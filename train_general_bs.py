import os
import os.path as osp
import argparse

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from dataset import scPDB_Dataset
from utils import set_seed
from model import ProMoSite, Config
import wandb

def bce_loss(pred, target, alpha=-1, gamma=2):
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return ce_loss.mean()

def calculate_iou(target, pred):
    threshold = 0.5
    pred_binary = [1 if p > threshold else 0 for p in pred]
    intersection = sum(t * p for t, p in zip(target, pred_binary))
    union = sum(t + p > 0 for t, p in zip(target, pred_binary))
    iou = intersection / union
    return iou

def load_data_to_device(data, device):
    data_dict = {}
    data_dict['lig_id'] = data.ligand_id
    data_dict['lig_x'] = data.x_lig.to(device)
    data_dict['lig_batch'] = data.x_lig_batch.to(device)

    lig_pair_batch = data.dis_map_lig_batch.to(device)
    data_dict['protein_x'] = data.x_seq.float().to(device)
    data_dict['protein_batch'] = data.x_seq_batch.to(device)

    protein_pair_batch = data.x_pair_batch.to(device)

    protein_pair_init = data.x_pair.to(device)
    lig_pair_init = data.dis_map_lig.to(device)

    #====================== Generate initial pair embs ======================
    _, protein_batch_count = torch.unique(data_dict['protein_batch'], return_counts=True)
    _, lig_batch_count = torch.unique(data_dict['lig_batch'], return_counts=True)
    assert protein_batch_count.size() == lig_batch_count.size()

    batch_n = protein_batch_count.size()[0]

    max_protein_size = protein_batch_count.max().item()
    max_lig_size = lig_batch_count.max().item()

    protein_pair = torch.zeros((batch_n, max_protein_size, max_protein_size, 1), device=device)
    lig_pair = torch.zeros((batch_n, max_lig_size, max_lig_size, 1), device=device)

    for i in range(batch_n):
        protein_size_square_i = (protein_pair_batch == i).sum()
        protein_size_i = int(protein_size_square_i**0.5)
        protein_pair[i, :protein_size_i, :protein_size_i] = protein_pair_init[protein_pair_batch == i].reshape(
                                                            (protein_size_i, protein_size_i, -1))
        lig_size_square_i = (lig_pair_batch == i).sum()
        lig_size_i = int(lig_size_square_i**0.5)
        lig_pair[i, :lig_size_i, :lig_size_i] = lig_pair_init[lig_pair_batch == i].reshape(
                                                                (lig_size_i, lig_size_i, -1))
    data_dict['protein_pair'] = protein_pair
    data_dict['lig_pair'] = lig_pair
    
    return data_dict

def get_binary_labels_from_dist(dist_list, cutoff):
    indices = np.where(np.array(dist_list) < cutoff)
    index_list = indices[0].tolist()

    length_seq = len(dist_list)

    y_seq = torch.zeros(length_seq)
    y_seq[index_list] = 1

    return y_seq

def train(model, loader, device, optimizer, y_cutoff):
    """
    Training process
    """
    loss_all = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    for data in tqdm(loader):
        data_dict = load_data_to_device(data, device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(data_dict)
            y_seq = get_binary_labels_from_dist(data.y_seq.tolist(), y_cutoff)
            loss = bce_loss(output, y_seq.float().to(device))
        
        loss_all += loss.item() * data.num_graphs
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return loss_all / len(loader.dataset)

def test(model, loader, device, y_cutoff):
    """
    Inference process
    """
    loss_all = 0
    all_targets = []
    all_preds = []

    model.eval()

    for data in tqdm(loader):
        data_dict = load_data_to_device(data, device)

        with torch.cuda.amp.autocast():
            output = model(data_dict)
        output_scaled = torch.sigmoid(output)
        y_seq = get_binary_labels_from_dist(data.y_seq.tolist(), y_cutoff)
        loss = bce_loss(output, y_seq.float().to(device))

        loss_all += loss.item() * data.num_graphs
        all_targets += y_seq.tolist()
        all_preds += output_scaled.tolist()
    all_targets = np.asarray(all_targets)
    all_preds = np.asarray(all_preds)

    auc = roc_auc_score(all_targets, all_preds)

    threshold = 0.5
    binary_pred = (all_preds >= threshold).astype(int)
    f1 = f1_score(all_targets, binary_pred)

    iou = calculate_iou(all_targets, all_preds)

    return loss_all / len(loader.dataset), auc, f1, iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=420, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=40, help='Max number of epochs to train.')
    parser.add_argument('--dim_interact', type=int, default=32, help='Size of hidden dimension for interaction')
    parser.add_argument('--dim_pair', type=int, default=64, help='Size of hidden dimension for pair embs')
    parser.add_argument('--n_module', type=int, default=2, help='Number of Update layers')
    parser.add_argument('--patience', type=int, default=4, help='Number of epochs for early stopping')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for data')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--y_cutoff', type=float, default=7.0, help='Cutoff for creating binary y')
    parser.add_argument('--task', type=str, default="coach420", choices=["coach420", "holo4k"], help='Name of task')
    parser.add_argument('--max_length', type=float, default=1280, help='Max protein length')
    args = parser.parse_args()

    if args.wandb:
        run = wandb.init(project="ProMoNet")
        wandb.config.update(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    # Load data as datasets
    if args.task == "coach420":
        full_dataset = scPDB_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                    name = 'scPDB_coach420',
                                    cif_dir = osp.join('.', 'data', 'Components-rel-alt.cif'),
                                    txt_dir = osp.join('.', 'data', 'scPDB_subset_test_coach420.txt'),
                                    dict_dir = osp.join('.', 'data', 'scPDB_unique_fasta_dict.txt'),
                                    esm_emb_dir = osp.join('.', 'data', 'esm', 'scPDB'),
                                    scPDB_dir = osp.join('.', 'data', 'scPDB'),
                                    max_length = args.max_length)
    elif args.task == "holo4k":
        full_dataset = scPDB_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                    name = 'scPDB_holo4k',
                                    cif_dir = osp.join('.', 'data', 'Components-rel-alt.cif'),
                                    txt_dir = osp.join('.', 'data', 'scPDB_subset_test_holo4k.txt'),
                                    dict_dir = osp.join('.', 'data', 'scPDB_unique_fasta_dict.txt'),
                                    esm_emb_dir = osp.join('.', 'data', 'esm', 'scPDB'),
                                    scPDB_dir = osp.join('.', 'data', 'scPDB'),
                                    max_length = args.max_length)
    else:
        raise ValueError("Invalid task.")
    print('Size of dataset:', len(full_dataset))

    # Create dataloaders
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'x_seq', 'x_pair'], shuffle=True, worker_init_fn=args.seed)

    # Setup model configurations
    config = Config(dim_interact=args.dim_interact, dim_pair=args.dim_pair, n_module=args.n_module, dropout=args.dropout)
    model = ProMoSite(config).to(device)

    # Training and inference processes
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    counter = 0

    print('=============================== Start training =====================================')
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, device, optimizer, args.y_cutoff)

        if args.wandb:
            torch.save(model.state_dict(), os.path.join("saved_model", "model_" + run.name + "_" + str(epoch + 1) + ".h5"))
            wandb.log({"Train Loss": train_loss})
        print('Epoch: {:03d}, Train MAE: {:.7f}'.format(epoch + 1, train_loss))
        
        if counter == args.patience:
            break

if __name__ == "__main__":
    main()