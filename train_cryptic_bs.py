import os
import os.path as osp
import argparse

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from dataset import scPDB_Dataset, PocketMiner_Dataset
from utils import set_seed
from model import ProMoSite, Config
import wandb


def bce_loss(pred, target, alpha=-1, gamma=2):
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    mask = (target == 2)
    ce_loss[mask] = 0
    return ce_loss.sum() / (~mask).sum()

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

def test(model, loader, device):
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
            loss = bce_loss(output, data.y_seq.float().to(device))
        
        loss_all += loss.item() * data.num_graphs
        all_targets += data.y_seq.tolist()
        all_preds += output_scaled.tolist()
    all_targets = np.asarray(all_targets)
    all_preds = np.asarray(all_preds)

    mask = (all_targets != 2)
    filtered_all_targets = all_targets[mask]
    filtered_all_preds = all_preds[mask]

    auc = roc_auc_score(filtered_all_targets, filtered_all_preds)
    aps = average_precision_score(filtered_all_targets, filtered_all_preds)

    return loss_all / len(loader.dataset), auc, aps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=920, help='Random seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=40, help='Max number of epochs to train.')
    parser.add_argument('--dim_interact', type=int, default=32, help='Size of hidden dimension for interaction')
    parser.add_argument('--dim_pair', type=int, default=64, help='Size of hidden dimension for pair embs')
    parser.add_argument('--n_module', type=int, default=2, help='Number of Update layers')
    parser.add_argument('--patience', type=int, default=6, help='Number of epochs for early stopping')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for data')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--y_cutoff', type=float, default=7.0, help='Cutoff for creating binary y')
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
    train_dataset = scPDB_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'scPDB_pocketminer',
                                  cif_dir = osp.join('.', 'data', 'Components-rel-alt.cif'),
                                  txt_dir = osp.join('.', 'data', 'scPDB_full_preprocessed_rcsb_20240410_line_blast_30_pm_biolip.txt'),
                                  dict_dir = osp.join('.', 'data', 'scPDB_full_preprocessed_rcsb_20240410_line_dict.txt'),
                                  esm_emb_dir = osp.join('.', 'data', 'esm', 'scPDB'),
                                  scPDB_dir = osp.join('.', 'data', 'scPDB'),
                                  max_length = args.max_length)
    
    val_dataset = PocketMiner_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                name = 'pocketminer_val',
                                cif_dir = osp.join('.', 'data', 'Components-rel-alt.cif'),
                                fasta_dir = osp.join('.', 'data', 'pocketminer_all_fasta.fasta'),
                                dict_dir = osp.join('.', 'data', 'pocketminer_val_id_mapping.txt'),
                                esm_emb_dir = osp.join('.', 'data', 'esm', 'pocketminer'),
                                label_dir = osp.join('.', 'data', 'pocketminer_fixed_label_dict.npy'),
                                max_length = args.max_length)

    test_dataset = PocketMiner_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                name = 'pocketminer_test',
                                cif_dir = osp.join('.', 'data', 'Components-rel-alt.cif'),
                                fasta_dir = osp.join('.', 'data', 'pocketminer_all_fasta.fasta'),
                                dict_dir = osp.join('.', 'data', 'pocketminer_test_id_mapping.txt'),
                                esm_emb_dir = osp.join('.', 'data', 'esm', 'pocketminer'),
                                label_dir = osp.join('.', 'data', 'pocketminer_fixed_label_dict.npy'),
                                max_length = args.max_length)

    print('Size of train_dataset:', len(train_dataset))
    print('Size of val_dataset:', len(val_dataset))
    print('Size of test_dataset:', len(test_dataset))

    # Use datasets to create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask', 'x_seq', 'x_pair'], shuffle=True, worker_init_fn=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask', 'x_seq', 'x_pair'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask', 'x_seq', 'x_pair'])

    # Setup model configurations
    config = Config(dim_interact=args.dim_interact, dim_pair=args.dim_pair, n_module=args.n_module, dropout=args.dropout)
    model = ProMoSite(config).to(device)

    # Training and inference processes
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_epoch = None
    best_val_loss = None
    best_val_auc = None
    best_val_aps = None
    counter = 0

    print('=============================== Start training =====================================')
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, device, optimizer, args.y_cutoff)
        val_loss, val_auc, val_aps = test(model, val_loader, device)
        test_loss, test_auc, test_aps = test(model, test_loader, device)
        
        if best_val_loss is None or val_loss <= best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_val_aps = val_aps
            counter = 0
            if args.wandb:
                torch.save(model.state_dict(), os.path.join("saved_model", "model_" + run.name + ".h5"))
        else:
            counter += 1

        if args.wandb:
            wandb.log({
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Val AUC": val_auc,
                "Val APS": val_aps,
                "Test Loss": test_loss,
                "Test AUC": test_auc,
                "Test APS": test_aps,
                "Best Val AUC": best_val_auc,
                "Best Val APS": best_val_aps})
        print('Epoch: {:03d}, Train loss: {:.7f}, Val loss: {:.7f}, Test loss: {:.7f}'.format(epoch + 1, train_loss, val_loss, test_loss))
        
        if counter == args.patience:
            break

    print('=============================== Training finished =====================================')
    print('Best Epoch:', best_epoch + 1)
    print('Best Val Loss:', best_val_loss)
    print('Best Val AUC:', best_val_auc)
    print('Best Val APS:', best_val_aps)

if __name__ == "__main__":
    main()