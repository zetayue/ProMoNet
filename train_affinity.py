import os
import os.path as osp
import argparse

from tqdm import tqdm
import numpy as np
from scipy import stats

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from dataset import PDBbind2020_finetune_Dataset
from utils import set_seed, rmse, mae, pearson
from model import ProMoBind, Config
import wandb


def load_data_to_device(data, embeddings_dict, device):
    data_dict = {}
    data_dict['lig_id'] = data.ligand_id
    data_dict['lig_x'] = data.x_lig.to(device)
    data_dict['lig_batch'] = data.x_lig_batch.to(device)
    data_dict['output_score'] = data.output_score.to(device)

    lig_pair_batch = data.dis_map_lig_batch.to(device)
    lig_pair_init = data.dis_map_lig.to(device)

    # create batch info
    integer_list = data.length_seq.to(device)
    indices = torch.arange(len(integer_list)).to(device)
    data_dict['protein_batch'] = torch.repeat_interleave(indices, integer_list)

    #====================== Generate initial protein pair embs ======================
    _, protein_batch_count = torch.unique(data_dict['protein_batch'], return_counts=True)
    batch_n = protein_batch_count.size()[0]

    max_protein_size = protein_batch_count.max().item()
    protein_pair = torch.zeros((batch_n, max_protein_size, max_protein_size, 1), device=device)

    x_seq_list = []

    esm_file_names = data.esm_file_name

    for i in range(batch_n):
        esm_file_name = esm_file_names[i]
        esm_seq_emb = embeddings_dict[esm_file_name]['seq_emb'].to(device)
        esm_pair_emb = embeddings_dict[esm_file_name]['pair_emb'].to(device)

        protein_size_i = int(esm_pair_emb.size()[0])

        protein_pair[i, :protein_size_i, :protein_size_i] = esm_pair_emb.reshape((protein_size_i, protein_size_i, -1))

        x_seq_list.append(esm_seq_emb)
    data_dict['protein_x'] = torch.cat(x_seq_list, dim=0)

    #====================== Generate initial pair embs ======================
    _, lig_batch_count = torch.unique(data_dict['lig_batch'], return_counts=True)
    max_lig_size = lig_batch_count.max().item()

    lig_pair = torch.zeros((batch_n, max_lig_size, max_lig_size, 1), device=device)

    for i in range(batch_n):
        lig_size_square_i = (lig_pair_batch == i).sum()
        lig_size_i = int(lig_size_square_i**0.5)
        lig_pair[i, :lig_size_i, :lig_size_i] = lig_pair_init[lig_pair_batch == i].reshape(
                                                                (lig_size_i, lig_size_i, -1))

    data_dict['protein_pair'] = protein_pair
    data_dict['lig_pair'] = lig_pair
    
    return data_dict


def train(model, loader, device, optimizer, embeddings_dict):
    """
    Training process
    """
    loss_all = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    for data in tqdm(loader):
        data_dict = load_data_to_device(data, embeddings_dict, device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(data_dict)
            loss = F.mse_loss(output, data.y.float().to(device))
        
        loss_all += loss.item() * data.num_graphs
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return loss_all / len(loader.dataset)


def test(model, loader, device, embeddings_dict):
    """
    Inference process
    """
    y_list = []
    pred_list = []

    model.eval()

    for data in tqdm(loader):
        data_dict = load_data_to_device(data, embeddings_dict, device)

        with torch.cuda.amp.autocast():
            output = model(data_dict)

        y_list += data.y.reshape(-1).tolist()
        pred_list += output.reshape(-1).tolist()

    y = np.array(y_list).reshape(-1,)
    pred = np.array(pred_list).reshape(-1,)

    return rmse(y, pred), mae(y, pred), stats.spearmanr(y, pred).correlation, pearson(y, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--cpu', type=int, default=4, help='Number of CPUs to use')
    parser.add_argument('--seed', type=int, default=920, help='Random seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Max number of epochs to train')
    parser.add_argument('--dim_interact', type=int, default=1024, help='Size of hidden dimension for interaction')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for data')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--max_length', type=float, default=3000, help='Max protein length')
    parser.add_argument('--task', type=str, default="affinity", help='Name of task')
    parser.add_argument('--saved', type=str, default='model', help='Name for loading model')
    args = parser.parse_args()

    if args.wandb:
        run = wandb.init(project="ProMoNet")
        wandb.config.update(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    os.environ["OMP_NUM_THREADS"] = str(args.cpu)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.cpu)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.cpu)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu)

    torch.set_num_threads(args.cpu)

    esm_emb_dir = osp.join('.', 'data', 'esm', 'pdbbind2020')

    # Load data as datasets
    train_dataset = PDBbind2020_finetune_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'pdbbind2020_train',
                                  csv_dir = osp.join('.', 'data', 'pdbbind2020', 'train.csv'),
                                  dict_dir = osp.join('.', 'data', 'pdbbind2020', 'pdbbind2020_dict.txt'),
                                  esm_emb_dir = esm_emb_dir,
                                  score_dir = osp.join('.', 'data', 'toy_model_koff_update', 'pdbbind2020_train_raw_3000_'+args.saved+'.pt'),
                                  max_length = args.max_length)

    val_dataset = PDBbind2020_finetune_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'pdbbind2020_val',
                                  csv_dir = osp.join('.', 'data', 'pdbbind2020', 'valid.csv'),
                                  dict_dir = osp.join('.', 'data', 'pdbbind2020', 'pdbbind2020_dict.txt'),
                                  esm_emb_dir = esm_emb_dir,
                                  score_dir = osp.join('.', 'data', 'toy_model_koff_update', 'pdbbind2020_val_raw_3000_'+args.saved+'.pt'),
                                  max_length = args.max_length)

    test_dataset = PDBbind2020_finetune_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'pdbbind2020_test',
                                  csv_dir = osp.join('.', 'data', 'pdbbind2020', 'test.csv'),
                                  dict_dir = osp.join('.', 'data', 'pdbbind2020', 'pdbbind2020_dict.txt'),
                                  esm_emb_dir = esm_emb_dir,
                                  score_dir = osp.join('.', 'data', 'toy_model_koff_update', 'pdbbind2020_test_raw_3000_'+args.saved+'.pt'),
                                  max_length = args.max_length)
    

    print('Size of train_dataset:', len(train_dataset))
    print('Size of val_dataset:', len(val_dataset))
    print('Size of val_dataset:', len(test_dataset))


    # Use datasets to create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask'], shuffle=True, worker_init_fn=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask'])

    
    # Setup model configurations
    config = Config(dim_interact=args.dim_interact, dropout=args.dropout, task=args.task)
    model = ProMoBind(config).to(device)

    # Training and inference processes
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # pre-load all esm embeddings
    embeddings_dict = {}

    for file_name in os.listdir(esm_emb_dir):
        if file_name.endswith('.pt'):
            esm_file_name = file_name[:-3]
            esm_embs = torch.load(osp.join(esm_emb_dir, esm_file_name + '.pt'))
            
            esm_seq_emb = esm_embs['representations'][33]
            esm_pair_emb = esm_embs['contacts']

            for idx in range(esm_pair_emb.size()[0]):
                esm_pair_emb[idx, idx] = 1

            # reverse mapping from contact prob to dist
            esm_pair_emb = 1 - esm_pair_emb

            # nan value to be treated as far away in dist
            nan_mask = torch.isnan(esm_pair_emb)
            esm_pair_emb[nan_mask] = 1

            embeddings_dict[esm_file_name] = {'seq_emb': esm_seq_emb, 'pair_emb': esm_pair_emb}


    best_epoch = None
    best_val_mae = None
    best_test_rmse = None
    best_test_mae = None
    best_test_sd = None
    best_test_p = None
    counter = 0

    print('=============================== Start training =====================================')
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, device, optimizer, embeddings_dict)
        val_rmse, val_mae, val_sd, val_p = test(model, val_loader, device, embeddings_dict)
        test_rmse, test_mae, test_sd, test_p = test(model, test_loader, device, embeddings_dict)
        
        if best_val_mae is None or val_mae <= best_val_mae:
            best_epoch = epoch
            best_val_mae = val_mae
            best_test_rmse = test_rmse
            best_test_mae = test_mae
            best_test_sd = test_sd
            best_test_p = test_p
            counter = 0
            if args.wandb:
                torch.save(model.state_dict(), os.path.join("saved_model", "model_" + run.name + ".h5"))
        else:
            counter += 1

        
        if args.wandb:
            wandb.log({
                "Train Loss": train_loss,
                "Val RMSE": val_rmse,
                "Val MAE": val_mae,
                "Val SP": val_sd,
                "Val P": val_p,
                "Test RMSE": test_rmse,
                "Test MAE": test_mae,
                "Test SP": test_sd,
                "Test P": test_p,
                "Best Val MAE": best_val_mae})
        print('Epoch: {:03d}, Train Loss: {:.7f}, \
            Test RMSE: {:.7f}, Test MAE: {:.7f}, Test P: {:.7f}, Test SP: {:.7f}'.format(epoch+1, train_loss, test_rmse, test_mae, test_p, test_sd))


    print('===================================================================================')
    print('Best Epoch:', best_epoch + 1)
    print('Testing RMSE:', best_test_rmse)
    print('Testing MAE:', best_test_mae)
    print('Testing P:', best_test_p)
    print('Testing SP:', best_test_sd)
    

if __name__ == "__main__":
    main()