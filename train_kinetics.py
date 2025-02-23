import os
import os.path as osp
import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from dataset import Kinetics_finetune_Dataset
from utils import set_seed, rmse, mae, pearson
from scipy import stats
from model_tankbind_att_dual_loss_dropout_unimol_mlp_promlp3_DTA_cat2_2_score_norm1_3layermlp import GNNModel, Config
import wandb


def load_data_to_device(data, device):
    data_dict = {}
    data_dict['lig_id'] = data.ligand_id
    data_dict['lig_x'] = data.x_lig.to(device)
    data_dict['lig_batch'] = data.x_lig_batch.to(device)
    data_dict['output_score'] = data.output_score.to(device)

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


def train(model, loader, device, optimizer):
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
            loss = F.mse_loss(output, data.y.float().to(device))
        
        loss_all += loss.item() * data.num_graphs
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return loss_all / len(loader.dataset)


def test(model, loader, device):
    """
    Inference process
    """
    y_list = []
    pred_list = []

    model.eval()

    for data in tqdm(loader):
        data_dict = load_data_to_device(data, device)

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
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Max number of epochs to train')
    parser.add_argument('--dim_interact', type=int, default=1024, help='Size of hidden dimension for interaction')
    parser.add_argument('--patience', type=int, default=30, help='Number of epochs for early stopping')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--saved', type=str, default='model', help='Name for loading model')
    parser.add_argument('--max_length', type=float, default=700, help='Max protein length')
    parser.add_argument('--task', type=str, default="kinetics", help='Name of task')
    parser.add_argument('--score_file_name', type=str, default='model', help='Name for score_file_name')
    parser.add_argument('--NUM_TRIALS', type=int, default=10, help='Number of runs')
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
    
    esm_emb_dir = osp.join('.', 'data', 'esm', 'binding_kinetics')

    # Load data as datasets
    full_dataset = Kinetics_finetune_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'Binding_kinetics',
                                  csv_dir = osp.join('.', 'data', 'binding_kinetics', 'binding_kinetics_pairs.csv'),
                                  esm_emb_dir = esm_emb_dir,
                                  score_dir = osp.join('.', 'koff_' + args.score_file_name + '.pt'),
                                  max_length = args.max_length)
    
    print('Size of full_dataset:', len(full_dataset))
    
    rmse_list = []
    mae_list = []
    pearson_list = []
    spearman_list = []
    
    NUM_TRIALS = args.NUM_TRIALS

    filecount = 0
    for _ in range(NUM_TRIALS):
        filecount = filecount + 1
        trainidxfile = osp.join('.', 'data', 'binding_kinetics', 'scaffold_split', 'trainidx_' + str(filecount) + '.npy')
        train_ix = np.load(trainidxfile)
        validateidxfile = osp.join('.', 'data', 'binding_kinetics', 'scaffold_split', 'validateidx_' + str(filecount) + '.npy')
        validate_ix = np.load(validateidxfile)
        testidxfile = osp.join('.', 'data', 'binding_kinetics', 'scaffold_split', 'testidx_' + str(filecount) + '.npy')
        test_ix = np.load(testidxfile)

        train_dataset = full_dataset[train_ix]
        val_dataset = full_dataset[validate_ix]
        test_dataset = full_dataset[test_ix]
        print('==============================================================')
        print('Size of train_dataset:', len(train_dataset))
        print('Size of val_dataset:', len(val_dataset))
        print('Size of test_dataset:', len(test_dataset))


        # Use datasets to create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask', 'x_seq', 'x_pair'], shuffle=True, worker_init_fn=args.seed)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask', 'x_seq', 'x_pair'])
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask', 'x_seq', 'x_pair'])

        # Setup model configurations
        config = Config(dim_interact=args.dim_interact, dropout=args.dropout, task=args.task)
        model = GNNModel(config).to(device)

        # Training and inference processes
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_mae = None
        best_test_rmse = None
        best_test_mae = None
        best_test_sp = None
        best_test_p = None
        counter = 0

        print('=============================== Start training =====================================')
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, device, optimizer)
            val_rmse, val_mae, val_sp, val_p = test(model, val_loader, device)
            test_rmse, test_mae, test_sp, test_p = test(model, test_loader, device)
            
            if best_val_mae is None or val_mae <= best_val_mae:
                best_val_mae = val_mae
                best_test_rmse = test_rmse
                best_test_mae = test_mae
                best_test_sp = test_sp
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
                    "Val P": val_p,
                    "Val SP": val_sp,
                    "Test RMSE": test_rmse,
                    "Test MAE": test_mae,
                    "Test P": test_p,
                    "Test SP": test_sp,
                    "Best Val MAE": best_val_mae})
            print('Epoch: {:03d}, Train Loss: {:.7f}, \
                Test RMSE: {:.7f}, Test MAE: {:.7f}, Test SP: {:.7f}, Test P: {:.7f}'.format(epoch+1, train_loss, test_rmse, test_mae, test_sp, test_p))
            
            if counter == args.patience:
                break
        
        rmse_list.append(best_test_rmse)
        mae_list.append(best_test_mae)
        pearson_list.append(best_test_p)
        spearman_list.append(best_test_sp)

    print('Testing RMSE:', np.mean(rmse_list), np.std(rmse_list))
    print('Testing MAE:', np.mean(mae_list), np.std(mae_list))
    print('Testing P:', np.mean(pearson_list), np.std(pearson_list))
    print('Testing SP:', np.mean(spearman_list), np.std(spearman_list))

if __name__ == "__main__":
    main()