import os
import os.path as osp
import argparse

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from dataset import Kinetics_pretrain_Dataset
from utils import set_seed
from model import ProMoSite, Config
import wandb


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


def test(model, loader, device, batch_size, name):
    """
    Inference process
    """
    model.eval()

    output_cpu_list = []

    for data in tqdm(loader):
        data_dict = load_data_to_device(data, device)
        protein_batch = data_dict['protein_batch']

        with torch.cuda.amp.autocast():
            output = model(data_dict).detach()
            output = torch.sigmoid(output)
            
        for i in range(batch_size):
            i_indices = torch.where(protein_batch == i)[0]
            output_i = output[i_indices]
            output_i = output_i.to('cpu')
            if output_i.size()[0] > 0:
                output_cpu_list.append(output_i)

    torch.save(output_cpu_list, name + '.pt')

    return True


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
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--max_length', type=float, default=700, help='Max protein length')
    parser.add_argument('--saved', type=str, default='model', help='Name for loading model')
    args = parser.parse_args()

    if args.wandb:
        run = wandb.init(project="ProMoNet")
        wandb.config.update(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    esm_emb_dir = osp.join('.', 'data', 'esm', 'binding_kinetics')

    # Load data as datasets
    full_dataset = Kinetics_pretrain_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'Binding_kinetics_score',
                                  csv_dir = osp.join('.', 'data', 'binding_kinetics', 'binding_kinetics_pairs.csv'),
                                  esm_emb_dir = esm_emb_dir,
                                  max_length = args.max_length)

    print('Size of full_dataset:', len(full_dataset))

    # Use datasets to create dataloaders
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask', 'x_seq', 'x_pair'])


    # Setup model configurations
    config = Config(dim_interact=args.dim_interact, dim_pair=args.dim_pair, n_module=args.n_module, dropout=args.dropout)
    model = ProMoSite(config).to(device)

    pretrained_dict = torch.load(os.path.join('.', "saved_model", "model_" + args.saved + ".h5"))
    model_dict = model.state_dict()

    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    _ = test(model, train_loader, device, args.batch_size, 'koff_'+args.saved)



if __name__ == "__main__":
    main()