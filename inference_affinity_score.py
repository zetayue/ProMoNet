import os
import os.path as osp
import argparse

from tqdm import tqdm

import torch
from torch_geometric.data import DataLoader

from dataset import PDBbind2020_pretrain_Dataset
from utils import set_seed
from model import ProMoSite, Config
import wandb


def load_data_to_device(data, embeddings_dict, device):
    data_dict = {}
    data_dict['lig_id'] = data.ligand_id
    data_dict['lig_x'] = data.x_lig.to(device)
    data_dict['lig_batch'] = data.x_lig_batch.to(device)

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


def test(model, loader, device, embeddings_dict, batch_size, name):
    """
    Inference process
    """
    model.eval()
    output_cpu_list = []

    for data in tqdm(loader):
        data_dict = load_data_to_device(data, embeddings_dict, device)
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
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--cpu', type=int, default=4, help='Number of CPUs to use')
    parser.add_argument('--seed', type=int, default=920, help='Random seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=40, help='Max number of epochs to train')
    parser.add_argument('--dim_interact', type=int, default=32, help='Size of hidden dimension for interaction')
    parser.add_argument('--dim_pair', type=int, default=64, help='Size of hidden dimension for pair embs')
    parser.add_argument('--n_module', type=int, default=2, help='Number of Update layers')
    parser.add_argument('--patience', type=int, default=6, help='Number of epochs for early stopping')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--max_length', type=float, default=3000, help='Max protein length')
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
    train_dataset = PDBbind2020_pretrain_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'pdbbind2020_train_score',
                                  csv_dir = osp.join('.', 'data', 'pdbbind2020', 'train.csv'),
                                  dict_dir = osp.join('.', 'data', 'pdbbind2020', 'pdbbind2020_dict.txt'),
                                  esm_emb_dir = esm_emb_dir,
                                  max_length = args.max_length)

    val_dataset = PDBbind2020_pretrain_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'pdbbind2020_val_score',
                                  csv_dir = osp.join('.', 'data', 'pdbbind2020', 'valid.csv'),
                                  dict_dir = osp.join('.', 'data', 'pdbbind2020', 'pdbbind2020_dict.txt'),
                                  esm_emb_dir = esm_emb_dir,
                                  max_length = args.max_length)

    test_dataset = PDBbind2020_pretrain_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'pdbbind2020_test_score',
                                  csv_dir = osp.join('.', 'data', 'pdbbind2020', 'test.csv'),
                                  dict_dir = osp.join('.', 'data', 'pdbbind2020', 'pdbbind2020_dict.txt'),
                                  esm_emb_dir = esm_emb_dir,
                                  max_length = args.max_length)



    print('Size of train_dataset:', len(train_dataset))
    print('Size of val_dataset:', len(val_dataset))
    print('Size of val_dataset:', len(test_dataset))


    # Use datasets to create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask'])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'LAS_mask'])


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

    # Setup model configurations
    config = Config(dim_interact=args.dim_interact, dim_pair=args.dim_pair, n_module=args.n_module, dropout=args.dropout)
    model = ProMoSite(config).to(device)

    pretrained_dict = torch.load(os.path.join('.', "saved_model", "model_" + args.saved + ".h5"))
    model_dict = model.state_dict()

    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    
    _ = test(model, train_loader, device, embeddings_dict, args.batch_size, 'pdbbind2020_train_'+args.saved)
    _ = test(model, val_loader, device, embeddings_dict, args.batch_size, 'pdbbind2020_val_'+args.saved)
    _ = test(model, test_loader, device, embeddings_dict, args.batch_size, 'pdbbind2020_test_'+args.saved)


if __name__ == "__main__":
    main()