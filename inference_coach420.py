import os
import os.path as osp
import argparse
import warnings

from tqdm import tqdm
import numpy as np
from Bio import PDB, BiopythonWarning
from scipy.cluster.hierarchy import linkage, fcluster

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from dataset import COACH420_Dataset
from utils import set_seed
from model import ProMoSite, Config
import wandb

warnings.simplefilter('ignore', BiopythonWarning)


def bce_loss(pred, target, alpha=-1, gamma=2):
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return ce_loss.mean()


def get_residue_lines_from_pdb(pdb_file, residue_idx):
    parser = PDB.PDBParser()
    structure = parser.get_structure('pdb', pdb_file)
    model = structure[0]

    residue_lines = []
    for chain in model:
        for idx in residue_idx:
            try:
                residue = chain.get_list()[idx]
                atom = residue.get_unpacked_list()[1]
                line = [atom.coord[0], atom.coord[1], atom.coord[2]]
                residue_lines.append(line)
            except IndexError:
                pass

        return residue_lines
    
def get_center_xyz_from_pdb(pdb_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure('pdb', pdb_file)
    model = structure[0]

    for chain in model:
        protein_center_xyz = np.array([0.0, 0.0, 0.0])
        residue_n = 0
        try:
            for residue in chain.get_list():
                atom = residue.get_unpacked_list()[1]
                protein_center_xyz += np.array([atom.coord[0], atom.coord[1], atom.coord[2]])
                residue_n += 1
        except IndexError:
            pass
        protein_center_xyz = protein_center_xyz / residue_n
        return protein_center_xyz


def split_list_by_label(res_indices, labels, confidences, xyzs):
    res_index_dict = {}
    confidence_dict = {}
    center_xyz_dict = {}

    for res_index, label, confidence, xyz in zip(res_indices, labels, confidences, xyzs):
        if label not in res_index_dict:
            res_index_dict[label] = []
            confidence_dict[label] = 0
            center_xyz_dict[label] = np.array([0.0, 0.0, 0.0])
        res_index_dict[label].append(res_index)
        confidence_dict[label] += confidence ** 2
        center_xyz_dict[label] += np.array(xyz)
    
    label_list = list(res_index_dict.keys())
    for label in label_list:
        if len(res_index_dict[label]) <= 3:  # Delete pocket has <= 3 residues
            del res_index_dict[label]
            del confidence_dict[label]
            del center_xyz_dict[label]
        else:
            center_xyz_dict[label] = list(center_xyz_dict[label] / len(res_index_dict[label]))

    return list(res_index_dict.values()), list(confidence_dict.values()), list(center_xyz_dict.values())


def select_top_lists(lists, confidences, center_xyzs, top_n):
    combined = sorted(zip(lists, confidences, center_xyzs), key=lambda x: x[1], reverse=True)

    if top_n > len(lists):
        top_list = [x[0] for x in combined[:]]
        confidence_list = [x[1] for x in combined[:]]
        center_xyz_list = [x[2] for x in combined[:]]
    else:
        top_list = [x[0] for x in combined[:top_n]]
        confidence_list = [x[1] for x in combined[:top_n]]
        center_xyz_list = [x[2] for x in combined[:top_n]]

    return top_list, confidence_list, center_xyz_list


def get_ligand_xyz_from_mol2(mol2_file):
    with open(mol2_file, 'r') as f:
        lines = f.readlines()
    
    xyz_list = []
    atom_section_start = lines.index('@<TRIPOS>ATOM\n')
    bond_section_start = lines.index('@<TRIPOS>BOND\n')
    
    atom_sequence = [line.strip() for line in lines[atom_section_start+1:bond_section_start] if line.strip() != '']

    for line in atom_sequence:
        x = float(line.split()[2])
        y = float(line.split()[3])
        z = float(line.split()[4])
        xyz_list.append([x, y, z])
    return xyz_list


def shortest_distance_xyz(xyz, xyz_list):
    xyz = np.array(xyz)
    xyz_list = np.array(xyz_list)
    distances = np.sqrt(np.sum((xyz_list - xyz)**2, axis=1))
    shortest_distance = np.min(distances)
    return shortest_distance


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


def test(model, loader, device, threshold, cutoff, id_cutoff):
    loss_all = 0
    all_targets = []
    all_preds = []

    pdb_success_rate_top_n = {}
    fail_count = 0

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

        # 3d based evaluation
        pdb_file_path = osp.join('..', 'p2rank-datasets', 'coach420', data.pair_name[0][:5] + '.pdb')

        ligand_mol2_path = osp.join('..', 'DeepPocket_data', 'coach420_ligands', data.pair_name[0])
        ligand_xyz_list = get_ligand_xyz_from_mol2(ligand_mol2_path)

        output_scaled_lst = np.array(output_scaled.tolist())
        max_value = np.max(output_scaled_lst)
        normed_output = ((output_scaled_lst) / (max_value )).tolist()

        # Clustering
        residue_idx = np.where(np.array(normed_output) > threshold)[0]

        if len(residue_idx) == 0:
            residue_idx = np.array([0])

        residue_xyzs = get_residue_lines_from_pdb(pdb_file_path, residue_idx)
        protein_center_xyz = get_center_xyz_from_pdb(pdb_file_path)

        contact_map = data_dict['protein_pair'].squeeze(0)
        contact_map = contact_map.squeeze(-1)

        contact_map = 1 - contact_map 
        zero_mask = torch.nonzero(contact_map == 0)
        contact_map[zero_mask] = 1e-5

        zero_mask = torch.nonzero(contact_map == 1)  # let value 1 stable
        contact_map[zero_mask] = 1

        contact_map = - 10 * torch.log(contact_map.float())   # to dist map
        contact_map = contact_map.cpu().numpy()

        new_contact_map = contact_map[np.ix_(residue_idx, residue_idx)]

        upper_triangular_indices = np.triu_indices(new_contact_map.shape[0], k=1)
        upper_triangular_values = new_contact_map[upper_triangular_indices]


        center_xyz_list = []
        confidence_list = []
        if len(residue_idx) == 1:
            center_xyz_list.append(residue_xyzs[0])
            confidence_list.append(0.0)
            fail_count += 1
        else:
            Z = linkage(upper_triangular_values, 'single')
            label_list = fcluster(Z, cutoff, criterion='distance')

            cluster_lists, confidence_list, center_xyz_list = split_list_by_label(residue_idx, label_list, normed_output, residue_xyzs)
            cluster_lists, confidence_list, center_xyz_list = select_top_lists(cluster_lists, confidence_list, center_xyz_list, 5)

            if len(center_xyz_list) == 0:
                center_xyz_list.append(protein_center_xyz)
                confidence_list.append(0.0)
                fail_count += 1

        # Compute DCA (distance between the pocket center and any ligand atom)
        min_dist_list = []
        min_dist_confidence_list = []
        idx = 0
        for center_xyz in center_xyz_list:
            min_dist = 1e5
            dist = shortest_distance_xyz(center_xyz, ligand_xyz_list)
            if dist < min_dist:
                min_dist = dist
            if min_dist > 300:
                print(min_dist)
                print(center_xyz, ligand_xyz_list)
            min_dist_list.append(min_dist)

            confidence = confidence_list[idx]
            min_dist_confidence_list.append(confidence)

            idx += 1

        # Compute Top-n
        if min_dist_list[0] <= id_cutoff:
            if data.pair_name[0][:5] not in pdb_success_rate_top_n.keys():
                pdb_success_rate_top_n[data.pair_name[0][:5]] = [1.0]
            else:
                pdb_success_rate_top_n[data.pair_name[0][:5]].append(1.0)
        else:
            if data.pair_name[0][:5] not in pdb_success_rate_top_n.keys():
                pdb_success_rate_top_n[data.pair_name[0][:5]] = [0.0]
            else:
                pdb_success_rate_top_n[data.pair_name[0][:5]].append(0.0)

    success_rate_all_sum = 0
    ligand_count = 0
    for key in pdb_success_rate_top_n.keys():
        success_rate_list = pdb_success_rate_top_n[key]
        success_rate_all_sum += sum(success_rate_list)
        ligand_count += len(success_rate_list)
    top_n_mean = success_rate_all_sum / ligand_count

    return top_n_mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=420, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs to train.')
    parser.add_argument('--dim_interact', type=int, default=32, help='Size of hidden dimension for interaction')
    parser.add_argument('--dim_pair', type=int, default=64, help='Size of hidden dimension for pair embs')
    parser.add_argument('--n_module', type=int, default=2, help='Number of Update layers')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for confidence')
    parser.add_argument('--cutoff', type=float, default=50, help='Cutoff for clustering')
    parser.add_argument('--id_cutoff', type=float, default=4.0, help='Cutoff for identification')
    parser.add_argument('--dataset', type=str, default='scPDB', help='Name for dataset')
    parser.add_argument('--saved', type=str, default='model', help='Name for loading model')
    args = parser.parse_args()

    if args.wandb:
        run = wandb.init(project="ProMoNet")
        wandb.config.update(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    # Load data as datasets
    test_dataset = COACH420_Dataset(root = osp.join('.', 'data', 'processed_dataset'),
                                  name = 'COACH420',
                                  cif_dir = osp.join('.', 'data', 'Components-rel-alt.cif'),
                                  dict_dir = osp.join('.', 'data', 'coach420_unique_fasta_dict.txt'),
                                  mol_dir = osp.join('.', 'data', 'DeepPocket_data', 'coach420_ligands'),
                                  fasta_dir = osp.join('.', 'data', 'p2rank-datasets', 'coach420', 'fasta'),
                                  esm_emb_dir = osp.join('.', 'data', 'esm', 'coach420'),
                                  residue_idx_dir = osp.join('.', 'data', 'PointSite_data', 'PointSite_TestData', 'COACH420_data', 'data'))
    
    print('Size of COACH420 dataset:', len(test_dataset))

    # Create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=['x_lig', 'pos_lig', 'dis_map_lig', 'x_seq', 'x_pair'])

    # Setup model configurations
    config = Config(dim_interact=args.dim_interact, dim_pair=args.dim_pair, n_module=args.n_module, dropout=args.dropout)
    model = ProMoSite(config).to(device)

    model.load_state_dict(torch.load(os.path.join("saved_model", "model_" + args.saved + ".h5")))

    top_n = test(model, test_loader, device, args.threshold, args.cutoff, args.id_cutoff)
    print('Top-n SR:', top_n)


if __name__ == "__main__":
    main()