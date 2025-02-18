import os
import re
import os.path as osp
import numpy as np
from tqdm import tqdm

from rdkit import Chem

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from pdbeccdutils.core import ccd_reader

from utils import ccdtoken2smiles, get_compound_pair_dis

from unimol_tools import UniMolRepr


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_lig':
            return self.x_lig.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def merge_sequences(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    sequence = ''
    for line in lines[1:]:
        line = line.strip()
        sequence += line.replace('?', '')  # delete '?'
    
    return sequence

def count_nonempty_lines(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                if line.strip()[-1] != 'H':
                    count += 1
    return count

def get_mol_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_names.append(file_name)
    return file_names

def get_ccd_token_from_mol(file_path):
    with open(file_path, 'r') as mol_file:
        lines = mol_file.readlines()

    molecule_section_start = lines.index('@<TRIPOS>ATOM\n')
    ccd_token = lines[molecule_section_start+1].strip().split()[-2][:3]
    return ccd_token


def get_resi_info(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    resi_seq = ''
    indice_list = []
    
    for line in lines:
        line = line.strip()
        resi_seq += line.split('|')[0]
        indice_list.append(int(line.split('|')[-1]))
    return resi_seq, indice_list


def get_ligand_xyz(mol2_file):
    with open(mol2_file, 'r') as f:
        lines = f.readlines()
    
    xyz_list = []
    
    atom_section_start = lines.index('@<TRIPOS>ATOM\n')
    bond_section_start = lines.index('@<TRIPOS>BOND\n')
    
    atom_sequence = [line.strip() for line in lines[atom_section_start+1:bond_section_start] if line.strip() != '']
    
    for line in atom_sequence:
        if line.split()[5] != 'H':
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


def extract_resseq(input_string):
    match = re.search(r'resseq=(-?\d+)', input_string)
    if match:
        return int(match.group(1))
    else:
        return None


class HOLO4K_Dataset(InMemoryDataset):
    def __init__(self, root, name, cif_dir = None, dict_dir = None, mol_dir = None, 
                 fasta_dir = None, esm_emb_dir = None, valid_pair_dir = None):
        """
        Dataset object for splitted Biolip data
        """
        self.name = name
        self.cif_dir = cif_dir
        self.dict_dir = dict_dir
        self.mol_dir = mol_dir
        self.fasta_dir = fasta_dir
        self.esm_emb_dir = esm_emb_dir
        self.valid_pair_dir = valid_pair_dir

        super(HOLO4K_Dataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)
    
    @property
    def processed_file_names(self):
        return 'processed.pt'

    def process(self):
        data_list = []
        parsed_components = ccd_reader.read_pdb_components_file(self.cif_dir)

        # get mol embeddings from unimol
        clf = UniMolRepr(data_type='molecule', remove_hs=True)

        smiles_list = []
        unique_ccd_list = set()
        unique_idx = 0
        ccd_idx_dict = {}


        ligand_mol_file_names = get_mol_file_names(self.mol_dir)
        print('len ligand_mol_file_names', len(ligand_mol_file_names))
        ligand_mol_name_dict = {}

        for ligand_mol_file_name in ligand_mol_file_names:
            pdb_name = ligand_mol_file_name[:4]
            if pdb_name not in ligand_mol_name_dict.keys():
                ligand_mol_name_dict[pdb_name] = [ligand_mol_file_name]
            else:
                ligand_mol_name_dict[pdb_name].append(ligand_mol_file_name)
        print('len ligand_mol_name_dict', len(ligand_mol_name_dict))



        with open(self.valid_pair_dir, 'r') as f:
            lines = f.readlines()

        valid_pair_dict = {}
        for line in lines:
            pdb_id_chain = line.strip().split()[0]
            mol2_name = line.strip().split()[1]
            if pdb_id_chain not in valid_pair_dict.keys():
                valid_pair_dict[pdb_id_chain] = [mol2_name]
            else:
                valid_pair_dict[pdb_id_chain].append(mol2_name)


        
        with open(self.dict_dir, 'r') as file:
            lines = file.readlines()

            for line in tqdm(lines, desc="Processing", unit="line", ncols=80):    # chain level
                fasta_name, esm_file_name = line.strip().split('\t')     # 1m6wB.pdb.seq.fasta	holo4k_1538
                pdb_name = fasta_name[:4]
                pdb_id_chain = fasta_name[:5]

                if pdb_id_chain not in valid_pair_dict.keys():
                    continue

                # ligand info
                ligand_mol_file_names = ligand_mol_name_dict[pdb_name]  # ligand mol2 names for a protein
                for ligand_mol_file_name in ligand_mol_file_names:
                    if ligand_mol_file_name in valid_pair_dict[pdb_id_chain]:
                        ligand_mol_file_path = osp.join(self.mol_dir, ligand_mol_file_name)
                        
                        smiles = ''
                        ccd_token = get_ccd_token_from_mol(ligand_mol_file_path)

                        if ccd_token == 'VA5':
                            ccd_token = 'VA'
                        elif ccd_token == 'U22':
                            ccd_token = 'U'
                                
                        if ccd_token not in unique_ccd_list:
                            unique_ccd_list.add(ccd_token)
                            ccd_idx_dict[ccd_token] = unique_idx

                            if ccd_token == 'CBS':
                                mol = Chem.MolFromMol2File(ligand_mol_file_path, removeHs=True)
                                smiles = Chem.MolToSmiles(mol, canonical=True)
                            else:
                                smiles = ccdtoken2smiles(parsed_components, ccd_token)

                            if len(smiles) == 0:
                                print(pdb_id_chain, ligand_mol_file_name, ccd_token, smiles)

                            if Chem.MolFromSmiles(smiles) is None:
                                print(pdb_id_chain, ligand_mol_file_name, ccd_token, smiles)
                            smiles_list.append(smiles)
                            unique_idx += 1
        
        unimol_repr_full = clf.get_repr(smiles_list, return_atomic_reprs=True)
        
        for i in unimol_repr_full.keys():
            print(i, np.array(unimol_repr_full[i]).shape)
        
        
        with open(self.dict_dir, 'r') as file:
            lines = file.readlines()

            for line in tqdm(lines, desc="Processing", unit="line", ncols=80):
                fasta_name, esm_file_name = line.strip().split('\t')     # 1lqdBB.pdb.seq.fasta	coach420_0
                pdb_name = fasta_name[:4]
                pdb_id_chain = fasta_name[:5]
                chain_name = fasta_name[4]

                if pdb_id_chain not in valid_pair_dict.keys():
                    continue

                # ligand info
                ligand_mol_file_names = ligand_mol_name_dict[pdb_name]  # ligand pdb file names for a pair
                for ligand_mol_file_name in ligand_mol_file_names:
                    if ligand_mol_file_name in valid_pair_dict[pdb_id_chain]:
                        ligand_mol_file_path = osp.join(self.mol_dir, ligand_mol_file_name)

                        ccd_token = get_ccd_token_from_mol(ligand_mol_file_path)

                        if ccd_token == 'VA5':
                            ccd_token = 'VA'
                        elif ccd_token == 'U22':
                            ccd_token = 'U'

                        # UniMol no H
                        unimol_idx = ccd_idx_dict[ccd_token]
                        atomic_reprs = unimol_repr_full['atomic_reprs'][unimol_idx]
                        atomic_coords = unimol_repr_full['atomic_coords'][unimol_idx]

                        pair_dis = get_compound_pair_dis(atomic_coords, bin_max=15)
                            
                        assert int(pair_dis.size()[0]) == int(pair_dis.size()[1])
                        assert int(pair_dis.size()[0]) == int(atomic_reprs.shape[0])
                        assert pair_dis.dim() == 2
                        

                        data = PairData()
                        data.__num_nodes__ = int(atomic_reprs.shape[0])
                        data.x_lig = torch.from_numpy(atomic_reprs).to(torch.float)      # n, 512
                        data.pos_lig = torch.from_numpy(atomic_coords).to(torch.float)   # n, 3
                        data.dis_map_lig = pair_dis.reshape(-1, 1)
                        data.ligand_id = ccd_token

                        # protein seq info
                        seq = merge_sequences(osp.join(self.fasta_dir, fasta_name))
                        seq_length = len(seq)
                        esm_embs = torch.load(osp.join(self.esm_emb_dir, esm_file_name + '.pt'))
                        esm_seq_emb = esm_embs['representations'][33]
                        esm_pair_emb = esm_embs['contacts']

                        assert(esm_seq_emb.size()[0] == seq_length)
                        assert(esm_pair_emb.size()[0] == seq_length)
                        assert(esm_pair_emb.size()[1] == seq_length)

                        # diagonal is 1 (has contact)
                        for i in range(esm_pair_emb.size()[0]):
                            esm_pair_emb[i, i] = 1
                        
                        # reverse mapping from contact prob to dist
                        esm_pair_emb = 1 - esm_pair_emb

                        # nan value to be treated as far away in dist
                        nan_mask = torch.isnan(esm_pair_emb)
                        esm_pair_emb[nan_mask] = 1

                        assert(torch.max(esm_pair_emb) <= 1)
                        assert(torch.min(esm_pair_emb) >= 0)

                        data.x_seq = esm_seq_emb
                        data.x_pair = esm_pair_emb.reshape(-1, 1)

                        # binding site indices info
                        data.length_seq = torch.tensor(int(seq_length), dtype=torch.int64)
                        data.pair_name = pdb_id_chain
                        data.pocket_name = ligand_mol_file_name
                        data.seq = seq
                        
                        data_list.append(data)

        print('Saving...')
        torch.save(self.collate(data_list), self.processed_paths[0])


