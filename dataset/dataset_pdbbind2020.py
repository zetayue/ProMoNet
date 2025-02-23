import os.path as osp
import csv

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from utils import get_compound_pair_dis

from unimol_tools import UniMolRepr


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_lig':
            return self.x_lig.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class PDBbind2020_pretrain_Dataset(InMemoryDataset):
    def __init__(self, root, name, csv_dir = None, dict_dir = None, 
                 esm_emb_dir = None, max_length = None):
        """
        Dataset object for splitted Biolip data
        """
        self.name = name
        self.csv_dir = csv_dir
        self.dict_dir = dict_dir
        self.esm_emb_dir = esm_emb_dir
        self.max_length = max_length

        super(PDBbind2020_pretrain_Dataset, self).__init__(root)
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

        # get mol embeddings from unimol
        clf = UniMolRepr(data_type='molecule', remove_hs=True)

        smiles_list = []
        unique_idx = 0
        smiles_to_idx_dict = {}

        with open(self.csv_dir, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                smiles = row['Ligand']
                if smiles not in smiles_list:
                    smiles_list.append(smiles)
                    smiles_to_idx_dict[smiles] = unique_idx
                    unique_idx += 1

        unimol_repr_full = clf.get_repr(smiles_list, return_atomic_reprs=True)

        esm_file_dict = {}
        with open(self.dict_dir, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                esm_file_dict[key] = value

        failed_pair_count = 0
        with open(self.csv_dir, mode='r') as file:
            reader = csv.DictReader(file)
            idx = 0
            for row in reader:
                smiles = row['Ligand']
                fasta = row['Protein']
                pair_idx = row['ID']
                pKi = float(row['regression_label'])
                esm_file_name = esm_file_dict[pair_idx]

                # save as data
                data = PairData()

                # UniMol no H
                unimol_idx = smiles_to_idx_dict[smiles]
                atomic_reprs = unimol_repr_full['atomic_reprs'][unimol_idx]
                atomic_coords = unimol_repr_full['atomic_coords'][unimol_idx]

                pair_dis = get_compound_pair_dis(atomic_coords, bin_max=15)
                
                assert int(pair_dis.size()[0]) == int(pair_dis.size()[1])
                assert int(pair_dis.size()[0]) == int(atomic_reprs.shape[0])
                assert pair_dis.dim() == 2

                data.__num_nodes__ = int(atomic_reprs.shape[0])
                data.x_lig = torch.from_numpy(atomic_reprs).to(torch.float)      # n, 512
                data.pos_lig = torch.from_numpy(atomic_coords).to(torch.float)   # n, 3
                data.dis_map_lig = pair_dis.reshape(-1, 1)
                data.ligand_id = smiles

                # protein info
                seq_length = len(fasta)

                try:
                    esm_embs = torch.load(osp.join(self.esm_emb_dir, esm_file_name + '.pt'))
                except:
                    failed_pair_count += 1
                    continue
                
                esm_seq_emb = esm_embs['representations'][33]
                esm_pair_emb = esm_embs['contacts']

                assert(esm_seq_emb.size()[0] == seq_length)
                assert(esm_pair_emb.size()[0] == seq_length)
                assert(esm_pair_emb.size()[1] == seq_length)

                data.y = torch.tensor(pKi, dtype=torch.float32)
                data.length_seq = torch.tensor(seq_length, dtype=torch.int64)
                data.pair_idx = pair_idx
                data.seq = fasta
                data.esm_file_name = esm_file_name
                
                data_list.append(data)

                idx += 1
        
        print('Saving...')
        torch.save(self.collate(data_list), self.processed_paths[0])


class PDBbind2020_finetune_Dataset(InMemoryDataset):
    def __init__(self, root, name, csv_dir = None, dict_dir = None, 
                 esm_emb_dir = None, score_dir = None, max_length = None):
        """
        Dataset object for splitted Biolip data
        """
        self.name = name
        self.csv_dir = csv_dir
        self.dict_dir = dict_dir
        self.esm_emb_dir = esm_emb_dir
        self.score_dir = score_dir
        self.max_length = max_length

        super(PDBbind2020_finetune_Dataset, self).__init__(root)
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

        # get mol embeddings from unimol
        clf = UniMolRepr(data_type='molecule', remove_hs=True)

        smiles_list = []
        unique_idx = 0
        smiles_to_idx_dict = {}

        output_score_list = torch.load(self.score_dir)

        with open(self.csv_dir, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                smiles = row['Ligand']
                if smiles not in smiles_list:
                    smiles_list.append(smiles)
                    smiles_to_idx_dict[smiles] = unique_idx
                    unique_idx += 1

        unimol_repr_full = clf.get_repr(smiles_list, return_atomic_reprs=True)

        esm_file_dict = {}
        with open(self.dict_dir, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                esm_file_dict[key] = value

        failed_pair_count = 0
        with open(self.csv_dir, mode='r') as file:
            reader = csv.DictReader(file)
            idx = 0
            for row in reader:
                smiles = row['Ligand']
                fasta = row['Protein']
                pair_idx = row['ID']
                pKi = float(row['regression_label'])
                esm_file_name = esm_file_dict[pair_idx]

                output_score = output_score_list[idx]

                # save as data
                data = PairData()

                # UniMol no H
                unimol_idx = smiles_to_idx_dict[smiles]
                atomic_reprs = unimol_repr_full['atomic_reprs'][unimol_idx]
                atomic_coords = unimol_repr_full['atomic_coords'][unimol_idx]

                pair_dis = get_compound_pair_dis(atomic_coords, bin_max=15)
                
                assert int(pair_dis.size()[0]) == int(pair_dis.size()[1])
                assert int(pair_dis.size()[0]) == int(atomic_reprs.shape[0])
                assert pair_dis.dim() == 2

                data.__num_nodes__ = int(atomic_reprs.shape[0])
                data.x_lig = torch.from_numpy(atomic_reprs).to(torch.float)      # n, 512
                data.pos_lig = torch.from_numpy(atomic_coords).to(torch.float)   # n, 3
                data.dis_map_lig = pair_dis.reshape(-1, 1)
                data.ligand_id = smiles

                # protein info
                seq_length = len(fasta)

                if seq_length <= self.max_length:
                    try:
                        esm_embs = torch.load(osp.join(self.esm_emb_dir, esm_file_name + '.pt'))
                    except:
                        failed_pair_count += 1
                        continue
                    
                    esm_seq_emb = esm_embs['representations'][33]
                    esm_pair_emb = esm_embs['contacts']

                    assert(esm_seq_emb.size()[0] == seq_length)
                    assert(esm_pair_emb.size()[0] == seq_length)
                    assert(esm_pair_emb.size()[1] == seq_length)
                    assert(output_score.size()[0] == seq_length)


                    data.y = torch.tensor(pKi, dtype=torch.float32)
                    data.length_seq = torch.tensor(seq_length, dtype=torch.int64)
                    data.pair_idx = pair_idx
                    data.seq = fasta
                    data.esm_file_name = esm_file_name
                    data.output_score = output_score
                    
                    data_list.append(data)

                    idx += 1
                else:
                    failed_pair_count += 1
        
        print('Saving...')
        torch.save(self.collate(data_list), self.processed_paths[0])
