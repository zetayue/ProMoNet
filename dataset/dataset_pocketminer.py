import os.path as osp
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from unimol_tools import UniMolRepr
from pdbeccdutils.core import ccd_reader

from utils import ccdtoken2smiles, get_compound_pair_dis


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_lig':
            return self.x_lig.size(0)
        return super().__inc__(key, value, *args, **kwargs)
    

class PocketMiner_Dataset(InMemoryDataset):
    def __init__(self, root, name, cif_dir = None, fasta_dir = None, dict_dir = None, 
                 esm_emb_dir = None, label_dir = None, max_length = None):
        """
        Dataset object for splitted Biolip data
        """
        self.name = name
        self.cif_dir = cif_dir
        self.fasta_dir = fasta_dir
        self.dict_dir = dict_dir
        self.esm_emb_dir = esm_emb_dir
        self.label_dir = label_dir
        self.max_length = max_length

        super(PocketMiner_Dataset, self).__init__(root)
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

        with open(self.dict_dir, 'r') as file:
            lines = file.readlines()

            for line in tqdm(lines, desc="Processing", unit="line", ncols=80):
                # ligand info
                line = line.strip()
                parts = line.split('\t')

                pdb_pair_name = parts[0]
                pdb_id = pdb_pair_name[:4].lower()

                ccd_token = parts[-1]

                if ccd_token == 'NULL':
                    ccd_token = 'PLP'
                    ccd_token_as_id = 'NULL'
                else:
                    ccd_token_as_id = ccd_token

                if ccd_token not in unique_ccd_list:
                    unique_ccd_list.add(ccd_token)
                    ccd_idx_dict[ccd_token] = unique_idx
                    smiles = ccdtoken2smiles(parsed_components, ccd_token)
                    smiles_list.append(smiles)
                    unique_idx += 1


        unimol_repr_full = clf.get_repr(smiles_list, return_atomic_reprs=True)
        
        for i in unimol_repr_full.keys():
            print(i, np.array(unimol_repr_full[i]).shape)


        esm_file_dict = {}
        with open(self.dict_dir, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                esm_file_dict[key] = value

        fasta_dict = {}
        current_name = ''

        with open(self.fasta_dir, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    current_name = line[1:]
                    fasta_dict[current_name] = ''
                else:
                    fasta_dict[current_name] += line


        pdb_mapping = {'4wbn': '6i5c', '3tuw': '5zgs', '4rzu': '5h70', '4nt3': '6lco', '1dx4': '6xyy', '2cmv': '5yzi', 
                    '2cmj': '5yzh', '3mw9': '6dhm', '5cto': '6j63', '4gn6': '6m7e', '3ogw': '7wyj', '2pdt': '6cny',
                    '4qe6': '6hl1', '4iqq': '5noo', '4n7a': '6lf7', '4otw': '6op9', '3mpe': '5qby', '3mpe': '6dhl',
                    '3mvq': '6dhl', '4p63': '7cmc', '4utd': '6cpf', '3q9k': '7dn6', '1hwz': '6dhd', '4dgo': '6qs5',
                    '3kwn': '5qc4', '3ql6': '7dn7', '3ake': '7ckj'}

        flipped_pdb_mapping = {v: k for k, v in pdb_mapping.items()}
        
        with open(self.dict_dir, 'r') as file:
            lines = file.readlines()

            for line in tqdm(lines, desc="Processing", unit="line", ncols=80):
                # ligand info
                line = line.strip()
                parts = line.split('\t')

                pdb_pair_name = parts[0]
                pdb_id = pdb_pair_name[:4].lower()

                ccd_token = parts[-1]

                if ccd_token == 'NULL':
                    ccd_token = 'PLP'
                    ccd_token_as_id = 'NULL'
                else:
                    ccd_token_as_id = ccd_token

                esm_file_name = pdb_pair_name

                # map the new pdb id to the old ones in scPDB
                if pdb_id in flipped_pdb_mapping:
                    pdb_id = flipped_pdb_mapping[pdb_id]

                # save as data
                data = PairData()

                # UniMol no H
                unimol_idx = ccd_idx_dict[ccd_token]
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
                data.ligand_id = ccd_token_as_id

                # protein info
                seq = fasta_dict[pdb_pair_name]
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


                # load label dict to get y_seq
                label_dictionary = np.load(self.label_dir, allow_pickle=True).item()
                if pdb_pair_name == '5x8ua_clean_h':
                    y_seq = label_dictionary['5x8ua']
                else:
                    y_seq = label_dictionary[pdb_id]
                assert len(y_seq) == seq_length
                
                data.y_seq = torch.tensor(y_seq, dtype=torch.float32)
                data.length_seq = torch.tensor(seq_length, dtype=torch.int64)
                data.pair_name = pdb_pair_name
                data.seq = seq
                
                data_list.append(data)

        
        print('Saving...')
        torch.save(self.collate(data_list), self.processed_paths[0])
