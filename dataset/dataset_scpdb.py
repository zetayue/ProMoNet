import os.path as osp
from tqdm import tqdm

import re
import warnings
from Bio import BiopythonWarning
from rdkit import Chem

import numpy as np
from Bio import PDB, BiopythonWarning

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from unimol_tools import UniMolRepr

from utils import get_compound_pair_dis


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_lig':
            return self.x_lig.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def extract_resseq(input_string):
    match = re.search(r'resseq=(-?\d+)', input_string)
    if match:
        return int(match.group(1))
    else:
        return None

def parse_pdb_for_chain(pdb_file, chain_id, rcsb_fasta):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BiopythonWarning)
        parser = PDB.PDBParser()
        structure = parser.get_structure('structure', pdb_file)

        residue_fasta = ''
        residue_xyz_list = []
        
        aa3to1 = {
            "ALA": "A", "VAL": "V", "PHE": "F", "PRO": "P", "MET": "M", "ILE": "I", "LEU": "L", "ASP": "D", "GLU": "E", "LYS": "K",
            "ARG": "R", "SER": "S", "THR": "T", "TYR": "Y", "HIS": "H", "CYS": "C", "ASN": "N", "GLN": "Q", "TRP": "W", "GLY": "G",
            "2AS": "D", "3AH": "H", "5HP": "E", "ACL": "R", "AIB": "A", "ALM": "A", "ALO": "T", "ALY": "K", "ARM": "R", "ASA": "D",
            "ASB": "D", "ASK": "D", "ASL": "D", "ASQ": "D", "AYA": "A", "BCS": "C", "BHD": "D", "BMT": "T", "BNN": "A", "BUC": "C",
            "BUG": "L", "C5C": "C", "C6C": "C", "CCS": "C", "CEA": "C", "CHG": "A", "CLE": "L", "CME": "C", "CSO": "C",
            "CSP": "C", "CSS": "C", "CSW": "C", "CXM": "M", "CY1": "C", "CY3": "C", "CYG": "C", "CYM": "C", "CYQ": "C", "DAH": "F",
            "DAL": "A", "DAR": "R", "DAS": "D", "DCY": "C", "DGL": "E", "DGN": "Q", "DHA": "A", "DHI": "H", "DIL": "I", "DIV": "V",
            "DLE": "L", "DLY": "K", "DNP": "A", "DPN": "F", "DPR": "P", "DSN": "S", "DSP": "D", "DTH": "T", "DTR": "W", "DTY": "Y",
            "DVA": "V", "EFC": "C", "FLA": "A", "FME": "M", "GGL": "E", "GLZ": "G", "GMA": "E", "GSC": "G", "HAC": "A", "HAR": "R",
            "HIC": "H", "HIP": "H", "HMR": "R", "HPQ": "F", "HTR": "W", "HYP": "P", "IIL": "I", "IYR": "Y", "KCX": "K", "LLP": "K",
            "LLY": "K", "LTR": "W", "LYM": "K", "LYZ": "K", "MAA": "A", "MEN": "N", "MHS": "H", "MIS": "S", "MLE": "L", "MPQ": "G",
            "MSA": "G", "MSE": "M", "MVA": "V", "NEM": "H", "NEP": "H", "NLE": "L", "NLN": "L", "NLP": "L", "NMC": "G", "OAS": "S",
            "OCS": "C", "OMT": "M", "PAQ": "Y", "PCA": "E", "PEC": "C", "PHI": "F", "PHL": "F", "PR3": "C", "PRR": "A", "PTR": "Y",
            "SAC": "S", "SAR": "G", "SCH": "C", "SCS": "C", "SCY": "C", "SEL": "S", "SEP": "S", "SET": "S", "SHC": "C", "SHR": "K",
            "SOC": "C", "STY": "Y", "SVA": "S", "TIH": "A", "TPL": "W", "TPO": "T", "TRG": "K", "TRO": "W", "TYB": "Y",
            "TYQ": "Y", "TYS": "Y", "TYY": "Y", "AGM": "R", "GL3": "G", "SMC": "C", "ASX": "B", "CGU": "E", "CSX": "C", "GLX": "Z",
            "LED": "L", "UNK": "X"
        }
        exclude_list = ['SEC']


        for model in structure:
            for chain in model:
                pre_residue_idx = -1e5
                rcsb_idx = 0
                shift = 0
                has_x = 0
                if chain.id == chain_id:
                    for residue in chain:
                        if 'CA' in residue:
                            
                            resseq_value = extract_resseq(residue.__repr__())
                            if pre_residue_idx == -1e5:
                                pre_residue_idx = resseq_value
                            if residue.get_resname() in aa3to1.keys() and residue.get_resname() not in exclude_list:
                                if resseq_value - pre_residue_idx <= 1:
                                    residue_fasta += aa3to1[residue.get_resname()]
                                    residue_xyz_list.append(residue['CA'].get_coord())
                                elif resseq_value - pre_residue_idx > 1 and residue.id[0] == ' ':
                                    residue_fasta += aa3to1[residue.get_resname()]
                                    residue_xyz_list.append(residue['CA'].get_coord())
                                pre_residue_idx = resseq_value
    return residue_fasta, residue_xyz_list

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


def get_chain_info(sequence, atom_sequence, pdb_chain):
    residue_fasta = ''
    residue_xyz_list = []
    
    amino_acids = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    # get chain start aa idx and end aa idx (not the value in 'PRO1', 'GLN2')
    chain_aa_list = []
    for line in sequence:
        chain_name = line.split()[5]
        aa_type = line.split()[3]
        aa_number = line.split()[0]
        aa_name = line.split()[6]

        if chain_name == pdb_chain and aa_name != 'HOH' and aa_type != 'GROUP':
            chain_aa_list.append(aa_number)

    try:
        start_aa_idx = int(chain_aa_list[0])
        end_aa_idx = int(chain_aa_list[-1])
    except:
        return [], []

    first_line = atom_sequence[0]
    fisrt_aa_idx = int(first_line.split()[6])

    idx = start_aa_idx
    finished_idx = []

    for line in atom_sequence:
        if idx > end_aa_idx:
            continue
        atom_name = line.split()[1]
        aa = line.split()[7]        # e.g. 'PRO1', 'GLN2'
        aa_idx = int(line.split()[6])        # not the value in 'PRO1', 'GLN2'
        aa_name = aa[:3]

        if aa_idx < start_aa_idx:
            continue

        if aa_idx == idx:
            if atom_name == 'CA':
                if aa_name in amino_acids:
                    residue_fasta += amino_acids[aa_name]
                    x = float(line.split()[2])
                    y = float(line.split()[3])
                    z = float(line.split()[4])
                    residue_xyz_list.append([x, y, z])
                    finished_idx.append(idx)
                    idx += 1
        elif idx < aa_idx and aa_idx not in finished_idx:     # has omited aa
            finished_idx.append(idx)
            idx += 1
       
    return residue_fasta, residue_xyz_list



def extract_chain(mol2_file, pdb_chain):
    with open(mol2_file, 'r') as f:
        lines = f.readlines()
    
    substructure_section_start = lines.index('@<TRIPOS>SUBSTRUCTURE\n')
    try:
        set_section_start = lines.index('@<TRIPOS>SET\n')
        sequence = [line.strip() for line in lines[substructure_section_start+1:set_section_start] if line.strip() != '']
    except:
        sequence = [line.strip() for line in lines[substructure_section_start+1:] if line.strip() != '']
    
    
    atom_section_start = lines.index('@<TRIPOS>ATOM\n')
    bond_section_start = lines.index('@<TRIPOS>BOND\n')
    atom_sequence = [line.strip() for line in lines[atom_section_start+1:bond_section_start] if line.strip() != '']
    
    residue_fasta, residue_xyz_list = get_chain_info(sequence, atom_sequence, pdb_chain)
    
    return residue_fasta, residue_xyz_list


def longest_common_subsequence(A, B):
    # Create a DP table to store lengths of longest common subsequences
    dp = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
    
    # Fill dp table
    for i in range(1, len(A) + 1):
        for j in range(1, len(B) + 1):
            if A[i-1] == B[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the LCS
    lcs = []
    i, j = len(A), len(B)
    while i > 0 and j > 0:
        if A[i-1] == B[j-1]:
            lcs.append(A[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    lcs.reverse()
    return lcs


def insert_instructions(A, B):
    lcs = longest_common_subsequence(A, B)
    instructions = []
    a_idx, b_idx, lcs_idx = 0, 0, 0
    idx_list = []

    while b_idx < len(B):
        # If current characters in A and B match the LCS, move to the next character in A, B and LCS
        if a_idx < len(A) and lcs_idx < len(lcs) and A[a_idx] == lcs[lcs_idx] and B[b_idx] == lcs[lcs_idx]:
            a_idx += 1
            lcs_idx += 1
        else:
            # If characters don't match, we need to insert the character from B into A
            instructions.append((a_idx, B[b_idx]))
            idx_list.append(a_idx)
        b_idx += 1

    return idx_list


def insert_x_at_indices(s, indices):
    """
    Insert the character 'X' into a string at specified indices.

    :param s: The original string.
    :param indices: A list of indices where 'X' will be inserted.
    :return: Modified string with 'X' inserted.
    """
    # Sort indices in descending order
    sorted_indices = sorted(indices, reverse=True)

    # Insert 'X' starting from the highest index
    for index in sorted_indices:
        if index > len(s):
            raise IndexError("Index is out of the bounds of the string")
        s = s[:index] + 'X' + s[index:]

    return s


def insert_same_chars_at_indices(s, indices):
    """
    Insert a copy of the character at each specified index in the string.

    :param s: The original string.
    :param indices: A list of indices where the character at each index will be duplicated.
    :return: Modified string with characters inserted.
    """
    # Sort indices in descending order
    sorted_indices = sorted(indices, reverse=True)

    # Insert character from the index starting from the highest index
    for index in sorted_indices:
        if index == len(s):
            s = s[:index] + [s[index - 1]]
        else:
            s = s[:index] + [s[index]] + s[index:]

    return s


def read_fasta_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def extract_fasta_chain(fasta_data, chain):
    sequences = fasta_data.strip().split('>')
    
    for seq in sequences:
        if seq:
            header, sequence = seq.split('\n', 1)
            if f"|Chain" in header and f"auth {chain}" in header:
                return f"{sequence}"
            
    for seq in sequences:
        if seq:
            header, sequence = seq.split('\n', 1)       
            if (f"Chain {chain}" in header) or (f"|Chains " in header and f" {chain}," in header) or (f", {chain}|" in header):
                return f"{sequence}"
    return f"Chain {chain} not found in the given FASTA data."


def calculate_similarity(str1, str2):
    if len(str1) != len(str2):
        return 0.0
    same_count = 0
    len_count = 0

    for char1, char2 in zip(str1, str2):
        if char1 != "X" and char2 != "X":
            if char1 == char2:
                same_count += 1
            len_count += 1
    similarity = same_count / len_count

    return similarity

class scPDB_Dataset(InMemoryDataset):
    def __init__(self, root, name, cif_dir = None, txt_dir = None, dict_dir = None, 
                 esm_emb_dir = None, scPDB_dir = None, max_length = None):
        """
        Dataset object for splitted Biolip data
        """
        self.name = name
        self.cif_dir = cif_dir
        self.txt_dir = txt_dir
        self.dict_dir = dict_dir
        self.esm_emb_dir = esm_emb_dir
        self.scPDB_dir = scPDB_dir
        self.max_length = max_length

        super(scPDB_Dataset, self).__init__(root)
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
        scPDB_dir = self.scPDB_dir
 
        smiles_list = []
        unique_subfolder_name_list = set()
        unique_idx = 0
        subfolder_name_idx_dict = {}

        with open(self.txt_dir, 'r') as file:
            lines = file.readlines()

            for line in tqdm(lines, desc="Processing", unit="line", ncols=80):
                # ligand info
                line = line.strip()
                parts = line.split('\t')

                # skip sequence with length > max_length
                if len(parts[-1]) > self.max_length:
                    continue

                subfolder_name = parts[1]
                try:
                    ligand_path = scPDB_dir + '/' + subfolder_name + '/' + 'ligand.sdf'

                    if subfolder_name not in unique_subfolder_name_list:
                        unique_subfolder_name_list.add(subfolder_name)
                        subfolder_name_idx_dict[subfolder_name] = unique_idx

                        suppl = Chem.SDMolSupplier(ligand_path, removeHs=True)
                        mol = suppl[0]
                        if mol is None:
                            print(subfolder_name)
                        smiles = Chem.MolToSmiles(mol, canonical=True)
                        assert len(smiles) > 0
                        smiles_list.append(smiles)
                        unique_idx += 1
                except:
                    ligand_path = scPDB_dir + '/' + subfolder_name + '/' + 'ligand.mol2'

                    if subfolder_name not in unique_subfolder_name_list:
                        unique_subfolder_name_list.add(subfolder_name)
                        subfolder_name_idx_dict[subfolder_name] = unique_idx

                        mol = Chem.MolFromMol2File(ligand_path, removeHs=True)
                        if mol is None:
                            print(subfolder_name)
                        smiles = Chem.MolToSmiles(mol, canonical=True)
                        assert len(smiles) > 0
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

        with open(self.txt_dir, 'r') as file:
            lines = file.readlines()
            count = 0

            for line in tqdm(lines, desc="Processing", unit="line", ncols=80):
                count += 1

                # ligand info
                line = line.strip()
                parts = line.split('\t')

                # skip sequence with length > max_length
                if len(parts[-1]) > self.max_length:
                    continue

                subfolder_name = parts[1]    # e.g. 4c4o_2
                fasta_info = parts[-1]
                ccd_token = parts[3]
                chain_name = parts[2]
                pdb_id = subfolder_name[:4]
                ligand_path = scPDB_dir + '/' + subfolder_name + '/' + 'ligand.sdf'
                fasta_path = '../scPDB/downloaded_fasta/' + subfolder_name + '.fasta'

                ligand_xyz_list = get_ligand_xyz(scPDB_dir + '/' + subfolder_name + '/' + 'ligand.mol2')
                esm_file_name = esm_file_dict[fasta_info]
                
                fasta_data = read_fasta_file(fasta_path)
                rcsb_fasta = extract_fasta_chain(fasta_data, chain_name)
                rcsb_fasta = rcsb_fasta.replace(" ", "")
                rcsb_fasta = rcsb_fasta.replace("\n", "")



                if pdb_id in ['4abz', '1hyh', '2hwc', '7cat', '1eah', '3pch', '2hwb', '6tmn', '3pce', '1gd1', '2rr1', '2ldb', '2rs3', '1ncq', 
                      '8cat', '3pcf', '1hrv', '2tmn', '1aqu', '3pci', '1r08', '3pcg', '2rs5', '4tmn', '1r09', '4tln', '1gpd', '2rm2', 
                      '1fx1', '2r07', '2rs1', '3pcn', '3pcb', '1lmc', '2r06', '3pcc', '2r04', '1hri', '1lxf', '2m56', '1cf4', '1e0a',
                      '2l1r', '1zy8', '3pcc', '3pcb', '2kwi', '2rlf', '1ohh', '2kot', '1aiy', '2aaz', '4egb', '2mse', '2krd', '1ai0',
                      '2klh', '2bru']:
                    protein_path = scPDB_dir + '/' + subfolder_name + '/' + 'protein.mol2'
                    residue_fasta, residue_xyz_list = extract_chain(protein_path, chain_name)
                else:
                    protein_path = '../scPDB/downloaded_pdb/' + subfolder_name + '.pdb'
                    residue_fasta, residue_xyz_list = parse_pdb_for_chain(protein_path, chain_name, rcsb_fasta)



                if (pdb_id == '3e2s' and chain_name == 'A') or (pdb_id == '2oa1' and chain_name == 'A') \
                    or (pdb_id == '3h6v' and chain_name == 'A') or (pdb_id == '2d32' and chain_name == 'B') \
                    or (pdb_id == '3qlr' and chain_name == 'A') or (pdb_id == '1tdn' and chain_name == 'A') \
                    or (pdb_id == '1tdo' and chain_name == 'A') or (pdb_id == '1jdb' and chain_name == 'H') \
                    or (pdb_id == '2cdq' and chain_name == 'A') or (pdb_id == '2cdq' and chain_name == 'B') \
                    or (pdb_id == '5cto' and chain_name == 'D') or (pdb_id == '3kwn' and chain_name == 'A') \
                    or (pdb_id == '2dfd' and chain_name == 'A') or (pdb_id == '3pzr' and chain_name == 'B'):
                    residue_fasta = residue_fasta[:-1]
                    residue_xyz_list = residue_xyz_list[:-1]
                elif (pdb_id == '2hw2' and chain_name == 'A') or (pdb_id == '4y9q' and chain_name == 'A') \
                    or (pdb_id == '5a3b' and chain_name == 'A') or (pdb_id == '3beg' and chain_name == 'A') \
                    or (pdb_id == '5a3c' and chain_name == 'A'):
                    residue_fasta = residue_fasta[:-2]
                    residue_xyz_list = residue_xyz_list[:-2]
                elif pdb_id == '1kh3' and chain_name == 'A':
                    residue_fasta = residue_fasta[:-3]
                    residue_xyz_list = residue_xyz_list[:-3]
                elif pdb_id == '3mvq' and chain_name == 'D':
                    residue_fasta = residue_fasta[5:]
                    residue_xyz_list = residue_xyz_list[5:]
                    n = 42
                    residue_fasta = residue_fasta[:n-1] + "G" + residue_fasta[n:]
                    n = 243
                    residue_fasta = residue_fasta[:n-1] + "A" + residue_fasta[n:]
                    n = 266
                    residue_fasta = residue_fasta[:n-1] + "V" + residue_fasta[n:]
                    n = 267
                    residue_fasta = residue_fasta[:n-1] + "A" + residue_fasta[n:]
                    n = 382
                    residue_fasta = residue_fasta[:n-1] + "K" + residue_fasta[n:]
                elif pdb_id == '3jsk' and chain_name == 'E':
                    n = 181
                    residue_fasta = residue_fasta[:n-1] + "S" + residue_fasta[n:]
                elif (pdb_id == '2cmj' and chain_name == 'A') or (pdb_id == '2cmj' and chain_name == 'B') \
                    or (pdb_id == '2cmv' and chain_name == 'A') or (pdb_id == '2cmv' and chain_name == 'B'):
                    n = 240
                    residue_fasta = residue_fasta[:n-1] + "K" + residue_fasta[n:]
                elif (pdb_id == '3mw9' and chain_name == 'A') or (pdb_id == '3mw9' and chain_name == 'C'):
                    n = 47
                    residue_fasta = residue_fasta[:n-1] + "G" + residue_fasta[n:]
                    n = 248
                    residue_fasta = residue_fasta[:n-1] + "A" + residue_fasta[n:]
                    n = 271
                    residue_fasta = residue_fasta[:n-1] + "V" + residue_fasta[n:]
                    n = 272
                    residue_fasta = residue_fasta[:n-1] + "A" + residue_fasta[n:]
                    n = 387
                    residue_fasta = residue_fasta[:n-1] + "K" + residue_fasta[n:]
                elif pdb_id == '4egb':
                    rcsb_fasta = 'MHHHHHHSSGVDLGTENLYFQSNAMNILVTGGAGFIGSNFVHYMLQSYETYKIINFDALTYSGNLNNVKSIQDHPNYYFVKGEIQNGELLEHVIKERDVQVIVNFAAESHVDRSIENPIPFYDTNVIGTVTLLELVKKYPHIKLVQVSTDEVYGSLGKTGRFTEETPLAPNSPYSSSKASADMIALAYYKTYQLPVIVTRCSNNYGPYQYPEKLIPLMVTNALEGKKLPLYGDGLNVRDWLHVTDHCSAIDVVLHKGRVGEVYNIGGNNEKTNVEVVEQIITLLGKTKKDIEYVTDRLGHDRRYAINAEKMKNEFDWEPKYTFEQGLQETVQWYEKNEEWWKPLKK'
                elif (pdb_id == '1hwz' and chain_name == 'C') or (pdb_id == '4n7a' and chain_name == 'A'):
                    residue_fasta = rcsb_fasta
                elif pdb_id == '3mpe':
                    rcsb_fasta = 'ILPDSVDWREKGCVTEVKYQGSCGASWAFSAVGALEAQLKLKTGKLVSLSAQNLVDCSTEKYGNKGCNGGFMTTAFQYIIDNKGIDSDASYPYKAMDQKCQYDSKYRAATCSKYTELPYGREDVLKEAVANKGPVSVGVDARHPSFFLYRSGVYYEPSCTQNVNHGVLVVGYGDLNGKEYWLVKNSWGHNFGEEGYIRMARNKGNHCGIASFPSYPEILQGGG'

                
                if len(residue_fasta) != 0:
                    insert_idx = insert_instructions(residue_fasta, rcsb_fasta)
                    inserted_residue_fasta = insert_x_at_indices(residue_fasta, insert_idx)
                    inserted_residue_xyz_list = insert_same_chars_at_indices(residue_xyz_list, insert_idx)
                    assert len(inserted_residue_fasta) == len(inserted_residue_xyz_list)
                
                    if calculate_similarity(rcsb_fasta, inserted_residue_fasta) < 0.9 or len(rcsb_fasta) != len(inserted_residue_fasta):
                        protein_path = scPDB_dir + '/' + subfolder_name + '/' + 'protein.mol2'
                        residue_fasta, residue_xyz_list = extract_chain(protein_path, chain_name)

                        insert_idx = insert_instructions(residue_fasta, rcsb_fasta)
                        inserted_residue_fasta = insert_x_at_indices(residue_fasta, insert_idx)
                        inserted_residue_xyz_list = insert_same_chars_at_indices(residue_xyz_list, insert_idx)
                        assert len(inserted_residue_fasta) == len(inserted_residue_xyz_list)

                        if calculate_similarity(rcsb_fasta, inserted_residue_fasta) < 0.9 or len(rcsb_fasta) != len(inserted_residue_fasta):
                            print('==================')
                            print('not similar')
                            print(subfolder_name, chain_name)
                            print(residue_fasta)
                            print(inserted_residue_fasta)
                            print(rcsb_fasta)
                            error_count += 1
                            continue

                # use rcsb_fasta and inserted_residue_xyz_list
                dist_list = []
                for residue_xyz in inserted_residue_xyz_list:
                    min_dist = shortest_distance_xyz(residue_xyz, ligand_xyz_list)
                    dist_list.append(min_dist)

                indices = np.where(np.array(dist_list) < 6.5)    # scPDB cutoff
                index_list = indices[0].tolist()

                if len(index_list) == 0:     # remove chains that does not have pocket
                    continue

                # save as data
                data = PairData()

                # UniMol no H
                unimol_idx = subfolder_name_idx_dict[subfolder_name]
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
                data.ligand_id = ccd_token

                # protein info
                seq = rcsb_fasta
                seq_length = len(seq)
                esm_embs = torch.load(osp.join(self.esm_emb_dir, esm_file_name + '.pt'))
                esm_seq_emb = esm_embs['representations'][33]
                esm_pair_emb = esm_embs['contacts']

                assert(esm_seq_emb.size()[0] == seq_length)
                assert(esm_pair_emb.size()[0] == seq_length)
                assert(esm_pair_emb.size()[1] == seq_length)
                assert(esm_seq_emb.size()[0] == len(dist_list))

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

                data.y_seq = torch.tensor(dist_list, dtype=torch.float32)
                data.length_seq = torch.tensor(seq_length, dtype=torch.int64)
                data.pair_name = subfolder_name
                data.seq = seq
                
                data_list.append(data)
        
        print('Saving...')
        torch.save(self.collate(data_list), self.processed_paths[0])
