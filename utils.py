import random
import numpy as np
import scipy
from rdkit import Chem

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear
from torch import Tensor
from math import sqrt, pi as PI



def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


#adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat


def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask


def get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=None, bin_max=15):
    pair_dis = scipy.spatial.distance.cdist(coords, coords)
    bin_size=1
    bin_min=-0.5
    if LAS_distance_constraint_mask is not None:
        pair_dis[LAS_distance_constraint_mask==0] = bin_max
        # diagonal is zero.
        for i in range(pair_dis.shape[0]):
            pair_dis[i, i] = 0
    pair_dis = torch.tensor(pair_dis, dtype=torch.float)
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    pair_dis_distribution = pair_dis_one_hot.float()
    return pair_dis_distribution


def get_compound_pair_dis(coords, bin_max=15):
    pair_dis = scipy.spatial.distance.cdist(coords, coords)
    pair_dis = torch.tensor(pair_dis, dtype=torch.float)
    pair_dis[pair_dis>bin_max] = bin_max
    return pair_dis




def ccdtoken2smiles(parsed_components, ccd_token):
    """
    Converts CCD 3-charactor token to 2D molecular graph Data object
    """
    ccd_component = parsed_components[ccd_token].component
    mol = ccd_component.mol

    ccd_smiles_dict = {'CLZ':'c1cc2c(c(c1N)Cl)c(nc(n2)N)N',
                       'ONP':'[Be-2](O[P@@](=O)(O)O[P@](=O)(O)OCCNc1ccccc1[N+](=O)[O-])(F)(F)F',
                       'NMQ':'[Be-2](O[P@@](=O)(O)O[P@](=O)(O)OCC[N@@](C)c1ccccc1[N+](=O)[O-])(F)(F)F',
                       'DAQ':'[Be-2](O[P@@](=O)(O)O[P@](=O)(O)OCCCNc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-])(F)(F)F',
                       'DAE':'[Be-2](O[P@@](=O)(O)O[P@](=O)(O)OCCNc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-])(F)(F)F',
                       'PNQ':'[Be-2](O[P@@](=O)(O)O[P@](=O)(O)OCCNc1ccc(cc1)[N+](=O)[O-])(F)(F)F'
                       }
    if ccd_token in ccd_smiles_dict.keys():
        return ccd_smiles_dict[ccd_token]
    else:
        #Chem.SanitizeMol(mol)
        #mol = Chem.RemoveAllHs(mol)
        return Chem.MolToSmiles(mol, canonical=True)



def MLP(channels):
    """
    Implementation of MLP
    """
    return Sequential(*[
        Sequential(Linear(channels[i - 1], channels[i]), SiLU())
        for i in range(1, len(channels))])


class SiLU(nn.Module):
    """
    SiLu non-linear function
    """
    def __init__(self):
        super().__init__() 

    def forward(self, input):
        return silu(input)


def silu(input):
    return input * torch.sigmoid(input)


class Res(nn.Module):
    """
    Residual module
    """
    def __init__(self, dim):
        super(Res, self).__init__()

        self.mlp = MLP([dim, dim, dim])

    def forward(self, m):
        m1 = self.mlp(m)
        m_out = m1 + m
        return m_out


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial: int, cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)
        thres = 1e-4
        dist[dist <= thres] = 1e-4
        dist = dist / self.cutoff
        #nan_mask = torch.isnan(dist)
        #dist[nan_mask] = 1.
        return (self.envelope(dist) * (self.freq * dist).sin())
    
class Envelope(torch.nn.Module):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: Tensor) -> Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 +
                c * x_pow_p2) * (x < 1.0).to(x.dtype)
    

def MLP_SiLU(channels, data_type=None):
    if data_type is not None:
        return Sequential(*[
            Sequential(Linear(channels[i - 1], channels[i], dtype=data_type), SiLU())
            for i in range(1, len(channels))])
    else:
        return Sequential(*[
            Sequential(Linear(channels[i - 1], channels[i]), SiLU())
            for i in range(1, len(channels))])