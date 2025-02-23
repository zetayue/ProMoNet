import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch.nn import LayerNorm, Linear, ReLU
from torch_scatter import scatter

from model import Config


class ProMoBind(nn.Module):
    """
    GNN model
    """
    def __init__(self, config: Config):
        super(ProMoBind, self).__init__()

        self.esm_feats = 1280
        self.dim_interact = config.dim_interact
        self.task = config.task
        
        if self.task == 'affinity':
            self.unimol_global_mlp = nn.Sequential(
                LayerNorm(512),
                Linear(512, self.dim_interact),
                ReLU(),
                LayerNorm(self.dim_interact),
                Linear(self.dim_interact, self.dim_interact),
                ReLU(),
                Linear(self.dim_interact, self.dim_interact),
            )
            self.esm_s_mlp = nn.Sequential(
                LayerNorm(self.esm_feats),
                Linear(self.esm_feats, self.dim_interact),
                ReLU(),
                LayerNorm(self.dim_interact),
                Linear(self.dim_interact, self.dim_interact),
                ReLU(),
                Linear(self.dim_interact, self.dim_interact),
            )
            self.concat_mlp = nn.Sequential(
                LayerNorm(self.dim_interact * 2),
                nn.Linear(self.dim_interact * 2, self.dim_interact * 2),
                nn.ReLU(),
                LayerNorm(self.dim_interact * 2),
                nn.Linear(self.dim_interact * 2, self.dim_interact),
                nn.ReLU(),
                nn.Linear(self.dim_interact, 1),
            )
        elif self.task == 'kinetics':
            self.unimol_global_mlp = nn.Sequential(
                Linear(512, self.dim_interact),
                ReLU(),
                LayerNorm(self.dim_interact),
                Linear(self.dim_interact, self.dim_interact),
                ReLU(),
                LayerNorm(self.dim_interact),
                Linear(self.dim_interact, self.dim_interact),
                ReLU(),
                LayerNorm(self.dim_interact),
                Linear(self.dim_interact, self.dim_interact),
            )
            self.esm_s_mlp = nn.Sequential(
                Linear(self.esm_feats, self.dim_interact),
                ReLU(),
                LayerNorm(self.dim_interact),
                Linear(self.dim_interact, self.dim_interact),
                ReLU(),
                LayerNorm(self.dim_interact),
                Linear(self.dim_interact, self.dim_interact),
                ReLU(),
                LayerNorm(self.dim_interact),
                Linear(self.dim_interact, self.dim_interact),
            )
            self.concat_mlp = nn.Sequential(
                nn.Linear(self.dim_interact * 2, self.dim_interact * 2),
                nn.ReLU(),
                LayerNorm(self.dim_interact * 2),
                nn.Linear(self.dim_interact * 2, self.dim_interact * 2),
                nn.ReLU(),
                LayerNorm(self.dim_interact * 2),
                nn.Linear(self.dim_interact * 2, self.dim_interact),
                nn.ReLU(),
                LayerNorm(self.dim_interact),
                nn.Linear(self.dim_interact, 1),
            )
        else:
            raise ValueError("Invalid task.")

        self.dropout = nn.Dropout2d(p=config.dropout)
        self.output_linear = Linear(1, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data_dict):
        lig_x = data_dict['lig_x']
        lig_batch = data_dict['lig_batch']

        protein_x = data_dict['protein_x']
        protein_batch = data_dict['protein_batch']
        protein_score = data_dict['output_score']

        #================================== ESM-2 =====================================
        protein_x = self.esm_s_mlp(protein_x)           # _, dim_interact

        if self.task == 'affinity':
            protein_x = protein_score.unsqueeze(1) * protein_x
            protein_score_sum = global_add_pool(protein_score, protein_batch)
            protein_x = protein_x / protein_score_sum[protein_batch].unsqueeze(1)
        elif self.task == 'kinetics':
            max_values = scatter(protein_score, protein_batch, reduce="max")
            protein_score_norm = protein_score / max_values[protein_batch]
            protein_x = protein_score_norm.unsqueeze(1) * protein_x
        else:
            raise ValueError("Invalid task.")
        
        protein_emb_global = global_add_pool(protein_x, protein_batch)
        
        #================================== Uni-Mol =====================================
        lig_emb_global = self.unimol_global_mlp(lig_x)

        if self.task == 'affinity':
            lig_emb_global = global_mean_pool(lig_emb_global, lig_batch)
        elif self.task == 'kinetics':
            lig_emb_global = global_add_pool(lig_emb_global, lig_batch)
        else:
            raise ValueError("Invalid task.")
        
        x_output = torch.cat([protein_emb_global, lig_emb_global], dim=1)
        x_output = self.concat_mlp(x_output).squeeze(-1)

        return x_output