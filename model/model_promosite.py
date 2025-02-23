import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_batch

from torch.nn import LayerNorm, Linear, ReLU

from utils import BesselBasisLayer, MLP_SiLU
from model import Config


class ProMoSite(nn.Module):
    """
    GNN model
    """
    def __init__(self, config: Config):
        super(ProMoSite, self).__init__()
        self.esm_feats = 1280
        self.dim_interact = config.dim_interact
        self.dim_interact_simple = 1024
        self.dim_pair = config.dim_pair
        self.n_trigonometry_module_stack = config.n_module
        self.factor = config.factor

        self.layernorm = torch.nn.LayerNorm(self.dim_interact)

        self.unimol_mlp = nn.Sequential(
            LayerNorm(512),
            Linear(512, self.dim_interact),
            ReLU(),
            Linear(self.dim_interact, self.dim_interact),
        )
        
        self.unimol_global_mlp = nn.Sequential(
            LayerNorm(512),
            Linear(512, self.dim_interact_simple),
            ReLU(),
            Linear(self.dim_interact_simple, self.dim_interact_simple),
        )

        self.protein_linear = Linear(self.dim_interact_simple, self.dim_interact)
        
        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            Linear(self.esm_feats, self.dim_interact_simple),
            ReLU(),
            Linear(self.dim_interact_simple, self.dim_interact_simple),
        )

        self.concat_mlp = nn.Sequential(
            LayerNorm(self.dim_interact_simple * 2),
            nn.Linear(self.dim_interact_simple * 2, self.dim_interact_simple),
            nn.ReLU(),
            nn.Linear(self.dim_interact_simple, 1),
        )

        self.protein_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, self.dim_interact_simple),
            nn.ReLU(),
            nn.Linear(self.dim_interact_simple, self.dim_interact_simple),
            nn.ReLU(),
            nn.Linear(self.dim_interact_simple, 1),
        )

        self.rbf_protein_pair = BesselBasisLayer(16, 1, envelope_exponent=5)
        self.rbf_lig_pair = BesselBasisLayer(16, 15, envelope_exponent=5)

        self.protein_pair_mlp = MLP_SiLU([16, self.dim_interact])
        self.lig_pair_mlp = MLP_SiLU([16, self.dim_interact])

        self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=self.dim_interact, c=self.dim_pair) for _ in range(self.n_trigonometry_module_stack)])
        self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=self.dim_interact) for _ in range(self.n_trigonometry_module_stack)])
        self.tranistion = Transition(embedding_channels=self.dim_interact, n=2)

        self.dropout = nn.Dropout2d(p=config.dropout)
        self.linear = Linear(self.dim_interact, 1)

    def forward(self, data_dict):
        lig_x = data_dict['lig_x']
        lig_batch = data_dict['lig_batch']

        protein_x = data_dict['protein_x']
        protein_batch = data_dict['protein_batch']

        protein_pair = data_dict['protein_pair']
        lig_pair = data_dict['lig_pair']
        lig_id = data_dict['lig_id']

        #================================== ESM-2 =====================================
        protein_emb_no_pair = self.esm_s_mlp(protein_x)           # _, dim_interact_simple
        protein_emb = self.protein_linear(protein_emb_no_pair)    # _, dim_interact

        y_protein = self.protein_mlp(protein_x).squeeze(-1)

        if lig_id != 'NULL':
            #================================== Uni-Mol =====================================
            lig_emb_global = global_add_pool(lig_x, lig_batch)

            lig_emb_global = self.unimol_global_mlp(lig_emb_global)
            lig_emb = self.unimol_mlp(lig_x)

            #================================== Interaction Module =====================================
            protein_emb_batched, protein_emb_mask = to_dense_batch(protein_emb, protein_batch)
            lig_emb_batched, lig_emb_mask = to_dense_batch(lig_emb, lig_batch)

            protein_emb_batched = self.layernorm(protein_emb_batched)
            lig_emb_batched = self.layernorm(lig_emb_batched)

            z = torch.einsum("bik,bjk->bijk", protein_emb_batched, lig_emb_batched)
            z_mask = torch.einsum("bi,bj->bij", protein_emb_mask, lig_emb_mask)

            protein_pair = self.rbf_protein_pair(protein_pair.squeeze(-1))      # BS*L*L, 1, 16
            lig_pair = self.rbf_lig_pair(lig_pair.squeeze(-1))         # BS*L*L, 1, 16

            protein_pair = self.protein_pair_mlp(protein_pair)    # BS, L, L, dim
            lig_pair = self.lig_pair_mlp(lig_pair)                # BS, L, L, dim

            for i_module in range(self.n_trigonometry_module_stack):
                z = z + self.dropout(self.protein_to_compound_list[i_module](z, protein_pair, lig_pair, z_mask.unsqueeze(-1)))
                z = z + self.dropout(self.triangle_self_attention_list[i_module](z, z_mask))
                z = self.tranistion(z)

            b = self.linear(z).squeeze(-1)
            b = b * z_mask
            b = torch.sum(b, dim=-1)
            b = b / torch.sum(lig_emb_mask, dim=-1).unsqueeze(1)
            y_pred = b[protein_emb_mask]

            x_lig_extend = lig_emb_global[protein_batch].view(-1, self.dim_interact_simple)

            x_output = torch.cat([protein_emb_no_pair, x_lig_extend], dim=1)
            x_output = self.concat_mlp(x_output).squeeze(-1)

            return y_pred * 0.5 + x_output * 0.5 + y_protein * self.factor
        else:
            return y_protein

class TriangleProteinToCompound_v2(torch.nn.Module):
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        self.gate_linear1 = Linear(embedding_channels, c)
        self.gate_linear2 = Linear(embedding_channels, c)

        self.linear1 = Linear(embedding_channels, c)
        self.linear2 = Linear(embedding_channels, c)

        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)
 
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(compound_pair)

        g = self.ending_gate_linear(z).sigmoid()
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)
        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z
    

class TriangleSelfAttentionRowWise(torch.nn.Module):
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.layernorm = torch.nn.LayerNorm(embedding_channels)

        self.linear_q = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = Linear(embedding_channels, self.all_head_size, bias=False)
        self.g = Linear(embedding_channels, self.all_head_size)
        self.final_linear = Linear(self.all_head_size, embedding_channels)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j
        z = self.layernorm(z)
        p_length = z.shape[1]
        batch_n = z.shape[0]

        z_i = z
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        # q, k, v of shape b, j, h, c
        q = self.reshape_last_dim(self.linear_q(z_i)) #  * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z_i))
        v = self.reshape_last_dim(self.linear_v(z_i))
        logits = torch.einsum('biqhc,bikhc->bihqk', q, k) + attention_mask_i
        weights = nn.Softmax(dim=-1)(logits)
        # weights of shape b, h, j, j
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v)
        g = self.reshape_last_dim(self.g(z_i)).sigmoid()
        output = g * weighted_avg
        new_output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*new_output_shape)
        # output of shape b, j, embedding.
        z = output
        z = self.final_linear(z) * z_mask.unsqueeze(-1)
        return z


class Transition(torch.nn.Module):
    def __init__(self, embedding_channels=256, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n*embedding_channels)
        self.linear2 = Linear(n*embedding_channels, embedding_channels)
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z