# Sequence-based Drug-Target Complex Pre-training Enhances Protein-Ligand Binding Process Predictions Tackling Crypticity

## Environment Setup
The environment can be built based on the [NVIDIA container image for PyTorch, release 23.01](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-01.html). Running NVIDIA container image requires **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** installed to run with GPU support.

Inside the docker container, following commands are needed to install additional required packages:
```
pip install fair-esm
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
pip install --verbose torch_scatter
pip install --verbose torch_sparse
pip install biopython rdkit==2023.9.5 pdbeccdutils pytorch-lightning==1.5.10 wandb einops==0.6.1
git clone https://github.com/dptech-corp/Uni-Core/tree/8f0e59dcaea1619a5c79d6086072fb9d24872f44
cd ./Uni-Core
python setup.py install
pip install pymatgen==2023.8.10 unimol_tools
```

## Datasets

**Chemical Component Dictionary:**

Download [mmCIF file](https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz). Unzip `Components-rel-alt.cif` under `./data`.

**scPDB:**

Download [scPDB dataset](http://bioinfo-pharma.u-strasbg.fr/scPDB/). Unzip folder `scPDB` under `./data`.

Compute ESM-2 embeddings for scPDB based on the fasta information provided:

    python esm2_infer_fairscale_fsdp_cpu_offloading.py esm2_t33_650M_UR50D ./data/scPDB_unique_fasta.fasta ./data/esm/scPDB --repr_layers 33 --include per_tok contacts --truncation_seq_length 1600 --toks_per_batch 1600

**COACH420 and HOLO4K:**

Download [protein fasta files](https://github.com/rdk/p2rank-datasets) and [ligand mol2 files](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/rishal_aggarwal_alumni_iiit_ac_in/EoJSrvuiKPlAluOJLjTzfpcBT2fVRdq8Sr4BMmil0_tvHw?e=kXUss4). Unzip folders `p2rank-datasets` and `DeepPocket_data` under `./data`.

Compute ESM-2 embeddings for COACH420 and HOLO4K based on the fasta information provided:

    python esm2_infer_fairscale_fsdp_cpu_offloading.py esm2_t33_650M_UR50D ./data/coach420_unique_fasta.fasta ./data/esm/coach420 --repr_layers 33 --include per_tok contacts --truncation_seq_length 1500 --toks_per_batch 1500

    python esm2_infer_fairscale_fsdp_cpu_offloading.py esm2_t33_650M_UR50D ./data/holo4k_unique_fasta.fasta ./data/esm/holo4k --repr_layers 33 --include per_tok contacts --truncation_seq_length 1500 --toks_per_batch 1500

**PocketMiner:**

Compute ESM-2 embeddings for PocketMiner based on the fasta information provided:

    python esm2_infer_fairscale_fsdp_cpu_offloading.py esm2_t33_650M_UR50D ./data/pocketminer_all_fasta.fasta ./data/esm/pocketminer --repr_layers 33 --include per_tok contacts --truncation_seq_length 1500 --toks_per_batch 1500

**PDBbind v2020:**

Compute ESM-2 embeddings for PDBbind v2020 based on the fasta information provided:

    python esm2_infer_fairscale_fsdp_cpu_offloading.py esm2_t33_650M_UR50D ./data/pdbbind2020/pdbbind2020_fasta.fasta ./data/esm/pdbbind2020 --repr_layers 33 --include per_tok contacts --truncation_seq_length 3000 --toks_per_batch 3000

**Curated binding kinetics dataset:**

Compute ESM-2 embeddings for binding kinetics dataset based on the fasta information provided:

    python esm2_infer_fairscale_fsdp_cpu_offloading.py esm2_t33_650M_UR50D ./data/binding_kinetics/binding_kinetics_fasta.fasta ./data/esm/binding_kinetics --repr_layers 33 --include per_tok contacts --truncation_seq_length 700 --toks_per_batch 700

## How to Run

### General binding site prediction

**Training on scPDB for inference on COACH420:**

    python train_general_bs.py --task 'coach420' --lr 5e-5 --batch_size 8 --epochs 10 --seed 705 --gpu 0 --wandb

**Training on scPDB for inference on HOLO4K:**

    python train_general_bs.py --task 'holo4k' --lr 5e-5 --batch_size 8 --epochs 10 --seed 705 --gpu 0 --wandb

**Inference on COACH420:**

    python inference_coach420.py --lr 5e-5 --saved saved_model --seed 705 --gpu 0 --wandb

**Inference on HOLO4K:**

    python inference_holo4k.py --lr 5e-5 --saved saved_model --seed 705 --gpu 0 --wandb

### Cryptic binding site prediction

**Training on scPDB, with validation and testing on PocketMiner:**

    python train_cryptic_bs.py --lr 1e-6 --batch_size 8 --epochs 40 --seed 705 --gpu 0 --data_txt 'scPDB_subset_pocketminer.txt' --max_length 1280 --wandb

### Pre-training on scPDB for downstream binding affinity and kinetics prediction

    python train_cryptic_bs.py --lr 1e-6 --batch_size 8 --epochs 40 --seed 705 --gpu 0 --data_txt 'scPDB_full.txt' --max_length 1600 --wandb

### Binding Affinity prediction

**Inference on PDBbind v2020 using pre-trained ProMoSite to obtain scores:**

    python inference_affinity_score.py --lr 1e-6 --saved saved_model --seed 705 --gpu 0 --wandb

**Training (fine-tuning), validation and testing on PDBbind v2020 using ProMoBind:**

    python train_affinity.py --lr 5e-5 --batch_size 256 --epochs 300 --seed 705 --gpu 0 --wandb

### Binding Kinetics prediction

**Inference on binding kinetics dataset using pre-trained ProMoSite to obtain scores:**

    python inference_kinetics_score.py --lr 1e-6 --saved saved_model --seed 705 --gpu 0 --wandb

**Training (fine-tuning), validation and testing on binding kinetics dataset using ProMoBind:**

    python train_kinetics.py --lr 5e-6 --batch_size 4 --epochs 500 --seed 705 --gpu 0 --wandb