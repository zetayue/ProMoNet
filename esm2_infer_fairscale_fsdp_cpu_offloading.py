# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import pathlib
import sys
import logging

import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

from esm import FastaBatchedDataset, pretrained, MSATransformer



def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        default='./sample.fasta',
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        default='./output_esm_embs',
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=1500, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=33,
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        default=["per_tok"],
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1500,
        help="truncate sequences longer than the given value",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(6)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # init the distributed world with world_size 1
    url = "tcp://localhost:23456"
    torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

    # initialize the model with FSDP wrapper
    fsdp_params = dict(
        mixed_precision=False,
        flatten_parameters=True,
        state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
        cpu_offload=True,  # enable cpu offloading
    )

    #fsdp_params = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        '''model, vocab = esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data
        )'''
        model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
        #batch_converter = vocab.get_batch_converter()
        model.eval()

        # Wrap each layer in FSDP separately
        for name, child in model.named_children():
            if name == "layers":
                for layer_name, layer in child.named_children():
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
        model = wrap(model)

    #model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    #model.eval()

    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    '''if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")'''

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            #if toks.size(0) > 1:
                #continue
            #else:
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device=device, non_blocking=True)
            try:
                out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            except RuntimeError as e:
                if e.args[0].startswith("CUDA out of memory"):
                    if len(toks) > 1:
                        logger.info(
                            f"Failed (CUDA out of memory) to predict batch of size {len(toks)}. "
                            "Try lowering `--max-tokens-per-batch`."
                        )
                    else:
                        logger.info(
                            f"Failed (CUDA out of memory) on sequence {labels[0]} of length {len(toks[0])}."
                        )

                    continue
                raise

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                    #print('size', result["representations"][33])
                    #print(fdsfas)
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()

                torch.save(
                    result,
                    args.output_file,
                )


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()