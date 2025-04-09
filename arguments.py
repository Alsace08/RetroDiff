import argparse
import torch
import numpy as np
import random
import os


def arg_parses():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_pretrained', action="store_true", default=False)

    # Whole Training
    subparser = parser.add_argument_group("Main")
    subparser.add_argument('--train_batch_size', type=int, default=120)
    subparser.add_argument('--val_batch_size', type=int, default=120)
    subparser.add_argument('--test_batch_size', type=int, default=120)
    subparser.add_argument('--train_epoch', type=int, default=500)
    subparser.add_argument('--num_workers', type=int, default=8)
    subparser.add_argument('--test_after_train', action="store_true", default=False)
    
    subparser.add_argument('--lambda_XE', type=float, default=5.0)
    subparser.add_argument('--lr', type=float, default=1e-4)
    subparser.add_argument('--loss_interval', type=int, default=10)
    subparser.add_argument('--val_interval', type=int, default=500)

    subparser.add_argument('--save_ckpt', action="store_true", default=False)
    subparser.add_argument('--visualization', action="store_true", default=False)
    subparser.add_argument('--from_ckpt', action="store_true", default=False)
    subparser.add_argument("--ckpt_path", type=str, default="/sharefs/yiming-w/RetroDiff/experiments/checkpoints/2023_08_28_14:03:46/model_step_33000.ckpt")
    subparser.add_argument('--dp', action="store_true", default=False)
    subparser.add_argument('--gpus', nargs='+', type=int, help='List of GPUs for training',default=[0,1,2,3,4,5,6,7])

    subparser.add_argument('--jointly', action="store_true", default=False)
    subparser.add_argument('--to_group_given_product', action="store_true", default=False)
    subparser.add_argument('--to_exbond_given_product_and_group', action="store_true", default=False)
    
    
    # Diffusion Processes
    subparser = parser.add_argument_group("Forward_Apply_Noise")
    subparser.add_argument('--total_diffusion_T', type=int, default=50)
    subparser.add_argument('--SDE_T', type=int, default=5)
    subparser.add_argument("--noise_schedule", type=str, default="cosine", choices=['cosine'],
                           help="noise schedule")
    subparser.add_argument("--transition_mode", type=str, default="marginal", choices=['uniform', 'marginal'],
                           help="transition mode")
                     
    subparser = parser.add_argument_group("Backward_Training (Graph Transformer)")
    subparser.add_argument('--n_layers', type=int, default=9)
    subparser.add_argument('--dims_mlp', type=dict, default={'X': 256, 'E': 128, 'y': 128},
                           help="node-edge-wise mlp dimensions")
    subparser.add_argument('--dims_hidden', type=dict, default={'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128},
                           help="hidden dimensions of graph transformer")
    
    subparser = parser.add_argument_group("Sampling")
    subparser.add_argument('--init_mode', type=str, default="marginal", choices=['uniform', 'marginal', 'product', 'reactant'])
    subparser.add_argument('--sample_atom_num', type=int, default=10)

    return parser.parse_args()
