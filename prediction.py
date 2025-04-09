import numpy as np
import pandas as pd
import argparse
import traceback
import pickle
import numpy as np
import os
import sys
sys.path.append("./")

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from diffusion.apply_noise import ForwardApplyNoise
from diffusion.de_noise import GraphTransformer
from diffusion.sampling import BackwardInference
from evaluation.metrics import GraphLoss
from visualization.draw import MolecularVisualization
from utils.file_utils import get_current_datetime, get_logger
from utils.data_utils import *
from utils.graph_utils import *




from _G2G_src.gcn_break_bond import RGCN, Linear, GCN


class Model1(nn.Module):
    def __init__(self, max_size, node_dim, bond_dim, args):
        """
        """
        super(Model1, self).__init__()
        self.max_size = max_size
        self.node_dim = node_dim
        self.bond_dim = bond_dim
        self.args = args   

        self.gcn = RGCN(self.node_dim, self.args.nhid, self.args.nout,
                         edge_dim=self.bond_dim,
                         num_layers=args.num_layers,
                         dropout=self.args.dropout,
                         normalization=False)
        inp_dim = 2 * self.args.nout
        #if self.args.has_class:
        inp_dim += 10
        if self.args.has_raw_node:
            inp_dim += 2 * self.node_dim
        if self.args.has_bond_feat:
            inp_dim += 4
        if self.args.has_graphemb:
            inp_dim += self.args.nout

        if not self.args.has_class:
            self.linear_class1 = Linear(self.args.nout, self.args.nout, self.args.pred_bias)
            self.linear_class2 = Linear(self.args.nout, 10, self.args.pred_bias) 

        self.linear1 = Linear(inp_dim, self.args.nout, self.args.pred_bias)

        self.linear2 = Linear(self.args.nout, 1, self.args.pred_bias)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        


    def forward(self, node_features, adj_features, classes=None):
        """
        Input: 
            node features: (B, max_size, 29)
            adj features: (B, 5, max_size, max_size)
            class: (optional) (B)
        Output:
            predicted label matrix: (B, max_size, max_size)
        """
        assert not (self.args.has_class and classes is None)
        batch_size = node_features.size(0)

        node_embeddings_org = self.gcn(node_features, adj_features) #(B, max_size, n_out)
        
        if self.args.has_raw_node:
            node_embeddings =torch.cat((node_embeddings_org, node_features), dim=-1)
        else:
            node_embeddings = node_embeddings_org

        if not self.args.has_class: #predict reaction class
            tmp_graphemb = torch.sum(node_embeddings_org, dim=1) #(B, n_out)
            pred_class_logits = self.act1(self.linear_class1(tmp_graphemb)) #(B, n_out)
            pred_class_logits = self.linear_class2(pred_class_logits) #(B, 10)
        

      
        node_embeddings_left = torch.unsqueeze(node_embeddings, 2).repeat(1, 1, self.max_size, 1)
        # (B, max_size, max_size_, n_out) [[1, 1],[2, 2]]
        node_embeddings_right = torch.unsqueeze(node_embeddings, 1).repeat(1, self.max_size, 1, 1)
        # (B, max_size_, max_size, n_out) [[1, 2],[1, 2]]
        node_pair_concat = torch.cat((node_embeddings_left, node_embeddings_right), 3)
        #if torch.isnan(node_pair_concat).any():
        #    print(node_pair_concat)
        #    print('concat1')
        #    exit(0)         
        # (B, max_size, max_size, 2 * n_out)
        node_pair_concat = node_pair_concat.view(-1, node_embeddings.size(-1) * 2)

        if self.args.has_class:
            assert classes is not None
            #print(classes.size())
            class_feat = torch.zeros((classes.size(0), 10)).cuda().scatter(1, (classes - 1).unsqueeze(1), 1.0).float().cuda()
            class_feat = class_feat.unsqueeze(1).repeat(1, self.max_size * self.max_size, 1).view(-1, 10)
            #print(class_feat.size(), node_pair_concat.size()) 
            node_pair_concat = torch.cat((node_pair_concat, class_feat), dim=1) #( *,2*n_out + 10)
        else:
            if classes is None:
                #class_feat_idx = torch.argmax(F.softmax(pred_class_logits, dim=-1), dim=-1) #(B,)
                #class_feat = torch.zeros((node_features.size(0), 10)).cuda().scatter(1, (class_feat_idx).unsqueeze(1), 1.0).float().cuda()
                class_feat = F.softmax(pred_class_logits, dim=-1)

            else:
                class_feat = torch.zeros((classes.size(0), 10)).cuda().scatter(1, (classes - 1).unsqueeze(1), 1.0).float().cuda()
            #class_feat = torch.zeros((batch_size, 10))
            if self.args.cuda:
                class_feat = class_feat.cuda()
            class_feat = class_feat.unsqueeze(1).repeat(1, self.max_size * self.max_size, 1).view(-1, 10)                
            node_pair_concat = torch.cat((node_pair_concat, class_feat), dim=1) #( *,2*n_out + 10)

        if self.args.has_bond_feat:
            #bond_feat = adj_features[:, :4].clone().permute(0,2,3,1) #(B, max_size, max_size, 4)
            bond_feat = adj_features.clone().permute(0,2,3,1) #(B, max_size, max_size, 4)

            bond_feat = bond_feat.contiguous().view(-1, 4)
            node_pair_concat = torch.cat((node_pair_concat, bond_feat), dim=1) # (*, 2*n_out + 10 + 4)
        #pred = torch.zeros (out.shape).cuda().scatter (1, a.unsqueeze (1), 1.0).long().cuda()
        if self.args.has_graphemb:
            graphemb = torch.sum(node_embeddings_org, dim=1, keepdim=True).repeat(1, self.max_size**2, 1) #(B, m**2, n_out)
            graphemb = graphemb.view(-1, self.args.nout)
            node_pair_concat = torch.cat((node_pair_concat, graphemb), dim=1) # (*, 3*n_out + 10 + 4)
        
        node_pair_concat = self.act1(self.linear1(node_pair_concat)) # (*, n_out)
        #if torch.isnan(node_pair_concat).any():
        #    print(node_pair_concat)
        #    print('concat2')
        #    exit(0)         
        node_pair_concat = self.linear2(node_pair_concat) # (*, 1)
        node_pair_concat = self.act2(node_pair_concat)
        #if torch.isnan(node_pair_concat).any():
        #    print(node_pair_concat)
        #    print('concat3')
        #    exit(0)            
        node_pair_concat = node_pair_concat.view(batch_size, self.max_size, self.max_size)
        if self.args.has_class:
            return node_pair_concat, None
        else:
            return node_pair_concat, pred_class_logits





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_time = get_current_datetime()


def arg_parses():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_pretrained', action="store_true", default=True)

    subparser = parser.add_argument_group("data preprocess")
    parser.add_argument('--dataset_name', type=str, default='QM9', choices=["MOSES, QM9"], help='pretrained dataset name')
    parser.add_argument('--do_preprocess', type=bool, default='True', help='if or not do preprocessing')

    subparser = parser.add_argument_group("data preload")
    subparser.add_argument('--train_batch_size', type=int, default=1024)
    subparser.add_argument('--train_epoch', type=int, default=1000)
    subparser.add_argument('--num_workers', type=int, default=8)
    
    subparser.add_argument('--lambda_XE', type=float, default=5.0)
    subparser.add_argument('--lr', type=float, default=2e-4)
    subparser.add_argument('--loss_interval', type=int, default=10)
    subparser.add_argument('--val_interval', type=int, default=1000)
    subparser.add_argument('--save_ckpt', action="store_true", default=False)
    subparser.add_argument('--from_ckpt', type=bool, default=False)
    subparser.add_argument('--ckpt_path', type=str, default=None)
    subparser.add_argument('--visualization', action="store_true", default=True)
    subparser.add_argument('--jointly', action="store_true", default=False)
    
    
    subparser = parser.add_argument_group("Forward_Apply_Noise")
    subparser.add_argument('--total_diffusion_T', type=int, default=500)
    subparser.add_argument("--noise_schedule", type=str, default="cosine", choices=['cosine'],
                           help="noise schedule")
    subparser.add_argument("--transition_mode", type=str, default="uniform", choices=['uniform', 'marginal'],
                           help="transition mode")
                           
    subparser = parser.add_argument_group("Backward_Training (Graph Transformer)")
    subparser.add_argument('--n_layers', type=int, default=9)
    subparser.add_argument('--dims_mlp', type=dict, default={'X': 256, 'E': 128, 'y': 128},
                           help="node-edge-wise mlp dimensions")
    subparser.add_argument('--dims_hidden', type=dict, default={'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128},
                           help="hidden dimensions of graph transformer")
    
    subparser = parser.add_argument_group("Sampling")
    subparser.add_argument('--init_mode', type=str, default="uniform", choices=['uniform', 'marginal', 'product', 'reactant'])

    return parser.parse_args()





if __name__ == '__main__':
    log = get_logger(__name__, current_time)
    args = arg_parses()
    args.do_preprocess = False
    args.save_ckpt = True
    log.info(f"{args}")


    dataset = RetroDiffDataset(dataset_name="USPTO50K",
                               train_batch_size=args.train_batch_size,
                               val_batch_size=args.val_batch_size,
                               test_batch_size=args.test_batch_size,
                               num_workers=args.num_workers,
                               )
    train_dataloader, val_dataloader, test_dataloader = dataset.prepare()
    train_dataiter = iter(train_dataloader)
    val_dataiter = iter(val_dataloader)
    test_dataiter = iter(test_dataloader)

    pred_model = Model1(self.max_size, 20, 5, args)



    print(f"Pre-trained Dataset Name: {args.dataset_name}")
    log.info(f"Pre-trained Dataset Name: {args.dataset_name}")
    print("Preprocessing Pre-trained Data ...")
    log.info("Preprocessing Pre-trained Data ...")
    dataprocess_module = DataProcess(args)
    if args.do_preprocess:
        dataprocess_module.preprocess()

    print("Loading Pre-trained Data ...")
    log.info("Loading Pre-trained Data ...")
    dataloader = dataprocess_module.preload()
    
    print("Loading Model ...")
    log.info("Loading Model ...")
    pretraining_module = PreTraining(args, dataloader)

    print("Start Pre-training!")
    log.info("Start Pre-training!")
    print(f"Training Batch Size: {args.train_batch_size}")
    log.info(f"Training Batch Size: {args.train_batch_size}")
    print(f"Pre-trained Data Size: {len(dataloader) * args.train_batch_size}")
    log.info(f"Pre-trained Data Size: {len(dataloader) * args.train_batch_size}")
    pretraining_module.train()
