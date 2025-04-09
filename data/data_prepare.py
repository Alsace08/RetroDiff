# -*- coding:utf-8 -*-
import os
import tqdm
import zipfile
import pickle
import numpy as np
from functools import partial


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch_geometric.utils import to_dense_batch, to_dense_adj

from rdkit import Chem

import sys
sys.path.append('../')
from utils.graph_utils import to_dense_atom_batch, to_dense_bond_batch



def get_file_names(path):
    list_name = []
    for file in os.listdir(path):
        list_name.append(os.path.join(path, file))
    return list_name


def merge_file_data(filename_list):
    data_list = []
    for filename in filename_list:
        f = open(filename, 'rb')
        inf = pickle.load(f)
        data_list.append(inf)
    
    return data_list



class RetroDiffDataset:
    def __init__(self,
                 dataset_name: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 num_workers: int,
                 ):
                 
        self.train_filedir = os.path.join('./data', dataset_name, 'train')
        self.train_filename_list = get_file_names(self.train_filedir)
        self.val_filedir = os.path.join('./data', dataset_name, 'valid')
        self.val_filename_list = get_file_names(self.val_filedir)
        self.test_filedir = os.path.join('./data', dataset_name, 'test')
        self.test_filename_list = get_file_names(self.test_filedir)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def prepare(self):
        # self.train_data = merge_file_data(self.train_filename_list)
        # self.val_data = merge_file_data(self.val_filename_list)
        # self.test_data = merge_file_data(self.test_filename_list)

        # self.train_dataset = RetroDiffDataModule(self.train_data)
        # self.val_dataset = RetroDiffDataModule(self.val_data)
        # self.test_dataset = RetroDiffDataModule(self.test_data)

        self.train_dataset = RetroDiffDataModule(self.train_filename_list)
        self.val_dataset = RetroDiffDataModule(self.val_filename_list)
        self.test_dataset = RetroDiffDataModule(self.test_filename_list)

        return \
        DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=True 
        ), \
        DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=True 
        ), \
        DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=True 
        )



    def collator(self, datamodule):
        rxn_types, \
        p_smiles, p_atom_nums, p_atom_symbols, p_bond_symbols, \
        r_smiles, r_atom_nums, r_atom_symbols, r_bond_symbols, \
        s_smiles, s_atom_nums, s_atom_symbols, s_bond_symbols, \
        g_smiles, g_atom_nums, g_atom_symbols, g_bond_symbols, \
        product_to_group_bond = zip(*datamodule)
        
        rxn_types = torch.stack([torch.tensor(types) for types in rxn_types])
        bs = rxn_types.size(0)
        
        # product information
        p_smiles = [smiles for smiles in p_smiles]
        p_atom_nums = torch.stack([torch.tensor(num) for num in p_atom_nums])
        p_atom_symbols, p_atom_masks = to_dense_atom_batch(bs, p_atom_nums, p_atom_symbols)
        p_bond_symbols = to_dense_bond_batch(bs, p_atom_nums, p_bond_symbols)


        # reactant information
        r_smiles = [smiles for smiles in r_smiles]
        r_atom_nums = torch.stack([torch.tensor(num) for num in r_atom_nums])
        r_atom_symbols, r_atom_masks = to_dense_atom_batch(bs, r_atom_nums, r_atom_symbols)
        r_bond_symbols = to_dense_bond_batch(bs, r_atom_nums, r_bond_symbols)


        # synthons information
        s_smiles = [smiles for smiles in s_smiles]
        s_atom_nums = torch.stack([torch.tensor(num) for num in s_atom_nums])
        s_atom_symbols, s_atom_masks = to_dense_atom_batch(bs, s_atom_nums, s_atom_symbols)
        s_bond_symbols = to_dense_bond_batch(bs, s_atom_nums, s_bond_symbols)


        # group information
        g_smiles = [smiles for smiles in g_smiles]
        g_atom_nums = torch.stack([torch.tensor(num) for num in g_atom_nums])
        g_atom_symbols, g_atom_masks = to_dense_atom_batch(bs, g_atom_nums, g_atom_symbols)
        g_bond_symbols = to_dense_bond_batch(bs, g_atom_nums, g_bond_symbols)


        # global information: empty
        p_y = torch.empty((0))
        r_y = torch.empty((0))
        s_y = torch.empty((0))
        g_y = torch.empty((0))

        product_to_group_bond = [item for item in product_to_group_bond]

        '''
        rxn_type: reactant type -> (bs)
        
        p_atom_nums: product atom numbers -> (bs)
        p_atom_symbols: product atom symbols -> (bs, p_atom_num_max, d_atom)
        p_atom_mask: product atom masks -> (bs, p_atom_num_max, p_atom_num_max): bool type
        p_bond_symbols: product bond symbols -> (bs, p_atom_num_max, p_atom_num_max, d_bond)
        p_adj: product graph adjacency matrix -> (bs, p_atom_num_max, p_atom_num_max)
        
        r_atom_nums: reactant atom numbers -> (bs)
        r_atom_symbols: reactant atom symbols -> (bs, r_atom_num_max, d_atom)
        r_atom_mask: reactant atom masks -> (bs, r_atom_num_max, r_atom_num_max): bool type
        r_bond_symbols: reactant bond symbols -> (bs, r_atom_num_max, r_atom_num_max, d_bond)
        r_adj: reactant graph adjacency matrix -> (bs, r_atom_num_max, r_atom_num_max)
        '''
        return Batch(
            rxn_types=rxn_types,
            p_smiles=p_smiles, p_atom_nums=p_atom_nums, p_atom_symbols=p_atom_symbols, p_atom_masks=p_atom_masks, p_bond_symbols=p_bond_symbols,
            r_smiles=r_smiles, r_atom_nums=r_atom_nums, r_atom_symbols=r_atom_symbols, r_atom_masks=r_atom_masks, r_bond_symbols=r_bond_symbols,
            s_smiles=s_smiles, s_atom_nums=s_atom_nums, s_atom_symbols=s_atom_symbols, s_atom_masks=s_atom_masks, s_bond_symbols=s_bond_symbols,
            g_smiles=g_smiles, g_atom_nums=g_atom_nums, g_atom_symbols=g_atom_symbols, g_atom_masks=g_atom_masks, g_bond_symbols=g_bond_symbols,
            p_y=p_y, r_y=r_y, s_y=s_y, g_y=g_y,
            product_to_group_bond=product_to_group_bond
        )


class RetroDiffDataModule(Dataset):
    def __init__(self,
                 data):
        self.data = data

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        ex = pickle.load(open(self.data[index], 'rb'))
        #ex = self.data[index]
        rxn_type = ex["rxn_type"]

        p_smiles = ex["product_smiles"]
        p_atom_num = ex["product_atom_num"]
        p_atom_symbols = [atom_features["symbol"] for atom_features in ex["product_atom_features"]]
        p_bond_symbols = ex["product_bond_features"]
        
        r_smiles = ex["reactant_smiles"]
        r_atom_num = ex["reactant_atom_num"]
        r_atom_symbols = [atom_features["symbol"] for atom_features in ex["reactant_atom_features"]]
        r_bond_symbols = ex["reactant_bond_features"]

        s_smiles = ex["synthons_smiles"]
        s_atom_num = ex["synthons_atom_num"]
        s_atom_symbols = [atom_features["symbol"] for atom_features in ex["synthons_atom_features"]]
        s_bond_symbols = ex["synthons_bond_features"]

        g_smiles = ex["group_smiles"]
        g_atom_num = ex["group_atom_num"]
        g_atom_symbols = [atom_features["symbol"] for atom_features in ex["group_atom_features"]]
        g_bond_symbols = ex["group_bond_features"]

        product_to_group_bond = ex["product_to_group_bond"]


        return rxn_type, \
              p_smiles, p_atom_num, p_atom_symbols, p_bond_symbols, \
              r_smiles, r_atom_num, r_atom_symbols, r_bond_symbols, \
              s_smiles, s_atom_num, s_atom_symbols, s_bond_symbols, \
              g_smiles, g_atom_num, g_atom_symbols, g_bond_symbols, \
              product_to_group_bond




class RetroDiffDataInfos:
    def __init__(self):
        self.id2atom = [None, 'C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br', 'Li', 'Na', 'K', 'Mg', 'B', 'Sn', 'I', 'Se', 'Cu', 'Zn']
        self.atom2id = {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'H':6, 'Si':7, 'P':8, 'Cl':9, 'Br':10, 'Li':11, 'Na':12, 'K':13, 'Mg':14, 'B':15, 'Sn':16, 'I':17, 'Se':18, 'Cu':19, 'Zn': 20}
        self.id2weights = [0, 12.011, 14.007, 15.999, 32.060, 18.998, 1.008, 28.086, 30.974, 35.450, 79.904, 6.940, 22.990, 39.098, 24.305, 10.810, 118.710, 126.900, 78.971, 63.546, 65.380]
        self.bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
        self.max_weight = 300
        
        # self.x_marginals = torch.tensor([8.5247e-01, 5.3982e-02, 6.2281e-02, 5.7737e-03, 1.2847e-02, 0.0000e+00, 4.5515e-04, 1.5255e-04, 7.5046e-03, 3.0400e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.6098e-05, 6.8260e-04, 7.1125e-05, 6.3962e-04, 4.5238e-06, 1.1812e-05, 1.3823e-05]) 
        # self.e_marginals = torch.tensor([0.88162,  0.11062,  5.9875e-03,  1.7758e-03, 0])
        self.x_marginals = torch.tensor([1.0] + [0.0] * (len(self.id2atom) - 1)) 
        self.e_marginals = torch.tensor([1.0] + [0.0] * 4)
        
    def compute_io_dims(self, dataloader):
        dataiter = iter(dataloader)
        batch = next(dataiter)
        assert (batch.p_atom_symbols.size(-1) == batch.r_atom_symbols.size(-1))
        assert (batch.p_bond_symbols.size(-1) == batch.r_bond_symbols.size(-1))
        assert (batch.p_y.size(-1) == batch.r_y.size(-1))
        
        bs = batch.p_atom_symbols.size(0)
        X = batch.p_atom_symbols
        E = batch.p_bond_symbols
        y = batch.p_y
        output_dims = {'X': X.size(-1), 'E': E.size(-1), 'y': y.size(-1) + 1}
        
        data = {'X_t': X, 'E_t': E, 'y_t': y, 't_int': torch.zeros(bs, 1)}
        extra_data = RetroDiffExtraFeature(data).to_extra_data(RetroDiffDataInfos())
        input_dims = {'X': output_dims['X'] + extra_data['X'].size(-1), 
                      'E': output_dims['E'] + extra_data['E'].size(-1), 
                      'y': output_dims['y'] - 1 + extra_data['y'].size(-1)}
                      
        return input_dims, output_dims
        
        


class RetroDiffExtraFeature:
    def __init__(self, noisy_data):
        self.noisy_X = noisy_data['X_t']
        self.noisy_E = noisy_data['E_t'].to(self.noisy_X.device)
        self.noisy_y = noisy_data['y_t'].to(self.noisy_X.device)
        self.t = noisy_data['t_int'].to(self.noisy_X.device)
        
        
    def to_extra_data(self, datainfos):
        X_weight = self.get_weight(self.noisy_X, datainfos)
        extra_y = torch.cat((self.t, X_weight), dim=1)
        
        #extra_X = self.get_valency(self.noisy_E)
        extra_X = torch.zeros((*self.noisy_X.shape[:-1], 0)).type_as(self.noisy_X)
        extra_E = torch.zeros((*self.noisy_E.shape[:-1], 0)).type_as(self.noisy_E)
        
        return {"X": extra_X, "E": extra_E, "y": extra_y}
        
        
    def get_weight(self, X, datainfos):
        '''get global extra feature'''
        X_symbols = torch.argmax(self.noisy_X, dim=-1)     # (bs, n)
        X_weights = torch.Tensor(datainfos.id2weights)[X_symbols].type_as(X)            # (bs, n)
        molecular_weight = X_weights.sum(dim=-1).unsqueeze(-1) / datainfos.max_weight     # (bs, 1)
        
        return molecular_weight
        
    def get_valency(self, E):
        '''get atom extra feature 1'''
        orders = torch.tensor([0, 1, 2, 3, 1.5], device=E.device).reshape(1, 1, 1, -1)
        E = E * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        
        return valencies



class Batch:
    def __init__(self,
                 rxn_types,
                 p_smiles, p_atom_nums, p_atom_symbols, p_atom_masks, p_bond_symbols,
                 r_smiles, r_atom_nums, r_atom_symbols, r_atom_masks, r_bond_symbols,
                 s_smiles, s_atom_nums, s_atom_symbols, s_atom_masks, s_bond_symbols,
                 g_smiles, g_atom_nums, g_atom_symbols, g_atom_masks, g_bond_symbols,
                 p_y, r_y, s_y, g_y,
                 product_to_group_bond
                 ):
        self.rxn_types = rxn_types
        
        self.p_smiles = p_smiles
        self.p_atom_nums = p_atom_nums
        self.p_atom_symbols = p_atom_symbols
        self.p_atom_masks = p_atom_masks
        self.p_bond_symbols = p_bond_symbols
        
        self.r_smiles = r_smiles
        self.r_atom_nums = r_atom_nums
        self.r_atom_symbols = r_atom_symbols
        self.r_atom_masks = r_atom_masks
        self.r_bond_symbols = r_bond_symbols

        self.s_smiles = s_smiles
        self.s_atom_nums = s_atom_nums
        self.s_atom_symbols = s_atom_symbols
        self.s_atom_masks = s_atom_masks
        self.s_bond_symbols = s_bond_symbols

        self.g_smiles = g_smiles
        self.g_atom_nums = g_atom_nums
        self.g_atom_symbols = g_atom_symbols
        self.g_atom_masks = g_atom_masks
        self.g_bond_symbols = g_bond_symbols
        
        self.p_y = p_y
        self.r_y = r_y
        self.s_y = s_y
        self.g_y = g_y

        self.product_to_group_bond = product_to_group_bond


    def to_device(self, device):
        self.rxn_types = self.rxn_types.to(device)
        
        self.p_atom_nums = self.p_atom_nums.to(device)
        self.p_atom_symbols = self.p_atom_symbols.to(device)
        self.p_atom_masks = self.p_atom_masks.to(device)
        self.p_bond_symbols = self.p_bond_symbols.to(device)
        
        self.r_atom_nums = self.r_atom_nums.to(device)
        self.r_atom_symbols = self.r_atom_symbols.to(device)
        self.r_atom_masks = self.r_atom_masks.to(device)
        self.r_bond_symbols = self.r_bond_symbols.to(device)

        self.s_atom_nums = self.s_atom_nums.to(device)
        self.s_atom_symbols = self.s_atom_symbols.to(device)
        self.s_atom_masks = self.s_atom_masks.to(device)
        self.s_bond_symbols = self.s_bond_symbols.to(device)

        self.g_atom_nums = self.g_atom_nums.to(device)
        self.g_atom_symbols = self.g_atom_symbols.to(device)
        self.g_atom_masks = self.g_atom_masks.to(device)
        self.g_bond_symbols = self.g_bond_symbols.to(device)
        
        self.p_y = self.p_y.to(device)
        self.r_y = self.r_y.to(device)
        self.s_y = self.s_y.to(device)
        self.g_y = self.g_y.to(device)
  

        return self

    def __len__(self):
        return self.rxn_types.size(0)

