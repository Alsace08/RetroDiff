### https://github.com/uta-smile/RetroXpert/blob/canonical_product/preprocessing.py

import numpy as np
import pandas as pd
import argparse
import os
import re
import pickle

import torch
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

import sys
sys.path.append('./')
from utils.data_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='USPTO50K',
                    help='dataset: USPTO50K or USPTO-full')

args = parser.parse_args()



### preprocess the whole dataset

def preprocess(save_dir, reactants, products, reaction_types=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 21 atoms
    symbol_list = [None, 'C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br', 'Li', 'Na', 'K', 'Mg', 'B', 'Sn', 'I', 'Se', 'Cu', 'Zn']

    exception = 0
    len0 = 0
    len1 = 0
    len2 = 0
    len3 = 0
    len4 = 0
    len4more = 0
    for index in tqdm(range(len(reactants))):
        rxn_type = reaction_types[index]
        product = products[index]
        reactant = reactants[index]
        synthons_list, break_atom_pair = get_synthons(product, reactant)
        prod_to_group_bond, flag = get_reaction_atom(product, reactant)
        #print(break_atom_pair)
        synthons = ".".join(synthons_list)
        group = get_external_group(reactant)

        # if flag == 0:
        #     print(product)
        #     print(reactant)
        #     print(group)
        #     print("--")

        if len(prod_to_group_bond) == 0:
            len0 += 1
        if len(prod_to_group_bond) == 1:
            len1 += 1
        if len(prod_to_group_bond) == 2:
            len2 += 1
        if len(prod_to_group_bond) == 3:
            len3 += 1
        if len(prod_to_group_bond) == 4:
            len4 += 1
        if len(prod_to_group_bond) > 4:
            len4more += 1


        # if flag == 1 and len(prod_to_group_bond) > 1:
        #     print(product)
        #     print(reactant)
        #     print(prod_to_group_bond)

        
        product_mol = Chem.MolFromSmiles(product)
        reactant_mol = Chem.MolFromSmiles(reactant)
        group_mol = Chem.MolFromSmiles(group)


        product_atom_num = len(product_mol.GetAtoms())
        reactant_atom_num = len(reactant_mol.GetAtoms())
        synthons_atom_num = product_atom_num

        try:
            group_atom_num = len(group_mol.GetAtoms())
        except Exception:
            print(traceback.format_exc())
            exception += 1
            continue

        if "train" in save_dir and group_atom_num > 10:
            exception += 1
            continue
        if "train" in save_dir and flag == 0:
            exception += 1
            continue
        if "train" in save_dir and len(prod_to_group_bond) > 2 and len(break_atom_pair) > 2:
            exception += 1
            continue
        
        product_bond_features = get_bond_features(product_mol)
        product_atom_features = get_atom_features(product_mol, symbol_list)
        reactant_bond_features = get_bond_features(reactant_mol)
        reactant_atom_features = get_atom_features(reactant_mol, symbol_list)
        synthons_atom_features = product_atom_features
        synthons_bond_features = product_bond_features
        group_bond_features = get_bond_features(group_mol)
        group_atom_features = get_atom_features(group_mol, symbol_list)


        # expand group to max_num
        expand_group_atom_num = 10
        expand_group_atom_features = group_atom_features
        for i in range(10 - group_atom_num):
            expand_group_atom_features.append({'symbol': [1] + [0] * (len(symbol_list) - 1)})
        expand_group_atom_features = expand_group_atom_features[:10]
         
        expand_group_bond_features = torch.zeros((10, 10, 5), dtype=int)
        expand_group_bond_features[:, :, 0] = 1
        for i in range(group_atom_num):
            if i >= 10:
                break
            for j in range(group_atom_num):
                if j >= 10:
                    break
                expand_group_bond_features[i, j, :] = torch.tensor(group_bond_features[i][j])
        expand_group_bond_features.tolist()

        # advance reaction atom
        group_bond_features, group_atom_features, prod_to_group_bond = advance_reaction_atom(group_bond_features, group_atom_features, prod_to_group_bond)

        # synthons
        if len(break_atom_pair) == 0:
            start_id = end_id = -1
        for break_pair in break_atom_pair:
            start_id = break_pair[0]
            end_id = break_pair[1]
            synthons_bond_features[start_id][end_id] = [1,0,0,0,0]
            synthons_bond_features[end_id][start_id] = [1,0,0,0,0]
        
        # print(product)
        # print(reactant)
        # print(group)
        # print(prod_to_group_bond)
        # print("---")

        rxn_data = {
            'rxn_type': rxn_type,
            # product
            'product_smiles': product,
            'product_atom_num': product_atom_num,  # n_atom
            #'product_adj': product_adj,  # (n_atom, n_atom)
            'product_bond_features': product_bond_features,  # (n_atom, n_atom, d_bond)
            'product_atom_features': product_atom_features,  # (n_atom, n_atom, d_atom)
            # reactant
            'reactant_smiles': reactant,
            'reactant_atom_num': reactant_atom_num,
            'reactant_bond_features': reactant_bond_features,
            'reactant_atom_features': reactant_atom_features,
            #'reactant_adj': reactant_adj,
            # synthons
            "synthons_smiles": synthons,
            'synthons_atom_num': synthons_atom_num,
            'synthons_bond_features': synthons_bond_features,
            'synthons_atom_features': synthons_atom_features,
            # group
            "group_smiles": group,
            'group_atom_num': expand_group_atom_num,
            'group_bond_features': expand_group_bond_features,
            'group_atom_features': expand_group_atom_features,
            # reaction atom and bond
            'product_to_group_bond': prod_to_group_bond
        }

        
        with open(os.path.join(save_dir, 'rxn_data_{}.pkl'.format(index)),
                  'wb') as f:
            pickle.dump(rxn_data, f)

    print(len0, len1, len2, len3, len4, len4more)
    print(exception)
    

if __name__ == '__main__':
    print('preprocessing dataset {}...'.format(args.dataset))
    assert args.dataset in ['USPTO50K', 'USPTO-full']

    datadir = 'data/{}/canonicalized_csv'.format(args.dataset)
    savedir = 'data/{}/'.format(args.dataset)

    # pkl format
    for data_set in ['test', 'valid', 'train']:
        save_dir = os.path.join(savedir, data_set)
        csv_path = os.path.join(datadir, data_set + '.csv')
        csv = pd.read_csv(csv_path)
        reaction_list = csv['rxn_smiles']
        reactant_smarts_list = list(
            map(lambda x: x.split('>>')[0], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split('>>')[1], reaction_list))
        reaction_class_list = list(map(lambda x: int(x) - 1, csv['class']))
        # Extract product adjacency matrix and atom features
        preprocess(
            save_dir,
            reactant_smarts_list,
            product_smarts_list,
            reaction_class_list,
        )


        
        
