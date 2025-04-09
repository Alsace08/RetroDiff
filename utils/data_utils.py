### https://github.com/uta-smile/RetroXpert/blob/canonical_product/preprocessing.py

import numpy as np
import pandas as pd
import argparse
import os
import re
import pickle
import traceback

import torch
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm



# Convert smarts to smiles by remove mapping numbers
def smarts2smiles(smarts, canonical=True):
    t = re.sub(':\d*', '', smarts)
    mol = Chem.MolFromSmiles(t, sanitize=False)
    return Chem.MolToSmiles(mol, canonical=canonical)


def del_index(smarts):
    t = re.sub(':\d*', '', smarts)
    return t


def onehot_encoding(x, total):
    return np.eye(total)[x]


def collate(data):
    return map(list, zip(*data))


# Get the mapping numbers in a SMARTS.
def get_idx(smarts_item):
    item = re.findall('(?<=:)\d+', smarts_item)
    item = list(map(int, item))
    return item


#  Get the dict maps each atom index to the mapping number.
def get_atomidx2mapidx(mol):
    atomidx2mapidx = {}
    for atom in mol.GetAtoms():
        atomidx2mapidx[atom.GetIdx()] = atom.GetAtomMapNum()
    return atomidx2mapidx


#  Get the dict maps each mapping number to the atom index .
def get_mapidx2atomidx(mol):
    mapidx2atomidx = {}
    mapidx2atomidx[0] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            mapidx2atomidx[0].append(atom.GetIdx())
        else:
            mapidx2atomidx[atom.GetAtomMapNum()] = atom.GetIdx()
    return mapidx2atomidx


# Get the reactant atom index list in the order of product atom index.
def get_order(product_mol, patomidx2pmapidx, rmapidx2ratomidx):
    order = []
    for atom in product_mol.GetAtoms():
        order.append(rmapidx2ratomidx[patomidx2pmapidx[atom.GetIdx()]])
    return order



def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)
    
def get_onehot(item, item_list):
    return list(map(lambda s: item == s, item_list))



### extract atom features

def get_atom_symbol_onehot(symbol, symbol_list):
    if symbol not in symbol_list:
        print(symbol)
        symbol = 'unk'
    
    atom_symbol = list(map(lambda s: symbol == s, symbol_list))
    return [int(item) for item in atom_symbol]


def get_single_atom_feature(atom, symbol_list):
    atom_feature = {'symbol': get_atom_symbol_onehot(atom.GetSymbol(), symbol_list),
                    # 'atom_mass': atom.GetMass(),
                    # 'is_aromatic': atom.GetIsAromatic(),
                    # 'degree': get_onehot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]),
                    # 'H_num': get_onehot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]),
                    # 'formal_charge': get_onehot(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]),
                    # 'chiral_tag': get_onehot(int(atom.GetChiralTag()), [0, 1, 2, 3]),
                    # 'hybridization': get_onehot(atom.GetHybridization(), [
                    #                             Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    #                             Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                    #                             Chem.rdchem.HybridizationType.SP3D2
                    #                         ])
                    }
    
    return atom_feature


def get_atom_features(mol, symbol_list):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(get_single_atom_feature(atom, symbol_list))
    return feats



### extract bond features

def get_bond_symbol_onehot(bt):
    #  1 empty bond + 4 bonds
    bond_symbol = [0, bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]
    
    if 1 not in bond_symbol:
        return [1,0,0,0,0]
    else:
        return bond_symbol



def get_single_bond_feature(bond):
    bt = bond.GetBondType()
    bond_feature = {'symbol': get_bond_symbol_onehot(bt),
                    # 'is_conjugated': bond.GetIsConjugated() if bt is not None else 0,
                    # 'is_inring': bond.IsInRing() if bt is not None else 0,
                    # 'stereo': get_onehot(int(bond.GetStereo()), list(range(6)))
                    }
                    
    return bond_feature
    
    
def get_bond_features(mol):
    atom_num = len(mol.GetAtoms())
    adj_array = np.zeros((atom_num, atom_num, 5), dtype=int)
    adj_array[:, :, 0] = 1
    
    for bond in mol.GetBonds():
        bond_feat = get_single_bond_feature(bond)
        adj_array[bond.GetBeginAtomIdx()][bond.GetEndAtomIdx()] = bond_feat['symbol']
        adj_array[bond.GetEndAtomIdx()][bond.GetBeginAtomIdx()] = bond_feat['symbol']

    return adj_array.tolist()
    


### split product into synthons
def get_synthons(product_smiles, reactant_smiles):
    p_smiles = product_smiles
    r_smiles = reactant_smiles
    p_mol = Chem.MolFromSmiles(p_smiles)
    r_mol = Chem.MolFromSmiles(r_smiles)
    # Chem.Kekulize(p_mol, clearAromaticFlags=True)
    # Chem.Kekulize(r_mol, clearAromaticFlags=True)

    p_atom_num = p_mol.GetNumAtoms()


    p_num2id = {}
    p_id2num = {}
    for atom in p_mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num:
            p_num2id[map_num] = atom.GetIdx()
            p_id2num[atom.GetIdx()] = map_num
            #print(atom.GetIdx(), atom.GetAtomMapNum())

    r_num2id = {}
    r_id2num = {}
    for atom in r_mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num:
            r_num2id[map_num] = atom.GetIdx()
            r_id2num[atom.GetIdx()] = map_num
            #print(atom.GetIdx(), atom.GetAtomMapNum())

    break_atom_pair = []
    break_atom_pair_num = []

    all_p_bonds = p_mol.GetBonds()
    len_p_bonds = len(all_p_bonds)
    for bond_ in all_p_bonds:
        type_ = bond_.GetBondType()

        begin_id = bond_.GetBeginAtomIdx()
        end_id = bond_.GetEndAtomIdx()
        begin_num = p_id2num[begin_id]
        end_num = p_id2num[end_id]
        
        r_bond = r_mol.GetBondBetweenAtoms(r_num2id[begin_num], r_num2id[end_num])
        if r_bond is None:
            break_atom_pair.append((begin_id, end_id))
            break_atom_pair_num.append((p_id2num[begin_id], p_id2num[end_id]))


    rw_mol_p = Chem.RWMol(p_mol)
    for break_pair in break_atom_pair:
        rw_mol_p.RemoveBond(break_pair[0], break_pair[1])
        
    all_res = r_smiles.split('.')
    #heuristic
    all_res.sort(key= lambda x:len(x),reverse=True)
    synthon_smiles = Chem.MolToSmiles(rw_mol_p)
    all_synthons = synthon_smiles.split('.')
    assert len(all_synthons) >= len(all_res)
    # for idx in range(len(all_res)):
    #     mol_synthons = [Chem.MolFromSmiles(sm) for sm in all_synthons]
    #     mol_synthons_first_map_num = [mol_.GetAtomWithIdx(0).GetAtomMapNum() for mol_ in mol_synthons]
    #     mol_res_map_num = [atom.GetAtomMapNum() for 
    #             atom in Chem.MolFromSmiles(all_res[idx]).GetAtoms()]
    #     #fps1 = [Chem.RDKFingerprint(x) for x in mol_synthons]
    #     #fps2= Chem.RDKFingerprint(Chem.MolFromSmiles(all_res[idx]))
    #     #sims = [DataStructs.FingerprintSimilarity(fps1[ss],fps2) for ss in range(len(fps1))]
    #     #max_idx = sims.index(max(sims))
    #     max_idx = -1
    #     for cur_idx in range(len(all_synthons)):
    #         if mol_synthons_first_map_num[cur_idx] in mol_res_map_num:
    #             max_idx = cur_idx
    #             break
    #     assert max_idx != -1

    return all_synthons, break_atom_pair


def get_external_group(reactant_smiles):
    mol_re = Chem.MolFromSmiles(reactant_smiles)
    # Chem.Kekulize(mol_re, clearAromaticFlags=True)
    mol_group = Chem.RWMol()

    group_idx = []
    re_to_group = {}
    for atom in mol_re.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num == 0:
            group_idx.append(atom.GetIdx())
            a = Chem.Atom(atom.GetSymbol())
            mol_group_idx = mol_group.AddAtom(a)
            re_to_group[atom.GetIdx()] = mol_group_idx

    for bond in mol_re.GetBonds():
        begin_id = bond.GetBeginAtomIdx()
        end_id = bond.GetEndAtomIdx()
        if (begin_id in group_idx) and (end_id in group_idx):
            mol_group.AddBond(re_to_group[begin_id], re_to_group[end_id], bond.GetBondType())


    try:
        mol_group = mol_group.GetMol()
    except Exception:
        print(traceback.format_exc())

    return Chem.MolToSmiles(mol_group)


def get_reaction_atom(product_smiles, reactant_smiles):
    p_smiles = product_smiles
    r_smiles = reactant_smiles
    p_mol = Chem.MolFromSmiles(p_smiles)
    r_mol = Chem.MolFromSmiles(r_smiles)
    # Chem.Kekulize(p_mol, clearAromaticFlags=True)
    # Chem.Kekulize(r_mol, clearAromaticFlags=True)

    p_atom_num = p_mol.GetNumAtoms()
    mol_group = Chem.RWMol()
    prod_to_group_bond = []

    re_to_prod = {}
    re_to_group = {}
    prod_idx = []
    group_idx = []
    for atom in r_mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num:
            re_to_prod[atom.GetIdx()] = map_num - 1
            prod_idx.append(atom.GetIdx())
        else:
            a = Chem.Atom(atom.GetSymbol())
            mol_group_idx = mol_group.AddAtom(a)
            re_to_group[atom.GetIdx()] = mol_group_idx
            group_idx.append(atom.GetIdx())

    flag = 1
    for bond in r_mol.GetBonds():
        begin_id = bond.GetBeginAtomIdx()
        end_id = bond.GetEndAtomIdx()
        if begin_id in prod_idx and end_id in group_idx:
            bond_type = [0, bond.GetBondType() == Chem.rdchem.BondType.SINGLE, bond.GetBondType()== Chem.rdchem.BondType.DOUBLE,
        bond.GetBondType() == Chem.rdchem.BondType.TRIPLE, bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
            prod_to_group_bond.append([re_to_prod[begin_id], re_to_group[end_id], bond_type])

        elif begin_id in group_idx and end_id in prod_idx:
            bond_type = [0, bond.GetBondType() == Chem.rdchem.BondType.SINGLE, bond.GetBondType()== Chem.rdchem.BondType.DOUBLE,
        bond.GetBondType() == Chem.rdchem.BondType.TRIPLE, bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
            prod_to_group_bond.append([re_to_prod[end_id], re_to_group[begin_id], bond_type])

        if begin_id in prod_idx and end_id in prod_idx:
            re_type = bond.GetBondType()
            try:
                prod_type = p_mol.GetBondBetweenAtoms(re_to_prod[begin_id], re_to_prod[end_id]).GetBondType()
            except Exception:
                prod_type = None
                print(traceback.format_exc())
            
            if re_type != prod_type:
                #print(begin_id, end_id, re_type, prod_type)
                flag = 0

    return prod_to_group_bond, flag
    


def advance_reaction_atom(group_bond_features, group_atom_features, prod_to_group_bond):
    g_bond, g_atom = group_bond_features, group_atom_features
    p2g_bond = prod_to_group_bond
    
    g_num = len(group_atom_features)
    g_map = np.arange(0, g_num)

    if len(prod_to_group_bond) >= 1:
        raw_id = prod_to_group_bond[0][1]

        tmp = g_atom[raw_id]
        g_atom[raw_id] = g_atom[0]
        g_atom[0] = tmp

        tmp = g_bond[raw_id][:]
        g_bond[raw_id][:] = g_bond[0][:]
        g_bond[0][:] = tmp

        tmp = g_bond[:][raw_id]
        g_bond[:][raw_id] = g_bond[:][0]
        g_bond[:][0] = tmp

        p2g_bond[0][1] = 0

        if len(prod_to_group_bond) == 2:
            if prod_to_group_bond[1][1] == prod_to_group_bond[0][1]:
                p2g_bond[1][1] = 0

            else:
                raw_id = prod_to_group_bond[1][1]

                tmp = g_atom[raw_id]
                g_atom[raw_id] = g_atom[1]
                g_atom[1] = tmp

                tmp = g_bond[raw_id][:]
                g_bond[raw_id][:] = g_bond[1][:]
                g_bond[1][:] = tmp

                tmp = g_bond[:][raw_id]
                g_bond[:][raw_id] = g_bond[:][1]
                g_bond[:][1] = tmp

                p2g_bond[1][1] = 1

    return g_bond, g_atom, p2g_bond





# s = get_external_group("O=C(c1ccc([N+](=O)[O-])cc1)[O:27][C@H:26]1[CH2:25][C@H:24]([n:23]2[c:6]3[n:5][cH:4][n:3][c:2]([NH2:1])[c:7]3[c:8](-[c:9]3[cH:10][cH:11][c:12]([O:13][c:14]4[cH:15][cH:16][cH:17][cH:18][cH:19]4)[cH:20][cH:21]3)[n:22]2)[CH2:28]1")
# print(s)
