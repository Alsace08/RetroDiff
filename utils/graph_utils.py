import os
import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from torch_geometric.utils import to_dense_batch, to_dense_adj
import numpy as np
    

class BatchGraphMask:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)
            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        
        return self



def to_dense_atom_batch(batch_size, atom_nums, atom_symbols):
    batch_id = np.arange(0, batch_size)
    batch_id = batch_id.repeat(atom_nums.numpy().tolist())
    batch_id = torch.tensor(batch_id)
    
    atom_symbols = torch.cat([torch.tensor(symbols) for symbols in atom_symbols])
    atom_symbols, atom_masks = to_dense_batch(x=atom_symbols, batch=batch_id)
    
    return atom_symbols, atom_masks



def to_dense_bond_batch(batch_size, atom_nums, bond_symbols):
    atom_nums_max = max(atom_nums)
    bond_index, bond_attr = [[], []], []
    batch_id = []

    for batch_idx in range(batch_size):
        atom_num = atom_nums[batch_idx]
        tmp_bond_symbols = bond_symbols[batch_idx]
        
        row = [[i + batch_idx * atom_nums_max] * atom_num for i in range(atom_num)]
        row = np.array(row).reshape(-1).tolist()
        col = [i + batch_idx * atom_nums_max for i in range(atom_num)] * atom_num
        tmp_bond_index = [row, col]
        tmp_bond_attr = np.array(tmp_bond_symbols).reshape(atom_num * atom_num, -1).tolist()
        
        bond_index = [bond_index[0] + tmp_bond_index[0], bond_index[1] + tmp_bond_index[1]]
        bond_attr = bond_attr + tmp_bond_attr
        batch_id += [batch_idx] * atom_nums_max
    
    bond_index, bond_attr, batch_id = torch.tensor(bond_index), torch.tensor(bond_attr), torch.tensor(batch_id)
    bond_symbols = to_dense_adj(edge_index=bond_index, batch=batch_id, edge_attr=bond_attr, max_num_nodes=atom_nums_max)
    
    return bond_symbols



def pack_reactant_with_product(r_atom_nums, r_atom_symbols, r_bond_symbols, 
                                p_atom_nums, p_atom_symbols, p_bond_symbols,
                                t_int, y):
    bs = len(r_atom_nums)
    rp_atom_nums = torch.tensor([r_atom_nums[i] + p_atom_nums[i] for i in range(bs)])

    rp_atom_symbols = []
    rp_bond_symbols = []

    for idx in range(bs):
        r_atom_num = r_atom_nums[idx]
        p_atom_num = p_atom_nums[idx]

        unzip_r_atom_symbols = r_atom_symbols[idx, :r_atom_nums[idx], :]
        unzip_p_atom_symbols = p_atom_symbols[idx, :p_atom_nums[idx], :]
        unzip_rp_atom_symbols = torch.cat((unzip_r_atom_symbols, unzip_p_atom_symbols), dim=0)
        rp_atom_symbols.append(unzip_rp_atom_symbols.cpu())

        unzip_r_bond_symbols = r_bond_symbols[idx, :r_atom_nums[idx], :r_atom_nums[idx], :]
        unzip_r_bond_symbols_int = torch.argmax(unzip_r_bond_symbols, dim=-1).cpu().numpy()
        unzip_p_bond_symbols = p_bond_symbols[idx, :p_atom_nums[idx], :p_atom_nums[idx], :]
        unzip_p_bond_symbols_int = torch.argmax(unzip_p_bond_symbols, dim=-1).cpu().numpy()

        top_matrix = np.column_stack((unzip_r_bond_symbols_int, np.zeros((r_atom_num, p_atom_num))))
        floor_matrix = np.column_stack((np.zeros((p_atom_num, r_atom_num)), unzip_p_bond_symbols_int))
        unzip_rp_bond_symbols_int = np.vstack((top_matrix, floor_matrix))

        index = torch.tensor(unzip_rp_bond_symbols_int, dtype=torch.int64).unsqueeze(-1)
        unzip_rp_bond_symbols = torch.zeros(unzip_rp_bond_symbols_int.shape[0], unzip_rp_bond_symbols_int.shape[1], 5).scatter_(2, index, 1)
        rp_bond_symbols.append(unzip_rp_bond_symbols.cpu())

    rp_atom_symbols, rp_atom_masks = to_dense_atom_batch(bs, rp_atom_nums, rp_atom_symbols)
    rp_bond_symbols = to_dense_bond_batch(bs, rp_atom_nums, rp_bond_symbols)

    Graph_rp = BatchGraphMask(X=rp_atom_symbols, E=rp_bond_symbols, y=y).type_as(y).mask(rp_atom_masks.type_as(y))
    noisy_data = {'t_int': t_int, 'X_t': Graph_rp.X, 'E_t': Graph_rp.E, 'y_t': Graph_rp.y, 'atom_mask': rp_atom_masks.type_as(y)}

    return noisy_data, rp_atom_nums



def unpack_reactant_with_product(rp_atom_symbols, rp_bond_symbols, y, rp_atom_nums, r_atom_nums):
    bs = len(rp_atom_nums)

    rp_atom_zero_symbols = torch.zeros_like(rp_atom_symbols)
    for i in range(len(rp_atom_nums)):
        rp_atom_symbols[i, r_atom_nums[i]: , :] = rp_atom_zero_symbols[i, r_atom_nums[i]: ,:]
 
    rp_bond_zero_symbols = torch.zeros_like(rp_bond_symbols)
    for i in range(len(rp_atom_nums)):
        rp_bond_symbols[i, r_atom_nums[i]: , :, :] = rp_bond_zero_symbols[i, r_atom_nums[i]: , :, :]
        rp_bond_symbols[i, :, r_atom_nums[i]: , :] = rp_bond_zero_symbols[i, :, r_atom_nums[i]: , :]

    max_num = torch.max(r_atom_nums)
    r_atom_symbols = rp_atom_symbols[:, :max_num, :]
    r_bond_symbols = rp_bond_symbols[:, :max_num, :max_num, :]

    Graph_r = BatchGraphMask(X=r_atom_symbols, E=r_bond_symbols, y=y).type_as(y)
    pred_data = {'X_0': Graph_r.X, 'E_0': Graph_r.E, 'y_0': Graph_r.y}

    return pred_data


def pack_group_with_product(g_atom_nums, g_atom_symbols, g_bond_symbols, 
                            p_atom_nums, p_atom_symbols, p_bond_symbols,
                            y, prod_to_group_bond, is_joint_external_bond):

    bs = len(g_atom_nums)
    gp_atom_nums = torch.tensor([g_atom_nums[i] + p_atom_nums[i] for i in range(bs)])

    gp_atom_symbols = []
    gp_bond_symbols = []

    for idx in range(bs):
        g_atom_num = g_atom_nums[idx]
        p_atom_num = p_atom_nums[idx]

        unzip_g_atom_symbols = g_atom_symbols[idx, :g_atom_nums[idx], :]
        unzip_p_atom_symbols = p_atom_symbols[idx, :p_atom_nums[idx], :]
        unzip_gp_atom_symbols = torch.cat((unzip_g_atom_symbols, unzip_p_atom_symbols), dim=0)
        gp_atom_symbols.append(unzip_gp_atom_symbols.cpu())

        unzip_g_bond_symbols = g_bond_symbols[idx, :g_atom_nums[idx], :g_atom_nums[idx], :]
        unzip_g_bond_symbols_int = torch.argmax(unzip_g_bond_symbols, dim=-1).cpu().numpy()
        unzip_p_bond_symbols = p_bond_symbols[idx, :p_atom_nums[idx], :p_atom_nums[idx], :]
        unzip_p_bond_symbols_int = torch.argmax(unzip_p_bond_symbols, dim=-1).cpu().numpy()

        top_matrix = np.column_stack((unzip_g_bond_symbols_int, np.zeros((g_atom_num, p_atom_num))))
        floor_matrix = np.column_stack((np.zeros((p_atom_num, g_atom_num)), unzip_p_bond_symbols_int))
        unzip_gp_bond_symbols_int = np.vstack((top_matrix, floor_matrix))

        index = torch.tensor(unzip_gp_bond_symbols_int, dtype=torch.int64).unsqueeze(-1)
        unzip_gp_bond_symbols = torch.zeros(unzip_gp_bond_symbols_int.shape[0], unzip_gp_bond_symbols_int.shape[1], 5).scatter_(2, index, 1)

        if is_joint_external_bond:
            break_bond_pair = prod_to_group_bond[idx]
            for break_bond in break_bond_pair:
                group_idx = break_bond[1]
                prod_idx = break_bond[0] + g_atom_num
                bond_type = break_bond[2]
                unzip_gp_bond_symbols[group_idx, prod_idx] = torch.tensor(bond_type)
                unzip_gp_bond_symbols[prod_idx, group_idx] = torch.tensor(bond_type)

        gp_bond_symbols.append(unzip_gp_bond_symbols.cpu())

    gp_atom_symbols, gp_atom_masks = to_dense_atom_batch(bs, gp_atom_nums, gp_atom_symbols)
    gp_bond_symbols = to_dense_bond_batch(bs, gp_atom_nums, gp_bond_symbols)

    Graph_gp = BatchGraphMask(X=gp_atom_symbols, E=gp_bond_symbols, y=y).type_as(y).mask(gp_atom_masks.type_as(y))

    return Graph_gp, gp_atom_masks.to(y.device), gp_atom_nums


def posterior_break_bond(X, E, num_gp):
    external_bond = []
    for row in range(10):
        for col in range(10, num_gp):
            if E[row, col] > 0:
                external_bond.append([col, row])
            
    if len(external_bond) == 2:
        E[external_bond[0][0], external_bond[1][0]] = 0
        E[external_bond[1][0], external_bond[0][0]] = 0

    # if len(external_bond) == 1:
        

    return X, E

