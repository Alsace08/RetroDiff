import scipy.spatial
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import time
#import wandb
from rdkit import Chem

from .eva_utils import *

class GraphLoss:     
    def __init__(self,
                step_loss={'group_X':0, 'group_E':0, 'reaction_E':0},
                epoch_loss={'group_X':0, 'group_E':0, 'reaction_E':0}, 
                total_loss={'group_X':0, 'group_E':0, 'reaction_E':0},
                current_epoch_step=0,
                total_step=0):
        #super().__init__()
        self.step_loss = step_loss
        self.epoch_loss = epoch_loss
        self.total_loss = total_loss
        self.current_epoch_step = current_epoch_step
        self.total_step = total_step

    
    def compute_graph_loss_jointly(self, pred, target):
        loss_X_group = 0
        loss_E_group = 0
        pred_X_group, pred_E_group = pred['X_0'][:, :10, :], pred['E_0'][:, :10, :10, :]
        true_X_group, true_E_group = target['X_0'][:, :10, :], target['E_0'][:, :10, :10, :]
        true_X_group = torch.reshape(true_X_group, (-1, true_X_group.size(-1)))
        true_E_group = torch.reshape(true_E_group, (-1, true_E_group.size(-1)))
        pred_X_group = torch.reshape(pred_X_group, (-1, pred_X_group.size(-1)))
        pred_E_group = torch.reshape(pred_E_group, (-1, pred_E_group.size(-1)))
        loss_X_group = self.cross_entropy_for_XEy(pred_X_group, true_X_group) if true_X_group.numel() > 0 else 0.0
        loss_E_group = self.cross_entropy_for_XEy(pred_E_group, true_E_group) if true_E_group.numel() > 0 else 0.0

        loss_E_center = 0
        # pred_E_center = pred['E_0'][:, 0:2, 10:, :]
        # true_E_center = target['E_0'][:, 0:2, 10:, :]
        # pred_E_center = torch.reshape(pred_E_center, (-1, pred_E_center.size(-1)))
        # true_E_center = torch.reshape(true_E_center, (-1, true_E_center.size(-1)))

        # mask_E_center = (true_E_center != 0.).any(dim=-1)
        # flat_true_E_center = true_E_center[mask_E_center, :]
        # flat_pred_E_center = pred_E_center[mask_E_center, :]


        # loss_E_center = self.cross_entropy_for_XEy(flat_pred_E_center, flat_true_E_center) if flat_true_E_center.numel() > 0 else 0.0

        ###
        # loss_coefficient = loss_coefficient[:, 0:2, 10:, :]
        # loss_coefficient = torch.reshape(loss_coefficient, (-1, loss_coefficient.size(-1)))
        # weight_pred_E_center = pred_E_center * loss_coefficient
        # flat_weight_pred_E_center = pred_E_center[mask_E_center, :]
        # loss_E_center += self.cross_entropy_for_XEy(flat_weight_pred_E_center, flat_true_E_center) if flat_true_E_center.numel() > 0 else 0.0
        ###

        self.step_loss = {'group_X': loss_X_group, 'group_E': loss_E_group, 'reaction_E': loss_E_center}
        
        self.epoch_loss['group_X'] = self.epoch_loss['group_X'] + loss_X_group
        self.epoch_loss['group_E'] = self.epoch_loss['group_E'] + loss_E_group
        self.epoch_loss['reaction_E'] = self.epoch_loss['reaction_E'] + loss_E_center
        
        self.total_loss['group_X'] = self.total_loss['group_X'] + loss_X_group
        self.total_loss['group_E'] = self.total_loss['group_E'] + loss_E_group
        self.total_loss['reaction_E'] = self.total_loss['reaction_E'] + loss_E_center
        
        self.current_epoch_step += 1
        self.total_step += 1
        
        return self
        

    # def compute_graph_loss(self, pred, target):
    #     bs = pred['X_0'].size(0)
        
    #     pred_X, pred_E, pred_y = pred['X_0'], pred['E_0'], pred['y_0']
    #     true_X, true_E, true_y = target['X_0'], target['E_0'], target['y_0']

    #     true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
    #     true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
    #     pred_X = torch.reshape(pred_X, (-1, pred_X.size(-1)))  # (bs * n, dx)
    #     pred_E = torch.reshape(pred_E, (-1, pred_E.size(-1)))  # (bs * n * n, de)

    #     # Remove masked rows
    #     mask_X = (true_X != 0.).any(dim=-1)
    #     mask_E = (true_E != 0.).any(dim=-1)

    #     flat_true_X = true_X[mask_X, :]
    #     flat_pred_X = pred_X[mask_X, :]

    #     flat_true_E = true_E[mask_E, :]
    #     flat_pred_E = pred_E[mask_E, :]

    #     loss_X = self.cross_entropy_for_XEy(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
    #     loss_E = self.cross_entropy_for_XEy(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0

        
    #     self.step_loss = {'X': loss_X, 'E': loss_E}
        
    #     self.epoch_loss['X'] = self.epoch_loss['X'] + loss_X
    #     self.epoch_loss['E'] = self.epoch_loss['E'] + loss_E
        
    #     self.total_loss['X'] = self.total_loss['X'] + loss_X
    #     self.total_loss['E'] = self.total_loss['E'] + loss_E
        
    #     self.current_epoch_step += 1
    #     self.total_step += 1
        
    #     return self

    
        
        
    def cross_entropy_for_XEy(self, pred: Tensor, target: Tensor):
        """
        Compute CE loss for node or edge
        pred: Predictions from model    (bs * n, d) or (bs * n * n, d)
        target: Ground truth values     (bs * n, d) or (bs * n * n, d). 
        """
        
        target = torch.argmax(target, dim=-1)
        #loss = F.cross_entropy(pred, target, reduction='sum')
        loss = F.cross_entropy(pred, target, reduction='mean')
        
        return loss
        
        
    def reset(self, step_start: bool, epoch_start: bool):
        if step_start:
            self.step_loss = {'group_X':0, 'group_E':0, 'reaction_E':0}
        if epoch_start:
            self.epoch_loss = {'group_X':0, 'group_E':0, 'reaction_E':0}
            self.current_epoch_step = 0
        
        return self
        


class MolecularMetrics:
    def __init__(self, dataset_info):
        self.id2atom = dataset_info.id2atom
        self.bond_dict = dataset_info.bond_dict

    def compute_graph_validity(self, atom_types, edge_types):
        valid = False
        num_components = -1

        mol, smiles = graph2molecule(atom_types, edge_types, self.id2atom, self.bond_dict)
            
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            num_components = len(mol_frags)
        except:
            pass

        if smiles is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                #largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                #smiles = mol2smiles(largest_mol)
                valid = True
            except Chem.rdchem.AtomValenceException:
                print("Valence error in GetmolFrags")
                smiles = None
            except Chem.rdchem.KekulizeException:
                print("Can't kekulize molecule")
                smiles = None

        return valid, num_components, smiles


