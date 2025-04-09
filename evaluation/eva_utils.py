import numpy as np
import scipy.spatial
import torch

from rdkit import Chem

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def graph2molecule(atom_types, edge_types, id2atom, bond_dict):
    mol = Chem.RWMol()

    remove_none_idx = []
    valid_num = 0
    for atom in atom_types:
        remove_none_idx.append(valid_num)
        if atom.item() == 0:
            continue

        a = Chem.Atom(id2atom[atom.item()])
        # print("Atom added: ", atom.item(), id2atom[atom.item()])
        mol.AddAtom(a)

        valid_num += 1
        
        
    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item() and atom_types[bond[0].item()] != 0 and atom_types[bond[1].item()] != 0:
            # print("bond added:", remove_none_idx[bond[0].item()], remove_none_idx[bond[1].item()], bond_dict[edge_types[bond[0], bond[1]].item()])
            mol.AddBond(remove_none_idx[bond[0].item()], remove_none_idx[bond[1].item()], bond_dict[edge_types[bond[0], bond[1]].item()])
            
    
    smiles = Chem.MolToSmiles(mol)
    
    return mol, smiles
    


