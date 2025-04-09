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



class DataModule(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ex = pickle.load(open(self.data[index], 'rb'))
        #ex = self.data[index]
        smiles = ex["smiles"]
        atom_num = ex["atom_num"]
        adj = ex["adj"]
        atom_symbols = [atom_features["symbol"] for atom_features in ex["atom_features"]]
        bond_symbols = ex["bond_features"]

        return smiles, atom_num, atom_symbols, bond_symbols, adj


class DataExtraFeature:
    def __init__(self, noisy_data):
        self.noisy_X = noisy_data['X_t']
        self.noisy_E = noisy_data['E_t'].to(self.noisy_X.device)
        self.noisy_y = noisy_data['y_t'].to(self.noisy_X.device)
        self.t = noisy_data['t_int'].to(self.noisy_X.device)
        
    def to_extra_data(self, datainfos):
        X_weight = self.get_weight(self.noisy_X, datainfos)
        extra_y = torch.cat((self.t, X_weight), dim=1)
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
                smiles,
                atom_nums,
                atom_symbols,
                atom_masks,
                bond_symbols,
                adjs,
                y
                ):
        self.smiles = smiles
        self.atom_nums = atom_nums
        self.atom_symbols = atom_symbols
        self.atom_masks = atom_masks
        self.bond_symbols = bond_symbols
        self.adjs = adjs
        self.y = y

    def to_device(self, device):
        self.atom_nums = self.atom_nums.to(device)
        self.atom_symbols = self.atom_symbols.to(device)
        self.atom_masks = self.atom_masks.to(device)
        self.bond_symbols = self.bond_symbols.to(device)
        self.adjs = self.adjs.to(device)
        self.y = self.y.to(device)
        
        return self

    def __len__(self):
        return self.atom_nums.size(0)
              

class DataProcess:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        if self.dataset_name == "MOSES":
            self.symbol_list = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
            self.id2atom = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
            self.atom2id = {'C':0, 'N':1, 'S':2, 'O':3, 'F':4, 'Cl':5, 'Br':6, 'H':7}
            self.id2weights = [12.011, 14.007, 32.060, 15.999, 18.998, 35.450, 79.904, 1.008]
            self.bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC]
            self.max_weight = 250

        elif self.dataset_name == "QM9":
            self.symbol_list = ['C', 'N', 'O', 'F']
            self.id2atom = ['C', 'N', 'O', 'F']
            self.atom2id = {'C':0, 'N':1, 'O':3, 'F':4}
            self.id2weights = [12.011, 14.007, 15.999, 18.998]
            self.bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC]
            self.max_weight = 150


    def preprocess(self):
        data_dir = './pre-train/data/{}/canonicalized_csv'.format(self.dataset_name)

        if self.dataset_name == "MOSES":
            data_path = os.path.join(data_dir, self.dataset_name + '.csv')
            csv = pd.read_csv(data_path)
            smiles_list = csv["SMILES"]

        elif self.dataset_name == "QM9":
            smiles_list = []
            data_path_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
            for filename in data_path_list:
                with open(filename, 'r') as f:
                    x = f.read().split('\n')[-3]
                    smiles = x.split('\t')[0]
                    smiles_list.append(smiles)

        save_dir = './pre-train/data/{}/processed_pkl'.format(self.dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        datasize = len(smiles_list)

        for index in tqdm(range(datasize)):
            smiles = smiles_list[index]
            mol = Chem.MolFromSmiles(smiles)
            atom_num = len(mol.GetAtoms())

            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj + np.eye(adj.shape[0]).tolist()
       
            bond_features = get_bond_features(mol)
            atom_features = get_atom_features(mol, self.symbol_list)

            pretrained_data = {
                'smiles': smiles,
                'atom_num': atom_num,  # n_atom
                'adj': adj,  # (n_atom, n_atom)
                'bond_features': bond_features,  # (n_atom, n_atom, d_bond)
                'atom_features': atom_features,  # (n_atom, n_atom, d_atom)
            }
            with open(os.path.join(save_dir, 'data_{}.pkl'.format(index)),
                    'wb') as f:
                pickle.dump(pretrained_data, f)

            # if index >= 100:
            #     break
        
        return 


    def preload(self):
        data_dir = './pre-train/data/{}/processed_pkl'.format(self.dataset_name)
        data_path_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        #data_list = [pickle.load(open(file_name, 'rb')) for file_name in data_path_list]
        dataset = DataModule(data_path_list)
        return DataLoader(
                dataset=dataset,
                batch_size=self.args.train_batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=self.collator,
                drop_last=True 
                )

    
    def collator(self, datamodule):
        smiles, atom_nums, atom_symbols, bond_symbols, adjs = zip(*datamodule)
        
        smiles = [s for s in smiles]
        atom_nums = torch.stack([torch.tensor(num) for num in atom_nums])
        bs = atom_nums.size(0)

        atom_symbols, atom_masks = to_dense_atom_batch(bs, atom_nums, atom_symbols)
        bond_symbols = to_dense_bond_batch(bs, atom_nums, bond_symbols)
        adjs = to_dense_bond_batch(bs, atom_nums, adjs)
        y = torch.tensor([])

        return Batch(
            smiles=smiles,
            atom_nums=atom_nums,
            atom_symbols=atom_symbols,
            atom_masks=atom_masks,
            bond_symbols=bond_symbols,
            adjs=adjs,
            y=y
        )

    def compute_io_dims(self, dataloader):
        dataiter = iter(dataloader)
        batch = next(dataiter)
        
        bs = batch.atom_symbols.size(0)
        X = batch.atom_symbols
        E = batch.bond_symbols
        y = batch.y
        output_dims = {'X': X.size(-1), 'E': E.size(-1), 'y': y.size(-1)}
        
        data = {'X_t': X, 'E_t': E, 'y_t': y, 't_int': torch.zeros(bs, 1)}
        extra_data = DataExtraFeature(data).to_extra_data(DataProcess(self.args))
        input_dims = {'X': output_dims['X'] + extra_data['X'].size(-1), 
                      'E': output_dims['E'] + extra_data['E'].size(-1), 
                      'y': output_dims['y'] + extra_data['y'].size(-1)}
                      
        return input_dims, output_dims
        

class PreTraining:
    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader
        self.dataiter = iter(dataloader)
        self.input_dims, self.output_dims = DataProcess(args).compute_io_dims(self.dataloader)

        self.current_step = 0
        self.current_epoch = 0
        self.train_loss = 0

        self.model = GraphTransformer(n_layers=args.n_layers,
                            input_dims=self.input_dims,
                            hidden_mlp_dims=args.dims_mlp,
                            hidden_dims=args.dims_hidden,
                            output_dims=self.output_dims,
                            act_fn_in=nn.ReLU(),
                            act_fn_out=nn.ReLU())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        if args.from_ckpt is True:
            log.info(f"Resume from {args.resume_ckpt_path}...")
            ckpt = torch.load(args.ckpt_path)
            self.current_step = ckpt["step"]
            self.current_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optim"])

        self.model = self.model.to(device)
        self.device = device
        
        self.training_epoch = args.train_epoch
        self.TrainLoss = GraphLoss()
        self.MolVis = MolecularVisualization(remove_h="False", dataset_infos=DataProcess(self.args))

        self.best_validity = 0
        self.best_ckpt_filename = None


    def train(self):
        while self.current_epoch <= self.training_epoch:
            try:
                batch = next(self.dataiter)
            except StopIteration:
                print(f"Epoch {current_epoch} Average: "
                        f"Loss_X: {self.TrainLoss.epoch_loss['X'] / self.TrainLoss.current_epoch_step :.4f} -- "
                        f"Loss_E: {self.TrainLoss.epoch_loss['E'] / self.TrainLoss.current_epoch_step :.4f}")
                log.info(f"Epoch {current_epoch} Average: "
                        f"Loss_X: {self.TrainLoss.epoch_loss['X'] / self.TrainLoss.current_epoch_step :.4f} -- "
                        f"Loss_E: {self.TrainLoss.epoch_loss['E'] / self.TrainLoss.current_epoch_step :.4f}")
                self.current_epoch += 1
                self.TrainLoss.reset(step_start=False, epoch_start=True)
                self.dataiter = iter(self.dataloader)
                batch = next(self.dataiter)
            except Exception as e:
                print(f"Error when loading data: {e}")
                print(traceback.format_exc())
                exit()

            self.model.train()
            self.optimizer.zero_grad()
            batch = batch.to_device(self.device)
            self.TrainLoss.reset(step_start=True, epoch_start=False)

            # step 1: apply noise
            apply_noise_module = ForwardApplyNoise(args=self.args,
                                                X=batch.atom_symbols,
                                                E=batch.bond_symbols,
                                                y=batch.y,
                                                node_mask=batch.atom_masks,
                                                fix_diffusion_T=False)
            noisy_data = apply_noise_module.apply_noise()

            # step 2: training graph transformer
            extra_data = DataExtraFeature(noisy_data).to_extra_data(DataProcess(self.args))
            X = torch.cat((noisy_data['X_t'], extra_data['X']), dim=2).float()
            E = torch.cat((noisy_data['E_t'], extra_data['E']), dim=3).float()
            y = torch.hstack((noisy_data['y_t'], extra_data['y'])).float()
            
            assert (next(self.model.parameters()).device == X.device)
            pred_data = self.model(X, E, y, noisy_data['atom_mask'])
            true_data = {'X_0': batch.atom_symbols, 'E_0': batch.bond_symbols, 'y_0': batch.y}
            
            self.TrainLoss.compute_graph_loss(pred_data, true_data)
            X_loss = self.TrainLoss.step_loss['X']
            E_loss = self.TrainLoss.step_loss['E']
            loss = X_loss + E_loss * self.args.lambda_XE
            
            loss.backward()
            self.optimizer.step()


            if self.current_step % args.loss_interval == 0:
                print(f"Epoch {self.current_epoch} -- Step {self.current_step}: "
                    f"Loss_X: {X_loss :.4f} -- Loss_E: {E_loss :.4f}")
                log.info(f"Epoch {self.current_epoch} -- Step {self.current_step}: "
                    f"Loss_X: {X_loss :.4f} -- Loss_E: {E_loss :.4f}")
            
            if self.current_step % args.val_interval == 0:
                self.val(self.current_step, self.current_epoch)

            self.current_step += 1


           

    def val(self, current_step, current_epoch):
        sampling_module = BackwardInference(args=self.args,
                                            model=self.model,
                                            output_dims=self.output_dims,
                                            dataloader=self.dataloader,
                                            device=device,
                                            DatasetInfos=DataProcess(self.args))
        _, Pred_X, Pred_E, validity = sampling_module.batch_sampling(args, device)


        if self.args.save_ckpt:
            ckpt = {
                    "epoch": current_epoch,
                    "step": current_step,
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    }
            ckpt_filename = os.path.join("checkpoints", self.args.dataset_name, f"model_step_{current_step}.ckpt")
            dirname = os.path.dirname(os.path.abspath(ckpt_filename))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            torch.save(ckpt, ckpt_filename)
            
                
        if self.args.visualization:
            legend_list = ["pred", "true"]
            if not os.path.exists(os.path.join("visualization", self.args.dataset_name)):
                os.makedirs(os.path.join("visualization", self.args.dataset_name))
            out_filename = os.path.join("visualization", self.args.dataset_name, "ValDraw2D_" + str(current_step) + ".png")
            self.MolVis.batch_graph2mol(Pred_X, Pred_E, legend_list, out_filename)


        print(f"step {current_step} val validity: {validity}")
        log.info(f"step {current_step} val validity: {validity}")
                
        if validity >= self.best_validity:
            self.best_validity = validity
            self.best_ckpt_filename = ckpt_filename if self.args.save_ckpt else None

        print(self.best_validity)
        log.info(f"best_validity: {self.best_validity}")
        print(self.best_ckpt_filename)
        log.info(f"best_ckpt_filename: {self.best_ckpt_filename}")





if __name__ == '__main__':
    log = get_logger(__name__, current_time)
    args = arg_parses()
    args.do_preprocess = False
    args.save_ckpt = True
    log.info(f"{args}")

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
