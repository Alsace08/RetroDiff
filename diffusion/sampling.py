import math
import tqdm
import numpy as np
import scipy.spatial
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from .apply_noise import *
from .de_noise import GraphTransformer

import sys
sys.path.append('../')
from evaluation.metrics import *
from data.data_prepare import RetroDiffExtraFeature, RetroDiffDataInfos
from utils.graph_utils import *


class sampling_Initialization:
    def __init__(self, output_dims, transition_mode, sample_num):
        self.output_dims_X = output_dims["X"]
        self.output_dims_E = output_dims["E"]
        self.output_dims_y = output_dims["y"]
        self.sample_num = sample_num

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if transition_mode == 'uniform':
            x_limit = torch.ones(self.output_dims_X ) / self.output_dims_X 
            e_limit = torch.ones(self.output_dims_E) / self.output_dims_E
            y_limit = torch.ones(self.output_dims_y) / self.output_dims_y
            self.limit_dist = BatchGraphMask(X=x_limit, E=e_limit, y=y_limit)

        elif transition_mode == 'marginal':
            x_limit = RetroDiffDataInfos().x_marginals
            e_limit = RetroDiffDataInfos().e_marginals
            y_limit = torch.ones(self.output_dims_y) / self.output_dims_y
            self.limit_dist = BatchGraphMask(X=x_limit, E=e_limit, y=y_limit)
        
        
    def get_init_sampling(self, batch_size):
        num_nodes = self.sample_num
        n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        bs, n_max = node_mask.shape
        x_limit = self.limit_dist.X[None, None, :].expand(bs, n_max, -1)
        e_limit = self.limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
        y_limit = self.limit_dist.y[None, :].expand(bs, -1)
        U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
        U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
        U_y = torch.empty((bs, 0))
    
        long_mask = node_mask.long()
        U_X = U_X.type_as(long_mask)
        U_E = U_E.type_as(long_mask)
        U_y = U_y.type_as(long_mask)
    
        U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
        U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()
    
        # Get upper triangular part of edge noise, without main diagonal
        upper_triangular_mask = torch.zeros_like(U_E)
        indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
        upper_triangular_mask[:, indices[0], indices[1], :] = 1
    
        U_E = U_E * upper_triangular_mask
        U_E = (U_E + torch.transpose(U_E, 1, 2))
    
        assert (U_E == torch.transpose(U_E, 1, 2)).all()
    
        return BatchGraphMask(X=U_X, E=U_E, y=U_y).mask(node_mask), node_mask
        
        
        

class BackwardInference:
    def __init__(self, args, model, output_dims, dataloader, device, DatasetInfos):
        self.args = args

        self.model = model.eval()
        self.device = device
        self.output_dims = output_dims

        self.dataloader = dataloader

        self.noise_schedule = args.noise_schedule
        self.T = args.total_diffusion_T
        self.transition_mode = args.transition_mode

        self.DatasetInfos = DatasetInfos
        
        
    def compute_loss(self, args):
        self.model.eval()
        TestLoss = GraphLoss()
        
        for batch_idx, batch in enumerate(self.dataloader):
            apply_noise_module = ForwardApplyNoise(X=batch.r_atom_symbols,
                                                   E=batch.r_bond_symbols,
                                                   y=torch.tensor([]),
                                                   node_mask=batch.r_atom_masks,
                                                   noise_schedule=args.noise_schedule,
                                                   total_diffusion_T=args.total_diffusion_T)
            noise_data = apply_noise_module.apply_noise()
            
            pred_data = self.model(noise_data['X_t'], noise_data['E_t'], noise_data['t_int'], noise_data['atom_mask'])
            true_data = {'X_0': batch.r_atom_symbols, 'E_0': batch.r_bond_symbols, 'y_0': noise_data['t_int']}
            TestLoss.compute_graph_loss(pred_data, true_data)
            
        return TestLoss

    









    def test_sampling(self, args, device, log, model):
        total_num = 0
        molmetrics = MolecularMetrics(self.DatasetInfos)
        acc_group = 0
        acc_center = 0
        acc_final = 0
        for batch_idx, batch in enumerate(self.dataloader):
            batch_size = len(batch)
            batch = batch.to_device(device)
        
            UniformInit = sampling_Initialization(self.output_dims, args.init_mode, self.args.sample_atom_num)
            init_graph, node_mask = UniformInit.get_init_sampling(batch_size)
            X, E, y = init_graph.X, init_graph.E, init_graph.y
            X, E, y, node_mask = X.to(batch.p_atom_symbols.device), E.to(batch.p_atom_symbols.device), y.to(batch.p_atom_symbols.device), node_mask.to(batch.p_atom_symbols.device)
        

            True_X = batch.g_atom_symbols
            True_E = batch.g_bond_symbols
            True_y = batch.g_y
            True_node_mask = batch.g_atom_masks
            True_Final_Graph_G = BatchGraphMask(True_X, True_E, True_y).type_as(True_X).mask(node_mask=True_node_mask, collapse=True)
            True_X_G, True_E_G, True_y_G = True_Final_Graph_G.X, True_Final_Graph_G.E, True_Final_Graph_G.y


            True_X = batch.r_atom_symbols
            True_E = batch.r_bond_symbols
            True_y = batch.r_y
            True_node_mask = batch.r_atom_masks
            True_Final_Graph_R = BatchGraphMask(True_X, True_E, True_y).type_as(True_X).mask(node_mask=True_node_mask, collapse=True)
            True_X_R, True_E_R, True_y_R = True_Final_Graph_R.X, True_Final_Graph_R.E, True_Final_Graph_R.y



            Graph_gp, gp_atom_masks, gp_atom_nums = pack_group_with_product(batch.g_atom_nums, 
                                                                                X, 
                                                                                E, 
                                                                                batch.p_atom_nums, 
                                                                                batch.p_atom_symbols, 
                                                                                batch.p_bond_symbols, 
                                                                                batch.g_y,
                                                                                batch.product_to_group_bond,
                                                                                is_joint_external_bond=False)
            X, E, y = Graph_gp.X, Graph_gp.E, Graph_gp.y
            for s_int in reversed(range(0, 500)):
                s_batch = s_int * torch.ones((batch_size, 1)).type(torch.LongTensor).to(X.device)
                t_batch = (s_batch + 1).to(X.device)

                Graph_s, _ = self.p_zs_given_zt(s_batch, t_batch, X, E, y, gp_atom_masks, self.model)
                Graph_s['X'][:, 10:, :] = Graph_gp.X[:, 10:, :]
                Graph_s['E'][:, 10:, :, :] = Graph_gp.E[:, 10:, :, :]
                Graph_s['E'][:, :, 10:, :] = Graph_gp.E[:, :, 10:, :]
                X, E, y = Graph_s['X'], Graph_s['E'], Graph_s['y']
            Group_Graph = BatchGraphMask(X, E, y).type_as(X).mask(node_mask=gp_atom_masks, collapse=True)
            Group_X, Group_E, Pred_y = Group_Graph.X, Group_Graph.E, Group_Graph.y
                



            
            Graph_gp, gp_atom_masks, gp_atom_nums = pack_group_with_product(batch.g_atom_nums, 
                                                                                batch.g_atom_symbols, 
                                                                                batch.g_bond_symbols,
                                                                                batch.p_atom_nums, 
                                                                                batch.p_atom_symbols, 
                                                                                batch.p_bond_symbols, 
                                                                                batch.g_y,
                                                                                batch.product_to_group_bond,
                                                                                is_joint_external_bond=False)
            X, E, y = Graph_gp.X, Graph_gp.E, Graph_gp.y
            for s_int in reversed(range(0, 50)):
                s_batch = s_int * torch.ones((batch_size, 1)).type(torch.LongTensor).to(X.device)
                t_batch = (s_batch + 1).to(X.device)

                Graph_s, _ = self.p_zs_given_zt(s_batch, t_batch, X, E, y, gp_atom_masks, model)
                Graph_s['X'][:, :, :] = Graph_gp.X
                Graph_s['E'][:, 10:, 10:, :] = Graph_gp.E[:, 10:, 10:, :]
                Graph_s['E'][:, :10, :10, :] = Graph_gp.E[:, :10, :10, :]
                Graph_s['E'][:, 2:10, :, :] = Graph_gp.E[:, 2:10, :, :]
                Graph_s['E'][:, :, 2:10, :] = Graph_gp.E[:, :, 2:10, :]
                X, E, y = Graph_s['X'], Graph_s['E'], Graph_s['y']
            Bond_Graph = BatchGraphMask(X, E, y).type_as(X).mask(node_mask=gp_atom_masks, collapse=True)
            Bond_X, Bond_E, Pred_y = Bond_Graph.X, Bond_Graph.E, Bond_Graph.y




            for i in range(batch_size):
                num_gp = gp_atom_nums[i]
                num_r = batch.r_atom_nums[i]

                tmp_true_bond = []
                for bond in batch.product_to_group_bond[i]:
                    bond_type = None
                    if bond[2][1]: bond_type = Chem.rdchem.BondType.SINGLE
                    if bond[2][2]: bond_type = Chem.rdchem.BondType.DOUBLE
                    if bond[2][3]: bond_type = Chem.rdchem.BondType.TRIPLE
                    if bond[2][4]: bond_type = Chem.rdchem.BondType.AROMATIC
                    tmp_true_bond.append([bond[0], bond[1], bond_type])

                tmp_pred_bond = []
                for row in range(10):
                    for col in range(10, num_gp):
                        if Bond_E[i,row,col] > 0:
                            bond_type = None
                            if Bond_E[i,row,col] == 1: bond_type = Chem.rdchem.BondType.SINGLE
                            if Bond_E[i,row,col] == 2: bond_type = Chem.rdchem.BondType.DOUBLE
                            if Bond_E[i,row,col] == 3: bond_type = Chem.rdchem.BondType.TRIPLE
                            if Bond_E[i,row,col] == 4: bond_type = Chem.rdchem.BondType.AROMATIC
                            tmp_pred_bond.append([col - 10, row, bond_type])

                
                
                _, _, pred_g_smiles = molmetrics.compute_graph_validity(Group_X[i, :10], Group_E[i, :10, :10])
                _, _, true_p_smiles = molmetrics.compute_graph_validity(Group_X[i, 10:num_gp], Group_E[i, 10:num_gp, 10:num_gp])
                _, _, true_g_smiles = molmetrics.compute_graph_validity(True_X_G[i], True_E_G[i])
                post_Pred_X, post_Pred_E = posterior_break_bond(Bond_X[i], Bond_E[i], num_gp)    
                _, _, pred_r_smiles = molmetrics.compute_graph_validity(post_Pred_X[:num_gp], post_Pred_E[:num_gp, :num_gp])
                _, _, true_r_smiles = molmetrics.compute_graph_validity(True_X_R[i, :num_r], True_E_R[i, :num_r, :num_r])
                
                if pred_g_smiles == true_g_smiles:
                    acc_group += 1

                if tmp_true_bond == tmp_pred_bond:
                    acc_center += 1

                if pred_g_smiles == true_g_smiles:
                    if pred_r_smiles == true_r_smiles:
                        acc_final += 1

                total_num += 1
                print(f"total num: {total_num}")
                log.info(f"total num: {total_num}")
                print(f"product: {batch.p_smiles[i]}")
                log.info(f"product: {batch.p_smiles[i]}")
                print(f"reactant: {batch.r_smiles[i]}")
                log.info(f"reactant: {batch.r_smiles[i]}")
                print(f"X.symbol: {Group_X[i, :10]}")
                log.info(f"X.symbol: {Group_X[i, :10]}")
                print(f"true_X.symbol: {Bond_X[i, :10]}")
                log.info(f"true_X.symbol: {Bond_X[i, :10]}")
                print(f"pred group: {pred_g_smiles} -- true group: {true_g_smiles}")
                log.info(f"pred group: {pred_g_smiles} -- true group: {true_g_smiles}")
                print(f"pred center: {tmp_pred_bond} -- true center: {tmp_true_bond}")
                log.info(f"pred center: {tmp_pred_bond} -- true center: {tmp_true_bond}")
                print(f"pred reactant: {pred_r_smiles} -- true reactant: {true_r_smiles}")
                log.info(f"pred reactant: {pred_r_smiles} -- true reactant: {true_r_smiles}")
                print(f"group acc: {acc_group / total_num}")
                log.info(f"group acc: {acc_group / total_num}")
                print(f"center acc: {acc_center / total_num}")
                log.info(f"center acc: {acc_center / total_num}")
                print(f"final acc: {acc_final / total_num}")
                log.info(f"final acc: {acc_final / total_num}")
                print("---")
                log.info("---")


        return 








    
    def batch_sampling(self, args, device, log, model):
        molmetrics = MolecularMetrics(self.DatasetInfos)
        args.init_mode = args.transition_mode

        val_group = 0
        val_reactant = 0
        acc_group = 0
        acc_reactant = 0
        acc_center = 0
        acc_final = 0

        product_smiles_list = []
        reactant_smiles_list = []
        pred_g_smiles_list = []
        true_g_smiles_list = []
        pred_r_smiles_list = []
        true_r_smiles_list = []
        pred_center = []
        true_center = []
        X_list = []
        E_list = []
        total_num = 0
        for batch_idx, batch in enumerate(self.dataloader):
            batch_size = len(batch)
            batch = batch.to_device(device)
            
            if args.init_mode == "product":
                X = batch.r_atom_symbols
                E = batch.r_bond_symbols
                y = batch.r_y
                node_mask = batch.r_atom_masks
            
            elif args.init_mode == "reactant":
                X = batch.p_atom_symbols
                E = batch.p_bond_symbols
                y = batch.p_y
                node_mask = batch.p_atom_masks
                
                apply_noise_module = ForwardApplyNoise(args=args, X=X,E=E,y=y,
                                                node_mask=node_mask,
                                                fix_diffusion_T=True)
                noisy_data = apply_noise_module.apply_noise()
                X, E, y = noisy_data['X_t'], noisy_data['E_t'], noisy_data['y_t']
            
            elif args.init_mode == "uniform" or "marginal":
                UniformInit = sampling_Initialization(self.output_dims, args.init_mode, self.args.sample_atom_num)
                init_graph, node_mask = UniformInit.get_init_sampling(batch_size)
                X, E, y = init_graph.X, init_graph.E, init_graph.y
                if args.is_pretrained is True:
                    X, E, y, node_mask = X.to(batch.atom_symbols.device), E.to(batch.atom_symbols.device), y.to(batch.atom_symbols.device), node_mask.to(batch.atom_symbols.device)
                else:
                    X, E, y, node_mask = X.to(batch.p_atom_symbols.device), E.to(batch.p_atom_symbols.device), y.to(batch.p_atom_symbols.device), node_mask.to(batch.p_atom_symbols.device)
        
            
            if args.to_group_given_product:
                Graph_gp, gp_atom_masks, gp_atom_nums = pack_group_with_product(batch.g_atom_nums, 
                                                                                X, 
                                                                                E, 
                                                                                batch.p_atom_nums, 
                                                                                batch.p_atom_symbols, 
                                                                                batch.p_bond_symbols, 
                                                                                batch.g_y,
                                                                                batch.product_to_group_bond,
                                                                                is_joint_external_bond=False)

                # true_Graph_gp, True_node_mask, _ = pack_group_with_product(batch.g_atom_nums, 
                #                                             batch.g_atom_symbols, 
                #                                             batch.g_bond_symbols, 
                #                                             batch.p_atom_nums, 
                #                                             batch.p_atom_symbols, 
                #                                             batch.p_bond_symbols, 
                #                                             batch.g_y,
                #                                             batch.product_to_group_bond,
                #                                             is_joint_external_bond=False)
                # true_X_no_collapse,  true_E_no_collapse,  true_y_no_collapse = true_Graph_gp.X, true_Graph_gp.E, true_Graph_gp.y
                # True_Final_Graph_G = BatchGraphMask(true_Graph_gp.X, true_Graph_gp.E, true_Graph_gp.y).type_as(true_Graph_gp.X).mask(node_mask=True_node_mask, collapse=True)
                # true_X_collapse, true_E_collapse, true_y_collapse = True_Final_Graph_G.X, True_Final_Graph_G.E, True_Final_Graph_G.y
                # true_no_collapse = {'X_0': true_X_no_collapse, 'E_0': true_E_no_collapse, 'y_0': true_y_no_collapse}
                # true_collapse = {'X_0': true_X_collapse, 'E_0': true_E_collapse, 'y_0': true_y_collapse}
                # print(f"X.symbol: {true_X_collapse[0, :10]}")
                # log.info(f"X.symbol: {true_X_collapse[0, :10]}")
                # # print(f"E.symbol: {true_E_collapse[0, :10, :10]}")
                # # log.info(f"E.symbol: {true_E_collapse[0, :10, :10]}")
                # print(f"X.symbol: {true_X_collapse[3, :10]}")
                # log.info(f"X.symbol: {true_X_collapse[3, :10]}")
                # print(f"X.symbol: {true_X_collapse[42, :10]}")
                # log.info(f"X.symbol: {true_X_collapse[42, :10]}")
                # # print(f"E.symbol: {true_E_collapse[3, :10, :10]}")
                # # log.info(f"E.symbol: {true_E_collapse[3, :10, :10]}")
                
                
                
                # TrainLoss = GraphLoss()
                X, E, y = Graph_gp.X, Graph_gp.E, Graph_gp.y
                for s_int in reversed(range(0, self.T)):
                    s_batch = s_int * torch.ones((batch_size, 1)).type(torch.LongTensor).to(X.device)
                    t_batch = (s_batch + 1).to(X.device)

                    Graph_s, pred_data = self.p_zs_given_zt(s_batch, t_batch, X, E, y, gp_atom_masks, self.model)
                    Graph_s['X'][:, 10:, :] = Graph_gp.X[:, 10:, :]
                    Graph_s['E'][:, 10:, :, :] = Graph_gp.E[:, 10:, :, :]
                    Graph_s['E'][:, :, 10:, :] = Graph_gp.E[:, :, 10:, :]
                    X, E, y = Graph_s['X'], Graph_s['E'], Graph_s['y']



                    # pred_X_no_collapse, pred_E_no_collapse, pred_y_no_collapse = pred_data['X_0'], pred_data['E_0'], pred_data['y_0']
                    # Pred_Final_Graph_G = BatchGraphMask(Graph_s['X'], Graph_s['E'], Graph_s['y']).type_as(Graph_s['X']).mask(node_mask=True_node_mask, collapse=True)
                    # pred_X_collapse, pred_E_collapse, pred_y_collapse = Pred_Final_Graph_G.X, Pred_Final_Graph_G.E, Pred_Final_Graph_G.y
                    # pred_no_collapse = {'X_0': pred_X_no_collapse, 'E_0': pred_E_no_collapse, 'y_0': pred_y_no_collapse}
                    # pred_collapse = {'X_0': pred_X_collapse, 'E_0': pred_E_collapse, 'y_0': pred_y_collapse}
                    
                    # with torch.no_grad():
                    #     TrainLoss.compute_graph_loss_jointly(pred_no_collapse, true_no_collapse)
                    #     group_X_loss = TrainLoss.step_loss['group_X']
                    #     group_E_loss = TrainLoss.step_loss['group_E']
                    #     reaction_E_loss = TrainLoss.step_loss['reaction_E']
                    #     print(f"Time {s_int}: loss_X: {group_X_loss} -- loss_E: {group_E_loss}")
                    #     log.info(f"Time {s_int}: loss_X: {group_X_loss} -- loss_E: {group_E_loss}")
                    #     TrainLoss.reset(step_start=True, epoch_start=False)

                    
                    # print(f"X.symbol: {pred_X_collapse[0, :10]}")
                    # log.info(f"X.symbol: {pred_X_collapse[0, :10]}")
                    # # print(f"E.symbol: {pred_E_collapse[0, :10, :10]}")
                    # # log.info(f"E.symbol: {pred_E_collapse[0, :10, :10]}")
                    # print(f"X.symbol: {pred_X_collapse[3, :10]}")
                    # log.info(f"X.symbol: {pred_X_collapse[3, :10]}")
                    # print(f"X.symbol: {pred_X_collapse[42, :10]}")
                    # log.info(f"X.symbol: {pred_X_collapse[42, :10]}")
                    # # print(f"E.symbol: {pred_E_collapse[3, :10, :10]}")
                    # # log.info(f"E.symbol: {pred_E_collapse[3, :10, :10]}")
                    # print("---")
                    # log.info("---")



            if args.to_exbond_given_product_and_group:
                Graph_gp, gp_atom_masks, gp_atom_nums = pack_group_with_product(batch.g_atom_nums, 
                                                                                batch.g_atom_symbols, 
                                                                                batch.g_bond_symbols,
                                                                                batch.p_atom_nums, 
                                                                                batch.p_atom_symbols, 
                                                                                batch.p_bond_symbols, 
                                                                                batch.g_y,
                                                                                batch.product_to_group_bond,
                                                                                is_joint_external_bond=False)
            
                X, E, y = Graph_gp.X, Graph_gp.E, Graph_gp.y
                for s_int in reversed(range(0, 50)):
                    s_batch = s_int * torch.ones((batch_size, 1)).type(torch.LongTensor).to(X.device)
                    t_batch = (s_batch + 1).to(X.device)

                    Graph_s = self.p_zs_given_zt(s_batch, t_batch, X, E, y, gp_atom_masks, self.model)
                    Graph_s['X'][:, :, :] = Graph_gp.X
                    Graph_s['E'][:, 10:, 10:, :] = Graph_gp.E[:, 10:, 10:, :]
                    Graph_s['E'][:, :10, :10, :] = Graph_gp.E[:, :10, :10, :]
                    Graph_s['E'][:, 2:10, :, :] = Graph_gp.E[:, 2:10, :, :]
                    Graph_s['E'][:, :, 2:10, :] = Graph_gp.E[:, :, 2:10, :]
                    X, E, y = Graph_s['X'], Graph_s['E'], Graph_s['y']

            
            if args.jointly:
                Graph_gp, gp_atom_masks, gp_atom_nums = pack_group_with_product(batch.g_atom_nums, 
                                                                                X, 
                                                                                E, 
                                                                                batch.p_atom_nums, 
                                                                                batch.p_atom_symbols, 
                                                                                batch.p_bond_symbols, 
                                                                                batch.g_y,
                                                                                batch.product_to_group_bond,
                                                                                is_joint_external_bond=False)
            
                X, E, y = Graph_gp.X, Graph_gp.E, Graph_gp.y
                for s_int in reversed(range(0, self.T)):
                    #print(BatchGraphMask(X, E, y).type_as(X).mask(node_mask=gp_atom_masks, collapse=True).X[0])
                    s_batch = s_int * torch.ones((batch_size, 1)).type(torch.LongTensor).to(X.device)
                    t_batch = (s_batch + 1).to(X.device)

                    Graph_s = self.p_zs_given_zt(s_batch, t_batch, X, E, y, gp_atom_masks, self.model)
                    Graph_s['X'][:, 10:, :] = Graph_gp.X[:, 10:, :]
                    Graph_s['E'][:, 10:, :, :] = Graph_gp.E[:, 10:, :, :]
                    Graph_s['E'][:, :, 10:, :] = Graph_gp.E[:, :, 10:, :]
                    X, E, y = Graph_s['X'], Graph_s['E'], Graph_s['y']
                
                Graph_gp.X, Graph_gp.E, Graph_gp.y = X, E, y

                for s_int in reversed(range(0, 200)):
                    s_batch = s_int * torch.ones((batch_size, 1)).type(torch.LongTensor).to(X.device)
                    t_batch = (s_batch + 1).to(X.device)

                    Graph_s = self.p_zs_given_zt(s_batch, t_batch, X, E, y, gp_atom_masks, model.eval())
                    Graph_s['X'][:, :, :] = Graph_gp.X
                    Graph_s['E'][:, 10:, 10:, :] = Graph_gp.E[:, 10:, 10:, :]
                    Graph_s['E'][:, :10, :10, :] = Graph_gp.E[:, :10, :10, :]
                    Graph_s['E'][:, 2:10, :, :] = Graph_gp.E[:, 2:10, :, :]
                    Graph_s['E'][:, :, 2:10, :] = Graph_gp.E[:, :, 2:10, :]
                    X, E, y = Graph_s['X'], Graph_s['E'], Graph_s['y']
            


            
            Final_Graph = BatchGraphMask(X, E, y).type_as(X).mask(node_mask=gp_atom_masks, collapse=True)
            Pred_X, Pred_E, Pred_y = Final_Graph.X, Final_Graph.E, Final_Graph.y
            
            True_X = batch.r_atom_symbols
            True_E = batch.r_bond_symbols
            True_y = batch.r_y
            True_node_mask = batch.r_atom_masks
            
            True_Final_Graph_R = BatchGraphMask(True_X, True_E, True_y).type_as(True_X).mask(node_mask=True_node_mask, collapse=True)
            True_X_R, True_E_R, True_y_R = True_Final_Graph_R.X, True_Final_Graph_R.E, True_Final_Graph_R.y

            True_X = batch.g_atom_symbols
            True_E = batch.g_bond_symbols
            True_y = batch.g_y
            True_node_mask = batch.g_atom_masks
            
            True_Final_Graph_G = BatchGraphMask(True_X, True_E, True_y).type_as(True_X).mask(node_mask=True_node_mask, collapse=True)
            True_X_G, True_E_G, True_y_G = True_Final_Graph_G.X, True_Final_Graph_G.E, True_Final_Graph_G.y
            
            for i in range(batch_size):
                num_gp = gp_atom_nums[i]
                num_r = batch.r_atom_nums[i]

                tmp_true_bond = []
                for bond in batch.product_to_group_bond[i]:
                    bond_type = None
                    if bond[2][1]: bond_type = Chem.rdchem.BondType.SINGLE
                    if bond[2][2]: bond_type = Chem.rdchem.BondType.DOUBLE
                    if bond[2][3]: bond_type = Chem.rdchem.BondType.TRIPLE
                    if bond[2][4]: bond_type = Chem.rdchem.BondType.AROMATIC
                    tmp_true_bond.append([bond[0], bond[1], bond_type])
                true_center.append(tmp_true_bond)

                tmp_pred_bond = []
                for row in range(10):
                    for col in range(10, num_gp):
                        if Pred_E[i,row,col] > 0:
                            bond_type = None
                            if Pred_E[i,row,col] == 1: bond_type = Chem.rdchem.BondType.SINGLE
                            if Pred_E[i,row,col] == 2: bond_type = Chem.rdchem.BondType.DOUBLE
                            if Pred_E[i,row,col] == 3: bond_type = Chem.rdchem.BondType.TRIPLE
                            if Pred_E[i,row,col] == 4: bond_type = Chem.rdchem.BondType.AROMATIC
                            tmp_pred_bond.append([col - 10, row, bond_type])
                pred_center.append(tmp_pred_bond)

                valid_group, pred_num_components, pred_g_smiles = molmetrics.compute_graph_validity(Pred_X[i, :10], Pred_E[i, :10, :10])
                _, _, true_g_smiles = molmetrics.compute_graph_validity(True_X_G[i], True_E_G[i])
                

                post_Pred_X, post_Pred_E = posterior_break_bond(Pred_X[i], Pred_E[i], num_gp)    
                valid_reactant, pred_num_components, pred_r_smiles = molmetrics.compute_graph_validity(post_Pred_X[:num_gp], post_Pred_E[:num_gp, :num_gp])
                valid_true, _, true_r_smiles = molmetrics.compute_graph_validity(True_X_R[i, :num_r], True_E_R[i, :num_r, :num_r])

                product_smiles_list.append(batch.p_smiles[i])
                reactant_smiles_list.append(batch.r_smiles[i])
                pred_g_smiles_list.append(pred_g_smiles)
                true_g_smiles_list.append(true_g_smiles)
                pred_r_smiles_list.append(pred_r_smiles)
                true_r_smiles_list.append(true_r_smiles)
                
                if valid_group:
                    val_group += 1
                if valid_reactant:
                    val_reactant += 1
                if pred_g_smiles == true_g_smiles:
                    acc_group += 1
                    if tmp_true_bond == tmp_pred_bond:
                        acc_reactant += 1
                if tmp_true_bond == tmp_pred_bond:
                    acc_center += 1

                if pred_r_smiles == true_r_smiles:
                    acc_final += 1

                X_list.append(Pred_X[i, :10].cpu())
                X_list.append(True_X_G[i].cpu())
                X_list.append(Pred_X[i, :num_gp].cpu())
                X_list.append(True_X_R[i, :num_r].cpu())
                E_list.append(Pred_E[i, :10, :10].cpu())
                E_list.append(True_E_G[i].cpu())
                E_list.append(Pred_E[i, :num_gp, :num_gp].cpu())
                E_list.append(True_E_R[i, :num_r, :num_r].cpu())


                total_num += 1
                print(f"epoch {batch_idx} -- total num: {total_num}")
                log.info(f"epoch {batch_idx} -- total num: {total_num}")
                print(f"X.symbol: {Pred_X[i, :10]}")
                log.info(f"X.symbol: {Pred_X[i, :10]}")
                print(f"pred group: {pred_g_smiles} -- true group: {true_g_smiles}")
                log.info(f"pred group: {pred_g_smiles} -- true group: {true_g_smiles}")
                print(f"pred center: {tmp_pred_bond} -- true center: {tmp_true_bond}")
                log.info(f"pred center: {tmp_pred_bond} -- true center: {tmp_true_bond}")
                print(f"pred reactant: {pred_r_smiles} -- true reactant: {true_r_smiles}")
                log.info(f"pred reactant: {pred_r_smiles} -- true reactant: {true_r_smiles}")
                print(f"group acc: {acc_group / total_num}")
                log.info(f"group acc: {acc_group / total_num}")
                print(f"center acc: {acc_center / total_num}")
                log.info(f"center acc: {acc_center / total_num}")
                print(f"final acc: {acc_final / total_num}")
                log.info(f"final acc: {acc_final / total_num}")
                print("---")
                log.info("---")


        
        return X_list, E_list, \
                product_smiles_list, reactant_smiles_list, pred_g_smiles_list, true_g_smiles_list, pred_r_smiles_list, true_r_smiles_list, \
                acc_center / ((batch_idx + 1) * batch_size), acc_group / ((batch_idx + 1) * batch_size), acc_reactant / ((batch_idx + 1) * batch_size), acc_final / ((batch_idx + 1) * batch_size), \
                val_group / ((batch_idx + 1) * batch_size), val_reactant / ((batch_idx + 1) * batch_size), \
                pred_center, true_center
            


        
    def p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, model):
        """
        Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well
        s = t - 1
        """
        raw_X_t, raw_E_t, raw_y_t = X_t, E_t, y_t
        bs, n, d_atom = X_t.shape
        
        predefine_alpha = PredefineAlpha(noise_schedule=self.noise_schedule, total_diffusion_T=self.T)
        predefine_transitional_matrix = TransitionMode(X_t.size(-1), E_t.size(-1), y_t.size(-1), self.args.transition_mode).MODE
        
        alpha_s_bar = predefine_alpha.get_alpha_t_bar(s)
        alpha_t_bar = predefine_alpha.get_alpha_t_bar(t)
        alpha_t = predefine_alpha.get_alpha_t(t)
        
        Q_s_bar = predefine_transitional_matrix.get_Q_t_bar(alpha_s_bar, X_t.device)
        Q_t_bar = predefine_transitional_matrix.get_Q_t_bar(alpha_t_bar, X_t.device)
        Q_t = predefine_transitional_matrix.get_Q_t(alpha_t, X_t.device)

        # if jointly:
        #     noisy_data, rp_atom_nums = pack_reactant_with_product(torch.tensor([self.args.sample_atom_num]*bs).to(batch.s_atom_nums.device), 
        #                                             X_t, E_t,
        #                                             batch.s_atom_nums, batch.s_atom_symbols, batch.s_bond_symbols,
        #                                             t, batch.g_y)
        #     X_t, E_t, y_t = noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        
        extra_data = RetroDiffExtraFeature({"X_t": X_t, "E_t": E_t, "y_t": y_t, "t_int": t}).to_extra_data(self.DatasetInfos)
        X_t = torch.cat((X_t, extra_data['X']), dim=2).float()
        E_t = torch.cat((E_t, extra_data['E']), dim=3).float()
        y_t = torch.hstack((y_t, extra_data['y'])).float()

        pred_data = model(X_t, E_t, y_t, node_mask)

        # if jointly:
        #     pred_data = self.model(X_t, E_t, y_t, noisy_data['atom_mask'])
        #     pred_data = unpack_reactant_with_product(pred_data['X_0'].cpu(), pred_data['E_0'].cpu(), pred_data['y_0'], rp_atom_nums.cpu(), torch.tensor([self.args.sample_atom_num]*bs).cpu())
        # else:
        #     pred_data = self.model(X_t, E_t, y_t, node_mask)

        pred_X = F.softmax(pred_data['X_0'], dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred_data['E_0'], dim=-1)               # bs, n, n, d0
        
        # x_t & x_0 -> x_s
        pX_s_given_t_and_0 = self.posterior_distribution(X_t=raw_X_t,
                                                        Q_t=Q_t['X'],
                                                        Q_s_bar=Q_s_bar['X'],
                                                        Q_t_bar=Q_t_bar['X'])

        pE_s_given_t_and_0 = self.posterior_distribution(X_t=raw_E_t,
                                                        Q_t=Q_t['E'],
                                                        Q_s_bar=Q_s_bar['E'],
                                                        Q_t_bar=Q_t_bar['E'])
                                      
        # x_t -> x_s                          
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * pX_s_given_t_and_0.type_as(pred_X)         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * pE_s_given_t_and_0.type_as(pred_E)        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
        
        
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        # z_s -> x_s
        X_s, E_s, y_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
        X_s = F.one_hot(X_s, num_classes=X_t.size(-1)).float()
        E_s = F.one_hot(E_s, num_classes=E_t.size(-1)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (raw_X_t.shape == X_s.shape) and (raw_E_t.shape == E_s.shape)
        
        
        Graph_s = BatchGraphMask(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0)).type_as(y_t).mask(node_mask)


        return {'X': Graph_s.X, 'E': Graph_s.E, 'y': Graph_s.y}, pred_data
        
        
    
        
    def posterior_distribution(self, X_t, Q_t, Q_s_bar, Q_t_bar):
        """ M: X or E
            Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
            X_t: bs, n, dt          or bs, n, n, dt
            Qt: bs, d_t-1, dt
            Qsb: bs, d0, d_t-1
            Qtb: bs, d0, dt.
        """
        # Flatten feature tensors
        # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
        # bs x (n ** 2) x d
        X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float64)         # bs x N x dt
    
        Q_t_transposed = Q_t.transpose(-1, -2)      # bs, dt, d_t-1
        left_term = X_t @ Q_t_transposed            # bs, N, d_t-1
        left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1
    
        right_term = Q_s_bar.unsqueeze(1)           # bs, 1, d0, d_t-1
        numerator = left_term * right_term          # bs, N, d0, d_t-1
    
    
        X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N
    
        prod = Q_t_bar @ X_t_transposed             # bs, d0, N
        prod = prod.transpose(-1, -2)               # bs, N, d0
        denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
        denominator[denominator == 0] = 1e-6
    
        out = numerator / denominator
        return out
   



