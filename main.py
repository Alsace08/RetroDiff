# -*- coding:utf-8 -*-

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import json
import math
import traceback
import time
import logging
import scipy.spatial
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
# from torch.utils.tensorboard import SummaryWriter


from arguments import arg_parses
from data.data_prepare import RetroDiffDataset, RetroDiffDataInfos, RetroDiffExtraFeature

from diffusion.apply_noise import ForwardApplyNoise
from diffusion.de_noise import GraphTransformer
from diffusion.sampling import BackwardInference

from evaluation.metrics import GraphLoss, MolecularMetrics

from visualization.draw import MolecularVisualization

from utils.file_utils import get_current_datetime, get_logger
from utils.graph_utils import *

from rdkit import Chem


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_time = get_current_datetime()



def view_model_parameters(model):
    for name, parameters in model.named_parameters():
        if parameters.requires_grad:
            print(name, ':', parameters.size())


def optimization(args, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch:0.98**epoch)

    return optimizer, scheduler



def train(args, log):
    log.info(f"Start Time: {get_current_datetime()}")
    
    print("Loading Datasets ...")
    log.info("Loading Datasets ...")
    tic = time.time()
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
    
    DatasetInfos = RetroDiffDataInfos()
    input_dims, output_dims = DatasetInfos.compute_io_dims(train_dataloader)
    toc = time.time()
    print("Finished Data Loading in %.3f seconds" % (toc - tic))
    
    
    print("Loading Model ...")
    log.info("Loading Model ...")
    tic = time.time()
    model = GraphTransformer(n_layers=args.n_layers,
                              input_dims=input_dims,
                              hidden_mlp_dims=args.dims_mlp,
                              hidden_dims=args.dims_hidden,
                              output_dims=output_dims,
                              act_fn_in=nn.ReLU(),
                              act_fn_out=nn.ReLU())
    optimizer, lr_scheduler = optimization(args, model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    toc = time.time()
    print("Finished Model Loading in %.3f seconds" % (toc - tic))

    current_step = 0
    current_epoch = 0


    if args.dp and torch.cuda.is_available() and len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus).to(f'cuda:{args.gpus[0]}')
        #optimizer = nn.DataParallel(optimizer, device_ids=args.gpus)
        print(f"Training using {len(args.gpus)} GPUs: {model.device_ids}")
        log.info(f"Training using {len(args.gpus)} GPUs: {model.device_ids}")

    if args.from_ckpt is True:
        log.info(f"Resume from {args.ckpt_path}...")
        ckpt = torch.load(args.ckpt_path)
        current_step = ckpt["step"]
        # current_epoch = current_step // len(train_dataloader)
        current_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])



    print("Loading Metrics ...")
    log.info("Loading Metrics ...")
    TrainLoss = GraphLoss()
    MolVis = MolecularVisualization(remove_h="False", dataset_infos=DatasetInfos)



    print("Start training!")
    log.info("Start training!")


    max_gnum = args.sample_atom_num = 10
    train_loss = 0

    best_acc_group = 0
    best_acc_center = 0
    best_acc_reactant = 0
    best_acc_final = 0
    best_val_group = 0
    best_val_reactant = 0

    best_ckpt_filename = None
    val_smile_list = []
    
    while current_epoch <= args.train_epoch:
        #print("current step: " + str(current_step))
        tic = time.time()
        try:
            batch = next(train_dataiter)
        except StopIteration:
            if args.to_group_given_product:
                log.info(f"Epoch {current_epoch} Average: "
                        f"Loss_X_group: {TrainLoss.epoch_loss['group_X'] / TrainLoss.current_epoch_step :.4f} -- "
                        f"Loss_E_group: {TrainLoss.epoch_loss['group_E'] / TrainLoss.current_epoch_step :.4f}")
            if args.to_exbond_given_product_and_group:
                log.info(f"Epoch {current_epoch} Average: "
                        f"Loss_E_reaction: {TrainLoss.epoch_loss['reaction_E'] / TrainLoss.current_epoch_step :.4f}")
                      
            current_epoch += 1
            #lr_scheduler.step()
            TrainLoss.reset(step_start=False, epoch_start=True)
            train_dataiter = iter(train_dataloader)
            batch = next(train_dataiter)
        except Exception as e:
            log.info(f"Error when loading data: {e}")
            log.info(traceback.format_exc())
            exit()
        toc = time.time()
        #print("Finished Batch Loading in %.3f seconds" % (toc - tic))


        model.train()
        optimizer.zero_grad()

        if args.dp and torch.cuda.is_available() and len(args.gpus) > 1:
            device = f'cuda:{args.gpus[0]}'
        batch = batch.to_device(device)
        TrainLoss.reset(step_start=True, epoch_start=False)

        # preprocess: merge product and group -- joint training
        if args.to_group_given_product:
            Graph_gp, gp_atom_masks, gp_atom_nums = pack_group_with_product(batch.g_atom_nums, batch.g_atom_symbols, batch.g_bond_symbols, 
                                                                            batch.p_atom_nums, batch.p_atom_symbols, batch.p_bond_symbols, 
                                                                            batch.g_y, batch.product_to_group_bond,
                                                                            is_joint_external_bond=False)
        if args.to_exbond_given_product_and_group:
            Graph_gp, gp_atom_masks, gp_atom_nums = pack_group_with_product(batch.g_atom_nums, batch.g_atom_symbols, batch.g_bond_symbols, 
                                                                            batch.p_atom_nums, batch.p_atom_symbols, batch.p_bond_symbols, 
                                                                            batch.g_y, batch.product_to_group_bond,
                                                                            is_joint_external_bond=True)


        # step 1: apply noise
        tic = time.time()
        apply_noise_module = ForwardApplyNoise(args=args,
                                               X=Graph_gp.X,
                                               E=Graph_gp.E,
                                               y=Graph_gp.y,
                                               node_mask=gp_atom_masks,
                                               fix_diffusion_T=False)
        noisy_data = apply_noise_module.apply_noise()

        if args.to_group_given_product:
            noisy_data['X_t'][:, 10:, :] = Graph_gp.X[:, 10:, :]
            noisy_data['E_t'][:, 10:, :, :] = Graph_gp.E[:, 10:, :, :]
            noisy_data['E_t'][:, :, 10:, :] = Graph_gp.E[:, :, 10:, :]
        if args.to_exbond_given_product_and_group:
            noisy_data['X_t'][:, :, :] = Graph_gp.X[:, :, :]
            noisy_data['E_t'][:, 10:, 10:, :] = Graph_gp.E[:, 10:, 10:, :]
            noisy_data['E_t'][:, :10, :10, :] = Graph_gp.E[:, :10, :10, :]
            noisy_data['E_t'][:, 2:10, :, :] = Graph_gp.E[:, 2:10, :, :]
            noisy_data['E_t'][:, :, 2:10, :] = Graph_gp.E[:, :, 2:10, :]
        toc = time.time()
        #print("Finished Applying Noise in %.3f seconds" % (toc - tic))

        


        # step 2: training graph transformer
        tic = time.time()
        extra_data = RetroDiffExtraFeature(noisy_data).to_extra_data(DatasetInfos)
        X = torch.cat((noisy_data['X_t'], extra_data['X']), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data['E']), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data['y'])).float()
        
        #assert (next(model.parameters()).device == X.device)
        true_data = {'X_0': Graph_gp.X, 'E_0': Graph_gp.E, 'y_0': Graph_gp.y}
        pred_data = model(X, E, y, noisy_data['atom_mask'])
        toc = time.time()
        #print("Finished Model Training in %.3f seconds" % (toc - tic))

        
        TrainLoss.compute_graph_loss_jointly(pred_data, true_data)
        group_X_loss = TrainLoss.step_loss['group_X']
        group_E_loss = TrainLoss.step_loss['group_E']
        reaction_E_loss = TrainLoss.step_loss['reaction_E']

        if args.to_group_given_product:
            #loss = group_X_loss + group_E_loss * args.lambda_XE + reaction_E_loss * max(1, 10 * (0.98 ** current_epoch))
            loss = group_X_loss + group_E_loss * args.lambda_XE
        if args.to_exbond_given_product_and_group:
            loss = reaction_E_loss


        tic = time.time()
        loss.backward()
        optimizer.step()
        toc = time.time()
        #print("Finished Undating Parameters in %.3f seconds" % (toc - tic))


        if current_step % args.loss_interval == 0:
            if args.to_group_given_product:
                print(f"Epoch {current_epoch} -- Step {current_step}: "
                        f"Loss_X_group: {group_X_loss :.4f} -- Loss_E_group: {group_E_loss :.4f} -- TOTAL LOSS: {loss: .4f}")
                log.info(f"Epoch {current_epoch} -- Step {current_step}: "
                        f"Loss_X_group: {group_X_loss :.4f} -- Loss_E_group: {group_E_loss :.4f} -- TOTAL LOSS: {loss: .4f}")
            if args.to_exbond_given_product_and_group:
                print(f"Epoch {current_epoch} -- Step {current_step}: "
                        f"Loss_E_reaction: {reaction_E_loss :.4f} -- TOTAL LOSS: {loss: .4f}")
                log.info(f"Epoch {current_epoch} -- Step {current_step}: "
                        f"Loss_E_reaction: {reaction_E_loss :.4f} -- TOTAL LOSS: {loss: .4f}")
        
        if current_step % args.val_interval == 0 and current_step > 0:
            if args.save_ckpt:
                ckpt = {
                    "epoch": current_epoch,
                    "step": current_step,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    #"scheduler": lr_scheduler.state_dict(),
                }
                ckpt_filename = os.path.join("experiments", "checkpoints", current_time, f"model_step_{current_step}.ckpt")
                dirname = os.path.dirname(os.path.abspath(ckpt_filename))
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                torch.save(ckpt, ckpt_filename)
            
            
            sampling_module = BackwardInference(args=args,
                                                model=model,
                                                output_dims=output_dims,
                                                dataloader=test_dataloader,
                                                device=device,
                                                DatasetInfos=DatasetInfos)
            Pred_X, Pred_E, \
            product_smiles_list, reactant_smiles_list, pred_g_smiles_list, true_g_smiles_list, pred_r_smiles_list, true_r_smiles_list, \
            acc_center, acc_group, acc_reactant, acc_final, \
            val_group, val_reactant, pred_center, true_center = sampling_module.batch_sampling(args, device, log, model)


            if args.visualization:
                legend_list = ["pred_group", "true_group", "pred_reactant", "true_reactant"] * 8
                if not os.path.exists("visualization/" + current_time):
                    os.makedirs("visualization/" + current_time)
                out_filename = "visualization/" + current_time + "/ValDraw2D_" + str(current_step) + ".png"
                MolVis.batch_graph2mol(Pred_X[:32], Pred_E[:32], legend_list, out_filename)

            # for i in range(len(pred_g_smiles_list)):
            #     print(f"id: {i}")
            #     log.info(f"id: {i}")
            #     print(f"pred group smiles: {pred_g_smiles_list[i]} --- true group smiles: {true_g_smiles_list[i]}")
            #     log.info(f"pred group smiles: {pred_g_smiles_list[i]} --- true group smiles: {true_g_smiles_list[i]}")
            #     print(f"pred reactent smiles: {pred_r_smiles_list[i]} --- true reactant smiles: {true_r_smiles_list[i]}")
            #     log.info(f"pred reactent smiles: {pred_r_smiles_list[i]} --- true reactant smiles: {true_r_smiles_list[i]}")
            #     print(f"pred center: {pred_center[i]} -- true center: {true_center[i]}")
            #     log.info(f"pred center: {pred_center[i]} -- true center: {true_center[i]}")
            #     print("---")
            #     log.info("---")

            
            print(f"step {current_step} group accuracy: {acc_group}")
            log.info(f"step {current_step} group accuracy: {acc_group}")
            print(f"step {current_step} center accuracy: {acc_center}")
            log.info(f"step {current_step} center accuracy: {acc_center}")
            # print(f"step {current_step} reactant accuracy: {acc_reactant}")
            # log.info(f"step {current_step} reactant accuracy: {acc_reactant}")
            print(f"step {current_step} final accuracy: {acc_final}")
            log.info(f"step {current_step} final accuracy: {acc_final}")
            

            print(f"step {current_step} group validity: {val_group}")
            log.info(f"step {current_step} group validity: {val_group}")
            print(f"step {current_step} reactant validity: {val_reactant}")
            log.info(f"step {current_step} reactant validity: {val_reactant}")

            
            if acc_group >= best_acc_group:
                best_acc_group = acc_group
            if acc_center >= best_acc_center:
                best_acc_center = acc_center
            # if acc_reactant >= best_acc_reactant:
            #     best_acc_reactant = acc_reactant
            #     best_ckpt_filename = ckpt_filename if args.save_ckpt else None
            if acc_final >= best_acc_final:
                best_acc_final = acc_final
                best_ckpt_filename = ckpt_filename if args.save_ckpt else None
            
            
            if val_group >= best_val_group:
                best_val_group = val_group
            if val_reactant >= best_val_reactant:
                best_val_reactant = val_reactant
                

            print(f"best_group_accuracy: {best_acc_group}")
            log.info(f"best_group_accuracy: {best_acc_group}")
            print(f"best_center_accuracy: {best_acc_center}")
            log.info(f"best_center_accuracy: {best_acc_center}")
            # print(f"best_reactant_accuracy: {best_acc_reactant}")
            # log.info(f"best_reactant_accuracy: {best_acc_reactant}")
            print(f"best_final_accuracy: {best_acc_final}")
            log.info(f"best_final_accuracy: {best_acc_final}")
                
            print(f"best_group_validity: {best_val_group}")
            log.info(f"best_group_validity: {best_val_group}")
            print(f"best_reactant_validity: {best_val_reactant}")
            log.info(f"best_reactant_validity: {best_val_reactant}")

            print(f"best_ckpt_filename: {best_ckpt_filename}")
            log.info(f"best_ckpt_filename: {best_ckpt_filename}")

        
        # writer.add_scalar("X_loss", X_loss.detach(), current_step)
        # writer.add_scalar("E_loss", E_loss.detach(), current_step)
        current_step += 1


    # writer.close()
    
    
    # if args.test_after_train:
        # step 3: sampling (inference)
        # sampling_module = BackwardInference(args=args,
        #                                     model=model,
        #                                     output_dims=output_dims,
        #                                     dataloader=test_dataloader,
        #                                     device=device,
        #                                     DatasetInfos=DatasetInfos)
                                            
        # TestLoss = sampling_module.compute_loss(args)  # class: GraphLoss
        # log.info(f"Final Testing: "
        #           f"Loss_X: {TestLoss.epoch_loss['X'] / TestLoss.current_epoch_step :.4f} -- "
        #           f"Loss_E: {TestLoss.epoch_loss['E'] / TestLoss.current_epoch_step :.4f}")
        
        # sampling_module.compute_sampling_accuracy(args)
        
        
                  
    log.info(f"End Time: {get_current_datetime()}")
    


def test(args, log):
    dataset = RetroDiffDataset(dataset_name="USPTO50K",
                               train_batch_size=args.train_batch_size,
                               val_batch_size=args.val_batch_size,
                               test_batch_size=args.test_batch_size,
                               num_workers=args.num_workers,
                               )
    train_dataloader, val_dataloader, test_dataloader = dataset.prepare()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DatasetInfos = RetroDiffDataInfos()
    input_dims, output_dims = DatasetInfos.compute_io_dims(train_dataloader)


    model_group = GraphTransformer(n_layers=10,
                              input_dims=input_dims,
                              hidden_mlp_dims=args.dims_mlp,
                              hidden_dims=args.dims_hidden,
                              output_dims=output_dims,
                              act_fn_in=nn.ReLU(),
                              act_fn_out=nn.ReLU())
    model_exbond = GraphTransformer(n_layers=9,
                              input_dims=input_dims,
                              hidden_mlp_dims=args.dims_mlp,
                              hidden_dims=args.dims_hidden,
                              output_dims=output_dims,
                              act_fn_in=nn.ReLU(),
                              act_fn_out=nn.ReLU())
    optimizer_group, _ = optimization(args, model_group)
    optimizer_exbond, _ = optimization(args, model_exbond)

    if args.dp and torch.cuda.is_available() and len(args.gpus) > 1:
        model_group = nn.DataParallel(model_group, device_ids=args.gpus).to(f'cuda:{args.gpus[0]}')
        model_exbond = nn.DataParallel(model_exbond, device_ids=args.gpus).to(f'cuda:{args.gpus[0]}')
        device = f'cuda:{args.gpus[0]}'
    #log.info(f"Best Checkpoint from {args.ckpt_path}...")
    
    ckpt_group = torch.load("/sharefs/yiming-w/RetroDiff/experiments/checkpoints/2023_08_07_01:22:41/model_step_73500.ckpt")
    ckpt_exbond = torch.load("/sharefs/yiming-w/RetroDiff/experiments/checkpoints/2023_08_28_14:03:46/model_step_30500.ckpt")
    
    model_group.load_state_dict(ckpt_group["model"])
    model_exbond.load_state_dict(ckpt_exbond["model"])
    optimizer_group.load_state_dict(ckpt_group["optim"])
    optimizer_exbond.load_state_dict(ckpt_exbond["optim"])



    sampling_module = BackwardInference(args=args,
                                        model=model_group,
                                        output_dims=output_dims,
                                        dataloader=test_dataloader,
                                        device=device,
                                        DatasetInfos=DatasetInfos)
    sampling_module.test_sampling(args, device, log, model_exbond)

    







def exbond_prediction(args):
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DatasetInfos = RetroDiffDataInfos()
    input_dims, output_dims = DatasetInfos.compute_io_dims(train_dataloader)

    current_step = 0
    current_epoch = 0

    model = GraphTransformer(n_layers=args.n_layers,
                              input_dims=input_dims,
                              hidden_mlp_dims=args.dims_mlp,
                              hidden_dims=args.dims_hidden,
                              output_dims=output_dims,
                              act_fn_in=nn.ReLU(),
                              act_fn_out=nn.ReLU())
    optimizer, lr_scheduler = optimization(args, model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    current_step = 0
    current_epoch = 0


    if args.dp and torch.cuda.is_available() and len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus).to(f'cuda:{args.gpus[0]}')
        #optimizer = nn.DataParallel(optimizer, device_ids=args.gpus)
        print(f"Training using {len(args.gpus)} GPUs: {model.device_ids}")
        log.info(f"Training using {len(args.gpus)} GPUs: {model.device_ids}")

    if args.from_ckpt is True:
        log.info(f"Resume from {args.ckpt_path}...")
        ckpt = torch.load(args.ckpt_path)
        current_step = ckpt["step"]
        # current_epoch = current_step // len(train_dataloader)
        current_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

    TrainLoss = GraphLoss()
    MolVis = MolecularVisualization(remove_h="False", dataset_infos=DatasetInfos)



    while current_epoch <= args.train_epoch:
        try:
            batch = next(train_dataiter)
        except StopIteration:
            if args.to_group_given_product:
                log.info(f"Epoch {current_epoch} Average: "
                        f"Loss_X_group: {TrainLoss.epoch_loss['group_X'] / TrainLoss.current_epoch_step :.4f} -- "
                        f"Loss_E_group: {TrainLoss.epoch_loss['group_E'] / TrainLoss.current_epoch_step :.4f}")
            if args.to_exbond_given_product_and_group:
                log.info(f"Epoch {current_epoch} Average: "
                        f"Loss_E_reaction: {TrainLoss.epoch_loss['reaction_E'] / TrainLoss.current_epoch_step :.4f}")
                      
            current_epoch += 1
            #lr_scheduler.step()
            TrainLoss.reset(step_start=False, epoch_start=True)
            train_dataiter = iter(train_dataloader)
            batch = next(train_dataiter)
        except Exception as e:
            log.info(f"Error when loading data: {e}")
            log.info(traceback.format_exc())
            exit()
        toc = time.time()
        #print("Finished Batch Loading in %.3f seconds" % (toc - tic))


        model.train()
        optimizer.zero_grad()

        if args.dp and torch.cuda.is_available() and len(args.gpus) > 1:
            device = f'cuda:{args.gpus[0]}'
        batch = batch.to_device(device)
        TrainLoss.reset(step_start=True, epoch_start=False)


        Graph_gp, gp_atom_masks, gp_atom_nums = pack_group_with_product(batch.g_atom_nums, batch.g_atom_symbols, batch.g_bond_symbols, 
                                                                            batch.p_atom_nums, batch.p_atom_symbols, batch.p_bond_symbols, 
                                                                            batch.g_y, batch.product_to_group_bond,
                                                                            is_joint_external_bond=False)


        noisy_data = {'X_t': Graph_gp.X, 'E_t': Graph_gp.E, 'y_t': Graph_gp.y, 't_int': torch.zeros(args.train_batch_size, 1)}
        extra_data = RetroDiffExtraFeature(noisy_data).to_extra_data(DatasetInfos)
        X = torch.cat((Graph_gp.X, extra_data['X']), dim=2).float()
        E = torch.cat((Graph_gp.E, extra_data['E']), dim=3).float()
        y = torch.hstack((Graph_gp.y, extra_data['y'])).float()
        
        #assert (next(model.parameters()).device == X.device)
        true_data = {'X_0': Graph_gp.X, 'E_0': Graph_gp.E, 'y_0': Graph_gp.y}
        pred_data = model(X, E, y, gp_atom_masks)
        
        loss_coefficient = torch.zeros(pred_data['E_0'].size()).to(device)
        for i in range(args.train_batch_size):
            exbond_list = batch.product_to_group_bond[i]
            for bond in exbond_list:
                loss_coefficient[i, bond[1], bond[0] + 10, :] = torch.tensor([10.0] * 5)
        
        TrainLoss.compute_graph_loss_jointly(pred_data, true_data, loss_coefficient)
        reaction_E_loss = TrainLoss.step_loss['reaction_E']
        loss = reaction_E_loss

        loss.backward()
        optimizer.step()

        if current_step % args.loss_interval == 0:
            print(f"Epoch {current_epoch} -- Step {current_step}: "
                    f"Loss_E_reaction: {reaction_E_loss :.4f} -- TOTAL LOSS: {loss: .4f}")
            log.info(f"Epoch {current_epoch} -- Step {current_step}: "
                    f"Loss_E_reaction: {reaction_E_loss :.4f} -- TOTAL LOSS: {loss: .4f}")
        
        if current_step % args.val_interval == 0:
            if args.save_ckpt:
                ckpt = {
                    "epoch": current_epoch,
                    "step": current_step,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    #"scheduler": lr_scheduler.state_dict(),
                }
                ckpt_filename = os.path.join("experiments", "checkpoints", current_time, f"model_step_{current_step}.ckpt")
                dirname = os.path.dirname(os.path.abspath(ckpt_filename))
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                torch.save(ckpt, ckpt_filename)
            
            
            acc = exbond_prediction_test(args=args,
                                        model=model,
                                        output_dims=output_dims,
                                        dataloader=test_dataloader,
                                        device=device,
                                        DatasetInfos=DatasetInfos)

            print(f"final accuracy: {acc}")
            log.info(f"final accuracy: {acc}")

        current_step += 1


def exbond_prediction_test(args, model, output_dims, dataloader, device, DatasetInfos):
    model = model.eval()
    acc_center = 0
    total_num = 0

    for batch_idx, batch in enumerate(dataloader):
        batch_size = len(batch)
        batch = batch.to_device(device)

        Graph_gp, gp_atom_masks, gp_atom_nums = pack_group_with_product(batch.g_atom_nums, 
                                                                        batch.g_atom_symbols, 
                                                                        batch.g_bond_symbols,
                                                                        batch.p_atom_nums, 
                                                                        batch.p_atom_symbols, 
                                                                        batch.p_bond_symbols, 
                                                                        batch.g_y,
                                                                        batch.product_to_group_bond,
                                                                        is_joint_external_bond=False)
            
        noisy_data = {'X_t': Graph_gp.X, 'E_t': Graph_gp.E, 'y_t': Graph_gp.y, 't_int': torch.zeros(batch_size, 1)}
        extra_data = RetroDiffExtraFeature(noisy_data).to_extra_data(DatasetInfos)
        X = torch.cat((Graph_gp.X, extra_data['X']), dim=2).float()
        E = torch.cat((Graph_gp.E, extra_data['E']), dim=3).float()
        y = torch.hstack((Graph_gp.y, extra_data['y'])).float()

        pred_data = model(X, E, y, gp_atom_masks)
        X, E, y = pred_data['X_0'], pred_data['E_0'], pred_data['y_0']
        
        Final_Graph = BatchGraphMask(X, E, y).type_as(X).mask(node_mask=gp_atom_masks, collapse=True)
        Pred_X, Pred_E, Pred_y = Final_Graph.X, Final_Graph.E, Final_Graph.y
        


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
                    if Pred_E[i,row,col] > 0:
                        bond_type = None
                        if Pred_E[i,row,col] == 1: bond_type = Chem.rdchem.BondType.SINGLE
                        if Pred_E[i,row,col] == 2: bond_type = Chem.rdchem.BondType.DOUBLE
                        if Pred_E[i,row,col] == 3: bond_type = Chem.rdchem.BondType.TRIPLE
                        if Pred_E[i,row,col] == 4: bond_type = Chem.rdchem.BondType.AROMATIC
                        tmp_pred_bond.append([col - 10, row, bond_type])
            
            total_num += 1
            print(f"epoch {batch_idx} -- total num: {total_num}")
            log.info(f"epoch {batch_idx} -- total num: {total_num}")
            print(f"pred center: {tmp_pred_bond} -- true center: {tmp_true_bond}")
            log.info(f"pred center: {tmp_pred_bond} -- true center: {tmp_true_bond}")
            print("---")
            log.info("---")

            if tmp_true_bond == tmp_pred_bond:
                acc_center += 1

    return acc_center / ((batch_idx + 1) * batch_size)






if __name__ == "__main__":
    # writer = SummaryWriter(log_dir="./tensorboard")
    log = get_logger(__name__, current_time)
    log.info(f"Checkpoint Directory: ./experiments/checkpoints/{current_time}")
    
    
    args = arg_parses()
    log.info(f"{args}")
    train(args, log)

