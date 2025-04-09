# -*- coding:utf-8 -*-
import sys
import numpy as np
import torch
import torch.nn.functional as F


sys.path.append('../')
#from prior.transition import TransitionDistribution
from data.data_prepare import RetroDiffDataInfos
from utils.graph_utils import BatchGraphMask



class PredefineAlpha(torch.nn.Module):
    def __init__(self, noise_schedule, total_diffusion_T):
        super(PredefineAlpha, self).__init__()
        self.noise_schedule = noise_schedule
        self.T = total_diffusion_T

            
    def get_alpha_t(self, t_int):
        if self.noise_schedule == 'cosine':
            alpha_all, _ = self.cosine_schedule(self.T)
        else:
            raise ValueError(noise_schedule)
            
        return alpha_all[t_int]
        
        
    def get_alpha_t_bar(self, t_int):
        if self.noise_schedule == 'cosine':
            _, alpha_bar_all = self.cosine_schedule(self.T)
        else:
            raise ValueError(noise_schedule)
            
        return alpha_bar_all[t_int]
    
    
    def cosine_schedule(self, s=0.008):
        """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
        steps = self.T + 2
        x = np.linspace(0, steps, steps)
    
        alpha_bar_all = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
        alpha_bar_all = alpha_bar_all / alpha_bar_all[0]
        
        alpha_all = (alpha_bar_all[1:] / alpha_bar_all[:-1])
        # beta_all = 1 - alpha_all
        # beta_bar_all = np.cumprod(beta_all)
        
        return torch.tensor(alpha_all.squeeze()), torch.tensor(alpha_bar_all.squeeze())




class UniformTransition:
    def __init__(self, 
                x_classes: int, 
                e_classes: int, 
                y_classes: int):
                
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        
        
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes
        

    def get_Q_t(self, alpha_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = alpha_t * I + (1 - alpha_t) / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        
        alpha_t = alpha_t.unsqueeze(1)
        alpha_t = alpha_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = (1 - alpha_t) * self.u_x + alpha_t * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = (1 - alpha_t) * self.u_e + alpha_t * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = (1 - alpha_t) * self.u_y + alpha_t * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return {'X': q_x, 'E': q_e, 'y': q_y}

    def get_Q_t_bar(self, alpha_t_bar, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t. alpha_t from 1 to 0.
        Qt = prod(alpha_t) * I + (1 - prod(alpha_t)) / K

        alpha_t_bar: (bs)
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_t_bar = alpha_t_bar.unsqueeze(1)  # (bs, 1, 1)
        alpha_t_bar = alpha_t_bar.to(device)
        
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        
        q_x = alpha_t_bar * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_t_bar) * self.u_x
        q_e = alpha_t_bar * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_t_bar) * self.u_e
        q_y = alpha_t_bar * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_t_bar) * self.u_y

        return {'X': q_x, 'E': q_e, 'y': q_y}
        
        
class MarginalTransition:
    def __init__(self, 
                x_classes: int, 
                e_classes: int, 
                y_classes: int):
                
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        
        x_marginals = RetroDiffDataInfos().x_marginals
        e_marginals = RetroDiffDataInfos().e_marginals
        
        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes
        

    def get_Q_t(self, alpha_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = alpha_t * I + (1 - alpha_t) / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        
        alpha_t = alpha_t.unsqueeze(1)
        alpha_t = alpha_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = (1 - alpha_t) * self.u_x + alpha_t * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = (1 - alpha_t) * self.u_e + alpha_t * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = (1 - alpha_t) * self.u_y + alpha_t * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return {'X': q_x, 'E': q_e, 'y': q_y}

    def get_Q_t_bar(self, alpha_t_bar, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t. alpha_t from 1 to 0.
        Qt = prod(alpha_t) * I + (1 - prod(alpha_t)) / K

        alpha_t_bar: (bs)
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_t_bar = alpha_t_bar.unsqueeze(1)  # (bs, 1, 1)
        alpha_t_bar = alpha_t_bar.to(device)
        
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        
        q_x = alpha_t_bar * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_t_bar) * self.u_x
        q_e = alpha_t_bar * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_t_bar) * self.u_e
        q_y = alpha_t_bar * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_t_bar) * self.u_y

        return {'X': q_x, 'E': q_e, 'y': q_y}




class TransitionMode:
    def __init__(self, 
                x_classes: int, 
                e_classes: int, 
                y_classes: int,
                transition_mode: str):
        if transition_mode == "uniform":
            self.MODE = UniformTransition(x_classes, e_classes, y_classes)
        elif transition_mode == "marginal":
            self.MODE = MarginalTransition(x_classes, e_classes, y_classes)



class ForwardApplyNoise:
    def __init__(self, args, X, E, y, node_mask, fix_diffusion_T):
        self.args = args
        self.X = X
        self.E = E
        self.y = y
        self.node_mask = node_mask
        self.noise_schedule = args.noise_schedule
        self.T = args.total_diffusion_T
        self.fix_diffusion_T = fix_diffusion_T
        

    def apply_noise(self):
        training = True
        lowest_t = 0 if training else 1
        
        if self.fix_diffusion_T:
            t_int = torch.tensor([self.T] * self.X.size(0))
        else:
            t_int = torch.randint(lowest_t, self.T + 1, size=(self.X.size(0), 1), device=self.X.device)  # (bs, 1)
           

        # Define alpha_bar
        predefine_alpha = PredefineAlpha(noise_schedule=self.noise_schedule, total_diffusion_T=self.T)
        # Define Q_bar
        predefine_transitional_matrix = TransitionMode(self.X.size(-1), self.E.size(-1), self.y.size(-1), self.args.transition_mode).MODE
        
        
        # Compute alpha_t_bar
        alpha_t_bar = predefine_alpha.get_alpha_t_bar(t_int)
        # Compute Q_t_bar
        Q_t_bar = predefine_transitional_matrix.get_Q_t_bar(alpha_t_bar, self.X.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        QX_t_bar = Q_t_bar['X']
        QE_t_bar = Q_t_bar['E']
        Qy_t_bar = Q_t_bar['y']
        
        assert (abs(QX_t_bar.sum(dim=2) - 1.) < 1e-4).all(), QX_t_bar.sum(dim=2) - 1
        assert (abs(QE_t_bar.sum(dim=2) - 1.) < 1e-4).all()
        
        
        # Compute transition probabilities in step t: x0 -> zt
        self.X = self.X.type_as(QX_t_bar)
        self.E = self.E.type_as(QE_t_bar)
        probX = self.X @ QX_t_bar  # (bs, n_atom, d_atom)
        probE = self.E @ QE_t_bar.unsqueeze(1)  # (bs, n_atom, n_atom, d_bond)
        
        # Discrete Sampling: zt -> xt
        X_t, E_t, y_t = sample_discrete_features(probX=probX, probE=probE, node_mask=self.node_mask)
        X_t = F.one_hot(X_t, num_classes=self.X.size(-1))
        E_t = F.one_hot(E_t, num_classes=self.E.size(-1))
        assert (self.X.shape == X_t.shape) and (self.E.shape == E_t.shape)
        
        
        Graph_t = BatchGraphMask(X=X_t, E=E_t, y=self.y).type_as(X_t).mask(self.node_mask)
        noisy_data = {'t_int': t_int, 'X_t': Graph_t.X, 'E_t': Graph_t.E, 'y_t': Graph_t.y, 'atom_mask': self.node_mask}
        
        return noisy_data
    

def sample_discrete_features(probX, probE, node_mask):
    ''' 
    Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n_atom, d_atom)
    
    # Sample X
    X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)     # (bs, n)
    
    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)    # (bs * n * n, de_out)
    

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))
    
    # y is empty
    y_t = torch.zeros(bs, 0).type_as(X_t)
    

    return X_t, E_t, y_t
