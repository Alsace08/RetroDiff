import scipy.spatial
import torch
import sys
from data.data_prepare import RetroDiffDataset, RetroDiffDataInfos
from arguments import arg_parses


class DatasetDistribution:
    def __init__(self, args):
        dataset = RetroDiffDataset(dataset_name="USPTO50K",
                               train_batch_size=args.train_batch_size,
                               val_batch_size=args.val_batch_size,
                               test_batch_size=args.test_batch_size,
                               num_workers=args.num_workers,
                               )
        self.train_dataloader, val_dataloader, test_dataloader = dataset.prepare()
        DatasetInfos = RetroDiffDataInfos()
        self.XE_dims, _ = DatasetInfos.compute_io_dims(val_dataloader)
        
        dim_X, dim_E = self.XE_dims['X'], self.XE_dims['E']
        self.x_dist = torch.zeros(dim_X)
        self.e_dist = torch.zeros(dim_E)
        self.total_x = 0
        self.total_e = 0
    
    
    def compute_marginal(self):
        for batch_idx, batch in enumerate(self.train_dataloader):
            print(batch_idx)
            bs = len(batch)
            r_X = batch.r_atom_symbols
            r_E = batch.r_bond_symbols
            p_X = batch.p_atom_symbols
            p_E = batch.p_bond_symbols
            
            for i in range(bs):
                self.update_x_dist(r_X[i])
                self.update_x_dist(p_X[i])
                self.update_e_dist(r_E[i])
                self.update_e_dist(p_E[i])
                
        self.x_dist /= self.total_x
        self.e_dist /= self.total_e
        
        return self
                
                
    def update_x_dist(self, X):
        for i in range(X.size(0)):
            x_type = torch.argmax(X[i], dim=-1)
            self.x_dist[x_type] += 1
            self.total_x += 1
        
        return self
        
    
    def update_e_dist(self, E):
        for i in range(E.size(0)):
            for j in range(i + 1, E.size(1)):
                e_type = torch.argmax(E[i,j], dim=-1)
                self.e_dist[e_type] += 1
                self.total_e += 1
        
        return self
    

if __name__ == "__main__":
    args = arg_parses()
    marginal_dist = DatasetDistribution(args)
    marginal_dist.compute_marginal()
    print(marginal_dist.x_dist, marginal_dist.e_dist)
    
    
        