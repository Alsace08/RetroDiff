import torch

class TransitionDistribution:
    def __init__(self,):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        
        self.u_x = torch.zeros(0)
        self.u_e = torch.zeros(0)
        self.u_y = torch.zeros(0)
        
        if q_type == "uniform":
            uniform()
        elif q_type == "marginal":
            marginal()
        elif q_type == "sample_specific":
            sample_specific()
        else:
            raise "Invalid Transitional Matrix!"
        
            
    
    def uniform(self):
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes
            
        return self
    
    
    def dataset_marginal(self):
        
    
    def transitional_marginal(self):
        pass
    
    def sample_specific(self):
        pass