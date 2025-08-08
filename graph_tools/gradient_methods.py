import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from graph_tools.graph_utils import create_Dh_numpy, create_Dh_torch
import scipy.linalg as scipy

# Example lambda function

def get_Sobolev_smoothness_function(x, Dh, Sobolev, type):
    assert type in ['integral', 'timewise'], "Type must be either 'integral' or 'timewise'."
    if type == 'timewise':
        x_diff = x @ Dh
        loss_sob = torch.diag(x_diff.T @ Sobolev @ x_diff)
        return loss_sob

    elif type == 'integral':
        x_diff = x @ Dh
        loss_sob = torch.diag(x_diff.T @ Sobolev @ x_diff)
        return torch.sum(loss_sob)

def compute_sobolev_matrix(Laplacian, epsilon, beta):
    sobolev = Laplacian + epsilon * torch.eye(Laplacian.shape[0])
    sobolev = 0.5 * (sobolev + sobolev.T)
    sobolev = scipy.fractional_matrix_power(sobolev, beta)
    return torch.tensor(sobolev, dtype=torch.float32)

class BaseLoss:
    def __init__(self, coefficient):
        self.coefficient = coefficient
        
    def __call__(self, x, y, mask):
        raise NotImplementedError("Each loss must implement the __call__ method.")

class LaplacianLoss(BaseLoss):
    def __init__(self, coefficient, Laplacian):
        super().__init__(coefficient)
        self.Laplacian = Laplacian

    def __call__(self, x, y, mask):
        loss_lap = torch.trace(x.T @ self.Laplacian @ x)
        return self.coefficient * loss_lap

class SobolevLoss(BaseLoss):
    def __init__(self, coefficient, Laplacian, epsilon, beta, M):
        super().__init__(coefficient)
        self.Sobolev = compute_sobolev_matrix(Laplacian, epsilon, beta)
        self.Dh = create_Dh_torch(M)

    # def compute_sobolev_matrix(self, Laplacian, epsilon, beta):
    #     sobolev = Laplacian + epsilon * np.eye(Laplacian.shape[0])
    #     sobolev = 0.5 * (sobolev + sobolev.T)
    #     sobolev = scipy.fractional_matrix_power(sobolev, beta)
    #     return torch.tensor(sobolev, dtype=torch.float32)

    def __call__(self, x, y, mask):
        x_diff = x @ self.Dh
        loss_sob = torch.trace(x_diff.T @ self.Sobolev @ x_diff)
        return self.coefficient * loss_sob

class WeightedSobolevLoss(BaseLoss):
    def __init__(self, coefficient, Laplacian, epsilon, beta, M):
        super().__init__(coefficient)
        self.Sobolev = compute_sobolev_matrix(Laplacian, epsilon, beta)
        self.Dh = create_Dh_torch(M)

    def __call__(self, x, y, mask, lambdas):
        x_diff = x @ self.Dh
        loss_sob = torch.diag(x_diff.T @ self.Sobolev @ x_diff)
        return lambdas @ loss_sob

class TemporalLoss(BaseLoss):
    def __init__(self, coefficient, M):
        super().__init__(coefficient)
        self.Dh = create_Dh_torch(M)
        self.coefficient = coefficient
    def __call__(self, x, y, mask):
        # print(x.shape, self.Dh.shape)
        x_diff = x @ self.Dh
        loss_temporal = torch.norm(x_diff, p=2)
        return self.coefficient * loss_temporal

class MSELoss(BaseLoss):
    def __init__(self, coefficient):
        super().__init__(coefficient)

    def __call__(self, x, y, mask):
        loss_mse = F.mse_loss(torch.mul(x, mask), y)
        return self.coefficient * loss_mse

class CombinedLoss:
    def __init__(self, *losses):
        self.losses = losses

    def __call__(self, x, y, mask):
        total_loss = 0.0
        for loss_fn in self.losses:
            total_loss += loss_fn(x, y, mask)
        return total_loss


def get_loss(method, coefficient_mse, coefficient_lap, coefficient_temp, coefficient_sob, Laplacian, epsilon, beta, M):
    
    # MSE is always used
    mse_loss = MSELoss(coefficient_mse)

    if method == 'MSE':
        return mse_loss
    
    if method == 'Tikhonov':
        
        lap_loss = LaplacianLoss(coefficient_lap, Laplacian)
        temporal_loss = TemporalLoss(coefficient_temp, M)

        return CombinedLoss(mse_loss, lap_loss, temporal_loss)
        
    elif method == 'Sobolev':
        
        sobolev_loss = SobolevLoss(coefficient_sob, Laplacian, epsilon, beta, M)

        return CombinedLoss(mse_loss, sobolev_loss)
        
    elif method == 'GraphRegularization':
        
        lap_loss = LaplacianLoss(coefficient_lap, Laplacian)

        return CombinedLoss(mse_loss, lap_loss)
    
    elif method == 'Temporal':
        
        temporal_loss = TemporalLoss(coefficient_temp, M)
        
        return CombinedLoss(mse_loss, temporal_loss)
     
     
    elif method == 'All':
        lap_loss = LaplacianLoss(coefficient_lap, Laplacian)
        sobolev_loss = SobolevLoss(coefficient_sob, Laplacian, epsilon, beta, M)
        temporal_loss = TemporalLoss(coefficient_temp, M)
        
        return CombinedLoss(mse_loss, temporal_loss, lap_loss, sobolev_loss)
    else:
        raise ValueError('Method not implemented')
    
    
    
    
class primal_dual_loss:
    def __init__(self, x, M, Laplacian, epsilon, beta, alpha, dual_step_mu, dual_step_lambdas, lambda_func):
        self.epsilon = epsilon
        self.mu = 1.
         
        self.Dh = create_Dh_torch(M)
        self.dual_step_mu = dual_step_mu
        self.dual_step_lambdas = dual_step_lambdas
        self.Sobolev = compute_sobolev_matrix(Laplacian, epsilon, beta)
        self.alpha = alpha

        # Create wrapper for lambda_func to only take x as parameter
        self.lambda_func = lambda_func
        
        # Initialize lambdas based on the output dimension of the lambda function
        with torch.no_grad():
            output_dim = self.lambda_func(x)
            # Handle scalar case (timewise returns a sum, which is scalar)
            if output_dim.dim() == 0:  # scalar tensor
                output_dim = 1
            else:
                output_dim = output_dim.shape[0]  # Use shape[0] for 1D tensor          
                  
        if output_dim == 1:
            self.lambdas = torch.tensor([1.0], requires_grad=True)
        else:
            self.lambdas = torch.ones(output_dim) / output_dim

        
    def primal(self, x, y, mask):

        loss_mse = self.mu * F.mse_loss(torch.mul(x, mask), y)

        lambda_output = self.lambda_func(x)
        if lambda_output.dim() == 0:  # scalar tensor
            loss_grad = lambda_output * self.lambdas[0]  # multiply scalar by first lambda
        else:  # vector tensor
            loss_grad = lambda_output @ self.lambdas  # matrix multiplication
           
        return loss_mse + loss_grad
    
    def dual(self, x, y, mask):
        """
        Update the lambdas using the primal-dual method.
        """
        self.mu = F.relu(self.mu + self.dual_step_mu * (F.mse_loss(torch.mul(x, mask), y)-self.alpha))

        self.lambdas = self.lambdas + self.dual_step_lambdas * self.lambda_func(x)
        
        self.lambdas = torch.max(self.lambdas, torch.zeros_like(self.lambdas))  # Ensure non-negativity
        self.lambdas = self.lambdas / torch.norm(self.lambdas, p=2) # Project onto the unit sphere
        
        pass