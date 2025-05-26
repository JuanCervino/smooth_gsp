import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from graph_tools.graph_utils import create_Dh_numpy, create_Dh_torch
import scipy.linalg as scipy



def compute_sobolev_matrix(Laplacian, epsilon, beta):
    sobolev = Laplacian + epsilon * np.eye(Laplacian.shape[0])
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

    # def compute_sobolev_matrix(self, Laplacian, epsilon, beta):
    #     sobolev = Laplacian + epsilon * np.eye(Laplacian.shape[0])
    #     sobolev = 0.5 * (sobolev + sobolev.T)
    #     sobolev = scipy.fractional_matrix_power(sobolev, beta)
    #     return torch.tensor(sobolev, dtype=torch.float32)

    def __call__(self, x, y, mask, lambdas):
        x_diff = x @ self.Dh
        loss_sob = torch.diag(x_diff.T @ self.Sobolev @ x_diff)
        return lambdas @ loss_sob

class TemporalLoss(BaseLoss):
    def __init__(self, coefficient, Dh):
        super().__init__(coefficient)
        self.Dh = Dh

    def __call__(self, x, y, mask):
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
     
    elif method == "PrimalDual":
        
        sobolev_loss = WeightedSobolevLoss(coefficient_sob, Laplacian, epsilon, beta, M)
        
        return CombinedLoss(mse_loss, temporal_loss)

     
    elif method == 'All':
        lap_loss = LaplacianLoss(coefficient_lap, Laplacian)
        sobolev_loss = SobolevLoss(coefficient_sob, Laplacian, epsilon, beta, M)
        temporal_loss = TemporalLoss(coefficient_temp, M)
        
        return CombinedLoss(mse_loss, temporal_loss, lap_loss, sobolev_loss)
    else:
        raise ValueError('Method not implemented')
    
    
    
    
class primal_dual_loss:
    def __init__(self, x, y, mask, Dh, Laplacian, epsilon, beta, alpha,lambdas, mu):
        self.epsilon = epsilon
        self.mu = 1.
        self.lambdas = lambdas
        self.Dh = Dh
        self.Sobolev = self.compute_sobolev_matrix(x.shape[1], epsilon, beta)
        
    def primal(self, x, y, mask):
        
        loss_mse = self.epsilon * F.mse_loss(torch.mul(x, mask), y)
        x_diff = x @ self.Dh
        loss_sob = torch.trace(x_diff.T @ self.Sobolev @ x_diff)
        return self.lambdas
    
    def dual(lambdas, x, y, mask, Dh, epsilon, beta):
        """
        Update the lambdas using the primal-dual method.
        """
        x_diff = x @ Dh
        loss_sob = torch.diag(x_diff.T @ self.Sobolev @ x_diff)
        
        # Update rule for lambdas
        lambdas = lambdas - epsilon * loss_sob
        
        return lambdas