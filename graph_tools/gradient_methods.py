import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from graph_tools.graph_utils import create_Dh_numpy, create_Dh_torch
import scipy.linalg as scipy

# class Sobolev:
    
#     def __init__(self, coefficients, Laplacian, epsilon, beta, M):
#         # super().__init__(learning_rate)
#         self.coefficients = coefficients
#         self.Laplacian = Laplacian
#         self.epsilon = epsilon
#         self.beta = beta
#         self.Sobolev = self.sobolev_matrix()
#         self.Dh = create_Dh_torch(M)
    
#     def sobolev_matrix(self):
#         # From MATLAB
#         # https://github.com/jhonygiraldo/GraphTRSS/blob/main/sea_surface_temperature_experiment/evolution_loss/evolution_loss_normalized.m
#         # sobolev_matrix = G.L+sob.epsilon_set(sob.best_epsilon)*eye(G.N);
#         # %% Symmetrization
#         # sobolev_matrix = 0.5*(sobolev_matrix+sobolev_matrix');
#         # sobolev_matrix = sparse(sobolev_matrix);
#         # TODO: Missing the power of beta
#         sobolev = self.Laplacian + self.epsilon * np.eye(self.Laplacian.shape[0])
#         # Missing power of beta
#         sobolev = 0.5 * (sobolev + sobolev.T)
#         sobolev = np.linalg.matrix_power(sobolev, self.beta)
#         # return sobolev
#         return torch.tensor(sobolev, dtype=torch.float32)  # Adjust dtype if needed
#     def loss(self, x, y, mask):
#         loss_1 = F.mse_loss(torch.mul(x,mask), y)
#         x_diff = x @ self.Dh
#         loss_2 = torch.trace( x_diff.T @  self.Sobolev @ x_diff)
#         return self.coefficients[0]*loss_1 + self.coefficients[1]*loss_2
    
# class Tikhonov:
#     def __init__(self, coefficients, Laplacian):
#         self.coefficients = coefficients
#         self.Laplacian = Laplacian

#     def loss(self, x, y, mask):
#         # x_tilde = torch.mul(x, mask)
#         loss_mse = F.mse_loss(x[mask], y[mask])
#         # loss_mse = F.mse_loss(torch.mul(x,mask),  y)
#         loss_lap = torch.trace(x.T@self.Laplacian@x)
#         return self.coefficients[0] * loss_mse + self.coefficients [1] * loss_lap
    
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
        self.Sobolev = self.compute_sobolev_matrix(Laplacian, epsilon, beta)
        self.Dh = create_Dh_torch(M)

    def compute_sobolev_matrix(self, Laplacian, epsilon, beta):
        sobolev = Laplacian + epsilon * np.eye(Laplacian.shape[0])
        sobolev = 0.5 * (sobolev + sobolev.T)
        sobolev = scipy.fractional_matrix_power(sobolev, beta)
        return torch.tensor(sobolev, dtype=torch.float32)

    def __call__(self, x, y, mask):
        x_diff = x @ self.Dh
        loss_sob = torch.trace(x_diff.T @ self.Sobolev @ x_diff)
        return self.coefficient * loss_sob

class WeightedSobolevLoss(BaseLoss):
    def __init__(self, coefficient, Laplacian, epsilon, beta, M):
        super().__init__(coefficient)
        self.Sobolev = self.compute_sobolev_matrix(Laplacian, epsilon, beta)
        self.Dh = create_Dh_torch(M)

    def compute_sobolev_matrix(self, Laplacian, epsilon, beta):
        sobolev = Laplacian + epsilon * np.eye(Laplacian.shape[0])
        sobolev = 0.5 * (sobolev + sobolev.T)
        sobolev = scipy.fractional_matrix_power(sobolev, beta)
        return torch.tensor(sobolev, dtype=torch.float32)

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
    def __init__(self, lambdas, x, y, mask, Dh, epsilon, beta):
        self.coefficient = coefficient
    
    def primal(self):
        return self.lambdas
    
    def __call__(self, x, y, mask):
        raise NotImplementedError("Each loss must implement the __call__ method.")

    def dual(lambdas, x, y, mask, Dh, epsilon, beta):
        """
        Update the lambdas using the primal-dual method.
        """
        x_diff = x @ Dh
        loss_sob = torch.diag(x_diff.T @ self.Sobolev @ x_diff)
        
        # Update rule for lambdas
        lambdas = lambdas - epsilon * loss_sob
        
        return lambdas