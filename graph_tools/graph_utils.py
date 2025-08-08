import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import scipy.linalg as scipy


def compute_sobolev_matrix( Laplacian, epsilon, beta):
        # Create eye matrix on the same device as Laplacian
        sobolev = Laplacian + epsilon * torch.eye(Laplacian.shape[0], device=Laplacian.device)
        sobolev = 0.5 * (sobolev + sobolev.T)
        if beta != 1.0:
            sobolev = scipy.fractional_matrix_power(sobolev.cpu().numpy(), beta)
            sobolev = torch.tensor(sobolev, dtype=torch.float32, device=Laplacian.device)
        return sobolev
def create_Dh_numpy(M):
    """
    Create a difference matrix Dh of size (M, M-1) using numpy.
    """
    Dh = np.zeros((M, M - 1))
    for i in range(M - 1):
        Dh[i, i] = -1
        Dh[i + 1, i] = 1
    return Dh

def create_Dh_torch(M, device='cpu'):
    """
    Create a difference matrix Dh of size (M, M-1) using PyTorch.
    """
    Dh = torch.zeros((M, M - 1), device=device)
    for i in range(M - 1):
        Dh[i, i] = -1
        Dh[i + 1, i] = 1
    return Dh

def get_mask(matrix, percentage_train):
    # Flatten the matrix
    num_elements = matrix.numel()
    num_train = int(num_elements * percentage_train)  # % for training
    # num_test = num_elements - num_train  # rest for testing

    # Create shuffled indices
    indices = torch.randperm(num_elements)

    # Create mask
    mask_flat = torch.zeros(num_elements, dtype=torch.bool)
    mask_flat[indices[:num_train]] = True  # Mark training points

    # Reshape the mask back to the original shape
    mask = mask_flat.view_as(matrix)

    # Create training and test sets
    train_set = matrix * mask
    test_set = matrix * (~mask)
    return train_set, test_set, mask


def solver_NNI(SampleMatrix, SampledData, Position):
    """
    Natural neighbor interpolation (approximated using linear interpolation).
    
    Parameters:
        SampleMatrix (ndarray): Sampling operator (N x T)
        SampledData (ndarray): Sampled data (N x T)
        Position (ndarray): Positions of all vertices (N x 2)
        
    Returns:
        x_recon (ndarray): Reconstructed graph signal (N x T)
    """
    N, T = SampledData.shape
    x_recon = SampledData.copy()
    
    xi, yi, ti, zi = [], [], [], []
    
    for TimeInd in range(T):
        temp1 = SampleMatrix[:, TimeInd]
        temp2 = SampledData[:, TimeInd]
        x1 = np.where(temp1 != 0)[0]
        z1 = temp2[x1]
        xi.extend(Position[x1, 0])
        yi.extend(Position[x1, 1])
        ti.extend([TimeInd] * len(x1))
        zi.extend(z1)

    points = np.column_stack((xi, yi, ti))
    values = np.array(zi)
    
    F = LinearNDInterpolator(points, values)  # approximate to 'natural' interpolation
    
    for TimeInd in range(T):
        temp1 = SampleMatrix[:, TimeInd]
        x2 = np.where(temp1 == 0)[0]
        interp_points = np.column_stack((Position[x2, 0], Position[x2, 1], [TimeInd] * len(x2)))
        interpolated_values = F(interp_points)
        x_recon[x2, TimeInd] = interpolated_values

    return x_recon


def solver_NNI_nan(SampleMatrix, SampledData, Position):
    """
    Natural neighbor interpolation (approximated using linear interpolation).
    
    Parameters:
        SampleMatrix (ndarray): Sampling operator (N x T)
        SampledData (ndarray): Sampled data (N x T)
        Position (ndarray): Positions of all vertices (N x 2)
        
    Returns:
        x_recon (ndarray): Reconstructed graph signal (N x T)
    """
    N, T = SampledData.shape
    x_recon = SampledData.copy()
    
    xi, yi, ti, zi = [], [], [], []
    
    for TimeInd in range(T):
        temp1 = SampleMatrix[:, TimeInd]
        temp2 = SampledData[:, TimeInd]
        x1 = np.where(temp1 != 0)[0]
        z1 = temp2[x1]
        xi.extend(Position[x1, 0])
        yi.extend(Position[x1, 1])
        ti.extend([TimeInd] * len(x1))
        zi.extend(z1)

    points = np.column_stack((xi, yi, ti))
    values = np.array(zi)
    
    # Build interpolators
    F_linear = LinearNDInterpolator(points, values)
    F_nearest = NearestNDInterpolator(points, values)

    for TimeInd in range(T):
        temp1 = SampleMatrix[:, TimeInd]
        x2 = np.where(temp1 == 0)[0]  # Indices where data is missing
        
        if len(x2) == 0:
            continue  # Nothing to interpolate for this time step
        
        # Create interpolation input points (x, y, time)
        interp_points = np.column_stack((
            Position[x2, 0],  # x
            Position[x2, 1],  # y
            np.full(len(x2), TimeInd)  # time
        ))
        
        # Try linear interpolation
        interpolated_values = F_linear(interp_points)
        
        # Replace NaNs (outside convex hull) with nearest neighbor interpolation
        if np.any(np.isnan(interpolated_values)):
            nan_mask = np.isnan(interpolated_values)
            interpolated_values[nan_mask] = F_nearest(interp_points[nan_mask])
        
        # Store reconstructed values
        x_recon[x2, TimeInd] = interpolated_values
        
    return x_recon