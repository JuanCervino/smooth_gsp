
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
from graph_tools.load_dataset import load_dataset
import graph_tools.graph_utils as gu

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def accuracy(y_tilde, y):
    return np.mean((y_tilde - y) ** 2)

import numpy as np
from scipy.interpolate import LinearNDInterpolator

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

def main(args):

    G, D = load_dataset(args.dataset, knn_param=args.knn)
    # Get data and adjacency matrix
    D = torch.Tensor(D)
    W = np.array(G['W'].toarray())
    
    # Create the Laplacian matrix
    L = np.diag(W@np.ones(W.shape[0])) - W
    L = torch.Tensor(L)
    
    # Create the mask
    # Sample Trajectories TODO: Add samplers
    train_set, test_set, mask = gu.get_mask(D, 0.5)

    # x_recon = solver_NNI(mask.cpu().numpy(), train_set.cpu().numpy(), G['coords'])
    x_recon = solver_NNI_nan(mask.cpu().numpy(), train_set.cpu().numpy(), G['coords'])

    print(x_recon)
    print(test_set.cpu().numpy())

    print(D.cpu().numpy())
    
    print("Total", accuracy(x_recon, D.cpu().numpy()))
    print("Train", accuracy(x_recon[mask], D[mask].cpu().numpy()))
    print("Test", accuracy(x_recon[~mask], D[~mask].cpu().numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Smooth Graph Signal')
    parser.add_argument('--dataset', type=str, default='sea_surface_temperature')
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--method', type=str, default='all')
    parser.add_argument('--lr', type=float, default=0.01)
    

    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.5)
    
    args = parser.parse_args()
    
    main(args)