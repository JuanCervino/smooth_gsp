import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse


from graph_tools.load_dataset import load_dataset
import numpy as np
import torch
import graph_tools.gradient_methods as gm
import torch.nn.functional as F
import graph_tools.graph_utils as gu

from torch import nn
from torch import optim
import torch.nn.functional as F

@torch.no_grad()
def accuracy(y_tilde, y):
    return F.mse_loss(y_tilde, y)

def mse(y_tilde, y):
    return np.mean((y_tilde - y) ** 2)


# [26] Qiu "Time-Varying Graph Signal Reconstruction" MSE + Temp
#  Giraldo  "Reconstuction of Time-Varying Graph Signals" MSE + Sob
#  [39]  Narang, "LOCALIZED ITERATIVE METHODS FOR INTERPOLATION IN GRAPH STRUCTURED DATA" MSE + High-Frequency


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
    train_set, test_set, mask = gu.get_mask(D, args.percentage)
    
    
    if args.method == 'nni':
        x_recon = gu.solver_NNI_nan(mask.cpu().numpy(), train_set.cpu().numpy(), G['coords'])
        print(x_recon)
        print(test_set.cpu().numpy())

        print(D.cpu().numpy())
        
        print("Total", mse(x_recon, D.cpu().numpy()))
        print("Train", mse(x_recon[mask], D[mask].cpu().numpy()))
        print("Test", mse(x_recon[~mask], D[~mask].cpu().numpy()))

    elif args.method in ['MSE', 'Tikhonov', 'Sobolev', 'GraphRegularization', 'Temporal', 'All']:
        # Create the matrix X
        X = torch.randn_like(D, requires_grad=True)

        # Create the optimizer
        optimizer = optim.Adam([X], lr=args.lr)
        
        # Get the loss function
        loss_fn = gm.get_loss(args.method, args.coefficient_mse, args.coefficient_lap, args.coefficient_temp, args.coefficient_sob, L, args.epsilon, args.beta, D.shape[1])
        
        ## Learning pipeline
        for e in range(args.epochs):
            
            # Zero the gradients before the forward pass
            optimizer.zero_grad()

            # Compute the loss
            loss = loss_fn(X, train_set, mask)

            # Backpropagate the loss
            loss.backward()

            # Update the parameters
            optimizer.step()

            if e % 100 == 0:
                acc_total = accuracy(X, D)
                acc_train = accuracy(X[mask], D[mask])
                acc_test = accuracy(X[~mask], D[~mask])

                print(f'Epoch {e}, Loss Total: {acc_total.item()}, Loss Train: {acc_train.item()}, Loss Test: {acc_test.item()}')
                
            # print(f'Gradient: {X.grad}')
        

    elif args.method in ['PrimalDual']:
        # Create the matrix X
        X = torch.randn_like(D, requires_grad=True)

        # Create the optimizer
        optimizer = optim.Adam([X], lr=args.lr)
        
        # Get the loss function
        loss_fn = gm.get_loss(args.method, args.coefficient_mse, args.coefficient_lap, args.coefficient_temp, args.coefficient_sob, L, args.epsilon, args.beta, D.shape[1])
        
        lambdas = torch.zeros(X.shape[1]-1)
        ## Learning pipeline
        for e in range(args.epochs):
            
            # Zero the gradients before the forward pass
            optimizer.zero_grad()

            # Compute the loss
            loss = loss_fn(X, train_set, mask, lambdas)
            
            # Compute the dual condition
            dual_update = gm.update_lambdas(lambdas.get_lambdas, X, train_set, mask, L, args.epsilon, args.beta)

            # Backpropagate the loss
            loss.backward()

            # Update the parameters
            optimizer.step()

            if e % 100 == 0:
                acc_total = accuracy(X, D)
                acc_train = accuracy(X[mask], D[mask])
                acc_test = accuracy(X[~mask], D[~mask])

                print(f'Epoch {e}, Loss Total: {acc_total.item()}, Loss Train: {acc_train.item()}, Loss Test: {acc_test.item()}')
            
            if e % args.primal_steps == 0:
                # Update the lambdas
                with torch.no_grad():
                    lambdas = gm.update_lambdas(X, train_set, mask, lambdas)
                    print(f'Updated lambdas: {lambdas}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Smooth Graph Signal')
    parser.add_argument('--dataset', type=str, default='sea_surface_temperature')
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--method', type=str, default='all')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--percentage', type=float, default=0.5, help='Percentage of data to be used for training between 0 and 1')
    
    
    parser.add_argument('--coefficient_mse', type=float, default=1)
    parser.add_argument('--coefficient_lap', type=float, default=0.01)
    parser.add_argument('--coefficient_sob', type=float, default=0.01)
    parser.add_argument('--coefficient_temp', type=float, default=0.01)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.5)
    
    args = parser.parse_args()
    
    main(args)