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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import json

@torch.no_grad()
def accuracy(y_tilde, y):
    return F.mse_loss(y_tilde, y)

def mse(y_tilde, y):
    return np.mean((y_tilde - y) ** 2)

#Julian: Extra functions for metric calculations
def mae(y_tilde, y):
    return F.l1_loss(y_tilde, y)

def mae_numpy(y_tilde, y):
    return np.mean(np.abs(y_tilde - y))


def mape(y_tilde, y, eps=1e-8):
    return (torch.abs((y - y_tilde) / (y + eps))).mean() 


def mape_numpy(y_tilde, y, eps=1e-8):
    return np.mean(np.abs((y - y_tilde) / (y + eps)))


# [26] Qiu "Time-Varying Graph Signal Reconstruction" MSE + Temp
#  Giraldo  "Reconstuction of Time-Varying Graph Signals" MSE + Sob
#  [39]  Narang, "LOCALIZED ITERATIVE METHODS FOR INTERPOLATION IN GRAPH STRUCTURED DATA" MSE + High-Frequency


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G, D, good_data,noise = load_dataset(args.dataset, knn_param=args.knn,idx_d=args.idx_eps_var)

    # Move tensors to GPU
    if noise is not None:
        noise=torch.Tensor(noise).to(device)
    if good_data is not None:
        good_data = torch.Tensor(good_data).to(device)
    D = torch.Tensor(D).to(device)
    W = np.array(G['W'].toarray())
    
    # Create the Laplacian matrix and move to GPU
    L = np.diag(W@np.ones(W.shape[0])) - W
    L = torch.Tensor(L).to(device)

    
    # Create the mask
    # Sample Trajectories TODO: Add samplers
    train_set, test_set, mask, test_mask = gu.sampler(args.type_sampler,D, args.percentage, good_data,noise, seed=args.seed)
    #train_set, test_set, mask = gu.get_mask(D, args.percentage)

    # Move train_set and mask to GPU
    train_set = train_set.to(device)
    mask = mask.to(device)
    
    if  args.method in ['PrimalDual']:
        # Create the matrix X
        X = torch.randn_like(D, requires_grad=True)
        X = X.to(device)
        M=D.shape[1]
        smooth_avg,smooth_time=gu.smooth_per_time(X,L, args.epsilon,M,device,args.beta)
        smooth_avg=smooth_avg/smooth_time.shape[0]
        # Convertir a numpy si es tensor de PyTorch
        smooth_time_np = smooth_time.detach().cpu().numpy()
        smooth_avg_val = smooth_avg.item()  # convierte tensor escalar a float

        # Eje X: rango de tiempo
        tiempo = range(smooth_time_np.shape[0])

        # Graficar
        if args.dataset not in ['paramAWDall_var_ep','ultra_paramAWDall_var_ep']:
            dataset= args.dataset
        else:
            dataset= f'{args.dataset}_{args.idx_eps_var}'
        os.makedirs(f'analysis/{dataset}',exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(tiempo, smooth_time_np, label='Smooth por instante', linewidth=2)
        plt.axhline(y=smooth_avg_val, color='red', linestyle='--', label='Smooth promedio')

        plt.xlabel('Tiempo')
        plt.ylabel('Smooth')
        plt.title('Smooth por tiempo vs Promedio Global')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'analysis/{dataset}/Smooth_per_time_{dataset}.png')
        plt.show()
        # Create the optimizer
        optimizer = optim.Adam([X], lr=args.lr)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=False)

        # Get the loss function
        Dh = gm.create_Dh_torch(D.shape[1],device)
        Sobolev = gu.compute_sobolev_matrix(L, args.epsilon, args.beta)
        dual_func = lambda x: gm.get_Sobolev_smoothness_function(x, Dh, Sobolev, args.dual_function_type,device)
        pm_loss = gm.primal_dual_loss(X, D.shape[1], L, args.epsilon, args.beta, args.alpha, args.dual_step_mu, args.dual_step_lambdas, dual_func,device)

        ## Learning pipeline
        for e in range(args.epochs):
            
            # Zero the gradients before the forward pass
            optimizer.zero_grad()

            # Compute the loss
            loss,time_wise = pm_loss.primal(X, train_set, mask,True)
            
            # Backpropagate the loss
            loss.backward()

            # Update the parameters
            optimizer.step()

            acc_train = accuracy(X[mask], D[mask])
            rmse_train = torch.sqrt(acc_train)
            scheduler.step(rmse_train)

            if e % 100 == 0:
                acc_total = accuracy(X, D)
                acc_train = accuracy(X[mask], D[mask])
                acc_test = accuracy(X[test_mask], D[test_mask])

                print(f'Epoch {e}, Loss Total: {acc_total.item()}, Loss Train: {acc_train.item()}, Loss Test: {acc_test.item()}')
            
            if e % args.primal_steps == 0:
                # Update the lambdas
                with torch.no_grad():
                    pm_loss.dual(X, train_set, mask)


    # Compute the RMSE
    #os.makedirs(f'./Error_per_time/{args.dataset}',exist_ok=True)
    if args.dataset not in ['paramAWDall_var_ep','ultra_paramAWDall_var_ep']:
        dataset= args.dataset
    else:
        dataset= f'{args.dataset}_{args.idx_eps_var}'

    plt.figure(figsize=(10, 5))
    plt.plot(time_wise.detach().cpu().numpy())
    plt.title(f"Error per time  {args.dataset}")
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.xlabel("Time step")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig(f'./Error_per_time/Error_per_time_{dataset}.jpg')
    plt.show()

    


    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Smooth Graph Signal')
    parser.add_argument('--dataset', type=str, default='synthetic')
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--method', type=str, default='PrimalDual')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--percentage', type=float, default=0.5, help='Percentage of data to be used for training between 0 and 1')
    
    
    parser.add_argument('--coefficient_mse', type=float, default=0.5)
    parser.add_argument('--coefficient_lap', type=float, default=0.01)
    parser.add_argument('--coefficient_sob', type=float, default=0.01)
    parser.add_argument('--coefficient_temp', type=float, default=0.01)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.5)
    
    #Primal Dual Parameters
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dual_step_mu', type=float, default=0.01)
    parser.add_argument('--dual_step_lambdas', type=float, default=0.01)
    parser.add_argument('--primal_steps', type=int, default=10, help='Number of steps for the primal update in primal dual method')
    parser.add_argument('--dual_function_type', type=str, default='timewise', choices=['timewise', 'integral'], help='Type of dual function to use in primal dual method')
    
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--type_sampler', type=str, default='random', help='type_sampler')
    parser.add_argument('--idx_eps_var', type=int, default=0, choices=[0,1,2,3,4,5,6], help='Index of the list of different epsilon thresholds')
    
    args = parser.parse_args()
    
    main(args)