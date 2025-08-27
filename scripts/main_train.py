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

    G, D, good_data = load_dataset(args.dataset, knn_param=args.knn)

    # Move tensors to GPU
    if good_data is not None:
        good_data = torch.Tensor(good_data).to(device)
    D = torch.Tensor(D).to(device)
    W = np.array(G['W'].toarray())
    
    # Create the Laplacian matrix and move to GPU
    L = np.diag(W@np.ones(W.shape[0])) - W
    L = torch.Tensor(L).to(device)

    
    # Create the mask
    # Sample Trajectories TODO: Add samplers
    train_set, test_set, mask, test_mask = gu.sampler(args.type_sampler,D, args.percentage, good_data, seed=args.seed)
    #train_set, test_set, mask = gu.get_mask(D, args.percentage)

    # Move train_set and mask to GPU
    train_set = train_set.to(device)
    mask = mask.to(device)
    
    if args.method == 'nni':
        x_recon = gu.solver_NNI_nan(mask.cpu().numpy(), train_set.cpu().numpy(), G['coords'])
        print(x_recon)
        print(test_set.cpu().numpy())

        print(D.cpu().numpy())
        
        print("Total", mse(x_recon, D.cpu().numpy()))
        print("Train", mse(x_recon[mask.cpu().numpy()], D[mask].cpu().numpy()))

        D_cpu = D.cpu()

        print("Test", mse(x_recon[test_mask.cpu().numpy()], D_cpu[test_mask.cpu()].numpy()))

    elif args.method in ['MSE', 'Tikhonov', 'Sobolev', 'GraphRegularization', 'Temporal', 'All']:
        
        # Create the matrix X
        X = torch.randn_like(D, requires_grad=True).to(device)

        # Create the optimizer
        optimizer = optim.Adam([X], lr=args.lr)
        
        #Julian: We added the scheduler because in the original paper Jhony used an algorithm to adapt the lr
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=False)
        
        # Get the loss function
        loss_fn = gm.get_loss(args.method, args.coefficient_mse, args.coefficient_lap, args.coefficient_temp, args.coefficient_sob, L, args.epsilon, args.beta, D.shape[1],device)
        
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

            acc_train = accuracy(X[mask], D[mask])
            rmse_train = torch.sqrt(acc_train)
            scheduler.step(rmse_train)

            if e % 100 == 0:
                acc_total = accuracy(X, D)
                acc_train = accuracy(X[mask], D[mask])
                acc_test = accuracy(X[test_mask], D[test_mask])
                print(f'Epoch {e}, Loss Total: {acc_total.item()}, Loss Train: {acc_train.item()}, Loss Test: {acc_test.item()}')
                
            # print(f'Gradient: {X.grad}')
        

    elif args.method in ['PrimalDual']:
        # Create the matrix X
        X = torch.randn_like(D, requires_grad=True)
        X = X.to(device)
        
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
            loss = pm_loss.primal(X, train_set, mask)
            
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
    if args.method == 'nni':
        MSE=mse(x_recon[test_mask.cpu().numpy()], D[test_mask].cpu().numpy())
        RMSE = np.sqrt(MSE)
        MAPE=mape_numpy(x_recon[test_mask.cpu().numpy()], D[test_mask].cpu().numpy())
        MAE=mae_numpy(x_recon[test_mask.cpu().numpy()], D[test_mask].cpu().numpy())
    else:
        MSE=accuracy(X[test_mask], D[test_mask])
        MAPE=mape(X[test_mask], D[test_mask])
        MAE=mae(X[test_mask], D[test_mask])
        RMSE = torch.sqrt(MSE).cpu()

    results={'RMSE':RMSE.item(),'MAPE':MAPE.item(),'MAE':MAE.item()}
    # Save the results to a JSON file.
    # The JSON stores results for a specific dataset and a single seed.
    # It contains entries for different percentage values, where each percentage
    # maps to its evaluation metrics:
    #    {
    #  "seeds": {
    #    "42": {
    #      "results_per_percentage": {
    #        "0.1": { "RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0 },
    #        "0.2": { "RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0 }
    #      }
    #    },
    #    "10": {
    #      "results_per_percentage": {
    #        "0.1": { "RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0 },
    #        "0.2": { "RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0 }
    #      }
    #    }
    #  }
    #}

    # Create the results folder for the given dataset and method
    os.makedirs(f'results/{args.dataset}/{args.method}', exist_ok=True)

    # Path to the JSON file for this sampler
    json_output_path = os.path.join(
        f'results/{args.dataset}/{args.method}', 
        f"{args.type_sampler}.json"
    )

    # Load existing data if the file exists, otherwise create an empty structure
    if os.path.exists(json_output_path):
        with open(json_output_path, "r") as f:
            data = json.load(f)
    else:
        data = {"seeds": {}}

    # Convert seed and percentage to strings (keys in JSON must be strings)
    seed_key = str(args.seed)
    percentage_key = str(args.percentage)

    # Ensure that the current seed entry exists
    if seed_key not in data["seeds"]:
        data["seeds"][seed_key] = {"results_per_percentage": {}}

    # Add results for this percentage if it does not already exist
    if percentage_key not in data["seeds"][seed_key]["results_per_percentage"]:
        data["seeds"][seed_key]["results_per_percentage"][percentage_key] = results
        with open(json_output_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Added results for seed={args.seed} and percentage={args.percentage}")
    else:
        print(f"Results for seed={args.seed} and percentage={args.percentage} already exist, skipping.")

    


    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Smooth Graph Signal')
    parser.add_argument('--dataset', type=str, default='sea_surface_temperature')
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--method', type=str, default='all')
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
    parser.add_argument('--dual_function_type', type=str, default='integral', choices=['timewise', 'integral'], help='Type of dual function to use in primal dual method')
    
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--type_sampler', type=str, default='random', help='type_sampler')
    args = parser.parse_args()
    
    main(args)