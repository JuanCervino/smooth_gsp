import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import torch
from itertools import product
from graph_tools.load_dataset import load_dataset
import graph_tools.gradient_methods as gm
import graph_tools.graph_utils as gu
from torch import optim
import torch.nn.functional as F
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

@torch.no_grad()
def accuracy(y_tilde, y):
    """Calculate the mean squared error between predicted and actual values."""
    return F.mse_loss(y_tilde, y)

def mse(y_tilde, y):
    """Calculate the mean squared error using numpy."""
    return np.mean((y_tilde - y) ** 2)

def train_with_params(method, D, L, G, train_set, test_set, mask, test_mask, params, epochs=1000, lr=0.1, device='cuda'):
    """
    Train the model with a specific set of parameters and return the RMSE.
    
    Args:
        method (str): The method to use (Tikhonov, Sobolev, or GraphRegularization)
        D (torch.Tensor): The complete dataset
        L (torch.Tensor): The Laplacian matrix
        G (dict): Graph information
        train_set (torch.Tensor): Training data
        test_set (torch.Tensor): Test data
        mask (torch.Tensor): Mask for training data
        test_mask (torch.Tensor): Mask for test data
        params (dict): Dictionary containing hyperparameters
        epochs (int): Number of training epochs
        lr (float): Learning rate
        device (str): Device to use for computation ('cuda' or 'cpu')
    
    Returns:
        tuple: (RMSE value, dictionary of training metrics)
    """
    # Move all tensors to the specified device
    D = D.to(device)
    L = L.to(device)
    train_set = train_set.to(device)
    test_set = test_set.to(device)
    mask = mask.to(device)
    test_mask = test_mask.to(device)
    lr = params['lr']
    
    # Initialize random solution and optimizer
    X = torch.randn_like(D, requires_grad=True, device=device)
    optimizer = optim.Adam([X], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=False)
    
    # Get the loss function for the specified method
    if method!='PrimalDual':
        loss_fn = gm.get_loss(method, 
                            params['coefficient_mse'],
                            params['coefficient_lap'],
                            params['coefficient_temp'],
                            params['coefficient_sob'],
                            L,
                            params['epsilon'],
                            params['beta'],
                            D.shape[1],
                            device)
    else:
        dual_function_type='integral'
        Dh = gm.create_Dh_torch(D.shape[1],device)
        Sobolev = gu.compute_sobolev_matrix(L, params['epsilon'], params['beta'])
        dual_func = lambda x: gm.get_Sobolev_smoothness_function(x, Dh, Sobolev, dual_function_type,device)
        loss_fn = gm.primal_dual_loss(X, D.shape[1], L, params['epsilon'], params['beta'], params['alpha'], params['dual_step_mu'], params['dual_step_lambdas'],dual_func,device)

    
    final_lr = lr
    
    # Training loop
    for e in range(epochs):
        optimizer.zero_grad()

        if method!='PrimalDual':
            loss = loss_fn(X, train_set, mask)
        else:
            loss = loss_fn.primal(X, train_set, mask)

        loss.backward()
        optimizer.step()

        # Calculate training metrics
        acc_train = accuracy(X[mask], D[mask])
        rmse_train = torch.sqrt(acc_train)
        scheduler.step(rmse_train)
        
        final_lr = optimizer.param_groups[0]['lr']
        
        # Print progress every 100 epochs
        if e % 100 == 0:
            rmse_total = torch.sqrt(accuracy(X, D))
            rmse_test = torch.sqrt(accuracy(X[test_mask], D[test_mask]))
            print(f'Epoch {e}, Loss Total: {rmse_total.item()}, Loss Train: {rmse_train.item()}, Loss Test: {rmse_test.item()}, LR: {final_lr}')

        if method=='PrimalDual':
            if e % params['primal_steps'] == 0:
                # Update the lambdas
                with torch.no_grad():
                    loss_fn.dual(X, train_set, mask)

    # Calculate final RMSE on test set
    RMSE = torch.sqrt(accuracy(X[test_mask], D[test_mask])).cpu()
    return RMSE.item(), {
        'final_train_loss': torch.sqrt(accuracy(X[mask], D[mask])).item(),
        'final_test_loss': RMSE.item(),
        'final_total_loss': torch.sqrt(accuracy(X, D)).item(),
        'final_lr': final_lr
    }

def get_param_combinations(method, coefficient_range, epsilon_range,primal_dual_params, lr_range):
    """
    Generate all possible combinations of parameters for a given method.
    
    Args:
        method (str): The method to generate parameters for
        coefficient_range (list): Range of coefficient values to test
        epsilon_range (list): Range of epsilon values to test
    
    Returns:
        list: List of dictionaries containing parameter combinations
    """
    if method == 'Tikhonov':
        return [{'coefficient_mse': 0.5, 'coefficient_lap': c1, 'coefficient_temp': c2, 
                'coefficient_sob': 0, 'epsilon': 0, 'beta': 1.0, 'lr': lr} 
                for c1,c2, lr in product(coefficient_range, coefficient_range, lr_range)]
    elif method == 'Sobolev':
        return [{'coefficient_mse': 0.5, 'coefficient_lap': 0, 'coefficient_temp': 0, 
                'coefficient_sob': c, 'epsilon': e, 'beta': 1.0, 'lr': lr} 
                for c, e, lr in product(coefficient_range, epsilon_range, lr_range)]
    elif method == 'GraphRegularization':
        return [{'coefficient_mse': 0.5, 'coefficient_lap': c, 'coefficient_temp': 0, 
                'coefficient_sob': 0, 'epsilon': 0, 'beta': 1.0, 'lr': lr} 
                for c, lr in product(coefficient_range, lr_range)]
    elif method == 'PrimalDual':
        return [{'coefficient_mse': 0.5, 'coefficient_lap': 0, 'coefficient_temp': 0, 
                'coefficient_sob': 0, 'epsilon': e, 'beta': 1.0, 'alpha': a, 'dual_step_mu': d, 'dual_step_lambdas': l, 'primal_steps': p, 'lr': lr} 
                for e,a, d, l, p, lr in product(primal_dual_params['epsilon_range'],primal_dual_params['alpha_range'], primal_dual_params['dual_step_mu_range'], primal_dual_params['dual_step_lambdas_range'], primal_dual_params['primal_steps_range'], lr_range)]

    else:
        raise ValueError(f"Method {method} not supported for hyperparameter search")

def main(args):
    """
    Main function to perform hyperparameter search across different methods and sampling densities.
    
    Args:
        args: Command line arguments containing:
            - dataset: Name of the dataset to use
            - knn: Number of nearest neighbors for graph construction
            - epochs: Number of training epochs
            - lr: Learning rate
    """
    # Set up device and random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset and construct graph
    G, D, good_data, noise = load_dataset(args.dataset, knn_param=args.knn)

    if noise is not None:
        noise=torch.Tensor(noise).to(device)
    D = torch.Tensor(D).to(device)
    if good_data is not None:
        good_data = torch.Tensor(good_data).to(device)
    W = np.array(G['W'].toarray())
    L = np.diag(W@np.ones(W.shape[0])) - W
    L = torch.Tensor(L).to(device)

    # Define ranges for hyperparameter search
    coefficient_range = [1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 1e2, 2e2, 5e2,1e3,2e3,5e3,1e4,2e4,5e4,1e5,2e5,5e5,1e6,2e6,5e6]
    epsilon_range = [1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 1e2, 2e2, 5e2,1e3,2e3,5e3,1e4,2e4,5e4,1e5,2e5,5e5,1e6,2e6,5e6]
    if 'covid' not in args.dataset:
        lr_range=[0.01,0.1,1]#for covid 1,10,1000
    else:
        lr_range=[1,10,1000]#for covid 

    #Primal Dual Parameters
    primal_dual_params={
    'epsilon_range' : [1e-2, 2e-2, 5e-2, 1, 2, 5, 10, 20, 50, 1e2, 2e2, 5e2],
    'alpha_range' : [1e-07,1e-05,1e-03,0.01],
    'dual_step_mu_range' : [0.01, 0.1,1,5],
    'dual_step_lambdas_range' : [1e-15,1e-9,1e-4,0.01, 0.1, 1],
    'primal_steps_range' : [1,10, 100],
    }
    
    # Set sampling density range based on dataset
    if args.dataset == 'PM2_5_concentration':
        sampling_density_range = [i/100 for i in range(10,55,5)]
    elif args.dataset == 'covid_19_new_cases_global' or args.dataset == 'covid_19_new_cases_USA':
        sampling_density_range = [0.5, 0.6, 0.7, 0.8, 0.9, 0.995]
    else:
        sampling_density_range = [x / 10 for x in range(1, 10)]

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results/{args.dataset}/hyperparameter_search'
    os.makedirs(results_dir, exist_ok=True)

    # Initialize results dictionary
    results = {
        'dataset': args.dataset,
        'knn': args.knn,
        'epochs': args.epochs,
        'initial_lr': args.lr,
        'timestamp': timestamp,
        'methods': {}
    }

    # Perform hyperparameter search for each method
    for method in args.method:
        print(f"\n{'='*50}")
        print(f"Testing method: {method}")
        print(f"{'='*50}\n")

        # Get all parameter combinations for this method
        param_combinations = get_param_combinations(method, coefficient_range, epsilon_range,primal_dual_params, lr_range)
        
        best_avg_rmse = float('inf')
        best_params = None
        best_metrics_per_density = {}

        # Test each parameter combination
        for i, params in enumerate(param_combinations):
            # print(f"\nTesting combination {i+1}/{len(param_combinations)}")
            # print(f"Parameters: {params}")
            
            rmse_per_density = {}
            metrics_per_density = {}
            
            try:
                # Test parameters across all sampling densities
                for density in sampling_density_range:
                    # print(f"\nTesting density: {density}")
                    
                    # Get training and test masks for current density
                    train_set, test_set, mask, test_mask = gu.sampler('random',D, density, good_data,noise, seed=seed) #gu.get_mask(D, density, good_data, seed=seed)
                    
                    # Train model and get metrics
                    rmse, metrics = train_with_params(method, D, L, G, train_set, test_set, mask, test_mask, params, 
                                                    epochs=args.epochs, lr=args.lr, device=device)
                    
                    rmse_per_density[density] = rmse
                    metrics_per_density[density] = metrics
                
                # Calculate average RMSE across all densities
                avg_rmse = sum(rmse_per_density.values()) / len(rmse_per_density)
                
                # Update best parameters if current combination is better
                if avg_rmse < best_avg_rmse:
                    best_avg_rmse = avg_rmse
                    best_params = params
                    best_metrics_per_density = metrics_per_density
                    print(f"New best for {method}: avg RMSE = {best_avg_rmse}, params = {best_params}")
                    # Save to a temporary file
                    with open(f'{results_dir}/temp_best_params_{method}.txt', 'a') as temp_file:
                        temp_file.write(f"Method: {method}\nBest avg RMSE: {best_avg_rmse}\nBest params: {best_params}\n---\n")
                    
            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue

        # Save results for this method
        results['methods'][method] = {
            'best_parameters': best_params,
            'best_metrics_per_density': best_metrics_per_density,
            'best_avg_rmse': best_avg_rmse
        }

        print(f"\nBest parameters for {method}:")
        print(f"Parameters: {best_params}")
        print(f"Average RMSE: {best_avg_rmse}")
        print(f"Metrics per density: {best_metrics_per_density}")

        # Save intermediate results
        with open(f'{results_dir}/best_parameters_{method}.json', 'w') as f:
            json.dump(results, f, indent=4)

    print("\nHyperparameter search completed for all methods!")
    print(f"Results saved in: {results_dir}/best_parameters_{method}.json")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter Search for Graph Signal Processing')
    parser.add_argument('--dataset', type=str, default='sea_surface_temperature')
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--method', nargs='+', default='Sobolev')
    parser.add_argument('--lr', type=float, default=0.1)
    
    args = parser.parse_args()
    main(args) 