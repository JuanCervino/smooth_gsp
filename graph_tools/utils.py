import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json


#Julian: This is a funtion to replicate the table 1 of the "Reconstruction of Time-Varying Graph Signals via Sobolev Smoothness"
def summary(dataset,sampling_type):
    #It lists the methods that has available results
    methods=os.listdir(f'./results/{dataset}')

    methods = [m for m in methods if m != "Summary" and m != "hyperparameter_search"]

    results = {method: {'RMSE':0 , 'MAE': 0 , 'MAPE': 0 } for method in methods}


    for method in methods:
        with open(f'./results/{dataset}/{method}/{sampling_type}.json','r') as f:
            data=json.load(f)

        results_seeds={'RMSE': [], 'MAE': [], 'MAPE': []}
        for seed in data['seeds']:
            results_percentage={'RMSE': [], 'MAE': [], 'MAPE': []}
            for result_per_percentage in data['seeds'][seed]['results_per_percentage']:
                current_results=data['seeds'][seed]['results_per_percentage'][result_per_percentage]

                results_percentage['RMSE'].append( float( current_results['RMSE'] ))
                results_percentage['MAE'].append( float( current_results['MAE']  ) )
                results_percentage['MAPE'].append( float( current_results['MAPE'] ) )
            
            results_seeds['RMSE'].append( np.mean (results_percentage['RMSE'] ) )
            results_seeds['MAE'].append( np.mean (results_percentage['MAE'] ) )
            results_seeds['MAPE'].append( np.mean (results_percentage['MAPE'] ) )
        
        results[method]['RMSE'] = np.mean(results_seeds['RMSE'])
        results[method]['MAE'] = np.mean(results_seeds['MAE'])
        results[method]['MAPE'] = np.mean(results_seeds['MAPE'])

    os.makedirs(f'./results/{dataset}/Summary',exist_ok=True)
    with open(f'./results/{dataset}/Summary/summary_{sampling_type}.json', 'w') as f:
        json.dump(results, f, indent=4)
            


def plot_rmse_vs_percentage(dataset, sampling_type):

    methods = [m for m in os.listdir(f'./results/{dataset}') if m != "Summary" and m != "hyperparameter_search"]

    results_percentage_all = {}

    for method in methods:
        path_json = f'./results/{dataset}/{method}/{sampling_type}.json'
        if not os.path.exists(path_json):
            continue

        with open(path_json, 'r') as f:
            data = json.load(f)

        percentage_keys = list(data['seeds'][next(iter(data['seeds']))]['results_per_percentage'].keys())
        percentage_keys = sorted(percentage_keys, key=lambda x: float(x))

        rmse_matrix = []
        for seed in data['seeds']:
            rmse_seed = []
            print(method,seed,percentage_keys)
            for perc in percentage_keys:
                rmse_seed.append(float(data['seeds'][seed]['results_per_percentage'][perc]['RMSE']))
            rmse_matrix.append(rmse_seed)

        # promedio por porcentaje en todas las semillas
        rmse_matrix = np.array(rmse_matrix)
        rmse_mean = np.mean(rmse_matrix, axis=0)

        results_percentage_all[method] = (percentage_keys, rmse_mean)

    # --------- PLOT ----------
    plt.figure(figsize=(10, 5))

    # estilos personalizados (puedes ampliar la lista si quieres)
    method_styles = {
        'TGSR':      {'color': 'red',    'linestyle': '-',   'marker': 'o'},
        'Sobolev':   {'color': 'blue',   'linestyle': ':',   'marker': 'D'},
        'nni':       {'color': 'black',  'linestyle': '-',   'marker': '*'},
        'GraphRegularization': {'color': 'deepskyblue', 'linestyle': '-', 'marker': 'X'},
        'Tikhonov':  {'color': 'green',  'linestyle': '-',   'marker': 's'},
    }
    default_colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    default_linestyles = ['-', '--', '-.', ':']
    default_markers = ['o', 's', '^', 'D', '*', 'x', '+']

    for i, method in enumerate(results_percentage_all):
        percentages, rmse_mean = results_percentage_all[method]

        style = method_styles.get(method, {
            'color': default_colors[i % len(default_colors)],
            'linestyle': default_linestyles[i % len(default_linestyles)],
            'marker': default_markers[i % len(default_markers)]
        })

        plt.plot(
            [float(p) for p in percentages], rmse_mean,
            color=style['color'],
            linestyle=style['linestyle'],
            marker=style['marker'],
            label=method,
            linewidth=2,
            markersize=8
        )

    if dataset in ['sea_surface_temperature', 'synthetic', 'weather','paramAWD_var_ep']:
        plt.yscale('log')
        y_min = min(np.min(v[1]) for v in results_percentage_all.values())
        y_max = max(np.max(v[1]) for v in results_percentage_all.values())
        y_min = 10**np.floor(np.log10(y_min))
        y_max = 10**np.ceil(np.log10(y_max))
        plt.yticks(np.logspace(np.log10(y_min), np.log10(y_max), num=5))
        plt.gca().yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))

    plt.xlabel("Sampling percentage")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs Sampling Percentage - {dataset}")
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(f'./results/{dataset}/Summary', exist_ok=True)
    plt.savefig(f'./results/{dataset}/Summary/RMSE_vs_percentage.png')
    plt.show()

            



def summary_results(dataset, methods):

    # Dictionary to store results for each method
    results = {method: {'RMSE': [], 'MAE': [], 'MAPE': []} for method in methods}
    
    # Collect results from all seeds
    seeds = [i for i in range(100)]  # Assuming seeds from 0 to 99
    for seed in seeds:
        for method in methods:
            result_path = f'results/{dataset}/{method}/results_{seed}.pt'
            if os.path.exists(result_path):
                results[method]['RMSE'].append(np.mean(torch.load(result_path)['RMSE'], axis=0).tolist())
                results[method]['MAE'].append(np.mean([x.cpu().detach().numpy() if hasattr(x, 'detach') and hasattr(x, 'numpy') and hasattr(x, 'cpu') else x for x in torch.load(result_path)['MAE']], axis=0).tolist())
                results[method]['MAPE'].append(np.mean([x.cpu().detach().numpy() if hasattr(x, 'detach') and hasattr(x, 'numpy') and hasattr(x, 'cpu') else x for x in torch.load(result_path)['MAPE']], axis=0).tolist())

    # Build a summary dictionary for all metrics
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in ['RMSE', 'MAE', 'MAPE']:
            if results[method][metric]:
                # Save the mean over all sampling densities
                summary[method][metric] = float(np.mean(results[method][metric]))
            else:
                summary[method][metric] = None

        # Save the summary as a JSON file
        summary_path = f'results/{dataset}/summary/summary.json'
        os.makedirs(f'results/{dataset}/summary', exist_ok=True)

        # Load existing summary if it exists
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                existing_summary = json.load(f)
        else:
            existing_summary = {}
        # Update with new methods
        for method in summary:
            existing_summary[method] = summary[method]
        # Save back
        with open(summary_path, 'w') as f:
            json.dump(existing_summary, f, indent=4)

#Julian: this function is to plot the RMSE vs Sampling density in Random sampling
def plot_results_from_seeds(dataset, methods, percentage_range=None):
    """
    Collect and plot results from different seeds for a specific dataset.
    
    Args:
        dataset (str): Name of the dataset
        methods (list): List of methods to plot
        percentage_range (list, optional): List of sampling percentages. If None, will use default range.
    """
    if percentage_range is None:
        percentage_range = [x / 10 for x in range(1, 10)]
    
    # Dictionary to store results for each method
    results_avg = {method: [] for method in methods}
    
    # Collect results from all seeds
    seeds = [i for i in range(100)]  # Assuming seeds from 0 to 99
    for seed in seeds:
        for method in methods:
            result_path = f'results/{dataset}/{method}/results_{seed}.pt'
            if os.path.exists(result_path):
                rmse_list = torch.load(result_path)
                rmse_list = rmse_list['RMSE']
                if isinstance(rmse_list, torch.Tensor):
                    rmse_list = rmse_list.cpu().numpy()
                # Ensure the array has the same length as percentage_range
                if len(rmse_list) == len(percentage_range):
                    results_avg[method].append(rmse_list)
    
    # Calculate average results per method
    for method in methods:
        if results_avg[method]:  # Check if we have results for this method
            # Stack arrays and take mean along first axis
            results_avg[method] = np.mean(np.stack(results_avg[method]), axis=0)
        else:
            results_avg[method] = np.array([])
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    # Dictionary of custom styles for known methods
    method_styles = {
        'TGSR':      {'color': 'red',    'linestyle': '-',   'marker': 'o'},
        'Sobolev':   {'color': 'blue','linestyle': ':','marker': 'D'},
        'nni':       {'color': 'black',  'linestyle': '-',   'marker': '*'},
        'GraphRegularization': {'color': 'deepskyblue', 'linestyle': '-', 'marker': 'X'},
        'Tikhonov': {'color': 'green',   'linestyle': '-',   'marker': 's'},
    }

    # Default styles for unlisted methods
    default_colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    default_linestyles = ['-', '--', '-.', ':']
    default_markers = ['o', 's', '^', 'D', '*', 'x', '+']

    for i, method in enumerate(methods):
        if len(results_avg[method]) > 0:  # Only plot if we have results
            style = method_styles.get(method, {
                'color': default_colors[i % len(default_colors)],
                'linestyle': default_linestyles[i % len(default_linestyles)],
                'marker': default_markers[i % len(default_markers)]
            })
            plt.plot(
                percentage_range, results_avg[method],
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                label=method,
                linewidth=2,
                markersize=8
            )
    
    # Set log scale for specific datasets
    if dataset in ['sea_surface_temperature', 'synthetic', 'weather']:
        plt.yscale('log')
        # Get the min and max values from all plotted data
        y_min = float('inf')
        y_max = float('-inf')
        for method in methods:
            if len(results_avg[method]) > 0:
                y_min = min(y_min, np.min(results_avg[method]))
                y_max = max(y_max, np.max(results_avg[method]))
        
        # Round to nearest power of 10
        y_min = 10**np.floor(np.log10(y_min))
        y_max = 10**np.ceil(np.log10(y_max))
        
        # Set yticks at powers of 10
        plt.yticks(np.logspace(np.log10(y_min), np.log10(y_max), num=5))
        plt.gca().yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
    #else:
    #    plt.yticks()
    #    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.xlabel('Sampling density')
    plt.ylabel('RMSE')
    plt.title(f'Random Sampling - {dataset}')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Save and show plot
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(f'results/plots/{dataset}_RMSE_from_seeds.png')
    plt.show()
    
    return results_avg
