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

        