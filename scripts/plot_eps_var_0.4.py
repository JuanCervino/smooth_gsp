import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_tools.utils import summary_sampling

def plot_rmse_vs_eps_var(datasets, methods):
    sampling_type = "random"
    pr_sp = "0.4"
    RMSE_vs_eps_var = {}

    for method in methods:
        RMSE_vs_eps_var[method] = []
        for dataset in datasets:
            data = summary_sampling(dataset, method, sampling_type, pr_sp)
            RMSE_vs_eps_var[method].append(data[method]['RMSE'])
    
    return RMSE_vs_eps_var

datasets = [
    'paramAWDall_var_ep_0','paramAWDall_var_ep_1','paramAWDall_var_ep_2',
    'paramAWDall_var_ep_3','paramAWDall_var_ep_4','paramAWDall_var_ep_5','paramAWDall_var_ep_6',
    'ultra_paramAWDall_var_ep_0','ultra_paramAWDall_var_ep_1','ultra_paramAWDall_var_ep_2',
    'ultra_paramAWDall_var_ep_4','ultra_paramAWDall_var_ep_5','ultra_paramAWDall_var_ep_6'
]
epsilon_max = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15, 20, 25, 40, 50, 100]




methods = ['Sobolev','PrimalDual','nni','Tikhonov']#,'GraphRegularization']

# Define estilos personalizados
style_dict = {
    'Sobolev':               {'color': 'blue',        'linestyle': ':',   'marker': 'D', 'label': 'GraphTRSS'},
    'PrimalDual':            {'color': 'orange',      'linestyle': '-',   'marker': 'v', 'label': 'PrimalDual'},  # ← este queda igual
    'nni':                   {'color': 'black',       'linestyle': '-',   'marker': '*', 'label': 'NNI'},
    'Tikhonov':              {'color': 'green',       'linestyle': '-',   'marker': 's', 'label': 'Tikhonov'},
    'GraphRegularization':   {'color': 'deepskyblue', 'linestyle': '-',   'marker': 'X', 'label': 'GraphRegularization'},
}


RMSE_vs_eps_var = plot_rmse_vs_eps_var(datasets, methods)

# === Figura principal ===
fig, ax = plt.subplots(figsize=(10, 5))
for method in methods:
    rmse = RMSE_vs_eps_var[method]
    style = style_dict[method]
    ax.plot(epsilon_max, rmse, label=style["label"], color=style["color"],
            linestyle=style["linestyle"], marker=style["marker"])

#Crear inset en la esquina inferior derecha
 #axins = inset_axes(ax, width="40%", height="40%", loc="lower center",)
 #
 ## Graficar solo esas dos curvas en el inset
 #for method in ['Sobolev', 'PrimalDual']:  # o ['GraphTRSS', 'PrimalDual'] si ya renombraste
 #    rmse = RMSE_vs_eps_var[method]
 #    style = style_dict[method]
 #    axins.plot(epsilon_max, rmse, label=style["label"], color=style["color"],
 #               linestyle=style["linestyle"], marker=style["marker"])

#axins.set_yscale("log")
 #xins.set_xlim(40, 100)       # rango de epsilon más informativo
 #xins.set_ylim(5, 20)     # enfócate solo en el rango donde hay diferencia
 # Ocultar etiquetas de ticks (números), pero mantener ticks para que el grid funcione
 #xins.tick_params(
 #   labelleft=False,     # oculta números eje Y
 #   labelbottom=False,   # oculta números eje X
 #   left=True,           # deja líneas de ticks eje Y (para grid)
 #   bottom=True          # deja líneas de ticks eje X (para grid)
 #
 #
 #axins.grid(True, linestyle='--', alpha=0.3)

# Conecta con líneas
#mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1)

ax.set_xlabel("Maximum epsilon")
ax.set_ylabel("RMSE")
#ax.set_yscale("log")
ax.set_title("RMSE vs Maximum Epsilon")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig("RMSE_vs_MaxEpsilon_0.4_sinlog.pdf")
plt.show()

# === Plot secundario (todo) ===
plt.figure(figsize=(10, 5))
plt.yscale("log")
for method in methods:
    rmse = RMSE_vs_eps_var[method]
    style = style_dict[method]
    plt.plot(epsilon_max, rmse, label=style["label"], color=style["color"],
             linestyle=style["linestyle"], marker=style["marker"])

plt.xlabel("Maximum epsilon")
plt.ylabel("RMSE")
plt.title("RMSE vs Maximum Epsilon")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# === Plot con zoom en epsilons bajos ===
plt.figure(figsize=(10, 5))
plt.yscale("log")
for method in methods:
    rmse = RMSE_vs_eps_var[method]
    style = style_dict[method]
    plt.plot(epsilon_max[:6], rmse[:6], label=style["label"], color=style["color"],
             linestyle=style["linestyle"], marker=style["marker"])

plt.xlabel("Maximum epsilon")
plt.ylabel("RMSE")
plt.title("RMSE vs Maximum Epsilon")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("RMSE_vs_MaxEpsilon_0.4_Zoom.pdf")
plt.show()
