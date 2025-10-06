import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from scipy.io import loadmat

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_tools.utils import summary_sampling

def get_signal_power_from_dataset(graph: str, idx: int = 0) -> float:
    """
    Carga un dataset por nombre y retorna su potencia promedio (sin ruido).
    Compatible con todos los casos de load_D.
    """
    if graph in ['paramAWD_var_ep', 'synthetic']:
        data = np.load(f'./datasets/{graph}.npz')
        D = data['Temp']

    elif graph in ['paramAWDall_var_ep', 'ultra_paramAWDall_var_ep']:
        data = np.load(f'./datasets/{graph}.npz')
        D = data['Tempall'][:, :, idx]

    elif graph in ['sea_surface_temperature', 'covid_19_new_cases_global', 'covid_19_new_cases_USA']:
        data = loadmat(f'./datasets/{graph}.mat')
        D = data['Data']
        if graph == 'sea_surface_temperature':
            D = D[:, :600]

    elif graph == 'PM2_5_concentration':
        data = loadmat(f'./datasets/{graph}.mat')
        D = data['myDataPM'][:, :220]

    else:
        raise ValueError(f"Unsupported dataset: {graph}")

    return float(np.mean(D ** 2))

def plot_snr_out_vs_eps_var(datasets, methods):
    sampling_type = "random"
    pr_sp = "0.4"
    SNR_vs_eps_var = {}

    for method in methods:
        SNR_vs_eps_var[method] = []
        for dataset in datasets:
            data = summary_sampling(dataset, method, sampling_type, pr_sp)
            rmse = data[method]['RMSE']
            signal_power = get_signal_power_from_dataset('PM2_5_concentration')
            snr_out = 10 * np.log10(signal_power / (rmse ** 2))
            SNR_vs_eps_var[method].append(snr_out)

    return SNR_vs_eps_var



#datasets = [
#    'paramAWDall_var_ep_0','paramAWDall_var_ep_1','paramAWDall_var_ep_2',
#    'paramAWDall_var_ep_3','paramAWDall_var_ep_4','paramAWDall_var_ep_5','paramAWDall_var_ep_6',
#    'ultra_paramAWDall_var_ep_0','ultra_paramAWDall_var_ep_1','ultra_paramAWDall_var_ep_2',
#    'ultra_paramAWDall_var_ep_4','ultra_paramAWDall_var_ep_5','ultra_paramAWDall_var_ep_6'
#]
#epsilon_max = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15, 20, 25, 40, 50, 100]

datasets = ['sea_surface_temperature_10.0','sea_surface_temperature_20.0','sea_surface_temperature_30.0','sea_surface_temperature_40.0','sea_surface_temperature_50.0'
            ,'sea_surface_temperature_60.0','sea_surface_temperature_70.0']

noise = [10.0,20.0,30.0,40.0,50.0,60.0,70.0]


methods = ['Sobolev','PrimalDual','nni','Tikhonov']#,'GraphRegularization']

# Define estilos personalizados
style_dict = {
    'Sobolev':               {'color': 'blue',        'linestyle': ':',   'marker': 'D', 'label': 'GraphTRSS'},
    'PrimalDual':            {'color': 'orange',      'linestyle': '-',   'marker': 'v', 'label': 'PrimalDual'},  # ← este queda igual
    'nni':                   {'color': 'black',       'linestyle': '-',   'marker': '*', 'label': 'NNI'},
    'Tikhonov':              {'color': 'green',       'linestyle': '-',   'marker': 's', 'label': 'Tikhonov'},
    'GraphRegularization':   {'color': 'deepskyblue', 'linestyle': '-',   'marker': 'X', 'label': 'GraphRegularization'},
}


SNR_vs_eps_var = plot_snr_out_vs_eps_var(datasets, methods)

# === Figura principal ===
fig, ax = plt.subplots(figsize=(10, 5))
for method in methods:
    rmse = SNR_vs_eps_var[method]
    style = style_dict[method]
    ax.plot(noise, rmse, label=style["label"], color=style["color"],
            linestyle=style["linestyle"], marker=style["marker"])

##Crear inset en la esquina inferior derecha
#axins = inset_axes(ax, width="40%", height="40%", loc="lower center",)
#
## Graficar solo esas dos curvas en el inset
#for method in ['Sobolev', 'PrimalDual']:  # o ['GraphTRSS', 'PrimalDual'] si ya renombraste
#    rmse = SNR_vs_eps_var[method]
#    style = style_dict[method]
#    axins.plot(epsilon_max, rmse, label=style["label"], color=style["color"],
#               linestyle=style["linestyle"], marker=style["marker"])
#
##axins.set_yscale("log")
#axins.set_xlim(40, 100)       # rango de epsilon más informativo
#axins.set_ylim(5, 20)     # enfócate solo en el rango donde hay diferencia
## Ocultar etiquetas de ticks (números), pero mantener ticks para que el grid funcione
#axins.tick_params(
#    labelleft=False,     # oculta números eje Y
#    labelbottom=False,   # oculta números eje X
#    left=True,           # deja líneas de ticks eje Y (para grid)
#    bottom=True          # deja líneas de ticks eje X (para grid)
#)
#
#axins.grid(True, linestyle='--', alpha=0.3)

# Conecta con líneas
#mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1)

ax.set_xlabel("Input SNR (dB)", fontsize=35)
ax.set_ylabel("Out SNR (dB)", fontsize=35)
#ax.set_yscale("log")
#ax.set_title("SNR Variation")
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', labelsize=29)
ax.legend(fontsize=26,framealpha=0.0)
plt.tight_layout()
plt.savefig("SNR_Variation_sea_surface_temperature_.pdf")
plt.show()
