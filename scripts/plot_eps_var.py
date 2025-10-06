import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_rmse_vs_eps_var(datasets,methods):

    RMSE_vs_eps_var = {}
    for method in methods:
        if not method in RMSE_vs_eps_var :
            RMSE_vs_eps_var[method] = []

            for dataset in datasets:

                with open (f'./results/{dataset}/Summary/summary_random.json', 'r') as f :
                    data = json.load(f)
                    
                RMSE_vs_eps_var[method].append(data[method]['RMSE'])
    
    return RMSE_vs_eps_var

datasets = ['paramAWDall_var_ep_0','paramAWDall_var_ep_1','paramAWDall_var_ep_2','paramAWDall_var_ep_3','paramAWDall_var_ep_4','paramAWDall_var_ep_5','paramAWDall_var_ep_6','ultra_paramAWDall_var_ep_0','ultra_paramAWDall_var_ep_1','ultra_paramAWDall_var_ep_2','ultra_paramAWDall_var_ep_4','ultra_paramAWDall_var_ep_5','ultra_paramAWDall_var_ep_6']
epsilon_max=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15, 20, 25, 40, 50, 100]

methods=['Sobolev','PrimalDual','nni','Tikhonov']

RMSE_vs_eps_var = plot_rmse_vs_eps_var(datasets,methods)

# Figura principal
fig, ax = plt.subplots(figsize=(10, 5))

# Curvas completas
for method in methods:
    rmse = RMSE_vs_eps_var[method]
    ax.plot(epsilon_max, rmse, marker='o', label=method)

ax.set_xlabel("Maximum epsilon")
ax.set_ylabel("RMSE")
ax.set_title("RMSE vs Maximum Epsilon")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

# 🔍 Lupa (inset) en la esquina
axins = inset_axes(ax, width="40%", height="40%", loc="upper left")  # tamaño y ubicación

# Curvas en la lupa (ε < 10 aprox)
for method in methods:
    rmse = RMSE_vs_eps_var[method]
    axins.plot(epsilon_max, rmse, marker='o', label=method)

# Limitar el zoom a la región de interés
axins.set_xlim(0, 10)        # rango de epsilon
axins.set_ylim(0, 4)         # rango de RMSE (ajusta si quieres)
axins.grid(True, linestyle='--', alpha=0.3)
axins.tick_params(labelsize=8)

# 🔗 Opcional: conectar visualmente con líneas
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1)

plt.tight_layout()
plt.savefig("RMSE vs Maximum Epsilon.pdf")
plt.show()

# Graficar RMSE vs epsilon_max
plt.figure(figsize=(10, 5))

for method in methods:
    rmse = RMSE_vs_eps_var[method]
    plt.plot(epsilon_max, rmse, marker='o', label=method)

plt.xlabel("Maximum epsilon")  
plt.ylabel("RMSE")             
plt.title("RMSE vs Maximum Epsilon") 
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Graficar RMSE vs epsilon_max
plt.figure(figsize=(10, 5))

for method in methods:
    rmse = RMSE_vs_eps_var[method]
    plt.plot(epsilon_max[:6], rmse[:6], marker='o', label=method)

plt.xlabel("Maximum epsilon")  
plt.ylabel("RMSE")             
plt.title("RMSE vs Maximum Epsilon") 
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()