import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_tools.utils import summary,plot_rmse_vs_percentage

dataset='sea_surface_temperature'
sampling_type='random'

summary(dataset=dataset,sampling_type=sampling_type)
plot_rmse_vs_percentage(dataset=dataset,sampling_type=sampling_type)