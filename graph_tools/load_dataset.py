import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors



def load_dataset(graph, knn_param=10):
    
    if graph == 'sea_surface_temperature':
        # From
        # https://github.com/jhonygiraldo/GraphTRSS/blob/main/sea_surface_temperature_experiment/graph_construction/graph_construction.m
        # Load .mat file
        data = loadmat('./datasets/sea_surface_temperature.mat')
        points = data['Position']

        N = points.shape[0]
        # Find k-nearest neighbors (k+1 because the first neighbor is the point itself)
        nbrs = NearestNeighbors(n_neighbors=knn_param + 1, algorithm='auto').fit(points)
        Dist, Idx = nbrs.kneighbors(points)

        # Compute sigma
        sigma = np.mean(Dist)

        # Initialize sparse weight matrix
        W = lil_matrix((N, N))

        # Fill W using Gaussian kernel (symmetric)
        for i in range(N):
            for j in range(1, knn_param + 1):  # skip the first one (self)
                dist_ij = Dist[i, j]
                neighbor_idx = Idx[i, j]
                weight = np.exp(-(dist_ij ** 2) / (sigma ** 2))
                W[i, neighbor_idx] = weight
                W[neighbor_idx, i] = weight

        # Construct graph dictionary (like MATLAB struct)
        G = {
            'N': N,
            'W': W.tocsr(),  # store sparse matrix efficiently
            'coords': points,
            'type': 'nearest neighbors',
            'sigma': sigma
        }
        
        D = data['Data']
        return G, D
    else:
        raise ValueError("Unsupported graph type")
        pass