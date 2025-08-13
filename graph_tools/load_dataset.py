import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
import netCDF4
import matplotlib.pyplot as plt


##This function automatically selects the dataset and performs the preprocessing
def load_D(graph,data):
    good_data = None
    if graph == 'sea_surface_temperature' or graph == 'covid_19_new_cases_global' or graph == 'covid_19_new_cases_USA':
        D=data['Data']
        if graph == 'sea_surface_temperature':
            D=D[:,:600]
    elif graph == 'synthetic':
        D=data['D']
    elif graph == 'PM2_5_concentration':
        D=data['myDataPM']
        D=D[:,:220]
        good_data = D>0
    return D,good_data

def load_dataset(graph, knn_param=10):
    
        # From
        # https://github.com/jhonygiraldo/GraphTRSS/blob/main/sea_surface_temperature_experiment/graph_construction/graph_construction.m
        # Load .mat file
        if graph in ['sea_surface_temperature', 'covid_19_new_cases_global', 'covid_19_new_cases_USA', 'PM2_5_concentration','synthetic']:
            data = loadmat(f'./datasets/{graph}.mat')
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


            D,good_data = load_D(graph,data)
            return G, D,good_data
        
        elif graph == 'weather':
            # Load the file
            folder = './datasets/run_trunc_6' 

            fp='output.nc'
            fp = folder + '/' + fp
            nc = netCDF4.Dataset(fp)
            data = nc.variables['vor'][:,0,:,:]  # Time, Lat, Lon
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            
            # Debug: Print longitude range
            print(lat.shape, lon.shape, data.shape)
            G = create_grid_graph(lat, lon)
            
            # Simple flatten: reshape to (time, N) where N = lat * lon
            data_flat = data.reshape(data.shape[0], -1)
            # Swap time and graph dimensions
            data_flat = data_flat.T
    
            # Check for dimension mismatch
            if data_flat.shape[1] != G['N']:
                print(f"WARNING: Data has {data_flat.shape[1]} nodes but graph has {G['N']} nodes!")
                print("This will cause matrix multiplication errors.")
            
            return G, data_flat,None
        
        else:
            raise ValueError(f"Dataset {graph} not found")

def create_grid_graph(lats, lons, periodic_lon=True, periodic_lat=False):
    """
    Create a graph from a grid of latitude and longitude points.
    Returns a sparse adjacency matrix and coordinates.
    
    Args:
        lats: Array of latitude values
        lons: Array of longitude values
        periodic_lon: Whether to connect the last longitude to first (across date line)
        periodic_lat: Whether to connect the last latitude to first (across poles)
    
    Returns:
        G: Dictionary containing:
            - W: Sparse adjacency matrix (CSR format)
            - coords: Array of shape (N, 2) with lat-lon coordinates
            - N: Number of nodes
            - type: String identifier
    """
    n_lats = len(lats)
    n_lons = len(lons)
    N = n_lats * n_lons
    
    # Create sparse adjacency matrix
    W = lil_matrix((N, N))
    
    # Create coordinates array
    coords = np.zeros((N, 2))
    
    # Helper function to get node index
    def get_node_idx(i, j):
        return i * n_lons + j
    
    # Fill coordinates array
    for i in range(n_lats):
        for j in range(n_lons):
            node_idx = get_node_idx(i, j)
            coords[node_idx] = [lats[i], lons[j]]
    
    # Connect points along same latitude (longitude lines)
    for i in range(n_lats):
        for j in range(n_lons - 1):
            node1 = get_node_idx(i, j)
            node2 = get_node_idx(i, j + 1)
            W[node1, node2] = 1
            W[node2, node1] = 1
        
        # Connect last longitude to first if periodic
        if periodic_lon and n_lons > 1:
            node1 = get_node_idx(i, n_lons - 1)
            node2 = get_node_idx(i, 0)
            W[node1, node2] = 1
            W[node2, node1] = 1
    
    # Connect points along same longitude (latitude lines)
    for j in range(n_lons):
        for i in range(n_lats - 1):
            node1 = get_node_idx(i, j)
            node2 = get_node_idx(i + 1, j)
            W[node1, node2] = 1
            W[node2, node1] = 1
        
        # Connect last latitude to first if periodic
        if periodic_lat and n_lats > 1:
            node1 = get_node_idx(n_lats - 1, j)
            node2 = get_node_idx(0, j)
            W[node1, node2] = 1
            W[node2, node1] = 1
    
    # Construct graph dictionary
    G = {
        'N': N,
        'W': W.tocsr(),  # Convert to CSR format for efficient operations
        'coords': coords,
        'type': 'grid',
        'shape': (n_lats, n_lons)  # Store grid shape for reshaping data
    }
    
    return G
