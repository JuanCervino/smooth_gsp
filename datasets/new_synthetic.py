# ========================================================================
# Time-varying Graph Signal Reconstruction,
#
# Copyright(c) 2017 Kai Qiu and Yuantao Gu
# All Rights Reserved.
# ----------------------------------------------------------------------
# 100 points are generated randomly.
# The dimension of the synthetic dataset is 100x600.
#
# Version 1.0
# Written by Kai Qiu (q1987k@163.com)
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


seed=42
# %% graph construction
N = 100
Position = np.zeros((N, 2), dtype=float)

# selected points on an N×N grid (randperm without replacement)
point_select = np.random.choice(N * N, size=N, replace=False) + 1  

for i_point in range(N):
    Position[i_point, 0] = (point_select[i_point] - 1) % N + 1  # x coordinate
    Position[i_point, 1] = (point_select[i_point] - 1) // N + 1  # y coordinate

# distances between each two points
Dist = np.zeros((N, N), dtype=float)
for i in range(N):
    for j in range(i + 1, N):
        Dist[i, j] = np.linalg.norm(Position[i, :] - Position[j, :])
        Dist[j, i] = Dist[i, j]

A = np.zeros((N, N), dtype=float)  # Adjacency matrix
W = np.zeros((N, N), dtype=float)  # Weighted matrix
k = 5  # k-NN method

for i in range(N):
    ind = np.argsort(Dist[i, :])  # ascending
    nn = ind[1:k + 1]             # skip self (at position 0)
    A[i, nn] = 1
    A[nn, i] = 1
    W[i, nn] = 1.0 / (Dist[i, nn] ** 2)
    W[nn, i] = W[i, nn]

W = W / np.max(W)
D = np.diag(np.sum(W, axis=1))
L = D - W

# %% plot the graph
# approximate linspecer(7) via a categorical colormap
C = plt.get_cmap('tab10')(np.arange(7))

plt.figure()
plt.plot(
    Position[:, 0],
    Position[:, 1],
    'o',
    color=C[3],
    markerfacecolor=C[3],
    markersize=4
)

ki, kj = np.nonzero(A)
for p in range(len(ki)):
    plt.plot(
        [Position[ki[p], 0], Position[kj[p], 0]],
        [Position[ki[p], 1], Position[kj[p], 1]],
        linewidth=1,
        color=C[1]
    )

# %% generate the time-varying graph signal
# eigen-decomposition of L (symmetric)
lam, V = np.linalg.eigh(L)
lam[0] = 0.0
lambdaHalfInv = 1.0 / np.sqrt(lam, where=lam > 0)
lambdaHalfInv[0] = 0.0
LHalfInv = V @ np.diag(lambdaHalfInv) @ V.T

T = 600
Temp = np.zeros((N, T), dtype=float)

ftmp = V.T @ np.random.randn(N)  # spectral coeffs ~ N(0,1)
ftmp[10:] = ftmp[10:] / 100.0    
ftmp = V @ ftmp
Temp[:, 0] = ftmp / np.linalg.norm(ftmp) * 100.0  # ||x1||2 = 100

###
rng = np.random.default_rng(seed)
ep_max = 1.0
epsilon = rng.uniform(1e-12, ep_max, size=T-1)
#eps = np.concatenate([eps, [ep_max]])
#epsilon = 1.0  # smoothness level
###
for k_,eps in zip( range(1, T), epsilon ):
    f = np.random.randn(N)
    f = f / np.linalg.norm(f) * eps
    fdc = np.random.randn() * 0.0 * np.ones(N)  # DC component (zeroed)
    Temp[:, k_] = Temp[:, k_ - 1] + LHalfInv @ f + fdc

# %% sampling matrix
SampleNum = int(np.floor(N * 0.4))  # the number of sampled points at each time
SampleMatrix = np.zeros((N, T), dtype=float)
for i in range(T):
    idx = np.random.choice(N, size=SampleNum, replace=False)
    SampleMatrix[idx, i] = 1.0

noise = 0.1 * np.random.randn(*Temp.shape)  # measurement noise


np.savez(
    './datasets/paramAWD_var_ep.npz',
    Position=Position,
    A=A,
    W=W,
    D=D,
    L=L,
    Temp=Temp,
    noise=noise,
    SampleMatrix=SampleMatrix
)

# %% generate data for Expreiment_A_5b
Tempall = np.zeros((N, T, 7), dtype=float)
Tempall[:, :, 3] = Temp  


lam2, V2 = np.linalg.eigh(L)
lam2[0] = 0.0
lambdaHalfInv2 = 1.0 / np.sqrt(lam2, where=lam2 > 0)
lambdaHalfInv2[0] = 0.0
LHalfInv2 = V2 @ np.diag(lambdaHalfInv2) @ V2.T

epsilon_set = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])


for i_epsilon in [0, 1, 2, 4, 5, 6]:
    epsilon = epsilon_set[i_epsilon]
    Tempall[:, 0, i_epsilon] = Tempall[:, 0, 3]  # same x1

    epsilon = rng.uniform(1e-12, epsilon, size=T-1)

    for k_,eps in zip(range(1, T),epsilon):
        f = np.random.randn(N)
        f = f / np.linalg.norm(f) * eps
        fdc = np.random.randn() * 0.0 * np.ones(N)  # DC component (zeroed)
        Tempall[:, k_, i_epsilon] = Tempall[:, k_ - 1, i_epsilon] + LHalfInv2 @ f + fdc
        


np.savez(
    './datasets/paramAWDall_var_ep.npz',
    Position=Position,
    A=A,
    W=W,
    D=D,
    L=L,
    Tempall=Tempall,
    noise=noise
)

plt.show()
