from scipy.linalg import eigh
import numpy as np
from ml.graph import embedding as graph_embedding

def diffusion_maps_embedding(x, n_components=3, t=4, sigma=2):
    w = graph_embedding(x, sigma)

    d = w.sum(axis=1).reshape(-1)
    d_root = np.sqrt(d)
    d_root_inv = 1 / d_root

    # right multiply
    s = w * d_root_inv.reshape(1, -1)
    # left multiply
    s = w * d_root_inv.reshape(-1, 1)

    # find n_components largest eigen values of M
    eig_vals, eig_vecs_s = eigh(s, subset_by_index=[len(w) - n_components, n_components])
    eig_vecs_m = eig_vecs_s * d_root_inv.reshape(-1, 1)  # left multiply

    eig_vals_power_t = eig_vals ** t
    embedding = eig_vecs_m * eig_vals_power_t.reshape(1, -1)  # left multiply

    return embedding