from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def embedding(x, sigma=1, distance_method='gaussian'):
    pairwise_dist = euclidean_distances(x, x)

    if distance_method == 'adjacency':
        return (pairwise_dist < sigma).astype('int')
    else:
        weights = np.exp(-(pairwise_dist ** 2) / (2 * (sigma ** 2)))
        # weights[weights<sigma] = 0

        return weights


def laplacian(x, sigma=1, method='gaussian'):
    w = embedding(x, sigma, distance_method=method)

    d = w.sum(axis=1).reshape(-1)
    d_root_inv = 1 / np.sqrt(d)

    # Sum product over elemetns in columns of w and d_root_inv
    n_w = w * d_root_inv.reshape(1, -1)

    return np.eye(len(w)) - n_w, d_root_inv
