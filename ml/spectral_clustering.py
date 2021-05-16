import numpy as np
from tqdm.notebook import tqdm
from scipy.linalg import eigh
from k_means import k_means
from sklearn.metrics.pairwise import euclidean_distances


def find_graph_weight_matrix(x, sigma=0.5, distance_method='gaussian'):
    print('Creating the weight matrix for graph.')
    pairwise_dist = euclidean_distances(x, x)

    if distance_method == 'adjacency':
        weights = (pairwise_dist < sigma)
    else:
        weights = np.exp(-(pairwise_dist ** 2) / (2 * (sigma ** 2)))

    # Fill diagonal entries with 0
    np.fill_diagonal(weights, 0)

    print('Done with weight matrix for graph.')
    return weights


def find_graph_degree_matrix(w):
    return np.diag(w.sum(axis=1).reshape(-1))


def find_graph_laplacian(x, sigma=0.5, method='gaussian'):
    w = find_graph_weight_matrix(x, sigma, distance_method=method)
    return find_graph_degree_matrix(w) - w


def find_k_smallest_eigenvectors(arr, k=2):
    _, ans = eigh(arr, subset_by_index=[0, k - 1])
    return ans


def spectral_clustering(x, sigma=0.5, method='gaussian', n_eigen_vectors=2, n_clusters=3, k_means_iterations=10):
    # Construct graph laplacian
    print('Finding Graph Laplacian')
    laplacian = find_graph_laplacian(x, sigma, method)

    # Find the k-smallest eigenvectors
    print('Finding Eigenvectors')
    eig_vecs = find_k_smallest_eigenvectors(laplacian, n_eigen_vectors)
    print(eig_vecs.shape)

    print('Using K-means')
    _, c = k_means(eig_vecs, k=n_clusters, max_iter=k_means_iterations)

    return c