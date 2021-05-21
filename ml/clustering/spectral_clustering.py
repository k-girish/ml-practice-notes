from scipy.linalg import eigh
from k_means import k_means
from ml.graph import laplacian as graph_laplacian


def spectral_clustering(x, sigma=1, method='gaussian', n_eigen_vectors=2, n_clusters=3, k_means_iterations=10):
    # Construct graph laplacian
    laplacian, d_root_inv = graph_laplacian(x, sigma, method)

    # Find the k-smallest eigenvectors
    eig_vecs = eigh(laplacian, subset_by_index=[1, n_eigen_vectors])
    eig_vecs = d_root_inv.reshape(-1, 1) * eig_vecs

    # K-means
    _, c = k_means(eig_vecs, k=n_clusters, max_iter=k_means_iterations)

    return c
