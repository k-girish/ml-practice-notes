import numpy as np


# Input is the data x and no. of centers k
# Output centroid for all the centers and index of cluster for each of the data
def k_means(x, k=3, max_iter=10, init_method='random'):
    n, p = x.shape
    c = np.zeros(n)

    # Initialize centroids mu
    # k random indices from uniform of range n
    # TODO: include the init methods
    mu_indices = np.random.choice(n, k)
    mu = x[mu_indices, :]

    # Iterate for some max-iterations
    for iter_idx in range(max_iter):

        # Update the cluster of all the points based on the closeness
        for x_idx in range(n):
            dists = np.linalg.norm(mu - x[x_idx, :].reshape(-1), axis=1)
            c[x_idx] = np.argmin(dists)

        # Update the centroids as the average of all the points within their clusters
        for mu_idx in range(k):
            mu[mu_idx, :] = np.average(x[(c == mu_idx), :], axis=0)

    return mu, c