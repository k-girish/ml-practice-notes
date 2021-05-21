import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import euclidean_distances


# Input is the data x and no. of centers k
# Output centroid for all the centers and index of cluster for each of the data
def k_means(x, k=3, max_iter=10, init_method='random', disable_tqdm=True):
    n, p = x.shape
    c = np.zeros(n)

    # Initialize centroids mu
    if init_method == 'farthest':
        mu = np.zeros((k, p))
        init_indices = []
        mu[0, :] = x[np.random.choice(n, 1), :].reshape(-1)
        for k_idx in range(1, k):
            dists = euclidean_distances(x, mu[:k_idx, :])
            dists = dists.sum(axis=1)
            max_idx = np.argmax(dists)
            init_indices.append(max_idx)
            mu[k_idx, :] = x[max_idx, :]

    else:
        init_indices = np.random.choice(n, k)
        mu = x[init_indices, :]

    # Iterate for some max-iterations
    for iter_idx in tqdm(range(max_iter), desc='k-means iterations', disable=disable_tqdm):

        # Update the cluster of all the points based on the closeness
        for x_idx in range(n):
            dists = np.linalg.norm(mu - x[x_idx, :].reshape(-1), axis=1)
            c[x_idx] = np.argmin(dists)

        # Update the centroids as the average of all the points within their clusters
        for mu_idx in range(k):
            temp = x[(c == mu_idx), :]
            if len(temp) > 0:
                mu[mu_idx, :] = np.mean(temp, axis=0)

    return mu, c
