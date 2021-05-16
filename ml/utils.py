import numpy as np


# Given a list of cluster indices for the data points, find the best matching label corresponding to the cluster index
def find_cluster_label_dict(c_indices, labels):
    cluster_label_dict = {}
    for c_idx in np.unique(c_indices):
        temp = (labels[c_indices == c_idx])
        temp = temp.astype('int')
        temp = np.bincount(temp)
        cluster_label_dict[c_idx] = np.argmax(temp)
    return cluster_label_dict


# Find the predicted label for data points corresponding to cluster indices
def find_cluster_labels(c_indices, labels):
    c_indices = c_indices.astype('int')
    cluster_label_dict = find_cluster_label_dict(c_indices, labels)
    label_pred = []
    for c_idx in c_indices:
        label_pred.append(cluster_label_dict[c_idx])
    return label_pred
