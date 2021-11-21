from operator import ne
import numpy as np
from numpy.core.numeric import identity
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from umap.umap_ import nearest_neighbors


def birch_clustering(feature, threshold=5.3):

    reducer = umap.UMAP(random_state=1, metric='manhattan')
    scaled_feature = StandardScaler().fit_transform(feature)
    embedding = reducer.fit_transform(scaled_feature)

    clustering = Birch(
        threshold=threshold,
        branching_factor=50,
        n_clusters=None,
        compute_labels=True
    ).fit(embedding)

    return clustering.labels_


def DBSCAN_clustering(feature, eps=3.1):

    reducer = umap.UMAP(random_state=1, metric='manhattan')
    scaled_feature = StandardScaler().fit_transform(feature)
    embedding = reducer.fit_transform(scaled_feature)

    clustering = DBSCAN(eps=eps).fit(embedding)

    return clustering.labels_


def mean_shift_clustering(feature, threshold=4.6):

    reducer = umap.UMAP(random_state=1, metric='manhattan')
    scaled_feature = StandardScaler().fit_transform(feature)
    embedding = reducer.fit_transform(scaled_feature)

    clustering = MeanShift(bandwidth=threshold).fit(embedding)

    return clustering.labels_


def get_nearest_neighbor(chosen_arr, labeled_arr):
    # The newest label. If we allow the user to choose who to compare, this should be passed to the function
    same_clust = labeled_arr[labeled_arr[:, -1] == chosen_arr[-1], :]
    same_clust = same_clust[same_clust[:, 2] != chosen_arr[2], :]

    if same_clust.size <= 5:
        return np.array(['No similar matches', '', '', '', ''])
    same_clust = same_clust[:5]
    nearest_neighbors = np.apply_along_axis(lambda x: 'cam ' + str(x[2]) + ' frame count ' + str(x[0]), 1, same_clust)
    return nearest_neighbors


def get_output(labeled_arr):
    misc_arr = labeled_arr[:, [0, 1, 2, -1]]
    # print(labeled_arr.shape)
    # additional_arr = np.apply_along_axis(func1d=get_nearest_neighbor, axis=1, arr=labeled_arr, labeled_arr=labeled_arr)

    additional_arr = [[]]
    for sample in labeled_arr:
        nearest_neighbors = get_nearest_neighbor(sample, labeled_arr)
        #print(nearest_neighbors, 5 - nearest_neighbors.shape[0])
        if nearest_neighbors.shape[0] < 5:
            nearest_neighbors = np.pad(nearest_neighbors, (0, 5 - nearest_neighbors.shape[0]), 'constant', constant_values=('', ''))
        if additional_arr == [[]]:
            additional_arr = [nearest_neighbors]
        else:
            additional_arr = np.concatenate((additional_arr, [nearest_neighbors]), axis=0)

    return np.concatenate((misc_arr, additional_arr), axis=1)
    # return additional_arr
