from operator import ne
import numpy as np
from numpy.core.numeric import identity
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from umap.umap_ import nearest_neighbors
import os
import shutil


def birch_clustering(feature, threshold=5.3):
    '''
    Function to perform Birch clustering
    Performance may be poor when the number of features and/or samples increases
    Predetermined threshold = 5.3
    feature: The numpy array contain all features of samples to be clustered
    Output: The label of input array
    '''
    reducer = umap.UMAP(random_state=1, metric='manhattan')
    scaled_feature = StandardScaler().fit_transform(feature)
    embedding = reducer.fit_transform(feature)

    clustering = Birch(
        threshold=threshold,
        branching_factor=50,
        n_clusters=None,
        compute_labels=True
    ).fit(embedding)

    return clustering.labels_


def DBSCAN_clustering(feature, eps=3.1):
    '''
    Function to perform DBSCAN clustering
    Predetermined epsilon = 3.1
    feature: The numpy array contain all features of samples to be clustered
    Output: The label of input array
    '''
    reducer = umap.UMAP(random_state=1, metric='manhattan')
    #scaled_feature = StandardScaler().fit_transform(feature)
    embedding = reducer.fit_transform(feature)

    clustering = DBSCAN(eps=eps).fit(embedding)

    return clustering.labels_


def mean_shift_clustering(feature, threshold=4.6):
    '''
    Function to perform Mean Shift clustering
    Predetermined threshold = 4.6
    feature: The numpy array contain all features of samples to be clustered
    Output: The label of input array
    '''
    reducer = umap.UMAP(random_state=1, metric='manhattan')
    scaled_feature = StandardScaler().fit_transform(feature)
    embedding = reducer.fit_transform(feature)

    clustering = MeanShift(bandwidth=threshold).fit(embedding)

    return clustering.labels_


def get_nearest_neighbor(chosen_arr, labeled_arr):
    '''
    Function to get the 5 neighbors with the oldest time recorded of a chosen array
    chosen_arr: The array to be compared
    labeled_arr: The labled dataset
    Output: An array contains 5 neighbors of the chosen point 
    '''
    same_clust = labeled_arr[labeled_arr[:, -1] == chosen_arr[-1], :]
    same_clust = same_clust[same_clust[:, 2] != chosen_arr[2], :]

    if same_clust.size <= 5:
        # In case there is no match
        return np.array(['No similar matches', '', '', '', ''])
    same_clust = same_clust[np.argsort(same_clust[:, 4])]
    same_clust = same_clust[:5]
    #print(same_clust[:, 0])
    nearest_neighbors = []
    for sample in same_clust:
        nearest_neighbors = np.append(nearest_neighbors, 'cam ' + str(sample[2]) + ' frame count ' + str(sample[0]))
    # print(nearest_neighbors)
    return nearest_neighbors


def get_output(labeled_arr):
    '''
    Function to get output array to be saved to csv
    labeled_arr: The labeled dataset
    Output: A numpy array with all the samples and their framecount - file name - cam id - label (person id) - 5 matches if possible, 
    '''
    misc_arr = labeled_arr[:, [0, 1, 2, 3, -1]]

    additional_arr = [[]]
    for sample in labeled_arr:
        nearest_neighbors = get_nearest_neighbor(sample, labeled_arr)
        # print(nearest_neighbors)
        # print()
        #print(nearest_neighbors, 5 - nearest_neighbors.shape[0])
        if nearest_neighbors.shape[0] < 5:
            nearest_neighbors = np.pad(nearest_neighbors, (0, 5 - nearest_neighbors.shape[0]), 'constant', constant_values=('', ''))
        if additional_arr == [[]]:
            additional_arr = [nearest_neighbors]
        else:
            additional_arr = np.concatenate((additional_arr, [nearest_neighbors]), axis=0)
    return np.concatenate((misc_arr, additional_arr), axis=1)


def visualization(out_df):
    current_dir = os.getcwd().replace('\\', '/')
    visualization_dir = current_dir + '/data/visualization'
    shutil.rmtree(visualization_dir, ignore_errors = True)
    if os.path.isdir(visualization_dir) == False:
        os.makedirs(visualization_dir)
    for person_id in range(out_df['Person ID'].min(), out_df['Person ID'].max()+1):
        person_dir = visualization_dir + '/' + str(person_id)
        if os.path.isdir(person_dir) == False:
            os.makedirs(person_dir)
        person_df = out_df[out_df['Person ID'] == person_id]
        for index in range(len(person_df['Person ID'].to_list())):
            person_data = person_df.values[index]
            source_file = current_dir + '/data/gallery/' + str(person_data[2]) + '/' + str(person_data[1])
            # print(source_file)
            filename = str(person_data[3]).replace('/', '_')
            filename = filename.replace(':', '_')
            filename = filename.replace(' ', '_')
            target_file = person_dir + '/' + 'cam_' + str(person_data[2]) + '_time_' + filename + '.jpg'
            # print(target_file)
            shutil.copyfile(source_file, target_file)
