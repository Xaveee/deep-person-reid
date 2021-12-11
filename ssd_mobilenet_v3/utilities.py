from operator import ne
import numpy as np
from numpy.core.numeric import identity
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import DBSCAN
from umap.umap_ import nearest_neighbors
import os
import shutil



# function performing clustering over features gallery
# it takes as a parameters feature list and predifined treshold, in this case setted to 3.1
# output is clustered labeled array
# defined in SDD section 3.2.3

def DBSCAN_clustering(feature, eps=3.1):
    # initializing parameters for clustering algorithm
    reducer = umap.UMAP(random_state=1, metric='manhattan')
    embedding = reducer.fit_transform(feature)

    # running DBSCAN clustering algorithm
    clustering = DBSCAN(eps=eps).fit(embedding)

    return clustering.labels_


# function that grabs list of features and finds 5 closest neighbors
# it takes as parameters chosen list of features and clusterd and labeled list of features
# it outputs list with 5 closest neighbors of input chosen list of features
# defined in SDD section 3.2.3

def get_nearest_neighbor(chosen_arr, labeled_arr):
    # grabing all list of features in the same cluster as chosen list of features

    same_clust = labeled_arr[labeled_arr[:, -1] == chosen_arr[-1], :]
    same_clust = same_clust[same_clust[:, 2] != chosen_arr[2], :]

    # if there is no matches filling list with message to a user
    if same_clust.size <= 5:
        return np.array(['No similar matches', '', '', '', ''])

    # sorting list of matches and grabbing first 5 elements
    same_clust = same_clust[np.argsort(same_clust[:, 4])]
    same_clust = same_clust[:5]
    
    # putting each element into output list
    nearest_neighbors = []
    for sample in same_clust:
        nearest_neighbors = np.append(nearest_neighbors, 'cam ' + str(sample[2]) + ' frame count ' + str(sample[0]))
    
    return nearest_neighbors

# function that prepars labeled array to be saved as csv file
# it grabs labaled dataset as input
# it outputs list ready to be converted to csv file
# defined in SDD section 3.2.3

def get_output(labeled_arr):
    
    misc_arr = labeled_arr[:, [0, 1, 2, 3, -1]]

    additional_arr = [[]]
    # looping through each person in labaled dataset
    for sample in labeled_arr:
        # getting nearest neighbors for current person
        nearest_neighbors = get_nearest_neighbor(sample, labeled_arr)
        
        # formating data in a list
        if nearest_neighbors.shape[0] < 5:
            nearest_neighbors = np.pad(nearest_neighbors, (0, 5 - nearest_neighbors.shape[0]), 'constant', constant_values=('', ''))
        if additional_arr == [[]]:
            additional_arr = [nearest_neighbors]
        else:
            additional_arr = np.concatenate((additional_arr, [nearest_neighbors]), axis=0)
    return np.concatenate((misc_arr, additional_arr), axis=1)

# function that saves images of a people into specific folder with correct label
# it grabs as input csv file
# defined in SDD section 3.2.3

def visualization(out_df):
    # getting current directory
    current_dir = os.getcwd().replace('\\', '/')
    visualization_dir = current_dir + '/data/visualization'
    shutil.rmtree(visualization_dir, ignore_errors = True)
    # creating visualization folder if it does not exist
    # otherwise opening visualization folder
    if os.path.isdir(visualization_dir) == False:
        os.makedirs(visualization_dir)
    # looping through each unique person in csv file
    for person_id in range(out_df['Person ID'].min(), out_df['Person ID'].max()+1):
        # creating a folder for a person if it does not exist
        person_dir = visualization_dir + '/' + str(person_id)
        if os.path.isdir(person_dir) == False:
            os.makedirs(person_dir)
        person_df = out_df[out_df['Person ID'] == person_id]
        # looping through each image of a current person and saving it to corresponding folder in visualization folder with correct label
        for index in range(len(person_df['Person ID'].to_list())):
            person_data = person_df.values[index]
            source_file = current_dir + '/data/gallery/' + str(person_data[2]) + '/' + str(person_data[1])
            
            filename = str(person_data[3]).replace('/', '_')
            filename = filename.replace(':', '_')
            filename = filename.replace(' ', '_')
            target_file = person_dir + '/' + 'cam_' + str(person_data[2]) + '_time_' + filename + '.jpg'
            
            shutil.copyfile(source_file, target_file)



# Copyright 2021 Missori State University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
