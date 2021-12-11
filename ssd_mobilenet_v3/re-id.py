import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.cluster.hierarchy import leaders
from torchreid.utils.feature_extractor import FeatureExtractor
import os
from multiprocessing import Process, Queue
from utilities import birch_clustering, DBSCAN_clustering, mean_shift_clustering, get_output, visualization


# function by which each camera thread runs
# Functional requirement 10 in SRS

def cam_worker(query_queue, feature_extractor, cam, cam_id):
    # Initializing arguments for human detection extarctor

    config_file = 'Models/detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'Models/detection/frozen_inference_graph.pb'
    detection_model = cv2.dnn_DetectionModel(frozen_model, config_file)
    detection_model.setInputSize(320, 320) 
    detection_model.setInputScale(1.0 / 127.5)
    detection_model.setInputMean((127.5, 127.5, 127.5)) 
    detection_model.setInputSwapRB(True)

    padding = 5
    frame_count = 0

    print('starting cam', cam_id)

    # next line can be uncommented if user wants to run program on already recorded videos
    # cap = cv2.VideoCapture(os.path.join('Videos/thread', cam))
    cap = cv2.VideoCapture(int(cam_id)+1)
    while cap.isOpened():

        # Increasing frame count with each iteration

        frame_count += 1

        # Gathering frame from camera input

        ret, frame = cap.read()

        # If camera is not active, printing message to a user
        # Hardware requirement 2 in SRS

        if (ret == False):
            print('cam', cam_id, 'ended')
            break

        # Resizing an gathered frame

        frame = cv2.resize(frame, (320, 320))

        # Executing main body of human detection and feature extraction in 5 frame intervals
        # Non-funcional requirement 1 in SRS

        if (frame_count % 5 == 0):

            # Human detection module with threshold set to 0.6
            # Start of human detection module described in SDD

            ClassIndex, confidence, bbox = detection_model.detect(
                frame, confThreshold=0.6
            )

            # If human figures got detected it starts cropping images of detected people
            # Non-functional requirement 5 in SRS

            if len(ClassIndex) != 0:

                # Looping through each detected person to perform feature extraction

                for ClassInd, conf, boxes in zip(
                    ClassIndex.flatten(), confidence.flatten(), bbox
                ):
                    x, y, w, h = boxes

                    # Extraction of crop image of a detected person
                    # And adding camera id and frame count to a image name

                    if ClassInd == 1:
                        crop_img = frame[y:y + h, x:x + w]
                        img_name = cam_id + '_' + str(frame_count) + '.jpg'
                        gal_folder = 'data/gallery/{}'.format(cam_id)
                        cv2.imwrite(
                            os.path.join(gal_folder, img_name),
                            crop_img
                        )
                        print('Extracting', img_name)

                        # Extraction of features from the image
                        # Functional requirement 2 in SRS

                        img_feature = feature_extractor(
                            os.path.join(gal_folder, img_name)
                        )

                        # Gathering time stamps when image got taken and on which camera
                        # Functional requirement 8 in SRS

                        img_feature_df = pd.DataFrame(img_feature.numpy())
                        img_feature_df.insert(0, 'actual_time', (datetime.now()-datetime(1970, 1, 1)).total_seconds())
                        img_feature_df.insert(0, 'display_time', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
                        img_feature_df.insert(0, 'cam_id', cam_id)
                        img_feature_df.insert(0, 'filename', img_name)
                        img_feature_df.insert(0, 'frame', frame_count)

                        # Adding feature list of a person to a queue

                        query_queue.put(img_feature_df)


        # Displaying camera input

        cv2.imshow('Camera ' + cam_id, cv2.resize(frame, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('killing cam', cam_id)
            query_queue.close()
            break
    
    cap.release()
    cv2.destroyAllWindows()


# function on which comparing server runs
# it takes as a parameter feature queue and gallery defined more specificaly in SDD section 2.2 
# function performs reidentification and updates visualization folder and output csv files

def comparing_worker(query_queue, gallery):

    # Gathering time when comparizon module starts

    latest_save_time = datetime.now()

    # Starting main loop of reidentification module described in detailed in SDD section 3.2.3

    while (1):

        # checking if query queue is empty

        if query_queue.empty():
            continue

        # Getting item from a query queue and grabing values for each feature and putting them in a list

        feature = query_queue.get()
        np_feature = feature.values
        
        # Initializing gallery with list of features if gallery is empty
        # Otherwise adding list of features into gallery

        if gallery == [[]]:
            gallery = np_feature
        else:
            gallery = np.concatenate((gallery, np_feature), axis=0)

        # Performing clastering each 100 frames input 

        if gallery.shape[0] % 100 != 0:
            continue

        # Starting Reidentification main module 
        # For each element in gallery grabing all features and putting it into 2D list 

        feature_arr = gallery[:, 4:]

        # Clustering and comparing of collected features and merging them with ids, new cluster is created for each new person
        # Functional requirement 3, 4 and 5 in SRS
        # Non-functional requirement 6

        label = DBSCAN_clustering(feature_arr, eps = 4.6).reshape(-1, 1)
        labeled_arr = np.concatenate((gallery, label), axis=1)

        
        # End of reidentification process
        # Start of saving clustered data into visualization folder and csv files

        # Gathering current time
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        # Saving process is executed each 10 seconds

        if (now-latest_save_time).total_seconds() >= 10:
            # Converting clustered list into formated data list with closest neighbors
            # Desribed in detailed in SDD section 3.2.3.4

            out_df = pd.DataFrame(get_output(labeled_arr))

            # Saving clustered data into csv files in local database with labeling
            # Logical database requirement 3 and 4 in SRS

            out_df.columns = ['Frame Count', 'File Name', 'Camera ID', 'Time Stamp', 'Person ID', '1st Match', '2nd Match', '3rd Match', '4th Match', '5th Match']
            out_df.to_csv('data/labeled_gal.csv', index=False)
            visualization(out_df)
            gal_df = pd.DataFrame(gallery)
            gal_df.to_csv('data/gallery.csv', header=False, index=False)
            print()
            print('SAVED AT', current_time, end='\n\n')

            # updating time of last time saving process ended

            latest_save_time = now
        else:
            pass




def main(cam_list, feature_extractor):
    # initialize query_queue and gallery
    # described in detailed in SDD section 2.2

    query_queue = Queue()
    gallery = [[]]

    # for each camera starting individual thread
    # functional requirement 10 in SRS

    for cam_id, cam in enumerate(cam_list):
        try:
            os.mkdir('data/gallery/' + str(cam_id))
        except:
            pass
        process = Process(
            target=cam_worker,
            args=(query_queue, feature_extractor, cam, str(cam_id))
        )
        process.start()

    # starting reidentification module
    # described in detailed in SDD section 3.2.3

    process = Process(target=comparing_worker, args=(query_queue, gallery))
    process.start()


if __name__ == '__main__':
    # creating list for cameras
    cam_list = os.listdir('Videos/thread')

    # initializing model for feature extractor
    feature_model = 'Models/osnet/osnet_ms_d_c.pth.tar'

    feature_extractor = FeatureExtractor(
        model_name='osnet_x1_0', model_path=feature_model, device='cpu'
    )

    # Number of cameras, can be changed by the user
    cam_num = 2
    cam_list = [''] * cam_num
    main(cam_list, feature_extractor)

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

