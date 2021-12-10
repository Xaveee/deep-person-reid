#from sys import last_traceback
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.cluster.hierarchy import leaders
from torchreid.utils.feature_extractor import FeatureExtractor
import os
from multiprocessing import Process, Queue
from utilities import birch_clustering, DBSCAN_clustering, mean_shift_clustering, get_output, visualization

# output folder and the id start number change these!!!!!!
outputFolder = "data\gallery\gal"
queryFolder = 'data\query\query'


def cam_worker(query_queue, feature_extractor, cam, cam_id):
    config_file = 'Models/detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'Models/detection/frozen_inference_graph.pb'
    detection_model = cv2.dnn_DetectionModel(frozen_model, config_file)
    detection_model.setInputSize(320, 320) # image resolution
    detection_model.setInputScale(1.0 / 127.5)
    detection_model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1,1]
    detection_model.setInputSwapRB(True)

    padding = 5
    frame_count = 0

    print('starting cam', cam_id)
    # cap = cv2.VideoCapture(os.path.join('Videos/thread', cam))
    cap = cv2.VideoCapture(int(cam_id)+1)
    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if (ret == False):
            print('cam', cam_id, 'ended')
            break
        frame = cv2.resize(frame, (320, 320))
        if (frame_count % 5 == 0):
            ClassIndex, confidence, bbox = detection_model.detect(
                frame, confThreshold=0.6
            )
            if len(ClassIndex) != 0:
                for ClassInd, conf, boxes in zip(
                    ClassIndex.flatten(), confidence.flatten(), bbox
                ):
                    x, y, w, h = boxes
                    if ClassInd == 1:
                        crop_img = frame[y:y + h, x:x + w]
                        img_name = cam_id + '_' + str(frame_count) + '.jpg'
                        gal_folder = 'data/gallery/{}'.format(cam_id)
                        cv2.imwrite(
                            os.path.join(gal_folder, img_name),
                            # cv2.resize(crop_img, (360, 640))
                            # cv2.resize(crop_img, (640, 480))
                            crop_img
                        )
                        print('Extracting', img_name)
                        img_feature = feature_extractor(
                            os.path.join(gal_folder, img_name)
                        )
                        # query_queue.put(img_feature)
                        # print(img_feature)

                        img_feature_df = pd.DataFrame(img_feature.numpy())
                        img_feature_df.insert(0, 'actual_time', (datetime.now()-datetime(1970, 1, 1)).total_seconds())
                        img_feature_df.insert(0, 'display_time', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
                        img_feature_df.insert(0, 'cam_id', cam_id)
                        img_feature_df.insert(0, 'filename', img_name)
                        img_feature_df.insert(0, 'frame', frame_count)
                        # print(img_feature_df)
                        query_queue.put(img_feature_df)

        cv2.imshow('Camera ' + cam_id, cv2.resize(frame, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('killing cam', cam_id)
            query_queue.close()
            break
        # Extract features
        # Append features to query queue
        # query_queue.put(feature_extractor)
    cap.release()
    cv2.destroyAllWindows()




def comparing_worker(query_queue, gallery):
    '''
    Comparing component
    This component get the features of a new image from the query queue, then cluster and label them.
    Output csv's columns: frame count - file name - cam id - person id - 5 people in the same cluster (not closest)
    '''
    # counter = 0  # unused
    latest_save_time = datetime.now()
    while (1):
        if query_queue.empty():
            # If the queue is empty, keep looping
            continue
        # Get item from query queue
        feature = query_queue.get()
        np_feature = feature.values
        # print('working...')
        if gallery == [[]]:
            # If gallery is empty, set the first sample as gallery
            gallery = np_feature
        else:
            # If the gallery has data, add the new sample to gallery
            gallery = np.concatenate((gallery, np_feature), axis=0)
        if gallery.shape[0] % 100 != 0:
            # Perform clustering every 100 frames input
            continue
        # START COMPARISON.
        feature_arr = gallery[:, 4:]
        # We can change between DBSCAN_clustering, birch_clustering and mean_shift_clustering
        label = DBSCAN_clustering(feature_arr, eps = 4.6).reshape(-1, 1)
        labeled_arr = np.concatenate((gallery, label), axis=1)

        # print(labeled_arr)
        # END COMPARISON
        # counter += 1
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if (now-latest_save_time).total_seconds() >= 10:
            # Update features csv and labeled csv every 30s
            out_df = pd.DataFrame(get_output(labeled_arr))
            out_df.columns = ['Frame Count', 'File Name', 'Camera ID', 'Time Stamp', 'Person ID', '1st Match', '2nd Match', '3rd Match', '4th Match', '5th Match']
            out_df.to_csv('data/labeled_gal.csv', index=False)
            visualization(out_df)
            gal_df = pd.DataFrame(gallery)
            gal_df.to_csv('data/gallery.csv', header=False, index=False)
            # reset latest save time
            print()
            print('SAVED AT', current_time, end='\n\n')
            latest_save_time = now
        else:
            pass

        # The item(features) from the query has a camID column. ONLY compare with other item that has DIFFERENT camID



def main(cam_list, feature_extractor):
    # initialize query_queue and gallery
    query_queue = Queue()
    gallery = [[]]
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

    process = Process(target=comparing_worker, args=(query_queue, gallery))
    process.start()


if __name__ == '__main__':
    cam_list = os.listdir('Videos/thread')
    feature_model = 'Models/osnet/osnet_ms_d_c.pth.tar'

    feature_extractor = FeatureExtractor(
        model_name='osnet_x1_0', model_path=feature_model, device='cpu'
    )
    # print(cam_list)
    cam_num = 2
    cam_list = [''] * cam_num
    main(cam_list, feature_extractor)
