import cv2
import numpy as np
import pandas as pd
from torchreid.utils.feature_extractor import FeatureExtractor
import os
from multiprocessing import Process, Queue
from utilities import birch_clustering, DBSCAN_clustering, mean_shift_clustering, get_output

gallery_feature_df = pd.read_csv('data\gallery_data.csv')

# output folder and the id start number change these!!!!!!
outputFolder = "data\gallery\gal"
queryFolder = 'data\query\query'
"""
cam_worker(query_queue, feature_extractor, cam):
    cap = cv2.VideoCapture(cam)
    while(cap.isOpen()):
        # Extract features
        # Append features to query queue
        query_queue.put(feature)
    cap.close()
"""


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
    cap = cv2.VideoCapture(os.path.join('Videos/thread', cam))
    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if (ret == False):
            print('cam', cam_id, 'ended')
            break
        frame = cv2.resize(frame, (320, 320))
        if (frame_count % 30 == 0):
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
                            cv2.resize(crop_img, (360, 640))
                        )
                        print('Extracting', img_name)
                        img_feature = feature_extractor(
                            os.path.join(gal_folder, img_name)
                        )
                        # query_queue.put(img_feature)
                        # print(img_feature)

                        img_feature_df = pd.DataFrame(img_feature.numpy())
                        img_feature_df.insert(0, 'id', -1)
                        img_feature_df.insert(0, 'cam_id', cam_id)
                        img_feature_df.insert(0, 'filename', img_name)
                        img_feature_df.insert(0, 'frame', frame_count)
                        # print(img_feature_df)
                        query_queue.put(img_feature_df)

        cv2.imshow('Camera ' + cam_id, cv2.resize(frame, (675, 1200)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('killing cam', cam_id)
            query_queue.close()
            break
        # Extract features
        # Append features to query queue
        # query_queue.put(feature_extractor)
    cap.release()
    cv2.destroyAllWindows()


"""
comparing_worker(query_queue, gallery):
    while(1):
        if query_queue.is_empty():
            continue
            
        # Get item from query queue
        query_queue.get()
        # Perform Comparison. Maybe every 5 feature or 5 second? Comparing every n
        # The item(features) from the query has a camID column. ONLY compare with other item that has DIFFERENT camID
"""


def comparing_worker(query_queue, gallery):
    # Output csv's columns: frame count - file name - cam id - person id - 5 people in the same cluster (not closest)
    counter = 0
    while (1):
        if query_queue.empty():
            continue
        # Get item from query queue
        feature = query_queue.get()
        np_feature = feature.values
        # print('working...')
        if gallery == [[]]:
            gallery = np_feature
        else:
            gallery = np.concatenate((gallery, np_feature), axis=0)
        if gallery.shape[0] % 100 != 0:
            continue
        # Perform Comparison. Maybe every 5 feature or 5 second? Comparing every n
        feature_arr = gallery[:, 4:]
        # print(feature_arr)
        label = mean_shift_clustering(feature_arr).reshape(-1, 1)
        labeled_arr = np.concatenate((gallery, label), axis=1)

        # print(labeled_arr)
        # End of comparison
        counter += 1
        if gallery.shape[0] % 100 == 0:
            # Update features csv
            out_df = pd.DataFrame(get_output(labeled_arr))
            out_df.to_csv('data/labeled_gal.csv', header=False, index=False)
            gal_df = pd.DataFrame(gallery)
            gal_df.to_csv('data/gallery.csv', header=False, index=False)
        else:
            pass

        # The item(features) from the query has a camID column. ONLY compare with other item that has DIFFERENT camID


"""
main(cam_list, feature_extractor):
    # initialize query_queue and gallery
    
    for cam in cam_list:
        process = Process(target=cam_worker, arg=(query_queue, feature_extractor, cam)
        process.start
        
    
    process = Process(target=comparing_worker, arg=(query_queue, gallery)
"""


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
    print(cam_list)
    main(cam_list, feature_extractor)
