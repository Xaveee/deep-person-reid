import cv2 # pip install opencv-python
import numpy as np
import pandas as pd
import math
from torchreid.utils.feature_extractor import FeatureExtractor
import os
from multiprocessing import Process, Queue

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

        if (frame_count % 30 == 0):
            ClassIndex, confidence, bbox = detection_model.detect(
                frame, confThreshold=0.6
            )
            if len(ClassIndex) != 0:
                for ClassInd, conf, boxes in zip(
                    ClassIndex.flatten(), confidence.flatten(), bbox
                ):
                    x, y, x1, y1 = boxes
                    if ClassInd == 1:
                        crop_img = frame[y - padding:y1 + y + padding,
                                         x - padding:x1 + x + padding]
                        img_name = cam_id + '_' + '.jpg'
                        cv2.imwrite(os.path.join('data', img_name), crop_img)
                        print('Extracting', img_name)
                        img_feature = feature_extractor(
                            os.path.join('data', img_name)
                        )
                        # query_queue.put(img_feature)
                        # print(img_feature)

                        img_feature_df = pd.DataFrame(img_feature.numpy())
                        img_feature_df.insert(0, 'id', -1)
                        img_feature_df.insert(0, 'cam_id', cam_id)
                        img_feature_df.insert(0, 'filename', img_name)
                        img_feature_df.insert(0, 'frame', frame_count)
                        print(img_feature_df)
                        query_queue.put(img_feature_df)

        cv2.imshow('Camera ' + cam_id, frame)
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

    counter = 0
    while (1):
        if query_queue.empty():
            continue
        # Get item from query queue
        feature = query_queue.get()
        # Perform Comparison. Maybe every 5 feature or 5 second? Comparing every n
        counter += 1
        if counter >= 5:
            # Perform Comparison
            pass
        else:
            counter = 0

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
    gallery = []
    for cam_id, cam in enumerate(cam_list):
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