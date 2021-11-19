import cv2 # pip install opencv-python
import sys
import numpy as np
import pandas as pd
import math
from torchreid.utils.feature_extractor import FeatureExtractor
import threading
import os
import glob
from queue import Queue
from multiprocessing import Process

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
feature_model = 'osnet_ibn_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
gallery_feature_df = pd.read_csv('data\gallery_data.csv')
model = cv2.dnn_DetectionModel(frozen_model, config_file)
extractor = FeatureExtractor(
    model_name='osnet_x1_0', model_path=feature_model, device='cpu'
)

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


def cam_worker(query_queue, feature_extractor, cam):
    cap = cv2.VideoCapture(cam)
    while (cap.isOpen()):
        # Extract features
        # Append features to query queue
        query_queue.put(feature_extractor)
    cap.close()


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
        if query_queue.is_empty():
            continue
    # Get item from query queue
    query_queue.get()
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
    for cam in cam_list:
        process = Process(
            target=cam_worker, arg=(query_queue, feature_extractor, cam)
        )
        process.start()

    process = Process(target=comparing_worker, arg=(query_queue, gallery))