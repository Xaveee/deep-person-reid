import cv2 # pip install opencv-python
import sys
import numpy as np
import pandas as pd
import math
from torchreid.utils.feature_extractor import FeatureExtractor

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
id = 0

# classLabels = []
# file_name = "labels.txt"
# with open(file_name, 'rt') as fpt:
#     classLabels = fpt.read().rstrip("\n").split("\n")
# print(classLabels)

model.setInputSize(320, 320) # image resolution
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1,1]
model.setInputSwapRB(True)

cap = cv2.VideoCapture("Videos/team_demo.mp4")
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

padding = 5
frameCount = 0

# Threshold for filtering the distance matrix
threshold = 4.8

def euclidean_dist(input1, input2):
    dist = 0
    for i in range(len(input1)):
        dist += math.pow(input1[i] - input2[i], 2)
        # dist += abs(input1[i] - input2[i])
    dist = math.sqrt(dist)

    return dist

def get_dist_matrix(query_df, gallery_df):
    distance_matrix = []
    for i, feature in gallery_df.iterrows():
        distance = euclidean_dist(feature.iloc[2:], query_df.iloc[0].iloc[2:])
        distance_matrix.append((i, int(gallery_df.iloc[i]['id']), distance))

    return distance_matrix


while cap.isOpened():
    ret, frame = cap.read()
    frameCount += 1
    if (frameCount % 30 == 0):
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)

        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(
                ClassIndex.flatten(), confidence.flatten(), bbox
            ):

            # crop the photo and save it in the output folder
                x, y, x1, y1 = boxes
                if ClassInd == 1:
                #cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    # frame[height:height+height, width:width+width] with additonal padding

                    crop_img = frame[y - padding:y1 + y + padding,
                                     x - padding:x1 + x + padding]
                    try:
                        #try to resize to fit the size torchreid accepts
                        crop_img = cv2.resize(crop_img, (128, 256))
                    except:
                        continue
                    # crop_img = remove_bg(crop_img)
                    image_name = queryFolder + str(frameCount) + ".jpg"
                    cv2.imwrite(image_name, crop_img)
                    try:
                        # Get image features and convert to DataFrame
                        print('Getting query image feature at frame', frameCount)
                        img_feature = extractor(image_name)
                        img_feature_df = pd.DataFrame(img_feature.numpy())
                        img_feature_df.insert(0, 'id', id)
                        img_feature_df.insert(0, 'filename', image_name)

                        if len(gallery_feature_df) == 0:
                            # append gallery data with the id of query image
                            print('No image in database yet')
                            print('Adding img', id, 'to the gallery...\n')
                            new_image_name = outputFolder + '_' + str(id) + '_'+ str(frameCount) + ".jpg"
                            cv2.imwrite(new_image_name, crop_img)
                            img_feature_df = img_feature_df.replace({'filename': image_name}, new_image_name)
                            gallery_feature_df = img_feature_df
                        else:
                            # get distance matrix (g_id, dist)
                            print('Calculating distance matrix')
                            distance_matrix = []
                            distance_matrix = get_dist_matrix(img_feature_df, gallery_feature_df)
                            # Sort ascending and Filter with threshold
                            distance_matrix = sorted(distance_matrix, key=lambda i: i[2])
                            distance_matrix = [i for i in distance_matrix if i[2] < threshold]
                            # print(distance_matrix)

                            if (len(distance_matrix) == 0):
                                print('No possible match')

                                print('Adding new person to gallery.', 'New ID:', id, '\n')
                                new_image_name = outputFolder + '_' + str(id) + '_'+ str(frameCount) + ".jpg"

                                img_feature_df = img_feature_df.replace({'filename': image_name}, new_image_name)
                                # append gallery data with the id of query image
                                gallery_feature_df = pd.concat([gallery_feature_df, img_feature_df], ignore_index=True)
                                cv2.imwrite(new_image_name, crop_img)

                            else:
                                print(len(distance_matrix), 'possible mathches')
                                print('Top 5 possible matches for', frameCount, ':')
                                for i, dist_id, dist in distance_matrix[:5]:
                                    print(gallery_feature_df.iloc[i]['filename'])

                                print('Adding old person to gallery...', 'ID:', distance_matrix[0][1],'\n')
                                new_image_name = outputFolder + str(distance_matrix[0][1]) + '_'+ str(frameCount) + ".jpg"

                                # append gallery data with the id of image with lowest dist
                                img_feature_df = img_feature_df.replace({'filename': image_name}, new_image_name)
                                img_feature_df = img_feature_df.replace({'id': id}, distance_matrix[0][1])
                                gallery_feature_df = pd.concat([gallery_feature_df, img_feature_df], ignore_index=True)
                                cv2.imwrite(new_image_name, crop_img)
                        id += 1


                        # image_name = outputFolder + str(id) + ".jpg"
                        # cv2.imwrite(image_name, crop_img)
                        # print(image_name) cv  

                    except:
                        continue

    # Display the resulting frame
    cv2.imshow('Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
