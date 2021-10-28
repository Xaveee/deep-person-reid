import cv2 # pip install opencv-python

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# output folder and the id start number change these!!!!!!
outputFolder = "C:\\Temp\\"
id = 0

classLabels = []
file_name = "labels.txt"
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip("\n").split("\n")
print(classLabels)

# This is probably able to be
model.setInputSize(320, 320) # image resolution
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1,1]
model.setInputSwapRB(True)

cap = cv2.VideoCapture("Videos/team_demo.mp4")
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

padding = 5
frameCount = 0

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        frameCount += 1
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)

        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(
                ClassIndex.flatten(), confidence.flatten(), bbox
            ):

                # crop the photo and save it in the output folder
                x, y, x1, y1 = boxes
                # draw the bounding box
                if ClassInd == 1:
                    #clscommented out for now since we don't want it drawn but not removing in case we need it for latercv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    if (frameCount % 10 == 0):
                        # frame[height:height+height, width:width+width] with additonal padding
                        crop_img = frame[y - padding:y1 + y + padding,
                                         x - padding:x1 + x + padding]

                        image_name = outputFolder + str(id) + ".jpg"
                        try:
                            cv2.imwrite(image_name, crop_img)
                            id += 1
                            print(image_name)
                        except:
                            continue

        # Display the resulting frame
        cv2.imshow('Person Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
cap.destroyAllWindows()
