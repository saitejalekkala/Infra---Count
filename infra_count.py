import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
import supervision as sv

import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')




cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print('Error: Please open the camera.')
else:

    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:

        ret, frame = cap.read()


        if not ret:
            print('Error: Please keep the frame nicely.')
            break


        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        img_neg = 255 - img_gray


        img_color = cv2.applyColorMap(img_neg, cv2.COLORMAP_JET)


        results = model(frame, size=1280)
        detections = sv.Detections.from_yolov5(results)
        detections = detections[(detections.class_id == 0)]


        num_persons = len(detections)


        text_position = (int(0.05*width), int(0.1*height))
        text_scale = 1
        text_thickness = 1
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img_color, 
            f"Person Count: {num_persons}", 
            text_position, 
            font, 
            text_scale, 
            text_color, 
            text_thickness
        )


        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        img_color = box_annotator.annotate(scene=img_color, detections=detections)


        img_color = cv2.resize(img_color, (width, height))


        cv2.imshow('Thermal-like Image', img_color)


        if cv2.waitKey(1) == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
