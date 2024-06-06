from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *


def main():
    motion_tracker = Sort()
    
    # load the number plate detector model
    license_plate_detector = YOLO("number-plate-detector.pt")
    coco_model = YOLO("yolov8n.pt")
    
    # load the video
    cap = cv2.VideoCapture("sample.mp4")
    ret = True
    detect = [3,4,6,8]    
    total = -1
    # read frame
    while ret:
        total+=1
        ret, frame = cap.read()
        # detect vehivlcle
        if total >= 1:
            break
        if ret:
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, confidence_score, class_id = detection
                if confidence_score in detect:
                    detections.append([x1, y1, x2, y2, confidence_score])
    
            # track vehicle
            motion_tracker.update(np.asarray(detections_))
    
            # detect license plate
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, confidence_score, class_id = license_plate
    
    
            # assign license plate to car
    
   
            # crop license plate

    
            # process license plate
            

            # read license plate number
        
        
            # write results 
    
    
    
if __name__ == '__main__':
    main()