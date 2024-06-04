from ultralytics import YOLO
import cv2

def main():
    # load the number plate detector model
    model = YOLO("number-plate-detector.pt")
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
        if total > 10:
            break
        if ret:
            detections = coco_model(frame)[0]
            for detection in detections.boxes.data.tolist():
                print(detection)
            # print(detections)
            
    
    
    
    
    
    # track vehicle
    
    
    
    
    # detect license plate
    
    
    
    
    
    # assign license plate to car
    
    
    
    
    
    # crop license plate
    
    
    
    
    
    # process license plate
    


    # read license plate number



    # write results 
    
    
    
if __name__ == '__main__':
    main()