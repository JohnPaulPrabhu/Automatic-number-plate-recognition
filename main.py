from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv

def main():
    motion_tracker = Sort()
    
    # load the number plate detector model
    license_plate_detector = YOLO("number-plate-detector.pt")
    coco_model = YOLO("yolov8n.pt")
    
    # load the video
    cap = cv2.VideoCapture("sample.mp4")
    ret = True
    detect = [2, 3, 5, 7]   
    frame_number = -1
    results = {}
    # read frame
    while ret:
        frame_number+=1
        results[frame_number] = {}
        ret, frame = cap.read()
        # # detect vehivlcle
        # if frame_numbeq
        if ret:
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                print(detection)
                x1, y1, x2, y2, confidence_score, class_id = detection
                if int(class_id) in detect:
                    detections_.append([x1, y1, x2, y2, confidence_score])
            
            # track vehicle
            track_ids = motion_tracker.update(np.asarray(detections_))
            
            # detect license plate
            license_plates = license_plate_detector(frame)
            for license_plate in license_plates[0].obb:
                x1, y1, x2, y2 = map(float,license_plate.xyxy[0])
                conf_score = license_plate.conf.item()
                cls_id = license_plate.cls.item()

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car((x1, y1, x2, y2), track_ids)

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    
                # process license plate
                license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
                # cv2.imshow("original crop", license_plate_crop)
                # cv2.imshow("thresholded image", license_plate_thresh)
                # cv2.waitKey(0)

                # read license plate number
                license_plate_text, license_plate_text_conf_score = read_license_plate(license_plate_thresh)
                
                if license_plate_text is not None:
                    results[frame_number][car_id] = {
                                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                'license_plate': {
                                                    'bbox': [x1, y1, x2, y2],
                                                    'license_plate_conf_score': conf_score,
                                                    'license_text': license_plate_text,
                                                    'license_plate_text_conf_score': license_plate_text_conf_score
                                                }
                                            }
                    print(results)
                    break
                break
        
            break
            # Visualize it
            for i in license_plates[0].obb:
                x1, y1, x2, y2 = map(int, i.xyxy[0])
                conf = i.conf
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=8)
                
                # Naming a window 
                cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
                  
                # Using resizeWindow() 
                cv2.resizeWindow("Resized_Window", 900,600)
                cv2.imshow("Resized_Window", frame)
            
            # Break loop on press `Q`
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # write results 
    write_csv(results, './test.csv')
        
    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()