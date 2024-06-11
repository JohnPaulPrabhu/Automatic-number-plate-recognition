from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import get_car

def main():
    motion_tracker = Sort()
    
    # load the number plate detector model
    # license_plate_detector = YOLO("number-plate-detector.pt")
    license_plate_detector = YOLO("number-plate-detector.pt")
    coco_model = YOLO("yolov8n.pt")
    
    # load the video
    cap = cv2.VideoCapture("sample.mp4")
    ret = True
    detect = [2, 3, 5, 7]   
    # total = -1
    # read frame
    while ret:
        # total+=1
        ret, frame = cap.read()
        # # detect vehivlcle
        # # if total >= 100:
        # #     break
        if ret:
        #     detections = coco_model(frame)[0]
        #     detections_ = []
        #     # for i in detections.boxes:
        #     #     print(i.xyxy[0])
        #         # print(i.xyxy[1])
        #     # break
        #     print(detections.boxes)
        #     for detection in detections.boxes.data.tolist():
        #         x1, y1, x2, y2, confidence_score, class_id = detection
        #         if int(class_id) in detect:
        #             detections_.append([x1, y1, x2, y2, confidence_score])
            
            # # track vehicle
            # track_ids = motion_tracker.update(np.asarray(detections_))
    
            # detect license plate
            license_plates = license_plate_detector(frame)
            # if license_plates.boxes is None:
                # print("============================================")
                # continue
            # print("license_plates", license_plates)
            # for license_plate in license_plates.boxes.data.tolist():
            #     x1, y1, x2, y2, confidence_score, class_id = license_plate
                
    
            #     # assign license plate to car
            #     get_car(license_plate, track_ids)
            #     print('1, ', license_plate)
            #     print('2, ', track_ids)
            # crop license plate

    
            # process license plate
            

            # read license plate number
        
        
            # write results 
            
            
            # Visualize it
            # for r in license_plates:
            #     print(r.plot())
            #     for obb in r.obb:
            #         print("=============================================")
            #         print(obb)
            #         print("=============================================")
            #         # Extract bbox and angle
            #         bbox = obb.xywhr[0][:4].tolist()  # Get (cx, cy, w, h)
            #         angle = obb.xywhr[0][4].item()  # Get the angle
            #         draw_obb(frame, bbox, angle)
            #         # Naming a window 
            #         cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
                      
            #         # Using resizeWindow() 
            #         cv2.resizeWindow("Resized_Window", 600,600)
            #         cv2.imshow("Resized_Window", frame)
            print(license_plates[0].obb)
            for i in license_plates[0].obb:
                print(i.xyxy[0])
                x1, y1, x2, y2 = map(int, i.xyxy[0])
                conf = i.conf
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=8)
                # Naming a window 
                cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
                  
                # Using resizeWindow() 
                cv2.resizeWindow("Resized_Window", 600,600)
                cv2.imshow("Resized_Window", frame)
            # for r in license_plates:
            #     # print("=============================================")
            #     # print(r.obb)
            #     # print("=============================================")
            #     # print(r.boxes)
            #     # print(r.obb.xywhr[0].int().tolist())
            #     x, y, w, h, angle = r.obb.xywhr[0].int().tolist() #, box.angle.item()
            #     # print("================")
            #     # print(x, y, w, h, angle)
            #     draw_obb(frame, (x, y, w, h), angle)
                
            #     im_bgr = r.plot()
            #     # print(im_bgr)
            #     # Naming a window 
            #     cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
                  
            #     # Using resizeWindow() 
            #     cv2.resizeWindow("Resized_Window", 600,600)
            #     cv2.imshow("Resized_Window", im_bgr)
            
            
            # for box in license_plates.boxes:
            #     x1, y1, x2, y2 = box.xyxy[0].int()
            #     conf = box.conf[0].item()
                
            #     # Draw bouding box
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            #     # Draw label with confidence score
            #     cv2.putText(frame, f'Plate {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
            # Display the frame
            # cv2.imshow("License plate detection", frame)
            
            # Break loop on press `Q`
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def draw_obb(image, bbox, angle, color=(0, 255, 0), thickness=2):
    cx, cy, w, h = bbox
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # Define the four corners of the bounding box
    box = np.array([
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
        [cx - w / 2, cy + h / 2]
    ])
    # Rotate the bounding box
    rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0]
    rotated_box = rotated_box.astype(int)
    # Draw the rotated bounding box
    for i in range(4):
        cv2.line(image, tuple(rotated_box[i]), tuple(rotated_box[(i + 1) % 4]), color, thickness)

    
if __name__ == '__main__':
    main()