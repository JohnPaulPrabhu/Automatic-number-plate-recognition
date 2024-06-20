from ultralytics import YOLOv10
import supervision as sv
import cv2
import numpy as np
from util import get_car, read_license_plate, write_csv, iou, KalmanBoxTracker, assign_detections_to_trackers
from collections import deque

byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()

def main():
    license_plate_detector = YOLOv10("best.pt")
    license_plate_detector.to('cuda')
    coco_model = YOLOv10("yolov10n.pt")
    coco_model.to('cuda')
    cap = cv2.VideoCapture("sample.mp4")
    ret = True
    detect = [2, 3, 5, 7]
    frame_number = -1
    results = {}
    trackers = []
    track_id_list = []
    max_age = 1
    min_hits = 3
    track_dict = {}
    memory = {}

    while ret:
        frame_number += 1
        results[frame_number] = {}
        ret, frame = cap.read()
        if not ret:
            print("End of video or error in reading frame.")
            break
           
        detections = coco_model(frame)[0]
        detections = sv.Detections.from_ultralytics(detections)
        tracks = byte_tracker.update_with_detections(detections)
        dets = np.array([[*d[0], d[2]] for d in detections if d[3] in detect])
        trks = np.zeros((len(trackers), 5))

        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(dets, trks, iou_threshold=0.3)

        for t, trk in enumerate(trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])
                track_id_list.append(trackers[t].id)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            trackers.append(trk)
            track_id_list.append(trk.id)

        i = len(trackers)
        for trk in reversed(trackers):
            d = trk.get_state()[0]
            if trk.time_since_update > max_age:
                trackers.pop(i)
            i -= 1

        track_bbs_ids = []
        for track in tracks:
            x1, y1, x2, y2  = track[0]
            track_id = track[2]
            track_bbs_ids.append(np.array([x1, y1, x2, y2, track_id]))
        track_bbs_ids = np.array(track_bbs_ids)

        license_plates = license_plate_detector(frame)
        for license_plate in license_plates[0].boxes.data.tolist():
            x1, y1, x2, y2, conf_score, cls_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car((x1, y1, x2, y2), track_bbs_ids)

            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_plate_text, license_plate_text_conf_score = read_license_plate(license_plate_thresh)
                if license_plate_text is not None:
                    if car_id not in memory:
                        memory[car_id] = deque(maxlen=5)
                    memory[car_id].append(license_plate_text)
                    if len(memory[car_id]) == 5:
                        consistent_license_plate = max(set(memory[car_id]), key=memory[car_id].count)
                        results[frame_number][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'license_plate_conf_score': conf_score,
                                'license_text': consistent_license_plate,
                                'license_plate_text_conf_score': license_plate_text_conf_score
                            }
                        }
        
        for license_plate in license_plates[0].boxes.data.tolist():
            x1, y1, x2, y2, conf_score, cls_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=8)
            cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Resized_Window", 900, 600)
            cv2.imshow("Resized_Window", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print('checking break')
    write_csv(results, './test.csv')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()












# # Copy paste the prignal main.py so that kalfman filter is implemented in the original version, not the below version


# from ultralytics import YOLOv10
# import supervision as sv
# import cv2
# import numpy as np
# from util import get_car, read_license_plate, write_csv
# from collections import deque

# # Initialize ByteTrack and BoxAnnotator from supervision
# byte_tracker = sv.ByteTrack()
# annotator = sv.BoxAnnotator()

# def main():
#     # Load license plate and COCO models
#     license_plate_detector = YOLOv10("best.pt")
#     license_plate_detector.to('cuda')
#     coco_model = YOLOv10("yolov10n.pt")
#     coco_model.to('cuda')
    
#     # Load video
#     cap = cv2.VideoCapture("sample.mp4")
#     if not cap.isOpened():
#         print("Error: Unable to open video file.")
#         return
    
#     frame_number = -1
#     results = {}
#     memory = {}  # Memory for storing license plate text
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("End of video or error in reading frame.")
#             break
        
#         frame_number += 1
#         results[frame_number] = {}
        
#         # Detect vehicles
#         detections = coco_model(frame)[0]
#         detections = sv.Detections.from_ultralytics(detections)
#         tracks = byte_tracker.update_with_detections(detections)
        
#         # Filter and prepare detections for tracking
#         # dets = np.array([[*d.boxes.data.tolist()[0][:4], d.boxes.data.tolist()[0][4]] for d in detections if int(d.boxes.data.tolist()[0][5]) in [2, 3, 5, 7]])
#         dets = np.array([[*d[0], d[2]] for d in detections if d[3] in [2, 3, 5, 7]])

#         # Collect track bounding boxes and IDs
#         track_bbs_ids = []
#         for track in tracks:
#             # print(track)
#             x1, y1, x2, y2  = track[0]
#             track_id = track[2]
#             track_bbs_ids.append(np.array([x1, y1, x2, y2, track_id]))
#         track_bbs_ids = np.array(track_bbs_ids)

#         # Detect license plates in the frame
#         license_plates = license_plate_detector(frame)
#         for license_plate in license_plates[0].boxes.data.tolist():
#             # print(license_plate)
#             x1, y1, x2, y2, conf_score, cls_id = license_plate
#             print("================================")
#             xcar1, ycar1, xcar2, ycar2, car_id = get_car((x1, y1, x2, y2), track_bbs_ids)
#             print(xcar1, ycar1, xcar2, ycar2, car_id)
#             print("================================")
#             if car_id != -1:
#                 # Crop and process license plate
#                 license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#                 license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                 _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

#                 # Read license plate text
#                 license_plate_text, license_plate_text_conf_score = read_license_plate(license_plate_thresh)
#                 if license_plate_text is not None:
#                     if car_id not in memory:
#                         memory[car_id] = deque(maxlen=5)
#                     memory[car_id].append(license_plate_text)
#                     if len(memory[car_id]) == 5:
#                         consistent_license_plate = max(set(memory[car_id]), key=memory[car_id].count)
#                         results[frame_number][car_id] = {
#                             'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                             'license_plate': {
#                                 'bbox': [x1, y1, x2, y2],
#                                 'license_plate_conf_score': conf_score,
#                                 'license_text': consistent_license_plate,
#                                 'license_plate_text_conf_score': license_plate_text_conf_score
#                             }
#                         }

#         # Visualize the results
#         for license_plate in license_plates[0].boxes.data.tolist():
#             x1, y1, x2, y2, conf_score, cls_id = license_plate
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=8)
#             cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
#             cv2.resizeWindow("Resized_Window", 900, 600)
#             cv2.imshow("Resized_Window", frame)

#         # Break loop on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     print(results)
#     # Write results to CSV
#     write_csv(results, './test.csv')
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
















# from ultralytics import YOLOv10
# import supervision as sv
# import cv2
# import numpy as np
# from util import get_car, read_license_plate, write_csv, iou, KalmanBoxTracker, assign_detections_to_trackers
# from collections import deque

# byte_tracker = sv.ByteTrack()
# annotator = sv.BoxAnnotator()

# def main():
#     license_plate_detector = YOLOv10("best.pt")
#     license_plate_detector.to('cuda')
#     coco_model = YOLOv10("yolov10n.pt")
#     coco_model.to('cuda')
#     cap = cv2.VideoCapture("sample.mp4")
#     ret = True
#     detect = [2, 3, 5, 7]
#     frame_number = -1
#     results = {}
#     trackers = []
#     track_id_list = []
#     max_age = 1
#     min_hits = 3
#     track_dict = {}
#     memory = {}

#     while ret:
#         frame_number += 1
#         results[frame_number] = {}
#         ret, frame = cap.read()
#         if not ret:
#             print("End of video or error in reading frame.")
#             break
#         if ret:
#             detections = coco_model(frame)[0]
#             detections = sv.Detections.from_ultralytics(detections)
#             tracks = byte_tracker.update_with_detections(detections)
#             # for detection in detections:
#             #     print(detection[0])
#             #     print(detection[1])
#             #     print(detection[2])
#             #     print(detection[3])
#             #     print(detection[4])

#             dets = np.array([[*d[0], d[2]] for d in detections if d[3] in detect])
#             trks = np.zeros((len(trackers), 5))

#             to_del = []
#             ret = []
#             for t, trk in enumerate(trks):
#                 pos = trackers[t].predict()[0]
#                 trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#                 if np.any(np.isnan(pos)):
#                     to_del.append(t)
#             trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

#             matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(dets, trks, iou_threshold=0.3)

#             for t, trk in enumerate(trackers):
#                 if t not in unmatched_trks:
#                     d = matched[np.where(matched[:, 1] == t)[0], 0]
#                     trk.update(dets[d, :][0])
#                     track_id_list.append(trackers[t].id)

#             for i in unmatched_dets:
#                 trk = KalmanBoxTracker(dets[i, :])
#                 trackers.append(trk)
#                 track_id_list.append(trk.id)

#             i = len(trackers)
#             for trk in reversed(trackers):
#                 d = trk.get_state()[0]
#                 if trk.time_since_update > max_age:
#                     trackers.pop(i)
#                 i -= 1

#             track_bbs_ids = []
#             for trk in trackers:
#                 pos = trk.get_state()[0]
#                 if (trk.hit_streak >= min_hits or frame_number <= min_hits) and trk.time_since_update <= max_age:
#                     track_bbs_ids.append(np.concatenate((pos, [trk.id])).reshape(1, -1))
#                     track_dict[trk.id] = deque(maxlen=100)
#                 if trk.id not in track_dict:
#                     track_dict[trk.id] = deque(maxlen=100)
#                 track_dict[trk.id].append((frame_number, pos[:4]))

#             track_bbs_ids = np.concatenate(track_bbs_ids) if len(track_bbs_ids) > 0 else []

#             license_plates = license_plate_detector(frame)
#             for license_plate in license_plates[0].boxes.data.tolist():
#                 x1, y1, x2, y2, conf_score, cls_id = license_plate
#                 xcar1, ycar1, xcar2, ycar2, car_id = get_car((x1, y1, x2, y2), track_bbs_ids)

#                 if car_id != -1:
#                     license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#                     license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                     _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

#                     license_plate_text, license_plate_text_conf_score = read_license_plate(license_plate_thresh)
#                     if license_plate_text is not None:
#                         if car_id not in memory:
#                             memory[car_id] = deque(maxlen=5)
#                         memory[car_id].append(license_plate_text)
#                         if len(memory[car_id]) == 5:
#                             consistent_license_plate = max(set(memory[car_id]), key=memory[car_id].count)
#                             results[frame_number][car_id] = {
#                                 'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                 'license_plate': {
#                                     'bbox': [x1, y1, x2, y2],
#                                     'license_plate_conf_score': conf_score,
#                                     'license_text': consistent_license_plate,
#                                     'license_plate_text_conf_score': license_plate_text_conf_score
#                                 }
#                             }

#             for license_plate in license_plates[0].boxes.data.tolist():
#                 x1, y1, x2, y2, conf_score, cls_id = license_plate
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=8)
#                 cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
#                 cv2.resizeWindow("Resized_Window", 900, 600)
#                 cv2.imshow("Resized_Window", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("checking break")
#                 break
#             print('checking')
#     write_csv(results, './test.csv')
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()








# # import torch
# from ultralytics import YOLOv10
# import supervision as sv
# import cv2
# import numpy as np
# # from sort.sort import *
# from util import get_car, read_license_plate, write_csv

# # from ByteTrack
# # from deep_sort.deep_sort import tracker
# # from deep_sort import Deepsort
# # from deep_sort.deep_sort import tracker
# # from yolox.tracker.byte_tracker import BYTETracker
# # from yolox.tracker.byte_tracker import BYTETracker
# # Initialize Deep SORT
# # deepsort = DeepSort("ckpt.t7", use_cuda=True)
# # Initialize BYTETracker
# # tracker = BYTETracker()
# byte_tracker = sv.ByteTrack()
# annotator = sv.BoxAnnotator()
# import torch
# # print('checking', torch.cuda.is_available())

# def main():
#     # motion_tracker = tracker.Tracker()
#     # motion_tracker = Sort()
#     # print(torch.cuda.is_available())
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # print(device)
    
#     # load the number plate detector model
#     license_plate_detector = YOLOv10("best.pt")
#     license_plate_detector.to('cuda')
#     coco_model = YOLOv10("yolov10n.pt")
#     coco_model.to('cuda')
#     # coco_model.track()
#     # load the video
#     cap = cv2.VideoCapture("sample.mp4")
#     ret = True
#     detect = [2, 3, 5, 7]   
#     frame_number = -1
#     results = {}
#     # read frame
#     while ret:
#         # if frame_number > 500:
#         #     break
#         frame_number+=1
#         results[frame_number] = {}
#         ret, frame = cap.read()
#         # # detect vehivlcle
#         # if frame_number
#         if ret:
#             detections = coco_model(frame)[0]
#             # detections = sv.Detections.from_ultralytics(detections)
#             detections = sv.Detections.from_ultralytics(detections)
#             track_ids = byte_tracker.update_with_detections(detections)
            
#             # detections_ = []
#             # print(detections)
#             # for detection in detections.boxes.data.tolist():
#             #     print(detection)
#             #     x1, y1, x2, y2, confidence_score, class_id = detection
#             #     if int(class_id) in detect:
#             #         detections_.append([x1, y1, x2, y2, confidence_score, class_id])
            
#             # track vehicle
#             # track_ids = motion_tracker.update(np.asarray(detections_))
            
#             # Update tracker
#             # track_ids = tracker.update(detections_, frame)
#             # track_ids = byte_tracker.update_with_detections(detections)
#             print("============================================")
#             print(track_ids)
#             print("============================================")
            
#             # detect license plate
#             license_plates = license_plate_detector(frame)
#             for license_plate in license_plates[0].boxes.data.tolist():
#                 print(license_plate)
#                 x1, y1, x2, y2, conf_score, cls_id = license_plate
#                 # conf_score = license_plate.conf.item()
#                 # cls_id = license_plate.cls.item()

#                 # Assign license plate to car
#                 xcar1, ycar1, xcar2, ycar2, car_id = get_car((x1, y1, x2, y2), track_ids)
                
#                 if car_id != -1:
#                     # crop license plate
#                     license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        
#                     # process license plate
#                     license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                     _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
#                     # cv2.imshow("original crop", license_plate_crop)
#                     # cv2.imshow("thresholded image", license_plate_thresh)
#                     # cv2.waitKey(0)
    
#                     # read license plate number
#                     license_plate_text, license_plate_text_conf_score = read_license_plate(license_plate_thresh)
#                     # print("----------------------------------")
#                     # print(license_plate_text, license_plate_text_conf_score)
#                     # print("----------------------------------")
#                     # cv2.imshow("Checking", license_plate_gray)
#                     # cv2.waitKey(0)
#                     if license_plate_text is not None:
#                         results[frame_number][car_id] = {
#                                     'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                     'license_plate': {
#                                                         'bbox': [x1, y1, x2, y2],
#                                                         'license_plate_conf_score': conf_score,
#                                                         'license_text': license_plate_text,
#                                                         'license_plate_text_conf_score': license_plate_text_conf_score
#                                                     }
#                                                 }
#                     # print(results)
#             #         break
#             #     break
        
#             # break
#             # Visualize it
#             for license_plate in license_plates[0].boxes.data.tolist():
#                 x1, y1, x2, y2, conf_score, cls_id = license_plate
#                 # x1, y1, x2, y2 = map(int, i.xyxy[0])
#                 # conf = i.conf
#                 color = (0, 255, 0)
#                 print(type(x1))
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=8)
                
#                 # Naming a window 
#                 cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
                  
#                 # Using resizeWindow() 
#                 cv2.resizeWindow("Resized_Window", 900,600)
#                 cv2.imshow("Resized_Window", frame)
            
#             # Break loop on press `Q`
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
    
#     # write results 
#     write_csv(results, './test.csv')
        
#     cap.release()
#     cv2.destroyAllWindows()

    
# if __name__ == '__main__':
#     main()