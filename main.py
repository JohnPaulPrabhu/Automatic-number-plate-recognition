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
    cap = cv2.VideoCapture("a.mp4")
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
    check_car_id = {}

    while ret:
        frame_number += 1
        print("frame_number", frame_number)
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
        for t, trk in enumerate(trks):
            pos = trackers[t].predict()
            trk[:] = [pos[0][0], pos[1][0], pos[2][0], pos[3][0], 0.0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(dets, trks, iou_threshold=0.3)

        for t, trk in enumerate(trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0][:4])
                track_id_list.append(trackers[t].id)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            trackers.append(trk)
            track_id_list.append(trk.id)
        
        i = len(trackers) - 1
        for trk in reversed(trackers):
            d = trk.get_state()[0]
            if trk.time_since_update > max_age:
                trackers.pop(i)
            i -= 1

        track_bbs_ids = []
        for trk in trackers:
            pos = trk.get_state()[0]
            if (trk.hit_streak >= min_hits or frame_number <= min_hits) and trk.time_since_update <= max_age:
                track_bbs_ids.append(np.concatenate((pos, [trk.id])).reshape(1, -1))
                track_dict[trk.id] = deque(maxlen=100)
            if trk.id not in track_dict:
                track_dict[trk.id] = deque(maxlen=100)
            track_dict[trk.id].append((frame_number, pos[:4]))

        track_bbs_ids = np.concatenate(track_bbs_ids) if len(track_bbs_ids) > 0 else []
        
        license_plates = license_plate_detector(frame)
        # print('================================================================')
        # print(license_plates[0].boxes.data.tolist())
        # print('================================================================')
        print('===========================================================================================================================')
        for license_plate in license_plates[0].boxes.data.tolist():
            x1, y1, x2, y2, conf_score, cls_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car((x1, y1, x2, y2), track_bbs_ids)
            print("car_id", car_id)
            
            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_thresh = cv2.threshold(license_plate_gray, 68, 255, cv2.THRESH_BINARY_INV)
                
                blurred = cv2.GaussianBlur(license_plate_gray, (25, 25), 0)
                grayscaleImage = license_plate_gray * ((license_plate_gray / blurred) > 0.01)  
                clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16,6))
                contrasted = clahe.apply(grayscaleImage)
                # _, license_plate_thresh2 = cv2.threshold(license_plate_gray, 68, 255, cv2.THRESH_BINARY_INV)
                
                # cv2.imshow("license_plate_thresh", license_plate_thresh)
                # cv2.imshow("contrasted", contrasted)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     print('checking break 0')
                #     break
                                
                license_plate_text, license_plate_text_conf_score = read_license_plate(contrasted)
                if car_id not in check_car_id:
                    check_car_id[car_id] = {"conf_score": conf_score, "license_plate_text": license_plate_text}
                elif check_car_id[car_id]["conf_score"] < conf_score and len(license_plate_text) > 6:
                    check_car_id[car_id]["license_plate_text"] = license_plate_text
                    check_car_id[car_id]["conf_score"] = conf_score
                # if license_plate_text is not None:
                # Update logic to check previous frame if current frame license plate text is 0
                # if license_plate_text == '0':
                #     if frame_number > 0 and car_id in results[frame_number - 1]:
                #         previous_frame_license_text = results[frame_number - 1][car_id]['license_plate']['license_text']
                #         license_plate_text = previous_frame_license_text
                #         print(f"Updated license plate text from previous frame: {license_plate_text}")
                # print("license_plate_text", license_plate_text)
                # if car_id not in memory:
                #     memory[car_id] = deque(maxlen=5)
                # memory[car_id].append(license_plate_text)
                # if len(memory[car_id]) == 5:
                    # consistent_license_plate = max(set(memory[car_id]), key=memory[car_id].count)
                results[frame_number][car_id] = {
                    'car': {'bbox': ' '.join(map(str, [xcar1, ycar1, xcar2, ycar2]))},
                    'license_plate': {
                        'bbox': ' '.join(map(str, [x1, y1, x2, y2])),
                        'license_plate_conf_score': check_car_id[car_id]["conf_score"],
                        'license_text': check_car_id[car_id]["license_plate_text"],
                        'license_plate_text_conf_score': license_plate_text_conf_score
                    }
                }
                # print('================================================================')
                # print(results)
                # print('================================================================')
        print('===========================================================================================================================')
        for license_plate in license_plates[0].boxes.data.tolist():
            x1, y1, x2, y2, conf_score, cls_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=8)
            cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Resized_Window", 900, 600)
            cv2.imshow("Resized_Window", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('checking break')
            break

    print(results)            
    write_csv(results, './test.csv')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()