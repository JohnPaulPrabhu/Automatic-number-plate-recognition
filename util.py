from PIL import Image
from typing import Tuple, List
import numpy as np
import csv
import easyocr
import string
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

reader = easyocr.Reader(['en'], gpu=True,)

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def write_csv(results: dict, output_path: str) -> None:
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_number', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_conf_score', 'license_text', 'license_plate_text_conf_score'])
        for frame_number, car_id_dict in results.items():
            for car_id, data in car_id_dict.items():
                car_bbox = data['car']['bbox']
                license_plate = data['license_plate']
                writer.writerow([frame_number, car_id, car_bbox, license_plate['bbox'], license_plate['license_plate_conf_score'], license_plate['license_text'], license_plate['license_plate_text_conf_score']])

def write_csv(results: dict, output_path: str) -> None:
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_number', 'car_id', 'car_bbox', 'license_plate_bbox','license_plate_conf_score', 'license_text', 'license_plate_text_conf_score'])
        
        # iterate over the dictionary and write each row
        for frame_number, car_id_dict in results.items():
            for car_id, data in car_id_dict.items():
                car_bbox = data['car']['bbox']
                license_plate = data['license_plate']
                writer.writerow([frame_number, car_id, car_bbox, license_plate['bbox'], license_plate['license_plate_conf_score'], license_plate['license_text'], license_plate['license_plate_text_conf_score']])


def get_car(license_plate: tuple, track_ids: List[np.ndarray]) -> List[float]:
    x1, y1, x2, y2 = license_plate
    for track_id in track_ids:
        xcar1, ycar1, xcar2, ycar2 = track_id[:4]
        car_id = track_id[4]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1

def read_license_plate(license_plate_cropped_img: Image) -> Tuple[str, float]:
    print("Checking read license plate")
    # detections = reader.readtext(license_plate_cropped_img, decoder=easyocr.GreedyDecoder(max_length=7))
    detections = reader.readtext(license_plate_cropped_img, allowlist=('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'), text_threshold=0.8,width_ths=1.5)
    print("detections", detections)
    text = ''
    score = 0
    for detection in detections:
        # print("detection", detection)
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        # print("text", text)
        if check_license_format(text):
            # print("checking if statement")
            return (check_format(text), score)
    return (text, score)

def check_license_format(text: str) -> bool:
    if len(text) != 7:
        return False
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_int_to_char.values()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_int_to_char.values()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
            return True
    return False

def check_format(text):
    license_plate_result = ''
    for idx, txt in enumerate(text):
        if (idx == 2 or idx == 3):
            if txt not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                license_plate_result += dict_char_to_int[txt]
                continue
        elif (txt in dict_int_to_char.keys()):
            license_plate_result += dict_int_to_char[txt]
            continue
        license_plate_result += txt
    return license_plate_result

def iou(box1, box2):
    x1, y1, x2, y2, _ = box1
    x1_p, y1_p, x2_p, y2_p, _ = box2
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def assign_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(matched_indices).T
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = bbox[:4].reshape((4, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox)

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        return self.kf.x[:4].reshape((1, 4))