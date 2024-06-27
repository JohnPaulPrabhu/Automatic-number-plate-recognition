import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

results = pd.read_csv('./test.csv')

# Load video
video_path = 'a.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_plate_conf_score'])
    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': results[(results['car_id'] == car_id) &
                                        (results['license_plate_conf_score'] == max_)]['license_text'].iloc[0]
    }
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                              (results['license_plate_conf_score'] == max_)]['frame_number'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_plate_conf_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop

frame_number = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_number += 1
    if ret:
        df_ = results[results['frame_number'] == frame_number]
        for row_indx in range(len(df_)):
            # Draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            # draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
            #             line_length_x=200, line_length_y=200)

            # Draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Crop license plate
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            # if license_crop is not None:
            H, W, _ = license_crop.shape
            print("fmandjlfn kld", license_crop.shape)
            car_top = int(car_y1) - 100
            car_center_x = int((car_x1 + car_x2) / 2)
            crop_start_y = car_top - H
            crop_start_x = car_center_x - W // 2

            try:
                # frame[crop_start_y:crop_start_y + H, crop_start_x:crop_start_x + W, :] = license_crop

                # Draw white background for text
                # text_bg_start_y = crop_start_y - 100
                # text_bg_end_y = crop_start_y
                # text_bg_start_x = crop_start_x
                # text_bg_end_x = crop_start_x + W

                # frame[text_bg_start_y:text_bg_end_y, text_bg_start_x:text_bg_end_x, :] = (255, 255, 255)

                # Put license plate text
                license_text = license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
                (text_width, text_height), _ = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                
                # text_x = car_center_x - text_width // 2
                # text_y = text_bg_start_y + text_height // 2
                print("==========================================================================================")
                # print(text_x, text_y, car_center_x, text_bg_start_y, text_width, text_height)
                print(license_text)
                print("==========================================================================================")
                cv2.putText(frame, license_text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # cv2.putText(img, "{}".format(int(bbox[4])), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_list[bbox[4]%79].tolist(), 2)

            except Exception as e:
                print(f"Error cropping and placing license plate: {e}")

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()
cap.release()
cv2.destroyAllWindows()
