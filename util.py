from PIL import Image
from typing import Tuple, List
import numpy as np
import csv


def write_csv(results: dict, output_path: str) -> None:
    """
    This method stores the results in a CSV file.

    Parameters
    ----------
    results : dict
        Results containing the bbox, confidence score of the license plate and car.
    otuput_path : str
        Directory path to store the CSV file.

    Returns
    -------
    This method does not return anything, just creates the CSV file

    """   
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
    """
    Retrieve vehicle coordinates and ID based on the license plate coordinates.

    Parameters
    ----------
    license_plate : object -> Need to change according to the original function call.
        DESCRIPTION.
    track_ids : List[np.ndarray]
        List of vehicle track IDs and their correspoding coordinates.

    Returns
    -------
    Tuple[float, float, float, float, float]
        Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.

    """
    x1, y1, x2, y2 = license_plate
    for track_id in track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track_id
        
        if x1 > xcar1 and y1 > ycar1 and x2 > xcar2 and y2 > ycar2:
            return track_id
    return -1, -1, -1, -1, -1


def read_license_plate(license_plate_cropped_img: Image) -> Tuple[str, float]:
    """
    Reads the licesnse plate text from the given crooped image.

    Parameters
    ----------
    license_plate_cropped_img : Image
        Cropped image containing the lincense plate

    Returns
    -------
    Tuple[str, float]
        Tuple containing the formatted text and its confidence score

    """
    
    return  0, 0