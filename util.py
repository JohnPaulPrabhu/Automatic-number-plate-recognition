from PIL import Image
from typing import Tuple, List
import numpy as np


def write_csv(results: dict, otuput_path: str) -> None:
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

def get_car(license_plate: object, track_ids: List[np.ndarray]) -> Tuple[float, float, float, float, float]:
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
    return 1,1,1,1,1


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