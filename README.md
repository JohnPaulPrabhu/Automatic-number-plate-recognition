---
# Automatic Number Plate Recognition

This project uses computer vision and deep learning techniques to implement an Automatic Number Plate Recognition (ANPR) system. The goal is to detect and recognize vehicle number plates from images or video streams.


https://github.com/JohnPaulPrabhu/Automatic-number-plate-recognition/assets/26264448/a3f555f2-701f-4123-bb96-616d46eb4053



## Features

- Detects and recognizes number plates from images and video.
- Utilizes YOLOv10 for object detection.
- Uses EasyOCR for Optical Character Recognition (OCR).
- Implements Kalman filter for tracking.
- Supports real-time processing with video input.

## Requirements

- Python 3.7+
- OpenCV
- Numpy
- EasyOCR
- PyTorch
- Ultralytics YOLOv10
- Supervision library

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/JohnPaulPrabhu/Automatic-number-plate-recognition.git
   cd Automatic-number-plate-recognition
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Follow the instructions from [this repository](https://github.com/JohnPaulPrabhu/Object-Detection-On-Custom-Dataset.git) to train and set up the custom license plate recognition model.

## Usage

1. Place the video file you want to process in the project directory.

2. Run the main script:
   ```sh
   python main.py
   ```

3. The output will be displayed in a window and saved as a CSV file.

## Project Structure

- `main.py`: Main script to run the ANPR system.
- `util.py`: Utility functions for detection, recognition, and tracking.
- `visualize.py`: Script to visualize the final output.
- `requirements.txt`: List of required packages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [YOLOv10 by Ultralytics](https://github.com/ultralytics/yolov10)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Custom Object Detection Model](https://github.com/JohnPaulPrabhu/Object-Detection-On-Custom-Dataset.git)

---
