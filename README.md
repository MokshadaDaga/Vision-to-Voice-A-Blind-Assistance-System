# Blind Assistance System

This project is a real-time object detection and text recognition system designed to assist visually impaired individuals. It leverages the YOLO (You Only Look Once) object detection model and Optical Character Recognition (OCR) using Tesseract to detect objects and text in a video stream and provide spoken feedback to the user.

## Features

- **Real-time Object Detection:** Detects and announces objects in a video feed using YOLOv3.
- **Distance Estimation:** Estimates the distance of detected objects from the camera.
- **Text Recognition (OCR):** Recognizes and reads text from the video stream using Tesseract OCR.
- **Speech Feedback:** Provides audio feedback using a text-to-speech engine.

## Dependencies

The project requires the following libraries:

- `opencv-python`
- `numpy`
- `pytesseract`
- `pyttsx3`
- `threading`
- `queue`

Additionally, you need:

- **YOLOv3 Weights and Config:**
  - Download the pre-trained YOLOv3 weights (`yolov3.weights`) from [here](https://pjreddie.com/media/files/yolov3.weights).
  - Download the YOLOv3 configuration file (`yolov3.cfg`) from [here](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg).
  
- **COCO Class Names:**
  - Download the `coco.names` file from [here](https://github.com/pjreddie/darknet/blob/master/data/coco.names).

- **Tesseract OCR:** 
  - Install Tesseract OCR. For instructions, visit [this link](https://github.com/tesseract-ocr/tesseract).

## Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
