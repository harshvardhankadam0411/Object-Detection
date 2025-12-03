# Object-Detection

#  Object Detection using Python & OpenCV

This project demonstrates real-time object detection using Python, OpenCV, and a pre-trained deep learning model. The system captures live video from the camera and detects multiple objects with bounding boxes and labels.

#  Features

Real-time video processing
Accurate object detection
Bounding boxes and class labels
Supports multiple object categories
Uses pre-trained deep learning model (YOLO)

#  Tech Stack

Python
OpenCV
NumPy
Pre-trained Deep Learning Model
(e.g YOLO)

#  Project Structure

/object-detection
  model/
    frozen_inference_graph.pb
    ssd_mobilenet_v3.pbtxt  (or your configuration file)
  object_detection.py
  requirements.txt
  README.md

#  How to Run

pip install -r requirements.txt
python object_detection.py

#  How It Works

Loads the pre-trained deep learning model
Reads frames from the webcam
Performs object detection
Draws bounding boxes with confidence score
Displays real-time output

#  Model Details

Model Type: (Add your model name Yolo)
Input Size: (e.g., 300×300, 416×416)
Framework: TensorFlow
Classes Supported: (e.g., 95 COCO classes)
