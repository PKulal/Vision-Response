
<div align="center">
  <h1 align="center">Vision Response: Empowering the Blind with Computer Vision</h1>  
    <img src="https://youaskweanswer.net/wp-content/uploads/2020/08/blind-man.gif" alt="Empowering the Blind with Computer Vision" />
    
</div>

## Overview

It is a computer vision-based machine learning project designed to empower individuals with visual impairments. The project focuses on Object Detection, Recognition, and Spatial Localization in indoor spaces, providing audio output to convey information about the surroundings.

## Demo

**Object Detection** 
- Detects all the objects that comes in the frame.
- Gives audio output for object name, confidence, object height in pixels and distance in metres.


**Object Location**
- Asks us what object we are looking for.
- Announces when that object comes into the frame.


## Table of Contents

- [Project Files](#project-files)
- [Getting Started](#getting-started)

## Project Files

1. **object_detection.py**: This file is responsible for detecting and identifying objects within the camera frame using the YOLOv3 model. It leverages the coco.names file to identify a variety of objects.

2. **object_finder.py**: Users can input the name of a specific object, and this file will respond when that particular object is detected within the camera frame. This enhances the user's ability to locate specific items in their environment.

3. **coco.names**: This file contains the names of objects that the YOLOv3 model is trained to identify. It serves as a reference for the object detection process.

4. **yolov3.weights**: Due to its large size, this file isn't included in the repository. Users need to download it from the internet to use it in conjunction with the YOLOv3 model.

5. **yolov8.tflite**: A converted version of YOLOv8 to TensorFlow Lite format. This enables integration with mobile applications. Work on developing a mobile app using Flutter is in progress.

## Getting Started

1. **Download yolov3.weights**: Download the `yolov3.weights` file from the internet and place it in the project directory.

2. **Install Dependencies**: Ensure you have the required dependencies installed. You may need to install libraries such as OpenCV, TensorFlow, and others as specified in the project files.

3. **Run Object Detection**: Execute the `object_detection.py` file to experience object detection using the YOLOv3 model.

4. **Run Object Finder**: Utilize the `object_finder.py` file to locate specific objects within the camera frame.

Feel free to contribute and make VisionAssist even more accessible for individuals with visual impairments!
