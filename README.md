
# Door Open Close Prediction
author: Deepak Malik

## Overview
This project involves extracting frames from video data, building a machine learning model, and deploying the model for real-time predictions. The workflow is divided into three main scripts:

1. **Extracting_frames.py**: Extracts frames from video data.
2. **Model_building.py**: Builds and trains the machine learning model.
3. **Live_prediction.py**: Deploys the trained model for real-time predictions.

## Requirements
Ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn

You can install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Scripts

### 1. Extracting_frames.py
This script is responsible for extracting frames from video files. It reads the video input, processes it frame by frame, and saves the frames for further use.

**Usage:**
```bash
python Extracting_frames.py --input <video_path> --output <frames_directory>
```

### 2. Model_building.py
This script builds and trains a machine learning model using the extracted frames. The model is saved for deployment in the next step.

**Usage:**
```bash
python Model_building.py --data <frames_directory> --model <model_output_path>
```

### 3. Live_prediction.py
This script deploys the trained model for making real-time predictions. It takes live video input, processes each frame, and predicts the output using the trained model.

**Usage:**
```bash
python Live_prediction.py --model <model_path> --input <live_video_source>
```

## How to Run
1. **Extract Frames**: Run `Extracting_frames.py` with your video file to extract frames.
2. **Build Model**: Use `Model_building.py` to train your model on the extracted frames.
3. **Live Prediction**: Deploy the model using `Live_prediction.py` to make predictions on live video input.

## Data Used
1. **Video of door opening**: https://youtu.be/uXIQt824S2Q
2. **Test Video**: was made using concatinating this video with itself multiple times at different playseeds 
3. **Trimming of Video**: This video was trimmed, and only first 130 frames were considered. Post that the video was found irrelevant for the scope of this project. 

## Directories
1. **data**: this folder contains the three videos Door_Opening.mp4, Door_Opening_Trim.mp4 and Test Video.mp4 and a folder named FramesExtracted.
2. **FramesExtracted**: this folder is for saving the extracted frames, closed door frames are stored in ClosedDoor folder and OpenDoor folder for open door frames.
3. **model**: folder saves all the trained models.

## Feedback
Would love to here your feedback. Mail me at d98malik@gmail.com

