# Mald-detector
Final project for CSE 455 - Computer Vision. For more detailed information on the project, see the [writeup website](https://sites.google.com/view/malddetector/home).

## How it works
Images are taken from the device's webcam and run through an MTCNN to find the bounding-box of any faces present in the image. The webcam image is then cropped to the dimensions returned, grayscaled, and sampled down to 48x48 pixels and run through a shallow CNN for emotion detection. Details on this network can be seen in the image below:

![alt text](https://github.com/jspaniac/Mald-detector/blob/main/cnn.png?raw=true)
This model achieved a 70.0% testing accuracy on positive/negative emotions and 45.6% on all 7 emotions, so results should be taken with a grain of salt.

A determination is then made on whether the percieved emotion is positive (happy, neutral) or negative (angry, disgusted, fearful, suprised) and displayed accordingly. The overall percentage that the user is displaying positive emotions is also printed to the console for use in future projects

## Usage
First, download the [emotion dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer) from kaggle. The train and test directories should be placed in ./root/data/emotion

Then, activating the application is as simple as:
```bash
python tryproject.py
```
After training the model, the display should appear
