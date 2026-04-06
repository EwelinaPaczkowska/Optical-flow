# Aircraft Trajectory Tracking using Optical Flow

## Description
This project uses optical flow techniques to track and visualize the trajectory of an aircraft based on video input. 
It applies computer vision methods to detect and follow key points across video frames.

The system uses the Lucas-Kanade optical flow algorithm to estimate motion between consecutive frames and reconstruct the movement path of the tracked object.

## Features
- Detection of good feature points (Shi-Tomasi method)
- Tracking points using Lucas-Kanade optical flow
- Visualization of motion trajectories
- Real-time video processing
- Display of active tracked points

## Technologies
- Python
- OpenCV
- NumPy

## How it works
1. The video is loaded frame by frame.
2. Key feature points are detected in the first frame.
3. Optical flow is used to track these points in subsequent frames.
4. The movement of points is visualized as trajectories.
5. The result is displayed in real-time.

## Usage
Run the script with a video file:

```bash
python lab4_tracking_gotowe.py --video your_video.mp4
