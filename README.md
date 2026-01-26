# Object Detection using SIFT and ORB with OpenCV

This project demonstrates how to detect and localize a specific object in both images and videos using feature detection and matching techniques. The keypoint detection methods used are **SIFT (Scale-Invariant Feature Transform)** and **ORB (Oriented FAST and Rotated BRIEF)**, both available in the OpenCV library.

The project showcases how to extract key features from an object image and find matches in a larger scene or video stream using feature descriptors and homography estimation.

## Project Overview

This project contains two main scripts:

1. `main_image_matching.py`: Detects an object in a static image using feature matching.
2. `main_video_detection.py`: Detects the same object in a continuous video stream frame-by-frame.

Each script uses either SIFT or ORB to extract features and match keypoints. Homography is then computed using the matched points to find the object's position in the target image or video frame.

## Project Directory Structure

```
Object-Detection-with-SIFT-ORB/
│
├── main_image_matching.py        # Python script for image detection
├── main_video_detection.py       # Python script for video detection
├── requirements.txt              # List of required Python packages
│
├── images/
│   ├── object.png                # The reference image of the object to detect
│   └── scene.jpg                 # The image in which to detect the object
│
├── videos/
│   └── traffic.mp4               # A sample video file used for object detection
│
├── results/
│   ├── localized_object.jpg      # Output image showing detected object
│   └── output_video.mp4          # Output video with detection results
```

## Installation Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/Object-Detection-with-SIFT-ORB.git
   cd Object-Detection-with-SIFT-ORB
   ```

2. **Install the Required Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   .v 
   ```

## How to Use

### 1. Detect an Object in a Static Image

To run the image detection script:
```bash
python main_image_matching.py
```

This will:
- Load the object image and the target scene image.
- Detect keypoints and descriptors using either SIFT or ORB.
- Match descriptors using a brute-force matcher.
- Use RANSAC to compute a homography from matched keypoints.
- Draw a bounding polygon around the detected object and save the result.

### 2. Detect an Object in a Video

To run object detection on a video file:
```bash
python main_video_detection.py
```

This will:
- Load a video and the reference object image.
- Process each frame of the video.
- Apply feature matching and homography in real-time.
- Draw the detected object in each frame if found.
- Save the output video with the drawn bounding box.

## Feature Detection Algorithms

### SIFT (Scale-Invariant Feature Transform)
- Detects distinctive features that are invariant to scale, rotation, and illumination changes.
- More accurate but slower than ORB.
- Requires the `opencv-contrib-python` package due to licensing restrictions.

### ORB (Oriented FAST and Rotated BRIEF)
- Faster and suitable for real-time applications.
- Free to use (no licensing restrictions).
- Works well in many practical use-cases though it is less accurate than SIFT.

You can switch between the two by changing the `detector_type` variable in the code:
```python
detector_type = 'SIFT'  # or 'ORB'
```

## Requirements

- Python 3.7 or higher
- OpenCV (`opencv-contrib-python` for SIFT)
- NumPy
- Optionally: Matplotlib (for additional plotting)

Install via pip:
```bash
pip install opencv-contrib-python numpy matplotlib
```

## Output

- Detected object in a static image will be saved as: `results/localized_object.jpg`
- Processed video with bounding box overlay will be saved as: `results/output_video.mp4`

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenCV documentation and tutorials for inspiration.
- Built as a demonstration of traditional computer vision techniques without relying on deep learning.
