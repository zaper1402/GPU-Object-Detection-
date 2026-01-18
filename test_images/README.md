# Test Images Directory

## Purpose

Place query images containing balls and/or books to test the object detection system.

## Generating Test Images

Run the provided script to automatically create test scenarios:

```bash
cd src
python generate_multiple_templates.py
```

This creates 5 test images in this directory:
- **test_ball_only.jpg** - Single ball
- **test_book_only.jpg** - Single book  
- **test_both_separated.jpg** - Ball and book far apart
- **test_both_close.jpg** - Ball and book close together
- **test_multiple_balls.jpg** - Multiple balls at different scales

## Test Image Requirements

- Format: JPG, PNG, BMP
- Resolution: 640x480 minimum, 1920x1080 recommended
- Objects should be visible and reasonably sized (at least 100x100 pixels)
- Can contain multiple objects (e.g., both ball and book)
- Can test different conditions:
  - Various lighting (bright, dim, shadows)
  - Different angles and perspectives
  - Partial occlusions
  - Scale variations (near/far objects)
  - Cluttered backgrounds

## Running Detection

```bash
cd src
# Single image
python object_detector_gpu.py --input ../test_images/test_both_separated.jpg --templates ../templates

# All images in directory
python object_detector_gpu.py --input ../test_images --templates ../templates
```

## Expected Output

Detection results will be saved to `../results/` with:
- Bounding boxes drawn around detected objects
- Labels indicating object type (ball/book)
- Template used (e.g., "ball using ball_2.jpg")
- Confidence scores
- Processing time metrics
