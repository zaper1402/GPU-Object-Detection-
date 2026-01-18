# Test Images Directory

## Purpose

Place query images containing balls and/or books to test the object detection system.

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

## Expected Output

Detection results will be saved to `../results/` with:
- Bounding boxes drawn around detected objects
- Labels indicating object type (ball/book)
- Confidence scores
- Processing time metrics
