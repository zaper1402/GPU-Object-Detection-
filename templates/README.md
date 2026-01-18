# Template Images Directory

## Required Reference Images

Place the following reference images in this directory:

1. **ball.jpg** - Clear image of a ball (basketball, soccer ball, tennis ball, etc.)
   - Recommended size: 300x300 to 800x800 pixels
   - Good lighting, minimal background clutter
   - Object should occupy 60-80% of image area

2. **book.jpg** - Clear image of a book cover
   - Recommended size: 300x400 to 800x1000 pixels
   - Book cover should be flat and well-lit
   - Distinctive features (title, artwork) clearly visible

## Image Capture Guidelines

- Use high resolution (minimum 640x480)
- Avoid motion blur
- Ensure uniform lighting
- Capture from a perpendicular angle
- Use plain background for better feature extraction
- Multiple images per object can improve detection accuracy

## Generating Sample Templates

If you don't have template images, you can:

1. Download sample images from free stock photo sites (Unsplash, Pexels)
2. Use a smartphone camera to capture your own reference objects
3. Use OpenCV to extract objects from existing images

Example Python code to prepare templates:

```python
import cv2

# Load and resize image
img = cv2.imread('original.jpg')
resized = cv2.resize(img, (600, 600))
cv2.imwrite('ball.jpg', resized)
```
