# Template Images Directory

## Multiple Template Support

This project supports multiple template variations per object for improved detection accuracy.

### Template Naming Convention

Templates should be named following this pattern:
- **Ball templates**: `ball_1.jpg`, `ball_2.jpg`, `ball_3.jpg`, ...
- **Book templates**: `book_1.jpg`, `book_2.jpg`, `book_3.jpg`, ...

Legacy single templates (`ball.jpg`, `book.jpg`) are also supported.

### Generating Templates

Run the provided script to create 3 variations of each object:

```bash
cd src
python generate_multiple_templates.py
```

This creates:
- `ball_1.jpg`, `ball_2.jpg`, `ball_3.jpg` (different sizes, rotations, lighting)
- `book_1.jpg`, `book_2.jpg`, `book_3.jpg` (different text, patterns, decorations)
- Test images in `../test_images/`

### Required Reference Images

Place reference images in this directory:

1. **ball_X.jpg** - Clear images of a ball (basketball, soccer ball, tennis ball, etc.)
   - Recommended size: 700x700 to 900x900 pixels
   - Good lighting, minimal background clutter
   - Object should occupy 60-80% of image area
   - Create 2-3 variations with different angles/lighting

2. **book_X.jpg** - Clear images of a book cover
   - Recommended size: 700x490 to 900x630 pixels
   - Book cover should be flat and well-lit
   - Distinctive features (title, artwork) clearly visible
   - Create 2-3 variations with different text/patterns

## How Multi-Template Detection Works

1. **Loading**: Detector finds all `ball_*.jpg` and `book_*.jpg` files
2. **Matching**: Tests query image against ALL template variations
3. **Selection**: Chooses best match based on confidence score
4. **Output**: Shows which template (`ball_2.jpg`) was used

Example output:
```
[INFO] Total ball templates: 3
[INFO] Total book templates: 3
[DETECT] ball (using ball_2.jpg): 142 matches, confidence 0.69
[DETECT] book (using book_1.jpg): 87 matches, confidence 0.70
```

## Image Capture Guidelines

- Use high resolution (minimum 640x480, prefer 800x800)
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
