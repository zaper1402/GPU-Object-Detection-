#!/usr/bin/env python3
"""
Sample template generator for testing
Creates synthetic ball and book templates if none exist

Author: Group U
"""

import cv2
import numpy as np
import os

def generate_ball_template(size=400):
    """Generate a synthetic ball template."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Draw circle (ball)
    center = (size // 2, size // 2)
    radius = size // 3
    
    # Add gradient for 3D effect
    for r in range(radius, 0, -5):
        intensity = int(255 * (1 - r / radius))
        color = (30 + intensity, 100 + intensity // 2, 200 - intensity // 3)
        cv2.circle(img, center, r, color, -1)
    
    # Add highlights
    highlight_center = (center[0] - radius // 3, center[1] - radius // 3)
    cv2.circle(img, highlight_center, radius // 4, (240, 240, 250), -1)
    cv2.circle(img, highlight_center, radius // 6, (255, 255, 255), -1)
    
    # Add texture (seams like basketball)
    cv2.line(img, (center[0] - radius, center[1]), 
             (center[0] + radius, center[1]), (50, 50, 50), 2)
    cv2.ellipse(img, center, (radius, radius // 2), 0, 0, 180, (50, 50, 50), 2)
    cv2.ellipse(img, center, (radius, radius // 2), 0, 180, 360, (50, 50, 50), 2)
    
    return img

def generate_book_template(size=(400, 300)):
    """Generate a synthetic book template."""
    width, height = size
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw book cover with border
    book_color = (180, 100, 50)  # Blue-ish book
    cv2.rectangle(img, (20, 20), (width-20, height-20), book_color, -1)
    cv2.rectangle(img, (20, 20), (width-20, height-20), (0, 0, 0), 3)
    
    # Add title text
    title = "SAMPLE BOOK"
    font = cv2.FONT_HERSHEY_BOLD
    text_size = cv2.getTextSize(title, font, 1.2, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 3
    
    cv2.putText(img, title, (text_x, text_y), font, 1.2, (255, 255, 255), 2)
    
    # Add author text
    author = "by AI Generator"
    text_size2 = cv2.getTextSize(author, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    text_x2 = (width - text_size2[0]) // 2
    text_y2 = text_y + 40
    cv2.putText(img, author, (text_x2, text_y2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Add decorative elements
    cv2.rectangle(img, (40, height - 80), (width - 40, height - 40), 
                  (220, 220, 220), 2)
    cv2.putText(img, "GPU Computing", (60, height - 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    # Add spine shadow
    cv2.rectangle(img, (20, 20), (40, height-20), (100, 50, 30), -1)
    
    return img

def generate_sample_test_image():
    """Generate a test image with both ball and book."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 220
    
    # Add ball
    ball = generate_ball_template(250)
    ball_h, ball_w = ball.shape[:2]
    
    # Place ball (with rotation and scale)
    center = (200, 200)
    angle = 30
    scale = 0.8
    M = cv2.getRotationMatrix2D((ball_w // 2, ball_h // 2), angle, scale)
    ball_rotated = cv2.warpAffine(ball, M, (ball_w, ball_h), 
                                   borderValue=(220, 220, 220))
    
    y1, y2 = center[1] - ball_h // 2, center[1] + ball_h // 2
    x1, x2 = center[0] - ball_w // 2, center[0] + ball_w // 2
    
    if y1 >= 0 and y2 <= 600 and x1 >= 0 and x2 <= 800:
        mask = cv2.cvtColor(ball_rotated, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)[1]
        img[y1:y2, x1:x2] = cv2.bitwise_and(img[y1:y2, x1:x2], 
                                             img[y1:y2, x1:x2], mask=~mask)
        img[y1:y2, x1:x2] = cv2.add(img[y1:y2, x1:x2], ball_rotated, mask=mask)
    
    # Add book
    book = generate_book_template((300, 200))
    book_h, book_w = book.shape[:2]
    
    center_book = (550, 350)
    angle_book = -15
    scale_book = 0.9
    M_book = cv2.getRotationMatrix2D((book_w // 2, book_h // 2), angle_book, scale_book)
    book_rotated = cv2.warpAffine(book, M_book, (book_w, book_h),
                                   borderValue=(220, 220, 220))
    
    y1, y2 = center_book[1] - book_h // 2, center_book[1] + book_h // 2
    x1, x2 = center_book[0] - book_w // 2, center_book[0] + book_w // 2
    
    if y1 >= 0 and y2 <= 600 and x1 >= 0 and x2 <= 800:
        mask_book = cv2.cvtColor(book_rotated, cv2.COLOR_BGR2GRAY)
        mask_book = cv2.threshold(mask_book, 250, 255, cv2.THRESH_BINARY_INV)[1]
        img[y1:y2, x1:x2] = cv2.bitwise_and(img[y1:y2, x1:x2],
                                             img[y1:y2, x1:x2], mask=~mask_book)
        img[y1:y2, x1:x2] = cv2.add(img[y1:y2, x1:x2], book_rotated, mask=mask_book)
    
    return img

def main():
    """Generate sample templates and test images."""
    print("="*60)
    print("Generating Sample Templates and Test Images")
    print("="*60)
    
    # Create directories
    templates_dir = '../templates'
    test_dir = '../test_images'
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate ball template
    ball_path = os.path.join(templates_dir, 'ball.jpg')
    if not os.path.exists(ball_path):
        print(f"\nGenerating ball template...")
        ball_img = generate_ball_template(400)
        cv2.imwrite(ball_path, ball_img)
        print(f"  ✓ Saved: {ball_path}")
    else:
        print(f"\n  ℹ Ball template already exists: {ball_path}")
    
    # Generate book template
    book_path = os.path.join(templates_dir, 'book.jpg')
    if not os.path.exists(book_path):
        print(f"Generating book template...")
        book_img = generate_book_template((400, 300))
        cv2.imwrite(book_path, book_img)
        print(f"  ✓ Saved: {book_path}")
    else:
        print(f"  ℹ Book template already exists: {book_path}")
    
    # Generate sample test image
    test_path = os.path.join(test_dir, 'sample_test.jpg')
    print(f"\nGenerating sample test image...")
    test_img = generate_sample_test_image()
    cv2.imwrite(test_path, test_img)
    print(f"  ✓ Saved: {test_path}")
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review generated templates in templates/")
    print("2. Add your own templates or test images if desired")
    print("3. Run detection:")
    print("   python object_detector_gpu.py --input ../test_images/sample_test.jpg --templates ../templates")
    print("="*60)

if __name__ == '__main__':
    main()
