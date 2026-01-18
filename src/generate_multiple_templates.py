#!/usr/bin/env python3
"""
Generate Multiple Template Images for Ball and Book Detection
Creates several variations of each object with different angles, scales, and lighting.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_ball_templates(output_dir='../templates', count=3):
    """
    Generate multiple ball template variations.
    
    Args:
        output_dir: Directory to save templates
        count: Number of template variations to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, count + 1):
        # Base parameters with variations
        size = 800 + (i - 2) * 100  # 700, 800, 900
        center = (size // 2, size // 2)
        radius = int(size * 0.35)
        
        # Create blank canvas
        img = np.ones((size, size, 3), dtype=np.uint8) * 250
        
        # Gradient circle with varying intensity
        for r in range(radius, 0, -1):
            intensity_factor = 1.0 + (i - 2) * 0.1  # Vary brightness
            color_val = int((r / radius) * 180 * intensity_factor)
            color_val = min(255, max(0, color_val))
            color = (color_val, color_val // 2, 50)
            cv2.circle(img, center, r, color, -1)
        
        # Pentagon pattern with rotation
        num_points = 5
        angle_offset = (i - 1) * 30  # Rotate pattern for each template
        for j in range(num_points):
            angle1 = 2 * np.pi * j / num_points + np.radians(angle_offset)
            angle2 = 2 * np.pi * (j + 1) / num_points + np.radians(angle_offset)
            pt1 = (
                int(center[0] + radius * 0.6 * np.cos(angle1)),
                int(center[1] + radius * 0.6 * np.sin(angle1))
            )
            pt2 = (
                int(center[0] + radius * 0.6 * np.cos(angle2)),
                int(center[1] + radius * 0.6 * np.sin(angle2))
            )
            cv2.line(img, pt1, pt2, (0, 0, 0), 3)
        
        # Add texture lines with varying density
        line_density = 8 + i * 2  # More lines for later templates
        for angle in np.linspace(0, 2 * np.pi, line_density, endpoint=False):
            pt1 = (
                int(center[0] + radius * 0.3 * np.cos(angle)),
                int(center[1] + radius * 0.3 * np.sin(angle))
            )
            pt2 = (
                int(center[0] + radius * 0.9 * np.cos(angle)),
                int(center[1] + radius * 0.9 * np.sin(angle))
            )
            cv2.line(img, pt1, pt2, (100, 100, 100), 1)
        
        # Add small circles at different positions
        circle_positions = [(0.4, 0.4), (0.6, 0.4), (0.5, 0.6)]
        for cx, cy in circle_positions:
            pos = (int(center[0] + (cx - 0.5) * radius), 
                   int(center[1] + (cy - 0.5) * radius))
            cv2.circle(img, pos, 8, (255, 255, 255), -1)
            cv2.circle(img, pos, 8, (0, 0, 0), 2)
        
        # Add subtle noise for more keypoints
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save template
        filename = os.path.join(output_dir, f'ball_{i}.jpg')
        cv2.imwrite(filename, img)
        print(f"[CREATED] {filename} - size {size}x{size}")
    
    return count

def create_book_templates(output_dir='../templates', count=3):
    """
    Generate multiple book template variations.
    
    Args:
        output_dir: Directory to save templates
        count: Number of template variations to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, count + 1):
        # Base parameters with variations
        width = 800 + (i - 2) * 100  # 700, 800, 900
        height = int(width * 0.7)  # Maintain aspect ratio
        
        # Create PIL image for text rendering
        pil_img = Image.new('RGB', (width, height), color=(240, 240, 235))
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load a font, fallback to default
        try:
            title_font = ImageFont.truetype("arial.ttf", size=int(width * 0.08))
            text_font = ImageFont.truetype("arial.ttf", size=int(width * 0.05))
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Draw title with variation
        titles = ["GPU COMPUTING", "OBJECT DETECTION", "COMPUTER VISION"]
        title_text = titles[(i - 1) % len(titles)]
        draw.text((width * 0.1, height * 0.15), title_text, fill=(20, 20, 100), font=title_font)
        
        # Draw subtitle
        subtitles = ["Feature Matching", "Visual Recognition", "Image Analysis"]
        subtitle_text = subtitles[(i - 1) % len(subtitles)]
        draw.text((width * 0.1, height * 0.3), subtitle_text, fill=(50, 50, 50), font=text_font)
        
        # Draw author/group info
        draw.text((width * 0.1, height * 0.45), f"Group U - Template {i}", fill=(80, 80, 80), font=text_font)
        
        # Add some decorative text
        for idx, line in enumerate(["Machine Learning", "CUDA Programming", "Parallel Computing"]):
            y_pos = height * 0.6 + idx * height * 0.08
            draw.text((width * 0.1, y_pos), line, fill=(100, 100, 100), font=text_font)
        
        # Convert back to OpenCV format
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Add border with varying thickness
        border_thickness = 8 + i * 2
        cv2.rectangle(img, (20, 20), (width - 20, height - 20), (100, 50, 20), border_thickness)
        cv2.rectangle(img, (40, 40), (width - 40, height - 40), (150, 100, 50), 2)
        
        # Add geometric patterns with rotation
        pattern_offset = (i - 1) * 15
        for j in range(5):
            angle = j * 72 + pattern_offset  # Pentagon pattern
            x = int(width * 0.85 + width * 0.08 * np.cos(np.radians(angle)))
            y = int(height * 0.2 + width * 0.08 * np.sin(np.radians(angle)))
            cv2.circle(img, (x, y), 5, (200, 100, 50), -1)
        
        # Add corner decorations
        corner_size = 30 + i * 5
        corners = [
            (60, 60), (width - 60, 60),
            (60, height - 60), (width - 60, height - 60)
        ]
        for corner in corners:
            cv2.rectangle(img, 
                         (corner[0] - corner_size // 2, corner[1] - corner_size // 2),
                         (corner[0] + corner_size // 2, corner[1] + corner_size // 2),
                         (180, 130, 80), 2)
        
        # Add texture with varying intensity
        noise_intensity = 10 + i * 3
        noise = np.random.randint(-noise_intensity, noise_intensity, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save template
        filename = os.path.join(output_dir, f'book_{i}.jpg')
        cv2.imwrite(filename, img)
        print(f"[CREATED] {filename} - size {width}x{height}")
    
    return count

def verify_templates(template_dir='../templates'):
    """Verify feature quality of generated templates."""
    print("\n[VERIFICATION] Checking template feature quality...")
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Check all templates
    template_files = [f for f in os.listdir(template_dir) 
                     if f.endswith('.jpg') and (f.startswith('ball_') or f.startswith('book_'))]
    template_files.sort()
    
    for filename in template_files:
        filepath = os.path.join(template_dir, filename)
        img = cv2.imread(filepath)
        
        if img is None:
            print(f"[ERROR] Failed to load {filename}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        status = "✓" if len(keypoints) >= 50 else "✗"
        warning = "" if len(keypoints) >= 50 else " [WARNING: Low keypoint count]"
        print(f"{status} {filename}: {len(keypoints)} keypoints{warning}")

def create_test_images(output_dir='../test_images', template_dir='../templates'):
    """
    Create test images with balls and books at different positions/scales.
    
    Args:
        output_dir: Directory to save test images
        template_dir: Directory containing templates
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load templates
    ball_templates = [cv2.imread(os.path.join(template_dir, f'ball_{i}.jpg')) 
                      for i in range(1, 4)]
    book_templates = [cv2.imread(os.path.join(template_dir, f'book_{i}.jpg')) 
                      for i in range(1, 4)]
    
    # Create various test scenarios
    test_configs = [
        {'name': 'ball_only', 'objects': [('ball', 0, 0.7, (300, 250))]},
        {'name': 'book_only', 'objects': [('book', 0, 0.6, (400, 300))]},
        {'name': 'both_separated', 'objects': [
            ('ball', 0, 0.5, (200, 200)),
            ('book', 1, 0.5, (600, 300))
        ]},
        {'name': 'both_close', 'objects': [
            ('ball', 1, 0.4, (300, 200)),
            ('book', 0, 0.4, (450, 250))
        ]},
        {'name': 'multiple_balls', 'objects': [
            ('ball', 0, 0.4, (200, 200)),
            ('ball', 1, 0.3, (500, 150)),
            ('ball', 2, 0.35, (350, 350))
        ]},
    ]
    
    for config in test_configs:
        # Create canvas
        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 230
        
        # Add objects
        for obj_type, template_idx, scale, position in config['objects']:
            if obj_type == 'ball':
                template = ball_templates[template_idx]
            else:
                template = book_templates[template_idx]
            
            # Resize template
            new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
            resized = cv2.resize(template, new_size)
            
            # Place on canvas
            x, y = position
            h, w = resized.shape[:2]
            
            # Ensure it fits
            x = max(0, min(x, canvas.shape[1] - w))
            y = max(0, min(y, canvas.shape[0] - h))
            
            # Blend onto canvas
            canvas[y:y+h, x:x+w] = resized
        
        # Add noise for realism
        noise = np.random.randint(-10, 10, canvas.shape, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save test image
        filename = os.path.join(output_dir, f'test_{config["name"]}.jpg')
        cv2.imwrite(filename, canvas)
        print(f"[CREATED] {filename}")

def main():
    """Generate all templates and test images."""
    print("=" * 60)
    print("TEMPLATE GENERATOR - Multiple Variations")
    print("=" * 60)
    
    # Generate templates
    print("\n[STEP 1] Generating ball templates...")
    ball_count = create_ball_templates()
    
    print("\n[STEP 2] Generating book templates...")
    book_count = create_book_templates()
    
    # Verify quality
    verify_templates()
    
    # Generate test images
    print("\n[STEP 3] Generating test images...")
    create_test_images()
    
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Generated {ball_count} ball and {book_count} book templates")
    print("[SUCCESS] Generated 5 test images")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run detector: python object_detector_gpu.py --input ../test_images --templates ../templates")
    print("  2. Run benchmark: python benchmark.py --input ../test_images --templates ../templates")
    print("=" * 60)

if __name__ == '__main__':
    main()
