#!/usr/bin/env python3
"""
GPU-Accelerated Object Detection using Feature Matching
Detects balls and books in static images using OpenCV CUDA + ORB features

Author: Group U (Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha)
"""

import cv2
import numpy as np
import time
import os
import sys
from typing import Dict, List, Tuple, Optional


class GPUObjectDetector:
    """
    GPU-accelerated object detector using ORB feature matching and homography.
    Uses vector matching instead of neural networks for efficient object detection.
    """
    
    def __init__(self, template_dir: str, min_matches: int = 10, ratio_threshold: float = 0.75):
        """
        Initialize GPU object detector.
        
        Args:
            template_dir: Directory containing template images (ball.jpg, book.jpg)
            min_matches: Minimum good matches required for detection
            ratio_threshold: Lowe's ratio test threshold (0.75 recommended)
        """
        self.template_dir = template_dir
        self.min_matches = min_matches
        self.ratio_threshold = ratio_threshold
        self.templates = {}
        self.use_gpu = False
        
        # Initialize GPU detection
        self._init_gpu()
        
        # Load templates
        self._load_templates()
    
    def __del__(self):
        """Cleanup GPU resources."""
        if self.use_gpu:
            try:
                for obj_name in self.templates:
                    for template in self.templates[obj_name]:
                        if 'descriptors' in template and hasattr(template['descriptors'], 'release'):
                            template['descriptors'].release()
            except:
                pass  # Ignore cleanup errors during shutdown
    
    def _init_gpu(self):
        """Initialize CUDA GPU and create feature detector + matcher."""
        cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
        
        if cuda_device_count > 0:
            print(f"[INFO] CUDA-enabled devices found: {cuda_device_count}")
            
            # Get GPU information
            gpu_info = cv2.cuda.printCudaDeviceInfo(0)
            print(f"[INFO] Using GPU device 0")
            
            # Create GPU-accelerated ORB feature detector
            self.feature_detector = cv2.cuda_ORB.create(
                nfeatures=2000,      # Number of keypoints to retain
                scaleFactor=1.2,     # Pyramid decimation ratio
                nlevels=8,           # Number of pyramid levels
                edgeThreshold=31,    # Border size
                firstLevel=0,        # First pyramid level
                WTA_K=2,            # Points to produce BRIEF descriptor
                patchSize=31        # Patch size for descriptor
            )
            
            # Create GPU-accelerated Brute Force Matcher
            # NORM_HAMMING for binary ORB descriptors
            self.matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
            
            self.use_gpu = True
            print("[INFO] GPU acceleration ENABLED")
        else:
            print("[WARNING] No CUDA-enabled GPU found. Falling back to CPU.")
            print("[WARNING] Install OpenCV with CUDA support for GPU acceleration.")
            
            # CPU fallback
            self.feature_detector = cv2.ORB_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.use_gpu = False
            print("[INFO] CPU mode ENABLED")
    
    def _load_templates(self):
        """Load and process multiple template images for ball and book."""
        # Find all template files matching pattern: ball_*.jpg, book_*.jpg
        if not os.path.exists(self.template_dir):
            print(f"[ERROR] Template directory not found: {self.template_dir}")
            return
        
        all_files = os.listdir(self.template_dir)
        
        # Group templates by object type
        template_groups = {'ball': [], 'book': []}
        
        for filename in all_files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                if filename.startswith('ball_') or filename == 'ball.jpg':
                    template_groups['ball'].append(filename)
                elif filename.startswith('book_') or filename == 'book.jpg':
                    template_groups['book'].append(filename)
        
        # Sort for consistent ordering
        for obj_name in template_groups:
            template_groups[obj_name].sort()
        
        # Load each template
        for obj_name, filenames in template_groups.items():
            if len(filenames) == 0:
                print(f"[WARNING] No templates found for {obj_name}")
                continue
            
            # Store all template variations for this object
            self.templates[obj_name] = []
            
            for filename in filenames:
                filepath = os.path.join(self.template_dir, filename)
                
                # Load template image
                template_img = cv2.imread(filepath)
                if template_img is None:
                    print(f"[ERROR] Failed to load template: {filepath}")
                    continue
                
                # Convert to grayscale
                gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
                h, w = gray_template.shape
                
                # Extract features
                if self.use_gpu:
                    # GPU processing
                    gpu_template = cv2.cuda_GpuMat()
                    gpu_template.upload(gray_template)
                    
                    gpu_keypoints, gpu_descriptors = self.feature_detector.detectAndComputeAsync(
                        gpu_template, None
                    )
                    
                    # Download keypoints (descriptors stay on GPU)
                    keypoints = self.feature_detector.convert(gpu_keypoints)
                    
                    if gpu_descriptors.empty():
                        print(f"[WARNING] No features detected in {filename}")
                        continue
                    
                    template_data = {
                        'filename': filename,
                        'image': template_img,
                        'gray': gray_template,
                        'keypoints': keypoints,
                        'descriptors': gpu_descriptors,  # Keep on GPU
                        'corners': np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2),
                        'size': (w, h)
                    }
                else:
                    # CPU processing
                    keypoints, descriptors = self.feature_detector.detectAndCompute(
                        gray_template, None
                    )
                    
                    if descriptors is None or len(keypoints) == 0:
                        print(f"[WARNING] No features detected in {filename}")
                        continue
                    
                    template_data = {
                        'filename': filename,
                        'image': template_img,
                        'gray': gray_template,
                        'keypoints': keypoints,
                        'descriptors': descriptors,
                        'corners': np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2),
                        'size': (w, h)
                    }
                
                self.templates[obj_name].append(template_data)
                print(f"[INFO] Loaded template '{filename}': {len(keypoints)} keypoints, size {w}x{h}")
            
            if len(self.templates[obj_name]) == 0:
                del self.templates[obj_name]
            else:
                print(f"[INFO] Total {obj_name} templates: {len(self.templates[obj_name])}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in query image.
        
        Args:
            image: Input BGR image (numpy array)
        
        Returns:
            List of detection dictionaries containing:
                - object: Object name ('ball' or 'book')
                - bbox: Bounding box corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                - confidence: Detection confidence (0-1)
                - matches: Number of good matches
                - center: Center point (x, y)
        """
        if len(self.templates) == 0:
            print("[ERROR] No templates loaded. Cannot perform detection.")
            return []
        
        # Convert to grayscale
        gray_query = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract query features
        start_time = time.time()
        
        if self.use_gpu:
            # GPU feature extraction
            gpu_query = cv2.cuda_GpuMat()
            gpu_query.upload(gray_query)
            
            gpu_query_kp, gpu_query_desc = self.feature_detector.detectAndComputeAsync(
                gpu_query, None
            )
            
            query_keypoints = self.feature_detector.convert(gpu_query_kp)
            
            if gpu_query_desc.empty():
                print("[WARNING] No features detected in query image")
                return []
        else:
            # CPU feature extraction
            query_keypoints, query_descriptors = self.feature_detector.detectAndCompute(
                gray_query, None
            )
            
            if query_descriptors is None or len(query_keypoints) == 0:
                print("[WARNING] No features detected in query image")
                return []
        
        feature_time = time.time() - start_time
        
        # Match against each template (now supporting multiple templates per object)
        all_candidate_detections = []  # Store all candidates for NMS
        
        for obj_name, template_list in self.templates.items():
            # Try each template variation for this object
            for template in template_list:
                match_start = time.time()
                
                # Perform matching
                if self.use_gpu:
                    # GPU matching (k=2 for ratio test)
                    matches = self.matcher.knnMatch(
                        template['descriptors'],
                        gpu_query_desc,
                        k=2
                    )
                else:
                    # CPU matching
                    matches = self.matcher.knnMatch(
                        template['descriptors'],
                        query_descriptors,
                        k=2
                    )
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.ratio_threshold * n.distance:
                            good_matches.append(m)
                
                match_time = time.time() - match_start
                
                # Debug output
                print(f"  [DEBUG] Template: {template['filename']}, "
                      f"Total matches: {len(matches)}, Good matches: {len(good_matches)}, "
                      f"Min required: {self.min_matches}")
                
                # Check if enough matches for homography
                if len(good_matches) >= self.min_matches:
                    # Extract matched keypoint coordinates
                    src_pts = np.float32([
                        template['keypoints'][m.queryIdx].pt for m in good_matches
                    ]).reshape(-1, 1, 2)
                    
                    dst_pts = np.float32([
                        query_keypoints[m.trainIdx].pt for m in good_matches
                    ]).reshape(-1, 1, 2)
                    
                    # Compute homography with RANSAC
                    homography_start = time.time()
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    homography_time = time.time() - homography_start
                    
                    if M is not None:
                        # Transform template corners to query image
                        transformed_corners = cv2.perspectiveTransform(
                            template['corners'], M
                        )
                        
                        # Validate homography quality
                        corners_flat = transformed_corners.reshape(-1, 2)
                        area = cv2.contourArea(corners_flat)
                        
                        # Skip degenerate transformations
                        if area < 100:  # Too small, likely collapsed
                            continue
                        
                        # Calculate confidence (inlier ratio)
                        inliers = np.sum(mask)
                        confidence = inliers / len(good_matches)
                        
                        # Calculate center point
                        center = np.mean(corners_flat, axis=0)
                        
                        # Calculate area ratio (for filtering very small/large detections)
                        image_area = image.shape[0] * image.shape[1]
                        area_ratio = area / image_area
                        
                        # Filter unrealistic detections (relaxed upper bound to 0.95)
                        if 0.005 < area_ratio < 0.95 and confidence > 0.25:
                            detection = {
                                'object': obj_name,
                                'template': template['filename'],
                                'bbox': transformed_corners,
                                'confidence': confidence,
                                'matches': len(good_matches),
                                'inliers': inliers,
                                'center': tuple(center.astype(int)),
                                'area_ratio': area_ratio,
                                'area': area,
                                'timings': {
                                    'feature_extraction': feature_time,
                                    'matching': match_time,
                                    'homography': homography_time
                                }
                            }
                            all_candidate_detections.append(detection)
        
        # Apply Non-Maximum Suppression to remove overlapping detections
        detections = self._apply_nms(all_candidate_detections, iou_threshold=0.5)
        
        # Print final detections
        for detection in detections:
            print(f"[DETECT] {detection['object']} (using {detection['template']}): "
                  f"{detection['matches']} matches, {detection['inliers']} inliers, "
                  f"confidence {detection['confidence']:.2f}")
        
        return detections
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1, bbox2: Bounding box corners as numpy arrays [4,1,2]
        
        Returns:
            IoU score (0-1)
        """
        # Convert to polygon format
        poly1 = bbox1.reshape(-1, 2).astype(np.float32)
        poly2 = bbox2.reshape(-1, 2).astype(np.float32)
        
        # Calculate areas
        area1 = cv2.contourArea(poly1)
        area2 = cv2.contourArea(poly2)
        
        if area1 <= 0 or area2 <= 0:
            return 0.0
        
        # Find intersection (approximate using bounding rectangles for efficiency)
        rect1 = cv2.boundingRect(poly1)
        rect2 = cv2.boundingRect(poly2)
        
        # Calculate intersection rectangle
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
        y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to filter overlapping detections.
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression (0.5 recommended)
        
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        # Sort by confidence (descending)
        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections_sorted) > 0:
            # Take highest confidence detection
            current = detections_sorted.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            filtered = []
            for det in detections_sorted:
                iou = self._calculate_iou(current['bbox'], det['bbox'])
                
                # Keep if IoU is below threshold OR different object type
                if iou < iou_threshold or det['object'] != current['object']:
                    filtered.append(det)
            
            detections_sorted = filtered
        
        return keep
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image
            detections: List of detections from detect()
        
        Returns:
            Annotated image
        """
        result = image.copy()
        
        # Color map for different objects
        colors = {
            'ball': (0, 255, 0),    # Green
            'book': (255, 0, 0)     # Blue
        }
        
        for detection in detections:
            obj_name = detection['object']
            bbox = detection['bbox']
            confidence = detection['confidence']
            matches = detection['matches']
            center = detection['center']
            
            # Draw bounding box
            color = colors.get(obj_name, (0, 255, 255))
            pts = bbox.reshape(-1, 2).astype(np.int32)
            cv2.polylines(result, [pts], True, color, 3)
            
            # Draw label
            label = f"{obj_name.upper()}: {confidence:.2f} ({matches} matches)"
            label_pos = tuple(pts[0])
            
            # Background for label
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                result,
                label_pos,
                (label_pos[0] + label_w, label_pos[1] - label_h - baseline),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                result,
                label,
                (label_pos[0], label_pos[1] - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw center point
            cv2.circle(result, center, 5, color, -1)
        
        return result
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Tuple[List[Dict], float]:
        """
        Process a single image and optionally save result.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save annotated result
        
        Returns:
            Tuple of (detections, total_time)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            return [], 0.0
        
        print(f"\n[INFO] Processing: {os.path.basename(image_path)}")
        print(f"[INFO] Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Detect objects
        start_time = time.time()
        detections = self.detect(image)
        total_time = time.time() - start_time
        
        print(f"[INFO] Detection time: {total_time*1000:.2f} ms")
        print(f"[INFO] Found {len(detections)} object(s)")
        
        # Visualize and save
        if output_path:
            result = self.visualize_detections(image, detections)
            cv2.imwrite(output_path, result)
            print(f"[INFO] Saved result to: {output_path}")
        
        return detections, total_time


def main():
    """Main function for testing object detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Object Detection using Feature Matching')
    parser.add_argument('--templates', type=str, default='../templates',
                        help='Directory containing template images')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default='../results',
                        help='Output directory for results')
    parser.add_argument('--min-matches', type=int, default=10,
                        help='Minimum matches for detection')
    parser.add_argument('--ratio', type=float, default=0.75,
                        help='Lowe\'s ratio test threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    print("="*60)
    print("GPU Object Detector - Feature Matching")
    print("="*60)
    
    detector = GPUObjectDetector(
        template_dir=args.templates,
        min_matches=args.min_matches,
        ratio_threshold=args.ratio
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        output_path = os.path.join(
            args.output,
            f"result_{os.path.basename(args.input)}"
        )
        detections, proc_time = detector.process_image(args.input, output_path)
        
        if len(detections) > 0:
            print("\n[SUCCESS] TEST PASSED - Objects detected successfully")
        else:
            print("\n[INFO] No objects detected in image")
    
    elif os.path.isdir(args.input):
        # Batch processing
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [
            f for f in os.listdir(args.input)
            if f.lower().endswith(image_extensions)
        ]
        
        total_time = 0.0
        total_detections = 0
        
        for img_file in image_files:
            input_path = os.path.join(args.input, img_file)
            output_path = os.path.join(args.output, f"result_{img_file}")
            
            detections, proc_time = detector.process_image(input_path, output_path)
            total_time += proc_time
            total_detections += len(detections)
        
        # Summary
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Images processed: {len(image_files)}")
        print(f"Total detections: {total_detections}")
        print(f"Average time per image: {(total_time/len(image_files))*1000:.2f} ms")
        print(f"Throughput: {len(image_files)/total_time:.2f} images/sec")
        print("[SUCCESS] TEST PASSED - Batch processing completed")
    
    else:
        print(f"[ERROR] Invalid input path: {args.input}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
