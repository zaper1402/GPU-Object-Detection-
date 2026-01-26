import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def extract_and_match(query_path, target_path, use_sift=True):
    # Load images
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    if query_img is None or target_img is None:
        print("Error loading images.")
        return

    # Initialize detector
    if use_sift:
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
        ratio_thresh = 0.75  # Lower ratio for SIFT
    else:
        detector = cv2.ORB_create(nfeatures=1000)
        norm_type = cv2.NORM_HAMMING
        ratio_thresh = 0.85  # Higher ratio for ORB

    # Detect keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(query_img, None)
    kp2, des2 = detector.detectAndCompute(target_img, None)

    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.")
        return

    # KNN Matching with ratio test
    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    print(f"Total good matches: {len(good_matches)}")

    # Initialize result image
    target_img_color = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
    localized = False

    # Only attempt homography if enough good matches
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            h, w = query_img.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            # Draw bounding box
            cv2.polylines(target_img_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            localized = True
        else:
            print("Homography computation failed.")
    else:
        print("Insufficient matches for homography.")

    # Save and display results
    if localized:
        result_filename = "localized_object.jpg"
        cv2.imwrite(result_filename, target_img_color)
        print(f"Result saved as {result_filename}")

        # Matplotlib display
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(target_img_color, cv2.COLOR_BGR2RGB))
        plt.title("Detected Object" if localized else "Detection Failed")
        plt.axis('off')
        plt.show()
    else:
        print("Object not localized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect new object by aligning scenery images.")
    parser.add_argument("scenery_path", help="Path to the scenery-only image.")
    parser.add_argument("scene_with_object_path", help="Path to the scenery image that contains the object.")
    args = parser.parse_args()
    extract_and_match(args.scene_with_object_path, args.scenery_path, use_sift=True)
