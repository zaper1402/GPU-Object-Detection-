import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _build_detector(use_sift: bool):
    if use_sift:
        return cv2.SIFT_create(), cv2.NORM_L2, 0.75
    return cv2.ORB_create(nfeatures=1500), cv2.NORM_HAMMING, 0.85


def detect_object_by_scene_diff(
    scenery_path: str,
    scene_with_object_path: str,
    use_sift: bool = True,
    min_matches: int = 12,
    min_inliers: int = 10,
    reproj_thresh: float = 4.0,
    diff_threshold: int = 30,
    min_contour_area: int = 250,
    display: bool = False,
):
    """Aligns scenery-only and scenery-with-object images, then diffs to localize the new object."""

    scenery_gray = cv2.imread(scenery_path, cv2.IMREAD_GRAYSCALE)
    target_gray = cv2.imread(scene_with_object_path, cv2.IMREAD_GRAYSCALE)
    target_color = cv2.imread(scene_with_object_path, cv2.IMREAD_COLOR)

    if scenery_gray is None or target_gray is None or target_color is None:
        print("Error loading images.")
        return

    detector, norm_type, ratio_thresh = _build_detector(use_sift)

    kp1, des1 = detector.detectAndCompute(scenery_gray, None)
    kp2, des2 = detector.detectAndCompute(target_gray, None)

    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.")
        return

    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    print(f"Total good matches: {len(good_matches)}")

    if len(good_matches) < min_matches:
        print("Not enough matches to align scenes.")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    inliers = int(mask.ravel().sum()) if mask is not None else 0

    if M is None or inliers < min_inliers:
        print("Homography computation failed or too few inliers.")
        return

    h, w = target_gray.shape
    warped_scenery = cv2.warpPerspective(scenery_gray, M, (w, h))

    diff = cv2.absdiff(target_gray, warped_scenery)
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, diff_mask = cv2.threshold(diff_blur, diff_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    diff_mask = cv2.dilate(diff_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        boxes.append((x, y, box_width, box_height))
        cv2.rectangle(
            target_color,
            (x, y),
            (x + box_width, y + box_height),
            (0, 0, 255),
            2,
        )

    cv2.imwrite("localized_object.jpg", target_color)
    cv2.imwrite("difference_mask.jpg", diff_mask)

    print(
        f"Saved localized_object.jpg with {len(boxes)} box(es); "
        f"difference_mask.jpg saved for debugging. Inliers: {inliers}"
    )

    if display:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(target_color, cv2.COLOR_BGR2RGB))
        plt.title("Localized Object")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(diff_mask, cmap="gray")
        plt.title("Difference Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


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
    parser.add_argument("--orb", action="store_true", help="Use ORB instead of SIFT.")
    parser.add_argument("--display", action="store_true", help="Display result and mask with matplotlib.")
    parser.add_argument("--diff-threshold", type=int, default=40, help="Pixel diff threshold for mask binarization.")
    parser.add_argument("--min-area", type=int, default=600, help="Minimum contour area to keep (in pixels).")
    parser.add_argument("--min-matches", type=int, default=12, help="Minimum good matches required for homography.")
    parser.add_argument("--min-inliers", type=int, default=10, help="Minimum inliers required for homography.")
    parser.add_argument("--reproj-thresh", type=float, default=4.0, help="RANSAC reprojection threshold.")

    args = parser.parse_args()

    detect_object_by_scene_diff(
        scenery_path=args.scenery_path,
        scene_with_object_path=args.scene_with_object_path,
        use_sift=not args.orb,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        reproj_thresh=args.reproj_thresh,
        diff_threshold=args.diff_threshold,
        min_contour_area=args.min_area,
        display=args.display,
    )
