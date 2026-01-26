import argparse
import time
import cv2
import numpy as np


"""
GPU-friendly variant of the scene-diff pipeline. Intended for Mahti:
- Mahti lacks a prebuilt OpenCV module; create a venv and install a CUDA-capable build.
  * CPU-only: pip install opencv-contrib-python-headless
  * GPU: use a wheel built with CUDA (e.g., from OpenCV forums) or build from source
    after loading CUDA on Mahti (e.g., module load cuda/11.8). If CUDA build is missing,
    this script gracefully falls back to CPU.
"""


def _build_detector(use_sift: bool, prefer_gpu: bool):
    cuda_ok = False
    if prefer_gpu:
        try:
            log()
            cuda_ok = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            cuda_ok = False

    if use_sift:
        if cuda_ok:
            return cv2.cuda.SIFT_create(), cv2.NORM_L2, 0.75, "cuda"
        return cv2.SIFT_create(), cv2.NORM_L2, 0.75, "cpu"

    if cuda_ok:
        return cv2.cuda.ORB_create(nfeatures=1500), cv2.NORM_HAMMING, 0.85, "cuda"
    return cv2.ORB_create(nfeatures=1500), cv2.NORM_HAMMING, 0.85, "cpu"


def _detect_and_compute(detector, gray_img, backend: str):
    if backend == "cuda":
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(gray_img)
        keypoints, descriptors = detector.detectAndComputeAsync(gpu_img, None)
        return keypoints, descriptors
    return detector.detectAndCompute(gray_img, None)


def _match_descriptors(des1, des2, norm_type: int, ratio_thresh: float, backend: str):
    if backend == "cuda":
        matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(norm_type)
        matches = matcher.knnMatch(des1, des2, k=2)
    else:
        matcher = cv2.BFMatcher(norm_type)
        matches = matcher.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < ratio_thresh * n.distance]


def _time_block():
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


def detect_object_by_scene_diff_gpu(
    scenery_path: str,
    scene_with_object_path: str,
    use_sift: bool = True,
    min_matches: int = 12,
    min_inliers: int = 10,
    reproj_thresh: float = 4.0,
    diff_threshold: int = 30,
    min_contour_area: int = 250,
    display: bool = False,
    prefer_gpu: bool = False,
):
    """Align two scenes, diff them, and localize new objects; GPU used when available."""

    t_total = _time_block()
    scenery_gray = cv2.imread(scenery_path, cv2.IMREAD_GRAYSCALE)
    target_gray = cv2.imread(scene_with_object_path, cv2.IMREAD_GRAYSCALE)
    target_color = cv2.imread(scene_with_object_path, cv2.IMREAD_COLOR)

    if scenery_gray is None or target_gray is None or target_color is None:
        print("Error loading images.")
        return

    detector, norm_type, ratio_thresh, backend = _build_detector(use_sift, prefer_gpu)
    print(f"Backend: {backend.upper()} | Detector: {'SIFT' if use_sift else 'ORB'}")

    t_detect = _time_block()
    kp1, des1 = _detect_and_compute(detector, scenery_gray, backend)
    kp2, des2 = _detect_and_compute(detector, target_gray, backend)
    detect_time = t_detect()

    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.")
        return

    t_match = _time_block()
    good_matches = _match_descriptors(des1, des2, norm_type, ratio_thresh, backend)
    match_time = t_match()

    print(f"Total good matches: {len(good_matches)} (detect: {detect_time:.3f}s, match: {match_time:.3f}s)")

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
        cv2.rectangle(target_color, (x, y), (x + box_width, y + box_height), (0, 0, 255), 2)

    cv2.imwrite("localized_object_gpu.jpg", target_color)
    cv2.imwrite("difference_mask_gpu.jpg", diff_mask)

    print(
        f"Saved localized_object_gpu.jpg with {len(boxes)} box(es); "
        f"difference_mask_gpu.jpg saved. Inliers: {inliers}. Total time: {t_total():.3f}s"
    )

    if display:
        import matplotlib.pyplot as plt

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


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-optional scene diff object detector.")
    parser.add_argument("scenery_path", help="Path to the scenery-only image.")
    parser.add_argument("scene_with_object_path", help="Path to the scenery image that contains the object.")
    parser.add_argument("--orb", action="store_true", help="Use ORB instead of SIFT.")
    parser.add_argument("--display", action="store_true", help="Display result and mask with matplotlib.")
    parser.add_argument("--diff-threshold", type=int, default=40, help="Pixel diff threshold for mask binarization.")
    parser.add_argument("--min-area", type=int, default=600, help="Minimum contour area to keep (pixels).")
    parser.add_argument("--min-matches", type=int, default=12, help="Minimum good matches required for homography.")
    parser.add_argument("--min-inliers", type=int, default=10, help="Minimum inliers required for homography.")
    parser.add_argument("--reproj-thresh", type=float, default=4.0, help="RANSAC reprojection threshold.")
    parser.add_argument("--gpu", action="store_true", help="Prefer CUDA if available; fallback to CPU.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_object_by_scene_diff_gpu(
        scenery_path=args.scenery_path,
        scene_with_object_path=args.scene_with_object_path,
        use_sift=not args.orb,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        reproj_thresh=args.reproj_thresh,
        diff_threshold=args.diff_threshold,
        min_contour_area=args.min_area,
        display=args.display,
        prefer_gpu=args.gpu,
    )
