import cv2
import numpy as np
import os

def detect_object_in_video(query_img_path, video_path, output_path='output_video', use_sift=True, prefer_cuda=True):
    # Load query image
    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        print("Error: Query image not found!")
        return
    print(f"[INFO] prefer_cuda={prefer_cuda}, use_sift={use_sift}, query_shape={query_img.shape}")
    cuda_env = {k: os.environ.get(k, "") for k in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "LD_LIBRARY_PATH")}
    print(f"[INFO] CUDA env: {cuda_env}")
    try:
        build_info = cv2.getBuildInformation()
        cuda_lines = [ln.strip() for ln in build_info.splitlines() if any(k in ln for k in ("CUDA", "cuDNN", "NVIDIA", "NVCUVID", "PTX", "SVM", "Use Cuda"))]
        compiled_with_cuda = next((ln.split()[-1] for ln in cuda_lines if ln.lower().startswith("use cuda")), "UNKNOWN")
        print("[INFO] OpenCV CUDA build summary:")
        for ln in cuda_lines:
            print(f"        {ln}")
        print(f"[INFO] OpenCV compiled with CUDA: {compiled_with_cuda}")
        print(f"[INFO] OpenCV module path: {cv2.__file__}")
    except Exception as e:
        print(f"[WARN] Could not read OpenCV build information: {e}")

    norm = cv2.NORM_L2 if use_sift else cv2.NORM_HAMMING
    use_cuda = False
    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception as e:
        device_count = 0
        print(f"[WARN] cv2.cuda.getCudaEnabledDeviceCount failed: {e}")
    print(f"[INFO] CUDA device count reported by OpenCV: {device_count}")
    if device_count == 0:
        print("[INFO] No CUDA devices detected by OpenCV on this node.")
        print("[HINT] On Mahti, request a GPU node (e.g., srun --partition=gpusmall --gres=gpu:1 --time=1:00:00 --pty bash),")
        print("       load cuda/opencv modules inside that session, and ensure CUDA_VISIBLE_DEVICES is set correctly.")
        print("       If OpenCV was built without CUDA, install a CUDA-enabled build (e.g., pip install opencv-contrib-python-cu11).")

    if prefer_cuda and device_count > 0:
        try:
            cv2.cuda.setDevice(0)
            print(f"[INFO] CUDA device set to: {cv2.cuda.getDevice()}")
            detector = cv2.cuda.SIFT_create() if use_sift else cv2.cuda_ORB.create()
            bf = cv2.cuda.DescriptorMatcher_createBFMatcher(norm)
            use_cuda = True
            print("[INFO] Using CUDA-accelerated detector/matcher.")
        except Exception as e:
            print(f"[WARN] CUDA init failed, falling back to CPU: {e}")

    if not use_cuda:
        detector = cv2.SIFT_create() if use_sift else cv2.ORB_create()
        bf = cv2.BFMatcher(norm, crossCheck=True)
        print("[INFO] Using CPU detector/matcher.")

    if use_cuda:
        q_gpu = cv2.cuda_GpuMat()
        q_gpu.upload(query_img)
        kp1_gpu, des1 = detector.detectAndComputeAsync(q_gpu, None)
        kp1 = detector.convert(kp1_gpu)
        print(f"[INFO] Query descriptors (CUDA): {'none' if des1 is None else des1.shape}")
    else:
        kp1, des1 = detector.detectAndCompute(query_img, None)
        print(f"[INFO] Query descriptors (CPU): {'none' if des1 is None else des1.shape}")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video file not opened!")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{output_path}.mp4', fourcc, fps, (frame_width * 2, frame_height))  # Double width for matches

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_cuda:
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame)
            frame_gray_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)
            kp2_gpu, des2 = detector.detectAndComputeAsync(frame_gray_gpu, None)
            kp2 = detector.convert(kp2_gpu)
            if des2 is None or des2.empty() or len(kp2) == 0:
                img_matches = cv2.drawMatches(query_img, kp1, frame, kp2, [], None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                out.write(img_matches)
                continue
            matches = bf.match(des1, des2)
        else:
            target_img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp2, des2 = detector.detectAndCompute(target_img_gray, None)
            if des2 is None or len(kp2) == 0:
                img_matches = cv2.drawMatches(query_img, kp1, frame, kp2, [], None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                out.write(img_matches)
                continue
            matches = bf.match(des1, des2)

        # Match descriptors
        if len(matches) == 0:
            img_matches = cv2.drawMatches(query_img, kp1, frame, kp2, [], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            out.write(img_matches)
            continue

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]  # Use top 50 matches for homography

        # Compute homography if enough matches
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Transform query image corners to target image
                h, w = query_img.shape
                corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
                
                # Draw bounding box on the frame
                frame_with_rect = frame.copy()
                cv2.polylines(frame_with_rect, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
                # Draw top 10 matches
                img_matches = cv2.drawMatches(query_img, kp1, frame_with_rect, kp2, good_matches[:10], None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            else:
                img_matches = cv2.drawMatches(query_img, kp1, frame, kp2, good_matches[:10], None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        else:
            img_matches = cv2.drawMatches(query_img, kp1, frame, kp2, matches[:10], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Resize img_matches to fit the video writer dimensions if necessary
        if img_matches.shape[1] != frame_width * 2 or img_matches.shape[0] != frame_height:
            img_matches = cv2.resize(img_matches, (frame_width * 2, frame_height))
        
        out.write(img_matches)

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved to {output_path}.mp4")

# Example usage:
query_img_path = "./Images/object2.png"
video_path = "./Videos/traffictrim - Trim.mp4"
output_path = "output_video"
use_sift = True  # Set to False for ORB

detect_object_in_video(query_img_path, video_path, output_path, use_sift, prefer_cuda=True)
