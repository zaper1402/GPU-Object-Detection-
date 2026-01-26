import cv2
import numpy as np

def detect_object_in_video(query_img_path, video_path, output_path='output_video', use_sift=True):
    # Load query image
    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        print("Error: Query image not found!")
        return

    # Initialize SIFT or ORB detector
    if use_sift:
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create()
        norm = cv2.NORM_HAMMING

    # Detect keypoints and descriptors in query image
    kp1, des1 = detector.detectAndCompute(query_img, None)

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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{output_path}.mp4', fourcc, fps, (frame_width * 2, frame_height))  # Double width for matches

    # BFMatcher with appropriate norm
    bf = cv2.BFMatcher(norm, crossCheck=True)

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        target_img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = detector.detectAndCompute(target_img_gray, None)

        if des2 is None or len(kp2) == 0:
            img_matches = cv2.drawMatches(query_img, kp1, frame, kp2, [], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            out.write(img_matches)
            continue

        # Match descriptors
        matches = bf.match(des1, des2)
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
output_path = "./Output/main_video_detection/output_video"
use_sift = True  # Set to False for ORB

detect_object_in_video(query_img_path, video_path, output_path, use_sift)
