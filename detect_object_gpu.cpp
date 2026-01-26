#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3) {
        cout << "Usage: ./detect_object_gpu scenery.jpg scene_with_object.jpg [--orb]" << endl;
        return 1;
    }

    bool use_sift = true;
    if (argc > 3 && string(argv[3]) == "--orb")
        use_sift = false;

    // Enable OpenCL (AMD GPU on Mahti)
    ocl::setUseOpenCL(true);
    cout << "OpenCL available: " << ocl::haveOpenCL() << endl;
    cout << "OpenCL in use:     " << ocl::useOpenCL() << endl;

    // Load images (UMat → GPU)
    UMat scenery_gray, target_gray, target_color;
    imread(argv[1], IMREAD_GRAYSCALE).copyTo(scenery_gray);
    imread(argv[2], IMREAD_GRAYSCALE).copyTo(target_gray);
    imread(argv[2], IMREAD_COLOR).copyTo(target_color);

    if (scenery_gray.empty() || target_gray.empty()) {
        cerr << "Error loading images." << endl;
        return 1;
    }

    // Feature detector
    Ptr<Feature2D> detector;
    int norm_type;
    float ratio_thresh;

    if (use_sift) {
        detector = SIFT::create();
        norm_type = NORM_L2;
        ratio_thresh = 0.75f;
        cout << "Using SIFT" << endl;
    } else {
        detector = ORB::create(1500);
        norm_type = NORM_HAMMING;
        ratio_thresh = 0.85f;
        cout << "Using ORB" << endl;
    }

    // Detect features (CPU internally if needed)
    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;

    detector->detectAndCompute(scenery_gray, noArray(), kp1, des1);
    detector->detectAndCompute(target_gray, noArray(), kp2, des2);

    if (des1.empty() || des2.empty()) {
        cerr << "Descriptor extraction failed." << endl;
        return 1;
    }

    // Matcher
    BFMatcher matcher(norm_type);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(des1, des2, knn_matches, 2);

    vector<DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance)
            good_matches.push_back(m[0]);
    }

    cout << "Good matches: " << good_matches.size() << endl;
    if (good_matches.size() < 12) {
        cerr << "Not enough matches for homography." << endl;
        return 1;
    }

    // Prepare points
    vector<Point2f> src_pts, dst_pts;
    for (const auto& m : good_matches) {
        src_pts.push_back(kp1[m.queryIdx].pt);
        dst_pts.push_back(kp2[m.trainIdx].pt);
    }

    // Homography
    Mat inlier_mask;
    Mat H = findHomography(src_pts, dst_pts, RANSAC, 4.0, inlier_mask);

    int inliers = countNonZero(inlier_mask);
    cout << "Inliers: " << inliers << endl;

    if (H.empty() || inliers < 10) {
        cerr << "Homography failed or too few inliers." << endl;
        return 1;
    }

    // Warp scenery → target frame (GPU)
    UMat warped;
    warpPerspective(scenery_gray, warped, H, target_gray.size());

    // Difference pipeline (GPU)
    UMat diff, blur, mask;
    absdiff(target_gray, warped, diff);
    GaussianBlur(diff, blur, Size(5, 5), 0);
    threshold(blur, mask, 30, 255, THRESH_BINARY);

    UMat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    dilate(mask, mask, kernel, Point(-1, -1), 2);

    // Back to CPU for contours
    Mat mask_cpu = mask.getMat(ACCESS_READ);
    vector<vector<Point>> contours;
    findContours(mask_cpu, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int boxes = 0;
    for (const auto& c : contours) {
        if (contourArea(c) < 250)
            continue;

        Rect box = boundingRect(c);
        rectangle(target_color, box, Scalar(0, 0, 255), 2);
        boxes++;
    }

    imwrite("localized_object.jpg", target_color);
    imwrite("difference_mask.jpg", mask_cpu);

    cout << "Saved localized_object.jpg" << endl;
    cout << "Saved difference_mask.jpg" << endl;
    cout << "Detected objects: " << boxes << endl;

    return 0;
}
