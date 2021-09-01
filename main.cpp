/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	The bachelor's thesis project, 
 * 	3D localization and 3D surface reconstruction
 * 	software for microprocessors
 */

#include <cstdint>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "src/constants.h"
#include "src/camera_module.h"
#include "src/features.h"
#include "src/debug_functions.h"

using namespace k3d;


int main()
{
	std::cout << "begin\n";

	// Camera cam(2);
 //    cam.set_resolution(1280, 720);

	// const cv::Mat frame = cam.get_frame_cv();

 //    cv::imwrite("../assets/test_img_2.png", frame);

    const cv::Mat frame1 = cv::imread("../assets/test_img_1.png");
    const cv::Mat frame2 = cv::imread("../assets/test_img_2.png");

    const auto [keypoints1, desc1] = detect_features_orb(frame1);
    const auto [keypoints2, desc2] = detect_features_orb(frame2);

    const std::vector<std::pair<uint32_t, uint32_t>> kp_matches = 
        match_features_bf_crosscheck(desc1, desc2);

    const auto kp_matches_filtered =
        radius_distance_filter_matches(kp_matches, keypoints1, keypoints2, FEATURE_DIST_MAX_RADIUS);

    DEBUG_visualize_matches(frame1, frame2, kp_matches_filtered, keypoints1, keypoints2);

	return 0;
}