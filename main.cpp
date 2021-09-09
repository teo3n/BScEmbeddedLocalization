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
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "src/constants.h"
#include "src/utilities.h"
#include "src/camera_module.h"
#include "src/features.h"
#include "src/debug_functions.h"
#include "src/localization_graph.h"


using namespace k3d;

/**
 *  TODO: 
 *      implement frame localization
 */


int main()
{
   std::cout << "begin\n";

   // Camera cam(0);
   // cam.set_resolution(1280, 720);
   // for (int i = 0; i < 60; i++)
   //    const cv::Mat frame = cam.get_frame_cv();

   // const cv::Mat frame = cam.get_frame_cv();

   // cv::imwrite("../assets/test_img_3.png", frame);
   // return 0;


   const std::shared_ptr<cv::Mat> frame1 = std::make_shared<cv::Mat>(cv::imread("../assets/test_img_1.png"));
   const std::shared_ptr<cv::Mat> frame2 = std::make_shared<cv::Mat>(cv::imread("../assets/test_img_2.png"));
   const std::shared_ptr<cv::Mat> frame3 = std::make_shared<cv::Mat>(cv::imread("../assets/test_img_3.png"));

   // camera intrinsics
   double intr_vec[3*3] = {
      775.34573964, 0.0, 618.4213988,
      0.0, 774.38086624, 357.46060439,
      0.0, 0.0, 1.0};
   const cv::Mat intr = utilities::restrucure_mat<double>(intr_vec, 3, 3);

   // distortion coefficients
   double dist_vec[5] = { 0.09075014, -0.23913447, 0.00068211, -0.00097758,  0.11341968 };
   const cv::Mat dist = utilities::restrucure_mat<double>(dist_vec, 1, 5);

   std::shared_ptr<Frame> f1 = frame_from_rgb(frame1, intr, dist);
   std::shared_ptr<Frame> f2 = frame_from_rgb(frame2, intr, dist);
   std::shared_ptr<Frame> f3 = frame_from_rgb(frame3, intr, dist);

   LGraph lgraph;
   lgraph.localize_frame(f1);
   lgraph.localize_frame(f2);
   lgraph.localize_frame(f3);

   lgraph.visualize_camera_tracks();

   return 0;
}
