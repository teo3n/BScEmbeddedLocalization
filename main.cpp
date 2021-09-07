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

   // Camera cam(4);
   // cam.set_resolution(1280, 720);
   // for (int i = 0; i < 60; i++)
   //    const cv::Mat frame = cam.get_frame_cv();

   // const cv::Mat frame = cam.get_frame_cv();

   // cv::imwrite("../assets/test_img_2.png", frame);
   // return 0;


   const std::shared_ptr<cv::Mat> frame1 = std::make_shared<cv::Mat>(cv::imread("../assets/test_img_1.png"));
   const std::shared_ptr<cv::Mat> frame2 = std::make_shared<cv::Mat>(cv::imread("../assets/test_img_2.png"));

   // camera intrinsics
   double intr_vec[3*3] = {
      779.52117063, 0.0, 602.92592355,
      0.0, 779.18371952, 372.3418880,
      0.0, 0.0, 1.0};
   const cv::Mat intr = utilities::restrucure_mat<double>(intr_vec, 3, 3);

   // distortion coefficients
   // 

   std::shared_ptr<Frame> f1 = frame_from_rgb(frame1, intr);
   std::shared_ptr<Frame> f2 = frame_from_rgb(frame2, intr);

   LGraph lgraph;
   lgraph.localize_frame(f1);
   lgraph.localize_frame(f2);

   return 0;
}
