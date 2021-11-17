/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	The bachelor's thesis project, 
 * 	3D localization and 3D surface reconstruction
 * 	software for microprocessors
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <exception>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <thread>
#include <vector>

#include "src/constants.h"
#include "src/utilities.h"
#include "src/camera_module.h"
#include "src/timer.h"
#include "src/features.h"
#include "src/debug_functions.h"
#include "src/localization_graph.h"
#include "src/point_receiver.h"


using namespace k3d;

/**
 *  TODO: 
 *            implement frame localization
 */


int main()
{
      std::cout << "begin\n";

#ifdef IS_SERVER

      std::cout << "await connection from remote\n";

      networking::StreamHandle stream_handle = networking::await_stream_connection(STREAM_PORT);


      while (true)
      {
            std::vector<Eigen::Vector3d> points, colors;
            Eigen::Vector3d position;
            Eigen::Matrix3d rotation;

            networking::receive_points_data(points, colors, position, rotation, stream_handle);
      }

#else

      // Camera cam(0);
      // cam.set_resolution(1280, 720);
      // for (int i = 0; i < 60; i++)
      //       const cv::Mat frame = cam.get_frame_cv();

      // const cv::Mat frame = cam.get_frame_cv();

      // cv::imwrite("../assets/test_img_3.png", frame);
      // return 0;


      // const std::shared_ptr<cv::Mat> frame1 = std::make_shared<cv::Mat>(cv::imread("../assets/test_img_1.png"));
      // const std::shared_ptr<cv::Mat> frame2 = std::make_shared<cv::Mat>(cv::imread("../assets/test_img_2.png"));
      // const std::shared_ptr<cv::Mat> frame3 = std::make_shared<cv::Mat>(cv::imread("../assets/test_img_3.png"));

      // camera intrinsics
      // double intr_vec[3*3] = {
      //       775.34573964, 0.0, 618.4213988,
      //       0.0, 774.38086624, 357.46060439,
      //       0.0, 0.0, 1.0};
      double intr_vec[3*3] = {
            912.91984103, 0.0, 637.74400638,
            0.0, 912.39513184, 358.96757428,
            0.0, 0.0, 1.0 };
      const cv::Mat intr = utilities::restrucure_mat<double>(intr_vec, 3, 3);

      // distortion coefficients
      // double dist_vec[5] = { 0.09075014, -0.23913447, 0.00068211, -0.00097758,  0.11341968 };
      double dist_vec[5] = { 1.66386586e-01, -5.26413220e-01, -1.01376611e-03, 1.59777094e-04, 4.65208008e-01 };
      const cv::Mat dist = utilities::restrucure_mat<double>(dist_vec, 1, 5);

      const std::shared_ptr<cv::Mat> frame1 = std::make_shared<cv::Mat>(cv::imread("../assets/rock/rgb_8.png"));
      std::shared_ptr<Frame> f1 = frame_from_rgb(frame1, intr, dist);

      LGraph lgraph;
      lgraph.localize_frame(f1);

      std::vector<uint32_t> times;

      int fail_count = 0;

      for (int ii = 35; ii < 89; ii++)
      {
            Timer t;

            const std::shared_ptr<cv::Mat> frame = std::make_shared<cv::Mat>(cv::imread("../assets/rock/rgb_" + std::to_string(ii) + ".png"));
            t.stop("read image");
            std::shared_ptr<Frame> ff = frame_from_rgb(frame, intr, dist);
            t.stop("detect orb features: " + std::to_string(ff->keypoints.size()));

            if (!lgraph.localize_frame(ff))
                  fail_count++;

            if (fail_count > 5)
                  break;

            times.push_back(t.lap_ms());

            t.stop("localize frame " + std::to_string(ii));

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      for (auto t : times)
            std::cout << t << ", ";
      std::cout << "\n";

#ifdef USE_OPEN3D
      lgraph.visualize_camera_tracks(true, false);
#else
      lgraph.print_camera_tracks();
#endif

#endif

      return 0;
}
