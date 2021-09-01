/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	An abstraction for detecting 2d image features
 */

#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "constants.h"


namespace k3d
{

static cv::Ptr<cv::ORB> orb_detector;


/**
 * 	@brief detects 2d image features using the ORB 
 * 		features detection algorithm
 */
std::vector<cv::KeyPoint> detect_features_orb(const cv::Mat& frame);


}