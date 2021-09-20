/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	A 
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <vector>
#include <memory>

#include <Eigen/Eigen>
#include <opencv2/core.hpp>

#include "features.h"
#include "utilities.h"
#include "constants.h"

namespace k3d
{

/**
 * 	@brief Holds camera parameters
 */
struct CameraParams
{
	cv::Mat intr;
	cv::Mat distortion;
};


/**
 * 	@brief Represents a frame in 3D space
 */
struct Frame
{
	std::shared_ptr<cv::Mat> rgb;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	// <feature_id, landmark_id>
	std::vector<std::pair<uint32_t, uint32_t>> feature_landmark_lookup;

	Eigen::Vector3d position;
	Eigen::Matrix3d rotation;

	Eigen::Matrix4d transformation;
	Mat34 projection;

	cv::Mat projection_cv;

	CameraParams params;
};

inline std::shared_ptr<Frame> frame_from_rgb(const std::shared_ptr<cv::Mat> rgb, const cv::Mat& intr, const cv::Mat& distortion)
{
	const auto [keypoints, descriptors] = features::detect_features_orb(rgb);

	const auto T_P = utilities::TP_from_Rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), intr);
	cv::Mat p_cv;
	cv::eigen2cv(T_P.second, p_cv);

	return std::make_shared<Frame>(
		Frame {
			rgb,
			keypoints,
			descriptors,
			std::vector<std::pair<uint32_t, uint32_t>>(),
			Eigen::Vector3d::Zero(),
			Eigen::Matrix3d::Identity(),
			T_P.first,
			T_P.second,
			p_cv,
			CameraParams { intr, distortion }
		});
}

};