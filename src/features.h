/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	Functionality used to detect and match features
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <stdexcept>
#include <sys/types.h>
#include <tuple>
#include <iostream>
#include <utility>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
// #include <opencv2/core/eigen.hpp>

#include "constants.h"
#include "timer.h"


namespace k3d::features
{

// no reason to use unnecessary classes, just define as static
static cv::Ptr<cv::ORB> orb_detector;

// use a brute force matcher with cross_check enabled for best results
static cv::BFMatcher bf_matcher (cv::NORM_HAMMING, true);

static cv::FlannBasedMatcher flann_matcher 
	(cv::makePtr<cv::flann::LshIndexParams>(FLANN_TABLE_NUMBER, FLANN_KEY_SIZE, FLANN_MULTI_PROBE_LEVEL));


/**
 * 	@brief detects 2d image features using the ORB 
 * 		features detection algorithm
 * 	@return [<detected keypoints, computed descriptors>]
 */
inline std::tuple<std::vector<cv::KeyPoint>, cv::Mat> detect_features_orb(const std::shared_ptr<cv::Mat> frame)
{
	std::vector<cv::KeyPoint> kp_features;

	if (!orb_detector)
	{
		orb_detector = cv::ORB::create(ORB_FEATURE_COUNT);
		std::cout << "initialized ORB detector\n";
	}

	cv::Mat descriptors;
	orb_detector->detectAndCompute(*frame, cv::noArray(), kp_features, descriptors);	

	return std::make_tuple(kp_features, descriptors);
}

/**
 * 	@brief matches 2 descriptors against earch other using 
 * 		a brute force matcher using crosscheck.
 * 	@return a vector of inlier features, per descriptor index
 */
inline std::vector<std::pair<uint32_t, uint32_t>> match_features_bf_crosscheck(const cv::Mat& desc1, const cv::Mat desc2)
{
	std::vector<cv::DMatch> nn_matches;
	bf_matcher.match(desc1, desc2, nn_matches);

	std::vector<std::pair<uint32_t, uint32_t>> inlier_features;
	inlier_features.reserve(nn_matches.size());

	for (int ii = 0; ii < nn_matches.size(); ii++)
	{
		// Filter out further outliers by using a distance check.
		// This could be omitted in favor of more performance.
		const float dist = nn_matches[ii].distance;
		if (dist > MATCH_MAX_DISTANCE)
			continue;

		inlier_features.push_back(std::make_pair(nn_matches[ii].queryIdx, nn_matches[ii].trainIdx));
	}
	nn_matches.shrink_to_fit();

	return inlier_features;
}

/**
 * 	@brief matches 2 descriptors using a flann based matcher.
 * 		Faster (in theory), and especially with larger feature counts.
 */
inline std::vector<std::pair<uint32_t, uint32_t>> 
	match_features_flann(const cv::Mat& desc1, const cv::Mat& desc2, const float distance_ratio = KNN_DISTANCE_RATIO)
{
	Timer t;
	std::vector<std::pair<uint32_t, uint32_t>> feature_matches;
	std::vector<std::vector<cv::DMatch>> knn_matches;
	feature_matches.reserve(knn_matches.size());

	flann_matcher.knnMatch(desc1, desc2, knn_matches, 2);

	for (uint32_t ii = 0; ii < knn_matches.size(); ii++)
	{
		// still no idea why it even could be 0, but it happens _sometimes_
		// when using a subset of indices
		if (knn_matches[ii].empty())
			continue;

		// do a distance check
		if (knn_matches[ii][0].distance < distance_ratio * knn_matches[ii][1].distance)
			feature_matches.push_back(std::make_pair(knn_matches[ii][0].queryIdx, knn_matches[ii][0].trainIdx));
	}

	feature_matches.shrink_to_fit();
	return feature_matches;
}

inline std::vector<std::pair<uint32_t, uint32_t>> take_only_indexed_matches(
	const std::vector<std::pair<uint32_t, uint32_t>>& matches, const std::vector<uint32_t>& feature_ids)
{
	std::vector<std::pair<uint32_t, uint32_t>> filtered;
	filtered.reserve(feature_ids.size());

	for (const auto& m : matches)
	{
		if (std::find_if(feature_ids.begin(), feature_ids.end(), [&m] (uint32_t elem) {
			return m.second == elem;
		}) != feature_ids.end())
		{
			filtered.push_back(m);
		}
	}

	filtered.shrink_to_fit();
	return filtered;
}

/**
 *	@brief filters the feature matches by radial distance. 
 * 		Only usable with brute-force matcher.
 * 	@return the filtered matches
 */ 
inline std::vector<std::pair<uint32_t, uint32_t>> 
	radius_distance_filter_matches(const std::vector<std::pair<uint32_t, uint32_t>>& matches,
	const std::vector<cv::KeyPoint>& fkp1, const std::vector<cv::KeyPoint>& fkp2,
	const float max_radius)
{
	std::vector<std::pair<uint32_t, uint32_t>> matches_filtered;

	matches_filtered.reserve(matches.size());

	for (int ii = 0; ii < matches.size(); ii++)
	{
		const cv::Point2f kp1 = fkp1[matches[ii].first].pt;
		const cv::Point2f kp2 = fkp2[matches[ii].second].pt;

		const float x_diff = abs(kp1.x - kp2.x);
		const float y_diff = abs(kp1.y - kp2.y);

		// accept the match only if the hypotenuse is shorter than max_radius
		if (sqrt(x_diff * x_diff + y_diff * y_diff) < max_radius)
			matches_filtered.push_back(matches[ii]);
	}

	matches_filtered.shrink_to_fit();
	return matches_filtered;
}

/**
 * 	@brief extracts an individual descriptor from a descriptor matrix
 */
inline cv::Mat get_individual_descriptor(const cv::Mat& desc, const uint32_t id)
{
	const cv::Mat idesc = desc.row(id);
    return idesc;
}

/**
 * 	@brief composes a descriptor from a subset of feature ids
 */
inline cv::Mat descriptor_from_feature_ids(const cv::Mat& desc, const std::vector<uint32_t>& feature_ids)
{
	cv::Mat idesc (cv::Size(desc.cols, feature_ids.size()), desc.type());

	for (int ii = 0; ii < feature_ids.size(); ii++)
	{
		// idesc.row(ii) = get_individual_descriptor(desc, feature_ids[ii]);

		// const uint8_t* mat_i = desc.ptr<uint8_t>(feature_ids[ii], 0);
		// uint8_t* idesc_ptr = idesc.ptr<uint8_t>(ii, 0);

		// for (int jj = 0; jj < desc.cols; jj++)
		// 	idesc_ptr[jj] = mat_i [jj];

		desc.row(feature_ids[ii]).copyTo(idesc.row(ii));
		// std::cout << "1: " << idesc.row(ii) << "\n2: " << desc.row(feature_ids[ii]) << "\n";
	}

	return idesc;
}

/**
 * 	@brief Fixes the indexing of <matches.first> from subset indices to 
 * 			whole vector indices
 */
inline void feature_subset_indices_to_whole(const std::vector<uint32_t>& subset_ids, std::vector<std::pair<uint32_t, uint32_t>>& matches,
	const bool is_first = true)
{
	for (int ii = 0; ii < matches.size(); ii++)
	{
		if (is_first)
			matches[ii].first = subset_ids[ii];
		else
			matches[ii].second = subset_ids[ii];
	}
}


/**
 * 	@brief normalizes the frame, i.e. transfers it from image coordinates to uv [0, 1] coordinates
 */
inline Eigen::Vector2d image2world(const Eigen::Vector2d& image_point, const cv::Mat& intr)
{
	const double x = image_point.x();
	const double y = image_point.y();

	const double f = intr.at<double>(0, 0);
	const double c1 = intr.at<double>(0, 2);
	const double c2 = intr.at<double>(1, 2);

	const double world_x = (x - c1) / f;
	const double world_y = (y - c2) / f;

	return Eigen::Vector2d(world_x, world_y);
}

/**
 * 	@brief triangulates a 3d point from 2d correspondences using 
 * 		multiview principles. Reduces the reprojection error and thus is 
 * 		more accurate, but is also considerably slower.
 */
inline cv::Point3f triangulate_multiview(const std::vector<cv::Point2f>& feature_points, 
		const std::vector<Mat34>& proj_matrices, const cv::Mat& intr = cv::Mat())
{
	if (proj_matrices.size() != feature_points.size())
		throw std::runtime_error("proj_matrices size did not match feature_points size!");

	Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

	for (size_t i = 0; i < feature_points.size(); i++)
	{
		const Eigen::Vector2d fpoint (feature_points[i].x, feature_points[i].y);
		// const Eigen::Vector2d npoint = image2world(fpoint, intr);

        const Eigen::Vector3d point = fpoint.homogeneous().normalized();
        const Mat34 term = proj_matrices[i] - point * point.transpose() * proj_matrices[i];
        A += term.transpose() * term;
  	}

  	Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

	const Eigen::Vector3d p3d = eigen_solver.eigenvectors().col(0).hnormalized();
	return cv::Point3f(p3d.x(), p3d.y(), p3d.z());
}

/**
 * 	@brief solves the multiview triangulation problem by minimizing the L2 error, 
 * 		the Eigen implementation. Taken from
 * 		https://github.com/colmap/colmap/blob/9f3a75ae9c72188244f2403eb085e51ecf4397a8/src/base/triangulation.cc
 */
inline cv::Point3f triangulate_multiview_eigen(const std::vector<cv::Point2f>& feature_points, 
	const std::vector<Mat34>& proj_matrices, const cv::Mat& intr = cv::Mat())
{
	std::vector<Eigen::Vector2d> feature_points_eigen;
	feature_points_eigen.reserve(feature_points.size());
	for (const auto& f : feature_points)
	{
		feature_points_eigen.emplace_back(Eigen::Vector2d(f.x, f.y));
		// feature_points_eigen.back() = image2world(feature_points_eigen.back(), intr);
	}

	Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

	for (size_t i = 0; i < feature_points_eigen.size(); i++)
	{
		const Eigen::Vector3d point = feature_points_eigen[i].homogeneous().normalized();
		const Mat34 term =
		    proj_matrices[i] - point * point.transpose() * proj_matrices[i];
		A += term.transpose() * term;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

	const Eigen::Vector3d p3d = eigen_solver.eigenvectors().col(0).hnormalized();
	return cv::Point3f(p3d.x(), p3d.y(), p3d.z());
}


};