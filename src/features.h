/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	Functionality used to detect and match features
 */

#pragma once

#include <cstdint>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <tuple>
#include <iostream>
#include <utility>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "constants.h"


namespace k3d
{

// no reason to use unnecessary classes, just define as static
static cv::Ptr<cv::ORB> orb_detector;

// use a brute force matcher with cross_check enabled for best results
static cv::BFMatcher bf_matcher (cv::NORM_HAMMING, true);


/**
 * 	@brief detects 2d image features using the ORB 
 * 		features detection algorithm
 * 	@return [<detected keypoints, computed descriptors>]
 */
inline std::tuple<std::vector<cv::KeyPoint>, cv::Mat> detect_features_orb(const cv::Mat& frame)
{
	std::vector<cv::KeyPoint> kp_features;

	if (!orb_detector)
	{
		orb_detector = cv::ORB::create(ORB_FEATURE_COUNT);
		std::cout << "initialized ORB detector\n";
	}

	cv::Mat descriptors;
	orb_detector->detectAndCompute(frame, cv::noArray(), kp_features, descriptors);	

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
 *	@brief filters the feature matches by radial distance
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

};