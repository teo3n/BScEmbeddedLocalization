#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

namespace k3d
{

inline void DEBUG_visualize_matches(const cv::Mat& frame_0, const cv::Mat& frame_1,
	const std::vector<std::pair<uint32_t, uint32_t>> feature_matches,
	const std::vector<cv::KeyPoint>& fkp1, const std::vector<cv::KeyPoint>& fkp2)
{
    std::vector<cv::DMatch> matches;
    matches.reserve(feature_matches.size());

    for (int ii = 0; ii < feature_matches.size(); ii++)
    	matches.push_back(cv::DMatch(feature_matches[ii].first, feature_matches[ii].second, 0));

    cv::Mat matchimg;
    cv::drawMatches(frame_0, fkp1, frame_1, fkp2, matches, matchimg);
    cv::imshow("DEBUG_visualize_matches", matchimg);
    cv::waitKey(0);
}


};