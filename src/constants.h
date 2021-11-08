#pragma once

#include <cstdint>
#include <Eigen/Eigen>

namespace k3d
{

#define LOG

typedef Eigen::Matrix<double, 3, 4> Mat34;

// specify the maximum number of features to be detected
static const uint32_t ORB_FEATURE_COUNT = 2000;

// specify the maximum distance the matches can have to be 
// considered valid
static const float MATCH_MAX_DISTANCE = 120.f;

// specify by how many pixels can a pixel's position differ
static const float FEATURE_DIST_MAX_RADIUS = 120.f;

static const float KNN_DISTANCE_RATIO = 0.7f;

// index parameters for the flann based matcher		// default
static const uint32_t FLANN_TABLE_NUMBER = 6; 		// 12
static const uint32_t FLANN_KEY_SIZE = 12;			// 20
static const uint32_t FLANN_MULTI_PROBE_LEVEL = 1;	// 2

// the minimum number of feature matches for good quality RANSAC
static const uint32_t MIN_MATCH_FEATURE_COUNT = 30;

// the number of features, below which new landmarks are triangulated
static const uint32_t MIN_MATCH_TRIANGULATE_NEW_COUNT = 75;

// the minimum amount of movement, which will be considered
// good enough to be triangulated
static const double TRIANGULATE_DIST_DIFF_MAGNITUDE = 1.0;

// when the number of feature-landmark matches, 
// find new landmarks
static const uint32_t MIN_FEATURE_LANDMARK_COUNT_NEW = 200;

// by how much can the new frame differ in position 
// compared to the previous frame
static const double FRAME_POSITION_DISTANCE_DEVIATION = 1.0;

static const double TRIANGULATE_DISTANCE_OUTLIER = 20.0;

// the minimum angle (in degrees) between frames for triangulation
static const double MIN_TRIANGULATION_ANGLE = 15.0;

// threshold, after which homography filtering is considered invalid
static const double HOMOGRAPHY_FILTER_MAX_DIST = 3.0;

// ransac threshold used in calculating homography
static const double HOMOGRAPHY_RANSAC_THRESHOLD = 2.5;

}