#pragma once

#include <cstdint>
#include <Eigen/Eigen>

namespace k3d
{

#define LOG

#define USE_FLANN
#define USE_FLANN_ESSENTIAL


typedef Eigen::Matrix<double, 3, 4> Mat34;

// specify the maximum number of features to be detected
static const uint32_t ORB_FEATURE_COUNT = 10000;

// specify the maximum distance the matches can have to be 
// considered valid
static const float MATCH_MAX_DISTANCE = 120.f;

// specify by how many pixels can a pixel's position differ
static const float FEATURE_DIST_MAX_RADIUS = 80.f;

static const float KNN_DISTANCE_RATIO = 0.7f;

// index parameters for the flann based matcher		// default
static const uint32_t FLANN_TABLE_NUMBER = 12; 		// 12
static const uint32_t FLANN_KEY_SIZE = 20;			// 20
static const uint32_t FLANN_MULTI_PROBE_LEVEL = 2;	// 2

// the minimum number of feature matches for good quality RANSAC
static const uint32_t MIN_MATCH_FEATURE_COUNT = 30;

// the number of feature matches, below which new landmarks are triangulated
static const uint32_t MIN_MATCH_TRIANGULATE_NEW_COUNT = 75;

// the minimum amount of movement, which will be considered
// good enough to be triangulated
static const double TRIANGULATE_DIST_DIFF_MAGNITUDE = 1.0;

// by how much can the new frame differ in position 
// compared to the previous frame
static const double FRAME_POSITION_DISTANCE_DEVIATION = 1.0;

static const double TRIANGULATE_DISTANCE_OUTLIER = 20.0;

// the minimum angle (in degrees) between frames for triangulation
static const double MIN_TRIANGULATION_ANGLE = 0.4;

// threshold, after which homography filtering is considered invalid
static const double HOMOGRAPHY_FILTER_MAX_DIST = 3.0;

// ransac threshold used in calculating homography
static const double HOMOGRAPHY_RANSAC_THRESHOLD = 2.5;

// when backpropagating matches, how far in the feature match tree
// should we go before the number of matches drops below a 
// percentual threshold
static const double STATISTICAL_FEATURE_COUNT = 0.5;

}