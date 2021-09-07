#pragma once

#include <cstdint>
#include <Eigen/Eigen>

namespace k3d
{

#define LOG

typedef Eigen::Matrix<double, 3, 4> Mat34;

// specify the maximum number of features to be detected
static const uint32_t ORB_FEATURE_COUNT = 10000;

// specify the maximum distance the matches can have to be 
// considered valid
static const float MATCH_MAX_DISTANCE = 300.f;

// specify by how many pixels can a pixel's position differ
static const float FEATURE_DIST_MAX_RADIUS = 100.f;

static const float KNN_DISTANCE_RATIO = 0.7f;

// index parameters for the flann based matcher		// default
static const uint32_t FLANN_TABLE_NUMBER = 6; 		// 12
static const uint32_t FLANN_KEY_SIZE = 12;			// 20
static const uint32_t FLANN_MULTI_PROBE_LEVEL = 1;	// 2

// the minimum amount of feature matches for good quality RANSAC
static const uint32_t MIN_MATCH_FEATURE_COUNT = 30;


}