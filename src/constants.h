/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	7.9.2021
 * 
 * 	A collection of constants used program-wide
 */

 #pragma once

#include <cstdint>
#include <Eigen/Eigen>

namespace k3d
{

#define LOG

#define USE_FLANN
#define USE_FLANN_ESSENTIAL
// #define IS_SERVER
// #define TRANSMIT_POSES
#define USE_OPEN3D

static const std::string STREAM_IP = "127.0.0.1";
static const uint32_t STREAM_PORT = 1234;

typedef Eigen::Matrix<double, 3, 4> Mat34;

// specify the maximum number of features to be detected
static const uint32_t ORB_FEATURE_COUNT = 10000;

static const uint32_t ORB_FEATURE_COUNT_DENSE = 25000;

// specify the maximum distance the matches can have to be 
// considered valid
static const float MATCH_MAX_DISTANCE = 120.f;

// specify by how many pixels can a pixel's position differ
static const float FEATURE_DIST_MAX_RADIUS = 80.f;

// the nearest-neighbour distance ratio, used in feature match distance check
static const float KNN_DISTANCE_RATIO = 0.75f;

// used when only a large quantity of matches is required,
// quality may be dubious
static const float KNN_DISTANCE_RATIO_LIBERAL = 0.85f;

// index parameters for the flann based matcher		// default
static const uint32_t FLANN_TABLE_NUMBER = 12; 		// 12
static const uint32_t FLANN_KEY_SIZE = 20;			// 20
static const uint32_t FLANN_MULTI_PROBE_LEVEL = 1;	// 2

// the minimum number of feature matches for good quality RANSAC
static const uint32_t MIN_MATCH_FEATURE_COUNT = 30;

// the number of feature matches, below which new landmarks are triangulated
static const uint32_t MIN_MATCH_TRIANGULATE_NEW_COUNT = 150;

// the minimum amount of movement, which will be considered
// good enough to be triangulated
static const double TRIANGULATE_DIST_DIFF_MAGNITUDE = 1.0;

// by how much can the new frame differ in position 
// compared to the previous frame
static const double FRAME_POSITION_DISTANCE_DEVIATION = 1.0;

static const double TRIANGULATE_DISTANCE_OUTLIER = 20.0;

// the minimum angle (in degrees) between frames for triangulation
static const double MIN_TRIANGULATION_ANGLE = 15.0;

// the minimum angle (in degrees) between frames for triangulation,
// used when projecting more points
static const double MIN_TRIANGULATION_ANGLE_LIBERAL = 7.5;

// threshold, after which homography filtering is considered invalid
static const double HOMOGRAPHY_FILTER_MAX_DIST = 3.0;

// ransac threshold used in calculating homography
static const double HOMOGRAPHY_RANSAC_THRESHOLD = 2.5;

static const uint32_t HOMOGRAPHY_MIN_FEATURE_COUNT = 9;

// the k-d search tree depth used in reconstructing the 3D mesh
static const uint32_t MESH_POISSON_DEPTH = 7;

// Laplacian values used to smooth the generated mesh
static const uint32_t LAPLACIAN_ITERATIONS = 3;
static const float LAPLACIAN_LAMBDA = 0.75f;

// how often should new dense points be projected
static const uint32_t DENSE_POINTS_EVERY_NTH_FRAME = 30;


}