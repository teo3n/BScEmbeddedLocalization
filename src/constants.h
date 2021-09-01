#pragma once

#include <cstdint>


// specify the maximum number of features to be detected
static const uint32_t ORB_FEATURE_COUNT = 10000;

// specify the maximum distance the matches can have to be 
// considered valid
static const float MATCH_MAX_DISTANCE = 200.f;

// specify by how many pixels can a pixel's position differ
static const float FEATURE_DIST_MAX_RADIUS = 100.f;

