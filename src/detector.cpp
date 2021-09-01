#include "detector.h"


using namespace k3d;

std::vector<cv::KeyPoint> detect_features_orb(const cv::Mat& frame)
{
	std::vector<cv::KeyPoint> features;

	if (!orb_detector)
	{
		orb_detector = cv::ORB::create(ORB_FEATURE_COUNT);
		std::cout << "was null, initialized\n";
	}


	return features;
}
