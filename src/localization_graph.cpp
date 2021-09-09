#include "localization_graph.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

using namespace k3d;

LGraph::LGraph()
{
    // double identity_vec[3*3] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    // identity_mat = utilities::restrucure_mat<double>(identity_vec, 3, 3);
}

void LGraph::localize_frame(std::shared_ptr<Frame> frame)
{
    // const std::vector<std::pair<uint32_t, uint32_t>> kp_matches = 
	   //  // match_features_bf_crosscheck(desc1, desc2);
    //     match_features_flann(desc1, desc2, KNN_DISTANCE_RATIO);

    // DEBUG_visualize_matches(frame1, frame2, kp_matches, keypoints1, keypoints2);

    /**
     *  
     */


	this->frames.push_back(frame);

    // cannot localize if only 1 frame
    if (frames.size() == 1)
        return;

    // localize using essential matrix decomposition
    else if (frames.size() == 2)
    {
        const auto pos_rot = localize_frame_essential(frames[0], frame);
        frame->position = pos_rot.first;
        frame->rotation = pos_rot.second;
    }

    // localize normally
}


std::pair<Eigen::Vector3d, Eigen::Matrix3d> LGraph::localize_frame_essential(
        const std::shared_ptr<Frame> ref_frame, std::shared_ptr<Frame> frame)
{
    Eigen::Vector3d position;
    Eigen::Matrix3d rotation;

    std::vector<std::pair<uint32_t, uint32_t>> matches = features::match_features_flann(
        ref_frame->descriptors, frame->descriptors, KNN_DISTANCE_RATIO);
    matches = features::radius_distance_filter_matches(matches, ref_frame->keypoints, frame->keypoints, FEATURE_DIST_MAX_RADIUS);


    if (matches.size() < MIN_MATCH_FEATURE_COUNT)
        throw std::runtime_error("not enough feature matches: " + std::to_string(matches.size()));

    DEBUG_visualize_matches(*ref_frame->rgb, *frame->rgb, matches, ref_frame->keypoints, frame->keypoints);

    std::vector<cv::Point2f> x1, x2;
    x1.reserve(matches.size());
    x2.reserve(matches.size());

    // populate the 2d feature position vectors
    for (const auto& m : matches)
    {
        x1.push_back(ref_frame->keypoints[m.first].pt);
        x2.push_back(frame->keypoints[m.second].pt);
    }
    cv::undistortPoints(x1, x1, frame->params.intr, frame->params.distortion, cv::noArray(), frame->params.intr);
    cv::undistortPoints(x2, x2, frame->params.intr, frame->params.distortion, cv::noArray(), frame->params.intr);

    cv::Mat mask;
    const cv::Mat essential = cv::findEssentialMat(x1, x2, frame->params.intr, cv::RANSAC, 0.999, 0.11, mask);

    cv::Mat local_R, local_t;
    cv::recoverPose(essential, x1, x2, frame->params.intr, local_R, local_t, mask);

    Eigen::Matrix3d dR;
    Eigen::Vector3d dt;
    cv::cv2eigen(local_R, dR);
    cv::cv2eigen(local_t, dt);

    frame->position = dt;
    frame->rotation = dR;

    const auto T_P = utilities::TP_from_Rt(dR, dt, frame->params.intr);
    frame->transformation = T_P.first;
    frame->projection = T_P.second;

    #ifdef LOG
    const  Eigen::AngleAxisd ax(dR);
    std::cout << "essential angle: " << ax.angle() * (180. / 3.1415) << "\n";
    #endif

    return std::make_pair(position, rotation);
}