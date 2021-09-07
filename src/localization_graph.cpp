#include "localization_graph.h"

using namespace k3d;


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

    const std::vector<std::pair<uint32_t, uint32_t>> matches = features::match_features_flann(
        ref_frame->descriptors, frame->descriptors, KNN_DISTANCE_RATIO);

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

    cv::Mat mask;
    const cv::Mat essential = cv::findEssentialMat(x1, x2, frame->intr, cv::RANSAC, 0.999, 0.11, mask);

    cv::Mat local_R, local_t;
    cv::recoverPose(essential, x1, x2, frame->intr, local_R, local_t, mask);

    Eigen::Matrix3d dR;
    Eigen::Vector3d dt;
    cv::cv2eigen(local_R, dR);
    cv::cv2eigen(local_t, dt);

    frame->position = dt;
    frame->rotation = dR;

    const auto T_P = utilities::TP_from_Rt(dR, dt, frame->intr);
    frame->transformation = T_P.first;
    frame->projection = T_P.second;

    #ifdef LOG
    const  Eigen::AngleAxisd ax(dR);
    std::cout << "essential angle: " << ax.angle() * (180. / 3.1415) << "\n";
    #endif

    return std::make_pair(position, rotation);
}