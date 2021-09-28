#include "localization_graph.h"
#include "features.h"
#include <Open3D/Geometry/Geometry.h>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/DrawGeometry.h>
#include <algorithm>
#include <cstdint>
#include <fmt/format.h>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <stdexcept>
#include <utility>

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
        localize_frame_essential(frames[0], frame);

    // localize using PnP
    else
        localize_frame_pnp(frames[frames.size() - 2], frame);
}


void LGraph::localize_frame_essential(const std::shared_ptr<Frame> ref_frame, std::shared_ptr<Frame> frame)
{
    std::vector<std::pair<uint32_t, uint32_t>> matches = features::match_features_bf_crosscheck(
        ref_frame->descriptors, frame->descriptors);
    matches = features::radius_distance_filter_matches(matches, ref_frame->keypoints, frame->keypoints, FEATURE_DIST_MAX_RADIUS);


    if (matches.size() < MIN_MATCH_FEATURE_COUNT)
        throw std::runtime_error("not enough feature matches: " + std::to_string(matches.size()));

    // DEBUG_visualize_matches(*ref_frame->rgb, *frame->rgb, matches, ref_frame->keypoints, frame->keypoints);

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

    const auto T_P = utilities::TP_from_Rt(dR.transpose(), -dR.transpose() * dt, frame->params.intr);
    frame->transformation = T_P.first;
    frame->projection = T_P.second;
    cv::eigen2cv(T_P.second, frame->projection_cv);

    #ifdef LOG
    const  Eigen::AngleAxisd ax(dR);
    std::cout << "essential angle: " << ax.angle() * (180. / 3.1415) << "\n";
    #endif

    create_landmarks_from_matches(ref_frame, frame, matches);
}

void LGraph::create_landmarks_from_matches(const std::shared_ptr<Frame> ref_frame, 
        const std::shared_ptr<Frame> frame, const std::vector<std::pair<uint32_t, uint32_t>>& matches)
{
    for (auto& m : matches)
    {
        Landmark lm;

        lm.first_frame = ref_frame;

        lm.descriptors.push_back(features::get_individual_descriptor(ref_frame->descriptors, m.first));
        lm.descriptors.push_back(features::get_individual_descriptor(frame->descriptors, m.second));

        const std::vector<cv::Point2f> x1 = { ref_frame->keypoints[m.first].pt };
        const std::vector<cv::Point2f> x2 = { frame->keypoints[m.second].pt };

        lm.first_feature_point = x1[0];

        cv::Mat p4d;
        cv::triangulatePoints(ref_frame->projection_cv, frame->projection_cv, x1, x2, p4d);
        lm.location = cv::Point3f(
                    p4d.at<float>(0, 0) / p4d.at<float>(3, 0),
                    p4d.at<float>(1, 0) / p4d.at<float>(3, 0),
                    p4d.at<float>(2, 0) / p4d.at<float>(3, 0));

        // if the triangulated point is behind the camera or too far away -> ignore 
        if (!utilities::point_infront_of_camera(frame->projection, Eigen::Vector3d(lm.location.x, lm.location.y, lm.location.z)) ||
            Eigen::Vector3d(lm.location.x, lm.location.y, lm.location.z).norm() > TRIANGULATE_DISTANCE_OUTLIER)
            continue;

        landmarks.push_back(lm);

        // <feature index, landmark index>
        frame->feature_landmark_lookup.push_back(std::make_pair(m.second, landmarks.size() - 1));
    }
}

void LGraph::update_landmarks(const std::shared_ptr<Frame> frame, 
        std::vector<uint32_t>& feature_ids, std::vector<uint32_t>& landmark_ids)
{
    for (int ii = 0; ii < feature_ids.size(); ii++)
    {
        Landmark& lm = landmarks[landmark_ids[ii]];

        // const std::vector<cv::Point2f> x1 = { lm.first_feature_point };
        // const std::vector<cv::Point2f> x2 = { frame->keypoints[feature_ids[ii]].pt };

        // cv::Mat p4d;
        // cv::triangulatePoints(lm.first_frame->projection_cv, frame->projection_cv, x1, x2, p4d);
        // const cv::Point3f p3d = cv::Point3f(
        //             p4d.at<float>(0, 0) / p4d.at<float>(3, 0),
        //             p4d.at<float>(1, 0) / p4d.at<float>(3, 0),
        //             p4d.at<float>(2, 0) / p4d.at<float>(3, 0));

        // // if the triangulated point is behind the camera -> ignore 
        // if (utilities::point_infront_of_camera(frame->projection, Eigen::Vector3d(p3d.x, p3d.y, p3d.z)) &&
        //     Eigen::Vector3d(p3d.x, p3d.y, p3d.z).norm() < TRIANGULATE_DISTANCE_OUTLIER)
        // {
        //     lm.location += p3d;
        //     lm.location /= 2.0;
        // }

        frame->feature_landmark_lookup.push_back(std::make_pair(feature_ids[ii], landmark_ids[ii]));
        lm.descriptors.push_back(features::get_individual_descriptor(frame->descriptors, feature_ids[ii]));
    }
}

void LGraph::localize_frame_pnp(const std::shared_ptr<Frame> prev_frame, std::shared_ptr<Frame> frame)
{
    /**
     *  - match against previous frame and find landmarks using lookup
     *  - PnP
     *  - new landmarks?
     */

    std::vector<std::pair<uint32_t, uint32_t>> matches = 
        features::match_features_bf_crosscheck(prev_frame->descriptors, frame->descriptors);
    matches = features::radius_distance_filter_matches(matches, prev_frame->keypoints, frame->keypoints, FEATURE_DIST_MAX_RADIUS);

    // DEBUG_visualize_matches(*prev_frame->rgb, *frame->rgb, matches, prev_frame->keypoints, frame->keypoints);

    // find landmark points for PnP
    std::vector<cv::Point3f> lm_points;
    std::vector<uint32_t> lm_ids;
    std::vector<cv::Point2f> feature_points;
    std::vector<uint32_t> feature_ids;

    // collect landmark point ids and current frame feature point ids
    // find prev-current matched form prev-landmarks
    for (const auto& flm : prev_frame->feature_landmark_lookup)
    {
        for (const auto& match : matches)
        {
            // feature match was found in landmark lookup
            if (match.first == flm.first)
            {
                feature_points.push_back(frame->keypoints[match.second].pt);
                feature_ids.push_back(match.second);

                lm_points.push_back(landmarks[flm.second].location);
                lm_ids.push_back(flm.second);
            }
        }
    }

    cv::Mat rcv, tcv, rcv_mat;
    cv::eigen2cv(prev_frame->position, tcv);
    cv::eigen2cv(prev_frame->rotation, rcv);

    std::vector<int> inliers;
    cv::solvePnPRansac(lm_points, feature_points, frame->params.intr, frame->params.distortion, rcv, tcv, true, 1000, 4.0, 0.987, inliers);
    cv::Rodrigues(rcv, rcv_mat);

    Eigen::Matrix3d dR;
    Eigen::Vector3d dt;
    cv::cv2eigen(rcv_mat, dR);
    cv::cv2eigen(tcv, dt);

    frame->rotation = dR;
    frame->position = dt;

    const auto T_P = utilities::TP_from_Rt(dR.transpose(), -dR.transpose() * dt, frame->params.intr);
    frame->transformation = T_P.first;
    frame->projection = T_P.second;
    cv::eigen2cv(T_P.second, frame->projection_cv);

    // reject the frame if it has too high location magnitude
    if (frame->position.norm() - prev_frame->position.norm() > FRAME_POSITION_DISTANCE_DEVIATION)
    {
        std::cout << "frame rejected for having position of " << frame->position.transpose() << 
            " compared to " << prev_frame->position.transpose() << "\n";

        frames.pop_back();
        return;
    }

    /**
     * TODO:
     *      fix camera positions and pointcloud positions being wack
     */

    // update matched landmarks using lookup
    update_landmarks(frame, feature_ids, lm_ids);

    if (feature_points.size() <= MIN_FEATURE_LANDMARK_COUNT_NEW)
    {
        const auto rframe = find_triangulatable_movement_frame(frame);
        if (rframe)
            new_landmarks_from_matched(rframe, frame);
    }
}


std::shared_ptr<Frame> LGraph::find_triangulatable_movement_frame(const std::shared_ptr<Frame> frame)
{
    const Eigen::Vector3d ref_pos = frame->position;

    for (int ii = frames.size() - 1; ii >= 0; ii--)
    {
        const std::shared_ptr<Frame> ref_frame = frames[ii];

        if ((ref_frame->position - ref_pos).norm() > TRIANGULATE_DIST_DIFF_MAGNITUDE)
            return ref_frame;
    }

    return nullptr;
}

void LGraph::new_landmarks_from_matched(const std::shared_ptr<Frame> ref_frame,
        const std::shared_ptr<Frame> frame)
{
    std::vector<std::pair<uint32_t, uint32_t>> matches = 
        features::match_features_bf_crosscheck(ref_frame->descriptors, frame->descriptors);

    matches = features::radius_distance_filter_matches(matches, ref_frame->keypoints, frame->keypoints, FEATURE_DIST_MAX_RADIUS);

    std::vector<std::pair<uint32_t, uint32_t>> new_matches;

    // filter matches for already existing landmarks
    for (const auto& m : matches)
    {
        // find if the feature id already exists in teh landmark lookup
        const auto it = std::find_if(ref_frame->feature_landmark_lookup.begin(), ref_frame->feature_landmark_lookup.end(),
            [&m](const std::pair<uint32_t, uint32_t>& feature_landmark) {
                return feature_landmark.first == m.first;
            });

        // the feature was not found --> create new landmarks
        if (it == ref_frame->feature_landmark_lookup.end())
            new_matches.push_back(m);
    }

    // std::cout << ref_frame->position.transpose() << ", " << frame->position.transpose() << "\n";

    create_landmarks_from_matches(ref_frame, frame, new_matches);
}

void LGraph::visualize_camera_tracks(const bool visualize_landmarks) const
{
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> debug_cameras;

    for (int ii = 0; ii < frames.size(); ii++)
    {
        std::cout << "frame " << ii << ": " << frames[ii]->position.transpose() << "\n";

        std::shared_ptr<open3d::geometry::TriangleMesh> camera_mesh = std::make_shared<open3d::geometry::TriangleMesh>(open3d::geometry::TriangleMesh());
        open3d::io::ReadTriangleMeshFromOBJ("../assets/debug_camera_mesh.obj", *camera_mesh, false);

        camera_mesh->Transform(frames[ii]->transformation);
        debug_cameras.push_back(camera_mesh);
    }

    if (visualize_landmarks)
    {
        std::vector<Eigen::Vector3d> landmark_points;

        for (const auto& lm : landmarks)
            landmark_points.push_back(Eigen::Vector3d(lm.location.x, lm.location.y, lm.location.z));

        auto lms_cloud = std::make_shared<open3d::geometry::PointCloud>(open3d::geometry::PointCloud(landmark_points));
        debug_cameras.push_back(lms_cloud);
    }

    open3d::visualization::DrawGeometries(debug_cameras);
}