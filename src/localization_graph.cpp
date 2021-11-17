#include "localization_graph.h"
#include "debug_functions.h"
#include "features.h"
#include <Eigen/src/Core/Matrix.h>

#ifdef USE_OPEN3D
    #include <Open3D/Geometry/Geometry.h>
    #include <Open3D/Geometry/PointCloud.h>
    #include <Open3D/Geometry/TriangleMesh.h>
    #include <Open3D/Visualization/Utility/DrawGeometry.h>
    #include <Open3D/IO/ClassIO/TriangleMeshIO.h>
#endif

#include <algorithm>
#include <cstdint>
// #include <fmt/format.h>
#include <iostream>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>

using namespace k3d;

LGraph::LGraph()
{
    // double identity_vec[3*3] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    // identity_mat = utilities::restrucure_mat<double>(identity_vec, 3, 3);
}

bool LGraph::localize_frame(std::shared_ptr<Frame> frame)
{
	frames.push_back(frame);

    // cannot localize if only 1 frame
    if (frames.size() == 1)
        return true;

    // localize using essential matrix decomposition
    else if (frames.size() == 2)
        return localize_frame_essential(frames[0], frame);

    // localize using PnP
    else
        return localize_frame_pnp(frames[frames.size() - 2], frame);
}


bool LGraph::localize_frame_essential(const std::shared_ptr<Frame> ref_frame, std::shared_ptr<Frame> frame)
{
    std::vector<std::pair<uint32_t, uint32_t>> matches = 
#if defined USE_FLANN_ESSENTIAL
        features::match_features_flann(ref_frame->descriptors, frame->descriptors);
#else
        features::match_features_bf_crosscheck(ref_frame->descriptors, frame->descriptors);
#endif

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

    // visualize_camera_tracks(true);

    return true;
}

void LGraph::create_landmarks_from_matches(const std::shared_ptr<Frame> ref_frame, 
        const std::shared_ptr<Frame> frame, const std::vector<std::pair<uint32_t, uint32_t>>& matches)
{
    for (auto& m : matches)
    {
        Landmark lm;

        lm.descriptors.push_back(features::get_individual_descriptor(ref_frame->descriptors, m.first));
        lm.descriptors.push_back(features::get_individual_descriptor(frame->descriptors, m.second));

        std::vector<cv::Point2f> x1 = { ref_frame->keypoints[m.first].pt };
        std::vector<cv::Point2f> x2 = { frame->keypoints[m.second].pt };

        cv::undistortPoints(x1, x1, frame->params.intr, ref_frame->params.distortion, cv::noArray(), ref_frame->params.intr);
        cv::undistortPoints(x2, x2, frame->params.intr, frame->params.distortion, cv::noArray(), frame->params.intr);

        lm.feature_2d_points.push_back(x1[0]);
        lm.feature_2d_points.push_back(x2[0]);

        lm.triangulate_2d_points.push_back(x1[0]);
        lm.triangulate_2d_points.push_back(x2[0]);

        lm.view_frames.push_back(ref_frame);
        lm.view_frames.push_back(frame);

        lm.triangulate_frames.push_back(ref_frame);
        lm.triangulate_frames.push_back(frame);

// use multiview triangulation. Considerably slower.
#if false
        const cv::Point3d new_p3d = features::triangulate_multiview_eigen(lm.triangulate_2d_points, [&lm] {
            std::vector<Mat34> pmats;
            for (const auto& f : lm.triangulate_frames)
                pmats.push_back(f->projection);
            return pmats;
        }());
// use 2view DLT. Fast.
#else
        cv::Mat p4d;
        cv::triangulatePoints(ref_frame->projection_cv, frame->projection_cv, x1, x2, p4d);
        const cv::Point3f new_p3d = cv::Point3f(
                    p4d.at<float>(0, 0) / p4d.at<float>(3, 0),
                    p4d.at<float>(1, 0) / p4d.at<float>(3, 0),
                    p4d.at<float>(2, 0) / p4d.at<float>(3, 0));
#endif

        // if the triangulated point is behind the camera or too far away -> ignore 
        if (!utilities::point_in_front(frame->projection, Eigen::Vector3d(new_p3d.x, new_p3d.y, new_p3d.z)) ||
            Eigen::Vector3d(new_p3d.x, new_p3d.y, new_p3d.z).norm() > TRIANGULATE_DISTANCE_OUTLIER)
            continue;

        lm.location = new_p3d;
        const cv::Vec3b col = frame->rgb->at<cv::Vec3b>(x2[0].y, x2[0].x);
        lm.color = Eigen::Vector3d((float)(col.val[2]) / 255.0, (float)(col.val[1]) / 255.0, (float)(col.val[0]) / 255.0);

        lm.normal = frame->position - Eigen::Vector3d(new_p3d.x, new_p3d.y, new_p3d.z);

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
        lm.feature_2d_points.push_back(frame->keypoints[feature_ids[ii]].pt);
        lm.view_frames.push_back(frame);

        // for performance's sake skip the incremential 3d improvement
        continue;

        const Eigen::Vector3d test_p3d (lm.location.x, lm.location.y, lm.location.z);

        // find a triangulatable frame
        for (int jj = lm.triangulate_frames.size() - 1; jj >= 0; jj--)
        {
            const double frame_angle = utilities::calculate_triangulation_angle(lm.triangulate_frames[jj]->position, frame->position, test_p3d);

            // std::cout << frame_angle << ", " << RAD2DEG(frame_angle) << "\n";

            if (RAD2DEG(frame_angle) < MIN_TRIANGULATION_ANGLE)
                continue;


            lm.triangulate_2d_points.push_back(frame->keypoints[feature_ids[ii]].pt);

            const cv::Point3d new_p3d = features::triangulate_multiview_eigen(lm.feature_2d_points, [&lm, &frame] {
                std::vector<Mat34> pmats;
                for (const auto& f : lm.view_frames)
                    pmats.push_back(f->projection);
                pmats.push_back(frame->projection);
                return pmats;
            }());

            if (std::isnan(new_p3d.x) || std::isnan(new_p3d.y) || std::isnan(new_p3d.z) ||
                std::isinf(new_p3d.x) || std::isinf(new_p3d.y) || std::isinf(new_p3d.z))
            {
                lm.triangulate_2d_points.pop_back();
                continue;
            }

            // if the triangulated point is behind the camera or too far away -> ignore 
            if (!utilities::point_in_front(frame->projection, Eigen::Vector3d(new_p3d.x, new_p3d.y, new_p3d.z)) ||
                Eigen::Vector3d(new_p3d.x, new_p3d.y, new_p3d.z).norm() > TRIANGULATE_DISTANCE_OUTLIER)
            {
                lm.triangulate_2d_points.pop_back();
                continue;
            }

            lm.location = new_p3d;
            lm.triangulate_frames.push_back(frame);

            std::cout << ii << " updated, " << frame_angle << ", " << RAD2DEG(frame_angle) << "\n";

            break;
        }
    }
}

bool LGraph::localize_frame_pnp(const std::shared_ptr<Frame> prev_frame, std::shared_ptr<Frame> frame)
{
    std::vector<std::pair<uint32_t, uint32_t>> matches = 
#if defined USE_FLANN
        features::match_features_flann(prev_frame->descriptors, frame->descriptors);
#else
        features::match_features_bf_crosscheck(prev_frame->descriptors, frame->descriptors);
#endif

    // matches = features::radius_distance_filter_matches(matches, prev_frame->keypoints, frame->keypoints, FEATURE_DIST_MAX_RADIUS);
    matches = homography_filter_matches(prev_frame, frame, matches);

    // DEBUG_visualize_matches(*prev_frame->rgb, *frame->rgb, matches, prev_frame->keypoints, frame->keypoints);

    // find landmark points for PnP
    std::vector<cv::Point3f> lm_points;
    std::vector<uint32_t> lm_ids;
    std::vector<cv::Point2f> feature_points;
    std::vector<uint32_t> feature_ids;

    for (const auto& lfc : find_landmark_feature_matches(prev_frame, matches))
    {
        feature_points.push_back(frame->keypoints[lfc.first].pt);
        feature_ids.push_back(lfc.first);

        lm_points.push_back(landmarks[lfc.second].location);
        lm_ids.push_back(lfc.second);
    }

    if (feature_points.size() < MIN_MATCH_FEATURE_COUNT)
    {
        std::cout << "could not localize frame, not enough matches: " << feature_points.size() << "\n";
        this->frames.pop_back();
        return false;
    }

    cv::undistortPoints(feature_points, feature_points, frame->params.intr, frame->params.distortion, cv::noArray(), frame->params.intr);

    cv::Mat rcv, tcv, rcv_mat;
    cv::eigen2cv(prev_frame->position, tcv);
    cv::eigen2cv(prev_frame->rotation, rcv);

    std::vector<int> inliers;
    cv::solvePnPRansac(lm_points, feature_points, frame->params.intr, frame->params.distortion, rcv, tcv, true, 1000, 8.0, 0.988, inliers);
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
        std::cout << "frame rejected for having an outlier position of " << frame->position.transpose() << 
            " compared to the previous frame pose of " << prev_frame->position.transpose() << "\n";

        frames.pop_back();
        return false;
    }

    // update matched landmarks using lookup
    update_landmarks(frame, feature_ids, lm_ids);

    if (feature_points.size() < MIN_MATCH_TRIANGULATE_NEW_COUNT)
    {
        // std::cout << "low number of active landmarks reached, creating new ones\n";
        new_landmarks_standalone(frame, lm_points);
    }

    if (frames.size() % DENSE_POINTS_EVERY_NTH_FRAME == 0)
    {
        std::cout << "project more points\n";
        project_more_points(find_triangulatable_point_frame(frame, lm_points, MIN_TRIANGULATION_ANGLE_LIBERAL), frame);
    }

    return true;
}

void LGraph::project_more_points(const std::shared_ptr<Frame> ref_frame, const std::shared_ptr<Frame> frame)
{
    // compute dense features
    const auto [ref_kps, ref_desc] = features::detect_features_orb(ref_frame->rgb, true);
    const auto [frame_kps, frame_desc] = features::detect_features_orb(frame->rgb, true);

    std::vector<std::pair<uint32_t, uint32_t>> matches = 
        features::match_features_flann(ref_desc, frame_desc, KNN_DISTANCE_RATIO_LIBERAL);

    matches = features::radius_distance_filter_matches(matches, ref_kps, frame_kps, FEATURE_DIST_MAX_RADIUS);
    // matches = homography_filter_matches(ref_frame, frame, matches);

    // DEBUG_visualize_matches(*ref_frame->rgb, *frame->rgb, matches, ref_frame->keypoints, frame->keypoints);

    std::vector<cv::Point2f> x1, x2;

    for (const auto& m : matches)
    {
        x1.push_back(ref_kps[m.first].pt);
        x2.push_back(frame_kps[m.second].pt);
    }

    cv::undistortPoints(x1, x1, frame->params.intr, frame->params.distortion, cv::noArray(), frame->params.intr);
    cv::undistortPoints(x2, x2, frame->params.intr, frame->params.distortion, cv::noArray(), frame->params.intr);

    cv::Mat p4ds;
    cv::triangulatePoints(ref_frame->projection_cv, frame->projection_cv, x1, x2, p4ds);

    // std::vector<Eigen::Vector3d> debug_points, debug_colors;

    // loop over all triangulated points and convert from homography to cartecian
    for (int ii = 0; ii < p4ds.cols; ii++)
    {
        const Eigen::Vector3d new_p3d (
                    p4ds.at<float>(0, ii) / p4ds.at<float>(3, ii),
                    p4ds.at<float>(1, ii) / p4ds.at<float>(3, ii),
                    p4ds.at<float>(2, ii) / p4ds.at<float>(3, ii));

        // if the triangulated point is behind the camera or too far away -> ignore 
        if (!utilities::point_in_front(frame->projection, new_p3d) || new_p3d.norm() > TRIANGULATE_DISTANCE_OUTLIER)
            continue;

        extra_3d_points.push_back(new_p3d);

        const cv::Vec3b col = frame->rgb->at<cv::Vec3b>(x2[ii].y, x2[ii].x);
        extra_3d_points_colors.push_back(Eigen::Vector3d((float)(col.val[2]) / 255.0, (float)(col.val[1]) / 255.0, (float)(col.val[0]) / 255.0));

        extra_3d_points_normals.push_back(frame->position - new_p3d);

        // debug_points.push_back(new_p3d);
        // debug_colors.push_back(Eigen::Vector3d((float)(col.val[2]) / 255.0, (float)(col.val[1]) / 255.0, (float)(col.val[0]) / 255.0));
    }

    // auto extra_cloud = std::make_shared<open3d::geometry::PointCloud>(open3d::geometry::PointCloud(debug_points));
    // extra_cloud->colors_ = debug_colors;

    // std::shared_ptr<open3d::geometry::TriangleMesh> camera_mesh = std::make_shared<open3d::geometry::TriangleMesh>(open3d::geometry::TriangleMesh());
    // open3d::io::ReadTriangleMeshFromOBJ("../assets/debug_camera_mesh.obj", *camera_mesh, false);
    // camera_mesh->Transform(frame->transformation);

    // open3d::visualization::DrawGeometries({ extra_cloud, camera_mesh });
}

std::vector<std::pair<uint32_t, uint32_t>> LGraph::find_landmark_feature_matches(const std::shared_ptr<Frame> ref_frame,
        const std::vector<std::pair<uint32_t, uint32_t>>& matches)
{
    // <2d feature, 3d landmark>
    std::vector<std::pair<uint32_t, uint32_t>> feature_lm_correspondences;
    feature_lm_correspondences.reserve(std::min(matches.size(), ref_frame->feature_landmark_lookup.size()));


    for (const auto& flm : ref_frame->feature_landmark_lookup)
    {
        for (const auto& match : matches)
        {
            // feature match was found in landmark lookup
            if (match.first == flm.first)
                feature_lm_correspondences.push_back(std::make_pair(match.second, flm.second));
        }
    }

    feature_lm_correspondences.shrink_to_fit();
    return feature_lm_correspondences;
}

std::vector<std::pair<uint32_t, uint32_t>> LGraph::backpropagate_future_matches(const std::shared_ptr<Frame> ref_frame,
        const std::shared_ptr<Frame> frame,
        const std::vector<std::pair<uint32_t, uint32_t>>& matches)
{
    throw std::runtime_error("this does not work, to the surprise of absolutely no one ");

    // <2d feature, 3d landmark>
    std::vector<std::pair<uint32_t, uint32_t>> feature_lm_correspondences;
    std::vector<uint32_t> ref_feature_ids;
    feature_lm_correspondences.reserve(std::min(matches.size(), ref_frame->feature_landmark_lookup.size()));

    // collect landmark point ids and current frame feature point ids
    // find prev-current matched form prev-landmarks
    for (const auto& flm : ref_frame->feature_landmark_lookup)
    {
        for (const auto& match : matches)
        {
            // feature match was found in landmark lookup
            if (match.first == flm.first)
            {
                feature_lm_correspondences.push_back(std::make_pair(match.second, flm.second));
                ref_feature_ids.push_back(match.first);
            }
        }
    }

    // not enough matches, backpropagate new ones
    // if (feature_lm_correspondences.size() < MIN_MATCH_FEATURE_COUNT)
    if (frames.size() > 5)
    {
        std::vector<std::pair<uint32_t, uint32_t>> new_matches;
        std::shared_ptr<Frame> latest_frame;

        // find a frame until STATISTICAL_FEATURE_COUNT condition not met
        for (int ii = frames.size() - 4; ii >= 0; ii--)
        {
            latest_frame = frames[ii];

            new_matches = features::match_features_flann(latest_frame->descriptors, ref_frame->descriptors);
            new_matches = features::take_only_indexed_matches(new_matches, ref_feature_ids);

            // new_matches = features::match_features_bf_crosscheck(latest_frame->descriptors, ref_desc);

            // new_matches = homography_filter_matches(latest_frame, ref_frame, new_matches);
            // new_matches = features::radius_distance_filter_matches(new_matches, ref_kps, latest_frame->keypoints, frame_kps;

            // DEBUG_visualize_matches(*latest_frame->rgb, *ref_frame->rgb, new_matches, latest_frame->keypoints, ref_frame->keypoints);

            // std::cout << ref_feature_ids.size() << ", " << new_matches.size() << ", " << matches.size() << "\n";

            // frame found, stop searching
            if (new_matches.size() < 250)
                break;
        }

        feature_lm_correspondences.shrink_to_fit();
        create_landmarks_from_matches(latest_frame, frame, new_matches);

        feature_lm_correspondences.clear();

        for (const auto& flm : ref_frame->feature_landmark_lookup)
        {
            for (const auto& match : matches)
            {
                // feature match was found in landmark lookup
                if (match.first == flm.first)
                {
                    feature_lm_correspondences.push_back(std::make_pair(match.second, flm.second));
                    ref_feature_ids.push_back(match.first);
                }
            }
        }
    }

    return feature_lm_correspondences;
}

void LGraph::new_landmarks_standalone(const std::shared_ptr<Frame> frame, const std::vector<cv::Point3f>& tr_angle_points)
{
    /**
     *  - find a triangulatable frame
     *  - match and filter features
     *  - triangulate
     *  - add to frame->feature_landmark_lookup
     */

    std::shared_ptr<Frame> ref_frame = find_triangulatable_point_frame(frame, tr_angle_points);

    std::vector<std::pair<uint32_t, uint32_t>> matches = 
#if defined USE_FLANN
        features::match_features_flann(ref_frame->descriptors, frame->descriptors);
#else
        features::match_features_bf_crosscheck(ref_frame->descriptors, frame->descriptors);
#endif

    // if (matches.size() > HOMOGRAPHY_MIN_FEATURE_COUNT)
    //     matches = homography_filter_matches(ref_frame, frame, matches);
    
    matches = features::radius_distance_filter_matches(matches, ref_frame->keypoints, frame->keypoints, FEATURE_DIST_MAX_RADIUS);

    // DEBUG_visualize_matches(*ref_frame->rgb, *frame->rgb, matches, ref_frame->keypoints, frame->keypoints);
    // visualize_camera_tracks(true);

    std::cout << "created new landmarks: " << matches.size() << "\n";

    // create landmarks and add to feature_landmark_lookup
    create_landmarks_from_matches(ref_frame, frame, matches);
}

void LGraph::backpropagate_new_landmarks_homography(const std::shared_ptr<Frame> frame,
        const cv::Point3f tr_angle_point)
{
    // TODO: find the largest difference frame to match against

    std::shared_ptr<Frame> ref_frame = nullptr;
    const Eigen::Vector3d tr_point (tr_angle_point.x, tr_angle_point.y, tr_angle_point.z);

    if (frames.size() < 4)
        return;

    for (int ii = frames.size() - 3; ii >= 0; ii--)
    {
        const double frame_angle = utilities::calculate_triangulation_angle(frames[ii]->position, frame->position, tr_point);

        if (RAD2DEG(frame_angle) < MIN_TRIANGULATION_ANGLE)
            continue;

        ref_frame = frames[ii];
        break;
    }

    if (!ref_frame)
        return;

    std::vector<std::pair<uint32_t, uint32_t>> matches = 
        features::match_features_bf_crosscheck(ref_frame->descriptors, frame->descriptors);
    matches = homography_filter_matches(ref_frame, frame, matches);


    // filter matches for already existing landmarks

    std::vector<std::pair<uint32_t, uint32_t>> new_matches;
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

    create_landmarks_from_matches(ref_frame, frame, new_matches);

    std::cout << "matches " << matches.size() << " out of " << new_matches.size() << "\n";

    // DEBUG_visualize_matches(*ref_frame->rgb, *frame->rgb, matches, ref_frame->keypoints, frame->keypoints);
    // visualize_camera_tracks(true);

}

std::vector<std::pair<uint32_t, uint32_t>> LGraph::homography_filter_matches(
        const std::shared_ptr<Frame> ref_frame, const std::shared_ptr<Frame> frame,
        const std::vector<std::pair<uint32_t, uint32_t>>& matches)
{
    if (matches.size() < HOMOGRAPHY_MIN_FEATURE_COUNT)
        throw std::runtime_error("not enough matches for computing homography");

    std::vector<cv::Point2f> fpoints1, fpoints2;
    fpoints1.reserve(matches.size());
    fpoints2.reserve(matches.size());

    for (const auto& m : matches)
    {
        fpoints1.push_back(ref_frame->keypoints[m.first].pt);
        fpoints2.push_back(frame->keypoints[m.second].pt);
    }

    cv::Mat homography;

    homography = findHomography(fpoints1, fpoints2, cv::RANSAC, HOMOGRAPHY_RANSAC_THRESHOLD, cv::noArray());

    std::vector<std::pair<uint32_t, uint32_t>> good_matches;
    good_matches.reserve(matches.size());

    for (size_t ii = 0; ii < matches.size(); ii++)
    {
        cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = fpoints1[ii].x;
        col.at<double>(1) = fpoints1[ii].y;
        col = homography * col;
        col /= col.at<double>(2);

        const double dist = sqrt(pow(col.at<double>(0) - fpoints2[ii].x, 2) + pow(col.at<double>(1) - fpoints2[ii].y, 2));
        if (dist < HOMOGRAPHY_FILTER_MAX_DIST)
            good_matches.push_back(matches[ii]);
    }

    good_matches.shrink_to_fit();
    return good_matches;
}


void LGraph::backpropagate_new_landmarks(const std::shared_ptr<Frame> frame,
        const std::shared_ptr<Frame> ref_frame,
        const std::vector<uint32_t>& ref_frame_fids_inv,
        const std::vector<uint32_t>& current_frame_fids_inv,
        const std::vector<std::pair<uint32_t, uint32_t>>& matches,
        const cv::Point3f tr_angle_point)
{
    // traverse the frames backwards, matching the features, until
    // a frame is found which has enough triangulatable movement

    // Create a data structure for holding the to-be-triangulated (tbt) feature chains.
    // Is also used as a lookup for pruning the triangulation list
    std::unordered_map<uint32_t, std::vector<uint32_t>> feature_chain;
    feature_chain.reserve(current_frame_fids_inv.size());

    // used to map feature ids from frame -> last_frame
    std::map<uint32_t, uint32_t> first_current_feature_lookup;

    for (int ii = 0; ii < current_frame_fids_inv.size(); ii++)
    {
        feature_chain.insert(std::make_pair(current_frame_fids_inv[ii], std::vector<uint32_t> { ref_frame_fids_inv[ii] }));
        first_current_feature_lookup.insert(std::make_pair(current_frame_fids_inv[ii], ref_frame_fids_inv[ii]));
    }

    std::vector<uint32_t> last_frame_fids = ref_frame_fids_inv;
    cv::Mat last_frame_descriptor = features::descriptor_from_feature_ids(ref_frame->descriptors, last_frame_fids);
    std::shared_ptr<Frame> last_frame = ref_frame;


    // -3: -1 for size, -2 for frame, -3 for ref_frame
    for (int ii = frames.size() - 3; ii >= 0; ii--)
    {
        const std::shared_ptr<Frame> current_frame = frames[ii];;

        // match the current latest descriptors against a new frame
        std::vector<std::pair<uint32_t, uint32_t>> matches = 
            features::match_features_bf_crosscheck(last_frame_descriptor, current_frame->descriptors);
        matches = features::radius_distance_filter_matches(matches, last_frame->keypoints, current_frame->keypoints, FEATURE_DIST_MAX_RADIUS);

        std::cout << "matches size: " << matches.size() << "\n";

        last_frame_fids.clear();

        // populate tbt feature chain lookup
        for (const auto& match : matches)
        {
            // find match.first in first_current_feature_lookup, acquire the "key", use the key to
            // index into feature_chain, append match.second
            const auto key_iter = std::find_if(
                first_current_feature_lookup.begin(),
                first_current_feature_lookup.end(),
                [match](const auto& fcfl) { return fcfl.second == match.first; });

            if (key_iter == first_current_feature_lookup.end())
                continue;

            const uint32_t key = key_iter->first;
            feature_chain.at(key).push_back(match.second);

            // append the current frame feature id, next iteration last frame id
            last_frame_fids.push_back(match.second);
        }

        // check if enough triangulatable distance

        // if enough movement, stop iterating and triangulate
        throw std::runtime_error("todo");

        last_frame = current_frame;
        last_frame_descriptor = features::descriptor_from_feature_ids(current_frame->descriptors, last_frame_fids);
    }
}


std::shared_ptr<Frame> LGraph::find_triangulatable_movement_frame(const std::shared_ptr<Frame> frame, const double angle_threshold)
{
    // use angle to find good frame

    const Eigen::Vector3d current_pos = frame->position;

    for (int ii = frames.size() - 2; ii >= 0; ii--)
    {
        const std::shared_ptr<Frame> ref_frame = frames[ii];

        if ((ref_frame->position - current_pos).norm() > angle_threshold)
            return ref_frame;
    }

    return nullptr;
}

std::shared_ptr<Frame> LGraph::find_triangulatable_point_frame(const std::shared_ptr<Frame> frame,
        const std::vector<cv::Point3f>& tr_angle_points, const double angle_threshold)
{
    const Eigen::Vector3d current_pos = frame->position;

    // take the average of all of the triangulated points
    const Eigen::Vector3d tr_point_eigen = [&tr_angle_points] {
        cv::Point3f trp (0.0, 0.0, 0.0);
        for (const auto& p : tr_angle_points)
            trp += p;
        return Eigen::Vector3d(trp.x, trp.y, trp.z) / (double)tr_angle_points.size();

    }();

    for (int ii = frames.size() - 2; ii >= 0; ii--)
    {
        const std::shared_ptr<Frame> ref_frame = frames[ii];

        const double frame_angle = utilities::calculate_triangulation_angle(frames[ii]->position, frame->position, tr_point_eigen);

        if (RAD2DEG(frame_angle) < angle_threshold)
            continue;

        std::cout << "use frame id " << ii << " for triangulation, current frame is " << frames.size() - 1 << "\n";
        return ref_frame;
    }

    std::cout << "use frame id 0 for triangulation\n";
    return frames[0];
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

    // std::cout << "new landmarks: " << new_matches.size() << "\n";

    // std::cout << ref_frame->position.transpose() << ", " << frame->position.transpose() << "\n";

    create_landmarks_from_matches(ref_frame, frame, new_matches);
}

#ifdef USE_OPEN3D
void LGraph::visualize_camera_tracks(const bool visualize_landmarks, bool generate_mesh) const
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
        std::vector<Eigen::Vector3d> landmark_colors;
        std::vector<Eigen::Vector3d> landmark_normals;

        for (const auto& lm : landmarks)
        {
            landmark_points.push_back(Eigen::Vector3d(lm.location.x, lm.location.y, lm.location.z));
            landmark_colors.push_back(lm.color);
            landmark_normals.push_back(lm.normal);
        }

        auto lms_cloud = std::make_shared<open3d::geometry::PointCloud>(open3d::geometry::PointCloud(landmark_points));
        lms_cloud->colors_ = landmark_colors;
        lms_cloud->normals_ = landmark_normals;

        if (generate_mesh)
        {
            auto mesh_cloud = std::make_shared<open3d::geometry::PointCloud>(open3d::geometry::PointCloud(extra_3d_points));
            mesh_cloud->colors_ = extra_3d_points_colors;
            mesh_cloud->normals_ = extra_3d_points_normals;

            *mesh_cloud += *lms_cloud;

            // mesh_cloud->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.5, 16));

            auto [new_mesh, trash] = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*mesh_cloud, MESH_POISSON_DEPTH);
            new_mesh = new_mesh->FilterSmoothLaplacian(LAPLACIAN_ITERATIONS, LAPLACIAN_LAMBDA);

            // new_mesh->Translate(Eigen::Vector3d(0.0, 20.0, 0.0));

            debug_cameras.push_back(new_mesh);
            // debug_cameras.push_back(mesh_cloud);
        }
        else
        {
            debug_cameras.push_back(lms_cloud);

            auto extra_cloud = std::make_shared<open3d::geometry::PointCloud>(open3d::geometry::PointCloud(extra_3d_points));
            extra_cloud->colors_ = extra_3d_points_colors;
            extra_cloud->normals_ = extra_3d_points_normals;

            // debug_cameras.push_back(extra_cloud);
        }
    }

    open3d::visualization::DrawGeometries(debug_cameras, "track visualization", 1920, 1080);
}
#endif

void LGraph::print_camera_tracks() const
{
    for (int ii = 0; ii < frames.size(); ii++)
    {
        std::cout << "frame " << ii << ": " << frames[ii]->position.transpose() << "\n";
    }
}