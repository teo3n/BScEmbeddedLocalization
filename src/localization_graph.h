/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	Localizes frames in respect to previously localized frames
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <map>
#include <unordered_map>
#include <opencv2/core/types.hpp>
#include <vector>

#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <Open3D/Open3D.h>

#include "frame.h"
#include "constants.h"
#include "utilities.h"
#include "debug_functions.h"
#include "timer.h"


namespace k3d
{

struct Landmark
{
    cv::Point3f location;
    std::vector<cv::Mat> descriptors;
    std::vector<std::shared_ptr<Frame>> view_frames;
    std::vector<cv::Point2f> feature_2d_points;

    std::vector<std::shared_ptr<Frame>> triangulate_frames;
    std::vector<cv::Point2f> triangulate_2d_points;

    Eigen::Vector3d color;
};

class LGraph
{

public:

    LGraph();
    ~LGraph() {};

    /**
     *  @brief Adds a new frame into the graph and 
     *      localizes it in respect to previous frames
     */
    void localize_frame(std::shared_ptr<Frame> frame);

    /**
     *  @brief Visualizes the camera tracks
     */
    void visualize_camera_tracks(const bool visualize_landmarks = false) const;

private:

    /**
     *  @brief Localizes a frame in reference to ref_frame
     *  @return <position, rotation_matrix>
     */
    void localize_frame_essential(const std::shared_ptr<Frame> ref_frame, std::shared_ptr<Frame> frame);

    /**
     *  @brief Localized a frame using the PnP algorithm
     *  @return <position, rotation matrix>
     */
    void localize_frame_pnp(const std::shared_ptr<Frame> prev_frame, std::shared_ptr<Frame> frame);

    /**
     *  @brief Creates new landmarks from feature matches
     */
    void create_landmarks_from_matches(const std::shared_ptr<Frame> ref_frame, 
        const std::shared_ptr<Frame> frame, const std::vector<std::pair<uint32_t, uint32_t>>& matches);

    /**
     *  @brief Updates the frame feature-landmark lookup and the
     *      corresponding landmarks
     */
    void update_landmarks(const std::shared_ptr<Frame> frame, 
        std::vector<uint32_t>& feature_ids, std::vector<uint32_t>& landmark_ids);

    /**
     *  @brief Creates new landmarks from previously unmatched features
     *  @param feature_ids, the previously matched features
     */
    void new_landmarks_from_matched(const std::shared_ptr<Frame> ref_frame,
        const std::shared_ptr<Frame> frame);

    /**
     *  @brief Propagates new landmarks from the features, which were not found
     *      to be current landmarks (the inverse of current_frame_feature_points) 
     *  @param ref_frame_fids_inv, the features, which correspond to frame's
     *      matched landmarks, but not landmarks
     *  @param matches, all of the matches between frame and ref_frame
     *  @param tr_angle_point, the 3D point which is used to calculate
     *      the triangulation angle
     */
    void backpropagate_new_landmarks(const std::shared_ptr<Frame> frame,
        const std::shared_ptr<Frame> ref_frame,
        const std::vector<uint32_t>& ref_frame_fids_inv,
        const std::vector<uint32_t>& current_frame_fids_inv,
        const std::vector<std::pair<uint32_t, uint32_t>>& matches,
        const cv::Point3f tr_angle_point);

    /**
     *  @brief Propagates new landmarks by traversing the frame-tree backwards
     *      until a frame is found, which fulfills the cheirality constraint
     *      to a satisfying degree. A more simplified version of backpropagate_new_landmarks
     */
    void backpropagate_new_landmarks_homography(const std::shared_ptr<Frame> frame,
        const cv::Point3f tr_angle_point);

    /**
     *  @brief Filters matches using homography, i.e. computer the homography
     *      between two frames and reprojects the features of ref_frame into 
     *      the transformation of frame. Then, checks the distance between the
     *      reprojected feature, if it is over threshold -> remove the feature
     */
    static std::vector<std::pair<uint32_t, uint32_t>> homography_filter_matches(
        const std::shared_ptr<Frame> ref_frame, const std::shared_ptr<Frame> frame,
        const std::vector<std::pair<uint32_t, uint32_t>>& matches);

    /**
     *  @brief Traverses the alredy localized frames backwards, and finds the first
     *      one with significant enough movement to be valid for triangulation
     */
    std::shared_ptr<Frame> find_triangulatable_movement_frame(const std::shared_ptr<Frame> frame);

    std::vector<std::shared_ptr<Frame>> frames;

    std::vector<Landmark> landmarks;
    
};

};