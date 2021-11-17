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
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#ifdef USE_OPEN3D
    #include <Open3D/Open3D.h>
#endif

#include "frame.h"
#include "constants.h"
#include "utilities.h"
#include "debug_functions.h"
#include "timer.h"
#include "point_streamer.h"


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

    Eigen::Vector3d normal;
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
    bool localize_frame(std::shared_ptr<Frame> frame);

#ifdef USE_OPEN3D
    /**
     *  @brief Visualizes the camera tracks
     */
    void visualize_camera_tracks(const bool visualize_landmarks = false, bool generate_mesh = false) const;
#endif

    /**
     *  @brief A terminal friendly camera location print-out
     */
    void print_camera_tracks() const;

private:

    std::vector<std::pair<uint32_t, uint32_t>> find_landmark_feature_matches(const std::shared_ptr<Frame> ref_frame,
        const std::vector<std::pair<uint32_t, uint32_t>>& matches);

    /**
     *  @brief Localizes a frame in reference to ref_frame
     *  @return <position, rotation_matrix>
     */
    bool localize_frame_essential(const std::shared_ptr<Frame> ref_frame, std::shared_ptr<Frame> frame);

    /**
     *  @brief Localized a frame using the PnP algorithm
     *  @return <position, rotation matrix>
     */
    bool localize_frame_pnp(const std::shared_ptr<Frame> prev_frame, std::shared_ptr<Frame> frame);

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
     *  @brief Filters matches using homography, i.e. computer the homography
     *      between two frames and reprojects the features of ref_frame into 
     *      the transformation of frame. Then, checks the distance between the
     *      reprojected feature, if it is over threshold -> remove the feature
     */
    static std::vector<std::pair<uint32_t, uint32_t>> homography_filter_matches(
        const std::shared_ptr<Frame> ref_frame, const std::shared_ptr<Frame> frame,
        const std::vector<std::pair<uint32_t, uint32_t>>& matches);


    /**
     *  @brief Create new landmarks between frame and suitably far-enough frame.
     *      Does not check for duplicates.
     */
    void new_landmarks_standalone(const std::shared_ptr<Frame> frame, const std::vector<cv::Point3f>& tr_angle_points);

    /**
     *  @brief Traverses the alredy localized frames backwards, and finds the first
     *      one with significant enough movement to be valid for triangulation
     */
    std::shared_ptr<Frame> find_triangulatable_movement_frame(const std::shared_ptr<Frame> frame,
        const double angle_threshold = TRIANGULATE_DIST_DIFF_MAGNITUDE);

    /**
     *  @brief Traverses the already localized frames backwards, and find the first 
     *      frame, which where tr_angle_point can be triangulated with at least difference of 
     *      angle_threshold. If no frame was found, return the first frame.
     */
    std::shared_ptr<Frame> find_triangulatable_point_frame(const std::shared_ptr<Frame> frame,
        const std::vector<cv::Point3f>& tr_angle_points, const double angle_threshold = MIN_TRIANGULATION_ANGLE);

    /**
     *  @brief Projects more 3D points more liberally and with minimal filtering.
     *      Quality not guaranteed.
     */
    void project_more_points(const std::shared_ptr<Frame> ref_frame, const std::shared_ptr<Frame> frame);

    std::vector<std::shared_ptr<Frame>> frames;

    std::vector<Landmark> landmarks;

    std::vector<Eigen::Vector3d> current_frame_points;
    std::vector<Eigen::Vector3d> current_frame_point_colors;

    /**
     *  Extra 3D points, not used in frame localization
     */
    std::vector<Eigen::Vector3d> extra_3d_points;
    std::vector<Eigen::Vector3d> extra_3d_points_colors;
    std::vector<Eigen::Vector3d> extra_3d_points_normals;

    networking::StreamHandle stream_handle;
    
};

};