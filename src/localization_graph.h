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


namespace k3d
{

struct Landmark
{
    cv::Point3f location;
    std::vector<cv::Mat> descriptors;
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
    void visualize_camera_tracks() const;

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


    std::vector<std::shared_ptr<Frame>> frames;

    std::vector<Landmark> landmarks;
};

};