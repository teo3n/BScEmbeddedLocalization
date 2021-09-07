/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	Localizes frames in respect to previously localized frames
 */

#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "frame.h"
#include "constants.h"
#include "utilities.h"
#include "debug_functions.h"


namespace k3d
{

class LGraph
{

public:

    LGraph() {};
    ~LGraph() {};

    /**
     *  @brief Adds a new frame into the graph and 
     *      localizes it in respect to previous frames
     */
    void localize_frame(std::shared_ptr<Frame> frame);


private:

    /**
     *  @brief localizes a frame in reference to ref_frame
     *  @return <position, rotation_matrix>
     */
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> localize_frame_essential(
        const std::shared_ptr<Frame> ref_frame, std::shared_ptr<Frame> frame);

    std::vector<std::shared_ptr<Frame>> frames;

};

};