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

#include "frame.h"
#include "constants.h"

namespace k3d
{

class LGraph
{

public:

    LGraph();
    ~LGraph();

    /**
     *  @brief Adds a new frame into the graph and 
     *      localizes it in respect to previous frames
     */
    void localize_frame(std::shared_ptr<Frame> frame);


private:


};

};