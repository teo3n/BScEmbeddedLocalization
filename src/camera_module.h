/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	A camera interface abstraction
 */

#pragma once

#include <iostream>
#include <memory>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


namespace k3d
{


/**
 * 	@brief An abstraction class for handling different 
 * 		methods of frame capture, in the case cv::VideoCapture
 * 		is not sufficient
 */
class Camera
{

public:

	Camera(const cv::uint32_t stream_id);
	~Camera();

	/**
	 * 	@brief Gets a new frame from the camera stream
	 */
	cv::Mat get_frame_cv();


private:

	std::unique_ptr<cv::VideoCapture> video_stream;

};


}
