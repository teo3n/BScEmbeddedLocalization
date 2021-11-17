#include "camera_module.h"
#include <opencv2/videoio.hpp>

using namespace k3d;


Camera::Camera(const uint32_t stream_id) :
	video_stream(std::make_unique<cv::VideoCapture>(cv::VideoCapture(stream_id)))
{

}

Camera::~Camera() { }

cv::Mat Camera::get_frame_cv()
{
	cv::Mat frame;
	if (!video_stream->read(frame))
	{
		throw std::runtime_error("could not read frame from stream");
	}

	return frame;
}


void Camera::set_resolution(const uint32_t width, const uint32_t height)
{
	video_stream->set(cv::CAP_PROP_FRAME_WIDTH, (double)width);
	video_stream->set(cv::CAP_PROP_FRAME_HEIGHT, (double)height);
}