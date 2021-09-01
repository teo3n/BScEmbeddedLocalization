#include "camera_module.h"
#include <opencv2/videoio.hpp>

using namespace k3d;


Camera::Camera(const cv::uint32_t stream_id) :
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
