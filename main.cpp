/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	1.9.2021
 * 
 * 	The bachelor's thesis project, 
 * 	3D localization and 3D surface reconstruction
 * 	software for microprocessors
 */

#include <iostream>
#include "src/camera_module.h"

using namespace k3d;


int main()
{
	std::cout << "begin\n";

	Camera cam(0);

	const cv::Mat frame = cam.get_frame_cv();



	return 0;
}