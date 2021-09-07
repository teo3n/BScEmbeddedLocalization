/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	7.9.2021
 * 
 * 	A collection of helper functions that cannot be easily
 * 	divided into separate headers
 */

#pragma once

#include <iostream>
#include <Eigen/Eigen>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "constants.h"


namespace k3d::utilities
{

/**
 * 	@brief restructures a buffer of data into shape
 */
template <typename T>
cv::Mat restrucure_mat(T* data, int rows, int cols, int chs = 1)
{
    cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataType<T>::type, chs));
    memcpy(mat.data, data, rows*cols*chs * sizeof(T));
    return mat;
}


/**
 *	@brief combined R and t into a 4x4 transformation matrix, [R|t]
 */
inline Eigen::Matrix4d compose_transform_Rt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
	Eigen::Matrix4d transform;
    Mat34 temp43;
    temp43 << R, t;
    transform << temp43, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0).transpose();
    return transform;
}

/**
 * 	@brief creates a projection matrix from k*[R|t], where intr = k
 */
inline Mat34 P_from_KRt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::Matrix3d& intr)
{
	Mat34 P;

	P.block<3, 3>(0, 0) = R;
    P.col(3) = t;
    P = intr * P;

	return P;
}

/**
 * 	@brief creates transformation and projection matrices from rotation and position
 */
inline std::pair<Eigen::Matrix4d, Mat34> TP_from_Rt(
	const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const cv::Mat& intr)
{
	const Eigen::Matrix4d T = compose_transform_Rt(R, t);

	Eigen::Matrix3d intr_eigen;
	cv::cv2eigen(intr, intr_eigen);
	const Mat34 P = P_from_KRt(R, t, intr_eigen);

	return std::make_pair(T, P);
}

}
