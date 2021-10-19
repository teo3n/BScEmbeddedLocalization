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
	const Mat34 P = P_from_KRt(R.transpose(), -R.transpose() * t, intr_eigen);

	return std::make_pair(T, P);
}

/**
 *  @brief converts the given point into homogenous coordinates
 */
inline Eigen::Vector4d euclidean_to_homogenous(const Eigen::Vector3d& eucl)
{
    return Eigen::Vector4d(eucl.x(), eucl.y(), eucl.z(), 1.0);
}

/**
 *  @brief converts the given homogenous coordinate into a 3d point
 */
inline Eigen::Vector3d homogeneous_to_euclidean(const Eigen::Vector4d& H)
{
    double w = H(3);
    Eigen::Vector3d eucl;
    eucl << H(0) / w, H(1) / w, H(2) / w;
    return eucl;
}


/**
 *  @brief checks if the given point is in front of the camera
 *  @param P projection matrix
 *  @param p the given 3d point
 */
inline bool point_infront_of_camera(const Mat34 P, const Eigen::Vector3d p)
{
    const Eigen::Vector4d h = euclidean_to_homogenous(p);

    double condition_1 = P.row(2).dot(h) * h[3];
    double condition_2 = h[2] * h[3];

    if (condition_1 > 0 && condition_2 > 0)
        return true;
    else
        return false;
}

/**
 *  @brief checks if point is in front of the given projection matrix,
 *      cheirality constraint.
 */
inline bool point_in_front(const Mat34& pmat, const Eigen::Vector3d& p3d)
{
  return pmat.row(2).dot(p3d.homogeneous()) >= std::numeric_limits<double>::epsilon();
}

/**
 *  @brief calculates the triangulation angle, to satisfy cheirality constraint
 *  @note borrowed from https://github.com/colmap/colmap/blob/9f3a75ae9c72188244f2403eb085e51ecf4397a8/src/base/triangulation.cc
 */
inline double calculate_triangulation_angle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3d)
{
  const double baseline_length_squared = (proj_center1 - proj_center2).squaredNorm();

  const double ray_length_squared1 = (point3d - proj_center1).squaredNorm();
  const double ray_length_squared2 = (point3d - proj_center2).squaredNorm();

  // Using "law of cosines" to compute the enclosing angle between rays.
  const double denominator = 2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
  if (denominator == 0.0)
    return 0.0;

  const double nominator = ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
  const double angle = std::abs(std::acos(nominator / denominator));

  // Triangulation is unstable for acute angles (far away points) and
  // obtuse angles (close points), so always compute the minimum angle
  // between the two intersecting rays.
  return std::min(angle, M_PI - angle);
}


}
