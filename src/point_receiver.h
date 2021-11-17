/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	17.11.2021
 * 
 * 	A collection of functions used to receive data from 
 * 	a remote localizer client
 */

#pragma once

#include <Eigen/Eigen>
#include <boost/asio/buffer.hpp>
#include <boost/asio/completion_condition.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/address.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/asio/write.hpp>
#include <boost/system/error_code.hpp>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <boost/asio.hpp>
#include <vector>

#include "frame.h"
#include "constants.h"
#include "point_streamer.h"


namespace k3d::networking
{

	using namespace boost::asio;
	using ip::tcp;

	inline std::vector<float> receive_buffer(const StreamHandle& stream_handle, const uint32_t num_bytes)
	{
		boost::system::error_code err;

		boost::asio::streambuf recv_buf;
		boost::asio::read(*stream_handle.stream_socket, recv_buf, boost::asio::transfer_exactly(num_bytes), err);

		if (err)
		{
			std::cout << "failed to receive data buffer: " << err.message() << "\n";
			return {};
		}

		const float* recv_data = boost::asio::buffer_cast<const float*>(recv_buf.data());

		std::cout << "received " << recv_buf.size() << " bytes out of " << num_bytes << " bytes wanted\n";

		std::vector<float> data_buf (recv_buf.size(), 0);
		// std::memcpy(&data_buf[0], recv_data, recv_buf.size() * sizeof(float));
		std::copy(recv_data, recv_data + recv_buf.size() / 4, data_buf.begin());

		data_buf.shrink_to_fit();

		return data_buf;
	}

	inline uint32_t receive_buffer_data_size(const StreamHandle& stream_handle)
	{
		boost::system::error_code err;

		boost::asio::streambuf recv_buf;
		boost::asio::read(*stream_handle.stream_socket, recv_buf, boost::asio::transfer_exactly(4), err);

		if (err)
		{
			std::cout << "failed to receive data size: " << err.message() << "\n";
			return 0;
		}

		const uint32_t* recv_data = boost::asio::buffer_cast<const uint32_t*>(recv_buf.data());
		return *recv_data;
	}


	/**
	 * 	@brief Streams a set of points and their respective point colors to
	 * 		a remote host. If frame is not nullptr, streams the appropriate 
	 * 		positional info as well.
	 * 	@return If receiving is finished, determined by remote connection closed
	 */
	inline bool receive_points_data(std::vector<Eigen::Vector3d>& points,
		std::vector<Eigen::Vector3d>& colors, Eigen::Vector3d& position, Eigen::Matrix3d& rotation,
		const StreamHandle& stream_handle)
	{
		std::cout << "receiving...\n";

		const uint32_t buf_size = receive_buffer_data_size(stream_handle);
		if (buf_size == 0)
			return true;

		const std::vector<float> recv_data = receive_buffer(stream_handle, buf_size);

		// 3 for position, 3*3 for Mat3x3, points and colors -> / 2
		const uint32_t points_raw_buf_size = ((buf_size / 4) - 3 - 3*3) / 2;

		points.reserve(points_raw_buf_size / 3);
		colors.reserve(points_raw_buf_size / 3);

		// recover points and colors
		for (int ii = 0; ii < points_raw_buf_size; ii += 3)
		{
			points.push_back(Eigen::Vector3d(
				recv_data[ii + 0],
				recv_data[ii + 1],
				recv_data[ii + 2]));

			colors.push_back(Eigen::Vector3d(
				recv_data[points_raw_buf_size + ii + 0],
				recv_data[points_raw_buf_size + ii + 1],
				recv_data[points_raw_buf_size + ii + 2]));
		}

		std::cout << "first data entry: " << recv_data[0] << ", " << recv_data[1] << ", " << recv_data[2] << "\n";
		std::cout << "points_raw_buf_size: " << points_raw_buf_size << "\n";

		// recover the position
		position.x() = recv_data[points_raw_buf_size * 2 + 0];
		position.y() = recv_data[points_raw_buf_size * 2 + 1];
		position.z() = recv_data[points_raw_buf_size * 2 + 2];

		// recover the rotation matrix
		rotation(0, 0) = recv_data[points_raw_buf_size * 2 + 3 + 0];
		rotation(0, 1) = recv_data[points_raw_buf_size * 2 + 3 + 1];
		rotation(0, 2) = recv_data[points_raw_buf_size * 2 + 3 + 2];

		rotation(1, 0) = recv_data[points_raw_buf_size * 2 + 3 + 3];
		rotation(1, 1) = recv_data[points_raw_buf_size * 2 + 3 + 4];
		rotation(1, 2) = recv_data[points_raw_buf_size * 2 + 3 + 5];

		rotation(2, 0) = recv_data[points_raw_buf_size * 2 + 3 + 6];
		rotation(2, 1) = recv_data[points_raw_buf_size * 2 + 3 + 7];
		rotation(2, 2) = recv_data[points_raw_buf_size * 2 + 3 + 8];

		return false;
	}

	/**
	 * 	@brief Constructs a stream handle and a socket connection
	 */
	inline StreamHandle await_stream_connection(const uint32_t port)
	{
		boost::asio::io_service ios;
		tcp::acceptor acc(ios, tcp::endpoint(tcp::v4(), port));
		auto sock = std::make_shared<tcp::socket>(ios);

		// wait for connection
		acc.accept(*sock);

		std::cout << "connection accepted from: " << sock->remote_endpoint().address().to_string() << "\n";

		return StreamHandle
		{
			sock,
			"",
			port
		};
	}

}