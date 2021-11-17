/**
 *  Teo Niemirepo
 * 	teo.niemirepo@tuni.fi
 * 	17.11.2021
 * 
 * 	A collection of functions used to stream data to the 
 *	remote host.
 */

#pragma once

#include <Eigen/Eigen>
#include <boost/asio/buffer.hpp>
#include <boost/asio/completion_condition.hpp>
#include <boost/asio/detail/array_fwd.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/address.hpp>
#include <boost/asio/write.hpp>
#include <boost/system/error_code.hpp>
#include <iostream>
#include <memory>
#include <boost/asio.hpp>
#include <vector>

#include "frame.h"
#include "constants.h"


namespace k3d::networking
{

	using namespace boost::asio;
	using ip::tcp;

	/**
	 * 	@brief Holds tcp connection related data
	 */
	struct StreamHandle
	{
		std::shared_ptr<tcp::socket> stream_socket;
		std::string ip;
		uint32_t port;
	};

	/**
	 * 	@brief Streams a set of points and their respective point colors to
	 * 		a remote host.Streams the appropriate positional info for frame as well
	 */
	inline void stream_points_camera(const std::vector<Eigen::Vector3d>& points,
		const std::vector<Eigen::Vector3d>& colors, const std::shared_ptr<Frame> frame,
		const StreamHandle& stream_handle)
	{
		// std::cout << "stream points and frame data\n";

		// cut the used data in half, no reason to transmit doubles when floats are "good enough"
		std::vector<float> data_buffer;

		// frame position (vec3), frame rotation (mat3x3), len of points, len of colors
		data_buffer.reserve(3 + 3*3 + points.size() * 3 + colors.size() * 3);

		// add point position
		for (int ii = 0; ii < points.size(); ii++)
		{
			const auto p = points[ii];

			data_buffer.push_back((float)p.x());
			data_buffer.push_back((float)p.y());
			data_buffer.push_back((float)p.z());
		}

		// add point colors
		for (int ii = 0; ii < colors.size(); ii++)
		{
			const auto c = colors[ii];

			data_buffer.push_back((float)c.x());
			data_buffer.push_back((float)c.y());
			data_buffer.push_back((float)c.z());
		}

		// add rotation and position
		{
			data_buffer.push_back((float)frame->position.x());
			data_buffer.push_back((float)frame->position.y());
			data_buffer.push_back((float)frame->position.z());

			data_buffer.push_back((float)frame->rotation(0, 0));
			data_buffer.push_back((float)frame->rotation(0, 1));
			data_buffer.push_back((float)frame->rotation(0, 2));

			data_buffer.push_back((float)frame->rotation(1, 0));
			data_buffer.push_back((float)frame->rotation(1, 1));
			data_buffer.push_back((float)frame->rotation(1, 2));

			data_buffer.push_back((float)frame->rotation(2, 0));
			data_buffer.push_back((float)frame->rotation(2, 1));
			data_buffer.push_back((float)frame->rotation(2, 2));
		}

		auto send_buffer = boost::asio::buffer(data_buffer);

		std::cout << "sending " << send_buffer.size() << " bytes, compared to actual buffer size of " << data_buffer.size() << " values\n";

		const uint32_t send_bytes = send_buffer.size();

		boost::system::error_code err;

		// transmit amount of bytes
		boost::asio::write(*stream_handle.stream_socket, boost::asio::buffer({ send_bytes }), boost::asio::transfer_exactly(4), err);

		// transmit the data
		boost::asio::write(*stream_handle.stream_socket, send_buffer, boost::asio::transfer_exactly(send_buffer.size()), err);

		if (err)
		{
			std::cout << "failed to send points: " << err.message() << "\n";
			return;
		}
	}

	/**
	 * 	@brief Constructs a stream handle and a socket connection
	 */
	inline StreamHandle acquire_stream_handle(const std::string& ip, const uint32_t port)
	{
		boost::asio::io_service ios;
		auto sock = std::make_shared<tcp::socket>(ios);
		sock->connect(tcp::endpoint(boost::asio::ip::address::from_string(ip), port));

		std::cout << "remote ip: " << sock->remote_endpoint().address().to_string() << "\n";

		return StreamHandle
		{
			sock,
			ip,
			port
		};
	}

}