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
	 * 	@brief Holds 
	 */
	struct StreamHandle
	{
		std::shared_ptr<tcp::socket> stream_socket;
		std::string ip;
		uint32_t port;
	};

	/**
	 * 	@brief Streams a set of points and their respective point colors to
	 * 		a remote host. If frame is not nullptr, streams the appropriate 
	 * 		positional info as well
	 */
	inline void stream_points_camera(const std::vector<Eigen::Vector3d>& points,
		const std::vector<Eigen::Vector3d>& colors, const std::shared_ptr<Frame> frame,
		const StreamHandle& stream_handle)
	{
		// convert the points and colors into transfer-friendly buffers
		std::vector<double> point_data, color_data;
		point_data.reserve(points.size() * 3);
		color_data.reserve(colors.size() * 3);
		for (int ii = 0; ii < points.size(); ii++)
		{
			const auto p = points[ii];
			const auto c = colors[ii];

			point_data.push_back(p.x());
			point_data.push_back(p.y());
			point_data.push_back(p.z());

			color_data.push_back(c.x());
			color_data.push_back(c.y());
			color_data.push_back(c.z());
		}

		std::cout << "stream points and frame data\n";

		// construct a header, which contains what will be sent
		std::vector<uint32_t> header;

		// how many points will be sent
		header.push_back(point_data.size());

		// whether camera location data will be sent
		header.push_back((uint32_t)(frame != nullptr));

		std::cout << "stream header: " << header[0] << ", " << header[1] << "\n";


		boost::system::error_code err;
		boost::asio::write(*stream_handle.stream_socket, boost::asio::buffer(header), err);

		// transmit the data

		if (err) { std::cout << "failed to send header: " << err.message() << "\n"; return; }

		boost::asio::write(*stream_handle.stream_socket, boost::asio::buffer(point_data), err);
		if (err) { std::cout << "failed to send points: " << err.message() << "\n"; return; }

		boost::asio::write(*stream_handle.stream_socket, boost::asio::buffer(color_data), err);
		if (err) { std::cout << "failed to send point colors: " << err.message() << "\n"; return; }

		if (frame)
		{
			// convert the frame positional data into transfer-friendly buffer format
			std::vector<double> frame_data;
			{
				frame_data.push_back(frame->position.x());
				frame_data.push_back(frame->position.y());
				frame_data.push_back(frame->position.z());

				frame_data.push_back(frame->rotation(0, 0));
				frame_data.push_back(frame->rotation(0, 1));
				frame_data.push_back(frame->rotation(0, 2));

				frame_data.push_back(frame->rotation(1, 0));
				frame_data.push_back(frame->rotation(1, 1));
				frame_data.push_back(frame->rotation(1, 2));

				frame_data.push_back(frame->rotation(2, 0));
				frame_data.push_back(frame->rotation(2, 1));
				frame_data.push_back(frame->rotation(2, 2));				
			}

			boost::asio::write(*stream_handle.stream_socket, boost::asio::buffer(frame_data), err);
			if (err) { std::cout << "failed to send frame data: " << err.message() << "\n"; return; }
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

		return StreamHandle
		{
			sock,
			ip,
			port
		};
	}

}