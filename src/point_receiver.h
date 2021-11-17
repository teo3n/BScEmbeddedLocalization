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

		std::cout << "received: " << recv_buf.size() << "\n";

		std::vector<float> data_buf (recv_buf.size(), 0);
		std::copy(recv_data, recv_data + recv_buf.size(), data_buf.begin());

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
			return -1;
		}

		const uint32_t* recv_data = boost::asio::buffer_cast<const uint32_t*>(recv_buf.data());
		return *recv_data;
	}


	/**
	 * 	@brief Streams a set of points and their respective point colors to
	 * 		a remote host. If frame is not nullptr, streams the appropriate 
	 * 		positional info as well
	 */
	inline void receive_points_data(std::vector<Eigen::Vector3d>& points,
		std::vector<Eigen::Vector3d>& colors, Eigen::Vector3d& position, Eigen::Matrix3d& rotation,
		const StreamHandle& stream_handle)
	{
		std::cout << "receiving...\n";

		const uint32_t buf_size = receive_buffer_data_size(stream_handle);
		const auto recv_data = receive_buffer(stream_handle, buf_size);

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