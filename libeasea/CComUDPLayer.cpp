#include "include/CComUDPLayer.h"

#include <boost/asio/placeholders.hpp>
#include <boost/bind/bind.hpp>

#include <sstream>
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

using namespace boost::asio;
using namespace boost::asio::ip;

CComSharedContext::CComSharedContext() : ctx(1), thread([this]() {
		auto work = make_work_guard(ctx.get_executor());
		ctx.run();
		/*std::cerr << "io_service ended.\n";*/})
{
}

CComUDPServer::CComUDPServer(unsigned short port) : socket(CComSharedContext::get(), udp::endpoint(udp::v4(), port))
{
	//std::cerr << "Server started on port: " << port << "\n";
	start_receive();
}

bool CComUDPServer::has_data() const
{
	return recv_queue.size() > 0;
}

std::vector<char> CComUDPServer::consume()
{
	if (!has_data())
		throw std::runtime_error("No data");
	auto ret = recv_queue.front();
	recv_queue.pop();
	return ret;
}

void CComUDPServer::start_receive()
{
	socket.async_receive_from(boost::asio::buffer(recv_buffer), last_endpoint,
				  boost::bind(&CComUDPServer::handle_receive, this, placeholders::error,
					      boost::asio::placeholders::bytes_transferred));
}

void CComUDPServer::handle_receive(const boost::system::error_code& error, std::size_t bytes_transfered)
{
	//std::cerr << "Received message of size: " << bytes_transfered << "\n";
	if (error) {
		std::cerr << "UDP error: " << error.message() << "\n";
	} else if (bytes_transfered != 0) {
		//std::cerr << "Appending message to buffer" << std::endl;
		std::vector<char> tmp(recv_buffer.begin(), recv_buffer.begin() + bytes_transfered);
		recv_queue.push(std::move(tmp));
		print_stats();
	}
	start_receive();
}

void CComUDPServer::print_stats() const
{
	std::istringstream iss(recv_queue.front().data());
	std::string tmp, last;
	while (std::getline(iss, tmp, ' ')) {
		last = std::move(tmp);
	}

	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);
	auto fmt = std::put_time(std::localtime(&in_time_t), "%H:%M:%S");

	std::stringstream ss;
	std::cout << "[" << fmt << "]"
		  << " Received individual (fitness = " << last << ") from " << last_endpoint.address().to_string()
		  << ":" << last_endpoint.port();
}

CComUDPClient::CComUDPClient(std::string const& ip, unsigned short port, std::string client_name_)
	: client_name(std::move(client_name_)), socket(CComSharedContext::get())
{
	udp::resolver resolver(CComSharedContext::get());
	dest = *resolver.resolve(udp::v4(), ip, client_name).begin(); // throw if error (safe)
	dest.port(port);
	socket.open(udp::v4());
}

std::string const& CComUDPClient::getClientName() const
{
	return client_name;
}

void CComUDPClient::send(std::string const& individual)
{
	socket.send_to(boost::asio::buffer(individual), dest);
}

std::string CComUDPClient::getIP() const
{
	return dest.address().to_string();
}

unsigned short CComUDPClient::getPort() const
{
	return dest.port();
}

bool isLocalMachine(std::string const& address)
{
	try {
		udp::resolver resolver(CComSharedContext::get());
		udp::endpoint ep = *resolver.resolve(udp::v4(), address, "resolve").begin(); // throw if error (safe)
		return ep.address().is_loopback();
	} catch (...) {
		return false;
	}
}

// NOTE: don't fall in the trap of using a regex, it is more complicated, delegate to boost
bool checkValidLine(std::string const& line)
{
	auto sep = std::find(std::begin(line), std::end(line), ':');
	if (sep == std::end(line))
		return false;
	auto ip = std::string(std::begin(line), sep);
	auto port = std::string(sep + 1, std::end(line));

	try {
		auto p = std::stoi(port);
		(void)(p);
		udp::resolver resolver(CComSharedContext::get());
		udp::endpoint ep = *resolver.resolve(udp::v4(), ip, "resolve").begin(); // throw if error (safe)
		(void)(ep);
	} catch (...) {
		return false;
	}
	return true;
}

std::unique_ptr<CComUDPClient> parse_line(std::string const& line)
{
	auto sep = std::find(std::begin(line), std::end(line), ':');
	if (sep == std::end(line))
		throw std::runtime_error("no ':' found");
	auto ip = std::string(std::begin(line), sep);
	auto port = std::string(sep + 1, std::end(line));
	auto p = std::stoi(port);

	return std::make_unique<CComUDPClient>(ip, p, "");
}

std::vector<std::unique_ptr<CComUDPClient>> parse_file(std::string const& file_name)
{
	std::ifstream ip_file(file_name);
	std::vector<std::unique_ptr<CComUDPClient>> ret;

	std::string tmp;
	int idx = 1;
	while (std::getline(ip_file, tmp)) {
		try {
			ret.push_back(parse_line(tmp));
		} catch (std::exception const& e) {
			std::cerr << "Error while reading ip file on line " << idx << ": " << tmp
				  << "\nError: " << e.what();
		}
		++idx;
	}
	return ret;
}
