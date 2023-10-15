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

CComSharedContext::CComSharedContext()
	: ctx(std::make_shared<boost::asio::io_context>()), thread([this]() {
		  auto my_ptr = ctx; // avoid use after free
		  auto work = make_work_guard(my_ptr->get_executor());
		  my_ptr->run();
		  /*std::cerr << "io_service ended.\n";*/
	  })
{
	thread.detach();
}

CComUDPServer::CComUDPServer(unsigned short port, bool verbose_)
	: socket(CComSharedContext::get(), udp::endpoint(udp::v4(), port)), verbose(verbose_)
{
	//std::cerr << "Server started on port: " << port << "\n";
	start_receive();
}

bool CComUDPServer::has_data() const
{
	return recv_queue.size() > 0;
}

std::pair<std::vector<char>, boost::asio::ip::udp::endpoint> CComUDPServer::consume()
{
	if (!has_data())
		throw std::runtime_error("No data");
	auto ret = recv_queue.front();
	recv_queue.pop();
	//std::cout << "DBG: data was consumed! -- Queue size=" << recv_queue.size() << "\n";
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
	if (error) {
		std::cerr << "UDP error: " << error.message() << "\n";
	} else if (bytes_transfered != 0) {
		std::vector<char> tmp(recv_buffer.begin(), recv_buffer.begin() + bytes_transfered);
		recv_queue.emplace(std::move(tmp), last_endpoint);
	}
	start_receive();
}

CComUDPClient::CComUDPClient(std::string const& ip, unsigned short port, std::string client_name_, bool verbose_)
	: client_name(std::move(client_name_)), socket(CComSharedContext::get()), verbose(verbose_)
{
	udp::resolver resolver(CComSharedContext::get());
	dest = *resolver.resolve(udp::v4(), ip, "").begin(); // throw if error (safe)
	dest.port(port);
	socket.open(udp::v4());
}

std::string const& CComUDPClient::getClientName() const
{
	return client_name;
}

void CComUDPClient::send(std::string const& individual)
{
	if (verbose) {
		auto now = std::chrono::system_clock::now();
		auto in_time_t = std::chrono::system_clock::to_time_t(now);
		std::cout << "[" << std::put_time(std::localtime(&in_time_t), "%H:%M:%S")
			  << "] Sending my best individual to ";
		if (getClientName().size() > 0) {
			std::cout << getClientName();
		} else {
			std::cout << getIP();
		}
		std::cout << ":" << getPort() << " \n";
	}
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
		udp::endpoint ep = *resolver.resolve(udp::v4(), address, "").begin(); // throw if error (safe)
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

std::unique_ptr<CComUDPClient> parse_line(std::string const& line, bool verbose_clients)
{
	auto sep = std::find(std::begin(line), std::end(line), ':');
	if (sep == std::end(line))
		throw std::runtime_error("no ':' found");
	auto ip = std::string(std::begin(line), sep);
	auto port = std::string(sep + 1, std::end(line));
	auto p = std::stoi(port);

	return std::make_unique<CComUDPClient>(ip, p, ip, verbose_clients);
}

std::vector<std::unique_ptr<CComUDPClient>> parse_file(std::string const& file_name, int thisPort, bool verbose_clients)
{
	std::ifstream ip_file(file_name);
	std::vector<std::unique_ptr<CComUDPClient>> ret;

	std::string tmp;
	int idx = 1;
	while (std::getline(ip_file, tmp)) {
		try {
			auto ptr = parse_line(tmp, verbose_clients);
			if (!(ptr->getPort() == thisPort && isLocalMachine(ptr->getIP())))
				ret.push_back(std::move(ptr));
		} catch (std::exception const& e) {
			std::cerr << "Error while reading ip file on line " << idx << " (\"" << tmp
				  << "\"): " << e.what();
		}
		++idx;
	}
	return ret;
}
