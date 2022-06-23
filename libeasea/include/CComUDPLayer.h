/**
 * @file CComUDPLayer.h
 * @brief Portable UDP implementation using Boost Asio
 * @author Léo Chéneau
 * @version 1.0
 * @date 2022-06-08
 *
 * Fix all portability problems...
 *
 * Warning: done (too) quickly to replace old pthread + UNIX implementation
 */

#ifndef CCOMUDPLAYER_H_
#define CCOMUDPLAYER_H_

#include <boost/asio/ip/udp.hpp>

#include <vector>
#include <queue>
#include <memory>
#include <thread>

/**
 * @brief Global common io_context
 */
class CComSharedContext
{
    public:
	using context_t = boost::asio::io_context;
	/**
     	* @brief Get global shared context
     	*
     	* @return global io_context
     	*/
	static inline context_t& get()
	{
		static CComSharedContext gctx;
		return gctx.ctx;
	}

    private:
	CComSharedContext();
	boost::asio::io_context ctx;
	std::thread thread;
};

class CComUDPServer
{
    public:
	/**
	* @brief Construct local UDP server
	*
	* @param port Port to listen on
	*/
	CComUDPServer(unsigned short port);

	CComUDPServer(CComUDPServer const&) = delete;
	CComUDPServer(CComUDPServer&&) = delete;
	CComUDPServer& operator=(CComUDPServer const&) = delete;
	CComUDPServer& operator=(CComUDPServer&&) = delete;
	~CComUDPServer() = default;

	/**
	 * @brief Buffer type used
	 */
	using buffer_t = std::array<char, 65535>;

	/**
	 * @brief Check if server has received data since last time
	 *
	 * @return true if new data can be consumed, false otherwise
	 */
	bool has_data() const;
	/**
	 * @brief Consume available data
	 * @waring Check if there's data first...
	 *
	 * @return A buffer with the data
	 */
	std::vector<char> consume();

    private:
	boost::asio::ip::udp::socket socket; ///< Socket used by the server
	std::queue<std::vector<char>> recv_queue; ///< Queue used to store data received until consumption
	buffer_t recv_buffer; ///< Buffer used to receive
	boost::asio::ip::udp::endpoint last_endpoint; /// Last endpoint who sent data

	/**
	 * @brief Receive asynchronously
	 */
	void start_receive();
	/**
	 * @brief Handle a reception event
	 *
	 * @param error Error codfe if any
	 * @param bytes_transfered Number of bytes transferred
	 */
	void handle_receive(const boost::system::error_code& error, std::size_t bytes_transfered);

	/**
	 * @brief Print stats of received individual
	 */
	void print_stats() const;
};

class CComUDPClient
{
    public:
	/**
	* @brief Create a new UDP Client
	*
	* @param port Port the destination is listening on
	* @param destination Resolvable adress of destination
	* @param client_name Human-readable name of client
	*/
	CComUDPClient(std::string const& destination, unsigned short port, std::string client_name = "");

	CComUDPClient(CComUDPClient const&) = delete;
	CComUDPClient(CComUDPClient&&) = delete;
	CComUDPClient& operator=(CComUDPClient const&) = delete;
	CComUDPClient& operator=(CComUDPClient&&) = delete;
	~CComUDPClient() = default;

	/**
	 * @brief Send data to this client
	 *
	 * @param individual Individual to send to destination
	 */
	void send(std::string const& individual);
	/**
	 * @brief Get IP related to this client
	 *
	 * @return A reference to the IP
	 */
	std::string getIP() const;
	/**
	 * @brief Get Port this client is connected to
	 *
	 * @return Port the destination is listening on
	 */
	unsigned short getPort() const;
	/**
	 * @brief Get Human-readable name related to this client
	 *
	 * @return A string representing the name
	 */
	std::string const& getClientName() const;

    private:
	std::string client_name; ///< Human-readble name
	boost::asio::ip::udp::endpoint dest; ///< Destination endpoint
	boost::asio::ip::udp::socket socket; ///< Socket used to communicate with destination
};

/**
 * @brief Check if this adress is a local adress
 *
 * Correct format is IPv4:port
 *
 * @param address Adress to check
 *
 * @return true if the adress is local
 */
bool isLocalMachine(std::string const& address);
/**
 * @brief Check if the line in the config is valid
 *
 * @param line Line to check
 *
 * @return true if the line is valid
 */
bool checkValidLine(std::string const& line);
/**
 * @brief Parse a config file and create associated clients
 *
 * @param file_name Path to config file
 *
 * @return A list of client generated from the config
 */
std::vector<std::unique_ptr<CComUDPClient>> parse_file(std::string const& file_name);

#endif /* CCOMUDPLAYER_H_ */
