#ifndef CCOMGRIDUDPSERVER
#define CCOMGRIDUDPSERVER
#include <queue>
#include <string>
#include "CComWorker.h"
#include <net/if.h>
#include <sys/socket.h> /* for socket(), bind(), and connect() */
#include <netdb.h> /* for gethostbyname */
#include <arpa/inet.h>  /* for sockaddr_in and inet_ntoa() */
#include <netinet/in.h> /* for IP Socket data types */
#include <unistd.h>     /* for close() */

class CComGridUDPServer {

public:
	int debug;
	//RECV_DATA *data;
	//int nb_data;
	std::queue<std::string> *data;
	CComGridUDPServer(unsigned short port, std::queue<std::string> *_data, int dbg);
	static void * UDP_server_thread(void *parm);
	~CComGridUDPServer();
	void read_data_lock();
	void read_data_unlock();
	int register_worker();
	int get_ipaddress(std::string &ip);
	int create_exp_path(char *path, char *expname);
	int determine_worker_name(std::string &workername);
	int send(char *individual, CommWorker destination);
	void read_thread();
	
private:
  	std::string workername;
	std::string fullpath;
        CommWorker *myself;
	void run();
	int ServerSocket;
	pthread_t thread;
	int Socket;
	struct sockaddr_in ServAddr;
	int dbg;
	bool cancel;
};
#endif