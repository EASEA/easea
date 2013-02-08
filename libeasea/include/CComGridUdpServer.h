#ifndef CCOMGRIDUDPSERVER
#define CCOMGRIDUDPSERVER
#include <queue>
#include <string>
#include "CComWorker.h"
class CComGridUDPServer {

public:
	int debug;
	//RECV_DATA *data;
	//int nb_data;
	std::queue<std::string> *data;
	CComUDPServer(unsigned short port, std::queue<std::string> *_data, int dbg);
	static void * UDP_server_thread(void *parm);
	~CComUDPServer();
	void read_data_lock();
	void read_data_unlock();
	int register_worker();
	int get_ipaddress(std::string &ip);
	
private:
        std::string fullpath;
        CommWorker *myself;
	void run();
	int ServerSocket;
	pthread_t thread;
	int Socket;
	int dbg;
};
#endif