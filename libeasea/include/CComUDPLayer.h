/* C UDP Communication Layer using Server-Client Modeli
   for implementation of distributed evolutionary computing
	@author: Pascal Comte, June 2010
*/

#ifndef CCOMUDPLAYER_H_
#define CCOMUDPLAYER_H_

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h> /* for socket(), bind(), and connect() */
#include <netdb.h> /* for gethostbyname */
#include <arpa/inet.h>  /* for sockaddr_in and inet_ntoa() */
#include <netinet/in.h> /* for IP Socket data types */
#include <unistd.h>     /* for close() */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <typeinfo>
#include <float.h>
#include <limits.h>
#if not defined(__APPLE__) 
#include <omp.h>
#endif
#include <string>
#include <list>
#include <set>
#include <vector>
#include <queue>

#define _MULTI_THREADED
#define MAXINDSIZE 50000 /*maximum size of an individual in number of characters*/

typedef struct {
        char data[MAXINDSIZE];
}RECV_DATA;

typedef struct {
	int Socket; /* Socket descriptor for server */
        struct sockaddr_in ServAddr;   /* Local address */
        int debug;
	//RECV_DATA *data;
	//int *nb_data;
}UDP_server_thread_parm_t;

class CComUDPServer {

public:
	int debug;
	//RECV_DATA *data;
	//int nb_data;
	std::queue<std::string> &data;
	UDP_server_thread_parm_t *parm;
	CComUDPServer(unsigned short port, std::queue<std::string> &_data, int dbg);
	static void * UDP_server_thread(void *parm);
	~CComUDPServer();
	void read_data_lock();
	void read_data_unlock();
private:
	void run();
	int ServerSocket;
	pthread_t thread;
	int Socket;
};

class CComFileServer{
  public:
	int debug;
	//RECV_DATA *data;
	//int nb_data;
	std::queue<std::string> &data;
	static void * File_server_thread(void *parm);
	CComFileServer(char *path, char *expname, std::queue<std::string> &_data, int dbg);
	std::string workername;
	std::string fullpath;
	std::set<std::string> processed_files;
	std::vector<std::string> worker_list;
	std::list<std::string> new_files;
	char buffer[MAXINDSIZE];
	int number_of_workers();
	int send_file(char *buffer, int dest);
	int refresh_worker_list();
	void read_data_lock();
	void read_data_unlock();

  private:
	pthread_t thread;
	long wait_time;
	int create_ind_repository();
	int determine_worker_name(int start=1);
	int determine_file_name(FILE*& fp, int& fd, int dest);
	int refresh_file_list();
	int file_read(const char* filename);
	void run();
};


	

class CComUDPClient {

public:
	int debug;
	void CComUDP_client_send(char *individual);
	CComUDPClient(unsigned short port, const char *ip,int dg);
    CComUDPClient(struct sockaddr_in* addr, int dg);
	~CComUDPClient();
	std::string getIP();
    int getPort();
private:
	struct sockaddr_in ServAddr;
	int Socket;
};

bool isLocalMachine(const char* address);
bool checkValidLine(std::string line);
struct sockaddr_in parse_addr_string(const char* line);
CComUDPClient** parse_file(const char* file_name, unsigned* p_no_client, int serverPort);

#endif /* CCOMUDPLAYER_H_ */
