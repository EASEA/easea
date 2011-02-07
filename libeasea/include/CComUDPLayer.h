/* C UDP Communication Layer using Server-Client Modeli
   for implementation of distributed evolutionary computing
	@author: Pascal Comte, June 2010
*/

#ifndef CCOMUDPLAYER_H_
#define CCOMUDPLAYER_H_

#include <sys/socket.h> /* for socket(), bind(), and connect() */
#include <netdb.h> /* for gethostbyname */
#include <arpa/inet.h>  /* for sockaddr_in and inet_ntoa() */
#include <netinet/in.h> /* for IP Socket data types */
#include <string.h>     /* for memset() */
#include <unistd.h>     /* for close() */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <typeinfo>
#include <float.h>
#include <limits.h>
#include <omp.h>
#include <string>

#define _MULTI_THREADED
#define MAXINDSIZE 50000 /*maximum size of an individual in number of characters*/

typedef struct {
        char data[MAXINDSIZE];
}RECV_DATA;

typedef struct {
	int Socket; /* Socket descriptor for server */
        struct sockaddr_in ServAddr;   /* Local address */
        int debug;
	RECV_DATA *data;
	int *nb_data;
}UDP_server_thread_parm_t;

class CComUDPServer {

public:
	int debug;
	RECV_DATA *data;
	int nb_data;
	UDP_server_thread_parm_t *parm;
	CComUDPServer(unsigned short port, int dg);
	static void * UDP_server_thread(void *parm);
	~CComUDPServer();
	void read_data_lock();
	void read_data_unlock();
private:
	int ServerSocket;
	pthread_t thread;
	int Socket;
};

class CComUDPClient {

public:
	int debug;
	void CComUDP_client_send(char *individual);
	CComUDPClient(unsigned short port, const char *ip,int dg);
	~CComUDPClient();
	std::string getIP();
private:
	struct sockaddr_in ServAddr;
	int Socket;
};

bool isLocalMachine(const char* address);

#endif /* CCOMUDPLAYER_H_ */
