#ifndef CCOMGRIDUDPSERVER
#define CCOMGRIDUDPSERVER
#include <queue>
#include <string>
#include <list>
#include <set>
#include "CComWorker.h"
#include <net/if.h>
#include <sys/socket.h> /* for socket(), bind(), and connect() */
#include <netdb.h> /* for gethostbyname */
#include <arpa/inet.h>  /* for sockaddr_in and inet_ntoa() */
#include <netinet/in.h> /* for IP Socket data types */
#include <unistd.h>     /* for close() */
#include "CComWorkerListManager.h"

class CComGridUDPServer {

public:
	int debug;
	//RECV_DATA *data;
	//int nb_data;
	std::queue<std::string> *data;
	CComGridUDPServer(char* path, char* expname, std::queue< std::string >* _data, short unsigned int port, int dbg);
	static void * UDP_server_thread(void *parm);
	~CComGridUDPServer();
	void read_data_lock();
	void read_data_unlock();
	int register_worker();
	int get_ipaddress(std::string &ip);
	int create_exp_path(char *path, char *expname);
	int determine_worker_name(std::string &workername);
	int send(char *individual, int dest);
	int send(char* individual, CommWorker dest);
	void read_thread();
	void refresh_thread();
	static void* read_thread_f(void *parm);
	static void* refresh_thread_f(void *parm);
	int init_network(short unsigned int& port);
	int number_of_clients();
        int create_tmp_file(int& fd, std::string workerdestname, std::string &tmpfilename);
	int send_file_worker(std::string buffer, std::string workerdestname);
	void send_individuals();
	int refresh_file_list();
	int file_read(const char* filename, char *buffer);
	void readfiles();
	int determine_file_name(std::string tmpfilename, std::string workerdestname);
	int send_file(char *buffer, CommWorker destination);

	
private:
  	std::string workername;
	std::string fullpath;
	std::list<std::pair<std::string,std::string> > writedata;
	std::set<std::string> processed_files;
	std::list<std::string> new_files;

	CComWorkerListManager *refresh_workers;
        CommWorker *myself;
	void run();
	int ServerSocket;
	pthread_t read_t,refresh_t;
	int Socket;
	struct sockaddr_in ServAddr;
	bool cancel;
};
#endif
