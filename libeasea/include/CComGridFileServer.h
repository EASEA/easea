/*
 *    Copyright (C) 2013  Waldo Cancino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#ifndef CCOMGRIDFILESERVER
#define CCOMGRIDFILESERVER
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
#include <utility>
#include "gfal_api.h"


#define _MULTI_THREADED
#define MAXINDSIZE 50000 /*maximum size of an individual in number of characters*/


class CComGridFileServer{
  public:
	int debug;
	//int cancel;git 
	//RECV_DATA *data;
	//int nb_data;
	std::queue<std::string> *data;
	std::list<std::pair<std::string,std::string> > writedata;
	static void * file_read_thread(void *parm);
	static void * file_write_thread(void *parm);
	static void * file_readwrite_thread(void *parm);
	CComGridFileServer(char *path, char *expname, std::queue<std::string> *_data, int dbg);
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
	~CComGridFileServer();
	void run_readwrite();

  private:
        
	pthread_t thread_read, thread_write;
	int cancel;
	long wait_time;
	int create_ind_repository();
	int determine_worker_name(int start=1);
	int determine_file_name(std::string tmpfilename, std::string workerdestname);
	int refresh_file_list();
	int file_read(const char* filename);
	void readfiles();
	void send_individuals();
	void run_read();
	void run_write();
        int create_tmp_file(int& fd, std::string workerdestname, std::string &tmpfilename);
	int send_file_worker(std::string buffer, std::string workerdestname);
	
};
#endif