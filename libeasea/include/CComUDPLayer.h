/* C UDP Communication Layer using Server-Client Modeli
   for implementation of distributed evolutionary computing
	@author: Pascal Comte, June 2010
*/
/**
 * @file CComUDPLayer.h
 * @author SONIC BFO, Pascal Comte 
 * @version 1.0
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation; either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Affero General Public License for more details at
 * http://www.gnu.org/licenses/
**/  

#ifndef CCOMUDPLAYER_H_
#define CCOMUDPLAYER_H_


#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h> /* for socket(), bind(), and connect() */
#include <netdb.h>      /* for gethostbyname */
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
#ifndef __clang__
#include <omp.h>
#endif
#include <string>
#include <vector>

#define _MULTI_THREADED
#define MAXINDSIZE 50000 /*maximum size of an individual in number of characters*/


/**
*  \struct   recv_data 
*  \brief    Simple wrapper to array of MAXINDSIZE char(byte)
*            for receiving individual
*  \details  Typedef RECV_DATA
*  
**/
typedef struct recv_data {
        char data[MAXINDSIZE];
}RECV_DATA;


/**
*  \struct   UDP_server_thread_parm_t 
*  \brief    Struct to pass parameter to thread
*  \details  pthread only allow one void* to be passed as argument, hence this
*            struct
*  
**/
typedef struct {
	int Socket;                          /* Socket descriptor for server */
        struct sockaddr_in ServAddr;   /* Local address */
        int debug;
	std::vector<recv_data>* data;
	int *nb_data;
}UDP_server_thread_parm_t;


/**
* \class    CComUDPServer 
* \brief    The UDP server for receiving individual from other island
* \details  The polling is lauched in a thread. in the near future, we will be
*           using boost::asio for communication 
*  
**/
class CComUDPServer {

  public:
    /*Constructors/Destructors----------------------------------------------------*/ 
    /**
    * \brief    Constructor of CComUDPServer
    * \details  Create the local scoket for all incoming UDP connections
    *
    * @param  port The server port
    * @param  dg   Debug flag 
    **/
    CComUDPServer(unsigned short port, int dg);
    /**
    * \brief    Destructor pf CComUDPServer
    * \details  Close socket
    *
    **/
    ~CComUDPServer();
    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    The main server polling loop
    * \details  
    *
    *  @param  parm Parameter structure of type UDP_server_thread_parm_t
    **/
    static void * UDP_server_thread(void *parm);
    
    /**
    * \brief    Method for current thread to lock the mutex on the server's data
    * \details  Used by CEvolutionaryAlgorithm  when adding received individuals
    *
    **/
    void read_data_lock();
    /**
    * \brief    Method for current thread to unlock the mutex on the server's data
    * \details  Used by CEvolutionaryAlgorithm  when adding received individuals
    *
    **/
    void read_data_unlock();
 
  public:
    /*Datas-----------------------------------------------------------------------*/
    int debug;
    std::vector<recv_data> data;
    int nb_data;
    UDP_server_thread_parm_t *parm;
  private:
    /*Datas-----------------------------------------------------------------------*/
    int ServerSocket;
    pthread_t thread;
};

class CComUDPClient {

  public:
    /*Constructors/Destructors----------------------------------------------------*/ 
    
    CComUDPClient(unsigned short port, const char *ip,int dg);
    CComUDPClient(struct sockaddr_in* addr, int dg);
    ~CComUDPClient();
    /*Methods---------------------------------------------------------------------*/
    std::string getIP();
    int getPort();
    void CComUDP_client_send(char *individual);
  
  public: 
    /*Datas-----------------------------------------------------------------------*/
    int debug;
  private:
    /*Datas-----------------------------------------------------------------------*/
    struct sockaddr_in ServAddr;
    int Socket;
};

/**
* \brief    Check is the given ip address is the same as the machine
* \details
* @param    address   IP address as XXX.XXX.XXX.XXX
* @return   var       True if it is the same machine
**/
bool isLocalMachine(const char* address);

/**
* \brief    Check is the string represent a valid IPv4 address.
* \details  TODO:Fix memory leak found by clang-analyzer.Check the validity
*           of an IP line. This line should have the form like : 1.2.3.4:5 (ip:port).
*
* @param    line  The line containing the ip and port description.
* @return   var   Boolean containing the result of the regex match.
**/
bool checkValidLine(std::string line);

/**
* \brief    Parse an IP line. This line should have the form like : 1.2.3.4:5 (ip:port).
* \details  This function should support the use of naming service instead of just ip adress.
*           Is was the case in older version of EASEA.
*
* @param    line  The line containing the ip and port description.
* @return   var   A sockaddr_in structure, containing the corresponding ip.
**/
struct sockaddr_in parse_addr_string(const char* line);

/**
 * \brief   Load an "ip" file and create a sockaddr_in per line.
 * \details Ip file line shouldn't do more than 512 char. no line shouldn't be more than 128.
 *
 * @param   file        A char* containing a patch to the ip file.
 * @param   no_client   The number of ip loaded from the ip file.
 * @return  var         An array of *(p_no_client) CComUDPClient related to each ip file line.
 *
 */
CComUDPClient** parse_file(const char* file_name, unsigned* p_no_client, int serverPort);

#endif /* CCOMUDPLAYER_H_ */
