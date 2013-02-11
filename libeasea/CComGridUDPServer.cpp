#include "include/CComGridUdpServer.h"
#include <sys/types.h>                                                                                           
#include <sys/socket.h>                                                                                          
#include <arpa/inet.h>                                                                                           
#include <net/if.h>                                                                                              

#include <errno.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <stdio.h>     
#include <stdlib.h>    
#include <string.h>    
#include <sstream>
#include "gfal_api.h"
#include <fcntl.h>

#define MAXINDSIZE 50000


CComGridUDPServer::CComGridUDPServer(char* path, char* expname, std::queue< std::string >* _data, int dbg) {
  
    data = _data;
    std::string exp(expname);
    std::string pathname(path);
    fullpath = pathname + expname;

    //cancel = 0;
    debug = dbg;
 
    // create the main directory
    
    int result = gfal_mkdir(fullpath.c_str(),0777);
    
    // check error condition
    printf("Trying to determine or create directory experiment\n");
    if(result<0 && errno!= EEXIST)
    {
        printf("Cannot create experiment folder %s; check user permissions or disk space", fullpath.c_str());
        printf("Result of gfal_mkdir = %d %d\n" ,result,errno);   
	exit(1);
    }
    
    result = gfal_chmod(fullpath.c_str(), 0777);
    
    // now determine the worker name, that is, the directory where where
    // the server will "listen" to new files
    
    
    
    if(determine_worker_name() != 0)
    {
        printf("Cannot create experiment worker folder; check user permissions or disk space");
	exit(1);
    }
    
    
    //gfal_pthr_init(Cglobals_get);
    // now create thread to listen for incoming files
     //if(pthread_create(&thread, NULL, &CComUDPServer::UDP_server_thread, (void *)this) != 0) {
     if( pthread_create(&thread_read,NULL,&CComGridFileServer::file_readwrite_thread, (void *)this) < 0) {
        printf("pthread create failed. exiting\n"); exit(1);
    }
    
/*    if( thread_read = Cthread_create(&CComGridFileServer::file_read_thread, (void *)this) < 0) {
        printf("pthread create failed. exiting\n"); exit(1);
    }
    if( thread_write = Cthread_create(&CComGridFileServer::file_write_thread, (void *)this) < 0) {
        printf("pthread create failed. exiting\n"); exit(1);
    }*/

}


int CComGridUDPServer::get_ipaddress(std::string &ip)
{
  
  struct ifaddrs *myaddrs, *ifa;
 int status;

  status = getifaddrs(&myaddrs);

  if (status != 0){
    perror("No network interface, communication impossible, finishing ...");
    exit(1);
  }

  for (ifa = myaddrs; ifa != NULL; ifa = ifa->ifa_next)
  {
      if (NULL == ifa->ifa_addr)continue;
      if ((ifa->ifa_flags & IFF_UP) == 0)continue;

      struct sockaddr_in *s4;                     
      /* ipv6 addresses have to fit in this buffer */
      char buf[64];                                  

      if (AF_INET == ifa->ifa_addr->sa_family)
      {
	  s4 = (struct sockaddr_in *)(ifa->ifa_addr);
	  if (NULL == inet_ntop(ifa->ifa_addr->sa_family, (void *)&(s4->sin_addr), buf, sizeof(buf)))
	    printf("%s: inet_ntop failed!\n", ifa->ifa_name);
	  else {
	    // external ip address
	    if(strlen(buf)>=13)
	    {  
	      ip = buf;
		freeifaddrs(myaddrs);
	      return 0;
	    }   
	  }
	}
  }
  freeifaddrs(myaddrs);
  return -1;
}  
  
  
int CComGridUDPServer::create_exp_path(char *path, char *expname)
{
    std::string exp(expname);
    std::string pathname(path);
    fullpath = pathname + expname;
    
    //cancel = 0;
    debug = dbg;
 
    // create the main directory
    
    int result = gfal_mkdir(fullpath.c_str(),0777);
    
    // check error condition
    printf("Trying to determine or create directory experiment\n");
    if(result<0 && errno!= EEXIST)
    {
        printf("Cannot create experiment folder %s; check user permissions or disk space", fullpath.c_str());
        printf("Result of gfal_mkdir = %d %d\n" ,result,errno);   
	exit(1);
    }
    
    result = gfal_chmod(fullpath.c_str(), 0777);
    return 0;
}

int CComGridUDPServer::determine_worker_name(std::string &workername)
{
  
  // scan experiment directory to find a suitable worker name
  int tries = 0;
  char hostname[256];
  
  gethostname(hostname, 256);
  
  workername = hostname;
  
  
  int start = 0;  
  while(tries < 5)
  {
      std::stringstream s,t;
      if(tries > 0)
      {	
	s << fullpath << "/worker_" << workername << '_' << start;
        t << "worker_" << workername << '_' << start;
      }
      else
      {
	s << fullpath << "/worker_" << workername;
        t << "worker_" << workername;
      }	
      DIR *dp;
  
     

      dp = gfal_opendir( s.str().c_str() );
      // verify if directory exist
      if(dp != NULL)
      {
		(void)gfal_closedir(dp);  
		start++;
      }
      else if(errno == ENOENT)
      {
	    int result = gfal_mkdir(s.str().c_str(), 0777);

// check error condition

	      if(result == 0)
	      {
		      if(debug)printf("Experiment worker folder sucessfuly created, the path is %s\n", s.str().c_str());
		      result = gfal_chmod( s.str().c_str(), 0777 );
		      workername = t.str();
		      return 0;
	      }	
	      else
	      {
		      printf("Cannot create worker experiment folder %s; error reported is %d\n", s.str().c_str(), errno);
		      tries++;
	      }
      }  
      else
      {
	      tries++;
	      printf("Cannot create worker experiment folder %s; error reported is %d\n", s.str().c_str(), errno);
      }
  }
  
  return -1;
}

// create and register worker
int CComGridUDPServer::register_worker()
{
    
  
    std::string fullfilename = fullpath + '/' + myself->get_name() + '/' + "worker_info.txt";
    
    int	fd = gfal_open( fullfilename.c_str(), O_CREAT | O_WRONLY, 0777);
    if (fd != -1) 
    {
	 std::stringstream s;
	 s << myself->get_name() + ':';
	 if(myself->get_ip() == "NOIP")s << "FILE::";
	 else s << "SOCKET:" << myself->get_ip() << ':' << myself->get_port();
	 
	 // now write file
	 
	 int result = gfal_write( fd, s.str().c_str(), s.str().size() );
	 
	 if(result > 0)
	 {  
	     printf("Create working information file %s \n :", fullfilename.c_str());
	     (void)gfal_close(fd);
	     return 0;
	     
	 }
    }
    return -1;
}  





int CComGridUDPServer::send(char *individual, CommWorker destination) {
   int sendSocket;
	#ifdef WIN32
	WSADATA wsadata;
	if (WSAStartup(MAKEWORD(1,1), &wsadata) == SOCKET_ERROR) {
		printf("Error creating socket.");
 		exit(1);
	}
	#endif
    if ((sendSocket = socket(AF_INET,SOCK_DGRAM,0)) < 0) {
        printf("Socket create problem."); exit(1);
    }
    
    
    int sendbuff=35000;
#ifdef WIN32
    setsockopt(sendSocket, SOL_SOCKET, SO_SNDBUF, (char*)&sendbuff, sizeof(sendbuff));
#else
    setsockopt(sendSocket, SOL_SOCKET, SO_SNDBUF, &sendbuff, sizeof(sendbuff));
#endif

	if(strlen(individual) < (unsigned)sendbuff ) { 
		sockaddr_in destaddr;
	        destaddr.sin_family = AF_INET;
		destaddr.sin_addr.s_addr = inet_addr(destination.get_ip().c_str());
		destaddr.sin_port = htons(destination.get_port());
		int n_sent = sendto(sendSocket,individual,strlen(individual),0,(struct sockaddr *)&destaddr,sizeof(destaddr));
		
		//int n_sent = sendto(this->Socket,t,strlen(individual),0,(struct sockaddr *)&this->ServAddr,sizeof(this->ServAddr));
		if( n_sent < 0){
			printf("Size of the individual %d\n", (int)strlen(individual));
			perror("! Error while sending the message !");
		}
	}
	else {fprintf(stderr,"Not sending individual with strlen(): %i, MAX msg size %i\n",(int)strlen(individual), sendbuff);}
#ifndef WIN32
	close(sendSocket);
#else
	closesocket(sendSocket);
 	WSACleanup();
#endif
}

void CComGridUDPServer::read_thread()
{
        struct sockaddr_in cliaddr; /* Client address */
        socklen_t len = sizeof(cliaddr);
        char buffer[MAXINDSIZE];
        unsigned int recvMsgSize;
        while(!cancel) {/*forever loop*/
                /*receive UDP datagrams from client*/
                if ((recvMsgSize = recvfrom(ServerSocket,buffer,MAXINDSIZE,0,(struct sockaddr *)&cliaddr,&len)) < 0) {
                        printf("\nError recvfrom()\n"); exit(1);
                }
		if(debug) {
                	buffer[recvMsgSize] = 0;
			printf("\nData entry[%i]\n", data->size());
                	printf("Received the following:\n");
                	printf("%s\n",buffer);
			printf("%d\n",(int)strlen(buffer));
		}
		printf("    Received individual from %s:%d\n", inet_ntoa(cliaddr.sin_addr), ntohs(cliaddr.sin_port));
		pthread_mutex_lock(&server_mutex);
		/*process received data */
		//memmove(p->data[(*p->nb_data)].data,buffer,sizeof(char)*MAXINDSIZE);
		//(*p->nb_data)++;
	//	printf("address %p\n",(p->data));
		//p->data = (RECV_DATA*)realloc(p->data,sizeof(RECV_DATA)*((*p->nb_data)+1));
		buffer[recvMsgSize] = 0;
		std::string bufferstream(buffer);
		data->push(bufferstream);
	//	printf("address %p\n",(p->data));
		pthread_mutex_unlock(&server_mutex);
		/*reset receiving buffer*/
		memset(buffer,0,MAXINDSIZE);
     }
     if(debug)printf("Finishing thread... bye\n");
}
