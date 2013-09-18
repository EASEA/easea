/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

*/

#include "include/networkcommworkercommunicator.h"
#include <stdio.h>
#include <sys/types.h>          /* See NOTES */
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <ifaddrs.h>

#define MAXINDSIZE 50000

extern pthread_mutex_t server_mutex;

int NetworkCommWorkerCommunicator::init()
{
  
    if(determine_ipaddress() == 0 )
    {  
	struct sockaddr_in ServAddr; /* Local address */

      
	    /* Create socket for incoming connections */
	if ((ServerSocket =  socket(AF_INET,SOCK_DGRAM,0)) < 0) {
		    printf("%d\n",socket(AF_INET,SOCK_DGRAM,0));
	    printf("Socket create problem.\n"); exit(1);
	}

	    /* Construct local address structure */
	
	int tries = 0;
	int port = myself->get_port();
	
	while(tries<5)
	{  
	    memset(&ServAddr, 0, sizeof(ServAddr));   /* Zero out structure */
	    ServAddr.sin_family = AF_INET;                /* Internet address family */
	    ServAddr.sin_addr.s_addr = htonl(INADDR_ANY); /* Any incoming interface */
	    ServAddr.sin_port = htons(port);              /* Local port */
	    
	    /* Bind to the local address */
	    if (bind(ServerSocket, (struct sockaddr *) &ServAddr, sizeof(ServAddr)) < 0) {
		printf("Can't bind to given port number. Trying a different one.\n"); 
		port++;
	    }
	    else
	    { 
	      myself->change_port(port);
	      return 0;
	    }  
	    tries++;
	}
    }
    return -1;
}

int NetworkCommWorkerCommunicator::receive()
{
	struct sockaddr_in cliaddr; /* Client address */
        socklen_t len = sizeof(cliaddr);
        char tmpbuffer[MAXINDSIZE];
        unsigned int recvMsgSize;
                /*receive UDP datagrams from client*/
	while(!cancel)
	{  
	    if ((recvMsgSize = recvfrom(ServerSocket,tmpbuffer,MAXINDSIZE,0,(struct sockaddr *)&cliaddr,&len)) < 0) {
		    printf("\nError recvfrom()\n");
		    return -1;
	    }
      
	    printf("    Received individual from %s:%d\n", inet_ntoa(cliaddr.sin_addr), ntohs(cliaddr.sin_port));
	    tmpbuffer[recvMsgSize] = 0;
	    std::string buffer(tmpbuffer);
	    pthread_mutex_lock(&server_mutex);
	    data->push(buffer);
	    pthread_mutex_unlock(&server_mutex);
	}
	return 0;
}

int NetworkCommWorkerCommunicator::send(char* individual, CommWorker& destination)
{
   int sendSocket;
   sockaddr_in destaddr;

   if ((sendSocket = socket(AF_INET,SOCK_DGRAM,0)) < 0) {
        printf("Socket create problem."); return -1;
    }
    
    
    int sendbuff=35000;

    setsockopt(sendSocket, SOL_SOCKET, SO_SNDBUF, &sendbuff, sizeof(sendbuff));

    //std::string complete_ind = myself->get_name() + "::" + individual;
    std::string complete_ind = individual;
	if( complete_ind.length() < (unsigned)sendbuff ) { 
		
	        destaddr.sin_family = AF_INET;
		destaddr.sin_addr.s_addr = inet_addr(destination.get_ip().c_str());
		destaddr.sin_port = htons(destination.get_port());
		int n_sent = sendto(sendSocket,complete_ind.c_str(),complete_ind.length(),0,(struct sockaddr *)&destaddr,sizeof(destaddr));
		
		//int n_sent = sendto(this->Socket,t,strlen(individual),0,(struct sockaddr *)&this->ServAddr,sizeof(this->ServAddr));
		if( n_sent < 0){
			printf("Size of the individual %d\n", (int)strlen(individual));
			perror("! Error while sending the message !");
			return -1;
		}
		else
		{  
		  if(debug)
		    printf("Individual sent to hostname: %s  ip: %s  port: %d\n", destination.get_name().c_str(),
			   destination.get_ip().c_str(),destination.get_port());
		}	   
	}
	else {fprintf(stderr,"Not sending individual with strlen(): %i, MAX msg size %i\n",(int)strlen(individual), sendbuff);}
	close(sendSocket);
	return 0;
}

int NetworkCommWorkerCommunicator::determine_ipaddress()
{
  
  struct ifaddrs *myaddrs, *ifa;
  int status;
  
  status = getifaddrs(&myaddrs);
  
  // read external ip obtained by running a script
  char external_ip[64];
  
  FILE *file_ip = fopen("external_ip.txt","r");
  
  if(file_ip!=NULL)
  {
    //read the data
    fgets(external_ip,64,file_ip);
    external_ip[strlen(external_ip)-1] = 0;
    if(debug)printf("external ip is: %s we will now check network interfaces\n",external_ip);
    fclose(file_ip);
  }
  else
  {
    if(debug)printf("cannot open external ip file ...\n");
    strcpy(external_ip,"noip");
  }  
  
  
  
  

  if (status != 0){
    perror("No network interface, communication impossible, finishing ...");
    exit(1);
  }
  unsigned int maxlen = 0;
  for (ifa = myaddrs; ifa != NULL; ifa = ifa->ifa_next)
  {
      //if (NULL == ifa->ifa_addr)continue;
      //if ((ifa->ifa_flags & IFF_UP) == 0)continue;

      void *s4=NULL;                     
      /* ipv6 addresses have to fit in this buffer */
      char buf[64];                                  
      memset(buf,0,64);
      
      
      
      if (AF_INET == ifa->ifa_addr->sa_family)
      {
	  s4 = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
	  if (NULL == inet_ntop(AF_INET, s4, buf, sizeof(buf)))
	    printf("%s: inet_ntop failed!\n", ifa->ifa_name);
	  else {
	    // external ip address is in the network interface
	    printf("comparing %s (%d) and %s (%d) = %d\n", buf, 
		   strlen(buf),external_ip,strlen(external_ip), strcmp(buf,external_ip));
	    
	    if( strcmp(buf,external_ip) == 0) 
	    {  
	        myself->set_netmask( ntohl(((struct sockaddr_in *)ifa->ifa_netmask)->sin_addr.s_addr) );
		myself->set_ip( buf );
		return 0;
	    }
	    // get an ip adress internal
	    else if(strlen(buf)> maxlen)
	    {  
	      //myself.set_ip( buf );
	      maxlen=strlen(buf);
	      //freeifaddrs(myaddrs);
	      //return 0;
	    }   
	  }
	}
  }
  if(myaddrs!=NULL)freeifaddrs(myaddrs);
  myself->set_ip( external_ip );
  return -1;
}  
