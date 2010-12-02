#include "include/CComUDPLayer.h"
#include <sys/ioctl.h>
#include <net/if.h>

pthread_mutex_t server_mutex = PTHREAD_MUTEX_INITIALIZER;
/* UDP SERVER*/
CComUDPServer::~CComUDPServer() {
	pthread_cancel(thread);
	close(this->ServerSocket);
};

void * CComUDPServer::UDP_server_thread(void *parm) {
	UDP_server_thread_parm_t *p = (UDP_server_thread_parm_t*)parm;
        struct sockaddr_in cliaddr; /* Client address */
        socklen_t len = sizeof(cliaddr);
        char buffer[MAXINDSIZE];
        unsigned int recvMsgSize;
        for(;;) {/*forever loop*/
                /*receive UDP datagrams from client*/
                if ((recvMsgSize = recvfrom(p->Socket,buffer,MAXINDSIZE,0,(struct sockaddr *)&cliaddr,&len)) < 0) {
                        printf("\nError recvfrom()\n"); exit(1);
                }
		if(p->debug) {
                	buffer[recvMsgSize] = 0;
			printf("\nData entry[%i]\n",*p->nb_data);
                	printf("Received the following:\n");
                	printf("%s\n",buffer);
		}
		printf("Received packet from %s:%d\n\n", inet_ntoa(cliaddr.sin_addr), ntohs(cliaddr.sin_port));
		pthread_mutex_lock(&server_mutex);
		/*process received data */
		memmove(p->data[(*p->nb_data)].data,buffer,sizeof(char)*MAXINDSIZE);
		(*p->nb_data)++;
	//	printf("address %p\n",(p->data));
		p->data = (RECV_DATA*)realloc(p->data,sizeof(RECV_DATA)*((*p->nb_data)+1));
	//	printf("address %p\n",(p->data));
		pthread_mutex_unlock(&server_mutex);
		/*reset receiving buffer*/
		memset(buffer,0,MAXINDSIZE);
        }
};

CComUDPServer::CComUDPServer(unsigned short port, int dg) {
        struct sockaddr_in ServAddr; /* Local address */
        debug = dg;
	this->nb_data = 0;
	this->data = (RECV_DATA*)calloc(1,sizeof(RECV_DATA));

        /* Create socket for incoming connections */
        if ((this->ServerSocket =  socket(AF_INET,SOCK_DGRAM,0)) < 0) {
                printf("Socket create problem.\n"); exit(1);
        }

        /* Construct local address structure */
        memset(&ServAddr, 0, sizeof(ServAddr));   /* Zero out structure */
        ServAddr.sin_family = AF_INET;                /* Internet address family */
        ServAddr.sin_addr.s_addr = htonl(INADDR_ANY); /* Any incoming interface */
        ServAddr.sin_port = htons(port);              /* Local port */

        /* Bind to the local address */
        if (bind(ServerSocket, (struct sockaddr *) &ServAddr, sizeof(ServAddr)) < 0) {
                printf("Can't bind to given port number. Try a different one.\n"); exit(1);
        }

        //UDP_server_thread_parm_t   *parm;
        this->parm = (UDP_server_thread_parm_t*)malloc(sizeof(UDP_server_thread_parm_t));
        this->parm->Socket = ServerSocket;
        this->parm->ServAddr = ServAddr;
	this->parm->nb_data = &this->nb_data;
	this->parm->data = this->data;
        this->parm->debug = this->debug;

        if(pthread_create(&thread, NULL, &CComUDPServer::UDP_server_thread, (void *)this->parm) != 0) {
                printf("pthread create failed. exiting\n"); exit(1);
        }
};

void CComUDPServer::read_data_lock() {
	pthread_mutex_lock(&server_mutex);
};

void CComUDPServer::read_data_unlock() {
	pthread_mutex_unlock(&server_mutex);
};
/*UDP SERVER*/

/*UDP CLIENT*/
CComUDPClient::~CComUDPClient() {};

CComUDPClient::CComUDPClient(unsigned short port, const char *ip,int dg){
	this->debug = dg;
        /* Construct local address structure */
        memset(&ServAddr, 0, sizeof(ServAddr));   /* Zero out structure */
        ServAddr.sin_family = AF_INET;            /* Internet address family */
        ServAddr.sin_addr.s_addr = inet_addr(ip);     /* Any incoming interface */
        ServAddr.sin_port = htons(port);          /* Local port */
};

void CComUDPClient::CComUDP_client_send(char *individual) {
        if ((this->Socket = socket(AF_INET,SOCK_DGRAM,0)) < 0) {
                printf("Socket create problem."); exit(1);
        }
	if(strlen(individual) < MAXINDSIZE ) { 
		//printf("Sending message...\n");
		sendto(this->Socket,individual,MAXINDSIZE,0,(struct sockaddr *)&this->ServAddr,sizeof(this->ServAddr));
	}
	else {fprintf(stderr,"Not sending individual with strlen(): %i\n",(int)strlen(individual));}
	close(this->Socket);
};

std::string CComUDPClient::getIP(){
	return inet_ntoa(this->ServAddr.sin_addr);
}
/*UDP CLIENT*/

bool isLocalMachine(const char*  address){
	char hostname[128];
	struct in_addr ipv4addr;
	struct hostent *he;
	int size;
	
	//retrieve local host name
	gethostname(hostname, sizeof(hostname));
	//printf("Local host name %s\n",hostname);

	//retrieve ip's host name
	inet_pton(AF_INET, address, &ipv4addr);
	he = gethostbyaddr(&ipv4addr, sizeof ipv4addr, AF_INET);
	//printf("Host name:%s\n",he->h_name);
	
	if(strlen(hostname)<strlen(he->h_name))
		size = strlen(hostname);
	else	
		size = strlen(he->h_name);

	if(strncmp(hostname,he->h_name,size)==0)
		return true;
	else
		return false;
	
}
