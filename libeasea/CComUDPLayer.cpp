#include "include/CComUDPLayer.h"
#ifndef WIN32
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <net/if.h>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <error.h>
#include <errno.h>
#include <fcntl.h>
#else
#include <winsock.h>
#include <winsock2.h>
#include "include/inet_pton.h"
#include <windows.h>
#define socklen_t int
#endif
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#ifndef WIN32
#include <ifaddrs.h>
#endif
using namespace std;

pthread_mutex_t server_mutex = PTHREAD_MUTEX_INITIALIZER;
/* UDP SERVER*/
CComUDPServer::~CComUDPServer() {
	pthread_cancel(thread);
#ifndef WIN32
	close(this->ServerSocket);
#else
	closesocket(this->ServerSocket);
	WSACleanup();
#endif
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
                    printf("%d\n",(int)strlen(buffer));
		}
		printf("    Received individual from %s:%d\n", inet_ntoa(cliaddr.sin_addr), ntohs(cliaddr.sin_port));
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
}

CComUDPServer::CComUDPServer(unsigned short port, int dg) {
    struct sockaddr_in ServAddr; /* Local address */
    debug = dg;
	this->nb_data = 0;
	this->data = (RECV_DATA*)calloc(1,sizeof(RECV_DATA));

	#ifdef WIN32
	WSADATA wsadata;
	if (WSAStartup(MAKEWORD(1,1), &wsadata) == SOCKET_ERROR) {
		printf("Error creating socket.");
		exit(1);
	}
	#endif

        /* Create socket for incoming connections */
    if ((this->ServerSocket =  socket(AF_INET,SOCK_DGRAM,0)) < 0) {
		printf("%d\n",socket(AF_INET,SOCK_DGRAM,0));
        printf("Socket create problem.\n"); exit(1);
    }

    /*int bufsize = 50000;
    socklen_t optlen = sizeof(bufsize);
    setsockopt(this->ServerSocket, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(int));
    getsockopt(this->ServerSocket, SOL_SOCKET, SO_RCVBUF, &bufsize, &optlen);
    printf("buf size %d\n",bufsize);*/

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

CComUDPClient::CComUDPClient(struct sockaddr_in* addr, int dg){
    this->debug = dg;
    memcpy(&ServAddr, addr, sizeof(ServAddr));
}

void CComUDPClient::CComUDP_client_send(char *individual) {
	#ifdef WIN32
	WSADATA wsadata;
	if (WSAStartup(MAKEWORD(1,1), &wsadata) == SOCKET_ERROR) {
		printf("Error creating socket.");
 		exit(1);
	}
	#endif
    if ((this->Socket = socket(AF_INET,SOCK_DGRAM,0)) < 0) {
        printf("Socket create problem."); exit(1);
    }

    int sendbuff=35000;
#ifdef WIN32
    setsockopt(this->Socket, SOL_SOCKET, SO_SNDBUF, (char*)&sendbuff, sizeof(sendbuff));
#else
    setsockopt(this->Socket, SOL_SOCKET, SO_SNDBUF, &sendbuff, sizeof(sendbuff));
#endif

	if(strlen(individual) < (unsigned)sendbuff ) { 
		int n_sent = sendto(this->Socket,individual,strlen(individual),0,(struct sockaddr *)&this->ServAddr,sizeof(this->ServAddr));
		//int n_sent = sendto(this->Socket,t,strlen(individual),0,(struct sockaddr *)&this->ServAddr,sizeof(this->ServAddr));
        if( n_sent < 0){
                printf("Size of the individual %d\n", (int)strlen(individual));
                perror("! Error while sending the message !");
        }
	}
	else {fprintf(stderr,"Not sending individual with strlen(): %i, MAX msg size %i\n",(int)strlen(individual), sendbuff);}
#ifndef WIN32
	close(this->Socket);
#else
	closesocket(this->Socket);
 	WSACleanup();
#endif
};

std::string CComUDPClient::getIP(){
	return inet_ntoa(this->ServAddr.sin_addr);
}

int CComUDPClient::getPort(){
    return ntohs(this->ServAddr.sin_port);
}

/*UDP CLIENT*/
#ifndef WIN32
bool isLocalMachine(const char*  address, int clientPort, int serverPort){
    struct ifaddrs * ifAddrStruct=NULL; 
    struct ifaddrs * ifa=NULL;
    void * tmpAddrPtr=NULL;

    getifaddrs(&ifAddrStruct);
    for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa ->ifa_addr->sa_family==AF_INET) { // check it is IP4
            // is a valid IP4 Address
            tmpAddrPtr=&((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            char addressBuffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
            //printf("%s IP Address %s\n", ifa->ifa_name, addressBuffer); 

	        if((strcmp(address,addressBuffer)==0)
                && serverPort==clientPort)
		        return true;
        }    
    }
    
    if(ifAddrStruct!=NULL) freeifaddrs(ifAddrStruct);

    return false;
}
#endif

/*bool isLocalMachine(const char*  address, int clientPort, int serverPort){
	char hostname[128];
	struct in_addr ipv4addr;
	struct hostent *he;
    struct ifaddrs * ifAddrStruct=NULL; 
        struct ifaddrs * ifa=NULL;
            void * tmpAddrPtr=NULL;
	//int size;
	
	//retrieve local host name
	gethostname(hostname, sizeof(hostname));
	///printf("Local host name %s\n",hostname);

	//retrieve ip's host name

	//inet_pton(AF_INET, address, &ipv4addr);
    inet_aton(address, &ipv4addr);
#ifdef WIN32
	unsigned long ip = inet_addr(address);
	//printf("IP : %d\n",ip);
	//printf("Adresse : %s\n",address);
	he = gethostbyaddr((const char*)&ip, 4, 0);
	if(he == NULL){
		printf("*** WARNING ***\nCouldn't find host, are you sure the host at %s exists or the local machine is connected to a network ?\n", address);
		return false;
	}
#else
	//he = gethostbyaddr((const char*)address, sizeof address, AF_INET);
	he = gethostbyaddr((void *)&ipv4addr, sizeof ipv4addr, AF_INET);
	if(he == NULL){
       // herror("PROB\n");
		//printf("**** WARNING ***\nCouldn't find host, are you sure the host at %s exists or the local machine is connected to a network ?\n", address);
		return false;
	}
#endif
	//printf("Host name:%s\n",he->h_name);
    //printf("Client Port:%d\n",clientPort);
    //printf("Server Port:%d\n",serverPort);
	
	if((strcmp(hostname,he->h_name)==0
               || strcmp(he->h_name,"localhost")==0)
            && serverPort==clientPort)
		return true;
	else
		return false;
}*/

/**
 * Check the validity of an IP line. This line should have the form like : 1.2.3.4:5 (ip:port).
 *
 * @ARG line : the line containing the ip and port description.
 * @ @RETURN : boolean containing the result of the regex match.
 *
 */
//www.dreamincode.net/forums/topic/168930-valid-or-not-for-a-ip-address/
bool checkValidLine(string line){
    const char* ligne = line.c_str();
    char* holder = (char*)malloc(strlen(ligne)+1);
    strcpy(holder,ligne);
    char* address = strtok(holder, ":");
    char* port = strtok(NULL,":");

    //printf("IP %s\n",address);
    //printf("port %s\n",port);

    //Check if there is an IP and a port
    if(address==NULL || port==NULL){
        cout << "*** WARNING ***\nThere is a problem with the following IP: " << line << "\t===> IGNORING IT\n";
        return false;
    }

    //Check if it is a valid ip
    char* byte = strtok(address,".");
    int nibble = 0, octets = 0, flag = 0;
    while(byte != NULL){
        octets++;
        if(octets>4){
            cout << "*** WARNING ***\nThere is a problem with the following IP: " << line << "\t===> IGNORING IT\n";
            return false;
        }
        nibble = atoi(byte);
        if((nibble<0)||(nibble>255)){
            cout << "*** WARNING ***\nThere is a problem with the following IP: " << line << "\t===> IGNORING IT\n";
            return false;
        }
        string s = byte;
        for(unsigned int i=0; i<s.length(); i++){
            if(!isdigit(s[i])){
                cout << "*** WARNING ***\nThere is a problem with the following IP: " << line << "\t===> IGNORING IT\n";
                return false;
            }
        }
        byte = strtok(NULL,".");
    }
    if(flag || octets<4){
            cout << "*** WARNING ***\nThere is a problem with the following IP: " << line << "\t===> IGNORING IT\n";
            return false;
    }

    //Check if it is a valid port
    nibble = atoi(port);
    if(nibble<0){
        cout << "*** WARNING ***\nThere is a problem with the following IP: " << line << "\t===> IGNORING IT\n";
        return false;
    }
    string s = port;
    for(unsigned int i=0; i<s.length(); i++){
        if(!isdigit(s[i])){
                cout << "*** WARNING ***\nThere is a problem with the following IP: " << line << "\t===> IGNORING IT\n";
                return false;
        }
    }

    free(holder);
    return true;
}

/**
 * Parse an IP line. This line should have the form like : 1.2.3.4:5 (ip:port).
 *
 * @ARG line : the line containing the ip and port description.
 * @ @RETURN : a sockaddr_in structure, containing the corresponding ip.
 *
 * @TODO : This function should support the use of naming service instead of just ip adress.
 *         Is was the case in older version of EASEA.
 */
struct sockaddr_in parse_addr_string(const char* line){
    char tmp_line[512];
    char* res_field_ptr;
    strncpy(tmp_line,line,512);
    struct sockaddr_in addr;
    unsigned short port;
                  
    res_field_ptr = strtok(tmp_line,":");
    //printf("addr : %s\n",res_field_ptr);
    addr.sin_family = AF_INET;            /* Internet address family */
    addr.sin_addr.s_addr = inet_addr(res_field_ptr);     /* Any incoming interface */
                        
    char* endptr;
    res_field_ptr = strtok(NULL,":");
    port = strtol(res_field_ptr,&endptr,10);
    addr.sin_port = htons(port);          /* Local port */
    //printf("addr : %s\n",res_field_ptr);

    return addr;
}

/**
 * Load an "ip" file and create a sockaddr_in per line.
 *
 * @ARG file : a char* containing a patch to the ip file.
 * @ARG no_client : the number of ip loaded from the ip file.
 *
 * @RETURN : an array of *(p_no_client) CComUDPClient related to each ip file line.
 *
 * @TODO : ip file line shouldn't do more than 512 char. no line shouldn't be more than 128.
 */
CComUDPClient** parse_file(const char* file_name, unsigned* p_no_client, int portServer){
    //char* tmp_line = new char[512];
    string tmp_line;
    //FILE* ip_file = fopen(file_name,"r");
    ifstream ip_file(file_name);
    //size_t n = 512;
    unsigned no_client = 0;
    struct sockaddr_in* client_addr = new struct sockaddr_in[128];

    //while(getline(&tmp_line,&n,ip_file)>0){
    while(getline(ip_file, tmp_line)){
        if(checkValidLine(tmp_line)){
            struct sockaddr_in tmpClient = parse_addr_string(tmp_line.c_str());
#ifndef WIN32
            if(!isLocalMachine(inet_ntoa(tmpClient.sin_addr), ntohs(tmpClient.sin_port), portServer)){
                client_addr[no_client++] = tmpClient;
            }
#else
                client_addr[no_client++] = tmpClient;
#endif
        }
    }
                  
    // copy the client_addr array in a fitted array. 
    CComUDPClient** clients = new CComUDPClient*[no_client];
    for( unsigned i=0 ; i<no_client ; i++ ){
        clients[i] = new CComUDPClient(&client_addr[i],0);
    }
                      
    (*p_no_client) = no_client;
    //delete[] tmp_line;
    delete[] client_addr;
    return clients;
}





int CComFileServer::refresh_worker_list()
{
  // read the directory list, each folder is a worker
  worker_list.clear();
  
  // taken from GNU C manual
  
  DIR *dp;
  struct dirent *ep;
     
  dp = opendir (fullpath.c_str());
  if (dp != NULL)
  {
       while ((ep = readdir (dp)))
       {
	 //only take into account folders
	 if(ep->d_type == DT_DIR)
	 {  
	      string s(ep->d_name);
              worker_list.push_back(s);
	      if(debug)
		printf("Worker %s added to the list\n",s.c_str());
	 }
       } 
       (void) closedir (dp);
  }      
  else
  {  
       printf ("Cannot scan the experiment directory for find workers");
       return -1;
  }      
  return 0;
}


int CComFileServer::refresh_file_list()
{
  // clear the file to process
  new_files.clear();
  
  // taken from GNU C manual
  
  DIR *dp;
  struct dirent *ep;
  
  std::string workerpath = fullpath + '/' + workername + '/';
  std::set<string>::iterator it;   
  dp = opendir (workerpath.c_str());
  if (dp != NULL)
  {
       while ((ep = readdir (dp)))
       {
	 //only take into account folders
	 if(ep->d_type == DT_REG)
	 {  
	      string s(ep->d_name);
	      it = processed_files.find(s);
	      // new file to be processed
	      if(it == processed_files.end() )
	      {
		if(debug)printf("New file found in path %s : %s\n", workerpath.c_str(), s.c_str());
		new_files.push_back(s);
	      }
	 }      
      }
      (void) closedir (dp);
  }      
  else
  {  
       printf ("Cannot scan the experiment directory for find workers");
       return -1;
  }      
  return 0;
}




int CComFileServer::determine_worker_name(int start)
{
  
  // scan experiment directory to find a suitable worker name
  while(1)
  {
      std::stringstream s;
      s << fullpath << "/worker_" << start;
  
      int result = mkdir( s.str().c_str(),0777);
    
    // check error condition
    
      if(result == 0)
      {
	  if(debug)printf("Experiment worker folder sucessfuly created, the path is %s\n", fullpath.c_str());
	  workername = s.str();
	  break;
      }	
      // worker already in use, increase worker number
      else if(result!= EEXIST)
	start++;
      
      else
      {
	  printf("Cannot create worker experiment folder; check user permissions or disk space");
	  return -1;
      }
  }
  return 0;
}

CComFileServer::CComFileServer(char *expname, char *path, int dg) {
  
    std::string exp(expname);
    std::string pathname(path);
    fullpath = pathname + expname + '/';
    
    
    debug = dg;
    this->nb_data = 0;
    this->data = (RECV_DATA*)calloc(1,sizeof(RECV_DATA));

    // create the main directory
    
    int result = mkdir(fullpath.c_str(),0777);
    
    // check error condition
    
    if(result!=0 && result!= EEXIST)
    {
        printf("Cannot create experiment folder; check user permissions or disk space");
	exit(1);
    }
    else if(debug)
    {
        
	printf("Experiment folder %s, the path is %s\n", (result==0 ? "created" : "exits already"), fullpath.c_str());
    }
    
    // now determine the worker name, that is, the directory where where
    // the server will "listen" to new files
    
    if(determine_worker_name() != 0)
    {
        printf("Cannot create experiment worker folder; check user permissions or disk space");
	exit(1);
    }
    
    // now create thread to listen for incoming files
    if(pthread_create(&thread, NULL, &CComFileServer::File_server_thread, (void *)this) != 0) {
        printf("pthread create failed. exiting\n"); exit(1);
    }
}


int CComFileServer::determine_file_name(FILE *&fp, int dest)
{
    while(1)
    {
        time_t tstamp = time(NULL);
	std::stringstream s;
	s << fullpath << '/' << worker_list[dest] << "/individual_" << tstamp << ".txt";
	string fullfilename = s.str();
	
	
	int fd;
 
	/* initialize file_name and new_file_mode */
	
	
	// try to open the file in exclusive mode
	fd = open( fullfilename.c_str(), O_CREAT | O_EXCL | O_WRONLY);
	if (fd != -1) {
	        //associate with file
	  	fp = fdopen(fd, "w");
		if (fp != NULL) {
		   if(debug)
		     printf("Create file for sending individual %s \n :", fullfilename.c_str());
		   break;
		}
		else return -1;
	}
	else
	{
	    if(debug)
	      printf("Failed to create filename %s failed, trying another name \n :", fullfilename.c_str());
	    continue;
	}  
    }
    return 0;
}

int CComFileServer::send_file(char *buffer, int dest)
{
     //first thing, prevent send to myself
     FILE *outputfile;
     
     if(workername == worker_list[dest])
     {
        if(debug)
	    printf("I will not send the individual to myself, it is not a fatal error, so continue\n");
        return 0;
     }
     
     // determine the name to send a file
     
     if( determine_file_name(outputfile, dest) )
     {
	  fputs(buffer,outputfile);
	  fclose(outputfile);
     }
     else
     {
        printf("Cannot write individual to file in path %s/%s", fullpath.c_str(), workername.c_str() ); 
     }
     return 0;
}


int CComFileServer::file_read(const char *filename)
{
    std::string workerfile(filename);
    std::string fullfilename = fullpath + '/' + workername + '/' + workerfile;
    ifstream inputfile(fullfilename.c_str());
    
    if(inputfile.is_open())
    {
        // get individual
	inputfile.getline(buffer,MAXINDSIZE);
	inputfile.close();
	// fail some read operation
	if(inputfile.fail())
	    return -1;
	processed_files.insert(workerfile);
	
    }
    else
    {
	return -1;
    }
    return 0;
}

void CComFileServer::run()
{
       for(;;) {/*forever loop*/
	  
		// check for new files
		if(refresh_file_list() == 0)
		{
		    std::list<string>::iterator it;
		    for(it= new_files.begin(); it != new_files.end(); it++)
		    {
		      if(file_read((*it).c_str()))
		      {
			  if(debug) {
			      printf("Reading file %s sucescully\n", (*it).c_str());
			      printf("\nData entry[%i]\n",nb_data);
			      printf("Received the following:\n");
			      printf("%s\n",buffer);
			      printf("%d\n",(int)strlen(buffer));
			  }
			    
			// blocking call
			pthread_mutex_lock(&server_mutex);
			/*process received data */
			memmove(data[nb_data].data,buffer,sizeof(char)*MAXINDSIZE);
			nb_data++;
		//	printf("address %p\n",(p->data));
			data = (RECV_DATA*)realloc(data,sizeof(RECV_DATA)*(nb_data+1));
		//	printf("address %p\n",(p->data));
			pthread_mutex_unlock(&server_mutex);
			/*reset receiving buffer*/
			memset(buffer,0,MAXINDSIZE);
		      }
		      else
		      {
			  printf("Error reading file %s , we will ignore it \n", (*it).c_str()); 
		      }	
		    }
		}
		else
		{
		     printf("Cannot get the list of files for this worker, we will try later again ...\n");
		}  
		// we should wit some time after 
		sleep(wait_time);
	}
}

void * CComFileServer::File_server_thread(void *parm) {
	CComFileServer *server = (CComFileServer*)parm;
	server->run();
	return NULL;
}	
		  
	  
 