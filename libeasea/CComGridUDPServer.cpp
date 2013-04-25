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
#include <queue>
extern "C"
{
#include "gfal_api.h"
#include <lcg_util.h>
}
#include "include/CComGridUdpServer.h"

#include <fcntl.h>
#include <time.h>

#define MAXINDSIZE 50000

pthread_mutex_t server_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t gfal_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t worker_list_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t sending_mutex = PTHREAD_MUTEX_INITIALIZER;


CComGridUDPServer::CComGridUDPServer(char* path, char* expname, std::queue< std::string >* _data, short unsigned int port, int dbg) {
  
    data = _data;
    std::string exp(expname);
    std::string pathname(path);
    fullpath = pathname + expname;

    cancel = false;
    debug = dbg;
    std::string hostname,ip;
    unsigned long int netmask;
    
    if(debug)printf("Creating experiment and worker folders...\n");
    // first create experiment path and worker directory
    if(create_exp_path(path,expname) == 0 && determine_worker_name(hostname) == 0)
    {
          // second, check worker connectivity, if external IP use sockert
	  // if no external ip, use files to receive data
	  
          if(get_ipaddress(ip,netmask) == 0 && this->init_network(port) == 0)
	  {  
	      this->myself = new CommWorker(hostname, ip, port);
	      this->myself->set_netmask(netmask);
	  }    
	  else this->myself = new CommWorker(hostname);
	  
	  // now register worker
	  if( this->register_worker() == 0 )
	  {
	      // create the
	      if(debug)printf("Worker Created --> hostname %s  ip %s  port %d\n", myself->get_name().c_str(), myself->get_ip().c_str(), myself->get_port());
	      refresh_workers = new CComWorkerListManager(fullpath, debug);
	      
	      
	      
	      // create threads
              if(  pthread_create(&refresh_t,NULL,&CComGridUDPServer::refresh_thread_f, (void *)this) < 0 ||
	       ( myself->get_ip() == "noip" ? 1 : pthread_create(&read_t,NULL,&CComGridUDPServer::read_thread_f, (void *)this)) < 0 ) 
	      {
		printf("pthread create failed. exiting\n"); 
		exit(1);
	      }
	      
              if(  pthread_create(&readf_t,NULL,&CComGridUDPServer::read_thread_files_f, (void *)this) < 0 ) 
	      {
		printf("pthread create failed. exiting\n"); 
		exit(1);
	      }
	      
              if(  pthread_create(&writef_t,NULL,&CComGridUDPServer::write_thread_files_f, (void *)this) < 0 ) 
	      {
		printf("pthread create failed. exiting\n"); 
		exit(1);
	      }


	  }  
	  
    }
    else
    {  
    // create the main directory
        printf("Cannot create experiment folder %s check user permissions or disk space", fullpath.c_str());
        printf("Error reported is = %d\n" , errno);   
	exit(1);
    }
}


int CComGridUDPServer::get_ipaddress(std::string &ip, unsigned long int &nm)
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
    strcpy(external_ip,"no_ip");
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
		ip = buf;
		nm = ntohl(((struct sockaddr_in *)ifa->ifa_netmask)->sin_addr.s_addr);
		return 0;
	    }
	    // get an ip adress internal
	    else if(strlen(buf)> maxlen)
	    {  
	      ip = buf;
	      maxlen=strlen(buf);
	      //freeifaddrs(myaddrs);
	      //return 0;
	    }   
	  }
	}
  }
  if(myaddrs!=NULL)freeifaddrs(myaddrs);
  return -1;
}  
  



int CComGridUDPServer::create_exp_path(char *path, char *expname)
{
    std::string exp(expname);
    std::string pathname(path);
    fullpath = pathname + expname;
    
    //cancel = 0;
 
    // create the main directory
    gfal_set_verbose(2);
    int result = gfal_mkdir(fullpath.c_str(),0777);
    
    // check error condition
    printf("Trying to determine or create directory experiment\n");
    if(result<0 && errno!=EEXIST)
    {
        printf("Cannot create experiment folder %s; check user permissions or disk space", fullpath.c_str());
        printf("Result of gfal_mkdir = %d %d\n" ,result,errno);   
	return -1;
    }
    printf("Experimenet folder is %s\n",fullpath.c_str());
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
      if(start > 0)
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



int CComGridUDPServer::init_network(short unsigned int &port) {
    struct sockaddr_in ServAddr; /* Local address */

    #ifdef WIN32
    WSADATA wsadata;
    if (WSAStartup(MAKEWORD(1,1), &wsadata) == SOCKET_ERROR) {
	    printf("Error creating socket.");
	    return -1;
    }
    #endif

        /* Create socket for incoming connections */
    if ((ServerSocket =  socket(AF_INET,SOCK_DGRAM,0)) < 0) {
		printf("%d\n",socket(AF_INET,SOCK_DGRAM,0));
        printf("Socket create problem.\n"); exit(1);
    }

        /* Construct local address structure */
    
    int tries = 0;
    
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
	    continue;
	}
	else return 0;
    }
    return -1;
}

// create and register worker
int CComGridUDPServer::register_worker()
{
    
    std::string fullfilename = fullpath + '/' + myself->get_name() + '/' + "worker_info.txt";
    
    int	fd = gfal_open( fullfilename.c_str(), O_WRONLY|O_CREAT, 0644 );
    if (fd != -1) 
    {
	 std::stringstream s;
	 s << myself->get_name() + ':';
	 if(myself->get_ip() == "noip")s << "FILE::";
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
    else
    {  
          printf("failed to create worker file %s error reported is %d\n", fullfilename.c_str(),errno);
	  std::string workerdir = fullpath + '/' + myself->get_name();
	  printf("Deleting worker directory %s\n", workerdir.c_str());
	  int result = gfal_rmdir(workerdir.c_str());
	  if(result==-1)printf("Cannot delete worker directory, error reported is %d\n", errno);
          exit(1);
          return -1;
    }  
}  


int CComGridUDPServer::send(char *individual, int dest)
{
     
     pthread_mutex_lock(&worker_list_mutex);
     CommWorker *workdest;
     // worker list can be changed since last refresh
     if(dest>= refresh_workers->get_nr_workers() ){
       
       if(refresh_workers->get_nr_workers()>0)
       {	 
          workdest = new CommWorker(refresh_workers->get_worker_nr(refresh_workers->get_nr_workers()-1));
          if(debug)
             printf("Invalid destination worker, send to another worker : %s\n", workdest->get_name().c_str());
       }	  
       else
       {	 
	     printf("No destination workers available \n");
	     pthread_mutex_unlock(&worker_list_mutex);
	     return 0;
       }     
     }     
     else  workdest = new CommWorker( refresh_workers->get_worker_nr( dest ) );     
     pthread_mutex_unlock(&worker_list_mutex);
     if(workdest->get_name() == myself->get_name())
     {
        if(debug)
	    printf("I will not send the individual to myself, it is not a fatal error, so continue\n");
	delete workdest;
	
        return 0;
     }
     
     if(workdest->get_ip() == "noip" 
       || !in_same_network(myself->get_ip().c_str(),workdest->get_ip().c_str()))send_file(individual,*workdest);
     else send( individual, *workdest);
     delete workdest;
     return 0;
  
}



int CComGridUDPServer::send(char *individual, CommWorker destination) {
   int sendSocket;
   sockaddr_in destaddr;
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
		
	        destaddr.sin_family = AF_INET;
		destaddr.sin_addr.s_addr = inet_addr(destination.get_ip().c_str());
		destaddr.sin_port = htons(destination.get_port());
		int n_sent = sendto(sendSocket,individual,strlen(individual),0,(struct sockaddr *)&destaddr,sizeof(destaddr));
		
		//int n_sent = sendto(this->Socket,t,strlen(individual),0,(struct sockaddr *)&this->ServAddr,sizeof(this->ServAddr));
		if( n_sent < 0){
			printf("Size of the individual %d\n", (int)strlen(individual));
			perror("! Error while sending the message !");
		}
		else if(debug)
		    printf("Individual sent to hostname: %s  ip: %s  port: %d\n", destination.get_name().c_str(),
			   destination.get_ip().c_str(),destination.get_port());
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

void* CComGridUDPServer::read_thread_f( void *parm)
{
    CComGridUDPServer *server = (CComGridUDPServer *)parm;
    server->read_thread();
    return NULL;
}


void* CComGridUDPServer::refresh_thread_f( void *parm)
{
    CComGridUDPServer *server = (CComGridUDPServer *)parm;
    server->refresh_thread();
    return NULL;
}

void CComGridUDPServer::refresh_thread()
{
     int counter = 0;
     while(!cancel) {/*forever loop*/
	  
 	  if(counter%5==0)refresh_workers->refresh_worker_list();
	  //readfiles();
	  //send_individuals();
	  counter++;
		// check for new files
	  sleep(4);
      }
}  


void* CComGridUDPServer::read_thread_files_f( void *parm)
{
    CComGridUDPServer *server = (CComGridUDPServer *)parm;
    server->readfiles_thread();
    return NULL;
}

void* CComGridUDPServer::write_thread_files_f( void *parm)
{
    CComGridUDPServer *server = (CComGridUDPServer *)parm;
    server->writefiles_thread();
    return NULL;
}



void CComGridUDPServer::readfiles_thread()
{
     while(!cancel) {/*forever loop*/
	  readfiles();
	  sleep(4);
      }
}  

void CComGridUDPServer::writefiles_thread()
{
     while(!cancel) {/*forever loop*/
	  send_individuals();
	  sleep(4);
      }
}  


void CComGridUDPServer::read_data_lock() {
	pthread_mutex_lock(&server_mutex);
}

void CComGridUDPServer::read_data_unlock() {
	pthread_mutex_unlock(&server_mutex);

}

int CComGridUDPServer::number_of_clients()
{
      return refresh_workers->get_nr_workers();
}

CComGridUDPServer::~CComGridUDPServer()	
{
    printf("Calling the destructor ....\n");
    this->cancel =1;
    this->refresh_workers->terminate();
    if( myself->get_ip() != "noip" )
    {  
      pthread_cancel(read_t);
      pthread_join(read_t, NULL);
    }  
    pthread_join(refresh_t,NULL);
    pthread_join(readf_t,NULL);
    pthread_join(writef_t,NULL);
    // erase working path
    printf("Filserver thread cancelled ....\n");
    int tries=0;
    std::string workerpath = fullpath + '/' + myself->get_name();
    while(tries < 10)
    {
	DIR *dp;
	struct dirent *ep;
	
	
	dp = gfal_opendir (workerpath.c_str());
	if (dp != NULL)
	{
	    int countfiles=0;
	    while ((ep = gfal_readdir (dp)))
	    {
	      //only take into account folders
	      std::string s(ep->d_name);
	      std::string fullfilename = workerpath + '/' + s;

	      struct stat statusfile;
	      int result = gfal_stat(fullfilename.c_str(), &statusfile);

	      if( result != -1 && S_ISREG(statusfile.st_mode))
	      {  
		    if( lcg_del((char *)fullfilename.c_str(), 5, NULL,NULL,NULL,0,0) != 0)
		    {
		      if(debug)printf("Finish worker : Cannot erase  the file %s\n", fullfilename.c_str());
		      break;
		    }  
	      }      
	      else if(result == -1)
	      {
			    (void)gfal_closedir (dp);
			  	sleep(4);
				tries++;
				continue;
	      }		
	    }
	    (void)gfal_closedir (dp);
            if( gfal_rmdir(workerpath.c_str()) == 0 )
	    {
		if(debug)
		    printf("Worker removed sucessfully, removing the path %s\n", workerpath.c_str());
	        break;
	    }	
	    else if(debug)
		      printf("Worker pâth %s be removed sucessfully, trying again\n", workerpath.c_str());
              		 
	}
	sleep(4);
        tries++;
    }
    if(tries == 10) printf("Cannot remove the worker path %s, worker not properly finished\n", workerpath.c_str());
    delete myself;
    delete this->refresh_workers;
}    


int CComGridUDPServer::determine_file_name(std::string fulltmpfilename, std::string workerdestname)
{
	
   int tries = 0;
   time_t tstamp = time(NULL);
    while(tries<3)
    {
 	std::stringstream s;
	s << fullpath << '/' << workerdestname << "/individual_" << tstamp << ".txt";
	std::string fullfilename = s.str();
        int fd = gfal_open( fullfilename.c_str(), O_RDONLY, 0777);
	// the file does not exit, so we can create the individual
	if(fd ==-1 && errno == ENOENT)
	{
	   int result = gfal_rename(fulltmpfilename.c_str(), fullfilename.c_str());
	   
	    if(result==0)
	    {
	      result=gfal_chmod(fullfilename.c_str(), 0777);
	      if(debug)
		printf("Individual file %s created sucessfully\n", fullfilename.c_str());
	      return 0;
	    }
	    else
	    {
	        tries++;
	        printf("unable to rename tmp file to %s, error code %d,  trying another name\n", fullfilename.c_str(), errno);
		DIR *dp;
		std::string workerpath = fullpath + '/' + workerdestname;
		dp = gfal_opendir (workerpath.c_str());
		if( dp == NULL)
		{  
		  printf("The worker path is not accesible %s, maybe worker finishes\n", workerpath.c_str());
		  return -1;
		}
		(void)gfal_closedir(dp);

	    }
	}
        tstamp++;
    }	
    return -1;
}  


int CComGridUDPServer::create_tmp_file(int &fd, std::string workerdestname, std::string &fullfilename)
{
   time_t tstamp = time(NULL);
   unsigned tries = 0;
    while(tries<3)
    {
        
	std::stringstream s;
	s << fullpath << '/' << workerdestname << "/temp_" << tstamp << ".txt";
	fullfilename = s.str();
	
	
	 
	/* initialize file_name and new_file_mode */
	
	
	// try to open the file in exclusive mode
	fd = gfal_open( fullfilename.c_str(), O_CREAT | O_WRONLY, 0777);
	if (fd != -1) {
	        //associate with file
	        
	 	   if(debug)
		     printf("Create file for sending individual %s \n :", fullfilename.c_str());
		   return 0;
	}
	else
	{
	    // error ocurred, so two alternatives
	    // the path does not exit anymore (race condition)
	  tries++;
	  printf("Failed to create filename %s, error code %d\n", fullfilename.c_str(), errno);
	  /*DIR *dp;
	  std::string workerpath = fullpath + '/' + workerdestname;
	  dp = gfal_opendir (workerpath.c_str());
	  if( dp == NULL)
	  {  
	     printf("The worker path is not accesible %s, maybe worker finishes\n", workerpath.c_str());
	     break;
	  }
	  (void)gfal_closedir(dp);*/
	  tstamp++;
	}  
    }
    return -1;
}


void CComGridUDPServer::send_individuals()
{
    //std::list<std::pair<std::string,std::string> >::iterator it = writedata.begin();
    //std::list<std::pair<std::string,std::string> >::iterator it_e = writedata.end();
    //std::list<std::pair<std::string,std::string> >::iterator it2 = it;
    if(writedata.size()==0)return;
    pthread_mutex_lock(&sending_mutex);
    std::pair<std::string,std::string> item = writedata.back();
    writedata.pop_back();
    pthread_mutex_unlock(&sending_mutex);


    /*while( it != it_e && !cancel )
    //{
        if(send_file_worker( (*it).first, (*it).second ) ==0 )
	{  
	     pthread_mutex_lock(&sending_mutex);
     	     it2 = it;
	     ++it;
	     writedata.erase(it2);
	     pthread_mutex_unlock(&sending_mutex);
	} 
	else ++it;
      
    } */
    if(cancel)printf("Stop sending individuals, thread canceled, remaining to send %d\n",writedata.size());
    send_file_worker( item.first, item.second );
}  


int CComGridUDPServer::send_file_worker(std::string buffer, std::string workerdestname)
{
    int fd;
    std::string tmpfilename;
    pthread_mutex_lock(&gfal_mutex);
    if( create_tmp_file(fd, workerdestname, tmpfilename) == 0)
     {
	  int result = gfal_write( fd, buffer.c_str(), buffer.size() );
	  if(result >0)
	  {
		printf("gfal_write returns %d for written %d bytes \n", result, buffer.size() );
		(void)gfal_close(fd);
		if( determine_file_name(tmpfilename, workerdestname) == 0)
		{
		  pthread_mutex_unlock(&gfal_mutex);
		  return 0;
		}  
		else
		{
		  pthread_mutex_unlock(&gfal_mutex);
		  return -1;
		}  
	  }
	  else
	  {
	    (void)gfal_close(fd);
	    pthread_mutex_unlock(&gfal_mutex);
	    return -1;
	    
	  }  
	  // now rename 
	  
     }
     else
     {
        printf("Cannot write individual to file in path %s/%s", fullpath.c_str(), workerdestname.c_str() ); 
	pthread_mutex_unlock(&gfal_mutex);
	return -1;
     }
     pthread_mutex_unlock(&gfal_mutex);
     return 0;
}
  
  

int CComGridUDPServer::send_file(char *buffer, CommWorker destination)
{
     //first thing, prevent send to myself

     std::string buffer_str(buffer);
     std::pair<std::string,std::string> item(buffer_str,destination.get_name());

     pthread_mutex_lock(&sending_mutex);
     if(debug)
       printf("Individual to be quicked to be sent to %s by file\n",destination.get_name().c_str());
     writedata.push_back(item);
     pthread_mutex_unlock(&sending_mutex);
     return 0;
}     
     
     
     
     
     // determine the name to send a file
     
 


int CComGridUDPServer::file_read(const char *filename, char *buffer)
{
    std::string workerfile(filename);
    std::string fullfilename = fullpath + '/' + myself->get_name() + '/' + workerfile;
    int fd = gfal_open( fullfilename.c_str(), O_RDONLY, 0777);;
    
    
    if( fd != -1)
    {
        // get individual
        int result = gfal_read(fd, buffer, MAXINDSIZE );
        if( result < 0)
	{  
	  (void)gfal_close(fd);
	  return -1;
	}    
	
	(void)gfal_close(fd);
	// fail some read operation
    }
     else 
	return -1;
     processed_files.insert(workerfile);
	
    return 0;
}


void CComGridUDPServer::readfiles()
{
      char buffer[MAXINDSIZE];
      if(refresh_file_list() == 0)
      {
	  std::list<std::string>::iterator it;
	  for(it= new_files.begin(); it != new_files.end() && !cancel; it++)
	  {
	    pthread_mutex_lock(&gfal_mutex);
	    if(file_read((*it).c_str(),buffer) == 0)
	    {
		if(debug) {
		    printf("Reading file %s sucescully\n", (*it).c_str());
		    printf("\nData entry[%i]\n",data->size());
		    printf("Received the following:\n");
		    printf("%s\n",buffer);
		    printf("%d\n",(int)strlen(buffer));
		}
		  
	      // blocking call
	      pthread_mutex_lock(&server_mutex);
	      std::string bufferstream(buffer);
	      data->push(buffer);
	      /*process received data */
	      //memmove(data[nb_data].data,buffer,sizeof(char)*MAXINDSIZE);
	      //nb_data++;
      //	printf("address %p\n",(p->data));
	      //data = (RECV_DATA*)realloc(data,sizeof(RECV_DATA)*(nb_data+1));
      //	printf("address %p\n",(p->data));
	      pthread_mutex_unlock(&server_mutex);
	      /*reset receiving buffer*/
	      memset(buffer,0,MAXINDSIZE);
	      pthread_mutex_unlock(&gfal_mutex);
	    }
	    else
	    {
		printf("Error reading file %s , we will ignore it \n", (*it).c_str()); 
	    }	
	  }
	  if(cancel) printf("Stop reading individuals, thread canceled\n");
      }
      else
      {
	    printf("Cannot get the list of files for this worker, we will try later again ...\n");
      }  
}  

int CComGridUDPServer::refresh_file_list()
{
  // clear the file to process
  new_files.clear();
  
  // taken from GNU C manual
  
  DIR *dp;
  struct dirent *ep;
  
  std::string workerpath = fullpath + '/' + myself->get_name();
  std::set<std::string>::iterator it;   

  //std::string command = "ls -a "+workerpath;
  //system(command.c_str());


  dp = gfal_opendir (workerpath.c_str());
  if (dp != NULL)
  {
       printf("Refreshing filelist (new individuals) in %s\n",workerpath.c_str());
       while ((ep = gfal_readdir (dp)) && !cancel)
       {
		 pthread_mutex_lock(&gfal_mutex);
		 //only take into account folders
		 std::string s(ep->d_name);
		 std::string fullfilename = workerpath + '/' + s;
		 struct stat statusfile;
		 int status = gfal_stat(fullfilename.c_str(), &statusfile);
		 
		 if( status!= -1 && S_ISREG(statusfile.st_mode) )
		 {  
			  
			  if(s.substr(0,10) == "individual")
			  {
				it = processed_files.find(s);
				// new file to be processed
				if(it == processed_files.end() )
				{
				  if(debug)printf("New file found in path %s : %s\n", workerpath.c_str(), s.c_str());
				  new_files.push_back(s);
				}
			  }
		 }
		 else if(status==-1) 
		 {
		   
			(void)gfal_closedir (dp);
			pthread_mutex_unlock(&gfal_mutex);
		    return 0;
		 }   
		  pthread_mutex_unlock(&gfal_mutex);
	        
      }
      if(cancel) printf("Stop scanning incoming files, thread canceled\n");
      (void)gfal_closedir (dp);
  }      
  else
  {  
       printf ("Cannot scan the experiment directory for find workers %s \n", workerpath.c_str() );
       return -1;
  }      
  return 0;
}


int CComGridUDPServer::in_same_network(const char *addr1, const char *addr2)
{
      struct sockaddr_in ip1,ip2;
      inet_aton(addr1, &ip1.sin_addr);
      inet_aton(addr2, &ip2.sin_addr);
      unsigned long int n_ip1, n_ip2;
      n_ip1 = ntohl(ip1.sin_addr.s_addr);
      n_ip2 = ntohl(ip2.sin_addr.s_addr);
      
      return ((n_ip1 & myself->get_netmask()) == (n_ip2 & myself->get_netmask()));
      
      //return ( (ip1.sin_addr.s_addr & netmask1.sin_addr.s_addr) == (ip2.sin_addr.s_addr & netmask2.sin_addr.s_addr) );
      
}




