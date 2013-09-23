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
#include <time.h>
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
pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

int CComGridUDPServer::cancel = 0;

CComGridUDPServer::CComGridUDPServer(char* path, char* expname, std::queue< std::string >* _data, short unsigned int port, int wn, int dbg) {
  
    data = _data;
    std::string exp(expname);
    std::string pathname(path);
    fullpath = pathname + expname;

    debug = dbg;
    std::string worker_id;
    std::string hostname,ip;
    unsigned long int netmask;
    worker_number = wn;
      char hn[256];
    
      gethostname(hn, 256);

    hostname = hn;

    
    if(debug)printf("Creating experiment and worker folders...\n");
    // first create experiment path and worker directory
    if(create_exp_path(path,expname) == 0)
    {
	  myself = new CommWorker();
	  myself->change_port(port);
	  myself->set_hostname(hostname);
	  networkcomm = new NetworkCommWorkerCommunicator(myself, data, debug);
	  filecomm = new FileCommWorkerCommunicator(myself, data,fullpath,worker_number,debug);
      
          // second, check worker connectivity, if external IP use sockert
	  // if no external ip, use files to receive data
	  
	  if(filecomm->init() == 0) 
	  {  
		networkcomm->init();
	  
	  
	  // now register worker
	    if( myself->register_worker(fullpath) == 0 )
	    {
		// create the
		if(debug)printf("Worker Created --> folder %s hostname %s  ip %s  port %d\n", myself->get_name().c_str(), myself->get_hostname().c_str(), myself->get_ip().c_str(), myself->get_port());
		refresh_workers = new Worker_WorkerListManager(fullpath, debug);
		
		
		
		// create threads
		if(  pthread_create(&refresh_t,NULL,&CComGridUDPServer::refresh_thread, (void *)refresh_workers) < 0 ||
		( myself->get_ip() == "noip" ? 1 : pthread_create(&read_t,NULL,&CComGridUDPServer::read_thread_f, (void *)networkcomm)) < 0 ) 
		{
		  printf("pthread create failed. exiting\n"); 
		  exit(1);
		}
		
		if(  pthread_create(&readf_t,NULL,&CComGridUDPServer::read_thread_files_f, (void *)filecomm) < 0 ) 
		{
		  printf("pthread create failed. exiting\n"); 
		  exit(1);
		}
		
		if(  pthread_create(&writef_t,NULL,&CComGridUDPServer::write_thread_files_f, (void *)filecomm) < 0 ) 
		{
		  printf("pthread create failed	. exiting\n"); 
		  exit(1);
		}


	    }
	  }
	  else
	  {
	      exit(1);
	  }  
	  
    }
    else
    {  
    // create the main directory
        printf("Cannot create experiment folder %s check user permissions or disk space", fullpath.c_str());
        printf("Error reported is = %d\n" , errno);   
	exit(1);
    }
    
    // finally create the log file_ip
    logfile_input = fopen("connections_receive.txt","w");
    logfile_output = fopen("connections_send.txt","w");
}


  



int CComGridUDPServer::create_exp_path(char *path, char *expname)
{
    std::string exp(expname);
    std::string pathname(path);
    fullpath = pathname + expname;
    
    //cancel = 0;
 
    // create the main directory
    //gfal_set_verbose(2);
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


int CComGridUDPServer::send(char *individual, int dest)
{
     
     //pthread_mutex_lock(&worker_list_mutex);
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
     //pthread_mutex_unlock(&worker_list_mutex);
     if(workdest->get_name() == myself->get_name())
     {
        if(debug)
	    printf("I will not send the individual to myself, it is not a fatal error, so continue\n");
	delete workdest;
	
        return 0;
     }
     // compose individual
     
     
     if(workdest->get_ip() == "noip" 
       || !in_same_network(myself->get_ip().c_str(),workdest->get_ip().c_str()))filecomm->send(individual, *workdest);
       //send_file(individual,*workdest);
     else networkcomm->send( individual, *workdest);
     
     delete workdest;
     return 0;
  
}



void* CComGridUDPServer::read_thread_f( void *parm)
{
    NetworkCommWorkerCommunicator *server = (NetworkCommWorkerCommunicator  *)parm;
    while(!cancel)
	server->receive();
    return NULL;
}



void* CComGridUDPServer::refresh_thread( void *parm )
{
     int counter = 0;
     Worker_WorkerListManager *server = (Worker_WorkerListManager *)parm;
     while(!cancel)
     {  
	server->refresh_worker_list();
	sleep(30);
     }
     return NULL;
}  


void* CComGridUDPServer::read_thread_files_f( void *parm)
{
    FileCommWorkerCommunicator *server = (FileCommWorkerCommunicator *)parm;
     while(!cancel)
     {  
	server->receive();
	sleep(5);
     }
    return NULL;
}

void* CComGridUDPServer::write_thread_files_f( void *parm)
{
    FileCommWorkerCommunicator *server = (FileCommWorkerCommunicator *)parm;
    while(!cancel)
    {  
      server->send();
      sleep(5);
    }  
    return NULL;
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

void CComGridUDPServer::terminate( std::string bestindividual)
{
    std::cout << "Finishing threads ..." << std::endl;
    char *dummy_message="finish";
    cancel = 1;
    //AbstractWorkerListManager::terminate();
    //CommWorkerCommunicator::terminate();
    //this->refresh_workers->terminate();
    
    // trick to finish networking communicator, send a dummy message
    
    
    
    if( myself->get_ip() != "noip" )
    {  
      //pthread_cancel(read_t);
      networkcomm->send(dummy_message,*myself);
      pthread_join(read_t, NULL);
    }  
    pthread_join(refresh_t,NULL);
    pthread_join(readf_t,NULL);
    pthread_join(writef_t,NULL);
    // erase working path
    if(logfile_input!=NULL)fclose(logfile_input);
    if(logfile_output!=NULL)fclose(logfile_output);
    printf("Thread cancelled ....\n");
    
    // artificial worker to send the results
    CommWorker result("results","results");
    filecomm->send((char *)bestindividual.c_str(),result);
    if( filecomm->send() != 0)std::cout << "Warning ... final result not stored for this worker...";
}  

CComGridUDPServer::~CComGridUDPServer()	
{
    printf("Calling the destructor ....\n");
    
    int tries=0;
    std::string workerpath = fullpath + '/' + myself->get_name();
    
    
    if( myself->unregister_worker(fullpath) != 0 )
      if(debug)printf("Finish worker : Cannot unregister worker \n");
    
    if( GFAL_Utils::rm_dir(workerpath) != 0 )
	printf("Cannot remove the worker path %s, worker not properly finished\n", workerpath.c_str());
    delete refresh_workers;
    delete networkcomm;
    delete filecomm;
    delete myself;
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


int CComGridUDPServer::log_connection(std::string source, std::string destination, std::string buffer)
{
      
      time_t t;
      time(&t);
      pthread_mutex_lock(&log_mutex);
      if(logfile_input!=NULL && logfile_output!=NULL)
      {	
	if(source == this->myself->get_name() )
       	    fprintf(logfile_output,"%ld,%s,%s,%s\n", t,source.c_str(), destination.c_str(), extract_fitness(buffer).c_str());
	else
	    fprintf(logfile_input,"%ld,%s,%s,%s\n", t,source.c_str(), destination.c_str(), extract_fitness(buffer).c_str());
      }
      pthread_mutex_unlock(&log_mutex);
      return 0;
}


std::string CComGridUDPServer::extract_fitness( std::string buffer )
{
    int pos =  buffer.rfind(" ");
    std::string fitness = buffer.substr(pos+1,buffer.length()-1);
    return fitness;
}

