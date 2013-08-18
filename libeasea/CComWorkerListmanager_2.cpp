#include "include/CComWorkerListManager_2.h"
#include "stdio.h"
extern "C"
{
#include <lcg_util.h>
}
#include "pthread.h"
#include <string>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <iostream>
#include <fstream>
#include <unistd.h>

extern pthread_mutex_t gfal_mutex;
extern pthread_mutex_t worker_list_mutex;

CComWorkerListManager_2::CComWorkerListManager_2(std::string path, int _debug):debug(_debug),cancel(false)
{
  workerlist_remote_filename = path+"/workers_info/allworkers_info.txt";
  workerlist_local_filename = "file:/home/ge-user/allworkers_info.txt";
}

int CComWorkerListManager_2::refresh_worker_list()
{
  
  active_workers_names.clear();  
  // create a list 
  for(unsigned int i=0; i< activeworkers.size(); i++) active_workers_names.insert(activeworkers[i].get_name());
  //  active_workers_names[ activeworkers[i].get_name()] = 0;
  if(debug)
  {  
    printf("Refreshing workers list (already in list)...\n");
    std::set<std::string>::iterator it = active_workers_names.begin();
    std::set<std::string>::iterator ite = active_workers_names.end();
    while(it!=ite)
    {
      printf("%s\n", (*it).c_str() );
      ++it;
    }  
  }  
  
  pthread_mutex_lock(&gfal_mutex);
  int result =  lcg_cp((char *)workerlist_remote_filename.c_str(),(char *)workerlist_local_filename.c_str(),NULL,1,NULL,0,1);
  pthread_mutex_unlock(&gfal_mutex);
  if(result != 0)
  {
         printf ("Cannot open worker information file %s\n", workerlist_remote_filename.c_str());
	 return -1;

  }  
  
  if ( process_workerlist_file() != 0 ) return -1;

  pthread_mutex_lock(&worker_list_mutex);
  for(int i = activeworkers.size()-1; i>=0; i--)
  {  
	if( active_workers_names.find( activeworkers[i].get_name() ) != active_workers_names.end() )
	{  
	  if(debug) printf("Delete worker %s is, no more active\n", activeworkers[i].get_name().c_str() );
	  inactiveworkers.push_back( *(activeworkers.begin() + i) );
	  activeworkers.erase(activeworkers.begin() + i);
	}  
  }

  pthread_mutex_unlock(&worker_list_mutex);
  
  return 0;
}


int CComWorkerListManager_2::process_workerlist_file()
{
    
    std::ifstream inputfile("/home/ge-user/allworkers_info.txt");
    
    
    if(inputfile.fail() )return -1;
    
    if(debug)printf("Reading worker info local file\n");
    
    int nworkers;
    // get the number of workers
    inputfile >> nworkers;
    if(inputfile.fail() )return -1;
    CommWorker *workerinfo=NULL;
    std::string line;
    
    pthread_mutex_lock(&worker_list_mutex);
    for(int i=0; i< nworkers; i++)
    {
        inputfile >> line;
	if( parse_worker_info_file(line.c_str(),workerinfo) == 0)
	{
	     if( active_workers_names.find( workerinfo->get_name() ) == active_workers_names.end() )
	     {  

		 activeworkers.push_back( *workerinfo );
		 if(debug)
		 {
		    printf("Worker %d added to the list, foldername %s hostname:%s ip:%s port/%d\n",
		    activeworkers.size(),
		      workerinfo->get_name().c_str(),
                                        workerinfo->get_hostname().c_str(),
					workerinfo->get_ip().c_str(),
					workerinfo->get_port());
		  }
	     }
	     else    active_workers_names.erase(workerinfo->get_name());
	     delete workerinfo;
	}    
    }
    
    pthread_mutex_unlock(&worker_list_mutex); 		
    inputfile.close();
    return 0;
}  


void CComWorkerListManager_2::terminate()
{
    cancel = true;
}

CommWorker CComWorkerListManager_2::get_worker_nr(int wn)
{
    return activeworkers[wn];
}



int CComWorkerListManager_2::parse_worker_info_file(const char *buffer, CommWorker *&workerinfo) const
{
 
    char* holder = (char*)malloc(strlen(buffer)+1);
    strcpy(holder,buffer);
    char *foldername = strtok(holder, ":");
    char *hostname = strtok(NULL, ":");
    char *mode = strtok(NULL, ":");
    
    
    
    // check first valid hostanme and protocol
    if(foldername == NULL || hostname == NULL || mode ==NULL)
    {  
        return -1;
    }
    
    
    // protocol = file so no IP, NO port, may be an external node
    if( strcmp(mode,"FILE") == 0 )
    {
        std::string fn = foldername;
        std::string hn = hostname;
	workerinfo = new CommWorker(fn,hn);
        return 0;
    }
    else if( strcmp(mode,"SOCKET") == 0 || strcmp(mode,"MPI") ==0) 
    {  
        // now check for ip and port/rank
	char* address = strtok(NULL, ":");
	char* port = strtok(NULL,":");
	
	if(check_ipaddress(address) == 0 && check_port(port) == 0) 
	{  
          std::string fn = foldername;
	  std::string hn = hostname;
	  std::string addr = address;
	  workerinfo = new CommWorker(fn, hn,addr, atoi(port) );	  
	  return 0;
	}  
        else return -1;
    }

    free(holder);
    return 0;
} 


int CComWorkerListManager_2::check_port(char *port) const
{
    int nibble = atoi(port);
    if(nibble<0){
	  return -1;
    }
    std::string s = port;
    for(unsigned int i=0; i<s.length(); i++){
        if(!isdigit(s[i])){
	      return -1;
        }
    }
    return 0;
 }  


int CComWorkerListManager_2::check_ipaddress(char *ipaddress) const
{
    char* holder = (char*)malloc(strlen(ipaddress)+1);
    strcpy(holder,ipaddress);

   char* byte = strtok(holder,".");
    int nibble = 0, octets = 0, flag = 0;
    while(byte != NULL){
        octets++;
        if(octets>4){
	    free(holder);
            return -1;
        }
        nibble = atoi(byte);
        if((nibble<0)||(nibble>255)){
	    free(holder);
            return -1;
        }
        std::string s = byte;
        for(unsigned int i=0; i<s.length(); i++){
            if(!isdigit(s[i])){
	      free(holder);
	      return -1;
            }
        }
        byte = strtok(NULL,".");
    }
    if(flag || octets<4){
            free(holder);
            return -1;
    }
    //free(holder);
    return 0;
}




int CComWorkerListManager_2::get_nr_workers() const
{
    return activeworkers.size();
}

int CComWorkerListManager_2::get_nr_inactive_workers() const
{
    return inactiveworkers.size();
}
