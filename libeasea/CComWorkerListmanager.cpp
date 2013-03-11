#include "include/CComWorkerListManager.h"
#include "stdio.h"
extern "C"
{
#include "gfal_api.h"
}
#include "pthread.h"
#include <string>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <map>

extern pthread_mutex_t gfal_mutex;
extern pthread_mutex_t worker_list_mutex;



int CComWorkerListManager::refresh_worker_list()
{
  
  // read the directory list, each folder is a worker
  
  pthread_mutex_lock(&gfal_mutex);
  //pthread_mutex_lock(&worker_list_mutex);
  
  // taken from GNU C manual, adapted to gfal api
  
  DIR *dp;
  struct dirent *ep;
  
  // information concerning a worker
  CommWorker *workerinfo;
     
  // refresh nfs file list
  
  //std::string command = "ls -a "+fullpath;
  //system(command.c_str());
  
  std::map<std::string, unsigned> active_workers_names;

  for(int i=0; i< activeworkers.size(); i++) active_workers_names[ activeworkers[i].get_name()] = i;
  if(debug)printf("Refreshing workers list ...\n");
  dp = gfal_opendir (workers_path.c_str());
  
  if (dp != NULL)
  {
       while ( (ep = gfal_readdir (dp)) && !cancel)
       {
			 //only take into account folders
			 std::string s(ep->d_name);
			 std::string fullpathworker = workers_path + '/' + s;
		
			 // query if the entry if a directory
			 struct stat statusfile;
			 int status = gfal_stat(fullpathworker.c_str(), &statusfile);

			 if( status!=-1 && S_ISDIR(statusfile.st_mode))
			 {  
				  
				  
				  if( s.substr(0,6) == "worker")
				      
				  {
				      // check if we have already information concerning this worker
				      if( active_workers_names.find( s.substr(7) ) == active_workers_names.end() )
				      {	
 					  printf("Testing reading worker info:%s\n", s.c_str());
					  if(read_worker_info_file( fullpathworker, workerinfo ) == 0 )					
					  {  
					      pthread_mutex_lock(&worker_list_mutex);
					      activeworkers.push_back( *workerinfo );
	      				      if(debug)
					      {
						  printf("Worker added to the list, hostname:%s ip:%s port/%d\n",
							  workerinfo->get_name().c_str(),
							  workerinfo->get_ip().c_str(),
							  workerinfo->get_port());
					      }
					      pthread_mutex_unlock(&worker_list_mutex); 		
					  }    

				      }	  
				      else
					  active_workers_names.erase(s.substr(7) );
				  }	  
			 }
			 else if(status==-1)
			 {
			    (void)gfal_closedir (dp);
			    printf ("Cannot scan the experiment directory for find workers %s\n", fullpathworker.c_str());
			    pthread_mutex_unlock(&gfal_mutex);
			    
			    return -1;
			 } 
	}
	if(cancel) printf("Stop finding workers\n");
	
        
        (void)gfal_closedir (dp);
	pthread_mutex_unlock(&gfal_mutex); 
	// updating the active workers_path
	// delete inactive workers
	pthread_mutex_lock(&worker_list_mutex);
	std::map<std::string, unsigned>::reverse_iterator it = active_workers_names.rbegin();
	while( it != active_workers_names.rend() )
	{  
	     //activeworkers.erase( activeworkers.begin() + (*it).second );
	     ++it;
	}
	pthread_mutex_unlock(&worker_list_mutex); 		
	
  }      
  else
  {  
       printf ("Cannot scan the experiment directory for find workers %s\n", workers_path.c_str());
       pthread_mutex_unlock(&gfal_mutex);    
       return -1;
  }      

  return 0;
}

void CComWorkerListManager::terminate()
{
    cancel = true;
}

CommWorker CComWorkerListManager::get_worker_nr(int wn)
{
    return activeworkers[wn];
}


int CComWorkerListManager::read_worker_info_file( std::string workerpath, CommWorker *&workerinfo)
{
      char buffer[256];
      memset(buffer,0,256);
      std::string fullfilename = workerpath + '/' + "worker_info.txt";
      int fd = gfal_open( fullfilename.c_str(), O_RDONLY, 0777);

      if( fd != -1)
      {
	  // get individual
	  
	  int result = gfal_read(fd, buffer, 256 );
	  if( result < 0)
	  {  
	    (void)gfal_close(fd);
	    return -1;
	  }    
	  if( parse_worker_info_file(buffer, workerinfo) == -1)
	  {  
	    printf("Problem parsing worker path file %s\n",fullfilename.c_str());
	    printf("Problematic line : %s\n",buffer);
	    (void)gfal_close(fd);
	    return -1;
	  }  
	  (void)gfal_close(fd);
	  return 0;
	  // fail some read operation
	  
      }
      else 
      {
	  printf("Problem examining worker path file %s, error code: %d\n",fullfilename.c_str(),errno);
	  return -1;
      }
}

int CComWorkerListManager::parse_worker_info_file(char *buffer, CommWorker *&workerinfo) const
{
 
    char* holder = (char*)malloc(strlen(buffer)+1);
    strcpy(holder,buffer);
    char *hostname = strtok(holder, ":");
    char *mode = strtok(NULL, ":");
    
    
    
    // check first valid hostanme and protocol
    if(hostname == NULL || mode ==NULL)
    {  
        printf("*** WARNING ***\nThere is a problem with the following IP line %s: ===> IGNORING IT\n",buffer);
        return -1;
    }
    
    
    // protocol = file so no IP, NO port, may be an external node
    if( strcmp(mode,"FILE") == 0 )
    {
        std::string hn = hostname;
	workerinfo = new CommWorker(hn);
        return 0;
    }
    else if( strcmp(mode,"SOCKET") == 0 || strcmp(mode,"MPI" ==0) ) 
    {  
        // now check for ip and port/rank
	char* address = strtok(NULL, ":");
	char* port = strtok(NULL,":");
	
	if(check_ipaddress(address) == 0 && check_port(port) == 0) 
	{  
	  std::string hn = hostname;
	  std::string addr = address;
	  workerinfo = new CommWorker(hn,addr, atoi(port) );	  
	  return 0;
	}  
        else return -1;
    }

    free(holder);
    return 0;
} 


int CComWorkerListManager::check_port(char *port) const
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


int CComWorkerListManager::check_ipaddress(char *ipaddress) const
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
    free(holder);
    return 0;
}




int CComWorkerListManager::get_nr_workers() const
{
    return activeworkers.size();
}
