#include "include/CComWorkerListManager.h"
#include "stdio.h"
#include "gfal_api.h"
#include "pthread.h"
#include <string>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

extern pthread_t gfal_mutex;




int CComWorkerListManager::refresh_worker_list()
{
  
  // read the directory list, each folder is a worker
  
  pthread_mutex_lock(&gfal_mutex);

  
  // taken from GNU C manual, adapted to gfal api
  
  DIR *dp;
  struct dirent *ep;
  
  // information concerning a worker
  CommWorker workerinfo;
     
  // refresh nfs file list
  
  //std::string command = "ls -a "+fullpath;
  //system(command.c_str());


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
				  
				  
				  if( s.substr(0,6) == "worker" && read_worker_info_file( fullpathworker, workerinfo ) == 0 )
				  {
				      activeworkers.push_back(s);
				      if(debug)
					    printf("Worker %s added to the list\n",s.c_str());
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
        pthread_mutex_unlock(&gfal_mutex); 		
        (void)gfal_closedir (dp);
  }      
  else
  {  
       printf ("Cannot scan the experiment directory for find workers %s\n", workers_path.c_str());
       pthread_mutex_unlock(&gfal_mutex);    
       return -1;
  }      

  return 0;
}

inline void CComWorkerListManager::cancel() const 
{
    cancel = true;
}

inline CommWorker CComWorkerListManager::get_worker_nr(int wn)
{
    return activeworkers[wn];
}


int CComWorkerListManager::read_worker_info_file( std::string workerpath, CommWorker &workerinfo)
{
      char buffer[256];
      std::string fullfilename = workerpath + '/' + "worker_info.txt";
      int fd = gfal_open( fullfilename.c_str(), O_RDONLY, 0777);;
      if( fd != -1)
      {
	  // get individual
	  int result = gfal_read(fd, buffer, 256 );
	  if( result < 0)
	  {  
	    (void)gfal_close(fd);
	    return -1;
	  }    
	  if( parse_worker_info_file(buffer, workerinfo) == -1) return -1;
	  (void)gfal_close(fd);
	  return 0;
	  // fail some read operation
	  
      }
      else 
	  return -1;
}

int CComWorkerListManager::parse_worker_info_file(char *buffer, CommWorker &workerinfo)
{
 
    if( checkValidLine(buffer) )
    {
	std::string hostname = strtok(buffer, ":");
	char* address = strtok(NULL, ":");
	std::string hostaddress = address;
	int port = atoi ( strtok(NULL,":") );
	 
	CommWorker tmp(hostname, hostaddress, port);
	workerinfo = tmp;    

    }
    else return -1;
      
} 


/**
 * Check the validity of an IP line. This line should have the form like : hostname:1.2.3.4:5 (ip:port).
 *
 * @ARG line : the line containing the ip and port description.
 * @ @RETURN : boolean containing the result of the regex match.
 *
 */
//www.dreamincode.net/forums/topic/168930-valid-or-not-for-a-ip-address/
int CComWorkerListManager::checkValidLine(char *line){
    char* holder = (char*)malloc(strlen(line)+1);
    strcpy(holder,line);
    char *hostname = strtok(holder, ":");
    char* address = strtok(NULL, ":");
    char* port = strtok(NULL,":");
    

    //printf("IP %s\n",address);
    //printf("port %s\n",port);

    //Check if there is an IP and a port
    if(hostname==NULL || address==NULL || port==NULL){
        printf("*** WARNING ***\nThere is a problem with the following IP line: " << line << "\t===> IGNORING IT\n";
        return -1;
    }

    //Check if it is a valid ip
    char* byte = strtok(address,".");
    int nibble = 0, octets = 0, flag = 0;
    while(byte != NULL){
        octets++;
        if(octets>4){
            printf("*** WARNING ***\nThere is a problem with the following IP line: " << line << "\t===> IGNORING IT\n";
            return -1;
        }
        nibble = atoi(byte);
        if((nibble<0)||(nibble>255)){
            printf("*** WARNING ***\nThere is a problem with the following IP line: " << line << "\t===> IGNORING IT\n";
            return -1;
        }
        std::string s = byte;
        for(unsigned int i=0; i<s.length(); i++){
            if(!isdigit(s[i])){
	      printf("*** WARNING ***\nThere is a problem with the following IP line: " << line << "\t===> IGNORING IT\n";
	      return -1;
            }
        }
        byte = strtok(NULL,".");
    }
    if(flag || octets<4){
            printf("*** WARNING ***\nThere is a problem with the following IP line: " << line << "\t===> IGNORING IT\n";
            return -1;
    }

    //Check if it is a valid port
    nibble = atoi(port);
    if(nibble<0){
	  printf("*** WARNING ***\nThere is a problem with the following IP line: " << line << "\t===> IGNORING IT\n";
	  return -1;
    }
    std::string s = port;
    for(unsigned int i=0; i<s.length(); i++){
        if(!isdigit(s[i])){
	      printf("*** WARNING ***\nThere is a problem with the following IP line: " << line << "\t===> IGNORING IT\n";
	      return -1;
        }
    }
    
    free(holder);
    return 0;
}


inline void CComWorkerListManager::cancel()
{
    cancel = true;
}


inline int CComWorkerListManager::get_nr_workers() const
{
    return activeworkers.size();
}