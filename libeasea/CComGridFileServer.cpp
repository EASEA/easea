#include "include/CComGridFileServer.h"
#ifndef WIN32
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <net/if.h>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
//#include <error.h>
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

extern pthread_mutex_t server_mutex;
pthread_mutex_t sending_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t worker_list_mutex = PTHREAD_MUTEX_INITIALIZER;

CComGridFileServer::CComGridFileServer(char* path, char* expname, std::queue< string >* _data, int dbg) {
  
    data = _data;
    std::string exp(expname);
    std::string pathname(path);
    fullpath = pathname + expname;
    wait_time = 1500;
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
    
     if( thread_read = Cthread_create(&CComGridFileServer::file_readwrite_thread, (void *)this) < 0) {
        printf("pthread create failed. exiting\n"); exit(1);
    }
    
/*    if( thread_read = Cthread_create(&CComGridFileServer::file_read_thread, (void *)this) < 0) {
        printf("pthread create failed. exiting\n"); exit(1);
    }
    if( thread_write = Cthread_create(&CComGridFileServer::file_write_thread, (void *)this) < 0) {
        printf("pthread create failed. exiting\n"); exit(1);
    }*/

}



// Gfal grid code

int CComGridFileServer::refresh_worker_list()
{
  
  // read the directory list, each folder is a worker
  
  pthread_mutex_lock(&worker_list_mutex);
  if(worker_list.size()>0)worker_list.clear();
  
  // taken from GNU C manual
  
  DIR *dp;
  struct dirent *ep;
     
  // refresh nfs file list
  
  //std::string command = "ls -a "+fullpath;
  //system(command.c_str());


  dp = gfal_opendir (fullpath.c_str());
  
  if (dp != NULL)
  {
       while ((ep = gfal_readdir (dp)) && !cancel)
       {
			 //only take into account folders
			 string s(ep->d_name);
			 string fullpathworker = fullpath + '/' + s;
			 struct stat statusfile;
			 int status = gfal_stat(fullpathworker.c_str(), &statusfile);

			 if( status!=-1 && S_ISDIR(statusfile.st_mode))
			 {  
				  
				  
				  if( s.substr(0,6) == "worker" && s!=workername)
				  {
				  worker_list.push_back(s);
				  if(debug)
					printf("Worker %s added to the list\n",s.c_str());
				  }	  
			 }
			 else if(status==-1)
			 {
			   (void)gfal_closedir (dp);
			   printf ("Cannot scan the experiment directory for find workers %s\n", fullpathworker.c_str());
			   pthread_mutex_unlock(&worker_list_mutex); 
			   return -1;
			 } 
	}
	if(cancel) printf("Stop finding workers, thread canceled\n");
        pthread_mutex_unlock(&worker_list_mutex); 		
        (void)gfal_closedir (dp);
  }      
  else
  {  
       printf ("Cannot scan the experiment directory for find workers %s\n", fullpath.c_str());
       pthread_mutex_unlock(&worker_list_mutex);    
       return -1;
  }      

  return 0;
}


int CComGridFileServer::refresh_file_list()
{
  // clear the file to process
  new_files.clear();
  
  // taken from GNU C manual
  
  DIR *dp;
  struct dirent *ep;
  
  std::string workerpath = fullpath + '/' + workername;
  std::set<string>::iterator it;   

  //std::string command = "ls -a "+workerpath;
  //system(command.c_str());


  dp = gfal_opendir (workerpath.c_str());
  if (dp != NULL)
  {
       printf("Refreshing filelist (new individuals) in %s\n",workerpath.c_str());
       while ((ep = gfal_readdir (dp)) && !cancel)
       {
		 //only take into account folders
		 string s(ep->d_name);
		 string fullfilename = workerpath + '/' + s;
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
		    return 0;
		 }   
	        
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




int CComGridFileServer::determine_worker_name(int start)
{
  
  // scan experiment directory to find a suitable worker name
  while(1)
  {
      std::stringstream s,t;
      s << fullpath << "/worker_" << start;
      t << "worker_" << start;

      DIR *dp;
  
     
  // refresh nfs file list
  
  
      //dp = gfal_opendir (fullpath.c_str());
      //(void)gfal_closedir(dp);

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
		      break;
	      }	
	      else
	      {
		      printf("Cannot create worker experiment folder %s; error reported is %d\n", s.str().c_str(), errno);
		      return -1;
	      }
      }  
      else
      {
	      printf("Cannot create worker experiment folder %s; error reported is %d\n", s.str().c_str(), errno);
	      return -1;
      }
  }
  return 0;
}




int CComGridFileServer::determine_file_name(std::string fulltmpfilename, std::string workerdestname)
{
	
   time_t tstamp = time(NULL);
    while(1)
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
	if(debug)printf("unable to rename tmp file to %s,  trying another name\n", fullfilename.c_str());
        tstamp++;
    }	
}  


int CComGridFileServer::create_tmp_file(int &fd, std::string workerdestname, std::string &fullfilename)
{
   time_t tstamp = time(NULL);
    while(1)
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
		   break;
	}
	else
	{
	    // error ocurred, so two alternatives
	    // the path does not exit anymore (race condition)
	  DIR *dp;
	  std::string workerpath = fullpath + '/' + workerdestname;
	  dp = gfal_opendir (workerpath.c_str());
	  if( dp == NULL)
	  {  
	     printf("The worker path is not accesible %s, maybe worker finishes\n", workerpath.c_str());
	     return -1;
	  }
	  (void)gfal_closedir(dp);
	  if(debug)
	      printf("Failed to create filename %s failed, trying another name \n :", fullfilename.c_str());
	    
	    tstamp++;
	    continue;
	  }  
    }
    return 0;
}


void CComGridFileServer::send_individuals()
{
    std::list<std::pair<std::string,std::string> >::iterator it = writedata.begin();
    std::list<std::pair<std::string,std::string> >::iterator it2 = it;
    std::pair<std::string,std::string> item;
    while( it != writedata.end() && !cancel )
    {
        if(send_file_worker( (*it).first, (*it).second ) ==0 )
	{  
	     pthread_mutex_lock(&sending_mutex);
     	     it2 = it;
	     ++it;
	     writedata.erase(it2);
	     pthread_mutex_unlock(&sending_mutex);
	} 
	else ++it;
      
    }  
    if(cancel)printf("Stop sending individuals, thread canceled\n");
    
}  

int CComGridFileServer::send_file_worker(std::string buffer, std::string workerdestname)
{
    int fd;
    std::string tmpfilename;
    if( create_tmp_file(fd, workerdestname, tmpfilename) == 0)
     {
	  int result = gfal_write( fd, buffer.c_str(), buffer.size() );
	  if(result >0)
	  {
		printf("gfal_write returns %d for written %d bytes \n", result, buffer.size() );
		(void)gfal_close(fd);
		if( determine_file_name(tmpfilename, workerdestname) == 0) return 0;
		else return -1;
	  }
	  else
	  {
	    (void)gfal_close(fd);
	    return -1;
	    
	  }  
	  // now rename 
	  
     }
     else
     {
        printf("Cannot write individual to file in path %s/%s", fullpath.c_str(), workername.c_str() ); 
	return -1;
     }
     return 0;
}
  
  

int CComGridFileServer::send_file(char *buffer, int dest)
{
     //first thing, prevent send to myself
     
     int fd;
     std::string tmpfilename;
     std::string workdest;
     
     
     pthread_mutex_lock(&worker_list_mutex);
     // worker list can be changed since last refresh
     if(dest>=worker_list.size()){
       
       if(worker_list.size()>0)
       {	 
          workdest = worker_list[worker_list.size()-1];
          if(debug)
             printf("Invalid destination worker, send to another worker : %s\n", workdest.c_str());
       }	  
       else
       {	 
	     printf("No destination workers available \n");
	     return 0;
       }     
     }     
     if(workername == worker_list[dest])
     {
        if(debug)
	    printf("I will not send the individual to myself, it is not a fatal error, so continue\n");
        return 0;
     }
          
     workdest = worker_list[dest];
     pthread_mutex_unlock(&worker_list_mutex);

     std::string buffer_str(buffer);
     std::pair<std::string,std::string> item(buffer_str,workdest);

     
     pthread_mutex_lock(&sending_mutex);
     writedata.push_back(item);
     pthread_mutex_unlock(&sending_mutex);
     return 0;
}     
     
     
     
     
     // determine the name to send a file
     
 


int CComGridFileServer::file_read(const char *filename)
{
    std::string workerfile(filename);
    std::string fullfilename = fullpath + '/' + workername + '/' + workerfile;
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


void CComGridFileServer::readfiles()
{
      if(refresh_file_list() == 0)
      {
	  std::list<string>::iterator it;
	  for(it= new_files.begin(); it != new_files.end() && !cancel; it++)
	  {
	    if(file_read((*it).c_str()) == 0)
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

void CComGridFileServer::run_read()
{
       while(!cancel) {/*forever loop*/
	  
		readfiles();
		refresh_worker_list();
		// check for new files
		sleep(4);
	}
}

void CComGridFileServer::run_write()
{
       while(!cancel) {/*forever loop*/
	  
		send_individuals();
		// check for new files
		sleep(4);
	}
}

void CComGridFileServer::run_readwrite()
{
      while(!cancel) {/*forever loop*/
	  
	  send_individuals();
	  readfiles();
	  refresh_worker_list();
		// check for new files
	sleep(4);
	}

}


void * CComGridFileServer::file_readwrite_thread(void *parm) {
	CComGridFileServer *server = (CComGridFileServer*)parm;

	server->run_readwrite();
	return NULL;
}	


void * CComGridFileServer::file_read_thread(void *parm) {
	CComGridFileServer *server = (CComGridFileServer*)parm;
	server->run_read();
	return NULL;
}	

void * CComGridFileServer::file_write_thread(void *parm) {
	CComGridFileServer *server = (CComGridFileServer*)parm;
	server->run_write();
	return NULL;
}	


int CComGridFileServer::number_of_workers()
{
      return worker_list.size();
}  

void CComGridFileServer::read_data_lock() {
	pthread_mutex_lock(&server_mutex);
};

void CComGridFileServer::read_data_unlock() {
	pthread_mutex_unlock(&server_mutex);
};


CComGridFileServer::~CComGridFileServer()	
{
    printf("Calling the destructor ....\n");
    this->cancel =1;
    Cthread_join(thread_read, NULL);
    Cthread_join(thread_write, NULL);
    // erase working path
    printf("Filserver thread cancelled ....\n");
    int tries=0;
    std::string workerpath = fullpath + '/' + workername;
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
		    if( gfal_unlink(fullfilename.c_str()) != 0)
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
}    


