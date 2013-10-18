/*
 *    Copyright (C) 2013  Waldo Cancino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include "include/CComWorkerListManager_3.h"
#include "stdio.h"
extern "C"
{
#include <gfal_api.h>
#include <lcg_util.h>
}
#include "pthread.h"
#include <string>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>

extern pthread_mutex_t gfal_mutex;
extern pthread_mutex_t worker_list_mutex;


CComWorkerListManager_3::CComWorkerListManager_3(std::string path,int _debug=1):debug(_debug),cancel(false),savedonce(false),savefailed(false)
{
    worker_info_path = path + "/workers_info/";
    allworkers_info_remote_filename = worker_info_path + "allworkers_info.txt";
};

int CComWorkerListManager_3::refresh_worker_list()
{
  
  
  DIR *dp;
  struct dirent *ep;
  bool newworkers = false;
  // information concerning a worker
  CommWorker *workerinfo;
     
  
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
  
  
  dp = gfal_opendir (worker_info_path.c_str());
  
  if (dp != NULL)
  {
       while ( (ep = gfal_readdir (dp)) )
       {
	      //only take into account folders
	      std::string filename(ep->d_name);
	      std::string remote_worker_info_filename = worker_info_path + filename;
	      int pos=filename.find('.');
		if( filename.substr(0,6) == "worker" && (pos!=std::string::npos))
		    
		{
		    std::string workername = filename.substr(0,pos);

		    // check if we have already information concerning this worker
		    if( active_workers_names.find( workername ) == active_workers_names.end() )
		    {	
			printf("Worker name (filename) to be added %s %s %d\n", workername.c_str(), ep->d_name, pos );
			printf("Testing reading worker info:%s\n", remote_worker_info_filename.c_str());
			if(read_worker_info_file( remote_worker_info_filename, workerinfo ) == 0 )					
			{  
			    
			    pthread_mutex_lock(&worker_list_mutex);
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
			    pthread_mutex_unlock(&worker_list_mutex);
			    newworkers=true;
			  
			}    

		    }	
		    else    active_workers_names.erase(workername);
		    
		}	  
	}
	
	pthread_mutex_lock(&worker_list_mutex);
	for(int i = activeworkers.size()-1; i>=0; i--)
	{  
	    if( active_workers_names.find( activeworkers[i].get_name() ) != active_workers_names.end() )
	    {  
	      if(debug) printf("Delete worker %s is, no more active\n", activeworkers[i].get_name().c_str() );
	      inactiveworkers.push_back( *(activeworkers.begin() + i) );
	      activeworkers.erase(activeworkers.begin() + i);
              newworkers=true;
	    }  
	}
	pthread_mutex_unlock(&worker_list_mutex);
	if(newworkers || savefailed)
	{
	  if(save_worker_info_file()!=0){
	      printf("Cannot save worker list file %s error code is %d", allworkers_info_remote_filename.c_str(), errno);
	      savefailed=true;
	      return -1;
	  }
	}  
  }      
  else
  {  
       printf ("Cannot scan the experiment directory for find workers %s\n", worker_info_path.c_str());
       return -1;
  }      
  savefailed=false;
  return 0;
}

void CComWorkerListManager_3::terminate()
{
    cancel = true;
}

bool CComWorkerListManager_3::is_over()
{
    return cancel;
}

CommWorker CComWorkerListManager_3::get_worker_nr(int wn)
{
    return activeworkers[wn];
}


int CComWorkerListManager_3::read_worker_info_file( std::string remote_worker_info_filename, CommWorker *&workerinfo)
{
      char directory[500];
      getcwd(directory,500);
      printf("Current directory is:%s \n", get_current_dir_name() );     
      printf("Current directory is:%s \n", directory );
      char *src_file="file:/home/ge-user/tmp_worker_info.txt";
      int result = lcg_cp((char *)remote_worker_info_filename.c_str(),"file:/home/ge-user/tmp_worker_info.txt",NULL,1,NULL,0,2);
  
      if( result == 0)
      {
	  // get individual
	  std::ifstream inputfile("/home/ge-user/tmp_worker_info.txt");
	  if( inputfile.fail() )return -1;
	  std::string line;	  
	  inputfile >> line;
	  inputfile.close();
	  if( (workerinfo = CommWorker::parse_worker_string(line)) != NULL)
	  {  
	    printf("Problem parsing worker path file %s\n",remote_worker_info_filename.c_str());
	    printf("Problematic line : %s\n",line.c_str());
	    return -1;
	  }  
	  // fail some read operation
	  
      }
      else 
      {
	  printf("Cannot copy remote worker information file: %s file: %s, error code: %d\n",src_file,remote_worker_info_filename.c_str(),errno);
	  return -1;
      }
      return 0;
}

int CComWorkerListManager_3::save_worker_info_file()
{
      int result=0;
      std::ofstream outputfile("/home/ge-user/all_workers_info.txt");
      char *dest_se = getenv("VO_VO_COMPLEX_SYSTEMS_EU_DEFAULT_SE");
      char guid[256];
      if( outputfile.fail() ) return -1;
      outputfile << activeworkers.size() << std::endl;
      for(int i=0; i<activeworkers.size(); i++)
      {
    	 /*std::stringstream s;

	 s << activeworkers[i].get_name() << ':' << activeworkers[i].get_hostname() << ':';
	 if(activeworkers[i].get_ip() == "noip")s << "FILE::";
	 else s << "SOCKET:" <<activeworkers[i].get_ip() << ':' << activeworkers[i].get_port();
	 outputfile << s.str() << std::endl;*/
	 outputfile << activeworkers[i] << std::endl;
      }
      outputfile.close();
      
      // replace the old file by the new one
      if(savedonce)
      {
	printf("Deleting old remote worker list file ....\n");
	result = lcg_del((char*)allworkers_info_remote_filename.c_str(), 5, NULL,NULL,NULL,0,0); 
      }
      if(result!=0)return -1;
      printf("Copying new  remote worker list file ....\n");
      char *src_file="file:///home/ge-user/all_workers_info.txt";
      result = lcg_cr("file:/home/ge-user/all_workers_info.txt",dest_se,NULL,(char*)allworkers_info_remote_filename.c_str(),NULL,NULL,1,NULL,0,2,guid);
      if( result != 0 )
      {	
	 printf("Canot copy remote worker information file %s file %s, error code: %d\n",src_file,allworkers_info_remote_filename.c_str(),errno); 
	 return -1;
      } 
      savedonce=true;
      return 0;
}


/*
int CComWorkerListManager_3::parse_worker_info_file(const char *buffer, CommWorker *&workerinfo) const
{
 
    char* holder = (char*)malloc(strlen(buffer)+1);
    strcpy(holder,buffer);
    char *foldername = strtok(holder, ":");
    char *hostname = strtok(NULL, ":");
    char *mode = strtok(NULL, ":");
    
    
    
    // check first valid hostanme and protocol
    if(foldername == NULL || hostname == NULL || mode ==NULL)
    {  
        printf("*** WARNING ***\nThere is a problem with the following IP line %s: ===> IGNORING IT\n",buffer);
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


int CComWorkerListManager_3::check_port(char *port) const
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


int CComWorkerListManager_3::check_ipaddress(char *ipaddress) const
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

*/


int CComWorkerListManager_3::get_nr_workers() const
{
    return activeworkers.size();
}


int CComWorkerListManager_3::get_nr_inactive_workers() const
{
    return inactiveworkers.size();
}
