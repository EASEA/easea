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

#include "include/filecommworkercommunicator.h"
#include <gfal_api.h>
#include <errno.h>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <pthread.h>

using namespace std;
extern pthread_mutex_t server_mutex;
pthread_mutex_t sending_mutex;


int FileCommWorkerCommunicator::write_file()
{
    string local_filename = "file:/home/ge-user/individual_write_tmp.txt";
    
    
    current_item = individualt_to_send.top();
    
    time_t tstamp = time(NULL);
    
    string remote_filename = exp_path + '/' + current_item.first + '/' + current_filename;
    
    ofstream outputfile("/home/ge-user/individual_write_tmp.txt");
    if(outputfile.fail())return -1;
    outputfile << current_item.second;
    outputfile.close();
    int tries = 0;
    
    while( tries < 3)
    {
        stringstream remote_filename;
	remote_filename << exp_path << '/' << current_item.first << "/individual_" << tstamp+tries << ".txt";
	if( GFAL_Utils::upload(local_filename, remote_filename.str()) == 0)break;
	else if(errno==0)tries++;
        else return -1; 
    }
    return 0;
}


int FileCommWorkerCommunicator::read_file(std::string &buffer)
{
    string local_filename = "file:/home/ge-user/individual_tmp.txt";
    string remote_filename = exp_path + '/' + myself.get_name() + '/' + current_filename;
    
    if( GFAL_Utils::download(remote_filename,local_filename) != 0 )return -1;
    ifstream inputfile("/home/ge-user/individual_tmp.txt");
    if(inputfile.fail())return -1;
    inputfile >> buffer;
    inputfile.close();
    return 0;
}


int FileCommWorkerCommunicator::receive()
{
    std::string buffer;
    while(!cancel)
    {  
	if( directory_scanner->dirscan("individual") == 0 )
	{  
	    vector<string> newfiles = directory_scanner->get_newfiles();
	    for(unsigned int i=0; i<newfiles.size(); i++)files_to_read.push(newfiles[i]);
	}
	
	while( !files_to_read.empty() && !cancel )
	{  
	    current_filename = files_to_read.front();
	    files_to_read.pop();
	    if( read_file(buffer) == 0)
	    {
	        pthread_mutex_lock(&server_mutex);
		data.push(buffer);
		pthread_mutex_unlock(&server_mutex);
	    }  
	      
	    else files_to_read.push(current_filename);
	}
    }
    return 0;
}

int FileCommWorkerCommunicator::send()
{
    while(!cancel)
    {
       if(!individualt_to_send.empty())
       {
	 pthread_mutex_lock(&sending_mutex);
	 current_item = individualt_to_send.top();
	 pthread_mutex_unlock(&sending_mutex);
	 if(write_file() ==0){
	   pthread_mutex_lock(&sending_mutex);
	   individualt_to_send.pop();
	   pthread_mutex_unlock(&sending_mutex);
	 }  
       } 
    }
    return 0;
}  




int FileCommWorkerCommunicator::send(char* individual, CommWorker& destination)
{
     string buffer(individual);
     pair<string,string> item(destination.get_name(),buffer);
  
     pthread_mutex_lock(&sending_mutex);
     individualt_to_send.push(item);
     pthread_mutex_unlock(&sending_mutex);
     return 0;
}

int FileCommWorkerCommunicator::init()
{
  // scan experiment directory to find a suitable worker name
  int tries = 0;
  int start = 0;  
  
  while(tries < 5)
  {
      std::stringstream s,t;
      if(start > 0)
      {	
	s << exp_path << "/worker_" << worker_number << '_' << start;
        t << "worker_" << worker_number << '_' << start;
      }
      else
      {
	s << exp_path << "/worker_" << worker_number;
        t << "worker_" << worker_number;
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
		      myself.set_name(t.str());
		      directory_scanner = new GFAL_Utils(s.str());
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


