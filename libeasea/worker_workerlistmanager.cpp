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

#include "include/worker_workerlistmanager.h"
#include "include/gfal_utils.h"
#include "stdio.h"
#include <fstream>
#include <unistd.h>

extern pthread_mutex_t gfal_mutex;
extern pthread_mutex_t worker_list_mutex;
using namespace std;

Worker_WorkerListManager::Worker_WorkerListManager(string exp_path, int debug):AbstractWorkerListManager(debug),workerfile_timestamp(0)
{
  workerlist_remote_filename = exp_path+"/workers_info/allworkers_info.txt";
  workerlist_local_filename = "file:/home/ge-user/allworkers_info.txt";
}  


int Worker_WorkerListManager::refresh_worker_list()
{
  while(!cancel)
  {  
	workernames_idx.clear();  
	// create a list 
	for(unsigned int i=0; i< activeworkers.size(); i++) workernames_idx.insert(activeworkers[i].get_name());
	
	pthread_mutex_lock(&gfal_mutex);
	int result =  GFAL_Utils::download(workerlist_remote_filename, workerlist_local_filename, workerfile_timestamp);
	pthread_mutex_unlock(&gfal_mutex);

	if(result != 0)
	{
	      printf ("Cannot open worker information file %s\n", workerlist_remote_filename.c_str());
	      return -1;

	}  
	process_workerlist_file();
	update_lists();
	sleep(30);
  }
  return 0;
}


void Worker_WorkerListManager::update_lists()
{
  pthread_mutex_lock(&worker_list_mutex);
  for(int i = activeworkers.size()-1; i>=0; i--)
  {  
		
	if( workernames_idx.find( activeworkers[i].get_name() ) != workernames_idx.end() )
	{  
	  if(debug) printf("Delete worker %s is, no more active\n", activeworkers[i].get_name().c_str() );
	  inactiveworkers.push_back( activeworkers[i] );
	  activeworkers.erase( activeworkers.begin() + i );
	}  
  }
  pthread_mutex_unlock(&worker_list_mutex);
}  


int Worker_WorkerListManager::process_workerlist_file()
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
	if( (workerinfo = CommWorker::parse_worker_string( line )) != NULL )
	{
	     if( workernames_idx.find( workerinfo->get_name() ) == workernames_idx.end() )
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
	     else    workernames_idx.erase(workerinfo->get_name());
	     delete workerinfo;
	}    
    }
    pthread_mutex_unlock(&worker_list_mutex);
    inputfile.close();
    return 0;
}  


CommWorker Worker_WorkerListManager::get_worker_nr(int wn) const
{
    pthread_mutex_lock(&worker_list_mutex);
    CommWorker returnworker = activeworkers[wn];
    pthread_mutex_unlock(&worker_list_mutex);
    return returnworker;
}


