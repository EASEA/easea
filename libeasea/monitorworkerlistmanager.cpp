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

#include "include/monitorworkerlistmanager.h"
#include "include/CComWorker.h"
#include <vector>
#include <fstream>
#include "errno.h"

using namespace std;
pthread_mutex_t worker_list_mutex;

MonitorWorkerListManager::MonitorWorkerListManager(string exp_path,int nw,int _debug=1):AbstractWorkerListManager(_debug),num_workers(nw),saved_once(false),save_failed(true),modified(false)
{
    workerinfo_path=exp_path+"/workers_info";
    this->directory_scanner = new GFAL_Utils(workerinfo_path);  
    chunk_size = num_workers/10;
    if( chunk_size < 10 )chunk_size=10;
    else if(chunk_size > 20)chunk_size=20;
}  


int MonitorWorkerListManager::refresh_worker_list()
{
    modified=false;
    enqueue_files();
    if(!cancel){
      process_worker_files();
      update_lists();
      if(modified || save_failed)return save_worker_file_info();
    }  
    return 0;
}


bool MonitorWorkerListManager::terminated()
{
    return cancel;
}

void MonitorWorkerListManager::enqueue_files()
{
   if( directory_scanner->dirscan("worker_") ==0)
   {  

	// insert the new files
	vector<string> newfiles=directory_scanner->get_newfiles();
	for(unsigned int i=0;i<newfiles.size();i++){
	  if(debug)cout << "new file found:" << newfiles[i] << endl; 
	  if(newfiles[i]!="worker_finishall")files_to_process.push( newfiles[i] );
	  else cancel=true;
	}  
   }
}

int MonitorWorkerListManager::read_worker_file(std::string &buffer) const
{
     ifstream localfile("/home/ge-user/tmp_worker_info.txt");
     if (localfile.fail())return -1;
     localfile >> buffer;
     localfile.close();
     return 0;
}  


string MonitorWorkerListManager::extract_worker_name(const string &filename) const
{
    string workername;
    unsigned pos = filename.find('.');
    if(pos!= string::npos)workername=filename.substr(0,pos);
    return workername;
}  



void MonitorWorkerListManager::process_worker_files()
{
   string current_filename,remote_filename,local_filename;
   string worker_string;
   int processed=0;
   CommWorker *worker=NULL;
   
   vector<string> failed_to_process;
    
   
   while( !files_to_process.empty() )
   {
	  // check for new workernames
 	  
	  current_filename = files_to_process.front();
	  files_to_process.pop();
	  
	  remote_filename = workerinfo_path + '/' + current_filename;
	  local_filename = "file://home/ge-user/tmp_worker_info.txt";
	  
	  // download remote workerinfo file
	  if( GFAL_Utils::download(remote_filename,local_filename) == 0 
		&& read_worker_file(worker_string) == 0 &&
		( worker = CommWorker::parse_worker_string(worker_string) ) != NULL)
		  
	  {
	  
	      activeworkers.push_back( *worker );
	      workernames_idx.insert( worker->get_name() );
	      cout << "adding worker: " << *worker << endl;
	      modified=true;
	      if( ++processed == chunk_size)break;
	  }
	  // if communication error when downloading file, try again later
	  else if(errno==ECOMM)failed_to_process.push_back( current_filename );
	  else cout << "failed to process file:" << current_filename << " error code:" << errno << endl;
   }  
   //reinsert in the queue
}   


int MonitorWorkerListManager::save_worker_file_info()
{
  
      //if(activeworkers.size()==0)return 0;
      std::ofstream outputfile("/home/ge-user/allworkers_info.txt");

      save_failed=true;
      
      if( outputfile.fail() ) return -1;
      outputfile << activeworkers.size() << std::endl;
      for(unsigned i=0; i<activeworkers.size(); i++)
      {
	 outputfile << activeworkers[i] << std::endl;
      }
      outputfile.close();
      
      
      string remote_workerlist_filename = workerinfo_path + "/allworkers_info.txt";
      // replace the old file by the new one
      if(saved_once)
	//printf("Deleting old remote worker list file ....\n");
	if( GFAL_Utils::delete_file( remote_workerlist_filename ) !=0)return -1;
      cout << "Saving " << activeworkers.size() << "  workers" << endl;
      printf("Copying new  remote worker list file ....\n");
      
      if( GFAL_Utils::upload("file:/home/ge-user/allworkers_info.txt",remote_workerlist_filename) !=0)
      {
	  cout << "failed to upload file to " << remote_workerlist_filename << " error code is:" << errno <<endl;
	  return -1;
      }
      saved_once=true;
      save_failed=false;
      return 0;
}  

   
void MonitorWorkerListManager::update_lists()   
{
   // take care of deleted workers_info
   string current_workername;
   
   // create index of deleted files
   vector<string> deleted_files=directory_scanner->get_deletedfiles();
   set<string> deleted_workernames_idx;
   for( unsigned i=0; i<deleted_files.size(); i++)deleted_workernames_idx.insert(  extract_worker_name( deleted_files[i] ) );
   
   
   for(int i=activeworkers.size()-1; i>=0; i--)
   {
        current_workername = activeworkers[i].get_name();
	
        if( deleted_workernames_idx.find( current_workername) != deleted_workernames_idx.end() )
	{  
	    inactiveworkers.push_back(activeworkers[i] );
	    activeworkers.erase(activeworkers.begin() + i);
	    workernames_idx.erase(current_workername);
	    cout << "delete worker " << current_workername << endl;
	    modified=true;

	}    
   }  
}  
  