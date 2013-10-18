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

extern "C" {
#include "gfal_api.h"
#include "lcg_util.h"
}
#include "include/gfal_utils.h"
#include <list>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <unistd.h>

using namespace std;

int GFAL_Utils::debug=0;

/**
 * @detailed Scan the directory finding files starting by start_with string
 * 
 */
int GFAL_Utils::dirscan(string start_with="")
{
  DIR *dp;
  struct dirent *ep;
  vector<string> dirfiles;
 
  // check if the directory has changed
  if( refresh_timestamp()==0 && !is_updated() )
  {
 
      if(debug) cout << "Reading directory ..." << remote_dir_path << endl;
      dp = gfal_opendir ( remote_dir_path.c_str() );
    
     // reading dir
    if (dp != NULL)
    {
	while ( (ep = gfal_readdir (dp)) )
	{
		//only take into account folders
		string filename(ep->d_name);

		unsigned int pos=filename.find(start_with);
		// add file if meet the requeriments
		if((pos!=string::npos))
		    dirfiles.push_back(filename);
		
	  }
    }
    else
    {
      list_update();
      return -1;
    }  
    gfal_closedir(dp);
    current_files = dirfiles;
  }  
  list_update();
  if(firstime)firstime=false;
  return 0;
}  


int GFAL_Utils::get_timestamp( string remote_filename, time_t &t)
{
  struct stat status;
  int result = gfal_stat( remote_filename.c_str(), &status);
  if(result!=0)return -1;
  t = status.st_mtime;
  return 0;
}  

int GFAL_Utils::refresh_timestamp()
{
   previous_timestamp = current_timestamp;
   if( GFAL_Utils::get_timestamp( remote_dir_path, current_timestamp) != 0)return -1;
   if(debug)cout << "directory timestamp:" << current_timestamp << endl;
   return 0;
}

bool GFAL_Utils::is_updated()
{
    return( firstime ? false : (previous_timestamp == current_timestamp) );
}

    

const std::vector<string>& GFAL_Utils::get_dirfiles() const
{
  return current_files;
}

const std::vector<string>& GFAL_Utils::get_newfiles() const
{
  return new_files;
}

const std::vector<string>& GFAL_Utils::get_deletedfiles() const
{
  return delete_files;
}


void GFAL_Utils::list_update()
{
   new_files.clear();
   //delete_files.clear();
   set<string> newindex;
   
   // check for new files
   for(unsigned int i=0; i<current_files.size(); i++)
   {
      newindex.insert(current_files[i]);
      // new files are not present in the last list
      if( lastfiles_idx.find( current_files[i] ) == lastfiles_idx.end() )
	 new_files.push_back( current_files[i] );
      else
	 lastfiles_idx.erase( current_files[i] );
   }  
   
   set<string>::iterator it, it_e;
   // check for deleted files
   for( it = lastfiles_idx.begin(); it!= lastfiles_idx.end(); ++it) 
         delete_files.push_back( *it );
   
   lastfiles_idx = newindex;
}


int GFAL_Utils::upload(string local_filename, string remote_filename, bool verbose)
{
    return lcg_cr( (char*)local_filename.c_str(), 
		   getenv("VO_VO_COMPLEX_SYSTEMS_EU_DEFAULT_SE"),
		   NULL,(char*)remote_filename.c_str(),NULL,NULL,1,
		   NULL,0,(verbose ? 2 : 0),NULL);
}  

int GFAL_Utils::download(string remote_filename, string local_filename, time_t &timestamp, bool verbose)
{
     time_t oldtimestamp=timestamp;
     if( GFAL_Utils::get_timestamp(remote_filename,timestamp) !=0 ) return -1;
     if(oldtimestamp!=timestamp)
     {  
	if( GFAL_Utils::download(remote_filename,local_filename,verbose) !=0)
	{
	    timestamp=oldtimestamp;
	    return -1;
	}
     }
     return 0; 
}  

int GFAL_Utils::download(string remote_filename, string local_filename, bool verbose)
{
      return lcg_cp((char *)remote_filename.c_str(),(char*)local_filename.c_str(),
		    NULL,1,NULL,0,(verbose ? 2 : 0));
}  

int GFAL_Utils::delete_file(string remote_filename)
{
      return lcg_del((char*)remote_filename.c_str(), 5, NULL,NULL,NULL,0,0); 
}  

int GFAL_Utils::rm_dir(string remote_dir_path, int ntries)
{
    string fullfilename;
    int tries;
    while(tries < ntries)
    {
	DIR *dp;
	struct dirent *ep;
	
	dp = gfal_opendir (remote_dir_path.c_str());
	if (dp != NULL)
	{
	    int countfiles=0;
	    
	    
	    while ((ep = gfal_readdir (dp)))
	    {
	      //only take into account folders
	      std::string s(ep->d_name);
	      fullfilename = remote_dir_path + '/' + s;

	      struct stat statusfile;
	      int result = gfal_stat(fullfilename.c_str(), &statusfile);

	      // delete a regular file
	      if( result != -1 && S_ISREG(statusfile.st_mode))
	      {  
		    if( delete_file( fullfilename ) != 0)
		    {
		      if(debug)printf("Finish worker : Cannot erase  the file %s\n", fullfilename.c_str());
		      break;
		    }  
	      }      
	      // error; try again
	      else if(result == -1)
	      {
			(void)gfal_closedir (dp);
			  sleep(4);
			tries++;
			continue;
	      }		
	    }
	    (void)gfal_closedir (dp);
            
	    // remove directory
            if( gfal_rmdir(remote_dir_path.c_str()) == 0 )
	    {
		if(debug)
		    printf("Worker removed sucessfully, removing the path %s\n", remote_dir_path.c_str());
	        return 0;
		break;
		
	    }	
	    else if(debug)
		      printf("Worker pâth %s be removed sucessfully, trying again\n", remote_dir_path.c_str());
              		 
	}
	sleep(4);
        tries++;
    }
    return -1;
}

