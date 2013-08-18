/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

extern "C" {
#include "gfal_api.h"
#include "lcg_util.h"
}
#include "include/gfal_utils.h"
#include <list>
#include <cstdlib>
#include <iostream>
using namespace std;

int GFAL_Utils::debug=0;

int GFAL_Utils::dirscan(string start_with="")
{
  DIR *dp;
  struct dirent *ep;
  vector<string> dirfiles;
  
  if( refresh_timestamp()==0 && !is_updated() )
  {
 
      if(debug) cout << "Reading directory ..." << remote_dir_path << endl;
      dp = gfal_opendir ( remote_dir_path.c_str() );
    
    if (dp != NULL)
    {
	while ( (ep = gfal_readdir (dp)) )
	{
		//only take into account folders
		string filename(ep->d_name);

		unsigned int pos=filename.find(start_with);
		
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
   delete_files.clear();
   set<string> newindex;
   for(unsigned int i=0; i<current_files.size(); i++)
   {
      newindex.insert(current_files[i]);
      if( lastfiles_idx.find( current_files[i] ) == lastfiles_idx.end() )
	 new_files.push_back( current_files[i] );
      else
	 lastfiles_idx.erase( current_files[i] );
   }  
   
   set<string>::iterator it, it_e;
   
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
