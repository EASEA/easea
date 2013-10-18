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


#ifndef CCOMWORKERLIST_3
#define CCOMWORKERLIST_3
#include "CComWorker.h"
#include <vector>
#include <string>
#include <set>


class CComWorkerListManager_3
{
    
  private:
      std::vector<CommWorker> activeworkers;
      std::vector<CommWorker> inactiveworkers;
      std::set<std::string> active_workers_names;
      std::string allworkers_info_remote_filename;
      std::string worker_info_path;
      int read_worker_info_file(std::string workerpath, CommWorker *&workerinfo);
      int parse_worker_info_file(const char *buffer, CommWorker *&workerinfo) const;
      int checkValidLine(char* line) const;
      int check_ipaddress(char* arg1) const;
      int check_port(char* arg1) const;
      int debug;
      bool cancel;
      bool savedonce;
      int save_worker_info_file();
      bool is_over();
      bool savefailed;
  public:
      CComWorkerListManager_3(std::string path,int _debug);
      CommWorker get_worker_nr(int wn);
      int refresh_worker_list();
      int get_nr_workers() const;
      int get_nr_inactive_workers() const;
      void terminate();
    
};  

#endif