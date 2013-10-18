/*
 *    Copyright (C) 2013 Waldo Cancino

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

#ifndef GFAL_UILS_H
#define GFAL_UILS_H
#include <set>
#include <string>
#include <vector>
#include <time.h>
#include <vector>

class GFAL_Utils
{
  private:
    std::string remote_dir_path;
    std::vector<std::string> current_files,delete_files,new_files;
    std::set<std::string> lastfiles_idx;
    bool firstime;
    time_t current_timestamp, previous_timestamp;

    void list_update();
    int refresh_timestamp();
    bool is_updated();
    static int debug;
  public:
    GFAL_Utils(std::string path, int db=1):remote_dir_path(path),firstime(true),current_timestamp(0),previous_timestamp(0) 
    { debug = db; };
    int dirscan(std::string start_with);
    const std::vector<std::string>& get_dirfiles() const;
    const std::vector<std::string>& get_newfiles() const;
    const std::vector<std::string>& get_deletedfiles() const;
    static int download(std::string remote_filename, std::string local_filename, time_t &timestamp, bool verbose=false);
    static int download(std::string remote_filename, std::string local_filename, bool verbose=false);
    static int upload(std::string local_filename, std::string remote_filename, bool verbose=false);
    static int get_timestamp(std::string filename, time_t &t);
    static int delete_file(std::string filename);
    static int rm_dir(std::string remote_dir_path, int ntries=3);
  
};



#endif // GFAL_UILS_H
