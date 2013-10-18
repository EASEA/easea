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

/**
 * @brief This class encapsulates some GFAL functions for grid filesystem access
 * 
 */
class GFAL_Utils
{
  private:
    std::string remote_dir_path;
    std::vector<std::string> current_files,delete_files,new_files; // for directory scanner, current files in the directory, deleted files and new files lists
    std::set<std::string> lastfiles_idx; // index for the names of the last current files
    bool firstime;
    time_t current_timestamp, previous_timestamp; // current and previous directory timestamp

    /**
     * @brief Updata list for current files, deleted files and new files
     * 
     */
    void list_update(); // update list
    /**
     * @brief Update directory timestamp
     * 
     * @return int
     */
    int refresh_timestamp(); 
    /**
     * @brief true if directory has not changed since last checking
     */
    bool is_updated(); 
    static int debug; // debug flag
  public:
    /**
     * @brief Constructor
     * @param path, directory to be scanned
     * @param db debug flag
     * 
     */
    GFAL_Utils(std::string path, int db=1):remote_dir_path(path),firstime(true),current_timestamp(0),previous_timestamp(0) 
    { debug = db; };
    /**
     * @brief scan directory looking for new and delete files
     * 
     * @param start_with filename should start with this string
     * @return 0 succces -1 error
     */
    int dirscan(std::string start_with);
    /**
     * @brief return current filenames in directory
     */
    const std::vector<std::string>& get_dirfiles() const;
    /**
     * @brief return new files in directory since the last scan
     */
    const std::vector<std::string>& get_newfiles() const;
    /**
     * @brief return deleted files in directory since the last scan
     * 
     */
    const std::vector<std::string>& get_deletedfiles() const;
    /**
     * @brief Download a file from the grid filesystem
     * 
     * @param remote_filename remote filename to download
     * @param local_filename local filename downloaded
     * @param timestamp it will download the file if it is newer than timestamp
     * @param verbose verbose output (GFAL messages)
     * @return 0 succces -1 error
     */
    static int download(std::string remote_filename, std::string local_filename, time_t &timestamp, bool verbose=false);
    /**
     * @brief Same as above without file timestamp check
     * 
     */
    static int download(std::string remote_filename, std::string local_filename, bool verbose=false);
    /**
     * @brief Upload a file to the grid filesystem
     * 
     * @param local_filename file to be uploaded
     * @param remote_filename destination grid filesystem filename
     * @param verbose verbose output
     * @return 0 succces -1 error
     */
    static int upload(std::string local_filename, std::string remote_filename, bool verbose=false);
    /**
     * @brief get the timestamp of a remote file in the grid filesystem
     * 
     * @param filename remote filename
     * @param t timestamp
     * @return 0 succces -1 error
     */
    static int get_timestamp(std::string filename, time_t &t);
    /**
     * @brief Delete a remote file in the grid filesystem
     * 
     * @param filename remote filename
     * @return 0 succces -1 error
     */
    static int delete_file(std::string filename);
    /**
     * @brief Remove a directory in the grid filesystem
     * 
     * @param remote_dir_path remote path
     * @param ntries number of tries
     * @return 0 succces -1 error
     */
    static int rm_dir(std::string remote_dir_path, int ntries=3);
  
};



#endif // GFAL_UILS_H
