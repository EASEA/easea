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

#ifndef MONITORWORKERLISTMANAGER_H
#define MONITORWORKERLISTMANAGER_H

#include "abstractworkerlistmanager.h"
#include <queue>
#include "gfal_utils.h"

/**
 * @brief This class implements the Monitor experiment worker list manager,
 * that scans the grid filesystem experiment path to find workers and write
 * all the information in a single file that will be read for each worker
 * in the experiment
 * 
 */
class MonitorWorkerListManager : public AbstractWorkerListManager
{
  private:
      std::string workerinfo_path; // remote grid filesystem path
      std::queue<std::string> files_to_process, failed_to_process; //files containing information about workers
      GFAL_Utils *directory_scanner; // directory scanner to find new files
      int chunk_size; // number of files processed at each iteration
      int num_workers; // number of workers found
      bool saved_once,save_failed, modified; // control flags
      /**
       * @brief update list of active and inactive workers
       */
      void update_lists();
      /**
       * @brief enqueue worker information files to be processed
       */
      void enqueue_files();
      /**
       * @brief read a worker file from a stream
       * 
       * @param buffer stream
       * @return 0 succces -1 error
       */
      
      int read_worker_file(std::string &buffer) const;
      /**
       * @brief extract worker name from the worker information filename
       * 
       * @param filename worker information filename
       * @return the worker name
       */
      std::string extract_worker_name(const std::string& filename) const;
      /**
       * @brief save the list of workers found in the grid filesystem
       * 
       * @return 0 succces -1 error
       */
      int save_worker_file_info();
      bool cancel; // cancel flag
  public:
    /**
     * @brief Constructor
     * 
     * @param exp_path grid filesystem experiment path
     * @param num_workers number of workers
     * @param _debug debug flag
     */
    MonitorWorkerListManager(std::string exp_path,int num_workers,int _debug);
    /**
     * @brief refresh the worker list, updating the internal list
     * 
     * @return 0 succces -1 error
     */
    virtual int refresh_worker_list();
    /**
     * @brief read the worker information files in the queue
     */
    void  process_worker_files();
    int upload_workers_info();
    /**
     * @brief returns true if the scanner process is cancelled
     * 
     * @return bool
     */
    bool terminated();
};

#endif // MONITORWORKERLISTMANAGER_H
