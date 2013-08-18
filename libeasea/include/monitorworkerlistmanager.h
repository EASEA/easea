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

#ifndef MONITORWORKERLISTMANAGER_H
#define MONITORWORKERLISTMANAGER_H

#include "abstractworkerlistmanager.h"
#include <queue>
#include "gfal_utils.h"

class MonitorWorkerListManager : public AbstractWorkerListManager
{
  private:
      std::string workerinfo_path;
      std::queue<std::string> files_to_process, failed_to_process;
      GFAL_Utils *directory_scanner;
      int chunk_size;
      int num_workers;
      bool saved_once,save_failed, modified;
      void update_lists();
      void enqueue_files();
      int read_worker_file(std::string &buffer) const;
      std::string extract_worker_name(const std::string& filename) const;
      int save_worker_file_info();
      bool cancel;
  public:
      MonitorWorkerListManager(std::string exp_path,int num_workers,int _debug);
      virtual int refresh_worker_list();
      void  process_worker_files();
      int upload_workers_info();
      bool terminated();
};

#endif // MONITORWORKERLISTMANAGER_H
