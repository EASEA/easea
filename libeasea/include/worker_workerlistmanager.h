/*
 *    Copyright (C) 2009  Waldo Cancino

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

#ifndef WORKER_WORKERLISTMANAGER_H
#define WORKER_WORKERLISTMANAGER_H

#include "abstractworkerlistmanager.h"
#include "time.h"


class Worker_WorkerListManager : public AbstractWorkerListManager
{
    private:
      std::string workerlist_remote_filename, workerlist_local_filename;
      time_t workerfile_timestamp;
      void update_lists();
      int process_workerlist_file();
      
    public:
      Worker_WorkerListManager(std::string exp_path,int _debug=1);
      virtual int refresh_worker_list();
      CommWorker get_worker_nr(int idx) const;
};

#endif // WORKER_WORKERLISTMANAGER_H
