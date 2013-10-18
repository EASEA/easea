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

#ifndef ABSTRACTWORKERLISTMANAGER_H
#define ABSTRACTWORKERLISTMANAGER_H
#include "CComWorker.h"
#include <vector>
#include <string>
#include <set>

/**
 * @brief Abstract base class for worker list management of worker objets
 * 
 */
class AbstractWorkerListManager
{
    protected:
      std::vector<CommWorker> activeworkers; // list active workers
      std::vector<CommWorker> inactiveworkers; // lis inactive (possibly finished) workers
      std::set<std::string> workernames_idx; // index of worker names
      static int debug; // debug flag
      static int cancel; // cancel operations
    public:
      /**
       * @brief Default constructor
       * 
       */
      AbstractWorkerListManager(int _debug) { debug = _debug; }
      virtual ~AbstractWorkerListManager() { }
      /**
       * @brief Refresh worker list 
       * 
       * @return 0 on sucess -1 on error
       */
      virtual int refresh_worker_list() = 0;
      /**
       * @brief get the number of active workers
       */
      inline int get_nr_workers() const {  return activeworkers.size(); } ;
      /**
       * @brief get the number of inactive workers
       */
      inline int get_nr_inactive_workers() const {  return inactiveworkers.size(); } ;
      /**
       * @brief return an active worker
       * 
       * @param wn index of active worker
       */
      
      CommWorker get_worker_nr(int wn);
      /**
       * @brief terminate the wor scan (for multithread scan)
       * 
       * @return void
       */
      static void terminate() { cancel = true; };
};


#endif // ABSTRACTWORKERLISTMANAGER_H
