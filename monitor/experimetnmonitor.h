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

#ifndef EXPERIMETNMONITOR_H
#define EXPERIMETNMONITOR_H
#include <string>
#include "../libeasea/include/monitorworkerlistmanager.h"

class ExperimentMonitor
{
  private:
      std::string expname,exppath;
      int nworkers;
      MonitorWorkerListManager *ListWorkersMonitor;
      int debug;
  public:
      ExperimentMonitor(  std::string _exppath, std::string _expname, int nw, int db=1):expname(_expname),exppath(_exppath),nworkers(nw),debug(db) {};
      int init();
      int run();
      
};

#endif // EXPERIMETNMONITOR_H
