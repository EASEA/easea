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

#ifndef COMMWORKERCOMUNITAROR_H
#define COMMWORKERCOMUNITAROR_H
#include "CComWorker.h"
#include <queue>

/**
 * @brief Base class for the worker file and network communication
 * 
 */
class CommWorkerCommunicator
{
  protected:
     CommWorker *myself; // worker
     std::queue<std::string> *data; // data queue received
     static int debug; // debug flag
     static bool cancel; // cancel communication flag
  public:
    /**
     * @brief Constructor
     * 
     * @param w the worker
     * @param d the data queue to receive individuals
     * @param db debugger flags
     * 
     */
    CommWorkerCommunicator(CommWorker *w, std::queue<std::string> *d, int db=1):myself(w),data(d) { debug = db; cancel=false;  };
    /**
     * @brief Terminato communications
     */
    static void terminate() { cancel = true; };
     virtual ~CommWorkerCommunicator() {};
     /**
      * @brief Init the communicator
      */
     virtual int init() = 0;
     /**
      * @brief Send an individual to a worker
      * 
      * @param individual individual string
      * @param destination destination worker
      * @return 0 succces -1 error
      */
     virtual int send(char *individual, CommWorker &destination) = 0;
     /**
      * @brief Receive and store an individual
      * 
      * @return 0 succces -1 error
      */
     virtual int receive() = 0;
};

#endif // COMMWORKERCOMUNITAROR_H
