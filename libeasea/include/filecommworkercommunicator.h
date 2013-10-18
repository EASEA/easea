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

#ifndef FILECOMMWORKERCOMMUNICATOR_H
#define FILECOMMWORKERCOMMUNICATOR_H

#include "commworkercomunitaror.h"
#include <stack>
#include <queue>
#include "gfal_utils.h"


/**
 * @brief Manages the grid filesystem worker communications
 * 
 */
class FileCommWorkerCommunicator : public CommWorkerCommunicator
{
  private:
  std::queue<std::string> files_to_read; // queue files received to be processed
  std::string exp_path, current_filename; // experiment path
  std::stack<std::pair<std::string,std::string> > individualt_to_send; // queue of individuals to send
  std::pair<std::string,std::string> current_item; 
  int worker_number; // worker number 
  GFAL_Utils *directory_scanner; // for find new files containing individuals
  /**
   * @brief Read  a individual from a stream
   * 
   * @param buffer stream
   * @return 0 succces -1 error
   */
  int read_file(std::string &buffer); 
  /**
   * @brief Write an invididual to the grid filesystem
   * 
   * @return 0 succces -1 error
   */
  int write_file(); 
public:
 
  /**
   * @brief Constructor
     * @param w the worker
     * @param d the data queue to receive individuals
     * @param path the worker path in the experiment grid filesystem
     * @param wn the worker number
     * @param db debugger flags
   */
  FileCommWorkerCommunicator(CommWorker *w, std::queue<std::string> *d, std::string path, int wn, int db=1):
      CommWorkerCommunicator(w,d,db),exp_path(path),worker_number(wn) { };
  ~FileCommWorkerCommunicator()  { delete directory_scanner; }  
  int receive();
  int send(char* individual, CommWorker& destination);
  int send();
  int init();
};

#endif // FILECOMMWORKERCOMMUNICATOR_H
