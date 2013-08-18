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

#ifndef FILECOMMWORKERCOMMUNICATOR_H
#define FILECOMMWORKERCOMMUNICATOR_H

#include "commworkercomunitaror.h"
#include <stack>
#include <queue>
#include "gfal_utils.h"


class FileCommWorkerCommunicator : public CommWorkerCommunicator
{
  private:
  std::queue<std::string> files_to_read;
  std::string exp_path, current_filename;
  std::stack<std::pair<std::string,std::string> > individualt_to_send;
  std::pair<std::string,std::string> current_item;
  int worker_number;
  GFAL_Utils *directory_scanner;
  int read_file(std::string &buffer);
  int write_file();
public:
  FileCommWorkerCommunicator(CommWorker &w, std::queue<std::string> &d, std::string path, int wn, int db=1):
      CommWorkerCommunicator(w,data,db),exp_path(path),worker_number(wn) {};
  int receive();
  int send(char* individual, CommWorker& destination);
  int send();
  int init();
};

#endif // FILECOMMWORKERCOMMUNICATOR_H
