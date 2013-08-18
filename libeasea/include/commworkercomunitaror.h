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

#ifndef COMMWORKERCOMUNITAROR_H
#define COMMWORKERCOMUNITAROR_H
#include "CComWorker.h"
#include <queue>

class CommWorkerCommunicator
{
  protected:
     CommWorker &myself;
     std::queue<std::string> &data;
     static int debug;
     static bool cancel;
  public:
     CommWorkerCommunicator(CommWorker &w, std::queue<std::string> &d, int db=1):myself(w),data(d) { debug = db; cancel=false;  };
     static void terminate() { cancel = true; };
     virtual int init() = 0;
     virtual int send(char *individual, CommWorker &destination) = 0;
     virtual int receive() = 0;
};

#endif // COMMWORKERCOMUNITAROR_H
