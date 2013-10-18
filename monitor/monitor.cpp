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


#include "experimetnmonitor.h"
#include <iostream>
#include <sstream>

using namespace std;

int main(int argc, char **argv)
{
    std::string exp_path = argv[1];
    std::string exp_name = argv[2];
    int number_workers;
    std::istringstream ( argv[3] ) >> number_workers;
    
    ExperimentMonitor runner(exp_path, exp_name, number_workers, 1);
    
    if( runner.init() == 0)
         runner.run();
    else
    {  
      cout << "Cannot init the experiments ... stop" << endl;
      return -1;
    }
    return 0;
}