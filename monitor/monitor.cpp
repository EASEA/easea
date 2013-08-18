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