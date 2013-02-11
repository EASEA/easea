#ifndef CCOMWORKERLIST
#define CCOMWORKERLIST
#include "CComWorker.h"
#include <vector>
#include <string>



class CComWorkerListManager
{
    
  private:
      std::vector<CommWorker> activeworkers;
      std::vector<CommWorker> inactiveworkers;
      std::string workers_path;
      int read_worker_info_file(std::string workerpath, CommWorker *&workerinfo);
      int parse_worker_info_file(char *buffer, CommWorker *&workerinfo) const;
      int checkValidLine(char* line) const;
      int check_ipaddress(char* arg1) const;
      int check_port(char* arg1) const;
      bool cancel;
      int debug;
  public:
      CComWorkerListManager(int _debug=0):debug(_debug) {};
      CommWorker get_worker_nr(int wn);
      int refresh_worker_list();
      int get_nr_workers() const;
      void terminate();
};  

#endif