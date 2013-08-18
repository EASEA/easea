#ifndef CCOMWORKERLIST_2
#define CCOMWORKERLIST_2
#include "CComWorker.h"
#include <vector>
#include <string>
#include <set>



class CComWorkerListManager_2
{
    
  private:
      std::vector<CommWorker> activeworkers;
      std::vector<CommWorker> inactiveworkers;
      std::string workerlist_remote_filename, workerlist_local_filename;
      std::set<std::string> active_workers_names;
      int process_workerlist_file();
      int parse_worker_info_file(const char *buffer, CommWorker *&workerinfo) const;
      int checkValidLine(char* line) const;
      int check_ipaddress(char* arg1) const;
      int check_port(char* arg1) const;
      int debug;
      bool cancel;
  public:
      CComWorkerListManager_2(std::string path,int _debug=1);
      CommWorker get_worker_nr(int wn);
      int refresh_worker_list();
      int get_nr_workers() const;
      int get_nr_inactive_workers() const;
      void terminate();
};  

#endif