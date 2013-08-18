#ifndef CCOMWORKERLIST_3
#define CCOMWORKERLIST_3
#include "CComWorker.h"
#include <vector>
#include <string>
#include <set>


class CComWorkerListManager_3
{
    
  private:
      std::vector<CommWorker> activeworkers;
      std::vector<CommWorker> inactiveworkers;
      std::set<std::string> active_workers_names;
      std::string allworkers_info_remote_filename;
      std::string worker_info_path;
      int read_worker_info_file(std::string workerpath, CommWorker *&workerinfo);
      int parse_worker_info_file(const char *buffer, CommWorker *&workerinfo) const;
      int checkValidLine(char* line) const;
      int check_ipaddress(char* arg1) const;
      int check_port(char* arg1) const;
      int debug;
      bool cancel;
      bool savedonce;
      int save_worker_info_file();
      bool is_over();
      bool savefailed;
  public:
      CComWorkerListManager_3(std::string path,int _debug);
      CommWorker get_worker_nr(int wn);
      int refresh_worker_list();
      int get_nr_workers() const;
      int get_nr_inactive_workers() const;
      void terminate();
    
};  

#endif