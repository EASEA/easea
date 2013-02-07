#ifndef CCOMWORKERLIST
#define CCOMWORKERLIST
#include "CComWorker.h"
#include <vector>



class CComWorkerListManager
{
    
  private:
      std::vector<CommWorker> activeworkers;
      std::vector<CommWorker> inactiveworkers;
      std::string workers_path;
      int read_worker_info_file();
      int& parse_worker_info_file(char *buffer, CommWorker &arg2);
      int checkValidLine(char* line);
      bool cancel;
  public:
      CComWorker();
      CommWorker get_worker_nr(int wn);
      int refresh_worker_list();
      int get_nr_workers() const;
      void cancel() const;
};  

#endif