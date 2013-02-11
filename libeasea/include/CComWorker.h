#ifndef CCOMWORKER
#define CCOMWORKER
#include <vector>
#include <string>

class CommWorker
{
  private:
    std::string workername;
    std::string ip;
    unsigned int port;
    bool active;
    unsigned int nfails;
  public:
    
    CommWorker(std::string wname, std::string wip, int wport);
    CommWorker(std::string wname);
    std::string get_name() const;
    std::string get_ip() const;
    int get_port() const;
    bool isactive() const;
    void desactivate();
    void activate();
};
#endif
