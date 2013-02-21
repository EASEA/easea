#ifndef CCOMWORKER
#define CCOMWORKER
#include <vector>
#include <string>

class CommWorker
{
  private:
    std::string workername;
    std::string ip;
    short unsigned int port;
    bool active;
    unsigned int nfails;
    bool internal_ip;
  public:
    
    CommWorker(std::string wname, std::string wip, short unsigned int wport);
    CommWorker(std::string wname);
    std::string get_name() const;
    std::string get_ip() const;
    short unsigned int get_port() const;
    bool isactive() const;
    void desactivate();
    void activate();
    void set_internal_ip(bool value);
    bool is_internal_ip();
};
#endif
