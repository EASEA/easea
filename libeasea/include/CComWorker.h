#ifndef CCOMWORKER
#define CCOMWORKER
#include <vector>
#include <string>

class CommWorker
{
  private:
    std::string workername;
    std::string hostname;
    std::string ip;
    short unsigned int port;
    bool active;
    unsigned int nfails;
    bool internal_ip;
    unsigned long int mynetmask;
  public:
    
    CommWorker(std::string wname, std::string hn, std::string wip, short unsigned int wport);
    CommWorker(std::string wname, std::string hn);
    std::string get_name() const;
    std::string get_hostname() const;
    std::string get_ip() const;
    short unsigned int get_port() const;
    unsigned long int get_netmask() const;
    void set_netmask(unsigned long int);
    bool isactive() const;
    void desactivate();
    void activate();
    void set_internal_ip(bool value);
    bool is_internal_ip();
};
#endif
