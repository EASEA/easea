#ifndef CCOMWORKER
#define CCOMWORKER
#include <vector>
#include <string>
#include <iostream>

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
    void change_port(short unsigned int p) { port = p; }
    void set_name(std::string name) { workername = name; }
    unsigned long int get_netmask() const;
    void set_netmask(unsigned long int);
    bool isactive() const;
    void desactivate();
    void activate();
    void set_internal_ip(bool value);
    bool is_internal_ip();
    static CommWorker* parse_worker_string(std::string s) { return CommWorker::parse_worker_string(s.c_str()); };
    static CommWorker* parse_worker_string(const char *buffer);
    static int check_ipaddress(char *ipaddress);
    static int check_port(char *port);
    
    friend std::ostream & operator<<(std::ostream &os, const CommWorker& myself);
};
#endif
