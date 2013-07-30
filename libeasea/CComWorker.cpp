#include "include/CComWorker.h"

CommWorker::CommWorker(std::string wname, std::string hn,std::string wip, short unsigned int wport):workername(wname),hostname(hn),ip(wip),port(wport),active(true),nfails(0)
{
}  

CommWorker::CommWorker(std::string wname, std::string hn):workername(wname),hostname(hn),ip("noip"),port(-1),active(true),nfails(0)
{
}  

std::string CommWorker::get_name() const 
{
    return workername;
}

std::string CommWorker::get_hostname() const
{
    return hostname;
}

std::string CommWorker::get_ip() const
{
     return ip;
}  

short unsigned int CommWorker::get_port() const
{
    return port;
}

bool CommWorker::isactive() const
{
     return active;
}

void CommWorker::desactivate()
{
   active =false;
}   

void CommWorker::activate()
{
   active = true; 
}

void CommWorker::set_internal_ip(bool value)
{
   internal_ip = value;
}

bool CommWorker::is_internal_ip()
{
  return internal_ip;
}

unsigned long int CommWorker::get_netmask() const
{
  return mynetmask;
}

void CommWorker::set_netmask(unsigned long int nm)
{
   mynetmask = nm;
}  
