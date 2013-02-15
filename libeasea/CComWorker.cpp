#include "include/CComWorker.h"

CommWorker::CommWorker(std::string wname, std::string wip, short unsigned int wport):workername(wname),ip(wip),port(wport),active(true),nfails(0)
{
}  

CommWorker::CommWorker(std::string wname):workername(wname),ip("noip"),port(-1),active(true),nfails(0)
{
}  

std::string CommWorker::get_name() const 
{
    return workername;
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
