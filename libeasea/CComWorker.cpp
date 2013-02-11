#include "include/CComWorker.h"

CommWorker::CommWorker(std::string wname, std::string wip, int wport):workername(wname),ip(wip),port(wport),active(true),nfails(0)
{
}  

CommWorker::CommWorker(std::string wname):workername(wname),ip("noip"),port(-1),active(true),nfails(0)
{
}  

inline std::string CommWorker::get_name() const 
{
    return workername;
}

inline std::string CommWorker::get_ip() const
{
     return ip;
}  

inline int CommWorker::get_port() const
{
    return port;
}

inline bool CommWorker::isactive() const
{
     return active;
}

inline void CommWorker::desactivate()
{
   active =false;
}   

inline void CommWorker::activate()
{
   active = true; 
}