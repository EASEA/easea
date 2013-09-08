#include "include/CComWorker.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"


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


CommWorker* CommWorker::parse_worker_string(const char *buffer) 
{
 
    char* holder = (char*)malloc(strlen(buffer)+1);
    strcpy(holder,buffer);
    char *foldername = strtok(holder, ":");
    char *hostname = strtok(NULL, ":");
    char *mode = strtok(NULL, ":");
    CommWorker *workerinfo;
    
    
    
    // check first valid hostanme and protocol
    if(foldername == NULL || hostname == NULL || mode ==NULL)
    {  
        return NULL;
    }
    
    
    // protocol = file so no IP, NO port, may be an external node
    if( strcmp(mode,"FILE") == 0 )
    {
        std::string fn = foldername;
        std::string hn = hostname;
	workerinfo = new CommWorker(fn,hn);
    }
    
    else if( strcmp(mode,"SOCKET") == 0 || strcmp(mode,"MPI") ==0) 
    {  
        // now check for ip and port/rank
	char* address = strtok(NULL, ":");
	char* port = strtok(NULL,":");
	
	if(CommWorker::check_ipaddress(address) == 0 && CommWorker::check_port(port) == 0) 
	{  
          std::string fn = foldername;
	  std::string hn = hostname;
	  std::string addr = address;
	  workerinfo = new CommWorker(fn, hn,addr, atoi(port) );	  
	}  
        else return NULL;
    }

    free(holder);
    return workerinfo;
} 


int CommWorker::check_port(char *port)
{
    int nibble = atoi(port);
    if(nibble<0){
	  return -1;
    }
    std::string s = port;
    for(unsigned int i=0; i<s.length(); i++){
        if(!isdigit(s[i])){
	      return -1;
        }
    }
    return 0;
 }  


int CommWorker::check_ipaddress(char *ipaddress) 
{
    char* holder = (char*)malloc(strlen(ipaddress)+1);
    strcpy(holder,ipaddress);

   char* byte = strtok(holder,".");
    int nibble = 0, octets = 0, flag = 0;
    while(byte != NULL){
        octets++;
        if(octets>4){
	    free(holder);
            return -1;
        }
        nibble = atoi(byte);
        if((nibble<0)||(nibble>255)){
	    free(holder);
            return -1;
        }
        std::string s = byte;
        for(unsigned int i=0; i<s.length(); i++){
            if(!isdigit(s[i])){
	      free(holder);
	      return -1;
            }
        }
        byte = strtok(NULL,".");
    }
    if(flag || octets<4){
            free(holder);
            return -1;
    }
    return 0;
}


std::ostream & operator<<(std::ostream &os, const CommWorker& myself)
{
     os << myself.get_name() << ':' << myself.get_hostname() << ':';
	 if(myself.get_ip() == "noip")os << "FILE::";
	 else os << "SOCKET:" << myself.get_ip() << ':' << myself.get_port();
    return os;
}  