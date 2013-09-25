#include "include/CComWorker.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include <sys/types.h>                                                                                           
#include <sys/socket.h>                                                                                          
#include <arpa/inet.h>                                                                                           
#include <net/if.h>                                                                                              
#include <netinet/in.h> /* for IP Socket data types */
#include <ifaddrs.h>
#include <unistd.h>
#include <sstream>
#include <fcntl.h>
#include "include/gfal_utils.h"
#include <errno.h>
extern "C"
{
#include "gfal_api.h"
#include <lcg_util.h>
}


CommWorker::CommWorker():active(true),nfails(0) {};

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

int CommWorker::determine_ipaddress(std::string &ip, unsigned long int &nm)
{
  
  struct ifaddrs *myaddrs, *ifa;
  int status;
  
  status = getifaddrs(&myaddrs);
  
  // read external ip obtained by running a script
  char external_ip[64];
  
  FILE *file_ip = fopen("external_ip.txt","r");
  
  if(file_ip!=NULL)
  {
    //read the data
    fgets(external_ip,64,file_ip);
    external_ip[strlen(external_ip)-1] = 0;
    //if(debug)printf("external ip is: %s we will now check network interfaces\n",external_ip);
    fclose(file_ip);
  }
  else
  {
    //if(debug)printf("cannot open external ip file ...\n");
    strcpy(external_ip,"no_ip");
  }  
  
  
  
  

  if (status != 0){
    perror("No network interface, communication impossible, finishing ...");
    exit(1);
  }
  unsigned int maxlen = 0;
  for (ifa = myaddrs; ifa != NULL; ifa = ifa->ifa_next)
  {
      //if (NULL == ifa->ifa_addr)continue;
      //if ((ifa->ifa_flags & IFF_UP) == 0)continue;

      void *s4=NULL;                     
      /* ipv6 addresses have to fit in this buffer */
      char buf[64];                                  
      memset(buf,0,64);
      
      
      
      if (AF_INET == ifa->ifa_addr->sa_family)
      {
	  s4 = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
	  if (NULL == inet_ntop(AF_INET, s4, buf, sizeof(buf)))
	    printf("%s: inet_ntop failed!\n", ifa->ifa_name);
	  else {
	    // external ip address is in the network interface
	    printf("comparing %s (%d) and %s (%d) = %d\n", buf, 
		   strlen(buf),external_ip,strlen(external_ip), strcmp(buf,external_ip));
	    
	    if( strcmp(buf,external_ip) == 0) 
	    {  
		ip = buf;
		nm = ntohl(((struct sockaddr_in *)ifa->ifa_netmask)->sin_addr.s_addr);
		return 0;
	    }
	    // get an ip adress internal
	    else if(strlen(buf)> maxlen)
	    {  
	      ip = buf;
	      maxlen=strlen(buf);
	      //freeifaddrs(myaddrs);
	      //return 0;
	    }   
	  }
	}
  }
  if(myaddrs!=NULL)freeifaddrs(myaddrs);
  return -1;
}  




int CommWorker::determine_worker_name(std::string fullpath, std::string &workername, int worker_number)
{
  // scan experiment directory to find a suitable worker name
  int tries = 0;
  //char hostname[256];
  
  //gethostname(hostname, 256);
  
  //workername = hostname;
  
  
  int start = 0;  
  while(tries < 5)
  {
      std::stringstream s,t;
      if(start > 0)
      {	
	s << fullpath << "/worker_" << worker_number << '_' << start;
        t << "worker_" << worker_number << '_' << start;
      }
      else
      {
	s << fullpath << "/worker_" << worker_number;
        t << "worker_" << worker_number;
      }	
      DIR *dp;
  
     

      dp = gfal_opendir( s.str().c_str() );
      // verify if directory exist
      if(dp != NULL)
      {
		(void)gfal_closedir(dp);  
		start++;
      }
      else if(errno == ENOENT)
      {
	    int result = gfal_mkdir(s.str().c_str(), 0777);

// check error condition

	      if(result == 0)
	      {
		      //if(debug)printf("Experiment worker folder sucessfuly created, the path is %s\n", s.str().c_str());
		      result = gfal_chmod( s.str().c_str(), 0777 );
		      workername = t.str();
		      return 0;
	      }	
	      else
	      {
		      printf("Cannot create worker experiment folder %s; error reported is %d\n", s.str().c_str(), errno);
		      tries++;
	      }
      }  
      else
      {
	      tries++;
	      printf("Cannot create worker experiment folder %s; error reported is %d\n", s.str().c_str(), errno);
      }
  }
  
  return -1;
}


int CommWorker::unregister_worker( std::string fullpath, int ntries)
{
    std::string fullfilename = fullpath + "/workers_info/" + get_name() + ".txt";
    for(int i=0; i< ntries; i++)
      if( GFAL_Utils::delete_file(fullfilename) == 0 )
	  return 0;
    return -1;
}  


int CommWorker::register_worker(std::string fullpath)
{
    
    std::string fullfilename = fullpath + "/workers_info/" + get_name() + ".txt";
    
    int	fd = gfal_open( fullfilename.c_str(), O_WRONLY|O_CREAT, 0644 );
    if (fd != -1) 
    {
	 std::stringstream s;
	 s << get_name() << ':' << get_hostname() << ':';
	 if( get_ip() == "noip")s << "FILE::";
	 else s << "SOCKET:" << get_ip() << ':' << get_port();
	 
	 // now write file
	 
	 int result = gfal_write( fd, s.str().c_str(), s.str().size() );
	 
	 if(result > 0)
	 {  
	     printf("Create working information file %s \n :", fullfilename.c_str());
	     (void)gfal_close(fd);
	 }
    }
    else
    {  
          printf("failed to create worker file %s error reported is %d\n", fullfilename.c_str(),errno);
	  std::string workerdir = fullpath + '/' + get_name();
	  printf("Deleting worker directory %s\n", workerdir.c_str());
	  int result = gfal_rmdir(workerdir.c_str());
	  if(result==-1)printf("Cannot delete worker directory, error reported is %d\n", errno);
          exit(1);
          return -1;
    }  
    return 0;
}

CommWorker * CommWorker::create(std::string fullpath, int port, int wn)
{
    std::string worker_id;
    std::string hostname,ip;
    unsigned long int netmask;

   
    CommWorker *worker = NULL;
    
    char hn[256];
  
    gethostname(hn, 256);

    hostname = hn;
    
    if( determine_worker_name( fullpath, worker_id, wn ) == 0)
    {
          // second, check worker connectivity, if external IP use sockert
	  // if no external ip, use files to receive data
	  
          if(determine_ipaddress(ip,netmask) == 0)
	      worker = new CommWorker(worker_id,hostname, ip, port);
	  else worker = new CommWorker(worker_id,hostname);
    }
    return worker;
}


std::ostream & operator<<(std::ostream &os, const CommWorker& myself)
{
     os << myself.get_name() << ':' << myself.get_hostname() << ':';
	 if(myself.get_ip() == "noip")os << "FILE::";
	 else os << "SOCKET:" << myself.get_ip() << ':' << myself.get_port();
    return os;
}  