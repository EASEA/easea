#include "include/CComGridUdpServer.h"
#include <sys/types.h>                                                                                           
#include <sys/socket.h>                                                                                          
#include <arpa/inet.h>                                                                                           
#include <net/if.h>                                                                                              

#include <errno.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <stdio.h>     
#include <stdlib.h>    
#include <string.h>    
#include "gfal_api.h"

int CComGridUDPServer::get_ipaddress(std::string &ip)
{
  
  struct ifaddrs *myaddrs, *ifa;
 int status;

  status = getifaddrs(&myaddrs);

  if (status != 0){
    perror("No network interface, communication impossible, finishing ...");
    exit(1);
  }

  for (ifa = myaddrs; ifa != NULL; ifa = ifa->ifa_next)
  {
      if (NULL == ifa->ifa_addr)continue;
      if ((ifa->ifa_flags & IFF_UP) == 0)continue;

      struct sockaddr_in *s4;                     
      /* ipv6 addresses have to fit in this buffer */
      char buf[64];                                  

      if (AF_INET == ifa->ifa_addr->sa_family)
      {
	  s4 = (struct sockaddr_in *)(ifa->ifa_addr);
	  if (NULL == inet_ntop(ifa->ifa_addr->sa_family, (void *)&(s4->sin_addr), buf, sizeof(buf)))
	    printf("%s: inet_ntop failed!\n", ifa->ifa_name);
	  else {
	    // external ip address
	    if(strlen(buf)>=13)
	    {  
	      ip = buf;
		freeifaddrs(myaddrs);
	      return 0;
	    }   
	  }
	}
  }
  freeifaddrs(myaddrs);
  return -1;
}  
  
  
int CComGridUDPServer::create_exp_path(char *path, char *expname)
{
    std::string exp(expname);
    std::string pathname(path);
    fullpath = pathname + expname;
    
    //cancel = 0;
    debug = dbg;
 
    // create the main directory
    
    int result = gfal_mkdir(fullpath.c_str(),0777);
    
    // check error condition
    printf("Trying to determine or create directory experiment\n");
    if(result<0 && errno!= EEXIST)
    {
        printf("Cannot create experiment folder %s; check user permissions or disk space", fullpath.c_str());
        printf("Result of gfal_mkdir = %d %d\n" ,result,errno);   
	exit(1);
    }
    
    result = gfal_chmod(fullpath.c_str(), 0777);
    return 0;
}

int CComGridUDPServer::determine_worker_name(std::string &workername)
{
  
  // scan experiment directory to find a suitable worker name
  int tries = 0;
  char hostname[256];
  
  gethostname(hostname, 256);
  
  workername = hostname;
  
  
  int start = 0;  
  while(tries < 5)
  {
      std::stringstream s,t;
      if(tries > 0)
      {	
	s << fullpath << "/worker_" << workername << '_' << start;
        t << "worker_" << workername << '_' << start;
      }
      else
      {
	s << fullpath << "/worker_" << workername;
        t << "worker_" << workername;
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
		      if(debug)printf("Experiment worker folder sucessfuly created, the path is %s\n", s.str().c_str());
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

// create and register worker
int CComGridUDPServer::register()
{
    
    std::string fullfilename = fullname + '/' + myself->getname() + '/' + "worker_info.txt";
    
    int	fd = gfal_open( fullfilename.c_str(), O_CREAT | O_WRONLY, 0777);
    if (fd != -1) 
    {
	 std::stringstream s;
	 s << myself->getname() + ':';
	 if(myself->get_ip() == "NOIP")s << "FILE::";
	 else s << "SOCKET:" + myself->get_ip() + ':' + myself->get_port();
	 
	 // now write file
	 
	 int result = gfal_write( fd, s.str().c_str(), s.str().size() );
	 
	 if(result > 0)
	 {  
	     printf("Create working information file %s \n :", fullfilename.c_str());
	     (void)gfal_close(fd);
	     return 0;
	     
	 }
    }
    return -1;
}  