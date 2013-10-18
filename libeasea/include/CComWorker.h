/*
 *    Copyright (C) 2013  Waldo Cancino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
 

#ifndef CCOMWORKER
#define CCOMWORKER
#include <vector>
#include <string>
#include <iostream>

/**
 * @brief This class stores information concerning worker for grid experiments
 * 
 */
class CommWorker
{
  private:
    std::string workername; // name of the worker
    std::string hostname; // hostname
    std::string ip; // ip address
    short unsigned int port; // port for listening incoming messages
    bool active; // is worker active ?
    unsigned int nfails; // number of fails (not used yet)
    bool internal_ip; // is ip internal or external
    unsigned long int mynetmask; // netmask
    
    /**
     * @brief determine the ip adress of the worker
     * 
     * @param ip ip number
     * @param nm netmask
     * @return 0 sucess -1 error
     */
    static int determine_ipaddress(std::string &ip, unsigned long int &nm);
    /**
     * @brief init the listen port 
     * 
     * @param port port number
     * @return 0 sucess -1 error
     */
    static int init_network(short unsigned int &port);
    /**
     * @brief determine the worker name 
     * 
     * @param fullpath experiment grid path
     * @param workername the name of the worker
     * @param worker_number the number of worker provided
     * @return 0 sucess -1 error
     */
    static int determine_worker_name(std::string fullpath, std::string& workername, int worker_number);
    
    
  public:
    /**
     * @brief Default constructor
     * 
     */
    CommWorker();
    /**
     * @brief Create a worker object using the provided parameters
     * 
     * @param wname worker name
     * @param hn hostname
     * @param wip ip address
     * @param wport port
     */
    CommWorker(std::string wname, std::string hn, std::string wip, short unsigned int wport);
    /**
     * @brief Create a worker object using the provided parameters
     * 
     * @param wname worker name
     * @param hn hostname
     */
    CommWorker(std::string wname, std::string hn);
    /**
     * @brief get worker name
     */
    std::string get_name() const;
    /**
     * @brief get worker's host name
     */
    std::string get_hostname() const;
    /**
     * @brief get worker ip address
     */
    std::string get_ip() const;
    /**
     * @brief get worker port number for incoming message
     */
    short unsigned int get_port() const;
    /**
     * @brief change the worker listen port
     */
    void change_port(short unsigned int p) { port = p; }
    /**
     * @brief set the worker name
     */
    void set_name(std::string name) { workername = name; }
    /**
     * @brief set worker hostname
     */
    void set_hostname(std::string hn) { hostname = hn; }
    /**
     * @brief get worker netmask
     */
    unsigned long int get_netmask() const;
    /**
     * @brief set worker netmask
     * 
     * @param  netmask
     */
    void set_netmask(unsigned long int);
    /**
     * @brief return if is running
     * 
     * @return true or false
     */
    bool isactive() const;
    /**
     * @brief manually deactivate the worker
     */
    void desactivate();
    /**
     * @brief manually activate the worker
     * 
     */
    void activate();
    /**
     * @brief set internal ip status
     * 
     * @param true worker has internal ip, false worker has external ip
     */
    void set_internal_ip(bool value);
    /**
     * @brief Change worker ip adress
     * 
     */
    void set_ip( std::string _ip ) { ip = _ip; }
    /**
     * @brief return true worker has internal ip, false worker has external ip
     */
    bool is_internal_ip();
    /**
     * @brief Create an worker object using the provided parameters
     * 
     * @param fullpath experiment grid worker path
     * @param port listen port for incoming connections
     * @param wn worker name
     * @return CommWorker* pointer to worker object
     */
    static CommWorker *create(std::string fullpath, int port, int wn);
    /**
     * @brief Create a worker object from a stream (string)
     * 
     * @return CommWorker* pointer to worker object
     */
    static CommWorker* parse_worker_string(std::string s) { return CommWorker::parse_worker_string(s.c_str()); };
    /**
     * @brief Same as above, just using a array of chars as input
     * 
     * @return CommWorker* pointer to worker object
     */
    static CommWorker* parse_worker_string(const char *buffer);
    /**
     * @brief check if the ip address argument is a valid ip
     * 
     * @param ipaddress ip address
     * @return 0 success -1 error
     */
    static int check_ipaddress(char *ipaddress);
    /**
     * @brief check if the port number provided is a valid port
     * 
     * @param port port number
     * @return 0 success -1 error
     */
    static int check_port(char *port);
    /**
     * @brief Register worker in the grid filesystem
     * 
     * @param  path of the grid filesystem
     * @return 0 success -1 error
     */
    int register_worker(std::string);
    /**
     * @brief Unregister worker from the grid filesystem
     * 
     * @param filename worker filename
     * @param ntries delete tries
     * @return int 0 success -1 error
     */
    int unregister_worker(std::string filename, int ntries=3);
    /**
     * @brief Worker stream
     * 
     */
    friend std::ostream & operator<<(std::ostream &os, const CommWorker& myself);
};
#endif
