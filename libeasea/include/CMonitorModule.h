/**
 * @file CMonitorModule.hpp
 * @author Pallamidessi Joseph
 * @version 1.0
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * http://www.gnu.org/copyleft/gpl.html
**/

#ifndef AUDIOMONITORMODULE_H__
#define AUDIOMONITORMODULE_H__

#include<iostream>
#include<cstring>
#include<cstdio>
#include<cstdlib>
#include<sys/types.h>
#include<sys/socket.h>
#include<sys/wait.h>
#include<unistd.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<signal.h>
#include"CMonitorUtils.h"

class MonitorParameter;

/**
*\class  CMonitorModule 
*\brief   Communication module between the easea program and a central
*         monitoring server.
*\details The module manage a TCP/IP connection with the server and send\
*         message at each generation. The message is a MonitorParameter derived class.

*
**/
class CMonitorModule{
  
  public:
    /*Constructors/Destructors----------------------------------------------------*/
    /**
    * \brief    Constructor of CMonitorModule 
    * \details  Create the TCP/IP connection. 
    *
    * @param    serverIP  The ip of the central server, as a string
    *                     ("xxx.xxx.xxx.xxx").By default localhost ("127.0.0.1").
    * @param    port      Listening port of the server, by default 27800.
    * @param    recvMsg   Monitoring protocol policy,send message notifying a
    *                     reception of individual.
    * @param    sendMsg   Monitoring protocol policy,send message notifying a
    *                     sending of individual.
    **/
    CMonitorModule(std::string serverIP="127.0.0.1",int port=27800,
                      bool recvMsg=true,bool sendMsg=false);
    
    /**
    * \brief    Destructor 
    * \details  Close the socket
    *
    **/
    virtual ~CMonitorModule ();


    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Send the data corresponding to a generation.
    * \details  Basic and unused method, only use for debugging purpose.
    *
    *  @param   best             Value of the best individual.
    *  @param   worst            Value of the worst individual.
    *  @param   stdev            Standart deviation of the population. 
    *  @param   averageFitness   Average note of the population.
    **/
    void sendGenerationData(float best,float worst,float stdev,float averageFitness);
    
    
    /**
    * \brief    Send the parameters to the central server
    * \details  The MonitorParameter derived parameter will use its fill and 
    *           serialization method.
    **/
    void send();
    
    
    /**
    * \brief    Send the parameters refering to a reception to the central server.
    * \details  The MonitorParameter derived parameters will use its aReception
    *           method.
    *
    **/
    void receivedIndividuals();
    
    
    /**
    * \brief    Send the parameters refering to a reception to the central server.
    * \details  The MonitorParameter derived parameters will use its aSending
    *           method.
    *
    **/
    void sendingIndividuals();
    
    /*Setters---------------------------------------------------------------------*/
    /**
    * \brief    Set the migration notification policy.
    * \details  Set if a reception or a sending is notified to the central server.
    *
    *  @param   onRecv Set the reception notification policy, default true.
    *  @param   onSend Set the sending notification policy, default false.
    **/
    void setMigrationNotification(bool onRecv=true,bool onSend=false);


    /**
    * \brief    Set which parameter will be used
    * \details  The module will work with any MonitorParameter derived parameter and
    *           the central server (AdioMonitorServer) will dynamically cast the
    *           received paramater to its right type.
    *
    * @param    params The parameter to use for the communication.
    **/
    void setParams(MonitorParameter* params);
 
 
 private  :
    /*Datas-----------------------------------------------------------------------*/
    MonitorParameter* params;
    int sockfd;
    bool debug;
    bool notifyReception;
    bool notifySending;
};

#endif /* end of include guard: AUDIOMONITORMODULE_H__ */


