/**
 * @file CMonitorModule.cpp
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

#include "include/CMonitorModule.h"

CMonitorModule::CMonitorModule(std::string serverIP,int port,bool recvMsg,bool sendMsg):
                    notifyReception(recvMsg),notifySending(sendMsg){
  struct sockaddr_in server;
  socklen_t addrlen;

  debug=true;

  /* socket factory*/
  if((sockfd = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP)) == -1)
  {
    perror("socket");
    exit(EXIT_FAILURE);
  }

  /* init remote addr structure and other params*/
  server.sin_family = AF_INET;
  server.sin_port   = htons(port);
  addrlen           = sizeof(struct sockaddr_in);

  /* convert serverIP to correct addr*/
  if(inet_pton(AF_INET,serverIP.c_str(),&server.sin_addr.s_addr) != 1)
  {
    perror("inet_pton");
    close(sockfd);
    exit(EXIT_FAILURE);
  }

  if (debug) {
    printf("Trying to connect to the remote host\n");
  }

  /* enable the TCP connection between this client and the AudioMonitorServer*/
  if(connect(sockfd,(struct sockaddr*)&server,addrlen) == -1)
  {
    perror("connect");
    exit(EXIT_FAILURE);
  }

  if (debug) {
    printf("Connection OK\n");
  }
  char buf[6]="Hello";
  /*Hello the server*/
  ::send(sockfd,buf,6,0);
}


CMonitorModule::~CMonitorModule(){
  close(sockfd);
}


void CMonitorModule::sendGenerationData(float best,float worst,float stdev,float averageFitness){
  float serial[4];

  serial[0]=best;
  serial[1]=worst;
  serial[2]=stdev;
  serial[3]=averageFitness;

  ::send(sockfd,serial,sizeof(float)*4,0);
}


void CMonitorModule::send(){
  params->fill(); 
  if (params->isData()) {
    ::send(sockfd,(void*)params->serialize(),params->serialSize(),0);
  }
}


void CMonitorModule::receivedIndividuals(){
  if (notifyReception) {
    params->reception();
    ::send(sockfd,(void*)params->serialize(),params->serialSize(),0);
  }
}


void CMonitorModule::sendingIndividuals(){

  if (notifySending) {
    params->sending();
    ::send(sockfd,(void*)params->serialize(),params->serialSize(),0);
  }
}


void CMonitorModule::setMigrationNotification(bool onRecvPolicy,bool onSendPolicy){
  notifyReception=onRecvPolicy;  
  notifySending=onSendPolicy;  
}


void CMonitorModule::setParams(MonitorParameter* params){
  this->params=params;
}
