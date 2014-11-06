#include "include/ClientMonitorParameter.h"

/*ClientMonitorParameter---------------------------------------------*/
ClientMonitorParameter::ClientMonitorParameter(CEvolutionaryAlgorithm* parent):
                        MonitorParameter(),source(parent){
  strType=SIMPLEDATA;
}


ClientMonitorParameter::~ClientMonitorParameter(){
}


void ClientMonitorParameter::fill(){
  
  if(source!=NULL){
    dataFlag=true;
    migration=false;
    best=source->population->Best->getFitness();
    worst=source->population->Worst->getFitness();
    stdev=source->cstats->currentStdDev;
    average=source->cstats->currentAverageFitness;
  }
}

void ClientMonitorParameter::processBuffer(std::string line){
  float unused;
  
  dataFlag=true;
  timeBegin=timeEnd;
  std::stringstream ss(line);
  ss>>unused;
  ss>>timeEnd;
  ss>>unused;
  ss>>best;
  ss>>average;
  ss>>stdev;
  ss>>worst;
}


void ClientMonitorParameter::sending(){
  dataFlag=false;
  migration=true;
  recv=false;
  send=true;
}

void ClientMonitorParameter::reception(){
  dataFlag=false;
  migration=true;
  recv=true;
  send=false;
}
/*HORRIBLE*/
char* ClientMonitorParameter::serialize(){
  char* serial=new char[sizeof(unsigned char)+4*sizeof(float)+4*sizeof(bool)];
  void* next;
  next=mempcpy(serial,&strType,sizeof(unsigned char));
  
  next=mempcpy(next,&dataFlag,sizeof(bool));
  next=mempcpy(next,&migration,sizeof(bool));
  next=mempcpy(next,&recv,sizeof(bool));
  next=mempcpy(next,&send,sizeof(bool));
  
  next=mempcpy(next,&best,sizeof(float));
  next=mempcpy(next,&worst,sizeof(float));
  next=mempcpy(next,&stdev,sizeof(float));
  next=mempcpy(next,&average,sizeof(float));

  return serial;
}

int ClientMonitorParameter::serialSize(){
  return sizeof(unsigned char)+4*sizeof(float)+4*sizeof(bool);
}

void ClientMonitorParameter::deserialize(char* data){
  int offset=0;
  memcpy(&strType,data,sizeof(unsigned char));
  
  offset+=sizeof(bool);
  memcpy(&dataFlag,data+offset,sizeof(bool));
  offset+=sizeof(bool);
  memcpy(&migration,data+offset,sizeof(bool));
  offset+=sizeof(bool);
  memcpy(&recv,data+offset,sizeof(bool));
  offset+=sizeof(bool);
  memcpy(&send,data+offset,sizeof(bool));
  offset+=sizeof(bool);


  memcpy(&best,data+offset,sizeof(float));
  offset+=sizeof(float);
  memcpy(&worst,data+offset,sizeof(float));
  offset+=sizeof(float);
  memcpy(&stdev,data+offset,sizeof(float));
  offset+=sizeof(float);
  memcpy(&average,data+offset,sizeof(float));
  offset+=sizeof(float);
}

size_t ClientMonitorParameter::size(){
  return sizeof(this);
}

int ClientMonitorParameter::getTime(){
  return MonitorParameter::getTime();
}
