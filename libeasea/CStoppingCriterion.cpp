/*
 * CStoppingCriterion.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include "include/CStoppingCriterion.h"
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include "include/CEvolutionaryAlgorithm.h"

/* ****************************************
   GenerationalCriterion class
****************************************/
CGenerationalCriterion::CGenerationalCriterion(unsigned generationalLimit){
  this->generationalLimit = generationalLimit;
}

void CGenerationalCriterion::setCounterEa(unsigned* ea_counter){
	this->currentGenerationPtr = ea_counter;
}

bool CGenerationalCriterion::reached(){
  if( generationalLimit <= *currentGenerationPtr ){
    std::cout << "Current generation " << *currentGenerationPtr << " Generational limit : " <<
      generationalLimit << std::endl;
    return true;
  }
  else return false;
}

unsigned* CGenerationalCriterion::getGenerationalLimit(){
	return &(this->generationalLimit);
}

/* ****************************************
   TimeCriterion class
****************************************/
CTimeCriterion::CTimeCriterion(unsigned timeLimit){
  this->timeLimit = timeLimit;
  this->elapsedTime = 0.0;
}

bool CTimeCriterion::reached(){
  if(timeLimit>0){
  	//gettimeofday(&(this->end),0);
  	//timersub(&(this->end),&(this->begin), &(this->res));
  	//if((unsigned)res.tv_sec>timeLimit-1){
  	if((unsigned)elapsedTime>timeLimit-1){
  	  std::cout << "Time Over" << std::endl;
	  std::cout << "Time Limit was " << timeLimit << " seconds" << std::endl;
  	  return true;
  	}
  	else return false;
  }
  else return false;
}

void CTimeCriterion::setElapsedTime(double elapsedTime){
	this->elapsedTime = elapsedTime;
}

double CTimeCriterion::getElapsedTime(){
	return this->elapsedTime;
}

/* ****************************************
   CtrlCStopingCriterion class
****************************************/
bool ARRET_DEMANDE;

CControlCStopingCriterion::CControlCStopingCriterion(){
  signal( SIGINT, signal_handler );
#ifdef WIN32
  signal( SIGTERM, signal_handler );
#else
  signal( SIGQUIT, signal_handler );
#endif
  ARRET_DEMANDE=false;
}

bool CControlCStopingCriterion::reached(){
  if(ARRET_DEMANDE)
	std::cout << "Algorithm stopped on user demand" << std::endl; 
  return ARRET_DEMANDE;
}

void signal_handler(int sig){
  signal(SIGINT, SIG_DFL);
#ifdef WIN32
  signal(SIGTERM, SIG_DFL);
#else
  signal(SIGQUIT, SIG_DFL);
#endif
  printf("Ctrl C entered ... closing down\nNext Ctrl C will kill the Program !!!\n");
  ARRET_DEMANDE=true;
}


