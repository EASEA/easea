/*
 * CRandomGenerator.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include "include/CRandomGenerator.h"
#include "include/global.h"
#include <stdlib.h>

CRandomGenerator::CRandomGenerator(unsigned int seed){
  srand(seed);
}

int CRandomGenerator::randInt(){
  return rand();
}

bool CRandomGenerator::tossCoin(){

  int rVal = rand();
  if( rVal >=(RNDMAX/2))
    return true;
  else return false;
}

bool CRandomGenerator::tossCoin(float bias){

  int rVal = rand();
  if( rVal <=(RNDMAX*bias) )
    return true;
  else return false;
}

int CRandomGenerator::randInt(int min, int max){

  int rValue = (((float)rand()/RNDMAX))*(max-min);
  //DEBUG_PRT("Int Random Value : %d",min+rValue);
  return rValue+min;

}

int CRandomGenerator::random(int min, int max){
  return randInt(min,max);
}

float CRandomGenerator::randFloat(float min, float max){
  float rValue = (((float)rand()/RNDMAX))*(max-min);
  //DEBUG_PRT("Float Random Value : %f",min+rValue);
  return rValue+min;
}

float CRandomGenerator::random(float min, float max){
  return randFloat(min,max);
}

double CRandomGenerator::random(double min, double max){
  return randFloat(min,max);
}

int CRandomGenerator::getRandomIntMax(int max){
  double r = rand();
  r = r / RNDMAX;
  r = r * max;
  return r;
}
