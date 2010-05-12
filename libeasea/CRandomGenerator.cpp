/*
 * CRandomGenerator.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include "include/CRandomGenerator.h"
#include "include/global.h"
//#include <stdlib.h>
#include <stdio.h>
#include <math.h>

CRandomGenerator::CRandomGenerator(unsigned int seed){
	this->seed = seed;
	this->mt_rnd = new MTRand(seed);
	//srand(seed);
}

CRandomGenerator::~CRandomGenerator(){
	delete this->mt_rnd;
}

int CRandomGenerator::randInt(){
  //return rand();
	return mt_rnd->randInt();
}

bool CRandomGenerator::tossCoin(){
  int rVal = mt_rnd->randInt(1);
  return (rVal==1?true:false);
}

bool CRandomGenerator::tossCoin(float bias){

  double rVal = mt_rnd->rand(1.);
  if( rVal <=bias )
    return true;
  else return false;
}

int CRandomGenerator::randInt(int min, int max){
	max--; // exclude upper bound
	return min+mt_rnd->randInt(max-min);
}

int CRandomGenerator::random(int min, int max){
  return this->randInt(min,max);
}

float CRandomGenerator::randFloat(float min, float max){
	return min+mt_rnd->randExc(max-min);
}

float CRandomGenerator::random(float min, float max){
  return this->randFloat(min,max);
}

double CRandomGenerator::random(double min, double max){
  return this->randFloat(min,max);
}

int CRandomGenerator::getRandomIntMax(int max){
  return this->randInt(0,max);
}


/**
   Box-Muller method for gaussian random distribution

   Not sure, this function is really working.
 */
void CRandomGenerator::random_gauss(float mean, float std_dev, float* z_0, float* z_1){
 float x1, x2, w;
 
 do {
   x1 = 2.0 * this->random(0.,1.) - 1.0;
   x2 = 2.0 * this->random(0.,1.) - 1.0;
   w = x1 * x1 + x2 * x2;
 } while ( w >= 1.0 );
 
 w = sqrt( (-2.0 * log( w ) ) / w );
 *z_0 = (x1 * w)*std_dev+mean;
 *z_1 = (x2 * w)*std_dev+mean;
}


float CRandomGenerator::random_gauss(float mean, float std_dev){

  float z_0,z_1;

  this->random_gauss(mean, std_dev, &z_0,&z_1);
  
  return (this->tossCoin(0.5)?z_0:z_1);
}



std::ostream & operator << (std::ostream & os, const CRandomGenerator& rg) {
	os<< "s : " << rg.seed << std::endl;
	return os;
}
