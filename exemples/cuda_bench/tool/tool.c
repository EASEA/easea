#include <stdlib.h>
#include <stdio.h>
#include "tool.h"
#include <math.h>

int tossCoin(double p){

  if( rand() < p*RAND_MAX)
    return 1;
  else 
    return 0;
}

int getRandomIntMax(int max){
  double r = rand();
  r = r / RAND_MAX;
  r = r * max;
  return r;
}


int randomLoc(int min, int max){
  return min+getRandomIntMax(max-min);
}

float getRandomFloatMax(float max){
  float r = rand();
  r = r / RAND_MAX;
  r = r * max;
  return r;
}

float fRandomLoc(float min,float max){
  return min+getRandomFloatMax(max-min);
}



size_t
partieEntiereSup(float E){
  int fl = floor(E);
  if( fl == E )
    return E;
  else
    return floor(E)+1;
}
