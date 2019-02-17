/*
 * CRandomGenerator.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef CRANDOMGENERATOR_H_
#define CRANDOMGENERATOR_H_

#include <iostream>
#include "MersenneTwister.h"
class CRandomGenerator {
  private:
    unsigned seed;
    MTRand* mt_rnd;


    double oldrand[55];
    int jrand;
//    void randomize();
    void warmup_random(double seed);
    void advance_random();
    double randomperc();
  public:
    CRandomGenerator(unsigned int seed);
    CRandomGenerator();
    ~CRandomGenerator();
    int randInt();
    bool tossCoin();
    bool tossCoin(float bias);
    int randInt(int min, int max);
    int getRandomIntMax(int max);
    float randFloat(float min, float max);
    int random(int min, int max);
    float random(float min, float max);
    double random(double min, double max);
    int rnd (int low, int high);
    double rndreal(double low, double high);


    void random_gauss(float min, float max, float* z_0, float* z_1);
    float random_gauss(float mean, float std_dev);

    unsigned get_seed()const {return this->seed;}
    friend std::ostream & operator << (std::ostream & os, const CRandomGenerator& rg);

};

#endif /* CRANDOMGENERATOR_H_ */
