/*
 * CCPseudoRandom.cpp
 *
 *  Created on: 26 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#include <math.h>
#include "CPseudoRandom.h"
/* Definition of random number generation routines
 * Based on the concept, which was implemented by Deb's in NSGAII
 * More info can be found on web site of Kanpur Genetic Algorithms Laboratory:
 * www.iitk.ac.in/kangal/index.shtml
 */
CRandomGenerator * CPseudoRandom::randomGenerator_ = nullptr ;

CPseudoRandom::CPseudoRandom() {
    if (CPseudoRandom::randomGenerator_ == nullptr)
        CPseudoRandom::randomGenerator_ = new CRandomGenerator();
}
/* Fetch a single double number between 0.0 and 1.0 */
double CPseudoRandom::randDouble() {
    if (CPseudoRandom::randomGenerator_ == nullptr)
        CPseudoRandom::randomGenerator_ = new CRandomGenerator();

    return CPseudoRandom::randomGenerator_->rndreal(0.0,1.0);
}
/* Fetch a single random integer number between lowerLimit and upperLimit */
int CPseudoRandom::randInt(int lowerLimit, int upperLimit) {
    if (CPseudoRandom::randomGenerator_ == nullptr)
        CPseudoRandom::randomGenerator_ = new CRandomGenerator();

    return CPseudoRandom::randomGenerator_->rnd(lowerLimit, upperLimit);
}
/* Fetch a single random double number between lowerLimit and upperLimit */
double CPseudoRandom::randDouble(double lowerLimit, double upperLimit) {
    if (CPseudoRandom::randomGenerator_ == nullptr)
        CPseudoRandom::randomGenerator_ = new CRandomGenerator();

    return CPseudoRandom::randomGenerator_->rndreal(lowerLimit, upperLimit);
}

