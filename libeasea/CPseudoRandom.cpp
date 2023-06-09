/*
 * CCPseudoRandom.cpp
 *
 *  Created on: 26 jullet 2018
 *      Author: Anna Ouskova Leonteva
 *  ===
 *
 *  Updated on: 26 july 2022 by Léo Chéneau
 *
 */
#include <math.h>
#include "CPseudoRandom.h"
/* Definition of random number generation routines
 * Based on the concept, which was implemented by Deb's in NSGAII
 * More info can be found on web site of Kanpur Genetic Algorithms Laboratory:
 * www.iitk.ac.in/kangal/index.shtml
 */
CRandomGenerator CPseudoRandom::randomGenerator_ = CRandomGenerator{};

CPseudoRandom::CPseudoRandom()
{
}

/* Fetch a single random integer number between lowerLimit and upperLimit */
int CPseudoRandom::randInt(int lowerLimit, int upperLimit)
{
	return CPseudoRandom::randomGenerator_.random(lowerLimit, upperLimit);
}

/* Fetch a single random double number between lowerLimit and upperLimit */
double CPseudoRandom::randDouble(double lowerLimit, double upperLimit)
{
	return CPseudoRandom::randomGenerator_.random(lowerLimit, upperLimit);
}

/* Fetch a single double number between 0.0 and 1.0 */
double CPseudoRandom::randDouble()
{
	return randDouble(0.0, 1.0);
}
