/*
 * CPseudoRandom.h
 *
 *  Created on: 26 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CPSEUDORANDOM_H
#define CPSEUDORANDOM_H

class CPseudoRandom {
public:
    CPseudoRandom();

    static double randDouble();
    static int randInt(int lowerLimit, int upperLimit);
    static double randDouble(double lowerLimit, double upperLimit);
};

#endif
