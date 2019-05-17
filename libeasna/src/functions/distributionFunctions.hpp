
#ifndef DISTRIBUTIONFUNCTIONS_H
#define DISTRIBUTIONFUNCTIONS_H

#include <iostream>
#include <random>
#include <string>

#include "../../include/pcg-cpp/pcg_random.hpp"

/**
 * @brief Use to print all distribution function implemented
 * 
 */
void printDistributionFunctionName();

/**
 * @brief Test according a name if a distribution function is implmented
 * 
 * @param dist Name of the distribution function
 * @return true 
 * @return false 
 */
bool isImplementedDistributionFunction(const std::string dist);

/**
 * @brief Create a generator to use according a seed
 * 
 * @param seed Seed to se for the pseudo-random generator
 * @return pcg32 PCg Engine
 */
pcg32 generateEngine(const int seed);

/**
 * @brief Generate a pseudo-random float according a distribution and a PCG engine
 * 
 * @param engine Reference to the PCG engine
 * @param dist Name of the distribution to use to generate pseudo-random float 
 * @param param1 Reference to the first parameter of the distribution
 * @param param2 Reference to the second parameter of the distribution
 * @return float 
 */
float generateRandom(pcg32 &engine, const std::string dist, const float param1, const float param2 );

#endif