/**
 * @file terminalUserInterface.hpp
 * @author Romain Orhand
 * @brief Header of terminal User Interface to gather pieces of information
 * @version 0.1
 * @date 2018-11-23
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#ifndef TERMINALUSERINTERFACE_H
#define TERMINALUSERINTERFACE_H

#include <limits>

#include "../userInterface/csvParser.hpp"

/**
 * @brief Get the User Input For Neuronal Network object and update all the references given in arguments
 * 
 * @param learningRate 
 * @param numberLayers 
 * @param neuronalNetwork
 */
void getUserInputForNeuronalNetwork(
	float &inertiaRate,
	float &learningRate,
    network &neuronalNetwork
);

/**
 * @brief Get the User Input For Initialization Of Perceptron Weights object and update all the references given in argument
 * 
 * @param seed 
 * @param dist 
 * @param paramOfDist1 
 * @param paramOfDist2 
 */
void getUserInputForInitializationOfPerceptronWeights(
	int &seed, 
	std::string &dist,
	float &paramOfDist1,
	float &paramOfDist2
);

/**
 * @brief Use getUserInputForNeuronalNetwork and getUserInputForInitializationOfPerceptronWeights functions
 * 
 * @param inertiaRate 
 * @param learningRate 
 * @param seed 
 * @param dist 
 * @param paramOfDist1 
 * @param paramOfDist2 
 * @param neuronalNetwork
 */
void useOfTerminal(
	float &inertiaRate,
	float &learningRate,
	int &seed, 
	std::string &dist,
	float &paramOfDist1,
	float &paramOfDist2,
    network &neuronalNetwork
);

#endif