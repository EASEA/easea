
#ifndef JSONPERCEPTRONARCHITECTUREPARSER_H
#define JSONPERCEPTRONARCHITECTUREPARSER_H

#include <cstdio>
#include <functional>
#include <iostream>
#include <string>

#include "../propagation.hpp"
#include "../functions/activationFunctions.hpp"
#include "../../include/rapidjson/document.h"
#include "../../include/rapidjson/filereadstream.h"
#include "../../include/rapidjson/filewritestream.h"
#include "../../include/rapidjson/stringbuffer.h"
#include "../../include/rapidjson/writer.h"

using namespace rapidjson;

/**
 * @brief Test if the JSON document is well written
 * 
 * @param doc Reference to the json document
 * @return true 
 * @return false 
 */
bool isJsonPerceptronArchitectureWellWritten(const Document &doc);

/**
 * @brief Read the JSON file and initialize all parameters of the perceptron
 * 
 * @param json Pointer to the json file descriptor
 * @param inertiaRate Reference to the inertia rate float
 * @param learningRate Reference to the learning rate float
 * @param seed Reference to the seed value
 * @param dist Reference to the distribution to use within weights initialization
 * @param paramOfDist1 Reference to the first parameter of the distribution
 * @param paramOfDist2 Reference to the second parameter of the distribution
 * @param neuronalNetwork
 * 
 * @return true 
 * @return false if an error occurs with the file descriptor
 */
bool readJsonPerceptronArchitecture(
    const char * json,
	float &inertiaRate,
	float &learningRate,
	int &seed, 
	std::string &dist,
	float &paramOfDist1,
	float &paramOfDist2,
    network &neuronalNetwork
);

/**
 * @brief Write all parameters used to create the perceptron in a JSON file
 * 
 * @param json Pointer to the json file descriptor
 * @param inertiaRate Reference to the inertia rate float
 * @param learningRate Reference to the learning rate float
 * @param seed Reference to the seed value
 * @param dist Reference to the distribution to use within weights initialization
 * @param paramOfDist1 Reference to the first parameter of the distribution
 * @param paramOfDist2 Reference to the second parameter of the distribution
 * @param neuronalNetwork
 * 
 * @return true 
 * @return false if an error occurs with the file descriptor
 */
bool writeJsonPerceptronArchitecture(
    const char * json,
	const float &inertiaRate,
	const float &learningRate,
	const int &seed,
	const std::string &dist,
	const float &paramOfDist1,
	const float &paramOfDist2,
    network &neuronalNetwork
);

#endif