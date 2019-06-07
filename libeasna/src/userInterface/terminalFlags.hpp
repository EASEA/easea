
#ifndef TERMINALFLAGS_H
#define TERMINALFLAGS_H

#include "../userInterface/jsonPerceptronArchitectureParser.hpp"
#include "../userInterface/terminalUserInterface.hpp"

/**
 * @brief Get user input from terminal to initialise perceptron architecture
 * 
 * @param inertiaRate Reference to the inertia rate float
 * @param learningRate Reference to the learning rate float
 * @param seed Reference to the seed value
 * @param dist Reference to the distribution to use within weights initialization
 * @param paramOfDist1 Reference to the first parameter of the distribution
 * @param paramOfDist2 Reference to the second parameter of the distribution
 * @param neuronalNetwork 
 * @param initComput
 */
void useTerminalToGetArchitecture(
	float &inertiaRate,
	float &learningRate,
	int &seed, 
	std::string &dist,
	float &paramOfDist1,
	float &paramOfDist2,
    network &neuronalNetwork,
	bool &initComput
);

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
 * @param initComput
 */
void parseJsonPerceptronArchitecture(
    const char * json,
	float &inertiaRate,
	float &learningRate,
	int &seed, 
	std::string &dist,
	float &paramOfDist1,
	float &paramOfDist2,
    network &neuronalNetwork,
	bool &initComput
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
 */
void saveJsonPerceptronArchitecture(
    const char * json,
	const float &inertiaRate,
	const float &learningRate,
	const int &seed, 
	const std::string &dist,
	const float &paramOfDist1,
	const float &paramOfDist2,
    network &neuronalNetwork
);

/**
 * @brief  Allows the user to perform an online learning on the perceptron with the given input file
 * 
 * @param learnFile 
 * @param inertiaRate 
 * @param learningRate
 * @param neuronalNetwork 
 */
void learnOnlineFromFile(
	char * learnFile,
	const float &inertiaRate,
	const float &learningRate,
    network &neuronalNetwork
);

/**
 * @brief  Allows the user to perform a batch learning on the perceptron with the given input file
 * 
 * @param learnFile 
 * @param batchSize 
 * @param batchErrorMethod 
 * @param inertiaRate 
 * @param learningRate
 * @param neuronalNetwork 
 */
void learnBatchFromFile(
	char * learnFile,
	const int &batchSize,
	const char * batchErrorMethod,
	const float &inertiaRate,
	const float &learningRate,
    network &neuronalNetwork
);

/**
 * @brief  Allows the user to get the output of the perceptron according to the given input
 * 
 * @param csvCompute
 * @param numberOfInputs
 * @param neuronalNetwork 
 */
void computeUserInputFile(
	char * csvCompute,
	int numberOfInputs,
    network &neuronalNetwork
);

#endif