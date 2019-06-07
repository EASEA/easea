
#ifndef PROPAGATION_H
#define PROPAGATION_H

#define COST_EVOLUTION_LENGTH 512

#include <cstring>
#include <sstream>

#include <assert.h>
#include <stdint.h>

#include "simdConfig.hpp"
#include "functions/activationFunctions.hpp"
#include "functions/costFunctions.hpp"
#include "functions/distributionFunctions.hpp"

/**
 * @brief Structure used to compute value in a neuronal network
 * 
 */
struct network {
	int numberLayers;
	bool * biasByLayers {nullptr};
	int * neuronsByLayers {nullptr};
	ActivationFunction * activationFunctionByLayers {nullptr};
	ActivationDerivFunction * activationFunctionDerivByLayers{nullptr};
    float ** inputs;
    float *** weights;
    float *** weightsPreviousChange;
	float ** errors;
	float costEvolution[COST_EVOLUTION_LENGTH];
	CostType costFunction;
	ActivationType activationFunctionArray[9] = {
		nullptr,
		identity,
		sigmoid,
		arctan,
		sinusoid,
		tanhyper,
		relu,
		gaussian,
		softmax
	};
	ActivationType activationDerivFunctionArray[9] = {
		nullptr,
		identity_deriv,
		sigmoid_deriv,
		arctan_deriv,
		sinusoid_deriv,
		tanhyper_deriv,
		relu_deriv,
		gaussian_deriv,
		softmax_deriv
	};
};

/**
 * @brief 
 * 
 * @param seed Seed value
 * @param dist Distribution to use within weights initialization
 * @param param1 First parameter of the distribution
 * @param param2 Second parameter of the distribution
 * @param neuronalNetwork Reference to the values within the neuronal network
 */
void initialize(
	const int &seed,
	const std::string &dist,
	const float &param1,
	const float &param2,
    network &neuronalNetwork
);

/**
 * @brief 
 * 
 * @param neuronalNetwork 
 */
void freeNetwork(network & neuronalNetwork);

/**
 * @brief Set the Input In Neuronal Network object
 * 
 * @param input Reference to a float array that will be copied in the first layer of the neuronal network
 * @param neuronalNetwork Reference to the values within the neuronal network
 */
void setInputInNeuronalNetwork(
	float * &input, 
    network &neuronalNetwork
);

/**
 * @brief Propagate the input values given into the neuronal network
 * 
 * @param neuronalNetwork Reference to the values within the neuronal network
 */
void propagate(
    network &neuronalNetwork
);

/**
 * @brief Retropropagate and update all weights according to the expected output and the learning rate and the inertia rate
 * 
 * @param inertiaRate Reference to the inertia rate float
 * @param learningRate Reference to the learning rate float
 * @param outputExpected Reference to a float array that will contains the expected output
 * @param iteration Integer to keep track of the evolution of the cost function
 * @param neuronalNetwork Reference to the values within the neuronal network
 */
void retropropagateOnline(
	const float &inertiaRate,
	const float &learningRate,
	float * &outputExpected, 
	const int iteration,
    network &neuronalNetwork
);

/**
 * @brief  Compute the error in batch mode
 * 
 * @param input Reference to a float array that will contains the expected output
 * @param iteration Integer to keep track of the evolution of the cost function
 * @param neuronalNetwork Reference to the values within the neuronal network
 */
void computeBatchError(
	float * &outputExpected,
	const int iteration,
    network &neuronalNetwork
);

/**
 * @brief Retropropagate by batch and update all weights according to the expected output and the learning rate and the inertia rate
 * 
 * @param batchSize Reference to the size of the current batch
 * @param batchErrorMethod Reference to the method used with batch error
 * @param inertiaRate Reference to the inertia rate float
 * @param learningRate Reference to the learning rate float
 * @param neuronalNetwork Reference to the values within the neuronal network
 */
void retropropagateBatch(
	const int &batchSize,
	const char * batchErrorMethod,
	const float &inertiaRate,
	const float &learningRate,
    network &neuronalNetwork
);

/**
 * @brief Print all values returned by all neurons in the neuronal network
 * 
 * @param neuronalNetwork 
 */
void printInputsOfNeuronalNetwork(const network &neuronalNetwork);

/**
 * @brief Print all weights of the neuronal network
 * 
 * @param neuronalNetwork 
 */
void printWeightsOfNeuronalNetwork(const network &neuronalNetwork);

/**
 * @brief Print the error values of the neuronal network computed
 * 
 * @param neuronalNetwork 
 */
void printErrorsOfNeuronalNetwork(const network &neuronalNetwork);

/**
 * @brief Print all values of the neuronal network : Inputs, Weights and Errors
 * 
 * @param neuronalNetwork 
 */
void printNeuronalNetwork(const network &neuronalNetwork);

#endif