#include "propagation.hpp"

float simple(float * &x, float * &weight, int length){
	float result = 0.f;
	for (int i = 0 ; i<length; i++) {
		result += x[i] * weight[i];
	}
	return result;
}

void initialize(
	const int &seed,
	const std::string &dist,
	const float &param1,
	const float &param2,
    network &neuronalNetwork
) {
	// Memory allocation for errors and inputs array
	neuronalNetwork.inputs = static_cast<float **>(malloc(neuronalNetwork.numberLayers * sizeof(float *)));
	neuronalNetwork.errors = static_cast<float **>(malloc(neuronalNetwork.numberLayers * sizeof(float *)));
    for (int i = 0; i < neuronalNetwork.numberLayers; i++) {
		if (neuronalNetwork.biasByLayers[i]) {
			neuronalNetwork.inputs[i] = static_cast<float *>(malloc((1 + neuronalNetwork.neuronsByLayers[i])* sizeof(float)));
			neuronalNetwork.errors[i] = static_cast<float *>(malloc((1 + neuronalNetwork.neuronsByLayers[i])* sizeof(float)));
			for(int j = 0; j < neuronalNetwork.neuronsByLayers[i]+1 ; j++) {
				neuronalNetwork.inputs[i][j] = 0.f;
				neuronalNetwork.errors[i][j] = 0.f;
			}
		} else {
			neuronalNetwork.inputs[i] = static_cast<float *>(malloc(neuronalNetwork.neuronsByLayers[i] * sizeof(float)));
			neuronalNetwork.errors[i] = static_cast<float *>(malloc(neuronalNetwork.neuronsByLayers[i] * sizeof(float)));
			for(int j = 0; j < neuronalNetwork.neuronsByLayers[i] ; j++) {
				neuronalNetwork.inputs[i][j] = 0.f;
				neuronalNetwork.errors[i][j] = 0.f;
			}
		}
	}
	// Weight initialisation
	pcg32 engine = generateEngine(seed);
	// Memory allocation for weights 
	neuronalNetwork.weights = static_cast<float ***>(malloc((neuronalNetwork.numberLayers - 1)* sizeof(float **)));
	neuronalNetwork.weightsPreviousChange = static_cast<float ***>(malloc((neuronalNetwork.numberLayers - 1)* sizeof(float **)));
	for (int i = 0; i < neuronalNetwork.numberLayers -1; i++) {
		int numberOfNeuronsInPreviousLayer = neuronalNetwork.neuronsByLayers[i];
		if (neuronalNetwork.biasByLayers[i]) numberOfNeuronsInPreviousLayer++;
		int numberOfNeuronsInFollowingLayer = neuronalNetwork.neuronsByLayers[i+1];
		neuronalNetwork.weights[i] = static_cast<float **>(malloc(numberOfNeuronsInFollowingLayer * sizeof(float *)));
		neuronalNetwork.weightsPreviousChange[i] = static_cast<float **>(malloc(numberOfNeuronsInFollowingLayer * sizeof(float *)));
		for(int j = 0; j < numberOfNeuronsInFollowingLayer; j++) {
			neuronalNetwork.weights[i][j] = static_cast<float *>(malloc(numberOfNeuronsInPreviousLayer * sizeof(float)));
			neuronalNetwork.weightsPreviousChange[i][j] = static_cast<float *>(malloc(numberOfNeuronsInPreviousLayer * sizeof(float)));
			for(int k = 0; k < numberOfNeuronsInPreviousLayer; k++) {
				neuronalNetwork.weights[i][j][k] = generateRandom(engine, dist, param1, param2);
				neuronalNetwork.weightsPreviousChange[i][j][k] = 0.f;
			}
		}
	}
	// Set up cost function in network structure
	neuronalNetwork.costFunction = costMatch(activationFunctionStringFromEnum(neuronalNetwork.activationFunctionByLayers[neuronalNetwork.numberLayers-1]));
}

void freeNetwork(network & neuronalNetwork) {
	// Free weights memory allocations
	for (int i = 0; i < neuronalNetwork.numberLayers -1; i++) {
		int numberOfNeuronsInPreviousLayer = neuronalNetwork.neuronsByLayers[i];
		if (neuronalNetwork.biasByLayers[i]) numberOfNeuronsInPreviousLayer++;
		int numberOfNeuronsInFollowingLayer = neuronalNetwork.neuronsByLayers[i+1];
		for(int j = 0; j < numberOfNeuronsInFollowingLayer; j++) {
			free(neuronalNetwork.weights[i][j]);
			free(neuronalNetwork.weightsPreviousChange[i][j]);
		}
		free(neuronalNetwork.weights[i]);
		free(neuronalNetwork.weightsPreviousChange[i]);
	}
	free(neuronalNetwork.weights);
	free(neuronalNetwork.weightsPreviousChange); 
	// Freee inputs and errors memory allocations
    for (int i = 0; i < neuronalNetwork.numberLayers; i++) {
		free(neuronalNetwork.inputs[i]);
		free(neuronalNetwork.errors[i]);
	}
    free(neuronalNetwork.inputs);
    free(neuronalNetwork.errors);
	// Free other dynamic arrays
	free(neuronalNetwork.biasByLayers);
	free(neuronalNetwork.neuronsByLayers);
	free(neuronalNetwork.activationFunctionByLayers);
	free(neuronalNetwork.activationFunctionDerivByLayers);
}

void setInputInNeuronalNetwork(
	float * &input, 
    network &neuronalNetwork
) {
	for(int i = 0; i < neuronalNetwork.neuronsByLayers[0]; i++){
		neuronalNetwork.inputs[0][i] = input[i];
	}
}

void propagate(
    network &neuronalNetwork
) {
	// We loop from the second layer to the last one because we need the previous one to compute the next one ...
	for(int i = 1; i < neuronalNetwork.numberLayers; i++) {
		// ... we get the size of the input layer that includes the bias ...
		int loopLength = neuronalNetwork.neuronsByLayers[i];
		// ... if there is a bias, because we are on the layer to compute we ignore it and ...
		if  (neuronalNetwork.biasByLayers[i]) {
			loopLength -= 1;
		}
		// ... forward by neuron. We do not care here about softmax because the trick makes the activation function behaves as identity ...
		for(int j = 0; j <loopLength; j++) {
			neuronalNetwork.inputs[i][j] = 
				neuronalNetwork.activationFunctionArray[neuronalNetwork.activationFunctionByLayers[i]](
					simple(
						neuronalNetwork.inputs[i-1],
						neuronalNetwork.weights[i-1][j],
						neuronalNetwork.neuronsByLayers[i-1]
					)
				);
		}
		// ... so we check at the end of the classical computation on the layer the use of softmax and update consequently
		if (activationFunctionStringFromEnum(neuronalNetwork.activationFunctionByLayers[i]).compare("softmax") == 0) {
			// Softmax computation
			softmax_real(neuronalNetwork.inputs[i], loopLength);
		}
	}
}

void retropropagateOnline(
	const float &inertiaRate,
	const float &learningRate,
	float * &outputExpected,
	const int iteration,
    network &neuronalNetwork
) {
	// Get the id of the last layer
	int lastLayerId = neuronalNetwork.numberLayers - 1;
	// Update the error evolution if necessary
	if (iteration >= 0 && iteration < COST_EVOLUTION_LENGTH) 
		neuronalNetwork.costEvolution[iteration] = neuronalNetwork.costFunction(
			outputExpected, 
			neuronalNetwork.inputs[lastLayerId], 
			neuronalNetwork.neuronsByLayers[lastLayerId]
		);
	// Compute the error on the output layer first...
	for(int i = 0; i < neuronalNetwork.neuronsByLayers[lastLayerId]; i++) {
		// we compute the error
		neuronalNetwork.errors[lastLayerId][i] = 
			neuronalNetwork.activationDerivFunctionArray[neuronalNetwork.activationFunctionDerivByLayers[lastLayerId]](
				simple(
					neuronalNetwork.inputs[lastLayerId-1],
					neuronalNetwork.weights[lastLayerId-1][i],
						neuronalNetwork.neuronsByLayers[lastLayerId-1]
				)
			) * (outputExpected[i] - neuronalNetwork.inputs[lastLayerId][i]);
	}
	// Then, for each hidden layer, starting from the end...
    for (int i = lastLayerId -1; i > 0; i--) {
		// and for each neuron of the current layer and the following one...
		int numberOfNeuronsInCurrentLayer = neuronalNetwork.neuronsByLayers[i];
		int numberOfNeuronsInFollowingCurrentLayer = neuronalNetwork.neuronsByLayers[i+1];
		// ... except the bias if exist
		if (neuronalNetwork.biasByLayers[i+1]) {
			numberOfNeuronsInFollowingCurrentLayer--;
		}
		for(int j = 0; j < numberOfNeuronsInCurrentLayer; j++) {
			// ... we just want to get the right weight array (tranpose)
			float * weightTranspose = static_cast<float *>(malloc(numberOfNeuronsInFollowingCurrentLayer * sizeof(float)));
			for(int t = 0; t<numberOfNeuronsInFollowingCurrentLayer; t++) {
				weightTranspose[t] = neuronalNetwork.weights[i][t][j];
			}
			// we compute the error value.
			neuronalNetwork.errors[i][j] = 
			neuronalNetwork.activationDerivFunctionArray[neuronalNetwork.activationFunctionDerivByLayers[i]](
					simple(
						neuronalNetwork.inputs[i-1],
						neuronalNetwork.weights[i-1][j],
						neuronalNetwork.neuronsByLayers[i-1]
					)
				) * simple(
						neuronalNetwork.errors[i+1],
						weightTranspose,
						numberOfNeuronsInFollowingCurrentLayer
				);
			free(weightTranspose);
		}
	}
	// Finally, we update all weights
	// For each stage between layers ...
	// lastLayerId = nbLayer - 1
	for(int i = 0; i < lastLayerId; i++) {
		// ... and each neurons of the current layer ...
		for(int j = 0; j < neuronalNetwork.neuronsByLayers[i+1]; j++) {
			// ... and each weight of the current neuron of the current layer ...
			for(int k = 0; k < neuronalNetwork.neuronsByLayers[i]; k++) {
				// W [i] [j] [k] denotes the weight between layers i and i + 1 of the neurons j of the layer i + 1 and the neuron k of the layer i 
				// Compute the new delta weight
				float weightDelta = 
					inertiaRate
					* learningRate 
					* neuronalNetwork.errors[i+1][j]
					* neuronalNetwork.inputs[i][k]
					+
					(1.f - inertiaRate)
					* neuronalNetwork.weightsPreviousChange[i][j][k];
				// Update the weights of the neuronal network
				neuronalNetwork.weights[i][j][k] += weightDelta;
				// Update the neuronal network memory with last delta weigth
				neuronalNetwork.weightsPreviousChange[i][j][k] = weightDelta;
			}
		}
	}
}

void computeBatchError(
	float * &outputExpected, 
	const int iteration,
    network &neuronalNetwork
){
	int lastLayerId = neuronalNetwork.numberLayers - 1;
	// Update the error evolution
	if (iteration >= 0 && iteration < COST_EVOLUTION_LENGTH) 
		neuronalNetwork.costEvolution[iteration] = neuronalNetwork.costFunction(
			outputExpected,
			neuronalNetwork.inputs[lastLayerId],
			neuronalNetwork.neuronsByLayers[lastLayerId]
		);
	for(int i = 0; i < neuronalNetwork.neuronsByLayers[lastLayerId]; i++) {
		// we compute the error
		neuronalNetwork.errors[lastLayerId][i] +=
			neuronalNetwork.activationDerivFunctionArray[neuronalNetwork.activationFunctionDerivByLayers[lastLayerId]](
				simple(
					neuronalNetwork.inputs[lastLayerId-1],
					neuronalNetwork.weights[lastLayerId-1][i],
					neuronalNetwork.neuronsByLayers[lastLayerId-1]
				)
			) * (outputExpected[i] - neuronalNetwork.inputs[lastLayerId][i]);
	}
}

void retropropagateBatch(
	const int &batchSize,
	const char * batchErrorMethod,
	const float &inertiaRate,
	const float &learningRate,
    network &neuronalNetwork
){
	int lastLayerId = neuronalNetwork.numberLayers - 1;
	// First, we check the method used about batch errors (only sum or average)
	if (strcmp(batchErrorMethod, "average") == 0) {
		// if average, we divide the error by the batch size
		for(int i = 0; i < neuronalNetwork.neuronsByLayers[lastLayerId]; i++) {
			// we compute the error
			neuronalNetwork.errors[lastLayerId][i] /= batchSize;
		}
	}
	// Then, for each hidden layer, starting from the end...
    for (int i = lastLayerId -1; i > 0; i--) {
		// and for each neuron of the current layer and the following one...
		int numberOfNeuronsInCurrentLayer = neuronalNetwork.neuronsByLayers[i];
		int numberOfNeuronsInFollowingCurrentLayer = neuronalNetwork.neuronsByLayers[i+1];
		// ... except the bias if exist
		if (neuronalNetwork.biasByLayers[i+1]) {
			numberOfNeuronsInFollowingCurrentLayer--;
		}
		for(int j = 0; j < numberOfNeuronsInCurrentLayer; j++) {
			// ... we just want to get the right weight array (tranpose)
			float * weightTranspose = static_cast<float *>(malloc(numberOfNeuronsInFollowingCurrentLayer * sizeof(float)));
			for(int t = 0; t<numberOfNeuronsInFollowingCurrentLayer; t++) {
				weightTranspose[t] = neuronalNetwork.weights[i][t][j];
			}
			// we compute the error value.
			neuronalNetwork.errors[i][j] = 
			neuronalNetwork.activationDerivFunctionArray[neuronalNetwork.activationFunctionDerivByLayers[i]](
					simple(
						neuronalNetwork.inputs[i-1],
						neuronalNetwork.weights[i-1][j],
						neuronalNetwork.neuronsByLayers[i-1]
					)
				) * simple(
						neuronalNetwork.errors[i+1],
						weightTranspose,
						numberOfNeuronsInFollowingCurrentLayer
				);
			free(weightTranspose);
		}
	}
	// Finally, we update all weights
	// For each stage between layers ...
	// lastLayerId = nbLayer - 1
	for(int i = 0; i < lastLayerId; i++) {
		// ... and each neurons of the current layer ...
		for(int j = 0; j < neuronalNetwork.neuronsByLayers[i+1]; j++) {
			// ... and each weight of the current neuron of the current layer ...
			for(int k = 0; k < neuronalNetwork.neuronsByLayers[i]; k++) {
				// W [i] [j] [k] denotes the weight between layers i and i + 1 of the neurons j of the layer i + 1 and the neuron k of the layer i 
				// Compute the new delta weight
				float weightDelta = 
					inertiaRate
					* learningRate 
					* neuronalNetwork.errors[i+1][j]
					* neuronalNetwork.inputs[i][k]
					+
					(1.f - inertiaRate)
					* neuronalNetwork.weightsPreviousChange[i][j][k];
				// Update the weights of the neuronal network
				neuronalNetwork.weights[i][j][k] += weightDelta;
				// Update the neuronal network memory with last delta weigth
				neuronalNetwork.weightsPreviousChange[i][j][k] = weightDelta;
			}
		}
	}
	// And we don't forget to reinitialize batch error !
	for(int i = 0; i < neuronalNetwork.neuronsByLayers[lastLayerId]; i++) {
		neuronalNetwork.errors[lastLayerId][i] = 0;
	}
}

void printInputsOfNeuronalNetwork(const network &neuronalNetwork) {
	std::cout << std::endl <<  "Print of Inputs : " << std::endl;
	for (int i = 0; i < neuronalNetwork.numberLayers; i++) {
		std::stringstream ss;
		for (int j = 0; j < neuronalNetwork.neuronsByLayers[i]; j++) {
			std::ostringstream strs;
			strs << neuronalNetwork.inputs[i][j];
			ss << strs.str() << "   ";
		}
		std::cout << "     " << i+1 << " : " << ss.str() << std::endl;
	}
}

void printWeightsOfNeuronalNetwork(const network &neuronalNetwork) {
	std::cout << std::endl <<  "Print of Weights : " << std::endl << std::endl;
	for (int i = 0; i < neuronalNetwork.numberLayers -1; i++) {
		for(int j = 0; j < neuronalNetwork.neuronsByLayers[i+1]; j++) {
			std::stringstream ss;
			for(int k = 0; k < neuronalNetwork.neuronsByLayers[i]; k++) {
				std::ostringstream strs;
				strs << neuronalNetwork.weights[i][j][k];
				ss << strs.str() << "   ";
			}
			std::cout << ss.str() << std::endl;
		}
		std::cout << std::endl;
	}
}

void printErrorsOfNeuronalNetwork(const network &neuronalNetwork) {
	std::cout  <<  "Print of Errors Evolution: " << std::endl;
	std::stringstream ss;
	for (int i = 0; i < COST_EVOLUTION_LENGTH; i++) {
		std::ostringstream strs;
		strs << neuronalNetwork.costEvolution[i];
		ss << strs.str() << "   ";
	}
	std::cout << ss.str() << std::endl;
}

void printNeuronalNetwork(const network &neuronalNetwork) {
	std::cout  <<  "=================================" << std::endl;
	printInputsOfNeuronalNetwork(neuronalNetwork);
	printWeightsOfNeuronalNetwork(neuronalNetwork);
	printErrorsOfNeuronalNetwork(neuronalNetwork);
	std::cout  <<  "=================================" << std::endl;
}