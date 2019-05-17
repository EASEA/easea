#include "terminalFlags.hpp"

void useTerminalToGetArchitecture(
	float &inertiaRate,
	float &learningRate,
	int &seed, 
	std::string &dist,
	float &paramOfDist1,
	float &paramOfDist2,
    network &neuronalNetwork,
	bool &initComput
){
	useOfTerminal(inertiaRate, learningRate, seed, dist, paramOfDist1, paramOfDist2, neuronalNetwork);
	initialize(seed, dist, paramOfDist1, paramOfDist2, neuronalNetwork);
	initComput = true;
}

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
){
	if (readJsonPerceptronArchitecture(json, inertiaRate, learningRate, seed, dist, paramOfDist1, paramOfDist2, neuronalNetwork)) {
		initialize(seed, dist, paramOfDist1, paramOfDist2, neuronalNetwork);
		initComput = true;
	} else {
		std::cout << "Error when parsing the json input file." << std::endl;
		exit(1);
	}
}

void saveJsonPerceptronArchitecture(
    const char * json,
	const float &inertiaRate,
	const float &learningRate,
	const int &seed, 
	const std::string &dist,
	const float &paramOfDist1,
	const float &paramOfDist2,
    network &neuronalNetwork
){
	if (writeJsonPerceptronArchitecture(json, inertiaRate, learningRate, seed, dist, paramOfDist1, paramOfDist2, neuronalNetwork)) {
		std::cout << "Perceptron Architecture saved as " << json << "." << std::endl;
	} else {
		std::cout << "Error when writing the json output file." << std::endl;
		exit(1);
	}
}

void learnOnlineFromFile(
	char * learnFile,
	const float &inertiaRate,
	const float &learningRate,
    network &neuronalNetwork
){
	// Number of inputs for reuse
	int numberOfInputs = neuronalNetwork.neuronsByLayers[0];
	// Number of outputs for reuse
	int numberOfOutputs = neuronalNetwork.neuronsByLayers[neuronalNetwork.numberLayers - 1];
	// Define input file stream for reading csv file
	std::ifstream myLearningDataset;
	myLearningDataset.open(learnFile);
	// arrays to use for propagation and retropropagation
	float * inputLearningData = static_cast<float *>(malloc(numberOfInputs * sizeof(float)));
	for(int i = 0; i<numberOfInputs; i++) inputLearningData[i] = 0.f;
	float * expectedResultData = static_cast<float *>(malloc(numberOfOutputs * sizeof(float)));
	for(int i = 0; i<numberOfOutputs; i++) expectedResultData[i] = 0.f;
	// Variable to get each line read by the getline function
	std::string line;
	// To skip the first line of csv file in which the column are described
	getline(myLearningDataset, line);
	// Variable to store the number of line read : will be used to manage the cost evolution
	int readLines = 0, iterationCost = 0;
	// For each input from csv file ...
	while(getline(myLearningDataset, line)) {
		// Increment in order to manage both arrays
		int i = 0;
		// Parse the line in order to set up both arrays
		std::string data;
		std::istringstream dataStream(line);
		while (getline(dataStream, data, ';')) {
			if (i < numberOfInputs) {
				inputLearningData[i] = stof(data);
			} else {
				expectedResultData[i - numberOfInputs] = stof(data);
			}
			i++;
		}
		// Set the input array in neuronal network
		setInputInNeuronalNetwork(inputLearningData,neuronalNetwork);
		// Propagate in neuronal network in order to compute outputs
		propagate(neuronalNetwork);
		// Retropropagate the error according the computed output and the iteration value to use to keep track of the cost function evolution
		if (readLines%COST_EVOLUTION_LENGTH == 0) {
			retropropagateOnline(inertiaRate, learningRate, expectedResultData, iterationCost, neuronalNetwork);
			iterationCost++;
		} else {
			retropropagateOnline(inertiaRate, learningRate, expectedResultData, -1, neuronalNetwork);
		}
		// Update readLines variable
		readLines++;
	}
	// Free memory allocation
	free(inputLearningData);
	free(expectedResultData);
	// close file descriptor
	myLearningDataset.close();
}

void learnBatchFromFile(
	char * learnFile,
	const int &batchSize,
	const char * batchErrorMethod,
	const float &inertiaRate,
	const float &learningRate,
    network &neuronalNetwork
){
	// batch increment
	int batchIncrement = 0;
	// Number of inputs for reuse
	int numberOfInputs = neuronalNetwork.neuronsByLayers[0];
	// Number of outputs for reuse
	int numberOfOutputs = neuronalNetwork.neuronsByLayers[neuronalNetwork.numberLayers - 1];
	// Define input file stream for reading csv file
	std::ifstream myLearningDataset;
	myLearningDataset.open(learnFile);
	// arrays to use for propagation and retropropagation
	float * inputLearningData = static_cast<float *>(malloc(numberOfInputs * sizeof(float)));
	for(int i = 0; i<numberOfInputs; i++) inputLearningData[i] = 0.f;
	float * expectedResultData = static_cast<float *>(malloc(numberOfOutputs * sizeof(float)));
	for(int i = 0; i<numberOfOutputs; i++) expectedResultData[i] = 0.f;
	// Variable to get each line read by the getline function
	std::string line;
	// To skip the first line of csv file in which the column are described
	getline(myLearningDataset, line);
	// Variable to store the number of line read : will be used to manage the cost evolution
	int readLines = 0, iterationCost = 0;
	// For each input from csv file ...
	while(getline(myLearningDataset, line)) {
		// Increment in order to manage both arrays
		int i = 0;
		// Parse the line in order to set up both arrays
		std::string data;
		std::istringstream dataStream(line);
		while (getline(dataStream, data, ';')) {
			if (i < numberOfInputs) {
				inputLearningData[i] = stof(data);
			} else {
				expectedResultData[i - numberOfInputs] = stof(data);
			}
			i++;
		}
		// Set the input array in neuronal network
		setInputInNeuronalNetwork(inputLearningData, neuronalNetwork);
		// Propagate in neuronal network in order to compute outputs
		propagate(neuronalNetwork);
		// Compute the batch error ...
		if ((readLines%COST_EVOLUTION_LENGTH) - batchSize == 0 ) {
			computeBatchError(expectedResultData, iterationCost, neuronalNetwork);
			iterationCost++;
		} else {
			computeBatchError(expectedResultData, -1, neuronalNetwork);
		}
		// ... and update batch increment
		batchIncrement++;
		// Check if retropropagation is needed
		if (batchIncrement >= batchSize) {
			retropropagateBatch(batchIncrement, batchErrorMethod, inertiaRate, learningRate, neuronalNetwork);
			// Do not forget to reinitialize batch increment and batch error (job done in retropropagateBatch function)
			batchIncrement = 0;
		}
		// Update readLines variable
		readLines++;
	}
	// Free memory allocation
	free(inputLearningData);
	free(expectedResultData);
	// close file descriptor
	myLearningDataset.close();
	// Check all batches have been used, if some data are unused, a retropropagation is forced
	if (batchIncrement) {
		retropropagateBatch(batchIncrement, batchErrorMethod, inertiaRate, learningRate, neuronalNetwork);
		std::cout << "Warning : Batch Size is not a multiple of Dataset Size !" << std::endl;
	}
}

void computeUserInputFile(
	char * csvCompute,
	int numberOfInputs,
    network &neuronalNetwork
){
	// Some precomputed value to help
	int lastLayerId = neuronalNetwork.numberLayers - 1;
	// Memory allocation of inputs and outputs
	float ** inputsFromUser = static_cast<float **>(malloc(numberOfInputs * sizeof(float *)));
	float ** outputsForUser = static_cast<float **>(malloc(numberOfInputs * sizeof(float *)));
	for (int i = 0; i < numberOfInputs; i++) {
		inputsFromUser[i] = static_cast<float *>(malloc(neuronalNetwork.neuronsByLayers[0] * sizeof(float)));
		outputsForUser[i] = static_cast<float *>(malloc(neuronalNetwork.neuronsByLayers[lastLayerId] * sizeof(float)));
	}
	// Read all inputs
	readComputeCsvFile(csvCompute, inputsFromUser);
	// For each input
	for (int i = 0 ; i < numberOfInputs ; i ++) {
		// Set the input array in neuronal network
		setInputInNeuronalNetwork(inputsFromUser[i],neuronalNetwork);
		// Propagate in neuronal network in order to compute outputs
		propagate(neuronalNetwork);
		// Store output
		for (int j = 0 ; j < neuronalNetwork.neuronsByLayers[lastLayerId] ; j++)
			outputsForUser[i][j] = neuronalNetwork.inputs[lastLayerId][j];
	}
	writeComputeCsvFile(csvCompute, numberOfInputs, inputsFromUser, neuronalNetwork.neuronsByLayers[0], outputsForUser, neuronalNetwork.neuronsByLayers[lastLayerId]);
	// Free memory
	for (int i = 0; i < numberOfInputs; i++) {
		free(inputsFromUser[i]);
		free(outputsForUser[i]);
	}
    free(inputsFromUser);
    free(outputsForUser);
}