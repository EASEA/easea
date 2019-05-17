#include "terminalUserInterface.hpp"

void getUserInputForNeuronalNetwork(
	float &inertiaRate,
	float &learningRate,
    network &neuronalNetwork
){
	// Input from user to define the inertia rate
	std::cout << "Define your inertia rate with a number between 0 and 1 (default 1.0)" << std::endl;
	if (std::cin.peek() == '\n') { 
		inertiaRate = 1.f;
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
		std::cout << "Default set." << std::endl << std::endl;
    } else {
		std::cin >> inertiaRate;
		if (std::cin.fail() || inertiaRate < 0.f || inertiaRate >1.f) {
			inertiaRate = 1.f;
			std::cin.clear();
			std::cout << "Default set." << std::endl << std::endl;
		}
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
	}
	// Input from user to define the learning rate
	std::cout << "Define your learning rate with a number between 0 and 1 (default 0.1)" << std::endl;
	if (std::cin.peek() == '\n') { 
		learningRate = 0.1f;
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
		std::cout << "Default set." << std::endl << std::endl;
    } else {
		std::cin >> learningRate;
		if (std::cin.fail() || learningRate < 0.f || learningRate > 1.f) {
			learningRate = 0.1f;
			std::cin.clear();
			std::cout << "Default set." << std::endl << std::endl;
		}
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
	}
	// Input from user to define the number of layers of the perceptron
	std::cout << "How many layers do you need with inputs and outputs ? (default 3)" << std::endl;
	if (std::cin.peek() == '\n') { 
		neuronalNetwork.numberLayers = 3;
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
		std::cout << "Default set." << std::endl << std::endl;
    } else {
		std::cin >> neuronalNetwork.numberLayers;
		if (std::cin.fail() || neuronalNetwork.numberLayers <1) {
			neuronalNetwork.numberLayers = 3;
			std::cin.clear();
			std::cout << "Default set." << std::endl << std::endl;
		}
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}
	// Input from user to define the number of neurons by layers of the perceptron
	neuronalNetwork.neuronsByLayers = static_cast<int *>(malloc(neuronalNetwork.numberLayers * sizeof(int)));
	for (int i = 0 ; i<neuronalNetwork.numberLayers ; i++) {
		int numberNeurons;
		while (1) {
			std::cout << "How many neurons do you need for the layer " << i+1 << " ? (without bias)" << std::endl;
			std::cin >> numberNeurons;
			if (std::cin.good() && numberNeurons > 0) {
				neuronalNetwork.neuronsByLayers[i] = numberNeurons;
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				break;
			} else {
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			}
		}
	}
	// Input from user to define the use of bias by layers of the perceptron
	neuronalNetwork.biasByLayers = static_cast<bool *>(malloc(neuronalNetwork.numberLayers * sizeof(bool)));
	for (int i = 0 ; i<neuronalNetwork.numberLayers ; i++) {
		std::string bias;
		while (true) {
			if (i == neuronalNetwork.numberLayers -1) {
				neuronalNetwork.biasByLayers[i] = false;
				break;
			}
			std::cout << "Do you want to use a bias for the layer " << i+1 << " ? (yes / no - Default no)" << std::endl;
			if (std::cin.peek() == '\n') { 
				neuronalNetwork.biasByLayers[i] = false;
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
				std::cout << "Default set." << std::endl << std::endl;
				break;
			}
			std::cin >> bias;
			if (std::cin.good() && (bias.compare("yes") == 0 || bias.compare("y") == 0)) {
				neuronalNetwork.biasByLayers[i] = true;
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
				break;
			} else if (std::cin.good() && (bias.compare("no") == 0 || bias.compare("n") == 0)) {
				neuronalNetwork.biasByLayers[i] = true;
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
				break;
			} else {
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			}
		}
	}
	// Input from user to define the activation function
	neuronalNetwork.activationFunctionByLayers = static_cast<ActivationFunction *>(malloc(neuronalNetwork.numberLayers * sizeof(ActivationFunction)));
	neuronalNetwork.activationFunctionDerivByLayers = static_cast<ActivationDerivFunction *>(malloc(neuronalNetwork.numberLayers * sizeof(ActivationDerivFunction)));
	for (int i = 0 ; i<neuronalNetwork.numberLayers ; i++) {
		if ( i == 0 ) {
			neuronalNetwork.activationFunctionByLayers[i] = NONE;
			neuronalNetwork.activationFunctionDerivByLayers[i] = NONE_DERIV;
		} else {
			std::cout << "Which activation function do you need for the layer " << i+1 << " ? (default identity)" << std::endl;
			if (std::cin.peek() == '\n') { 
				activationFunctionEnumFromString("identity", i, neuronalNetwork.activationFunctionByLayers, neuronalNetwork.activationFunctionDerivByLayers);
				std::cout << activationFunctionStringFromEnum(neuronalNetwork.activationFunctionByLayers[i]) << " chosen " << std::endl;
			} else {
				std::string activationFunctionString;
				std::cin >> activationFunctionString;
				activationFunctionEnumFromString(activationFunctionString, i, neuronalNetwork.activationFunctionByLayers, neuronalNetwork.activationFunctionDerivByLayers);
				std::cout << activationFunctionStringFromEnum(neuronalNetwork.activationFunctionByLayers[i]) << " chosen " << std::endl;
			}
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
		}
	}
}

void getUserInputForInitializationOfPerceptronWeights(
	int &seed, 
	std::string &dist,
	float &paramOfDist1,
	float &paramOfDist2
){
	// Input from user to define the random engine
	while (true) {
		std::cout << "Do you want to use a seeded random engine ? (Int value - Default 0)" << std::endl;
		if (std::cin.peek() == '\n') { 
			seed = 0;
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
			std::cout << "Default set." << std::endl << std::endl;
			break;
		}
		int seedinput;
		std::cin >> seedinput;
		if (std::cin.good()) {
			seed = seedinput;
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
			break;
		} else {
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
	}
	// Input from user to define the distribution function to use for initialization of perceptron's weights
	while (true) {
		std::cout << "Which distribution function do you want to use ? (bernoulli / uniform / normal / poisson - Default normal)" << std::endl;
		if (std::cin.peek() == '\n') { 
			dist = "normal";
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
			std::cout << "Default set." << std::endl << std::endl;
			break;
		}
		std::string distinput;
		std::cin >> distinput;
		if (isImplementedDistributionFunction(distinput)) {
			dist = distinput;
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
			std::cout << dist << " set." << std::endl << std::endl;
			break;
		} else {
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
	}
	// Input from user to define the first parameter of the distribution function 
	while (true) {
		std::cout << "Define the first parameter of your distribution function (float expected - Default 0.)" << std::endl;
		if (std::cin.peek() == '\n') { 
			paramOfDist1 = 0.f;
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
			std::cout << "Default set." << std::endl << std::endl;
			break;
		}
		float param1input;
		std::cin >> param1input;
		if (std::cin.good()) {
			paramOfDist1 = param1input;
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			break;
		} else {
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
	}
	// Input from user to define the first parameter of the distribution function 
	while (true) {
		std::cout << "Define the second parameter of your distribution function (float expected - Default 0.1)" << std::endl;
		if (std::cin.peek() == '\n') { 
			paramOfDist2 = 0.1f;
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
			std::cout << "Default set." << std::endl << std::endl;
			break;
		}
		float param2input;
		std::cin >> param2input;
		if (std::cin.good()) {
			paramOfDist2 = param2input;
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			break;
		} else {
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
	}
}

void useOfTerminal(
	float &inertiaRate,
	float &learningRate,
	int &seed, 
	std::string &dist,
	float &paramOfDist1,
	float &paramOfDist2,
    network &neuronalNetwork
){
	getUserInputForNeuronalNetwork(inertiaRate, learningRate, neuronalNetwork);
	getUserInputForInitializationOfPerceptronWeights(seed, dist, paramOfDist1, paramOfDist2);
}
