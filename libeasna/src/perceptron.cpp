#include "perceptron.hpp"

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
	char ** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return find(begin, end, option) != end;
}

int EASNAmain(int argc, char **argv)
{
	// Execution Time
	clock_t tStart = clock();
	// Parameters needed
	float inertiaRate;
	float learningRate;
	int seed;
	std::string distributionFunction;
	float paramOfDist1;
	float paramOfDist2;
	network neuronalNetwork;
	// Temporary variable for control
	bool initComput = false;
	bool verbose = cmdOptionExists(argv, argv+argc, "--verbose");
	// Scope of code where the program returns the list of commands, flags and pieces of information that can be used 
	if (cmdOptionExists(argv, argv+argc, "--help")) {
		std::cout << "Add '--help.activation' to get the name of the implemented activation functions. "  << std::endl;
		std::cout << "Add '--help.cost' to get the name of the implemented cost functions. "  << std::endl;
		std::cout << "Add '--help.examples' to get some classical examples of use of this program." << std::endl;
		std::cout << "Add '--help.randomDistribution' to get the name of the implemented distribution functions for rondomization."  << std::endl << std::endl;
		std::cout << "Add '--batch.size' to define the size of batch used in batch retropropagation."  << std::endl;
		std::cout << "Add '--batch.error' to define the way the error can be considered (average or sum) used in batch retropropagation."  << std::endl;
		std::cout << "Add '--compute' to compute inputs from a csv file into the perceptron." << std::endl;
		std::cout << "Add '--learn.online' to start learning process only if the perceptron has been initialized, that is to say '--term' or 'parse' have been added and have run successfully. A well-formed csv file is expected in input and you should have initialized your neuronal network through '--term' or '--parse'."  << std::endl;
		std::cout << "Add '--learn.batch' to start learning process only if the perceptron has been initialized, that is to say '--term' or 'parse' have been added and have run successfully. A well-formed csv file is expected in input and you should have initialized your neuronal network through '--term' or '--parse'. '--batch.size' and '--batch.error' have to be defined."  << std::endl;
		std::cout << "Add '--parse' to get perceptron parameters through a json file."  << std::endl;
		std::cout << "Add '--save.architecture filename' to get perceptron parameters through the terminal and save it in a json file. Could only be used with '--term' option."  << std::endl;
		std::cout << "Add '--save.errors' to store the error matrix in a csv file. You should have initialized your neuronal network through '--term' or '--parse'."  << std::endl;
		std::cout << "Add '--save.weights' to store the 3D weight matrix in a csv file. You should have initialized your neuronal network through '--term' or '--parse'."  << std::endl;
		std::cout << "Add '--term' to get perceptron parameters through the terminal."  << std::endl;
		std::cout << "Add '--update.weights' to update weights from a csv file."  << std::endl;
		std::cout << "Add '--verbose' to get more information in terminal."  << std::endl;
	}
	if (cmdOptionExists(argv, argv+argc, "--help.activation")) {
		printActivationFunctionName();
	}
	if (cmdOptionExists(argv, argv+argc, "--help.cost")) {
		printCostFunctionName();
	}
	if (cmdOptionExists(argv, argv+argc, "--help.examples")) {
		std::cout << "./easena --parse data/architecture.nz --learn.batch data/learn.csv --batch.size 32 --batch.error average --save.weights data/weights.csv" << std::endl;
		std::cout << "./easena --parse data/architecture.nz --learn.online data/learn.csv --save.weights data/weights.csv"  << std::endl;
		std::cout << "./easena --term --save.architecture data/architecture.nz"  << std::endl;
	}
	if (cmdOptionExists(argv, argv+argc, "--help.randomDistribution")) {
		printDistributionFunctionName();
	}

	// Scope of code where perceptron architecture is defined
	if (cmdOptionExists(argv, argv+argc, "--term")) {
		useTerminalToGetArchitecture(inertiaRate, learningRate, seed, distributionFunction, paramOfDist1, paramOfDist2, neuronalNetwork, initComput);		
		// echo state of neuronal network
		if (verbose) printNeuronalNetwork(neuronalNetwork);
		if (cmdOptionExists(argv, argv+argc, "--save.architecture")) {
			char * jsonInput = getCmdOption(argv, argv + argc, "--save.architecture");
			if (jsonInput) {
				saveJsonPerceptronArchitecture(jsonInput, inertiaRate, learningRate, seed, distributionFunction, paramOfDist1, paramOfDist2, neuronalNetwork);
			} else {
				std::cout << "Error when getting the json output file." << std::endl;
				exit(1);
			}
		}
	} else 	if (cmdOptionExists(argv, argv+argc, "--parse")) {
		char * jsonInput = getCmdOption(argv, argv + argc, "--parse");
		if (jsonInput) {
			parseJsonPerceptronArchitecture(jsonInput, inertiaRate, learningRate, seed, distributionFunction, paramOfDist1, paramOfDist2, neuronalNetwork, initComput);
		// echo state of neuronal network
		if (verbose) printNeuronalNetwork(neuronalNetwork);
		} else {
			std::cout << "Error when getting the json input file." << std::endl;
			exit(1);
		}
	}

	// Scope of code where the perceptron weights are updated
	if (initComput && cmdOptionExists(argv, argv+argc, "--update.weights")) {
		char * weightsFile = getCmdOption(argv, argv + argc, "--update.weights");
		if (weightsFile) {
			readWeightsCsvFile(weightsFile, neuronalNetwork);
			if (verbose) printWeightsOfNeuronalNetwork(neuronalNetwork);
		} else {
			std::cout << "Error when getting the weights csv file." << std::endl;
			exit(1);
		}
	}

	// Allows users to perform online gradient retropropagation on perceptron
	if (initComput && cmdOptionExists(argv, argv+argc, "--learn.online")) {
		char * learnFile = getCmdOption(argv, argv + argc, "--learn.online");
		if (learnFile) {
			learnOnlineFromFile(learnFile, inertiaRate, learningRate, neuronalNetwork);
		} else {
			std::cout << "Error when getting the learning dataset file." << std::endl;
			exit(1);
		}
	}

	// Allows users to perform batch gradient retropropagation on perceptron
	if (initComput && cmdOptionExists(argv, argv+argc, "--learn.batch")) {
		char * learnFile = getCmdOption(argv, argv + argc, "--learn.batch");
		if (learnFile) {
			int batchSize = 0;
			char * batchErrorMethod = nullptr;
			// Get the batch size for batch retropropagation
			if (cmdOptionExists(argv, argv+argc, "--batch.size")) {
				try {
					batchSize = std::stoi(getCmdOption(argv, argv + argc, "--batch.size"));
				} catch (...) {
					std::cout << "Error : --batch.size has to be defined with an integer." << std::endl;
					exit(1);
				}
			}
			// Get the method used on error for batch retropropagation
			if (cmdOptionExists(argv, argv+argc, "--batch.error")) {
				batchErrorMethod = getCmdOption(argv, argv + argc, "--batch.error");
				if (strcmp(batchErrorMethod, "average") && strcmp(batchErrorMethod, "sum")) {
					std::cout << "Error : --batch.error has to be defined with 'average' or 'sum'." << std::endl;
					exit(1);
				}
			}
			// Compute learning through batch retropropagation
			if (batchSize && batchErrorMethod) {
				learnBatchFromFile(learnFile, batchSize, batchErrorMethod, inertiaRate, learningRate, neuronalNetwork);
			} else {
				std::cout << "Error when trying batch learning. Check batch size and batch error method." << std::endl;
				exit(1);
			}
		} else {
			std::cout << "Error when trying batch learning. Check csv learning file, batch size and batch error method." << std::endl;
			exit(1);
		}
	}

	// Allows users to get the output of the perceptron according to the given input
	if (initComput && cmdOptionExists(argv, argv+argc, "--compute")) {
		char * csvCompute = getCmdOption(argv, argv + argc, "--compute");
		if (csvCompute) {
			int count = 0;
			std::string line;
			/* Creating input filestream */ 
			std::ifstream file(csvCompute);
			while (getline(file, line))
				count++;
			file.close();
			computeUserInputFile(csvCompute, count-1, neuronalNetwork);
		} else {
			std::cout << "Error when getting the csv compute file." << std::endl;
			exit(1);
		}
	}

	// Allows users to store the computed weights in the file given in argument
	if (initComput && cmdOptionExists(argv, argv+argc, "--save.weights")) {
		char * csvInput = getCmdOption(argv, argv + argc, "--save.weights");
		if (csvInput) {
			writeWeigthsCsvFile(csvInput, neuronalNetwork);
		} else {
			std::cout << "Error when getting the weights csv output file." << std::endl;
			exit(1);
		}
	}

	// To refactor later
	if (initComput && cmdOptionExists(argv, argv+argc, "--save.errors")) {
		char * csvInput = getCmdOption(argv, argv + argc, "--save.errors");
		if (csvInput) {
			writeErrorsCsvFile(csvInput, neuronalNetwork);
		} else {
			std::cout << "Error when getting the errors csv output file." << std::endl;
			exit(1);
		}
	}

	// Free every malloc
	freeNetwork(neuronalNetwork);

	// Return execution time
	printf("Total time taken: %.2fs\n", static_cast<double>(clock() - tStart)/CLOCKS_PER_SEC);
	return 0;
}

//  valgrind --leak-check=yes ./perceptron --parse data/architecture.json --learn.batch data/learn.csv --batch.size 32 --batch.error average --save.weights data/weights.csv

//  valgrind --leak-check=yes ./perceptron --parse data/architecture.json --learn.online data/learn.csv --save.weights data/weights.csv

//  valgrind --leak-check=yes ./perceptron --parse data/architecture.json --update.weights data/weights.csv --compute data/inputs.csv

// ./perceptron --parse data/architecture.json  --update.weights data/mninst/weights_archi1_pass4.csv --learn.online data/mninst/TrainingImage.csv --save.weights data/mninst/weights_archi1_pass5.csv   

//  ./perceptron --parse data/architecture.json --update.weights data/mninst/weights_archi1_pass1.csv  --compute data/mninst/TestImage_withoutLabel_archi1_pass1.csv