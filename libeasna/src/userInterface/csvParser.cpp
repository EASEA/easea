/*

*/
#include "csvParser.hpp"

void readComputeCsvFile(const char * file, float ** & inputs) {
    // Define input file stream for reading weights file
    std::ifstream myComputeFile;
    myComputeFile.open(file);
    // Variable to get each line read by the getline function
    std::string line;
    // To skip the first line of csv file in which the column are described
    getline(myComputeFile, line);
    // Increment
    int i = 0;
    // For each input from csv file ...
    while(getline(myComputeFile, line)) {
        // A array is created and pushed in its container.
        std::string data;
        std::istringstream dataStream(line);
        // Increment
        int j = 0;
        while (getline(dataStream, data, ';')) {
            inputs[i][j] = stof(data);
            j++;
        }
        i++;
    }
    myComputeFile.close();
}

void readWeightsCsvFile(const char * file, network &n) {
    // Define input file stream for reading weights file
    std::ifstream myWeightFile;
    myWeightFile.open(file);
    // Variable to get each line read by the getline function
    std::string line;
    // To skip the first line of csv file in which the column are described
    getline(myWeightFile, line);
    // For each input from csv file ...
    while(getline(myWeightFile, line)) {
        int i = -1;
        int j = -1;
        int k = -1;
        // Parse the line in order to get all elements
        std::string data;
        std::istringstream dataStream(line);
        while (getline(dataStream, data, ';')) {
            if (i == -1) {
                i = stoi(data);
            } else if (j == -1) {
                j = stoi(data);
            } else if (k == -1) {
                k = stoi(data);
            } else {
                n.weights[i][j][k] = stof(data);
                i = -1;
                j = -1;
                k = -1;
            }
        }
    }
    myWeightFile.close();
}

void writeComputeCsvFile(const char * file, int numberOfInputs, float ** & inputs, int sizeOfInput, float ** & outputs,int sizeOfOutput) {
    std::ofstream myFile;
    myFile.open(file);
    // Set headers of csv file
    for(int i = 0 ; i < sizeOfInput; i++) {
        myFile << "input " << i << " ; ";
    }
    for(int i = 0 ; i < sizeOfOutput ; i++) {
        if (i == (sizeOfOutput -1))
            myFile << "output " << i << std::endl;
        else 
            myFile << "output " << i << " ; ";
    }
    //
    for(int i = 0 ; i <numberOfInputs ; i++) {
        for(int j = 0 ; j < sizeOfInput; j++) {
            myFile << inputs[i][j] << " ; ";
        }
        for(int j = 0 ; j < sizeOfOutput ; j++) {
            if (j == (sizeOfOutput -1))
                myFile << outputs[i][j] << std::endl;
            else 
                myFile << outputs[i][j] << " ; ";
        }
    }
    myFile.close();
}

void writeWeigthsCsvFile(const char * file, network &n) {
    std::ofstream myWeightsFile;
    myWeightsFile.open(file);
    // W [i] [j] [k] denotes the weight between layers i and i + 1 of the neurons j of the layer i + 1 and the neuron k of the layer i 
    myWeightsFile << "Between layer i and i+1 ; Neuron of layer i+1 ; Neuron of layer i ; Value" << std::endl;
	for (int i = 0; i < n.numberLayers -1; i++) {
		for(int j = 0; j < n.neuronsByLayers[i+1]; j++) {
			for(int k = 0; k < n.neuronsByLayers[i]; k++) {
                myWeightsFile << i <<";"<< j <<";"<< k <<";"<< n.weights[i][j][k] << std::endl;
            }
        }
    }
    myWeightsFile.close();
}

void writeErrorsCsvFile(const char * file, network &n) {
    std::ofstream myErrorsFile;
    myErrorsFile.open(file);
    myErrorsFile << "t;value" << std::endl;
    for(int i = 0 ; i < COST_EVOLUTION_LENGTH ; i++) {
        myErrorsFile << i <<";"<< n.costEvolution[i] << std::endl;
    }
    myErrorsFile.close();
}