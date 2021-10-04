//=======================================================================
// Copyright (c) 2019 Romain Orhand
//=======================================================================

//=======================================================================
// Copyright (c) 2017 Adrian Schneider
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>

#include "mnist/mnist_reader.hpp"

std::string constructStringOutput(std::string value){
    if (value.compare("0") == 0) {
        return "1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0";
    } else if (value.compare("1") == 0) {
        return "0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0";
    } else if (value.compare("2") == 0) {
        return "0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0";
    } else if (value.compare("3") == 0) {
        return "0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0";
    } else if (value.compare("4") == 0) {
        return "0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0";
    } else if (value.compare("5") == 0) {
        return "0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0";
    } else if (value.compare("6") == 0) {
        return "0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0";
    } else if (value.compare("7") == 0) {
        return "0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0";
    } else if (value.compare("8") == 0) {
        return "0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0";
    } else if (value.compare("9") == 0) {
        return "0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 1";
    } 
    return "";
}

void readCsvFile(const char * file, std::vector<std::vector<float>> & inputs) {
    // Define input file stream for reading weights file
    std::ifstream myComputeFile;
    myComputeFile.open(file);
    // Variable to get each line read by the getline function
    std::string line;
    // To skip the first line of csv file in which the column are described
    getline(myComputeFile, line);
    // For each input from csv file ...
    while(getline(myComputeFile, line)) {
        // A vector is created and pushed in its container.
        std::vector<float> input;
        std::string data;
        std::istringstream dataStream(line);
        while (getline(dataStream, data, ';')) {
            input.push_back(stof(data));
        }
        inputs.push_back(input);
        input.clear();
    }
    myComputeFile.close();
}

void writeTrainingImageCsvFile(const char * file, const std::vector<std::vector<uint8_t>>& inputs, const std::vector<uint8_t> & outputs) {
    std::ofstream myFile;
    myFile.open(file);
    unsigned int numberOfInputs = inputs[0].size();
    // Set headers of csv file
    for(unsigned int i = 0 ; i < numberOfInputs; i++) {
        myFile << "input " << i << " ; ";
    }
    for(unsigned int i = 0 ; i < 10; i++) {
        if (i == (10 - 1))
            myFile << "output " << i << std::endl;
        else 
            myFile << "output " << i << " ; ";
    }
    //
    for(unsigned int i = 0 ; i < inputs.size() ; i++) {
        for(unsigned int j = 0 ; j < numberOfInputs; j++) {
            myFile << std::to_string(inputs[i][j]) << " ; ";
        }
        myFile << constructStringOutput(std::to_string(outputs[i])) << std::endl;
    }
    myFile.close();
}

void writeTestImageWithoutLabelCsvFile(const char * file, const std::vector<std::vector<uint8_t>>& inputs) {
    std::ofstream myFile;
    myFile.open(file);
    unsigned int numberOfInputs = inputs[0].size();
    // Set headers of csv file
    for(unsigned int i = 0 ; i < numberOfInputs; i++) {
        myFile << "input " << i << " ; ";
    }
    myFile << std::endl;
    //
    for(unsigned int i = 0 ; i < inputs.size() ; i++) {
        for(unsigned int j = 0 ; j < numberOfInputs; j++) {
            if (j == (numberOfInputs - 1))
                myFile << std::to_string(inputs[i][j]) << std::endl;
            else 
                myFile << std::to_string(inputs[i][j]) << " ; ";
        }
    }
    myFile.close();
}

void writeTestImageOnlyLabelCsvFile(const char * file, const std::vector<uint8_t> & outputs) {
    std::ofstream myFile;
    myFile.open(file);
    unsigned int numberOfInputs = outputs.size();
    // Set headers of csv file
    for(unsigned int i = 0 ; i < 10; i++) {
        if (i == (10 - 1))
            myFile << "output " << i << std::endl;
        else 
            myFile << "output " << i << " ; ";
    }
    //
    for(unsigned int i = 0 ; i < numberOfInputs ; i++) {
        myFile << std::to_string(outputs[i]) << std::endl;
    }
    myFile.close();
}

float computeScore(std::vector<std::vector<float>> labels, std::vector<std::vector<float>> computedValues) {
    unsigned int numberOfComputedValues = computedValues.size();
    float score = 0.f; 
    for(unsigned int i = 0 ; i < numberOfComputedValues ; i++) {
        std::vector<float>::iterator indexSearch = std::max_element(computedValues[i].end()-10, computedValues[i].end());
        int activatedNeuron = std::distance(computedValues[i].end()-10, indexSearch);
        if (activatedNeuron == (int) labels[i][0])
            score ++;
    }
    return score / numberOfComputedValues;
}

void learnAndScore(
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset,
    std::string pathProgram,
    std::string pathArchitecture,
    std::string pathWeights,
    int numberOfEpochs,
    float * results, 
    int clean
){
    std::vector<std::vector<float>> labels, computedValues ;
    for(int i = 0 ; i < numberOfEpochs ; i++) {
        char * command;
        if (i == 0) {
            std::string commandStr = pathProgram
                + std::string(" --parse ") + pathArchitecture
                + std::string(" --learn.batch TrainingImage.csv --batch.size 1 --batch.error sum ")
                + std::string("--save.weights ") + pathWeights + std::to_string(i) + std::string("_weights_archi_pass.csv");
            command = new char [commandStr.length()+1];
            strcpy(command, commandStr.c_str());
        } else {
            std::string commandStr = pathProgram
                + std::string(" --parse ") + pathArchitecture
                + std::string(" --update.weights ") + pathWeights + std::to_string(i-1) + std::string("_weights_archi_pass.csv")
                + std::string(" --learn.online TrainingImage.csv ")
                + std::string(" --save.weights ") + pathWeights + std::to_string(i) + std::string("_weights_archi_pass.csv");
            command = new char [commandStr.length()+1];
            strcpy(command, commandStr.c_str());
        }
        std::cout << std::endl << "Epoch n°" << std::to_string(i) << std::endl;
        system(command);
        delete [] command;
        // Create TestImage set
        writeTestImageWithoutLabelCsvFile("TestImage_withoutLabel_pass.csv", dataset.test_images);
        // Evaluation
        std::string commandStr = pathProgram
            + std::string(" --parse ") + pathArchitecture
            + std::string(" --update.weights ") + pathWeights + std::to_string(i) + std::string("_weights_archi_pass.csv")
            + std::string(" --compute TestImage_withoutLabel_pass.csv");
        command = new char [commandStr.length()+1];
        strcpy(command, commandStr.c_str());
        system(command);
        delete [] command;
        // Compute score
        readCsvFile("TestImage_OnlyLabel.csv", labels);
        readCsvFile("TestImage_withoutLabel_pass.csv", computedValues);
        results[i] = computeScore(labels, computedValues);
        std::cout << computeScore(labels, computedValues) << std::endl;
        labels.clear();
        computedValues.clear();
        system("rm TestImage_withoutLabel_pass.csv");
    }
    // If there is only one arg, mnist weights are deleted, else they aren't and we write again the evaluation file.
    if (clean == 1) {
        for(int i = 0 ; i < numberOfEpochs ; i++) {
            std::string cmdStr = std::string("rm ./") + std::to_string(i) + std::string("_weights_archi_pass.csv");
            char * cmd = new char [cmdStr.length()+1];
            strcpy(cmd, cmdStr.c_str());
            system(cmd);
            delete [] cmd;
        }
    } else {
        writeTestImageWithoutLabelCsvFile("TestImage.csv", dataset.test_images);
    }
}

inline bool almostEqual(float x, float y) {
    const float epsilon = std::numeric_limits<float>::epsilon();
    return std::abs(x - y) <= epsilon * std::abs(x);
}

int main(int argc, char* argv[]) {
    (void) argv;
    // Load MNIST data
     mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    // Prepare data
    writeTrainingImageCsvFile("TrainingImage.csv", dataset.training_images, dataset.training_labels);
    writeTestImageOnlyLabelCsvFile("TestImage_OnlyLabel.csv", dataset.test_labels);

    // Compute on Mnist datasets global accuracy and compare...
    const int numberOfEpochs = 6;
    int mnistTestScore = 0;
    float * mnistResults = NULL;
    mnistResults = static_cast<float*>(malloc(numberOfEpochs * sizeof(float)));
    learnAndScore( dataset,"./../../easena", "./architecture_mnist.nz", "./", numberOfEpochs, mnistResults, argc);
    float expectedMnistResults[6];
    expectedMnistResults[0] = 0.8482f ; 
    expectedMnistResults[1] = 0.883f ; 
    expectedMnistResults[2] = 0.894f ; 
    expectedMnistResults[3] = 0.9023f ; 
    expectedMnistResults[4] = 0.9135f ; 
    expectedMnistResults[5] = 0.917f ;
    for(int i = 0; i < numberOfEpochs; i++) {
       if (almostEqual(expectedMnistResults[i],mnistResults[i])) mnistTestScore++;
    }
    // For personal debug, do not use it
    // std::cout<< std::endl << "Mnist regression test score is " << std::to_string(mnistTestScore) << " / " << std::to_string(numberOfEpochs) << std::endl;

    // Testing memory leaks : openmp and valgrind have issues if used together
    #if defined(__linux__)
        system("valgrind --log-file='Testing_memory_leaks_mnist.txt' ./../../easena --parse ./architecture_xor3.nz --learn.online dataset_xor3.csv --save.weights weights.csv ");
        if (system("grep --silent 'All heap blocks were freed -- no leaks are possible' Testing_memory_leaks_mnist.txt") == 0 && system("grep --silent 'ERROR SUMMARY: 0 errors from 0 contexts' Testing_memory_leaks_mnist.txt") == 0) {
            std::cout << "No memory leaks or errors detected in online learning process !" << std::endl;
            system("rm Testing_memory_leaks_mnist.txt");
        } else {
            std::cout << "Warning : memory leaks or errors detected in online learning process !" << std::endl;
        }

        system("valgrind --log-file='Testing_memory_leaks_mnist.txt' ./../../easena --parse ./architecture_xor3.nz --learn.batch dataset_xor3.csv --batch.size 1 --batch.error average ");
        if (system("grep --silent 'All heap blocks were freed -- no leaks are possible' Testing_memory_leaks_mnist.txt") == 0 && system("grep --silent 'ERROR SUMMARY: 0 errors from 0 contexts' Testing_memory_leaks_mnist.txt") == 0) {
            std::cout << "No memory leaks or errors detected in batch learning process !" << std::endl << std::endl;
            system("rm Testing_memory_leaks_mnist.txt");
        } else {
            std::cout << "Warning : memory leaks or errors detected in batch learning process !" << std::endl;
        }

        system("valgrind --log-file='Testing_memory_leaks_mnist.txt' ./../../easena --parse ./architecture_xor3.nz --update.weights weights.csv");
        if (system("grep --silent 'All heap blocks were freed -- no leaks are possible' Testing_memory_leaks_mnist.txt") == 0 && system("grep --silent 'ERROR SUMMARY: 0 errors from 0 contexts' Testing_memory_leaks_mnist.txt") == 0) {
            std::cout << "No memory leaks or errors detected in updating weights process !" << std::endl;
            system("rm Testing_memory_leaks_mnist.txt");
        } else {
            std::cout << "Warning : memory leaks or errors detected in updating weights process !" << std::endl;
        }
        system("rm weights.csv");
    #endif
    // Free memory
    free(mnistResults);

    // Delete data if no args
    if (argc == 1) {
        system("rm TrainingImage.csv");
        system("rm TestImage_OnlyLabel.csv");
    }
    // Return execution time
    std::cout << std::endl;
    return 0;
}
