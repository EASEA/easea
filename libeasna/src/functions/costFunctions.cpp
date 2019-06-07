/*
List of possible cost functions
*/
#include "costFunctions.hpp"

void printCostFunctionName() {
	std::cout << "The implemented functions are :" << std::endl;
	std::cout << "     - Cross Entropy : Use softmax as activation function on the last layer." << std::endl;
	std::cout << "     - Mean Squarred Error : Use another activation function on the last layer." << std::endl;
}

CostType costMatch(const std::string s) {
	if (s.compare("softmax") == 0) {
		return crossEntropy;
	} else { 
		return meanSquarredError;
	}
}

float crossEntropy(float * &expected, float * &outputs, int length){
	float result = 0.f;
	for (int i = 0 ; i<length ; i++) {
		result -= expected[i] * std::log(outputs[i]);
	}
	return result;
}

float meanSquarredError(float * &expected, float * &outputs, int length){
	float result = 0.f;
	for (int i = 0 ; i<length ; i++) {
		result += (expected[i] - outputs[i]) * (expected[i] - outputs[i]) /2.f;
	}
	return result;
}