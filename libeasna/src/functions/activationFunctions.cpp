/*
List of possible activation functions
*/
#include "activationFunctions.hpp"

void printActivationFunctionName() {
	std::cout << "The implemented functions are :" << std::endl;
	std::cout << "     - identity" << std::endl;
	std::cout << "     - sigmoid" << std::endl;
	std::cout << "     - arctan" << std::endl;
	std::cout << "     - sinusoid" << std::endl;
	std::cout << "     - tanhyper" << std::endl;
	std::cout << "     - relu" << std::endl;
	std::cout << "     - gaussian" << std::endl;
	std::cout << "     - softmax" << std::endl;
}

void activationFunctionEnumFromString(
	const std::string s, 
	const int i,
	ActivationFunction * &activationFunctionByLayers,
	ActivationDerivFunction * &activationFunctionDerivByLayers
) {
	if (s.compare("identity") == 0) {
		activationFunctionByLayers[i] = IDENTITY;
		activationFunctionDerivByLayers[i] = IDENTITY_DERIV;
	} else if (s.compare("sigmoid") == 0) {
		activationFunctionByLayers[i] = SIGMOID;
		activationFunctionDerivByLayers[i] = SIGMOID_DERIV;
	} else if (s.compare("arctan") == 0) {
		activationFunctionByLayers[i] = ARCTAN;
		activationFunctionDerivByLayers[i] = ARCTAN_DERIV;
	} else if (s.compare("sinusoid") == 0) {
		activationFunctionByLayers[i] = SINUSOID;
		activationFunctionDerivByLayers[i] = SINUSOID_DERIV;
	}  else if (s.compare("tanhyper") == 0) {
		activationFunctionByLayers[i] = TANHYPER;
		activationFunctionDerivByLayers[i] = TANHYPER_DERIV;
	} else  if (s.compare("relu") == 0) {
		activationFunctionByLayers[i] = RELU;
		activationFunctionDerivByLayers[i] = RELU_DERIV;
	} else  if (s.compare("gaussian") == 0) {
		activationFunctionByLayers[i] = GAUSSIAN;
		activationFunctionDerivByLayers[i] = GAUSSIAN_DERIV;
	} else  if (s.compare("softmax") == 0) {
		activationFunctionByLayers[i] = SOFTMAX;
		activationFunctionDerivByLayers[i] = SOFTMAX_DERIV;
	} else {
		activationFunctionByLayers[i] = NONE;
		activationFunctionDerivByLayers[i] = NONE_DERIV;
	}
}

std::string activationFunctionStringFromEnum(ActivationFunction af)
{
	if (af == IDENTITY)  {
		return "identity";
	} 
	if (af == SIGMOID)  {
		return "sigmoid";
	}
	if (af == ARCTAN)  {
		return "arctan";
	}
	if (af == SINUSOID)  {
		return "sinusoid";
	} 
	if (af == TANHYPER)  {
		return "tanhyper";
	} 
	if (af == RELU)  {
		return "relu";
	} 
	if (af == GAUSSIAN)  {
		return "gaussian";
	} 
	if (af == SOFTMAX)  {
		return "softmax";
	}
	return "null";
}

float identity(const float x) {
	return x;
}

float identity_deriv(const float x) {
	(void) x;
	return 1.f;
}

float sigmoid(const float x) {
	return 0.5f + 0.5f * std::tanh(x / 2.f);
}

float sigmoid_deriv(const float x) {
	float tmp = sigmoid(x);
	return tmp * (1.f - tmp);
}

float arctan(const float x) {
	return std::atan(x);
}

float arctan_deriv(const float x) {
	float tmp = std::atan(x);
	return 1.f / (1.f + tmp*tmp);
}

float sinusoid(const float x) {
	return std::sin(x);
}

float sinusoid_deriv(const float x) {
	return std::cos(x);
}

float tanhyper(const float x) {
	return std::tanh(x);
}

float tanhyper_deriv(const float x) {
	float tmp = std::tanh(x);
	return 1.f - tmp*tmp;
}

float relu(const float x) {
	if (x < 0.f)
		return 0.f;
	else
		return x;
}

float relu_deriv(const float x) {
	if (x < 0.f)
		return 0.f;
	else
		return 1.f;
}

float gaussian(const float x) {
	return std::exp(-x * x);
}

float gaussian_deriv(const float x) {
	return -2.f * x * std::exp(-x * x);
}

float softmax(const float x) {
	return x;
}

float softmax_deriv(const float x) {
	(void) x;
	return 1.f;
}

void softmax_real(float * & x, int length) {
	float max = x[0];
	// Search for the max in computed_combination
	for (int i = 1; i < length; i++) {
		if (max < x[i]) max = x[i];
	}
	// Compute the exp function on each values
	float sum_acc = 0.f;
	for(int i = 0; i < length; i++) {
		float exp_calcul = std::exp(x[i] - max);
		x[i] = exp_calcul;
		sum_acc += exp_calcul;
	}
	// Divide by the sum of exp
	for(int i = 0; i < length; i++) {
		x[i] /= sum_acc;
	}
}