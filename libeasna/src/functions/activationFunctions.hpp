
#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>


typedef float (*ActivationType) (const float x);
enum ActivationFunction {NONE, IDENTITY, SIGMOID, ARCTAN, SINUSOID, TANHYPER, RELU, GAUSSIAN, SOFTMAX};
enum ActivationDerivFunction {NONE_DERIV, IDENTITY_DERIV, SIGMOID_DERIV, ARCTAN_DERIV, SINUSOID_DERIV, TANHYPER_DERIV, RELU_DERIV, GAUSSIAN_DERIV, SOFTMAX_DERIV};

/**
 * @brief Print all activation functions implemeted
 * 
 */
void printActivationFunctionName();

/**
 * @brief From the name of the function, update the arrays that contain all these functions
 * 
 * @param s 
 * @param i
 * @param activationFunctionByLayers 
 * @param activationFunctionDerivByLayers 
 */
void activationFunctionEnumFromString(
	const std::string& s,
	const int i,
	ActivationFunction * &activationFunctionByLayers,
	ActivationDerivFunction * &activationFunctionDerivByLayers
);

/**
 * @brief Return the name of the activation function according a function reference
 * 
 * @param f Reference of the function
 * @return string Name of the function used
 */
std::string activationFunctionStringFromEnum(ActivationFunction af);

/**
 * @brief Identity function
 * 
 * @param x 
 * @return float 
 */
float identity(const float x);

/**
 * @brief Derivate of the identity function
 * 
 * @param x 
 * @return float 
 */
float identity_deriv(const float x);

/**
 * @brief Sigmoid function
 * 
 * @param x 
 * @return float 
 */
float sigmoid(const float x);

/**
 * @brief Derivate of the sigmoid function
 * 
 * @param x 
 * @return float 
 */
float sigmoid_deriv(const float x);

/**
 * @brief Arctan function
 * 
 * @param x 
 * @return float 
 */
float arctan(const float x);

/**
 * @brief Derivate of the arctan function
 * 
 * @param x 
 * @return float 
 */
float arctan_deriv(const float x);

/**
 * @brief Sinusoid function
 * 
 * @param x 
 * @return float 
 */
float sinusoid(const float x);

/**
 * @brief Derivate of the sinusoid function
 * 
 * @param x 
 * @return float 
 */
float sinusoid_deriv(const float x);

/**
 * @brief Hyperbolic tan function
 * 
 * @param x 
 * @return float 
 */
float tanhyper(const float x);

/**
 * @brief Derivate of the hyperbolic tan function
 * 
 * @param x 
 * @return float 
 */
float tanhyper_deriv(const float x);

/**
 * @brief Rectified linear unit function
 * 
 * @param x 
 * @return float 
 */
float relu(const float x);

/**
 * @brief Derivate of the rectified linear unit function
 * 
 * @param x 
 * @return float 
 */
float relu_deriv(const float x);

/**
 * @brief Gaussian function
 * 
 * @param x 
 * @return float 
 */
float gaussian(const float x);

/**
 * @brief Derivate of the gaussian function
 * 
 * @param x 
 * @return float 
 */
float gaussian_deriv(const float x);

/**
 * @brief Softmax function
 * 	Trick used to implement this. Just act as identity function because the softmax layer is computed in propagation
 * @param x 
 * @return float 
 */
float softmax(const float x);

/**
 * @brief Derivate of the softmax function
 * 	Trick used to implement this. Just act as identity function because the softmax layer is computed in propagation
 * @param x 
 * @return float 
 */
float softmax_deriv(const float x);

/**
 * @brief  Real Softmax function that compute the layer inplace
 *  Be carefull, the version implemented is the stable one where the max is used.
 * @param x
 * @param length
 */
void softmax_real(float * & x, int length);

#endif
