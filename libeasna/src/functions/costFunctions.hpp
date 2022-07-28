
#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include <cmath>
#include <functional>
#include <iostream>
#include <string>

typedef float (*CostType) (float * &expected, float * &outputs, int length);

/**
 * @brief Print all coST functions implemented
 * 
 */
void printCostFunctionName();

/**
 * @brief From the name of the function, return the reference of the cost function
 * 
 * @param s 
 * @return fucntion pointer
 */
CostType costMatch(const std::string& s);

/**
 * @brief Compute error through cross entropy formula 
 * 
 * @param expected
 * @param outputs
 * @return float 
 */
float crossEntropy(float * &expected, float * &outputs, int length);

/**
 * @brief Compute error through cross entropy formula 
 * 
 * @param expected
 * @param outputs
 * @return float 
 */
float meanSquarredError(float * &expected, float * &outputs, int length);

#endif
