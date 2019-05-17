
#ifndef CSVPARSER_H
#define CSVPARSER_H

#include <fstream>
#include <iostream>

#include "../propagation.hpp"

/**
 * @brief Read the csv file that contains all values to compute with the neuronal network and update the vecotr of all inputs
 * 
 * @param file 
 * @param inputs 
 */
void readComputeCsvFile(const char * file, float ** & inputs);

/**
 * @brief Read the csv file that contains some weight values and update the neuronal network
 * 
 * @param file 
 * @param n 
 */
void readWeightsCsvFile(const char * file, network &n);

/**
 * @brief Write in the compute csv file the output for each input value returned by the neuronal network
 * 
 * @param file 
 * @param numberOfInputs
 * @param inputs 
 * @param sizeOfInput
 * @param outputs 
 * @param sizeOfOutput
 */
void writeComputeCsvFile(const char * file, int numberOfInputs, float ** & inputs, int sizeOfInput, float ** & outputs,int sizeOfOutput);

/**
 * @brief Write in the a csv file all the weight value from a neuronal network
 * 
 * @param file 
 * @param n 
 */
void writeWeigthsCsvFile(const char * file, network &n);

/**
 * @brief To rewrite
 * @deprecated
 * @param file 
 * @param n 
 */
void writeErrorsCsvFile(const char * file, network &n);

#endif