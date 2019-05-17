
#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <algorithm>
#include <sstream>
#include <time.h>

#include "userInterface/csvParser.hpp"
#include "userInterface/jsonPerceptronArchitectureParser.hpp"
#include "userInterface/terminalUserInterface.hpp"
#include "userInterface/terminalFlags.hpp"

/**
 * @brief Get the command option object
 * 
 * @param begin 
 * @param end 
 * @param option 
 * 
 * @return char* 
 */
char* getCmdOption(char ** begin, char ** end, const std::string & option);

/**
 * @brief Check is a command option exists
 * 
 * @param begin 
 * @param end 
 * @param option 
 * 
 * @return true 
 * @return false 
 */
bool cmdOptionExists(char** begin, char** end, const std::string& option);

/**
 * @brief Main
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int EASNAmain(int argc, char **argv);

#endif