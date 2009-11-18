/*
 * COptionParser.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef COPTIONPARSER_H_
#define COPTIONPARSER_H_


void parseArguments(const char* parametersFileName, int ac, char** av);
int setVariable(const std::string optionName, int defaultValue);
float setVariable(const std::string optionName, float defaultValue);
std::string setVariable(const std::string optionName, std::string defaultValue);



#endif /* COPTIONPARSER_H_ */
