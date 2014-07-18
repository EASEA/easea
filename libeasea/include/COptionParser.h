/*
 * COptionParser.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */
/**
 * @file COptionParser.h
 * @author SONIC BFO, Ogier Maitre
 * @version 1.0
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation; either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Affero General Public License for more details at
 * http://www.gnu.org/licenses/
**/  


#ifndef COPTIONPARSER_H_
#define COPTIONPARSER_H_


/**
* \brief    Parse the arguments, from commandline  and file
* \details  Parse easea defined option (--nbGen,etc ..) from cmdline and from
*           prm files
*
* @param    parametersFileName  Name of the prm file (relative path)
* @param    ac                  Number of option from cmdline 
* @param    av                  Option list from cmdline
**/
void parseArguments(const char* parametersFileName, int ac, char** av);

/**
* \brief    Set option variable value
* \details  The value of the option is an integer. if no value for this variable
*           is found in the file or in the cmdline then defaultValue is used
*
* @param    optionName    The option name
* @param    defaultValue  Value to set by defaut 
* @return   var           The value that was used
**/
int setVariable(const std::string optionName, int defaultValue);

/**
* \brief    Set option variable value
* \details  The value of the option is a float. if no value for this variable
*           is found in the file or in the cmdline then defaultValue is used
*
* @param    optionName    The option name
* @param    defaultValue  Value to set by defaut 
* @return   var           The value that was used
**/
float setVariable(const std::string optionName, float defaultValue);

/**
* \brief    Set option variable value
* \details  The value of the option is a string. if no value for this variable
*           is found in the file or in the cmdline then defaultValue is used
*
* @param    optionName    The option name
* @param    defaultValue  Value to set by defaut 
* @return   var           The value that was used
**/
std::string setVariable(const std::string optionName, std::string defaultValue);



#endif /* COPTIONPARSER_H_ */
