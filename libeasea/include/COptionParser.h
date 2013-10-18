/*
 *    Copyright (C) 2009  Ogier Maitre

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef COPTIONPARSER_H_
#define COPTIONPARSER_H_


void parseArguments(const char* parametersFileName, int ac, char** av);
int setVariable(const std::string optionName, int defaultValue);
float setVariable(const std::string optionName, float defaultValue);
std::string setVariable(const std::string optionName, std::string defaultValue);



#endif /* COPTIONPARSER_H_ */
