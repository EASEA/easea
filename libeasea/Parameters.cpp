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

#include "include/Parameters.h"
#include <stdio.h>

#ifdef WIN32
Parameters::Parameters(){
}

Parameters::~Parameters(){
}
#endif

int Parameters::setReductionSizes(int popSize, float popReducSize){
        if(popReducSize<1.0 && popReducSize>=0.0)
                return (int)(popReducSize*popSize);
        if(popReducSize<0.0)
                return 0;
	if(popReducSize == 1.0)
		return popSize;
        if((int)popReducSize>popSize){
                printf("*WARNING* ReductionSize greater than PopulationSize !!!\n");
                printf("*WARNING* ReductionSize will be PopulationSize\n");
                return popSize;
        }
        else
                return (int)popReducSize;
}

