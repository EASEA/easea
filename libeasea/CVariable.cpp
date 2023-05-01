/*
 * CVaribale.cpp
 *
 *  Created on: 24 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */

#include "CVariable.h"
#include <cstddef>


// destructor
CVariable::~CVariable() {  }


// Gets the type of the variable
std::string CVariable::getVariableType() {
  return typeid(this).name() ;
}
