/*
 * CStoppingCriterion.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include "include/CStoppingCriterion.h"
#include <iostream>
#include "include/CEvolutionaryAlgorithm.h"

/* ****************************************
   GenerationalCriterion class
****************************************/
CGenerationalCriterion::CGenerationalCriterion(size_t generationalLimit){
  this->generationalLimit = generationalLimit;
}

void CGenerationalCriterion::setCounterEa(size_t* ea_counter){
	this->currentGenerationPtr = ea_counter;
}

bool CGenerationalCriterion::reached(){
  if( generationalLimit <= *currentGenerationPtr ){
    std::cout << "Current generation " << *currentGenerationPtr << " Generational limit : " <<
      generationalLimit << std::endl;
    return true;
  }
  else return false;
}

size_t* CGenerationalCriterion::getGenerationalLimit(){
	return &(this->generationalLimit);
}
