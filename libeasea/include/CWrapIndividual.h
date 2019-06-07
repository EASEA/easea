/*
 * CWrapIndividual.h
 *
 *  Created on: 26 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CWRAPINDIVIDUAL_H
#define CWRAPINDIVIDUAL_H

#include "CIndividual.h"

/* Wrapper for different types of variables */
class CWrapIndividual {

public:
  CWrapIndividual(){};
  CWrapIndividual(CIndividual * individual){individual_ = individual;};
  inline double getValue (const int &index) { return individual_->getIndividualVariables()[index]->getValue(); };
  inline void setValue(const int &index, const double &value) { individual_->getIndividualVariables()[index]->setValue(value); };
  double getLowerBound (const int &index)  { return individual_->getIndividualVariables()[index]->getLowerBound(); };
  double getUpperBound (const int &index)  { return individual_->getIndividualVariables()[index]->getUpperBound(); };
  int getNumberOfIndividualVariables()  { return individual_->getNumberOfVariables(); };
  int size()  { return individual_->getNumberOfVariables(); };
  inline CIndividual * getIndividual()  { return individual_ ;};

private:
  CIndividual * individual_;
 

};

#endif //CWRAPINDIVIDUAL
