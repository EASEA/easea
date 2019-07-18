/*
 * CIndividual.cpp
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#include "CIndividual.h"
#include "CDoubleType.h"
#include <CLogger.h>

CIndividual::CIndividual() : upperBound_(nullptr),
                            lowerBound_(nullptr),
                            objective_(nullptr),
                            variable_(nullptr)
{
  // TODO Auto-generated constructor stub

}
CIndividual::CIndividual(CIndividual *genome) {

}

CIndividual::~CIndividual() {
  // TODO Auto-generated destructor stub
}

void CIndividual::setObjective(int index, double val) {
    if (index < 0 || index >= nbObj_)
	LOG_ERROR(errorCode::value, "Index of CIndividual is out range");

    objective_[index] = val;
}

void CIndividual::setAllObjectives(double* __restrict out, double* __restrict in){
    for (auto i = 0; i < nbObj_; i++)
        out[i] = in[i];
}

void CIndividual::setAllVariables(CVariable ** __restrict out, CVariable ** __restrict in){
    for (auto i = 0; i < nbVar_; i++)
        out[i] = (CDoubleType*)in[i]->deepCopy();
}

double CIndividual::getObjective(const int index) {
//    if (index < 0 || index >= nbObj_)
//        LOG_ERROR(errorCode::value, "index of CIndividual is out of range");
    return objective_[index];
}

