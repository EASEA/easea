/*
 * CRankComparator.h
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CRANKCOMPARATOR_H
#define CRANKCOMPARATOR_H

#include <CComparator.h>

/* This is class of rank comparation */

template<typename TIndividual>
class CRankComparator : public CComparator<TIndividual> {

public:

         
  int match(TIndividual * ind1, TIndividual * ind2){
  if (ind1 == nullptr)
    return 1;
  else if (ind2 == nullptr)
    return -1;


  if (ind1->getRank() < ind2->getRank())
    return -1;

  if (ind1->getRank() > ind2->getRank())
    return 1;

  return 0;

};

};

#endif
