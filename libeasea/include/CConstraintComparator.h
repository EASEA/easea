
/*
 * CConstraintComparator .h
 *
 *  Created on: 2 aout 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CCONSTRAINTCOMPARATOR
#define CCONSTRAINTCOMPARATOR

#include <CComparator.h>

/* This class for constraint comparating */

template <typename TIndividual>
class CConstraintComparator : public CComparator<TIndividual> {

public:
  int match(TIndividual * o1,TIndividual * o2){
  double val1, val2;
  val1 = o1->getConstraint();
  val2 = o2->getConstraint();

  if ((val1 < 0) && (val2 < 0)) {
    if (val1 > val2){
      return -1;
    } else if (val2 > val1){
      return 1;
    } else {
      return 0;
    }
  } else if ((val1 == 0) && (val2 < 0)) {
    return -1;
  } else if ((val1 < 0) && (val2 == 0)) {
    return 1;
  } else {
    return 0;
  }

};

};

#endif