/*
 * CCrowdingDistanceComparator.h
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CCROWDINGDISTANCECOMPARATOR_H
#define CCROWDINGDISTANCECOMPARATOR_H

#include <CComparator.h>
/* This is class of crowding distance comparation */
template<typename TIndividual>
class CCrowdingDistanceComparator : public CComparator<TIndividual> {

public:
  int match(TIndividual * ind1, TIndividual * ind2){
  if (ind1 == nullptr)
    return 1;
  else if (ind2 == nullptr)
    return -1;

  double dist1 = ind1->crowdingDistance_;
  double dist2 = ind2->crowdingDistance_;
  if (dist1 >  dist2)
    return -1; //ind1 < ind2

  if (dist1 < dist2)
    return 1;  //ind1 > ind2

  return 0; // ind1 = ind2

};
};

#endif