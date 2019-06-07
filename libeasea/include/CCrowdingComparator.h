/*
 * CCrowdingComparator.h
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CCROWDINGCOMPARATOR_H
#define CCROWDINGCOMPARATOR_H

#include "CComparator.h"
#include "CRankComparator.h"

/* This is class of comparator algorithm according NSGA-II */
template<typename TIndividual>
class CCrowdingComparator : public CComparator<TIndividual> {

private:
  CRankComparator<TIndividual> * c_;

public:
  CCrowdingComparator() : CComparator<TIndividual>() { c_ = new CRankComparator<TIndividual>();};
  ~CCrowdingComparator() { delete c_; };

  int match(TIndividual * ind1, TIndividual * ind2){
  if (ind1 == nullptr)
    return 1;
  else if (ind2 == nullptr)
    return -1;

  int flag = c_->match(ind1,ind2); // Ranking comparator

  if (flag != 0)
    return flag;

  /* if ranks are the same => it must be crowding distance comparation */
  double dist1 = ind1->crowdingDistance_;
  double dist2 = ind2->crowdingDistance_;
  if (dist1 >  dist2)
    return -1;

  if (dist1 < dist2)
    return 1;

  return 0;

};

};
#endif