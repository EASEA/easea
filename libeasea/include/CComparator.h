/*
 * CCopmarator.h
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */

/* This is a base class for comparator 
 * It's useful for having abstraction for different
 * type of comparators. 
 * For example for sorting population.
 */

#ifndef CCOMPARATOR_H
#define CCOMPARATOR_H
template<typename T>
class CComparator {

public:
  virtual ~CComparator(){};
  virtual int match(T * ind1, T * ind2) = 0; // pure virtual function for comparing individuals

};

#endif
