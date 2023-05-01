/*
 * CVaribale.h
 *
 *  Created on: 24 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CVARIABLE_H
#define CVARIABLE_H

#include <typeinfo>
#include <string>

/* This is abstract class for defining variables */

class CVariable {

public:
  
  virtual ~CVariable() = 0;
  virtual const CVariable * deepCopy() = 0;
  virtual double getValue() const = 0;
  virtual void setValue(const double &value) = 0;
  virtual double getLowerBound() = 0;
  virtual double getUpperBound() = 0;
  void setLowerBound(double bound);
  void setUpperBound(double bound);
  virtual std::string getVariableType();

  virtual std::string toString()=0;

};
#endif

