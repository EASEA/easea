/*
 * CDoubleType.h
 *
 *  Created on: 24 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */

#ifndef CDOUBLETYPE_H
#define CDOUBLETYPE_H

#include "CVariable.h"
#include "CPseudoRandom.h"
#include <sstream>

/* This class is for define Type Double for variables */
class CDoubleType : public CVariable {

public:

	CDoubleType(){ value_ = 0.0; };
  	CDoubleType(double lowerBound, double upperBound){
  		value_ = CPseudoRandom::randDouble(lowerBound, upperBound);
  		lowerBound_ = lowerBound;
		upperBound_ = upperBound;
	};
  	
	CDoubleType(CVariable * variable){
  		lowerBound_ = variable->getLowerBound();
		upperBound_ = variable->getUpperBound();
		value_      = variable->getValue();
	};

  	~CDoubleType(){};

  	inline const double getValue() const{return value_; } ;
  	inline void setValue (const double &value) { value_ = value; };
  	CVariable * deepCopy(){CDoubleType * cp = new CDoubleType(this); return cp;};
  	double getLowerBound()  { return lowerBound_; };
  	double getUpperBound()  { return upperBound_; };
  	void setLowerBound(const double &bound) { lowerBound_ = bound; };
  	void setUpperBound(const double &bound) { upperBound_ = bound; };
  	string toString(){  
		std::ostringstream stringStream;
  		stringStream << value_ ;
  		string aux = stringStream.str() + " ";

  		return aux ;
	} ;

private:
  double value_;
  double lowerBound_;
  double upperBound_;
};

#endif //CDOUBLETYPE_H
