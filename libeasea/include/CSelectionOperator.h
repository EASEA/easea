/*
 * CSelectionOperator.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef CSELECTIONOPERATOR_H_
#define CSELECTIONOPERATOR_H_

#include <stdlib.h>
#include "CIndividual.h"

class CSelectionOperator {
public:
  virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  virtual float getExtremum() = 0 ;
protected:
  CIndividual** population;
  float currentSelectionPressure;
};


/* ****************************************
   Tournament classes (min and max)
****************************************/
class MaxTournament : public CSelectionOperator{
public:
  MaxTournament(CRandomGenerator* rg){ this->rg = rg; }
  virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
private:
  CRandomGenerator* rg;

};



class MinTournament : public CSelectionOperator{
public:
  MinTournament(CRandomGenerator* rg){ this->rg = rg; }
  virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
private:
  CRandomGenerator* rg;

};


class MaxDeterministic : public CSelectionOperator{
 public:
  virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;
};

class MinDeterministic : public CSelectionOperator{
 public:
  virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;

};


class MaxRandom : public CSelectionOperator{
 public:
  MaxRandom(CRandomGenerator* globalCRandomGenerator);
  virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;
  CRandomGenerator* rg;

};

class MinRandom : public CSelectionOperator{
 public:
  MinRandom(CRandomGenerator* globalCRandomGenerator);
  virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;
  CRandomGenerator* rg;
};



#endif /* CSELECTIONOPERATOR_H_ */
