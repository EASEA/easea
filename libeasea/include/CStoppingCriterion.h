/*
 * CStoppingCriterion.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef CSTOPPINGCRITERION_H_
#define CSTOPPINGCRITERION_H_

#include <stdlib.h>
class CEvolutionaryAlgorithm;

/* ****************************************
   StoppingCriterion class
****************************************/
class CStoppingCriterion {

public:
  virtual bool reached() = 0;

};


/* ****************************************
   GenerationalCriterion class
****************************************/
class CGenerationalCriterion : public CStoppingCriterion {
 private:
  size_t* currentGenerationPtr;
  size_t generationalLimit;
 public:
  virtual bool reached();
  CGenerationalCriterion(size_t generationalLimit);
  void setCounterEa(size_t* ea_counter);
  size_t *getGenerationalLimit();
};

#endif /* CSTOPPINGCRITERION_H_ */
