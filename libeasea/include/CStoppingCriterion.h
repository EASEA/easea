/*
 * CStoppingCriterion.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef CSTOPPINGCRITERION_H_
#define CSTOPPINGCRITERION_H_

#include <stdlib.h>
#include <signal.h>
#ifndef WIN32
#include <sys/time.h>
#endif
#ifdef WIN32
#include <windows.h>
#endif
#include <time.h>

class CEvolutionaryAlgorithm;

/* ****************************************
   StoppingCriterion class
****************************************/
class CStoppingCriterion {

public:
  virtual bool reached() = 0;
  virtual ~CStoppingCriterion(){;}
};


/* ****************************************
   GenerationalCriterion class
****************************************/
class CGenerationalCriterion : public CStoppingCriterion {
 private:
  unsigned* currentGenerationPtr;
  unsigned generationalLimit;
 public:
  virtual bool reached();
  CGenerationalCriterion(unsigned generationalLimit);
  void setCounterEa(unsigned* ea_counter);
  unsigned *getGenerationalLimit();
};

/* ****************************************
   TimeCriterion class
****************************************/
class CTimeCriterion : public CStoppingCriterion {
 private:
  unsigned timeLimit;
  double elapsedTime;
 public:
  virtual bool reached();
  CTimeCriterion(unsigned timeLimit);
  void setElapsedTime(double elapsedTime);
  double getElapsedTime();
};

/* ****************************************
   ControlCStopingCriterion class
****************************************/
extern void signal_handler(int sig);

class CControlCStopingCriterion : public CStoppingCriterion {
 private:
 public:
  virtual bool reached();
  CControlCStopingCriterion();
};
#endif /* CSTOPPINGCRITERION_H_ */
