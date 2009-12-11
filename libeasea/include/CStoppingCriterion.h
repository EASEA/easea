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

/* ****************************************
   TimeCriterion class
****************************************/
class CTimeCriterion : public CStoppingCriterion {
 private:
  size_t timeLimit;
  size_t elapsedTime;
 public:
  virtual bool reached();
  CTimeCriterion(size_t timeLimit);
  void setElapsedTime(size_t elapsedTime);
};

/* ****************************************
   Goal stopping criterion class
****************************************/
class CGoalCriterion : public CStoppingCriterion {
 private:
  double goal;
  bool minimize;
 public:
  virtual bool reached();
  CGoalCriterion( double goal, bool minimize );
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
