/*
 *    Copyright (C) 2009  Ogier Maitre

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
