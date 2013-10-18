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

#ifndef CSELECTIONOPERATOR_H_
#define CSELECTIONOPERATOR_H_

#include <stdlib.h>
#include "CIndividual.h"
#include <string>

class CSelectionOperator {
public:
  virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  virtual float getExtremum() = 0 ;
  virtual ~CSelectionOperator(){;}
protected:
  CIndividual** population;
  float currentSelectionPressure;
};

extern float getSelectionPressure(std::string selectop);
extern CSelectionOperator* getSelectionOperator(std::string selectop, int minimizing, CRandomGenerator* globalRandomGenerator);

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

/* ****************************************
 *    Roulette classes (MAX)
 *****************************************/

class MaxRoulette : public CSelectionOperator{
public:
  MaxRoulette(CRandomGenerator* rg){ this->rg = rg; }
  virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
private:
  size_t populationSize;
  CRandomGenerator* rg;
};


#endif /* CSELECTIONOPERATOR_H_ */
