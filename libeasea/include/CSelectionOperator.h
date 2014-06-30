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
#include <string>
#include <omp.h>

/*TODO: Refactor CSelectionOperator*/
class CSelectionOperator {
  public:
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    virtual float getExtremum() = 0 ;
    virtual ~CSelectionOperator(){;}
    virtual size_t selectNext(size_t populationSize, int rgId);
    virtual void setThreadRg();
    virtual void setRg(CRandomGenerator* rg);
    virtual CSelectionOperator* copy(size_t populationSize,CRandomGenerator* rg);
    CRandomGenerator* rg;
  protected:
    CIndividual** population;
    float currentSelectionPressure;
    CRandomGenerator** threadRg;
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
    virtual size_t selectNext(size_t populationSize, int rgId);
    virtual CSelectionOperator* copy(size_t populationSize,CRandomGenerator* rg);
    void setThreadRg();
    float getExtremum();
    CSelectionOperator* copy();
  private:
};



class MinTournament : public CSelectionOperator{
  public:
    MinTournament(CRandomGenerator* rg){ this->rg = rg; }
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    virtual size_t selectNext(size_t populationSize, int rgId);
    virtual CSelectionOperator* copy(size_t populationSize,CRandomGenerator* rg);
    float getExtremum();
    void setThreadRg();
  private:

};


class MaxDeterministic : public CSelectionOperator{
  public:
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    size_t selectNext(size_t populationSize);
    float getExtremum();
  private:
    size_t populationSize;
};

class MinDeterministic : public CSelectionOperator{
  public:
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    size_t selectNext(size_t populationSize);
    float getExtremum();
  private:
    size_t populationSize;

};


class MaxRandom : public CSelectionOperator{
  public:
    MaxRandom(CRandomGenerator* globalCRandomGenerator);
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    virtual size_t selectNext(size_t populationSize, int rgId);
    float getExtremum();
    void setThreadRg();
  private:

};

class MinRandom : public CSelectionOperator{
  public:
    MinRandom(CRandomGenerator* globalCRandomGenerator);
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    virtual size_t selectNext(size_t populationSize, int rgId);
    float getExtremum();
    void setThreadRg();
  private:
};

/* ****************************************
 *    Roulette classes (MAX)
 *****************************************/

class MaxRoulette : public CSelectionOperator{
  public:
    MaxRoulette(CRandomGenerator* rg){ this->rg = rg; }
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    virtual size_t selectNext(size_t populationSize, int rgId);
    float getExtremum();
    void setThreadRg();
  private:
    size_t populationSize;
};


#endif /* CSELECTIONOPERATOR_H_ */
