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

class CSelectionOperator {
  public:
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    virtual float getExtremum() = 0 ;
    virtual ~CSelectionOperator(){;}
    inline std::string getSelectorName(){return name;}
  protected:
    CIndividual** population;
    float currentSelectionPressure;
    std::string name;
};

float getSelectionPressure(std::string const& selectop);
CSelectionOperator* getSelectionOperator(std::string const& selectop, int minimizing);

/* ****************************************
   Tournament classes (min and max)
 ****************************************/
class MaxTournament : public CSelectionOperator{
  public:
    MaxTournament(std::string const& name){ this->name = name; }
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    float getExtremum();
};



class MinTournament : public CSelectionOperator{
  public:
    MinTournament(std::string const& name){ this->name = name; }
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    float getExtremum();
};


class MaxDeterministic : public CSelectionOperator{
  public:
    MaxDeterministic(std::string const& name) { this->name = name; }
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    float getExtremum();
  private:
    size_t populationSize;
};

class MinDeterministic : public CSelectionOperator{
  public:
    MinDeterministic(std::string const& name) { this->name = name; }
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    float getExtremum();
  private:
    size_t populationSize;

};


class MaxRandom : public CSelectionOperator{
  public:
    MaxRandom(std::string const& name);
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    float getExtremum();
};

class MinRandom : public CSelectionOperator{
  public:
    MinRandom(std::string const& name);
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    float getExtremum();
};

/* ****************************************
 *    Roulette classes (MAX)
 *****************************************/

class MaxRoulette : public CSelectionOperator{
  public:
    MaxRoulette(std::string const& name){ this->name = name; }
    virtual void initialize(CIndividual** population, float selectionPressure, size_t populationSize);
    virtual size_t selectNext(size_t populationSize);
    float getExtremum();
  private:
    size_t populationSize;
};


#endif /* CSELECTIONOPERATOR_H_ */
