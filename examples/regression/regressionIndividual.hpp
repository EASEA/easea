

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

//#include "CRandomGenerator.h"
#include <stdlib.h>
#include <iostream>
#include <CIndividual.h>
#include <Parameters.h>
#include <string>
#include <CStats.h>
#include <list>
#include <map>
#include "CGPNode.h"

using namespace std;

class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;
class CCuda;




class IndividualImpl : public CIndividual {

public: // in regression the genome is public (for user functions,...)
	// Class members
  	// Class members 
  GPNode *root;



public:
	IndividualImpl();
	IndividualImpl(const IndividualImpl& indiv);
	virtual ~IndividualImpl();
	float evaluate();
	static unsigned getCrossoverArrity(){ return 2; }
	float getFitness(){ return this->fitness; }
	CIndividual* crossover(CIndividual** p2);
	void printOn(std::ostream& O) const;
	CIndividual* clone();

	unsigned mutate(float pMutationPerGene);

	void boundChecking();

	string serialize();
	void deserialize(string EASEA_Line);
	void copyToCudaBuffer(void* buffer, unsigned id);

	friend std::ostream& operator << (std::ostream& O, const IndividualImpl& B) ;
	void initRandomGenerator(CRandomGenerator* rg){ IndividualImpl::rg = rg;}
};


class ParametersImpl : public Parameters {
public:
	void setDefaultParameters(int argc, char** argv);
	CEvolutionaryAlgorithm* newEvolutionaryAlgorithm();
};

/**
 * @TODO ces functions devraient s'appeler weierstrassInit, weierstrassFinal etc... (en gros regressionFinal dans le tpl).
 *
 */

void regressionInit(int argc, char** argv);
void regressionFinal(CPopulation* pop);
void regressionBeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void regressionEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void regressionGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm);


class EvolutionaryAlgorithmImpl: public CEvolutionaryAlgorithm {
public:
	EvolutionaryAlgorithmImpl(Parameters* params);
	virtual ~EvolutionaryAlgorithmImpl();
	void initializeParentPopulation();
};

class PopulationImpl: public CPopulation {
public:
  //CCuda *cuda;
  void* cudaBuffer;

public:
  PopulationImpl(unsigned parentPopulationSize, unsigned offspringPopulationSize, float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params, CStats* stats);
        virtual ~PopulationImpl();
        void evaluateParentPopulation();
	void evaluateOffspringPopulation();
};

#endif /* PROBLEM_DEP_H */

