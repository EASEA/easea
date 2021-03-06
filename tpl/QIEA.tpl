\TEMPLATE_START
/***********************************************************************
| QIEA   Single Objective Quantum Inspired Algorithm Template                            |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2019-10                                                         |
|                                                                       |
 ***********************************************************************/


\ANALYSE_PARAMETERS
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include "COptionParser.h"
#include "CRandomGenerator.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "EASEAIndividual.hpp"

using namespace std;

/** Global variables for the whole algorithm */

CIndividual** pPopulation = NULL;
CIndividual*  bBest = NULL;
float* pEZ_MUT_PROB = NULL;
float* pEZ_XOVER_PROB = NULL;
unsigned *EZ_NB_GEN;
unsigned *EZ_current_generation;
int EZ_POP_SIZE;
int OFFSPRING_SIZE;

CEvolutionaryAlgorithm* EA;

int main(int argc, char** argv){


	parseArguments("EASEA.prm",argc,argv);

	ParametersImpl p;
	p.setDefaultParameters(argc,argv);
	CEvolutionaryAlgorithm* ea = p.newEvolutionaryAlgorithm();

	EA = ea;

	EASEAInit(argc,argv);

	CPopulation* pop = ea->getPopulation();

	ea->runEvolutionaryLoop();

	EASEAFinal(pop);

	delete pop;


	return 0;
}

\START_CUDA_GENOME_CU_TPL

#include <fstream>
#include <time.h>
#include <cstring>
#include <sstream>
#include <chrono>
#include <vector>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"
#include <CLogger.h>
#include <problems/CProblem.h>

using namespace std;
bool bReevaluate = false;

#include "EASEAIndividual.hpp"
#define EASENA
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;
#define STD_TPL
typedef std::mt19937 TRandom;
typedef double TT;
typedef easea::problem::CProblem<TT> TP;
typedef TP::TV TV;
typedef TP::TV TO;
TRandom m_generator;
size_t limitGen;
bool reset;
typedef typename easea::shared::CBoundary<TT>::TBoundary TBoundary;

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES


\INSERT_USER_FUNCTIONS

\INSERT_INITIALISATION_FUNCTION
\INSERT_FINALIZATION_FUNCTION

size_t m_quantDomain = 5;
size_t m_classPopSize = -1;
std::vector<std::vector<TT>> quantPop;

/*
 * \brief Initialize quantum population
 */

std::vector<std::vector<TT>> initQuantumPopulation(const std::vector<std::pair<TT,TT>>&boundary, const size_t nbDomain)
{
	size_t iSize = boundary.size();
	std::vector<std::vector<TT>> qPop(nbDomain, std::vector<TT>(iSize*2)); 
	
	/* nbDomain - size of quantum population */
	for (size_t i = 0; i < nbDomain; ++i){
		/* Initialize quantum gene */
		for (size_t j = 0; j < iSize; ++j){
			double pulse = (boundary[j].second - boundary[j].first)/(double)nbDomain;
			/* Alpha = mean of pulse */
			if (i == 0)
			    qPop[i][j*2] = -boundary[j].second + pulse/2.;
			else 
			    qPop[i][j*2] = qPop[i-1][j] + pulse;
			/* Beta = width of pulse */
			qPop[i][j*2+1] = pulse;	
		}
	}
	return qPop;
} 
/*Initialize classical population */
std::vector<std::vector<TT>> observation(TRandom &random, const std::vector<std::pair<TT,TT>>&boundary, const std::vector<std::vector<TT>> qPop, const size_t nbDomain)
{
	std::vector<std::vector<TT>> cPop(m_classPopSize, std::vector<double>(boundary.size()));

        uniform_real_distribution<TT> dist(0, 1 /*nbDomain*/);
	
	int currQuant = 0;
	int nbQuant = int(m_classPopSize / nbDomain);
	for (size_t i = 0; i < m_classPopSize; i+=nbQuant){
		for (size_t j = 0; j < nbQuant; j++){
			for (size_t k = 0; k < boundary.size(); ++k)
			    cPop[i+j][k] = dist(random) * qPop[currQuant][2*k+1] + (qPop[currQuant][0] - qPop[currQuant][2*k+1]/(double)2.);
		}
		currQuant++;
	}
	return cPop;
}
pair<double,int> get_max(vector<double>& F, int type){

        double n_max = F[0]; int idx = 0;

        for( int i = 1 ; i < F.size() ; ++i){
                if( type == 1 ) {
                        if( n_max < F[i] ){
                                n_max = F[i];
                                idx = i;
                        }
                }
                else if( type == -1 ){
                        if( n_max > F[i] ){
                                n_max = F[i];
                                idx = i;
                        }
                }
        }
                
    
                
        pair<double,int> ans;
        ans.first = n_max; ans.second = idx;

        return ans;
}

void breeding(TRandom random, size_t size, std::vector<std::vector<TT>>&currPop, std::vector<std::vector<TT>> &prevPop){
	double rate = 0.;
	double prob = 0.;
	double beta = 0.;

	uniform_real_distribution<double> dProb(0, 1); /* Distribution for crossover probability */
	uniform_int_distribution<int> dIndex(0, m_classPopSize - 1); /*Distribution for index individual */

	int iFirst, iSecond;
	std::vector<double> indFirst(size);
	std::vector<double> indSecond(size);

	/* Random Selection */
	for ( size_t i = 0; i < m_classPopSize; ++i ){
		rate = dProb(random);
		do iFirst = dIndex(random); while (iFirst == i);
		do iSecond = dIndex(random); while (iSecond == i || iSecond == iFirst);		
		
		for ( size_t j = 0; j < size; ++j ){
		    prob = dProb(random);
		    beta = dProb(random);

		    indFirst[j] = prob * currPop[iFirst][j] + beta *prevPop[iSecond][j]; 
		    indSecond[j] = prob * prevPop[iSecond][j] + beta * currPop[iFirst][j];
		}
	
		/* DE Crossover */
	//	if (rate <= 0.6){
		    currPop[iFirst] = indFirst;
		    currPop[iSecond] = indSecond;
	//	}
	}

	
}

void evale_pop_chunk(CIndividual** population, int popSize){
  \INSTEAD_EVAL_FUNCTION
}

void EASEAInit(int argc, char** argv){
	\INSERT_INIT_FCT_CALL
	if (m_classPopSize <= 0) LOG_ERROR(errorCode::value, "Wrong size of parent population");
	if ((m_quantDomain <= 0) || (m_quantDomain > m_classPopSize/2)) LOG_ERROR(errorCode::value, "Wrong size of quantum domain number");
	//quantPop = initQuantumPopulation(m_problem.getBoundary(), m_quantDomain);
}

void EASEAFinal(CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;
}
 void AESAEBeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	\INSERT_BEGIN_GENERATION_FUNCTION
    
    }
void EASEABeginningGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	\INSERT_BEGIN_GENERATION_FUNCTION
 
}
void AESAEEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    \INSERT_END_GENERATION_FUNCTION
}

void EASEAEndGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	\INSERT_END_GENERATION_FUNCTION
}

void EASEAGenerationFunctionBeforeReplace(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	\INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}
void AESAEGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    \INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}

IndividualImpl::IndividualImpl() : CIndividual() {
  \GENOME_CTOR 
  \INSERT_EO_INITIALISER
  valid = false;
  isImmigrant = false;
 
}
IndividualImpl::IndividualImpl(std::vector<double> ind)
{
	for (size_t i = 0; i < m_problem.getBoundary().size(); i++)
	    x[i] = ind[i];
}

CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

IndividualImpl::~IndividualImpl(){
  \GENOME_DTOR
}


float IndividualImpl::evaluate(){
 /* if(valid)
    return fitness;
  else{
    valid = true;*/
    \INSERT_EVALUATOR
  //}
}

void IndividualImpl::boundChecking(){
	\INSERT_BOUND_CHECKING
}

string IndividualImpl::serialize(){
    ostringstream AESAE_Line(ios_base::app);
    \GENOME_SERIAL
    AESAE_Line << this->fitness;
    return AESAE_Line.str();
}

void IndividualImpl::deserialize(string Line){
    istringstream AESAE_Line(Line);
    string line;
    \GENOME_DESERIAL
    AESAE_Line >> this->fitness;
    this->valid=true;
    this->isImmigrant = false;
}


IndividualImpl::IndividualImpl(const IndividualImpl& genome){

  // ********************
  // Problem specific part
  \COPY_CTOR


  // ********************
  // Generic part
  this->valid = genome.valid;
  this->fitness = genome.fitness;
  this->isImmigrant = false;
}


CIndividual* IndividualImpl::crossover(CIndividual** ps){
	// ********************
	// Generic part
	IndividualImpl** tmp = (IndividualImpl**)ps;
	IndividualImpl parent1(*this);
	IndividualImpl parent2(*tmp[0]);
	IndividualImpl child(*this);

	//DEBUG_PRT("Xover");
	/*   cout << "p1 : " << parent1 << endl; */
	/*   cout << "p2 : " << parent2 << endl; */

	// ********************
	// Problem specific part
  	\INSERT_CROSSOVER


	child.valid = false;
	/*   cout << "child : " << child << endl; */
	return new IndividualImpl(child);
}


void IndividualImpl::printOn(std::ostream& os) const{
	\INSERT_DISPLAY
}

std::ostream& operator << (std::ostream& O, const IndividualImpl& B)
{
  // ********************
  // Problem specific part
  O << "\nIndividualImpl : "<< std::endl;
  O << "\t\t\t";
  B.printOn(O);

  if( B.valid ) O << "\t\t\tfitness : " << B.fitness;
  else O << "fitness is not yet computed" << std::endl;
  return O;
}


unsigned IndividualImpl::mutate( float pMutationPerGene ){
  this->valid=false;


  // ********************
  // Problem specific part
  \INSERT_MUTATOR
}

void ParametersImpl::setDefaultParameters(int argc, char** argv){

	this->minimizing = \MINIMAXI;
	this->nbGen = setVariable("nbGen",(int)\NB_GEN);

	seed = setVariable("seed",(int)time(0));
	globalRandomGenerator = new CRandomGenerator(seed);
	this->randomGenerator = globalRandomGenerator;


	selectionOperator = getSelectionOperator(setVariable("selectionOperator","\SELECTOR_OPERATOR"), this->minimizing, globalRandomGenerator);
	replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator","\RED_FINAL_OPERATOR"),this->minimizing, globalRandomGenerator);
	parentReductionOperator = getSelectionOperator(setVariable("reduceParentsOperator","\RED_PAR_OPERATOR"),this->minimizing, globalRandomGenerator);
	offspringReductionOperator = getSelectionOperator(setVariable("reduceOffspringOperator","\RED_OFF_OPERATOR"),this->minimizing, globalRandomGenerator);
	selectionPressure = setVariable("selectionPressure",(float)\SELECT_PRM);
	replacementPressure = setVariable("reduceFinalPressure",(float)\RED_FINAL_PRM);
	parentReductionPressure = setVariable("reduceParentsPressure",(float)\RED_PAR_PRM);
	offspringReductionPressure = setVariable("reduceOffspringPressure",(float)\RED_OFF_PRM);
	pCrossover = \XOVER_PROB;
	pMutation = \MUT_PROB;
	pMutationPerGene = 0.05;

	parentPopulationSize = setVariable("popSize",(int)\POP_SIZE);
	offspringPopulationSize = setVariable("nbOffspring",(int)\OFF_SIZE);
	m_classPopSize = parentPopulationSize;

	parentReductionSize = setReductionSizes(parentPopulationSize, setVariable("survivingParents",(float)\SURV_PAR_SIZE));
	offspringReductionSize = setReductionSizes(offspringPopulationSize, setVariable("survivingOffspring",(float)\SURV_OFF_SIZE));

	this->elitSize = setVariable("elite",(int)\ELITE_SIZE);
	this->strongElitism = setVariable("eliteType",(int)\ELITISM);

	if((this->parentReductionSize + this->offspringReductionSize) < this->parentPopulationSize){
		printf("*WARNING* parentReductionSize + offspringReductionSize < parentPopulationSize\n");
		printf("*WARNING* change Sizes in .prm or .ez\n");
		printf("EXITING\n");
		exit(1);	
	} 
	if((this->parentPopulationSize-this->parentReductionSize)>this->parentPopulationSize-this->elitSize){
		printf("*WARNING* parentPopulationSize - parentReductionSize > parentPopulationSize - elitSize\n");
		printf("*WARNING* change Sizes in .prm or .ez\n");
		printf("EXITING\n");
		exit(1);	
	} 
	if(!this->strongElitism && ((this->offspringPopulationSize - this->offspringReductionSize)>this->offspringPopulationSize-this->elitSize)){
		printf("*WARNING* offspringPopulationSize - offspringReductionSize > offspringPopulationSize - elitSize\n");
		printf("*WARNING* change Sizes in .prm or .ez\n");
		printf("EXITING\n");
		exit(1);	
	} 
	

	/*
	 * The reduction is set to true if reductionSize (parent or offspring) is set to a size less than the
	 * populationSize. The reduction size is set to populationSize by default
	 */
	if(offspringReductionSize<offspringPopulationSize) offspringReduction = true;
	else offspringReduction = false;

	if(parentReductionSize<parentPopulationSize) parentReduction = true;
	else parentReduction = false;

	generationalCriterion = new CGenerationalCriterion(setVariable("nbGen",(int)\NB_GEN));
	controlCStopingCriterion = new CControlCStopingCriterion();
	timeCriterion = new CTimeCriterion(setVariable("timeLimit",\TIME_LIMIT));

	this->optimise = 0;

	this->printStats = setVariable("printStats",\PRINT_STATS);
	this->generateCSVFile = setVariable("generateCSVFile",\GENERATE_CSV_FILE);
	this->generatePlotScript = setVariable("generatePlotScript",\GENERATE_GNUPLOT_SCRIPT);
	this->generateRScript = setVariable("generateRScript",\GENERATE_R_SCRIPT);
	this->plotStats = setVariable("plotStats",\PLOT_STATS);
	this->printInitialPopulation = setVariable("printInitialPopulation",0);
	this->printFinalPopulation = setVariable("printFinalPopulation",0);
	this->savePopulation = setVariable("savePopulation",\SAVE_POPULATION);
	this->startFromFile = setVariable("startFromFile",\START_FROM_FILE);

	this->outputFilename = (char*)"EASEA";
	this->plotOutputFilename = (char*)"EASEA.png";

	this->remoteIslandModel = setVariable("remoteIslandModel",\REMOTE_ISLAND_MODEL);
	std::string* ipFilename=new std::string();
	*ipFilename=setVariable("ipFile","\IP_FILE");

	this->ipFile =(char*)ipFilename->c_str();
	this->migrationProbability = setVariable("migrationProbability",(float)\MIGRATION_PROBABILITY);
    this->serverPort = setVariable("serverPort",\SERVER_PORT);
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	//EZ_NB_GEN = (unsigned*)setVariable("nbGen",\NB_GEN);
	EZ_current_generation=0;
  EZ_POP_SIZE = parentPopulationSize;
  OFFSPRING_SIZE = offspringPopulationSize;

	CEvolutionaryAlgorithm* ea = new EvolutionaryAlgorithmImpl(this);
	generationalCriterion->setCounterEa(ea->getCurrentGenerationPtr());
	ea->addStoppingCriterion(generationalCriterion);
	ea->addStoppingCriterion(controlCStopingCriterion);
	ea->addStoppingCriterion(timeCriterion);	

	EZ_NB_GEN=((CGenerationalCriterion*)ea->stoppingCriteria[0])->getGenerationalLimit();
	EZ_current_generation=&(ea->currentGeneration);

	 return ea;
}

void EvolutionaryAlgorithmImpl::initializeParentPopulation(){
/*	if(this->params->startFromFile){
	  ifstream AESAE_File("EASEA.pop");
	  string AESAE_Line;
  	  for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
	  	  getline(AESAE_File, AESAE_Line);
		  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
		  ((IndividualImpl*)this->population->parents[i])->deserialize(AESAE_Line);
	  }
	  
	}
	else{
  	  for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
		  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
	  }
	}*/
        this->population->actualParentPopulationSize = this->params->parentPopulationSize;

	quantPop = initQuantumPopulation(m_problem.getBoundary(), m_quantDomain);


}

void EvolutionaryAlgorithmImpl::runEvolutionaryLoop(){

	std::vector<std::vector<TT>> classPop(this->params->parentPopulationSize, std::vector<double>(m_problem.getBoundary().size()));
	std::vector<std::vector<TT>> classPopPrev(this->params->parentPopulationSize, std::vector<double>(m_problem.getBoundary().size()));
	
	LOG_MSG(msgType::INFO, "QIEA starting....");
	auto tmStart = std::chrono::system_clock::now();
	limitGen = EZ_NB_GEN[0];
	size_t gSize = m_problem.getBoundary().size();
	std::vector<double> f(this->params->parentPopulationSize);

        double best;// = powf(2,64) - 1;
        int coef = 1;
	reset = false;
	currentGeneration = 0;
	this->initializeParentPopulation();
	while( this->allCriteria() == false){
	    if (currentGeneration == 0){
		this->initializeParentPopulation();
		best = powf(2,64) - 1;
		reset = false;

	    }
		EASEABeginningGeneration(this);

		classPop = observation(m_generator, m_problem.getBoundary(), quantPop, m_quantDomain);

		ostringstream ss;
		ss << "Generation: " << currentGeneration << std::endl;
		LOG_MSG(msgType::INFO, ss.str());
		if (currentGeneration == 0)
		{
			for (size_t i = 0; i < this->params->parentPopulationSize; i++){
				  population->addIndividualParentPopulation(new IndividualImpl(classPop[i]),i);
				  classPopPrev[i] = classPop[i];
				  f[i] = population->parents[i]->evaluate();
			}
		//	population->evaluateParentPopulation();

		}else{
			breeding(m_generator, gSize, classPop, classPopPrev);
			for (size_t i = 0; i < this->params->parentPopulationSize; i++){
				for (size_t j = 0; j < NB_VARIABLES; j++)
				   ((IndividualImpl *)population->parents[i])->x[j] = classPop[i][j];
				 classPopPrev[i] = classPop[i];
				 f[i] = population->parents[i]->evaluate();
			}
		//	population->evaluateOffspringPopulation();
		}
                /* Update by RULE 1/5 */
                uniform_real_distribution<double> dist(0, 1);
                double uProb = dist(m_generator);
	//	std::vector<double> f(this->params->parentPopulationSize);
                if ( uProb < 0.5 ){
                        int acc = 0;
                        for (size_t i = 0; i < this->params->parentPopulationSize; i++){
                                if ( f[i] < best) acc++;
                        }
                        double phi = (double)acc / (double) this->params->parentPopulationSize;
                        double factor = 1.;
                        if (phi < 0.2) factor = 0.82;
                        else if ( phi > 0.2 ) factor = 1.0/(double)0.82;
                        for( size_t i = 0 ; i < quantPop.size() ; i++ ){
                             for( size_t j = 1 ; j < quantPop[0].size() ; j+=2){
                                    quantPop[i][j] = quantPop[i][j]*factor;
                             }
                         }
                             
                }
                /* Move pulse */
                if ( currentGeneration > 0 && ( currentGeneration + 1)%coef == 0 ){
                        double lambda = 0.5;
                        for( size_t i = 0 ; i < quantPop.size() ; i++ ){
                                pair<double, int> ans = get_max(f, -1);
                                for( size_t j = 0, k = 0; j < quantPop[0].size() ; j+=2, k++ ){
                                        quantPop[i][j] = quantPop[i][j] + lambda * (classPop[ans.second][k] - quantPop[i][j]);

                                }
                        }
                }
		pair<double, int> ans = get_max(f, -1);
		best = (ans.first < best) ? ans.first : best;
		printf("Best: %f\n", best);

		currentGeneration += 1;
		EASEAEndGeneration(this);
		if (reset == true)
		{
		    currentGeneration = 0;
		    reset = false;
		}

	}

	std::chrono::duration<double> tmDur = std::chrono::system_clock::now() - tmStart;
	ostringstream ss;
        ss << "Total execution time (in sec.): " << tmDur.count() << std::endl;
	LOG_MSG(msgType::INFO, ss.str());

}

EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){
	;
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

\START_CUDA_GENOME_H_TPL

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

//#include "CRandomGenerator.h"
#include <stdlib.h>
#include <iostream>
#include <CIndividual.h>
#include <Parameters.h>
#include <cstring>
#include <sstream>

using namespace std;

class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;

extern int EZ_POP_SIZE;
extern int OFFSPRING_SIZE;

\INSERT_USER_CLASSES

class IndividualImpl : public CIndividual {

public: // in EASEA the genome is public (for user functions,...)
	// Class members
  	\INSERT_GENOME

public:
	IndividualImpl();
	IndividualImpl(const IndividualImpl& indiv);
        IndividualImpl(std::vector<double> ind);
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
	void deserialize(string AESAE_Line);

	friend std::ostream& operator << (std::ostream& O, const IndividualImpl& B) ;
	void initRandomGenerator(CRandomGenerator* rg){ IndividualImpl::rg = rg;}

};


class ParametersImpl : public Parameters {
public:
	void setDefaultParameters(int argc, char** argv);
	CEvolutionaryAlgorithm* newEvolutionaryAlgorithm();
};

/**
 * @TODO ces functions devraient s'appeler weierstrassInit, weierstrassFinal etc... (en gros EASEAFinal dans le tpl).
 *
 */

void EASEAInit(int argc, char** argv);
void EASEAFinal(CPopulation* pop);
void EASEABeginningGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAEndGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAGenerationFunctionBeforeReplace(CEvolutionaryAlgorithm* evolutionaryAlgorithm);


class EvolutionaryAlgorithmImpl: public CEvolutionaryAlgorithm {
public:
	EvolutionaryAlgorithmImpl(Parameters* params);
	virtual ~EvolutionaryAlgorithmImpl();
	void initializeParentPopulation();
	void runEvolutionaryLoop();

};

#endif /* PROBLEM_DEP_H */

\START_CUDA_MAKEFILE_TPL

UNAME := $(shell uname)

ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif

ifneq ("$(OS)","")
	EZ_PATH=../../
endif

EASEALIB_PATH=$(EZ_PATH)/libeasea/

CXXFLAGS =  -std=c++14 -fopenmp -O2 -g -Wall -fmessage-length=0 -I$(EASEALIB_PATH)include

OBJS = EASEA.o EASEAIndividual.o 

LIBS = -lpthread -fopenmp 
ifneq ("$(OS)","")
	LIBS += -lws2_32 -lwinmm -L"C:\MinGW\lib"
endif

#USER MAKEFILE OPTIONS :
\INSERT_MAKEFILE_OPTION
#END OF USER MAKEFILE OPTIONS

TARGET =	EASEA

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS) -g $(EASEALIB_PATH)/libeasea.a $(LIBS)

	
#%.o:%.cpp
#	$(CXX) -c $(CXXFLAGS) $^

all:	$(TARGET)
clean:
ifneq ("$(OS)","")
	-del $(OBJS) $(TARGET).exe
else
	rm -f $(OBJS) $(TARGET)
endif
easeaclean:
ifneq ("$(OS)","")
	-del $(TARGET).exe *.o *.cpp *.hpp EASEA.png EASEA.dat EASEA.prm EASEA.mak Makefile EASEA.vcproj EASEA.csv EASEA.r EASEA.plot EASEA.pop
else
	rm -f $(TARGET) *.o *.cpp *.hpp EASEA.png EASEA.dat EASEA.prm EASEA.mak Makefile EASEA.vcproj EASEA.csv EASEA.r EASEA.plot EASEA.pop
endif

\START_EO_PARAM_TPL#****************************************
#                                         
#  EASEA.prm
#                                         
#  Parameter file generated by STD.tpl AESAE v1.0
#                                         
#***************************************
# --seed=0   # -S : Random number seed. It is possible to give a specific seed.

######    Evolution Engine    ######
--popSize=\POP_SIZE # -P : Population Size
--nbOffspring=\OFF_SIZE # -O : Nb of offspring (percentage or absolute)

######	  Stopping Criterions    #####
--nbGen=\NB_GEN #Nb of generations
--timeLimit=\TIME_LIMIT # Time Limit: desactivate with (0) (in Seconds)

######    Evolution Engine / Replacement    ######
--elite=\ELITE_SIZE  # Nb of elite parents (absolute)
--eliteType=\ELITISM # Strong (1) or weak (0) elitism (set elite to 0 for none)
--survivingParents=\SURV_PAR_SIZE # Nb of surviving parents (percentage or absolute)
--survivingOffspring=\SURV_OFF_SIZE  # Nb of surviving offspring (percentage or absolute)
--selectionOperator=\SELECTOR_OPERATOR # Selector: Deterministic, Tournament, Random, Roulette 
--selectionPressure=\SELECT_PRM
--reduceParentsOperator=\RED_PAR_OPERATOR 
--reduceParentsPressure=\RED_PAR_PRM
--reduceOffspringOperator=\RED_OFF_OPERATOR 
--reduceOffspringPressure=\RED_OFF_PRM
--reduceFinalOperator=\RED_FINAL_OPERATOR
--reduceFinalPressure=\RED_FINAL_PRM

#####	Stats Ouput 	#####
--printStats=\PRINT_STATS #print Stats to screen
--plotStats=\PLOT_STATS #plot Stats
--printInitialPopulation=0 #Print initial population
--printFinalPopulation=0 #Print final population
--generateCSVFile=\GENERATE_CSV_FILE
--generatePlotScript=\GENERATE_GNUPLOT_SCRIPT
--generateRScript=\GENERATE_R_SCRIPT

#### Population save	####
--savePopulation=\SAVE_POPULATION #save population to EASEA.pop file
--startFromFile=\START_FROM_FILE #start optimisation from EASEA.pop file

#### Remote Island Model ####
--remoteIslandModel=\REMOTE_ISLAND_MODEL #To initialize communications with remote AESAE's
--ipFile=\IP_FILE
--migrationProbability=\MIGRATION_PROBABILITY #Probability to send an individual every generation
--serverPort=\SERVER_PORT
\TEMPLATE_END
