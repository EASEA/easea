\TEMPLATE_START
/***********************************************************************
| QAES Single Objective Quantum Inspired Algorithm Template             |
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
	ParametersImpl p("EASEA.prm", argc, argv);

	CEvolutionaryAlgorithm* ea = p.newEvolutionaryAlgorithm();
	pPopulation = ea->population->parents;

	EA = ea;

	EASEAInit(argc, argv, p);

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
#include <limits>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"
#include <CLogger.h>
#include <problems/CProblem.h>
#include <shared/distributions/Norm.h>

#include "config.h"
#ifdef USE_OPENMP
	#include <omp.h>
#endif

using namespace std;
#define AESAE
bool bReevaluate = false;

#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

#define ACC_1 100
#define ACC_2 100
#define LIMIT_UPDATE 30
#define SZ_POP_MAX 5000

extern CEvolutionaryAlgorithm* EA;
extern std::ofstream easea::log_file;
extern easea::log_stream logg;

#ifndef CUSTOM_PRECISION_TYPE
	typedef double TT;
#else
	using TT = CUSTOM_PRECISION_TYPE;
#endif

typedef easea::problem::CProblem<TT> TP;
typedef TP::TV TV;
typedef TP::TV TO;
//TRandom m_generator;
size_t limitGen;
bool reset;
int num = 0;

typedef typename easea::shared::CBoundary<TT>::TBoundary TBoundary;

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES


\INSERT_USER_FUNCTIONS

\INSERT_FINALIZATION_FUNCTION



int getGlobalSolutionIndex(int size, std::vector<TT> F)
{
        int i, ind = 0;
    
        double minVal = F[0];

        for( i=1; i < size; i++)
        {
            if( F[i] < minVal )
            {
                minVal = F[i];
		ind = i;
            }
        }
        return ind;
} 


void evale_pop_chunk([[maybe_unused]] CIndividual** population, [[maybe_unused]] int popSize) {
  \INSTEAD_EVAL_FUNCTION
}

void EASEAInit(int argc, char* argv[], ParametersImpl& p){
	(void)argc;(void)argv;(void)p;
	auto setVariable = [&](std::string const& arg, auto def) {
		return p.setVariable(arg, std::forward<decltype(def)>(def));
	}; // for compatibility
	(void)setVariable;

	\INSERT_INITIALISATION_FUNCTION
}

void EASEAFinal(CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;
}

void AESAEBeginningGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_BEGIN_GENERATION_FUNCTION

}
void EASEABeginningGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    \INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_END_GENERATION_FUNCTION
}
void EASEAEndGeneration(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    \INSERT_END_GENERATION_FUNCTION
}
void EASEAGenerationFunctionBeforeReplace(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    \INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}

void AESAEGenerationFunctionBeforeReplacement([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}

IndividualImpl::IndividualImpl()
{
	for (size_t i = 0; i < m_problem.getBoundary().size(); i++)
	    this->\GENOME_NAME[i] = (rand()/(RAND_MAX+1.0)*(m_problem.getBoundary()[i].second-m_problem.getBoundary()[i].first)+m_problem.getBoundary()[i].first);
}
IndividualImpl::IndividualImpl(std::vector<double> ind)
     {
    for (size_t i = 0; i < m_problem.getBoundary().size(); i++)
       this->\GENOME_NAME[i] = ind[i];
     }


CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

float IndividualImpl::evaluate(){
	\INSERT_EVALUATOR
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


void IndividualImpl::printOn([[maybe_unused]] std::ostream& os) const {
	\INSERT_DISPLAY
}

unsigned IndividualImpl::mutate([[maybe_unused]] float pMutationPerGene ) {
  this->valid=false;


  // ********************
  // Problem specific part
  \INSERT_MUTATOR
  return 0;
}

ParametersImpl::ParametersImpl(std::string const& file, int argc, char* argv[]) : Parameters(file, argc, argv) {

	this->minimizing = \MINIMAXI;
	this->nbGen = setVariable("nbGen", (int)\NB_GEN);

	#ifdef USE_OPENMP
	omp_set_num_threads(nbCPUThreads);
	#endif
	srand(seed);

	selectionOperator = getSelectionOperator(setVariable("selectionOperator", "\SELECTOR_OPERATOR"), this->minimizing);
	replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator", "\RED_FINAL_OPERATOR"),this->minimizing);
	parentReductionOperator = getSelectionOperator(setVariable("reduceParentsOperator", "\RED_PAR_OPERATOR"),this->minimizing);
	offspringReductionOperator = getSelectionOperator(setVariable("reduceOffspringOperator", "\RED_OFF_OPERATOR"),this->minimizing);
	selectionPressure = setVariable("selectionPressure", (float)\SELECT_PRM);
	replacementPressure = setVariable("reduceFinalPressure", (float)\RED_FINAL_PRM);
	parentReductionPressure = setVariable("reduceParentsPressure", (float)\RED_PAR_PRM);
	offspringReductionPressure = setVariable("reduceOffspringPressure", (float)\RED_OFF_PRM);
	pCrossover = \XOVER_PROB;
	pMutation = \MUT_PROB;
	pMutationPerGene = 0.05;
	
	parentPopulationSize = setVariable("popSize", (int)\POP_SIZE);
	offspringPopulationSize = getOffspringSize((int)\OFF_SIZE, \POP_SIZE);

	parentReductionSize = setReductionSizes(parentPopulationSize, setVariable("survivingParents", (float)\SURV_PAR_SIZE));
	offspringReductionSize = setReductionSizes(offspringPopulationSize, setVariable("survivingOffspring", (float)\SURV_OFF_SIZE));

	this->elitSize = setVariable("elite", (int)\ELITE_SIZE);
	this->strongElitism = setVariable("eliteType", (int)\ELITISM);

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

	generationalCriterion = new CGenerationalCriterion(setVariable("nbGen", (int)\NB_GEN));
	controlCStopingCriterion = new CControlCStopingCriterion();
	timeCriterion = new CTimeCriterion(setVariable("timeLimit", \TIME_LIMIT));


	this->printStats = setVariable("printStats", \PRINT_STATS);
	this->generateCSVFile = setVariable("generateCSVFile", \GENERATE_CSV_FILE);
	this->generatePlotScript = setVariable("generatePlotScript", \GENERATE_GNUPLOT_SCRIPT);
	this->generateRScript = setVariable("generateRScript", \GENERATE_R_SCRIPT);
	this->plotStats = setVariable("plotStats", \PLOT_STATS);
	this->savePopulation = setVariable("savePopulation", \SAVE_POPULATION);
	this->startFromFile = setVariable("startFromFile", \START_FROM_FILE);

	this->outputFilename = setVariable("outputFile", "EASEA");
	this->inputFilename = setVariable("inputFile", "EASEA.pop");
	this->plotOutputFilename = (char*)"EASEA.png";

	this->remoteIslandModel = setVariable("remoteIslandModel", \REMOTE_ISLAND_MODEL);
	this->ipFile = setVariable("ipFile", "\IP_FILE");
	this->migrationProbability = setVariable("migrationProbability", (float)\MIGRATION_PROBABILITY);
    	this->serverPort = setVariable("serverPort", \SERVER_PORT);
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	//pEZ_MUT_PROB = &pMutationPerGene;
	//pEZ_XOVER_PROB = &pCrossover;
	//EZ_NB_GEN = (unsigned*)setVariable("nbGen", \NB_GEN);
	EZ_current_generation=0;
        EZ_POP_SIZE = parentPopulationSize;
        OFFSPRING_SIZE = offspringPopulationSize;
	if (SZ_POP_MAX < EZ_POP_SIZE){
	    LOG_ERROR(errorCode::value, string("Max value of population must be ") + to_string(SZ_POP_MAX));
	    exit(1);
	}

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
        this->population->actualParentPopulationSize = this->params->parentPopulationSize;
/*	for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
	    this->population->addIndividualParentPopulation(new IndividualImpl(), i);
	}*/
}

/*
 * \brief Main Evolutionary Loop
 *  F(x) - objective function = Potential well V(x)
 *  best - global optimal solution = Wavefunction at the ground state
 *  sample - for Gaussina sampling - Wavefunction converges to the ground state
 *  local optimum - it is metastable state at high energy level
 *  replacement of optimal solution - transition from a metastable state to the ground state 
 */

void EvolutionaryAlgorithmImpl::runEvolutionaryLoop(){

	
	/* Get the size of serach space dimension from problem difinition by user in ez file*/
	const size_t nbVar = m_problem.getBoundary().size();
	
	const size_t szPopMax = SZ_POP_MAX; /* max value of possible population size */;
	size_t szPop = this->params->parentPopulationSize;

	/* Define quantum population */
	std::vector<std::vector<TT>>pop_x(szPopMax, std::vector<TT>(nbVar));
	std::vector<std::vector<TT>>pop_pos_best(szPopMax, std::vector<TT>(nbVar));
	

	/* Objective function f(x) is the potential well V(x) in Schrödinger equation */
	std::vector<TT> F(szPopMax);
	/* Best objective function ever been for each particle */
	std::vector<TT> bestF(szPopMax);
	/* Weight function for each particle */ 
	std::vector<int> bestM(szPopMax);
	/* Memory - needed to avoid local optimum - not used yet */
	std::vector<TT> memory(nbVar);
	/* Mean value of solution */
	std::vector<TT> meanSolution(nbVar);
	/* global solution */
	std::vector<TT> globalSolution(nbVar);
	/* flags = 0 - particle is deleted, flags = 1 - particle is alive */ 
	std::vector<TT> flags(szPopMax);

	int globalIndex = 0;
	double bestGlobal;    /* Global best value */

	int nbV = 0;  
	int nbI = 0;
	int nbDeriv = 0;
	std::vector<TT> derivF(szPopMax);
	TT memoryF;
	double epsilon = 0.01;

	/* Start logging */
	LOG_MSG(msgType::INFO, "QAES starting....");
	auto tmStart = std::chrono::system_clock::now();
        string log_fichier_name = params->outputFilename;
        time_t t = std::chrono::system_clock::to_time_t(tmStart);
        std::tm * ptm = std::localtime(&t);
        char buf_start_time[32];
	std::strftime(buf_start_time, 32, "%Y-%m-%d_%H-%M-%S", ptm);
        easea::log_file.open(log_fichier_name.c_str() + std::string("_") + std::string(buf_start_time) + std::string(".log"));


	/* koeff controls the local optimums - if koeff = 1, we are very probably in local optimum */
	double koeff = 0;
	double V = 0;
	double Vtmp = 0;
	bBest = new IndividualImpl();
	bBest->fitness = std::numeric_limits<double>::max();


	limitGen = EZ_NB_GEN[0];  /* Get maximal number of geneation from the prm file, defined by user */

	int Nt = szPop;
	double cAcc_1, cAcc_2;
	currentGeneration = 0;  /* Counter of generation */
	size_t currentEval; 	/* Counter of evaluation number */

	/* Check all cases of stop critaria */ 
	while( this->allCriteria() == false){
	
	    clock_t starttime, endtime;
	    double totaltime;
	    starttime = clock();
	    
	    /* Launch user settings from section "befor everything else" in ez file */
	    EASEABeginningGeneration(this);
	    
	    if (currentGeneration == 0){
	    /* First generation settings */
		/* Init and evaluate populations */
		for (int i = 0; i < szPopMax; i++){
		    for (int j = 0; j < nbVar; j++){
			pop_x[i][j] = (rand() / (double)(RAND_MAX + 1.)*(m_problem.getBoundary()[j].second - m_problem.getBoundary()[j].first) + m_problem.getBoundary()[j].first);
			pop_pos_best[i][j] = pop_x[i][j];
		    }
		    IndividualImpl *tmp = new IndividualImpl(pop_x[i]);
		    bestF[i] = F[i] = tmp->evaluate();
		    delete(tmp);
		    derivF[i] = std::numeric_limits<double>::max();
		    if (i < szPop) flags[i] = 1;
		    else { flags[i] = 0; V+=F[i]; }
		}  
		/* Avarage cost function value (potential energy) */
		V = V/(double)szPop;

		/* Get index of current global optimum */
		globalIndex = getGlobalSolutionIndex(szPop, F);

		/* Get desicion variables and value of fitness function of current global solution */
		for (int i = 0; i < nbVar; i++)
		    globalSolution[i] = pop_pos_best[globalIndex][i];
		bestGlobal = bestF[globalIndex];
	    }

	    double tmp = 0;
	    /* Get mean value of all current solutions */
	    for (int i = 0; i < nbVar; i++)
	    {	    
		tmp = 0;
		for ( int j = 0; j < szPop; j++)
			tmp = tmp + pop_pos_best[j][i];
			meanSolution[i] = tmp / szPop;		
	    }
	    /* Alpha for QPSO part */
	    double a = 0.5 * (this->params->nbGen - currentGeneration)/(this->params->nbGen) + 0.5;
	    if (currentGeneration == 0){
		currentEval = 0;
		/* Initialization of  population if the first generation */
		//this->initializeParentPopulation();
	    }

	    /* Logging current generation number */
	    ostringstream ss;
	    nbI = 0; 
	    Vtmp = 0;  
	    int Nnext = Nt;
	    int tt = 0;

	    for ( int i = 0; i < szPopMax; i++){
		if (flags[i] == 1){
		nbV = 0;
		for (int j = 0; j < nbVar; j++){
		    double fi1 = rand()/(double)(RAND_MAX+1.0);
		    double fi2 = rand()/(double)(RAND_MAX+1.0);

		    if (koeff == 0){
			cAcc_1 = ACC_1;
			cAcc_2 = ACC_2;
		    }else { cAcc_1 = ACC_1; cAcc_2 = ACC_1;}
		    /* Crossover "à la QPSO" */
		    double fi = cAcc_1*fi1/(cAcc_1*fi1+cAcc_2*fi2);
		    double p = pop_pos_best[i][j]*fi + (1-fi)*globalSolution[j];
		    /* if there is local optimum -> let's make the large diffusion displacement */
		    if (koeff == 1){
			 p = easea::shared::distributions::norm(p,0.1*nbVar/(3*fabs(meanSolution[j]-pop_pos_best[i][j])));
		    }

		    if ((p - globalSolution[j]) < epsilon) nbV++;
		    double u = rand()/(double)(RAND_MAX+1.0);
		    double b = a * fabs(meanSolution[j] - pop_x[i][j]);
		    double v = log(1/(double)u);
		    double z = rand()/(double)(RAND_MAX+1.0);
		    if ( z < 0.5) pop_x[i][j] = (p + b * v);
		    else pop_x[i][j] = (p - b * v);
		    if (pop_x[i][j] < m_problem.getBoundary()[j].first) pop_x[i][j] = m_problem.getBoundary()[j].first;
		    if (pop_x[i][j] > m_problem.getBoundary()[j].second) pop_x[i][j] = m_problem.getBoundary()[j].second;
		    
		}
		if (nbV >= (nbVar-1)) nbI++;
		IndividualImpl *tmp = new IndividualImpl(pop_x[i]);
		F[i] = tmp->evaluate();
		delete(tmp);
		Vtmp += F[i]; tt++;
		int m = (int)std::floor(std::exp(-(F[i]-V)*0.001));
		if (m < 0) m = 0;
		if (m > 1){ bestM[i] = m;
		/*  int newPoint = std::distance( flags.begin(), std::find(flags.begin(), flags.end(), 0) );
		    if ( newPoint < szPopMax ) {
			pop_x[newPoint] = pop_x[i]; Nnext++; flags[newPoint]=1;
		    }*/
		}else bestM[i] = 0;
		if (koeff==1){
		    for (int j = 0; j < nbVar; j++)
			pop_pos_best[i][j] = pop_x[i][j];
		    bestF[i] = F[i];
		    
		    globalIndex = getGlobalSolutionIndex(szPop, F);
		    for (int k = 0; k < nbVar; k++){
			    memory[k] = globalSolution[k];
		            globalSolution[k] = pop_pos_best[globalIndex][k];
		    }
		    memoryF = bestGlobal;
		    bestGlobal = bestF[globalIndex];
		    
		}

                if (F[i] < bestF[i]){
                    for (int j = 0; j < nbVar; j++)
                        pop_pos_best[i][j] = pop_x[i][j];
                    bestF[i] = F[i];


                }if (bestF[i] < bestGlobal){
                        for (int j = 0; j < nbVar; j++)
                                globalSolution[j] = pop_pos_best[i][j];
                        bestGlobal = bestF[i];
                }

	    }
	    }V = Vtmp/(double)tt;
	    //V = V + 0.5*(1-(Nnext/(double)Nt));
	    //Nt = Nnext;
	    //szPop = Nnext;

	    double epsilonF = 0.00001;
	    for (int i = 0; i < szPop; i++){
		if (bestM[i] == 0){ /* If weight function of particle is bad - let's make small diffusion displacement */
		    for (int j = 0; j < nbVar; j++)
			pop_x[i][j] = easea::shared::distributions::norm(pop_x[i][j], 0.5*fabs(meanSolution[j]-pop_pos_best[i][j]));
		    IndividualImpl *tmp = new IndividualImpl(pop_x[i]);
		    F[i] = tmp->evaluate();
		    delete(tmp);
		    if (F[i] < bestF[i]){
			pop_pos_best[i] = pop_x[i]; 
			bestF[i] = F[i]; 
			if (bestF[i] < bestGlobal){
			    globalSolution = pop_pos_best[i]; 
			    bestGlobal = bestF[i];
			}
		     }

		}
	    }
	    /* Control of local optimum */
	    if (nbI >= szPop){    
		int count = 0;
		for (int i = 0; i < szPop; i++){
		    double curDerivF = bestF[i] - bestGlobal; 
		    if (fabs(curDerivF - derivF[i]) <= epsilonF) count++;
		    derivF[i] = curDerivF;
		}
		if (count >= szPop-1) nbDeriv++; 
    	    }else nbDeriv = 0;
	    if (bestGlobal < 1) nbDeriv = 0;
	
	    if (nbDeriv >= LIMIT_UPDATE) koeff = 1;
	    else koeff = 0;


		currentGeneration++;
		EASEAEndGeneration(this);
		if (reset == true)
		{
		    currentGeneration = 0;
		    reset = false;
		}
	
		if (bestGlobal < bBest->fitness){
		for (int j = 0; j < nbVar; j++)
		    ((IndividualImpl*)(bBest))->\GENOME_NAME[j] = globalSolution[j];
		
	    	bBest->fitness = bestGlobal;
		}
		ss << "Generation: " << currentGeneration << " Best solution: " << bBest->fitness << " Current best solution: " << bestGlobal << std::endl;
		LOG_MSG(msgType::INFO, ss.str());
	}
	std::chrono::duration<double> tmDur = std::chrono::system_clock::now() - tmStart;
	ostringstream ss;
        ss << "Total execution time (in sec.): " << tmDur.count() << std::endl;
	LOG_MSG(msgType::INFO, ss.str());



	delete(bBest);

}

EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

\START_CUDA_GENOME_H_TPL

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

#include <stdlib.h>
#include <iostream>
#include <CIndividual.h>
#include <Parameters.h>
#include <cstring>
#include <sstream>

\INSERT_USER_HEADER

using namespace std;

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
	virtual ~IndividualImpl() {
		\GENOME_DTOR
	}
	float evaluate() override;
	CIndividual* crossover(CIndividual** p2) override;
	void printOn(std::ostream& O) const override;
	CIndividual* clone() override;

	unsigned mutate(float pMutationPerGene) overrid overridee;

	void boundChecking() override;

	string serialize() override;
	void deserialize(string AESAE_Line) override;
};


class ParametersImpl : public Parameters {
public:
	ParametersImpl(std::string const& file, int argc, char* argv[]);
	CEvolutionaryAlgorithm* newEvolutionaryAlgorithm();
};

void EASEAInit(int argc, char* argv[], ParametersImpl& p);
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

\START_CMAKELISTS
cmake_minimum_required(VERSION 3.9) # 3.9: OpenMP improved support
set(EZ_ROOT $ENV{EZ_PATH})

project(EASEA)
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release")
endif()

file(GLOB EASEA_src ${CMAKE_SOURCE_DIR}/*.cpp ${CMAKE_SOURCE_DIR}/*.c)
add_executable(EASEA ${EASEA_src})

target_compile_features(EASEA PUBLIC cxx_std_17)
target_compile_options(EASEA PUBLIC
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /W3>
	$<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -march=native -mtune=native -Wall -Wextra -pedantic>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/W4 /DEBUG:FULL>
	$<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -Wall -Wextra -pedantic>
	)

find_library(libeasea_LIB
	NAMES libeasea easea
	HINTS ${EZ_ROOT} ${CMAKE_INSTALL_PREFIX}/easea ${CMAKE_INSTALL_PREFIX}/AESAE
	PATH_SUFFIXES lib libeasea easea)
find_path(libeasea_INCLUDE
	NAMES CLogger.h
	HINTS ${EZ_ROOT}/libeasea ${CMAKE_INSTALL_PREFIX}/*/libeasea
	PATH_SUFFIXES include easea libeasea)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
	add_definitions(-DBOOST_ALL_NO_LIB)
	set(Boost_USE_STATIC_LIBS ON)
	set(Boost_USE_MULTITHREADED ON)
	set(Boost_USE_STATIC_RUNTIME OFF)
endif()
find_package(Boost REQUIRED COMPONENTS program_options)

find_package(OpenMP)

target_include_directories(EASEA PUBLIC ${Boost_INCLUDE_DIRS} ${libeasea_INCLUDE})
target_link_libraries(EASEA PUBLIC ${libeasea_LIB} $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX> $<$<CXX_COMPILER_ID:MSVC>:winmm> ${Boost_LIBRARIES})

if (SANITIZE)
        target_compile_options(EASEA PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr> $<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>
)
        target_link_options(EASEA PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr> $<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>)
endif()

\INSERT_USER_CMAKE

\START_EO_PARAM_TPL#****************************************
#                                         
#  EASEA.prm
#                                         
#  Parameter file generated by STD.tpl AESAE v1.0
#                                         
#***************************************
# --seed=0   # -S : Random number seed. It is possible to give a specific seed.

######    Evolution Engine    ######
--popSize=10 # -P : Population Size
--nbOffspring=10 # -O : Nb of offspring (percentage or absolute)

######	  Stopping Criterions    #####
--nbGen=100000 #Nb of generations
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
