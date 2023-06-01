\TEMPLATE_START
/**
 This is program entry for CMAES template for EASEA

*/

\ANALYSE_PARAMETERS
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include "COptionParser.h"
#include "CRandomGenerator.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "EASEAIndividual.hpp"
#include "CCmaes.h"

using namespace std;

/** Global variables for the whole algorithm */
CIndividual** pPopulation = NULL;
CIndividual *bBest = NULL;
float* pEZ_MUT_PROB = NULL;
float* pEZ_XOVER_PROB = NULL;
unsigned *EZ_NB_GEN;
unsigned *EZ_current_generation;
int EZ_POP_SIZE;
int OFFSPRING_SIZE;

CEvolutionaryAlgorithm* EA;

CCmaes *cma;

int main(int argc, char** argv){
	ParametersImpl p("EASEA.prm", argc, argv);

	CEvolutionaryAlgorithm* ea = p.newEvolutionaryAlgorithm();

	EA = ea;

	EASEAInit(argc,argv,p);

	CPopulation* pop = ea->getPopulation();

	ea->runEvolutionaryLoop();

	EASEAFinal(pop);

	delete pop;

	return 0;
}

\START_CUDA_GENOME_CU_TPL

#include <fstream>
#include <time.h>
#include <string>
#include <sstream>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"
#include "CCmaes.h"

#include "config.h"
#ifdef USE_OPENMP
	#include <omp.h>
#endif


using namespace std;
bool bReevaluate = false;

#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;
#define STD_TPL

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

\INSERT_USER_CLASSES

\INSERT_USER_FUNCTIONS

\INSERT_FINALIZATION_FUNCTION

void evale_pop_chunk([[maybe_unused]] CIndividual** population, [[maybe_unused]] int popSize) {
  \INSTEAD_EVAL_FUNCTION
}

void EASEAInit(int argc, char** argv, ParametersImpl& p){
	(void)argc;(void)argv;(void)p;
	auto setVariable = [&](std::string const& arg, auto def) {
		return p.setVariable(arg, std::forward<decltype(def)>(def));
	}; // for compatibility
	(void)setVariable;

	\INSERT_INITIALISATION_FUNCTION
  	cma = new CCmaes(p.offspringPopulationSize, p.parentPopulationSize, \PROBLEM_DIM);
}

void EASEAFinal(CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;
	delete(cma);
}

void AESAEBeginningGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
    cma->cmaes_UpdateEigensystem(0);
    cma->TestMinStdDevs();
    \INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_END_GENERATION_FUNCTION
	evolutionaryAlgorithm->population->sortParentPopulation();
	double **popparent;
	double *fitpar;
	popparent = (double**)malloc(evolutionaryAlgorithm->population->parentPopulationSize*sizeof(double *));
	fitpar = (double*)malloc(evolutionaryAlgorithm->population->parentPopulationSize*sizeof(double));
	for(int i=0; i<(signed)evolutionaryAlgorithm->population->parentPopulationSize; i++)
		popparent[i] = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	for(int i=0; i<(signed)evolutionaryAlgorithm->population->parentPopulationSize; i++){
		for(int j=0; j<\PROBLEM_DIM; j++){
			IndividualImpl *tmp = (IndividualImpl *)evolutionaryAlgorithm->population->parents[i];
			popparent[i][j] = tmp->\GENOME_NAME[j];
			//cout << popparent[i][j] << "|";
		}
		fitpar[i] = evolutionaryAlgorithm->population->parents[i]->fitness;
		//cout << fitpar[i] << endl;
	}
	cma->cmaes_update(popparent, fitpar);
	for(int i=0; i<(signed)evolutionaryAlgorithm->population->parentPopulationSize; i++){
		free(popparent[i]);
	}
	free(popparent);
	free(fitpar);

}

void AESAEGenerationFunctionBeforeReplacement([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
        \INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}



IndividualImpl::IndividualImpl() : CIndividual() {
  \GENOME_CTOR 
  for(int i=0; i<\PROBLEM_DIM; i++ ) {
	this->\GENOME_NAME[i] = 0.5 + (cma->sigma * cma->rgD[i] * cma->alea.alea_Gauss());
  }
  valid = false;
  isImmigrant = false;
}

CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

IndividualImpl::~IndividualImpl(){
  \GENOME_DTOR
}


float IndividualImpl::evaluate(){
  if(valid)
    return fitness;
  else{
    valid = true;
    \INSERT_EVALUATOR
  }
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
    this->isImmigrant=false;
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
  	for (int i = 0; i < \PROBLEM_DIM; ++i)
		cma->rgdTmp[i] = cma->rgD[i] * cma->alea.alea_Gauss();

	child.valid = false;
	/*   cout << "child : " << child << endl; */
	return new IndividualImpl(child);
}


void IndividualImpl::printOn([[maybe_unused]] std::ostream& os) const {
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


unsigned IndividualImpl::mutate([[maybe_unused]] float pMutationPerGene ) {
  this->valid=false;


  // ********************
  // Problem specific part
  double sum;
  for (int i = 0; i < \PROBLEM_DIM; ++i) {
	sum = 0.;
	for (int j = 0; j < \PROBLEM_DIM; ++j)
		sum += cma->B[i][j] * cma->rgdTmp[j];
	this->\GENOME_NAME[i] = cma->rgxmean[i] + cma->sigma * sum;
  }

  return 0;
}

ParametersImpl::ParametersImpl(std::string const& file, int argc, char* argv[]) : Parameters(file, argc, argv) {

        this->minimizing =1;
        this->nbGen = setVariable("nbGen", (int)\NB_GEN);
	#ifdef USE_OPENMP
	omp_set_num_threads(nbCPUThreads);
	#endif

        globalRandomGenerator = new CRandomGenerator(seed);
        this->randomGenerator = globalRandomGenerator;

        selectionOperator = getSelectionOperator(setVariable("selectionOperator", "Tournament"), this->minimizing, globalRandomGenerator);
        replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator", "\RED_FINAL_OPERATOR"),this->minimizing, globalRandomGenerator);
        selectionPressure = 1;
        replacementPressure = setVariable("reduceFinalPressure", (float)\RED_FINAL_PRM);
	parentReductionOperator = NULL;
	offspringReductionOperator = NULL;
	offspringReductionPressure = 1.;
	parentReductionPressure = 1.;
        pCrossover = 1;
        pMutation = 1;
        pMutationPerGene = 1;

        parentPopulationSize = setVariable("popSize", (int)\POP_SIZE);
        offspringPopulationSize = getOffspringSize((int)\OFF_SIZE, \POP_SIZE);

        this->elitSize = 0;

	offspringReduction = parentReduction = false;

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
        this->ipFile = (char*)setVariable("ipFile", "\IP_FILE").c_str();
        this->migrationProbability = setVariable("migrationProbability", (float)\MIGRATION_PROBABILITY);
        this->serverPort = setVariable("serverPort", \SERVER_PORT);
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	//EZ_NB_GEN = (unsigned*)setVariable("nbGen", \NB_GEN);
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
        if(this->params->startFromFile){
          ifstream AESAE_File(this->params->inputFilename);
          string AESAE_Line;
          for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
                  getline(AESAE_File, AESAE_Line);
                  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
                  ((IndividualImpl*)this->population->parents[i])->deserialize(AESAE_Line);
          }

        }
        else {
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for(int i=0 ; i< static_cast<int>(this->params->parentPopulationSize) ; i++){
                  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
          }
        }
        this->population->actualParentPopulationSize = this->params->parentPopulationSize;
}


EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){

}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

\START_CUDA_GENOME_H_TPL

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

//#include "CRandomGenerator.h"
#include <stdlib.h>
#include <iostream>
#include <string>
#include <CIndividual.h>
#include <Parameters.h>
#include <CCmaes.h>

using namespace std;

class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class CCmaes;
class Parameters;

extern int EZ_POP_SIZE;
extern int OFFSPRING_SIZE;

extern CCmaes *cma;

\INSERT_USER_CLASSES_DEFINITIONS

class IndividualImpl : public CIndividual {

public: // in EASEA the genome is public (for user functions,...)
	// Class members
  	\INSERT_GENOME


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
        void deserialize(string AESAE_Line);

	friend std::ostream& operator << (std::ostream& O, const IndividualImpl& B) ;
	void initRandomGenerator(CRandomGenerator* rg){ IndividualImpl::rg = rg;}

};


class ParametersImpl : public Parameters {
public:
	ParametersImpl(std::string const& file, int argc, char* argv[]);
	CEvolutionaryAlgorithm* newEvolutionaryAlgorithm();
};

void EASEAInit(int argc, char* argv[], ParametersImpl& p);
void EASEAFinal(CPopulation* pop);
void EASEABeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm);


class EvolutionaryAlgorithmImpl: public CEvolutionaryAlgorithm {
public:
	EvolutionaryAlgorithmImpl(Parameters* params);
	virtual ~EvolutionaryAlgorithmImpl();
	void initializeParentPopulation();
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
	HINTS ${EZ_ROOT} ${CMAKE_INSTALL_PREFIX}/easena ${CMAKE_INSTALL_PREFIX}/EASENA
	PATH_SUFFIXES lib libeasea easea easena)
find_path(libeasea_INCLUDE
	NAMES CLogger.h
	HINTS ${EZ_ROOT}/libeasea ${CMAKE_INSTALL_PREFIX}/*/libeasea
	PATH_SUFFIXES include easena libeasea)
find_package(Boost)
find_package(OpenMP)

target_include_directories(EASEA PUBLIC ${Boost_INCLUDE_DIRS} ${libeasea_INCLUDE})
target_link_libraries(EASEA PUBLIC ${libeasea_LIB} $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX> $<$<CXX_COMPILER_ID:MSVC>:winmm>)

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
#  Parameter file generated by CMAES.tpl AESAE v0.1
#
#***************************************
# --seed=0   # -S : Random number seed. It is possible to give a specific seed.

######    Evolution Engine    ######
--popSize=\POP_SIZE # -P : Population Size
--nbOffspring=\OFF_SIZE # -O : Nb of offspring (percentage or absolute)

######    Stopping Criterions    #####
--nbGen=\NB_GEN #Nb of generations
--timeLimit=\TIME_LIMIT # Time Limit: desactivate with (0) (in Seconds)

######    Evolution Engine / Replacement    ######
--elite=\ELITE_SIZE  # Nb of elite parents (absolute)
--eliteType=\ELITISM # Strong (1) or weak (0) elitism (set elite to 0 for none)
--reduceFinalOperator=\RED_FINAL_OPERATOR
--reduceFinalPressure=\RED_FINAL_PRM

#####   Stats Ouput     #####
--printStats=\PRINT_STATS #print Stats to screen
--plotStats=\PLOT_STATS #plot Stats 
--printInitialPopulation=0 #Print initial population
--printFinalPopulation=0 #Print final population
--generateCSVFile=\GENERATE_CSV_FILE
--generatePlotScript=\GENERATE_GNUPLOT_SCRIPT
--generateRScript=\GENERATE_R_SCRIPT

#### Population save    ####
--savePopulation=\SAVE_POPULATION #save population to EASEA.pop file
--startFromFile=\START_FROM_FILE #start optimisation from EASEA.pop file

#### Remote Island Model ####
--remoteIslandModel=\REMOTE_ISLAND_MODEL #To initialize communications with remote AESAE's
--ipFile=\IP_FILE
--migrationProbability=\MIGRATION_PROBABILITY #Probability to send an individual every generation
--serverPort=\SERVER_PORT
\TEMPLATE_END
