\TEMPLATE_START
/***********************************************************************
| Classical DE  Template                                                |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2023-03                                                         |
|                                                                       |
 ***********************************************************************/


\ANALYSE_PARAMETERS
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <CLogger.h>
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
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include <CLogger.h>


#include <problems/CProblem.h>
#include <algorithms/CAlgorithmWrapper.h>
#include <operators/crossover/C3x1CrossoverLauncher.h>
#include <variables/continuous/uniform.h>
#include <algorithms/soea/Ccde.h>
#include <problems/CProblem.h>
#include <algorithms/CAlgorithmWrapper.h>
#include <operators/crossover/continuous/CdeCrossover.h>

#define NB_OBJECTIVES 1
#define NB_CONSTRAINTS 0


#include "config.h"
#ifdef USE_OPENMP
	#include <omp.h>
#endif


using namespace std;
bool bReevaluate = false;

#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

extern CEvolutionaryAlgorithm* EA;

typedef std::mt19937 TRandom;

#ifndef CUSTOM_PRECISION_TYPE
	typedef double TT;
#else
	using TT = CUSTOM_PRECISION_TYPE;
#endif

typedef easea::problem::CProblem<TT> TP;
typedef TP::TV TV;
typedef TP::TO TO;

typedef typename easea::Individual<TT, TV> TIndividual;
typedef typename easea::shared::CBoundary<TT>::TBoundary TBoundary;
typedef easea::algorithms::cde::Ccde< TIndividual, TRandom &> TAlgorithm;
typedef easea::operators::crossover::continuous::de::CdeCrossover<TT, TRandom &> TCrossover;

TAlgorithm *m_algorithm;
size_t m_popSize = -1;
std::time_t m_seed = std::time(nullptr);
TRandom m_generator{static_cast<unsigned int>(m_seed)};

\INSERT_USER_DECLARATIONS

\ANALYSE_USER_CLASSES


\INSERT_USER_FUNCTIONS

\INSERT_FINALIZATION_FUNCTION



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

void EASEAFinal([[maybe_unused]] CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;

        delete(m_algorithm);
	LOG_MSG(msgType::INFO, "Classical DE finished");

}

void AESAEBeginningGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_END_GENERATION_FUNCTION
}

void AESAEGenerationFunctionBeforeReplacement([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}

template <typename TO, typename TV>
easea::Individual<TO, TV>::Individual(void)
{
}

template <typename TO, typename TV>
size_t easea::Individual<TO, TV>::evaluate()
{
        \INSERT_EVALUATOR
}



ParametersImpl::ParametersImpl(std::string const& file, int argc, char* argv[]) : Parameters(file, argc, argv) {

	this->minimizing = \MINIMAXI;
	this->nbGen = setVariable("nbGen", (int)\NB_GEN);

	#ifdef USE_OPENMP
	omp_set_num_threads(nbCPUThreads);
	#endif

        parentReductionPressure = setVariable("reduceParentsPressure", (float)\RED_PAR_PRM);
        offspringReductionPressure = setVariable("reduceOffspringPressure", (float)\RED_OFF_PRM);

	pCrossover = \XOVER_PROB;
	pMutation = \MUT_PROB;
	pMutationPerGene = 0.05;

	parentPopulationSize = setVariable("popSize", (int)\POP_SIZE);
	offspringPopulationSize = getOffspringSize((int)\OFF_SIZE, \POP_SIZE);
	m_popSize = parentPopulationSize;
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

        if (!this->noLogFile) {

        auto tmStart = std::chrono::system_clock::now();
        time_t t = std::chrono::system_clock::to_time_t(tmStart);
        std::tm * ptm = std::localtime(&t);
        char buf_start_time[32];
        string log_fichier_name = this->outputFilename;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tmStart.time_since_epoch()) % 1000;
	if (!this->noLogFile) {
	    std::strftime(buf_start_time, 32, "%Y-%m-%d_%H-%M-%S", ptm);
	    easena::log_file.open(log_fichier_name.c_str() + std::string("_") + std::string(buf_start_time) + std::string("-") + std::to_string(ms.count()) + std::string(".log"));
	    logg("DATA of TEST;", std::string(buf_start_time).c_str());
	    logg("\n__RUN SETTINGS__");
	    logg("SEED;", m_seed);
	    logg("nCURRENT_GEN;", this->nbGen);
	    logg("POP_SIZE;", this->parentPopulationSize);
	    logg("CPU_THREADS_NB;", nbCPUThreads);
	}
    }


	this->remoteIslandModel = setVariable("remoteIslandModel", \REMOTE_ISLAND_MODEL);
	this->ipFile = setVariable("ipFile", "\IP_FILE");
	this->migrationProbability = setVariable("migrationProbability", (float)\MIGRATION_PROBABILITY);
        this->serverPort = setVariable("serverPort", \SERVER_PORT);
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){
	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	EZ_current_generation=0;
        EZ_POP_SIZE = parentPopulationSize;
        OFFSPRING_SIZE = offspringPopulationSize;

	if (m_popSize <= 0){ LOG_ERROR(errorCode::value, "Wrong size of parent population");  };
    	const std::vector<TV> initPop = easea::variables::continuous::uniform(m_generator, m_problem.getBoundary(), m_popSize);

    	m_algorithm  = new TAlgorithm(m_generator, m_problem, initPop, m_crossover);

	CEvolutionaryAlgorithm* ea = new CAlgorithmWrapper(this, m_algorithm);
	generationalCriterion->setCounterEa(ea->getCurrentGenerationPtr());
	ea->addStoppingCriterion(generationalCriterion);
	ea->addStoppingCriterion(controlCStopingCriterion);
	ea->addStoppingCriterion(timeCriterion);

	EZ_NB_GEN=((CGenerationalCriterion*)ea->stoppingCriteria[0])->getGenerationalLimit();
	EZ_current_generation=&(ea->currentGeneration);

	return ea;
}

\START_CUDA_GENOME_H_TPL

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

#include <stdlib.h>
#include <iostream>
#include <core/CmoIndividual.h>
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

namespace easea
{

template <typename TObjective, typename TVariable>
class Individual : public easea::CmoIndividual<TObjective, TVariable>
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CmoIndividual<TO, TV> TI;

        Individual(void);
        ~Individual() = default;
	size_t evaluate() override;
};
}

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

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
	add_definitions(-DBOOST_ALL_NO_LIB)
	set(Boost_USE_STATIC_LIBS ON)
	set(Boost_USE_MULTITHREADED ON)
	set(Boost_USE_STATIC_RUNTIME OFF)
endif()
find_package(Boost REQUIRED COMPONENTS iostreams serialization program_options)

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
--popSize=\POP_SIZE # -P : Population Size
--nbOffspring=\OFF_SIZE # -O : Nb of offspring (percentage or absolute)

######	  Stopping Criterions    #####
--nbGen=\NB_GEN #Nb of generations
--timeLimit=\TIME_LIMIT # Time Limit: desactivate with (0) (in Seconds)

#####	Stats Ouput 	#####
--printStats=\PRINT_STATS #print Stats to screen

#### Population save	####
--savePopulation=\SAVE_POPULATION #save population to EASEA.pop file
--startFromFile=\START_FROM_FILE #start optimisation from EASEA.pop file

#### Remote Island Model ####
--remoteIslandModel=\REMOTE_ISLAND_MODEL #To initialize communications with remote AESAE's
--ipFile=\IP_FILE
--migrationProbability=\MIGRATION_PROBABILITY #Probability to send an individual every generation
--serverPort=\SERVER_PORT
\TEMPLATE_END
