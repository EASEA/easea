\TEMPLATE_START
/***********************************************************************
| MOEAD Multi objective algorithm template                              |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2019-05                                                         |
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

	EA = ea;

	EASEAInit(argc, argv, p);

	CPopulation* pop = ea->getPopulation();

	ea->runEvolutionaryLoop();

	EASEAFinal(pop);

	delete pop;

	return 0;
}

\START_CUDA_GENOME_CU_TPL

#include <chrono>
#include <fstream>
#include <time.h>
#include <cstring>
#include <sstream>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"

#include <CLogger.h>
#include <variables/continuous/uniform.h>
#include <shared/functions/ndi.h>
#include <shared/functions/weight.h>
#include <algorithms/moea/Cmoead.h>

#include <CQMetrics.h>
#include <CQMetricsHV.h>
#include <CQMetricsGD.h>
#include <CQMetricsIGD.h>
#include <problems/CProblem.h>
#include <algorithms/CAlgorithmWrapper.h>
#include <operators/crossover/C2x2CrossoverLauncher.h>

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
typedef std::mt19937 TRandom;
typedef double TT;
typedef easea::problem::CProblem<TT> TP;
typedef TP::TV TV;
typedef TP::TO TO;

typedef typename easea::Individual<TT, TV> TIndividual;
typedef typename easea::shared::CBoundary<TT>::TBoundary TBoundary;

\INSERT_USER_DECLARATIONS
TCrossover m_crossover = crossover;
//easea::operators::crossover::CWrap2x2Crossover<TT, TV>(crossover);
//easea::operators::crossover::C2x2CrossoverLauncher<TT, TV, TRandom &> m_crossover(crossover, m_generator);

typedef easea::algorithms::moead::Cmoead< TIndividual, TRandom &> TAlgorithm;
TAlgorithm *m_algorithm;
size_t m_popSize = -1;

\ANALYSE_USER_CLASSES


\INSERT_USER_FUNCTIONS
/*
 * \brief Set number of reference point division
    
 */ 

size_t setNumberOfReferencePointDiv( const int nbObjectives)
{
    size_t division;

    if (nbObjectives == 1) division = 100;
    else if (nbObjectives == 2) division = 99;
    else if (nbObjectives == 3) division = 12;
    else if (nbObjectives == 4) division = 8;
    else if (nbObjectives == 5) division = 6;
    else if (nbObjectives == 6) division = 5;
    else if (nbObjectives == 7) division = 3;
    else if (nbObjectives == 8) division = 3;
    else if (nbObjectives == 9) division = 3;
    else if (nbObjectives == 10) division = 3;
    else division = 2;

    return division;
}

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

void EASEAFinal(CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;
/*EASEAFinalization(pop);*/
        string file_o = "objectives";
        string file_s = "solutions";
        std::ofstream out_o(file_o.c_str());
        std::ofstream out_s(file_s.c_str());
        cout.setf(ios::fixed);
	
	LOG_MSG(msgType::INFO, "Saving Pareto Front in file objectives");

        const auto &population = m_algorithm->getPopulation();
        for (size_t i = 0; i < population.size(); ++i)
        {
                const auto &objective = population[i].m_objective;
                const auto &solution = population[i].m_variable;
                for (size_t j = 0; j < objective.size(); ++j)
                    out_o << objective[j] << ' ';
                out_o << endl;
                for(size_t j = 0; j < solution.size(); j++)
                    out_s << solution[j] << ' ';
                out_s << endl;
    	}
    	out_o.close();
        out_s.close();
	LOG_MSG(msgType::INFO, "Pareto Front is saved ");


#ifdef QMETRICS
        LOG_MSG(msgType::INFO, "Calculating performance metrics ");
	LOG_MSG(msgType::INFO, "Statistic begin");

        auto metrics = make_unique<CQMetrics>("objectives", PARETO_TRUE_FILE, m_problem.getNumberOfObjectives());
        auto hv = metrics->getMetric<CQMetricsHV>();
        auto gd = metrics->getMetric<CQMetricsGD>();
        auto igd = metrics->getMetric<CQMetricsIGD>();
        std::ostringstream statInfo;
        statInfo << "Quality Metrics: " << std::endl
        << "HyperVolume = " << hv << std::endl
        << "Generational distance = " << gd << std::endl
        << "Inverted generational distance  = " << igd << std::endl;
        auto statistics = (statInfo.str());
        LOG_MSG(msgType::INFO, statistics);
	LOG_MSG(msgType::INFO, "Statistic end");


#endif

     delete(m_algorithm);
     LOG_MSG(msgType::INFO, "MOEAD finished");

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
        m_minDistance = -1;
}

template <typename TO, typename TV>
easea::Individual<TO, TV>::~Individual(void)
{
}

template <typename TO, typename TV>
size_t easea::Individual<TO, TV>::evaluate()
{
/*    if(valid)
        return fitness;
    else{
       valid = true;*/
        \INSERT_EVALUATOR
//    }

}


ParametersImpl::ParametersImpl(std::string const& file, int argc, char* argv[]) : Parameters(file, argc, argv) {
	this->minimizing = \MINIMAXI;
	this->nbGen = setVariable("nbGen", (int)\NB_GEN);

	globalRandomGenerator = new CRandomGenerator(seed);
	this->randomGenerator = globalRandomGenerator;

	#ifdef USE_OPENMP
	omp_set_num_threads(nbCPUThreads);
	#endif

/*	selectionOperator = getSelectionOperator(setVariable("selectionOperator", "\SELECTOR_OPERATOR"), this->minimizing, globalRandomGenerator);
	replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator", "\RED_FINAL_OPERATOR"),this->minimizing, globalRandomGenerator);
	parentReductionOperator = getSelectionOperator(setVariable("reduceParentsOperator", "\RED_PAR_OPERATOR"),this->minimizing, globalRandomGenerator);
	offspringReductionOperator = getSelectionOperator(setVariable("reduceOffspringOperator", "\RED_OFF_OPERATOR"),this->minimizing, globalRandomGenerator);
	selectionPressure = setVariable("selectionPressure", (float)\SELECT_PRM);
	replacementPressure = setVariable("reduceFinalPressure", (float)\RED_FINAL_PRM);
	parentReductionPressure = setVariable("reduceParentsPressure", (float)\RED_PAR_PRM);
	offspringReductionPressure = setVariable("reduceOffspringPressure", (float)\RED_OFF_PRM);
*/	pCrossover = \XOVER_PROB;
	pMutation = \MUT_PROB;
	pMutationPerGene = 0.05;

	parentPopulationSize = setVariable("popSize", (int)\POP_SIZE);
	offspringPopulationSize = getOffspringSize((int)\OFF_SIZE, \POP_SIZE);
	m_popSize = parentPopulationSize;

	/*parentReductionSize = setReductionSizes(parentPopulationSize, setVariable("survivingParents", (float)\SURV_PAR_SIZE));
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
	*/

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
	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	EZ_current_generation=0;
        EZ_POP_SIZE = parentPopulationSize;
        OFFSPRING_SIZE = offspringPopulationSize;

        if (m_popSize <= 0){ LOG_ERROR(errorCode::value, "Wrong size of parent population"); };
        const size_t nbObjectives = m_problem.getNumberOfObjectives();

        size_t division = setNumberOfReferencePointDiv(nbObjectives);
        auto weight = easea::shared::function::runNbi<TO>(nbObjectives, 43/*division*/);
        std::vector<std::vector<TO>> tmp_weight(weight.begin(), weight.end());
        for (size_t i = 0; i < tmp_weight.size(); ++i){
                easea::shared::function::adjustWeight(tmp_weight[i], 0.00001);
		printf("WEIGHT: %i\n",tmp_weight.size());}
        const std::vector<TV> initPop = easea::variables::continuous::uniform(m_generator, m_problem.getBoundary(), tmp_weight.size()/* m_popSize*/);

        m_algorithm  = new TAlgorithm(m_generator, m_problem, initPop, m_crossover, m_mutation, tmp_weight, initPop.size() / 10);

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

//#include "CRandomGenerator.h"
#include <stdlib.h>
#include <iostream>
#include <core/CmoIndividual.h>
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

namespace easea
{
template <typename TObjective, typename TVariable>
class Individual : public easea::CmoIndividual<TObjective, TVariable>
{
public:
        typedef TObjective TO;
        typedef TVariable TV;
        typedef CmoIndividual<TO, TV> TI;

        TO m_minDistance;
	std::vector<TO> m_trObjective;
	float fitness; 		// this is variable for return the value 1 from function evaluate()

        Individual(void);
        ~Individual(void);
	size_t evaluate();
};
}

class ParametersImpl : public Parameters {
public:
	ParametersImpl(std::string const& file, int argc, char* argv[]);
	CEvolutionaryAlgorithm* newEvolutionaryAlgorithm();
};

/**
 * @TODO ces functions devraient s'appeler weierstrassInit, weierstrassFinal etc... (en gros EASEAFinal dans le tpl).
 *
 */

void EASEAInit(int argc, char* argv[], ParametersImpl& p);
void EASEAFinal(CPopulation* pop);
void EASEABeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm);

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

CXXFLAGS =  -std=c++14 -O3 -DNDEBUG -fopenmp  -w -Wno-deprecated -Wno-write-strings -fmessage-length=0 -I$(EASEALIB_PATH)include

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
find_package(Boost REQUIRED COMPONENTS iostreams serialization)

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
