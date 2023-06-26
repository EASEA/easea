\TEMPLATE_START
/***********************************************************************
| NSGA-II Multi Objective Algorithm Template                            |
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


#include <CQMetrics.h>
#include <CQMetricsHV.h>
#include <CQMetricsGD.h>
#include <CQMetricsIGD.h>
#include <problems/CProblem.h>
#include <algorithms/CAlgorithmWrapper.h>
#include <operators/crossover/C2x2CrossoverLauncher.h>
#include <variables/continuous/uniform.h>
#include <algorithms/moea/Cnsga-ii.h>

#include "config.h"
#ifdef USE_OPENMP
	#include <omp.h>
#endif


using namespace std;
bool bReevaluate = false;

#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

extern CEvolutionaryAlgorithm* EA;
#define STD_TPL
typedef std::mt19937 TRandom;
typedef double TT;
typedef easea::problem::CProblem<TT> TP;
typedef TP::TV TV;
typedef TP::TO TO;

typedef typename easea::Individual<TT, TV> TIndividual;
typedef typename easea::shared::CBoundary<TT>::TBoundary TBoundary;

std::time_t m_seed = std::time(nullptr);
TRandom m_generator{static_cast<std::mt19937::result_type>(m_seed)};

typedef easea::algorithms::nsga_ii::Cnsga_ii< TIndividual, TRandom &> TAlgorithm;
TAlgorithm *m_algorithm;
size_t m_popSize = -1;


\INSERT_USER_DECLARATIONS
easea::operators::crossover::C2x2CrossoverLauncher<TT, TV, TRandom &> m_crossover(crossover, m_generator);


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
        out_o.close();
        out_s.close();
        delete(m_algorithm);
	 LOG_MSG(msgType::INFO, "NSGAII finished");

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
        m_crowdingDistance = -1;
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

	this->remoteIslandModel = setVariable("remoteIslandModel", \REMOTE_ISLAND_MODEL);
	this->ipFile = setVariable("ipFile", "\IP_FILE");
	this->migrationProbability = setVariable("migrationProbability", (float)\MIGRATION_PROBABILITY);
	this->serverPort = setVariable("serverPort", \SERVER_PORT);

        if (!this->noLogFile) {
    	    auto tmStart = std::chrono::system_clock::now();
            time_t t = std::chrono::system_clock::to_time_t(tmStart);
            std::tm * ptm = std::localtime(&t);
            char buf_start_time[32];
            string log_fichier_name = this->outputFilename;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tmStart.time_since_epoch()) % 1000;
            std::strftime(buf_start_time, 32, "%Y-%m-%d_%H-%M-%S", ptm);
	    easena::log_file.open(log_fichier_name.c_str() +std::string("_") + std::string(buf_start_time) +std::string("-") + std::to_string(ms.count()) + std::string(".log"));
            logg("DATA of TEST;", std::string(buf_start_time).c_str());
            logg("\n__RUN SETTINGS__");
            logg("SEED;", m_seed);
            logg("nCURRENT_GEN;", this->nbGen);
            logg("POP_SIZE;", this->parentPopulationSize);
            logg("CPU_THREADS_NB;", nbCPUThreads);
       }
     

}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){
	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	EZ_current_generation=0;
        EZ_POP_SIZE = parentPopulationSize;
        OFFSPRING_SIZE = offspringPopulationSize;

	if (m_popSize <= 0){ LOG_ERROR(errorCode::value, "Wrong size of parent population");  };
	const std::vector<TV> initPop = easea::variables::continuous::uniform(m_generator, m_problem.getBoundary(), m_popSize);

	m_algorithm  = new TAlgorithm(m_generator, m_problem, initPop, m_crossover, m_mutation);

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

        TO m_crowdingDistance;

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
void AESAEBeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void AESAEEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void AESAEGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm);

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
