\TEMPLATE_START
/**
 This is program entry for GP template for EASEA

*/

\ANALYSE_PARAMETERS
\ANALYSE_GP_OPCODE
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
std::vector<char *> vArgv;

CEvolutionaryAlgorithm* EA;

int main(int argc, char** argv){
	if (argc > 1){
    	    for (int i = 1; i < argc; i++){
        	if ((argv[i][0]=='-')&&(argv[i][1]=='-')) break;
            	    vArgv.push_back(argv[i]);
    	    }
    	}

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
#include <string>
#include <sstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"

#include "config.h"
#ifdef USE_OPENMP
	#include <omp.h>
#endif


using namespace std;
bool bReevaluate = false;

#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

extern CEvolutionaryAlgorithm* EA;
#define GP_TPL



unsigned aborded_crossover;
float** inputs;
float* outputs;


\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

\INSERT_GP_PARAMETERS

GPNode* ramped_hh(){
  return RAMPED_H_H(INIT_TREE_DEPTH_MIN,INIT_TREE_DEPTH_MAX,EA->population->actualParentPopulationSize,EA->population->parentPopulationSize,0, VAR_LEN, OPCODE_SIZE,opArity, OP_ERC);
}

std::string toString(GPNode* root){
  return toString(root,opArity,opCodeName,OP_ERC);
}


\INSERT_USER_CLASSES
\INSERT_USER_FUNCTIONS

\INSERT_FINALIZATION_FUNCTION

float recEval(GPNode* root, float* input) {
  float OP1=0, OP2= 0, RESULT = 0;
  if( opArity[(int)root->opCode]>=1) OP1 = recEval(root->children[0],input);
  if( opArity[(int)root->opCode]>=2) OP2 = recEval(root->children[1],input);
  switch( root->opCode ){
\INSERT_GP_CPU_SWITCH
  default:
    fprintf(stderr,"error unknown terminal opcode %d\n",root->opCode);
    exit(-1);
  }
  return RESULT;
}


GPNode* pickNthNode(GPNode* root, int N, int* childId){

  GPNode* stack[TREE_DEPTH_MAX*MAX_ARITY];
  GPNode* parentStack[TREE_DEPTH_MAX*MAX_ARITY];
  int stackPointer = 0;

  parentStack[stackPointer] = NULL;
  stack[stackPointer++] = root;

  for( int i=0 ; i<N ; i++ ){
    GPNode* currentNode = stack[stackPointer-1];
    stackPointer--;
    for( int j=opArity[(int)currentNode->opCode] ; j>0 ; j--){
      parentStack[stackPointer] = currentNode;
      stack[stackPointer++] = currentNode->children[j-1];
    }
  }

  //assert(stackPointer>0);
  if( stackPointer )
    stackPointer--;

  for( unsigned i=0 ; i<opArity[(int)parentStack[stackPointer]->opCode] ; i++ ){
    if( parentStack[stackPointer]->children[i]==stack[stackPointer] ){
      (*childId)=i;
      break;
    }
  }
  return parentStack[stackPointer];
}

void simple_mutator(IndividualImpl* Genome){

  // Cassical  mutation
  // select a node
  int mutationPointChildId = 0;
  int mutationPointDepth = 0;
  GPNode* mutationPointParent = selectNode(Genome->root, &mutationPointChildId, &mutationPointDepth, opArity);
  
  
  if( !mutationPointParent ){
    mutationPointParent = Genome->root;
    mutationPointDepth = 0;
  }
  delete mutationPointParent->children[mutationPointChildId] ;
  mutationPointParent->children[mutationPointChildId] =
    construction_method( VAR_LEN+1, OPCODE_SIZE , 1, TREE_DEPTH_MAX-mutationPointDepth ,0,opArity,OP_ERC);
}


void simpleCrossOver(IndividualImpl& p1, IndividualImpl& p2, IndividualImpl& c){
  int depthP1 = depthOfTree(p1.root, opArity);
  int depthP2 = depthOfTree(p2.root, opArity);

  int nbNodeP1 = enumTreeNodes(p1.root, opArity);
   int nbNodeP2 = enumTreeNodes(p2.root, opArity);

  int stockPointChildId=0;
  int graftPointChildId=0;

  bool stockCouldBeTerminal = tossCoin(0.1f);
  bool graftCouldBeTerminal = tossCoin(0.1f);

  int childrenDepth = 0, Np1 = 0 , Np2 = 0;
  GPNode* stockParentNode = NULL;
  GPNode* graftParentNode = NULL;

  unsigned tries = 0;
  do{
  choose_node:
    
    tries++;
    if( tries>=10 ){
      aborded_crossover++;
      Np1=0;
      Np2=0;
      break;
    }

    if( nbNodeP1<2 ) Np1=0;
    else Np1 = (int)random((int)0,(int)nbNodeP1);
    if( nbNodeP2<2 ) Np2=0;
    else Np2 = (int)random((int)0,(int)nbNodeP2);


    
    if( Np1!=0 ) stockParentNode = pickNthNode(c.root, MIN(Np1,nbNodeP1) ,&stockPointChildId);
    if( Np2!=0 ) graftParentNode = pickNthNode(p2.root, MIN(Np2,nbNodeP1) ,&graftPointChildId);

    // is the stock and the graft an authorized type of node (leaf or inner-node)
    if( Np1 && !stockCouldBeTerminal && opArity[(int)stockParentNode->children[stockPointChildId]->opCode]==0 ) goto choose_node;
    if( Np2 && !graftCouldBeTerminal && opArity[(int)graftParentNode->children[graftPointChildId]->opCode]==0 ) goto choose_node;
    
    if( Np2 && Np1)
      childrenDepth = depthOfNode(c.root,stockParentNode, opArity)+depthOfTree(graftParentNode->children[graftPointChildId], opArity);
    else if( Np1 ) childrenDepth = depthOfNode(c.root,stockParentNode, opArity)+depthP1;
    else if( Np2 ) childrenDepth = depthOfTree(graftParentNode->children[graftPointChildId], opArity);
    else childrenDepth = depthP2;
    
  }while( childrenDepth>TREE_DEPTH_MAX );

  
  if( Np1 && Np2 ){
    delete stockParentNode->children[stockPointChildId];
    stockParentNode->children[stockPointChildId] = graftParentNode->children[graftPointChildId];
    graftParentNode->children[graftPointChildId] = NULL;
  }
  else if( Np1 ){ // && Np2==NULL
    // We want to use the root of the parent 2 as graft
    delete stockParentNode->children[stockPointChildId];
    stockParentNode->children[stockPointChildId] = p2.root;
    p2.root = NULL;
  }else if( Np2 ){ // && Np1==NULL
    // We want to use the root of the parent 1 as stock
    delete c.root;
    c.root = graftParentNode->children[graftPointChildId];
    graftParentNode->children[graftPointChildId] = NULL;
  }else{
    // We want to switch root nodes between parents
    delete c.root;
    c.root  = p2.root;
    p2.root = NULL;
  }
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

void AESAEEndGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_END_GENERATION_FUNCTION
}

void AESAEGenerationFunctionBeforeReplacement([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}


IndividualImpl::IndividualImpl() : CIndividual() {
  \GENOME_CTOR 
  \INSERT_EO_INITIALISER
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
  float error; 
 float sum = 0;
  \INSERT_GENOME_EVAL_HDR

   for( int i=0 ; i<NO_FITNESS_CASES ; i++ ){
     float EVOLVED_VALUE = recEval(this->root,inputs[i]);
     \INSERT_GENOME_EVAL_BDY
     sum += error;
   }
  error = sum;
  \INSERT_GENOME_EVAL_FTR    
}


void IndividualImpl::boundChecking(){
	\INSERT_BOUND_CHECKING
}

void IndividualImpl::serialize_impl(boost::archive::text_iarchive& ar, const unsigned version) {
	serialize_(ar, version);
}

void IndividualImpl::serialize_impl(boost::archive::text_oarchive& ar, const unsigned version) {
	serialize_(ar, version);
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
	omp_set_num_threads(this->nbCPUThreads);
	#endif

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
        this->remoteIslandModel = setVariable("remoteIslandModel", \REMOTE_ISLAND_MODEL);
	this->ipFile = setVariable("ipFile", "\IP_FILE");
        this->migrationProbability = setVariable("migrationProbability", (float)\MIGRATION_PROBABILITY);
        this->serverPort = setVariable("serverPort", \SERVER_PORT);


}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
//	EZ_NB_GEN = (unsigned*)setVariable("nbGen", \NB_GEN);
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
	  boost::archive::text_iarchive ia{AESAE_File};
  	  for( unsigned i=0 ; i< this->params->parentPopulationSize ; i++){
		  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
	  	  ia >> *((IndividualImpl*)this->population->parents[i]);
	  }
	  
	}
	else{
  	  #ifdef USE_OPENMP
  	  #pragma omp parallel for
  	  #endif
  	  for(int i = 0; i < static_cast<int>(this->params->parentPopulationSize); ++i) {
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

#include <stdlib.h>
#include <iostream>
#include <CIndividual.h>
#include <Parameters.h>
#include <string>
#include <list>
#include <map>
#include "CGPNode.h"

\INSERT_USER_HEADER

using namespace std;

class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;

extern int EZ_POP_SIZE;
extern int OFFSPRING_SIZE;

/* Insert declarations about opcodes*/
\INSERT_GP_OPCODE_DECL

\INSERT_USER_CLASSES_DEFINITIONS

class IndividualImpl : public CIndividual {

public: // in EASEA the genome is public (for user functions,...)
	// Class members
  	\INSERT_GENOME


public:
	IndividualImpl();
	IndividualImpl(const IndividualImpl& indiv);
	virtual ~IndividualImpl();
	float evaluate() override;
	CIndividual* crossover(CIndividual** p2) override;
	void printOn(std::ostream& O) const override;
	CIndividual* clone() override;

	unsigned mutate(float pMutationPerGene) override;

	void boundChecking() override;

	void serialize_impl(boost::archive::text_iarchive& ar, const unsigned version) override;
	void serialize_impl(boost::archive::text_oarchive& ar, const unsigned version) override;

	template <typename Archive>
	void serialize_(Archive& ar, [[maybe_unused]] const unsigned version) {
	    \GENOME_SERIAL
	    ar & this->fitness;
	}
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

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
	add_definitions(-DBOOST_ALL_NO_LIB)
	set(Boost_USE_STATIC_LIBS ON)
	set(Boost_USE_MULTITHREADED ON)
	set(Boost_USE_STATIC_RUNTIME OFF)
endif()
find_package(Boost REQUIRED COMPONENTS program_options serialization)

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
