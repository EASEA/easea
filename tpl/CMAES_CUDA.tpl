\TEMPLATE_START
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libEasea.lib")
#pragma comment(lib, "Winmm.lib")
#endif
/**
 This is program entry for CMAES_CUDA template for EASEA

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
#include "CCmaesCuda.h"

using namespace std;

/** Global variables for the whole algorithm */
CIndividual** pPopulation = NULL;
CIndividual* bBest = NULL;
float* pEZ_MUT_PROB = NULL;
float* pEZ_XOVER_PROB = NULL;
size_t *EZ_NB_GEN;
size_t *EZ_current_generation;
CEvolutionaryAlgorithm *EA;

CCmaesCuda *cma;

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
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libEasea.lib")
#define WIN32
#endif

using namespace std;
#include <string.h>
#include <fstream>
#ifndef WIN32
#include <sys/time.h>
#else
#include <time.h>
#endif
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"
#include "CCuda.h"
#include "CCmaesCuda.h"
#include <vector_types.h>


#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;
#define CUDA_TPL

void* d_offspringPopulationcuda;
float* d_fitnessescuda;
dim3 dimBlockcuda, dimGridcuda;

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

\INSERT_USER_CLASSES

\INSERT_USER_FUNCTIONS

void cudaPreliminaryProcess(size_t populationSize, dim3* dimBlock, dim3* dimGrid, void** allocatedDeviceBuffer,float** deviceFitness){

        size_t nbThreadPB, nbThreadLB, nbBlock;
        cudaError_t lastError;

        lastError = cudaMalloc(allocatedDeviceBuffer,populationSize*(sizeof(IndividualImpl)));
        //DEBUG_PRT("Population buffer allocation : %s",cudaGetErrorString(lastError));
        lastError = cudaMalloc(((void**)deviceFitness),populationSize*sizeof(float));
        //DEBUG_PRT("Fitness buffer allocation : %s",cudaGetErrorString(lastError));

        if( !repartition(populationSize, &nbBlock, &nbThreadPB, &nbThreadLB,30, 240))
                 exit( -1 );

        //DEBUG_PRT("repartition is \n\tnbBlock %lu \n\tnbThreadPB %lu \n\tnbThreadLD %lu",nbBlock,nbThreadPB,nbThreadLB);

        if( nbThreadLB!=0 )
                   dimGrid->x = (nbBlock+1);
        else
        dimGrid->x = (nbBlock);

        dimBlock->x = nbThreadPB;
        std::cout << "Number of grid : " << dimGrid->x << std::endl;
        std::cout << "Number of block : " << dimBlock->x << std::endl;
}

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
	if(*EZ_current_generation==1){
		cudaPreliminaryProcess(((PopulationImpl*)evolutionaryAlgorithm->population)->offspringPopulationSize,&dimBlockcuda, &dimGridcuda, &d_offspringPopulationcuda,&d_fitnessescuda);
	}
	cma->cmaes_UpdateEigensystem(0);
	cma->TestMinStdDevs();
	\INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_END_GENERATION_FUNCTION
        evolutionaryAlgorithm->population->sortParentPopulation();
        float **popparent;
        float *fitpar;
        popparent = (float**)malloc(evolutionaryAlgorithm->population->parentPopulationSize*sizeof(float *));
        fitpar = (float*)malloc(evolutionaryAlgorithm->population->parentPopulationSize*sizeof(float));
        for(int i=0; i<(signed)evolutionaryAlgorithm->population->parentPopulationSize; i++)
                popparent[i] = (float*)malloc(\PROBLEM_DIM*sizeof(float));
        for(int i=0; i<(signed)evolutionaryAlgorithm->population->parentPopulationSize; i++){
                for(int j=0; j<\PROBLEM_DIM; j++){
                        IndividualImpl *tmp = (IndividualImpl *)evolutionaryAlgorithm->population->parents[i];
                        popparent[i][j] = (float)tmp->\GENOME_NAME[j];
                        //cout << popparent[i][j] << "|";
                }
                fitpar[i] = (float)evolutionaryAlgorithm->population->parents[i]->fitness;
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
  for(int i=0; i<\PROBLEM_DIM; i++) {
	this->\GENOME_NAME[i] = (float)(0.5 + (cma->sigma * cma->rgD[i] * cma->alea.alea_Gauss()));
  }
  valid = false;
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


IndividualImpl::IndividualImpl(const IndividualImpl& genome){

  // ********************
  // Problem specific part
  \COPY_CTOR


  // ********************
  // Generic part
  this->valid = genome.valid;
  this->fitness = genome.fitness;
}


CIndividual* IndividualImpl::crossover(CIndividual** ps){
	// ********************
	// Generic part
	IndividualImpl** tmp = (IndividualImpl**)ps;
	IndividualImpl parent1(*this);
	IndividualImpl parent2(*tmp[0]);
	IndividualImpl child(*this);

	////DEBUG_PRT("Xover");
	/*   cout << "p1 : " << parent1 << endl; */
	/*   cout << "p2 : " << parent2 << endl; */

	// ********************
	// Problem specific part
	for(int i=0; i<\PROBLEM_DIM; ++i)
		cma->rgdTmp[i] = (float)(cma->rgD[i] * cma->alea.alea_Gauss());

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
  float sum;
  for (int i = 0; i < \PROBLEM_DIM; ++i) {
        sum = 0.;
        for (int j = 0; j < \PROBLEM_DIM; ++j)
                sum += cma->B[i][j] * cma->rgdTmp[j];
        this->\GENOME_NAME[i] = (float)(cma->rgxmean[i] + cma->sigma * sum);
  }

  return 0;
}


__device__ __host__ inline IndividualImpl* INDIVIDUAL_ACCESS(void* buffer,size_t id){
  return (IndividualImpl*)buffer+id;
}

__device__ float cudaEvaluate(void* devBuffer, size_t id, struct gpuOptions initOpts){
  \INSERT_CUDA_EVALUATOR
}
  

__global__ void cudaEvaluatePopulation(void* d_population, size_t popSize, float* d_fitnesses, struct gpuOptions initOpts){

        size_t id = (blockDim.x*blockIdx.x)+threadIdx.x;  // id of the individual computed by this thread

  	// escaping for the last block
        if(blockIdx.x == (gridDim.x-1)) if( id >= popSize ) return;
  
       //void* indiv = ((char*)d_population)+id*(\GENOME_SIZE+sizeof(IndividualImpl*)); // compute the offset of the current individual
  
        d_fitnesses[id] = cudaEvaluate(d_population,id,initOpts);
  
}


void PopulationImpl::evaluateParentPopulation(){
        float* fitnesses = new float[this->actualParentPopulationSize];
        void* allocatedDeviceBuffer;
        float* deviceFitness;
        cudaError_t lastError;
        dim3 dimBlock, dimGrid;
        size_t actualPopulationSize = this->actualParentPopulationSize;

	// ICI il faut allouer la tailler max (entre parentPopualtionSize et offspringpopulationsize)
        cudaPreliminaryProcess(actualPopulationSize,&dimBlock,&dimGrid,&allocatedDeviceBuffer,&deviceFitness);

        //compute the repartition over MP and SP
        //lastError = cudaMemcpy(allocatedDeviceBuffer,this->cuda->cudaParentBuffer,(\GENOME_SIZE+sizeof(Individual*))*actualPopulationSize,cudaMemcpyHostToDevice);
        lastError = cudaMemcpy(allocatedDeviceBuffer,this->cuda->cudaParentBuffer,(sizeof(IndividualImpl)*actualPopulationSize),cudaMemcpyHostToDevice);
        //DEBUG_PRT("Parent population buffer copy : %s",cudaGetErrorString(lastError));
        cudaEvaluatePopulation<<< dimGrid, dimBlock>>>(allocatedDeviceBuffer,actualPopulationSize,deviceFitness,this->cuda->initOpts);
        lastError = cudaThreadSynchronize();
        //DEBUG_PRT("Kernel execution : %s",cudaGetErrorString(lastError));

       lastError = cudaMemcpy(fitnesses,deviceFitness,actualPopulationSize*sizeof(float),cudaMemcpyDeviceToHost);
       //DEBUG_PRT("Parent's fitnesses gathering : %s",cudaGetErrorString(lastError));

       cudaFree(deviceFitness);
       cudaFree(allocatedDeviceBuffer);



#ifdef COMPARE_HOST_DEVICE
       this->CPopulation::evaluateParentPopulation();
#endif

       for( size_t i=0 ; i<actualPopulationSize ; i++ ){
#ifdef COMPARE_HOST_DEVICE
               float error = (this->parents[i]->getFitness()-fitnesses[i])/this->parents[i]->getFitness();
               printf("Difference for individual %lu is : %f %f|%f\n",i,error,this->parents[i]->getFitness(), fitnesses[i]);
               if( error > 0.2 )
                     exit(-1);
#else
                //DEBUG_PRT("%lu : %f\n",i,fitnesses[i]);
                this->parents[i]->fitness = fitnesses[i];
                this->parents[i]->valid = true;
#endif
        }
}

void PopulationImpl::evaluateOffspringPopulation(){
  cudaError_t lastError;
  size_t actualPopulationSize = this->actualOffspringPopulationSize;
  float* fitnesses = new float[actualPopulationSize];

  for( size_t i=0 ; i<this->actualOffspringPopulationSize ; i++ )
      ((IndividualImpl*)this->offsprings[i])->copyToCudaBuffer(this->cuda->cudaOffspringBuffer,i);
  
  lastError = cudaMemcpy(d_offspringPopulationcuda,this->cuda->cudaOffspringBuffer,sizeof(IndividualImpl)*actualPopulationSize, cudaMemcpyHostToDevice);
  //DEBUG_PRT("Parent population buffer copy : %s",cudaGetErrorString(lastError));

  cudaEvaluatePopulation<<< dimGridcuda, dimBlockcuda>>>(d_offspringPopulationcuda,actualPopulationSize,d_fitnessescuda,this->cuda->initOpts);
  lastError = cudaGetLastError();
  //DEBUG_PRT("Kernel execution : %s",cudaGetErrorString(lastError));

  lastError = cudaMemcpy(fitnesses,d_fitnessescuda,actualPopulationSize*sizeof(float),cudaMemcpyDeviceToHost);
  //DEBUG_PRT("Offspring's fitnesses gathering : %s",cudaGetErrorString(lastError));


#ifdef COMPARE_HOST_DEVICE
  this->CPopulation::evaluateOffspringPopulation();
#endif

  for( size_t i=0 ; i<actualPopulationSize ; i++ ){
#ifdef COMPARE_HOST_DEVICE
    float error = (this->offsprings[i]->getFitness()-fitnesses[i])/this->offsprings[i]->getFitness();
    printf("Difference for individual %lu is : %f %f|%f\n",i,error, this->offsprings[i]->getFitness(),fitnesses[i]);
    if( error > 0.2 )
      exit(-1);

#else
    //DEBUG_PRT("%lu : %f\n",i,fitnesses[i]);
    this->offsprings[i]->fitness = fitnesses[i];
    this->offsprings[i]->valid = true;
#endif
  }
  
}

ParametersImpl::ParametersImpl(std::string const& file, int argc, char* argv[]) : Parameters(file, argc, argv) {

        this->minimizing =1;
        this->nbGen = setVariable("nbGen", (int)\NB_GEN);

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

        globalRandomGenerator = new CRandomGenerator(seed);
        this->randomGenerator = globalRandomGenerator;

        this->printStats = setVariable("printStats", \PRINT_STATS);
        this->generateCSVFile = setVariable("generateCSVFile", \GENERATE_CSV_FILE);
        this->generatePlotScript = setVariable("generatePlotScript", \GENERATE_GNUPLOT_SCRIPT);
        this->generateRScript = setVariable("generateRScript", \GENERATE_R_SCRIPT);
        this->plotStats = setVariable("plotStats", \PLOT_STATS);

        this->outputFilename = setVariable("outputFile", "EASEA");
        this->inputFilename = setVariable("inputFile", "EASEA.pop");
        this->plotOutputFilename = (char*)"EASEA.png";
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	EZ_NB_GEN = (size_t*)setVariable("nbGen", \NB_GEN);
	EZ_current_generation=0;

	CEvolutionaryAlgorithm* ea = new EvolutionaryAlgorithmImpl(this);
	generationalCriterion->setCounterEa(ea->getCurrentGenerationPtr());
	 ea->addStoppingCriterion(generationalCriterion);
	 ea->addStoppingCriterion(controlCStopingCriterion);
	ea->addStoppingCriterion(timeCriterion);

	  EZ_NB_GEN=((CGenerationalCriterion*)ea->stoppingCriteria[0])->getGenerationalLimit();
	  EZ_current_generation=&(ea->currentGeneration);

	 return ea;
}

inline void IndividualImpl::copyToCudaBuffer(void* buffer, size_t id){
  
 memcpy(((IndividualImpl*)buffer)+id,this,sizeof(IndividualImpl)); 
  
}

void EvolutionaryAlgorithmImpl::initializeParentPopulation(){

  //DEBUG_PRT("Creation of %lu/%lu parents (other could have been loaded from input file)",this->params->parentPopulationSize-this->params->actualParentPopulationSize,this->params->parentPopulationSize);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
for(int i=0 ; i< this->params->parentPopulationSize ; i++){
	this->population->addIndividualParentPopulation(new IndividualImpl(),i);
  }

  this->population->actualParentPopulationSize = this->params->parentPopulationSize;
  this->population->actualOffspringPopulationSize = 0;
  
  // Copy parent population in the cuda buffer.
  for( size_t i=0 ; i<this->population->actualParentPopulationSize ; i++ ){
    ((IndividualImpl*)this->population->parents[i])->copyToCudaBuffer(((PopulationImpl*)this->population)->cuda->cudaParentBuffer,i); 
  }

}


EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){
	this->population = (CPopulation*)new PopulationImpl(this->params->parentPopulationSize,this->params->offspringPopulationSize, this->params->pCrossover,this->params->pMutation,this->params->pMutationPerGene,this->params->randomGenerator,this->params);
	((PopulationImpl*)this->population)->cuda = new CCuda(params->parentPopulationSize, params->offspringPopulationSize, sizeof(IndividualImpl));
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

PopulationImpl::PopulationImpl(size_t parentPopulationSize, size_t offspringPopulationSize, float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params) : CPopulation(parentPopulationSize, offspringPopulationSize, pCrossover, pMutation, pMutationPerGene, rg, params){
}

PopulationImpl::~PopulationImpl(){
}


\START_CUDA_GENOME_H_TPL

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

//#include "CRandomGenerator.h"
#include <stdlib.h>
#include <iostream>
#include <CIndividual.h>
#include <Parameters.h>
#include "CCuda.h"
#include "CCmaesCuda.h"
class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;
class CCuda;
class CCmaesCuda;

extern CCmaesCuda *cma;

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
	static size_t getCrossoverArrity(){ return 2; }
	float getFitness(){ return this->fitness; }
	CIndividual* crossover(CIndividual** p2);
	void printOn(std::ostream& O) const;
	CIndividual* clone();

	void mutate(float pMutationPerGene);

	void boundChecking();

	void copyToCudaBuffer(void* buffer, size_t id);

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

class PopulationImpl: public CPopulation {
public:
	CCuda *cuda;
public:
	PopulationImpl(size_t parentPopulationSize, size_t offspringPopulationSize, float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params);
        virtual ~PopulationImpl();
        void evaluateParentPopulation();
	void evaluateOffspringPopulation();
};

#endif /* PROBLEM_DEP_H */

\START_CMAKELISTS
cmake_minimum_required(VERSION 3.9) # 3.9: OpenMP improved support
set(EZ_ROOT $ENV{EZ_PATH})

project(EASEA LANGUAGES CUDA CXX C)
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release")
endif()

file(GLOB EASEA_src ${CMAKE_SOURCE_DIR}/*.cpp ${CMAKE_SOURCE_DIR}/*.c ${CMAKE_SOURCE_DIR}/*.cu)
list(FILTER EASEA_src EXCLUDE REGEX .*EASEAIndividual.cpp)
add_executable(EASEA ${EASEA_src})
set_target_properties(EASEA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(EASEA PUBLIC cxx_std_17)
target_compile_options(EASEA PRIVATE
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /W3>
	$<$<AND:$<NOT:$<COMPILE_LANGUAGE:CUDA>>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -march=native -mtune=native -Wall -Wextra>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/O1 /W4 /DEBUG:FULL>
	$<$<AND:$<NOT:$<COMPILE_LANGUAGE:CUDA>>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -Wall -Wextra>
	)

find_library(libeasea_LIB
	NAMES libeasea easea
	HINTS ${EZ_ROOT} ${CMAKE_INSTALL_PREFIX}/easena ${CMAKE_INSTALL_PREFIX}/EASENA
	PATH_SUFFIXES lib libeasea easea easena)
find_path(libeasea_INCLUDE
	NAMES CLogger.h
	HINTS ${EZ_ROOT}/libeasea ${CMAKE_INSTALL_PREFIX}/*/libeasea
	PATH_SUFFIXES include easena libeasea)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

message(STATUS ${libeasea_INCLUDE} ${CLOGGER} ${CUDAToolkit_INCLUDE_DIRS})

target_include_directories(EASEA PUBLIC ${Boost_INCLUDE_DIRS} ${libeasea_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
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
#  Parameter file generated by CMAES_CUDA.tpl AESAE v1.0
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

\TEMPLATE_END
