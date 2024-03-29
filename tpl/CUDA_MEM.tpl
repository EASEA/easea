\TEMPLATE_START
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libEasea.lib")
#pragma comment(lib, "Winmm.lib")
#endif
/**
 This is program entry for CUDA_MEM template for EASEA

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

using namespace std;

/** Global variables for the whole algorithm */
CIndividual** pPopulation = NULL;
CIndividual* bBest = NULL;
float* pEZ_MUT_PROB = NULL;
float* pEZ_XOVER_PROB = NULL;
unsigned *EZ_NB_GEN;
unsigned *EZ_current_generation;

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
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#define WIN32
#endif

#include <fstream>
#ifndef WIN32
#include <sys/time.h>
#else
#include <time.h>
#endif
#include <string>
#include <sstream>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"
#include "CCuda.h"
#include <vector_types.h>

#include "config.h"
#ifdef USE_OPENMP
	#include <omp.h>
#endif

using namespace std;

#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

extern CEvolutionaryAlgorithm *EA:

#define CUDA_TPL

struct gpuArg* gpuArgs;

void* d_offspringPopulationcuda;
void* d_offspringPopulationTmpcuda;
float* d_fitnessescuda;
dim3 dimBlockcuda, dimGridcuda;

void* allocatedDeviceBuffer;
void* allocatedDeviceTmpBuffer;
float* deviceFitness;
dim3 dimBlock, dimGrid;

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

\INSERT_USER_CLASSES

\INSERT_USER_FUNCTIONS

void cudaPreliminaryProcess(unsigned populationSize, dim3* dimBlock, dim3* dimGrid, void** allocatedDeviceBuffer,void** allocatedDeviceTmpBuffer, float** deviceFitness){

        unsigned nbThreadPB, nbThreadLB, nbBlock;
        cudaError_t lastError;

        lastError = cudaMalloc(allocatedDeviceBuffer,populationSize*(sizeof(IndividualImpl)));
        lastError = cudaMalloc(allocatedDeviceTmpBuffer,populationSize*(sizeof(IndividualImpl)));
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
        //std::cout << "Number of grid : " << dimGrid->x << std::endl;
        //std::cout << "Number of block : " << dimBlock->x << std::endl;
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
	cudaFree(d_offspringPopulationcuda);
	cudaFree(d_offspringPopulationTmpcuda);
	cudaFree(d_fitnessescuda);
}

void AESAEBeginningGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	if(*EZ_current_generation==1){
		cudaPreliminaryProcess(((PopulationImpl*)evolutionaryAlgorithm->population)->offspringPopulationSize,&dimBlockcuda, &dimGridcuda, &d_offspringPopulationcuda,&d_offspringPopulationTmpcuda, &d_fitnessescuda);
	}
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
    this->isImmigrant=false;
}

void IndividualImpl::optimise(int currentIteration){
    \INSERT_OPTIMISER
}


IndividualImpl::IndividualImpl(const IndividualImpl& genome){

  // ********************
  // Problem specific part
  \COPY_CTOR


  // ********************
  // Generic part
  this->valid = genome.valid;
  this->fitness = genome.fitness;
  this->isImmgrant = false;
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

__device__ __host__ inline IndividualImpl* INDIVIDUAL_ACCESS(void* buffer,unsigned id){
  return (IndividualImpl*)buffer+id;
}

__device__  void cudaCopyIndividual(IndividualImpl* src, IndividualImpl* dest){
  \COPY_CUDA_CTOR
  //Generic Part
  dest->fitnessImpl = src->fitnessImpl;
}

__device__ float cudaEvaluate(void* devBuffer, unsigned id, struct gpuOptions initOpts){
  \INSERT_CUDA_EVALUATOR
}
  
__device__ void cudaOptimiseIndividual(void* devBuffer, unsigned id, struct gpuOptions initOpts, int currentIteration){
  \INSERT_CUDA_OPTIMISOR
}

__device__ void cudaOptimise(void* devBuffer, void* tmpBuffer, unsigned id, struct gpuOptions initOpts,  int OptimiseIterations){
	for(int currentIteration=0; currentIteration<OptimiseIterations; currentIteration++){
		cudaOptimiseIndividual(tmpBuffer, id, initOpts, currentIteration);
		INDIVIDUAL_ACCESS(tmpBuffer,id)->fitnessImpl = cudaEvaluate(tmpBuffer, id, initOpts); 

		if( (\MINIMAXI && INDIVIDUAL_ACCESS(tmpBuffer,id)->fitnessImpl < INDIVIDUAL_ACCESS(devBuffer,id)->fitnessImpl) || (!\MINIMAXI && INDIVIDUAL_ACCESS(tmpBuffer,id)->fitnessImpl > INDIVIDUAL_ACCESS(devBuffer,id)->fitnessImpl))
			cudaCopyIndividual(INDIVIDUAL_ACCESS(tmpBuffer, id), INDIVIDUAL_ACCESS(devBuffer, id));
		else 
			cudaCopyIndividual(INDIVIDUAL_ACCESS(devBuffer, id), INDIVIDUAL_ACCESS(tmpBuffer, id));
	}
}

__global__ void cudaEvaluatePopulation(void* d_population, unsigned popSize, float* d_fitnesses, struct gpuOptions initOpts){

        unsigned id = (blockDim.x*blockIdx.x)+threadIdx.x;  // id of the individual computed by this thread

  	// escaping for the last block
        if(blockIdx.x == (gridDim.x-1)) if( id >= popSize ) return;
  
       //void* indiv = ((char*)d_population)+id*(\GENOME_SIZE+sizeof(IndividualImpl*)); // compute the offset of the current individual
  
        d_fitnesses[id] = cudaEvaluate(d_population,id,initOpts);
  
}

__global__ void cudaOptimisePopulation(void* d_population, void* d_populationTmp, unsigned popSize, struct gpuOptions initOpts, int OptimiseIterations){

        unsigned id = (blockDim.x*blockIdx.x)+threadIdx.x;  // id of the individual computed by this thread

  	// escaping for the last block
        if(blockIdx.x == (gridDim.x-1)) if( id >= popSize ) return;
  
       //void* indiv = ((char*)d_population)+id*(\GENOME_SIZE+sizeof(IndividualImpl*)); // compute the offset of the current individual
  
        cudaOptimise(d_population, d_populationTmp,id,initOpts, OptimiseIterations);
  
}

void PopulationImpl::evaluateParentPopulation(){
        float* fitnesses = new float[this->actualParentPopulationSize];
        /*void* allocatedDeviceBuffer;
	void* allocatedDeviceTmpBuffer;
        float* deviceFitness;*/
        cudaError_t lastError;
//        dim3 dimBlock, dimGrid;
        unsigned actualPopulationSize = this->actualParentPopulationSize;

	// ICI il faut allouer la tailler max (entre parentPopualtionSize et offspringpopulationsize)
	cudaPreliminaryProcess(actualPopulationSize,&dimBlock,&dimGrid,&allocatedDeviceBuffer,&allocatedDeviceTmpBuffer, &deviceFitness);

        //compute the repartition over MP and SP
        //lastError = cudaMemcpy(allocatedDeviceBuffer,this->cuda->cudaParentBuffer,(\GENOME_SIZE+sizeof(Individual*))*actualPopulationSize,cudaMemcpyHostToDevice);
        lastError = cudaMemcpy(allocatedDeviceBuffer,this->cuda->cudaParentBuffer,(sizeof(IndividualImpl)*actualPopulationSize),cudaMemcpyHostToDevice);
        //DEBUG_PRT("Parent population buffer copy : %s",cudaGetErrorString(lastError));
        cudaEvaluatePopulation<<< dimGrid, dimBlock>>>(allocatedDeviceBuffer,actualPopulationSize,deviceFitness,this->cuda->initOpts);
        lastError = cudaThreadSynchronize();
        //DEBUG_PRT("Kernel execution : %s",cudaGetErrorString(lastError));

       lastError = cudaMemcpy(fitnesses,deviceFitness,actualPopulationSize*sizeof(float),cudaMemcpyDeviceToHost);
       //DEBUG_PRT("Parent's fitnesses gathering : %s",cudaGetErrorString(lastError));

       /*cudaFree(deviceFitness);
       cudaFree(allocatedDeviceBuffer);
       cudaFree(allocatedDeviceTmpBuffer);*/


#ifdef COMPARE_HOST_DEVICE
       this->CPopulation::evaluateParentPopulation();
#endif

       for( unsigned i=0 ; i<actualPopulationSize ; i++ ){
#ifdef COMPARE_HOST_DEVICE
               float error = (this->parents[i]->getFitness()-fitnesses[i])/this->parents[i]->getFitness();
               printf("Difference for individual %lu is : %f %f|%f\n",i,error,this->parents[i]->getFitness(), fitnesses[i]);
               if( error > 0.2 )
                     exit(-1);
#else
                //DEBUG_PRT("%lu : %f\n",i,fitnesses[i]);
                this->parents[i]->fitness = ((IndividualImpl*)this->parents[i])->fitnessImpl = fitnesses[i];
                this->parents[i]->valid = true;
#endif
        }
}

void PopulationImpl::optimiseParentPopulation(){
  cudaError_t lastError;
/*  void* allocatedDeviceBuffer;
  void* allocatedDeviceTmpBuffer;
  float* deviceFitness;*/
  unsigned actualPopulationSize = this->actualParentPopulationSize;
//  dim3 dimBlock, dimGrid;

// ICI il faut allouer la tailler max (entre parentPopualtionSize et offspringpopulationsize)
  //cudaPreliminaryProcess(actualPopulationSize,&dimBlock,&dimGrid,&allocatedDeviceBuffer,&allocatedDeviceTmpBuffer, &deviceFitness);
  

//Copy population into buffer
  for( unsigned i=0 ; i<this->actualParentPopulationSize ; i++){
	((IndividualImpl*)this->parents[i])->copyToCudaBuffer(((PopulationImpl*)this)->cuda->cudaParentBuffer,i);
  }

  //Copy population to GPU
  lastError = cudaMemcpy(allocatedDeviceBuffer,this->cuda->cudaParentBuffer,(sizeof(IndividualImpl)*actualPopulationSize),cudaMemcpyHostToDevice);
  lastError = cudaMemcpy(allocatedDeviceTmpBuffer,this->cuda->cudaParentBuffer,(sizeof(IndividualImpl)*actualPopulationSize),cudaMemcpyHostToDevice);

  //Optimise Population
  cudaOptimisePopulation<<< dimGrid, dimBlock>>>(allocatedDeviceBuffer, allocatedDeviceTmpBuffer, actualPopulationSize,this->cuda->initOpts,params->optimiseIterations);
  lastError = cudaThreadSynchronize();
  if( lastError )fprintf(stderr,"Kernel execution : %s\n",cudaGetErrorString(lastError));

  //Copy Fitnesses and population back to CPU
  if(!params->baldwinism)
    lastError = cudaMemcpy(this->cuda->cudaParentBuffer,allocatedDeviceBuffer,actualPopulationSize*sizeof(IndividualImpl),cudaMemcpyDeviceToHost);

  cudaFree(allocatedDeviceBuffer);
  cudaFree(allocatedDeviceTmpBuffer);
  cudaFree(deviceFitness);

  //Copies the new individuals back into the population
  for( unsigned i=0; i<actualPopulationSize ; i++){
    if(!params->baldwinism)
      ((IndividualImpl*)this->parents[i])->copyFromCudaBuffer(this->cuda->cudaParentBuffer,i);
    this->parents[i]->fitness = ((IndividualImpl*)this->parents[i])->fitnessImpl;
    this->parents[i]->valid = true;
  }
}

void PopulationImpl::evaluateOffspringPopulation(){
  cudaError_t lastError;
  unsigned actualPopulationSize = this->actualOffspringPopulationSize;
  float* fitnesses = new float[actualPopulationSize];

  for( unsigned i=0 ; i<this->actualOffspringPopulationSize ; i++ )
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

  for( unsigned i=0 ; i<actualPopulationSize ; i++ ){
#ifdef COMPARE_HOST_DEVICE
    float error = (this->offsprings[i]->getFitness()-fitnesses[i])/this->offsprings[i]->getFitness();
    printf("Difference for individual %lu is : %f %f|%f\n",i,error, this->offsprings[i]->getFitness(),fitnesses[i]);
    if( error > 0.2 )
      exit(-1);

#else
    //DEBUG_PRT("%lu : %f\n",i,fitnesses[i]);
    this->offsprings[i]->fitness = ((IndividualImpl*)this->offsprings[i])->fitnessImpl = fitnesses[i];
    this->offsprings[i]->valid = true;
#endif
  }
  
}

void PopulationImpl::optimiseOffspringPopulation(){
  cudaError_t lastError;
  unsigned actualPopulationSize = this->actualOffspringPopulationSize;

  //Copy population to buffer
  for( unsigned i=0; i<actualPopulationSize; i++)
    ((IndividualImpl*)this->offsprings[i])->copyToCudaBuffer(this->cuda->cudaOffspringBuffer,i);

  //Copy population to GPU
  lastError = cudaMemcpy(d_offspringPopulationcuda,this->cuda->cudaOffspringBuffer,sizeof(IndividualImpl)*actualPopulationSize, cudaMemcpyHostToDevice);
  lastError = cudaMemcpy(d_offspringPopulationTmpcuda,this->cuda->cudaOffspringBuffer,sizeof(IndividualImpl)*actualPopulationSize, cudaMemcpyHostToDevice);

  //Otpimise children
  cudaOptimisePopulation<<<dimGridcuda, dimBlockcuda>>>(d_offspringPopulationcuda,d_offspringPopulationTmpcuda,actualPopulationSize, this->cuda->initOpts, params->optimiseIterations);
  lastError = cudaThreadSynchronize();

  //Copy fitnesses and population back to the CPU
  if(!params->baldwinism)
    lastError = cudaMemcpy(this->cuda->cudaOffspringBuffer,d_offspringPopulationcuda,actualPopulationSize*sizeof(IndividualImpl*), cudaMemcpyDeviceToHost);

  //Replace the newly optimised population
  for( unsigned i=0; i<actualPopulationSize ; i++){
    if(!params->baldwinism)
      ((IndividualImpl*)this->offsprings[i])->copyFromCudaBuffer(this->cuda->cudaOffspringBuffer,i);
    this->offsprings[i]->fitness = ((IndividualImpl*)this->offsprings[i])->fitnessImpl;
    this->offsprings[i]->valid = true;
  }
}

ParametersImpl::ParametersImpl(std::string const& file, int argc, char* argv[]) : Parameters(file, argc, argv) {
        this->minimizing = \MINIMAXI;
        this->nbGen = setVariable("nbGen", (int)\NB_GEN);
	#ifdef USE_OPENMP
	omp_set_num_threads(nbCPUThreads);
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
        if((this->parentPopulationSize - this->parentReductionSize)>this->parentPopulationSize-this->elitSize){
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
        if(offspringReductionSize<offspringPopulationSize) offspringReduction = true;
        else offspringReduction = false;

        if(parentReductionSize<parentPopulationSize) parentReduction = true;
        else parentReduction = false;

        generationalCriterion = new CGenerationalCriterion(setVariable("nbGen", (int)\NB_GEN));
        controlCStopingCriterion = new CControlCStopingCriterion();
        timeCriterion = new CTimeCriterion(setVariable("timeLimit", \TIME_LIMIT));
        this->optimiseIterations = setVariable("optimiseIterations", (int)\NB_OPT_IT);
        this->baldwinism = setVariable("baldwinism", (int)\BALDWINISM);

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

	CEvolutionaryAlgorithm* ea = new EvolutionaryAlgorithmImpl(this);
	generationalCriterion->setCounterEa(ea->getCurrentGenerationPtr());
	 ea->addStoppingCriterion(generationalCriterion);
	ea->addStoppingCriterion(controlCStopingCriterion);
	ea->addStoppingCriterion(timeCriterion);

	  EZ_NB_GEN=((CGenerationalCriterion*)ea->stoppingCriteria[0])->getGenerationalLimit();
	  EZ_current_generation=&(ea->currentGeneration);

	 return ea;
}

inline void IndividualImpl::copyToCudaBuffer(void* buffer, unsigned id){
  
 memcpy(((IndividualImpl*)buffer)+id,this,sizeof(IndividualImpl)); 
  
}

inline void IndividualImpl::copyFromCudaBuffer(void* buffer, unsigned id){

        memcpy(this, ((IndividualImpl*)buffer)+id,sizeof(IndividualImpl));

}

void EvolutionaryAlgorithmImpl::initializeParentPopulation(){

  //DEBUG_PRT("Creation of %lu/%lu parents (other could have been loaded from input file)",this->params->parentPopulationSize-this->params->actualParentPopulationSize,this->params->parentPopulationSize);
    int index,Size = this->params->parentPopulationSize;

    if(this->params->startFromFile){
          ifstream AESAE_File(this->params->inputFilename);
          string AESAE_Line;
          for( index=(Size-1); index>=0; index--) {
             getline(AESAE_File, AESAE_Line);
            this->population->addIndividualParentPopulation(new IndividualImpl(),index);
            ((IndividualImpl*)this->population->parents[index])->deserialize(AESAE_Line);
            ((IndividualImpl*)this->population->parents[index])->copyToCudaBuffer(((PopulationImpl*)this->population)->cuda->cudaBuffer,index);
         }

        }
        else{
                for( index=(Size-1); index>=0; index--) {
                         this->population->addIndividualParentPopulation(new IndividualImpl(),index);
                        ((IndividualImpl*)this->population->parents[index])->copyToCudaBuffer(((PopulationImpl*)this->population)->cuda->cudaBuffer,index);
                }
    }

    this->population->actualOffspringPopulationSize = 0;
    this->population->actualParentPopulationSize = Size;

}


EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){
	this->population = (CPopulation*)new PopulationImpl(this->params->parentPopulationSize,this->params->offspringPopulationSize, this->params->pCrossover,this->params->pMutation,this->params->pMutationPerGene,this->params->randomGenerator,this->params);
	((PopulationImpl*)this->population)->cuda = new CCuda(params->parentPopulationSize, params->offspringPopulationSize, sizeof(IndividualImpl));
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

PopulationImpl::PopulationImpl(unsigned parentPopulationSize, unsigned offspringPopulationSize, float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params) : CPopulation(parentPopulationSize, offspringPopulationSize, pCrossover, pMutation, pMutationPerGene, rg, params){
	
}

PopulationImpl::~PopulationImpl(){
}


\START_CUDA_GENOME_H_TPL

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

#include <stdlib.h>
#include <iostream>
#include <string>
#include <CIndividual.h>
#include <Parameters.h>
#include <CCuda.h>

\INSERT_USER_HEADER

using namespace std;

class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;
class CCuda;


\INSERT_USER_CLASSES_DEFINITIONS

class IndividualImpl : public CIndividual {

public: // in EASEA the genome is public (for user functions,...)
	// Class members
  	\INSERT_GENOME
	
	float fitnessImpl;


public:
	IndividualImpl();
	IndividualImpl(const IndividualImpl& indiv);
	virtual ~IndividualImpl() {
		\GENOME_DTOR
	}
	float evaluate() override;
	void optimise(int currentIteration);
	CIndividual* crossover(CIndividual** p2) override;
	void printOn(std::ostream& O) const override;
	CIndividual* clone() override;

	unsigned mutate(float pMutationPerGene) override;

	void boundChecking() override;

        string serialize() override;
        void deserialize(string AESAE_Line) override;
	void copyToCudaBuffer(void* buffer, unsigned id);
	void copyFromCudaBuffer(void* buffer, unsigned id);};


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
	PopulationImpl(unsigned parentPopulationSize, unsigned offspringPopulationSize, float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params);
        virtual ~PopulationImpl();
        void evaluateParentPopulation();
	void evaluateOffspringPopulation();
	void optimiseParentPopulation();
	void optimiseOffspringPopulation();
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
find_package(Boost REQUIRED program_options)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
cmake_policy(SET CMP0104 NEW)

target_include_directories(EASEA PUBLIC ${Boost_INCLUDE_DIRS} ${libeasea_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
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
#  Parameter file generated by CUDA.tpl AESAE v1.0
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

######    Local Optimisation    ######
--optimiseIterations=\NB_OPT_IT
--baldwinism=\BALDWINISM # True (1) or False (0) baldwinism : keep optimised genome

#####   Stats Ouput     #####
--printStats=\PRINT_STATS #print Stats to screen
--plotStats=\PLOT_STATS #plot Stats
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
