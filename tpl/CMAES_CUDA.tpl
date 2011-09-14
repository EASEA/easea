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


	parseArguments("EASEA.prm",argc,argv);

	ParametersImpl p;
	p.setDefaultParameters(argc,argv);
	CEvolutionaryAlgorithm* ea = p.newEvolutionaryAlgorithm();

	EASEAInit(argc,argv);

	CPopulation* pop = ea->getPopulation();

	ea->runEvolutionaryLoop();

	EASEAFinal(pop);

	delete pop;

#ifdef WIN32
	system("pause");
#endif

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

\INSERT_INITIALISATION_FUNCTION
\INSERT_FINALIZATION_FUNCTION

void evale_pop_chunk(CIndividual** population, int popSize){
  \INSTEAD_EVAL_FUNCTION
}

void EASEAInit(int argc, char** argv){
	\INSERT_INIT_FCT_CALL
	cma = new CCmaesCuda(setVariable("nbOffspring",\OFF_SIZE), setVariable("popSize",\POP_SIZE), \PROBLEM_DIM);
}

void EASEAFinal(CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;
	delete(cma);
}

void AESAEBeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	if(*EZ_current_generation==1){
		cudaPreliminaryProcess(((PopulationImpl*)evolutionaryAlgorithm->population)->offspringPopulationSize,&dimBlockcuda, &dimGridcuda, &d_offspringPopulationcuda,&d_fitnessescuda);
	}
	cma->cmaes_UpdateEigensystem(0);
	cma->TestMinStdDevs();
	\INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
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

void AESAEGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
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


size_t IndividualImpl::mutate( float pMutationPerGene ){
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





void ParametersImpl::setDefaultParameters(int argc, char** argv){

        this->minimizing =1;
        this->nbGen = setVariable("nbGen",(int)\NB_GEN);

        selectionOperator = getSelectionOperator(setVariable("selectionOperator","Tournament"), this->minimizing, globalRandomGenerator);
        replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator","\RED_FINAL_OPERATOR"),this->minimizing, globalRandomGenerator);
        selectionPressure = 1;
        replacementPressure = setVariable("reduceFinalPressure",(float)\RED_FINAL_PRM);
        pCrossover = 1;
        pMutation = 1;
        pMutationPerGene = 1;

        parentPopulationSize = setVariable("popSize",(int)\POP_SIZE);
        offspringPopulationSize = setVariable("nbOffspring",(int)\OFF_SIZE);

        this->elitSize = 0;

	offspringReduction = parentReduction = false;

        generationalCriterion = new CGenerationalCriterion(setVariable("nbGen",(int)\NB_GEN));
        controlCStopingCriterion = new CControlCStopingCriterion();
        timeCriterion = new CTimeCriterion(setVariable("timeLimit",\TIME_LIMIT));

	this->optimise = 0;

        seed = setVariable("seed",(int)time(0));
        globalRandomGenerator = new CRandomGenerator(seed);
        this->randomGenerator = globalRandomGenerator;

        this->printStats = setVariable("printStats",\PRINT_STATS);
        this->generateCSVFile = setVariable("generateCSVFile",\GENERATE_CSV_FILE);
        this->generatePlotScript = setVariable("generatePlotScript",\GENERATE_GNUPLOT_SCRIPT);
        this->generateRScript = setVariable("generateRScript",\GENERATE_R_SCRIPT);
        this->plotStats = setVariable("plotStats",\PLOT_STATS);
        this->printInitialPopulation = setVariable("printInitialPopulation",0);
        this->printFinalPopulation = setVariable("printFinalPopulation",0);

        this->outputFilename = (char*)"EASEA";
        this->plotOutputFilename = (char*)"EASEA.png";
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	EZ_NB_GEN = (size_t*)setVariable("nbGen",\NB_GEN);
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
  for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
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
	;
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

PopulationImpl::PopulationImpl(size_t parentPopulationSize, size_t offspringPopulationSize, float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params) : CPopulation(parentPopulationSize, offspringPopulationSize, pCrossover, pMutation, pMutationPerGene, rg, params){
	;
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

	size_t mutate(float pMutationPerGene);

	void boundChecking();

	void copyToCudaBuffer(void* buffer, size_t id);

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

\START_CUDA_MAKEFILE_TPL
NVCC= nvcc
CPPC= g++
LIBAESAE=\EZ_PATHlibeasea/
CXXFLAGS+=-g -Wall -O2 -I$(LIBAESAE)include
LDFLAGS=-lboost_program_options $(LIBAESAE)libeasea.a

#USER MAKEFILE OPTIONS :
\INSERT_MAKEFILE_OPTION#END OF USER MAKEFILE OPTIONS

CPPFLAGS+= -I$(LIBAESAE)include
NVCCFLAGS+=


EASEA_SRC= EASEAIndividual.cpp
EASEA_MAIN_HDR= EASEA.cpp
EASEA_UC_HDR= EASEAIndividual.hpp

EASEA_HDR= $(EASEA_SRC:.cpp=.hpp) 

SRC= $(EASEA_SRC) $(EASEA_MAIN_HDR)
CUDA_SRC = EASEAIndividual.cu
HDR= $(EASEA_HDR) $(EASEA_UC_HDR)
OBJ= $(EASEA_SRC:.cpp=.o) $(EASEA_MAIN_HDR:.cpp=.o)

BIN= EASEA
  
all:$(BIN)

$(BIN):$(OBJ)
	$(NVCC) $^ -o $@ $(LDFLAGS) 

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -c -DTIMING $(CPPFLAGS) -g -Xcompiler -Wall

easeaclean: clean
	rm -f Makefile EASEA.prm $(SRC) $(HDR) EASEA.mak $(CUDA_SRC) *.linkinfo EASEA.png EASEA.dat EASEA.vcproj EASEA.plot EASEA.r EASEA.csv
clean:
	rm -f $(OBJ) $(BIN) 	
	
\START_VISUAL_TPL<?xml version="1.0" encoding="Windows-1252"?>
<VisualStudioProject
	ProjectType="Visual C++"
	Version="9,00"
	Name="EASEA"
	ProjectGUID="{E73D5A89-F262-4F0E-A876-3CF86175BC30}"
	RootNamespace="EASEA"
	Keyword="WIN32Proj"
	TargetFrameworkVersion="196613"
	>
	<Platforms>
		<Platform
			Name="WIN32"
		/>
	</Platforms>
	<ToolFiles>
		<ToolFile
			RelativePath="\CUDA_RULE_DIRcommon\Cuda.rules"
		/>
	</ToolFiles>
	<Configurations>
		<Configuration
			Name="Release|WIN32"
			OutputDirectory="$(SolutionDir)"
			IntermediateDirectory="$(ConfigurationName)"
			ConfigurationType="1"
			CharacterSet="1"
			WholeProgramOptimization="1"
			>
			<Tool
				Name="VCPreBuildEventTool"
			/>
			<Tool
				Name="VCCustomBuildTool"
			/>
			<Tool
				Name="CUDA Build Rule"
				Include="\EZ_PATHlibEasea"
				Keep="false"
				Runtime="0"
			/>
			<Tool
				Name="VCXMLDataGeneratorTool"
			/>
			<Tool
				Name="VCWebServiceProxyGeneratorTool"
			/>
			<Tool
				Name="VCMIDLTool"
			/>
			<Tool
				Name="VCCLCompilerTool"
				Optimization="2"
				EnableIntrinsicFunctions="true"
				AdditionalIncludeDirectories="&quot;\EZ_PATHlibEasea&quot;"
				PreprocessorDefinitions="WIN32;NDEBUG;_CONSOLE"
				RuntimeLibrary="0"
				EnableFunctionLevelLinking="true"
				UsePrecompiledHeader="0"
				WarningLevel="3"
				DEBUGInformationFormat="3"
			/>
			<Tool
				Name="VCManagedResourceCompilerTool"
			/>
			<Tool
				Name="VCResourceCompilerTool"
			/>
			<Tool
				Name="VCPreLinkEventTool"
			/>
			<Tool
				Name="VCLinkerTool"
				AdditionalDependencies="$(CUDA_LIB_PATH)\cudart.lib"
				LinkIncremental="1"
				AdditionalLibraryDirectories="&quot;\EZ_PATHlibEasea&quot;"
				GenerateDEBUGInformation="true"
				SubSystem="1"
				OptimizeReferences="2"
				EnableCOMDATFolding="2"
				TargetMachine="1"
			/>
			<Tool
				Name="VCALinkTool"
			/>
			<Tool
				Name="VCManifestTool"
			/>
			<Tool
				Name="VCXDCMakeTool"
			/>
			<Tool
				Name="VCBscMakeTool"
			/>
			<Tool
				Name="VCFxCopTool"
			/>
			<Tool
				Name="VCAppVerifierTool"
			/>
			<Tool
				Name="VCPostBuildEventTool"
			/>
		</Configuration>
	</Configurations>
	<References>
	</References>
	<Files>
		<Filter
			Name="Source Files"
			Filter="cpp;c;cc;cxx;def;odl;idl;hpj;bat;asm;asmx"
			UniqueIdentifier="{4FC737F1-C7A5-4376-A066-2A32D752A2FF}"
			>
			<File
				RelativePath=".\EASEA.cpp"
				>
			</File>
			<File
				RelativePath=".\EASEAIndividual.cu"
				>
			</File>
		</Filter>
		<Filter
			Name="Header Files"
			Filter="h;hpp;hxx;hm;inl;inc;xsd"
			UniqueIdentifier="{93995380-89BD-4b04-88EB-625FBE52EBFB}"
			>
			<File
				RelativePath=".\EASEAIndividual.hpp"
				>
			</File>
		</Filter>
		<Filter
			Name="Resource Files"
			Filter="rc;ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe;resx;tiff;tif;png;wav"
			UniqueIdentifier="{67DA6AB6-F800-4c08-8B7A-83BB121AAD01}"
			>
		</Filter>
	</Files>
	<Globals>
	</Globals>
</VisualStudioProject>

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
