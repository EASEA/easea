\TEMPLATE_START
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libEasea.lib")
#pragma comment(lib, "Winmm.lib")
#endif
/**
 This is program entry for STD template for EASEA
*/

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
CIndividual* bBest = NULL;
float* pEZ_MUT_PROB = NULL;
float* pEZ_XOVER_PROB = NULL;
unsigned *EZ_NB_GEN;
unsigned *EZ_current_generation;
CEvolutionaryAlgorithm* EA;
std::vector<char *> vArgv;
int EZ_POP_SIZE;
int OFFSPRING_SIZE;


int main(int argc, char** argv){
	if (argc > 1){
    	    for (int i = 1; i < argc; i++){
        	if ((argv[i][0]=='-')&&(argv[i][1]=='-')) break;
            	    vArgv.push_back(argv[i]);
    	    }
        }


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

#ifdef WIN32
	system("pause");
#endif
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
#include <vector_types.h>
#include "CCuda.h"
bool bReevaluate = false;


using namespace std;
extern "C" __global__ void cudaEvaluatePopulation(void* d_population, unsigned popSize, float* d_fitnesses, int offset);
#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm *EA;

#define CUDA_TPL

struct gpuEvaluationData* gpuData;

int fstGpu = 0;
int lstGpu = 0;


struct gpuEvaluationData* globalGpuData;
float* fitnessTemp;  
bool freeGPU = false;
bool first_generation = true;
int num_gpus = 0;       // number of CUDA GPUs

PopulationImpl* Pop = NULL;

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

\INSERT_USER_FUNCTIONS


void dispatchPopulation(int populationSize){
  int noTotalMP = 0; // number of MP will be used to distribute the population
  int count = 0;

  //Recuperation of each device information's.
  for( int index = 0; index < num_gpus; index++){
    cudaDeviceProp deviceProp;
    cudaError_t lastError = cudaGetDeviceProperties(&deviceProp, index+fstGpu);
    if( lastError!=cudaSuccess ){
      std::cerr << "Cannot get device information for device no : " << index+fstGpu << std::endl;
      exit(-1);
    }

    globalGpuData[index].num_MP =  deviceProp.multiProcessorCount; 
    globalGpuData[index].num_Warp = deviceProp.warpSize;
    noTotalMP += globalGpuData[index].num_MP;
    globalGpuData[index].gpuProp = deviceProp;
  }

  for( int index = 0; index < num_gpus; index++){

    globalGpuData[index].indiv_start = count;

    if(index != (num_gpus - 1)) {
      globalGpuData[index].sh_pop_size = ceil((float)populationSize * (((float)globalGpuData[index].num_MP) / (float)noTotalMP) );
    
    }
    //On the last card we are going to place the remaining individuals.  
    else 
      globalGpuData[index].sh_pop_size = populationSize - count;
	     
    count += globalGpuData[index].sh_pop_size;	     
  }
}

void cudaPreliminaryProcess(struct gpuEvaluationData* localGpuData, int populationSize){


  //  here we will compute how to spread the population to evaluate on GPGPU cores

  struct cudaFuncAttributes attr;
  CUDA_SAFE_CALL(cudaFuncGetAttributes(&attr,cudaEvaluatePopulation));

  int thLimit = attr.maxThreadsPerBlock;
  int N = localGpuData->sh_pop_size;
  int w = localGpuData->gpuProp.warpSize;

  int b=0,t=0;
	      
  do{
    b += localGpuData->num_MP;
    t = ceilf( MIN(thLimit,(float)N/b)/w)*w;
  } while( (b*t<N) || t>thLimit );
	      
  if( localGpuData->d_population!=NULL ){ cudaFree(localGpuData->d_population); }
  if( localGpuData->d_fitness!=NULL ){ cudaFree(localGpuData->d_fitness); }

  CUDA_SAFE_CALL(cudaMalloc(&localGpuData->d_population,localGpuData->sh_pop_size*(sizeof(IndividualImpl))));
  CUDA_SAFE_CALL(cudaMalloc(((void**)&localGpuData->d_fitness),localGpuData->sh_pop_size*sizeof(float)));


  std::cout << "card (" << localGpuData->threadId << ") " << localGpuData->gpuProp.name << " has " << localGpuData->sh_pop_size << " individual to evaluate" 
	    << ": t=" << t << " b: " << b << std::endl;
   localGpuData->dimGrid = b;
   localGpuData->dimBlock = t;

}

__device__ __host__ inline IndividualImpl* INDIVIDUAL_ACCESS(void* buffer,unsigned id){
  return (IndividualImpl*)buffer+id;
}

__device__ float cudaEvaluate(void* devBuffer, unsigned id){
  \INSERT_CUDA_EVALUATOR
}
  

extern "C" 
__global__ void cudaEvaluatePopulation(void* d_population, unsigned popSize, float* d_fitnesses, int offset){

        unsigned id = (blockDim.x*blockIdx.x)+threadIdx.x + offset;  // id of the individual computed by this thread

  	// escaping for the last block
        if( id >= popSize ) return;
  
        //void* indiv = ((char*)d_population)+id*(\GENOME_SIZE+sizeof(IndividualImpl*)); // compute the offset of the current individual
        d_fitnesses[id] = cudaEvaluate(d_population,id);
}



void* gpuThreadMain(void* arg){

  cudaError_t lastError;
  struct gpuEvaluationData* localGpuData = (struct gpuEvaluationData*)arg;
  //std::cout << " gpuId : " << localGpuData->gpuId << std::endl;

  lastError = cudaSetDevice(localGpuData->gpuId);

  if( lastError != cudaSuccess ){
    std::cerr << "Error, cannot set device properly for device no " << localGpuData->gpuId << std::endl;
    exit(-1);
  }
  
  int nbr_cudaPreliminaryProcess = 2;

  //struct my_struct_gpu* localGpuInfo = gpu_infos+localArg->threadId;


  if( lastError != cudaSuccess ){
    std::cerr << "Error, cannot get function attribute for cudaEvaluatePopulation on card: " << localGpuData->gpuProp.name  << std::endl;
    exit(-1);
  }
  
  // Because of the context of each GPU thread, we have to put all user's CUDA 
  // initialisation here if we want to use them in the GPU, otherwise they are
  // not found in the GPU context
  \INSERT_USER_CUDA

  // Wait for population to evaluate
   while(1){
	    sem_wait(&localGpuData->sem_in);

	    if( freeGPU ) {
	      // do we need to free gpu memory ?
	      cudaFree(localGpuData->d_fitness);
	      cudaFree(localGpuData->d_population);
	      break;
	    }

	    if(nbr_cudaPreliminaryProcess > 0) {
	      
	      if( nbr_cudaPreliminaryProcess==2 ) 
		cudaPreliminaryProcess(localGpuData,EA->population->parentPopulationSize);
	      else {
		cudaPreliminaryProcess(localGpuData,EA->population->offspringPopulationSize);
	      }
	      nbr_cudaPreliminaryProcess--;

	      if( localGpuData->dimBlock*localGpuData->dimGrid!=localGpuData->sh_pop_size ){
		// due to lack of individuals, the population distribution is not optimal according to core organisation
		// warn the user and propose a proper configuration
		std::cerr << "Warning, population distribution is not optimal, consider adding " << (localGpuData->dimBlock*localGpuData->dimGrid-localGpuData->sh_pop_size) 
			  << " individuals to " << (nbr_cudaPreliminaryProcess==2?"parent":"offspring")<<" population" << std::endl;
	      }
            }
	    
	    // transfer data to GPU memory
            lastError = cudaMemcpy(localGpuData->d_population,(IndividualImpl*)(Pop->cudaBuffer)+localGpuData->indiv_start,
				   (sizeof(IndividualImpl)*localGpuData->sh_pop_size),cudaMemcpyHostToDevice);

	    CUDA_SAFE_CALL(lastError);
/********************************************************************/
/* This part of code is added to avoid a force terminating kernel by timeout */
/* It could happend when execution time of kernel > 10s  */
/* For this we launch each kernel for one block          */
/* And use sreames to avoid lost of time                 */
int x_offset = 0;
cudaStream_t stream[localGpuData->dimGrid];

for (int u=0; u != localGpuData->dimGrid; u+=1)
{
    cudaStreamCreate(&stream[u]);
    x_offset = u*localGpuData->dimBlock;
    cudaEvaluatePopulation<<< 1, localGpuData->dimBlock, 0, stream[u]>>>(localGpuData->d_population, localGpuData->sh_pop_size, localGpuData->d_fitness, x_offset);
}
for (int u = 0; u != localGpuData->dimGrid; u+=1)
{
    cudaStreamSynchronize(stream[u]);
    cudaStreamDestroy(stream[u]);
}
	    
/********************************************************************/	    	    
	    //std::cout << localGpuData->sh_pop_size << ";" << localGpuData->dimGrid << ";"<<  localGpuData->dimBlock << std::endl;				      
	    // the real GPU computation (kernel launch)

/***********************************************************************/
/* This part of code is replaced by the code above */ 
/*	    cudaEvaluatePopulation<<< localGpuData->dimGrid, localGpuData->dimBlock>>>(localGpuData->d_population, localGpuData->sh_pop_size, localGpuData->d_fitness);
	    lastError = cudaGetLastError();
	    CUDA_SAFE_CALL(lastError);

	    if( cudaGetLastError()!=cudaSuccess ){ std::cerr << "Error during synchronize" << std::endl; }

	    // be sure the GPU has finished computing evaluations, and get results to CPU
	    lastError = cudaThreadSynchronize();
	    if( lastError!=cudaSuccess ){ std::cerr << "Error during synchronize" << std::endl; }
*/	
	    lastError = cudaMemcpy(fitnessTemp + localGpuData->indiv_start, localGpuData->d_fitness, localGpuData->sh_pop_size*sizeof(float), cudaMemcpyDeviceToHost);
	    
	    // this thread has finished its phase, so lets tell it to the main thread
	    sem_post(&localGpuData->sem_out);
   }
  sem_post(&localGpuData->sem_out);
  fflush(stdout);
  return NULL;
}
				
void wake_up_gpu_thread(){
	for( int i=0 ; i<num_gpus ; i++ ){
		sem_post(&(globalGpuData[i].sem_in));
	
  	}
	for( int i=0 ; i<num_gpus ; i++ ){
	  sem_wait(&globalGpuData[i].sem_out);
  	}

}
				
void InitialiseGPUs(){
	//MultiGPU part on one CPU
	globalGpuData = (struct gpuEvaluationData*)malloc(sizeof(struct gpuEvaluationData)*num_gpus);
	pthread_t* t = (pthread_t*)malloc(sizeof(pthread_t)*num_gpus);
	int gpuId = fstGpu;
	//here we want to create on thread per GPU
	for( int i=0 ; i<num_gpus ; i++ ){
	  
		globalGpuData[i].d_fitness = NULL;
		globalGpuData[i].d_population = NULL;
		
		globalGpuData[i].gpuId = gpuId++;

	  	globalGpuData[i].threadId = i;
	  	sem_init(&globalGpuData[i].sem_in,0,0);
	  	sem_init(&globalGpuData[i].sem_out,0,0);
	  	if( pthread_create(t+i,NULL,gpuThreadMain,globalGpuData+i) ){ perror("pthread_create : "); }
	}
}

\INSERT_INITIALISATION_FUNCTION
\INSERT_FINALIZATION_FUNCTION

void evale_pop_chunk(CIndividual** population, int popSize){
  \INSTEAD_EVAL_FUNCTION
}

void EASEAInit(int argc, char** argv){
  fstGpu = setVariable("fstgpu",0);
  lstGpu = setVariable("lstgpu",0);

	if( lstGpu==fstGpu && fstGpu==0 ){
	  // use all gpus available
	  cudaGetDeviceCount(&num_gpus);
	}
	else{
	  int queryGpuNum;
	  cudaGetDeviceCount(&queryGpuNum);
	  if( (lstGpu - fstGpu)>queryGpuNum || fstGpu<0 || lstGpu>queryGpuNum){
	    std::cerr << "Error, not enough devices found on the system ("<< queryGpuNum <<") to satisfy user configuration ["<<fstGpu<<","<<lstGpu<<"["<<std::endl;
	    exit(-1);
	  }
	  else{
	    num_gpus = lstGpu-fstGpu;
	  }
	}

	//globalGpuData = (struct gpuEvaluationData*)malloc(sizeof(struct gpuEvaluationData)*num_gpus);
	InitialiseGPUs();
	\INSERT_INIT_FCT_CALL
}

void EASEAFinal(CPopulation* pop){
	freeGPU=true;
	wake_up_gpu_thread();
        free(globalGpuData);
	
	\INSERT_FINALIZATION_FCT_CALL;
}

void AESAEBeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	\INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	\INSERT_END_GENERATION_FUNCTION
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


void IndividualImpl::mutate( float pMutationPerGene ){
  this->valid=false;


  // ********************
  // Problem specific part
  \INSERT_MUTATOR
}


void PopulationImpl::evaluateParentPopulation(){
        unsigned actualPopulationSize = this->actualParentPopulationSize;
	fitnessTemp = new float[actualPopulationSize];
	int index;
	static bool dispatchedParents = false;
	
	if( dispatchedParents==false ){
	  dispatchPopulation(EA->population->parentPopulationSize);
	  dispatchedParents=true;
	}

	       	
 	wake_up_gpu_thread(); 

 	
	for( index=(actualPopulationSize-1); index>=0; index--){
		this->parents[index]->fitness = fitnessTemp[index];
		this->parents[index]->valid = true;
	}  

        delete[](fitnessTemp);

}

void PopulationImpl::evaluateOffspringPopulation(){
	unsigned actualPopulationSize = this->actualOffspringPopulationSize;
	fitnessTemp = new float[actualPopulationSize];
	int index;
	static bool dispatchedOffspring = false;
	
	if( dispatchedOffspring==false ){
	  dispatchPopulation(EA->population->offspringPopulationSize);
	  dispatchedOffspring=true;
	}

        for( index=(actualPopulationSize-1); index>=0; index--)
	    ((IndividualImpl*)this->offsprings[index])->copyToCudaBuffer(this->cudaBuffer,index);

        wake_up_gpu_thread(); 

	for( index=(actualPopulationSize-1); index>=0; index--){
		this->offsprings[index]->fitness = fitnessTemp[index];
		this->offsprings[index]->valid = true;
	}	  
 
        first_generation = false;
        delete[](fitnessTemp);
}





void ParametersImpl::setDefaultParameters(int argc, char** argv){
        this->minimizing = \MINIMAXI;
        this->nbGen = setVariable("nbGen",(int)\NB_GEN);
	this->nbCPUThreads = setVariable("nbCPUThreads", 1);
	omp_set_num_threads(this->nbCPUThreads);
	this->reevaluateImmigrants = setVariable("reevaluateImmigrants", 0);
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

        generationalCriterion = new CGenerationalCriterion(setVariable("nbGen",(int)\NB_GEN));
        controlCStopingCriterion = new CControlCStopingCriterion();
        timeCriterion = new CTimeCriterion(setVariable("timeLimit",\TIME_LIMIT));

	this->optimise=0;

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
	EZ_NB_GEN = (unsigned*)setVariable("nbGen",\NB_GEN);
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

inline void IndividualImpl::copyToCudaBuffer(void* buffer, unsigned id){
  
 memcpy(((IndividualImpl*)buffer)+id,this,sizeof(IndividualImpl)); 
  
}

void EvolutionaryAlgorithmImpl::initializeParentPopulation(){
    //DEBUG_PRT("Creation of %lu/%lu parents (other could have been loaded from input file)",this->params->parentPopulationSize-this->params->actualParentPopulationSize,this->params->parentPopulationSize);
    int index,Size = this->params->parentPopulationSize;
    
    if(this->params->startFromFile){
          ifstream AESAE_File("EASEA.pop");
          string AESAE_Line;
          for( index=(Size-1); index>=0; index--) {
             getline(AESAE_File, AESAE_Line);
            this->population->addIndividualParentPopulation(new IndividualImpl(),index);
            ((IndividualImpl*)this->population->parents[index])->deserialize(AESAE_Line);
            ((IndividualImpl*)this->population->parents[index])->copyToCudaBuffer(((PopulationImpl*)this->population)->cudaBuffer,index);
         }

        }
        else{
                for( index=(Size-1); index>=0; index--) {
                         this->population->addIndividualParentPopulation(new IndividualImpl(),index);
                        ((IndividualImpl*)this->population->parents[index])->copyToCudaBuffer(((PopulationImpl*)this->population)->cudaBuffer,index);
                }
    }
    
    this->population->actualOffspringPopulationSize = 0;
    this->population->actualParentPopulationSize = Size;
}


EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){

  // warning cstats parameter is null
  this->population = (CPopulation*)new
  PopulationImpl( this->params->parentPopulationSize,this->params->offspringPopulationSize,
                  this->params->pCrossover,this->params->pMutation,this->params->pMutationPerGene,
                  this->params->randomGenerator,this->params,this->cstats);

  int popSize = (params->parentPopulationSize>params->offspringPopulationSize?params->parentPopulationSize:params->offspringPopulationSize);
  ((PopulationImpl*)this->population)->cudaBuffer = (void*)malloc(sizeof(IndividualImpl)*( popSize ));
  
  // = new CCuda(params->parentPopulationSize, params->offspringPopulationSize, sizeof(IndividualImpl));
  Pop = ((PopulationImpl*)this->population);
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

PopulationImpl::PopulationImpl(unsigned parentPopulationSize, unsigned offspringPopulationSize, float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params, CStats* stats) : CPopulation(parentPopulationSize, offspringPopulationSize, pCrossover, pMutation, pMutationPerGene, rg, params, stats){
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
#include <string>
#include <CStats.h>
#include "CCuda.h"
#include <sstream>

using namespace std;

class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;
class CCuda;


\INSERT_USER_CLASSES

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

	void mutate(float pMutationPerGene);

	void boundChecking();

	string serialize();
	void deserialize(string AESAE_Line);
	void copyToCudaBuffer(void* buffer, unsigned id);

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
  //CCuda *cuda;
  void* cudaBuffer;

public:
  PopulationImpl(unsigned parentPopulationSize, unsigned offspringPopulationSize, float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params, CStats* stats);
        virtual ~PopulationImpl();
        void evaluateParentPopulation();
	void evaluateOffspringPopulation();
};

#endif /* PROBLEM_DEP_H */

\START_CUDA_MAKEFILE_TPL
NVCC= nvcc 
CPPC= g++ 
LIBAESAE=$(EZ_PATH)libeasea/
CXXFLAGS+= -std=c++11 -g -Wall -O2 -I$(LIBAESAE)include 
LDFLAGS=$(LIBAESAE)libeasea.a -lpthread 
 


EASEA_SRC= EASEAIndividual.cpp
EASEA_MAIN_HDR= EASEA.cpp
EASEA_UC_HDR= EASEAIndividual.hpp

EASEA_HDR= $(EASEA_SRC:.cpp=.hpp) 

SRC= $(EASEA_SRC) $(EASEA_MAIN_HDR)
CUDA_SRC = EASEAIndividual.cu
HDR= $(EASEA_HDR) $(EASEA_UC_HDR)
OBJ= $(EASEA_SRC:.cpp=.o) $(EASEA_MAIN_HDR:.cpp=.o)

#USER MAKEFILE OPTIONS :
\INSERT_MAKEFILE_OPTION#END OF USER MAKEFILE OPTIONS

CPPFLAGS+= -I$(LIBAESAE)include  -I/usr/local/cuda/include/
NVCCFLAGS+= -std=c++11 #--ptxas-options="-v"# --gpu-architecture sm_23 --compiler-options -fpermissive 


BIN= EASEA
  
all:$(BIN)

$(BIN):$(OBJ)
	$(NVCC) $^ -o $@ $(LDFLAGS) -Xcompiler -fopenmp

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -c -DTIMING $(CPPFLAGS) -g -Xcompiler -fopenmp 

easeaclean: clean
	rm -f Makefile EASEA.prm $(SRC) $(HDR) EASEA.mak $(CUDA_SRC) *.linkinfo EASEA.png EASEA.dat EASEA.vcproj EASEA.plot EASEA.r EASEA.csv EASEA.pop
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
				DebugInformationFormat="3"
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
				GenerateDebugInformation="true"
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
