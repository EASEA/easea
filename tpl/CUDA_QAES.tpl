\TEMPLATE_START
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libEasea.lib")
#pragma comment(lib, "Winmm.lib")
#endif
/**
 This is program entry for CUDA QAES template for EASEA
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
int EZ_POP_SIZE;
int OFFSPRING_SIZE;
CEvolutionaryAlgorithm* EA;
std::vector<char *> vArgv;


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
#include <fstream>
#include <time.h>
#include <cstring>
#include <sstream>
#include <chrono>

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

#include <shared/distributions/Norm.h>
#include <shared/distributions/Cauchy.h>
#include <shared/distributions/Unif.h>

#include "config.h"
#ifdef USE_OPENMP
	#include <omp.h>
#endif


bool bReevaluate = false;


using namespace std;
#include "EASEAIndividual.hpp"
extern "C" __global__ void cudaEvaluatePopulation(void* d_population, unsigned popSize, TO* d_fitnesses, int offset);

bool INSTEAD_EVAL_STEP = false;
extern std::ofstream easena::log_file;
extern easena::log_stream logg;
CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm *EA;
extern CEvolutionaryAlgorithm *EA;
size_t limitGen;
bool reset;

#define CUDA_TPL

struct gpuEvaluationData<TO>* gpuData;

int fstGpu = 0;
int lstGpu = 0;

double ACC_1 = 0.4;
double ACC_2 = 0.6;
double LIMIT_UPDATE = 200;
#define SZ_POP_MAX 100000

struct gpuEvaluationData<TO>* globalGpuData;
TO* fitnessTemp;  
bool freeGPU = false;
bool first_generation = true;
int num_gpus = 0;       // number of CUDA GPUs

PopulationImpl* Pop = NULL;

\INSERT_USER_DECLARATIONS
void* cudaBuffer;

  /* Get the size of serach space dimension from problem difinition by user in ez file*/
        const size_t nbVar = m_problem.getBoundary().size();

        const size_t szPopMax = SZ_POP_MAX; /* max value of possible population size */;
        size_t szPop;

        /* Define quantum population */
        std::vector<std::vector<TV>>pop_x(szPopMax, std::vector<TV>(nbVar));
	std::vector<std::vector<TV>>pop_bank_1(szPopMax, std::vector<TV>(nbVar));

	std::vector<std::vector<TV>>pop_bank_2(szPopMax, std::vector<TV>(nbVar));
	std::vector<std::vector<TV>>pop_bank_3(szPopMax, std::vector<TV>(nbVar));

        std::vector<std::vector<TV>>pop_pos_best(szPopMax, std::vector<TV>(nbVar));


        /* Objective function f(x) is the potential well V(x) in Schrödinger equation */
        std::vector<TO> F(szPopMax);
        /* Best objective function ever been for each particle */
        std::vector<TO> bestF(szPopMax);
        /* Weight function for each particle */
        std::vector<TV> bestM(szPopMax);
        /* Memory - needed to avoid local optimum - not used yet */
        std::vector<TV> memory(nbVar);
        /* Mean value of solution */
        std::vector<TV> meanSolution(nbVar);
        /* global solution */
        std::vector<TV> globalSolution(nbVar);
        /* flags = 0 - particle is deleted, flags = 1 - particle is alive */
        std::vector<int> flags(szPopMax);

        int globalIndex = 0;
        TO bestGlobal;    /* Global best value */

        int nbDeriv = 0;
        std::vector<TO> derivF(szPopMax);
        TO memoryF;
        TO koeff = 0;
        TO V = 0;
        TO Vtmp = 0;
	int szOldPop;



\ANALYSE_USER_CLASSES

\INSERT_USER_FUNCTIONS

int getGlobalSolutionIndex(int size, std::vector<TO> F)
{
        int i, ind = 0;

        TO minVal = F[0];

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
      globalGpuData[index].sh_pop_size = ceil((TO)populationSize * (((TO)globalGpuData[index].num_MP) / (TO)noTotalMP) );
    
    }
    //On the last card we are going to place the remaining individuals.  
    else 
      globalGpuData[index].sh_pop_size = populationSize - count;
	     
    count += globalGpuData[index].sh_pop_size;	     
  }
}

void cudaPreliminaryProcess(struct gpuEvaluationData<TO>* localGpuData, int populationSize){


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
  CUDA_SAFE_CALL(cudaMalloc(((void**)&localGpuData->d_fitness),localGpuData->sh_pop_size*sizeof(TO)));


//  std::cout << "card (" << localGpuData->threadId << ") " << localGpuData->gpuProp.name << " has " << localGpuData->sh_pop_size << " individual to evaluate" 
//	    << ": t=" << t << " b: " << b << std::endl;
   localGpuData->dimGrid = b;
   localGpuData->dimBlock = t;

}

__device__ __host__ inline IndividualImpl* INDIVIDUAL_ACCESS(void* buffer,unsigned id){
  return (IndividualImpl*)buffer+id;
}

__device__ TO cudaEvaluate(void* devBuffer, unsigned id){
  \INSERT_CUDA_EVALUATOR
}
  

extern "C" 
__global__ void cudaEvaluatePopulation(void* d_population, unsigned popSize, TO* d_fitnesses, int offset){

        unsigned id = (blockDim.x*blockIdx.x)+threadIdx.x + offset;  // id of the individual computed by this thread

  	// escaping for the last block
        if( id >= popSize ) return;
  
        //void* indiv = ((char*)d_population)+id*(\GENOME_SIZE+sizeof(IndividualImpl*)); // compute the offset of the current individual
        d_fitnesses[id] = cudaEvaluate(d_population,id);
}



void* gpuThreadMain(void* arg){

  cudaError_t lastError;
  struct gpuEvaluationData<TO>* localGpuData = (struct gpuEvaluationData<TO>*)arg;
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
	    if (szOldPop != szPop) nbr_cudaPreliminaryProcess = 1;

	    if(nbr_cudaPreliminaryProcess > 0) {
	      
	      if( nbr_cudaPreliminaryProcess==2 ) 
		cudaPreliminaryProcess(localGpuData, szPopMax);  //EA->population->parentPopulationSize);
	      else {
		cudaPreliminaryProcess(localGpuData, szPop); //EA->population->offspringPopulationSize);
	      }
	      nbr_cudaPreliminaryProcess--;

	      if( localGpuData->dimBlock*localGpuData->dimGrid!=localGpuData->sh_pop_size ){
		// due to lack of individuals, the population distribution is not optimal according to core organisation
		// warn the user and propose a proper configuration
		//std::cerr << "Warning, population distribution is not optimal, consider adding " << (localGpuData->dimBlock*localGpuData->dimGrid-localGpuData->sh_pop_size) 
		//	  << " individuals to " << (nbr_cudaPreliminaryProcess==2?"parent":"offspring")<<" population" << std::endl;
	      }
            }
	    
	    // transfer data to GPU memory
            lastError = cudaMemcpy(localGpuData->d_population,(IndividualImpl*)(/*Pop->*/cudaBuffer)+localGpuData->indiv_start,
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
	    lastError = cudaMemcpy(fitnessTemp + localGpuData->indiv_start, localGpuData->d_fitness, localGpuData->sh_pop_size*sizeof(TO), cudaMemcpyDeviceToHost);
	    
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
	globalGpuData = (struct gpuEvaluationData<TO>*)malloc(sizeof(struct gpuEvaluationData<TO>)*num_gpus);
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
                     if (bestGlobal < bBest->fitness){
                     for (int j = 0; j < nbVar; j++)
                         ((IndividualImpl*)(bBest))->\GENOME_NAME[j] = globalSolution[j];
}
    \INSERT_END_GENERATION_FUNCTION
}
void EASEAGenerationFunctionBeforeReplace([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    \INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}

void AESAEGenerationFunctionBeforeReplacement([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
        \INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}


void evaluateParentPopulation(){
        unsigned actualPopulationSize = szPopMax;
        fitnessTemp = new TO[actualPopulationSize];
        int index;
        static bool dispatchedParents = false;

        if( dispatchedParents==false ){
          dispatchPopulation(szPopMax);
          dispatchedParents=true;
        }
        for( index=(actualPopulationSize-1); index>=0; index--){
             IndividualImpl *tmp = new IndividualImpl(pop_x[index]);

            ((IndividualImpl*)tmp)->copyToCudaBuffer(cudaBuffer,index);
                delete(tmp);
        }
        wake_up_gpu_thread();

        for( index=(actualPopulationSize-1); index>=0; index--){
                bestF[index] = F[index] = fitnessTemp[index];
                derivF[index] = std::numeric_limits<TO>::max();
		if (flags[index] == 1) V+=F[index];
        }

        first_generation = false;
        delete[](fitnessTemp);

}
void evaluateOffspringPopulation(){
    
	static bool dispatchedOffspring = false;
	if (szOldPop != szPop) dispatchedOffspring = false;

        unsigned actualPopulationSize = szPop;
        fitnessTemp = new TO[actualPopulationSize];
        int index;

        if( dispatchedOffspring==false ){
          dispatchPopulation(szPop);
          dispatchedOffspring=true;
        }
        for( index=(actualPopulationSize-1); index>=0; index--){
          if (flags[index] == 1){

             IndividualImpl *tmp = new IndividualImpl(pop_x[index]);
        
            ((IndividualImpl*)tmp)->copyToCudaBuffer(cudaBuffer,index);
                delete(tmp);
                }
        }
        wake_up_gpu_thread();
        
        for( index=(actualPopulationSize-1); index>=0; index--){
        if (flags[index] == 1){
        
         F[index] = fitnessTemp[index];
                if (koeff==1){
                    for (int j = 0; j < nbVar; j++)
                        pop_pos_best[index][j] = pop_x[index][j];
                    bestF[index] = F[index];

                    globalIndex = getGlobalSolutionIndex(szPop, F);
                    for (int k = 0; k < nbVar; k++){
			    if (memoryF > bestGlobal)
                            memory[k] = globalSolution[k];
                            globalSolution[k] = pop_pos_best[globalIndex][k];
                    }
		    if (memoryF > bestGlobal)
                    memoryF = bestGlobal;
                    bestGlobal = bestF[globalIndex];

                }

                if (F[index] < bestF[index]){
                    for (int j = 0; j < nbVar; j++)
                        pop_pos_best[index][j] = pop_x[index][j];
                    bestF[index] = F[index];
                        
                    
                }/*else {
		    for (int j = 0; j < nbVar; j++)
			pop_x[index][j] = pop_pos_best[index][j];
		    F[index] = bestF[index];
		}*/
		if (bestF[index] < bestGlobal){
                        for (int j = 0; j < nbVar; j++)
                                globalSolution[j] = pop_pos_best[index][j];
                        bestGlobal = bestF[index];
                }

        }}      
	delete[](fitnessTemp);
}
 
void EvolutionaryAlgorithmImpl::runEvolutionaryLoop(){
        /* Start logging */
        LOG_MSG(msgType::INFO, "QAES CUDA version starting....");
        auto tmStart = std::chrono::system_clock::now();

        string log_fichier_name = params->outputFilename;
        time_t t = std::chrono::system_clock::to_time_t(tmStart);
        std::tm * ptm = std::localtime(&t);
        char buf_start_time[32];
	std::strftime(buf_start_time, 32, "%Y-%m-%d_%H-%M-%S", ptm);
        easena::log_file.open(log_fichier_name.c_str() + std::string("_") + std::string(buf_start_time) + std::string(".log"));

        /* koeff controls the local optimums - if koeff = 1, we are very probably in local optimum */
        bBest = new IndividualImpl();
        bBest->fitness = std::numeric_limits<TO>::max();

        szPop = this->params->parentPopulationSize;
	cudaBuffer = (void*)malloc(sizeof(IndividualImpl)*( szPopMax ));
	szOldPop = szPop;
        limitGen = EZ_NB_GEN[0];  /* Get maximal number of geneation from the prm file, defined by user */

        
        TV cAcc_1, cAcc_2, a;
        currentGeneration = 0;  /* Counter of generation */
        size_t currentEval;     /* Counter of evaluation number */

        /* Check all cases of stop critaria */
        while( this->allCriteria() == false){

            clock_t starttime, endtime;
            double totaltime;
            starttime = clock();

            /* Launch user settings from section "befor everything else" in ez file */
            EASEABeginningGeneration(this);
            if (currentGeneration == 0){
            /* First generation settings */
		a = 0.9;
                /* Init and evaluate populations */
                for (int i = 0; i < szPopMax; i++){
                    for (int j = 0; j < nbVar; j++){
                        pop_x[i][j] = (rand() / (TV)(RAND_MAX + 1.)*(m_problem.getBoundary()[j].second -
m_problem.getBoundary()[j].first) + m_problem.getBoundary()[j].first);
                        pop_pos_best[i][j] = pop_x[i][j];
		/*	pop_bank_1[i][j] = pop_x[i][j];
			pop_bank_2[i][j] = pop_x[i][j];
			pop_bank_3[i][j] = pop_x[i][j];*/
			 if (i < szPop) flags[i] = 1;
                         else { flags[i] = 0;} 
                    }
                }

                evaluateParentPopulation();
                /* Avarage cost function value (potential energy) */
                V = V/(TO)szPop;

                /* Get index of current global optimum */
                globalIndex = getGlobalSolutionIndex(szPop, F);

                /* Get desicion variables and value of fitness function of current global solution */
                for (int i = 0; i < nbVar; i++)
                    globalSolution[i] = pop_pos_best[globalIndex][i];
                bestGlobal = bestF[globalIndex];
		memoryF = bestF[globalIndex];
            }
            TV tmp = 0.;
            /* Get mean value of all current solutions */
            for (int i = 0; i < nbVar; i++)
            {
                tmp = 0.;
                for ( int j = 0; j < szPop; j++)
                        tmp = tmp + pop_pos_best[j][i];
                        meanSolution[i] = tmp / (TV)szPop;
            }
            /* Alpha for QPSO part */
        //    TV a =  0.5 * (this->params->nbGen - currentGeneration)/(this->params->nbGen) + 0.5;
            if (currentGeneration == 0){
                currentEval = 0;
                /* Initialization of  population if the first generation */
                //this->initializeParentPopulation();
            }
	    

            /* Logging current generation number */
            ostringstream ss;
            Vtmp = 0;
            
            
 for ( int i = 0; i < szPopMax; i++){
              if (flags[i] == 1){
              for (int j = 0; j < nbVar; j++){
                  TV fi1 = rand()/(TV)(RAND_MAX+1.0);
                  TV fi2 = rand()/(TV)(RAND_MAX+1.0);

                  if (koeff == 0){
                      cAcc_1 = ACC_1;
                      cAcc_2 = ACC_2;
		      a = 0.85;
                  }else { cAcc_1 = 0.65; cAcc_2 = 0.45; a= easea::shared::distributions::unif(0.65,0.95);}
                  /* Crossover "à la QPSO" */
                  TV fi = cAcc_1*fi1/(cAcc_1*fi1+cAcc_2*fi2);
                  TV p = pop_pos_best[i][j]*fi + (1-fi)*globalSolution[j];
                  /* if there is local optimum -> let's make the large diffusion displacement */

                  TV u = rand()/(TV)(RAND_MAX+1.0);
                  TV b;
                  TV v = log(1/(TV)u);
                  TV z = rand()/(TV)(RAND_MAX+1.0);
                  TV t = pop_x[i][j];
                  TV sig = fabs(meanSolution[j] - pop_x[i][j]);
                  /*if (sig < 0.0001)*/ 
		  sig = (1./(TV)sqrt(sig));

                  if (koeff == 1)
    		  {
		    TV z = rand()/(TV)(RAND_MAX+1.0);
		    if (z < 0.5) t = easea::shared::distributions::cauchy(pop_x[i][j], sig);
		  }

                  b = a * fabs(meanSolution[j] - t);
                  if ( z < 0.5) t = (p + b * v);
                       else t = (p - b * v);
                  if (t < m_problem.getBoundary()[j].first) t = m_problem.getBoundary()[j].first;
                  if (t > m_problem.getBoundary()[j].second) t = m_problem.getBoundary()[j].second;
/*
		  pop_bank_3[i][j] = pop_bank_2[i][j];
		  pop_bank_2[i][j] = pop_bank_1[i][j];
		  pop_bank_1[i][j] = pop_x[i][j];     */
		  pop_x[i][j] = t;
              }
            }}
 if (koeff == 1){
     int curSzPop = std::count (flags.begin(), flags.end(), 1);
         szOldPop = szPop;
//	if (currentGeneration > 2000) LIMIT_UPDATE = 1500;

         for (int h = 0; h < 10; h++){
	    flags[curSzPop] = 1;

           for (int j = 0; j < nbVar; j++){
	        double ii = 1./(TV)sqrt(fabs(meanSolution[j] - memory[j]));
	        pop_x[curSzPop+h][j] = easea::shared::distributions::cauchy(memory[j], ii);
    
         }
         curSzPop++;
         szPop++;
         }
     }

        evaluateOffspringPopulation();
      TO epsilonF = 0.001;
      /* Control of local optimum */
          int count = 0;
          for (int i = 0; i < szPop; i++){
              TO curDerivF = fabs(bestF[i] - bestGlobal);
              if (fabs(curDerivF - derivF[i]) <= epsilonF) count++;
              derivF[i] = curDerivF;
          }
          int curSzPop = std::count (flags.begin(), flags.end(), 1);

      if (count >= curSzPop-1) nbDeriv++;
      if (bestGlobal < 2) nbDeriv = 0;
      if (nbDeriv >= LIMIT_UPDATE) {koeff = 1;nbDeriv = 0;}
    	else koeff = 0;
                currentGeneration++;
                EASEAEndGeneration(this);
                if (reset == true)
                {
                    currentGeneration = 0;
                    reset = false;
                }

               if (bestGlobal < bBest->fitness){
            /*    for (int j = 0; j < nbVar; j++)
                    ((IndividualImpl*)(bBest))->\GENOME_NAME[j] = globalSolution[j];
	    */
                bBest->fitness = bestGlobal;
                }
                ss << "Generation: " << currentGeneration << " Best solution: " << bBest->fitness /*<< " Current best solution: " << bestGlobal*/ << std::endl;
                LOG_MSG(msgType::INFO, ss.str());
        }
        std::chrono::duration<double> tmDur = std::chrono::system_clock::now() - tmStart;
        ostringstream ss;
        ss << "Total execution time (in sec.): " << tmDur.count() << std::endl;
        LOG_MSG(msgType::INFO, ss.str());
        delete(bBest);


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

  fstGpu = setVariable("fstgpu", 0);
  lstGpu = setVariable("lstgpu", 0);

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

	//globalGpuData = (struct gpuEvaluationData<TO>*)malloc(sizeof(struct gpuEvaluationData<TO>)*num_gpus);
	InitialiseGPUs();
	\INSERT_INITIALISATION_FUNCTION
}

void EASEAFinal(CPopulation* pop){
	freeGPU=true;
	wake_up_gpu_thread();
        free(globalGpuData);
	
	\INSERT_FINALIZATION_FCT_CALL;
}
/*
void AESAEBeginningGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
	\INSERT_END_GENERATION_FUNCTION
}

void AESAEGenerationFunctionBeforeReplacement([[maybe_unused]] CEvolutionaryAlgorithm* evolutionaryAlgorithm) {
        \INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}
*/


IndividualImpl::IndividualImpl() : CIndividual() {
  \GENOME_CTOR 
  \INSERT_EO_INITIALISER
  valid = false;
  isImmigrant = false;
}

IndividualImpl::IndividualImpl(std::vector<TV> ind)
     {
    for (size_t i = 0; i < m_problem.getBoundary().size(); i++)
       this->\GENOME_NAME[i] = ind[i];
     }
CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

IndividualImpl::~IndividualImpl(){
  \GENOME_DTOR
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
  \INSERT_MUTATOR

  return 0;
}

/*
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
}*/


ParametersImpl::ParametersImpl(std::string const& file, int argc, char* argv[]) : Parameters(file, argc, argv) {
        this->minimizing = \MINIMAXI;
        this->nbGen = setVariable("nbGen", (int)\NB_GEN);

	#ifdef USE_OPENMP
	omp_set_num_threads(this->nbCPUThreads);
	#endif

        globalRandomGenerator = new CRandomGenerator(seed);
        this->randomGenerator = globalRandomGenerator;

        selectionOperator = getSelectionOperator(setVariable("selectionOperator", "\SELECTOR_OPERATOR"), this->minimizing, globalRandomGenerator);
        replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator", "\RED_FINAL_OPERATOR"),this->minimizing, globalRandomGenerator);
        parentReductionOperator = getSelectionOperator(setVariable("reduceParentsOperator", "\RED_PAR_OPERATOR"),this->minimizing, globalRandomGenerator);
        offspringReductionOperator = getSelectionOperator(setVariable("reduceOffspringOperator", "\RED_OFF_OPERATOR"),this->minimizing, globalRandomGenerator);
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
	//EZ_NB_GEN = (unsigned*)setVariable("nbGen", \NB_GEN);
	EZ_current_generation=0;


	CEvolutionaryAlgorithm* ea = new EvolutionaryAlgorithmImpl(this);
	cudaBuffer = (void*)malloc(sizeof(IndividualImpl)*( szPop ));
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
          ifstream AESAE_File(this->params->inputFilename);
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
#include <problems/CProblem.h>
#include <sstream>

using namespace std;

class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;
class CCuda;


typedef double TV;
typedef double TO;

typedef easea::problem::CProblem<TV> TP;

typedef typename easea::shared::CBoundary<TV>::TBoundary TBoundary;


\INSERT_USER_CLASSES

class IndividualImpl : public CIndividual {

public: // in EASEA the genome is public (for user functions,...)
	// Class members
	TO fitness;
  	\INSERT_GENOME


public:
	IndividualImpl();
	IndividualImpl(const IndividualImpl& indiv);
	IndividualImpl(std::vector<TV> ind);
	virtual ~IndividualImpl();
	float evaluate() override;
	CIndividual* crossover(CIndividual** p2);
	void printOn(std::ostream& O) const;
	CIndividual* clone();

	unsigned mutate(float pMutationPerGene);

	void boundChecking();

	string serialize();
	void deserialize(string AESAE_Line);
	void copyToCudaBuffer(void* buffer, unsigned id);

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
 void runEvolutionaryLoop();

};

class PopulationImpl: public CPopulation {
public:
  //CCuda *cuda;
  void* cudaBuffer;

public:
  PopulationImpl(unsigned parentPopulationSize, unsigned offspringPopulationSize, float pCrossover, float pMutation, float pMutationPerGene, CRandomGenerator* rg, Parameters* params, CStats* stats);
        virtual ~PopulationImpl();
//        void evaluateParentPopulation();
//	void evaluateOffspringPopulation();
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
#  Parameter file generated by CUDA.tpl AESAE v1.0
#
#***************************************
# --seed=0   # -S : Random number seed. It is possible to give a specific seed.

######    Evolution Engine    ######
--popSize=10 #\POP_SIZE # -P : Population Size
--nbOffspring=10 #\OFF_SIZE # -O : Nb of offspring (percentage or absolute)

######    Stopping Criterions    #####
--nbGen=200000 #\NB_GEN #Nb of generations
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
