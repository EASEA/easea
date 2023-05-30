\TEMPLATE_START
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libEasea.lib")
#pragma comment(lib, "Winmm.lib")
#endif
/**
 This is program entry for CUDA_GP template for EASEA
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


	ParametersImpl p("EASEA.prm", argc, argv);
	CEvolutionaryAlgorithm* ea = p.newEvolutionaryAlgorithm();

	EA = ea;

	EASEAInit(argc, argv, p);

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
#include "CGPNode.h"

#include "config.h"
#ifdef USE_OPENMP
	#include <omp.h>
#endif


using namespace std;
bool bReevaluate = false;
extern "C"
__global__ void 
EvaluatePostFixIndividuals( const float * k_progs, const int maxprogssize,  const int popsize, const float * k_inputs, const float * outputs, const int trainingSetSize, float * k_results,  int* k_indexes );


#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm *EA;

#define CUDAGP_TPL

typedef float TO;
typedef float TV;

#define HIT_LEVEL  0.01f
#define PROBABLY_ZERO  1.11E-15f
#define BIG_NUMBER 1.0E15f


unsigned aborded_crossover;
float** inputs;
float* outputs;


struct gpuEvaluationData<TO>* gpuData;

int fstGpu = 0;
int lstGpu = 0;


struct gpuEvaluationData<TO>* globalGpuData;
float* fitnessTemp;  
bool freeGPU = false;
bool first_generation = true;
int num_gpus = 0;       // number of CUDA GPUs

PopulationImpl* Pop = NULL;

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

\INSERT_GP_PARAMETERS
\ANALYSE_GP_OPCODE
/* Insert declarations about opcodes*/
\INSERT_GP_OPCODE_DECL




GPNode* ramped_hh(){
  return RAMPED_H_H(INIT_TREE_DEPTH_MIN,INIT_TREE_DEPTH_MAX,EA->population->actualParentPopulationSize,EA->population->parentPopulationSize,0, VAR_LEN, OPCODE_SIZE,opArity, OP_ERC);
}

std::string toString(GPNode* root){
  return toString(root,opArity,opCodeName,OP_ERC);
}


\INSERT_USER_CLASSES
\INSERT_USER_FUNCTIONS

\INSERT_FINALIZATION_FUNCTION

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

void cudaPreliminaryProcessGP(struct gpuEvaluationData<TO>* localGpuData){

  //  here we will compute how to spread the population to evaluate on GPGPU cores
  struct cudaFuncAttributes attr;

  CUDA_SAFE_CALL(cudaFuncGetAttributes(&attr,EvaluatePostFixIndividuals));

  int thLimit = attr.maxThreadsPerBlock;
  //int N = localGpuData->sh_pop_size;
  //int w = localGpuData->gpuProp.warpSize;

  int b=0,t=0;

  if( thLimit < NUMTHREAD ){
    
  }

  b = ceilf(((float)localGpuData->sh_pop_size)/localGpuData->num_MP)*localGpuData->num_MP;
  t = NUMTHREAD;

  b = ( b<localGpuData->gpuProp.maxGridSize[0] ? b : localGpuData->gpuProp.maxGridSize[0]);
	      
  if( localGpuData->d_population!=NULL ){ cudaFree(localGpuData->d_population); }
  if( localGpuData->d_fitness!=NULL ){ cudaFree(localGpuData->d_fitness); }

  localGpuData->indexes = new int[localGpuData->sh_pop_size];
  localGpuData->fitness = new float[localGpuData->sh_pop_size];
  //std::cout << "mem : " << (sizeof(*localGpuData->d_indexes)*localGpuData->sh_pop_size) << std::endl;
  CUDA_SAFE_CALL(cudaMalloc(&localGpuData->d_indexes,sizeof(*localGpuData->d_indexes)*localGpuData->sh_pop_size));
  CUDA_SAFE_CALL(cudaMalloc(&localGpuData->d_fitness,sizeof(*localGpuData->d_fitness)*localGpuData->sh_pop_size));

  

  std::cout << "card (" << localGpuData->threadId << ") " << localGpuData->gpuProp.name << " has " << localGpuData->sh_pop_size << " individual to evaluate" 
	    << ": t=" << t << " b: " << b << std::endl;
   localGpuData->dimGrid = b;
   localGpuData->dimBlock = t;

}


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

__device__ float eval_tree_gpu(const float * k_progs, const float * input){
  float RESULT;
  float OP1, OP2;
  float stack[MAX_STACK];
  int sp=0;
  int start_prog = 0;
  int codop =  k_progs[start_prog++];


  while (codop != OP_RETURN){
    switch(codop){

\INSERT_GP_GPU_SWITCH
    }
    codop =  k_progs[start_prog++];
  }

  
  return stack[0];
}


extern "C"
__global__ void 
EvaluatePostFixIndividuals( const float * k_progs, const int maxprogssize,  const int popsize, const float * k_inputs,
			   const float * outputs, const int trainingSetSize, float * k_results,  int* k_indexes )
{
  __shared__ float tmpresult[NUMTHREAD];
  
  const int tid = threadIdx.x; //0 to NUM_THREADS-1
  const int bid = blockIdx.x; // 0 to NUM_BLOCKS-1

   
  for( int index = bid; index<popsize ; index+=gridDim.x ){
    //    int index;   // index of the prog processed by the block 
    float sum = 0.0;
    float error;

    // index = bid; // one program per block => block ID = program number
 
    if (index >= popsize) // idle block (should never occur)
      return;
    if (k_progs[index] == -1.0) // already evaluated
      return;

    // Here, it's a busy thread
    sum = 0.0;
  
    // Loop on training cases, per cluster of 32 cases (= number of thread)
    // (even if there are only 8 stream processors, we must spawn at least 32 threads) 
    // We loop from 0 to upper bound INCLUDED in case trainingSetSize is not 
    // a multiple of NUMTHREAD

    \INSERT_GENOME_EVAL_HDR;
    
    for (int i=tid; i < trainingSetSize ; i+=NUMTHREAD) {
    
      // are we on a busy thread?
      if (i >= trainingSetSize) // no!
	continue;
         
      float EVOLVED_VALUE = eval_tree_gpu( k_progs+k_indexes[index], k_inputs+i*VAR_LEN);
 
      \INSERT_GENOME_EVAL_BDY_GPU;

    
      if (!(error < BIG_NUMBER)) error = BIG_NUMBER;
      else if (error < PROBABLY_ZERO) error = 0.0;
    
    
      sum += error; // sum raw error on all training cases
    
    } // LOOP ON TRAINING CASES
  
    // gather results from all threads => we need to synchronize
    tmpresult[tid] = sum;

    __syncthreads();

    if (tid == 0) {
      for (int i = 1; i < NUMTHREAD; i++) {
	tmpresult[0] += tmpresult[i];
      }    
      error = tmpresult[0];
      \INSERT_GENOME_EVAL_FTR_GPU;
    }  
    // here results and hits have been stored in their respective array: we can leave
  }
}



int flattening_tree_rpn( GPNode* root, float* buf, int* index){
  for( unsigned i=0 ; i<opArity[(int)root->opCode] ; i++ ){
    flattening_tree_rpn(root->children[i],buf,index);
  }

  if( (*index)+2>MAX_PROGS_SIZE )return 0;
  buf[(*index)++] = root->opCode;
  if( root->opCode == OP_ERC ) buf[(*index)++] = root->erc_value;
  return 1;
}


int flatteningSubPopulation( struct gpuEvaluationData<TO>* localGpuData, IndividualImpl** population){
  int index = 0;
  for( int i=0 ; i<localGpuData->sh_pop_size ; i++ ){
    localGpuData->indexes[i] = index;
    flattening_tree_rpn( population[localGpuData->indiv_start+i]->root, localGpuData->progs, &index);
    localGpuData->progs[index++] = OP_RETURN;
    if( index > MAX_PROGS_SIZE ){
      std::cerr << "Error, impossible to flatten the population. Consider to increase the MAX_PROGS_SIZE. " << std::endl;
      exit(-1);
    }
  }
  return index;
}


void* gpuThreadMain(void* arg){

  int index = 0;
  int nbr_cudaPreliminaryProcess = 2;

  cudaError_t lastError;
  struct gpuEvaluationData<TO>* localGpuData = (struct gpuEvaluationData<TO>*)arg;

  CUDA_SAFE_CALL(cudaSetDevice(localGpuData->gpuId));

  CUDA_SAFE_CALL(cudaMalloc(&localGpuData->d_inputs,sizeof(*localGpuData->d_inputs)*VAR_LEN*NO_FITNESS_CASES));
  CUDA_SAFE_CALL(cudaMalloc(&localGpuData->d_outputs,sizeof(*localGpuData->d_outputs)*NO_FITNESS_CASES));
  // transfert inputs to GPGPU
  CUDA_SAFE_CALL(cudaMemcpy( localGpuData->d_inputs,localGpuData->flatInputs,sizeof(*localGpuData->d_inputs)*VAR_LEN*NO_FITNESS_CASES,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy( localGpuData->d_outputs,outputs,sizeof(*localGpuData->d_outputs)*NO_FITNESS_CASES,cudaMemcpyHostToDevice));

  // allocation of program buffers (GPU and CPU sides)
  localGpuData->progs = new float[MAX_PROGS_SIZE];
  CUDA_SAFE_CALL(cudaMalloc( &localGpuData->d_progs, sizeof(*localGpuData->d_progs)*MAX_PROGS_SIZE));


  
  // Because of the context of each GPU thread, we have to put all user's CUDA 
  // initialisation here if we want to use them in the GPU, otherwise they are
  // not found in the GPU context
  \INSERT_USER_CUDA;
  
  // Wait for population to evaluate
  while(1){
    sem_wait(&localGpuData->sem_in);
    
    if( freeGPU ) {
      // do we need to free gpu memory ?
      cudaFree(localGpuData->d_fitness);
      //cudaFree(localGpuData->d_population);

      cudaFree(localGpuData->d_indexes);
      cudaFree(localGpuData->d_progs);

      delete[] localGpuData->progs;
      delete[] localGpuData->indexes;	break;
    }

    if(nbr_cudaPreliminaryProcess > 0) {
	      
      if( nbr_cudaPreliminaryProcess==2 ) 
	cudaPreliminaryProcessGP(localGpuData);
      else {
	cudaPreliminaryProcessGP(localGpuData);
      }
      nbr_cudaPreliminaryProcess--;
	
      //if( localGpuData->sh_pop_size%localGpuData->num_MP!=0 ){
	//std::cerr << "Warning, population distribution is not optimial, consider adding " << ceilf(((float)localGpuData->sh_pop_size)/localGpuData->num_MP)*localGpuData->num_MP-localGpuData->sh_pop_size  << " individuals to " << (nbr_cudaPreliminaryProcess==2?"parent":"offspring")<<" population" << std::endl;
      //}
    }


    if( nbr_cudaPreliminaryProcess==1 ){ index = flatteningSubPopulation(localGpuData,(IndividualImpl**)EA->population->parents); }
    else{ index = flatteningSubPopulation(localGpuData,(IndividualImpl**)EA->population->offsprings); }

    // transfer the programs to the GPU
    CUDA_SAFE_CALL(cudaMemcpy( localGpuData->d_progs, localGpuData->progs, sizeof(*localGpuData->d_progs)*index, cudaMemcpyHostToDevice ));
    CUDA_SAFE_CALL(cudaMemcpy( localGpuData->d_indexes, localGpuData->indexes, sizeof(*localGpuData->d_indexes)*localGpuData->sh_pop_size, cudaMemcpyHostToDevice ));

	    
    cudaStream_t st;
    cudaStreamCreate(&st);
	    
				      
    // the real GPU computation (kernel launch)
    EvaluatePostFixIndividuals<<<localGpuData->dimGrid,NUMTHREAD,0,st>>>
      ( localGpuData->d_progs,index,localGpuData->sh_pop_size,localGpuData->d_inputs,localGpuData->d_outputs,NO_FITNESS_CASES,localGpuData->d_fitness,localGpuData->d_indexes );
    
    CUDA_SAFE_CALL(cudaStreamSynchronize(st));
    
    // be sure the GPU has finished computing evaluations, and get results to CPU
    lastError = cudaThreadSynchronize();
    if( lastError!=cudaSuccess ){ std::cerr << "Error during synchronize" << std::endl; }
    lastError = cudaMemcpy(localGpuData->fitness, localGpuData->d_fitness, localGpuData->sh_pop_size*sizeof(float), cudaMemcpyDeviceToHost);

    if( nbr_cudaPreliminaryProcess==1 ){
      for( int i=0 ; i<localGpuData->sh_pop_size ; i++ ){
	EA->population->parents[i+localGpuData->indiv_start]->fitness = localGpuData->fitness[i];
	//std::cout << i+localGpuData->indiv_start << ":" << localGpuData->fitness[i] <<std::endl;
      }
    }
    else{
       for( int i=0 ; i<localGpuData->sh_pop_size ; i++ ){
	 EA->population->offsprings[i+localGpuData->indiv_start]->fitness = localGpuData->fitness[i];

	 //float t = ((IndividualImpl*)EA->population->offsprings[i+localGpuData->indiv_start])->evaluate();
	 //std::cout << i+localGpuData->indiv_start << ":" << localGpuData->fitness[i] << " : " << t <<std::endl;
      }
    }
    
    // this thread has finished its phase, so lets tell it to the main thread
    sem_post(&localGpuData->sem_out);
  }
  sem_post(&localGpuData->sem_out);
  fflush(stdout);
  return NULL;
}
				
void wake_up_gpu_thread(){
	for( int i=0 ; i<num_gpus ; i++ ){ sem_post(&(globalGpuData[i].sem_in)); }
	for( int i=0 ; i<num_gpus ; i++ ){ sem_wait(&globalGpuData[i].sem_out); }

}
				
void InitialiseGPUs(){

  // We will use flat inputs data for GPGPU(s)
  float* flatInputs = new float[NO_FITNESS_CASES*VAR_LEN];
  for( int i=0 ; i<NO_FITNESS_CASES ; i++ ){
    memcpy( flatInputs+(i*VAR_LEN),inputs[i],sizeof(float)*VAR_LEN);
  }

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

    globalGpuData[i].flatInputs = flatInputs;
  }
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
  GPNode* mutationPointParent = selectNode(Genome->root, &mutationPointChildId, &mutationPointDepth);
  
  
  if( !mutationPointParent ){
    mutationPointParent = Genome->root;
    mutationPointDepth = 0;
  }
  delete mutationPointParent->children[mutationPointChildId] ;
  mutationPointParent->children[mutationPointChildId] =
    construction_method( VAR_LEN+1, OPCODE_SIZE , 1, TREE_DEPTH_MAX-mutationPointDepth ,0,opArity,OP_ERC);
}

void simpleCrossOver(IndividualImpl& p1, IndividualImpl& p2, IndividualImpl& c){
  int depthP1 = depthOfTree(p1.root);
  int depthP2 = depthOfTree(p2.root);

  int nbNodeP1 = enumTreeNodes(p1.root);
   int nbNodeP2 = enumTreeNodes(p2.root);

  int stockPointChildId=0;
  int graftPointChildId=0;

  bool stockCouldBeTerminal = globalRandomGenerator->tossCoin(0.1);
  bool graftCouldBeTerminal = globalRandomGenerator->tossCoin(0.1);

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
    else Np1 = (int)globalRandomGenerator->random((int)0,(int)nbNodeP1);
    if( nbNodeP2<2 ) Np2=0;
    else Np2 = (int)globalRandomGenerator->random((int)0,(int)nbNodeP2);


    
    if( Np1!=0 ) stockParentNode = pickNthNode(c.root, MIN(Np1,nbNodeP1) ,&stockPointChildId);
    if( Np2!=0 ) graftParentNode = pickNthNode(p2.root, MIN(Np2,nbNodeP1) ,&graftPointChildId);

    // is the stock and the graft an authorized type of node (leaf or inner-node)
    if( Np1 && !stockCouldBeTerminal && opArity[(int)stockParentNode->children[stockPointChildId]->opCode]==0 ) goto choose_node;
    if( Np2 && !graftCouldBeTerminal && opArity[(int)graftParentNode->children[graftPointChildId]->opCode]==0 ) goto choose_node;
    
    if( Np2 && Np1)
      childrenDepth = depthOfNode(c.root,stockParentNode)+depthOfTree(graftParentNode->children[graftPointChildId]);
    else if( Np1 ) childrenDepth = depthOfNode(c.root,stockParentNode)+depthP1;
    else if( Np2 ) childrenDepth = depthOfTree(graftParentNode->children[graftPointChildId]);
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
	\INSERT_INITIALISATION_FUNCTION

	InitialiseGPUs();
}

void EASEAFinal(CPopulation* pop){
	freeGPU=true;
	wake_up_gpu_thread();
        free(globalGpuData);
	
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
  this->valid = true;
  error = sum;
  \INSERT_GENOME_EVAL_FTR    
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


void PopulationImpl::evaluateParentPopulation(){
  static bool dispatchedParents = false;
  
  if( dispatchedParents==false ){
    dispatchPopulation(EA->population->parentPopulationSize);
    dispatchedParents=true;
  }
  
  wake_up_gpu_thread(); 
}

void PopulationImpl::evaluateOffspringPopulation(){
  unsigned actualPopulationSize = this->actualOffspringPopulationSize;
  int index;
  static bool dispatchedOffspring = false;
  
  if( dispatchedOffspring==false ){
    dispatchPopulation(EA->population->offspringPopulationSize);
    dispatchedOffspring=true;
  }
  
  for( index=(actualPopulationSize-1); index>=0; index--)
    ((IndividualImpl*)this->offsprings[index])->copyToCudaBuffer(this->cudaBuffer,index);
  
  wake_up_gpu_thread(); 
  first_generation = false;
}





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
        this->intputFilename = setVariable("inputFile", "EASEA.pop");
        this->plotOutputFilename = (char*)"EASEA.png";

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
  this->population = (CPopulation*)new PopulationImpl(this->params->parentPopulationSize,this->params->offspringPopulationSize, this->params->pCrossover,this->params->pMutation,this->params->pMutationPerGene,this->params->randomGenerator,this->params, this->cstats); // NULL);
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
#include <list>
#include <map>
#include "CGPNode.h"

using namespace std;

class CRandomGenerator;
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
	void copyToCudaBuffer(void* buffer, unsigned id);

	friend std::ostream& operator << (std::ostream& O, const IndividualImpl& B) ;
	void initRandomGenerator(CRandomGenerator* rg){ IndividualImpl::rg = rg;}
};


class ParametersImpl : public Parameters {
public:
	ParametersImpl(std::string const& file, int argc, char* argv[]);
	CEvolutionaryAlgorithm* newEvolutionaryAlgorithm();
};

/**
 * @TODO ces functions devraient s'appeler weierstrassInit, weierstrassFinal etc... (en gros EASEAFinal dans le tpl).
 *
 */

void EASEAInit(int argc, char** argv, ParametersImpl& p);
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
CXXFLAGS+= -std=c++14 -g -Wall -O2 -I$(LIBAESAE)include 
LDFLAGS= $(LIBAESAE)libeasea.a -lpthread 



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
