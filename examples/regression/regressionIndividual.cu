
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

using namespace std;

extern "C"
__global__ void 
EvaluatePostFixIndividuals( const float * k_progs, const int maxprogssize,  const int popsize, const float * k_inputs, const float * outputs, const int trainingSetSize, float * k_results,  int* k_indexes );


#include "regressionIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm *EA;

#define CUDAGP_TPL

#define HIT_LEVEL  0.01f
#define PROBABLY_ZERO  1.11E-15f
#define BIG_NUMBER 1.0E15f


unsigned aborded_crossover;
float** inputs;
float* outputs;


struct gpuEvaluationData* gpuData;

int fstGpu = 0;
int lstGpu = 0;


struct gpuEvaluationData* globalGpuData;
float* fitnessTemp;  
bool freeGPU = false;
bool first_generation = true;
int num_gpus = 0;       // number of CUDA GPUs

PopulationImpl* Pop = NULL;

// User declarations
#line 17 "regression.ez"


// these 3 defines are mandatory here. Adjust as you like.
#define NO_FITNESS_CASES 1024
#define VAR_LEN 1
#define GROW_FULL_RATIO 0.5

// this is the number of learning cases computed in parallel.
// note that on 1024 is the maximum size on fermi architectures 512 on older cards.
#define NUMTHREAD 1024
#define MAX_STACK 15


#define PI (3.141592653589793)



#define TREE_DEPTH_MAX 8
#define INIT_TREE_DEPTH_MAX 4
#define INIT_TREE_DEPTH_MIN 2
#define MAX_PROGS_SIZE 100000000
#define NB_GPU 0


/* Insert declarations about opcodes*/
enum OPCODE              {OP_X,OP_ERC,OP_ADD,OP_SUB,OP_MUL,OP_DIV,OPCODE_SIZE, OP_RETURN};
const char* opCodeName[]={"x","ERC","+","-","*","/"};
unsigned opArity[]=     {0,0,2,2,2,2};





GPNode* ramped_hh(){
  return RAMPED_H_H(INIT_TREE_DEPTH_MIN,INIT_TREE_DEPTH_MAX,EA->population->actualParentPopulationSize,EA->population->parentPopulationSize,0, VAR_LEN, OPCODE_SIZE,opArity, OP_ERC);
}

std::string toString(GPNode* root){
  return toString(root,opArity,opCodeName,OP_ERC);
}


// User classes

// User functions

#line 34 "regression.ez"

#define POLY(x) x*x*x-3*x*x+x
/**
   This function generates data NO_FITNESS_CASES fitness cases,
   from the polynome POLY(X) with X globalRandomGenerator->randomly picked between (-1,1)

   @inputs address of the inputs array. (array will be allocated here)
   @outputs adddress of the outputs array. (array will be allocated here)

   @ret number of loaded fitness cases (should be equal to NO_FITNESS_CASES).   
 */
int generateData(float*** inputs, float** outputs){
  int i=0;

  (*inputs) = new float*[NO_FITNESS_CASES];
  (*outputs) = new float[NO_FITNESS_CASES];
  
  for( i=0 ; i<NO_FITNESS_CASES ; i++ ){
    (*inputs)[i]=new float[VAR_LEN];
    float x = globalRandomGenerator->random(-10.,+10.);
    (*inputs)[i][0] = x;
    (*outputs)[i] = POLY(x);
  }

  return NO_FITNESS_CASES;
}


void free_data(){
  for( int i=0 ; i<NO_FITNESS_CASES ;i++ ) delete[] inputs[i] ;

  delete[] outputs;
  delete[] inputs;
} 


// Initialisation function
void EASEAInitFunction(int argc, char *argv[]){
#line 71 "regression.ez"

{
  generateData(&inputs,&outputs);
}
}

// Finalization function
void EASEAFinalization(CPopulation* population){
#line 77 "regression.ez"

{
  std::cout << toString(((IndividualImpl*)EA->population->Best)->root) << std::endl;

  free_data();
}
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
      globalGpuData[index].sh_pop_size = ceil((float)populationSize * (((float)globalGpuData[index].num_MP) / (float)noTotalMP) );
    
    }
    //On the last card we are going to place the remaining individuals.  
    else 
      globalGpuData[index].sh_pop_size = populationSize - count;
	     
    count += globalGpuData[index].sh_pop_size;	     
  }
}

void cudaPreliminaryProcessGP(struct gpuEvaluationData* localGpuData){

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
  case OP_X :
    RESULT=input[0];
    break;
  case OP_ERC :
    RESULT=root->erc_value;
    break;
  case OP_ADD :
    RESULT=OP1+OP2;
    break;
  case OP_SUB :
    RESULT=OP1-OP2;
    break;
  case OP_MUL :
    RESULT=OP1*OP2;
    break;
  case OP_DIV :
    
    if( !OP2 ) RESULT = 1;
    else RESULT = OP1/OP2;
  
    break;

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

    case OP_X :
      RESULT=input[0];
      stack[sp++] = RESULT;
      break;
    case OP_ERC :
      RESULT=k_progs[start_prog++];;
      stack[sp++] = RESULT;
      break;
    case OP_ADD :
      OP2 = stack[--sp];
      OP1 = stack[--sp];
      RESULT=OP1+OP2;
      stack[sp++] = RESULT;
      break;
    case OP_SUB :
      OP2 = stack[--sp];
      OP1 = stack[--sp];
      RESULT=OP1-OP2;
      stack[sp++] = RESULT;
      break;
    case OP_MUL :
      OP2 = stack[--sp];
      OP1 = stack[--sp];
      RESULT=OP1*OP2;
      stack[sp++] = RESULT;
      break;
    case OP_DIV :
      OP2 = stack[--sp];
      OP1 = stack[--sp];
      
      if( !OP2 ) RESULT = 1;
      else RESULT = OP1/OP2;
    
      stack[sp++] = RESULT;
      break;

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
    float ERROR;

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

      
  ;
    
    for (int i=tid; i < trainingSetSize ; i+=NUMTHREAD) {
    
      // are we on a busy thread?
      if (i >= trainingSetSize) // no!
	continue;
         
      float EVOLVED_VALUE = eval_tree_gpu( k_progs+k_indexes[index], k_inputs+i*VAR_LEN);
 
            
      float expected_value = outputs[i];
      ERROR = (expected_value-EVOLVED_VALUE)*(expected_value-EVOLVED_VALUE);
      ;

    
      if (!(ERROR < BIG_NUMBER)) ERROR = BIG_NUMBER;
      else if (ERROR < PROBABLY_ZERO) ERROR = 0.0;
    
    
      sum += ERROR; // sum raw error on all training cases
    
    } // LOOP ON TRAINING CASES
  
    // gather results from all threads => we need to synchronize
    tmpresult[tid] = sum;

    __syncthreads();

    if (tid == 0) {
      for (int i = 1; i < NUMTHREAD; i++) {
	tmpresult[0] += tmpresult[i];
      }    
      ERROR = tmpresult[0];
        
  k_results[index] =sqrtf(ERROR);
  ;
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


int flatteningSubPopulation( struct gpuEvaluationData* localGpuData, IndividualImpl** population){
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
  struct gpuEvaluationData* localGpuData = (struct gpuEvaluationData*)arg;

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
  ;
  
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
	
      if( localGpuData->sh_pop_size%localGpuData->num_MP!=0 ){

	std::cerr << "Warning, population distribution is not optimial, consider adding " << ceilf(((float)localGpuData->sh_pop_size)/localGpuData->num_MP)*localGpuData->num_MP-localGpuData->sh_pop_size  << " individuals to " << (nbr_cudaPreliminaryProcess==2?"parent":"offspring")<<" population" << std::endl;
      }
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

void evale_pop_chunk(CIndividual** population, int popSize){
  
// No Instead evaluation step function.

}

void regressionInit(int argc, char** argv){
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
	
  EASEAInitFunction(argc, argv);


	InitialiseGPUs();
}

void regressionFinal(CPopulation* pop){
	freeGPU=true;
	wake_up_gpu_thread();
        free(globalGpuData);
	
	
  EASEAFinalization(pop);
;
}

void EASEABeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	#line 207 "regression.ez"
{
#line 85 "regression.ez"

{
  //cout << "At the beginning of each generation function called" << endl;
}		    
}
}

void EASEAEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	{

{		 
  //cout << "At the end of each generation function called" << endl;
}
}
}

void EASEAGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
        {

 //cout << "At each generation before replacement function called" << endl;
}
}


IndividualImpl::IndividualImpl() : CIndividual() {
      root=NULL;
 
  // Genome Initialiser
#line 113 "regression.ez"

{
  (*this).root = ramped_hh();
}

  valid = false;
  isImmigrant = false;
}

CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

IndividualImpl::~IndividualImpl(){
  // Destructing pointers
  if (root) delete root;
  root=NULL;

}


float IndividualImpl::evaluate(){
  float ERROR; 
 float sum = 0;
    
  

   for( int i=0 ; i<NO_FITNESS_CASES ; i++ ){
     float EVOLVED_VALUE = recEval(this->root,inputs[i]);
           
      float expected_value = outputs[i];
      ERROR = (expected_value-EVOLVED_VALUE)*(expected_value-EVOLVED_VALUE);
      
     sum += ERROR;
   }
  this->valid = true;
  ERROR = sum;
    
  return fitness=sqrtf(ERROR);
      
}

void IndividualImpl::boundChecking(){
        
// No Bound checking function.

}


string IndividualImpl::serialize(){
    ostringstream EASEA_Line(ios_base::app);
    // Memberwise serialization
	// Serialize function for "GPNode"
	//cout << "Now serializing individual " << toString(this->root) << endl;
	// build map used to associate GPNode pointers to indexes
	map<GPNode*,int> indexes;

	// breadth-first visit of the tree
	int currentIndex = 0;
	list<GPNode*> nodesToVisit;
	nodesToVisit.push_back(this->root);

	while(nodesToVisit.size() != 0)
	{
	 // remove current node from nodes to visit
	 GPNode* currentNode = nodesToVisit.front();
	 nodesToVisit.pop_front();
	
	 // add children of current node (by default, the tree is binary)
	 if( currentNode->children[0] != NULL ) nodesToVisit.push_back( currentNode->children[0] ); 
	 if( currentNode->children[1] != NULL ) nodesToVisit.push_back( currentNode->children[1] ); 
	 // assign code to current node       
	 indexes[currentNode] = currentIndex; 
	 currentIndex++;
	}
	// the very first item in the line is the number of nodes in the tree 
	EASEA_Line << currentIndex << " ";
	// another visit to finally serialize the nodes 
	vector<double> ercValues;          
	nodesToVisit.push_back(this->root);
	while(nodesToVisit.size() != 0)                     
	{                                                   
	 // remove current node from nodes to visit  
	 GPNode* currentNode = nodesToVisit.front(); 
	 nodesToVisit.pop_front();	
	 // add children of current node (hoping it's binary)                                                       
	 if( currentNode->children[0] != NULL ) nodesToVisit.push_back( currentNode->children[0] );                 
	 if( currentNode->children[1] != NULL ) nodesToVisit.push_back( currentNode->children[1] );                 
	                                                                                                           
	 // node to string: format is <index> <var_id> <opCode> <indexOfChild1> <indexOfChild2>                     
	 EASEA_Line << indexes[currentNode] << " " << currentNode->var_id << " " << (int)currentNode->opCode << " ";
	 // if the children are not NULL, put their index; otherwise, put "0"              
	 if( currentNode->children[0] != NULL )                                            
	  EASEA_Line << indexes[ currentNode->children[0] ] << " ";                 
	 else                                                                              
	  EASEA_Line << "0 ";                                                       
	                                                                                  
	 if( currentNode->children[1] != NULL )                                            
	  EASEA_Line << indexes[ currentNode->children[1] ] << " ";                 
	 else             
	  EASEA_Line << "0 ";                                                       
	                 
	 // if the node is an ERC, the floating point value is stored for later            
	 if( currentNode->opCode == OP_ERC ) ercValues.push_back( currentNode->erc_value );
	} 
	// finally, put all the floating point ERC values             
	for(unsigned int i = 0; i < ercValues.size(); i++)            
	 EASEA_Line << ercValues[i] << " ";                    
	                                                              
	// debug                                                      
	//cout << "EASEA_Line: " << EASEA_Line.str() << endl; 

    EASEA_Line << this->fitness;
    return EASEA_Line.str();
}

void IndividualImpl::deserialize(string Line){
    istringstream EASEA_Line(Line);
    string line;
    // Memberwise deserialization
	// debug
	//cout << "Reading received individual..." << endl;
	//cout << Line << endl;
	
	// first, read number of nodes
	int numberOfNodes; 
	EASEA_Line >> numberOfNodes;
	// debug
	//cout << "The received individual has " << numberOfNodes << " nodes." << endl; 
	
	// iterate through the line, creating a map of <index> <GPNode*> <indexOfChild1> <indexOfChild2> 
	map< int, pair< GPNode*,vector<int> > > nodeMap;                              
	for(int n = 0; n < numberOfNodes; n++)                                        
	{                                                                             
	 int index, temp;                                                      
	 int opCode;                                                           
	 int var_id;                                                           
	 vector<int> childrenIndexes;                                          
	  
	 // format is <index> <var_id> <opCode> <indexOfChild1> <indexOfChild2>
	 EASEA_Line >> index;             
	 EASEA_Line >> var_id;            
	 EASEA_Line >> opCode;            
	 EASEA_Line >> temp;              
	 childrenIndexes.push_back(temp); 
	 EASEA_Line >> temp;              
	 childrenIndexes.push_back(temp); 
	
	 // create GPNode                   
	 GPNode* currentNode = new GPNode();
	 currentNode->var_id = var_id;
	 currentNode->opCode = opCode;
	
	 // debug 
	 //cout 	<< "Read node: " << index << " " << var_id << " " << opCode << " " 
	 //<< childrenIndexes[0] << " " << childrenIndexes[1] << endl;
	
	 // put everything into the map                                        
	 pair< GPNode*, vector<int> > tempPair (currentNode, childrenIndexes); 
	 nodeMap[index] = tempPair;
	}
	 
	// rebuild the individual structure 
	for(int n = 0; n < numberOfNodes; n++) 
	{ 
	 // now, rebuild the individual by adding the pointers to the children                                  
	 if( nodeMap[n].second[0] != 0 ) nodeMap[n].first->children[0] = nodeMap[ nodeMap[n].second[0] ].first; 
	 if( nodeMap[n].second[1] != 0 ) nodeMap[n].first->children[1] = nodeMap[ nodeMap[n].second[1] ].first; 
	
	 // also, if the opCode of the node is the same as the OP_ERC, find the
	 // corresponding real value at the end of the EASEA_Line
	 if( nodeMap[n].first->opCode == OP_ERC )
	 {                                          
	                                           
	  double temp;                       
	  EASEA_Line >> temp;                
	  nodeMap[n].first->erc_value = temp;
	// debug
	//cout << "-- Found ERC variable! Read value " << temp << " from the end of EASEA_Line." << endl;
	 }
	}
	
	// link the tree to the current individual 
	this->root = nodeMap[0].first;
	
	// debug 
	//cout << "Individual received: " << toString(this->root) << endl;

    EASEA_Line >> this->fitness;
    this->valid=true;
    this->isImmigrant=false;
}

IndividualImpl::IndividualImpl(const IndividualImpl& genome){

  // ********************
  // Problem specific part
  // Memberwise copy
    root=(genome.root ? new GPNode(*(genome.root)) : NULL);



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
  	#line 119 "regression.ez"

{
  simpleCrossOver(parent1,parent2,child);
  child.valid = false;
}



	child.valid = false;
	/*   cout << "child : " << child << endl; */
	return new IndividualImpl(child);
}


void IndividualImpl::printOn(std::ostream& os) const{
	


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


unsigned IndividualImpl::mutate( float pMutationPerGene ){
  this->valid=false;


  // ********************
  // Problem specific part
  #line 126 "regression.ez"
 // Must return the number of mutations
{
  simple_mutator(&(*this));

  return  1>0?true:false;
}

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





void ParametersImpl::setDefaultParameters(int argc, char** argv){
        this->minimizing = true;
        this->nbGen = setVariable("nbGen",(int)50);

        seed = setVariable("seed",(int)time(0));
        globalRandomGenerator = new CRandomGenerator(seed);
        this->randomGenerator = globalRandomGenerator;

        selectionOperator = getSelectionOperator(setVariable("selectionOperator","Tournament"), this->minimizing, globalRandomGenerator);
        replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator","Tournament"),this->minimizing, globalRandomGenerator);
        parentReductionOperator = getSelectionOperator(setVariable("reduceParentsOperator","Tournament"),this->minimizing, globalRandomGenerator);
        offspringReductionOperator = getSelectionOperator(setVariable("reduceOffspringOperator","Tournament"),this->minimizing, globalRandomGenerator);
        selectionPressure = setVariable("selectionPressure",(float)7.000000);
        replacementPressure = setVariable("reduceFinalPressure",(float)7.000000);
        parentReductionPressure = setVariable("reduceParentsPressure",(float)2.000000);
        offspringReductionPressure = setVariable("reduceOffspringPressure",(float)2.000000);
        pCrossover = 0.900000;
        pMutation = 0.100000;
        pMutationPerGene = 0.05;

        parentPopulationSize = setVariable("popSize",(int)5000);
        offspringPopulationSize = setVariable("nbOffspring",(int)5000);


        parentReductionSize = setReductionSizes(parentPopulationSize, setVariable("survivingParents",(float)1.000000));
        offspringReductionSize = setReductionSizes(offspringPopulationSize, setVariable("survivingOffspring",(float)1.000000));

        this->elitSize = setVariable("elite",(int)1);
        this->strongElitism = setVariable("eliteType",(int)1);

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

        generationalCriterion = new CGenerationalCriterion(setVariable("nbGen",(int)50));
        controlCStopingCriterion = new CControlCStopingCriterion();
        timeCriterion = new CTimeCriterion(setVariable("timeLimit",0));

	this->optimise=0;

        this->printStats = setVariable("printStats",1);
        this->generateCSVFile = setVariable("generateCSVFile",0);
        this->generatePlotScript = setVariable("generatePlotScript",0);
        this->generateRScript = setVariable("generateRScript",0);
        this->plotStats = setVariable("plotStats",1);
	this->printInitialPopulation = setVariable("printInitialPopulation",0);
	this->printFinalPopulation = setVariable("printFinalPopulation",0);
	this->savePopulation = setVariable("savePopulation",0);
	this->startFromFile = setVariable("startFromFile",0);

        this->outputFilename = (char*)"regression";
        this->plotOutputFilename = (char*)"regression.png";

	this->remoteIslandModel = setVariable("remoteIslandModel",0);
//    	this->ipFile = (char*)setVariable("ipFile","NULL").c_str();
	std::string *ipFilename = new std::string();
	*ipFilename = setVariable("ipFile", "NULL");
	this->ipFile = (char *)ipFilename->c_str();
	this->migrationProbability = setVariable("migrationProbability",(float)0.000000);
	this->serverPort = setVariable("serverPort",0);

}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	EZ_NB_GEN = (unsigned*)setVariable("nbGen",50);
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
          ifstream EASEA_File("regression.pop");
          string EASEA_Line;
          for( index=(Size-1); index>=0; index--) {
             getline(EASEA_File, EASEA_Line);
            this->population->addIndividualParentPopulation(new IndividualImpl(),index);
            ((IndividualImpl*)this->population->parents[index])->deserialize(EASEA_Line);
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


