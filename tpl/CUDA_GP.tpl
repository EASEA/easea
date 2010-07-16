\TEMPLATE_START
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libEasea.lib")
#endif
/**
 This is program entry for TreeGP template for EASEA

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
CIndividual*  bBest = NULL;
float* pEZ_MUT_PROB = NULL;
float* pEZ_XOVER_PROB = NULL;
size_t *EZ_NB_GEN;
size_t *EZ_current_generation;
CEvolutionaryAlgorithm* EA;



\ANALYSE_GP_OPCODE


int main(int argc, char** argv){


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
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libEasea.lib")
#endif

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
#include "CGPNode.h"

float* input_k;
float* output_k;
int* indexes_k;
float* progs_k;
float* results_k;
int* hits_k;

using namespace std;

#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

int fitnessCasesSetLength;


float** inputs;
float* outputs;
int* indexes;
int* hits;
float* results;
float* progs;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;
#define CUDA_GP_TPL
#define GROW_FULL_RATIO 0.5


#define NUMTHREAD2 128
#define MAX_STACK 50
#define LOGNUMTHREAD2 7

#define HIT_LEVEL  0.01f
#define PROBABLY_ZERO  1.11E-15f
#define BIG_NUMBER 1.0E15f


/* Insert declarations about opcodes*/
\INSERT_GP_OPCODE_DECL

float recEvaleDrone(GPNode* root, float* input) {
  float OP1=0, OP2= 0, RESULT = 0;
  if( opArity[root->opCode]>=1) OP1 = recEvaleDrone(root->children[0],input);
  if( opArity[root->opCode]>=2) OP2 = recEvaleDrone(root->children[1],input);
  switch( root->opCode ){
\INSERT_GP_CPU_SWITCH
  default:
    fprintf(stderr,"error unknown terminal opcode %d\n",root->opCode);
    exit(-1);
  }
  return RESULT;
}

__device__ float eval_tree_gpu(unsigned fc_id, const float * k_progs, const float * input){
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


/**
   Send input and output data on the GPU memory.
   Allocate
 */
void initialDataToGPU(float* input_f, int length_input, float* output_f, int length_output){
  // allocate and copy input/output arrays
  CUDA_SAFE_CALL(cudaMalloc((void**)(&input_k),sizeof(float)*length_input));
  CUDA_SAFE_CALL(cudaMalloc((void**)(&output_k),sizeof(float)*length_output));
  CUDA_SAFE_CALL(cudaMemcpy(input_k,input_f,sizeof(float)*length_input,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(output_k,output_f,sizeof(float)*length_output,cudaMemcpyHostToDevice));

  // allocate indexes and programs arrays
  int maxPopSize = MAX(EA->population->parentPopulationSize,EA->population->offspringPopulationSize);
  CUDA_SAFE_CALL(cudaMalloc((void**)&indexes_k,sizeof(*indexes_k)*maxPopSize));
  CUDA_SAFE_CALL( cudaMalloc((void**)&progs_k,sizeof(*progs_k)*MAX_PROGS_SIZE));

  // allocate hits and results arrays
  CUDA_SAFE_CALL(cudaMalloc((void**)&results_k,sizeof(*indexes_k)*maxPopSize));
  CUDA_SAFE_CALL(cudaMalloc((void**)&hits_k,sizeof(*indexes_k)*maxPopSize));
}

/**
   Free gpu memory from the input and ouput arrays.
 */
void free_gpu(){
  cudaFree(input_k);
  cudaFree(output_k);
  cudaFree(progs_k);
  cudaFree(indexes_k);
  cout << "GPU freed" << endl;
}








__global__ static void 
EvaluatePostFixIndividuals_128(const float * k_progs,
			       const int maxprogssize,
			       const int popsize,
			       const float * k_inputs,
			       const float * outputs,
			       const int trainingSetSize,
			       float * k_results,
			       int *k_hits,
			       int* k_indexes
			       )
{
  __shared__ float tmpresult[NUMTHREAD2];
  __shared__ float tmphits[NUMTHREAD2];
  
  const int tid = threadIdx.x; //0 to NUM_THREADS-1
  const int bid = blockIdx.x; // 0 to NUM_BLOCKS-1

  
  int index;   // index of the prog processed by the block 
  float sum = 0.0;
  int hits = 0 ; // hits number

  float currentOutput;
  float ERROR;

  index = bid; // one program per block => block ID = program number
 
  if (index >= popsize) // idle block (should never occur)
    return;
  if (k_progs[index] == -1.0) // already evaluated
    return;

  // Here, it's a busy thread

  sum = 0.0;
  hits = 0 ; // hits number
  
  // Loop on training cases, per cluster of 32 cases (= number of thread)
  // (even if there are only 8 stream processors, we must spawn at least 32 threads) 
  // We loop from 0 to upper bound INCLUDED in case trainingSetSize is not 
  // a multiple of NUMTHREAD

  \INSERT_GENOME_EVAL_HDR

  for (int i=0; i < ((trainingSetSize-1)>>LOGNUMTHREAD2)+1; i++) {
    
    // are we on a busy thread?
    if (i*NUMTHREAD2+tid >= trainingSetSize) // no!
      continue;
    currentOutput = outputs[i*NUMTHREAD2+tid];

    float EVOLVED_VALUE = eval_tree_gpu(i, k_progs+k_indexes[index], k_inputs+(i*NUMTHREAD2+tid));
\INSERT_GENOME_EVAL_BDY

    
    if (!(ERROR < BIG_NUMBER))
      ERROR = BIG_NUMBER;
    else if (ERROR < PROBABLY_ZERO)
      ERROR = 0.0;
    
    if (ERROR <= HIT_LEVEL)
      hits++;
    
    sum += ERROR; // sum raw error on all training cases
    
  } // LOOP ON TRAINING CASES
  
  // gather results from all threads => we need to synchronize
  tmpresult[tid] = sum;
  tmphits[tid] = hits;
  __syncthreads();

  if (tid == 0) {
    for (int i = 1; i < NUMTHREAD2; i++) {
      tmpresult[0] += tmpresult[i];
      tmphits[0] += tmphits[i];
    }    
    ERROR = tmpresult[0];
    \INSERT_GENOME_EVAL_FTR_GPU
    k_results[index] = sqrtf(tmpresult[0]/512);
    k_hits[index] = tmphits[0];
  }  
  // here results and hits have been stored in their respective array: we can leave
}

int flattening_tree_rpn( GPNode* root, float* buf, int* index){
  int i;

  for( i=0 ; i<opArity[(int)root->opCode] ; i++ ){
    flattening_tree_rpn(root->children[i],buf,index);
  }
  if( (*index)+2>MAX_PROGS_SIZE )return 0;
  buf[(*index)++] = root->opCode;
  if( root->opCode == OP_ERC ) buf[(*index)++] = root->erc_value;
  return 1;
}



GPNode* pickNthNode(GPNode* root, int N, int* childId){

  GPNode* stack[TREE_DEPTH_MAX*MAX_ARITY];
  GPNode* parentStack[TREE_DEPTH_MAX*MAX_ARITY];
  int stackPointer = 0;

  parentStack[stackPointer] = NULL;
  stack[stackPointer++] = root;

  for( int i=0 ; i<N ; i++ ){
    GPNode* currentNode = stack[stackPointer-1];
    //cout <<  currentNode << endl;
    stackPointer--;
    for( int j=opArity[currentNode->opCode] ; j>0 ; j--){
      parentStack[stackPointer] = currentNode;
      stack[stackPointer++] = currentNode->children[j-1];
    }
  }

  //assert(stackPointer>0);
  if( stackPointer )
    stackPointer--;

  //cout << "f : \n\t n :" << stack[stackPointer ] << "\n\t p :" << parentStack[stackPointer] << " cId : " << \
  //(*childId) << endl;

  for( int i=0 ; i<opArity[parentStack[stackPointer]->opCode] ; i++ ){
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
  //toDotFile( Genome.root[tree], "out/mutation/p", tree);
  GPNode* mutationPointParent = selectNode(Genome->root, &mutationPointChildId, &mutationPointDepth);
  
  
  if( !mutationPointParent ){
    mutationPointParent = Genome->root;
    mutationPointDepth = 0;
  }
  delete mutationPointParent->children[mutationPointChildId] ;
  mutationPointParent->children[mutationPointChildId] = NULL;
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

  do{
  choose_node:
    /* Np1 = (int)globalRandomGenerator->random((int)0,(int)nbNodeP1); */
    /* Np2 = (int)globalRandomGenerator->random((int)0,(int)nbNodeP2); */

    if( nbNodeP1<2 ) Np1=0;
    else Np1 = (int)globalRandomGenerator->random((int)0,(int)nbNodeP1);
    if( nbNodeP2<2 ) Np2=0;
    else Np2 = (int)globalRandomGenerator->random((int)0,(int)nbNodeP2);

    
    if( Np1!=0 ) stockParentNode = pickNthNode(c.root, MIN(Np1,nbNodeP1) ,&stockPointChildId);
    if( Np2!=0 ) graftParentNode = pickNthNode(p2.root, MIN(Np2,nbNodeP1) ,&graftPointChildId);

    // is the stock and the graft an authorized type of node (leaf or inner-node)
    if( Np1 && !stockCouldBeTerminal && opArity[stockParentNode->children[stockPointChildId]->opCode]==0 ) goto choose_node;
    if( Np2 && !graftCouldBeTerminal && opArity[graftParentNode->children[graftPointChildId]->opCode]==0 ) goto choose_node;
    
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






float IndividualImpl::evaluate(){
  float ERROR;
  float sum = 0;
  \INSERT_GENOME_EVAL_HDR

   for( int i=0 ; i<fitnessCasesSetLength-1 ; i++ ){
     float EVOLVED_VALUE = recEvaleDrone(this->root,inputs[i]);
     \INSERT_GENOME_EVAL_BDY
     sum += ERROR;
   }
  this->valid = true;
  ERROR = sum;
  \INSERT_GENOME_EVAL_FTR    
}



\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

\INSERT_USER_CLASSES

\INSERT_USER_FUNCTIONS

\INSERT_INITIALISATION_FUNCTION
\INSERT_FINALIZATION_FUNCTION

\INSERT_BOUND_CHECKING

void evale_pop_chunk(CIndividual** population, int popSize){
  int index = 0;
  for( int i=0 ; i<popSize ; i++ ){
    indexes[i] = index;
    flattening_tree_rpn( ((IndividualImpl*)population[i])->root, progs, &index);
    progs[index++] = OP_RETURN;
  }

  CUDA_SAFE_CALL(cudaMemcpy( progs_k, progs, sizeof(float)*index, cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL(cudaMemcpy( indexes_k, indexes, sizeof(float)*popSize, cudaMemcpyHostToDevice ));

  cudaStream_t st;
  cudaStreamCreate(&st);

  // Here we will do the real GPU evaluation
  EvaluatePostFixIndividuals_128<<<popSize,128,st>>>( progs_k, index, popSize, input_k, output_k, fitnessCasesSetLength, results_k, hits_k, indexes_k);
  CUDA_SAFE_CALL(cudaStreamSynchronize(st));


  //CUDA_SAFE_CALL(cudaMemcpy( hits, hits_k, sizeof(float)*popSize, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy( results, results_k, sizeof(float)*popSize, cudaMemcpyDeviceToHost));


  for( int i=0 ; i<popSize ; i++ ){
    population[i]->fitness = results[i];
    //population[i]->valid = false;
    //float res = population[i]->evaluate();
    //float err = sqrtf(res-results[i]);
    //if( err>res*0.1 )
    //printf("error in evaluation of %d : %f\n",i,err);
    population[i]->valid = true;
  }
  


  \INSTEAD_EVAL_FUNCTION
}

void EASEAInit(int argc, char** argv){

  int maxPopSize = MAX(EA->population->parentPopulationSize,EA->population->offspringPopulationSize);

  // load data from csv file.
  cout<<"Before everything else function called "<<endl;
  //fitnessCasesSetLength = load_data(&inputs,&outputs,"data_koza_sextic.csv");
  fitnessCasesSetLength = generateData(&inputs,&outputs);
  cout << "number of point in fitness cases set : " << fitnessCasesSetLength << endl;

  float* inputs_f = NULL;

  flattenDatas2D(inputs,fitnessCasesSetLength,VAR_LEN,&inputs_f);

  indexes = new int[maxPopSize];
  hits    = new int[maxPopSize];
  results = new float[maxPopSize];
  progs   = new float[MAX_PROGS_SIZE];
  
  INSTEAD_EVAL_STEP=true;

  initialDataToGPU(inputs_f, fitnessCasesSetLength*VAR_LEN, outputs, fitnessCasesSetLength);



  \INSERT_INIT_FCT_CALL
}

void EASEAFinal(CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;

  // not sure that the population is sorted now. So lets do another time (or check in the code;))
  // and dump the best individual in a graphviz file.
  //population->sortParentPopulation();
  //toDotFile( ((IndividualImpl*)population->parents[0])->root[0], "best-of-run",0);
  
  // delete some global arrays
  delete[] indexes; delete[] hits;
  delete[] results; delete[] progs;
  
  //free_gpu();
  free_data();
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
}

CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

IndividualImpl::~IndividualImpl(){
  \GENOME_DTOR
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


size_t IndividualImpl::mutate( float pMutationPerGene ){
  this->valid=false;


  // ********************
  // Problem specific part
  \INSERT_MUTATOR
}




void ParametersImpl::setDefaultParameters(int argc, char** argv){

	seed = setVariable("seed",(int)time(0));
	globalRandomGenerator = new CRandomGenerator(seed);
	this->randomGenerator = globalRandomGenerator;


	this->minimizing = \MINIMAXI;
	this->nbGen = setVariable("nbGen",(int)\NB_GEN);

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

	cout << "Parent red " << parentReduction << " " << parentReductionSize << "/"<< parentPopulationSize << endl;
	cout << "Parent red " << offspringReduction << " " << offspringReductionSize << "/" << offspringPopulationSize << endl;

	generationalCriterion = new CGenerationalCriterion(setVariable("nbGen",(int)\NB_GEN));
	controlCStopingCriterion = new CControlCStopingCriterion();
	timeCriterion = new CTimeCriterion(setVariable("timeLimit",\TIME_LIMIT));
	
	this->optimise = 0;


	this->printStats = setVariable("printStats",\PRINT_STATS);
	this->generateCSVFile = setVariable("generateCSVFile",\GENERATE_CSV_FILE);
	this->generateGnuplotScript = setVariable("generateGnuplotScript",\GENERATE_GNUPLOT_SCRIPT);
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

void EvolutionaryAlgorithmImpl::initializeParentPopulation(){
	for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
		this->population->addIndividualParentPopulation(new IndividualImpl());
	}
}


EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){
	;
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){
  
}

\START_CUDA_GENOME_H_TPL

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

\INSERT_GP_PARAMETERS

//#include "CRandomGenerator.h"
#include <stdlib.h>
#include <iostream>
#include <CIndividual.h>
#include <Parameters.h>
#include <CGPNode.h>
class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;

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

#endif /* PROBLEM_DEP_H */

\START_CUDA_MAKEFILE_TPL

NVCC=nvcc
EASEALIB_PATH=\EZ_PATHlibeasea/#/home/kruger/Bureau/Easea/libeasea/

CXXFLAGS =  -g  -I$(EASEALIB_PATH)include

\INSERT_MAKEFILE_OPTION#END OF USER MAKEFILE OPTIONS


OBJS = EASEA.o EASEAIndividual.o 

LIBS = -lboost_program_options 

TARGET =	EASEA

$(TARGET):	$(OBJS)
	$(NVCC) -o $(TARGET) $(OBJS) $(LIBS) -g $(EASEALIB_PATH)libeasea.a

	
%.o:%.cu
	$(NVCC) -c $(CXXFLAGS) $^ $(NVCC_OPT)

all:	$(TARGET)
clean:
	rm -f $(OBJS) $(TARGET)
easeaclean:
	rm -f $(TARGET) *.o *.cpp *.hpp EASEA.png EASEA.dat EASEA.prm EASEA.mak Makefile EASEA.vcproj EASEA.csv EASEA.r EASEA.plot
	
\START_VISUAL_TPL<?xml version="1.0" encoding="Windows-1252"?>
<VisualStudioProject
	ProjectType="Visual C++"
	Version="9,00"
	Name="EASEA"
	ProjectGUID="{E73D5A89-F262-4F0E-A876-3CF86175BC30}"
	RootNamespace="EASEA"
	Keyword="Win32Proj"
	TargetFrameworkVersion="196613"
	>
	<Platforms>
		<Platform
			Name="Win32"
		/>
	</Platforms>
	<ToolFiles>
	</ToolFiles>
	<Configurations>
		<Configuration
			Name="Release|Win32"
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
				RuntimeLibrary="2"
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
				RelativePath=".\EASEAIndividual.cpp"
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
--plotStats=\PLOT_STATS #plot Stats with gnuplot (requires Gnuplot)
--printInitialPopulation=0 #Print initial population
--printFinalPopulation=0 #Print final population
--generateCSV=\GENERATE_CSV_FILE
--generateGnuplotScript=\GENERATE_GNUPLOT_SCRIPT
--generateRScript=\GENERATE_R_SCRIPT
\TEMPLATE_END
