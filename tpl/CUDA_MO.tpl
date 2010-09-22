\TEMPLATE_START/**
 This is program entry for multi-objective 
 CUDA template for EASEA

*/
\ANALYSE_PARAMETERS
using namespace std;
#include <iostream>
#include "EASEATools.hpp"
#include "EASEAIndividual.hpp"
#include <time.h>

RandomGenerator* globalRandomGenerator;


int main(int argc, char** argv){


  parseArguments("EASEA.prm",argc,argv);

  size_t parentPopulationSize = setVariable("popSize",\POP_SIZE);
  size_t offspringPopulationSize = setVariable("nbOffspring",\OFF_SIZE);
  float pCrossover = \XOVER_PROB;
  float pMutation = \MUT_PROB;
  float pMutationPerGene = 0.05;

  time_t seed = setVariable("seed",time(0));
  globalRandomGenerator = new RandomGenerator(seed);

  std::cout << "Seed is : " << seed << std::endl;

  SelectionOperator* selectionOperator = new \SELECTOR;
  SelectionOperator* replacementOperator = new \RED_FINAL;
  float selectionPressure = \SELECT_PRM;
  float replacementPressure = \RED_FINAL_PRM;
  string outputfile = setVariable("outputfile","");
  string inputfile = setVariable("inputfile","");

  EASEAInit(argc,argv);
    
  EvolutionaryAlgorithm ea(parentPopulationSize,offspringPopulationSize,selectionPressure,replacementPressure,
			   selectionOperator,replacementOperator,pCrossover, pMutation, pMutationPerGene,outputfile,inputfile);

  StoppingCriterion* sc = new GenerationalCriterion(&ea,setVariable("nbGen",\NB_GEN));
  ea.addStoppingCriterion(sc);
  Population* pop = ea.getPopulation();


  ea.runEvolutionaryLoop();

  EASEAFinal(pop);

  delete pop;
  delete sc;
  delete selectionOperator;
  delete replacementOperator;
  delete globalRandomGenerator;


  return 0;
}


\START_CUDA_GENOME_CU_TPL
#include "EASEAIndividual.hpp"
#include "EASEAUserClasses.hpp"
#include <string.h>
#include <fstream>
#include <sys/time.h>
#include "EASEATools.hpp"

#define CUDA_TPL

extern RandomGenerator* globalRandomGenerator;

\INSERT_USER_DECLARATIONS
struct gpuOptions initOpts;

\ANALYSE_USER_CLASSES

\INSERT_USER_FUNCTIONS

\INSERT_INITIALISATION_FUNCTION
\INSERT_FINALIZATION_FUNCTION
\INSERT_GENERATION_FUNCTION
\INSERT_BOUND_CHECKING



void EASEAFinal(Population* pop){
  \INSERT_FINALIZATION_FCT_CALL
}

void EASEAInit(int argc, char** argv){
  \INSERT_INIT_FCT_CALL
}


using namespace std;

RandomGenerator* Individual::rg;

Individual::Individual(){
  \GENOME_CTOR 
  \INSERT_EO_INITIALISER
  valid = false;
}


Individual::~Individual(){
  \GENOME_DTOR
}


float Individual::evaluate(){
  if(valid)
    return fitness;
  else{
    valid = true;
    \INSERT_EVALUATOR
  } 
}


/**
   This function allows to acces to the Individual stored in cudaBuffer as a standard
   individual.
   @TODO This should be a macro, at this time it is a function for debuging purpose
*/
__device__ __host__ inline Individual* INDIVIDUAL_ACCESS(void* buffer,size_t id){
  return ((Individual*)(((char*)buffer)+(\GENOME_SIZE+sizeof(void*))*id));
}


__device__ float cudaEvaluate(void* devBuffer, size_t id, struct gpuOptions initOpts, float f[2]){
  \INSERT_CUDA_EVALUATOR
}




inline void Individual::copyToCudaBuffer(void* buffer, size_t id){
  
/*   DEBUG_PRT("%p\n",(char*)this+sizeof(Individual*)); */
/*   DEBUG_PRT("%p\n",&this->sigma); */
/*   DEBUG_PRT("%lu\n",id); */
  
  memcpy(((char*)buffer)+(\GENOME_SIZE+sizeof(Individual*))*id,((char*)this),\GENOME_SIZE+sizeof(Individual*));
  
}

Individual::Individual(const Individual& genome){

  // ********************
  // Problem specific part
  \COPY_CTOR
  
  // ********************
  // Generic part
  this->valid = genome.valid;
  this->fitness = genome.fitness;
}


Individual* Individual::crossover(Individual** ps){
  // ********************
  // Generic part  
  Individual parent1(*this);
  Individual parent2(*ps[0]);
  Individual child(*this);

  //DEBUG_PRT("Xover");
/*   cout << "p1 : " << parent1 << endl; */
/*   cout << "p2 : " << parent2 << endl; */

  // ********************
  // Problem specific part
  \INSERT_CROSSOVER

    child.valid = false;
/*   cout << "child : " << child << endl; */
  return new Individual(child);
}


void Individual::printOn(std::ostream& os) const{
  \INSERT_DISPLAY
}

std::ostream& operator << (std::ostream& O, const Individual& B) 
{ 
  // ********************
  // Problem specific part
  O << "\nIndividual : "<< std::endl;
  O << "\t\t\t";
  B.printOn(O);
    
  if( B.valid ) O << "\t\t\tfitness : " << B.fitness;
  else O << "fitness is not yet computed" << std::endl;
  return O; 
} 


size_t Individual::mutate( float pMutationPerGene ){
  this->valid=false;
  // ********************
  // Problem specific part
  \INSERT_MUTATOR  
}


size_t Individual::sizeOfGenome=\GENOME_SIZE;

/* ****************************************
   EvolutionaryAlgorithm class
****************************************/

/**
   @DEPRECATED This contructor will be deleted. It was for test only, because it
   is too much constrained (default selection/replacement operator)
 */
EvolutionaryAlgorithm::EvolutionaryAlgorithm( size_t parentPopulationSize,
					      size_t offspringPopulationSize,
					      float selectionPressure, float replacementPressure,
					      float pCrossover, float pMutation, 
					      float pMutationPerGene){
  RandomGenerator* rg = globalRandomGenerator;


  SelectionOperator* so = new MaxTournament(rg);
  SelectionOperator* ro = new MaxTournament(rg);
  
  Individual::initRandomGenerator(rg);
  Population::initPopulation(so,ro,selectionPressure,replacementPressure);
  
  this->population = new Population(parentPopulationSize,offspringPopulationSize,
				    pCrossover,pMutation,pMutationPerGene,rg);

  this->currentGeneration = 0;

  this->reduceParents = 0;
  this->reduceOffsprings = 0;


}

EvolutionaryAlgorithm::EvolutionaryAlgorithm( size_t parentPopulationSize,
					      size_t offspringPopulationSize,
					      float selectionPressure, float replacementPressure,
					      SelectionOperator* selectionOperator, SelectionOperator* replacementOperator,
					      float pCrossover, float pMutation, 
					      float pMutationPerGene, string& outputfile, string& inputfile){

  RandomGenerator* rg = globalRandomGenerator;

  SelectionOperator* so = selectionOperator;
  SelectionOperator* ro = replacementOperator;
  
  Individual::initRandomGenerator(rg);
  Population::initPopulation(so,ro,selectionPressure,replacementPressure);
  
  this->population = new Population(parentPopulationSize,offspringPopulationSize,
				    pCrossover,pMutation,pMutationPerGene,rg);

  this->currentGeneration = 0;

  this->reduceParents = 0;
  this->reduceOffsprings = 0;

  if( outputfile.length() )
    this->outputfile = new string(outputfile);
  else
    this->outputfile = NULL;

  if( inputfile.length() )
    this->inputfile = new std::string(inputfile);
  else
    this->inputfile = NULL;
  


}

// do the repartition of data accross threads
__global__ void 
cudaEvaluatePopulation(void* d_population, size_t popSize, float* d_fitnesses, struct gpuOptions initOpts){

  size_t id = (blockDim.x*blockIdx.x)+threadIdx.x;  // id of the individual computed by this thread

  // escaping for the last block
  if(blockIdx.x == (gridDim.x-1)) if( id >= popSize ) return;

  //void* indiv = ((char*)d_population)+id*(\GENOME_SIZE+sizeof(Individual*)); // compute the offset of the current individual
  cudaEvaluate(d_population,id,initOpts,d_fitnesses+2*id);
}


#define NB_MP 16
inline size_t
partieEntiereSup(float E){
  int fl = floor(E);
  if( fl == E )
    return E;
  else
    return floor(E)+1;
}

inline int 
puissanceDeuxSup(float n){
  int tmp = 2;
  while(tmp<n)tmp*=2;
  return tmp;
}



bool
repartition(size_t popSize, size_t* nbBlock, size_t* nbThreadPB, size_t* nbThreadLB, 
	    size_t nbMP, size_t maxBlockSize){
  
  (*nbThreadLB) = 0;
  
  DEBUG_PRT("repartition : %d",popSize);
  
  if( ((float)popSize / (float)nbMP) <= maxBlockSize ){
    //la population repartie sur les MP tient dans une bloc par MP
    (*nbThreadPB) = partieEntiereSup( (float)popSize/(float)nbMP);
    (*nbBlock) = popSize/(*nbThreadPB);
    if( popSize%nbMP != 0 ){
      //on fait MP-1 block de equivalent et un plus petit
      (*nbThreadLB) = popSize - (*nbThreadPB)*(*nbBlock);
    }
  }
  else{
    //la population est trop grande pour etre repartie sur les MP
    //directement
    //(*nbBlock) = partieEntiereSup( (float)popSize/((float)maxBlockSize*NB_MP));
    (*nbBlock) = puissanceDeuxSup( (float)popSize/((float)maxBlockSize*NB_MP));
    (*nbBlock) *= NB_MP;
    (*nbThreadPB) = popSize/(*nbBlock);
    if( popSize%maxBlockSize!=0){
      (*nbThreadLB) = popSize - (*nbThreadPB)*(*nbBlock);
      
      // Le rest est trop grand pour etre place dans un seul block (c'est possible uniquement qd 
      // le nombre de block depasse maxBlockSize 
      while( (*nbThreadLB) > maxBlockSize ){
	//on augmente le nombre de blocs principaux jusqu'a ce que nbthreadLB retombe en dessous de maxBlockSize
	//(*nbBlock) += nbMP;
	(*nbBlock) *= 2;
 	(*nbThreadPB) = popSize/(*nbBlock);
	(*nbThreadLB) = popSize - (*nbThreadPB)*(*nbBlock);
      }
    }
  }
  
  if((((*nbBlock)*(*nbThreadPB) + (*nbThreadLB))  == popSize) 
     && ((*nbThreadLB) <= maxBlockSize) && ((*nbThreadPB) <= maxBlockSize))
    return true;
  else 
    return false;
}


/**
   Allocate buffer for populationSize individuals and fitnesses
   compute the repartition
 */
void EvolutionaryAlgorithm::cudaPreliminaryProcess(size_t populationSize, dim3* dimBlock, dim3* dimGrid, void** allocatedDeviceBuffer,
						   float** deviceFitness){

  size_t nbThreadPB, nbThreadLB, nbBlock;
  cudaError_t lastError;

  lastError = cudaMalloc(allocatedDeviceBuffer,populationSize*(\GENOME_SIZE+sizeof(Individual*)));
  DEBUG_PRT("Population buffer allocation : %s",cudaGetErrorString(lastError));
  lastError = cudaMalloc(((void**)deviceFitness),NB_OBJECTIVE*populationSize*sizeof(float));
  DEBUG_PRT("Fitness buffer allocation : %s",cudaGetErrorString(lastError));
  
  if( !repartition(populationSize, &nbBlock, &nbThreadPB, &nbThreadLB,30, 240))
    exit( -1 );

  DEBUG_PRT("repartition is \n\tnbBlock %lu \n\tnbThreadPB %lu \n\tnbThreadLD %lu",nbBlock,nbThreadPB,nbThreadLB); 

  if( nbThreadLB!=0 )
    dimGrid->x = (nbBlock+1);
  else
    dimGrid->x = (nbBlock);

  dimBlock->x = nbThreadPB;
  cout << "Number of grid : " << dimGrid->x << endl;
  cout << "Number of block : " << dimBlock->x << endl;
}


void EvolutionaryAlgorithm::cudaOffspringEvaluate(void* d_offspringPopulation, float* d_fitnesses, dim3 dimBlock, dim3 dimGrid){
  cudaError_t lastError;
  size_t actualPopulationSize = this->population->actualOffspringPopulationSize;
  float* fitnesses = new float[actualPopulationSize*NB_OBJECTIVE];

  
  lastError = cudaMemcpy(d_offspringPopulation,population->cudaOffspringBuffer,(\GENOME_SIZE+sizeof(Individual*))*actualPopulationSize,
			 cudaMemcpyHostToDevice);
  DEBUG_PRT("Parent population buffer copy : %s",cudaGetErrorString(lastError));

  cudaEvaluatePopulation<<< dimGrid, dimBlock>>>(d_offspringPopulation,actualPopulationSize,d_fitnesses,initOpts);
  lastError = cudaGetLastError();
  DEBUG_PRT("Kernel execution : %s",cudaGetErrorString(lastError));

  lastError = cudaMemcpy(fitnesses,d_fitnesses,actualPopulationSize*NB_OBJECTIVE*sizeof(float),cudaMemcpyDeviceToHost);
  DEBUG_PRT("Offspring's fitnesses gathering : %s",cudaGetErrorString(lastError));


/* #ifdef COMPARE_HOST_DEVICE */
/*   population->evaluateOffspringPopulation(); */
/* #endif */

  for( size_t i=0 ; i<actualPopulationSize ; i++ ){
/* #ifdef COMPARE_HOST_DEVICE */
/*     float error = (population->offsprings[i]->getFitness()-fitnesses[i])/population->offsprings[i]->getFitness(); */
/*     printf("Difference for individual %lu is : %f %f|%f\n",i,error, population->offsprings[i]->getFitness(),fitnesses[i]); */
/*     if( error > 0.2 ) */
/*       exit(-1); */

/* #else */
    population->offsprings[i]->f1 = fitnesses[i*NB_OBJECTIVE];
    population->offsprings[i]->f2 = fitnesses[i*NB_OBJECTIVE+1];
    population->offsprings[i]->valid = true;
/* #endif */
  }
  
}

/**
   Evaluate parent population on the GPU. This is special because this evaluation occures
   only one time. Buffers are allocated and freed here.
 */
void EvolutionaryAlgorithm::cudaParentEvaluate(){
  float* fitnesses = new float[this->population->actualParentPopulationSize*NB_OBJECTIVE];
  void* allocatedDeviceBuffer;
  float* deviceFitness;
  cudaError_t lastError;
  dim3 dimBlock, dimGrid;
  size_t actualPopulationSize = this->population->actualParentPopulationSize;

  cudaPreliminaryProcess(actualPopulationSize,&dimBlock,&dimGrid,&allocatedDeviceBuffer,&deviceFitness);
    
  //compute the repartition over MP and SP
  lastError = cudaMemcpy(allocatedDeviceBuffer,this->population->cudaParentBuffer,(\GENOME_SIZE+sizeof(Individual*))*actualPopulationSize,
			 cudaMemcpyHostToDevice);
  DEBUG_PRT("Parent population buffer copy : %s",cudaGetErrorString(lastError));


  cudaEvaluatePopulation<<< dimGrid, dimBlock>>>(allocatedDeviceBuffer,actualPopulationSize,deviceFitness,initOpts);
  lastError = cudaThreadSynchronize();
  DEBUG_PRT("Kernel execution : %s",cudaGetErrorString(lastError));
 
  lastError = cudaMemcpy(fitnesses,deviceFitness,actualPopulationSize*NB_OBJECTIVE*sizeof(float),cudaMemcpyDeviceToHost);
  DEBUG_PRT("Parent's fitnesses gathering : %s",cudaGetErrorString(lastError));

  cudaFree(deviceFitness);
  cudaFree(allocatedDeviceBuffer);

/* #ifdef COMPARE_HOST_DEVICE */
/*   population->evaluateParentPopulation(); */
/* #endif */

  for( size_t i=0 ; i<actualPopulationSize ; i++ ){
/* #ifdef COMPARE_HOST_DEVICE */
/*     float error = (population->parents[i]->getFitness()-fitnesses[i])/population->parents[i]->getFitness(); */
/*     printf("Difference for individual %lu is : %f %f|%f\n",i,error, */
/* 	   population->parents[i]->getFitness(), fitnesses[i]); */
/*     if( error > 0.2 ) */
/*       exit(-1); */
/* #else */
    population->parents[i]->f1 = fitnesses[i*NB_OBJECTIVE];
    population->parents[i]->f2 = fitnesses[i*NB_OBJECTIVE+1];
    cout << i << " f1 : " << population->parents[i]->f1 << " f2 : " << population->parents[i]->f2 << endl;
    population->parents[i]->valid = true;
/* #endif */
  }
}

void EvolutionaryAlgorithm::addStoppingCriterion(StoppingCriterion* sc){
  this->stoppingCriteria.push_back(sc);
}

void EvolutionaryAlgorithm::runEvolutionaryLoop(){
  std::vector<Individual*> tmpVect;


  std::cout << "Parent's population initializing "<< std::endl;
  this->population->initializeCudaParentPopulation();
  cudaParentEvaluate();
  population->evaluateMoPopulation();
  std::cout << *population << std::endl;


  DECLARE_TIME(eval);
  DECLARE_TIME(moPop);
  struct timeval begin,accuEval;
  gettimeofday(&begin,NULL);
  accuEval.tv_sec = 0;
  accuEval.tv_usec = 0;

  struct timeval accuMo = {0,0};


  void* d_offspringPopulation;
  float* d_fitnesses;
  dim3 dimBlock, dimGrid;

  cudaPreliminaryProcess(this->population->offspringPopulationSize,&dimBlock,&dimGrid,&d_offspringPopulation,&d_fitnesses);
  
  while( this->allCriteria() == false ){    

    population->produceOffspringPopulation();
    \INSERT_BOUND_CHECKING_FCT_CALL

    TIME_ST(eval);
    for( size_t i=0 ; i<this->population->actualOffspringPopulationSize ; i++ )
      this->population->offsprings[i]->copyToCudaBuffer(this->population->cudaOffspringBuffer,i); 

    cudaOffspringEvaluate(d_offspringPopulation,d_fitnesses,dimBlock,dimGrid);
    TIME_END(eval);
    
    TIME_ST(moPop);
    population->evaluateMoPopulation();
    TIME_END(moPop);

    COMPUTE_TIME(eval);
    //SHOW_TIME(eval);
    COMPUTE_TIME(moPop);
    timeradd(&accuMo,&moPop_res,&accuMo);
    timeradd(&accuEval,&eval_res,&accuEval);
    

    if(reduceParents)
      population->reduceParentPopulation(reduceParents);
    
    if(reduceOffsprings)
      population->reduceOffspringPopulation(reduceOffsprings);
    
    population->reduceTotalPopulation();
     
    \INSERT_GEN_FCT_CALL    

    showPopulationStats(begin);
    currentGeneration += 1;
    //SHOW_TIME(moPop);
  }  
  population->sortParentPopulation();
  std::cout << *population << std::endl;
  std::cout << *population->parents[0] << std::endl;
  
  std::cout << "Generation : " << currentGeneration << std::endl;
  SHOW_SIMPLE_TIME(accuMo);
  SHOW_SIMPLE_TIME(accuEval);

  cudaFree(d_offspringPopulation);
  cudaFree(d_fitnesses);
}


void EvolutionaryAlgorithm::showPopulationStats(struct timeval beginTime){

  float currentAverageFitness=0.0;
  float currentSTDEV=0.0;

  //Calcul de la moyenne et de l'ecart type
  population->Best=population->parents[0];

  for(size_t i=0; i<population->parentPopulationSize; i++){
    currentAverageFitness+=population->parents[i]->getFitness();
#if \MINIMAXI
    if(population->parents[i]->getFitness()<population->Best->getFitness())
#else
    if(population->parents[i]->getFitness()>population->Best->getFitness())
#endif
      population->Best=population->parents[i];
  }

  currentAverageFitness/=population->parentPopulationSize;

  for(size_t i=0; i<population->parentPopulationSize; i++){
    currentSTDEV+=(population->parents[i]->getFitness()-currentAverageFitness)*(population->parents[i]->getFitness()-currentAverageFitness);
  }
  currentSTDEV/=population->parentPopulationSize;
  currentSTDEV=sqrt(currentSTDEV);
  
  //Affichage
  if(currentGeneration==0)
    printf("GEN\tTIME\tEVAL\tBEST\t\tAVG\t\tSTDEV\n\n");

  
  struct timeval end, res;
  gettimeofday(&end,0);
  timersub(&end,&beginTime,&res);
  printf("%lu\t%lu.%06lu\t%lu\t%f\t%f\t%f\n",currentGeneration,res.tv_sec,res.tv_usec,population->currentEvaluationNb,
	 population->Best->getFitness(),currentAverageFitness,currentSTDEV);
}

bool EvolutionaryAlgorithm::allCriteria(){

  for( size_t i=0 ; i<stoppingCriteria.size(); i++ ){
    if( stoppingCriteria.at(i)->reached() ){
      std::cout << "Stopping criterion reached : " << i << std::endl;
      return true;
    }
  }
  return false;
}



\START_CUDA_USER_CLASSES_H_TPL
#include <iostream>
#include <ostream>
#include <sstream>
using namespace std;
\INSERT_USER_CLASSES

\START_CUDA_GENOME_H_TPL
#ifndef __INDIVIDUAL
#define __INDIVIDUAL
#include "EASEATools.hpp"
#include <iostream>
#include <vector_types.h>
/* #include <boost/archive/text_oarchive.hpp> */
/* #include <boost/archive/text_iarchive.hpp> */

#define NB_OBJECTIVE 2

\INSERT_USER_CLASSES_DEFINITIONS

void EASEAInit(int argc, char *argv[]);
void EASEAFinal(Population* population);
void EASEAFinalization(Population* population);

class Individual{

 public: // in AESAE the genome is public (for user functions,...)
  \INSERT_GENOME
  bool valid;
  float fitness;
  static RandomGenerator* rg;

  float f1,f2; // this is fitness for dual-objective implementation

 public:
  Individual();
  Individual(const Individual& indiv);
  virtual ~Individual();
  float evaluate();
  static size_t getCrossoverArrity(){ return 2; }
  float getFitness(){ return this->fitness; }
  Individual* crossover(Individual** p2);
  void printOn(std::ostream& O) const;
  
  size_t mutate(float pMutationPerGene);
  void copyToCudaBuffer(void* buffer, size_t id);

  friend std::ostream& operator << (std::ostream& O, const Individual& B) ;
  static void initRandomGenerator(RandomGenerator* rg){ Individual::rg = rg;}
  static size_t sizeOfGenome;

/*  private: */
/*   friend class boost::serialization::access; */
/*   template <class Archive> void serialize(Archive& ar, const unsigned int version){ */

/*     ar & fitness; */
/*     DEBUG_PRT("(de)serialization of %f fitness",fitness); */
/*     ar & valid; */
/*     DEBUG_PRT("(de)serialization of %d valid",valid); */
/*     \GENOME_SERIAL */
/*   } */

  
};


/* ****************************************
   EvolutionaryAlgorithm class
****************************************/
class EvolutionaryAlgorithm{
public:
  EvolutionaryAlgorithm(  size_t parentPopulationSize, size_t offspringPopulationSize,
			  float selectionPressure, float replacementPressure, 
			  float pCrossover, float pMutation, float pMutationPerGene);
  EvolutionaryAlgorithm( size_t parentPopulationSize,size_t offspringPopulationSize,
			 float selectionPressure, float replacementPressure,
			 SelectionOperator* selectionOperator, SelectionOperator* replacementOperator,
			 float pCrossover, float pMutation, 
			 float pMutationPerGene, std::string& outputfile, std::string& inputfile);

  size_t* getCurrentGenerationPtr(){ return &currentGeneration;}
  void addStoppingCriterion(StoppingCriterion* sc);
  void runEvolutionaryLoop();
  bool allCriteria();
  Population* getPopulation(){ return population;}
  size_t getCurrentGeneration() { return currentGeneration;}
  void cudaParentEvaluate();
  void cudaOffspringEvaluate(void* d_offspringPopulation, float* fitnesses, dim3 dimBlock, dim3 dimGrid);
  void cudaPreliminaryProcess(size_t populationSize, dim3* dimBlock, dim3* dimGrid, void** allocatedDeviceBuffer,
						     float** deviceFitness);
  

public:
  size_t currentGeneration;
  Population* population;
  size_t reduceParents;
  size_t reduceOffsprings;
  //void showPopulationStats();
  void showPopulationStats(struct timeval beginTime);
  

  std::vector<StoppingCriterion*> stoppingCriteria;

  std::string* outputfile;
  std::string* inputfile;
};


#endif


\START_CUDA_TOOLS_CPP_TPL/* ****************************************
			    
   RandomGenerator class

****************************************/
#include "EASEATools.hpp"
#include "EASEAIndividual.hpp"
#include <stdio.h>
#include <iostream>
#include <values.h>
#include <string.h>
#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>


RandomGenerator::RandomGenerator(unsigned int seed){
  srand(seed);
}

int RandomGenerator::randInt(){
  return rand();
}

bool RandomGenerator::tossCoin(){

  int rVal = rand();
  if( rVal >=(RAND_MAX/2))
    return true;
  else return false;
}


bool RandomGenerator::tossCoin(float bias){

  int rVal = rand();
  if( rVal <=(RAND_MAX*bias) )
    return true;
  else return false;
}



int RandomGenerator::randInt(int min, int max){

  int rValue = (((float)rand()/RAND_MAX))*(max-min);
  //DEBUG_PRT("Int Random Value : %d",min+rValue);
  return rValue+min;

}

int RandomGenerator::random(int min, int max){
  return randInt(min,max);
}

float RandomGenerator::randFloat(float min, float max){
  float rValue = (((float)rand()/RAND_MAX))*(max-min);
  //DEBUG_PRT("Float Random Value : %f",min+rValue);
  return rValue+min;
}

float RandomGenerator::random(float min, float max){
  return randFloat(min,max);
}

double RandomGenerator::random(double min, double max){
  return randFloat(min,max);
}


int RandomGenerator::getRandomIntMax(int max){
  double r = rand();
  r = r / RAND_MAX;
  r = r * max;
  return r;
}


/* ****************************************
   Tournament class (min and max)
****************************************/
void MaxTournament::initialize(Individual** population, float selectionPressure, size_t populationSize) {
  SelectionOperator::initialize(population,selectionPressure,populationSize);
}


float MaxTournament::getExtremum(){
  return -FLT_MAX;
}

size_t MaxTournament::selectNext(size_t populationSize){
  size_t bestIndex = 0;
  float bestFitness = -FLT_MAX;

  //std::cout << "MaxTournament selection " ;
  if( currentSelectionPressure >= 2 ){
    for( size_t i = 0 ; i<currentSelectionPressure ; i++ ){
      size_t selectedIndex = rg->getRandomIntMax(populationSize);
      //std::cout << selectedIndex << " ";
      float currentFitness = population[selectedIndex]->getFitness();
      
      if( bestFitness < currentFitness ){
	bestIndex = selectedIndex;
	bestFitness = currentFitness;
      }

    }
  }
  else if( currentSelectionPressure <= 1 && currentSelectionPressure > 0 ){
    size_t i1 = rg->getRandomIntMax(populationSize);
    size_t i2 = rg->getRandomIntMax(populationSize);

    if( rg->tossCoin(currentSelectionPressure) ){
      if( population[i1]->getFitness() > population[i2]->getFitness() ){
	bestIndex = i1;
      }
    }
    else{
      if( population[i1]->getFitness() > population[i2]->getFitness() ){
	bestIndex = i2;
      }
    }
  }
  else{
    std::cerr << " MaxTournament selection operator doesn't handle selection pressure : " 
	      << currentSelectionPressure << std::endl;
  }
  //std::cout << std::endl;
  return bestIndex;
}


void MinTournament::initialize(Individual** population, float selectionPressure, size_t populationSize) {
  SelectionOperator::initialize(population,selectionPressure,populationSize);
}

float MinTournament::getExtremum(){
  return FLT_MAX;
}


size_t MinTournament::selectNext(size_t populationSize){
  size_t bestIndex = 0;
  float bestFitness = FLT_MAX;

  //std::cout << "MinTournament selection " ;
  if( currentSelectionPressure >= 2 ){
    for( size_t i = 0 ; i<currentSelectionPressure ; i++ ){
      size_t selectedIndex = rg->getRandomIntMax(populationSize);
      //std::cout << selectedIndex << " ";
      float currentFitness = population[selectedIndex]->getFitness();
      
      if( bestFitness > currentFitness ){
	bestIndex = selectedIndex;
	bestFitness = currentFitness;
      }

    }
  }
  else if( currentSelectionPressure <= 1 && currentSelectionPressure > 0 ){
    size_t i1 = rg->getRandomIntMax(populationSize);
    size_t i2 = rg->getRandomIntMax(populationSize);

    if( rg->tossCoin(currentSelectionPressure) ){
      if( population[i1]->getFitness() < population[i2]->getFitness() ){
	bestIndex = i1;
      }
    }
    else{
      if( population[i1]->getFitness() < population[i2]->getFitness() ){
	bestIndex = i2;
      }
    }
  }
  else{
    std::cerr << " MinTournament selection operator doesn't handle selection pressure : " 
	      << currentSelectionPressure << std::endl;
  }

  //std::cout << std::endl;
  return bestIndex;
}


/* ****************************************
   SelectionOperator class
****************************************/
void SelectionOperator::initialize(Individual** population, float selectionPressure, size_t populationSize){
  this->population = population;
  this->currentSelectionPressure = selectionPressure;
}

size_t SelectionOperator::selectNext(size_t populationSize){ return 0; }


/* ****************************************
   GenerationalCriterion class
****************************************/
GenerationalCriterion::GenerationalCriterion(EvolutionaryAlgorithm* ea, size_t generationalLimit){
  this->currentGenerationPtr = ea->getCurrentGenerationPtr();
  this->generationalLimit = generationalLimit;
}

bool GenerationalCriterion::reached(){
  if( generationalLimit <= *currentGenerationPtr ){
    std::cout << "Current generation " << *currentGenerationPtr << " Generational limit : " <<
      generationalLimit << std::endl;
    return true;
  }
  else return false;
}


/* ****************************************
   Population class
****************************************/
SelectionOperator* Population::selectionOperator;
SelectionOperator* Population::replacementOperator;

float Population::selectionPressure;
float Population::replacementPressure;


Population::Population(){
}

Population::Population(size_t parentPopulationSize, size_t offspringPopulationSize,
		       float pCrossover, float pMutation, float pMutationPerGene,
		       RandomGenerator* rg){
  
  this->parents     = new Individual*[parentPopulationSize];
  this->offsprings  = new Individual*[offspringPopulationSize];
  
  this->parentPopulationSize     = parentPopulationSize;
  this->offspringPopulationSize  = offspringPopulationSize;
    
  this->actualParentPopulationSize    = 0;
  this->actualOffspringPopulationSize = 0;

  this->pCrossover       = pCrossover;
  this->pMutation        = pMutation;
  this->pMutationPerGene = pMutationPerGene;

  this->rg = rg;

  this->currentEvaluationNb = 0;

  this->cudaParentBuffer = (void*)malloc((\GENOME_SIZE+sizeof(Individual*))*parentPopulationSize);
  this->cudaOffspringBuffer = (void*)malloc((\GENOME_SIZE+sizeof(Individual*))*offspringPopulationSize);

}

void Population::syncInVector(){
  for( size_t i = 0 ; i<actualParentPopulationSize ; i++ ){
    parents[i] = pop_vect.at(i);
  }
}

void Population::syncOutVector(){
  pop_vect.clear();
  for( size_t i = 0 ; i<actualParentPopulationSize ; i++ ){
    pop_vect.push_back(parents[i]);
  }
  DEBUG_PRT("Size of outVector %lu",pop_vect.size());
}

Population::~Population(){
  for( size_t i=0 ; i<actualOffspringPopulationSize ; i++ ) delete(offsprings[i]);
  for( size_t i=0 ; i<actualParentPopulationSize ; i++ )    delete(parents[i]);

  delete[](this->parents);
  delete[](this->offsprings);

  free(cudaParentBuffer);
  free(cudaOffspringBuffer);
}

void Population::initPopulation(SelectionOperator* selectionOperator, 
				SelectionOperator* replacementOperator,
				float selectionPressure, float replacementPressure){
  Population::selectionOperator   = selectionOperator;
  Population::replacementOperator = replacementOperator;
  Population::selectionPressure   = selectionPressure;
  Population::replacementPressure = replacementPressure;
}


void Population::initializeParentPopulation(){

  DEBUG_PRT("Creation of %lu/%lu parents (other could have been loaded from input file)",parentPopulationSize-actualParentPopulationSize,parentPopulationSize);
  for( size_t i=actualParentPopulationSize ; i<parentPopulationSize ; i++ )
    parents[i] = new Individual();

  actualParentPopulationSize = parentPopulationSize;
  actualOffspringPopulationSize = 0;
  
  evaluateParentPopulation();
}


/**
   Initialize parent population for CUDA template.
   i.e. create new individuals, copy them to the cuda parent's buffer
   but don't evaluate them.
 */
void Population::initializeCudaParentPopulation(){

  DEBUG_PRT("Creation of %lu/%lu parents (other could have been loaded from input file)",parentPopulationSize-actualParentPopulationSize,parentPopulationSize);
  for( size_t i=actualParentPopulationSize ; i<parentPopulationSize ; i++ )
    parents[i] = new Individual();

  actualParentPopulationSize = parentPopulationSize;
  actualOffspringPopulationSize = 0;
  
  // Copy parent population in the cuda buffer.
  for( size_t i=0 ; i<actualParentPopulationSize ; i++ ){
    parents[i]->copyToCudaBuffer(cudaParentBuffer,i); 
  }

}


void Population::evaluatePopulation(Individual** population, size_t populationSize){
  for( size_t i=0 ; i < populationSize ; i++ )
    population[i]->evaluate();
  currentEvaluationNb += populationSize;
}


void Population::evaluateParentPopulation(){
  evaluatePopulation(parents,parentPopulationSize);
}


void Population::evaluateOffspringPopulation(){
  evaluatePopulation(offsprings,offspringPopulationSize);
}


/**
   Strong dominance
 */
int check_dominance(Individual* i1 , Individual* i2){

  bool flag1=0,flag2=0;

  if( i1->f1 < i2->f1 )
    flag1 = 1;
  else 
    if( i1->f1 > i2->f1 )
      flag2 = 1;

  if( i1->f2 < i2->f2 )
    flag1 = 1;
  else 
    if( i1->f2 > i2->f2 )
      flag2 = 1;



  if( flag1 == 1 && flag2 == 0 )
    return 1;
  else 
    if( flag1 == 0 && flag2 == 1)
      return -1;
  return 0;
}


struct Individual_list{
  Individual* content;
  struct Individual_list* next;
  struct Individual_list* prev;
};


struct Individual_list* new_Individual_list_reverse(Individual** population, unsigned int size){
  struct Individual_list* head = NULL;

  for( int i=(size-1) ; i>=0 ; i--){
    struct Individual_list* tmp=(struct Individual_list*)malloc(sizeof(struct Individual_list));
    tmp->next = head;
    tmp->content = population[i];
    tmp->prev = NULL;
    if(head) head->prev = tmp;
    head = tmp; 
  }
  return head;
}

struct Individual_list* Individual_list_add_element_top(struct Individual_list* head, Individual* elt){

  struct Individual_list* ret = (struct Individual_list*)malloc(sizeof(struct Individual_list));
  ret->content = elt;
  ret->next = head;
  ret->prev = NULL;
  if( head != NULL)
    head->prev = ret;

  return ret;
}

void show_Individual_list_content(struct Individual_list* head){
  std::cout << "Printing list" << std::endl;
  unsigned int ctr = 0;
  while(head!=NULL){
    std::cout << *head->content << std::endl;//"\t\t current : " << head  << " next : " << head->next << " prev : " << head->prev ;
    head = head->next;
    ctr++;
  }
  std::cout << "\nEnd of list : "<< ctr << std::endl;
}


struct Individual_list* Individual_list_get(struct Individual_list* head, unsigned int){   
  for( struct Individual_list* h1=head ; h1!=NULL ; h1=h1->next){
    return h1;
 }
  return NULL;
}

/**
   remove an element (current) from a list (head)
   if list become empty, it is freed and become head become null
 */
void Individual_list_remove(struct Individual_list** head, struct Individual_list* current){
  struct Individual_list* tmp = current;

  if(*head==current){
    if(current->next){
      *head = (*head)->next;
      (*head)->prev = NULL;
    }
    else{
      // if this element is the last from the list
      free(tmp);
      *head=NULL;
      return ;
    }
  }
  else{
    current->prev->next = current->next;
    current->next->prev = current->prev;
  } 
  free(tmp);
}


using namespace std;
void Population::evaluateMoPopulation(){    
  size_t actualGlobalSize = actualParentPopulationSize+actualOffspringPopulationSize;
  Individual** globalPopulation = new Individual*[actualGlobalSize]();

  unsigned int rank = 1;

  memcpy(globalPopulation,parents,sizeof(Individual*)*actualParentPopulationSize);
  memcpy(globalPopulation+actualParentPopulationSize,offsprings,sizeof(Individual*)*actualOffspringPopulationSize);

  vector<Individual*>* currentVectPopulation = new vector<Individual*>();

  for( size_t i=0 ; i<actualGlobalSize ; i++ ){
    currentVectPopulation->push_back(globalPopulation[i]);
  }

  delete[](globalPopulation);

  int** dom = new int*[actualGlobalSize]();
  for( unsigned int i=0 ; i<actualGlobalSize ; i++ )
    dom[i] = new int[actualGlobalSize]();

  std::vector<Individual*>* pareto_front = new std::vector<Individual*>();
  std::vector<Individual*>* dominated_solutions = new std::vector<Individual*>();


  while( currentVectPopulation->size() ){
    
    for( size_t i = 0 ; i<currentVectPopulation->size() ; i++ ){
      for( size_t j = 0 ; j<currentVectPopulation->size() ; j++ ){
	dom[i][j] = check_dominance(currentVectPopulation->at(i),currentVectPopulation->at(j));
      }
      
    }

    int flag;

    for( unsigned int i=0 ;  i<currentVectPopulation->size(); i++){
      //printf("%02d ",i);
      flag=0;
      for( size_t j = 0 ; j<currentVectPopulation->size() ; j++ ){

	//std::cout << (dom[i][j]==1?" 1":(dom[i][j]==-1?"-1":" 0"))  << " ";

	if( dom[i][j] < 0 ){
	  flag = 1; //i is dominated by j
	}
	
      }
      //std::cout << "\t f1 : " << currentVectPopulation->at(i)->f1 << " f2 : " << currentVectPopulation->at(i)->f2 <<std::endl; 

      if( !flag ) {
	Individual* currentIndividual = currentVectPopulation->at(i);
	currentIndividual->fitness = rank;
	pareto_front->push_back(currentIndividual);
      }
      else{
	dominated_solutions->push_back(currentVectPopulation->at(i));
      }
    }
    
    rank++;
    //std::cout << "end of selection of pareto front " << rank << std::endl;
    pareto_front->clear();
    currentVectPopulation->clear();
    vector<Individual*>* tmp = currentVectPopulation;
    currentVectPopulation = dominated_solutions;
    dominated_solutions = tmp;
    
  }

  actualGlobalSize = actualParentPopulationSize+actualOffspringPopulationSize;
  for( unsigned int i=0 ; i<actualGlobalSize ; i++ )
    delete[](dom[i]);
  delete[](dom);

}


/* void Population::evaluateMoPopulation(){     */
/*   size_t actualGlobalSize = actualParentPopulationSize+actualOffspringPopulationSize; */
/*   Individual** globalPopulation = new Individual*[actualGlobalSize](); */

/*   unsigned int rank = 1; */

/*   memcpy(globalPopulation,parents,sizeof(Individual*)*actualParentPopulationSize); */
/*   memcpy(globalPopulation+actualParentPopulationSize,offsprings,sizeof(Individual*)*actualOffspringPopulationSize); */

/*   struct Individual_list* head = new_Individual_list_reverse(globalPopulation,actualGlobalSize); */

/*   delete[](globalPopulation); */

/*   struct Individual_list* h1,* h2; */
/*   int** dom = new int*[actualGlobalSize](); */
/*   for( unsigned int i=0 ; i<actualGlobalSize ; i++ ) */
/*     dom[i] = new int[actualGlobalSize](); */

/*   std::vector<Individual*> pareto_front; */

/*   show_Individual_list_content(head); */

/*   while(head!=NULL){ */
/*     unsigned int i=0,j=0; */
/*     for( h1=head ; h1!=NULL ; h1=h1->next,i++){ */
/*       j=0; */
/*       for( h2=head ; h2!=NULL ; h2=h2->next,j++){ */
/* 	dom[i][j] = check_dominance(h1->content,h2->content); */
/*       } */
      
/*     } */

/*     int flag, dummy_counter=0; */
/*     unsigned int non_dominated_ctr = 0; */

/*     for( unsigned int i=0 ;  i<actualGlobalSize; i++){ */
/*       printf("%02d ",i); */
/*       flag=0; */
/*       for( size_t j = 0 ; j<actualGlobalSize ; j++ ){ */
/* 	std::cout << (dom[i][j]==1?" 1":(dom[i][j]==-1?"-1":" 0"))  << " "; */
/* 	if( dom[i][j] < 0 ){ */
/* 	  flag = 1; //i is dominated by j */
/* 	} */
	
/*       } */
/*       std::cout << "\t f1 : " << Individual_list_get(head,i+dummy_counter)->content->f1 << " f2 : " << Individual_list_get(head,i+dummy_counter)->content->f2 <<std::endl; */
/*       if( !flag ) { */
	
/* 	struct Individual_list*  non_dominated_individual = Individual_list_get(head,i+dummy_counter); */
/* 	std::cout << *non_dominated_individual->content << std::endl; */
/* 	non_dominated_individual->content->fitness = rank; */
/* 	pareto_front.push_back(non_dominated_individual->content);   */
/* 	Individual_list_remove(&head,non_dominated_individual); */
/* 	non_dominated_ctr++; */
/* 	//Individual_list_remove(&head, */
/*       } */
/*       else{ */
/* 	dummy_counter++; */
/*       } */
/*     } */
    
/*     //for( unsigned int i=0 ; i<pareto_front.size() ; i++ ){ */
/*       //std::cout << "p" << i<<" : " << *pareto_front[i] << std::endl;  */
/*     //} */
/*     actualGlobalSize = actualGlobalSize-non_dominated_ctr; */
/*     //show_Individual_list_content(head); */
/*     rank++; */
/*     //std::cout << "end of selection of pareto front " << rank << std::endl; */
/*     pareto_front.clear(); */
/*   } */

/*   actualGlobalSize = actualParentPopulationSize+actualOffspringPopulationSize; */
/*   for( unsigned int i=0 ; i<actualGlobalSize ; i++ ) */
/*     delete[](dom[i]); */
/*   delete[](dom); */

/* } */




/**
   Reduit la population population de taille populationSize 
   a une population reducedPopulation de taille obSize.
   reducedPopulation doit etre alloue a obSize.

   Ici on pourrait avoir le best fitness de la prochaine population de parents.
   

 */
void Population::reducePopulation(Individual** population, size_t populationSize,
					  Individual** reducedPopulation, size_t obSize,
					  SelectionOperator* replacementOperator){
  

  replacementOperator->initialize(population,replacementPressure,populationSize);

  for( size_t i=0 ; i<obSize ; i++ ){
    
    // select an individual and add it to the reduced population
    size_t selectedIndex = replacementOperator->selectNext(populationSize - i);
    // std::cout << "Selected " << selectedIndex << "/" << populationSize
    // 	      << " replaced by : " << populationSize-(i+1)<< std::endl;
    reducedPopulation[i] = population[selectedIndex];
    
    // erase it to the std population by swapping last individual end current
    population[selectedIndex] = population[populationSize-(i+1)];
    //population[populationSize-(i+1)] = NULL;
  }

  //return reducedPopulation;
}


Individual** Population::reduceParentPopulation(size_t obSize){
  Individual** nextGeneration = new Individual*[obSize];

  reducePopulation(parents,actualParentPopulationSize,nextGeneration,obSize,
		   Population::replacementOperator);

  // free no longer needed individuals
  for( size_t i=0 ; i<actualParentPopulationSize-obSize ; i++ )
    delete(parents[i]);
  delete[](parents);

  this->actualParentPopulationSize = obSize;
  parents = nextGeneration;
  

  return nextGeneration;
}


Individual** Population::reduceOffspringPopulation(size_t obSize){
  Individual** nextGeneration = new Individual*[obSize];

  reducePopulation(offsprings,actualOffspringPopulationSize,nextGeneration,obSize,
		   Population::replacementOperator);

  // free no longer needed individuals
  for( size_t i=0 ; i<actualOffspringPopulationSize-obSize ; i++ )
    delete(parents[i]);
  delete[](parents);

  this->actualParentPopulationSize = obSize;
  parents = nextGeneration;
  return nextGeneration;
}


static int individualCompare(const void* p1, const void* p2){
  Individual** p1_i = (Individual**)p1;
  Individual** p2_i = (Individual**)p2;

  return p1_i[0]->getFitness() > p2_i[0]->getFitness();
}

static int individualRCompare(const void* p1, const void* p2){
  Individual** p1_i = (Individual**)p1;
  Individual** p2_i = (Individual**)p2;

  return p1_i[0]->getFitness() < p2_i[0]->getFitness();
}


void Population::sortPopulation(Individual** population, size_t populationSize){
  qsort(population,populationSize,sizeof(Individual*),individualCompare);
}

void Population::sortRPopulation(Individual** population, size_t populationSize){
  qsort(population,populationSize,sizeof(Individual*),individualRCompare);
}


/**
   Reduit les populations en faisant l'operation de remplacement.

   @TODO : on aurait voulu eviter la recopie des deux populations en une seule
   mais cela semble incompatible avec SelectionOperator (notamment l'operation 
   d'initialisation.
*/
void Population::reduceTotalPopulation(){

  Individual** nextGeneration = new Individual*[parentPopulationSize];

#if ((\ELITE_SIZE!=0) && (\ELITISM==true))                     // If there is elitism and it is strong
  Population::elitism(\ELITE_SIZE,parents,actualParentPopulationSize,
		      nextGeneration,parentPopulationSize); // do the elitism on the parent population only
  actualParentPopulationSize -= \ELITE_SIZE;                // decrement the parent population size
#endif

  size_t actualGlobalSize = actualParentPopulationSize+actualOffspringPopulationSize;
  Individual** globalPopulation = new Individual*[actualGlobalSize]();


  memcpy(globalPopulation,parents,sizeof(Individual*)*actualParentPopulationSize);
  memcpy(globalPopulation+actualParentPopulationSize,offsprings,
   	 sizeof(Individual*)*actualOffspringPopulationSize);
  replacementOperator->initialize(globalPopulation, replacementPressure,actualGlobalSize);

#if ((\ELITE_SIZE!=0) && (\ELITISM==false))                    // If there is elitism and it is weak
  Population::elitism(\ELITE_SIZE,globalPopulation,actualGlobalSize,
		      nextGeneration,parentPopulationSize); // do the elitism on the global (already merged) population
  actualGlobalSize -= \ELITE_SIZE;                // decrement the parent population size
#endif

    
  Population::reducePopulation(globalPopulation,actualGlobalSize,\ELITE_SIZE+nextGeneration,
			       parentPopulationSize-\ELITE_SIZE,replacementOperator);

  for( size_t i=0 ; i<offspringPopulationSize ; i++ )
    delete(globalPopulation[i]);
    
  delete[](parents);
  delete[](globalPopulation);

  actualParentPopulationSize = parentPopulationSize;
  actualOffspringPopulationSize = 0;
  parents = nextGeneration;
  
}


void Population::produceOffspringPopulation(){

  size_t crossoverArrity = Individual::getCrossoverArrity();
  Individual* p1;
  Individual** ps = new Individual*[crossoverArrity]();
  Individual* child;

  selectionOperator->initialize(parents,selectionPressure,actualParentPopulationSize);

  for( size_t i=0 ; i<offspringPopulationSize ; i++ ){
    size_t index = selectionOperator->selectNext(parentPopulationSize);
    p1 = parents[index];
    
    if( rg->tossCoin(pCrossover) ){
      for( size_t j=0 ; j<crossoverArrity-1 ; j++ ){
	index = selectionOperator->selectNext(parentPopulationSize);
	ps[j] = parents[index];
      }
      child = p1->crossover(ps);
    }
    else child = new Individual(*parents[index]);

    if( rg->tossCoin(pMutation) ){
      child->mutate(pMutationPerGene);
    }
    
    offsprings[actualOffspringPopulationSize++] = child;
  }
  delete[](ps);
  }




/**
   Here we save elit individuals to the replacement
   
   @ARG elitismSize the number of individuals save by elitism
   @ARG population the population where the individuals are save
   @ARG populationSize the size of the population
   @ARG outPopulation the output population, this must be allocated with size greather than elitism
   @ARG outPopulationSize the size of the output population
   
*/
void Population::elitism(size_t elitismSize, Individual** population, size_t populationSize, 
			 Individual** outPopulation, size_t outPopulationSize){
  
  float bestFitness = population[0]->getFitness();
  size_t bestIndividual = 0;
  
  if( elitismSize >= 5 )DEBUG_PRT("Warning, elitism has O(n) complexity, elitismSize is maybe too big (%lu)",elitismSize);
  
  
  for(size_t i = 0 ; i<elitismSize ; i++ ){
    bestFitness = replacementOperator->getExtremum();
    bestIndividual = 0;
    for( size_t j=0 ; j<populationSize-i ; j++ ){
#if \MINIMAXI
      if( bestFitness < population[j]->getFitness() ){
#else
      if( bestFitness > population[j]->getFitness() ){
#endif
	bestFitness = population[j]->getFitness();
	bestIndividual = j;
      }
    }
    outPopulation[i] = population[bestIndividual];
    population[bestIndividual] = population[populationSize-(i+1)];
    population[populationSize-(i+1)] = NULL;
  }
}



std::ostream& operator << (std::ostream& O, const Population& B) 
{ 
  
  size_t offspringPopulationSize = B.offspringPopulationSize;
  size_t realOffspringPopulationSize = B.actualOffspringPopulationSize;

  size_t parentPopulationSize = B.parentPopulationSize;
  size_t realParentPopulationSize = B.actualParentPopulationSize;


  O << "Population : "<< std::endl;
  O << "\t Parents size : "<< realParentPopulationSize << "/" << 
    parentPopulationSize << std::endl;
  
  for( size_t i=0 ; i<realParentPopulationSize ; i++){
    O << "\t\t" << *B.parents[i] ;
  } 

  O << "\t Offspring size : "<< realOffspringPopulationSize << "/" << 
    offspringPopulationSize << std::endl;
  for( size_t i=0 ; i<realOffspringPopulationSize ; i++){
    O << "\t\t" << *B.offsprings[i] << std::endl;
  }  
  return O; 
} 



void MaxDeterministic::initialize(Individual** population, float selectionPressure,size_t populationSize){
  SelectionOperator::initialize(population,selectionPressure,populationSize);
  Population::sortPopulation(population,populationSize);
  populationSize = populationSize;
}


size_t MaxDeterministic::selectNext(size_t populationSize){
  return populationSize-1;
}

float MaxDeterministic::getExtremum(){
  return -FLT_MAX;
}



void MinDeterministic::initialize(Individual** population, float selectionPressure,size_t populationSize){
  SelectionOperator::initialize(population,selectionPressure,populationSize);
  Population::sortRPopulation(population,populationSize);
  populationSize = populationSize;
}


size_t MinDeterministic::selectNext(size_t populationSize){
  return populationSize-1;
}

float MinDeterministic::getExtremum(){
  return FLT_MAX;
}

MaxRandom::MaxRandom(RandomGenerator* globalRandomGenerator){
  rg = globalRandomGenerator;
}

void MaxRandom::initialize(Individual** population, float selectionPressure, size_t populationSize){
  SelectionOperator::initialize(population,selectionPressure,populationSize);
}

size_t MaxRandom::selectNext(size_t populationSize){
  return rg->random(0,populationSize-1);
}

float MaxRandom::getExtremum(){
  return -FLT_MAX;
}

MinRandom::MinRandom(RandomGenerator* globalRandomGenerator){
  rg = globalRandomGenerator;
}

void MinRandom::initialize(Individual** population, float selectionPressure, size_t populationSize){
  SelectionOperator::initialize(population,selectionPressure,populationSize);
}

size_t MinRandom::selectNext(size_t populationSize){
  return rg->random(0,populationSize-1);
}

float MinRandom::getExtremum(){
  return -FLT_MAX;
}

namespace po = boost::program_options;


po::variables_map vm;
po::variables_map vm_file;

using namespace std;

string setVariable(string argumentName, string defaultValue, po::variables_map vm, po::variables_map vm_file){
  string ret;

  if( vm.count(argumentName) ){
    ret = vm[argumentName].as<string>();
    cout << argumentName << " is declared in user command line as "<< ret << endl;
  }
  else if( vm_file.count(argumentName) ){
    ret = vm_file[argumentName].as<string>();
    cout <<  argumentName << " is declared configuration file as "<< ret << endl;
  }
  else {
    ret = defaultValue;
    cout << argumentName << " is not declared, default value is "<< ret<< endl;
  }
  return ret;
}

int setVariable(string argumentName, int defaultValue, po::variables_map vm, po::variables_map vm_file ){
  int ret;

  if( vm.count(argumentName) ){
    ret = vm[argumentName].as<int>();
    cout << argumentName << " is declared in user command line as "<< ret << endl;
  }
  else if( vm_file.count(argumentName) ){
    ret = vm_file[argumentName].as<int>();
    cout <<  argumentName << " is declared configuration file as "<< ret << endl;
  }
  else {
    ret = defaultValue;
    cout << argumentName << " is not declared, default value is "<< ret<< endl;
  }
  return ret;
}


int loadParametersFile(const string& filename, char*** outputContainer){

  FILE* paramFile = fopen(filename.c_str(),"r");
  char buffer[512];
  vector<char*> tmpContainer;
  
  char* padding = (char*)malloc(sizeof(char));
  padding[0] = 0;

  tmpContainer.push_back(padding);
  
  while( fgets(buffer,512,paramFile)){
    for( size_t i=0 ; i<512 ; i++ )
      if( buffer[i] == '#' || buffer[i] == '\n' || buffer[i] == '\0' || buffer[i]==' '){
	buffer[i] = '\0';
	break;
      } 
    int str_len;
    if( (str_len = strlen(buffer)) ){
      cout << "line : " <<buffer << endl;
      char* nLine = (char*)malloc(sizeof(char)*(str_len+1));
      strcpy(nLine,buffer);
      tmpContainer.push_back(nLine);
    }    
  }

  (*outputContainer) = (char**)malloc(sizeof(char*)*tmpContainer.size());
 
  for ( size_t i=0 ; i<tmpContainer.size(); i++)
    (*outputContainer)[i] = tmpContainer.at(i);

  fclose(paramFile);
  return tmpContainer.size();
}


void parseArguments(const char* parametersFileName, int ac, char** av, 
		    po::variables_map& vm, po::variables_map& vm_file){

  char** argv;
  int argc = loadParametersFile(parametersFileName,&argv);
  
  po::options_description desc("Allowed options ");
  desc.add_options()
    ("help", "produce help message")
    ("compression", po::value<int>(), "set compression level")
    ("seed", po::value<int>(), "set the global seed of the pseudo random generator")
    ("popSize",po::value<int>(),"set the population size")
    ("nbOffspring",po::value<int>(),"set the offspring population size")
    ("elite",po::value<int>(),"Nb of elite parents (absolute)")
    ("eliteType",po::value<int>(),"Strong (1) or weak (1)")
    ("nbGen",po::value<int>(),"Set the number of generation")
    ("surviveParents",po::value<int>()," Nb of surviving parents (absolute)")
    ("surviveOffsprings",po::value<int>()," Nb of surviving offsprings (absolute)")
    ("outputfile",po::value<string>(),"Set an output file for the final population (default : none)")
    ("inputfile",po::value<string>(),"Set an input file for the initial population (default : none)")
    ("u1",po::value<string>(),"User defined parameter 1")
    ("u2",po::value<string>(),"User defined parameter 2")
    ("u3",po::value<string>(),"User defined parameter 3")
    ("u4",po::value<string>(),"User defined parameter 4")
    ;
    
  try{
    po::store(po::parse_command_line(ac, av, desc,0), vm);
    po::store(po::parse_command_line(argc, argv, desc,0), vm_file);
  }
  catch(po::unknown_option& e){
    cerr << "Unknown option  : " << e.what() << endl;    
    cout << desc << endl;
    exit(1);
  }
  
  po::notify(vm);    
  po::notify(vm_file);    

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }

  for( int i = 0 ; i<argc ; i++ )
    free(argv[i]);
  free(argv);
 
}

void parseArguments(const char* parametersFileName, int ac, char** av){
  parseArguments(parametersFileName,ac,av,vm,vm_file);
}


int setVariable(const string optionName, int defaultValue){
  return setVariable(optionName,defaultValue,vm,vm_file);
}

string setVariable(const string optionName, string defaultValue){
  return setVariable(optionName,defaultValue,vm,vm_file);
}








\START_CUDA_TOOLS_H_TPL#ifndef TIMING_H
#define TIMING_H

#include <time.h>             //gettimeofday
#include <sys/time.h>
#include <stdio.h>



#ifdef TIMING
#define DECLARE_TIME(t)				\
  struct timeval t##_beg, t##_end, t##_res
#define TIME_ST(t)				\
  gettimeofday(&t##_beg,NULL)
#define TIME_END(t)				\
  gettimeofday(&t##_end,NULL)
#define SHOW_TIME(t)						\
  timersub(&t##_end,&t##_beg,&t##_res);				\
  printf("%s : %lu.%06lu\n",#t,t##_res.tv_sec,t##_res.tv_usec)
#define SHOW_SIMPLE_TIME(t)					\
  printf("%s : %lu.%06lu\n",#t,t.tv_sec,t.tv_usec)
#define COMPUTE_TIME(t)						\
  timersub(&t##_end,&t##_beg,&t##_res)
#else
#define DECLARE_TIME(t)
#define TIME_ST(t)
#define TIME_END(t)
#define SHOW_TIME(t)
#define SHOW_SIMPLE_TIME(t)
#endif


#endif

/* ****************************************
   Some tools classes for algorithm
****************************************/
#include <stdlib.h>
#include <vector>
#include <iostream>
/* #include <boost/archive/text_oarchive.hpp> //for serialization (dumping) */
/* #include <boost/archive/text_iarchive.hpp> //for serialization (loading) */
/* #include <boost/serialization/vector.hpp> */

class EvolutionaryAlgorithm;
class Individual;
class Population;

#define EZ_MINIMIZE \MINIMAXI
#define EZ_MINIMISE \MINIMAXI
#define EZ_MAXIMIZE !\MINIMAXI
#define EZ_MAXIMISE !\MINIMAXI

#ifdef DEBUG
#define DEBUG_PRT(format, args...) fprintf (stdout,"***DBG***  %s-%d: "format"\n",__FILE__,__LINE__,##args)
#define DEBUG_YACC(format, args...) fprintf (stdout,"***DBG_YACC***  %s-%d: "format"\n",__FILE__,__LINE__,##args)
#else
#define DEBUG_PRT(format, args...) 
#define DEBUG_YACC(format, args...)
#endif


/* ****************************************
   StoppingCriterion class
****************************************/
#ifndef __EASEATOOLS
#define __EASEATOOLS
class StoppingCriterion {

public:
  virtual bool reached() = 0;

};


/* ****************************************
   GenerationalCriterion class
****************************************/
class GenerationalCriterion : public StoppingCriterion {
 private:
  size_t* currentGenerationPtr;
  size_t generationalLimit;
 public:
  virtual bool reached();
  GenerationalCriterion(EvolutionaryAlgorithm* ea, size_t generationalLimit);
  
};


/* ****************************************
   RandomGenerator class
****************************************/
class RandomGenerator{
public:
  RandomGenerator(unsigned int seed);
  int randInt();
  bool tossCoin();
  bool tossCoin(float bias);
  int randInt(int min, int max);
  int getRandomIntMax(int max);
  float randFloat(float min, float max);
  int random(int min, int max);
  float random(float min, float max);
  double random(double min, double max);
};



/* ****************************************
   Selection Operator class
****************************************/
class SelectionOperator{
public:
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  virtual float getExtremum() = 0 ;
protected:
  Individual** population;
  float currentSelectionPressure;
};


/* ****************************************
   Tournament classes (min and max)
****************************************/
class MaxTournament : public SelectionOperator{
public:
  MaxTournament(RandomGenerator* rg){ this->rg = rg; }
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
private:
  RandomGenerator* rg;
  
};



class MinTournament : public SelectionOperator{
public:
  MinTournament(RandomGenerator* rg){ this->rg = rg; }
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
private:
  RandomGenerator* rg;
  
};


class MaxDeterministic : public SelectionOperator{
 public:
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;
};

class MinDeterministic : public SelectionOperator{
 public:
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;

};


class MaxRandom : public SelectionOperator{
 public:
  MaxRandom(RandomGenerator* globalRandomGenerator);
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;
  RandomGenerator* rg;

};

class MinRandom : public SelectionOperator{
 public:
  MinRandom(RandomGenerator* globalRandomGenerator);
  virtual void initialize(Individual** population, float selectionPressure, size_t populationSize);
  virtual size_t selectNext(size_t populationSize);
  float getExtremum();
 private:
  size_t populationSize;
  RandomGenerator* rg;
};



class Population {
  
 public:
  
  float pCrossover;
  float pMutation;
  float pMutationPerGene;

  Individual* Best;
  
  Individual** parents;
  Individual** offsprings;

  size_t parentPopulationSize;
  size_t offspringPopulationSize;

  size_t actualParentPopulationSize;
  size_t actualOffspringPopulationSize;

  static SelectionOperator* selectionOperator;
  static SelectionOperator* replacementOperator;

  size_t currentEvaluationNb;
  RandomGenerator* rg;
  std::vector<Individual*> pop_vect;

  void* cudaParentBuffer;
  void* cudaOffspringBuffer;

 public:
  Population();
  Population(size_t parentPopulationSize, size_t offspringPopulationSize, 
	     float pCrossover, float pMutation, float pMutationPerGene, RandomGenerator* rg);
  virtual ~Population();

  void initializeParentPopulation();  
  void initializeCudaParentPopulation();
  void evaluatePopulation(Individual** population, size_t populationSize);
  void evaluateParentPopulation();
  void evaluateMoPopulation();

  static void elitism(size_t elitismSize, Individual** population, size_t populationSize, Individual** outPopulation,
		      size_t outPopulationSize);

  void evaluateOffspringPopulation();
  Individual** reducePopulations(Individual** population, size_t populationSize,
			       Individual** reducedPopulation, size_t obSize);
  Individual** reduceParentPopulation(size_t obSize);
  Individual** reduceOffspringPopulation(size_t obSize);
  void reduceTotalPopulation();
  void evolve();

  static float selectionPressure;
  static float replacementPressure;
  static void initPopulation(SelectionOperator* selectionOperator, 
			     SelectionOperator* replacementOperator,
			     float selectionPressure, float replacementPressure);

  static void sortPopulation(Individual** population, size_t populationSize);

  static void sortRPopulation(Individual** population, size_t populationSize);


  void sortParentPopulation(){ Population::sortPopulation(parents,actualParentPopulationSize);}

  void produceOffspringPopulation();

  friend std::ostream& operator << (std::ostream& O, const Population& B);


  void setParentPopulation(Individual** population, size_t actualParentPopulationSize){ 
    this->parents = population;
    this->actualParentPopulationSize = actualParentPopulationSize;
  }

  static void reducePopulation(Individual** population, size_t populationSize,
				       Individual** reducedPopulation, size_t obSize,
				       SelectionOperator* replacementOperator);
  void syncOutVector();
  void syncInVector();

/*  private: */
/*   friend class boost::serialization::access; */
/*   template <class Archive> void serialize(Archive& ar, const unsigned int version){ */

/*     ar & actualParentPopulationSize; */
/*     DEBUG_PRT("(de)serialization of %d parents",actualParentPopulationSize); */
/*     ar & pop_vect; */
/*     DEBUG_PRT("(de)serialization of %d offspring",actualOffspringPopulationSize); */
/*   } */
};

/* namespace boost{ */
/*   namespace serialization{ */
/*     template<class Archive> */
/*       void serialize(Archive & ar,std::vector<Individual*> population, const unsigned int version){ */
/*       ar & population; */
/*     } */
/*   } */
/* } */

void parseArguments(const char* parametersFileName, int ac, char** av);
int setVariable(const std::string optionName, int defaultValue);
std::string setVariable(const std::string optionName, std::string defaultValue);



#endif



\START_CUDA_MAKEFILE_TPL

NVCC= nvcc
CPPC= g++
CXXFLAGS+=-g -Wall -O2
LDFLAGS=-lboost_program_options -lboost_serialization

#USER MAKEFILE OPTIONS :
\INSERT_MAKEFILE_OPTION#END OF USER MAKEFILE OPTIONS

CPPFLAGS+=
NVCCFLAGS+=


EASEA_SRC= EASEATools.cpp EASEAIndividual.cpp
EASEA_MAIN_HDR= EASEA.cpp
EASEA_UC_HDR= EASEAUserClasses.hpp

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
	rm -f Makefile EASEA.prm $(SRC) $(HDR) EASEA.mak $(CUDA_SRC)
clean:
	rm -f $(OBJ) $(BIN) *.linkinfo 

\START_EO_PARAM_TPL#****************************************
#                                         
#  EASEA.prm
#                                         
#  Parameter file generated by AESAE-EO v0.7b
#                                         
#***************************************
# --seed=0   # -S : Random number seed. It is possible to give a specific seed.

######    Evolution Engine    ######
--popSize=\POP_SIZE # -P : Population Size
--nbOffspring=\OFF_SIZE # -O : Nb of offspring (percentage or absolute)

######    Evolution Engine / Replacement    ######
--elite=\ELITE_SIZE  # Nb of elite parents (percentage or absolute)
--eliteType=\ELITISM # Strong (true) or weak (false) elitism (set elite to 0 for none)
--surviveParents=\SURV_PAR_SIZE # Nb of surviving parents (percentage or absolute)
# --reduceParents=Ranking # Parents reducer: Deterministic, EP(T), DetTour(T), StochTour(t), Uniform
--surviveOffspring=\SURV_OFF_SIZE  # Nb of surviving offspring (percentage or absolute)
# --reduceOffspring=Roulette # Offspring reducer: Deterministic, EP(T), DetTour(T), StochTour(t), Uniform
# --reduceFinal=DetTour(2) # Final reducer: Deterministic, EP(T), DetTour(T), StochTour(t), Uniform


\TEMPLATE_END
