\TEMPLATE_START
/**
 This is program entry for STD template for EASEA

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
unsigned *EZ_NB_GEN;
unsigned *EZ_current_generation;
int EZ_POP_SIZE;
int OFFSPRING_SIZE;

CEvolutionaryAlgorithm* EA;

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


	return 0;
}

\START_CUDA_GENOME_CU_TPL

#include <fstream>
#include <time.h>
#include <cstring>
#include <sstream>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"

using namespace std;

#include "EASEAIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;
#define STD_TPL

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES


\INSERT_USER_FUNCTIONS

\INSERT_INITIALISATION_FUNCTION
\INSERT_FINALIZATION_FUNCTION


void evale_pop_chunk(CIndividual** population, int popSize){
  \INSTEAD_EVAL_FUNCTION
}

void EASEAInit(int argc, char** argv){
	\INSERT_INIT_FCT_CALL
}

void EASEAFinal(CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;
}

void AESAEBeginningGenerationFunction(CEvolutionaryAlgorithm* ea){
	\INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction(CEvolutionaryAlgorithm* ea){
	\INSERT_END_GENERATION_FUNCTION
  
  /* In this section we override the selection and reduction process of the genetic
   * engine (libeasea)
   * We also use a custom result printer
   */

  /* Selection and reduction code*/
  selectNSGA(*ea->population->offsprings, ea->population->offspringPopulationSize,
  *ea->population->parents, ea->population->parentPopulationSize, false);
 
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
  return 0.0f;
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
    this->isImmigrant = false;
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
  
  for (int i = 0; i < \NB_OBJECTIVE; i++) {
    this->f[i] = genome.f[i];
  }

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


unsigned IndividualImpl::mutate( float pMutationPerGene ){
  this->valid=false;


  // ********************
  // Problem specific part
  \INSERT_MUTATOR
}

void ParametersImpl::setDefaultParameters(int argc, char** argv){

	this->minimizing = \MINIMAXI;
	this->nbGen = setVariable("nbGen",(int)\NB_GEN);

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

	generationalCriterion = new CGenerationalCriterion(setVariable("nbGen",(int)\NB_GEN));
	controlCStopingCriterion = new CControlCStopingCriterion();
	timeCriterion = new CTimeCriterion(setVariable("timeLimit",\TIME_LIMIT));

	this->optimise = 0;

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
	//EZ_NB_GEN = (unsigned*)setVariable("nbGen",\NB_GEN);
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
	  ifstream AESAE_File("EASEA.pop");
	  string AESAE_Line;
  	  for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
	  	  getline(AESAE_File, AESAE_Line);
		  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
		  ((IndividualImpl*)this->population->parents[i])->deserialize(AESAE_Line);
	  }
	  
	}
	else{
  	  for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
		  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
	  }
	}
        this->population->actualParentPopulationSize = this->params->parentPopulationSize;
}


EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){
	;
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

/********************************************************************/
/* NSGA-II routines 				     		    */
/* code readapated from Deb's original NSGA-II code version 1.1.6   */
/********************************************************************/

int nsga_popsize=0;
double nsga_seed;
double nsga_oldrand[55];
int nsga_jrand;
int nsga_nobj=\NB_OBJECTIVE;

/* Get nsga_seed number for random and start it up */
void randomize() {
    int j1;
    for(j1=0; j1<=54; j1++) {
        nsga_oldrand[j1] = 0.0;
    }
    nsga_jrand=0;
    warmup_random (nsga_seed);
    return;
}

/* Get randomize off and running */
void warmup_random (double nsga_seed) {
    int j1, ii;
    double new_random, prev_random;
    nsga_oldrand[54] = nsga_seed;
    new_random = 0.000000001;
    prev_random = nsga_seed;
    for(j1=1; j1<=54; j1++) {
        ii = (21*j1)%54;
        nsga_oldrand[ii] = new_random;
        new_random = prev_random-new_random;
        if(new_random<0.0) {
            new_random += 1.0;
        }
        prev_random = nsga_oldrand[ii];
    }
    advance_random ();
    advance_random ();
    advance_random ();
    nsga_jrand = 0;
    return;
}

/* Create next batch of 55 random numbers */
void advance_random () {
    int j1;
    double new_random;
    for(j1=0; j1<24; j1++) {
        new_random = nsga_oldrand[j1]-nsga_oldrand[j1+31];
        if(new_random<0.0) {
            new_random = new_random+1.0;
        }
        nsga_oldrand[j1] = new_random;
    }
    for(j1=24; j1<55; j1++) {
        new_random = nsga_oldrand[j1]-nsga_oldrand[j1-24];
        if(new_random<0.0) {
            new_random = new_random+1.0;
        }
        nsga_oldrand[j1] = new_random;
    }
}

/* Fetch a single random number between 0.0 and 1.0 */
double randomperc() {
    nsga_jrand++;
    if(nsga_jrand>=55) {
        nsga_jrand = 1;
        advance_random();
    }
    return((double)nsga_oldrand[nsga_jrand]);
}

/* Fetch a single random integer between low and high including the bounds */
int rnd (int low, int high) {
    int res;
    if (low >= high) {
        res = low;
    } else {
        res = low + (randomperc()*(high-low+1));
        if (res > high) {
            res = high;
        }
    }
    return (res);
}

/* Fetch a single random real number between low and high including the bounds */
double rndreal (double low, double high) {
    return (low + (high-low)*randomperc());
}


/* Function to allocate memory to a NSGA_population */
void allocate_memory_pop (NSGA_population *pop, int size) {
    int i;
    pop->ind = (individual *)malloc(size*sizeof(individual));
    for (i=0; i<size; i++) {
        allocate_memory_ind (&(pop->ind[i]));
    }
    return;
}

/* Function to allocate memory to an individual */
void allocate_memory_ind (individual *ind) {
    ind->obj = (double *)malloc(nsga_nobj*sizeof(double));
    return;
}

/* Function to deallocate memory to a NSGA_population */
void deallocate_memory_pop (NSGA_population *pop, int size) {
    int i;
    for (i=0; i<size; i++) {
        deallocate_memory_ind (&(pop->ind[i]));
    }
    free (pop->ind);
    return;
}

/* Function to deallocate memory to an individual */
void deallocate_memory_ind (individual *ind) {
    free(ind->obj);
    return;
}
/* Crowding distance computation routines */


/* Routine to compute crowding distance based on ojbective function values when the NSGA_population in in the form of a listNSGA */
void assign_crowding_distance_listNSGA (NSGA_population *pop, listNSGA *lst,
                                        int front_size) {
    int **obj_array;
    int *dist;
    int i, j;
    listNSGA *temp;
    temp = lst;
    if (front_size==1) {
        pop->ind[lst->index].crowd_dist = INF;
        return;
    }
    if (front_size==2) {
        pop->ind[lst->index].crowd_dist = INF;
        pop->ind[lst->child->index].crowd_dist = INF;
        return;
    }
    obj_array = (int **)malloc(nsga_nobj*sizeof(int*));
    dist = (int *)malloc(front_size*sizeof(int));
    for (i=0; i<nsga_nobj; i++) {
        obj_array[i] = (int *)malloc(front_size*sizeof(int));
    }
    for (j=0; j<front_size; j++) {
        dist[j] = temp->index;
        temp = temp->child;
    }
    assign_crowding_distance (pop, dist, obj_array, front_size);
    free (dist);
    for (i=0; i<nsga_nobj; i++) {
        free (obj_array[i]);
    }
    free (obj_array);
    return;
}

/* Routine to compute crowding distance based on objective function values when the NSGA_population in in the form of an array */
void assign_crowding_distance_indices (NSGA_population *pop, int c1, int c2) {
    int **obj_array;
    int *dist;
    int i, j;
    int front_size;
    front_size = c2-c1+1;
    if (front_size==1) {
        pop->ind[c1].crowd_dist = INF;
        return;
    }
    if (front_size==2) {
        pop->ind[c1].crowd_dist = INF;
        pop->ind[c2].crowd_dist = INF;
        return;
    }
    obj_array = (int **)malloc(nsga_nobj*sizeof(int*));
    dist = (int *)malloc(front_size*sizeof(int));
    for (i=0; i<nsga_nobj; i++) {
        obj_array[i] = (int *)malloc(front_size*sizeof(int));
    }
    for (j=0; j<front_size; j++) {
        dist[j] = c1++;
    }
    assign_crowding_distance (pop, dist, obj_array, front_size);
    free (dist);
    for (i=0; i<nsga_nobj; i++) {
        free (obj_array[i]);
    }
    free (obj_array);
    return;
}

/* Routine to compute crowding distances */
void assign_crowding_distance (NSGA_population *pop, int *dist, int **obj_array,
                               int front_size) {
    int i, j;
    for (i=0; i<nsga_nobj; i++) {
        for (j=0; j<front_size; j++) {
            obj_array[i][j] = dist[j];
        }
        quicksort_front_obj (pop, i, obj_array[i], front_size);
    }
    for (j=0; j<front_size; j++) {
        pop->ind[dist[j]].crowd_dist = 0.0;
    }
    for (i=0; i<nsga_nobj; i++) {
        pop->ind[obj_array[i][0]].crowd_dist = INF;
    }
    for (i=0; i<nsga_nobj; i++) {
        for (j=1; j<front_size-1; j++) {
            if (pop->ind[obj_array[i][j]].crowd_dist != INF) {
                if (pop->ind[obj_array[i][front_size-1]].obj[i] ==
                        pop->ind[obj_array[i][0]].obj[i]) {
                    pop->ind[obj_array[i][j]].crowd_dist += 0.0;
                } else {
                    pop->ind[obj_array[i][j]].crowd_dist += (pop->ind[obj_array[i][j+1]].obj[i] -
                                                            pop->ind[obj_array[i][j-1]].obj[i])/(pop->ind[obj_array[i][front_size-1]].obj[i]
                                                                    - pop->ind[obj_array[i][0]].obj[i]);
                }
            }
        }
    }
    for (j=0; j<front_size; j++) {
        if (pop->ind[dist[j]].crowd_dist != INF) {
            pop->ind[dist[j]].crowd_dist = (pop->ind[dist[j]].crowd_dist)/nsga_nobj;
        }
    }
    return;
}
/* Domination checking routines */


/* Routine for usual non-domination checking
   It will return the following values
   1 if a dominates b
   -1 if b dominates a
   0 if both a and b are non-dominated */

int check_dominance (individual *a, individual *b) {
    int i;
    int flag1;
    int flag2;
    flag1 = 0;
    flag2 = 0;
    for (i=0; i<nsga_nobj; i++) {
        if (a->obj[i] < b->obj[i]) {
            flag1 = 1;

        } else {
            if (a->obj[i] > b->obj[i]) {
                flag2 = 1;
            }
        }
    }
    if (flag1==1 && flag2==0) {
        return (1);
    } else {
        if (flag1==0 && flag2==1) {
            return (-1);
        } else {
            return (0);
        }
    }
}
/* A custom doubly linked listNSGA implemenation */


/* Insert an element X into the listNSGA at location specified by NODE */
void insert (listNSGA *node, int x) {
    listNSGA *temp;
    if (node==NULL) {
        printf("\n Error!! asked to enter after a NULL pointer, hence exiting \n");
        exit(1);
    }
    temp = (listNSGA *)malloc(sizeof(listNSGA));
    temp->index = x;
    temp->child = node->child;
    temp->parent = node;
    if (node->child != NULL) {
        node->child->parent = temp;
    }
    node->child = temp;
    return;
}

/* Delete the node NODE from the listNSGA */
listNSGA* del (listNSGA *node) {
    listNSGA *temp;
    if (node==NULL) {
        printf("\n Error!! asked to delete a NULL pointer, hence exiting \n");
        exit(1);
    }
    temp = node->parent;
    temp->child = node->child;
    if (temp->child!=NULL) {
        temp->child->parent = temp;
    }
    free (node);
    return (temp);
}
/* Routines for randomized recursive quick-sort */


/* Randomized quick sort routine to sort a NSGA_population based on a particular objective chosen */
void quicksort_front_obj(NSGA_population *pop, int objcount, int obj_array[],
                         int obj_array_size) {
    q_sort_front_obj (pop, objcount, obj_array, 0, obj_array_size-1);
    return;
}

/* Actual implementation of the randomized quick sort used to sort a NSGA_population based on a particular objective chosen */
void q_sort_front_obj(NSGA_population *pop, int objcount, int obj_array[],
                      int left, int right) {
    int index;
    int temp;
    int i, j;
    double pivot;
    if (left<right) {
        index = rnd (left, right);
        temp = obj_array[right];
        obj_array[right] = obj_array[index];
        obj_array[index] = temp;
        pivot = pop->ind[obj_array[right]].obj[objcount];
        i = left-1;
        for (j=left; j<right; j++) {
            if (pop->ind[obj_array[j]].obj[objcount] <= pivot) {
                i+=1;
                temp = obj_array[j];
                obj_array[j] = obj_array[i];
                obj_array[i] = temp;
            }
        }
        index=i+1;
        temp = obj_array[index];
        obj_array[index] = obj_array[right];
        obj_array[right] = temp;
        q_sort_front_obj (pop, objcount, obj_array, left, index-1);
        q_sort_front_obj (pop, objcount, obj_array, index+1, right);
    }
    return;
}

/* Randomized quick sort routine to sort a NSGA_population based on crowding distance */
void quicksort_dist(NSGA_population *pop, int *dist, int front_size) {
    q_sort_dist (pop, dist, 0, front_size-1);
    return;
}

/* Actual implementation of the randomized quick sort used to sort a NSGA_population based on crowding distance */
void q_sort_dist(NSGA_population *pop, int *dist, int left, int right) {
    int index;
    int temp;
    int i, j;
    double pivot;
    if (left<right) {
        index = rnd (left, right);
        temp = dist[right];
        dist[right] = dist[index];
        dist[index] = temp;
        pivot = pop->ind[dist[right]].crowd_dist;
        i = left-1;
        for (j=left; j<right; j++) {
            if (pop->ind[dist[j]].crowd_dist <= pivot) {
                i+=1;
                temp = dist[j];
                dist[j] = dist[i];
                dist[i] = temp;
            }
        }
        index=i+1;
        temp = dist[index];
        dist[index] = dist[right];
        dist[right] = temp;
        q_sort_dist (pop, dist, left, index-1);
        q_sort_dist (pop, dist, index+1, right);
    }
    return;
}
/* Rank assignment routine */


/* Function to assign rank and crowding distance to a NSGA_population of size pop_size*/
void assign_rank_and_crowding_distance (NSGA_population *new_pop) {
    int flag;
    int i;
    int end;
    int front_size;
    int rank=1;
    listNSGA *orig;
    listNSGA *cur;
    listNSGA *temp1, *temp2;
    orig = (listNSGA *)malloc(sizeof(listNSGA));
    cur = (listNSGA *)malloc(sizeof(listNSGA));
    front_size = 0;
    orig->index = -1;
    orig->parent = NULL;
    orig->child = NULL;
    cur->index = -1;
    cur->parent = NULL;
    cur->child = NULL;
    temp1 = orig;
    for (i=0; i<nsga_popsize; i++) {
        insert (temp1,i);
        temp1 = temp1->child;
    }
    do {
        if (orig->child->child == NULL) {
            new_pop->ind[orig->child->index].rank = rank;
            new_pop->ind[orig->child->index].crowd_dist = INF;
            break;
        }
        temp1 = orig->child;
        insert (cur, temp1->index);
        front_size = 1;
        temp2 = cur->child;
        temp1 = del (temp1);
        temp1 = temp1->child;
        do {
            temp2 = cur->child;
            do {
                end = 0;
                flag = check_dominance (&(new_pop->ind[temp1->index]),
                                        &(new_pop->ind[temp2->index]));
                if (flag == 1) {
                    insert (orig, temp2->index);
                    temp2 = del (temp2);
                    front_size--;
                    temp2 = temp2->child;
                }
                if (flag == 0) {
                    temp2 = temp2->child;
                }
                if (flag == -1) {
                    end = 1;
                }
            } while (end!=1 && temp2!=NULL);
            if (flag == 0 || flag == 1) {
                insert (cur, temp1->index);
                front_size++;
                temp1 = del (temp1);
            }
            temp1 = temp1->child;
        } while (temp1 != NULL);
        temp2 = cur->child;
        do {
            new_pop->ind[temp2->index].rank = rank;
            temp2 = temp2->child;
        } while (temp2 != NULL);
        assign_crowding_distance_listNSGA (new_pop, cur->child, front_size);
        temp2 = cur->child;
        do {
            temp2 = del (temp2);
            temp2 = temp2->child;
        } while (cur->child !=NULL);
        rank+=1;
    } while (orig->child!=NULL);
    free (orig);
    free (cur);
    return;
}

int comparator(const void *p, const void *q) {

    if (((individual *) p)->rank != ((individual *) q)->rank)
        return (((individual *) p)->rank >  ((individual *) q)->rank);
    else
        return ( (individual *) p )->crowd_dist > ((individual *) q)->crowd_dist;

}




void selectNSGA(CIndividual* children, size_t numChildren,
            CIndividual* parents, size_t numParents,
            bool mixedpop) {

    int i;
    int j;
    NSGA_population *nsga_pop;


    if (!mixedpop)
        nsga_popsize = numChildren + numParents;
    else
        nsga_popsize = numChildren;


    nsga_pop = (NSGA_population *)malloc(sizeof(NSGA_population));
    allocate_memory_pop (nsga_pop,nsga_popsize);

    randomize(); /*FIXME: check how to deal with EASEA own random  routines*/




    if (!mixedpop) {
        for (i = 0; i < (int) numParents; i++) {
            nsga_pop->ind[i].id=i;
            for(j=0; j<nsga_nobj; j++) {
                nsga_pop->ind[i].obj[j]
                = reinterpret_cast<IndividualImpl*>(parents)[i].f[j];
            }

        }

        for (i = 0; i < (int) numChildren; i++) {
            nsga_pop->ind[numParents + i].id = i + numParents;
            for(j=0; j<nsga_nobj; j++) {
                nsga_pop->ind[numParents + i].obj[j] =
                    reinterpret_cast<IndividualImpl*>(children)[i].f[j];
            }

        }

    } else {
        for (i=0; i<(int)nsga_popsize; i++) {
            nsga_pop->ind[i].id=i;
            for(j=0; j<nsga_nobj; j++) {
                nsga_pop->ind[i].obj[j] = reinterpret_cast<IndividualImpl*>(children)[i].f[j];
            }

        }
    }

    assign_rank_and_crowding_distance (nsga_pop);
    qsort((void *)nsga_pop->ind,nsga_popsize,sizeof(nsga_pop->ind[0]),comparator);


    for (i=0; i<nsga_popsize; i++) {
        /*
        		printf("[%d]->id %d\n",i,nsga_pop->ind[i].id);
        		printf("[%d]->crowd_dist %f\n",i,nsga_pop->ind[i].crowd_dist);
        		printf("[%d]->rank %d\n",i,nsga_pop->ind[i].rank);
        		printf("[%d]->f0 %d %f ->f1 %f\n",i,nsga_pop->ind[i].obj[0], nsga_pop->ind[i].obj[1]);
        */
        if (!mixedpop) {
            if (nsga_pop->ind[i].id < numParents) {
                CIndividual* indiv = &(parents[nsga_pop->ind[i].id]);
                indiv->rank = nsga_pop->ind[i].rank;
            } else {
                CIndividual* indiv = &(children[nsga_pop->ind[i].id - numParents]);
                indiv->rank = nsga_pop->ind[i].rank;
            }
        } else {
            CIndividual* indiv = &(parents[nsga_pop->ind[i].id]);
            indiv->rank = nsga_pop->ind[i].rank;
        }
    }

    deallocate_memory_pop(nsga_pop,nsga_popsize);
}

\START_CUDA_GENOME_H_TPL

#ifndef PROBLEM_DEP_H
#define PROBLEM_DEP_H

//#include "CRandomGenerator.h"
#include <stdlib.h>
#include <iostream>
#include <CIndividual.h>
#include <Parameters.h>
#include <cstring>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>


using namespace std;

class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class Parameters;

extern int EZ_POP_SIZE;
extern int OFFSPRING_SIZE;

\INSERT_USER_CLASSES

class IndividualImpl : public CIndividual {

public: // in EASEA the genome is public (for user functions,...)
	// Class members
  	\INSERT_GENOME
    float f[\NB_OBJECTIVE];

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

/* Memory allocation and deallocation routines */

# define NSGA2_NOBJ 3
# define INF 1.0e14
# define EPS 1.0e-14
# define E  2.71828182845905
# define PI 3.14159265358979
# define GNUPLOT_COMMAND "gnuplot -persist"

typedef struct {
    int rank;
    double *obj;
    double crowd_dist;
    int id;
}
individual;

typedef struct {
    individual *ind;
}
NSGA_population;

typedef struct listNSGAs {
    int index;
    struct listNSGAs *parent;
    struct listNSGAs *child;
}
listNSGA;


void allocate_memory_pop (NSGA_population *pop, int size);
void allocate_memory_ind (individual *ind);
void deallocate_memory_pop (NSGA_population *pop, int size);
void deallocate_memory_ind (individual *ind);
void assign_crowding_distance_listNSGA (NSGA_population *pop, listNSGA *lst,
                                        int front_size);
void assign_crowding_distance_indices (NSGA_population *pop, int c1, int c2);
void assign_crowding_distance (NSGA_population *pop, int *dist, int **obj_array,
                               int front_size);
int check_dominance (individual *a, individual *b);
void insert (listNSGA *node, int x);
void quicksort_front_obj(NSGA_population *pop, int objcount, int obj_array[],
                         int obj_array_size);
void q_sort_front_obj(NSGA_population *pop, int objcount, int obj_array[],
                      int left, int right);
void quicksort_dist(NSGA_population *pop, int *dist, int front_size);
void q_sort_dist(NSGA_population *pop, int *dist, int left, int right);
void assign_rank_and_crowding_distance (NSGA_population *new_pop);
void advance_random ();
void randomize();
void warmup_random (double nsga_seed);

void selectNSGA(CIndividual* children, size_t numChildren,
            CIndividual* parents, size_t numParents,
            bool mixedpop);

/* Definition of random number generation routines */


#endif /* PROBLEM_DEP_H */

\START_CUDA_MAKEFILE_TPL

UNAME := $(shell uname)

ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif

ifneq ("$(OS)","")
	EZ_PATH=../../
endif

EASEALIB_PATH=$(EZ_PATH)/libeasea/

CXXFLAGS =  -fopenmp	-O2 -g -Wall -fmessage-length=0 -I$(EASEALIB_PATH)include -I$(EZ_PATH)boost

OBJS = EASEA.o EASEAIndividual.o 

LIBS = -lpthread -fopenmp 
ifneq ("$(OS)","")
	LIBS += -lws2_32 -lwinmm -L"C:\MinGW\lib"
endif

#USER MAKEFILE OPTIONS :
\INSERT_MAKEFILE_OPTION
#END OF USER MAKEFILE OPTIONS

TARGET =	EASEA

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS) -g $(EASEALIB_PATH)/libeasea.a $(EZ_PATH)boost/program_options.a $(LIBS)

	
#%.o:%.cpp
#	$(CXX) -c $(CXXFLAGS) $^

all:	$(TARGET)
clean:
ifneq ("$(OS)","")
	-del $(OBJS) $(TARGET).exe
else
	rm -f $(OBJS) $(TARGET)
endif
easeaclean:
ifneq ("$(OS)","")
	-del $(TARGET).exe *.o *.cpp *.hpp EASEA.png EASEA.dat EASEA.prm EASEA.mak Makefile EASEA.vcproj EASEA.csv EASEA.r EASEA.plot EASEA.pop
else
	rm -f $(TARGET) *.o *.cpp *.hpp EASEA.png EASEA.dat EASEA.prm EASEA.mak Makefile EASEA.vcproj EASEA.csv EASEA.r EASEA.plot EASEA.pop
endif

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
--printInitialPopulation=0 #Print initial population
--printFinalPopulation=0 #Print final population
--generateCSV=\GENERATE_CSV_FILE
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
