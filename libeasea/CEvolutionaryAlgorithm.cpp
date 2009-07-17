#ifdef WIN32
#pragma comment(lib, "WinMM.lib")
#endif
/*
 * CEvolutionaryAlgorithm.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include "include/CEvolutionaryAlgorithm.h"
#include <string>
#ifndef WIN32
#include <sys/time.h>
#endif
#ifdef WIN32
#include <windows.h>
#endif
#include <time.h>
#include <math.h>
#include "include/CIndividual.h"
#include "include/Parameters.h"
#include "include/CGnuplot.h"
#include "include/global.h"

extern CRandomGenerator* globalRandomGenerator;
void EASEABeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
/**
 * @DEPRECATED the next contructor has to be used instead of this one.
 */
CEvolutionaryAlgorithm::CEvolutionaryAlgorithm( size_t parentPopulationSize,
					      size_t offspringPopulationSize,
					      float selectionPressure, float replacementPressure, float parentReductionPressure, float offspringReductionPressure,
					      CSelectionOperator* selectionOperator, CSelectionOperator* replacementOperator,
					      CSelectionOperator* parentReductionOperator, CSelectionOperator* offspringReductionOperator,
					      float pCrossover, float pMutation,
					      float pMutationPerGene){

  CRandomGenerator* rg = globalRandomGenerator;

  CSelectionOperator* so = selectionOperator;
  CSelectionOperator* ro = replacementOperator;

  //CIndividual::initRandomGenerator(rg);
  CPopulation::initPopulation(so,ro,parentReductionOperator,offspringReductionOperator,selectionPressure,replacementPressure,parentReductionPressure,offspringReductionPressure);

  this->population = new CPopulation(parentPopulationSize,offspringPopulationSize,
				    pCrossover,pMutation,pMutationPerGene,rg,NULL);

  this->currentGeneration = 0;

  this->reduceParents = 0;
  this->reduceOffsprings = 0;
}

CEvolutionaryAlgorithm::CEvolutionaryAlgorithm(Parameters* params){
	this->params = params;

	CPopulation::initPopulation(params->selectionOperator,params->replacementOperator,params->parentReductionOperator,params->offspringReductionOperator,
			params->selectionPressure,params->replacementPressure,params->parentReductionPressure,params->offspringReductionPressure);

	this->population = new CPopulation(params->parentPopulationSize,params->offspringPopulationSize,
			params->pCrossover,params->pMutation,params->pMutationPerGene,params->randomGenerator,params);

	this->currentGeneration = 0;

	this->reduceParents = 0;
	this->reduceOffsprings = 0;
	this->gnuplot = NULL;
	#ifdef __linux__
	remove("stats.dat");
	if(params->plotStats){
		this->gnuplot = new CGnuplot();
		if(!this->gnuplot->valid)
			printf("Attention, erreur lors de l'utilisation de gnulplot, l'algo va continuer sans plot");
	}
	#endif
}

void CEvolutionaryAlgorithm::addStoppingCriterion(CStoppingCriterion* sc){
  this->stoppingCriteria.push_back(sc);
}



void CEvolutionaryAlgorithm::runEvolutionaryLoop(){
  std::vector<CIndividual*> tmpVect;


  std::cout << "Parent's population initializing "<< std::endl;
  this->initializeParentPopulation();
  this->population->evaluateParentPopulation();
  this->population->currentEvaluationNb += this->params->parentPopulationSize;
  if(this->params->printInitialPopulation){
  	std::cout << *population << std::endl;
  }


  struct timeval begin;
  gettimeofday(&begin,NULL);

  while( this->allCriteria() == false ){

    EASEABeginningGenerationFunction(this);
    population->produceOffspringPopulation();

    population->evaluateOffspringPopulation();

    population->currentEvaluationNb += this->params->offspringPopulationSize;

	//EASEAEndGenerationFunction(this);

	if( params->parentReduction )
      population->reduceParentPopulation(params->parentReductionSize);



	if( params->offspringReduction )
      population->reduceOffspringPopulation( params->offspringReductionSize );

    population->reduceTotalPopulation();

	EASEAEndGenerationFunction(this);


    showPopulationStats(begin);
    currentGeneration += 1;
  }
  population->sortParentPopulation();

  if(this->params->printFinalPopulation)
  	std::cout << *population << std::endl;
  std::cout << "Generation : " << currentGeneration << std::endl;

  #ifdef __linux__
  if(this->params->plotStats && this->gnuplot->valid)
  	delete this->gnuplot;
  #endif
}


void CEvolutionaryAlgorithm::showPopulationStats(struct timeval beginTime){

  float currentAverageFitness=0.0;
  float currentSTDEV=0.0;

  //Calcul de la moyenne et de l'ecart type
  population->Best=population->parents[0];

  for(size_t i=0; i<population->parentPopulationSize; i++){
    currentAverageFitness+=population->parents[i]->getFitness();

    // here we are looking for the smaller individual's fitness if we are minimizing
    // or the greatest one if we are not
    if( (params->minimizing && population->parents[i]->getFitness()<population->Best->getFitness()) ||
    (!params->minimizing && population->parents[i]->getFitness()>population->Best->getFitness()))
      population->Best=population->parents[i];
  }

  currentAverageFitness/=population->parentPopulationSize;

  for(size_t i=0; i<population->parentPopulationSize; i++){
    currentSTDEV+=(population->parents[i]->getFitness()-currentAverageFitness)*(population->parents[i]->getFitness()-currentAverageFitness);
  }
  currentSTDEV/=population->parentPopulationSize;
  currentSTDEV=sqrt(currentSTDEV);

  struct timeval end, res;
  gettimeofday(&end,0);
  timersub(&end,&beginTime,&res);

  //Affichage
  if(params->printStats){
	  if(currentGeneration==0)
	    printf("GEN\tTIME\t\tEVAL\tBEST\t\tAVG\t\tSTDEV\n\n");
	  printf("%lu\t%ld.%06ld\t%lu\t%.15e\t%.15e\t%.15e\n",currentGeneration,res.tv_sec,res.tv_usec,population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV);
  }

  //print Gnuplot
  #ifdef __linux__
  if(this->params->plotStats && this->gnuplot->valid){
 	 FILE *f;
 	 f = fopen("stats.dat","a");
 	 if(f!=NULL){
 	 	fprintf(f,"%lu\t%d.%06d\t%lu\t%.15e\t%.15e\t%.15e\n",currentGeneration,res.tv_sec,res.tv_usec,population->currentEvaluationNb, population->Best->getFitness(),currentAverageFitness,currentSTDEV);
		fclose(f);
	}
	if(currentGeneration==0)
		fprintf(this->gnuplot->fWrit,"plot \'stats.dat\' using 3:4 t \'Best Fitness\' w lines, \'stats.dat\' using 3:5 t  \'Average\' w lines, \'stats.dat\' using 3:6 t \'StdDev\' w lines\n");
	else if((currentGeneration+1)==(*EZ_NB_GEN)){
		fprintf(this->gnuplot->fWrit,"set term png\n");
		fprintf(this->gnuplot->fWrit,"set output \"plot.png\"\n");
		fprintf(this->gnuplot->fWrit,"replot \n");
	}
	else
		fprintf(this->gnuplot->fWrit,"replot\n");
	fflush(this->gnuplot->fWrit);
 }
#endif
}

bool CEvolutionaryAlgorithm::allCriteria(){

  for( size_t i=0 ; i<stoppingCriteria.size(); i++ ){
    if( stoppingCriteria.at(i)->reached() ){
      std::cout << "Stopping criterion reached : " << i << std::endl;
      return true;
    }
  }
  return false;
}

#ifdef WIN32
int gettimeofday
      (struct timeval* tp, void* tzp) {
    DWORD t;
    t = timeGetTime();
    tp->tv_sec = t / 1000;
    tp->tv_usec = t % 1000;
    /* 0 indicates success. */
    return 0;
}

void timersub( const timeval * tvp, const timeval * uvp, timeval* vvp )
{
    vvp->tv_sec = tvp->tv_sec - uvp->tv_sec;
    vvp->tv_usec = tvp->tv_usec - uvp->tv_usec;
    if( vvp->tv_usec < 0 )
    {
       --vvp->tv_sec;
       vvp->tv_usec += 1000000;
    }
} 
#endif
