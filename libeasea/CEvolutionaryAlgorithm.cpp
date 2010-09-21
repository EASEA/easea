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

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string>
#include "include/CIndividual.h"
#include "include/Parameters.h"
#include "include/CGnuplot.h"
#include "include/global.h"
#include <stdio.h>

using namespace std;

extern CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;
void EASEABeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm);

extern void evale_pop_chunk(CIndividual** pop, int popSize);
extern bool INSTEAD_EVAL_STEP;
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
	if(params->plotStats || params->generateGnuplotScript){
		string fichier = params->outputFilename;
		fichier.append(".dat");
		remove(fichier.c_str());
	}
	if(params->generateGnuplotScript){
		string fichier = params->outputFilename;
		fichier.append(".plot");
		remove(fichier.c_str());
	}
	if(params->generateRScript || params->generateCSVFile){
		string fichier = params->outputFilename;
		fichier.append(".csv");
		remove(fichier.c_str());
	}
	if(params->generateRScript){
		string fichier = params->outputFilename;
		fichier.append(".r");
		remove(fichier.c_str());
	}
	#ifdef __linux__
	if(params->plotStats){
		this->gnuplot = new CGnuplot((this->params->offspringPopulationSize*this->params->nbGen)+this->params->parentPopulationSize);
	}
	#endif
}

CEvolutionaryAlgorithm::~CEvolutionaryAlgorithm(){
  delete population;
}
void CEvolutionaryAlgorithm::addStoppingCriterion(CStoppingCriterion* sc){
  this->stoppingCriteria.push_back(sc);
}



void CEvolutionaryAlgorithm::runEvolutionaryLoop(){
  CIndividual** elitistPopulation;

#ifdef WIN32
   clock_t begin(clock());
#else
  struct timeval begin;
  gettimeofday(&begin,0);
#endif

  std::cout << "Parent's population initializing "<< std::endl;
  this->initializeParentPopulation();

  if(!INSTEAD_EVAL_STEP)
    this->population->evaluateParentPopulation();
  else
    evale_pop_chunk(population->parents, population->parentPopulationSize);

  if(this->params->optimise){
        population->optimiseParentPopulation();
  }

  this->population->currentEvaluationNb += this->params->parentPopulationSize;
  if(this->params->printInitialPopulation){
  	std::cout << *population << std::endl;
  }

  showPopulationStats(begin);
  currentGeneration += 1;

  //Initialize elitPopulation
  if(params->elitSize)
		elitistPopulation = (CIndividual**)malloc(params->elitSize*sizeof(CIndividual*));	

  while( this->allCriteria() == false){

    EASEABeginningGenerationFunction(this);

    population->produceOffspringPopulation();

    if(!INSTEAD_EVAL_STEP)
      population->evaluateOffspringPopulation();
    else
      evale_pop_chunk(population->offsprings, population->offspringPopulationSize);
    population->currentEvaluationNb += this->params->offspringPopulationSize;

    if(this->params->optimise){
          population->optimiseOffspringPopulation();
    }

    EASEAGenerationFunctionBeforeReplacement(this);

    /* ELITISM */
    if(params->elitSize && this->params->parentPopulationSize>=params->elitSize){
	/* STRONG ELITISM */
	if(params->strongElitism){
		population->strongElitism(params->elitSize, population->parents, this->params->parentPopulationSize, elitistPopulation, params->elitSize);
		population->actualParentPopulationSize -= params->elitSize;
	}
	/* WEAK ELITISM */
	else{
		population->weakElitism(params->elitSize, population->parents, population->offsprings, &(population->actualParentPopulationSize), &(population->actualOffspringPopulationSize), elitistPopulation, params->elitSize);
	}
	
    }

    if( params->parentReduction )
      population->reduceParentPopulation(params->parentReductionSize);

    if( params->offspringReduction )
      population->reduceOffspringPopulation( params->offspringReductionSize );

    population->reduceTotalPopulation(elitistPopulation);

    population->sortParentPopulation();
    if( this->params->printStats  || this->params->generateCSVFile )
      showPopulationStats(begin);
    bBest = population->Best;
    EASEAEndGenerationFunction(this);

    currentGeneration += 1;
  }
#ifdef __linux__
  if(this->params->plotStats && this->gnuplot->valid){
  	outputGraph();
  	delete this->gnuplot;
  }
#endif

  if(this->params->printFinalPopulation){
  	population->sortParentPopulation();
  	std::cout << *population << std::endl;
  }

  if(this->params->generateGnuplotScript || !this->params->plotStats)
	generateGnuplotScript();

  if(this->params->generateRScript)
	generateRScript();
  
  if(params->elitSize)
  	free(elitistPopulation);
}


#ifdef WIN32
void CEvolutionaryAlgorithm::showPopulationStats(clock_t beginTime){
#else
void CEvolutionaryAlgorithm::showPopulationStats(struct timeval beginTime){
#endif
  
  currentAverageFitness=0.0;
  currentSTDEV=0.0;

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

#ifdef WIN32
  clock_t end(clock());
  double duration;
  duration = (double)(end-beginTime)/CLOCKS_PER_SEC;
#else
  struct timeval end, res;
  gettimeofday(&end,0);
  timersub(&end,&beginTime,&res);
#endif

  //Affichage
  if(params->printStats){
	  if(currentGeneration==0)
	    printf("GEN\tTIME\t\tEVAL\tBEST\t\tAVG\t\tSTDEV\n\n");
#ifdef WIN32
            printf("%lu\t%2.6f\t%lu\t%.15e\t%.15e\t%.15e\n",currentGeneration,duration,population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV);
#else
	    printf("%d\t%ld.%06ld\t%d\t%.15e\t%.15e\t%.15e\n",currentGeneration,res.tv_sec,res.tv_usec,population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV);
#endif
	  //printf("%lu\t%ld.%06ld\t%lu\t%f\t%f\t%f\n",currentGeneration,res.tv_sec,res.tv_usec,population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV);
  }

  if((this->params->plotStats && this->gnuplot->valid) || this->params->generateGnuplotScript){
 	FILE *f;
	string fichier = params->outputFilename;
	fichier.append(".dat");
 	f = fopen(fichier.c_str(),"a"); //ajouter .csv
	if(f!=NULL){
	  if(currentGeneration==0)
		fprintf(f,"#GEN\tTIME\t\tEVAL\tBEST\t\tAVG\t\tSTDEV\n\n");
#ifdef WIN32
          fprintf(f,"%lu\t%2.6f\t%lu\t%.15e\t%.15e\t%.15e\n",currentGeneration,duration,population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV);
#else
	  fprintf(f,"%d\t%ld.%06ld\t%d\t%.15e\t%.15e\t%.15e\n",currentGeneration,res.tv_sec,res.tv_usec,population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV);
#endif
	  fclose(f);
        }
  }
  if(params->generateCSVFile || params->generateRScript){ //Generation du fichier CSV;
 	FILE *f;
	string fichier = params->outputFilename;
	fichier.append(".csv");
 	f = fopen(fichier.c_str(),"a"); //ajouter .csv
	if(f!=NULL){
	  if(currentGeneration==0)
		fprintf(f,"GEN,TIME,EVAL,BEST,AVG,STDEV\n");
#ifdef WIN32
          fprintf(f,"%lu,%2.6f,%lu,%.15e,%.15e,%.15e\n",currentGeneration,duration,population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV);
#else
	  fprintf(f,"%d,%ld.%06ld,%d,%f,%f,%f\n",currentGeneration,res.tv_sec,res.tv_usec,population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV);
#endif
	  fclose(f);
        }
  }
  //print Gnuplot
  #ifdef __linux__
  if(this->params->plotStats && this->gnuplot->valid){
	if(currentGeneration==0)
		fprintf(this->gnuplot->fWrit,"plot \'%s.dat\' using 3:4 t \'Best Fitness\' w lines ls 1, \'%s.dat\' using 3:5 t  \'Average\' w lines ls 4, \'%s.dat\' using 3:6 t \'StdDev\' w lines ls 3\n", params->outputFilename,params->outputFilename,params->outputFilename);
	else
		fprintf(this->gnuplot->fWrit,"replot\n");
	fflush(this->gnuplot->fWrit);
 }
 
#endif

  params->timeCriterion->setElapsedTime(res.tv_sec);
}

void CEvolutionaryAlgorithm::outputGraph(){
      	fprintf(this->gnuplot->fWrit,"set term png\n");
      	fprintf(this->gnuplot->fWrit,"set output \"%s\"\n",params->plotOutputFilename);
	fprintf(this->gnuplot->fWrit,"set xrange[0:%d]\n",population->currentEvaluationNb);
	fprintf(this->gnuplot->fWrit,"set xlabel \"Number of Evaluations\"\n");
        fprintf(this->gnuplot->fWrit,"set ylabel \"Fitness\"\n");
        fprintf(this->gnuplot->fWrit,"replot \n");
	fflush(this->gnuplot->fWrit);
}

void CEvolutionaryAlgorithm::generateGnuplotScript(){
	FILE* f;
	string fichier = this->params->outputFilename;
	fichier.append(".plot");
	f = fopen(fichier.c_str(),"a");
	fprintf(f,"set term png\n");
	fprintf(f,"set output \"%s\"\n",params->plotOutputFilename);
	fprintf(f,"set xrange[0:%d]\n",population->currentEvaluationNb);
	fprintf(f,"set xlabel \"Number of Evaluations\"\n");
        fprintf(f,"set ylabel \"Fitness\"\n");
	fprintf(f,"plot \'%s.dat\' using 3:4 t \'Best Fitness\' w lines, \'%s.dat\' using 3:5 t  \'Average\' w lines, \'%s.dat\' using 3:6 t \'StdDev\' w lines\n", params->outputFilename,params->outputFilename,params->outputFilename);
	fclose(f);	
}

void CEvolutionaryAlgorithm::generateRScript(){
	FILE* f;
	string fichier = this->params->outputFilename;
	fichier.append(".r");
	f=fopen(fichier.c_str(),"a");
	fprintf(f,"#Plotting for R\n"),
	fprintf(f,"png(\"%s\")\n",params->plotOutputFilename);
	fprintf(f,"data <- read.table(\"./%s.csv\",sep=\",\")\n",params->outputFilename);
	fprintf(f,"plot(0, type = \"n\", main = \"Plot Title\", xlab = \"Number of Evaluations\", ylab = \"Fitness\", xlim = c(0,%d) )\n",population->currentEvaluationNb);
	fprintf(f,"grid() # add grid\n");
	fprintf(f,"lines(data[,3], data[,4], lty = 1) #draw first dataset\n");
	fprintf(f,"lines(data[,3], data[,5], lty = 2) #draw second dataset\n");
	fprintf(f,"lines(data[,3], data[,6], lty = 3) #draw third dataset\n");
	fprintf(f,"legend(\"topright\", c(\"Best Fitness\", \"Average\", \"StdDev\"), lty = c(1, 2, 3) )\n");
	fclose(f);
	
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

