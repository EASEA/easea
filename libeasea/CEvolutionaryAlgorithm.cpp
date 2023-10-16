/*
 * CEvolutionaryAlgorithm.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 *  =======
 *  Updated on june 2022 to restore Windows compatibility
 *  TODO: switch to C++....
 */

#include "include/CEvolutionaryAlgorithm.h"
#include "config.h"

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "include/CIndividual.h"
#include "include/Parameters.h"
#include "include/CGrapher.h"
#include "include/global.h"
#include "include/CComUDPLayer.h"
#include "include/CRandomGenerator.h"
#include "include/CLogger.h"
#include "include/CLogFile.h"
#include "include/CProgressBar.h"
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
//#define INSTRUMENTED
#ifdef INSTRUMENTED
#define TIMING
#include <timing.h>
#else
#define TIME_ST(f)
#define TIME_END(f)
#define TIME_ACC(f)
#endif

using namespace std;

extern CEvolutionaryAlgorithm* EA;
void EASEABeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm);
void EASEAGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm);

extern void evale_pop_chunk(CIndividual** pop, int popSize);
extern bool INSTEAD_EVAL_STEP;
extern bool bReevaluate;
std::ofstream easena::log_file; 
easena::log_stream logg;

/*****
 * REAL CONSTRUCTOR
 */

CEvolutionaryAlgorithm::CEvolutionaryAlgorithm(Parameters* params){
	this->params = params;
	this->cstats = new CStats();

	CPopulation::initPopulation(params->selectionOperator,params->replacementOperator,params->parentReductionOperator,params->offspringReductionOperator,
        params->selectionPressure,params->replacementPressure,params->parentReductionPressure,params->offspringReductionPressure);

	this->population = new CPopulation(params->parentPopulationSize,params->offspringPopulationSize,
        params->pCrossover,params->pMutation,params->pMutationPerGene,params, this->cstats);

	this->currentGeneration = 0;
	this->reduceParents = 0;
	this->reduceOffsprings = 0;
	this->grapher = NULL;
	if(params->plotStats || params->generatePlotScript){
		string fichier = params->outputFilename;
		fichier.append(".dat");
		remove(fichier.c_str());
	}
	if(params->generatePlotScript){
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
  if(params->plotStats){
        string str = "Plotting of the evolution of ";;
        string str2 = this->params->outputFilename;
        str.append(str2);
    //this->grapher = new CGrapher((this->params->offspringPopulationSize*this->params->nbGen)+this->params->parentPopulationSize, (char*)str.c_str());
    this->grapher = new CGrapher(this->params, (char*)str.c_str());
  }


  // INITIALIZE SERVER OBJECT ISLAND MODEL
  if(params->remoteIslandModel){
    this->treatedIndividuals = 0;
    this->numberOfClients = 0;
    this->myClientNumber=0; 
    this->initializeClients();
    server = std::make_unique<CComUDPServer>(params->serverPort, !params->silentNetwork);
  }
}

/* DESTRUCTOR */
CEvolutionaryAlgorithm::~CEvolutionaryAlgorithm(){
  delete population;
}
void CEvolutionaryAlgorithm::addStoppingCriterion(CStoppingCriterion* sc){
  this->stoppingCriteria.push_back(sc);
}

/* MAIN FUNCTION TO RUN THE EVOLUTIONARY LOOP */
void CEvolutionaryAlgorithm::runEvolutionaryLoop(){
  CIndividual** elitistPopulation = NULL;
    
#ifdef INSTRUMENTED
  const char* timing_file_name = "timing.csv";
  FILE* timing_file = NULL;
  if( access(timing_file_name,W_OK)!=0 ){
    // if file does not already exist, start by describing each field
    timing_file = fopen("timing.csv","w");
    fprintf(timing_file,"gen,popSize,init,eval,breeding,reduction\n");
  }
  else{
    timing_file = fopen("timing.csv","a");
  }
  DECLARE_TIME(init);
  DECLARE_TIME_ACC(eval);
  //DECLARE_TIME_ACC(optim);
  DECLARE_TIME_ACC(breeding);
  DECLARE_TIME_ACC(reduction);  

#endif

  //std::cout << "Population initialisation (Generation 0)... "<< std::endl; 
  auto start = std::chrono::system_clock::now();
  string log_fichier_name = params->outputFilename;

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()) % 1000;

  time_t t = std::chrono::system_clock::to_time_t(start);
  std::tm * ptm = std::localtime(&t);
  char buf_start_time[32];
  if (!params->noLogFile){
    std::strftime(buf_start_time, 32, "%Y-%m-%d_%H-%M-%S", ptm);
    easena::log_file.open(log_fichier_name.c_str() + std::string("_") + std::string(buf_start_time) + std::string("-") + std::to_string(ms.count()) + std::string(".log"));
  }

  TIME_ST(init);this->initializeParentPopulation();TIME_END(init);

  TIME_ST(eval);
  if(!INSTEAD_EVAL_STEP) {
    this->population->evaluateParentPopulation();
  } else {
    evale_pop_chunk(population->parents, static_cast<int>(population->parentPopulationSize));
    this->population->currentEvaluationNb += this->params->parentPopulationSize;
  }

  if(this->params->optimise){
        population->optimiseParentPopulation();
  }
  TIME_END(eval);
  TIME_ACC(eval);

  if(this->params->printInitialPopulation){
    std::cout << *population << std::endl;
  }

  showPopulationStats(start);
  bBest = population->Best;
  currentGeneration += 1;

  //Initialize elitPopulation
  if(params->elitSize)
    elitistPopulation = (CIndividual**)malloc(params->elitSize*sizeof(CIndividual*)); 

  // EVOLUTIONARY LOOP
// auto start = std::chrono::system_clock::now();
  const int pbCounts = this->params->nbGen-1;

  const char pbComplited = '#';
  const char pbIncomplited = '-';
  easena::CProgressBar pb(pbCounts, pbComplited, pbIncomplited);

  if (this->params->printStats == 0)
      pb.init();
  while( this->allCriteria() == false){
    
    if (this->params->printStats == 0){
	if(currentGeneration % 1 == 0)
	    pb.display();
	++pb;
    }

    EASEABeginningGenerationFunction(this);
auto tmpElitSize =  params->elitSize;
auto tmpPrntReduceSize =  this->params->parentReductionSize;
auto tmpPrntReduct = params->parentReduction; 

if (bReevaluate == true){
params->elitSize = 0;
this->params->parentReductionSize = 0;
params->parentReduction = 1;
}
    //  individuals if remote island model
    if(params->remoteIslandModel && this->numberOfClients>0)
      this->sendIndividual();
    TIME_ST(breeding);
    population->produceOffspringPopulation();
    TIME_END(breeding);
    TIME_ACC(breeding);

    TIME_ST(eval);
    if(!INSTEAD_EVAL_STEP){
//      population->evaluateParentPopulation();
      population->evaluateOffspringPopulation();
    } else {
      evale_pop_chunk(population->offsprings, static_cast<int>(population->offspringPopulationSize));
    population->currentEvaluationNb += this->params->offspringPopulationSize;
    }

    if(this->params->optimise){
          population->optimiseOffspringPopulation();
    }
    TIME_END(eval);
    TIME_ACC(eval);

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
    

    TIME_ST(reduction);
//if (bReevaluate == true)
//population->reduceParentPopulation(0);//params->parentReductionSize);
    if( params->parentReduction ){
      population->reduceParentPopulation(params->parentReductionSize);
    }
    if( params->offspringReduction )
      population->reduceOffspringPopulation( params->offspringReductionSize );

    population->reduceTotalPopulation(elitistPopulation);

    TIME_END(reduction);
    TIME_ACC(reduction);

    population->sortParentPopulation();
    bBest = population->Best;
    //if( this->params->printStats  || this->params->generateCSVFile )
    showPopulationStats(start); // (always calculate stats)
    EASEAEndGenerationFunction(this);

    //Receiving individuals if cluster island model
    if(params->remoteIslandModel){
  this->receiveIndividuals();
    }

    currentGeneration += 1;
  params->elitSize = tmpElitSize ;
  this->params->parentReductionSize = tmpPrntReduceSize;
 params->parentReduction = tmpPrntReduct ;
  }

    auto end = std::chrono::system_clock::now();
 
    std::chrono::duration<double> elapsed_seconds = end-start;
    //std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (this->params->printStats == 0){
	pb.complited();
	//std::cout << "Stopping criterion is reached " << std::endl;
	std::cout << "Best fitness: " << population->Best->getFitness() << std::endl;
	std::cout << "Best individual: " << std::endl;
	 population->Best->printOn(std::cout);
    }
    else{
	
	/* Logging out the results */
	LOG_MSG(msgType::INFO, "Seed: %d", params->seed);
	LOG_MSG(msgType::INFO, "Best fitness: %ld", static_cast<long>(population->Best->getFitness()));
	LOG_MSG(msgType::INFO, "Elapsed time: %f", elapsed_seconds.count());
    }

if (!params->noLogFile){
    logg("Run configuration:");
    logg("Start time: ", std::string(buf_start_time));
    logg("Seed: ", params->seed);
    logg("Number of generations: ", params->nbGen);
    logg("Population size: ", params->parentPopulationSize);
    logg("CPU Threads number: ", params->nbCPUThreads);
    logg("Evaluation goal: ", params->minimizing);
    logg("____________________________________________________");
    logg("Special options: ");
    logg("Offspring population size: ",  params->offspringPopulationSize);
    logg("Mutation probability: ", params->pMutation);
    logg("Crossover probability: ", params->pCrossover);
    logg("Selection operator: ", (params->selectionOperator) ? params->selectionOperator->getSelectorName() : "None");
    logg("Selection pressure: ", params->selectionPressure);
    logg("Reduce parent pressure: ", params->parentReductionPressure);
    logg("Reduce offspring pressure: ", params->offspringReductionPressure);
    logg("Reduce parents operator: ", (params->parentReductionOperator) ? params->parentReductionOperator->getSelectorName() : "None");
    logg("Reduce offspring operator: ", (params->offspringReductionOperator) ? params->offspringReductionOperator->getSelectorName() : "None");
    logg("Surviving parents: ", params->parentReductionSize);
    logg("Surviving offspring: ", params->offspringReductionSize);
    logg("Replacement operator: ", (params->replacementOperator) ? params->replacementOperator->getSelectorName() : "None");
    logg("Replacement pressure: ", params->replacementPressure);
    logg("Elitism: ", params->strongElitism);
    logg("Elite size: ",params->elitSize);

    logg("____________________________________________________");
    logg("Remote island model: ");
    logg("Remote island model: ", params->remoteIslandModel);
    logg("Ip file: ", params->ipFile);
    logg("Migration probability: ", params->migrationProbability);
    logg("Server port: ", params->serverPort);
    logg("Reevaluate immigrants: ", params->reevaluateImmigrants);
    
    
    logg("_____________________________________________________");
    logg("Result: ");
    logg("Best fitness: ", population->Best->getFitness());
    logg("Best individual: ");
    logg(population->Best);
//    population->Best->printOn(std::cout);
    logg("\n");
    logg("Elapsed time: ", elapsed_seconds.count(), " s");
    logg("_____________________________________________________");
    logg("User's messages: \n");
}

/*    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";*/

  if(this->params->printFinalPopulation){
    population->sortParentPopulation();
    std::cout << *population << std::endl;
  }

  //IF SAVING THE POPULATION, ERASE THE OLD FILE
  if(params->savePopulation){
  string fichier = params->outputFilename;
  fichier.append(".pop");
    population->serializePopulation(fichier);
  }

  if(this->params->generatePlotScript /*|| !this->params->plotStats*/)
  generatePlotScript();

  if(this->params->generateRScript)
  generateRScript();
  
  if(params->elitSize)
    free(elitistPopulation);

  if(this->params->plotStats){
      delete this->grapher;
  }

#ifdef INSTRUMENTED
  COMPUTE_TIME(init);
  fprintf(timing_file,"%d,%d,%ld.%06ld,%ld.%06ld,%ld.%06ld,%ld.%06ld\n",
    currentGeneration, population->parentPopulationSize,
    init_res.tv_sec,init_res.tv_usec,
    eval_acc.tv_sec,eval_acc.tv_usec,
    breeding_acc.tv_sec,breeding_acc.tv_usec,
    reduction_acc.tv_sec,reduction_acc.tv_usec);
  fclose(timing_file);
#endif
}

void CEvolutionaryAlgorithm::showPopulationStats(std::chrono::time_point<std::chrono::system_clock> const& beginTime){

  //Calcul de la moyenne et de l'ecart type
  population->Best = population->Worst = population->parents[0];
  for(unsigned i=0; i<population->parentPopulationSize; i++){
    this->cstats->currentAverageFitness+=population->parents[i]->getFitness();

    // here we are looking for the smaller individual's fitness if we are minimizing
    // or the greatest one if we are not
    if( (params->minimizing && population->parents[i]->getFitness()<population->Best->getFitness()) ||
    (!params->minimizing && population->parents[i]->getFitness()>population->Best->getFitness()))
      population->Best=population->parents[i];

    // keep track of worst individual too, for statistical purposes
    if( (params->minimizing && population->parents[i]->getFitness() > population->Worst->getFitness()) ||
    (!params->minimizing && population->parents[i]->getFitness() < population->Worst->getFitness()))
      population->Worst=population->parents[i];

    if( params->remoteIslandModel && population->parents[i]->isImmigrant){
        //Count number of Immigrants
       this->cstats->currentNumberOfImmigrants++; 
    }
  }
  bBest = population->Best;

  this->cstats->currentAverageFitness/=population->parentPopulationSize;

  for(unsigned i=0; i<population->parentPopulationSize; i++){
    this->cstats->currentStdDev+=(population->parents[i]->getFitness()-this->cstats->currentAverageFitness)*(population->parents[i]->getFitness()-this->cstats->currentAverageFitness);
  }
  this->cstats->currentStdDev/=population->parentPopulationSize;
  this->cstats->currentStdDev=sqrt(this->cstats->currentStdDev);

  auto elapsed = std::chrono::system_clock::now() - beginTime;
  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  auto elapsed_s = static_cast<float>(elapsed_us) / 1e6f;

  //Affichage
  

  if(params->printStats){
    if(currentGeneration==0){

      printf("--------------------------------------------------------------------------------------------------------\n");
      printf("|GENER.|    ELAPSED    |    PLANNED    |     ACTUAL    | BEST INDIVIDUAL |   AVG   |  STAND  |  WORST  |\n");
      printf("|NUMBER|     TIME      | EVALUATION NB | EVALUATION NB |     FITNESS     | FITNESS |   DEV   | FITNESS |\n");
      printf("--------------------------------------------------------------------------------------------------------\n");
    }
    printf("%7u %14.3fs %15u %15u % .10e % .2e % .2e % .2e\n",currentGeneration,elapsed_s,(int)population->currentEvaluationNb,(int)population->realEvaluationNb,population->Best->getFitness(),this->cstats->currentAverageFitness,this->cstats->currentStdDev, population->Worst->getFitness());
  }

  if((this->params->plotStats && this->grapher->valid) || this->params->generatePlotScript){
  FILE *f;
  string fichier = params->outputFilename;
  fichier.append(".dat");
  f = fopen(fichier.c_str(),"a"); //ajouter .csv
  if(f!=NULL){
    if(currentGeneration==0)
      fprintf(f,"#GEN\tTIME\t\tEVAL\tBEST\t\tAVG\t\tSTDDEV\t\tWORST\n\n");
    fprintf(f,"%u\t%2.6f\t%u\t%.2e\t%.2e\t%.2e\t%.2e\n",currentGeneration,elapsed_s,population->currentEvaluationNb,population->Best->getFitness(),this->cstats->currentAverageFitness,population->Worst->getFitness(),this->cstats->currentStdDev);
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
    fprintf(f, "Run configuration:\nNB_GEN = %i POP_SIZE = %i OFFSPRING_SIZE = %i MUT_PROB = %f  XOVER_PROB = %f \n\n", (*EZ_NB_GEN), EZ_POP_SIZE, OFFSPRING_SIZE, (*pEZ_MUT_PROB), (*pEZ_XOVER_PROB));

    fprintf(f,"GEN,TIME,EVAL,BEST,AVG,STDDEV,WORST\n");
    fprintf(f,"%u,%2.6f,%u,%.2e,%.2e,%.2e,%.2e\n",currentGeneration,elapsed_s,population->currentEvaluationNb,population->Best->getFitness(),this->cstats->currentAverageFitness,this->cstats->currentStdDev, population->Worst->getFitness());
    fclose(f);
        }
  }
  //print grapher
  if(this->params->plotStats && this->grapher->valid){
  //if(currentGeneration==0)
  //  fprintf(this->grapher->fWrit,"plot \'%s.dat\' using 3:4 t \'Best Fitness\' w lines ls 1, \'%s.dat\' using 3:5 t  \'Average\' w lines ls 4, \'%s.dat\' using 3:6 t \'StdDev\' w lines ls 3\n", params->outputFilename,params->outputFilename,params->outputFilename);
  //else
  //  fprintf(this->grapher->fWrit,"replot\n");
    fprintf(this->grapher->fWrit,"add coordinate:%u;%f;%f;%f\n",population->
currentEvaluationNb, population->Best->fitness, this->cstats->currentAverageFitness, this->cstats->currentStdDev);

    if(this->params->remoteIslandModel){
        if(population->Best->isImmigrant){
            fprintf(this->grapher->fWrit,"set immigrant\n");
        }
        fprintf(this->grapher->fWrit,"add stat:%u;%u;%u\n",currentGeneration, this->cstats->currentNumberOfImmigrants, this->cstats->currentNumberOfImmigrantReproductions);
    }
    if(currentGeneration==0){
        fprintf(this->grapher->fWrit,"paint\n");
    }
    else{
        fprintf(this->grapher->fWrit,"repaint\n");
    }
  fflush(this->grapher->fWrit);
 }

  params->timeCriterion->setElapsedTime(elapsed_s);

  //Reset Current Gen Stats
  this->cstats->resetCurrentStats();
}

  
//REMOTE ISLAND MODEL FUNCTIONS
void CEvolutionaryAlgorithm::initializeClients(){
    this->refreshClient();

    if(this->numberOfClients<=0){
    cout << "***WARNING***\nNo islands to communicate with." << endl;
  }
}

void CEvolutionaryAlgorithm::refreshClient(){
    this->Clients = parse_file(this->params->ipFile, this->params->serverPort, !params->silentNetwork);
    this->numberOfClients = Clients.size();

    cout << "ip file : " << this->params->ipFile << " contains " << numberOfClients << " client ip(s)" << endl;
}

void CEvolutionaryAlgorithm::sendIndividual(){
  //Sending an individual every n generations 
  if(random(0.0,1.0)<=params->migrationProbability){
  //if((this->currentGeneration+this->myClientNumber)%3==0 && this->currentGeneration!=0){
    this->population->selectionOperator->initialize(this->population->parents, params->selectionPressure, this->population->actualParentPopulationSize);
    //unsigned index = this->population->selectionOperator->selectNext(this->population->actualParentPopulationSize);
  
    //selecting a client randomly
    int client = random(0, static_cast<int>(this->numberOfClients));

    this->Clients[client]->send(bBest);
  }
}

void CEvolutionaryAlgorithm::receiveIndividuals(){

  //Checking every generation for received individuals
  if(server->has_data()){
    CSelectionOperator *antiTournament = getSelectionOperator("Tournament",!this->params->minimizing);   


    //Treating all the individuals before continuing
    while(server->has_data()){
      //selecting the individual to erase
      antiTournament->initialize(this->population->parents, 7, this->population->actualParentPopulationSize);
      unsigned index = antiTournament->selectNext(this->population->actualParentPopulationSize);
      
      //We're selecting the worst element to replace
      server->consume_into(this->population->parents[index]);
      int reeval = params->reevaluateImmigrants;
      // Reevaluate individaul if the flag reevaluateImmigrants == 1	
      if (bReevaluate == true){ params->reevaluateImmigrants = 1;}
      if (params->reevaluateImmigrants == 1){
	    this->population->realEvaluationNb++;
	    this->population->parents[index]->evaluate_wrapper(true);
      }
      params->reevaluateImmigrants = reeval;
      //TAG THE INDIVIDUAL AS IMMIGRANT
      this->population->parents[index]->isImmigrant = true;
    }
  }
}

void CEvolutionaryAlgorithm::outputGraph(){
    /*    fprintf(this->grapher->fWrit,"set term png\n");
        fprintf(this->grapher->fWrit,"set output \"%s\"\n",params->plotOutputFilename);
  fprintf(this->grapher->fWrit,"set xrange[0:%d]\n",(int)population->currentEvaluationNb);
  fprintf(this->grapher->fWrit,"set xlabel \"Number of Evaluations\"\n");
        fprintf(this->grapher->fWrit,"set ylabel \"Fitness\"\n");
        fprintf(this->grapher->fWrit,"replot \n");
  fflush(this->grapher->fWrit);*/
}

void CEvolutionaryAlgorithm::generatePlotScript(){
  FILE* f;
  string fichier = this->params->outputFilename;
  fichier.append(".plot");
  f = fopen(fichier.c_str(),"a");
  fprintf(f,"set term png\n");
  fprintf(f,"set output \"%s\"\n",params->plotOutputFilename);
  fprintf(f,"set xrange[0:%d]\n",(int)population->currentEvaluationNb);
  fprintf(f,"set xlabel \"Number of Evaluations\"\n");
        fprintf(f,"set ylabel \"Fitness\"\n");
  fprintf(f,"plot \'%s.dat\' using 3:4 t \'Best Fitness\' w lines, \'%s.dat\' using 3:5 t  \'Average\' w lines, \'%s.dat\' using 3:6 t \'StdDev\' w lines\n", params->outputFilename.c_str(),params->outputFilename.c_str(),params->outputFilename.c_str());
  fclose(f);  
}

void CEvolutionaryAlgorithm::generateRScript(){
  FILE* f;
  string fichier = this->params->outputFilename;
  fichier.append(".r");
  f=fopen(fichier.c_str(),"a");
  fprintf(f,"#Plotting for R\n"),
  fprintf(f,"png(\"%s\")\n",params->plotOutputFilename);
  fprintf(f,"data <- read.table(\"./%s.csv\",sep=\",\")\n",params->outputFilename.c_str());
  fprintf(f,"plot(0, type = \"n\", main = \"Plot Title\", xlab = \"Number of Evaluations\", ylab = \"Fitness\", xlim = c(0,%d) )\n",(int)population->currentEvaluationNb);
  fprintf(f,"grid() # add grid\n");
  fprintf(f,"lines(data[,3], data[,4], lty = 1) #draw first dataset\n");
  fprintf(f,"lines(data[,3], data[,5], lty = 2) #draw second dataset\n");
  fprintf(f,"lines(data[,3], data[,6], lty = 3) #draw third dataset\n");
  fprintf(f,"legend(\"topright\", c(\"Best Fitness\", \"Average\", \"StdDev\"), lty = c(1, 2, 3) )\n");
  fclose(f);
  
}

bool CEvolutionaryAlgorithm::allCriteria(){

  for( unsigned i=0 ; i<stoppingCriteria.size(); i++ ){
    if( stoppingCriteria.at(i)->reached() ){
      //std::cout << "Stopping criterion reached" << std::endl;
      return true;
    }
  }
  return false;
}
