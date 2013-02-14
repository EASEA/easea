//#ifdef WIN32
//#pragma comment(lib, "WinMM.lib")
//#endif
/*
 * CEvolutionaryAlgorithm.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include "include/CEvolutionaryAlgorithm.h"
#ifndef WIN32
#include <sys/time.h>
#endif
#ifdef WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "include/CIndividual.h"
#include "include/Parameters.h"
#include "include/CGrapher.h"
#include "include/global.h"
//#include "include/CComUDPLayer.h"
#include "include/CComGridUdpServer.h"
#include "include/CRandomGenerator.h"
#include <stdio.h>
#include <sstream>
#include <fstream>

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
/*CEvolutionaryAlgorithm::CEvolutionaryAlgorithm( unsigned parentPopulationSize,
					      unsigned offspringPopulationSize,
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
  
  // INITIALIZE SERVER OBJECT ISLAND MODEL
  if(params->remoteIslandModel){
  	this->server = new CComUDPServer(2909,0);
	this->treatedIndividuals = 0;
	this->numberOfClients = 0;

	this->initializeClients();
  }
}*/

/*****
 * REAL CONSTRUCTOR
 */
CEvolutionaryAlgorithm::CEvolutionaryAlgorithm(Parameters* params){
	this->params = params;
    this->cstats = new CStats();

	CPopulation::initPopulation(params->selectionOperator,params->replacementOperator,params->parentReductionOperator,params->offspringReductionOperator,
			params->selectionPressure,params->replacementPressure,params->parentReductionPressure,params->offspringReductionPressure);

	this->population = new CPopulation(params->parentPopulationSize,params->offspringPopulationSize,
			params->pCrossover,params->pMutation,params->pMutationPerGene,params->randomGenerator,params, this->cstats);

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
	#ifndef WIN32 
	if(params->plotStats){
        string str = "Plotting of the evolution of ";;
        string str2 = this->params->outputFilename;
        str.append(str2);
		//this->grapher = new CGrapher((this->params->offspringPopulationSize*this->params->nbGen)+this->params->parentPopulationSize, (char*)str.c_str());
		this->grapher = new CGrapher(this->params, (char*)str.c_str());
	}
	#endif


	// INITIALIZE SERVER OBJECT ISLAND MODEL
	if(params->remoteIslandModel){
		this->treatedIndividuals = 0;
		this->treatedFileIndividuals = 0;
		this->numberOfClients = 0;
		this->myClientNumber=0;	
		fileserver = new CComGridUDPServer(params->working_path,params->expId, &receivedIndividuals, (unsigned short int) params->serverPort, 1);
		this->initializeClients();
		//if(params->remoteIslandModel)
		//server = new CComUDPServer(params->serverPort,&receivedIndividuals,0); //1 if debug
		
	}
}

/* DESTRUCTOR */
CEvolutionaryAlgorithm::~CEvolutionaryAlgorithm(){
	delete population;
	
        if(this->params->remoteIslandModel){
                //delete this->server;
		delete this->fileserver;
                /*if(this->numberOfClients>0){
                        for(int i=0; (unsigned)i<this->numberOfClients; i++)
                                delete this->Clients[i];
                        delete [] this->Clients;
                }*/
        }
        delete cstats;
        
}
void CEvolutionaryAlgorithm::addStoppingCriterion(CStoppingCriterion* sc){
  this->stoppingCriteria.push_back(sc);
}

/* MAIN FUNCTION TO RUN THE EVOLUTIONARY LOOP */
void CEvolutionaryAlgorithm::runEvolutionaryLoop(){
  CIndividual** elitistPopulation;

#ifdef WIN32
   clock_t begin(clock());
#else
  struct timeval begin;
  gettimeofday(&begin,0);
#endif

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

  std::cout << "Population initialisation (Generation 0)... "<< std::endl;

  TIME_ST(init);this->initializeParentPopulation();TIME_END(init);

  TIME_ST(eval);
  if(!INSTEAD_EVAL_STEP)
    this->population->evaluateParentPopulation();
  else
    evale_pop_chunk(population->parents, population->parentPopulationSize);

  if(this->params->optimise){
        population->optimiseParentPopulation();
  }
  TIME_END(eval);
  TIME_ACC(eval);

  this->population->currentEvaluationNb += this->params->parentPopulationSize;
  if(this->params->printInitialPopulation){
  	std::cout << *population << std::endl;
  }

  showPopulationStats(begin);
  bBest = population->Best;
  currentGeneration += 1;

  //Initialize elitPopulation
  if(params->elitSize)
		elitistPopulation = (CIndividual**)malloc(params->elitSize*sizeof(CIndividual*));	

  // EVOLUTIONARY LOOP
  while( this->allCriteria() == false){

    EASEABeginningGenerationFunction(this);

    // Sending individuals if remote island model
    if(params->remoteIslandModel && this->numberOfClients>0)
	    this->sendIndividual();
    TIME_ST(breeding);
    population->produceOffspringPopulation();
    TIME_END(breeding);
    TIME_ACC(breeding);

    TIME_ST(eval);
    if(!INSTEAD_EVAL_STEP)
      population->evaluateOffspringPopulation();
    else
      evale_pop_chunk(population->offsprings, population->offspringPopulationSize);
    population->currentEvaluationNb += this->params->offspringPopulationSize;

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
    if( params->parentReduction )
      population->reduceParentPopulation(params->parentReductionSize);

    if( params->offspringReduction )
      population->reduceOffspringPopulation( params->offspringReductionSize );

    population->reduceTotalPopulation(elitistPopulation);
    TIME_END(reduction);
    TIME_ACC(reduction);

    population->sortParentPopulation();
    //if( this->params->printStats  || this->params->generateCSVFile )
    showPopulationStats(begin); // (always calculate stats)
    bBest = population->Best;
    EASEAEndGenerationFunction(this);

    //Receiving individuals if cluster island model
    if(params->remoteIslandModel){
	
	//fileserver->refresh_worker_list();
	this->refreshClient();
	this->receiveIndividuals();
    }
    
    currentGeneration += 1;
  }
//#ifdef __linux__
  //if(this->params->plotStats && this->grapher->valid){
  	//outputGraph();
  	//delete this->grapher;
  //}
//#endif

  if(this->params->printFinalPopulation){
  	population->sortParentPopulation();
  	std::cout << *population << std::endl;
  }

  //IF SAVING THE POPULATION, ERASE THE OLD FILE
  if(params->savePopulation){
	
	string fichier = params->outputFilename;
	fichier.append(".pop");
	remove(fichier.c_str());
  	population->serializePopulation();
  }

  if(this->params->generatePlotScript || !this->params->plotStats)
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


#ifdef WIN32
void CEvolutionaryAlgorithm::showPopulationStats(clock_t beginTime){
#else
void CEvolutionaryAlgorithm::showPopulationStats(struct timeval beginTime){
#endif

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

  this->cstats->currentAverageFitness/=population->parentPopulationSize;

  for(unsigned i=0; i<population->parentPopulationSize; i++){
    this->cstats->currentStdDev+=(population->parents[i]->getFitness()-this->cstats->currentAverageFitness)*(population->parents[i]->getFitness()-this->cstats->currentAverageFitness);
  }
  this->cstats->currentStdDev/=population->parentPopulationSize;
  this->cstats->currentStdDev=sqrt(this->cstats->currentStdDev);

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
	    printf("GEN\tTIME\t\tEVAL\t\tBEST      AVG       STDDEV    WORST\n\n");
#ifdef WIN32
	  printf("%lu\t%2.6f\t%lu\t%.2e\t%.2e\t%.2e\t%.2e\n",currentGeneration,duration,population->currentEvaluationNb,population->Best->getFitness(),this->cstats->currentAverageFitness,this->cstats->currentStdDev, population->Worst->getFitness());
#else
	    //printf("%d\t%ld.%01ld\t%d\t%.2e\t%.2e\t%.2e\t%.2e\n",(int)currentGeneration,res.tv_sec,res.tv_usec,(int)population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV, population->Worst->getFitness());
	    printf("%d\t%ld.%d\t\t%d\t\t%.2e  %.2e  %.2e  %.2e\n",(int)currentGeneration,res.tv_sec,(int)res.tv_usec/10000,(int)population->currentEvaluationNb,population->Best->getFitness(),this->cstats->currentAverageFitness,this->cstats->currentStdDev, population->Worst->getFitness());
#endif
  }

  if((this->params->plotStats && this->grapher->valid) || this->params->generatePlotScript){
 	FILE *f;
	string fichier = params->outputFilename;
	fichier.append(".dat");
 	f = fopen(fichier.c_str(),"a"); //ajouter .csv
	if(f!=NULL){
	  if(currentGeneration==0)
	    fprintf(f,"#GEN\tTIME\t\tEVAL\tBEST\t\tAVG\t\tSTDDEV\t\tWORST\n\n");
#ifdef WIN32
	  fprintf(f,"%lu\t%2.6f\t%lu\t%.2e\t%.2e\t%.2e\t%.2e\n",currentGeneration,duration,population->currentEvaluationNb,population->Best->getFitness(),this->cstats->currentAverageFitness,this->cstats->currentStdDev, population->Worst->getFitness());
#else
	    //printf("%d\t%ld.%01ld\t%d\t%.2e\t%.2e\t%.2e\t%.2e\n",(int)currentGeneration,res.tv_sec,res.tv_usec,(int)population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV, population->Worst->getFitness());
	    fprintf(f,"%d\t%ld.%d\t\t%d\t\t%.2e  %.2e  %.2e  %.2e\n",(int)currentGeneration,res.tv_sec,(int)res.tv_usec/10000,(int)population->currentEvaluationNb,population->Best->getFitness(),this->cstats->currentAverageFitness,this->cstats->currentStdDev, population->Worst->getFitness());
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
		fprintf(f,"GEN,TIME,EVAL,BEST,AVG,STDDEV,WORST\n");
#ifdef WIN32
	  fprintf(f,"%lu\t%2.6f\t%lu\t%.2e\t%.2e\t%.2e\t%.2e\n",currentGeneration,duration,population->currentEvaluationNb,population->Best->getFitness(),this->cstats->currentAverageFitness,this->cstats->currentStdDev, population->Worst->getFitness());
#else
	    //printf("%d\t%ld.%01ld\t%d\t%.2e\t%.2e\t%.2e\t%.2e\n",(int)currentGeneration,res.tv_sec,res.tv_usec,(int)population->currentEvaluationNb,population->Best->getFitness(),currentAverageFitness,currentSTDEV, population->Worst->getFitness());
	    fprintf(f,"%d\t%ld.%d\t\t%d\t\t%.2e  %.2e  %.2e  %.2e\n",(int)currentGeneration,res.tv_sec,(int)res.tv_usec/10000,(int)population->currentEvaluationNb,population->Best->getFitness(),this->cstats->currentAverageFitness,this->cstats->currentStdDev, population->Worst->getFitness());
#endif
	  fclose(f);
        }
  }
  //print grapher
  #ifndef WIN32
  if(this->params->plotStats && this->grapher->valid){
	//if(currentGeneration==0)
	//	fprintf(this->grapher->fWrit,"plot \'%s.dat\' using 3:4 t \'Best Fitness\' w lines ls 1, \'%s.dat\' using 3:5 t  \'Average\' w lines ls 4, \'%s.dat\' using 3:6 t \'StdDev\' w lines ls 3\n", params->outputFilename,params->outputFilename,params->outputFilename);
	//else
	//	fprintf(this->grapher->fWrit,"replot\n");
    fprintf(this->grapher->fWrit,"add coordinate:%d;%f;%f;%f\n",population->currentEvaluationNb, population->Best->fitness, this->cstats->currentAverageFitness, this->cstats->currentStdDev);
    if(this->params->remoteIslandModel){
        if(population->Best->isImmigrant){
            fprintf(this->grapher->fWrit,"set immigrant\n");
        }
        fprintf(this->grapher->fWrit,"add stat:%d;%d;%d\n",currentGeneration, this->cstats->currentNumberOfImmigrants, this->cstats->currentNumberOfImmigrantReproductions);
    }
    if(currentGeneration==0){
        fprintf(this->grapher->fWrit,"paint\n");
    }
    else{
        fprintf(this->grapher->fWrit,"repaint\n");
    }
	fflush(this->grapher->fWrit);
 }
#endif
 
#ifdef __linux__
  double elapsedTime = res.tv_sec + 0.0;
  double micSec = res.tv_usec + 0.0;

  while(micSec>1){
	micSec /= 10.;
  }
  elapsedTime += micSec + 0.0;


  params->timeCriterion->setElapsedTime(elapsedTime);

#endif

  //Reset Current Gen Stats
  this->cstats->resetCurrentStats();
}

//REMOTE ISLAND MODEL FUNCTIONS
void CEvolutionaryAlgorithm::initializeClients(){
	/*int clientNumber=0;
	char (*clients)[16] = (char(*)[16])calloc(1,sizeof(char)*16);
	
	cout << "Reading IP address file: " << this->params->ipFile << endl;
	ifstream IP_File(this->params->ipFile);
	string line;
	while(getline(IP_File, line)){
		if(!isLocalMachine(line.c_str())){
			memmove(clients[this->numberOfClients],line.c_str(),sizeof(char)*16);
			this->numberOfClients++;
			clients = (char(*)[16])realloc(clients,sizeof(char)*16*(this->numberOfClients*16));
			clientNumber++;
		}
		else{
			this->myClientNumber = clientNumber;	
		}
	}*/
    this->refreshClient();

	/*if(this->numberOfClients>0){
		this->Clients = (CComUDPClient**)malloc(this->numberOfClients*sizeof(CComUDPClient*));
		for(int i=0; i<(signed)this->numberOfClients; i++){
			this->Clients[i] = new CComUDPClient(2909,(const char*)clients[i],0);
		}
	}
	else{*/
    if(this->numberOfClients<=0){
		cout << "***WARNING***\nNo islands to communicate with." << endl;
	//	params->remoteIslandModel=0;
	}
}

void CEvolutionaryAlgorithm::refreshClient(){
    //unsigned no_client;
    //this->Clients = parse_file(this->params->ipFile,&no_client, this->params->serverPort);

    //cout << "ip file : " << this->params->ipFile << " contains " << no_client << " client ip(s)" << endl;
    this->numberOfClients = fileserver->number_of_clients();
    //fileserver->refresh_worker_list();
}

void CEvolutionaryAlgorithm::sendIndividual(){
	//Sending an individual every n generations	

	if(globalRandomGenerator->random(0.0,1.0)<=params->migrationProbability){
	//if((this->currentGeneration+this->myClientNumber)%3==0 && this->currentGeneration!=0){
		//cout << "I'm going to send an Individual now" << endl;
		this->population->selectionOperator->initialize(this->population->parents, params->selectionPressure, this->population->actualParentPopulationSize);
		//unsigned index = this->population->selectionOperator->selectNext(this->population->actualParentPopulationSize);
	
		//selecting a client randomly
		int client = globalRandomGenerator->getRandomIntMax(fileserver->number_of_clients());
		//for(int client=0; client<this->numberOfClients; client++){
		cout << "    Going to send an individual to client " << client << endl;
		//cout << "Sending individual " << index << " to client " << client << " now" << endl;
		//cout << this->population->parents[index]->serialize() << endl;
		/*if(client < this->numberOfClients)
		{
		    cout << "    His IP is " << this->Clients[client]->getIP() << " and his port is " << this->Clients[client]->getPort() <<endl;
		    this->Clients[client]->CComUDP_client_send((char*)bBest->serialize().c_str());
		}    
		else*/
		    fileserver->send((char*)bBest->serialize().c_str(), client);
	}
}

void CEvolutionaryAlgorithm::receiveIndividuals(){
	//Checking every generation for received individuals
	if(receivedIndividuals.size()>0)
	{
	  CSelectionOperator *antiTournament = getSelectionOperator("Tournament",!this->params->minimizing, globalRandomGenerator);		

	  while(!receivedIndividuals.empty())
	  {
    		antiTournament->initialize(this->population->parents, 7, this->population->actualParentPopulationSize);
		unsigned index = antiTournament->selectNext(this->population->actualParentPopulationSize);
		this->fileserver->read_data_lock();
		this->population->parents[index]->deserialize(receivedIndividuals.front());
		this->population->parents[index]->isImmigrant = true;
		this->fileserver->read_data_unlock();
		receivedIndividuals.pop();
			//cout << "new Individual :" << this->population->parents[index]->serialize() << endl;
		this->treatedIndividuals++;
		
	  }
	  
	}

/*	if(this->treatedIndividuals<(unsigned)this->server->nb_data){
		//cout << "number of received individuals :" << this->server->nb_data << endl;
		//cout << "number of treated individuals :" << this->treatedIndividuals << endl;
		CSelectionOperator *antiTournament = getSelectionOperator("Tournament",!this->params->minimizing, globalRandomGenerator);		

		//Treating all the individuals before continuing
		while(this->treatedIndividuals < (unsigned)this->server->nb_data){
			//selecting the individual to erase
			antiTournament->initialize(this->population->parents, 7, this->population->actualParentPopulationSize);
			unsigned index = antiTournament->selectNext(this->population->actualParentPopulationSize);
			
			//We're selecting the worst element to replace
			//size_t index = this->population->getWorstIndividualIndex(this->population->parents);

			//cout << "old individual fitness :" << this->population->parents[index]->fitness << endl;
			//cout << "old Individual :" << this->population->parents[index]->serialize() << endl;
			this->server->read_data_lock();
			string line = this->server->parm->data[this->treatedIndividuals].data;
			this->population->parents[index]->deserialize(line);
            //TAG THE INDIVIDUAL AS IMMIGRANT
            this->population->parents[index]->isImmigrant = true;

			this->server->read_data_unlock();
			//cout << "new Individual :" << this->population->parents[index]->serialize() << endl;
			this->treatedIndividuals++;
		}
	}
	
	// now with the fileserver
	if(this->treatedFileIndividuals<(unsigned)this->fileserver->nb_data){
		cout << "number of received individuals :" << this->fileserver->nb_data << endl;
		//cout << "number of treated individuals :" << this->treatedIndividuals << endl;
		CSelectionOperator *antiTournament = getSelectionOperator("Tournament",!this->params->minimizing, globalRandomGenerator);		

		//Treating all the individuals before continuing
		while(this->treatedFileIndividuals < (unsigned)this->fileserver->nb_data){
			//selecting the individual to erase
			antiTournament->initialize(this->population->parents, 7, this->population->actualParentPopulationSize);
			unsigned index = antiTournament->selectNext(this->population->actualParentPopulationSize);
			
			//We're selecting the worst element to replace
			//size_t index = this->population->getWorstIndividualIndex(this->population->parents);

			//cout << "old individual fitness :" << this->population->parents[index]->fitness << endl;
			//cout << "old Individual :" << this->population->parents[index]->serialize() << endl;
			this->fileserver->read_data_lock();
			string line = this->fileserver->data[this->treatedFileIndividuals].data;
			this->population->parents[index]->deserialize(line);
            //TAG THE INDIVIDUAL AS IMMIGRANT
			this->population->parents[index]->isImmigrant = true;

			this->fileserver->read_data_unlock();
			//cout << "new Individual :" << this->population->parents[index]->serialize() << endl;
			this->treatedFileIndividuals++;
		}
	}*/

	
	
}

void CEvolutionaryAlgorithm::outputGraph(){
    /*  	fprintf(this->grapher->fWrit,"set term png\n");
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
			std::cout << "Stopping criterion reached" << std::endl;
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

