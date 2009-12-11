\TEMPLATE_START
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libEasea.lib")
#endif
/**
 This is program entry for CMAES template for EASEA

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
#include "CCmaes.h"

using namespace std;

/** Global variables for the whole algorithm */
CIndividual** pPopulation = NULL;
CIndividual *bBest = NULL;
float* pEZ_MUT_PROB = NULL;
float* pEZ_XOVER_PROB = NULL;
size_t *EZ_NB_GEN;
size_t *EZ_current_generation;

CCmaes *cma;
//CMA cma;

int main(int argc, char** argv){


	parseArguments("EASEA.prm",argc,argv);
	ParametersImpl p;
	p.setDefaultParameters(argc,argv);
	CEvolutionaryAlgorithm* ea = p.newEvolutionaryAlgorithm();

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
#include "CCmaes.h"

using namespace std;

#include "EASEAIndividual.hpp"


CRandomGenerator* globalRandomGenerator;

#define STD_TPL

\INSERT_USER_DECLARATIONS
\ANALYSE_USER_CLASSES

\INSERT_USER_CLASSES

\INSERT_USER_FUNCTIONS

\INSERT_INITIALISATION_FUNCTION
\INSERT_FINALIZATION_FUNCTION

\INSERT_BOUND_CHECKING


void EASEAInit(int argc, char** argv){
	\INSERT_INIT_FCT_CALL
  	cma = new CCmaes(setVariable("nbOffspring",\OFF_SIZE), setVariable("popSize",\POP_SIZE), \PROBLEM_DIM);
}

void EASEAFinal(CPopulation* pop){
	\INSERT_FINALIZATION_FCT_CALL;
	delete(cma);
}

void AESAEBeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
    cma->cmaes_UpdateEigensystem(0);
    cma->TestMinStdDevs();
    \INSERT_BEGIN_GENERATION_FUNCTION
}

void AESAEEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	\INSERT_END_GENERATION_FUNCTION
	evolutionaryAlgorithm->population->sortParentPopulation();
	double **popparent;
	double *fitpar;
	popparent = (double**)malloc(evolutionaryAlgorithm->population->parentPopulationSize*sizeof(double *));
	fitpar = (double*)malloc(evolutionaryAlgorithm->population->parentPopulationSize*sizeof(double));
	for(int i=0; i<(signed)evolutionaryAlgorithm->population->parentPopulationSize; i++)
		popparent[i] = (double*)malloc(\PROBLEM_DIM*sizeof(double));
	for(int i=0; i<(signed)evolutionaryAlgorithm->population->parentPopulationSize; i++){
		for(int j=0; j<\PROBLEM_DIM; j++){
			IndividualImpl *tmp = (IndividualImpl *)evolutionaryAlgorithm->population->parents[i];
			popparent[i][j] = tmp->\GENOME_NAME[j];
			//cout << popparent[i][j] << "|";
		}
		fitpar[i] = evolutionaryAlgorithm->population->parents[i]->fitness;
		//cout << fitpar[i] << endl;
	}
	cma->cmaes_update(popparent, fitpar);
	for(int i=0; i<(signed)evolutionaryAlgorithm->population->parentPopulationSize; i++){
		free(popparent[i]);
	}
	free(popparent);
	free(fitpar);

}

void AESAEGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
        \INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT
}



IndividualImpl::IndividualImpl() : CIndividual() {
  \GENOME_CTOR 
  for(int i=0; i<\PROBLEM_DIM; i++ ) {
	this->\GENOME_NAME[i] = 0.5 + (cma->sigma * cma->rgD[i] * cma->alea.alea_Gauss());
  }
  valid = false;
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
  	for (int i = 0; i < \PROBLEM_DIM; ++i)
		cma->rgdTmp[i] = cma->rgD[i] * cma->alea.alea_Gauss();

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
  double sum;
  for (int i = 0; i < \PROBLEM_DIM; ++i) {
	sum = 0.;
	for (int j = 0; j < \PROBLEM_DIM; ++j)
		sum += cma->B[i][j] * cma->rgdTmp[j];
	this->\GENOME_NAME[i] = cma->rgxmean[i] + cma->sigma * sum;
  }
  return 0;
}




void ParametersImpl::setDefaultParameters(int argc, char** argv){

        this->minimizing =1;
        this->nbGen = setVariable("nbGen",(int)\NB_GEN);

        selectionOperator = getSelectionOperator(setVariable("selectionOperator","Tournament"), this->minimizing, globalRandomGenerator);
        replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator","\RED_FINAL_OPERATOR"),this->minimizing, globalRandomGenerator);
        selectionPressure = 1;
        replacementPressure = setVariable("reduceFinalPressure",(float)\RED_FINAL_PRM);
        pCrossover = 1;
        pMutation = 1;
        pMutationPerGene = 1;

        parentPopulationSize = setVariable("popSize",(int)\POP_SIZE);
        offspringPopulationSize = setVariable("nbOffspring",(int)\OFF_SIZE);

        this->elitSize = 0;

	offspringReduction = parentReduction = false;

        generationalCriterion = new CGenerationalCriterion(setVariable("nbGen",(int)\NB_GEN));
        controlCStopingCriterion = new CControlCStopingCriterion();
        timeCriterion = new CTimeCriterion(setVariable("timeLimit",\TIME_LIMIT));


        seed = setVariable("seed",(int)time(0));
        globalRandomGenerator = new CRandomGenerator(seed);
        this->randomGenerator = globalRandomGenerator;

        this->printStats = setVariable("printStats",\PRINT_STATS);
        this->generateCVSFile = setVariable("generateCVSFile",\GENERATE_CVS_FILE);
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

//#include "CRandomGenerator.h"
#include <stdlib.h>
#include <iostream>
#include <CIndividual.h>
#include <Parameters.h>
#include <CCmaes.h>
class CRandomGenerator;
class CSelectionOperator;
class CGenerationalCriterion;
class CEvolutionaryAlgorithm;
class CPopulation;
class CCmaes;
class Parameters;

extern CCmaes *cma;

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

EASEALIB_PATH=\EZ_PATHlibeasea/

CXXFLAGS =	-O2 -g -Wall -fmessage-length=0 -I$(EASEALIB_PATH)include

OBJS = EASEA.o EASEAIndividual.o 

LIBS = -lboost_program_options 

\INSERT_MAKEFILE_OPTION

TARGET =	EASEA

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS) -g $(EASEALIB_PATH)libeasea.a

	
#%.o:%.cpp
#	$(CXX) -c $(CXXFLAGS) $^

all:	$(TARGET)
clean:
	rm -f $(OBJS) $(TARGET)
easeaclean:
	rm -f $(TARGET) *.o *.cpp *.hpp EASEA.png EASEA.dat EASEA.prm EASEA.mak Makefile EASEA.vcproj EASEA.r EASEA.plot EASEA.csv
	
	
	
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
#  Parameter file generated by CMAES.tpl AESAE v0.1
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
--reduceFinalOperator=\RED_FINAL_OPERATOR
--reduceFinalPressure=\RED_FINAL_PRM

#####   Stats Ouput     #####
--printStats=\PRINT_STATS #print Stats to screen
--plotStats=\PLOT_STATS #plot Stats with gnuplot (requires Gnuplot)
--printInitialPopulation=0 #Print initial population
--printFinalPopulation=0 #Print final population
--generateCVSFile=\GENERATE_CVS_FILE
--generateGnuplotScript=\GENERATE_GNUPLOT_SCRIPT
--generateRScript=\GENERATE_R_SCRIPT

\TEMPLATE_END
