/*
 * COptionParser.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 *  Changed 25.08.2018 by Anna Ouskova Leonteva
 */

#include <iostream>
#include <stdio.h>
#include <memory>
#include <sstream>
#include "include/COptionParser.h"
#include <third_party/cxxopts/cxxopts.hpp>
#include <CLogger.h>
#include "include/define.h"


using namespace cxxopts;
using namespace std;

/* Option parser is based on lightweighted header only library
 * cxxopts.
 */

std::unique_ptr<cxxopts::ParseResult> vm = nullptr;
std::unique_ptr<cxxopts::ParseResult> vm_file = nullptr;
cxxopts::Options options("Allowed options");


template<typename TypeVariable>
TypeVariable setVariable(const std::string argumentName, TypeVariable defaultValue, std::unique_ptr<cxxopts::ParseResult> &vm,  std::unique_ptr<cxxopts::ParseResult>& vm_file){

    TypeVariable ret;

    if (vm->count(argumentName)){
        auto ptr = *vm;
        ret = ptr[argumentName].as<TypeVariable>();
        return ret;
    }else if( vm_file->count(argumentName) ){
        auto ptr = *vm_file;
        ret = ptr[argumentName].as<TypeVariable>();
        return ret;
     }else {
        ret = defaultValue;
    //    msg << argumentName <<" is not declared, default value is  " << ret;
    //    LOG_MSG(msgType::INFO, msg.str());
        return ret;
    }
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

void parseArguments(const char* parametersFileName, int ac, char** av, std::unique_ptr<cxxopts::ParseResult> &vm, std::unique_ptr<cxxopts::ParseResult>& vm_file){

    char** argv;
    int argc;
    if( parametersFileName )
        argc = loadParametersFile(parametersFileName,&argv);
    else{
        argc = 0;
        argv = nullptr;
    }

    options
        .add_options()
        ("help", "produce help message")
        ("compression", "set compression level", cxxopts::value<int>())
        ("seed", "set the global seed of the pseudo random generator", cxxopts::value<int>())
        ("popSize","set the population size",cxxopts::value<int>())
        ("nbOffspring","set the offspring population size", cxxopts::value<int>())
        ("survivingParents","set the reduction size for parent population", cxxopts::value<float>())
        ("survivingOffspring","set the reduction size for offspring population",cxxopts::value<float>())
        ("elite","Elite size",cxxopts::value<int>())
        ("eliteType","Strong (1) or weak (0)",cxxopts::value<int>())
        ("nbCPUThreads","Set the number of threads", cxxopts::value<int>())
        ("isLogg","Set 0 if you want to swith off the logging", cxxopts::value<int>())
        ("reevaluateImmigrants","Set 1 if you want evaluate immigrant", cxxopts::value<int>())
        ("nbGen","Set the number of generation", cxxopts::value<int>())
        ("timeLimit","Set the timeLimit, (0) to desactivate",cxxopts::value<int>())
        ("selectionOperator","Set the Selection Operator (default : Tournament)",cxxopts::value<string>())
        ("selectionPressure","Set the Selection Pressure (default : 2.0)",cxxopts::value<float>())
        ("reduceParentsOperator","Set the Parents Reducing Operator (default : Tournament)",cxxopts::value<string>())
        ("reduceParentsPressure","Set the Parents Reducing Pressure (default : 2.0)",cxxopts::value<float>())
        ("reduceOffspringOperator","Set the Offspring Reducing Operator (default : Tournament)",cxxopts::value<string>())
        ("reduceOffspringPressure","Set the Offspring Reducing Pressure (default : 2.0)",cxxopts::value<float>())
        ("reduceFinalOperator","Set the Final Reducing Operator (default : Tournament)",cxxopts::value<string>())
        ("reduceFinalPressure","Set the Final Reducing Pressure (default : 2.0)",cxxopts::value<float>())
        ("optimiseIterations","Set the number of optimisation iterations (default : 100)",cxxopts::value<int>())
        ("baldwinism","Only keep fitness (default : 0)",cxxopts::value<int>())
        ("remoteIslandModel","Boolean to activate the individual exchange with remote islands (default : 0)",cxxopts::value<int>())
        ("ipFile","File containing all the IPs of the remote islands)",cxxopts::value<string>())
        ("migrationProbability","Probability to send an individual each generation", cxxopts::value<float>())
        ("serverPort","Port of the Server", cxxopts::value<int>())
        ("outputfile","Set an output file for the final population (default : none)",cxxopts::value<string>())
        ("inputfile","Set an input file for the initial population (default : none)",cxxopts::value<string>())
        ("printStats","Print the Stats (default : 1)",cxxopts::value<int>())
        ("plotStats","Plot the Stats (default : 0)",cxxopts::value<int>())
        ("generateCSVFile","Prints the Stats to a CSV File (Filename: ProjectName.dat) (default : 0)",cxxopts::value<int>())
        ("generatePlotScript","Generate a Gnuplot script to plot the Stats (Filename: ProjectName.plot) (default : 0)",cxxopts::value<int>())
        ("generateRScript","Generate a R script to plot the Stats (Filename: ProjectName.r) (default : 0)",cxxopts::value<int>())
//  ("printStatsFile",cxxopts::value<int>(),"Prints the Stats to a File (Filename: ProjectName.dat) (default : 0)")
        ("printInitialPopulation","Print the initial population (default : 0)",cxxopts::value<int>())
        ("printFinalPopulation","Print the final population (default : 0)",cxxopts::value<int>())
        ("savePopulation","Save population at the end (default : 0)",cxxopts::value<int>())
        ("startFromFile","Load the population from a .pop file (default : 0",cxxopts::value<int>())
        ("fstgpu","The number of the first GPU used for computation",cxxopts::value<int>())
        ("lstgpu","The number of the first GPU NOT used for computation",cxxopts::value<int>())
        ("u1","User defined parameter 1",cxxopts::value<string>())
        ("u2","User defined parameter 2",cxxopts::value<string>())
        ("u3","User defined parameter 3", cxxopts::value<int>())
        ("u4","User defined parameter 4",cxxopts::value<int>())
        ("u5","User defined parameter 5",cxxopts::value<int>());
    try{
        auto vm_value = options.parse(ac,av);
        vm = std::make_unique<cxxopts::ParseResult>(move(vm_value));
        if (vm->count("help")) {
            ostringstream msg;
            LOG_MSG(msgType::INFO,options.help({""}));
	    exit(1);
        }
        if(parametersFileName){
            auto vm_file_value = options.parse(argc, argv);
            vm_file = std::make_unique<cxxopts::ParseResult>(move(vm_file_value));
        }
    }
    catch(const cxxopts::OptionException& e){
        ostringstream msg;
        LOG_ERROR(errorCode::value, msg.str());
  }

    for(auto i = 0 ; i<argc ; i++)
        free(argv[i]);
    if( argv )
        free(argv);
}

void parseArguments(const char* parametersFileName, int ac, char** av){

    parseArguments( parametersFileName,ac,av, vm, vm_file);
}

std::string setVariable(const std::string optionName, string defaultValue){
    return setVariable(optionName, defaultValue, vm, vm_file);
}

float setVariable(const std::string optionName, float defaultValue){
    return setVariable( optionName, defaultValue, vm, vm_file);
}

int setVariable(const std::string optionName, int defaultValue){
    return setVariable(optionName, defaultValue, vm, vm_file);
}

