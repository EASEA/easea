/*
 * COptionParser.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 *  Changed 25.08.2018 by Anna Ouskova Leonteva
 */

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <memory>
#include <sstream>
#include "COptionParser.h"
#include <CLogger.h>

using namespace cxxopts;
using namespace std;

/* Option parser is based on lightweighted header only library
 * cxxopts.
 */

std::vector<std::string> loadParametersFile(const string& filename)
{
	std::ifstream paramFile{filename, std::ios_base::in};
	std::vector<std::string> ret;
	ret.emplace_back("");
	std::string buffer;

	while (std::getline(paramFile, buffer)) {
		if ((buffer.find('#') == std::string::npos) && buffer.size() > 0) {
			auto space = buffer.find_first_of(" \t");
			if (space != std::string::npos)
				buffer.resize(space);
			ret.push_back(std::move(buffer));
		}
	}

	return ret;
}

void parseArguments(const char* parametersFileName, int ac, char** av, std::unique_ptr<cxxopts::ParseResult> &vm, std::unique_ptr<cxxopts::ParseResult>& vm_file){

    std::vector<std::string> argv;
    int argc = 0;
    if( parametersFileName ) {
	argv = loadParametersFile(parametersFileName);
        argc = argv.size();
    }


    cxxopts::Options options("Allowed options");
    options
        .add_options()
        ("h,help", "produce help message")
        ("s,seed", "set the global seed of the pseudo random generator", cxxopts::value<int>())
        ("p,popSize","set the population size",cxxopts::value<int>())
        ("nbOffspring","set the offspring population size", cxxopts::value<int>())
        ("survivingParents","set the reduction size for parent population", cxxopts::value<float>())
        ("survivingOffspring","set the reduction size for offspring population",cxxopts::value<float>())
        ("elite","Elite size",cxxopts::value<int>())
        ("eliteType","Strong (1) or weak (0)",cxxopts::value<int>())
        ("t,nbCPUThreads","Set the number of threads", cxxopts::value<int>())
        ("noLogFile","Disable logging to .log file", cxxopts::value<bool>()->default_value("false"))
	("alwaysEvaluate","Always evaluate individual, do not reuse previous fitness", cxxopts::value<bool>()->default_value("false"))
        ("reevaluateImmigrants","Set flag if immigrants need to be reevaluated", cxxopts::value<bool>()->default_value("false"))
        ("g,nbGen","Set the number of generation", cxxopts::value<int>())
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
        ("baldwinism","Only keep fitness",cxxopts::value<bool>()->default_value("false"))
        ("remoteIslandModel","Boolean to activate the individual exchange with remote islands (default : 0)",cxxopts::value<int>())
        ("ipFile","File containing all the IPs of the remote islands)",cxxopts::value<string>())
        ("migrationProbability","Probability to send an individual each generation", cxxopts::value<float>())
        ("serverPort","Port of the Server", cxxopts::value<int>())
        ("outputFile","Set an output file for the final population (default : none)",cxxopts::value<string>())
        ("inputFile","Set an input file for the initial population (default : none)",cxxopts::value<string>())
        ("printStats","Print the Stats (default : 1)",cxxopts::value<int>())
        ("plotStats","Plot the Stats (default : 0)",cxxopts::value<int>())
        ("generateCSVFile","Prints the Stats to a CSV File (Filename: ProjectName.dat) (default : 0)",cxxopts::value<int>())
        ("generatePlotScript","Generate a Gnuplot script to plot the Stats (Filename: ProjectName.plot) (default : 0)",cxxopts::value<int>())
        ("generateRScript","Generate a R script to plot the Stats (Filename: ProjectName.r) (default : 0)",cxxopts::value<int>())
//  ("printStatsFile",cxxopts::value<int>(),"Prints the Stats to a File (Filename: ProjectName.dat) (default : 0)")
        ("printInitialPopulation","Print the initial population",cxxopts::value<bool>()->default_value("false"))
        ("printFinalPopulation","Print the final population",cxxopts::value<bool>()->default_value("false"))
        ("savePopulation","Save population at the end (default : 0)",cxxopts::value<int>())
        ("startFromFile","Load the population from a .pop file (default : 0",cxxopts::value<int>())
        ("fstgpu","The number of the first GPU used for computation",cxxopts::value<int>())
        ("lstgpu","The number of the first GPU NOT used for computation",cxxopts::value<int>())
        ("u1","User defined parameter 1",cxxopts::value<string>())
        ("u2","User defined parameter 2",cxxopts::value<string>())
        ("u3","User defined parameter 3", cxxopts::value<string>())
        ("u4","User defined parameter 4",cxxopts::value<string>())
        ("u5","User defined parameter 5",cxxopts::value<string>());
    try{
        auto vm_value = options.parse(ac,av);
        vm = std::make_unique<cxxopts::ParseResult>(std::move(vm_value));
        if (vm->count("help")) {
            ostringstream msg;
            LOG_MSG(msgType::INFO,options.help({""}));
	    exit(1);
        }
        if(parametersFileName){
	    std::vector<const char*> cstr;
	    std::transform(argv.cbegin(), argv.cend(), std::back_inserter(cstr), [](auto const& s) {return s.c_str();});
	    auto carr = cstr.data();
            auto vm_file_value = options.parse(argc, carr);
            vm_file = std::make_unique<cxxopts::ParseResult>(std::move(vm_file_value));
        }
    }
    catch(const std::exception& e){
	std::cerr << "Bad command line argument(s): " << e.what() << "\n" <<
		options.help({""});
	exit(1);
        //LOG_ERROR(errorCode::value, msg.str());
  }
}
