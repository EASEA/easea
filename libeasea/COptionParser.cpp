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
#include <memory>
#include <sstream>

#include "COptionParser.h"
#include "CLogger.h"
#include "version.h"

namespace po = boost::program_options;

std::vector<std::string> loadParametersFile(const std::string& filename)
{
	std::ifstream paramFile{filename, std::ios_base::in};
	std::vector<std::string> ret;
	ret.emplace_back("");
	std::string buffer;

	while (std::getline(paramFile, buffer)) {
	    std::string::size_type id_find = buffer.find('#');
	    if ((id_find > 0) && (buffer.size() > 0)) {
			auto space = buffer.find_first_of(" \t");
			if ((id_find != std::string::npos) && (id_find >1))
			    buffer.erase(id_find);
			if (space != std::string::npos)
				buffer.resize(space);
			ret.push_back(std::move(buffer));
		}
	}

	return ret;
}

void parseArguments(const char* parametersFileName, int ac, char** av, std::unique_ptr<vm_t> &vm){
    po::options_description options("Allowed options");
    options.add_options()
        ("help,h", "produce help message")
	("version,v", "Print version")
        ("seed,s", po::value<int>(), "set the global seed of the pseudo random generator")
        ("popSize,p", po::value<int>(), "set the population size")
        ("nbOffspring", po::value<float>(), "set the offspring population size")
        ("survivingParents", po::value<float>(), "set the reduction size for parent population")
        ("survivingOffspring", po::value<float>(), "set the reduction size for offspring population")
        ("elite", po::value<int>(), "Elite size")
        ("eliteType", po::value<int>(), "Strong (1) or weak (0)")
        ("nbCPUThreads,t", po::value<int>(), "Set the number of threads")
        ("noLogFile", po::value<bool>()->default_value("false"), "Disable logging to .log file")
	("alwaysEvaluate", po::value<bool>()->default_value("false"), "Always evaluate individual, do not reuse previous fitness")
        ("reevaluateImmigrants", po::value<bool>()->default_value("false"), "Set flag if immigrants need to be reevaluated")
        ("nbGen,g", po::value<int>(), "Set the number of generation")
        ("timeLimit", po::value<int>(), "Set the timeLimit, (0) to desactivate")
        ("selectionOperator", po::value<std::string>(), "Set the Selection Operator (default : Tournament)")
        ("selectionPressure", po::value<float>(), "Set the Selection Pressure (default : 2.0)")
        ("reduceParentsOperator", po::value<std::string>(), "Set the Parents Reducing Operator (default : Tournament)")
        ("reduceParentsPressure", po::value<float>(), "Set the Parents Reducing Pressure (default : 2.0)")
        ("reduceOffspringOperator", po::value<std::string>(), "Set the Offspring Reducing Operator (default : Tournament)")
        ("reduceOffspringPressure", po::value<float>(), "Set the Offspring Reducing Pressure (default : 2.0)")
        ("reduceFinalOperator", po::value<std::string>(), "Set the Final Reducing Operator (default : Tournament)")
        ("reduceFinalPressure", po::value<float>(), "Set the Final Reducing Pressure (default : 2.0)")
        ("optimiseIterations", po::value<int>(), "Set the number of optimisation iterations (default : 100)")
        ("baldwinism", po::value<bool>()->default_value("false"), "Only keep fitness")
        ("remoteIslandModel", po::value<int>(), "Boolean to activate the individual exchange with remote islands (default : 0)")
	("silentNetwork", po::value<bool>()->default_value("false"), "Do not log informations about sent and received individuals")
        ("ipFile", po::value<std::string>(), "File containing all the IPs of the remote islands)")
        ("migrationProbability", po::value<float>(), "Probability to send an individual each generation")
        ("serverPort", po::value<int>(), "Port of the Server")
        ("outputFile", po::value<std::string>(), "Set an output file for the final population (default : none)")
        ("inputFile", po::value<std::string>(), "Set an input file for the initial population (default : none)")
        ("printStats", po::value<int>(), "Print the Stats (default : 1)")
        ("plotStats", po::value<int>(), "Plot the Stats (default : 0)")
        ("generateCSVFile", po::value<int>(), "Prints the Stats to a CSV File (Filename: ProjectName.dat) (default : 0)")
        ("generatePlotScript", po::value<int>(), "Generate a Gnuplot script to plot the Stats (Filename: ProjectName.plot) (default : 0)")
        ("generateRScript", po::value<int>(), "Generate a R script to plot the Stats (Filename: ProjectName.r) (default : 0)")
//  ("printStatsFile",po::value<int>(),"Prints the Stats to a File (Filename: ProjectName.dat) (default : 0)")
        ("printInitialPopulation", po::value<bool>()->default_value("false"), "Print the initial population")
        ("printFinalPopulation", po::value<bool>()->default_value("false"), "Print the final population")
        ("savePopulation", po::value<int>(), "Save population at the end (default : 0)")
        ("startFromFile", po::value<int>(), "Load the population from a .pop file (default : 0")
        ("fstgpu", po::value<int>(), "The number of the first GPU used for computation")
        ("lstgpu", po::value<int>(), "The number of the first GPU NOT used for computation")
        ("u1", po::value<std::string>(), "User defined parameter 1")
        ("u2", po::value<std::string>(), "User defined parameter 2")
        ("u3", po::value<std::string>(), "User defined parameter 3")
        ("u4", po::value<std::string>(), "User defined parameter 4")
        ("u5", po::value<std::string>(), "User defined parameter 5");
    try{
	vm = std::make_unique<vm_t>();
	po::store(po::parse_command_line(ac, av, options), *vm);
	po::notify(*vm);

        if(parametersFileName){
	    std::vector<std::string> argv = loadParametersFile(parametersFileName);
	    std::vector<const char*> cstr;
	    std::transform(argv.cbegin(), argv.cend(), std::back_inserter(cstr), [](auto const& s) {return s.c_str();});
	    po::store(po::parse_command_line(cstr.size(), cstr.data(), options), *vm);
	    po::notify(*vm);
        }

	if (vm->count("help")) {
	    std::cout << options;
	    exit(0);
        } else if (vm->count("version")) {
	   std::cout << "EASENA version " << easea::version::as_string() << "\n" <<
		   "Compiled in " << EZ_BUILD_TYPE << " mode using " << EZ_BUILT_BY << " " << EZ_BUILT_BY_VERSION << "\n";
	   exit(0);
	}
    }
    catch(const std::exception& e){
	std::cerr << "Bad command line argument(s): " << e.what() << "\n" <<
		options;
	exit(1);
  }
}
