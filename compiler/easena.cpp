
#include "Easea.h"
#include "EaseaLex.h"
#include "NeuralLex.h"
#include "EaseaParse.hpp"
#include "NeuralParse.hpp"
#include "../libeasna/src/perceptron.hpp"
#include <version.h>
#include <config.h>
#include <iostream>
#include <regex>
#include <string>
#include <CLogger.h>

enum Mode {ERROR, EVOLUTIONNARY_ALGORITHM, NEURAL_ALGORITHM, MIXED, END};

void usage() {
	std::cout << "EASEA : ./easena [-asrea | -cmaes | -cuda | -cuda_mo | -cuda_gp | -fastemo | -gp | -memetic | -nsgaii | nsgaiii | cdas | -std | -std_mo | -v ] file.ez | [--version]" << std::endl;
	std::cout << "EASNA : ./easena [--help] [--help.activation] [--help.examples] [--help.randomDistribution] [--batch.size int] [--batch.error (average|sum)] [--compute string] [--learn.online string] [--learn.batch string] [--parse stringfile.nz] [--save.architecture stringfile.nz] [--save.weights string] [--term] [--update.weights string]" << std::endl;
}

Mode detectModeUsed(int argc, char** argv) {
	Mode result = ERROR;
	const std::regex EZregex(".+\\.ez(\"\')?");
	const std::regex NZregex(".+\\.nz(\"\')?");
	const std::regex NEZregex(".+\\.nez(\"\')?");
	

	char *sTemp;
	if (argc == 1){
	    LOG_ERROR(errorCode::io, "Expected argument following easena");
	    return result;
	}
        if ((argv[1][0]=='-')&&(argv[1][1]=='-')){
            sTemp=&(argv[1][2]);
            if (!mystricmp(sTemp,"version")){
             std::cout << "EASENA version: " << easea::version::as_string() << std::endl;
             result = END;
            }else{ 
		 LOG_ERROR(errorCode::io, std::string("Unrecognised input option: ")+ std::string(sTemp));
		 result = ERROR;
	    }
	    return result;
	    
	}    
	for(int i = 0; i < argc; i++)
	{
		if (result == ERROR) {
			if (std::regex_match(argv[i], EZregex))
				result = EVOLUTIONNARY_ALGORITHM;
			else if (std::regex_match(argv[i], NZregex))
				result = NEURAL_ALGORITHM;
			else if (std::regex_match(argv[i], NEZregex))
				result = MIXED;
		} else {
			if (std::regex_match(argv[i], EZregex)
				|| std::regex_match(argv[i], NZregex)
				|| std::regex_match(argv[i], NEZregex)) {
				result = ERROR;
				break;
			}
		}
	}
	
	return result;
}

int main(int argc, char** argv)
{
	try{

	switch (detectModeUsed(argc, argv)) {
		case EVOLUTIONNARY_ALGORITHM:
			easeaParse(argc, argv);
			break;

		case NEURAL_ALGORITHM:
			EASNAmain(argc,argv);
			break;

		case MIXED:
			std::cout << "EASENA !!! Coming soon ..." << std::endl;
			neuralParse(argc, argv);
			break;

		case ERROR:
			std::cout << "Usage :" << std::endl;
			usage();
			break;
		case END:
			break;
	
		default:
			std::cout << "This shouldn't have happened !!!" << std::endl;
			break;
		}
		}
		catch(Exception& e)
		{
		    printf("%s\n",e.what());
		    return 0;
		}
	
	return 0;
}
