
#include "Easea.h"
#include "EaseaLex.h"
#include "NeuralLex.h"
#include "EaseaParse.hpp"
#include "NeuralParse.hpp"
#include "../libeasna/src/perceptron.hpp"

#include <iostream>
#include <regex>
#include <string>

enum Mode {ERROR, EVOLUTIONNARY_ALGORITHM, NEURAL_ALGORITHM, MIXED};

void usage() {
	std::cout << "EASEA : ./easena [-asrea | -cmaes | -cuda | -cuda_mo | -cuda_gp | -fastemo | -gp | -memetic | -nsgaii | -std | -std_mo | -v] file.ez" << std::endl;
	std::cout << "EASNA : ./easena [--help] [--help.activation] [--help.combinaison] [--help.examples] [--help.randomDistribution] [--batch.size int] [--batch.error (average|sum)] [--compute string] [--learn.online string] [--learn.batch string] [--parse stringfile.nz] [--save.architecture stringfile.nz] [--save.weights string] [--term] [--update.weights string]" << std::endl;
}

Mode detectModeUsed(int argc, char** argv) {
	Mode result = ERROR;
	const std::regex EZregex(".+\\.ez(\"\')?");
	const std::regex NZregex(".+\\.nz(\"\')?");
	const std::regex NEZregex(".+\\.nez(\"\')?");
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
	
		default:
			std::cout << "This shouldn't have happened !!!" << std::endl;
			break;
	}
	return 0;
}
