#include "config.h"
#ifdef OS_WINDOWS
	#define YY_NO_UNISTD_H
	#include <io.h>
	using ssize_t = signed long long;
#endif

#include "Easea.h"
#include "EaseaLex.h"
#include "EaseaParse.hpp"
#include <version.h>
#include <config.h>
#include <iostream>
#include <regex>
#include <string>
#include <CLogger.h>

enum Mode {ERROR, EVOLUTIONNARY_ALGORITHM, NEURAL_ALGORITHM, MIXED};

void usage() {
	std::cout << "EASEA : ./easea [-asrea | -cmaes | -cuda | -cuda_mo | -cuda_gp | -fastemo | -gp | -memetic | -nsgaii | nsgaiii | cdas | -std | -std_mo | -v ] file.ez | [--version]" << std::endl;
}

Mode detectModeUsed(int argc, char** argv) {
	Mode result = ERROR;
	const std::regex EZregex(".+\\.ez(\"\')?");
	const std::regex NZregex(".+\\.nz(\"\')?");
	const std::regex NEZregex(".+\\.nez(\"\')?");
	

	char *sTemp;
	if (argc == 1){
	    std::cerr << "Error: expected argument following easea\n";
	    usage();
	    exit(1);
	}
        if ((argv[1][0]=='-')&&(argv[1][1]=='-')){
            sTemp=&(argv[1][2]);
            if (!mystricmp(sTemp,"version")){
             std::cout << "EASEA version " << easea::version::as_string() << "\n" <<
		   "Compiled in " << EZ_BUILD_TYPE << " mode using " << EZ_BUILT_BY << " " << EZ_BUILT_BY_VERSION << "\n";
	     exit(0);
            }else{
		 std::cerr << "Error: unrecognised input option: " << sTemp << "\n";
		 usage();
		 exit(1);
	    }
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
		case MIXED:
			std::cout << "Sorry, .nz/.nez files are no longer supported as of version 3.2.0" << std::endl;
			exit(1);
			break;
		case ERROR:
			std::cout << "Usage :" << std::endl;
			usage();
			exit(1);
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