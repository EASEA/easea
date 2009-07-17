/*
 * COptionParser.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;


po::variables_map vm;
po::variables_map vm_file;

using namespace std;

string setVariable(string argumentName, string defaultValue, po::variables_map vm, po::variables_map vm_file){
  string ret;

  if( vm.count(argumentName) ){
    ret = vm[argumentName].as<string>();
    cout << argumentName << " is declared in user command line as "<< ret << endl;
  }
  else if( vm_file.count(argumentName) ){
    ret = vm_file[argumentName].as<string>();
    cout <<  argumentName << " is declared configuration file as "<< ret << endl;
  }
  else {
    ret = defaultValue;
    cout << argumentName << " is not declared, default value is "<< ret<< endl;
  }
  return ret;
}

int setVariable(string argumentName, int defaultValue, po::variables_map vm, po::variables_map vm_file ){
  int ret;

  if( vm.count(argumentName) ){
    ret = vm[argumentName].as<int>();
    cout << argumentName << " is declared in user command line as "<< ret << endl;
  }
  else if( vm_file.count(argumentName) ){
    ret = vm_file[argumentName].as<int>();
    cout <<  argumentName << " is declared configuration file as "<< ret << endl;
  }
  else {
    ret = defaultValue;
    cout << argumentName << " is not declared, default value is "<< ret<< endl;
  }
  return ret;
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
      cout << "line : " <<buffer << endl;
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


void parseArguments(const char* parametersFileName, int ac, char** av,
		    po::variables_map& vm, po::variables_map& vm_file){

  char** argv;
  int argc = loadParametersFile(parametersFileName,&argv);

  po::options_description desc("Allowed options ");
  desc.add_options()
    ("help", "produce help message")
    ("compression", po::value<int>(), "set compression level")
    ("seed", po::value<int>(), "set the global seed of the pseudo random generator")
    ("popSize",po::value<int>(),"set the population size")
    ("nbOffspring",po::value<int>(),"set the offspring population size")
    ("parentReductionSize",po::value<int>(),"set the reduction size for parent population")
    ("offspringReductionSize",po::value<int>(),"set the reduction size for offspring population")
    ("elite",po::value<int>(),"Nb of elite parents (absolute)")
    ("eliteType",po::value<int>(),"Strong (1) or weak (1)")
    ("nbGen",po::value<int>(),"Set the number of generation")
    ("surviveParents",po::value<int>()," Nb of surviving parents (absolute)")
    ("surviveOffsprings",po::value<int>()," Nb of surviving offsprings (absolute)")
    ("outputfile",po::value<string>(),"Set an output file for the final population (default : none)")
    ("inputfile",po::value<string>(),"Set an input file for the initial population (default : none)")
    ("printStats",po::value<int>(),"Print the Stats (default : 1)")
    ("plotStats",po::value<int>(),"Plot the Stats with gnuplot (default : 0)")
    ("printInitialPopulation",po::value<int>(),"Prints the initial population (default : 0)")
    ("printFinalPopulation",po::value<int>(),"Prints the final population (default : 0)")
    ("u1",po::value<string>(),"User defined parameter 1")
    ("u2",po::value<string>(),"User defined parameter 2")
    ("u3",po::value<string>(),"User defined parameter 3")
    ("u4",po::value<string>(),"User defined parameter 4")
    ;

  try{
    po::store(po::parse_command_line(ac, av, desc,0), vm);
    po::store(po::parse_command_line(argc, argv, desc,0), vm_file);
  }
  catch(po::unknown_option& e){
    cerr << "Unknown option  : " << e.what() << endl;
    cout << desc << endl;
    exit(1);
  }

  po::notify(vm);
  po::notify(vm_file);

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }

  for( int i = 0 ; i<argc ; i++ )
    free(argv[i]);
  free(argv);

}

void parseArguments(const char* parametersFileName, int ac, char** av){
  parseArguments(parametersFileName,ac,av,vm,vm_file);
}


int setVariable(const string optionName, int defaultValue){
  return setVariable(optionName,defaultValue,vm,vm_file);
}

string setVariable(const string optionName, string defaultValue){
  return setVariable(optionName,defaultValue,vm,vm_file);
}
