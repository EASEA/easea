/*
 * COptionParser.cpp
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */
#include <boost/program_options.hpp>
#include <iostream>
#include <stdio.h>

namespace po = boost::program_options;


po::variables_map vm;
po::variables_map vm_file;

using namespace std;

string setVariable(string argumentName, string defaultValue, po::variables_map vm, po::variables_map vm_file){
  string ret;

  if( vm.count(argumentName) ){
    ret = vm[argumentName].as<string>();
//    cout << argumentName << " is declared in user command line as "<< ret << endl;
  }
  else if( vm_file.count(argumentName) ){
    ret = vm_file[argumentName].as<string>();
//    cout <<  argumentName << " is declared configuration file as "<< ret << endl;
  }
  else {
    ret = defaultValue;
//   cout << argumentName << " is not declared, default value is "<< ret<< endl;
  }
  return ret;
}

int setVariable(string argumentName, int defaultValue, po::variables_map vm, po::variables_map vm_file ){
  int ret;

  if( vm.count(argumentName) ){
    ret = vm[argumentName].as<int>();
//    cout << argumentName << " is declared in user command line as "<< ret << endl;
  }
  else if( vm_file.count(argumentName) ){
    ret = vm_file[argumentName].as<int>();
//    cout <<  argumentName << " is declared configuration file as "<< ret << endl;
  }
  else {
    ret = defaultValue;
//    cout << argumentName << " is not declared, default value is "<< ret<< endl;
  }
  return ret;
}

float setVariable(string argumentName, float defaultValue, po::variables_map vm, po::variables_map vm_file ){
  float ret;

  if( vm.count(argumentName) ){
    ret = vm[argumentName].as<float>();
//    cout << argumentName << " is declared in user command line as "<< ret << endl;
  }
  else if( vm_file.count(argumentName) ){
    ret = vm_file[argumentName].as<float>();
//    cout <<  argumentName << " is declared configuration file as "<< ret << endl;
  }
  else {
    ret = defaultValue;
//    cout << argumentName << " is not declared, default value is "<< ret<< endl;
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
//      cout << "line : " <<buffer << endl;
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
  int argc;
  if( parametersFileName )
    argc = loadParametersFile(parametersFileName,&argv);
  else{ 
    argc = 0;
    argv = NULL;
  }
  po::options_description desc("Allowed options ");
  desc.add_options()
	("help", "produce help message")
	("compression", po::value<int>(), "set compression level")
	("seed", po::value<int>(), "set the global seed of the pseudo random generator")
	("popSize",po::value<int>(),"set the population size")
	("nbOffspring",po::value<int>(),"set the offspring population size")
	("survivingParents",po::value<float>(),"set the reduction size for parent population")
	("survivingOffspring",po::value<float>(),"set the reduction size for offspring population")
	("elite",po::value<int>(),"Nb of elite parents (absolute), 0 for no elite")
	("eliteType",po::value<int>(),"Strong (1) or weak (0)")
	("nbGen",po::value<int>(),"Set the number of generation")
	("timeLimit",po::value<int>(),"Set the timeLimit, (0) to deactivate")
	("selectionOperator",po::value<string>(),"Set the Selection Operator (default : Tournament)")
	("selectionPressure",po::value<float>(),"Set the Selection Pressure (default : 2.0)")
	("reduceParentsOperator",po::value<string>(),"Set the Parents Reducing Operator (default : Tournament)")
	("reduceParentsPressure",po::value<float>(),"Set the Parents Reducing Pressure (default : 2.0)")
	("reduceOffspringOperator",po::value<string>(),"Set the Offspring Reducing Operator (default : Tournament)")
	("reduceOffspringPressure",po::value<float>(),"Set the Offspring Reducing Pressure (default : 2.0)")	
	("reduceFinalOperator",po::value<string>(),"Set the Final Reducing Operator (default : Tournament)")
	("reduceFinalPressure",po::value<float>(),"Set the Final Reducing Pressure (default : 2.0)")
	("optimiseIterations",po::value<int>(),"Set the number of optimisation iterations (default : 100)")
	("baldwinism",po::value<int>(),"Only keep fitness (default : 0)")
	("remoteIslandModel",po::value<int>(),"Boolean to activate the individual exachange with remote islands (default : 0)")
	("ipFile",po::value<string>(),"File containing all the IPs of the remote islands)")
	("migrationProbability", po::value<float>(),"Probability to send an individual each generation")
    ("serverPort", po::value<int>(),"Port of the Server")
	("outputfile",po::value<string>(),"Set an output file for the final population (default : none)")
	("inputfile",po::value<string>(),"Set an input file for the initial population (default : none)")
	("printStats",po::value<int>(),"Print the Stats (default : 1)")
	("plotStats",po::value<int>(),"Plot the Stats (default : 0)")
	("generateCSVFile",po::value<int>(),"Print the Stats to a CSV File (Filename: ProjectName.dat) (default : 0)")
	("generatePlotScript",po::value<int>(),"Generates a Gnuplot script to plat the Stats (Filename: ProjectName.plot) (default : 0)")
	("generateRScript",po::value<int>(),"Generates a R script to plat the Stats (Filename: ProjectName.r) (default : 0)")
//	("printStatsFile",po::value<int>(),"Print the Stats to a File (Filename: ProjectName.dat) (default : 0)")
	("printInitialPopulation",po::value<int>(),"Prints the initial population (default : 0)")
	("printFinalPopulation",po::value<int>(),"Prints the final population (default : 0)")
	("savePopulation",po::value<int>(),"Saves population at the end (default : 0)")
	("startFromFile",po::value<int>(),"Loads the population from a .pop file (default : 0")
	("u1",po::value<string>(),"User defined parameter 1")
	("u2",po::value<string>(),"User defined parameter 2")
	("u3",po::value<int>(),"User defined parameter 3")
	("u4",po::value<int>(),"User defined parameter 4")
	("u5",po::value<int>(),"User defined parameter 5")
	;

  try{
    po::store(po::parse_command_line(ac, av, desc,0), vm);
    if( parametersFileName )
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
  if( argv )
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

float setVariable(const string optionName, float defaultValue){
  return setVariable(optionName,defaultValue,vm,vm_file);
}
