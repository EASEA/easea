#!/bin/bash

#Commit before easea use flex/bison (alexyacc) : 96e438ca057cab19df99a171f252bdb40404bc56
#(Reference for testing purpose)

result_path=""

clean(){
	rm -f $1 *.o *.cpp *.hpp *.png *.dat *.prm *.mak Makefile *.vcproj  *.r *.plot *.pop 2> /dev/null;
    true
}


generate(){
  directory_name=$1
  easea_cflag=$2
  program_name=""

  if [ -z "$3" ]; then
    program_name=$directory_name;
  else
    program_name=$3
  fi

  
  resultat_directory="${program_name}_${easea_cflag}"

  cd $directory_name

  clean $program_name
  easea $easea_cflag $program_name.ez > /dev/null

  mkdir -p $result_path/$resultat_directory
  mv *.cpp $result_path/$resultat_directory
  mv *.hpp $result_path/$resultat_directory
  mv *.cu $result_path/$resultat_directory 2> /dev/null
  
  cd ..;
}

main(){
  rm -rf $result_path 2> /dev/null
  mkdir -p $result_path


  generate ant "" "" "";
  generate bbob2013 "" "" "";
  generate listsort "" "" "";
  generate michalewicz "" "" "1";
  generate rastrigin "" "" "";
  generate sphere "" "" "";
  generate weierstrass "" "" "1";
  generate memetic_std "" "memetic_weierstrass" "1";
  generate memetic_std_custom "" "memetic_weierstrass" "1";
  
  generate memetic_cuda "cuda_mem" "memetic_weierstrass" "1";
  generate memetic_cuda_custom "cuda_mem" "memetic_weierstrass" "1";
  generate weierstrass "cuda" "" "";
  generate michalewicz "cuda" "" "";
  generate rastrigin "cuda" "" "";

  generate regression "gp" "" "";
  generate regression-network "gp" "regression" "";
  generate regression_okto_simulation "gp" "regression" "1";

  generate regression "cuda_gp" "" "";
  generate regression-network "cuda_gp" "regression" "";
  generate regression_okto_simulation "cuda_gp" "regression" "";
 
  generate cmaes_tests "cmaes" "cigtab" "";
 
  #generate cmaes_cuda_test "cmaes_cuda" "cigtabGPU" "";
}

if [ $# -lt 2 ]
then
    echo "Usage $0 <results_directory_name> <easea_example_directory>"
    exit
fi
result_path=`pwd`"/$1"
cd $2

main
