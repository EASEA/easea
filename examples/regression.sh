#!/bin/bash

#TODO color_echo

clean(){
	rm -f $1 *.o *.cpp *.hpp *.png *.dat *.prm *.mak Makefile *.vcproj  *.r *.plot *.pop 2> /dev/null;
}


test_case(){
  error_code=0;
  directory_name=$1
  easea_cflag=$2
  program_name=""
  number_gen=5 
  cd $directory_name;

  if [ -z "$3" ]; then
    program_name=$directory_name;
  else
    program_name=$3
  fi

  if [ ! -z "$4" ]; then
    number_gen=$4;
  fi

  echo"" 
  echo"" 
  echo "$(tput setaf 2)* $program_name *$(tput sgr 0)";
  echo "$(tput setaf 2)* --------------- *$(tput sgr 0)";
  clean $program_name;
  
  echo "$(tput setaf 6)Testing easea compilation of $easea_cflag $directory_name $(tput sgr 0)"
  easea $easea_cflag $program_name.ez > /dev/null;
  error_code=$?;
  if [[ $error_code != 0 ]] ; then
    echo "$(tput setaf 1)Error, easea compilation case $easea_cflag $directory_name $(tput sgr 0)"
    clean $program_name;
    #exit 1;
  else 
    echo "$(tput setaf 2)Pass !!! $(tput sgr 0)"

    echo "$(tput setaf 6)Testing compilation of $ $easea_cflag $directory_name generated source code$(tput sgr 0)"
    make > /dev/null;
    error_code=$?;
    if [[ $error_code != 0 ]] ; then
      echo $(tput setaf 1)"Error, compilation case $easea_cflag $directory_name $(tput sgr 0)"
      clean $program_name;
      #exit 2;
    else 
      echo "$(tput setaf 2)Pass !!! $(tput sgr 0)"

      echo "$(tput setaf 6)Testing runtime execution of $easea_cflag $directory_name $(tput sgr 0)"
      ./$program_name --plotStats 0 --remoteIslandModel 0 --nbGen $number_gen > /dev/null;
      error_code=$?;
      if [[ $error_code != 0 ]] ; then
        echo "$(tput setaf 1)Error, runtime $(tput sgr 0)"
        clean $program_name;
        #exit 3;
      else 
        echo "$(tput setaf 2)Pass !!! $(tput sgr 0)"
      fi 
    fi
  fi
  clean $program_name;
  cd ..;
}

main(){

  echo "$(tput setaf 2)****************$(tput sgr 0)";
  echo "$(tput setaf 2)*STD tests case *$(tput sgr 0)";
  echo "$(tput setaf 2)****************$(tput sgr 0)";
  test_case ant "" "" "";
  test_case bbob2013 "" "" "";
  test_case listsort "" "" "";
  test_case michalewicz "" "" "1";
  test_case rastrigin "" "" "";
  test_case sphere "" "" "";
  test_case weierstrass "" "" "1";
  test_case cmaes_tests "" "cigtab" "";
  test_case memetic_std "" "memetic_weierstrass" "1";
  test_case memetic_std_custom "" "memetic_weierstrass" "1";
  
  echo "";
  echo "$(tput setaf 2)******************$(tput sgr 0)";
  echo "$(tput setaf 2)*CUDA tests case *$(tput sgr 0)"
  echo "$(tput setaf 2)******************$(tput sgr 0)";
  test_case cmaes_cuda_test "cuda" "cigtabGPU" "";
  test_case memetic_cuda "cuda_mem" "memetic_weierstrass" "1";
  test_case memetic_cuda_custom "cuda_mem" "memetic_weierstrass" "1";
  test_case weierstrass "cuda" "" "";
  test_case michalewicz "cuda" "" "";
  test_case rastrigin "cuda" "" "";

  echo "";
  echo "$(tput setaf 2)****************$(tput sgr 0)";
  echo "$(tput setaf 2)*GP tests case *$(tput sgr 0)";
  echo "$(tput setaf 2)****************$(tput sgr 0)";
  test_case regression "gp" "" "";
  test_case regression-network "gp" "regression" "";
  test_case regression_okto_simulation "gp" "regression" "1";

  echo "";
  echo "$(tput setaf 2)*********************$(tput sgr 0)";
  echo "$(tput setaf 2)*CUDA GP tests case *$(tput sgr 0)";
  echo "$(tput setaf 2)*********************$(tput sgr 0)";
  test_case regression "cuda_gp" "" "";
  test_case regression-network "cuda_gp" "regression" "";
  test_case regression_okto_simulation "cuda_gp" "regression" "";
  
  echo ""
  echo "$(tput setaf 2)*********************$(tput sgr 0)";
  echo "$(tput setaf 2)*CMAES tests case    *$(tput sgr 0)"
  echo "$(tput setaf 2)*********************$(tput sgr 0)";
}

main ;
