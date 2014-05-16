#!/bin/bash

#TODO color_echo

clean(){
	rm -f $1 *.o *.cpp *.hpp *.png *.dat *.prm *.mak Makefile *.vcproj *.csv *.r *.plot *.pop 2> /dev/null;
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
  clean $program_name;
  
  echo "$(tput setaf 6)Testing easea compilation of $program_name $(tput sgr 0)"
  easea $easea_cflag $program_name.ez > /dev/null;
  error_code=$?;
  if [[ $error_code != 0 ]] ; then
    echo "$(tput setaf 1)Error, easea compilation case $easea_cflag $program_name $(tput sgr 0)"
    clean $program_name;
    #exit 1;
  else 
    echo "$(tput setaf 2)Pass !!! $(tput sgr 0)"

    echo "$(tput setaf 6)Testing compilation of $program_name generated source code$(tput sgr 0)"
    make > /dev/null;
    error_code=$?;
    if [[ $error_code != 0 ]] ; then
      echo $(tput setaf 1)"Error, compilation case $easea_cflag $program_name $(tput sgr 0)"
      clean $program_name;
      #exit 2;
    else 
      echo "$(tput setaf 2)Pass !!! $(tput sgr 0)"

      echo "$(tput setaf 6)Testing runtime execution of $program_name$(tput sgr 0)"
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
  cd ..;
}

main(){

  echo"STD tests case"
  test_case ant "" "" "";
  test_case bbob2013 "" "" "";
  test_case listsort "" "" "";
  test_case michalewicz "" "" "";
  test_case rastrigin "" "" "";
  test_case sphere "" "" "";
  test_case weierstrass "" "" "";
  test_case cmaes_tests "" "cigtab" "";
  test_case memetic_std "" "memetic_weierstrass" "";
  test_case memetic_std_custom "" "memetic_weierstrass" "";

  echo"CUDA tests case"
  test_case cmaes_cuda_test "cuda" "cigtabGPU" "";
  test_case memetic_cuda "cuda" "memetic_weierstrass" "";
  test_case memetic_cuda_custom "cuda" "memetic_weierstrass" "";
  test_case weierstrass "cuda" "" "";

  echo"GP tests case"
  echo"CUDA GP tests case"
  echo"CMAES tests case"
}

test_case ant "" "" "";
test_case bbob2013 "" "" "";
test_case listsort "" "";
test_case michalewicz "" "" "1";
test_case rastrigin "" "" "";
test_case sphere "" "" "";
test_case weierstrass "" "" "1";
test_case cmaes_tests "" "cigtab" "";
test_case memetic_std "" "memetic_weierstrass" "";
test_case memetic_std_custom "" "memetic_weierstrass" "";

