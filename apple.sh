#!/bin/zsh

cmake -DCMAKE_C_COMPILER="/usr/local/opt/gcc@9/bin/gcc-9" -DCMAKE_CXX_COMPILER="/usr/local/opt/gcc@9/bin/g++-9" . ;

 make -j 4 ;