#!/bin/zsh

cmake -DCMAKE_C_COMPILER="gcc-9" -DCMAKE_CXX_COMPILER="g++-9" . ;
#cmake -DCMAKE_C_COMPILER="/usr/local/Cellar/gcc/9.1.0/bin/gcc-9" -DCMAKE_CXX_COMPILER="/usr/local/Cellar/gcc/9.1.0/bin/g++-9" . ;
#cmake -DCMAKE_C_COMPILER="/usr/local/Cellar/gcc@8/8.3.0/bin/gcc-8" -DCMAKE_CXX_COMPILER="/usr/local/Cellar/gcc@8/8.3.0/bin/g++-8" . ;
#cmake -DCMAKE_C_COMPILER="/usr/local/Cellar/gcc@7/7.4.0_2/bin/gcc-7" -DCMAKE_CXX_COMPILER="/usr/local/Cellar/gcc@7/7.4.0_2/bin/g++-7" . ;
#cmake -DCMAKE_C_COMPILER="/usr/local/Cellar/gcc@6/6.5.0_2/bin/gcc-6" -DCMAKE_CXX_COMPILER="/usr/local/Cellar/gcc@6/6.5.0_2/bin/g++-6" . ;
#cmake -DCMAKE_C_COMPILER="/usr/local/Cellar/gcc@5/5.5.0_3/bin/gcc-5" -DCMAKE_CXX_COMPILER="/usr/local/Cellar/gcc@5/5.5.0_3/bin/g++-5" . ;

make -j 4 ;