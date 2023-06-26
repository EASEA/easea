#!/bin/bash
echo "Removing old files...."
make clean
rm -f dtlz1.cpp
rm -f dtlz1.prm
rm -f dtlz1.mak
rm -f dtlz1Individual.cpp
rm -f dtlz1Individual.hpp
rm -f Makefile
rm -f objectives
rm -f dtlz1_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"