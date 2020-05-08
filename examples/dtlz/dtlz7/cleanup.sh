#!/bin/bash
echo "Removing old files...."
make clean
rm -f dtlz7.cpp
rm -f dtlz7.prm
rm -f dtlz7.mak
rm -f dtlz7Individual.cpp
rm -f dtlz7Individual.hpp
rm -f Makefile
rm -f objectives
rm -f dtlz7_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"