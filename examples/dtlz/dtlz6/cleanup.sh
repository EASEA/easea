#!/bin/bash
echo "Removing old files...."
make clean
rm -f dtlz6.cpp
rm -f dtlz6.prm
rm -f dtlz6.mak
rm -f dtlz6Individual.cpp
rm -f dtlz6Individual.hpp
rm -f Makefile
rm -f objectives
rm -f dtlz6_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"