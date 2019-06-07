#!/bin/bash
echo "Removing old files...."
make clean
rm -f dtlz3.cpp
rm -f dtlz3.prm
rm -f dtlz3.mak
rm -f dtlz3Individual.cpp
rm -f dtlz3Individual.hpp
rm -f Makefile
rm -f objectives
rm -f dtlz3_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"