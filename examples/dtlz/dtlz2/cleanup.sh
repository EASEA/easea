#!/bin/bash
echo "Removing old files...."
make clean
rm -f dtlz2.cpp
rm -f dtlz2.prm
rm -f dtlz2.mak
rm -f dtlz2Individual.cpp
rm -f dtlz2Individual.hpp
rm -f Makefile
rm -f objectives
rm -f dtlz2_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"