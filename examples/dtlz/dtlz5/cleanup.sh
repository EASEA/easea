#!/bin/bash
echo "Removing old files...."
make clean
rm -f dtlz5.cpp
rm -f dtlz5.prm
rm -f dtlz5.mak
rm -f dtlz5Individual.cpp
rm -f dtlz5Individual.hpp
rm -f Makefile
rm -f objectives
rm -f dtlz5_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"