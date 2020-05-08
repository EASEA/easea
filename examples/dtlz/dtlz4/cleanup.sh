#!/bin/bash
echo "Removing old files...."
make clean
rm -f dtlz4.cpp
rm -f dtlz4.prm
rm -f dtlz4.mak
rm -f dtlz4Individual.cpp
rm -f dtlz4Individual.hpp
rm -f Makefile
rm -f objectives
rm -f dtlz4_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"