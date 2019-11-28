#!/bin/bash
echo "Removing old files...."
make clean
rm -f zdt3.cpp
rm -f zdt3.prm
rm -f zdt3.mak
rm -f zdt3Individual.cpp
rm -f zdt3Individual.hpp
rm -f Makefile
rm -f objectives
rm -f zdt3_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"