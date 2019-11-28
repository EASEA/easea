#!/bin/bash
echo "Removing old files...."
make clean
rm -f zdt1.cpp
rm -f zdt1.prm
rm -f zdt1.mak
rm -f zdt1Individual.cpp
rm -f zdt1Individual.hpp
rm -f Makefile
rm -f objectives
rm -f zdt1_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"