#!/bin/bash
echo "Removing old files...."
make clean
rm -f zdt2.cpp
rm -f zdt2.prm
rm -f zdt2.mak
rm -f zdt2Individual.cpp
rm -f zdt2Individual.hpp
rm -f Makefile
rm -f objectives
rm -f zdt2_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"