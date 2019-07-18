#!/bin/bash
echo "Removing old files...."
make clean
rm -f zdt4.cpp
rm -f zdt4.prm
rm -f zdt4.mak
rm -f zdt4Individual.cpp
rm -f zdt4Individual.hpp
rm -f Makefile
rm -f objectives
rm -f zdt4_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"