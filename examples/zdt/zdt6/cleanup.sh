#!/bin/bash
echo "Removing old files...."
make clean
rm -f zdt6.cpp
rm -f zdt6.prm
rm -f zdt6.mak
rm -f zdt6Individual.cpp
rm -f zdt6Individual.hpp
rm -f Makefile
rm -f objectives
rm -f zdt6_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"