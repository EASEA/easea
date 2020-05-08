#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf3.cpp
rm -f uf3.prm
rm -f uf3.mak
rm -f uf3Individual.cpp
rm -f uf3Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf3_pf.png
rm -f .RData
rm -f plot_info
echo "Done"