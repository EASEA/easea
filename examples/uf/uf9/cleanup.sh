#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf9.cpp
rm -f uf9.prm
rm -f uf9.mak
rm -f uf9Individual.cpp
rm -f uf9Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf9_pf.png
rm -f .RData
rm -f plot_info
echo "Done"