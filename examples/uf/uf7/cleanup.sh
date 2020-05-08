#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf7.cpp
rm -f uf7.prm
rm -f uf7.mak
rm -f uf7Individual.cpp
rm -f uf7Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf7_pf.png
rm -f .RData
rm -f plot_info
echo "Done"