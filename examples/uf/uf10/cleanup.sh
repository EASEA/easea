#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf10.cpp
rm -f uf10.prm
rm -f uf10.mak
rm -f uf10Individual.cpp
rm -f uf10Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf10_pf.png
rm -f .RData
rm -f plot_info
echo "Done"