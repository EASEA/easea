#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf1.cpp
rm -f uf1.prm
rm -f uf1.mak
rm -f uf1Individual.cpp
rm -f uf1Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf1_pf.png
rm -f .RData
rm -f plot_info
echo "Done"