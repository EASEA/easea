#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf2.cpp
rm -f uf2.prm
rm -f uf2.mak
rm -f uf2Individual.cpp
rm -f uf2Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf2_pf.png
rm -f .RData
rm -f plot_info
echo "Done"