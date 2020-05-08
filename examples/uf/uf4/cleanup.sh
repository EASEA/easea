#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf4.cpp
rm -f uf4.prm
rm -f uf4.mak
rm -f uf4Individual.cpp
rm -f uf4Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf4_pf.png
rm -f .RData
rm -f plot_info
echo "Done"