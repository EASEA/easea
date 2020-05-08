#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf6.cpp
rm -f uf6.prm
rm -f uf6.mak
rm -f uf6Individual.cpp
rm -f uf6Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf6_pf.png
rm -f .RData
rm -f plot_info
echo "Done"