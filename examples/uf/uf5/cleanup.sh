#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf5.cpp
rm -f uf5.prm
rm -f uf5.mak
rm -f uf5Individual.cpp
rm -f uf5Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf5_pf.png
rm -f .RData
rm -f plot_info
echo "Done"