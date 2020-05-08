#!/bin/bash
echo "Removing old files...."
make clean
rm -f uf8.cpp
rm -f uf8.prm
rm -f uf8.mak
rm -f uf8Individual.cpp
rm -f uf8Individual.hpp
rm -f Makefile
rm -f objectives
rm -f uf8_pf.png
rm -f .RData
rm -f plot_info
echo "Done"