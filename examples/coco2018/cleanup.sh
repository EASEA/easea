#!/bin/bash
echo "Removing old files...."
make clean
rm -f coco2018.cpp
rm -f coco2018.prm
rm -f coco2018.mak
rm -f coco2018Individual.cpp
rm -f coco2018Individual.hpp
rm -f Makefile
rm -f objectives
rm -f coco2018_pf.png
rm -f .RData
rm -f plot_info
rm -f easea.log
echo "Done"