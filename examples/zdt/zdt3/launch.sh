#!/bin/bash

echo "Launch ZDT3...."
./zdt3
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
