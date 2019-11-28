#!/bin/bash

echo "Launch ZDT2...."
./zdt2
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
