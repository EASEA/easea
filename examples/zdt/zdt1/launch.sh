#!/bin/bash

echo "Launch ZDT1...."
./zdt1
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
