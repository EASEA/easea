#!/bin/bash

echo "Launch ZDT6...."
./zdt6
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
