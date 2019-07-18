#!/bin/bash

echo "Launch ZDT4...."
./zdt4
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
