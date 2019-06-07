#!/bin/bash

echo "Launch DTLZ2...."
./dtlz2
echo "Start plot script..."
R < script3D.R --save > plot_info

echo "Done"
