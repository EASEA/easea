#!/bin/bash

echo "Launch DTLZ6...."
./dtlz6
echo "Start plot script..."
R < script3D.R --save > plot_info

echo "Done"
