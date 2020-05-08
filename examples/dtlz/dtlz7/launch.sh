#!/bin/bash

echo "Launch DTLZ7...."
./dtlz7
echo "Start plot script..."
R < script3D.R --save > plot_info

echo "Done"
