#!/bin/bash

echo "Launch DTLZ3...."
./dtlz3
echo "Start plot script..."
R < script3D.R --save > plot_info

echo "Done"
