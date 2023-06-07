#!/bin/bash

echo "Launch DTLZ1...."
./dtlz1
echo "Start plot script..."
R < script3D.R --save > plot_info

echo "Done"
