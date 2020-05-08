#!/bin/bash

echo "Launch DTLZ5...."
./dtlz5
echo "Start plot script..."
R < script3D.R --save > plot_info

echo "Done"
