#!/bin/bash

echo "Launch DTLZ4...."
./dtlz4
echo "Start plot script..."
R < script3D.R --save > plot_info

echo "Done"
