#!/bin/bash

echo "Launch UF10...."
./uf10
echo "Start plot script..."
R < script3D.R --save > plot_info

echo "Done"
