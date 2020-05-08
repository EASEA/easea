#!/bin/bash

echo "Launch UF8...."
./uf9
echo "Start plot script..."
R < script3D.R --save > plot_info

echo "Done"
