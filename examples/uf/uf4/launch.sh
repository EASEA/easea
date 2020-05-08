#!/bin/bash

echo "Launch UF4...."
./uf4
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
