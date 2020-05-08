#!/bin/bash

echo "Launch UF6...."
./uf6
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
