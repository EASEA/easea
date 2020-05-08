#!/bin/bash

echo "Launch UF7...."
./uf7
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
