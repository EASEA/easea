#!/bin/bash

echo "Launch UF1...."
./uf1
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
