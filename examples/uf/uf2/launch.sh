#!/bin/bash

echo "Launch UF2...."
./uf2
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
