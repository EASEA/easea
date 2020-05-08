#!/bin/bash

echo "Launch UF3...."
./uf3
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
