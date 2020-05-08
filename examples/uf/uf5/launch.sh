#!/bin/bash

echo "Launch UF5...."
./uf5
echo "Start plot script..."
R < script2D.R --save > plot_info

echo "Done"
