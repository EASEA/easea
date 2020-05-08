#!/bin/bash

#Place here -nsgaiii / -nsgaii / -cdas
echo "Compiling .ez to .cpp .hpp files..."
$EZ_PATH/bin/easena -fastemo  dtlz6.ez
echo "Compiling .cpp file..."
make
echo "Done"
