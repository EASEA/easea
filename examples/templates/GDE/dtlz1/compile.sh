#!/bin/bash

echo "Compiling .ez to .cpp .hpp files..."
$EZ_PATH/bin/easena -gde dtlz1.ez
echo "Compiling .cpp file..."
cmake . && cmake --build . --config Release

echo "Done"
