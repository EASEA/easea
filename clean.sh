#!/bin/sh

rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake Makefile build install_manifest.txt

rm compiler/EaseaParse.hpp compiler/EaseaParse.cpp compiler/EaseaLex.cpp

rm compiler/NeuralParse.hpp compiler/NeuralParse.cpp compiler/NeuralLex.cpp compiler/NeuralLex.h

rm easena libeasea.a libeasea/libeasea.a libeasna.a libeasna/libeasna.a

rm -rf Test/xor_mnist/CMakeFiles Test/xor_mnist/cmake_install.cmake Test/xor_mnist/Makefile Test/xor_mnist/easna_regression_test