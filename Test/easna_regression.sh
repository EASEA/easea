#!/bin/bash

# Download Yann Lecun Mnist datasets
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz ;
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz ;
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz ;
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz ;

# Extract datasets from archives
gunzip train-images-idx3-ubyte.gz ;
gunzip train-labels-idx1-ubyte.gz ;
gunzip t10k-images-idx3-ubyte.gz ;
gunzip t10k-labels-idx1-ubyte.gz ;

# Move Datatsets in the correct folder to test
mv train-images-idx3-ubyte xor_mnist ;
mv train-labels-idx1-ubyte xor_mnist ;
mv t10k-images-idx3-ubyte xor_mnist ;
mv t10k-labels-idx1-ubyte xor_mnist ;

# Move in the right folder
cd xor_mnist ;

# Call the test program and get the results of the tests
FILE_EASENA="./../../easena"
if [ -f "$FILE_EASENA" ];
then
   ./easna_regression_test $1
fi

# Remove every files to keep the repository clean and back to the root test folder
rm train-images-idx3-ubyte ;
rm train-labels-idx1-ubyte ;
rm t10k-images-idx3-ubyte ;
rm t10k-labels-idx1-ubyte ;

# Go back in the test folder
cd .. ;
