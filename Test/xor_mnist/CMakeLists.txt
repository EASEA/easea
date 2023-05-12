CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(easna_regression_test)

# .. -> hint, that the mnist package is one directory level above.
# When using just "find_package(MNIST REQUIRED)", "MNIST_DIR"
#    cmake variable has to be set correctly.
FIND_PACKAGE(MNIST PATHS .)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

INCLUDE_DIRECTORIES(${MNIST_INCLUDE_DIR})
ADD_EXECUTABLE(easna_regression_test main.cpp)
target_compile_options(easna_regression_test PUBLIC
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-pedantic -Wall -Wextra -Woverloaded-virtual -Wfloat-equal -Wwrite-strings -Wpointer-arith -Wcast-qual -Wcast-align -Wshadow -Wredundant-decls -Wdouble-promotion -Winit-self -Wswitch-default -std=c++11 -O3 $<$<NOT:$<CXX_COMPILER_ID:Clang>>:-s -Wlogical-op>>
	)
TARGET_COMPILE_FEATURES(easna_regression_test PRIVATE cxx_range_for)

# Pass MNIST data directory to main.cpp
TARGET_COMPILE_DEFINITIONS(easna_regression_test PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
