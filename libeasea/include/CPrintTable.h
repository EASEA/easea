#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>

namespace easena {

class CPrintTable {

private:

    const int szGenome;
    const int maxNbCol = 5;

    const int widthIndex100  = 4;
    const int widthIndex10   = widthIndex100 + 1;
    const int widthIndex1    = widthIndex100 + 1;

    const int widthSmbl = 2;
    const int precision = 4;
    const int widthGenome = precision * 2;

    int nbRow = 1;
    int nbCol = szGenome;
    const float *genome;


public:
    CPrintTable(int size, const float *gen) :
            szGenome{size} {
	nbCol = size;
	if (size > maxNbCol){
	    nbCol = maxNbCol;
	    nbRow = size/nbCol;
	}
	genome = gen;

    }


    void display() const {

	 std::cout << std::fixed;
         for( int i = 0 ; i < nbRow ; i++ ){
            for ( int j = 0; j < nbCol; j++ ){
                std::stringstream stream;
                stream << "x[" << i * nbCol + j << "]";
                if (( i*nbCol+j ) < 10 )
                    std::cout << std::setw(widthIndex1) << std::right << stream.str() << std::setw(widthSmbl) << std::right << " = ";
                else if (( i*nbCol+j ) < 100 )
		    std::cout << std::setw(widthIndex10) << std::right << stream.str() << std::setw(widthSmbl) << std::right << " = ";
		else std::cout << std::setw(widthIndex100) << std::right << stream.str() << std::setw(widthSmbl) << std::right << " = ";

            std::cout << std::setw( widthGenome ) << std::right <<  std::setprecision( precision ) << genome[j+i*nbCol] << std::setw( widthSmbl ) << std::right << " ";
            }
            std::cout<<std::endl;
         }
    }
    
};
}
