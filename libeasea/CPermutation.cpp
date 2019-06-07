/*
 * CPermutation.cpp
 *
 *  Created on: 4 auot 2018
 *      Author: Anna Ouskova Leonteva
 */

#include <CPermutation.h>

int * CPermutation::intPermutation(int length) {

  int * aux    = new int[length];
  int * result = new int[length];
  
  // Create an array from 0 to length - 1.
  // Create an random array of size length
  for (int i = 0; i < length; i++) {
    result[i] = i;
    aux[i] = CPseudoRandom::randInt(0,length-1);
  } 
    
  // Sort the random array with effect in result, and then we obtain a
  // permutation array between 0 and length - 1
  for (int i = 0; i < length; i++) {
    for (int j = i + 1; j < length; j++) {
      if (aux[i] > aux[j]) {
        int tmp;
        tmp = aux[i];
        aux[i] = aux[j];
        aux[j] = tmp;
        tmp = result[i];
        result[i] = result[j];
        result[j] = tmp;
      } 
    } 
  } 
  delete[] aux;

  return result;

}