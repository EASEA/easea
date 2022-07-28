/*
 * CMeta.h
 *
 *  Created on: 5 septembre 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef META_H
#define META_H

#include <stdlib.h>
/* Some useful things for High Performance */

/* Metaprogrammation => function power in compiling time */
template<int n>
inline
double power(const double& m){
  return power<n-1>(m) * m;
}
template <>
inline
double power<1>(const double& m){
  return m;
}
template <>
inline
double power<0>(const double&){
  return 1.0;
}
/* Metaprogrammation => condition in compiling time */
template<typename Ttype>
void TrueStatementIf(Ttype arg1, Ttype arg2){
    arg1 = arg2;
}
template<bool predicate>
class IF
{
};
template<>
class IF<true>
{
public :
template<typename Ttype>
    static inline void func(Ttype arg1, Ttype arg2){
        TrueStatementIf(arg1, arg2);
    }
};

#endif


