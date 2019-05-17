/*
 * COperator.h
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef COPERATOR_H
#define COPERATOR_H

#include <map>
#include <string>
#include <iostream>
#include <tuple>
#include <typeinfo>

#include "CLogger.h"

#define NOINLINE __attribute__((noinline))
#define ALWAYS_INLINE __attribute__((always_inline))


/* This is a base class of operators ( Selection, Crossover, Mutation)
 * It takes input variable variables for each type of operator.
 * Some  utilis for getting tuple's elements with a runtime index
 * Actually, it's used for access to parameters of COpertaor.
 *
 * */

template<class T, class U>
struct retVal {
    static T& takeAt(U& u) NOINLINE {
        LOG_ERROR(errorCode::type, "return type is mismatched");
    }
};
template<class T>
struct retVal<T,T> {
    static T& takeAt(T& t) ALWAYS_INLINE {  return t; }
};

template<class TOut, class TTuple, unsigned size>
struct getElemById {
    static TOut& takeAt(TTuple& t, int id) ALWAYS_INLINE {
        constexpr unsigned limit = size-1;
        if(id == limit) {
            typedef typename std::tuple_element<limit,TTuple>::type TElem;
            return retVal<TOut, TElem>::takeAt(std::get<limit>(t));
        }
        else
            return getElemById<TOut,TTuple,limit>::takeAt(t, id);
    }
};
template<class TOut, class TTuple>
struct getElemById<TOut, TTuple, 0> {
    static TOut& takeAt(TTuple&, int) NOINLINE {
        LOG_ERROR(errorCode::value, "bad index");
    }
};


template<class TOut, class TTuple>
TOut& tupleAt(TTuple& t, int id) {
    constexpr unsigned size = std::tuple_size<TTuple>::value;
    return getElemById<TOut, TTuple, size>::takeAt(t, id);
};

template<typename... T>
class COperator {

private:
    static const unsigned short int size = sizeof...(T);
    std::tuple<T...> parametres_;

public:
    COperator(const T&... t) : parametres_(t...){};

    template<typename TElem>
    auto get( const long unsigned int id){

        if (id < size)
            return tupleAt<TElem>(parametres_, id);
	return (TElem)0;
	
    }

    virtual ~COperator(){};

};

#endif
