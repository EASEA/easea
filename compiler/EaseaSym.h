/****************************************************************************
EaseaSym.h
Symbol table and other functions for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@inria.fr)
Copyright EVOlutionary LABoratory
INRIA Rocquencourt, Projet FRACTALES
Domaine de Voluceau
Rocquencourt BP 105
78153 Le Chesnay CEDEX
****************************************************************************/

#ifndef SYMBOL_H
#define SYMBOL_H
#include "Easea.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "debug.h"
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <list>
#include <memory>

enum EObjectType { oUserClass, oBaseClass, oObject, oPointer, oArray, oMacro, oUndefined, oArrayPointer };

/////////////////////////////////////////////////////////////////////////////
// Symbol

class CSymbol
{
    public:
	CSymbol(const char* s);

	// Attributes
    public:
	std::string sName; // symbol name
	double dValue; // value attached to symbol
	std::string sString; // attached string
	int nSize; // size of the object represented by the symbol
	bool bAlreadyPrinted; // no comment
	EObjectType ObjectType; // variable, array, pointer, class,...
	int ObjectQualifier; // 0=Normal, 1=Static, 2=oObservable, ...
	CSymbol* pType; // pointer to the variable/array/.. type
	CSymbol* pClass; // pointer to the variable class in which it is defined.
	std::list<std::unique_ptr<CSymbol>> pSymbolList; // pointer on a list of class members (if the symbol is a class)
		// pointer on the class (if the symbol is a variable)

	// Operations
    public:
	void print(FILE* f);
	void printClasses(FILE* f);
	template <typename Iterator>
	void printAllSymbols(FILE* f, char*, EObjectType, Iterator begin, Iterator end);
	void printUserClasses(FILE* fp);
	void printUC(FILE* fp);
	void serializeIndividual(FILE* fp, char* sCompleteName);
	void dtor(FILE* fp);
};

/////////////////////////////////////////////////////////////////////////////
// symboltable

class CSymbolTable
{
    public:
	CSymbol* insert(std::unique_ptr<CSymbol>&& symbol);
	CSymbol* find(const char* s);

	CSymbolTable()=default;
	CSymbolTable(CSymbolTable const&)=delete;
	CSymbolTable& operator=(CSymbolTable const&)=delete;
	CSymbolTable(CSymbolTable&&)=default;
	CSymbolTable& operator=(CSymbolTable&&)=default;

    protected:
	std::map<std::string, std::unique_ptr<CSymbol>> hashmap;
};

class OPCodeDesc
{
    public:
	unsigned arity;
	std::string opcode;
	std::string realName;
	std::ostringstream cpuCodeStream;
	std::ostringstream gpuCodeStream;
	bool isERC;

	void show(void);
	//static void sort(OPCodeDesc** opDescs, unsigned len);
	OPCodeDesc();
};

#endif
