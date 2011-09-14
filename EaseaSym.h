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


#include <iostream>
#include <sstream>

extern void debug (char *);

enum EObjectType {oUserClass, oBaseClass, oObject, oPointer, oArray, oMacro, oUndefined, oArrayPointer};

/////////////////////////////////////////////////////////////////////////////
// Object

template <class T> class CLList;

template <class T> class CListItem {
friend class CLList<T>;
// Operations
public:     
  CListItem(const CListItem<T> *pOldObject, const T &NewObject)
    :Object(NewObject), pNext((CListItem<T>*)pOldObject){}
  ~CListItem();
// Attributes  
  T Object;                     // The object
  CListItem<T> *pNext; // pointer to the next object in the list      
};

/////////////////////////////////////////////////////////////////////////////
// Linked List

template <class T> class CLList {
private :
  CListItem<T> *pHead;
  CListItem<T> *pCurrentObject;
  CListItem<T> *pNextItem;
public :
  CLList():pHead(NULL),pCurrentObject(NULL){}
  ~CLList();
  void addLast(const T &p);
  void addFirst(const T &NewObject) {pHead = new CListItem<T>(pHead, NewObject);}
  void reset(){pCurrentObject=pNextItem=pHead;}
  CListItem<T> * getHead(){return pHead;}
  T CurrentObject(){return pCurrentObject?pCurrentObject->Object:NULL;}
  T nextObject(){return pNextItem ?pNextItem->Object:NULL;}
  CListItem<T> *walkToNextItem();
  CListItem<T> *remove(T *p);
  };

 /////////////////////////////////////////////////////////////////////////////
// Symbol

class CSymbol {
public:
 CSymbol(char *s);
 virtual ~CSymbol();

// Attributes
public:
  CSymbol* pNextInBucket; // next symbol in bucket list (hash code)

  char *sName;     // symbol name
  double dValue;     // value attached to symbol
  char *sString;    // attached string
  int nSize;              // size of the object represented by the symbol
  bool bAlreadyPrinted; // no comment
  EObjectType ObjectType;  // variable, array, pointer, class,...
  int ObjectQualifier;  // 0=Normal, 1=Static, 2=oObservable, ...
  CSymbol *pType; // pointer to the variable/array/.. type
  CSymbol *pClass;  // pointer to the variable class in which it is defined.
  CLList<CSymbol *> *pSymbolList; // pointer on a list of class members (if the symbol is a class)
                                        // pointer on the class (if the symbol is a variable)
  
// Operations
public:
  void print(FILE *f);
  void printClasses(FILE *f);      
  void printAllSymbols(FILE *f, char *, EObjectType, CListItem<CSymbol *> *pSym);
  void printUserClasses(FILE* fp);
  void printUC(FILE* fp);
  void serializeIndividual(FILE *fp, char* sCompleteName);
  void deserializeIndividual(FILE *fp, char* sCompleteName);

};

/////////////////////////////////////////////////////////////////////////////
// symboltable

#define BUCKET_SIZE 4093       // prime number

class CSymbolTable {
public:
  CSymbolTable();
  virtual ~CSymbolTable();
  
// Attributes
protected:
  CSymbol* saBucket[BUCKET_SIZE];    // array of buckets

// Operations
protected:
  int hash(const char* sName) const;
public:
  CSymbol* insert(CSymbol *pSymbol);
  CSymbol* find(const char* s);
};


using namespace std;

class OPCodeDesc {
 public:
  unsigned arity;
  string* opcode;
  string* realName;
  ostringstream cpuCodeStream;
  ostringstream gpuCodeStream;
  bool isERC;

  void show(void);
  static void sort(OPCodeDesc** opDescs, unsigned len);
  OPCodeDesc();
};

#endif
