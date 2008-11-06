/****************************************************************************
EaseaSym.cpp
Symbol table and other functions for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@inria.fr)
Copyright EVOlutionary LABoratory
INRIA Rocquencourt, Projet FRACTALES
Domaine de Voluceau
Rocquencourt BP 105
78153 Le Chesnay CEDEX
****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "EaseaSym.h"
                  
void debug(char *s){
  #ifdef _DEBUG
  printf (s);
  getchar ();
  #endif
  return;
  }

/////////////////////////////////////////////////////////////////////////////
// LinkedList commands

// template <class T> void CLList<T>::addLast(const T &p){
//   CListItem<T> *pSentinel, *pCurrent;
//   pSentinel=pCurrent=pHead;
//   while (pCurrent!=NULL){pSentinel=pCurrent; pCurrent=pCurrent->pNext;}
//   if (pSentinel != NULL) pSentinel->pNext=new CListItem<T>(pCurrent,NewObject);
//   else pHead=new CListItem<T>(pCurrent,NewObject);
//   }

template <class T> CListItem<T> *CLList<T>::walkToNextItem(){
  if (pNextItem==NULL) return NULL;
  if (pNextItem==pHead){
    pNextItem=pHead->pNext;
    return pHead;
    }
  pCurrentObject=pNextItem;
  pNextItem=pNextItem->pNext;
  return pCurrentObject;
  }
  
/////////////////////////////////////////////////////////////////////////////
// symbol construction/destruction

CSymbol::CSymbol(char *s){
  assert(s != NULL);
  int nLength = strlen(s);
  sName = new char[nLength + 1];
  strcpy(sName, s);      
  dValue = 0.0;
  nSize = 0;
  ObjectType=oUndefined;   
  bAlreadyPrinted=false;
  pType=pClass=NULL;
  pNextInBucket = NULL;
  pSymbolList=NULL;
  sString=NULL;
  }
  
CSymbol::~CSymbol(){
  delete[] sName;
  }

/////////////////////////////////////////////////////////////////////////////
// symbol  commands

void CSymbol::print(FILE *fp){
  CListItem<CSymbol*> *pSym;
  int i;

  if (strcmp(sName,"Genome")&&(TARGET!=DREAM)){   // If we are printing a user class other than the genome

    fprintf(fp,"\nclass %s {\npublic:\n// Default methods for class %s\n",sName,sName); // class  header

    fprintf(fp,"// Class members \n"); // Now, we must print the class members
    pSymbolList->reset();
    while (pSym=pSymbolList->walkToNextItem()){
      if (pSym->Object->ObjectType==oObject)
        fprintf(fp,"  %s %s;\n",pSym->Object->pType->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oPointer)
        fprintf(fp,"  %s *%s;\n",pSym->Object->pType->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray)
        fprintf(fp,"  %s %s[%d];\n",pSym->Object->pType->sName,pSym->Object->sName,pSym->Object->nSize/pSym->Object->pType->nSize);
    }

    fprintf(fp,"  %s(){  // Constructor\n",sName); // constructor
          pSymbolList->reset(); // in which we initialise all pointers to NULL
          while (pSym=pSymbolList->walkToNextItem())
            if (pSym->Object->ObjectType==oPointer)
              fprintf(fp,"    %s=NULL;\n",pSym->Object->sName);
    fprintf(fp,"  }\n"); // constructor
 
    fprintf(fp,"  %s(%s &EASEA_Var) {  // Copy constructor\n",sName,sName); // copy constructor
          pSymbolList->reset();
          while (pSym=pSymbolList->walkToNextItem()){
            if (pSym->Object->ObjectType==oObject)
              fprintf(fp,"    %s=EASEA_Var.%s;\n",pSym->Object->sName,pSym->Object->sName);
            if (pSym->Object->ObjectType==oArray){
              fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
              fprintf(fp,"       %s[EASEA_Ndx]=EASEA_Var.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
            }
            if (pSym->Object->ObjectType==oPointer){
              fprintf(fp,"    %s=(EASEA_Var.%s ? new %s(*(EASEA_Var.%s)) : NULL);\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
            }
          }
    fprintf(fp,"  }\n"); // copy constructor

    fprintf(fp,"  ~%s() {  // Destructor\n",sName); // destructor
          pSymbolList->reset();
          while (pSym=pSymbolList->walkToNextItem()){
            if (pSym->Object->ObjectType==oPointer)
              fprintf(fp,"    if (%s) delete %s;\n    %s=NULL;\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
          }
    fprintf(fp,"  }\n"); // destructor

    fprintf(fp,"  %s& operator=(%s &EASEA_Var) {  // Operator=\n",sName,sName); // operator=
    fprintf(fp,"    if (&EASEA_Var == this) return *this;\n");
          pSymbolList->reset();
          while (pSym=pSymbolList->walkToNextItem()){
            if (pSym->Object->ObjectType==oObject)
              fprintf(fp,"    %s = EASEA_Var.%s;\n",pSym->Object->sName,pSym->Object->sName);
            if (pSym->Object->ObjectType==oArray){
              fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
              fprintf(fp,"       %s[EASEA_Ndx] = EASEA_Var.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
            }
             if (pSym->Object->ObjectType==oPointer){
              fprintf(fp,"    if (%s) delete %s;\n",pSym->Object->sName,pSym->Object->sName);
              fprintf(fp,"    %s = (EASEA_Var.%s? new %s(*(EASEA_Var.%s)) : NULL);\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
            }
          }
    fprintf(fp,"  return *this;\n  }\n\n"); // operator<=

    fprintf(fp,"  bool operator==(%s &EASEA_Var) const {  // Operator==\n",sName); // operator==
          pSymbolList->reset();
          while (pSym=pSymbolList->walkToNextItem()){
            if (TARGET==GALIB){
              if (pSym->Object->ObjectType==oObject)
                fprintf(fp,"    if (%s!=EASEA_Var.%s) return gaFalse;\n",pSym->Object->sName,pSym->Object->sName);
              if (pSym->Object->ObjectType==oArray){
                fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
                fprintf(fp,"       if (%s[EASEA_Ndx]!=EASEA_Var.%s[EASEA_Ndx]) return gaFalse;}\n",pSym->Object->sName,pSym->Object->sName);
              }
               if (pSym->Object->ObjectType==oPointer){
                fprintf(fp,"    if (((%s) && (!EASEA_Var.%s)) || ((!%s) && (EASEA_Var.%s))) return gaFalse;\n",pSym->Object->sName,pSym->Object->sName, pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
                fprintf(fp,"    if ((%s)&&(%s!=EASEA_Var.%s)) return gaFalse;\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName, pSym->Object->pType->sName,pSym->Object->sName,pSym->Object->sName);
              }                                               
            }
            if (TARGET==EO){
              if (pSym->Object->ObjectType==oObject)
                fprintf(fp,"    if (%s!=EASEA_Var.%s) return false;\n",pSym->Object->sName,pSym->Object->sName);
              if (pSym->Object->ObjectType==oArray){
                fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
                fprintf(fp,"       if (%s[EASEA_Ndx]!=EASEA_Var.%s[EASEA_Ndx]) return false;}\n",pSym->Object->sName,pSym->Object->sName);
              }
               if (pSym->Object->ObjectType==oPointer){
                fprintf(fp,"    if (((%s) && (!EASEA_Var.%s)) || ((!%s) && (EASEA_Var.%s))) return false;\n",pSym->Object->sName,pSym->Object->sName, pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
                fprintf(fp,"    if ((%s)&&(%s!=EASEA_Var.%s)) return false;\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName, pSym->Object->pType->sName,pSym->Object->sName,pSym->Object->sName);
              }                                               
            }
          }
    if (TARGET==GALIB)  fprintf(fp,"  return gaTrue;\n  }\n\n"); // operator==
    if (TARGET==EO)  fprintf(fp,"  return true;\n  }\n\n"); // operator==

    fprintf(fp,"  bool operator!=(%s &EASEA_Var) const {return !(*this==EASEA_Var);} // operator!=\n\n",sName); // operator!=

    fprintf(fp,"  friend ostream& operator<< (ostream& os, const %s& EASEA_Var) { // Output stream insertion operator\n",sName); // Output stream insertion operator
          pSymbolList->reset();
          while (pSym=pSymbolList->walkToNextItem()){
            if (pSym->Object->ObjectType==oObject)
              fprintf(fp,"    os <<  \"%s:\" << EASEA_Var.%s << \"\\n\";\n",pSym->Object->sName,pSym->Object->sName);
            if (pSym->Object->ObjectType==oArray){
              fprintf(fp,"    {os << \"Array %s : \";\n",pSym->Object->sName);
              fprintf(fp,"     for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
              fprintf(fp,"       os << \"[\" << EASEA_Ndx << \"]:\" << EASEA_Var.%s[EASEA_Ndx] << \"\\t\";}\n    os << \"\\n\";\n",pSym->Object->sName);
            }
            if (pSym->Object->ObjectType==oPointer)
              fprintf(fp,"    if (EASEA_Var.%s) os << \"%s:\" << *(EASEA_Var.%s) << \"\\n\";\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
          }
    fprintf(fp,"    return os;\n  }\n\n"); // Output stream insertion operator

    fprintf(fp,"  friend istream& operator>> (istream& is, %s& EASEA_Var) { // Input stream extraction operator\n",sName); // Output stream insertion operator
          pSymbolList->reset();
          while (pSym=pSymbolList->walkToNextItem()){
            if ((pSym->Object->ObjectType==oObject)&&(strcmp(pSym->Object->pType->sName, "bool"))) 
              fprintf(fp,"    is >> EASEA_Var.%s;\n",pSym->Object->sName);
            if ((pSym->Object->ObjectType==oArray)&&(strcmp(pSym->Object->pType->sName, "bool"))) {
              fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
              fprintf(fp,"       is >> EASEA_Var.%s[EASEA_Ndx];}\n",pSym->Object->sName);
            }                                         
          }
    fprintf(fp,"    return is;\n  }\n\n"); // Input stream extraction operator

    if (sString) {
      if (bVERBOSE) printf ("Inserting Methods into %s Class.\n",sName);
      fprintf(fpOutputFile,"// User-defined methods:\n\n");
      fprintf(fpOutputFile,"%s\n",sString);
    }
  }
  else if (strcmp(sName,"Genome")&&(TARGET==DREAM)){   // If we are printing a user class other than the genome AND we are producing code for DREAM
    // We must first create a new file called Class.java
    char sFileName[1000];
    strcpy(sFileName, sRAW_PROJECT_NAME);
    for (i=strlen(sFileName);(sFileName[i]!='/')&&(sFileName[i]!='\\')&&(i!=0);i--);
    sFileName[i]=0;
    if (i!=0) {sFileName[i]='/';sFileName[i+1]=0;}
    if ((i!=0)&&(OPERATING_SYSTEM==WINDOWS)) {sFileName[i]='\\';sFileName[i+1]=0;}
    strcat(sFileName,sName); strcat(sFileName,".java");
    if (bVERBOSE) printf("Creating %s...\n",sFileName);
    fp=fopen(sFileName,"w");

    fprintf(fp,"//********************************************\n//\n//  %s.java\n//\n//  Java file generated by EASEA-DREAM v0.7patch17\n",sName);
    fprintf(fp,"//\n//********************************************\n\n");
    fprintf(fp,"\nimport drm.agentbase.Logger;\n");
    fprintf(fp,"import java.util.Arrays;\nimport java.util.Vector;\nimport java.util.Collection;\nimport java.util.Iterator;\nimport java.io.*;\n\n");
  
    fprintf(fp,"public class %s implements java.io.Serializable { \n",sName); // Now, we must print the class members
    fprintf(fp,"// Class members \n"); // Now, we must print the class members
    pSymbolList->reset();
    while (pSym=pSymbolList->walkToNextItem()){
      if ((pSym->Object->ObjectType==oObject)||(pSym->Object->ObjectType==oPointer))
        fprintf(fp,"  %s %s;\n",pSym->Object->pType->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray)
        fprintf(fp,"  public %s[] %s = new %s[%d];\n",pSym->Object->pType->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->nSize/pSym->Object->pType->nSize);
    }

    fprintf(fp,"\n  %s(){  // Constructor\n",sName); // constructor
          pSymbolList->reset(); // in which we initialise all pointers to NULL
          while (pSym=pSymbolList->walkToNextItem())
            if (pSym->Object->ObjectType==oPointer)
              fprintf(fp,"    %s=null;\n",pSym->Object->sName);
    fprintf(fp,"  }\n\n"); // constructor
 
    fprintf(fp,"  public %s (%s EZ_%s) {  \n// Memberwise Cloning\n",sName,sName,sName); // copy constructor
          pSymbolList->reset();
          while (pSym=pSymbolList->walkToNextItem()){
            if (pSym->Object->ObjectType==oObject)
              fprintf(fp,"    %s=EZ_%s.%s;\n",pSym->Object->sName,sName,pSym->Object->sName);
            if (pSym->Object->ObjectType==oArray){
              fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
              if (pSym->Object->pType->ObjectType==oUserClass) fprintf(fp,"       %s[EASEA_Ndx]=new %s(EZ_%s.%s[EASEA_Ndx]);}\n",pSym->Object->sName,pSym->Object->pType->sName,sName,pSym->Object->sName);
              else fprintf(fp,"       %s[EASEA_Ndx]=EZ_%s.%s[EASEA_Ndx];}\n",pSym->Object->sName,sName,pSym->Object->sName);
            }
            if (pSym->Object->ObjectType==oPointer){
              fprintf(fp,"    %s=(EZ_%s.%s!=null ? new %s(EZ_%s.%s) : null);\n",pSym->Object->sName,sName,pSym->Object->sName,pSym->Object->pType->sName,sName,pSym->Object->sName);
            }
          }
    fprintf(fp,"  }\n\n"); // copy constructor

    fprintf(fp,"  public Object copy() throws CloneNotSupportedException {\n");
    fprintf(fp,"    %s EZ_%s = new %s();\n",sName,sName,sName);
    fprintf(fp,"    // Memberwise copy\n");
          pSymbolList->reset();
          while (pSym=pSymbolList->walkToNextItem()){
            if (pSym->Object->ObjectType==oObject)
              fprintf(fp,"    EZ_%s.%s = %s;\n",sName,pSym->Object->sName,pSym->Object->sName);
            if (pSym->Object->ObjectType==oArray){
              fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
              if (pSym->Object->pType->ObjectType==oUserClass) fprintf(fp,"       EZ_%s.%s[EASEA_Ndx] = new %s(%s[EASEA_Ndx]);}\n",sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
              else fprintf(fp,"       EZ_%s.%s[EASEA_Ndx] = %s[EASEA_Ndx];}\n",sName,pSym->Object->sName,pSym->Object->sName);
            }
             if (pSym->Object->ObjectType==oPointer){
              fprintf(fp,"    EZ_%s.%s = (%s != null ? new %s(%s) : null);\n",sName,pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
            }
          }
    fprintf(fp,"    return EZ_%s;\n  }\n\n",sName); // operator<=

    fprintf(fp,"  public String toString() {\n");
    fprintf(fp,"    String EASEA_S = new String();\n");
    fprintf(fp,"    //Default display function\n");
          pSymbolList->reset();
          while (pSym=pSymbolList->walkToNextItem()){
            if (pSym->Object->ObjectType==oObject)
              fprintf(fp,"    EASEA_S = EASEA_S +  \"%s:\" + %s + \"\\n\";\n",pSym->Object->sName,pSym->Object->sName);
            if (pSym->Object->ObjectType==oArray){
              fprintf(fp,"    {EASEA_S = EASEA_S + \"Array %s : \";\n",pSym->Object->sName);
              fprintf(fp,"     for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
              fprintf(fp,"       EASEA_S = EASEA_S + \"[\" + EASEA_Ndx + \"]:\" + %s[EASEA_Ndx] + \"\\t\";}\n    EASEA_S = EASEA_S + \"\\n\";\n",pSym->Object->sName);
            }
            if (pSym->Object->ObjectType==oPointer)
              fprintf(fp,"    if (%s!=null) EASEA_S = EASEA_S + \"%s:\" + %s + \"\\n\";\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
          }
    fprintf(fp,"    return EASEA_S;\n  }\n}\n\n"); // Output stream insertion operator
    fclose(fp);
  }
  else {
    fprintf(fp,"// Class members \n"); // Now, we must print the class members
    pSymbolList->reset();
    while (pSym=pSymbolList->walkToNextItem()){
      if (pSym->Object->ObjectQualifier==1) // 1=Static
        fprintf(fp,"  static");
      if (pSym->Object->ObjectType==oObject)
        fprintf(fp,"  %s %s;\n",pSym->Object->pType->sName,pSym->Object->sName);
      if ((pSym->Object->ObjectType==oPointer)&&(TARGET==DREAM))
        fprintf(fp,"  %s %s;\n",pSym->Object->pType->sName,pSym->Object->sName);
      if ((pSym->Object->ObjectType==oPointer)&&(TARGET!=DREAM))
        fprintf(fp,"  %s *%s;\n",pSym->Object->pType->sName,pSym->Object->sName);
      if ((pSym->Object->ObjectType==oArray)&&(TARGET==DREAM))
        fprintf(fp,"  public %s[] %s = new %s[%d];\n",pSym->Object->pType->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->nSize/pSym->Object->pType->nSize);
      if ((pSym->Object->ObjectType==oArray)&&(TARGET!=DREAM))
        fprintf(fp,"  %s %s[%d];\n",pSym->Object->pType->sName,pSym->Object->sName,pSym->Object->nSize/pSym->Object->pType->nSize);
    }
  }

  if (strcmp(sName,"Genome"))
    fprintf(fp,"};\n",sName);
}

void CSymbol::printClasses(FILE *fp){
  CListItem<CSymbol*> *pSym;
  if (bAlreadyPrinted) return;
  bAlreadyPrinted=true;
  pSymbolList->reset();
  while (pSym=pSymbolList->walkToNextItem())
    if ((pSym->Object->pType->ObjectType==oUserClass)&&(!pSym->Object->pType->bAlreadyPrinted))
      pSym->Object->pType->printClasses(fp);
  print(fp);
}

void CSymbol::printAllSymbols(FILE *fp, char *sCompleteName, EObjectType FatherType, CListItem<CSymbol *> *pSym){
  char sNewCompleteName[1000], s[20];
  strcpy(sNewCompleteName, sCompleteName);
  do {
    if (pSym->Object->pType->ObjectType==oUserClass){
      if (FatherType==oPointer) 
        strcat(sNewCompleteName,"->");
      else strcat(sNewCompleteName,".");
      strcat(sNewCompleteName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray) {
        strcat(sNewCompleteName,"[");
        sprintf(s,"%d",pSym->Object->nSize/pSym->Object->pType->nSize);
        strcat(sNewCompleteName,s);
        strcat(sNewCompleteName,"]");
      }
      if (pSym->Object->pType==pSym->Object->pClass) 
        fprintf(fp,"%s\n",sNewCompleteName);
      else printAllSymbols(fp, sNewCompleteName, pSym->Object->ObjectType, pSym->Object->pType->pSymbolList->getHead());
      strcpy(sNewCompleteName, sCompleteName);
    }
    else {
      if (FatherType==oPointer) 
        strcat(sNewCompleteName,"->");
      else strcat(sNewCompleteName,".");
      strcat(sNewCompleteName,pSym->Object->sName);
      if (sNewCompleteName,pSym->Object->ObjectType==oArray) {
        strcat(sNewCompleteName,"[");
        sprintf(s,"%d",pSym->Object->nSize/pSym->Object->pType->nSize);
        strcat(sNewCompleteName,s);
        strcat(sNewCompleteName,"]");
      }
      fprintf(fp,"%s\n",sNewCompleteName);
      strcpy(sNewCompleteName, sCompleteName);
    }
  } while (pSym=pSym->pNext);
}
  

/////////////////////////////////////////////////////////////////////////////
// symboltable construction/destruction

CSymbolTable::CSymbolTable(){
  for (int i = 0; i < BUCKET_SIZE; i++) {
    saBucket[i] = NULL;
    }
  }

CSymbolTable::~CSymbolTable(){
  for (int i = 0; i < BUCKET_SIZE; i++) {
    CSymbol* pSym = saBucket[i];
    while (pSym!= NULL) {
      CSymbol* pNextSym = pSym->pNextInBucket;
      delete pSym;
      pSym = pNextSym;
      }
    }
  }

/////////////////////////////////////////////////////////////////////////////
// symbol table  commands

int CSymbolTable::hash(const char* s) const{
  assert(s != NULL);
  int i = 0;
  while (*s != '\0') {
    i = i << 1 ^ *s++;
    }
  i %= BUCKET_SIZE;
  if (i < 0)  i *= -1;
  return i;
  }

CSymbol* CSymbolTable::insert(CSymbol *pSymbol){
  int i = hash(pSymbol->sName);
  CSymbol* pSym;
  for (pSym = saBucket[i]; pSym != NULL; pSym = pSym->pNextInBucket)
    if (strcmp(pSym->sName, pSymbol->sName) == 0){
      delete pSymbol;
      return pSym;
      }
  pSym = pSymbol;
  pSym->pNextInBucket = saBucket[i];
  saBucket[i] = pSym;
  return pSym;
  }

CSymbol* CSymbolTable::find(const char *s){
  int i = hash(s);
  for (CSymbol* pSym = saBucket[i]; pSym != NULL; pSym = pSym->pNextInBucket)
    if (strcmp(pSym->sName,s) == 0)
      return pSym;     
  return NULL;
  }
