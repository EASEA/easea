#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif
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
#include "debug.h"

using std::cout;
using std::endl;

void debug(char *s)
{
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

template <class T> CListItem<T> *CLList<T>::walkToNextItem()
{
	if (pNextItem==NULL) return NULL;
	if (pNextItem==pHead)
	{
		pNextItem=pHead->pNext;
		return pHead;
	}
	pCurrentObject=pNextItem;
	pNextItem=pNextItem->pNext;
	return pCurrentObject;
}

/////////////////////////////////////////////////////////////////////////////
// symbol construction/destruction

CSymbol::CSymbol(const char *s)
{
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

CSymbol::~CSymbol()
{
	delete[] sName;
}



/////////////////////////////////////////////////////////////////////////////
// symbol  commands
void CSymbol::print(FILE *fp)
{
///*
	CListItem<CSymbol*> *pSym;
	int i;

	// check: are we printing a user class that is different from Genome?
	if (strcmp(sName,"Genome"))
	{   
		// if we are printing a user class other than the genome
		fprintf(fp,"\nclass %s {\npublic:\n// Default methods for class %s\n",sName,sName); // class  header

		//fprintf(fp,"// Class members \n"); // Now, we must print the class members
		//  pSymbolList->reset();
		//  while (pSym=pSymbolList->walkToNextItem()){
		//  if (pSym->Object->ObjectType==oObject)
		//  fprintf(fp,"  %s %s;\n",pSym->Object->pType->sName,pSym->Object->sName);
		//  if (pSym->Object->ObjectType==oPointer)
		//  fprintf(fp,"  %s *%s;\n",pSym->Object->pType->sName,pSym->Object->sName);
		//  if (pSym->Object->ObjectType==oArray)
		//  fprintf(fp,"  %s %s[%d];\n",pSym->Object->pType->sName,pSym->Object->sName,pSym->Object->nSize/pSym->Object->pType->nSize);
		// }

		// check on the type of target
		if( TARGET==CUDA )
		{ 
			// here we we are generating function to copy objects from host memory to gpu's.
			bool isFlatClass = true;
			pSymbolList->reset();
			while ((pSym=pSymbolList->walkToNextItem()))
			{
				//DEBUG_PRT("analyse flat %s",pSym->Object->pType->sName);
				if( pSym->Object->ObjectType == oPointer ) //|| (pSym->Object->pType->ObjectType == oObject) )
				{  
					isFlatClass = false;
					break;
				}
			}

			//DEBUG_PRT("Does %s flat class : %s",sName,(isFlatClass?"yes":"no"));
			pSymbolList->reset();      
			fprintf(fp,"  %s* cudaSendToGpu%s(){\n",sName,sName);
			fprintf(fp,"    %s* ret=NULL;\n",sName);

			if( isFlatClass )
			{
				fprintf(fp,"    cudaMalloc((void**)&ret,sizeof(%s));\n",sName);
				fprintf(fp,"    cudaMemcpy(ret,this,sizeof(%s),cudaMemcpyHostToDevice);\n",sName);
				fprintf(fp,"    return ret;\n");	
			}
			else
			{
				fprintf(fp,"    %s tmp;\n",sName);
				fprintf(fp,"    memcpy(&tmp,this,sizeof(%s));\n",sName);

				while ((pSym=pSymbolList->walkToNextItem()))
				{
					if( pSym->Object->ObjectType == oPointer )  //|| (pSym->Object->pType->ObjectType == oObject) )
					{
						fprintf(fp,"    tmp.%s=this->%s->cudaSendToGpu%s();\n",
								pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName);
					}
				}
				fprintf(fp,"    cudaMalloc((void**)&ret,sizeof(%s));\n",sName);
				fprintf(fp,"    cudaMemcpy(ret,&tmp,sizeof(%s),cudaMemcpyHostToDevice);\n",sName);
				fprintf(fp,"    return ret;\n");
			}

			fprintf(fp,"  }\n\n");

			// another CUDA-specific function
			fprintf(fp,"  void cudaGetFromGpu%s(%s* dev_ptr){\n",sName,sName);
			fprintf(fp,"    %s* ret=NULL;\n",sName); 	

			if( isFlatClass )
			{
				fprintf(fp,"    ret = (%s*)malloc(sizeof(%s));\n",sName,sName);
				fprintf(fp,"    cudaMemcpy(ret,dev_ptr,sizeof(%s),cudaMemcpyDeviceToHost);\n",sName);
				//while (pSym=pSymbolList->walkToNextItem())
				//fprintf(fp,"    this->%s=ret->%s;\n",pSym->Object->sName,pSym->Object->sName);      
			}
		  fprintf(fp,"  }\n\n");
		}

		// creation of class constructor
		fprintf(fp,"  %s(){  // Constructor\n",sName); // constructor
		pSymbolList->reset(); // in which we initialise all pointers to NULL

		while ((pSym=pSymbolList->walkToNextItem()))
		{
			if (pSym->Object->ObjectType==oPointer)
				fprintf(fp,"    %s=NULL;\n",pSym->Object->sName);

			if (pSym->Object->ObjectType==oArrayPointer)
			{
				fprintf(fp,"    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",(int)(pSym->Object->nSize/sizeof(char*)));
				fprintf(fp,"         %s[EASEA_Ndx]=NULL;\n",pSym->Object->sName);
			}
		}
		fprintf(fp,"  }\n"); // end of constructor

		// creation of copy constructor
		fprintf(fp,"  %s(const %s &EASEA_Var) {  // Copy constructor\n",sName,sName); // copy constructor
		pSymbolList->reset();

		while ((pSym=pSymbolList->walkToNextItem()))
		{
			if (pSym->Object->ObjectType==oObject)
				fprintf(fp,"    %s=EASEA_Var.%s;\n",pSym->Object->sName,pSym->Object->sName);

			if (pSym->Object->ObjectType==oArray)
			{
				fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						pSym->Object->nSize/pSym->Object->pType->nSize);
				fprintf(fp,"       %s[EASEA_Ndx]=EASEA_Var.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
			}

			if (pSym->Object->ObjectType==oPointer)
			{
				fprintf(fp,"    %s=(EASEA_Var.%s ? new %s(*(EASEA_Var.%s)) : NULL);\n",
						pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
			}

			if( pSym->Object->ObjectType==oArrayPointer )
			{
				fprintf(fp,"    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",(int)(pSym->Object->nSize/sizeof(char*)));
				fprintf(fp,"        if( EASEA_Var.%s[EASEA_Ndx] ) %s[EASEA_Ndx] = new %s(*(EASEA_Var.%s[EASEA_Ndx]));\n",
						pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
				fprintf(fp,"        else  %s[EASEA_Ndx] = NULL;\n",pSym->Object->sName);
			}
		}
		fprintf(fp,"  }\n"); // end of copy constructor

		// creation of destructor
		fprintf(fp,"  virtual ~%s() {  // Destructor\n",sName); // destructor
		pSymbolList->reset();
		while ((pSym=pSymbolList->walkToNextItem()))
		{
			if (pSym->Object->ObjectType==oPointer)
				fprintf(fp,"    if (%s) delete %s;\n    %s=NULL;\n",
						pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);

			if( pSym->Object->ObjectType==oArrayPointer )
			{
				fprintf(fp,"    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						(int)(pSym->Object->nSize/sizeof(char*)));
				fprintf(fp,"        if( %s[EASEA_Ndx] ) delete %s[EASEA_Ndx];\n",
						pSym->Object->sName,pSym->Object->sName);
			}
		}
		fprintf(fp,"  }\n"); // end of destructor

		// creation of serializer
		fprintf(fp,"  string serializer() {  // serialize\n"); // serializer
		fprintf(fp,"  \tostringstream EASEA_Line(ios_base::app);\n");
		pSymbolList->reset();
		while ((pSym=pSymbolList->walkToNextItem()))
		{
			// check: is it a user-defined class?
			if(pSym->Object->pType->ObjectType==oUserClass)
			{
				if (pSym->Object->ObjectType==oArrayPointer)
				{
					// it's an array of pointers
					fprintf(fp,"\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++){\n",
							(int)(pSym->Object->nSize/sizeof(char*)));
					fprintf(fpOutputFile,"\t\tif(this->%s[EASEA_Ndx] != NULL){\n",pSym->Object->sName);
					fprintf(fpOutputFile,"\t\t\tEASEA_Line << \"\\a \";\n");
					fprintf(fpOutputFile,"\t\t\tEASEA_Line << this->%s[EASEA_Ndx]->serializer() << \" \";\n",
							pSym->Object->sName);
					fprintf(fpOutputFile,"\t}\n");
					fprintf(fpOutputFile,"\t\telse\n");
					fprintf(fpOutputFile,"\t\t\tEASEA_Line << \"NULL\" << \" \";\n");
					fprintf(fpOutputFile,"}\n");
				}
				else
				{
					// it's not an array of pointers
					fprintf(fpOutputFile,"\tif(this->%s != NULL){\n",pSym->Object->sName);
					fprintf(fpOutputFile,"\t\tEASEA_Line << \"\\a \";\n");
					fprintf(fpOutputFile,"\t\tEASEA_Line << this->%s->serializer() << \" \";\n",pSym->Object->sName);
					fprintf(fpOutputFile,"}\n");
					fprintf(fpOutputFile,"\telse\n");
					fprintf(fpOutputFile,"\t\tEASEA_Line << \"NULL\" << \" \";\n");
				}
			}
			else
			{
				// it's not a user-defined class
				if (pSym->Object->ObjectType==oObject)
				{
					fprintf(fpOutputFile,"\tEASEA_Line << this->%s << \" \";\n",pSym->Object->sName);
				}

				// it's a classical array
				if(pSym->Object->ObjectType==oArray)
				{
					fprintf(fpOutputFile,"\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
							pSym->Object->nSize/pSym->Object->pType->nSize);
					fprintf(fpOutputFile,"\t\tEASEA_Line << this->%s[EASEA_Ndx] <<\" \";\n", pSym->Object->sName);
				}
			}
		} // end while

		fprintf(fp,"  \treturn EASEA_Line.str();\n");
		fprintf(fp,"  }\n"); // end of serializer

		// creation of deserializer
		fprintf(fp,"  void deserializer(istringstream* EASEA_Line) {  // deserialize\n"); // deserializer
		fprintf(fp,"  \tstring line;\n");
		pSymbolList->reset();
		while ((pSym=pSymbolList->walkToNextItem()))
		{
			if(pSym->Object->pType->ObjectType==oUserClass)
			{
				if (pSym->Object->ObjectType==oArrayPointer)
				{
					fprintf(fp,"\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++){\n",
							(int)(pSym->Object->nSize/sizeof(char*)));
					fprintf(fpOutputFile,"\t\t(*EASEA_Line) >> line;\n");
					fprintf(fpOutputFile,"\t\tif(strcmp(line.c_str(),\"NULL\")==0)\n");
					fprintf(fpOutputFile,"\t\t\tthis->%s[EASEA_Ndx] = NULL;\n",pSym->Object->sName);
					fprintf(fpOutputFile,"\t\telse{\n");
					fprintf(fpOutputFile,"\t\t\tthis->%s[EASEA_Ndx] = new %s;\n",
							pSym->Object->sName, pSym->Object->pType->sName);
					fprintf(fpOutputFile,"\t\t\tthis->%s[EASEA_Ndx]->deserializer(EASEA_Line);\n",
							pSym->Object->sName);
					fprintf(fpOutputFile,"\t\t}");
					fprintf(fpOutputFile,"\t}");
				}
				else
				{
					fprintf(fpOutputFile,"\t(*EASEA_Line) >> line;\n");
					fprintf(fpOutputFile,"\tif(strcmp(line.c_str(),\"NULL\")==0)\n");
					fprintf(fpOutputFile,"\t\tthis->%s = NULL;\n",pSym->Object->sName);
					fprintf(fpOutputFile,"\telse{\n");
					fprintf(fpOutputFile,"\t\tthis->%s = new %s;\n",pSym->Object->sName, sName);
					fprintf(fpOutputFile,"\t\tthis->%s->deserializer(EASEA_Line);\n",pSym->Object->sName);
					fprintf(fpOutputFile,"\t}");
				}
			}
			else
			{
				if (pSym->Object->ObjectType==oObject)
				{
					fprintf(fpOutputFile,"\t(*EASEA_Line) >> this->%s;\n",pSym->Object->sName);
				}

				if(pSym->Object->ObjectType==oArray)
				{
					fprintf(fpOutputFile,"\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
							pSym->Object->nSize/pSym->Object->pType->nSize);
					fprintf(fpOutputFile,"\t\t(*EASEA_Line) >> this->%s[EASEA_Ndx];\n", pSym->Object->sName);
				}
			}
		} // end while
		fprintf(fp,"  }\n"); // end of deserializer

		// creation of operator to assess individual equality
		fprintf(fp,"  %s& operator=(const %s &EASEA_Var) {  // Operator=\n",sName,sName); // operator=
		fprintf(fp,"    if (&EASEA_Var == this) return *this;\n");
		pSymbolList->reset();

		while ((pSym=pSymbolList->walkToNextItem()))
		{
			if (pSym->Object->ObjectType==oObject)
				fprintf(fp,"    %s = EASEA_Var.%s;\n",pSym->Object->sName,pSym->Object->sName);

			if (pSym->Object->ObjectType==oArray)
			{
				fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						pSym->Object->nSize/pSym->Object->pType->nSize);
				fprintf(fp,"       %s[EASEA_Ndx] = EASEA_Var.%s[EASEA_Ndx];}\n",
						pSym->Object->sName,pSym->Object->sName);
			}

			if (pSym->Object->ObjectType==oArrayPointer)
			{
				fprintf(fp,"    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",(int)(pSym->Object->nSize/sizeof(char*)));
				fprintf(fp,"      if(EASEA_Var.%s[EASEA_Ndx]) %s[EASEA_Ndx] = new %s(*(EASEA_Var.%s[EASEA_Ndx]));\n",
						pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
			}

			if (pSym->Object->ObjectType==oPointer)
			{
				fprintf(fp,"    if (%s) delete %s;\n",pSym->Object->sName,pSym->Object->sName);
				fprintf(fp,"    %s = (EASEA_Var.%s? new %s(*(EASEA_Var.%s)) : NULL);\n",
						pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
			}

		}// end while
		fprintf(fp,"  return *this;\n  }\n\n"); // end of operator <=

		// creation of operator ==
		fprintf(fp,"  bool operator==(%s &EASEA_Var) const {  // Operator==\n",sName); // operator==
		pSymbolList->reset();
		while ((pSym=pSymbolList->walkToNextItem()))
		{
			if (TARGET==CUDA || TARGET==STD)
			{
				if (pSym->Object->ObjectType==oObject)
					fprintf(fp,"    if (%s!=EASEA_Var.%s) return false;\n",pSym->Object->sName,pSym->Object->sName);
				if (pSym->Object->ObjectType==oArray )
				{
					fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						pSym->Object->nSize/pSym->Object->pType->nSize);
					fprintf(fp,"       if (%s[EASEA_Ndx]!=EASEA_Var.%s[EASEA_Ndx]) return false;}\n",
						pSym->Object->sName,pSym->Object->sName);
				}
				if ( pSym->Object->ObjectType==oArrayPointer)
				{
					fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						(int)(pSym->Object->nSize/sizeof(char*)));
					fprintf(fp,"       if (%s[EASEA_Ndx]!=EASEA_Var.%s[EASEA_Ndx]) return false;}\n",
						pSym->Object->sName,pSym->Object->sName);
				}
				if (pSym->Object->ObjectType==oPointer)
				{
					fprintf(fp,"    if (((%s) && (!EASEA_Var.%s)) || ((!%s) && (EASEA_Var.%s))) return false;\n",
						pSym->Object->sName,pSym->Object->sName, pSym->Object->sName,
						pSym->Object->sName);
					fprintf(fp,"    if ((%s)&&(%s!=EASEA_Var.%s)) return false;\n",
						pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
				}                                               
			}
		}
		if (TARGET==CUDA || TARGET==STD)  
			fprintf(fp,"  return true;\n  }\n\n"); // end of operator==

		// creation of operator !=
		fprintf(fp,"  bool operator!=(%s &EASEA_Var) const {return !(*this==EASEA_Var);} // operator!=\n\n",sName); // operator!=
		
		// creation of output stream insertion operator
		fprintf(fp,"  friend ostream& operator<< (ostream& os, const %s& EASEA_Var) { // Output stream insertion operator\n",sName); 
		pSymbolList->reset();
		while ((pSym=pSymbolList->walkToNextItem()))
		{
			if (pSym->Object->ObjectType==oObject)
				fprintf(fp,"    os <<  \"%s:\" << EASEA_Var.%s << \"\\n\";\n",pSym->Object->sName,pSym->Object->sName);
			
			if (pSym->Object->ObjectType==oArray )
			{
				fprintf(fp,"    {os << \"Array %s : \";\n",pSym->Object->sName);
				fprintf(fp,"     for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					pSym->Object->nSize/pSym->Object->pType->nSize);
				fprintf(fp,"       os << \"[\" << EASEA_Ndx << \"]:\" << EASEA_Var.%s[EASEA_Ndx] << \"\\t\";}\n    os << \"\\n\";\n",pSym->Object->sName);
			}
			
			if( pSym->Object->ObjectType==oArrayPointer)
			{
				fprintf(fp,"    {os << \"Array %s : \";\n",pSym->Object->sName);
				fprintf(fp,"     for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					(int)(pSym->Object->nSize/sizeof(char*)));
				fprintf(fp,"       if( EASEA_Var.%s[EASEA_Ndx] ) os << \"[\" << EASEA_Ndx << \"]:\" << *(EASEA_Var.%s[EASEA_Ndx]) << \"\\t\";}\n    os << \"\\n\";\n",pSym->Object->sName,pSym->Object->sName);
			}
			
			if (pSym->Object->ObjectType==oPointer)
				fprintf(fp,"    if (EASEA_Var.%s) os << \"%s:\" << *(EASEA_Var.%s) << \"\\n\";\n",
				pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
		}
		fprintf(fp,"    return os;\n  }\n\n"); // end of output stream insertion operator

		//     fprintf(fp,"  friend istream& operator>> (istream& is, %s& EASEA_Var) { // Input stream extraction operator\n",sName); // Output stream insertion operator
		//           pSymbolList->reset();
		//           while (pSym=pSymbolList->walkToNextItem()){
		//             if ((pSym->Object->ObjectType==oObject)&&(strcmp(pSym->Object->pType->sName, "bool"))) 
		//               fprintf(fp,"    is >> EASEA_Var.%s;\n",pSym->Object->sName);
		//             if ((pSym->Object->ObjectType==oArray)&&(strcmp(pSym->Object->pType->sName, "bool"))) {
		//               fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
		//               fprintf(fp,"       is >> EASEA_Var.%s[EASEA_Ndx];}\n",pSym->Object->sName);
		//             }                                         
		//           }
		//     fprintf(fp,"    return is;\n  }\n\n"); // Input stream extraction operator

		if (sString) 
		{
			if (bVERBOSE) printf ("Inserting Methods into %s Class.\n",sName);
			fprintf(fpOutputFile,"// User-defined methods:\n\n");
			fprintf(fpOutputFile,"%s\n",sString);
		}
	}

	fprintf(fp,"// Class members \n"); // Now, we must print the class members
	pSymbolList->reset();
	while ((pSym=pSymbolList->walkToNextItem()))
	{
		if (pSym->Object->ObjectQualifier==1) // 1=Static
			fprintf(fp,"  static");
		if (pSym->Object->ObjectType==oObject)
			fprintf(fp,"  %s %s;\n",pSym->Object->pType->sName,pSym->Object->sName);
		if (pSym->Object->ObjectType==oPointer)
			fprintf(fp,"  %s *%s;\n",pSym->Object->pType->sName,pSym->Object->sName);
		if (pSym->Object->ObjectType==oArray)
			fprintf(fp,"  %s %s[%d];\n",pSym->Object->pType->sName,pSym->Object->sName,pSym->Object->nSize/pSym->Object->pType->nSize);
		if (pSym->Object->ObjectType==oArrayPointer)
			fprintf(fp,"  %s* %s[%d];\n",
			pSym->Object->pType->sName,pSym->Object->sName,(int)(pSym->Object->nSize/sizeof(char*)));
	}

	if (strcmp(sName,"Genome"))
		fprintf(fp,"};\n");
	
	return;
}
//*/

void CSymbol::printClasses(FILE *fp)
{
	CListItem<CSymbol*> *pSym;
	if (bAlreadyPrinted) return;
	bAlreadyPrinted=true;
	pSymbolList->reset();
	while ((pSym=pSymbolList->walkToNextItem()))
		if ((pSym->Object->pType->ObjectType==oUserClass)&&(!pSym->Object->pType->bAlreadyPrinted))
			pSym->Object->pType->printClasses(fp);
	print(fp);
}

// prints user class definitions
void CSymbol::printUC(FILE* fp)
{
	//DEBUG_PRT("print user classes definitions");
	if (strcmp(sName,"Genome") && (TARGET==CUDA || TARGET==STD)) // If we are printing a user class other than the genome
	{   
		fprintf(fp,"\nclass %s;\n",sName); // class  header
	}
	//DEBUG_PRT("%s",sName);

	return;
}

// prints user classes
void CSymbol::printUserClasses(FILE *fp)
{
	CListItem<CSymbol*> *pSym;
	pSymbolList->reset();

	if (bAlreadyPrinted) return;

	bAlreadyPrinted=true;  
	while ((pSym=pSymbolList->walkToNextItem()))
	{
		if (pSym->Object->pType->ObjectType==oUserClass)
			pSym->Object->pType->printUC(fp);
	}

	return;
}

// This function fills the "serialize" part of the individual class produced by EASEA
// the "serialize" is used to code individuals in strings, in order to push them into UDP
// packets and send them to other instances of EASEA running on other network-connected machines
void CSymbol::serializeIndividual(FILE *fp, char* sCompleteName)
{
	CListItem<CSymbol*> *pSym;
	pSymbolList->reset();
	char sNewCompleteName[1000];
	strcpy(sNewCompleteName, sCompleteName);

	while((pSym=pSymbolList->walkToNextItem()))
	{
		// check the type of object 
		if(pSym->Object->pType->ObjectType==oUserClass)
		{
			// if it's an user-defined class
			if (pSym->Object->ObjectType==oArrayPointer)
			{
				// if it's an array of pointers
				fprintf(fp,"\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++){\n",(int)(pSym->Object->nSize/sizeof(char*)));
				fprintf(fpOutputFile,"\t\tif(this->%s[EASEA_Ndx] != NULL){\n",pSym->Object->sName);
				fprintf(fpOutputFile,"\t\t\tEASEA_Line << \"\\a \";\n");
				fprintf(fpOutputFile,"\t\t\tEASEA_Line << this->%s[EASEA_Ndx]->serializer() << \" \";\n",pSym->Object->sName);
				fprintf(fpOutputFile,"\t}\n");
				fprintf(fpOutputFile,"\t\telse\n");
				fprintf(fpOutputFile,"\t\t\tEASEA_Line << \"NULL\" << \" \";\n");
				fprintf(fpOutputFile,"}\n");

			}
			else
			{
				// if it's not an array of pointers
				fprintf(fpOutputFile,"\tif(this->%s != NULL){\n",pSym->Object->sName);
				fprintf(fpOutputFile,"\t\tEASEA_Line << \"\\a \";\n");
				fprintf(fpOutputFile,"\t\tEASEA_Line << this->%s->serializer() << \" \";\n",pSym->Object->sName);
				fprintf(fpOutputFile,"\t}\n");
				fprintf(fpOutputFile,"\telse\n");
				fprintf(fpOutputFile,"\t\tEASEA_Line << \"NULL\" << \" \";\n");
			}
		}
		else
		{
			// if it's not a user-defined class
			if (pSym->Object->ObjectType==oObject)
			{
				fprintf(fpOutputFile,"\tEASEA_Line << this->%s << \" \";\n",pSym->Object->sName);
			}
			else if(pSym->Object->ObjectType==oArray)
			{
				// if it's an array of floats	
				fprintf(fpOutputFile,"\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
				fprintf(fpOutputFile,"\t\tEASEA_Line << this->%s[EASEA_Ndx] <<\" \";\n", pSym->Object->sName);
			}
			else if(pSym->Object->ObjectType==oPointer && strcmp(pSym->Object->pType->sName, "GPNode") == 0)
			{
				// it's a pointer to a GPNode!
				fprintf(fpOutputFile,"\t// Serialize function for \"%s\"\n", pSym->Object->pType->sName);
					
				// serialize function: it needs the <map> and <vector> includes, but those are added
				// at the top of the class if it's a GPNode individual
				fprintf(fpOutputFile,"\tcout << \"Now serializing individual \" << toString(this->root) << endl;\n");
				fprintf(fpOutputFile,"\t// build map used to associate GPNode pointers to indexes\n");
				fprintf(fpOutputFile,"\tmap<GPNode*,int> indexes;\n");
				fprintf(fpOutputFile,"\n");
				fprintf(fpOutputFile,"\t// breadth-first visit of the tree\n");
				fprintf(fpOutputFile,"\tint currentIndex = 0;\n");
				fprintf(fpOutputFile,"\tlist<GPNode*> nodesToVisit;\n");
				fprintf(fpOutputFile,"\tnodesToVisit.push_back(this->root);\n");
				fprintf(fpOutputFile,"\n");
				fprintf(fpOutputFile,"\twhile(nodesToVisit.size() != 0)\n");
				fprintf(fpOutputFile,"\t{\n");
				fprintf(fpOutputFile,"\t // remove current node from nodes to visit\n");
				fprintf(fpOutputFile,"\t GPNode* currentNode = nodesToVisit.front();\n");
				fprintf(fpOutputFile,"\t nodesToVisit.pop_front();\n");
				fprintf(fpOutputFile,"\t\n");
				fprintf(fpOutputFile,"\t // add children of current node (by default, the tree is binary)\n");
				fprintf(fpOutputFile,"\t if( currentNode->children[0] != NULL ) nodesToVisit.push_back( currentNode->children[0] ); \n");
				fprintf(fpOutputFile,"\t if( currentNode->children[1] != NULL ) nodesToVisit.push_back( currentNode->children[1] ); \n");
				fprintf(fpOutputFile,"\t // assign code to current node       \n");
				fprintf(fpOutputFile,"\t indexes[currentNode] = currentIndex; \n");
				fprintf(fpOutputFile,"\t currentIndex++;\n");
				fprintf(fpOutputFile,"\t}\n");
				fprintf(fpOutputFile,"\t// the very first item in the line is the number of nodes in the tree \n");
				fprintf(fpOutputFile,"\tEASEA_Line << currentIndex << \" \";\n");
				fprintf(fpOutputFile,"\t// another visit to finally serialize the nodes \n");
				fprintf(fpOutputFile,"\tvector<double> ercValues;          \n");
				fprintf(fpOutputFile,"\tnodesToVisit.push_back(this->root);\n");
				fprintf(fpOutputFile,"\twhile(nodesToVisit.size() != 0)                     \n");
				fprintf(fpOutputFile,"\t{                                                   \n");
				fprintf(fpOutputFile,"\t // remove current node from nodes to visit  \n");
				fprintf(fpOutputFile,"\t GPNode* currentNode = nodesToVisit.front(); \n");
				fprintf(fpOutputFile,"\t nodesToVisit.pop_front();	\n");
				fprintf(fpOutputFile,"\t // add children of current node (hoping it's binary)                                                       \n");
				fprintf(fpOutputFile,"\t if( currentNode->children[0] != NULL ) nodesToVisit.push_back( currentNode->children[0] );                 \n");
				fprintf(fpOutputFile,"\t if( currentNode->children[1] != NULL ) nodesToVisit.push_back( currentNode->children[1] );                 \n");
				fprintf(fpOutputFile,"\t                                                                                                           \n");
				fprintf(fpOutputFile,"\t // node to string: format is <index> <var_id> <opCode> <indexOfChild1> <indexOfChild2>                     \n");
				fprintf(fpOutputFile,"\t EASEA_Line << indexes[currentNode] << \" \" << currentNode->var_id << \" \" << (int)currentNode->opCode << \" \";\n");
				fprintf(fpOutputFile,"\t // if the children are not NULL, put their index; otherwise, put \"0\"              \n");
				fprintf(fpOutputFile,"\t if( currentNode->children[0] != NULL )                                            \n");
				fprintf(fpOutputFile,"\t  EASEA_Line << indexes[ currentNode->children[0] ] << \" \";                 \n");
				fprintf(fpOutputFile,"\t else                                                                              \n");
				fprintf(fpOutputFile,"\t  EASEA_Line << \"0 \";                                                       \n");
				fprintf(fpOutputFile,"\t                                                                                  \n");
				fprintf(fpOutputFile,"\t if( currentNode->children[1] != NULL )                                            \n");
				fprintf(fpOutputFile,"\t  EASEA_Line << indexes[ currentNode->children[1] ] << \" \";                 \n");
				fprintf(fpOutputFile,"\t else             \n");
				fprintf(fpOutputFile,"\t  EASEA_Line << \"0 \";                                                       \n");
				fprintf(fpOutputFile,"\t                 \n");
				fprintf(fpOutputFile,"\t // if the node is an ERC, the floating point value is stored for later            \n");
				fprintf(fpOutputFile,"\t if( currentNode->opCode == OP_ERC ) ercValues.push_back( currentNode->erc_value );\n");
				fprintf(fpOutputFile,"\t} \n");
				fprintf(fpOutputFile,"\t// finally, put all the floating point ERC values             \n");
				fprintf(fpOutputFile,"\tfor(unsigned int i = 0; i < ercValues.size(); i++)            \n");
				fprintf(fpOutputFile,"\t EASEA_Line << ercValues[i] << \" \";                    \n");
				fprintf(fpOutputFile,"\t                                                              \n");
				fprintf(fpOutputFile,"\t// debug                                                      \n");
				fprintf(fpOutputFile,"\t//cout << \"EASEA_Line: \" << EASEA_Line.str() << endl; \n");
			}

		} // end if it's a user-defined class
	}// end while

	return;
}

void CSymbol::deserializeIndividual(FILE *fp, char* sCompleteName){
	CListItem<CSymbol*> *pSym;
	pSymbolList->reset();
	while ((pSym=pSymbolList->walkToNextItem())){
		if(pSym->Object->pType->ObjectType==oUserClass){
			if (pSym->Object->ObjectType==oArrayPointer){
				fprintf(fpOutputFile,"\tEASEA_Line >> line;\n");
				fprintf(fp,"\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++){\n",(int)(pSym->Object->nSize/sizeof(char*)));
				fprintf(fpOutputFile,"\t\tif(strcmp(line.c_str(),\"NULL\")==0)\n");
				fprintf(fpOutputFile,"\t\t\tthis->%s[EASEA_Ndx] = NULL;\n",pSym->Object->sName);
				fprintf(fpOutputFile,"\t\telse{\n");
				fprintf(fpOutputFile,"\t\t\tthis->%s[EASEA_Ndx] = new %s;\n",pSym->Object->sName, pSym->Object->pType->sName);
				fprintf(fpOutputFile,"\t\t\tthis->%s[EASEA_Ndx]->deserializer(&EASEA_Line);\n",pSym->Object->sName);
				fprintf(fpOutputFile,"\t\t}");
				fprintf(fpOutputFile,"\t}");
			}
			else{
				fprintf(fpOutputFile,"\tEASEA_Line >> line;\n");
				fprintf(fpOutputFile,"\tif(strcmp(line.c_str(),\"NULL\")==0)\n");
				fprintf(fpOutputFile,"\t\tthis->%s = NULL;\n",pSym->Object->sName);
				fprintf(fpOutputFile,"\telse{\n");
				fprintf(fpOutputFile,"\t\tthis->%s = new %s;\n",pSym->Object->sName, pSym->Object->pType->sName);
				fprintf(fpOutputFile,"\t\tthis->%s->deserializer(&EASEA_Line);\n",pSym->Object->sName);
				fprintf(fpOutputFile,"\t}");
			}
		}
		else
		{
			if (pSym->Object->ObjectType==oObject){
				fprintf(fpOutputFile,"\tEASEA_Line >> this->%s;\n",pSym->Object->sName);
			}

			if(pSym->Object->ObjectType==oArray){
				fprintf(fpOutputFile,"\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
				fprintf(fpOutputFile,"\t\tEASEA_Line >> this->%s[EASEA_Ndx];\n", pSym->Object->sName);
			}

			if(pSym->Object->ObjectType==oPointer && strcmp(pSym->Object->pType->sName, "GPNode") == 0)
			{
				// it's a GPNode, so a tree-like structure used for GP
				fprintf(fpOutputFile, "\t// debug\n");	
				fprintf(fpOutputFile, "\t//cout << \"Reading received individual...\" << endl;\n");	
				fprintf(fpOutputFile, "\t//cout << Line << endl;\n");	
				fprintf(fpOutputFile, "\t\n");	
				fprintf(fpOutputFile, "\t// first, read number of nodes\n");	
				fprintf(fpOutputFile, "\tint numberOfNodes; \n");	
				fprintf(fpOutputFile, "\tEASEA_Line >> numberOfNodes;\n");	
				fprintf(fpOutputFile, "\t// debug\n");	
				fprintf(fpOutputFile, "\t//cout << \"The received individual has \" << numberOfNodes << \" nodes.\" << endl; \n");	
				fprintf(fpOutputFile, "\t\n");	
				fprintf(fpOutputFile, "\t// iterate through the line, creating a map of <index> <GPNode*> <indexOfChild1> <indexOfChild2> \n");	
				fprintf(fpOutputFile, "\tmap< int, pair< GPNode*,vector<int> > > nodeMap;                              \n");	
				fprintf(fpOutputFile, "\tfor(int n = 0; n < numberOfNodes; n++)                                        \n");	
				fprintf(fpOutputFile, "\t{                                                                             \n");	
				fprintf(fpOutputFile, "\t int index, temp;                                                      \n");	
				fprintf(fpOutputFile, "\t int opCode;                                                           \n");	
				fprintf(fpOutputFile, "\t int var_id;                                                           \n");	
				fprintf(fpOutputFile, "\t vector<int> childrenIndexes;                                          \n");	
				fprintf(fpOutputFile, "\t  \n");	
				fprintf(fpOutputFile, "\t // format is <index> <var_id> <opCode> <indexOfChild1> <indexOfChild2>\n");	
				fprintf(fpOutputFile, "\t EASEA_Line >> index;             \n");	
				fprintf(fpOutputFile, "\t EASEA_Line >> var_id;            \n");	
				fprintf(fpOutputFile, "\t EASEA_Line >> opCode;            \n");	
				fprintf(fpOutputFile, "\t EASEA_Line >> temp;              \n");	
				fprintf(fpOutputFile, "\t childrenIndexes.push_back(temp); \n");	
				fprintf(fpOutputFile, "\t EASEA_Line >> temp;              \n");	
				fprintf(fpOutputFile, "\t childrenIndexes.push_back(temp); \n");	
				fprintf(fpOutputFile, "\t\n");	
				fprintf(fpOutputFile, "\t // create GPNode                   \n");	
				fprintf(fpOutputFile, "\t GPNode* currentNode = new GPNode();\n");	
				fprintf(fpOutputFile, "\t currentNode->var_id = var_id;\n");	
				fprintf(fpOutputFile, "\t currentNode->opCode = opCode;\n");	
				fprintf(fpOutputFile, "\t\n");	
				fprintf(fpOutputFile, "\t // debug \n");	
				fprintf(fpOutputFile, "\t //cout 	<< \"Read node: \" << index << \" \" << var_id << \" \" << opCode << \" \" \n");	
				fprintf(fpOutputFile, "\t //<< childrenIndexes[0] << \" \" << childrenIndexes[1] << endl;\n");	
				fprintf(fpOutputFile, "\t\n");	
				fprintf(fpOutputFile, "\t // put everything into the map                                        \n");	
				fprintf(fpOutputFile, "\t pair< GPNode*, vector<int> > tempPair (currentNode, childrenIndexes); \n");	
				fprintf(fpOutputFile, "\t nodeMap[index] = tempPair;\n"); 
				fprintf(fpOutputFile, "\t}\n");	
				fprintf(fpOutputFile, "\t \n");	
				fprintf(fpOutputFile, "\t// rebuild the individual structure \n");	
				fprintf(fpOutputFile, "\tfor(int n = 0; n < numberOfNodes; n++) \n");	
				fprintf(fpOutputFile, "\t{ \n");	
				fprintf(fpOutputFile, "\t // now, rebuild the individual by adding the pointers to the children                                  \n");	
				fprintf(fpOutputFile, "\t if( nodeMap[n].second[0] != 0 ) nodeMap[n].first->children[0] = nodeMap[ nodeMap[n].second[0] ].first; \n");	
				fprintf(fpOutputFile, "\t if( nodeMap[n].second[1] != 0 ) nodeMap[n].first->children[1] = nodeMap[ nodeMap[n].second[1] ].first; \n");	
				fprintf(fpOutputFile, "\t\n");	
				fprintf(fpOutputFile, "\t // also, if the opCode of the node is the same as the OP_ERC, find the\n");	
				fprintf(fpOutputFile, "\t // corresponding real value at the end of the EASEA_Line\n");	
				fprintf(fpOutputFile, "\t if( nodeMap[n].first->opCode == OP_ERC )\n");	
				fprintf(fpOutputFile, "\t {                                          \n");	
				fprintf(fpOutputFile, "\t                                           \n");	
				fprintf(fpOutputFile, "\t  double temp;                       \n");	
				fprintf(fpOutputFile, "\t  EASEA_Line >> temp;                \n");	
				fprintf(fpOutputFile, "\t  nodeMap[n].first->erc_value = temp;\n");	
				fprintf(fpOutputFile, "\t// debug\n");	
				fprintf(fpOutputFile, "\t//cout << \"-- Found ERC variable! Read value \" << temp << \" from the end of EASEA_Line.\" << endl;\n");	
				fprintf(fpOutputFile, "\t }\n");	
				fprintf(fpOutputFile, "\t}\n");	
				fprintf(fpOutputFile, "\t\n");	
				fprintf(fpOutputFile, "\t// link the tree to the current individual \n");	
				fprintf(fpOutputFile, "\tthis->root = nodeMap[0].first;\n");	
				fprintf(fpOutputFile, "\t\n");	
				fprintf(fpOutputFile, "\t// debug \n");	
				fprintf(fpOutputFile, "\t//cout << \"Individual received: \" << toString(this->root) << endl;\n");	
			}
		}
	}
}


void CSymbol::printAllSymbols(FILE *fp, char *sCompleteName, EObjectType FatherType, CListItem<CSymbol *> *pSym){
	char sNewCompleteName[000], s[20];
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
			if (pSym->Object->ObjectType==oArray) {
				strcat(sNewCompleteName,"[");
				sprintf(s,"%d",pSym->Object->nSize/pSym->Object->pType->nSize);
				strcat(sNewCompleteName,s);
				strcat(sNewCompleteName,"]");
			}
			fprintf(fp,"%s\n",sNewCompleteName);
			strcpy(sNewCompleteName, sCompleteName);
		}
	} while ((pSym=pSym->pNext));
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


OPCodeDesc::OPCodeDesc(){
	isERC = false;
	arity = 0;
}
int OPCodeDescCompare(const void* a, const void* b){
	OPCodeDesc** f1, ** f2;
	f1 = (OPCodeDesc**)a;
	f2 = (OPCodeDesc**)b;

	if( (*f1)->arity==(*f2)->arity && (*f1)->arity==0 ){
		return (*f1)->isERC-(*f2)->isERC;
	}
	else{
		return (*f1)->arity-(*f2)->arity;
	}
}

void OPCodeDesc::sort(OPCodeDesc** opDescs, unsigned len){
	qsort(opDescs,len,sizeof(OPCodeDesc*),OPCodeDescCompare);
}

void OPCodeDesc::show(void){

	cout << "OPCode : " << *this->opcode << endl;
	cout << "Real name : " << *this->realName << endl;
	cout << "Arity : " << this->arity << endl;
	cout << "cpu code : \n" << this->cpuCodeStream.str() << endl;
	cout << "gpu code : \n" << this->gpuCodeStream.str() << endl;

}
