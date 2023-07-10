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

extern int yylineno;

void debug(char* s)
{
#ifdef _DEBUG
	printf(s);
	getchar();
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

/////////////////////////////////////////////////////////////////////////////
// symbol construction/destruction

CSymbol::CSymbol(const char* s)
	: sName(s), dValue(0.0), sString(), nSize(0), bAlreadyPrinted(false), ObjectType(oUndefined),
	  ObjectQualifier(0), pType(nullptr), pClass(nullptr), pSymbolList()
{
	assert(s != NULL);
}

/////////////////////////////////////////////////////////////////////////////
// symbol  commands
void CSymbol::print(FILE* fp)
{
	// check: are we printing a user class that is different from Genome?
	if (sName != "Genome") {
		// if we are printing a user class other than the genome
		fprintf(fp, "\nclass %s {\npublic:\n// Default methods for class %s\n", sName.c_str(),
			sName.c_str()); // class  header

		//fprintf(fp,"// Class members \n"); // Now, we must print the class members
		//  pSymbolList->reset();
		//  while (pSym=pSymbolList->walkToNextItem()){
		//  if (pSym->Object->ObjectType==oObject)
		//  fprintf(fp,"  %s %s;\n",pSym->Object->pType->sName.c_str(),pSym->Object->sName.c_str());
		//  if (pSym->Object->ObjectType==oPointer)
		//  fprintf(fp,"  %s *%s;\n",pSym->Object->pType->sName.c_str(),pSym->Object->sName.c_str());
		//  if (pSym->Object->ObjectType==oArray)
		//  fprintf(fp,"  %s %s[%d];\n",pSym->Object->pType->sName.c_str(),pSym->Object->sName.c_str(),pSym->Object->nSize/pSym->Object->pType->nSize);
		// }

		// check on the type of target
		if (TARGET == CUDA) {
			// here we we are generating function to copy objects from host memory to gpu's.
			bool isFlatClass = true;

			for (auto const& sym : pSymbolList) {
				{
					//DEBUG_PRT("analyse flat %s",pSym->Object->pType->sName.c_str());
					if (sym->ObjectType ==
					    oPointer) //|| (pSym->Object->pType->ObjectType == oObject) )
					{
						isFlatClass = false;
						break;
					}
				}
			}

			//DEBUG_PRT("Does %s flat class : %s",sName,(isFlatClass?"yes":"no"));
			fprintf(fp, "  %s* cudaSendToGpu%s(){\n", sName.c_str(), sName.c_str());
			fprintf(fp, "    %s* ret=NULL;\n", sName.c_str());

			if (isFlatClass) {
				fprintf(fp, "    cudaMalloc((void**)&ret,sizeof(%s));\n", sName.c_str());
				fprintf(fp, "    cudaMemcpy(ret,this,sizeof(%s),cudaMemcpyHostToDevice);\n",
					sName.c_str());
				fprintf(fp, "    return ret;\n");
			} else {
				fprintf(fp, "    %s tmp;\n", sName.c_str());
				fprintf(fp, "    memcpy(&tmp,this,sizeof(%s));\n", sName.c_str());

				for (auto const& sym : pSymbolList) {
					if (sym->ObjectType ==
					    oPointer) //|| (pSym->Object->pType->ObjectType == oObject) )
					{
						fprintf(fp, "    tmp.%s=this->%s->cudaSendToGpu%s();\n",
							sym->sName.c_str(), sym->sName.c_str(), sym->pType->sName.c_str());
					}
				}
				fprintf(fp, "    cudaMalloc((void**)&ret,sizeof(%s));\n", sName.c_str());
				fprintf(fp, "    cudaMemcpy(ret,&tmp,sizeof(%s),cudaMemcpyHostToDevice);\n",
					sName.c_str());
				fprintf(fp, "    return ret;\n");
			}

			fprintf(fp, "  }\n\n");

			// another CUDA-specific function
			fprintf(fp, "  void cudaGetFromGpu%s(%s* dev_ptr){\n", sName.c_str(), sName.c_str());
			fprintf(fp, "    %s* ret=NULL;\n", sName.c_str());

			if (isFlatClass) {
				fprintf(fp, "    ret = (%s*)malloc(sizeof(%s));\n", sName.c_str(), sName.c_str());
				fprintf(fp, "    cudaMemcpy(ret,dev_ptr,sizeof(%s),cudaMemcpyDeviceToHost);\n",
					sName.c_str());
				//while (pSym=pSymbolList->walkToNextItem())
				//fprintf(fp,"    this->%s=ret->%s;\n",pSym->Object->sName.c_str(),pSym->Object->sName.c_str());
			}
			fprintf(fp, "  }\n\n");
		}

		// creation of class constructor
		fprintf(fp, "  %s(){  // Constructor\n", sName.c_str()); // constructor

		for (auto const& sym : pSymbolList) {
			if (sym->ObjectType == oPointer)
				fprintf(fp, "    %s=NULL;\n", sym->sName.c_str());

			if (sym->ObjectType == oArrayPointer) {
				fprintf(fp, "    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					(int)(sym->nSize / sizeof(char*)));
				fprintf(fp, "         %s[EASEA_Ndx]=NULL;\n", sym->sName.c_str());
			}
		}
		fprintf(fp, "  }\n"); // end of constructor

		// creation of copy constructor
		fprintf(fp, "  %s(const %s &EASEA_Var) {  // Copy constructor\n", sName.c_str(),
			sName.c_str()); // copy constructor

		
		for (auto const& sym : pSymbolList) {
			if (sym->ObjectType == oObject)
				fprintf(fp, "    %s=EASEA_Var.%s;\n", sym->sName.c_str(), sym->sName.c_str());

			if (sym->ObjectType == oArray) {
				fprintf(fp, "    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					sym->nSize / sym->pType->nSize);
				fprintf(fp, "       %s[EASEA_Ndx]=EASEA_Var.%s[EASEA_Ndx];}\n", sym->sName.c_str(),
					sym->sName.c_str());
			}

			if (sym->ObjectType == oPointer) {
				fprintf(fp, "    %s=(EASEA_Var.%s ? new %s(*(EASEA_Var.%s)) : NULL);\n", sym->sName.c_str(),
					sym->sName.c_str(), sym->pType->sName.c_str(), sym->sName.c_str());
			}

			if (sym->ObjectType == oArrayPointer) {
				fprintf(fp, "    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					(int)(sym->nSize / sizeof(char*)));
				fprintf(fp,
					"        if( EASEA_Var.%s[EASEA_Ndx] ) %s[EASEA_Ndx] = new %s(*(EASEA_Var.%s[EASEA_Ndx]));\n",
					sym->sName.c_str(), sym->sName.c_str(), sym->pType->sName.c_str(), sym->sName.c_str());
				fprintf(fp, "        else  %s[EASEA_Ndx] = NULL;\n", sym->sName.c_str());
			}
		}
		fprintf(fp, "  }\n"); // end of copy constructor

		// creation of destructor
		fprintf(fp, "  virtual ~%s() {  // Destructor\n", sName.c_str()); // destructor

		
		for (auto const& sym : pSymbolList) {
			if (sym->ObjectType == oPointer)
				fprintf(fp, "    if (%s) delete %s;\n    %s=NULL;\n", sym->sName.c_str(),
					sym->sName.c_str(), sym->sName.c_str());

			if (sym->ObjectType == oArrayPointer) {
				fprintf(fp, "    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					(int)(sym->nSize / sizeof(char*)));
				fprintf(fp, "        if( %s[EASEA_Ndx] ) delete %s[EASEA_Ndx];\n", sym->sName.c_str(),
					sym->sName.c_str());
			}
		}
		fprintf(fp, "  }\n"); // end of destructor

		// creation of serializer
		fprintf(fp, "  string serializer() {  // serialize\n"); // serializer
		fprintf(fp, "  \tostringstream EASEA_Line(ios_base::app);\n");
		
		for (auto const& sym : pSymbolList) {
			// check: is it a user-defined class?
			if (sym->pType->ObjectType == oUserClass) {
				if (sym->ObjectType == oArrayPointer) {
					// it's an array of pointers
					fprintf(fp, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++){\n",
						(int)(sym->nSize / sizeof(char*)));
					fprintf(fpOutputFile, "\t\tif(this->%s[EASEA_Ndx] != NULL){\n",
						sym->sName.c_str());
					fprintf(fpOutputFile, "\t\t\tEASEA_Line << \"\\a \";\n");
					fprintf(fpOutputFile,
						"\t\t\tEASEA_Line << this->%s[EASEA_Ndx]->serializer() << \" \";\n",
						sym->sName.c_str());
					fprintf(fpOutputFile, "\t}\n");
					fprintf(fpOutputFile, "\t\telse\n");
					fprintf(fpOutputFile, "\t\t\tEASEA_Line << \"NULL\" << \" \";\n");
					fprintf(fpOutputFile, "}\n");
				}
				// it's a classical array
				else if (sym->ObjectType == oArray) {
					fprintf(fpOutputFile, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						sym->nSize / sym->pType->nSize);
					fprintf(fpOutputFile,
						"\t\tEASEA_Line << this->%s[EASEA_Ndx].serializer() <<\" \";\n",
						sym->sName.c_str());
				}
				// it's a simple struct/class
				else if (sym->ObjectType == oObject) {
					fprintf(fpOutputFile, "\t\tEASEA_Line << this->%s.serializer() <<\" \";\n",
						sym->sName.c_str());
				} else {
					// it's a pointer to an user-defined clas
					fprintf(fpOutputFile, "\tif(this->%s != NULL){\n", sym->sName.c_str());
					fprintf(fpOutputFile, "\t\tEASEA_Line << \"\\a \";\n");
					fprintf(fpOutputFile, "\t\tEASEA_Line << this->%s->serializer() << \" \";\n",
						sym->sName.c_str());
					fprintf(fpOutputFile, "}\n");
					fprintf(fpOutputFile, "\telse\n");
					fprintf(fpOutputFile, "\t\tEASEA_Line << \"NULL\" << \" \";\n");
				}
			} else {
				// it's not a user-defined class
				if (sym->ObjectType == oObject) {
					fprintf(fpOutputFile, "\tEASEA_Line << this->%s << \" \";\n",
						sym->sName.c_str());
				}

				// it's a classical array
				if (sym->ObjectType == oArray) {
					fprintf(fpOutputFile, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						sym->nSize / sym->pType->nSize);
					fprintf(fpOutputFile, "\t\tEASEA_Line << this->%s[EASEA_Ndx] <<\" \";\n",
						sym->sName.c_str());
				}
			}
		} // end while

		fprintf(fp, "  \treturn EASEA_Line.str();\n");
		fprintf(fp, "  }\n"); // end of serializer

		// creation of deserializer
		fprintf(fp,
			"  void deserializer(istringstream* EASEA_Line) {  // deserialize\n"); // deserializer
		fprintf(fp, "  \tstring line;\n");
		
		for (auto const& sym : pSymbolList) {
			if (sym->pType->ObjectType == oUserClass) {
				/* it's an array of pointer */
				if (sym->ObjectType == oArrayPointer) {
					fprintf(fp, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++){\n",
						(int)(sym->nSize / sizeof(char*)));
					fprintf(fpOutputFile, "\t\t(*EASEA_Line) >> line;\n");
					fprintf(fpOutputFile, "\t\tif(strcmp(line.c_str(),\"NULL\")==0)\n");
					fprintf(fpOutputFile, "\t\t\tthis->%s[EASEA_Ndx] = NULL;\n", sym->sName.c_str());
					fprintf(fpOutputFile, "\t\telse{\n");
					fprintf(fpOutputFile, "\t\t\tthis->%s[EASEA_Ndx] = new %s;\n",
						sym->sName.c_str(), sym->pType->sName.c_str());
					fprintf(fpOutputFile, "\t\t\tthis->%s[EASEA_Ndx]->deserializer(EASEA_Line);\n",
						sym->sName.c_str());
					fprintf(fpOutputFile, "\t\t}");
					fprintf(fpOutputFile, "\t}");
				}
				/* it's a classical array*/
				else if (sym->ObjectType == oArray) {
					fprintf(fpOutputFile, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						sym->nSize / sym->pType->nSize);
					fprintf(fpOutputFile, "\t\tthis->%s[EASEA_Ndx].deserializer(EASEA_Line) ;\n",
						sym->sName.c_str());
				}
				/* it's a simple struct/class */
				else if (sym->ObjectType == oObject) {
					fprintf(fpOutputFile, "\t\tthis->%s.deserializer(EASEA_Line);\n",
						sym->sName.c_str());
				}
				/*it's a pointer*/
				else {
					fprintf(fpOutputFile, "\t(*EASEA_Line) >> line;\n");
					fprintf(fpOutputFile, "\tif(strcmp(line.c_str(),\"NULL\")==0)\n");
					fprintf(fpOutputFile, "\t\tthis->%s = NULL;\n", sym->sName.c_str());
					fprintf(fpOutputFile, "\telse{\n");
					fprintf(fpOutputFile, "\t\tthis->%s = new %s;\n", sym->sName.c_str(),
						sName.c_str());
					fprintf(fpOutputFile, "\t\tthis->%s->deserializer(EASEA_Line);\n",
						sym->sName.c_str());
					fprintf(fpOutputFile, "\t}");
				}
			} else {
				if (sym->ObjectType == oObject) {
					fprintf(fpOutputFile, "\t(*EASEA_Line) >> this->%s;\n", sym->sName.c_str());
				}

				if (sym->ObjectType == oArray) {
					fprintf(fpOutputFile, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						sym->nSize / sym->pType->nSize);
					fprintf(fpOutputFile, "\t\t(*EASEA_Line) >> this->%s[EASEA_Ndx];\n",
						sym->sName.c_str());
				}
			}
		} // end while
		fprintf(fp, "  }\n"); // end of deserializer

		// creation of operator to assess individual equality
		fprintf(fp, "  %s& operator=(const %s &EASEA_Var) {  // Operator=\n", sName.c_str(),
			sName.c_str()); // operator=
		fprintf(fp, "    if (&EASEA_Var == this) return *this;\n");
		
		for (auto const& sym : pSymbolList) {
			if (sym->ObjectType == oObject)
				fprintf(fp, "    %s = EASEA_Var.%s;\n", sym->sName.c_str(), sym->sName.c_str());

			if (sym->ObjectType == oArray) {
				fprintf(fp, "    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					sym->nSize / sym->pType->nSize);
				fprintf(fp, "       %s[EASEA_Ndx] = EASEA_Var.%s[EASEA_Ndx];}\n", sym->sName.c_str(),
					sym->sName.c_str());
			}

			if (sym->ObjectType == oArrayPointer) {
				fprintf(fp, "    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					(int)(sym->nSize / sizeof(char*)));
				fprintf(fp,
					"      if(EASEA_Var.%s[EASEA_Ndx]) %s[EASEA_Ndx] = new %s(*(EASEA_Var.%s[EASEA_Ndx]));\n",
					sym->sName.c_str(), sym->sName.c_str(), sym->pType->sName.c_str(), sym->sName.c_str());
			}

			if (sym->ObjectType == oPointer) {
				fprintf(fp, "    if (%s) delete %s;\n", sym->sName.c_str(), sym->sName.c_str());
				fprintf(fp, "    %s = (EASEA_Var.%s? new %s(*(EASEA_Var.%s)) : NULL);\n",
					sym->sName.c_str(), sym->sName.c_str(), sym->pType->sName.c_str(), sym->sName.c_str());
			}

		} // end while
		fprintf(fp, "  return *this;\n  }\n\n"); // end of operator <=

		// creation of operator ==
		fprintf(fp, "  bool operator==(%s &EASEA_Var) const {  // Operator==\n", sName.c_str()); // operator==
		
		for (auto const& sym : pSymbolList) {
			if (TARGET == CUDA || TARGET == STD) {
				if (sym->ObjectType == oObject)
					fprintf(fp, "    if (%s!=EASEA_Var.%s) return false;\n", sym->sName.c_str(),
						sym->sName.c_str());
				if (sym->ObjectType == oArray) {
					fprintf(fp, "    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						sym->nSize / sym->pType->nSize);
					fprintf(fp,
						"       if (%s[EASEA_Ndx]!=EASEA_Var.%s[EASEA_Ndx]) return false;}\n",
						sym->sName.c_str(), sym->sName.c_str());
				}
				if (sym->ObjectType == oArrayPointer) {
					fprintf(fp, "    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
						(int)(sym->nSize / sizeof(char*)));
					fprintf(fp,
						"       if (%s[EASEA_Ndx]!=EASEA_Var.%s[EASEA_Ndx]) return false;}\n",
						sym->sName.c_str(), sym->sName.c_str());
				}
				if (sym->ObjectType == oPointer) {
					fprintf(fp,
						"    if (((%s) && (!EASEA_Var.%s)) || ((!%s) && (EASEA_Var.%s))) return false;\n",
						sym->sName.c_str(), sym->sName.c_str(), sym->sName.c_str(),
						sym->sName.c_str());
					fprintf(fp, "    if ((%s)&&(%s!=EASEA_Var.%s)) return false;\n",
						sym->sName.c_str(), sym->sName.c_str(), sym->sName.c_str());
				}
			}
		}
		if (TARGET == CUDA || TARGET == STD)
			fprintf(fp, "  return true;\n  }\n\n"); // end of operator==

		// creation of operator !=
		fprintf(fp, "  bool operator!=(%s &EASEA_Var) const {return !(*this==EASEA_Var);} // operator!=\n\n",
			sName.c_str()); // operator!=

		// creation of output stream insertion operator
		fprintf(fp,
			"  friend ostream& operator<< (ostream& os, const %s& EASEA_Var) { // Output stream insertion operator\n",
			sName.c_str());
		
		for (auto const& sym : pSymbolList) {
			if (sym->ObjectType == oObject)
				fprintf(fp, "    os <<  \"%s:\" << EASEA_Var.%s << \"\\n\";\n", sym->sName.c_str(),
					sym->sName.c_str());

			if (sym->ObjectType == oArray) {
				fprintf(fp, "    {os << \"Array %s : \";\n", sym->sName.c_str());
				fprintf(fp, "     for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					sym->nSize / sym->pType->nSize);
				fprintf(fp,
					"       os << \"[\" << EASEA_Ndx << \"]:\" << EASEA_Var.%s[EASEA_Ndx] << \"\\t\";}\n    os << \"\\n\";\n",
					sym->sName.c_str());
			}

			if (sym->ObjectType == oArrayPointer) {
				fprintf(fp, "    {os << \"Array %s : \";\n", sym->sName.c_str());
				fprintf(fp, "     for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					(int)(sym->nSize / sizeof(char*)));
				fprintf(fp,
					"       if( EASEA_Var.%s[EASEA_Ndx] ) os << \"[\" << EASEA_Ndx << \"]:\" << *(EASEA_Var.%s[EASEA_Ndx]) << \"\\t\";}\n    os << \"\\n\";\n",
					sym->sName.c_str(), sym->sName.c_str());
			}

			if (sym->ObjectType == oPointer)
				fprintf(fp, "    if (EASEA_Var.%s) os << \"%s:\" << *(EASEA_Var.%s) << \"\\n\";\n",
					sym->sName.c_str(), sym->sName.c_str(), sym->sName.c_str());
		}
		fprintf(fp, "    return os;\n  }\n\n"); // end of output stream insertion operator

		//     fprintf(fp,"  friend istream& operator>> (istream& is, %s& EASEA_Var) { // Input stream extraction operator\n",sName); // Output stream insertion operator
		//           pSymbolList->reset();
		//           while (pSym=pSymbolList->walkToNextItem()){
		//             if ((sym->ObjectType==oObject)&&(strcmp(sym->pType->sName.c_str(), "bool")))
		//               fprintf(fp,"    is >> EASEA_Var.%s;\n",sym->sName.c_str());
		//             if ((sym->ObjectType==oArray)&&(strcmp(sym->pType->sName.c_str(), "bool"))) {
		//               fprintf(fp,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",sym->nSize/sym->pType->nSize);
		//               fprintf(fp,"       is >> EASEA_Var.%s[EASEA_Ndx];}\n",sym->sName.c_str());
		//             }
		//           }
		//     fprintf(fp,"    return is;\n  }\n\n"); // Input stream extraction operator

		if (sString != "") {
			if (bVERBOSE)
				printf("Inserting Methods into %s Class.\n", sName.c_str());
			fprintf(fpOutputFile, "// User-defined methods:\n\n");
			fprintf(fpOutputFile, "%s\n", sString.c_str());
		}
	}

	fprintf(fp, "// Class members \n"); // Now, we must print the class members
	
	for (auto const& sym : pSymbolList) {
		if (sym->ObjectQualifier == 1) // 1=Static
			fprintf(fp, "  static");
		if (sym->ObjectType == oObject)
			fprintf(fp, "  %s %s;\n", sym->pType->sName.c_str(), sym->sName.c_str());
		if (sym->ObjectType == oPointer)
			fprintf(fp, "  %s *%s;\n", sym->pType->sName.c_str(), sym->sName.c_str());
		if (sym->ObjectType == oArray)
			fprintf(fp, "  %s %s[%d];\n", sym->pType->sName.c_str(), sym->sName.c_str(),
				sym->nSize / sym->pType->nSize);
		if (sym->ObjectType == oArrayPointer)
			fprintf(fp, "  %s* %s[%d];\n", sym->pType->sName.c_str(), sym->sName.c_str(),
				(int)(sym->nSize / sizeof(char*)));
	}

	if (sName != "Genome")
		fprintf(fp, "};\n");

	return;
}
//*/

void CSymbol::printClasses(FILE* fp)
{
	if (bAlreadyPrinted)
		return;
	bAlreadyPrinted = true;
	
	for (auto const& sym : pSymbolList) {
		if ((sym->pType->ObjectType == oUserClass) && (!sym->pType->bAlreadyPrinted))
			sym->pType->printClasses(fp);
	}
	print(fp);
}

// prints user class definitions
void CSymbol::printUC(FILE* fp)
{
	//DEBUG_PRT("print user classes definitions");
	if (sName != "Genome" &&
	    (TARGET == CUDA || TARGET == STD)) // If we are printing a user class other than the genome
	{
		fprintf(fp, "\nclass %s;\n", sName.c_str()); // class  header
	}
	//DEBUG_PRT("%s",sName);

	return;
}

// prints user classes
void CSymbol::printUserClasses(FILE* fp)
{
	if (bAlreadyPrinted)
		return;

	bAlreadyPrinted = true;
	
	for (auto const& sym : pSymbolList) {
		if (sym->pType->ObjectType == oUserClass)
			sym->pType->printUC(fp);
	}

	return;
}

// This function fills the "serialize" part of the individual class produced by EASEA
// the "serialize" is used to code individuals in strings, in order to push them into UDP
// packets and send them to other instances of EASEA running on other network-connected machines
void CSymbol::serializeIndividual(FILE* fp, char* sCompleteName)
{
	std::string sNewCompleteName(sCompleteName);

	
	for (auto const& sym : pSymbolList) {
		// check the type of object
		if (sym->pType->ObjectType == oUserClass) {
			// if it's an user-defined class
			if (sym->ObjectType == oArrayPointer) {
				// if it's an array of pointers
				fprintf(fp, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++){\n",
					(int)(sym->nSize / sizeof(char*)));
				fprintf(fpOutputFile, "\t\tif(this->%s[EASEA_Ndx] != NULL){\n", sym->sName.c_str());
				fprintf(fpOutputFile, "\t\t\tEASEA_Line << \"\\a \";\n");
				fprintf(fpOutputFile,
					"\t\t\tEASEA_Line << this->%s[EASEA_Ndx]->serializer() << \" \";\n",
					sym->sName.c_str());
				fprintf(fpOutputFile, "\t}\n");
				fprintf(fpOutputFile, "\t\telse\n");
				fprintf(fpOutputFile, "\t\t\tEASEA_Line << \"NULL\" << \" \";\n");
				fprintf(fpOutputFile, "}\n");

			}
			// it's a classical array
			else if (sym->ObjectType == oArray) {
				/*TODO: not clean at all*/
				fprintf(fpOutputFile, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					sym->nSize / sym->pType->nSize);
				fprintf(fpOutputFile, "\t\tEASEA_Line << this->%s[EASEA_Ndx].serializer() <<\" \";\n",
					sym->sName.c_str());
			}
			// it's a simple struct/class
			else if (sym->ObjectType == oObject) {
				fprintf(fpOutputFile, "\tEASEA_Line << \"\\a \";\n");
				fprintf(fpOutputFile, "\tEASEA_Line << this->%s.serializer() << \" \";\n",
					sym->sName.c_str());
			} else {
				// if it's not an array of pointers
				fprintf(fpOutputFile, "\tif(this->%s != NULL){\n", sym->sName.c_str());
				fprintf(fpOutputFile, "\t\tEASEA_Line << \"\\a \";\n");
				fprintf(fpOutputFile, "\t\tEASEA_Line << this->%s->serializer() << \" \";\n",
					sym->sName.c_str());
				fprintf(fpOutputFile, "\t}\n");
				fprintf(fpOutputFile, "\telse\n");
				fprintf(fpOutputFile, "\t\tEASEA_Line << \"NULL\" << \" \";\n");
			}
		} else {
			// if it's not a user-defined class
			if (sym->ObjectType == oObject) {
				fprintf(fpOutputFile, "\tEASEA_Line << this->%s << \" \";\n", sym->sName.c_str());
			} else if (sym->ObjectType == oArray) {
				// if it's an array of floats
				fprintf(fpOutputFile, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					sym->nSize / sym->pType->nSize);
				fprintf(fpOutputFile, "\t\tEASEA_Line << this->%s[EASEA_Ndx] <<\" \";\n",
					sym->sName.c_str());
			} else if (sym->ObjectType == oPointer && strcmp(sym->pType->sName.c_str(), "GPNode") == 0) {
				// it's a pointer to a GPNode!
				fprintf(fpOutputFile, "\t// Serialize function for \"%s\"\n", sym->pType->sName.c_str());

				// serialize function: it needs the <map> and <vector> includes, but those are added
				// at the top of the class if it's a GPNode individual
				fprintf(fpOutputFile,
					"\tcout << \"Now serializing individual \" << toString(this->root) << endl;\n");
				fprintf(fpOutputFile, "\t// build map used to associate GPNode pointers to indexes\n");
				fprintf(fpOutputFile, "\tmap<GPNode*,int> indexes;\n");
				fprintf(fpOutputFile, "\n");
				fprintf(fpOutputFile, "\t// breadth-first visit of the tree\n");
				fprintf(fpOutputFile, "\tint currentIndex = 0;\n");
				fprintf(fpOutputFile, "\tlist<GPNode*> nodesToVisit;\n");
				fprintf(fpOutputFile, "\tnodesToVisit.push_back(this->root);\n");
				fprintf(fpOutputFile, "\n");
				fprintf(fpOutputFile, "\twhile(nodesToVisit.size() != 0)\n");
				fprintf(fpOutputFile, "\t{\n");
				fprintf(fpOutputFile, "\t // remove current node from nodes to visit\n");
				fprintf(fpOutputFile, "\t GPNode* currentNode = nodesToVisit.front();\n");
				fprintf(fpOutputFile, "\t nodesToVisit.pop_front();\n");
				fprintf(fpOutputFile, "\t\n");
				fprintf(fpOutputFile,
					"\t // add children of current node (by default, the tree is binary)\n");
				fprintf(fpOutputFile,
					"\t if( currentNode->children[0] != NULL ) nodesToVisit.push_back( currentNode->children[0] ); \n");
				fprintf(fpOutputFile,
					"\t if( currentNode->children[1] != NULL ) nodesToVisit.push_back( currentNode->children[1] ); \n");
				fprintf(fpOutputFile, "\t // assign code to current node       \n");
				fprintf(fpOutputFile, "\t indexes[currentNode] = currentIndex; \n");
				fprintf(fpOutputFile, "\t currentIndex++;\n");
				fprintf(fpOutputFile, "\t}\n");
				fprintf(fpOutputFile,
					"\t// the very first item in the line is the number of nodes in the tree \n");
				fprintf(fpOutputFile, "\tEASEA_Line << currentIndex << \" \";\n");
				fprintf(fpOutputFile, "\t// another visit to finally serialize the nodes \n");
				fprintf(fpOutputFile, "\tvector<double> ercValues;          \n");
				fprintf(fpOutputFile, "\tnodesToVisit.push_back(this->root);\n");
				fprintf(fpOutputFile, "\twhile(nodesToVisit.size() != 0)                     \n");
				fprintf(fpOutputFile, "\t{                                                   \n");
				fprintf(fpOutputFile, "\t // remove current node from nodes to visit  \n");
				fprintf(fpOutputFile, "\t GPNode* currentNode = nodesToVisit.front(); \n");
				fprintf(fpOutputFile, "\t nodesToVisit.pop_front();	\n");
				fprintf(fpOutputFile,
					"\t // add children of current node (hoping it's binary)                                                       \n");
				fprintf(fpOutputFile,
					"\t if( currentNode->children[0] != NULL ) nodesToVisit.push_back( currentNode->children[0] );                 \n");
				fprintf(fpOutputFile,
					"\t if( currentNode->children[1] != NULL ) nodesToVisit.push_back( currentNode->children[1] );                 \n");
				fprintf(fpOutputFile,
					"\t                                                                                                           \n");
				fprintf(fpOutputFile,
					"\t // node to string: format is <index> <var_id> <opCode> <indexOfChild1> <indexOfChild2>                     \n");
				fprintf(fpOutputFile,
					"\t EASEA_Line << indexes[currentNode] << \" \" << currentNode->var_id << \" \" << (int)currentNode->opCode << \" \";\n");
				fprintf(fpOutputFile,
					"\t // if the children are not NULL, put their index; otherwise, put \"0\"              \n");
				fprintf(fpOutputFile,
					"\t if( currentNode->children[0] != NULL )                                            \n");
				fprintf(fpOutputFile,
					"\t  EASEA_Line << indexes[ currentNode->children[0] ] << \" \";                 \n");
				fprintf(fpOutputFile,
					"\t else                                                                              \n");
				fprintf(fpOutputFile,
					"\t  EASEA_Line << \"0 \";                                                       \n");
				fprintf(fpOutputFile,
					"\t                                                                                  \n");
				fprintf(fpOutputFile,
					"\t if( currentNode->children[1] != NULL )                                            \n");
				fprintf(fpOutputFile,
					"\t  EASEA_Line << indexes[ currentNode->children[1] ] << \" \";                 \n");
				fprintf(fpOutputFile, "\t else             \n");
				fprintf(fpOutputFile,
					"\t  EASEA_Line << \"0 \";                                                       \n");
				fprintf(fpOutputFile, "\t                 \n");
				fprintf(fpOutputFile,
					"\t // if the node is an ERC, the floating point value is stored for later            \n");
				fprintf(fpOutputFile,
					"\t if( currentNode->opCode == OP_ERC ) ercValues.push_back( currentNode->erc_value );\n");
				fprintf(fpOutputFile, "\t} \n");
				fprintf(fpOutputFile,
					"\t// finally, put all the floating point ERC values             \n");
				fprintf(fpOutputFile,
					"\tfor(unsigned int i = 0; i < ercValues.size(); i++)            \n");
				fprintf(fpOutputFile, "\t EASEA_Line << ercValues[i] << \" \";                    \n");
				fprintf(fpOutputFile,
					"\t                                                              \n");
				fprintf(fpOutputFile,
					"\t// debug                                                      \n");
				fprintf(fpOutputFile, "\t//cout << \"EASEA_Line: \" << EASEA_Line.str() << endl; \n");
			}

		} // end if it's a user-defined class
	} // end while

	return;
}

void CSymbol::deserializeIndividual(FILE* fp, char* sCompleteName)
{
	
	for (auto const& sym : pSymbolList) {
		if (sym->pType->ObjectType == oUserClass) {
			if (sym->ObjectType == oArrayPointer) {
				fprintf(fpOutputFile, "\tEASEA_Line >> line;\n");
				fprintf(fp, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++){\n",
					(int)(sym->nSize / sizeof(char*)));
				fprintf(fpOutputFile, "\t\tif(strcmp(line.c_str(),\"NULL\")==0)\n");
				fprintf(fpOutputFile, "\t\t\tthis->%s[EASEA_Ndx] = NULL;\n", sym->sName.c_str());
				fprintf(fpOutputFile, "\t\telse{\n");
				fprintf(fpOutputFile, "\t\t\tthis->%s[EASEA_Ndx] = new %s;\n", sym->sName.c_str(),
					sym->pType->sName.c_str());
				fprintf(fpOutputFile, "\t\t\tthis->%s[EASEA_Ndx]->deserializer(&EASEA_Line);\n",
					sym->sName.c_str());
				fprintf(fpOutputFile, "\t\t}");
				fprintf(fpOutputFile, "\t}");
			} else if (sym->ObjectType == oArray) {
				fprintf(fpOutputFile, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					sym->nSize / sym->pType->nSize);
				fprintf(fpOutputFile, "\t\tthis->%s[EASEA_Ndx].deserializer(&EASEA_Line);\n",
					sym->sName.c_str());
			} else if (sym->ObjectType == oObject) {
				fprintf(fpOutputFile, "\t this->%s.deserializer(&EASEA_Line);", sym->sName.c_str());
			} else {
				fprintf(fpOutputFile, "\tEASEA_Line >> line;\n");
				fprintf(fpOutputFile, "\tif(strcmp(line.c_str(),\"NULL\")==0)\n");
				fprintf(fpOutputFile, "\t\tthis->%s = NULL;\n", sym->sName.c_str());
				fprintf(fpOutputFile, "\telse{\n");
				fprintf(fpOutputFile, "\t\tthis->%s = new %s;\n", sym->sName.c_str(), sym->pType->sName.c_str());
				fprintf(fpOutputFile, "\t\tthis->%s->deserializer(&EASEA_Line);\n", sym->sName.c_str());
				fprintf(fpOutputFile, "\t}");
			}
		} else {
			if (sym->ObjectType == oObject) {
				fprintf(fpOutputFile, "\tEASEA_Line >> this->%s;\n", sym->sName.c_str());
			}

			if (sym->ObjectType == oArray) {
				fprintf(fpOutputFile, "\tfor(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",
					sym->nSize / sym->pType->nSize);
				fprintf(fpOutputFile, "\t\tEASEA_Line >> this->%s[EASEA_Ndx];\n", sym->sName.c_str());
			}

			if (sym->ObjectType == oPointer && strcmp(sym->pType->sName.c_str(), "GPNode") == 0) {
				// it's a GPNode, so a tree-like structure used for GP
				fprintf(fpOutputFile, "\t// debug\n");
				fprintf(fpOutputFile, "\t//cout << \"Reading received individual...\" << endl;\n");
				fprintf(fpOutputFile, "\t//cout << Line << endl;\n");
				fprintf(fpOutputFile, "\t\n");
				fprintf(fpOutputFile, "\t// first, read number of nodes\n");
				fprintf(fpOutputFile, "\tint numberOfNodes; \n");
				fprintf(fpOutputFile, "\tEASEA_Line >> numberOfNodes;\n");
				fprintf(fpOutputFile, "\t// debug\n");
				fprintf(fpOutputFile,
					"\t//cout << \"The received individual has \" << numberOfNodes << \" nodes.\" << endl; \n");
				fprintf(fpOutputFile, "\t\n");
				fprintf(fpOutputFile,
					"\t// iterate through the line, creating a map of <index> <GPNode*> <indexOfChild1> <indexOfChild2> \n");
				fprintf(fpOutputFile,
					"\tmap< int, pair< GPNode*,vector<int> > > nodeMap;                              \n");
				fprintf(fpOutputFile,
					"\tfor(int n = 0; n < numberOfNodes; n++)                                        \n");
				fprintf(fpOutputFile,
					"\t{                                                                             \n");
				fprintf(fpOutputFile,
					"\t int index, temp;                                                      \n");
				fprintf(fpOutputFile,
					"\t int opCode;                                                           \n");
				fprintf(fpOutputFile,
					"\t int var_id;                                                           \n");
				fprintf(fpOutputFile,
					"\t vector<int> childrenIndexes;                                          \n");
				fprintf(fpOutputFile, "\t  \n");
				fprintf(fpOutputFile,
					"\t // format is <index> <var_id> <opCode> <indexOfChild1> <indexOfChild2>\n");
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
				fprintf(fpOutputFile,
					"\t //cout 	<< \"Read node: \" << index << \" \" << var_id << \" \" << opCode << \" \" \n");
				fprintf(fpOutputFile,
					"\t //<< childrenIndexes[0] << \" \" << childrenIndexes[1] << endl;\n");
				fprintf(fpOutputFile, "\t\n");
				fprintf(fpOutputFile,
					"\t // put everything into the map                                        \n");
				fprintf(fpOutputFile,
					"\t pair< GPNode*, vector<int> > tempPair (currentNode, childrenIndexes); \n");
				fprintf(fpOutputFile, "\t nodeMap[index] = tempPair;\n");
				fprintf(fpOutputFile, "\t}\n");
				fprintf(fpOutputFile, "\t \n");
				fprintf(fpOutputFile, "\t// rebuild the individual structure \n");
				fprintf(fpOutputFile, "\tfor(int n = 0; n < numberOfNodes; n++) \n");
				fprintf(fpOutputFile, "\t{ \n");
				fprintf(fpOutputFile,
					"\t // now, rebuild the individual by adding the pointers to the children                                  \n");
				fprintf(fpOutputFile,
					"\t if( nodeMap[n].second[0] != 0 ) nodeMap[n].first->children[0] = nodeMap[ nodeMap[n].second[0] ].first; \n");
				fprintf(fpOutputFile,
					"\t if( nodeMap[n].second[1] != 0 ) nodeMap[n].first->children[1] = nodeMap[ nodeMap[n].second[1] ].first; \n");
				fprintf(fpOutputFile, "\t\n");
				fprintf(fpOutputFile,
					"\t // also, if the opCode of the node is the same as the OP_ERC, find the\n");
				fprintf(fpOutputFile, "\t // corresponding real value at the end of the EASEA_Line\n");
				fprintf(fpOutputFile, "\t if( nodeMap[n].first->opCode == OP_ERC )\n");
				fprintf(fpOutputFile, "\t {                                          \n");
				fprintf(fpOutputFile, "\t                                           \n");
				fprintf(fpOutputFile, "\t  double temp;                       \n");
				fprintf(fpOutputFile, "\t  EASEA_Line >> temp;                \n");
				fprintf(fpOutputFile, "\t  nodeMap[n].first->erc_value = temp;\n");
				fprintf(fpOutputFile, "\t// debug\n");
				fprintf(fpOutputFile,
					"\t//cout << \"-- Found ERC variable! Read value \" << temp << \" from the end of EASEA_Line.\" << endl;\n");
				fprintf(fpOutputFile, "\t }\n");
				fprintf(fpOutputFile, "\t}\n");
				fprintf(fpOutputFile, "\t\n");
				fprintf(fpOutputFile, "\t// link the tree to the current individual \n");
				fprintf(fpOutputFile, "\tthis->root = nodeMap[0].first;\n");
				fprintf(fpOutputFile, "\t\n");
				fprintf(fpOutputFile, "\t// debug \n");
				fprintf(fpOutputFile,
					"\t//cout << \"Individual received: \" << toString(this->root) << endl;\n");
			}
		}
	}
}

template <typename Iterator>
void CSymbol::printAllSymbols(FILE* fp, char* sCompleteName, EObjectType FatherType, Iterator iSym,
			      Iterator end)
{
	std::string sNewCompleteName(sCompleteName);
	while (iSym != end) {
		if (iSym->pType->ObjectType == oUserClass) {
			if (FatherType == oPointer)
				sNewCompleteName += "->";
			else
				sNewCompleteName += ".";
			sNewCompleteName += iSym->sName.c_str();
			if (iSym->ObjectType == oArray) {
				sNewCompleteName += "[";
				sNewCompleteName += std::to_string(iSym->nSize / iSym->pType->nSize);
				sNewCompleteName += "]";
			}
			if (iSym->pType == iSym->pClass)
				fprintf(fp, "%s\n", sNewCompleteName);
			else
				printAllSymbols(fp, sNewCompleteName, iSym->ObjectType,
						iSym->pType->pSymbolList->getHead());
			sNewCompleteName = sCompleteName;
		} else {
			if (FatherType == oPointer)
				sNewCompleteName += "->";
			else
				sNewCompleteName += ".";
			sNewCompleteName += iSym->sName;
			if (iSym->ObjectType == oArray) {
				sNewCompleteName += "[";
				sNewCompleteName += std::to_string(iSym->nSize / iSym->pType->nSize);
				sNewCompleteName += "]";
			}
			fprintf(fp, "%s\n", sNewCompleteName);
			sNewCompleteName = sCompleteName;
		}
		iSym++;
	}
}

/////////////////////////////////////////////////////////////////////////////
// symbol table  commands

CSymbol* CSymbolTable::insert(std::unique_ptr<CSymbol>&& symbol)
{
	if (hashmap.find(symbol->sName) != hashmap.end() && symbol->sName.compare("CUSTOM_PRECISION_TYPE") != 0) {
		std::cerr << "\n" << sEZ_FILE_NAME << " - Warning line " << yylineno << ": Multiple definitions of symbol '" << symbol->sName << "', this may lead to compile errors!\n";
	}
	auto h = symbol->sName;
	hashmap.emplace(h, std::move(symbol));
	return hashmap.at(h).get();
}

CSymbol* CSymbolTable::find(const char* s)
{
	auto h = std::string(s);
	auto res = hashmap.find(h);
	if (res != hashmap.end()) {
		return res->second.get();
	} else {
		return nullptr;
	}
}

OPCodeDesc::OPCodeDesc()
{
	isERC = false;
	arity = 0;
}

void OPCodeDesc::show(void)
{
	cout << "OPCode : " << this->opcode << endl;
	cout << "Real name : " << this->realName << endl;
	cout << "Arity : " << this->arity << endl;
	cout << "cpu code : \n" << this->cpuCodeStream.str() << endl;
	cout << "gpu code : \n" << this->gpuCodeStream.str() << endl;
}
