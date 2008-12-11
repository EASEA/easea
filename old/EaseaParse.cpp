#include <cyacc.h>

#line 1 "C:\\repo\\src\\EaseaParse.y"

/****************************************************************************
EaseaLex.y
Parser for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@polytechnique.fr)
Ecole Polytechnique
Centre de Mathématiques Appliquées
91128 Palaiseau cedex
****************************************************************************/

#include "Easea.h"

// Globals     
CSymbol *pCURRENT_CLASS;
CSymbol *pCURRENT_TYPE;
CSymbol *pGENOME;
char sRAW_PROJECT_NAME[1000];
char sPROJECT_NAME[1000];
char sLOWER_CASE_PROJECT_NAME[1000];
char sEZ_FILE_NAME[1000];
char sEO_DIR[1000];
char sEZ_PATH[1000];
int TARGET;
int OPERATING_SYSTEM;
int nWARNINGS=0;
int nERRORS=0;
char sREPLACEMENT[50];
char sSELECTOR[50];
char sSELECT_PRM[50];
char sRED_PAR[50];
char sRED_PAR_PRM[50];
char sRED_OFF[50];
char sRED_OFF_PRM[50];
char sRED_FINAL[50];
char sRED_FINAL_PRM[50];
char sDISCARD[50];
char sDISCARD_PRM[50];
int nMINIMISE=2;
int nELITE;
bool bELITISM=0;
bool bVERBOSE=0;
int nPOP_SIZE, nOFF_SIZE, nSURV_PAR_SIZE, nSURV_OFF_SIZE;
int nNB_GEN;
int nNB_ISLANDS;
bool bPROP_SEQ;
float fMUT_PROB;
float fXOVER_PROB;
float fREPL_PERC=100;
FILE *fpOutputFile, *fpTemplateFile, *fpGenomeFile;//, *fpExplodedGenomeFile;
int nMIG_CLONE, nNB_MIG, nIMMIG_REPL;
char sMIG_SEL[50], sMIGRATOR[50], sIMMIG_SEL[50], sMIG_TARGET_SELECTOR[50];
float fMIG_FREQ;


#line 60 "C:\\repo\\src\\EaseaParse.cpp"
// repeated because of possible precompiled header
#include <cyacc.h>

#include "EaseaParse.h"

/////////////////////////////////////////////////////////////////////////////
// constructor

YYPARSENAME::YYPARSENAME()
{
	yytables();
#line 158 "C:\\repo\\src\\EaseaParse.y"

      CSymbol *pNewBaseType;

      if (TARGET!=DREAM) pNewBaseType=new CSymbol("bool");
      else pNewBaseType=new CSymbol("boolean");
      pNewBaseType->nSize=sizeof(bool);   
      pNewBaseType->ObjectType=oBaseClass;
      SymbolTable.insert(pNewBaseType);

      pNewBaseType=new CSymbol("int");
      pNewBaseType->nSize=sizeof(int);   
      pNewBaseType->ObjectType=oBaseClass;
      SymbolTable.insert(pNewBaseType);

      pNewBaseType=new CSymbol("double");
      pNewBaseType->nSize=sizeof(double);   
      pNewBaseType->ObjectType=oBaseClass;
      SymbolTable.insert(pNewBaseType);

      pNewBaseType=new CSymbol("float");
      pNewBaseType->nSize=sizeof(float);   
      pNewBaseType->ObjectType=oBaseClass;
      SymbolTable.insert(pNewBaseType);

      pNewBaseType=new CSymbol("char");
      pNewBaseType->nSize=sizeof(char);   
      pNewBaseType->ObjectType=oBaseClass;
      SymbolTable.insert(pNewBaseType);

      pNewBaseType=new CSymbol("pointer");
      pNewBaseType->nSize=sizeof(char *);   
      pNewBaseType->ObjectType=oBaseClass;
      SymbolTable.insert(pNewBaseType);

#line 107 "C:\\repo\\src\\EaseaParse.cpp"
}

#ifndef YYSTYPE
#define YYSTYPE int
#endif
#ifndef YYSTACK_SIZE
#define YYSTACK_SIZE 100
#endif

// yyattribute
#ifdef YYDEBUG
void YYFAR* YYPARSENAME::yyattribute1(int index) const
{
	YYSTYPE YYFAR* p = &((YYSTYPE YYFAR*)yyattributestackptr)[yytop + index];
	return p;
}
#define yyattribute(index) (*(YYSTYPE YYFAR*)yyattribute1(index))
#else
#define yyattribute(index) (((YYSTYPE YYFAR*)yyattributestackptr)[yytop + (index)])
#endif

#ifdef YYDEBUG
void YYPARSENAME::yyinitdebug(void YYFAR** p, int count) const
{
	yyassert(p != NULL);
	yyassert(count >= 1);

	YYSTYPE YYFAR** p1 = (YYSTYPE YYFAR**)p;
	for (int i = 0; i < count; i++) {
		p1[i] = &((YYSTYPE YYFAR*)yyattributestackptr)[yytop + i - (count - 1)];
	}
}
#endif

void YYPARSENAME::yyaction(int action)
{
	switch (action) {
	case 0:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 199 "C:\\repo\\src\\EaseaParse.y"

        if (bVERBOSE) printf("                    _______________________________________\n");
        if ((TARGET==DREAM)&& bVERBOSE) printf ("\nGeneration of the JAVA source files for %s.\n\n",sPROJECT_NAME);
        if ((TARGET!=DREAM)&& bVERBOSE) printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);
      
#line 158 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 1:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 205 "C:\\repo\\src\\EaseaParse.y"

        if (bVERBOSE) printf("                    _______________________________________\n");
        if ((TARGET==DREAM)&& bVERBOSE) printf ("\nGeneration of the JAVA source files for %s.\n\n",sPROJECT_NAME);
        if ((TARGET!=DREAM)&& bVERBOSE) printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);
      
#line 175 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 2:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 214 "C:\\repo\\src\\EaseaParse.y"

      if (bVERBOSE) printf("Declaration of user classes :\n\n");
#line 189 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 3:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 217 "C:\\repo\\src\\EaseaParse.y"

      if (bVERBOSE) printf("No user class declaration found other than GenomeClass.\n");
#line 203 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 4:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 227 "C:\\repo\\src\\EaseaParse.y"

      pCURRENT_CLASS=SymbolTable.insert(yyattribute(1 - 1).pSymbol);  
      pCURRENT_CLASS->pSymbolList=new CLList<CSymbol *>();
      yyattribute(1 - 1).pSymbol->ObjectType=oUserClass;
    
#line 220 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 5:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[6];
			yyinitdebug((void YYFAR**)yya, 6);
#endif
			{
#line 232 "C:\\repo\\src\\EaseaParse.y"

      if (bVERBOSE) printf("Class %s declared for %d bytes.\n\n",yyattribute(1 - 5).pSymbol->sName,yyattribute(1 - 5).pSymbol->nSize);
    
#line 235 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 6:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 244 "C:\\repo\\src\\EaseaParse.y"
pCURRENT_TYPE=yyattribute(2 - 2).pSymbol; pCURRENT_TYPE->ObjectQualifier=yyattribute(1 - 2).ObjectQualifier;
#line 248 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 7:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 244 "C:\\repo\\src\\EaseaParse.y"

#line 261 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 8:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 245 "C:\\repo\\src\\EaseaParse.y"
pCURRENT_TYPE=yyattribute(2 - 2).pSymbol; pCURRENT_TYPE->ObjectQualifier=yyattribute(1 - 2).ObjectQualifier;
#line 274 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 9:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 245 "C:\\repo\\src\\EaseaParse.y"

#line 287 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 10:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 250 "C:\\repo\\src\\EaseaParse.y"

    pCURRENT_CLASS->sString = new char[strlen(yyattribute(2 - 2).szString) + 1];
    strcpy(pCURRENT_CLASS->sString, yyattribute(2 - 2).szString);      
    if (bVERBOSE) printf("\n    The following methods have been declared:\n\n%s\n\n",pCURRENT_CLASS->sString);
    
#line 304 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 11:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 258 "C:\\repo\\src\\EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).ObjectQualifier=1;
#line 317 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 12:
		{
#line 259 "C:\\repo\\src\\EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).ObjectQualifier=0;
#line 325 "C:\\repo\\src\\EaseaParse.cpp"
		}
		break;
	case 13:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 264 "C:\\repo\\src\\EaseaParse.y"

#line 337 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 14:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 268 "C:\\repo\\src\\EaseaParse.y"

#line 350 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 15:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 269 "C:\\repo\\src\\EaseaParse.y"

#line 363 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 16:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 282 "C:\\repo\\src\\EaseaParse.y"
  
      CSymbol *pSym=SymbolTable.find(yyattribute(1 - 1).pSymbol->sName);
      if (pSym==NULL) {
        fprintf(stderr,"\n%s - Error line %d: Class \"%s\" was not defined.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 1).pSymbol->sName);
        fprintf(stderr,"Only base types (bool, int, float, double, char) or new user classes defined\nwithin the \"User classes\" sections are allowed.\n");
        exit(1);
      }       
      else (*(YYSTYPE YYFAR*)yyvalptr).pSymbol=pSym;
    
#line 384 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 17:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 303 "C:\\repo\\src\\EaseaParse.y"

//      CSymbol *pSym;
//      pSym=$1;
      if (TARGET==DREAM){                             
        if (pCURRENT_TYPE->ObjectType==oBaseClass){
          yyattribute(1 - 1).pSymbol->nSize=pCURRENT_TYPE->nSize;
          yyattribute(1 - 1).pSymbol->pClass=pCURRENT_CLASS;
          yyattribute(1 - 1).pSymbol->pType=pCURRENT_TYPE;
          yyattribute(1 - 1).pSymbol->ObjectType=oObject;
          yyattribute(1 - 1).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
          pCURRENT_CLASS->nSize+=yyattribute(1 - 1).pSymbol->nSize;
          pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(1 - 1).pSymbol));
          if (bVERBOSE) printf("    %s variable declared (%d bytes)\n",yyattribute(1 - 1).pSymbol->sName,yyattribute(1 - 1).pSymbol->nSize);
        }
        else {
          yyattribute(1 - 1).pSymbol->nSize=sizeof (char *);
          yyattribute(1 - 1).pSymbol->pClass=pCURRENT_CLASS;
          yyattribute(1 - 1).pSymbol->pType=pCURRENT_TYPE;
          yyattribute(1 - 1).pSymbol->ObjectType=oPointer;
          yyattribute(1 - 1).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
          pCURRENT_CLASS->nSize+=yyattribute(1 - 1).pSymbol->nSize;
          pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(1 - 1).pSymbol));
          if (bVERBOSE) printf("    %s pointer declared (%d bytes)\n",yyattribute(1 - 1).pSymbol->sName,yyattribute(1 - 1).pSymbol->nSize);
      } }
      else {
        yyattribute(1 - 1).pSymbol->nSize=pCURRENT_TYPE->nSize;
        yyattribute(1 - 1).pSymbol->pClass=pCURRENT_CLASS;
        yyattribute(1 - 1).pSymbol->pType=pCURRENT_TYPE;
        yyattribute(1 - 1).pSymbol->ObjectType=oObject;
        yyattribute(1 - 1).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
        pCURRENT_CLASS->nSize+=yyattribute(1 - 1).pSymbol->nSize;
        pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(1 - 1).pSymbol));
        if (bVERBOSE) printf("    %s variable declared (%d bytes)\n",yyattribute(1 - 1).pSymbol->sName,yyattribute(1 - 1).pSymbol->nSize);
    } 
#line 430 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 18:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 337 "C:\\repo\\src\\EaseaParse.y"

      yyattribute(2 - 2).pSymbol->nSize=sizeof (char *);
      yyattribute(2 - 2).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(2 - 2).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(2 - 2).pSymbol->ObjectType=oPointer;
      yyattribute(2 - 2).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(2 - 2).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(2 - 2).pSymbol));
      if (bVERBOSE) printf("    %s pointer declared (%d bytes)\n",yyattribute(2 - 2).pSymbol->sName,yyattribute(2 - 2).pSymbol->nSize);
    
#line 452 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 19:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 347 "C:\\repo\\src\\EaseaParse.y"

      yyattribute(1 - 4).pSymbol->nSize=pCURRENT_TYPE->nSize*(int)yyattribute(3 - 4).dValue;
      yyattribute(1 - 4).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(1 - 4).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(1 - 4).pSymbol->ObjectType=oArray;
      yyattribute(1 - 4).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(1 - 4).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(1 - 4).pSymbol));
      if (bVERBOSE) printf("    %s array declared (%d bytes)\n",yyattribute(1 - 4).pSymbol->sName,yyattribute(1 - 4).pSymbol->nSize);
    
#line 474 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 20:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 365 "C:\\repo\\src\\EaseaParse.y"

#line 487 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 21:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 378 "C:\\repo\\src\\EaseaParse.y"

      if (bVERBOSE) printf ("\nGenome declaration analysis :\n\n");
      pGENOME=new CSymbol("Genome");
      pCURRENT_CLASS=SymbolTable.insert(pGENOME);  
      pGENOME->pSymbolList=new CLList<CSymbol *>();
      pGENOME->ObjectType=oUserClass;
      pGENOME->ObjectQualifier=0;
      pGENOME->sString=NULL;
    
#line 508 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 22:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[6];
			yyinitdebug((void YYFAR**)yya, 6);
#endif
			{
#line 387 "C:\\repo\\src\\EaseaParse.y"

#line 521 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 23:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 404 "C:\\repo\\src\\EaseaParse.y"

#line 534 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 24:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 408 "C:\\repo\\src\\EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).pSymbol=yyattribute(1 - 1).pSymbol;
#line 547 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 25:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 417 "C:\\repo\\src\\EaseaParse.y"
         
      if (bVERBOSE) printf("Inserting genome initialiser (taken from .ez file).\n");
      switch (TARGET) {
        case GALIB : 
          fprintf(fpOutputFile,"void %sGenome::Initializer(GAGenome& g) {\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  %sGenome & genome = (%sGenome &)g;\n",sPROJECT_NAME,sPROJECT_NAME);
          break;
        case EO :
          fprintf(fpOutputFile,"%sGenome InitialiserFunction() {\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  %sGenome *genome = new %sGenome;\n",sPROJECT_NAME,sPROJECT_NAME);
    } 
#line 570 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 26:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 428 "C:\\repo\\src\\EaseaParse.y"
                                    
      switch (TARGET) {
        case GALIB : fprintf(fpOutputFile,"genome._evaluated=gaFalse;\n}\n\n");break;
        case EO : fprintf(fpOutputFile,"return *genome;\n}");
    } 
#line 587 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 27:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 433 "C:\\repo\\src\\EaseaParse.y"

      if (bVERBOSE) printf("Inserting user genome crossover (taken from .ez file).\n");
      switch (TARGET) {
        case GALIB : 
          fprintf(fpOutputFile,"int %sGenome::Crossover(const GAGenome& EZ_a, const GAGenome& EZ_b, GAGenome* pEZ_c, GAGenome* pEZ_d) {\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  %sGenome *pMom = (%sGenome *) &EZ_a;\n",sPROJECT_NAME,sPROJECT_NAME);
          fprintf(fpOutputFile,"  %sGenome *pDad = (%sGenome *) &EZ_b;\n",sPROJECT_NAME,sPROJECT_NAME);
          fprintf(fpOutputFile,"  %sGenome *pSis = (%sGenome *) pEZ_c;\n",sPROJECT_NAME,sPROJECT_NAME);
          fprintf(fpOutputFile,"  %sGenome *pBro = (%sGenome *) pEZ_d;\n",sPROJECT_NAME,sPROJECT_NAME);
          fprintf(fpOutputFile,"  if (pBro) pBro->_evaluated=gaFalse;\n  if (pSis) pSis->_evaluated=gaFalse;\n");
          fprintf(fpOutputFile,"  int EZ_GeneratedChildren=2;\n");
          fprintf(fpOutputFile,"  if (pBro) *pBro=*pDad;\n");
          fprintf(fpOutputFile,"  if (pSis) *pSis=*pMom;\n");
          fprintf(fpOutputFile,"  if (!pBro) {pBro = new %sGenome(*pMom);pBro->_evaluated=pSis->_evaluated=gaFalse;EZ_GeneratedChildren=3;}\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  if (!pSis) {pSis = new %sGenome(*pDad);pBro->_evaluated=pSis->_evaluated=gaFalse;EZ_GeneratedChildren=4;}\n",sPROJECT_NAME);
          break;
    } 
#line 616 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 28:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 450 "C:\\repo\\src\\EaseaParse.y"

      switch (TARGET) {
        case GALIB : 
          fprintf(fpOutputFile,"  if(EZ_GeneratedChildren==3) {delete pBro; EZ_GeneratedChildren=1;}\n");
          fprintf(fpOutputFile,"  if(EZ_GeneratedChildren==4) {delete pSis; EZ_GeneratedChildren=1;}\n");
          fprintf(fpOutputFile,"return EZ_GeneratedChildren;\n}\n\n");break;
    } 
#line 635 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 29:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 457 "C:\\repo\\src\\EaseaParse.y"

      if (bVERBOSE) printf("Inserting user genome mutator (taken from .ez file).\n");
      switch (TARGET) {
        case GALIB : 
          fprintf(fpOutputFile,"int %sGenome::Mutator(GAGenome& g, float PMut) {\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  if (!GAFlipCoin((float)EZ_MUT_PROB)) return 0;\n");
          fprintf(fpOutputFile,"  %sGenome & genome = (%sGenome &)g;\n",sPROJECT_NAME,sPROJECT_NAME);
          fprintf(fpOutputFile,"  genome._evaluated=gaFalse;\n");
          break;
    } 
#line 657 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 30:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 467 "C:\\repo\\src\\EaseaParse.y"
   
      if (TARGET==DREAM) fprintf(fpOutputFile,"  }\n");
      fprintf(fpOutputFile,"}");
    
#line 673 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 31:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 471 "C:\\repo\\src\\EaseaParse.y"
 
      if (bVERBOSE) printf("Inserting user genome evaluator (taken from .ez file).\n");
      switch (TARGET) {
        case GALIB : 
          fprintf(fpOutputFile,"float %sGenome::Evaluator(GAGenome & g) {\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  clock_t EZ_t1;\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  %sGenome & genome = (%sGenome &)g;\n",sPROJECT_NAME,sPROJECT_NAME);
          fprintf(fpOutputFile,"  EZ_NB_EVALUATIONS++;\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  const GAPopulation *pPopulation;\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  if (g.geneticAlgorithm()!=NULL) pPopulation=&(g.geneticAlgorithm()->population());  // to circumvent a bug in GALib\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  EZ_t1=clock();\n",sPROJECT_NAME);
          break;
        case EO : 
          fprintf(fpOutputFile,"  EZ_NB_EVALUATIONS++;\n",sPROJECT_NAME);
    } 
#line 700 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 32:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 486 "C:\\repo\\src\\EaseaParse.y"

          if (TARGET!=EO) fprintf(fpOutputFile,"}\n");
    
#line 715 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 33:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 498 "C:\\repo\\src\\EaseaParse.y"
nNB_GEN=(int)yyattribute(2 - 2).dValue;
#line 728 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 34:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 500 "C:\\repo\\src\\EaseaParse.y"
nNB_ISLANDS=(int)yyattribute(2 - 2).dValue;
#line 741 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 35:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 501 "C:\\repo\\src\\EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"proportionally")) bPROP_SEQ=true;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"sequentially")) bPROP_SEQ=false;
      else {
         fprintf(stderr,"\n%s - Error line %d: Looking for \"Proportionally\" or \"Sequentially\".\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
     } 
#line 760 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 36:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 509 "C:\\repo\\src\\EaseaParse.y"
fMUT_PROB=(float)yyattribute(2 - 2).dValue;
#line 773 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 37:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 511 "C:\\repo\\src\\EaseaParse.y"
fXOVER_PROB=(float)yyattribute(2 - 2).dValue;
#line 786 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 38:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 513 "C:\\repo\\src\\EaseaParse.y"
nPOP_SIZE=(int)yyattribute(2 - 2).dValue;
#line 799 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 39:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 514 "C:\\repo\\src\\EaseaParse.y"

      strcpy(sSELECTOR, yyattribute(2 - 2).pSymbol->sName);
      switch (TARGET) {
        case DREAM : if (!mystricmp(sSELECTOR,"EPTrn")){
            fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
            exit(1);
          }
          else if (!mystricmp(sSELECTOR,"Tournament")) sprintf(sSELECTOR,"TournamentSelector(MAX, 1, 2)");
          else if (!mystricmp(sSELECTOR,"StochTrn")) {
            fprintf(stderr,"\n%s - Warning line %d: The Stochatic Tournament selector needs a parameter in [0.5,1].\nDefault value 1 inserted",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
            sprintf(sSELECTOR,"TournamentSelector(MAX, 1, 2)");
          }
          else if (!mystricmp(sSELECTOR,"RouletteWheel")) sprintf(sSELECTOR,"RouletteWheelSelector(MAX)");
          else if (!mystricmp(sSELECTOR,"Random")) sprintf(sSELECTOR,"RandomSelector()");
          else if (!mystricmp(sSELECTOR,"Sequential")) sprintf(sSELECTOR,"BestNSelector(MAX,%d)",nPOP_SIZE);
          else if (!mystricmp(sSELECTOR,"Ranking")) sprintf(sSELECTOR,"RankingSelector(MAX)");
          else {
            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sSELECTOR);
            exit(1);
          }
          break;
        case GALIB : if (!mystricmp(sSELECTOR,"Ranking")){
                                  fprintf(stderr,"\n%s - Error line %d: The Ranking selector is not implemented in GALib.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                  exit(1);
                                }
                                else if (!mystricmp(sSELECTOR,"Tournament")) sprintf(sSELECTOR,"Tournament");
                                else if (!mystricmp(sSELECTOR,"StochTrn")) sprintf(sSELECTOR,"TournamentSelector(true, 1, 2)");
                                else if (!mystricmp(sSELECTOR,"RouletteWheel")) sprintf(sSELECTOR,"RouletteWheel");
                                else if (!mystricmp(sSELECTOR,"Random")) sprintf(sSELECTOR,"Uniform");
                                else if (!mystricmp(sSELECTOR,"Sequential")) sprintf(sSELECTOR,"Rank");
                                else {
                                  fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sSELECTOR);
                                  exit(1);
                                }
                                break;
        case EO : if (!mystricmp(sSELECTOR,"RouletteWheel")){
                            if (nMINIMISE==1) {
                              fprintf(stderr,"\n%s - Error line %d: The RouletteWheel selection scheme cannot be\n selected when \"minimising the fitness\" is the evaluator goal.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                              exit(1);
                            }
                            else sprintf(sSELECTOR,"Roulette");
                          }
                          else if (!mystricmp(sSELECTOR,"Tournament")) sprintf(sSELECTOR,"DetTour");
                          else if (!mystricmp(sSELECTOR,"StochTrn")) sprintf(sSELECTOR,"StochTour");
                          else if (!mystricmp(sSELECTOR,"Random")) sprintf(sSELECTOR,"Random");
                          else if (!mystricmp(sSELECTOR,"Ranking")) sprintf(sSELECTOR,"Ranking");
                          else if (!mystricmp(sSELECTOR,"Sequential")) sprintf(sSELECTOR,"Sequential");
                          else {
                            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist in EO.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sSELECTOR);
                            exit(1);
                          }
      }
      
#line 864 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 40:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 567 "C:\\repo\\src\\EaseaParse.y"

      sprintf(sSELECTOR, yyattribute(2 - 3).pSymbol->sName);   
      switch (TARGET) {
        case DREAM : if (!mystricmp(sSELECTOR,"EPTrn")){
                                  fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                  exit(1);
                                }
                                else if (!mystricmp(sSELECTOR,"Tournament")) {
                                  if (yyattribute(3 - 3).dValue>=2) sprintf(sSELECTOR,"TournamentSelector(MAX, 1, %d)",(int)yyattribute(3 - 3).dValue);
                                  else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sSELECTOR,"TournamentSelector(MAX, %f, 2)",(float)yyattribute(3 - 3).dValue);
                                  else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                }
                                else if (!mystricmp(sSELECTOR,"StochTrn")) {
                                  if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sSELECTOR,"TournamentSelector(MAX, %f, 2)",(float)yyattribute(3 - 3).dValue);
                                  else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                }
                                else if (!mystricmp(sSELECTOR,"RouletteWheel")) sprintf(sSELECTOR,"RouletteWheelSelector(MAX)");
                                else if (!mystricmp(sSELECTOR,"Random")) sprintf(sSELECTOR,"RandomSelector()");
                                else if (!mystricmp(sSELECTOR,"Sequential")) sprintf(sSELECTOR,"BestNSelector(MAX,%d)",nPOP_SIZE);
                                else if (!mystricmp(sSELECTOR,"Ranking")) sprintf(sSELECTOR,"RankingSelector(MAX)");
                                else {
                                  fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sSELECTOR);
                                  exit(1);
                                }
                                break;
        case GALIB : fprintf(stderr,"\n%s - Warning line %d: No GALib selector takes parameters yet. The parameter will be ignored.\n",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                                if (!mystricmp(sSELECTOR,"Ranking")) sprintf(sSELECTOR,"Rank");
                                else if (!mystricmp(sSELECTOR,"Tournament")) sprintf(sSELECTOR,"Tournament");
                                else if (!mystricmp(sSELECTOR,"StochTrn")) sprintf(sSELECTOR,"TournamentSelector(true, 1, 2)");
                                else if (!mystricmp(sSELECTOR,"RouletteWheel")) sprintf(sSELECTOR,"RouletteWheel");
                                else if (!mystricmp(sSELECTOR,"Random")) sprintf(sSELECTOR,"Uniform");
                                else if (!mystricmp(sSELECTOR,"Sequential")){
                                          fprintf(stderr,"\n%s - Error line %d: The Sequential selector does not exist under GALib.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                          exit(1);
                                        }
                                else {
                                  fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sSELECTOR);
                                  exit(1);
                                }
                                break;
        case EO :  if (!mystricmp(sSELECTOR,"Tournament")||!mystricmp(sSELECTOR,"StochTrn")) {
                             if (yyattribute(3 - 3).dValue>=2) {sprintf(sSELECTOR,"DetTour");sprintf(sSELECT_PRM,"(%d)",(int) yyattribute(3 - 3).dValue);}
                             else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) {sprintf(sSELECTOR,"StochTour");sprintf(sSELECT_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sSELECTOR,"RouletteWheel")) {
                            sprintf(sSELECTOR,"Roulette");
                            if (yyattribute(3 - 3).dValue<1) {fprintf(stderr,"\n%s - Warning line %d: The parameter of RouletteWheel must be greater than one.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;}
                            else sprintf(sSELECT_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);
                          }
                          else if (!mystricmp(sSELECTOR,"Random")) {
                            sprintf(sSELECTOR,"Random");
                            fprintf(stderr,"\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in EO.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
                          else if (!mystricmp(sSELECTOR,"Ranking")) {
                            sprintf(sSELECTOR,"Ranking");
                            if ((yyattribute(3 - 3).dValue<=1)||(yyattribute(3 - 3).dValue>2)) {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sSELECT_PRM,"(2)");
                            }
                            else sprintf(sSELECT_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);
                          }
                          else if (!mystricmp(sSELECTOR,"Sequential")) {
                            sprintf(sSELECTOR,"Sequential");
                            if (yyattribute(3 - 3).dValue==0) sprintf(sSELECT_PRM,"(unordered)");
                            else if (yyattribute(3 - 3).dValue==1) sprintf(sSELECT_PRM,"(ordered)");
                            else {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Sequential must be either 0 (unordered) or 1 (ordered).\nThe parameter will default to 1.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sSELECT_PRM,"(ordered)");
                            }
                          }
                          else {
                            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sSELECTOR);
                            exit(1);
                          }
     }
#line 952 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 41:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 643 "C:\\repo\\src\\EaseaParse.y"

        sprintf(sRED_PAR, yyattribute(2 - 2).pSymbol->sName);
        switch (TARGET) {
          case DREAM :
            if (!mystricmp(sRED_PAR,"EPTrn")){
              fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
              exit(1);
            }
            else if (!mystricmp(sRED_PAR,"Tournament")) sprintf(sRED_PAR,"TournamentSelector(MAX, 1, 2)");
            else if (!mystricmp(sRED_PAR,"StochTrn")) {
              fprintf(stderr,"\n%s - Warning line %d: The Stochatic Tournament selector needs a parameter in [0.5,1].\nDefault value 1 inserted",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
              sprintf(sRED_PAR,"TournamentSelector(MAX, 1, 2)");
            }
            else if (!mystricmp(sRED_PAR,"RouletteWheel")) sprintf(sRED_PAR,"RouletteWheelSelector(MAX)");
            else if (!mystricmp(sRED_PAR,"Random")) sprintf(sRED_PAR,"RandomSelector()");
            else if (!mystricmp(sRED_PAR,"Sequential")) sprintf(sRED_PAR,"BestNSelector(MAX,%d)",nPOP_SIZE);
            else if (!mystricmp(sRED_PAR,"Ranking")) sprintf(sRED_PAR,"RankingSelector(MAX)");
            else {
              fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_PAR);
              exit(1);
            }
          break;
          case EO : if (!mystricmp(sRED_PAR,"RouletteWheel")){
                              if (nMINIMISE==1) {
                                fprintf(stderr,"\n%s - Error line %d: The RouletteWheel selection scheme cannot be\n selected when \"minimising the fitness\" is the evaluator goal.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                exit(1);
                              }
                              else sprintf(sRED_PAR,"Roulette");
                            }
                            else if (!mystricmp(sRED_PAR,"Tournament")) sprintf(sRED_PAR,"DetTour");
                            else if (!mystricmp(sRED_PAR,"StochTrn")) sprintf(sRED_PAR,"StochTour");
                            else if (!mystricmp(sRED_PAR,"Random")) sprintf(sRED_PAR,"Random");
                            else if (!mystricmp(sRED_PAR,"Ranking")) sprintf(sRED_PAR,"Ranking");
                            else if (!mystricmp(sRED_PAR,"Sequential")) sprintf(sRED_PAR,"Sequential");
                            else {
                              fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist in EO.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_PAR);
                              exit(1);
                            }
       }
       
#line 1004 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 42:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 683 "C:\\repo\\src\\EaseaParse.y"

        sprintf(sRED_PAR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
          case DREAM : if (!mystricmp(sRED_PAR,"EPTrn")){
                                        fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                        exit(1);
                                      }
                                      else if (!mystricmp(sRED_PAR,"Tournament")) {
                                        if (yyattribute(3 - 3).dValue>=2) sprintf(sRED_PAR,"TournamentSelector(MAX, 1, %d)",(int)yyattribute(3 - 3).dValue);
                                        else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sRED_PAR,"TournamentSelector(MAX, %f, 2)",(float)yyattribute(3 - 3).dValue);
                                        else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                      }
                                      else if (!mystricmp(sRED_PAR,"StochTrn")) {
                                        if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sRED_PAR,"TournamentSelector(MAX, %f, 2)",(float)yyattribute(3 - 3).dValue);
                                        else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                      }
                                      else if (!mystricmp(sRED_PAR,"RouletteWheel")) sprintf(sRED_PAR,"RouletteWheelSelector(MAX)");
                                      else if (!mystricmp(sRED_PAR,"Random")) sprintf(sRED_PAR,"RandomSelector()");
                                      else if (!mystricmp(sRED_PAR,"Sequential")) sprintf(sRED_PAR,"BestNSelector(MAX,%d)",nPOP_SIZE);
                                      else if (!mystricmp(sRED_PAR,"Ranking")) sprintf(sRED_PAR,"RankingSelector(MAX)");
                                      else {
                                        fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_PAR);
                                        exit(1);
                                      }
          break;
        case EO :  if (!mystricmp(sRED_PAR,"Tournament")||!mystricmp(sRED_PAR,"StochTrn")) {
                             if (yyattribute(3 - 3).dValue>=2) {sprintf(sRED_PAR,"DetTour");sprintf(sRED_PAR_PRM,"(%d)",(int) yyattribute(3 - 3).dValue);}
                             else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) {sprintf(sRED_PAR,"StochTour");sprintf(sRED_PAR_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sRED_PAR,"RouletteWheel")) {
                            sprintf(sRED_PAR,"Roulette");
                            if (yyattribute(3 - 3).dValue<1) {fprintf(stderr,"\n%s - Warning line %d: The parameter of RouletteWheel must be greater than one.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;}
                            else sprintf(sRED_PAR_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);
                          }
                          else if (!mystricmp(sRED_PAR,"Random")) {
                            sprintf(sRED_PAR,"Random");
                            fprintf(stderr,"\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in EO.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
                          else if (!mystricmp(sRED_PAR,"Ranking")) {
                            sprintf(sRED_PAR,"Ranking");
                            if ((yyattribute(3 - 3).dValue<=1)||(yyattribute(3 - 3).dValue>2)) {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_PAR_PRM,"(2)");
                            }
                            else sprintf(sRED_PAR_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);
                          }
                          else if (!mystricmp(sRED_PAR,"Sequential")) {
                            sprintf(sRED_PAR,"Sequential");
                            if (yyattribute(3 - 3).dValue==0) sprintf(sRED_PAR_PRM,"(unordered)");
                            else if (yyattribute(3 - 3).dValue==1) sprintf(sRED_PAR_PRM,"(ordered)");
                            else {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Sequential must be either 0 (unordered) or 1 (ordered).\nThe parameter will default to 1.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_PAR_PRM,"(ordered)");
                            }
                          }
                          else {
                            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_PAR);
                            exit(1);
                          }
       }
#line 1077 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 43:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 744 "C:\\repo\\src\\EaseaParse.y"

      sprintf(sRED_OFF, yyattribute(2 - 2).pSymbol->sName);
      switch (TARGET) {
          case DREAM : if (!mystricmp(sRED_OFF,"EPTrn")){
                                        fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                        exit(1);
                                      }
                                      else if (!mystricmp(sRED_OFF,"Tournament")) sprintf(sRED_OFF,"TournamentSelector(MAX, 1, 2)");
                                      else if (!mystricmp(sRED_OFF,"StochTrn")) {
                                        fprintf(stderr,"\n%s - Warning line %d: The Stochatic Tournament selector needs a parameter in [0.5,1].\nDefault value 1 inserted",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                                        sprintf(sRED_OFF,"TournamentSelector(MAX, 1, 2)");
                                      }
                                      else if (!mystricmp(sRED_OFF,"RouletteWheel")) sprintf(sRED_OFF,"RouletteWheelSelector(MAX)");
                                      else if (!mystricmp(sRED_OFF,"Random")) sprintf(sRED_OFF,"RandomSelector()");
                                      else if (!mystricmp(sRED_OFF,"Sequential")) sprintf(sRED_OFF,"BestNSelector(MAX,%d)",nPOP_SIZE);
                                      else if (!mystricmp(sRED_OFF,"Ranking")) sprintf(sRED_OFF,"RankingSelector(MAX)");
                                      else {
                                        fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_OFF);
                                        exit(1);
                                      }
            break;
          case EO : if (!mystricmp(sRED_OFF,"RouletteWheel")){
                              if (nMINIMISE==1) {
                                fprintf(stderr,"\n%s - Error line %d: The RouletteWheel selection scheme cannot be\n selected when \"minimising the fitness\" is the evaluator goal.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                exit(1);
                              }
                              else sprintf(sRED_OFF,"Roulette");
                            }
                            else if (!mystricmp(sRED_OFF,"Tournament")) sprintf(sRED_OFF,"DetTour");
                            else if (!mystricmp(sRED_OFF,"StochTrn")) sprintf(sRED_OFF,"StochTour");
                            else if (!mystricmp(sRED_OFF,"Random")) sprintf(sRED_OFF,"Random");
                            else if (!mystricmp(sRED_OFF,"Ranking")) sprintf(sRED_OFF,"Ranking");
                            else if (!mystricmp(sRED_OFF,"Sequential")) sprintf(sRED_OFF,"Sequential");
                            else {
                              fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist in EO.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_OFF);
                              exit(1);
                            }
       }
#line 1127 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 44:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 782 "C:\\repo\\src\\EaseaParse.y"

        strcpy(sRED_OFF, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
          case DREAM : if (!mystricmp(sRED_OFF,"EPTrn")){
                                        fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                        exit(1);
                                      }
                                      else if (!mystricmp(sRED_OFF,"Tournament")) {
                                        if (yyattribute(3 - 3).dValue>=2) sprintf(sRED_OFF,"TournamentSelector(MAX, 1, %d)",(int)yyattribute(3 - 3).dValue);
                                        else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sRED_OFF,"TournamentSelector(MAX, %f, 2)",(float)yyattribute(3 - 3).dValue);
                                        else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                      }
                                      else if (!mystricmp(sRED_OFF,"StochTrn")) {
                                        if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sRED_OFF,"TournamentSelector(MAX, %f, 2)",(float)yyattribute(3 - 3).dValue);
                                        else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                      }
                                      else if (!mystricmp(sRED_OFF,"RouletteWheel")) sprintf(sRED_OFF,"RouletteWheelSelector(MAX)");
                                      else if (!mystricmp(sRED_OFF,"Random")) sprintf(sRED_OFF,"RandomSelector()");
                                      else if (!mystricmp(sRED_OFF,"Sequential")) sprintf(sRED_OFF,"BestNSelector(MAX,%d)",nPOP_SIZE);
                                      else if (!mystricmp(sRED_OFF,"Ranking")) sprintf(sRED_OFF,"RankingSelector(MAX)");
                                      else {
                                        fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_OFF);
                                        exit(1);
                                      }
          break;
        case EO :  if (!mystricmp(sRED_OFF,"Tournament")||!mystricmp(sRED_OFF,"StochTrn")) {
                             if (yyattribute(3 - 3).dValue>=2) {sprintf(sRED_OFF,"DetTour");sprintf(sRED_OFF_PRM,"(%d)",(int) yyattribute(3 - 3).dValue);}
                             else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) {sprintf(sRED_OFF,"StochTour");sprintf(sRED_OFF_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sRED_OFF,"RouletteWheel")) {
                            sprintf(sRED_OFF,"Roulette");
                            if (yyattribute(3 - 3).dValue<1) {fprintf(stderr,"\n%s - Warning line %d: The parameter of RouletteWheel must be greater than one.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;}
                            else sprintf(sRED_OFF_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);
                          }
                          else if (!mystricmp(sRED_OFF,"Random")) {
                            sprintf(sRED_OFF,"Random");
                            fprintf(stderr,"\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in EO.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
                          else if (!mystricmp(sRED_OFF,"Ranking")) {
                            sprintf(sRED_OFF,"Ranking");
                            if ((yyattribute(3 - 3).dValue<=1)||(yyattribute(3 - 3).dValue>2)) {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_OFF_PRM,"(2)");
                            }
                            else sprintf(sRED_OFF_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);
                          }
                          else if (!mystricmp(sRED_OFF,"Sequential")) {
                            sprintf(sRED_OFF,"Sequential");
                            if (yyattribute(3 - 3).dValue==0) sprintf(sRED_OFF_PRM,"(unordered)");
                            else if (yyattribute(3 - 3).dValue==1) sprintf(sRED_OFF_PRM,"(ordered)");
                            else {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Sequential must be either 0 (unordered) or 1 (ordered).\nThe parameter will default to 1.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_OFF_PRM,"(ordered)");
                            }
                          }
                          else {
                            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_OFF);
                            exit(1);
                          }
       }
#line 1200 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 45:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 843 "C:\\repo\\src\\EaseaParse.y"

        strcpy(sRED_FINAL, yyattribute(2 - 2).pSymbol->sName);
        switch (TARGET) {
          case DREAM : if (!mystricmp(sRED_FINAL,"EPTrn")){
                                        fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                        exit(1);
                                      }
                                      else if (!mystricmp(sRED_FINAL,"Tournament")) sprintf(sRED_FINAL,"TournamentSelector(false, 1, 2)");
                                      else if (!mystricmp(sRED_FINAL,"StochTrn")) {
                                        fprintf(stderr,"\n%s - Warning line %d: The Stochatic Tournament selector needs a parameter in [0.5,1].\nDefault value 1 inserted",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                                        sprintf(sRED_FINAL,"TournamentSelector(false, 1, 2)");
                                      }
                                      else if (!mystricmp(sRED_FINAL,"RouletteWheel")) sprintf(sRED_FINAL,"RouletteWheelSelector(%s)",nMINIMISE?"true":"false");
                                      else if (!mystricmp(sRED_FINAL,"Random")) sprintf(sRED_FINAL,"RandomSelector()");
                                      else if (!mystricmp(sRED_FINAL,"Sequential")) sprintf(sRED_FINAL,"BestNSelector(%s,%d)",nMINIMISE?"true":"false",nPOP_SIZE);
                                      else if (!mystricmp(sRED_FINAL,"Ranking")) sprintf(sRED_FINAL,"RankingSelector(%s)",nMINIMISE?"true":"false");
                                      else {
                                        fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_FINAL);
                                        exit(1);
                                      }
            break;
          case EO : if (!mystricmp(sRED_FINAL,"RouletteWheel")){
                              if (nMINIMISE==1) {
                                fprintf(stderr,"\n%s - Error line %d: The RouletteWheel selection scheme cannot be\n selected when \"minimising the fitness\" is the evaluator goal.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                exit(1);
                              }
                              else sprintf(sRED_FINAL,"Roulette");
                            }
                            else if (!mystricmp(sRED_FINAL,"Tournament")) sprintf(sRED_FINAL,"DetTour");
                            else if (!mystricmp(sRED_FINAL,"StochTrn")) sprintf(sRED_FINAL,"StochTour");
                            else if (!mystricmp(sRED_FINAL,"Random")) sprintf(sRED_FINAL,"Random");
                            else if (!mystricmp(sRED_FINAL,"Ranking")) sprintf(sRED_FINAL,"Ranking");
                            else if (!mystricmp(sRED_FINAL,"Sequential")) sprintf(sRED_FINAL,"Sequential");
                            else {
                              fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist in EO.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_FINAL);
                              exit(1);
                            }
       }
#line 1250 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 46:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 881 "C:\\repo\\src\\EaseaParse.y"

        strcpy(sRED_FINAL, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
          case DREAM : if (!mystricmp(sRED_FINAL,"EPTrn")){
                                        fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                        exit(1);
                                      }
                                      else if (!mystricmp(sRED_FINAL,"Tournament")) {
                                        if (yyattribute(3 - 3).dValue>=2) sprintf(sRED_FINAL,"TournamentSelector(%s, 1, %d)",(nMINIMISE?"true":"false"),(int)yyattribute(3 - 3).dValue);
                                        else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sRED_FINAL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"true":"false"),(float)yyattribute(3 - 3).dValue);
                                        else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                      }
                                      else if (!mystricmp(sRED_FINAL,"StochTrn")) {
                                        if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sRED_FINAL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"true":"false"),(float)yyattribute(3 - 3).dValue);
                                        else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                      }
                                      else if (!mystricmp(sRED_FINAL,"RouletteWheel")) sprintf(sRED_FINAL,"RouletteWheelSelector(%s)",nMINIMISE?"true":"false");
                                      else if (!mystricmp(sRED_FINAL,"Random")) sprintf(sRED_FINAL,"RandomSelector()");
                                      else if (!mystricmp(sRED_FINAL,"Sequential")) sprintf(sRED_FINAL,"BestNSelector(%s,%d)",nMINIMISE?"true":"false",nPOP_SIZE);
                                      else if (!mystricmp(sRED_FINAL,"Ranking")) sprintf(sRED_FINAL,"RankingSelector(%s)",nMINIMISE?"true":"false");
                                      else {
                                        fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_OFF);
                                        exit(1);
                                      }
          break;
        case EO :  if (!mystricmp(sRED_FINAL,"Tournament")||!mystricmp(sRED_FINAL,"StochTrn")) {
                             if (yyattribute(3 - 3).dValue>=2) {sprintf(sRED_FINAL,"DetTour");sprintf(sRED_FINAL_PRM,"(%d)",(int) yyattribute(3 - 3).dValue);}
                             else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) {sprintf(sRED_FINAL,"StochTour");sprintf(sRED_FINAL_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sRED_FINAL,"RouletteWheel")) {
                            sprintf(sRED_FINAL,"Roulette");
                            if (yyattribute(3 - 3).dValue<1) {fprintf(stderr,"\n%s - Warning line %d: The parameter of RouletteWheel must be greater than one.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;}
                            else sprintf(sRED_FINAL_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);
                          }
                          else if (!mystricmp(sRED_FINAL,"Random")) {
                            sprintf(sRED_FINAL,"Random");
                            fprintf(stderr,"\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in EO.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
                          else if (!mystricmp(sRED_FINAL,"Ranking")) {
                            sprintf(sRED_FINAL,"Ranking");
                            if ((yyattribute(3 - 3).dValue<=1)||(yyattribute(3 - 3).dValue>2)) {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_FINAL_PRM,"(2)");
                            }
                            else sprintf(sRED_FINAL_PRM,"(%f)",(float) yyattribute(3 - 3).dValue);
                          }
                          else if (!mystricmp(sRED_FINAL,"Sequential")) {
                            sprintf(sRED_FINAL,"Sequential");
                            if (yyattribute(3 - 3).dValue==0) sprintf(sRED_FINAL_PRM,"(unordered)");
                            else if (yyattribute(3 - 3).dValue==1) sprintf(sRED_FINAL_PRM,"(ordered)");
                            else {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Sequential must be either 0 (unordered) or 1 (ordered).\nThe parameter will default to 1.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_FINAL_PRM,"(ordered)");
                            }
                          }
                          else {
                            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_FINAL);
                            exit(1);
                          }
       }
#line 1323 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 47:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 942 "C:\\repo\\src\\EaseaParse.y"
nOFF_SIZE=(int)yyattribute(2 - 2).dValue;
#line 1336 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 48:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 943 "C:\\repo\\src\\EaseaParse.y"
nOFF_SIZE=(int)(yyattribute(2 - 3).dValue*nPOP_SIZE/100);
#line 1349 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 49:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 944 "C:\\repo\\src\\EaseaParse.y"
nSURV_PAR_SIZE=(int)yyattribute(2 - 2).dValue;
#line 1362 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 50:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 945 "C:\\repo\\src\\EaseaParse.y"
nSURV_PAR_SIZE=(int)(yyattribute(2 - 3).dValue*nPOP_SIZE/100);
#line 1375 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 51:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 946 "C:\\repo\\src\\EaseaParse.y"
nSURV_OFF_SIZE=(int)yyattribute(2 - 2).dValue;
#line 1388 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 52:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 947 "C:\\repo\\src\\EaseaParse.y"
nSURV_OFF_SIZE=(int)(yyattribute(2 - 3).dValue*nPOP_SIZE/100);
#line 1401 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 53:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 948 "C:\\repo\\src\\EaseaParse.y"
                                       
  // Generational
      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Generational")){
        if ((nOFF_SIZE+fREPL_PERC) && ((nOFF_SIZE!=nPOP_SIZE)||(fREPL_PERC!=100))){
          fprintf(stderr,"\n%s - Warning line %d: The \"Generational\" replacement strategy\nreplaces the whole population.\n",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
          fprintf(stderr,"COMPILATION WILL THEREFORE CONTINUE WITH AN OFFSPRING POPULATION SIZE OF %d !\n",nPOP_SIZE);
          if (TARGET==GALIB) fREPL_PERC=0; else fREPL_PERC=100;
        }
        switch (TARGET) {
          case GALIB : strcpy(sREPLACEMENT, "Simple"); break;
          case EO : strcpy(sREPLACEMENT, "Generational");
      }}
// Steadystate
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"SteadyState"))
        switch (TARGET) {
          case GALIB : strcpy(sREPLACEMENT, "Incremental");
                                  fREPL_PERC=0;
                                  break;
          case EO : strcpy(sREPLACEMENT, "SSGA");
        }
// Plus
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"ESPlus"))
        switch (TARGET) {
          case GALIB : strcpy(sREPLACEMENT, "SteadyState"); break;
          case EO : strcpy(sREPLACEMENT, "Plus");
        }
// Comma
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"ESComma"))
        switch (TARGET) {
          case GALIB : fprintf(stderr,"\n%s - Error line %d: The Comma replacement strategie is not yet available in EASEA-GALib.\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);
          case EO : strcpy(sREPLACEMENT, "Comma");
        }  
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Custom"))
        switch (TARGET) {
          case GALIB : fprintf(stderr,"\n%s - Error line %d: The Custom replacement strategie is not yet available in EASEA-GALib.\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);
          case EO : strcpy(sREPLACEMENT, "Custom");
        }  
      else {
         fprintf(stderr,"\n%s - Error line %d: The %s replacement strategy does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, yyattribute(2 - 2).pSymbol->sName);
         exit(1);
     }
#line 1454 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 54:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 989 "C:\\repo\\src\\EaseaParse.y"

      strcpy(sDISCARD, yyattribute(2 - 2).pSymbol->sName);
      switch (TARGET) {
        case GALIB : if (!mystricmp(sDISCARD,"Worst")) {
                                  strcpy(sDISCARD,"WORST"); break;
                                }
                                else if (!mystricmp(sDISCARD,"Parent")) {
                                  if (mystricmp(sREPLACEMENT,"Incremental")){  
                                    fprintf(stderr,"\n%s - Warning line %d: GALib only takes \"%s\" with a steady-state replacement strategy. Default operator \"Worst\" is selected.\n",sEZ_FILE_NAME,EASEALexer.yylineno,sDISCARD);nWARNINGS++;
                                    strcpy(sDISCARD,"WORST"); break;
                                  }
                                  strcpy(sDISCARD,"PARENT");
                                  break;
                                }
                                else if (!mystricmp(sDISCARD,"Random")) {
                                  if (mystricmp(sREPLACEMENT,"Incremental")){  
                                    fprintf(stderr,"\n%s - Warning line %d: GALib only takes \"%s\" with a steady-state replacement strategy. Default operator \"Worst\" is selected.\n",sEZ_FILE_NAME,EASEALexer.yylineno,sDISCARD);nWARNINGS++;
                                    strcpy(sDISCARD,"WORST"); break;
                                  }
                                  strcpy(sDISCARD,"RANDOM"); break;
                                }
                                else if (!mystricmp(sDISCARD,"Best")) {
                                  if (mystricmp(sREPLACEMENT,"Incremental")){  
                                    fprintf(stderr,"\n%s - Warning line %d: GALib only takes \"%s\" with a steady-state replacement strategy. Default operator \"Worst\" is selected.\n",sEZ_FILE_NAME,EASEALexer.yylineno,sDISCARD);nWARNINGS++;
                                    strcpy(sDISCARD,"WORST"); break;
                                  }
                                  strcpy(sDISCARD,"BEST"); break;
                                }
                                else {
                                  fprintf(stderr,"\n%s - Warning line %d: GALib does not take \"%s\" as a discarding\noperator. Default operator \"Worst\" is selected.\n",sEZ_FILE_NAME,EASEALexer.yylineno,sDISCARD);nWARNINGS++;
                                  strcpy(sDISCARD,"WORST"); break;
                                }
          break;
        case EO : if (!mystricmp(sREPLACEMENT,"SSGA")){
                           if (!mystricmp(sDISCARD,"Tournament")) {
                              strcat(sREPLACEMENT,"DetTournament");
                              strcpy(sDISCARD_PRM,"(2)");
                            }
                           else if (!mystricmp(sDISCARD,"Worst")) {
                              strcat(sREPLACEMENT,"Worse");
                            }
                           else {
                              fprintf(stderr,"\n%s - Warning line %d: EO does not take \"%s\" as a discarding\noperator. Default operator \"Worst\" is selected.\n",sEZ_FILE_NAME,EASEALexer.yylineno,sDISCARD);nWARNINGS++;
                              strcat(sREPLACEMENT,"Worse"); break;
                            }
       }}
#line 1512 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 55:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[6];
			yyinitdebug((void YYFAR**)yya, 6);
#endif
			{
#line 1035 "C:\\repo\\src\\EaseaParse.y"

      strcpy(sDISCARD, yyattribute(2 - 5).pSymbol->sName);
      switch (TARGET) {
        case GALIB : fprintf(stderr,"\n%s - Warning line %d: No GALib selector takes parameters yet. The parameter will be ignored.\n",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                                if (!mystricmp(sDISCARD,"Worst")) {
                                  strcpy(sDISCARD,"WORST"); break;
                                }
                                else if (!mystricmp(sDISCARD,"Parent")) {
                                  if (mystricmp(sREPLACEMENT,"Incremental")){  
                                    fprintf(stderr,"\n%s - Warning line %d: GALib only takes \"%s\" with a steady-state replacement strategy. Default operator \"Worst\" is selected.\n",sEZ_FILE_NAME,EASEALexer.yylineno,sDISCARD);nWARNINGS++;
                                    strcpy(sDISCARD,"WORST"); break;
                                  }
                                  strcpy(sDISCARD,"PARENT");
                                  break;
                                }
                                else if (!mystricmp(sDISCARD,"Random")) {
                                  if (mystricmp(sREPLACEMENT,"Incremental")){  
                                    fprintf(stderr,"\n%s - Warning line %d: GALib only takes \"%s\" with a steady-state replacement strategy. Default operator \"Worst\" is selected.\n",sEZ_FILE_NAME,EASEALexer.yylineno,sDISCARD);nWARNINGS++;
                                    strcpy(sDISCARD,"WORST"); break;
                                  }
                                  strcpy(sDISCARD,"RANDOM"); break;
                                }
                                else if (!mystricmp(sDISCARD,"Best")) {
                                  if (mystricmp(sREPLACEMENT,"Incremental")){  
                                    fprintf(stderr,"\n%s - Warning line %d: GALib only takes \"%s\" with a steady-state replacement strategy. Default operator \"Worst\" is selected.\n",sEZ_FILE_NAME,EASEALexer.yylineno,sDISCARD);nWARNINGS++;
                                    strcpy(sDISCARD,"WORST"); break;
                                  }
                                  strcpy(sDISCARD,"BEST"); break;
                                }
                                else {
                                  fprintf(stderr,"\n%s - Warning line %d: GALib does not take \"%s\" as a discarding\noperator. Default operator \"Worst\" is selected.\n",sEZ_FILE_NAME,EASEALexer.yylineno,sDISCARD);nWARNINGS++;
                                  strcpy(sDISCARD,"WORST"); break;
                                }
          break;
        case EO : if (!mystricmp(sDISCARD,"Tournament")) {
                             if (yyattribute(4 - 5).dValue>=2) {strcpy(sDISCARD,"DetTournament");sprintf(sDISCARD_PRM,"(%d)",(int) yyattribute(4 - 5).dValue);}
                             else if ((yyattribute(4 - 5).dValue>.5)&&(yyattribute(4 - 5).dValue<=1.0)) {strcpy(sDISCARD,"StochTournament");sprintf(sDISCARD_PRM,"(%f)",(float) yyattribute(4 - 5).dValue);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sDISCARD,"Worst")) {
                             strcpy(sDISCARD,"Worse");
                             fprintf(stderr,"\n%s - Warning line %d: The Worst discarding operator does not take parameters. The parameter will be ignored.\n",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
      }
#line 1568 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 56:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1079 "C:\\repo\\src\\EaseaParse.y"

      if ((!mystricmp(yyattribute(2 - 2).pSymbol->sName,"maximise")) || (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"maximize"))) nMINIMISE=0;
      else if ((!mystricmp(yyattribute(2 - 2).pSymbol->sName,"minimise")) || (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"minimize"))) nMINIMISE=1;
      else {
         fprintf(stderr,"\n%s - Error line %d: The evaluator goal default parameter can only take\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         fprintf(stderr,"two values : maximi[sz]e or minimi[sz]e.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
       }
      
#line 1589 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 57:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1088 "C:\\repo\\src\\EaseaParse.y"

        if (yyattribute(2 - 2).dValue!=0) bELITISM=1;
        else bELITISM=0;
        nELITE=(int)yyattribute(2 - 2).dValue;
        
#line 1606 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 58:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 1093 "C:\\repo\\src\\EaseaParse.y"

        if (yyattribute(2 - 3).dValue!=0) bELITISM=1;
        else bELITISM=0;
        nELITE=(int)yyattribute(2 - 3).dValue*nPOP_SIZE/100;
        
#line 1623 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 59:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1098 "C:\\repo\\src\\EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"weak")) bELITISM=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"strong")) bELITISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Elitism must be \"Strong\" or \"Weak\".\nDefault value \"Strong\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bELITISM=1;
       }
#line 1642 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 60:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1105 "C:\\repo\\src\\EaseaParse.y"

       if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"clone")) nMIG_CLONE=1;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"move")) nMIG_CLONE=0;
      else {
         fprintf(stderr,"\n%s - Error line %d: Looking for \"Move\" or \"Clone\".\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
     } 
#line 1661 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 61:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1112 "C:\\repo\\src\\EaseaParse.y"

        strcpy(sMIG_SEL, yyattribute(2 - 2).pSymbol->sName);
        if (!mystricmp(sMIG_SEL,"EPTrn")){
          fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
          exit(1);
        }
        else if (!mystricmp(sMIG_SEL,"Tournament")) sprintf(sMIG_SEL,"TournamentSelector(false, 1, 2)");
        else if (!mystricmp(sMIG_SEL,"StochTrn")) {
          fprintf(stderr,"\n%s - Warning line %d: The Stochatic Tournament selector needs a parameter in [0.5,1].\nDefault value 1 inserted.\n",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
          sprintf(sMIG_SEL,"TournamentSelector(false, 1, 2)");
        }
        else if (!mystricmp(sMIG_SEL,"RouletteWheel")) sprintf(sMIG_SEL,"RouletteWheelSelector(%s)",nMINIMISE?"false":"true");
        else if (!mystricmp(sMIG_SEL,"Random")) sprintf(sMIG_SEL,"RandomSelector()");
        else if (!mystricmp(sMIG_SEL,"Sequential")) sprintf(sMIG_SEL,"BestNSelector(%s,%d)",nMINIMISE?"false":"true",nPOP_SIZE);
        else if (!mystricmp(sMIG_SEL,"Ranking")) sprintf(sMIG_SEL,"RankingSelector(%s)",nMINIMISE?"false":"true");
        else {
          fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sMIG_SEL);
          exit(1);
        }
     
#line 1693 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 62:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 1132 "C:\\repo\\src\\EaseaParse.y"

        strcpy(sMIG_SEL, yyattribute(2 - 3).pSymbol->sName);
        if (!mystricmp(sMIG_SEL,"EPTrn")){
          fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
          exit(1);
        }
        else if (!mystricmp(sMIG_SEL,"Tournament")) {
          if (yyattribute(3 - 3).dValue>=2) sprintf(sMIG_SEL,"TournamentSelector(%s, 1, %d)",(nMINIMISE?"false":"true"),(int)yyattribute(3 - 3).dValue);
          else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sMIG_SEL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"false":"true"),(float)yyattribute(3 - 3).dValue);
          else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
        }
        else if (!mystricmp(sMIG_SEL,"StochTrn")) {
          if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sMIG_SEL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"false":"true"),(float)yyattribute(3 - 3).dValue);
          else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
        }
        else if (!mystricmp(sMIG_SEL,"RouletteWheel")) sprintf(sMIG_SEL,"RouletteWheelSelector(%s)",nMINIMISE?"false":"true");
        else if (!mystricmp(sMIG_SEL,"Random")) sprintf(sMIG_SEL,"RandomSelector()");
        else if (!mystricmp(sMIG_SEL,"Sequential")) sprintf(sMIG_SEL,"BestNSelector(%s,%d)",nMINIMISE?"false":"true",nPOP_SIZE);
        else if (!mystricmp(sMIG_SEL,"Ranking")) sprintf(sMIG_SEL,"RankingSelector(%s)",nMINIMISE?"false":"true");
        else {
          fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_OFF);
          exit(1);
        }
     
#line 1729 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 63:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1156 "C:\\repo\\src\\EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"neighbours")) {
        sprintf(sMIGRATOR,"DefaultEMigrator");
        sprintf(sMIG_TARGET_SELECTOR,", new RandomTargetSelector()");
      }
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"mesh")) {
        sprintf(sMIGRATOR,"ToAllEmigrator");
        sMIG_TARGET_SELECTOR[0]=0;     
      }
      else {
         fprintf(stderr,"\n%s - Error line %d: Looking for \"Move\" or \"Clone\".\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
     } 
#line 1754 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 64:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1170 "C:\\repo\\src\\EaseaParse.y"
fMIG_FREQ=(float)yyattribute(2 - 2).dValue;
#line 1767 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 65:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1172 "C:\\repo\\src\\EaseaParse.y"
nNB_MIG=(int)yyattribute(2 - 2).dValue;
#line 1780 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 66:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1173 "C:\\repo\\src\\EaseaParse.y"

        strcpy(sIMMIG_SEL, yyattribute(2 - 2).pSymbol->sName);
        if (!mystricmp(sIMMIG_SEL,"EPTrn")){
          fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
          exit(1);
        }
        else if (!mystricmp(sIMMIG_SEL,"Tournament")) sprintf(sIMMIG_SEL,"TournamentSelector(false, 1, 2)");
        else if (!mystricmp(sIMMIG_SEL,"StochTrn")) {
          fprintf(stderr,"\n%s - Warning line %d: The Stochatic Tournament selector needs a parameter in [0.5,1].\nDefault value 1 inserted",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
          sprintf(sIMMIG_SEL,"TournamentSelector(false, 1, 2)");
        }
        else if (!mystricmp(sIMMIG_SEL,"RouletteWheel")) sprintf(sIMMIG_SEL,"RouletteWheelSelector(%s)",nMINIMISE?"false":"true");
        else if (!mystricmp(sIMMIG_SEL,"Random")) sprintf(sIMMIG_SEL,"RandomSelector()");
        else if (!mystricmp(sIMMIG_SEL,"Sequential")) sprintf(sIMMIG_SEL,"BestNSelector(%s,%d)",nMINIMISE?"false":"true",nPOP_SIZE);
        else if (!mystricmp(sIMMIG_SEL,"Ranking")) sprintf(sIMMIG_SEL,"RankingSelector(%s)",nMINIMISE?"false":"true");
        else {
          fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sIMMIG_SEL);
          exit(1);
        }
     
#line 1812 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 67:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 1193 "C:\\repo\\src\\EaseaParse.y"

        strcpy(sIMMIG_SEL, yyattribute(2 - 3).pSymbol->sName);
        if (!mystricmp(sIMMIG_SEL,"EPTrn")){
          fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
          exit(1);
        }
        else if (!mystricmp(sIMMIG_SEL,"Tournament")) {
          if (yyattribute(3 - 3).dValue>=2) sprintf(sIMMIG_SEL,"TournamentSelector(%s, 1, %d)",(nMINIMISE?"false":"true"),(int)yyattribute(3 - 3).dValue);
          else if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sIMMIG_SEL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"false":"true"),(float)yyattribute(3 - 3).dValue);
          else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
        }
        else if (!mystricmp(sIMMIG_SEL,"StochTrn")) {
          if ((yyattribute(3 - 3).dValue>.5)&&(yyattribute(3 - 3).dValue<=1.0)) sprintf(sIMMIG_SEL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"false":"true"),(float)yyattribute(3 - 3).dValue);
          else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
        }
        else if (!mystricmp(sIMMIG_SEL,"RouletteWheel")) sprintf(sIMMIG_SEL,"RouletteWheelSelector(%s)",nMINIMISE?"false":"true");
        else if (!mystricmp(sIMMIG_SEL,"Random")) sprintf(sIMMIG_SEL,"RandomSelector()");
        else if (!mystricmp(sIMMIG_SEL,"Sequential")) sprintf(sIMMIG_SEL,"BestNSelector(%s,%d)",nMINIMISE?"false":"true",nPOP_SIZE);
        else if (!mystricmp(sIMMIG_SEL,"Ranking")) sprintf(sIMMIG_SEL,"RankingSelector(%s)",nMINIMISE?"false":"true");
        else {
          fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_OFF);
          exit(1);
        }
     
#line 1848 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 68:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1217 "C:\\repo\\src\\EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"replace")) nIMMIG_REPL=1;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"add")) nIMMIG_REPL=0;
      else {
         fprintf(stderr,"\n%s - Error line %d: Looking for \"Add\" or \"Replace\".\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
     } 
#line 1867 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 69:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 1227 "C:\\repo\\src\\EaseaParse.y"
 
      if (SymbolTable.find(yyattribute(1 - 3).pSymbol->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 3).pSymbol->sName);
         exit(1);
      }
      (*(YYSTYPE YYFAR*)yyvalptr).dValue = assign(SymbolTable.find(yyattribute(1 - 3).pSymbol->sName), yyattribute(3 - 3).dValue);
    
#line 1886 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 70:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 1234 "C:\\repo\\src\\EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue + yyattribute(3 - 3).dValue; 
#line 1899 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 71:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 1235 "C:\\repo\\src\\EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue - yyattribute(3 - 3).dValue; 
#line 1912 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 72:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 1236 "C:\\repo\\src\\EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue * yyattribute(3 - 3).dValue; 
#line 1925 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 73:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 1237 "C:\\repo\\src\\EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = divide(yyattribute(1 - 3).dValue, yyattribute(3 - 3).dValue); 
#line 1938 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 74:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 1238 "C:\\repo\\src\\EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(2 - 3).dValue; 
#line 1951 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 75:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 1239 "C:\\repo\\src\\EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = -yyattribute(2 - 2).dValue; 
#line 1964 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 76:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 1240 "C:\\repo\\src\\EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 1).dValue; 
#line 1977 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	case 77:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 1241 "C:\\repo\\src\\EaseaParse.y"

      if (SymbolTable.find(yyattribute(1 - 1).pSymbol->sName)==NULL){
	fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 1).pSymbol->sName);
         exit(1);
      }
      (*(YYSTYPE YYFAR*)yyvalptr).dValue = (SymbolTable.find(yyattribute(1 - 1).pSymbol->sName))->dValue;
    
#line 1996 "C:\\repo\\src\\EaseaParse.cpp"
			}
		}
		break;
	default:
		yyassert(0);
		break;
	}
}
#line 1250 "C:\\repo\\src\\EaseaParse.y"

                       
/////////////////////////////////////////////////////////////////////////////
// main

int main(int argc, char *argv[]){
  int n = YYEXIT_FAILURE;
  int nParamNb=0;
  char *sTemp;
  int i=0;
  
  TARGET=bVERBOSE=0;
  sRAW_PROJECT_NAME[0]=0; // used to ask for a filename if no filename is found on the command line.

  while ((++nParamNb) < argc) {
    sTemp=&(argv[nParamNb][0]);
    if ((argv[nParamNb][0]=='-')||(argv[nParamNb][0]=='/')) sTemp=&(argv[nParamNb][1]);
    if (!mystricmp(sTemp,"eo"))  TARGET=EO;
    else if (!mystricmp(sTemp,"galib"))  TARGET=GALIB;
    else if (!mystricmp(sTemp,"dream"))  TARGET=DREAM;
    else if (!mystricmp(sTemp,"v"))  bVERBOSE=true;
    else if (!mystricmp(sTemp,"path"))  {
      if (argv[++nParamNb][0]=='"') {
        strcpy(sEZ_PATH,&(argv[nParamNb][1]));
        while (argv[++nParamNb][strlen(argv[nParamNb])]!='"')
          strcat(sEZ_PATH,argv[nParamNb]);
          argv[nParamNb][strlen(argv[nParamNb])]=0;
          strcat(sEZ_PATH,argv[nParamNb]);
      }
      else {
        if (argv[nParamNb][strlen(argv[nParamNb])-1]=='"') argv[nParamNb][strlen(argv[nParamNb])-1]=0;
        strcpy(sEZ_PATH,argv[nParamNb]);
      }
    }
    else if (!mystricmp(sTemp,"eo_dir"))  {
      if (argv[++nParamNb][0]=='"') {
        strcpy(sEO_DIR,&(argv[nParamNb][1]));
        while (argv[++nParamNb][strlen(argv[nParamNb])]!='"')
          strcat(sEO_DIR,argv[nParamNb]);
          argv[nParamNb][strlen(argv[nParamNb])]=0;
          strcat(sEO_DIR,argv[nParamNb]);
      }
      else {
        if (argv[nParamNb][strlen(argv[nParamNb])-1]=='"') argv[nParamNb][strlen(argv[nParamNb])-1]=0;
        strcpy(sEO_DIR,argv[nParamNb]);
      }
    }
    else strcpy(sRAW_PROJECT_NAME,argv[nParamNb]);
  }

  CEASEAParser Parser;
  if (Parser.create())  n = Parser.yyparse();
  exit(n);
  return n;
}

/////////////////////////////////////////////////////////////////////////////
// EASEAParser commands

int CEASEAParser::create()
{
  if (!yycreate(&EASEALexer)) {
    return 0;
  }
  if (!EASEALexer.create(this, &SymbolTable)) {
    return 0;
  }
  return 1; // success
}

/////////////////////////////////////////////////////////////////////////////
// calc_parser attribute commands

double CEASEAParser::assign(CSymbol* spIdentifier, double dValue)
{
  assert(spIdentifier != NULL);

  spIdentifier->dValue = dValue;
  return spIdentifier->dValue;
}

double CEASEAParser::divide(double a, double b)
{
  if (b == 0) {
    printf("division by zero\n");
    yyforceerror();   // causes a syntax error
    return 0;
  }
  else {
    return a / b;
  }
}

void CEASEAParser::yysyntaxerror(){
  
  fprintf(stderr,"%s \t Error line %d\n",sEZ_FILE_NAME,EASEALexer.yylineno);
  exit(-1);
}

#line 2104 "C:\\repo\\src\\EaseaParse.cpp"
void YYPARSENAME::yytables()
{
	yyattribute_size = sizeof(YYSTYPE);
	yysstack_size = YYSTACK_SIZE;

#ifdef YYDEBUG
	static const yysymbol_t YYNEARFAR YYBASED_CODE symbol[] = {
		{ "$end", 0 },
		{ "\'%\'", 37 },
		{ "\'(\'", 40 },
		{ "\')\'", 41 },
		{ "\'*\'", 42 },
		{ "\'+\'", 43 },
		{ "\',\'", 44 },
		{ "\'-\'", 45 },
		{ "\'/\'", 47 },
		{ "\':\'", 58 },
		{ "\';\'", 59 },
		{ "\'=\'", 61 },
		{ "\'[\'", 91 },
		{ "\']\'", 93 },
		{ "\'{\'", 123 },
		{ "\'}\'", 125 },
		{ "error", 256 },
		{ "UMINUS", 257 },
		{ "CLASSES", 258 },
		{ "GENOME", 259 },
		{ "USER_CTOR", 260 },
		{ "USER_XOVER", 261 },
		{ "USER_MUTATOR", 262 },
		{ "USER_EVALUATOR", 263 },
		{ "END_OF_FUNCTION", 264 },
		{ "END_METHODS", 265 },
		{ "IDENTIFIER", 266 },
		{ "IDENTIFIER2", 267 },
		{ "BOOL", 268 },
		{ "INT", 269 },
		{ "DOUBLE", 270 },
		{ "FLOAT", 271 },
		{ "CHAR", 272 },
		{ "POINTER", 273 },
		{ "NUMBER", 274 },
		{ "NUMBER2", 275 },
		{ "METHODS", 276 },
		{ "STATIC", 277 },
		{ "NB_GEN", 278 },
		{ "NB_ISLANDS", 279 },
		{ "PROP_SEQ", 280 },
		{ "MUT_PROB", 281 },
		{ "XOVER_PROB", 282 },
		{ "POP_SIZE", 283 },
		{ "SELECTOR", 284 },
		{ "RED_PAR", 285 },
		{ "RED_OFF", 286 },
		{ "RED_FINAL", 287 },
		{ "OFFSPRING", 288 },
		{ "SURVPAR", 289 },
		{ "SURVOFF", 290 },
		{ "REPLACEMENT", 291 },
		{ "DISCARD", 292 },
		{ "MINIMAXI", 293 },
		{ "ELITISM", 294 },
		{ "ELITE", 295 },
		{ "MIG_CLONE", 296 },
		{ "MIG_SEL", 297 },
		{ "MIGRATOR", 298 },
		{ "MIG_FREQ", 299 },
		{ "NB_MIG", 300 },
		{ "IMMIG_SEL", 301 },
		{ "IMMIG_REPL", 302 },
		{ NULL, 0 }
	};
	yysymbol = symbol;

	static const char* const YYNEARFAR YYBASED_CODE rule[] = {
		"$accept: EASEA",
		"EASEA: RunParameters GenomeAnalysis",
		"$$1:",
		"GenomeAnalysis: ClassDeclarationsSection GenomeDeclarationSection $$1 StandardFunctionsAnalysis",
		"$$2:",
		"GenomeAnalysis: GenomeDeclarationSection $$2 StandardFunctionsAnalysis",
		"$$3:",
		"ClassDeclarationsSection: CLASSES $$3 ClassDeclarations",
		"ClassDeclarationsSection: CLASSES",
		"ClassDeclarations: ClassDeclaration",
		"ClassDeclarations: ClassDeclarations ClassDeclaration",
		"$$4:",
		"ClassDeclaration: Symbol $$4 \'{\' VariablesDeclarations \'}\'",
		"VariablesDeclarations: VariablesDeclaration",
		"VariablesDeclarations: VariablesDeclarations VariablesDeclaration",
		"$$5:",
		"VariablesDeclaration: Qualifier BaseType $$5 BaseObjects",
		"$$6:",
		"VariablesDeclaration: Qualifier UserType $$6 UserObjects",
		"VariablesDeclaration: MethodsDeclaration",
		"MethodsDeclaration: METHODS END_METHODS",
		"Qualifier: STATIC",
		"Qualifier:",
		"BaseObjects: Objects \';\'",
		"BaseObjects: Objects \':\' BaseConstructorParameters \';\'",
		"UserObjects: Objects \';\'",
		"UserObjects: Objects \':\' UserConstructorParameters \';\'",
		"BaseType: BOOL",
		"BaseType: INT",
		"BaseType: DOUBLE",
		"BaseType: FLOAT",
		"BaseType: CHAR",
		"BaseType: POINTER",
		"UserType: Symbol",
		"Objects: Object",
		"Objects: Objects \',\' Object",
		"Object: Symbol",
		"Object: \'*\' Symbol",
		"Object: Symbol \'[\' Expr \']\'",
		"BaseConstructorParameters: BaseConstructorParameter",
		"BaseConstructorParameters: BaseConstructorParameters BaseConstructorParameter",
		"BaseConstructorParameter: NUMBER",
		"$$7:",
		"GenomeDeclarationSection: GENOME $$7 \'{\' VariablesDeclarations \'}\'",
		"UserConstructorParameters: UserConstructorParameter",
		"UserConstructorParameters: UserConstructorParameters UserConstructorParameter",
		"UserConstructorParameter: Symbol",
		"Symbol: IDENTIFIER",
		"StandardFunctionsAnalysis: StandardFunctionAnalysis",
		"StandardFunctionsAnalysis: StandardFunctionsAnalysis StandardFunctionAnalysis",
		"$$8:",
		"StandardFunctionAnalysis: USER_CTOR $$8 END_OF_FUNCTION",
		"$$9:",
		"StandardFunctionAnalysis: USER_XOVER $$9 END_OF_FUNCTION",
		"$$10:",
		"StandardFunctionAnalysis: USER_MUTATOR $$10 END_OF_FUNCTION",
		"$$11:",
		"StandardFunctionAnalysis: USER_EVALUATOR $$11 END_OF_FUNCTION",
		"RunParameters: Parameter",
		"RunParameters: RunParameters Parameter",
		"Parameter: NB_GEN NUMBER2",
		"Parameter: NB_ISLANDS NUMBER2",
		"Parameter: PROP_SEQ IDENTIFIER2",
		"Parameter: MUT_PROB NUMBER2",
		"Parameter: XOVER_PROB NUMBER2",
		"Parameter: POP_SIZE NUMBER2",
		"Parameter: SELECTOR IDENTIFIER2",
		"Parameter: SELECTOR IDENTIFIER2 NUMBER2",
		"Parameter: RED_PAR IDENTIFIER2",
		"Parameter: RED_PAR IDENTIFIER2 NUMBER2",
		"Parameter: RED_OFF IDENTIFIER2",
		"Parameter: RED_OFF IDENTIFIER2 NUMBER2",
		"Parameter: RED_FINAL IDENTIFIER2",
		"Parameter: RED_FINAL IDENTIFIER2 NUMBER2",
		"Parameter: OFFSPRING NUMBER2",
		"Parameter: OFFSPRING NUMBER2 \'%\'",
		"Parameter: SURVPAR NUMBER2",
		"Parameter: SURVPAR NUMBER2 \'%\'",
		"Parameter: SURVOFF NUMBER2",
		"Parameter: SURVOFF NUMBER2 \'%\'",
		"Parameter: REPLACEMENT IDENTIFIER2",
		"Parameter: DISCARD IDENTIFIER2",
		"Parameter: DISCARD IDENTIFIER2 \'(\' NUMBER2 \')\'",
		"Parameter: MINIMAXI IDENTIFIER2",
		"Parameter: ELITE NUMBER2",
		"Parameter: ELITE NUMBER2 \'%\'",
		"Parameter: ELITISM IDENTIFIER2",
		"Parameter: MIG_CLONE IDENTIFIER2",
		"Parameter: MIG_SEL IDENTIFIER2",
		"Parameter: MIG_SEL IDENTIFIER2 NUMBER2",
		"Parameter: MIGRATOR IDENTIFIER2",
		"Parameter: MIG_FREQ NUMBER2",
		"Parameter: NB_MIG NUMBER2",
		"Parameter: IMMIG_SEL IDENTIFIER2",
		"Parameter: IMMIG_SEL IDENTIFIER2 NUMBER2",
		"Parameter: IMMIG_REPL IDENTIFIER2",
		"Expr: IDENTIFIER \'=\' Expr",
		"Expr: Expr \'+\' Expr",
		"Expr: Expr \'-\' Expr",
		"Expr: Expr \'*\' Expr",
		"Expr: Expr \'/\' Expr",
		"Expr: \'(\' Expr \')\'",
		"Expr: \'-\' Expr",
		"Expr: NUMBER",
		"Expr: IDENTIFIER"
	};
	yyrule = rule;
#endif

	static const yyreduction_t YYNEARFAR YYBASED_CODE reduction[] = {
		{ 0, 1, -1 },
		{ 1, 2, -1 },
		{ 3, 0, 0 },
		{ 2, 4, -1 },
		{ 4, 0, 1 },
		{ 2, 3, -1 },
		{ 6, 0, 2 },
		{ 5, 3, -1 },
		{ 5, 1, 3 },
		{ 7, 1, -1 },
		{ 7, 2, -1 },
		{ 9, 0, 4 },
		{ 8, 5, 5 },
		{ 10, 1, -1 },
		{ 10, 2, -1 },
		{ 12, 0, 6 },
		{ 11, 4, 7 },
		{ 13, 0, 8 },
		{ 11, 4, 9 },
		{ 11, 1, -1 },
		{ 14, 2, 10 },
		{ 15, 1, 11 },
		{ 15, 0, 12 },
		{ 16, 2, -1 },
		{ 16, 4, 13 },
		{ 17, 2, 14 },
		{ 17, 4, 15 },
		{ 18, 1, -1 },
		{ 18, 1, -1 },
		{ 18, 1, -1 },
		{ 18, 1, -1 },
		{ 18, 1, -1 },
		{ 18, 1, -1 },
		{ 19, 1, 16 },
		{ 20, 1, -1 },
		{ 20, 3, -1 },
		{ 21, 1, 17 },
		{ 21, 2, 18 },
		{ 21, 4, 19 },
		{ 22, 1, -1 },
		{ 22, 2, -1 },
		{ 23, 1, 20 },
		{ 25, 0, 21 },
		{ 24, 5, 22 },
		{ 26, 1, -1 },
		{ 26, 2, -1 },
		{ 27, 1, 23 },
		{ 28, 1, 24 },
		{ 29, 1, -1 },
		{ 29, 2, -1 },
		{ 31, 0, 25 },
		{ 30, 3, 26 },
		{ 32, 0, 27 },
		{ 30, 3, 28 },
		{ 33, 0, 29 },
		{ 30, 3, 30 },
		{ 34, 0, 31 },
		{ 30, 3, 32 },
		{ 35, 1, -1 },
		{ 35, 2, -1 },
		{ 36, 2, 33 },
		{ 36, 2, 34 },
		{ 36, 2, 35 },
		{ 36, 2, 36 },
		{ 36, 2, 37 },
		{ 36, 2, 38 },
		{ 36, 2, 39 },
		{ 36, 3, 40 },
		{ 36, 2, 41 },
		{ 36, 3, 42 },
		{ 36, 2, 43 },
		{ 36, 3, 44 },
		{ 36, 2, 45 },
		{ 36, 3, 46 },
		{ 36, 2, 47 },
		{ 36, 3, 48 },
		{ 36, 2, 49 },
		{ 36, 3, 50 },
		{ 36, 2, 51 },
		{ 36, 3, 52 },
		{ 36, 2, 53 },
		{ 36, 2, 54 },
		{ 36, 5, 55 },
		{ 36, 2, 56 },
		{ 36, 2, 57 },
		{ 36, 3, 58 },
		{ 36, 2, 59 },
		{ 36, 2, 60 },
		{ 36, 2, 61 },
		{ 36, 3, 62 },
		{ 36, 2, 63 },
		{ 36, 2, 64 },
		{ 36, 2, 65 },
		{ 36, 2, 66 },
		{ 36, 3, 67 },
		{ 36, 2, 68 },
		{ 37, 3, 69 },
		{ 37, 3, 70 },
		{ 37, 3, 71 },
		{ 37, 3, 72 },
		{ 37, 3, 73 },
		{ 37, 3, 74 },
		{ 37, 2, 75 },
		{ 37, 1, 76 },
		{ 37, 1, 77 }
	};
	yyreduction = reduction;

	static const yytokenaction_t YYNEARFAR YYBASED_CODE tokenaction[] = {
		{ 27, YYAT_SHIFT, 54 },
		{ 156, YYAT_SHIFT, 138 },
		{ 145, YYAT_SHIFT, 158 },
		{ 148, YYAT_SHIFT, 160 },
		{ 120, YYAT_SHIFT, 123 },
		{ 142, YYAT_ERROR, 0 },
		{ 156, YYAT_SHIFT, 139 },
		{ 93, YYAT_SHIFT, 105 },
		{ 93, YYAT_SHIFT, 106 },
		{ 93, YYAT_SHIFT, 107 },
		{ 93, YYAT_SHIFT, 108 },
		{ 93, YYAT_SHIFT, 109 },
		{ 93, YYAT_SHIFT, 110 },
		{ 150, YYAT_SHIFT, 162 },
		{ 150, YYAT_SHIFT, 153 },
		{ 150, YYAT_SHIFT, 154 },
		{ 140, YYAT_SHIFT, 152 },
		{ 150, YYAT_SHIFT, 155 },
		{ 136, YYAT_ERROR, 0 },
		{ 150, YYAT_SHIFT, 156 },
		{ 27, YYAT_SHIFT, 1 },
		{ 27, YYAT_SHIFT, 2 },
		{ 27, YYAT_SHIFT, 3 },
		{ 27, YYAT_SHIFT, 4 },
		{ 27, YYAT_SHIFT, 5 },
		{ 27, YYAT_SHIFT, 6 },
		{ 27, YYAT_SHIFT, 7 },
		{ 27, YYAT_SHIFT, 8 },
		{ 27, YYAT_SHIFT, 9 },
		{ 27, YYAT_SHIFT, 10 },
		{ 27, YYAT_SHIFT, 11 },
		{ 27, YYAT_SHIFT, 12 },
		{ 27, YYAT_SHIFT, 13 },
		{ 27, YYAT_SHIFT, 14 },
		{ 27, YYAT_SHIFT, 15 },
		{ 27, YYAT_SHIFT, 16 },
		{ 27, YYAT_SHIFT, 17 },
		{ 27, YYAT_SHIFT, 18 },
		{ 27, YYAT_SHIFT, 19 },
		{ 27, YYAT_SHIFT, 20 },
		{ 27, YYAT_SHIFT, 21 },
		{ 27, YYAT_SHIFT, 22 },
		{ 27, YYAT_SHIFT, 23 },
		{ 27, YYAT_SHIFT, 24 },
		{ 27, YYAT_SHIFT, 25 },
		{ 130, YYAT_SHIFT, 133 },
		{ 163, YYAT_SHIFT, 153 },
		{ 163, YYAT_SHIFT, 154 },
		{ 134, YYAT_ERROR, 0 },
		{ 163, YYAT_SHIFT, 155 },
		{ 133, YYAT_SHIFT, 124 },
		{ 163, YYAT_SHIFT, 156 },
		{ 97, YYAT_SHIFT, 82 },
		{ 97, YYAT_SHIFT, 83 },
		{ 97, YYAT_SHIFT, 84 },
		{ 97, YYAT_SHIFT, 85 },
		{ 125, YYAT_SHIFT, 132 },
		{ 142, YYAT_SHIFT, 157 },
		{ 103, YYAT_REDUCE, 22 },
		{ 130, YYAT_SHIFT, 136 },
		{ 130, YYAT_SHIFT, 137 },
		{ 86, YYAT_SHIFT, 82 },
		{ 86, YYAT_SHIFT, 83 },
		{ 86, YYAT_SHIFT, 84 },
		{ 86, YYAT_SHIFT, 85 },
		{ 81, YYAT_SHIFT, 82 },
		{ 81, YYAT_SHIFT, 83 },
		{ 81, YYAT_SHIFT, 84 },
		{ 81, YYAT_SHIFT, 85 },
		{ 166, YYAT_SHIFT, 153 },
		{ 165, YYAT_SHIFT, 153 },
		{ 127, YYAT_SHIFT, 134 },
		{ 127, YYAT_SHIFT, 135 },
		{ 101, YYAT_SHIFT, 119 },
		{ 166, YYAT_SHIFT, 156 },
		{ 165, YYAT_SHIFT, 156 },
		{ 0, YYAT_ERROR, 0 },
		{ 0, YYAT_ERROR, 0 },
		{ 100, YYAT_SHIFT, 118 },
		{ 99, YYAT_SHIFT, 117 },
		{ 98, YYAT_SHIFT, 116 },
		{ 94, YYAT_SHIFT, 114 },
		{ 91, YYAT_SHIFT, 104 },
		{ 89, YYAT_SHIFT, 103 },
		{ 78, YYAT_SHIFT, 76 },
		{ 75, YYAT_SHIFT, 88 },
		{ 72, YYAT_SHIFT, 80 },
		{ 67, YYAT_SHIFT, 75 },
		{ 57, YYAT_SHIFT, 55 },
		{ 54, YYAT_REDUCE, 6 },
		{ 52, YYAT_SHIFT, 70 },
		{ 48, YYAT_SHIFT, 69 },
		{ 46, YYAT_SHIFT, 68 },
		{ 43, YYAT_SHIFT, 67 },
		{ 41, YYAT_SHIFT, 66 },
		{ 40, YYAT_SHIFT, 65 },
		{ 39, YYAT_SHIFT, 64 },
		{ 38, YYAT_SHIFT, 63 },
		{ 37, YYAT_SHIFT, 62 },
		{ 36, YYAT_SHIFT, 61 },
		{ 35, YYAT_SHIFT, 60 },
		{ 26, YYAT_ACCEPT, 0 },
		{ 25, YYAT_SHIFT, 53 },
		{ 24, YYAT_SHIFT, 52 },
		{ 23, YYAT_SHIFT, 51 },
		{ 22, YYAT_SHIFT, 50 },
		{ 21, YYAT_SHIFT, 49 },
		{ 20, YYAT_SHIFT, 48 },
		{ 19, YYAT_SHIFT, 47 },
		{ 18, YYAT_SHIFT, 46 },
		{ 17, YYAT_SHIFT, 45 },
		{ 16, YYAT_SHIFT, 44 },
		{ 15, YYAT_SHIFT, 43 },
		{ 14, YYAT_SHIFT, 42 },
		{ 13, YYAT_SHIFT, 41 },
		{ 12, YYAT_SHIFT, 40 },
		{ 11, YYAT_SHIFT, 39 },
		{ 10, YYAT_SHIFT, 38 },
		{ 9, YYAT_SHIFT, 37 },
		{ 8, YYAT_SHIFT, 36 },
		{ 7, YYAT_SHIFT, 35 },
		{ 6, YYAT_SHIFT, 34 },
		{ 5, YYAT_SHIFT, 33 },
		{ 4, YYAT_SHIFT, 32 },
		{ 3, YYAT_SHIFT, 31 },
		{ 2, YYAT_SHIFT, 30 },
		{ 1, YYAT_SHIFT, 29 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 120, YYAT_SHIFT, 91 },
		{ 120, YYAT_SHIFT, 92 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 148, YYAT_SHIFT, 76 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 145, YYAT_SHIFT, 144 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 156, YYAT_SHIFT, 140 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 156, YYAT_SHIFT, 141 }
	};
	yytokenaction = tokenaction;
	yytokenaction_size = 236;

	static const yystateaction_t YYNEARFAR YYBASED_CODE stateaction[] = {
		{ -182, 1, YYAT_DEFAULT, 27 },
		{ -149, 1, YYAT_DEFAULT, 67 },
		{ -150, 1, YYAT_DEFAULT, 67 },
		{ -143, 1, YYAT_DEFAULT, 25 },
		{ -152, 1, YYAT_DEFAULT, 67 },
		{ -153, 1, YYAT_DEFAULT, 67 },
		{ -154, 1, YYAT_DEFAULT, 67 },
		{ -147, 1, YYAT_DEFAULT, 25 },
		{ -148, 1, YYAT_DEFAULT, 25 },
		{ -149, 1, YYAT_DEFAULT, 25 },
		{ -150, 1, YYAT_DEFAULT, 25 },
		{ -159, 1, YYAT_DEFAULT, 67 },
		{ -160, 1, YYAT_DEFAULT, 67 },
		{ -161, 1, YYAT_DEFAULT, 67 },
		{ -154, 1, YYAT_DEFAULT, 25 },
		{ -155, 1, YYAT_DEFAULT, 25 },
		{ -156, 1, YYAT_DEFAULT, 25 },
		{ -157, 1, YYAT_DEFAULT, 25 },
		{ -166, 1, YYAT_DEFAULT, 67 },
		{ -159, 1, YYAT_DEFAULT, 25 },
		{ -160, 1, YYAT_DEFAULT, 25 },
		{ -161, 1, YYAT_DEFAULT, 25 },
		{ -170, 1, YYAT_DEFAULT, 67 },
		{ -171, 1, YYAT_DEFAULT, 67 },
		{ -164, 1, YYAT_DEFAULT, 25 },
		{ -165, 1, YYAT_ERROR, 0 },
		{ 101, 1, YYAT_ERROR, 0 },
		{ -258, 1, YYAT_DEFAULT, 57 },
		{ 0, 0, YYAT_REDUCE, 58 },
		{ 0, 0, YYAT_REDUCE, 60 },
		{ 0, 0, YYAT_REDUCE, 61 },
		{ 0, 0, YYAT_REDUCE, 62 },
		{ 0, 0, YYAT_REDUCE, 63 },
		{ 0, 0, YYAT_REDUCE, 64 },
		{ 0, 0, YYAT_REDUCE, 65 },
		{ -175, 1, YYAT_REDUCE, 66 },
		{ -176, 1, YYAT_REDUCE, 68 },
		{ -177, 1, YYAT_REDUCE, 70 },
		{ -178, 1, YYAT_REDUCE, 72 },
		{ 59, 1, YYAT_REDUCE, 74 },
		{ 58, 1, YYAT_REDUCE, 76 },
		{ 57, 1, YYAT_REDUCE, 78 },
		{ 0, 0, YYAT_REDUCE, 80 },
		{ 53, 1, YYAT_REDUCE, 81 },
		{ 0, 0, YYAT_REDUCE, 83 },
		{ 0, 0, YYAT_REDUCE, 86 },
		{ 55, 1, YYAT_REDUCE, 84 },
		{ 0, 0, YYAT_REDUCE, 87 },
		{ -184, 1, YYAT_REDUCE, 88 },
		{ 0, 0, YYAT_REDUCE, 90 },
		{ 0, 0, YYAT_REDUCE, 91 },
		{ 0, 0, YYAT_REDUCE, 92 },
		{ -185, 1, YYAT_REDUCE, 93 },
		{ 0, 0, YYAT_REDUCE, 95 },
		{ -177, 1, YYAT_REDUCE, 8 },
		{ 0, 0, YYAT_REDUCE, 42 },
		{ 0, 0, YYAT_REDUCE, 1 },
		{ -171, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 4 },
		{ 0, 0, YYAT_REDUCE, 59 },
		{ 0, 0, YYAT_REDUCE, 67 },
		{ 0, 0, YYAT_REDUCE, 69 },
		{ 0, 0, YYAT_REDUCE, 71 },
		{ 0, 0, YYAT_REDUCE, 73 },
		{ 0, 0, YYAT_REDUCE, 75 },
		{ 0, 0, YYAT_REDUCE, 77 },
		{ 0, 0, YYAT_REDUCE, 79 },
		{ -188, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 85 },
		{ 0, 0, YYAT_REDUCE, 89 },
		{ 0, 0, YYAT_REDUCE, 94 },
		{ 0, 0, YYAT_DEFAULT, 136 },
		{ -37, 1, YYAT_DEFAULT, 89 },
		{ 0, 0, YYAT_REDUCE, 2 },
		{ 0, 0, YYAT_DEFAULT, 81 },
		{ 44, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 47 },
		{ 0, 0, YYAT_REDUCE, 11 },
		{ -182, 1, YYAT_REDUCE, 7 },
		{ 0, 0, YYAT_REDUCE, 9 },
		{ 0, 0, YYAT_DEFAULT, 103 },
		{ -195, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 50 },
		{ 0, 0, YYAT_REDUCE, 52 },
		{ 0, 0, YYAT_REDUCE, 54 },
		{ 0, 0, YYAT_REDUCE, 56 },
		{ -199, 1, YYAT_REDUCE, 5 },
		{ 0, 0, YYAT_REDUCE, 48 },
		{ 0, 0, YYAT_REDUCE, 82 },
		{ -40, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 10 },
		{ -183, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 21 },
		{ -261, 1, YYAT_DEFAULT, 136 },
		{ -44, 1, YYAT_DEFAULT, 120 },
		{ 0, 0, YYAT_REDUCE, 13 },
		{ 0, 0, YYAT_REDUCE, 19 },
		{ -208, 1, YYAT_REDUCE, 3 },
		{ -184, 1, YYAT_DEFAULT, 101 },
		{ -185, 1, YYAT_DEFAULT, 101 },
		{ -186, 1, YYAT_DEFAULT, 101 },
		{ -191, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 49 },
		{ -67, 1, YYAT_DEFAULT, 120 },
		{ 0, 0, YYAT_REDUCE, 20 },
		{ 0, 0, YYAT_REDUCE, 27 },
		{ 0, 0, YYAT_REDUCE, 28 },
		{ 0, 0, YYAT_REDUCE, 29 },
		{ 0, 0, YYAT_REDUCE, 30 },
		{ 0, 0, YYAT_REDUCE, 31 },
		{ 0, 0, YYAT_REDUCE, 32 },
		{ 0, 0, YYAT_REDUCE, 15 },
		{ 0, 0, YYAT_REDUCE, 17 },
		{ 0, 0, YYAT_REDUCE, 33 },
		{ 0, 0, YYAT_REDUCE, 43 },
		{ 0, 0, YYAT_REDUCE, 14 },
		{ 0, 0, YYAT_REDUCE, 51 },
		{ 0, 0, YYAT_REDUCE, 53 },
		{ 0, 0, YYAT_REDUCE, 55 },
		{ 0, 0, YYAT_REDUCE, 57 },
		{ -121, 1, YYAT_REDUCE, 22 },
		{ 0, 0, YYAT_DEFAULT, 133 },
		{ 0, 0, YYAT_DEFAULT, 133 },
		{ 0, 0, YYAT_REDUCE, 12 },
		{ 0, 0, YYAT_DEFAULT, 136 },
		{ -35, 1, YYAT_REDUCE, 36 },
		{ 0, 0, YYAT_REDUCE, 16 },
		{ 13, 1, YYAT_DEFAULT, 130 },
		{ 0, 0, YYAT_REDUCE, 34 },
		{ 0, 0, YYAT_REDUCE, 18 },
		{ 1, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 37 },
		{ 0, 0, YYAT_DEFAULT, 156 },
		{ 8, 1, YYAT_DEFAULT, 136 },
		{ -11, 1, YYAT_DEFAULT, 145 },
		{ 0, 0, YYAT_REDUCE, 23 },
		{ -41, 1, YYAT_DEFAULT, 148 },
		{ 0, 0, YYAT_REDUCE, 25 },
		{ 0, 0, YYAT_DEFAULT, 156 },
		{ 0, 0, YYAT_DEFAULT, 156 },
		{ -45, 1, YYAT_REDUCE, 104 },
		{ 0, 0, YYAT_REDUCE, 103 },
		{ -36, 1, YYAT_DEFAULT, 150 },
		{ 0, 0, YYAT_REDUCE, 35 },
		{ 0, 0, YYAT_REDUCE, 41 },
		{ -57, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 39 },
		{ 0, 0, YYAT_REDUCE, 46 },
		{ -56, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 44 },
		{ -28, 1, YYAT_DEFAULT, 75 },
		{ 0, 0, YYAT_REDUCE, 102 },
		{ 0, 0, YYAT_DEFAULT, 156 },
		{ 0, 0, YYAT_DEFAULT, 156 },
		{ 0, 0, YYAT_DEFAULT, 156 },
		{ 0, 0, YYAT_DEFAULT, 156 },
		{ -39, 1, YYAT_DEFAULT, 136 },
		{ 0, 0, YYAT_REDUCE, 38 },
		{ 0, 0, YYAT_REDUCE, 24 },
		{ 0, 0, YYAT_REDUCE, 40 },
		{ 0, 0, YYAT_REDUCE, 26 },
		{ 0, 0, YYAT_REDUCE, 45 },
		{ 0, 0, YYAT_REDUCE, 101 },
		{ 4, 1, YYAT_REDUCE, 96 },
		{ 0, 0, YYAT_REDUCE, 99 },
		{ 28, 1, YYAT_REDUCE, 97 },
		{ 27, 1, YYAT_REDUCE, 98 },
		{ 0, 0, YYAT_REDUCE, 100 }
	};
	yystateaction = stateaction;

	static const yynontermgoto_t YYNEARFAR YYBASED_CODE nontermgoto[] = {
		{ 27, 56 },
		{ 0, 26 },
		{ 133, 143 },
		{ 27, 57 },
		{ 78, 90 },
		{ 93, 111 },
		{ 93, 112 },
		{ 121, 126 },
		{ 156, 167 },
		{ 133, 125 },
		{ 122, 129 },
		{ 121, 127 },
		{ 121, 128 },
		{ 122, 130 },
		{ 122, 128 },
		{ 93, 113 },
		{ 120, 115 },
		{ 148, 161 },
		{ 148, 147 },
		{ 120, 96 },
		{ 120, 93 },
		{ 155, 166 },
		{ 27, 58 },
		{ 154, 165 },
		{ 78, 77 },
		{ 136, 148 },
		{ 136, 149 },
		{ 134, 145 },
		{ 134, 146 },
		{ 103, 120 },
		{ 103, 95 },
		{ 81, 97 },
		{ 81, 87 },
		{ 153, 164 },
		{ 27, 59 },
		{ 0, 27 },
		{ 0, 28 },
		{ 71, 78 },
		{ 71, 79 },
		{ 152, 163 },
		{ 145, 159 },
		{ 139, 151 },
		{ 138, 150 },
		{ 132, 142 },
		{ 124, 131 },
		{ 112, 122 },
		{ 111, 121 },
		{ 97, 102 },
		{ 85, 101 },
		{ 84, 100 },
		{ 83, 99 },
		{ 82, 98 },
		{ 80, 94 },
		{ 77, 89 },
		{ 74, 86 },
		{ 73, 81 },
		{ 58, 74 },
		{ 57, 73 },
		{ 55, 72 },
		{ 54, 71 }
	};
	yynontermgoto = nontermgoto;
	int yynontermgoto_size = 60;

	static const yystategoto_t YYNEARFAR YYBASED_CODE stategoto[] = {
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ -2, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 53, -1 },
		{ 33, -1 },
		{ 0, -1 },
		{ 33, -1 },
		{ 52, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 30, 78 },
		{ 0, -1 },
		{ 52, -1 },
		{ 25, 81 },
		{ 0, -1 },
		{ 0, -1 },
		{ 44, -1 },
		{ -4, -1 },
		{ 0, -1 },
		{ 42, 103 },
		{ 2, -1 },
		{ 20, -1 },
		{ 18, -1 },
		{ 16, -1 },
		{ 14, -1 },
		{ 0, 97 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ -13, -1 },
		{ 0, 120 },
		{ 0, -1 },
		{ 0, -1 },
		{ 17, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 19, 120 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 34, -1 },
		{ 32, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 5, -1 },
		{ -9, 133 },
		{ -7, 133 },
		{ 0, -1 },
		{ 16, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 6, -1 },
		{ -19, -1 },
		{ 5, -1 },
		{ 0, -1 },
		{ -1, 148 },
		{ 0, -1 },
		{ 5, -1 },
		{ 4, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 17, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ -10, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 2, -1 },
		{ -4, -1 },
		{ -14, -1 },
		{ -16, -1 },
		{ -29, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 }
	};
	yystategoto = stategoto;

	yydestructorptr = NULL;

	yytokendestptr = NULL;
	yytokendest_size = 0;
	yytokendestbase = 0;
}
