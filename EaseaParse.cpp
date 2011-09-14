#include <cyacc.h>

#line 1 "EaseaParse.y"

/****************************************************************************
EaseaLex.y
Parser for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@polytechnique.fr)
Ecole Polytechnique
Centre de Math�matiques Appliqu�es
91128 Palaiseau cedex
****************************************************************************/

#include "Easea.h"
#include "debug.h"
#include "EaseaYTools.h"

// Globals     
CSymbol *pCURRENT_CLASS;
CSymbol *pCURRENT_TYPE;
CSymbol *pGENOME;
CSymbol* pCLASSES[128];
char sRAW_PROJECT_NAME[1000];
int nClasses_nb = 0;
char sPROJECT_NAME[1000];
char sLOWER_CASE_PROJECT_NAME[1000];
char sEZ_FILE_NAME[1000];
char sEO_DIR[1000];
char sEZ_PATH[1000];
char sTPL_DIR[1000];
int TARGET,TARGET_FLAVOR;
int OPERATING_SYSTEM;
int nWARNINGS=0;
int nERRORS=0;
char sSELECTOR[50], sSELECTOR_OPERATOR[50];
float fSELECT_PRM=0.0;
char sRED_PAR[50], sRED_PAR_OPERATOR[50];
float fRED_PAR_PRM=0.0;//[50] = {0};
char sRED_OFF[50], sRED_OFF_OPERATOR[50];
float fRED_OFF_PRM;//[50] = {0};
char sRED_FINAL[50], sRED_FINAL_OPERATOR[50];
float fRED_FINAL_PRM=0.0;//[50];
int nMINIMISE=2;
int nELITE=0;
bool bELITISM=0;
bool bVERBOSE=0;
bool bLINE_NUM_EZ_FILE=1;
bool bPRINT_STATS=1;
bool bPLOT_STATS=0;
bool bGENERATE_CSV_FILE=0, bGENERATE_R_SCRIPT=0, bGENERATE_GNUPLOT_SCRIPT=0;
bool bSAVE_POPULATION=0, bSTART_FROM_FILE=0;
bool bBALDWINISM=0; //memetic
bool bREMOTE_ISLAND_MODEL=0; //remote island model
float fMIGRATION_PROBABILITY=0.0;
char sIP_FILE[128]; //remote island model
int nPOP_SIZE, nOFF_SIZE;
float fSURV_PAR_SIZE=-1.0, fSURV_OFF_SIZE=-1.0;
char *nGENOME_NAME;
int nPROBLEM_DIM;
int nNB_GEN=0;
int nNB_OPT_IT=0;
int nTIME_LIMIT=0;
float fMUT_PROB;
float fXOVER_PROB;
FILE *fpOutputFile, *fpTemplateFile, *fpGenomeFile;//, *fpExplodedGenomeFile;

 unsigned iMAX_INIT_TREE_D,iMIN_INIT_TREE_D,iMAX_TREE_D,iNB_GPU,iPRG_BUF_SIZE,iMAX_TREE_DEPTH,iMAX_XOVER_DEPTH,iNO_FITNESS_CASES;

#line 71 "EaseaParse.cpp"
// repeated because of possible precompiled header
#include <cyacc.h>

#include "EaseaParse.h"

/////////////////////////////////////////////////////////////////////////////
// constructor

YYPARSENAME::YYPARSENAME()
{
	yytables();
#line 183 "EaseaParse.y"

      CSymbol *pNewBaseType;

      pNewBaseType=new CSymbol("bool");
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

#line 117 "EaseaParse.cpp"
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
#line 223 "EaseaParse.y"

        if (bVERBOSE){ printf("                    _______________________________________\n");
        printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);}
      
#line 167 "EaseaParse.cpp"
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
#line 228 "EaseaParse.y"

        if (bVERBOSE) printf("                    _______________________________________\n");
        if (bVERBOSE) printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);
      
#line 183 "EaseaParse.cpp"
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
#line 236 "EaseaParse.y"

    if (bVERBOSE) printf("Declaration of user classes :\n\n");
#line 197 "EaseaParse.cpp"
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
#line 239 "EaseaParse.y"

      if (bVERBOSE) printf("No user class declaration found other than GenomeClass.\n");
#line 211 "EaseaParse.cpp"
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
#line 249 "EaseaParse.y"

      pCURRENT_CLASS=SymbolTable.insert(yyattribute(1 - 1).pSymbol);  
      pCURRENT_CLASS->pSymbolList=new CLList<CSymbol *>();
      yyattribute(1 - 1).pSymbol->ObjectType=oUserClass;
      //DEBUG_PRT("Yacc Symbol declaration %s %d",$1->sName,$1->nSize);
      pCLASSES[nClasses_nb++] = yyattribute(1 - 1).pSymbol;
    
#line 230 "EaseaParse.cpp"
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
#line 256 "EaseaParse.y"

      if (bVERBOSE) printf("Class %s declared for %d bytes.\n\n",yyattribute(1 - 5).pSymbol->sName,yyattribute(1 - 5).pSymbol->nSize);
      //DEBUG_PRT("Yacc variable declaration %s %d",$1->sName,$1->nSize);
    
#line 246 "EaseaParse.cpp"
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
#line 269 "EaseaParse.y"
pCURRENT_TYPE=yyattribute(2 - 2).pSymbol; pCURRENT_TYPE->ObjectQualifier=yyattribute(1 - 2).ObjectQualifier;
#line 259 "EaseaParse.cpp"
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
#line 269 "EaseaParse.y"

#line 272 "EaseaParse.cpp"
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
#line 270 "EaseaParse.y"
pCURRENT_TYPE=yyattribute(2 - 2).pSymbol; pCURRENT_TYPE->ObjectQualifier=yyattribute(1 - 2).ObjectQualifier;
#line 285 "EaseaParse.cpp"
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
#line 270 "EaseaParse.y"

#line 298 "EaseaParse.cpp"
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
#line 275 "EaseaParse.y"

    pCURRENT_CLASS->sString = new char[strlen(yyattribute(2 - 2).szString) + 1];
    strcpy(pCURRENT_CLASS->sString, yyattribute(2 - 2).szString);      
    if (bVERBOSE) printf("\n    The following methods have been declared:\n\n%s\n\n",pCURRENT_CLASS->sString);
    
#line 315 "EaseaParse.cpp"
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
#line 283 "EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).ObjectQualifier=1;
#line 328 "EaseaParse.cpp"
			}
		}
		break;
	case 12:
		{
#line 284 "EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).ObjectQualifier=0;
#line 336 "EaseaParse.cpp"
		}
		break;
	case 13:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 289 "EaseaParse.y"

#line 348 "EaseaParse.cpp"
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
#line 293 "EaseaParse.y"

#line 361 "EaseaParse.cpp"
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
#line 294 "EaseaParse.y"

#line 374 "EaseaParse.cpp"
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
#line 308 "EaseaParse.y"
  
      CSymbol *pSym=SymbolTable.find(yyattribute(1 - 1).pSymbol->sName);
      if (pSym==NULL) {
        fprintf(stderr,"\n%s - Error line %d: Class \"%s\" was not defined.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 1).pSymbol->sName);
        fprintf(stderr,"Only base types (bool, int, float, double, char) or new user classes defined\nwithin the \"User classes\" sections are allowed.\n");
        exit(1);
      }       
      else (*(YYSTYPE YYFAR*)yyvalptr).pSymbol=pSym;
    
#line 395 "EaseaParse.cpp"
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
#line 329 "EaseaParse.y"

//      CSymbol *pSym;
//      pSym=$1;
        yyattribute(1 - 1).pSymbol->nSize=pCURRENT_TYPE->nSize;
        yyattribute(1 - 1).pSymbol->pClass=pCURRENT_CLASS;
        yyattribute(1 - 1).pSymbol->pType=pCURRENT_TYPE;
        yyattribute(1 - 1).pSymbol->ObjectType=oObject;
        yyattribute(1 - 1).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
        pCURRENT_CLASS->nSize+=yyattribute(1 - 1).pSymbol->nSize;
        pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(1 - 1).pSymbol));
        if (bVERBOSE) printf("    %s variable declared (%d bytes)\n",yyattribute(1 - 1).pSymbol->sName,yyattribute(1 - 1).pSymbol->nSize);
    
#line 419 "EaseaParse.cpp"
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
#line 341 "EaseaParse.y"

      yyattribute(2 - 2).pSymbol->nSize=sizeof (char *);
      yyattribute(2 - 2).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(2 - 2).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(2 - 2).pSymbol->ObjectType=oPointer;
      yyattribute(2 - 2).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(2 - 2).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(2 - 2).pSymbol));
      if (bVERBOSE) printf("    %s pointer declared (%d bytes)\n",yyattribute(2 - 2).pSymbol->sName,yyattribute(2 - 2).pSymbol->nSize);
    
#line 441 "EaseaParse.cpp"
			}
		}
		break;
	case 19:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 351 "EaseaParse.y"

      yyattribute(2 - 2).pSymbol->nSize=sizeof (char *);
      yyattribute(2 - 2).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(2 - 2).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(2 - 2).pSymbol->ObjectType=oPointer;
      yyattribute(2 - 2).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(2 - 2).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(2 - 2).pSymbol));
      if (bVERBOSE) printf("    %s NULL pointer declared (%d bytes)\n",yyattribute(2 - 2).pSymbol->sName,yyattribute(2 - 2).pSymbol->nSize);
    
#line 463 "EaseaParse.cpp"
			}
		}
		break;
	case 20:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 361 "EaseaParse.y"

      yyattribute(3 - 3).pSymbol->nSize=sizeof (char *);
      yyattribute(3 - 3).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(3 - 3).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(3 - 3).pSymbol->ObjectType=oPointer;
      yyattribute(3 - 3).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(3 - 3).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(3 - 3).pSymbol));
      if (bVERBOSE) printf("    %s pointer of pointer declared (%d bytes)\n",yyattribute(3 - 3).pSymbol->sName,yyattribute(3 - 3).pSymbol->nSize);
      fprintf(stderr,"Pointer of pointer doesn't work properly yet\n");
      exit(-1);
    
#line 487 "EaseaParse.cpp"
			}
		}
		break;
	case 21:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 374 "EaseaParse.y"

      if((TARGET_FLAVOR==CMAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName,"Genome")==0) { nGENOME_NAME=yyattribute(1 - 4).pSymbol->sName; nPROBLEM_DIM=(int)yyattribute(3 - 4).dValue;}

      //printf("DEBUG : size of $3 %d nSize %d\n",(int)$3,pCURRENT_TYPE->nSize);

      yyattribute(1 - 4).pSymbol->nSize=pCURRENT_TYPE->nSize*(int)yyattribute(3 - 4).dValue;
      yyattribute(1 - 4).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(1 - 4).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(1 - 4).pSymbol->ObjectType=oArray;
      yyattribute(1 - 4).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(1 - 4).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(1 - 4).pSymbol));
      if (bVERBOSE) printf("    %s array declared (%d bytes)\n",yyattribute(1 - 4).pSymbol->sName,yyattribute(1 - 4).pSymbol->nSize);
    
#line 513 "EaseaParse.cpp"
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
#line 388 "EaseaParse.y"


    // this is for support of pointer array. This should be done in a more generic way in a later version
      if((TARGET_FLAVOR==CMAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName,"Genome")==0) { 
	nGENOME_NAME=yyattribute(2 - 5).pSymbol->sName; nPROBLEM_DIM=(int)yyattribute(4 - 5).dValue;
      }
      
      //pCURRENT_CLASS->nSize

      yyattribute(2 - 5).pSymbol->nSize=sizeof(char*)*(int)yyattribute(4 - 5).dValue;
      yyattribute(2 - 5).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(2 - 5).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(2 - 5).pSymbol->ObjectType=oArrayPointer;
      yyattribute(2 - 5).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(2 - 5).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(2 - 5).pSymbol));

      printf("DEBUG : size of $4 %d nSize %d\n",(int)yyattribute(4 - 5).dValue,pCURRENT_TYPE->nSize);
      if (bVERBOSE) printf("    %s array of pointers declared (%d bytes)\n",yyattribute(2 - 5).pSymbol->sName,yyattribute(2 - 5).pSymbol->nSize);
    
#line 545 "EaseaParse.cpp"
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
#line 416 "EaseaParse.y"

#line 558 "EaseaParse.cpp"
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
#line 429 "EaseaParse.y"

    ////DEBUG_PRT("Yacc genome decl %s",$1.pSymbol->sName);
      if (bVERBOSE) printf ("\nGenome declaration analysis :\n\n");
      pGENOME=new CSymbol("Genome");
      pCURRENT_CLASS=SymbolTable.insert(pGENOME);  
      pGENOME->pSymbolList=new CLList<CSymbol *>();
      pGENOME->ObjectType=oUserClass;
      pGENOME->ObjectQualifier=0;
      pGENOME->sString=NULL;
    
#line 580 "EaseaParse.cpp"
			}
		}
		break;
	case 25:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[6];
			yyinitdebug((void YYFAR**)yya, 6);
#endif
			{
#line 439 "EaseaParse.y"

#line 593 "EaseaParse.cpp"
			}
		}
		break;
	case 26:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 456 "EaseaParse.y"

#line 606 "EaseaParse.cpp"
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
#line 460 "EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).pSymbol=yyattribute(1 - 1).pSymbol;
#line 619 "EaseaParse.cpp"
			}
		}
		break;
	case 28:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 469 "EaseaParse.y"
         
      if (bVERBOSE) printf("Inserting genome initialiser (taken from .ez file).\n");
    
#line 634 "EaseaParse.cpp"
			}
		}
		break;
	case 29:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 472 "EaseaParse.y"

#line 647 "EaseaParse.cpp"
			}
		}
		break;
	case 30:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 473 "EaseaParse.y"

      if (bVERBOSE) printf("Inserting user genome crossover (taken from .ez file).\n");
    
#line 662 "EaseaParse.cpp"
			}
		}
		break;
	case 31:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 476 "EaseaParse.y"

#line 675 "EaseaParse.cpp"
			}
		}
		break;
	case 32:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 477 "EaseaParse.y"

      if (bVERBOSE) printf("Inserting user genome mutator (taken from .ez file).\n");
    
#line 690 "EaseaParse.cpp"
			}
		}
		break;
	case 33:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 480 "EaseaParse.y"

#line 703 "EaseaParse.cpp"
			}
		}
		break;
	case 34:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 481 "EaseaParse.y"
 
      if (bVERBOSE) printf("Inserting user genome evaluator (taken from .ez file).\n");
    
#line 718 "EaseaParse.cpp"
			}
		}
		break;
	case 35:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 484 "EaseaParse.y"

#line 731 "EaseaParse.cpp"
			}
		}
		break;
	case 36:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 485 "EaseaParse.y"
 
      if (bVERBOSE) printf("Inserting user genome optimiser (taken from .ez file).\n");
    
#line 746 "EaseaParse.cpp"
			}
		}
		break;
	case 37:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 488 "EaseaParse.y"

#line 759 "EaseaParse.cpp"
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
#line 489 "EaseaParse.y"

     //DEBUG_PRT("User makefile options have been reduced");
     
#line 774 "EaseaParse.cpp"
			}
		}
		break;
	case 39:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 492 "EaseaParse.y"

#line 787 "EaseaParse.cpp"
			}
		}
		break;
	case 40:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 502 "EaseaParse.y"
nNB_GEN=(int)yyattribute(2 - 2).dValue;
#line 800 "EaseaParse.cpp"
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
#line 504 "EaseaParse.y"
nNB_OPT_IT=(int)yyattribute(2 - 2).dValue;
#line 813 "EaseaParse.cpp"
			}
		}
		break;
	case 42:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 506 "EaseaParse.y"
nTIME_LIMIT=(int)yyattribute(2 - 2).dValue;
#line 826 "EaseaParse.cpp"
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
#line 508 "EaseaParse.y"
fMUT_PROB=(float)yyattribute(2 - 2).dValue;
#line 839 "EaseaParse.cpp"
			}
		}
		break;
	case 44:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 510 "EaseaParse.y"
fXOVER_PROB=(float)yyattribute(2 - 2).dValue;
#line 852 "EaseaParse.cpp"
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
#line 512 "EaseaParse.y"
nPOP_SIZE=(int)yyattribute(2 - 2).dValue;
#line 865 "EaseaParse.cpp"
			}
		}
		break;
	case 46:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 513 "EaseaParse.y"

      strcpy(sSELECTOR, yyattribute(2 - 2).pSymbol->sName);
      strcpy(sSELECTOR_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelector(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME,&EASEALexer);
	break;
      }
    
#line 887 "EaseaParse.cpp"
			}
		}
		break;
	case 47:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 523 "EaseaParse.y"

      sprintf(sSELECTOR, yyattribute(2 - 3).pSymbol->sName);   
      sprintf(sSELECTOR_OPERATOR, yyattribute(2 - 3).pSymbol->sName);   
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelectorArgument(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME,(float)yyattribute(3 - 3).dValue,&EASEALexer);
	break;
      }
    
#line 909 "EaseaParse.cpp"
			}
		}
		break;
	case 48:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 533 "EaseaParse.y"

        sprintf(sRED_PAR, yyattribute(2 - 2).pSymbol->sName);
	sprintf(sRED_PAR_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME,&EASEALexer);
	  break;
	}
    
#line 931 "EaseaParse.cpp"
			}
		}
		break;
	case 49:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 543 "EaseaParse.y"

        sprintf(sRED_PAR, yyattribute(2 - 3).pSymbol->sName);
	sprintf(sRED_PAR_OPERATOR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME,(float)yyattribute(3 - 3).dValue,&EASEALexer);
	  break;
	}
    
#line 953 "EaseaParse.cpp"
			}
		}
		break;
	case 50:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 553 "EaseaParse.y"

	sprintf(sRED_OFF, yyattribute(2 - 2).pSymbol->sName);
	sprintf(sRED_OFF_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
      switch (TARGET) {
      case STD:
      case CUDA:
	pickupSTDSelector(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME,&EASEALexer);
	break;
      }
    
#line 975 "EaseaParse.cpp"
			}
		}
		break;
	case 51:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 563 "EaseaParse.y"

        sprintf(sRED_OFF, yyattribute(2 - 3).pSymbol->sName);
	sprintf(sRED_OFF_OPERATOR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
	case STD:
	case CUDA:
	  pickupSTDSelectorArgument(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME,yyattribute(3 - 3).dValue,&EASEALexer);
       }
#line 995 "EaseaParse.cpp"
			}
		}
		break;
	case 52:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 571 "EaseaParse.y"

        sprintf(sRED_FINAL, yyattribute(2 - 2).pSymbol->sName);
        sprintf(sRED_FINAL_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME,&EASEALexer);
	  break;
       }
#line 1016 "EaseaParse.cpp"
			}
		}
		break;
	case 53:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 580 "EaseaParse.y"

        sprintf(sRED_FINAL, yyattribute(2 - 3).pSymbol->sName);
        sprintf(sRED_FINAL_OPERATOR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME,yyattribute(3 - 3).dValue,&EASEALexer);
	  break;
	}
#line 1037 "EaseaParse.cpp"
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
#line 589 "EaseaParse.y"
nOFF_SIZE=(int)yyattribute(2 - 2).dValue;
#line 1050 "EaseaParse.cpp"
			}
		}
		break;
	case 55:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 590 "EaseaParse.y"
nOFF_SIZE=(int)(yyattribute(2 - 3).dValue*nPOP_SIZE/100);
#line 1063 "EaseaParse.cpp"
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
#line 591 "EaseaParse.y"
fSURV_PAR_SIZE=(float)yyattribute(2 - 2).dValue;
#line 1076 "EaseaParse.cpp"
			}
		}
		break;
	case 57:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 592 "EaseaParse.y"
fSURV_PAR_SIZE=(float)(yyattribute(2 - 3).dValue/100);
#line 1089 "EaseaParse.cpp"
			}
		}
		break;
	case 58:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 593 "EaseaParse.y"
fSURV_OFF_SIZE=(float)yyattribute(2 - 2).dValue;
#line 1102 "EaseaParse.cpp"
			}
		}
		break;
	case 59:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 594 "EaseaParse.y"
fSURV_OFF_SIZE=(float)(yyattribute(2 - 3).dValue/100);
#line 1115 "EaseaParse.cpp"
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
#line 595 "EaseaParse.y"

      if ((!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Maximise")) || (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Maximize"))) nMINIMISE=0;
      else if ((!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Minimise")) || (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Minimize"))) nMINIMISE=1;
      else {
         fprintf(stderr,"\n%s - Error line %d: The evaluator goal default parameter can only take\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         fprintf(stderr,"two values : maximi[sz]e or minimi[sz]e.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
       }
      
#line 1136 "EaseaParse.cpp"
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
#line 604 "EaseaParse.y"

        nELITE=(int)yyattribute(2 - 2).dValue;
        
#line 1151 "EaseaParse.cpp"
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
#line 607 "EaseaParse.y"

        nELITE=(int)yyattribute(2 - 3).dValue*nPOP_SIZE/100;
        
#line 1166 "EaseaParse.cpp"
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
#line 610 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Weak")) bELITISM=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Strong")) bELITISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Elitism must be \"Strong\" or \"Weak\".\nDefault value \"Strong\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bELITISM=1;
       }
#line 1185 "EaseaParse.cpp"
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
#line 617 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bBALDWINISM=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bBALDWINISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Baldwinism must be \"True\" or \"False\".\nDefault value \"True\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bBALDWINISM=1;
       }
#line 1204 "EaseaParse.cpp"
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
#line 625 "EaseaParse.y"

	if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bREMOTE_ISLAND_MODEL=0;
	else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bREMOTE_ISLAND_MODEL=1;
	else {
	  fprintf(stderr,"\n%s - Warning line %d: remote island model must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
	  bREMOTE_ISLAND_MODEL=0;
	}
#line 1223 "EaseaParse.cpp"
			}
		}
		break;
	case 66:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 632 "EaseaParse.y"

        sprintf(sIP_FILE, yyattribute(2 - 4).pSymbol->sName);
	strcat(sIP_FILE,".");
	strcat(sIP_FILE,yyattribute(4 - 4).pSymbol->sName);
	
#line 1240 "EaseaParse.cpp"
			}
		}
		break;
	case 67:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 637 "EaseaParse.y"

	fMIGRATION_PROBABILITY=(float)yyattribute(2 - 2).dValue;
	
#line 1255 "EaseaParse.cpp"
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
#line 641 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bPRINT_STATS=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bPRINT_STATS=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Print stats must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bPRINT_STATS=0;
       }
#line 1274 "EaseaParse.cpp"
			}
		}
		break;
	case 69:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 648 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bPLOT_STATS=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bPLOT_STATS=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate stats must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bPLOT_STATS=0;
       }
#line 1293 "EaseaParse.cpp"
			}
		}
		break;
	case 70:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 655 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bGENERATE_CSV_FILE=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bGENERATE_CSV_FILE=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate csv file must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bGENERATE_CSV_FILE=0;
       }
#line 1312 "EaseaParse.cpp"
			}
		}
		break;
	case 71:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 662 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bGENERATE_GNUPLOT_SCRIPT=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bGENERATE_GNUPLOT_SCRIPT=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate gnuplot script must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bGENERATE_GNUPLOT_SCRIPT=0;
       }
#line 1331 "EaseaParse.cpp"
			}
		}
		break;
	case 72:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 669 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bGENERATE_R_SCRIPT=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bGENERATE_R_SCRIPT=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate R script must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bGENERATE_R_SCRIPT=0;
       }
#line 1350 "EaseaParse.cpp"
			}
		}
		break;
	case 73:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 676 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bSAVE_POPULATION=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bSAVE_POPULATION=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: SavePopulation must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bSAVE_POPULATION=0;
       }
#line 1369 "EaseaParse.cpp"
			}
		}
		break;
	case 74:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 683 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bSTART_FROM_FILE=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bSTART_FROM_FILE=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: StartFromFile must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bSTART_FROM_FILE=0;
       }
#line 1388 "EaseaParse.cpp"
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
#line 690 "EaseaParse.y"
iMAX_INIT_TREE_D = (unsigned)yyattribute(2 - 2).dValue;
#line 1401 "EaseaParse.cpp"
			}
		}
		break;
	case 76:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 691 "EaseaParse.y"
iMIN_INIT_TREE_D = (unsigned)yyattribute(2 - 2).dValue;
#line 1414 "EaseaParse.cpp"
			}
		}
		break;
	case 77:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 692 "EaseaParse.y"
iMAX_TREE_D = (unsigned)yyattribute(2 - 2).dValue;
#line 1427 "EaseaParse.cpp"
			}
		}
		break;
	case 78:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 693 "EaseaParse.y"
iNB_GPU = (unsigned)yyattribute(2 - 2).dValue;
#line 1440 "EaseaParse.cpp"
			}
		}
		break;
	case 79:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 694 "EaseaParse.y"
iPRG_BUF_SIZE = (unsigned)yyattribute(2 - 2).dValue;
#line 1453 "EaseaParse.cpp"
			}
		}
		break;
	case 80:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 697 "EaseaParse.y"
iNO_FITNESS_CASES = (unsigned)yyattribute(2 - 2).dValue;
#line 1466 "EaseaParse.cpp"
			}
		}
		break;
	case 81:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 701 "EaseaParse.y"
 
      if (SymbolTable.find(yyattribute(1 - 3).pSymbol->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 3).pSymbol->sName);
         exit(1);
      }
      (*(YYSTYPE YYFAR*)yyvalptr).dValue = assign(SymbolTable.find(yyattribute(1 - 3).pSymbol->sName), yyattribute(3 - 3).dValue);
    
#line 1485 "EaseaParse.cpp"
			}
		}
		break;
	case 82:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 708 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue + yyattribute(3 - 3).dValue; 
#line 1498 "EaseaParse.cpp"
			}
		}
		break;
	case 83:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 709 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue - yyattribute(3 - 3).dValue; 
#line 1511 "EaseaParse.cpp"
			}
		}
		break;
	case 84:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 710 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue * yyattribute(3 - 3).dValue; 
#line 1524 "EaseaParse.cpp"
			}
		}
		break;
	case 85:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 711 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = divide(yyattribute(1 - 3).dValue, yyattribute(3 - 3).dValue); 
#line 1537 "EaseaParse.cpp"
			}
		}
		break;
	case 86:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 712 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(2 - 3).dValue; 
#line 1550 "EaseaParse.cpp"
			}
		}
		break;
	case 87:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 713 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = -yyattribute(2 - 2).dValue; 
#line 1563 "EaseaParse.cpp"
			}
		}
		break;
	case 88:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 714 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 1).dValue; 
#line 1576 "EaseaParse.cpp"
			}
		}
		break;
	case 89:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 715 "EaseaParse.y"

      if (SymbolTable.find(yyattribute(1 - 1).pSymbol->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 1).pSymbol->sName);
         exit(1);
      }
      (*(YYSTYPE YYFAR*)yyvalptr).dValue = (SymbolTable.find(yyattribute(1 - 1).pSymbol->sName))->dValue;
    
#line 1595 "EaseaParse.cpp"
			}
		}
		break;
	default:
		yyassert(0);
		break;
	}
}
#line 724 "EaseaParse.y"

                       
/////////////////////////////////////////////////////////////////////////////
// main

int main(int argc, char *argv[]){
  int n = YYEXIT_FAILURE;
  int nParamNb=0;
  char *sTemp;
  int i=0;
  
  TARGET=STD;
  bVERBOSE=0;
  sRAW_PROJECT_NAME[0]=0; // used to ask for a filename if no filename is found on the command line.

  while ((++nParamNb) < argc) {
    sTemp=&(argv[nParamNb][0]);
    if ((argv[nParamNb][0]=='-')||(argv[nParamNb][0]=='/')) sTemp=&(argv[nParamNb][1]);
    if (!mystricmp(sTemp,"cuda")){
      TARGET=CUDA;
      TARGET_FLAVOR = CUDA_FLAVOR_SO;
    }
    else if( !mystricmp(sTemp,"cuda_mo") ){
      TARGET=CUDA;
      TARGET_FLAVOR = CUDA_FLAVOR_MO;
    }
    else if( !mystricmp(sTemp,"cuda_gp") ){
      printf("tpl argu : is gp\n");
      TARGET=CUDA;
      TARGET_FLAVOR = CUDA_FLAVOR_GP;
    }
    else if (!mystricmp(sTemp,"std"))  {
      TARGET=STD;
      TARGET_FLAVOR = STD_FLAVOR_SO;
    }
    else if (!mystricmp(sTemp,"std_mo")) {
      TARGET=STD;
      TARGET_FLAVOR = STD_FLAVOR_MO;
    }
    else if (!mystricmp(sTemp,"cmaes"))  {
      TARGET_FLAVOR = CMAES;
    }
    else if (!mystricmp(sTemp,"memetic"))  {
      TARGET_FLAVOR = MEMETIC;
    }

    else if (!mystricmp(sTemp,"v"))  bVERBOSE=true;
    else if (!mystricmp(sTemp,"tl")){
      bLINE_NUM_EZ_FILE=false;
    }
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

  printf("Syntax Error at line : %d\nFor more details during the EASEA compiling, use the \"-v\" option\n",EASEALexer.yylineno);
}


#line 1734 "EaseaParse.cpp"
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
		{ "\'.\'", 46 },
		{ "\'/\'", 47 },
		{ "\'0\'", 48 },
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
		{ "USER_OPTIMISER", 264 },
		{ "MAKEFILE_OPTION", 265 },
		{ "END_OF_FUNCTION", 266 },
		{ "END_METHODS", 267 },
		{ "IDENTIFIER", 268 },
		{ "IDENTIFIER2", 269 },
		{ "BOOL", 270 },
		{ "INT", 271 },
		{ "DOUBLE", 272 },
		{ "FLOAT", 273 },
		{ "GPNODE", 274 },
		{ "CHAR", 275 },
		{ "POINTER", 276 },
		{ "NUMBER", 277 },
		{ "NUMBER2", 278 },
		{ "METHODS", 279 },
		{ "STATIC", 280 },
		{ "NB_GEN", 281 },
		{ "NB_OPT_IT", 282 },
		{ "BALDWINISM", 283 },
		{ "MUT_PROB", 284 },
		{ "XOVER_PROB", 285 },
		{ "POP_SIZE", 286 },
		{ "SELECTOR", 287 },
		{ "RED_PAR", 288 },
		{ "RED_OFF", 289 },
		{ "RED_FINAL", 290 },
		{ "OFFSPRING", 291 },
		{ "SURVPAR", 292 },
		{ "SURVOFF", 293 },
		{ "MINIMAXI", 294 },
		{ "ELITISM", 295 },
		{ "ELITE", 296 },
		{ "REMOTE_ISLAND_MODEL", 297 },
		{ "IP_FILE", 298 },
		{ "MIGRATION_PROBABILITY", 299 },
		{ "PRINT_STATS", 300 },
		{ "PLOT_STATS", 301 },
		{ "GENERATE_CSV_FILE", 302 },
		{ "GENERATE_GNUPLOT_SCRIPT", 303 },
		{ "GENERATE_R_SCRIPT", 304 },
		{ "SAVE_POPULATION", 305 },
		{ "START_FROM_FILE", 306 },
		{ "TIME_LIMIT", 307 },
		{ "MAX_INIT_TREE_D", 308 },
		{ "MIN_INIT_TREE_D", 309 },
		{ "MAX_TREE_D", 312 },
		{ "NB_GPU", 313 },
		{ "PRG_BUF_SIZE", 314 },
		{ "NO_FITNESS_CASES", 315 },
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
		"BaseType: GPNODE",
		"UserType: Symbol",
		"Objects: Object",
		"Objects: Objects \',\' Object",
		"Object: Symbol",
		"Object: \'*\' Symbol",
		"Object: \'0\' Symbol",
		"Object: \'*\' \'*\' Symbol",
		"Object: Symbol \'[\' Expr \']\'",
		"Object: \'*\' Symbol \'[\' Expr \']\'",
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
		"$$12:",
		"StandardFunctionAnalysis: USER_OPTIMISER $$12 END_OF_FUNCTION",
		"StandardFunctionAnalysis: MAKEFILE_OPTION END_OF_FUNCTION",
		"StandardFunctionAnalysis: MAKEFILE_OPTION",
		"RunParameters: Parameter",
		"RunParameters: RunParameters Parameter",
		"Parameter: NB_GEN NUMBER2",
		"Parameter: NB_OPT_IT NUMBER2",
		"Parameter: TIME_LIMIT NUMBER2",
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
		"Parameter: MINIMAXI IDENTIFIER2",
		"Parameter: ELITE NUMBER2",
		"Parameter: ELITE NUMBER2 \'%\'",
		"Parameter: ELITISM IDENTIFIER2",
		"Parameter: BALDWINISM IDENTIFIER2",
		"Parameter: REMOTE_ISLAND_MODEL IDENTIFIER2",
		"Parameter: IP_FILE IDENTIFIER2 \'.\' IDENTIFIER2",
		"Parameter: MIGRATION_PROBABILITY NUMBER2",
		"Parameter: PRINT_STATS IDENTIFIER2",
		"Parameter: PLOT_STATS IDENTIFIER2",
		"Parameter: GENERATE_CSV_FILE IDENTIFIER2",
		"Parameter: GENERATE_GNUPLOT_SCRIPT IDENTIFIER2",
		"Parameter: GENERATE_R_SCRIPT IDENTIFIER2",
		"Parameter: SAVE_POPULATION IDENTIFIER2",
		"Parameter: START_FROM_FILE IDENTIFIER2",
		"Parameter: MAX_INIT_TREE_D NUMBER2",
		"Parameter: MIN_INIT_TREE_D NUMBER2",
		"Parameter: MAX_TREE_D NUMBER2",
		"Parameter: NB_GPU NUMBER2",
		"Parameter: PRG_BUF_SIZE NUMBER2",
		"Parameter: NO_FITNESS_CASES NUMBER2",
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
		{ 18, 1, -1 },
		{ 19, 1, 16 },
		{ 20, 1, -1 },
		{ 20, 3, -1 },
		{ 21, 1, 17 },
		{ 21, 2, 18 },
		{ 21, 2, 19 },
		{ 21, 3, 20 },
		{ 21, 4, 21 },
		{ 21, 5, 22 },
		{ 22, 1, -1 },
		{ 22, 2, -1 },
		{ 23, 1, 23 },
		{ 25, 0, 24 },
		{ 24, 5, 25 },
		{ 26, 1, -1 },
		{ 26, 2, -1 },
		{ 27, 1, 26 },
		{ 28, 1, 27 },
		{ 29, 1, -1 },
		{ 29, 2, -1 },
		{ 31, 0, 28 },
		{ 30, 3, 29 },
		{ 32, 0, 30 },
		{ 30, 3, 31 },
		{ 33, 0, 32 },
		{ 30, 3, 33 },
		{ 34, 0, 34 },
		{ 30, 3, 35 },
		{ 35, 0, 36 },
		{ 30, 3, 37 },
		{ 30, 2, 38 },
		{ 30, 1, 39 },
		{ 36, 1, -1 },
		{ 36, 2, -1 },
		{ 37, 2, 40 },
		{ 37, 2, 41 },
		{ 37, 2, 42 },
		{ 37, 2, 43 },
		{ 37, 2, 44 },
		{ 37, 2, 45 },
		{ 37, 2, 46 },
		{ 37, 3, 47 },
		{ 37, 2, 48 },
		{ 37, 3, 49 },
		{ 37, 2, 50 },
		{ 37, 3, 51 },
		{ 37, 2, 52 },
		{ 37, 3, 53 },
		{ 37, 2, 54 },
		{ 37, 3, 55 },
		{ 37, 2, 56 },
		{ 37, 3, 57 },
		{ 37, 2, 58 },
		{ 37, 3, 59 },
		{ 37, 2, 60 },
		{ 37, 2, 61 },
		{ 37, 3, 62 },
		{ 37, 2, 63 },
		{ 37, 2, 64 },
		{ 37, 2, 65 },
		{ 37, 4, 66 },
		{ 37, 2, 67 },
		{ 37, 2, 68 },
		{ 37, 2, 69 },
		{ 37, 2, 70 },
		{ 37, 2, 71 },
		{ 37, 2, 72 },
		{ 37, 2, 73 },
		{ 37, 2, 74 },
		{ 37, 2, 75 },
		{ 37, 2, 76 },
		{ 37, 2, 77 },
		{ 37, 2, 78 },
		{ 37, 2, 79 },
		{ 37, 2, 80 },
		{ 38, 3, 81 },
		{ 38, 3, 82 },
		{ 38, 3, 83 },
		{ 38, 3, 84 },
		{ 38, 3, 85 },
		{ 38, 3, 86 },
		{ 38, 2, 87 },
		{ 38, 1, 88 },
		{ 38, 1, 89 }
	};
	yyreduction = reduction;

	static const yytokenaction_t YYNEARFAR YYBASED_CODE tokenaction[] = {
		{ 35, YYAT_SHIFT, 70 },
		{ 181, YYAT_SHIFT, 162 },
		{ 169, YYAT_SHIFT, 183 },
		{ 172, YYAT_SHIFT, 185 },
		{ 139, YYAT_SHIFT, 142 },
		{ 166, YYAT_SHIFT, 182 },
		{ 181, YYAT_SHIFT, 163 },
		{ 174, YYAT_ERROR, 0 },
		{ 108, YYAT_SHIFT, 122 },
		{ 108, YYAT_SHIFT, 123 },
		{ 108, YYAT_SHIFT, 124 },
		{ 108, YYAT_SHIFT, 125 },
		{ 108, YYAT_SHIFT, 126 },
		{ 108, YYAT_SHIFT, 127 },
		{ 108, YYAT_SHIFT, 128 },
		{ 112, YYAT_SHIFT, 96 },
		{ 112, YYAT_SHIFT, 97 },
		{ 112, YYAT_SHIFT, 98 },
		{ 112, YYAT_SHIFT, 99 },
		{ 112, YYAT_SHIFT, 100 },
		{ 112, YYAT_SHIFT, 101 },
		{ 147, YYAT_SHIFT, 156 },
		{ 147, YYAT_SHIFT, 157 },
		{ 35, YYAT_SHIFT, 1 },
		{ 35, YYAT_SHIFT, 2 },
		{ 35, YYAT_SHIFT, 3 },
		{ 35, YYAT_SHIFT, 4 },
		{ 35, YYAT_SHIFT, 5 },
		{ 35, YYAT_SHIFT, 6 },
		{ 35, YYAT_SHIFT, 7 },
		{ 35, YYAT_SHIFT, 8 },
		{ 35, YYAT_SHIFT, 9 },
		{ 35, YYAT_SHIFT, 10 },
		{ 35, YYAT_SHIFT, 11 },
		{ 35, YYAT_SHIFT, 12 },
		{ 35, YYAT_SHIFT, 13 },
		{ 35, YYAT_SHIFT, 14 },
		{ 35, YYAT_SHIFT, 15 },
		{ 35, YYAT_SHIFT, 16 },
		{ 35, YYAT_SHIFT, 17 },
		{ 35, YYAT_SHIFT, 18 },
		{ 35, YYAT_SHIFT, 19 },
		{ 35, YYAT_SHIFT, 20 },
		{ 35, YYAT_SHIFT, 21 },
		{ 35, YYAT_SHIFT, 22 },
		{ 35, YYAT_SHIFT, 23 },
		{ 35, YYAT_SHIFT, 24 },
		{ 35, YYAT_SHIFT, 25 },
		{ 35, YYAT_SHIFT, 26 },
		{ 35, YYAT_SHIFT, 27 },
		{ 35, YYAT_SHIFT, 28 },
		{ 35, YYAT_SHIFT, 29 },
		{ 0, YYAT_ERROR, 0 },
		{ 0, YYAT_ERROR, 0 },
		{ 35, YYAT_SHIFT, 30 },
		{ 35, YYAT_SHIFT, 31 },
		{ 35, YYAT_SHIFT, 32 },
		{ 35, YYAT_SHIFT, 33 },
		{ 150, YYAT_SHIFT, 155 },
		{ 174, YYAT_SHIFT, 187 },
		{ 102, YYAT_SHIFT, 96 },
		{ 102, YYAT_SHIFT, 97 },
		{ 102, YYAT_SHIFT, 98 },
		{ 102, YYAT_SHIFT, 99 },
		{ 102, YYAT_SHIFT, 100 },
		{ 102, YYAT_SHIFT, 101 },
		{ 95, YYAT_SHIFT, 96 },
		{ 95, YYAT_SHIFT, 97 },
		{ 95, YYAT_SHIFT, 98 },
		{ 95, YYAT_SHIFT, 99 },
		{ 95, YYAT_SHIFT, 100 },
		{ 95, YYAT_SHIFT, 101 },
		{ 150, YYAT_SHIFT, 158 },
		{ 150, YYAT_SHIFT, 159 },
		{ 175, YYAT_SHIFT, 188 },
		{ 175, YYAT_SHIFT, 178 },
		{ 175, YYAT_SHIFT, 179 },
		{ 155, YYAT_SHIFT, 143 },
		{ 175, YYAT_SHIFT, 180 },
		{ 164, YYAT_SHIFT, 177 },
		{ 175, YYAT_SHIFT, 181 },
		{ 189, YYAT_SHIFT, 178 },
		{ 189, YYAT_SHIFT, 179 },
		{ 155, YYAT_SHIFT, 144 },
		{ 189, YYAT_SHIFT, 180 },
		{ 192, YYAT_SHIFT, 178 },
		{ 189, YYAT_SHIFT, 181 },
		{ 191, YYAT_SHIFT, 178 },
		{ 158, YYAT_ERROR, 0 },
		{ 156, YYAT_ERROR, 0 },
		{ 192, YYAT_SHIFT, 181 },
		{ 152, YYAT_SHIFT, 161 },
		{ 191, YYAT_SHIFT, 181 },
		{ 145, YYAT_SHIFT, 154 },
		{ 143, YYAT_SHIFT, 151 },
		{ 120, YYAT_REDUCE, 22 },
		{ 117, YYAT_SHIFT, 138 },
		{ 116, YYAT_SHIFT, 137 },
		{ 115, YYAT_SHIFT, 136 },
		{ 114, YYAT_SHIFT, 135 },
		{ 113, YYAT_SHIFT, 134 },
		{ 109, YYAT_SHIFT, 132 },
		{ 106, YYAT_SHIFT, 121 },
		{ 104, YYAT_SHIFT, 120 },
		{ 101, YYAT_SHIFT, 118 },
		{ 92, YYAT_SHIFT, 90 },
		{ 86, YYAT_SHIFT, 94 },
		{ 84, YYAT_SHIFT, 89 },
		{ 73, YYAT_SHIFT, 71 },
		{ 70, YYAT_REDUCE, 6 },
		{ 54, YYAT_SHIFT, 84 },
		{ 52, YYAT_SHIFT, 83 },
		{ 49, YYAT_SHIFT, 82 },
		{ 48, YYAT_SHIFT, 81 },
		{ 47, YYAT_SHIFT, 80 },
		{ 46, YYAT_SHIFT, 79 },
		{ 45, YYAT_SHIFT, 78 },
		{ 44, YYAT_SHIFT, 77 },
		{ 43, YYAT_SHIFT, 76 },
		{ 34, YYAT_ACCEPT, 0 },
		{ 33, YYAT_SHIFT, 69 },
		{ 32, YYAT_SHIFT, 68 },
		{ 31, YYAT_SHIFT, 67 },
		{ 30, YYAT_SHIFT, 66 },
		{ 29, YYAT_SHIFT, 65 },
		{ 28, YYAT_SHIFT, 64 },
		{ 27, YYAT_SHIFT, 63 },
		{ 26, YYAT_SHIFT, 62 },
		{ 25, YYAT_SHIFT, 61 },
		{ 24, YYAT_SHIFT, 60 },
		{ 23, YYAT_SHIFT, 59 },
		{ 22, YYAT_SHIFT, 58 },
		{ 21, YYAT_SHIFT, 57 },
		{ 20, YYAT_SHIFT, 56 },
		{ 19, YYAT_SHIFT, 55 },
		{ 18, YYAT_SHIFT, 54 },
		{ 17, YYAT_SHIFT, 53 },
		{ 16, YYAT_SHIFT, 52 },
		{ 15, YYAT_SHIFT, 51 },
		{ 14, YYAT_SHIFT, 50 },
		{ 13, YYAT_SHIFT, 49 },
		{ 12, YYAT_SHIFT, 48 },
		{ 11, YYAT_SHIFT, 47 },
		{ 10, YYAT_SHIFT, 46 },
		{ 9, YYAT_SHIFT, 45 },
		{ 8, YYAT_SHIFT, 44 },
		{ 7, YYAT_SHIFT, 43 },
		{ 6, YYAT_SHIFT, 42 },
		{ 5, YYAT_SHIFT, 41 },
		{ 4, YYAT_SHIFT, 40 },
		{ 3, YYAT_SHIFT, 39 },
		{ 2, YYAT_SHIFT, 38 },
		{ 1, YYAT_SHIFT, 37 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 139, YYAT_SHIFT, 106 },
		{ 139, YYAT_SHIFT, 107 },
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
		{ 172, YYAT_SHIFT, 90 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 169, YYAT_SHIFT, 168 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 181, YYAT_SHIFT, 164 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 181, YYAT_SHIFT, 165 }
	};
	yytokenaction = tokenaction;
	yytokenaction_size = 239;

	static const yystateaction_t YYNEARFAR YYBASED_CODE stateaction[] = {
		{ -206, 1, YYAT_DEFAULT, 35 },
		{ -126, 1, YYAT_DEFAULT, 33 },
		{ -127, 1, YYAT_DEFAULT, 33 },
		{ -119, 1, YYAT_DEFAULT, 84 },
		{ -129, 1, YYAT_DEFAULT, 33 },
		{ -130, 1, YYAT_DEFAULT, 33 },
		{ -131, 1, YYAT_DEFAULT, 33 },
		{ -123, 1, YYAT_DEFAULT, 84 },
		{ -124, 1, YYAT_DEFAULT, 84 },
		{ -125, 1, YYAT_DEFAULT, 84 },
		{ -126, 1, YYAT_DEFAULT, 84 },
		{ -136, 1, YYAT_DEFAULT, 33 },
		{ -137, 1, YYAT_DEFAULT, 33 },
		{ -138, 1, YYAT_DEFAULT, 33 },
		{ -130, 1, YYAT_DEFAULT, 84 },
		{ -131, 1, YYAT_DEFAULT, 84 },
		{ -141, 1, YYAT_DEFAULT, 33 },
		{ -133, 1, YYAT_DEFAULT, 84 },
		{ -134, 1, YYAT_DEFAULT, 84 },
		{ -144, 1, YYAT_DEFAULT, 33 },
		{ -136, 1, YYAT_DEFAULT, 84 },
		{ -137, 1, YYAT_DEFAULT, 84 },
		{ -138, 1, YYAT_DEFAULT, 84 },
		{ -139, 1, YYAT_DEFAULT, 84 },
		{ -140, 1, YYAT_DEFAULT, 84 },
		{ -141, 1, YYAT_DEFAULT, 84 },
		{ -142, 1, YYAT_DEFAULT, 84 },
		{ -152, 1, YYAT_DEFAULT, 33 },
		{ -153, 1, YYAT_DEFAULT, 33 },
		{ -154, 1, YYAT_DEFAULT, 33 },
		{ -155, 1, YYAT_DEFAULT, 33 },
		{ -156, 1, YYAT_DEFAULT, 33 },
		{ -157, 1, YYAT_DEFAULT, 33 },
		{ -158, 1, YYAT_ERROR, 0 },
		{ 119, 1, YYAT_ERROR, 0 },
		{ -258, 1, YYAT_DEFAULT, 73 },
		{ 0, 0, YYAT_REDUCE, 66 },
		{ 0, 0, YYAT_REDUCE, 68 },
		{ 0, 0, YYAT_REDUCE, 69 },
		{ 0, 0, YYAT_REDUCE, 92 },
		{ 0, 0, YYAT_REDUCE, 71 },
		{ 0, 0, YYAT_REDUCE, 72 },
		{ 0, 0, YYAT_REDUCE, 73 },
		{ -160, 1, YYAT_REDUCE, 74 },
		{ -161, 1, YYAT_REDUCE, 76 },
		{ -162, 1, YYAT_REDUCE, 78 },
		{ -163, 1, YYAT_REDUCE, 80 },
		{ 77, 1, YYAT_REDUCE, 82 },
		{ 76, 1, YYAT_REDUCE, 84 },
		{ 75, 1, YYAT_REDUCE, 86 },
		{ 0, 0, YYAT_REDUCE, 88 },
		{ 0, 0, YYAT_REDUCE, 91 },
		{ 74, 1, YYAT_REDUCE, 89 },
		{ 0, 0, YYAT_REDUCE, 93 },
		{ 64, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 95 },
		{ 0, 0, YYAT_REDUCE, 96 },
		{ 0, 0, YYAT_REDUCE, 97 },
		{ 0, 0, YYAT_REDUCE, 98 },
		{ 0, 0, YYAT_REDUCE, 99 },
		{ 0, 0, YYAT_REDUCE, 100 },
		{ 0, 0, YYAT_REDUCE, 101 },
		{ 0, 0, YYAT_REDUCE, 102 },
		{ 0, 0, YYAT_REDUCE, 70 },
		{ 0, 0, YYAT_REDUCE, 103 },
		{ 0, 0, YYAT_REDUCE, 104 },
		{ 0, 0, YYAT_REDUCE, 105 },
		{ 0, 0, YYAT_REDUCE, 106 },
		{ 0, 0, YYAT_REDUCE, 107 },
		{ 0, 0, YYAT_REDUCE, 108 },
		{ -159, 1, YYAT_REDUCE, 8 },
		{ 0, 0, YYAT_REDUCE, 46 },
		{ 0, 0, YYAT_REDUCE, 1 },
		{ -151, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 4 },
		{ 0, 0, YYAT_REDUCE, 67 },
		{ 0, 0, YYAT_REDUCE, 75 },
		{ 0, 0, YYAT_REDUCE, 77 },
		{ 0, 0, YYAT_REDUCE, 79 },
		{ 0, 0, YYAT_REDUCE, 81 },
		{ 0, 0, YYAT_REDUCE, 83 },
		{ 0, 0, YYAT_REDUCE, 85 },
		{ 0, 0, YYAT_REDUCE, 87 },
		{ 0, 0, YYAT_REDUCE, 90 },
		{ -162, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_DEFAULT, 158 },
		{ -17, 1, YYAT_DEFAULT, 104 },
		{ 0, 0, YYAT_REDUCE, 2 },
		{ 0, 0, YYAT_DEFAULT, 95 },
		{ 0, 0, YYAT_REDUCE, 94 },
		{ 0, 0, YYAT_REDUCE, 51 },
		{ 0, 0, YYAT_REDUCE, 11 },
		{ -163, 1, YYAT_REDUCE, 7 },
		{ 0, 0, YYAT_REDUCE, 9 },
		{ 0, 0, YYAT_DEFAULT, 120 },
		{ -194, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 54 },
		{ 0, 0, YYAT_REDUCE, 56 },
		{ 0, 0, YYAT_REDUCE, 58 },
		{ 0, 0, YYAT_REDUCE, 60 },
		{ 0, 0, YYAT_REDUCE, 62 },
		{ -162, 1, YYAT_REDUCE, 65 },
		{ -200, 1, YYAT_REDUCE, 5 },
		{ 0, 0, YYAT_REDUCE, 52 },
		{ -20, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 10 },
		{ -165, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 21 },
		{ -262, 1, YYAT_DEFAULT, 158 },
		{ -24, 1, YYAT_DEFAULT, 139 },
		{ 0, 0, YYAT_REDUCE, 13 },
		{ 0, 0, YYAT_REDUCE, 19 },
		{ -245, 1, YYAT_REDUCE, 3 },
		{ -166, 1, YYAT_DEFAULT, 117 },
		{ -167, 1, YYAT_DEFAULT, 117 },
		{ -168, 1, YYAT_DEFAULT, 117 },
		{ -169, 1, YYAT_DEFAULT, 117 },
		{ -170, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 64 },
		{ 0, 0, YYAT_REDUCE, 53 },
		{ -30, 1, YYAT_DEFAULT, 139 },
		{ 0, 0, YYAT_REDUCE, 20 },
		{ 0, 0, YYAT_REDUCE, 27 },
		{ 0, 0, YYAT_REDUCE, 28 },
		{ 0, 0, YYAT_REDUCE, 29 },
		{ 0, 0, YYAT_REDUCE, 30 },
		{ 0, 0, YYAT_REDUCE, 33 },
		{ 0, 0, YYAT_REDUCE, 31 },
		{ 0, 0, YYAT_REDUCE, 32 },
		{ 0, 0, YYAT_REDUCE, 15 },
		{ 0, 0, YYAT_REDUCE, 17 },
		{ 0, 0, YYAT_REDUCE, 34 },
		{ 0, 0, YYAT_REDUCE, 47 },
		{ 0, 0, YYAT_REDUCE, 14 },
		{ 0, 0, YYAT_REDUCE, 55 },
		{ 0, 0, YYAT_REDUCE, 57 },
		{ 0, 0, YYAT_REDUCE, 59 },
		{ 0, 0, YYAT_REDUCE, 61 },
		{ 0, 0, YYAT_REDUCE, 63 },
		{ -121, 1, YYAT_REDUCE, 22 },
		{ 0, 0, YYAT_DEFAULT, 155 },
		{ 0, 0, YYAT_DEFAULT, 155 },
		{ 0, 0, YYAT_REDUCE, 12 },
		{ 52, 1, YYAT_DEFAULT, 158 },
		{ 0, 0, YYAT_DEFAULT, 158 },
		{ 2, 1, YYAT_REDUCE, 37 },
		{ 0, 0, YYAT_REDUCE, 16 },
		{ -37, 1, YYAT_DEFAULT, 150 },
		{ 0, 0, YYAT_REDUCE, 35 },
		{ 0, 0, YYAT_REDUCE, 18 },
		{ 14, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_DEFAULT, 158 },
		{ 0, 1, YYAT_REDUCE, 38 },
		{ 0, 0, YYAT_REDUCE, 39 },
		{ 0, 0, YYAT_DEFAULT, 181 },
		{ 35, 1, YYAT_DEFAULT, 158 },
		{ 30, 1, YYAT_DEFAULT, 169 },
		{ 0, 0, YYAT_REDUCE, 23 },
		{ 29, 1, YYAT_DEFAULT, 172 },
		{ 0, 0, YYAT_REDUCE, 25 },
		{ 0, 0, YYAT_REDUCE, 40 },
		{ 0, 0, YYAT_DEFAULT, 181 },
		{ 0, 0, YYAT_DEFAULT, 181 },
		{ 0, 0, YYAT_DEFAULT, 181 },
		{ 18, 1, YYAT_REDUCE, 117 },
		{ 0, 0, YYAT_REDUCE, 116 },
		{ -88, 1, YYAT_DEFAULT, 174 },
		{ 0, 0, YYAT_REDUCE, 36 },
		{ 0, 0, YYAT_REDUCE, 45 },
		{ -57, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 43 },
		{ 0, 0, YYAT_REDUCE, 50 },
		{ -56, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 48 },
		{ -34, 1, YYAT_DEFAULT, 175 },
		{ 33, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 115 },
		{ 0, 0, YYAT_DEFAULT, 181 },
		{ 0, 0, YYAT_DEFAULT, 181 },
		{ 0, 0, YYAT_DEFAULT, 181 },
		{ 0, 0, YYAT_DEFAULT, 181 },
		{ -39, 1, YYAT_DEFAULT, 158 },
		{ 0, 0, YYAT_REDUCE, 41 },
		{ 0, 0, YYAT_REDUCE, 24 },
		{ 0, 0, YYAT_REDUCE, 44 },
		{ 0, 0, YYAT_REDUCE, 26 },
		{ 0, 0, YYAT_REDUCE, 49 },
		{ 0, 0, YYAT_REDUCE, 42 },
		{ 0, 0, YYAT_REDUCE, 114 },
		{ 39, 1, YYAT_REDUCE, 109 },
		{ 0, 0, YYAT_REDUCE, 112 },
		{ 45, 1, YYAT_REDUCE, 110 },
		{ 43, 1, YYAT_REDUCE, 111 },
		{ 0, 0, YYAT_REDUCE, 113 }
	};
	yystateaction = stateaction;

	static const yynontermgoto_t YYNEARFAR YYBASED_CODE nontermgoto[] = {
		{ 35, 72 },
		{ 0, 34 },
		{ 155, 167 },
		{ 35, 73 },
		{ 92, 105 },
		{ 108, 129 },
		{ 108, 130 },
		{ 140, 146 },
		{ 181, 193 },
		{ 155, 145 },
		{ 141, 149 },
		{ 140, 147 },
		{ 140, 148 },
		{ 141, 150 },
		{ 141, 148 },
		{ 108, 131 },
		{ 139, 133 },
		{ 172, 186 },
		{ 172, 171 },
		{ 139, 111 },
		{ 139, 108 },
		{ 180, 192 },
		{ 35, 74 },
		{ 179, 191 },
		{ 92, 91 },
		{ 158, 172 },
		{ 158, 173 },
		{ 156, 169 },
		{ 156, 170 },
		{ 120, 139 },
		{ 120, 110 },
		{ 95, 112 },
		{ 95, 103 },
		{ 85, 92 },
		{ 85, 93 },
		{ 35, 75 },
		{ 0, 35 },
		{ 0, 36 },
		{ 178, 190 },
		{ 177, 189 },
		{ 169, 184 },
		{ 163, 176 },
		{ 162, 175 },
		{ 161, 174 },
		{ 154, 166 },
		{ 151, 160 },
		{ 144, 153 },
		{ 143, 152 },
		{ 130, 141 },
		{ 129, 140 },
		{ 112, 119 },
		{ 100, 117 },
		{ 99, 116 },
		{ 98, 115 },
		{ 97, 114 },
		{ 96, 113 },
		{ 94, 109 },
		{ 91, 104 },
		{ 88, 102 },
		{ 87, 95 },
		{ 74, 88 },
		{ 73, 87 },
		{ 71, 86 },
		{ 70, 85 }
	};
	yynontermgoto = nontermgoto;
	yynontermgoto_size = 64;

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
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 57, -1 },
		{ 37, -1 },
		{ 0, -1 },
		{ 37, -1 },
		{ 56, -1 },
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
		{ 26, 92 },
		{ 0, -1 },
		{ 56, -1 },
		{ 29, 95 },
		{ 0, -1 },
		{ 0, -1 },
		{ 48, -1 },
		{ -4, -1 },
		{ 0, -1 },
		{ 46, 120 },
		{ 2, -1 },
		{ 24, -1 },
		{ 22, -1 },
		{ 20, -1 },
		{ 18, -1 },
		{ 16, -1 },
		{ 0, -1 },
		{ 0, 112 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ -13, -1 },
		{ 0, 139 },
		{ 0, -1 },
		{ 0, -1 },
		{ 20, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 19, 139 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 37, -1 },
		{ 35, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 5, -1 },
		{ -9, 155 },
		{ -7, 155 },
		{ 0, -1 },
		{ 19, -1 },
		{ 18, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 17, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 6, -1 },
		{ -19, -1 },
		{ 5, -1 },
		{ 0, -1 },
		{ -1, 172 },
		{ 0, -1 },
		{ 0, -1 },
		{ 5, -1 },
		{ 4, -1 },
		{ 3, -1 },
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
		{ 0, -1 },
		{ 1, -1 },
		{ 0, -1 },
		{ -15, -1 },
		{ -17, -1 },
		{ -30, -1 },
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
		{ 0, -1 }
	};
	yystategoto = stategoto;

	yydestructorptr = NULL;

	yytokendestptr = NULL;
	yytokendest_size = 0;
	yytokendestbase = 0;
}
