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
char sEXPID[128]; //experiment ID
char sWORKING_PATH[512];  // working path
int nPOP_SIZE, nOFF_SIZE;
float fSURV_PAR_SIZE=-1.0, fSURV_OFF_SIZE=-1.0;
char *nGENOME_NAME;
int nPROBLEM_DIM;
int nNB_GEN=0;
int nNB_OPT_IT=0;
int nTIME_LIMIT=0;
int nSERVER_PORT=0;
float fMUT_PROB;
float fXOVER_PROB;
FILE *fpOutputFile, *fpTemplateFile, *fpGenomeFile;//, *fpExplodedGenomeFile;

 unsigned iMAX_INIT_TREE_D,iMIN_INIT_TREE_D,iMAX_TREE_D,iNB_GPU,iPRG_BUF_SIZE,iMAX_TREE_DEPTH,iMAX_XOVER_DEPTH,iNO_FITNESS_CASES;

#line 74 "EaseaParse.cpp"
// repeated because of possible precompiled header
#include <cyacc.h>

#include "EaseaParse.h"

/////////////////////////////////////////////////////////////////////////////
// constructor

YYPARSENAME::YYPARSENAME()
{
	yytables();
#line 190 "EaseaParse.y"

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

#line 120 "EaseaParse.cpp"
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
#line 230 "EaseaParse.y"

        if (bVERBOSE){ printf("                    _______________________________________\n");
        printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);}
      
#line 170 "EaseaParse.cpp"
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
#line 235 "EaseaParse.y"

        if (bVERBOSE) printf("                    _______________________________________\n");
        if (bVERBOSE) printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);
      
#line 186 "EaseaParse.cpp"
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
#line 243 "EaseaParse.y"

    if (bVERBOSE) printf("Declaration of user classes :\n\n");
#line 200 "EaseaParse.cpp"
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
#line 246 "EaseaParse.y"

      if (bVERBOSE) printf("No user class declaration found other than GenomeClass.\n");
#line 214 "EaseaParse.cpp"
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
#line 256 "EaseaParse.y"

      pCURRENT_CLASS=SymbolTable.insert(yyattribute(1 - 1).pSymbol);  
      pCURRENT_CLASS->pSymbolList=new CLList<CSymbol *>();
      yyattribute(1 - 1).pSymbol->ObjectType=oUserClass;
      //DEBUG_PRT("Yacc Symbol declaration %s %d",$1->sName,$1->nSize);
      pCLASSES[nClasses_nb++] = yyattribute(1 - 1).pSymbol;
    
#line 233 "EaseaParse.cpp"
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
#line 263 "EaseaParse.y"

      if (bVERBOSE) printf("Class %s declared for %d bytes.\n\n",yyattribute(1 - 5).pSymbol->sName,yyattribute(1 - 5).pSymbol->nSize);
      //DEBUG_PRT("Yacc variable declaration %s %d",$1->sName,$1->nSize);
    
#line 249 "EaseaParse.cpp"
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
#line 276 "EaseaParse.y"
pCURRENT_TYPE=yyattribute(2 - 2).pSymbol; pCURRENT_TYPE->ObjectQualifier=yyattribute(1 - 2).ObjectQualifier;
#line 262 "EaseaParse.cpp"
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
#line 276 "EaseaParse.y"

#line 275 "EaseaParse.cpp"
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
#line 277 "EaseaParse.y"
pCURRENT_TYPE=yyattribute(2 - 2).pSymbol; pCURRENT_TYPE->ObjectQualifier=yyattribute(1 - 2).ObjectQualifier;
#line 288 "EaseaParse.cpp"
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
#line 277 "EaseaParse.y"

#line 301 "EaseaParse.cpp"
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
#line 282 "EaseaParse.y"

    pCURRENT_CLASS->sString = new char[strlen(yyattribute(2 - 2).szString) + 1];
    strcpy(pCURRENT_CLASS->sString, yyattribute(2 - 2).szString);      
    if (bVERBOSE) printf("\n    The following methods have been declared:\n\n%s\n\n",pCURRENT_CLASS->sString);
    
#line 318 "EaseaParse.cpp"
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
#line 290 "EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).ObjectQualifier=1;
#line 331 "EaseaParse.cpp"
			}
		}
		break;
	case 12:
		{
#line 291 "EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).ObjectQualifier=0;
#line 339 "EaseaParse.cpp"
		}
		break;
	case 13:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 296 "EaseaParse.y"

#line 351 "EaseaParse.cpp"
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
#line 300 "EaseaParse.y"

#line 364 "EaseaParse.cpp"
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
#line 301 "EaseaParse.y"

#line 377 "EaseaParse.cpp"
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
#line 315 "EaseaParse.y"
  
      CSymbol *pSym=SymbolTable.find(yyattribute(1 - 1).pSymbol->sName);
      if (pSym==NULL) {
        fprintf(stderr,"\n%s - Error line %d: Class \"%s\" was not defined.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 1).pSymbol->sName);
        fprintf(stderr,"Only base types (bool, int, float, double, char) or new user classes defined\nwithin the \"User classes\" sections are allowed.\n");
        exit(1);
      }       
      else (*(YYSTYPE YYFAR*)yyvalptr).pSymbol=pSym;
    
#line 398 "EaseaParse.cpp"
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
#line 336 "EaseaParse.y"

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
    
#line 422 "EaseaParse.cpp"
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
#line 348 "EaseaParse.y"

      yyattribute(2 - 2).pSymbol->nSize=sizeof (char *);
      yyattribute(2 - 2).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(2 - 2).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(2 - 2).pSymbol->ObjectType=oPointer;
      yyattribute(2 - 2).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(2 - 2).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(2 - 2).pSymbol));
      if (bVERBOSE) printf("    %s pointer declared (%d bytes)\n",yyattribute(2 - 2).pSymbol->sName,yyattribute(2 - 2).pSymbol->nSize);
    
#line 444 "EaseaParse.cpp"
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
#line 358 "EaseaParse.y"

      yyattribute(2 - 2).pSymbol->nSize=sizeof (char *);
      yyattribute(2 - 2).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(2 - 2).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(2 - 2).pSymbol->ObjectType=oPointer;
      yyattribute(2 - 2).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(2 - 2).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(2 - 2).pSymbol));
      if (bVERBOSE) printf("    %s NULL pointer declared (%d bytes)\n",yyattribute(2 - 2).pSymbol->sName,yyattribute(2 - 2).pSymbol->nSize);
    
#line 466 "EaseaParse.cpp"
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
#line 368 "EaseaParse.y"

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
    
#line 490 "EaseaParse.cpp"
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
#line 381 "EaseaParse.y"

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
    
#line 516 "EaseaParse.cpp"
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
#line 395 "EaseaParse.y"


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
    
#line 548 "EaseaParse.cpp"
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
#line 423 "EaseaParse.y"

#line 561 "EaseaParse.cpp"
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
#line 436 "EaseaParse.y"

    ////DEBUG_PRT("Yacc genome decl %s",$1.pSymbol->sName);
      if (bVERBOSE) printf ("\nGenome declaration analysis :\n\n");
      pGENOME=new CSymbol("Genome");
      pCURRENT_CLASS=SymbolTable.insert(pGENOME);  
      pGENOME->pSymbolList=new CLList<CSymbol *>();
      pGENOME->ObjectType=oUserClass;
      pGENOME->ObjectQualifier=0;
      pGENOME->sString=NULL;
    
#line 583 "EaseaParse.cpp"
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
#line 446 "EaseaParse.y"

#line 596 "EaseaParse.cpp"
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
#line 463 "EaseaParse.y"

#line 609 "EaseaParse.cpp"
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
#line 467 "EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).pSymbol=yyattribute(1 - 1).pSymbol;
#line 622 "EaseaParse.cpp"
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
#line 476 "EaseaParse.y"
         
      if (bVERBOSE) printf("Inserting genome initialiser (taken from .ez file).\n");
    
#line 637 "EaseaParse.cpp"
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
#line 479 "EaseaParse.y"

#line 650 "EaseaParse.cpp"
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
#line 480 "EaseaParse.y"

      if (bVERBOSE) printf("Inserting user genome crossover (taken from .ez file).\n");
    
#line 665 "EaseaParse.cpp"
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
#line 483 "EaseaParse.y"

#line 678 "EaseaParse.cpp"
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
#line 484 "EaseaParse.y"

      if (bVERBOSE) printf("Inserting user genome mutator (taken from .ez file).\n");
    
#line 693 "EaseaParse.cpp"
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
#line 487 "EaseaParse.y"

#line 706 "EaseaParse.cpp"
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
#line 488 "EaseaParse.y"
 
      if (bVERBOSE) printf("Inserting user genome evaluator (taken from .ez file).\n");
    
#line 721 "EaseaParse.cpp"
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
#line 491 "EaseaParse.y"

#line 734 "EaseaParse.cpp"
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
#line 492 "EaseaParse.y"
 
      if (bVERBOSE) printf("Inserting user genome optimiser (taken from .ez file).\n");
    
#line 749 "EaseaParse.cpp"
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
#line 495 "EaseaParse.y"

#line 762 "EaseaParse.cpp"
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
#line 496 "EaseaParse.y"

     //DEBUG_PRT("User makefile options have been reduced");
     
#line 777 "EaseaParse.cpp"
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
#line 499 "EaseaParse.y"

#line 790 "EaseaParse.cpp"
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
#line 509 "EaseaParse.y"
nNB_GEN=(int)yyattribute(2 - 2).dValue;
#line 803 "EaseaParse.cpp"
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
#line 511 "EaseaParse.y"
nNB_OPT_IT=(int)yyattribute(2 - 2).dValue;
#line 816 "EaseaParse.cpp"
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
#line 513 "EaseaParse.y"
nTIME_LIMIT=(int)yyattribute(2 - 2).dValue;
#line 829 "EaseaParse.cpp"
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
#line 515 "EaseaParse.y"
fMUT_PROB=(float)yyattribute(2 - 2).dValue;
#line 842 "EaseaParse.cpp"
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
#line 517 "EaseaParse.y"
fXOVER_PROB=(float)yyattribute(2 - 2).dValue;
#line 855 "EaseaParse.cpp"
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
#line 519 "EaseaParse.y"
nPOP_SIZE=(int)yyattribute(2 - 2).dValue;
#line 868 "EaseaParse.cpp"
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
#line 520 "EaseaParse.y"

      strcpy(sSELECTOR, yyattribute(2 - 2).pSymbol->sName);
      strcpy(sSELECTOR_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelector(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME,&EASEALexer);
	break;
      }
    
#line 890 "EaseaParse.cpp"
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
#line 530 "EaseaParse.y"

      sprintf(sSELECTOR, yyattribute(2 - 3).pSymbol->sName);   
      sprintf(sSELECTOR_OPERATOR, yyattribute(2 - 3).pSymbol->sName);   
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelectorArgument(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME,(float)yyattribute(3 - 3).dValue,&EASEALexer);
	break;
      }
    
#line 912 "EaseaParse.cpp"
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
#line 540 "EaseaParse.y"

        sprintf(sRED_PAR, yyattribute(2 - 2).pSymbol->sName);
	sprintf(sRED_PAR_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME,&EASEALexer);
	  break;
	}
    
#line 934 "EaseaParse.cpp"
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
#line 550 "EaseaParse.y"

        sprintf(sRED_PAR, yyattribute(2 - 3).pSymbol->sName);
	sprintf(sRED_PAR_OPERATOR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME,(float)yyattribute(3 - 3).dValue,&EASEALexer);
	  break;
	}
    
#line 956 "EaseaParse.cpp"
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
#line 560 "EaseaParse.y"

	sprintf(sRED_OFF, yyattribute(2 - 2).pSymbol->sName);
	sprintf(sRED_OFF_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
      switch (TARGET) {
      case STD:
      case CUDA:
	pickupSTDSelector(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME,&EASEALexer);
	break;
      }
    
#line 978 "EaseaParse.cpp"
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
#line 570 "EaseaParse.y"

        sprintf(sRED_OFF, yyattribute(2 - 3).pSymbol->sName);
	sprintf(sRED_OFF_OPERATOR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
	case STD:
	case CUDA:
	  pickupSTDSelectorArgument(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME,yyattribute(3 - 3).dValue,&EASEALexer);
       }
#line 998 "EaseaParse.cpp"
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
#line 578 "EaseaParse.y"

        sprintf(sRED_FINAL, yyattribute(2 - 2).pSymbol->sName);
        sprintf(sRED_FINAL_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME,&EASEALexer);
	  break;
       }
#line 1019 "EaseaParse.cpp"
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
#line 587 "EaseaParse.y"

        sprintf(sRED_FINAL, yyattribute(2 - 3).pSymbol->sName);
        sprintf(sRED_FINAL_OPERATOR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME,yyattribute(3 - 3).dValue,&EASEALexer);
	  break;
	}
#line 1040 "EaseaParse.cpp"
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
#line 596 "EaseaParse.y"
nOFF_SIZE=(int)yyattribute(2 - 2).dValue;
#line 1053 "EaseaParse.cpp"
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
#line 597 "EaseaParse.y"
nOFF_SIZE=(int)(yyattribute(2 - 3).dValue*nPOP_SIZE/100);
#line 1066 "EaseaParse.cpp"
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
#line 598 "EaseaParse.y"
fSURV_PAR_SIZE=(float)yyattribute(2 - 2).dValue;
#line 1079 "EaseaParse.cpp"
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
#line 599 "EaseaParse.y"
fSURV_PAR_SIZE=(float)(yyattribute(2 - 3).dValue/100);
#line 1092 "EaseaParse.cpp"
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
#line 600 "EaseaParse.y"
fSURV_OFF_SIZE=(float)yyattribute(2 - 2).dValue;
#line 1105 "EaseaParse.cpp"
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
#line 601 "EaseaParse.y"
fSURV_OFF_SIZE=(float)(yyattribute(2 - 3).dValue/100);
#line 1118 "EaseaParse.cpp"
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
#line 602 "EaseaParse.y"

      if ((!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Maximise")) || (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Maximize"))) nMINIMISE=0;
      else if ((!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Minimise")) || (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Minimize"))) nMINIMISE=1;
      else {
         fprintf(stderr,"\n%s - Error line %d: The evaluator goal default parameter can only take\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         fprintf(stderr,"two values : maximi[sz]e or minimi[sz]e.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
       }
      
#line 1139 "EaseaParse.cpp"
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
#line 611 "EaseaParse.y"

        nELITE=(int)yyattribute(2 - 2).dValue;
        
#line 1154 "EaseaParse.cpp"
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
#line 614 "EaseaParse.y"

        nELITE=(int)yyattribute(2 - 3).dValue*nPOP_SIZE/100;
        
#line 1169 "EaseaParse.cpp"
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
#line 617 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Weak")) bELITISM=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Strong")) bELITISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Elitism must be \"Strong\" or \"Weak\".\nDefault value \"Strong\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bELITISM=1;
       }
#line 1188 "EaseaParse.cpp"
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
#line 624 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bBALDWINISM=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bBALDWINISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Baldwinism must be \"True\" or \"False\".\nDefault value \"True\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bBALDWINISM=1;
       }
#line 1207 "EaseaParse.cpp"
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
#line 632 "EaseaParse.y"

	if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bREMOTE_ISLAND_MODEL=0;
	else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bREMOTE_ISLAND_MODEL=1;
	else {
	  fprintf(stderr,"\n%s - Warning line %d: remote island model must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
	  bREMOTE_ISLAND_MODEL=0;
	}
#line 1226 "EaseaParse.cpp"
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
#line 639 "EaseaParse.y"

        sprintf(sIP_FILE, yyattribute(2 - 4).pSymbol->sName);
	strcat(sIP_FILE,".");
	strcat(sIP_FILE,yyattribute(4 - 4).pSymbol->sName);
	
#line 1243 "EaseaParse.cpp"
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
#line 644 "EaseaParse.y"

        sprintf(sEXPID, yyattribute(2 - 2).pSymbol->sName);
	
#line 1258 "EaseaParse.cpp"
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
#line 647 "EaseaParse.y"

        sprintf(sWORKING_PATH, yyattribute(2 - 2).pSymbol->sName);
	
#line 1273 "EaseaParse.cpp"
			}
		}
		break;
	case 69:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 651 "EaseaParse.y"

        sprintf(sIP_FILE, yyattribute(2 - 4).pSymbol->sName);
	strcat(sIP_FILE,".");
	strcat(sIP_FILE,yyattribute(4 - 4).pSymbol->sName);
	
#line 1290 "EaseaParse.cpp"
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
#line 657 "EaseaParse.y"

	fMIGRATION_PROBABILITY=(float)yyattribute(2 - 2).dValue;
	
#line 1305 "EaseaParse.cpp"
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
#line 660 "EaseaParse.y"

      nSERVER_PORT=(int)yyattribute(2 - 2).dValue;
    
#line 1320 "EaseaParse.cpp"
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
#line 663 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bPRINT_STATS=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bPRINT_STATS=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Print stats must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bPRINT_STATS=0;
       }
#line 1339 "EaseaParse.cpp"
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
#line 670 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bPLOT_STATS=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bPLOT_STATS=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate stats must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bPLOT_STATS=0;
       }
#line 1358 "EaseaParse.cpp"
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
#line 677 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bGENERATE_CSV_FILE=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bGENERATE_CSV_FILE=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate csv file must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bGENERATE_CSV_FILE=0;
       }
#line 1377 "EaseaParse.cpp"
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
#line 684 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bGENERATE_GNUPLOT_SCRIPT=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bGENERATE_GNUPLOT_SCRIPT=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate gnuplot script must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bGENERATE_GNUPLOT_SCRIPT=0;
       }
#line 1396 "EaseaParse.cpp"
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

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bGENERATE_R_SCRIPT=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bGENERATE_R_SCRIPT=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate R script must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bGENERATE_R_SCRIPT=0;
       }
#line 1415 "EaseaParse.cpp"
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
#line 698 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bSAVE_POPULATION=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bSAVE_POPULATION=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: SavePopulation must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bSAVE_POPULATION=0;
       }
#line 1434 "EaseaParse.cpp"
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
#line 705 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bSTART_FROM_FILE=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bSTART_FROM_FILE=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: StartFromFile must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bSTART_FROM_FILE=0;
       }
#line 1453 "EaseaParse.cpp"
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
#line 712 "EaseaParse.y"
iMAX_INIT_TREE_D = (unsigned)yyattribute(2 - 2).dValue;
#line 1466 "EaseaParse.cpp"
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
#line 713 "EaseaParse.y"
iMIN_INIT_TREE_D = (unsigned)yyattribute(2 - 2).dValue;
#line 1479 "EaseaParse.cpp"
			}
		}
		break;
	case 81:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 714 "EaseaParse.y"
iMAX_TREE_D = (unsigned)yyattribute(2 - 2).dValue;
#line 1492 "EaseaParse.cpp"
			}
		}
		break;
	case 82:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 715 "EaseaParse.y"
iNB_GPU = (unsigned)yyattribute(2 - 2).dValue;
#line 1505 "EaseaParse.cpp"
			}
		}
		break;
	case 83:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 716 "EaseaParse.y"
iPRG_BUF_SIZE = (unsigned)yyattribute(2 - 2).dValue;
#line 1518 "EaseaParse.cpp"
			}
		}
		break;
	case 84:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 719 "EaseaParse.y"
iNO_FITNESS_CASES = (unsigned)yyattribute(2 - 2).dValue;
#line 1531 "EaseaParse.cpp"
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
#line 723 "EaseaParse.y"
 
      if (SymbolTable.find(yyattribute(1 - 3).pSymbol->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 3).pSymbol->sName);
         exit(1);
      }
      (*(YYSTYPE YYFAR*)yyvalptr).dValue = assign(SymbolTable.find(yyattribute(1 - 3).pSymbol->sName), yyattribute(3 - 3).dValue);
    
#line 1550 "EaseaParse.cpp"
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
#line 730 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue + yyattribute(3 - 3).dValue; 
#line 1563 "EaseaParse.cpp"
			}
		}
		break;
	case 87:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 731 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue - yyattribute(3 - 3).dValue; 
#line 1576 "EaseaParse.cpp"
			}
		}
		break;
	case 88:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 732 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue * yyattribute(3 - 3).dValue; 
#line 1589 "EaseaParse.cpp"
			}
		}
		break;
	case 89:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 733 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = divide(yyattribute(1 - 3).dValue, yyattribute(3 - 3).dValue); 
#line 1602 "EaseaParse.cpp"
			}
		}
		break;
	case 90:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 734 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(2 - 3).dValue; 
#line 1615 "EaseaParse.cpp"
			}
		}
		break;
	case 91:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 735 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = -yyattribute(2 - 2).dValue; 
#line 1628 "EaseaParse.cpp"
			}
		}
		break;
	case 92:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 736 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 1).dValue; 
#line 1641 "EaseaParse.cpp"
			}
		}
		break;
	case 93:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 737 "EaseaParse.y"

      if (SymbolTable.find(yyattribute(1 - 1).pSymbol->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 1).pSymbol->sName);
         exit(1);
      }
      (*(YYSTYPE YYFAR*)yyvalptr).dValue = (SymbolTable.find(yyattribute(1 - 1).pSymbol->sName))->dValue;
    
#line 1660 "EaseaParse.cpp"
			}
		}
		break;
	default:
		yyassert(0);
		break;
	}
}
#line 746 "EaseaParse.y"

                       
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
      printf("tpl is cuda gp\n");
      TARGET=CUDA;
      TARGET_FLAVOR = FLAVOR_GP;
    }
    else if( !mystricmp(sTemp,"gp") ){
      printf("tpl is gp\n");
      TARGET=STD;
      TARGET_FLAVOR = FLAVOR_GP;
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

  printf("Syntax Error at line : %d (on text : %s)\nFor more details during the EASEA compiling, use the \"-v\" option\n",
	 EASEALexer.yylineno,EASEALexer.yytext);
}


#line 1805 "EaseaParse.cpp"
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
		{ "PATH_IDENTIFIER", 270 },
		{ "BOOL", 271 },
		{ "INT", 272 },
		{ "DOUBLE", 273 },
		{ "FLOAT", 274 },
		{ "GPNODE", 275 },
		{ "CHAR", 276 },
		{ "POINTER", 277 },
		{ "NUMBER", 278 },
		{ "NUMBER2", 279 },
		{ "METHODS", 280 },
		{ "STATIC", 281 },
		{ "NB_GEN", 282 },
		{ "NB_OPT_IT", 283 },
		{ "BALDWINISM", 284 },
		{ "MUT_PROB", 285 },
		{ "XOVER_PROB", 286 },
		{ "POP_SIZE", 287 },
		{ "SELECTOR", 288 },
		{ "RED_PAR", 289 },
		{ "RED_OFF", 290 },
		{ "RED_FINAL", 291 },
		{ "OFFSPRING", 292 },
		{ "SURVPAR", 293 },
		{ "SURVOFF", 294 },
		{ "MINIMAXI", 295 },
		{ "ELITISM", 296 },
		{ "ELITE", 297 },
		{ "REMOTE_ISLAND_MODEL", 298 },
		{ "IP_FILE", 299 },
		{ "EXPID", 300 },
		{ "WORKING_PATH", 301 },
		{ "MIGRATION_PROBABILITY", 302 },
		{ "SERVER_PORT", 303 },
		{ "PRINT_STATS", 304 },
		{ "PLOT_STATS", 305 },
		{ "GENERATE_CSV_FILE", 306 },
		{ "GENERATE_GNUPLOT_SCRIPT", 307 },
		{ "GENERATE_R_SCRIPT", 308 },
		{ "SAVE_POPULATION", 309 },
		{ "START_FROM_FILE", 310 },
		{ "TIME_LIMIT", 311 },
		{ "MAX_INIT_TREE_D", 312 },
		{ "MIN_INIT_TREE_D", 313 },
		{ "MAX_TREE_D", 316 },
		{ "NB_GPU", 317 },
		{ "PRG_BUF_SIZE", 318 },
		{ "NO_FITNESS_CASES", 319 },
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
		"Parameter: EXPID IDENTIFIER2",
		"Parameter: WORKING_PATH PATH_IDENTIFIER",
		"Parameter: IP_FILE IDENTIFIER2 \'.\' IDENTIFIER2",
		"Parameter: MIGRATION_PROBABILITY NUMBER2",
		"Parameter: SERVER_PORT NUMBER2",
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
		{ 37, 4, 69 },
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
		{ 37, 2, 81 },
		{ 37, 2, 82 },
		{ 37, 2, 83 },
		{ 37, 2, 84 },
		{ 38, 3, 85 },
		{ 38, 3, 86 },
		{ 38, 3, 87 },
		{ 38, 3, 88 },
		{ 38, 3, 89 },
		{ 38, 3, 90 },
		{ 38, 2, 91 },
		{ 38, 1, 92 },
		{ 38, 1, 93 }
	};
	yyreduction = reduction;

	static const yytokenaction_t YYNEARFAR YYBASED_CODE tokenaction[] = {
		{ 38, YYAT_SHIFT, 76 },
		{ 187, YYAT_SHIFT, 168 },
		{ 175, YYAT_SHIFT, 189 },
		{ 178, YYAT_SHIFT, 191 },
		{ 145, YYAT_SHIFT, 148 },
		{ 180, YYAT_ERROR, 0 },
		{ 187, YYAT_SHIFT, 169 },
		{ 114, YYAT_SHIFT, 128 },
		{ 114, YYAT_SHIFT, 129 },
		{ 114, YYAT_SHIFT, 130 },
		{ 114, YYAT_SHIFT, 131 },
		{ 114, YYAT_SHIFT, 132 },
		{ 114, YYAT_SHIFT, 133 },
		{ 114, YYAT_SHIFT, 134 },
		{ 118, YYAT_SHIFT, 102 },
		{ 118, YYAT_SHIFT, 103 },
		{ 118, YYAT_SHIFT, 104 },
		{ 118, YYAT_SHIFT, 105 },
		{ 118, YYAT_SHIFT, 106 },
		{ 118, YYAT_SHIFT, 107 },
		{ 153, YYAT_SHIFT, 162 },
		{ 153, YYAT_SHIFT, 163 },
		{ 0, YYAT_ERROR, 0 },
		{ 0, YYAT_ERROR, 0 },
		{ 38, YYAT_SHIFT, 1 },
		{ 38, YYAT_SHIFT, 2 },
		{ 38, YYAT_SHIFT, 3 },
		{ 38, YYAT_SHIFT, 4 },
		{ 38, YYAT_SHIFT, 5 },
		{ 38, YYAT_SHIFT, 6 },
		{ 38, YYAT_SHIFT, 7 },
		{ 38, YYAT_SHIFT, 8 },
		{ 38, YYAT_SHIFT, 9 },
		{ 38, YYAT_SHIFT, 10 },
		{ 38, YYAT_SHIFT, 11 },
		{ 38, YYAT_SHIFT, 12 },
		{ 38, YYAT_SHIFT, 13 },
		{ 38, YYAT_SHIFT, 14 },
		{ 38, YYAT_SHIFT, 15 },
		{ 38, YYAT_SHIFT, 16 },
		{ 38, YYAT_SHIFT, 17 },
		{ 38, YYAT_SHIFT, 18 },
		{ 38, YYAT_SHIFT, 19 },
		{ 38, YYAT_SHIFT, 20 },
		{ 38, YYAT_SHIFT, 21 },
		{ 38, YYAT_SHIFT, 22 },
		{ 38, YYAT_SHIFT, 23 },
		{ 38, YYAT_SHIFT, 24 },
		{ 38, YYAT_SHIFT, 25 },
		{ 38, YYAT_SHIFT, 26 },
		{ 38, YYAT_SHIFT, 27 },
		{ 38, YYAT_SHIFT, 28 },
		{ 38, YYAT_SHIFT, 29 },
		{ 38, YYAT_SHIFT, 30 },
		{ 38, YYAT_SHIFT, 31 },
		{ 38, YYAT_SHIFT, 32 },
		{ 172, YYAT_SHIFT, 188 },
		{ 180, YYAT_SHIFT, 193 },
		{ 38, YYAT_SHIFT, 33 },
		{ 38, YYAT_SHIFT, 34 },
		{ 38, YYAT_SHIFT, 35 },
		{ 38, YYAT_SHIFT, 36 },
		{ 108, YYAT_SHIFT, 102 },
		{ 108, YYAT_SHIFT, 103 },
		{ 108, YYAT_SHIFT, 104 },
		{ 108, YYAT_SHIFT, 105 },
		{ 108, YYAT_SHIFT, 106 },
		{ 108, YYAT_SHIFT, 107 },
		{ 101, YYAT_SHIFT, 102 },
		{ 101, YYAT_SHIFT, 103 },
		{ 101, YYAT_SHIFT, 104 },
		{ 101, YYAT_SHIFT, 105 },
		{ 101, YYAT_SHIFT, 106 },
		{ 101, YYAT_SHIFT, 107 },
		{ 156, YYAT_SHIFT, 161 },
		{ 181, YYAT_SHIFT, 194 },
		{ 181, YYAT_SHIFT, 184 },
		{ 181, YYAT_SHIFT, 185 },
		{ 161, YYAT_SHIFT, 149 },
		{ 181, YYAT_SHIFT, 186 },
		{ 170, YYAT_SHIFT, 183 },
		{ 181, YYAT_SHIFT, 187 },
		{ 195, YYAT_SHIFT, 184 },
		{ 195, YYAT_SHIFT, 185 },
		{ 161, YYAT_SHIFT, 150 },
		{ 195, YYAT_SHIFT, 186 },
		{ 198, YYAT_SHIFT, 184 },
		{ 195, YYAT_SHIFT, 187 },
		{ 156, YYAT_SHIFT, 164 },
		{ 156, YYAT_SHIFT, 165 },
		{ 197, YYAT_SHIFT, 184 },
		{ 198, YYAT_SHIFT, 187 },
		{ 164, YYAT_ERROR, 0 },
		{ 162, YYAT_ERROR, 0 },
		{ 158, YYAT_SHIFT, 167 },
		{ 197, YYAT_SHIFT, 187 },
		{ 151, YYAT_SHIFT, 160 },
		{ 149, YYAT_SHIFT, 157 },
		{ 126, YYAT_REDUCE, 22 },
		{ 123, YYAT_SHIFT, 144 },
		{ 122, YYAT_SHIFT, 143 },
		{ 121, YYAT_SHIFT, 142 },
		{ 120, YYAT_SHIFT, 141 },
		{ 119, YYAT_SHIFT, 140 },
		{ 115, YYAT_SHIFT, 138 },
		{ 112, YYAT_SHIFT, 127 },
		{ 110, YYAT_SHIFT, 126 },
		{ 107, YYAT_SHIFT, 124 },
		{ 98, YYAT_SHIFT, 96 },
		{ 92, YYAT_SHIFT, 100 },
		{ 90, YYAT_SHIFT, 95 },
		{ 79, YYAT_SHIFT, 77 },
		{ 76, YYAT_REDUCE, 6 },
		{ 57, YYAT_SHIFT, 90 },
		{ 55, YYAT_SHIFT, 89 },
		{ 52, YYAT_SHIFT, 88 },
		{ 51, YYAT_SHIFT, 87 },
		{ 50, YYAT_SHIFT, 86 },
		{ 49, YYAT_SHIFT, 85 },
		{ 48, YYAT_SHIFT, 84 },
		{ 47, YYAT_SHIFT, 83 },
		{ 46, YYAT_SHIFT, 82 },
		{ 37, YYAT_ACCEPT, 0 },
		{ 36, YYAT_SHIFT, 75 },
		{ 35, YYAT_SHIFT, 74 },
		{ 34, YYAT_SHIFT, 73 },
		{ 33, YYAT_SHIFT, 72 },
		{ 32, YYAT_SHIFT, 71 },
		{ 31, YYAT_SHIFT, 70 },
		{ 30, YYAT_SHIFT, 69 },
		{ 29, YYAT_SHIFT, 68 },
		{ 28, YYAT_SHIFT, 67 },
		{ 27, YYAT_SHIFT, 66 },
		{ 26, YYAT_SHIFT, 65 },
		{ 25, YYAT_SHIFT, 64 },
		{ 24, YYAT_SHIFT, 63 },
		{ 23, YYAT_SHIFT, 62 },
		{ 22, YYAT_SHIFT, 61 },
		{ 21, YYAT_SHIFT, 60 },
		{ 20, YYAT_SHIFT, 59 },
		{ 19, YYAT_SHIFT, 58 },
		{ 18, YYAT_SHIFT, 57 },
		{ 17, YYAT_SHIFT, 56 },
		{ 16, YYAT_SHIFT, 55 },
		{ 15, YYAT_SHIFT, 54 },
		{ 14, YYAT_SHIFT, 53 },
		{ 13, YYAT_SHIFT, 52 },
		{ 12, YYAT_SHIFT, 51 },
		{ 11, YYAT_SHIFT, 50 },
		{ 10, YYAT_SHIFT, 49 },
		{ 9, YYAT_SHIFT, 48 },
		{ 8, YYAT_SHIFT, 47 },
		{ 7, YYAT_SHIFT, 46 },
		{ 6, YYAT_SHIFT, 45 },
		{ 5, YYAT_SHIFT, 44 },
		{ 4, YYAT_SHIFT, 43 },
		{ 3, YYAT_SHIFT, 42 },
		{ 2, YYAT_SHIFT, 41 },
		{ 1, YYAT_SHIFT, 40 },
		{ 145, YYAT_SHIFT, 112 },
		{ 145, YYAT_SHIFT, 113 },
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
		{ 178, YYAT_SHIFT, 96 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 175, YYAT_SHIFT, 174 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 187, YYAT_SHIFT, 170 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 187, YYAT_SHIFT, 171 }
	};
	yytokenaction = tokenaction;
	yytokenaction_size = 240;

	static const yystateaction_t YYNEARFAR YYBASED_CODE stateaction[] = {
		{ -236, 1, YYAT_DEFAULT, 38 },
		{ -121, 1, YYAT_DEFAULT, 36 },
		{ -122, 1, YYAT_DEFAULT, 36 },
		{ -113, 1, YYAT_DEFAULT, 90 },
		{ -124, 1, YYAT_DEFAULT, 36 },
		{ -125, 1, YYAT_DEFAULT, 36 },
		{ -126, 1, YYAT_DEFAULT, 36 },
		{ -117, 1, YYAT_DEFAULT, 90 },
		{ -118, 1, YYAT_DEFAULT, 90 },
		{ -119, 1, YYAT_DEFAULT, 90 },
		{ -120, 1, YYAT_DEFAULT, 90 },
		{ -131, 1, YYAT_DEFAULT, 36 },
		{ -132, 1, YYAT_DEFAULT, 36 },
		{ -133, 1, YYAT_DEFAULT, 36 },
		{ -124, 1, YYAT_DEFAULT, 90 },
		{ -125, 1, YYAT_DEFAULT, 90 },
		{ -136, 1, YYAT_DEFAULT, 36 },
		{ -127, 1, YYAT_DEFAULT, 90 },
		{ -128, 1, YYAT_DEFAULT, 90 },
		{ -129, 1, YYAT_DEFAULT, 90 },
		{ -131, 1, YYAT_ERROR, 0 },
		{ -141, 1, YYAT_DEFAULT, 36 },
		{ -142, 1, YYAT_DEFAULT, 36 },
		{ -133, 1, YYAT_DEFAULT, 90 },
		{ -134, 1, YYAT_DEFAULT, 90 },
		{ -135, 1, YYAT_DEFAULT, 90 },
		{ -136, 1, YYAT_DEFAULT, 90 },
		{ -137, 1, YYAT_DEFAULT, 90 },
		{ -138, 1, YYAT_DEFAULT, 90 },
		{ -139, 1, YYAT_DEFAULT, 90 },
		{ -150, 1, YYAT_DEFAULT, 36 },
		{ -151, 1, YYAT_DEFAULT, 36 },
		{ -152, 1, YYAT_DEFAULT, 36 },
		{ -153, 1, YYAT_DEFAULT, 36 },
		{ -154, 1, YYAT_DEFAULT, 36 },
		{ -155, 1, YYAT_DEFAULT, 36 },
		{ -156, 1, YYAT_ERROR, 0 },
		{ 122, 1, YYAT_ERROR, 0 },
		{ -258, 1, YYAT_DEFAULT, 79 },
		{ 0, 0, YYAT_REDUCE, 66 },
		{ 0, 0, YYAT_REDUCE, 68 },
		{ 0, 0, YYAT_REDUCE, 69 },
		{ 0, 0, YYAT_REDUCE, 92 },
		{ 0, 0, YYAT_REDUCE, 71 },
		{ 0, 0, YYAT_REDUCE, 72 },
		{ 0, 0, YYAT_REDUCE, 73 },
		{ -158, 1, YYAT_REDUCE, 74 },
		{ -159, 1, YYAT_REDUCE, 76 },
		{ -160, 1, YYAT_REDUCE, 78 },
		{ -161, 1, YYAT_REDUCE, 80 },
		{ 80, 1, YYAT_REDUCE, 82 },
		{ 79, 1, YYAT_REDUCE, 84 },
		{ 78, 1, YYAT_REDUCE, 86 },
		{ 0, 0, YYAT_REDUCE, 88 },
		{ 0, 0, YYAT_REDUCE, 91 },
		{ 77, 1, YYAT_REDUCE, 89 },
		{ 0, 0, YYAT_REDUCE, 93 },
		{ 67, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 95 },
		{ 0, 0, YYAT_REDUCE, 96 },
		{ 0, 0, YYAT_REDUCE, 98 },
		{ 0, 0, YYAT_REDUCE, 99 },
		{ 0, 0, YYAT_REDUCE, 100 },
		{ 0, 0, YYAT_REDUCE, 101 },
		{ 0, 0, YYAT_REDUCE, 102 },
		{ 0, 0, YYAT_REDUCE, 103 },
		{ 0, 0, YYAT_REDUCE, 104 },
		{ 0, 0, YYAT_REDUCE, 105 },
		{ 0, 0, YYAT_REDUCE, 106 },
		{ 0, 0, YYAT_REDUCE, 70 },
		{ 0, 0, YYAT_REDUCE, 107 },
		{ 0, 0, YYAT_REDUCE, 108 },
		{ 0, 0, YYAT_REDUCE, 109 },
		{ 0, 0, YYAT_REDUCE, 110 },
		{ 0, 0, YYAT_REDUCE, 111 },
		{ 0, 0, YYAT_REDUCE, 112 },
		{ -156, 1, YYAT_REDUCE, 8 },
		{ 0, 0, YYAT_REDUCE, 46 },
		{ 0, 0, YYAT_REDUCE, 1 },
		{ -148, 1, YYAT_ERROR, 0 },
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
		{ -159, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_DEFAULT, 164 },
		{ -14, 1, YYAT_DEFAULT, 110 },
		{ 0, 0, YYAT_REDUCE, 2 },
		{ 0, 0, YYAT_DEFAULT, 101 },
		{ 0, 0, YYAT_REDUCE, 94 },
		{ 0, 0, YYAT_REDUCE, 51 },
		{ 0, 0, YYAT_REDUCE, 11 },
		{ -160, 1, YYAT_REDUCE, 7 },
		{ 0, 0, YYAT_REDUCE, 9 },
		{ 0, 0, YYAT_DEFAULT, 126 },
		{ -192, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 54 },
		{ 0, 0, YYAT_REDUCE, 56 },
		{ 0, 0, YYAT_REDUCE, 58 },
		{ 0, 0, YYAT_REDUCE, 60 },
		{ 0, 0, YYAT_REDUCE, 62 },
		{ -159, 1, YYAT_REDUCE, 65 },
		{ -198, 1, YYAT_REDUCE, 5 },
		{ 0, 0, YYAT_REDUCE, 52 },
		{ -17, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 10 },
		{ -162, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 21 },
		{ -264, 1, YYAT_DEFAULT, 164 },
		{ -21, 1, YYAT_DEFAULT, 145 },
		{ 0, 0, YYAT_REDUCE, 13 },
		{ 0, 0, YYAT_REDUCE, 19 },
		{ -246, 1, YYAT_REDUCE, 3 },
		{ -163, 1, YYAT_DEFAULT, 123 },
		{ -164, 1, YYAT_DEFAULT, 123 },
		{ -165, 1, YYAT_DEFAULT, 123 },
		{ -166, 1, YYAT_DEFAULT, 123 },
		{ -167, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 64 },
		{ 0, 0, YYAT_REDUCE, 53 },
		{ -27, 1, YYAT_DEFAULT, 145 },
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
		{ 0, 0, YYAT_DEFAULT, 161 },
		{ 0, 0, YYAT_DEFAULT, 161 },
		{ 0, 0, YYAT_REDUCE, 12 },
		{ 55, 1, YYAT_DEFAULT, 164 },
		{ 0, 0, YYAT_DEFAULT, 164 },
		{ 5, 1, YYAT_REDUCE, 37 },
		{ 0, 0, YYAT_REDUCE, 16 },
		{ -38, 1, YYAT_DEFAULT, 156 },
		{ 0, 0, YYAT_REDUCE, 35 },
		{ 0, 0, YYAT_REDUCE, 18 },
		{ 30, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_DEFAULT, 164 },
		{ 3, 1, YYAT_REDUCE, 38 },
		{ 0, 0, YYAT_REDUCE, 39 },
		{ 0, 0, YYAT_DEFAULT, 187 },
		{ 36, 1, YYAT_DEFAULT, 164 },
		{ 34, 1, YYAT_DEFAULT, 175 },
		{ 0, 0, YYAT_REDUCE, 23 },
		{ 33, 1, YYAT_DEFAULT, 178 },
		{ 0, 0, YYAT_REDUCE, 25 },
		{ 0, 0, YYAT_REDUCE, 40 },
		{ 0, 0, YYAT_DEFAULT, 187 },
		{ 0, 0, YYAT_DEFAULT, 187 },
		{ 0, 0, YYAT_DEFAULT, 187 },
		{ 19, 1, YYAT_REDUCE, 121 },
		{ 0, 0, YYAT_REDUCE, 120 },
		{ -37, 1, YYAT_DEFAULT, 180 },
		{ 0, 0, YYAT_REDUCE, 36 },
		{ 0, 0, YYAT_REDUCE, 45 },
		{ -57, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 43 },
		{ 0, 0, YYAT_REDUCE, 50 },
		{ -56, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 48 },
		{ -36, 1, YYAT_DEFAULT, 181 },
		{ 34, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 119 },
		{ 0, 0, YYAT_DEFAULT, 187 },
		{ 0, 0, YYAT_DEFAULT, 187 },
		{ 0, 0, YYAT_DEFAULT, 187 },
		{ 0, 0, YYAT_DEFAULT, 187 },
		{ -39, 1, YYAT_DEFAULT, 164 },
		{ 0, 0, YYAT_REDUCE, 41 },
		{ 0, 0, YYAT_REDUCE, 24 },
		{ 0, 0, YYAT_REDUCE, 44 },
		{ 0, 0, YYAT_REDUCE, 26 },
		{ 0, 0, YYAT_REDUCE, 49 },
		{ 0, 0, YYAT_REDUCE, 42 },
		{ 0, 0, YYAT_REDUCE, 118 },
		{ 40, 1, YYAT_REDUCE, 113 },
		{ 0, 0, YYAT_REDUCE, 116 },
		{ 48, 1, YYAT_REDUCE, 114 },
		{ 44, 1, YYAT_REDUCE, 115 },
		{ 0, 0, YYAT_REDUCE, 117 }
	};
	yystateaction = stateaction;

	static const yynontermgoto_t YYNEARFAR YYBASED_CODE nontermgoto[] = {
		{ 38, 78 },
		{ 0, 37 },
		{ 161, 173 },
		{ 38, 79 },
		{ 98, 111 },
		{ 114, 135 },
		{ 114, 136 },
		{ 146, 152 },
		{ 187, 199 },
		{ 161, 151 },
		{ 147, 155 },
		{ 146, 153 },
		{ 146, 154 },
		{ 147, 156 },
		{ 147, 154 },
		{ 114, 137 },
		{ 145, 139 },
		{ 178, 192 },
		{ 178, 177 },
		{ 145, 117 },
		{ 145, 114 },
		{ 186, 198 },
		{ 38, 80 },
		{ 185, 197 },
		{ 98, 97 },
		{ 164, 178 },
		{ 164, 179 },
		{ 162, 175 },
		{ 162, 176 },
		{ 126, 145 },
		{ 126, 116 },
		{ 101, 118 },
		{ 101, 109 },
		{ 91, 98 },
		{ 91, 99 },
		{ 38, 81 },
		{ 0, 38 },
		{ 0, 39 },
		{ 184, 196 },
		{ 183, 195 },
		{ 175, 190 },
		{ 169, 182 },
		{ 168, 181 },
		{ 167, 180 },
		{ 160, 172 },
		{ 157, 166 },
		{ 150, 159 },
		{ 149, 158 },
		{ 136, 147 },
		{ 135, 146 },
		{ 118, 125 },
		{ 106, 123 },
		{ 105, 122 },
		{ 104, 121 },
		{ 103, 120 },
		{ 102, 119 },
		{ 100, 115 },
		{ 97, 110 },
		{ 94, 108 },
		{ 93, 101 },
		{ 80, 94 },
		{ 79, 93 },
		{ 77, 92 },
		{ 76, 91 }
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
		{ 26, 98 },
		{ 0, -1 },
		{ 56, -1 },
		{ 29, 101 },
		{ 0, -1 },
		{ 0, -1 },
		{ 48, -1 },
		{ -4, -1 },
		{ 0, -1 },
		{ 46, 126 },
		{ 2, -1 },
		{ 24, -1 },
		{ 22, -1 },
		{ 20, -1 },
		{ 18, -1 },
		{ 16, -1 },
		{ 0, -1 },
		{ 0, 118 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ -13, -1 },
		{ 0, 145 },
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
		{ 19, 145 },
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
		{ -9, 161 },
		{ -7, 161 },
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
		{ -1, 178 },
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
