#include <cyacc.h>

#line 1 "EaseaParse.y"

/****************************************************************************
EaseaLex.y
Parser for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@polytechnique.fr)
Ecole Polytechnique
Centre de Mathématiques Appliquées
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
bool bBALDWINISM=0;
bool bREMOTE_ISLAND_MODEL=0;
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

#line 69 "EaseaParse.cpp"
// repeated because of possible precompiled header
#include <cyacc.h>

#include "EaseaParse.h"

/////////////////////////////////////////////////////////////////////////////
// constructor

YYPARSENAME::YYPARSENAME()
{
	yytables();
#line 177 "EaseaParse.y"

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

#line 115 "EaseaParse.cpp"
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
#line 217 "EaseaParse.y"

        if (bVERBOSE){ printf("                    _______________________________________\n");
        printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);}
      
#line 165 "EaseaParse.cpp"
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
#line 222 "EaseaParse.y"

        if (bVERBOSE) printf("                    _______________________________________\n");
        if (bVERBOSE) printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);
      
#line 181 "EaseaParse.cpp"
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
#line 230 "EaseaParse.y"

    if (bVERBOSE) printf("Declaration of user classes :\n\n");
#line 195 "EaseaParse.cpp"
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
#line 233 "EaseaParse.y"

      if (bVERBOSE) printf("No user class declaration found other than GenomeClass.\n");
#line 209 "EaseaParse.cpp"
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
#line 243 "EaseaParse.y"

      pCURRENT_CLASS=SymbolTable.insert(yyattribute(1 - 1).pSymbol);  
      pCURRENT_CLASS->pSymbolList=new CLList<CSymbol *>();
      yyattribute(1 - 1).pSymbol->ObjectType=oUserClass;
      //DEBUG_PRT("Yacc Symbol declaration %s %d",$1->sName,$1->nSize);
      pCLASSES[nClasses_nb++] = yyattribute(1 - 1).pSymbol;
    
#line 228 "EaseaParse.cpp"
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
#line 250 "EaseaParse.y"

      if (bVERBOSE) printf("Class %s declared for %d bytes.\n\n",yyattribute(1 - 5).pSymbol->sName,yyattribute(1 - 5).pSymbol->nSize);
      //DEBUG_PRT("Yacc variable declaration %s %d",$1->sName,$1->nSize);
    
#line 244 "EaseaParse.cpp"
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
#line 263 "EaseaParse.y"
pCURRENT_TYPE=yyattribute(2 - 2).pSymbol; pCURRENT_TYPE->ObjectQualifier=yyattribute(1 - 2).ObjectQualifier;
#line 257 "EaseaParse.cpp"
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
#line 263 "EaseaParse.y"

#line 270 "EaseaParse.cpp"
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
#line 264 "EaseaParse.y"
pCURRENT_TYPE=yyattribute(2 - 2).pSymbol; pCURRENT_TYPE->ObjectQualifier=yyattribute(1 - 2).ObjectQualifier;
#line 283 "EaseaParse.cpp"
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
#line 264 "EaseaParse.y"

#line 296 "EaseaParse.cpp"
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
#line 269 "EaseaParse.y"

    pCURRENT_CLASS->sString = new char[strlen(yyattribute(2 - 2).szString) + 1];
    strcpy(pCURRENT_CLASS->sString, yyattribute(2 - 2).szString);      
    if (bVERBOSE) printf("\n    The following methods have been declared:\n\n%s\n\n",pCURRENT_CLASS->sString);
    
#line 313 "EaseaParse.cpp"
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
#line 277 "EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).ObjectQualifier=1;
#line 326 "EaseaParse.cpp"
			}
		}
		break;
	case 12:
		{
#line 278 "EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).ObjectQualifier=0;
#line 334 "EaseaParse.cpp"
		}
		break;
	case 13:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[5];
			yyinitdebug((void YYFAR**)yya, 5);
#endif
			{
#line 283 "EaseaParse.y"

#line 346 "EaseaParse.cpp"
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
#line 287 "EaseaParse.y"

#line 359 "EaseaParse.cpp"
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
#line 288 "EaseaParse.y"

#line 372 "EaseaParse.cpp"
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
#line 302 "EaseaParse.y"
  
      CSymbol *pSym=SymbolTable.find(yyattribute(1 - 1).pSymbol->sName);
      if (pSym==NULL) {
        fprintf(stderr,"\n%s - Error line %d: Class \"%s\" was not defined.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 1).pSymbol->sName);
        fprintf(stderr,"Only base types (bool, int, float, double, char) or new user classes defined\nwithin the \"User classes\" sections are allowed.\n");
        exit(1);
      }       
      else (*(YYSTYPE YYFAR*)yyvalptr).pSymbol=pSym;
    
#line 393 "EaseaParse.cpp"
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
#line 323 "EaseaParse.y"

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
    
#line 417 "EaseaParse.cpp"
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
#line 335 "EaseaParse.y"

      yyattribute(2 - 2).pSymbol->nSize=sizeof (char *);
      yyattribute(2 - 2).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(2 - 2).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(2 - 2).pSymbol->ObjectType=oPointer;
      yyattribute(2 - 2).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(2 - 2).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(2 - 2).pSymbol));
      if (bVERBOSE) printf("    %s pointer declared (%d bytes)\n",yyattribute(2 - 2).pSymbol->sName,yyattribute(2 - 2).pSymbol->nSize);
    
#line 439 "EaseaParse.cpp"
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
#line 345 "EaseaParse.y"

      yyattribute(2 - 2).pSymbol->nSize=sizeof (char *);
      yyattribute(2 - 2).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(2 - 2).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(2 - 2).pSymbol->ObjectType=oPointer;
      yyattribute(2 - 2).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(2 - 2).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(2 - 2).pSymbol));
      if (bVERBOSE) printf("    %s NULL pointer declared (%d bytes)\n",yyattribute(2 - 2).pSymbol->sName,yyattribute(2 - 2).pSymbol->nSize);
    
#line 461 "EaseaParse.cpp"
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
#line 355 "EaseaParse.y"

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
    
#line 485 "EaseaParse.cpp"
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
#line 368 "EaseaParse.y"

      if((TARGET_FLAVOR==STD_FLAVOR_CMAES || TARGET_FLAVOR==CUDA_FLAVOR_CMAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName,"Genome")==0) { nGENOME_NAME=yyattribute(1 - 4).pSymbol->sName; nPROBLEM_DIM=(int)yyattribute(3 - 4).dValue;}

      printf("DEBUG : size of $3 %d nSize %d\n",(int)yyattribute(3 - 4).dValue,pCURRENT_TYPE->nSize);

      yyattribute(1 - 4).pSymbol->nSize=pCURRENT_TYPE->nSize*(int)yyattribute(3 - 4).dValue;
      yyattribute(1 - 4).pSymbol->pClass=pCURRENT_CLASS;
      yyattribute(1 - 4).pSymbol->pType=pCURRENT_TYPE;
      yyattribute(1 - 4).pSymbol->ObjectType=oArray;
      yyattribute(1 - 4).pSymbol->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=yyattribute(1 - 4).pSymbol->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)(yyattribute(1 - 4).pSymbol));
      if (bVERBOSE) printf("    %s array declared (%d bytes)\n",yyattribute(1 - 4).pSymbol->sName,yyattribute(1 - 4).pSymbol->nSize);
    
#line 511 "EaseaParse.cpp"
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
#line 382 "EaseaParse.y"


    // this is for support of pointer array. This should be done in a more generic way in a later version
      if((TARGET_FLAVOR==STD_FLAVOR_CMAES || TARGET_FLAVOR==CUDA_FLAVOR_CMAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName,"Genome")==0) { 
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
    
#line 543 "EaseaParse.cpp"
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
#line 410 "EaseaParse.y"

#line 556 "EaseaParse.cpp"
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
#line 423 "EaseaParse.y"

    ////DEBUG_PRT("Yacc genome decl %s",$1.pSymbol->sName);
      if (bVERBOSE) printf ("\nGenome declaration analysis :\n\n");
      pGENOME=new CSymbol("Genome");
      pCURRENT_CLASS=SymbolTable.insert(pGENOME);  
      pGENOME->pSymbolList=new CLList<CSymbol *>();
      pGENOME->ObjectType=oUserClass;
      pGENOME->ObjectQualifier=0;
      pGENOME->sString=NULL;
    
#line 578 "EaseaParse.cpp"
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
#line 433 "EaseaParse.y"

#line 591 "EaseaParse.cpp"
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
#line 450 "EaseaParse.y"

#line 604 "EaseaParse.cpp"
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
#line 454 "EaseaParse.y"
(*(YYSTYPE YYFAR*)yyvalptr).pSymbol=yyattribute(1 - 1).pSymbol;
#line 617 "EaseaParse.cpp"
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
#line 463 "EaseaParse.y"
         
      if (bVERBOSE) printf("Inserting genome initialiser (taken from .ez file).\n");
    
#line 632 "EaseaParse.cpp"
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
#line 466 "EaseaParse.y"

#line 645 "EaseaParse.cpp"
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
#line 467 "EaseaParse.y"

      if (bVERBOSE) printf("Inserting user genome crossover (taken from .ez file).\n");
    
#line 660 "EaseaParse.cpp"
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
#line 470 "EaseaParse.y"

#line 673 "EaseaParse.cpp"
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
#line 471 "EaseaParse.y"

      if (bVERBOSE) printf("Inserting user genome mutator (taken from .ez file).\n");
    
#line 688 "EaseaParse.cpp"
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
#line 474 "EaseaParse.y"

#line 701 "EaseaParse.cpp"
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
#line 475 "EaseaParse.y"
 
      if (bVERBOSE) printf("Inserting user genome evaluator (taken from .ez file).\n");
    
#line 716 "EaseaParse.cpp"
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
#line 478 "EaseaParse.y"

#line 729 "EaseaParse.cpp"
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
#line 479 "EaseaParse.y"
 
      if (bVERBOSE) printf("Inserting user genome optimiser (taken from .ez file).\n");
    
#line 744 "EaseaParse.cpp"
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
#line 482 "EaseaParse.y"

#line 757 "EaseaParse.cpp"
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
#line 483 "EaseaParse.y"

     //DEBUG_PRT("User makefile options have been reduced");
     
#line 772 "EaseaParse.cpp"
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
#line 486 "EaseaParse.y"

#line 785 "EaseaParse.cpp"
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
#line 496 "EaseaParse.y"
nNB_GEN=(int)yyattribute(2 - 2).dValue;
#line 798 "EaseaParse.cpp"
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
#line 498 "EaseaParse.y"
nNB_OPT_IT=(int)yyattribute(2 - 2).dValue;
#line 811 "EaseaParse.cpp"
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
#line 500 "EaseaParse.y"
nTIME_LIMIT=(int)yyattribute(2 - 2).dValue;
#line 824 "EaseaParse.cpp"
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
#line 502 "EaseaParse.y"
fMUT_PROB=(float)yyattribute(2 - 2).dValue;
#line 837 "EaseaParse.cpp"
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
#line 504 "EaseaParse.y"
fXOVER_PROB=(float)yyattribute(2 - 2).dValue;
#line 850 "EaseaParse.cpp"
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
#line 506 "EaseaParse.y"
nPOP_SIZE=(int)yyattribute(2 - 2).dValue;
#line 863 "EaseaParse.cpp"
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
#line 507 "EaseaParse.y"

      strcpy(sSELECTOR, yyattribute(2 - 2).pSymbol->sName);
      strcpy(sSELECTOR_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelector(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME,&EASEALexer);
	break;
      }
    
#line 885 "EaseaParse.cpp"
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
#line 517 "EaseaParse.y"

      sprintf(sSELECTOR, yyattribute(2 - 3).pSymbol->sName);   
      sprintf(sSELECTOR_OPERATOR, yyattribute(2 - 3).pSymbol->sName);   
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelectorArgument(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME,(float)yyattribute(3 - 3).dValue,&EASEALexer);
	break;
      }
    
#line 907 "EaseaParse.cpp"
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
#line 527 "EaseaParse.y"

        sprintf(sRED_PAR, yyattribute(2 - 2).pSymbol->sName);
	sprintf(sRED_PAR_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME,&EASEALexer);
	  break;
	}
    
#line 929 "EaseaParse.cpp"
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
#line 537 "EaseaParse.y"

        sprintf(sRED_PAR, yyattribute(2 - 3).pSymbol->sName);
	sprintf(sRED_PAR_OPERATOR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME,(float)yyattribute(3 - 3).dValue,&EASEALexer);
	  break;
	}
    
#line 951 "EaseaParse.cpp"
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
#line 547 "EaseaParse.y"

	sprintf(sRED_OFF, yyattribute(2 - 2).pSymbol->sName);
	sprintf(sRED_OFF_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
      switch (TARGET) {
      case STD:
      case CUDA:
	pickupSTDSelector(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME,&EASEALexer);
	break;
      }
    
#line 973 "EaseaParse.cpp"
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
#line 557 "EaseaParse.y"

        sprintf(sRED_OFF, yyattribute(2 - 3).pSymbol->sName);
	sprintf(sRED_OFF_OPERATOR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
	case STD:
	case CUDA:
	  pickupSTDSelectorArgument(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME,yyattribute(3 - 3).dValue,&EASEALexer);
       }
#line 993 "EaseaParse.cpp"
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
#line 565 "EaseaParse.y"

        sprintf(sRED_FINAL, yyattribute(2 - 2).pSymbol->sName);
        sprintf(sRED_FINAL_OPERATOR, yyattribute(2 - 2).pSymbol->sName);
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME,&EASEALexer);
	  break;
       }
#line 1014 "EaseaParse.cpp"
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
#line 574 "EaseaParse.y"

        sprintf(sRED_FINAL, yyattribute(2 - 3).pSymbol->sName);
        sprintf(sRED_FINAL_OPERATOR, yyattribute(2 - 3).pSymbol->sName);
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME,yyattribute(3 - 3).dValue,&EASEALexer);
	  break;
	}
#line 1035 "EaseaParse.cpp"
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
#line 583 "EaseaParse.y"
nOFF_SIZE=(int)yyattribute(2 - 2).dValue;
#line 1048 "EaseaParse.cpp"
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
#line 584 "EaseaParse.y"
nOFF_SIZE=(int)(yyattribute(2 - 3).dValue*nPOP_SIZE/100);
#line 1061 "EaseaParse.cpp"
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
#line 585 "EaseaParse.y"
fSURV_PAR_SIZE=(float)yyattribute(2 - 2).dValue;
#line 1074 "EaseaParse.cpp"
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
#line 586 "EaseaParse.y"
fSURV_PAR_SIZE=(float)(yyattribute(2 - 3).dValue/100);
#line 1087 "EaseaParse.cpp"
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
#line 587 "EaseaParse.y"
fSURV_OFF_SIZE=(float)yyattribute(2 - 2).dValue;
#line 1100 "EaseaParse.cpp"
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
#line 588 "EaseaParse.y"
fSURV_OFF_SIZE=(float)(yyattribute(2 - 3).dValue/100);
#line 1113 "EaseaParse.cpp"
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
#line 589 "EaseaParse.y"

      if ((!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Maximise")) || (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Maximize"))) nMINIMISE=0;
      else if ((!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Minimise")) || (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Minimize"))) nMINIMISE=1;
      else {
         fprintf(stderr,"\n%s - Error line %d: The evaluator goal default parameter can only take\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         fprintf(stderr,"two values : maximi[sz]e or minimi[sz]e.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
       }
      
#line 1134 "EaseaParse.cpp"
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
#line 598 "EaseaParse.y"

        nELITE=(int)yyattribute(2 - 2).dValue;
        
#line 1149 "EaseaParse.cpp"
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
#line 601 "EaseaParse.y"

        nELITE=(int)yyattribute(2 - 3).dValue*nPOP_SIZE/100;
        
#line 1164 "EaseaParse.cpp"
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
#line 604 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Weak")) bELITISM=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"Strong")) bELITISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Elitism must be \"Strong\" or \"Weak\".\nDefault value \"Strong\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bELITISM=1;
       }
#line 1183 "EaseaParse.cpp"
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
#line 611 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bBALDWINISM=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bBALDWINISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Baldwinism must be \"True\" or \"False\".\nDefault value \"True\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bBALDWINISM=1;
       }
#line 1202 "EaseaParse.cpp"
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
#line 619 "EaseaParse.y"

	if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bREMOTE_ISLAND_MODEL=0;
	else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bREMOTE_ISLAND_MODEL=1;
	else {
	  fprintf(stderr,"\n%s - Warning line %d: remote island model must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
	  bREMOTE_ISLAND_MODEL=0;
	}
#line 1221 "EaseaParse.cpp"
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
#line 627 "EaseaParse.y"

      if((int)yyattribute(2 - 2).dValue>=1)
	 bPRINT_STATS=1;
      else
	 bPRINT_STATS=0;
    
#line 1239 "EaseaParse.cpp"
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
#line 633 "EaseaParse.y"

      if((int)yyattribute(2 - 2).dValue>=1)
	 bPLOT_STATS=1;
      else
	 bPLOT_STATS=0;
    
#line 1257 "EaseaParse.cpp"
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
#line 639 "EaseaParse.y"

      if((int)yyattribute(2 - 2).dValue>=1)
	 bGENERATE_CSV_FILE=1;
      else
	 bGENERATE_CSV_FILE=0;
    
#line 1275 "EaseaParse.cpp"
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
#line 645 "EaseaParse.y"

      if((int)yyattribute(2 - 2).dValue>=1)
	 bGENERATE_GNUPLOT_SCRIPT=1;
      else
	 bGENERATE_GNUPLOT_SCRIPT=0;
    
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
#line 651 "EaseaParse.y"

      if((int)yyattribute(2 - 2).dValue>=1)
	 bGENERATE_R_SCRIPT=1;
      else
	 bGENERATE_R_SCRIPT=0;
    
#line 1311 "EaseaParse.cpp"
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
#line 657 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bSAVE_POPULATION=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bSAVE_POPULATION=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: SavePopulation must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bSAVE_POPULATION=0;
       }
#line 1330 "EaseaParse.cpp"
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
#line 664 "EaseaParse.y"

      if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"False")) bSTART_FROM_FILE=0;
      else if (!mystricmp(yyattribute(2 - 2).pSymbol->sName,"True")) bSTART_FROM_FILE=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: StartFromFile must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bSTART_FROM_FILE=0;
       }
#line 1349 "EaseaParse.cpp"
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
#line 671 "EaseaParse.y"
iMAX_INIT_TREE_D = (unsigned)yyattribute(2 - 2).dValue;
#line 1362 "EaseaParse.cpp"
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
#line 672 "EaseaParse.y"
iMIN_INIT_TREE_D = (unsigned)yyattribute(2 - 2).dValue;
#line 1375 "EaseaParse.cpp"
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
#line 673 "EaseaParse.y"
iMAX_TREE_D = (unsigned)yyattribute(2 - 2).dValue;
#line 1388 "EaseaParse.cpp"
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
#line 674 "EaseaParse.y"
iNB_GPU = (unsigned)yyattribute(2 - 2).dValue;
#line 1401 "EaseaParse.cpp"
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
#line 675 "EaseaParse.y"
iPRG_BUF_SIZE = (unsigned)yyattribute(2 - 2).dValue;
#line 1414 "EaseaParse.cpp"
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
#line 678 "EaseaParse.y"
iNO_FITNESS_CASES = (unsigned)yyattribute(2 - 2).dValue;
#line 1427 "EaseaParse.cpp"
			}
		}
		break;
	case 79:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 682 "EaseaParse.y"
 
      if (SymbolTable.find(yyattribute(1 - 3).pSymbol->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 3).pSymbol->sName);
         exit(1);
      }
      (*(YYSTYPE YYFAR*)yyvalptr).dValue = assign(SymbolTable.find(yyattribute(1 - 3).pSymbol->sName), yyattribute(3 - 3).dValue);
    
#line 1446 "EaseaParse.cpp"
			}
		}
		break;
	case 80:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[4];
			yyinitdebug((void YYFAR**)yya, 4);
#endif
			{
#line 689 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue + yyattribute(3 - 3).dValue; 
#line 1459 "EaseaParse.cpp"
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
#line 690 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue - yyattribute(3 - 3).dValue; 
#line 1472 "EaseaParse.cpp"
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
#line 691 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 3).dValue * yyattribute(3 - 3).dValue; 
#line 1485 "EaseaParse.cpp"
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
#line 692 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = divide(yyattribute(1 - 3).dValue, yyattribute(3 - 3).dValue); 
#line 1498 "EaseaParse.cpp"
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
#line 693 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(2 - 3).dValue; 
#line 1511 "EaseaParse.cpp"
			}
		}
		break;
	case 85:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[3];
			yyinitdebug((void YYFAR**)yya, 3);
#endif
			{
#line 694 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = -yyattribute(2 - 2).dValue; 
#line 1524 "EaseaParse.cpp"
			}
		}
		break;
	case 86:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 695 "EaseaParse.y"
 (*(YYSTYPE YYFAR*)yyvalptr).dValue = yyattribute(1 - 1).dValue; 
#line 1537 "EaseaParse.cpp"
			}
		}
		break;
	case 87:
		{
#ifdef YYDEBUG
			YYSTYPE YYFAR* yya[2];
			yyinitdebug((void YYFAR**)yya, 2);
#endif
			{
#line 696 "EaseaParse.y"

      if (SymbolTable.find(yyattribute(1 - 1).pSymbol->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,yyattribute(1 - 1).pSymbol->sName);
         exit(1);
      }
      (*(YYSTYPE YYFAR*)yyvalptr).dValue = (SymbolTable.find(yyattribute(1 - 1).pSymbol->sName))->dValue;
    
#line 1556 "EaseaParse.cpp"
			}
		}
		break;
	default:
		yyassert(0);
		break;
	}
}
#line 705 "EaseaParse.y"

                       
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
      TARGET=STD;
      TARGET_FLAVOR = STD_FLAVOR_CMAES;
    }
    else if (!mystricmp(sTemp,"cmaes_cuda"))  {
      TARGET=CUDA;
      TARGET_FLAVOR = CUDA_FLAVOR_CMAES;
    }
    else if (!mystricmp(sTemp,"memetic"))  {
      TARGET=STD;
      TARGET_FLAVOR = STD_FLAVOR_MEMETIC;
    }
    else if (!mystricmp(sTemp,"memetic_cuda"))  {
      TARGET=CUDA;
      TARGET_FLAVOR = CUDA_FLAVOR_MEMETIC;
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

#line 1697 "EaseaParse.cpp"
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
		{ "PRINT_STATS", 298 },
		{ "PLOT_STATS", 299 },
		{ "GENERATE_CSV_FILE", 300 },
		{ "GENERATE_GNUPLOT_SCRIPT", 301 },
		{ "GENERATE_R_SCRIPT", 302 },
		{ "SAVE_POPULATION", 303 },
		{ "START_FROM_FILE", 304 },
		{ "TIME_LIMIT", 305 },
		{ "MAX_INIT_TREE_D", 306 },
		{ "MIN_INIT_TREE_D", 307 },
		{ "MAX_TREE_D", 310 },
		{ "NB_GPU", 311 },
		{ "PRG_BUF_SIZE", 312 },
		{ "NO_FITNESS_CASES", 313 },
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
		"Parameter: PRINT_STATS NUMBER2",
		"Parameter: PLOT_STATS NUMBER2",
		"Parameter: GENERATE_CSV_FILE NUMBER2",
		"Parameter: GENERATE_GNUPLOT_SCRIPT NUMBER2",
		"Parameter: GENERATE_R_SCRIPT NUMBER2",
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
		{ 37, 2, 66 },
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
		{ 38, 3, 79 },
		{ 38, 3, 80 },
		{ 38, 3, 81 },
		{ 38, 3, 82 },
		{ 38, 3, 83 },
		{ 38, 3, 84 },
		{ 38, 2, 85 },
		{ 38, 1, 86 },
		{ 38, 1, 87 }
	};
	yyreduction = reduction;

	static const yytokenaction_t YYNEARFAR YYBASED_CODE tokenaction[] = {
		{ 33, YYAT_SHIFT, 66 },
		{ 175, YYAT_SHIFT, 156 },
		{ 163, YYAT_SHIFT, 177 },
		{ 166, YYAT_SHIFT, 179 },
		{ 133, YYAT_SHIFT, 136 },
		{ 168, YYAT_ERROR, 0 },
		{ 175, YYAT_SHIFT, 157 },
		{ 102, YYAT_SHIFT, 116 },
		{ 102, YYAT_SHIFT, 117 },
		{ 102, YYAT_SHIFT, 118 },
		{ 102, YYAT_SHIFT, 119 },
		{ 102, YYAT_SHIFT, 120 },
		{ 102, YYAT_SHIFT, 121 },
		{ 102, YYAT_SHIFT, 122 },
		{ 106, YYAT_SHIFT, 90 },
		{ 106, YYAT_SHIFT, 91 },
		{ 106, YYAT_SHIFT, 92 },
		{ 106, YYAT_SHIFT, 93 },
		{ 106, YYAT_SHIFT, 94 },
		{ 106, YYAT_SHIFT, 95 },
		{ 141, YYAT_SHIFT, 150 },
		{ 141, YYAT_SHIFT, 151 },
		{ 160, YYAT_SHIFT, 176 },
		{ 33, YYAT_SHIFT, 1 },
		{ 33, YYAT_SHIFT, 2 },
		{ 33, YYAT_SHIFT, 3 },
		{ 33, YYAT_SHIFT, 4 },
		{ 33, YYAT_SHIFT, 5 },
		{ 33, YYAT_SHIFT, 6 },
		{ 33, YYAT_SHIFT, 7 },
		{ 33, YYAT_SHIFT, 8 },
		{ 33, YYAT_SHIFT, 9 },
		{ 33, YYAT_SHIFT, 10 },
		{ 33, YYAT_SHIFT, 11 },
		{ 33, YYAT_SHIFT, 12 },
		{ 33, YYAT_SHIFT, 13 },
		{ 33, YYAT_SHIFT, 14 },
		{ 33, YYAT_SHIFT, 15 },
		{ 33, YYAT_SHIFT, 16 },
		{ 33, YYAT_SHIFT, 17 },
		{ 33, YYAT_SHIFT, 18 },
		{ 33, YYAT_SHIFT, 19 },
		{ 33, YYAT_SHIFT, 20 },
		{ 33, YYAT_SHIFT, 21 },
		{ 33, YYAT_SHIFT, 22 },
		{ 33, YYAT_SHIFT, 23 },
		{ 33, YYAT_SHIFT, 24 },
		{ 33, YYAT_SHIFT, 25 },
		{ 33, YYAT_SHIFT, 26 },
		{ 33, YYAT_SHIFT, 27 },
		{ 0, YYAT_ERROR, 0 },
		{ 0, YYAT_ERROR, 0 },
		{ 33, YYAT_SHIFT, 28 },
		{ 33, YYAT_SHIFT, 29 },
		{ 33, YYAT_SHIFT, 30 },
		{ 33, YYAT_SHIFT, 31 },
		{ 144, YYAT_SHIFT, 149 },
		{ 168, YYAT_SHIFT, 181 },
		{ 96, YYAT_SHIFT, 90 },
		{ 96, YYAT_SHIFT, 91 },
		{ 96, YYAT_SHIFT, 92 },
		{ 96, YYAT_SHIFT, 93 },
		{ 96, YYAT_SHIFT, 94 },
		{ 96, YYAT_SHIFT, 95 },
		{ 89, YYAT_SHIFT, 90 },
		{ 89, YYAT_SHIFT, 91 },
		{ 89, YYAT_SHIFT, 92 },
		{ 89, YYAT_SHIFT, 93 },
		{ 89, YYAT_SHIFT, 94 },
		{ 89, YYAT_SHIFT, 95 },
		{ 144, YYAT_SHIFT, 152 },
		{ 144, YYAT_SHIFT, 153 },
		{ 169, YYAT_SHIFT, 182 },
		{ 169, YYAT_SHIFT, 172 },
		{ 169, YYAT_SHIFT, 173 },
		{ 149, YYAT_SHIFT, 137 },
		{ 169, YYAT_SHIFT, 174 },
		{ 158, YYAT_SHIFT, 171 },
		{ 169, YYAT_SHIFT, 175 },
		{ 183, YYAT_SHIFT, 172 },
		{ 183, YYAT_SHIFT, 173 },
		{ 149, YYAT_SHIFT, 138 },
		{ 183, YYAT_SHIFT, 174 },
		{ 186, YYAT_SHIFT, 172 },
		{ 183, YYAT_SHIFT, 175 },
		{ 185, YYAT_SHIFT, 172 },
		{ 152, YYAT_ERROR, 0 },
		{ 150, YYAT_ERROR, 0 },
		{ 186, YYAT_SHIFT, 175 },
		{ 146, YYAT_SHIFT, 155 },
		{ 185, YYAT_SHIFT, 175 },
		{ 139, YYAT_SHIFT, 148 },
		{ 137, YYAT_SHIFT, 145 },
		{ 114, YYAT_REDUCE, 22 },
		{ 111, YYAT_SHIFT, 132 },
		{ 110, YYAT_SHIFT, 131 },
		{ 109, YYAT_SHIFT, 130 },
		{ 108, YYAT_SHIFT, 129 },
		{ 107, YYAT_SHIFT, 128 },
		{ 103, YYAT_SHIFT, 126 },
		{ 100, YYAT_SHIFT, 115 },
		{ 98, YYAT_SHIFT, 114 },
		{ 95, YYAT_SHIFT, 112 },
		{ 86, YYAT_SHIFT, 84 },
		{ 81, YYAT_SHIFT, 88 },
		{ 69, YYAT_SHIFT, 67 },
		{ 66, YYAT_REDUCE, 6 },
		{ 50, YYAT_SHIFT, 79 },
		{ 47, YYAT_SHIFT, 78 },
		{ 46, YYAT_SHIFT, 77 },
		{ 45, YYAT_SHIFT, 76 },
		{ 44, YYAT_SHIFT, 75 },
		{ 43, YYAT_SHIFT, 74 },
		{ 42, YYAT_SHIFT, 73 },
		{ 41, YYAT_SHIFT, 72 },
		{ 32, YYAT_ACCEPT, 0 },
		{ 31, YYAT_SHIFT, 65 },
		{ 30, YYAT_SHIFT, 64 },
		{ 29, YYAT_SHIFT, 63 },
		{ 28, YYAT_SHIFT, 62 },
		{ 27, YYAT_SHIFT, 61 },
		{ 26, YYAT_SHIFT, 60 },
		{ 25, YYAT_SHIFT, 59 },
		{ 24, YYAT_SHIFT, 58 },
		{ 23, YYAT_SHIFT, 57 },
		{ 22, YYAT_SHIFT, 56 },
		{ 21, YYAT_SHIFT, 55 },
		{ 20, YYAT_SHIFT, 54 },
		{ 19, YYAT_SHIFT, 53 },
		{ 18, YYAT_SHIFT, 52 },
		{ 17, YYAT_SHIFT, 51 },
		{ 16, YYAT_SHIFT, 50 },
		{ 15, YYAT_SHIFT, 49 },
		{ 14, YYAT_SHIFT, 48 },
		{ 13, YYAT_SHIFT, 47 },
		{ 12, YYAT_SHIFT, 46 },
		{ 11, YYAT_SHIFT, 45 },
		{ 10, YYAT_SHIFT, 44 },
		{ 9, YYAT_SHIFT, 43 },
		{ 8, YYAT_SHIFT, 42 },
		{ 7, YYAT_SHIFT, 41 },
		{ 6, YYAT_SHIFT, 40 },
		{ 5, YYAT_SHIFT, 39 },
		{ 4, YYAT_SHIFT, 38 },
		{ 3, YYAT_SHIFT, 37 },
		{ 2, YYAT_SHIFT, 36 },
		{ 1, YYAT_SHIFT, 35 },
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
		{ 133, YYAT_SHIFT, 100 },
		{ 133, YYAT_SHIFT, 101 },
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
		{ 166, YYAT_SHIFT, 84 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 163, YYAT_SHIFT, 162 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 175, YYAT_SHIFT, 158 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ -1, YYAT_ERROR, 0 },
		{ 175, YYAT_SHIFT, 159 }
	};
	yytokenaction = tokenaction;
	yytokenaction_size = 239;

	static const yystateaction_t YYNEARFAR YYBASED_CODE stateaction[] = {
		{ -208, 1, YYAT_DEFAULT, 33 },
		{ -132, 1, YYAT_DEFAULT, 31 },
		{ -133, 1, YYAT_DEFAULT, 31 },
		{ -125, 1, YYAT_DEFAULT, 24 },
		{ -135, 1, YYAT_DEFAULT, 31 },
		{ -136, 1, YYAT_DEFAULT, 31 },
		{ -137, 1, YYAT_DEFAULT, 31 },
		{ -129, 1, YYAT_DEFAULT, 24 },
		{ -130, 1, YYAT_DEFAULT, 24 },
		{ -131, 1, YYAT_DEFAULT, 24 },
		{ -132, 1, YYAT_DEFAULT, 24 },
		{ -142, 1, YYAT_DEFAULT, 31 },
		{ -143, 1, YYAT_DEFAULT, 31 },
		{ -144, 1, YYAT_DEFAULT, 31 },
		{ -136, 1, YYAT_DEFAULT, 24 },
		{ -137, 1, YYAT_DEFAULT, 24 },
		{ -147, 1, YYAT_DEFAULT, 31 },
		{ -139, 1, YYAT_DEFAULT, 24 },
		{ -149, 1, YYAT_DEFAULT, 31 },
		{ -150, 1, YYAT_DEFAULT, 31 },
		{ -151, 1, YYAT_DEFAULT, 31 },
		{ -152, 1, YYAT_DEFAULT, 31 },
		{ -153, 1, YYAT_DEFAULT, 31 },
		{ -145, 1, YYAT_DEFAULT, 24 },
		{ -146, 1, YYAT_ERROR, 0 },
		{ -156, 1, YYAT_DEFAULT, 31 },
		{ -157, 1, YYAT_DEFAULT, 31 },
		{ -158, 1, YYAT_DEFAULT, 31 },
		{ -159, 1, YYAT_DEFAULT, 31 },
		{ -160, 1, YYAT_DEFAULT, 31 },
		{ -161, 1, YYAT_DEFAULT, 31 },
		{ -162, 1, YYAT_ERROR, 0 },
		{ 115, 1, YYAT_ERROR, 0 },
		{ -258, 1, YYAT_DEFAULT, 69 },
		{ 0, 0, YYAT_REDUCE, 66 },
		{ 0, 0, YYAT_REDUCE, 68 },
		{ 0, 0, YYAT_REDUCE, 69 },
		{ 0, 0, YYAT_REDUCE, 92 },
		{ 0, 0, YYAT_REDUCE, 71 },
		{ 0, 0, YYAT_REDUCE, 72 },
		{ 0, 0, YYAT_REDUCE, 73 },
		{ -164, 1, YYAT_REDUCE, 74 },
		{ -165, 1, YYAT_REDUCE, 76 },
		{ -166, 1, YYAT_REDUCE, 78 },
		{ -167, 1, YYAT_REDUCE, 80 },
		{ 73, 1, YYAT_REDUCE, 82 },
		{ 72, 1, YYAT_REDUCE, 84 },
		{ 71, 1, YYAT_REDUCE, 86 },
		{ 0, 0, YYAT_REDUCE, 88 },
		{ 0, 0, YYAT_REDUCE, 91 },
		{ 70, 1, YYAT_REDUCE, 89 },
		{ 0, 0, YYAT_REDUCE, 93 },
		{ 0, 0, YYAT_REDUCE, 94 },
		{ 0, 0, YYAT_REDUCE, 95 },
		{ 0, 0, YYAT_REDUCE, 96 },
		{ 0, 0, YYAT_REDUCE, 97 },
		{ 0, 0, YYAT_REDUCE, 98 },
		{ 0, 0, YYAT_REDUCE, 99 },
		{ 0, 0, YYAT_REDUCE, 100 },
		{ 0, 0, YYAT_REDUCE, 70 },
		{ 0, 0, YYAT_REDUCE, 101 },
		{ 0, 0, YYAT_REDUCE, 102 },
		{ 0, 0, YYAT_REDUCE, 103 },
		{ 0, 0, YYAT_REDUCE, 104 },
		{ 0, 0, YYAT_REDUCE, 105 },
		{ 0, 0, YYAT_REDUCE, 106 },
		{ -162, 1, YYAT_REDUCE, 8 },
		{ 0, 0, YYAT_REDUCE, 46 },
		{ 0, 0, YYAT_REDUCE, 1 },
		{ -154, 1, YYAT_ERROR, 0 },
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
		{ 0, 0, YYAT_DEFAULT, 152 },
		{ -19, 1, YYAT_DEFAULT, 98 },
		{ 0, 0, YYAT_REDUCE, 2 },
		{ 0, 0, YYAT_DEFAULT, 89 },
		{ 0, 0, YYAT_REDUCE, 51 },
		{ 0, 0, YYAT_REDUCE, 11 },
		{ -165, 1, YYAT_REDUCE, 7 },
		{ 0, 0, YYAT_REDUCE, 9 },
		{ 0, 0, YYAT_DEFAULT, 114 },
		{ -196, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 54 },
		{ 0, 0, YYAT_REDUCE, 56 },
		{ 0, 0, YYAT_REDUCE, 58 },
		{ 0, 0, YYAT_REDUCE, 60 },
		{ 0, 0, YYAT_REDUCE, 62 },
		{ -164, 1, YYAT_REDUCE, 65 },
		{ -202, 1, YYAT_REDUCE, 5 },
		{ 0, 0, YYAT_REDUCE, 52 },
		{ -22, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 10 },
		{ -167, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 21 },
		{ -263, 1, YYAT_DEFAULT, 152 },
		{ -26, 1, YYAT_DEFAULT, 133 },
		{ 0, 0, YYAT_REDUCE, 13 },
		{ 0, 0, YYAT_REDUCE, 19 },
		{ -246, 1, YYAT_REDUCE, 3 },
		{ -168, 1, YYAT_DEFAULT, 111 },
		{ -169, 1, YYAT_DEFAULT, 111 },
		{ -170, 1, YYAT_DEFAULT, 111 },
		{ -171, 1, YYAT_DEFAULT, 111 },
		{ -172, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 64 },
		{ 0, 0, YYAT_REDUCE, 53 },
		{ -32, 1, YYAT_DEFAULT, 133 },
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
		{ 0, 0, YYAT_DEFAULT, 149 },
		{ 0, 0, YYAT_DEFAULT, 149 },
		{ 0, 0, YYAT_REDUCE, 12 },
		{ 50, 1, YYAT_DEFAULT, 152 },
		{ 0, 0, YYAT_DEFAULT, 152 },
		{ 0, 1, YYAT_REDUCE, 37 },
		{ 0, 0, YYAT_REDUCE, 16 },
		{ -38, 1, YYAT_DEFAULT, 144 },
		{ 0, 0, YYAT_REDUCE, 35 },
		{ 0, 0, YYAT_REDUCE, 18 },
		{ 12, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_DEFAULT, 152 },
		{ -2, 1, YYAT_REDUCE, 38 },
		{ 0, 0, YYAT_REDUCE, 39 },
		{ 0, 0, YYAT_DEFAULT, 175 },
		{ 33, 1, YYAT_DEFAULT, 152 },
		{ 28, 1, YYAT_DEFAULT, 163 },
		{ 0, 0, YYAT_REDUCE, 23 },
		{ 27, 1, YYAT_DEFAULT, 166 },
		{ 0, 0, YYAT_REDUCE, 25 },
		{ 0, 0, YYAT_REDUCE, 40 },
		{ 0, 0, YYAT_DEFAULT, 175 },
		{ 0, 0, YYAT_DEFAULT, 175 },
		{ 0, 0, YYAT_DEFAULT, 175 },
		{ 16, 1, YYAT_REDUCE, 115 },
		{ 0, 0, YYAT_REDUCE, 114 },
		{ -71, 1, YYAT_DEFAULT, 168 },
		{ 0, 0, YYAT_REDUCE, 36 },
		{ 0, 0, YYAT_REDUCE, 45 },
		{ -57, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 43 },
		{ 0, 0, YYAT_REDUCE, 50 },
		{ -56, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 48 },
		{ -36, 1, YYAT_DEFAULT, 169 },
		{ 31, 1, YYAT_ERROR, 0 },
		{ 0, 0, YYAT_REDUCE, 113 },
		{ 0, 0, YYAT_DEFAULT, 175 },
		{ 0, 0, YYAT_DEFAULT, 175 },
		{ 0, 0, YYAT_DEFAULT, 175 },
		{ 0, 0, YYAT_DEFAULT, 175 },
		{ -39, 1, YYAT_DEFAULT, 152 },
		{ 0, 0, YYAT_REDUCE, 41 },
		{ 0, 0, YYAT_REDUCE, 24 },
		{ 0, 0, YYAT_REDUCE, 44 },
		{ 0, 0, YYAT_REDUCE, 26 },
		{ 0, 0, YYAT_REDUCE, 49 },
		{ 0, 0, YYAT_REDUCE, 42 },
		{ 0, 0, YYAT_REDUCE, 112 },
		{ 37, 1, YYAT_REDUCE, 107 },
		{ 0, 0, YYAT_REDUCE, 110 },
		{ 43, 1, YYAT_REDUCE, 108 },
		{ 41, 1, YYAT_REDUCE, 109 },
		{ 0, 0, YYAT_REDUCE, 111 }
	};
	yystateaction = stateaction;

	static const yynontermgoto_t YYNEARFAR YYBASED_CODE nontermgoto[] = {
		{ 33, 68 },
		{ 0, 32 },
		{ 149, 161 },
		{ 33, 69 },
		{ 86, 99 },
		{ 102, 123 },
		{ 102, 124 },
		{ 134, 140 },
		{ 175, 187 },
		{ 149, 139 },
		{ 135, 143 },
		{ 134, 141 },
		{ 134, 142 },
		{ 135, 144 },
		{ 135, 142 },
		{ 102, 125 },
		{ 133, 127 },
		{ 166, 180 },
		{ 166, 165 },
		{ 133, 105 },
		{ 133, 102 },
		{ 174, 186 },
		{ 33, 70 },
		{ 173, 185 },
		{ 86, 85 },
		{ 152, 166 },
		{ 152, 167 },
		{ 150, 163 },
		{ 150, 164 },
		{ 114, 133 },
		{ 114, 104 },
		{ 89, 106 },
		{ 89, 97 },
		{ 80, 86 },
		{ 80, 87 },
		{ 33, 71 },
		{ 0, 33 },
		{ 0, 34 },
		{ 172, 184 },
		{ 171, 183 },
		{ 163, 178 },
		{ 157, 170 },
		{ 156, 169 },
		{ 155, 168 },
		{ 148, 160 },
		{ 145, 154 },
		{ 138, 147 },
		{ 137, 146 },
		{ 124, 135 },
		{ 123, 134 },
		{ 106, 113 },
		{ 94, 111 },
		{ 93, 110 },
		{ 92, 109 },
		{ 91, 108 },
		{ 90, 107 },
		{ 88, 103 },
		{ 85, 98 },
		{ 83, 96 },
		{ 82, 89 },
		{ 70, 83 },
		{ 69, 82 },
		{ 67, 81 },
		{ 66, 80 }
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
		{ 26, 86 },
		{ 0, -1 },
		{ 56, -1 },
		{ 29, 89 },
		{ 0, -1 },
		{ 48, -1 },
		{ -4, -1 },
		{ 0, -1 },
		{ 46, 114 },
		{ 2, -1 },
		{ 24, -1 },
		{ 22, -1 },
		{ 20, -1 },
		{ 18, -1 },
		{ 16, -1 },
		{ 0, -1 },
		{ 0, 106 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ 0, -1 },
		{ -13, -1 },
		{ 0, 133 },
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
		{ 19, 133 },
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
		{ -9, 149 },
		{ -7, 149 },
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
		{ -1, 166 },
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
