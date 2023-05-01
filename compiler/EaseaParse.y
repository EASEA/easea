%{
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
#include "EaseaParse.hpp"

#define YYEXIT_FAILURE	1
#define YYEXIT_SUCCESS	0

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
int nPOP_SIZE, nOFF_SIZE, nARCH_SIZE;
float fSURV_PAR_SIZE=-1.0, fSURV_OFF_SIZE=-1.0;
const char *nGENOME_NAME;
int nPROBLEM_DIM;
int nNB_GEN=0;
int nNB_OPT_IT=0;
int nTIME_LIMIT=0;
int nSERVER_PORT=0;
float fMUT_PROB;
float fXOVER_PROB;
FILE *fpOutputFile, *fpTemplateFile, *fpGenomeFile;//, *fpExplodedGenomeFile;

CSymbolTable SymbolTable;    // the symbol table

unsigned iMAX_INIT_TREE_D,iMIN_INIT_TREE_D,iMAX_TREE_D,iNB_GPU,iPRG_BUF_SIZE,iMAX_TREE_DEPTH,iMAX_XOVER_DEPTH,iNO_FITNESS_CASES;

extern int yylex();
extern char * yytext;

void yyerror(const char * s){
  printf("%s\nSyntax Error at line : %d (on text : %s)\nFor more details during the EASEA compiling, use the \"-v\" option\n",
	 s,yylineno,yytext);
}

char* concat_delete(char* lhs, char* rhs) {
	int len = strlen(lhs) + strlen(rhs) + 1;
	char* ret = new char[len]();
	ret = strncat(ret, lhs, len);
	ret = strncat(ret, rhs, len);
	delete[] lhs;
	return ret;
}

char* sdup(const char* str) {
	int len = strlen(str)+1;
	char* ret = new char[len];
	memcpy(ret, str, len);
	return ret;
}

char* itos(int n) {
	char buf[32];
	snprintf(buf, 32, "%d", n);
	return sdup(buf);
}

%}

%code provides {
// forward references
class CSymbol;
}

%code requires {
#include <memory>
}

// attribute type
%union {
  CSymbol* pSymbol;
  double dValue;
  int ObjectQualifier;
  int nValue;
  char *szString;
}

// nonterminals
%type <dValue> Expr
%type <ObjectQualifier> Qualifier
%type <pSymbol> BaseType
%type <pSymbol> UserType
%type <pSymbol> Symbol
%type <szString> Template TemplateType TemplateList

// tokens
%right '='
%left '+' '-'
%left '*' '/'
%right UMINUS

// keywords

// .ez file tokens
%token CLASSES
%token GENOME
%token USER_CTOR
%token USER_XOVER
%token USER_MUTATOR
%token USER_EVALUATOR                   
%token USER_OPTIMISER
%token MAKEFILE_OPTION
%token END_OF_FUNCTION                   
//%token DELETE
%token <szString> END_METHODS
%token <pSymbol> IDENTIFIER
%token <pSymbol> IDENTIFIER2
%token <pSymbol> BOOL
%token <pSymbol> INT
%token <pSymbol> DOUBLE
%token <pSymbol> FLOAT
%token <pSymbol> GPNODE
%token <pSymbol> CHAR
%token <pSymbol> POINTER
%token <dValue> NUMBER
%token <dValue> NUMBER2
%token METHODS
%token STATIC       
%token NB_GEN       
%token NB_OPT_IT //Memetic 
%token BALDWINISM //Memetic
%token MUT_PROB
%token XOVER_PROB                       
%token POP_SIZE
%token ARCH_SIZE
%token SELECTOR
%token RED_PAR
%token RED_OFF
%token RED_FINAL
%token OFFSPRING
%token SURVPAR
%token SURVOFF
%token MINIMAXI
%token ELITISM
%token ELITE
%token REMOTE_ISLAND_MODEL //island model
%token IP_FILE  //island model
%token MIGRATION_PROBABILITY //island model
%token SERVER_PORT //server port
%token PRINT_STATS
%token PLOT_STATS
%token GENERATE_CSV_FILE
%token GENERATE_GNUPLOT_SCRIPT
%token GENERATE_R_SCRIPT
%token SAVE_POPULATION
%token START_FROM_FILE
%token TIME_LIMIT
%token MAX_INIT_TREE_D
%token MIN_INIT_TREE_D
%token MAX_XOVER_DEPTH
%token MAX_MUTAT_DEPTH
%token MAX_TREE_D
%token NB_GPU
%token PRG_BUF_SIZE
%token NO_FITNESS_CASES
%token TEMPLATE_END

// include file
%code provides {
#include "EaseaSym.h"
#include "EaseaLex.h"

  int easeaParse(int argc, char *argv[]);
  int  CEASEAParser_create();
  
  double  CEASEAParser_assign(CSymbol* pIdentifier, double dValue);
  double  CEASEAParser_divide(double dDividend, double dDivisor);
  CSymbol*  CEASEAParser_insert();

  //virtual void yysyntaxerror();
}

%start EASEA
%%

EASEA :  RunParameters GenomeAnalysis TEMPLATE_END {return 0;};

GenomeAnalysis
    : ClassDeclarationsSection GenomeDeclarationSection {
        if (bVERBOSE){ printf("                    _______________________________________\n");
        printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);}
      }
      StandardFunctionsAnalysis
    | GenomeDeclarationSection {
        if (bVERBOSE) printf("                    _______________________________________\n");
        if (bVERBOSE) printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);
      }
      StandardFunctionsAnalysis
  ;

ClassDeclarationsSection
  : CLASSES {
    if (bVERBOSE) printf("Declaration of user classes :\n\n");} 
    ClassDeclarations
  | CLASSES {
      if (bVERBOSE) printf("No user class declaration found other than GenomeClass.\n");} 
  ;
  
ClassDeclarations
: ClassDeclaration 
| ClassDeclarations ClassDeclaration
  ;
  
ClassDeclaration
  : Symbol {
      $1->ObjectType=oUserClass;
      pCLASSES[nClasses_nb++] = $1;
      //auto sym = std::unique_ptr<CSymbol>{$1};
      pCURRENT_CLASS = $1;
      //pCURRENT_CLASS=SymbolTable.insert(std::move(sym));  
      //DEBUG_PRT("Yacc Symbol declaration %s %d",$1->sName.c_str(),$1->nSize);
    }
  '{' VariablesDeclarations '}' {
      if (bVERBOSE) printf("Class %s declared for %d bytes.\n\n",$1->sName.c_str(),$1->nSize);
      //DEBUG_PRT("Yacc variable declaration %s %d",$1->sName.c_str(),$1->nSize);
    }                       
  ;

VariablesDeclarations
  : VariablesDeclaration    
  | VariablesDeclarations VariablesDeclaration    
  ;                             
  
// TypeSize initialisation in a separate rule to avoid reduce/reduce conflict
VariablesDeclaration
  : Qualifier BaseType {pCURRENT_TYPE=$2; pCURRENT_TYPE->ObjectQualifier=$1;} BaseObjects {}
  | Qualifier UserType {pCURRENT_TYPE=$2; pCURRENT_TYPE->ObjectQualifier=$1;} UserObjects {}
  | MethodsDeclaration
  ;

MethodsDeclaration
  : METHODS END_METHODS{
    pCURRENT_CLASS->sString = std::string($2);
    if (bVERBOSE) printf("\n    The following methods have been declared:\n\n%s\n\n",pCURRENT_CLASS->sString.c_str());
    }
  ;

Qualifier
  : STATIC {$$=1;}  // 1=Static
  | {$$=0;} // 0=Normal
  ;
  
BaseObjects
  : Objects ';'
  | Objects ':' BaseConstructorParameters ';' {}
  ;

UserObjects
  : Objects ';' {}
  | Objects ':' UserConstructorParameters ';' {}
  ;
  
BaseType
  : BOOL
  | INT
  | DOUBLE
  | FLOAT
  | CHAR
  | POINTER
  | GPNODE
  ;
  
UserType
  : Symbol {  
      CSymbol *pSym=SymbolTable.find($1->sName.c_str());
      if (pSym==NULL) {
        /*fprintf(stderr,"\n%s - Error line %d: Class \"%s\" was not defined.\n",sEZ_FILE_NAME,yylineno,$1->sName.c_str());
        fprintf(stderr,"Only base types (bool, int, float, double, char) or new user classes defined\nwithin the \"User classes\" sections are allowed.\n");
        exit(1);*/
	auto pNewBaseType=std::make_unique<CSymbol>($1->sName.c_str());
  	pNewBaseType->nSize=0;
  	pNewBaseType->ObjectType=oObject;
  	SymbolTable.insert(std::move(pNewBaseType));
	pSym = SymbolTable.find($1->sName.c_str());
      }       
      else $$=pSym;
    }
  | Symbol '<' Template {
      char* buf = concat_delete(concat_delete(sdup($1->sName.c_str()), sdup("<")), $3);
      CSymbol *pSym=SymbolTable.find(buf);
      if (pSym==NULL) {
        fprintf(stderr, "DEBUG: %s\n", buf);
	auto pNewBaseType=std::make_unique<CSymbol>(buf);
  	pNewBaseType->nSize=0;
  	pNewBaseType->ObjectType=oObject;
  	SymbolTable.insert(std::move(pNewBaseType));
	pSym = SymbolTable.find(buf);
      }
      $$=pSym;
    }
  ;

TemplateType
  : BaseType { $$ = sdup($1->sName.c_str()); }
  | UserType { $$ = sdup($1->sName.c_str()); }
  | NUMBER { $$ = itos($1); }

TemplateList
  : ',' TemplateType '>' {$$ = concat_delete(concat_delete(sdup(","), $2), sdup(">")); }
  | ',' TemplateType TemplateList { $$ = concat_delete(concat_delete(sdup(","), $2), $3); }

Template
  : TemplateType '>' { $$ = concat_delete($1, sdup(">")); }
  | TemplateType TemplateList { $$ = concat_delete($1, $2);}
  | '>' { $$ = sdup(">"); }

Objects
  : Object
  | Objects ',' Object
  ;

// Attention : il reste � g�rer correctement les tableaux de pointeurs
// les indirections multiples et les tableaux � plusieurs dimensions.
// Je sais, il faudrait aussi utiliser un peu de polymorphisme pour les symboles
  
Object
  : Symbol {

      auto uptr = std::make_unique<CSymbol>($1->sName.c_str());
        uptr->nSize=pCURRENT_TYPE->nSize;
        uptr->pClass=pCURRENT_CLASS;
        uptr->pType=pCURRENT_TYPE;
        uptr->ObjectType=oObject;
        uptr->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
        pCURRENT_CLASS->nSize+=uptr->nSize;
        if (bVERBOSE) printf("    %s variable declared (%d bytes)\n",uptr->sName.c_str(),uptr->nSize);
        pCURRENT_CLASS->pSymbolList.push_front(std::move(uptr));
    } 
  | '*' Symbol {
      auto uptr = std::make_unique<CSymbol>($2->sName.c_str());
      uptr->nSize=sizeof (char *);
      uptr->pClass=pCURRENT_CLASS;
      uptr->pType=pCURRENT_TYPE;
      uptr->ObjectType=oPointer;
      uptr->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=uptr->nSize;
      if (bVERBOSE) printf("    %s pointer declared (%d bytes)\n",uptr->sName.c_str(),uptr->nSize);
      pCURRENT_CLASS->pSymbolList.push_front(std::move(uptr));
    }
  | '0' Symbol {
      auto uptr = std::make_unique<CSymbol>($2->sName.c_str());
      uptr->nSize=sizeof (char *);
      uptr->pClass=pCURRENT_CLASS;
      uptr->pType=pCURRENT_TYPE;
      uptr->ObjectType=oPointer;
      uptr->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=uptr->nSize;
      if (bVERBOSE) printf("    %s NULL pointer declared (%d bytes)\n",uptr->sName.c_str(),uptr->nSize);
      pCURRENT_CLASS->pSymbolList.push_front(std::move(uptr));
    }
  | '*''*' Symbol {
      auto uptr = std::make_unique<CSymbol>($3->sName.c_str());
      uptr->nSize=sizeof (char *);
      uptr->pClass=pCURRENT_CLASS;
      uptr->pType=pCURRENT_TYPE;
      uptr->ObjectType=oPointer;
      uptr->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=uptr->nSize;
      if (bVERBOSE) printf("    %s pointer of pointer declared (%d bytes)\n",uptr->sName.c_str(),uptr->nSize);
      pCURRENT_CLASS->pSymbolList.push_front(std::move(uptr));
      fprintf(stderr,"Pointer of pointer doesn't work properly yet\n");
      exit(-1);
    }

  | Symbol  '[' Expr ']' {
      if((TARGET_FLAVOR==CMAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName.c_str(),"Genome")==0) { nGENOME_NAME=$1->sName.c_str(); nPROBLEM_DIM=(int)$3;}
if((TARGET_FLAVOR==QAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName.c_str(),"Genome")==0) { nGENOME_NAME=$1->sName.c_str(); nPROBLEM_DIM=(int)$3;}
if((TARGET_FLAVOR==CUDA_QAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName.c_str(),"Genome")==0) { nGENOME_NAME=$1->sName.c_str(); nPROBLEM_DIM=(int)$3;}
if((TARGET_FLAVOR==QIEA) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName.c_str(),"Genome")==0) { nGENOME_NAME=$1->sName.c_str(); nPROBLEM_DIM=(int)$3;}

      //printf("DEBUG : size of $3 %d nSize %d\n",(int)$3,pCURRENT_TYPE->nSize);

      auto uptr = std::make_unique<CSymbol>($1->sName.c_str());
      uptr->nSize=pCURRENT_TYPE->nSize*(int)$3;
      uptr->pClass=pCURRENT_CLASS;
      uptr->pType=pCURRENT_TYPE;
      uptr->ObjectType=oArray;
      uptr->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=uptr->nSize;
      if (bVERBOSE) printf("    %s array declared (%d bytes)\n",uptr->sName.c_str(),uptr->nSize);
      pCURRENT_CLASS->pSymbolList.push_front(std::move(uptr));
    }
  | '*' Symbol  '[' Expr ']' {

    // this is for support of pointer array. This should be done in a more generic way in a later version
      if((TARGET_FLAVOR==CMAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName.c_str(),"Genome")==0) { 
	nGENOME_NAME=$2->sName.c_str(); nPROBLEM_DIM=(int)$4;
      }
if((TARGET_FLAVOR==QAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName.c_str(),"Genome")==0) {
    nGENOME_NAME=$2->sName.c_str(); nPROBLEM_DIM=(int)$4;}
if((TARGET_FLAVOR==CUDA_QAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName.c_str(),"Genome")==0) {
        nGENOME_NAME=$2->sName.c_str(); nPROBLEM_DIM=(int)$4;}
if((TARGET_FLAVOR==QIEA) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName.c_str(),"Genome")==0) {
      nGENOME_NAME=$2->sName.c_str(); nPROBLEM_DIM=(int)$4;

}

      
      //pCURRENT_CLASS->nSize

      auto uptr = std::make_unique<CSymbol>($2->sName.c_str());
      uptr->nSize=sizeof(char*)*(int)$4;
      uptr->pClass=pCURRENT_CLASS;
      uptr->pType=pCURRENT_TYPE;
      uptr->ObjectType=oArrayPointer;
      uptr->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=uptr->nSize;
      if (bVERBOSE) printf("    %s array of pointers declared (%d bytes)\n",uptr->sName.c_str(),uptr->nSize);
      pCURRENT_CLASS->pSymbolList.push_front(std::move(uptr));
    }
  ;  

BaseConstructorParameters                           
  : BaseConstructorParameter
  | BaseConstructorParameters BaseConstructorParameter
  ;                                                   

BaseConstructorParameter 
  : NUMBER {}
  ;
  
//Methods
//  : Method
//  | Methods Method
//  ;

//Method
//  : NUMBER {}
//  ;
  
GenomeDeclarationSection
  : GENOME {
    ////DEBUG_PRT("Yacc genome decl %s",$1.pSymbol->sName.c_str());
      pGENOME=new CSymbol("Genome");
      auto uptr = std::unique_ptr<CSymbol>(pGENOME);
      pCURRENT_CLASS=SymbolTable.insert(std::move(uptr));
      pGENOME->ObjectType=oUserClass;
      pGENOME->ObjectQualifier=0;
    }
'{' VariablesDeclarations '}' {}
  ;

//GenomeMethodsDeclaration
//  : GENOME_METHODS GENOME_END_METHODS{
//    pCURRENT_CLASS->sString = new char[strlen($2) + 1];
//    strcpy(pCURRENT_CLASS->sString, $2);      
//    }
//  |
//  ;

UserConstructorParameters
  : UserConstructorParameter
  | UserConstructorParameters UserConstructorParameter
  ;

UserConstructorParameter
  : Symbol {}
  ;
                         
Symbol                                        
  : IDENTIFIER {$$=$1;}
  ;                                             
  
StandardFunctionsAnalysis
  : StandardFunctionAnalysis
  | StandardFunctionsAnalysis StandardFunctionAnalysis
  ;
  
StandardFunctionAnalysis
  : USER_CTOR {         
      if (bVERBOSE) printf("Inserting genome initialiser (taken from .ez file).\n");
    } 
    END_OF_FUNCTION {} 
  | USER_XOVER {
      if (bVERBOSE) printf("Inserting user genome crossover (taken from .ez file).\n");
    } 
    END_OF_FUNCTION {} 
  | USER_MUTATOR {
      if (bVERBOSE) printf("Inserting user genome mutator (taken from .ez file).\n");
    } 
    END_OF_FUNCTION {}
  | USER_EVALUATOR { 
      if (bVERBOSE) printf("Inserting user genome evaluator (taken from .ez file).\n");
    } 
    END_OF_FUNCTION {}
  | USER_OPTIMISER { 
      if (bVERBOSE) printf("Inserting user genome optimiser (taken from .ez file).\n");
    } 
    END_OF_FUNCTION {}
   | MAKEFILE_OPTION END_OF_FUNCTION {
     //DEBUG_PRT("User makefile options have been reduced");
     }
   | MAKEFILE_OPTION {}
  ;

RunParameters
  : Parameter
  | RunParameters Parameter
  ;           
  
Parameter
  :  NB_GEN NUMBER2
      {nNB_GEN=(int)$2;}
  |  NB_OPT_IT NUMBER2
      {nNB_OPT_IT=(int)$2;}
  |  TIME_LIMIT NUMBER2
      {nTIME_LIMIT=(int)$2;}
  |  MUT_PROB NUMBER2
      {fMUT_PROB=(float)$2;}
  |  XOVER_PROB NUMBER2
      {fXOVER_PROB=(float)$2;}
  |  POP_SIZE NUMBER2
      {nPOP_SIZE=(int)$2;}
  |  ARCH_SIZE NUMBER2
      {nARCH_SIZE=(int)$2;}
  |  SELECTOR IDENTIFIER2{
      strcpy(sSELECTOR, $2->sName.c_str());
      strcpy(sSELECTOR_OPERATOR, $2->sName.c_str());
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelector(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME);
	break;
      }
    }
  |  SELECTOR IDENTIFIER2 NUMBER2 {
      sprintf(sSELECTOR, "%s", $2->sName.c_str());   
      sprintf(sSELECTOR_OPERATOR, "%s", $2->sName.c_str());   
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelectorArgument(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME,(float)$3);
	break;
      }
    }
  |  RED_PAR IDENTIFIER2{
        sprintf(sRED_PAR, "%s", $2->sName.c_str());
	sprintf(sRED_PAR_OPERATOR, "%s", $2->sName.c_str());
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME);
	  break;
	}
    }
|  RED_PAR IDENTIFIER2 NUMBER2 {
        sprintf(sRED_PAR, "%s", $2->sName.c_str());
	sprintf(sRED_PAR_OPERATOR, "%s", $2->sName.c_str());
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME,(float)$3);
	  break;
	}
    }
  |  RED_OFF IDENTIFIER2{
	sprintf(sRED_OFF, "%s", $2->sName.c_str());
	sprintf(sRED_OFF_OPERATOR, "%s", $2->sName.c_str());
      switch (TARGET) {
      case STD:
      case CUDA:
	pickupSTDSelector(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME);
	break;
      }
    }
  |  RED_OFF IDENTIFIER2 NUMBER2 {
        sprintf(sRED_OFF, "%s", $2->sName.c_str());
	sprintf(sRED_OFF_OPERATOR, "%s", $2->sName.c_str());
        switch (TARGET) {
	case STD:
	case CUDA:
	  pickupSTDSelectorArgument(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME,$3);
       }}
  |  RED_FINAL IDENTIFIER2{
        sprintf(sRED_FINAL, "%s", $2->sName.c_str());
        sprintf(sRED_FINAL_OPERATOR, "%s", $2->sName.c_str());
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME);
	  break;
       }}
  |  RED_FINAL IDENTIFIER2 NUMBER2 {
        sprintf(sRED_FINAL, "%s", $2->sName.c_str());
        sprintf(sRED_FINAL_OPERATOR, "%s", $2->sName.c_str());
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME,$3);
	  break;
	}}
  |  OFFSPRING NUMBER2 {nOFF_SIZE=(int)$2;}
  |  OFFSPRING NUMBER2 '%' {nOFF_SIZE=(int)($2*nPOP_SIZE/100);}
  |  SURVPAR NUMBER2 {fSURV_PAR_SIZE=(float)$2;}
  |  SURVPAR NUMBER2 '%' {fSURV_PAR_SIZE=(float)($2/100);}
  |  SURVOFF NUMBER2 {fSURV_OFF_SIZE=(float)$2;}
  |  SURVOFF NUMBER2 '%' {fSURV_OFF_SIZE=(float)($2/100);}
  |  MINIMAXI IDENTIFIER2 {
      if ((!mystricmp($2->sName.c_str(),"Maximise")) || (!mystricmp($2->sName.c_str(),"Maximize"))) nMINIMISE=0;
      else if ((!mystricmp($2->sName.c_str(),"Minimise")) || (!mystricmp($2->sName.c_str(),"Minimize"))) nMINIMISE=1;
      else {
         fprintf(stderr,"\n%s - Error line %d: The evaluator goal default parameter can only take\n",sEZ_FILE_NAME,yylineno);
         fprintf(stderr,"two values : maximi[sz]e or minimi[sz]e.\n",sEZ_FILE_NAME,yylineno);
         exit(1);
       }
      }
  |  ELITE NUMBER2 {
        nELITE=(int)$2;
        }
  |  ELITE NUMBER2 '%' {
        nELITE=(int)$2*nPOP_SIZE/100;
        }
  |  ELITISM IDENTIFIER2{
      if (!mystricmp($2->sName.c_str(),"Weak")) bELITISM=0;
      else if (!mystricmp($2->sName.c_str(),"Strong")) bELITISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Elitism must be \"Strong\" or \"Weak\".\nDefault value \"Strong\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bELITISM=1;
       }}
  | BALDWINISM IDENTIFIER2{
      if (!mystricmp($2->sName.c_str(),"False")) bBALDWINISM=0;
      else if (!mystricmp($2->sName.c_str(),"True")) bBALDWINISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Baldwinism must be \"True\" or \"False\".\nDefault value \"True\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bBALDWINISM=1;
       }}

  | REMOTE_ISLAND_MODEL IDENTIFIER2{
	if (!mystricmp($2->sName.c_str(),"False")) bREMOTE_ISLAND_MODEL=0;
	else if (!mystricmp($2->sName.c_str(),"True")) bREMOTE_ISLAND_MODEL=1;
	else {
	  fprintf(stderr,"\n%s - Warning line %d: remote island model must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n",sEZ_FILE_NAME,yylineno);nWARNINGS++;
	  bREMOTE_ISLAND_MODEL=0;
	}}
  | IP_FILE IDENTIFIER2'.'IDENTIFIER2{
        sprintf(sIP_FILE, $2->sName.c_str());
	strcat(sIP_FILE,".");
	strcat(sIP_FILE,$4->sName.c_str());
	}
  | MIGRATION_PROBABILITY NUMBER2{
	fMIGRATION_PROBABILITY=(float)$2;
	}
  | SERVER_PORT NUMBER2{
      nSERVER_PORT=(int)$2;
    }
  | PRINT_STATS IDENTIFIER2{
      if (!mystricmp($2->sName.c_str(),"False")) bPRINT_STATS=0;
      else if (!mystricmp($2->sName.c_str(),"True")) bPRINT_STATS=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Print stats must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bPRINT_STATS=0;
       }}
  | PLOT_STATS IDENTIFIER2{
      if (!mystricmp($2->sName.c_str(),"False")) bPLOT_STATS=0;
      else if (!mystricmp($2->sName.c_str(),"True")) bPLOT_STATS=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate stats must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bPLOT_STATS=0;
       }}
  | GENERATE_CSV_FILE IDENTIFIER2{
      if (!mystricmp($2->sName.c_str(),"False")) bGENERATE_CSV_FILE=0;
      else if (!mystricmp($2->sName.c_str(),"True")) bGENERATE_CSV_FILE=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate csv file must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bGENERATE_CSV_FILE=0;
       }}
  | GENERATE_GNUPLOT_SCRIPT IDENTIFIER2{
      if (!mystricmp($2->sName.c_str(),"False")) bGENERATE_GNUPLOT_SCRIPT=0;
      else if (!mystricmp($2->sName.c_str(),"True")) bGENERATE_GNUPLOT_SCRIPT=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate gnuplot script must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bGENERATE_GNUPLOT_SCRIPT=0;
       }}
  | GENERATE_R_SCRIPT IDENTIFIER2{
      if (!mystricmp($2->sName.c_str(),"False")) bGENERATE_R_SCRIPT=0;
      else if (!mystricmp($2->sName.c_str(),"True")) bGENERATE_R_SCRIPT=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate R script must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bGENERATE_R_SCRIPT=0;
       }}
  | SAVE_POPULATION IDENTIFIER2{
      if (!mystricmp($2->sName.c_str(),"False")) bSAVE_POPULATION=0;
      else if (!mystricmp($2->sName.c_str(),"True")) bSAVE_POPULATION=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: SavePopulation must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bSAVE_POPULATION=0;
       }}
  | START_FROM_FILE IDENTIFIER2{
      if (!mystricmp($2->sName.c_str(),"False")) bSTART_FROM_FILE=0;
      else if (!mystricmp($2->sName.c_str(),"True")) bSTART_FROM_FILE=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: StartFromFile must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bSTART_FROM_FILE=0;
       }}
| MAX_INIT_TREE_D NUMBER2 {iMAX_INIT_TREE_D = (unsigned)$2;}
| MIN_INIT_TREE_D NUMBER2 {iMIN_INIT_TREE_D = (unsigned)$2;}
| MAX_TREE_D NUMBER2 {iMAX_TREE_D = (unsigned)$2;}
| NB_GPU NUMBER2 {iNB_GPU = (unsigned)$2;}
| PRG_BUF_SIZE NUMBER2 {iPRG_BUF_SIZE = (unsigned)$2;}
//| MAX_XOVER_DEPTH NUMBER2 {iMAX_TREE_DEPTH = (unsigned)$2;}
 //| MAX_MUTAT_DEPTH NUMBER2 {iMAX_MUTAT_DEPTH = (unsigned)$2;}  
| NO_FITNESS_CASES NUMBER2 {iNO_FITNESS_CASES = (unsigned)$2;}
  ;
  
Expr
  : IDENTIFIER '=' Expr { 
      $$ = CEASEAParser_assign($1, $3);
    }
  | Expr '+' Expr                    { $$ = $1 + $3; }
  | Expr '-' Expr                     { $$ = $1 - $3; }
  | Expr '*' Expr                     { $$ = $1 * $3; }
  | Expr '/' Expr                     { /* CEASEAParser_divide can't be used because g++ can't goto the label YYERROR go,
                                           So I directely use its code. I don't know why g++ can't compile CEASEAParser_divide */
                                        if($3 == 0) {$$=0; YYERROR; return 0;}
                                        else $$ = $1 / $3; }
  | '(' Expr ')'                          { $$ = $2; }
  | '-' Expr %prec UMINUS { $$ = -$2; }
  | NUMBER                         { $$ = $1; }
  | IDENTIFIER{
      $$ = $1->dValue;
    }
  ;

%%
                       
/////////////////////////////////////////////////////////////////////////////
// main
#include <version.h>

int easeaParse(int argc, char *argv[]){
  int n = YYEXIT_FAILURE;
  int nParamNb=0;
  char *sTemp;
  int i=0;
  
  TARGET=STD;
  bVERBOSE=0;
  sRAW_PROJECT_NAME[0]=0; // used to ask for a filename if no filename is found on the command line.
    
    std::cout << "EASENA version: " << easea::version::as_string() << std::endl;

  while ((++nParamNb) < argc) {
    sTemp=&(argv[nParamNb][0]);
    if ((argv[nParamNb][0]=='-')||(argv[nParamNb][0]=='/')) sTemp=&(argv[nParamNb][1]);
    if (!mystricmp(sTemp,"cuda")){
      TARGET=CUDA;
      printf("Compiled with CUDA template\n");
      TARGET_FLAVOR = CUDA_FLAVOR_SO;
    }
    else if( !mystricmp(sTemp,"cuda_mo") ){
      TARGET=CUDA;
      TARGET_FLAVOR = CUDA_FLAVOR_MO;
    }
    else if( !mystricmp(sTemp,"cuda_gp") ){
      printf("Compiled with CUDA GP template\n");
      TARGET=CUDA;
      TARGET_FLAVOR = FLAVOR_GP;
    }
    else if( !mystricmp(sTemp,"gp") ){
      printf("Compiled with GP template\n");
      TARGET=STD;
      TARGET_FLAVOR = FLAVOR_GP;
    }
    else if (!mystricmp(sTemp,"std"))  {
      TARGET=STD;
      printf("Compiled with STD template\n");
      TARGET_FLAVOR = STD_FLAVOR_SO;
    }
    else if (!mystricmp(sTemp,"std_mo")) {
      TARGET=STD;
      TARGET_FLAVOR = STD_FLAVOR_MO;
    }
    else if (!mystricmp(sTemp,"cmaes"))  {
      printf("Compiled with CMAES template\n");
      TARGET_FLAVOR = CMAES;
    }
    else if (!mystricmp(sTemp,"de"))  {
      printf("Compiled with DE template\n");
      TARGET_FLAVOR = DE;
    }
    else if (!mystricmp(sTemp,"gde"))  {
           printf("Compiled with GDE template\n");
           TARGET_FLAVOR = GDE;
    }
    else if (!mystricmp(sTemp,"qaes"))  {
      printf("Compiled with QAES template\n");
      TARGET_FLAVOR = QAES;
    }
    else if (!mystricmp(sTemp, "cuda_qaes")){
	printf("Compiled with CUDA_QAES template\n");
	TARGET = CUDA;
	TARGET_FLAVOR = CUDA_QAES;
    }
    else if (!mystricmp(sTemp,"memetic"))  {
      TARGET_FLAVOR = MEMETIC;
    }
    else if (!mystricmp(sTemp,"nsgaii"))  {
      printf("Compiled with NSGAII template\n");
      TARGET_FLAVOR = NSGAII;
    }
    else if (!mystricmp(sTemp,"asrea"))  {
      printf("Compiled with ASREA template\n");
      TARGET_FLAVOR = ASREA;
    }
    else if (!mystricmp(sTemp,"fastemo"))  {
      printf("Compiled with FASTEMO template\n");
      TARGET_FLAVOR = FASTEMO;
    }
        else if (!mystricmp(sTemp,"fastemoii"))  {
           TARGET_FLAVOR = FASTEMOII;
     }
    else if (!mystricmp(sTemp,"nsgaiii"))  {
	   printf("Compiled with NSGAIII template\n");
           TARGET_FLAVOR = NSGAIII;
         }
    else if (!mystricmp(sTemp,"ibea"))  {
	    printf("Compiled with IBEA template\n");
            TARGET_FLAVOR = IBEA;
     }
    else if (!mystricmp(sTemp,"moead"))  {
		printf("Compiled with MOEAD template\n");
                TARGET_FLAVOR = MOEAD;
     }
     else if (!mystricmp(sTemp,"qiea"))  {
		printf("Compiled with QIEA template\n");
                 TARGET_FLAVOR = QIEA;
      }
    else if (!mystricmp(sTemp,"cdas"))  {
	    printf("Compiled with CDAS template\n");
             TARGET_FLAVOR = CDAS;
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

  CEASEAParser_create();
  n = yyparse();
  exit(n);
  return n;
}

/////////////////////////////////////////////////////////////////////////////
// EASEAParser commands

#define helper_create(sym) { auto ptr = std::make_unique<CSymbol>(#sym); ptr->nSize = sizeof(sym); ptr->ObjectType = oBaseClass; SymbolTable.insert(std::move(ptr)); }

int CEASEAParser_create()
{
  helper_create(bool);
  helper_create(int);
  helper_create(double);
  helper_create(float);
  helper_create(char);
  helper_create(char*);

  auto ptr = std::make_unique<CSymbol>("pointer");
  ptr->nSize=sizeof(char *);   
  ptr->ObjectType=oBaseClass;
  SymbolTable.insert(std::move(ptr));


  //if (!yycreate(&EASEALexer)) {
  //  return 0;
  //}
  if (!CEASEALexer_create(&SymbolTable)) {
    return 0;
  }
  return 1; // success
}

/////////////////////////////////////////////////////////////////////////////
// calc_parser attribute commands

double CEASEAParser_assign(CSymbol* spIdentifier, double dValue)
{
  assert(spIdentifier != NULL);

  spIdentifier->dValue = dValue;
  return spIdentifier->dValue;
}

/*double CEASEAParser_divide(double a, double b)
{
  if (b == 0) {
    printf("division by zero\n");
    YYERROR;
    return 0;
  }
  else {
    return a / b;
  }
}
*/

void CEASEAParser_yysyntaxerror(){

  printf("Syntax Error at line : %d (on text : %s)\nFor more details during the EASEA compiling, use the \"-v\" option\n",
	 yylineno,yytext);
}

