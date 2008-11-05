%{
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

%}

// include file
%include {
// forward references
class CSymbol;
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

// tokens
%right '='
%left '+', '-'
%left '*', '/'
%right UMINUS

// keywords

// .ez file tokens
%token CLASSES
%token GENOME
%token USER_CTOR
%token USER_XOVER
%token USER_MUTATOR
%token USER_EVALUATOR                   
%token END_OF_FUNCTION                   
//%token DELETE
%token <szString> END_METHODS
%token <pSymbol> IDENTIFIER
%token <pSymbol> IDENTIFIER2
%token <pSymbol> BOOL
%token <pSymbol> INT
%token <pSymbol> DOUBLE
%token <pSymbol> FLOAT
%token <pSymbol> CHAR
%token <pSymbol> POINTER
%token <dValue> NUMBER
%token <dValue> NUMBER2
%token METHODS
%token STATIC       
%token NB_GEN       
%token NB_ISLANDS
%token PROP_SEQ
%token MUT_PROB
%token XOVER_PROB                       
%token POP_SIZE
%token SELECTOR
%token RED_PAR
%token RED_OFF
%token RED_FINAL
%token OFFSPRING
%token SURVPAR
%token SURVOFF
%token REPLACEMENT
%token DISCARD
%token MINIMAXI
%token ELITISM
%token ELITE
%token MIG_CLONE
%token MIG_SEL
%token MIGRATOR
%token MIG_FREQ
%token NB_MIG
%token IMMIG_SEL
%token IMMIG_REPL

// include file
%include {
#include "EaseaSym.h"
#include "EaseaLex.h"
}

// parser name and class definition
%name CEASEAParser
{
protected:
  CEASEALexer EASEALexer;       // the lexical analyser
  
public:
  CSymbolTable SymbolTable;    // the symbol table

  int create();
  
  double assign(CSymbol* pIdentifier, double dValue);
  double divide(double dDividend, double dDivisor);
  CSymbol* insert() const;
}

// constructor
{
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
}

%start EASEA
%%

EASEA :  RunParameters GenomeAnalysis;

GenomeAnalysis
    : ClassDeclarationsSection GenomeDeclarationSection {
        if (bVERBOSE) printf("                    _______________________________________\n");
        if ((TARGET==DREAM)&& bVERBOSE) printf ("\nGeneration of the JAVA source files for %s.\n\n",sPROJECT_NAME);
        if ((TARGET!=DREAM)&& bVERBOSE) printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);
      }
      StandardFunctionsAnalysis
    | GenomeDeclarationSection {
        if (bVERBOSE) printf("                    _______________________________________\n");
        if ((TARGET==DREAM)&& bVERBOSE) printf ("\nGeneration of the JAVA source files for %s.\n\n",sPROJECT_NAME);
        if ((TARGET!=DREAM)&& bVERBOSE) printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);
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
      pCURRENT_CLASS=SymbolTable.insert($1);  
      pCURRENT_CLASS->pSymbolList=new CLList<CSymbol *>();
      $1->ObjectType=oUserClass;
    }
  '{' VariablesDeclarations '}' {
      if (bVERBOSE) printf("Class %s declared for %d bytes.\n\n",$1->sName,$1->nSize);
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
    pCURRENT_CLASS->sString = new char[strlen($2) + 1];
    strcpy(pCURRENT_CLASS->sString, $2);      
    if (bVERBOSE) printf("\n    The following methods have been declared:\n\n%s\n\n",pCURRENT_CLASS->sString);
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
  ;
  
UserType
  : Symbol {  
      CSymbol *pSym=SymbolTable.find($1->sName);
      if (pSym==NULL) {
        fprintf(stderr,"\n%s - Error line %d: Class \"%s\" was not defined.\n",sEZ_FILE_NAME,EASEALexer.yylineno,$1->sName);
        fprintf(stderr,"Only base types (bool, int, float, double, char) or new user classes defined\nwithin the \"User classes\" sections are allowed.\n");
        exit(1);
      }       
      else $$=pSym;
    }
  ;

Objects
  : Object
  | Objects ',' Object
  ;

// Attention : il reste à gérer correctement les tableaux de pointeurs
// les indirections multiples et les tableaux à plusieurs dimensions.
// Je sais, il faudrait aussi utiliser un peu de polymorphisme pour les symboles
  
Object
  : Symbol {
//      CSymbol *pSym;
//      pSym=$1;
      if (TARGET==DREAM){                             
        if (pCURRENT_TYPE->ObjectType==oBaseClass){
          $1->nSize=pCURRENT_TYPE->nSize;
          $1->pClass=pCURRENT_CLASS;
          $1->pType=pCURRENT_TYPE;
          $1->ObjectType=oObject;
          $1->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
          pCURRENT_CLASS->nSize+=$1->nSize;
          pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)($1));
          if (bVERBOSE) printf("    %s variable declared (%d bytes)\n",$1->sName,$1->nSize);
        }
        else {
          $1->nSize=sizeof (char *);
          $1->pClass=pCURRENT_CLASS;
          $1->pType=pCURRENT_TYPE;
          $1->ObjectType=oPointer;
          $1->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
          pCURRENT_CLASS->nSize+=$1->nSize;
          pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)($1));
          if (bVERBOSE) printf("    %s pointer declared (%d bytes)\n",$1->sName,$1->nSize);
      } }
      else {
        $1->nSize=pCURRENT_TYPE->nSize;
        $1->pClass=pCURRENT_CLASS;
        $1->pType=pCURRENT_TYPE;
        $1->ObjectType=oObject;
        $1->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
        pCURRENT_CLASS->nSize+=$1->nSize;
        pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)($1));
        if (bVERBOSE) printf("    %s variable declared (%d bytes)\n",$1->sName,$1->nSize);
    } }
  | '*' Symbol {
      $2->nSize=sizeof (char *);
      $2->pClass=pCURRENT_CLASS;
      $2->pType=pCURRENT_TYPE;
      $2->ObjectType=oPointer;
      $2->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=$2->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)($2));
      if (bVERBOSE) printf("    %s pointer declared (%d bytes)\n",$2->sName,$2->nSize);
    }
  | Symbol  '[' Expr ']' {
      $1->nSize=pCURRENT_TYPE->nSize*(int)$3;
      $1->pClass=pCURRENT_CLASS;
      $1->pType=pCURRENT_TYPE;
      $1->ObjectType=oArray;
      $1->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=$1->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)($1));
      if (bVERBOSE) printf("    %s array declared (%d bytes)\n",$1->sName,$1->nSize);
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
      if (bVERBOSE) printf ("\nGenome declaration analysis :\n\n");
      pGENOME=new CSymbol("Genome");
      pCURRENT_CLASS=SymbolTable.insert(pGENOME);  
      pGENOME->pSymbolList=new CLList<CSymbol *>();
      pGENOME->ObjectType=oUserClass;
      pGENOME->ObjectQualifier=0;
      pGENOME->sString=NULL;
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
      switch (TARGET) {
        case GALIB : 
          fprintf(fpOutputFile,"void %sGenome::Initializer(GAGenome& g) {\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  %sGenome & genome = (%sGenome &)g;\n",sPROJECT_NAME,sPROJECT_NAME);
          break;
        case EO :
          fprintf(fpOutputFile,"%sGenome InitialiserFunction() {\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  %sGenome *genome = new %sGenome;\n",sPROJECT_NAME,sPROJECT_NAME);
    } }
    END_OF_FUNCTION {                                    
      switch (TARGET) {
        case GALIB : fprintf(fpOutputFile,"genome._evaluated=gaFalse;\n}\n\n");break;
        case EO : fprintf(fpOutputFile,"return *genome;\n}");
    } }
  | USER_XOVER {
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
    } }
    END_OF_FUNCTION {
      switch (TARGET) {
        case GALIB : 
          fprintf(fpOutputFile,"  if(EZ_GeneratedChildren==3) {delete pBro; EZ_GeneratedChildren=1;}\n");
          fprintf(fpOutputFile,"  if(EZ_GeneratedChildren==4) {delete pSis; EZ_GeneratedChildren=1;}\n");
          fprintf(fpOutputFile,"return EZ_GeneratedChildren;\n}\n\n");break;
    } }
  | USER_MUTATOR {
      if (bVERBOSE) printf("Inserting user genome mutator (taken from .ez file).\n");
      switch (TARGET) {
        case GALIB : 
          fprintf(fpOutputFile,"int %sGenome::Mutator(GAGenome& g, float PMut) {\n",sPROJECT_NAME);
          fprintf(fpOutputFile,"  if (!GAFlipCoin((float)EZ_MUT_PROB)) return 0;\n");
          fprintf(fpOutputFile,"  %sGenome & genome = (%sGenome &)g;\n",sPROJECT_NAME,sPROJECT_NAME);
          fprintf(fpOutputFile,"  genome._evaluated=gaFalse;\n");
          break;
    } }
    END_OF_FUNCTION {   
      if (TARGET==DREAM) fprintf(fpOutputFile,"  }\n");
      fprintf(fpOutputFile,"}");
    }
  | USER_EVALUATOR { 
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
    } }
    END_OF_FUNCTION {
          if (TARGET!=EO) fprintf(fpOutputFile,"}\n");
    }
  ;

RunParameters
  : Parameter
  | RunParameters Parameter
  ;           
  
Parameter
  :  NB_GEN NUMBER2
      {nNB_GEN=(int)$2;}
  |  NB_ISLANDS NUMBER2
      {nNB_ISLANDS=(int)$2;}
  |  PROP_SEQ IDENTIFIER2{
      if (!mystricmp($2->sName,"proportionally")) bPROP_SEQ=true;
      else if (!mystricmp($2->sName,"sequentially")) bPROP_SEQ=false;
      else {
         fprintf(stderr,"\n%s - Error line %d: Looking for \"Proportionally\" or \"Sequentially\".\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
     } }
  |  MUT_PROB NUMBER2
      {fMUT_PROB=(float)$2;}
  |  XOVER_PROB NUMBER2
      {fXOVER_PROB=(float)$2;}
  |  POP_SIZE NUMBER2
      {nPOP_SIZE=(int)$2;}
  |  SELECTOR IDENTIFIER2{
      strcpy(sSELECTOR, $2->sName);
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
      }
  |  SELECTOR IDENTIFIER2 NUMBER2 {
      sprintf(sSELECTOR, $2->sName);   
      switch (TARGET) {
        case DREAM : if (!mystricmp(sSELECTOR,"EPTrn")){
                                  fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                  exit(1);
                                }
                                else if (!mystricmp(sSELECTOR,"Tournament")) {
                                  if ($3>=2) sprintf(sSELECTOR,"TournamentSelector(MAX, 1, %d)",(int)$3);
                                  else if (($3>.5)&&($3<=1.0)) sprintf(sSELECTOR,"TournamentSelector(MAX, %f, 2)",(float)$3);
                                  else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                }
                                else if (!mystricmp(sSELECTOR,"StochTrn")) {
                                  if (($3>.5)&&($3<=1.0)) sprintf(sSELECTOR,"TournamentSelector(MAX, %f, 2)",(float)$3);
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
                             if ($3>=2) {sprintf(sSELECTOR,"DetTour");sprintf(sSELECT_PRM,"(%d)",(int) $3);}
                             else if (($3>.5)&&($3<=1.0)) {sprintf(sSELECTOR,"StochTour");sprintf(sSELECT_PRM,"(%f)",(float) $3);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sSELECTOR,"RouletteWheel")) {
                            sprintf(sSELECTOR,"Roulette");
                            if ($3<1) {fprintf(stderr,"\n%s - Warning line %d: The parameter of RouletteWheel must be greater than one.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;}
                            else sprintf(sSELECT_PRM,"(%f)",(float) $3);
                          }
                          else if (!mystricmp(sSELECTOR,"Random")) {
                            sprintf(sSELECTOR,"Random");
                            fprintf(stderr,"\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in EO.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
                          else if (!mystricmp(sSELECTOR,"Ranking")) {
                            sprintf(sSELECTOR,"Ranking");
                            if (($3<=1)||($3>2)) {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sSELECT_PRM,"(2)");
                            }
                            else sprintf(sSELECT_PRM,"(%f)",(float) $3);
                          }
                          else if (!mystricmp(sSELECTOR,"Sequential")) {
                            sprintf(sSELECTOR,"Sequential");
                            if ($3==0) sprintf(sSELECT_PRM,"(unordered)");
                            else if ($3==1) sprintf(sSELECT_PRM,"(ordered)");
                            else {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Sequential must be either 0 (unordered) or 1 (ordered).\nThe parameter will default to 1.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sSELECT_PRM,"(ordered)");
                            }
                          }
                          else {
                            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sSELECTOR);
                            exit(1);
                          }
     }}
  |  RED_PAR IDENTIFIER2{
        sprintf(sRED_PAR, $2->sName);
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
       }
  |  RED_PAR IDENTIFIER2 NUMBER2 {
        sprintf(sRED_PAR, $2->sName);
        switch (TARGET) {
          case DREAM : if (!mystricmp(sRED_PAR,"EPTrn")){
                                        fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                        exit(1);
                                      }
                                      else if (!mystricmp(sRED_PAR,"Tournament")) {
                                        if ($3>=2) sprintf(sRED_PAR,"TournamentSelector(MAX, 1, %d)",(int)$3);
                                        else if (($3>.5)&&($3<=1.0)) sprintf(sRED_PAR,"TournamentSelector(MAX, %f, 2)",(float)$3);
                                        else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                      }
                                      else if (!mystricmp(sRED_PAR,"StochTrn")) {
                                        if (($3>.5)&&($3<=1.0)) sprintf(sRED_PAR,"TournamentSelector(MAX, %f, 2)",(float)$3);
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
                             if ($3>=2) {sprintf(sRED_PAR,"DetTour");sprintf(sRED_PAR_PRM,"(%d)",(int) $3);}
                             else if (($3>.5)&&($3<=1.0)) {sprintf(sRED_PAR,"StochTour");sprintf(sRED_PAR_PRM,"(%f)",(float) $3);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sRED_PAR,"RouletteWheel")) {
                            sprintf(sRED_PAR,"Roulette");
                            if ($3<1) {fprintf(stderr,"\n%s - Warning line %d: The parameter of RouletteWheel must be greater than one.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;}
                            else sprintf(sRED_PAR_PRM,"(%f)",(float) $3);
                          }
                          else if (!mystricmp(sRED_PAR,"Random")) {
                            sprintf(sRED_PAR,"Random");
                            fprintf(stderr,"\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in EO.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
                          else if (!mystricmp(sRED_PAR,"Ranking")) {
                            sprintf(sRED_PAR,"Ranking");
                            if (($3<=1)||($3>2)) {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_PAR_PRM,"(2)");
                            }
                            else sprintf(sRED_PAR_PRM,"(%f)",(float) $3);
                          }
                          else if (!mystricmp(sRED_PAR,"Sequential")) {
                            sprintf(sRED_PAR,"Sequential");
                            if ($3==0) sprintf(sRED_PAR_PRM,"(unordered)");
                            else if ($3==1) sprintf(sRED_PAR_PRM,"(ordered)");
                            else {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Sequential must be either 0 (unordered) or 1 (ordered).\nThe parameter will default to 1.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_PAR_PRM,"(ordered)");
                            }
                          }
                          else {
                            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_PAR);
                            exit(1);
                          }
       }}
  |  RED_OFF IDENTIFIER2{
      sprintf(sRED_OFF, $2->sName);
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
       }}
  |  RED_OFF IDENTIFIER2 NUMBER2 {
        strcpy(sRED_OFF, $2->sName);
        switch (TARGET) {
          case DREAM : if (!mystricmp(sRED_OFF,"EPTrn")){
                                        fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                        exit(1);
                                      }
                                      else if (!mystricmp(sRED_OFF,"Tournament")) {
                                        if ($3>=2) sprintf(sRED_OFF,"TournamentSelector(MAX, 1, %d)",(int)$3);
                                        else if (($3>.5)&&($3<=1.0)) sprintf(sRED_OFF,"TournamentSelector(MAX, %f, 2)",(float)$3);
                                        else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                      }
                                      else if (!mystricmp(sRED_OFF,"StochTrn")) {
                                        if (($3>.5)&&($3<=1.0)) sprintf(sRED_OFF,"TournamentSelector(MAX, %f, 2)",(float)$3);
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
                             if ($3>=2) {sprintf(sRED_OFF,"DetTour");sprintf(sRED_OFF_PRM,"(%d)",(int) $3);}
                             else if (($3>.5)&&($3<=1.0)) {sprintf(sRED_OFF,"StochTour");sprintf(sRED_OFF_PRM,"(%f)",(float) $3);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sRED_OFF,"RouletteWheel")) {
                            sprintf(sRED_OFF,"Roulette");
                            if ($3<1) {fprintf(stderr,"\n%s - Warning line %d: The parameter of RouletteWheel must be greater than one.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;}
                            else sprintf(sRED_OFF_PRM,"(%f)",(float) $3);
                          }
                          else if (!mystricmp(sRED_OFF,"Random")) {
                            sprintf(sRED_OFF,"Random");
                            fprintf(stderr,"\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in EO.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
                          else if (!mystricmp(sRED_OFF,"Ranking")) {
                            sprintf(sRED_OFF,"Ranking");
                            if (($3<=1)||($3>2)) {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_OFF_PRM,"(2)");
                            }
                            else sprintf(sRED_OFF_PRM,"(%f)",(float) $3);
                          }
                          else if (!mystricmp(sRED_OFF,"Sequential")) {
                            sprintf(sRED_OFF,"Sequential");
                            if ($3==0) sprintf(sRED_OFF_PRM,"(unordered)");
                            else if ($3==1) sprintf(sRED_OFF_PRM,"(ordered)");
                            else {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Sequential must be either 0 (unordered) or 1 (ordered).\nThe parameter will default to 1.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_OFF_PRM,"(ordered)");
                            }
                          }
                          else {
                            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_OFF);
                            exit(1);
                          }
       }}
  |  RED_FINAL IDENTIFIER2{
        strcpy(sRED_FINAL, $2->sName);
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
       }}
  |  RED_FINAL IDENTIFIER2 NUMBER2 {
        strcpy(sRED_FINAL, $2->sName);
        switch (TARGET) {
          case DREAM : if (!mystricmp(sRED_FINAL,"EPTrn")){
                                        fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
                                        exit(1);
                                      }
                                      else if (!mystricmp(sRED_FINAL,"Tournament")) {
                                        if ($3>=2) sprintf(sRED_FINAL,"TournamentSelector(%s, 1, %d)",(nMINIMISE?"true":"false"),(int)$3);
                                        else if (($3>.5)&&($3<=1.0)) sprintf(sRED_FINAL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"true":"false"),(float)$3);
                                        else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                                      }
                                      else if (!mystricmp(sRED_FINAL,"StochTrn")) {
                                        if (($3>.5)&&($3<=1.0)) sprintf(sRED_FINAL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"true":"false"),(float)$3);
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
                             if ($3>=2) {sprintf(sRED_FINAL,"DetTour");sprintf(sRED_FINAL_PRM,"(%d)",(int) $3);}
                             else if (($3>.5)&&($3<=1.0)) {sprintf(sRED_FINAL,"StochTour");sprintf(sRED_FINAL_PRM,"(%f)",(float) $3);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sRED_FINAL,"RouletteWheel")) {
                            sprintf(sRED_FINAL,"Roulette");
                            if ($3<1) {fprintf(stderr,"\n%s - Warning line %d: The parameter of RouletteWheel must be greater than one.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;}
                            else sprintf(sRED_FINAL_PRM,"(%f)",(float) $3);
                          }
                          else if (!mystricmp(sRED_FINAL,"Random")) {
                            sprintf(sRED_FINAL,"Random");
                            fprintf(stderr,"\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in EO.\nThe parameter will therefore be ignored.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
                          else if (!mystricmp(sRED_FINAL,"Ranking")) {
                            sprintf(sRED_FINAL,"Ranking");
                            if (($3<=1)||($3>2)) {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_FINAL_PRM,"(2)");
                            }
                            else sprintf(sRED_FINAL_PRM,"(%f)",(float) $3);
                          }
                          else if (!mystricmp(sRED_FINAL,"Sequential")) {
                            sprintf(sRED_FINAL,"Sequential");
                            if ($3==0) sprintf(sRED_FINAL_PRM,"(unordered)");
                            else if ($3==1) sprintf(sRED_FINAL_PRM,"(ordered)");
                            else {
                              fprintf(stderr,"\n%s - Warning line %d: The parameter of Sequential must be either 0 (unordered) or 1 (ordered).\nThe parameter will default to 1.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                              sprintf(sRED_FINAL_PRM,"(ordered)");
                            }
                          }
                          else {
                            fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, sRED_FINAL);
                            exit(1);
                          }
       }}
  |  OFFSPRING NUMBER2 {nOFF_SIZE=(int)$2;}
  |  OFFSPRING NUMBER2 '%' {nOFF_SIZE=(int)($2*nPOP_SIZE/100);}
  |  SURVPAR NUMBER2 {nSURV_PAR_SIZE=(int)$2;}
  |  SURVPAR NUMBER2 '%' {nSURV_PAR_SIZE=(int)($2*nPOP_SIZE/100);}
  |  SURVOFF NUMBER2 {nSURV_OFF_SIZE=(int)$2;}
  |  SURVOFF NUMBER2 '%' {nSURV_OFF_SIZE=(int)($2*nPOP_SIZE/100);}
  |  REPLACEMENT IDENTIFIER2 {                                       
  // Generational
      if (!mystricmp($2->sName,"Generational")){
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
      else if (!mystricmp($2->sName,"SteadyState"))
        switch (TARGET) {
          case GALIB : strcpy(sREPLACEMENT, "Incremental");
                                  fREPL_PERC=0;
                                  break;
          case EO : strcpy(sREPLACEMENT, "SSGA");
        }
// Plus
      else if (!mystricmp($2->sName,"ESPlus"))
        switch (TARGET) {
          case GALIB : strcpy(sREPLACEMENT, "SteadyState"); break;
          case EO : strcpy(sREPLACEMENT, "Plus");
        }
// Comma
      else if (!mystricmp($2->sName,"ESComma"))
        switch (TARGET) {
          case GALIB : fprintf(stderr,"\n%s - Error line %d: The Comma replacement strategie is not yet available in EASEA-GALib.\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);
          case EO : strcpy(sREPLACEMENT, "Comma");
        }  
      else if (!mystricmp($2->sName,"Custom"))
        switch (TARGET) {
          case GALIB : fprintf(stderr,"\n%s - Error line %d: The Custom replacement strategie is not yet available in EASEA-GALib.\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);
          case EO : strcpy(sREPLACEMENT, "Custom");
        }  
      else {
         fprintf(stderr,"\n%s - Error line %d: The %s replacement strategy does not exist.\n",sEZ_FILE_NAME,EASEALexer.yylineno, $2->sName);
         exit(1);
     }}
  |  DISCARD IDENTIFIER2 {
      strcpy(sDISCARD, $2->sName);
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
       }}}
  |  DISCARD IDENTIFIER2 '(' NUMBER2 ')' {
      strcpy(sDISCARD, $2->sName);
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
                             if ($4>=2) {strcpy(sDISCARD,"DetTournament");sprintf(sDISCARD_PRM,"(%d)",(int) $4);}
                             else if (($4>.5)&&($4<=1.0)) {strcpy(sDISCARD,"StochTournament");sprintf(sDISCARD_PRM,"(%f)",(float) $4);}
                             else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
                          }
                          else if (!mystricmp(sDISCARD,"Worst")) {
                             strcpy(sDISCARD,"Worse");
                             fprintf(stderr,"\n%s - Warning line %d: The Worst discarding operator does not take parameters. The parameter will be ignored.\n",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
                          }
      }}
  |  MINIMAXI IDENTIFIER2 {
      if ((!mystricmp($2->sName,"maximise")) || (!mystricmp($2->sName,"maximize"))) nMINIMISE=0;
      else if ((!mystricmp($2->sName,"minimise")) || (!mystricmp($2->sName,"minimize"))) nMINIMISE=1;
      else {
         fprintf(stderr,"\n%s - Error line %d: The evaluator goal default parameter can only take\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         fprintf(stderr,"two values : maximi[sz]e or minimi[sz]e.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
       }
      }
  |  ELITE NUMBER2 {
        if ($2!=0) bELITISM=1;
        else bELITISM=0;
        nELITE=(int)$2;
        }
  |  ELITE NUMBER2 '%' {
        if ($2!=0) bELITISM=1;
        else bELITISM=0;
        nELITE=(int)$2*nPOP_SIZE/100;
        }
  |  ELITISM IDENTIFIER2{
      if (!mystricmp($2->sName,"weak")) bELITISM=0;
      else if (!mystricmp($2->sName,"strong")) bELITISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Elitism must be \"Strong\" or \"Weak\".\nDefault value \"Strong\" inserted.\n.",sEZ_FILE_NAME,EASEALexer.yylineno);nWARNINGS++;
         bELITISM=1;
       }}
  |  MIG_CLONE IDENTIFIER2{
       if (!mystricmp($2->sName,"clone")) nMIG_CLONE=1;
      else if (!mystricmp($2->sName,"move")) nMIG_CLONE=0;
      else {
         fprintf(stderr,"\n%s - Error line %d: Looking for \"Move\" or \"Clone\".\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
     } }
  |  MIG_SEL IDENTIFIER2{
        strcpy(sMIG_SEL, $2->sName);
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
     }
  |  MIG_SEL IDENTIFIER2 NUMBER2 {
        strcpy(sMIG_SEL, $2->sName);
        if (!mystricmp(sMIG_SEL,"EPTrn")){
          fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
          exit(1);
        }
        else if (!mystricmp(sMIG_SEL,"Tournament")) {
          if ($3>=2) sprintf(sMIG_SEL,"TournamentSelector(%s, 1, %d)",(nMINIMISE?"false":"true"),(int)$3);
          else if (($3>.5)&&($3<=1.0)) sprintf(sMIG_SEL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"false":"true"),(float)$3);
          else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
        }
        else if (!mystricmp(sMIG_SEL,"StochTrn")) {
          if (($3>.5)&&($3<=1.0)) sprintf(sMIG_SEL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"false":"true"),(float)$3);
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
     }
  |  MIGRATOR IDENTIFIER2{
      if (!mystricmp($2->sName,"neighbours")) {
        sprintf(sMIGRATOR,"DefaultEMigrator");
        sprintf(sMIG_TARGET_SELECTOR,", new RandomTargetSelector()");
      }
      else if (!mystricmp($2->sName,"mesh")) {
        sprintf(sMIGRATOR,"ToAllEmigrator");
        sMIG_TARGET_SELECTOR[0]=0;     
      }
      else {
         fprintf(stderr,"\n%s - Error line %d: Looking for \"Move\" or \"Clone\".\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
     } }
  |  MIG_FREQ NUMBER2
     {fMIG_FREQ=(float)$2;}
  |  NB_MIG NUMBER2
     {nNB_MIG=(int)$2;}
  |  IMMIG_SEL IDENTIFIER2{
        strcpy(sIMMIG_SEL, $2->sName);
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
     }
  |  IMMIG_SEL IDENTIFIER2 NUMBER2 {
        strcpy(sIMMIG_SEL, $2->sName);
        if (!mystricmp(sIMMIG_SEL,"EPTrn")){
          fprintf(stderr,"\n%s - Error line %d: The EP-Tournament selector is not implemented in DREAM.\n",sEZ_FILE_NAME,EASEALexer.yylineno);
          exit(1);
        }
        else if (!mystricmp(sIMMIG_SEL,"Tournament")) {
          if ($3>=2) sprintf(sIMMIG_SEL,"TournamentSelector(%s, 1, %d)",(nMINIMISE?"false":"true"),(int)$3);
          else if (($3>.5)&&($3<=1.0)) sprintf(sIMMIG_SEL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"false":"true"),(float)$3);
          else {fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",sEZ_FILE_NAME,EASEALexer.yylineno); exit(1);}
        }
        else if (!mystricmp(sIMMIG_SEL,"StochTrn")) {
          if (($3>.5)&&($3<=1.0)) sprintf(sIMMIG_SEL,"TournamentSelector(%s, %f, 2)",(nMINIMISE?"false":"true"),(float)$3);
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
     }
  |  IMMIG_REPL IDENTIFIER2{
      if (!mystricmp($2->sName,"replace")) nIMMIG_REPL=1;
      else if (!mystricmp($2->sName,"add")) nIMMIG_REPL=0;
      else {
         fprintf(stderr,"\n%s - Error line %d: Looking for \"Add\" or \"Replace\".\n",sEZ_FILE_NAME,EASEALexer.yylineno);
         exit(1);
     } }
  ;
  
Expr
  : IDENTIFIER '=' Expr { 
      if (SymbolTable.find($1->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,$1->sName);
         exit(1);
      }
      $$ = assign(SymbolTable.find($1->sName), $3);
    }
  | Expr '+' Expr                    { $$ = $1 + $3; }
  | Expr '-' Expr                     { $$ = $1 - $3; }
  | Expr '*' Expr                     { $$ = $1 * $3; }
  | Expr '/' Expr                     { $$ = divide($1, $3); }
  | '(' Expr ')'                          { $$ = $2; }
  | '-' Expr %prec UMINUS { $$ = -$2; }
  | NUMBER                         { $$ = $1; }
  | IDENTIFIER{
      if (SymbolTable.find($1->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,EASEALexer.yylineno,$1->sName);
         exit(1);
      }
      $$ = (SymbolTable.find($1->sName))->dValue;
    }
  ;

%%
                       
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

