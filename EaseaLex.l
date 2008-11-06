%{
  /****************************************************************************
EaseaLex.l
Lexical analyser for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@polytechnique.fr)
Ecole Polytechnique
Centre de Mathématiques Appliquées
91128 Palaiseau cedex
  ****************************************************************************/

#include "Easea.h"
#include "EaseaParse.h"
  %}

// include file
%include {
  // forward references
  class CEASEAParser;
  class CSymbolTable;
 }                                                     
                                         
%start GENOME_ANALYSIS TEMPLATE_ANALYSIS MACRO_IDENTIFIER MACRO_DEFINITION 
%start COPY_USER_DECLARATIONS COPY_INITIALISATION_FUNCTION ANALYSE_USER_CLASSES
%start COPY_EO_INITIALISER 
%start COPY COPY_INITIALISER COPY_CROSSOVER COPY_MUTATOR COPY_EVALUATOR
%start COPY_DISPLAY COPY_USER_FUNCTION COPY_USER_GENERATION PARAMETERS_ANALYSIS GET_PARAMETERS 
%start COPY_USER_FUNCTIONS COPY_GENERATION_FUNCTION GET_METHODS

 // lexical analyser name and class definition
%name CEASEALexer {
 protected:
  CSymbolTable *pSymbolTable;   // the symbol table
  bool bSymbolInserted,bWithinEvaluator;  // used to change evalutor type from double to float 
  bool bInitFunction,bDisplayFunction,bFunction, bNotFinishedYet, bWithinEO_Function;
  bool bDoubleQuotes,bWithinDisplayFunction,bWithinInitialiser,bWithinMutator,bWithinXover;
  bool bWaitingForSemiColon,bFinishNB_GEN,bFinishMINIMISE,bFinishMINIMIZE,bGenerationFunction;
  bool bCatchNextSemiColon,bWaitingToClosePopulation, bMethodsInGenome;
  CSymbol *pASymbol;

 public:
  int create(CEASEAParser* pParser, CSymbolTable* pSymbolTable);
  int yywrap();
  double myStrtod() const;                              
 }

// constructor
{                                
  bFunction=bWithinEvaluator=bDisplayFunction=bInitFunction=bNotFinishedYet=0;
  bSymbolInserted=bDoubleQuotes=bWithinDisplayFunction=bWithinInitialiser=bWithinMutator=bWithinXover=0;
  bWaitingForSemiColon=bFinishNB_GEN=bFinishMINIMISE=bFinishMINIMIZE=bGenerationFunction=0;
  bCatchNextSemiColon,bWaitingToClosePopulation=bMethodsInGenome=0;
}

// macros
exponent  ([Ee][+-]?[0-9]+)

%%                                              

%{
  // extract yylval for use later on in actions
  YYSTYPE& yylval = *(YYSTYPE*)yyparserptr->yylvalptr;
  %}

// Gobbles up ^Ms, to be compatible with unix
\r {}

// switch immediately to TEMPLATE_ANALYSIS state
<INITIAL>.    {
  BEGIN TEMPLATE_ANALYSIS; yyless(yyleng-1);
 }

//****************************************
//  GA_GENERATION lexer (feeds on template file and occasionally on .ez file)
//****************************************

<TEMPLATE_ANALYSIS>"\\TEMPLATE_START"  {             
  char sFileName[1000];
  strcpy(sFileName, sRAW_PROJECT_NAME);
  switch (TARGET) {
  case DREAM : strcat(sFileName,".java"); break;
  default : strcat(sFileName,".cpp");
  } 
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"AESAE" {fprintf(fpOutputFile,"EASEA");}        
<TEMPLATE_ANALYSIS>"EASEA" {fprintf(fpOutputFile,"%s",sPROJECT_NAME);}        
<TEMPLATE_ANALYSIS>"\\EZ_PATH" {fprintf(fpOutputFile,"%s",sEZ_PATH);}        
<TEMPLATE_ANALYSIS>"\\EO_DIR" {fprintf(fpOutputFile,"%s",sEO_DIR);}        
<TEMPLATE_ANALYSIS>"LC_EASEA" {fprintf(fpOutputFile,"%s",sLOWER_CASE_PROJECT_NAME);}        
<TEMPLATE_ANALYSIS>"\\OPERATING_SYSTEM"  {switch (OPERATING_SYSTEM) {
  case UNIX : fprintf(fpOutputFile,"UNIX_OS"); break;
  case WINDOWS : fprintf(fpOutputFile,"WINDOWS_OS"); break;
  case UNKNOWN_OS : fprintf(fpOutputFile,"UNKNOWN_OS"); break;
  }
 }
<TEMPLATE_ANALYSIS>"\\INSERT_USER_DECLARATIONS"  {
  if (bVERBOSE) printf ("Inserting user declarations.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_USER_DECLARATIONS;
 }
<TEMPLATE_ANALYSIS>"\\INSERT_INITIALISATION_FUNCTION"  {
  if (bVERBOSE) printf ("Inserting initialisation function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_INITIALISATION_FUNCTION;
 }
<TEMPLATE_ANALYSIS>"\\INSERT_GENERATION_FUNCTION"  {
  if (bVERBOSE) printf ("Inserting generation function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_GENERATION_FUNCTION;
 }
<TEMPLATE_ANALYSIS>"\\ANALYSE_USER_CLASSES"  {
  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 }
<TEMPLATE_ANALYSIS>"\\INSERT_USER_CLASSES"  {
  if (bVERBOSE) printf ("Inserting User classes.\n");
  if (TARGET!=DREAM) fprintf (fpOutputFile,"// User classes\n");
  CListItem<CSymbol*> *pSym;
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem())
    if ((pSym->Object->pType->ObjectType==oUserClass)&&(!pSym->Object->pType->bAlreadyPrinted))
      pSym->Object->pType->printClasses(fpOutputFile);
 }
<TEMPLATE_ANALYSIS>"\\GENOME_CTOR"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if ((pSym->Object->ObjectType==oArray)&&(TARGET==DREAM))
      fprintf(fpOutputFile,"    %s = new %s[%d];\n",pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->nSize/pSym->Object->pType->nSize);
    if (pSym->Object->ObjectType==oPointer){
      if (TARGET==DREAM) fprintf(fpOutputFile,"    %s=null;\n",pSym->Object->sName);
      else fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 }
<TEMPLATE_ANALYSIS>"\\INSERT_GENOME"  {        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 }                                            
<TEMPLATE_ANALYSIS>"\\ASSIGNMENT_OP"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default assignment constructor.\n");
  fprintf (fpOutputFile,"// Memberwise assignment\n");             
  pGENOME->pSymbolList->reset();                                      
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oObject)
      fprintf(fpOutputFile,"    %s=genome.%s;\n",pSym->Object->sName,pSym->Object->sName);
    if (pSym->Object->ObjectType==oPointer)
      fprintf(fpOutputFile,"    %s=new %s(*(genome.%s));\n",pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
    if (pSym->Object->ObjectType==oArray){
      fprintf(fpOutputFile,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
      fprintf(fpOutputFile,"       %s[EASEA_Ndx]=genome.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
    }
  }
 }
<TEMPLATE_ANALYSIS>"\\CLONE"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default clone method.\n");
  fprintf (fpOutputFile,"// Memberwise Cloning\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (TARGET==DREAM){
      if (pSym->Object->ObjectType==oObject)
	fprintf(fpOutputFile,"    %s=EZ_genome.%s;\n",pSym->Object->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oPointer)
	fprintf(fpOutputFile,"    %s=((EZ_genome.%s!=null) ? new %s(EZ_genome.%s) : null);\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray){
	fprintf(fpOutputFile,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	if (pSym->Object->pType->ObjectType==oUserClass) fprintf(fpOutputFile,"       this.%s[EASEA_Ndx]=new %s(EZ_genome.%s[EASEA_Ndx]);}\n",pSym->Object->sName, pSym->Object->pType->sName, pSym->Object->sName);
	else fprintf(fpOutputFile,"       this.%s[EASEA_Ndx]=EZ_genome.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
      }
    } 
    else {
      if (pSym->Object->ObjectType==oObject)
	fprintf(fpOutputFile,"    %s=genome.%s;\n",pSym->Object->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oPointer)
	fprintf(fpOutputFile,"    %s=(genome.%s ? new %s(*(genome.%s)) : NULL);\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray){
	fprintf(fpOutputFile,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	fprintf(fpOutputFile,"       %s[EASEA_Ndx]=genome.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
      }
    }
  }
 }

<TEMPLATE_ANALYSIS>"\\COPY_CTOR"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default copy constructor.\n");
  fprintf (fpOutputFile,"// Memberwise copy\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (TARGET==DREAM){
      if (pSym->Object->ObjectType==oObject)
	fprintf(fpOutputFile,"    EZ_genome.%s=%s;\n",pSym->Object->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oPointer)
	fprintf(fpOutputFile,"    EZ_genome.%s=(%s!=null ? new %s(%s) : null);\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray){
	fprintf(fpOutputFile,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	if (pSym->Object->pType->ObjectType==oUserClass) fprintf(fpOutputFile,"       EZ_genome.%s[EASEA_Ndx]=new %s(this.%s[EASEA_Ndx]);}\n",pSym->Object->sName, pSym->Object->pType->sName, pSym->Object->sName);
	else fprintf(fpOutputFile,"       EZ_genome.%s[EASEA_Ndx]=this.%s[EASEA_Ndx];}\n",pSym->Object->sName, pSym->Object->sName);
      }
    }
    else {
      if (pSym->Object->ObjectType==oObject)
	fprintf(fpOutputFile,"    %s=genome.%s;\n",pSym->Object->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oPointer)
	fprintf(fpOutputFile,"    %s=(genome.%s ? new %s(*(genome.%s)) : NULL);\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray){
	fprintf(fpOutputFile,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	fprintf(fpOutputFile,"       %s[EASEA_Ndx]=genome.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
      }
    }
  }
 }
<TEMPLATE_ANALYSIS>"\\GENOME_DTOR"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default destructor.\n");
  fprintf (fpOutputFile,"// Destructing pointers\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem())
    if (pSym->Object->ObjectType==oPointer){
      if (TARGET==DREAM) fprintf(fpOutputFile,"  if (%s) delete %s;\n  %s=null;\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
      else fprintf(fpOutputFile,"  if (%s) delete %s;\n  %s=NULL;\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
    }
 }
<TEMPLATE_ANALYSIS>"\\EQUAL"  {       
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default diversity test.\n");
  fprintf (fpOutputFile,"// Default diversity test (required by GALib)\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()) {
    if (pSym->Object->ObjectType==oObject)
      fprintf(fpOutputFile,"  if (%s!=genome.%s) return 0;\n",pSym->Object->sName,pSym->Object->sName);
    if (pSym->Object->ObjectType==oPointer)
      fprintf(fpOutputFile,"  if (*%s!=*(genome.%s)) return 0;\n",pSym->Object->sName,pSym->Object->sName);
    if (pSym->Object->ObjectType==oArray){
      fprintf(fpOutputFile,"  {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
      fprintf(fpOutputFile,"     if (%s[EASEA_Ndx]!=genome.%s[EASEA_Ndx]) return 0;}\n",pSym->Object->sName,pSym->Object->sName);
    }
  }
 }
<TEMPLATE_ANALYSIS>"\\COMPARE"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default genome comparator.\n");
  fprintf (fpOutputFile,"// Default genome comparator (required by GALib)\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()) {
    if (pSym->Object->ObjectType==oObject)
      fprintf(fpOutputFile,"  if (sis.%s!=bro.%s) diff++;\n",pSym->Object->sName,pSym->Object->sName);
    if (pSym->Object->ObjectType==oPointer)
      fprintf(fpOutputFile,"  if (*(sis.%s)!=*(bro.%s)) diff++;\n",pSym->Object->sName,pSym->Object->sName);
    if (pSym->Object->ObjectType==oArray){
      fprintf(fpOutputFile,"  {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
      fprintf(fpOutputFile,"     if (sis.%s[EASEA_Ndx]!=bro.%s[EASEA_Ndx]) diff++;}\n",pSym->Object->sName,pSym->Object->sName);
    }
  }
 }
<TEMPLATE_ANALYSIS>"\\READ"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (TARGET==GALIB) {
      if ((pSym->Object->ObjectType==oObject)&&(strcmp(pSym->Object->pType->sName, "bool")))
	fprintf(fpOutputFile,"  is >> %s;\n",pSym->Object->sName);                                  
      if ((pSym->Object->ObjectType==oArray)&&(strcmp(pSym->Object->pType->sName, "bool"))){
	fprintf(fpOutputFile,"  {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	fprintf(fpOutputFile,"     is >> %s[EASEA_Ndx];}\n",pSym->Object->sName);
      }
      if (pSym->Object->ObjectType==oPointer)
	fprintf(fpOutputFile,"  is >> *%s;\n",pSym->Object->sName);
    }
    if (TARGET==EO) {
      if (pSym->Object->ObjectType==oObject)
	fprintf(fpOutputFile,"  is >> %s;\n",pSym->Object->sName);                                  
      if (pSym->Object->ObjectType==oArray){
	fprintf(fpOutputFile,"  {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	fprintf(fpOutputFile,"     is >> %s[EASEA_Ndx];}\n",pSym->Object->sName);
      }
      if (pSym->Object->ObjectType==oPointer)
	fprintf(fpOutputFile,"  is >> *(%s);\n",pSym->Object->sName);
    }
  }
 }
<TEMPLATE_ANALYSIS>"\\INSERT_DISPLAY"  {        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 }
<TEMPLATE_ANALYSIS>"\\WRITE"  {
  CListItem<CSymbol*> *pSym;
  if (!bDisplayFunction){
    if (bVERBOSE) printf ("*** Creating default display function. ***\n");
    fprintf (fpOutputFile,"// Default display function\n");
    pGENOME->pSymbolList->reset();
    while (pSym=pGENOME->pSymbolList->walkToNextItem()){
      if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
      if (TARGET==DREAM) {
	if (pSym->Object->ObjectType==oObject){
	  if (bDisplayFunction) printf("//");
	  fprintf(fpOutputFile,"  EASEA_S = EASEA_S + \"%s:\" + %s + \"\\n\";\n",pSym->Object->sName,pSym->Object->sName);
	}
	if (pSym->Object->ObjectType==oArray){
	  if (bDisplayFunction) printf("//");
	  fprintf(fpOutputFile,"  {EASEA_S = \"Array %s : \";\n",pSym->Object->sName);
	  if (bDisplayFunction) printf("//");
	  fprintf(fpOutputFile,"   for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	  if (bDisplayFunction) printf("//");
	  fprintf(fpOutputFile,"     EASEA_S = EASEA_S + \"\\t\"+ %s[EASEA_Ndx];}\n",pSym->Object->sName);
	  if (bDisplayFunction) printf("//");
	}         
	if (pSym->Object->ObjectType==oPointer){
	  if (bDisplayFunction) printf("//");
	  fprintf(fpOutputFile,"  if (%s!=null) EASEA_S = EASEA_S + \"%s:\" + %s + \"\\n\";\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
	}
      }
      else {
	if (pSym->Object->ObjectType==oObject){
	  if (bDisplayFunction) printf("//");
	  if (TARGET==GALIB) fprintf(fpOutputFile,"  os << \"%s:\" << %s << \"\\n\";\n",pSym->Object->sName,pSym->Object->sName);
	  if (TARGET==EO) fprintf(fpOutputFile,"  os << \"%s:\" << %s << \"\\n\";\n",pSym->Object->sName,pSym->Object->sName);
	}
	if (pSym->Object->ObjectType==oArray){
	  if (bDisplayFunction) printf("//");
	  if (TARGET == GALIB) fprintf(fpOutputFile,"  {os << \"Array %s : \";\n",pSym->Object->sName);
	  if (TARGET == DREAM) fprintf(fpOutputFile,"  {EASEA_S = \"Array %s : \";\n",pSym->Object->sName);
	  if (TARGET == EO) fprintf(fpOutputFile,"  {",pSym->Object->sName);
	  if (bDisplayFunction) printf("//");
	  fprintf(fpOutputFile,"   for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	  if (bDisplayFunction) printf("//");
	  if (TARGET==GALIB) fprintf(fpOutputFile,"     os << \"[\" << EASEA_Ndx << \"]:\" << %s[EASEA_Ndx] << \"\\t\";}\n",pSym->Object->sName);
	  if (TARGET==EO) fprintf(fpOutputFile,"     os << %s[EASEA_Ndx];}\n",pSym->Object->sName);
	  if (TARGET==DREAM) fprintf(fpOutputFile,"     EASEA_S = EASEA_S + \"\\t\"+ %s[EASEA_Ndx];}\n",pSym->Object->sName);
	  if (bDisplayFunction) printf("//");
	  if (TARGET!=DREAM) fprintf(fpOutputFile,"  os << \"\\n\";\n",pSym->Object->sName);
	}         
	if (pSym->Object->ObjectType==oPointer){
	  if (bDisplayFunction) printf("//");
	  if (TARGET==GALIB) fprintf(fpOutputFile,"  os << \"%s:\" << *%s << \"\\n\";\n",pSym->Object->sName,pSym->Object->sName);
	  if (TARGET==EO) fprintf(fpOutputFile,"  os << \"%s:\" << *%s << \"\\n\";\n",pSym->Object->sName,pSym->Object->sName);
	}
      }
    }
  }                      
 }    
<TEMPLATE_ANALYSIS>"\\INSERT_USER_FUNCTIONS"  {
  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 }
<TEMPLATE_ANALYSIS>"\\INSERT_EO_INITIALISER"  {        
  yyreset();
  bWithinEO_Function=1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 }
<TEMPLATE_ANALYSIS>"\\INSERT_INITIALISER"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_INITIALISER;   
 }
<TEMPLATE_ANALYSIS>"\\INSERT_CROSSOVER"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_CROSSOVER;   
 }
<TEMPLATE_ANALYSIS>"\\INSERT_MUTATOR"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_MUTATOR;   
 }
<TEMPLATE_ANALYSIS>"\\INSERT_EVALUATOR"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EVALUATOR;   
 }
<TEMPLATE_ANALYSIS>"\\ANALYSE_PARAMETERS"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 }
<TEMPLATE_ANALYSIS>"\\INSERT_GEN_FCT_CALL"  {
  if (bGenerationFunction) fprintf(fpOutputFile,"\n    EASEAGenerationFunction(ga);\n");
 }        
<TEMPLATE_ANALYSIS>"\\INSERT_INIT_FCT_CALL"  {
  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 }        
<TEMPLATE_ANALYSIS>"\\SELECTOR"  {fprintf(fpOutputFile,"%s",sSELECTOR);}
//<TEMPLATE_ANALYSIS>"\\GEN_SELECTOR"  {fprintf(fpOutputFile,"%s(%s,%d)",sGEN_SELECTOR);}
<TEMPLATE_ANALYSIS>"\\SELECT_PRM"  {fprintf(fpOutputFile,"%s",sSELECT_PRM);}
<TEMPLATE_ANALYSIS>"\\POP_SIZE"  {fprintf(fpOutputFile,"%d",nPOP_SIZE);}
<TEMPLATE_ANALYSIS>"\\OFF_SIZE"  {fprintf(fpOutputFile,"%d",nOFF_SIZE);}
<TEMPLATE_ANALYSIS>"\\ELITE_SIZE"  {fprintf(fpOutputFile,"%d",nELITE);}
<TEMPLATE_ANALYSIS>"\\RED_PAR"  {fprintf(fpOutputFile,"%s",sRED_PAR);}
<TEMPLATE_ANALYSIS>"\\RED_PAR_PRM"  {fprintf(fpOutputFile,"%s",sRED_PAR_PRM);}
<TEMPLATE_ANALYSIS>"\\RED_OFF"  {fprintf(fpOutputFile,"%s",sRED_OFF);}
<TEMPLATE_ANALYSIS>"\\RED_OFF_PRM"  {fprintf(fpOutputFile,"%s",sRED_OFF_PRM);}
<TEMPLATE_ANALYSIS>"\\RED_FINAL"  {fprintf(fpOutputFile,"%s",sRED_FINAL);}
<TEMPLATE_ANALYSIS>"\\RED_FINAL_PRM"  {fprintf(fpOutputFile,"%s",sRED_FINAL_PRM);}
<TEMPLATE_ANALYSIS>"\\SURV_PAR_SIZE"  {fprintf(fpOutputFile,"%d",nSURV_PAR_SIZE);}
<TEMPLATE_ANALYSIS>"\\SURV_OFF_SIZE"  {fprintf(fpOutputFile,"%d",nSURV_OFF_SIZE);}
<TEMPLATE_ANALYSIS>"\\NB_GEN"  {fprintf(fpOutputFile,"%d",nNB_GEN);}
<TEMPLATE_ANALYSIS>"\\NB_ISLANDS"  {fprintf(fpOutputFile,"%d",nNB_ISLANDS);}
<TEMPLATE_ANALYSIS>"\\PROP_OR_SEQ"  {fprintf(fpOutputFile,"%s",bPROP_SEQ?"Prop":"Seq");}
<TEMPLATE_ANALYSIS>"\\MUT_PROB"  {if (TARGET==GALIB)fprintf(fpOutputFile,"(float)%f",fMUT_PROB);
  else fprintf(fpOutputFile,"%f",fMUT_PROB);}
<TEMPLATE_ANALYSIS>"\\XOVER_PROB"  {if (TARGET==GALIB)fprintf(fpOutputFile,"(float)%f",fXOVER_PROB);
  else fprintf(fpOutputFile,"%f",fXOVER_PROB);}
<TEMPLATE_ANALYSIS>"\\REPLACEMENT"  {fprintf(fpOutputFile,"%s",sREPLACEMENT);}
<TEMPLATE_ANALYSIS>"\\REPL_PERC"  {fprintf(fpOutputFile,"(float)%f/100",fREPL_PERC);}
<TEMPLATE_ANALYSIS>"\\DISCARD_PRM"  {fprintf(fpOutputFile,"%s",sDISCARD_PRM);}
<TEMPLATE_ANALYSIS>"\\STEADYSTATE"  {if ((fREPL_PERC==0)||(!strcmp(sREPLACEMENT,"Incremental"))||(!strcmp(sREPLACEMENT,"Simple")))
     fprintf(fpOutputFile,"// undefined ");}
<TEMPLATE_ANALYSIS>"\\COMMENT"  {if (mystricmp(sREPLACEMENT,"SSGA")) fprintf(fpOutputFile,"//");}
<TEMPLATE_ANALYSIS>"\\MINIMAXI"  {switch (TARGET) { case GALIB : fprintf(fpOutputFile,"%d",nMINIMISE? -1:1); break;
  case EO : fprintf(fpOutputFile,"%s",nMINIMISE? "eoMinimizingFitness" : "eoMaximizingFitness"); break;
  case DREAM : fprintf(fpOutputFile,"%s",nMINIMISE? "false" : "true"); break;
  }                                  }
<TEMPLATE_ANALYSIS>"\\ELITIST_REPLACE_NAME"  {if (TARGET==EO) {if (bELITISM) fprintf(fpOutputFile,"standardR");
    else fprintf(fpOutputFile,"r");}}
<TEMPLATE_ANALYSIS>"\\ELITISM"  {switch (TARGET) { case GALIB : if (!mystricmp(sREPLACEMENT,"simple"))
       if (bELITISM) fprintf(fpOutputFile,"  ga.elitist(gaTrue);\n");
       else fprintf(fpOutputFile,"  ga.elitist(gaFalse);\n");
      break;
  case EO :  fprintf(fpOutputFile,"%d",bELITISM);
  }                                 }
<TEMPLATE_ANALYSIS>"\\IMMIG_SEL" {fprintf(fpOutputFile,"%s",sIMMIG_SEL);}
<TEMPLATE_ANALYSIS>"\\IMMIG_REPL" {fprintf(fpOutputFile,"%d",nIMMIG_REPL);}
<TEMPLATE_ANALYSIS>"\\MIGRATOR" {fprintf(fpOutputFile,"%s",sMIGRATOR);}
<TEMPLATE_ANALYSIS>"\\MIG_SEL" {fprintf(fpOutputFile,"%s",sMIG_SEL);}
<TEMPLATE_ANALYSIS>"\\MIG_CLONE" {fprintf(fpOutputFile,"%d",nMIG_CLONE);}
<TEMPLATE_ANALYSIS>"\\NB_MIG" {fprintf(fpOutputFile,"%d",nNB_MIG);}
<TEMPLATE_ANALYSIS>"\\MIG_FREQ" {fprintf(fpOutputFile,"%f",fMIG_FREQ);}
<TEMPLATE_ANALYSIS>"\\MIG_TARGET_SELECTOR" {fprintf(fpOutputFile,"%s",sMIG_TARGET_SELECTOR);}

<TEMPLATE_ANALYSIS>"\\START_EO_GENOME_H_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Genome.h");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 }
<TEMPLATE_ANALYSIS>"\\START_EO_EVAL_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"EvalFunc.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EO_INITER_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Init.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EO_MUT_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Mutation.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EO_QUAD_XOVER_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"QuadCrossover.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EO_CONTINUE_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"_make_continue.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EO_MAKE_GENOTYPE_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"_make_genotype.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EO_MAKE_OPERATORS_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"_make_operators.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EO_PARAM_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EO_MAKEFILE_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EVAL_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Evaluator.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_INITER_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Initer.java");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 }
<TEMPLATE_ANALYSIS>"\\START_GENOME_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Genome.java");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 }
<TEMPLATE_ANALYSIS>"\\START_MUT_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Mutator.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_RECOMB_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Recombine.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_IHDEF_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"IHDefIniter.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_FITINIT_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"FitnessIniter.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_ISLAND_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Island.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\TEMPLATE_END"  {
  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded with no warning.\n");
  if (OPERATING_SYSTEM==UNIX){
    if (TARGET==GALIB) {
      printf ("\nYou may now compile your file with:\n");
      printf ("\ng++ %s.cpp -o %s %sgdc.o %sgdchart.o %sprice_conv.o -lga -L%sgdchart0.94b/gd1.3 -lgd -lm\n\n",sPROJECT_NAME,sPROJECT_NAME,sEZ_PATH,sEZ_PATH,sEZ_PATH,sEZ_PATH);
    }
    if (TARGET==EO) {
      printf ("\nYou may now compile your file with:\n");
      printf ("\nmake -f %s.mak\n\n",sPROJECT_NAME);
      printf ("and run it with:\n\n",sPROJECT_NAME);
      printf ("%s @%s.prm\n\n",sPROJECT_NAME,sPROJECT_NAME);
    } }
  printf ("Have a nice compile time.\n");
  if (TARGET==EO) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 }

<TEMPLATE_ANALYSIS>.|\n {putc(yytext[0],fpOutputFile);}                                      

//****************************************
//  use information found in .ez file
//****************************************

<COPY_EO_INITIALISER>"\\GenomeClass::initiali"[sz]"er"[ \t\n]*":" {
  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  BEGIN COPY;
 }                                                               
<COPY_EO_INITIALISER><<EOF>> {
  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_EO_INITIALISER>.|\n {}                                      

<COPY_USER_DECLARATIONS>"\\User"[ \t\n]+"declarations"[ \t\n]*":" {
  fprintf (fpOutputFile,"// User declarations\n"); 
  BEGIN COPY;
 }                                                               
<COPY_USER_DECLARATIONS><<EOF>> {
  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_USER_DECLARATIONS>.|\n {}                                      

<COPY_USER_FUNCTIONS>"\\User"[ \t\n]+"functions"[ \t\n]*":" {
  fprintf (fpOutputFile,"// User functions\n\n"); 
  BEGIN COPY;
 }                                                               
<COPY_USER_FUNCTIONS><<EOF>> {
  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_USER_FUNCTIONS>.|\n {}                                      

<COPY_INITIALISATION_FUNCTION>"\\Initialisation"[ \t\n]+"function"[ \t\n]*":" {
  if (TARGET==DREAM)
    fprintf (fpOutputFile,"// Evaluator Constructor\n\n  public %sEvaluator(){",sPROJECT_NAME);
  else
    fprintf (fpOutputFile,"// Initialisation function\n\nvoid EASEAInitFunction(int argc, char *argv[]){");
  bFunction=1; bInitFunction=1;
      
  BEGIN COPY;
 }
<COPY_INITIALISATION_FUNCTION><<EOF>> {bInitFunction=0; // No initialisation function was found in the .ez file
  if (bVERBOSE) printf("*** No initialisation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No initialisation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_INITIALISATION_FUNCTION>.|\n {}                                      

<COPY_GENERATION_FUNCTION>"\\At"[ \t\n]+"each"[ \t\n]+"new"[ \t\n]+"generation"[ \t\n]*":" {
  fprintf (fpOutputFile,"// Function called at each new generation\n\nvoid EASEAGenerationFunction(GAGeneticAlgorithm & g){\n");
  fprintf(fpOutputFile,"  const GAPopulation *pPopulation;\n",sPROJECT_NAME);
  fprintf(fpOutputFile,"  pPopulation=&(g.population());  // to circumvent a bug in GALib\n",sPROJECT_NAME);
  bFunction=1; bGenerationFunction=1;
  BEGIN COPY_USER_GENERATION;
 }
<COPY_GENERATION_FUNCTION><<EOF>> {bGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_GENERATION_FUNCTION>.|\n {}                                      

<ANALYSE_USER_CLASSES>"\\User"[ \t\n]+"classes"[ \t\n]*":" {
  BEGIN GENOME_ANALYSIS; return CLASSES;}
<ANALYSE_USER_CLASSES>.|\n {}                                      

//****************************************
//  Basic copy to cpp file with minor changes
//****************************************

<COPY>"\""  {(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");}
<COPY>"\\\""  {fprintf(fpOutputFile,"\\\"");}

<COPY>"currentGeneration" |
<COPY>"currentGeneration"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"EZ_currentGeneration"); break;
    case EO : fprintf(fpOutputFile,"generationCounter.value()");
    }} // local genome name
<COPY>"NB_GEN" |
<COPY>"NB_GEN"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"EZ_NB_GEN");} // local genome name
<COPY>"POP_SIZE" |
<COPY>"POP_SIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");} // local genome name
<COPY>"MUT_PROB" |
<COPY>"MUT_PROB"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");} // local genome name
<COPY>"XOVER_PROB" |
<COPY>"XOVER_PROB"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");} // local genome name
<COPY>"REPL_PERC" |
<COPY>"REPL_PERC"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");} // local genome name
<COPY>"MINIMISE" |
<COPY>"MINIMISE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");} // local genome name
<COPY>"MINIMIZE" |
<COPY>"MINIMIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");} // local genome name

<COPY>"currentGeneration"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);}
<COPY>"NB_GEN"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"POP_SIZE"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"MUT_PROB"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"XOVER_PROB"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"REPL_PERC"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"MINIMISE"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"MINIMIZE"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"false"  {if (TARGET==GALIB) fprintf(fpOutputFile,"gaFalse");
  else fprintf(fpOutputFile,"false");} // local name
<COPY>"true"  {if (TARGET==GALIB) fprintf(fpOutputFile,"gaTrue");
  else fprintf(fpOutputFile,"true");} // local name
<COPY>[^a-zA-Z0-9_]"bool"[^a-zA-Z0-9_]  {if (TARGET==DREAM) fprintf(fpOutputFile," boolean ");
  else fprintf(fpOutputFile,yytext);}
<COPY>"tossCoin"  {if (TARGET==GALIB) fprintf(fpOutputFile,"tossCoin");
  if (TARGET==EO) fprintf(fpOutputFile,"rng.flip");
  if (TARGET==DREAM) fprintf(fpOutputFile,"Math.random()<");
 } // local random name 
<COPY>"random"  {if (TARGET==DREAM) fprintf(fpOutputFile,"%s.random",sPROJECT_NAME);
  else fprintf(fpOutputFile,"random");}
<COPY>"Genome" {if (bWithinEO_Function) fprintf(fpOutputFile,"_genotype");
  else fprintf(fpOutputFile,"Genome");} // local genome name
<COPY>"identicalGenome"  {fprintf(fpOutputFile,"genome._evaluated");} // local name
<COPY>"#define"[ \t]*  {fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;} 
<COPY>[a-zA-Z0-9_]*[ \t]+"GenomeClass::" {
  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);}         
<COPY>"GenomeClass::"  {fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);}
<COPY>"GenomeClass"  {if (TARGET==EO) fprintf(fpOutputFile,"Indi");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);} // local name
<COPY>"\\end"  {if (bFunction==1) {fprintf (fpOutputFile,"}\n"); bFunction=0;}
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;} // Back to the template file
<COPY>.|\n {putc(yytext[0],fpOutputFile);}                                      

// Looking for an identifier
<MACRO_IDENTIFIER>[a-zA-Z0-9_]* {fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; }
<MACRO_IDENTIFIER>/.|\n {BEGIN COPY;} // If no identifier was found, we're not interested
// Looking for the first number following the macro definition
<MACRO_DEFINITION>[ \t]*  {fprintf(fpOutputFile,"%s",yytext);} // gobbles up spaces and tabs
<MACRO_DEFINITION>[0-9]+"."[0-9]*{exponent}?  |
<MACRO_DEFINITION>"."[0-9]+{exponent}?  |
<MACRO_DEFINITION>[0-9]+{exponent} {fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;}
<MACRO_DEFINITION>[0-9]+           {fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;}
<MACRO_DEFINITION>/.|\n {if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;} // If no number was found, we're not interested

//****************************************
//  GENOME_ANALYSIS lexer
//****************************************

// takes care of C++-like comments
<GENOME_ANALYSIS>\/\/[^\n]* ;         // blah blah on a line containing sth else
<GENOME_ANALYSIS>^[ \t]*\/\/[^\n]*\n ; // blah blah on its own on a single line
<GENOME_ANALYSIS>\/\*[^\*]*\*\/ ;  /* blah blah on a line containing sth else */
<GENOME_ANALYSIS>^[ \t]*\/\*[^\*]*\*\/[ \t]*\n ; /* blah blah with nothing before the comment */

// gobbles up white spaces, tabs or carriage returns
<GENOME_ANALYSIS>[ \t]             { /* do nothing */ }
<GENOME_ANALYSIS>^[ \t]*\n      { /*return '\n';*/ }
<GENOME_ANALYSIS>\n                 { /*return '\n';*/ }

// keywords
<GENOME_ANALYSIS>"bool"                  {if (TARGET!=DREAM) yylval.pSymbol = pSymbolTable->find("bool");
  else yylval.pSymbol = pSymbolTable->find("boolean");
  return BOOL;}
<GENOME_ANALYSIS>"boolean"            {if (TARGET==DREAM){
    yylval.pSymbol = pSymbolTable->find("boolean");
    return BOOL;
  }
  else {
    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
  }  }
<GENOME_ANALYSIS>"static"                  {return STATIC;}
<GENOME_ANALYSIS>"int"                  {yylval.pSymbol = pSymbolTable->find("int"); return INT;}
<GENOME_ANALYSIS>"double"                  {yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;}
<GENOME_ANALYSIS>"float"                  {yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;}
<GENOME_ANALYSIS>"char"                  {yylval.pSymbol = pSymbolTable->find("char"); return CHAR;}
<GENOME_ANALYSIS>"pointer"                  {yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;}
<GENOME_ANALYSIS>"\\end"   {rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;} // Back to the template file
  
<GENOME_ANALYSIS>"GenomeClass"        {return GENOME; }                         

<GENOME_ANALYSIS>"Methods:" {BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;}
// number
<GENOME_ANALYSIS>[0-9]+"."[0-9]*{exponent}?  |
<GENOME_ANALYSIS>"."[0-9]+{exponent}?  |
<GENOME_ANALYSIS>[0-9]+{exponent}     { yylval.dValue = myStrtod(); return NUMBER; }
<GENOME_ANALYSIS>[0-9]+ {yylval.dValue=atoi(yytext); return NUMBER;}

// identifier
<GENOME_ANALYSIS>[a-zA-Z_][a-zA-Z0-9_]*    { yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; }
<GET_METHODS>[^\}]*   {BEGIN GENOME_ANALYSIS; return END_METHODS;}

//****************************************
//  Looking for a user-supplied display function.
//****************************************

<COPY_DISPLAY>"\\GenomeClass::display"[ \t\n]*":" { 
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 }
<COPY_DISPLAY><<EOF>> {bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_DISPLAY>.|\n {}                                      

//****************************************
//  Looks for standard and user functions in the .ez file
//****************************************

<COPY_INITIALISER>"\\GenomeClass::initiali"[sz]"er"[ \t\n]*":" {
  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  return USER_CTOR;
 }
<COPY_INITIALISER>.|\n {}                                      
<COPY_CROSSOVER>"\\GenomeClass::crossover"[ \t\n]*":" {
  bWithinXover=1;
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 }
<COPY_CROSSOVER>.|\n {}                                      
<COPY_MUTATOR>"\\GenomeClass::mutator"[ \t\n]*":" {
  bWithinMutator=1;
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 }
<COPY_MUTATOR>.|\n {}                                      
<COPY_EVALUATOR>"\\GenomeClass::evaluator"[ \t\n]*":" {
  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  return USER_EVALUATOR;
 }
<COPY_EVALUATOR>.|\n {}                                      

//****************************************
//  Basic copy to .cpp file with major changes
//****************************************

<COPY_USER_GENERATION>\/\/[^\n]* {fprintf(fpOutputFile,yytext);}         // blah blah on a line containing sth else
<COPY_USER_GENERATION>^[ \t]*\/\/[^\n]*\n {fprintf(fpOutputFile,yytext);} // blah blah on its own on a single line
<COPY_USER_GENERATION>\/\*[^\*]*\*\/ {fprintf(fpOutputFile,yytext);}  /* blah blah on a line containing sth else */
<COPY_USER_GENERATION>^[ \t]*\/\*[^\*]*\*\/[ \t]*\n {fprintf(fpOutputFile,yytext);} /* blah blah with nothing before the comment */
<COPY_USER_GENERATION>"Genome"  {fprintf(fpOutputFile,"genome");} // local genome name
<COPY_USER_GENERATION>"\""  {(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");}
<COPY_USER_GENERATION>"\\\""  {fprintf(fpOutputFile,"\\\"");}

<COPY_USER_GENERATION>"pPopulation"[ \t\n]*"[" {bWaitingToClosePopulation=true;
  switch (TARGET) {
  case EO: fprintf(fpOutputFile,"pPopulation ["); break;
  case GALIB: fprintf(fpOutputFile,"((%sGenome *)&(pPopulation->individual(",sPROJECT_NAME);
  }
 }
<COPY_USER_GENERATION>"]" {if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {fprintf(fpOutputFile,")))"); bWaitingToClosePopulation=false;}}

<COPY_USER_GENERATION>"currentGeneration" |
<COPY_USER_GENERATION>"currentGeneration"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.generation()"); break;
    case EO : fprintf(fpOutputFile,"generationCounter.value()");
    }} // local genome name
<COPY_USER_GENERATION>"NB_GEN" |
<COPY_USER_GENERATION>"NB_GEN"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.nGenerations()"); break;
    case EO : fprintf(fpOutputFile,"ptEZ_NbGen.value()");
    }} // local genome name                           
<COPY_USER_GENERATION>"POP_SIZE" |
<COPY_USER_GENERATION>"POP_SIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.populationSize()"); break;
    case EO : fprintf(fpOutputFile,"EZ_POP_SIZE");
    }} // local genome name
<COPY_USER_GENERATION>"MUT_PROB" |
<COPY_USER_GENERATION>"MUT_PROB"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.pMutation()"); break;
    case EO : fprintf(fpOutputFile,"EZ_MUT_PROB");
    }} // local genome name
<COPY_USER_GENERATION>"XOVER_PROB" |
<COPY_USER_GENERATION>"XOVER_PROB"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.pCrossover()"); break;
    case EO : fprintf(fpOutputFile,"EZ_XOVER_PROB");
    }} // local genome name
<COPY_USER_GENERATION>"REPL_PERC" |
<COPY_USER_GENERATION>"REPL_PERC"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.pReplacement()*100"); break;
    case EO : fprintf(stderr,"\n%s - Error line %d: The REPL_PERC variable cannot be accessed yet undeer EO.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }} // local genome name
<COPY_USER_GENERATION>"MINIMISE" |
<COPY_USER_GENERATION>"MINIMISE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.minimaxi()"); break;
    case EO : fprintf(stderr,"\n%s - Error line %d: The MINIMISE variable cannot be accessed yet undeer EO.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }} // local genome name
<COPY_USER_GENERATION>"MINIMIZE" |
<COPY_USER_GENERATION>"MINIMIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.minimaxi()"); break;
    case EO : fprintf(stderr,"\n%s - Error line %d: The MINIMIZE variable cannot be accessed yet undeer EO.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }} // local genome name

<COPY_USER_GENERATION>"currentGeneration"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 } 
<COPY_USER_GENERATION>"NB_GEN"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.nGenerations((EZ_NB_GEN=");
      bWaitingForSemiColon=bFinishNB_GEN=1; break;
    case EO : fprintf(fpOutputFile,"ptEZ_NbGen.value((EZ_NB_GEN=)");
      bWaitingForSemiColon=1;
    }} // local genome name
<COPY_USER_GENERATION>"POP_SIZE"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.populationSize((EZ_POP_SIZE=");
      bWaitingForSemiColon=1; break;
    case EO : fprintf(stderr,"\n%s - Error line %d: The population size cannot be changed during the run in EO yet.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }} // local genome name
<COPY_USER_GENERATION>"MUT_PROB"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.pMutation((EZ_MUT_PROB=");
      bWaitingForSemiColon=1; break;
    case EO : fprintf(fpOutputFile,"EZ_MUT_PROB=");
    }} // local genome name
<COPY_USER_GENERATION>"XOVER_PROB"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB :fprintf(fpOutputFile,"g.pCrossover((EZ_XOVER_PROB=");
      bWaitingForSemiColon=1; break;
    case EO : fprintf(fpOutputFile,"EZ_MUT_PROB=");
    }} // local genome name
<COPY_USER_GENERATION>"REPL_PERC"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB :fprintf(fpOutputFile,"g.pReplacement((1/100)*(EZ_REPL_PERC=");
      bWaitingForSemiColon=1; break;
    case EO : fprintf(stderr,"\n%s - Error line %d: REPL_PERC cannot be changed during the run in EO yet.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }} // local genome name
<COPY_USER_GENERATION>"MINIMISE"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB :fprintf(fpOutputFile,"MINIMISE=");
      bWaitingForSemiColon=bFinishMINIMISE=1; break;
    case EO : fprintf(stderr,"\n%s - Error line %d: MINIMISE cannot be changed during the run in EO yet.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }}
<COPY_USER_GENERATION>"MINIMIZE"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB :fprintf(fpOutputFile,"MINIMIZE=");
      bWaitingForSemiColon=bFinishMINIMISE=1; break;
    case EO : fprintf(stderr,"\n%s - Error line %d: MINIMIZE cannot be changed during the run in EO yet.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }}
<COPY_USER_GENERATION>"false"  {if (TARGET==GALIB) fprintf(fpOutputFile,"gaFalse");
  else fprintf(fpOutputFile,"false");} // local name
<COPY_USER_GENERATION>"true"  {if (TARGET==GALIB) fprintf(fpOutputFile,"gaTrue");
  else fprintf(fpOutputFile,"true");} // local name
<COPY_USER_GENERATION>[^a-zA-Z0-9_]"bool"[^a-zA-Z0-9_]  {if (TARGET==DREAM) fprintf(fpOutputFile," boolean ");
  else fprintf(fpOutputFile,yytext);}
<COPY_USER_GENERATION>"tossCoin"  {if (TARGET==GALIB) fprintf(fpOutputFile,"tossCoin");
  if (TARGET==EO) fprintf(fpOutputFile,"rng.flip");
  if (TARGET==DREAM) fprintf(fpOutputFile,"Math.random()<");
 } // local random name
<COPY_USER_GENERATION>"random"  {if (TARGET==DREAM) fprintf(fpOutputFile,"%s.random",sPROJECT_NAME);
  else fprintf(fpOutputFile,"random");}
<COPY_USER_GENERATION>"identicalGenome"  {fprintf(fpOutputFile,"genome._evaluated");} // local name
<COPY_USER_GENERATION>"\\end" {rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");} // Back to the template file
<COPY_USER_GENERATION>";" {if (bWaitingForSemiColon){
    bWaitingForSemiColon=0;
    if (bFinishMINIMISE) {fprintf(fpOutputFile,");\n  if (MINIMISE) g.minimize() else g.maximize();\n"); bFinishMINIMISE=0;}
    if (bFinishMINIMIZE) {fprintf(fpOutputFile,");\n  if (MINIMIZE) g.minimize() else g.maximize();\n"); bFinishMINIMISE=0;}
    if ((bFinishNB_GEN)&&(OPERATING_SYSTEM==UNIX))
      {fprintf(fpOutputFile,"));\n  if ( (EZ_daFITNESS = (double *) realloc(EZ_daFITNESS, (EZ_NB_GEN +1)* sizeof (double) )) == NULL){\n");
	fprintf(fpOutputFile,"    fprintf(stderr,\"Not enough memory... bailing out.\");\n    exit(1);");}
    else if (bFinishNB_GEN) {fprintf(fpOutputFile,"));"); bFinishNB_GEN=0;}
    else fprintf(fpOutputFile,"));");
  }
  else fprintf(fpOutputFile,";");}

<COPY_USER_GENERATION>.|\n {putc(yytext[0],fpOutputFile);}                                      

//****************************************
//  Basic copy to .cpp file with minor changes
//****************************************

// takes care of C++-like comments
<COPY_USER_FUNCTION>\/\/[^\n]* {fprintf(fpOutputFile,yytext);}         // blah blah on a line containing sth else
<COPY_USER_FUNCTION>^[ \t]*\/\/[^\n]*\n {fprintf(fpOutputFile,yytext);} // blah blah on its own on a single line
<COPY_USER_FUNCTION>\/\*[^\*]*\*\/ {fprintf(fpOutputFile,yytext);}  /* blah blah on a line containing sth else */
<COPY_USER_FUNCTION>^[ \t]*\/\*[^\*]*\*\/[ \t]*\n {fprintf(fpOutputFile,yytext);} /* blah blah with nothing before the comment */

<COPY_USER_FUNCTION>"GenomeClass"  {if (TARGET==EO) fprintf(fpOutputFile, "GenotypeT");
  else fprintf(fpOutputFile,"Genome.");
 }
<COPY_USER_FUNCTION>"Genome"  {if (bWithinDisplayFunction) fprintf(fpOutputFile,"(*this)");
  else if ((TARGET==EO)&&(bWithinInitialiser)) fprintf(fpOutputFile, "(*genome)");
  else if ((TARGET==EO)&&(bWithinMutator)) fprintf(fpOutputFile, "_genotype");
  else fprintf(fpOutputFile,"genome");} // local genome name
<COPY_USER_FUNCTION>"\""  {(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");}
<COPY_USER_FUNCTION>"\\\""  {fprintf(fpOutputFile,"\\\"");}

<COPY_USER_FUNCTION>"pPopulation"[ \t\n]*"[" {bWaitingToClosePopulation=true;
  switch (TARGET) {
  case EO: fprintf(fpOutputFile,"pPopulation ["); break;
  case GALIB: fprintf(fpOutputFile,"((%sGenome *)&(pPopulation->individual(",sPROJECT_NAME);
  }
 }
<COPY_USER_FUNCTION>"]" {if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {fprintf(fpOutputFile,")))"); bWaitingToClosePopulation=false;}}

<COPY_USER_FUNCTION>"currentGeneration" |
<COPY_USER_FUNCTION>"currentGeneration"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"EZ_currentGeneration"); break;
    case EO : fprintf(fpOutputFile,"generationCounter.value()");
    }} // local genome name
<COPY_USER_FUNCTION>"NB_GEN" |
<COPY_USER_FUNCTION>"NB_GEN"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"EZ_NB_GEN");} // local genome name
<COPY_USER_FUNCTION>"POP_SIZE" |
<COPY_USER_FUNCTION>"POP_SIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");} // local genome name
<COPY_USER_FUNCTION>"MUT_PROB" |
<COPY_USER_FUNCTION>"MUT_PROB"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");} // local genome name
<COPY_USER_FUNCTION>"XOVER_PROB" |
<COPY_USER_FUNCTION>"XOVER_PROB"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");} // local genome name
<COPY_USER_FUNCTION>"REPL_PERC" |
<COPY_USER_FUNCTION>"REPL_PERC"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");} // local genome name

<COPY_USER_FUNCTION>"currentGeneration"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);}
<COPY_USER_FUNCTION>"NB_GEN"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY_USER_FUNCTION>"POP_SIZE"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY_USER_FUNCTION>"MUT_PROB"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY_USER_FUNCTION>"XOVER_PROB"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY_USER_FUNCTION>"REPL_PERC"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY_USER_FUNCTION>"false"  {if (TARGET==GALIB) fprintf(fpOutputFile,"gaFalse");
  else fprintf(fpOutputFile,"false");} // local name
<COPY_USER_FUNCTION>"true"  {if (TARGET==GALIB) fprintf(fpOutputFile,"gaTrue");
  else fprintf(fpOutputFile,"true");} // local name
<COPY_USER_FUNCTION>[^a-zA-Z0-9_]"bool"[^a-zA-Z0-9_]  {if (TARGET==DREAM) fprintf(fpOutputFile," boolean ");
  else fprintf(fpOutputFile,yytext);}
<COPY_USER_FUNCTION>"tossCoin"  {if (TARGET==GALIB) fprintf(fpOutputFile,"tossCoin");
  if (TARGET==EO) fprintf(fpOutputFile,"rng.flip");
  if (TARGET==DREAM) fprintf(fpOutputFile,"Math.random()<");} // local random name
<COPY_USER_FUNCTION>"random"  {if (TARGET==DREAM) fprintf(fpOutputFile,"%s.random",sPROJECT_NAME);
  else fprintf(fpOutputFile,"random");}
<COPY_USER_FUNCTION>"child1"  {if ((bWithinXover)&&(TARGET==GALIB)) fprintf(fpOutputFile, "(*pBro)");
  else fprintf(fpOutputFile,"child1");
 }
<COPY_USER_FUNCTION>"child2"  {if ((bWithinXover)&&(TARGET==GALIB)) fprintf(fpOutputFile, "(*pSis)");
  else fprintf(fpOutputFile,"child2");
 }
<COPY_USER_FUNCTION>"parent1"  {if ((bWithinXover)&&(TARGET==GALIB)) fprintf(fpOutputFile, "(*pDad)");
  else fprintf(fpOutputFile,"parent1");
 }
<COPY_USER_FUNCTION>"parent2"  {if ((bWithinXover)&&(TARGET==GALIB)) fprintf(fpOutputFile, "(*pMom)");
  else fprintf(fpOutputFile,"parent2");
 }
<COPY_USER_FUNCTION>"identicalGenome"  {fprintf(fpOutputFile,"genome._evaluated");} // local name
<COPY_USER_FUNCTION>"return"  {if (bWithinEvaluator) {
    if (TARGET==GALIB) fprintf(fpOutputFile,"EZ_EVAL+=(double)(clock()-EZ_t1);\n  return (float)");
    if (TARGET==EO)  {fprintf(fpOutputFile,"genome.fitness(");bCatchNextSemiColon=true;}// changes function type// changes function type
    if (TARGET==DREAM) {fprintf(fpOutputFile,"infoHabitant.setFitness(new Double(");bCatchNextSemiColon=true;}// changes function type
    bWithinEvaluator=0;
  }
  else if ((bWithinMutator)&&(TARGET!=GALIB)) {
    fprintf(fpOutputFile,"return ");
    bCatchNextSemiColon=true;
  }
  else fprintf(fpOutputFile,"return"); }
<COPY_USER_FUNCTION>";" {if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if ((TARGET!=GALIB)&&(bWithinMutator)){fprintf(fpOutputFile,">0?true:false;"); bWithinMutator=false;}
  else if (TARGET==EO) fprintf(fpOutputFile,");");
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 }
<COPY_USER_FUNCTION>"\\end" {rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=0;
  if (bWithinDisplayFunction) bWithinDisplayFunction=0; // display function
  else return END_OF_FUNCTION;} // Back to the template file
<COPY_USER_FUNCTION>.|\n {putc(yytext[0],fpOutputFile);}                                      

//****************************************
//  Looks for run parameters in the .ez file
//****************************************

<PARAMETERS_ANALYSIS>"\\Default run"[ \t\n]+"parameters"[ \t\n]*":" {
  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 }
<PARAMETERS_ANALYSIS>.|\n {}                                      

// takes care of C++-like comments
<GET_PARAMETERS>\/\/[^\n]* ;         // blah blah on a line containing sth else
<GET_PARAMETERS>^[ \t]*\/\/[^\n]*\n ; // blah blah on its own on a single line
<GET_PARAMETERS>\/\*[^\*]*\*\/ ;  /* blah blah on a line containing sth else */
<GET_PARAMETERS>^[ \t]*\/\*[^\*]*\*\/[ \t]*\n ; /* blah blah with nothing before the comment */
// gobbles up white spaces, tabs or carriage returns
<GET_PARAMETERS>[ \t]             { /* do nothing */ }
<GET_PARAMETERS>^[ \t]*\n      { /*return '\n';*/ }
<GET_PARAMETERS>\n                 { /*return '\n';*/ }

<GET_PARAMETERS>"Number"[ \t\n]+"of"[ \t\n]+"generations"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;}
<GET_PARAMETERS>"Number"[ \t\n]+"of"[ \t\n]+"islands"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tNb of Isl...\n");return NB_ISLANDS;}
<GET_PARAMETERS>"Operators"[ \t\n]+"are"[ \t\n]+"called"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tOp Seq...\n");return PROP_SEQ;}
<GET_PARAMETERS>"Evolutionary"[ \t\n]+"engine"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tEvol Engine...\n");return REPLACEMENT;}
<GET_PARAMETERS>"Population"[ \t\n]+"size"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;}
<GET_PARAMETERS>"Elite"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;}
<GET_PARAMETERS>"Fertility"[ \t\n]*":"[ \ta-zA-Z0-9_]* {if (bVERBOSE) printf ("\tFertil...\n");}
<GET_PARAMETERS>"Genitors"[ \t\n]+"selector"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tGenitors Sel...\n");return SELECTOR;}
<GET_PARAMETERS>"Selected"[ \t\n]+"genitors"[ \t\n]*":"[ \ta-zA-Z0-9_]* {if (bVERBOSE) printf ("\tSel Genitors...\n");}
<GET_PARAMETERS>"Mutation"[ \t\n]+"probability"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;}
<GET_PARAMETERS>"Crossover"[ \t\n]+"probability"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;}
<GET_PARAMETERS>"Offspring"[ \t\n]+"size"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;}
<GET_PARAMETERS>"Reduce"[ \t\n]+"parents"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tReduce Par...\n");return RED_PAR;}
<GET_PARAMETERS>"Surviving"[ \t\n]+"parents"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;}
<GET_PARAMETERS>"Reduce"[ \t\n]+"offspring"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tRed Off...\n");return RED_OFF;}
<GET_PARAMETERS>"Surviving"[ \t\n]+"offspring"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;}
<GET_PARAMETERS>"Final"[ \t\n]+"reduce"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tFinal Red...\n");return RED_FINAL;}// DISCARD;}
<GET_PARAMETERS>"Elitism"[ \t\n]*":" {if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;}
<GET_PARAMETERS>"Evaluator"[ \t\n]+"goal"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;}
<GET_PARAMETERS>"Emigration"[ \t\n]+"policy"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tMig Policy...\n");return MIG_CLONE;}
<GET_PARAMETERS>"Migrants"[ \t\n]+"selector"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tMig Sel...\n");return MIG_SEL;}
<GET_PARAMETERS>"Migrants"[ \t\n]+"destination"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tMig Dest...\n");return MIGRATOR;}
<GET_PARAMETERS>"Migration"[ \t\n]+"frequency"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tMig Freq...\n");return MIG_FREQ;}
<GET_PARAMETERS>"Number"[ \t\n]+"of"[ \t\n]+"emigrants"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tNb Mig...\n");return NB_MIG;}
<GET_PARAMETERS>"Immigration"[ \t\n]+"replacement"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tImmig Repl...\n");return IMMIG_SEL;}
<GET_PARAMETERS>"Immigration"[ \t\n]+"policy"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tImmig Policy...\n");return IMMIG_REPL;}
 
// number
<GET_PARAMETERS>[0-9]+"."[0-9]*{exponent}?  |
<GET_PARAMETERS>"."[0-9]+{exponent}?  |
<GET_PARAMETERS>[0-9]+{exponent}     { yylval.dValue = myStrtod(); return NUMBER2; }
<GET_PARAMETERS>[0-9]+ {yylval.dValue=atof(yytext); return NUMBER2;}

// identifier
<GET_PARAMETERS>[a-zA-Z_][a-zA-Z0-9_]*    { yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; }
// end of .ez file : resume tpl analysis
<GET_PARAMETERS>"\\end" {rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;}  // switch back to .tpl file and analyser

//<GET_PARAMETERS>.|\n {}                                      


//****************************************
// all other characters                      
.                      {return  (char)yytext[0];}

%%

		       /////////////////////////////////////////////////////////////////////////////

inline char  mytolower(char c) {
  return ((c>=65)&&(c<=90)) ? c+=32:c;
}

inline int mystricmp(char *string1, char *string2){
  int i;
  for (i=0; string1[i]&&string2[i];i++){
    if (mytolower(string1[i])<mytolower(string2[i])) return -(i+1);
    if (mytolower(string1[i])>mytolower(string2[i])) return i+1;
  }
  if (string2[i]) return  -(i+1);
  if (string1[i]) return  i+1;
  return 0;
}                                  

inline int isLetter(char c){ 
  if (((c>=65)&&(c<=90))||((c>=97)&&(c<=122))) return 1;
  if ((c==45)||(c==46)||(c==95)) return 1;
  return 0;
}

inline int isFigure(char c){ 
  if ((c>=48)&&(c<=57)) return 1;
  return 0;
}

/////////////////////////////////////////////////////////////////////////////
// EASEALexer commands

int CEASEALexer::yywrap(){
  if (bNotFinishedYet) {bNotFinishedYet=0; return 0;}
  else return 1;
}

int CEASEALexer::create(CEASEAParser* pParser, CSymbolTable* pSymTable)
{ 
  int i;
  char sTemp[1000];
#if defined UNIX_OS
  OPERATING_SYSTEM=UNIX;
#elif defined WINDOWS_OS
  OPERATING_SYSTEM=WINDOWS;
#else
  OPERATING_SYSTEM=OTHER_SYSTEM;
#endif
  assert(pParser != NULL);
  assert(pSymTable != NULL);
  
  pSymbolTable = pSymTable;
  if (!yycreate(pParser)) return 0;    

  if (bVERBOSE) printf("\n                                                                   ");
  if (bVERBOSE) printf("\n                                   E A S E A                   (v0.7b)");
  if (bVERBOSE) printf("\n                              ___________________     ");
  if (bVERBOSE) printf("\n                                                                    ");
  
  if (sRAW_PROJECT_NAME[0]==0){
    printf("\nInsert a .ez file name or a local project name: ");
    scanf("%s",sRAW_PROJECT_NAME);
  }                         
  if (bVERBOSE) printf("\n");
  
  if (TARGET==0) {
    printf("\nPlease select a target library (DREAM, EO or GALib): ");
    scanf("%s",sTemp);
    if (!mystricmp(sTemp,"eo")) TARGET=EO;
    else if (!mystricmp(sTemp,"galib")) TARGET=GALIB;
    else if (!mystricmp(sTemp,"dream")) TARGET=DREAM;
    else return 0;
  }
                                                                           
  /////////////////////////////////////////////////////////  
  //strcpy(sTemp,"e:\\lutton\\easea\\debug");pour tester sous windows
  if ((sEZ_PATH==NULL)||(sEZ_PATH[0]==0)) {
    if (getenv("EZ_PATH")==NULL){
      printf("\n\nHmmm, it looks like you are running EASEA without GUIDE for the first time.\n");
      printf("Please, add the path of the installation directory to the PATH variable and\n");
      printf("into the \"EZ_PATH\" environment variable, so that EASEA knows where to look for\n");
      printf("its template files.\n");                                                           
      exit(1);
    }
    strcpy(sEZ_PATH,getenv("EZ_PATH"));
  }
  
  switch (OPERATING_SYSTEM) {
  case UNIX : if (sEZ_PATH[strlen(sEZ_PATH)-1] != '/') strcat (sEZ_PATH,"/"); break;
  case WINDOWS : if (sEZ_PATH[strlen(sEZ_PATH)-1] != '\\') strcat (sEZ_PATH,"\\"); break;
  case UNKNOWN_OS : fprintf(fpOutputFile,"UNKNOWN_OS"); break;
  }
  strcpy(sTemp,sEZ_PATH);
  if (TARGET==EO){
    strcat(sTemp,"EO.tpl");
    if (!(yyin = fpTemplateFile = fopen(sTemp, "r"))){
      fprintf(stderr,"\n*** Could not open %s.\n",sTemp);
      fprintf(stderr,"*** Please modify the EZ_PATH environment variable.\n");
      exit(1);
    } }
  if (TARGET==GALIB){
    strcat(sTemp,"GALib.tpl");
    if (!(yyin = fpTemplateFile = fopen(sTemp, "r"))){
      fprintf(stderr,"\n*** Could not open %s.\n",sTemp);
      fprintf(stderr,"*** Please modify the EZ_PATH environment variable.\n");
      exit(1);
    } }
  if (TARGET==DREAM){
    strcat(sTemp,"DREAM.tpl");
    if (!(yyin = fpTemplateFile = fopen(sTemp, "r"))){
      fprintf(stderr,"\n*** Could not open %s.\n",sTemp);
      fprintf(stderr,"*** Please modify the EZ_PATH environment variable.\n");
      exit(1);
    } }

  if (TARGET==EO){
    if ((sEO_DIR==NULL)||(sEO_DIR[0]==0)) {
      if ((getenv("EO_DIR")==NULL)){
        printf("Please, set the \"EO_DIR\" environment variable to the EO installation directory.\n");
        exit(1);
      }
      strcpy(sEO_DIR,getenv("EO_DIR"));
    }
    switch (OPERATING_SYSTEM) {
    case UNIX : if (sEO_DIR[strlen(sEO_DIR)-1] != '/') strcat (sEO_DIR,"/"); break;
    case WINDOWS : if (sEO_DIR[strlen(sEO_DIR)-1] != '\\') strcat (sEO_DIR,"\\"); break;
    case UNKNOWN_OS : fprintf(fpOutputFile,"UNKNOWN_OS"); break;
    }
  }
  
  if ((sRAW_PROJECT_NAME[0]=='"')&&(OPERATING_SYSTEM!=WINDOWS)){
    strcpy(sRAW_PROJECT_NAME,&(sRAW_PROJECT_NAME[1]));
    sRAW_PROJECT_NAME[strlen(sRAW_PROJECT_NAME)-1]=0;
  }
  if (strlen(sRAW_PROJECT_NAME)>3) 
    if (!mystricmp(".EZ",&(sRAW_PROJECT_NAME[strlen(sRAW_PROJECT_NAME)-3])))
      sRAW_PROJECT_NAME[strlen(sRAW_PROJECT_NAME)-3]=0;

  strcpy(sEZ_FILE_NAME, sRAW_PROJECT_NAME);
  strcat(sEZ_FILE_NAME,".ez");

  for (i=strlen(sRAW_PROJECT_NAME)-1;isFigure(sRAW_PROJECT_NAME[i]) || isLetter(sRAW_PROJECT_NAME[i]);i--);
  strcpy (sPROJECT_NAME,&(sRAW_PROJECT_NAME[i+1]));
  
  for(i=0;i<(int)strlen(sPROJECT_NAME);i++) sLOWER_CASE_PROJECT_NAME[i]=mytolower(sPROJECT_NAME[i]);
  
  if ((!isLetter(sPROJECT_NAME[0]))&&(sPROJECT_NAME[0]!='"')&&(sPROJECT_NAME[0]!='/')&&(sPROJECT_NAME[0]!='\\')) {
    fprintf(stderr,"\n*** Project names starting with non-letters are invalid.\n*** Please choose another name.\n"); return 0;}
                                                                           
  if (!(fpGenomeFile = fopen(sEZ_FILE_NAME, "r"))){
    fprintf(stderr,"\n*** Could not open %s\n",sEZ_FILE_NAME); return 0;}
 
  return 1;
}


/////////////////////////////////////////////////////////////////////////////
// calc_lexer attribute commands

double CEASEALexer::myStrtod() const{
  errno = 0;    // clear error flag
  char* endp;
  double d = strtod(yytext, &endp);
  if ((d == +HUGE_VAL || d == -HUGE_VAL) && errno == ERANGE) {
    printf("number too large\n");
  }
  return d;
}                               
