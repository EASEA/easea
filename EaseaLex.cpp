#include <clex.h>

#line 2 "EaseaLex.l"

/****************************************************************************
EaseaLex.l
Lexical analyser for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@polytechnique.fr)
Ecole Polytechnique
Centre de Mathématiques Appliquées
91128 Palaiseau cedex
  ****************************************************************************/
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libAlexYacc.lib")
#endif
#include "Easea.h"
#include "EaseaParse.h"
#ifdef WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif
#include "debug.h"

  size_t genomeSize;
  bool genomeSizeValidity=false;
  int lineCounter = 0;

  // local functions
  char* selectorDetermination(int nMINIMISE, char* sSELECTOR){

  char selectorName[50];  
  selectorName[3] = 0;
  if( nMINIMISE )
    strcpy(selectorName,"Min");
  else
    strcpy(selectorName,"Max");
  

  if( mystricmp("Tournament",sSELECTOR)==0 )
    strcat(selectorName,"Tournament(globalRandomGenerator)");
  else if( mystricmp("dettour",sSELECTOR)==0 )
    strcat(selectorName,"Tournament(globalRandomGenerator)");
  else if( mystricmp("Deterministic",sSELECTOR)==0 )
    strcat(selectorName,"Deterministic()");
  else if( mystricmp("deterministic",sSELECTOR)==0 )
    strcat(selectorName,"Deterministic()");
  else if( mystricmp("Random",sSELECTOR)==0 )
    strcat(selectorName,"Random(globalRandomGenerator)");
  else if( mystricmp("Roulette",sSELECTOR)==0 )
    strcat(selectorName,"Roulette(globalRandomGenerator)");

  else{
    //DEBUG_PRT_PRT("%s is not yet implemented",sSELECTOR);
    return NULL;
  }
  
  char* ret = (char*)malloc((strlen(selectorName)+1)*sizeof(char));
  strcpy(ret,selectorName);

  return ret;
}

  
#line 68 "EaseaLex.cpp"
// repeated because of possible precompiled header
#include <clex.h>

#include "EaseaLex.h"

#line 72 "EaseaLex.l"
                                         
#line 80 "EaseaLex.l"
 // lexical analyser name and class definition
#line 78 "EaseaLex.cpp"
/////////////////////////////////////////////////////////////////////////////
// constructor

YYLEXNAME::YYLEXNAME()
{
	yytables();
#line 100 "EaseaLex.l"
                                
  bFunction=bWithinEvaluator=bWithinOptimiser=bDisplayFunction=bInitFunction=bNotFinishedYet=0;
  bSymbolInserted=bDoubleQuotes=bWithinDisplayFunction=bWithinInitialiser=bWithinMutator=bWithinXover=0;
  bWaitingForSemiColon=bFinishNB_GEN=bFinishMINIMISE=bFinishMINIMIZE=bGenerationReplacementFunction=0;
  bCatchNextSemiColon,bWaitingToClosePopulation=bMethodsInGenome=0;
  bBoundCheckingFunction = bWithinCUDA_Initializer=bWithinMAKEFILEOPTION =bWithinCUDA_Evaluator=0;
  bIsParentReduce = bIsOffspringReduce = false;
  bGenerationFunctionBeforeReplacement = bEndGeneration = bBeginGeneration = bEndGenerationFunction = bBeginGenerationFunction = false;

#line 95 "EaseaLex.cpp"
}

#ifndef YYTEXT_SIZE
#define YYTEXT_SIZE 100
#endif
#ifndef YYUNPUT_SIZE
#define YYUNPUT_SIZE YYTEXT_SIZE
#endif

// backwards compatability with lex
#ifdef input
int YYLEXNAME::yyinput()
{
	return input();
}
#else
#define input yyinput
#endif

#ifdef output
void YYLEXNAME::yyoutput(int ch)
{
	output(ch);
}
#else
#define output yyoutput
#endif

#ifdef unput
void YYLEXNAME::yyunput(int ch)
{
	unput(ch);
}
#else
#define unput yyunput
#endif

#ifndef YYNBORLANDWARN
#ifdef __BORLANDC__
#pragma warn -rch		// <warning: unreachable code> off
#endif
#endif

int YYLEXNAME::yyaction(int action)
{
#line 117 "EaseaLex.l"

  // extract yylval for use later on in actions
  YYSTYPE& yylval = *(YYSTYPE*)yyparserptr->yylvalptr;
  
#line 146 "EaseaLex.cpp"
	yyreturnflg = 1;
	switch (action) {
	case 1:
		{
#line 123 "EaseaLex.l"

#line 153 "EaseaLex.cpp"
		}
		break;
	case 2:
		{
#line 126 "EaseaLex.l"

  BEGIN TEMPLATE_ANALYSIS; yyless(yyleng-1);
 
#line 162 "EaseaLex.cpp"
		}
		break;
	case 3:
		{
#line 134 "EaseaLex.l"
             
  char sFileName[1000];
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".cpp"); 
  fpOutputFile=fopen(sFileName,"w");
 
#line 174 "EaseaLex.cpp"
		}
		break;
	case 4:
		{
#line 140 "EaseaLex.l"
fprintf(fpOutputFile,"EASEA");
#line 181 "EaseaLex.cpp"
		}
		break;
	case 5:
		{
#line 141 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sPROJECT_NAME);
#line 188 "EaseaLex.cpp"
		}
		break;
	case 6:
		{
#line 142 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sEZ_PATH);
#line 195 "EaseaLex.cpp"
		}
		break;
	case 7:
		{
#line 143 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sTPL_DIR);
#line 202 "EaseaLex.cpp"
		}
		break;
	case 8:
		{
#line 144 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sEO_DIR);
#line 209 "EaseaLex.cpp"
		}
		break;
	case 9:
		{
#line 145 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sLOWER_CASE_PROJECT_NAME);
#line 216 "EaseaLex.cpp"
		}
		break;
	case 10:
		{
#line 146 "EaseaLex.l"
switch (OPERATING_SYSTEM) {
  case UNIX : fprintf(fpOutputFile,"UNIX_OS"); break;
  case WINDOWS : fprintf(fpOutputFile,"WINDOWS_OS"); break;
  case UNKNOWN_OS : fprintf(fpOutputFile,"UNKNOWN_OS"); break;
  }
 
#line 228 "EaseaLex.cpp"
		}
		break;
	case 11:
		{
#line 152 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user declarations.\n");
  yyreset();
  yyin = fpGenomeFile;                                                    // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_USER_DECLARATIONS;
 
#line 241 "EaseaLex.cpp"
		}
		break;
	case 12:
		{
#line 159 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting initialisation function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_INITIALISATION_FUNCTION;
 
#line 253 "EaseaLex.cpp"
		}
		break;
	case 13:
		{
#line 166 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting generation before reduce function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bGenerationFunctionBeforeReplacement = true;
  BEGIN COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT;
 
#line 266 "EaseaLex.cpp"
		}
		break;
	case 14:
		{
#line 175 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  if (bVERBOSE) printf ("Inserting at the begining of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bBeginGeneration = true;
  bEndGeneration = false;
  BEGIN COPY_BEG_GENERATION_FUNCTION;
 
#line 281 "EaseaLex.cpp"
		}
		break;
	case 15:
		{
#line 186 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  yyreset();
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Evaluation population in a single function!!.\n");
  BEGIN COPY_INSTEAD_EVAL;
 
#line 294 "EaseaLex.cpp"
		}
		break;
	case 16:
		{
#line 195 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert end");
  if (bVERBOSE) printf ("Inserting at the end of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bEndGeneration = true;
  bBeginGeneration = false;
  BEGIN COPY_END_GENERATION_FUNCTION;
 
#line 309 "EaseaLex.cpp"
		}
		break;
	case 17:
		{
#line 206 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Bound Checking function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_BOUND_CHECKING_FUNCTION;
 
#line 321 "EaseaLex.cpp"
		}
		break;
	case 18:
		{
#line 214 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 
#line 333 "EaseaLex.cpp"
		}
		break;
	case 19:
		{
#line 221 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting User classes.\n");
  fprintf (fpOutputFile,"// User classes\n");
  CListItem<CSymbol*> *pSym;
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem())
    if ((pSym->Object->pType->ObjectType==oUserClass)&&(!pSym->Object->pType->bAlreadyPrinted)){
      //DEBUG_PRT_PRT("%p",pSym->Object->pType);
      pSym->Object->pType->printClasses(fpOutputFile);
    }


  if( TARGET == CUDA ){
    //DEBUG_PRT_PRT("User classes are :");
    for( int i = nClasses_nb-1 ; i>=0 ; i-- ){
      //DEBUG_PRT_PRT(" %s, %p ,%d| ",pCLASSES[i]->sName,pCLASSES[i],pCLASSES[i]->bAlreadyPrinted);
      if( !pCLASSES[i]->bAlreadyPrinted ){
	fprintf(fpOutputFile,"// User class not refereced by the Genome");
	pCLASSES[i]->printClasses(fpOutputFile);
      }
    }
    //DEBUG_PRT_PRT("\n");
  }
 
#line 363 "EaseaLex.cpp"
		}
		break;
	case 20:
		{
#line 245 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 
#line 380 "EaseaLex.cpp"
		}
		break;
	case 21:
		{
#line 257 "EaseaLex.l"

  size_t size_of_genome=0;
  if (bVERBOSE) printf ("Inserting default genome size.\n");
  if( !genomeSizeValidity ){
    if (bVERBOSE) printf ("\tComputing default genome size.\n");  
    CListItem<CSymbol*> *pSym;
    pGENOME->pSymbolList->reset();
    while (pSym=pGENOME->pSymbolList->walkToNextItem()){
      //DEBUG_PRT_PRT("%s has size : %lu",pSym->Object->sName,pSym->Object->nSize);
      size_of_genome+=pSym->Object->nSize;
    }
    //DEBUG_PRT_PRT("Total genome size is %lu",size_of_genome); 
    genomeSize = size_of_genome;
    genomeSizeValidity=true;
  }
  else{
    size_of_genome = genomeSize;
  }
  fprintf(fpOutputFile,"%d",size_of_genome);
 
#line 406 "EaseaLex.cpp"
		}
		break;
	case 22:
		{
#line 278 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();

  fprintf(fpOutputFile,"\tmemcpy(GENOME_ACCESS(id,buffer),this,Individual::sizeOfGenome);");

  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
  
 
#line 427 "EaseaLex.cpp"
		}
		break;
#line 296 "EaseaLex.l"
  
#line 310 "EaseaLex.l"
      
#line 434 "EaseaLex.cpp"
	case 23:
		{
#line 317 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    //DEBUG_PRT_PRT("found new symbol %s",pSym->Object->sName);
    fprintf(fpOutputFile," ar & %s;\n",pSym->Object->sName);
  }
 
#line 447 "EaseaLex.cpp"
		}
		break;
	case 24:
		{
#line 327 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 462 "EaseaLex.cpp"
		}
		break;
	case 25:
		{
#line 336 "EaseaLex.l"

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
 
#line 485 "EaseaLex.cpp"
		}
		break;
	case 26:
		{
#line 353 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default clone method.\n");
  fprintf (fpOutputFile,"// Memberwise Cloning\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
      if (pSym->Object->ObjectType==oObject)
	fprintf(fpOutputFile,"    %s=genome.%s;\n",pSym->Object->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oPointer)
	fprintf(fpOutputFile,"    %s=(genome.%s ? new %s(*(genome.%s)) : NULL);\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray){
	fprintf(fpOutputFile,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	fprintf(fpOutputFile,"       %s[EASEA_Ndx]=genome.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
      }
  }
 
#line 508 "EaseaLex.cpp"
		}
		break;
	case 27:
		{
#line 371 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default copy constructor.\n");
  fprintf (fpOutputFile,"// Memberwise copy\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
      if (pSym->Object->ObjectType==oObject)
	fprintf(fpOutputFile,"    %s=genome.%s;\n",pSym->Object->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oPointer)
	fprintf(fpOutputFile,"    %s=(genome.%s ? new %s(*(genome.%s)) : NULL);\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray){
	fprintf(fpOutputFile,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	fprintf(fpOutputFile,"       %s[EASEA_Ndx]=genome.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
      }
      else if( pSym->Object->ObjectType==oArrayPointer ){ 
	// here we handle array of pointer (developped for Tree GP)
	fprintf(fpOutputFile,"    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/sizeof(char*));
	fprintf(fpOutputFile,"      if(genome.%s[EASEA_Ndx]) %s[EASEA_Ndx] = new %s(*(genome.%s[EASEA_Ndx]));\n",pSym->Object->sName,
		pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
	fprintf(fpOutputFile,"      else %s[EASEA_Ndx] = NULL;\n",pSym->Object->sName);
	
      }
      
  }
 
#line 540 "EaseaLex.cpp"
		}
		break;
	case 28:
		{
#line 398 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default copy constructor.\n");
  fprintf (fpOutputFile,"// Memberwise copy\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
      if (pSym->Object->ObjectType==oObject)
	fprintf(fpOutputFile,"    dest->%s=src->%s;\n",pSym->Object->sName,pSym->Object->sName);
      if (pSym->Object->ObjectType==oArray){
	fprintf(fpOutputFile,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	fprintf(fpOutputFile,"       dest->%s[EASEA_Ndx]=src->%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName);
      }
  }
 
#line 561 "EaseaLex.cpp"
		}
		break;
	case 29:
		{
#line 414 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default destructor.\n");
  fprintf (fpOutputFile,"// Destructing pointers\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"  if (%s) delete %s;\n  %s=NULL;\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
    }
    else if( pSym->Object->ObjectType==oArrayPointer ){ 
      // here we handle array of pointer (developped for Tree GP)
      fprintf(fpOutputFile,"    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/sizeof(char*));
      fprintf(fpOutputFile,"      if(%s[EASEA_Ndx]) delete %s[EASEA_Ndx];\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
    }
  }
 
#line 583 "EaseaLex.cpp"
		}
		break;
	case 30:
		{
#line 431 "EaseaLex.l"
       
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
    else if( pSym->Object->ObjectType==oArrayPointer ){ 
      // here we handle array of pointer (developped for Tree GP)
      fprintf(fpOutputFile,"    for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/sizeof(char*));
      fprintf(fpOutputFile,"      if(%s[EASEA_Ndx] != genome.%s[EASEA_Ndx]) return 0;\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName);
    }
    
  }
 
#line 611 "EaseaLex.cpp"
		}
		break;
	case 31:
		{
#line 453 "EaseaLex.l"

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
 
#line 633 "EaseaLex.cpp"
		}
		break;
	case 32:
		{
#line 469 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 648 "EaseaLex.cpp"
		}
		break;
	case 33:
		{
#line 478 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 660 "EaseaLex.cpp"
		}
		break;
	case 34:
		{
#line 486 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 672 "EaseaLex.cpp"
		}
		break;
	case 35:
		{
#line 493 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (!bDisplayFunction){
    if (bVERBOSE) printf ("*** Creating default display function. ***\n");
    fprintf (fpOutputFile,"// Default display function\n");
    pGENOME->pSymbolList->reset();
    while (pSym=pGENOME->pSymbolList->walkToNextItem()){
      if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
	if (pSym->Object->ObjectType==oObject){
	  if (bDisplayFunction) printf("//");
	}
	if (pSym->Object->ObjectType==oArray){
	  if (bDisplayFunction) printf("//");
	  if (bDisplayFunction) printf("//");
	  fprintf(fpOutputFile,"   for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize);
	  if (bDisplayFunction) printf("//");
	  if (bDisplayFunction) printf("//");
	  fprintf(fpOutputFile,"  os << \"\\n\";\n",pSym->Object->sName);
	}         
	if (pSym->Object->ObjectType==oPointer){
	  if (bDisplayFunction) printf("//");
	}
    }
  }                      
 
#line 703 "EaseaLex.cpp"
		}
		break;
	case 36:
		{
#line 518 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 716 "EaseaLex.cpp"
		}
		break;
	case 37:
		{
#line 525 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 730 "EaseaLex.cpp"
		}
		break;
	case 38:
		{
#line 534 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_INITIALISER;   
 
#line 741 "EaseaLex.cpp"
		}
		break;
	case 39:
		{
#line 540 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 753 "EaseaLex.cpp"
		}
		break;
	case 40:
		{
#line 547 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 765 "EaseaLex.cpp"
		}
		break;
	case 41:
		{
#line 553 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 777 "EaseaLex.cpp"
		}
		break;
	case 42:
		{
#line 559 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 789 "EaseaLex.cpp"
		}
		break;
	case 43:
		{
#line 565 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 801 "EaseaLex.cpp"
		}
		break;
	case 44:
		{
#line 571 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 814 "EaseaLex.cpp"
		}
		break;
	case 45:
		{
#line 578 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 827 "EaseaLex.cpp"
		}
		break;
	case 46:
		{
#line 586 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 838 "EaseaLex.cpp"
		}
		break;
	case 47:
		{
#line 591 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 852 "EaseaLex.cpp"
		}
		break;
	case 48:
		{
#line 600 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 866 "EaseaLex.cpp"
		}
		break;
	case 49:
		{
#line 609 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 880 "EaseaLex.cpp"
		}
		break;
	case 50:
		{
#line 619 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 893 "EaseaLex.cpp"
		}
		break;
	case 51:
		{
#line 627 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 902 "EaseaLex.cpp"
		}
		break;
	case 52:
		{
#line 631 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 911 "EaseaLex.cpp"
		}
		break;
	case 53:
		{
#line 635 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 920 "EaseaLex.cpp"
		}
		break;
	case 54:
		{
#line 639 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 929 "EaseaLex.cpp"
		}
		break;
	case 55:
		{
#line 643 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 939 "EaseaLex.cpp"
		}
		break;
	case 56:
		{
#line 648 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD){
    //DEBUG_PRT_PRT("Selector is \"%s\" | Goal is %s",sSELECTOR,(nMINIMISE?"Minimize":"Maximize"));
    char* selectorClass = selectorDetermination(nMINIMISE,sSELECTOR);
    if( !selectorClass ){
      fprintf(stderr,"Error %d : selection operator %s doesn't exist in CUDA/STD template\n",yylineno,sSELECTOR);
      return -1;
    }    
    //DEBUG_PRT_PRT("Created class is %s",selectorClass);
    fprintf(fpOutputFile,"%s",selectorClass);
  }
  else fprintf(fpOutputFile,"%s",sSELECTOR);

#line 958 "EaseaLex.cpp"
		}
		break;
	case 57:
		{
#line 661 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 965 "EaseaLex.cpp"
		}
		break;
	case 58:
		{
#line 662 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 972 "EaseaLex.cpp"
		}
		break;
	case 59:
		{
#line 663 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 979 "EaseaLex.cpp"
		}
		break;
	case 60:
		{
#line 664 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 986 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 665 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 993 "EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 666 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1000 "EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 667 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1007 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 668 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1014 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 669 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1021 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 670 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1028 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 671 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",nELITE);
  ////DEBUG_PRT_PRT("elitism is %d, elite size is %d",bELITISM, nELITE);
 
#line 1038 "EaseaLex.cpp"
		}
		break;
	case 68:
		{
#line 676 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD){
    //DEBUG_PRT_PRT("Parent reduction is \"%s\" | Goal is %s",sSELECTOR,(nMINIMISE?"Minimize":"Maximize"));
    char* selectorClass = selectorDetermination(nMINIMISE,sSELECTOR);
    if( !selectorClass ){
      fprintf(stderr,"Error %d : selection operator %s doesn't exist in CUDA/STD template\n",yylineno,sSELECTOR);
      return -1;
    }    
    //DEBUG_PRT_PRT("Created class is %s",selectorClass);
    fprintf(fpOutputFile,"%s",selectorClass);
  }
  else fprintf(fpOutputFile,"%s",sRED_PAR);
 
#line 1057 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 689 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD){
    //DEBUG_PRT_PRT("Offspring reduction is \"%s\" | Goal is %s",sSELECTOR,(nMINIMISE?"Minimize":"Maximize"));
    char* selectorClass = selectorDetermination(nMINIMISE,sSELECTOR);
    if( !selectorClass ){
      fprintf(stderr,"Error %d : selection operator %s doesn't exist in CUDA/STD template\n",yylineno,sSELECTOR);
      return -1;
    }    
    //DEBUG_PRT_PRT("Created class is %s",selectorClass);
    fprintf(fpOutputFile,"%s",selectorClass);
  }
  else fprintf(fpOutputFile,"%s",sRED_OFF);
 
#line 1076 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 702 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD){
    //DEBUG_PRT_PRT("Replacement selector is \"%s\" | Goal is %s",sRED_FINAL,(nMINIMISE?"Minimize":"Maximize"));
    char* selectorClass = selectorDetermination(nMINIMISE,sRED_FINAL);
    if( !selectorClass ){
      fprintf(stderr,"Error %d : replacement operator %s doesn't exist in CUDA/TPL template\n",yylineno,sRED_FINAL);
      return -1;
    }    
    //DEBUG_PRT_PRT("Created class is %s",selectorClass);
    fprintf(fpOutputFile,"%s",selectorClass);
  }
  else fprintf(fpOutputFile,"%s",sRED_FINAL);
 
#line 1095 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 715 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1105 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 719 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1112 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 720 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1119 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 721 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1126 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 722 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1133 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 723 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1140 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 724 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1147 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 725 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1154 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 726 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1161 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 727 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1168 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 728 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1175 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 729 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1182 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 730 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1189 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 732 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1196 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 733 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1203 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 734 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1210 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 735 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1217 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 736 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1224 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 738 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1238 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 746 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  if( TARGET==CUDA )
    strcat(sFileName,"Individual.cu");
  else if( TARGET==STD )
    strcat(sFileName,"Individual.cpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1255 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 757 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1269 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 765 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1283 "EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 774 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1297 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 783 "EaseaLex.l"

  char sFileName[1000];
  char sPathName[1000];
  char sFullFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");

  // get the path without fillename
  int fileNameLength = -1;
  for( int i=strlen(sRAW_PROJECT_NAME) ; i>=0 ; i-- )
    if( sRAW_PROJECT_NAME[i] == '/' ){
      fileNameLength = i;
      break;
    }
  if( fileNameLength != -1 ){
    // add "Makefile" at the end of path
    //char* cdn = get_current_dir_name();
    char cdn[4096];
    #ifdef WIN32 
    _getcwd(cdn,4096);
    #else
    getcwd(cdn,4096);
    #endif
    size_t cdnLength = strlen(cdn);
    strcpy(sFullFileName,cdn);
    strcat(sFullFileName,"/\0");
    strcat(sFullFileName,sFileName);
    
    strncpy(sPathName,sRAW_PROJECT_NAME,fileNameLength);
    strcpy(sPathName+fileNameLength,"/Makefile");
    
    //DEBUG_PRT_PRT("PathName is %s",sPathName);
    //DEBUG_PRT_PRT("FullFileName is %s",sFullFileName);
    
  
    // create a symbolic link from Makefile to EASEA.mak
#ifndef WIN32
    symlink(sFullFileName,sPathName);
#endif
  }
  else{
    //DEBUG_PRT_PRT("file name : %s",sFileName);
#ifndef WIN32
    if( symlink(sFileName,"Makefile") ) perror("Symlink creation error ");
#endif
  }
  if (bVERBOSE){
    printf("Creating %s...\n",sFileName);
    printf("Creating %s symbolic link...\n",sPathName);
  }
  fpOutputFile=fopen(sFileName,"w");
  if( !fpOutputFile ) {
    fprintf(stderr,"Error in %s creation\n",sFileName);
    exit(-1);
  }

#line 1360 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 840 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded with no warning.\n");
  printf ("Have a nice compile time.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 1377 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 852 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1384 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 858 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1396 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 864 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1409 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 871 "EaseaLex.l"

#line 1416 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 872 "EaseaLex.l"
lineCounter++;
#line 1423 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 874 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1435 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 880 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1448 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 888 "EaseaLex.l"

#line 1455 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 889 "EaseaLex.l"

  lineCounter++;
 
#line 1464 "EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 893 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1476 "EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 899 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1490 "EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 907 "EaseaLex.l"

#line 1497 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 908 "EaseaLex.l"

  lineCounter++;
 
#line 1506 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 912 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){");
  bFunction=1; bInitFunction=1;
      
  BEGIN COPY;
 
#line 1518 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 918 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1532 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 926 "EaseaLex.l"

#line 1539 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 931 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){");
  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 1550 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 937 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1564 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 945 "EaseaLex.l"

#line 1571 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 948 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 1587 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 959 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1603 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 969 "EaseaLex.l"

#line 1610 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 972 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    bBeginGeneration = 0;
    bBeginGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 1626 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 983 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 1640 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 992 "EaseaLex.l"

#line 1647 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 994 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1663 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1006 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1679 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1016 "EaseaLex.l"

#line 1686 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1020 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 1701 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1030 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1716 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1039 "EaseaLex.l"

#line 1723 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1042 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 1736 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1049 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1750 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1057 "EaseaLex.l"

#line 1757 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1061 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 1765 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1063 "EaseaLex.l"

#line 1772 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1069 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 1779 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1070 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 1786 "EaseaLex.cpp"
		}
		break;
	case 134:
	case 135:
		{
#line 1073 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 1797 "EaseaLex.cpp"
		}
		break;
	case 136:
	case 137:
		{
#line 1078 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 1806 "EaseaLex.cpp"
		}
		break;
	case 138:
	case 139:
		{
#line 1081 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 1815 "EaseaLex.cpp"
		}
		break;
	case 140:
	case 141:
		{
#line 1084 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 1832 "EaseaLex.cpp"
		}
		break;
	case 142:
	case 143:
		{
#line 1095 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 1846 "EaseaLex.cpp"
		}
		break;
	case 144:
	case 145:
		{
#line 1103 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 1855 "EaseaLex.cpp"
		}
		break;
	case 146:
	case 147:
		{
#line 1106 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 1864 "EaseaLex.cpp"
		}
		break;
	case 148:
	case 149:
		{
#line 1109 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 1873 "EaseaLex.cpp"
		}
		break;
	case 150:
	case 151:
		{
#line 1112 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 1882 "EaseaLex.cpp"
		}
		break;
	case 152:
	case 153:
		{
#line 1115 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 1891 "EaseaLex.cpp"
		}
		break;
	case 154:
	case 155:
		{
#line 1119 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 1903 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1125 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 1910 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1126 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1917 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1127 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1924 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1128 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 1934 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1133 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1941 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1134 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1948 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1135 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1955 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1136 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1962 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1137 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1969 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1138 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1976 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1139 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 1983 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1140 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 1990 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1141 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 1998 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1143 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2006 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1145 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2014 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1147 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2024 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1151 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2031 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1152 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2038 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1153 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2049 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1158 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2056 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1159 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"Individual");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2065 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1162 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2077 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1168 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2086 "EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 1171 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2098 "EaseaLex.cpp"
		}
		break;
	case 180:
		{
#line 1177 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2109 "EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 1182 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2125 "EaseaLex.cpp"
		}
		break;
	case 182:
		{
#line 1192 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2132 "EaseaLex.cpp"
		}
		break;
	case 183:
		{
#line 1195 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2141 "EaseaLex.cpp"
		}
		break;
	case 184:
		{
#line 1198 "EaseaLex.l"
BEGIN COPY;
#line 2148 "EaseaLex.cpp"
		}
		break;
	case 185:
		{
#line 1200 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2155 "EaseaLex.cpp"
		}
		break;
	case 186:
	case 187:
	case 188:
		{
#line 1203 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2168 "EaseaLex.cpp"
		}
		break;
	case 189:
		{
#line 1208 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2179 "EaseaLex.cpp"
		}
		break;
	case 190:
		{
#line 1213 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 2188 "EaseaLex.cpp"
		}
		break;
	case 191:
		{
#line 1222 "EaseaLex.l"
;
#line 2195 "EaseaLex.cpp"
		}
		break;
	case 192:
		{
#line 1223 "EaseaLex.l"
;
#line 2202 "EaseaLex.cpp"
		}
		break;
	case 193:
		{
#line 1224 "EaseaLex.l"
;
#line 2209 "EaseaLex.cpp"
		}
		break;
	case 194:
		{
#line 1225 "EaseaLex.l"
;
#line 2216 "EaseaLex.cpp"
		}
		break;
	case 195:
		{
#line 1228 "EaseaLex.l"
 /* do nothing */ 
#line 2223 "EaseaLex.cpp"
		}
		break;
	case 196:
		{
#line 1229 "EaseaLex.l"
 /*return '\n';*/ 
#line 2230 "EaseaLex.cpp"
		}
		break;
	case 197:
		{
#line 1230 "EaseaLex.l"
 /*return '\n';*/ 
#line 2237 "EaseaLex.cpp"
		}
		break;
	case 198:
		{
#line 1233 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("boolean");
  return BOOL;
#line 2246 "EaseaLex.cpp"
		}
		break;
	case 199:
		{
#line 1236 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 2256 "EaseaLex.cpp"
		}
		break;
	case 200:
		{
#line 1240 "EaseaLex.l"
return STATIC;
#line 2263 "EaseaLex.cpp"
		}
		break;
	case 201:
		{
#line 1241 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 2270 "EaseaLex.cpp"
		}
		break;
	case 202:
		{
#line 1242 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 2277 "EaseaLex.cpp"
		}
		break;
	case 203:
		{
#line 1243 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 2284 "EaseaLex.cpp"
		}
		break;
	case 204:
		{
#line 1244 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 2291 "EaseaLex.cpp"
		}
		break;
	case 205:
		{
#line 1245 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 2298 "EaseaLex.cpp"
		}
		break;
	case 206:
		{
#line 1247 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 2305 "EaseaLex.cpp"
		}
		break;
#line 1248 "EaseaLex.l"
  
#line 2310 "EaseaLex.cpp"
	case 207:
		{
#line 1249 "EaseaLex.l"
return GENOME; 
#line 2315 "EaseaLex.cpp"
		}
		break;
	case 208:
		{
#line 1251 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 2325 "EaseaLex.cpp"
		}
		break;
	case 209:
	case 210:
	case 211:
		{
#line 1258 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 2334 "EaseaLex.cpp"
		}
		break;
	case 212:
		{
#line 1259 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 2341 "EaseaLex.cpp"
		}
		break;
	case 213:
		{
#line 1262 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 2349 "EaseaLex.cpp"
		}
		break;
	case 214:
		{
#line 1264 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 2356 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1270 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 2368 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1276 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2381 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1283 "EaseaLex.l"

#line 2388 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1285 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 2399 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1296 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 2414 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1306 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 2425 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1312 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 2434 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1316 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 2449 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1329 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 2461 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1335 "EaseaLex.l"

#line 2468 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1336 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 2481 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1343 "EaseaLex.l"

#line 2488 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1344 "EaseaLex.l"
lineCounter++;
#line 2495 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1345 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 2508 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1352 "EaseaLex.l"

#line 2515 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1353 "EaseaLex.l"
lineCounter++;
#line 2522 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1355 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 2535 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1362 "EaseaLex.l"

#line 2542 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1363 "EaseaLex.l"
lineCounter++;
#line 2549 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1365 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 2562 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1372 "EaseaLex.l"

#line 2569 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1373 "EaseaLex.l"
lineCounter++;
#line 2576 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1379 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2583 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1380 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2590 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1381 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2597 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1382 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2604 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1383 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 2611 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1384 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2618 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1385 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2625 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1387 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 2634 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1390 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 2647 "EaseaLex.cpp"
		}
		break;
	case 246:
	case 247:
		{
#line 1399 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 2658 "EaseaLex.cpp"
		}
		break;
	case 248:
	case 249:
		{
#line 1404 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 2667 "EaseaLex.cpp"
		}
		break;
	case 250:
	case 251:
		{
#line 1407 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 2676 "EaseaLex.cpp"
		}
		break;
	case 252:
	case 253:
		{
#line 1410 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 2688 "EaseaLex.cpp"
		}
		break;
	case 254:
	case 255:
		{
#line 1416 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 2701 "EaseaLex.cpp"
		}
		break;
	case 256:
	case 257:
		{
#line 1423 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 2710 "EaseaLex.cpp"
		}
		break;
	case 258:
	case 259:
		{
#line 1426 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 2719 "EaseaLex.cpp"
		}
		break;
	case 260:
	case 261:
		{
#line 1429 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 2728 "EaseaLex.cpp"
		}
		break;
	case 262:
	case 263:
		{
#line 1432 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 2737 "EaseaLex.cpp"
		}
		break;
	case 264:
	case 265:
		{
#line 1435 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 2746 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1438 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 2755 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1441 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 2765 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1445 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 2773 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1447 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 2784 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1452 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 2795 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1457 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 2803 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1459 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 2811 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1461 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 2819 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1463 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 2827 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1465 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 2835 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1467 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2842 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1468 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2849 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1469 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2857 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1471 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2865 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1473 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2873 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1475 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2880 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1476 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 2892 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1482 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 2901 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1485 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 2911 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1489 "EaseaLex.l"
if (bWaitingForSemiColon){
    bWaitingForSemiColon=0;
    if (bFinishMINIMISE) {fprintf(fpOutputFile,");\n  if (MINIMISE) g.minimize() else g.maximize();\n"); bFinishMINIMISE=0;}
    if (bFinishMINIMIZE) {fprintf(fpOutputFile,");\n  if (MINIMIZE) g.minimize() else g.maximize();\n"); bFinishMINIMISE=0;}
    if ((bFinishNB_GEN)&&(OPERATING_SYSTEM==UNIX))
      {fprintf(fpOutputFile,"));\n  if ( (EZ_daFITNESS = (double *) realloc(EZ_daFITNESS, ((*EZ_NB_GEN) +1)* sizeof (double) )) == NULL){\n");
	fprintf(fpOutputFile,"    fprintf(stderr,\"Not enough memory... bailing out.\");\n    exit(1);");}
    else if (bFinishNB_GEN) {fprintf(fpOutputFile,"));"); bFinishNB_GEN=0;}
    else fprintf(fpOutputFile,"));");
  }
  else fprintf(fpOutputFile,";");
#line 2928 "EaseaLex.cpp"
		}
		break;
	case 286:
	case 287:
		{
#line 1501 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 2938 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1504 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2945 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1511 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2952 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1512 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2959 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1513 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 2966 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1514 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2973 "EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1516 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 2982 "EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1520 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 2995 "EaseaLex.cpp"
		}
		break;
	case 295:
		{
#line 1528 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3008 "EaseaLex.cpp"
		}
		break;
	case 296:
		{
#line 1537 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3021 "EaseaLex.cpp"
		}
		break;
	case 297:
		{
#line 1546 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3036 "EaseaLex.cpp"
		}
		break;
	case 298:
		{
#line 1556 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3043 "EaseaLex.cpp"
		}
		break;
	case 299:
		{
#line 1557 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3050 "EaseaLex.cpp"
		}
		break;
	case 300:
	case 301:
		{
#line 1560 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3061 "EaseaLex.cpp"
		}
		break;
	case 302:
	case 303:
		{
#line 1565 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3070 "EaseaLex.cpp"
		}
		break;
	case 304:
	case 305:
		{
#line 1568 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3079 "EaseaLex.cpp"
		}
		break;
	case 306:
	case 307:
		{
#line 1571 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3092 "EaseaLex.cpp"
		}
		break;
	case 308:
	case 309:
		{
#line 1578 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3105 "EaseaLex.cpp"
		}
		break;
	case 310:
	case 311:
		{
#line 1585 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3114 "EaseaLex.cpp"
		}
		break;
	case 312:
		{
#line 1588 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3121 "EaseaLex.cpp"
		}
		break;
	case 313:
		{
#line 1589 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3128 "EaseaLex.cpp"
		}
		break;
	case 314:
		{
#line 1590 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3135 "EaseaLex.cpp"
		}
		break;
	case 315:
		{
#line 1591 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3145 "EaseaLex.cpp"
		}
		break;
	case 316:
		{
#line 1596 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3152 "EaseaLex.cpp"
		}
		break;
	case 317:
		{
#line 1597 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3159 "EaseaLex.cpp"
		}
		break;
	case 318:
		{
#line 1598 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3166 "EaseaLex.cpp"
		}
		break;
	case 319:
		{
#line 1599 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3173 "EaseaLex.cpp"
		}
		break;
	case 320:
		{
#line 1600 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3181 "EaseaLex.cpp"
		}
		break;
	case 321:
		{
#line 1602 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3189 "EaseaLex.cpp"
		}
		break;
	case 322:
		{
#line 1604 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3197 "EaseaLex.cpp"
		}
		break;
	case 323:
		{
#line 1606 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3205 "EaseaLex.cpp"
		}
		break;
	case 324:
		{
#line 1608 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 3213 "EaseaLex.cpp"
		}
		break;
	case 325:
		{
#line 1610 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 3221 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1612 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 3229 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1614 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3236 "EaseaLex.cpp"
		}
		break;
	case 328:
		{
#line 1615 "EaseaLex.l"
if (bWithinEvaluator) {
    if( TARGET==CUDA || TARGET==STD) {
      if( bWithinCUDA_Evaluator ){
	fprintf(fpOutputFile,"return "); 
	//bWithinCUDA_Evaluator = 0;
      }
      else
	fprintf(fpOutputFile,"return fitness = "); 
      bCatchNextSemiColon=false;
    }
    //bWithinEvaluator=0;
  }
  else if ((bWithinMutator)) {
    fprintf(fpOutputFile,"return ");
    bCatchNextSemiColon=true;
  }
  else fprintf(fpOutputFile,"return"); 
#line 3259 "EaseaLex.cpp"
		}
		break;
	case 329:
		{
#line 1632 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 3270 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1637 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 3284 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1645 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3291 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 1651 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 3301 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 1655 "EaseaLex.l"

#line 3308 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 1658 "EaseaLex.l"
;
#line 3315 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 1659 "EaseaLex.l"
;
#line 3322 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 1660 "EaseaLex.l"
;
#line 3329 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 1661 "EaseaLex.l"
;
#line 3336 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1663 "EaseaLex.l"
 /* do nothing */ 
#line 3343 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 1664 "EaseaLex.l"
 /*return '\n';*/ 
#line 3350 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 1665 "EaseaLex.l"
 /*return '\n';*/ 
#line 3357 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 1667 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 3364 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 1668 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 3371 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 1669 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 3378 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 1670 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 3385 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 1671 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 3392 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 1672 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 3399 "EaseaLex.cpp"
		}
		break;
	case 347:
		{
#line 1673 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 3406 "EaseaLex.cpp"
		}
		break;
	case 348:
		{
#line 1674 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 3413 "EaseaLex.cpp"
		}
		break;
	case 349:
		{
#line 1675 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 3420 "EaseaLex.cpp"
		}
		break;
	case 350:
		{
#line 1677 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 3427 "EaseaLex.cpp"
		}
		break;
	case 351:
		{
#line 1678 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 3434 "EaseaLex.cpp"
		}
		break;
	case 352:
		{
#line 1679 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 3441 "EaseaLex.cpp"
		}
		break;
	case 353:
		{
#line 1680 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 3448 "EaseaLex.cpp"
		}
		break;
	case 354:
		{
#line 1681 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 3455 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 1683 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 3466 "EaseaLex.cpp"
		}
		break;
	case 356:
		{
#line 1688 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 3473 "EaseaLex.cpp"
		}
		break;
	case 357:
		{
#line 1690 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 3484 "EaseaLex.cpp"
		}
		break;
	case 358:
		{
#line 1695 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 3491 "EaseaLex.cpp"
		}
		break;
	case 359:
		{
#line 1698 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 3498 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 1699 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 3505 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 1700 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 3512 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 1701 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 3519 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 1702 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 3526 "EaseaLex.cpp"
		}
		break;
#line 1703 "EaseaLex.l"
 
#line 3531 "EaseaLex.cpp"
	case 364:
	case 365:
	case 366:
		{
#line 1707 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 3538 "EaseaLex.cpp"
		}
		break;
	case 367:
		{
#line 1708 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 3545 "EaseaLex.cpp"
		}
		break;
	case 368:
		{
#line 1711 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 3553 "EaseaLex.cpp"
		}
		break;
	case 369:
		{
#line 1714 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 3560 "EaseaLex.cpp"
		}
		break;
	case 370:
		{
#line 1716 "EaseaLex.l"

  lineCounter++;

#line 3569 "EaseaLex.cpp"
		}
		break;
	case 371:
		{
#line 1723 "EaseaLex.l"
return  (char)yytext[0];
#line 3576 "EaseaLex.cpp"
		}
		break;
	default:
		yyassert(0);
		break;
	}
	yyreturnflg = 0;
	return 0;
}

#ifndef YYNBORLANDWARN
#ifdef __BORLANDC__
#pragma warn .rch		// <warning: unreachable code> to the old state
#endif
#endif
#line 1725 "EaseaLex.l"


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
#elif defined WIN32 //WINDOWS_OS
  OPERATING_SYSTEM=WINDOWS;
//#else
//  OPERATING_SYSTEM=OTHER_SYSTEM;
#endif
  assert(pParser != NULL);
  assert(pSymTable != NULL);
  
  pSymbolTable = pSymTable;
  if (!yycreate(pParser)) return 0;    

  if (bVERBOSE) printf("\n                                                                   ");
  if (bVERBOSE) printf("\n                                   E A S E A                   (v1.0)");
  if (bVERBOSE) printf("\n                              ___________________     ");
  if (bVERBOSE) printf("\n                                                                    ");
  
  if (sRAW_PROJECT_NAME[0]==0){
    printf("\nInsert a .ez file name or a local project name: ");
    scanf("%s",sRAW_PROJECT_NAME);
  }                         
  if (bVERBOSE) printf("\n");
  
  if (TARGET==0) {
/*     printf("\nPlease select a target library (GALib STD or CUDA): "); */
/*     scanf("%s",sTemp); */
/*     else if (!mystricmp(sTemp,"cuda")) TARGET=CUDA; */
/*     else if (!mystricmp(sTemp,"std")) TARGET=STD; */
/*     else return 0; */
    TARGET = STD;
  }

  /////////////////////////////////////////////////////////  
  //strcpy(sTemp,"e:\\lutton\\easea\\debug");pour tester sous windows
  if ((sEZ_PATH==NULL)||(sEZ_PATH[0]==0)) {
    if (getenv("EZ_PATH")==NULL){
      //strcpy(sEZ_PATH,"./tpl/");	
      strcpy(sEZ_PATH,"./");	
    }else
      strcpy(sEZ_PATH,getenv("EZ_PATH"));
  }

  strcpy(sTPL_DIR,sEZ_PATH);
  strcat(sTPL_DIR,"tpl/");
  
 
  switch (OPERATING_SYSTEM) {
  case UNIX : if (sEZ_PATH[strlen(sEZ_PATH)-1] != '/') strcat (sEZ_PATH,"./"); break;
  case WINDOWS : if (sEZ_PATH[strlen(sEZ_PATH)-1] != '\\') strcat (sEZ_PATH,"\\"); break;
  case UNKNOWN_OS : fprintf(fpOutputFile,"UNKNOWN_OS"); break;
  }
  //strcpy(sTemp,sEZ_PATH);
  strcpy(sTemp,sTPL_DIR);

  if (TARGET==CUDA){
    if(TARGET_FLAVOR == CUDA_FLAVOR_SO )
      strcat(sTemp,"CUDA.tpl");
    else if(TARGET_FLAVOR == CUDA_FLAVOR_CMAES )
      strcat(sTemp,"CMAES_CUDA.tpl");
    else if(TARGET_FLAVOR == CUDA_FLAVOR_MEMETIC )
      strcat(sTemp,"CUDA_MEM.tpl");
    else 
      strcat(sTemp,"CUDA_MO.tpl");

    if (!(yyin = fpTemplateFile = fopen(sTemp, "r"))){
      fprintf(stderr,"\n*** Could not open %s.\n",sTemp);
      fprintf(stderr,"*** Please modify the EZ_PATH environment variable.\n");
      exit(1);
    } }

  if (TARGET==STD){
    if(TARGET_FLAVOR == STD_FLAVOR_SO)
      strcat(sTemp,"STD.tpl");
    else if (TARGET_FLAVOR == STD_FLAVOR_CMAES)
      strcat(sTemp,"CMAES.tpl");
    else if (TARGET_FLAVOR == STD_FLAVOR_MEMETIC )
      strcat(sTemp,"STD_MEM.tpl");
    else
      strcat(sTemp,"STD_MO.tpl");
    if (!(yyin = fpTemplateFile = fopen(sTemp, "r"))){
      fprintf(stderr,"\n*** Could not open %s.\n",sTemp);
      fprintf(stderr,"*** Please modify the EZ_PATH environment variable.\n");
      exit(1);
    } }
  
  
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

#line 3761 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		152,
		-153,
		0,
		144,
		-145,
		0,
		142,
		-143,
		0,
		134,
		-135,
		0,
		136,
		-137,
		0,
		138,
		-139,
		0,
		140,
		-141,
		0,
		146,
		-147,
		0,
		148,
		-149,
		0,
		150,
		-151,
		0,
		-184,
		0,
		-190,
		0,
		302,
		-303,
		0,
		304,
		-305,
		0,
		306,
		-307,
		0,
		300,
		-301,
		0,
		250,
		-251,
		0,
		252,
		-253,
		0,
		246,
		-247,
		0,
		258,
		-259,
		0,
		260,
		-261,
		0,
		262,
		-263,
		0,
		264,
		-265,
		0,
		248,
		-249,
		0,
		310,
		-311,
		0,
		256,
		-257,
		0,
		308,
		-309,
		0,
		254,
		-255,
		0
	};
	yymatch = match;

	yytransitionmax = 4198;
	static const yytransition_t YYNEARFAR YYBASED_CODE transition[] = {
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2011, 38 },
		{ 1369, 1343 },
		{ 1370, 1343 },
		{ 1691, 1794 },
		{ 1798, 38 },
		{ 2156, 2139 },
		{ 2156, 2139 },
		{ 2154, 2137 },
		{ 2154, 2137 },
		{ 2167, 2149 },
		{ 2167, 2149 },
		{ 63, 3 },
		{ 64, 3 },
		{ 2050, 41 },
		{ 2051, 41 },
		{ 61, 1 },
		{ 2266, 2266 },
		{ 2266, 2266 },
		{ 59, 1 },
		{ 2246, 2246 },
		{ 2246, 2246 },
		{ 2016, 2012 },
		{ 0, 1821 },
		{ 2011, 38 },
		{ 1369, 1343 },
		{ 1801, 38 },
		{ 2033, 2032 },
		{ 0, 2068 },
		{ 2156, 2139 },
		{ 1790, 1792 },
		{ 2154, 2137 },
		{ 152, 154 },
		{ 2167, 2149 },
		{ 157, 155 },
		{ 63, 3 },
		{ 2108, 2087 },
		{ 2050, 41 },
		{ 2108, 2087 },
		{ 2010, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 62, 3 },
		{ 1814, 38 },
		{ 2049, 41 },
		{ 1407, 1401 },
		{ 1371, 1343 },
		{ 2157, 2139 },
		{ 141, 139 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1800, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1808, 38 },
		{ 1806, 38 },
		{ 1819, 38 },
		{ 1807, 38 },
		{ 1819, 38 },
		{ 1810, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1809, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1409, 1403 },
		{ 1802, 38 },
		{ 1804, 38 },
		{ 1262, 1240 },
		{ 1819, 38 },
		{ 1261, 1239 },
		{ 1819, 38 },
		{ 1817, 38 },
		{ 1805, 38 },
		{ 1819, 38 },
		{ 1818, 38 },
		{ 1811, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1816, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1803, 38 },
		{ 1819, 38 },
		{ 1815, 38 },
		{ 1819, 38 },
		{ 1812, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 1819, 38 },
		{ 146, 4 },
		{ 147, 4 },
		{ 1387, 1373 },
		{ 1388, 1373 },
		{ 2169, 2151 },
		{ 2169, 2151 },
		{ 2183, 2166 },
		{ 2183, 2166 },
		{ 1263, 1241 },
		{ 1264, 1242 },
		{ 1265, 1243 },
		{ 1813, 37 },
		{ 1624, 35 },
		{ 2212, 2195 },
		{ 2212, 2195 },
		{ 1266, 1244 },
		{ 1267, 1245 },
		{ 1268, 1246 },
		{ 1269, 1247 },
		{ 1270, 1248 },
		{ 1273, 1251 },
		{ 1274, 1252 },
		{ 1275, 1253 },
		{ 146, 4 },
		{ 1276, 1254 },
		{ 1387, 1373 },
		{ 1277, 1255 },
		{ 2169, 2151 },
		{ 1279, 1258 },
		{ 2183, 2166 },
		{ 1396, 1386 },
		{ 1397, 1386 },
		{ 1317, 1300 },
		{ 1318, 1300 },
		{ 1813, 37 },
		{ 1624, 35 },
		{ 2212, 2195 },
		{ 76, 4 },
		{ 145, 4 },
		{ 75, 4 },
		{ 75, 4 },
		{ 75, 4 },
		{ 75, 4 },
		{ 75, 4 },
		{ 75, 4 },
		{ 75, 4 },
		{ 75, 4 },
		{ 75, 4 },
		{ 75, 4 },
		{ 1799, 37 },
		{ 1611, 35 },
		{ 1280, 1259 },
		{ 1281, 1260 },
		{ 1396, 1386 },
		{ 1389, 1373 },
		{ 1317, 1300 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 73, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 74, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 1398, 1386 },
		{ 72, 4 },
		{ 1319, 1300 },
		{ 1282, 1261 },
		{ 77, 4 },
		{ 1283, 1262 },
		{ 77, 4 },
		{ 65, 4 },
		{ 70, 4 },
		{ 68, 4 },
		{ 77, 4 },
		{ 69, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 67, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 71, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 66, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 77, 4 },
		{ 1201, 19 },
		{ 1354, 1338 },
		{ 1355, 1338 },
		{ 1284, 1263 },
		{ 1188, 19 },
		{ 2191, 2174 },
		{ 2191, 2174 },
		{ 2214, 2197 },
		{ 2214, 2197 },
		{ 2217, 2200 },
		{ 2217, 2200 },
		{ 2227, 2210 },
		{ 2227, 2210 },
		{ 2228, 2211 },
		{ 2228, 2211 },
		{ 2230, 2213 },
		{ 2230, 2213 },
		{ 1285, 1264 },
		{ 1286, 1265 },
		{ 1287, 1266 },
		{ 1288, 1267 },
		{ 1289, 1268 },
		{ 1290, 1270 },
		{ 1201, 19 },
		{ 1354, 1338 },
		{ 1189, 19 },
		{ 1202, 19 },
		{ 1293, 1273 },
		{ 2191, 2174 },
		{ 1294, 1274 },
		{ 2214, 2197 },
		{ 1295, 1275 },
		{ 2217, 2200 },
		{ 1296, 1276 },
		{ 2227, 2210 },
		{ 1297, 1277 },
		{ 2228, 2211 },
		{ 1299, 1279 },
		{ 2230, 2213 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1300, 1280 },
		{ 1301, 1281 },
		{ 1302, 1282 },
		{ 1303, 1283 },
		{ 1356, 1338 },
		{ 2192, 2174 },
		{ 1304, 1284 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1205, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1194, 19 },
		{ 1192, 19 },
		{ 1207, 19 },
		{ 1193, 19 },
		{ 1207, 19 },
		{ 1196, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1195, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1305, 1285 },
		{ 1190, 19 },
		{ 1203, 19 },
		{ 1306, 1286 },
		{ 1197, 19 },
		{ 1307, 1287 },
		{ 1207, 19 },
		{ 1208, 19 },
		{ 1191, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1198, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1206, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1209, 19 },
		{ 1207, 19 },
		{ 1204, 19 },
		{ 1207, 19 },
		{ 1199, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1207, 19 },
		{ 1785, 36 },
		{ 1357, 1339 },
		{ 1358, 1339 },
		{ 1308, 1288 },
		{ 1610, 36 },
		{ 2232, 2215 },
		{ 2232, 2215 },
		{ 2233, 2216 },
		{ 2233, 2216 },
		{ 1309, 1290 },
		{ 1312, 1293 },
		{ 2135, 2114 },
		{ 2135, 2114 },
		{ 2138, 2117 },
		{ 2138, 2117 },
		{ 1313, 1294 },
		{ 1314, 1295 },
		{ 1315, 1297 },
		{ 1316, 1299 },
		{ 1213, 1191 },
		{ 1320, 1301 },
		{ 1321, 1302 },
		{ 1322, 1303 },
		{ 1785, 36 },
		{ 1357, 1339 },
		{ 1615, 36 },
		{ 1326, 1305 },
		{ 1327, 1306 },
		{ 2232, 2215 },
		{ 1323, 1303 },
		{ 2233, 2216 },
		{ 1328, 1307 },
		{ 1360, 1340 },
		{ 1361, 1340 },
		{ 2135, 2114 },
		{ 1329, 1308 },
		{ 2138, 2117 },
		{ 1330, 1309 },
		{ 1784, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1333, 1313 },
		{ 1625, 36 },
		{ 1334, 1314 },
		{ 1335, 1315 },
		{ 1359, 1339 },
		{ 1336, 1316 },
		{ 1360, 1340 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1612, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1620, 36 },
		{ 1618, 36 },
		{ 1628, 36 },
		{ 1619, 36 },
		{ 1628, 36 },
		{ 1622, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1621, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1338, 1320 },
		{ 1616, 36 },
		{ 1362, 1340 },
		{ 1339, 1321 },
		{ 1628, 36 },
		{ 1340, 1322 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1617, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1613, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1614, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1627, 36 },
		{ 1628, 36 },
		{ 1626, 36 },
		{ 1628, 36 },
		{ 1623, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 1628, 36 },
		{ 2436, 42 },
		{ 2437, 42 },
		{ 1363, 1341 },
		{ 1364, 1341 },
		{ 59, 42 },
		{ 1324, 1304 },
		{ 2244, 2224 },
		{ 2244, 2224 },
		{ 1341, 1323 },
		{ 1342, 1324 },
		{ 1343, 1325 },
		{ 1344, 1326 },
		{ 1325, 1304 },
		{ 1345, 1327 },
		{ 1346, 1328 },
		{ 1347, 1329 },
		{ 1348, 1330 },
		{ 1350, 1333 },
		{ 1351, 1334 },
		{ 1352, 1335 },
		{ 1353, 1336 },
		{ 1216, 1192 },
		{ 1217, 1193 },
		{ 2436, 42 },
		{ 1221, 1195 },
		{ 1363, 1341 },
		{ 1222, 1196 },
		{ 1223, 1197 },
		{ 1372, 1344 },
		{ 2244, 2224 },
		{ 1366, 1342 },
		{ 1367, 1342 },
		{ 1373, 1345 },
		{ 1374, 1346 },
		{ 1376, 1350 },
		{ 1377, 1351 },
		{ 1378, 1352 },
		{ 2065, 42 },
		{ 2435, 42 },
		{ 2064, 42 },
		{ 2064, 42 },
		{ 2064, 42 },
		{ 2064, 42 },
		{ 2064, 42 },
		{ 2064, 42 },
		{ 2064, 42 },
		{ 2064, 42 },
		{ 2064, 42 },
		{ 2064, 42 },
		{ 1379, 1353 },
		{ 1386, 1372 },
		{ 1224, 1198 },
		{ 1390, 1374 },
		{ 1366, 1342 },
		{ 1365, 1341 },
		{ 0, 2224 },
		{ 2066, 42 },
		{ 2063, 42 },
		{ 2058, 42 },
		{ 2066, 42 },
		{ 2055, 42 },
		{ 2062, 42 },
		{ 2060, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2057, 42 },
		{ 2052, 42 },
		{ 2059, 42 },
		{ 2054, 42 },
		{ 2066, 42 },
		{ 2061, 42 },
		{ 2056, 42 },
		{ 2053, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 1368, 1342 },
		{ 2067, 42 },
		{ 1392, 1376 },
		{ 1393, 1377 },
		{ 2066, 42 },
		{ 1394, 1378 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 2066, 42 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1395, 1379 },
		{ 1226, 1199 },
		{ 1401, 1392 },
		{ 1402, 1393 },
		{ 1225, 1199 },
		{ 1403, 1394 },
		{ 1404, 1395 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1229, 1204 },
		{ 1408, 1402 },
		{ 1230, 1205 },
		{ 1410, 1404 },
		{ 1214, 1420 },
		{ 1413, 1408 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 1214, 1420 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 1414, 1410 },
		{ 1416, 1413 },
		{ 1417, 1414 },
		{ 1418, 1416 },
		{ 1419, 1417 },
		{ 1214, 1418 },
		{ 1231, 1206 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 1232, 1208 },
		{ 1233, 1209 },
		{ 1236, 1213 },
		{ 1237, 1216 },
		{ 2066, 2240 },
		{ 1238, 1217 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 2066, 2240 },
		{ 1220, 1194 },
		{ 1215, 1419 },
		{ 0, 1419 },
		{ 1239, 1218 },
		{ 1240, 1219 },
		{ 1241, 1220 },
		{ 1242, 1221 },
		{ 1243, 1222 },
		{ 1219, 1194 },
		{ 1244, 1223 },
		{ 1246, 1224 },
		{ 1247, 1225 },
		{ 1248, 1226 },
		{ 1245, 1223 },
		{ 1251, 1229 },
		{ 1252, 1230 },
		{ 1253, 1231 },
		{ 1254, 1232 },
		{ 1255, 1233 },
		{ 1258, 1236 },
		{ 1218, 1194 },
		{ 1259, 1237 },
		{ 1260, 1238 },
		{ 100, 86 },
		{ 1215, 1419 },
		{ 101, 87 },
		{ 103, 89 },
		{ 104, 90 },
		{ 110, 95 },
		{ 111, 96 },
		{ 112, 98 },
		{ 113, 99 },
		{ 114, 100 },
		{ 115, 101 },
		{ 117, 103 },
		{ 118, 104 },
		{ 123, 110 },
		{ 124, 111 },
		{ 125, 112 },
		{ 126, 113 },
		{ 127, 115 },
		{ 128, 117 },
		{ 129, 118 },
		{ 130, 123 },
		{ 131, 124 },
		{ 132, 125 },
		{ 133, 127 },
		{ 2364, 2364 },
		{ 134, 128 },
		{ 135, 129 },
		{ 136, 130 },
		{ 137, 133 },
		{ 138, 134 },
		{ 0, 1419 },
		{ 139, 135 },
		{ 140, 138 },
		{ 2070, 2052 },
		{ 2071, 2053 },
		{ 2074, 2054 },
		{ 2075, 2055 },
		{ 2077, 2056 },
		{ 2072, 2054 },
		{ 2079, 2057 },
		{ 2080, 2058 },
		{ 2073, 2054 },
		{ 2081, 2059 },
		{ 2082, 2060 },
		{ 2083, 2061 },
		{ 2084, 2062 },
		{ 2076, 2055 },
		{ 2364, 2364 },
		{ 2085, 2063 },
		{ 2066, 2066 },
		{ 2091, 2070 },
		{ 2092, 2071 },
		{ 2093, 2072 },
		{ 2078, 2056 },
		{ 2094, 2073 },
		{ 2095, 2074 },
		{ 2096, 2075 },
		{ 2097, 2076 },
		{ 2098, 2077 },
		{ 2099, 2078 },
		{ 2100, 2079 },
		{ 2101, 2080 },
		{ 2102, 2081 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2103, 2082 },
		{ 2104, 2083 },
		{ 2105, 2084 },
		{ 2106, 2085 },
		{ 2113, 2091 },
		{ 2114, 2092 },
		{ 1420, 1419 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 59, 7 },
		{ 2115, 2093 },
		{ 2116, 2094 },
		{ 2117, 2095 },
		{ 2364, 2364 },
		{ 2118, 2096 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2119, 2097 },
		{ 2120, 2098 },
		{ 2121, 2099 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 2122, 2100 },
		{ 2123, 2101 },
		{ 2124, 2102 },
		{ 2125, 2103 },
		{ 2126, 2104 },
		{ 2127, 2105 },
		{ 2128, 2106 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 2134, 2113 },
		{ 81, 65 },
		{ 2136, 2115 },
		{ 2137, 2116 },
		{ 1073, 7 },
		{ 142, 140 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 1073, 7 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 2139, 2118 },
		{ 2141, 2119 },
		{ 2142, 2120 },
		{ 2143, 2121 },
		{ 2140, 2118 },
		{ 2144, 2122 },
		{ 2145, 2123 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 2146, 2124 },
		{ 2147, 2125 },
		{ 2148, 2126 },
		{ 2149, 2127 },
		{ 0, 1278 },
		{ 2150, 2128 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1278 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 2151, 2134 },
		{ 2153, 2136 },
		{ 143, 142 },
		{ 144, 143 },
		{ 2158, 2140 },
		{ 2159, 2141 },
		{ 2160, 2142 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 2161, 2143 },
		{ 2162, 2144 },
		{ 2163, 2145 },
		{ 2164, 2146 },
		{ 0, 1690 },
		{ 2165, 2147 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1690 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 2166, 2148 },
		{ 2168, 2150 },
		{ 82, 66 },
		{ 2171, 2153 },
		{ 2174, 2158 },
		{ 2175, 2159 },
		{ 2178, 2161 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 2179, 2162 },
		{ 2177, 2160 },
		{ 2180, 2163 },
		{ 2181, 2164 },
		{ 0, 1884 },
		{ 2176, 2160 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 0, 1884 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 2182, 2165 },
		{ 83, 67 },
		{ 2185, 2168 },
		{ 2188, 2171 },
		{ 2193, 2175 },
		{ 2194, 2176 },
		{ 2195, 2177 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 2196, 2178 },
		{ 2197, 2179 },
		{ 2198, 2180 },
		{ 2199, 2181 },
		{ 77, 144 },
		{ 2200, 2182 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 77, 144 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 2204, 2185 },
		{ 2207, 2188 },
		{ 2210, 2193 },
		{ 2211, 2194 },
		{ 84, 68 },
		{ 2213, 2196 },
		{ 85, 69 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 2215, 2198 },
		{ 2216, 2199 },
		{ 86, 70 },
		{ 2221, 2204 },
		{ 1073, 1073 },
		{ 2224, 2207 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 1073, 1073 },
		{ 87, 71 },
		{ 89, 73 },
		{ 90, 74 },
		{ 95, 81 },
		{ 0, 2221 },
		{ 0, 2221 },
		{ 96, 82 },
		{ 97, 83 },
		{ 98, 84 },
		{ 99, 85 },
		{ 0, 2513 },
		{ 169, 160 },
		{ 179, 160 },
		{ 171, 160 },
		{ 121, 108 },
		{ 166, 160 },
		{ 121, 108 },
		{ 170, 160 },
		{ 106, 92 },
		{ 168, 160 },
		{ 106, 92 },
		{ 2300, 2283 },
		{ 2291, 2274 },
		{ 177, 160 },
		{ 176, 160 },
		{ 167, 160 },
		{ 175, 160 },
		{ 0, 2221 },
		{ 172, 160 },
		{ 174, 160 },
		{ 165, 160 },
		{ 2512, 47 },
		{ 1124, 1123 },
		{ 173, 160 },
		{ 178, 160 },
		{ 1079, 1076 },
		{ 1566, 1565 },
		{ 1076, 1076 },
		{ 1076, 1076 },
		{ 1076, 1076 },
		{ 1076, 1076 },
		{ 1076, 1076 },
		{ 1076, 1076 },
		{ 1076, 1076 },
		{ 1076, 1076 },
		{ 1076, 1076 },
		{ 1076, 1076 },
		{ 1083, 1080 },
		{ 2129, 2107 },
		{ 1083, 1080 },
		{ 2129, 2107 },
		{ 2131, 2110 },
		{ 119, 105 },
		{ 2131, 2110 },
		{ 119, 105 },
		{ 1990, 1987 },
		{ 2389, 2379 },
		{ 91, 75 },
		{ 1080, 1076 },
		{ 75, 75 },
		{ 75, 75 },
		{ 75, 75 },
		{ 75, 75 },
		{ 75, 75 },
		{ 75, 75 },
		{ 75, 75 },
		{ 75, 75 },
		{ 75, 75 },
		{ 75, 75 },
		{ 2399, 2390 },
		{ 418, 378 },
		{ 423, 378 },
		{ 420, 378 },
		{ 419, 378 },
		{ 422, 378 },
		{ 417, 378 },
		{ 1086, 1082 },
		{ 416, 378 },
		{ 1086, 1082 },
		{ 1184, 1183 },
		{ 92, 75 },
		{ 421, 378 },
		{ 1088, 1085 },
		{ 424, 378 },
		{ 1088, 1085 },
		{ 1448, 1447 },
		{ 1184, 1183 },
		{ 1608, 1607 },
		{ 1842, 1818 },
		{ 415, 378 },
		{ 1080, 1076 },
		{ 2086, 2064 },
		{ 1448, 1447 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 1121, 1120 },
		{ 2240, 2221 },
		{ 1474, 1473 },
		{ 2275, 2255 },
		{ 1519, 1518 },
		{ 1563, 1562 },
		{ 1841, 1818 },
		{ 2657, 2656 },
		{ 1857, 1836 },
		{ 92, 75 },
		{ 1886, 1868 },
		{ 2087, 2064 },
		{ 1079, 1079 },
		{ 1079, 1079 },
		{ 1079, 1079 },
		{ 1079, 1079 },
		{ 1079, 1079 },
		{ 1079, 1079 },
		{ 1079, 1079 },
		{ 1079, 1079 },
		{ 1079, 1079 },
		{ 1079, 1079 },
		{ 1900, 1883 },
		{ 1081, 1081 },
		{ 1081, 1081 },
		{ 1081, 1081 },
		{ 1081, 1081 },
		{ 1081, 1081 },
		{ 1081, 1081 },
		{ 1081, 1081 },
		{ 1081, 1081 },
		{ 1081, 1081 },
		{ 1081, 1081 },
		{ 1082, 1079 },
		{ 2183, 2183 },
		{ 2183, 2183 },
		{ 2230, 2230 },
		{ 2230, 2230 },
		{ 2459, 2458 },
		{ 1666, 1647 },
		{ 78, 47 },
		{ 2499, 2498 },
		{ 1641, 1617 },
		{ 2087, 2064 },
		{ 1085, 1081 },
		{ 93, 93 },
		{ 93, 93 },
		{ 93, 93 },
		{ 93, 93 },
		{ 93, 93 },
		{ 93, 93 },
		{ 93, 93 },
		{ 93, 93 },
		{ 93, 93 },
		{ 93, 93 },
		{ 1640, 1617 },
		{ 2504, 2503 },
		{ 2183, 2183 },
		{ 2276, 2256 },
		{ 2230, 2230 },
		{ 1688, 1669 },
		{ 1579, 1578 },
		{ 2239, 2220 },
		{ 2030, 2029 },
		{ 2545, 2544 },
		{ 1082, 1079 },
		{ 108, 93 },
		{ 2035, 2034 },
		{ 1522, 1521 },
		{ 2597, 2596 },
		{ 1495, 1494 },
		{ 1827, 1805 },
		{ 2638, 2637 },
		{ 1137, 1136 },
		{ 2660, 2659 },
		{ 2668, 2667 },
		{ 1085, 1081 },
		{ 1675, 1656 },
		{ 2086, 2086 },
		{ 2086, 2086 },
		{ 2086, 2086 },
		{ 2086, 2086 },
		{ 2086, 2086 },
		{ 2086, 2086 },
		{ 2086, 2086 },
		{ 2086, 2086 },
		{ 2086, 2086 },
		{ 2086, 2086 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 108, 93 },
		{ 2107, 2086 },
		{ 2562, 2562 },
		{ 2562, 2562 },
		{ 1524, 1523 },
		{ 1075, 9 },
		{ 2609, 2609 },
		{ 2609, 2609 },
		{ 2044, 2043 },
		{ 59, 9 },
		{ 2282, 2264 },
		{ 2110, 2088 },
		{ 2209, 2190 },
		{ 91, 91 },
		{ 91, 91 },
		{ 91, 91 },
		{ 91, 91 },
		{ 91, 91 },
		{ 91, 91 },
		{ 91, 91 },
		{ 91, 91 },
		{ 91, 91 },
		{ 91, 91 },
		{ 2295, 2278 },
		{ 1695, 1676 },
		{ 2562, 2562 },
		{ 2303, 2286 },
		{ 1546, 1545 },
		{ 1075, 9 },
		{ 2609, 2609 },
		{ 1722, 1706 },
		{ 2222, 2222 },
		{ 2222, 2222 },
		{ 2107, 2086 },
		{ 105, 91 },
		{ 2154, 2154 },
		{ 2154, 2154 },
		{ 2326, 2309 },
		{ 2202, 2183 },
		{ 2201, 2183 },
		{ 2251, 2230 },
		{ 2250, 2230 },
		{ 1077, 9 },
		{ 2110, 2088 },
		{ 1076, 9 },
		{ 1076, 9 },
		{ 1076, 9 },
		{ 1076, 9 },
		{ 1076, 9 },
		{ 1076, 9 },
		{ 1076, 9 },
		{ 1076, 9 },
		{ 1076, 9 },
		{ 1076, 9 },
		{ 2222, 2222 },
		{ 2233, 2233 },
		{ 2233, 2233 },
		{ 2335, 2319 },
		{ 2154, 2154 },
		{ 2138, 2138 },
		{ 2138, 2138 },
		{ 2244, 2244 },
		{ 2244, 2244 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 2337, 2321 },
		{ 105, 91 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2356, 2356 },
		{ 2356, 2356 },
		{ 2495, 2495 },
		{ 2495, 2495 },
		{ 2167, 2167 },
		{ 2167, 2167 },
		{ 2342, 2327 },
		{ 2217, 2217 },
		{ 2217, 2217 },
		{ 2233, 2233 },
		{ 2523, 2523 },
		{ 2523, 2523 },
		{ 2173, 2155 },
		{ 2138, 2138 },
		{ 2347, 2332 },
		{ 2244, 2244 },
		{ 2355, 2341 },
		{ 2257, 2257 },
		{ 2232, 2232 },
		{ 2232, 2232 },
		{ 2357, 2343 },
		{ 2292, 2292 },
		{ 1724, 1709 },
		{ 2356, 2356 },
		{ 1726, 1711 },
		{ 2495, 2495 },
		{ 2376, 2362 },
		{ 2167, 2167 },
		{ 2037, 2037 },
		{ 2037, 2037 },
		{ 2217, 2217 },
		{ 2214, 2214 },
		{ 2214, 2214 },
		{ 2523, 2523 },
		{ 2359, 2359 },
		{ 2359, 2359 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 2297, 2297 },
		{ 2297, 2297 },
		{ 2563, 2562 },
		{ 2232, 2232 },
		{ 2532, 2532 },
		{ 2532, 2532 },
		{ 2610, 2609 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2576, 2576 },
		{ 2576, 2576 },
		{ 1776, 1774 },
		{ 2379, 2367 },
		{ 2037, 2037 },
		{ 2169, 2169 },
		{ 2169, 2169 },
		{ 2214, 2214 },
		{ 2385, 2374 },
		{ 2241, 2222 },
		{ 2359, 2359 },
		{ 1558, 1557 },
		{ 2228, 2228 },
		{ 2390, 2380 },
		{ 2297, 2297 },
		{ 2617, 2617 },
		{ 2617, 2617 },
		{ 2242, 2222 },
		{ 2532, 2532 },
		{ 1105, 1104 },
		{ 2402, 2393 },
		{ 2329, 2329 },
		{ 2415, 2409 },
		{ 2576, 2576 },
		{ 2417, 2411 },
		{ 2172, 2154 },
		{ 2423, 2418 },
		{ 2429, 2428 },
		{ 2169, 2169 },
		{ 2223, 2206 },
		{ 1140, 1139 },
		{ 2524, 2523 },
		{ 2462, 2461 },
		{ 2236, 2217 },
		{ 2471, 2470 },
		{ 2484, 2483 },
		{ 2225, 2208 },
		{ 1567, 1566 },
		{ 2617, 2617 },
		{ 1179, 1178 },
		{ 2507, 2506 },
		{ 1582, 1581 },
		{ 1898, 1881 },
		{ 1899, 1882 },
		{ 1496, 1495 },
		{ 2254, 2233 },
		{ 1498, 1497 },
		{ 2535, 2534 },
		{ 2189, 2172 },
		{ 2155, 2138 },
		{ 2234, 2217 },
		{ 2265, 2244 },
		{ 2561, 2560 },
		{ 2277, 2257 },
		{ 2235, 2217 },
		{ 1917, 1904 },
		{ 1930, 1915 },
		{ 2309, 2292 },
		{ 2591, 2590 },
		{ 2370, 2356 },
		{ 2496, 2495 },
		{ 1931, 1916 },
		{ 2184, 2167 },
		{ 2524, 2523 },
		{ 2600, 2599 },
		{ 2608, 2607 },
		{ 1125, 1124 },
		{ 1991, 1988 },
		{ 2632, 2631 },
		{ 2006, 2005 },
		{ 2641, 2640 },
		{ 2651, 2650 },
		{ 1443, 1442 },
		{ 1669, 1650 },
		{ 2253, 2232 },
		{ 2662, 2661 },
		{ 2032, 2031 },
		{ 2671, 2670 },
		{ 1648, 1623 },
		{ 1837, 1812 },
		{ 2369, 2355 },
		{ 1647, 1623 },
		{ 1836, 1812 },
		{ 1737, 1724 },
		{ 2038, 2037 },
		{ 2135, 2135 },
		{ 2135, 2135 },
		{ 2231, 2214 },
		{ 1756, 1746 },
		{ 2373, 2359 },
		{ 1764, 1756 },
		{ 2248, 2228 },
		{ 1406, 1400 },
		{ 2314, 2297 },
		{ 2226, 2209 },
		{ 1540, 1539 },
		{ 1541, 1540 },
		{ 2533, 2532 },
		{ 1156, 1155 },
		{ 2396, 2386 },
		{ 2344, 2329 },
		{ 1550, 1549 },
		{ 2577, 2576 },
		{ 1158, 1157 },
		{ 1436, 1435 },
		{ 1858, 1837 },
		{ 1877, 1856 },
		{ 2186, 2169 },
		{ 2135, 2135 },
		{ 1879, 1858 },
		{ 2433, 2432 },
		{ 1882, 1861 },
		{ 2454, 2453 },
		{ 1437, 1436 },
		{ 2245, 2225 },
		{ 2466, 2465 },
		{ 1172, 1171 },
		{ 2618, 2617 },
		{ 1575, 1574 },
		{ 1173, 1172 },
		{ 1464, 1463 },
		{ 2255, 2234 },
		{ 2259, 2238 },
		{ 1598, 1597 },
		{ 1599, 1598 },
		{ 1604, 1603 },
		{ 2518, 2516 },
		{ 1465, 1464 },
		{ 1471, 1470 },
		{ 1472, 1471 },
		{ 2539, 2538 },
		{ 1133, 1132 },
		{ 1490, 1489 },
		{ 1667, 1648 },
		{ 1491, 1490 },
		{ 2305, 2288 },
		{ 1674, 1655 },
		{ 2312, 2295 },
		{ 1400, 1391 },
		{ 2047, 2046 },
		{ 1686, 1667 },
		{ 1109, 1108 },
		{ 1094, 1093 },
		{ 1514, 1513 },
		{ 2650, 2649 },
		{ 1515, 1514 },
		{ 2348, 2333 },
		{ 2350, 2336 },
		{ 2351, 2337 },
		{ 1148, 1147 },
		{ 1155, 1154 },
		{ 2416, 2410 },
		{ 1476, 1475 },
		{ 2296, 2279 },
		{ 2424, 2422 },
		{ 2427, 2425 },
		{ 2034, 2033 },
		{ 2302, 2285 },
		{ 1848, 1827 },
		{ 1678, 1659 },
		{ 2456, 2455 },
		{ 2040, 2039 },
		{ 2311, 2294 },
		{ 1096, 1095 },
		{ 2046, 2045 },
		{ 2482, 2481 },
		{ 2315, 2298 },
		{ 2493, 2492 },
		{ 1869, 1848 },
		{ 1526, 1525 },
		{ 2336, 2320 },
		{ 2237, 2218 },
		{ 2338, 2322 },
		{ 1450, 1449 },
		{ 1705, 1688 },
		{ 1123, 1122 },
		{ 2522, 2520 },
		{ 1186, 1185 },
		{ 1469, 1468 },
		{ 1552, 1551 },
		{ 1500, 1499 },
		{ 2252, 2231 },
		{ 2360, 2346 },
		{ 2363, 2349 },
		{ 2152, 2135 },
		{ 1560, 1559 },
		{ 2589, 2588 },
		{ 1659, 1640 },
		{ 1771, 1766 },
		{ 1984, 1977 },
		{ 1987, 1982 },
		{ 1103, 1102 },
		{ 2383, 2372 },
		{ 2630, 2629 },
		{ 2273, 2253 },
		{ 1565, 1564 },
		{ 2003, 2000 },
		{ 2392, 2382 },
		{ 1117, 1116 },
		{ 1670, 1651 },
		{ 2287, 2270 },
		{ 2403, 2394 },
		{ 2404, 2395 },
		{ 1150, 1149 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 1709, 1694 },
		{ 2264, 2242 },
		{ 2409, 2401 },
		{ 2322, 2305 },
		{ 2279, 2259 },
		{ 2534, 2533 },
		{ 2331, 2314 },
		{ 2384, 2373 },
		{ 1605, 1604 },
		{ 2268, 2248 },
		{ 2358, 2344 },
		{ 1868, 1847 },
		{ 2293, 2276 },
		{ 1545, 1544 },
		{ 2343, 2328 },
		{ 1637, 1614 },
		{ 1773, 1770 },
		{ 2327, 2310 },
		{ 2431, 2430 },
		{ 1291, 1271 },
		{ 1997, 1994 },
		{ 2413, 2413 },
		{ 2000, 1998 },
		{ 1779, 1778 },
		{ 1234, 1210 },
		{ 1636, 1614 },
		{ 1473, 1472 },
		{ 2464, 2463 },
		{ 1107, 1106 },
		{ 1671, 1652 },
		{ 1673, 1654 },
		{ 1843, 1820 },
		{ 2486, 2485 },
		{ 2492, 2491 },
		{ 2349, 2335 },
		{ 1847, 1826 },
		{ 1116, 1115 },
		{ 1525, 1524 },
		{ 1484, 1483 },
		{ 2509, 2508 },
		{ 1862, 1841 },
		{ 2267, 2247 },
		{ 1864, 1843 },
		{ 2272, 2252 },
		{ 1866, 1845 },
		{ 1584, 1583 },
		{ 2375, 2361 },
		{ 1592, 1591 },
		{ 1534, 1533 },
		{ 2537, 2536 },
		{ 1704, 1687 },
		{ 2544, 2543 },
		{ 1166, 1165 },
		{ 1458, 1457 },
		{ 2290, 2273 },
		{ 1897, 1880 },
		{ 1544, 1543 },
		{ 1719, 1703 },
		{ 2593, 2592 },
		{ 2395, 2385 },
		{ 1256, 1234 },
		{ 2602, 2601 },
		{ 2397, 2387 },
		{ 1142, 1141 },
		{ 1912, 1896 },
		{ 1430, 1429 },
		{ 1499, 1498 },
		{ 2634, 2633 },
		{ 1470, 1469 },
		{ 2410, 2402 },
		{ 2643, 2642 },
		{ 1962, 1944 },
		{ 1963, 1945 },
		{ 1652, 1629 },
		{ 1508, 1507 },
		{ 2422, 2417 },
		{ 2664, 2663 },
		{ 2320, 2303 },
		{ 1989, 1986 },
		{ 2673, 2672 },
		{ 2227, 2227 },
		{ 2227, 2227 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2584, 2584 },
		{ 2584, 2584 },
		{ 2477, 2477 },
		{ 2477, 2477 },
		{ 2625, 2625 },
		{ 2625, 2625 },
		{ 2465, 2464 },
		{ 1668, 1649 },
		{ 1942, 1928 },
		{ 1533, 1532 },
		{ 2418, 2413 },
		{ 2480, 2479 },
		{ 1580, 1579 },
		{ 1964, 1946 },
		{ 2256, 2235 },
		{ 2487, 2486 },
		{ 1975, 1962 },
		{ 1976, 1963 },
		{ 1780, 1779 },
		{ 2227, 2227 },
		{ 1986, 1981 },
		{ 2212, 2212 },
		{ 1457, 1456 },
		{ 2584, 2584 },
		{ 2505, 2504 },
		{ 2477, 2477 },
		{ 2366, 2352 },
		{ 2625, 2625 },
		{ 1128, 1127 },
		{ 2510, 2509 },
		{ 2269, 2249 },
		{ 1585, 1584 },
		{ 1591, 1590 },
		{ 1676, 1657 },
		{ 2519, 2517 },
		{ 1999, 1997 },
		{ 1235, 1212 },
		{ 2278, 2258 },
		{ 1845, 1823 },
		{ 2281, 2263 },
		{ 2386, 2375 },
		{ 2538, 2537 },
		{ 1846, 1825 },
		{ 1271, 1249 },
		{ 1177, 1176 },
		{ 2546, 2545 },
		{ 2555, 2554 },
		{ 2111, 2089 },
		{ 1429, 1428 },
		{ 2571, 2570 },
		{ 2572, 2571 },
		{ 2574, 2573 },
		{ 1507, 1506 },
		{ 1859, 1838 },
		{ 2587, 2586 },
		{ 2036, 2035 },
		{ 1311, 1292 },
		{ 1706, 1689 },
		{ 2594, 2593 },
		{ 2406, 2397 },
		{ 2598, 2597 },
		{ 1108, 1107 },
		{ 1143, 1142 },
		{ 2603, 2602 },
		{ 1711, 1696 },
		{ 1441, 1440 },
		{ 2615, 2614 },
		{ 1721, 1705 },
		{ 1881, 1860 },
		{ 2628, 2627 },
		{ 1165, 1164 },
		{ 2321, 2304 },
		{ 1483, 1482 },
		{ 2635, 2634 },
		{ 1654, 1632 },
		{ 2639, 2638 },
		{ 1658, 1639 },
		{ 2330, 2313 },
		{ 2644, 2643 },
		{ 2649, 2648 },
		{ 2432, 2431 },
		{ 1745, 1733 },
		{ 102, 88 },
		{ 1758, 1749 },
		{ 1904, 1887 },
		{ 1570, 1569 },
		{ 2665, 2664 },
		{ 2460, 2459 },
		{ 2669, 2668 },
		{ 1770, 1765 },
		{ 2247, 2227 },
		{ 1138, 1137 },
		{ 2229, 2212 },
		{ 2674, 2673 },
		{ 2585, 2584 },
		{ 1509, 1508 },
		{ 2478, 2477 },
		{ 1548, 1547 },
		{ 2626, 2625 },
		{ 1135, 1135 },
		{ 1135, 1135 },
		{ 1577, 1577 },
		{ 1577, 1577 },
		{ 2666, 2666 },
		{ 2666, 2666 },
		{ 2636, 2636 },
		{ 2636, 2636 },
		{ 2457, 2457 },
		{ 2457, 2457 },
		{ 2381, 2381 },
		{ 2381, 2381 },
		{ 2595, 2595 },
		{ 2595, 2595 },
		{ 2502, 2502 },
		{ 2502, 2502 },
		{ 2042, 2041 },
		{ 1593, 1592 },
		{ 1913, 1897 },
		{ 2316, 2299 },
		{ 1992, 1989 },
		{ 1310, 1291 },
		{ 1485, 1484 },
		{ 1135, 1135 },
		{ 1535, 1534 },
		{ 1577, 1577 },
		{ 1459, 1458 },
		{ 2666, 2666 },
		{ 1720, 1704 },
		{ 2636, 2636 },
		{ 2008, 2007 },
		{ 2457, 2457 },
		{ 1885, 1866 },
		{ 2381, 2381 },
		{ 1775, 1773 },
		{ 2595, 2595 },
		{ 1431, 1430 },
		{ 2502, 2502 },
		{ 1494, 1493 },
		{ 1692, 1673 },
		{ 1167, 1166 },
		{ 2187, 2170 },
		{ 2579, 2579 },
		{ 2579, 2579 },
		{ 2472, 2472 },
		{ 2472, 2472 },
		{ 1119, 1119 },
		{ 1119, 1119 },
		{ 2613, 2613 },
		{ 2613, 2613 },
		{ 1572, 1572 },
		{ 1572, 1572 },
		{ 1561, 1561 },
		{ 1561, 1561 },
		{ 2655, 2655 },
		{ 2655, 2655 },
		{ 2620, 2620 },
		{ 2620, 2620 },
		{ 1130, 1130 },
		{ 1130, 1130 },
		{ 1153, 1152 },
		{ 2368, 2354 },
		{ 2310, 2293 },
		{ 1883, 1862 },
		{ 1884, 1864 },
		{ 2579, 2579 },
		{ 1278, 1256 },
		{ 2472, 2472 },
		{ 1996, 1993 },
		{ 1119, 1119 },
		{ 2530, 2529 },
		{ 2613, 2613 },
		{ 1596, 1595 },
		{ 1572, 1572 },
		{ 1538, 1537 },
		{ 1561, 1561 },
		{ 1488, 1487 },
		{ 2655, 2655 },
		{ 1512, 1511 },
		{ 2620, 2620 },
		{ 1170, 1169 },
		{ 1130, 1130 },
		{ 1606, 1605 },
		{ 1903, 1886 },
		{ 1574, 1573 },
		{ 2031, 2030 },
		{ 1736, 1723 },
		{ 1446, 1445 },
		{ 1856, 1835 },
		{ 1182, 1181 },
		{ 1748, 1738 },
		{ 1679, 1660 },
		{ 1961, 1943 },
		{ 1136, 1135 },
		{ 1521, 1520 },
		{ 1578, 1577 },
		{ 2659, 2658 },
		{ 2667, 2666 },
		{ 1132, 1131 },
		{ 2637, 2636 },
		{ 2408, 2400 },
		{ 2458, 2457 },
		{ 1690, 1671 },
		{ 2391, 2381 },
		{ 1375, 1349 },
		{ 2596, 2595 },
		{ 2353, 2339 },
		{ 2503, 2502 },
		{ 1434, 1433 },
		{ 2301, 2284 },
		{ 1655, 1635 },
		{ 1101, 1100 },
		{ 1462, 1461 },
		{ 2526, 2525 },
		{ 1097, 1097 },
		{ 1097, 1097 },
		{ 2553, 2552 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 1210, 1201 },
		{ 1629, 1785 },
		{ 1820, 2011 },
		{ 2554, 2553 },
		{ 1444, 1443 },
		{ 1583, 1582 },
		{ 2361, 2347 },
		{ 2580, 2579 },
		{ 1211, 1190 },
		{ 2473, 2472 },
		{ 1097, 1097 },
		{ 1120, 1119 },
		{ 2570, 2569 },
		{ 2614, 2613 },
		{ 2299, 2282 },
		{ 1573, 1572 },
		{ 1944, 1930 },
		{ 1562, 1561 },
		{ 2573, 2572 },
		{ 2656, 2655 },
		{ 2367, 2353 },
		{ 2621, 2620 },
		{ 1945, 1931 },
		{ 1131, 1130 },
		{ 2132, 2132 },
		{ 2132, 2132 },
		{ 2132, 2132 },
		{ 2132, 2132 },
		{ 2132, 2132 },
		{ 2132, 2132 },
		{ 2132, 2132 },
		{ 2132, 2132 },
		{ 2132, 2132 },
		{ 2132, 2132 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 1089, 1089 },
		{ 1089, 1089 },
		{ 1089, 1089 },
		{ 1089, 1089 },
		{ 1089, 1089 },
		{ 1089, 1089 },
		{ 1089, 1089 },
		{ 1089, 1089 },
		{ 1089, 1089 },
		{ 1089, 1089 },
		{ 1151, 1151 },
		{ 1151, 1151 },
		{ 122, 122 },
		{ 122, 122 },
		{ 122, 122 },
		{ 122, 122 },
		{ 122, 122 },
		{ 122, 122 },
		{ 122, 122 },
		{ 122, 122 },
		{ 122, 122 },
		{ 122, 122 },
		{ 0, 1077 },
		{ 0, 76 },
		{ 2548, 2548 },
		{ 2548, 2548 },
		{ 0, 2065 },
		{ 2488, 2488 },
		{ 2488, 2488 },
		{ 2566, 2566 },
		{ 2566, 2566 },
		{ 1212, 1190 },
		{ 2463, 2462 },
		{ 1151, 1151 },
		{ 1098, 1097 },
		{ 107, 107 },
		{ 107, 107 },
		{ 107, 107 },
		{ 107, 107 },
		{ 107, 107 },
		{ 107, 107 },
		{ 107, 107 },
		{ 107, 107 },
		{ 107, 107 },
		{ 107, 107 },
		{ 1447, 1446 },
		{ 1127, 1126 },
		{ 2548, 2548 },
		{ 1656, 1636 },
		{ 1180, 1179 },
		{ 2488, 2488 },
		{ 2592, 2591 },
		{ 2566, 2566 },
		{ 2206, 2187 },
		{ 0, 1077 },
		{ 0, 76 },
		{ 1141, 1140 },
		{ 1778, 1776 },
		{ 0, 2065 },
		{ 1084, 1084 },
		{ 1084, 1084 },
		{ 1084, 1084 },
		{ 1084, 1084 },
		{ 1084, 1084 },
		{ 1084, 1084 },
		{ 1084, 1084 },
		{ 1084, 1084 },
		{ 1084, 1084 },
		{ 1084, 1084 },
		{ 120, 120 },
		{ 120, 120 },
		{ 120, 120 },
		{ 120, 120 },
		{ 120, 120 },
		{ 120, 120 },
		{ 120, 120 },
		{ 120, 120 },
		{ 120, 120 },
		{ 120, 120 },
		{ 1087, 1087 },
		{ 1087, 1087 },
		{ 1087, 1087 },
		{ 1087, 1087 },
		{ 1087, 1087 },
		{ 1087, 1087 },
		{ 1087, 1087 },
		{ 1087, 1087 },
		{ 1087, 1087 },
		{ 1087, 1087 },
		{ 2380, 2368 },
		{ 2313, 2296 },
		{ 2485, 2484 },
		{ 2601, 2600 },
		{ 2170, 2152 },
		{ 1660, 1641 },
		{ 1183, 1182 },
		{ 2387, 2376 },
		{ 1710, 1695 },
		{ 1176, 1175 },
		{ 1106, 1105 },
		{ 1152, 1151 },
		{ 1824, 1802 },
		{ 1569, 1568 },
		{ 2328, 2311 },
		{ 1994, 1991 },
		{ 2508, 2507 },
		{ 2633, 2632 },
		{ 1603, 1602 },
		{ 2400, 2391 },
		{ 2401, 2392 },
		{ 2333, 2316 },
		{ 1440, 1439 },
		{ 2274, 2254 },
		{ 2642, 2641 },
		{ 2549, 2548 },
		{ 1178, 1177 },
		{ 2489, 2488 },
		{ 1442, 1441 },
		{ 2567, 2566 },
		{ 2339, 2323 },
		{ 1733, 1719 },
		{ 2529, 2528 },
		{ 2005, 2003 },
		{ 1547, 1546 },
		{ 1292, 1272 },
		{ 1549, 1548 },
		{ 2536, 2535 },
		{ 2663, 2662 },
		{ 2283, 2265 },
		{ 2286, 2269 },
		{ 1914, 1898 },
		{ 2352, 2338 },
		{ 1638, 1616 },
		{ 2430, 2429 },
		{ 2672, 2671 },
		{ 2354, 2340 },
		{ 1928, 1912 },
		{ 2550, 2549 },
		{ 2583, 2582 },
		{ 2611, 2610 },
		{ 1126, 1125 },
		{ 2007, 2006 },
		{ 59, 5 },
		{ 1568, 1567 },
		{ 2624, 2623 },
		{ 2476, 2475 },
		{ 2564, 2563 },
		{ 2569, 2568 },
		{ 2378, 2366 },
		{ 2556, 2555 },
		{ 2345, 2330 },
		{ 1998, 1996 },
		{ 1571, 1570 },
		{ 2575, 2574 },
		{ 1129, 1128 },
		{ 2516, 2514 },
		{ 1649, 1626 },
		{ 2028, 2027 },
		{ 2491, 2490 },
		{ 2205, 2186 },
		{ 1650, 1626 },
		{ 1631, 1611 },
		{ 1272, 1250 },
		{ 2219, 2202 },
		{ 1822, 1799 },
		{ 1115, 1114 },
		{ 1630, 1611 },
		{ 2528, 2527 },
		{ 1825, 1802 },
		{ 1821, 1799 },
		{ 2578, 2577 },
		{ 2517, 2514 },
		{ 1557, 1556 },
		{ 2271, 2251 },
		{ 2013, 2010 },
		{ 2288, 2271 },
		{ 2238, 2219 },
		{ 1787, 1784 },
		{ 2619, 2618 },
		{ 2012, 2010 },
		{ 1228, 1202 },
		{ 2531, 2530 },
		{ 1786, 1784 },
		{ 1766, 1758 },
		{ 1576, 1575 },
		{ 2394, 2384 },
		{ 1134, 1133 },
		{ 2298, 2281 },
		{ 1486, 1485 },
		{ 1536, 1535 },
		{ 1164, 1163 },
		{ 1777, 1775 },
		{ 1449, 1448 },
		{ 2304, 2287 },
		{ 161, 5 },
		{ 1331, 1310 },
		{ 2551, 2550 },
		{ 2308, 2291 },
		{ 162, 5 },
		{ 1639, 1616 },
		{ 1456, 1455 },
		{ 1590, 1589 },
		{ 1946, 1932 },
		{ 2414, 2408 },
		{ 1332, 1311 },
		{ 163, 5 },
		{ 2089, 2067 },
		{ 2565, 2564 },
		{ 1428, 1427 },
		{ 2568, 2567 },
		{ 1095, 1094 },
		{ 1685, 1666 },
		{ 2317, 2300 },
		{ 1823, 1800 },
		{ 2425, 2423 },
		{ 1594, 1593 },
		{ 1977, 1964 },
		{ 1981, 1974 },
		{ 1460, 1459 },
		{ 1689, 1670 },
		{ 1149, 1148 },
		{ 160, 5 },
		{ 1506, 1505 },
		{ 2586, 2585 },
		{ 1693, 1674 },
		{ 2588, 2587 },
		{ 2455, 2454 },
		{ 1551, 1550 },
		{ 1696, 1678 },
		{ 1122, 1121 },
		{ 1995, 1992 },
		{ 1432, 1431 },
		{ 1168, 1167 },
		{ 1559, 1558 },
		{ 2249, 2229 },
		{ 1708, 1692 },
		{ 2346, 2331 },
		{ 1860, 1839 },
		{ 1861, 1840 },
		{ 1510, 1509 },
		{ 2479, 2478 },
		{ 2612, 2611 },
		{ 1249, 1227 },
		{ 2481, 2480 },
		{ 1632, 1612 },
		{ 2258, 2237 },
		{ 1633, 1613 },
		{ 1634, 1613 },
		{ 1250, 1228 },
		{ 2263, 2241 },
		{ 2627, 2626 },
		{ 1564, 1563 },
		{ 2629, 2628 },
		{ 2490, 2489 },
		{ 1878, 1857 },
		{ 2027, 2026 },
		{ 1185, 1184 },
		{ 2494, 2493 },
		{ 1518, 1517 },
		{ 2497, 2496 },
		{ 1099, 1098 },
		{ 2501, 2500 },
		{ 88, 72 },
		{ 1475, 1474 },
		{ 1118, 1117 },
		{ 1482, 1481 },
		{ 2372, 2358 },
		{ 1887, 1869 },
		{ 2652, 2651 },
		{ 1895, 1877 },
		{ 1657, 1637 },
		{ 2043, 2042 },
		{ 2280, 2260 },
		{ 1749, 1739 },
		{ 2045, 2044 },
		{ 1114, 1113 },
		{ 2520, 2518 },
		{ 2285, 2268 },
		{ 1532, 1531 },
		{ 2203, 2184 },
		{ 2527, 2526 },
		{ 1157, 1156 },
		{ 1902, 1885 },
		{ 1765, 1757 },
		{ 2654, 2653 },
		{ 2547, 2546 },
		{ 1867, 1846 },
		{ 1687, 1668 },
		{ 2498, 2497 },
		{ 2133, 2111 },
		{ 2616, 2615 },
		{ 1602, 1601 },
		{ 1257, 1235 },
		{ 2220, 2203 },
		{ 1635, 1613 },
		{ 2521, 2519 },
		{ 1697, 1679 },
		{ 116, 102 },
		{ 1880, 1859 },
		{ 1839, 1816 },
		{ 1677, 1658 },
		{ 1100, 1099 },
		{ 2500, 2499 },
		{ 2599, 2598 },
		{ 2382, 2370 },
		{ 1929, 1914 },
		{ 2582, 2581 },
		{ 2260, 2239 },
		{ 2475, 2474 },
		{ 2506, 2505 },
		{ 1581, 1580 },
		{ 1468, 1467 },
		{ 568, 516 },
		{ 2461, 2460 },
		{ 2640, 2639 },
		{ 2294, 2277 },
		{ 2670, 2669 },
		{ 1725, 1710 },
		{ 2552, 2551 },
		{ 1139, 1138 },
		{ 2623, 2622 },
		{ 2340, 2324 },
		{ 569, 516 },
		{ 2323, 2306 },
		{ 2289, 2272 },
		{ 2307, 2290 },
		{ 1820, 1814 },
		{ 1210, 1203 },
		{ 1629, 1625 },
		{ 189, 166 },
		{ 1154, 1153 },
		{ 1982, 1975 },
		{ 187, 166 },
		{ 1651, 1627 },
		{ 188, 166 },
		{ 2483, 2482 },
		{ 1104, 1103 },
		{ 2306, 2289 },
		{ 570, 516 },
		{ 1523, 1522 },
		{ 1988, 1984 },
		{ 2270, 2250 },
		{ 2190, 2173 },
		{ 186, 166 },
		{ 2039, 2038 },
		{ 2622, 2621 },
		{ 2411, 2403 },
		{ 1607, 1606 },
		{ 2041, 2040 },
		{ 1489, 1488 },
		{ 2362, 2348 },
		{ 1835, 1811 },
		{ 1915, 1900 },
		{ 2631, 2630 },
		{ 1916, 1903 },
		{ 2319, 2302 },
		{ 1463, 1462 },
		{ 1181, 1180 },
		{ 2428, 2427 },
		{ 1391, 1375 },
		{ 1838, 1815 },
		{ 2374, 2360 },
		{ 2324, 2307 },
		{ 1774, 1771 },
		{ 2284, 2267 },
		{ 1171, 1170 },
		{ 1943, 1929 },
		{ 2581, 2580 },
		{ 1513, 1512 },
		{ 2332, 2315 },
		{ 2653, 2652 },
		{ 2208, 2189 },
		{ 1435, 1434 },
		{ 1497, 1496 },
		{ 2658, 2657 },
		{ 1597, 1596 },
		{ 1445, 1444 },
		{ 2661, 2660 },
		{ 2590, 2589 },
		{ 2525, 2524 },
		{ 1539, 1538 },
		{ 1738, 1725 },
		{ 2341, 2326 },
		{ 1102, 1101 },
		{ 2393, 2383 },
		{ 2474, 2473 },
		{ 1746, 1736 },
		{ 2029, 2028 },
		{ 2218, 2201 },
		{ 1520, 1519 },
		{ 393, 355 },
		{ 619, 567 },
		{ 672, 624 },
		{ 763, 722 },
		{ 1162, 17 },
		{ 1454, 23 },
		{ 1530, 29 },
		{ 59, 17 },
		{ 59, 23 },
		{ 59, 29 },
		{ 1480, 25 },
		{ 1504, 27 },
		{ 1092, 11 },
		{ 59, 25 },
		{ 59, 27 },
		{ 59, 11 },
		{ 394, 355 },
		{ 2452, 43 },
		{ 939, 915 },
		{ 671, 624 },
		{ 59, 43 },
		{ 947, 924 },
		{ 960, 938 },
		{ 620, 567 },
		{ 1681, 1662 },
		{ 1682, 1663 },
		{ 965, 944 },
		{ 969, 948 },
		{ 970, 949 },
		{ 764, 722 },
		{ 999, 984 },
		{ 1004, 990 },
		{ 1871, 1850 },
		{ 1872, 1851 },
		{ 1010, 1000 },
		{ 1020, 1011 },
		{ 1026, 1017 },
		{ 1061, 1060 },
		{ 260, 224 },
		{ 274, 235 },
		{ 1702, 1684 },
		{ 279, 240 },
		{ 285, 245 },
		{ 294, 254 },
		{ 310, 269 },
		{ 1894, 1876 },
		{ 313, 272 },
		{ 319, 277 },
		{ 324, 282 },
		{ 337, 298 },
		{ 1717, 1701 },
		{ 358, 320 },
		{ 361, 323 },
		{ 367, 329 },
		{ 378, 341 },
		{ 1910, 1893 },
		{ 384, 347 },
		{ 215, 182 },
		{ 404, 364 },
		{ 219, 186 },
		{ 425, 379 },
		{ 428, 382 },
		{ 438, 390 },
		{ 439, 391 },
		{ 449, 401 },
		{ 457, 410 },
		{ 487, 434 },
		{ 496, 441 },
		{ 498, 443 },
		{ 499, 444 },
		{ 501, 446 },
		{ 513, 460 },
		{ 520, 467 },
		{ 530, 477 },
		{ 543, 488 },
		{ 544, 490 },
		{ 549, 495 },
		{ 220, 187 },
		{ 573, 519 },
		{ 588, 534 },
		{ 596, 542 },
		{ 609, 554 },
		{ 618, 566 },
		{ 226, 193 },
		{ 634, 581 },
		{ 244, 208 },
		{ 1160, 17 },
		{ 1452, 23 },
		{ 1528, 29 },
		{ 673, 625 },
		{ 685, 636 },
		{ 687, 638 },
		{ 1478, 25 },
		{ 1502, 27 },
		{ 1090, 11 },
		{ 688, 639 },
		{ 692, 643 },
		{ 709, 660 },
		{ 759, 718 },
		{ 2450, 43 },
		{ 253, 217 },
		{ 788, 749 },
		{ 796, 757 },
		{ 1661, 1642 },
		{ 806, 767 },
		{ 839, 800 },
		{ 854, 818 },
		{ 868, 836 },
		{ 887, 858 },
		{ 890, 861 },
		{ 902, 874 },
		{ 918, 892 },
		{ 1849, 1828 },
		{ 930, 905 },
		{ 937, 913 },
		{ 59, 39 },
		{ 59, 55 },
		{ 59, 13 },
		{ 59, 49 },
		{ 59, 53 },
		{ 59, 21 },
		{ 59, 33 },
		{ 59, 15 },
		{ 59, 45 },
		{ 59, 31 },
		{ 59, 51 },
		{ 59, 57 },
		{ 1924, 1909 },
		{ 1922, 1908 },
		{ 1983, 1976 },
		{ 430, 384 },
		{ 432, 384 },
		{ 1729, 1715 },
		{ 400, 359 },
		{ 1925, 1909 },
		{ 1923, 1908 },
		{ 397, 358 },
		{ 654, 607 },
		{ 398, 358 },
		{ 728, 680 },
		{ 655, 608 },
		{ 433, 384 },
		{ 536, 483 },
		{ 459, 412 },
		{ 466, 419 },
		{ 467, 419 },
		{ 431, 384 },
		{ 377, 340 },
		{ 1920, 1906 },
		{ 306, 265 },
		{ 1832, 1808 },
		{ 725, 677 },
		{ 468, 419 },
		{ 198, 171 },
		{ 269, 230 },
		{ 399, 358 },
		{ 197, 171 },
		{ 617, 565 },
		{ 1831, 1808 },
		{ 202, 174 },
		{ 1853, 1832 },
		{ 406, 366 },
		{ 199, 171 },
		{ 330, 288 },
		{ 480, 428 },
		{ 387, 349 },
		{ 386, 349 },
		{ 210, 177 },
		{ 974, 953 },
		{ 479, 428 },
		{ 1830, 1808 },
		{ 470, 421 },
		{ 1664, 1645 },
		{ 473, 423 },
		{ 204, 174 },
		{ 203, 174 },
		{ 474, 423 },
		{ 540, 487 },
		{ 277, 238 },
		{ 209, 177 },
		{ 529, 476 },
		{ 541, 487 },
		{ 1854, 1833 },
		{ 729, 681 },
		{ 730, 682 },
		{ 643, 594 },
		{ 973, 953 },
		{ 207, 175 },
		{ 247, 211 },
		{ 734, 686 },
		{ 205, 175 },
		{ 471, 421 },
		{ 542, 487 },
		{ 206, 175 },
		{ 2024, 39 },
		{ 2605, 55 },
		{ 1111, 13 },
		{ 2514, 49 },
		{ 2558, 53 },
		{ 1425, 21 },
		{ 1587, 33 },
		{ 1145, 15 },
		{ 2468, 45 },
		{ 1554, 31 },
		{ 2541, 51 },
		{ 2646, 57 },
		{ 261, 225 },
		{ 557, 502 },
		{ 1018, 1009 },
		{ 644, 595 },
		{ 922, 896 },
		{ 1093, 1090 },
		{ 773, 733 },
		{ 646, 597 },
		{ 938, 914 },
		{ 221, 188 },
		{ 714, 665 },
		{ 445, 397 },
		{ 1147, 1145 },
		{ 846, 809 },
		{ 850, 814 },
		{ 262, 225 },
		{ 389, 351 },
		{ 2453, 2450 },
		{ 628, 575 },
		{ 997, 982 },
		{ 1644, 1620 },
		{ 518, 465 },
		{ 1002, 987 },
		{ 192, 168 },
		{ 510, 457 },
		{ 194, 169 },
		{ 485, 432 },
		{ 355, 318 },
		{ 193, 168 },
		{ 282, 243 },
		{ 195, 169 },
		{ 807, 768 },
		{ 813, 774 },
		{ 818, 779 },
		{ 823, 784 },
		{ 827, 788 },
		{ 832, 794 },
		{ 316, 274 },
		{ 509, 457 },
		{ 364, 326 },
		{ 183, 165 },
		{ 505, 451 },
		{ 855, 819 },
		{ 865, 833 },
		{ 185, 165 },
		{ 631, 578 },
		{ 315, 274 },
		{ 314, 274 },
		{ 873, 841 },
		{ 874, 842 },
		{ 880, 848 },
		{ 184, 165 },
		{ 506, 452 },
		{ 370, 332 },
		{ 511, 458 },
		{ 904, 876 },
		{ 906, 878 },
		{ 910, 882 },
		{ 916, 890 },
		{ 917, 891 },
		{ 320, 278 },
		{ 647, 598 },
		{ 923, 898 },
		{ 323, 281 },
		{ 275, 236 },
		{ 524, 471 },
		{ 528, 475 },
		{ 385, 348 },
		{ 959, 937 },
		{ 450, 402 },
		{ 961, 939 },
		{ 534, 481 },
		{ 245, 209 },
		{ 335, 295 },
		{ 710, 661 },
		{ 978, 960 },
		{ 983, 966 },
		{ 990, 974 },
		{ 991, 976 },
		{ 711, 662 },
		{ 258, 222 },
		{ 1000, 985 },
		{ 717, 668 },
		{ 718, 670 },
		{ 723, 675 },
		{ 1011, 1001 },
		{ 1013, 1003 },
		{ 724, 676 },
		{ 339, 300 },
		{ 1663, 1644 },
		{ 1024, 1015 },
		{ 1025, 1016 },
		{ 341, 302 },
		{ 1030, 1021 },
		{ 1045, 1036 },
		{ 477, 426 },
		{ 1072, 1071 },
		{ 559, 504 },
		{ 563, 509 },
		{ 743, 698 },
		{ 744, 699 },
		{ 751, 709 },
		{ 757, 716 },
		{ 567, 515 },
		{ 403, 362 },
		{ 771, 731 },
		{ 482, 430 },
		{ 1851, 1830 },
		{ 777, 737 },
		{ 785, 746 },
		{ 301, 260 },
		{ 228, 195 },
		{ 732, 684 },
		{ 633, 580 },
		{ 883, 851 },
		{ 884, 852 },
		{ 885, 854 },
		{ 402, 361 },
		{ 502, 447 },
		{ 1354, 1354 },
		{ 1357, 1357 },
		{ 1360, 1360 },
		{ 1363, 1363 },
		{ 1889, 1871 },
		{ 1366, 1366 },
		{ 1369, 1369 },
		{ 892, 863 },
		{ 338, 299 },
		{ 903, 875 },
		{ 645, 596 },
		{ 1055, 1049 },
		{ 461, 415 },
		{ 907, 879 },
		{ 552, 498 },
		{ 553, 499 },
		{ 1387, 1387 },
		{ 508, 456 },
		{ 1699, 1681 },
		{ 662, 617 },
		{ 374, 337 },
		{ 1317, 1317 },
		{ 560, 505 },
		{ 1354, 1354 },
		{ 1357, 1357 },
		{ 1360, 1360 },
		{ 1363, 1363 },
		{ 793, 754 },
		{ 1366, 1366 },
		{ 1369, 1369 },
		{ 1396, 1396 },
		{ 935, 911 },
		{ 795, 756 },
		{ 321, 279 },
		{ 565, 513 },
		{ 943, 919 },
		{ 214, 181 },
		{ 381, 344 },
		{ 345, 306 },
		{ 1387, 1387 },
		{ 580, 526 },
		{ 522, 469 },
		{ 831, 793 },
		{ 594, 540 },
		{ 1317, 1317 },
		{ 225, 192 },
		{ 213, 180 },
		{ 1421, 1421 },
		{ 614, 559 },
		{ 851, 815 },
		{ 331, 291 },
		{ 995, 980 },
		{ 362, 324 },
		{ 1396, 1396 },
		{ 532, 479 },
		{ 621, 568 },
		{ 259, 223 },
		{ 1716, 1700 },
		{ 886, 856 },
		{ 546, 492 },
		{ 548, 494 },
		{ 735, 687 },
		{ 1227, 1354 },
		{ 1227, 1357 },
		{ 1227, 1360 },
		{ 1227, 1363 },
		{ 740, 694 },
		{ 1227, 1366 },
		{ 1227, 1369 },
		{ 742, 696 },
		{ 1421, 1421 },
		{ 365, 327 },
		{ 639, 586 },
		{ 745, 700 },
		{ 455, 407 },
		{ 278, 239 },
		{ 329, 287 },
		{ 371, 333 },
		{ 1227, 1387 },
		{ 1743, 1731 },
		{ 1744, 1732 },
		{ 343, 304 },
		{ 649, 600 },
		{ 1227, 1317 },
		{ 650, 602 },
		{ 652, 605 },
		{ 936, 912 },
		{ 238, 203 },
		{ 350, 311 },
		{ 658, 610 },
		{ 1893, 1875 },
		{ 661, 615 },
		{ 1227, 1396 },
		{ 802, 763 },
		{ 951, 928 },
		{ 954, 931 },
		{ 955, 932 },
		{ 805, 766 },
		{ 566, 514 },
		{ 351, 313 },
		{ 808, 769 },
		{ 811, 772 },
		{ 1907, 1890 },
		{ 478, 427 },
		{ 302, 261 },
		{ 820, 781 },
		{ 821, 782 },
		{ 822, 783 },
		{ 201, 173 },
		{ 1227, 1421 },
		{ 584, 530 },
		{ 587, 533 },
		{ 697, 648 },
		{ 702, 653 },
		{ 336, 296 },
		{ 589, 535 },
		{ 592, 538 },
		{ 852, 816 },
		{ 1940, 1926 },
		{ 1941, 1927 },
		{ 309, 268 },
		{ 489, 436 },
		{ 857, 824 },
		{ 862, 830 },
		{ 441, 393 },
		{ 866, 834 },
		{ 497, 442 },
		{ 1701, 1683 },
		{ 870, 838 },
		{ 537, 484 },
		{ 1058, 1056 },
		{ 443, 395 },
		{ 1062, 1061 },
		{ 727, 679 },
		{ 390, 352 },
		{ 357, 319 },
		{ 356, 319 },
		{ 264, 226 },
		{ 263, 226 },
		{ 604, 549 },
		{ 605, 549 },
		{ 491, 438 },
		{ 492, 438 },
		{ 493, 439 },
		{ 494, 439 },
		{ 231, 197 },
		{ 556, 501 },
		{ 191, 167 },
		{ 230, 197 },
		{ 555, 501 },
		{ 242, 206 },
		{ 464, 418 },
		{ 682, 634 },
		{ 790, 751 },
		{ 268, 229 },
		{ 283, 244 },
		{ 241, 206 },
		{ 190, 167 },
		{ 1911, 1894 },
		{ 1718, 1702 },
		{ 731, 683 },
		{ 465, 418 },
		{ 683, 634 },
		{ 284, 244 },
		{ 267, 229 },
		{ 326, 284 },
		{ 693, 644 },
		{ 576, 522 },
		{ 651, 604 },
		{ 427, 381 },
		{ 475, 424 },
		{ 860, 827 },
		{ 1926, 1910 },
		{ 1731, 1717 },
		{ 327, 285 },
		{ 1826, 1803 },
		{ 976, 958 },
		{ 1064, 1063 },
		{ 977, 959 },
		{ 255, 219 },
		{ 979, 961 },
		{ 252, 216 },
		{ 270, 231 },
		{ 819, 780 },
		{ 1662, 1643 },
		{ 1700, 1682 },
		{ 993, 978 },
		{ 1890, 1872 },
		{ 407, 367 },
		{ 1665, 1646 },
		{ 545, 491 },
		{ 876, 844 },
		{ 442, 394 },
		{ 1850, 1829 },
		{ 606, 550 },
		{ 547, 493 },
		{ 940, 916 },
		{ 1855, 1834 },
		{ 240, 205 },
		{ 690, 641 },
		{ 963, 941 },
		{ 893, 864 },
		{ 967, 946 },
		{ 679, 631 },
		{ 681, 633 },
		{ 797, 758 },
		{ 211, 178 },
		{ 1694, 1675 },
		{ 243, 207 },
		{ 1959, 1940 },
		{ 1643, 1619 },
		{ 856, 823 },
		{ 1645, 1621 },
		{ 436, 387 },
		{ 535, 482 },
		{ 507, 455 },
		{ 920, 894 },
		{ 550, 496 },
		{ 766, 724 },
		{ 926, 901 },
		{ 405, 365 },
		{ 931, 906 },
		{ 932, 907 },
		{ 933, 908 },
		{ 1008, 998 },
		{ 696, 647 },
		{ 641, 588 },
		{ 1829, 1807 },
		{ 779, 739 },
		{ 1833, 1809 },
		{ 782, 743 },
		{ 877, 845 },
		{ 783, 744 },
		{ 881, 849 },
		{ 824, 785 },
		{ 538, 485 },
		{ 1031, 1022 },
		{ 1730, 1716 },
		{ 1042, 1033 },
		{ 1043, 1034 },
		{ 952, 929 },
		{ 1048, 1039 },
		{ 786, 747 },
		{ 1056, 1051 },
		{ 1057, 1052 },
		{ 526, 473 },
		{ 1921, 1907 },
		{ 957, 934 },
		{ 446, 398 },
		{ 844, 806 },
		{ 1754, 1743 },
		{ 232, 198 },
		{ 656, 609 },
		{ 1007, 997 },
		{ 271, 232 },
		{ 392, 354 },
		{ 836, 797 },
		{ 1012, 1002 },
		{ 657, 609 },
		{ 944, 920 },
		{ 1017, 1008 },
		{ 945, 922 },
		{ 595, 541 },
		{ 949, 926 },
		{ 346, 307 },
		{ 1714, 1698 },
		{ 564, 510 },
		{ 1027, 1018 },
		{ 229, 196 },
		{ 525, 472 },
		{ 1905, 1888 },
		{ 1040, 1031 },
		{ 956, 933 },
		{ 481, 429 },
		{ 905, 877 },
		{ 408, 368 },
		{ 1049, 1042 },
		{ 1050, 1043 },
		{ 1054, 1048 },
		{ 799, 760 },
		{ 909, 881 },
		{ 572, 518 },
		{ 413, 376 },
		{ 1852, 1831 },
		{ 1059, 1057 },
		{ 747, 705 },
		{ 462, 416 },
		{ 972, 952 },
		{ 1071, 1070 },
		{ 919, 893 },
		{ 975, 957 },
		{ 755, 714 },
		{ 624, 571 },
		{ 812, 773 },
		{ 924, 899 },
		{ 980, 963 },
		{ 925, 900 },
		{ 984, 967 },
		{ 989, 973 },
		{ 627, 574 },
		{ 761, 720 },
		{ 246, 210 },
		{ 722, 674 },
		{ 363, 325 },
		{ 585, 531 },
		{ 426, 380 },
		{ 281, 242 },
		{ 516, 463 },
		{ 1006, 996 },
		{ 622, 569 },
		{ 689, 640 },
		{ 847, 810 },
		{ 216, 183 },
		{ 375, 338 },
		{ 218, 185 },
		{ 601, 546 },
		{ 603, 548 },
		{ 701, 652 },
		{ 401, 360 },
		{ 635, 582 },
		{ 411, 372 },
		{ 562, 508 },
		{ 613, 558 },
		{ 328, 286 },
		{ 791, 752 },
		{ 825, 786 },
		{ 308, 267 },
		{ 749, 707 },
		{ 1069, 1068 },
		{ 539, 486 },
		{ 551, 497 },
		{ 383, 346 },
		{ 1891, 1873 },
		{ 1892, 1874 },
		{ 495, 440 },
		{ 792, 753 },
		{ 828, 789 },
		{ 1046, 1037 },
		{ 1047, 1038 },
		{ 829, 790 },
		{ 632, 579 },
		{ 237, 202 },
		{ 1876, 1855 },
		{ 1053, 1047 },
		{ 265, 227 },
		{ 248, 212 },
		{ 988, 971 },
		{ 843, 804 },
		{ 888, 859 },
		{ 889, 860 },
		{ 992, 977 },
		{ 691, 642 },
		{ 994, 979 },
		{ 1065, 1064 },
		{ 726, 678 },
		{ 636, 583 },
		{ 895, 867 },
		{ 897, 869 },
		{ 292, 252 },
		{ 660, 612 },
		{ 1005, 993 },
		{ 217, 184 },
		{ 809, 770 },
		{ 575, 521 },
		{ 776, 736 },
		{ 322, 280 },
		{ 817, 778 },
		{ 437, 389 },
		{ 297, 257 },
		{ 962, 940 },
		{ 739, 693 },
		{ 1684, 1665 },
		{ 334, 294 },
		{ 869, 837 },
		{ 921, 895 },
		{ 713, 664 },
		{ 971, 950 },
		{ 629, 576 },
		{ 1032, 1023 },
		{ 1037, 1028 },
		{ 715, 666 },
		{ 1041, 1032 },
		{ 659, 611 },
		{ 748, 706 },
		{ 1215, 1215 },
		{ 872, 840 },
		{ 523, 470 },
		{ 484, 431 },
		{ 298, 258 },
		{ 296, 256 },
		{ 914, 887 },
		{ 483, 431 },
		{ 299, 258 },
		{ 664, 618 },
		{ 663, 618 },
		{ 913, 887 },
		{ 665, 618 },
		{ 1052, 1045 },
		{ 894, 865 },
		{ 765, 723 },
		{ 607, 552 },
		{ 900, 872 },
		{ 901, 873 },
		{ 608, 553 },
		{ 376, 339 },
		{ 236, 201 },
		{ 447, 399 },
		{ 1215, 1215 },
		{ 778, 738 },
		{ 286, 246 },
		{ 908, 880 },
		{ 780, 740 },
		{ 845, 807 },
		{ 720, 672 },
		{ 333, 293 },
		{ 1873, 1852 },
		{ 1874, 1853 },
		{ 849, 813 },
		{ 571, 517 },
		{ 452, 404 },
		{ 515, 462 },
		{ 789, 750 },
		{ 574, 520 },
		{ 454, 406 },
		{ 996, 981 },
		{ 354, 316 },
		{ 998, 983 },
		{ 859, 826 },
		{ 675, 627 },
		{ 929, 904 },
		{ 678, 630 },
		{ 864, 832 },
		{ 456, 408 },
		{ 680, 632 },
		{ 934, 910 },
		{ 582, 528 },
		{ 340, 301 },
		{ 684, 635 },
		{ 458, 411 },
		{ 1015, 1006 },
		{ 368, 330 },
		{ 222, 189 },
		{ 941, 917 },
		{ 1022, 1013 },
		{ 2648, 2646 },
		{ 875, 843 },
		{ 1227, 1215 },
		{ 1906, 1889 },
		{ 266, 228 },
		{ 1908, 1891 },
		{ 1909, 1892 },
		{ 810, 771 },
		{ 946, 923 },
		{ 879, 847 },
		{ 591, 537 },
		{ 372, 334 },
		{ 1033, 1024 },
		{ 1034, 1025 },
		{ 359, 321 },
		{ 1039, 1030 },
		{ 469, 420 },
		{ 558, 503 },
		{ 531, 478 },
		{ 756, 715 },
		{ 958, 936 },
		{ 412, 374 },
		{ 1715, 1699 },
		{ 758, 717 },
		{ 648, 599 },
		{ 472, 422 },
		{ 373, 335 },
		{ 762, 721 },
		{ 307, 266 },
		{ 1531, 1528 },
		{ 1757, 1748 },
		{ 638, 585 },
		{ 1589, 1587 },
		{ 719, 671 },
		{ 1870, 1849 },
		{ 460, 413 },
		{ 1481, 1478 },
		{ 948, 925 },
		{ 871, 839 },
		{ 800, 761 },
		{ 1974, 1961 },
		{ 517, 464 },
		{ 1739, 1726 },
		{ 1505, 1502 },
		{ 1455, 1452 },
		{ 1427, 1425 },
		{ 626, 573 },
		{ 1680, 1661 },
		{ 1163, 1160 },
		{ 1932, 1917 },
		{ 833, 795 },
		{ 353, 315 },
		{ 981, 964 },
		{ 1009, 999 },
		{ 834, 795 },
		{ 982, 965 },
		{ 444, 396 },
		{ 826, 787 },
		{ 986, 969 },
		{ 987, 970 },
		{ 1051, 1044 },
		{ 741, 695 },
		{ 896, 868 },
		{ 391, 353 },
		{ 224, 191 },
		{ 344, 305 },
		{ 695, 646 },
		{ 968, 947 },
		{ 630, 577 },
		{ 396, 357 },
		{ 667, 620 },
		{ 750, 708 },
		{ 733, 685 },
		{ 305, 264 },
		{ 953, 930 },
		{ 623, 570 },
		{ 738, 692 },
		{ 803, 764 },
		{ 234, 200 },
		{ 816, 777 },
		{ 287, 247 },
		{ 235, 200 },
		{ 964, 942 },
		{ 514, 461 },
		{ 593, 539 },
		{ 1683, 1664 },
		{ 289, 249 },
		{ 752, 711 },
		{ 476, 425 },
		{ 700, 651 },
		{ 899, 871 },
		{ 360, 322 },
		{ 600, 545 },
		{ 703, 654 },
		{ 760, 719 },
		{ 1834, 1810 },
		{ 704, 655 },
		{ 1060, 1058 },
		{ 705, 656 },
		{ 1927, 1911 },
		{ 708, 659 },
		{ 1698, 1680 },
		{ 1063, 1062 },
		{ 380, 343 },
		{ 602, 547 },
		{ 1934, 1920 },
		{ 1936, 1922 },
		{ 1937, 1923 },
		{ 1938, 1924 },
		{ 1939, 1925 },
		{ 1068, 1067 },
		{ 767, 726 },
		{ 1070, 1069 },
		{ 838, 799 },
		{ 911, 883 },
		{ 769, 729 },
		{ 915, 888 },
		{ 325, 283 },
		{ 772, 732 },
		{ 712, 663 },
		{ 775, 735 },
		{ 382, 345 },
		{ 180, 161 },
		{ 653, 606 },
		{ 448, 400 },
		{ 410, 370 },
		{ 853, 817 },
		{ 781, 741 },
		{ 928, 903 },
		{ 200, 172 },
		{ 250, 214 },
		{ 490, 437 },
		{ 1646, 1622 },
		{ 858, 825 },
		{ 1732, 1718 },
		{ 312, 271 },
		{ 787, 748 },
		{ 861, 828 },
		{ 1875, 1854 },
		{ 414, 377 },
		{ 863, 831 },
		{ 1741, 1729 },
		{ 388, 350 },
		{ 251, 215 },
		{ 347, 308 },
		{ 669, 622 },
		{ 369, 331 },
		{ 223, 190 },
		{ 674, 626 },
		{ 429, 383 },
		{ 317, 275 },
		{ 1888, 1870 },
		{ 577, 523 },
		{ 801, 762 },
		{ 463, 417 },
		{ 737, 689 },
		{ 434, 385 },
		{ 196, 170 },
		{ 254, 218 },
		{ 586, 532 },
		{ 303, 262 },
		{ 276, 237 },
		{ 512, 459 },
		{ 637, 584 },
		{ 590, 536 },
		{ 1044, 1035 },
		{ 1422, 1422 },
		{ 1422, 1422 },
		{ 1767, 1767 },
		{ 1767, 1767 },
		{ 1918, 1918 },
		{ 1918, 1918 },
		{ 1978, 1978 },
		{ 1978, 1978 },
		{ 1367, 1367 },
		{ 1367, 1367 },
		{ 1358, 1358 },
		{ 1358, 1358 },
		{ 1370, 1370 },
		{ 1370, 1370 },
		{ 1318, 1318 },
		{ 1318, 1318 },
		{ 1397, 1397 },
		{ 1397, 1397 },
		{ 1361, 1361 },
		{ 1361, 1361 },
		{ 1781, 1781 },
		{ 1781, 1781 },
		{ 256, 220 },
		{ 1422, 1422 },
		{ 233, 199 },
		{ 1767, 1767 },
		{ 503, 449 },
		{ 1918, 1918 },
		{ 625, 572 },
		{ 1978, 1978 },
		{ 293, 253 },
		{ 1367, 1367 },
		{ 527, 474 },
		{ 1358, 1358 },
		{ 379, 342 },
		{ 1370, 1370 },
		{ 273, 234 },
		{ 1318, 1318 },
		{ 435, 386 },
		{ 1397, 1397 },
		{ 616, 564 },
		{ 1361, 1361 },
		{ 2026, 2024 },
		{ 1781, 1781 },
		{ 1388, 1388 },
		{ 1388, 1388 },
		{ 1355, 1355 },
		{ 1355, 1355 },
		{ 2001, 2001 },
		{ 2001, 2001 },
		{ 1750, 1750 },
		{ 1750, 1750 },
		{ 1423, 1422 },
		{ 686, 637 },
		{ 1768, 1767 },
		{ 599, 544 },
		{ 1919, 1918 },
		{ 519, 466 },
		{ 1979, 1978 },
		{ 942, 918 },
		{ 1368, 1367 },
		{ 288, 248 },
		{ 1359, 1358 },
		{ 814, 775 },
		{ 1371, 1370 },
		{ 500, 445 },
		{ 1319, 1318 },
		{ 1388, 1388 },
		{ 1398, 1397 },
		{ 1355, 1355 },
		{ 1362, 1361 },
		{ 2001, 2001 },
		{ 1782, 1781 },
		{ 1750, 1750 },
		{ 1752, 1752 },
		{ 1752, 1752 },
		{ 1364, 1364 },
		{ 1364, 1364 },
		{ 1947, 1947 },
		{ 1947, 1947 },
		{ 1949, 1949 },
		{ 1949, 1949 },
		{ 1951, 1951 },
		{ 1951, 1951 },
		{ 1953, 1953 },
		{ 1953, 1953 },
		{ 1955, 1955 },
		{ 1955, 1955 },
		{ 1957, 1957 },
		{ 1957, 1957 },
		{ 1727, 1727 },
		{ 1727, 1727 },
		{ 597, 543 },
		{ 598, 543 },
		{ 1762, 1762 },
		{ 1762, 1762 },
		{ 1389, 1388 },
		{ 1752, 1752 },
		{ 1356, 1355 },
		{ 1364, 1364 },
		{ 2002, 2001 },
		{ 1947, 1947 },
		{ 1751, 1750 },
		{ 1949, 1949 },
		{ 1067, 1066 },
		{ 1951, 1951 },
		{ 1036, 1027 },
		{ 1953, 1953 },
		{ 349, 310 },
		{ 1955, 1955 },
		{ 318, 276 },
		{ 1957, 1957 },
		{ 694, 645 },
		{ 1727, 1727 },
		{ 1972, 1972 },
		{ 1972, 1972 },
		{ 985, 968 },
		{ 1762, 1762 },
		{ 1433, 1432 },
		{ 794, 755 },
		{ 835, 796 },
		{ 966, 945 },
		{ 927, 902 },
		{ 670, 623 },
		{ 1896, 1879 },
		{ 1723, 1708 },
		{ 1753, 1752 },
		{ 1014, 1004 },
		{ 1365, 1364 },
		{ 774, 734 },
		{ 1948, 1947 },
		{ 1016, 1007 },
		{ 1950, 1949 },
		{ 721, 673 },
		{ 1952, 1951 },
		{ 1487, 1486 },
		{ 1954, 1953 },
		{ 1972, 1972 },
		{ 1956, 1955 },
		{ 840, 801 },
		{ 1958, 1957 },
		{ 1019, 1010 },
		{ 1728, 1727 },
		{ 912, 885 },
		{ 1537, 1536 },
		{ 1021, 1012 },
		{ 1763, 1762 },
		{ 1960, 1941 },
		{ 842, 803 },
		{ 798, 759 },
		{ 753, 712 },
		{ 736, 688 },
		{ 182, 163 },
		{ 1029, 1020 },
		{ 716, 667 },
		{ 1169, 1168 },
		{ 1349, 1331 },
		{ 1595, 1594 },
		{ 1461, 1460 },
		{ 1703, 1686 },
		{ 1001, 986 },
		{ 882, 850 },
		{ 1755, 1744 },
		{ 1003, 989 },
		{ 848, 812 },
		{ 1511, 1510 },
		{ 1973, 1972 },
		{ 1742, 1730 },
		{ 561, 507 },
		{ 1113, 1111 },
		{ 2543, 2541 },
		{ 1035, 1026 },
		{ 1828, 1806 },
		{ 1935, 1921 },
		{ 1840, 1817 },
		{ 280, 241 },
		{ 453, 405 },
		{ 208, 176 },
		{ 1642, 1618 },
		{ 1971, 1959 },
		{ 1761, 1754 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 1552, 1552 },
		{ 1552, 1552 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2156, 2156 },
		{ 2156, 2156 },
		{ 1608, 1608 },
		{ 1608, 1608 },
		{ 1585, 1585 },
		{ 1585, 1585 },
		{ 2539, 2539 },
		{ 2539, 2539 },
		{ 2363, 2363 },
		{ 2363, 2363 },
		{ 2396, 2396 },
		{ 2396, 2396 },
		{ 1158, 1158 },
		{ 1158, 1158 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 1028, 1019 },
		{ 2433, 2433 },
		{ 272, 233 },
		{ 1552, 1552 },
		{ 642, 591 },
		{ 2308, 2308 },
		{ 311, 270 },
		{ 2156, 2156 },
		{ 257, 221 },
		{ 1608, 1608 },
		{ 342, 303 },
		{ 1585, 1585 },
		{ 300, 259 },
		{ 2539, 2539 },
		{ 486, 433 },
		{ 2363, 2363 },
		{ 227, 194 },
		{ 2396, 2396 },
		{ 610, 555 },
		{ 1158, 1158 },
		{ 1556, 1554 },
		{ 2399, 2399 },
		{ 2607, 2605 },
		{ 1500, 1500 },
		{ 1500, 1500 },
		{ 1526, 1526 },
		{ 1526, 1526 },
		{ 2434, 2433 },
		{ 1038, 1029 },
		{ 1553, 1552 },
		{ 611, 556 },
		{ 2325, 2308 },
		{ 698, 649 },
		{ 2157, 2156 },
		{ 699, 650 },
		{ 1609, 1608 },
		{ 746, 702 },
		{ 1586, 1585 },
		{ 612, 557 },
		{ 2540, 2539 },
		{ 488, 435 },
		{ 2377, 2363 },
		{ 290, 250 },
		{ 2405, 2396 },
		{ 615, 563 },
		{ 1159, 1158 },
		{ 1500, 1500 },
		{ 2407, 2399 },
		{ 1526, 1526 },
		{ 578, 524 },
		{ 1109, 1109 },
		{ 1109, 1109 },
		{ 2644, 2644 },
		{ 2644, 2644 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2404, 2404 },
		{ 2404, 2404 },
		{ 1143, 1143 },
		{ 1143, 1143 },
		{ 2466, 2466 },
		{ 2466, 2466 },
		{ 2510, 2510 },
		{ 2510, 2510 },
		{ 2191, 2191 },
		{ 2191, 2191 },
		{ 2047, 2047 },
		{ 2047, 2047 },
		{ 2603, 2603 },
		{ 2603, 2603 },
		{ 2556, 2556 },
		{ 2556, 2556 },
		{ 1501, 1500 },
		{ 1109, 1109 },
		{ 1527, 1526 },
		{ 2644, 2644 },
		{ 804, 765 },
		{ 2317, 2317 },
		{ 579, 525 },
		{ 2404, 2404 },
		{ 706, 657 },
		{ 1143, 1143 },
		{ 754, 713 },
		{ 2466, 2466 },
		{ 707, 658 },
		{ 2510, 2510 },
		{ 1993, 1990 },
		{ 2191, 2191 },
		{ 291, 251 },
		{ 2047, 2047 },
		{ 867, 835 },
		{ 2603, 2603 },
		{ 2470, 2468 },
		{ 2556, 2556 },
		{ 581, 527 },
		{ 2223, 2223 },
		{ 2223, 2223 },
		{ 1450, 1450 },
		{ 1450, 1450 },
		{ 1110, 1109 },
		{ 332, 292 },
		{ 2645, 2644 },
		{ 583, 529 },
		{ 2334, 2317 },
		{ 366, 328 },
		{ 2412, 2404 },
		{ 521, 468 },
		{ 1144, 1143 },
		{ 815, 776 },
		{ 2467, 2466 },
		{ 666, 619 },
		{ 2511, 2510 },
		{ 440, 392 },
		{ 2192, 2191 },
		{ 668, 621 },
		{ 2048, 2047 },
		{ 348, 309 },
		{ 2604, 2603 },
		{ 2223, 2223 },
		{ 2557, 2556 },
		{ 1450, 1450 },
		{ 2560, 2558 },
		{ 2414, 2414 },
		{ 2414, 2414 },
		{ 2415, 2415 },
		{ 2415, 2415 },
		{ 2416, 2416 },
		{ 2416, 2416 },
		{ 2378, 2378 },
		{ 2378, 2378 },
		{ 2301, 2301 },
		{ 2301, 2301 },
		{ 2350, 2350 },
		{ 2350, 2350 },
		{ 2351, 2351 },
		{ 2351, 2351 },
		{ 2424, 2424 },
		{ 2424, 2424 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 1186, 1186 },
		{ 1186, 1186 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 2243, 2223 },
		{ 2414, 2414 },
		{ 1451, 1450 },
		{ 2415, 2415 },
		{ 878, 846 },
		{ 2416, 2416 },
		{ 1066, 1065 },
		{ 2378, 2378 },
		{ 409, 369 },
		{ 2301, 2301 },
		{ 768, 728 },
		{ 2350, 2350 },
		{ 554, 500 },
		{ 2351, 2351 },
		{ 770, 730 },
		{ 2424, 2424 },
		{ 304, 263 },
		{ 2261, 2261 },
		{ 239, 204 },
		{ 1186, 1186 },
		{ 249, 213 },
		{ 2226, 2226 },
		{ 676, 628 },
		{ 2245, 2245 },
		{ 2245, 2245 },
		{ 1476, 1476 },
		{ 1476, 1476 },
		{ 2419, 2414 },
		{ 677, 629 },
		{ 2420, 2415 },
		{ 352, 314 },
		{ 2421, 2416 },
		{ 950, 927 },
		{ 2388, 2378 },
		{ 830, 792 },
		{ 2318, 2301 },
		{ 181, 162 },
		{ 2364, 2350 },
		{ 891, 862 },
		{ 2365, 2351 },
		{ 295, 255 },
		{ 2426, 2424 },
		{ 504, 450 },
		{ 2262, 2261 },
		{ 212, 179 },
		{ 1187, 1186 },
		{ 2245, 2245 },
		{ 2246, 2226 },
		{ 1476, 1476 },
		{ 533, 480 },
		{ 2357, 2357 },
		{ 2357, 2357 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 837, 798 },
		{ 395, 356 },
		{ 898, 870 },
		{ 1023, 1014 },
		{ 451, 403 },
		{ 784, 745 },
		{ 841, 802 },
		{ 640, 587 },
		{ 1980, 1973 },
		{ 1337, 1319 },
		{ 1772, 1768 },
		{ 1759, 1751 },
		{ 1385, 1371 },
		{ 1965, 1948 },
		{ 1985, 1979 },
		{ 1760, 1753 },
		{ 1966, 1950 },
		{ 1383, 1365 },
		{ 2266, 2245 },
		{ 2357, 2357 },
		{ 1477, 1476 },
		{ 2389, 2389 },
		{ 1967, 1952 },
		{ 1380, 1356 },
		{ 1968, 1954 },
		{ 1382, 1362 },
		{ 1969, 1956 },
		{ 1384, 1368 },
		{ 1970, 1958 },
		{ 1740, 1728 },
		{ 1424, 1423 },
		{ 1933, 1919 },
		{ 1399, 1389 },
		{ 1783, 1782 },
		{ 1769, 1763 },
		{ 1405, 1398 },
		{ 1381, 1359 },
		{ 2004, 2002 },
		{ 1467, 1466 },
		{ 1411, 1406 },
		{ 1412, 1407 },
		{ 1542, 1541 },
		{ 1543, 1542 },
		{ 1438, 1437 },
		{ 1439, 1438 },
		{ 2371, 2357 },
		{ 1516, 1515 },
		{ 2398, 2389 },
		{ 1517, 1516 },
		{ 1492, 1491 },
		{ 1493, 1492 },
		{ 1600, 1599 },
		{ 1601, 1600 },
		{ 1174, 1173 },
		{ 1175, 1174 },
		{ 1415, 1411 },
		{ 1466, 1465 },
		{ 2675, 2674 },
		{ 2440, 2440 },
		{ 2437, 2440 },
		{ 150, 150 },
		{ 147, 150 },
		{ 1734, 1722 },
		{ 1735, 1722 },
		{ 1712, 1697 },
		{ 1713, 1697 },
		{ 1788, 1788 },
		{ 2014, 2014 },
		{ 2069, 2049 },
		{ 2439, 2435 },
		{ 155, 151 },
		{ 2019, 2015 },
		{ 2445, 2441 },
		{ 2068, 2049 },
		{ 2438, 2435 },
		{ 154, 151 },
		{ 2018, 2015 },
		{ 2444, 2441 },
		{ 2112, 2090 },
		{ 1793, 1789 },
		{ 80, 62 },
		{ 2440, 2440 },
		{ 149, 145 },
		{ 150, 150 },
		{ 1792, 1789 },
		{ 79, 62 },
		{ 1796, 1795 },
		{ 148, 145 },
		{ 2020, 2017 },
		{ 1788, 1788 },
		{ 2014, 2014 },
		{ 2022, 2021 },
		{ 1865, 1844 },
		{ 109, 94 },
		{ 156, 153 },
		{ 2446, 2443 },
		{ 2441, 2440 },
		{ 2448, 2447 },
		{ 151, 150 },
		{ 158, 157 },
		{ 1672, 1653 },
		{ 1794, 1791 },
		{ 2009, 2008 },
		{ 1863, 1842 },
		{ 1789, 1788 },
		{ 2015, 2014 },
		{ 1747, 1737 },
		{ 2021, 2019 },
		{ 1791, 1787 },
		{ 2443, 2439 },
		{ 153, 149 },
		{ 1653, 1631 },
		{ 94, 80 },
		{ 2447, 2445 },
		{ 2017, 2013 },
		{ 2090, 2069 },
		{ 1795, 1793 },
		{ 1844, 1822 },
		{ 2434, 2434 },
		{ 2434, 2434 },
		{ 0, 1118 },
		{ 0, 2608 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 0, 2547 },
		{ 2157, 2157 },
		{ 2157, 2157 },
		{ 0, 2612 },
		{ 2334, 2334 },
		{ 2334, 2334 },
		{ 0, 2494 },
		{ 2243, 2243 },
		{ 2243, 2243 },
		{ 0, 2616 },
		{ 1796, 1796 },
		{ 1797, 1796 },
		{ 2398, 2398 },
		{ 2398, 2398 },
		{ 0, 2619 },
		{ 2365, 2365 },
		{ 2365, 2365 },
		{ 2434, 2434 },
		{ 0, 1096 },
		{ 0, 2501 },
		{ 0, 2624 },
		{ 2262, 2262 },
		{ 2448, 2448 },
		{ 2449, 2448 },
		{ 2157, 2157 },
		{ 0, 2312 },
		{ 0, 2561 },
		{ 2334, 2334 },
		{ 0, 1576 },
		{ 0, 2369 },
		{ 2243, 2243 },
		{ 158, 158 },
		{ 159, 158 },
		{ 1796, 1796 },
		{ 0, 2565 },
		{ 2398, 2398 },
		{ 2405, 2405 },
		{ 2405, 2405 },
		{ 2365, 2365 },
		{ 0, 2456 },
		{ 0, 2406 },
		{ 0, 2635 },
		{ 2407, 2407 },
		{ 2407, 2407 },
		{ 0, 2036 },
		{ 2448, 2448 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 0, 2342 },
		{ 2412, 2412 },
		{ 2412, 2412 },
		{ 0, 2575 },
		{ 0, 1560 },
		{ 0, 2345 },
		{ 158, 158 },
		{ 0, 2578 },
		{ 2377, 2377 },
		{ 2377, 2377 },
		{ 0, 2522 },
		{ 2405, 2405 },
		{ 2318, 2318 },
		{ 2318, 2318 },
		{ 2022, 2022 },
		{ 2023, 2022 },
		{ 0, 2583 },
		{ 2407, 2407 },
		{ 0, 1134 },
		{ 0, 2471 },
		{ 0, 2654 },
		{ 2371, 2371 },
		{ 2419, 2419 },
		{ 2419, 2419 },
		{ 2412, 2412 },
		{ 2420, 2420 },
		{ 2420, 2420 },
		{ 2421, 2421 },
		{ 2421, 2421 },
		{ 0, 2275 },
		{ 0, 2476 },
		{ 2377, 2377 },
		{ 0, 1129 },
		{ 0, 2531 },
		{ 0, 1571 },
		{ 2318, 2318 },
		{ 0, 2594 },
		{ 2022, 2022 },
		{ 0, 2236 },
		{ 0, 2665 },
		{ 2426, 2426 },
		{ 2426, 2426 },
		{ 0, 2522 },
		{ 2325, 2325 },
		{ 2325, 2325 },
		{ 2419, 2419 },
		{ 0, 2280 },
		{ 0, 1150 },
		{ 2420, 2420 },
		{ 0, 2487 },
		{ 2421, 2421 },
		{ 2388, 2388 },
		{ 2388, 2388 },
		{ 0, 2205 },
		{ 1332, 1332 },
		{ 1075, 1075 },
		{ 1790, 1786 },
		{ 2442, 2438 },
		{ 2016, 2018 },
		{ 0, 79 },
		{ 1691, 1672 },
		{ 0, 2205 },
		{ 152, 148 },
		{ 2426, 2426 },
		{ 2442, 2444 },
		{ 0, 1630 },
		{ 2325, 2325 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2388, 2388 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1332, 1332 },
		{ 1075, 1075 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -60, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 42, 115, 0 },
		{ -164, 2350, 0 },
		{ 5, 0, 0 },
		{ -1074, 846, -31 },
		{ 7, 0, -31 },
		{ -1078, 1541, -33 },
		{ 9, 0, -33 },
		{ -1091, 2613, 102 },
		{ 11, 0, 102 },
		{ -1112, 2715, 110 },
		{ 13, 0, 110 },
		{ -1146, 2720, 0 },
		{ 15, 0, 0 },
		{ -1161, 2605, 98 },
		{ 17, 0, 98 },
		{ -1200, 229, 0 },
		{ 19, 0, 0 },
		{ -1426, 2718, 0 },
		{ 21, 0, 0 },
		{ -1453, 2606, 0 },
		{ 23, 0, 0 },
		{ -1479, 2611, 0 },
		{ 25, 0, 0 },
		{ -1503, 2612, 0 },
		{ 27, 0, 0 },
		{ -1529, 2607, 0 },
		{ 29, 0, 0 },
		{ -1555, 2722, 113 },
		{ 31, 0, 113 },
		{ -1588, 2719, 216 },
		{ 33, 0, 216 },
		{ 36, 127, 0 },
		{ -1624, 343, 0 },
		{ 38, 126, 0 },
		{ -1813, 1, 0 },
		{ -2025, 2713, 0 },
		{ 39, 0, 0 },
		{ 42, 14, 0 },
		{ -78, 457, 0 },
		{ -2451, 2618, 106 },
		{ 43, 0, 106 },
		{ -2469, 2721, 125 },
		{ 45, 0, 125 },
		{ 2513, 1362, 0 },
		{ 47, 0, 0 },
		{ -2515, 2716, 222 },
		{ 49, 0, 222 },
		{ -2542, 2723, 128 },
		{ 51, 0, 128 },
		{ -2559, 2717, 122 },
		{ 53, 0, 122 },
		{ -2606, 2714, 116 },
		{ 55, 0, 116 },
		{ -2647, 2724, 121 },
		{ 57, 0, 121 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 370 },
		{ 2447, 3985, 371 },
		{ 0, 0, 195 },
		{ 0, 0, 197 },
		{ 144, 827, 213 },
		{ 144, 1015, 213 },
		{ 144, 1095, 213 },
		{ 144, 1172, 213 },
		{ 144, 1177, 213 },
		{ 144, 1210, 213 },
		{ 144, 1233, 213 },
		{ 2651, 2382, 371 },
		{ 144, 1244, 213 },
		{ 144, 1245, 213 },
		{ 2651, 1355, 212 },
		{ 93, 2175, 371 },
		{ 144, 0, 213 },
		{ 0, 0, 371 },
		{ -79, 4168, 191 },
		{ -80, 4017, 0 },
		{ 144, 1236, 213 },
		{ 144, 1253, 213 },
		{ 144, 1235, 213 },
		{ 144, 1235, 213 },
		{ 144, 1242, 213 },
		{ 144, 656, 213 },
		{ 144, 650, 213 },
		{ 2673, 1923, 0 },
		{ 144, 646, 213 },
		{ 144, 641, 213 },
		{ 2651, 1510, 209 },
		{ 107, 1319, 0 },
		{ 2651, 1444, 210 },
		{ 2447, 3993, 0 },
		{ 144, 650, 213 },
		{ 144, 643, 213 },
		{ 144, 0, 201 },
		{ 144, 662, 213 },
		{ 144, 664, 213 },
		{ 144, 648, 213 },
		{ 144, 653, 213 },
		{ 2653, 2418, 0 },
		{ 144, 653, 213 },
		{ 144, 661, 213 },
		{ 120, 1353, 0 },
		{ 107, 0, 0 },
		{ 2132, 2208, 211 },
		{ 122, 1315, 0 },
		{ 0, 0, 193 },
		{ 144, 665, 198 },
		{ 144, 662, 213 },
		{ 144, 660, 213 },
		{ 144, 653, 213 },
		{ 144, 0, 204 },
		{ 144, 654, 213 },
		{ 0, 0, 206 },
		{ 144, 662, 213 },
		{ 144, 661, 213 },
		{ 120, 0, 0 },
		{ 2132, 2242, 209 },
		{ 122, 0, 0 },
		{ 2132, 2185, 210 },
		{ 144, 676, 213 },
		{ 144, 675, 213 },
		{ 144, 674, 213 },
		{ 144, 0, 203 },
		{ 144, 675, 213 },
		{ 144, 677, 213 },
		{ 144, 679, 213 },
		{ 144, 670, 213 },
		{ 144, 0, 200 },
		{ 144, 0, 202 },
		{ 144, 667, 213 },
		{ 144, 715, 213 },
		{ 144, 669, 213 },
		{ 144, 0, 199 },
		{ 144, 0, 205 },
		{ 144, 677, 213 },
		{ 144, 7, 213 },
		{ 144, 845, 213 },
		{ 0, 0, 208 },
		{ 144, 941, 213 },
		{ 144, 942, 213 },
		{ 2673, 1146, 207 },
		{ 2447, 3987, 371 },
		{ 150, 0, 195 },
		{ 0, 0, 196 },
		{ -148, 4171, 191 },
		{ -149, 4015, 0 },
		{ 2665, 3998, 0 },
		{ 2447, 3975, 0 },
		{ 0, 0, 192 },
		{ 2447, 3994, 0 },
		{ -154, 31, 0 },
		{ -155, 1, 0 },
		{ 158, 0, 193 },
		{ 2447, 3999, 0 },
		{ 2665, 4093, 0 },
		{ 0, 0, 194 },
		{ 2646, 1290, 96 },
		{ 1925, 3433, 96 },
		{ 2605, 3864, 96 },
		{ 1941, 3631, 96 },
		{ 0, 0, 96 },
		{ 2646, 2788, 0 },
		{ 1940, 2474, 0 },
		{ 1910, 3013, 0 },
		{ 1888, 2762, 0 },
		{ 1888, 2764, 0 },
		{ 1925, 3468, 0 },
		{ 2450, 2688, 0 },
		{ 1925, 3440, 0 },
		{ 1927, 2962, 0 },
		{ 2450, 2701, 0 },
		{ 1940, 2722, 0 },
		{ 2541, 3657, 0 },
		{ 2646, 2705, 0 },
		{ 1940, 3063, 0 },
		{ 2605, 3872, 0 },
		{ 1871, 2899, 0 },
		{ 1871, 2889, 0 },
		{ 1893, 2573, 0 },
		{ 1874, 3171, 0 },
		{ 1855, 3221, 0 },
		{ 1874, 3173, 0 },
		{ 1893, 2575, 0 },
		{ 1893, 2593, 0 },
		{ 2450, 2741, 0 },
		{ 2646, 3304, 0 },
		{ 1925, 3458, 0 },
		{ 1044, 3374, 0 },
		{ 1871, 2898, 0 },
		{ 1893, 2599, 0 },
		{ 2605, 3700, 0 },
		{ 1871, 2845, 0 },
		{ 1888, 3126, 0 },
		{ 1910, 3004, 0 },
		{ 1940, 3108, 0 },
		{ 2024, 3502, 0 },
		{ 2605, 3393, 0 },
		{ 2646, 3268, 0 },
		{ 1855, 3201, 0 },
		{ 1927, 2941, 0 },
		{ 2605, 3846, 0 },
		{ 1910, 3054, 0 },
		{ 2646, 3013, 0 },
		{ 1940, 3065, 0 },
		{ 1893, 2601, 0 },
		{ 1830, 2805, 0 },
		{ 1888, 3159, 0 },
		{ 1833, 2713, 0 },
		{ 1855, 3205, 0 },
		{ 2605, 3848, 0 },
		{ 1925, 3441, 0 },
		{ 1925, 3454, 0 },
		{ 1910, 3037, 0 },
		{ 1893, 2616, 0 },
		{ 1925, 3469, 0 },
		{ 1910, 3035, 0 },
		{ 2024, 3500, 0 },
		{ 2605, 3692, 0 },
		{ 1830, 2813, 0 },
		{ 1927, 2910, 0 },
		{ 1893, 2554, 0 },
		{ 1925, 2748, 0 },
		{ 1940, 2994, 0 },
		{ 1855, 3204, 0 },
		{ 2646, 3311, 0 },
		{ 1940, 3021, 0 },
		{ 677, 2676, 0 },
		{ 1910, 3038, 0 },
		{ 1888, 3112, 0 },
		{ 2605, 3686, 0 },
		{ 2024, 3514, 0 },
		{ 1893, 2555, 0 },
		{ 1830, 2797, 0 },
		{ 1925, 3472, 0 },
		{ 1833, 2703, 0 },
		{ 1927, 2929, 0 },
		{ 1893, 2557, 0 },
		{ 2541, 3655, 0 },
		{ 1888, 3164, 0 },
		{ 1830, 2762, 0 },
		{ 1961, 3020, 0 },
		{ 1893, 2558, 0 },
		{ 2646, 3272, 0 },
		{ 1925, 3391, 0 },
		{ 2024, 3539, 0 },
		{ 1925, 3397, 0 },
		{ 2605, 3726, 0 },
		{ 2605, 3772, 0 },
		{ 1855, 3218, 0 },
		{ 2024, 3508, 0 },
		{ 1893, 2559, 0 },
		{ 2605, 3868, 0 },
		{ 2646, 3252, 0 },
		{ 1855, 3228, 0 },
		{ 2646, 3255, 0 },
		{ 2605, 3696, 0 },
		{ 1871, 2844, 0 },
		{ 1927, 2958, 0 },
		{ 1925, 3471, 0 },
		{ 2605, 3844, 0 },
		{ 1044, 3383, 0 },
		{ 677, 2671, 0 },
		{ 1961, 3337, 0 },
		{ 1874, 3185, 0 },
		{ 1927, 2974, 0 },
		{ 1893, 2560, 0 },
		{ 2605, 3690, 0 },
		{ 1925, 3446, 0 },
		{ 1893, 2562, 0 },
		{ 0, 0, 32 },
		{ 1940, 2784, 0 },
		{ 1925, 3461, 0 },
		{ 1941, 3589, 0 },
		{ 1893, 2563, 0 },
		{ 1830, 2793, 0 },
		{ 1871, 2886, 0 },
		{ 1855, 3225, 0 },
		{ 1830, 2796, 0 },
		{ 1893, 2564, 0 },
		{ 1925, 3428, 0 },
		{ 1910, 3021, 0 },
		{ 1910, 3030, 0 },
		{ 1874, 3182, 0 },
		{ 1927, 2930, 0 },
		{ 0, 2687, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 1871, 2903, 0 },
		{ 2605, 3784, 0 },
		{ 2646, 3277, 0 },
		{ 1855, 3232, 0 },
		{ 1830, 2806, 0 },
		{ 1927, 2968, 0 },
		{ 0, 0, 30 },
		{ 1893, 2565, 0 },
		{ 1871, 2861, 0 },
		{ 1830, 2821, 0 },
		{ 2646, 3299, 0 },
		{ 1830, 2825, 0 },
		{ 2605, 3694, 0 },
		{ 1927, 2935, 0 },
		{ 1044, 3375, 0 },
		{ 1871, 2891, 0 },
		{ 1888, 3122, 0 },
		{ 1925, 3455, 0 },
		{ 2605, 3800, 0 },
		{ 1941, 3587, 0 },
		{ 1927, 2942, 0 },
		{ 0, 0, 26 },
		{ 1927, 2953, 0 },
		{ 2605, 3858, 0 },
		{ 1044, 3361, 0 },
		{ 2646, 3288, 0 },
		{ 0, 0, 35 },
		{ 1830, 2760, 0 },
		{ 1940, 2992, 0 },
		{ 1893, 2567, 0 },
		{ 2646, 3321, 0 },
		{ 1925, 3402, 0 },
		{ 1893, 2568, 0 },
		{ 1871, 2905, 0 },
		{ 1888, 3161, 0 },
		{ 1830, 2772, 0 },
		{ 1927, 2925, 0 },
		{ 2605, 3788, 0 },
		{ 1893, 2569, 0 },
		{ 2646, 3303, 0 },
		{ 1925, 3457, 0 },
		{ 1830, 2786, 0 },
		{ 1927, 2931, 0 },
		{ 2646, 3318, 0 },
		{ 721, 3334, 0 },
		{ 0, 0, 8 },
		{ 1871, 2873, 0 },
		{ 1874, 3172, 0 },
		{ 2646, 3267, 0 },
		{ 1906, 2668, 0 },
		{ 1893, 2570, 0 },
		{ 2024, 3512, 0 },
		{ 1925, 3414, 0 },
		{ 1871, 2890, 0 },
		{ 1925, 3432, 0 },
		{ 1874, 3190, 0 },
		{ 1893, 2572, 0 },
		{ 1830, 2800, 0 },
		{ 2450, 2692, 0 },
		{ 1925, 3453, 0 },
		{ 2450, 2748, 0 },
		{ 1927, 2988, 0 },
		{ 1044, 3373, 0 },
		{ 1888, 3113, 0 },
		{ 1940, 2532, 0 },
		{ 2605, 3883, 0 },
		{ 1044, 3379, 0 },
		{ 1941, 2680, 0 },
		{ 1906, 2654, 0 },
		{ 1874, 3177, 0 },
		{ 1871, 2851, 0 },
		{ 1830, 2837, 0 },
		{ 0, 0, 76 },
		{ 1893, 2574, 0 },
		{ 1940, 3077, 0 },
		{ 1832, 2684, 0 },
		{ 1910, 3044, 0 },
		{ 1888, 3133, 0 },
		{ 2605, 3836, 0 },
		{ 1925, 3436, 0 },
		{ 0, 0, 7 },
		{ 1874, 3179, 0 },
		{ 0, 0, 6 },
		{ 2646, 3328, 0 },
		{ 0, 0, 81 },
		{ 1888, 3140, 0 },
		{ 1925, 3450, 0 },
		{ 2646, 1348, 0 },
		{ 1893, 2576, 0 },
		{ 1888, 3163, 0 },
		{ 1910, 3025, 0 },
		{ 1893, 2577, 0 },
		{ 1925, 3460, 0 },
		{ 2024, 2674, 0 },
		{ 1925, 3467, 0 },
		{ 2024, 3516, 0 },
		{ 1940, 3070, 0 },
		{ 0, 0, 31 },
		{ 1855, 3227, 0 },
		{ 1893, 2578, 68 },
		{ 1893, 2579, 69 },
		{ 2605, 3796, 0 },
		{ 1927, 2978, 0 },
		{ 1910, 3048, 0 },
		{ 1927, 2985, 0 },
		{ 1044, 3366, 0 },
		{ 2450, 2743, 0 },
		{ 1940, 3105, 0 },
		{ 2646, 3269, 0 },
		{ 1925, 3435, 0 },
		{ 1893, 2580, 0 },
		{ 1830, 2802, 0 },
		{ 2605, 3886, 0 },
		{ 2646, 3282, 0 },
		{ 2541, 3656, 0 },
		{ 2646, 3286, 0 },
		{ 1927, 2928, 0 },
		{ 2646, 3295, 0 },
		{ 0, 0, 9 },
		{ 1893, 2581, 0 },
		{ 2646, 3301, 0 },
		{ 1906, 2664, 0 },
		{ 1961, 3344, 0 },
		{ 0, 0, 66 },
		{ 1871, 2865, 0 },
		{ 1888, 3144, 0 },
		{ 1925, 3465, 0 },
		{ 1940, 3018, 0 },
		{ 1940, 2677, 0 },
		{ 2646, 3323, 0 },
		{ 2605, 2717, 0 },
		{ 2646, 3332, 0 },
		{ 2450, 2702, 0 },
		{ 1910, 3026, 0 },
		{ 1925, 3399, 0 },
		{ 1830, 2828, 0 },
		{ 1927, 2957, 0 },
		{ 2450, 2695, 0 },
		{ 1888, 3131, 0 },
		{ 1830, 2839, 0 },
		{ 2646, 3254, 0 },
		{ 1830, 2759, 0 },
		{ 2605, 3698, 0 },
		{ 1893, 2582, 0 },
		{ 2605, 3724, 0 },
		{ 1927, 2975, 0 },
		{ 1925, 3442, 0 },
		{ 1940, 2998, 0 },
		{ 1940, 3000, 0 },
		{ 1855, 3194, 0 },
		{ 1893, 2583, 56 },
		{ 1927, 2980, 0 },
		{ 1893, 2584, 0 },
		{ 1893, 2585, 0 },
		{ 2024, 3543, 0 },
		{ 1893, 2586, 0 },
		{ 1871, 2852, 0 },
		{ 0, 0, 65 },
		{ 2024, 3504, 0 },
		{ 2605, 3870, 0 },
		{ 1830, 2774, 0 },
		{ 1830, 2785, 0 },
		{ 0, 0, 78 },
		{ 0, 0, 80 },
		{ 1940, 3072, 0 },
		{ 1871, 2870, 0 },
		{ 1925, 2772, 0 },
		{ 1830, 2787, 0 },
		{ 1925, 3473, 0 },
		{ 1893, 2587, 0 },
		{ 1925, 3394, 0 },
		{ 2646, 3283, 0 },
		{ 1888, 3165, 0 },
		{ 1961, 3350, 0 },
		{ 2450, 2753, 0 },
		{ 2024, 3535, 0 },
		{ 1893, 2588, 0 },
		{ 2605, 3790, 0 },
		{ 1871, 2894, 0 },
		{ 840, 3247, 0 },
		{ 1830, 2798, 0 },
		{ 1888, 3127, 0 },
		{ 1940, 3102, 0 },
		{ 2024, 3510, 0 },
		{ 1830, 2799, 0 },
		{ 1833, 2705, 0 },
		{ 1893, 2589, 0 },
		{ 2646, 3325, 0 },
		{ 1871, 2907, 0 },
		{ 2605, 3877, 0 },
		{ 1830, 2804, 0 },
		{ 1940, 3071, 0 },
		{ 1906, 2663, 0 },
		{ 1927, 2983, 0 },
		{ 1940, 3092, 0 },
		{ 1874, 3188, 0 },
		{ 1961, 2721, 0 },
		{ 1893, 2590, 0 },
		{ 0, 0, 27 },
		{ 1893, 2591, 0 },
		{ 1910, 3046, 0 },
		{ 1927, 2913, 0 },
		{ 1910, 3051, 0 },
		{ 1927, 2914, 0 },
		{ 1893, 2592, 70 },
		{ 1940, 3074, 0 },
		{ 1874, 3189, 0 },
		{ 1871, 2867, 0 },
		{ 1871, 2868, 0 },
		{ 2605, 3840, 0 },
		{ 1910, 3005, 0 },
		{ 2450, 2733, 0 },
		{ 2646, 3324, 0 },
		{ 1830, 2830, 0 },
		{ 1871, 2875, 0 },
		{ 0, 0, 82 },
		{ 2541, 3648, 0 },
		{ 1874, 3180, 0 },
		{ 1830, 2831, 0 },
		{ 1888, 3124, 0 },
		{ 0, 0, 77 },
		{ 0, 0, 67 },
		{ 1871, 2887, 0 },
		{ 1927, 2952, 0 },
		{ 1830, 2836, 0 },
		{ 1940, 2464, 0 },
		{ 2646, 3281, 0 },
		{ 1888, 3139, 0 },
		{ 1893, 2594, 0 },
		{ 2646, 3285, 0 },
		{ 1855, 3223, 0 },
		{ 1910, 3023, 0 },
		{ 1925, 3463, 0 },
		{ 2605, 3733, 0 },
		{ 2605, 3762, 0 },
		{ 1871, 2893, 0 },
		{ 2605, 3778, 0 },
		{ 2646, 3298, 0 },
		{ 2605, 3786, 0 },
		{ 1927, 2964, 0 },
		{ 1888, 3162, 0 },
		{ 1925, 3470, 0 },
		{ 1927, 2965, 0 },
		{ 1893, 2595, 0 },
		{ 1927, 2969, 0 },
		{ 1925, 3475, 0 },
		{ 2646, 3317, 0 },
		{ 1927, 2970, 0 },
		{ 1925, 3395, 0 },
		{ 1871, 2896, 0 },
		{ 1888, 3120, 0 },
		{ 1893, 2596, 0 },
		{ 2541, 3572, 0 },
		{ 2024, 3533, 0 },
		{ 1925, 3403, 0 },
		{ 1874, 3174, 0 },
		{ 1925, 3415, 0 },
		{ 1874, 3175, 0 },
		{ 1940, 2996, 0 },
		{ 1910, 3050, 0 },
		{ 0, 0, 58 },
		{ 2646, 3263, 0 },
		{ 2646, 3266, 0 },
		{ 1893, 2597, 0 },
		{ 2605, 3702, 0 },
		{ 2605, 3714, 0 },
		{ 2605, 3722, 0 },
		{ 1874, 3181, 0 },
		{ 1871, 2901, 0 },
		{ 0, 0, 85 },
		{ 0, 0, 79 },
		{ 0, 0, 83 },
		{ 2605, 3728, 0 },
		{ 2024, 3518, 0 },
		{ 677, 2679, 0 },
		{ 1893, 2598, 0 },
		{ 2646, 2539, 0 },
		{ 1927, 2909, 0 },
		{ 1874, 3168, 0 },
		{ 1044, 3385, 0 },
		{ 1888, 3150, 0 },
		{ 2024, 3506, 0 },
		{ 1961, 3355, 0 },
		{ 1888, 3157, 0 },
		{ 2450, 2750, 0 },
		{ 1855, 3237, 0 },
		{ 1044, 3378, 0 },
		{ 1830, 2778, 0 },
		{ 1855, 3200, 0 },
		{ 1871, 2847, 0 },
		{ 1893, 2600, 0 },
		{ 1874, 3178, 0 },
		{ 1855, 3215, 0 },
		{ 1925, 3474, 0 },
		{ 1961, 3340, 0 },
		{ 1927, 2926, 0 },
		{ 2605, 3889, 0 },
		{ 1940, 3083, 0 },
		{ 0, 0, 20 },
		{ 0, 0, 21 },
		{ 2605, 3688, 0 },
		{ 0, 0, 29 },
		{ 0, 0, 74 },
		{ 1833, 2710, 0 },
		{ 2450, 2735, 0 },
		{ 1871, 2863, 0 },
		{ 2450, 2739, 0 },
		{ 1830, 2794, 0 },
		{ 2646, 3331, 0 },
		{ 1927, 2936, 0 },
		{ 0, 0, 60 },
		{ 1927, 2938, 0 },
		{ 0, 0, 62 },
		{ 1910, 3024, 0 },
		{ 1927, 2939, 0 },
		{ 1925, 3434, 0 },
		{ 1906, 2658, 0 },
		{ 1906, 2661, 0 },
		{ 1961, 3117, 0 },
		{ 1927, 2943, 0 },
		{ 840, 3243, 0 },
		{ 1855, 3219, 0 },
		{ 0, 0, 75 },
		{ 0, 0, 84 },
		{ 1927, 2945, 0 },
		{ 0, 0, 95 },
		{ 1871, 2872, 0 },
		{ 2024, 3262, 0 },
		{ 2605, 3794, 0 },
		{ 1044, 3380, 0 },
		{ 2605, 3798, 0 },
		{ 1925, 3456, 0 },
		{ 1941, 3602, 0 },
		{ 1893, 2535, 0 },
		{ 1893, 2605, 0 },
		{ 1925, 3459, 0 },
		{ 2646, 3291, 0 },
		{ 2605, 3850, 0 },
		{ 2605, 3856, 0 },
		{ 2646, 3293, 0 },
		{ 1940, 3060, 0 },
		{ 2646, 3296, 0 },
		{ 1940, 3061, 0 },
		{ 1940, 3019, 0 },
		{ 2646, 3300, 0 },
		{ 1893, 2606, 0 },
		{ 2024, 3531, 0 },
		{ 1893, 2607, 0 },
		{ 1893, 2611, 0 },
		{ 1874, 3169, 0 },
		{ 1910, 3055, 0 },
		{ 1855, 3211, 0 },
		{ 1893, 2612, 0 },
		{ 1910, 3022, 0 },
		{ 1941, 3591, 0 },
		{ 1044, 3376, 0 },
		{ 1940, 3082, 0 },
		{ 1927, 2966, 0 },
		{ 2605, 3716, 0 },
		{ 2605, 3718, 0 },
		{ 1925, 3400, 0 },
		{ 1874, 3176, 0 },
		{ 1927, 2967, 0 },
		{ 1925, 3404, 0 },
		{ 1925, 3407, 0 },
		{ 1925, 3409, 0 },
		{ 2605, 3764, 0 },
		{ 2605, 3768, 0 },
		{ 1925, 3411, 0 },
		{ 1893, 2613, 0 },
		{ 1830, 2807, 0 },
		{ 1830, 2812, 0 },
		{ 1925, 3430, 0 },
		{ 1855, 3235, 0 },
		{ 2450, 2742, 0 },
		{ 1855, 3240, 0 },
		{ 1941, 3633, 0 },
		{ 1830, 2815, 0 },
		{ 0, 0, 24 },
		{ 1830, 2816, 0 },
		{ 1961, 3342, 0 },
		{ 2646, 3276, 0 },
		{ 1941, 3612, 0 },
		{ 1888, 3160, 0 },
		{ 1830, 2817, 0 },
		{ 1830, 2820, 0 },
		{ 0, 2673, 0 },
		{ 1855, 3214, 0 },
		{ 1927, 2987, 0 },
		{ 1906, 2660, 0 },
		{ 1833, 2708, 0 },
		{ 1833, 2709, 0 },
		{ 1910, 3016, 0 },
		{ 1871, 2846, 0 },
		{ 1044, 3382, 0 },
		{ 2450, 2715, 0 },
		{ 1927, 2915, 0 },
		{ 1941, 3630, 0 },
		{ 1925, 3466, 0 },
		{ 0, 0, 25 },
		{ 0, 0, 23 },
		{ 1044, 3386, 0 },
		{ 1855, 3230, 0 },
		{ 1927, 2920, 0 },
		{ 1044, 3371, 0 },
		{ 1927, 2923, 0 },
		{ 0, 0, 71 },
		{ 1830, 2832, 0 },
		{ 1830, 2833, 0 },
		{ 1927, 2927, 0 },
		{ 0, 0, 64 },
		{ 2605, 3720, 0 },
		{ 0, 0, 72 },
		{ 0, 0, 73 },
		{ 1888, 3143, 0 },
		{ 840, 3244, 0 },
		{ 1874, 3186, 0 },
		{ 1044, 3381, 0 },
		{ 1830, 2834, 0 },
		{ 0, 0, 3 },
		{ 1925, 3398, 0 },
		{ 1941, 3629, 0 },
		{ 2605, 3766, 0 },
		{ 1888, 3149, 0 },
		{ 2646, 3326, 0 },
		{ 1830, 2835, 0 },
		{ 2646, 3330, 0 },
		{ 1893, 2614, 0 },
		{ 1925, 3405, 0 },
		{ 1888, 3158, 0 },
		{ 0, 3335, 0 },
		{ 1925, 2545, 0 },
		{ 2646, 3262, 0 },
		{ 1940, 3075, 0 },
		{ 0, 0, 33 },
		{ 1925, 3422, 0 },
		{ 0, 0, 41 },
		{ 2605, 3838, 0 },
		{ 1925, 3426, 0 },
		{ 2605, 3842, 0 },
		{ 1830, 2838, 0 },
		{ 1925, 3429, 0 },
		{ 2450, 2738, 0 },
		{ 1941, 3608, 0 },
		{ 1925, 3431, 0 },
		{ 1855, 3224, 0 },
		{ 1830, 2841, 0 },
		{ 2646, 3271, 0 },
		{ 1940, 3085, 0 },
		{ 2646, 3274, 0 },
		{ 1925, 3438, 0 },
		{ 0, 0, 28 },
		{ 1940, 3087, 0 },
		{ 1940, 3089, 0 },
		{ 2605, 3887, 0 },
		{ 1830, 2842, 0 },
		{ 1940, 3099, 0 },
		{ 1925, 3447, 0 },
		{ 1893, 2617, 0 },
		{ 2646, 3284, 0 },
		{ 1910, 3009, 0 },
		{ 1874, 3183, 0 },
		{ 1855, 3195, 0 },
		{ 1871, 2880, 0 },
		{ 1941, 3598, 0 },
		{ 1871, 2885, 0 },
		{ 1893, 2618, 0 },
		{ 1940, 3062, 0 },
		{ 1941, 3628, 0 },
		{ 1888, 3137, 0 },
		{ 1961, 3348, 0 },
		{ 1925, 3464, 0 },
		{ 1927, 2947, 0 },
		{ 1044, 3387, 0 },
		{ 2605, 3760, 0 },
		{ 1927, 2951, 0 },
		{ 1893, 2620, 0 },
		{ 1830, 2764, 0 },
		{ 1927, 2954, 0 },
		{ 1855, 3222, 0 },
		{ 2646, 3314, 0 },
		{ 1927, 2955, 0 },
		{ 1888, 3151, 0 },
		{ 1830, 2765, 0 },
		{ 2024, 3541, 0 },
		{ 2605, 3792, 0 },
		{ 1925, 3390, 0 },
		{ 1855, 3226, 0 },
		{ 1830, 2766, 0 },
		{ 1910, 3039, 0 },
		{ 1927, 2959, 0 },
		{ 1927, 2960, 0 },
		{ 1927, 2961, 0 },
		{ 1830, 2767, 0 },
		{ 1940, 3091, 0 },
		{ 1874, 3184, 0 },
		{ 1044, 3367, 0 },
		{ 1830, 2768, 0 },
		{ 1855, 3196, 0 },
		{ 1855, 3199, 0 },
		{ 0, 0, 10 },
		{ 2605, 3862, 0 },
		{ 1871, 2895, 0 },
		{ 1830, 2769, 0 },
		{ 2605, 3365, 0 },
		{ 1941, 3599, 0 },
		{ 1888, 3114, 0 },
		{ 2605, 3882, 0 },
		{ 1925, 3424, 0 },
		{ 1893, 2621, 0 },
		{ 1941, 3618, 0 },
		{ 2605, 3888, 0 },
		{ 1941, 3627, 0 },
		{ 1855, 3207, 0 },
		{ 0, 0, 42 },
		{ 1940, 3106, 0 },
		{ 2646, 3275, 0 },
		{ 0, 0, 40 },
		{ 2450, 2745, 0 },
		{ 1874, 3170, 0 },
		{ 0, 0, 43 },
		{ 1941, 3643, 0 },
		{ 2646, 3280, 0 },
		{ 2450, 2746, 0 },
		{ 1871, 2902, 0 },
		{ 1927, 2971, 0 },
		{ 1925, 3437, 0 },
		{ 1893, 2622, 0 },
		{ 1830, 2775, 0 },
		{ 0, 0, 22 },
		{ 0, 0, 59 },
		{ 0, 0, 61 },
		{ 1940, 3068, 0 },
		{ 1927, 2976, 0 },
		{ 1925, 3444, 0 },
		{ 2646, 3290, 0 },
		{ 1910, 3027, 0 },
		{ 1925, 3448, 0 },
		{ 0, 0, 93 },
		{ 1927, 2977, 0 },
		{ 1925, 3451, 0 },
		{ 2646, 3294, 0 },
		{ 1830, 2776, 0 },
		{ 1927, 2979, 0 },
		{ 2605, 3774, 0 },
		{ 1893, 2623, 0 },
		{ 1855, 3233, 0 },
		{ 1927, 2982, 0 },
		{ 1961, 3347, 0 },
		{ 0, 3246, 0 },
		{ 1830, 2781, 0 },
		{ 1830, 2782, 0 },
		{ 2646, 3308, 0 },
		{ 1910, 3047, 0 },
		{ 1940, 3088, 0 },
		{ 2605, 3832, 0 },
		{ 2646, 3316, 0 },
		{ 1830, 2783, 0 },
		{ 1940, 3090, 0 },
		{ 1941, 3640, 0 },
		{ 1871, 2848, 0 },
		{ 1871, 2849, 0 },
		{ 0, 0, 86 },
		{ 1871, 2850, 0 },
		{ 0, 0, 88 },
		{ 1927, 2912, 0 },
		{ 0, 0, 57 },
		{ 1893, 2624, 0 },
		{ 1855, 3208, 0 },
		{ 1855, 3209, 0 },
		{ 1893, 2625, 0 },
		{ 2605, 3866, 0 },
		{ 1871, 2860, 0 },
		{ 1940, 3058, 0 },
		{ 2646, 3261, 0 },
		{ 0, 0, 38 },
		{ 1855, 3216, 0 },
		{ 1044, 3372, 0 },
		{ 1855, 3217, 0 },
		{ 2605, 3884, 0 },
		{ 1925, 3401, 0 },
		{ 2646, 3264, 0 },
		{ 2646, 3265, 0 },
		{ 1893, 2626, 0 },
		{ 1871, 2862, 0 },
		{ 1830, 2788, 0 },
		{ 1888, 3132, 0 },
		{ 1830, 2789, 0 },
		{ 1871, 2866, 0 },
		{ 2646, 3273, 0 },
		{ 1888, 3138, 0 },
		{ 1830, 2790, 0 },
		{ 1925, 3425, 0 },
		{ 0, 0, 46 },
		{ 1941, 3622, 0 },
		{ 0, 0, 63 },
		{ 1941, 3259, 0 },
		{ 1925, 3427, 0 },
		{ 0, 0, 91 },
		{ 1830, 2791, 0 },
		{ 1830, 2792, 0 },
		{ 1893, 2627, 19 },
		{ 1888, 3147, 0 },
		{ 1940, 3073, 0 },
		{ 1855, 3234, 0 },
		{ 2450, 2736, 0 },
		{ 0, 0, 47 },
		{ 1830, 2795, 0 },
		{ 1888, 3152, 0 },
		{ 1888, 3154, 0 },
		{ 1940, 3076, 0 },
		{ 1941, 3601, 0 },
		{ 1925, 3439, 0 },
		{ 2646, 3292, 0 },
		{ 1893, 2629, 0 },
		{ 1940, 3078, 0 },
		{ 1940, 3079, 0 },
		{ 1940, 3080, 0 },
		{ 0, 0, 51 },
		{ 2646, 3297, 0 },
		{ 1871, 2884, 0 },
		{ 1927, 2940, 0 },
		{ 1893, 2630, 0 },
		{ 2450, 2740, 0 },
		{ 1893, 2534, 0 },
		{ 1910, 3052, 0 },
		{ 2646, 3305, 0 },
		{ 2024, 3537, 0 },
		{ 1871, 2888, 0 },
		{ 1888, 3117, 0 },
		{ 0, 0, 53 },
		{ 1888, 3119, 0 },
		{ 2646, 3315, 0 },
		{ 1893, 2537, 0 },
		{ 1961, 3346, 0 },
		{ 1888, 3121, 0 },
		{ 2605, 3860, 0 },
		{ 1927, 2948, 0 },
		{ 1940, 3097, 0 },
		{ 1044, 3384, 0 },
		{ 1927, 2949, 0 },
		{ 1927, 2950, 0 },
		{ 1888, 3130, 0 },
		{ 1940, 3104, 0 },
		{ 0, 0, 18 },
		{ 2646, 3327, 0 },
		{ 1830, 2801, 0 },
		{ 1893, 2538, 0 },
		{ 1830, 2803, 0 },
		{ 1855, 3229, 0 },
		{ 1940, 3057, 0 },
		{ 1925, 3393, 0 },
		{ 0, 0, 36 },
		{ 1893, 2542, 0 },
		{ 1941, 3600, 0 },
		{ 1940, 3059, 0 },
		{ 1044, 3377, 0 },
		{ 1893, 2543, 17 },
		{ 1893, 2544, 0 },
		{ 1855, 3236, 0 },
		{ 0, 0, 37 },
		{ 1888, 3145, 0 },
		{ 2450, 2712, 0 },
		{ 0, 0, 44 },
		{ 0, 0, 45 },
		{ 0, 0, 15 },
		{ 1888, 3148, 0 },
		{ 1910, 3032, 0 },
		{ 1910, 3034, 0 },
		{ 1830, 2808, 0 },
		{ 1910, 3036, 0 },
		{ 0, 0, 92 },
		{ 1888, 3153, 0 },
		{ 1044, 3362, 0 },
		{ 1044, 3365, 0 },
		{ 1830, 2809, 0 },
		{ 1888, 3155, 0 },
		{ 1941, 3595, 0 },
		{ 1044, 3368, 0 },
		{ 1044, 3369, 0 },
		{ 1855, 3206, 0 },
		{ 0, 0, 34 },
		{ 1888, 3156, 0 },
		{ 1830, 2810, 0 },
		{ 0, 0, 52 },
		{ 1830, 2811, 0 },
		{ 1855, 3210, 0 },
		{ 1910, 3042, 0 },
		{ 1855, 3212, 0 },
		{ 1871, 2904, 0 },
		{ 2646, 3287, 0 },
		{ 2450, 2751, 0 },
		{ 2646, 3289, 0 },
		{ 1893, 2546, 0 },
		{ 1830, 2814, 0 },
		{ 1941, 3639, 0 },
		{ 2450, 2754, 0 },
		{ 0, 0, 48 },
		{ 1941, 3642, 0 },
		{ 1893, 2547, 0 },
		{ 0, 0, 87 },
		{ 0, 0, 89 },
		{ 1855, 3220, 0 },
		{ 0, 0, 94 },
		{ 0, 0, 11 },
		{ 1888, 3166, 0 },
		{ 1888, 3111, 0 },
		{ 1940, 3081, 0 },
		{ 1044, 3363, 0 },
		{ 1893, 2550, 0 },
		{ 1830, 2818, 0 },
		{ 1888, 3115, 0 },
		{ 1830, 2819, 0 },
		{ 1941, 3606, 0 },
		{ 0, 0, 90 },
		{ 2646, 3302, 0 },
		{ 1941, 3610, 0 },
		{ 1888, 3118, 0 },
		{ 2450, 2734, 0 },
		{ 1941, 3620, 0 },
		{ 1893, 2551, 0 },
		{ 1941, 3624, 0 },
		{ 2646, 3306, 0 },
		{ 2605, 3885, 0 },
		{ 1830, 2823, 0 },
		{ 1830, 2824, 0 },
		{ 1893, 2552, 0 },
		{ 1888, 3125, 0 },
		{ 2605, 3684, 0 },
		{ 1941, 3632, 0 },
		{ 1830, 2826, 0 },
		{ 1940, 3093, 0 },
		{ 1855, 3238, 0 },
		{ 2646, 3319, 0 },
		{ 2646, 3320, 0 },
		{ 2541, 3651, 0 },
		{ 1941, 3585, 0 },
		{ 1855, 3239, 0 },
		{ 2605, 3712, 0 },
		{ 2646, 3322, 0 },
		{ 1888, 3129, 0 },
		{ 1855, 3241, 0 },
		{ 1940, 3095, 0 },
		{ 1940, 3096, 0 },
		{ 1925, 3476, 0 },
		{ 1830, 2827, 0 },
		{ 1855, 3197, 0 },
		{ 1855, 3198, 0 },
		{ 1940, 3098, 0 },
		{ 0, 0, 39 },
		{ 0, 0, 54 },
		{ 1888, 3134, 0 },
		{ 1888, 3135, 0 },
		{ 0, 3370, 0 },
		{ 2646, 3260, 0 },
		{ 0, 0, 49 },
		{ 1855, 3203, 0 },
		{ 1888, 3136, 0 },
		{ 1871, 2864, 0 },
		{ 0, 0, 12 },
		{ 1940, 3100, 0 },
		{ 1940, 3101, 0 },
		{ 0, 0, 50 },
		{ 0, 0, 16 },
		{ 0, 0, 55 },
		{ 1927, 2984, 0 },
		{ 1888, 3142, 0 },
		{ 1925, 3408, 0 },
		{ 0, 0, 14 },
		{ 1893, 2553, 0 },
		{ 1927, 2986, 0 },
		{ 1925, 3413, 0 },
		{ 1910, 3033, 0 },
		{ 1855, 3213, 0 },
		{ 2605, 3834, 0 },
		{ 1941, 3583, 0 },
		{ 1925, 3421, 0 },
		{ 1874, 3187, 0 },
		{ 1925, 3423, 0 },
		{ 1888, 3146, 0 },
		{ 1830, 2829, 0 },
		{ 0, 0, 13 },
		{ 2673, 1221, 183 },
		{ 0, 0, 184 },
		{ 2032, 4165, 185 },
		{ 2651, 1333, 189 },
		{ 1081, 2174, 190 },
		{ 0, 0, 190 },
		{ 2651, 1411, 186 },
		{ 1084, 1348, 0 },
		{ 2651, 1422, 187 },
		{ 1087, 1377, 0 },
		{ 1084, 0, 0 },
		{ 2132, 2232, 188 },
		{ 1089, 1383, 0 },
		{ 1087, 0, 0 },
		{ 2132, 2252, 186 },
		{ 1089, 0, 0 },
		{ 2132, 2173, 187 },
		{ 2450, 2737, 103 },
		{ 0, 0, 103 },
		{ 0, 0, 104 },
		{ 2649, 1688, 0 },
		{ 2651, 2330, 0 },
		{ 2629, 1710, 0 },
		{ 1097, 3989, 0 },
		{ 2665, 2155, 0 },
		{ 2651, 2380, 0 },
		{ 2669, 2423, 0 },
		{ 2658, 2053, 0 },
		{ 2660, 2507, 0 },
		{ 2629, 1738, 0 },
		{ 2660, 2460, 0 },
		{ 2670, 1564, 0 },
		{ 2671, 2215, 0 },
		{ 2672, 1784, 0 },
		{ 2673, 1902, 0 },
		{ 2649, 1687, 0 },
		{ 2674, 3790, 0 },
		{ 0, 0, 101 },
		{ 2541, 3649, 111 },
		{ 0, 0, 111 },
		{ 2651, 2395, 0 },
		{ 2618, 2284, 0 },
		{ 2672, 1792, 0 },
		{ 2629, 1745, 0 },
		{ 2651, 2384, 0 },
		{ 1119, 3966, 0 },
		{ 2665, 2087, 0 },
		{ 2656, 1329, 0 },
		{ 2651, 2349, 0 },
		{ 2629, 1722, 0 },
		{ 2390, 1255, 0 },
		{ 2670, 1611, 0 },
		{ 2623, 2257, 0 },
		{ 2671, 2162, 0 },
		{ 2673, 1869, 0 },
		{ 2574, 2272, 0 },
		{ 1130, 4050, 0 },
		{ 2665, 2099, 0 },
		{ 2658, 2040, 0 },
		{ 2649, 1677, 0 },
		{ 2651, 2306, 0 },
		{ 1135, 4035, 0 },
		{ 2666, 2041, 0 },
		{ 2667, 1403, 0 },
		{ 2673, 1932, 0 },
		{ 2669, 2441, 0 },
		{ 2670, 1575, 0 },
		{ 2671, 2172, 0 },
		{ 2672, 1819, 0 },
		{ 2673, 1903, 0 },
		{ 2674, 3798, 0 },
		{ 0, 0, 109 },
		{ 2450, 2744, 131 },
		{ 0, 0, 131 },
		{ 2649, 1695, 0 },
		{ 2651, 2340, 0 },
		{ 2629, 1750, 0 },
		{ 1151, 4067, 0 },
		{ 2669, 2222, 0 },
		{ 2658, 2002, 0 },
		{ 2660, 2454, 0 },
		{ 2649, 1696, 0 },
		{ 2649, 1644, 0 },
		{ 2651, 2401, 0 },
		{ 2649, 1649, 0 },
		{ 2674, 3736, 0 },
		{ 0, 0, 130 },
		{ 1961, 3357, 99 },
		{ 0, 0, 99 },
		{ 0, 0, 100 },
		{ 2651, 2310, 0 },
		{ 2673, 1911, 0 },
		{ 2672, 1808, 0 },
		{ 2299, 1981, 0 },
		{ 2651, 2352, 0 },
		{ 1941, 3634, 0 },
		{ 2658, 2022, 0 },
		{ 2660, 2489, 0 },
		{ 2649, 1662, 0 },
		{ 2649, 1665, 0 },
		{ 2674, 3942, 0 },
		{ 2674, 3943, 0 },
		{ 2671, 2214, 0 },
		{ 2673, 1885, 0 },
		{ 2671, 2231, 0 },
		{ 2670, 1584, 0 },
		{ 2671, 2165, 0 },
		{ 2660, 2481, 0 },
		{ 2658, 2031, 0 },
		{ 2671, 2211, 0 },
		{ 2649, 1308, 0 },
		{ 2651, 2376, 0 },
		{ 2629, 1724, 0 },
		{ 2674, 3880, 0 },
		{ 0, 0, 97 },
		{ 1203, 0, 1 },
		{ 1203, 0, 132 },
		{ 1203, 2151, 182 },
		{ 1418, 254, 182 },
		{ 1418, 421, 182 },
		{ 1418, 409, 182 },
		{ 1418, 665, 182 },
		{ 1418, 411, 182 },
		{ 1418, 423, 182 },
		{ 1418, 398, 182 },
		{ 1418, 420, 182 },
		{ 1418, 480, 182 },
		{ 1203, 0, 182 },
		{ 1215, 2079, 182 },
		{ 1203, 2301, 182 },
		{ 2324, 2450, 178 },
		{ 1418, 526, 182 },
		{ 1418, 524, 182 },
		{ 1418, 571, 182 },
		{ 1418, 0, 182 },
		{ 1418, 632, 182 },
		{ 1418, 619, 182 },
		{ 2672, 1780, 0 },
		{ 0, 0, 133 },
		{ 2673, 1877, 0 },
		{ 1418, 586, 0 },
		{ 1418, 0, 0 },
		{ 2032, 3311, 0 },
		{ 1418, 606, 0 },
		{ 1418, 623, 0 },
		{ 1418, 649, 0 },
		{ 1418, 656, 0 },
		{ 1418, 647, 0 },
		{ 1418, 650, 0 },
		{ 1418, 657, 0 },
		{ 1418, 639, 0 },
		{ 1418, 632, 0 },
		{ 1418, 624, 0 },
		{ 1418, 627, 0 },
		{ 2651, 2362, 0 },
		{ 2651, 2368, 0 },
		{ 1419, 634, 0 },
		{ 1419, 635, 0 },
		{ 1418, 645, 0 },
		{ 1418, 646, 0 },
		{ 1418, 637, 0 },
		{ 2672, 1816, 0 },
		{ 2653, 2413, 0 },
		{ 1418, 635, 0 },
		{ 1418, 680, 0 },
		{ 1418, 657, 0 },
		{ 1418, 2, 0 },
		{ 1418, 22, 0 },
		{ 1418, 59, 0 },
		{ 1418, 64, 0 },
		{ 1418, 58, 0 },
		{ 1418, 38, 0 },
		{ 1418, 29, 0 },
		{ 1418, 26, 0 },
		{ 1418, 41, 0 },
		{ 1418, 28, 0 },
		{ 2673, 1884, 0 },
		{ 2618, 2281, 0 },
		{ 1418, 44, 0 },
		{ 1418, 34, 0 },
		{ 1419, 36, 0 },
		{ 1418, 33, 0 },
		{ 1418, 38, 0 },
		{ 2658, 2008, 0 },
		{ 0, 0, 181 },
		{ 1418, 51, 0 },
		{ 1418, 106, 0 },
		{ 1418, 93, 0 },
		{ 1418, 129, 0 },
		{ 1418, 134, 0 },
		{ 1418, 164, 0 },
		{ 1418, 173, 0 },
		{ 1418, 161, 0 },
		{ 1418, 139, 0 },
		{ 1418, 143, 0 },
		{ 1418, 158, 0 },
		{ 1418, 0, 167 },
		{ 1418, 193, 0 },
		{ 2672, 1775, 0 },
		{ 2671, 2240, 0 },
		{ 1418, 154, 0 },
		{ 1418, 158, 0 },
		{ 1418, 153, 0 },
		{ 1418, 155, 0 },
		{ 1418, 156, 0 },
		{ -1298, 921, 0 },
		{ 1419, 165, 0 },
		{ 1418, 209, 0 },
		{ 1418, 215, 0 },
		{ 1418, 207, 0 },
		{ 1418, 217, 0 },
		{ 1418, 220, 0 },
		{ 1418, 225, 0 },
		{ 1418, 243, 0 },
		{ 1418, 220, 0 },
		{ 1418, 239, 0 },
		{ 1418, 0, 166 },
		{ 1418, 250, 0 },
		{ 2299, 1962, 0 },
		{ 2673, 1897, 0 },
		{ 1418, 253, 0 },
		{ 1418, 266, 0 },
		{ 1418, 263, 0 },
		{ 1418, 0, 180 },
		{ 1418, 261, 0 },
		{ 0, 0, 168 },
		{ 1418, 254, 0 },
		{ 1420, 147, -13 },
		{ 1418, 282, 0 },
		{ 1418, 294, 0 },
		{ 1418, 291, 0 },
		{ 1418, 388, 0 },
		{ 1418, 298, 0 },
		{ 1418, 310, 0 },
		{ 1418, 284, 0 },
		{ 1418, 292, 0 },
		{ 1418, 284, 0 },
		{ 2651, 2315, 0 },
		{ 2651, 2324, 0 },
		{ 1418, 0, 170 },
		{ 1418, 334, 171 },
		{ 1418, 304, 0 },
		{ 1418, 307, 0 },
		{ 1418, 335, 0 },
		{ 1318, 2948, 0 },
		{ 2665, 3551, 0 },
		{ 2002, 3895, 157 },
		{ 1418, 365, 0 },
		{ 1418, 371, 0 },
		{ 1418, 370, 0 },
		{ 1418, 405, 0 },
		{ 1418, 406, 0 },
		{ 1418, 407, 0 },
		{ 1418, 395, 0 },
		{ 1418, 397, 0 },
		{ 1418, 379, 0 },
		{ 1418, 386, 0 },
		{ 1419, 372, 0 },
		{ 1941, 3635, 0 },
		{ 2032, 4164, 173 },
		{ 1418, 375, 0 },
		{ 1418, 387, 0 },
		{ 1418, 369, 0 },
		{ 1418, 385, 0 },
		{ 0, 0, 137 },
		{ 1420, 230, -16 },
		{ 1420, 344, -19 },
		{ 1420, 375, -22 },
		{ 1420, 459, -25 },
		{ 1420, 487, -28 },
		{ 1420, 2, -1 },
		{ 1418, 415, 0 },
		{ 1418, 431, 0 },
		{ 1418, 404, 0 },
		{ 1418, 0, 155 },
		{ 1418, 0, 169 },
		{ 2658, 2046, 0 },
		{ 1418, 403, 0 },
		{ 1418, 393, 0 },
		{ 1418, 397, 0 },
		{ 1419, 405, 0 },
		{ 1355, 2927, 0 },
		{ 2665, 3583, 0 },
		{ 2002, 3909, 158 },
		{ 1358, 2928, 0 },
		{ 2665, 3547, 0 },
		{ 2002, 3922, 159 },
		{ 1361, 2929, 0 },
		{ 2665, 3555, 0 },
		{ 2002, 3911, 162 },
		{ 1364, 2930, 0 },
		{ 2665, 3613, 0 },
		{ 2002, 3903, 163 },
		{ 1367, 2932, 0 },
		{ 2665, 3545, 0 },
		{ 2002, 3913, 164 },
		{ 1370, 2933, 0 },
		{ 2665, 3549, 0 },
		{ 2002, 3898, 165 },
		{ 1418, 450, 0 },
		{ 1420, 117, -4 },
		{ 1418, 423, 0 },
		{ 2660, 2483, 0 },
		{ 1418, 435, 0 },
		{ 1418, 480, 0 },
		{ 1418, 442, 0 },
		{ 1418, 489, 0 },
		{ 0, 0, 139 },
		{ 0, 0, 141 },
		{ 0, 0, 147 },
		{ 0, 0, 149 },
		{ 0, 0, 151 },
		{ 0, 0, 153 },
		{ 1420, 145, -7 },
		{ 1388, 2943, 0 },
		{ 2665, 3581, 0 },
		{ 2002, 3918, 161 },
		{ 1418, 0, 154 },
		{ 2649, 1684, 0 },
		{ 1418, 477, 0 },
		{ 1418, 492, 0 },
		{ 1419, 485, 0 },
		{ 1418, 482, 0 },
		{ 1397, 2957, 0 },
		{ 2665, 3553, 0 },
		{ 2002, 3921, 160 },
		{ 0, 0, 145 },
		{ 2649, 1638, 0 },
		{ 1418, 4, 176 },
		{ 1419, 514, 0 },
		{ 1418, 1, 179 },
		{ 1418, 529, 0 },
		{ 0, 0, 143 },
		{ 2674, 3928, 0 },
		{ 2674, 3929, 0 },
		{ 1418, 517, 0 },
		{ 0, 0, 177 },
		{ 1418, 549, 0 },
		{ 2674, 3944, 0 },
		{ 0, 0, 175 },
		{ 1418, 557, 0 },
		{ 1418, 562, 0 },
		{ 0, 0, 174 },
		{ 1418, 567, 0 },
		{ 1418, 558, 0 },
		{ 1419, 560, 172 },
		{ 1420, 722, 0 },
		{ 1421, 532, -10 },
		{ 1422, 2974, 0 },
		{ 2665, 3537, 0 },
		{ 2002, 3916, 156 },
		{ 0, 0, 135 },
		{ 1961, 3354, 224 },
		{ 0, 0, 224 },
		{ 2651, 2328, 0 },
		{ 2673, 1889, 0 },
		{ 2672, 1821, 0 },
		{ 2299, 1977, 0 },
		{ 2651, 2351, 0 },
		{ 1941, 3597, 0 },
		{ 2658, 2050, 0 },
		{ 2660, 2496, 0 },
		{ 2649, 1650, 0 },
		{ 2649, 1659, 0 },
		{ 2674, 3932, 0 },
		{ 2674, 3933, 0 },
		{ 2671, 2227, 0 },
		{ 2673, 1906, 0 },
		{ 2671, 2233, 0 },
		{ 2670, 1617, 0 },
		{ 2671, 2076, 0 },
		{ 2660, 2500, 0 },
		{ 2658, 2029, 0 },
		{ 2671, 2161, 0 },
		{ 2649, 1314, 0 },
		{ 2651, 2312, 0 },
		{ 2629, 1720, 0 },
		{ 2674, 3837, 0 },
		{ 0, 0, 223 },
		{ 1961, 3353, 226 },
		{ 0, 0, 226 },
		{ 0, 0, 227 },
		{ 2651, 2320, 0 },
		{ 2673, 1863, 0 },
		{ 2672, 1809, 0 },
		{ 2299, 1967, 0 },
		{ 2651, 2338, 0 },
		{ 1941, 3637, 0 },
		{ 2658, 2054, 0 },
		{ 2660, 2480, 0 },
		{ 2649, 1666, 0 },
		{ 2649, 1673, 0 },
		{ 2674, 3945, 0 },
		{ 2674, 3927, 0 },
		{ 2669, 2433, 0 },
		{ 2629, 1725, 0 },
		{ 2672, 1824, 0 },
		{ 2649, 1674, 0 },
		{ 2649, 1675, 0 },
		{ 2672, 1782, 0 },
		{ 2656, 1331, 0 },
		{ 2651, 2383, 0 },
		{ 2629, 1699, 0 },
		{ 2674, 3909, 0 },
		{ 0, 0, 225 },
		{ 1961, 3345, 229 },
		{ 0, 0, 229 },
		{ 0, 0, 230 },
		{ 2651, 2385, 0 },
		{ 2673, 1913, 0 },
		{ 2672, 1794, 0 },
		{ 2299, 1963, 0 },
		{ 2651, 2308, 0 },
		{ 1941, 3614, 0 },
		{ 2658, 2018, 0 },
		{ 2660, 2473, 0 },
		{ 2649, 1678, 0 },
		{ 2649, 1680, 0 },
		{ 2674, 3938, 0 },
		{ 2674, 3939, 0 },
		{ 2299, 1979, 0 },
		{ 2667, 1400, 0 },
		{ 2670, 1589, 0 },
		{ 2660, 2497, 0 },
		{ 2670, 1591, 0 },
		{ 2672, 1822, 0 },
		{ 2629, 1727, 0 },
		{ 2674, 3763, 0 },
		{ 0, 0, 228 },
		{ 1961, 3352, 232 },
		{ 0, 0, 232 },
		{ 0, 0, 233 },
		{ 2651, 2342, 0 },
		{ 2673, 1893, 0 },
		{ 2672, 1830, 0 },
		{ 2299, 1937, 0 },
		{ 2651, 2359, 0 },
		{ 1941, 3644, 0 },
		{ 2658, 2020, 0 },
		{ 2660, 2492, 0 },
		{ 2649, 1689, 0 },
		{ 2649, 1691, 0 },
		{ 2674, 3935, 0 },
		{ 2674, 3937, 0 },
		{ 2651, 2378, 0 },
		{ 2656, 1333, 0 },
		{ 2660, 2513, 0 },
		{ 2658, 2036, 0 },
		{ 2667, 1398, 0 },
		{ 2660, 2463, 0 },
		{ 2670, 1433, 0 },
		{ 2672, 1793, 0 },
		{ 2629, 1716, 0 },
		{ 2674, 3765, 0 },
		{ 0, 0, 231 },
		{ 1961, 3338, 235 },
		{ 0, 0, 235 },
		{ 0, 0, 236 },
		{ 2651, 2398, 0 },
		{ 2673, 1850, 0 },
		{ 2672, 1804, 0 },
		{ 2299, 1965, 0 },
		{ 2651, 2309, 0 },
		{ 1941, 3623, 0 },
		{ 2658, 2016, 0 },
		{ 2660, 2504, 0 },
		{ 2649, 1641, 0 },
		{ 2649, 1642, 0 },
		{ 2674, 3930, 0 },
		{ 2674, 3931, 0 },
		{ 2672, 1812, 0 },
		{ 2533, 1768, 0 },
		{ 2670, 1456, 0 },
		{ 2671, 2239, 0 },
		{ 2299, 1939, 0 },
		{ 2671, 2241, 0 },
		{ 2649, 1647, 0 },
		{ 2651, 2347, 0 },
		{ 2629, 1726, 0 },
		{ 2674, 3720, 0 },
		{ 0, 0, 234 },
		{ 2605, 3704, 114 },
		{ 0, 0, 114 },
		{ 2618, 2291, 0 },
		{ 2670, 1556, 0 },
		{ 2651, 2353, 0 },
		{ 2629, 1732, 0 },
		{ 1561, 4022, 0 },
		{ 2665, 2093, 0 },
		{ 2656, 1334, 0 },
		{ 2651, 2371, 0 },
		{ 2629, 1742, 0 },
		{ 2390, 1259, 0 },
		{ 2670, 1582, 0 },
		{ 2623, 2260, 0 },
		{ 2671, 2218, 0 },
		{ 2673, 1926, 0 },
		{ 2574, 2270, 0 },
		{ 1572, 4052, 0 },
		{ 2665, 2091, 0 },
		{ 2658, 2026, 0 },
		{ 2649, 1664, 0 },
		{ 2651, 2304, 0 },
		{ 1577, 3997, 0 },
		{ 2666, 2043, 0 },
		{ 2667, 1391, 0 },
		{ 2673, 1853, 0 },
		{ 2669, 2432, 0 },
		{ 2670, 1586, 0 },
		{ 2671, 2077, 0 },
		{ 2672, 1801, 0 },
		{ 2673, 1872, 0 },
		{ 2674, 3728, 0 },
		{ 0, 0, 112 },
		{ 1961, 3341, 217 },
		{ 0, 0, 217 },
		{ 2651, 2321, 0 },
		{ 2673, 1873, 0 },
		{ 2672, 1803, 0 },
		{ 2299, 1958, 0 },
		{ 2651, 2335, 0 },
		{ 1941, 3636, 0 },
		{ 2658, 2014, 0 },
		{ 2660, 2499, 0 },
		{ 2649, 1669, 0 },
		{ 2649, 1670, 0 },
		{ 2674, 3940, 0 },
		{ 2674, 3941, 0 },
		{ 2653, 2412, 0 },
		{ 2671, 2223, 0 },
		{ 2649, 1671, 0 },
		{ 2533, 1763, 0 },
		{ 2658, 2024, 0 },
		{ 2660, 2471, 0 },
		{ 2390, 1310, 0 },
		{ 2674, 3726, 0 },
		{ 0, 0, 215 },
		{ 1625, 0, 1 },
		{ 1784, 2340, 331 },
		{ 2651, 2364, 331 },
		{ 2660, 2418, 331 },
		{ 2673, 1782, 331 },
		{ 1625, 0, 298 },
		{ 1625, 2319, 331 },
		{ 2667, 1385, 331 },
		{ 2541, 3658, 331 },
		{ 1940, 3067, 331 },
		{ 2450, 2752, 331 },
		{ 1940, 3069, 331 },
		{ 1925, 3443, 331 },
		{ 2672, 1628, 331 },
		{ 1625, 0, 331 },
		{ 2324, 2451, 329 },
		{ 2660, 2280, 331 },
		{ 2660, 2457, 331 },
		{ 0, 0, 331 },
		{ 2672, 1829, 0 },
		{ -1630, 4174, 289 },
		{ -1631, 4016, 0 },
		{ 2673, 1915, 0 },
		{ 0, 0, 294 },
		{ 0, 0, 295 },
		{ 2658, 2052, 0 },
		{ 2671, 2164, 0 },
		{ 2651, 2390, 0 },
		{ 0, 0, 299 },
		{ 2673, 1917, 0 },
		{ 2629, 1734, 0 },
		{ 2671, 2210, 0 },
		{ 1893, 2619, 0 },
		{ 1910, 3040, 0 },
		{ 1830, 2822, 0 },
		{ 1833, 2697, 0 },
		{ 1910, 3045, 0 },
		{ 2667, 1369, 0 },
		{ 2649, 1679, 0 },
		{ 2673, 1848, 0 },
		{ 2670, 1618, 0 },
		{ 2629, 1746, 0 },
		{ 2672, 1785, 0 },
		{ 2447, 4000, 0 },
		{ 2672, 1786, 0 },
		{ 2649, 1682, 0 },
		{ 2670, 1408, 0 },
		{ 2673, 1874, 0 },
		{ 2653, 2421, 0 },
		{ 2629, 1706, 0 },
		{ 2658, 2033, 0 },
		{ 1961, 3356, 0 },
		{ 1893, 2540, 0 },
		{ 1893, 2541, 0 },
		{ 1925, 3396, 0 },
		{ 1855, 3231, 0 },
		{ 2651, 2331, 0 },
		{ 2649, 1686, 0 },
		{ 2653, 2408, 0 },
		{ 2667, 1390, 0 },
		{ 2651, 2339, 0 },
		{ 2658, 2044, 0 },
		{ 0, 4169, 0 },
		{ 2299, 1980, 0 },
		{ 2651, 2344, 0 },
		{ 1940, 3064, 0 },
		{ 2670, 1453, 0 },
		{ 0, 0, 330 },
		{ 2651, 2348, 0 },
		{ 2653, 2417, 0 },
		{ 1925, 3412, 0 },
		{ 1871, 2871, 0 },
		{ 1910, 3041, 0 },
		{ 1927, 2981, 0 },
		{ 1893, 2556, 0 },
		{ 0, 0, 319 },
		{ 1941, 3638, 0 },
		{ 2672, 1806, 0 },
		{ 2629, 1721, 0 },
		{ 2673, 1898, 0 },
		{ -1707, 996, 0 },
		{ 0, 0, 291 },
		{ 2651, 2355, 0 },
		{ 0, 0, 318 },
		{ 2533, 1755, 0 },
		{ 2671, 2213, 0 },
		{ 2673, 1905, 0 },
		{ 1722, 3962, 0 },
		{ 1888, 3123, 0 },
		{ 2646, 3329, 0 },
		{ 1927, 2911, 0 },
		{ 1893, 2566, 0 },
		{ 1910, 3015, 0 },
		{ 2672, 1813, 0 },
		{ 2299, 1969, 0 },
		{ 2673, 1908, 0 },
		{ 2670, 1459, 0 },
		{ 0, 0, 320 },
		{ 1941, 3604, 297 },
		{ 2670, 1520, 0 },
		{ 2669, 2439, 0 },
		{ 2670, 1522, 0 },
		{ 0, 0, 323 },
		{ 0, 0, 324 },
		{ 1727, 0, -35 },
		{ 1906, 2653, 0 },
		{ 1940, 3094, 0 },
		{ 1910, 3029, 0 },
		{ 1925, 3445, 0 },
		{ 2671, 2236, 0 },
		{ 0, 0, 322 },
		{ 0, 0, 328 },
		{ 0, 3960, 0 },
		{ 2658, 2028, 0 },
		{ 2649, 1629, 0 },
		{ 2660, 2505, 0 },
		{ 1961, 3351, 0 },
		{ 2665, 3627, 0 },
		{ 2002, 3915, 313 },
		{ 1925, 3452, 0 },
		{ 2541, 3647, 0 },
		{ 1927, 2933, 0 },
		{ 1927, 2934, 0 },
		{ 2673, 1922, 0 },
		{ 0, 0, 325 },
		{ 0, 0, 326 },
		{ 2660, 2510, 0 },
		{ 2008, 4007, 0 },
		{ 2658, 2032, 0 },
		{ 2651, 2393, 0 },
		{ 0, 0, 303 },
		{ 1750, 0, -38 },
		{ 1752, 0, -41 },
		{ 1940, 3107, 0 },
		{ 1941, 3641, 0 },
		{ 0, 0, 321 },
		{ 2649, 1634, 0 },
		{ 0, 0, 296 },
		{ 1961, 3339, 0 },
		{ 2673, 1924, 0 },
		{ 2665, 3587, 0 },
		{ 2002, 3897, 314 },
		{ 2665, 3611, 0 },
		{ 2002, 3901, 315 },
		{ 2541, 3660, 0 },
		{ 1762, 0, -71 },
		{ 2649, 1636, 0 },
		{ 2651, 2403, 0 },
		{ 2651, 2303, 0 },
		{ 0, 0, 305 },
		{ 0, 0, 307 },
		{ 1767, 0, -77 },
		{ 2665, 3631, 0 },
		{ 2002, 3920, 317 },
		{ 0, 0, 293 },
		{ 2673, 1930, 0 },
		{ 2629, 1735, 0 },
		{ 2665, 3539, 0 },
		{ 2002, 3896, 316 },
		{ 0, 0, 311 },
		{ 2672, 1772, 0 },
		{ 2660, 2487, 0 },
		{ 0, 0, 309 },
		{ 2299, 1975, 0 },
		{ 2670, 1547, 0 },
		{ 2651, 2311, 0 },
		{ 2671, 2173, 0 },
		{ 0, 0, 327 },
		{ 2672, 1779, 0 },
		{ 2673, 1859, 0 },
		{ 1781, 0, -44 },
		{ 2665, 3557, 0 },
		{ 2002, 3919, 312 },
		{ 0, 0, 301 },
		{ 1625, 2356, 331 },
		{ 1788, 2080, 331 },
		{ -1786, 4165, 289 },
		{ -1787, 4013, 0 },
		{ 2447, 4004, 0 },
		{ 2447, 3984, 0 },
		{ 0, 0, 290 },
		{ 2447, 4001, 0 },
		{ -1792, 29, 0 },
		{ -1793, 4021, 0 },
		{ 1796, 3, 0 },
		{ 2447, 3986, 0 },
		{ 2665, 4072, 0 },
		{ 0, 0, 292 },
		{ 1814, 0, 1 },
		{ 2010, 2343, 288 },
		{ 2651, 2333, 288 },
		{ 1814, 0, 242 },
		{ 1814, 2288, 288 },
		{ 1910, 3031, 288 },
		{ 1814, 0, 245 },
		{ 2667, 1401, 288 },
		{ 2541, 3652, 288 },
		{ 1940, 3084, 288 },
		{ 2646, 2696, 288 },
		{ 1940, 3086, 288 },
		{ 1925, 3406, 288 },
		{ 2660, 2475, 288 },
		{ 2672, 1629, 288 },
		{ 1814, 0, 288 },
		{ 2324, 2449, 285 },
		{ 2660, 2484, 288 },
		{ 2653, 2420, 288 },
		{ 2541, 3654, 288 },
		{ 2660, 1335, 288 },
		{ 0, 0, 288 },
		{ 2672, 1787, 0 },
		{ -1821, 22, 237 },
		{ -1822, 4022, 0 },
		{ 2673, 1879, 0 },
		{ 0, 0, 243 },
		{ 2673, 1883, 0 },
		{ 2672, 1791, 0 },
		{ 2629, 1705, 0 },
		{ 1893, 2628, 0 },
		{ 1910, 3049, 0 },
		{ 0, 2840, 0 },
		{ 1888, 3141, 0 },
		{ 0, 2683, 0 },
		{ 0, 2707, 0 },
		{ 1910, 3053, 0 },
		{ 2658, 2030, 0 },
		{ 2667, 1338, 0 },
		{ 2649, 1651, 0 },
		{ 2673, 1894, 0 },
		{ 2651, 2357, 0 },
		{ 2651, 2358, 0 },
		{ 2672, 1796, 0 },
		{ 2008, 4004, 0 },
		{ 2672, 1798, 0 },
		{ 2447, 3992, 0 },
		{ 2672, 1800, 0 },
		{ 2653, 2407, 0 },
		{ 2533, 1766, 0 },
		{ 2629, 1715, 0 },
		{ 1961, 3343, 0 },
		{ 1893, 2548, 0 },
		{ 1893, 2549, 0 },
		{ 2646, 3278, 0 },
		{ 2646, 3279, 0 },
		{ 1925, 3449, 0 },
		{ 0, 3202, 0 },
		{ 2649, 1652, 0 },
		{ 2651, 2374, 0 },
		{ 2649, 1655, 0 },
		{ 2653, 2419, 0 },
		{ 2673, 1909, 0 },
		{ 2649, 1657, 0 },
		{ 2658, 2005, 0 },
		{ 0, 0, 287 },
		{ 2658, 2006, 0 },
		{ 0, 0, 239 },
		{ 2299, 1973, 0 },
		{ 0, 0, 284 },
		{ 2667, 1340, 0 },
		{ 2651, 2387, 0 },
		{ 1925, 3462, 0 },
		{ 0, 2857, 0 },
		{ 1910, 3043, 0 },
		{ 1874, 3191, 0 },
		{ 0, 3192, 0 },
		{ 1927, 2944, 0 },
		{ 1893, 2561, 0 },
		{ 2651, 2389, 0 },
		{ 0, 0, 277 },
		{ 1941, 3603, 0 },
		{ 2672, 1811, 0 },
		{ 2670, 1587, 0 },
		{ 2670, 1588, 0 },
		{ 2667, 1352, 0 },
		{ -1901, 1071, 0 },
		{ 2651, 2402, 0 },
		{ 2658, 2025, 0 },
		{ 2673, 1925, 0 },
		{ 0, 3128, 0 },
		{ 2646, 3310, 0 },
		{ 1927, 2956, 0 },
		{ 2646, 3312, 0 },
		{ 2646, 3313, 0 },
		{ 0, 2571, 0 },
		{ 1910, 3014, 0 },
		{ 0, 0, 276 },
		{ 2672, 1820, 0 },
		{ 2299, 1959, 0 },
		{ 2671, 2246, 0 },
		{ 0, 0, 283 },
		{ 2660, 2476, 0 },
		{ 0, 0, 278 },
		{ 0, 0, 241 },
		{ 2660, 2478, 0 },
		{ 2670, 1600, 0 },
		{ 1918, 0, -68 },
		{ 0, 2669, 0 },
		{ 1940, 3103, 0 },
		{ 1909, 2656, 0 },
		{ 1906, 2655, 0 },
		{ 0, 3028, 0 },
		{ 1925, 3410, 0 },
		{ 2671, 2252, 0 },
		{ 0, 0, 280 },
		{ 2669, 2427, 0 },
		{ 2670, 1601, 0 },
		{ 2670, 1606, 0 },
		{ 1961, 3358, 0 },
		{ 2665, 3541, 0 },
		{ 2002, 3917, 267 },
		{ 1925, 3416, 0 },
		{ 2541, 3653, 0 },
		{ 1925, 3417, 0 },
		{ 1925, 3418, 0 },
		{ 1925, 3419, 0 },
		{ 0, 3420, 0 },
		{ 1927, 2972, 0 },
		{ 0, 2973, 0 },
		{ 2673, 1849, 0 },
		{ 2660, 2490, 0 },
		{ 2671, 2088, 0 },
		{ 2671, 2094, 0 },
		{ 2651, 2322, 0 },
		{ 0, 0, 249 },
		{ 1947, 0, -47 },
		{ 1949, 0, -50 },
		{ 1951, 0, -56 },
		{ 1953, 0, -59 },
		{ 1955, 0, -62 },
		{ 1957, 0, -65 },
		{ 0, 3066, 0 },
		{ 0, 3626, 0 },
		{ 0, 0, 279 },
		{ 2658, 2034, 0 },
		{ 2672, 1827, 0 },
		{ 2672, 1828, 0 },
		{ 2673, 1854, 0 },
		{ 2665, 3615, 0 },
		{ 2002, 3899, 268 },
		{ 2665, 3617, 0 },
		{ 2002, 3902, 269 },
		{ 2665, 3619, 0 },
		{ 2002, 3908, 272 },
		{ 2665, 3621, 0 },
		{ 2002, 3910, 273 },
		{ 2665, 3623, 0 },
		{ 2002, 3912, 274 },
		{ 2665, 3625, 0 },
		{ 2002, 3914, 275 },
		{ 2541, 3659, 0 },
		{ 1972, 0, -74 },
		{ 0, 3349, 0 },
		{ 2673, 1857, 0 },
		{ 2673, 1858, 0 },
		{ 2651, 2336, 0 },
		{ 0, 0, 251 },
		{ 0, 0, 253 },
		{ 0, 0, 259 },
		{ 0, 0, 261 },
		{ 0, 0, 263 },
		{ 0, 0, 265 },
		{ 1978, 0, -80 },
		{ 2665, 3651, 0 },
		{ 2002, 3894, 271 },
		{ 2651, 2337, 0 },
		{ 2660, 2455, 0 },
		{ 0, 2649, 282 },
		{ 2629, 1736, 0 },
		{ 2665, 3543, 0 },
		{ 2002, 3900, 270 },
		{ 0, 0, 257 },
		{ 2673, 1861, 0 },
		{ 2629, 1737, 0 },
		{ 0, 0, 244 },
		{ 2660, 2464, 0 },
		{ 0, 0, 255 },
		{ 2672, 1834, 0 },
		{ 2390, 1278, 0 },
		{ 2670, 1612, 0 },
		{ 2299, 1961, 0 },
		{ 2605, 3770, 0 },
		{ 2671, 2220, 0 },
		{ 2651, 2350, 0 },
		{ 2658, 2010, 0 },
		{ 2672, 1776, 0 },
		{ 0, 0, 281 },
		{ 2574, 2269, 0 },
		{ 2673, 1876, 0 },
		{ 2672, 1778, 0 },
		{ 2001, 0, -53 },
		{ 2629, 1743, 0 },
		{ 2665, 3585, 0 },
		{ 0, 3923, 266 },
		{ 2671, 2238, 0 },
		{ 0, 0, 247 },
		{ 2670, 1614, 0 },
		{ 2623, 2258, 0 },
		{ 2299, 1971, 0 },
		{ 0, 4003, 0 },
		{ 0, 0, 286 },
		{ 1814, 2353, 288 },
		{ 2014, 2081, 288 },
		{ -2012, 21, 237 },
		{ -2013, 4019, 0 },
		{ 2447, 4005, 0 },
		{ 2447, 3976, 0 },
		{ 0, 0, 238 },
		{ 2447, 3988, 0 },
		{ -2018, 4167, 0 },
		{ -2019, 4012, 0 },
		{ 2022, 0, 239 },
		{ 2447, 3991, 0 },
		{ 2665, 4124, 0 },
		{ 0, 0, 240 },
		{ 0, 3520, 333 },
		{ 0, 0, 333 },
		{ 2651, 2375, 0 },
		{ 2618, 2276, 0 },
		{ 2660, 2511, 0 },
		{ 2667, 1393, 0 },
		{ 2658, 2027, 0 },
		{ 2670, 1621, 0 },
		{ 0, 4, 0 },
		{ 2629, 1703, 0 },
		{ 2667, 1397, 0 },
		{ 2673, 1896, 0 },
		{ 2037, 4003, 0 },
		{ 2665, 1633, 0 },
		{ 2660, 2468, 0 },
		{ 2629, 1708, 0 },
		{ 2660, 2472, 0 },
		{ 2299, 1957, 0 },
		{ 2651, 2391, 0 },
		{ 2670, 1437, 0 },
		{ 2651, 2394, 0 },
		{ 2629, 1711, 0 },
		{ 2649, 1685, 0 },
		{ 2674, 3806, 0 },
		{ 0, 0, 332 },
		{ 2447, 3973, 371 },
		{ 0, 0, 338 },
		{ 0, 0, 340 },
		{ 2119, 669, 368 },
		{ 2185, 682, 368 },
		{ 2198, 680, 368 },
		{ 2163, 681, 368 },
		{ 2182, 689, 368 },
		{ 2119, 675, 368 },
		{ 2198, 679, 368 },
		{ 2081, 693, 368 },
		{ 2182, 695, 368 },
		{ 2182, 696, 368 },
		{ 2185, 693, 368 },
		{ 2147, 704, 368 },
		{ 2651, 1389, 367 },
		{ 2088, 2178, 371 },
		{ 2221, 693, 368 },
		{ 2651, 2326, 371 },
		{ -2068, 27, 334 },
		{ -2069, 4020, 0 },
		{ 2221, 694, 368 },
		{ 2221, 695, 368 },
		{ 2124, 693, 368 },
		{ 2185, 702, 368 },
		{ 2188, 697, 368 },
		{ 2185, 704, 368 },
		{ 2147, 713, 368 },
		{ 2127, 703, 368 },
		{ 2198, 698, 368 },
		{ 2165, 697, 368 },
		{ 2188, 703, 368 },
		{ 2066, 713, 368 },
		{ 2207, 716, 368 },
		{ 2177, 727, 368 },
		{ 2207, 718, 368 },
		{ 2127, 721, 368 },
		{ 2651, 1477, 364 },
		{ 2109, 2, 0 },
		{ 2651, 1487, 365 },
		{ 2673, 1888, 0 },
		{ 2447, 3978, 0 },
		{ 2066, 732, 368 },
		{ 2182, 730, 368 },
		{ 2119, 743, 368 },
		{ 2207, 751, 368 },
		{ 2165, 746, 368 },
		{ 2165, 748, 368 },
		{ 2127, 783, 368 },
		{ 2182, 791, 368 },
		{ 2163, 775, 368 },
		{ 2147, 807, 368 },
		{ 2204, 790, 368 },
		{ 2204, 791, 368 },
		{ 2182, 806, 368 },
		{ 2119, 791, 368 },
		{ 2147, 812, 368 },
		{ 2177, 810, 368 },
		{ 2130, 1349, 0 },
		{ 2109, 0, 0 },
		{ 2132, 2163, 366 },
		{ 2132, 1352, 0 },
		{ 2653, 2410, 0 },
		{ 0, 0, 336 },
		{ 2182, 836, 368 },
		{ 2224, 354, 368 },
		{ 2127, 831, 368 },
		{ 2165, 824, 368 },
		{ 2224, 356, 368 },
		{ 2185, 878, 368 },
		{ 2066, 863, 368 },
		{ 2126, 882, 368 },
		{ 2185, 877, 368 },
		{ 2165, 868, 368 },
		{ 2204, 870, 368 },
		{ 2066, 900, 368 },
		{ 2198, 899, 368 },
		{ 2066, 915, 368 },
		{ 2066, 907, 368 },
		{ 2066, 898, 368 },
		{ 2130, 0, 0 },
		{ 2132, 2119, 364 },
		{ 2132, 0, 0 },
		{ 2065, 2153, 365 },
		{ 0, 0, 369 },
		{ 2198, 940, 368 },
		{ 2665, 1737, 0 },
		{ 2147, 958, 368 },
		{ 2224, 8, 368 },
		{ 2665, 1595, 0 },
		{ 2240, 6, 368 },
		{ 2204, 943, 368 },
		{ 2147, 962, 368 },
		{ 2165, 944, 368 },
		{ 2163, 969, 368 },
		{ 2185, 983, 368 },
		{ 2188, 978, 368 },
		{ 2198, 976, 368 },
		{ 2066, 995, 368 },
		{ 2182, 1028, 368 },
		{ 2224, 10, 368 },
		{ 2185, 1025, 368 },
		{ 2224, 119, 368 },
		{ 2671, 2209, 0 },
		{ 2165, 1016, 368 },
		{ 2665, 1571, 0 },
		{ 2670, 1510, 0 },
		{ 2674, 3724, 0 },
		{ 2665, 4063, 344 },
		{ 2221, 1024, 368 },
		{ 2165, 1018, 368 },
		{ 2185, 1062, 368 },
		{ 2185, 1030, 368 },
		{ 2188, 1051, 368 },
		{ 2066, 1046, 368 },
		{ 2185, 1060, 368 },
		{ 2066, 1088, 368 },
		{ 2224, 121, 368 },
		{ 2665, 1609, 0 },
		{ 2207, 1096, 368 },
		{ 2672, 1657, 0 },
		{ 2299, 1982, 0 },
		{ 2185, 1102, 368 },
		{ 2670, 1593, 0 },
		{ 2660, 2466, 0 },
		{ 2240, 234, 368 },
		{ 2188, 1097, 368 },
		{ 2188, 1098, 368 },
		{ 2066, 1110, 368 },
		{ 2207, 1127, 368 },
		{ 2207, 1128, 368 },
		{ 2182, 1138, 368 },
		{ 2207, 1130, 368 },
		{ 2066, 1141, 368 },
		{ 2672, 1472, 0 },
		{ 2651, 2399, 0 },
		{ 2066, 1174, 368 },
		{ 2618, 2278, 0 },
		{ 2671, 2169, 0 },
		{ 2066, 1169, 368 },
		{ 2660, 2495, 0 },
		{ 2670, 1441, 0 },
		{ 2674, 3804, 0 },
		{ 0, 0, 360 },
		{ 2198, 1167, 368 },
		{ 2207, 1172, 368 },
		{ 2224, 128, 368 },
		{ 2199, 1181, 368 },
		{ 2224, 236, 368 },
		{ 2066, 1198, 368 },
		{ 2066, 1210, 368 },
		{ 2224, 238, 368 },
		{ 2660, 2512, 0 },
		{ 2618, 2282, 0 },
		{ 2653, 2414, 0 },
		{ 2066, 1200, 368 },
		{ 2222, 4069, 0 },
		{ 2670, 1574, 0 },
		{ 2066, 1207, 368 },
		{ 2670, 1581, 0 },
		{ 2649, 1640, 0 },
		{ 2224, 240, 368 },
		{ 2224, 242, 368 },
		{ 2665, 1940, 0 },
		{ 2224, 244, 368 },
		{ 2665, 1636, 0 },
		{ 2224, 348, 368 },
		{ 2224, 350, 368 },
		{ 2669, 1612, 0 },
		{ 2629, 1718, 0 },
		{ 2618, 2295, 0 },
		{ 2667, 1392, 0 },
		{ 2224, 1339, 368 },
		{ 2672, 1567, 0 },
		{ 2674, 3835, 0 },
		{ 2240, 463, 368 },
		{ 2649, 1660, 0 },
		{ 2674, 3882, 0 },
		{ 2665, 1938, 0 },
		{ 2672, 1641, 0 },
		{ 2651, 2354, 0 },
		{ 2672, 1474, 0 },
		{ 2629, 1728, 0 },
		{ 2665, 1623, 0 },
		{ 2665, 1591, 0 },
		{ 2649, 1667, 0 },
		{ 2673, 1855, 0 },
		{ 2257, 4042, 0 },
		{ 2651, 2365, 0 },
		{ 2649, 1668, 0 },
		{ 2669, 2429, 0 },
		{ 2261, 607, 368 },
		{ 2651, 2369, 0 },
		{ 2533, 1756, 0 },
		{ 2665, 4069, 342 },
		{ 2665, 1597, 0 },
		{ 2674, 3907, 0 },
		{ 2266, 20, 351 },
		{ 2672, 1797, 0 },
		{ 2533, 1764, 0 },
		{ 2673, 1871, 0 },
		{ 2660, 2465, 0 },
		{ 2618, 2292, 0 },
		{ 2672, 1799, 0 },
		{ 2629, 1741, 0 },
		{ 2671, 2228, 0 },
		{ 2656, 1332, 0 },
		{ 2667, 1388, 0 },
		{ 2665, 1599, 0 },
		{ 2673, 1878, 0 },
		{ 2533, 1759, 0 },
		{ 2651, 2392, 0 },
		{ 2674, 3878, 0 },
		{ 2665, 4060, 363 },
		{ 2673, 1880, 0 },
		{ 2670, 1439, 0 },
		{ 2671, 2244, 0 },
		{ 1672, 17, 350 },
		{ 2660, 2488, 0 },
		{ 2651, 2397, 0 },
		{ 2671, 2245, 0 },
		{ 2629, 1747, 0 },
		{ 2618, 2294, 0 },
		{ 2324, 2447, 0 },
		{ 2672, 1810, 0 },
		{ 2283, 1244, 0 },
		{ 2292, 4033, 0 },
		{ 2533, 1767, 0 },
		{ 2669, 2437, 0 },
		{ 2670, 1452, 0 },
		{ 2629, 1700, 0 },
		{ 2297, 4054, 0 },
		{ 2651, 2307, 0 },
		{ 2671, 2086, 0 },
		{ 0, 1243, 0 },
		{ 2658, 2051, 0 },
		{ 2629, 1704, 0 },
		{ 2670, 1455, 0 },
		{ 2651, 2313, 0 },
		{ 2649, 1681, 0 },
		{ 2660, 2461, 0 },
		{ 2324, 2448, 0 },
		{ 2651, 2317, 0 },
		{ 2665, 1603, 0 },
		{ 2658, 2004, 0 },
		{ 2629, 1709, 0 },
		{ 2649, 1683, 0 },
		{ 2671, 2206, 0 },
		{ 2672, 1643, 0 },
		{ 2629, 1713, 0 },
		{ 0, 1960, 0 },
		{ 2651, 2332, 0 },
		{ 2674, 3870, 0 },
		{ 2660, 2479, 0 },
		{ 2672, 1833, 0 },
		{ 2673, 1912, 0 },
		{ 2533, 1758, 0 },
		{ 2324, 2446, 0 },
		{ 2660, 2486, 0 },
		{ 2674, 3722, 0 },
		{ 2670, 1466, 0 },
		{ 2672, 1773, 0 },
		{ 2671, 2219, 0 },
		{ 2329, 3985, 0 },
		{ 2673, 1918, 0 },
		{ 2533, 1761, 0 },
		{ 2660, 2493, 0 },
		{ 2671, 2226, 0 },
		{ 2674, 3794, 0 },
		{ 2665, 4122, 361 },
		{ 2670, 1486, 0 },
		{ 2629, 1717, 0 },
		{ 2670, 1494, 0 },
		{ 2629, 1719, 0 },
		{ 2671, 2235, 0 },
		{ 0, 2444, 0 },
		{ 2665, 4153, 349 },
		{ 2660, 2506, 0 },
		{ 2670, 1504, 0 },
		{ 2533, 1769, 0 },
		{ 2672, 1650, 0 },
		{ 2574, 2268, 0 },
		{ 2651, 2356, 0 },
		{ 2670, 1512, 0 },
		{ 2649, 1692, 0 },
		{ 2665, 4066, 343 },
		{ 2672, 1790, 0 },
		{ 2649, 1693, 0 },
		{ 2649, 1694, 0 },
		{ 2671, 2247, 0 },
		{ 2658, 2048, 0 },
		{ 2671, 2251, 0 },
		{ 2670, 1514, 0 },
		{ 2356, 4004, 0 },
		{ 2670, 1518, 0 },
		{ 2533, 1765, 0 },
		{ 2359, 4013, 0 },
		{ 2629, 1729, 0 },
		{ 2671, 2078, 0 },
		{ 2660, 2474, 0 },
		{ 2629, 1730, 0 },
		{ 2674, 3872, 0 },
		{ 2674, 3874, 0 },
		{ 2673, 1867, 0 },
		{ 2671, 2092, 0 },
		{ 2658, 2003, 0 },
		{ 2649, 1626, 0 },
		{ 2665, 1605, 0 },
		{ 2674, 3934, 0 },
		{ 2651, 2386, 0 },
		{ 2672, 1639, 0 },
		{ 2660, 2485, 0 },
		{ 2672, 1802, 0 },
		{ 2670, 1524, 0 },
		{ 2674, 3732, 0 },
		{ 2673, 768, 346 },
		{ 2665, 4077, 356 },
		{ 2574, 2266, 0 },
		{ 2670, 1548, 0 },
		{ 2671, 2205, 0 },
		{ 2381, 3998, 0 },
		{ 2669, 2426, 0 },
		{ 2665, 4108, 354 },
		{ 2629, 1739, 0 },
		{ 2533, 1762, 0 },
		{ 2670, 1553, 0 },
		{ 2673, 1881, 0 },
		{ 2671, 2212, 0 },
		{ 2665, 4118, 345 },
		{ 2674, 3868, 0 },
		{ 2390, 1279, 0 },
		{ 2670, 1558, 0 },
		{ 2666, 2051, 0 },
		{ 2629, 1744, 0 },
		{ 2660, 2508, 0 },
		{ 2651, 2305, 0 },
		{ 2672, 1815, 0 },
		{ 2649, 1645, 0 },
		{ 2672, 1818, 0 },
		{ 2665, 4161, 358 },
		{ 2674, 3936, 0 },
		{ 0, 1292, 0 },
		{ 2671, 2224, 0 },
		{ 2671, 2225, 0 },
		{ 2670, 1565, 0 },
		{ 2629, 1748, 0 },
		{ 2629, 1749, 0 },
		{ 2674, 3734, 0 },
		{ 2673, 1900, 0 },
		{ 2665, 4074, 347 },
		{ 2674, 3738, 0 },
		{ 2658, 2042, 0 },
		{ 2533, 1757, 0 },
		{ 2672, 1825, 0 },
		{ 2660, 2470, 0 },
		{ 2674, 3796, 0 },
		{ 2665, 4098, 341 },
		{ 2413, 4006, 0 },
		{ 2665, 4104, 348 },
		{ 2651, 2323, 0 },
		{ 2670, 1567, 0 },
		{ 2629, 1698, 0 },
		{ 2670, 1569, 0 },
		{ 2665, 4111, 359 },
		{ 2671, 1856, 0 },
		{ 2674, 3862, 0 },
		{ 2674, 3864, 0 },
		{ 2674, 3866, 0 },
		{ 2672, 1831, 0 },
		{ 2670, 1571, 0 },
		{ 2665, 4132, 352 },
		{ 2665, 4135, 353 },
		{ 2665, 4137, 355 },
		{ 2629, 1701, 0 },
		{ 2651, 2334, 0 },
		{ 2674, 3876, 0 },
		{ 2629, 1702, 0 },
		{ 2665, 4150, 357 },
		{ 2660, 2482, 0 },
		{ 2670, 1572, 0 },
		{ 2671, 2249, 0 },
		{ 2672, 1774, 0 },
		{ 2673, 1921, 0 },
		{ 2649, 1656, 0 },
		{ 2674, 3718, 0 },
		{ 2665, 4056, 362 },
		{ 2447, 3974, 371 },
		{ 2440, 0, 338 },
		{ 0, 0, 339 },
		{ -2438, 4166, 334 },
		{ -2439, 4014, 0 },
		{ 2665, 3996, 0 },
		{ 2447, 3977, 0 },
		{ 0, 0, 335 },
		{ 2447, 3995, 0 },
		{ -2444, 4173, 0 },
		{ -2445, 4018, 0 },
		{ 2448, 0, 336 },
		{ 0, 3997, 0 },
		{ 2665, 4084, 0 },
		{ 0, 0, 337 },
		{ 0, 2749, 107 },
		{ 0, 0, 107 },
		{ 0, 0, 108 },
		{ 2649, 1658, 0 },
		{ 2651, 2346, 0 },
		{ 2629, 1707, 0 },
		{ 2457, 4008, 0 },
		{ 2666, 2049, 0 },
		{ 2667, 1368, 0 },
		{ 2673, 1928, 0 },
		{ 2669, 2435, 0 },
		{ 2670, 1577, 0 },
		{ 2671, 2148, 0 },
		{ 2672, 1783, 0 },
		{ 2673, 1847, 0 },
		{ 2649, 1661, 0 },
		{ 2674, 3800, 0 },
		{ 0, 0, 105 },
		{ 2605, 3776, 126 },
		{ 0, 0, 126 },
		{ 2670, 1579, 0 },
		{ 2472, 4037, 0 },
		{ 2665, 2085, 0 },
		{ 2660, 2509, 0 },
		{ 2669, 2430, 0 },
		{ 2623, 2262, 0 },
		{ 2477, 4046, 0 },
		{ 2665, 1944, 0 },
		{ 2651, 2360, 0 },
		{ 2673, 1852, 0 },
		{ 2651, 2363, 0 },
		{ 2629, 1712, 0 },
		{ 2660, 2459, 0 },
		{ 2670, 1580, 0 },
		{ 2671, 2207, 0 },
		{ 2672, 1788, 0 },
		{ 2673, 1856, 0 },
		{ 2488, 4070, 0 },
		{ 2665, 2239, 0 },
		{ 2651, 2373, 0 },
		{ 2618, 2277, 0 },
		{ 2672, 1789, 0 },
		{ 2629, 1714, 0 },
		{ 2651, 2377, 0 },
		{ 2495, 3963, 0 },
		{ 2665, 1607, 0 },
		{ 2651, 2379, 0 },
		{ 2653, 2409, 0 },
		{ 2667, 1371, 0 },
		{ 2669, 2424, 0 },
		{ 2651, 2381, 0 },
		{ 2502, 3988, 0 },
		{ 2666, 2055, 0 },
		{ 2667, 1386, 0 },
		{ 2673, 1865, 0 },
		{ 2669, 2431, 0 },
		{ 2670, 1585, 0 },
		{ 2671, 2221, 0 },
		{ 2672, 1795, 0 },
		{ 2673, 1870, 0 },
		{ 2674, 3802, 0 },
		{ 0, 0, 124 },
		{ 2513, 0, 1 },
		{ -2513, 1229, 214 },
		{ 2651, 2291, 220 },
		{ 0, 0, 220 },
		{ 2649, 1672, 0 },
		{ 2673, 1875, 0 },
		{ 2651, 2396, 0 },
		{ 2653, 2416, 0 },
		{ 2629, 1723, 0 },
		{ 0, 0, 219 },
		{ 2523, 4052, 0 },
		{ 2665, 1615, 0 },
		{ 2660, 2503, 0 },
		{ 2552, 2056, 0 },
		{ 2651, 2400, 0 },
		{ 2618, 2286, 0 },
		{ 2671, 2237, 0 },
		{ 2658, 2012, 0 },
		{ 2651, 2301, 0 },
		{ 2532, 4041, 0 },
		{ 2672, 1647, 0 },
		{ 0, 1760, 0 },
		{ 2670, 1592, 0 },
		{ 2671, 2242, 0 },
		{ 2672, 1805, 0 },
		{ 2673, 1882, 0 },
		{ 2649, 1676, 0 },
		{ 2674, 3730, 0 },
		{ 0, 0, 218 },
		{ 0, 3650, 129 },
		{ 0, 0, 129 },
		{ 2672, 1807, 0 },
		{ 2667, 1394, 0 },
		{ 2673, 1886, 0 },
		{ 2653, 2406, 0 },
		{ 2548, 3972, 0 },
		{ 2669, 2236, 0 },
		{ 2623, 2254, 0 },
		{ 2651, 2316, 0 },
		{ 2669, 2440, 0 },
		{ 0, 2059, 0 },
		{ 2671, 2075, 0 },
		{ 2673, 1887, 0 },
		{ 2574, 2267, 0 },
		{ 2674, 3810, 0 },
		{ 0, 0, 127 },
		{ 2605, 3805, 123 },
		{ 0, 0, 123 },
		{ 2670, 1597, 0 },
		{ 2562, 3981, 0 },
		{ 2670, 1538, 0 },
		{ 2623, 2263, 0 },
		{ 2651, 2327, 0 },
		{ 2566, 4007, 0 },
		{ 2665, 2241, 0 },
		{ 2651, 2329, 0 },
		{ 2574, 2265, 0 },
		{ 2671, 2084, 0 },
		{ 2673, 1890, 0 },
		{ 2673, 1891, 0 },
		{ 2671, 2090, 0 },
		{ 2673, 1892, 0 },
		{ 0, 2271, 0 },
		{ 2576, 4011, 0 },
		{ 2672, 1652, 0 },
		{ 2618, 2289, 0 },
		{ 2579, 4025, 0 },
		{ 2665, 2083, 0 },
		{ 2660, 2491, 0 },
		{ 2669, 2428, 0 },
		{ 2623, 2255, 0 },
		{ 2584, 4032, 0 },
		{ 2665, 1942, 0 },
		{ 2651, 2343, 0 },
		{ 2673, 1895, 0 },
		{ 2651, 2345, 0 },
		{ 2629, 1733, 0 },
		{ 2660, 2502, 0 },
		{ 2670, 1603, 0 },
		{ 2671, 2167, 0 },
		{ 2672, 1814, 0 },
		{ 2673, 1899, 0 },
		{ 2595, 4053, 0 },
		{ 2666, 2053, 0 },
		{ 2667, 1399, 0 },
		{ 2673, 1901, 0 },
		{ 2669, 2425, 0 },
		{ 2670, 1609, 0 },
		{ 2671, 2208, 0 },
		{ 2672, 1817, 0 },
		{ 2673, 1904, 0 },
		{ 2674, 3808, 0 },
		{ 0, 0, 118 },
		{ 0, 3706, 117 },
		{ 0, 0, 117 },
		{ 2670, 1610, 0 },
		{ 2609, 3952, 0 },
		{ 2670, 1542, 0 },
		{ 2623, 2256, 0 },
		{ 2651, 2361, 0 },
		{ 2613, 3973, 0 },
		{ 2665, 2089, 0 },
		{ 2673, 1907, 0 },
		{ 2653, 2411, 0 },
		{ 2617, 3969, 0 },
		{ 2672, 1667, 0 },
		{ 0, 2297, 0 },
		{ 2620, 3984, 0 },
		{ 2665, 2097, 0 },
		{ 2660, 2469, 0 },
		{ 2669, 2442, 0 },
		{ 0, 2261, 0 },
		{ 2625, 3988, 0 },
		{ 2665, 1946, 0 },
		{ 2651, 2370, 0 },
		{ 2673, 1910, 0 },
		{ 2651, 2372, 0 },
		{ 0, 1740, 0 },
		{ 2660, 2477, 0 },
		{ 2670, 1613, 0 },
		{ 2671, 2222, 0 },
		{ 2672, 1823, 0 },
		{ 2673, 1914, 0 },
		{ 2636, 4010, 0 },
		{ 2666, 2047, 0 },
		{ 2667, 1402, 0 },
		{ 2673, 1916, 0 },
		{ 2669, 2436, 0 },
		{ 2670, 1615, 0 },
		{ 2671, 2229, 0 },
		{ 2672, 1826, 0 },
		{ 2673, 1919, 0 },
		{ 2674, 3792, 0 },
		{ 0, 0, 115 },
		{ 0, 3307, 120 },
		{ 0, 0, 120 },
		{ 2673, 1920, 0 },
		{ 0, 1690, 0 },
		{ 2670, 1616, 0 },
		{ 0, 2388, 0 },
		{ 2660, 2494, 0 },
		{ 0, 2405, 0 },
		{ 2655, 4038, 0 },
		{ 2665, 2095, 0 },
		{ 0, 1336, 0 },
		{ 2660, 2498, 0 },
		{ 0, 2038, 0 },
		{ 2667, 1404, 0 },
		{ 0, 2501, 0 },
		{ 2670, 1620, 0 },
		{ 2671, 2243, 0 },
		{ 2672, 1832, 0 },
		{ 2673, 1927, 0 },
		{ 2666, 4056, 0 },
		{ 2618, 2045, 0 },
		{ 0, 1405, 0 },
		{ 2673, 1929, 0 },
		{ 0, 2438, 0 },
		{ 0, 1622, 0 },
		{ 0, 2250, 0 },
		{ 0, 1835, 0 },
		{ 0, 1934, 0 },
		{ 0, 3946, 0 },
		{ 0, 0, 119 }
	};
	yystate = state;

	static const yybackup_t YYNEARFAR YYBASED_CODE backup[] = {
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		1,
		0,
		0,
		0,
		0,
		0,
		1,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0
	};
	yybackup = backup;
}
