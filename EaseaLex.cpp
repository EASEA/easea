#include <clex.h>

#line 1 "C:\\repo\\src\\EaseaLex.l"

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
  
#line 19 "C:\\repo\\src\\EaseaLex.cpp"
// repeated because of possible precompiled header
#include <clex.h>

#include "EaseaLex.h"

#line 22 "C:\\repo\\src\\EaseaLex.l"
                                         
#line 30 "C:\\repo\\src\\EaseaLex.l"
 // lexical analyser name and class definition
#line 29 "C:\\repo\\src\\EaseaLex.cpp"
/////////////////////////////////////////////////////////////////////////////
// constructor

YYLEXNAME::YYLEXNAME()
{
	yytables();
#line 48 "C:\\repo\\src\\EaseaLex.l"
                                
  bFunction=bWithinEvaluator=bDisplayFunction=bInitFunction=bNotFinishedYet=0;
  bSymbolInserted=bDoubleQuotes=bWithinDisplayFunction=bWithinInitialiser=bWithinMutator=bWithinXover=0;
  bWaitingForSemiColon=bFinishNB_GEN=bFinishMINIMISE=bFinishMINIMIZE=bGenerationFunction=0;
  bCatchNextSemiColon,bWaitingToClosePopulation=bMethodsInGenome=0;

#line 43 "C:\\repo\\src\\EaseaLex.cpp"
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
#line 60 "C:\\repo\\src\\EaseaLex.l"

  // extract yylval for use later on in actions
  YYSTYPE& yylval = *(YYSTYPE*)yyparserptr->yylvalptr;
  
#line 94 "C:\\repo\\src\\EaseaLex.cpp"
	yyreturnflg = 1;
	switch (action) {
	case 1:
		{
#line 66 "C:\\repo\\src\\EaseaLex.l"

#line 101 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 2:
		{
#line 69 "C:\\repo\\src\\EaseaLex.l"

  BEGIN TEMPLATE_ANALYSIS; yyless(yyleng-1);
 
#line 110 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 3:
		{
#line 77 "C:\\repo\\src\\EaseaLex.l"
             
  char sFileName[1000];
  strcpy(sFileName, sRAW_PROJECT_NAME);
  switch (TARGET) {
  case DREAM : strcat(sFileName,".java"); break;
  default : strcat(sFileName,".cpp");
  } 
  fpOutputFile=fopen(sFileName,"w");
 
#line 125 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 4:
		{
#line 86 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"EASEA");
#line 132 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 5:
		{
#line 87 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sPROJECT_NAME);
#line 139 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 6:
		{
#line 88 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sEZ_PATH);
#line 146 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 7:
		{
#line 89 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sEO_DIR);
#line 153 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 8:
		{
#line 90 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sLOWER_CASE_PROJECT_NAME);
#line 160 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 9:
		{
#line 91 "C:\\repo\\src\\EaseaLex.l"
switch (OPERATING_SYSTEM) {
  case UNIX : fprintf(fpOutputFile,"UNIX_OS"); break;
  case WINDOWS : fprintf(fpOutputFile,"WINDOWS_OS"); break;
  case UNKNOWN_OS : fprintf(fpOutputFile,"UNKNOWN_OS"); break;
  }
 
#line 172 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 10:
		{
#line 97 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user declarations.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_USER_DECLARATIONS;
 
#line 184 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 11:
		{
#line 103 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf ("Inserting initialisation function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_INITIALISATION_FUNCTION;
 
#line 196 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 12:
		{
#line 109 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf ("Inserting generation function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_GENERATION_FUNCTION;
 
#line 208 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 13:
		{
#line 115 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 
#line 220 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 14:
		{
#line 121 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf ("Inserting User classes.\n");
  if (TARGET!=DREAM) fprintf (fpOutputFile,"// User classes\n");
  CListItem<CSymbol*> *pSym;
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem())
    if ((pSym->Object->pType->ObjectType==oUserClass)&&(!pSym->Object->pType->bAlreadyPrinted))
      pSym->Object->pType->printClasses(fpOutputFile);
 
#line 235 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 15:
		{
#line 130 "C:\\repo\\src\\EaseaLex.l"

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
 
#line 255 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 16:
		{
#line 144 "C:\\repo\\src\\EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 270 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 17:
		{
#line 153 "C:\\repo\\src\\EaseaLex.l"

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
 
#line 293 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 18:
		{
#line 170 "C:\\repo\\src\\EaseaLex.l"

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
 
#line 329 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 19:
		{
#line 201 "C:\\repo\\src\\EaseaLex.l"

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
 
#line 365 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 20:
		{
#line 231 "C:\\repo\\src\\EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default destructor.\n");
  fprintf (fpOutputFile,"// Destructing pointers\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem())
    if (pSym->Object->ObjectType==oPointer){
      if (TARGET==DREAM) fprintf(fpOutputFile,"  if (%s) delete %s;\n  %s=null;\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
      else fprintf(fpOutputFile,"  if (%s) delete %s;\n  %s=NULL;\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->sName);
    }
 
#line 382 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 21:
		{
#line 242 "C:\\repo\\src\\EaseaLex.l"
       
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
 
#line 404 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 22:
		{
#line 258 "C:\\repo\\src\\EaseaLex.l"

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
 
#line 426 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 23:
		{
#line 274 "C:\\repo\\src\\EaseaLex.l"

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
 
#line 461 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 24:
		{
#line 303 "C:\\repo\\src\\EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 473 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 25:
		{
#line 309 "C:\\repo\\src\\EaseaLex.l"

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
 
#line 535 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 26:
		{
#line 365 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 547 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 27:
		{
#line 371 "C:\\repo\\src\\EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 559 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 28:
		{
#line 377 "C:\\repo\\src\\EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_INITIALISER;   
 
#line 570 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 29:
		{
#line 382 "C:\\repo\\src\\EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_CROSSOVER;   
 
#line 581 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 30:
		{
#line 387 "C:\\repo\\src\\EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_MUTATOR;   
 
#line 592 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 31:
		{
#line 392 "C:\\repo\\src\\EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EVALUATOR;   
 
#line 603 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 32:
		{
#line 397 "C:\\repo\\src\\EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 614 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 33:
		{
#line 402 "C:\\repo\\src\\EaseaLex.l"

  if (bGenerationFunction) fprintf(fpOutputFile,"\n    EASEAGenerationFunction(ga);\n");
 
#line 623 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 34:
		{
#line 405 "C:\\repo\\src\\EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 632 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 35:
		{
#line 408 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR);
#line 639 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 36:
		{
#line 410 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECT_PRM);
#line 646 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 37:
		{
#line 411 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 653 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 38:
		{
#line 412 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 660 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 39:
		{
#line 413 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nELITE);
#line 667 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 40:
		{
#line 414 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR);
#line 674 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 41:
		{
#line 415 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_PRM);
#line 681 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 42:
		{
#line 416 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF);
#line 688 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 43:
		{
#line 417 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_PRM);
#line 695 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 44:
		{
#line 418 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL);
#line 702 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 45:
		{
#line 419 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_PRM);
#line 709 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 46:
		{
#line 420 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nSURV_PAR_SIZE);
#line 716 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 47:
		{
#line 421 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nSURV_OFF_SIZE);
#line 723 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 48:
		{
#line 422 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 730 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 49:
		{
#line 423 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_ISLANDS);
#line 737 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 50:
		{
#line 424 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",bPROP_SEQ?"Prop":"Seq");
#line 744 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 51:
		{
#line 425 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB)fprintf(fpOutputFile,"(float)%f",fMUT_PROB);
  else fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 752 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 52:
		{
#line 427 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB)fprintf(fpOutputFile,"(float)%f",fXOVER_PROB);
  else fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 760 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 53:
		{
#line 429 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sREPLACEMENT);
#line 767 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 54:
		{
#line 430 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"(float)%f/100",fREPL_PERC);
#line 774 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 55:
		{
#line 431 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sDISCARD_PRM);
#line 781 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 56:
		{
#line 432 "C:\\repo\\src\\EaseaLex.l"
if ((fREPL_PERC==0)||(!strcmp(sREPLACEMENT,"Incremental"))||(!strcmp(sREPLACEMENT,"Simple")))
     fprintf(fpOutputFile,"// undefined ");
#line 789 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 57:
		{
#line 434 "C:\\repo\\src\\EaseaLex.l"
if (mystricmp(sREPLACEMENT,"SSGA")) fprintf(fpOutputFile,"//");
#line 796 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 58:
		{
#line 435 "C:\\repo\\src\\EaseaLex.l"
switch (TARGET) { case GALIB : fprintf(fpOutputFile,"%d",nMINIMISE? -1:1); break;
  case EO : fprintf(fpOutputFile,"%s",nMINIMISE? "eoMinimizingFitness" : "eoMaximizingFitness"); break;
  case DREAM : fprintf(fpOutputFile,"%s",nMINIMISE? "false" : "true"); break;
  }                                  
#line 806 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 59:
		{
#line 439 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==EO) {if (bELITISM) fprintf(fpOutputFile,"standardR");
    else fprintf(fpOutputFile,"r");}
#line 814 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 60:
		{
#line 441 "C:\\repo\\src\\EaseaLex.l"
switch (TARGET) { case GALIB : if (!mystricmp(sREPLACEMENT,"simple"))
       if (bELITISM) fprintf(fpOutputFile,"  ga.elitist(gaTrue);\n");
       else fprintf(fpOutputFile,"  ga.elitist(gaFalse);\n");
      break;
  case EO :  fprintf(fpOutputFile,"%d",bELITISM);
  }                                 
#line 826 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 447 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sIMMIG_SEL);
#line 833 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 448 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nIMMIG_REPL);
#line 840 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 449 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sMIGRATOR);
#line 847 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 450 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sMIG_SEL);
#line 854 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 451 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nMIG_CLONE);
#line 861 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 452 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_MIG);
#line 868 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 453 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%f",fMIG_FREQ);
#line 875 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 68:
		{
#line 454 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",sMIG_TARGET_SELECTOR);
#line 882 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 456 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Genome.h");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 896 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 464 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"EvalFunc.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 910 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 472 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Init.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 924 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 480 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Mutation.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 938 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 488 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"QuadCrossover.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 952 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 496 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"_make_continue.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 966 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 504 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"_make_genotype.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 980 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 512 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"_make_operators.h");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 994 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 520 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1008 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 528 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1022 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 536 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Evaluator.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1036 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 544 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Initer.java");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1050 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 552 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Genome.java");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1064 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 560 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Mutator.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1078 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 568 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Recombine.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1092 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 576 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"IHDefIniter.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1106 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 584 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"FitnessIniter.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1120 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 592 "C:\\repo\\src\\EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Island.java");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1134 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 600 "C:\\repo\\src\\EaseaLex.l"

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
 
#line 1162 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 623 "C:\\repo\\src\\EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1169 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 629 "C:\\repo\\src\\EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  BEGIN COPY;
 
#line 1179 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 633 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1192 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 640 "C:\\repo\\src\\EaseaLex.l"

#line 1199 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 642 "C:\\repo\\src\\EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  BEGIN COPY;
 
#line 1209 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 646 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1222 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 653 "C:\\repo\\src\\EaseaLex.l"

#line 1229 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 655 "C:\\repo\\src\\EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  BEGIN COPY;
 
#line 1239 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 659 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1253 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 667 "C:\\repo\\src\\EaseaLex.l"

#line 1260 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 669 "C:\\repo\\src\\EaseaLex.l"

  if (TARGET==DREAM)
    fprintf (fpOutputFile,"// Evaluator Constructor\n\n  public %sEvaluator(){",sPROJECT_NAME);
  else
    fprintf (fpOutputFile,"// Initialisation function\n\nvoid EASEAInitFunction(int argc, char *argv[]){");
  bFunction=1; bInitFunction=1;
      
  BEGIN COPY;
 
#line 1275 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 678 "C:\\repo\\src\\EaseaLex.l"
bInitFunction=0; // No initialisation function was found in the .ez file
  if (bVERBOSE) printf("*** No initialisation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No initialisation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1289 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 686 "C:\\repo\\src\\EaseaLex.l"

#line 1296 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 688 "C:\\repo\\src\\EaseaLex.l"

  fprintf (fpOutputFile,"// Function called at each new generation\n\nvoid EASEAGenerationFunction(GAGeneticAlgorithm & g){\n");
  fprintf(fpOutputFile,"  const GAPopulation *pPopulation;\n",sPROJECT_NAME);
  fprintf(fpOutputFile,"  pPopulation=&(g.population());  // to circumvent a bug in GALib\n",sPROJECT_NAME);
  bFunction=1; bGenerationFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 1309 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 695 "C:\\repo\\src\\EaseaLex.l"
bGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1323 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 703 "C:\\repo\\src\\EaseaLex.l"

#line 1330 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 705 "C:\\repo\\src\\EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 1338 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 707 "C:\\repo\\src\\EaseaLex.l"

#line 1345 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 713 "C:\\repo\\src\\EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 1352 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 714 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 1359 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 108:
	case 109:
		{
#line 717 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"EZ_currentGeneration"); break;
    case EO : fprintf(fpOutputFile,"generationCounter.value()");
    }
#line 1371 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 110:
	case 111:
		{
#line 723 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"EZ_NB_GEN");
#line 1380 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 112:
	case 113:
		{
#line 726 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 1389 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 114:
	case 115:
		{
#line 729 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
#line 1398 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 116:
	case 117:
		{
#line 732 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 1407 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 118:
	case 119:
		{
#line 735 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 1416 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 120:
	case 121:
		{
#line 738 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 1425 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 122:
	case 123:
		{
#line 741 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 1434 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 744 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 1441 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 745 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1448 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 746 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1455 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 747 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1462 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 748 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1469 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 749 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1476 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 750 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1483 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 751 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 1490 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 752 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB) fprintf(fpOutputFile,"gaFalse");
  else fprintf(fpOutputFile,"false");
#line 1498 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 754 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB) fprintf(fpOutputFile,"gaTrue");
  else fprintf(fpOutputFile,"true");
#line 1506 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 756 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==DREAM) fprintf(fpOutputFile," boolean ");
  else fprintf(fpOutputFile,yytext);
#line 1514 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 758 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB) fprintf(fpOutputFile,"tossCoin");
  if (TARGET==EO) fprintf(fpOutputFile,"rng.flip");
  if (TARGET==DREAM) fprintf(fpOutputFile,"Math.random()<");
 
#line 1524 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 762 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==DREAM) fprintf(fpOutputFile,"%s.random",sPROJECT_NAME);
  else fprintf(fpOutputFile,"random");
#line 1532 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 764 "C:\\repo\\src\\EaseaLex.l"
if (bWithinEO_Function) fprintf(fpOutputFile,"_genotype");
  else fprintf(fpOutputFile,"Genome");
#line 1540 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 766 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 1547 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 767 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 1554 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 768 "C:\\repo\\src\\EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 1565 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 773 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 1572 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 774 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==EO) fprintf(fpOutputFile,"Indi");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 1580 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 776 "C:\\repo\\src\\EaseaLex.l"
if (bFunction==1) {fprintf (fpOutputFile,"}\n"); bFunction=0;}
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 1590 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 780 "C:\\repo\\src\\EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1597 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 783 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 1606 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 786 "C:\\repo\\src\\EaseaLex.l"
BEGIN COPY;
#line 1613 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 788 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 1620 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 148:
	case 149:
	case 150:
		{
#line 791 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 1633 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 796 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 1644 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 801 "C:\\repo\\src\\EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 1653 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 810 "C:\\repo\\src\\EaseaLex.l"
;
#line 1660 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 811 "C:\\repo\\src\\EaseaLex.l"
;
#line 1667 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 812 "C:\\repo\\src\\EaseaLex.l"
;
#line 1674 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 813 "C:\\repo\\src\\EaseaLex.l"
;
#line 1681 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 816 "C:\\repo\\src\\EaseaLex.l"
 /* do nothing */ 
#line 1688 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 817 "C:\\repo\\src\\EaseaLex.l"
 /*return '\n';*/ 
#line 1695 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 818 "C:\\repo\\src\\EaseaLex.l"
 /*return '\n';*/ 
#line 1702 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 821 "C:\\repo\\src\\EaseaLex.l"
if (TARGET!=DREAM) yylval.pSymbol = pSymbolTable->find("bool");
  else yylval.pSymbol = pSymbolTable->find("boolean");
  return BOOL;
#line 1711 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 824 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==DREAM){
    yylval.pSymbol = pSymbolTable->find("boolean");
    return BOOL;
  }
  else {
    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
  }  
#line 1725 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 832 "C:\\repo\\src\\EaseaLex.l"
return STATIC;
#line 1732 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 833 "C:\\repo\\src\\EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 1739 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 834 "C:\\repo\\src\\EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 1746 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 835 "C:\\repo\\src\\EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 1753 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 836 "C:\\repo\\src\\EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 1760 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 837 "C:\\repo\\src\\EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 1767 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 838 "C:\\repo\\src\\EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 1774 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
#line 839 "C:\\repo\\src\\EaseaLex.l"
  
#line 1779 "C:\\repo\\src\\EaseaLex.cpp"
	case 169:
		{
#line 840 "C:\\repo\\src\\EaseaLex.l"
return GENOME; 
#line 1784 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 842 "C:\\repo\\src\\EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 1794 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 171:
	case 172:
	case 173:
		{
#line 849 "C:\\repo\\src\\EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 1803 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 850 "C:\\repo\\src\\EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 1810 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 853 "C:\\repo\\src\\EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 1818 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 855 "C:\\repo\\src\\EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 1825 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 861 "C:\\repo\\src\\EaseaLex.l"
 
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 1835 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 865 "C:\\repo\\src\\EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1848 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 872 "C:\\repo\\src\\EaseaLex.l"

#line 1855 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 180:
		{
#line 878 "C:\\repo\\src\\EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  return USER_CTOR;
 
#line 1866 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 883 "C:\\repo\\src\\EaseaLex.l"

#line 1873 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 182:
		{
#line 884 "C:\\repo\\src\\EaseaLex.l"

  bWithinXover=1;
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 1884 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 183:
		{
#line 889 "C:\\repo\\src\\EaseaLex.l"

#line 1891 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 184:
		{
#line 890 "C:\\repo\\src\\EaseaLex.l"

  bWithinMutator=1;
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 1902 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 185:
		{
#line 895 "C:\\repo\\src\\EaseaLex.l"

#line 1909 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 186:
		{
#line 896 "C:\\repo\\src\\EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  return USER_EVALUATOR;
 
#line 1920 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 187:
		{
#line 901 "C:\\repo\\src\\EaseaLex.l"

#line 1927 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 188:
		{
#line 907 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 1934 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 189:
		{
#line 908 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 1941 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 190:
		{
#line 909 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 1948 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 191:
		{
#line 910 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 1955 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 192:
		{
#line 911 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 1962 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 193:
		{
#line 912 "C:\\repo\\src\\EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 1969 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 194:
		{
#line 913 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 1976 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 195:
		{
#line 915 "C:\\repo\\src\\EaseaLex.l"
bWaitingToClosePopulation=true;
  switch (TARGET) {
  case EO: fprintf(fpOutputFile,"pPopulation ["); break;
  case GALIB: fprintf(fpOutputFile,"((%sGenome *)&(pPopulation->individual(",sPROJECT_NAME);
  }
 
#line 1988 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 196:
		{
#line 921 "C:\\repo\\src\\EaseaLex.l"
if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {fprintf(fpOutputFile,")))"); bWaitingToClosePopulation=false;}
#line 1996 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 197:
	case 198:
		{
#line 925 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.generation()"); break;
    case EO : fprintf(fpOutputFile,"generationCounter.value()");
    }
#line 2008 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 199:
	case 200:
		{
#line 931 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.nGenerations()"); break;
    case EO : fprintf(fpOutputFile,"ptEZ_NbGen.value()");
    }
#line 2020 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 201:
	case 202:
		{
#line 937 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.populationSize()"); break;
    case EO : fprintf(fpOutputFile,"EZ_POP_SIZE");
    }
#line 2032 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 203:
	case 204:
		{
#line 943 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.pMutation()"); break;
    case EO : fprintf(fpOutputFile,"EZ_MUT_PROB");
    }
#line 2044 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 205:
	case 206:
		{
#line 949 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.pCrossover()"); break;
    case EO : fprintf(fpOutputFile,"EZ_XOVER_PROB");
    }
#line 2056 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 207:
	case 208:
		{
#line 955 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.pReplacement()*100"); break;
    case EO : fprintf(stderr,"\n%s - Error line %d: The REPL_PERC variable cannot be accessed yet undeer EO.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }
#line 2069 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 209:
	case 210:
		{
#line 962 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.minimaxi()"); break;
    case EO : fprintf(stderr,"\n%s - Error line %d: The MINIMISE variable cannot be accessed yet undeer EO.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }
#line 2082 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 211:
	case 212:
		{
#line 969 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.minimaxi()"); break;
    case EO : fprintf(stderr,"\n%s - Error line %d: The MINIMIZE variable cannot be accessed yet undeer EO.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }
#line 2095 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 213:
		{
#line 976 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 2104 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 214:
		{
#line 979 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.nGenerations((EZ_NB_GEN=");
      bWaitingForSemiColon=bFinishNB_GEN=1; break;
    case EO : fprintf(fpOutputFile,"ptEZ_NbGen.value((EZ_NB_GEN=)");
      bWaitingForSemiColon=1;
    }
#line 2117 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 986 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.populationSize((EZ_POP_SIZE=");
      bWaitingForSemiColon=1; break;
    case EO : fprintf(stderr,"\n%s - Error line %d: The population size cannot be changed during the run in EO yet.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }
#line 2130 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 993 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"g.pMutation((EZ_MUT_PROB=");
      bWaitingForSemiColon=1; break;
    case EO : fprintf(fpOutputFile,"EZ_MUT_PROB=");
    }
#line 2142 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 999 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB :fprintf(fpOutputFile,"g.pCrossover((EZ_XOVER_PROB=");
      bWaitingForSemiColon=1; break;
    case EO : fprintf(fpOutputFile,"EZ_MUT_PROB=");
    }
#line 2154 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1005 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB :fprintf(fpOutputFile,"g.pReplacement((1/100)*(EZ_REPL_PERC=");
      bWaitingForSemiColon=1; break;
    case EO : fprintf(stderr,"\n%s - Error line %d: REPL_PERC cannot be changed during the run in EO yet.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }
#line 2167 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1012 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB :fprintf(fpOutputFile,"MINIMISE=");
      bWaitingForSemiColon=bFinishMINIMISE=1; break;
    case EO : fprintf(stderr,"\n%s - Error line %d: MINIMISE cannot be changed during the run in EO yet.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }
#line 2180 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1019 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else switch  (TARGET) {
    case GALIB :fprintf(fpOutputFile,"MINIMIZE=");
      bWaitingForSemiColon=bFinishMINIMISE=1; break;
    case EO : fprintf(stderr,"\n%s - Error line %d: MINIMIZE cannot be changed during the run in EO yet.\n",sEZ_FILE_NAME,yylineno);
      exit(1);
    }
#line 2193 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1026 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB) fprintf(fpOutputFile,"gaFalse");
  else fprintf(fpOutputFile,"false");
#line 2201 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1028 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB) fprintf(fpOutputFile,"gaTrue");
  else fprintf(fpOutputFile,"true");
#line 2209 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1030 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==DREAM) fprintf(fpOutputFile," boolean ");
  else fprintf(fpOutputFile,yytext);
#line 2217 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1032 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB) fprintf(fpOutputFile,"tossCoin");
  if (TARGET==EO) fprintf(fpOutputFile,"rng.flip");
  if (TARGET==DREAM) fprintf(fpOutputFile,"Math.random()<");
 
#line 2227 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1036 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==DREAM) fprintf(fpOutputFile,"%s.random",sPROJECT_NAME);
  else fprintf(fpOutputFile,"random");
#line 2235 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1038 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2242 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1039 "C:\\repo\\src\\EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 2252 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1043 "C:\\repo\\src\\EaseaLex.l"
if (bWaitingForSemiColon){
    bWaitingForSemiColon=0;
    if (bFinishMINIMISE) {fprintf(fpOutputFile,");\n  if (MINIMISE) g.minimize() else g.maximize();\n"); bFinishMINIMISE=0;}
    if (bFinishMINIMIZE) {fprintf(fpOutputFile,");\n  if (MINIMIZE) g.minimize() else g.maximize();\n"); bFinishMINIMISE=0;}
    if ((bFinishNB_GEN)&&(OPERATING_SYSTEM==UNIX))
      {fprintf(fpOutputFile,"));\n  if ( (EZ_daFITNESS = (double *) realloc(EZ_daFITNESS, (EZ_NB_GEN +1)* sizeof (double) )) == NULL){\n");
	fprintf(fpOutputFile,"    fprintf(stderr,\"Not enough memory... bailing out.\");\n    exit(1);");}
    else if (bFinishNB_GEN) {fprintf(fpOutputFile,"));"); bFinishNB_GEN=0;}
    else fprintf(fpOutputFile,"));");
  }
  else fprintf(fpOutputFile,";");
#line 2269 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1055 "C:\\repo\\src\\EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2276 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1062 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2283 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1063 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2290 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1064 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2297 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1065 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 2304 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1067 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==EO) fprintf(fpOutputFile, "GenotypeT");
  else fprintf(fpOutputFile,"Genome.");
 
#line 2313 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1070 "C:\\repo\\src\\EaseaLex.l"
if (bWithinDisplayFunction) fprintf(fpOutputFile,"(*this)");
  else if ((TARGET==EO)&&(bWithinInitialiser)) fprintf(fpOutputFile, "(*genome)");
  else if ((TARGET==EO)&&(bWithinMutator)) fprintf(fpOutputFile, "_genotype");
  else fprintf(fpOutputFile,"genome");
#line 2323 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1074 "C:\\repo\\src\\EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2330 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1075 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2337 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1077 "C:\\repo\\src\\EaseaLex.l"
bWaitingToClosePopulation=true;
  switch (TARGET) {
  case EO: fprintf(fpOutputFile,"pPopulation ["); break;
  case GALIB: fprintf(fpOutputFile,"((%sGenome *)&(pPopulation->individual(",sPROJECT_NAME);
  }
 
#line 2349 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1083 "C:\\repo\\src\\EaseaLex.l"
if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {fprintf(fpOutputFile,")))"); bWaitingToClosePopulation=false;}
#line 2357 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 240:
	case 241:
		{
#line 1087 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case GALIB : fprintf(fpOutputFile,"EZ_currentGeneration"); break;
    case EO : fprintf(fpOutputFile,"generationCounter.value()");
    }
#line 2369 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 242:
	case 243:
		{
#line 1093 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"EZ_NB_GEN");
#line 2378 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 244:
	case 245:
		{
#line 1096 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2387 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 246:
	case 247:
		{
#line 1099 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
#line 2396 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 248:
	case 249:
		{
#line 1102 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 2405 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 250:
	case 251:
		{
#line 1105 "C:\\repo\\src\\EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2414 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1108 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2421 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 253:
		{
#line 1109 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2428 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1110 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2435 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1111 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2442 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1112 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2449 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1113 "C:\\repo\\src\\EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2456 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1114 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB) fprintf(fpOutputFile,"gaFalse");
  else fprintf(fpOutputFile,"false");
#line 2464 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1116 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB) fprintf(fpOutputFile,"gaTrue");
  else fprintf(fpOutputFile,"true");
#line 2472 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1118 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==DREAM) fprintf(fpOutputFile," boolean ");
  else fprintf(fpOutputFile,yytext);
#line 2480 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1120 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==GALIB) fprintf(fpOutputFile,"tossCoin");
  if (TARGET==EO) fprintf(fpOutputFile,"rng.flip");
  if (TARGET==DREAM) fprintf(fpOutputFile,"Math.random()<");
#line 2489 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1123 "C:\\repo\\src\\EaseaLex.l"
if (TARGET==DREAM) fprintf(fpOutputFile,"%s.random",sPROJECT_NAME);
  else fprintf(fpOutputFile,"random");
#line 2497 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1125 "C:\\repo\\src\\EaseaLex.l"
if ((bWithinXover)&&(TARGET==GALIB)) fprintf(fpOutputFile, "(*pBro)");
  else fprintf(fpOutputFile,"child1");
 
#line 2506 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1128 "C:\\repo\\src\\EaseaLex.l"
if ((bWithinXover)&&(TARGET==GALIB)) fprintf(fpOutputFile, "(*pSis)");
  else fprintf(fpOutputFile,"child2");
 
#line 2515 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1131 "C:\\repo\\src\\EaseaLex.l"
if ((bWithinXover)&&(TARGET==GALIB)) fprintf(fpOutputFile, "(*pDad)");
  else fprintf(fpOutputFile,"parent1");
 
#line 2524 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1134 "C:\\repo\\src\\EaseaLex.l"
if ((bWithinXover)&&(TARGET==GALIB)) fprintf(fpOutputFile, "(*pMom)");
  else fprintf(fpOutputFile,"parent2");
 
#line 2533 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1137 "C:\\repo\\src\\EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2540 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1138 "C:\\repo\\src\\EaseaLex.l"
if (bWithinEvaluator) {
    if (TARGET==GALIB) fprintf(fpOutputFile,"EZ_EVAL+=(double)(clock()-EZ_t1);\n  return (float)");
    if (TARGET==EO)  {fprintf(fpOutputFile,"genome.fitness(");bCatchNextSemiColon=true;}// changes function type// changes function type
    if (TARGET==DREAM) {fprintf(fpOutputFile,"infoHabitant.setFitness(new Double(");bCatchNextSemiColon=true;}// changes function type
    bWithinEvaluator=0;
  }
  else if ((bWithinMutator)&&(TARGET!=GALIB)) {
    fprintf(fpOutputFile,"return ");
    bCatchNextSemiColon=true;
  }
  else fprintf(fpOutputFile,"return"); 
#line 2557 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1149 "C:\\repo\\src\\EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if ((TARGET!=GALIB)&&(bWithinMutator)){fprintf(fpOutputFile,">0?true:false;"); bWithinMutator=false;}
  else if (TARGET==EO) fprintf(fpOutputFile,");");
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 2569 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1155 "C:\\repo\\src\\EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=0;
  if (bWithinDisplayFunction) bWithinDisplayFunction=0; // display function
  else return END_OF_FUNCTION;
#line 2581 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1161 "C:\\repo\\src\\EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2588 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1167 "C:\\repo\\src\\EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 2598 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1171 "C:\\repo\\src\\EaseaLex.l"

#line 2605 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1174 "C:\\repo\\src\\EaseaLex.l"
;
#line 2612 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1175 "C:\\repo\\src\\EaseaLex.l"
;
#line 2619 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1176 "C:\\repo\\src\\EaseaLex.l"
;
#line 2626 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1177 "C:\\repo\\src\\EaseaLex.l"
;
#line 2633 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1179 "C:\\repo\\src\\EaseaLex.l"
 /* do nothing */ 
#line 2640 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1180 "C:\\repo\\src\\EaseaLex.l"
 /*return '\n';*/ 
#line 2647 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1181 "C:\\repo\\src\\EaseaLex.l"
 /*return '\n';*/ 
#line 2654 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1183 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 2661 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1184 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Isl...\n");return NB_ISLANDS;
#line 2668 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1185 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tOp Seq...\n");return PROP_SEQ;
#line 2675 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1186 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tEvol Engine...\n");return REPLACEMENT;
#line 2682 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1187 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 2689 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1188 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 2696 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1189 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tFertil...\n");
#line 2703 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1190 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tGenitors Sel...\n");return SELECTOR;
#line 2710 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1191 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 2717 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1192 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 2724 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1193 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 2731 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1194 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 2738 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1195 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tReduce Par...\n");return RED_PAR;
#line 2745 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1196 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 2752 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 295:
		{
#line 1197 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tRed Off...\n");return RED_OFF;
#line 2759 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 296:
		{
#line 1198 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 2766 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 297:
		{
#line 1199 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Red...\n");return RED_FINAL;
#line 2773 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 298:
		{
#line 1200 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 2780 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 299:
		{
#line 1201 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 2787 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 300:
		{
#line 1202 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tMig Policy...\n");return MIG_CLONE;
#line 2794 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 301:
		{
#line 1203 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tMig Sel...\n");return MIG_SEL;
#line 2801 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 302:
		{
#line 1204 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tMig Dest...\n");return MIGRATOR;
#line 2808 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 303:
		{
#line 1205 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tMig Freq...\n");return MIG_FREQ;
#line 2815 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 304:
		{
#line 1206 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tNb Mig...\n");return NB_MIG;
#line 2822 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 305:
		{
#line 1207 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tImmig Repl...\n");return IMMIG_SEL;
#line 2829 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 306:
		{
#line 1208 "C:\\repo\\src\\EaseaLex.l"
if (bVERBOSE) printf ("\tImmig Policy...\n");return IMMIG_REPL;
#line 2836 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
#line 1209 "C:\\repo\\src\\EaseaLex.l"
 
#line 2841 "C:\\repo\\src\\EaseaLex.cpp"
	case 307:
	case 308:
	case 309:
		{
#line 1213 "C:\\repo\\src\\EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 2848 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 310:
		{
#line 1214 "C:\\repo\\src\\EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 2855 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 311:
		{
#line 1217 "C:\\repo\\src\\EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 2863 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 312:
		{
#line 1220 "C:\\repo\\src\\EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 2870 "C:\\repo\\src\\EaseaLex.cpp"
		}
		break;
	case 313:
		{
#line 1227 "C:\\repo\\src\\EaseaLex.l"
return  (char)yytext[0];
#line 2877 "C:\\repo\\src\\EaseaLex.cpp"
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
#line 1229 "C:\\repo\\src\\EaseaLex.l"


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

#line 3064 "C:\\repo\\src\\EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		116,
		-117,
		0,
		108,
		-109,
		0,
		110,
		-111,
		0,
		112,
		-113,
		0,
		114,
		-115,
		0,
		120,
		-121,
		0,
		122,
		-123,
		0,
		118,
		-119,
		0,
		-146,
		0,
		-152,
		0,
		207,
		-208,
		0,
		248,
		-249,
		0,
		205,
		-206,
		0,
		244,
		-245,
		0,
		246,
		-247,
		0,
		242,
		-243,
		0,
		250,
		-251,
		0,
		197,
		-198,
		0,
		240,
		-241,
		0,
		201,
		-202,
		0,
		203,
		-204,
		0,
		209,
		-210,
		0,
		211,
		-212,
		0,
		199,
		-200,
		0
	};
	yymatch = match;

	yytransitionmax = 3662;
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
		{ 1460, 32 },
		{ 1134, 1126 },
		{ 1135, 1126 },
		{ 1088, 1088 },
		{ 1284, 32 },
		{ 1792, 1773 },
		{ 1792, 1773 },
		{ 1825, 1806 },
		{ 1825, 1806 },
		{ 1857, 1838 },
		{ 1857, 1838 },
		{ 48, 3 },
		{ 49, 3 },
		{ 1680, 37 },
		{ 1681, 37 },
		{ 0, 1 },
		{ 2092, 2094 },
		{ 0, 1305 },
		{ 45, 1 },
		{ 137, 139 },
		{ 1646, 1642 },
		{ 0, 64 },
		{ 1465, 1467 },
		{ 1460, 32 },
		{ 1134, 1126 },
		{ 1287, 32 },
		{ 1088, 1088 },
		{ 1663, 1662 },
		{ 1792, 1773 },
		{ 1465, 1461 },
		{ 1825, 1806 },
		{ 2092, 2088 },
		{ 1857, 1838 },
		{ 1470, 1468 },
		{ 48, 3 },
		{ 1765, 1742 },
		{ 1680, 37 },
		{ 1765, 1742 },
		{ 1459, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 47, 3 },
		{ 1300, 32 },
		{ 1679, 37 },
		{ 1144, 1139 },
		{ 1136, 1126 },
		{ 1793, 1773 },
		{ 126, 124 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1286, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1294, 32 },
		{ 1292, 32 },
		{ 1303, 32 },
		{ 1293, 32 },
		{ 1303, 32 },
		{ 1296, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1295, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1049, 1033 },
		{ 1288, 32 },
		{ 1290, 32 },
		{ 1050, 1034 },
		{ 1303, 32 },
		{ 1051, 1035 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1291, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1297, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1302, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1289, 32 },
		{ 1303, 32 },
		{ 1301, 32 },
		{ 1303, 32 },
		{ 1298, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1303, 32 },
		{ 1641, 34 },
		{ 1077, 1064 },
		{ 1078, 1064 },
		{ 1041, 1024 },
		{ 1473, 34 },
		{ 1832, 1813 },
		{ 1832, 1813 },
		{ 1858, 1839 },
		{ 1858, 1839 },
		{ 1860, 1841 },
		{ 1860, 1841 },
		{ 1299, 31 },
		{ 1044, 1027 },
		{ 1488, 33 },
		{ 1861, 1842 },
		{ 1861, 1842 },
		{ 1045, 1028 },
		{ 1046, 1029 },
		{ 1048, 1032 },
		{ 1052, 1036 },
		{ 1053, 1037 },
		{ 1054, 1038 },
		{ 1055, 1039 },
		{ 1641, 34 },
		{ 1077, 1064 },
		{ 1476, 34 },
		{ 1056, 1041 },
		{ 1059, 1044 },
		{ 1832, 1813 },
		{ 1060, 1045 },
		{ 1858, 1839 },
		{ 1061, 1046 },
		{ 1860, 1841 },
		{ 1063, 1048 },
		{ 1299, 31 },
		{ 1064, 1049 },
		{ 1488, 33 },
		{ 1861, 1842 },
		{ 1640, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1285, 31 },
		{ 1489, 34 },
		{ 1474, 33 },
		{ 1065, 1050 },
		{ 1079, 1064 },
		{ 1833, 1813 },
		{ 1066, 1051 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1475, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1483, 34 },
		{ 1481, 34 },
		{ 1492, 34 },
		{ 1482, 34 },
		{ 1492, 34 },
		{ 1485, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1484, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1067, 1052 },
		{ 1477, 34 },
		{ 1479, 34 },
		{ 1068, 1053 },
		{ 1492, 34 },
		{ 1069, 1054 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1480, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1486, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1491, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1478, 34 },
		{ 1492, 34 },
		{ 1490, 34 },
		{ 1492, 34 },
		{ 1487, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 1492, 34 },
		{ 131, 4 },
		{ 132, 4 },
		{ 1104, 1093 },
		{ 1105, 1093 },
		{ 1868, 1849 },
		{ 1868, 1849 },
		{ 1869, 1850 },
		{ 1869, 1850 },
		{ 1070, 1056 },
		{ 1073, 1059 },
		{ 1074, 1060 },
		{ 1871, 1852 },
		{ 1871, 1852 },
		{ 1798, 1778 },
		{ 1798, 1778 },
		{ 1075, 1061 },
		{ 1076, 1063 },
		{ 996, 978 },
		{ 1080, 1065 },
		{ 1081, 1066 },
		{ 1084, 1068 },
		{ 1082, 1067 },
		{ 1085, 1069 },
		{ 131, 4 },
		{ 1086, 1070 },
		{ 1104, 1093 },
		{ 1089, 1074 },
		{ 1868, 1849 },
		{ 1083, 1067 },
		{ 1869, 1850 },
		{ 1107, 1094 },
		{ 1108, 1094 },
		{ 1110, 1095 },
		{ 1111, 1095 },
		{ 1871, 1852 },
		{ 1090, 1075 },
		{ 1798, 1778 },
		{ 61, 4 },
		{ 130, 4 },
		{ 60, 4 },
		{ 60, 4 },
		{ 60, 4 },
		{ 60, 4 },
		{ 60, 4 },
		{ 60, 4 },
		{ 60, 4 },
		{ 60, 4 },
		{ 60, 4 },
		{ 60, 4 },
		{ 1091, 1076 },
		{ 1093, 1080 },
		{ 1094, 1081 },
		{ 1095, 1082 },
		{ 1107, 1094 },
		{ 1106, 1093 },
		{ 1110, 1095 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 58, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 59, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 1109, 1094 },
		{ 57, 4 },
		{ 1112, 1095 },
		{ 1096, 1083 },
		{ 62, 4 },
		{ 1097, 1084 },
		{ 62, 4 },
		{ 50, 4 },
		{ 55, 4 },
		{ 53, 4 },
		{ 62, 4 },
		{ 54, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 52, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 56, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 51, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 62, 4 },
		{ 2086, 38 },
		{ 2087, 38 },
		{ 1113, 1096 },
		{ 1114, 1096 },
		{ 45, 38 },
		{ 1098, 1085 },
		{ 1874, 1855 },
		{ 1874, 1855 },
		{ 1099, 1086 },
		{ 1879, 1859 },
		{ 1879, 1859 },
		{ 1883, 1862 },
		{ 1883, 1862 },
		{ 1884, 1863 },
		{ 1884, 1863 },
		{ 1101, 1089 },
		{ 1102, 1090 },
		{ 1103, 1091 },
		{ 999, 979 },
		{ 1000, 980 },
		{ 1003, 982 },
		{ 1002, 981 },
		{ 1116, 1097 },
		{ 2086, 38 },
		{ 1117, 1098 },
		{ 1113, 1096 },
		{ 1895, 1872 },
		{ 1895, 1872 },
		{ 1119, 1101 },
		{ 1874, 1855 },
		{ 1127, 1117 },
		{ 1128, 1117 },
		{ 1879, 1859 },
		{ 1001, 981 },
		{ 1883, 1862 },
		{ 1120, 1102 },
		{ 1884, 1863 },
		{ 1694, 38 },
		{ 2085, 38 },
		{ 1693, 38 },
		{ 1693, 38 },
		{ 1693, 38 },
		{ 1693, 38 },
		{ 1693, 38 },
		{ 1693, 38 },
		{ 1693, 38 },
		{ 1693, 38 },
		{ 1693, 38 },
		{ 1693, 38 },
		{ 1895, 1872 },
		{ 1121, 1103 },
		{ 1126, 1116 },
		{ 1004, 983 },
		{ 1127, 1117 },
		{ 1115, 1096 },
		{ 1875, 1855 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1690, 38 },
		{ 1695, 38 },
		{ 1684, 38 },
		{ 1686, 38 },
		{ 1687, 38 },
		{ 1695, 38 },
		{ 1692, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1689, 38 },
		{ 1682, 38 },
		{ 1683, 38 },
		{ 1685, 38 },
		{ 1695, 38 },
		{ 1691, 38 },
		{ 1688, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1129, 1117 },
		{ 1696, 38 },
		{ 1131, 1119 },
		{ 1132, 1120 },
		{ 1695, 38 },
		{ 1133, 1121 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 1695, 38 },
		{ 987, 19 },
		{ 1896, 1873 },
		{ 1896, 1873 },
		{ 1005, 984 },
		{ 975, 19 },
		{ 1931, 1909 },
		{ 1931, 1909 },
		{ 1808, 1787 },
		{ 1808, 1787 },
		{ 1139, 1131 },
		{ 1140, 1132 },
		{ 1141, 1133 },
		{ 1007, 985 },
		{ 1145, 1140 },
		{ 1146, 1141 },
		{ 1006, 985 },
		{ 1149, 1145 },
		{ 1150, 1146 },
		{ 1152, 1149 },
		{ 1153, 1150 },
		{ 1154, 1152 },
		{ 1155, 1153 },
		{ 997, 1154 },
		{ 987, 19 },
		{ 1896, 1873 },
		{ 976, 19 },
		{ 988, 19 },
		{ 1010, 989 },
		{ 1931, 1909 },
		{ 1011, 990 },
		{ 1808, 1787 },
		{ 1012, 991 },
		{ 1015, 996 },
		{ 1016, 999 },
		{ 1017, 1000 },
		{ 1018, 1001 },
		{ 1019, 1002 },
		{ 1020, 1003 },
		{ 1021, 1004 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 1022, 1005 },
		{ 1023, 1006 },
		{ 1024, 1007 },
		{ 1027, 1010 },
		{ 1028, 1011 },
		{ 1029, 1012 },
		{ 1032, 1015 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 990, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 981, 19 },
		{ 979, 19 },
		{ 992, 19 },
		{ 980, 19 },
		{ 992, 19 },
		{ 983, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 982, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 1033, 1016 },
		{ 977, 19 },
		{ 1034, 1017 },
		{ 1035, 1018 },
		{ 992, 19 },
		{ 1036, 1019 },
		{ 992, 19 },
		{ 992, 19 },
		{ 978, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 984, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 991, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 989, 19 },
		{ 992, 19 },
		{ 985, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 992, 19 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 1037, 1020 },
		{ 1038, 1021 },
		{ 1039, 1022 },
		{ 1040, 1023 },
		{ 113, 102 },
		{ 114, 103 },
		{ 115, 108 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 116, 109 },
		{ 117, 110 },
		{ 118, 112 },
		{ 119, 113 },
		{ 997, 1156 },
		{ 120, 114 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 997, 1156 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 121, 115 },
		{ 122, 118 },
		{ 123, 119 },
		{ 124, 120 },
		{ 125, 123 },
		{ 66, 50 },
		{ 127, 125 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 128, 127 },
		{ 129, 128 },
		{ 67, 51 },
		{ 68, 52 },
		{ 1695, 1915 },
		{ 69, 53 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 1695, 1915 },
		{ 998, 1155 },
		{ 0, 1155 },
		{ 70, 54 },
		{ 71, 55 },
		{ 72, 56 },
		{ 74, 58 },
		{ 75, 59 },
		{ 80, 66 },
		{ 1699, 1682 },
		{ 1701, 1683 },
		{ 1703, 1684 },
		{ 1704, 1684 },
		{ 1705, 1685 },
		{ 1706, 1686 },
		{ 1875, 1875 },
		{ 1708, 1687 },
		{ 1709, 1688 },
		{ 1707, 1686 },
		{ 1712, 1689 },
		{ 1700, 1683 },
		{ 1702, 1684 },
		{ 1713, 1690 },
		{ 1714, 1691 },
		{ 998, 1155 },
		{ 1715, 1692 },
		{ 1695, 1695 },
		{ 1721, 1699 },
		{ 1722, 1700 },
		{ 1723, 1701 },
		{ 1725, 1702 },
		{ 1711, 1689 },
		{ 1726, 1703 },
		{ 1710, 1688 },
		{ 1727, 1704 },
		{ 1728, 1705 },
		{ 1729, 1706 },
		{ 1730, 1707 },
		{ 1875, 1875 },
		{ 1731, 1708 },
		{ 1732, 1709 },
		{ 1733, 1710 },
		{ 1734, 1711 },
		{ 1735, 1712 },
		{ 1724, 1702 },
		{ 1736, 1713 },
		{ 1737, 1714 },
		{ 1738, 1715 },
		{ 1745, 1721 },
		{ 1746, 1722 },
		{ 1747, 1723 },
		{ 1748, 1724 },
		{ 1749, 1725 },
		{ 0, 1155 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1750, 1726 },
		{ 1751, 1727 },
		{ 1752, 1728 },
		{ 1753, 1729 },
		{ 1754, 1730 },
		{ 1755, 1731 },
		{ 1756, 1732 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1757, 1733 },
		{ 1758, 1734 },
		{ 1759, 1735 },
		{ 1760, 1736 },
		{ 1875, 1875 },
		{ 1156, 1155 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 1875, 1875 },
		{ 2043, 2043 },
		{ 1761, 1737 },
		{ 1762, 1738 },
		{ 1768, 1745 },
		{ 1769, 1746 },
		{ 1770, 1747 },
		{ 1771, 1748 },
		{ 1772, 1749 },
		{ 1773, 1750 },
		{ 1775, 1751 },
		{ 1776, 1752 },
		{ 1777, 1753 },
		{ 1774, 1750 },
		{ 1778, 1754 },
		{ 1779, 1755 },
		{ 1780, 1756 },
		{ 1781, 1757 },
		{ 1782, 1758 },
		{ 1783, 1759 },
		{ 1784, 1760 },
		{ 1785, 1761 },
		{ 1786, 1762 },
		{ 1787, 1768 },
		{ 2043, 2043 },
		{ 1788, 1769 },
		{ 1789, 1770 },
		{ 1790, 1771 },
		{ 1791, 1772 },
		{ 81, 67 },
		{ 1794, 1774 },
		{ 1795, 1775 },
		{ 1796, 1776 },
		{ 1797, 1777 },
		{ 82, 68 },
		{ 1799, 1779 },
		{ 1800, 1780 },
		{ 1801, 1781 },
		{ 1802, 1782 },
		{ 1805, 1784 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 1803, 1783 },
		{ 1806, 1785 },
		{ 1807, 1786 },
		{ 83, 69 },
		{ 1809, 1788 },
		{ 1810, 1789 },
		{ 1804, 1783 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 45, 7 },
		{ 1811, 1790 },
		{ 1812, 1791 },
		{ 1813, 1794 },
		{ 2043, 2043 },
		{ 1814, 1795 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 1815, 1796 },
		{ 1816, 1797 },
		{ 1818, 1799 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 1819, 1800 },
		{ 1820, 1801 },
		{ 1821, 1802 },
		{ 1822, 1803 },
		{ 1823, 1804 },
		{ 1824, 1805 },
		{ 84, 70 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 1826, 1807 },
		{ 1828, 1809 },
		{ 1829, 1810 },
		{ 1830, 1811 },
		{ 870, 7 },
		{ 1831, 1812 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 870, 7 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 85, 71 },
		{ 1834, 1814 },
		{ 1835, 1815 },
		{ 1836, 1816 },
		{ 1838, 1818 },
		{ 1839, 1819 },
		{ 1840, 1820 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 1841, 1821 },
		{ 1842, 1822 },
		{ 1843, 1823 },
		{ 1844, 1824 },
		{ 0, 1047 },
		{ 1847, 1826 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 0, 1047 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 1849, 1828 },
		{ 1850, 1829 },
		{ 1851, 1830 },
		{ 1852, 1831 },
		{ 1853, 1834 },
		{ 1854, 1835 },
		{ 1855, 1836 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 86, 72 },
		{ 88, 74 },
		{ 1859, 1840 },
		{ 89, 75 },
		{ 62, 129 },
		{ 95, 80 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 62, 129 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 1862, 1843 },
		{ 1863, 1844 },
		{ 1866, 1847 },
		{ 96, 81 },
		{ 1870, 1851 },
		{ 97, 83 },
		{ 1872, 1853 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 1873, 1854 },
		{ 98, 84 },
		{ 99, 85 },
		{ 100, 86 },
		{ 870, 870 },
		{ 102, 88 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 870, 870 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 1887, 1866 },
		{ 1893, 1870 },
		{ 103, 89 },
		{ 108, 95 },
		{ 109, 96 },
		{ 1909, 1887 },
		{ 110, 97 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 111, 98 },
		{ 112, 100 },
		{ 0, 2143 },
		{ 104, 90 },
		{ 0, 1363 },
		{ 104, 90 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1363 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 1740, 1717 },
		{ 106, 93 },
		{ 1740, 1717 },
		{ 106, 93 },
		{ 880, 877 },
		{ 1963, 1940 },
		{ 880, 877 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 91, 77 },
		{ 883, 879 },
		{ 91, 77 },
		{ 883, 879 },
		{ 0, 1546 },
		{ 1936, 1914 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1546 },
		{ 0, 1893 },
		{ 0, 1893 },
		{ 154, 145 },
		{ 2069, 2058 },
		{ 156, 145 },
		{ 164, 145 },
		{ 151, 145 },
		{ 885, 882 },
		{ 155, 145 },
		{ 885, 882 },
		{ 153, 145 },
		{ 1763, 1739 },
		{ 2078, 2072 },
		{ 1763, 1739 },
		{ 162, 145 },
		{ 161, 145 },
		{ 152, 145 },
		{ 160, 145 },
		{ 2023, 2004 },
		{ 157, 145 },
		{ 159, 145 },
		{ 150, 145 },
		{ 2052, 2035 },
		{ 0, 1893 },
		{ 158, 145 },
		{ 163, 145 },
		{ 1716, 1693 },
		{ 2142, 43 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 1931, 1931 },
		{ 1931, 1931 },
		{ 971, 970 },
		{ 1890, 1867 },
		{ 1282, 1281 },
		{ 1888, 1867 },
		{ 2060, 2049 },
		{ 1889, 1867 },
		{ 2129, 2128 },
		{ 971, 970 },
		{ 876, 873 },
		{ 1717, 1693 },
		{ 873, 873 },
		{ 873, 873 },
		{ 873, 873 },
		{ 873, 873 },
		{ 873, 873 },
		{ 873, 873 },
		{ 873, 873 },
		{ 873, 873 },
		{ 873, 873 },
		{ 873, 873 },
		{ 1184, 1183 },
		{ 1931, 1931 },
		{ 1209, 1208 },
		{ 1252, 1251 },
		{ 1313, 1291 },
		{ 1993, 1971 },
		{ 1361, 1342 },
		{ 1184, 1183 },
		{ 1660, 1659 },
		{ 1665, 1664 },
		{ 1365, 1348 },
		{ 877, 873 },
		{ 1825, 1825 },
		{ 1825, 1825 },
		{ 1229, 1228 },
		{ 1884, 1884 },
		{ 1884, 1884 },
		{ 1312, 1291 },
		{ 925, 924 },
		{ 1255, 1254 },
		{ 2108, 2107 },
		{ 1717, 1693 },
		{ 76, 60 },
		{ 1500, 1480 },
		{ 60, 60 },
		{ 60, 60 },
		{ 60, 60 },
		{ 60, 60 },
		{ 60, 60 },
		{ 60, 60 },
		{ 60, 60 },
		{ 60, 60 },
		{ 60, 60 },
		{ 60, 60 },
		{ 1525, 1508 },
		{ 1825, 1825 },
		{ 1548, 1533 },
		{ 1339, 1320 },
		{ 1884, 1884 },
		{ 1876, 1856 },
		{ 928, 927 },
		{ 1382, 1366 },
		{ 1399, 1383 },
		{ 877, 873 },
		{ 1409, 1396 },
		{ 77, 60 },
		{ 1450, 1448 },
		{ 1948, 1926 },
		{ 1951, 1929 },
		{ 966, 965 },
		{ 1861, 1861 },
		{ 1861, 1861 },
		{ 1915, 1893 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 78, 78 },
		{ 78, 78 },
		{ 78, 78 },
		{ 78, 78 },
		{ 78, 78 },
		{ 78, 78 },
		{ 78, 78 },
		{ 78, 78 },
		{ 78, 78 },
		{ 78, 78 },
		{ 1861, 1861 },
		{ 1742, 1718 },
		{ 1257, 1256 },
		{ 1965, 1943 },
		{ 77, 60 },
		{ 1986, 1964 },
		{ 63, 43 },
		{ 1991, 1969 },
		{ 1954, 1931 },
		{ 1179, 1178 },
		{ 1953, 1931 },
		{ 93, 78 },
		{ 2008, 1988 },
		{ 76, 76 },
		{ 76, 76 },
		{ 76, 76 },
		{ 76, 76 },
		{ 76, 76 },
		{ 76, 76 },
		{ 76, 76 },
		{ 76, 76 },
		{ 76, 76 },
		{ 76, 76 },
		{ 876, 876 },
		{ 876, 876 },
		{ 876, 876 },
		{ 876, 876 },
		{ 876, 876 },
		{ 876, 876 },
		{ 876, 876 },
		{ 876, 876 },
		{ 876, 876 },
		{ 876, 876 },
		{ 1742, 1718 },
		{ 90, 76 },
		{ 2018, 1999 },
		{ 912, 911 },
		{ 2030, 2012 },
		{ 2038, 2020 },
		{ 919, 918 },
		{ 1559, 1545 },
		{ 1846, 1825 },
		{ 1845, 1825 },
		{ 93, 78 },
		{ 879, 876 },
		{ 1906, 1884 },
		{ 878, 878 },
		{ 878, 878 },
		{ 878, 878 },
		{ 878, 878 },
		{ 878, 878 },
		{ 878, 878 },
		{ 878, 878 },
		{ 878, 878 },
		{ 878, 878 },
		{ 878, 878 },
		{ 1716, 1716 },
		{ 1716, 1716 },
		{ 1716, 1716 },
		{ 1716, 1716 },
		{ 1716, 1716 },
		{ 1716, 1716 },
		{ 1716, 1716 },
		{ 1716, 1716 },
		{ 1716, 1716 },
		{ 1716, 1716 },
		{ 90, 76 },
		{ 882, 878 },
		{ 1882, 1861 },
		{ 1879, 1879 },
		{ 1879, 1879 },
		{ 872, 9 },
		{ 1869, 1869 },
		{ 1869, 1869 },
		{ 2058, 2046 },
		{ 45, 9 },
		{ 879, 876 },
		{ 1739, 1716 },
		{ 1857, 1857 },
		{ 1857, 1857 },
		{ 1896, 1896 },
		{ 1896, 1896 },
		{ 901, 900 },
		{ 1881, 1861 },
		{ 1798, 1798 },
		{ 1798, 1798 },
		{ 1574, 1563 },
		{ 2072, 2061 },
		{ 1585, 1573 },
		{ 2083, 2082 },
		{ 1631, 1629 },
		{ 1342, 1323 },
		{ 1879, 1879 },
		{ 2111, 2110 },
		{ 872, 9 },
		{ 1869, 1869 },
		{ 2120, 2119 },
		{ 1860, 1860 },
		{ 1860, 1860 },
		{ 882, 878 },
		{ 1230, 1229 },
		{ 1857, 1857 },
		{ 1662, 1661 },
		{ 1896, 1896 },
		{ 1667, 1667 },
		{ 1667, 1667 },
		{ 2137, 2136 },
		{ 1798, 1798 },
		{ 874, 9 },
		{ 1739, 1716 },
		{ 873, 9 },
		{ 873, 9 },
		{ 873, 9 },
		{ 873, 9 },
		{ 873, 9 },
		{ 873, 9 },
		{ 873, 9 },
		{ 873, 9 },
		{ 873, 9 },
		{ 873, 9 },
		{ 1860, 1860 },
		{ 1895, 1895 },
		{ 1895, 1895 },
		{ 1808, 1808 },
		{ 1808, 1808 },
		{ 2126, 2126 },
		{ 2126, 2126 },
		{ 1667, 1667 },
		{ 1232, 1231 },
		{ 1378, 1362 },
		{ 1674, 1673 },
		{ 1509, 1487 },
		{ 1321, 1298 },
		{ 1206, 1205 },
		{ 1508, 1487 },
		{ 1320, 1298 },
		{ 1926, 1904 },
		{ 1207, 1206 },
		{ 943, 942 },
		{ 1224, 1223 },
		{ 1225, 1224 },
		{ 1428, 1418 },
		{ 1437, 1428 },
		{ 944, 943 },
		{ 1895, 1895 },
		{ 946, 945 },
		{ 1808, 1808 },
		{ 1967, 1945 },
		{ 2126, 2126 },
		{ 1973, 1951 },
		{ 1138, 1130 },
		{ 1247, 1246 },
		{ 1248, 1247 },
		{ 959, 958 },
		{ 1526, 1509 },
		{ 2019, 2000 },
		{ 1541, 1524 },
		{ 2025, 2007 },
		{ 2026, 2008 },
		{ 1543, 1526 },
		{ 960, 959 },
		{ 905, 904 },
		{ 2054, 2038 },
		{ 1272, 1271 },
		{ 1273, 1272 },
		{ 1278, 1277 },
		{ 1143, 1138 },
		{ 890, 889 },
		{ 2074, 2064 },
		{ 1172, 1171 },
		{ 1173, 1172 },
		{ 1901, 1879 },
		{ 1900, 1879 },
		{ 917, 916 },
		{ 2103, 2102 },
		{ 1340, 1321 },
		{ 936, 935 },
		{ 2115, 2114 },
		{ 1892, 1869 },
		{ 1677, 1676 },
		{ 1357, 1338 },
		{ 1908, 1886 },
		{ 1359, 1340 },
		{ 1911, 1889 },
		{ 1877, 1857 },
		{ 1199, 1198 },
		{ 1918, 1896 },
		{ 1200, 1199 },
		{ 1902, 1880 },
		{ 1817, 1798 },
		{ 2009, 1989 },
		{ 1259, 1258 },
		{ 973, 972 },
		{ 1445, 1439 },
		{ 2024, 2006 },
		{ 1626, 1620 },
		{ 1913, 1891 },
		{ 2029, 2011 },
		{ 1350, 1331 },
		{ 899, 898 },
		{ 1880, 1860 },
		{ 1234, 1233 },
		{ 1186, 1185 },
		{ 1927, 1905 },
		{ 1928, 1906 },
		{ 1664, 1663 },
		{ 1211, 1210 },
		{ 1668, 1667 },
		{ 1517, 1500 },
		{ 1944, 1922 },
		{ 1670, 1669 },
		{ 1885, 1864 },
		{ 1952, 1930 },
		{ 1377, 1361 },
		{ 2105, 2104 },
		{ 1676, 1675 },
		{ 938, 937 },
		{ 1534, 1517 },
		{ 892, 891 },
		{ 1977, 1955 },
		{ 1979, 1957 },
		{ 2135, 2134 },
		{ 1330, 1311 },
		{ 1331, 1312 },
		{ 1917, 1895 },
		{ 1827, 1808 },
		{ 2127, 2126 },
		{ 1204, 1203 },
		{ 2007, 1987 },
		{ 1971, 1949 },
		{ 1533, 1516 },
		{ 1930, 1908 },
		{ 1989, 1967 },
		{ 1279, 1278 },
		{ 1348, 1329 },
		{ 1997, 1975 },
		{ 1429, 1419 },
		{ 1013, 993 },
		{ 921, 920 },
		{ 1449, 1447 },
		{ 2006, 1986 },
		{ 1208, 1207 },
		{ 1454, 1452 },
		{ 953, 952 },
		{ 2011, 1991 },
		{ 930, 929 },
		{ 1344, 1325 },
		{ 1346, 1327 },
		{ 1512, 1493 },
		{ 1516, 1499 },
		{ 1916, 1894 },
		{ 1258, 1257 },
		{ 1218, 1217 },
		{ 1924, 1902 },
		{ 1266, 1265 },
		{ 2053, 2036 },
		{ 1529, 1512 },
		{ 1531, 1514 },
		{ 903, 902 },
		{ 2059, 2048 },
		{ 1193, 1192 },
		{ 1057, 1042 },
		{ 1376, 1360 },
		{ 1939, 1917 },
		{ 1030, 1013 },
		{ 1558, 1544 },
		{ 1950, 1928 },
		{ 1166, 1165 },
		{ 1233, 1232 },
		{ 1570, 1557 },
		{ 1205, 1204 },
		{ 2113, 2112 },
		{ 1391, 1375 },
		{ 1609, 1596 },
		{ 1241, 1240 },
		{ 1630, 1628 },
		{ 1976, 1954 },
		{ 1325, 1304 },
		{ 2139, 2138 },
		{ 1635, 1633 },
		{ 1329, 1310 },
		{ 1987, 1965 },
		{ 1871, 1871 },
		{ 1871, 1871 },
		{ 1858, 1858 },
		{ 1858, 1858 },
		{ 2130, 2130 },
		{ 2130, 2130 },
		{ 964, 963 },
		{ 1366, 1349 },
		{ 1992, 1970 },
		{ 1240, 1239 },
		{ 1996, 1974 },
		{ 1042, 1025 },
		{ 1527, 1510 },
		{ 1165, 1164 },
		{ 1327, 1307 },
		{ 1328, 1309 },
		{ 1383, 1367 },
		{ 926, 925 },
		{ 1393, 1377 },
		{ 2020, 2001 },
		{ 1545, 1528 },
		{ 1014, 995 },
		{ 910, 909 },
		{ 1871, 1871 },
		{ 1416, 1406 },
		{ 1858, 1858 },
		{ 1177, 1176 },
		{ 2130, 2130 },
		{ 1921, 1899 },
		{ 2031, 2013 },
		{ 1563, 1549 },
		{ 2040, 2022 },
		{ 2045, 2027 },
		{ 1217, 1216 },
		{ 1430, 1420 },
		{ 1341, 1322 },
		{ 1929, 1907 },
		{ 1594, 1583 },
		{ 1438, 1429 },
		{ 1932, 1910 },
		{ 2064, 2053 },
		{ 1610, 1597 },
		{ 2070, 2059 },
		{ 1619, 1609 },
		{ 904, 903 },
		{ 1628, 1624 },
		{ 2082, 2079 },
		{ 1447, 1443 },
		{ 1265, 1264 },
		{ 952, 951 },
		{ 1636, 1635 },
		{ 931, 930 },
		{ 2109, 2108 },
		{ 1455, 1454 },
		{ 1192, 1191 },
		{ 2114, 2113 },
		{ 922, 921 },
		{ 1072, 1058 },
		{ 1743, 1719 },
		{ 1666, 1665 },
		{ 87, 73 },
		{ 2133, 2132 },
		{ 1978, 1956 },
		{ 1362, 1343 },
		{ 1982, 1960 },
		{ 2140, 2139 },
		{ 1514, 1496 },
		{ 1515, 1498 },
		{ 1988, 1966 },
		{ 923, 923 },
		{ 923, 923 },
		{ 1167, 1166 },
		{ 1883, 1883 },
		{ 1883, 1883 },
		{ 2106, 2106 },
		{ 2106, 2106 },
		{ 1392, 1376 },
		{ 1672, 1671 },
		{ 1547, 1531 },
		{ 1364, 1346 },
		{ 1912, 1890 },
		{ 1267, 1266 },
		{ 1451, 1449 },
		{ 1632, 1630 },
		{ 2073, 2062 },
		{ 1194, 1193 },
		{ 954, 953 },
		{ 1219, 1218 },
		{ 1571, 1558 },
		{ 1071, 1057 },
		{ 1228, 1227 },
		{ 1242, 1241 },
		{ 923, 923 },
		{ 897, 896 },
		{ 1894, 1871 },
		{ 1883, 1883 },
		{ 1878, 1858 },
		{ 2106, 2106 },
		{ 2131, 2130 },
		{ 406, 358 },
		{ 404, 358 },
		{ 405, 358 },
		{ 1170, 1169 },
		{ 403, 358 },
		{ 1925, 1903 },
		{ 402, 358 },
		{ 2121, 2121 },
		{ 2121, 2121 },
		{ 1427, 1417 },
		{ 407, 358 },
		{ 1937, 1937 },
		{ 1937, 1937 },
		{ 1608, 1595 },
		{ 1524, 1507 },
		{ 1047, 1030 },
		{ 1197, 1196 },
		{ 1363, 1344 },
		{ 401, 358 },
		{ 915, 914 },
		{ 1933, 1911 },
		{ 1998, 1976 },
		{ 941, 940 },
		{ 1270, 1269 },
		{ 1338, 1319 },
		{ 1222, 1221 },
		{ 1245, 1244 },
		{ 2016, 1997 },
		{ 957, 956 },
		{ 1546, 1529 },
		{ 2121, 2121 },
		{ 1661, 1660 },
		{ 2021, 2002 },
		{ 1381, 1365 },
		{ 1937, 1937 },
		{ 1182, 1181 },
		{ 1961, 1938 },
		{ 1962, 1939 },
		{ 2028, 2010 },
		{ 1280, 1279 },
		{ 969, 968 },
		{ 1254, 1253 },
		{ 1562, 1548 },
		{ 2039, 2021 },
		{ 1118, 1100 },
		{ 1351, 1332 },
		{ 2050, 2032 },
		{ 1408, 1395 },
		{ 1920, 1898 },
		{ 893, 893 },
		{ 893, 893 },
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
		{ 0, 874 },
		{ 0, 61 },
		{ 924, 923 },
		{ 0, 1694 },
		{ 1304, 1460 },
		{ 1905, 1883 },
		{ 1493, 1641 },
		{ 2107, 2106 },
		{ 993, 987 },
		{ 920, 919 },
		{ 2046, 2028 },
		{ 893, 893 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1764, 1764 },
		{ 1764, 1764 },
		{ 1764, 1764 },
		{ 1764, 1764 },
		{ 1764, 1764 },
		{ 1764, 1764 },
		{ 1764, 1764 },
		{ 1764, 1764 },
		{ 1764, 1764 },
		{ 1764, 1764 },
		{ 0, 874 },
		{ 0, 61 },
		{ 2048, 2030 },
		{ 0, 1694 },
		{ 1452, 1450 },
		{ 1176, 1175 },
		{ 2122, 2121 },
		{ 1277, 1276 },
		{ 1984, 1962 },
		{ 929, 928 },
		{ 1960, 1937 },
		{ 1766, 1766 },
		{ 1766, 1766 },
		{ 1766, 1766 },
		{ 1766, 1766 },
		{ 1766, 1766 },
		{ 1766, 1766 },
		{ 1766, 1766 },
		{ 1766, 1766 },
		{ 1766, 1766 },
		{ 1766, 1766 },
		{ 92, 92 },
		{ 92, 92 },
		{ 92, 92 },
		{ 92, 92 },
		{ 92, 92 },
		{ 92, 92 },
		{ 92, 92 },
		{ 92, 92 },
		{ 92, 92 },
		{ 92, 92 },
		{ 881, 881 },
		{ 881, 881 },
		{ 881, 881 },
		{ 881, 881 },
		{ 881, 881 },
		{ 881, 881 },
		{ 881, 881 },
		{ 881, 881 },
		{ 881, 881 },
		{ 881, 881 },
		{ 939, 939 },
		{ 939, 939 },
		{ 1958, 1958 },
		{ 1958, 1958 },
		{ 1178, 1177 },
		{ 994, 977 },
		{ 894, 893 },
		{ 105, 105 },
		{ 105, 105 },
		{ 105, 105 },
		{ 105, 105 },
		{ 105, 105 },
		{ 105, 105 },
		{ 105, 105 },
		{ 105, 105 },
		{ 105, 105 },
		{ 105, 105 },
		{ 1497, 1477 },
		{ 2061, 2050 },
		{ 1394, 1378 },
		{ 1180, 1179 },
		{ 1406, 1391 },
		{ 1934, 1912 },
		{ 939, 939 },
		{ 1572, 1559 },
		{ 1958, 1958 },
		{ 884, 884 },
		{ 884, 884 },
		{ 884, 884 },
		{ 884, 884 },
		{ 884, 884 },
		{ 884, 884 },
		{ 884, 884 },
		{ 884, 884 },
		{ 884, 884 },
		{ 884, 884 },
		{ 886, 886 },
		{ 886, 886 },
		{ 886, 886 },
		{ 886, 886 },
		{ 886, 886 },
		{ 886, 886 },
		{ 886, 886 },
		{ 886, 886 },
		{ 886, 886 },
		{ 886, 886 },
		{ 521, 474 },
		{ 913, 912 },
		{ 517, 474 },
		{ 1583, 1570 },
		{ 516, 474 },
		{ 1940, 1918 },
		{ 518, 474 },
		{ 1943, 1921 },
		{ 1308, 1288 },
		{ 2010, 1990 },
		{ 519, 474 },
		{ 1183, 1182 },
		{ 1596, 1585 },
		{ 522, 474 },
		{ 520, 474 },
		{ 2017, 1998 },
		{ 1419, 1409 },
		{ 963, 962 },
		{ 2112, 2111 },
		{ 902, 901 },
		{ 965, 964 },
		{ 2022, 2003 },
		{ 916, 915 },
		{ 967, 966 },
		{ 911, 910 },
		{ 1914, 1892 },
		{ 995, 977 },
		{ 2027, 2009 },
		{ 970, 969 },
		{ 1970, 1948 },
		{ 1332, 1313 },
		{ 2138, 2137 },
		{ 1633, 1631 },
		{ 2032, 2014 },
		{ 2036, 2018 },
		{ 1974, 1952 },
		{ 1058, 1043 },
		{ 45, 5 },
		{ 1498, 1477 },
		{ 1868, 1868 },
		{ 1868, 1868 },
		{ 2125, 2124 },
		{ 2057, 2045 },
		{ 2015, 1996 },
		{ 940, 939 },
		{ 1957, 1934 },
		{ 1980, 1958 },
		{ 2003, 1982 },
		{ 1322, 1301 },
		{ 1848, 1827 },
		{ 1865, 1846 },
		{ 1923, 1901 },
		{ 1323, 1301 },
		{ 1462, 1459 },
		{ 1886, 1865 },
		{ 1658, 1657 },
		{ 1306, 1285 },
		{ 1043, 1026 },
		{ 1461, 1459 },
		{ 1495, 1474 },
		{ 1643, 1640 },
		{ 1305, 1285 },
		{ 1868, 1868 },
		{ 1945, 1923 },
		{ 1494, 1474 },
		{ 1642, 1640 },
		{ 1009, 988 },
		{ 1549, 1534 },
		{ 1556, 1541 },
		{ 1088, 1072 },
		{ 1243, 1242 },
		{ 1307, 1286 },
		{ 1561, 1547 },
		{ 1164, 1163 },
		{ 1210, 1209 },
		{ 1309, 1288 },
		{ 1025, 1008 },
		{ 1935, 1913 },
		{ 1453, 1451 },
		{ 1367, 1350 },
		{ 1374, 1357 },
		{ 1251, 1250 },
		{ 1942, 1920 },
		{ 1216, 1215 },
		{ 1185, 1184 },
		{ 1026, 1009 },
		{ 1947, 1925 },
		{ 1597, 1586 },
		{ 1949, 1927 },
		{ 146, 5 },
		{ 1719, 1696 },
		{ 1380, 1364 },
		{ 1496, 1475 },
		{ 147, 5 },
		{ 1191, 1190 },
		{ 1955, 1932 },
		{ 1220, 1219 },
		{ 2055, 2039 },
		{ 2056, 2040 },
		{ 1959, 1936 },
		{ 148, 5 },
		{ 1620, 1610 },
		{ 1624, 1618 },
		{ 891, 890 },
		{ 1264, 1263 },
		{ 1168, 1167 },
		{ 2062, 2051 },
		{ 1966, 1944 },
		{ 972, 971 },
		{ 1195, 1194 },
		{ 1268, 1267 },
		{ 1837, 1817 },
		{ 1634, 1632 },
		{ 1975, 1953 },
		{ 2079, 2073 },
		{ 1898, 1877 },
		{ 145, 5 },
		{ 1899, 1878 },
		{ 937, 936 },
		{ 1343, 1324 },
		{ 2104, 2103 },
		{ 1903, 1881 },
		{ 1904, 1882 },
		{ 895, 894 },
		{ 1985, 1963 },
		{ 955, 954 },
		{ 1907, 1885 },
		{ 1891, 1868 },
		{ 1528, 1511 },
		{ 1657, 1656 },
		{ 1910, 1888 },
		{ 945, 944 },
		{ 1420, 1410 },
		{ 1349, 1330 },
		{ 73, 57 },
		{ 2128, 2127 },
		{ 1239, 1238 },
		{ 1542, 1525 },
		{ 2132, 2131 },
		{ 951, 950 },
		{ 2134, 2133 },
		{ 1358, 1339 },
		{ 1919, 1897 },
		{ 1087, 1071 },
		{ 1439, 1430 },
		{ 1673, 1672 },
		{ 2013, 1993 },
		{ 1443, 1436 },
		{ 1675, 1674 },
		{ 1360, 1341 },
		{ 1544, 1527 },
		{ 2066, 2055 },
		{ 101, 87 },
		{ 1368, 1351 },
		{ 1276, 1275 },
		{ 1532, 1515 },
		{ 2000, 1978 },
		{ 1856, 1837 },
		{ 1767, 1743 },
		{ 1347, 1328 },
		{ 1324, 1302 },
		{ 1031, 1014 },
		{ 1511, 1491 },
		{ 377, 335 },
		{ 382, 335 },
		{ 379, 335 },
		{ 2051, 2034 },
		{ 378, 335 },
		{ 1897, 1876 },
		{ 1584, 1572 },
		{ 2035, 2017 },
		{ 380, 335 },
		{ 896, 895 },
		{ 2110, 2109 },
		{ 1964, 1942 },
		{ 1407, 1394 },
		{ 381, 335 },
		{ 542, 496 },
		{ 2004, 1984 },
		{ 1203, 1202 },
		{ 1969, 1947 },
		{ 2049, 2031 },
		{ 2124, 2123 },
		{ 927, 926 },
		{ 1619, 1619 },
		{ 1619, 1619 },
		{ 2014, 1994 },
		{ 543, 496 },
		{ 1310, 1289 },
		{ 1438, 1438 },
		{ 1438, 1438 },
		{ 993, 986 },
		{ 1946, 1924 },
		{ 1304, 1300 },
		{ 1493, 1489 },
		{ 1972, 1950 },
		{ 1990, 1968 },
		{ 172, 151 },
		{ 914, 913 },
		{ 1510, 1490 },
		{ 170, 151 },
		{ 1864, 1845 },
		{ 171, 151 },
		{ 544, 496 },
		{ 1130, 1118 },
		{ 1311, 1289 },
		{ 1629, 1626 },
		{ 1619, 1619 },
		{ 1448, 1445 },
		{ 1938, 1916 },
		{ 1396, 1381 },
		{ 169, 151 },
		{ 1438, 1438 },
		{ 1223, 1222 },
		{ 918, 917 },
		{ 900, 899 },
		{ 1246, 1245 },
		{ 1181, 1180 },
		{ 1271, 1270 },
		{ 1417, 1407 },
		{ 1573, 1562 },
		{ 1418, 1408 },
		{ 2034, 2016 },
		{ 1319, 1297 },
		{ 1659, 1658 },
		{ 942, 941 },
		{ 1198, 1197 },
		{ 1994, 1972 },
		{ 1231, 1230 },
		{ 1956, 1933 },
		{ 2123, 2122 },
		{ 1595, 1584 },
		{ 1999, 1977 },
		{ 1253, 1252 },
		{ 2001, 1979 },
		{ 2002, 1980 },
		{ 1922, 1900 },
		{ 1171, 1170 },
		{ 958, 957 },
		{ 1669, 1668 },
		{ 1256, 1255 },
		{ 2136, 2135 },
		{ 1671, 1670 },
		{ 1281, 1280 },
		{ 1507, 1486 },
		{ 2012, 1992 },
		{ 1968, 1946 },
		{ 898, 897 },
		{ 968, 967 },
		{ 680, 645 },
		{ 289, 249 },
		{ 372, 331 },
		{ 262, 225 },
		{ 586, 541 },
		{ 783, 760 },
		{ 788, 765 },
		{ 792, 770 },
		{ 1353, 1334 },
		{ 1354, 1335 },
		{ 794, 772 },
		{ 827, 812 },
		{ 841, 831 },
		{ 849, 843 },
		{ 856, 850 },
		{ 283, 244 },
		{ 263, 225 },
		{ 1625, 1619 },
		{ 373, 331 },
		{ 198, 169 },
		{ 293, 252 },
		{ 297, 256 },
		{ 1444, 1438 },
		{ 314, 273 },
		{ 319, 278 },
		{ 679, 645 },
		{ 587, 541 },
		{ 335, 295 },
		{ 1373, 1356 },
		{ 348, 308 },
		{ 358, 318 },
		{ 290, 249 },
		{ 1518, 1501 },
		{ 363, 322 },
		{ 199, 170 },
		{ 222, 189 },
		{ 398, 354 },
		{ 237, 203 },
		{ 410, 361 },
		{ 415, 367 },
		{ 1389, 1372 },
		{ 416, 368 },
		{ 435, 384 },
		{ 1536, 1519 },
		{ 1537, 1520 },
		{ 444, 394 },
		{ 445, 396 },
		{ 471, 422 },
		{ 472, 423 },
		{ 474, 425 },
		{ 493, 448 },
		{ 498, 453 },
		{ 510, 467 },
		{ 248, 213 },
		{ 1555, 1540 },
		{ 528, 480 },
		{ 255, 220 },
		{ 552, 505 },
		{ 570, 523 },
		{ 579, 533 },
		{ 585, 540 },
		{ 260, 223 },
		{ 1568, 1554 },
		{ 612, 572 },
		{ 635, 596 },
		{ 644, 605 },
		{ 650, 611 },
		{ 652, 613 },
		{ 653, 614 },
		{ 655, 616 },
		{ 1333, 1314 },
		{ 676, 642 },
		{ 196, 167 },
		{ 681, 646 },
		{ 690, 656 },
		{ 700, 667 },
		{ 715, 683 },
		{ 723, 692 },
		{ 729, 698 },
		{ 737, 706 },
		{ 739, 708 },
		{ 45, 39 },
		{ 45, 27 },
		{ 45, 41 },
		{ 45, 13 },
		{ 45, 35 },
		{ 45, 17 },
		{ 45, 29 },
		{ 45, 21 },
		{ 45, 23 },
		{ 45, 25 },
		{ 45, 11 },
		{ 45, 15 },
		{ 305, 263 },
		{ 1579, 1567 },
		{ 446, 397 },
		{ 306, 263 },
		{ 357, 317 },
		{ 1577, 1565 },
		{ 383, 336 },
		{ 600, 559 },
		{ 1580, 1567 },
		{ 601, 560 },
		{ 1402, 1387 },
		{ 668, 630 },
		{ 184, 159 },
		{ 334, 294 },
		{ 824, 809 },
		{ 244, 210 },
		{ 304, 263 },
		{ 307, 263 },
		{ 584, 539 },
		{ 279, 240 },
		{ 713, 681 },
		{ 387, 342 },
		{ 425, 377 },
		{ 453, 405 },
		{ 671, 633 },
		{ 428, 378 },
		{ 1504, 1483 },
		{ 186, 159 },
		{ 185, 159 },
		{ 426, 377 },
		{ 454, 405 },
		{ 427, 378 },
		{ 191, 162 },
		{ 566, 519 },
		{ 460, 410 },
		{ 1522, 1505 },
		{ 429, 378 },
		{ 252, 217 },
		{ 1503, 1483 },
		{ 459, 410 },
		{ 563, 517 },
		{ 226, 192 },
		{ 771, 744 },
		{ 1336, 1317 },
		{ 190, 162 },
		{ 355, 315 },
		{ 663, 624 },
		{ 820, 803 },
		{ 593, 547 },
		{ 745, 714 },
		{ 2102, 2100 },
		{ 456, 407 },
		{ 354, 315 },
		{ 565, 519 },
		{ 431, 380 },
		{ 935, 933 },
		{ 853, 847 },
		{ 889, 887 },
		{ 1316, 1294 },
		{ 567, 520 },
		{ 200, 171 },
		{ 360, 319 },
		{ 359, 319 },
		{ 223, 190 },
		{ 351, 311 },
		{ 490, 445 },
		{ 595, 549 },
		{ 2100, 39 },
		{ 1236, 27 },
		{ 2117, 41 },
		{ 907, 13 },
		{ 1654, 35 },
		{ 948, 17 },
		{ 1261, 29 },
		{ 1161, 21 },
		{ 1188, 23 },
		{ 1213, 25 },
		{ 887, 11 },
		{ 933, 15 },
		{ 489, 445 },
		{ 177, 154 },
		{ 212, 182 },
		{ 807, 788 },
		{ 809, 790 },
		{ 213, 182 },
		{ 178, 154 },
		{ 1520, 1503 },
		{ 811, 792 },
		{ 813, 794 },
		{ 816, 798 },
		{ 480, 431 },
		{ 482, 433 },
		{ 825, 810 },
		{ 605, 565 },
		{ 838, 827 },
		{ 610, 570 },
		{ 214, 182 },
		{ 848, 842 },
		{ 312, 270 },
		{ 424, 376 },
		{ 855, 849 },
		{ 643, 604 },
		{ 860, 856 },
		{ 865, 861 },
		{ 288, 248 },
		{ 647, 608 },
		{ 501, 456 },
		{ 651, 612 },
		{ 505, 461 },
		{ 316, 275 },
		{ 515, 473 },
		{ 656, 617 },
		{ 658, 619 },
		{ 287, 248 },
		{ 286, 248 },
		{ 660, 621 },
		{ 318, 277 },
		{ 666, 627 },
		{ 667, 629 },
		{ 524, 476 },
		{ 669, 631 },
		{ 670, 632 },
		{ 364, 323 },
		{ 533, 486 },
		{ 678, 644 },
		{ 1335, 1316 },
		{ 536, 489 },
		{ 541, 495 },
		{ 685, 650 },
		{ 687, 652 },
		{ 688, 653 },
		{ 689, 655 },
		{ 366, 325 },
		{ 698, 665 },
		{ 235, 201 },
		{ 705, 674 },
		{ 707, 676 },
		{ 558, 511 },
		{ 331, 292 },
		{ 721, 690 },
		{ 250, 215 },
		{ 295, 254 },
		{ 343, 303 },
		{ 575, 528 },
		{ 462, 412 },
		{ 746, 715 },
		{ 756, 727 },
		{ 767, 739 },
		{ 463, 413 },
		{ 775, 752 },
		{ 780, 757 },
		{ 187, 160 },
		{ 1077, 1077 },
		{ 763, 735 },
		{ 188, 160 },
		{ 276, 237 },
		{ 296, 255 },
		{ 320, 279 },
		{ 532, 484 },
		{ 227, 193 },
		{ 786, 763 },
		{ 1157, 1157 },
		{ 787, 764 },
		{ 299, 258 },
		{ 618, 579 },
		{ 621, 584 },
		{ 799, 778 },
		{ 805, 785 },
		{ 539, 493 },
		{ 194, 165 },
		{ 376, 334 },
		{ 548, 500 },
		{ 483, 435 },
		{ 1370, 1353 },
		{ 310, 268 },
		{ 1077, 1077 },
		{ 1551, 1536 },
		{ 822, 806 },
		{ 195, 166 },
		{ 1104, 1104 },
		{ 1107, 1107 },
		{ 1110, 1110 },
		{ 1113, 1113 },
		{ 497, 452 },
		{ 1157, 1157 },
		{ 204, 175 },
		{ 830, 817 },
		{ 449, 401 },
		{ 503, 459 },
		{ 726, 695 },
		{ 728, 697 },
		{ 850, 844 },
		{ 1127, 1127 },
		{ 852, 846 },
		{ 353, 314 },
		{ 315, 274 },
		{ 1134, 1134 },
		{ 513, 471 },
		{ 514, 472 },
		{ 207, 178 },
		{ 755, 726 },
		{ 594, 548 },
		{ 1104, 1104 },
		{ 1107, 1107 },
		{ 1110, 1110 },
		{ 1113, 1113 },
		{ 420, 372 },
		{ 422, 374 },
		{ 346, 306 },
		{ 551, 504 },
		{ 1372, 1355 },
		{ 367, 326 },
		{ 1566, 1552 },
		{ 553, 506 },
		{ 1008, 1077 },
		{ 1127, 1127 },
		{ 554, 507 },
		{ 764, 736 },
		{ 308, 264 },
		{ 1134, 1134 },
		{ 349, 309 },
		{ 236, 202 },
		{ 777, 754 },
		{ 1008, 1157 },
		{ 439, 388 },
		{ 781, 758 },
		{ 1388, 1371 },
		{ 443, 393 },
		{ 1592, 1581 },
		{ 1593, 1582 },
		{ 217, 185 },
		{ 384, 337 },
		{ 324, 284 },
		{ 791, 769 },
		{ 508, 465 },
		{ 672, 634 },
		{ 509, 466 },
		{ 588, 542 },
		{ 447, 398 },
		{ 392, 347 },
		{ 313, 271 },
		{ 1008, 1104 },
		{ 1008, 1107 },
		{ 1008, 1110 },
		{ 1008, 1113 },
		{ 1414, 1404 },
		{ 1415, 1405 },
		{ 812, 793 },
		{ 596, 550 },
		{ 814, 796 },
		{ 599, 557 },
		{ 455, 406 },
		{ 183, 158 },
		{ 254, 219 },
		{ 1008, 1127 },
		{ 609, 569 },
		{ 341, 301 },
		{ 611, 571 },
		{ 1008, 1134 },
		{ 833, 821 },
		{ 708, 677 },
		{ 277, 238 },
		{ 844, 836 },
		{ 464, 414 },
		{ 619, 581 },
		{ 535, 488 },
		{ 470, 421 },
		{ 419, 371 },
		{ 540, 494 },
		{ 1554, 1539 },
		{ 733, 702 },
		{ 734, 703 },
		{ 246, 211 },
		{ 174, 152 },
		{ 578, 531 },
		{ 245, 211 },
		{ 333, 293 },
		{ 332, 293 },
		{ 211, 181 },
		{ 441, 391 },
		{ 257, 222 },
		{ 210, 181 },
		{ 258, 222 },
		{ 173, 152 },
		{ 180, 156 },
		{ 710, 679 },
		{ 259, 222 },
		{ 181, 156 },
		{ 720, 689 },
		{ 557, 510 },
		{ 789, 767 },
		{ 859, 855 },
		{ 1334, 1315 },
		{ 711, 679 },
		{ 393, 348 },
		{ 864, 860 },
		{ 1337, 1318 },
		{ 649, 610 },
		{ 421, 373 },
		{ 301, 260 },
		{ 1390, 1373 },
		{ 538, 492 },
		{ 686, 651 },
		{ 458, 409 },
		{ 738, 707 },
		{ 1552, 1537 },
		{ 220, 187 },
		{ 742, 711 },
		{ 1499, 1478 },
		{ 256, 221 },
		{ 1404, 1389 },
		{ 488, 444 },
		{ 751, 721 },
		{ 545, 497 },
		{ 823, 807 },
		{ 615, 575 },
		{ 330, 290 },
		{ 1569, 1555 },
		{ 826, 811 },
		{ 231, 197 },
		{ 828, 813 },
		{ 465, 415 },
		{ 709, 678 },
		{ 835, 824 },
		{ 773, 746 },
		{ 1581, 1568 },
		{ 1519, 1502 },
		{ 466, 416 },
		{ 637, 598 },
		{ 1523, 1506 },
		{ 845, 838 },
		{ 232, 198 },
		{ 716, 685 },
		{ 1371, 1354 },
		{ 718, 687 },
		{ 719, 688 },
		{ 176, 153 },
		{ 175, 153 },
		{ 225, 191 },
		{ 568, 521 },
		{ 487, 443 },
		{ 730, 699 },
		{ 209, 180 },
		{ 1606, 1592 },
		{ 388, 343 },
		{ 224, 191 },
		{ 390, 345 },
		{ 836, 825 },
		{ 1502, 1482 },
		{ 790, 768 },
		{ 1505, 1484 },
		{ 840, 829 },
		{ 221, 188 },
		{ 1425, 1414 },
		{ 702, 670 },
		{ 192, 163 },
		{ 796, 775 },
		{ 744, 713 },
		{ 800, 779 },
		{ 500, 455 },
		{ 527, 479 },
		{ 633, 594 },
		{ 634, 595 },
		{ 858, 854 },
		{ 242, 208 },
		{ 529, 481 },
		{ 641, 602 },
		{ 337, 297 },
		{ 868, 867 },
		{ 818, 801 },
		{ 1315, 1293 },
		{ 1578, 1566 },
		{ 386, 341 },
		{ 1317, 1295 },
		{ 598, 552 },
		{ 506, 462 },
		{ 1403, 1388 },
		{ 507, 463 },
		{ 414, 364 },
		{ 857, 853 },
		{ 768, 741 },
		{ 817, 800 },
		{ 399, 356 },
		{ 862, 858 },
		{ 564, 518 },
		{ 537, 490 },
		{ 436, 385 },
		{ 869, 868 },
		{ 511, 468 },
		{ 602, 562 },
		{ 450, 402 },
		{ 608, 568 },
		{ 573, 526 },
		{ 696, 663 },
		{ 831, 818 },
		{ 1386, 1369 },
		{ 832, 820 },
		{ 496, 451 },
		{ 577, 530 },
		{ 208, 179 },
		{ 1564, 1550 },
		{ 1521, 1504 },
		{ 440, 390 },
		{ 325, 285 },
		{ 243, 209 },
		{ 479, 430 },
		{ 321, 280 },
		{ 846, 840 },
		{ 753, 723 },
		{ 592, 546 },
		{ 714, 682 },
		{ 760, 730 },
		{ 461, 411 },
		{ 338, 298 },
		{ 369, 328 },
		{ 724, 693 },
		{ 556, 509 },
		{ 574, 527 },
		{ 1553, 1538 },
		{ 797, 776 },
		{ 418, 370 },
		{ 576, 529 },
		{ 675, 641 },
		{ 302, 261 },
		{ 362, 321 },
		{ 512, 470 },
		{ 580, 535 },
		{ 683, 648 },
		{ 197, 168 },
		{ 205, 176 },
		{ 636, 597 },
		{ 281, 242 },
		{ 638, 599 },
		{ 555, 508 },
		{ 589, 543 },
		{ 502, 458 },
		{ 837, 826 },
		{ 583, 538 },
		{ 839, 828 },
		{ 389, 344 },
		{ 704, 673 },
		{ 642, 603 },
		{ 740, 709 },
		{ 523, 475 },
		{ 743, 712 },
		{ 795, 773 },
		{ 1540, 1523 },
		{ 272, 234 },
		{ 851, 845 },
		{ 339, 299 },
		{ 547, 499 },
		{ 747, 716 },
		{ 802, 782 },
		{ 804, 784 },
		{ 748, 718 },
		{ 749, 719 },
		{ 808, 789 },
		{ 750, 720 },
		{ 863, 859 },
		{ 591, 545 },
		{ 467, 417 },
		{ 866, 864 },
		{ 457, 408 },
		{ 249, 214 },
		{ 717, 686 },
		{ 762, 732 },
		{ 654, 615 },
		{ 819, 802 },
		{ 684, 649 },
		{ 766, 738 },
		{ 240, 206 },
		{ 345, 305 },
		{ 478, 429 },
		{ 216, 184 },
		{ 1356, 1337 },
		{ 774, 751 },
		{ 725, 694 },
		{ 662, 623 },
		{ 269, 231 },
		{ 691, 657 },
		{ 664, 625 },
		{ 834, 823 },
		{ 784, 761 },
		{ 606, 566 },
		{ 998, 998 },
		{ 273, 235 },
		{ 701, 669 },
		{ 623, 585 },
		{ 622, 585 },
		{ 274, 235 },
		{ 624, 585 },
		{ 559, 513 },
		{ 867, 865 },
		{ 1565, 1551 },
		{ 909, 907 },
		{ 1567, 1553 },
		{ 631, 592 },
		{ 829, 816 },
		{ 1387, 1370 },
		{ 560, 514 },
		{ 433, 382 },
		{ 241, 207 },
		{ 271, 233 },
		{ 438, 387 },
		{ 712, 680 },
		{ 300, 259 },
		{ 328, 288 },
		{ 998, 998 },
		{ 201, 172 },
		{ 193, 164 },
		{ 801, 780 },
		{ 1538, 1521 },
		{ 317, 276 },
		{ 604, 564 },
		{ 682, 647 },
		{ 546, 498 },
		{ 495, 450 },
		{ 239, 205 },
		{ 261, 224 },
		{ 397, 353 },
		{ 336, 296 },
		{ 215, 183 },
		{ 854, 848 },
		{ 452, 404 },
		{ 617, 577 },
		{ 530, 482 },
		{ 779, 756 },
		{ 697, 664 },
		{ 356, 316 },
		{ 699, 666 },
		{ 476, 427 },
		{ 785, 762 },
		{ 352, 312 },
		{ 765, 737 },
		{ 442, 392 },
		{ 340, 300 },
		{ 448, 399 },
		{ 1618, 1608 },
		{ 1263, 1261 },
		{ 1238, 1236 },
		{ 1215, 1213 },
		{ 280, 241 },
		{ 1436, 1427 },
		{ 1586, 1574 },
		{ 1410, 1399 },
		{ 1535, 1518 },
		{ 1008, 998 },
		{ 278, 239 },
		{ 1352, 1333 },
		{ 950, 948 },
		{ 1163, 1161 },
		{ 1190, 1188 },
		{ 757, 728 },
		{ 219, 186 },
		{ 327, 287 },
		{ 375, 333 },
		{ 758, 728 },
		{ 218, 186 },
		{ 572, 525 },
		{ 626, 587 },
		{ 423, 375 },
		{ 368, 327 },
		{ 590, 544 },
		{ 803, 783 },
		{ 203, 174 },
		{ 847, 841 },
		{ 412, 363 },
		{ 413, 363 },
		{ 1405, 1390 },
		{ 736, 705 },
		{ 451, 403 },
		{ 370, 329 },
		{ 371, 330 },
		{ 815, 797 },
		{ 1412, 1402 },
		{ 1539, 1522 },
		{ 282, 243 },
		{ 741, 710 },
		{ 674, 636 },
		{ 266, 228 },
		{ 350, 310 },
		{ 821, 805 },
		{ 677, 643 },
		{ 285, 246 },
		{ 251, 216 },
		{ 202, 173 },
		{ 1550, 1535 },
		{ 614, 574 },
		{ 561, 515 },
		{ 562, 516 },
		{ 291, 250 },
		{ 165, 146 },
		{ 182, 157 },
		{ 430, 379 },
		{ 168, 150 },
		{ 628, 589 },
		{ 432, 381 },
		{ 692, 659 },
		{ 693, 660 },
		{ 571, 524 },
		{ 468, 418 },
		{ 179, 155 },
		{ 1355, 1336 },
		{ 769, 742 },
		{ 843, 835 },
		{ 770, 743 },
		{ 434, 383 },
		{ 772, 745 },
		{ 391, 346 },
		{ 525, 477 },
		{ 639, 600 },
		{ 703, 671 },
		{ 778, 755 },
		{ 640, 601 },
		{ 1582, 1569 },
		{ 361, 320 },
		{ 706, 675 },
		{ 1369, 1352 },
		{ 298, 257 },
		{ 1588, 1577 },
		{ 1590, 1579 },
		{ 1591, 1580 },
		{ 396, 351 },
		{ 238, 204 },
		{ 1506, 1485 },
		{ 645, 606 },
		{ 531, 483 },
		{ 229, 195 },
		{ 365, 324 },
		{ 400, 357 },
		{ 485, 440 },
		{ 486, 442 },
		{ 793, 771 },
		{ 322, 281 },
		{ 408, 359 },
		{ 409, 360 },
		{ 657, 618 },
		{ 722, 691 },
		{ 491, 446 },
		{ 492, 447 },
		{ 661, 622 },
		{ 344, 304 },
		{ 494, 449 },
		{ 411, 362 },
		{ 1318, 1296 },
		{ 806, 786 },
		{ 230, 196 },
		{ 550, 503 },
		{ 264, 226 },
		{ 1598, 1598 },
		{ 1598, 1598 },
		{ 1600, 1600 },
		{ 1600, 1600 },
		{ 1602, 1602 },
		{ 1602, 1602 },
		{ 1604, 1604 },
		{ 1604, 1604 },
		{ 1434, 1434 },
		{ 1434, 1434 },
		{ 1158, 1158 },
		{ 1158, 1158 },
		{ 1616, 1616 },
		{ 1616, 1616 },
		{ 1440, 1440 },
		{ 1440, 1440 },
		{ 1621, 1621 },
		{ 1621, 1621 },
		{ 1400, 1400 },
		{ 1400, 1400 },
		{ 1575, 1575 },
		{ 1575, 1575 },
		{ 646, 607 },
		{ 1598, 1598 },
		{ 613, 573 },
		{ 1600, 1600 },
		{ 247, 212 },
		{ 1602, 1602 },
		{ 582, 537 },
		{ 1604, 1604 },
		{ 233, 199 },
		{ 1434, 1434 },
		{ 394, 349 },
		{ 1158, 1158 },
		{ 1656, 1654 },
		{ 1616, 1616 },
		{ 477, 428 },
		{ 1440, 1440 },
		{ 484, 436 },
		{ 1621, 1621 },
		{ 294, 253 },
		{ 1400, 1400 },
		{ 481, 432 },
		{ 1575, 1575 },
		{ 1128, 1128 },
		{ 1128, 1128 },
		{ 1078, 1078 },
		{ 1078, 1078 },
		{ 1114, 1114 },
		{ 1114, 1114 },
		{ 1421, 1421 },
		{ 1421, 1421 },
		{ 1599, 1598 },
		{ 861, 857 },
		{ 1601, 1600 },
		{ 1375, 1359 },
		{ 1603, 1602 },
		{ 1244, 1243 },
		{ 1605, 1604 },
		{ 329, 289 },
		{ 1435, 1434 },
		{ 759, 729 },
		{ 1159, 1158 },
		{ 665, 626 },
		{ 1617, 1616 },
		{ 1169, 1168 },
		{ 1441, 1440 },
		{ 1128, 1128 },
		{ 1622, 1621 },
		{ 1078, 1078 },
		{ 1401, 1400 },
		{ 1114, 1114 },
		{ 1576, 1575 },
		{ 1421, 1421 },
		{ 1423, 1423 },
		{ 1423, 1423 },
		{ 1637, 1637 },
		{ 1637, 1637 },
		{ 1108, 1108 },
		{ 1108, 1108 },
		{ 1456, 1456 },
		{ 1456, 1456 },
		{ 1105, 1105 },
		{ 1105, 1105 },
		{ 1111, 1111 },
		{ 1111, 1111 },
		{ 1135, 1135 },
		{ 1135, 1135 },
		{ 1221, 1220 },
		{ 1100, 1087 },
		{ 1196, 1195 },
		{ 727, 696 },
		{ 629, 590 },
		{ 1607, 1593 },
		{ 265, 227 },
		{ 469, 419 },
		{ 1129, 1128 },
		{ 1423, 1423 },
		{ 1079, 1078 },
		{ 1637, 1637 },
		{ 1115, 1114 },
		{ 1108, 1108 },
		{ 1422, 1421 },
		{ 1456, 1456 },
		{ 731, 700 },
		{ 1105, 1105 },
		{ 659, 620 },
		{ 1111, 1111 },
		{ 167, 148 },
		{ 1135, 1135 },
		{ 1426, 1415 },
		{ 694, 661 },
		{ 956, 955 },
		{ 1395, 1380 },
		{ 292, 251 },
		{ 752, 722 },
		{ 323, 283 },
		{ 673, 635 },
		{ 1269, 1268 },
		{ 842, 832 },
		{ 1557, 1543 },
		{ 1615, 1606 },
		{ 437, 386 },
		{ 1501, 1481 },
		{ 1433, 1425 },
		{ 1413, 1403 },
		{ 1424, 1423 },
		{ 616, 576 },
		{ 1638, 1637 },
		{ 1589, 1578 },
		{ 1109, 1108 },
		{ 534, 487 },
		{ 1457, 1456 },
		{ 189, 161 },
		{ 1106, 1105 },
		{ 1314, 1292 },
		{ 1112, 1111 },
		{ 303, 262 },
		{ 1136, 1135 },
		{ 905, 905 },
		{ 905, 905 },
		{ 1832, 1832 },
		{ 1832, 1832 },
		{ 1211, 1211 },
		{ 1211, 1211 },
		{ 931, 931 },
		{ 931, 931 },
		{ 1234, 1234 },
		{ 1234, 1234 },
		{ 1973, 1973 },
		{ 1973, 1973 },
		{ 1874, 1874 },
		{ 1874, 1874 },
		{ 2015, 2015 },
		{ 2015, 2015 },
		{ 1259, 1259 },
		{ 1259, 1259 },
		{ 1959, 1959 },
		{ 1959, 1959 },
		{ 973, 973 },
		{ 973, 973 },
		{ 603, 563 },
		{ 905, 905 },
		{ 761, 731 },
		{ 1832, 1832 },
		{ 270, 232 },
		{ 1211, 1211 },
		{ 206, 177 },
		{ 931, 931 },
		{ 395, 350 },
		{ 1234, 1234 },
		{ 810, 791 },
		{ 1973, 1973 },
		{ 607, 567 },
		{ 1874, 1874 },
		{ 309, 265 },
		{ 2015, 2015 },
		{ 549, 501 },
		{ 1259, 1259 },
		{ 648, 609 },
		{ 1959, 1959 },
		{ 234, 200 },
		{ 973, 973 },
		{ 374, 332 },
		{ 2019, 2019 },
		{ 2019, 2019 },
		{ 1961, 1961 },
		{ 1961, 1961 },
		{ 906, 905 },
		{ 311, 269 },
		{ 1833, 1832 },
		{ 581, 536 },
		{ 1212, 1211 },
		{ 228, 194 },
		{ 932, 931 },
		{ 526, 478 },
		{ 1235, 1234 },
		{ 732, 701 },
		{ 1995, 1973 },
		{ 776, 753 },
		{ 1875, 1874 },
		{ 473, 424 },
		{ 2033, 2015 },
		{ 326, 286 },
		{ 1260, 1259 },
		{ 735, 704 },
		{ 1981, 1959 },
		{ 2019, 2019 },
		{ 974, 973 },
		{ 1961, 1961 },
		{ 695, 662 },
		{ 2052, 2052 },
		{ 2052, 2052 },
		{ 2115, 2115 },
		{ 2115, 2115 },
		{ 1792, 1792 },
		{ 1792, 1792 },
		{ 2054, 2054 },
		{ 2054, 2054 },
		{ 946, 946 },
		{ 946, 946 },
		{ 2056, 2056 },
		{ 2056, 2056 },
		{ 2057, 2057 },
		{ 2057, 2057 },
		{ 2023, 2023 },
		{ 2023, 2023 },
		{ 2024, 2024 },
		{ 2024, 2024 },
		{ 2060, 2060 },
		{ 2060, 2060 },
		{ 2025, 2025 },
		{ 2025, 2025 },
		{ 2037, 2019 },
		{ 2052, 2052 },
		{ 1983, 1961 },
		{ 2115, 2115 },
		{ 475, 426 },
		{ 1792, 1792 },
		{ 2119, 2117 },
		{ 2054, 2054 },
		{ 782, 759 },
		{ 946, 946 },
		{ 499, 454 },
		{ 2056, 2056 },
		{ 620, 583 },
		{ 2057, 2057 },
		{ 342, 302 },
		{ 2023, 2023 },
		{ 284, 245 },
		{ 2024, 2024 },
		{ 625, 586 },
		{ 2060, 2060 },
		{ 385, 339 },
		{ 2025, 2025 },
		{ 627, 588 },
		{ 2026, 2026 },
		{ 2026, 2026 },
		{ 1186, 1186 },
		{ 1186, 1186 },
		{ 2063, 2052 },
		{ 275, 236 },
		{ 2116, 2115 },
		{ 504, 460 },
		{ 1793, 1792 },
		{ 630, 591 },
		{ 2065, 2054 },
		{ 253, 218 },
		{ 947, 946 },
		{ 632, 593 },
		{ 2067, 2056 },
		{ 267, 229 },
		{ 2068, 2057 },
		{ 347, 307 },
		{ 2041, 2023 },
		{ 597, 551 },
		{ 2042, 2024 },
		{ 798, 777 },
		{ 2071, 2060 },
		{ 2026, 2026 },
		{ 2043, 2025 },
		{ 1186, 1186 },
		{ 268, 230 },
		{ 2066, 2066 },
		{ 2066, 2066 },
		{ 2069, 2069 },
		{ 2069, 2069 },
		{ 2070, 2070 },
		{ 2070, 2070 },
		{ 1919, 1919 },
		{ 1919, 1919 },
		{ 2029, 2029 },
		{ 2029, 2029 },
		{ 2074, 2074 },
		{ 2074, 2074 },
		{ 2078, 2078 },
		{ 2078, 2078 },
		{ 1985, 1985 },
		{ 1985, 1985 },
		{ 2140, 2140 },
		{ 2140, 2140 },
		{ 1282, 1282 },
		{ 1282, 1282 },
		{ 2083, 2083 },
		{ 2083, 2083 },
		{ 2044, 2026 },
		{ 2066, 2066 },
		{ 1187, 1186 },
		{ 2069, 2069 },
		{ 754, 725 },
		{ 2070, 2070 },
		{ 569, 522 },
		{ 1919, 1919 },
		{ 417, 369 },
		{ 2029, 2029 },
		{ 166, 147 },
		{ 2074, 2074 },
		{ 1160, 1159 },
		{ 2078, 2078 },
		{ 1639, 1638 },
		{ 1985, 1985 },
		{ 1411, 1401 },
		{ 2140, 2140 },
		{ 1442, 1435 },
		{ 1282, 1282 },
		{ 1431, 1422 },
		{ 2083, 2083 },
		{ 1587, 1576 },
		{ 1677, 1677 },
		{ 1677, 1677 },
		{ 1623, 1617 },
		{ 1122, 1106 },
		{ 2075, 2066 },
		{ 1432, 1424 },
		{ 2076, 2069 },
		{ 1458, 1457 },
		{ 2077, 2070 },
		{ 1124, 1112 },
		{ 1941, 1919 },
		{ 1627, 1622 },
		{ 2047, 2029 },
		{ 1446, 1441 },
		{ 2080, 2074 },
		{ 1611, 1599 },
		{ 2081, 2078 },
		{ 1123, 1109 },
		{ 2005, 1985 },
		{ 1612, 1601 },
		{ 2141, 2140 },
		{ 1092, 1079 },
		{ 1283, 1282 },
		{ 1677, 1677 },
		{ 2084, 2083 },
		{ 1613, 1603 },
		{ 1137, 1129 },
		{ 1614, 1605 },
		{ 1142, 1136 },
		{ 1125, 1115 },
		{ 962, 961 },
		{ 1151, 1147 },
		{ 1201, 1200 },
		{ 1202, 1201 },
		{ 1147, 1143 },
		{ 1148, 1144 },
		{ 1274, 1273 },
		{ 1275, 1274 },
		{ 1174, 1173 },
		{ 1175, 1174 },
		{ 1249, 1248 },
		{ 1250, 1249 },
		{ 1226, 1225 },
		{ 1227, 1226 },
		{ 961, 960 },
		{ 2090, 2090 },
		{ 2087, 2090 },
		{ 135, 135 },
		{ 132, 135 },
		{ 1678, 1677 },
		{ 1384, 1368 },
		{ 1385, 1368 },
		{ 1397, 1382 },
		{ 1398, 1382 },
		{ 1644, 1644 },
		{ 1463, 1463 },
		{ 2095, 2091 },
		{ 140, 136 },
		{ 1468, 1464 },
		{ 1698, 1679 },
		{ 65, 47 },
		{ 2094, 2091 },
		{ 139, 136 },
		{ 1467, 1464 },
		{ 1697, 1679 },
		{ 64, 47 },
		{ 2089, 2085 },
		{ 1649, 1645 },
		{ 2090, 2090 },
		{ 134, 130 },
		{ 135, 135 },
		{ 2088, 2085 },
		{ 1648, 1645 },
		{ 1345, 1326 },
		{ 133, 130 },
		{ 1469, 1466 },
		{ 1471, 1470 },
		{ 1644, 1644 },
		{ 1463, 1463 },
		{ 143, 142 },
		{ 1530, 1513 },
		{ 1744, 1720 },
		{ 94, 79 },
		{ 2091, 2090 },
		{ 2096, 2093 },
		{ 136, 135 },
		{ 2098, 2097 },
		{ 1650, 1647 },
		{ 1652, 1651 },
		{ 141, 138 },
		{ 1720, 1698 },
		{ 1647, 1643 },
		{ 1645, 1644 },
		{ 1464, 1463 },
		{ 2093, 2089 },
		{ 138, 134 },
		{ 1466, 1462 },
		{ 79, 65 },
		{ 2097, 2095 },
		{ 1651, 1649 },
		{ 142, 140 },
		{ 1513, 1495 },
		{ 1326, 1306 },
		{ 2037, 2037 },
		{ 2037, 2037 },
		{ 1983, 1983 },
		{ 1983, 1983 },
		{ 2063, 2063 },
		{ 2063, 2063 },
		{ 0, 2105 },
		{ 143, 143 },
		{ 144, 143 },
		{ 2065, 2065 },
		{ 2065, 2065 },
		{ 1941, 1941 },
		{ 1941, 1941 },
		{ 2067, 2067 },
		{ 2067, 2067 },
		{ 2068, 2068 },
		{ 2068, 2068 },
		{ 2041, 2041 },
		{ 2041, 2041 },
		{ 2042, 2042 },
		{ 2042, 2042 },
		{ 2071, 2071 },
		{ 2071, 2071 },
		{ 2037, 2037 },
		{ 0, 892 },
		{ 1983, 1983 },
		{ 0, 2120 },
		{ 2063, 2063 },
		{ 1652, 1652 },
		{ 1653, 1652 },
		{ 143, 143 },
		{ 0, 2125 },
		{ 2065, 2065 },
		{ 0, 938 },
		{ 1941, 1941 },
		{ 0, 1666 },
		{ 2067, 2067 },
		{ 0, 2129 },
		{ 2068, 2068 },
		{ 0, 1935 },
		{ 2041, 2041 },
		{ 0, 922 },
		{ 2042, 2042 },
		{ 0, 1494 },
		{ 2071, 2071 },
		{ 2044, 2044 },
		{ 2044, 2044 },
		{ 2075, 2075 },
		{ 2075, 2075 },
		{ 2076, 2076 },
		{ 2076, 2076 },
		{ 1652, 1652 },
		{ 2077, 2077 },
		{ 2077, 2077 },
		{ 2005, 2005 },
		{ 2005, 2005 },
		{ 2047, 2047 },
		{ 2047, 2047 },
		{ 2080, 2080 },
		{ 2080, 2080 },
		{ 2081, 2081 },
		{ 2081, 2081 },
		{ 2084, 2084 },
		{ 2084, 2084 },
		{ 1867, 1848 },
		{ 1867, 1848 },
		{ 1793, 1793 },
		{ 1793, 1793 },
		{ 2044, 2044 },
		{ 1646, 1648 },
		{ 2075, 2075 },
		{ 0, 1697 },
		{ 2076, 2076 },
		{ 1995, 1995 },
		{ 1995, 1995 },
		{ 2077, 2077 },
		{ 872, 872 },
		{ 2005, 2005 },
		{ 137, 133 },
		{ 2047, 2047 },
		{ 0, 0 },
		{ 2080, 2080 },
		{ 0, 0 },
		{ 2081, 2081 },
		{ 0, 0 },
		{ 2084, 2084 },
		{ 0, 0 },
		{ 1867, 1848 },
		{ 0, 0 },
		{ 1793, 1793 },
		{ 1471, 1471 },
		{ 1472, 1471 },
		{ 2033, 2033 },
		{ 2033, 2033 },
		{ 2098, 2098 },
		{ 2099, 2098 },
		{ 1995, 1995 },
		{ 1981, 1981 },
		{ 1981, 1981 },
		{ 872, 872 },
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
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1471, 1471 },
		{ 0, 0 },
		{ 2033, 2033 },
		{ 0, 0 },
		{ 2098, 2098 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1981, 1981 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -46, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 38, 229, 0 },
		{ -149, 2143, 0 },
		{ 5, 0, 0 },
		{ -871, 927, -25 },
		{ 7, 0, -25 },
		{ -875, 1620, -27 },
		{ 9, 0, -27 },
		{ -888, 2446, 93 },
		{ 11, 0, 93 },
		{ -908, 2439, 99 },
		{ 13, 0, 99 },
		{ -934, 2447, 0 },
		{ 15, 0, 0 },
		{ -949, 2441, 90 },
		{ 17, 0, 90 },
		{ -986, 457, 0 },
		{ 19, 0, 0 },
		{ -1162, 2443, 0 },
		{ 21, 0, 0 },
		{ -1189, 2444, 0 },
		{ 23, 0, 0 },
		{ -1214, 2445, 0 },
		{ 25, 0, 0 },
		{ -1237, 2437, 0 },
		{ 27, 0, 0 },
		{ -1262, 2442, 178 },
		{ 29, 0, 178 },
		{ 32, 126, 0 },
		{ -1299, 1, 0 },
		{ 34, 128, 0 },
		{ -1488, 115, 0 },
		{ -1655, 2440, 0 },
		{ 35, 0, 0 },
		{ 38, 14, 0 },
		{ -63, 343, 0 },
		{ -2101, 2436, 96 },
		{ 39, 0, 96 },
		{ -2118, 2438, 102 },
		{ 41, 0, 102 },
		{ 2143, 1439, 0 },
		{ 43, 0, 0 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 2097, 3456, 313 },
		{ 0, 0, 157 },
		{ 0, 0, 159 },
		{ 129, 559, 175 },
		{ 129, 584, 175 },
		{ 129, 591, 175 },
		{ 129, 592, 175 },
		{ 129, 624, 175 },
		{ 129, 629, 175 },
		{ 129, 623, 175 },
		{ 2133, 2152, 313 },
		{ 129, 634, 175 },
		{ 129, 635, 175 },
		{ 2133, 1461, 174 },
		{ 78, 1932, 313 },
		{ 129, 0, 175 },
		{ 0, 0, 313 },
		{ -64, 21, 153 },
		{ -65, 3493, 0 },
		{ 129, 626, 175 },
		{ 129, 789, 175 },
		{ 129, 775, 175 },
		{ 129, 793, 175 },
		{ 129, 880, 175 },
		{ 129, 963, 175 },
		{ 129, 1063, 175 },
		{ 2139, 1790, 0 },
		{ 129, 1059, 175 },
		{ 129, 1055, 175 },
		{ 2133, 1523, 171 },
		{ 92, 1350, 0 },
		{ 2133, 1500, 172 },
		{ 2097, 3473, 0 },
		{ 129, 1065, 175 },
		{ 129, 1097, 175 },
		{ 129, 0, 163 },
		{ 129, 1117, 175 },
		{ 129, 1147, 175 },
		{ 129, 1131, 175 },
		{ 129, 1136, 175 },
		{ 2055, 2171, 0 },
		{ 129, 1137, 175 },
		{ 129, 1183, 175 },
		{ 105, 1278, 0 },
		{ 92, 0, 0 },
		{ 1766, 2005, 173 },
		{ 107, 1318, 0 },
		{ 0, 0, 155 },
		{ 129, 1187, 160 },
		{ 129, 1184, 175 },
		{ 129, 1183, 175 },
		{ 129, 1202, 175 },
		{ 129, 0, 166 },
		{ 129, 1203, 175 },
		{ 0, 0, 168 },
		{ 129, 485, 175 },
		{ 129, 484, 175 },
		{ 105, 0, 0 },
		{ 1766, 2032, 171 },
		{ 107, 0, 0 },
		{ 1766, 1942, 172 },
		{ 129, 499, 175 },
		{ 129, 524, 175 },
		{ 129, 523, 175 },
		{ 129, 0, 165 },
		{ 129, 524, 175 },
		{ 129, 525, 175 },
		{ 129, 528, 175 },
		{ 129, 555, 175 },
		{ 129, 0, 162 },
		{ 129, 0, 164 },
		{ 129, 552, 175 },
		{ 129, 600, 175 },
		{ 129, 553, 175 },
		{ 129, 0, 161 },
		{ 129, 0, 167 },
		{ 129, 561, 175 },
		{ 129, 7, 175 },
		{ 129, 574, 175 },
		{ 0, 0, 170 },
		{ 129, 583, 175 },
		{ 129, 584, 175 },
		{ 2139, 1077, 169 },
		{ 2097, 3465, 313 },
		{ 135, 0, 157 },
		{ 0, 0, 158 },
		{ -133, 3609, 153 },
		{ -134, 3491, 0 },
		{ 2129, 3476, 0 },
		{ 2097, 3453, 0 },
		{ 0, 0, 154 },
		{ 2097, 3480, 0 },
		{ -139, 19, 0 },
		{ -140, 3496, 0 },
		{ 143, 0, 155 },
		{ 2097, 3470, 0 },
		{ 2129, 3539, 0 },
		{ 0, 0, 156 },
		{ 2117, 1362, 88 },
		{ 1580, 2983, 88 },
		{ 2117, 3360, 88 },
		{ 1593, 3151, 88 },
		{ 0, 0, 88 },
		{ 1580, 2986, 0 },
		{ 1592, 2240, 0 },
		{ 1568, 2664, 0 },
		{ 1550, 2720, 0 },
		{ 1550, 2463, 0 },
		{ 1580, 2993, 0 },
		{ 1592, 2669, 0 },
		{ 1580, 2984, 0 },
		{ 1582, 2631, 0 },
		{ 2100, 2404, 0 },
		{ 1592, 2533, 0 },
		{ 1606, 3177, 0 },
		{ 2100, 2420, 0 },
		{ 1592, 2737, 0 },
		{ 1553, 2897, 0 },
		{ 1536, 2547, 0 },
		{ 1536, 2556, 0 },
		{ 1554, 2345, 0 },
		{ 1538, 2812, 0 },
		{ 1554, 2292, 0 },
		{ 1554, 2307, 0 },
		{ 2100, 2436, 0 },
		{ 1553, 2896, 0 },
		{ 1580, 2977, 0 },
		{ 841, 2955, 0 },
		{ 1536, 2563, 0 },
		{ 1538, 2813, 0 },
		{ 2117, 3212, 0 },
		{ 1536, 2577, 0 },
		{ 1550, 2782, 0 },
		{ 1592, 2724, 0 },
		{ 1568, 2662, 0 },
		{ 2117, 2477, 0 },
		{ 1553, 2909, 0 },
		{ 1523, 2858, 0 },
		{ 1582, 2609, 0 },
		{ 2117, 2949, 0 },
		{ 1568, 2687, 0 },
		{ 1592, 2734, 0 },
		{ 1554, 2308, 0 },
		{ 1503, 2440, 0 },
		{ 1608, 2728, 0 },
		{ 1505, 2416, 0 },
		{ 1536, 2537, 0 },
		{ 2117, 3238, 0 },
		{ 1580, 3019, 0 },
		{ 1580, 3038, 0 },
		{ 1568, 2700, 0 },
		{ 1568, 2712, 0 },
		{ 1654, 3072, 0 },
		{ 2117, 3226, 0 },
		{ 1503, 2511, 0 },
		{ 1582, 2600, 0 },
		{ 1554, 2310, 0 },
		{ 1580, 3015, 0 },
		{ 1553, 2905, 0 },
		{ 1523, 2855, 0 },
		{ 1553, 2889, 0 },
		{ 1592, 2746, 0 },
		{ 1550, 2787, 0 },
		{ 809, 2387, 0 },
		{ 1568, 2656, 0 },
		{ 1654, 3068, 0 },
		{ 1554, 2326, 0 },
		{ 1523, 2848, 0 },
		{ 1503, 2517, 0 },
		{ 1580, 2976, 0 },
		{ 1505, 2412, 0 },
		{ 2117, 3312, 0 },
		{ 1582, 2632, 0 },
		{ 1554, 2329, 0 },
		{ 1568, 2690, 0 },
		{ 1608, 2670, 0 },
		{ 1554, 2334, 0 },
		{ 1553, 2906, 0 },
		{ 1582, 2289, 0 },
		{ 1580, 3040, 0 },
		{ 1593, 3137, 0 },
		{ 1580, 2971, 0 },
		{ 2117, 3316, 0 },
		{ 2117, 3327, 0 },
		{ 1523, 2863, 0 },
		{ 2117, 3210, 0 },
		{ 1553, 2890, 0 },
		{ 1523, 2832, 0 },
		{ 1580, 2877, 0 },
		{ 2117, 3306, 0 },
		{ 1536, 2533, 0 },
		{ 1582, 2640, 0 },
		{ 1608, 2937, 0 },
		{ 809, 2391, 0 },
		{ 1608, 2931, 0 },
		{ 1538, 2815, 0 },
		{ 1580, 2968, 0 },
		{ 1554, 2288, 0 },
		{ 2117, 3294, 0 },
		{ 1580, 2975, 0 },
		{ 0, 0, 23 },
		{ 1592, 2495, 0 },
		{ 2117, 2304, 0 },
		{ 1580, 2982, 0 },
		{ 1593, 3157, 0 },
		{ 1554, 2293, 0 },
		{ 1654, 3082, 0 },
		{ 1503, 2518, 0 },
		{ 1536, 2534, 0 },
		{ 1554, 2294, 0 },
		{ 1580, 3010, 0 },
		{ 1536, 2541, 0 },
		{ 1553, 2893, 0 },
		{ 1568, 2680, 0 },
		{ 1538, 2807, 0 },
		{ 2117, 3182, 0 },
		{ 1593, 2394, 0 },
		{ 1582, 2597, 0 },
		{ 2117, 3220, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 1536, 2552, 0 },
		{ 2117, 3234, 0 },
		{ 1503, 2475, 0 },
		{ 1582, 2619, 0 },
		{ 0, 0, 21 },
		{ 1554, 2296, 0 },
		{ 1536, 2573, 0 },
		{ 1503, 2486, 0 },
		{ 1553, 2900, 0 },
		{ 1503, 2493, 0 },
		{ 1554, 2297, 0 },
		{ 1536, 2535, 0 },
		{ 1550, 2789, 0 },
		{ 1580, 3025, 0 },
		{ 0, 0, 18 },
		{ 1593, 3159, 0 },
		{ 1582, 2611, 0 },
		{ 1550, 2786, 0 },
		{ 2117, 3248, 0 },
		{ 841, 2945, 0 },
		{ 1553, 2894, 0 },
		{ 1593, 3102, 0 },
		{ 1568, 2697, 0 },
		{ 0, 0, 25 },
		{ 1503, 2515, 0 },
		{ 1592, 2658, 0 },
		{ 809, 2385, 0 },
		{ 1554, 2300, 0 },
		{ 1553, 2908, 0 },
		{ 1592, 2749, 0 },
		{ 1550, 2796, 0 },
		{ 1523, 2834, 0 },
		{ 1608, 2925, 0 },
		{ 1582, 2635, 0 },
		{ 2117, 3292, 0 },
		{ 1503, 2519, 0 },
		{ 1580, 3033, 0 },
		{ 1523, 2856, 0 },
		{ 1582, 2587, 0 },
		{ 2117, 3318, 0 },
		{ 1554, 2302, 0 },
		{ 1582, 2599, 0 },
		{ 1580, 2972, 0 },
		{ 1503, 2441, 0 },
		{ 737, 2921, 0 },
		{ 0, 0, 7 },
		{ 1536, 2572, 0 },
		{ 1538, 2429, 0 },
		{ 1553, 2916, 0 },
		{ 1565, 2375, 0 },
		{ 1554, 2303, 0 },
		{ 1582, 2440, 0 },
		{ 1580, 3007, 0 },
		{ 1538, 2808, 0 },
		{ 1554, 2306, 0 },
		{ 1503, 2499, 0 },
		{ 1580, 3020, 0 },
		{ 1503, 2509, 0 },
		{ 1582, 2590, 0 },
		{ 841, 2952, 0 },
		{ 1550, 2797, 0 },
		{ 1580, 2963, 0 },
		{ 1580, 2964, 0 },
		{ 1592, 2291, 0 },
		{ 2117, 3228, 0 },
		{ 841, 2946, 0 },
		{ 1536, 2548, 0 },
		{ 1608, 2213, 0 },
		{ 1565, 2377, 0 },
		{ 1582, 2610, 0 },
		{ 0, 0, 48 },
		{ 2117, 3298, 0 },
		{ 0, 0, 66 },
		{ 1592, 2754, 0 },
		{ 681, 2394, 0 },
		{ 1592, 2726, 0 },
		{ 1523, 2824, 0 },
		{ 1592, 2728, 0 },
		{ 1580, 3000, 0 },
		{ 1582, 2618, 0 },
		{ 1568, 2675, 0 },
		{ 1654, 3074, 0 },
		{ 2117, 3214, 0 },
		{ 1580, 3014, 0 },
		{ 0, 0, 6 },
		{ 1553, 2907, 0 },
		{ 1554, 2309, 0 },
		{ 0, 0, 60 },
		{ 1550, 2765, 0 },
		{ 1580, 3021, 0 },
		{ 2100, 1872, 0 },
		{ 1580, 3026, 0 },
		{ 1580, 3027, 0 },
		{ 1554, 2311, 0 },
		{ 1580, 3035, 0 },
		{ 1654, 2960, 0 },
		{ 1592, 2760, 0 },
		{ 0, 0, 22 },
		{ 0, 0, 57 },
		{ 1554, 2312, 40 },
		{ 1554, 2314, 42 },
		{ 2117, 3358, 0 },
		{ 1538, 2804, 0 },
		{ 1582, 2646, 0 },
		{ 1582, 2585, 0 },
		{ 1568, 2679, 0 },
		{ 1582, 2586, 0 },
		{ 841, 2951, 0 },
		{ 1503, 2476, 0 },
		{ 1592, 2404, 0 },
		{ 1550, 2414, 0 },
		{ 1580, 2985, 0 },
		{ 2100, 2430, 0 },
		{ 1580, 2988, 0 },
		{ 1553, 2888, 0 },
		{ 1580, 2998, 0 },
		{ 1554, 2315, 0 },
		{ 1550, 2769, 0 },
		{ 1606, 3166, 0 },
		{ 1553, 2891, 0 },
		{ 1582, 2603, 0 },
		{ 0, 0, 64 },
		{ 1550, 2785, 0 },
		{ 531, 2659, 0 },
		{ 1608, 2924, 0 },
		{ 1582, 2606, 0 },
		{ 1554, 2318, 0 },
		{ 0, 0, 8 },
		{ 1554, 2319, 0 },
		{ 1565, 2373, 0 },
		{ 1582, 2617, 0 },
		{ 1608, 2926, 0 },
		{ 0, 0, 38 },
		{ 1536, 2565, 0 },
		{ 1550, 2773, 0 },
		{ 1580, 2962, 0 },
		{ 1553, 2911, 0 },
		{ 1592, 2405, 0 },
		{ 1582, 2630, 0 },
		{ 2100, 2427, 0 },
		{ 1523, 2847, 0 },
		{ 1568, 2684, 0 },
		{ 2100, 2415, 0 },
		{ 1550, 2795, 0 },
		{ 1503, 2521, 0 },
		{ 1503, 2525, 0 },
		{ 1582, 2642, 0 },
		{ 1568, 2702, 0 },
		{ 1568, 2708, 0 },
		{ 1523, 2845, 0 },
		{ 1580, 2992, 0 },
		{ 1593, 3138, 0 },
		{ 0, 0, 35 },
		{ 1582, 2645, 0 },
		{ 1554, 2320, 0 },
		{ 1554, 2321, 0 },
		{ 2117, 3246, 0 },
		{ 1554, 2322, 0 },
		{ 2117, 3282, 0 },
		{ 1553, 2918, 0 },
		{ 1654, 3078, 0 },
		{ 1523, 2857, 0 },
		{ 1550, 2788, 0 },
		{ 1503, 2467, 0 },
		{ 1593, 3085, 0 },
		{ 1503, 2468, 0 },
		{ 0, 0, 37 },
		{ 1536, 2550, 0 },
		{ 1654, 3080, 0 },
		{ 0, 0, 51 },
		{ 0, 0, 58 },
		{ 0, 0, 63 },
		{ 1580, 3022, 0 },
		{ 0, 0, 67 },
		{ 1580, 3023, 0 },
		{ 1592, 2722, 0 },
		{ 1568, 2692, 0 },
		{ 1580, 2457, 0 },
		{ 1580, 3030, 0 },
		{ 1580, 3031, 0 },
		{ 1554, 2323, 0 },
		{ 1580, 3034, 0 },
		{ 1553, 2904, 0 },
		{ 1550, 2780, 0 },
		{ 1536, 2561, 0 },
		{ 1554, 2324, 0 },
		{ 2117, 3288, 0 },
		{ 1592, 2741, 0 },
		{ 1503, 2483, 0 },
		{ 0, 0, 61 },
		{ 1523, 2820, 0 },
		{ 1536, 2566, 0 },
		{ 2117, 3308, 0 },
		{ 1503, 2485, 0 },
		{ 1592, 2757, 0 },
		{ 1592, 2759, 0 },
		{ 0, 0, 19 },
		{ 1582, 2613, 0 },
		{ 1582, 2615, 0 },
		{ 1554, 2325, 44 },
		{ 1550, 2771, 0 },
		{ 0, 0, 54 },
		{ 1538, 2809, 0 },
		{ 1536, 2575, 0 },
		{ 1536, 2576, 0 },
		{ 1503, 2487, 0 },
		{ 1608, 2052, 0 },
		{ 1523, 2828, 0 },
		{ 1503, 2496, 0 },
		{ 1580, 3001, 0 },
		{ 2117, 3240, 0 },
		{ 1592, 2742, 0 },
		{ 1554, 2328, 0 },
		{ 1592, 2747, 0 },
		{ 1553, 2913, 0 },
		{ 1580, 3018, 0 },
		{ 1536, 2536, 0 },
		{ 0, 0, 65 },
		{ 1503, 2500, 0 },
		{ 1606, 3175, 0 },
		{ 1582, 2644, 0 },
		{ 1503, 2503, 0 },
		{ 1550, 2768, 0 },
		{ 0, 0, 39 },
		{ 1568, 2682, 0 },
		{ 1536, 2546, 0 },
		{ 1582, 2647, 0 },
		{ 1503, 2504, 0 },
		{ 1592, 2227, 0 },
		{ 1568, 2694, 0 },
		{ 1553, 2903, 0 },
		{ 1523, 2835, 0 },
		{ 1536, 2549, 0 },
		{ 2117, 3222, 0 },
		{ 0, 0, 62 },
		{ 1580, 3039, 0 },
		{ 1582, 2588, 0 },
		{ 1554, 2330, 0 },
		{ 1582, 2592, 0 },
		{ 1582, 2595, 0 },
		{ 1538, 2817, 0 },
		{ 1538, 2800, 0 },
		{ 1568, 2670, 0 },
		{ 1503, 2514, 0 },
		{ 0, 0, 36 },
		{ 1553, 2879, 0 },
		{ 1553, 2887, 0 },
		{ 1580, 2980, 0 },
		{ 1580, 2981, 0 },
		{ 1505, 2415, 0 },
		{ 1550, 2767, 0 },
		{ 2117, 2429, 0 },
		{ 2100, 2435, 0 },
		{ 1592, 2721, 0 },
		{ 2117, 3356, 0 },
		{ 1554, 2331, 0 },
		{ 1580, 2991, 0 },
		{ 841, 2949, 0 },
		{ 1550, 2775, 0 },
		{ 1538, 2801, 0 },
		{ 1503, 2520, 0 },
		{ 1538, 2805, 0 },
		{ 1550, 2781, 0 },
		{ 0, 2654, 0 },
		{ 0, 0, 49 },
		{ 1554, 2332, 0 },
		{ 0, 0, 52 },
		{ 1538, 2810, 0 },
		{ 2117, 3236, 0 },
		{ 1654, 3070, 0 },
		{ 1523, 2822, 0 },
		{ 809, 2390, 0 },
		{ 1554, 2333, 0 },
		{ 1554, 2299, 0 },
		{ 1582, 2616, 0 },
		{ 1538, 2818, 0 },
		{ 841, 2953, 0 },
		{ 1523, 2844, 0 },
		{ 1550, 2792, 0 },
		{ 2100, 2424, 0 },
		{ 1536, 2579, 0 },
		{ 1503, 2443, 0 },
		{ 1582, 2627, 0 },
		{ 2117, 3320, 0 },
		{ 1592, 2756, 0 },
		{ 0, 0, 15 },
		{ 0, 0, 20 },
		{ 0, 0, 41 },
		{ 0, 0, 43 },
		{ 1582, 2629, 0 },
		{ 0, 0, 53 },
		{ 1565, 2378, 0 },
		{ 1565, 2380, 0 },
		{ 0, 0, 56 },
		{ 1550, 2772, 0 },
		{ 2117, 3206, 0 },
		{ 1553, 2901, 0 },
		{ 1503, 2470, 0 },
		{ 0, 2869, 0 },
		{ 2117, 3218, 0 },
		{ 1550, 2774, 0 },
		{ 1582, 2634, 0 },
		{ 1503, 2472, 0 },
		{ 1582, 2636, 0 },
		{ 1554, 2336, 0 },
		{ 1654, 3066, 0 },
		{ 1580, 2979, 0 },
		{ 1568, 2696, 0 },
		{ 1606, 3171, 0 },
		{ 1553, 2912, 0 },
		{ 0, 0, 50 },
		{ 1536, 2542, 0 },
		{ 0, 0, 55 },
		{ 1582, 2643, 0 },
		{ 0, 0, 87 },
		{ 2117, 3290, 0 },
		{ 1536, 2543, 0 },
		{ 1654, 2881, 0 },
		{ 2117, 3296, 0 },
		{ 841, 2950, 0 },
		{ 2117, 3300, 0 },
		{ 1580, 2987, 0 },
		{ 1593, 3135, 0 },
		{ 2117, 3310, 0 },
		{ 1553, 2884, 0 },
		{ 2117, 3314, 0 },
		{ 1592, 2743, 0 },
		{ 1592, 2744, 0 },
		{ 1554, 2337, 0 },
		{ 1538, 2814, 0 },
		{ 1568, 2709, 0 },
		{ 1538, 2816, 0 },
		{ 1580, 3002, 0 },
		{ 1580, 3005, 0 },
		{ 1592, 2748, 0 },
		{ 1523, 2826, 0 },
		{ 1503, 2478, 0 },
		{ 1554, 2338, 0 },
		{ 1580, 3017, 0 },
		{ 1654, 3064, 0 },
		{ 1503, 2482, 0 },
		{ 2117, 3224, 0 },
		{ 1568, 2678, 0 },
		{ 1554, 2339, 0 },
		{ 1503, 2484, 0 },
		{ 1554, 2340, 0 },
		{ 1554, 2341, 0 },
		{ 1523, 2851, 0 },
		{ 1554, 2342, 0 },
		{ 1503, 2488, 0 },
		{ 1580, 3028, 0 },
		{ 1503, 2489, 0 },
		{ 1593, 3149, 0 },
		{ 1503, 2492, 0 },
		{ 1580, 3032, 0 },
		{ 1523, 2862, 0 },
		{ 2100, 2422, 0 },
		{ 1523, 2865, 0 },
		{ 1593, 3106, 0 },
		{ 1503, 2494, 0 },
		{ 0, 0, 16 },
		{ 1503, 2495, 0 },
		{ 809, 2383, 0 },
		{ 1503, 2497, 0 },
		{ 1503, 2498, 0 },
		{ 1505, 2399, 0 },
		{ 1582, 2614, 0 },
		{ 1593, 3160, 0 },
		{ 1580, 2970, 0 },
		{ 0, 0, 17 },
		{ 0, 0, 45 },
		{ 0, 0, 46 },
		{ 0, 0, 47 },
		{ 1538, 2806, 0 },
		{ 1554, 2344, 0 },
		{ 1580, 2974, 0 },
		{ 1503, 2501, 0 },
		{ 1554, 2298, 0 },
		{ 1554, 2346, 0 },
		{ 1553, 2902, 0 },
		{ 1538, 2811, 0 },
		{ 1523, 2853, 0 },
		{ 1503, 2505, 0 },
		{ 1568, 2683, 0 },
		{ 1503, 2506, 0 },
		{ 1503, 2507, 0 },
		{ 0, 0, 82 },
		{ 1503, 2508, 0 },
		{ 1554, 2347, 0 },
		{ 1523, 2864, 0 },
		{ 0, 0, 3 },
		{ 1580, 2989, 0 },
		{ 1580, 2990, 0 },
		{ 1593, 3154, 0 },
		{ 2117, 3255, 0 },
		{ 1550, 2776, 0 },
		{ 1553, 2915, 0 },
		{ 1503, 2510, 0 },
		{ 1553, 2917, 0 },
		{ 1554, 2348, 0 },
		{ 0, 0, 24 },
		{ 1553, 2874, 0 },
		{ 1592, 2736, 0 },
		{ 1580, 3003, 0 },
		{ 0, 0, 30 },
		{ 1523, 2825, 0 },
		{ 1503, 2512, 0 },
		{ 1580, 3008, 0 },
		{ 1503, 2513, 0 },
		{ 1582, 2639, 0 },
		{ 1568, 2703, 0 },
		{ 1608, 2675, 0 },
		{ 1553, 2892, 0 },
		{ 0, 2393, 0 },
		{ 1550, 2793, 0 },
		{ 1554, 2349, 0 },
		{ 0, 0, 79 },
		{ 1568, 2713, 0 },
		{ 1523, 2849, 0 },
		{ 1568, 2715, 0 },
		{ 1568, 2716, 0 },
		{ 1568, 2669, 0 },
		{ 1503, 2516, 0 },
		{ 1580, 3029, 0 },
		{ 1554, 2350, 0 },
		{ 1538, 2799, 0 },
		{ 1523, 2861, 0 },
		{ 1536, 2567, 0 },
		{ 1593, 3134, 0 },
		{ 1536, 2568, 0 },
		{ 1554, 2351, 0 },
		{ 1592, 2723, 0 },
		{ 1593, 3147, 0 },
		{ 2117, 3242, 0 },
		{ 1582, 2649, 0 },
		{ 1582, 2650, 0 },
		{ 2117, 3250, 0 },
		{ 1580, 2961, 0 },
		{ 1554, 2352, 0 },
		{ 1568, 2685, 0 },
		{ 1554, 2353, 0 },
		{ 1523, 2827, 0 },
		{ 1580, 2969, 0 },
		{ 1568, 2688, 0 },
		{ 1523, 2829, 0 },
		{ 1592, 2739, 0 },
		{ 2100, 2425, 0 },
		{ 1503, 2522, 0 },
		{ 1523, 2836, 0 },
		{ 0, 0, 84 },
		{ 1523, 2839, 0 },
		{ 1523, 2840, 0 },
		{ 1523, 2842, 0 },
		{ 1568, 2693, 0 },
		{ 1593, 3158, 0 },
		{ 1550, 2791, 0 },
		{ 0, 0, 9 },
		{ 2117, 3354, 0 },
		{ 1536, 2578, 0 },
		{ 1503, 2523, 0 },
		{ 2117, 2948, 0 },
		{ 1593, 3104, 0 },
		{ 1550, 2794, 0 },
		{ 2117, 3208, 0 },
		{ 1523, 2850, 0 },
		{ 0, 0, 31 },
		{ 0, 0, 29 },
		{ 1536, 2531, 0 },
		{ 1582, 2596, 0 },
		{ 0, 2922, 0 },
		{ 1523, 2854, 0 },
		{ 1503, 2524, 0 },
		{ 0, 0, 72 },
		{ 1550, 2763, 0 },
		{ 1580, 2995, 0 },
		{ 1580, 2997, 0 },
		{ 1505, 2417, 0 },
		{ 1580, 2999, 0 },
		{ 1568, 2705, 0 },
		{ 0, 0, 80 },
		{ 0, 0, 86 },
		{ 0, 0, 81 },
		{ 0, 0, 83 },
		{ 1523, 2860, 0 },
		{ 1503, 2526, 0 },
		{ 2117, 3244, 0 },
		{ 1582, 2601, 0 },
		{ 1580, 3004, 0 },
		{ 1553, 2914, 0 },
		{ 1503, 2527, 0 },
		{ 1582, 2604, 0 },
		{ 2117, 3286, 0 },
		{ 1554, 2278, 0 },
		{ 1523, 2867, 0 },
		{ 1553, 2919, 0 },
		{ 1536, 2538, 0 },
		{ 1536, 2540, 0 },
		{ 1554, 2279, 0 },
		{ 0, 0, 70 },
		{ 1568, 2671, 0 },
		{ 1592, 2731, 0 },
		{ 1582, 2612, 0 },
		{ 1554, 2280, 0 },
		{ 1580, 3024, 0 },
		{ 1554, 2283, 0 },
		{ 1523, 2830, 0 },
		{ 0, 0, 85 },
		{ 1592, 2738, 0 },
		{ 1538, 2803, 0 },
		{ 2117, 3322, 0 },
		{ 1536, 2544, 0 },
		{ 1592, 2740, 0 },
		{ 1553, 2898, 0 },
		{ 0, 0, 28 },
		{ 1523, 2837, 0 },
		{ 841, 2954, 0 },
		{ 1523, 2838, 0 },
		{ 1536, 2545, 0 },
		{ 1580, 3037, 0 },
		{ 0, 0, 32 },
		{ 1503, 2459, 0 },
		{ 1523, 2841, 0 },
		{ 1503, 2460, 0 },
		{ 2117, 3216, 0 },
		{ 1503, 2464, 0 },
		{ 1582, 2626, 0 },
		{ 1503, 2465, 0 },
		{ 0, 0, 77 },
		{ 1582, 2628, 0 },
		{ 1580, 2965, 0 },
		{ 1503, 2466, 0 },
		{ 0, 0, 14 },
		{ 1550, 2764, 0 },
		{ 1592, 2751, 0 },
		{ 1523, 2852, 0 },
		{ 2100, 2423, 0 },
		{ 0, 0, 33 },
		{ 1580, 2973, 0 },
		{ 1536, 2555, 0 },
		{ 1568, 2695, 0 },
		{ 0, 0, 71 },
		{ 0, 2386, 0 },
		{ 1503, 2469, 0 },
		{ 1568, 2699, 0 },
		{ 1554, 2284, 0 },
		{ 1568, 2701, 0 },
		{ 0, 0, 68 },
		{ 0, 0, 59 },
		{ 1553, 2885, 0 },
		{ 1536, 2564, 0 },
		{ 1550, 2777, 0 },
		{ 0, 0, 34 },
		{ 1550, 2779, 0 },
		{ 1582, 2638, 0 },
		{ 0, 0, 13 },
		{ 1523, 2866, 0 },
		{ 1568, 2704, 0 },
		{ 1592, 2729, 0 },
		{ 1523, 2821, 0 },
		{ 1503, 2471, 0 },
		{ 1523, 2823, 0 },
		{ 1592, 2733, 0 },
		{ 0, 0, 26 },
		{ 1554, 2285, 0 },
		{ 1593, 3162, 0 },
		{ 0, 0, 27 },
		{ 0, 0, 69 },
		{ 1580, 2996, 0 },
		{ 1582, 2641, 0 },
		{ 0, 0, 78 },
		{ 1568, 2711, 0 },
		{ 0, 0, 74 },
		{ 1550, 2790, 0 },
		{ 0, 2956, 0 },
		{ 1503, 2474, 0 },
		{ 1554, 2286, 0 },
		{ 1536, 2569, 0 },
		{ 1523, 2833, 0 },
		{ 1536, 2571, 0 },
		{ 2100, 2432, 0 },
		{ 1553, 2910, 0 },
		{ 1503, 2477, 0 },
		{ 1554, 2287, 0 },
		{ 0, 0, 73 },
		{ 0, 0, 10 },
		{ 1550, 2762, 0 },
		{ 1592, 2745, 0 },
		{ 1568, 2672, 0 },
		{ 1503, 2479, 0 },
		{ 1593, 3096, 0 },
		{ 1550, 2766, 0 },
		{ 1523, 2843, 0 },
		{ 1568, 2676, 0 },
		{ 1503, 2480, 0 },
		{ 0, 0, 12 },
		{ 0, 0, 75 },
		{ 1523, 2846, 0 },
		{ 1553, 2880, 0 },
		{ 0, 0, 76 },
		{ 1592, 2750, 0 },
		{ 1550, 2770, 0 },
		{ 0, 0, 11 },
		{ 2139, 1152, 145 },
		{ 0, 0, 146 },
		{ 1662, 3608, 147 },
		{ 2133, 1427, 151 },
		{ 878, 1931, 152 },
		{ 0, 0, 152 },
		{ 2133, 1533, 148 },
		{ 881, 1321, 0 },
		{ 2133, 1556, 149 },
		{ 884, 1351, 0 },
		{ 881, 0, 0 },
		{ 1766, 2015, 150 },
		{ 886, 1389, 0 },
		{ 884, 0, 0 },
		{ 1766, 2051, 148 },
		{ 886, 0, 0 },
		{ 1766, 2061, 149 },
		{ 2100, 2433, 94 },
		{ 0, 0, 94 },
		{ 2114, 1610, 0 },
		{ 2133, 2121, 0 },
		{ 2134, 1662, 0 },
		{ 893, 3465, 0 },
		{ 2129, 1979, 0 },
		{ 2133, 2141, 0 },
		{ 2123, 2192, 0 },
		{ 2032, 1825, 0 },
		{ 2135, 2269, 0 },
		{ 2134, 1643, 0 },
		{ 2135, 2237, 0 },
		{ 2136, 1524, 0 },
		{ 2137, 2033, 0 },
		{ 2138, 1705, 0 },
		{ 2139, 1774, 0 },
		{ 2114, 1604, 0 },
		{ 2140, 3240, 0 },
		{ 0, 0, 92 },
		{ 1553, 2882, 100 },
		{ 0, 0, 100 },
		{ 2139, 1752, 0 },
		{ 2137, 2038, 0 },
		{ 2136, 1478, 0 },
		{ 2137, 2015, 0 },
		{ 2135, 2220, 0 },
		{ 2032, 1850, 0 },
		{ 2137, 2036, 0 },
		{ 2114, 1616, 0 },
		{ 2135, 2236, 0 },
		{ 2136, 1481, 0 },
		{ 2137, 1904, 0 },
		{ 2138, 1685, 0 },
		{ 2139, 1786, 0 },
		{ 923, 3480, 0 },
		{ 2129, 1900, 0 },
		{ 2107, 1386, 0 },
		{ 2139, 1747, 0 },
		{ 2123, 2203, 0 },
		{ 2136, 1409, 0 },
		{ 2137, 1936, 0 },
		{ 2138, 1692, 0 },
		{ 2139, 1781, 0 },
		{ 2140, 3246, 0 },
		{ 0, 0, 98 },
		{ 2100, 2431, 105 },
		{ 0, 0, 105 },
		{ 2114, 1619, 0 },
		{ 2133, 2136, 0 },
		{ 2134, 1660, 0 },
		{ 939, 3475, 0 },
		{ 2129, 2064, 0 },
		{ 2032, 1853, 0 },
		{ 2135, 2247, 0 },
		{ 2114, 1581, 0 },
		{ 2114, 1586, 0 },
		{ 2133, 2149, 0 },
		{ 2114, 1588, 0 },
		{ 2140, 3320, 0 },
		{ 0, 0, 104 },
		{ 1608, 2939, 91 },
		{ 0, 0, 91 },
		{ 2133, 2157, 0 },
		{ 2139, 1779, 0 },
		{ 2138, 1690, 0 },
		{ 2062, 1817, 0 },
		{ 2133, 2143, 0 },
		{ 1593, 3155, 0 },
		{ 2032, 1859, 0 },
		{ 2135, 2260, 0 },
		{ 2114, 1596, 0 },
		{ 2114, 1603, 0 },
		{ 1274, 3424, 0 },
		{ 1274, 3410, 0 },
		{ 2137, 2031, 0 },
		{ 2139, 1736, 0 },
		{ 2137, 2034, 0 },
		{ 2136, 1418, 0 },
		{ 2137, 2037, 0 },
		{ 2135, 2270, 0 },
		{ 2032, 1871, 0 },
		{ 2137, 2042, 0 },
		{ 2114, 1350, 0 },
		{ 2133, 2126, 0 },
		{ 2134, 1636, 0 },
		{ 2140, 3260, 0 },
		{ 0, 0, 89 },
		{ 986, 0, 1 },
		{ 986, 0, 106 },
		{ 986, 2044, 144 },
		{ 1154, 138, 144 },
		{ 1154, 304, 144 },
		{ 1154, 292, 144 },
		{ 1154, 300, 144 },
		{ 1154, 293, 144 },
		{ 1154, 335, 144 },
		{ 1154, 372, 144 },
		{ 1154, 367, 144 },
		{ 1994, 2212, 144 },
		{ 998, 1910, 144 },
		{ 986, 2085, 144 },
		{ 1154, 396, 144 },
		{ 1154, 394, 144 },
		{ 1154, 397, 144 },
		{ 1154, 0, 144 },
		{ 2138, 1684, 0 },
		{ 0, 0, 107 },
		{ 2139, 1751, 0 },
		{ 1154, 384, 0 },
		{ 1154, 0, 0 },
		{ 1662, 2936, 0 },
		{ 1154, 404, 0 },
		{ 1154, 420, 0 },
		{ 1154, 417, 0 },
		{ 1154, 424, 0 },
		{ 1154, 417, 0 },
		{ 1154, 424, 0 },
		{ 1154, 407, 0 },
		{ 1154, 399, 0 },
		{ 1154, 402, 0 },
		{ 2133, 2094, 0 },
		{ 2133, 2103, 0 },
		{ 1155, 408, 0 },
		{ 1155, 409, 0 },
		{ 1154, 419, 0 },
		{ 2138, 1711, 0 },
		{ 2055, 2180, 0 },
		{ 1154, 407, 0 },
		{ 1154, 477, 0 },
		{ 1154, 455, 0 },
		{ 1154, 456, 0 },
		{ 1154, 480, 0 },
		{ 1154, 521, 0 },
		{ 1154, 515, 0 },
		{ 1154, 477, 0 },
		{ 1154, 492, 0 },
		{ 1154, 12, 0 },
		{ 2139, 1741, 0 },
		{ 1923, 2074, 0 },
		{ 1154, 36, 0 },
		{ 1154, 29, 0 },
		{ 1155, 31, 0 },
		{ 2032, 1846, 0 },
		{ 0, 0, 143 },
		{ 1154, 41, 0 },
		{ 1154, 23, 0 },
		{ 1154, 12, 0 },
		{ 1154, 17, 0 },
		{ 1154, 66, 0 },
		{ 1154, 62, 0 },
		{ 1154, 50, 0 },
		{ 1154, 45, 0 },
		{ 1154, 0, 133 },
		{ 1154, 83, 0 },
		{ 2138, 1708, 0 },
		{ 2137, 2050, 0 },
		{ 1154, 40, 0 },
		{ 1154, 44, 0 },
		{ 1154, 39, 0 },
		{ -1062, 1002, 0 },
		{ 1155, 47, 0 },
		{ 1154, 81, 0 },
		{ 1154, 103, 0 },
		{ 1154, 97, 0 },
		{ 1154, 133, 0 },
		{ 1154, 114, 0 },
		{ 1154, 131, 0 },
		{ 1154, 0, 132 },
		{ 1154, 135, 0 },
		{ 2062, 1820, 0 },
		{ 2139, 1787, 0 },
		{ 1154, 138, 0 },
		{ 1154, 147, 0 },
		{ 1154, 148, 0 },
		{ 0, 0, 134 },
		{ 1154, 138, 0 },
		{ 1156, 116, -7 },
		{ 1154, 166, 0 },
		{ 1154, 178, 0 },
		{ 1154, 176, 0 },
		{ 1154, 178, 0 },
		{ 1154, 191, 0 },
		{ 1154, 157, 0 },
		{ 2133, 2161, 0 },
		{ 2133, 2087, 0 },
		{ 1154, 0, 136 },
		{ 1154, 197, 137 },
		{ 1154, 174, 0 },
		{ 1154, 216, 0 },
		{ 1078, 2604, 0 },
		{ 2129, 3147, 0 },
		{ 1638, 3398, 125 },
		{ 1154, 219, 0 },
		{ 1154, 223, 0 },
		{ 1154, 221, 0 },
		{ 1154, 254, 0 },
		{ 1154, 243, 0 },
		{ 1154, 275, 0 },
		{ 1155, 250, 0 },
		{ 1593, 3132, 0 },
		{ 1662, 4, 139 },
		{ 1154, 259, 0 },
		{ 1154, 271, 0 },
		{ 1154, 268, 0 },
		{ 0, 0, 111 },
		{ 1156, 231, -10 },
		{ 1156, 259, -13 },
		{ 1156, 261, -16 },
		{ 1156, 345, -19 },
		{ 1154, 295, 0 },
		{ 1154, 309, 0 },
		{ 1154, 0, 135 },
		{ 2032, 1875, 0 },
		{ 1154, 283, 0 },
		{ 1154, 279, 0 },
		{ 1155, 292, 0 },
		{ 1105, 2631, 0 },
		{ 2129, 3183, 0 },
		{ 1638, 3380, 126 },
		{ 1108, 2632, 0 },
		{ 2129, 3179, 0 },
		{ 1638, 3394, 127 },
		{ 1111, 2633, 0 },
		{ 2129, 3185, 0 },
		{ 1638, 3386, 130 },
		{ 1114, 2634, 0 },
		{ 2129, 3149, 0 },
		{ 1638, 3406, 131 },
		{ 1154, 337, 0 },
		{ 1156, 373, -22 },
		{ 2135, 2226, 0 },
		{ 1154, 321, 0 },
		{ 1154, 366, 0 },
		{ 1154, 338, 0 },
		{ 0, 0, 113 },
		{ 0, 0, 115 },
		{ 0, 0, 121 },
		{ 0, 0, 123 },
		{ 1156, 2, -1 },
		{ 1128, 2644, 0 },
		{ 2129, 3145, 0 },
		{ 1638, 3403, 129 },
		{ 2114, 1593, 0 },
		{ 1154, 360, 0 },
		{ 1154, 375, 0 },
		{ 1154, 363, 0 },
		{ 1135, 2648, 0 },
		{ 2129, 3187, 0 },
		{ 1638, 3405, 128 },
		{ 0, 0, 119 },
		{ 2114, 1609, 0 },
		{ 1154, 4, 142 },
		{ 1155, 369, 0 },
		{ 1154, 383, 0 },
		{ 0, 0, 117 },
		{ 1274, 3414, 0 },
		{ 1274, 3415, 0 },
		{ 1154, 371, 0 },
		{ 1154, 367, 0 },
		{ 1274, 3411, 0 },
		{ 0, 0, 141 },
		{ 1154, 375, 0 },
		{ 1154, 380, 0 },
		{ 0, 0, 140 },
		{ 1154, 385, 0 },
		{ 1154, 376, 0 },
		{ 1155, 378, 138 },
		{ 1156, 721, 0 },
		{ 1157, 532, -4 },
		{ 1158, 2613, 0 },
		{ 2129, 3111, 0 },
		{ 1638, 3366, 124 },
		{ 0, 0, 109 },
		{ 1608, 2940, 181 },
		{ 0, 0, 181 },
		{ 2133, 2091, 0 },
		{ 2139, 1743, 0 },
		{ 2138, 1714, 0 },
		{ 2062, 1802, 0 },
		{ 2133, 2123, 0 },
		{ 1593, 3108, 0 },
		{ 2032, 1834, 0 },
		{ 2135, 2259, 0 },
		{ 2114, 1612, 0 },
		{ 2114, 1613, 0 },
		{ 1274, 3418, 0 },
		{ 1274, 3419, 0 },
		{ 2137, 1932, 0 },
		{ 2139, 1756, 0 },
		{ 2137, 1972, 0 },
		{ 2136, 1451, 0 },
		{ 2137, 1988, 0 },
		{ 2135, 2239, 0 },
		{ 2032, 1866, 0 },
		{ 2137, 2025, 0 },
		{ 2114, 1370, 0 },
		{ 2133, 2102, 0 },
		{ 2134, 1646, 0 },
		{ 2140, 3359, 0 },
		{ 0, 0, 180 },
		{ 1608, 2941, 183 },
		{ 0, 0, 183 },
		{ 2133, 2112, 0 },
		{ 2139, 1784, 0 },
		{ 2138, 1707, 0 },
		{ 2062, 1816, 0 },
		{ 2133, 2127, 0 },
		{ 1593, 3133, 0 },
		{ 2032, 1847, 0 },
		{ 2135, 2248, 0 },
		{ 2114, 1628, 0 },
		{ 2114, 1630, 0 },
		{ 1274, 3412, 0 },
		{ 1274, 3413, 0 },
		{ 2123, 2199, 0 },
		{ 2134, 1671, 0 },
		{ 2138, 1717, 0 },
		{ 2114, 1576, 0 },
		{ 2114, 1580, 0 },
		{ 2138, 1688, 0 },
		{ 1251, 1369, 0 },
		{ 2133, 2092, 0 },
		{ 2134, 1650, 0 },
		{ 2140, 3244, 0 },
		{ 0, 0, 182 },
		{ 1608, 2930, 185 },
		{ 0, 0, 185 },
		{ 2133, 2101, 0 },
		{ 2139, 1763, 0 },
		{ 2138, 1699, 0 },
		{ 2062, 1818, 0 },
		{ 2133, 2114, 0 },
		{ 1593, 3131, 0 },
		{ 2032, 1856, 0 },
		{ 2135, 2235, 0 },
		{ 2114, 1582, 0 },
		{ 2114, 1583, 0 },
		{ 1274, 3422, 0 },
		{ 1274, 3423, 0 },
		{ 2062, 1821, 0 },
		{ 2107, 1382, 0 },
		{ 2136, 1542, 0 },
		{ 2135, 2250, 0 },
		{ 2136, 1570, 0 },
		{ 2138, 1715, 0 },
		{ 2134, 1645, 0 },
		{ 2140, 3248, 0 },
		{ 0, 0, 184 },
		{ 1608, 2929, 187 },
		{ 0, 0, 187 },
		{ 2133, 2154, 0 },
		{ 2139, 1739, 0 },
		{ 2138, 1721, 0 },
		{ 2062, 1822, 0 },
		{ 2133, 2088, 0 },
		{ 1593, 3100, 0 },
		{ 2032, 1857, 0 },
		{ 2135, 2238, 0 },
		{ 2114, 1594, 0 },
		{ 2114, 1595, 0 },
		{ 1274, 3420, 0 },
		{ 1274, 3421, 0 },
		{ 2133, 2099, 0 },
		{ 0, 1370, 0 },
		{ 2135, 2255, 0 },
		{ 2032, 1872, 0 },
		{ 2107, 1387, 0 },
		{ 2135, 2262, 0 },
		{ 2136, 1444, 0 },
		{ 2138, 1698, 0 },
		{ 2134, 1635, 0 },
		{ 2140, 3256, 0 },
		{ 0, 0, 186 },
		{ 1608, 2928, 179 },
		{ 0, 0, 179 },
		{ 2133, 2122, 0 },
		{ 2139, 1778, 0 },
		{ 2138, 1701, 0 },
		{ 2062, 1812, 0 },
		{ 2133, 2128, 0 },
		{ 1593, 3161, 0 },
		{ 2032, 1854, 0 },
		{ 2135, 2240, 0 },
		{ 2114, 1606, 0 },
		{ 2114, 1607, 0 },
		{ 1274, 3416, 0 },
		{ 0, 3417, 0 },
		{ 2055, 2173, 0 },
		{ 2137, 1934, 0 },
		{ 2114, 1608, 0 },
		{ 1975, 1679, 0 },
		{ 2032, 1870, 0 },
		{ 2135, 2265, 0 },
		{ 2072, 1346, 0 },
		{ 2140, 3402, 0 },
		{ 0, 0, 177 },
		{ 1300, 0, 1 },
		{ 1459, 2133, 271 },
		{ 2133, 2089, 271 },
		{ 1300, 0, 236 },
		{ 1300, 2093, 271 },
		{ 2135, 2227, 271 },
		{ 1300, 0, 239 },
		{ 2124, 1385, 271 },
		{ 1606, 3179, 271 },
		{ 1592, 2752, 271 },
		{ 2100, 2434, 271 },
		{ 1592, 2755, 271 },
		{ 1580, 3036, 271 },
		{ 2135, 2245, 271 },
		{ 2138, 1579, 271 },
		{ 1300, 0, 271 },
		{ 1994, 2214, 269 },
		{ 2135, 2070, 271 },
		{ 2055, 2179, 271 },
		{ 0, 0, 271 },
		{ 2138, 1724, 0 },
		{ -1305, 17, 230 },
		{ -1306, 3498, 0 },
		{ 2139, 1744, 0 },
		{ 0, 0, 237 },
		{ 2139, 1745, 0 },
		{ 2138, 1727, 0 },
		{ 2134, 1666, 0 },
		{ 2134, 1667, 0 },
		{ 2137, 2044, 0 },
		{ 1554, 2343, 0 },
		{ 1568, 2673, 0 },
		{ 1503, 2502, 0 },
		{ 1505, 2418, 0 },
		{ 1568, 2677, 0 },
		{ 2032, 1855, 0 },
		{ 2107, 1405, 0 },
		{ 2114, 1618, 0 },
		{ 2139, 1765, 0 },
		{ 2136, 1533, 0 },
		{ 2133, 2137, 0 },
		{ 2138, 1693, 0 },
		{ 2097, 3464, 0 },
		{ 2138, 1694, 0 },
		{ 2055, 2178, 0 },
		{ 1975, 1680, 0 },
		{ 2133, 2151, 0 },
		{ 2134, 1642, 0 },
		{ 2032, 1876, 0 },
		{ 1608, 2938, 0 },
		{ 1554, 2281, 0 },
		{ 1554, 2282, 0 },
		{ 1580, 2994, 0 },
		{ 1523, 2859, 0 },
		{ 2114, 1623, 0 },
		{ 2133, 2159, 0 },
		{ 2114, 1625, 0 },
		{ 2055, 2168, 0 },
		{ 2107, 1374, 0 },
		{ 2139, 1793, 0 },
		{ 2032, 1848, 0 },
		{ 0, 0, 232 },
		{ 2062, 1810, 0 },
		{ 0, 0, 270 },
		{ 2107, 1378, 0 },
		{ 2139, 1737, 0 },
		{ 2133, 2097, 0 },
		{ 2055, 2172, 0 },
		{ 1580, 3009, 0 },
		{ 1536, 2551, 0 },
		{ 1568, 2714, 0 },
		{ 1582, 2589, 0 },
		{ 1554, 2301, 0 },
		{ 2133, 2098, 0 },
		{ 0, 0, 259 },
		{ 1593, 3098, 0 },
		{ 2138, 1709, 0 },
		{ 2134, 1657, 0 },
		{ 2136, 1571, 0 },
		{ -1379, 1227, 0 },
		{ 2133, 2109, 0 },
		{ 2032, 1864, 0 },
		{ 2136, 1410, 0 },
		{ 2139, 1746, 0 },
		{ 1382, 3439, 0 },
		{ 1550, 2778, 0 },
		{ 1553, 2886, 0 },
		{ 1582, 2605, 0 },
		{ 1554, 2313, 0 },
		{ 1568, 2681, 0 },
		{ 0, 0, 258 },
		{ 2138, 1719, 0 },
		{ 2062, 1807, 0 },
		{ 2139, 1748, 0 },
		{ 2137, 1987, 0 },
		{ 0, 0, 260 },
		{ 1593, 3156, 235 },
		{ 2135, 2232, 0 },
		{ 0, 3441, 0 },
		{ 2136, 1411, 0 },
		{ 0, 0, 263 },
		{ 0, 0, 264 },
		{ 1400, 0, -44 },
		{ 1565, 2381, 0 },
		{ 1592, 2758, 0 },
		{ 1568, 2691, 0 },
		{ 1580, 2960, 0 },
		{ 2137, 1989, 0 },
		{ 0, 0, 262 },
		{ 0, 0, 268 },
		{ 2123, 2195, 0 },
		{ 2032, 1878, 0 },
		{ 2136, 1413, 0 },
		{ 0, 0, 265 },
		{ 0, 0, 266 },
		{ 1608, 2934, 0 },
		{ 2129, 3119, 0 },
		{ 1638, 3370, 253 },
		{ 1580, 2966, 0 },
		{ 1606, 3169, 0 },
		{ 1582, 2624, 0 },
		{ 1582, 2625, 0 },
		{ 2139, 1754, 0 },
		{ 2135, 2241, 0 },
		{ 2135, 2243, 0 },
		{ 2137, 2030, 0 },
		{ 2133, 2150, 0 },
		{ 0, 0, 243 },
		{ 1421, 0, -38 },
		{ 1423, 0, -41 },
		{ 1592, 2735, 0 },
		{ 1593, 3153, 0 },
		{ 0, 0, 261 },
		{ 2032, 1840, 0 },
		{ 2114, 1584, 0 },
		{ 2138, 1683, 0 },
		{ 2139, 1764, 0 },
		{ 2129, 3151, 0 },
		{ 1638, 3374, 254 },
		{ 2129, 3175, 0 },
		{ 1638, 3382, 255 },
		{ 1606, 3168, 0 },
		{ 1434, 0, -47 },
		{ 1608, 2932, 0 },
		{ 2114, 1585, 0 },
		{ 2139, 1768, 0 },
		{ 2133, 2162, 0 },
		{ 0, 0, 245 },
		{ 0, 0, 247 },
		{ 1440, 0, -32 },
		{ 2129, 3109, 0 },
		{ 1638, 3372, 257 },
		{ 2133, 2165, 0 },
		{ 0, 0, 234 },
		{ 2129, 2299, 0 },
		{ 2134, 1637, 0 },
		{ 2129, 3115, 0 },
		{ 1638, 3390, 256 },
		{ 0, 0, 251 },
		{ 2139, 1777, 0 },
		{ 0, 0, 238 },
		{ 2135, 2230, 0 },
		{ 0, 0, 249 },
		{ 2138, 1686, 0 },
		{ 2136, 1415, 0 },
		{ 2062, 1813, 0 },
		{ 2137, 1931, 0 },
		{ 2133, 2096, 0 },
		{ 2138, 1689, 0 },
		{ 0, 0, 267 },
		{ 2139, 1783, 0 },
		{ 1456, 0, -53 },
		{ 2129, 3181, 0 },
		{ 1638, 3384, 252 },
		{ 0, 0, 241 },
		{ 1300, 2130, 271 },
		{ 1463, 1906, 271 },
		{ -1461, 29, 230 },
		{ -1462, 3492, 0 },
		{ 2097, 3484, 0 },
		{ 2097, 3454, 0 },
		{ 0, 0, 231 },
		{ 2097, 3466, 0 },
		{ -1467, 22, 0 },
		{ -1468, 1, 0 },
		{ 1471, 0, 232 },
		{ 2097, 3467, 0 },
		{ 2129, 3622, 0 },
		{ 0, 0, 233 },
		{ 1489, 0, 1 },
		{ 1640, 2136, 229 },
		{ 2133, 2110, 229 },
		{ 1489, 0, 193 },
		{ 1489, 2056, 229 },
		{ 1568, 2689, 229 },
		{ 1489, 0, 196 },
		{ 2107, 1391, 229 },
		{ 1606, 3167, 229 },
		{ 1592, 2730, 229 },
		{ 2100, 2414, 229 },
		{ 1592, 2732, 229 },
		{ 1580, 3016, 229 },
		{ 2135, 2266, 229 },
		{ 2138, 1578, 229 },
		{ 1489, 0, 229 },
		{ 1994, 2215, 228 },
		{ 2135, 2221, 229 },
		{ 2055, 2181, 229 },
		{ 0, 0, 229 },
		{ 2138, 1695, 0 },
		{ -1494, 3574, 188 },
		{ -1495, 3497, 0 },
		{ 2139, 1796, 0 },
		{ 0, 0, 194 },
		{ 2139, 1797, 0 },
		{ 2138, 1696, 0 },
		{ 2134, 1652, 0 },
		{ 1554, 2305, 0 },
		{ 1568, 2707, 0 },
		{ 0, 2463, 0 },
		{ 1550, 2784, 0 },
		{ 0, 2410, 0 },
		{ 1568, 2710, 0 },
		{ 2032, 1845, 0 },
		{ 2107, 1402, 0 },
		{ 2114, 1597, 0 },
		{ 2139, 1742, 0 },
		{ 2133, 2146, 0 },
		{ 2138, 1703, 0 },
		{ 2097, 3471, 0 },
		{ 2138, 1704, 0 },
		{ 2055, 2174, 0 },
		{ 1975, 1676, 0 },
		{ 2134, 1661, 0 },
		{ 1608, 2935, 0 },
		{ 1554, 2316, 0 },
		{ 1554, 2317, 0 },
		{ 1553, 2899, 0 },
		{ 1580, 2967, 0 },
		{ 0, 2831, 0 },
		{ 2114, 1599, 0 },
		{ 2133, 2155, 0 },
		{ 2114, 1602, 0 },
		{ 2055, 2169, 0 },
		{ 2139, 1750, 0 },
		{ 2032, 1860, 0 },
		{ 0, 0, 190 },
		{ 2062, 1809, 0 },
		{ 0, 0, 227 },
		{ 2107, 1404, 0 },
		{ 2133, 2085, 0 },
		{ 1580, 2978, 0 },
		{ 0, 2554, 0 },
		{ 1568, 2686, 0 },
		{ 0, 2802, 0 },
		{ 1582, 2648, 0 },
		{ 1554, 2327, 0 },
		{ 2133, 2086, 0 },
		{ 0, 0, 222 },
		{ 1593, 3163, 0 },
		{ 2138, 1712, 0 },
		{ 2136, 1482, 0 },
		{ -1560, 1302, 0 },
		{ 2133, 2090, 0 },
		{ 2032, 1873, 0 },
		{ 2139, 1760, 0 },
		{ 0, 2783, 0 },
		{ 1553, 2881, 0 },
		{ 1582, 2591, 0 },
		{ 0, 2883, 0 },
		{ 0, 2335, 0 },
		{ 1568, 2698, 0 },
		{ 0, 0, 221 },
		{ 2138, 1716, 0 },
		{ 2062, 1819, 0 },
		{ 2137, 1992, 0 },
		{ 0, 0, 223 },
		{ 0, 0, 192 },
		{ 2135, 2242, 0 },
		{ 2136, 1528, 0 },
		{ 1575, 0, -68 },
		{ 0, 2376, 0 },
		{ 1592, 2753, 0 },
		{ 1565, 2379, 0 },
		{ 0, 2706, 0 },
		{ 1580, 3006, 0 },
		{ 2137, 2017, 0 },
		{ 0, 0, 225 },
		{ 2123, 2189, 0 },
		{ 2136, 1530, 0 },
		{ 1608, 2933, 0 },
		{ 2129, 3121, 0 },
		{ 1638, 3376, 214 },
		{ 1580, 3011, 0 },
		{ 1606, 3173, 0 },
		{ 1580, 3012, 0 },
		{ 0, 3013, 0 },
		{ 1582, 2607, 0 },
		{ 0, 2608, 0 },
		{ 2139, 1767, 0 },
		{ 2135, 2253, 0 },
		{ 2137, 2026, 0 },
		{ 2133, 2105, 0 },
		{ 0, 0, 200 },
		{ 1598, 0, -56 },
		{ 1600, 0, -59 },
		{ 1602, 0, -62 },
		{ 1604, 0, -65 },
		{ 0, 2725, 0 },
		{ 0, 3136, 0 },
		{ 0, 0, 224 },
		{ 2032, 1844, 0 },
		{ 2138, 1720, 0 },
		{ 2139, 1771, 0 },
		{ 2129, 3101, 0 },
		{ 1638, 3392, 215 },
		{ 2129, 3103, 0 },
		{ 1638, 3396, 216 },
		{ 2129, 3105, 0 },
		{ 1638, 3402, 219 },
		{ 2129, 3107, 0 },
		{ 1638, 3404, 220 },
		{ 0, 3165, 0 },
		{ 1616, 0, -29 },
		{ 0, 2927, 0 },
		{ 2139, 1773, 0 },
		{ 2133, 2119, 0 },
		{ 0, 0, 202 },
		{ 0, 0, 204 },
		{ 0, 0, 210 },
		{ 0, 0, 212 },
		{ 1621, 0, -35 },
		{ 2129, 3113, 0 },
		{ 1638, 3379, 218 },
		{ 2133, 2120, 0 },
		{ 2129, 2294, 0 },
		{ 2134, 1639, 0 },
		{ 2129, 3117, 0 },
		{ 1638, 3388, 217 },
		{ 0, 0, 208 },
		{ 2139, 1775, 0 },
		{ 0, 0, 195 },
		{ 2135, 2228, 0 },
		{ 0, 0, 206 },
		{ 2138, 1722, 0 },
		{ 2136, 1532, 0 },
		{ 2062, 1814, 0 },
		{ 2137, 2046, 0 },
		{ 2133, 2130, 0 },
		{ 2138, 1726, 0 },
		{ 0, 0, 226 },
		{ 2139, 1780, 0 },
		{ 1637, 0, -50 },
		{ 2129, 3177, 0 },
		{ 0, 3368, 213 },
		{ 0, 0, 198 },
		{ 1489, 2137, 229 },
		{ 1644, 1908, 229 },
		{ -1642, 20, 188 },
		{ -1643, 3487, 0 },
		{ 2097, 3483, 0 },
		{ 2097, 3463, 0 },
		{ 0, 0, 189 },
		{ 2097, 3478, 0 },
		{ -1648, 3600, 0 },
		{ -1649, 3495, 0 },
		{ 1652, 0, 190 },
		{ 2097, 3479, 0 },
		{ 2129, 3560, 0 },
		{ 0, 0, 191 },
		{ 0, 3076, 273 },
		{ 0, 0, 273 },
		{ 2133, 2147, 0 },
		{ 1923, 2072, 0 },
		{ 2135, 2246, 0 },
		{ 2107, 1376, 0 },
		{ 2032, 1862, 0 },
		{ 2136, 1544, 0 },
		{ 0, 5, 0 },
		{ 2134, 1649, 0 },
		{ 2107, 1377, 0 },
		{ 2139, 1789, 0 },
		{ 1667, 3464, 0 },
		{ 2129, 1653, 0 },
		{ 2135, 2261, 0 },
		{ 2134, 1654, 0 },
		{ 2135, 2264, 0 },
		{ 2062, 1808, 0 },
		{ 2133, 2163, 0 },
		{ 2136, 1572, 0 },
		{ 2133, 2166, 0 },
		{ 2134, 1659, 0 },
		{ 2114, 1622, 0 },
		{ 2140, 3429, 0 },
		{ 0, 0, 272 },
		{ 2097, 3455, 313 },
		{ 0, 0, 278 },
		{ 0, 0, 280 },
		{ 1749, 621, 311 },
		{ 1747, 637, 311 },
		{ 1805, 632, 311 },
		{ 1866, 631, 311 },
		{ 1847, 642, 311 },
		{ 1824, 644, 311 },
		{ 1824, 645, 311 },
		{ 1847, 643, 311 },
		{ 1870, 637, 311 },
		{ 1824, 651, 311 },
		{ 1794, 645, 311 },
		{ 2133, 1405, 310 },
		{ 1718, 1934, 313 },
		{ 1893, 634, 311 },
		{ 2133, 2108, 313 },
		{ -1697, 3602, 274 },
		{ -1698, 3486, 0 },
		{ 1794, 647, 311 },
		{ 1824, 656, 311 },
		{ 1695, 656, 311 },
		{ 1866, 662, 311 },
		{ 1847, 656, 311 },
		{ 1847, 658, 311 },
		{ 1747, 652, 311 },
		{ 1870, 651, 311 },
		{ 1887, 656, 311 },
		{ 1887, 658, 311 },
		{ 1777, 661, 311 },
		{ 1870, 656, 311 },
		{ 1826, 655, 311 },
		{ 1840, 669, 311 },
		{ 1866, 663, 311 },
		{ 1819, 675, 311 },
		{ 1794, 667, 311 },
		{ 2133, 1566, 307 },
		{ 1741, 1317, 0 },
		{ 2133, 1490, 308 },
		{ 2139, 1788, 0 },
		{ 2097, 3472, 0 },
		{ 1695, 679, 311 },
		{ 1870, 664, 311 },
		{ 1828, 664, 311 },
		{ 1777, 672, 311 },
		{ 1777, 673, 311 },
		{ 1826, 677, 311 },
		{ 1840, 691, 311 },
		{ 1749, 678, 311 },
		{ 1826, 680, 311 },
		{ 1851, 700, 311 },
		{ 1847, 693, 311 },
		{ 1824, 698, 311 },
		{ 1805, 708, 311 },
		{ 1851, 730, 311 },
		{ 1870, 714, 311 },
		{ 1828, 714, 311 },
		{ 1749, 742, 311 },
		{ 1847, 755, 311 },
		{ 1764, 1393, 0 },
		{ 1741, 0, 0 },
		{ 1766, 1964, 309 },
		{ 1766, 2, 0 },
		{ 2055, 2177, 0 },
		{ 0, 0, 276 },
		{ 1824, 760, 311 },
		{ 1851, 765, 311 },
		{ 1695, 751, 311 },
		{ 1749, 747, 311 },
		{ 1695, 748, 311 },
		{ 1847, 765, 311 },
		{ 1870, 753, 311 },
		{ 1777, 760, 311 },
		{ 1847, 764, 311 },
		{ 1777, 763, 311 },
		{ 1826, 756, 311 },
		{ 1761, 774, 311 },
		{ 1847, 769, 311 },
		{ 1826, 759, 311 },
		{ 1851, 779, 311 },
		{ 1828, 762, 311 },
		{ 1695, 779, 311 },
		{ 1840, 776, 311 },
		{ 1764, 0, 0 },
		{ 1766, 1974, 307 },
		{ 1766, 0, 0 },
		{ 1694, 1995, 308 },
		{ 0, 0, 312 },
		{ 1870, 766, 311 },
		{ 1826, 766, 311 },
		{ 1870, 769, 311 },
		{ 1826, 768, 311 },
		{ 1851, 788, 311 },
		{ 1915, 6, 311 },
		{ 1828, 772, 311 },
		{ 1851, 791, 311 },
		{ 1851, 792, 311 },
		{ 1695, 782, 311 },
		{ 1915, 242, 311 },
		{ 1866, 781, 311 },
		{ 1826, 777, 311 },
		{ 1805, 776, 311 },
		{ 1847, 790, 311 },
		{ 1887, 797, 311 },
		{ 1866, 785, 311 },
		{ 1824, 807, 311 },
		{ 1870, 795, 311 },
		{ 1915, 464, 311 },
		{ 1866, 800, 311 },
		{ 1847, 807, 311 },
		{ 1847, 836, 311 },
		{ 1826, 826, 311 },
		{ 2140, 3316, 0 },
		{ 2129, 3598, 286 },
		{ 1695, 834, 311 },
		{ 1826, 829, 311 },
		{ 1826, 856, 311 },
		{ 1847, 868, 311 },
		{ 2134, 1633, 0 },
		{ 1870, 860, 311 },
		{ 1824, 884, 311 },
		{ 1847, 881, 311 },
		{ 1866, 876, 311 },
		{ 1826, 872, 311 },
		{ 1847, 884, 311 },
		{ 1695, 872, 311 },
		{ 1915, 8, 311 },
		{ 1851, 921, 311 },
		{ 2138, 1672, 0 },
		{ 1870, 905, 311 },
		{ 1887, 910, 311 },
		{ 1866, 910, 311 },
		{ 1866, 912, 311 },
		{ 1915, 120, 311 },
		{ 1847, 956, 311 },
		{ 1847, 957, 311 },
		{ 1826, 947, 311 },
		{ 2133, 2129, 0 },
		{ 1828, 949, 311 },
		{ 1695, 965, 311 },
		{ 1887, 956, 311 },
		{ 1887, 983, 311 },
		{ 1828, 979, 311 },
		{ 1866, 984, 311 },
		{ 1695, 995, 311 },
		{ 2138, 1488, 0 },
		{ 1695, 982, 311 },
		{ 1923, 2066, 0 },
		{ 1695, 1020, 311 },
		{ 1840, 1033, 311 },
		{ 1887, 1027, 311 },
		{ 1870, 1024, 311 },
		{ 2140, 3242, 0 },
		{ 0, 0, 298 },
		{ 1866, 1028, 311 },
		{ 1866, 1029, 311 },
		{ 1893, 1020, 311 },
		{ 2055, 2176, 0 },
		{ 1915, 10, 311 },
		{ 1915, 122, 311 },
		{ 1695, 1067, 311 },
		{ 1915, 124, 311 },
		{ 1915, 129, 311 },
		{ 1887, 1100, 311 },
		{ 1870, 1097, 311 },
		{ 2135, 2223, 0 },
		{ 1923, 2067, 0 },
		{ 1695, 1107, 311 },
		{ 2129, 3596, 0 },
		{ 1915, 233, 311 },
		{ 1915, 235, 311 },
		{ 1695, 1117, 311 },
		{ 1915, 240, 311 },
		{ 1887, 1106, 311 },
		{ 1887, 1133, 311 },
		{ 1915, 349, 311 },
		{ 2107, 1407, 0 },
		{ 2129, 1627, 0 },
		{ 2130, 1833, 0 },
		{ 1915, 352, 311 },
		{ 2129, 1646, 0 },
		{ 2129, 1526, 0 },
		{ 1915, 354, 311 },
		{ 1915, 356, 311 },
		{ 2134, 1655, 0 },
		{ 1923, 2071, 0 },
		{ 1695, 1174, 311 },
		{ 1848, 1365, 0 },
		{ 2135, 2149, 0 },
		{ 2129, 1621, 0 },
		{ 1695, 1172, 311 },
		{ 2130, 1831, 0 },
		{ 1915, 369, 311 },
		{ 1915, 458, 311 },
		{ 2140, 3252, 0 },
		{ 2139, 735, 287 },
		{ 2123, 2188, 0 },
		{ 2133, 2133, 0 },
		{ 2133, 2135, 0 },
		{ 2138, 1618, 0 },
		{ 2134, 1632, 0 },
		{ 2133, 2139, 0 },
		{ 2133, 2140, 0 },
		{ 2129, 1903, 0 },
		{ 2129, 1491, 0 },
		{ 2133, 2144, 0 },
		{ 2114, 1624, 0 },
		{ 1695, 1180, 311 },
		{ 2133, 2148, 0 },
		{ 2114, 1626, 0 },
		{ 2062, 1811, 0 },
		{ 2134, 1640, 0 },
		{ 2137, 2039, 0 },
		{ 1915, 1416, 311 },
		{ 2138, 1697, 0 },
		{ 2129, 1670, 0 },
		{ 2129, 1629, 0 },
		{ 2133, 2160, 0 },
		{ 2032, 1879, 0 },
		{ 2139, 1758, 0 },
		{ 2135, 2258, 0 },
		{ 1923, 2068, 0 },
		{ 2138, 1700, 0 },
		{ 2032, 1836, 0 },
		{ 2114, 1579, 0 },
		{ 2134, 1647, 0 },
		{ 2134, 1648, 0 },
		{ 2139, 1766, 0 },
		{ 1975, 1677, 0 },
		{ 1915, 462, 311 },
		{ 2139, 1769, 0 },
		{ 2032, 1851, 0 },
		{ 2137, 1990, 0 },
		{ 2133, 2095, 0 },
		{ 1940, 1276, 0 },
		{ 1937, 607, 311 },
		{ 2135, 2231, 0 },
		{ 2138, 1710, 0 },
		{ 2137, 2019, 0 },
		{ 2140, 3390, 0 },
		{ 2133, 2100, 0 },
		{ 2137, 2021, 0 },
		{ 2134, 1653, 0 },
		{ 0, 2080, 0 },
		{ 1994, 2213, 0 },
		{ 2133, 2104, 0 },
		{ 2136, 1416, 0 },
		{ 2133, 2106, 0 },
		{ 2138, 1713, 0 },
		{ 2136, 1417, 0 },
		{ 2134, 1656, 0 },
		{ 2134, 1454, 0 },
		{ 2133, 2113, 0 },
		{ 2135, 2251, 0 },
		{ 2045, 2061, 0 },
		{ 1958, 3481, 0 },
		{ 2133, 2117, 0 },
		{ 2133, 1941, 0 },
		{ 2032, 1867, 0 },
		{ 2032, 1868, 0 },
		{ 0, 1243, 0 },
		{ 2129, 3543, 297 },
		{ 2123, 2194, 0 },
		{ 2136, 1445, 0 },
		{ 2133, 2125, 0 },
		{ 2114, 1590, 0 },
		{ 2135, 2268, 0 },
		{ 2123, 2200, 0 },
		{ 2137, 2043, 0 },
		{ 0, 1674, 0 },
		{ 1994, 2216, 0 },
		{ 2114, 1592, 0 },
		{ 2137, 2049, 0 },
		{ 2133, 2131, 0 },
		{ 2138, 1723, 0 },
		{ 2134, 1663, 0 },
		{ 2139, 1792, 0 },
		{ 2134, 1664, 0 },
		{ 2129, 2066, 0 },
		{ 2140, 3258, 0 },
		{ 2139, 1794, 0 },
		{ 2140, 3287, 0 },
		{ 2137, 1935, 0 },
		{ 2133, 2142, 0 },
		{ 2136, 1447, 0 },
		{ 2138, 1728, 0 },
		{ 2139, 1798, 0 },
		{ 1975, 1678, 0 },
		{ 1994, 2217, 0 },
		{ 2136, 1449, 0 },
		{ 2139, 1738, 0 },
		{ 2107, 1373, 0 },
		{ 2135, 2249, 0 },
		{ 2140, 3250, 0 },
		{ 2139, 1740, 0 },
		{ 0, 1681, 0 },
		{ 2032, 1852, 0 },
		{ 2135, 2254, 0 },
		{ 2055, 2175, 0 },
		{ 2135, 2256, 0 },
		{ 2135, 2257, 0 },
		{ 2129, 3629, 292 },
		{ 2045, 2063, 0 },
		{ 2129, 3534, 299 },
		{ 2123, 2198, 0 },
		{ 2140, 3398, 0 },
		{ 2138, 1687, 0 },
		{ 2134, 1672, 0 },
		{ 2136, 1454, 0 },
		{ 2134, 1634, 0 },
		{ 2137, 2023, 0 },
		{ 2138, 1691, 0 },
		{ 2135, 2267, 0 },
		{ 2133, 2164, 0 },
		{ 0, 2207, 0 },
		{ 2129, 3605, 293 },
		{ 2045, 2059, 0 },
		{ 2032, 1858, 0 },
		{ 2137, 2029, 0 },
		{ 2136, 1477, 0 },
		{ 2114, 1598, 0 },
		{ 2139, 1749, 0 },
		{ 2032, 1863, 0 },
		{ 2137, 2035, 0 },
		{ 2072, 1322, 0 },
		{ 2129, 3586, 285 },
		{ 2134, 1638, 0 },
		{ 2114, 1600, 0 },
		{ 2114, 1601, 0 },
		{ 2137, 2041, 0 },
		{ 2032, 1869, 0 },
		{ 2134, 1641, 0 },
		{ 2136, 1479, 0 },
		{ 2139, 1759, 0 },
		{ 2137, 2047, 0 },
		{ 2140, 3254, 0 },
		{ 2135, 2244, 0 },
		{ 2123, 2190, 0 },
		{ 2137, 2048, 0 },
		{ 2140, 3285, 0 },
		{ 2136, 1480, 0 },
		{ 2032, 1874, 0 },
		{ 2139, 1761, 0 },
		{ 2140, 3326, 0 },
		{ 2140, 3328, 0 },
		{ 2140, 3332, 0 },
		{ 2140, 3357, 0 },
		{ 2139, 1762, 0 },
		{ 2137, 1905, 0 },
		{ 2140, 3392, 0 },
		{ 2137, 1929, 0 },
		{ 2123, 2201, 0 },
		{ 0, 1877, 0 },
		{ 2129, 3624, 295 },
		{ 2123, 2186, 0 },
		{ 2072, 1326, 0 },
		{ 2138, 1702, 0 },
		{ 2129, 3532, 282 },
		{ 2114, 1605, 0 },
		{ 2133, 2115, 0 },
		{ 2133, 2116, 0 },
		{ 2129, 3549, 300 },
		{ 2129, 3551, 288 },
		{ 2139, 849, 289 },
		{ 2129, 3577, 294 },
		{ 0, 2058, 0 },
		{ 2136, 1516, 0 },
		{ 2129, 3588, 301 },
		{ 2138, 1706, 0 },
		{ 2072, 1348, 0 },
		{ 2137, 1986, 0 },
		{ 2133, 2124, 0 },
		{ 2140, 3312, 0 },
		{ 2139, 1770, 0 },
		{ 2140, 3318, 0 },
		{ 0, 2170, 0 },
		{ 2140, 3322, 0 },
		{ 2140, 3324, 0 },
		{ 2072, 1307, 0 },
		{ 2139, 1772, 0 },
		{ 2140, 3330, 0 },
		{ 2136, 1529, 0 },
		{ 0, 1815, 0 },
		{ 2129, 3536, 306 },
		{ 2114, 1611, 0 },
		{ 2129, 3541, 304 },
		{ 2140, 3384, 0 },
		{ 2129, 3545, 284 },
		{ 2129, 3547, 296 },
		{ 2140, 3386, 0 },
		{ 2140, 3388, 0 },
		{ 2129, 3553, 303 },
		{ 0, 1316, 0 },
		{ 2133, 2132, 0 },
		{ 2140, 3394, 0 },
		{ 2129, 3579, 283 },
		{ 2129, 3581, 290 },
		{ 2129, 3584, 302 },
		{ 2140, 3396, 0 },
		{ 2139, 1776, 0 },
		{ 2129, 3590, 281 },
		{ 2129, 3592, 291 },
		{ 2136, 1531, 0 },
		{ 2140, 3404, 0 },
		{ 2129, 3594, 305 },
		{ 2097, 3462, 313 },
		{ 2090, 0, 278 },
		{ 0, 0, 279 },
		{ -2088, 31, 274 },
		{ -2089, 3490, 0 },
		{ 2129, 3474, 0 },
		{ 2097, 3452, 0 },
		{ 0, 0, 275 },
		{ 2097, 3475, 0 },
		{ -2094, 16, 0 },
		{ -2095, 3494, 0 },
		{ 2098, 0, 276 },
		{ 0, 3477, 0 },
		{ 2129, 3626, 0 },
		{ 0, 0, 277 },
		{ 0, 2426, 97 },
		{ 0, 0, 97 },
		{ 2114, 1617, 0 },
		{ 2133, 2138, 0 },
		{ 2134, 1658, 0 },
		{ 2106, 3445, 0 },
		{ 2129, 1905, 0 },
		{ 0, 1388, 0 },
		{ 2139, 1782, 0 },
		{ 2123, 2193, 0 },
		{ 2136, 1535, 0 },
		{ 2137, 2032, 0 },
		{ 2138, 1718, 0 },
		{ 2139, 1785, 0 },
		{ 0, 1620, 0 },
		{ 2140, 3314, 0 },
		{ 0, 0, 95 },
		{ 0, 3284, 103 },
		{ 0, 0, 103 },
		{ 2136, 1538, 0 },
		{ 2121, 3466, 0 },
		{ 2133, 1937, 0 },
		{ 2135, 2252, 0 },
		{ 0, 2202, 0 },
		{ 0, 2056, 0 },
		{ 2126, 3462, 0 },
		{ 2139, 1674, 0 },
		{ 2133, 2153, 0 },
		{ 0, 1352, 0 },
		{ 2130, 3475, 0 },
		{ 2125, 1835, 0 },
		{ 2133, 2156, 0 },
		{ 2139, 1791, 0 },
		{ 0, 2158, 0 },
		{ 0, 1665, 0 },
		{ 0, 2263, 0 },
		{ 0, 1548, 0 },
		{ 0, 2045, 0 },
		{ 0, 1725, 0 },
		{ 0, 1795, 0 },
		{ 2129, 3400, 0 },
		{ 0, 0, 101 },
		{ 2143, 0, 1 },
		{ -2143, 1195, 176 }
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
		0,
		0,
		0
	};
	yybackup = backup;
}
