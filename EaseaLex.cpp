#include <clex.h>

#line 1 "EaseaLex.l"

/****************************************************************************
EaseaLex.l
Lexical analyser for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@polytechnique.fr)
Ecole Polytechnique
Centre de Math�matiques Appliqu�es
91128 Palaiseau cedex
  ****************************************************************************/
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma comment(lib, "libAlexYacc.lib")
#endif
#if defined __WIN32__ || defined __WIN64__
#include "Easea.h"
#include "EaseaParse.h"
#else
#include "Easea.h"
#include "EaseaParse.h"
#endif
#if defined WIN32 || defined WIN64
#include <direct.h>
#else
#include <unistd.h>
#endif
#ifndef SIZE_MAX
# define SIZE_MAX ((size_t) -1)
#endif
#ifndef SSIZE_MAX
# define SSIZE_MAX ((ssize_t) (SIZE_MAX / 2))
#endif
#include "debug.h"
#include <iostream>
#include <sstream>
#if defined __WIN64__ || defined __APPLE__ || defined __WIN32__
/* Mac OS X don't have strndup even if _GNU_SOURCE is defined */
char *strndup (const char *s, size_t n){
    size_t len = strlen (s);
    char *ret;

    if (len <= n)
	return strdup (s);

    ret = (char *)malloc(n + 1);
    strncpy(ret, s, n);
    ret[n] = '\0';
    return ret;
}
ssize_t getline (char **lineptr, size_t *n, FILE *fp){
    ssize_t result;
    size_t cur_len = 0;

    if (lineptr == NULL || n == NULL || fp == NULL){
	errno = EINVAL;
	return -1;
    }

    if (*lineptr == NULL || *n == 0){
	*n = 120;
	*lineptr = (char *) malloc (*n);
	if (*lineptr == NULL){
	    result = -1;
	    goto end;
	}
    }

    for (;;){
	int i;

	i = getc (fp);
	if (i == EOF){
	    result = -1;
	    break;
	}

	/* Make enough space for len+1 (for final NUL) bytes.  */
	if (cur_len + 1 >= *n){
	    size_t needed_max =	SSIZE_MAX < SIZE_MAX ? (size_t) SSIZE_MAX + 1 : SIZE_MAX;
	    size_t needed = 2 * *n + 1;   /* Be generous. */
	    char *new_lineptr;

	    if (needed_max < needed)
		needed = needed_max;
	    if (cur_len + 1 >= needed){
		result = -1;
		goto end;
	    }

	    new_lineptr = (char *) realloc (*lineptr, needed);
	    if (new_lineptr == NULL){
		result = -1;
		goto end;
	    }

	    *lineptr = new_lineptr;
	    *n = needed;
	}

	(*lineptr)[cur_len] = i;
	cur_len++;

	if (i == '\n')
	    break;
    }
    (*lineptr)[cur_len] = '\0';
    result = cur_len ? (ssize_t) cur_len : result;

end:
    return result;
}
#endif

/* getline implementation is copied from glibc. */


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
  
#line 160 "EaseaLex.cpp"
// repeated because of possible precompiled header
#include <clex.h>

#include "EaseaLex.h"

#line 164 "EaseaLex.l"
                                         
#line 172 "EaseaLex.l"
 // lexical analyser name and class definition
#line 170 "EaseaLex.cpp"
/////////////////////////////////////////////////////////////////////////////
// constructor

YYLEXNAME::YYLEXNAME()
{
	yytables();
#line 201 "EaseaLex.l"
                                
  bFunction=bWithinEvaluator=bWithinOptimiser=bDisplayFunction=bInitFunction=bNotFinishedYet=0;
  bSymbolInserted=bDoubleQuotes=bWithinDisplayFunction=bWithinInitialiser=bWithinMutator=bWithinXover=0;
  bWaitingForSemiColon=bFinishNB_GEN=bFinishMINIMISE=bFinishMINIMIZE=bGenerationReplacementFunction=0;
  bCatchNextSemiColon,bWaitingToClosePopulation=bMethodsInGenome=0;
  bBoundCheckingFunction = bWithinCUDA_Initializer=bWithinMAKEFILEOPTION =bWithinCUDA_Evaluator=0;
  bIsParentReduce = bIsOffspringReduce = false;
  bGPOPCODE_ANALYSIS = bGenerationFunctionBeforeReplacement = bEndGeneration = bBeginGeneration = bEndGenerationFunction = bBeginGenerationFunction = false;
  iGP_OPCODE_FIELD = accolade_counter = 0;
  iNoOp = 0;
  bCOPY_GP_EVAL_GPU,bIsCopyingGPEval = false;

#line 190 "EaseaLex.cpp"
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
#line 221 "EaseaLex.l"

  // extract yylval for use later on in actions
  YYSTYPE& yylval = *(YYSTYPE*)yyparserptr->yylvalptr;
  
#line 241 "EaseaLex.cpp"
	yyreturnflg = 1;
	switch (action) {
	case 1:
		{
#line 227 "EaseaLex.l"

#line 248 "EaseaLex.cpp"
		}
		break;
	case 2:
		{
#line 230 "EaseaLex.l"

  BEGIN TEMPLATE_ANALYSIS; yyless(yyleng-1);
 
#line 257 "EaseaLex.cpp"
		}
		break;
	case 3:
		{
#line 238 "EaseaLex.l"
             
  char sFileName[1000];
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".cpp"); 
  fpOutputFile=fopen(sFileName,"w");
 
#line 269 "EaseaLex.cpp"
		}
		break;
	case 4:
		{
#line 244 "EaseaLex.l"
fprintf(fpOutputFile,"EASEA");
#line 276 "EaseaLex.cpp"
		}
		break;
	case 5:
		{
#line 245 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sPROJECT_NAME);
#line 283 "EaseaLex.cpp"
		}
		break;
	case 6:
		{
#line 246 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sEZ_PATH);
#line 290 "EaseaLex.cpp"
		}
		break;
	case 7:
		{
#line 247 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sTPL_DIR);
#line 297 "EaseaLex.cpp"
		}
		break;
	case 8:
		{
#line 248 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sEO_DIR);
#line 304 "EaseaLex.cpp"
		}
		break;
	case 9:
		{
#line 249 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sLOWER_CASE_PROJECT_NAME);
#line 311 "EaseaLex.cpp"
		}
		break;
	case 10:
		{
#line 250 "EaseaLex.l"
switch (OPERATING_SYSTEM) {
  case UNIX : fprintf(fpOutputFile,"UNIX_OS"); break;
  case WINDOWS : fprintf(fpOutputFile,"WINDOWS_OS"); break;
  case UNKNOWN_OS : fprintf(fpOutputFile,"UNKNOWN_OS"); break;
  }
 
#line 323 "EaseaLex.cpp"
		}
		break;
	case 11:
		{
#line 256 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user declarations.\n");
  yyreset();
  yyin = fpGenomeFile;                                                    // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_USER_DECLARATIONS;
 
#line 336 "EaseaLex.cpp"
		}
		break;
	case 12:
		{
#line 264 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user cuda.\n");
  yyreset();
  yyin = fpGenomeFile;                                                    // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_USER_CUDA;
 
#line 349 "EaseaLex.cpp"
		}
		break;
	case 13:
		{
#line 272 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting initialisation function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISATION_FUNCTION;
 
#line 362 "EaseaLex.cpp"
		}
		break;
	case 14:
		{
#line 280 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting generation before reduce function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bGenerationFunctionBeforeReplacement = true;
  BEGIN COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT;
 
#line 375 "EaseaLex.cpp"
		}
		break;
	case 15:
		{
#line 289 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  if (bVERBOSE) printf ("Inserting at the begining of each generation function.\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter+1, sRAW_PROJECT_NAME);

  yyreset();
  yyin = fpGenomeFile;
  bBeginGeneration = true;
  bEndGeneration = false;
  lineCounter = 1;
  BEGIN COPY_BEG_GENERATION_FUNCTION;
 
#line 394 "EaseaLex.cpp"
		}
		break;
	case 16:
		{
#line 308 "EaseaLex.l"

  if( bVERBOSE )printf("inserting gp parameters\n");
  //  fprintf(fpOutputFile,"#define MAX_XOVER_DEPTH",%d
  fprintf(fpOutputFile,"#define TREE_DEPTH_MAX %d\n",iMAX_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MAX %d\n",iMAX_INIT_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MIN %d\n",iMIN_INIT_TREE_D);

  fprintf(fpOutputFile,"#define MAX_PROGS_SIZE %d\n",iPRG_BUF_SIZE);
  fprintf(fpOutputFile,"#define NB_GPU %d\n",iNB_GPU);

  fprintf(fpOutputFile,"#define NO_FITNESS_CASES %d\n",iNO_FITNESS_CASES);

#line 412 "EaseaLex.cpp"
		}
		break;
	case 17:
		{
#line 326 "EaseaLex.l"

  
  fprintf(fpOutputFile,"enum OPCODE              {"); 
  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"%s,",opDesc[i]->opcode->c_str());
  }
  fprintf(fpOutputFile,"OPCODE_SIZE, OP_RETURN};\n");


  fprintf(fpOutputFile,"const char* opCodeName[]={"); 
  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"%s",opDesc[i]->realName->c_str());
    if( i<(iNoOp-1) )fprintf(fpOutputFile,",");
  }
  fprintf(fpOutputFile,"};\n"); 
  fprintf(fpOutputFile,"unsigned opArity[]=     {"); 
  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"%d",opDesc[i]->arity);
    if( i<(iNoOp-1) )fprintf(fpOutputFile,",");
  }
  fprintf(fpOutputFile,"};\n"); 


  // count the number of variable (arity zero and non-erc operator)
  unsigned var_len = 0;
  for( unsigned i=0 ; i<iNoOp ; i++ ){
    if( opDesc[i]->arity==0 && !opDesc[i]->isERC ) var_len++;
  }
  if( bVERBOSE ) printf("var length is %d\n",var_len);
  fprintf(fpOutputFile,"#define VAR_LEN %d\n",var_len); 
 
#line 449 "EaseaLex.cpp"
		}
		break;
	case 18:
		{
#line 358 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"    case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"      %s",opDesc[i]->gpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"      break;\n");

  }
 
#line 463 "EaseaLex.cpp"
		}
		break;
	case 19:
		{
#line 367 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"  case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"    %s\n",opDesc[i]->cpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"    break;\n");
  }
 
#line 476 "EaseaLex.cpp"
		}
		break;
	case 20:
		{
#line 376 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Analysing GP OP code from ez file\n");
  BEGIN COPY_GP_OPCODE;
 
#line 489 "EaseaLex.cpp"
		}
		break;
	case 21:
		{
#line 385 "EaseaLex.l"

  if (bVERBOSE) printf ("found begin section\n");
  bGPOPCODE_ANALYSIS = true;
  BEGIN GP_RULE_ANALYSIS;
 
#line 500 "EaseaLex.cpp"
		}
		break;
	case 22:
		{
#line 391 "EaseaLex.l"
 
  if (bVERBOSE) printf ("found end section\n");
  if( bGPOPCODE_ANALYSIS ){
    rewind(fpGenomeFile);
    yyin = fpTemplateFile;
    bGPOPCODE_ANALYSIS = false;
    
    OPCodeDesc::sort(opDesc,iNoOp);
    /*for( unsigned i=0 ; i<iNoOp ; i++ ){
      opDesc[i]->show();
      }*/
    BEGIN TEMPLATE_ANALYSIS; 
  }  
 
#line 520 "EaseaLex.cpp"
		}
		break;
	case 23:
		{
#line 406 "EaseaLex.l"

  if (bVERBOSE) printf("*** No GP OP codes were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
 
#line 532 "EaseaLex.cpp"
		}
		break;
	case 24:
		{
#line 412 "EaseaLex.l"

#line 539 "EaseaLex.cpp"
		}
		break;
	case 25:
		{
#line 413 "EaseaLex.l"
if( bGPOPCODE_ANALYSIS )printf("\n");lineCounter++;
#line 546 "EaseaLex.cpp"
		}
		break;
	case 26:
		{
#line 420 "EaseaLex.l"

  // this rule match the OP_NAME
  if( iGP_OPCODE_FIELD != 0 ){
    fprintf(stderr,"Error, OP_CODE name must be given first\n");
    exit(-1);
  }
  opDesc[iNoOp] = new OPCodeDesc();
  opDesc[iNoOp]->opcode = new string(yytext);
 
#line 561 "EaseaLex.cpp"
		}
		break;
	case 27:
		{
#line 430 "EaseaLex.l"

#line 568 "EaseaLex.cpp"
		}
		break;
	case 28:
		{
#line 432 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 1 ){
    fprintf(stderr,"Error, op code real name must be given at the second place\n");
    exit(-1);
  }
  opDesc[iNoOp]->realName = new string(yytext);
 
#line 581 "EaseaLex.cpp"
		}
		break;
	case 29:
		{
#line 441 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 2 ){
    fprintf(stderr,"Error, arity must be given at the third place\n");
    exit(-1);
  }
  char* endptr;
  
  opDesc[iNoOp]->arity = strtol(yytext,&endptr,10);
  if( endptr==yytext ){
    fprintf(stderr, "warning, unable to translate this arity %s assuming 0\n",yytext);
    opDesc[iNoOp]->arity = 0;
  }
 
#line 600 "EaseaLex.cpp"
		}
		break;
	case 30:
		{
#line 455 "EaseaLex.l"

#line 607 "EaseaLex.cpp"
		}
		break;
	case 31:
		{
#line 456 "EaseaLex.l"

  iGP_OPCODE_FIELD = 0;
  iNoOp++;
 
#line 617 "EaseaLex.cpp"
		}
		break;
	case 32:
		{
#line 461 "EaseaLex.l"

  if( bGPOPCODE_ANALYSIS ) iGP_OPCODE_FIELD++;
 
#line 626 "EaseaLex.cpp"
		}
		break;
	case 33:
		{
#line 466 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 3 ){
    fprintf(stderr,"Error, code must be given at the forth place\n");
    exit(-1);
  }
  if( bVERBOSE ) printf("begining of the code part\n");
  accolade_counter=1;

//  printf("arity : %d\n",opDesc[iNoOp]->arity);
  if( opDesc[iNoOp]->arity>=2 )
    opDesc[iNoOp]->gpuCodeStream << "OP2 = stack[--sp];\n      ";
  if( opDesc[iNoOp]->arity>=1 )
    opDesc[iNoOp]->gpuCodeStream << "OP1 = stack[--sp];\n      ";

  BEGIN GP_COPY_OPCODE_CODE;
 
#line 648 "EaseaLex.cpp"
		}
		break;
	case 34:
		{
#line 487 "EaseaLex.l"

  accolade_counter++;
  opDesc[iNoOp]->cpuCodeStream << "{";
  opDesc[iNoOp]->gpuCodeStream << "{";
 
#line 659 "EaseaLex.cpp"
		}
		break;
	case 35:
		{
#line 493 "EaseaLex.l"

  accolade_counter--;
  if( accolade_counter==0 ){
    opDesc[iNoOp]->gpuCodeStream << "\n      stack[sp++] = RESULT;\n";

    BEGIN GP_RULE_ANALYSIS;
  }
  else{
    opDesc[iNoOp]->cpuCodeStream << "}";
    opDesc[iNoOp]->gpuCodeStream << "}";
  }
 
#line 677 "EaseaLex.cpp"
		}
		break;
	case 36:
		{
#line 506 "EaseaLex.l"

  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
//  printf("input no : %d\n",no_input);
  opDesc[iNoOp]->cpuCodeStream << "input["<< no_input <<"]" ;
  opDesc[iNoOp]->gpuCodeStream << "input["<< no_input << "]";  
 
#line 690 "EaseaLex.cpp"
		}
		break;
	case 37:
		{
#line 514 "EaseaLex.l"

  opDesc[iNoOp]->isERC = true;
  opDesc[iNoOp]->cpuCodeStream << "root->erc_value" ;
  opDesc[iNoOp]->gpuCodeStream << "k_progs[start_prog++];" ;
//  printf("ERC matched\n");

#line 702 "EaseaLex.cpp"
		}
		break;
	case 38:
		{
#line 521 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << "\n  ";
  opDesc[iNoOp]->gpuCodeStream << "\n    ";
 
#line 712 "EaseaLex.cpp"
		}
		break;
	case 39:
		{
#line 527 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << yytext;
  opDesc[iNoOp]->gpuCodeStream << yytext;
 
#line 722 "EaseaLex.cpp"
		}
		break;
	case 40:
		{
#line 532 "EaseaLex.l"
 
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  if( bVERBOSE ) printf("Insert GP eval header\n");
  iCOPY_GP_EVAL_STATUS = EVAL_HDR;
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 738 "EaseaLex.cpp"
		}
		break;
	case 41:
		{
#line 543 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = false;
  BEGIN COPY_GP_EVAL;
 
#line 755 "EaseaLex.cpp"
		}
		break;
	case 42:
		{
#line 557 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 772 "EaseaLex.cpp"
		}
		break;
	case 43:
		{
#line 571 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 788 "EaseaLex.cpp"
		}
		break;
	case 44:
		{
#line 582 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 805 "EaseaLex.cpp"
		}
		break;
	case 45:
		{
#line 595 "EaseaLex.l"

  switch(iCOPY_GP_EVAL_STATUS){
  case EVAL_HDR:
    fprintf(stderr,"Error, no evaluator header has been defined\n");
    exit(-1);
  case EVAL_BDY:
    fprintf(stderr,"Error, no evaluator body has been defined\n");
    exit(-1);
  case EVAL_FTR:
    fprintf(stderr,"Error, no evaluator footer has been defined\n");
    exit(-1);
  }
 
#line 824 "EaseaLex.cpp"
		}
		break;
	case 46:
		{
#line 610 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_HDR){
    bIsCopyingGPEval = true;
  }
 
#line 835 "EaseaLex.cpp"
		}
		break;
	case 47:
		{
#line 616 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_BDY){
    bIsCopyingGPEval = true;
  }
 
#line 846 "EaseaLex.cpp"
		}
		break;
	case 48:
		{
#line 624 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_FTR){
    bIsCopyingGPEval = true;
  }
 
#line 857 "EaseaLex.cpp"
		}
		break;
	case 49:
		{
#line 630 "EaseaLex.l"

  if( bIsCopyingGPEval ){
    bIsCopyingGPEval = false;
    bCOPY_GP_EVAL_GPU = false;
    rewind(fpGenomeFile);
    yyin = fpTemplateFile;
    BEGIN TEMPLATE_ANALYSIS;
  }
 
#line 872 "EaseaLex.cpp"
		}
		break;
	case 50:
		{
#line 640 "EaseaLex.l"

  if( bIsCopyingGPEval ) fprintf(fpOutputFile,"%s",yytext);
 
#line 881 "EaseaLex.cpp"
		}
		break;
	case 51:
		{
#line 644 "EaseaLex.l"

  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[i*NUMTHREAD2+tid]" );
    else fprintf(fpOutputFile, "outputs[i]" );
  
 
#line 894 "EaseaLex.cpp"
		}
		break;
	case 52:
		{
#line 652 "EaseaLex.l"

  char* endptr;
  unsigned no_output = strtol(yytext+strlen("OUTPUT["),&endptr,10);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[(i+%d)*NUMTHREAD2+tid]", no_output);
    else fprintf(fpOutputFile, "outputs[i+%d]", no_output );
  
 
#line 909 "EaseaLex.cpp"
		}
		break;
	case 53:
		{
#line 662 "EaseaLex.l"

	char *var;
	var = strndup(yytext+strlen("OUTPUT["), strlen(yytext) - strlen("OUTPUT[") - 1);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[(i+%s)*NUMTHREAD2+tid]", var);
    else fprintf(fpOutputFile, "outputs[i+%s]", var);
  
 
#line 924 "EaseaLex.cpp"
		}
		break;
	case 54:
		{
#line 672 "EaseaLex.l"

  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[i*NUMTHREAD2+tid]" );
    else fprintf(fpOutputFile, "inputs[i][0]" );
  
 
#line 937 "EaseaLex.cpp"
		}
		break;
	case 55:
		{
#line 680 "EaseaLex.l"

  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[(i+%d)*NUMTHREAD2+tid]", no_input);
    else fprintf(fpOutputFile, "inputs[i+%d][0]", no_input );
  
 
#line 952 "EaseaLex.cpp"
		}
		break;
	case 56:
		{
#line 690 "EaseaLex.l"

	char *var;
	var = strndup(yytext+strlen("INPUT["), strlen(yytext) - strlen("INPUT[") - 1);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[(i+%s)*NUMTHREAD2+tid]", var);
    else fprintf(fpOutputFile, "inputs[i+%s][0]", var);
  
 
#line 967 "EaseaLex.cpp"
		}
		break;
	case 57:
		{
#line 700 "EaseaLex.l"

  if( bIsCopyingGPEval )
    if( iCOPY_GP_EVAL_STATUS==EVAL_FTR )
      if( bCOPY_GP_EVAL_GPU ){
	fprintf(fpOutputFile,"k_results[index] =");
      }
      else fprintf(fpOutputFile,"return fitness=");
 
#line 981 "EaseaLex.cpp"
		}
		break;
	case 58:
		{
#line 712 "EaseaLex.l"

  if( bIsCopyingGPEval )
    switch(iCOPY_GP_EVAL_STATUS){
    case EVAL_FTR:
    case EVAL_HDR:
      fprintf(fpOutputFile,"\n  ");
      break;
    case EVAL_BDY:
      fprintf(fpOutputFile,"\n      ");
      break;
    }
 
#line 999 "EaseaLex.cpp"
		}
		break;
	case 59:
		{
#line 726 "EaseaLex.l"

  if( bIsCopyingGPEval )
    fprintf(fpOutputFile,"return fitness = "); 
 
#line 1009 "EaseaLex.cpp"
		}
		break;
	case 60:
		{
#line 733 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  yyreset();
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Evaluation population in a single function!!.\n");
  lineCounter = 1;
  BEGIN COPY_INSTEAD_EVAL;
 
#line 1023 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 742 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting at the end of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bEndGeneration = true;
  bBeginGeneration = false;
  BEGIN COPY_END_GENERATION_FUNCTION;
 
#line 1037 "EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 751 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Bound Checking function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_BOUND_CHECKING_FUNCTION;
 
#line 1049 "EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 758 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 
#line 1061 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 765 "EaseaLex.l"

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
 
#line 1090 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 788 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 
#line 1107 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 800 "EaseaLex.l"

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
 
#line 1133 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 821 "EaseaLex.l"

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
  
 
#line 1154 "EaseaLex.cpp"
		}
		break;
#line 839 "EaseaLex.l"
  
#line 853 "EaseaLex.l"
      
#line 1161 "EaseaLex.cpp"
	case 68:
		{
#line 861 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 1174 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 870 "EaseaLex.l"

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
 
#line 1197 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 887 "EaseaLex.l"

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
 
#line 1220 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 905 "EaseaLex.l"

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
 
#line 1252 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 932 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  fprintf (fpOutputFile,"// Memberwise serialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->serializeIndividual(fpOutputFile, "this");
  //fprintf(fpOutputFile,"\tEASEA_Line << endl;\n");
 
#line 1266 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 941 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome deserializer.\n");
  fprintf (fpOutputFile,"// Memberwise deserialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->deserializeIndividual(fpOutputFile, "this");
 
#line 1279 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 949 "EaseaLex.l"

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
 
#line 1300 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 965 "EaseaLex.l"

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
 
#line 1322 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 982 "EaseaLex.l"
       
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
 
#line 1350 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 1004 "EaseaLex.l"

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
 
#line 1372 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 1020 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 1387 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 1029 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 1399 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 1037 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 1411 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 1044 "EaseaLex.l"

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
 
#line 1442 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 1069 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 1455 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 1076 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 1469 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 1085 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISER;   
 
#line 1481 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 1092 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 1494 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 1100 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 1506 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 1106 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 1518 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 1112 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 1530 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 1118 "EaseaLex.l"
      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 1543 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 1125 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 1556 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 1132 "EaseaLex.l"
 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 1570 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 1141 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 1581 "EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 1146 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 1595 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 1155 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 1609 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 1164 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 1623 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 1174 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 1636 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 1182 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 1645 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 1186 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 1654 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 1190 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 1663 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 1194 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 1672 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 1198 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 1682 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 1203 "EaseaLex.l"

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

#line 1701 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 1216 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 1708 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 1217 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 1715 "EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 1218 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 1722 "EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 1219 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 1729 "EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 1220 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 1736 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 1221 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1743 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 1222 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1750 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 1223 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1757 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 1224 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1764 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 1225 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1771 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 1226 "EaseaLex.l"
 fprintf(fpOutputFile,"%d",nELITE); 
#line 1778 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 1227 "EaseaLex.l"
 fprintf(fpOutputFile,"%d",iNO_FITNESS_CASES); 
#line 1785 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 1230 "EaseaLex.l"

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
 
#line 1804 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 1243 "EaseaLex.l"

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
 
#line 1823 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 1256 "EaseaLex.l"

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
 
#line 1842 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 1269 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1852 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 1273 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1859 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 1274 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1866 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 1275 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1873 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1276 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1880 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1277 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1887 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1278 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1894 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1279 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1901 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1280 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1908 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1281 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1915 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1282 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1922 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1284 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1929 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1285 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1936 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1287 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bREMOTE_ISLAND_MODEL);
#line 1943 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1288 "EaseaLex.l"
if(strlen(sIP_FILE)>0)fprintf(fpOutputFile,"%s",sIP_FILE); else fprintf(fpOutputFile,"NULL");
#line 1950 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1289 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMIGRATION_PROBABILITY);
#line 1957 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1290 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nSERVER_PORT);
#line 1964 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1292 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1971 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1293 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1978 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1294 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1985 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1295 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1992 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1296 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1999 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1298 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 2006 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1299 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 2013 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1301 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 2027 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1309 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  if( TARGET==CUDA )
    strcat(sFileName,"Individual.cu");
  else if( TARGET==STD )
    strcat(sFileName,"Individual.cpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 2044 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1320 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2058 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1328 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2072 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1337 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2086 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1346 "EaseaLex.l"

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

#line 2149 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1403 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded without warnings.\n");
  printf ("You can now type \"make\" to compile your project.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 2166 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1415 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2173 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1421 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2185 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1427 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2198 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1434 "EaseaLex.l"

#line 2205 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1435 "EaseaLex.l"
lineCounter++;
#line 2212 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1437 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2224 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1443 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2237 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1451 "EaseaLex.l"

#line 2244 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1452 "EaseaLex.l"

  lineCounter++;
 
#line 2253 "EaseaLex.cpp"
		}
		break;
#line 1455 "EaseaLex.l"
               
#line 2258 "EaseaLex.cpp"
	case 158:
		{
#line 1456 "EaseaLex.l"

  fprintf (fpOutputFile,"// User CUDA\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2268 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1462 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user CUDA were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2281 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1470 "EaseaLex.l"

#line 2288 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1471 "EaseaLex.l"

  lineCounter++;
 
#line 2297 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1475 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2309 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1481 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2323 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1489 "EaseaLex.l"

#line 2330 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1490 "EaseaLex.l"

  lineCounter++;
 
#line 2339 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1494 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
      
  BEGIN COPY;
 
#line 2353 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1502 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2368 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1511 "EaseaLex.l"

#line 2375 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1512 "EaseaLex.l"
lineCounter++;
#line 2382 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1517 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2396 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1526 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2410 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1534 "EaseaLex.l"

#line 2417 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1535 "EaseaLex.l"
lineCounter++;
#line 2424 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1538 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2440 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1549 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2456 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1559 "EaseaLex.l"

#line 2463 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1562 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    bBeginGeneration = 0;
    bBeginGenerationFunction = 1;
    if( bLINE_NUM_EZ_FILE )
      fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2481 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1575 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2498 "EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 1587 "EaseaLex.l"

#line 2505 "EaseaLex.cpp"
		}
		break;
	case 180:
		{
#line 1588 "EaseaLex.l"
lineCounter++;
#line 2512 "EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 1590 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2528 "EaseaLex.cpp"
		}
		break;
	case 182:
		{
#line 1602 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2544 "EaseaLex.cpp"
		}
		break;
	case 183:
		{
#line 1612 "EaseaLex.l"
lineCounter++;
#line 2551 "EaseaLex.cpp"
		}
		break;
	case 184:
		{
#line 1613 "EaseaLex.l"

#line 2558 "EaseaLex.cpp"
		}
		break;
	case 185:
		{
#line 1617 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2573 "EaseaLex.cpp"
		}
		break;
	case 186:
		{
#line 1627 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2588 "EaseaLex.cpp"
		}
		break;
	case 187:
		{
#line 1636 "EaseaLex.l"

#line 2595 "EaseaLex.cpp"
		}
		break;
	case 188:
		{
#line 1639 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    //fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    fprintf (fpOutputFile,"{\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2609 "EaseaLex.cpp"
		}
		break;
	case 189:
		{
#line 1647 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2623 "EaseaLex.cpp"
		}
		break;
	case 190:
		{
#line 1655 "EaseaLex.l"

#line 2630 "EaseaLex.cpp"
		}
		break;
	case 191:
		{
#line 1659 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2638 "EaseaLex.cpp"
		}
		break;
	case 192:
		{
#line 1661 "EaseaLex.l"

#line 2645 "EaseaLex.cpp"
		}
		break;
	case 193:
		{
#line 1667 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2652 "EaseaLex.cpp"
		}
		break;
	case 194:
		{
#line 1668 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2659 "EaseaLex.cpp"
		}
		break;
	case 195:
	case 196:
		{
#line 1671 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2670 "EaseaLex.cpp"
		}
		break;
	case 197:
	case 198:
		{
#line 1676 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2679 "EaseaLex.cpp"
		}
		break;
	case 199:
	case 200:
		{
#line 1679 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2688 "EaseaLex.cpp"
		}
		break;
	case 201:
	case 202:
		{
#line 1682 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2705 "EaseaLex.cpp"
		}
		break;
	case 203:
	case 204:
		{
#line 1693 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2719 "EaseaLex.cpp"
		}
		break;
	case 205:
	case 206:
		{
#line 1701 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2728 "EaseaLex.cpp"
		}
		break;
	case 207:
	case 208:
		{
#line 1704 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2737 "EaseaLex.cpp"
		}
		break;
	case 209:
	case 210:
		{
#line 1707 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2746 "EaseaLex.cpp"
		}
		break;
	case 211:
	case 212:
		{
#line 1710 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2755 "EaseaLex.cpp"
		}
		break;
	case 213:
	case 214:
		{
#line 1713 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2764 "EaseaLex.cpp"
		}
		break;
	case 215:
	case 216:
		{
#line 1717 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2776 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1723 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2783 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1724 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2790 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1725 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2797 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1726 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2807 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1731 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2814 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1732 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2821 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1733 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2828 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1734 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2835 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1735 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2842 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1736 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2849 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1737 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2856 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1738 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2863 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1739 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2871 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1741 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2879 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1743 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2887 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1745 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2897 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1749 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2904 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1750 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2911 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1751 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2922 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1756 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2929 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1757 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2938 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1760 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2950 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1766 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2959 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1769 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2971 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1775 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2982 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1780 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2998 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1790 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3005 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1793 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 3014 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1796 "EaseaLex.l"
BEGIN COPY;
#line 3021 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1798 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 3028 "EaseaLex.cpp"
		}
		break;
	case 247:
	case 248:
	case 249:
		{
#line 1801 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 3041 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1806 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 3052 "EaseaLex.cpp"
		}
		break;
	case 251:
		{
#line 1811 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 3061 "EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1820 "EaseaLex.l"
;
#line 3068 "EaseaLex.cpp"
		}
		break;
	case 253:
		{
#line 1821 "EaseaLex.l"
;
#line 3075 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1822 "EaseaLex.l"
;
#line 3082 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1823 "EaseaLex.l"
;
#line 3089 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1826 "EaseaLex.l"
 /* do nothing */ 
#line 3096 "EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1827 "EaseaLex.l"
 /*return '\n';*/ 
#line 3103 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1828 "EaseaLex.l"
 /*return '\n';*/ 
#line 3110 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1831 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 3119 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1834 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 3129 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1838 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
//  printf("match gpnode\n");
  return GPNODE;
 
#line 3141 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1845 "EaseaLex.l"
return STATIC;
#line 3148 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1846 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 3155 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1847 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 3162 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1848 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 3169 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1849 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 3176 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1850 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 3183 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1852 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 3190 "EaseaLex.cpp"
		}
		break;
#line 1853 "EaseaLex.l"
  
#line 3195 "EaseaLex.cpp"
	case 269:
		{
#line 1854 "EaseaLex.l"
return GENOME; 
#line 3200 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1856 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 3210 "EaseaLex.cpp"
		}
		break;
	case 271:
	case 272:
	case 273:
		{
#line 1863 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 3219 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1864 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 3226 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1867 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 3234 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1869 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 3241 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1875 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 3253 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1881 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 3266 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1888 "EaseaLex.l"

#line 3273 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1890 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3284 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1901 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3299 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1911 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3310 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1917 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3319 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1921 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3334 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1934 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3346 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1940 "EaseaLex.l"

#line 3353 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1941 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3366 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1948 "EaseaLex.l"

#line 3373 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1949 "EaseaLex.l"
lineCounter++;
#line 3380 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1950 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3393 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1957 "EaseaLex.l"

#line 3400 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1958 "EaseaLex.l"
lineCounter++;
#line 3407 "EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1960 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3420 "EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1967 "EaseaLex.l"

#line 3427 "EaseaLex.cpp"
		}
		break;
	case 295:
		{
#line 1968 "EaseaLex.l"
lineCounter++;
#line 3434 "EaseaLex.cpp"
		}
		break;
	case 296:
		{
#line 1970 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3447 "EaseaLex.cpp"
		}
		break;
	case 297:
		{
#line 1977 "EaseaLex.l"

#line 3454 "EaseaLex.cpp"
		}
		break;
	case 298:
		{
#line 1978 "EaseaLex.l"
lineCounter++;
#line 3461 "EaseaLex.cpp"
		}
		break;
	case 299:
		{
#line 1984 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3468 "EaseaLex.cpp"
		}
		break;
	case 300:
		{
#line 1985 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3475 "EaseaLex.cpp"
		}
		break;
	case 301:
		{
#line 1986 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3482 "EaseaLex.cpp"
		}
		break;
	case 302:
		{
#line 1987 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3489 "EaseaLex.cpp"
		}
		break;
	case 303:
		{
#line 1988 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3496 "EaseaLex.cpp"
		}
		break;
	case 304:
		{
#line 1989 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3503 "EaseaLex.cpp"
		}
		break;
	case 305:
		{
#line 1990 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3510 "EaseaLex.cpp"
		}
		break;
	case 306:
		{
#line 1992 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3519 "EaseaLex.cpp"
		}
		break;
	case 307:
		{
#line 1995 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3532 "EaseaLex.cpp"
		}
		break;
	case 308:
	case 309:
		{
#line 2004 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3543 "EaseaLex.cpp"
		}
		break;
	case 310:
	case 311:
		{
#line 2009 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3552 "EaseaLex.cpp"
		}
		break;
	case 312:
	case 313:
		{
#line 2012 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3561 "EaseaLex.cpp"
		}
		break;
	case 314:
	case 315:
		{
#line 2015 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3573 "EaseaLex.cpp"
		}
		break;
	case 316:
	case 317:
		{
#line 2021 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3586 "EaseaLex.cpp"
		}
		break;
	case 318:
	case 319:
		{
#line 2028 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3595 "EaseaLex.cpp"
		}
		break;
	case 320:
	case 321:
		{
#line 2031 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3604 "EaseaLex.cpp"
		}
		break;
	case 322:
	case 323:
		{
#line 2034 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3613 "EaseaLex.cpp"
		}
		break;
	case 324:
	case 325:
		{
#line 2037 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3622 "EaseaLex.cpp"
		}
		break;
	case 326:
	case 327:
		{
#line 2040 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3631 "EaseaLex.cpp"
		}
		break;
	case 328:
		{
#line 2043 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3640 "EaseaLex.cpp"
		}
		break;
	case 329:
		{
#line 2046 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3650 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 2050 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3658 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 2052 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3669 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 2057 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3680 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 2062 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3688 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 2064 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3696 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 2066 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3704 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 2068 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3712 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 2070 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3720 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 2072 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3727 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 2073 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3734 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 2074 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3742 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 2076 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3750 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 2078 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3758 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 2080 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3765 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 2081 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3777 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 2087 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3786 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 2090 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3796 "EaseaLex.cpp"
		}
		break;
	case 347:
		{
#line 2094 "EaseaLex.l"
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
#line 3813 "EaseaLex.cpp"
		}
		break;
	case 348:
	case 349:
		{
#line 2106 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3823 "EaseaLex.cpp"
		}
		break;
	case 350:
		{
#line 2109 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3830 "EaseaLex.cpp"
		}
		break;
	case 351:
		{
#line 2116 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3837 "EaseaLex.cpp"
		}
		break;
	case 352:
		{
#line 2117 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3844 "EaseaLex.cpp"
		}
		break;
	case 353:
		{
#line 2118 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3851 "EaseaLex.cpp"
		}
		break;
	case 354:
		{
#line 2119 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3858 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 2120 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3865 "EaseaLex.cpp"
		}
		break;
	case 356:
		{
#line 2122 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3874 "EaseaLex.cpp"
		}
		break;
	case 357:
		{
#line 2126 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3887 "EaseaLex.cpp"
		}
		break;
	case 358:
		{
#line 2134 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3900 "EaseaLex.cpp"
		}
		break;
	case 359:
		{
#line 2143 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3913 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 2152 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3928 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 2162 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3935 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 2163 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3942 "EaseaLex.cpp"
		}
		break;
	case 363:
	case 364:
		{
#line 2166 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3953 "EaseaLex.cpp"
		}
		break;
	case 365:
	case 366:
		{
#line 2171 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3962 "EaseaLex.cpp"
		}
		break;
	case 367:
	case 368:
		{
#line 2174 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3971 "EaseaLex.cpp"
		}
		break;
	case 369:
	case 370:
		{
#line 2177 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3984 "EaseaLex.cpp"
		}
		break;
	case 371:
	case 372:
		{
#line 2184 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3997 "EaseaLex.cpp"
		}
		break;
	case 373:
	case 374:
		{
#line 2191 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 4006 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2194 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 4013 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2195 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4020 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2196 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4027 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2197 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 4037 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2202 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4044 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2203 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4051 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2204 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 4058 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2205 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 4065 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2206 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 4073 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2208 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 4081 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2210 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 4089 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2212 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 4097 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2214 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 4105 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2216 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 4113 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2218 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 4121 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2220 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 4128 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2221 "EaseaLex.l"
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
#line 4151 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2238 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 4162 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2243 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 4176 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2251 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 4183 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2257 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 4193 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2261 "EaseaLex.l"

#line 4200 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2264 "EaseaLex.l"
;
#line 4207 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2265 "EaseaLex.l"
;
#line 4214 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2266 "EaseaLex.l"
;
#line 4221 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2267 "EaseaLex.l"
;
#line 4228 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2269 "EaseaLex.l"
 /* do nothing */ 
#line 4235 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2270 "EaseaLex.l"
 /*return '\n';*/ 
#line 4242 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2271 "EaseaLex.l"
 /*return '\n';*/ 
#line 4249 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2273 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 4256 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2274 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 4263 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2275 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 4270 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2276 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4277 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2277 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4284 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2278 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4291 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2279 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4298 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2280 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4305 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2281 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4312 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2283 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4319 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2284 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4326 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2285 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4333 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2286 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4340 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2287 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4347 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2289 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4354 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2290 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4361 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2292 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4372 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2297 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4379 "EaseaLex.cpp"
		}
		break;
	case 422:
		{
#line 2299 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4390 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2304 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4397 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2307 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4404 "EaseaLex.cpp"
		}
		break;
	case 425:
		{
#line 2308 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4411 "EaseaLex.cpp"
		}
		break;
	case 426:
		{
#line 2309 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4418 "EaseaLex.cpp"
		}
		break;
	case 427:
		{
#line 2310 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4425 "EaseaLex.cpp"
		}
		break;
	case 428:
		{
#line 2311 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4432 "EaseaLex.cpp"
		}
		break;
#line 2312 "EaseaLex.l"
 
#line 4437 "EaseaLex.cpp"
	case 429:
		{
#line 2313 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4442 "EaseaLex.cpp"
		}
		break;
	case 430:
		{
#line 2314 "EaseaLex.l"
if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;
#line 4449 "EaseaLex.cpp"
		}
		break;
	case 431:
		{
#line 2315 "EaseaLex.l"
if(bVERBOSE) printf("\tMigration Probability...\n"); return MIGRATION_PROBABILITY;
#line 4456 "EaseaLex.cpp"
		}
		break;
	case 432:
		{
#line 2316 "EaseaLex.l"
if(bVERBOSE) printf("\tServer port...\n"); return SERVER_PORT;
#line 4463 "EaseaLex.cpp"
		}
		break;
#line 2318 "EaseaLex.l"
 
#line 4468 "EaseaLex.cpp"
	case 433:
	case 434:
	case 435:
		{
#line 2322 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4475 "EaseaLex.cpp"
		}
		break;
	case 436:
		{
#line 2323 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4482 "EaseaLex.cpp"
		}
		break;
	case 437:
		{
#line 2326 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4490 "EaseaLex.cpp"
		}
		break;
	case 438:
		{
#line 2329 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4497 "EaseaLex.cpp"
		}
		break;
	case 439:
		{
#line 2331 "EaseaLex.l"

  lineCounter++;

#line 4506 "EaseaLex.cpp"
		}
		break;
	case 440:
		{
#line 2334 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4516 "EaseaLex.cpp"
		}
		break;
	case 441:
		{
#line 2339 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4526 "EaseaLex.cpp"
		}
		break;
	case 442:
		{
#line 2344 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4536 "EaseaLex.cpp"
		}
		break;
	case 443:
		{
#line 2349 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4546 "EaseaLex.cpp"
		}
		break;
	case 444:
		{
#line 2354 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4556 "EaseaLex.cpp"
		}
		break;
	case 445:
		{
#line 2359 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4566 "EaseaLex.cpp"
		}
		break;
	case 446:
		{
#line 2368 "EaseaLex.l"
return  (char)yytext[0];
#line 4573 "EaseaLex.cpp"
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
#line 2370 "EaseaLex.l"


		       /////////////////////////////////////////////////////////////////////////////

inline char  mytolower(char c) {
  return ((c>=65)&&(c<=90)) ? c+=32:c;
}

inline int mystricmp(const char *string1, const char *string2){
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

  if (bVERBOSE) {
	printf("\n                                                                   ");
  	printf("\n                                   E A S E A                   (v1.1)");
  	printf("\n                              ___________________     ");
 	printf("\n                                                                    ");
	printf("\n		Stochastic Optimisation and Nature Inspired Computing");
	printf("\nLaboratoire des Sciences de l'Image, de l'Informatique et de la Teledetection");
	printf("\n			Universite de Strasbourg - France");
	printf("\n		Ogier Maitre - Frederic Kruger - Pierre Collet");
 	printf("\n                                                                    ");
  	printf("\n                              ___________________     ");
 	printf("\n                                                                    ");
  }
  
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
    else if(TARGET_FLAVOR == CMAES )
      strcat(sTemp,"CMAES_CUDA.tpl");
    else if( TARGET_FLAVOR == CUDA_FLAVOR_GP )
      strcat(sTemp,"CUDA_GP.tpl");
    else if(TARGET_FLAVOR == MEMETIC )
      strcat(sTemp,"CUDA_MEM.tpl");
    else 
      strcat(sTemp,"CUDA_MO.tpl");
    printf("tpl file : %s\n",sTemp);
    if (!(yyin = fpTemplateFile = fopen(sTemp, "r"))){
      fprintf(stderr,"\n*** Could not open %s.\n",sTemp);
      fprintf(stderr,"*** Please modify the EZ_PATH environment variable.\n");
      exit(1);
    } }

  if (TARGET==STD){
    if(TARGET_FLAVOR == STD_FLAVOR_SO)
      strcat(sTemp,"STD.tpl");
    else if (TARGET_FLAVOR == CMAES)
      strcat(sTemp,"CMAES.tpl");
    else if (TARGET_FLAVOR == MEMETIC )
      strcat(sTemp,"STD_MEM.tpl");
    //else if (TARGET_FLAVOR == STD_FLAVOR_GP )
   //   strcat(sTemp,"GP.tpl");
    else
      strcat(sTemp,"STD_MO.tpl");
    if (!(yyin = fpTemplateFile = fopen(sTemp, "r"))){
      fprintf(stderr,"\n*** Could not open %s.\n",sTemp);
      fprintf(stderr,"*** Please modify the EZ_PATH environment variable.\n");
      exit(1);
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

#line 4772 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		195,
		-196,
		0,
		197,
		-198,
		0,
		199,
		-200,
		0,
		201,
		-202,
		0,
		207,
		-208,
		0,
		209,
		-210,
		0,
		211,
		-212,
		0,
		213,
		-214,
		0,
		205,
		-206,
		0,
		203,
		-204,
		0,
		-245,
		0,
		-251,
		0,
		312,
		-313,
		0,
		314,
		-315,
		0,
		308,
		-309,
		0,
		320,
		-321,
		0,
		322,
		-323,
		0,
		324,
		-325,
		0,
		326,
		-327,
		0,
		310,
		-311,
		0,
		373,
		-374,
		0,
		318,
		-319,
		0,
		371,
		-372,
		0,
		316,
		-317,
		0,
		365,
		-366,
		0,
		367,
		-368,
		0,
		369,
		-370,
		0,
		363,
		-364,
		0
	};
	yymatch = match;

	yytransitionmax = 5248;
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
		{ 3098, 63 },
		{ 3098, 63 },
		{ 1918, 2021 },
		{ 1543, 1526 },
		{ 1544, 1526 },
		{ 2427, 2399 },
		{ 2427, 2399 },
		{ 0, 89 },
		{ 2017, 2013 },
		{ 2398, 2367 },
		{ 2398, 2367 },
		{ 73, 3 },
		{ 74, 3 },
		{ 2277, 45 },
		{ 2278, 45 },
		{ 71, 1 },
		{ 2857, 2859 },
		{ 2243, 2245 },
		{ 69, 1 },
		{ 0, 2299 },
		{ 167, 163 },
		{ 1918, 1899 },
		{ 167, 169 },
		{ 3098, 63 },
		{ 1412, 1411 },
		{ 3096, 63 },
		{ 1543, 1526 },
		{ 3156, 3151 },
		{ 2427, 2399 },
		{ 2260, 2259 },
		{ 1580, 1564 },
		{ 1581, 1564 },
		{ 2398, 2367 },
		{ 105, 90 },
		{ 73, 3 },
		{ 3100, 63 },
		{ 2277, 45 },
		{ 88, 63 },
		{ 3095, 63 },
		{ 3097, 63 },
		{ 3097, 63 },
		{ 3097, 63 },
		{ 3097, 63 },
		{ 3097, 63 },
		{ 3097, 63 },
		{ 3097, 63 },
		{ 3097, 63 },
		{ 3097, 63 },
		{ 3097, 63 },
		{ 72, 3 },
		{ 3099, 63 },
		{ 2276, 45 },
		{ 1633, 1627 },
		{ 1580, 1564 },
		{ 2428, 2399 },
		{ 1545, 1526 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 1582, 1564 },
		{ 3093, 63 },
		{ 1635, 1629 },
		{ 1506, 1485 },
		{ 3094, 63 },
		{ 1507, 1486 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3094, 63 },
		{ 3101, 63 },
		{ 2238, 42 },
		{ 1583, 1565 },
		{ 1584, 1565 },
		{ 1502, 1480 },
		{ 2025, 42 },
		{ 2486, 2457 },
		{ 2486, 2457 },
		{ 2404, 2372 },
		{ 2404, 2372 },
		{ 2407, 2375 },
		{ 2407, 2375 },
		{ 2040, 41 },
		{ 1503, 1481 },
		{ 1851, 39 },
		{ 2425, 2397 },
		{ 2425, 2397 },
		{ 1505, 1484 },
		{ 1508, 1487 },
		{ 1509, 1488 },
		{ 1510, 1489 },
		{ 1511, 1490 },
		{ 1512, 1491 },
		{ 1513, 1492 },
		{ 2238, 42 },
		{ 1583, 1565 },
		{ 2028, 42 },
		{ 1514, 1493 },
		{ 1515, 1494 },
		{ 2486, 2457 },
		{ 1516, 1496 },
		{ 2404, 2372 },
		{ 1519, 1499 },
		{ 2407, 2375 },
		{ 1520, 1500 },
		{ 2040, 41 },
		{ 1521, 1501 },
		{ 1851, 39 },
		{ 2425, 2397 },
		{ 2237, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2026, 41 },
		{ 2041, 42 },
		{ 1838, 39 },
		{ 1522, 1502 },
		{ 1585, 1565 },
		{ 2487, 2457 },
		{ 1523, 1503 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2027, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2035, 42 },
		{ 2033, 42 },
		{ 2046, 42 },
		{ 2034, 42 },
		{ 2046, 42 },
		{ 2037, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2036, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 1525, 1505 },
		{ 2029, 42 },
		{ 2031, 42 },
		{ 1526, 1506 },
		{ 2046, 42 },
		{ 1527, 1507 },
		{ 2046, 42 },
		{ 2044, 42 },
		{ 2032, 42 },
		{ 2046, 42 },
		{ 2045, 42 },
		{ 2038, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2043, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2030, 42 },
		{ 2046, 42 },
		{ 2042, 42 },
		{ 2046, 42 },
		{ 2039, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 2046, 42 },
		{ 1427, 23 },
		{ 1586, 1566 },
		{ 1587, 1566 },
		{ 1528, 1508 },
		{ 1414, 23 },
		{ 2434, 2405 },
		{ 2434, 2405 },
		{ 2447, 2418 },
		{ 2447, 2418 },
		{ 2450, 2421 },
		{ 2450, 2421 },
		{ 2461, 2432 },
		{ 2461, 2432 },
		{ 2474, 2445 },
		{ 2474, 2445 },
		{ 2475, 2446 },
		{ 2475, 2446 },
		{ 1529, 1509 },
		{ 1530, 1510 },
		{ 1531, 1511 },
		{ 1532, 1512 },
		{ 1533, 1513 },
		{ 1534, 1514 },
		{ 1427, 23 },
		{ 1586, 1566 },
		{ 1415, 23 },
		{ 1428, 23 },
		{ 1535, 1516 },
		{ 2434, 2405 },
		{ 1538, 1519 },
		{ 2447, 2418 },
		{ 1539, 1520 },
		{ 2450, 2421 },
		{ 1540, 1521 },
		{ 2461, 2432 },
		{ 1541, 1523 },
		{ 2474, 2445 },
		{ 1542, 1525 },
		{ 2475, 2446 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1439, 1417 },
		{ 1546, 1527 },
		{ 1547, 1528 },
		{ 1552, 1531 },
		{ 1588, 1566 },
		{ 1553, 1532 },
		{ 1554, 1533 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1431, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1420, 23 },
		{ 1418, 23 },
		{ 1433, 23 },
		{ 1419, 23 },
		{ 1433, 23 },
		{ 1422, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1421, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1555, 1534 },
		{ 1416, 23 },
		{ 1429, 23 },
		{ 1556, 1535 },
		{ 1423, 23 },
		{ 1559, 1539 },
		{ 1433, 23 },
		{ 1434, 23 },
		{ 1417, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1424, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1432, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1435, 23 },
		{ 1433, 23 },
		{ 1430, 23 },
		{ 1433, 23 },
		{ 1425, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 1433, 23 },
		{ 2012, 40 },
		{ 1589, 1567 },
		{ 1590, 1567 },
		{ 1548, 1529 },
		{ 1837, 40 },
		{ 2519, 2490 },
		{ 2519, 2490 },
		{ 2525, 2496 },
		{ 2525, 2496 },
		{ 1550, 1530 },
		{ 1549, 1529 },
		{ 2532, 2503 },
		{ 2532, 2503 },
		{ 2545, 2517 },
		{ 2545, 2517 },
		{ 1560, 1540 },
		{ 1551, 1530 },
		{ 1561, 1541 },
		{ 1562, 1542 },
		{ 1564, 1546 },
		{ 1565, 1547 },
		{ 1566, 1548 },
		{ 1567, 1549 },
		{ 2012, 40 },
		{ 1589, 1567 },
		{ 1842, 40 },
		{ 2546, 2518 },
		{ 2546, 2518 },
		{ 2519, 2490 },
		{ 1568, 1550 },
		{ 2525, 2496 },
		{ 1569, 1551 },
		{ 1592, 1568 },
		{ 1593, 1568 },
		{ 2532, 2503 },
		{ 1570, 1552 },
		{ 2545, 2517 },
		{ 1571, 1553 },
		{ 2011, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 2546, 2518 },
		{ 1852, 40 },
		{ 1572, 1554 },
		{ 1573, 1555 },
		{ 1591, 1567 },
		{ 1574, 1556 },
		{ 1592, 1568 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1839, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1847, 40 },
		{ 1845, 40 },
		{ 1855, 40 },
		{ 1846, 40 },
		{ 1855, 40 },
		{ 1849, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1848, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1576, 1559 },
		{ 1843, 40 },
		{ 1594, 1568 },
		{ 1577, 1560 },
		{ 1855, 40 },
		{ 1578, 1561 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1844, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1840, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1841, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1854, 40 },
		{ 1855, 40 },
		{ 1853, 40 },
		{ 1855, 40 },
		{ 1850, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 1855, 40 },
		{ 2851, 46 },
		{ 2852, 46 },
		{ 1595, 1569 },
		{ 1596, 1569 },
		{ 69, 46 },
		{ 2551, 2523 },
		{ 2551, 2523 },
		{ 1579, 1562 },
		{ 1442, 1418 },
		{ 1443, 1419 },
		{ 1447, 1421 },
		{ 2554, 2526 },
		{ 2554, 2526 },
		{ 2558, 2530 },
		{ 2558, 2530 },
		{ 1448, 1422 },
		{ 1449, 1423 },
		{ 1598, 1570 },
		{ 1599, 1571 },
		{ 1600, 1572 },
		{ 1602, 1576 },
		{ 1603, 1577 },
		{ 1604, 1578 },
		{ 2851, 46 },
		{ 1605, 1579 },
		{ 1595, 1569 },
		{ 2559, 2531 },
		{ 2559, 2531 },
		{ 2551, 2523 },
		{ 1612, 1598 },
		{ 1613, 1599 },
		{ 1614, 1599 },
		{ 1622, 1612 },
		{ 1623, 1612 },
		{ 2554, 2526 },
		{ 1450, 1424 },
		{ 2558, 2530 },
		{ 2293, 46 },
		{ 2850, 46 },
		{ 2292, 46 },
		{ 2292, 46 },
		{ 2292, 46 },
		{ 2292, 46 },
		{ 2292, 46 },
		{ 2292, 46 },
		{ 2292, 46 },
		{ 2292, 46 },
		{ 2292, 46 },
		{ 2292, 46 },
		{ 2559, 2531 },
		{ 1616, 1600 },
		{ 1618, 1602 },
		{ 1619, 1603 },
		{ 1613, 1599 },
		{ 1597, 1569 },
		{ 1622, 1612 },
		{ 2294, 46 },
		{ 2290, 46 },
		{ 2285, 46 },
		{ 2294, 46 },
		{ 2282, 46 },
		{ 2289, 46 },
		{ 2287, 46 },
		{ 2294, 46 },
		{ 2291, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2284, 46 },
		{ 2279, 46 },
		{ 2286, 46 },
		{ 2281, 46 },
		{ 2294, 46 },
		{ 2288, 46 },
		{ 2283, 46 },
		{ 2280, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 1615, 1599 },
		{ 2298, 46 },
		{ 1624, 1612 },
		{ 1620, 1604 },
		{ 2294, 46 },
		{ 1621, 1605 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2295, 46 },
		{ 2296, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2297, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 2294, 46 },
		{ 161, 4 },
		{ 162, 4 },
		{ 2573, 2542 },
		{ 2573, 2542 },
		{ 2331, 2302 },
		{ 2331, 2302 },
		{ 2355, 2323 },
		{ 2355, 2323 },
		{ 2378, 2346 },
		{ 2378, 2346 },
		{ 2379, 2347 },
		{ 2379, 2347 },
		{ 2395, 2364 },
		{ 2395, 2364 },
		{ 1446, 1420 },
		{ 1452, 1425 },
		{ 1627, 1618 },
		{ 1628, 1619 },
		{ 1451, 1425 },
		{ 1629, 1620 },
		{ 1630, 1621 },
		{ 1455, 1430 },
		{ 1445, 1420 },
		{ 161, 4 },
		{ 1634, 1628 },
		{ 2573, 2542 },
		{ 1456, 1431 },
		{ 2331, 2302 },
		{ 1636, 1630 },
		{ 2355, 2323 },
		{ 1639, 1634 },
		{ 2378, 2346 },
		{ 1640, 1636 },
		{ 2379, 2347 },
		{ 1444, 1420 },
		{ 2395, 2364 },
		{ 1642, 1639 },
		{ 86, 4 },
		{ 160, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 1643, 1640 },
		{ 1644, 1642 },
		{ 0, 2542 },
		{ 1645, 1643 },
		{ 1440, 1644 },
		{ 1457, 1432 },
		{ 1458, 1434 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 76, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 84, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 1459, 1435 },
		{ 83, 4 },
		{ 1462, 1439 },
		{ 1463, 1442 },
		{ 87, 4 },
		{ 1464, 1443 },
		{ 87, 4 },
		{ 75, 4 },
		{ 81, 4 },
		{ 79, 4 },
		{ 87, 4 },
		{ 80, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 78, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 82, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 77, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 87, 4 },
		{ 3105, 3104 },
		{ 1465, 1444 },
		{ 1466, 1445 },
		{ 3104, 3104 },
		{ 1467, 1446 },
		{ 1468, 1447 },
		{ 1469, 1448 },
		{ 1472, 1450 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 1473, 1451 },
		{ 3104, 3104 },
		{ 1474, 1452 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 1470, 1449 },
		{ 1477, 1455 },
		{ 1478, 1456 },
		{ 1479, 1457 },
		{ 1471, 1449 },
		{ 1480, 1458 },
		{ 1481, 1459 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 1484, 1462 },
		{ 1485, 1463 },
		{ 1486, 1464 },
		{ 1487, 1465 },
		{ 1488, 1466 },
		{ 1489, 1467 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 3104, 3104 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1490, 1468 },
		{ 1491, 1469 },
		{ 1492, 1470 },
		{ 1493, 1471 },
		{ 1494, 1472 },
		{ 1495, 1473 },
		{ 1496, 1474 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1499, 1477 },
		{ 1500, 1478 },
		{ 1501, 1479 },
		{ 156, 154 },
		{ 1440, 1646 },
		{ 108, 93 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 1440, 1646 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 109, 94 },
		{ 110, 95 },
		{ 111, 96 },
		{ 112, 97 },
		{ 113, 98 },
		{ 114, 99 },
		{ 116, 101 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 108 },
		{ 125, 109 },
		{ 2294, 2567 },
		{ 126, 111 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 2294, 2567 },
		{ 1441, 1645 },
		{ 0, 1645 },
		{ 127, 112 },
		{ 128, 113 },
		{ 129, 114 },
		{ 131, 116 },
		{ 136, 122 },
		{ 137, 123 },
		{ 138, 124 },
		{ 139, 125 },
		{ 140, 126 },
		{ 141, 127 },
		{ 142, 129 },
		{ 143, 131 },
		{ 2756, 2756 },
		{ 144, 136 },
		{ 145, 137 },
		{ 146, 138 },
		{ 2301, 2279 },
		{ 2303, 2280 },
		{ 2307, 2282 },
		{ 2306, 2281 },
		{ 2310, 2283 },
		{ 1441, 1645 },
		{ 2304, 2281 },
		{ 2317, 2285 },
		{ 2309, 2283 },
		{ 2305, 2281 },
		{ 2315, 2284 },
		{ 2318, 2286 },
		{ 2308, 2282 },
		{ 2319, 2287 },
		{ 2320, 2288 },
		{ 2321, 2289 },
		{ 2322, 2290 },
		{ 2323, 2291 },
		{ 2294, 2294 },
		{ 2756, 2756 },
		{ 2316, 2295 },
		{ 2302, 2296 },
		{ 2314, 2284 },
		{ 2311, 2283 },
		{ 2312, 2283 },
		{ 2313, 2297 },
		{ 2330, 2301 },
		{ 147, 139 },
		{ 2327, 2295 },
		{ 2332, 2303 },
		{ 2333, 2304 },
		{ 2334, 2305 },
		{ 2335, 2306 },
		{ 2336, 2307 },
		{ 0, 1645 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2337, 2308 },
		{ 2340, 2310 },
		{ 2341, 2311 },
		{ 2342, 2312 },
		{ 2343, 2313 },
		{ 2344, 2314 },
		{ 2345, 2315 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 69, 7 },
		{ 2347, 2316 },
		{ 2348, 2317 },
		{ 2349, 2318 },
		{ 2756, 2756 },
		{ 1646, 1645 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2756, 2756 },
		{ 2350, 2319 },
		{ 2353, 2321 },
		{ 2354, 2322 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 2338, 2309 },
		{ 148, 140 },
		{ 2346, 2327 },
		{ 2362, 2330 },
		{ 2364, 2332 },
		{ 2365, 2333 },
		{ 2339, 2309 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 2366, 2334 },
		{ 2367, 2335 },
		{ 2368, 2336 },
		{ 2369, 2337 },
		{ 1254, 7 },
		{ 2370, 2338 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 1254, 7 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 2371, 2339 },
		{ 2372, 2340 },
		{ 2373, 2341 },
		{ 2374, 2342 },
		{ 2375, 2343 },
		{ 2376, 2344 },
		{ 2377, 2345 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 149, 142 },
		{ 2380, 2348 },
		{ 2381, 2349 },
		{ 2382, 2350 },
		{ 0, 1917 },
		{ 2383, 2351 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 1917 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 2384, 2352 },
		{ 2385, 2353 },
		{ 2386, 2354 },
		{ 2393, 2362 },
		{ 150, 143 },
		{ 2396, 2365 },
		{ 2397, 2366 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 151, 144 },
		{ 2399, 2368 },
		{ 2401, 2369 },
		{ 2402, 2370 },
		{ 0, 2111 },
		{ 2400, 2368 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 0, 2111 },
		{ 2351, 2320 },
		{ 2403, 2371 },
		{ 2405, 2373 },
		{ 2406, 2374 },
		{ 152, 146 },
		{ 2408, 2376 },
		{ 2409, 2377 },
		{ 2413, 2380 },
		{ 2414, 2381 },
		{ 2352, 2320 },
		{ 2415, 2382 },
		{ 2416, 2383 },
		{ 2417, 2384 },
		{ 2418, 2385 },
		{ 2419, 2386 },
		{ 2421, 2393 },
		{ 2424, 2396 },
		{ 2429, 2400 },
		{ 2430, 2401 },
		{ 2431, 2402 },
		{ 2432, 2403 },
		{ 153, 149 },
		{ 2435, 2406 },
		{ 2437, 2408 },
		{ 2438, 2409 },
		{ 2442, 2413 },
		{ 2443, 2414 },
		{ 2444, 2415 },
		{ 2445, 2416 },
		{ 2446, 2417 },
		{ 154, 150 },
		{ 2448, 2419 },
		{ 2454, 2424 },
		{ 2457, 2429 },
		{ 2458, 2430 },
		{ 2460, 2431 },
		{ 155, 152 },
		{ 2464, 2435 },
		{ 2466, 2437 },
		{ 2459, 2431 },
		{ 2467, 2438 },
		{ 2471, 2442 },
		{ 2472, 2443 },
		{ 2473, 2444 },
		{ 91, 75 },
		{ 2477, 2448 },
		{ 2483, 2454 },
		{ 157, 155 },
		{ 2488, 2458 },
		{ 2489, 2459 },
		{ 2490, 2460 },
		{ 2494, 2464 },
		{ 2496, 2466 },
		{ 2497, 2467 },
		{ 2501, 2471 },
		{ 2502, 2472 },
		{ 2503, 2473 },
		{ 2508, 2477 },
		{ 2514, 2483 },
		{ 2517, 2488 },
		{ 2518, 2489 },
		{ 158, 157 },
		{ 2523, 2494 },
		{ 159, 158 },
		{ 2526, 2497 },
		{ 2530, 2501 },
		{ 2531, 2502 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 2537, 2508 },
		{ 2542, 2514 },
		{ 94, 77 },
		{ 95, 78 },
		{ 96, 79 },
		{ 97, 80 },
		{ 98, 81 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 99, 82 },
		{ 101, 84 },
		{ 106, 91 },
		{ 107, 92 },
		{ 87, 159 },
		{ 0, 2928 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 87, 159 },
		{ 92, 76 },
		{ 1264, 1261 },
		{ 132, 117 },
		{ 1264, 1261 },
		{ 132, 117 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 2388, 2356 },
		{ 2390, 2359 },
		{ 2388, 2356 },
		{ 2390, 2359 },
		{ 2927, 51 },
		{ 2648, 2622 },
		{ 93, 76 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1267, 1263 },
		{ 2357, 2325 },
		{ 1267, 1263 },
		{ 2357, 2325 },
		{ 1254, 1254 },
		{ 2659, 2633 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 1254, 1254 },
		{ 0, 2537 },
		{ 0, 2537 },
		{ 1269, 1266 },
		{ 134, 120 },
		{ 1269, 1266 },
		{ 134, 120 },
		{ 2810, 2795 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 118, 103 },
		{ 2813, 2798 },
		{ 118, 103 },
		{ 2217, 2214 },
		{ 1835, 1834 },
		{ 1319, 1318 },
		{ 0, 2537 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 1793, 1792 },
		{ 88, 51 },
		{ 2794, 2778 },
		{ 2623, 2592 },
		{ 3094, 3094 },
		{ 1790, 1789 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 3094, 3094 },
		{ 1868, 1844 },
		{ 1316, 1315 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 1700, 1699 },
		{ 1867, 1844 },
		{ 3074, 3073 },
		{ 1745, 1744 },
		{ 3171, 3170 },
		{ 2960, 2959 },
		{ 2567, 2537 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 2127, 2110 },
		{ 2693, 2667 },
		{ 3013, 3012 },
		{ 1893, 1874 },
		{ 0, 1504 },
		{ 1915, 1896 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 0, 1504 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3054, 3053 },
		{ 1806, 1805 },
		{ 3077, 3076 },
		{ 3085, 3084 },
		{ 2257, 2256 },
		{ 2521, 2492 },
		{ 2262, 2261 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 1748, 1747 },
		{ 1721, 1720 },
		{ 3159, 3154 },
		{ 2874, 2873 },
		{ 2054, 2032 },
		{ 3143, 3138 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 2914, 2913 },
		{ 2919, 2918 },
		{ 2624, 2593 },
		{ 1332, 1331 },
		{ 2084, 2063 },
		{ 2566, 2536 },
		{ 2113, 2095 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3174, 3173 },
		{ 3190, 3187 },
		{ 3164, 3161 },
		{ 3196, 3193 },
		{ 2712, 2687 },
		{ 2716, 2691 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3161, 3161 },
		{ 3160, 3155 },
		{ 3160, 3155 },
		{ 3160, 3155 },
		{ 3160, 3155 },
		{ 3160, 3155 },
		{ 3160, 3155 },
		{ 3160, 3155 },
		{ 3160, 3155 },
		{ 3160, 3155 },
		{ 3160, 3155 },
		{ 3153, 3149 },
		{ 3153, 3149 },
		{ 3153, 3149 },
		{ 3153, 3149 },
		{ 3153, 3149 },
		{ 3153, 3149 },
		{ 3153, 3149 },
		{ 3153, 3149 },
		{ 3153, 3149 },
		{ 3153, 3149 },
		{ 184, 175 },
		{ 195, 175 },
		{ 186, 175 },
		{ 1379, 1378 },
		{ 181, 175 },
		{ 191, 175 },
		{ 185, 175 },
		{ 2069, 2045 },
		{ 183, 175 },
		{ 1674, 1673 },
		{ 1379, 1378 },
		{ 2726, 2702 },
		{ 193, 175 },
		{ 192, 175 },
		{ 182, 175 },
		{ 190, 175 },
		{ 1674, 1673 },
		{ 187, 175 },
		{ 189, 175 },
		{ 180, 175 },
		{ 2003, 2001 },
		{ 2732, 2708 },
		{ 188, 175 },
		{ 194, 175 },
		{ 3113, 65 },
		{ 0, 3155 },
		{ 2745, 2725 },
		{ 69, 65 },
		{ 2068, 2045 },
		{ 2747, 2727 },
		{ 2762, 2742 },
		{ 2763, 2743 },
		{ 1750, 1749 },
		{ 1772, 1771 },
		{ 1260, 1257 },
		{ 0, 3149 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 3153, 3153 },
		{ 3153, 3153 },
		{ 3153, 3153 },
		{ 3153, 3153 },
		{ 3153, 3153 },
		{ 3153, 3153 },
		{ 3153, 3153 },
		{ 3153, 3153 },
		{ 3153, 3153 },
		{ 3153, 3153 },
		{ 2324, 2292 },
		{ 1261, 1257 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2773, 2753 },
		{ 468, 421 },
		{ 473, 421 },
		{ 470, 421 },
		{ 469, 421 },
		{ 472, 421 },
		{ 467, 421 },
		{ 1785, 1784 },
		{ 466, 421 },
		{ 2778, 2760 },
		{ 2788, 2771 },
		{ 2325, 2292 },
		{ 471, 421 },
		{ 1320, 1319 },
		{ 474, 421 },
		{ 3112, 65 },
		{ 2795, 2779 },
		{ 2798, 2782 },
		{ 2541, 2513 },
		{ 3111, 65 },
		{ 465, 421 },
		{ 1261, 1257 },
		{ 102, 85 },
		{ 3158, 3153 },
		{ 85, 85 },
		{ 85, 85 },
		{ 85, 85 },
		{ 85, 85 },
		{ 85, 85 },
		{ 85, 85 },
		{ 85, 85 },
		{ 85, 85 },
		{ 85, 85 },
		{ 85, 85 },
		{ 1669, 1668 },
		{ 2816, 2801 },
		{ 2830, 2824 },
		{ 2832, 2826 },
		{ 2838, 2833 },
		{ 2844, 2843 },
		{ 2543, 2515 },
		{ 1794, 1793 },
		{ 2877, 2876 },
		{ 2325, 2292 },
		{ 2886, 2885 },
		{ 103, 85 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2899, 2898 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2356, 2324 },
		{ 2125, 2108 },
		{ 3109, 65 },
		{ 2126, 2109 },
		{ 3110, 65 },
		{ 1397, 1396 },
		{ 2922, 2921 },
		{ 2484, 2455 },
		{ 1809, 1808 },
		{ 2144, 2131 },
		{ 103, 85 },
		{ 2359, 2326 },
		{ 104, 104 },
		{ 104, 104 },
		{ 104, 104 },
		{ 104, 104 },
		{ 104, 104 },
		{ 104, 104 },
		{ 104, 104 },
		{ 104, 104 },
		{ 104, 104 },
		{ 104, 104 },
		{ 3160, 3160 },
		{ 3160, 3160 },
		{ 3160, 3160 },
		{ 3160, 3160 },
		{ 3160, 3160 },
		{ 3160, 3160 },
		{ 3160, 3160 },
		{ 3160, 3160 },
		{ 3160, 3160 },
		{ 3160, 3160 },
		{ 2356, 2324 },
		{ 120, 104 },
		{ 2157, 2142 },
		{ 2158, 2143 },
		{ 2950, 2949 },
		{ 1408, 1407 },
		{ 2977, 2976 },
		{ 2218, 2215 },
		{ 2233, 2232 },
		{ 3007, 3006 },
		{ 1286, 1285 },
		{ 2359, 2326 },
		{ 3016, 3015 },
		{ 102, 102 },
		{ 102, 102 },
		{ 102, 102 },
		{ 102, 102 },
		{ 102, 102 },
		{ 102, 102 },
		{ 102, 102 },
		{ 102, 102 },
		{ 102, 102 },
		{ 102, 102 },
		{ 1260, 1260 },
		{ 1260, 1260 },
		{ 1260, 1260 },
		{ 1260, 1260 },
		{ 1260, 1260 },
		{ 1260, 1260 },
		{ 1260, 1260 },
		{ 1260, 1260 },
		{ 1260, 1260 },
		{ 1260, 1260 },
		{ 120, 104 },
		{ 117, 102 },
		{ 3163, 3160 },
		{ 3024, 3023 },
		{ 3128, 67 },
		{ 1256, 9 },
		{ 2498, 2468 },
		{ 69, 67 },
		{ 2499, 2469 },
		{ 69, 9 },
		{ 3048, 3047 },
		{ 1263, 1260 },
		{ 2579, 2548 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 3057, 3056 },
		{ 3068, 3067 },
		{ 1722, 1721 },
		{ 2259, 2258 },
		{ 3079, 3078 },
		{ 1256, 9 },
		{ 1724, 1723 },
		{ 3088, 3087 },
		{ 1896, 1877 },
		{ 2271, 2270 },
		{ 117, 102 },
		{ 1266, 1262 },
		{ 2512, 2481 },
		{ 2611, 2580 },
		{ 1902, 1883 },
		{ 2516, 2485 },
		{ 2456, 2426 },
		{ 3138, 3133 },
		{ 2631, 2602 },
		{ 1258, 9 },
		{ 1263, 1260 },
		{ 1257, 9 },
		{ 1257, 9 },
		{ 1257, 9 },
		{ 1257, 9 },
		{ 1257, 9 },
		{ 1257, 9 },
		{ 1257, 9 },
		{ 1257, 9 },
		{ 1257, 9 },
		{ 1257, 9 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 2379, 2379 },
		{ 2379, 2379 },
		{ 1266, 1262 },
		{ 3125, 67 },
		{ 2527, 2527 },
		{ 2527, 2527 },
		{ 2551, 2551 },
		{ 2551, 2551 },
		{ 2646, 2620 },
		{ 3126, 67 },
		{ 2474, 2474 },
		{ 2474, 2474 },
		{ 1335, 1334 },
		{ 2978, 2978 },
		{ 2978, 2978 },
		{ 3025, 3025 },
		{ 3025, 3025 },
		{ 2528, 2528 },
		{ 2528, 2528 },
		{ 2746, 2746 },
		{ 2746, 2746 },
		{ 2652, 2626 },
		{ 3123, 67 },
		{ 2379, 2379 },
		{ 2538, 2538 },
		{ 2538, 2538 },
		{ 1922, 1903 },
		{ 2527, 2527 },
		{ 2662, 2636 },
		{ 2551, 2551 },
		{ 2425, 2425 },
		{ 2425, 2425 },
		{ 1374, 1373 },
		{ 2474, 2474 },
		{ 1949, 1933 },
		{ 3122, 3121 },
		{ 2978, 2978 },
		{ 1951, 1936 },
		{ 3025, 3025 },
		{ 3176, 3175 },
		{ 2528, 2528 },
		{ 2701, 2675 },
		{ 2746, 2746 },
		{ 1953, 1938 },
		{ 3127, 67 },
		{ 2559, 2559 },
		{ 2559, 2559 },
		{ 2538, 2538 },
		{ 2398, 2398 },
		{ 2398, 2398 },
		{ 2573, 2573 },
		{ 2573, 2573 },
		{ 3203, 3201 },
		{ 2425, 2425 },
		{ 2594, 2594 },
		{ 2594, 2594 },
		{ 2649, 2649 },
		{ 2649, 2649 },
		{ 2447, 2447 },
		{ 2447, 2447 },
		{ 2910, 2910 },
		{ 2910, 2910 },
		{ 1353, 1352 },
		{ 1632, 1626 },
		{ 2938, 2938 },
		{ 2938, 2938 },
		{ 2525, 2525 },
		{ 2525, 2525 },
		{ 2559, 2559 },
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2398, 2398 },
		{ 2848, 2847 },
		{ 2573, 2573 },
		{ 2404, 2404 },
		{ 2404, 2404 },
		{ 1662, 1661 },
		{ 2594, 2594 },
		{ 2869, 2868 },
		{ 2649, 2649 },
		{ 1802, 1801 },
		{ 2447, 2447 },
		{ 2064, 2039 },
		{ 2910, 2910 },
		{ 1663, 1662 },
		{ 2063, 2039 },
		{ 2881, 2880 },
		{ 2938, 2938 },
		{ 1875, 1850 },
		{ 2525, 2525 },
		{ 2592, 2560 },
		{ 1874, 1850 },
		{ 2264, 2264 },
		{ 2532, 2532 },
		{ 2532, 2532 },
		{ 2596, 2564 },
		{ 2411, 2379 },
		{ 2404, 2404 },
		{ 2461, 2461 },
		{ 2461, 2461 },
		{ 2554, 2554 },
		{ 2554, 2554 },
		{ 1367, 1366 },
		{ 2604, 2571 },
		{ 2558, 2558 },
		{ 2558, 2558 },
		{ 1825, 1824 },
		{ 2412, 2379 },
		{ 2583, 2551 },
		{ 2582, 2551 },
		{ 1826, 1825 },
		{ 2555, 2527 },
		{ 2505, 2474 },
		{ 2504, 2474 },
		{ 2495, 2495 },
		{ 2495, 2495 },
		{ 2532, 2532 },
		{ 3178, 3178 },
		{ 2568, 2538 },
		{ 1831, 1830 },
		{ 2979, 2978 },
		{ 2461, 2461 },
		{ 3026, 3025 },
		{ 2554, 2554 },
		{ 2556, 2528 },
		{ 2766, 2746 },
		{ 2569, 2538 },
		{ 2558, 2558 },
		{ 1368, 1367 },
		{ 2407, 2407 },
		{ 2407, 2407 },
		{ 1690, 1689 },
		{ 2939, 2938 },
		{ 2933, 2931 },
		{ 2749, 2749 },
		{ 2749, 2749 },
		{ 2455, 2425 },
		{ 2495, 2495 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 3178, 3178 },
		{ 1691, 1690 },
		{ 2632, 2604 },
		{ 1697, 1696 },
		{ 2947, 2947 },
		{ 2947, 2947 },
		{ 2954, 2953 },
		{ 2450, 2450 },
		{ 2450, 2450 },
		{ 1698, 1697 },
		{ 1894, 1875 },
		{ 2591, 2559 },
		{ 2407, 2407 },
		{ 1296, 1295 },
		{ 2426, 2398 },
		{ 2274, 2273 },
		{ 2605, 2573 },
		{ 2749, 2749 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2625, 2594 },
		{ 2331, 2331 },
		{ 2675, 2649 },
		{ 2476, 2447 },
		{ 2939, 2938 },
		{ 2911, 2910 },
		{ 2562, 2532 },
		{ 2947, 2947 },
		{ 2666, 2640 },
		{ 2553, 2525 },
		{ 2450, 2450 },
		{ 1901, 1882 },
		{ 2265, 2264 },
		{ 2546, 2546 },
		{ 2546, 2546 },
		{ 3033, 3033 },
		{ 3033, 3033 },
		{ 2433, 2404 },
		{ 2678, 2652 },
		{ 2655, 2655 },
		{ 2655, 2655 },
		{ 2992, 2992 },
		{ 1716, 1715 },
		{ 2560, 2532 },
		{ 1913, 1894 },
		{ 2704, 2704 },
		{ 2704, 2704 },
		{ 2561, 2532 },
		{ 2680, 2680 },
		{ 2680, 2680 },
		{ 2395, 2395 },
		{ 2395, 2395 },
		{ 1717, 1716 },
		{ 1328, 1327 },
		{ 2535, 2506 },
		{ 1275, 1274 },
		{ 2546, 2546 },
		{ 3067, 3066 },
		{ 3033, 3033 },
		{ 2539, 2511 },
		{ 1403, 1402 },
		{ 2491, 2461 },
		{ 2655, 2655 },
		{ 2586, 2554 },
		{ 1740, 1739 },
		{ 3181, 3178 },
		{ 2733, 2709 },
		{ 2590, 2558 },
		{ 2704, 2704 },
		{ 2734, 2710 },
		{ 3180, 3178 },
		{ 2680, 2680 },
		{ 3179, 3178 },
		{ 2395, 2395 },
		{ 2736, 2713 },
		{ 2737, 2716 },
		{ 1741, 1740 },
		{ 2524, 2495 },
		{ 2544, 2516 },
		{ 1964, 1951 },
		{ 1983, 1973 },
		{ 2765, 2745 },
		{ 1991, 1983 },
		{ 1290, 1289 },
		{ 1626, 1617 },
		{ 2774, 2754 },
		{ 1343, 1342 },
		{ 1766, 1765 },
		{ 1767, 1766 },
		{ 1350, 1349 },
		{ 1776, 1775 },
		{ 2436, 2407 },
		{ 3166, 3165 },
		{ 3167, 3166 },
		{ 2085, 2064 },
		{ 2805, 2789 },
		{ 2769, 2749 },
		{ 2104, 2083 },
		{ 2106, 2085 },
		{ 2109, 2088 },
		{ 2363, 2331 },
		{ 2574, 2543 },
		{ 1351, 1350 },
		{ 2661, 2635 },
		{ 2548, 2520 },
		{ 2493, 2463 },
		{ 2948, 2947 },
		{ 2897, 2896 },
		{ 2552, 2524 },
		{ 2479, 2450 },
		{ 2908, 2907 },
		{ 2677, 2651 },
		{ 1752, 1751 },
		{ 1998, 1993 },
		{ 2682, 2656 },
		{ 1318, 1317 },
		{ 1277, 1276 },
		{ 2261, 2260 },
		{ 1702, 1701 },
		{ 2937, 2935 },
		{ 2993, 2992 },
		{ 2713, 2688 },
		{ 1886, 1867 },
		{ 2717, 2692 },
		{ 2563, 2533 },
		{ 2267, 2266 },
		{ 1345, 1344 },
		{ 2422, 2422 },
		{ 2422, 2422 },
		{ 2273, 2272 },
		{ 3005, 3004 },
		{ 1778, 1777 },
		{ 2075, 2054 },
		{ 1298, 1297 },
		{ 1897, 1878 },
		{ 2577, 2546 },
		{ 2751, 2731 },
		{ 3034, 3033 },
		{ 2755, 2735 },
		{ 3046, 3045 },
		{ 2585, 2553 },
		{ 2681, 2655 },
		{ 2096, 2075 },
		{ 2587, 2555 },
		{ 2588, 2556 },
		{ 1787, 1786 },
		{ 1381, 1380 },
		{ 2728, 2704 },
		{ 2706, 2680 },
		{ 2423, 2395 },
		{ 2422, 2422 },
		{ 1905, 1886 },
		{ 1792, 1791 },
		{ 2777, 2759 },
		{ 1395, 1394 },
		{ 2785, 2768 },
		{ 1312, 1311 },
		{ 1932, 1915 },
		{ 1726, 1725 },
		{ 2617, 2586 },
		{ 2800, 2784 },
		{ 2621, 2590 },
		{ 1676, 1675 },
		{ 1399, 1398 },
		{ 3147, 3143 },
		{ 1284, 1283 },
		{ 2817, 2802 },
		{ 2818, 2804 },
		{ 2211, 2204 },
		{ 2831, 2825 },
		{ 2214, 2209 },
		{ 2639, 2613 },
		{ 2839, 2837 },
		{ 2842, 2840 },
		{ 1695, 1694 },
		{ 1405, 1404 },
		{ 3178, 3177 },
		{ 2441, 2412 },
		{ 3186, 3183 },
		{ 2653, 2627 },
		{ 3194, 3191 },
		{ 2871, 2870 },
		{ 2230, 2227 },
		{ 3206, 3205 },
		{ 2378, 2378 },
		{ 2378, 2378 },
		{ 2828, 2828 },
		{ 2828, 2828 },
		{ 1936, 1921 },
		{ 2451, 2422 },
		{ 2475, 2475 },
		{ 2475, 2475 },
		{ 2608, 2577 },
		{ 1393, 1392 },
		{ 2095, 2074 },
		{ 2692, 2666 },
		{ 2650, 2624 },
		{ 2786, 2769 },
		{ 2742, 2721 },
		{ 2743, 2722 },
		{ 2949, 2948 },
		{ 2620, 2589 },
		{ 1771, 1770 },
		{ 2748, 2728 },
		{ 2707, 2681 },
		{ 1832, 1831 },
		{ 2492, 2462 },
		{ 2378, 2378 },
		{ 2602, 2569 },
		{ 2828, 2828 },
		{ 2627, 2596 },
		{ 1407, 1406 },
		{ 2824, 2815 },
		{ 2475, 2475 },
		{ 2727, 2703 },
		{ 1864, 1841 },
		{ 2522, 2493 },
		{ 2093, 2072 },
		{ 2924, 2923 },
		{ 1392, 1391 },
		{ 2452, 2422 },
		{ 1734, 1733 },
		{ 2607, 2576 },
		{ 2462, 2433 },
		{ 1288, 1287 },
		{ 1863, 1841 },
		{ 2616, 2585 },
		{ 1482, 1460 },
		{ 1696, 1695 },
		{ 1311, 1310 },
		{ 2952, 2951 },
		{ 2124, 2107 },
		{ 2959, 2958 },
		{ 1931, 1914 },
		{ 1811, 1810 },
		{ 2772, 2752 },
		{ 1819, 1818 },
		{ 1361, 1360 },
		{ 2139, 2123 },
		{ 3009, 3008 },
		{ 1946, 1930 },
		{ 2643, 2617 },
		{ 3018, 3017 },
		{ 1751, 1750 },
		{ 2647, 2621 },
		{ 1699, 1698 },
		{ 2189, 2171 },
		{ 2190, 2172 },
		{ 3050, 3049 },
		{ 1760, 1759 },
		{ 1656, 1655 },
		{ 3059, 3058 },
		{ 2804, 2788 },
		{ 2216, 2213 },
		{ 2806, 2790 },
		{ 1410, 1409 },
		{ 2663, 2637 },
		{ 3081, 3080 },
		{ 1770, 1769 },
		{ 2224, 2221 },
		{ 3090, 3089 },
		{ 2227, 2225 },
		{ 1710, 1709 },
		{ 2825, 2816 },
		{ 2000, 1997 },
		{ 1398, 1397 },
		{ 2688, 2662 },
		{ 2006, 2005 },
		{ 2837, 2832 },
		{ 1879, 1856 },
		{ 3144, 3139 },
		{ 1517, 1497 },
		{ 2702, 2676 },
		{ 1337, 1336 },
		{ 2846, 2845 },
		{ 1460, 1436 },
		{ 1684, 1683 },
		{ 2584, 2552 },
		{ 2070, 2047 },
		{ 2074, 2053 },
		{ 2410, 2378 },
		{ 1725, 1724 },
		{ 2833, 2828 },
		{ 2879, 2878 },
		{ 1898, 1879 },
		{ 3177, 3176 },
		{ 2506, 2475 },
		{ 1900, 1881 },
		{ 2730, 2706 },
		{ 3183, 3180 },
		{ 2089, 2068 },
		{ 2901, 2900 },
		{ 2907, 2906 },
		{ 2520, 2491 },
		{ 2091, 2070 },
		{ 3205, 3203 },
		{ 2735, 2712 },
		{ 2892, 2892 },
		{ 2892, 2892 },
		{ 3041, 3041 },
		{ 3041, 3041 },
		{ 2519, 2519 },
		{ 2519, 2519 },
		{ 2545, 2545 },
		{ 2545, 2545 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 1933, 1916 },
		{ 2360, 2328 },
		{ 1683, 1682 },
		{ 2953, 2952 },
		{ 1333, 1332 },
		{ 1938, 1923 },
		{ 2108, 2087 },
		{ 2961, 2960 },
		{ 2970, 2969 },
		{ 1323, 1322 },
		{ 2789, 2772 },
		{ 2987, 2986 },
		{ 2988, 2987 },
		{ 2892, 2892 },
		{ 2990, 2989 },
		{ 3041, 3041 },
		{ 1948, 1932 },
		{ 2519, 2519 },
		{ 1461, 1438 },
		{ 2545, 2545 },
		{ 3003, 3002 },
		{ 3000, 3000 },
		{ 1733, 1732 },
		{ 1411, 1410 },
		{ 1372, 1371 },
		{ 3010, 3009 },
		{ 1972, 1960 },
		{ 3014, 3013 },
		{ 2131, 2114 },
		{ 1881, 1859 },
		{ 3019, 3018 },
		{ 1985, 1976 },
		{ 1885, 1866 },
		{ 3031, 3030 },
		{ 1997, 1992 },
		{ 2169, 2155 },
		{ 3044, 3043 },
		{ 2820, 2806 },
		{ 1497, 1475 },
		{ 2689, 2663 },
		{ 3051, 3050 },
		{ 2691, 2665 },
		{ 3055, 3054 },
		{ 1655, 1654 },
		{ 2191, 2173 },
		{ 3060, 3059 },
		{ 3066, 3065 },
		{ 2202, 2189 },
		{ 2439, 2410 },
		{ 2705, 2679 },
		{ 2440, 2411 },
		{ 2203, 2190 },
		{ 1289, 1288 },
		{ 3082, 3081 },
		{ 2213, 2208 },
		{ 3086, 3085 },
		{ 2847, 2846 },
		{ 1895, 1876 },
		{ 3091, 3090 },
		{ 2007, 2006 },
		{ 1797, 1796 },
		{ 1338, 1337 },
		{ 3106, 3102 },
		{ 2593, 2561 },
		{ 2875, 2874 },
		{ 1360, 1359 },
		{ 2226, 2224 },
		{ 3135, 3130 },
		{ 2880, 2879 },
		{ 3139, 3134 },
		{ 1807, 1806 },
		{ 1709, 1708 },
		{ 1759, 1758 },
		{ 3151, 3147 },
		{ 2895, 2894 },
		{ 2072, 2050 },
		{ 2073, 2052 },
		{ 2609, 2578 },
		{ 2902, 2901 },
		{ 1903, 1884 },
		{ 1812, 1811 },
		{ 1818, 1817 },
		{ 2263, 2262 },
		{ 1667, 1666 },
		{ 2893, 2892 },
		{ 2920, 2919 },
		{ 3042, 3041 },
		{ 2086, 2065 },
		{ 2547, 2519 },
		{ 2758, 2738 },
		{ 2576, 2545 },
		{ 2925, 2924 },
		{ 3001, 3000 },
		{ 2540, 2512 },
		{ 1390, 1389 },
		{ 2626, 2595 },
		{ 1537, 1518 },
		{ 2934, 2932 },
		{ 2628, 2597 },
		{ 2630, 2601 },
		{ 115, 100 },
		{ 3011, 3011 },
		{ 3011, 3011 },
		{ 3083, 3083 },
		{ 3083, 3083 },
		{ 2434, 2434 },
		{ 2434, 2434 },
		{ 2872, 2872 },
		{ 2872, 2872 },
		{ 2581, 2581 },
		{ 2581, 2581 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 2783, 2783 },
		{ 2783, 2783 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 3052, 3052 },
		{ 3052, 3052 },
		{ 1804, 1804 },
		{ 1804, 1804 },
		{ 1330, 1330 },
		{ 1330, 1330 },
		{ 2140, 2124 },
		{ 3011, 3011 },
		{ 2219, 2216 },
		{ 3083, 3083 },
		{ 1536, 1517 },
		{ 2434, 2434 },
		{ 1711, 1710 },
		{ 2872, 2872 },
		{ 1761, 1760 },
		{ 2581, 2581 },
		{ 2482, 2453 },
		{ 2355, 2355 },
		{ 2683, 2657 },
		{ 2783, 2783 },
		{ 1685, 1684 },
		{ 2917, 2917 },
		{ 1947, 1931 },
		{ 3052, 3052 },
		{ 2235, 2234 },
		{ 1804, 1804 },
		{ 2112, 2093 },
		{ 1330, 1330 },
		{ 2002, 2000 },
		{ 1657, 1656 },
		{ 1720, 1719 },
		{ 1919, 1900 },
		{ 1362, 1361 },
		{ 1735, 1734 },
		{ 3193, 3190 },
		{ 2550, 2522 },
		{ 1774, 1773 },
		{ 2269, 2268 },
		{ 3148, 3144 },
		{ 1820, 1819 },
		{ 3029, 3029 },
		{ 3029, 3029 },
		{ 1325, 1325 },
		{ 1325, 1325 },
		{ 1799, 1799 },
		{ 1799, 1799 },
		{ 3072, 3072 },
		{ 3072, 3072 },
		{ 3036, 3036 },
		{ 3036, 3036 },
		{ 1788, 1788 },
		{ 1788, 1788 },
		{ 2995, 2995 },
		{ 2995, 2995 },
		{ 1764, 1763 },
		{ 2887, 2887 },
		{ 2887, 2887 },
		{ 1314, 1314 },
		{ 1314, 1314 },
		{ 1714, 1713 },
		{ 1738, 1737 },
		{ 1282, 1281 },
		{ 1833, 1832 },
		{ 3029, 3029 },
		{ 2130, 2113 },
		{ 1325, 1325 },
		{ 1801, 1800 },
		{ 1799, 1799 },
		{ 2258, 2257 },
		{ 3072, 3072 },
		{ 2660, 2634 },
		{ 3036, 3036 },
		{ 1963, 1950 },
		{ 1788, 1788 },
		{ 2740, 2719 },
		{ 2995, 2995 },
		{ 2549, 2521 },
		{ 3012, 3011 },
		{ 2887, 2887 },
		{ 3084, 3083 },
		{ 1314, 1314 },
		{ 2463, 2434 },
		{ 2823, 2814 },
		{ 2873, 2872 },
		{ 1672, 1671 },
		{ 2612, 2581 },
		{ 2664, 2638 },
		{ 2387, 2355 },
		{ 2083, 2062 },
		{ 2799, 2783 },
		{ 1348, 1347 },
		{ 2918, 2917 },
		{ 2676, 2650 },
		{ 3053, 3052 },
		{ 1975, 1965 },
		{ 1805, 1804 },
		{ 1906, 1887 },
		{ 1331, 1330 },
		{ 2761, 2741 },
		{ 2188, 2170 },
		{ 2945, 2944 },
		{ 3162, 3157 },
		{ 1747, 1746 },
		{ 2764, 2744 },
		{ 1365, 1364 },
		{ 1917, 1898 },
		{ 1601, 1575 },
		{ 3173, 3172 },
		{ 1660, 1659 },
		{ 2565, 2535 },
		{ 1882, 1862 },
		{ 1377, 1376 },
		{ 1688, 1687 },
		{ 1327, 1326 },
		{ 3076, 3075 },
		{ 2110, 2089 },
		{ 2111, 2091 },
		{ 2787, 2770 },
		{ 1504, 1482 },
		{ 3199, 3196 },
		{ 2449, 2420 },
		{ 2223, 2220 },
		{ 1823, 1822 },
		{ 2529, 2529 },
		{ 2529, 2529 },
		{ 1278, 1278 },
		{ 1278, 1278 },
		{ 2670, 2670 },
		{ 2670, 2670 },
		{ 2671, 2671 },
		{ 2671, 2671 },
		{ 2941, 2940 },
		{ 3030, 3029 },
		{ 2968, 2967 },
		{ 1326, 1325 },
		{ 0, 2293 },
		{ 1800, 1799 },
		{ 1856, 2012 },
		{ 3073, 3072 },
		{ 2047, 2238 },
		{ 3037, 3036 },
		{ 1436, 1427 },
		{ 1789, 1788 },
		{ 2679, 2653 },
		{ 2996, 2995 },
		{ 2900, 2899 },
		{ 2529, 2529 },
		{ 2888, 2887 },
		{ 1278, 1278 },
		{ 1315, 1314 },
		{ 2670, 2670 },
		{ 1371, 1370 },
		{ 2671, 2671 },
		{ 133, 133 },
		{ 133, 133 },
		{ 133, 133 },
		{ 133, 133 },
		{ 133, 133 },
		{ 133, 133 },
		{ 133, 133 },
		{ 133, 133 },
		{ 133, 133 },
		{ 133, 133 },
		{ 3049, 3048 },
		{ 2141, 2125 },
		{ 2155, 2139 },
		{ 1287, 1286 },
		{ 0, 2293 },
		{ 1265, 1265 },
		{ 1265, 1265 },
		{ 1265, 1265 },
		{ 1265, 1265 },
		{ 1265, 1265 },
		{ 1265, 1265 },
		{ 1265, 1265 },
		{ 1265, 1265 },
		{ 1265, 1265 },
		{ 1265, 1265 },
		{ 135, 135 },
		{ 135, 135 },
		{ 135, 135 },
		{ 135, 135 },
		{ 135, 135 },
		{ 135, 135 },
		{ 135, 135 },
		{ 135, 135 },
		{ 135, 135 },
		{ 135, 135 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2658, 2658 },
		{ 2658, 2658 },
		{ 3097, 3097 },
		{ 3097, 3097 },
		{ 3097, 3097 },
		{ 3097, 3097 },
		{ 3097, 3097 },
		{ 3097, 3097 },
		{ 3097, 3097 },
		{ 3097, 3097 },
		{ 3097, 3097 },
		{ 3097, 3097 },
		{ 2779, 2761 },
		{ 2782, 2764 },
		{ 3058, 3057 },
		{ 1960, 1946 },
		{ 2557, 2529 },
		{ 1389, 1388 },
		{ 1279, 1278 },
		{ 2171, 2157 },
		{ 2696, 2670 },
		{ 1437, 1416 },
		{ 2697, 2671 },
		{ 2658, 2658 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 119, 119 },
		{ 119, 119 },
		{ 119, 119 },
		{ 119, 119 },
		{ 119, 119 },
		{ 119, 119 },
		{ 119, 119 },
		{ 119, 119 },
		{ 119, 119 },
		{ 119, 119 },
		{ 1346, 1346 },
		{ 1346, 1346 },
		{ 0, 1258 },
		{ 0, 86 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2963, 2963 },
		{ 2963, 2963 },
		{ 3198, 3198 },
		{ 2982, 2982 },
		{ 2982, 2982 },
		{ 2641, 2641 },
		{ 2641, 2641 },
		{ 2903, 2903 },
		{ 2903, 2903 },
		{ 1346, 1346 },
		{ 1438, 1416 },
		{ 2172, 2158 },
		{ 2684, 2658 },
		{ 2923, 2922 },
		{ 1373, 1372 },
		{ 2790, 2773 },
		{ 2453, 2423 },
		{ 2622, 2591 },
		{ 1830, 1829 },
		{ 3080, 3079 },
		{ 0, 1258 },
		{ 0, 86 },
		{ 2703, 2677 },
		{ 2963, 2963 },
		{ 2420, 2387 },
		{ 3198, 3198 },
		{ 2982, 2982 },
		{ 1406, 1405 },
		{ 2641, 2641 },
		{ 1666, 1665 },
		{ 2903, 2903 },
		{ 2709, 2683 },
		{ 2944, 2943 },
		{ 3089, 3088 },
		{ 1796, 1795 },
		{ 1322, 1321 },
		{ 2814, 2799 },
		{ 2815, 2800 },
		{ 2951, 2950 },
		{ 2513, 2482 },
		{ 1668, 1667 },
		{ 2719, 2694 },
		{ 1518, 1498 },
		{ 2633, 2605 },
		{ 2636, 2609 },
		{ 2637, 2611 },
		{ 2638, 2612 },
		{ 2969, 2968 },
		{ 1865, 1843 },
		{ 1670, 1669 },
		{ 1375, 1374 },
		{ 2005, 2003 },
		{ 2986, 2985 },
		{ 2468, 2439 },
		{ 2738, 2717 },
		{ 2989, 2988 },
		{ 2469, 2440 },
		{ 2741, 2720 },
		{ 2845, 2844 },
		{ 1673, 1672 },
		{ 2221, 2218 },
		{ 2744, 2724 },
		{ 2657, 2631 },
		{ 3008, 3007 },
		{ 1773, 1772 },
		{ 1409, 1408 },
		{ 1883, 1863 },
		{ 2752, 2732 },
		{ 1810, 1809 },
		{ 2051, 2029 },
		{ 3017, 3016 },
		{ 2878, 2877 },
		{ 2760, 2740 },
		{ 2232, 2230 },
		{ 3186, 3186 },
		{ 2481, 2452 },
		{ 1347, 1346 },
		{ 1937, 1922 },
		{ 1775, 1774 },
		{ 1887, 1868 },
		{ 1336, 1335 },
		{ 1378, 1377 },
		{ 3040, 3039 },
		{ 2781, 2763 },
		{ 2999, 2998 },
		{ 2234, 2233 },
		{ 2980, 2979 },
		{ 1795, 1794 },
		{ 69, 5 },
		{ 2891, 2890 },
		{ 2964, 2963 },
		{ 3198, 3195 },
		{ 2983, 2982 },
		{ 3027, 3026 },
		{ 2667, 2641 },
		{ 3200, 3198 },
		{ 2904, 2903 },
		{ 3186, 3186 },
		{ 2965, 2964 },
		{ 2672, 2646 },
		{ 1321, 1320 },
		{ 2780, 2762 },
		{ 2985, 2984 },
		{ 1388, 1387 },
		{ 1798, 1797 },
		{ 2729, 2705 },
		{ 2971, 2970 },
		{ 2225, 2223 },
		{ 2991, 2990 },
		{ 1324, 1323 },
		{ 2776, 2758 },
		{ 2615, 2584 },
		{ 2931, 2929 },
		{ 3129, 3123 },
		{ 1876, 1853 },
		{ 1866, 1843 },
		{ 2718, 2693 },
		{ 2640, 2614 },
		{ 1877, 1853 },
		{ 3035, 3034 },
		{ 2465, 2436 },
		{ 2534, 2505 },
		{ 2906, 2905 },
		{ 2564, 2534 },
		{ 1858, 1838 },
		{ 2943, 2942 },
		{ 1498, 1476 },
		{ 2049, 2026 },
		{ 2932, 2929 },
		{ 1857, 1838 },
		{ 2394, 2363 },
		{ 2614, 2583 },
		{ 2048, 2026 },
		{ 2510, 2479 },
		{ 1310, 1309 },
		{ 1784, 1783 },
		{ 2052, 2029 },
		{ 2240, 2237 },
		{ 2739, 2718 },
		{ 2255, 2254 },
		{ 176, 5 },
		{ 2014, 2011 },
		{ 2239, 2237 },
		{ 3130, 3123 },
		{ 177, 5 },
		{ 2994, 2993 },
		{ 2013, 2011 },
		{ 1454, 1428 },
		{ 2050, 2027 },
		{ 100, 83 },
		{ 1708, 1707 },
		{ 178, 5 },
		{ 2571, 2540 },
		{ 1762, 1761 },
		{ 1912, 1893 },
		{ 2685, 2659 },
		{ 1817, 1816 },
		{ 3002, 3001 },
		{ 2222, 2219 },
		{ 3004, 3003 },
		{ 2690, 2664 },
		{ 2578, 2547 },
		{ 2829, 2823 },
		{ 1475, 1453 },
		{ 1916, 1897 },
		{ 1476, 1454 },
		{ 3189, 3186 },
		{ 175, 5 },
		{ 1394, 1393 },
		{ 1920, 1901 },
		{ 1821, 1820 },
		{ 1923, 1905 },
		{ 2840, 2838 },
		{ 2589, 2557 },
		{ 1712, 1711 },
		{ 2087, 2066 },
		{ 2500, 2470 },
		{ 3028, 3027 },
		{ 2088, 2067 },
		{ 2595, 2563 },
		{ 1352, 1351 },
		{ 1276, 1275 },
		{ 2870, 2869 },
		{ 2601, 2568 },
		{ 1359, 1358 },
		{ 3043, 3042 },
		{ 2721, 2696 },
		{ 3045, 3044 },
		{ 2722, 2697 },
		{ 2254, 2253 },
		{ 2507, 2476 },
		{ 1935, 1919 },
		{ 1297, 1296 },
		{ 1402, 1401 },
		{ 2731, 2707 },
		{ 2105, 2084 },
		{ 1675, 1674 },
		{ 1777, 1776 },
		{ 2894, 2893 },
		{ 1557, 1536 },
		{ 2896, 2895 },
		{ 1682, 1681 },
		{ 1859, 1839 },
		{ 3069, 3068 },
		{ 2618, 2587 },
		{ 2619, 2588 },
		{ 1860, 1840 },
		{ 1861, 1840 },
		{ 2270, 2269 },
		{ 2905, 2904 },
		{ 1558, 1537 },
		{ 2272, 2271 },
		{ 2114, 2096 },
		{ 2909, 2908 },
		{ 2122, 2104 },
		{ 2912, 2911 },
		{ 1786, 1785 },
		{ 2916, 2915 },
		{ 1732, 1731 },
		{ 1329, 1328 },
		{ 2629, 2598 },
		{ 3102, 3093 },
		{ 2754, 2734 },
		{ 1344, 1343 },
		{ 1791, 1790 },
		{ 2759, 2739 },
		{ 2129, 2112 },
		{ 1976, 1966 },
		{ 2635, 2608 },
		{ 1686, 1685 },
		{ 1736, 1735 },
		{ 3133, 3127 },
		{ 3134, 3129 },
		{ 1380, 1379 },
		{ 2935, 2933 },
		{ 1992, 1984 },
		{ 2768, 2748 },
		{ 1993, 1985 },
		{ 2942, 2941 },
		{ 2770, 2750 },
		{ 1884, 1864 },
		{ 3152, 3148 },
		{ 2644, 2618 },
		{ 2946, 2945 },
		{ 2645, 2619 },
		{ 1363, 1362 },
		{ 1654, 1653 },
		{ 1744, 1743 },
		{ 1387, 1386 },
		{ 2004, 2002 },
		{ 2173, 2159 },
		{ 3170, 3169 },
		{ 2656, 2630 },
		{ 1313, 1312 },
		{ 1862, 1840 },
		{ 2328, 2298 },
		{ 1280, 1279 },
		{ 2470, 2441 },
		{ 2966, 2965 },
		{ 1658, 1657 },
		{ 3182, 3179 },
		{ 1803, 1802 },
		{ 1309, 1308 },
		{ 2665, 2639 },
		{ 1317, 1316 },
		{ 3191, 3188 },
		{ 2204, 2191 },
		{ 2208, 2201 },
		{ 2981, 2980 },
		{ 2674, 2648 },
		{ 2984, 2983 },
		{ 2802, 2786 },
		{ 1758, 1757 },
		{ 2478, 2449 },
		{ 1701, 1700 },
		{ 2392, 2360 },
		{ 2913, 2912 },
		{ 3032, 3031 },
		{ 3140, 3135 },
		{ 1483, 1461 },
		{ 2936, 2934 },
		{ 2094, 2073 },
		{ 2066, 2043 },
		{ 2962, 2961 },
		{ 3188, 3185 },
		{ 1401, 1400 },
		{ 3071, 3070 },
		{ 130, 115 },
		{ 1829, 1828 },
		{ 1914, 1895 },
		{ 3108, 3106 },
		{ 2654, 2628 },
		{ 2107, 2086 },
		{ 2536, 2507 },
		{ 1904, 1885 },
		{ 2750, 2730 },
		{ 1924, 1906 },
		{ 3056, 3055 },
		{ 3087, 3086 },
		{ 639, 579 },
		{ 2921, 2920 },
		{ 2784, 2766 },
		{ 2890, 2889 },
		{ 2651, 2625 },
		{ 3184, 3181 },
		{ 3039, 3038 },
		{ 3187, 3184 },
		{ 1281, 1280 },
		{ 2967, 2966 },
		{ 640, 579 },
		{ 2598, 2566 },
		{ 3015, 3014 },
		{ 2156, 2141 },
		{ 2876, 2875 },
		{ 3195, 3192 },
		{ 1694, 1693 },
		{ 2998, 2997 },
		{ 1808, 1807 },
		{ 2915, 2914 },
		{ 3202, 3200 },
		{ 1952, 1937 },
		{ 1334, 1333 },
		{ 1404, 1403 },
		{ 2724, 2699 },
		{ 2694, 2668 },
		{ 641, 579 },
		{ 1436, 1429 },
		{ 2047, 2041 },
		{ 2642, 2616 },
		{ 1386, 1383 },
		{ 1856, 1852 },
		{ 2669, 2643 },
		{ 2720, 2695 },
		{ 2673, 2647 },
		{ 205, 181 },
		{ 1349, 1348 },
		{ 1973, 1963 },
		{ 203, 181 },
		{ 1824, 1823 },
		{ 204, 181 },
		{ 2266, 2265 },
		{ 2215, 2211 },
		{ 2268, 2267 },
		{ 2634, 2607 },
		{ 1689, 1688 },
		{ 1739, 1738 },
		{ 1283, 1282 },
		{ 2889, 2888 },
		{ 202, 181 },
		{ 1396, 1395 },
		{ 2687, 2661 },
		{ 2142, 2127 },
		{ 2801, 2785 },
		{ 2597, 2565 },
		{ 3038, 3037 },
		{ 2143, 2130 },
		{ 1878, 1854 },
		{ 2898, 2897 },
		{ 2515, 2484 },
		{ 2062, 2038 },
		{ 1765, 1764 },
		{ 2695, 2669 },
		{ 3047, 3046 },
		{ 1723, 1722 },
		{ 1617, 1601 },
		{ 2753, 2733 },
		{ 2699, 2673 },
		{ 2065, 2042 },
		{ 2170, 2156 },
		{ 3165, 3162 },
		{ 1834, 1833 },
		{ 2001, 1998 },
		{ 2826, 2817 },
		{ 2613, 2582 },
		{ 3172, 3171 },
		{ 2485, 2456 },
		{ 2708, 2682 },
		{ 3175, 3174 },
		{ 1746, 1745 },
		{ 2710, 2684 },
		{ 1376, 1375 },
		{ 1366, 1365 },
		{ 3070, 3069 },
		{ 1749, 1748 },
		{ 1285, 1284 },
		{ 3185, 3182 },
		{ 2997, 2996 },
		{ 3075, 3074 },
		{ 2771, 2751 },
		{ 2843, 2842 },
		{ 3078, 3077 },
		{ 2533, 2504 },
		{ 3192, 3189 },
		{ 1671, 1670 },
		{ 2256, 2255 },
		{ 1715, 1714 },
		{ 1661, 1660 },
		{ 3006, 3005 },
		{ 2668, 2642 },
		{ 2725, 2701 },
		{ 3201, 3199 },
		{ 2580, 2549 },
		{ 1965, 1952 },
		{ 2209, 2202 },
		{ 2940, 2939 },
		{ 883, 830 },
		{ 438, 394 },
		{ 701, 638 },
		{ 768, 710 },
		{ 1680, 27 },
		{ 1706, 29 },
		{ 1273, 11 },
		{ 69, 27 },
		{ 69, 29 },
		{ 69, 11 },
		{ 2974, 57 },
		{ 1307, 15 },
		{ 1730, 31 },
		{ 69, 57 },
		{ 69, 15 },
		{ 69, 31 },
		{ 1294, 13 },
		{ 439, 394 },
		{ 1357, 19 },
		{ 69, 13 },
		{ 767, 710 },
		{ 69, 19 },
		{ 2867, 47 },
		{ 1116, 1088 },
		{ 702, 638 },
		{ 69, 47 },
		{ 884, 830 },
		{ 1756, 33 },
		{ 1782, 35 },
		{ 1385, 21 },
		{ 69, 33 },
		{ 69, 35 },
		{ 69, 21 },
		{ 3064, 61 },
		{ 1129, 1102 },
		{ 1135, 1109 },
		{ 69, 61 },
		{ 1142, 1117 },
		{ 1143, 1118 },
		{ 2076, 2055 },
		{ 1157, 1138 },
		{ 1158, 1139 },
		{ 1176, 1159 },
		{ 1181, 1165 },
		{ 1189, 1177 },
		{ 1201, 1190 },
		{ 1207, 1196 },
		{ 1242, 1241 },
		{ 1888, 1869 },
		{ 268, 227 },
		{ 278, 236 },
		{ 285, 243 },
		{ 2098, 2077 },
		{ 2099, 2078 },
		{ 300, 255 },
		{ 308, 263 },
		{ 315, 269 },
		{ 325, 279 },
		{ 342, 295 },
		{ 345, 298 },
		{ 353, 305 },
		{ 354, 306 },
		{ 359, 311 },
		{ 374, 329 },
		{ 1908, 1889 },
		{ 2121, 2103 },
		{ 1909, 1890 },
		{ 399, 355 },
		{ 402, 358 },
		{ 410, 366 },
		{ 421, 378 },
		{ 428, 385 },
		{ 437, 393 },
		{ 234, 198 },
		{ 440, 395 },
		{ 2137, 2120 },
		{ 453, 406 },
		{ 238, 202 },
		{ 475, 422 },
		{ 1929, 1911 },
		{ 478, 426 },
		{ 488, 434 },
		{ 489, 435 },
		{ 503, 449 },
		{ 512, 460 },
		{ 545, 484 },
		{ 1678, 27 },
		{ 1704, 29 },
		{ 1271, 11 },
		{ 555, 492 },
		{ 558, 495 },
		{ 1944, 1928 },
		{ 2973, 57 },
		{ 1305, 15 },
		{ 1728, 31 },
		{ 559, 496 },
		{ 563, 500 },
		{ 576, 515 },
		{ 1292, 13 },
		{ 580, 519 },
		{ 1355, 19 },
		{ 584, 523 },
		{ 594, 533 },
		{ 609, 546 },
		{ 2865, 47 },
		{ 610, 548 },
		{ 615, 553 },
		{ 631, 569 },
		{ 239, 203 },
		{ 1754, 33 },
		{ 1780, 35 },
		{ 1383, 21 },
		{ 648, 583 },
		{ 661, 596 },
		{ 664, 599 },
		{ 3062, 61 },
		{ 673, 608 },
		{ 689, 623 },
		{ 690, 624 },
		{ 700, 637 },
		{ 245, 209 },
		{ 720, 656 },
		{ 246, 210 },
		{ 769, 711 },
		{ 781, 722 },
		{ 783, 724 },
		{ 785, 726 },
		{ 790, 731 },
		{ 810, 751 },
		{ 821, 761 },
		{ 825, 765 },
		{ 826, 766 },
		{ 856, 800 },
		{ 875, 822 },
		{ 267, 226 },
		{ 913, 862 },
		{ 923, 872 },
		{ 938, 887 },
		{ 975, 928 },
		{ 979, 932 },
		{ 995, 951 },
		{ 1011, 971 },
		{ 1036, 1000 },
		{ 1039, 1003 },
		{ 1047, 1012 },
		{ 1057, 1022 },
		{ 1075, 1042 },
		{ 1094, 1060 },
		{ 1101, 1069 },
		{ 1103, 1071 },
		{ 69, 17 },
		{ 69, 25 },
		{ 69, 59 },
		{ 445, 399 },
		{ 69, 49 },
		{ 446, 399 },
		{ 444, 399 },
		{ 69, 37 },
		{ 69, 43 },
		{ 69, 53 },
		{ 69, 55 },
		{ 221, 189 },
		{ 2151, 2136 },
		{ 2149, 2135 },
		{ 3149, 3145 },
		{ 219, 189 },
		{ 3155, 3150 },
		{ 3121, 3120 },
		{ 2210, 2203 },
		{ 2152, 2136 },
		{ 2150, 2135 },
		{ 836, 776 },
		{ 447, 399 },
		{ 480, 428 },
		{ 482, 428 },
		{ 514, 462 },
		{ 601, 540 },
		{ 522, 469 },
		{ 523, 469 },
		{ 2147, 2133 },
		{ 222, 189 },
		{ 220, 189 },
		{ 1956, 1942 },
		{ 404, 360 },
		{ 483, 428 },
		{ 524, 469 },
		{ 743, 686 },
		{ 744, 687 },
		{ 420, 377 },
		{ 481, 428 },
		{ 448, 400 },
		{ 295, 250 },
		{ 535, 478 },
		{ 1132, 1105 },
		{ 2059, 2035 },
		{ 338, 291 },
		{ 1138, 1112 },
		{ 699, 636 },
		{ 833, 773 },
		{ 455, 408 },
		{ 215, 186 },
		{ 537, 478 },
		{ 2058, 2035 },
		{ 214, 186 },
		{ 2080, 2059 },
		{ 367, 319 },
		{ 536, 478 },
		{ 645, 580 },
		{ 208, 183 },
		{ 216, 186 },
		{ 210, 183 },
		{ 644, 580 },
		{ 929, 878 },
		{ 209, 183 },
		{ 2057, 2035 },
		{ 930, 879 },
		{ 816, 756 },
		{ 431, 387 },
		{ 430, 387 },
		{ 643, 580 },
		{ 642, 580 },
		{ 261, 221 },
		{ 229, 193 },
		{ 1147, 1122 },
		{ 526, 471 },
		{ 815, 756 },
		{ 2081, 2060 },
		{ 837, 777 },
		{ 838, 778 },
		{ 1340, 17 },
		{ 1651, 25 },
		{ 3021, 59 },
		{ 529, 473 },
		{ 2883, 49 },
		{ 228, 193 },
		{ 530, 473 },
		{ 1814, 37 },
		{ 2251, 43 },
		{ 2929, 53 },
		{ 2956, 55 },
		{ 1891, 1872 },
		{ 1146, 1122 },
		{ 731, 671 },
		{ 272, 230 },
		{ 527, 471 },
		{ 593, 532 },
		{ 225, 190 },
		{ 304, 259 },
		{ 925, 874 },
		{ 223, 190 },
		{ 606, 545 },
		{ 305, 260 },
		{ 224, 190 },
		{ 542, 482 },
		{ 607, 545 },
		{ 286, 244 },
		{ 1342, 1340 },
		{ 2511, 2480 },
		{ 560, 497 },
		{ 3119, 3117 },
		{ 1274, 1271 },
		{ 1197, 1186 },
		{ 1198, 1187 },
		{ 1199, 1188 },
		{ 3132, 3126 },
		{ 608, 545 },
		{ 1079, 1046 },
		{ 817, 757 },
		{ 543, 482 },
		{ 433, 389 },
		{ 287, 244 },
		{ 1102, 1070 },
		{ 765, 708 },
		{ 766, 709 },
		{ 3141, 3136 },
		{ 240, 204 },
		{ 1871, 1847 },
		{ 3146, 3142 },
		{ 627, 564 },
		{ 714, 650 },
		{ 986, 941 },
		{ 990, 946 },
		{ 499, 445 },
		{ 842, 782 },
		{ 1295, 1292 },
		{ 2868, 2865 },
		{ 582, 521 },
		{ 732, 672 },
		{ 1172, 1155 },
		{ 1301, 1300 },
		{ 734, 674 },
		{ 1179, 1162 },
		{ 893, 841 },
		{ 211, 184 },
		{ 573, 512 },
		{ 1151, 1129 },
		{ 1890, 1871 },
		{ 1156, 1136 },
		{ 212, 184 },
		{ 832, 772 },
		{ 349, 301 },
		{ 1165, 1147 },
		{ 1166, 1149 },
		{ 302, 257 },
		{ 406, 362 },
		{ 1177, 1160 },
		{ 409, 365 },
		{ 2078, 2057 },
		{ 572, 512 },
		{ 685, 619 },
		{ 853, 796 },
		{ 1190, 1178 },
		{ 1192, 1180 },
		{ 854, 797 },
		{ 269, 228 },
		{ 864, 810 },
		{ 872, 819 },
		{ 1205, 1194 },
		{ 3120, 3119 },
		{ 1206, 1195 },
		{ 567, 505 },
		{ 1211, 1202 },
		{ 1226, 1217 },
		{ 568, 506 },
		{ 1253, 1252 },
		{ 891, 839 },
		{ 413, 369 },
		{ 3137, 3132 },
		{ 898, 846 },
		{ 908, 857 },
		{ 574, 513 },
		{ 355, 307 },
		{ 717, 653 },
		{ 358, 310 },
		{ 3145, 3141 },
		{ 283, 241 },
		{ 429, 386 },
		{ 939, 888 },
		{ 945, 894 },
		{ 3150, 3146 },
		{ 951, 901 },
		{ 956, 906 },
		{ 961, 913 },
		{ 967, 921 },
		{ 974, 927 },
		{ 588, 527 },
		{ 735, 675 },
		{ 592, 531 },
		{ 504, 450 },
		{ 745, 688 },
		{ 996, 952 },
		{ 1008, 968 },
		{ 199, 180 },
		{ 1016, 976 },
		{ 1017, 977 },
		{ 1021, 981 },
		{ 201, 180 },
		{ 1022, 982 },
		{ 1028, 988 },
		{ 599, 538 },
		{ 372, 326 },
		{ 311, 266 },
		{ 376, 331 },
		{ 200, 180 },
		{ 1059, 1024 },
		{ 1061, 1026 },
		{ 1065, 1030 },
		{ 1072, 1039 },
		{ 1074, 1041 },
		{ 378, 333 },
		{ 533, 476 },
		{ 1087, 1053 },
		{ 394, 351 },
		{ 629, 566 },
		{ 812, 753 },
		{ 813, 754 },
		{ 1105, 1073 },
		{ 1113, 1082 },
		{ 539, 480 },
		{ 1128, 1101 },
		{ 634, 572 },
		{ 1130, 1103 },
		{ 820, 760 },
		{ 638, 578 },
		{ 822, 762 },
		{ 348, 300 },
		{ 451, 403 },
		{ 831, 771 },
		{ 621, 559 },
		{ 1170, 1153 },
		{ 258, 219 },
		{ 516, 465 },
		{ 1926, 1908 },
		{ 571, 511 },
		{ 347, 300 },
		{ 346, 300 },
		{ 259, 219 },
		{ 630, 567 },
		{ 417, 374 },
		{ 1031, 992 },
		{ 1032, 993 },
		{ 1033, 995 },
		{ 754, 699 },
		{ 1580, 1580 },
		{ 1583, 1583 },
		{ 1586, 1586 },
		{ 1589, 1589 },
		{ 1592, 1592 },
		{ 1595, 1595 },
		{ 878, 825 },
		{ 1042, 1006 },
		{ 879, 826 },
		{ 356, 308 },
		{ 1058, 1023 },
		{ 636, 576 },
		{ 383, 338 },
		{ 1062, 1027 },
		{ 424, 381 },
		{ 1613, 1613 },
		{ 1236, 1230 },
		{ 312, 267 },
		{ 232, 196 },
		{ 920, 869 },
		{ 922, 871 },
		{ 1083, 1049 },
		{ 1622, 1622 },
		{ 1580, 1580 },
		{ 1583, 1583 },
		{ 1586, 1586 },
		{ 1589, 1589 },
		{ 1592, 1592 },
		{ 1595, 1595 },
		{ 655, 590 },
		{ 586, 525 },
		{ 1099, 1066 },
		{ 1543, 1543 },
		{ 2116, 2098 },
		{ 233, 197 },
		{ 671, 606 },
		{ 403, 359 },
		{ 554, 491 },
		{ 1613, 1613 },
		{ 1108, 1076 },
		{ 368, 322 },
		{ 597, 536 },
		{ 695, 629 },
		{ 332, 285 },
		{ 966, 920 },
		{ 1622, 1622 },
		{ 1647, 1647 },
		{ 248, 212 },
		{ 603, 542 },
		{ 375, 330 },
		{ 244, 208 },
		{ 719, 655 },
		{ 564, 501 },
		{ 992, 948 },
		{ 450, 402 },
		{ 1543, 1543 },
		{ 620, 558 },
		{ 840, 780 },
		{ 733, 673 },
		{ 835, 775 },
		{ 1018, 978 },
		{ 434, 390 },
		{ 1453, 1580 },
		{ 1453, 1583 },
		{ 1453, 1586 },
		{ 1453, 1589 },
		{ 1453, 1592 },
		{ 1453, 1595 },
		{ 619, 557 },
		{ 1647, 1647 },
		{ 341, 294 },
		{ 726, 662 },
		{ 1239, 1237 },
		{ 730, 668 },
		{ 1243, 1242 },
		{ 843, 783 },
		{ 3116, 3112 },
		{ 1453, 1613 },
		{ 1034, 997 },
		{ 850, 792 },
		{ 852, 794 },
		{ 307, 262 },
		{ 510, 457 },
		{ 1049, 1014 },
		{ 1453, 1622 },
		{ 855, 798 },
		{ 407, 363 },
		{ 317, 271 },
		{ 218, 188 },
		{ 737, 677 },
		{ 738, 679 },
		{ 740, 682 },
		{ 380, 335 },
		{ 414, 370 },
		{ 1453, 1543 },
		{ 637, 577 },
		{ 1928, 1910 },
		{ 749, 691 },
		{ 752, 696 },
		{ 753, 697 },
		{ 366, 318 },
		{ 1100, 1067 },
		{ 763, 706 },
		{ 388, 343 },
		{ 534, 477 },
		{ 389, 345 },
		{ 1943, 1927 },
		{ 284, 242 },
		{ 1453, 1647 },
		{ 934, 883 },
		{ 937, 886 },
		{ 1120, 1092 },
		{ 1123, 1095 },
		{ 1124, 1096 },
		{ 659, 594 },
		{ 395, 352 },
		{ 940, 889 },
		{ 943, 892 },
		{ 663, 598 },
		{ 1137, 1111 },
		{ 333, 286 },
		{ 2120, 2102 },
		{ 1970, 1958 },
		{ 1971, 1959 },
		{ 1139, 1113 },
		{ 953, 903 },
		{ 954, 904 },
		{ 955, 905 },
		{ 795, 736 },
		{ 801, 742 },
		{ 665, 600 },
		{ 669, 604 },
		{ 2134, 2117 },
		{ 547, 486 },
		{ 260, 220 },
		{ 373, 327 },
		{ 602, 541 },
		{ 556, 493 },
		{ 492, 438 },
		{ 993, 949 },
		{ 495, 441 },
		{ 498, 444 },
		{ 999, 958 },
		{ 1005, 965 },
		{ 612, 550 },
		{ 1009, 969 },
		{ 703, 639 },
		{ 2167, 2153 },
		{ 2168, 2154 },
		{ 614, 552 },
		{ 207, 182 },
		{ 251, 214 },
		{ 397, 353 },
		{ 396, 353 },
		{ 250, 214 },
		{ 518, 467 },
		{ 681, 615 },
		{ 682, 615 },
		{ 549, 488 },
		{ 550, 488 },
		{ 206, 182 },
		{ 551, 489 },
		{ 552, 489 },
		{ 289, 245 },
		{ 288, 245 },
		{ 626, 563 },
		{ 519, 467 },
		{ 254, 217 },
		{ 625, 563 },
		{ 265, 224 },
		{ 255, 217 },
		{ 294, 249 },
		{ 313, 268 },
		{ 520, 468 },
		{ 778, 720 },
		{ 264, 224 },
		{ 457, 410 },
		{ 1958, 1944 },
		{ 787, 728 },
		{ 256, 217 },
		{ 314, 268 },
		{ 293, 249 },
		{ 1024, 984 },
		{ 521, 468 },
		{ 779, 720 },
		{ 497, 443 },
		{ 2138, 2121 },
		{ 791, 732 },
		{ 952, 902 },
		{ 684, 617 },
		{ 2077, 2056 },
		{ 362, 314 },
		{ 531, 474 },
		{ 611, 549 },
		{ 2082, 2061 },
		{ 739, 681 },
		{ 2153, 2137 },
		{ 363, 315 },
		{ 696, 631 },
		{ 613, 551 },
		{ 1245, 1244 },
		{ 477, 424 },
		{ 651, 586 },
		{ 277, 235 },
		{ 707, 643 },
		{ 1149, 1127 },
		{ 1150, 1128 },
		{ 708, 644 },
		{ 1927, 1909 },
		{ 1152, 1130 },
		{ 915, 864 },
		{ 709, 645 },
		{ 296, 251 },
		{ 263, 223 },
		{ 1002, 961 },
		{ 1168, 1151 },
		{ 596, 535 },
		{ 3117, 3115 },
		{ 280, 238 },
		{ 398, 354 },
		{ 1889, 1870 },
		{ 839, 779 },
		{ 2053, 2030 },
		{ 2117, 2099 },
		{ 1945, 1929 },
		{ 493, 439 },
		{ 1892, 1873 },
		{ 2480, 2451 },
		{ 3136, 3131 },
		{ 494, 440 },
		{ 1186, 1174 },
		{ 1187, 1175 },
		{ 784, 725 },
		{ 1104, 1072 },
		{ 3142, 3137 },
		{ 271, 229 },
		{ 1095, 1061 },
		{ 1096, 1062 },
		{ 1097, 1063 },
		{ 561, 498 },
		{ 2056, 2034 },
		{ 794, 735 },
		{ 270, 229 },
		{ 2060, 2036 },
		{ 924, 873 },
		{ 443, 398 },
		{ 509, 456 },
		{ 590, 529 },
		{ 1870, 1846 },
		{ 2148, 2134 },
		{ 811, 752 },
		{ 1872, 1848 },
		{ 725, 661 },
		{ 454, 407 },
		{ 728, 664 },
		{ 1212, 1203 },
		{ 1223, 1214 },
		{ 1224, 1215 },
		{ 252, 215 },
		{ 1229, 1220 },
		{ 1121, 1093 },
		{ 1237, 1232 },
		{ 1238, 1233 },
		{ 1957, 1943 },
		{ 2186, 2167 },
		{ 858, 804 },
		{ 1025, 985 },
		{ 1126, 1098 },
		{ 570, 510 },
		{ 1029, 989 },
		{ 486, 431 },
		{ 617, 555 },
		{ 1133, 1106 },
		{ 824, 764 },
		{ 775, 717 },
		{ 777, 719 },
		{ 1981, 1970 },
		{ 886, 832 },
		{ 1140, 1115 },
		{ 957, 907 },
		{ 1043, 1007 },
		{ 958, 908 },
		{ 301, 256 },
		{ 557, 494 },
		{ 896, 844 },
		{ 600, 539 },
		{ 901, 849 },
		{ 904, 853 },
		{ 984, 938 },
		{ 905, 854 },
		{ 266, 225 },
		{ 911, 860 },
		{ 1077, 1044 },
		{ 500, 446 },
		{ 230, 194 },
		{ 604, 543 },
		{ 1921, 1902 },
		{ 1090, 1056 },
		{ 997, 956 },
		{ 1185, 1173 },
		{ 747, 690 },
		{ 1159, 1140 },
		{ 1164, 1146 },
		{ 249, 213 },
		{ 405, 361 },
		{ 971, 924 },
		{ 748, 690 },
		{ 538, 479 },
		{ 517, 466 },
		{ 647, 582 },
		{ 579, 518 },
		{ 1109, 1077 },
		{ 1110, 1079 },
		{ 1183, 1171 },
		{ 1184, 1172 },
		{ 436, 392 },
		{ 860, 806 },
		{ 1118, 1090 },
		{ 458, 411 },
		{ 870, 817 },
		{ 1191, 1179 },
		{ 931, 880 },
		{ 1196, 1185 },
		{ 463, 419 },
		{ 1125, 1097 },
		{ 1060, 1025 },
		{ 660, 595 },
		{ 830, 770 },
		{ 1064, 1029 },
		{ 310, 265 },
		{ 1208, 1199 },
		{ 881, 828 },
		{ 384, 339 },
		{ 1221, 1212 },
		{ 944, 893 },
		{ 1076, 1043 },
		{ 589, 528 },
		{ 741, 683 },
		{ 1230, 1223 },
		{ 1231, 1224 },
		{ 1941, 1925 },
		{ 2132, 2115 },
		{ 1235, 1229 },
		{ 476, 423 },
		{ 710, 646 },
		{ 3115, 3111 },
		{ 1145, 1121 },
		{ 1088, 1054 },
		{ 1240, 1238 },
		{ 1148, 1126 },
		{ 1089, 1055 },
		{ 713, 649 },
		{ 1252, 1251 },
		{ 569, 509 },
		{ 3131, 3125 },
		{ 635, 573 },
		{ 1153, 1133 },
		{ 672, 607 },
		{ 2079, 2058 },
		{ 297, 252 },
		{ 909, 858 },
		{ 364, 316 },
		{ 704, 640 },
		{ 235, 199 },
		{ 678, 612 },
		{ 680, 614 },
		{ 987, 942 },
		{ 427, 384 },
		{ 2118, 2100 },
		{ 2119, 2101 },
		{ 449, 401 },
		{ 633, 571 },
		{ 605, 544 },
		{ 618, 556 },
		{ 694, 628 },
		{ 907, 856 },
		{ 721, 657 },
		{ 786, 727 },
		{ 461, 415 },
		{ 862, 808 },
		{ 340, 293 },
		{ 917, 866 },
		{ 418, 375 },
		{ 959, 911 },
		{ 623, 561 },
		{ 237, 201 },
		{ 876, 823 },
		{ 1250, 1249 },
		{ 799, 740 },
		{ 1081, 1047 },
		{ 1213, 1204 },
		{ 1218, 1209 },
		{ 1144, 1119 },
		{ 1082, 1047 },
		{ 1222, 1213 },
		{ 1080, 1047 },
		{ 323, 277 },
		{ 357, 309 },
		{ 291, 247 },
		{ 1227, 1218 },
		{ 1228, 1219 },
		{ 918, 867 },
		{ 834, 774 },
		{ 962, 914 },
		{ 1234, 1228 },
		{ 963, 915 },
		{ 553, 490 },
		{ 328, 282 },
		{ 715, 651 },
		{ 622, 560 },
		{ 718, 654 },
		{ 1163, 1144 },
		{ 487, 433 },
		{ 1037, 1001 },
		{ 1246, 1245 },
		{ 1038, 1002 },
		{ 1167, 1150 },
		{ 983, 936 },
		{ 1169, 1152 },
		{ 1040, 1004 },
		{ 371, 325 },
		{ 382, 337 },
		{ 1045, 1010 },
		{ 1114, 1084 },
		{ 849, 791 },
		{ 1182, 1168 },
		{ 1048, 1013 },
		{ 818, 758 },
		{ 897, 845 },
		{ 722, 658 },
		{ 941, 890 },
		{ 899, 847 },
		{ 236, 200 },
		{ 650, 585 },
		{ 950, 900 },
		{ 273, 231 },
		{ 1131, 1104 },
		{ 1911, 1892 },
		{ 1068, 1035 },
		{ 751, 693 },
		{ 2103, 2082 },
		{ 788, 729 },
		{ 859, 805 },
		{ 1012, 972 },
		{ 1013, 973 },
		{ 1078, 1045 },
		{ 616, 554 },
		{ 587, 526 },
		{ 1441, 1441 },
		{ 861, 807 },
		{ 750, 692 },
		{ 1020, 980 },
		{ 329, 283 },
		{ 541, 481 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 330, 283 },
		{ 540, 481 },
		{ 756, 700 },
		{ 755, 700 },
		{ 1070, 1036 },
		{ 757, 700 },
		{ 1203, 1192 },
		{ 2135, 2118 },
		{ 2136, 2119 },
		{ 1069, 1036 },
		{ 415, 371 },
		{ 686, 620 },
		{ 1023, 983 },
		{ 736, 676 },
		{ 1115, 1087 },
		{ 1441, 1441 },
		{ 687, 621 },
		{ 1027, 987 },
		{ 1214, 1205 },
		{ 1215, 1206 },
		{ 942, 891 },
		{ 1299, 1299 },
		{ 1220, 1211 },
		{ 462, 417 },
		{ 400, 356 },
		{ 804, 745 },
		{ 871, 818 },
		{ 805, 746 },
		{ 1127, 1100 },
		{ 873, 820 },
		{ 241, 205 },
		{ 419, 376 },
		{ 501, 447 },
		{ 1233, 1226 },
		{ 292, 248 },
		{ 257, 218 },
		{ 646, 581 },
		{ 1044, 1008 },
		{ 885, 831 },
		{ 578, 517 },
		{ 506, 453 },
		{ 649, 584 },
		{ 1055, 1020 },
		{ 1056, 1021 },
		{ 508, 455 },
		{ 337, 290 },
		{ 392, 348 },
		{ 657, 592 },
		{ 900, 848 },
		{ 1942, 1926 },
		{ 976, 929 },
		{ 1063, 1028 },
		{ 977, 930 },
		{ 828, 768 },
		{ 1453, 1441 },
		{ 902, 850 },
		{ 1300, 1299 },
		{ 511, 458 },
		{ 985, 939 },
		{ 1073, 1040 },
		{ 327, 281 },
		{ 513, 461 },
		{ 2100, 2079 },
		{ 2101, 2080 },
		{ 989, 945 },
		{ 377, 332 },
		{ 316, 270 },
		{ 771, 713 },
		{ 3065, 3062 },
		{ 774, 716 },
		{ 1171, 1154 },
		{ 914, 863 },
		{ 1173, 1156 },
		{ 411, 367 },
		{ 776, 718 },
		{ 1001, 960 },
		{ 667, 602 },
		{ 1093, 1059 },
		{ 1004, 964 },
		{ 491, 437 },
		{ 1007, 967 },
		{ 780, 721 },
		{ 1098, 1065 },
		{ 456, 409 },
		{ 595, 534 },
		{ 370, 324 },
		{ 628, 565 },
		{ 1194, 1183 },
		{ 525, 470 },
		{ 360, 312 },
		{ 528, 472 },
		{ 1106, 1074 },
		{ 789, 730 },
		{ 2133, 2116 },
		{ 882, 829 },
		{ 1085, 1051 },
		{ 1086, 1052 },
		{ 416, 372 },
		{ 1731, 1728 },
		{ 2201, 2188 },
		{ 2097, 2076 },
		{ 712, 648 },
		{ 515, 463 },
		{ 1984, 1975 },
		{ 932, 881 },
		{ 1757, 1754 },
		{ 1907, 1888 },
		{ 724, 660 },
		{ 1019, 979 },
		{ 339, 292 },
		{ 2159, 2144 },
		{ 1681, 1678 },
		{ 1358, 1355 },
		{ 581, 520 },
		{ 827, 767 },
		{ 1117, 1089 },
		{ 1966, 1953 },
		{ 1653, 1651 },
		{ 1707, 1704 },
		{ 1174, 1157 },
		{ 1175, 1158 },
		{ 1816, 1814 },
		{ 968, 922 },
		{ 381, 336 },
		{ 848, 790 },
		{ 290, 246 },
		{ 969, 922 },
		{ 1188, 1176 },
		{ 716, 652 },
		{ 1161, 1142 },
		{ 1162, 1143 },
		{ 851, 793 },
		{ 705, 641 },
		{ 1232, 1225 },
		{ 391, 347 },
		{ 793, 734 },
		{ 243, 207 },
		{ 935, 884 },
		{ 1141, 1116 },
		{ 759, 702 },
		{ 960, 912 },
		{ 442, 397 },
		{ 336, 289 },
		{ 1122, 1094 },
		{ 746, 689 },
		{ 435, 391 },
		{ 1046, 1011 },
		{ 841, 781 },
		{ 863, 809 },
		{ 496, 442 },
		{ 1154, 1134 },
		{ 1155, 1135 },
		{ 1071, 1037 },
		{ 479, 427 },
		{ 350, 302 },
		{ 484, 429 },
		{ 412, 368 },
		{ 761, 704 },
		{ 652, 587 },
		{ 903, 851 },
		{ 352, 304 },
		{ 991, 947 },
		{ 275, 233 },
		{ 1925, 1907 },
		{ 393, 349 },
		{ 994, 950 },
		{ 334, 287 },
		{ 770, 712 },
		{ 910, 859 },
		{ 276, 234 },
		{ 1092, 1058 },
		{ 1000, 959 },
		{ 912, 861 },
		{ 662, 597 },
		{ 1003, 962 },
		{ 575, 514 },
		{ 306, 261 },
		{ 1006, 966 },
		{ 577, 516 },
		{ 666, 601 },
		{ 532, 475 },
		{ 723, 659 },
		{ 2102, 2081 },
		{ 846, 786 },
		{ 668, 603 },
		{ 1014, 974 },
		{ 318, 272 },
		{ 927, 876 },
		{ 928, 877 },
		{ 670, 605 },
		{ 320, 274 },
		{ 452, 404 },
		{ 1959, 1945 },
		{ 217, 187 },
		{ 933, 882 },
		{ 2115, 2097 },
		{ 677, 611 },
		{ 423, 380 },
		{ 679, 613 },
		{ 1968, 1956 },
		{ 401, 357 },
		{ 425, 382 },
		{ 426, 383 },
		{ 798, 739 },
		{ 502, 448 },
		{ 548, 487 },
		{ 1873, 1849 },
		{ 1035, 998 },
		{ 802, 743 },
		{ 1225, 1216 },
		{ 866, 813 },
		{ 949, 898 },
		{ 803, 744 },
		{ 242, 206 },
		{ 1134, 1107 },
		{ 460, 413 },
		{ 806, 747 },
		{ 874, 821 },
		{ 809, 750 },
		{ 742, 684 },
		{ 361, 313 },
		{ 279, 237 },
		{ 880, 827 },
		{ 1050, 1015 },
		{ 1054, 1019 },
		{ 1241, 1239 },
		{ 344, 297 },
		{ 814, 755 },
		{ 1244, 1243 },
		{ 2154, 2138 },
		{ 464, 420 },
		{ 432, 388 },
		{ 1249, 1248 },
		{ 213, 185 },
		{ 1251, 1250 },
		{ 2161, 2147 },
		{ 2163, 2149 },
		{ 2164, 2150 },
		{ 2165, 2151 },
		{ 2166, 2152 },
		{ 887, 834 },
		{ 889, 837 },
		{ 196, 176 },
		{ 892, 840 },
		{ 2061, 2037 },
		{ 385, 340 },
		{ 895, 843 },
		{ 1066, 1031 },
		{ 1910, 1891 },
		{ 303, 258 },
		{ 978, 931 },
		{ 1581, 1581 },
		{ 1581, 1581 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 1977, 1977 },
		{ 1977, 1977 },
		{ 1979, 1979 },
		{ 1979, 1979 },
		{ 1590, 1590 },
		{ 1590, 1590 },
		{ 2174, 2174 },
		{ 2174, 2174 },
		{ 2176, 2176 },
		{ 2176, 2176 },
		{ 2178, 2178 },
		{ 2178, 2178 },
		{ 2180, 2180 },
		{ 2180, 2180 },
		{ 2182, 2182 },
		{ 2182, 2182 },
		{ 2184, 2184 },
		{ 2184, 2184 },
		{ 800, 741 },
		{ 1581, 1581 },
		{ 782, 723 },
		{ 2228, 2228 },
		{ 583, 522 },
		{ 1977, 1977 },
		{ 946, 895 },
		{ 1979, 1979 },
		{ 998, 957 },
		{ 1590, 1590 },
		{ 947, 896 },
		{ 2174, 2174 },
		{ 1302, 1301 },
		{ 2176, 2176 },
		{ 698, 635 },
		{ 2178, 2178 },
		{ 281, 239 },
		{ 2180, 2180 },
		{ 422, 379 },
		{ 2182, 2182 },
		{ 565, 503 },
		{ 2184, 2184 },
		{ 1954, 1954 },
		{ 1954, 1954 },
		{ 674, 609 },
		{ 675, 609 },
		{ 1989, 1989 },
		{ 1989, 1989 },
		{ 2199, 2199 },
		{ 2199, 2199 },
		{ 1582, 1581 },
		{ 319, 273 },
		{ 2229, 2228 },
		{ 868, 815 },
		{ 1978, 1977 },
		{ 1107, 1075 },
		{ 1980, 1979 },
		{ 485, 430 },
		{ 1591, 1590 },
		{ 299, 254 },
		{ 2175, 2174 },
		{ 591, 530 },
		{ 2177, 2176 },
		{ 1111, 1080 },
		{ 2179, 2178 },
		{ 1954, 1954 },
		{ 2181, 2180 },
		{ 1112, 1081 },
		{ 2183, 2182 },
		{ 1989, 1989 },
		{ 2185, 2184 },
		{ 2199, 2199 },
		{ 1648, 1648 },
		{ 1648, 1648 },
		{ 1994, 1994 },
		{ 1994, 1994 },
		{ 2145, 2145 },
		{ 2145, 2145 },
		{ 2205, 2205 },
		{ 2205, 2205 },
		{ 1593, 1593 },
		{ 1593, 1593 },
		{ 1584, 1584 },
		{ 1584, 1584 },
		{ 1596, 1596 },
		{ 1596, 1596 },
		{ 1544, 1544 },
		{ 1544, 1544 },
		{ 1623, 1623 },
		{ 1623, 1623 },
		{ 1587, 1587 },
		{ 1587, 1587 },
		{ 2008, 2008 },
		{ 2008, 2008 },
		{ 1955, 1954 },
		{ 1648, 1648 },
		{ 253, 216 },
		{ 1994, 1994 },
		{ 1990, 1989 },
		{ 2145, 2145 },
		{ 2200, 2199 },
		{ 2205, 2205 },
		{ 324, 278 },
		{ 1593, 1593 },
		{ 676, 610 },
		{ 1584, 1584 },
		{ 2253, 2251 },
		{ 1596, 1596 },
		{ 711, 647 },
		{ 1544, 1544 },
		{ 877, 824 },
		{ 1623, 1623 },
		{ 1015, 975 },
		{ 1587, 1587 },
		{ 562, 499 },
		{ 2008, 2008 },
		{ 1614, 1614 },
		{ 1614, 1614 },
		{ 980, 933 },
		{ 982, 935 },
		{ 1737, 1736 },
		{ 1248, 1247 },
		{ 1687, 1686 },
		{ 1210, 1201 },
		{ 1649, 1648 },
		{ 1067, 1033 },
		{ 1995, 1994 },
		{ 867, 814 },
		{ 2146, 2145 },
		{ 3157, 3152 },
		{ 2206, 2205 },
		{ 387, 342 },
		{ 1594, 1593 },
		{ 1178, 1161 },
		{ 1585, 1584 },
		{ 762, 705 },
		{ 1597, 1596 },
		{ 1217, 1208 },
		{ 1545, 1544 },
		{ 1614, 1614 },
		{ 1624, 1623 },
		{ 1180, 1164 },
		{ 1588, 1587 },
		{ 844, 784 },
		{ 2009, 2008 },
		{ 845, 785 },
		{ 1982, 1971 },
		{ 988, 944 },
		{ 226, 191 },
		{ 1051, 1016 },
		{ 2123, 2106 },
		{ 1930, 1913 },
		{ 1052, 1017 },
		{ 1030, 990 },
		{ 792, 733 },
		{ 1659, 1658 },
		{ 2187, 2168 },
		{ 1160, 1141 },
		{ 3118, 3116 },
		{ 1763, 1762 },
		{ 1575, 1557 },
		{ 819, 759 },
		{ 1713, 1712 },
		{ 970, 923 },
		{ 1822, 1821 },
		{ 1084, 1050 },
		{ 1193, 1181 },
		{ 1364, 1363 },
		{ 1615, 1614 },
		{ 1136, 1110 },
		{ 1195, 1184 },
		{ 829, 769 },
		{ 1950, 1935 },
		{ 921, 870 },
		{ 764, 707 },
		{ 351, 303 },
		{ 1200, 1189 },
		{ 894, 842 },
		{ 1202, 1191 },
		{ 198, 178 },
		{ 1091, 1057 },
		{ 926, 875 },
		{ 1216, 1207 },
		{ 1869, 1845 },
		{ 1308, 1305 },
		{ 1969, 1957 },
		{ 2162, 2148 },
		{ 507, 454 },
		{ 2198, 2186 },
		{ 632, 570 },
		{ 309, 264 },
		{ 2067, 2044 },
		{ 227, 192 },
		{ 2055, 2033 },
		{ 964, 916 },
		{ 2958, 2956 },
		{ 1988, 1981 },
		{ 865, 811 },
		{ 2685, 2685 },
		{ 2685, 2685 },
		{ 2881, 2881 },
		{ 2881, 2881 },
		{ 1835, 1835 },
		{ 1835, 1835 },
		{ 2660, 2660 },
		{ 2660, 2660 },
		{ 2755, 2755 },
		{ 2755, 2755 },
		{ 2805, 2805 },
		{ 2805, 2805 },
		{ 2689, 2689 },
		{ 2689, 2689 },
		{ 2810, 2810 },
		{ 2810, 2810 },
		{ 2813, 2813 },
		{ 2813, 2813 },
		{ 2690, 2690 },
		{ 2690, 2690 },
		{ 2274, 2274 },
		{ 2274, 2274 },
		{ 919, 868 },
		{ 2685, 2685 },
		{ 598, 537 },
		{ 2881, 2881 },
		{ 807, 748 },
		{ 1835, 1835 },
		{ 808, 749 },
		{ 2660, 2660 },
		{ 282, 240 },
		{ 2755, 2755 },
		{ 758, 701 },
		{ 2805, 2805 },
		{ 1204, 1193 },
		{ 2689, 2689 },
		{ 390, 346 },
		{ 2810, 2810 },
		{ 760, 703 },
		{ 2813, 2813 },
		{ 544, 483 },
		{ 2690, 2690 },
		{ 343, 296 },
		{ 2274, 2274 },
		{ 1209, 1200 },
		{ 2541, 2541 },
		{ 2541, 2541 },
		{ 2486, 2486 },
		{ 2486, 2486 },
		{ 2711, 2685 },
		{ 869, 816 },
		{ 2882, 2881 },
		{ 546, 485 },
		{ 1836, 1835 },
		{ 331, 284 },
		{ 2686, 2660 },
		{ 3023, 3021 },
		{ 2775, 2755 },
		{ 490, 436 },
		{ 2819, 2805 },
		{ 298, 253 },
		{ 2714, 2689 },
		{ 321, 275 },
		{ 2821, 2810 },
		{ 459, 412 },
		{ 2822, 2813 },
		{ 936, 885 },
		{ 2715, 2690 },
		{ 2541, 2541 },
		{ 2275, 2274 },
		{ 2486, 2486 },
		{ 322, 276 },
		{ 2818, 2818 },
		{ 2818, 2818 },
		{ 2954, 2954 },
		{ 2954, 2954 },
		{ 2427, 2427 },
		{ 2427, 2427 },
		{ 3019, 3019 },
		{ 3019, 3019 },
		{ 2544, 2544 },
		{ 2544, 2544 },
		{ 1752, 1752 },
		{ 1752, 1752 },
		{ 1303, 1303 },
		{ 1303, 1303 },
		{ 2698, 2698 },
		{ 2698, 2698 },
		{ 2829, 2829 },
		{ 2829, 2829 },
		{ 2830, 2830 },
		{ 2830, 2830 },
		{ 2831, 2831 },
		{ 2831, 2831 },
		{ 2572, 2541 },
		{ 2818, 2818 },
		{ 2487, 2486 },
		{ 2954, 2954 },
		{ 1219, 1210 },
		{ 2427, 2427 },
		{ 823, 763 },
		{ 3019, 3019 },
		{ 335, 288 },
		{ 2544, 2544 },
		{ 1303, 1302 },
		{ 1752, 1752 },
		{ 772, 714 },
		{ 1303, 1303 },
		{ 773, 715 },
		{ 2698, 2698 },
		{ 683, 616 },
		{ 2829, 2829 },
		{ 2220, 2217 },
		{ 2830, 2830 },
		{ 379, 334 },
		{ 2831, 2831 },
		{ 1010, 970 },
		{ 1812, 1812 },
		{ 1812, 1812 },
		{ 1290, 1290 },
		{ 1290, 1290 },
		{ 2827, 2818 },
		{ 262, 222 },
		{ 2955, 2954 },
		{ 727, 663 },
		{ 2428, 2427 },
		{ 585, 524 },
		{ 3020, 3019 },
		{ 729, 667 },
		{ 2575, 2544 },
		{ 948, 897 },
		{ 1753, 1752 },
		{ 888, 836 },
		{ 1304, 1303 },
		{ 365, 317 },
		{ 2723, 2698 },
		{ 890, 838 },
		{ 2834, 2829 },
		{ 688, 622 },
		{ 2835, 2830 },
		{ 1812, 1812 },
		{ 2836, 2831 },
		{ 1290, 1290 },
		{ 441, 396 },
		{ 2736, 2736 },
		{ 2736, 2736 },
		{ 2737, 2737 },
		{ 2737, 2737 },
		{ 2839, 2839 },
		{ 2839, 2839 },
		{ 1726, 1726 },
		{ 1726, 1726 },
		{ 2971, 2971 },
		{ 2971, 2971 },
		{ 2776, 2776 },
		{ 2776, 2776 },
		{ 2570, 2570 },
		{ 2570, 2570 },
		{ 1778, 1778 },
		{ 1778, 1778 },
		{ 2674, 2674 },
		{ 2674, 2674 },
		{ 2478, 2478 },
		{ 2478, 2478 },
		{ 2574, 2574 },
		{ 2574, 2574 },
		{ 1813, 1812 },
		{ 2736, 2736 },
		{ 1291, 1290 },
		{ 2737, 2737 },
		{ 653, 588 },
		{ 2839, 2839 },
		{ 691, 625 },
		{ 1726, 1726 },
		{ 692, 626 },
		{ 2971, 2971 },
		{ 693, 627 },
		{ 2776, 2776 },
		{ 654, 589 },
		{ 2570, 2570 },
		{ 1026, 986 },
		{ 1778, 1778 },
		{ 197, 177 },
		{ 2674, 2674 },
		{ 656, 591 },
		{ 2478, 2478 },
		{ 697, 634 },
		{ 2574, 2574 },
		{ 274, 232 },
		{ 2848, 2848 },
		{ 2848, 2848 },
		{ 1338, 1338 },
		{ 1338, 1338 },
		{ 2756, 2736 },
		{ 1247, 1246 },
		{ 2757, 2737 },
		{ 658, 593 },
		{ 2841, 2839 },
		{ 2885, 2883 },
		{ 1727, 1726 },
		{ 2976, 2973 },
		{ 2972, 2971 },
		{ 326, 280 },
		{ 2792, 2776 },
		{ 965, 918 },
		{ 2603, 2570 },
		{ 847, 789 },
		{ 1779, 1778 },
		{ 369, 323 },
		{ 2700, 2674 },
		{ 906, 855 },
		{ 2509, 2478 },
		{ 2848, 2848 },
		{ 2606, 2574 },
		{ 1338, 1338 },
		{ 796, 737 },
		{ 1702, 1702 },
		{ 1702, 1702 },
		{ 2787, 2787 },
		{ 2787, 2787 },
		{ 2925, 2925 },
		{ 2925, 2925 },
		{ 2599, 2599 },
		{ 2599, 2599 },
		{ 2747, 2747 },
		{ 2747, 2747 },
		{ 1381, 1381 },
		{ 1381, 1381 },
		{ 2791, 2791 },
		{ 2791, 2791 },
		{ 2793, 2793 },
		{ 2793, 2793 },
		{ 2794, 2794 },
		{ 2794, 2794 },
		{ 2579, 2579 },
		{ 2579, 2579 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 2849, 2848 },
		{ 1702, 1702 },
		{ 1339, 1338 },
		{ 2787, 2787 },
		{ 797, 738 },
		{ 2925, 2925 },
		{ 972, 925 },
		{ 2599, 2599 },
		{ 973, 926 },
		{ 2747, 2747 },
		{ 1041, 1005 },
		{ 1381, 1381 },
		{ 1783, 1780 },
		{ 2791, 2791 },
		{ 386, 341 },
		{ 2793, 2793 },
		{ 505, 451 },
		{ 2794, 2794 },
		{ 624, 562 },
		{ 2579, 2579 },
		{ 706, 642 },
		{ 3060, 3060 },
		{ 566, 504 },
		{ 2796, 2796 },
		{ 2796, 2796 },
		{ 2797, 2797 },
		{ 2797, 2797 },
		{ 1703, 1702 },
		{ 1119, 1091 },
		{ 2803, 2787 },
		{ 247, 211 },
		{ 2926, 2925 },
		{ 857, 801 },
		{ 2600, 2599 },
		{ 981, 934 },
		{ 2767, 2747 },
		{ 916, 865 },
		{ 1382, 1381 },
		{ 231, 195 },
		{ 2807, 2791 },
		{ 408, 364 },
		{ 2808, 2793 },
		{ 1053, 1018 },
		{ 2809, 2794 },
		{ 3194, 3194 },
		{ 2610, 2579 },
		{ 2796, 2796 },
		{ 3061, 3060 },
		{ 2797, 2797 },
		{ 2196, 2183 },
		{ 1353, 1353 },
		{ 1353, 1353 },
		{ 1676, 1676 },
		{ 1676, 1676 },
		{ 3202, 3202 },
		{ 3206, 3206 },
		{ 1610, 1594 },
		{ 2197, 2185 },
		{ 1967, 1955 },
		{ 1650, 1649 },
		{ 2160, 2146 },
		{ 1625, 1615 },
		{ 2010, 2009 },
		{ 1996, 1990 },
		{ 1631, 1624 },
		{ 1607, 1585 },
		{ 2231, 2229 },
		{ 3194, 3194 },
		{ 2207, 2200 },
		{ 1563, 1545 },
		{ 1999, 1995 },
		{ 1986, 1978 },
		{ 2811, 2796 },
		{ 1353, 1353 },
		{ 2812, 2797 },
		{ 1676, 1676 },
		{ 1611, 1597 },
		{ 3202, 3202 },
		{ 3206, 3206 },
		{ 2192, 2175 },
		{ 2212, 2206 },
		{ 1987, 1980 },
		{ 2193, 2177 },
		{ 1609, 1591 },
		{ 2194, 2179 },
		{ 1606, 1582 },
		{ 2195, 2181 },
		{ 1608, 1588 },
		{ 1664, 1663 },
		{ 1665, 1664 },
		{ 1692, 1691 },
		{ 1827, 1826 },
		{ 3168, 3167 },
		{ 3197, 3194 },
		{ 3169, 3168 },
		{ 1828, 1827 },
		{ 1693, 1692 },
		{ 1742, 1741 },
		{ 3092, 3091 },
		{ 1354, 1353 },
		{ 1743, 1742 },
		{ 1677, 1676 },
		{ 1370, 1369 },
		{ 3204, 3202 },
		{ 3207, 3206 },
		{ 1641, 1637 },
		{ 1718, 1717 },
		{ 1719, 1718 },
		{ 1369, 1368 },
		{ 1637, 1632 },
		{ 1638, 1633 },
		{ 1413, 1412 },
		{ 1768, 1767 },
		{ 1769, 1768 },
		{ 2855, 2855 },
		{ 2852, 2855 },
		{ 165, 165 },
		{ 162, 165 },
		{ 1961, 1949 },
		{ 1962, 1949 },
		{ 1939, 1924 },
		{ 1940, 1924 },
		{ 2015, 2015 },
		{ 2241, 2241 },
		{ 164, 160 },
		{ 2860, 2856 },
		{ 2300, 2276 },
		{ 170, 166 },
		{ 2246, 2242 },
		{ 163, 160 },
		{ 2859, 2856 },
		{ 2299, 2276 },
		{ 169, 166 },
		{ 2245, 2242 },
		{ 2247, 2244 },
		{ 2020, 2016 },
		{ 2854, 2850 },
		{ 2855, 2855 },
		{ 90, 72 },
		{ 165, 165 },
		{ 2019, 2016 },
		{ 2853, 2850 },
		{ 2249, 2248 },
		{ 89, 72 },
		{ 3103, 3095 },
		{ 2015, 2015 },
		{ 2241, 2241 },
		{ 2361, 2329 },
		{ 2092, 2071 },
		{ 121, 105 },
		{ 171, 168 },
		{ 2861, 2858 },
		{ 2856, 2855 },
		{ 2863, 2862 },
		{ 166, 165 },
		{ 173, 172 },
		{ 1899, 1880 },
		{ 2021, 2018 },
		{ 2023, 2022 },
		{ 1974, 1964 },
		{ 2016, 2015 },
		{ 2242, 2241 },
		{ 2236, 2235 },
		{ 2090, 2069 },
		{ 2244, 2240 },
		{ 2022, 2020 },
		{ 2071, 2049 },
		{ 172, 170 },
		{ 2248, 2246 },
		{ 2329, 2300 },
		{ 2858, 2854 },
		{ 2018, 2014 },
		{ 168, 164 },
		{ 1880, 1858 },
		{ 2862, 2860 },
		{ 0, 2562 },
		{ 0, 2994 },
		{ 0, 1277 },
		{ 2827, 2827 },
		{ 2827, 2827 },
		{ 0, 2909 },
		{ 0, 2263 },
		{ 0, 2999 },
		{ 2711, 2711 },
		{ 2711, 2711 },
		{ 0, 2629 },
		{ 2023, 2023 },
		{ 2024, 2023 },
		{ 2714, 2714 },
		{ 2714, 2714 },
		{ 0, 2765 },
		{ 0, 2916 },
		{ 2834, 2834 },
		{ 2834, 2834 },
		{ 2835, 2835 },
		{ 2835, 2835 },
		{ 2836, 2836 },
		{ 2836, 2836 },
		{ 2715, 2715 },
		{ 2715, 2715 },
		{ 0, 3010 },
		{ 2827, 2827 },
		{ 2767, 2767 },
		{ 2767, 2767 },
		{ 0, 2539 },
		{ 0, 2632 },
		{ 2711, 2711 },
		{ 2841, 2841 },
		{ 2841, 2841 },
		{ 2023, 2023 },
		{ 0, 2672 },
		{ 2714, 2714 },
		{ 0, 1298 },
		{ 2249, 2249 },
		{ 2250, 2249 },
		{ 2834, 2834 },
		{ 0, 2774 },
		{ 2835, 2835 },
		{ 0, 1329 },
		{ 2836, 2836 },
		{ 0, 3024 },
		{ 2715, 2715 },
		{ 2775, 2775 },
		{ 2775, 2775 },
		{ 0, 2777 },
		{ 2767, 2767 },
		{ 2849, 2849 },
		{ 2849, 2849 },
		{ 2600, 2600 },
		{ 2600, 2600 },
		{ 2841, 2841 },
		{ 0, 2678 },
		{ 0, 3028 },
		{ 0, 2937 },
		{ 0, 2780 },
		{ 0, 2781 },
		{ 2249, 2249 },
		{ 0, 3032 },
		{ 0, 2726 },
		{ 0, 1787 },
		{ 0, 3035 },
		{ 2572, 2572 },
		{ 2572, 2572 },
		{ 0, 2729 },
		{ 0, 3040 },
		{ 2775, 2775 },
		{ 2863, 2863 },
		{ 2864, 2863 },
		{ 0, 2946 },
		{ 2849, 2849 },
		{ 0, 1345 },
		{ 2600, 2600 },
		{ 0, 1798 },
		{ 2606, 2606 },
		{ 2606, 2606 },
		{ 2575, 2575 },
		{ 2575, 2575 },
		{ 0, 2644 },
		{ 0, 2871 },
		{ 2686, 2686 },
		{ 2686, 2686 },
		{ 2792, 2792 },
		{ 2792, 2792 },
		{ 0, 2645 },
		{ 2572, 2572 },
		{ 0, 2937 },
		{ 0, 3051 },
		{ 0, 2498 },
		{ 0, 2499 },
		{ 2863, 2863 },
		{ 2610, 2610 },
		{ 2610, 2610 },
		{ 0, 2500 },
		{ 0, 2550 },
		{ 0, 2962 },
		{ 0, 1313 },
		{ 2606, 2606 },
		{ 0, 1324 },
		{ 2575, 2575 },
		{ 0, 2615 },
		{ 0, 2654 },
		{ 0, 2886 },
		{ 2686, 2686 },
		{ 0, 1803 },
		{ 2792, 2792 },
		{ 2803, 2803 },
		{ 2803, 2803 },
		{ 173, 173 },
		{ 174, 173 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 0, 2891 },
		{ 0, 3071 },
		{ 2610, 2610 },
		{ 2700, 2700 },
		{ 2700, 2700 },
		{ 0, 2977 },
		{ 2428, 2428 },
		{ 2428, 2428 },
		{ 0, 2465 },
		{ 0, 2394 },
		{ 0, 2981 },
		{ 2509, 2509 },
		{ 2509, 2509 },
		{ 0, 2623 },
		{ 0, 2510 },
		{ 2819, 2819 },
		{ 2819, 2819 },
		{ 2803, 2803 },
		{ 0, 3082 },
		{ 173, 173 },
		{ 0, 2820 },
		{ 2809, 2809 },
		{ 0, 2510 },
		{ 2821, 2821 },
		{ 2821, 2821 },
		{ 0, 2902 },
		{ 2700, 2700 },
		{ 2822, 2822 },
		{ 2822, 2822 },
		{ 2428, 2428 },
		{ 0, 2991 },
		{ 2757, 2757 },
		{ 2757, 2757 },
		{ 1558, 1558 },
		{ 2509, 2509 },
		{ 1256, 1256 },
		{ 1400, 1399 },
		{ 1391, 1390 },
		{ 2819, 2819 },
		{ 0, 1857 },
		{ 0, 2394 },
		{ 2243, 2239 },
		{ 2857, 2853 },
		{ 3107, 3103 },
		{ 0, 2048 },
		{ 2017, 2019 },
		{ 2821, 2821 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2822, 2822 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2757, 2757 },
		{ 0, 0 },
		{ 1558, 1558 },
		{ 0, 0 },
		{ 1256, 1256 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -70, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 46, 572, 0 },
		{ -179, 3020, 0 },
		{ 5, 0, 0 },
		{ -1255, 1017, -31 },
		{ 7, 0, -31 },
		{ -1259, 2037, -33 },
		{ 9, 0, -33 },
		{ -1272, 3346, 155 },
		{ 11, 0, 155 },
		{ -1293, 3356, 159 },
		{ 13, 0, 159 },
		{ -1306, 3351, 167 },
		{ 15, 0, 167 },
		{ -1341, 3487, 0 },
		{ 17, 0, 0 },
		{ -1356, 3358, 151 },
		{ 19, 0, 151 },
		{ -1384, 3369, 23 },
		{ 21, 0, 23 },
		{ -1426, 230, 0 },
		{ 23, 0, 0 },
		{ -1652, 3488, 0 },
		{ 25, 0, 0 },
		{ -1679, 3344, 0 },
		{ 27, 0, 0 },
		{ -1705, 3345, 0 },
		{ 29, 0, 0 },
		{ -1729, 3352, 0 },
		{ 31, 0, 0 },
		{ -1755, 3367, 0 },
		{ 33, 0, 0 },
		{ -1781, 3368, 171 },
		{ 35, 0, 171 },
		{ -1815, 3494, 278 },
		{ 37, 0, 278 },
		{ 40, 129, 0 },
		{ -1851, 344, 0 },
		{ 42, 127, 0 },
		{ -2040, 116, 0 },
		{ -2252, 3495, 0 },
		{ 43, 0, 0 },
		{ 46, 14, 0 },
		{ -88, 458, 0 },
		{ -2866, 3362, 163 },
		{ 47, 0, 163 },
		{ -2884, 3491, 186 },
		{ 49, 0, 186 },
		{ 2928, 1438, 0 },
		{ 51, 0, 0 },
		{ -2930, 3496, 284 },
		{ 53, 0, 284 },
		{ -2957, 3497, 189 },
		{ 55, 0, 189 },
		{ -2975, 3350, 182 },
		{ 57, 0, 182 },
		{ -3022, 3489, 175 },
		{ 59, 0, 175 },
		{ -3063, 3373, 181 },
		{ 61, 0, 181 },
		{ 46, 1, 0 },
		{ 63, 0, 0 },
		{ -3114, 1855, 0 },
		{ 65, 0, 0 },
		{ -3124, 2035, 45 },
		{ 67, 0, 45 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 439 },
		{ 3095, 4994, 446 },
		{ 0, 0, 256 },
		{ 0, 0, 258 },
		{ 159, 1223, 275 },
		{ 159, 1352, 275 },
		{ 159, 1253, 275 },
		{ 159, 1260, 275 },
		{ 159, 1260, 275 },
		{ 159, 1264, 275 },
		{ 159, 1269, 275 },
		{ 159, 1289, 275 },
		{ 3188, 2993, 446 },
		{ 159, 1300, 275 },
		{ 3188, 1885, 274 },
		{ 104, 2865, 446 },
		{ 159, 0, 275 },
		{ 0, 0, 446 },
		{ -89, 7, 252 },
		{ -90, 1, 0 },
		{ 159, 1291, 275 },
		{ 159, 1325, 275 },
		{ 159, 722, 275 },
		{ 159, 772, 275 },
		{ 159, 754, 275 },
		{ 159, 754, 275 },
		{ 159, 761, 275 },
		{ 159, 776, 275 },
		{ 159, 769, 275 },
		{ 3147, 2542, 0 },
		{ 159, 759, 275 },
		{ 3188, 1973, 271 },
		{ 119, 1486, 0 },
		{ 3188, 1940, 272 },
		{ 3095, 5000, 0 },
		{ 159, 794, 275 },
		{ 159, 792, 275 },
		{ 159, 793, 275 },
		{ 159, 789, 275 },
		{ 159, 0, 263 },
		{ 159, 809, 275 },
		{ 159, 839, 275 },
		{ 159, 823, 275 },
		{ 159, 828, 275 },
		{ 3185, 3132, 0 },
		{ 159, 835, 275 },
		{ 133, 1391, 0 },
		{ 119, 0, 0 },
		{ 3097, 2873, 273 },
		{ 135, 1472, 0 },
		{ 0, 0, 254 },
		{ 159, 839, 259 },
		{ 159, 841, 275 },
		{ 159, 833, 275 },
		{ 159, 838, 275 },
		{ 159, 836, 275 },
		{ 159, 829, 275 },
		{ 159, 0, 266 },
		{ 159, 830, 275 },
		{ 0, 0, 268 },
		{ 159, 836, 275 },
		{ 133, 0, 0 },
		{ 3097, 2774, 271 },
		{ 135, 0, 0 },
		{ 3097, 2799, 272 },
		{ 159, 852, 275 },
		{ 159, 849, 275 },
		{ 159, 850, 275 },
		{ 159, 880, 275 },
		{ 159, 975, 275 },
		{ 159, 0, 265 },
		{ 159, 1082, 275 },
		{ 159, 1129, 275 },
		{ 159, 1148, 275 },
		{ 159, 0, 261 },
		{ 159, 1227, 275 },
		{ 159, 0, 262 },
		{ 159, 0, 264 },
		{ 159, 1197, 275 },
		{ 159, 1205, 275 },
		{ 159, 0, 260 },
		{ 159, 1218, 275 },
		{ 159, 0, 267 },
		{ 159, 772, 275 },
		{ 159, 1240, 275 },
		{ 0, 0, 270 },
		{ 159, 1236, 275 },
		{ 159, 1238, 275 },
		{ 3205, 1309, 269 },
		{ 3095, 4980, 446 },
		{ 165, 0, 256 },
		{ 0, 0, 257 },
		{ -163, 20, 252 },
		{ -164, 5028, 0 },
		{ 3151, 5005, 0 },
		{ 3095, 4983, 0 },
		{ 0, 0, 253 },
		{ 3095, 5001, 0 },
		{ -169, 22, 0 },
		{ -170, 5023, 0 },
		{ 173, 0, 254 },
		{ 3095, 5006, 0 },
		{ 3151, 5176, 0 },
		{ 0, 0, 255 },
		{ 3152, 1776, 149 },
		{ 2152, 4382, 149 },
		{ 3021, 4777, 149 },
		{ 3152, 4574, 149 },
		{ 0, 0, 149 },
		{ 3137, 3633, 0 },
		{ 2167, 3203, 0 },
		{ 3137, 3833, 0 },
		{ 3137, 3480, 0 },
		{ 3125, 3565, 0 },
		{ 2152, 4373, 0 },
		{ 3142, 3474, 0 },
		{ 2152, 4333, 0 },
		{ 3112, 3759, 0 },
		{ 3146, 3446, 0 },
		{ 3112, 3520, 0 },
		{ 3152, 4543, 0 },
		{ 2956, 4588, 0 },
		{ 3142, 3499, 0 },
		{ 2167, 3968, 0 },
		{ 3021, 4871, 0 },
		{ 2098, 3688, 0 },
		{ 2098, 3704, 0 },
		{ 2120, 3328, 0 },
		{ 2101, 4039, 0 },
		{ 2082, 4109, 0 },
		{ 2101, 4061, 0 },
		{ 2120, 3332, 0 },
		{ 2120, 3363, 0 },
		{ 3142, 3540, 0 },
		{ 3062, 4166, 0 },
		{ 2152, 4353, 0 },
		{ 1225, 4275, 0 },
		{ 2098, 3720, 0 },
		{ 2120, 3375, 0 },
		{ 2120, 3377, 0 },
		{ 3021, 4863, 0 },
		{ 2098, 3717, 0 },
		{ 3125, 3978, 0 },
		{ 3137, 3827, 0 },
		{ 2167, 3932, 0 },
		{ 2251, 4490, 0 },
		{ 3021, 3855, 0 },
		{ 3062, 4171, 0 },
		{ 3112, 3664, 0 },
		{ 3112, 3805, 0 },
		{ 2060, 3485, 0 },
		{ 3021, 4717, 0 },
		{ 3137, 3886, 0 },
		{ 3062, 3849, 0 },
		{ 2167, 3964, 0 },
		{ 2120, 3389, 0 },
		{ 2120, 3304, 0 },
		{ 3146, 3580, 0 },
		{ 3125, 3917, 0 },
		{ 2060, 3507, 0 },
		{ 2082, 4112, 0 },
		{ 3021, 4783, 0 },
		{ 2152, 4302, 0 },
		{ 2152, 4309, 0 },
		{ 3137, 3876, 0 },
		{ 2120, 3305, 0 },
		{ 2152, 4361, 0 },
		{ 3137, 3891, 0 },
		{ 2251, 4430, 0 },
		{ 3021, 4625, 0 },
		{ 3146, 3601, 0 },
		{ 3112, 3778, 0 },
		{ 2120, 3306, 0 },
		{ 3146, 3536, 0 },
		{ 3137, 3837, 0 },
		{ 1225, 4264, 0 },
		{ 2082, 4075, 0 },
		{ 3062, 4170, 0 },
		{ 2167, 3855, 0 },
		{ 1112, 3452, 0 },
		{ 3137, 3885, 0 },
		{ 3125, 4034, 0 },
		{ 3021, 4655, 0 },
		{ 2251, 4453, 0 },
		{ 2120, 3309, 0 },
		{ 2167, 3956, 0 },
		{ 3146, 3569, 0 },
		{ 2152, 4389, 0 },
		{ 2060, 3511, 0 },
		{ 2060, 3515, 0 },
		{ 2152, 4316, 0 },
		{ 3112, 3752, 0 },
		{ 2120, 3310, 0 },
		{ 2956, 4586, 0 },
		{ 3125, 4004, 0 },
		{ 3146, 3627, 0 },
		{ 2098, 3687, 0 },
		{ 2188, 3854, 0 },
		{ 2120, 3311, 0 },
		{ 3062, 4202, 0 },
		{ 3112, 3758, 0 },
		{ 2152, 4326, 0 },
		{ 2251, 4445, 0 },
		{ 2152, 4330, 0 },
		{ 3021, 4657, 0 },
		{ 3021, 4666, 0 },
		{ 2082, 4073, 0 },
		{ 2251, 4496, 0 },
		{ 2120, 3312, 0 },
		{ 3021, 4797, 0 },
		{ 3062, 4196, 0 },
		{ 2082, 4084, 0 },
		{ 3062, 4136, 0 },
		{ 3021, 4649, 0 },
		{ 2098, 3713, 0 },
		{ 3112, 3791, 0 },
		{ 2152, 4306, 0 },
		{ 3021, 4697, 0 },
		{ 1225, 4281, 0 },
		{ 3062, 4181, 0 },
		{ 1112, 3456, 0 },
		{ 2188, 4247, 0 },
		{ 2101, 4056, 0 },
		{ 3112, 3741, 0 },
		{ 2120, 3313, 0 },
		{ 3021, 4637, 0 },
		{ 2152, 4366, 0 },
		{ 2120, 3314, 0 },
		{ 0, 0, 78 },
		{ 3137, 3665, 0 },
		{ 3146, 3566, 0 },
		{ 2152, 4294, 0 },
		{ 3152, 4570, 0 },
		{ 2152, 4300, 0 },
		{ 2120, 3315, 0 },
		{ 2120, 3316, 0 },
		{ 3146, 3597, 0 },
		{ 2098, 3679, 0 },
		{ 2082, 4074, 0 },
		{ 3146, 3599, 0 },
		{ 2120, 3317, 0 },
		{ 3062, 4225, 0 },
		{ 2152, 4360, 0 },
		{ 3137, 3864, 0 },
		{ 3137, 3870, 0 },
		{ 2101, 4037, 0 },
		{ 3021, 4729, 0 },
		{ 3112, 3771, 0 },
		{ 879, 3468, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2098, 3710, 0 },
		{ 3021, 4803, 0 },
		{ 3062, 4221, 0 },
		{ 2082, 4097, 0 },
		{ 3146, 3626, 0 },
		{ 3112, 3806, 0 },
		{ 0, 0, 76 },
		{ 2120, 3318, 0 },
		{ 2098, 3719, 0 },
		{ 3146, 3628, 0 },
		{ 3062, 4201, 0 },
		{ 3146, 3635, 0 },
		{ 3021, 4709, 0 },
		{ 3112, 3763, 0 },
		{ 1225, 4262, 0 },
		{ 2082, 4098, 0 },
		{ 2098, 3682, 0 },
		{ 3125, 4007, 0 },
		{ 2152, 4385, 0 },
		{ 3021, 4847, 0 },
		{ 3152, 4526, 0 },
		{ 3112, 3774, 0 },
		{ 0, 0, 70 },
		{ 3112, 3776, 0 },
		{ 3021, 4631, 0 },
		{ 1225, 4273, 0 },
		{ 3062, 4182, 0 },
		{ 2152, 4304, 0 },
		{ 0, 0, 81 },
		{ 3146, 3638, 0 },
		{ 3112, 3786, 0 },
		{ 3137, 3826, 0 },
		{ 3137, 3892, 0 },
		{ 2120, 3322, 0 },
		{ 3062, 4160, 0 },
		{ 2152, 4340, 0 },
		{ 2120, 3323, 0 },
		{ 2098, 3706, 0 },
		{ 2133, 3443, 0 },
		{ 3125, 3979, 0 },
		{ 3146, 3570, 0 },
		{ 3112, 3757, 0 },
		{ 3021, 4873, 0 },
		{ 3146, 3572, 0 },
		{ 2120, 3324, 0 },
		{ 3062, 4209, 0 },
		{ 2152, 4296, 0 },
		{ 3146, 3592, 0 },
		{ 3112, 3764, 0 },
		{ 3062, 4146, 0 },
		{ 1052, 4234, 0 },
		{ 0, 0, 8 },
		{ 2098, 3665, 0 },
		{ 2101, 4058, 0 },
		{ 3062, 4167, 0 },
		{ 2133, 3448, 0 },
		{ 2120, 3325, 0 },
		{ 2251, 4432, 0 },
		{ 2152, 4337, 0 },
		{ 2098, 3684, 0 },
		{ 2152, 4341, 0 },
		{ 2152, 4342, 0 },
		{ 2101, 4043, 0 },
		{ 2120, 3326, 0 },
		{ 3146, 3602, 0 },
		{ 3146, 3483, 0 },
		{ 2152, 4371, 0 },
		{ 3142, 3534, 0 },
		{ 3112, 3732, 0 },
		{ 1225, 4284, 0 },
		{ 3125, 3990, 0 },
		{ 2120, 3327, 0 },
		{ 2167, 3272, 0 },
		{ 2120, 3329, 0 },
		{ 3021, 4738, 0 },
		{ 1225, 4280, 0 },
		{ 2167, 3919, 0 },
		{ 3152, 3436, 0 },
		{ 2133, 3450, 0 },
		{ 2101, 4046, 0 },
		{ 2098, 3724, 0 },
		{ 3146, 3652, 0 },
		{ 2152, 4331, 0 },
		{ 0, 0, 123 },
		{ 2120, 3331, 0 },
		{ 2167, 3927, 0 },
		{ 2059, 3461, 0 },
		{ 3062, 4219, 0 },
		{ 3137, 3849, 0 },
		{ 3125, 3993, 0 },
		{ 3021, 4659, 0 },
		{ 2152, 4355, 0 },
		{ 0, 0, 7 },
		{ 2101, 4054, 0 },
		{ 0, 0, 6 },
		{ 3062, 4159, 0 },
		{ 0, 0, 128 },
		{ 3125, 3998, 0 },
		{ 2152, 4370, 0 },
		{ 3152, 1844, 0 },
		{ 2120, 3333, 0 },
		{ 3125, 4018, 0 },
		{ 3137, 3874, 0 },
		{ 0, 0, 132 },
		{ 2120, 3335, 0 },
		{ 2152, 4293, 0 },
		{ 3152, 3456, 0 },
		{ 2152, 4295, 0 },
		{ 2251, 4451, 0 },
		{ 2167, 3944, 0 },
		{ 0, 0, 77 },
		{ 2082, 4089, 0 },
		{ 2120, 3336, 115 },
		{ 2120, 3337, 116 },
		{ 3021, 4653, 0 },
		{ 3062, 4215, 0 },
		{ 3112, 3809, 0 },
		{ 3137, 3898, 0 },
		{ 3137, 3902, 0 },
		{ 3112, 3811, 0 },
		{ 1225, 4288, 0 },
		{ 3137, 3858, 0 },
		{ 3112, 3812, 0 },
		{ 3142, 3547, 0 },
		{ 2167, 3967, 0 },
		{ 3062, 4168, 0 },
		{ 2152, 4344, 0 },
		{ 2120, 3338, 0 },
		{ 3146, 3614, 0 },
		{ 3021, 4849, 0 },
		{ 0, 0, 114 },
		{ 3062, 4176, 0 },
		{ 2956, 4583, 0 },
		{ 3062, 4180, 0 },
		{ 2167, 3920, 0 },
		{ 3112, 3753, 0 },
		{ 3062, 4193, 0 },
		{ 0, 0, 9 },
		{ 2120, 3339, 0 },
		{ 3062, 4197, 0 },
		{ 2133, 3435, 0 },
		{ 2188, 4240, 0 },
		{ 0, 0, 112 },
		{ 2098, 3658, 0 },
		{ 3125, 3983, 0 },
		{ 3137, 3839, 0 },
		{ 2167, 3857, 0 },
		{ 3125, 3449, 0 },
		{ 3062, 4224, 0 },
		{ 3142, 3509, 0 },
		{ 3062, 4226, 0 },
		{ 3142, 3500, 0 },
		{ 3137, 3865, 0 },
		{ 2152, 4320, 0 },
		{ 3146, 3636, 0 },
		{ 3112, 3775, 0 },
		{ 3142, 3471, 0 },
		{ 3125, 3982, 0 },
		{ 3146, 3644, 0 },
		{ 3062, 4137, 0 },
		{ 3146, 3534, 0 },
		{ 3021, 4635, 0 },
		{ 2120, 3340, 0 },
		{ 3021, 4647, 0 },
		{ 3112, 3804, 0 },
		{ 2152, 4345, 0 },
		{ 3137, 3832, 0 },
		{ 3137, 3835, 0 },
		{ 2082, 4083, 0 },
		{ 2098, 3707, 0 },
		{ 2120, 3344, 102 },
		{ 3112, 3808, 0 },
		{ 2167, 3957, 0 },
		{ 2120, 3345, 0 },
		{ 2120, 3350, 0 },
		{ 3142, 3523, 0 },
		{ 2167, 3913, 0 },
		{ 2251, 4508, 0 },
		{ 2120, 3351, 0 },
		{ 2098, 3722, 0 },
		{ 0, 0, 111 },
		{ 2251, 4434, 0 },
		{ 3021, 4855, 0 },
		{ 3146, 3586, 0 },
		{ 3146, 3589, 0 },
		{ 0, 0, 125 },
		{ 0, 0, 127 },
		{ 3125, 4028, 0 },
		{ 2167, 3942, 0 },
		{ 2098, 3660, 0 },
		{ 2152, 3575, 0 },
		{ 3146, 3596, 0 },
		{ 2152, 4315, 0 },
		{ 2120, 3352, 0 },
		{ 2152, 4318, 0 },
		{ 3062, 4175, 0 },
		{ 3125, 3985, 0 },
		{ 2120, 3354, 0 },
		{ 2188, 4251, 0 },
		{ 3142, 3551, 0 },
		{ 2251, 4418, 0 },
		{ 2120, 3356, 0 },
		{ 3021, 4721, 0 },
		{ 2098, 3700, 0 },
		{ 980, 4125, 0 },
		{ 3146, 3611, 0 },
		{ 3125, 4011, 0 },
		{ 2167, 3921, 0 },
		{ 2251, 4455, 0 },
		{ 3146, 3613, 0 },
		{ 2060, 3509, 0 },
		{ 2120, 3357, 0 },
		{ 3062, 4220, 0 },
		{ 3137, 3889, 0 },
		{ 2098, 3711, 0 },
		{ 3021, 4619, 0 },
		{ 3146, 3625, 0 },
		{ 2167, 3959, 0 },
		{ 2133, 3436, 0 },
		{ 3112, 3807, 0 },
		{ 2098, 3718, 0 },
		{ 2167, 3969, 0 },
		{ 2101, 4048, 0 },
		{ 3152, 3533, 0 },
		{ 2120, 3358, 0 },
		{ 0, 0, 71 },
		{ 2120, 3360, 0 },
		{ 3137, 3866, 0 },
		{ 3112, 3815, 0 },
		{ 3137, 3872, 0 },
		{ 3112, 3820, 0 },
		{ 2120, 3361, 117 },
		{ 2082, 4123, 0 },
		{ 2167, 3945, 0 },
		{ 2101, 4049, 0 },
		{ 3112, 3739, 0 },
		{ 2098, 3726, 0 },
		{ 2098, 3655, 0 },
		{ 2082, 4086, 0 },
		{ 2101, 4060, 0 },
		{ 3021, 4851, 0 },
		{ 3137, 3841, 0 },
		{ 3142, 3543, 0 },
		{ 3062, 4222, 0 },
		{ 3146, 3639, 0 },
		{ 2098, 3664, 0 },
		{ 0, 0, 129 },
		{ 2120, 3362, 0 },
		{ 2956, 4585, 0 },
		{ 2101, 4047, 0 },
		{ 3146, 3646, 0 },
		{ 3125, 4030, 0 },
		{ 0, 0, 124 },
		{ 0, 0, 113 },
		{ 2098, 3681, 0 },
		{ 3112, 3766, 0 },
		{ 3146, 3649, 0 },
		{ 2167, 3175, 0 },
		{ 3152, 3490, 0 },
		{ 3062, 4172, 0 },
		{ 3125, 3984, 0 },
		{ 2120, 3367, 0 },
		{ 3062, 4177, 0 },
		{ 2082, 4110, 0 },
		{ 3137, 3875, 0 },
		{ 2152, 4298, 0 },
		{ 3021, 4765, 0 },
		{ 3021, 4773, 0 },
		{ 2098, 3699, 0 },
		{ 3021, 4779, 0 },
		{ 3062, 4183, 0 },
		{ 3021, 4791, 0 },
		{ 3112, 3785, 0 },
		{ 3125, 4001, 0 },
		{ 2120, 3368, 0 },
		{ 2152, 4313, 0 },
		{ 3112, 3789, 0 },
		{ 2120, 3369, 0 },
		{ 3112, 3801, 0 },
		{ 2152, 4319, 0 },
		{ 3062, 4212, 0 },
		{ 2152, 4324, 0 },
		{ 3112, 3802, 0 },
		{ 2152, 4329, 0 },
		{ 2098, 3705, 0 },
		{ 3125, 4032, 0 },
		{ 2120, 3371, 0 },
		{ 3152, 4440, 0 },
		{ 2251, 4498, 0 },
		{ 2152, 4336, 0 },
		{ 2101, 4040, 0 },
		{ 2152, 4338, 0 },
		{ 2101, 4041, 0 },
		{ 3137, 3830, 0 },
		{ 3021, 4705, 0 },
		{ 3137, 3862, 0 },
		{ 0, 0, 104 },
		{ 3146, 3575, 0 },
		{ 3062, 4147, 0 },
		{ 3062, 4152, 0 },
		{ 3021, 4733, 0 },
		{ 2120, 3372, 0 },
		{ 2120, 3373, 0 },
		{ 3021, 4767, 0 },
		{ 3021, 4769, 0 },
		{ 3021, 4771, 0 },
		{ 2101, 4050, 0 },
		{ 2098, 3712, 0 },
		{ 0, 0, 136 },
		{ 3137, 3871, 0 },
		{ 0, 0, 126 },
		{ 0, 0, 130 },
		{ 3021, 4781, 0 },
		{ 2251, 4428, 0 },
		{ 1112, 3458, 0 },
		{ 2120, 3374, 0 },
		{ 3062, 3279, 0 },
		{ 3112, 3817, 0 },
		{ 2101, 4038, 0 },
		{ 1225, 4271, 0 },
		{ 3021, 4853, 0 },
		{ 3137, 3877, 0 },
		{ 3137, 3880, 0 },
		{ 3137, 3884, 0 },
		{ 3125, 4019, 0 },
		{ 2251, 4502, 0 },
		{ 2188, 4239, 0 },
		{ 3125, 4026, 0 },
		{ 3142, 3544, 0 },
		{ 2082, 4085, 0 },
		{ 1225, 4267, 0 },
		{ 3146, 3598, 0 },
		{ 2082, 4087, 0 },
		{ 2098, 3721, 0 },
		{ 2120, 3376, 0 },
		{ 2101, 4052, 0 },
		{ 2082, 4106, 0 },
		{ 2152, 4321, 0 },
		{ 2188, 4245, 0 },
		{ 2167, 3926, 0 },
		{ 3112, 3742, 0 },
		{ 3021, 4719, 0 },
		{ 2167, 3928, 0 },
		{ 0, 0, 65 },
		{ 0, 0, 66 },
		{ 3021, 4723, 0 },
		{ 3112, 3744, 0 },
		{ 0, 0, 75 },
		{ 0, 0, 121 },
		{ 2060, 3506, 0 },
		{ 3142, 3552, 0 },
		{ 2098, 3728, 0 },
		{ 3142, 3555, 0 },
		{ 3146, 3612, 0 },
		{ 3062, 4149, 0 },
		{ 3112, 3760, 0 },
		{ 0, 0, 106 },
		{ 3112, 3761, 0 },
		{ 0, 0, 108 },
		{ 3137, 3868, 0 },
		{ 3112, 3762, 0 },
		{ 3125, 4012, 0 },
		{ 2152, 4359, 0 },
		{ 0, 0, 134 },
		{ 2133, 3446, 0 },
		{ 2133, 3447, 0 },
		{ 3146, 3615, 0 },
		{ 1225, 4283, 0 },
		{ 2188, 3982, 0 },
		{ 3112, 3768, 0 },
		{ 980, 4128, 0 },
		{ 2082, 4116, 0 },
		{ 0, 0, 122 },
		{ 0, 0, 135 },
		{ 3112, 3769, 0 },
		{ 3112, 3770, 0 },
		{ 0, 0, 148 },
		{ 2098, 3669, 0 },
		{ 3152, 4144, 0 },
		{ 3021, 4627, 0 },
		{ 1225, 4278, 0 },
		{ 3021, 4633, 0 },
		{ 2152, 4297, 0 },
		{ 3152, 4530, 0 },
		{ 3112, 3773, 0 },
		{ 3152, 4569, 0 },
		{ 3142, 3537, 0 },
		{ 3142, 3538, 0 },
		{ 3125, 3275, 0 },
		{ 2120, 3378, 0 },
		{ 2152, 4307, 0 },
		{ 3062, 4203, 0 },
		{ 3021, 4701, 0 },
		{ 3021, 4703, 0 },
		{ 3062, 4205, 0 },
		{ 2167, 3948, 0 },
		{ 3062, 4210, 0 },
		{ 2167, 3949, 0 },
		{ 2167, 3858, 0 },
		{ 3062, 4217, 0 },
		{ 2120, 3379, 0 },
		{ 2251, 4416, 0 },
		{ 2120, 3380, 0 },
		{ 3137, 3905, 0 },
		{ 2120, 3381, 0 },
		{ 2101, 4053, 0 },
		{ 3137, 3851, 0 },
		{ 2082, 4118, 0 },
		{ 3062, 4228, 0 },
		{ 2120, 3382, 0 },
		{ 3137, 3860, 0 },
		{ 3152, 4549, 0 },
		{ 1225, 4274, 0 },
		{ 2167, 3915, 0 },
		{ 3112, 3799, 0 },
		{ 3021, 4810, 0 },
		{ 3021, 4837, 0 },
		{ 2152, 4343, 0 },
		{ 2101, 4064, 0 },
		{ 2251, 4414, 0 },
		{ 3112, 3800, 0 },
		{ 2152, 4348, 0 },
		{ 2152, 4352, 0 },
		{ 3062, 4161, 0 },
		{ 3062, 4163, 0 },
		{ 2152, 4356, 0 },
		{ 3021, 4621, 0 },
		{ 3021, 4623, 0 },
		{ 2152, 4358, 0 },
		{ 2120, 3383, 0 },
		{ 2167, 3924, 0 },
		{ 3146, 3640, 0 },
		{ 3146, 3641, 0 },
		{ 2152, 4367, 0 },
		{ 3142, 3490, 0 },
		{ 3142, 3532, 0 },
		{ 2082, 4104, 0 },
		{ 3152, 4556, 0 },
		{ 3146, 3648, 0 },
		{ 2120, 3384, 68 },
		{ 3146, 3650, 0 },
		{ 3021, 4695, 0 },
		{ 2167, 3947, 0 },
		{ 2120, 3385, 0 },
		{ 2120, 3386, 0 },
		{ 2188, 4252, 0 },
		{ 3062, 4189, 0 },
		{ 3152, 4566, 0 },
		{ 3125, 4002, 0 },
		{ 3146, 3653, 0 },
		{ 3146, 3565, 0 },
		{ 1112, 3459, 0 },
		{ 2082, 4079, 0 },
		{ 3112, 3730, 0 },
		{ 2133, 3431, 0 },
		{ 2060, 3491, 0 },
		{ 2060, 3492, 0 },
		{ 3137, 3894, 0 },
		{ 2098, 3727, 0 },
		{ 1225, 4286, 0 },
		{ 3142, 3548, 0 },
		{ 3112, 3746, 0 },
		{ 3152, 4538, 0 },
		{ 3152, 4540, 0 },
		{ 2152, 4323, 0 },
		{ 0, 0, 69 },
		{ 0, 0, 72 },
		{ 3021, 4801, 0 },
		{ 1225, 4263, 0 },
		{ 2082, 4101, 0 },
		{ 3112, 3750, 0 },
		{ 1225, 4270, 0 },
		{ 3112, 3751, 0 },
		{ 0, 0, 118 },
		{ 3146, 3576, 0 },
		{ 3146, 3579, 0 },
		{ 3112, 3756, 0 },
		{ 0, 0, 110 },
		{ 2120, 3387, 0 },
		{ 3021, 4865, 0 },
		{ 0, 0, 119 },
		{ 0, 0, 120 },
		{ 2167, 3939, 0 },
		{ 2082, 4119, 0 },
		{ 3125, 3991, 0 },
		{ 980, 4127, 0 },
		{ 2101, 4055, 0 },
		{ 1225, 4287, 0 },
		{ 3146, 3581, 0 },
		{ 2956, 4593, 0 },
		{ 0, 0, 3 },
		{ 2152, 4350, 0 },
		{ 3152, 4522, 0 },
		{ 2251, 4447, 0 },
		{ 3021, 4645, 0 },
		{ 3125, 3994, 0 },
		{ 3062, 4162, 0 },
		{ 3146, 3582, 0 },
		{ 3062, 4165, 0 },
		{ 2152, 4357, 0 },
		{ 2120, 3388, 0 },
		{ 2101, 4062, 0 },
		{ 2251, 4504, 0 },
		{ 2098, 3676, 0 },
		{ 2098, 3678, 0 },
		{ 2152, 4362, 0 },
		{ 3125, 4006, 0 },
		{ 1052, 4231, 0 },
		{ 2152, 3281, 0 },
		{ 3062, 4174, 0 },
		{ 2167, 3951, 0 },
		{ 0, 0, 79 },
		{ 2152, 4380, 0 },
		{ 0, 0, 87 },
		{ 3021, 4727, 0 },
		{ 2152, 4381, 0 },
		{ 3021, 4731, 0 },
		{ 3146, 3591, 0 },
		{ 2152, 4383, 0 },
		{ 3142, 3557, 0 },
		{ 3152, 4572, 0 },
		{ 2152, 4386, 0 },
		{ 2167, 3958, 0 },
		{ 2082, 4105, 0 },
		{ 3146, 3594, 0 },
		{ 2082, 4108, 0 },
		{ 3062, 4184, 0 },
		{ 2167, 3960, 0 },
		{ 3062, 4191, 0 },
		{ 2152, 4299, 0 },
		{ 0, 0, 74 },
		{ 2167, 3961, 0 },
		{ 2167, 3963, 0 },
		{ 3021, 4805, 0 },
		{ 2101, 4051, 0 },
		{ 3146, 3595, 0 },
		{ 3125, 4035, 0 },
		{ 2152, 4308, 0 },
		{ 2167, 3965, 0 },
		{ 2152, 4312, 0 },
		{ 2120, 3390, 0 },
		{ 3062, 4207, 0 },
		{ 3137, 3883, 0 },
		{ 3021, 4869, 0 },
		{ 2101, 4057, 0 },
		{ 2082, 4078, 0 },
		{ 3021, 4617, 0 },
		{ 2098, 3689, 0 },
		{ 3152, 4568, 0 },
		{ 2098, 3690, 0 },
		{ 2120, 3391, 0 },
		{ 2167, 3918, 0 },
		{ 2060, 3512, 0 },
		{ 3152, 4576, 0 },
		{ 2152, 4327, 0 },
		{ 2152, 4328, 0 },
		{ 879, 3475, 0 },
		{ 0, 3478, 0 },
		{ 3125, 3996, 0 },
		{ 2188, 4242, 0 },
		{ 2152, 4334, 0 },
		{ 3112, 3780, 0 },
		{ 1225, 4276, 0 },
		{ 3021, 4661, 0 },
		{ 3112, 3781, 0 },
		{ 2120, 3392, 0 },
		{ 3146, 3603, 0 },
		{ 3112, 3787, 0 },
		{ 2082, 4107, 0 },
		{ 3062, 4156, 0 },
		{ 3112, 3788, 0 },
		{ 3125, 4009, 0 },
		{ 3146, 3604, 0 },
		{ 2251, 4420, 0 },
		{ 2251, 4424, 0 },
		{ 3021, 4725, 0 },
		{ 2152, 4351, 0 },
		{ 0, 0, 73 },
		{ 2082, 4111, 0 },
		{ 3146, 3606, 0 },
		{ 3137, 3861, 0 },
		{ 3112, 3796, 0 },
		{ 3112, 3797, 0 },
		{ 3112, 3798, 0 },
		{ 3146, 3607, 0 },
		{ 2167, 3953, 0 },
		{ 2167, 3955, 0 },
		{ 0, 0, 140 },
		{ 0, 0, 141 },
		{ 2101, 4059, 0 },
		{ 1225, 4279, 0 },
		{ 3146, 3608, 0 },
		{ 2082, 4080, 0 },
		{ 2082, 4082, 0 },
		{ 2956, 4590, 0 },
		{ 0, 0, 10 },
		{ 3021, 4799, 0 },
		{ 0, 0, 12 },
		{ 2098, 3714, 0 },
		{ 3146, 3609, 0 },
		{ 3021, 4266, 0 },
		{ 3152, 4558, 0 },
		{ 3125, 3980, 0 },
		{ 3021, 4839, 0 },
		{ 3021, 4841, 0 },
		{ 3146, 3610, 0 },
		{ 2120, 3393, 0 },
		{ 3062, 4186, 0 },
		{ 3062, 4188, 0 },
		{ 2152, 4390, 0 },
		{ 2120, 3394, 0 },
		{ 3152, 4513, 0 },
		{ 3021, 4867, 0 },
		{ 3152, 4514, 0 },
		{ 2082, 4094, 0 },
		{ 0, 0, 88 },
		{ 2167, 3962, 0 },
		{ 3062, 4194, 0 },
		{ 0, 0, 86 },
		{ 3142, 3545, 0 },
		{ 2101, 4042, 0 },
		{ 0, 0, 89 },
		{ 3152, 4542, 0 },
		{ 3062, 4200, 0 },
		{ 3142, 3546, 0 },
		{ 2152, 4301, 0 },
		{ 2098, 3723, 0 },
		{ 3112, 3810, 0 },
		{ 2152, 4305, 0 },
		{ 2120, 3395, 0 },
		{ 3146, 3616, 0 },
		{ 0, 0, 67 },
		{ 0, 0, 105 },
		{ 0, 0, 107 },
		{ 2167, 3972, 0 },
		{ 2251, 4422, 0 },
		{ 3112, 3813, 0 },
		{ 2152, 4311, 0 },
		{ 3062, 4211, 0 },
		{ 3137, 3887, 0 },
		{ 2152, 4314, 0 },
		{ 0, 0, 146 },
		{ 3062, 4214, 0 },
		{ 3112, 3814, 0 },
		{ 2152, 4317, 0 },
		{ 3062, 4216, 0 },
		{ 3146, 3617, 0 },
		{ 3112, 3816, 0 },
		{ 3021, 4711, 0 },
		{ 2120, 3396, 0 },
		{ 2082, 4120, 0 },
		{ 2082, 4121, 0 },
		{ 2152, 4325, 0 },
		{ 2251, 4506, 0 },
		{ 3146, 3619, 0 },
		{ 3146, 3620, 0 },
		{ 3112, 3731, 0 },
		{ 2188, 4246, 0 },
		{ 0, 4129, 0 },
		{ 3146, 3621, 0 },
		{ 3146, 3623, 0 },
		{ 3062, 4148, 0 },
		{ 3137, 3855, 0 },
		{ 2167, 3940, 0 },
		{ 3021, 4775, 0 },
		{ 3062, 4153, 0 },
		{ 3146, 3624, 0 },
		{ 2167, 3943, 0 },
		{ 3152, 4548, 0 },
		{ 0, 0, 20 },
		{ 2098, 3666, 0 },
		{ 2098, 3667, 0 },
		{ 0, 0, 137 },
		{ 2098, 3668, 0 },
		{ 0, 0, 139 },
		{ 3112, 3749, 0 },
		{ 2152, 4347, 0 },
		{ 0, 0, 103 },
		{ 2120, 3397, 0 },
		{ 2082, 4090, 0 },
		{ 2082, 4092, 0 },
		{ 2120, 3398, 0 },
		{ 2082, 4096, 0 },
		{ 3021, 4843, 0 },
		{ 2098, 3677, 0 },
		{ 2167, 3954, 0 },
		{ 3062, 4173, 0 },
		{ 0, 0, 84 },
		{ 2082, 4099, 0 },
		{ 1225, 4285, 0 },
		{ 2120, 3399, 0 },
		{ 2082, 4103, 0 },
		{ 3112, 3754, 0 },
		{ 2152, 4363, 0 },
		{ 3152, 4544, 0 },
		{ 3152, 4547, 0 },
		{ 3021, 4875, 0 },
		{ 2152, 4364, 0 },
		{ 3062, 4178, 0 },
		{ 3062, 4179, 0 },
		{ 2120, 3400, 0 },
		{ 2098, 3680, 0 },
		{ 3146, 3630, 0 },
		{ 3125, 4000, 0 },
		{ 3146, 3631, 0 },
		{ 2098, 3683, 0 },
		{ 3062, 4187, 0 },
		{ 3125, 4003, 0 },
		{ 3146, 3632, 0 },
		{ 2152, 4387, 0 },
		{ 0, 0, 92 },
		{ 3152, 4520, 0 },
		{ 0, 0, 109 },
		{ 2082, 4115, 0 },
		{ 3152, 4146, 0 },
		{ 2152, 4292, 0 },
		{ 0, 0, 144 },
		{ 3146, 3633, 0 },
		{ 3062, 4195, 0 },
		{ 3146, 3634, 0 },
		{ 2120, 3401, 64 },
		{ 3125, 4010, 0 },
		{ 2167, 3966, 0 },
		{ 2082, 4122, 0 },
		{ 3142, 3531, 0 },
		{ 2956, 4076, 0 },
		{ 0, 0, 93 },
		{ 2098, 3691, 0 },
		{ 3152, 4560, 0 },
		{ 1052, 4232, 0 },
		{ 0, 4233, 0 },
		{ 3146, 3637, 0 },
		{ 3125, 4022, 0 },
		{ 3125, 4025, 0 },
		{ 2167, 3971, 0 },
		{ 3152, 4575, 0 },
		{ 2152, 4310, 0 },
		{ 3062, 4213, 0 },
		{ 2120, 3402, 0 },
		{ 2167, 3910, 0 },
		{ 2167, 3911, 0 },
		{ 2167, 3912, 0 },
		{ 0, 0, 97 },
		{ 3062, 4218, 0 },
		{ 2098, 3701, 0 },
		{ 3112, 3772, 0 },
		{ 0, 0, 131 },
		{ 2120, 3403, 0 },
		{ 3142, 3536, 0 },
		{ 2120, 3404, 0 },
		{ 3137, 3906, 0 },
		{ 3146, 3642, 0 },
		{ 3062, 4227, 0 },
		{ 2251, 4449, 0 },
		{ 2098, 3709, 0 },
		{ 3125, 3986, 0 },
		{ 0, 0, 99 },
		{ 3125, 3987, 0 },
		{ 2251, 4457, 0 },
		{ 2251, 4461, 0 },
		{ 3146, 3643, 0 },
		{ 0, 0, 16 },
		{ 2082, 4100, 0 },
		{ 0, 0, 18 },
		{ 0, 0, 19 },
		{ 3062, 4150, 0 },
		{ 2120, 3278, 0 },
		{ 2188, 4253, 0 },
		{ 3125, 3992, 0 },
		{ 3021, 4861, 0 },
		{ 3112, 3782, 0 },
		{ 2167, 3934, 0 },
		{ 1225, 4282, 0 },
		{ 3112, 3783, 0 },
		{ 3112, 3784, 0 },
		{ 3125, 3999, 0 },
		{ 2167, 3941, 0 },
		{ 0, 0, 63 },
		{ 3062, 4164, 0 },
		{ 3146, 3645, 0 },
		{ 2120, 3289, 0 },
		{ 3146, 3647, 0 },
		{ 2082, 4113, 0 },
		{ 1112, 3454, 0 },
		{ 2167, 3946, 0 },
		{ 2152, 4354, 0 },
		{ 0, 0, 82 },
		{ 2120, 3290, 0 },
		{ 3152, 4564, 0 },
		{ 3112, 3790, 0 },
		{ 0, 3457, 0 },
		{ 3112, 3795, 0 },
		{ 0, 0, 17 },
		{ 2167, 3952, 0 },
		{ 1225, 4277, 0 },
		{ 2120, 3292, 62 },
		{ 2120, 3293, 0 },
		{ 2082, 4069, 0 },
		{ 0, 0, 83 },
		{ 3125, 4021, 0 },
		{ 3152, 3506, 0 },
		{ 0, 0, 90 },
		{ 0, 0, 91 },
		{ 0, 0, 60 },
		{ 3125, 4024, 0 },
		{ 3137, 3878, 0 },
		{ 3137, 3879, 0 },
		{ 3146, 3561, 0 },
		{ 3137, 3882, 0 },
		{ 0, 0, 145 },
		{ 0, 0, 133 },
		{ 3125, 4031, 0 },
		{ 1225, 4289, 0 },
		{ 1225, 4290, 0 },
		{ 3146, 3563, 0 },
		{ 0, 0, 40 },
		{ 2120, 3295, 41 },
		{ 2120, 3296, 43 },
		{ 3125, 3976, 0 },
		{ 3152, 4552, 0 },
		{ 1225, 4268, 0 },
		{ 1225, 4269, 0 },
		{ 2082, 4088, 0 },
		{ 0, 0, 80 },
		{ 3125, 3977, 0 },
		{ 3146, 3567, 0 },
		{ 0, 0, 98 },
		{ 3146, 3568, 0 },
		{ 2082, 4093, 0 },
		{ 3137, 3888, 0 },
		{ 2082, 4095, 0 },
		{ 2098, 3656, 0 },
		{ 3062, 4206, 0 },
		{ 3142, 3553, 0 },
		{ 3062, 4208, 0 },
		{ 2188, 4257, 0 },
		{ 2188, 4258, 0 },
		{ 2120, 3297, 0 },
		{ 3146, 3571, 0 },
		{ 3152, 4528, 0 },
		{ 3142, 3556, 0 },
		{ 0, 0, 94 },
		{ 3152, 4536, 0 },
		{ 2120, 3298, 0 },
		{ 0, 0, 138 },
		{ 0, 0, 142 },
		{ 2082, 4102, 0 },
		{ 0, 0, 147 },
		{ 0, 0, 11 },
		{ 3125, 3988, 0 },
		{ 3125, 3989, 0 },
		{ 2167, 3973, 0 },
		{ 3137, 3903, 0 },
		{ 3137, 3904, 0 },
		{ 1225, 4266, 0 },
		{ 2120, 3299, 0 },
		{ 3146, 3577, 0 },
		{ 3125, 3995, 0 },
		{ 3146, 3578, 0 },
		{ 3152, 4561, 0 },
		{ 0, 0, 143 },
		{ 3062, 4223, 0 },
		{ 3152, 4565, 0 },
		{ 3125, 3997, 0 },
		{ 3142, 3526, 0 },
		{ 3142, 3527, 0 },
		{ 3142, 3528, 0 },
		{ 3152, 4571, 0 },
		{ 2120, 3300, 0 },
		{ 3152, 4573, 0 },
		{ 3062, 4142, 0 },
		{ 3021, 4629, 0 },
		{ 3146, 3583, 0 },
		{ 3146, 3585, 0 },
		{ 2120, 3301, 0 },
		{ 0, 0, 42 },
		{ 0, 0, 44 },
		{ 3125, 4005, 0 },
		{ 3021, 4639, 0 },
		{ 3152, 4518, 0 },
		{ 3146, 3587, 0 },
		{ 2167, 3929, 0 },
		{ 2082, 4067, 0 },
		{ 3062, 4154, 0 },
		{ 3062, 4155, 0 },
		{ 2956, 4578, 0 },
		{ 3152, 4532, 0 },
		{ 2082, 4068, 0 },
		{ 3021, 4693, 0 },
		{ 3062, 4158, 0 },
		{ 3125, 4008, 0 },
		{ 2082, 4071, 0 },
		{ 2167, 3930, 0 },
		{ 2167, 3931, 0 },
		{ 2152, 4349, 0 },
		{ 3146, 3588, 0 },
		{ 2082, 4076, 0 },
		{ 2082, 4077, 0 },
		{ 2167, 3933, 0 },
		{ 0, 0, 85 },
		{ 0, 0, 100 },
		{ 3125, 4013, 0 },
		{ 3125, 4014, 0 },
		{ 0, 4272, 0 },
		{ 3062, 4169, 0 },
		{ 0, 0, 95 },
		{ 2082, 4081, 0 },
		{ 3125, 4017, 0 },
		{ 2098, 3686, 0 },
		{ 0, 0, 13 },
		{ 2167, 3935, 0 },
		{ 2167, 3936, 0 },
		{ 0, 0, 96 },
		{ 0, 0, 61 },
		{ 0, 0, 101 },
		{ 3112, 3743, 0 },
		{ 3125, 4023, 0 },
		{ 2152, 4365, 0 },
		{ 0, 0, 15 },
		{ 2120, 3302, 0 },
		{ 3112, 3745, 0 },
		{ 2152, 4368, 0 },
		{ 3137, 3873, 0 },
		{ 2082, 4091, 0 },
		{ 3021, 4789, 0 },
		{ 3152, 4516, 0 },
		{ 2152, 4372, 0 },
		{ 2101, 4063, 0 },
		{ 2152, 4374, 0 },
		{ 3125, 4027, 0 },
		{ 3146, 3590, 0 },
		{ 0, 0, 14 },
		{ 3205, 1389, 244 },
		{ 0, 0, 245 },
		{ 3151, 5215, 246 },
		{ 3188, 1829, 250 },
		{ 1262, 2864, 251 },
		{ 0, 0, 251 },
		{ 3188, 1983, 247 },
		{ 1265, 1390, 0 },
		{ 3188, 2006, 248 },
		{ 1268, 1437, 0 },
		{ 1265, 0, 0 },
		{ 3097, 2789, 249 },
		{ 1270, 1471, 0 },
		{ 1268, 0, 0 },
		{ 3097, 2853, 247 },
		{ 1270, 0, 0 },
		{ 3097, 2863, 248 },
		{ 3142, 3525, 156 },
		{ 0, 0, 156 },
		{ 0, 0, 157 },
		{ 3166, 2185, 0 },
		{ 3188, 3025, 0 },
		{ 3205, 2247, 0 },
		{ 1278, 4975, 0 },
		{ 3185, 2785, 0 },
		{ 3188, 3100, 0 },
		{ 3200, 3153, 0 },
		{ 3196, 2622, 0 },
		{ 3199, 3194, 0 },
		{ 3205, 2296, 0 },
		{ 3199, 3232, 0 },
		{ 3201, 1902, 0 },
		{ 3088, 2730, 0 },
		{ 3203, 2358, 0 },
		{ 3147, 2494, 0 },
		{ 3166, 2213, 0 },
		{ 3206, 4770, 0 },
		{ 0, 0, 154 },
		{ 3142, 3549, 160 },
		{ 0, 0, 160 },
		{ 0, 0, 161 },
		{ 3166, 2143, 0 },
		{ 3188, 3036, 0 },
		{ 3205, 2264, 0 },
		{ 1299, 5043, 0 },
		{ 3152, 4198, 0 },
		{ 3142, 3554, 0 },
		{ 2251, 4426, 0 },
		{ 3021, 4699, 0 },
		{ 3206, 4735, 0 },
		{ 0, 0, 158 },
		{ 2956, 4580, 168 },
		{ 0, 0, 168 },
		{ 0, 0, 169 },
		{ 3188, 3106, 0 },
		{ 3034, 2977, 0 },
		{ 3203, 2363, 0 },
		{ 3205, 2287, 0 },
		{ 3188, 3097, 0 },
		{ 1314, 5072, 0 },
		{ 3188, 2717, 0 },
		{ 3170, 1477, 0 },
		{ 3188, 3108, 0 },
		{ 3205, 2246, 0 },
		{ 2798, 1413, 0 },
		{ 3201, 1806, 0 },
		{ 3195, 2941, 0 },
		{ 3088, 2875, 0 },
		{ 3147, 2451, 0 },
		{ 2990, 2951, 0 },
		{ 1325, 5074, 0 },
		{ 3188, 2702, 0 },
		{ 3196, 2674, 0 },
		{ 3166, 2183, 0 },
		{ 3188, 3063, 0 },
		{ 1330, 5014, 0 },
		{ 3198, 2664, 0 },
		{ 3193, 1642, 0 },
		{ 3147, 2446, 0 },
		{ 3200, 3167, 0 },
		{ 3201, 2001, 0 },
		{ 3088, 2920, 0 },
		{ 3203, 2407, 0 },
		{ 3147, 2503, 0 },
		{ 3206, 4842, 0 },
		{ 0, 0, 166 },
		{ 3142, 3521, 192 },
		{ 0, 0, 192 },
		{ 3166, 2216, 0 },
		{ 3188, 3067, 0 },
		{ 3205, 2257, 0 },
		{ 1346, 5049, 0 },
		{ 3200, 2922, 0 },
		{ 3196, 2651, 0 },
		{ 3199, 3183, 0 },
		{ 3166, 2219, 0 },
		{ 3166, 2232, 0 },
		{ 3188, 3024, 0 },
		{ 3166, 2052, 0 },
		{ 3206, 4939, 0 },
		{ 0, 0, 191 },
		{ 2188, 4250, 152 },
		{ 0, 0, 152 },
		{ 0, 0, 153 },
		{ 3188, 3028, 0 },
		{ 3147, 2507, 0 },
		{ 3203, 2371, 0 },
		{ 3190, 2592, 0 },
		{ 3188, 3089, 0 },
		{ 3152, 4562, 0 },
		{ 3196, 2665, 0 },
		{ 3199, 3229, 0 },
		{ 3166, 2092, 0 },
		{ 3166, 2118, 0 },
		{ 3168, 4948, 0 },
		{ 3168, 4942, 0 },
		{ 3088, 2715, 0 },
		{ 3147, 2466, 0 },
		{ 3088, 2854, 0 },
		{ 3201, 2021, 0 },
		{ 3088, 2890, 0 },
		{ 3199, 3228, 0 },
		{ 3196, 2672, 0 },
		{ 3088, 2921, 0 },
		{ 3166, 1729, 0 },
		{ 3188, 3077, 0 },
		{ 3205, 2277, 0 },
		{ 3206, 4877, 0 },
		{ 0, 0, 150 },
		{ 2699, 3176, 24 },
		{ 0, 0, 24 },
		{ 0, 0, 25 },
		{ 3188, 3092, 0 },
		{ 2990, 2945, 0 },
		{ 3088, 2779, 0 },
		{ 3147, 2536, 0 },
		{ 3151, 5194, 0 },
		{ 3203, 2353, 0 },
		{ 2948, 2326, 0 },
		{ 3188, 3012, 0 },
		{ 3205, 2285, 0 },
		{ 3199, 3197, 0 },
		{ 3201, 1865, 0 },
		{ 3203, 2399, 0 },
		{ 3205, 2294, 0 },
		{ 3151, 5193, 0 },
		{ 3185, 3130, 0 },
		{ 3188, 3037, 0 },
		{ 3166, 2190, 0 },
		{ 3200, 3168, 0 },
		{ 3205, 2306, 0 },
		{ 3088, 2867, 0 },
		{ 2948, 2344, 0 },
		{ 3201, 1897, 0 },
		{ 3088, 2905, 0 },
		{ 3203, 2389, 0 },
		{ 3147, 2465, 0 },
		{ 3151, 2, 0 },
		{ 3168, 4951, 0 },
		{ 0, 0, 21 },
		{ 1429, 0, 1 },
		{ 1429, 0, 193 },
		{ 1429, 2854, 243 },
		{ 1644, 171, 243 },
		{ 1644, 409, 243 },
		{ 1644, 397, 243 },
		{ 1644, 530, 243 },
		{ 1644, 398, 243 },
		{ 1644, 413, 243 },
		{ 1644, 388, 243 },
		{ 1644, 405, 243 },
		{ 1644, 485, 243 },
		{ 1429, 0, 243 },
		{ 1441, 2712, 243 },
		{ 1429, 2992, 243 },
		{ 2699, 3173, 239 },
		{ 1644, 505, 243 },
		{ 1644, 506, 243 },
		{ 1644, 535, 243 },
		{ 1644, 0, 243 },
		{ 1644, 570, 243 },
		{ 1644, 583, 243 },
		{ 3203, 2409, 0 },
		{ 0, 0, 194 },
		{ 3147, 2460, 0 },
		{ 1644, 551, 0 },
		{ 1644, 0, 0 },
		{ 3151, 4192, 0 },
		{ 1644, 571, 0 },
		{ 1644, 588, 0 },
		{ 1644, 612, 0 },
		{ 1644, 619, 0 },
		{ 1644, 611, 0 },
		{ 1644, 614, 0 },
		{ 1644, 621, 0 },
		{ 1644, 619, 0 },
		{ 1644, 594, 0 },
		{ 1644, 588, 0 },
		{ 1644, 592, 0 },
		{ 3188, 3007, 0 },
		{ 3188, 3009, 0 },
		{ 1645, 610, 0 },
		{ 1645, 611, 0 },
		{ 1644, 621, 0 },
		{ 1644, 623, 0 },
		{ 1644, 614, 0 },
		{ 3203, 2361, 0 },
		{ 3185, 3124, 0 },
		{ 1644, 638, 0 },
		{ 1644, 682, 0 },
		{ 1644, 659, 0 },
		{ 1644, 660, 0 },
		{ 1644, 683, 0 },
		{ 1644, 684, 0 },
		{ 1644, 725, 0 },
		{ 1644, 719, 0 },
		{ 1644, 695, 0 },
		{ 1644, 686, 0 },
		{ 1644, 683, 0 },
		{ 1644, 698, 0 },
		{ 1644, 685, 0 },
		{ 3147, 2480, 0 },
		{ 3034, 2969, 0 },
		{ 1644, 727, 0 },
		{ 1644, 717, 0 },
		{ 1645, 719, 0 },
		{ 1644, 13, 0 },
		{ 1644, 25, 0 },
		{ 3196, 2679, 0 },
		{ 0, 0, 242 },
		{ 1644, 40, 0 },
		{ 1644, 26, 0 },
		{ 1644, 14, 0 },
		{ 1644, 62, 0 },
		{ 1644, 66, 0 },
		{ 1644, 67, 0 },
		{ 1644, 63, 0 },
		{ 1644, 51, 0 },
		{ 1644, 29, 0 },
		{ 1644, 36, 0 },
		{ 1644, 51, 0 },
		{ 1644, 0, 228 },
		{ 1644, 87, 0 },
		{ 3203, 2405, 0 },
		{ 3088, 2882, 0 },
		{ 1644, 45, 0 },
		{ 1644, 49, 0 },
		{ 1644, 44, 0 },
		{ 1644, 61, 0 },
		{ 1644, 63, 0 },
		{ -1524, 1548, 0 },
		{ 1645, 97, 0 },
		{ 1644, 132, 0 },
		{ 1644, 139, 0 },
		{ 1644, 160, 0 },
		{ 1644, 183, 0 },
		{ 1644, 184, 0 },
		{ 1644, 163, 0 },
		{ 1644, 179, 0 },
		{ 1644, 155, 0 },
		{ 1644, 145, 0 },
		{ 1644, 0, 227 },
		{ 1644, 155, 0 },
		{ 3190, 2570, 0 },
		{ 3147, 2538, 0 },
		{ 1644, 159, 0 },
		{ 1644, 169, 0 },
		{ 1644, 167, 0 },
		{ 1644, 0, 241 },
		{ 1644, 166, 0 },
		{ 0, 0, 229 },
		{ 1644, 160, 0 },
		{ 1646, 4, -4 },
		{ 1644, 199, 0 },
		{ 1644, 211, 0 },
		{ 1644, 273, 0 },
		{ 1644, 279, 0 },
		{ 1644, 211, 0 },
		{ 1644, 224, 0 },
		{ 1644, 195, 0 },
		{ 1644, 226, 0 },
		{ 1644, 219, 0 },
		{ 3188, 3043, 0 },
		{ 3188, 3054, 0 },
		{ 1644, 0, 231 },
		{ 1644, 259, 232 },
		{ 1644, 269, 0 },
		{ 1644, 273, 0 },
		{ 1644, 300, 0 },
		{ 1544, 3776, 0 },
		{ 3151, 4539, 0 },
		{ 2229, 4906, 218 },
		{ 1644, 303, 0 },
		{ 1644, 307, 0 },
		{ 1644, 305, 0 },
		{ 1644, 306, 0 },
		{ 1644, 313, 0 },
		{ 1644, 315, 0 },
		{ 1644, 306, 0 },
		{ 1644, 308, 0 },
		{ 1644, 303, 0 },
		{ 1644, 310, 0 },
		{ 1645, 297, 0 },
		{ 3152, 4555, 0 },
		{ 3151, 5213, 234 },
		{ 1644, 327, 0 },
		{ 1644, 341, 0 },
		{ 1644, 324, 0 },
		{ 1644, 373, 0 },
		{ 0, 0, 198 },
		{ 1646, 31, -7 },
		{ 1646, 117, -10 },
		{ 1646, 231, -13 },
		{ 1646, 345, -16 },
		{ 1646, 376, -19 },
		{ 1646, 460, -22 },
		{ 1644, 405, 0 },
		{ 1644, 418, 0 },
		{ 1644, 391, 0 },
		{ 1644, 0, 216 },
		{ 1644, 0, 230 },
		{ 3196, 2667, 0 },
		{ 1644, 390, 0 },
		{ 1644, 380, 0 },
		{ 1644, 384, 0 },
		{ 1645, 381, 0 },
		{ 1581, 3744, 0 },
		{ 3151, 4451, 0 },
		{ 2229, 4922, 219 },
		{ 1584, 3745, 0 },
		{ 3151, 4535, 0 },
		{ 2229, 4902, 220 },
		{ 1587, 3746, 0 },
		{ 3151, 4543, 0 },
		{ 2229, 4924, 223 },
		{ 1590, 3747, 0 },
		{ 3151, 4459, 0 },
		{ 2229, 4920, 224 },
		{ 1593, 3748, 0 },
		{ 3151, 4533, 0 },
		{ 2229, 4893, 225 },
		{ 1596, 3749, 0 },
		{ 3151, 4537, 0 },
		{ 2229, 4913, 226 },
		{ 1644, 430, 0 },
		{ 1646, 488, -25 },
		{ 1644, 422, 0 },
		{ 3199, 3212, 0 },
		{ 1644, 403, 0 },
		{ 1644, 448, 0 },
		{ 1644, 441, 0 },
		{ 1644, 453, 0 },
		{ 0, 0, 200 },
		{ 0, 0, 202 },
		{ 0, 0, 208 },
		{ 0, 0, 210 },
		{ 0, 0, 212 },
		{ 0, 0, 214 },
		{ 1646, 490, -28 },
		{ 1614, 3759, 0 },
		{ 3151, 4569, 0 },
		{ 2229, 4898, 222 },
		{ 1644, 0, 215 },
		{ 3166, 2214, 0 },
		{ 1644, 482, 0 },
		{ 1644, 497, 0 },
		{ 1645, 490, 0 },
		{ 1644, 487, 0 },
		{ 1623, 3766, 0 },
		{ 3151, 4541, 0 },
		{ 2229, 4901, 221 },
		{ 0, 0, 206 },
		{ 3166, 2053, 0 },
		{ 1644, 4, 237 },
		{ 1645, 495, 0 },
		{ 1644, 3, 240 },
		{ 1644, 512, 0 },
		{ 0, 0, 204 },
		{ 3168, 4949, 0 },
		{ 3168, 4950, 0 },
		{ 1644, 500, 0 },
		{ 0, 0, 238 },
		{ 1644, 497, 0 },
		{ 3168, 4945, 0 },
		{ 0, 0, 236 },
		{ 1644, 508, 0 },
		{ 1644, 525, 0 },
		{ 0, 0, 235 },
		{ 1644, 530, 0 },
		{ 1644, 522, 0 },
		{ 1645, 524, 233 },
		{ 1646, 925, 0 },
		{ 1647, 736, -1 },
		{ 1648, 3790, 0 },
		{ 3151, 4525, 0 },
		{ 2229, 4896, 217 },
		{ 0, 0, 196 },
		{ 2188, 4255, 286 },
		{ 0, 0, 286 },
		{ 3188, 3090, 0 },
		{ 3147, 2485, 0 },
		{ 3203, 2384, 0 },
		{ 3190, 2589, 0 },
		{ 3188, 3103, 0 },
		{ 3152, 4550, 0 },
		{ 3196, 2669, 0 },
		{ 3199, 3244, 0 },
		{ 3166, 2066, 0 },
		{ 3166, 2074, 0 },
		{ 3168, 4928, 0 },
		{ 3168, 4929, 0 },
		{ 3088, 2869, 0 },
		{ 3147, 2525, 0 },
		{ 3088, 2880, 0 },
		{ 3201, 1827, 0 },
		{ 3088, 2889, 0 },
		{ 3199, 3241, 0 },
		{ 3196, 2645, 0 },
		{ 3088, 2899, 0 },
		{ 3166, 1735, 0 },
		{ 3188, 3040, 0 },
		{ 3205, 2293, 0 },
		{ 3206, 4941, 0 },
		{ 0, 0, 285 },
		{ 2188, 4249, 288 },
		{ 0, 0, 288 },
		{ 0, 0, 289 },
		{ 3188, 3045, 0 },
		{ 3147, 2444, 0 },
		{ 3203, 2410, 0 },
		{ 3190, 2580, 0 },
		{ 3188, 3073, 0 },
		{ 3152, 4517, 0 },
		{ 3196, 2673, 0 },
		{ 3199, 3192, 0 },
		{ 3166, 2121, 0 },
		{ 3166, 2131, 0 },
		{ 3168, 4930, 0 },
		{ 3168, 4936, 0 },
		{ 3200, 3161, 0 },
		{ 3205, 2305, 0 },
		{ 3203, 2362, 0 },
		{ 3166, 2133, 0 },
		{ 3166, 2139, 0 },
		{ 3203, 2379, 0 },
		{ 3170, 1488, 0 },
		{ 3188, 3118, 0 },
		{ 3205, 2249, 0 },
		{ 3206, 4867, 0 },
		{ 0, 0, 287 },
		{ 2188, 4256, 291 },
		{ 0, 0, 291 },
		{ 0, 0, 292 },
		{ 3188, 2994, 0 },
		{ 3147, 2513, 0 },
		{ 3203, 2396, 0 },
		{ 3190, 2572, 0 },
		{ 3188, 3018, 0 },
		{ 3152, 4557, 0 },
		{ 3196, 2620, 0 },
		{ 3199, 3243, 0 },
		{ 3166, 2172, 0 },
		{ 3166, 2182, 0 },
		{ 3168, 4946, 0 },
		{ 3168, 4947, 0 },
		{ 3190, 2590, 0 },
		{ 3193, 1598, 0 },
		{ 3201, 1950, 0 },
		{ 3199, 3211, 0 },
		{ 3201, 1954, 0 },
		{ 3203, 2415, 0 },
		{ 3205, 2289, 0 },
		{ 3206, 4801, 0 },
		{ 0, 0, 290 },
		{ 2188, 4236, 294 },
		{ 0, 0, 294 },
		{ 0, 0, 295 },
		{ 3188, 3062, 0 },
		{ 3147, 2464, 0 },
		{ 3203, 2355, 0 },
		{ 3190, 2593, 0 },
		{ 3188, 3074, 0 },
		{ 3152, 4515, 0 },
		{ 3196, 2621, 0 },
		{ 3199, 3193, 0 },
		{ 3166, 2194, 0 },
		{ 3166, 2206, 0 },
		{ 3168, 4937, 0 },
		{ 3168, 4940, 0 },
		{ 3188, 3091, 0 },
		{ 3170, 1491, 0 },
		{ 3199, 3226, 0 },
		{ 3196, 2663, 0 },
		{ 3193, 1597, 0 },
		{ 3199, 3231, 0 },
		{ 3201, 1757, 0 },
		{ 3203, 2377, 0 },
		{ 3205, 2243, 0 },
		{ 3206, 4733, 0 },
		{ 0, 0, 293 },
		{ 2188, 4243, 297 },
		{ 0, 0, 297 },
		{ 0, 0, 298 },
		{ 3188, 3116, 0 },
		{ 3147, 2514, 0 },
		{ 3203, 2383, 0 },
		{ 3190, 2574, 0 },
		{ 3188, 2997, 0 },
		{ 3152, 4554, 0 },
		{ 3196, 2615, 0 },
		{ 3199, 3208, 0 },
		{ 3166, 2217, 0 },
		{ 3166, 2218, 0 },
		{ 3168, 4952, 0 },
		{ 3168, 4953, 0 },
		{ 3203, 2392, 0 },
		{ 2948, 2335, 0 },
		{ 3201, 1758, 0 },
		{ 3088, 2904, 0 },
		{ 3190, 2596, 0 },
		{ 3088, 2918, 0 },
		{ 3166, 2220, 0 },
		{ 3188, 3041, 0 },
		{ 3205, 2262, 0 },
		{ 3206, 4809, 0 },
		{ 0, 0, 296 },
		{ 3021, 4845, 172 },
		{ 0, 0, 172 },
		{ 0, 0, 173 },
		{ 3034, 2978, 0 },
		{ 3201, 1800, 0 },
		{ 3188, 3060, 0 },
		{ 3205, 2276, 0 },
		{ 1788, 5036, 0 },
		{ 3188, 2710, 0 },
		{ 3170, 1449, 0 },
		{ 3188, 3068, 0 },
		{ 3205, 2283, 0 },
		{ 2798, 1441, 0 },
		{ 3201, 1834, 0 },
		{ 3195, 2928, 0 },
		{ 3088, 2874, 0 },
		{ 3147, 2502, 0 },
		{ 2990, 2946, 0 },
		{ 1799, 5049, 0 },
		{ 3188, 2704, 0 },
		{ 3196, 2627, 0 },
		{ 3166, 2070, 0 },
		{ 3188, 3105, 0 },
		{ 1804, 5079, 0 },
		{ 3198, 2662, 0 },
		{ 3193, 1565, 0 },
		{ 3147, 2512, 0 },
		{ 3200, 3163, 0 },
		{ 3201, 1868, 0 },
		{ 3088, 2908, 0 },
		{ 3203, 2368, 0 },
		{ 3147, 2522, 0 },
		{ 3206, 4768, 0 },
		{ 0, 0, 170 },
		{ 2188, 4259, 279 },
		{ 0, 0, 279 },
		{ 3188, 3000, 0 },
		{ 3147, 2523, 0 },
		{ 3203, 2370, 0 },
		{ 3190, 2599, 0 },
		{ 3188, 3014, 0 },
		{ 3152, 4559, 0 },
		{ 3196, 2683, 0 },
		{ 3199, 3186, 0 },
		{ 3166, 2096, 0 },
		{ 3166, 2100, 0 },
		{ 3168, 4931, 0 },
		{ 3168, 4935, 0 },
		{ 3185, 3133, 0 },
		{ 3088, 2858, 0 },
		{ 3166, 2109, 0 },
		{ 2948, 2338, 0 },
		{ 3196, 2623, 0 },
		{ 3199, 3218, 0 },
		{ 2798, 1412, 0 },
		{ 3206, 4655, 0 },
		{ 0, 0, 277 },
		{ 1852, 0, 1 },
		{ 2011, 3027, 394 },
		{ 3188, 3046, 394 },
		{ 3199, 3102, 394 },
		{ 3185, 2360, 394 },
		{ 1852, 0, 361 },
		{ 1852, 2959, 394 },
		{ 3195, 1490, 394 },
		{ 2956, 4579, 394 },
		{ 2167, 3922, 394 },
		{ 3142, 3541, 394 },
		{ 2167, 3925, 394 },
		{ 2152, 4346, 394 },
		{ 3205, 2082, 394 },
		{ 1852, 0, 394 },
		{ 2699, 3177, 392 },
		{ 3199, 2962, 394 },
		{ 3199, 3204, 394 },
		{ 0, 0, 394 },
		{ 3203, 2403, 0 },
		{ -1857, 5218, 351 },
		{ -1858, 5029, 0 },
		{ 3147, 2471, 0 },
		{ 0, 0, 357 },
		{ 0, 0, 358 },
		{ 3196, 2671, 0 },
		{ 3088, 2906, 0 },
		{ 3188, 3084, 0 },
		{ 0, 0, 362 },
		{ 3147, 2474, 0 },
		{ 3205, 2253, 0 },
		{ 3088, 2919, 0 },
		{ 2120, 3303, 0 },
		{ 3137, 3893, 0 },
		{ 3146, 3562, 0 },
		{ 2060, 3504, 0 },
		{ 3137, 3899, 0 },
		{ 3193, 1525, 0 },
		{ 3166, 2140, 0 },
		{ 3147, 2499, 0 },
		{ 3201, 1956, 0 },
		{ 3205, 2265, 0 },
		{ 3203, 2418, 0 },
		{ 3095, 5007, 0 },
		{ 3203, 2421, 0 },
		{ 3166, 2161, 0 },
		{ 3201, 1962, 0 },
		{ 3147, 2521, 0 },
		{ 3185, 3139, 0 },
		{ 3205, 2282, 0 },
		{ 3196, 2657, 0 },
		{ 2188, 4244, 0 },
		{ 2120, 3319, 0 },
		{ 2120, 3321, 0 },
		{ 2152, 4388, 0 },
		{ 2082, 4114, 0 },
		{ 3188, 2998, 0 },
		{ 3166, 2174, 0 },
		{ 3185, 3134, 0 },
		{ 3193, 1527, 0 },
		{ 3188, 3008, 0 },
		{ 3196, 2666, 0 },
		{ 0, 21, 354 },
		{ 3190, 2591, 0 },
		{ 3188, 3013, 0 },
		{ 2167, 3970, 0 },
		{ 3201, 2015, 0 },
		{ 0, 0, 393 },
		{ 3188, 3015, 0 },
		{ 3185, 3141, 0 },
		{ 2152, 4303, 0 },
		{ 2098, 3659, 0 },
		{ 3137, 3881, 0 },
		{ 3112, 3767, 0 },
		{ 2120, 3334, 0 },
		{ 0, 0, 382 },
		{ 3152, 4546, 0 },
		{ 3203, 2367, 0 },
		{ 3205, 2288, 0 },
		{ 3147, 2442, 0 },
		{ -1934, 1092, 0 },
		{ 0, 0, 353 },
		{ 3188, 3035, 0 },
		{ 0, 0, 381 },
		{ 2948, 2321, 0 },
		{ 3088, 2917, 0 },
		{ 3147, 2447, 0 },
		{ 1949, 4969, 0 },
		{ 3125, 4015, 0 },
		{ 3062, 4185, 0 },
		{ 3112, 3777, 0 },
		{ 2120, 3346, 0 },
		{ 3137, 3897, 0 },
		{ 3203, 2374, 0 },
		{ 3190, 2582, 0 },
		{ 3147, 2458, 0 },
		{ 3201, 2023, 0 },
		{ 0, 0, 383 },
		{ 3152, 4567, 360 },
		{ 3201, 2026, 0 },
		{ 3200, 3166, 0 },
		{ 3201, 2032, 0 },
		{ 0, 0, 386 },
		{ 0, 0, 387 },
		{ 1954, 0, -71 },
		{ 2133, 3442, 0 },
		{ 2167, 3937, 0 },
		{ 3137, 3850, 0 },
		{ 2152, 4332, 0 },
		{ 3088, 2777, 0 },
		{ 0, 0, 385 },
		{ 0, 0, 391 },
		{ 0, 4967, 0 },
		{ 3196, 2633, 0 },
		{ 3166, 2209, 0 },
		{ 3199, 3250, 0 },
		{ 2188, 4254, 0 },
		{ 3151, 4495, 0 },
		{ 2229, 4895, 376 },
		{ 2152, 4339, 0 },
		{ 2956, 4581, 0 },
		{ 3112, 3793, 0 },
		{ 3112, 3794, 0 },
		{ 3147, 2468, 0 },
		{ 0, 0, 388 },
		{ 0, 0, 389 },
		{ 3199, 3184, 0 },
		{ 2235, 5011, 0 },
		{ 3196, 2655, 0 },
		{ 3188, 3071, 0 },
		{ 0, 0, 366 },
		{ 1977, 0, -74 },
		{ 1979, 0, -77 },
		{ 2167, 3950, 0 },
		{ 3152, 4541, 0 },
		{ 0, 0, 384 },
		{ 3166, 2210, 0 },
		{ 0, 0, 359 },
		{ 2188, 4241, 0 },
		{ 3147, 2473, 0 },
		{ 3151, 4455, 0 },
		{ 2229, 4908, 377 },
		{ 3151, 4457, 0 },
		{ 2229, 4918, 378 },
		{ 2956, 4592, 0 },
		{ 1989, 0, -59 },
		{ 3166, 2212, 0 },
		{ 3188, 3079, 0 },
		{ 3188, 3081, 0 },
		{ 0, 0, 368 },
		{ 0, 0, 370 },
		{ 1994, 0, -65 },
		{ 3151, 4499, 0 },
		{ 2229, 4900, 380 },
		{ 0, 0, 356 },
		{ 3147, 2476, 0 },
		{ 3205, 2244, 0 },
		{ 3151, 4527, 0 },
		{ 2229, 4907, 379 },
		{ 0, 0, 374 },
		{ 3203, 2398, 0 },
		{ 3199, 3219, 0 },
		{ 0, 0, 372 },
		{ 3190, 2588, 0 },
		{ 3201, 1745, 0 },
		{ 3188, 3093, 0 },
		{ 3088, 2891, 0 },
		{ 0, 0, 390 },
		{ 3203, 2401, 0 },
		{ 3147, 2501, 0 },
		{ 2008, 0, -80 },
		{ 3151, 4545, 0 },
		{ 2229, 4899, 375 },
		{ 0, 0, 364 },
		{ 1852, 3044, 394 },
		{ 2015, 2708, 394 },
		{ -2013, 8, 351 },
		{ -2014, 5027, 0 },
		{ 3151, 5011, 0 },
		{ 3095, 4991, 0 },
		{ 0, 0, 352 },
		{ 3095, 5008, 0 },
		{ -2019, 5224, 0 },
		{ -2020, 5021, 0 },
		{ 2023, 2, 354 },
		{ 3095, 5009, 0 },
		{ 3151, 5075, 0 },
		{ 0, 0, 355 },
		{ 2041, 0, 1 },
		{ 2237, 3030, 350 },
		{ 3188, 2992, 350 },
		{ 2041, 0, 304 },
		{ 2041, 2980, 350 },
		{ 3137, 3895, 350 },
		{ 2041, 0, 307 },
		{ 3193, 1601, 350 },
		{ 2956, 4589, 350 },
		{ 2167, 3914, 350 },
		{ 3142, 3479, 350 },
		{ 2167, 3917, 350 },
		{ 2152, 4384, 350 },
		{ 3199, 3207, 350 },
		{ 3205, 2076, 350 },
		{ 2041, 0, 350 },
		{ 2699, 3174, 347 },
		{ 3199, 3215, 350 },
		{ 3185, 3127, 350 },
		{ 2956, 4587, 350 },
		{ 3199, 1751, 350 },
		{ 0, 0, 350 },
		{ 3203, 2412, 0 },
		{ -2048, 5223, 299 },
		{ -2049, 5022, 0 },
		{ 3147, 2517, 0 },
		{ 0, 0, 305 },
		{ 3147, 2518, 0 },
		{ 3203, 2413, 0 },
		{ 3205, 2263, 0 },
		{ 2120, 3294, 0 },
		{ 3137, 3863, 0 },
		{ 3146, 3573, 0 },
		{ 3125, 4033, 0 },
		{ 0, 3466, 0 },
		{ 0, 3490, 0 },
		{ 3137, 3867, 0 },
		{ 3196, 2649, 0 },
		{ 3193, 1643, 0 },
		{ 3166, 2224, 0 },
		{ 3147, 2529, 0 },
		{ 3188, 3019, 0 },
		{ 3188, 3022, 0 },
		{ 3203, 2424, 0 },
		{ 2235, 5015, 0 },
		{ 3203, 2428, 0 },
		{ 3095, 4999, 0 },
		{ 3203, 2351, 0 },
		{ 3185, 3126, 0 },
		{ 2948, 2327, 0 },
		{ 3205, 2273, 0 },
		{ 2188, 4238, 0 },
		{ 2120, 3307, 0 },
		{ 2120, 3308, 0 },
		{ 3062, 4198, 0 },
		{ 3062, 4199, 0 },
		{ 2152, 4322, 0 },
		{ 0, 4117, 0 },
		{ 3166, 2227, 0 },
		{ 3188, 3039, 0 },
		{ 3166, 2228, 0 },
		{ 3185, 3137, 0 },
		{ 3147, 2448, 0 },
		{ 3166, 2229, 0 },
		{ 3196, 2676, 0 },
		{ 0, 0, 349 },
		{ 3196, 2677, 0 },
		{ 0, 0, 301 },
		{ 3190, 2586, 0 },
		{ 0, 0, 346 },
		{ 3193, 1645, 0 },
		{ 3188, 3056, 0 },
		{ 2152, 4335, 0 },
		{ 0, 3703, 0 },
		{ 3137, 3896, 0 },
		{ 2101, 4044, 0 },
		{ 0, 4045, 0 },
		{ 3112, 3792, 0 },
		{ 2120, 3320, 0 },
		{ 3188, 3058, 0 },
		{ 0, 0, 339 },
		{ 3152, 4545, 0 },
		{ 3203, 2365, 0 },
		{ 3201, 1861, 0 },
		{ 3201, 1863, 0 },
		{ 3193, 1522, 0 },
		{ -2128, 1167, 0 },
		{ 3188, 3070, 0 },
		{ 3196, 2625, 0 },
		{ 3147, 2470, 0 },
		{ 3125, 4016, 0 },
		{ 3062, 4229, 0 },
		{ 3112, 3803, 0 },
		{ 3062, 4143, 0 },
		{ 3062, 4144, 0 },
		{ 0, 3330, 0 },
		{ 3137, 3859, 0 },
		{ 0, 0, 338 },
		{ 3203, 2372, 0 },
		{ 3190, 2566, 0 },
		{ 3088, 2728, 0 },
		{ 0, 0, 345 },
		{ 3199, 3199, 0 },
		{ 0, 0, 340 },
		{ 0, 0, 303 },
		{ 3199, 3203, 0 },
		{ 3201, 1869, 0 },
		{ 2145, 0, -56 },
		{ 0, 3439, 0 },
		{ 2167, 3923, 0 },
		{ 2136, 3430, 0 },
		{ 2133, 3429, 0 },
		{ 3137, 3869, 0 },
		{ 2152, 4369, 0 },
		{ 3088, 2729, 0 },
		{ 0, 0, 342 },
		{ 3200, 3158, 0 },
		{ 3201, 1894, 0 },
		{ 3201, 1895, 0 },
		{ 2188, 4248, 0 },
		{ 3151, 4529, 0 },
		{ 2229, 4897, 329 },
		{ 2152, 4375, 0 },
		{ 2956, 4582, 0 },
		{ 2152, 4376, 0 },
		{ 2152, 4377, 0 },
		{ 2152, 4378, 0 },
		{ 0, 4379, 0 },
		{ 3112, 3818, 0 },
		{ 3112, 3819, 0 },
		{ 3147, 2477, 0 },
		{ 3199, 3216, 0 },
		{ 3088, 2781, 0 },
		{ 3088, 2851, 0 },
		{ 3188, 3094, 0 },
		{ 0, 0, 311 },
		{ 2174, 0, -35 },
		{ 2176, 0, -38 },
		{ 2178, 0, -44 },
		{ 2180, 0, -47 },
		{ 2182, 0, -50 },
		{ 2184, 0, -53 },
		{ 0, 3938, 0 },
		{ 3152, 4551, 0 },
		{ 0, 0, 341 },
		{ 3196, 2660, 0 },
		{ 3203, 2380, 0 },
		{ 3203, 2381, 0 },
		{ 3147, 2486, 0 },
		{ 3151, 4461, 0 },
		{ 2229, 4916, 330 },
		{ 3151, 4463, 0 },
		{ 2229, 4919, 331 },
		{ 3151, 4465, 0 },
		{ 2229, 4921, 334 },
		{ 3151, 4467, 0 },
		{ 2229, 4923, 335 },
		{ 3151, 4469, 0 },
		{ 2229, 4886, 336 },
		{ 3151, 4471, 0 },
		{ 2229, 4894, 337 },
		{ 2956, 4584, 0 },
		{ 2199, 0, -62 },
		{ 0, 4237, 0 },
		{ 3147, 2489, 0 },
		{ 3147, 2493, 0 },
		{ 3188, 3110, 0 },
		{ 0, 0, 313 },
		{ 0, 0, 315 },
		{ 0, 0, 321 },
		{ 0, 0, 323 },
		{ 0, 0, 325 },
		{ 0, 0, 327 },
		{ 2205, 0, -68 },
		{ 3151, 4501, 0 },
		{ 2229, 4905, 333 },
		{ 3188, 3111, 0 },
		{ 3199, 3251, 0 },
		{ 3150, 3427, 344 },
		{ 3205, 2299, 0 },
		{ 3151, 4531, 0 },
		{ 2229, 4917, 332 },
		{ 0, 0, 319 },
		{ 3147, 2496, 0 },
		{ 3205, 2301, 0 },
		{ 0, 0, 306 },
		{ 3199, 3189, 0 },
		{ 0, 0, 317 },
		{ 3203, 2387, 0 },
		{ 2798, 1411, 0 },
		{ 3201, 1899, 0 },
		{ 3190, 2568, 0 },
		{ 3021, 4707, 0 },
		{ 3088, 2900, 0 },
		{ 3188, 3002, 0 },
		{ 3196, 2682, 0 },
		{ 3203, 2393, 0 },
		{ 0, 0, 343 },
		{ 2990, 2949, 0 },
		{ 3147, 2508, 0 },
		{ 3203, 2395, 0 },
		{ 2228, 0, -41 },
		{ 3205, 2313, 0 },
		{ 3151, 4453, 0 },
		{ 0, 4903, 328 },
		{ 3088, 2913, 0 },
		{ 0, 0, 309 },
		{ 3201, 1900, 0 },
		{ 3195, 2926, 0 },
		{ 3190, 2584, 0 },
		{ 0, 5014, 0 },
		{ 0, 0, 348 },
		{ 2041, 3040, 350 },
		{ 2241, 2710, 350 },
		{ -2239, 5220, 299 },
		{ -2240, 5020, 0 },
		{ 3151, 5012, 0 },
		{ 3095, 4984, 0 },
		{ 0, 0, 300 },
		{ 3095, 4985, 0 },
		{ -2245, 17, 0 },
		{ -2246, 5024, 0 },
		{ 2249, 0, 301 },
		{ 3095, 4993, 0 },
		{ 3151, 5102, 0 },
		{ 0, 0, 302 },
		{ 0, 4500, 396 },
		{ 0, 0, 396 },
		{ 3188, 3033, 0 },
		{ 3034, 2982, 0 },
		{ 3199, 3242, 0 },
		{ 3193, 1568, 0 },
		{ 3196, 2629, 0 },
		{ 3201, 1951, 0 },
		{ 3151, 7, 0 },
		{ 3205, 2248, 0 },
		{ 3193, 1570, 0 },
		{ 3147, 2524, 0 },
		{ 2264, 4967, 0 },
		{ 3151, 2165, 0 },
		{ 3199, 3188, 0 },
		{ 3205, 2256, 0 },
		{ 3199, 3190, 0 },
		{ 3190, 2597, 0 },
		{ 3188, 3052, 0 },
		{ 3201, 1957, 0 },
		{ 3188, 3055, 0 },
		{ 3205, 2260, 0 },
		{ 3166, 2145, 0 },
		{ 3206, 4671, 0 },
		{ 0, 0, 395 },
		{ 3095, 4982, 446 },
		{ 0, 0, 401 },
		{ 0, 0, 403 },
		{ 2296, 835, 437 },
		{ 2477, 848, 437 },
		{ 2501, 847, 437 },
		{ 2442, 846, 437 },
		{ 2297, 859, 437 },
		{ 2295, 857, 437 },
		{ 2501, 845, 437 },
		{ 2318, 861, 437 },
		{ 2473, 864, 437 },
		{ 2473, 865, 437 },
		{ 2477, 862, 437 },
		{ 2415, 871, 437 },
		{ 2294, 889, 437 },
		{ 3188, 1851, 436 },
		{ 2326, 2735, 446 },
		{ 2537, 861, 437 },
		{ 2477, 875, 437 },
		{ 2330, 875, 437 },
		{ 2477, 872, 437 },
		{ 3188, 3099, 446 },
		{ -2299, 19, 397 },
		{ -2300, 5025, 0 },
		{ 2537, 869, 437 },
		{ 2542, 576, 437 },
		{ 2537, 872, 437 },
		{ 2381, 870, 437 },
		{ 2477, 878, 437 },
		{ 2483, 873, 437 },
		{ 2477, 880, 437 },
		{ 2415, 900, 437 },
		{ 2501, 967, 437 },
		{ 2442, 880, 437 },
		{ 2415, 902, 437 },
		{ 2501, 886, 437 },
		{ 2294, 879, 437 },
		{ 2444, 886, 437 },
		{ 2327, 900, 437 },
		{ 2294, 911, 437 },
		{ 2483, 921, 437 },
		{ 2294, 931, 437 },
		{ 2514, 952, 437 },
		{ 2537, 1190, 437 },
		{ 2514, 953, 437 },
		{ 2385, 956, 437 },
		{ 2542, 578, 437 },
		{ 3188, 1907, 433 },
		{ 2358, 1438, 0 },
		{ 3188, 1918, 434 },
		{ 2514, 967, 437 },
		{ 3147, 2443, 0 },
		{ 3095, 4998, 0 },
		{ 2294, 980, 437 },
		{ 3203, 2234, 0 },
		{ 2473, 978, 437 },
		{ 2369, 963, 437 },
		{ 2514, 998, 437 },
		{ 2444, 993, 437 },
		{ 2444, 994, 437 },
		{ 2385, 1003, 437 },
		{ 2473, 1012, 437 },
		{ 2442, 1032, 437 },
		{ 2473, 1050, 437 },
		{ 2501, 1038, 437 },
		{ 2442, 1035, 437 },
		{ 2473, 1053, 437 },
		{ 2415, 1058, 437 },
		{ 2501, 1042, 437 },
		{ 2542, 580, 437 },
		{ 2542, 582, 437 },
		{ 2508, 1069, 437 },
		{ 2508, 1070, 437 },
		{ 2473, 1085, 437 },
		{ 2369, 1071, 437 },
		{ 2483, 1114, 437 },
		{ 2415, 1129, 437 },
		{ 2460, 1127, 437 },
		{ 3198, 2654, 0 },
		{ 2389, 1404, 0 },
		{ 2358, 0, 0 },
		{ 3097, 2887, 435 },
		{ 2391, 1405, 0 },
		{ 3185, 3120, 0 },
		{ 0, 0, 399 },
		{ 2473, 1127, 437 },
		{ 3034, 2973, 0 },
		{ 2542, 584, 437 },
		{ 2385, 1122, 437 },
		{ 2444, 1115, 437 },
		{ 2542, 10, 437 },
		{ 2477, 1158, 437 },
		{ 2294, 1143, 437 },
		{ 2383, 1162, 437 },
		{ 2473, 1190, 437 },
		{ 2542, 123, 437 },
		{ 2444, 1176, 437 },
		{ 2477, 1188, 437 },
		{ 2542, 125, 437 },
		{ 2444, 1179, 437 },
		{ 2415, 1199, 437 },
		{ 3151, 2420, 0 },
		{ 3201, 2096, 0 },
		{ 2508, 1182, 437 },
		{ 2294, 1186, 437 },
		{ 2501, 1186, 437 },
		{ 2294, 1202, 437 },
		{ 2444, 1186, 437 },
		{ 2294, 1195, 437 },
		{ 2294, 1185, 437 },
		{ 3088, 2864, 0 },
		{ 2389, 0, 0 },
		{ 3097, 2809, 433 },
		{ 2391, 0, 0 },
		{ 3097, 2843, 434 },
		{ 0, 0, 438 },
		{ 2501, 1191, 437 },
		{ 2422, 5127, 0 },
		{ 3196, 2286, 0 },
		{ 2415, 1209, 437 },
		{ 2542, 130, 437 },
		{ 3166, 2144, 0 },
		{ 2567, 6, 437 },
		{ 2508, 1192, 437 },
		{ 2415, 1211, 437 },
		{ 2444, 1193, 437 },
		{ 2501, 1196, 437 },
		{ 3151, 2170, 0 },
		{ 2542, 235, 437 },
		{ 2442, 1194, 437 },
		{ 3203, 2225, 0 },
		{ 2477, 1208, 437 },
		{ 2444, 1198, 437 },
		{ 3147, 2490, 0 },
		{ 3147, 2492, 0 },
		{ 3205, 2308, 0 },
		{ 2483, 1204, 437 },
		{ 2501, 1202, 437 },
		{ 2294, 1220, 437 },
		{ 2473, 1217, 437 },
		{ 2473, 1218, 437 },
		{ 2542, 237, 437 },
		{ 2477, 1216, 437 },
		{ 3196, 2681, 0 },
		{ 2542, 239, 437 },
		{ 3198, 2363, 0 },
		{ 3088, 2856, 0 },
		{ 2444, 1206, 437 },
		{ 3166, 2126, 0 },
		{ 3201, 1964, 0 },
		{ 3206, 4727, 0 },
		{ 3151, 5186, 407 },
		{ 2537, 1214, 437 },
		{ 2444, 1208, 437 },
		{ 2477, 1224, 437 },
		{ 2542, 241, 437 },
		{ 3203, 2357, 0 },
		{ 3198, 2648, 0 },
		{ 2477, 1222, 437 },
		{ 3034, 2963, 0 },
		{ 2483, 1217, 437 },
		{ 2477, 1225, 437 },
		{ 3088, 2893, 0 },
		{ 3088, 2896, 0 },
		{ 3188, 3101, 0 },
		{ 2294, 1213, 437 },
		{ 2477, 1227, 437 },
		{ 2294, 1217, 437 },
		{ 2542, 243, 437 },
		{ 2542, 245, 437 },
		{ 3205, 2154, 0 },
		{ 2514, 1225, 437 },
		{ 3188, 3117, 0 },
		{ 3203, 2243, 0 },
		{ 3137, 3900, 0 },
		{ 3088, 2915, 0 },
		{ 3190, 2576, 0 },
		{ 2477, 1231, 437 },
		{ 3201, 1867, 0 },
		{ 3199, 3223, 0 },
		{ 2567, 121, 437 },
		{ 2483, 1227, 437 },
		{ 2483, 1228, 437 },
		{ 2294, 1240, 437 },
		{ 3151, 2194, 0 },
		{ 2948, 2339, 0 },
		{ 3205, 2236, 0 },
		{ 2514, 1231, 437 },
		{ 2495, 5085, 0 },
		{ 2514, 1232, 437 },
		{ 2483, 1232, 437 },
		{ 3201, 1931, 0 },
		{ 3201, 1933, 0 },
		{ 3188, 3020, 0 },
		{ 2473, 1243, 437 },
		{ 2514, 1235, 437 },
		{ 2294, 1245, 437 },
		{ 3203, 2106, 0 },
		{ 3151, 2426, 0 },
		{ 3188, 3034, 0 },
		{ 2294, 1242, 437 },
		{ 3206, 4813, 0 },
		{ 3034, 2976, 0 },
		{ 3142, 3522, 0 },
		{ 3201, 1960, 0 },
		{ 3088, 2879, 0 },
		{ 2294, 1237, 437 },
		{ 3199, 3206, 0 },
		{ 3201, 1963, 0 },
		{ 3206, 4698, 0 },
		{ 0, 0, 425 },
		{ 2501, 1235, 437 },
		{ 2514, 1240, 437 },
		{ 2542, 349, 437 },
		{ 3203, 2427, 0 },
		{ 3193, 1569, 0 },
		{ 3203, 2350, 0 },
		{ 2502, 1249, 437 },
		{ 3151, 2210, 0 },
		{ 2542, 351, 437 },
		{ 2514, 1244, 437 },
		{ 2527, 5049, 0 },
		{ 2528, 5050, 0 },
		{ 2529, 5070, 0 },
		{ 2294, 1241, 437 },
		{ 2294, 1253, 437 },
		{ 2542, 355, 437 },
		{ 3199, 3239, 0 },
		{ 3034, 2964, 0 },
		{ 3166, 2184, 0 },
		{ 3185, 3138, 0 },
		{ 2294, 1252, 437 },
		{ 3151, 5191, 430 },
		{ 2538, 5100, 0 },
		{ 3166, 2189, 0 },
		{ 3147, 2535, 0 },
		{ 3201, 1811, 0 },
		{ 2294, 1258, 437 },
		{ 3201, 1833, 0 },
		{ 3166, 2208, 0 },
		{ 2542, 357, 437 },
		{ 2542, 370, 437 },
		{ 3151, 2537, 0 },
		{ 3205, 2235, 0 },
		{ 3196, 2637, 0 },
		{ 3190, 2595, 0 },
		{ 2542, 463, 437 },
		{ 3205, 2239, 0 },
		{ 3151, 2162, 0 },
		{ 2542, 469, 437 },
		{ 3201, 2100, 0 },
		{ 3201, 2113, 0 },
		{ 3185, 2783, 0 },
		{ 2542, 471, 437 },
		{ 2542, 484, 437 },
		{ 3200, 2189, 0 },
		{ 3205, 2255, 0 },
		{ 3034, 2966, 0 },
		{ 3196, 2670, 0 },
		{ 3193, 1644, 0 },
		{ 2542, 1503, 437 },
		{ 3203, 2120, 0 },
		{ 2570, 5044, 0 },
		{ 3188, 2996, 0 },
		{ 3206, 4696, 0 },
		{ 2567, 574, 437 },
		{ 3166, 2231, 0 },
		{ 3206, 4731, 0 },
		{ 3151, 2539, 0 },
		{ 3203, 2269, 0 },
		{ 3188, 3005, 0 },
		{ 3201, 1937, 0 },
		{ 3199, 3249, 0 },
		{ 2581, 5069, 0 },
		{ 3203, 2102, 0 },
		{ 3203, 2411, 0 },
		{ 3205, 2271, 0 },
		{ 3151, 2196, 0 },
		{ 3205, 2274, 0 },
		{ 3205, 2275, 0 },
		{ 3188, 3017, 0 },
		{ 3151, 2200, 0 },
		{ 3166, 2141, 0 },
		{ 3166, 2080, 0 },
		{ 3147, 2505, 0 },
		{ 2594, 4958, 0 },
		{ 3188, 3023, 0 },
		{ 3166, 2085, 0 },
		{ 3199, 3201, 0 },
		{ 3200, 3156, 0 },
		{ 2599, 811, 437 },
		{ 3188, 3027, 0 },
		{ 2948, 2341, 0 },
		{ 3206, 4807, 0 },
		{ 3166, 2093, 0 },
		{ 3151, 5130, 405 },
		{ 3166, 2146, 0 },
		{ 3206, 4815, 0 },
		{ 3151, 5144, 414 },
		{ 3203, 2356, 0 },
		{ 2948, 2325, 0 },
		{ 3147, 2519, 0 },
		{ 3206, 4885, 0 },
		{ 3201, 1961, 0 },
		{ 3198, 2652, 0 },
		{ 3199, 3221, 0 },
		{ 3034, 2974, 0 },
		{ 2990, 2953, 0 },
		{ 3203, 2360, 0 },
		{ 3205, 2290, 0 },
		{ 3188, 3048, 0 },
		{ 3188, 3049, 0 },
		{ 2948, 2334, 0 },
		{ 3205, 2292, 0 },
		{ 3088, 2857, 0 },
		{ 3170, 1447, 0 },
		{ 3193, 1641, 0 },
		{ 3166, 2150, 0 },
		{ 3147, 2537, 0 },
		{ 2948, 2343, 0 },
		{ 3147, 2540, 0 },
		{ 3188, 3064, 0 },
		{ 3206, 4873, 0 },
		{ 3151, 5117, 428 },
		{ 3147, 2541, 0 },
		{ 3201, 1966, 0 },
		{ 0, 0, 443 },
		{ 3166, 2132, 0 },
		{ 3088, 2883, 0 },
		{ 3151, 5142, 413 },
		{ 3199, 3191, 0 },
		{ 3188, 3072, 0 },
		{ 3088, 2884, 0 },
		{ 3151, 5159, 432 },
		{ 3088, 2885, 0 },
		{ 3088, 2886, 0 },
		{ 3205, 2302, 0 },
		{ 3034, 2960, 0 },
		{ 2641, 5079, 0 },
		{ 2699, 3175, 0 },
		{ 3203, 2375, 0 },
		{ 3188, 3086, 0 },
		{ 3188, 3088, 0 },
		{ 3201, 1997, 0 },
		{ 3203, 2378, 0 },
		{ 2633, 1330, 0 },
		{ 2649, 5087, 0 },
		{ 2948, 2329, 0 },
		{ 3200, 3149, 0 },
		{ 3201, 2010, 0 },
		{ 3205, 2310, 0 },
		{ 3185, 3136, 0 },
		{ 2655, 4972, 0 },
		{ 3188, 3096, 0 },
		{ 3088, 2902, 0 },
		{ 2658, 5004, 0 },
		{ 0, 1363, 0 },
		{ 3196, 2631, 0 },
		{ 3205, 2234, 0 },
		{ 3201, 2017, 0 },
		{ 3203, 2390, 0 },
		{ 3196, 2647, 0 },
		{ 3188, 3107, 0 },
		{ 3166, 2158, 0 },
		{ 3151, 2941, 0 },
		{ 3199, 3246, 0 },
		{ 2699, 3178, 0 },
		{ 2670, 5055, 0 },
		{ 2671, 5061, 0 },
		{ 3195, 2940, 0 },
		{ 2699, 3180, 0 },
		{ 3188, 3113, 0 },
		{ 3166, 2152, 0 },
		{ 3196, 2653, 0 },
		{ 3205, 2242, 0 },
		{ 3166, 2168, 0 },
		{ 3088, 2707, 0 },
		{ 2680, 5069, 0 },
		{ 3203, 2275, 0 },
		{ 3205, 2245, 0 },
		{ 3190, 2578, 0 },
		{ 3200, 2858, 0 },
		{ 3188, 2999, 0 },
		{ 3206, 4657, 0 },
		{ 3199, 3198, 0 },
		{ 3203, 2400, 0 },
		{ 3147, 2481, 0 },
		{ 3188, 3004, 0 },
		{ 3147, 2483, 0 },
		{ 2948, 2328, 0 },
		{ 3193, 1523, 0 },
		{ 2699, 3171, 0 },
		{ 3199, 3209, 0 },
		{ 3185, 2787, 0 },
		{ 3185, 2789, 0 },
		{ 2698, 5050, 0 },
		{ 3199, 3214, 0 },
		{ 3206, 4811, 0 },
		{ 3201, 2030, 0 },
		{ 3203, 2406, 0 },
		{ 3088, 2862, 0 },
		{ 2704, 5018, 0 },
		{ 3147, 2491, 0 },
		{ 3190, 2284, 0 },
		{ 2948, 2337, 0 },
		{ 3199, 3224, 0 },
		{ 3088, 2871, 0 },
		{ 3199, 3227, 0 },
		{ 3206, 4651, 0 },
		{ 3151, 5148, 426 },
		{ 3201, 1677, 0 },
		{ 3205, 2252, 0 },
		{ 3206, 4663, 0 },
		{ 3206, 4669, 0 },
		{ 3201, 1678, 0 },
		{ 3205, 2254, 0 },
		{ 3034, 2959, 0 },
		{ 3088, 2881, 0 },
		{ 2699, 3179, 0 },
		{ 3188, 3030, 0 },
		{ 3188, 3032, 0 },
		{ 3206, 4737, 0 },
		{ 0, 3170, 0 },
		{ 3151, 5183, 412 },
		{ 3199, 3247, 0 },
		{ 3201, 1736, 0 },
		{ 2948, 2347, 0 },
		{ 3203, 2281, 0 },
		{ 2990, 2947, 0 },
		{ 3203, 2422, 0 },
		{ 3188, 3038, 0 },
		{ 3201, 1746, 0 },
		{ 3166, 2196, 0 },
		{ 3166, 2199, 0 },
		{ 3151, 5072, 406 },
		{ 3203, 2430, 0 },
		{ 3166, 2204, 0 },
		{ 3151, 5077, 418 },
		{ 3151, 5087, 419 },
		{ 3166, 2205, 0 },
		{ 3088, 2894, 0 },
		{ 3034, 2981, 0 },
		{ 3196, 2635, 0 },
		{ 3088, 2897, 0 },
		{ 2948, 2331, 0 },
		{ 2948, 2332, 0 },
		{ 0, 0, 442 },
		{ 3088, 2901, 0 },
		{ 3201, 1751, 0 },
		{ 2746, 5021, 0 },
		{ 3201, 1754, 0 },
		{ 2948, 2336, 0 },
		{ 2749, 5030, 0 },
		{ 3185, 3140, 0 },
		{ 3205, 2267, 0 },
		{ 3088, 2907, 0 },
		{ 3199, 3213, 0 },
		{ 3188, 3066, 0 },
		{ 3205, 2269, 0 },
		{ 3206, 4795, 0 },
		{ 3206, 4797, 0 },
		{ 3147, 2531, 0 },
		{ 3188, 3069, 0 },
		{ 3088, 2912, 0 },
		{ 3196, 2659, 0 },
		{ 3201, 1755, 0 },
		{ 3201, 1756, 0 },
		{ 3196, 2664, 0 },
		{ 3166, 2211, 0 },
		{ 3166, 2115, 0 },
		{ 3206, 4875, 0 },
		{ 3188, 3080, 0 },
		{ 3203, 2230, 0 },
		{ 3188, 3083, 0 },
		{ 3199, 3236, 0 },
		{ 3203, 2369, 0 },
		{ 3201, 1793, 0 },
		{ 3166, 2215, 0 },
		{ 3206, 4659, 0 },
		{ 3205, 939, 409 },
		{ 3151, 5211, 421 },
		{ 2990, 2952, 0 },
		{ 3205, 2284, 0 },
		{ 3201, 1802, 0 },
		{ 3088, 2774, 0 },
		{ 3195, 2942, 0 },
		{ 3195, 2924, 0 },
		{ 3088, 2775, 0 },
		{ 2783, 4986, 0 },
		{ 3200, 3147, 0 },
		{ 3151, 5091, 417 },
		{ 3205, 2286, 0 },
		{ 2948, 2330, 0 },
		{ 3196, 2678, 0 },
		{ 3201, 1803, 0 },
		{ 3147, 2452, 0 },
		{ 3088, 2855, 0 },
		{ 2791, 5056, 0 },
		{ 3151, 5111, 408 },
		{ 3206, 4805, 0 },
		{ 2793, 5064, 0 },
		{ 2798, 1443, 0 },
		{ 3201, 1809, 0 },
		{ 2796, 5074, 0 },
		{ 2797, 5075, 0 },
		{ 3201, 1810, 0 },
		{ 3198, 2656, 0 },
		{ 3205, 2291, 0 },
		{ 3199, 3200, 0 },
		{ 3188, 3115, 0 },
		{ 3206, 4869, 0 },
		{ 3203, 2386, 0 },
		{ 3166, 2225, 0 },
		{ 3203, 2388, 0 },
		{ 3206, 4879, 0 },
		{ 3151, 5150, 423 },
		{ 3206, 4881, 0 },
		{ 3206, 4883, 0 },
		{ 2798, 1397, 0 },
		{ 3206, 4912, 0 },
		{ 3206, 4914, 0 },
		{ 0, 1409, 0 },
		{ 3088, 2876, 0 },
		{ 3088, 2877, 0 },
		{ 3201, 1828, 0 },
		{ 3205, 2297, 0 },
		{ 3151, 5174, 429 },
		{ 3205, 2298, 0 },
		{ 3206, 4661, 0 },
		{ 3147, 2479, 0 },
		{ 0, 0, 445 },
		{ 0, 0, 444 },
		{ 3151, 5178, 410 },
		{ 3206, 4665, 0 },
		{ 0, 0, 441 },
		{ 0, 0, 440 },
		{ 3206, 4667, 0 },
		{ 3196, 2643, 0 },
		{ 2948, 2345, 0 },
		{ 3203, 2397, 0 },
		{ 3199, 3220, 0 },
		{ 3206, 4723, 0 },
		{ 3151, 5195, 404 },
		{ 2828, 5104, 0 },
		{ 3151, 5203, 431 },
		{ 3151, 5207, 411 },
		{ 3188, 3006, 0 },
		{ 3201, 1829, 0 },
		{ 3205, 2300, 0 },
		{ 3201, 1830, 0 },
		{ 3151, 5067, 424 },
		{ 3151, 2422, 0 },
		{ 3206, 4739, 0 },
		{ 3206, 4741, 0 },
		{ 3206, 4743, 0 },
		{ 3203, 2402, 0 },
		{ 3201, 1831, 0 },
		{ 3151, 5081, 415 },
		{ 3151, 5083, 416 },
		{ 3151, 5085, 420 },
		{ 3205, 2303, 0 },
		{ 3188, 3016, 0 },
		{ 3206, 4799, 0 },
		{ 3205, 2304, 0 },
		{ 3151, 5096, 422 },
		{ 3199, 3237, 0 },
		{ 3201, 1832, 0 },
		{ 3088, 2898, 0 },
		{ 3203, 2408, 0 },
		{ 3147, 2498, 0 },
		{ 3166, 2062, 0 },
		{ 3206, 4840, 0 },
		{ 3151, 5115, 427 },
		{ 3095, 4992, 446 },
		{ 2855, 0, 401 },
		{ 0, 0, 402 },
		{ -2853, 5221, 397 },
		{ -2854, 5026, 0 },
		{ 3151, 5003, 0 },
		{ 3095, 4981, 0 },
		{ 0, 0, 398 },
		{ 3095, 5002, 0 },
		{ -2859, 16, 0 },
		{ -2860, 5030, 0 },
		{ 2863, 0, 399 },
		{ 3095, 5004, 0 },
		{ 3151, 5135, 0 },
		{ 0, 0, 400 },
		{ 3142, 3550, 164 },
		{ 0, 0, 164 },
		{ 0, 0, 165 },
		{ 3166, 2068, 0 },
		{ 3188, 3026, 0 },
		{ 3205, 2312, 0 },
		{ 2872, 5054, 0 },
		{ 3198, 2650, 0 },
		{ 3193, 1600, 0 },
		{ 3147, 2506, 0 },
		{ 3200, 3159, 0 },
		{ 3201, 1835, 0 },
		{ 3088, 2911, 0 },
		{ 3203, 2417, 0 },
		{ 3147, 2510, 0 },
		{ 3166, 2076, 0 },
		{ 3206, 4653, 0 },
		{ 0, 0, 162 },
		{ 3021, 4793, 187 },
		{ 0, 0, 187 },
		{ 3201, 1837, 0 },
		{ 2887, 5078, 0 },
		{ 3188, 2715, 0 },
		{ 3199, 3195, 0 },
		{ 3200, 3148, 0 },
		{ 3195, 2930, 0 },
		{ 2892, 5086, 0 },
		{ 3151, 2533, 0 },
		{ 3188, 3042, 0 },
		{ 3147, 2516, 0 },
		{ 3188, 3044, 0 },
		{ 3205, 2238, 0 },
		{ 3199, 3205, 0 },
		{ 3201, 1849, 0 },
		{ 3088, 2709, 0 },
		{ 3203, 2425, 0 },
		{ 3147, 2520, 0 },
		{ 2903, 5116, 0 },
		{ 3151, 2943, 0 },
		{ 3188, 3053, 0 },
		{ 3034, 2965, 0 },
		{ 3203, 2426, 0 },
		{ 3205, 2241, 0 },
		{ 3188, 3057, 0 },
		{ 2910, 4964, 0 },
		{ 3205, 2156, 0 },
		{ 3188, 3059, 0 },
		{ 3185, 3121, 0 },
		{ 3193, 1639, 0 },
		{ 3200, 3164, 0 },
		{ 3188, 3061, 0 },
		{ 2917, 4987, 0 },
		{ 3198, 2658, 0 },
		{ 3193, 1640, 0 },
		{ 3147, 2527, 0 },
		{ 3200, 3146, 0 },
		{ 3201, 1866, 0 },
		{ 3088, 2853, 0 },
		{ 3203, 2352, 0 },
		{ 3147, 2533, 0 },
		{ 3206, 4871, 0 },
		{ 0, 0, 185 },
		{ 2928, 0, 1 },
		{ -2928, 1280, 276 },
		{ 3188, 2972, 282 },
		{ 0, 0, 282 },
		{ 3166, 2123, 0 },
		{ 3147, 2539, 0 },
		{ 3188, 3078, 0 },
		{ 3185, 3125, 0 },
		{ 3205, 2250, 0 },
		{ 0, 0, 281 },
		{ 2938, 5054, 0 },
		{ 3190, 2160, 0 },
		{ 3199, 3252, 0 },
		{ 2967, 2693, 0 },
		{ 3188, 3082, 0 },
		{ 3034, 2968, 0 },
		{ 3088, 2872, 0 },
		{ 3196, 2661, 0 },
		{ 3188, 3087, 0 },
		{ 2947, 5035, 0 },
		{ 3203, 2240, 0 },
		{ 0, 2333, 0 },
		{ 3201, 1896, 0 },
		{ 3088, 2878, 0 },
		{ 3203, 2364, 0 },
		{ 3147, 2445, 0 },
		{ 3166, 2136, 0 },
		{ 3206, 4725, 0 },
		{ 0, 0, 280 },
		{ 0, 4591, 190 },
		{ 0, 0, 190 },
		{ 3203, 2366, 0 },
		{ 3193, 1494, 0 },
		{ 3147, 2449, 0 },
		{ 3185, 3128, 0 },
		{ 2963, 5073, 0 },
		{ 3200, 2936, 0 },
		{ 3195, 2939, 0 },
		{ 3188, 3102, 0 },
		{ 3200, 3154, 0 },
		{ 0, 2695, 0 },
		{ 3088, 2887, 0 },
		{ 3147, 2450, 0 },
		{ 2990, 2948, 0 },
		{ 3206, 4803, 0 },
		{ 0, 0, 188 },
		{ 3021, 4795, 184 },
		{ 0, 0, 183 },
		{ 0, 0, 184 },
		{ 3201, 1898, 0 },
		{ 2978, 5078, 0 },
		{ 3201, 2109, 0 },
		{ 3195, 2927, 0 },
		{ 3188, 3112, 0 },
		{ 2982, 5101, 0 },
		{ 3151, 2939, 0 },
		{ 3188, 3114, 0 },
		{ 2990, 2944, 0 },
		{ 3088, 2892, 0 },
		{ 3147, 2453, 0 },
		{ 3147, 2454, 0 },
		{ 3088, 2895, 0 },
		{ 3147, 2456, 0 },
		{ 0, 2950, 0 },
		{ 2992, 5108, 0 },
		{ 3203, 2254, 0 },
		{ 3034, 2988, 0 },
		{ 2995, 4973, 0 },
		{ 3188, 2712, 0 },
		{ 3199, 3234, 0 },
		{ 3200, 3162, 0 },
		{ 3195, 2925, 0 },
		{ 3000, 4977, 0 },
		{ 3151, 2541, 0 },
		{ 3188, 3001, 0 },
		{ 3147, 2462, 0 },
		{ 3188, 3003, 0 },
		{ 3205, 2261, 0 },
		{ 3199, 3245, 0 },
		{ 3201, 1901, 0 },
		{ 3088, 2903, 0 },
		{ 3203, 2373, 0 },
		{ 3147, 2467, 0 },
		{ 3011, 4996, 0 },
		{ 3198, 2644, 0 },
		{ 3193, 1524, 0 },
		{ 3147, 2469, 0 },
		{ 3200, 3157, 0 },
		{ 3201, 1904, 0 },
		{ 3088, 2910, 0 },
		{ 3203, 2376, 0 },
		{ 3147, 2472, 0 },
		{ 3206, 4729, 0 },
		{ 0, 0, 177 },
		{ 0, 4651, 176 },
		{ 0, 0, 176 },
		{ 3201, 1928, 0 },
		{ 3025, 5002, 0 },
		{ 3201, 2111, 0 },
		{ 3195, 2934, 0 },
		{ 3188, 3021, 0 },
		{ 3029, 5029, 0 },
		{ 3188, 2700, 0 },
		{ 3147, 2475, 0 },
		{ 3185, 3122, 0 },
		{ 3033, 5024, 0 },
		{ 3203, 2271, 0 },
		{ 0, 2962, 0 },
		{ 3036, 5037, 0 },
		{ 3188, 2708, 0 },
		{ 3199, 3202, 0 },
		{ 3200, 3151, 0 },
		{ 3195, 2923, 0 },
		{ 3041, 5039, 0 },
		{ 3151, 2535, 0 },
		{ 3188, 3029, 0 },
		{ 3147, 2478, 0 },
		{ 3188, 3031, 0 },
		{ 3205, 2270, 0 },
		{ 3199, 3210, 0 },
		{ 3201, 1935, 0 },
		{ 3088, 2727, 0 },
		{ 3203, 2382, 0 },
		{ 3147, 2482, 0 },
		{ 3052, 5062, 0 },
		{ 3198, 2660, 0 },
		{ 3193, 1564, 0 },
		{ 3147, 2484, 0 },
		{ 3200, 3143, 0 },
		{ 3201, 1948, 0 },
		{ 3088, 2776, 0 },
		{ 3203, 2385, 0 },
		{ 3147, 2487, 0 },
		{ 3206, 4887, 0 },
		{ 0, 0, 174 },
		{ 0, 4204, 179 },
		{ 0, 0, 179 },
		{ 0, 0, 180 },
		{ 3147, 2488, 0 },
		{ 3166, 2187, 0 },
		{ 3201, 1949, 0 },
		{ 3188, 3047, 0 },
		{ 3199, 3230, 0 },
		{ 3185, 3131, 0 },
		{ 3072, 5089, 0 },
		{ 3188, 2706, 0 },
		{ 3170, 1490, 0 },
		{ 3199, 3235, 0 },
		{ 3196, 2675, 0 },
		{ 3193, 1566, 0 },
		{ 3199, 3238, 0 },
		{ 3201, 1952, 0 },
		{ 3088, 2859, 0 },
		{ 3203, 2391, 0 },
		{ 3147, 2495, 0 },
		{ 3083, 5105, 0 },
		{ 3198, 2646, 0 },
		{ 3193, 1567, 0 },
		{ 3147, 2497, 0 },
		{ 3200, 3144, 0 },
		{ 3201, 1955, 0 },
		{ 0, 2873, 0 },
		{ 3203, 2394, 0 },
		{ 3147, 2500, 0 },
		{ 3168, 4938, 0 },
		{ 0, 0, 178 },
		{ 3188, 3065, 446 },
		{ 3205, 1471, 26 },
		{ 0, 4995, 446 },
		{ 3104, 0, 446 },
		{ 2293, 2821, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 3147, 2504, 0 },
		{ -3103, 5222, 0 },
		{ 3205, 661, 0 },
		{ 0, 0, 28 },
		{ 3185, 3135, 0 },
		{ 0, 0, 27 },
		{ 0, 0, 22 },
		{ 0, 0, 34 },
		{ 0, 0, 35 },
		{ 3125, 4020, 39 },
		{ 0, 3747, 39 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 3137, 3890, 0 },
		{ 3152, 4553, 0 },
		{ 3142, 3524, 0 },
		{ 0, 0, 37 },
		{ 3146, 3584, 0 },
		{ 3150, 3426, 0 },
		{ 3160, 2047, 0 },
		{ 0, 0, 36 },
		{ 3188, 2987, 50 },
		{ 0, 0, 50 },
		{ 0, 4029, 50 },
		{ 3142, 3529, 50 },
		{ 3188, 3075, 50 },
		{ 0, 0, 58 },
		{ 3188, 3076, 0 },
		{ 3147, 2509, 0 },
		{ 3137, 3901, 0 },
		{ 3146, 3593, 0 },
		{ 3201, 1965, 0 },
		{ 3147, 2511, 0 },
		{ 3185, 3123, 0 },
		{ 3142, 3539, 0 },
		{ 0, 3907, 0 },
		{ 3193, 1602, 0 },
		{ 3203, 2404, 0 },
		{ 0, 0, 49 },
		{ 3146, 3600, 0 },
		{ 0, 3542, 0 },
		{ 3205, 2295, 0 },
		{ 3190, 2598, 0 },
		{ 3150, 3423, 54 },
		{ 0, 3605, 0 },
		{ 0, 2515, 0 },
		{ 3188, 3085, 0 },
		{ 3154, 1783, 0 },
		{ 0, 3425, 51 },
		{ 0, 5, 59 },
		{ 0, 4524, 0 },
		{ 3160, 1839, 0 },
		{ 3205, 1623, 0 },
		{ 3161, 1773, 0 },
		{ 0, 0, 57 },
		{ 3196, 2662, 0 },
		{ 0, 0, 55 },
		{ 0, 0, 56 },
		{ 3097, 1950, 0 },
		{ 3205, 1698, 0 },
		{ 3199, 3217, 0 },
		{ 0, 0, 52 },
		{ 0, 0, 53 },
		{ 3166, 2222, 0 },
		{ 0, 2223, 0 },
		{ 3168, 4932, 0 },
		{ 0, 4934, 0 },
		{ 3188, 3095, 0 },
		{ 0, 1492, 0 },
		{ 3199, 3222, 0 },
		{ 3196, 2668, 0 },
		{ 3193, 1672, 0 },
		{ 3199, 3225, 0 },
		{ 3201, 2028, 0 },
		{ 3203, 2419, 0 },
		{ 3205, 2307, 0 },
		{ 3199, 2213, 0 },
		{ 3188, 3104, 0 },
		{ 3203, 2423, 0 },
		{ 3200, 3150, 0 },
		{ 3199, 3233, 0 },
		{ 3205, 2309, 0 },
		{ 3200, 3152, 0 },
		{ 0, 3129, 0 },
		{ 3188, 3010, 0 },
		{ 3193, 1673, 0 },
		{ 0, 3109, 0 },
		{ 3199, 3240, 0 },
		{ 0, 2594, 0 },
		{ 3205, 2311, 0 },
		{ 3200, 3160, 0 },
		{ 0, 1675, 0 },
		{ 3206, 4933, 0 },
		{ 0, 2932, 0 },
		{ 0, 2680, 0 },
		{ 0, 0, 46 },
		{ 3151, 2938, 0 },
		{ 0, 3248, 0 },
		{ 0, 3165, 0 },
		{ 0, 2041, 0 },
		{ 3206, 4943, 0 },
		{ 0, 2429, 0 },
		{ 0, 0, 47 },
		{ 0, 2314, 0 },
		{ 3168, 4944, 0 },
		{ 0, 0, 48 }
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
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
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
		0,
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
