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
  /*fprintf(fpOutputFile,"#define NO_FITNESS_CASES %d\n",iNO_FITNESS_CASES);*/

#line 411 "EaseaLex.cpp"
		}
		break;
	case 17:
		{
#line 325 "EaseaLex.l"

  
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

  /*
  // count the number of variable (arity zero and non-erc operator)
  unsigned var_len = 0;
  for( unsigned i=0 ; i<iNoOp ; i++ ){
    if( opDesc[i]->arity==0 && !opDesc[i]->isERC ) var_len++;
  }
  if( bVERBOSE ) printf("var length is %d\n",var_len);
  fprintf(fpOutputFile,"#define VAR_LEN %d\n",var_len); */
 
#line 448 "EaseaLex.cpp"
		}
		break;
	case 18:
		{
#line 357 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"    case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"      %s",opDesc[i]->gpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"      break;\n");

  }
 
#line 462 "EaseaLex.cpp"
		}
		break;
	case 19:
		{
#line 366 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"  case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"    %s\n",opDesc[i]->cpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"    break;\n");
  }
 
#line 475 "EaseaLex.cpp"
		}
		break;
	case 20:
		{
#line 375 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Analysing GP OP code from ez file\n");
  BEGIN COPY_GP_OPCODE;
 
#line 488 "EaseaLex.cpp"
		}
		break;
	case 21:
		{
#line 384 "EaseaLex.l"

  if (bVERBOSE) printf ("found begin section\n");
  bGPOPCODE_ANALYSIS = true;
  BEGIN GP_RULE_ANALYSIS;
 
#line 499 "EaseaLex.cpp"
		}
		break;
	case 22:
		{
#line 390 "EaseaLex.l"
 
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
 
#line 518 "EaseaLex.cpp"
		}
		break;
	case 23:
		{
#line 404 "EaseaLex.l"

  if (bVERBOSE) printf("*** No GP OP codes were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
 
#line 530 "EaseaLex.cpp"
		}
		break;
	case 24:
		{
#line 410 "EaseaLex.l"

#line 537 "EaseaLex.cpp"
		}
		break;
	case 25:
		{
#line 411 "EaseaLex.l"
if( bGPOPCODE_ANALYSIS )printf("\n");lineCounter++;
#line 544 "EaseaLex.cpp"
		}
		break;
	case 26:
		{
#line 418 "EaseaLex.l"

  // this rule match the OP_NAME
  if( iGP_OPCODE_FIELD != 0 ){
    fprintf(stderr,"Error, OP_CODE name must be given first\n");
    exit(-1);
  }
  opDesc[iNoOp] = new OPCodeDesc();
  opDesc[iNoOp]->opcode = new string(yytext);
 
#line 559 "EaseaLex.cpp"
		}
		break;
	case 27:
		{
#line 428 "EaseaLex.l"

#line 566 "EaseaLex.cpp"
		}
		break;
	case 28:
		{
#line 430 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 1 ){
    fprintf(stderr,"Error, op code real name must be given at the second place\n");
    exit(-1);
  }
  opDesc[iNoOp]->realName = new string(yytext);
 
#line 579 "EaseaLex.cpp"
		}
		break;
	case 29:
		{
#line 439 "EaseaLex.l"

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
 
#line 598 "EaseaLex.cpp"
		}
		break;
	case 30:
		{
#line 453 "EaseaLex.l"

#line 605 "EaseaLex.cpp"
		}
		break;
	case 31:
		{
#line 454 "EaseaLex.l"

  iGP_OPCODE_FIELD = 0;
  iNoOp++;
 
#line 615 "EaseaLex.cpp"
		}
		break;
	case 32:
		{
#line 459 "EaseaLex.l"

  if( bGPOPCODE_ANALYSIS ) iGP_OPCODE_FIELD++;
 
#line 624 "EaseaLex.cpp"
		}
		break;
	case 33:
		{
#line 464 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 3 ){
    fprintf(stderr,"Error, code must be given at the forth place\n");
    exit(-1);
  }
  accolade_counter=1;

//  printf("arity : %d\n",opDesc[iNoOp]->arity);
  if( opDesc[iNoOp]->arity>=2 )
    opDesc[iNoOp]->gpuCodeStream << "OP2 = stack[--sp];\n      ";
  if( opDesc[iNoOp]->arity>=1 )
    opDesc[iNoOp]->gpuCodeStream << "OP1 = stack[--sp];\n      ";

  BEGIN GP_COPY_OPCODE_CODE;
 
#line 645 "EaseaLex.cpp"
		}
		break;
	case 34:
		{
#line 484 "EaseaLex.l"

  accolade_counter++;
  opDesc[iNoOp]->cpuCodeStream << "{";
  opDesc[iNoOp]->gpuCodeStream << "{";
 
#line 656 "EaseaLex.cpp"
		}
		break;
	case 35:
		{
#line 490 "EaseaLex.l"

  accolade_counter--;
  if( accolade_counter==0 ){
    opDesc[iNoOp]->gpuCodeStream << "\n      stack[sp++] = RESULT;\n";

    BEGIN GP_RULE_ANALYSIS;
  }
  else{
    opDesc[iNoOp]->cpuCodeStream << "}";
    opDesc[iNoOp]->gpuCodeStream << "}";
  }
 
#line 674 "EaseaLex.cpp"
		}
		break;
	case 36:
		{
#line 503 "EaseaLex.l"

  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
//  printf("input no : %d\n",no_input);
  opDesc[iNoOp]->cpuCodeStream << "input["<< no_input <<"]" ;
  opDesc[iNoOp]->gpuCodeStream << "input["<< no_input << "]";  
 
#line 687 "EaseaLex.cpp"
		}
		break;
	case 37:
		{
#line 511 "EaseaLex.l"

  opDesc[iNoOp]->isERC = true;
  opDesc[iNoOp]->cpuCodeStream << "root->erc_value" ;
  opDesc[iNoOp]->gpuCodeStream << "k_progs[start_prog++];" ;
//  printf("ERC matched\n");

#line 699 "EaseaLex.cpp"
		}
		break;
	case 38:
		{
#line 518 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << "\n  ";
  opDesc[iNoOp]->gpuCodeStream << "\n    ";
 
#line 709 "EaseaLex.cpp"
		}
		break;
	case 39:
		{
#line 524 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << yytext;
  opDesc[iNoOp]->gpuCodeStream << yytext;
 
#line 719 "EaseaLex.cpp"
		}
		break;
	case 40:
		{
#line 529 "EaseaLex.l"
 
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  if( bVERBOSE ) printf("Insert GP eval header\n");
  iCOPY_GP_EVAL_STATUS = EVAL_HDR;
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 735 "EaseaLex.cpp"
		}
		break;
	case 41:
		{
#line 540 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = false;
  BEGIN COPY_GP_EVAL;
 
#line 752 "EaseaLex.cpp"
		}
		break;
	case 42:
		{
#line 554 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 769 "EaseaLex.cpp"
		}
		break;
	case 43:
		{
#line 568 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 785 "EaseaLex.cpp"
		}
		break;
	case 44:
		{
#line 579 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 802 "EaseaLex.cpp"
		}
		break;
	case 45:
		{
#line 592 "EaseaLex.l"

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
 
#line 821 "EaseaLex.cpp"
		}
		break;
	case 46:
		{
#line 607 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_HDR){
    bIsCopyingGPEval = true;
  }
 
#line 832 "EaseaLex.cpp"
		}
		break;
	case 47:
		{
#line 613 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_BDY){
    bIsCopyingGPEval = true;
  }
 
#line 843 "EaseaLex.cpp"
		}
		break;
	case 48:
		{
#line 621 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_FTR){
    bIsCopyingGPEval = true;
  }
 
#line 854 "EaseaLex.cpp"
		}
		break;
	case 49:
		{
#line 627 "EaseaLex.l"

  if( bIsCopyingGPEval ){
    bIsCopyingGPEval = false;
    bCOPY_GP_EVAL_GPU = false;
    rewind(fpGenomeFile);
    yyin = fpTemplateFile;
    BEGIN TEMPLATE_ANALYSIS;
  }
 
#line 869 "EaseaLex.cpp"
		}
		break;
	case 50:
		{
#line 637 "EaseaLex.l"

  if( bIsCopyingGPEval ) fprintf(fpOutputFile,"%s",yytext);
 
#line 878 "EaseaLex.cpp"
		}
		break;
	case 51:
		{
#line 641 "EaseaLex.l"

  if( bIsCopyingGPEval) 
    //if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[i]" );
    //else fprintf(fpOutputFile, "outputs[i]" );
  
 
#line 891 "EaseaLex.cpp"
		}
		break;
	case 52:
		{
#line 649 "EaseaLex.l"

  char* endptr;
  unsigned no_output = strtol(yytext+strlen("OUTPUT["),&endptr,10);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[(i+%d)*NUMTHREAD+tid]", no_output);
    else fprintf(fpOutputFile, "outputs[i+%d]", no_output );
  
 
#line 906 "EaseaLex.cpp"
		}
		break;
	case 53:
		{
#line 659 "EaseaLex.l"

	char *var;
	var = strndup(yytext+strlen("OUTPUT["), strlen(yytext) - strlen("OUTPUT[") - 1);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[(i+%s)*NUMTHREAD+tid]", var);
    else fprintf(fpOutputFile, "outputs[i+%s]", var);
  
 
#line 921 "EaseaLex.cpp"
		}
		break;
	case 54:
		{
#line 669 "EaseaLex.l"

  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[i*NUMTHREAD+tid]" );
    else fprintf(fpOutputFile, "inputs[i][0]" );
  
 
#line 934 "EaseaLex.cpp"
		}
		break;
	case 55:
		{
#line 677 "EaseaLex.l"

  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[(i+%d)*NUMTHREAD+tid]", no_input);
    else fprintf(fpOutputFile, "inputs[i+%d][0]", no_input );
  
 
#line 949 "EaseaLex.cpp"
		}
		break;
	case 56:
		{
#line 687 "EaseaLex.l"

	char *var;
	var = strndup(yytext+strlen("INPUT["), strlen(yytext) - strlen("INPUT[") - 1);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[(i+%s)*NUMTHREAD+tid]", var);
    else fprintf(fpOutputFile, "inputs[i+%s][0]", var);
  
 
#line 964 "EaseaLex.cpp"
		}
		break;
	case 57:
		{
#line 697 "EaseaLex.l"

  if( bIsCopyingGPEval )
    if( iCOPY_GP_EVAL_STATUS==EVAL_FTR )
      if( bCOPY_GP_EVAL_GPU ){
	fprintf(fpOutputFile,"k_results[index] =");
      }
      else fprintf(fpOutputFile,"return fitness=");
 
#line 978 "EaseaLex.cpp"
		}
		break;
	case 58:
		{
#line 709 "EaseaLex.l"

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
 
#line 996 "EaseaLex.cpp"
		}
		break;
	case 59:
		{
#line 723 "EaseaLex.l"

  if( bIsCopyingGPEval )
    fprintf(fpOutputFile,"return fitness = "); 
 
#line 1006 "EaseaLex.cpp"
		}
		break;
	case 60:
		{
#line 730 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  yyreset();
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Evaluation population in a single function!!.\n");
  lineCounter = 1;
  BEGIN COPY_INSTEAD_EVAL;
 
#line 1020 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 739 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting at the end of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bEndGeneration = true;
  bBeginGeneration = false;
  BEGIN COPY_END_GENERATION_FUNCTION;
 
#line 1034 "EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 748 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Bound Checking function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_BOUND_CHECKING_FUNCTION;
 
#line 1046 "EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 755 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 
#line 1058 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 762 "EaseaLex.l"

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
 
#line 1087 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 785 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 
#line 1104 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 797 "EaseaLex.l"

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
 
#line 1130 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 818 "EaseaLex.l"

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
  
 
#line 1151 "EaseaLex.cpp"
		}
		break;
#line 836 "EaseaLex.l"
  
#line 850 "EaseaLex.l"
      
#line 1158 "EaseaLex.cpp"
	case 68:
		{
#line 858 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 1171 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 867 "EaseaLex.l"

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
 
#line 1194 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 884 "EaseaLex.l"

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
 
#line 1217 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 902 "EaseaLex.l"

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
 
#line 1249 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 929 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  fprintf (fpOutputFile,"// Memberwise serialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->serializeIndividual(fpOutputFile, "this");
  //fprintf(fpOutputFile,"\tEASEA_Line << endl;\n");
 
#line 1263 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 938 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome deserializer.\n");
  fprintf (fpOutputFile,"// Memberwise deserialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->deserializeIndividual(fpOutputFile, "this");
 
#line 1276 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 946 "EaseaLex.l"

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
 
#line 1297 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 962 "EaseaLex.l"

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
 
#line 1319 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 979 "EaseaLex.l"
       
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
 
#line 1347 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 1001 "EaseaLex.l"

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
 
#line 1369 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 1017 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 1384 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 1026 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 1396 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 1034 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 1408 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 1041 "EaseaLex.l"

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
 
#line 1439 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 1066 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 1452 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 1073 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 1466 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 1082 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISER;   
 
#line 1478 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 1089 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 1491 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 1097 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 1503 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 1103 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 1515 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 1109 "EaseaLex.l"

  printf("evaluator insert\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 1528 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 1116 "EaseaLex.l"
      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 1541 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 1123 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 1554 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 1130 "EaseaLex.l"
 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 1568 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 1139 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 1579 "EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 1144 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 1593 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 1153 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 1607 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 1162 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 1621 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 1172 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 1634 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 1180 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 1643 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 1184 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 1652 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 1188 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 1661 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 1192 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 1670 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 1196 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 1680 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 1201 "EaseaLex.l"

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

#line 1699 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 1214 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 1706 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 1215 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 1713 "EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 1216 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 1720 "EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 1217 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 1727 "EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 1218 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 1734 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 1219 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1741 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 1220 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1748 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 1221 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1755 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 1222 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1762 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 1223 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1769 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 1224 "EaseaLex.l"
 fprintf(fpOutputFile,"%d",nELITE); 
#line 1776 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 1225 "EaseaLex.l"
 fprintf(fpOutputFile,"%d",iNO_FITNESS_CASES); 
#line 1783 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 1228 "EaseaLex.l"

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
 
#line 1802 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 1241 "EaseaLex.l"

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
 
#line 1821 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 1254 "EaseaLex.l"

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
 
#line 1840 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 1267 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1850 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 1271 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1857 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 1272 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1864 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 1273 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1871 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1274 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1878 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1275 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1885 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1276 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1892 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1277 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1899 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1278 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1906 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1279 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1913 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1280 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1920 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1282 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1927 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1283 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1934 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1285 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bREMOTE_ISLAND_MODEL);
#line 1941 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1286 "EaseaLex.l"
if(strlen(sIP_FILE)>0)fprintf(fpOutputFile,"%s",sIP_FILE); else fprintf(fpOutputFile,"NULL");
#line 1948 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1287 "EaseaLex.l"
if(strlen(sEXPID)>0)fprintf(fpOutputFile,"%s",sEXPID); else fprintf(fpOutputFile,"NULL");
#line 1955 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1288 "EaseaLex.l"
if(strlen(sWORKING_PATH)>0)fprintf(fpOutputFile,"%s",sWORKING_PATH); else fprintf(fpOutputFile,"NULL");
#line 1962 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1289 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMIGRATION_PROBABILITY);
#line 1969 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1290 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nSERVER_PORT);
#line 1976 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1292 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1983 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1293 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1990 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1294 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1997 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1295 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 2004 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1296 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 2011 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1298 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 2018 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1299 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 2025 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1301 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 2039 "EaseaLex.cpp"
		}
		break;
	case 145:
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
 
#line 2056 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1320 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2070 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1328 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2084 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1337 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2098 "EaseaLex.cpp"
		}
		break;
	case 149:
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

#line 2161 "EaseaLex.cpp"
		}
		break;
	case 150:
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
 
#line 2178 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1415 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2185 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1421 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2197 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1427 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2210 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1434 "EaseaLex.l"

#line 2217 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1435 "EaseaLex.l"
lineCounter++;
#line 2224 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1437 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2236 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1443 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2249 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1451 "EaseaLex.l"

#line 2256 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1452 "EaseaLex.l"

  lineCounter++;
 
#line 2265 "EaseaLex.cpp"
		}
		break;
#line 1455 "EaseaLex.l"
               
#line 2270 "EaseaLex.cpp"
	case 160:
		{
#line 1456 "EaseaLex.l"

  fprintf (fpOutputFile,"// User CUDA\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2280 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1462 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user CUDA were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2293 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1470 "EaseaLex.l"

#line 2300 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1471 "EaseaLex.l"

  lineCounter++;
 
#line 2309 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1475 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2321 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1481 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2335 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1489 "EaseaLex.l"

#line 2342 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1490 "EaseaLex.l"

  lineCounter++;
 
#line 2351 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1494 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
      
  BEGIN COPY;
 
#line 2365 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1502 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2380 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1511 "EaseaLex.l"

#line 2387 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1512 "EaseaLex.l"
lineCounter++;
#line 2394 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1517 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2408 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1526 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2422 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1534 "EaseaLex.l"

#line 2429 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1535 "EaseaLex.l"
lineCounter++;
#line 2436 "EaseaLex.cpp"
		}
		break;
	case 176:
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
 
#line 2452 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1549 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2468 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1559 "EaseaLex.l"

#line 2475 "EaseaLex.cpp"
		}
		break;
	case 179:
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
 
#line 2493 "EaseaLex.cpp"
		}
		break;
	case 180:
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
 
#line 2510 "EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 1587 "EaseaLex.l"

#line 2517 "EaseaLex.cpp"
		}
		break;
	case 182:
		{
#line 1588 "EaseaLex.l"
lineCounter++;
#line 2524 "EaseaLex.cpp"
		}
		break;
	case 183:
		{
#line 1590 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2540 "EaseaLex.cpp"
		}
		break;
	case 184:
		{
#line 1602 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2556 "EaseaLex.cpp"
		}
		break;
	case 185:
		{
#line 1612 "EaseaLex.l"
lineCounter++;
#line 2563 "EaseaLex.cpp"
		}
		break;
	case 186:
		{
#line 1613 "EaseaLex.l"

#line 2570 "EaseaLex.cpp"
		}
		break;
	case 187:
		{
#line 1617 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2585 "EaseaLex.cpp"
		}
		break;
	case 188:
		{
#line 1627 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2600 "EaseaLex.cpp"
		}
		break;
	case 189:
		{
#line 1636 "EaseaLex.l"

#line 2607 "EaseaLex.cpp"
		}
		break;
	case 190:
		{
#line 1639 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    //fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    fprintf (fpOutputFile,"{\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2621 "EaseaLex.cpp"
		}
		break;
	case 191:
		{
#line 1647 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2635 "EaseaLex.cpp"
		}
		break;
	case 192:
		{
#line 1655 "EaseaLex.l"

#line 2642 "EaseaLex.cpp"
		}
		break;
	case 193:
		{
#line 1659 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2650 "EaseaLex.cpp"
		}
		break;
	case 194:
		{
#line 1661 "EaseaLex.l"

#line 2657 "EaseaLex.cpp"
		}
		break;
	case 195:
		{
#line 1667 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2664 "EaseaLex.cpp"
		}
		break;
	case 196:
		{
#line 1668 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2671 "EaseaLex.cpp"
		}
		break;
	case 197:
	case 198:
		{
#line 1671 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2682 "EaseaLex.cpp"
		}
		break;
	case 199:
	case 200:
		{
#line 1676 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2691 "EaseaLex.cpp"
		}
		break;
	case 201:
	case 202:
		{
#line 1679 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2700 "EaseaLex.cpp"
		}
		break;
	case 203:
	case 204:
		{
#line 1682 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2717 "EaseaLex.cpp"
		}
		break;
	case 205:
	case 206:
		{
#line 1693 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2731 "EaseaLex.cpp"
		}
		break;
	case 207:
	case 208:
		{
#line 1701 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2740 "EaseaLex.cpp"
		}
		break;
	case 209:
	case 210:
		{
#line 1704 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2749 "EaseaLex.cpp"
		}
		break;
	case 211:
	case 212:
		{
#line 1707 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2758 "EaseaLex.cpp"
		}
		break;
	case 213:
	case 214:
		{
#line 1710 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2767 "EaseaLex.cpp"
		}
		break;
	case 215:
	case 216:
		{
#line 1713 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2776 "EaseaLex.cpp"
		}
		break;
	case 217:
	case 218:
		{
#line 1717 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2788 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1723 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2795 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1724 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2802 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1725 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2809 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1726 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2819 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1731 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2826 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1732 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2833 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1733 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2840 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1734 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2847 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1735 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2854 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1736 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2861 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1737 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2868 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1738 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2875 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1739 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2883 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1741 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2891 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1743 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2899 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1745 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2909 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1749 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2916 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1750 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2923 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1751 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2934 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1756 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2941 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1757 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2950 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1760 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2962 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1766 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2971 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1769 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2983 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1775 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2994 "EaseaLex.cpp"
		}
		break;
	case 244:
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
#line 3010 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1790 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3017 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1793 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 3026 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1796 "EaseaLex.l"
BEGIN COPY;
#line 3033 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1798 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 3040 "EaseaLex.cpp"
		}
		break;
	case 249:
	case 250:
	case 251:
		{
#line 1801 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 3053 "EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1806 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 3064 "EaseaLex.cpp"
		}
		break;
	case 253:
		{
#line 1811 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 3073 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1820 "EaseaLex.l"
;
#line 3080 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1821 "EaseaLex.l"
;
#line 3087 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1822 "EaseaLex.l"
;
#line 3094 "EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1823 "EaseaLex.l"
;
#line 3101 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1826 "EaseaLex.l"
 /* do nothing */ 
#line 3108 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1827 "EaseaLex.l"
 /*return '\n';*/ 
#line 3115 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1828 "EaseaLex.l"
 /*return '\n';*/ 
#line 3122 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1831 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 3131 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1834 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 3141 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1838 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
//  printf("match gpnode\n");
  return GPNODE;
 
#line 3153 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1845 "EaseaLex.l"
return STATIC;
#line 3160 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1846 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 3167 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1847 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 3174 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1848 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 3181 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1849 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 3188 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1850 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 3195 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1852 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 3202 "EaseaLex.cpp"
		}
		break;
#line 1853 "EaseaLex.l"
  
#line 3207 "EaseaLex.cpp"
	case 271:
		{
#line 1854 "EaseaLex.l"
return GENOME; 
#line 3212 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1856 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 3222 "EaseaLex.cpp"
		}
		break;
	case 273:
	case 274:
	case 275:
		{
#line 1863 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 3231 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1864 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 3238 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1867 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 3246 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1869 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 3253 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1875 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 3265 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1881 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 3278 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1888 "EaseaLex.l"

#line 3285 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1890 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3296 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1901 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3311 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1911 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3322 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1917 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3331 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1921 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3346 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1934 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3358 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1940 "EaseaLex.l"

#line 3365 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1941 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3378 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1948 "EaseaLex.l"

#line 3385 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1949 "EaseaLex.l"
lineCounter++;
#line 3392 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1950 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3405 "EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1957 "EaseaLex.l"

#line 3412 "EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1958 "EaseaLex.l"
lineCounter++;
#line 3419 "EaseaLex.cpp"
		}
		break;
	case 295:
		{
#line 1960 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3432 "EaseaLex.cpp"
		}
		break;
	case 296:
		{
#line 1967 "EaseaLex.l"

#line 3439 "EaseaLex.cpp"
		}
		break;
	case 297:
		{
#line 1968 "EaseaLex.l"
lineCounter++;
#line 3446 "EaseaLex.cpp"
		}
		break;
	case 298:
		{
#line 1971 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3459 "EaseaLex.cpp"
		}
		break;
	case 299:
		{
#line 1978 "EaseaLex.l"

#line 3466 "EaseaLex.cpp"
		}
		break;
	case 300:
		{
#line 1979 "EaseaLex.l"
lineCounter++;
#line 3473 "EaseaLex.cpp"
		}
		break;
	case 301:
		{
#line 1985 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3480 "EaseaLex.cpp"
		}
		break;
	case 302:
		{
#line 1986 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3487 "EaseaLex.cpp"
		}
		break;
	case 303:
		{
#line 1987 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3494 "EaseaLex.cpp"
		}
		break;
	case 304:
		{
#line 1988 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3501 "EaseaLex.cpp"
		}
		break;
	case 305:
		{
#line 1989 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3508 "EaseaLex.cpp"
		}
		break;
	case 306:
		{
#line 1990 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3515 "EaseaLex.cpp"
		}
		break;
	case 307:
		{
#line 1991 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3522 "EaseaLex.cpp"
		}
		break;
	case 308:
		{
#line 1993 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3531 "EaseaLex.cpp"
		}
		break;
	case 309:
		{
#line 1996 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3544 "EaseaLex.cpp"
		}
		break;
	case 310:
	case 311:
		{
#line 2005 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3555 "EaseaLex.cpp"
		}
		break;
	case 312:
	case 313:
		{
#line 2010 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3564 "EaseaLex.cpp"
		}
		break;
	case 314:
	case 315:
		{
#line 2013 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3573 "EaseaLex.cpp"
		}
		break;
	case 316:
	case 317:
		{
#line 2016 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3585 "EaseaLex.cpp"
		}
		break;
	case 318:
	case 319:
		{
#line 2022 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3598 "EaseaLex.cpp"
		}
		break;
	case 320:
	case 321:
		{
#line 2029 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3607 "EaseaLex.cpp"
		}
		break;
	case 322:
	case 323:
		{
#line 2032 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3616 "EaseaLex.cpp"
		}
		break;
	case 324:
	case 325:
		{
#line 2035 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3625 "EaseaLex.cpp"
		}
		break;
	case 326:
	case 327:
		{
#line 2038 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3634 "EaseaLex.cpp"
		}
		break;
	case 328:
	case 329:
		{
#line 2041 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3643 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 2044 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3652 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 2047 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3662 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 2051 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3670 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 2053 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3681 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 2058 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3692 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 2063 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3700 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 2065 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3708 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 2067 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3716 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 2069 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3724 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 2071 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3732 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 2073 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3739 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 2074 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3746 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 2075 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3754 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 2077 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3762 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 2079 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3770 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 2081 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3777 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 2082 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3789 "EaseaLex.cpp"
		}
		break;
	case 347:
		{
#line 2088 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3798 "EaseaLex.cpp"
		}
		break;
	case 348:
		{
#line 2091 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3808 "EaseaLex.cpp"
		}
		break;
	case 349:
		{
#line 2095 "EaseaLex.l"
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
#line 3825 "EaseaLex.cpp"
		}
		break;
	case 350:
	case 351:
		{
#line 2107 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3835 "EaseaLex.cpp"
		}
		break;
	case 352:
		{
#line 2110 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3842 "EaseaLex.cpp"
		}
		break;
	case 353:
		{
#line 2117 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3849 "EaseaLex.cpp"
		}
		break;
	case 354:
		{
#line 2118 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3856 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 2119 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3863 "EaseaLex.cpp"
		}
		break;
	case 356:
		{
#line 2120 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3870 "EaseaLex.cpp"
		}
		break;
	case 357:
		{
#line 2121 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3877 "EaseaLex.cpp"
		}
		break;
	case 358:
		{
#line 2123 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3886 "EaseaLex.cpp"
		}
		break;
	case 359:
		{
#line 2127 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3899 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 2135 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3912 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 2144 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3925 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 2153 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3940 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 2163 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3947 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 2164 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3954 "EaseaLex.cpp"
		}
		break;
	case 365:
	case 366:
		{
#line 2167 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3965 "EaseaLex.cpp"
		}
		break;
	case 367:
	case 368:
		{
#line 2172 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3974 "EaseaLex.cpp"
		}
		break;
	case 369:
	case 370:
		{
#line 2175 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3983 "EaseaLex.cpp"
		}
		break;
	case 371:
	case 372:
		{
#line 2178 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3996 "EaseaLex.cpp"
		}
		break;
	case 373:
	case 374:
		{
#line 2185 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 4009 "EaseaLex.cpp"
		}
		break;
	case 375:
	case 376:
		{
#line 2192 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 4018 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2195 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 4025 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2196 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4032 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2197 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4039 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2198 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 4049 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2203 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4056 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2204 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4063 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2205 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 4070 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2206 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 4077 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2207 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 4085 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2209 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 4093 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2211 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 4101 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2213 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 4109 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2215 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 4117 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2217 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 4125 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2219 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 4133 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2221 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 4140 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2222 "EaseaLex.l"
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
#line 4163 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2239 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 4174 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2244 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 4188 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2252 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 4195 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2258 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 4205 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2262 "EaseaLex.l"

#line 4212 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2265 "EaseaLex.l"
;
#line 4219 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2266 "EaseaLex.l"
;
#line 4226 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2267 "EaseaLex.l"
;
#line 4233 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2268 "EaseaLex.l"
;
#line 4240 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2270 "EaseaLex.l"
 /* do nothing */ 
#line 4247 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2271 "EaseaLex.l"
 /*return '\n';*/ 
#line 4254 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2272 "EaseaLex.l"
 /*return '\n';*/ 
#line 4261 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2274 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 4268 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2275 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 4275 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2276 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 4282 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2277 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4289 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2278 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4296 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2279 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4303 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2280 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4310 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2281 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4317 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2282 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4324 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2284 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4331 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2285 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4338 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2286 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4345 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2287 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4352 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2288 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4359 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2290 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4366 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2291 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4373 "EaseaLex.cpp"
		}
		break;
	case 422:
		{
#line 2293 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4384 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2298 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4391 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2300 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4402 "EaseaLex.cpp"
		}
		break;
	case 425:
		{
#line 2305 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4409 "EaseaLex.cpp"
		}
		break;
	case 426:
		{
#line 2308 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4416 "EaseaLex.cpp"
		}
		break;
	case 427:
		{
#line 2309 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4423 "EaseaLex.cpp"
		}
		break;
	case 428:
		{
#line 2310 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4430 "EaseaLex.cpp"
		}
		break;
	case 429:
		{
#line 2311 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4437 "EaseaLex.cpp"
		}
		break;
	case 430:
		{
#line 2312 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4444 "EaseaLex.cpp"
		}
		break;
#line 2313 "EaseaLex.l"
 
#line 4449 "EaseaLex.cpp"
	case 431:
		{
#line 2314 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4454 "EaseaLex.cpp"
		}
		break;
	case 432:
		{
#line 2315 "EaseaLex.l"
if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;
#line 4461 "EaseaLex.cpp"
		}
		break;
	case 433:
		{
#line 2316 "EaseaLex.l"
if(bVERBOSE) printf("\tExperiment Id...\n"); return EXPID;
#line 4468 "EaseaLex.cpp"
		}
		break;
	case 434:
		{
#line 2317 "EaseaLex.l"
if(bVERBOSE) printf("\tGrid Working Path...\n"); return WORKING_PATH;
#line 4475 "EaseaLex.cpp"
		}
		break;
	case 435:
		{
#line 2318 "EaseaLex.l"
if(bVERBOSE) printf("\tMigration Probability...\n"); return MIGRATION_PROBABILITY;
#line 4482 "EaseaLex.cpp"
		}
		break;
	case 436:
		{
#line 2319 "EaseaLex.l"
if(bVERBOSE) printf("\tServer port...\n"); return SERVER_PORT;
#line 4489 "EaseaLex.cpp"
		}
		break;
#line 2321 "EaseaLex.l"
 
#line 4494 "EaseaLex.cpp"
	case 437:
	case 438:
	case 439:
		{
#line 2325 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4501 "EaseaLex.cpp"
		}
		break;
	case 440:
		{
#line 2326 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4508 "EaseaLex.cpp"
		}
		break;
	case 441:
		{
#line 2329 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4516 "EaseaLex.cpp"
		}
		break;
	case 442:
		{
#line 2332 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return PATH_IDENTIFIER; 
#line 4524 "EaseaLex.cpp"
		}
		break;
	case 443:
		{
#line 2337 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4531 "EaseaLex.cpp"
		}
		break;
	case 444:
		{
#line 2339 "EaseaLex.l"

  lineCounter++;

#line 4540 "EaseaLex.cpp"
		}
		break;
	case 445:
		{
#line 2342 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4550 "EaseaLex.cpp"
		}
		break;
	case 446:
		{
#line 2347 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4560 "EaseaLex.cpp"
		}
		break;
	case 447:
		{
#line 2352 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4570 "EaseaLex.cpp"
		}
		break;
	case 448:
		{
#line 2357 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4580 "EaseaLex.cpp"
		}
		break;
	case 449:
		{
#line 2362 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4590 "EaseaLex.cpp"
		}
		break;
	case 450:
		{
#line 2367 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4600 "EaseaLex.cpp"
		}
		break;
	case 451:
		{
#line 2376 "EaseaLex.l"
return  (char)yytext[0];
#line 4607 "EaseaLex.cpp"
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
#line 2378 "EaseaLex.l"


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
    else if( TARGET_FLAVOR == FLAVOR_GP )
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
    else if (TARGET_FLAVOR == FLAVOR_GP)
      strcat(sTemp,"GP.tpl");
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

#line 4807 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
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
		203,
		-204,
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
		215,
		-216,
		0,
		207,
		-208,
		0,
		205,
		-206,
		0,
		-247,
		0,
		-253,
		0,
		369,
		-370,
		0,
		371,
		-372,
		0,
		365,
		-366,
		0,
		314,
		-315,
		0,
		316,
		-317,
		0,
		310,
		-311,
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
		328,
		-329,
		0,
		312,
		-313,
		0,
		375,
		-376,
		0,
		320,
		-321,
		0,
		373,
		-374,
		0,
		318,
		-319,
		0,
		367,
		-368,
		0
	};
	yymatch = match;

	yytransitionmax = 5411;
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
		{ 3152, 63 },
		{ 3152, 63 },
		{ 1933, 2036 },
		{ 1558, 1541 },
		{ 1559, 1541 },
		{ 2455, 2424 },
		{ 2455, 2424 },
		{ 0, 89 },
		{ 0, 1872 },
		{ 2430, 2395 },
		{ 2430, 2395 },
		{ 73, 3 },
		{ 74, 3 },
		{ 2292, 45 },
		{ 2293, 45 },
		{ 71, 1 },
		{ 2911, 2913 },
		{ 2258, 2254 },
		{ 69, 1 },
		{ 0, 2063 },
		{ 2032, 2034 },
		{ 167, 163 },
		{ 2032, 2028 },
		{ 3152, 63 },
		{ 1406, 1405 },
		{ 3150, 63 },
		{ 1558, 1541 },
		{ 3210, 3205 },
		{ 2455, 2424 },
		{ 1427, 1426 },
		{ 1595, 1579 },
		{ 1596, 1579 },
		{ 2430, 2395 },
		{ 2033, 2029 },
		{ 73, 3 },
		{ 3154, 63 },
		{ 2292, 45 },
		{ 88, 63 },
		{ 3149, 63 },
		{ 3151, 63 },
		{ 3151, 63 },
		{ 3151, 63 },
		{ 3151, 63 },
		{ 3151, 63 },
		{ 3151, 63 },
		{ 3151, 63 },
		{ 3151, 63 },
		{ 3151, 63 },
		{ 3151, 63 },
		{ 72, 3 },
		{ 3153, 63 },
		{ 2291, 45 },
		{ 1648, 1642 },
		{ 1595, 1579 },
		{ 2456, 2424 },
		{ 1560, 1541 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 1597, 1579 },
		{ 3147, 63 },
		{ 0, 2481 },
		{ 1650, 1644 },
		{ 3148, 63 },
		{ 1524, 1503 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3148, 63 },
		{ 3155, 63 },
		{ 3159, 3158 },
		{ 2433, 2398 },
		{ 2433, 2398 },
		{ 3158, 3158 },
		{ 2442, 2406 },
		{ 2442, 2406 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 2513, 2480 },
		{ 3158, 3158 },
		{ 2547, 2513 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 2433, 2398 },
		{ 1525, 1504 },
		{ 1526, 1505 },
		{ 2442, 2406 },
		{ 1527, 1506 },
		{ 1528, 1507 },
		{ 1529, 1508 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 1530, 1509 },
		{ 3158, 3158 },
		{ 1531, 1511 },
		{ 3158, 3158 },
		{ 1534, 1514 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 2253, 42 },
		{ 1598, 1580 },
		{ 1599, 1580 },
		{ 1535, 1515 },
		{ 2040, 42 },
		{ 2521, 2489 },
		{ 2521, 2489 },
		{ 2453, 2422 },
		{ 2453, 2422 },
		{ 2463, 2431 },
		{ 2463, 2431 },
		{ 2055, 41 },
		{ 1536, 1516 },
		{ 1866, 39 },
		{ 2477, 2445 },
		{ 2477, 2445 },
		{ 1537, 1517 },
		{ 1538, 1518 },
		{ 1540, 1520 },
		{ 1541, 1521 },
		{ 1542, 1522 },
		{ 1543, 1523 },
		{ 1544, 1524 },
		{ 2253, 42 },
		{ 1598, 1580 },
		{ 2043, 42 },
		{ 1545, 1525 },
		{ 1546, 1526 },
		{ 2521, 2489 },
		{ 1547, 1527 },
		{ 2453, 2422 },
		{ 1548, 1528 },
		{ 2463, 2431 },
		{ 1549, 1529 },
		{ 2055, 41 },
		{ 1550, 1531 },
		{ 1866, 39 },
		{ 2477, 2445 },
		{ 2252, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2041, 41 },
		{ 2056, 42 },
		{ 1853, 39 },
		{ 1553, 1534 },
		{ 1600, 1580 },
		{ 2522, 2489 },
		{ 1554, 1535 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2042, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2050, 42 },
		{ 2048, 42 },
		{ 2061, 42 },
		{ 2049, 42 },
		{ 2061, 42 },
		{ 2052, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2051, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 1555, 1536 },
		{ 2044, 42 },
		{ 2046, 42 },
		{ 1556, 1538 },
		{ 2061, 42 },
		{ 1557, 1540 },
		{ 2061, 42 },
		{ 2059, 42 },
		{ 2047, 42 },
		{ 2061, 42 },
		{ 2060, 42 },
		{ 2053, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2058, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2045, 42 },
		{ 2061, 42 },
		{ 2057, 42 },
		{ 2061, 42 },
		{ 2054, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2061, 42 },
		{ 2905, 46 },
		{ 2906, 46 },
		{ 1601, 1581 },
		{ 1602, 1581 },
		{ 69, 46 },
		{ 2482, 2449 },
		{ 2482, 2449 },
		{ 1454, 1432 },
		{ 1561, 1542 },
		{ 1562, 1543 },
		{ 1563, 1544 },
		{ 2494, 2461 },
		{ 2494, 2461 },
		{ 2508, 2475 },
		{ 2508, 2475 },
		{ 1565, 1545 },
		{ 1567, 1546 },
		{ 1564, 1544 },
		{ 1568, 1547 },
		{ 1569, 1548 },
		{ 1570, 1549 },
		{ 1571, 1550 },
		{ 1566, 1545 },
		{ 2905, 46 },
		{ 1574, 1554 },
		{ 1601, 1581 },
		{ 2509, 2476 },
		{ 2509, 2476 },
		{ 2482, 2449 },
		{ 1575, 1555 },
		{ 1604, 1582 },
		{ 1605, 1582 },
		{ 1607, 1583 },
		{ 1608, 1583 },
		{ 2494, 2461 },
		{ 1576, 1556 },
		{ 2508, 2475 },
		{ 2308, 46 },
		{ 2904, 46 },
		{ 2307, 46 },
		{ 2307, 46 },
		{ 2307, 46 },
		{ 2307, 46 },
		{ 2307, 46 },
		{ 2307, 46 },
		{ 2307, 46 },
		{ 2307, 46 },
		{ 2307, 46 },
		{ 2307, 46 },
		{ 2509, 2476 },
		{ 1577, 1557 },
		{ 1579, 1561 },
		{ 1580, 1562 },
		{ 1604, 1582 },
		{ 1603, 1581 },
		{ 1607, 1583 },
		{ 2309, 46 },
		{ 2305, 46 },
		{ 2300, 46 },
		{ 2309, 46 },
		{ 2297, 46 },
		{ 2304, 46 },
		{ 2302, 46 },
		{ 2309, 46 },
		{ 2306, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2299, 46 },
		{ 2294, 46 },
		{ 2301, 46 },
		{ 2296, 46 },
		{ 2309, 46 },
		{ 2303, 46 },
		{ 2298, 46 },
		{ 2295, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 1606, 1582 },
		{ 2314, 46 },
		{ 1609, 1583 },
		{ 1581, 1563 },
		{ 2309, 46 },
		{ 1582, 1564 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2310, 46 },
		{ 2311, 46 },
		{ 2312, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2313, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 2309, 46 },
		{ 161, 4 },
		{ 162, 4 },
		{ 1610, 1584 },
		{ 1611, 1584 },
		{ 2558, 2526 },
		{ 2558, 2526 },
		{ 2564, 2532 },
		{ 2564, 2532 },
		{ 1583, 1565 },
		{ 1584, 1566 },
		{ 1585, 1567 },
		{ 2571, 2539 },
		{ 2571, 2539 },
		{ 2350, 2318 },
		{ 2350, 2318 },
		{ 1586, 1568 },
		{ 1587, 1569 },
		{ 1588, 1570 },
		{ 1589, 1571 },
		{ 1591, 1574 },
		{ 1592, 1575 },
		{ 1593, 1576 },
		{ 1594, 1577 },
		{ 161, 4 },
		{ 1457, 1433 },
		{ 1610, 1584 },
		{ 1458, 1434 },
		{ 2558, 2526 },
		{ 1462, 1436 },
		{ 2564, 2532 },
		{ 1628, 1614 },
		{ 1629, 1614 },
		{ 1637, 1627 },
		{ 1638, 1627 },
		{ 2571, 2539 },
		{ 1463, 1437 },
		{ 2350, 2318 },
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
		{ 1464, 1438 },
		{ 1613, 1585 },
		{ 1614, 1586 },
		{ 1615, 1587 },
		{ 1628, 1614 },
		{ 1612, 1584 },
		{ 1637, 1627 },
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
		{ 1630, 1614 },
		{ 83, 4 },
		{ 1639, 1627 },
		{ 1617, 1591 },
		{ 87, 4 },
		{ 1618, 1592 },
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
		{ 1442, 23 },
		{ 2585, 2555 },
		{ 2585, 2555 },
		{ 1619, 1593 },
		{ 1429, 23 },
		{ 2587, 2557 },
		{ 2587, 2557 },
		{ 2592, 2562 },
		{ 2592, 2562 },
		{ 2595, 2565 },
		{ 2595, 2565 },
		{ 2599, 2569 },
		{ 2599, 2569 },
		{ 2600, 2570 },
		{ 2600, 2570 },
		{ 2615, 2582 },
		{ 2615, 2582 },
		{ 1620, 1594 },
		{ 1627, 1613 },
		{ 1465, 1439 },
		{ 1631, 1615 },
		{ 1633, 1617 },
		{ 1634, 1618 },
		{ 1442, 23 },
		{ 2585, 2555 },
		{ 1430, 23 },
		{ 1443, 23 },
		{ 1635, 1619 },
		{ 2587, 2557 },
		{ 1461, 1435 },
		{ 2592, 2562 },
		{ 1636, 1620 },
		{ 2595, 2565 },
		{ 1642, 1633 },
		{ 2599, 2569 },
		{ 1643, 1634 },
		{ 2600, 2570 },
		{ 1460, 1435 },
		{ 2615, 2582 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1459, 1435 },
		{ 1467, 1440 },
		{ 1644, 1635 },
		{ 1645, 1636 },
		{ 1466, 1440 },
		{ 1470, 1445 },
		{ 1649, 1643 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1446, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1435, 23 },
		{ 1433, 23 },
		{ 1448, 23 },
		{ 1434, 23 },
		{ 1448, 23 },
		{ 1437, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1436, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1471, 1446 },
		{ 1431, 23 },
		{ 1444, 23 },
		{ 1651, 1645 },
		{ 1438, 23 },
		{ 1654, 1649 },
		{ 1448, 23 },
		{ 1449, 23 },
		{ 1432, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1439, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1447, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1450, 23 },
		{ 1448, 23 },
		{ 1445, 23 },
		{ 1448, 23 },
		{ 1440, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 1448, 23 },
		{ 2027, 40 },
		{ 2376, 2341 },
		{ 2376, 2341 },
		{ 1655, 1651 },
		{ 1852, 40 },
		{ 2619, 2586 },
		{ 2619, 2586 },
		{ 2401, 2366 },
		{ 2401, 2366 },
		{ 2402, 2367 },
		{ 2402, 2367 },
		{ 2420, 2386 },
		{ 2420, 2386 },
		{ 2423, 2389 },
		{ 2423, 2389 },
		{ 1657, 1654 },
		{ 1658, 1655 },
		{ 1659, 1657 },
		{ 1660, 1658 },
		{ 1455, 1659 },
		{ 1472, 1447 },
		{ 1473, 1449 },
		{ 1474, 1450 },
		{ 2027, 40 },
		{ 2376, 2341 },
		{ 1857, 40 },
		{ 1477, 1454 },
		{ 1478, 1457 },
		{ 2619, 2586 },
		{ 1479, 1458 },
		{ 2401, 2366 },
		{ 1480, 1459 },
		{ 2402, 2367 },
		{ 1481, 1460 },
		{ 2420, 2386 },
		{ 1482, 1461 },
		{ 2423, 2389 },
		{ 1483, 1462 },
		{ 2026, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1484, 1463 },
		{ 1867, 40 },
		{ 1485, 1464 },
		{ 1487, 1465 },
		{ 1488, 1466 },
		{ 0, 2586 },
		{ 1486, 1464 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1854, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1862, 40 },
		{ 1860, 40 },
		{ 1870, 40 },
		{ 1861, 40 },
		{ 1870, 40 },
		{ 1864, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1863, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1489, 1467 },
		{ 1858, 40 },
		{ 1492, 1470 },
		{ 1493, 1471 },
		{ 1870, 40 },
		{ 1494, 1472 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1859, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1855, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1856, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1869, 40 },
		{ 1870, 40 },
		{ 1868, 40 },
		{ 1870, 40 },
		{ 1865, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1870, 40 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1495, 1473 },
		{ 1496, 1474 },
		{ 1499, 1477 },
		{ 1500, 1478 },
		{ 1501, 1479 },
		{ 1502, 1480 },
		{ 1503, 1481 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1504, 1482 },
		{ 1505, 1483 },
		{ 1506, 1484 },
		{ 1507, 1485 },
		{ 1455, 1661 },
		{ 1508, 1486 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 1455, 1661 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 1509, 1487 },
		{ 1510, 1488 },
		{ 1511, 1489 },
		{ 1514, 1492 },
		{ 1515, 1493 },
		{ 1516, 1494 },
		{ 1517, 1495 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 1518, 1496 },
		{ 1520, 1499 },
		{ 1521, 1500 },
		{ 1522, 1501 },
		{ 2481, 2547 },
		{ 1523, 1502 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2481, 2547 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2416, 2381 },
		{ 156, 154 },
		{ 116, 101 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 108 },
		{ 125, 109 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 126, 111 },
		{ 127, 112 },
		{ 128, 113 },
		{ 129, 114 },
		{ 2309, 2609 },
		{ 131, 116 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 2309, 2609 },
		{ 1456, 1660 },
		{ 0, 1660 },
		{ 136, 122 },
		{ 137, 123 },
		{ 138, 124 },
		{ 139, 125 },
		{ 140, 126 },
		{ 141, 127 },
		{ 142, 129 },
		{ 143, 131 },
		{ 144, 136 },
		{ 145, 137 },
		{ 146, 138 },
		{ 2317, 2294 },
		{ 2809, 2809 },
		{ 2319, 2295 },
		{ 2322, 2296 },
		{ 2323, 2297 },
		{ 2332, 2299 },
		{ 2320, 2296 },
		{ 2327, 2298 },
		{ 2334, 2300 },
		{ 2321, 2296 },
		{ 1456, 1660 },
		{ 2326, 2298 },
		{ 2335, 2301 },
		{ 2338, 2303 },
		{ 2324, 2297 },
		{ 2336, 2302 },
		{ 2325, 2297 },
		{ 2331, 2299 },
		{ 2339, 2304 },
		{ 2340, 2305 },
		{ 2341, 2306 },
		{ 2309, 2309 },
		{ 2345, 2310 },
		{ 2333, 2311 },
		{ 2809, 2809 },
		{ 2318, 2312 },
		{ 2328, 2298 },
		{ 2329, 2298 },
		{ 2337, 2302 },
		{ 2330, 2313 },
		{ 2349, 2317 },
		{ 2346, 2311 },
		{ 147, 139 },
		{ 2351, 2319 },
		{ 2352, 2320 },
		{ 2353, 2321 },
		{ 2354, 2322 },
		{ 2355, 2323 },
		{ 2356, 2324 },
		{ 0, 1660 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2357, 2325 },
		{ 2360, 2327 },
		{ 2361, 2328 },
		{ 2362, 2329 },
		{ 2363, 2330 },
		{ 2364, 2331 },
		{ 2365, 2332 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 69, 7 },
		{ 2367, 2333 },
		{ 2368, 2334 },
		{ 2369, 2335 },
		{ 2809, 2809 },
		{ 1661, 1660 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2370, 2336 },
		{ 2371, 2337 },
		{ 2374, 2339 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 2358, 2326 },
		{ 2375, 2340 },
		{ 148, 140 },
		{ 2381, 2345 },
		{ 2366, 2346 },
		{ 2384, 2349 },
		{ 2359, 2326 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 2386, 2351 },
		{ 2387, 2352 },
		{ 2388, 2353 },
		{ 2389, 2354 },
		{ 1269, 7 },
		{ 2390, 2355 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 1269, 7 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 2391, 2356 },
		{ 2392, 2357 },
		{ 2393, 2358 },
		{ 2394, 2359 },
		{ 2395, 2360 },
		{ 2396, 2361 },
		{ 2397, 2362 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 2398, 2363 },
		{ 2399, 2364 },
		{ 2400, 2365 },
		{ 149, 142 },
		{ 0, 1932 },
		{ 2403, 2368 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 1932 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 2404, 2369 },
		{ 2405, 2370 },
		{ 2406, 2371 },
		{ 2407, 2372 },
		{ 2408, 2373 },
		{ 2409, 2374 },
		{ 2410, 2375 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 150, 143 },
		{ 2418, 2384 },
		{ 151, 144 },
		{ 2421, 2387 },
		{ 0, 2126 },
		{ 2422, 2388 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 0, 2126 },
		{ 2372, 2338 },
		{ 2424, 2390 },
		{ 2426, 2391 },
		{ 2427, 2392 },
		{ 2428, 2393 },
		{ 2425, 2390 },
		{ 2429, 2394 },
		{ 152, 146 },
		{ 2431, 2396 },
		{ 2373, 2338 },
		{ 2432, 2397 },
		{ 2434, 2399 },
		{ 2435, 2400 },
		{ 2439, 2403 },
		{ 2440, 2404 },
		{ 2441, 2405 },
		{ 2443, 2407 },
		{ 2444, 2408 },
		{ 2445, 2409 },
		{ 2446, 2410 },
		{ 2449, 2418 },
		{ 2452, 2421 },
		{ 153, 149 },
		{ 154, 150 },
		{ 2457, 2425 },
		{ 2458, 2426 },
		{ 2459, 2427 },
		{ 2460, 2428 },
		{ 2461, 2429 },
		{ 2464, 2432 },
		{ 2466, 2434 },
		{ 2467, 2435 },
		{ 2471, 2439 },
		{ 2472, 2440 },
		{ 2473, 2441 },
		{ 2475, 2443 },
		{ 2476, 2444 },
		{ 155, 152 },
		{ 2478, 2446 },
		{ 2486, 2452 },
		{ 2489, 2457 },
		{ 2490, 2458 },
		{ 2491, 2459 },
		{ 2493, 2460 },
		{ 91, 75 },
		{ 2497, 2464 },
		{ 2499, 2466 },
		{ 2492, 2460 },
		{ 2500, 2467 },
		{ 2504, 2471 },
		{ 2505, 2472 },
		{ 2506, 2473 },
		{ 157, 155 },
		{ 158, 157 },
		{ 2511, 2478 },
		{ 159, 158 },
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
		{ 2518, 2486 },
		{ 2523, 2490 },
		{ 2524, 2491 },
		{ 2525, 2492 },
		{ 2526, 2493 },
		{ 2530, 2497 },
		{ 2532, 2499 },
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
		{ 2533, 2500 },
		{ 2537, 2504 },
		{ 2538, 2505 },
		{ 2539, 2506 },
		{ 87, 159 },
		{ 2545, 2511 },
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
		{ 94, 77 },
		{ 2552, 2518 },
		{ 2555, 2523 },
		{ 2556, 2524 },
		{ 2557, 2525 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 2562, 2530 },
		{ 95, 78 },
		{ 2565, 2533 },
		{ 2569, 2537 },
		{ 2570, 2538 },
		{ 93, 76 },
		{ 96, 79 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 2577, 2545 },
		{ 97, 80 },
		{ 2582, 2552 },
		{ 98, 81 },
		{ 1269, 1269 },
		{ 2586, 2556 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 99, 82 },
		{ 101, 84 },
		{ 106, 91 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 107, 92 },
		{ 108, 93 },
		{ 109, 94 },
		{ 110, 95 },
		{ 111, 96 },
		{ 112, 97 },
		{ 113, 98 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 114, 99 },
		{ 1282, 1278 },
		{ 2981, 51 },
		{ 1282, 1278 },
		{ 0, 1519 },
		{ 0, 2982 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 1519 },
		{ 0, 2577 },
		{ 0, 2577 },
		{ 1284, 1281 },
		{ 134, 120 },
		{ 1284, 1281 },
		{ 134, 120 },
		{ 2695, 2667 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 118, 103 },
		{ 2378, 2343 },
		{ 118, 103 },
		{ 2378, 2343 },
		{ 2707, 2679 },
		{ 2864, 2849 },
		{ 0, 2577 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 1279, 1276 },
		{ 2412, 2377 },
		{ 1279, 1276 },
		{ 2412, 2377 },
		{ 3148, 3148 },
		{ 2867, 2852 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 88, 51 },
		{ 2232, 2229 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 2414, 2380 },
		{ 132, 117 },
		{ 2414, 2380 },
		{ 132, 117 },
		{ 1850, 1849 },
		{ 1334, 1333 },
		{ 2609, 2577 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 1808, 1807 },
		{ 2848, 2832 },
		{ 3213, 3208 },
		{ 2668, 2635 },
		{ 1805, 1804 },
		{ 1331, 1330 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3208, 3208 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 1715, 1714 },
		{ 3128, 3127 },
		{ 1760, 1759 },
		{ 3225, 3224 },
		{ 3014, 3013 },
		{ 2069, 2047 },
		{ 2743, 2716 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3067, 3066 },
		{ 1347, 1346 },
		{ 3218, 3215 },
		{ 2560, 2528 },
		{ 3108, 3107 },
		{ 2099, 2078 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3215, 3215 },
		{ 3214, 3209 },
		{ 3214, 3209 },
		{ 3214, 3209 },
		{ 3214, 3209 },
		{ 3214, 3209 },
		{ 3214, 3209 },
		{ 3214, 3209 },
		{ 3214, 3209 },
		{ 3214, 3209 },
		{ 3214, 3209 },
		{ 3207, 3203 },
		{ 3207, 3203 },
		{ 3207, 3203 },
		{ 3207, 3203 },
		{ 3207, 3203 },
		{ 3207, 3203 },
		{ 3207, 3203 },
		{ 3207, 3203 },
		{ 3207, 3203 },
		{ 3207, 3203 },
		{ 0, 2448 },
		{ 3131, 3130 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 0, 2448 },
		{ 2084, 2060 },
		{ 184, 175 },
		{ 195, 175 },
		{ 186, 175 },
		{ 1394, 1393 },
		{ 181, 175 },
		{ 191, 175 },
		{ 185, 175 },
		{ 3139, 3138 },
		{ 183, 175 },
		{ 1689, 1688 },
		{ 1394, 1393 },
		{ 0, 3209 },
		{ 193, 175 },
		{ 192, 175 },
		{ 182, 175 },
		{ 190, 175 },
		{ 1689, 1688 },
		{ 187, 175 },
		{ 189, 175 },
		{ 180, 175 },
		{ 2083, 2060 },
		{ 0, 3203 },
		{ 188, 175 },
		{ 194, 175 },
		{ 102, 85 },
		{ 1883, 1859 },
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
		{ 2128, 2110 },
		{ 2142, 2125 },
		{ 1882, 1859 },
		{ 1908, 1889 },
		{ 1930, 1911 },
		{ 1821, 1820 },
		{ 2928, 2927 },
		{ 2272, 2271 },
		{ 3197, 3192 },
		{ 2968, 2967 },
		{ 2973, 2972 },
		{ 103, 85 },
		{ 1275, 1272 },
		{ 2480, 2448 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 3207, 3207 },
		{ 3207, 3207 },
		{ 3207, 3207 },
		{ 3207, 3207 },
		{ 3207, 3207 },
		{ 3207, 3207 },
		{ 3207, 3207 },
		{ 3207, 3207 },
		{ 3207, 3207 },
		{ 3207, 3207 },
		{ 2669, 2636 },
		{ 1276, 1272 },
		{ 2277, 2276 },
		{ 2608, 2576 },
		{ 1763, 1762 },
		{ 1736, 1735 },
		{ 3228, 3227 },
		{ 3244, 3241 },
		{ 2342, 2307 },
		{ 103, 85 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 3250, 3247 },
		{ 2343, 2307 },
		{ 2763, 2737 },
		{ 1276, 1272 },
		{ 2767, 2741 },
		{ 3212, 3207 },
		{ 2777, 2752 },
		{ 2779, 2754 },
		{ 1739, 1738 },
		{ 2784, 2759 },
		{ 2797, 2776 },
		{ 1278, 1275 },
		{ 2799, 2778 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2343, 2307 },
		{ 2377, 2342 },
		{ 2815, 2794 },
		{ 2816, 2795 },
		{ 1911, 1892 },
		{ 3167, 65 },
		{ 2286, 2285 },
		{ 2827, 2806 },
		{ 69, 65 },
		{ 2581, 2551 },
		{ 1278, 1275 },
		{ 2380, 2344 },
		{ 2832, 2813 },
		{ 1277, 1277 },
		{ 1277, 1277 },
		{ 1277, 1277 },
		{ 1277, 1277 },
		{ 1277, 1277 },
		{ 1277, 1277 },
		{ 1277, 1277 },
		{ 1277, 1277 },
		{ 1277, 1277 },
		{ 1277, 1277 },
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
		{ 2377, 2342 },
		{ 1281, 1277 },
		{ 2402, 2402 },
		{ 2402, 2402 },
		{ 2842, 2825 },
		{ 1271, 9 },
		{ 2566, 2566 },
		{ 2566, 2566 },
		{ 1917, 1898 },
		{ 69, 9 },
		{ 2380, 2344 },
		{ 120, 104 },
		{ 2849, 2833 },
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
		{ 3182, 67 },
		{ 2852, 2836 },
		{ 2402, 2402 },
		{ 69, 67 },
		{ 2583, 2553 },
		{ 1271, 9 },
		{ 2566, 2566 },
		{ 2519, 2487 },
		{ 3166, 65 },
		{ 2870, 2855 },
		{ 1281, 1277 },
		{ 117, 102 },
		{ 3165, 65 },
		{ 2884, 2878 },
		{ 2886, 2880 },
		{ 2892, 2887 },
		{ 2898, 2897 },
		{ 1350, 1349 },
		{ 1937, 1918 },
		{ 1273, 9 },
		{ 120, 104 },
		{ 1272, 9 },
		{ 1272, 9 },
		{ 1272, 9 },
		{ 1272, 9 },
		{ 1272, 9 },
		{ 1272, 9 },
		{ 1272, 9 },
		{ 1272, 9 },
		{ 1272, 9 },
		{ 1272, 9 },
		{ 3214, 3214 },
		{ 3214, 3214 },
		{ 3214, 3214 },
		{ 3214, 3214 },
		{ 3214, 3214 },
		{ 3214, 3214 },
		{ 3214, 3214 },
		{ 3214, 3214 },
		{ 3214, 3214 },
		{ 3214, 3214 },
		{ 2931, 2930 },
		{ 2940, 2939 },
		{ 117, 102 },
		{ 3175, 3175 },
		{ 3175, 3175 },
		{ 3175, 3175 },
		{ 3175, 3175 },
		{ 3175, 3175 },
		{ 3175, 3175 },
		{ 3175, 3175 },
		{ 3175, 3175 },
		{ 3175, 3175 },
		{ 3175, 3175 },
		{ 478, 430 },
		{ 483, 430 },
		{ 480, 430 },
		{ 479, 430 },
		{ 482, 430 },
		{ 477, 430 },
		{ 2953, 2952 },
		{ 476, 430 },
		{ 3163, 65 },
		{ 3179, 67 },
		{ 3164, 65 },
		{ 481, 430 },
		{ 1389, 1388 },
		{ 484, 430 },
		{ 1964, 1948 },
		{ 3180, 67 },
		{ 2592, 2592 },
		{ 2592, 2592 },
		{ 1966, 1951 },
		{ 475, 430 },
		{ 2976, 2975 },
		{ 2437, 2402 },
		{ 3217, 3214 },
		{ 2508, 2508 },
		{ 2508, 2508 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 1968, 1953 },
		{ 3177, 67 },
		{ 3032, 3032 },
		{ 3032, 3032 },
		{ 2018, 2016 },
		{ 2438, 2402 },
		{ 3079, 3079 },
		{ 3079, 3079 },
		{ 3176, 3175 },
		{ 2596, 2566 },
		{ 2578, 2578 },
		{ 2578, 2578 },
		{ 2592, 2592 },
		{ 2798, 2798 },
		{ 2798, 2798 },
		{ 2423, 2423 },
		{ 2423, 2423 },
		{ 2600, 2600 },
		{ 2600, 2600 },
		{ 2508, 2508 },
		{ 1765, 1764 },
		{ 2567, 2567 },
		{ 1787, 1786 },
		{ 3181, 67 },
		{ 2534, 2501 },
		{ 3032, 3032 },
		{ 2453, 2453 },
		{ 2453, 2453 },
		{ 3004, 3003 },
		{ 3079, 3079 },
		{ 2615, 2615 },
		{ 2615, 2615 },
		{ 2535, 2502 },
		{ 2578, 2578 },
		{ 2637, 2637 },
		{ 2637, 2637 },
		{ 2798, 2798 },
		{ 3031, 3030 },
		{ 2423, 2423 },
		{ 1800, 1799 },
		{ 2600, 2600 },
		{ 2696, 2696 },
		{ 2696, 2696 },
		{ 2477, 2477 },
		{ 2477, 2477 },
		{ 2964, 2964 },
		{ 2964, 2964 },
		{ 2571, 2571 },
		{ 2571, 2571 },
		{ 2453, 2453 },
		{ 2279, 2279 },
		{ 2279, 2279 },
		{ 1335, 1334 },
		{ 2615, 2615 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 3061, 3060 },
		{ 2637, 2637 },
		{ 2531, 2531 },
		{ 2531, 2531 },
		{ 2494, 2494 },
		{ 2494, 2494 },
		{ 2430, 2430 },
		{ 2430, 2430 },
		{ 2696, 2696 },
		{ 2488, 2454 },
		{ 2477, 2477 },
		{ 3070, 3069 },
		{ 2964, 2964 },
		{ 3078, 3077 },
		{ 2571, 2571 },
		{ 2595, 2595 },
		{ 2595, 2595 },
		{ 2279, 2279 },
		{ 2599, 2599 },
		{ 2599, 2599 },
		{ 1684, 1683 },
		{ 2992, 2992 },
		{ 2564, 2564 },
		{ 2564, 2564 },
		{ 1809, 1808 },
		{ 2531, 2531 },
		{ 3102, 3101 },
		{ 2494, 2494 },
		{ 2622, 2589 },
		{ 2430, 2430 },
		{ 3111, 3110 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 3122, 3121 },
		{ 2140, 2123 },
		{ 2626, 2592 },
		{ 2625, 2592 },
		{ 2550, 2516 },
		{ 2595, 2595 },
		{ 2802, 2802 },
		{ 2802, 2802 },
		{ 2599, 2599 },
		{ 2542, 2508 },
		{ 2541, 2508 },
		{ 3133, 3132 },
		{ 2564, 2564 },
		{ 2482, 2482 },
		{ 2482, 2482 },
		{ 2610, 2578 },
		{ 2597, 2567 },
		{ 3001, 3001 },
		{ 3001, 3001 },
		{ 2141, 2124 },
		{ 3033, 3032 },
		{ 2433, 2433 },
		{ 3142, 3141 },
		{ 2611, 2578 },
		{ 3080, 3079 },
		{ 2554, 2520 },
		{ 2350, 2350 },
		{ 2350, 2350 },
		{ 1412, 1411 },
		{ 2802, 2802 },
		{ 2819, 2798 },
		{ 2603, 2571 },
		{ 2454, 2423 },
		{ 2993, 2992 },
		{ 2634, 2600 },
		{ 1824, 1823 },
		{ 2482, 2482 },
		{ 3046, 3046 },
		{ 3046, 3046 },
		{ 2656, 2623 },
		{ 3001, 3001 },
		{ 2159, 2146 },
		{ 2172, 2157 },
		{ 2487, 2453 },
		{ 2587, 2587 },
		{ 2587, 2587 },
		{ 2173, 2158 },
		{ 2649, 2615 },
		{ 2601, 2571 },
		{ 2350, 2350 },
		{ 3192, 3187 },
		{ 2670, 2637 },
		{ 2602, 2571 },
		{ 2677, 2646 },
		{ 3087, 3087 },
		{ 3087, 3087 },
		{ 2703, 2703 },
		{ 2703, 2703 },
		{ 2724, 2696 },
		{ 2510, 2477 },
		{ 3046, 3046 },
		{ 2965, 2964 },
		{ 2693, 2665 },
		{ 1423, 1422 },
		{ 2280, 2279 },
		{ 2993, 2992 },
		{ 2700, 2672 },
		{ 2587, 2587 },
		{ 2755, 2755 },
		{ 2755, 2755 },
		{ 2233, 2230 },
		{ 2711, 2683 },
		{ 2563, 2531 },
		{ 2248, 2247 },
		{ 2527, 2494 },
		{ 1301, 1300 },
		{ 2462, 2430 },
		{ 3087, 3087 },
		{ 1737, 1736 },
		{ 2703, 2703 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 3230, 3229 },
		{ 3232, 3232 },
		{ 2751, 2724 },
		{ 2629, 2595 },
		{ 2274, 2273 },
		{ 3257, 3255 },
		{ 2633, 2599 },
		{ 1343, 1342 },
		{ 1890, 1865 },
		{ 2755, 2755 },
		{ 2594, 2564 },
		{ 1889, 1865 },
		{ 2420, 2420 },
		{ 2420, 2420 },
		{ 2079, 2054 },
		{ 1290, 1289 },
		{ 2902, 2901 },
		{ 2078, 2054 },
		{ 2465, 2433 },
		{ 1418, 1417 },
		{ 2923, 2922 },
		{ 2730, 2730 },
		{ 1755, 1754 },
		{ 2635, 2601 },
		{ 3232, 3232 },
		{ 2935, 2934 },
		{ 2823, 2802 },
		{ 2640, 2606 },
		{ 1756, 1755 },
		{ 2648, 2613 },
		{ 1979, 1966 },
		{ 1998, 1988 },
		{ 2006, 1998 },
		{ 2514, 2482 },
		{ 1305, 1304 },
		{ 2420, 2420 },
		{ 1641, 1632 },
		{ 3002, 3001 },
		{ 1358, 1357 },
		{ 2987, 2985 },
		{ 2678, 2648 },
		{ 1781, 1780 },
		{ 1782, 1781 },
		{ 3008, 3007 },
		{ 1365, 1364 },
		{ 1791, 1790 },
		{ 2385, 2350 },
		{ 2100, 2079 },
		{ 2715, 2687 },
		{ 2119, 2098 },
		{ 2728, 2700 },
		{ 2121, 2100 },
		{ 2575, 2543 },
		{ 2124, 2103 },
		{ 1366, 1365 },
		{ 2579, 2549 },
		{ 1368, 1367 },
		{ 3047, 3046 },
		{ 1647, 1641 },
		{ 3121, 3120 },
		{ 1677, 1676 },
		{ 2584, 2554 },
		{ 1817, 1816 },
		{ 2785, 2760 },
		{ 2620, 2587 },
		{ 2786, 2761 },
		{ 2788, 2764 },
		{ 2789, 2767 },
		{ 1678, 1677 },
		{ 1382, 1381 },
		{ 1840, 1839 },
		{ 1841, 1840 },
		{ 2818, 2797 },
		{ 1846, 1845 },
		{ 3088, 3087 },
		{ 1383, 1382 },
		{ 2731, 2703 },
		{ 1705, 1704 },
		{ 2828, 2807 },
		{ 1706, 1705 },
		{ 1712, 1711 },
		{ 1713, 1712 },
		{ 1909, 1890 },
		{ 1311, 1310 },
		{ 3220, 3219 },
		{ 3221, 3220 },
		{ 2289, 2288 },
		{ 2859, 2843 },
		{ 2780, 2755 },
		{ 3235, 3232 },
		{ 1916, 1897 },
		{ 2616, 2583 },
		{ 1731, 1730 },
		{ 1928, 1909 },
		{ 3234, 3232 },
		{ 1732, 1731 },
		{ 3233, 3232 },
		{ 1920, 1901 },
		{ 2757, 2730 },
		{ 2710, 2682 },
		{ 2589, 2559 },
		{ 2450, 2450 },
		{ 2450, 2450 },
		{ 2951, 2950 },
		{ 2593, 2563 },
		{ 2962, 2961 },
		{ 2726, 2698 },
		{ 1807, 1806 },
		{ 1410, 1409 },
		{ 2732, 2704 },
		{ 1327, 1326 },
		{ 2451, 2420 },
		{ 1947, 1930 },
		{ 2540, 2507 },
		{ 1741, 1740 },
		{ 2991, 2989 },
		{ 2764, 2738 },
		{ 1691, 1690 },
		{ 2768, 2742 },
		{ 2605, 2573 },
		{ 1414, 1413 },
		{ 1299, 1298 },
		{ 2226, 2219 },
		{ 2229, 2224 },
		{ 2450, 2450 },
		{ 3059, 3058 },
		{ 1710, 1709 },
		{ 1420, 1419 },
		{ 2245, 2242 },
		{ 1767, 1766 },
		{ 2013, 2008 },
		{ 2804, 2783 },
		{ 3100, 3099 },
		{ 2808, 2787 },
		{ 2628, 2594 },
		{ 1333, 1332 },
		{ 2630, 2596 },
		{ 2631, 2597 },
		{ 1292, 1291 },
		{ 2276, 2275 },
		{ 1717, 1716 },
		{ 1901, 1882 },
		{ 2831, 2812 },
		{ 2282, 2281 },
		{ 2839, 2822 },
		{ 1360, 1359 },
		{ 2470, 2438 },
		{ 2288, 2287 },
		{ 2662, 2629 },
		{ 2854, 2838 },
		{ 2666, 2633 },
		{ 1793, 1792 },
		{ 3201, 3197 },
		{ 2090, 2069 },
		{ 1313, 1312 },
		{ 2871, 2856 },
		{ 2872, 2858 },
		{ 1912, 1893 },
		{ 2885, 2879 },
		{ 2111, 2090 },
		{ 2686, 2658 },
		{ 2893, 2891 },
		{ 2896, 2894 },
		{ 2483, 2450 },
		{ 2529, 2496 },
		{ 3232, 3231 },
		{ 1802, 1801 },
		{ 3240, 3237 },
		{ 1396, 1395 },
		{ 3248, 3245 },
		{ 2701, 2673 },
		{ 2925, 2924 },
		{ 3260, 3259 },
		{ 2401, 2401 },
		{ 2401, 2401 },
		{ 2882, 2882 },
		{ 2882, 2882 },
		{ 2509, 2509 },
		{ 2509, 2509 },
		{ 2653, 2620 },
		{ 1786, 1785 },
		{ 1847, 1846 },
		{ 2742, 2715 },
		{ 2697, 2669 },
		{ 2665, 2632 },
		{ 2840, 2823 },
		{ 2794, 2772 },
		{ 2795, 2773 },
		{ 3003, 3002 },
		{ 1422, 1421 },
		{ 2758, 2731 },
		{ 2801, 2780 },
		{ 1951, 1936 },
		{ 1408, 1407 },
		{ 2484, 2450 },
		{ 2646, 2611 },
		{ 2401, 2401 },
		{ 2673, 2640 },
		{ 2882, 2882 },
		{ 2528, 2495 },
		{ 2509, 2509 },
		{ 2778, 2753 },
		{ 2878, 2869 },
		{ 2110, 2089 },
		{ 1879, 1856 },
		{ 2205, 2187 },
		{ 2787, 2763 },
		{ 1775, 1774 },
		{ 2978, 2977 },
		{ 1671, 1670 },
		{ 2231, 2228 },
		{ 2507, 2474 },
		{ 2651, 2618 },
		{ 1425, 1424 },
		{ 1878, 1856 },
		{ 1785, 1784 },
		{ 2661, 2628 },
		{ 2239, 2236 },
		{ 2242, 2240 },
		{ 3006, 3005 },
		{ 1725, 1724 },
		{ 3013, 3012 },
		{ 2015, 2012 },
		{ 1413, 1412 },
		{ 2021, 2020 },
		{ 2826, 2805 },
		{ 1894, 1871 },
		{ 1532, 1512 },
		{ 3063, 3062 },
		{ 1352, 1351 },
		{ 1475, 1451 },
		{ 3072, 3071 },
		{ 2690, 2662 },
		{ 1699, 1698 },
		{ 2694, 2666 },
		{ 2085, 2062 },
		{ 2089, 2068 },
		{ 3104, 3103 },
		{ 1740, 1739 },
		{ 1913, 1894 },
		{ 3113, 3112 },
		{ 1915, 1896 },
		{ 2858, 2842 },
		{ 2104, 2083 },
		{ 2860, 2844 },
		{ 2106, 2085 },
		{ 3135, 3134 },
		{ 2712, 2684 },
		{ 2108, 2087 },
		{ 3144, 3143 },
		{ 1407, 1406 },
		{ 1749, 1748 },
		{ 1303, 1302 },
		{ 2879, 2870 },
		{ 1497, 1475 },
		{ 1711, 1710 },
		{ 2738, 2711 },
		{ 1326, 1325 },
		{ 2891, 2886 },
		{ 3198, 3193 },
		{ 2139, 2122 },
		{ 1946, 1929 },
		{ 2752, 2725 },
		{ 1826, 1825 },
		{ 2900, 2899 },
		{ 1834, 1833 },
		{ 1376, 1375 },
		{ 2154, 2138 },
		{ 2627, 2593 },
		{ 2436, 2401 },
		{ 1961, 1945 },
		{ 2887, 2882 },
		{ 1766, 1765 },
		{ 2543, 2509 },
		{ 2933, 2932 },
		{ 3231, 3230 },
		{ 2495, 2462 },
		{ 1714, 1713 },
		{ 3237, 3234 },
		{ 2559, 2527 },
		{ 2782, 2757 },
		{ 2955, 2954 },
		{ 2961, 2960 },
		{ 2204, 2186 },
		{ 3259, 3257 },
		{ 2561, 2529 },
		{ 2946, 2946 },
		{ 2946, 2946 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 2558, 2558 },
		{ 2558, 2558 },
		{ 2585, 2585 },
		{ 2585, 2585 },
		{ 3054, 3054 },
		{ 3054, 3054 },
		{ 2676, 2645 },
		{ 2206, 2188 },
		{ 2217, 2204 },
		{ 2218, 2205 },
		{ 3007, 3006 },
		{ 1304, 1303 },
		{ 2228, 2223 },
		{ 1910, 1891 },
		{ 3015, 3014 },
		{ 3024, 3023 },
		{ 2022, 2021 },
		{ 1812, 1811 },
		{ 3041, 3040 },
		{ 2946, 2946 },
		{ 3042, 3041 },
		{ 3095, 3095 },
		{ 3044, 3043 },
		{ 2558, 2558 },
		{ 2843, 2826 },
		{ 2585, 2585 },
		{ 1353, 1352 },
		{ 3054, 3054 },
		{ 3057, 3056 },
		{ 1375, 1374 },
		{ 2241, 2239 },
		{ 1822, 1821 },
		{ 3064, 3063 },
		{ 2468, 2436 },
		{ 3068, 3067 },
		{ 2469, 2437 },
		{ 1724, 1723 },
		{ 3073, 3072 },
		{ 1774, 1773 },
		{ 2087, 2065 },
		{ 3085, 3084 },
		{ 2088, 2067 },
		{ 1918, 1899 },
		{ 3098, 3097 },
		{ 1827, 1826 },
		{ 2874, 2860 },
		{ 1833, 1832 },
		{ 3105, 3104 },
		{ 2739, 2712 },
		{ 3109, 3108 },
		{ 2741, 2714 },
		{ 2278, 2277 },
		{ 3114, 3113 },
		{ 3120, 3119 },
		{ 1682, 1681 },
		{ 2101, 2080 },
		{ 1405, 1404 },
		{ 2756, 2729 },
		{ 1552, 1533 },
		{ 115, 100 },
		{ 3136, 3135 },
		{ 1948, 1931 },
		{ 3140, 3139 },
		{ 1698, 1697 },
		{ 2901, 2900 },
		{ 3145, 3144 },
		{ 1348, 1347 },
		{ 1953, 1938 },
		{ 2123, 2102 },
		{ 3160, 3156 },
		{ 2382, 2347 },
		{ 2636, 2602 },
		{ 2929, 2928 },
		{ 2638, 2604 },
		{ 3189, 3184 },
		{ 1338, 1337 },
		{ 3193, 3188 },
		{ 2934, 2933 },
		{ 1963, 1947 },
		{ 1476, 1453 },
		{ 3205, 3201 },
		{ 1748, 1747 },
		{ 2949, 2948 },
		{ 1426, 1425 },
		{ 1387, 1386 },
		{ 1987, 1975 },
		{ 2956, 2955 },
		{ 2654, 2621 },
		{ 2146, 2129 },
		{ 1896, 1874 },
		{ 2947, 2946 },
		{ 2000, 1991 },
		{ 3096, 3095 },
		{ 1900, 1881 },
		{ 2588, 2558 },
		{ 2974, 2973 },
		{ 2618, 2585 },
		{ 2580, 2550 },
		{ 3055, 3054 },
		{ 2012, 2007 },
		{ 2979, 2978 },
		{ 2811, 2790 },
		{ 2184, 2170 },
		{ 1512, 1490 },
		{ 2672, 2639 },
		{ 2988, 2986 },
		{ 1670, 1669 },
		{ 2674, 2641 },
		{ 3065, 3065 },
		{ 3065, 3065 },
		{ 3137, 3137 },
		{ 3137, 3137 },
		{ 2624, 2624 },
		{ 2624, 2624 },
		{ 2926, 2926 },
		{ 2926, 2926 },
		{ 2463, 2463 },
		{ 2463, 2463 },
		{ 2376, 2376 },
		{ 2376, 2376 },
		{ 2837, 2837 },
		{ 2837, 2837 },
		{ 2971, 2971 },
		{ 2971, 2971 },
		{ 3106, 3106 },
		{ 3106, 3106 },
		{ 1819, 1819 },
		{ 1819, 1819 },
		{ 1345, 1345 },
		{ 1345, 1345 },
		{ 1700, 1699 },
		{ 3065, 3065 },
		{ 1962, 1946 },
		{ 3137, 3137 },
		{ 2250, 2249 },
		{ 2624, 2624 },
		{ 2127, 2108 },
		{ 2926, 2926 },
		{ 2017, 2015 },
		{ 2463, 2463 },
		{ 2733, 2705 },
		{ 2376, 2376 },
		{ 1672, 1671 },
		{ 2837, 2837 },
		{ 1735, 1734 },
		{ 2971, 2971 },
		{ 1934, 1915 },
		{ 3106, 3106 },
		{ 1377, 1376 },
		{ 1819, 1819 },
		{ 1750, 1749 },
		{ 1345, 1345 },
		{ 1789, 1788 },
		{ 2284, 2283 },
		{ 1835, 1834 },
		{ 2591, 2561 },
		{ 2155, 2139 },
		{ 2234, 2231 },
		{ 3247, 3244 },
		{ 2517, 2485 },
		{ 1551, 1532 },
		{ 1726, 1725 },
		{ 3202, 3198 },
		{ 1776, 1775 },
		{ 3083, 3083 },
		{ 3083, 3083 },
		{ 1340, 1340 },
		{ 1340, 1340 },
		{ 1814, 1814 },
		{ 1814, 1814 },
		{ 3126, 3126 },
		{ 3126, 3126 },
		{ 3090, 3090 },
		{ 3090, 3090 },
		{ 1803, 1803 },
		{ 1803, 1803 },
		{ 3049, 3049 },
		{ 3049, 3049 },
		{ 2125, 2104 },
		{ 2941, 2941 },
		{ 2941, 2941 },
		{ 1329, 1329 },
		{ 1329, 1329 },
		{ 2126, 2106 },
		{ 1519, 1497 },
		{ 2238, 2235 },
		{ 1838, 1837 },
		{ 3083, 3083 },
		{ 1779, 1778 },
		{ 1340, 1340 },
		{ 2590, 2560 },
		{ 1814, 1814 },
		{ 2708, 2680 },
		{ 3126, 3126 },
		{ 1729, 1728 },
		{ 3090, 3090 },
		{ 1753, 1752 },
		{ 1803, 1803 },
		{ 2792, 2770 },
		{ 3049, 3049 },
		{ 1297, 1296 },
		{ 3066, 3065 },
		{ 2941, 2941 },
		{ 3138, 3137 },
		{ 1329, 1329 },
		{ 2657, 2624 },
		{ 2877, 2868 },
		{ 2927, 2926 },
		{ 2713, 2685 },
		{ 2496, 2463 },
		{ 1848, 1847 },
		{ 2411, 2376 },
		{ 2145, 2128 },
		{ 2853, 2837 },
		{ 2725, 2697 },
		{ 2972, 2971 },
		{ 1816, 1815 },
		{ 3107, 3106 },
		{ 2273, 2272 },
		{ 1820, 1819 },
		{ 2479, 2447 },
		{ 1346, 1345 },
		{ 2814, 2793 },
		{ 1978, 1965 },
		{ 2999, 2998 },
		{ 3216, 3211 },
		{ 1687, 1686 },
		{ 2817, 2796 },
		{ 2098, 2077 },
		{ 2607, 2575 },
		{ 1363, 1362 },
		{ 3227, 3226 },
		{ 1990, 1980 },
		{ 1921, 1902 },
		{ 2203, 2185 },
		{ 1762, 1761 },
		{ 1380, 1379 },
		{ 1932, 1913 },
		{ 3130, 3129 },
		{ 1616, 1590 },
		{ 1675, 1674 },
		{ 2841, 2824 },
		{ 1897, 1877 },
		{ 3253, 3250 },
		{ 1392, 1391 },
		{ 1703, 1702 },
		{ 1342, 1341 },
		{ 1293, 1293 },
		{ 1293, 1293 },
		{ 2719, 2719 },
		{ 2719, 2719 },
		{ 2720, 2720 },
		{ 2720, 2720 },
		{ 2568, 2568 },
		{ 2568, 2568 },
		{ 2572, 2540 },
		{ 3084, 3083 },
		{ 2995, 2994 },
		{ 1341, 1340 },
		{ 3022, 3021 },
		{ 1815, 1814 },
		{ 1871, 2027 },
		{ 3127, 3126 },
		{ 2062, 2253 },
		{ 3091, 3090 },
		{ 1451, 1442 },
		{ 1804, 1803 },
		{ 2447, 2411 },
		{ 3050, 3049 },
		{ 2485, 2451 },
		{ 1293, 1293 },
		{ 2942, 2941 },
		{ 2719, 2719 },
		{ 1330, 1329 },
		{ 2720, 2720 },
		{ 2020, 2018 },
		{ 2568, 2568 },
		{ 2379, 2379 },
		{ 2379, 2379 },
		{ 2379, 2379 },
		{ 2379, 2379 },
		{ 2379, 2379 },
		{ 2379, 2379 },
		{ 2379, 2379 },
		{ 2379, 2379 },
		{ 2379, 2379 },
		{ 2379, 2379 },
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
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 2954, 2953 },
		{ 2706, 2706 },
		{ 2706, 2706 },
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
		{ 0, 1273 },
		{ 0, 86 },
		{ 1361, 1361 },
		{ 1361, 1361 },
		{ 3017, 3017 },
		{ 3017, 3017 },
		{ 3252, 3252 },
		{ 3103, 3102 },
		{ 1294, 1293 },
		{ 2729, 2701 },
		{ 2746, 2719 },
		{ 2706, 2706 },
		{ 2747, 2720 },
		{ 1688, 1687 },
		{ 2598, 2568 },
		{ 3151, 3151 },
		{ 3151, 3151 },
		{ 3151, 3151 },
		{ 3151, 3151 },
		{ 3151, 3151 },
		{ 3151, 3151 },
		{ 3151, 3151 },
		{ 3151, 3151 },
		{ 3151, 3151 },
		{ 3151, 3151 },
		{ 1361, 1361 },
		{ 2236, 2233 },
		{ 3017, 3017 },
		{ 1788, 1787 },
		{ 3252, 3252 },
		{ 1424, 1423 },
		{ 1898, 1878 },
		{ 0, 1273 },
		{ 0, 86 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2415, 2415 },
		{ 2415, 2415 },
		{ 2415, 2415 },
		{ 2415, 2415 },
		{ 2415, 2415 },
		{ 2415, 2415 },
		{ 2415, 2415 },
		{ 2415, 2415 },
		{ 2415, 2415 },
		{ 2415, 2415 },
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
		{ 0, 2308 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 3036, 3036 },
		{ 3036, 3036 },
		{ 3112, 3111 },
		{ 2734, 2706 },
		{ 2957, 2957 },
		{ 2957, 2957 },
		{ 2688, 2688 },
		{ 2688, 2688 },
		{ 2833, 2814 },
		{ 2836, 2817 },
		{ 1825, 1824 },
		{ 2066, 2044 },
		{ 2247, 2245 },
		{ 2977, 2976 },
		{ 1952, 1937 },
		{ 1790, 1789 },
		{ 2844, 2827 },
		{ 1362, 1361 },
		{ 1902, 1883 },
		{ 3018, 3017 },
		{ 3134, 3133 },
		{ 0, 2308 },
		{ 1351, 1350 },
		{ 3036, 3036 },
		{ 3254, 3252 },
		{ 2667, 2634 },
		{ 2753, 2726 },
		{ 2957, 2957 },
		{ 2551, 2517 },
		{ 2688, 2688 },
		{ 2501, 2468 },
		{ 2502, 2469 },
		{ 3143, 3142 },
		{ 2998, 2997 },
		{ 2760, 2733 },
		{ 1393, 1392 },
		{ 2604, 2572 },
		{ 2868, 2853 },
		{ 3005, 3004 },
		{ 2869, 2854 },
		{ 1386, 1385 },
		{ 2156, 2140 },
		{ 2770, 2744 },
		{ 1452, 1431 },
		{ 2170, 2154 },
		{ 2679, 2649 },
		{ 2683, 2654 },
		{ 3023, 3022 },
		{ 2684, 2656 },
		{ 2685, 2657 },
		{ 1302, 1301 },
		{ 1975, 1961 },
		{ 3040, 3039 },
		{ 1404, 1403 },
		{ 2186, 2172 },
		{ 3043, 3042 },
		{ 2187, 2173 },
		{ 2790, 2768 },
		{ 1388, 1387 },
		{ 2899, 2898 },
		{ 2793, 2771 },
		{ 2516, 2484 },
		{ 1845, 1844 },
		{ 3062, 3061 },
		{ 2796, 2775 },
		{ 2705, 2677 },
		{ 1421, 1420 },
		{ 1681, 1680 },
		{ 1811, 1810 },
		{ 2805, 2784 },
		{ 3071, 3070 },
		{ 3240, 3240 },
		{ 1337, 1336 },
		{ 2932, 2931 },
		{ 1683, 1682 },
		{ 2813, 2792 },
		{ 1533, 1513 },
		{ 1880, 1858 },
		{ 2067, 2044 },
		{ 1685, 1684 },
		{ 1390, 1389 },
		{ 1810, 1809 },
		{ 3081, 3080 },
		{ 1336, 1335 },
		{ 2834, 2815 },
		{ 2835, 2816 },
		{ 3019, 3018 },
		{ 3094, 3093 },
		{ 2721, 2693 },
		{ 3037, 3036 },
		{ 2249, 2248 },
		{ 3252, 3249 },
		{ 3053, 3052 },
		{ 2958, 2957 },
		{ 3240, 3240 },
		{ 2716, 2688 },
		{ 69, 5 },
		{ 2800, 2779 },
		{ 2945, 2944 },
		{ 3034, 3033 },
		{ 3039, 3038 },
		{ 1403, 1402 },
		{ 2660, 2627 },
		{ 1813, 1812 },
		{ 2240, 2238 },
		{ 3025, 3024 },
		{ 3045, 3044 },
		{ 1339, 1338 },
		{ 2830, 2811 },
		{ 2671, 2638 },
		{ 1453, 1431 },
		{ 2781, 2756 },
		{ 2985, 2983 },
		{ 3183, 3177 },
		{ 1891, 1868 },
		{ 1873, 1853 },
		{ 2255, 2252 },
		{ 2606, 2574 },
		{ 1892, 1868 },
		{ 1513, 1491 },
		{ 1872, 1853 },
		{ 2254, 2252 },
		{ 3048, 3047 },
		{ 2419, 2385 },
		{ 3089, 3088 },
		{ 2548, 2514 },
		{ 2659, 2626 },
		{ 2270, 2269 },
		{ 2986, 2983 },
		{ 2791, 2769 },
		{ 2064, 2041 },
		{ 1325, 1324 },
		{ 2960, 2959 },
		{ 2769, 2743 },
		{ 2997, 2996 },
		{ 2063, 2041 },
		{ 2498, 2465 },
		{ 1799, 1798 },
		{ 2687, 2659 },
		{ 2574, 2542 },
		{ 1469, 1443 },
		{ 2029, 2026 },
		{ 205, 181 },
		{ 3184, 3177 },
		{ 1881, 1858 },
		{ 203, 181 },
		{ 2028, 2026 },
		{ 204, 181 },
		{ 176, 5 },
		{ 1572, 1551 },
		{ 1697, 1696 },
		{ 1874, 1854 },
		{ 177, 5 },
		{ 2735, 2707 },
		{ 206, 181 },
		{ 2621, 2588 },
		{ 202, 181 },
		{ 1875, 1855 },
		{ 1876, 1855 },
		{ 178, 5 },
		{ 2285, 2284 },
		{ 2740, 2713 },
		{ 3056, 3055 },
		{ 3243, 3240 },
		{ 2536, 2503 },
		{ 3058, 3057 },
		{ 1573, 1552 },
		{ 2287, 2286 },
		{ 2883, 2877 },
		{ 2129, 2111 },
		{ 2137, 2119 },
		{ 1801, 1800 },
		{ 2632, 2598 },
		{ 1747, 1746 },
		{ 2544, 2510 },
		{ 175, 5 },
		{ 1344, 1343 },
		{ 2894, 2892 },
		{ 1359, 1358 },
		{ 1806, 1805 },
		{ 2639, 2605 },
		{ 2144, 2127 },
		{ 3082, 3081 },
		{ 1991, 1981 },
		{ 2645, 2610 },
		{ 1701, 1700 },
		{ 1751, 1750 },
		{ 2924, 2923 },
		{ 2772, 2746 },
		{ 2773, 2747 },
		{ 3097, 3096 },
		{ 1395, 1394 },
		{ 3099, 3098 },
		{ 2007, 1999 },
		{ 2008, 2000 },
		{ 1899, 1879 },
		{ 1378, 1377 },
		{ 1669, 1668 },
		{ 2783, 2758 },
		{ 1759, 1758 },
		{ 1402, 1401 },
		{ 2019, 2017 },
		{ 2188, 2174 },
		{ 2948, 2947 },
		{ 2663, 2630 },
		{ 1877, 1855 },
		{ 2950, 2949 },
		{ 2664, 2631 },
		{ 1328, 1327 },
		{ 3123, 3122 },
		{ 1295, 1294 },
		{ 2347, 2314 },
		{ 1673, 1672 },
		{ 1818, 1817 },
		{ 2959, 2958 },
		{ 1324, 1323 },
		{ 1332, 1331 },
		{ 2219, 2206 },
		{ 2963, 2962 },
		{ 2223, 2216 },
		{ 2966, 2965 },
		{ 1773, 1772 },
		{ 2970, 2969 },
		{ 2675, 2642 },
		{ 1716, 1715 },
		{ 2065, 2042 },
		{ 3156, 3147 },
		{ 2807, 2786 },
		{ 100, 83 },
		{ 1723, 1722 },
		{ 2812, 2791 },
		{ 2682, 2653 },
		{ 1777, 1776 },
		{ 2503, 2470 },
		{ 1927, 1908 },
		{ 1832, 1831 },
		{ 3187, 3181 },
		{ 3188, 3183 },
		{ 2237, 2234 },
		{ 2989, 2987 },
		{ 1490, 1468 },
		{ 2822, 2801 },
		{ 2691, 2663 },
		{ 2996, 2995 },
		{ 2824, 2803 },
		{ 2692, 2664 },
		{ 3206, 3202 },
		{ 1931, 1912 },
		{ 3000, 2999 },
		{ 1491, 1469 },
		{ 1409, 1408 },
		{ 1935, 1916 },
		{ 2512, 2479 },
		{ 1836, 1835 },
		{ 2704, 2676 },
		{ 1938, 1920 },
		{ 3224, 3223 },
		{ 1727, 1726 },
		{ 2102, 2081 },
		{ 2103, 2082 },
		{ 1367, 1366 },
		{ 1291, 1290 },
		{ 3020, 3019 },
		{ 1374, 1373 },
		{ 3236, 3233 },
		{ 2714, 2686 },
		{ 2269, 2268 },
		{ 1950, 1934 },
		{ 1312, 1311 },
		{ 3245, 3242 },
		{ 2723, 2695 },
		{ 1417, 1416 },
		{ 3035, 3034 },
		{ 2120, 2099 },
		{ 3038, 3037 },
		{ 2856, 2840 },
		{ 1690, 1689 },
		{ 2613, 2580 },
		{ 1792, 1791 },
		{ 1929, 1910 },
		{ 2967, 2966 },
		{ 3086, 3085 },
		{ 3194, 3189 },
		{ 2576, 2544 },
		{ 2990, 2988 },
		{ 2417, 2382 },
		{ 2122, 2101 },
		{ 3016, 3015 },
		{ 3242, 3239 },
		{ 1919, 1900 },
		{ 3125, 3124 },
		{ 1939, 1921 },
		{ 2702, 2674 },
		{ 1498, 1476 },
		{ 3162, 3160 },
		{ 2109, 2088 },
		{ 2081, 2058 },
		{ 1416, 1415 },
		{ 130, 115 },
		{ 2803, 2782 },
		{ 1844, 1843 },
		{ 3110, 3109 },
		{ 3141, 3140 },
		{ 2642, 2608 },
		{ 2975, 2974 },
		{ 2838, 2819 },
		{ 2944, 2943 },
		{ 1709, 1708 },
		{ 3238, 3235 },
		{ 3093, 3092 },
		{ 3241, 3238 },
		{ 1823, 1822 },
		{ 3021, 3020 },
		{ 1967, 1952 },
		{ 3069, 3068 },
		{ 1349, 1348 },
		{ 2930, 2929 },
		{ 3249, 3246 },
		{ 1419, 1418 },
		{ 3052, 3051 },
		{ 652, 591 },
		{ 2969, 2968 },
		{ 3256, 3254 },
		{ 1296, 1295 },
		{ 2171, 2156 },
		{ 2698, 2670 },
		{ 1451, 1444 },
		{ 2689, 2661 },
		{ 2062, 2056 },
		{ 1401, 1398 },
		{ 653, 591 },
		{ 1871, 1867 },
		{ 2718, 2690 },
		{ 2771, 2745 },
		{ 2722, 2694 },
		{ 2775, 2749 },
		{ 2744, 2717 },
		{ 1381, 1380 },
		{ 1764, 1763 },
		{ 2994, 2993 },
		{ 1300, 1299 },
		{ 1686, 1685 },
		{ 2271, 2270 },
		{ 1730, 1729 },
		{ 1676, 1675 },
		{ 1980, 1967 },
		{ 654, 591 },
		{ 2553, 2519 },
		{ 2680, 2651 },
		{ 2224, 2217 },
		{ 1364, 1363 },
		{ 1988, 1978 },
		{ 2943, 2942 },
		{ 1839, 1838 },
		{ 2737, 2710 },
		{ 2641, 2607 },
		{ 3092, 3091 },
		{ 2281, 2280 },
		{ 2855, 2839 },
		{ 2520, 2488 },
		{ 2230, 2226 },
		{ 2952, 2951 },
		{ 2283, 2282 },
		{ 1704, 1703 },
		{ 3101, 3100 },
		{ 1754, 1753 },
		{ 2745, 2718 },
		{ 1298, 1297 },
		{ 1411, 1410 },
		{ 2749, 2722 },
		{ 3219, 3216 },
		{ 2157, 2142 },
		{ 2806, 2785 },
		{ 2158, 2145 },
		{ 1893, 1869 },
		{ 3226, 3225 },
		{ 2754, 2727 },
		{ 2658, 2625 },
		{ 3229, 3228 },
		{ 2880, 2871 },
		{ 2077, 2053 },
		{ 2573, 2541 },
		{ 2759, 2732 },
		{ 1780, 1779 },
		{ 3124, 3123 },
		{ 2761, 2734 },
		{ 3239, 3236 },
		{ 1738, 1737 },
		{ 1632, 1616 },
		{ 3129, 3128 },
		{ 3051, 3050 },
		{ 2080, 2057 },
		{ 3132, 3131 },
		{ 3246, 3243 },
		{ 2185, 2171 },
		{ 1849, 1848 },
		{ 2897, 2896 },
		{ 2825, 2804 },
		{ 2016, 2013 },
		{ 1761, 1760 },
		{ 3060, 3059 },
		{ 3255, 3253 },
		{ 2623, 2590 },
		{ 1391, 1390 },
		{ 2717, 2689 },
		{ 2776, 2751 },
		{ 448, 403 },
		{ 715, 651 },
		{ 783, 724 },
		{ 898, 845 },
		{ 1400, 21 },
		{ 1695, 27 },
		{ 1721, 29 },
		{ 69, 21 },
		{ 69, 27 },
		{ 69, 29 },
		{ 3028, 57 },
		{ 1288, 11 },
		{ 1322, 15 },
		{ 69, 57 },
		{ 69, 11 },
		{ 69, 15 },
		{ 449, 403 },
		{ 1745, 31 },
		{ 1309, 13 },
		{ 782, 724 },
		{ 69, 31 },
		{ 69, 13 },
		{ 2921, 47 },
		{ 716, 651 },
		{ 1372, 19 },
		{ 69, 47 },
		{ 1222, 1211 },
		{ 69, 19 },
		{ 1771, 33 },
		{ 899, 845 },
		{ 1797, 35 },
		{ 69, 33 },
		{ 3118, 61 },
		{ 69, 35 },
		{ 1257, 1256 },
		{ 69, 61 },
		{ 272, 229 },
		{ 282, 238 },
		{ 290, 246 },
		{ 305, 258 },
		{ 314, 267 },
		{ 321, 273 },
		{ 331, 283 },
		{ 349, 300 },
		{ 2136, 2118 },
		{ 1923, 1904 },
		{ 1924, 1905 },
		{ 352, 303 },
		{ 361, 311 },
		{ 362, 312 },
		{ 367, 317 },
		{ 382, 335 },
		{ 408, 363 },
		{ 411, 366 },
		{ 2152, 2135 },
		{ 419, 374 },
		{ 430, 386 },
		{ 437, 393 },
		{ 446, 401 },
		{ 1944, 1926 },
		{ 236, 198 },
		{ 450, 404 },
		{ 463, 415 },
		{ 240, 202 },
		{ 485, 431 },
		{ 488, 435 },
		{ 498, 443 },
		{ 499, 444 },
		{ 1959, 1943 },
		{ 502, 447 },
		{ 514, 459 },
		{ 523, 470 },
		{ 556, 494 },
		{ 567, 503 },
		{ 570, 506 },
		{ 571, 507 },
		{ 575, 511 },
		{ 588, 526 },
		{ 592, 530 },
		{ 596, 534 },
		{ 606, 544 },
		{ 621, 557 },
		{ 622, 559 },
		{ 627, 564 },
		{ 644, 581 },
		{ 241, 203 },
		{ 1398, 21 },
		{ 1693, 27 },
		{ 1719, 29 },
		{ 661, 595 },
		{ 674, 608 },
		{ 677, 611 },
		{ 3027, 57 },
		{ 1286, 11 },
		{ 1320, 15 },
		{ 686, 620 },
		{ 703, 636 },
		{ 704, 637 },
		{ 714, 650 },
		{ 1743, 31 },
		{ 1307, 13 },
		{ 248, 210 },
		{ 734, 669 },
		{ 249, 211 },
		{ 2919, 47 },
		{ 784, 725 },
		{ 1370, 19 },
		{ 796, 736 },
		{ 798, 738 },
		{ 800, 740 },
		{ 1769, 33 },
		{ 805, 745 },
		{ 1795, 35 },
		{ 825, 766 },
		{ 3116, 61 },
		{ 836, 776 },
		{ 840, 780 },
		{ 841, 781 },
		{ 871, 815 },
		{ 890, 837 },
		{ 271, 228 },
		{ 928, 877 },
		{ 938, 887 },
		{ 953, 902 },
		{ 990, 943 },
		{ 994, 947 },
		{ 1010, 966 },
		{ 1026, 986 },
		{ 1051, 1015 },
		{ 1054, 1018 },
		{ 1062, 1027 },
		{ 1072, 1037 },
		{ 1090, 1057 },
		{ 1109, 1075 },
		{ 1116, 1084 },
		{ 2091, 2070 },
		{ 1118, 1086 },
		{ 1131, 1103 },
		{ 1144, 1117 },
		{ 1150, 1124 },
		{ 1157, 1132 },
		{ 1158, 1133 },
		{ 1172, 1153 },
		{ 1173, 1154 },
		{ 1191, 1174 },
		{ 1903, 1884 },
		{ 1196, 1180 },
		{ 1204, 1192 },
		{ 2113, 2092 },
		{ 2114, 2093 },
		{ 1216, 1205 },
		{ 2442, 2442 },
		{ 2442, 2442 },
		{ 69, 17 },
		{ 69, 25 },
		{ 69, 59 },
		{ 455, 408 },
		{ 69, 49 },
		{ 456, 408 },
		{ 454, 408 },
		{ 69, 37 },
		{ 69, 43 },
		{ 69, 53 },
		{ 69, 55 },
		{ 223, 189 },
		{ 2166, 2151 },
		{ 2164, 2150 },
		{ 3203, 3199 },
		{ 221, 189 },
		{ 3209, 3204 },
		{ 3175, 3174 },
		{ 2225, 2218 },
		{ 2167, 2151 },
		{ 2165, 2150 },
		{ 2442, 2442 },
		{ 457, 408 },
		{ 490, 437 },
		{ 492, 437 },
		{ 759, 701 },
		{ 458, 409 },
		{ 413, 368 },
		{ 851, 791 },
		{ 525, 472 },
		{ 224, 189 },
		{ 222, 189 },
		{ 613, 551 },
		{ 2162, 2148 },
		{ 493, 437 },
		{ 533, 479 },
		{ 534, 479 },
		{ 1971, 1957 },
		{ 429, 385 },
		{ 491, 437 },
		{ 758, 700 },
		{ 1147, 1120 },
		{ 216, 186 },
		{ 535, 479 },
		{ 1153, 1127 },
		{ 215, 186 },
		{ 713, 649 },
		{ 848, 788 },
		{ 300, 253 },
		{ 345, 296 },
		{ 546, 488 },
		{ 217, 186 },
		{ 2074, 2050 },
		{ 465, 417 },
		{ 2095, 2074 },
		{ 2699, 2699 },
		{ 2699, 2699 },
		{ 658, 592 },
		{ 945, 894 },
		{ 548, 488 },
		{ 2073, 2050 },
		{ 657, 592 },
		{ 375, 325 },
		{ 209, 183 },
		{ 547, 488 },
		{ 211, 183 },
		{ 944, 893 },
		{ 745, 684 },
		{ 210, 183 },
		{ 656, 592 },
		{ 655, 592 },
		{ 537, 481 },
		{ 2072, 2050 },
		{ 231, 193 },
		{ 440, 395 },
		{ 439, 395 },
		{ 2474, 2442 },
		{ 1162, 1137 },
		{ 2699, 2699 },
		{ 1355, 17 },
		{ 1666, 25 },
		{ 3075, 59 },
		{ 605, 543 },
		{ 2937, 49 },
		{ 310, 263 },
		{ 230, 193 },
		{ 1829, 37 },
		{ 2266, 43 },
		{ 2983, 53 },
		{ 3010, 55 },
		{ 540, 483 },
		{ 538, 481 },
		{ 831, 771 },
		{ 541, 483 },
		{ 227, 190 },
		{ 1161, 1137 },
		{ 311, 264 },
		{ 225, 190 },
		{ 940, 889 },
		{ 618, 556 },
		{ 226, 190 },
		{ 830, 771 },
		{ 265, 223 },
		{ 619, 556 },
		{ 276, 232 },
		{ 2096, 2075 },
		{ 852, 792 },
		{ 853, 793 },
		{ 1906, 1887 },
		{ 291, 247 },
		{ 553, 492 },
		{ 1289, 1286 },
		{ 1212, 1201 },
		{ 1213, 1202 },
		{ 620, 556 },
		{ 3173, 3171 },
		{ 1214, 1203 },
		{ 1094, 1061 },
		{ 832, 772 },
		{ 242, 204 },
		{ 3186, 3180 },
		{ 1117, 1085 },
		{ 780, 722 },
		{ 781, 723 },
		{ 292, 247 },
		{ 554, 492 },
		{ 2727, 2699 },
		{ 442, 397 },
		{ 1886, 1862 },
		{ 640, 576 },
		{ 3195, 3190 },
		{ 728, 663 },
		{ 1001, 956 },
		{ 3200, 3196 },
		{ 1005, 961 },
		{ 510, 455 },
		{ 857, 797 },
		{ 2549, 2515 },
		{ 1310, 1307 },
		{ 594, 532 },
		{ 746, 685 },
		{ 2922, 2919 },
		{ 1187, 1170 },
		{ 1316, 1315 },
		{ 748, 687 },
		{ 1194, 1177 },
		{ 908, 856 },
		{ 1357, 1355 },
		{ 572, 508 },
		{ 585, 523 },
		{ 212, 184 },
		{ 1205, 1193 },
		{ 1207, 1195 },
		{ 868, 811 },
		{ 869, 812 },
		{ 213, 184 },
		{ 699, 632 },
		{ 879, 825 },
		{ 1220, 1209 },
		{ 1221, 1210 },
		{ 887, 834 },
		{ 1226, 1217 },
		{ 1241, 1232 },
		{ 584, 523 },
		{ 422, 377 },
		{ 1268, 1267 },
		{ 579, 516 },
		{ 906, 854 },
		{ 580, 517 },
		{ 913, 861 },
		{ 923, 872 },
		{ 363, 313 },
		{ 586, 524 },
		{ 366, 316 },
		{ 731, 666 },
		{ 3174, 3173 },
		{ 287, 243 },
		{ 438, 394 },
		{ 954, 903 },
		{ 960, 909 },
		{ 966, 916 },
		{ 971, 921 },
		{ 976, 928 },
		{ 982, 936 },
		{ 3191, 3186 },
		{ 989, 942 },
		{ 199, 180 },
		{ 600, 538 },
		{ 749, 688 },
		{ 604, 542 },
		{ 201, 180 },
		{ 515, 460 },
		{ 3199, 3195 },
		{ 1011, 967 },
		{ 1023, 983 },
		{ 760, 702 },
		{ 1031, 991 },
		{ 200, 180 },
		{ 3204, 3200 },
		{ 1032, 992 },
		{ 1036, 996 },
		{ 1037, 997 },
		{ 1043, 1003 },
		{ 380, 332 },
		{ 611, 549 },
		{ 317, 270 },
		{ 384, 338 },
		{ 1074, 1039 },
		{ 1076, 1041 },
		{ 1080, 1045 },
		{ 1087, 1054 },
		{ 1089, 1056 },
		{ 386, 340 },
		{ 403, 359 },
		{ 1102, 1068 },
		{ 544, 486 },
		{ 355, 305 },
		{ 642, 578 },
		{ 827, 768 },
		{ 1120, 1088 },
		{ 1128, 1097 },
		{ 828, 769 },
		{ 1143, 1116 },
		{ 550, 490 },
		{ 1145, 1118 },
		{ 354, 305 },
		{ 353, 305 },
		{ 647, 584 },
		{ 1905, 1886 },
		{ 835, 775 },
		{ 651, 590 },
		{ 837, 777 },
		{ 461, 412 },
		{ 356, 306 },
		{ 1166, 1144 },
		{ 1171, 1151 },
		{ 846, 786 },
		{ 2093, 2072 },
		{ 847, 787 },
		{ 1180, 1162 },
		{ 1181, 1164 },
		{ 307, 260 },
		{ 415, 370 },
		{ 1192, 1175 },
		{ 418, 373 },
		{ 273, 230 },
		{ 697, 629 },
		{ 262, 221 },
		{ 220, 188 },
		{ 937, 886 },
		{ 1098, 1064 },
		{ 219, 188 },
		{ 1637, 1637 },
		{ 263, 221 },
		{ 598, 536 },
		{ 412, 367 },
		{ 1114, 1081 },
		{ 1558, 1558 },
		{ 2131, 2113 },
		{ 684, 618 },
		{ 376, 328 },
		{ 565, 501 },
		{ 339, 290 },
		{ 1123, 1091 },
		{ 609, 547 },
		{ 251, 213 },
		{ 709, 642 },
		{ 383, 336 },
		{ 981, 935 },
		{ 1662, 1662 },
		{ 615, 553 },
		{ 247, 209 },
		{ 460, 411 },
		{ 576, 512 },
		{ 733, 668 },
		{ 1637, 1637 },
		{ 426, 382 },
		{ 1007, 963 },
		{ 633, 570 },
		{ 634, 571 },
		{ 1558, 1558 },
		{ 855, 795 },
		{ 747, 686 },
		{ 527, 475 },
		{ 1185, 1168 },
		{ 583, 522 },
		{ 1941, 1923 },
		{ 643, 579 },
		{ 364, 314 },
		{ 391, 345 },
		{ 1046, 1007 },
		{ 1047, 1008 },
		{ 1662, 1662 },
		{ 1048, 1010 },
		{ 769, 713 },
		{ 1595, 1595 },
		{ 893, 840 },
		{ 1598, 1598 },
		{ 1601, 1601 },
		{ 1604, 1604 },
		{ 1607, 1607 },
		{ 1610, 1610 },
		{ 1057, 1021 },
		{ 894, 841 },
		{ 649, 588 },
		{ 1073, 1038 },
		{ 433, 389 },
		{ 318, 271 },
		{ 1077, 1042 },
		{ 234, 196 },
		{ 1628, 1628 },
		{ 1251, 1245 },
		{ 235, 197 },
		{ 668, 602 },
		{ 1468, 1637 },
		{ 935, 884 },
		{ 1115, 1082 },
		{ 778, 720 },
		{ 1595, 1595 },
		{ 1468, 1558 },
		{ 1598, 1598 },
		{ 1601, 1601 },
		{ 1604, 1604 },
		{ 1607, 1607 },
		{ 1610, 1610 },
		{ 545, 487 },
		{ 289, 245 },
		{ 404, 360 },
		{ 1958, 1942 },
		{ 672, 606 },
		{ 949, 898 },
		{ 1468, 1662 },
		{ 952, 901 },
		{ 1628, 1628 },
		{ 1135, 1107 },
		{ 3170, 3166 },
		{ 1138, 1110 },
		{ 1139, 1111 },
		{ 340, 291 },
		{ 676, 610 },
		{ 955, 904 },
		{ 958, 907 },
		{ 264, 222 },
		{ 1152, 1126 },
		{ 678, 612 },
		{ 2135, 2117 },
		{ 1985, 1973 },
		{ 1986, 1974 },
		{ 1154, 1128 },
		{ 968, 918 },
		{ 969, 919 },
		{ 970, 920 },
		{ 810, 750 },
		{ 816, 757 },
		{ 682, 616 },
		{ 558, 496 },
		{ 2149, 2132 },
		{ 1468, 1595 },
		{ 381, 333 },
		{ 1468, 1598 },
		{ 1468, 1601 },
		{ 1468, 1604 },
		{ 1468, 1607 },
		{ 1468, 1610 },
		{ 443, 398 },
		{ 614, 552 },
		{ 568, 504 },
		{ 503, 448 },
		{ 506, 451 },
		{ 1008, 964 },
		{ 509, 454 },
		{ 624, 561 },
		{ 1468, 1628 },
		{ 1014, 973 },
		{ 1020, 980 },
		{ 626, 563 },
		{ 1024, 984 },
		{ 717, 652 },
		{ 2182, 2168 },
		{ 2183, 2169 },
		{ 348, 299 },
		{ 850, 790 },
		{ 1033, 993 },
		{ 632, 569 },
		{ 313, 266 },
		{ 416, 371 },
		{ 740, 675 },
		{ 1254, 1252 },
		{ 744, 681 },
		{ 1258, 1257 },
		{ 858, 798 },
		{ 1049, 1012 },
		{ 865, 807 },
		{ 867, 809 },
		{ 521, 467 },
		{ 323, 275 },
		{ 1064, 1029 },
		{ 870, 813 },
		{ 261, 220 },
		{ 388, 342 },
		{ 423, 378 },
		{ 751, 690 },
		{ 752, 692 },
		{ 754, 695 },
		{ 374, 324 },
		{ 650, 589 },
		{ 396, 350 },
		{ 1943, 1925 },
		{ 764, 705 },
		{ 767, 710 },
		{ 768, 711 },
		{ 397, 352 },
		{ 560, 498 },
		{ 561, 498 },
		{ 562, 499 },
		{ 563, 499 },
		{ 254, 215 },
		{ 639, 575 },
		{ 208, 182 },
		{ 253, 215 },
		{ 638, 575 },
		{ 406, 361 },
		{ 405, 361 },
		{ 294, 248 },
		{ 293, 248 },
		{ 529, 477 },
		{ 694, 627 },
		{ 695, 627 },
		{ 207, 182 },
		{ 257, 218 },
		{ 2619, 2619 },
		{ 2619, 2619 },
		{ 258, 218 },
		{ 793, 734 },
		{ 319, 272 },
		{ 299, 252 },
		{ 530, 477 },
		{ 531, 478 },
		{ 269, 226 },
		{ 799, 739 },
		{ 3196, 3191 },
		{ 259, 218 },
		{ 320, 272 },
		{ 794, 734 },
		{ 268, 226 },
		{ 298, 252 },
		{ 967, 917 },
		{ 532, 478 },
		{ 371, 321 },
		{ 802, 742 },
		{ 2153, 2136 },
		{ 1973, 1959 },
		{ 1260, 1259 },
		{ 2619, 2619 },
		{ 244, 206 },
		{ 806, 746 },
		{ 608, 546 },
		{ 2092, 2071 },
		{ 487, 433 },
		{ 281, 237 },
		{ 407, 362 },
		{ 2168, 2152 },
		{ 2097, 2076 },
		{ 1164, 1142 },
		{ 1165, 1143 },
		{ 698, 630 },
		{ 1167, 1145 },
		{ 267, 225 },
		{ 753, 694 },
		{ 301, 254 },
		{ 284, 240 },
		{ 370, 320 },
		{ 1183, 1166 },
		{ 710, 644 },
		{ 930, 879 },
		{ 504, 449 },
		{ 1017, 976 },
		{ 623, 560 },
		{ 566, 502 },
		{ 1942, 1924 },
		{ 1201, 1189 },
		{ 1202, 1190 },
		{ 664, 598 },
		{ 721, 656 },
		{ 722, 657 },
		{ 1119, 1087 },
		{ 3171, 3169 },
		{ 723, 658 },
		{ 625, 562 },
		{ 854, 794 },
		{ 2132, 2114 },
		{ 505, 450 },
		{ 2068, 2045 },
		{ 1039, 999 },
		{ 2652, 2619 },
		{ 467, 419 },
		{ 1960, 1944 },
		{ 1904, 1885 },
		{ 3190, 3185 },
		{ 508, 453 },
		{ 542, 484 },
		{ 1907, 1888 },
		{ 2515, 2483 },
		{ 275, 231 },
		{ 1044, 1004 },
		{ 630, 567 },
		{ 255, 216 },
		{ 1148, 1121 },
		{ 839, 779 },
		{ 790, 731 },
		{ 274, 231 },
		{ 792, 733 },
		{ 1996, 1985 },
		{ 901, 847 },
		{ 1155, 1130 },
		{ 972, 922 },
		{ 1058, 1022 },
		{ 973, 923 },
		{ 569, 505 },
		{ 612, 550 },
		{ 911, 859 },
		{ 511, 456 },
		{ 916, 864 },
		{ 919, 868 },
		{ 999, 953 },
		{ 920, 869 },
		{ 453, 407 },
		{ 926, 875 },
		{ 1092, 1059 },
		{ 464, 416 },
		{ 616, 554 },
		{ 573, 509 },
		{ 1936, 1917 },
		{ 1105, 1071 },
		{ 1012, 971 },
		{ 1200, 1188 },
		{ 1110, 1076 },
		{ 1111, 1077 },
		{ 1112, 1078 },
		{ 520, 466 },
		{ 2071, 2049 },
		{ 809, 749 },
		{ 2075, 2051 },
		{ 939, 888 },
		{ 232, 194 },
		{ 602, 540 },
		{ 496, 440 },
		{ 1885, 1861 },
		{ 2163, 2149 },
		{ 826, 767 },
		{ 1887, 1863 },
		{ 739, 674 },
		{ 306, 259 },
		{ 742, 677 },
		{ 1227, 1218 },
		{ 1238, 1229 },
		{ 1239, 1230 },
		{ 582, 521 },
		{ 1244, 1235 },
		{ 1136, 1108 },
		{ 1252, 1247 },
		{ 1253, 1248 },
		{ 1972, 1958 },
		{ 2201, 2182 },
		{ 873, 819 },
		{ 1040, 1000 },
		{ 1141, 1113 },
		{ 270, 227 },
		{ 762, 704 },
		{ 924, 873 },
		{ 402, 358 },
		{ 986, 939 },
		{ 601, 539 },
		{ 685, 619 },
		{ 763, 704 },
		{ 1124, 1092 },
		{ 1198, 1186 },
		{ 1199, 1187 },
		{ 1125, 1094 },
		{ 648, 585 },
		{ 468, 420 },
		{ 1133, 1105 },
		{ 473, 428 },
		{ 1206, 1194 },
		{ 581, 520 },
		{ 1211, 1200 },
		{ 316, 269 },
		{ 875, 821 },
		{ 1140, 1112 },
		{ 445, 400 },
		{ 1075, 1040 },
		{ 946, 895 },
		{ 2147, 2130 },
		{ 885, 832 },
		{ 1956, 1940 },
		{ 1223, 1214 },
		{ 1079, 1044 },
		{ 660, 594 },
		{ 1236, 1227 },
		{ 486, 432 },
		{ 392, 346 },
		{ 845, 785 },
		{ 1091, 1058 },
		{ 1245, 1238 },
		{ 1246, 1239 },
		{ 1250, 1244 },
		{ 896, 843 },
		{ 959, 908 },
		{ 2094, 2073 },
		{ 549, 489 },
		{ 1160, 1136 },
		{ 1255, 1253 },
		{ 591, 529 },
		{ 1163, 1141 },
		{ 1103, 1069 },
		{ 3169, 3165 },
		{ 1267, 1266 },
		{ 1104, 1070 },
		{ 673, 607 },
		{ 302, 255 },
		{ 1168, 1148 },
		{ 528, 476 },
		{ 755, 696 },
		{ 414, 369 },
		{ 3185, 3179 },
		{ 1174, 1155 },
		{ 1179, 1161 },
		{ 252, 214 },
		{ 724, 659 },
		{ 727, 662 },
		{ 471, 424 },
		{ 1002, 957 },
		{ 239, 201 },
		{ 2133, 2115 },
		{ 2134, 2116 },
		{ 646, 583 },
		{ 617, 555 },
		{ 631, 568 },
		{ 427, 383 },
		{ 708, 641 },
		{ 922, 871 },
		{ 735, 670 },
		{ 801, 741 },
		{ 372, 322 },
		{ 877, 823 },
		{ 237, 199 },
		{ 932, 881 },
		{ 636, 573 },
		{ 974, 926 },
		{ 347, 298 },
		{ 436, 392 },
		{ 891, 838 },
		{ 1265, 1264 },
		{ 814, 754 },
		{ 459, 410 },
		{ 718, 653 },
		{ 691, 624 },
		{ 693, 626 },
		{ 1129, 1099 },
		{ 864, 806 },
		{ 1197, 1183 },
		{ 1063, 1028 },
		{ 833, 773 },
		{ 912, 860 },
		{ 736, 671 },
		{ 956, 905 },
		{ 914, 862 },
		{ 663, 597 },
		{ 296, 250 },
		{ 965, 915 },
		{ 628, 565 },
		{ 1146, 1119 },
		{ 1926, 1907 },
		{ 1083, 1050 },
		{ 766, 707 },
		{ 2118, 2097 },
		{ 803, 743 },
		{ 874, 820 },
		{ 1027, 987 },
		{ 1028, 988 },
		{ 1093, 1060 },
		{ 334, 286 },
		{ 1096, 1062 },
		{ 1228, 1219 },
		{ 1233, 1224 },
		{ 1159, 1134 },
		{ 1097, 1062 },
		{ 1237, 1228 },
		{ 1095, 1062 },
		{ 379, 331 },
		{ 564, 500 },
		{ 497, 442 },
		{ 1242, 1233 },
		{ 1243, 1234 },
		{ 933, 882 },
		{ 849, 789 },
		{ 977, 929 },
		{ 1249, 1243 },
		{ 978, 930 },
		{ 390, 344 },
		{ 635, 572 },
		{ 729, 664 },
		{ 238, 200 },
		{ 732, 667 },
		{ 1178, 1159 },
		{ 277, 233 },
		{ 1052, 1016 },
		{ 1261, 1260 },
		{ 1053, 1017 },
		{ 1182, 1165 },
		{ 998, 951 },
		{ 1184, 1167 },
		{ 1055, 1019 },
		{ 329, 281 },
		{ 365, 315 },
		{ 1060, 1025 },
		{ 599, 537 },
		{ 876, 822 },
		{ 308, 261 },
		{ 1035, 995 },
		{ 1456, 1456 },
		{ 765, 706 },
		{ 1314, 1314 },
		{ 1314, 1314 },
		{ 335, 287 },
		{ 552, 491 },
		{ 1022, 982 },
		{ 1085, 1051 },
		{ 336, 287 },
		{ 551, 491 },
		{ 771, 714 },
		{ 770, 714 },
		{ 1084, 1051 },
		{ 772, 714 },
		{ 786, 727 },
		{ 1209, 1198 },
		{ 789, 730 },
		{ 424, 379 },
		{ 791, 732 },
		{ 517, 463 },
		{ 519, 465 },
		{ 1218, 1207 },
		{ 1121, 1089 },
		{ 1456, 1456 },
		{ 795, 735 },
		{ 1314, 1314 },
		{ 288, 244 },
		{ 680, 614 },
		{ 322, 274 },
		{ 522, 468 },
		{ 1130, 1102 },
		{ 1229, 1220 },
		{ 1230, 1221 },
		{ 1038, 998 },
		{ 1235, 1226 },
		{ 428, 384 },
		{ 524, 471 },
		{ 1042, 1002 },
		{ 409, 364 },
		{ 804, 744 },
		{ 957, 906 },
		{ 260, 219 },
		{ 1142, 1115 },
		{ 641, 577 },
		{ 607, 545 },
		{ 1248, 1241 },
		{ 378, 330 },
		{ 1957, 1941 },
		{ 750, 689 },
		{ 886, 833 },
		{ 243, 205 },
		{ 888, 835 },
		{ 819, 760 },
		{ 820, 761 },
		{ 1059, 1023 },
		{ 700, 633 },
		{ 2115, 2094 },
		{ 2116, 2095 },
		{ 701, 634 },
		{ 501, 446 },
		{ 1315, 1314 },
		{ 344, 295 },
		{ 1468, 1456 },
		{ 1070, 1035 },
		{ 1071, 1036 },
		{ 900, 846 },
		{ 536, 480 },
		{ 368, 318 },
		{ 539, 482 },
		{ 466, 418 },
		{ 400, 355 },
		{ 1078, 1043 },
		{ 358, 308 },
		{ 991, 944 },
		{ 992, 945 },
		{ 659, 593 },
		{ 915, 863 },
		{ 1088, 1055 },
		{ 3119, 3116 },
		{ 297, 251 },
		{ 1000, 954 },
		{ 917, 865 },
		{ 472, 426 },
		{ 1004, 960 },
		{ 1186, 1169 },
		{ 843, 783 },
		{ 1188, 1171 },
		{ 662, 596 },
		{ 420, 375 },
		{ 2148, 2131 },
		{ 333, 285 },
		{ 2150, 2133 },
		{ 2151, 2134 },
		{ 590, 528 },
		{ 670, 604 },
		{ 929, 878 },
		{ 1108, 1074 },
		{ 1016, 975 },
		{ 512, 457 },
		{ 1019, 979 },
		{ 385, 339 },
		{ 1113, 1080 },
		{ 1100, 1066 },
		{ 1101, 1067 },
		{ 756, 697 },
		{ 897, 844 },
		{ 425, 380 },
		{ 1772, 1769 },
		{ 1981, 1968 },
		{ 2216, 2203 },
		{ 346, 297 },
		{ 1132, 1104 },
		{ 447, 402 },
		{ 2112, 2091 },
		{ 1373, 1370 },
		{ 738, 673 },
		{ 726, 661 },
		{ 947, 896 },
		{ 1696, 1693 },
		{ 1999, 1990 },
		{ 593, 531 },
		{ 2174, 2159 },
		{ 1189, 1172 },
		{ 1831, 1829 },
		{ 1190, 1173 },
		{ 1034, 994 },
		{ 1922, 1903 },
		{ 1722, 1719 },
		{ 842, 782 },
		{ 1746, 1743 },
		{ 526, 473 },
		{ 1668, 1666 },
		{ 983, 937 },
		{ 719, 654 },
		{ 1176, 1157 },
		{ 1177, 1158 },
		{ 984, 937 },
		{ 866, 808 },
		{ 444, 399 },
		{ 1247, 1240 },
		{ 295, 249 },
		{ 808, 748 },
		{ 399, 354 },
		{ 950, 899 },
		{ 1156, 1131 },
		{ 774, 716 },
		{ 975, 927 },
		{ 507, 452 },
		{ 343, 294 },
		{ 1137, 1109 },
		{ 761, 703 },
		{ 246, 208 },
		{ 1061, 1026 },
		{ 856, 796 },
		{ 878, 824 },
		{ 389, 343 },
		{ 1169, 1149 },
		{ 1170, 1150 },
		{ 452, 406 },
		{ 863, 805 },
		{ 730, 665 },
		{ 1203, 1191 },
		{ 683, 617 },
		{ 1029, 989 },
		{ 432, 388 },
		{ 942, 891 },
		{ 943, 892 },
		{ 410, 365 },
		{ 434, 390 },
		{ 690, 623 },
		{ 1974, 1960 },
		{ 435, 391 },
		{ 948, 897 },
		{ 2130, 2112 },
		{ 692, 625 },
		{ 369, 319 },
		{ 351, 302 },
		{ 1983, 1971 },
		{ 513, 458 },
		{ 559, 497 },
		{ 470, 422 },
		{ 813, 753 },
		{ 279, 235 },
		{ 280, 236 },
		{ 1888, 1864 },
		{ 1050, 1013 },
		{ 817, 758 },
		{ 1240, 1231 },
		{ 881, 828 },
		{ 964, 913 },
		{ 818, 759 },
		{ 441, 396 },
		{ 1149, 1122 },
		{ 474, 429 },
		{ 821, 762 },
		{ 889, 836 },
		{ 824, 765 },
		{ 757, 698 },
		{ 309, 262 },
		{ 393, 347 },
		{ 895, 842 },
		{ 1065, 1030 },
		{ 1069, 1034 },
		{ 1256, 1254 },
		{ 357, 307 },
		{ 829, 770 },
		{ 1259, 1258 },
		{ 2169, 2153 },
		{ 245, 207 },
		{ 360, 310 },
		{ 1264, 1263 },
		{ 489, 436 },
		{ 1266, 1265 },
		{ 2176, 2162 },
		{ 2178, 2164 },
		{ 2179, 2165 },
		{ 2180, 2166 },
		{ 2181, 2167 },
		{ 902, 849 },
		{ 904, 852 },
		{ 421, 376 },
		{ 907, 855 },
		{ 2076, 2052 },
		{ 494, 438 },
		{ 910, 858 },
		{ 1081, 1046 },
		{ 1925, 1906 },
		{ 341, 292 },
		{ 993, 946 },
		{ 1086, 1052 },
		{ 401, 356 },
		{ 196, 176 },
		{ 665, 599 },
		{ 312, 265 },
		{ 776, 718 },
		{ 324, 276 },
		{ 918, 866 },
		{ 326, 278 },
		{ 1006, 962 },
		{ 283, 239 },
		{ 1940, 1922 },
		{ 218, 187 },
		{ 1009, 965 },
		{ 675, 609 },
		{ 785, 726 },
		{ 925, 874 },
		{ 587, 525 },
		{ 1107, 1073 },
		{ 1015, 974 },
		{ 927, 876 },
		{ 543, 485 },
		{ 1018, 977 },
		{ 589, 527 },
		{ 679, 613 },
		{ 1021, 981 },
		{ 214, 185 },
		{ 681, 615 },
		{ 737, 672 },
		{ 462, 413 },
		{ 2117, 2096 },
		{ 861, 801 },
		{ 1608, 1608 },
		{ 1608, 1608 },
		{ 1599, 1599 },
		{ 1599, 1599 },
		{ 1611, 1611 },
		{ 1611, 1611 },
		{ 1559, 1559 },
		{ 1559, 1559 },
		{ 1638, 1638 },
		{ 1638, 1638 },
		{ 1602, 1602 },
		{ 1602, 1602 },
		{ 2023, 2023 },
		{ 2023, 2023 },
		{ 1629, 1629 },
		{ 1629, 1629 },
		{ 1596, 1596 },
		{ 1596, 1596 },
		{ 2243, 2243 },
		{ 2243, 2243 },
		{ 1992, 1992 },
		{ 1992, 1992 },
		{ 892, 839 },
		{ 1608, 1608 },
		{ 1013, 972 },
		{ 1599, 1599 },
		{ 256, 217 },
		{ 1611, 1611 },
		{ 961, 910 },
		{ 1559, 1559 },
		{ 962, 911 },
		{ 1638, 1638 },
		{ 1122, 1090 },
		{ 1602, 1602 },
		{ 2268, 2266 },
		{ 2023, 2023 },
		{ 337, 288 },
		{ 1629, 1629 },
		{ 330, 282 },
		{ 1596, 1596 },
		{ 595, 533 },
		{ 2243, 2243 },
		{ 1126, 1095 },
		{ 1992, 1992 },
		{ 1994, 1994 },
		{ 1994, 1994 },
		{ 1605, 1605 },
		{ 1605, 1605 },
		{ 2189, 2189 },
		{ 2189, 2189 },
		{ 2191, 2191 },
		{ 2191, 2191 },
		{ 1609, 1608 },
		{ 1127, 1096 },
		{ 1600, 1599 },
		{ 431, 387 },
		{ 1612, 1611 },
		{ 815, 755 },
		{ 1560, 1559 },
		{ 574, 510 },
		{ 1639, 1638 },
		{ 797, 737 },
		{ 1603, 1602 },
		{ 304, 257 },
		{ 2024, 2023 },
		{ 2681, 2652 },
		{ 1630, 1629 },
		{ 1994, 1994 },
		{ 1597, 1596 },
		{ 1605, 1605 },
		{ 2244, 2243 },
		{ 2189, 2189 },
		{ 1993, 1992 },
		{ 2191, 2191 },
		{ 2193, 2193 },
		{ 2193, 2193 },
		{ 2195, 2195 },
		{ 2195, 2195 },
		{ 2197, 2197 },
		{ 2197, 2197 },
		{ 2199, 2199 },
		{ 2199, 2199 },
		{ 1969, 1969 },
		{ 1969, 1969 },
		{ 687, 621 },
		{ 688, 621 },
		{ 2004, 2004 },
		{ 2004, 2004 },
		{ 2214, 2214 },
		{ 2214, 2214 },
		{ 1663, 1663 },
		{ 1663, 1663 },
		{ 2009, 2009 },
		{ 2009, 2009 },
		{ 2160, 2160 },
		{ 2160, 2160 },
		{ 1995, 1994 },
		{ 2193, 2193 },
		{ 1606, 1605 },
		{ 2195, 2195 },
		{ 2190, 2189 },
		{ 2197, 2197 },
		{ 2192, 2191 },
		{ 2199, 2199 },
		{ 725, 660 },
		{ 1969, 1969 },
		{ 2220, 2220 },
		{ 2220, 2220 },
		{ 883, 830 },
		{ 2004, 2004 },
		{ 1317, 1316 },
		{ 2214, 2214 },
		{ 689, 622 },
		{ 1663, 1663 },
		{ 325, 277 },
		{ 2009, 2009 },
		{ 1030, 990 },
		{ 2160, 2160 },
		{ 577, 514 },
		{ 712, 648 },
		{ 495, 439 },
		{ 603, 541 },
		{ 285, 241 },
		{ 1099, 1065 },
		{ 1208, 1196 },
		{ 1379, 1378 },
		{ 2194, 2193 },
		{ 1151, 1125 },
		{ 2196, 2195 },
		{ 2220, 2220 },
		{ 2198, 2197 },
		{ 1210, 1199 },
		{ 2200, 2199 },
		{ 844, 784 },
		{ 1970, 1969 },
		{ 1965, 1950 },
		{ 936, 885 },
		{ 3211, 3206 },
		{ 2005, 2004 },
		{ 779, 721 },
		{ 2215, 2214 },
		{ 395, 349 },
		{ 1664, 1663 },
		{ 1215, 1204 },
		{ 2010, 2009 },
		{ 909, 857 },
		{ 2161, 2160 },
		{ 1217, 1206 },
		{ 228, 191 },
		{ 1106, 1072 },
		{ 941, 890 },
		{ 995, 948 },
		{ 997, 950 },
		{ 1752, 1751 },
		{ 1263, 1262 },
		{ 1702, 1701 },
		{ 1225, 1216 },
		{ 1082, 1048 },
		{ 2221, 2220 },
		{ 882, 829 },
		{ 359, 309 },
		{ 1193, 1176 },
		{ 777, 719 },
		{ 3172, 3170 },
		{ 1232, 1223 },
		{ 1195, 1179 },
		{ 859, 799 },
		{ 860, 800 },
		{ 1997, 1986 },
		{ 1003, 959 },
		{ 198, 178 },
		{ 1066, 1031 },
		{ 2138, 2121 },
		{ 1945, 1928 },
		{ 1067, 1032 },
		{ 1045, 1005 },
		{ 807, 747 },
		{ 1674, 1673 },
		{ 2202, 2183 },
		{ 1175, 1156 },
		{ 1778, 1777 },
		{ 1590, 1572 },
		{ 834, 774 },
		{ 1728, 1727 },
		{ 985, 938 },
		{ 1837, 1836 },
		{ 1984, 1972 },
		{ 2177, 2163 },
		{ 315, 268 },
		{ 2213, 2201 },
		{ 518, 464 },
		{ 229, 192 },
		{ 2082, 2059 },
		{ 645, 582 },
		{ 2070, 2048 },
		{ 979, 931 },
		{ 2003, 1996 },
		{ 880, 826 },
		{ 1231, 1222 },
		{ 3012, 3010 },
		{ 1884, 1860 },
		{ 1323, 1320 },
		{ 2841, 2841 },
		{ 2841, 2841 },
		{ 2622, 2622 },
		{ 2622, 2622 },
		{ 1396, 1396 },
		{ 1396, 1396 },
		{ 2799, 2799 },
		{ 2799, 2799 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2847, 2847 },
		{ 2847, 2847 },
		{ 2848, 2848 },
		{ 2848, 2848 },
		{ 2800, 2800 },
		{ 2800, 2800 },
		{ 2850, 2850 },
		{ 2850, 2850 },
		{ 2851, 2851 },
		{ 2851, 2851 },
		{ 2581, 2581 },
		{ 2581, 2581 },
		{ 610, 548 },
		{ 2841, 2841 },
		{ 773, 715 },
		{ 2622, 2622 },
		{ 328, 280 },
		{ 1396, 1396 },
		{ 775, 717 },
		{ 2799, 2799 },
		{ 2235, 2232 },
		{ 2845, 2845 },
		{ 884, 831 },
		{ 2847, 2847 },
		{ 555, 493 },
		{ 2848, 2848 },
		{ 278, 234 },
		{ 2800, 2800 },
		{ 557, 495 },
		{ 2850, 2850 },
		{ 1234, 1225 },
		{ 2851, 2851 },
		{ 500, 445 },
		{ 2581, 2581 },
		{ 1318, 1317 },
		{ 2935, 2935 },
		{ 2935, 2935 },
		{ 1368, 1368 },
		{ 1368, 1368 },
		{ 2857, 2841 },
		{ 342, 293 },
		{ 2655, 2622 },
		{ 951, 900 },
		{ 1397, 1396 },
		{ 197, 177 },
		{ 2820, 2799 },
		{ 469, 421 },
		{ 2861, 2845 },
		{ 387, 341 },
		{ 2862, 2847 },
		{ 3077, 3075 },
		{ 2863, 2848 },
		{ 838, 778 },
		{ 2821, 2800 },
		{ 233, 195 },
		{ 2865, 2850 },
		{ 373, 323 },
		{ 2866, 2851 },
		{ 2935, 2935 },
		{ 2614, 2581 },
		{ 1368, 1368 },
		{ 1025, 985 },
		{ 1691, 1691 },
		{ 1691, 1691 },
		{ 2584, 2584 },
		{ 2584, 2584 },
		{ 2735, 2735 },
		{ 2735, 2735 },
		{ 2681, 2681 },
		{ 2681, 2681 },
		{ 2859, 2859 },
		{ 2859, 2859 },
		{ 2708, 2708 },
		{ 2708, 2708 },
		{ 2864, 2864 },
		{ 2864, 2864 },
		{ 2867, 2867 },
		{ 2867, 2867 },
		{ 2808, 2808 },
		{ 2808, 2808 },
		{ 3073, 3073 },
		{ 3073, 3073 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 2936, 2935 },
		{ 1691, 1691 },
		{ 1369, 1368 },
		{ 2584, 2584 },
		{ 787, 728 },
		{ 2735, 2735 },
		{ 788, 729 },
		{ 2681, 2681 },
		{ 696, 628 },
		{ 2859, 2859 },
		{ 332, 284 },
		{ 2708, 2708 },
		{ 286, 242 },
		{ 2864, 2864 },
		{ 963, 912 },
		{ 2867, 2867 },
		{ 741, 676 },
		{ 2808, 2808 },
		{ 903, 851 },
		{ 3073, 3073 },
		{ 451, 405 },
		{ 3008, 3008 },
		{ 905, 853 },
		{ 2739, 2739 },
		{ 2739, 2739 },
		{ 2740, 2740 },
		{ 2740, 2740 },
		{ 1692, 1691 },
		{ 743, 680 },
		{ 2617, 2584 },
		{ 597, 535 },
		{ 2762, 2735 },
		{ 266, 224 },
		{ 2709, 2681 },
		{ 702, 635 },
		{ 2873, 2859 },
		{ 629, 566 },
		{ 2736, 2708 },
		{ 1041, 1001 },
		{ 2875, 2864 },
		{ 666, 600 },
		{ 2876, 2867 },
		{ 1262, 1261 },
		{ 2829, 2808 },
		{ 705, 638 },
		{ 3074, 3073 },
		{ 2739, 2739 },
		{ 3009, 3008 },
		{ 2740, 2740 },
		{ 706, 639 },
		{ 2512, 2512 },
		{ 2512, 2512 },
		{ 2872, 2872 },
		{ 2872, 2872 },
		{ 1850, 1850 },
		{ 1850, 1850 },
		{ 2455, 2455 },
		{ 2455, 2455 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 1767, 1767 },
		{ 1767, 1767 },
		{ 1318, 1318 },
		{ 1318, 1318 },
		{ 2883, 2883 },
		{ 2883, 2883 },
		{ 2884, 2884 },
		{ 2884, 2884 },
		{ 2885, 2885 },
		{ 2885, 2885 },
		{ 1827, 1827 },
		{ 1827, 1827 },
		{ 2765, 2739 },
		{ 2512, 2512 },
		{ 2766, 2740 },
		{ 2872, 2872 },
		{ 707, 640 },
		{ 1850, 1850 },
		{ 667, 601 },
		{ 2455, 2455 },
		{ 377, 329 },
		{ 2289, 2289 },
		{ 1798, 1795 },
		{ 1767, 1767 },
		{ 669, 603 },
		{ 1318, 1318 },
		{ 980, 933 },
		{ 2883, 2883 },
		{ 711, 647 },
		{ 2884, 2884 },
		{ 394, 348 },
		{ 2885, 2885 },
		{ 862, 804 },
		{ 1827, 1827 },
		{ 921, 870 },
		{ 2748, 2748 },
		{ 2748, 2748 },
		{ 3025, 3025 },
		{ 3025, 3025 },
		{ 2546, 2512 },
		{ 671, 605 },
		{ 2881, 2872 },
		{ 987, 940 },
		{ 1851, 1850 },
		{ 1056, 1020 },
		{ 2456, 2455 },
		{ 988, 941 },
		{ 2290, 2289 },
		{ 303, 256 },
		{ 1768, 1767 },
		{ 811, 751 },
		{ 1319, 1318 },
		{ 3030, 3027 },
		{ 2888, 2883 },
		{ 2939, 2937 },
		{ 2889, 2884 },
		{ 812, 752 },
		{ 2890, 2885 },
		{ 2748, 2748 },
		{ 1828, 1827 },
		{ 3025, 3025 },
		{ 1134, 1106 },
		{ 2612, 2612 },
		{ 2612, 2612 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 2788, 2788 },
		{ 2788, 2788 },
		{ 2789, 2789 },
		{ 2789, 2789 },
		{ 2830, 2830 },
		{ 2830, 2830 },
		{ 2616, 2616 },
		{ 2616, 2616 },
		{ 2521, 2521 },
		{ 2521, 2521 },
		{ 2723, 2723 },
		{ 2723, 2723 },
		{ 2902, 2902 },
		{ 2902, 2902 },
		{ 2774, 2748 },
		{ 2612, 2612 },
		{ 3026, 3025 },
		{ 1305, 1305 },
		{ 250, 212 },
		{ 2893, 2893 },
		{ 350, 301 },
		{ 1741, 1741 },
		{ 516, 461 },
		{ 2788, 2788 },
		{ 417, 372 },
		{ 2789, 2789 },
		{ 996, 949 },
		{ 2830, 2830 },
		{ 720, 655 },
		{ 2616, 2616 },
		{ 931, 880 },
		{ 2521, 2521 },
		{ 1068, 1033 },
		{ 2723, 2723 },
		{ 872, 816 },
		{ 2902, 2902 },
		{ 637, 574 },
		{ 1793, 1793 },
		{ 1793, 1793 },
		{ 2979, 2979 },
		{ 2979, 2979 },
		{ 2647, 2612 },
		{ 934, 883 },
		{ 1306, 1305 },
		{ 578, 515 },
		{ 2895, 2893 },
		{ 1219, 1208 },
		{ 1742, 1741 },
		{ 398, 353 },
		{ 2809, 2788 },
		{ 338, 289 },
		{ 2810, 2789 },
		{ 822, 763 },
		{ 2846, 2830 },
		{ 823, 764 },
		{ 2650, 2616 },
		{ 1224, 1215 },
		{ 2522, 2521 },
		{ 327, 279 },
		{ 2750, 2723 },
		{ 1793, 1793 },
		{ 2903, 2902 },
		{ 2979, 2979 },
		{ 1646, 1639 },
		{ 1353, 1353 },
		{ 1353, 1353 },
		{ 1717, 1717 },
		{ 1717, 1717 },
		{ 2643, 2643 },
		{ 2643, 2643 },
		{ 3114, 3114 },
		{ 3114, 3114 },
		{ 3248, 3248 },
		{ 3256, 3256 },
		{ 3260, 3260 },
		{ 1622, 1600 },
		{ 2246, 2244 },
		{ 2222, 2215 },
		{ 1578, 1560 },
		{ 2014, 2010 },
		{ 2001, 1993 },
		{ 1626, 1612 },
		{ 2207, 2190 },
		{ 2227, 2221 },
		{ 2002, 1995 },
		{ 2208, 2192 },
		{ 1794, 1793 },
		{ 1353, 1353 },
		{ 2980, 2979 },
		{ 1717, 1717 },
		{ 1624, 1606 },
		{ 2643, 2643 },
		{ 2209, 2194 },
		{ 3114, 3114 },
		{ 1621, 1597 },
		{ 3248, 3248 },
		{ 3256, 3256 },
		{ 3260, 3260 },
		{ 2210, 2196 },
		{ 1623, 1603 },
		{ 2211, 2198 },
		{ 1625, 1609 },
		{ 2212, 2200 },
		{ 1982, 1970 },
		{ 1665, 1664 },
		{ 2175, 2161 },
		{ 1640, 1630 },
		{ 2025, 2024 },
		{ 2011, 2005 },
		{ 1784, 1783 },
		{ 1679, 1678 },
		{ 1680, 1679 },
		{ 1707, 1706 },
		{ 1354, 1353 },
		{ 3222, 3221 },
		{ 1718, 1717 },
		{ 3223, 3222 },
		{ 2644, 2643 },
		{ 1842, 1841 },
		{ 3115, 3114 },
		{ 1843, 1842 },
		{ 3251, 3248 },
		{ 3258, 3256 },
		{ 3261, 3260 },
		{ 1708, 1707 },
		{ 3146, 3145 },
		{ 1757, 1756 },
		{ 1758, 1757 },
		{ 1385, 1384 },
		{ 1656, 1652 },
		{ 1733, 1732 },
		{ 1734, 1733 },
		{ 1384, 1383 },
		{ 1652, 1647 },
		{ 1653, 1648 },
		{ 1428, 1427 },
		{ 1783, 1782 },
		{ 2909, 2909 },
		{ 2906, 2909 },
		{ 165, 165 },
		{ 162, 165 },
		{ 1954, 1939 },
		{ 1955, 1939 },
		{ 1976, 1964 },
		{ 1977, 1964 },
		{ 2256, 2256 },
		{ 2030, 2030 },
		{ 164, 160 },
		{ 2914, 2910 },
		{ 170, 166 },
		{ 2316, 2291 },
		{ 90, 72 },
		{ 163, 160 },
		{ 2913, 2910 },
		{ 169, 166 },
		{ 2315, 2291 },
		{ 89, 72 },
		{ 2038, 2037 },
		{ 2261, 2257 },
		{ 2908, 2904 },
		{ 2909, 2909 },
		{ 2035, 2031 },
		{ 165, 165 },
		{ 2260, 2257 },
		{ 2907, 2904 },
		{ 2262, 2259 },
		{ 2034, 2031 },
		{ 2264, 2263 },
		{ 2256, 2256 },
		{ 2030, 2030 },
		{ 2107, 2086 },
		{ 2915, 2912 },
		{ 2917, 2916 },
		{ 2448, 2416 },
		{ 171, 168 },
		{ 2910, 2909 },
		{ 173, 172 },
		{ 166, 165 },
		{ 3157, 3149 },
		{ 121, 105 },
		{ 2383, 2348 },
		{ 1914, 1895 },
		{ 2036, 2033 },
		{ 2257, 2256 },
		{ 2031, 2030 },
		{ 1989, 1979 },
		{ 2251, 2250 },
		{ 2105, 2084 },
		{ 2348, 2316 },
		{ 105, 90 },
		{ 1895, 1873 },
		{ 172, 170 },
		{ 2259, 2255 },
		{ 2037, 2035 },
		{ 2912, 2908 },
		{ 2086, 2064 },
		{ 168, 164 },
		{ 2263, 2261 },
		{ 2916, 2914 },
		{ 0, 1813 },
		{ 0, 2668 },
		{ 2709, 2709 },
		{ 2709, 2709 },
		{ 0, 1328 },
		{ 0, 3045 },
		{ 0, 2956 },
		{ 2873, 2873 },
		{ 2873, 2873 },
		{ 0, 3048 },
		{ 0, 2874 },
		{ 2875, 2875 },
		{ 2875, 2875 },
		{ 2876, 2876 },
		{ 2876, 2876 },
		{ 0, 2579 },
		{ 0, 3053 },
		{ 0, 2671 },
		{ 0, 2963 },
		{ 2456, 2456 },
		{ 2456, 2456 },
		{ 2810, 2810 },
		{ 2810, 2810 },
		{ 2881, 2881 },
		{ 2881, 2881 },
		{ 2709, 2709 },
		{ 0, 1339 },
		{ 0, 2675 },
		{ 0, 2970 },
		{ 0, 3064 },
		{ 2873, 2873 },
		{ 2762, 2762 },
		{ 2762, 2762 },
		{ 0, 2534 },
		{ 2875, 2875 },
		{ 0, 2535 },
		{ 2876, 2876 },
		{ 2888, 2888 },
		{ 2888, 2888 },
		{ 2889, 2889 },
		{ 2889, 2889 },
		{ 0, 2818 },
		{ 2456, 2456 },
		{ 0, 2678 },
		{ 2810, 2810 },
		{ 0, 3078 },
		{ 2881, 2881 },
		{ 2890, 2890 },
		{ 2890, 2890 },
		{ 2765, 2765 },
		{ 2765, 2765 },
		{ 2766, 2766 },
		{ 2766, 2766 },
		{ 0, 2536 },
		{ 2762, 2762 },
		{ 2820, 2820 },
		{ 2820, 2820 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 0, 2721 },
		{ 2888, 2888 },
		{ 0, 3082 },
		{ 2889, 2889 },
		{ 2821, 2821 },
		{ 2821, 2821 },
		{ 173, 173 },
		{ 174, 173 },
		{ 2614, 2614 },
		{ 2614, 2614 },
		{ 0, 3086 },
		{ 2890, 2890 },
		{ 0, 2991 },
		{ 2765, 2765 },
		{ 0, 2828 },
		{ 2766, 2766 },
		{ 2644, 2644 },
		{ 2644, 2644 },
		{ 0, 3089 },
		{ 2820, 2820 },
		{ 0, 1818 },
		{ 2895, 2895 },
		{ 2903, 2903 },
		{ 2903, 2903 },
		{ 2829, 2829 },
		{ 2829, 2829 },
		{ 0, 2831 },
		{ 2821, 2821 },
		{ 0, 3094 },
		{ 173, 173 },
		{ 0, 2834 },
		{ 2614, 2614 },
		{ 2617, 2617 },
		{ 2617, 2617 },
		{ 0, 2835 },
		{ 0, 3000 },
		{ 0, 2777 },
		{ 2917, 2917 },
		{ 2918, 2917 },
		{ 2644, 2644 },
		{ 0, 2728 },
		{ 0, 2498 },
		{ 2650, 2650 },
		{ 2650, 2650 },
		{ 0, 2991 },
		{ 2903, 2903 },
		{ 0, 3105 },
		{ 2829, 2829 },
		{ 0, 2781 },
		{ 0, 1292 },
		{ 0, 2925 },
		{ 0, 2591 },
		{ 0, 2278 },
		{ 0, 2691 },
		{ 0, 2692 },
		{ 2617, 2617 },
		{ 2846, 2846 },
		{ 2846, 2846 },
		{ 0, 3016 },
		{ 0, 1313 },
		{ 2917, 2917 },
		{ 2736, 2736 },
		{ 2736, 2736 },
		{ 2038, 2038 },
		{ 2039, 2038 },
		{ 2650, 2650 },
		{ 2655, 2655 },
		{ 2655, 2655 },
		{ 2546, 2546 },
		{ 2546, 2546 },
		{ 2264, 2264 },
		{ 2265, 2264 },
		{ 0, 2660 },
		{ 0, 3125 },
		{ 0, 2940 },
		{ 0, 2548 },
		{ 0, 1344 },
		{ 2857, 2857 },
		{ 2857, 2857 },
		{ 2846, 2846 },
		{ 0, 3031 },
		{ 0, 2702 },
		{ 0, 2945 },
		{ 0, 2548 },
		{ 2736, 2736 },
		{ 0, 1802 },
		{ 2038, 2038 },
		{ 0, 3035 },
		{ 0, 1360 },
		{ 2655, 2655 },
		{ 0, 3136 },
		{ 2546, 2546 },
		{ 0, 2419 },
		{ 2264, 2264 },
		{ 2863, 2863 },
		{ 2863, 2863 },
		{ 2750, 2750 },
		{ 2750, 2750 },
		{ 0, 2603 },
		{ 1271, 1271 },
		{ 2857, 2857 },
		{ 1573, 1573 },
		{ 2275, 2274 },
		{ 1415, 1414 },
		{ 2258, 2260 },
		{ 167, 169 },
		{ 2911, 2907 },
		{ 3161, 3157 },
		{ 0, 2315 },
		{ 1933, 1914 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2863, 2863 },
		{ 0, 0 },
		{ 2750, 2750 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1271, 1271 },
		{ 0, 2419 },
		{ 1573, 1573 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -70, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 46, 433, 0 },
		{ -179, 3145, 0 },
		{ 5, 0, 0 },
		{ -1270, 1095, -31 },
		{ 7, 0, -31 },
		{ -1274, 2088, -33 },
		{ 9, 0, -33 },
		{ -1287, 3469, 157 },
		{ 11, 0, 157 },
		{ -1308, 3476, 161 },
		{ 13, 0, 161 },
		{ -1321, 3470, 169 },
		{ 15, 0, 169 },
		{ -1356, 3608, 0 },
		{ 17, 0, 0 },
		{ -1371, 3482, 153 },
		{ 19, 0, 153 },
		{ -1399, 3462, 23 },
		{ 21, 0, 23 },
		{ -1441, 547, 0 },
		{ 23, 0, 0 },
		{ -1667, 3609, 0 },
		{ 25, 0, 0 },
		{ -1694, 3463, 0 },
		{ 27, 0, 0 },
		{ -1720, 3464, 0 },
		{ 29, 0, 0 },
		{ -1744, 3475, 0 },
		{ 31, 0, 0 },
		{ -1770, 3486, 0 },
		{ 33, 0, 0 },
		{ -1796, 3488, 173 },
		{ 35, 0, 173 },
		{ -1830, 3615, 280 },
		{ 37, 0, 280 },
		{ 40, 218, 0 },
		{ -1866, 661, 0 },
		{ 42, 216, 0 },
		{ -2055, 205, 0 },
		{ -2267, 3616, 0 },
		{ 43, 0, 0 },
		{ 46, 14, 0 },
		{ -88, 319, 0 },
		{ -2920, 3480, 165 },
		{ 47, 0, 165 },
		{ -2938, 3612, 188 },
		{ 49, 0, 188 },
		{ 2982, 1615, 0 },
		{ 51, 0, 0 },
		{ -2984, 3617, 286 },
		{ 53, 0, 286 },
		{ -3011, 3618, 191 },
		{ 55, 0, 191 },
		{ -3029, 3468, 184 },
		{ 57, 0, 184 },
		{ -3076, 3610, 177 },
		{ 59, 0, 177 },
		{ -3117, 3490, 183 },
		{ 61, 0, 183 },
		{ 46, 1, 0 },
		{ 63, 0, 0 },
		{ -3168, 2054, 0 },
		{ 65, 0, 0 },
		{ -3178, 2105, 45 },
		{ 67, 0, 45 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 444 },
		{ 3149, 5137, 451 },
		{ 0, 0, 258 },
		{ 0, 0, 260 },
		{ 159, 1301, 277 },
		{ 159, 1419, 277 },
		{ 159, 1384, 277 },
		{ 159, 1406, 277 },
		{ 159, 1410, 277 },
		{ 159, 1441, 277 },
		{ 159, 1447, 277 },
		{ 159, 1469, 277 },
		{ 3242, 3189, 451 },
		{ 159, 1480, 277 },
		{ 3242, 1904, 276 },
		{ 104, 2919, 451 },
		{ 159, 0, 277 },
		{ 0, 0, 451 },
		{ -89, 7, 254 },
		{ -90, 5175, 0 },
		{ 159, 1471, 277 },
		{ 159, 1515, 277 },
		{ 159, 1484, 277 },
		{ 159, 1498, 277 },
		{ 159, 1480, 277 },
		{ 159, 1480, 277 },
		{ 159, 1487, 277 },
		{ 159, 1502, 277 },
		{ 159, 1521, 277 },
		{ 3201, 2606, 0 },
		{ 159, 833, 277 },
		{ 3242, 2057, 273 },
		{ 119, 1632, 0 },
		{ 3242, 2034, 274 },
		{ 3149, 5160, 0 },
		{ 159, 842, 277 },
		{ 159, 840, 277 },
		{ 159, 841, 277 },
		{ 159, 837, 277 },
		{ 159, 0, 265 },
		{ 159, 882, 277 },
		{ 159, 884, 277 },
		{ 159, 868, 277 },
		{ 159, 873, 277 },
		{ 3239, 3260, 0 },
		{ 159, 881, 277 },
		{ 133, 1710, 0 },
		{ 119, 0, 0 },
		{ 3151, 2929, 275 },
		{ 135, 1618, 0 },
		{ 0, 0, 256 },
		{ 159, 913, 261 },
		{ 159, 915, 277 },
		{ 159, 907, 277 },
		{ 159, 912, 277 },
		{ 159, 910, 277 },
		{ 159, 903, 277 },
		{ 159, 0, 268 },
		{ 159, 904, 277 },
		{ 0, 0, 270 },
		{ 159, 910, 277 },
		{ 133, 0, 0 },
		{ 3151, 2993, 273 },
		{ 135, 0, 0 },
		{ 3151, 2896, 274 },
		{ 159, 925, 277 },
		{ 159, 922, 277 },
		{ 159, 923, 277 },
		{ 159, 958, 277 },
		{ 159, 1054, 277 },
		{ 159, 0, 267 },
		{ 159, 1163, 277 },
		{ 159, 1236, 277 },
		{ 159, 1228, 277 },
		{ 159, 0, 263 },
		{ 159, 1308, 277 },
		{ 159, 0, 264 },
		{ 159, 0, 266 },
		{ 159, 1276, 277 },
		{ 159, 1276, 277 },
		{ 159, 0, 262 },
		{ 159, 1297, 277 },
		{ 159, 0, 269 },
		{ 159, 890, 277 },
		{ 159, 1323, 277 },
		{ 0, 0, 272 },
		{ 159, 1306, 277 },
		{ 159, 1308, 277 },
		{ 3259, 1376, 271 },
		{ 3149, 5133, 451 },
		{ 165, 0, 258 },
		{ 0, 0, 259 },
		{ -163, 21, 254 },
		{ -164, 5182, 0 },
		{ 3205, 5158, 0 },
		{ 3149, 5135, 0 },
		{ 0, 0, 255 },
		{ 3149, 5155, 0 },
		{ -169, 5381, 0 },
		{ -170, 5177, 0 },
		{ 173, 0, 256 },
		{ 3149, 5157, 0 },
		{ 3205, 5283, 0 },
		{ 0, 0, 257 },
		{ 3206, 1861, 151 },
		{ 2167, 4503, 151 },
		{ 3075, 4793, 151 },
		{ 3206, 4705, 151 },
		{ 0, 0, 151 },
		{ 3191, 3738, 0 },
		{ 2182, 3128, 0 },
		{ 3191, 3969, 0 },
		{ 3191, 3606, 0 },
		{ 3179, 3693, 0 },
		{ 2167, 4527, 0 },
		{ 3196, 3587, 0 },
		{ 2167, 4513, 0 },
		{ 3166, 3790, 0 },
		{ 3200, 3567, 0 },
		{ 3166, 3639, 0 },
		{ 3206, 4683, 0 },
		{ 3010, 4727, 0 },
		{ 3196, 3621, 0 },
		{ 2182, 4086, 0 },
		{ 3075, 4803, 0 },
		{ 2113, 3847, 0 },
		{ 2113, 3850, 0 },
		{ 2135, 3433, 0 },
		{ 2116, 4189, 0 },
		{ 2097, 4247, 0 },
		{ 2116, 4176, 0 },
		{ 2135, 3436, 0 },
		{ 2135, 3458, 0 },
		{ 3196, 3655, 0 },
		{ 3116, 4318, 0 },
		{ 3191, 3995, 0 },
		{ 2167, 4480, 0 },
		{ 1240, 4422, 0 },
		{ 2113, 3809, 0 },
		{ 2135, 3474, 0 },
		{ 2135, 3476, 0 },
		{ 3075, 4981, 0 },
		{ 2113, 3803, 0 },
		{ 3179, 4170, 0 },
		{ 3191, 3960, 0 },
		{ 2182, 4048, 0 },
		{ 2652, 4560, 0 },
		{ 3075, 3985, 0 },
		{ 3116, 4309, 0 },
		{ 3166, 3937, 0 },
		{ 3166, 3792, 0 },
		{ 3166, 3881, 0 },
		{ 2075, 3637, 0 },
		{ 3075, 4865, 0 },
		{ 3191, 4008, 0 },
		{ 3116, 3986, 0 },
		{ 2182, 4109, 0 },
		{ 2135, 3493, 0 },
		{ 2135, 3409, 0 },
		{ 3200, 3782, 0 },
		{ 3179, 4053, 0 },
		{ 2075, 3639, 0 },
		{ 2097, 4250, 0 },
		{ 3075, 4775, 0 },
		{ 2167, 4454, 0 },
		{ 2167, 4455, 0 },
		{ 3191, 4000, 0 },
		{ 2135, 3410, 0 },
		{ 2167, 4511, 0 },
		{ 3191, 4011, 0 },
		{ 2652, 4656, 0 },
		{ 3075, 4845, 0 },
		{ 3200, 3713, 0 },
		{ 3116, 4294, 0 },
		{ 3166, 3865, 0 },
		{ 2135, 3411, 0 },
		{ 3200, 3661, 0 },
		{ 3191, 3965, 0 },
		{ 1240, 4411, 0 },
		{ 2097, 4213, 0 },
		{ 3116, 4347, 0 },
		{ 2182, 3987, 0 },
		{ 1127, 3580, 0 },
		{ 3191, 4010, 0 },
		{ 3179, 4162, 0 },
		{ 3075, 4941, 0 },
		{ 2652, 4597, 0 },
		{ 2135, 3412, 0 },
		{ 2182, 4094, 0 },
		{ 3200, 3778, 0 },
		{ 995, 4264, 0 },
		{ 2167, 4470, 0 },
		{ 2075, 3619, 0 },
		{ 2075, 3631, 0 },
		{ 2167, 4505, 0 },
		{ 3166, 3923, 0 },
		{ 2135, 3413, 0 },
		{ 3010, 4724, 0 },
		{ 3179, 4129, 0 },
		{ 3200, 3742, 0 },
		{ 2113, 3845, 0 },
		{ 2203, 3984, 0 },
		{ 2135, 3414, 0 },
		{ 3116, 4296, 0 },
		{ 3166, 3934, 0 },
		{ 2167, 4507, 0 },
		{ 2652, 4648, 0 },
		{ 2167, 4509, 0 },
		{ 3075, 5021, 0 },
		{ 3075, 4765, 0 },
		{ 2097, 4258, 0 },
		{ 2652, 4572, 0 },
		{ 2135, 3415, 0 },
		{ 3075, 4843, 0 },
		{ 3116, 4358, 0 },
		{ 2097, 4226, 0 },
		{ 3116, 4276, 0 },
		{ 2652, 4570, 0 },
		{ 3075, 5013, 0 },
		{ 2113, 3800, 0 },
		{ 3166, 3877, 0 },
		{ 2167, 4499, 0 },
		{ 3075, 4789, 0 },
		{ 1240, 4419, 0 },
		{ 3116, 4329, 0 },
		{ 1127, 3581, 0 },
		{ 2203, 4380, 0 },
		{ 2116, 4193, 0 },
		{ 3166, 3919, 0 },
		{ 2135, 3416, 0 },
		{ 3075, 4983, 0 },
		{ 2167, 4448, 0 },
		{ 2135, 3420, 0 },
		{ 0, 0, 78 },
		{ 3191, 3767, 0 },
		{ 3200, 3770, 0 },
		{ 2167, 4476, 0 },
		{ 3116, 4340, 0 },
		{ 3206, 4695, 0 },
		{ 2167, 4481, 0 },
		{ 2135, 3421, 0 },
		{ 2135, 3422, 0 },
		{ 3200, 3708, 0 },
		{ 2113, 3826, 0 },
		{ 2097, 4259, 0 },
		{ 3200, 3710, 0 },
		{ 2135, 3423, 0 },
		{ 3116, 4335, 0 },
		{ 2167, 4447, 0 },
		{ 3191, 4012, 0 },
		{ 3191, 3989, 0 },
		{ 2116, 4187, 0 },
		{ 3075, 4805, 0 },
		{ 3166, 3943, 0 },
		{ 894, 3596, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2113, 3798, 0 },
		{ 3075, 4913, 0 },
		{ 3116, 4314, 0 },
		{ 2097, 4234, 0 },
		{ 3200, 3740, 0 },
		{ 3166, 3897, 0 },
		{ 0, 0, 76 },
		{ 2135, 3424, 0 },
		{ 2113, 3805, 0 },
		{ 0, 0, 133 },
		{ 3200, 3743, 0 },
		{ 3116, 4368, 0 },
		{ 3200, 3749, 0 },
		{ 3075, 4797, 0 },
		{ 3166, 3938, 0 },
		{ 1240, 4426, 0 },
		{ 2097, 4244, 0 },
		{ 2113, 3827, 0 },
		{ 3179, 4143, 0 },
		{ 2167, 4471, 0 },
		{ 3075, 4923, 0 },
		{ 3206, 4676, 0 },
		{ 3166, 3945, 0 },
		{ 0, 0, 70 },
		{ 3166, 3950, 0 },
		{ 3075, 5011, 0 },
		{ 1240, 4413, 0 },
		{ 3116, 4338, 0 },
		{ 2167, 4502, 0 },
		{ 0, 0, 81 },
		{ 3179, 4113, 0 },
		{ 3200, 3750, 0 },
		{ 3166, 3866, 0 },
		{ 3191, 3963, 0 },
		{ 3191, 4001, 0 },
		{ 2135, 3425, 0 },
		{ 3116, 4306, 0 },
		{ 2167, 4439, 0 },
		{ 2135, 3426, 0 },
		{ 2113, 3793, 0 },
		{ 2148, 3558, 0 },
		{ 3179, 4166, 0 },
		{ 3200, 3779, 0 },
		{ 3166, 3924, 0 },
		{ 3075, 4987, 0 },
		{ 3200, 3781, 0 },
		{ 2135, 3428, 0 },
		{ 3116, 4356, 0 },
		{ 2167, 4492, 0 },
		{ 3200, 3701, 0 },
		{ 3166, 3939, 0 },
		{ 3116, 4285, 0 },
		{ 1067, 4375, 0 },
		{ 0, 0, 8 },
		{ 2113, 3814, 0 },
		{ 2116, 4182, 0 },
		{ 3116, 4303, 0 },
		{ 2148, 3569, 0 },
		{ 2135, 3429, 0 },
		{ 2652, 4589, 0 },
		{ 2167, 4436, 0 },
		{ 2113, 3844, 0 },
		{ 2167, 4440, 0 },
		{ 2167, 4443, 0 },
		{ 2116, 4194, 0 },
		{ 2135, 3430, 0 },
		{ 3200, 3714, 0 },
		{ 3200, 3611, 0 },
		{ 2167, 4463, 0 },
		{ 3196, 3663, 0 },
		{ 3166, 3903, 0 },
		{ 1240, 4409, 0 },
		{ 3179, 4132, 0 },
		{ 2135, 3431, 0 },
		{ 2203, 4382, 0 },
		{ 2182, 3389, 0 },
		{ 2135, 3434, 0 },
		{ 3075, 4853, 0 },
		{ 1240, 4429, 0 },
		{ 2182, 4068, 0 },
		{ 3206, 3557, 0 },
		{ 2148, 3557, 0 },
		{ 2116, 4198, 0 },
		{ 2113, 3810, 0 },
		{ 3200, 3769, 0 },
		{ 2167, 4530, 0 },
		{ 0, 0, 123 },
		{ 2135, 3435, 0 },
		{ 2182, 4071, 0 },
		{ 2074, 3586, 0 },
		{ 3116, 4337, 0 },
		{ 3191, 4036, 0 },
		{ 3179, 4123, 0 },
		{ 3075, 4795, 0 },
		{ 2167, 4452, 0 },
		{ 0, 0, 7 },
		{ 2116, 4174, 0 },
		{ 0, 0, 6 },
		{ 3116, 4350, 0 },
		{ 0, 0, 128 },
		{ 3179, 4125, 0 },
		{ 2167, 4465, 0 },
		{ 3206, 2103, 0 },
		{ 2135, 3437, 0 },
		{ 3179, 4142, 0 },
		{ 3191, 3999, 0 },
		{ 0, 0, 132 },
		{ 2135, 3438, 0 },
		{ 2167, 4483, 0 },
		{ 3206, 3577, 0 },
		{ 2167, 4495, 0 },
		{ 2652, 4654, 0 },
		{ 2182, 4088, 0 },
		{ 0, 0, 77 },
		{ 2097, 4236, 0 },
		{ 2135, 3439, 115 },
		{ 2135, 3440, 116 },
		{ 3075, 4781, 0 },
		{ 3116, 4327, 0 },
		{ 2135, 3442, 0 },
		{ 3166, 3906, 0 },
		{ 3191, 4016, 0 },
		{ 3191, 4032, 0 },
		{ 3166, 3907, 0 },
		{ 1240, 4418, 0 },
		{ 3191, 4040, 0 },
		{ 3166, 3909, 0 },
		{ 3196, 3671, 0 },
		{ 2182, 4063, 0 },
		{ 3116, 4366, 0 },
		{ 2167, 4450, 0 },
		{ 2135, 3443, 0 },
		{ 3200, 3728, 0 },
		{ 3075, 4985, 0 },
		{ 0, 0, 114 },
		{ 3116, 4287, 0 },
		{ 3010, 4726, 0 },
		{ 3116, 4288, 0 },
		{ 2182, 4081, 0 },
		{ 3166, 3933, 0 },
		{ 3116, 4297, 0 },
		{ 0, 0, 9 },
		{ 2135, 3444, 0 },
		{ 3116, 4304, 0 },
		{ 2148, 3560, 0 },
		{ 2203, 4400, 0 },
		{ 0, 0, 112 },
		{ 2113, 3821, 0 },
		{ 3179, 4164, 0 },
		{ 3191, 3977, 0 },
		{ 2182, 3989, 0 },
		{ 3179, 3578, 0 },
		{ 3116, 4334, 0 },
		{ 3196, 3627, 0 },
		{ 3116, 4336, 0 },
		{ 3196, 3629, 0 },
		{ 3191, 4041, 0 },
		{ 2167, 4522, 0 },
		{ 3200, 3752, 0 },
		{ 3166, 3864, 0 },
		{ 3196, 3600, 0 },
		{ 3179, 4152, 0 },
		{ 3200, 3760, 0 },
		{ 3116, 4277, 0 },
		{ 3200, 3662, 0 },
		{ 3075, 4773, 0 },
		{ 2135, 3445, 0 },
		{ 3075, 4777, 0 },
		{ 3166, 3894, 0 },
		{ 2167, 4451, 0 },
		{ 3191, 3954, 0 },
		{ 3191, 3956, 0 },
		{ 2097, 4235, 0 },
		{ 2113, 3799, 0 },
		{ 3191, 4019, 0 },
		{ 2135, 3446, 102 },
		{ 3166, 3905, 0 },
		{ 2182, 4060, 0 },
		{ 2135, 3447, 0 },
		{ 2135, 3448, 0 },
		{ 3196, 3684, 0 },
		{ 2182, 4073, 0 },
		{ 2652, 4593, 0 },
		{ 2135, 3449, 0 },
		{ 2113, 3811, 0 },
		{ 0, 0, 111 },
		{ 2652, 4652, 0 },
		{ 3075, 5007, 0 },
		{ 3200, 3703, 0 },
		{ 3200, 3705, 0 },
		{ 0, 0, 125 },
		{ 0, 0, 127 },
		{ 3179, 4127, 0 },
		{ 2182, 4099, 0 },
		{ 2113, 3823, 0 },
		{ 2167, 3701, 0 },
		{ 3200, 3709, 0 },
		{ 2167, 4518, 0 },
		{ 2135, 3450, 0 },
		{ 2167, 4524, 0 },
		{ 3116, 4361, 0 },
		{ 3179, 4155, 0 },
		{ 2135, 3451, 0 },
		{ 2203, 4390, 0 },
		{ 3196, 3675, 0 },
		{ 2652, 4574, 0 },
		{ 2135, 3452, 0 },
		{ 3075, 4863, 0 },
		{ 2113, 3792, 0 },
		{ 995, 4262, 0 },
		{ 3200, 3724, 0 },
		{ 3179, 4115, 0 },
		{ 2182, 4087, 0 },
		{ 2652, 4655, 0 },
		{ 3200, 3726, 0 },
		{ 2075, 3617, 0 },
		{ 2135, 3453, 0 },
		{ 3116, 4312, 0 },
		{ 3191, 3997, 0 },
		{ 2113, 3802, 0 },
		{ 3075, 4761, 0 },
		{ 3200, 3741, 0 },
		{ 2182, 4061, 0 },
		{ 2148, 3563, 0 },
		{ 3166, 3904, 0 },
		{ 2113, 3808, 0 },
		{ 2182, 4072, 0 },
		{ 2116, 4180, 0 },
		{ 3206, 3653, 0 },
		{ 2135, 3454, 0 },
		{ 0, 0, 71 },
		{ 2135, 3455, 0 },
		{ 3191, 4018, 0 },
		{ 3166, 3910, 0 },
		{ 3191, 4029, 0 },
		{ 3166, 3914, 0 },
		{ 2135, 3456, 117 },
		{ 2097, 4215, 0 },
		{ 3075, 4869, 0 },
		{ 2182, 4047, 0 },
		{ 2116, 4181, 0 },
		{ 3166, 3922, 0 },
		{ 2113, 3816, 0 },
		{ 2113, 3817, 0 },
		{ 2097, 4245, 0 },
		{ 2116, 4191, 0 },
		{ 3075, 4999, 0 },
		{ 3191, 3961, 0 },
		{ 3196, 3665, 0 },
		{ 3116, 4311, 0 },
		{ 3200, 3754, 0 },
		{ 2113, 3825, 0 },
		{ 0, 0, 129 },
		{ 2135, 3457, 0 },
		{ 3010, 4729, 0 },
		{ 2116, 4179, 0 },
		{ 3200, 3764, 0 },
		{ 3179, 4122, 0 },
		{ 0, 0, 124 },
		{ 0, 0, 113 },
		{ 2113, 3842, 0 },
		{ 3166, 3944, 0 },
		{ 3200, 3767, 0 },
		{ 2182, 3313, 0 },
		{ 3206, 3611, 0 },
		{ 3116, 4343, 0 },
		{ 3179, 4140, 0 },
		{ 2135, 3462, 0 },
		{ 3116, 4355, 0 },
		{ 2097, 4212, 0 },
		{ 3191, 4023, 0 },
		{ 2167, 4504, 0 },
		{ 3075, 4873, 0 },
		{ 3075, 4911, 0 },
		{ 2113, 3851, 0 },
		{ 3075, 4917, 0 },
		{ 3116, 4362, 0 },
		{ 3075, 4933, 0 },
		{ 3166, 3868, 0 },
		{ 3179, 4161, 0 },
		{ 2135, 3463, 0 },
		{ 2167, 4515, 0 },
		{ 3166, 3878, 0 },
		{ 2135, 3464, 0 },
		{ 3166, 3883, 0 },
		{ 2167, 4525, 0 },
		{ 3116, 4295, 0 },
		{ 2167, 4528, 0 },
		{ 3166, 3893, 0 },
		{ 2167, 4434, 0 },
		{ 2113, 3797, 0 },
		{ 3179, 4116, 0 },
		{ 2135, 3468, 0 },
		{ 3206, 4620, 0 },
		{ 2652, 4646, 0 },
		{ 2167, 4441, 0 },
		{ 2116, 4200, 0 },
		{ 2167, 4446, 0 },
		{ 2116, 4201, 0 },
		{ 3191, 3968, 0 },
		{ 3075, 4841, 0 },
		{ 3200, 3783, 0 },
		{ 3191, 4006, 0 },
		{ 0, 0, 104 },
		{ 3200, 3693, 0 },
		{ 3116, 4323, 0 },
		{ 3116, 4326, 0 },
		{ 3075, 4867, 0 },
		{ 2135, 3469, 0 },
		{ 2135, 3470, 0 },
		{ 3075, 4877, 0 },
		{ 3075, 4882, 0 },
		{ 3075, 4909, 0 },
		{ 2116, 4183, 0 },
		{ 2113, 3804, 0 },
		{ 0, 0, 138 },
		{ 3191, 4014, 0 },
		{ 0, 0, 126 },
		{ 0, 0, 130 },
		{ 3075, 4921, 0 },
		{ 2652, 4653, 0 },
		{ 1127, 3578, 0 },
		{ 2135, 3471, 0 },
		{ 3116, 3396, 0 },
		{ 3166, 3916, 0 },
		{ 2116, 4199, 0 },
		{ 1240, 4404, 0 },
		{ 3075, 4991, 0 },
		{ 3191, 4024, 0 },
		{ 3191, 4025, 0 },
		{ 3191, 4028, 0 },
		{ 3179, 4171, 0 },
		{ 2652, 4638, 0 },
		{ 2203, 4386, 0 },
		{ 3179, 4172, 0 },
		{ 3196, 3667, 0 },
		{ 2097, 4246, 0 },
		{ 1240, 4431, 0 },
		{ 3200, 3711, 0 },
		{ 2097, 4248, 0 },
		{ 2113, 3812, 0 },
		{ 2135, 3475, 0 },
		{ 2116, 4185, 0 },
		{ 2097, 4209, 0 },
		{ 2167, 4529, 0 },
		{ 2203, 4385, 0 },
		{ 2182, 4093, 0 },
		{ 3166, 3925, 0 },
		{ 3075, 4849, 0 },
		{ 2182, 4095, 0 },
		{ 0, 0, 65 },
		{ 0, 0, 66 },
		{ 3075, 4861, 0 },
		{ 3166, 3927, 0 },
		{ 0, 0, 75 },
		{ 0, 0, 121 },
		{ 2075, 3602, 0 },
		{ 3196, 3676, 0 },
		{ 2113, 3820, 0 },
		{ 3196, 3680, 0 },
		{ 3200, 3725, 0 },
		{ 3116, 4316, 0 },
		{ 3166, 3940, 0 },
		{ 0, 0, 106 },
		{ 3166, 3941, 0 },
		{ 0, 0, 108 },
		{ 3191, 4009, 0 },
		{ 3166, 3942, 0 },
		{ 3179, 4165, 0 },
		{ 1067, 4373, 0 },
		{ 2167, 4469, 0 },
		{ 0, 0, 136 },
		{ 2148, 3571, 0 },
		{ 2148, 3556, 0 },
		{ 3200, 3732, 0 },
		{ 1240, 4421, 0 },
		{ 2203, 4118, 0 },
		{ 3166, 3947, 0 },
		{ 995, 4267, 0 },
		{ 2097, 4219, 0 },
		{ 0, 0, 122 },
		{ 0, 0, 137 },
		{ 3166, 3948, 0 },
		{ 3166, 3949, 0 },
		{ 0, 0, 150 },
		{ 2113, 3832, 0 },
		{ 3206, 4284, 0 },
		{ 3075, 4763, 0 },
		{ 1240, 4416, 0 },
		{ 3075, 4767, 0 },
		{ 2167, 4506, 0 },
		{ 3206, 4697, 0 },
		{ 3166, 3856, 0 },
		{ 3206, 4674, 0 },
		{ 3196, 3658, 0 },
		{ 3196, 3659, 0 },
		{ 3179, 3392, 0 },
		{ 2135, 3478, 0 },
		{ 2167, 4516, 0 },
		{ 3116, 4282, 0 },
		{ 3075, 4837, 0 },
		{ 3075, 4839, 0 },
		{ 3116, 4284, 0 },
		{ 2182, 4051, 0 },
		{ 3116, 4286, 0 },
		{ 2182, 4053, 0 },
		{ 2182, 3985, 0 },
		{ 3116, 4292, 0 },
		{ 2135, 3480, 0 },
		{ 2652, 4595, 0 },
		{ 2135, 3481, 0 },
		{ 3191, 3980, 0 },
		{ 2135, 3482, 0 },
		{ 2116, 4186, 0 },
		{ 3191, 3990, 0 },
		{ 2097, 4221, 0 },
		{ 3116, 4307, 0 },
		{ 2135, 3484, 0 },
		{ 3191, 3996, 0 },
		{ 3206, 4711, 0 },
		{ 1240, 4412, 0 },
		{ 2182, 4083, 0 },
		{ 3166, 3891, 0 },
		{ 3075, 4943, 0 },
		{ 3075, 4949, 0 },
		{ 2167, 4453, 0 },
		{ 2116, 4197, 0 },
		{ 2652, 4591, 0 },
		{ 0, 0, 134 },
		{ 3166, 3892, 0 },
		{ 2167, 4458, 0 },
		{ 2167, 4462, 0 },
		{ 3116, 4320, 0 },
		{ 3116, 4321, 0 },
		{ 2167, 4466, 0 },
		{ 3075, 5015, 0 },
		{ 3075, 5017, 0 },
		{ 2167, 4468, 0 },
		{ 2135, 3486, 0 },
		{ 2182, 4091, 0 },
		{ 3200, 3755, 0 },
		{ 3200, 3758, 0 },
		{ 2167, 4477, 0 },
		{ 3196, 3637, 0 },
		{ 3196, 3654, 0 },
		{ 2097, 4207, 0 },
		{ 3206, 4717, 0 },
		{ 3200, 3766, 0 },
		{ 2135, 3488, 68 },
		{ 3200, 3768, 0 },
		{ 3075, 4801, 0 },
		{ 2182, 4050, 0 },
		{ 2135, 3489, 0 },
		{ 2135, 3490, 0 },
		{ 2203, 4398, 0 },
		{ 3116, 4353, 0 },
		{ 3206, 4668, 0 },
		{ 3179, 4144, 0 },
		{ 3200, 3773, 0 },
		{ 3200, 3775, 0 },
		{ 1127, 3579, 0 },
		{ 2097, 4240, 0 },
		{ 3166, 3920, 0 },
		{ 2148, 3559, 0 },
		{ 2075, 3641, 0 },
		{ 2075, 3642, 0 },
		{ 3191, 4030, 0 },
		{ 2113, 3819, 0 },
		{ 1240, 4424, 0 },
		{ 3196, 3672, 0 },
		{ 3166, 3929, 0 },
		{ 3206, 4701, 0 },
		{ 3206, 4702, 0 },
		{ 2167, 4532, 0 },
		{ 0, 0, 69 },
		{ 0, 0, 72 },
		{ 3075, 4925, 0 },
		{ 1240, 4430, 0 },
		{ 2097, 4204, 0 },
		{ 3166, 3931, 0 },
		{ 1240, 4408, 0 },
		{ 3166, 3932, 0 },
		{ 0, 0, 118 },
		{ 3200, 3690, 0 },
		{ 3200, 3691, 0 },
		{ 3166, 3936, 0 },
		{ 0, 0, 110 },
		{ 2135, 3491, 0 },
		{ 3075, 4997, 0 },
		{ 0, 0, 119 },
		{ 0, 0, 120 },
		{ 2182, 4106, 0 },
		{ 2097, 4222, 0 },
		{ 3179, 4130, 0 },
		{ 995, 4263, 0 },
		{ 2116, 4188, 0 },
		{ 1240, 4425, 0 },
		{ 3200, 3694, 0 },
		{ 3010, 4733, 0 },
		{ 0, 0, 3 },
		{ 2167, 4460, 0 },
		{ 3206, 4694, 0 },
		{ 2652, 4642, 0 },
		{ 3075, 4771, 0 },
		{ 3179, 4136, 0 },
		{ 3116, 4317, 0 },
		{ 3200, 3697, 0 },
		{ 3116, 4319, 0 },
		{ 2167, 4467, 0 },
		{ 2135, 3492, 0 },
		{ 2116, 4195, 0 },
		{ 2652, 4556, 0 },
		{ 2113, 3834, 0 },
		{ 2113, 3841, 0 },
		{ 2167, 4472, 0 },
		{ 3179, 4149, 0 },
		{ 1067, 4374, 0 },
		{ 2167, 3402, 0 },
		{ 3116, 4333, 0 },
		{ 2182, 4055, 0 },
		{ 0, 0, 79 },
		{ 2167, 4490, 0 },
		{ 0, 0, 87 },
		{ 3075, 4851, 0 },
		{ 2167, 4491, 0 },
		{ 3075, 4855, 0 },
		{ 3200, 3704, 0 },
		{ 2167, 4493, 0 },
		{ 3196, 3682, 0 },
		{ 3206, 4680, 0 },
		{ 2167, 4496, 0 },
		{ 2182, 4062, 0 },
		{ 2097, 4208, 0 },
		{ 3200, 3706, 0 },
		{ 2097, 4211, 0 },
		{ 3116, 4344, 0 },
		{ 2182, 4064, 0 },
		{ 3116, 4349, 0 },
		{ 2167, 4508, 0 },
		{ 0, 0, 74 },
		{ 2182, 4065, 0 },
		{ 2182, 4067, 0 },
		{ 3075, 4927, 0 },
		{ 2116, 4184, 0 },
		{ 3200, 3707, 0 },
		{ 3179, 4112, 0 },
		{ 2167, 4517, 0 },
		{ 2182, 4069, 0 },
		{ 2167, 4521, 0 },
		{ 2135, 3494, 0 },
		{ 3116, 4363, 0 },
		{ 3191, 4015, 0 },
		{ 3075, 4993, 0 },
		{ 2116, 4190, 0 },
		{ 2097, 4239, 0 },
		{ 3075, 5005, 0 },
		{ 2113, 3853, 0 },
		{ 3206, 4671, 0 },
		{ 2113, 3787, 0 },
		{ 2135, 3495, 0 },
		{ 2182, 4085, 0 },
		{ 2075, 3633, 0 },
		{ 3206, 4685, 0 },
		{ 2167, 4437, 0 },
		{ 2167, 4438, 0 },
		{ 894, 3600, 0 },
		{ 0, 3592, 0 },
		{ 3179, 4134, 0 },
		{ 2203, 4387, 0 },
		{ 2167, 4444, 0 },
		{ 3166, 3869, 0 },
		{ 1240, 4414, 0 },
		{ 3075, 4791, 0 },
		{ 3166, 3871, 0 },
		{ 2135, 3496, 0 },
		{ 3200, 3715, 0 },
		{ 3166, 3879, 0 },
		{ 2097, 4210, 0 },
		{ 3116, 4308, 0 },
		{ 3166, 3880, 0 },
		{ 3179, 4150, 0 },
		{ 3200, 3716, 0 },
		{ 2652, 4562, 0 },
		{ 2652, 4564, 0 },
		{ 3075, 4847, 0 },
		{ 2167, 4461, 0 },
		{ 0, 0, 73 },
		{ 2097, 4214, 0 },
		{ 3200, 3717, 0 },
		{ 3191, 3987, 0 },
		{ 3166, 3888, 0 },
		{ 3166, 3889, 0 },
		{ 3166, 3890, 0 },
		{ 3200, 3718, 0 },
		{ 2182, 4057, 0 },
		{ 2182, 4059, 0 },
		{ 0, 0, 142 },
		{ 0, 0, 143 },
		{ 2116, 4192, 0 },
		{ 1240, 4417, 0 },
		{ 3200, 3719, 0 },
		{ 2097, 4241, 0 },
		{ 2097, 4243, 0 },
		{ 3010, 4731, 0 },
		{ 0, 0, 10 },
		{ 3075, 4919, 0 },
		{ 0, 0, 12 },
		{ 2113, 3806, 0 },
		{ 3200, 3720, 0 },
		{ 3075, 4408, 0 },
		{ 3206, 4719, 0 },
		{ 3179, 4114, 0 },
		{ 3075, 4935, 0 },
		{ 3075, 4939, 0 },
		{ 3200, 3722, 0 },
		{ 2135, 3497, 0 },
		{ 3116, 4341, 0 },
		{ 3116, 4342, 0 },
		{ 2167, 4500, 0 },
		{ 2135, 3498, 0 },
		{ 3206, 4686, 0 },
		{ 3075, 4989, 0 },
		{ 3206, 4687, 0 },
		{ 2097, 4255, 0 },
		{ 0, 0, 88 },
		{ 2182, 4066, 0 },
		{ 3116, 4348, 0 },
		{ 0, 0, 86 },
		{ 3196, 3668, 0 },
		{ 2116, 4175, 0 },
		{ 0, 0, 89 },
		{ 3206, 4704, 0 },
		{ 3116, 4351, 0 },
		{ 3196, 3670, 0 },
		{ 2167, 4510, 0 },
		{ 2113, 3815, 0 },
		{ 3166, 3908, 0 },
		{ 2167, 4514, 0 },
		{ 2135, 3499, 0 },
		{ 3200, 3730, 0 },
		{ 0, 0, 67 },
		{ 0, 0, 105 },
		{ 0, 0, 107 },
		{ 2182, 4076, 0 },
		{ 2652, 4558, 0 },
		{ 3166, 3912, 0 },
		{ 2167, 4520, 0 },
		{ 3116, 4365, 0 },
		{ 3191, 4017, 0 },
		{ 2167, 4523, 0 },
		{ 0, 0, 148 },
		{ 3116, 4367, 0 },
		{ 3166, 3913, 0 },
		{ 2167, 4526, 0 },
		{ 3116, 4274, 0 },
		{ 3200, 3731, 0 },
		{ 3166, 3915, 0 },
		{ 3075, 4810, 0 },
		{ 2135, 3500, 0 },
		{ 2097, 4223, 0 },
		{ 2097, 4224, 0 },
		{ 2167, 4435, 0 },
		{ 2652, 4650, 0 },
		{ 3200, 3733, 0 },
		{ 3200, 3736, 0 },
		{ 3166, 3921, 0 },
		{ 2203, 4395, 0 },
		{ 0, 4265, 0 },
		{ 3200, 3737, 0 },
		{ 3200, 3738, 0 },
		{ 3116, 4301, 0 },
		{ 3191, 4034, 0 },
		{ 2182, 4107, 0 },
		{ 3075, 4871, 0 },
		{ 3116, 4305, 0 },
		{ 3200, 3739, 0 },
		{ 2182, 4046, 0 },
		{ 3206, 4710, 0 },
		{ 0, 0, 20 },
		{ 2113, 3828, 0 },
		{ 2113, 3829, 0 },
		{ 0, 0, 139 },
		{ 2113, 3831, 0 },
		{ 0, 0, 141 },
		{ 3166, 3930, 0 },
		{ 2167, 4457, 0 },
		{ 0, 0, 103 },
		{ 2135, 3501, 0 },
		{ 2097, 4251, 0 },
		{ 2097, 4253, 0 },
		{ 2135, 3502, 0 },
		{ 2097, 4257, 0 },
		{ 3075, 4937, 0 },
		{ 2113, 3840, 0 },
		{ 2182, 4058, 0 },
		{ 3116, 4322, 0 },
		{ 0, 0, 84 },
		{ 2097, 4260, 0 },
		{ 1240, 4423, 0 },
		{ 2135, 3503, 0 },
		{ 2097, 4206, 0 },
		{ 3166, 3935, 0 },
		{ 2167, 4473, 0 },
		{ 3206, 4706, 0 },
		{ 3206, 4709, 0 },
		{ 3075, 4995, 0 },
		{ 2167, 4474, 0 },
		{ 3116, 4331, 0 },
		{ 3116, 4332, 0 },
		{ 2135, 3504, 0 },
		{ 2113, 3843, 0 },
		{ 3200, 3744, 0 },
		{ 3179, 4133, 0 },
		{ 3200, 3745, 0 },
		{ 2113, 3846, 0 },
		{ 3116, 4339, 0 },
		{ 3179, 4139, 0 },
		{ 3200, 3746, 0 },
		{ 2167, 4497, 0 },
		{ 0, 0, 92 },
		{ 3206, 4692, 0 },
		{ 0, 0, 109 },
		{ 2097, 4218, 0 },
		{ 3206, 4281, 0 },
		{ 2167, 4501, 0 },
		{ 0, 0, 146 },
		{ 3200, 3747, 0 },
		{ 3116, 4345, 0 },
		{ 3200, 3748, 0 },
		{ 2135, 3505, 64 },
		{ 3179, 4145, 0 },
		{ 2182, 4070, 0 },
		{ 2097, 4225, 0 },
		{ 3196, 3653, 0 },
		{ 3010, 4237, 0 },
		{ 0, 0, 93 },
		{ 2113, 3788, 0 },
		{ 3206, 4658, 0 },
		{ 1067, 4371, 0 },
		{ 0, 4372, 0 },
		{ 3200, 3751, 0 },
		{ 3179, 4157, 0 },
		{ 3179, 4160, 0 },
		{ 2182, 4075, 0 },
		{ 3206, 4684, 0 },
		{ 2167, 4519, 0 },
		{ 3116, 4364, 0 },
		{ 2135, 3506, 0 },
		{ 2182, 4078, 0 },
		{ 2182, 4079, 0 },
		{ 2182, 4080, 0 },
		{ 0, 0, 97 },
		{ 3116, 4369, 0 },
		{ 2113, 3794, 0 },
		{ 3166, 3855, 0 },
		{ 0, 0, 131 },
		{ 2135, 3507, 0 },
		{ 3196, 3657, 0 },
		{ 2135, 3509, 0 },
		{ 3191, 4026, 0 },
		{ 3200, 3756, 0 },
		{ 3116, 4290, 0 },
		{ 2652, 4566, 0 },
		{ 2113, 3801, 0 },
		{ 3179, 4118, 0 },
		{ 0, 0, 99 },
		{ 3179, 4121, 0 },
		{ 2652, 4576, 0 },
		{ 2652, 4587, 0 },
		{ 3200, 3757, 0 },
		{ 0, 0, 16 },
		{ 2097, 4203, 0 },
		{ 0, 0, 18 },
		{ 0, 0, 19 },
		{ 3116, 4298, 0 },
		{ 2135, 3510, 0 },
		{ 2203, 4381, 0 },
		{ 3179, 4124, 0 },
		{ 3075, 4954, 0 },
		{ 3166, 3873, 0 },
		{ 2182, 4101, 0 },
		{ 1240, 4420, 0 },
		{ 3166, 3875, 0 },
		{ 3166, 3876, 0 },
		{ 3179, 4131, 0 },
		{ 2182, 4108, 0 },
		{ 0, 0, 63 },
		{ 3116, 4310, 0 },
		{ 3200, 3759, 0 },
		{ 2135, 3511, 0 },
		{ 3200, 3761, 0 },
		{ 2097, 4216, 0 },
		{ 1127, 3573, 0 },
		{ 2182, 4049, 0 },
		{ 2167, 4464, 0 },
		{ 0, 0, 82 },
		{ 2135, 3512, 0 },
		{ 3206, 4662, 0 },
		{ 3166, 3882, 0 },
		{ 0, 3576, 0 },
		{ 3166, 3887, 0 },
		{ 0, 0, 17 },
		{ 2182, 4056, 0 },
		{ 1240, 4415, 0 },
		{ 2135, 3513, 62 },
		{ 2135, 3514, 0 },
		{ 2097, 4230, 0 },
		{ 0, 0, 83 },
		{ 3179, 4153, 0 },
		{ 3206, 3631, 0 },
		{ 0, 0, 90 },
		{ 0, 0, 91 },
		{ 0, 0, 60 },
		{ 3179, 4156, 0 },
		{ 3191, 4004, 0 },
		{ 3191, 4005, 0 },
		{ 3200, 3771, 0 },
		{ 3191, 4007, 0 },
		{ 0, 0, 147 },
		{ 0, 0, 135 },
		{ 3179, 4163, 0 },
		{ 1240, 4427, 0 },
		{ 1240, 4428, 0 },
		{ 3200, 3772, 0 },
		{ 0, 0, 40 },
		{ 2135, 3515, 41 },
		{ 2135, 3516, 43 },
		{ 3179, 4168, 0 },
		{ 3206, 4714, 0 },
		{ 1240, 4405, 0 },
		{ 1240, 4406, 0 },
		{ 2097, 4249, 0 },
		{ 0, 0, 80 },
		{ 3179, 4169, 0 },
		{ 3200, 3776, 0 },
		{ 0, 0, 98 },
		{ 3200, 3777, 0 },
		{ 2097, 4254, 0 },
		{ 3191, 4013, 0 },
		{ 2097, 4256, 0 },
		{ 2113, 3822, 0 },
		{ 3116, 4352, 0 },
		{ 3196, 3678, 0 },
		{ 3116, 4354, 0 },
		{ 2203, 4392, 0 },
		{ 2203, 4394, 0 },
		{ 2135, 3517, 0 },
		{ 3200, 3780, 0 },
		{ 3206, 4696, 0 },
		{ 3196, 3681, 0 },
		{ 0, 0, 94 },
		{ 3206, 4700, 0 },
		{ 2135, 3519, 0 },
		{ 0, 0, 140 },
		{ 0, 0, 144 },
		{ 2097, 4205, 0 },
		{ 0, 0, 149 },
		{ 0, 0, 11 },
		{ 3179, 4119, 0 },
		{ 3179, 4120, 0 },
		{ 2182, 4077, 0 },
		{ 3191, 4021, 0 },
		{ 3191, 4022, 0 },
		{ 1240, 4432, 0 },
		{ 2135, 3520, 0 },
		{ 3200, 3688, 0 },
		{ 3179, 4126, 0 },
		{ 3200, 3689, 0 },
		{ 3206, 4659, 0 },
		{ 0, 0, 145 },
		{ 3116, 4283, 0 },
		{ 3206, 4666, 0 },
		{ 3179, 4128, 0 },
		{ 3196, 3648, 0 },
		{ 3196, 3649, 0 },
		{ 3196, 3652, 0 },
		{ 3206, 4678, 0 },
		{ 2135, 3523, 0 },
		{ 3206, 4682, 0 },
		{ 3116, 4289, 0 },
		{ 3075, 5009, 0 },
		{ 3200, 3695, 0 },
		{ 3200, 3696, 0 },
		{ 2135, 3399, 0 },
		{ 0, 0, 42 },
		{ 0, 0, 44 },
		{ 3179, 4138, 0 },
		{ 3075, 5019, 0 },
		{ 3206, 4691, 0 },
		{ 3200, 3698, 0 },
		{ 2182, 4096, 0 },
		{ 2097, 4228, 0 },
		{ 3116, 4299, 0 },
		{ 3116, 4300, 0 },
		{ 3010, 4734, 0 },
		{ 3206, 4699, 0 },
		{ 2097, 4229, 0 },
		{ 3075, 4779, 0 },
		{ 3116, 4302, 0 },
		{ 3179, 4141, 0 },
		{ 2097, 4232, 0 },
		{ 2182, 4097, 0 },
		{ 2182, 4098, 0 },
		{ 2167, 4459, 0 },
		{ 3200, 3699, 0 },
		{ 2097, 4237, 0 },
		{ 2097, 4238, 0 },
		{ 2182, 4100, 0 },
		{ 0, 0, 85 },
		{ 0, 0, 100 },
		{ 3179, 4146, 0 },
		{ 3179, 4147, 0 },
		{ 0, 4410, 0 },
		{ 3116, 4313, 0 },
		{ 0, 0, 95 },
		{ 2097, 4242, 0 },
		{ 3179, 4148, 0 },
		{ 2113, 3849, 0 },
		{ 0, 0, 13 },
		{ 2182, 4102, 0 },
		{ 2182, 4103, 0 },
		{ 0, 0, 96 },
		{ 0, 0, 61 },
		{ 0, 0, 101 },
		{ 3166, 3926, 0 },
		{ 3179, 4154, 0 },
		{ 2167, 4475, 0 },
		{ 0, 0, 15 },
		{ 2135, 3407, 0 },
		{ 3166, 3928, 0 },
		{ 2167, 4478, 0 },
		{ 3191, 3993, 0 },
		{ 2097, 4252, 0 },
		{ 3075, 4875, 0 },
		{ 3206, 4689, 0 },
		{ 2167, 4482, 0 },
		{ 2116, 4196, 0 },
		{ 2167, 4484, 0 },
		{ 3179, 4159, 0 },
		{ 3200, 3702, 0 },
		{ 0, 0, 14 },
		{ 3259, 1457, 246 },
		{ 0, 0, 247 },
		{ 3205, 5376, 248 },
		{ 3242, 1928, 252 },
		{ 1277, 2918, 253 },
		{ 0, 0, 253 },
		{ 3242, 1968, 249 },
		{ 1280, 1665, 0 },
		{ 3242, 2024, 250 },
		{ 1283, 1584, 0 },
		{ 1280, 0, 0 },
		{ 3151, 3004, 251 },
		{ 1285, 1617, 0 },
		{ 1283, 0, 0 },
		{ 3151, 2906, 249 },
		{ 1285, 0, 0 },
		{ 3151, 2916, 250 },
		{ 3196, 3647, 158 },
		{ 0, 0, 158 },
		{ 0, 0, 159 },
		{ 3220, 2267, 0 },
		{ 3242, 3222, 0 },
		{ 3259, 2391, 0 },
		{ 1293, 5235, 0 },
		{ 3239, 2895, 0 },
		{ 3242, 3171, 0 },
		{ 3254, 3286, 0 },
		{ 3250, 2749, 0 },
		{ 3253, 3332, 0 },
		{ 3259, 2374, 0 },
		{ 3253, 3305, 0 },
		{ 3255, 2244, 0 },
		{ 3142, 3007, 0 },
		{ 3257, 2508, 0 },
		{ 3201, 2558, 0 },
		{ 3220, 2286, 0 },
		{ 3260, 5013, 0 },
		{ 0, 0, 156 },
		{ 3196, 3674, 162 },
		{ 0, 0, 162 },
		{ 0, 0, 163 },
		{ 3220, 2335, 0 },
		{ 3242, 3229, 0 },
		{ 3259, 2407, 0 },
		{ 1314, 5278, 0 },
		{ 3206, 4334, 0 },
		{ 3196, 3679, 0 },
		{ 2652, 4644, 0 },
		{ 3075, 4783, 0 },
		{ 3260, 4951, 0 },
		{ 0, 0, 160 },
		{ 3010, 4737, 170 },
		{ 0, 0, 170 },
		{ 0, 0, 171 },
		{ 3242, 3176, 0 },
		{ 3088, 3091, 0 },
		{ 3257, 2513, 0 },
		{ 3259, 2363, 0 },
		{ 3242, 3169, 0 },
		{ 1329, 5130, 0 },
		{ 3242, 2829, 0 },
		{ 3224, 1672, 0 },
		{ 3242, 3177, 0 },
		{ 3259, 2388, 0 },
		{ 2852, 1636, 0 },
		{ 3255, 2132, 0 },
		{ 3249, 3041, 0 },
		{ 3142, 3029, 0 },
		{ 3201, 2622, 0 },
		{ 3044, 3066, 0 },
		{ 1340, 5152, 0 },
		{ 3242, 2814, 0 },
		{ 3250, 2795, 0 },
		{ 3220, 2259, 0 },
		{ 3242, 3137, 0 },
		{ 1345, 5260, 0 },
		{ 3252, 2776, 0 },
		{ 3247, 1744, 0 },
		{ 3201, 2613, 0 },
		{ 3254, 3278, 0 },
		{ 3255, 2016, 0 },
		{ 3142, 2979, 0 },
		{ 3257, 2485, 0 },
		{ 3201, 2573, 0 },
		{ 3260, 5083, 0 },
		{ 0, 0, 168 },
		{ 3196, 3683, 194 },
		{ 0, 0, 194 },
		{ 3220, 2290, 0 },
		{ 3242, 3139, 0 },
		{ 3259, 2398, 0 },
		{ 1361, 5275, 0 },
		{ 3254, 2980, 0 },
		{ 3250, 2779, 0 },
		{ 3253, 3315, 0 },
		{ 3220, 2296, 0 },
		{ 3220, 2306, 0 },
		{ 3242, 3221, 0 },
		{ 3220, 2308, 0 },
		{ 3260, 4842, 0 },
		{ 0, 0, 193 },
		{ 2203, 4384, 154 },
		{ 0, 0, 154 },
		{ 0, 0, 155 },
		{ 3242, 3224, 0 },
		{ 3201, 2576, 0 },
		{ 3257, 2522, 0 },
		{ 3244, 2696, 0 },
		{ 3242, 3157, 0 },
		{ 3206, 4660, 0 },
		{ 3250, 2785, 0 },
		{ 3253, 3302, 0 },
		{ 3220, 2321, 0 },
		{ 3220, 2327, 0 },
		{ 3222, 5102, 0 },
		{ 3222, 5098, 0 },
		{ 3142, 2997, 0 },
		{ 3201, 2631, 0 },
		{ 3142, 3015, 0 },
		{ 3255, 2065, 0 },
		{ 3142, 3037, 0 },
		{ 3253, 3368, 0 },
		{ 3250, 2793, 0 },
		{ 3142, 2992, 0 },
		{ 3220, 1814, 0 },
		{ 3242, 3152, 0 },
		{ 3259, 2421, 0 },
		{ 3260, 4799, 0 },
		{ 0, 0, 152 },
		{ 2749, 3293, 24 },
		{ 0, 0, 24 },
		{ 0, 0, 25 },
		{ 3242, 3161, 0 },
		{ 3044, 3060, 0 },
		{ 3142, 3010, 0 },
		{ 3201, 2603, 0 },
		{ 3205, 2, 0 },
		{ 3257, 2506, 0 },
		{ 3002, 2448, 0 },
		{ 3242, 3211, 0 },
		{ 3259, 2361, 0 },
		{ 3253, 3333, 0 },
		{ 3255, 2197, 0 },
		{ 3257, 2479, 0 },
		{ 3259, 2373, 0 },
		{ 3205, 5357, 0 },
		{ 3239, 3259, 0 },
		{ 3242, 3232, 0 },
		{ 3220, 2271, 0 },
		{ 3254, 3281, 0 },
		{ 3259, 2380, 0 },
		{ 3142, 3023, 0 },
		{ 3002, 2444, 0 },
		{ 3255, 2232, 0 },
		{ 3142, 2912, 0 },
		{ 3257, 2469, 0 },
		{ 3201, 2630, 0 },
		{ 3205, 7, 0 },
		{ 3222, 5105, 0 },
		{ 0, 0, 21 },
		{ 1444, 0, 1 },
		{ 1444, 0, 195 },
		{ 1444, 3071, 245 },
		{ 1659, 218, 245 },
		{ 1659, 400, 245 },
		{ 1659, 389, 245 },
		{ 1659, 520, 245 },
		{ 1659, 391, 245 },
		{ 1659, 408, 245 },
		{ 1659, 396, 245 },
		{ 1659, 478, 245 },
		{ 1659, 495, 245 },
		{ 1444, 0, 245 },
		{ 1456, 2824, 245 },
		{ 1444, 3102, 245 },
		{ 2749, 3290, 241 },
		{ 1659, 513, 245 },
		{ 1659, 537, 245 },
		{ 1659, 590, 245 },
		{ 1659, 0, 245 },
		{ 1659, 625, 245 },
		{ 1659, 612, 245 },
		{ 3257, 2486, 0 },
		{ 0, 0, 196 },
		{ 3201, 2626, 0 },
		{ 1659, 582, 0 },
		{ 1659, 0, 0 },
		{ 3205, 4332, 0 },
		{ 1659, 602, 0 },
		{ 1659, 619, 0 },
		{ 1659, 617, 0 },
		{ 1659, 625, 0 },
		{ 1659, 617, 0 },
		{ 1659, 621, 0 },
		{ 1659, 639, 0 },
		{ 1659, 621, 0 },
		{ 1659, 614, 0 },
		{ 1659, 606, 0 },
		{ 1659, 637, 0 },
		{ 3242, 3201, 0 },
		{ 3242, 3210, 0 },
		{ 1660, 644, 0 },
		{ 1660, 645, 0 },
		{ 1659, 656, 0 },
		{ 1659, 693, 0 },
		{ 1659, 684, 0 },
		{ 3257, 2510, 0 },
		{ 3239, 3255, 0 },
		{ 1659, 682, 0 },
		{ 1659, 726, 0 },
		{ 1659, 703, 0 },
		{ 1659, 704, 0 },
		{ 1659, 727, 0 },
		{ 1659, 754, 0 },
		{ 1659, 759, 0 },
		{ 1659, 753, 0 },
		{ 1659, 729, 0 },
		{ 1659, 721, 0 },
		{ 1659, 757, 0 },
		{ 1659, 772, 0 },
		{ 1659, 759, 0 },
		{ 3201, 2650, 0 },
		{ 3088, 3079, 0 },
		{ 1659, 775, 0 },
		{ 1659, 765, 0 },
		{ 1660, 767, 0 },
		{ 1659, 763, 0 },
		{ 1659, 793, 0 },
		{ 3250, 2733, 0 },
		{ 0, 0, 244 },
		{ 1659, 805, 0 },
		{ 1659, 838, 0 },
		{ 1659, 825, 0 },
		{ 1659, 830, 0 },
		{ 1659, 20, 0 },
		{ 1659, 73, 0 },
		{ 1659, 69, 0 },
		{ 1659, 58, 0 },
		{ 1659, 36, 0 },
		{ 1659, 40, 0 },
		{ 1659, 82, 0 },
		{ 1659, 0, 230 },
		{ 1659, 118, 0 },
		{ 3257, 2483, 0 },
		{ 3142, 3033, 0 },
		{ 1659, 76, 0 },
		{ 1659, 108, 0 },
		{ 1659, 110, 0 },
		{ 1659, 114, 0 },
		{ 1659, 114, 0 },
		{ -1539, 1535, 0 },
		{ 1660, 122, 0 },
		{ 1659, 155, 0 },
		{ 1659, 161, 0 },
		{ 1659, 153, 0 },
		{ 1659, 163, 0 },
		{ 1659, 167, 0 },
		{ 1659, 146, 0 },
		{ 1659, 163, 0 },
		{ 1659, 140, 0 },
		{ 1659, 131, 0 },
		{ 1659, 0, 229 },
		{ 1659, 138, 0 },
		{ 3244, 2708, 0 },
		{ 3201, 2605, 0 },
		{ 1659, 157, 0 },
		{ 1659, 168, 0 },
		{ 1659, 191, 0 },
		{ 1659, 0, 243 },
		{ 1659, 191, 0 },
		{ 0, 0, 231 },
		{ 1659, 185, 0 },
		{ 1661, 4, -4 },
		{ 1659, 246, 0 },
		{ 1659, 258, 0 },
		{ 1659, 255, 0 },
		{ 1659, 260, 0 },
		{ 1659, 264, 0 },
		{ 1659, 277, 0 },
		{ 1659, 248, 0 },
		{ 1659, 253, 0 },
		{ 1659, 244, 0 },
		{ 3242, 3110, 0 },
		{ 3242, 3127, 0 },
		{ 1659, 0, 233 },
		{ 1659, 285, 234 },
		{ 1659, 258, 0 },
		{ 1659, 266, 0 },
		{ 1659, 307, 0 },
		{ 1559, 3869, 0 },
		{ 3205, 4599, 0 },
		{ 2244, 5045, 220 },
		{ 1659, 310, 0 },
		{ 1659, 314, 0 },
		{ 1659, 344, 0 },
		{ 1659, 346, 0 },
		{ 1659, 381, 0 },
		{ 1659, 382, 0 },
		{ 1659, 370, 0 },
		{ 1659, 375, 0 },
		{ 1659, 357, 0 },
		{ 1659, 364, 0 },
		{ 1660, 350, 0 },
		{ 3206, 4716, 0 },
		{ 3205, 5378, 236 },
		{ 1659, 353, 0 },
		{ 1659, 365, 0 },
		{ 1659, 347, 0 },
		{ 1659, 363, 0 },
		{ 0, 0, 200 },
		{ 1661, 31, -7 },
		{ 1661, 206, -10 },
		{ 1661, 321, -13 },
		{ 1661, 349, -16 },
		{ 1661, 351, -19 },
		{ 1661, 435, -22 },
		{ 1659, 413, 0 },
		{ 1659, 426, 0 },
		{ 1659, 399, 0 },
		{ 1659, 0, 218 },
		{ 1659, 0, 232 },
		{ 3250, 2788, 0 },
		{ 1659, 430, 0 },
		{ 1659, 421, 0 },
		{ 1659, 454, 0 },
		{ 1660, 463, 0 },
		{ 1596, 3907, 0 },
		{ 3205, 4609, 0 },
		{ 2244, 5061, 221 },
		{ 1599, 3909, 0 },
		{ 3205, 4595, 0 },
		{ 2244, 5042, 222 },
		{ 1602, 3910, 0 },
		{ 3205, 4603, 0 },
		{ 2244, 5066, 225 },
		{ 1605, 3911, 0 },
		{ 3205, 4639, 0 },
		{ 2244, 5057, 226 },
		{ 1608, 3912, 0 },
		{ 3205, 4593, 0 },
		{ 2244, 5068, 227 },
		{ 1611, 3913, 0 },
		{ 3205, 4597, 0 },
		{ 2244, 5048, 228 },
		{ 1659, 508, 0 },
		{ 1661, 463, -25 },
		{ 1659, 481, 0 },
		{ 3253, 3353, 0 },
		{ 1659, 462, 0 },
		{ 1659, 507, 0 },
		{ 1659, 472, 0 },
		{ 1659, 486, 0 },
		{ 0, 0, 202 },
		{ 0, 0, 204 },
		{ 0, 0, 210 },
		{ 0, 0, 212 },
		{ 0, 0, 214 },
		{ 0, 0, 216 },
		{ 1661, 465, -28 },
		{ 1629, 3922, 0 },
		{ 3205, 4607, 0 },
		{ 2244, 5073, 224 },
		{ 1659, 0, 217 },
		{ 3220, 2288, 0 },
		{ 1659, 474, 0 },
		{ 1659, 490, 0 },
		{ 1660, 497, 0 },
		{ 1659, 494, 0 },
		{ 1638, 3864, 0 },
		{ 3205, 4601, 0 },
		{ 2244, 5030, 223 },
		{ 0, 0, 208 },
		{ 3220, 2310, 0 },
		{ 1659, 4, 239 },
		{ 1660, 501, 0 },
		{ 1659, 4, 242 },
		{ 1659, 544, 0 },
		{ 0, 0, 206 },
		{ 3222, 5103, 0 },
		{ 3222, 5104, 0 },
		{ 1659, 532, 0 },
		{ 0, 0, 240 },
		{ 1659, 557, 0 },
		{ 3222, 5099, 0 },
		{ 0, 0, 238 },
		{ 1659, 576, 0 },
		{ 1659, 581, 0 },
		{ 0, 0, 237 },
		{ 1659, 586, 0 },
		{ 1659, 577, 0 },
		{ 1660, 579, 235 },
		{ 1661, 1003, 0 },
		{ 1662, 736, -1 },
		{ 1663, 3881, 0 },
		{ 3205, 4683, 0 },
		{ 2244, 5071, 219 },
		{ 0, 0, 198 },
		{ 2203, 4401, 288 },
		{ 0, 0, 288 },
		{ 3242, 3158, 0 },
		{ 3201, 2653, 0 },
		{ 3257, 2465, 0 },
		{ 3244, 2690, 0 },
		{ 3242, 3173, 0 },
		{ 3206, 4712, 0 },
		{ 3250, 2789, 0 },
		{ 3253, 3309, 0 },
		{ 3220, 2312, 0 },
		{ 3220, 2320, 0 },
		{ 3222, 5080, 0 },
		{ 3222, 5081, 0 },
		{ 3142, 3024, 0 },
		{ 3201, 2601, 0 },
		{ 3142, 3031, 0 },
		{ 3255, 2156, 0 },
		{ 3142, 3036, 0 },
		{ 3253, 3306, 0 },
		{ 3250, 2775, 0 },
		{ 3142, 2895, 0 },
		{ 3220, 1820, 0 },
		{ 3242, 3237, 0 },
		{ 3259, 2370, 0 },
		{ 3260, 4867, 0 },
		{ 0, 0, 287 },
		{ 2203, 4388, 290 },
		{ 0, 0, 290 },
		{ 0, 0, 291 },
		{ 3242, 3111, 0 },
		{ 3201, 2610, 0 },
		{ 3257, 2489, 0 },
		{ 3244, 2678, 0 },
		{ 3242, 3146, 0 },
		{ 3206, 4690, 0 },
		{ 3250, 2794, 0 },
		{ 3253, 3328, 0 },
		{ 3220, 2329, 0 },
		{ 3220, 2331, 0 },
		{ 3222, 5082, 0 },
		{ 3222, 5094, 0 },
		{ 3254, 3270, 0 },
		{ 3259, 2379, 0 },
		{ 3257, 2511, 0 },
		{ 3220, 2332, 0 },
		{ 3220, 2333, 0 },
		{ 3257, 2533, 0 },
		{ 3224, 1709, 0 },
		{ 3242, 3185, 0 },
		{ 3259, 2393, 0 },
		{ 3260, 5085, 0 },
		{ 0, 0, 289 },
		{ 2203, 4397, 293 },
		{ 0, 0, 293 },
		{ 0, 0, 294 },
		{ 3242, 3190, 0 },
		{ 3201, 2583, 0 },
		{ 3257, 2476, 0 },
		{ 3244, 2709, 0 },
		{ 3242, 3218, 0 },
		{ 3206, 4718, 0 },
		{ 3250, 2743, 0 },
		{ 3253, 3308, 0 },
		{ 3220, 2344, 0 },
		{ 3220, 2347, 0 },
		{ 3222, 5100, 0 },
		{ 3222, 5101, 0 },
		{ 3244, 2692, 0 },
		{ 3247, 1884, 0 },
		{ 3255, 2247, 0 },
		{ 3253, 3352, 0 },
		{ 3255, 1918, 0 },
		{ 3257, 2494, 0 },
		{ 3259, 2367, 0 },
		{ 3260, 5017, 0 },
		{ 0, 0, 292 },
		{ 2203, 4399, 296 },
		{ 0, 0, 296 },
		{ 0, 0, 297 },
		{ 3242, 3134, 0 },
		{ 3201, 2628, 0 },
		{ 3257, 2507, 0 },
		{ 3244, 2698, 0 },
		{ 3242, 3147, 0 },
		{ 3206, 4688, 0 },
		{ 3250, 2745, 0 },
		{ 3253, 3330, 0 },
		{ 3220, 2274, 0 },
		{ 3220, 2280, 0 },
		{ 3222, 5096, 0 },
		{ 3222, 5097, 0 },
		{ 3242, 3160, 0 },
		{ 3224, 1711, 0 },
		{ 3253, 3364, 0 },
		{ 3250, 2784, 0 },
		{ 3247, 1883, 0 },
		{ 3253, 3303, 0 },
		{ 3255, 2100, 0 },
		{ 3257, 2528, 0 },
		{ 3259, 2382, 0 },
		{ 3260, 4949, 0 },
		{ 0, 0, 295 },
		{ 2203, 4377, 299 },
		{ 0, 0, 299 },
		{ 0, 0, 300 },
		{ 3242, 3182, 0 },
		{ 3201, 2585, 0 },
		{ 3257, 2463, 0 },
		{ 3244, 2711, 0 },
		{ 3242, 3193, 0 },
		{ 3206, 4715, 0 },
		{ 3250, 2737, 0 },
		{ 3253, 3348, 0 },
		{ 3220, 2293, 0 },
		{ 3220, 2294, 0 },
		{ 3222, 5106, 0 },
		{ 3222, 5079, 0 },
		{ 3257, 2471, 0 },
		{ 3002, 2435, 0 },
		{ 3255, 2102, 0 },
		{ 3142, 2910, 0 },
		{ 3244, 2700, 0 },
		{ 3142, 2972, 0 },
		{ 3220, 2297, 0 },
		{ 3242, 3239, 0 },
		{ 3259, 2404, 0 },
		{ 3260, 5056, 0 },
		{ 0, 0, 298 },
		{ 3075, 4915, 174 },
		{ 0, 0, 174 },
		{ 0, 0, 175 },
		{ 3088, 3097, 0 },
		{ 3255, 2119, 0 },
		{ 3242, 3132, 0 },
		{ 3259, 2419, 0 },
		{ 1803, 5270, 0 },
		{ 3242, 2822, 0 },
		{ 3224, 1671, 0 },
		{ 3242, 3140, 0 },
		{ 3259, 2360, 0 },
		{ 2852, 1664, 0 },
		{ 3255, 2160, 0 },
		{ 3249, 3039, 0 },
		{ 3142, 3025, 0 },
		{ 3201, 2564, 0 },
		{ 3044, 3062, 0 },
		{ 1814, 5126, 0 },
		{ 3242, 2816, 0 },
		{ 3250, 2765, 0 },
		{ 3220, 2314, 0 },
		{ 3242, 3174, 0 },
		{ 1819, 5204, 0 },
		{ 3252, 2774, 0 },
		{ 3247, 1850, 0 },
		{ 3201, 2578, 0 },
		{ 3254, 3274, 0 },
		{ 3255, 2204, 0 },
		{ 3142, 2967, 0 },
		{ 3257, 2519, 0 },
		{ 3201, 2591, 0 },
		{ 3260, 4959, 0 },
		{ 0, 0, 172 },
		{ 2203, 4393, 281 },
		{ 0, 0, 281 },
		{ 3242, 3196, 0 },
		{ 3201, 2593, 0 },
		{ 3257, 2521, 0 },
		{ 3244, 2702, 0 },
		{ 3242, 3214, 0 },
		{ 3206, 4720, 0 },
		{ 3250, 2735, 0 },
		{ 3253, 3318, 0 },
		{ 3220, 2322, 0 },
		{ 3220, 2323, 0 },
		{ 3222, 5088, 0 },
		{ 3222, 5090, 0 },
		{ 3239, 3262, 0 },
		{ 3142, 3019, 0 },
		{ 3220, 2325, 0 },
		{ 3002, 2436, 0 },
		{ 3250, 2759, 0 },
		{ 3253, 3360, 0 },
		{ 2852, 1635, 0 },
		{ 3260, 4943, 0 },
		{ 0, 0, 279 },
		{ 1867, 0, 1 },
		{ 2026, 3135, 396 },
		{ 3242, 3112, 396 },
		{ 3253, 3170, 396 },
		{ 3239, 2471, 396 },
		{ 1867, 0, 363 },
		{ 1867, 3105, 396 },
		{ 3249, 1847, 396 },
		{ 3010, 4736, 396 },
		{ 2182, 4089, 396 },
		{ 3196, 3664, 396 },
		{ 2182, 4092, 396 },
		{ 2167, 4456, 396 },
		{ 3259, 2264, 396 },
		{ 1867, 0, 396 },
		{ 2749, 3295, 394 },
		{ 3253, 3079, 396 },
		{ 3253, 3339, 396 },
		{ 0, 0, 396 },
		{ 3257, 2482, 0 },
		{ -1872, 8, 353 },
		{ -1873, 5176, 0 },
		{ 3201, 2636, 0 },
		{ 0, 0, 359 },
		{ 0, 0, 360 },
		{ 3250, 2791, 0 },
		{ 3142, 2913, 0 },
		{ 3242, 3156, 0 },
		{ 0, 0, 364 },
		{ 3201, 2640, 0 },
		{ 3259, 2394, 0 },
		{ 3142, 2975, 0 },
		{ 2135, 3518, 0 },
		{ 3191, 4038, 0 },
		{ 3200, 3765, 0 },
		{ 2075, 3643, 0 },
		{ 3191, 4042, 0 },
		{ 3247, 1848, 0 },
		{ 3220, 2334, 0 },
		{ 3201, 2560, 0 },
		{ 3255, 1947, 0 },
		{ 3259, 2410, 0 },
		{ 3257, 2495, 0 },
		{ 3149, 5162, 0 },
		{ 3257, 2497, 0 },
		{ 3220, 2342, 0 },
		{ 3255, 1984, 0 },
		{ 3201, 2589, 0 },
		{ 3239, 3251, 0 },
		{ 3259, 2350, 0 },
		{ 3250, 2782, 0 },
		{ 2203, 4396, 0 },
		{ 2135, 3418, 0 },
		{ 2135, 3419, 0 },
		{ 2167, 4498, 0 },
		{ 2097, 4217, 0 },
		{ 3242, 3195, 0 },
		{ 3220, 2345, 0 },
		{ 3239, 3241, 0 },
		{ 3247, 1849, 0 },
		{ 3242, 3208, 0 },
		{ 3250, 2786, 0 },
		{ 0, 5385, 356 },
		{ 3244, 2694, 0 },
		{ 3242, 3212, 0 },
		{ 2182, 4074, 0 },
		{ 3255, 2017, 0 },
		{ 0, 0, 395 },
		{ 3242, 3216, 0 },
		{ 3239, 3253, 0 },
		{ 2167, 4512, 0 },
		{ 2113, 3824, 0 },
		{ 3191, 4020, 0 },
		{ 3166, 3946, 0 },
		{ 2135, 3432, 0 },
		{ 0, 0, 384 },
		{ 3206, 4708, 0 },
		{ 3257, 2517, 0 },
		{ 3259, 2365, 0 },
		{ 3201, 2608, 0 },
		{ -1949, 1170, 0 },
		{ 0, 0, 355 },
		{ 3242, 3228, 0 },
		{ 0, 0, 383 },
		{ 3002, 2447, 0 },
		{ 3142, 2971, 0 },
		{ 3201, 2614, 0 },
		{ 1964, 5120, 0 },
		{ 3179, 4137, 0 },
		{ 3116, 4315, 0 },
		{ 3166, 3867, 0 },
		{ 2135, 3441, 0 },
		{ 3191, 4037, 0 },
		{ 3257, 2526, 0 },
		{ 3244, 2680, 0 },
		{ 3201, 2625, 0 },
		{ 3255, 2067, 0 },
		{ 0, 0, 385 },
		{ 3206, 4670, 362 },
		{ 3255, 2071, 0 },
		{ 3254, 3276, 0 },
		{ 3255, 2080, 0 },
		{ 0, 0, 388 },
		{ 0, 0, 389 },
		{ 1969, 0, -80 },
		{ 2148, 3568, 0 },
		{ 2182, 4104, 0 },
		{ 3191, 3992, 0 },
		{ 2167, 4442, 0 },
		{ 3142, 3008, 0 },
		{ 0, 0, 387 },
		{ 0, 0, 393 },
		{ 0, 5122, 0 },
		{ 3250, 2772, 0 },
		{ 3220, 2282, 0 },
		{ 3253, 3310, 0 },
		{ 2203, 4378, 0 },
		{ 3205, 4675, 0 },
		{ 2244, 5070, 378 },
		{ 2167, 4449, 0 },
		{ 3010, 4722, 0 },
		{ 3166, 3885, 0 },
		{ 3166, 3886, 0 },
		{ 3201, 2632, 0 },
		{ 0, 0, 390 },
		{ 0, 0, 391 },
		{ 3253, 3316, 0 },
		{ 2250, 5167, 0 },
		{ 3250, 2781, 0 },
		{ 3242, 3144, 0 },
		{ 0, 0, 368 },
		{ 1992, 0, -35 },
		{ 1994, 0, -38 },
		{ 2182, 4054, 0 },
		{ 3206, 4703, 0 },
		{ 0, 0, 386 },
		{ 3220, 2283, 0 },
		{ 0, 0, 361 },
		{ 2203, 4389, 0 },
		{ 3201, 2638, 0 },
		{ 3205, 4613, 0 },
		{ 2244, 5047, 379 },
		{ 3205, 4637, 0 },
		{ 2244, 5051, 380 },
		{ 3010, 4732, 0 },
		{ 2004, 0, -68 },
		{ 3220, 2284, 0 },
		{ 3242, 3154, 0 },
		{ 3242, 3155, 0 },
		{ 0, 0, 370 },
		{ 0, 0, 372 },
		{ 2009, 0, -74 },
		{ 3205, 4679, 0 },
		{ 2244, 5075, 382 },
		{ 0, 0, 358 },
		{ 3201, 2646, 0 },
		{ 3259, 2383, 0 },
		{ 3205, 4685, 0 },
		{ 2244, 5046, 381 },
		{ 0, 0, 376 },
		{ 3257, 2478, 0 },
		{ 3253, 3363, 0 },
		{ 0, 0, 374 },
		{ 3244, 2686, 0 },
		{ 3255, 2084, 0 },
		{ 3242, 3162, 0 },
		{ 3142, 2827, 0 },
		{ 0, 0, 392 },
		{ 3257, 2480, 0 },
		{ 3201, 2563, 0 },
		{ 2023, 0, -41 },
		{ 3205, 4605, 0 },
		{ 2244, 5074, 377 },
		{ 0, 0, 366 },
		{ 1867, 3161, 396 },
		{ 2030, 2820, 396 },
		{ -2028, 22, 353 },
		{ -2029, 1, 0 },
		{ 3205, 5165, 0 },
		{ 3149, 5147, 0 },
		{ 0, 0, 354 },
		{ 3149, 5163, 0 },
		{ -2034, 20, 0 },
		{ -2035, 5179, 0 },
		{ 2038, 2, 356 },
		{ 3149, 5138, 0 },
		{ 3205, 5340, 0 },
		{ 0, 0, 357 },
		{ 2056, 0, 1 },
		{ 2252, 3150, 352 },
		{ 3242, 3186, 352 },
		{ 2056, 0, 306 },
		{ 2056, 3039, 352 },
		{ 3191, 4033, 352 },
		{ 2056, 0, 309 },
		{ 3247, 1715, 352 },
		{ 3010, 4730, 352 },
		{ 2182, 4082, 352 },
		{ 3196, 3608, 352 },
		{ 2182, 4084, 352 },
		{ 2167, 4494, 352 },
		{ 3253, 3345, 352 },
		{ 3259, 2270, 352 },
		{ 2056, 0, 352 },
		{ 2749, 3292, 349 },
		{ 3253, 3356, 352 },
		{ 3239, 3258, 352 },
		{ 3010, 4728, 352 },
		{ 3253, 1828, 352 },
		{ 0, 0, 352 },
		{ 3257, 2491, 0 },
		{ -2063, 19, 301 },
		{ -2064, 5181, 0 },
		{ 3201, 2586, 0 },
		{ 0, 0, 307 },
		{ 3201, 2588, 0 },
		{ 3257, 2492, 0 },
		{ 3259, 2406, 0 },
		{ 2135, 3508, 0 },
		{ 3191, 3998, 0 },
		{ 3200, 3774, 0 },
		{ 3179, 4151, 0 },
		{ 0, 3587, 0 },
		{ 0, 3640, 0 },
		{ 3191, 4003, 0 },
		{ 3250, 2777, 0 },
		{ 3247, 1748, 0 },
		{ 3220, 2299, 0 },
		{ 3201, 2602, 0 },
		{ 3242, 3219, 0 },
		{ 3242, 3220, 0 },
		{ 3257, 2499, 0 },
		{ 2250, 5169, 0 },
		{ 3257, 2501, 0 },
		{ 3149, 5151, 0 },
		{ 3257, 2504, 0 },
		{ 3239, 3257, 0 },
		{ 3002, 2458, 0 },
		{ 3259, 2412, 0 },
		{ 2203, 4383, 0 },
		{ 2135, 3521, 0 },
		{ 2135, 3522, 0 },
		{ 3116, 4324, 0 },
		{ 3116, 4325, 0 },
		{ 2167, 4531, 0 },
		{ 0, 4220, 0 },
		{ 3220, 2301, 0 },
		{ 3242, 3234, 0 },
		{ 3220, 2303, 0 },
		{ 3239, 3248, 0 },
		{ 3201, 2615, 0 },
		{ 3220, 2305, 0 },
		{ 3250, 2727, 0 },
		{ 0, 0, 351 },
		{ 3250, 2732, 0 },
		{ 0, 0, 303 },
		{ 3244, 2684, 0 },
		{ 0, 0, 348 },
		{ 3247, 1845, 0 },
		{ 3242, 3130, 0 },
		{ 2167, 4445, 0 },
		{ 0, 3796, 0 },
		{ 3191, 4031, 0 },
		{ 2116, 4177, 0 },
		{ 0, 4178, 0 },
		{ 3166, 3884, 0 },
		{ 2135, 3417, 0 },
		{ 3242, 3131, 0 },
		{ 0, 0, 341 },
		{ 3206, 4707, 0 },
		{ 3257, 2516, 0 },
		{ 3255, 2170, 0 },
		{ 3255, 2188, 0 },
		{ 3247, 1846, 0 },
		{ -2143, 1245, 0 },
		{ 3242, 3142, 0 },
		{ 3250, 2761, 0 },
		{ 3201, 2635, 0 },
		{ 3179, 4135, 0 },
		{ 3116, 4357, 0 },
		{ 3166, 3895, 0 },
		{ 3116, 4359, 0 },
		{ 3116, 4360, 0 },
		{ 0, 3427, 0 },
		{ 3191, 3991, 0 },
		{ 0, 0, 340 },
		{ 3257, 2523, 0 },
		{ 3244, 2704, 0 },
		{ 3142, 2998, 0 },
		{ 0, 0, 347 },
		{ 3253, 3336, 0 },
		{ 0, 0, 342 },
		{ 0, 0, 305 },
		{ 3253, 3338, 0 },
		{ 3255, 2210, 0 },
		{ 2160, 0, -65 },
		{ 0, 3564, 0 },
		{ 2182, 4090, 0 },
		{ 2151, 3551, 0 },
		{ 2148, 3550, 0 },
		{ 3191, 4002, 0 },
		{ 2167, 4479, 0 },
		{ 3142, 3001, 0 },
		{ 0, 0, 344 },
		{ 3254, 3287, 0 },
		{ 3255, 2211, 0 },
		{ 3255, 2215, 0 },
		{ 2203, 4391, 0 },
		{ 3205, 4687, 0 },
		{ 2244, 5072, 331 },
		{ 2167, 4485, 0 },
		{ 3010, 4723, 0 },
		{ 2167, 4486, 0 },
		{ 2167, 4487, 0 },
		{ 2167, 4488, 0 },
		{ 0, 4489, 0 },
		{ 3166, 3917, 0 },
		{ 3166, 3918, 0 },
		{ 3201, 2649, 0 },
		{ 3253, 3359, 0 },
		{ 3142, 3011, 0 },
		{ 3142, 3013, 0 },
		{ 3242, 3163, 0 },
		{ 0, 0, 313 },
		{ 2189, 0, -44 },
		{ 2191, 0, -47 },
		{ 2193, 0, -53 },
		{ 2195, 0, -56 },
		{ 2197, 0, -59 },
		{ 2199, 0, -62 },
		{ 0, 4105, 0 },
		{ 3206, 4713, 0 },
		{ 0, 0, 343 },
		{ 3250, 2783, 0 },
		{ 3257, 2539, 0 },
		{ 3257, 2461, 0 },
		{ 3201, 2554, 0 },
		{ 3205, 4641, 0 },
		{ 2244, 5049, 332 },
		{ 3205, 4643, 0 },
		{ 2244, 5052, 333 },
		{ 3205, 4667, 0 },
		{ 2244, 5059, 336 },
		{ 3205, 4669, 0 },
		{ 2244, 5065, 337 },
		{ 3205, 4671, 0 },
		{ 2244, 5067, 338 },
		{ 3205, 4673, 0 },
		{ 2244, 5069, 339 },
		{ 3010, 4725, 0 },
		{ 2214, 0, -71 },
		{ 0, 4379, 0 },
		{ 3201, 2555, 0 },
		{ 3201, 2556, 0 },
		{ 3242, 3178, 0 },
		{ 0, 0, 315 },
		{ 0, 0, 317 },
		{ 0, 0, 323 },
		{ 0, 0, 325 },
		{ 0, 0, 327 },
		{ 0, 0, 329 },
		{ 2220, 0, -77 },
		{ 3205, 4681, 0 },
		{ 2244, 5044, 335 },
		{ 3242, 3180, 0 },
		{ 3253, 3314, 0 },
		{ 3204, 3548, 346 },
		{ 3259, 2375, 0 },
		{ 3205, 4699, 0 },
		{ 2244, 5050, 334 },
		{ 0, 0, 321 },
		{ 3201, 2559, 0 },
		{ 3259, 2376, 0 },
		{ 0, 0, 308 },
		{ 3253, 3325, 0 },
		{ 0, 0, 319 },
		{ 3257, 2466, 0 },
		{ 2852, 1620, 0 },
		{ 3255, 2239, 0 },
		{ 3244, 2705, 0 },
		{ 3075, 4769, 0 },
		{ 3142, 2908, 0 },
		{ 3242, 3199, 0 },
		{ 3250, 2734, 0 },
		{ 3257, 2473, 0 },
		{ 0, 0, 345 },
		{ 3044, 3063, 0 },
		{ 3201, 2577, 0 },
		{ 3257, 2474, 0 },
		{ 2243, 0, -50 },
		{ 3259, 2381, 0 },
		{ 3205, 4611, 0 },
		{ 0, 5043, 330 },
		{ 3142, 2969, 0 },
		{ 0, 0, 311 },
		{ 3255, 2242, 0 },
		{ 3249, 3048, 0 },
		{ 3244, 2682, 0 },
		{ 0, 5168, 0 },
		{ 0, 0, 350 },
		{ 2056, 3136, 352 },
		{ 2256, 2822, 352 },
		{ -2254, 17, 301 },
		{ -2255, 5178, 0 },
		{ 3205, 5164, 0 },
		{ 3149, 5144, 0 },
		{ 0, 0, 302 },
		{ 3149, 5146, 0 },
		{ -2260, 5380, 0 },
		{ -2261, 5183, 0 },
		{ 2264, 0, 303 },
		{ 3149, 5148, 0 },
		{ 3205, 5347, 0 },
		{ 0, 0, 304 },
		{ 2652, 4568, 398 },
		{ 0, 0, 398 },
		{ 3242, 3227, 0 },
		{ 3088, 3087, 0 },
		{ 3253, 3307, 0 },
		{ 3247, 1852, 0 },
		{ 3250, 2767, 0 },
		{ 3255, 2255, 0 },
		{ 3205, 5356, 0 },
		{ 3259, 2392, 0 },
		{ 3247, 1881, 0 },
		{ 3201, 2598, 0 },
		{ 2279, 5226, 0 },
		{ 3205, 2237, 0 },
		{ 3253, 3322, 0 },
		{ 3259, 2396, 0 },
		{ 3253, 3327, 0 },
		{ 3244, 2701, 0 },
		{ 3242, 3121, 0 },
		{ 3255, 1949, 0 },
		{ 3242, 3128, 0 },
		{ 3259, 2400, 0 },
		{ 3220, 2338, 0 },
		{ 3260, 4947, 0 },
		{ 0, 0, 397 },
		{ 3149, 5136, 451 },
		{ 0, 0, 403 },
		{ 0, 0, 405 },
		{ 2312, 908, 441 },
		{ 2511, 922, 441 },
		{ 2537, 920, 441 },
		{ 2471, 921, 441 },
		{ 2313, 935, 441 },
		{ 2311, 925, 441 },
		{ 2537, 919, 441 },
		{ 2335, 935, 441 },
		{ 2537, 939, 441 },
		{ 2506, 937, 441 },
		{ 2511, 938, 441 },
		{ 2441, 947, 441 },
		{ 2309, 965, 441 },
		{ 3242, 1958, 440 },
		{ 2344, 2982, 451 },
		{ 2577, 937, 441 },
		{ 2335, 945, 441 },
		{ 2511, 951, 441 },
		{ 2349, 952, 441 },
		{ 2511, 949, 441 },
		{ 3242, 3172, 451 },
		{ -2315, 5384, 399 },
		{ -2316, 5174, 0 },
		{ 2577, 946, 441 },
		{ 2586, 446, 441 },
		{ 2577, 949, 441 },
		{ 2404, 947, 441 },
		{ 2511, 955, 441 },
		{ 2518, 950, 441 },
		{ 2511, 957, 441 },
		{ 2441, 966, 441 },
		{ 2404, 963, 441 },
		{ 2537, 1045, 441 },
		{ 2471, 958, 441 },
		{ 2441, 980, 441 },
		{ 2537, 964, 441 },
		{ 2309, 957, 441 },
		{ 2556, 964, 441 },
		{ 2346, 978, 441 },
		{ 2309, 989, 441 },
		{ 2518, 999, 441 },
		{ 2309, 1009, 441 },
		{ 2552, 1030, 441 },
		{ 2511, 1036, 441 },
		{ 2577, 1268, 441 },
		{ 2552, 1032, 441 },
		{ 2409, 1046, 441 },
		{ 2586, 662, 441 },
		{ 3242, 1991, 437 },
		{ 2379, 1633, 0 },
		{ 3242, 2001, 438 },
		{ 2552, 1046, 441 },
		{ 2552, 1047, 441 },
		{ 3201, 2617, 0 },
		{ 3149, 5161, 0 },
		{ 2309, 1060, 441 },
		{ 3257, 2302, 0 },
		{ 2506, 1085, 441 },
		{ 2391, 1070, 441 },
		{ 2552, 1078, 441 },
		{ 2556, 1073, 441 },
		{ 2556, 1075, 441 },
		{ 2409, 1120, 441 },
		{ 2506, 1128, 441 },
		{ 2506, 1129, 441 },
		{ 2471, 1113, 441 },
		{ 2506, 1131, 441 },
		{ 2537, 1119, 441 },
		{ 2471, 1116, 441 },
		{ 2506, 1160, 441 },
		{ 2441, 1165, 441 },
		{ 2537, 1149, 441 },
		{ 2586, 668, 441 },
		{ 2586, 670, 441 },
		{ 2545, 1151, 441 },
		{ 2545, 1188, 441 },
		{ 2506, 1203, 441 },
		{ 2493, 1205, 441 },
		{ 2391, 1189, 441 },
		{ 2518, 1196, 441 },
		{ 2441, 1211, 441 },
		{ 2493, 1209, 441 },
		{ 3252, 2766, 0 },
		{ 2413, 1666, 0 },
		{ 2379, 0, 0 },
		{ 3151, 2886, 439 },
		{ 2415, 1709, 0 },
		{ 2309, 889, 441 },
		{ 3239, 3247, 0 },
		{ 0, 0, 401 },
		{ 2506, 1236, 441 },
		{ 3088, 3083, 0 },
		{ 2586, 672, 441 },
		{ 2409, 1231, 441 },
		{ 2556, 1225, 441 },
		{ 2586, 674, 441 },
		{ 2511, 1268, 441 },
		{ 2309, 1253, 441 },
		{ 2537, 1257, 441 },
		{ 2407, 1273, 441 },
		{ 2506, 1273, 441 },
		{ 2586, 10, 441 },
		{ 2556, 1260, 441 },
		{ 2511, 1273, 441 },
		{ 2586, 117, 441 },
		{ 2556, 1263, 441 },
		{ 2441, 1283, 441 },
		{ 3205, 2531, 0 },
		{ 3255, 2085, 0 },
		{ 2545, 1266, 441 },
		{ 2309, 1270, 441 },
		{ 2537, 1269, 441 },
		{ 2586, 120, 441 },
		{ 2309, 1285, 441 },
		{ 2556, 1269, 441 },
		{ 2309, 1278, 441 },
		{ 2309, 1268, 441 },
		{ 3142, 2819, 0 },
		{ 2413, 0, 0 },
		{ 3151, 2973, 437 },
		{ 2415, 0, 0 },
		{ 3151, 2983, 438 },
		{ 3149, 5154, 0 },
		{ 0, 0, 443 },
		{ 2537, 1274, 441 },
		{ 2450, 5307, 0 },
		{ 3250, 2370, 0 },
		{ 2441, 1292, 441 },
		{ 2586, 212, 441 },
		{ 3220, 2202, 0 },
		{ 2609, 6, 441 },
		{ 2545, 1277, 441 },
		{ 2441, 1296, 441 },
		{ 2511, 1289, 441 },
		{ 2556, 1279, 441 },
		{ 2537, 1282, 441 },
		{ 3205, 2249, 0 },
		{ 2586, 214, 441 },
		{ 2471, 1279, 441 },
		{ 3257, 2274, 0 },
		{ 2511, 1293, 441 },
		{ 2556, 1283, 441 },
		{ 3201, 2580, 0 },
		{ 3201, 2582, 0 },
		{ 3259, 2399, 0 },
		{ 2518, 1289, 441 },
		{ 2537, 1287, 441 },
		{ 2309, 1305, 441 },
		{ 3205, 3610, 0 },
		{ 2506, 1302, 441 },
		{ 2506, 1303, 441 },
		{ 2586, 219, 441 },
		{ 2511, 1301, 441 },
		{ 3250, 2769, 0 },
		{ 2481, 1867, 0 },
		{ 2586, 324, 441 },
		{ 3252, 2459, 0 },
		{ 3142, 2821, 0 },
		{ 2556, 1291, 441 },
		{ 3220, 2213, 0 },
		{ 3255, 2145, 0 },
		{ 3260, 4945, 0 },
		{ 3205, 5237, 409 },
		{ 2577, 1299, 441 },
		{ 2556, 1293, 441 },
		{ 2577, 1301, 441 },
		{ 2511, 1310, 441 },
		{ 2586, 330, 441 },
		{ 3257, 2532, 0 },
		{ 3252, 2764, 0 },
		{ 2511, 1308, 441 },
		{ 3088, 3096, 0 },
		{ 2518, 1303, 441 },
		{ 2511, 1311, 441 },
		{ 3142, 2987, 0 },
		{ 3142, 2988, 0 },
		{ 3242, 3194, 0 },
		{ 2309, 1299, 441 },
		{ 2511, 1313, 441 },
		{ 2556, 1303, 441 },
		{ 3257, 2467, 0 },
		{ 2586, 332, 441 },
		{ 2586, 345, 441 },
		{ 3259, 2230, 0 },
		{ 2552, 1312, 441 },
		{ 3242, 3213, 0 },
		{ 2481, 33, 442 },
		{ 2547, 36, 442 },
		{ 3257, 2289, 0 },
		{ 3191, 4043, 0 },
		{ 3142, 3018, 0 },
		{ 3244, 2707, 0 },
		{ 2511, 1329, 441 },
		{ 3255, 2006, 0 },
		{ 3253, 3324, 0 },
		{ 2609, 210, 441 },
		{ 2518, 1324, 441 },
		{ 2506, 1335, 441 },
		{ 2518, 1326, 441 },
		{ 2309, 1338, 441 },
		{ 3205, 2247, 0 },
		{ 3002, 2454, 0 },
		{ 3259, 2417, 0 },
		{ 2552, 1329, 441 },
		{ 2531, 5215, 0 },
		{ 2552, 1330, 441 },
		{ 2518, 1356, 441 },
		{ 3255, 2104, 0 },
		{ 3255, 2112, 0 },
		{ 3242, 3125, 0 },
		{ 2506, 1367, 441 },
		{ 2552, 1359, 441 },
		{ 2309, 1369, 441 },
		{ 3259, 2366, 0 },
		{ 3257, 2183, 0 },
		{ 3205, 2535, 0 },
		{ 3242, 3135, 0 },
		{ 2309, 1367, 441 },
		{ 3260, 4939, 0 },
		{ 2481, 27, 442 },
		{ 3088, 3085, 0 },
		{ 3196, 3673, 0 },
		{ 3255, 2173, 0 },
		{ 3142, 2985, 0 },
		{ 2309, 1390, 441 },
		{ 3253, 3312, 0 },
		{ 3255, 2194, 0 },
		{ 3260, 5027, 0 },
		{ 0, 0, 427 },
		{ 2537, 1388, 441 },
		{ 2552, 1393, 441 },
		{ 2552, 1394, 441 },
		{ 2586, 437, 441 },
		{ 3257, 2535, 0 },
		{ 3247, 1746, 0 },
		{ 3257, 2541, 0 },
		{ 2538, 1412, 441 },
		{ 3205, 2245, 0 },
		{ 2586, 439, 441 },
		{ 2552, 1407, 441 },
		{ 2566, 5144, 0 },
		{ 2567, 5146, 0 },
		{ 2568, 5180, 0 },
		{ 2309, 1404, 441 },
		{ 2309, 1416, 441 },
		{ 2586, 444, 441 },
		{ 3021, 2805, 0 },
		{ 3253, 3346, 0 },
		{ 3088, 3099, 0 },
		{ 3220, 2304, 0 },
		{ 3239, 3245, 0 },
		{ 2309, 1433, 441 },
		{ 3205, 5345, 432 },
		{ 2381, 814, 442 },
		{ 2578, 5258, 0 },
		{ 3220, 2307, 0 },
		{ 3201, 2644, 0 },
		{ 3255, 1952, 0 },
		{ 2309, 1440, 441 },
		{ 3255, 2003, 0 },
		{ 3220, 2313, 0 },
		{ 2586, 548, 441 },
		{ 2309, 1437, 441 },
		{ 2586, 552, 441 },
		{ 3205, 2648, 0 },
		{ 3259, 2353, 0 },
		{ 3250, 2739, 0 },
		{ 3244, 2703, 0 },
		{ 2586, 554, 441 },
		{ 3259, 2357, 0 },
		{ 3205, 2265, 0 },
		{ 2586, 556, 441 },
		{ 3255, 2089, 0 },
		{ 3255, 2185, 0 },
		{ 3239, 2901, 0 },
		{ 2586, 558, 441 },
		{ 2586, 560, 441 },
		{ 3254, 2234, 0 },
		{ 3142, 2993, 0 },
		{ 3259, 2372, 0 },
		{ 3088, 3077, 0 },
		{ 3250, 2778, 0 },
		{ 3247, 1882, 0 },
		{ 2586, 1649, 441 },
		{ 3257, 2197, 0 },
		{ 2612, 5184, 0 },
		{ 3242, 3238, 0 },
		{ 3260, 4815, 0 },
		{ 2586, 562, 441 },
		{ 3220, 2343, 0 },
		{ 3260, 4869, 0 },
		{ 3205, 2650, 0 },
		{ 2609, 666, 441 },
		{ 3257, 2320, 0 },
		{ 3242, 3116, 0 },
		{ 3255, 2164, 0 },
		{ 3253, 3367, 0 },
		{ 2624, 5235, 0 },
		{ 3257, 2176, 0 },
		{ 3257, 2524, 0 },
		{ 3259, 2387, 0 },
		{ 3205, 2258, 0 },
		{ 3259, 2389, 0 },
		{ 3259, 2390, 0 },
		{ 3242, 3133, 0 },
		{ 3205, 2261, 0 },
		{ 3220, 2204, 0 },
		{ 3220, 2275, 0 },
		{ 3201, 2618, 0 },
		{ 2637, 5269, 0 },
		{ 3201, 2620, 0 },
		{ 3242, 3141, 0 },
		{ 3220, 2279, 0 },
		{ 3253, 3320, 0 },
		{ 3254, 3266, 0 },
		{ 2643, 889, 441 },
		{ 3242, 3145, 0 },
		{ 3002, 2450, 0 },
		{ 3260, 5011, 0 },
		{ 3220, 2281, 0 },
		{ 3205, 5285, 407 },
		{ 3220, 2217, 0 },
		{ 3260, 5025, 0 },
		{ 3205, 5309, 416 },
		{ 3257, 2468, 0 },
		{ 3205, 4042, 0 },
		{ 3002, 2434, 0 },
		{ 3201, 2634, 0 },
		{ 3260, 4797, 0 },
		{ 3255, 2208, 0 },
		{ 3252, 2760, 0 },
		{ 3253, 3342, 0 },
		{ 3088, 3086, 0 },
		{ 3044, 3061, 0 },
		{ 3257, 2472, 0 },
		{ 3259, 2401, 0 },
		{ 3242, 3165, 0 },
		{ 3242, 3168, 0 },
		{ 3002, 2439, 0 },
		{ 3259, 2403, 0 },
		{ 3142, 2982, 0 },
		{ 3224, 1670, 0 },
		{ 3247, 1879, 0 },
		{ 3220, 2221, 0 },
		{ 3044, 3068, 0 },
		{ 3201, 2651, 0 },
		{ 3002, 2452, 0 },
		{ 3201, 2654, 0 },
		{ 3242, 3184, 0 },
		{ 3260, 5087, 0 },
		{ 3205, 5293, 430 },
		{ 3201, 2553, 0 },
		{ 3255, 2222, 0 },
		{ 0, 0, 448 },
		{ 3220, 2292, 0 },
		{ 3142, 3002, 0 },
		{ 3205, 5319, 415 },
		{ 3253, 3313, 0 },
		{ 0, 4599, 0 },
		{ 3242, 3192, 0 },
		{ 3142, 3003, 0 },
		{ 3205, 5343, 436 },
		{ 3142, 3005, 0 },
		{ 3142, 3006, 0 },
		{ 3259, 2413, 0 },
		{ 3088, 3098, 0 },
		{ 2688, 5260, 0 },
		{ 2749, 3291, 0 },
		{ 3257, 2488, 0 },
		{ 3242, 3203, 0 },
		{ 3242, 3206, 0 },
		{ 3255, 2231, 0 },
		{ 3257, 2490, 0 },
		{ 2679, 1542, 0 },
		{ 2696, 5113, 0 },
		{ 3002, 2438, 0 },
		{ 3254, 3288, 0 },
		{ 2699, 5164, 0 },
		{ 3255, 2235, 0 },
		{ 3259, 2423, 0 },
		{ 3239, 3254, 0 },
		{ 2703, 5143, 0 },
		{ 3242, 3215, 0 },
		{ 3142, 3022, 0 },
		{ 2706, 5171, 0 },
		{ 0, 1557, 0 },
		{ 3250, 2741, 0 },
		{ 3260, 4873, 0 },
		{ 3259, 2352, 0 },
		{ 3255, 2240, 0 },
		{ 3257, 2503, 0 },
		{ 3250, 2757, 0 },
		{ 3242, 3226, 0 },
		{ 3220, 2300, 0 },
		{ 3205, 3059, 0 },
		{ 3253, 3369, 0 },
		{ 2749, 3296, 0 },
		{ 2719, 5239, 0 },
		{ 2720, 5240, 0 },
		{ 3249, 3046, 0 },
		{ 2749, 3298, 0 },
		{ 3242, 3231, 0 },
		{ 3220, 2228, 0 },
		{ 3250, 2763, 0 },
		{ 3259, 2359, 0 },
		{ 3205, 3667, 0 },
		{ 3220, 2302, 0 },
		{ 3142, 2891, 0 },
		{ 2730, 5258, 0 },
		{ 3257, 2332, 0 },
		{ 3259, 2362, 0 },
		{ 3244, 2688, 0 },
		{ 3254, 2966, 0 },
		{ 3242, 3114, 0 },
		{ 3260, 4877, 0 },
		{ 3205, 5220, 433 },
		{ 3253, 3319, 0 },
		{ 3257, 2512, 0 },
		{ 3201, 2595, 0 },
		{ 3242, 3122, 0 },
		{ 3201, 2597, 0 },
		{ 3002, 2437, 0 },
		{ 3247, 1716, 0 },
		{ 2749, 3300, 0 },
		{ 3253, 3331, 0 },
		{ 3239, 2897, 0 },
		{ 3239, 2899, 0 },
		{ 2748, 5228, 0 },
		{ 3253, 3334, 0 },
		{ 3260, 5029, 0 },
		{ 3255, 2253, 0 },
		{ 3257, 2518, 0 },
		{ 3142, 2983, 0 },
		{ 3253, 3341, 0 },
		{ 2755, 5215, 0 },
		{ 3201, 2604, 0 },
		{ 3244, 2356, 0 },
		{ 3002, 2445, 0 },
		{ 3253, 3347, 0 },
		{ 3142, 2991, 0 },
		{ 3253, 3350, 0 },
		{ 3260, 4871, 0 },
		{ 3205, 5338, 428 },
		{ 3255, 1912, 0 },
		{ 3259, 2369, 0 },
		{ 3260, 4912, 0 },
		{ 3260, 4914, 0 },
		{ 3255, 1914, 0 },
		{ 3259, 2371, 0 },
		{ 3088, 3093, 0 },
		{ 3142, 2999, 0 },
		{ 2749, 3297, 0 },
		{ 3242, 3149, 0 },
		{ 3242, 3150, 0 },
		{ 3260, 4984, 0 },
		{ 0, 3299, 0 },
		{ 3205, 5373, 414 },
		{ 3253, 3370, 0 },
		{ 3255, 1916, 0 },
		{ 3002, 2456, 0 },
		{ 3255, 1917, 0 },
		{ 3257, 2344, 0 },
		{ 3044, 3070, 0 },
		{ 3257, 2536, 0 },
		{ 3242, 3159, 0 },
		{ 3255, 1919, 0 },
		{ 3220, 2315, 0 },
		{ 3220, 2317, 0 },
		{ 3205, 5249, 408 },
		{ 3257, 2462, 0 },
		{ 3220, 2318, 0 },
		{ 3205, 5267, 420 },
		{ 3205, 5269, 421 },
		{ 3220, 2319, 0 },
		{ 3142, 3014, 0 },
		{ 3088, 3089, 0 },
		{ 3250, 2747, 0 },
		{ 3142, 3017, 0 },
		{ 3002, 2441, 0 },
		{ 3002, 2442, 0 },
		{ 0, 0, 447 },
		{ 3142, 3021, 0 },
		{ 3255, 1920, 0 },
		{ 2798, 5207, 0 },
		{ 3255, 1922, 0 },
		{ 3249, 3055, 0 },
		{ 3002, 2446, 0 },
		{ 2802, 5223, 0 },
		{ 3239, 3261, 0 },
		{ 3259, 2384, 0 },
		{ 3142, 3026, 0 },
		{ 3253, 3337, 0 },
		{ 3242, 3188, 0 },
		{ 3259, 2386, 0 },
		{ 3260, 5019, 0 },
		{ 3260, 5021, 0 },
		{ 3201, 2648, 0 },
		{ 3242, 3191, 0 },
		{ 3142, 3032, 0 },
		{ 3250, 2771, 0 },
		{ 3255, 1945, 0 },
		{ 3255, 1946, 0 },
		{ 3250, 2776, 0 },
		{ 3220, 2324, 0 },
		{ 3220, 2200, 0 },
		{ 3260, 4801, 0 },
		{ 3260, 4809, 0 },
		{ 3242, 3202, 0 },
		{ 3257, 2282, 0 },
		{ 3242, 3205, 0 },
		{ 3253, 3362, 0 },
		{ 3257, 2481, 0 },
		{ 3255, 1950, 0 },
		{ 3220, 2330, 0 },
		{ 3260, 4883, 0 },
		{ 3259, 1017, 411 },
		{ 3205, 5239, 423 },
		{ 3044, 3067, 0 },
		{ 3259, 2395, 0 },
		{ 3255, 1955, 0 },
		{ 3142, 2965, 0 },
		{ 3249, 3042, 0 },
		{ 3249, 3043, 0 },
		{ 3142, 2966, 0 },
		{ 2837, 5166, 0 },
		{ 3254, 3268, 0 },
		{ 3205, 5273, 419 },
		{ 3205, 5281, 434 },
		{ 3259, 2397, 0 },
		{ 3002, 2440, 0 },
		{ 3250, 2790, 0 },
		{ 3255, 1980, 0 },
		{ 3201, 2571, 0 },
		{ 3142, 2973, 0 },
		{ 2845, 5242, 0 },
		{ 3205, 5301, 410 },
		{ 3260, 5023, 0 },
		{ 2847, 5254, 0 },
		{ 2852, 1665, 0 },
		{ 3255, 1988, 0 },
		{ 2850, 5258, 0 },
		{ 2851, 5262, 0 },
		{ 3255, 2000, 0 },
		{ 3252, 2768, 0 },
		{ 3259, 2402, 0 },
		{ 3253, 3323, 0 },
		{ 3242, 3236, 0 },
		{ 3260, 4795, 0 },
		{ 3257, 2498, 0 },
		{ 3220, 2339, 0 },
		{ 3257, 2500, 0 },
		{ 3260, 4803, 0 },
		{ 3205, 5333, 425 },
		{ 3260, 4805, 0 },
		{ 3260, 4807, 0 },
		{ 2852, 1559, 0 },
		{ 3260, 4811, 0 },
		{ 3260, 4813, 0 },
		{ 0, 1592, 0 },
		{ 3142, 2994, 0 },
		{ 3142, 2996, 0 },
		{ 3255, 2008, 0 },
		{ 3259, 2408, 0 },
		{ 3205, 5354, 431 },
		{ 3259, 2409, 0 },
		{ 3260, 4875, 0 },
		{ 3201, 2592, 0 },
		{ 0, 0, 450 },
		{ 0, 0, 449 },
		{ 3205, 5371, 412 },
		{ 3260, 4879, 0 },
		{ 0, 0, 446 },
		{ 0, 0, 445 },
		{ 3260, 4881, 0 },
		{ 3250, 2755, 0 },
		{ 3002, 2457, 0 },
		{ 3257, 2509, 0 },
		{ 3253, 3344, 0 },
		{ 3260, 4941, 0 },
		{ 3205, 5225, 406 },
		{ 2882, 5132, 0 },
		{ 3205, 5229, 435 },
		{ 3205, 5231, 413 },
		{ 3242, 3129, 0 },
		{ 3255, 2012, 0 },
		{ 3259, 2411, 0 },
		{ 3255, 2013, 0 },
		{ 3205, 5241, 426 },
		{ 3205, 2533, 0 },
		{ 3260, 4953, 0 },
		{ 3260, 4955, 0 },
		{ 3260, 4957, 0 },
		{ 3257, 2514, 0 },
		{ 3255, 2014, 0 },
		{ 3205, 5255, 417 },
		{ 3205, 5257, 418 },
		{ 3205, 5265, 422 },
		{ 3259, 2414, 0 },
		{ 3242, 3138, 0 },
		{ 3260, 5015, 0 },
		{ 3259, 2415, 0 },
		{ 3205, 5275, 424 },
		{ 3253, 3361, 0 },
		{ 3255, 2015, 0 },
		{ 3142, 3016, 0 },
		{ 3257, 2520, 0 },
		{ 3201, 2611, 0 },
		{ 3220, 2268, 0 },
		{ 3260, 5031, 0 },
		{ 3205, 5299, 429 },
		{ 3149, 5145, 451 },
		{ 2909, 0, 403 },
		{ 0, 0, 404 },
		{ -2907, 5382, 399 },
		{ -2908, 5180, 0 },
		{ 3205, 5156, 0 },
		{ 3149, 5134, 0 },
		{ 0, 0, 400 },
		{ 3149, 5152, 0 },
		{ -2913, 16, 0 },
		{ -2914, 5184, 0 },
		{ 2917, 0, 401 },
		{ 3149, 5153, 0 },
		{ 3205, 5314, 0 },
		{ 0, 0, 402 },
		{ 3196, 3677, 166 },
		{ 0, 0, 166 },
		{ 0, 0, 167 },
		{ 3220, 2272, 0 },
		{ 3242, 3148, 0 },
		{ 3259, 2424, 0 },
		{ 2926, 5234, 0 },
		{ 3252, 2762, 0 },
		{ 3247, 1851, 0 },
		{ 3201, 2619, 0 },
		{ 3254, 3279, 0 },
		{ 3255, 2040, 0 },
		{ 3142, 3030, 0 },
		{ 3257, 2530, 0 },
		{ 3201, 2624, 0 },
		{ 3220, 2277, 0 },
		{ 3260, 4840, 0 },
		{ 0, 0, 164 },
		{ 3075, 4947, 189 },
		{ 0, 0, 189 },
		{ 3255, 2041, 0 },
		{ 2941, 5259, 0 },
		{ 3242, 2827, 0 },
		{ 3253, 3317, 0 },
		{ 3254, 3269, 0 },
		{ 3249, 3056, 0 },
		{ 2946, 5265, 0 },
		{ 3205, 2644, 0 },
		{ 3242, 3164, 0 },
		{ 3201, 2629, 0 },
		{ 3242, 3167, 0 },
		{ 3259, 2356, 0 },
		{ 3253, 3326, 0 },
		{ 3255, 2059, 0 },
		{ 3142, 2869, 0 },
		{ 3257, 2537, 0 },
		{ 3201, 2633, 0 },
		{ 2957, 5135, 0 },
		{ 3205, 3057, 0 },
		{ 3242, 3175, 0 },
		{ 3088, 3092, 0 },
		{ 3257, 2538, 0 },
		{ 3259, 2358, 0 },
		{ 3242, 3179, 0 },
		{ 2964, 5131, 0 },
		{ 3259, 2232, 0 },
		{ 3242, 3181, 0 },
		{ 3239, 3242, 0 },
		{ 3247, 1854, 0 },
		{ 3254, 3284, 0 },
		{ 3242, 3183, 0 },
		{ 2971, 5153, 0 },
		{ 3252, 2770, 0 },
		{ 3247, 1855, 0 },
		{ 3201, 2642, 0 },
		{ 3254, 3267, 0 },
		{ 3255, 2073, 0 },
		{ 3142, 2970, 0 },
		{ 3257, 2464, 0 },
		{ 3201, 2647, 0 },
		{ 3260, 5058, 0 },
		{ 0, 0, 187 },
		{ 2982, 0, 1 },
		{ -2982, 1506, 278 },
		{ 3242, 3089, 284 },
		{ 0, 0, 284 },
		{ 3220, 2291, 0 },
		{ 3201, 2652, 0 },
		{ 3242, 3200, 0 },
		{ 3239, 3246, 0 },
		{ 3259, 2368, 0 },
		{ 0, 0, 283 },
		{ 2992, 5221, 0 },
		{ 3244, 2241, 0 },
		{ 3253, 3304, 0 },
		{ 3021, 2807, 0 },
		{ 3242, 3204, 0 },
		{ 3088, 3094, 0 },
		{ 3142, 2990, 0 },
		{ 3250, 2773, 0 },
		{ 3242, 3209, 0 },
		{ 3001, 5210, 0 },
		{ 3257, 2293, 0 },
		{ 0, 2443, 0 },
		{ 3255, 2108, 0 },
		{ 3142, 2995, 0 },
		{ 3257, 2475, 0 },
		{ 3201, 2557, 0 },
		{ 3220, 2295, 0 },
		{ 3260, 4887, 0 },
		{ 0, 0, 282 },
		{ 0, 4735, 192 },
		{ 0, 0, 192 },
		{ 3257, 2477, 0 },
		{ 3247, 1714, 0 },
		{ 3201, 2561, 0 },
		{ 3239, 3249, 0 },
		{ 3017, 5245, 0 },
		{ 3254, 2982, 0 },
		{ 3249, 3044, 0 },
		{ 3242, 3223, 0 },
		{ 3254, 3275, 0 },
		{ 0, 2809, 0 },
		{ 3142, 3004, 0 },
		{ 3201, 2562, 0 },
		{ 3044, 3064, 0 },
		{ 3260, 4986, 0 },
		{ 0, 0, 190 },
		{ 3075, 4945, 186 },
		{ 0, 0, 185 },
		{ 0, 0, 186 },
		{ 3255, 2117, 0 },
		{ 3032, 5250, 0 },
		{ 3255, 2189, 0 },
		{ 3249, 3057, 0 },
		{ 3242, 3233, 0 },
		{ 3036, 5275, 0 },
		{ 3205, 3053, 0 },
		{ 3242, 3235, 0 },
		{ 3044, 3059, 0 },
		{ 3142, 3009, 0 },
		{ 3201, 2565, 0 },
		{ 3201, 2567, 0 },
		{ 3142, 3012, 0 },
		{ 3201, 2569, 0 },
		{ 0, 3065, 0 },
		{ 3046, 5121, 0 },
		{ 3257, 2313, 0 },
		{ 3088, 3082, 0 },
		{ 3049, 5135, 0 },
		{ 3242, 2824, 0 },
		{ 3253, 3355, 0 },
		{ 3254, 3282, 0 },
		{ 3249, 3050, 0 },
		{ 3054, 5140, 0 },
		{ 3205, 2652, 0 },
		{ 3242, 3123, 0 },
		{ 3201, 2575, 0 },
		{ 3242, 3126, 0 },
		{ 3259, 2378, 0 },
		{ 3253, 3365, 0 },
		{ 3255, 2136, 0 },
		{ 3142, 3020, 0 },
		{ 3257, 2484, 0 },
		{ 3201, 2579, 0 },
		{ 3065, 5154, 0 },
		{ 3252, 2756, 0 },
		{ 3247, 1743, 0 },
		{ 3201, 2581, 0 },
		{ 3254, 3277, 0 },
		{ 3255, 2147, 0 },
		{ 3142, 3027, 0 },
		{ 3257, 2487, 0 },
		{ 3201, 2584, 0 },
		{ 3260, 4885, 0 },
		{ 0, 0, 179 },
		{ 0, 4799, 178 },
		{ 0, 0, 178 },
		{ 3255, 2149, 0 },
		{ 3079, 5156, 0 },
		{ 3255, 2193, 0 },
		{ 3249, 3040, 0 },
		{ 3242, 3143, 0 },
		{ 3083, 5187, 0 },
		{ 3242, 2812, 0 },
		{ 3201, 2587, 0 },
		{ 3239, 3243, 0 },
		{ 3087, 5185, 0 },
		{ 3257, 2330, 0 },
		{ 0, 3084, 0 },
		{ 3090, 5203, 0 },
		{ 3242, 2820, 0 },
		{ 3253, 3321, 0 },
		{ 3254, 3272, 0 },
		{ 3249, 3045, 0 },
		{ 3095, 5211, 0 },
		{ 3205, 2646, 0 },
		{ 3242, 3151, 0 },
		{ 3201, 2590, 0 },
		{ 3242, 3153, 0 },
		{ 3259, 2385, 0 },
		{ 3253, 3329, 0 },
		{ 3255, 2162, 0 },
		{ 3142, 2889, 0 },
		{ 3257, 2493, 0 },
		{ 3201, 2594, 0 },
		{ 3106, 5230, 0 },
		{ 3252, 2772, 0 },
		{ 3247, 1747, 0 },
		{ 3201, 2596, 0 },
		{ 3254, 3264, 0 },
		{ 3255, 2166, 0 },
		{ 3142, 2959, 0 },
		{ 3257, 2496, 0 },
		{ 3201, 2599, 0 },
		{ 3260, 5089, 0 },
		{ 0, 0, 176 },
		{ 0, 4346, 181 },
		{ 0, 0, 181 },
		{ 0, 0, 182 },
		{ 3201, 2600, 0 },
		{ 3220, 2311, 0 },
		{ 3255, 2169, 0 },
		{ 3242, 3170, 0 },
		{ 3253, 3349, 0 },
		{ 3239, 3252, 0 },
		{ 3126, 5258, 0 },
		{ 3242, 2818, 0 },
		{ 3224, 1710, 0 },
		{ 3253, 3354, 0 },
		{ 3250, 2787, 0 },
		{ 3247, 1796, 0 },
		{ 3253, 3357, 0 },
		{ 3255, 2180, 0 },
		{ 3142, 2977, 0 },
		{ 3257, 2502, 0 },
		{ 3201, 2607, 0 },
		{ 3137, 5274, 0 },
		{ 3252, 2758, 0 },
		{ 3247, 1816, 0 },
		{ 3201, 2609, 0 },
		{ 3254, 3265, 0 },
		{ 3255, 2191, 0 },
		{ 0, 2989, 0 },
		{ 3257, 2505, 0 },
		{ 3201, 2612, 0 },
		{ 3222, 5095, 0 },
		{ 0, 0, 180 },
		{ 3242, 3187, 451 },
		{ 3259, 1617, 26 },
		{ 0, 5159, 451 },
		{ 3158, 0, 451 },
		{ 2308, 2954, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 3201, 2616, 0 },
		{ -3157, 5383, 0 },
		{ 3259, 91, 0 },
		{ 0, 0, 28 },
		{ 3239, 3256, 0 },
		{ 0, 0, 27 },
		{ 0, 0, 22 },
		{ 0, 0, 34 },
		{ 0, 0, 35 },
		{ 3179, 4158, 39 },
		{ 0, 3874, 39 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 3191, 4027, 0 },
		{ 3206, 4698, 0 },
		{ 3196, 3651, 0 },
		{ 0, 0, 37 },
		{ 3200, 3712, 0 },
		{ 3204, 3547, 0 },
		{ 3214, 2111, 0 },
		{ 0, 0, 36 },
		{ 3242, 3104, 50 },
		{ 0, 0, 50 },
		{ 0, 4167, 50 },
		{ 3196, 3656, 50 },
		{ 3242, 3197, 50 },
		{ 0, 0, 58 },
		{ 3242, 3198, 0 },
		{ 3201, 2621, 0 },
		{ 3191, 4039, 0 },
		{ 3200, 3721, 0 },
		{ 3255, 2219, 0 },
		{ 3201, 2623, 0 },
		{ 3239, 3244, 0 },
		{ 3196, 3666, 0 },
		{ 0, 3981, 0 },
		{ 3247, 1853, 0 },
		{ 3257, 2515, 0 },
		{ 0, 0, 49 },
		{ 3200, 3729, 0 },
		{ 0, 3669, 0 },
		{ 3259, 2405, 0 },
		{ 3244, 2710, 0 },
		{ 3204, 3544, 54 },
		{ 0, 3735, 0 },
		{ 0, 2627, 0 },
		{ 3242, 3207, 0 },
		{ 3208, 1854, 0 },
		{ 0, 3546, 51 },
		{ 0, 5, 59 },
		{ 0, 4672, 0 },
		{ 3214, 1938, 0 },
		{ 3259, 1694, 0 },
		{ 3215, 1844, 0 },
		{ 0, 0, 57 },
		{ 3250, 2774, 0 },
		{ 0, 0, 55 },
		{ 0, 0, 56 },
		{ 3151, 2098, 0 },
		{ 3259, 1769, 0 },
		{ 3253, 3335, 0 },
		{ 0, 0, 52 },
		{ 0, 0, 53 },
		{ 3220, 2336, 0 },
		{ 0, 2337, 0 },
		{ 3222, 5084, 0 },
		{ 0, 5086, 0 },
		{ 3242, 3217, 0 },
		{ 0, 1712, 0 },
		{ 3253, 3340, 0 },
		{ 3250, 2780, 0 },
		{ 3247, 1885, 0 },
		{ 3253, 3343, 0 },
		{ 3255, 2251, 0 },
		{ 3257, 2531, 0 },
		{ 3259, 2418, 0 },
		{ 3253, 2359, 0 },
		{ 3242, 3225, 0 },
		{ 3257, 2534, 0 },
		{ 3254, 3271, 0 },
		{ 3253, 3351, 0 },
		{ 3259, 2420, 0 },
		{ 3254, 3273, 0 },
		{ 0, 3250, 0 },
		{ 3242, 3124, 0 },
		{ 3247, 1886, 0 },
		{ 0, 3230, 0 },
		{ 3253, 3358, 0 },
		{ 0, 2706, 0 },
		{ 3259, 2422, 0 },
		{ 3254, 3280, 0 },
		{ 0, 1909, 0 },
		{ 3260, 5091, 0 },
		{ 0, 3049, 0 },
		{ 0, 2792, 0 },
		{ 0, 0, 46 },
		{ 3205, 2984, 0 },
		{ 0, 3366, 0 },
		{ 0, 3285, 0 },
		{ 0, 2256, 0 },
		{ 3260, 5092, 0 },
		{ 0, 2540, 0 },
		{ 0, 0, 47 },
		{ 0, 2425, 0 },
		{ 3222, 5093, 0 },
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
		0,
		0,
		0,
		0
	};
	yybackup = backup;
}
