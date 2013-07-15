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
#line 1296 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GENOME_FILE);
#line 2004 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1297 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_IND_FILE);
#line 2011 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1298 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_TXT_GEN_FILE);
#line 2018 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1300 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 2025 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1301 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 2032 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1303 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 2039 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1304 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 2046 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1306 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 2060 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1314 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  if( TARGET==CUDA )
    strcat(sFileName,"Individual.cu");
  else if( TARGET==STD )
    strcat(sFileName,"Individual.cpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 2077 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1325 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2091 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1333 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2105 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1342 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2119 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1351 "EaseaLex.l"

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

#line 2182 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1408 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded without warnings.\n");
  printf ("You can now type \"make\" to compile your project.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 2199 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1420 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2206 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1426 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2218 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1432 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2231 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1439 "EaseaLex.l"

#line 2238 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1440 "EaseaLex.l"
lineCounter++;
#line 2245 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1442 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2257 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1448 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2270 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1456 "EaseaLex.l"

#line 2277 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1457 "EaseaLex.l"

  lineCounter++;
 
#line 2286 "EaseaLex.cpp"
		}
		break;
#line 1460 "EaseaLex.l"
               
#line 2291 "EaseaLex.cpp"
	case 163:
		{
#line 1461 "EaseaLex.l"

  fprintf (fpOutputFile,"// User CUDA\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2301 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1467 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user CUDA were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2314 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1475 "EaseaLex.l"

#line 2321 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1476 "EaseaLex.l"

  lineCounter++;
 
#line 2330 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1480 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2342 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1486 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2356 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1494 "EaseaLex.l"

#line 2363 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1495 "EaseaLex.l"

  lineCounter++;
 
#line 2372 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1499 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
      
  BEGIN COPY;
 
#line 2386 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1507 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2401 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1516 "EaseaLex.l"

#line 2408 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1517 "EaseaLex.l"
lineCounter++;
#line 2415 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1522 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2429 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1531 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2443 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1539 "EaseaLex.l"

#line 2450 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1540 "EaseaLex.l"
lineCounter++;
#line 2457 "EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 1543 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2473 "EaseaLex.cpp"
		}
		break;
	case 180:
		{
#line 1554 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2489 "EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 1564 "EaseaLex.l"

#line 2496 "EaseaLex.cpp"
		}
		break;
	case 182:
		{
#line 1567 "EaseaLex.l"

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
 
#line 2514 "EaseaLex.cpp"
		}
		break;
	case 183:
		{
#line 1580 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2531 "EaseaLex.cpp"
		}
		break;
	case 184:
		{
#line 1592 "EaseaLex.l"

#line 2538 "EaseaLex.cpp"
		}
		break;
	case 185:
		{
#line 1593 "EaseaLex.l"
lineCounter++;
#line 2545 "EaseaLex.cpp"
		}
		break;
	case 186:
		{
#line 1595 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2561 "EaseaLex.cpp"
		}
		break;
	case 187:
		{
#line 1607 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2577 "EaseaLex.cpp"
		}
		break;
	case 188:
		{
#line 1617 "EaseaLex.l"
lineCounter++;
#line 2584 "EaseaLex.cpp"
		}
		break;
	case 189:
		{
#line 1618 "EaseaLex.l"

#line 2591 "EaseaLex.cpp"
		}
		break;
	case 190:
		{
#line 1622 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2606 "EaseaLex.cpp"
		}
		break;
	case 191:
		{
#line 1632 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2621 "EaseaLex.cpp"
		}
		break;
	case 192:
		{
#line 1641 "EaseaLex.l"

#line 2628 "EaseaLex.cpp"
		}
		break;
	case 193:
		{
#line 1644 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    //fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    fprintf (fpOutputFile,"{\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2642 "EaseaLex.cpp"
		}
		break;
	case 194:
		{
#line 1652 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2656 "EaseaLex.cpp"
		}
		break;
	case 195:
		{
#line 1660 "EaseaLex.l"

#line 2663 "EaseaLex.cpp"
		}
		break;
	case 196:
		{
#line 1664 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2671 "EaseaLex.cpp"
		}
		break;
	case 197:
		{
#line 1666 "EaseaLex.l"

#line 2678 "EaseaLex.cpp"
		}
		break;
	case 198:
		{
#line 1672 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2685 "EaseaLex.cpp"
		}
		break;
	case 199:
		{
#line 1673 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2692 "EaseaLex.cpp"
		}
		break;
	case 200:
	case 201:
		{
#line 1676 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2703 "EaseaLex.cpp"
		}
		break;
	case 202:
	case 203:
		{
#line 1681 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2712 "EaseaLex.cpp"
		}
		break;
	case 204:
	case 205:
		{
#line 1684 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2721 "EaseaLex.cpp"
		}
		break;
	case 206:
	case 207:
		{
#line 1687 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2738 "EaseaLex.cpp"
		}
		break;
	case 208:
	case 209:
		{
#line 1698 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2752 "EaseaLex.cpp"
		}
		break;
	case 210:
	case 211:
		{
#line 1706 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2761 "EaseaLex.cpp"
		}
		break;
	case 212:
	case 213:
		{
#line 1709 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2770 "EaseaLex.cpp"
		}
		break;
	case 214:
	case 215:
		{
#line 1712 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2779 "EaseaLex.cpp"
		}
		break;
	case 216:
	case 217:
		{
#line 1715 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2788 "EaseaLex.cpp"
		}
		break;
	case 218:
	case 219:
		{
#line 1718 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2797 "EaseaLex.cpp"
		}
		break;
	case 220:
	case 221:
		{
#line 1722 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2809 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1728 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2816 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1729 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2823 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1730 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2830 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1731 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2840 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1736 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2847 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1737 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2854 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1738 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2861 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1739 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2868 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1740 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2875 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1741 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2882 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1742 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2889 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1743 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2896 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1744 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2904 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1746 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2912 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1748 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2920 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1750 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2930 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1754 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2937 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1755 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2944 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1756 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2955 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1761 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2962 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1762 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2971 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1765 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2983 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1771 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2992 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1774 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 3004 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1780 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 3015 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1785 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 3031 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1795 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3038 "EaseaLex.cpp"
		}
		break;
	case 249:
		{
#line 1798 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 3047 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1801 "EaseaLex.l"
BEGIN COPY;
#line 3054 "EaseaLex.cpp"
		}
		break;
	case 251:
		{
#line 1803 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 3061 "EaseaLex.cpp"
		}
		break;
	case 252:
	case 253:
	case 254:
		{
#line 1806 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 3074 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1811 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 3085 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1816 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 3094 "EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1825 "EaseaLex.l"
;
#line 3101 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1826 "EaseaLex.l"
;
#line 3108 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1827 "EaseaLex.l"
;
#line 3115 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1828 "EaseaLex.l"
;
#line 3122 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1831 "EaseaLex.l"
 /* do nothing */ 
#line 3129 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1832 "EaseaLex.l"
 /*return '\n';*/ 
#line 3136 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1833 "EaseaLex.l"
 /*return '\n';*/ 
#line 3143 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1836 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 3152 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1839 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 3162 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1843 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
//  printf("match gpnode\n");
  return GPNODE;
 
#line 3174 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1850 "EaseaLex.l"
return STATIC;
#line 3181 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1851 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 3188 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1852 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 3195 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1853 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 3202 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1854 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 3209 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1855 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 3216 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1857 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 3223 "EaseaLex.cpp"
		}
		break;
#line 1858 "EaseaLex.l"
  
#line 3228 "EaseaLex.cpp"
	case 274:
		{
#line 1859 "EaseaLex.l"
return GENOME; 
#line 3233 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1861 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 3243 "EaseaLex.cpp"
		}
		break;
	case 276:
	case 277:
	case 278:
		{
#line 1868 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 3252 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1869 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 3259 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1872 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 3267 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1874 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 3274 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1880 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 3286 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1886 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 3299 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1893 "EaseaLex.l"

#line 3306 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1895 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3317 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1906 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3332 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1916 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3343 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1922 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3352 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1926 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3367 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1939 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3379 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1945 "EaseaLex.l"

#line 3386 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1946 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3399 "EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1953 "EaseaLex.l"

#line 3406 "EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1954 "EaseaLex.l"
lineCounter++;
#line 3413 "EaseaLex.cpp"
		}
		break;
	case 295:
		{
#line 1955 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3426 "EaseaLex.cpp"
		}
		break;
	case 296:
		{
#line 1962 "EaseaLex.l"

#line 3433 "EaseaLex.cpp"
		}
		break;
	case 297:
		{
#line 1963 "EaseaLex.l"
lineCounter++;
#line 3440 "EaseaLex.cpp"
		}
		break;
	case 298:
		{
#line 1965 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3453 "EaseaLex.cpp"
		}
		break;
	case 299:
		{
#line 1972 "EaseaLex.l"

#line 3460 "EaseaLex.cpp"
		}
		break;
	case 300:
		{
#line 1973 "EaseaLex.l"
lineCounter++;
#line 3467 "EaseaLex.cpp"
		}
		break;
	case 301:
		{
#line 1976 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3480 "EaseaLex.cpp"
		}
		break;
	case 302:
		{
#line 1983 "EaseaLex.l"

#line 3487 "EaseaLex.cpp"
		}
		break;
	case 303:
		{
#line 1984 "EaseaLex.l"
lineCounter++;
#line 3494 "EaseaLex.cpp"
		}
		break;
	case 304:
		{
#line 1990 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3501 "EaseaLex.cpp"
		}
		break;
	case 305:
		{
#line 1991 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3508 "EaseaLex.cpp"
		}
		break;
	case 306:
		{
#line 1992 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3515 "EaseaLex.cpp"
		}
		break;
	case 307:
		{
#line 1993 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3522 "EaseaLex.cpp"
		}
		break;
	case 308:
		{
#line 1994 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3529 "EaseaLex.cpp"
		}
		break;
	case 309:
		{
#line 1995 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3536 "EaseaLex.cpp"
		}
		break;
	case 310:
		{
#line 1996 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3543 "EaseaLex.cpp"
		}
		break;
	case 311:
		{
#line 1998 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3552 "EaseaLex.cpp"
		}
		break;
	case 312:
		{
#line 2001 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3565 "EaseaLex.cpp"
		}
		break;
	case 313:
	case 314:
		{
#line 2010 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3576 "EaseaLex.cpp"
		}
		break;
	case 315:
	case 316:
		{
#line 2015 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3585 "EaseaLex.cpp"
		}
		break;
	case 317:
	case 318:
		{
#line 2018 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3594 "EaseaLex.cpp"
		}
		break;
	case 319:
	case 320:
		{
#line 2021 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3606 "EaseaLex.cpp"
		}
		break;
	case 321:
	case 322:
		{
#line 2027 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3619 "EaseaLex.cpp"
		}
		break;
	case 323:
	case 324:
		{
#line 2034 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3628 "EaseaLex.cpp"
		}
		break;
	case 325:
	case 326:
		{
#line 2037 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3637 "EaseaLex.cpp"
		}
		break;
	case 327:
	case 328:
		{
#line 2040 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3646 "EaseaLex.cpp"
		}
		break;
	case 329:
	case 330:
		{
#line 2043 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3655 "EaseaLex.cpp"
		}
		break;
	case 331:
	case 332:
		{
#line 2046 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3664 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 2049 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3673 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 2052 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3683 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 2056 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3691 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 2058 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3702 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 2063 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3713 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 2068 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3721 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 2070 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3729 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 2072 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3737 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 2074 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3745 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 2076 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3753 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 2078 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3760 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 2079 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3767 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 2080 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3775 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 2082 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3783 "EaseaLex.cpp"
		}
		break;
	case 347:
		{
#line 2084 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3791 "EaseaLex.cpp"
		}
		break;
	case 348:
		{
#line 2086 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3798 "EaseaLex.cpp"
		}
		break;
	case 349:
		{
#line 2087 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3810 "EaseaLex.cpp"
		}
		break;
	case 350:
		{
#line 2093 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3819 "EaseaLex.cpp"
		}
		break;
	case 351:
		{
#line 2096 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3829 "EaseaLex.cpp"
		}
		break;
	case 352:
		{
#line 2100 "EaseaLex.l"
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
#line 3846 "EaseaLex.cpp"
		}
		break;
	case 353:
	case 354:
		{
#line 2112 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3856 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 2115 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3863 "EaseaLex.cpp"
		}
		break;
	case 356:
		{
#line 2122 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3870 "EaseaLex.cpp"
		}
		break;
	case 357:
		{
#line 2123 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3877 "EaseaLex.cpp"
		}
		break;
	case 358:
		{
#line 2124 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3884 "EaseaLex.cpp"
		}
		break;
	case 359:
		{
#line 2125 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3891 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 2126 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3898 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 2128 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3907 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 2132 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3920 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 2140 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3933 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 2149 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3946 "EaseaLex.cpp"
		}
		break;
	case 365:
		{
#line 2158 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3961 "EaseaLex.cpp"
		}
		break;
	case 366:
		{
#line 2168 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3968 "EaseaLex.cpp"
		}
		break;
	case 367:
		{
#line 2169 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3975 "EaseaLex.cpp"
		}
		break;
	case 368:
	case 369:
		{
#line 2172 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3986 "EaseaLex.cpp"
		}
		break;
	case 370:
	case 371:
		{
#line 2177 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3995 "EaseaLex.cpp"
		}
		break;
	case 372:
	case 373:
		{
#line 2180 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 4004 "EaseaLex.cpp"
		}
		break;
	case 374:
	case 375:
		{
#line 2183 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 4017 "EaseaLex.cpp"
		}
		break;
	case 376:
	case 377:
		{
#line 2190 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 4030 "EaseaLex.cpp"
		}
		break;
	case 378:
	case 379:
		{
#line 2197 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 4039 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2200 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 4046 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2201 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4053 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2202 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4060 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2203 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 4070 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2208 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4077 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2209 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4084 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2210 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 4091 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2211 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 4098 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2212 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 4106 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2214 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 4114 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2216 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 4122 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2218 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 4130 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2220 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 4138 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2222 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 4146 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2224 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 4154 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2226 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 4161 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2227 "EaseaLex.l"
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
#line 4184 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2244 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 4195 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2249 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 4209 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2257 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 4216 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2263 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 4226 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2267 "EaseaLex.l"

#line 4233 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2270 "EaseaLex.l"
;
#line 4240 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2271 "EaseaLex.l"
;
#line 4247 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2272 "EaseaLex.l"
;
#line 4254 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2273 "EaseaLex.l"
;
#line 4261 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2275 "EaseaLex.l"
 /* do nothing */ 
#line 4268 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2276 "EaseaLex.l"
 /*return '\n';*/ 
#line 4275 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2277 "EaseaLex.l"
 /*return '\n';*/ 
#line 4282 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2279 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 4289 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2280 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 4296 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2281 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 4303 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2282 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4310 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2283 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4317 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2284 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4324 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2285 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4331 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2286 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4338 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2287 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4345 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2289 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4352 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2290 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4359 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2291 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4366 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2293 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Genome File...\n");return GENERATE_GENOME_FILE;
#line 4373 "EaseaLex.cpp"
		}
		break;
	case 422:
		{
#line 2295 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to Ind.csv File...\n");return GENERATE_CSV_IND_FILE;
#line 4380 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2297 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Gen to Gen.txt File...\n");return GENERATE_TXT_GEN_FILE;
#line 4387 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2299 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4394 "EaseaLex.cpp"
		}
		break;
	case 425:
		{
#line 2300 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4401 "EaseaLex.cpp"
		}
		break;
	case 426:
		{
#line 2302 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4408 "EaseaLex.cpp"
		}
		break;
	case 427:
		{
#line 2303 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4415 "EaseaLex.cpp"
		}
		break;
	case 428:
		{
#line 2305 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4426 "EaseaLex.cpp"
		}
		break;
	case 429:
		{
#line 2310 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4433 "EaseaLex.cpp"
		}
		break;
	case 430:
		{
#line 2312 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4444 "EaseaLex.cpp"
		}
		break;
	case 431:
		{
#line 2317 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4451 "EaseaLex.cpp"
		}
		break;
	case 432:
		{
#line 2320 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4458 "EaseaLex.cpp"
		}
		break;
	case 433:
		{
#line 2321 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4465 "EaseaLex.cpp"
		}
		break;
	case 434:
		{
#line 2322 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4472 "EaseaLex.cpp"
		}
		break;
	case 435:
		{
#line 2323 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4479 "EaseaLex.cpp"
		}
		break;
	case 436:
		{
#line 2324 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4486 "EaseaLex.cpp"
		}
		break;
#line 2325 "EaseaLex.l"
 
#line 4491 "EaseaLex.cpp"
	case 437:
		{
#line 2326 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4496 "EaseaLex.cpp"
		}
		break;
	case 438:
		{
#line 2327 "EaseaLex.l"
if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;
#line 4503 "EaseaLex.cpp"
		}
		break;
	case 439:
		{
#line 2328 "EaseaLex.l"
if(bVERBOSE) printf("\tExperiment Id...\n"); return EXPID;
#line 4510 "EaseaLex.cpp"
		}
		break;
	case 440:
		{
#line 2329 "EaseaLex.l"
if(bVERBOSE) printf("\tGrid Working Path...\n"); return WORKING_PATH;
#line 4517 "EaseaLex.cpp"
		}
		break;
	case 441:
		{
#line 2330 "EaseaLex.l"
if(bVERBOSE) printf("\tMigration Probability...\n"); return MIGRATION_PROBABILITY;
#line 4524 "EaseaLex.cpp"
		}
		break;
	case 442:
		{
#line 2331 "EaseaLex.l"
if(bVERBOSE) printf("\tServer port...\n"); return SERVER_PORT;
#line 4531 "EaseaLex.cpp"
		}
		break;
#line 2333 "EaseaLex.l"
 
#line 4536 "EaseaLex.cpp"
	case 443:
	case 444:
	case 445:
		{
#line 2337 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4543 "EaseaLex.cpp"
		}
		break;
	case 446:
		{
#line 2338 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4550 "EaseaLex.cpp"
		}
		break;
	case 447:
		{
#line 2341 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4558 "EaseaLex.cpp"
		}
		break;
	case 448:
		{
#line 2344 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return PATH_IDENTIFIER; 
#line 4566 "EaseaLex.cpp"
		}
		break;
	case 449:
		{
#line 2349 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4573 "EaseaLex.cpp"
		}
		break;
	case 450:
		{
#line 2351 "EaseaLex.l"

  lineCounter++;

#line 4582 "EaseaLex.cpp"
		}
		break;
	case 451:
		{
#line 2354 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4592 "EaseaLex.cpp"
		}
		break;
	case 452:
		{
#line 2359 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4602 "EaseaLex.cpp"
		}
		break;
	case 453:
		{
#line 2364 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4612 "EaseaLex.cpp"
		}
		break;
	case 454:
		{
#line 2369 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4622 "EaseaLex.cpp"
		}
		break;
	case 455:
		{
#line 2374 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4632 "EaseaLex.cpp"
		}
		break;
	case 456:
		{
#line 2379 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4642 "EaseaLex.cpp"
		}
		break;
	case 457:
		{
#line 2388 "EaseaLex.l"
return  (char)yytext[0];
#line 4649 "EaseaLex.cpp"
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
#line 2390 "EaseaLex.l"


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

#line 4849 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		200,
		-201,
		0,
		202,
		-203,
		0,
		204,
		-205,
		0,
		206,
		-207,
		0,
		212,
		-213,
		0,
		214,
		-215,
		0,
		216,
		-217,
		0,
		218,
		-219,
		0,
		210,
		-211,
		0,
		208,
		-209,
		0,
		-250,
		0,
		-256,
		0,
		376,
		-377,
		0,
		321,
		-322,
		0,
		370,
		-371,
		0,
		372,
		-373,
		0,
		374,
		-375,
		0,
		368,
		-369,
		0,
		317,
		-318,
		0,
		319,
		-320,
		0,
		313,
		-314,
		0,
		325,
		-326,
		0,
		327,
		-328,
		0,
		329,
		-330,
		0,
		331,
		-332,
		0,
		315,
		-316,
		0,
		378,
		-379,
		0,
		323,
		-324,
		0
	};
	yymatch = match;

	yytransitionmax = 5516;
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
		{ 3227, 63 },
		{ 3227, 63 },
		{ 1963, 2066 },
		{ 1588, 1571 },
		{ 1589, 1571 },
		{ 2485, 2454 },
		{ 2485, 2454 },
		{ 0, 1902 },
		{ 2288, 2284 },
		{ 2460, 2425 },
		{ 2460, 2425 },
		{ 73, 3 },
		{ 74, 3 },
		{ 2322, 45 },
		{ 2323, 45 },
		{ 71, 1 },
		{ 2986, 2988 },
		{ 0, 2093 },
		{ 69, 1 },
		{ 2062, 2064 },
		{ 167, 163 },
		{ 2062, 2058 },
		{ 2288, 2290 },
		{ 3227, 63 },
		{ 1457, 1456 },
		{ 3225, 63 },
		{ 1588, 1571 },
		{ 3285, 3280 },
		{ 2485, 2454 },
		{ 2305, 2304 },
		{ 1625, 1609 },
		{ 1626, 1609 },
		{ 2460, 2425 },
		{ 168, 164 },
		{ 73, 3 },
		{ 3229, 63 },
		{ 2322, 45 },
		{ 88, 63 },
		{ 3224, 63 },
		{ 3226, 63 },
		{ 3226, 63 },
		{ 3226, 63 },
		{ 3226, 63 },
		{ 3226, 63 },
		{ 3226, 63 },
		{ 3226, 63 },
		{ 3226, 63 },
		{ 3226, 63 },
		{ 3226, 63 },
		{ 72, 3 },
		{ 3228, 63 },
		{ 2321, 45 },
		{ 1678, 1672 },
		{ 1625, 1609 },
		{ 2486, 2454 },
		{ 1590, 1571 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 1627, 1609 },
		{ 3222, 63 },
		{ 0, 2511 },
		{ 1680, 1674 },
		{ 3223, 63 },
		{ 1554, 1533 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3223, 63 },
		{ 3230, 63 },
		{ 3234, 3233 },
		{ 2463, 2428 },
		{ 2463, 2428 },
		{ 3233, 3233 },
		{ 2472, 2436 },
		{ 2472, 2436 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 2543, 2510 },
		{ 3233, 3233 },
		{ 2577, 2543 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 2463, 2428 },
		{ 1555, 1534 },
		{ 1556, 1535 },
		{ 2472, 2436 },
		{ 1557, 1536 },
		{ 1558, 1537 },
		{ 1559, 1538 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 1560, 1539 },
		{ 3233, 3233 },
		{ 1561, 1541 },
		{ 3233, 3233 },
		{ 1564, 1544 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 3233, 3233 },
		{ 2283, 42 },
		{ 1628, 1610 },
		{ 1629, 1610 },
		{ 1565, 1545 },
		{ 2070, 42 },
		{ 2551, 2519 },
		{ 2551, 2519 },
		{ 2483, 2452 },
		{ 2483, 2452 },
		{ 2493, 2461 },
		{ 2493, 2461 },
		{ 2085, 41 },
		{ 1566, 1546 },
		{ 1896, 39 },
		{ 2507, 2475 },
		{ 2507, 2475 },
		{ 1567, 1547 },
		{ 1568, 1548 },
		{ 1570, 1550 },
		{ 1571, 1551 },
		{ 1572, 1552 },
		{ 1573, 1553 },
		{ 1574, 1554 },
		{ 2283, 42 },
		{ 1628, 1610 },
		{ 2073, 42 },
		{ 1575, 1555 },
		{ 1576, 1556 },
		{ 2551, 2519 },
		{ 1577, 1557 },
		{ 2483, 2452 },
		{ 1578, 1558 },
		{ 2493, 2461 },
		{ 1579, 1559 },
		{ 2085, 41 },
		{ 1580, 1561 },
		{ 1896, 39 },
		{ 2507, 2475 },
		{ 2282, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2071, 41 },
		{ 2086, 42 },
		{ 1883, 39 },
		{ 1583, 1564 },
		{ 1630, 1610 },
		{ 2552, 2519 },
		{ 1584, 1565 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2072, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2080, 42 },
		{ 2078, 42 },
		{ 2091, 42 },
		{ 2079, 42 },
		{ 2091, 42 },
		{ 2082, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2081, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 1585, 1566 },
		{ 2074, 42 },
		{ 2076, 42 },
		{ 1586, 1568 },
		{ 2091, 42 },
		{ 1587, 1570 },
		{ 2091, 42 },
		{ 2089, 42 },
		{ 2077, 42 },
		{ 2091, 42 },
		{ 2090, 42 },
		{ 2083, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2088, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2075, 42 },
		{ 2091, 42 },
		{ 2087, 42 },
		{ 2091, 42 },
		{ 2084, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2091, 42 },
		{ 2980, 46 },
		{ 2981, 46 },
		{ 1631, 1611 },
		{ 1632, 1611 },
		{ 69, 46 },
		{ 2512, 2479 },
		{ 2512, 2479 },
		{ 1484, 1462 },
		{ 1591, 1572 },
		{ 1592, 1573 },
		{ 1593, 1574 },
		{ 2524, 2491 },
		{ 2524, 2491 },
		{ 2538, 2505 },
		{ 2538, 2505 },
		{ 1595, 1575 },
		{ 1597, 1576 },
		{ 1594, 1574 },
		{ 1598, 1577 },
		{ 1599, 1578 },
		{ 1600, 1579 },
		{ 1601, 1580 },
		{ 1596, 1575 },
		{ 2980, 46 },
		{ 1604, 1584 },
		{ 1631, 1611 },
		{ 2539, 2506 },
		{ 2539, 2506 },
		{ 2512, 2479 },
		{ 1605, 1585 },
		{ 1634, 1612 },
		{ 1635, 1612 },
		{ 1637, 1613 },
		{ 1638, 1613 },
		{ 2524, 2491 },
		{ 1606, 1586 },
		{ 2538, 2505 },
		{ 2338, 46 },
		{ 2979, 46 },
		{ 2337, 46 },
		{ 2337, 46 },
		{ 2337, 46 },
		{ 2337, 46 },
		{ 2337, 46 },
		{ 2337, 46 },
		{ 2337, 46 },
		{ 2337, 46 },
		{ 2337, 46 },
		{ 2337, 46 },
		{ 2539, 2506 },
		{ 1607, 1587 },
		{ 1609, 1591 },
		{ 1610, 1592 },
		{ 1634, 1612 },
		{ 1633, 1611 },
		{ 1637, 1613 },
		{ 2339, 46 },
		{ 2335, 46 },
		{ 2330, 46 },
		{ 2339, 46 },
		{ 2327, 46 },
		{ 2334, 46 },
		{ 2332, 46 },
		{ 2339, 46 },
		{ 2336, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2329, 46 },
		{ 2324, 46 },
		{ 2331, 46 },
		{ 2326, 46 },
		{ 2339, 46 },
		{ 2333, 46 },
		{ 2328, 46 },
		{ 2325, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 1636, 1612 },
		{ 2344, 46 },
		{ 1639, 1613 },
		{ 1611, 1593 },
		{ 2339, 46 },
		{ 1612, 1594 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2340, 46 },
		{ 2341, 46 },
		{ 2342, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2343, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 2339, 46 },
		{ 161, 4 },
		{ 162, 4 },
		{ 1640, 1614 },
		{ 1641, 1614 },
		{ 2588, 2556 },
		{ 2588, 2556 },
		{ 2594, 2562 },
		{ 2594, 2562 },
		{ 1613, 1595 },
		{ 1614, 1596 },
		{ 1615, 1597 },
		{ 2601, 2569 },
		{ 2601, 2569 },
		{ 2380, 2348 },
		{ 2380, 2348 },
		{ 1616, 1598 },
		{ 1617, 1599 },
		{ 1618, 1600 },
		{ 1619, 1601 },
		{ 1621, 1604 },
		{ 1622, 1605 },
		{ 1623, 1606 },
		{ 1624, 1607 },
		{ 161, 4 },
		{ 1487, 1463 },
		{ 1640, 1614 },
		{ 1488, 1464 },
		{ 2588, 2556 },
		{ 1492, 1466 },
		{ 2594, 2562 },
		{ 1658, 1644 },
		{ 1659, 1644 },
		{ 1667, 1657 },
		{ 1668, 1657 },
		{ 2601, 2569 },
		{ 1493, 1467 },
		{ 2380, 2348 },
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
		{ 1494, 1468 },
		{ 1643, 1615 },
		{ 1644, 1616 },
		{ 1645, 1617 },
		{ 1658, 1644 },
		{ 1642, 1614 },
		{ 1667, 1657 },
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
		{ 1660, 1644 },
		{ 83, 4 },
		{ 1669, 1657 },
		{ 1647, 1621 },
		{ 87, 4 },
		{ 1648, 1622 },
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
		{ 1472, 23 },
		{ 2615, 2585 },
		{ 2615, 2585 },
		{ 1649, 1623 },
		{ 1459, 23 },
		{ 2617, 2587 },
		{ 2617, 2587 },
		{ 2622, 2592 },
		{ 2622, 2592 },
		{ 2625, 2595 },
		{ 2625, 2595 },
		{ 2629, 2599 },
		{ 2629, 2599 },
		{ 2630, 2600 },
		{ 2630, 2600 },
		{ 2647, 2612 },
		{ 2647, 2612 },
		{ 1650, 1624 },
		{ 1657, 1643 },
		{ 1495, 1469 },
		{ 1661, 1645 },
		{ 1663, 1647 },
		{ 1664, 1648 },
		{ 1472, 23 },
		{ 2615, 2585 },
		{ 1460, 23 },
		{ 1473, 23 },
		{ 1665, 1649 },
		{ 2617, 2587 },
		{ 1491, 1465 },
		{ 2622, 2592 },
		{ 1666, 1650 },
		{ 2625, 2595 },
		{ 1672, 1663 },
		{ 2629, 2599 },
		{ 1673, 1664 },
		{ 2630, 2600 },
		{ 1490, 1465 },
		{ 2647, 2612 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1489, 1465 },
		{ 1497, 1470 },
		{ 1674, 1665 },
		{ 1675, 1666 },
		{ 1496, 1470 },
		{ 1500, 1475 },
		{ 1679, 1673 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1476, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1465, 23 },
		{ 1463, 23 },
		{ 1478, 23 },
		{ 1464, 23 },
		{ 1478, 23 },
		{ 1467, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1466, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1501, 1476 },
		{ 1461, 23 },
		{ 1474, 23 },
		{ 1681, 1675 },
		{ 1468, 23 },
		{ 1684, 1679 },
		{ 1478, 23 },
		{ 1479, 23 },
		{ 1462, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1469, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1477, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1480, 23 },
		{ 1478, 23 },
		{ 1475, 23 },
		{ 1478, 23 },
		{ 1470, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 1478, 23 },
		{ 2057, 40 },
		{ 2406, 2371 },
		{ 2406, 2371 },
		{ 1685, 1681 },
		{ 1882, 40 },
		{ 2651, 2616 },
		{ 2651, 2616 },
		{ 2431, 2396 },
		{ 2431, 2396 },
		{ 2432, 2397 },
		{ 2432, 2397 },
		{ 2450, 2416 },
		{ 2450, 2416 },
		{ 2453, 2419 },
		{ 2453, 2419 },
		{ 1687, 1684 },
		{ 1688, 1685 },
		{ 1689, 1687 },
		{ 1690, 1688 },
		{ 1485, 1689 },
		{ 1502, 1477 },
		{ 1503, 1479 },
		{ 1504, 1480 },
		{ 2057, 40 },
		{ 2406, 2371 },
		{ 1887, 40 },
		{ 1507, 1484 },
		{ 1508, 1487 },
		{ 2651, 2616 },
		{ 1509, 1488 },
		{ 2431, 2396 },
		{ 1510, 1489 },
		{ 2432, 2397 },
		{ 1511, 1490 },
		{ 2450, 2416 },
		{ 1512, 1491 },
		{ 2453, 2419 },
		{ 1513, 1492 },
		{ 2056, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1514, 1493 },
		{ 1897, 40 },
		{ 1515, 1494 },
		{ 1517, 1495 },
		{ 1518, 1496 },
		{ 0, 2616 },
		{ 1516, 1494 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1884, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1892, 40 },
		{ 1890, 40 },
		{ 1900, 40 },
		{ 1891, 40 },
		{ 1900, 40 },
		{ 1894, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1893, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1519, 1497 },
		{ 1888, 40 },
		{ 1522, 1500 },
		{ 1523, 1501 },
		{ 1900, 40 },
		{ 1524, 1502 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1889, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1885, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1886, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1899, 40 },
		{ 1900, 40 },
		{ 1898, 40 },
		{ 1900, 40 },
		{ 1895, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1900, 40 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1525, 1503 },
		{ 1526, 1504 },
		{ 1529, 1507 },
		{ 1530, 1508 },
		{ 1531, 1509 },
		{ 1532, 1510 },
		{ 1533, 1511 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1534, 1512 },
		{ 1535, 1513 },
		{ 1536, 1514 },
		{ 1537, 1515 },
		{ 1485, 1691 },
		{ 1538, 1516 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 1485, 1691 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 1539, 1517 },
		{ 1540, 1518 },
		{ 1541, 1519 },
		{ 1544, 1522 },
		{ 1545, 1523 },
		{ 1546, 1524 },
		{ 1547, 1525 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 1548, 1526 },
		{ 1550, 1529 },
		{ 1551, 1530 },
		{ 1552, 1531 },
		{ 2511, 2577 },
		{ 1553, 1532 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2511, 2577 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2446, 2411 },
		{ 156, 154 },
		{ 116, 101 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 108 },
		{ 125, 109 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 126, 111 },
		{ 127, 112 },
		{ 128, 113 },
		{ 129, 114 },
		{ 2339, 2641 },
		{ 131, 116 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 2339, 2641 },
		{ 1486, 1690 },
		{ 0, 1690 },
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
		{ 2347, 2324 },
		{ 2860, 2860 },
		{ 2349, 2325 },
		{ 2352, 2326 },
		{ 2353, 2327 },
		{ 2362, 2329 },
		{ 2350, 2326 },
		{ 2357, 2328 },
		{ 2364, 2330 },
		{ 2351, 2326 },
		{ 1486, 1690 },
		{ 2356, 2328 },
		{ 2365, 2331 },
		{ 2368, 2333 },
		{ 2354, 2327 },
		{ 2366, 2332 },
		{ 2355, 2327 },
		{ 2361, 2329 },
		{ 2369, 2334 },
		{ 2370, 2335 },
		{ 2371, 2336 },
		{ 2339, 2339 },
		{ 2375, 2340 },
		{ 2363, 2341 },
		{ 2860, 2860 },
		{ 2348, 2342 },
		{ 2358, 2328 },
		{ 2359, 2328 },
		{ 2367, 2332 },
		{ 2360, 2343 },
		{ 2379, 2347 },
		{ 2376, 2341 },
		{ 147, 139 },
		{ 2381, 2349 },
		{ 2382, 2350 },
		{ 2383, 2351 },
		{ 2384, 2352 },
		{ 2385, 2353 },
		{ 2386, 2354 },
		{ 0, 1690 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2387, 2355 },
		{ 2390, 2357 },
		{ 2391, 2358 },
		{ 2392, 2359 },
		{ 2393, 2360 },
		{ 2394, 2361 },
		{ 2395, 2362 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 69, 7 },
		{ 2397, 2363 },
		{ 2398, 2364 },
		{ 2399, 2365 },
		{ 2860, 2860 },
		{ 1691, 1690 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2400, 2366 },
		{ 2401, 2367 },
		{ 2404, 2369 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 2388, 2356 },
		{ 2405, 2370 },
		{ 148, 140 },
		{ 2411, 2375 },
		{ 2396, 2376 },
		{ 2414, 2379 },
		{ 2389, 2356 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 2416, 2381 },
		{ 2417, 2382 },
		{ 2418, 2383 },
		{ 2419, 2384 },
		{ 1299, 7 },
		{ 2420, 2385 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 1299, 7 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 2421, 2386 },
		{ 2422, 2387 },
		{ 2423, 2388 },
		{ 2424, 2389 },
		{ 2425, 2390 },
		{ 2426, 2391 },
		{ 2427, 2392 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 2428, 2393 },
		{ 2429, 2394 },
		{ 2430, 2395 },
		{ 149, 142 },
		{ 0, 1962 },
		{ 2433, 2398 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 1962 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 2434, 2399 },
		{ 2435, 2400 },
		{ 2436, 2401 },
		{ 2437, 2402 },
		{ 2438, 2403 },
		{ 2439, 2404 },
		{ 2440, 2405 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 150, 143 },
		{ 2448, 2414 },
		{ 151, 144 },
		{ 2451, 2417 },
		{ 0, 2156 },
		{ 2452, 2418 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 0, 2156 },
		{ 2402, 2368 },
		{ 2454, 2420 },
		{ 2456, 2421 },
		{ 2457, 2422 },
		{ 2458, 2423 },
		{ 2455, 2420 },
		{ 2459, 2424 },
		{ 152, 146 },
		{ 2461, 2426 },
		{ 2403, 2368 },
		{ 2462, 2427 },
		{ 2464, 2429 },
		{ 2465, 2430 },
		{ 2469, 2433 },
		{ 2470, 2434 },
		{ 2471, 2435 },
		{ 2473, 2437 },
		{ 2474, 2438 },
		{ 2475, 2439 },
		{ 2476, 2440 },
		{ 2479, 2448 },
		{ 2482, 2451 },
		{ 153, 149 },
		{ 154, 150 },
		{ 2487, 2455 },
		{ 2488, 2456 },
		{ 2489, 2457 },
		{ 2490, 2458 },
		{ 2491, 2459 },
		{ 2494, 2462 },
		{ 2496, 2464 },
		{ 2497, 2465 },
		{ 2501, 2469 },
		{ 2502, 2470 },
		{ 2503, 2471 },
		{ 2505, 2473 },
		{ 2506, 2474 },
		{ 155, 152 },
		{ 2508, 2476 },
		{ 2516, 2482 },
		{ 2519, 2487 },
		{ 2520, 2488 },
		{ 2521, 2489 },
		{ 2523, 2490 },
		{ 91, 75 },
		{ 2527, 2494 },
		{ 2529, 2496 },
		{ 2522, 2490 },
		{ 2530, 2497 },
		{ 2534, 2501 },
		{ 2535, 2502 },
		{ 2536, 2503 },
		{ 157, 155 },
		{ 158, 157 },
		{ 2541, 2508 },
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
		{ 2548, 2516 },
		{ 2553, 2520 },
		{ 2554, 2521 },
		{ 2555, 2522 },
		{ 2556, 2523 },
		{ 2560, 2527 },
		{ 2562, 2529 },
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
		{ 2563, 2530 },
		{ 2567, 2534 },
		{ 2568, 2535 },
		{ 2569, 2536 },
		{ 87, 159 },
		{ 2575, 2541 },
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
		{ 2582, 2548 },
		{ 2585, 2553 },
		{ 2586, 2554 },
		{ 2587, 2555 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 2592, 2560 },
		{ 95, 78 },
		{ 2595, 2563 },
		{ 2599, 2567 },
		{ 2600, 2568 },
		{ 93, 76 },
		{ 96, 79 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 2607, 2575 },
		{ 97, 80 },
		{ 2612, 2582 },
		{ 98, 81 },
		{ 1299, 1299 },
		{ 2616, 2586 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 1299, 1299 },
		{ 99, 82 },
		{ 101, 84 },
		{ 106, 91 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 107, 92 },
		{ 108, 93 },
		{ 109, 94 },
		{ 110, 95 },
		{ 111, 96 },
		{ 112, 97 },
		{ 113, 98 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 114, 99 },
		{ 1314, 1311 },
		{ 3056, 51 },
		{ 1314, 1311 },
		{ 0, 1549 },
		{ 0, 3057 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 1549 },
		{ 0, 2607 },
		{ 0, 2607 },
		{ 134, 120 },
		{ 118, 103 },
		{ 134, 120 },
		{ 118, 103 },
		{ 2746, 2715 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 2408, 2373 },
		{ 1309, 1306 },
		{ 2408, 2373 },
		{ 1309, 1306 },
		{ 2731, 2701 },
		{ 2924, 2906 },
		{ 0, 2607 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 2442, 2407 },
		{ 2444, 2410 },
		{ 2442, 2407 },
		{ 2444, 2410 },
		{ 3223, 3223 },
		{ 2927, 2909 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 3223, 3223 },
		{ 88, 51 },
		{ 1364, 1363 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 132, 117 },
		{ 1312, 1308 },
		{ 132, 117 },
		{ 1312, 1308 },
		{ 1838, 1837 },
		{ 2262, 2259 },
		{ 2641, 2607 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 1880, 1879 },
		{ 2905, 2886 },
		{ 3288, 3283 },
		{ 2870, 2846 },
		{ 2795, 2765 },
		{ 3203, 3202 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3283, 3283 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 1790, 1789 },
		{ 2702, 2667 },
		{ 1835, 1834 },
		{ 2824, 2796 },
		{ 1361, 1360 },
		{ 1745, 1744 },
		{ 3300, 3299 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3089, 3088 },
		{ 2129, 2108 },
		{ 3293, 3290 },
		{ 2158, 2140 },
		{ 3142, 3141 },
		{ 2172, 2155 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3290, 3290 },
		{ 3289, 3284 },
		{ 3289, 3284 },
		{ 3289, 3284 },
		{ 3289, 3284 },
		{ 3289, 3284 },
		{ 3289, 3284 },
		{ 3289, 3284 },
		{ 3289, 3284 },
		{ 3289, 3284 },
		{ 3289, 3284 },
		{ 3282, 3278 },
		{ 3282, 3278 },
		{ 3282, 3278 },
		{ 3282, 3278 },
		{ 3282, 3278 },
		{ 3282, 3278 },
		{ 3282, 3278 },
		{ 3282, 3278 },
		{ 3282, 3278 },
		{ 3282, 3278 },
		{ 0, 2478 },
		{ 1938, 1919 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 0, 2478 },
		{ 2114, 2090 },
		{ 184, 175 },
		{ 195, 175 },
		{ 186, 175 },
		{ 1719, 1718 },
		{ 181, 175 },
		{ 191, 175 },
		{ 185, 175 },
		{ 3183, 3182 },
		{ 183, 175 },
		{ 1424, 1423 },
		{ 1719, 1718 },
		{ 0, 3284 },
		{ 193, 175 },
		{ 192, 175 },
		{ 182, 175 },
		{ 190, 175 },
		{ 1424, 1423 },
		{ 187, 175 },
		{ 189, 175 },
		{ 180, 175 },
		{ 2113, 2090 },
		{ 0, 3278 },
		{ 188, 175 },
		{ 194, 175 },
		{ 102, 85 },
		{ 1913, 1889 },
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
		{ 1960, 1941 },
		{ 3206, 3205 },
		{ 1912, 1889 },
		{ 3214, 3213 },
		{ 1851, 1850 },
		{ 2302, 2301 },
		{ 2705, 2670 },
		{ 2307, 2306 },
		{ 2640, 2606 },
		{ 3003, 3002 },
		{ 1793, 1792 },
		{ 103, 85 },
		{ 1305, 1302 },
		{ 2510, 2478 },
		{ 1302, 1302 },
		{ 1302, 1302 },
		{ 1302, 1302 },
		{ 1302, 1302 },
		{ 1302, 1302 },
		{ 1302, 1302 },
		{ 1302, 1302 },
		{ 1302, 1302 },
		{ 1302, 1302 },
		{ 1302, 1302 },
		{ 3282, 3282 },
		{ 3282, 3282 },
		{ 3282, 3282 },
		{ 3282, 3282 },
		{ 3282, 3282 },
		{ 3282, 3282 },
		{ 3282, 3282 },
		{ 3282, 3282 },
		{ 3282, 3282 },
		{ 3282, 3282 },
		{ 3272, 3267 },
		{ 1306, 1302 },
		{ 3043, 3042 },
		{ 3048, 3047 },
		{ 1766, 1765 },
		{ 2099, 2077 },
		{ 2785, 2755 },
		{ 1377, 1376 },
		{ 2372, 2337 },
		{ 103, 85 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 1305, 1305 },
		{ 2590, 2558 },
		{ 2373, 2337 },
		{ 3303, 3302 },
		{ 1306, 1302 },
		{ 3319, 3316 },
		{ 3287, 3282 },
		{ 3325, 3322 },
		{ 2823, 2795 },
		{ 1941, 1922 },
		{ 2825, 2797 },
		{ 2827, 2799 },
		{ 1308, 1305 },
		{ 2316, 2315 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2373, 2337 },
		{ 2407, 2372 },
		{ 2832, 2804 },
		{ 2845, 2821 },
		{ 2846, 2822 },
		{ 3242, 65 },
		{ 2850, 2826 },
		{ 2866, 2842 },
		{ 69, 65 },
		{ 2867, 2843 },
		{ 1308, 1305 },
		{ 2410, 2374 },
		{ 2611, 2581 },
		{ 1307, 1307 },
		{ 1307, 1307 },
		{ 1307, 1307 },
		{ 1307, 1307 },
		{ 1307, 1307 },
		{ 1307, 1307 },
		{ 1307, 1307 },
		{ 1307, 1307 },
		{ 1307, 1307 },
		{ 1307, 1307 },
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
		{ 2407, 2372 },
		{ 1311, 1307 },
		{ 2432, 2432 },
		{ 2432, 2432 },
		{ 1947, 1928 },
		{ 1301, 9 },
		{ 2613, 2583 },
		{ 2671, 2671 },
		{ 2671, 2671 },
		{ 69, 9 },
		{ 2410, 2374 },
		{ 120, 104 },
		{ 2549, 2517 },
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
		{ 2881, 2857 },
		{ 1380, 1379 },
		{ 2432, 2432 },
		{ 3257, 67 },
		{ 2886, 2864 },
		{ 1301, 9 },
		{ 69, 67 },
		{ 2671, 2671 },
		{ 3241, 65 },
		{ 2892, 2870 },
		{ 1311, 1307 },
		{ 117, 102 },
		{ 3240, 65 },
		{ 2894, 2872 },
		{ 2899, 2879 },
		{ 1967, 1948 },
		{ 2906, 2887 },
		{ 2909, 2890 },
		{ 1419, 1418 },
		{ 1303, 9 },
		{ 120, 104 },
		{ 1302, 9 },
		{ 1302, 9 },
		{ 1302, 9 },
		{ 1302, 9 },
		{ 1302, 9 },
		{ 1302, 9 },
		{ 1302, 9 },
		{ 1302, 9 },
		{ 1302, 9 },
		{ 1302, 9 },
		{ 3289, 3289 },
		{ 3289, 3289 },
		{ 3289, 3289 },
		{ 3289, 3289 },
		{ 3289, 3289 },
		{ 3289, 3289 },
		{ 3289, 3289 },
		{ 3289, 3289 },
		{ 3289, 3289 },
		{ 3289, 3289 },
		{ 1994, 1978 },
		{ 2931, 2913 },
		{ 117, 102 },
		{ 3250, 3250 },
		{ 3250, 3250 },
		{ 3250, 3250 },
		{ 3250, 3250 },
		{ 3250, 3250 },
		{ 3250, 3250 },
		{ 3250, 3250 },
		{ 3250, 3250 },
		{ 3250, 3250 },
		{ 3250, 3250 },
		{ 478, 430 },
		{ 483, 430 },
		{ 480, 430 },
		{ 479, 430 },
		{ 482, 430 },
		{ 477, 430 },
		{ 2933, 2915 },
		{ 476, 430 },
		{ 3238, 65 },
		{ 2952, 2944 },
		{ 3239, 65 },
		{ 481, 430 },
		{ 3254, 67 },
		{ 484, 430 },
		{ 2601, 2601 },
		{ 2601, 2601 },
		{ 2622, 2622 },
		{ 2622, 2622 },
		{ 3255, 67 },
		{ 475, 430 },
		{ 2954, 2946 },
		{ 2467, 2432 },
		{ 3292, 3289 },
		{ 2538, 2538 },
		{ 2538, 2538 },
		{ 2596, 2596 },
		{ 2596, 2596 },
		{ 2597, 2597 },
		{ 2597, 2597 },
		{ 3107, 3107 },
		{ 3107, 3107 },
		{ 3252, 67 },
		{ 2468, 2432 },
		{ 3154, 3154 },
		{ 3154, 3154 },
		{ 3251, 3250 },
		{ 2706, 2671 },
		{ 2601, 2601 },
		{ 2962, 2955 },
		{ 2622, 2622 },
		{ 2973, 2971 },
		{ 2608, 2608 },
		{ 2608, 2608 },
		{ 2732, 2732 },
		{ 2732, 2732 },
		{ 1996, 1981 },
		{ 2538, 2538 },
		{ 1998, 1983 },
		{ 2596, 2596 },
		{ 3006, 3005 },
		{ 2597, 2597 },
		{ 3015, 3014 },
		{ 3107, 3107 },
		{ 3256, 67 },
		{ 2848, 2848 },
		{ 2848, 2848 },
		{ 3154, 3154 },
		{ 2849, 2849 },
		{ 2849, 2849 },
		{ 2483, 2483 },
		{ 2483, 2483 },
		{ 2630, 2630 },
		{ 2630, 2630 },
		{ 3028, 3027 },
		{ 2608, 2608 },
		{ 2048, 2046 },
		{ 2732, 2732 },
		{ 2453, 2453 },
		{ 2453, 2453 },
		{ 2647, 2647 },
		{ 2647, 2647 },
		{ 2507, 2507 },
		{ 2507, 2507 },
		{ 3039, 3039 },
		{ 3039, 3039 },
		{ 1795, 1794 },
		{ 2632, 2601 },
		{ 2848, 2848 },
		{ 2633, 2601 },
		{ 1817, 1816 },
		{ 2849, 2849 },
		{ 3051, 3050 },
		{ 2483, 2483 },
		{ 2564, 2531 },
		{ 2630, 2630 },
		{ 2565, 2532 },
		{ 1830, 1829 },
		{ 2635, 2601 },
		{ 3067, 3067 },
		{ 3067, 3067 },
		{ 2453, 2453 },
		{ 1365, 1364 },
		{ 2647, 2647 },
		{ 2518, 2484 },
		{ 2507, 2507 },
		{ 3079, 3078 },
		{ 3039, 3039 },
		{ 2524, 2524 },
		{ 2524, 2524 },
		{ 2460, 2460 },
		{ 2460, 2460 },
		{ 2625, 2625 },
		{ 2625, 2625 },
		{ 1714, 1713 },
		{ 2631, 2601 },
		{ 2629, 2629 },
		{ 2629, 2629 },
		{ 3106, 3105 },
		{ 2634, 2601 },
		{ 1839, 1838 },
		{ 2654, 2619 },
		{ 3067, 3067 },
		{ 3136, 3135 },
		{ 2170, 2153 },
		{ 2594, 2594 },
		{ 2594, 2594 },
		{ 2309, 2309 },
		{ 2309, 2309 },
		{ 2658, 2622 },
		{ 2657, 2622 },
		{ 2524, 2524 },
		{ 3145, 3144 },
		{ 2460, 2460 },
		{ 3153, 3152 },
		{ 2625, 2625 },
		{ 2572, 2538 },
		{ 2571, 2538 },
		{ 2580, 2546 },
		{ 2629, 2629 },
		{ 2561, 2561 },
		{ 2561, 2561 },
		{ 2171, 2154 },
		{ 2626, 2596 },
		{ 3177, 3176 },
		{ 2627, 2597 },
		{ 2642, 2608 },
		{ 3108, 3107 },
		{ 2594, 2594 },
		{ 2584, 2550 },
		{ 2309, 2309 },
		{ 3155, 3154 },
		{ 2380, 2380 },
		{ 2380, 2380 },
		{ 2643, 2608 },
		{ 3186, 3185 },
		{ 2853, 2853 },
		{ 2853, 2853 },
		{ 2617, 2617 },
		{ 2617, 2617 },
		{ 2763, 2732 },
		{ 3197, 3196 },
		{ 1442, 1441 },
		{ 2561, 2561 },
		{ 1854, 1853 },
		{ 2742, 2742 },
		{ 2742, 2742 },
		{ 3068, 3067 },
		{ 3076, 3076 },
		{ 3076, 3076 },
		{ 3208, 3207 },
		{ 2872, 2848 },
		{ 2463, 2463 },
		{ 2463, 2463 },
		{ 2873, 2849 },
		{ 2380, 2380 },
		{ 2517, 2483 },
		{ 2690, 2655 },
		{ 2666, 2630 },
		{ 2853, 2853 },
		{ 3217, 3216 },
		{ 2617, 2617 },
		{ 3121, 3121 },
		{ 3121, 3121 },
		{ 2484, 2453 },
		{ 2189, 2176 },
		{ 2683, 2647 },
		{ 2540, 2507 },
		{ 2742, 2742 },
		{ 3040, 3039 },
		{ 2202, 2187 },
		{ 3076, 3076 },
		{ 2203, 2188 },
		{ 2800, 2800 },
		{ 2800, 2800 },
		{ 2463, 2463 },
		{ 2713, 2680 },
		{ 3162, 3162 },
		{ 3162, 3162 },
		{ 3068, 3067 },
		{ 2512, 2512 },
		{ 2512, 2512 },
		{ 2729, 2699 },
		{ 2772, 2772 },
		{ 2772, 2772 },
		{ 3121, 3121 },
		{ 1453, 1452 },
		{ 2739, 2708 },
		{ 3267, 3262 },
		{ 2263, 2260 },
		{ 2750, 2719 },
		{ 2557, 2524 },
		{ 2278, 2277 },
		{ 2492, 2460 },
		{ 2765, 2734 },
		{ 2661, 2625 },
		{ 2800, 2800 },
		{ 1331, 1330 },
		{ 1767, 1766 },
		{ 2665, 2629 },
		{ 3162, 3162 },
		{ 2793, 2763 },
		{ 2304, 2303 },
		{ 2512, 2512 },
		{ 1769, 1768 },
		{ 3305, 3304 },
		{ 2772, 2772 },
		{ 3307, 3307 },
		{ 2624, 2594 },
		{ 2808, 2779 },
		{ 2310, 2309 },
		{ 2812, 2783 },
		{ 3332, 3330 },
		{ 2450, 2450 },
		{ 2450, 2450 },
		{ 2109, 2084 },
		{ 1920, 1895 },
		{ 2009, 1996 },
		{ 2108, 2084 },
		{ 1919, 1895 },
		{ 2028, 2018 },
		{ 2036, 2028 },
		{ 1335, 1334 },
		{ 2593, 2561 },
		{ 2977, 2976 },
		{ 1671, 1662 },
		{ 2998, 2997 },
		{ 1388, 1387 },
		{ 2714, 2682 },
		{ 3010, 3009 },
		{ 3307, 3307 },
		{ 1811, 1810 },
		{ 1812, 1811 },
		{ 1395, 1394 },
		{ 2415, 2380 },
		{ 1821, 1820 },
		{ 2450, 2450 },
		{ 2130, 2109 },
		{ 2877, 2853 },
		{ 2754, 2723 },
		{ 2652, 2617 },
		{ 2149, 2128 },
		{ 2151, 2130 },
		{ 2770, 2739 },
		{ 3062, 3060 },
		{ 2605, 2573 },
		{ 2154, 2133 },
		{ 2773, 2742 },
		{ 1396, 1395 },
		{ 3083, 3082 },
		{ 3077, 3076 },
		{ 2609, 2579 },
		{ 2796, 2766 },
		{ 1398, 1397 },
		{ 2495, 2463 },
		{ 1677, 1671 },
		{ 1707, 1706 },
		{ 2614, 2584 },
		{ 1847, 1846 },
		{ 1708, 1707 },
		{ 1412, 1411 },
		{ 1870, 1869 },
		{ 1871, 1870 },
		{ 2833, 2805 },
		{ 3122, 3121 },
		{ 2834, 2806 },
		{ 3196, 3195 },
		{ 2836, 2809 },
		{ 2837, 2812 },
		{ 1876, 1875 },
		{ 1413, 1412 },
		{ 1735, 1734 },
		{ 1736, 1735 },
		{ 1742, 1741 },
		{ 2869, 2845 },
		{ 2828, 2800 },
		{ 1743, 1742 },
		{ 1939, 1920 },
		{ 1341, 1340 },
		{ 3163, 3162 },
		{ 2319, 2318 },
		{ 1946, 1927 },
		{ 2544, 2512 },
		{ 2802, 2772 },
		{ 2882, 2858 },
		{ 2648, 2613 },
		{ 1761, 1760 },
		{ 1958, 1939 },
		{ 1762, 1761 },
		{ 1373, 1372 },
		{ 1320, 1319 },
		{ 1448, 1447 },
		{ 3295, 3294 },
		{ 3296, 3295 },
		{ 1785, 1784 },
		{ 3310, 3307 },
		{ 2919, 2900 },
		{ 2667, 2631 },
		{ 2674, 2638 },
		{ 1786, 1785 },
		{ 3309, 3307 },
		{ 2682, 2645 },
		{ 3308, 3307 },
		{ 2943, 2931 },
		{ 1771, 1770 },
		{ 1721, 1720 },
		{ 2637, 2603 },
		{ 1444, 1443 },
		{ 3026, 3025 },
		{ 2809, 2780 },
		{ 3037, 3036 },
		{ 1329, 1328 },
		{ 2481, 2450 },
		{ 2813, 2784 },
		{ 2256, 2249 },
		{ 2259, 2254 },
		{ 1740, 1739 },
		{ 1450, 1449 },
		{ 2275, 2272 },
		{ 1797, 1796 },
		{ 3066, 3064 },
		{ 2043, 2038 },
		{ 2660, 2624 },
		{ 1363, 1362 },
		{ 2662, 2626 },
		{ 2663, 2627 },
		{ 1322, 1321 },
		{ 2306, 2305 },
		{ 2855, 2831 },
		{ 3134, 3133 },
		{ 2859, 2835 },
		{ 1747, 1746 },
		{ 1931, 1912 },
		{ 2312, 2311 },
		{ 1390, 1389 },
		{ 2500, 2468 },
		{ 3175, 3174 },
		{ 2318, 2317 },
		{ 2696, 2661 },
		{ 2700, 2665 },
		{ 1823, 1822 },
		{ 2120, 2099 },
		{ 2885, 2863 },
		{ 1343, 1342 },
		{ 1942, 1923 },
		{ 2141, 2120 },
		{ 2896, 2876 },
		{ 2722, 2692 },
		{ 2559, 2526 },
		{ 1832, 1831 },
		{ 1426, 1425 },
		{ 2914, 2895 },
		{ 2740, 2709 },
		{ 1950, 1931 },
		{ 2749, 2718 },
		{ 2619, 2589 },
		{ 3276, 3272 },
		{ 2480, 2480 },
		{ 2480, 2480 },
		{ 2934, 2916 },
		{ 2935, 2918 },
		{ 2623, 2593 },
		{ 1837, 1836 },
		{ 2953, 2945 },
		{ 2768, 2737 },
		{ 1440, 1439 },
		{ 2965, 2961 },
		{ 2969, 2966 },
		{ 1357, 1356 },
		{ 3307, 3306 },
		{ 2774, 2743 },
		{ 3315, 3312 },
		{ 1977, 1960 },
		{ 3323, 3320 },
		{ 2570, 2537 },
		{ 3000, 2999 },
		{ 3335, 3334 },
		{ 2948, 2948 },
		{ 2948, 2948 },
		{ 2734, 2703 },
		{ 2480, 2480 },
		{ 2539, 2539 },
		{ 2539, 2539 },
		{ 2431, 2431 },
		{ 2431, 2431 },
		{ 2843, 2818 },
		{ 2558, 2525 },
		{ 2140, 2119 },
		{ 2687, 2652 },
		{ 2897, 2877 },
		{ 2852, 2828 },
		{ 1816, 1815 },
		{ 1877, 1876 },
		{ 2699, 2664 },
		{ 3078, 3077 },
		{ 2736, 2705 },
		{ 2826, 2798 },
		{ 2784, 2754 },
		{ 1452, 1451 },
		{ 1981, 1966 },
		{ 2948, 2948 },
		{ 1438, 1437 },
		{ 2680, 2643 },
		{ 2709, 2674 },
		{ 2539, 2539 },
		{ 2803, 2773 },
		{ 2431, 2431 },
		{ 2944, 2932 },
		{ 2842, 2817 },
		{ 2045, 2042 },
		{ 1909, 1886 },
		{ 1443, 1442 },
		{ 2051, 2050 },
		{ 3053, 3052 },
		{ 1924, 1901 },
		{ 1562, 1542 },
		{ 1382, 1381 },
		{ 1505, 1481 },
		{ 2726, 2696 },
		{ 2513, 2480 },
		{ 1908, 1886 },
		{ 1729, 1728 },
		{ 2730, 2700 },
		{ 2115, 2092 },
		{ 2119, 2098 },
		{ 3081, 3080 },
		{ 2880, 2856 },
		{ 3088, 3087 },
		{ 1770, 1769 },
		{ 1943, 1924 },
		{ 1945, 1926 },
		{ 2134, 2113 },
		{ 2136, 2115 },
		{ 2138, 2117 },
		{ 3138, 3137 },
		{ 2751, 2720 },
		{ 1437, 1436 },
		{ 3147, 3146 },
		{ 1779, 1778 },
		{ 1333, 1332 },
		{ 1527, 1505 },
		{ 1741, 1740 },
		{ 1356, 1355 },
		{ 3179, 3178 },
		{ 2733, 2703 },
		{ 2169, 2152 },
		{ 2918, 2899 },
		{ 3188, 3187 },
		{ 2780, 2750 },
		{ 2920, 2901 },
		{ 2514, 2480 },
		{ 1976, 1959 },
		{ 1856, 1855 },
		{ 1864, 1863 },
		{ 3210, 3209 },
		{ 1406, 1405 },
		{ 2184, 2168 },
		{ 3219, 3218 },
		{ 2797, 2767 },
		{ 2659, 2623 },
		{ 1991, 1975 },
		{ 2945, 2933 },
		{ 1796, 1795 },
		{ 2525, 2492 },
		{ 1744, 1743 },
		{ 2589, 2557 },
		{ 2961, 2954 },
		{ 3273, 3268 },
		{ 2234, 2216 },
		{ 2591, 2559 },
		{ 2235, 2217 },
		{ 1805, 1804 },
		{ 2975, 2974 },
		{ 2955, 2948 },
		{ 1701, 1700 },
		{ 2261, 2258 },
		{ 2830, 2802 },
		{ 2573, 2539 },
		{ 2537, 2504 },
		{ 2466, 2431 },
		{ 2685, 2650 },
		{ 1455, 1454 },
		{ 3008, 3007 },
		{ 3306, 3305 },
		{ 2835, 2808 },
		{ 1815, 1814 },
		{ 3312, 3309 },
		{ 2695, 2660 },
		{ 2269, 2266 },
		{ 3030, 3029 },
		{ 3036, 3035 },
		{ 2272, 2270 },
		{ 3334, 3332 },
		{ 1755, 1754 },
		{ 3021, 3021 },
		{ 3021, 3021 },
		{ 3170, 3170 },
		{ 3170, 3170 },
		{ 2615, 2615 },
		{ 2615, 2615 },
		{ 2588, 2588 },
		{ 2588, 2588 },
		{ 3129, 3129 },
		{ 3129, 3129 },
		{ 2052, 2051 },
		{ 1842, 1841 },
		{ 1383, 1382 },
		{ 1405, 1404 },
		{ 2271, 2269 },
		{ 1852, 1851 },
		{ 3082, 3081 },
		{ 2498, 2466 },
		{ 2499, 2467 },
		{ 1754, 1753 },
		{ 3090, 3089 },
		{ 3099, 3098 },
		{ 1804, 1803 },
		{ 3021, 3021 },
		{ 2117, 2095 },
		{ 3170, 3170 },
		{ 3116, 3115 },
		{ 2615, 2615 },
		{ 3117, 3116 },
		{ 2588, 2588 },
		{ 3119, 3118 },
		{ 3129, 3129 },
		{ 2900, 2880 },
		{ 2118, 2097 },
		{ 3132, 3131 },
		{ 1948, 1929 },
		{ 1857, 1856 },
		{ 1863, 1862 },
		{ 3139, 3138 },
		{ 2308, 2307 },
		{ 3143, 3142 },
		{ 1712, 1711 },
		{ 2781, 2751 },
		{ 3148, 3147 },
		{ 2783, 2753 },
		{ 2131, 2110 },
		{ 3160, 3159 },
		{ 1435, 1434 },
		{ 1582, 1563 },
		{ 3173, 3172 },
		{ 115, 100 },
		{ 1978, 1961 },
		{ 2937, 2920 },
		{ 3180, 3179 },
		{ 1728, 1727 },
		{ 3184, 3183 },
		{ 2801, 2771 },
		{ 1378, 1377 },
		{ 3189, 3188 },
		{ 3195, 3194 },
		{ 1983, 1968 },
		{ 2153, 2132 },
		{ 2412, 2377 },
		{ 2669, 2633 },
		{ 2670, 2634 },
		{ 2672, 2636 },
		{ 3211, 3210 },
		{ 1368, 1367 },
		{ 3215, 3214 },
		{ 1993, 1977 },
		{ 1506, 1483 },
		{ 3220, 3219 },
		{ 1778, 1777 },
		{ 2976, 2975 },
		{ 1456, 1455 },
		{ 3235, 3231 },
		{ 1417, 1416 },
		{ 2017, 2005 },
		{ 2688, 2653 },
		{ 2176, 2159 },
		{ 3264, 3259 },
		{ 3004, 3003 },
		{ 3268, 3263 },
		{ 1926, 1904 },
		{ 2030, 2021 },
		{ 3009, 3008 },
		{ 3280, 3276 },
		{ 1930, 1911 },
		{ 2610, 2580 },
		{ 2042, 2037 },
		{ 3024, 3023 },
		{ 2703, 2668 },
		{ 2214, 2200 },
		{ 1542, 1520 },
		{ 3022, 3021 },
		{ 3031, 3030 },
		{ 3171, 3170 },
		{ 2708, 2673 },
		{ 2650, 2615 },
		{ 1700, 1699 },
		{ 2618, 2588 },
		{ 2710, 2675 },
		{ 3130, 3129 },
		{ 2862, 2838 },
		{ 2712, 2679 },
		{ 3049, 3048 },
		{ 2236, 2218 },
		{ 2247, 2234 },
		{ 3054, 3053 },
		{ 2248, 2235 },
		{ 1334, 1333 },
		{ 2258, 2253 },
		{ 1940, 1921 },
		{ 3063, 3061 },
		{ 2911, 2911 },
		{ 2911, 2911 },
		{ 2891, 2891 },
		{ 2891, 2891 },
		{ 1849, 1849 },
		{ 1849, 1849 },
		{ 1375, 1375 },
		{ 1375, 1375 },
		{ 2493, 2493 },
		{ 2493, 2493 },
		{ 2951, 2951 },
		{ 2951, 2951 },
		{ 3046, 3046 },
		{ 3046, 3046 },
		{ 2656, 2656 },
		{ 2656, 2656 },
		{ 3140, 3140 },
		{ 3140, 3140 },
		{ 2406, 2406 },
		{ 2406, 2406 },
		{ 3001, 3001 },
		{ 3001, 3001 },
		{ 1407, 1406 },
		{ 2911, 2911 },
		{ 2775, 2744 },
		{ 2891, 2891 },
		{ 1780, 1779 },
		{ 1849, 1849 },
		{ 1819, 1818 },
		{ 1375, 1375 },
		{ 2314, 2313 },
		{ 2493, 2493 },
		{ 1865, 1864 },
		{ 2951, 2951 },
		{ 2621, 2591 },
		{ 3046, 3046 },
		{ 2185, 2169 },
		{ 2656, 2656 },
		{ 2264, 2261 },
		{ 3140, 3140 },
		{ 2547, 2515 },
		{ 2406, 2406 },
		{ 1581, 1562 },
		{ 3001, 3001 },
		{ 2847, 2847 },
		{ 2847, 2847 },
		{ 3181, 3181 },
		{ 3181, 3181 },
		{ 3212, 3212 },
		{ 3212, 3212 },
		{ 1756, 1755 },
		{ 1806, 1805 },
		{ 1730, 1729 },
		{ 1992, 1976 },
		{ 2280, 2279 },
		{ 2157, 2138 },
		{ 2764, 2733 },
		{ 3322, 3319 },
		{ 2047, 2045 },
		{ 1702, 1701 },
		{ 1765, 1764 },
		{ 1964, 1945 },
		{ 3277, 3273 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 1359, 1359 },
		{ 1359, 1359 },
		{ 2847, 2847 },
		{ 2898, 2878 },
		{ 3181, 3181 },
		{ 1878, 1877 },
		{ 3212, 3212 },
		{ 2175, 2158 },
		{ 1370, 1370 },
		{ 1370, 1370 },
		{ 3201, 3201 },
		{ 3201, 3201 },
		{ 3165, 3165 },
		{ 3165, 3165 },
		{ 1844, 1844 },
		{ 1844, 1844 },
		{ 3124, 3124 },
		{ 3124, 3124 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 1846, 1845 },
		{ 3158, 3158 },
		{ 2752, 2721 },
		{ 1359, 1359 },
		{ 1833, 1833 },
		{ 1833, 1833 },
		{ 2303, 2302 },
		{ 2509, 2477 },
		{ 2929, 2911 },
		{ 2912, 2893 },
		{ 2910, 2891 },
		{ 1370, 1370 },
		{ 1850, 1849 },
		{ 3201, 3201 },
		{ 1376, 1375 },
		{ 3165, 3165 },
		{ 2526, 2493 },
		{ 1844, 1844 },
		{ 2958, 2951 },
		{ 3124, 3124 },
		{ 3047, 3046 },
		{ 3016, 3016 },
		{ 2691, 2656 },
		{ 2008, 1995 },
		{ 3141, 3140 },
		{ 1717, 1716 },
		{ 2441, 2406 },
		{ 1833, 1833 },
		{ 3002, 3001 },
		{ 2840, 2815 },
		{ 2767, 2736 },
		{ 2128, 2107 },
		{ 2639, 2605 },
		{ 1393, 1392 },
		{ 2020, 2010 },
		{ 1951, 1932 },
		{ 2233, 2215 },
		{ 1792, 1791 },
		{ 1410, 1409 },
		{ 2940, 2928 },
		{ 2865, 2841 },
		{ 1962, 1943 },
		{ 1646, 1620 },
		{ 2950, 2941 },
		{ 2868, 2844 },
		{ 3291, 3286 },
		{ 1705, 1704 },
		{ 1927, 1907 },
		{ 1422, 1421 },
		{ 1733, 1732 },
		{ 1372, 1371 },
		{ 3302, 3301 },
		{ 2871, 2847 },
		{ 3074, 3073 },
		{ 3182, 3181 },
		{ 2155, 2134 },
		{ 3213, 3212 },
		{ 2156, 2136 },
		{ 2967, 2964 },
		{ 1549, 1527 },
		{ 2268, 2265 },
		{ 1868, 1867 },
		{ 1809, 1808 },
		{ 2620, 2590 },
		{ 3205, 3204 },
		{ 1759, 1758 },
		{ 3328, 3325 },
		{ 1783, 1782 },
		{ 1327, 1326 },
		{ 2747, 2716 },
		{ 3159, 3158 },
		{ 2602, 2570 },
		{ 1360, 1359 },
		{ 3070, 3069 },
		{ 2758, 2758 },
		{ 2758, 2758 },
		{ 2759, 2759 },
		{ 2759, 2759 },
		{ 1323, 1323 },
		{ 1323, 1323 },
		{ 1371, 1370 },
		{ 3097, 3096 },
		{ 3202, 3201 },
		{ 1901, 2057 },
		{ 3166, 3165 },
		{ 2092, 2283 },
		{ 1845, 1844 },
		{ 1481, 1472 },
		{ 3125, 3124 },
		{ 2581, 2547 },
		{ 3017, 3016 },
		{ 2598, 2598 },
		{ 2598, 2598 },
		{ 2701, 2666 },
		{ 2531, 2498 },
		{ 2532, 2499 },
		{ 1834, 1833 },
		{ 2758, 2758 },
		{ 1423, 1422 },
		{ 2759, 2759 },
		{ 2636, 2602 },
		{ 1323, 1323 },
		{ 1310, 1310 },
		{ 1310, 1310 },
		{ 1310, 1310 },
		{ 1310, 1310 },
		{ 1310, 1310 },
		{ 1310, 1310 },
		{ 1310, 1310 },
		{ 1310, 1310 },
		{ 1310, 1310 },
		{ 1310, 1310 },
		{ 2887, 2865 },
		{ 3178, 3177 },
		{ 2598, 2598 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2409, 2409 },
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
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 1315, 1315 },
		{ 1315, 1315 },
		{ 1315, 1315 },
		{ 1315, 1315 },
		{ 1315, 1315 },
		{ 1315, 1315 },
		{ 1315, 1315 },
		{ 1315, 1315 },
		{ 1315, 1315 },
		{ 1315, 1315 },
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
		{ 2788, 2758 },
		{ 0, 1303 },
		{ 2789, 2759 },
		{ 0, 86 },
		{ 1324, 1323 },
		{ 3226, 3226 },
		{ 3226, 3226 },
		{ 3226, 3226 },
		{ 3226, 3226 },
		{ 3226, 3226 },
		{ 3226, 3226 },
		{ 3226, 3226 },
		{ 3226, 3226 },
		{ 3226, 3226 },
		{ 3226, 3226 },
		{ 1391, 1391 },
		{ 1391, 1391 },
		{ 2628, 2598 },
		{ 2745, 2745 },
		{ 2745, 2745 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 0, 2338 },
		{ 2890, 2868 },
		{ 1416, 1415 },
		{ 0, 1303 },
		{ 3029, 3028 },
		{ 0, 86 },
		{ 2893, 2871 },
		{ 2186, 2170 },
		{ 1391, 1391 },
		{ 2798, 2768 },
		{ 3187, 3186 },
		{ 2745, 2745 },
		{ 2445, 2445 },
		{ 2445, 2445 },
		{ 2445, 2445 },
		{ 2445, 2445 },
		{ 2445, 2445 },
		{ 2445, 2445 },
		{ 2445, 2445 },
		{ 2445, 2445 },
		{ 2445, 2445 },
		{ 2445, 2445 },
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
		{ 0, 2338 },
		{ 3092, 3092 },
		{ 3092, 3092 },
		{ 3327, 3327 },
		{ 3111, 3111 },
		{ 3111, 3111 },
		{ 2724, 2724 },
		{ 2724, 2724 },
		{ 3032, 3032 },
		{ 3032, 3032 },
		{ 1482, 1461 },
		{ 2200, 2184 },
		{ 1332, 1331 },
		{ 2805, 2775 },
		{ 2901, 2881 },
		{ 2005, 1991 },
		{ 2715, 2683 },
		{ 2719, 2688 },
		{ 3052, 3051 },
		{ 2720, 2690 },
		{ 3209, 3208 },
		{ 2815, 2786 },
		{ 2721, 2691 },
		{ 1434, 1433 },
		{ 3092, 3092 },
		{ 2216, 2202 },
		{ 3327, 3327 },
		{ 3111, 3111 },
		{ 2217, 2203 },
		{ 2724, 2724 },
		{ 1418, 1417 },
		{ 3032, 3032 },
		{ 3218, 3217 },
		{ 2546, 2514 },
		{ 2928, 2910 },
		{ 1875, 1874 },
		{ 3073, 3072 },
		{ 1451, 1450 },
		{ 2932, 2914 },
		{ 1711, 1710 },
		{ 1841, 1840 },
		{ 3080, 3079 },
		{ 2744, 2713 },
		{ 1392, 1391 },
		{ 1367, 1366 },
		{ 1713, 1712 },
		{ 2776, 2745 },
		{ 2941, 2929 },
		{ 2838, 2813 },
		{ 1563, 1543 },
		{ 2841, 2816 },
		{ 1910, 1888 },
		{ 3098, 3097 },
		{ 1715, 1714 },
		{ 2844, 2820 },
		{ 1420, 1419 },
		{ 2477, 2441 },
		{ 3115, 3114 },
		{ 2515, 2481 },
		{ 2050, 2048 },
		{ 3118, 3117 },
		{ 1718, 1717 },
		{ 2964, 2958 },
		{ 2856, 2832 },
		{ 2266, 2263 },
		{ 1818, 1817 },
		{ 2864, 2840 },
		{ 2974, 2973 },
		{ 3137, 3136 },
		{ 1454, 1453 },
		{ 1928, 1908 },
		{ 2771, 2740 },
		{ 1855, 1854 },
		{ 2096, 2074 },
		{ 2277, 2275 },
		{ 3315, 3315 },
		{ 3146, 3145 },
		{ 1483, 1461 },
		{ 1982, 1967 },
		{ 1820, 1819 },
		{ 1932, 1913 },
		{ 1381, 1380 },
		{ 3007, 3006 },
		{ 2851, 2827 },
		{ 3156, 3155 },
		{ 2760, 2729 },
		{ 1366, 1365 },
		{ 2279, 2278 },
		{ 3094, 3093 },
		{ 3169, 3168 },
		{ 69, 5 },
		{ 3093, 3092 },
		{ 1840, 1839 },
		{ 3112, 3111 },
		{ 3327, 3324 },
		{ 2755, 2724 },
		{ 3329, 3327 },
		{ 3033, 3032 },
		{ 3315, 3315 },
		{ 2888, 2866 },
		{ 3128, 3127 },
		{ 2889, 2867 },
		{ 3020, 3019 },
		{ 3109, 3108 },
		{ 3114, 3113 },
		{ 1369, 1368 },
		{ 2884, 2862 },
		{ 1433, 1432 },
		{ 1843, 1842 },
		{ 3100, 3099 },
		{ 3120, 3119 },
		{ 2707, 2672 },
		{ 2694, 2659 },
		{ 2829, 2801 },
		{ 2270, 2268 },
		{ 3060, 3058 },
		{ 1921, 1898 },
		{ 3258, 3252 },
		{ 1911, 1888 },
		{ 2693, 2658 },
		{ 1922, 1898 },
		{ 2094, 2071 },
		{ 1355, 1354 },
		{ 2528, 2495 },
		{ 3123, 3122 },
		{ 1829, 1828 },
		{ 2093, 2071 },
		{ 3164, 3163 },
		{ 2059, 2056 },
		{ 205, 181 },
		{ 2604, 2572 },
		{ 3061, 3058 },
		{ 203, 181 },
		{ 2058, 2056 },
		{ 204, 181 },
		{ 2723, 2693 },
		{ 2839, 2814 },
		{ 3035, 3034 },
		{ 2814, 2785 },
		{ 1903, 1883 },
		{ 2097, 2074 },
		{ 206, 181 },
		{ 176, 5 },
		{ 202, 181 },
		{ 1902, 1883 },
		{ 3072, 3071 },
		{ 177, 5 },
		{ 3259, 3252 },
		{ 2638, 2604 },
		{ 2285, 2282 },
		{ 1543, 1521 },
		{ 2449, 2415 },
		{ 2578, 2544 },
		{ 178, 5 },
		{ 2284, 2282 },
		{ 2300, 2299 },
		{ 1499, 1473 },
		{ 3113, 3112 },
		{ 2668, 2632 },
		{ 2794, 2764 },
		{ 2930, 2912 },
		{ 1389, 1388 },
		{ 1836, 1835 },
		{ 2174, 2157 },
		{ 2673, 2637 },
		{ 2021, 2011 },
		{ 1731, 1730 },
		{ 3318, 3315 },
		{ 2679, 2642 },
		{ 175, 5 },
		{ 1781, 1780 },
		{ 1425, 1424 },
		{ 3131, 3130 },
		{ 2037, 2029 },
		{ 3133, 3132 },
		{ 2038, 2030 },
		{ 2949, 2940 },
		{ 1929, 1909 },
		{ 1408, 1407 },
		{ 1699, 1698 },
		{ 2817, 2788 },
		{ 2818, 2789 },
		{ 2957, 2950 },
		{ 1789, 1788 },
		{ 1432, 1431 },
		{ 2049, 2047 },
		{ 2218, 2204 },
		{ 1358, 1357 },
		{ 2966, 2962 },
		{ 2697, 2662 },
		{ 3157, 3156 },
		{ 2698, 2663 },
		{ 2970, 2967 },
		{ 1325, 1324 },
		{ 2831, 2803 },
		{ 2377, 2344 },
		{ 1703, 1702 },
		{ 1848, 1847 },
		{ 3172, 3171 },
		{ 1354, 1353 },
		{ 3174, 3173 },
		{ 1362, 1361 },
		{ 2999, 2998 },
		{ 2249, 2236 },
		{ 2253, 2246 },
		{ 1803, 1802 },
		{ 1746, 1745 },
		{ 2095, 2072 },
		{ 2711, 2676 },
		{ 100, 83 },
		{ 1753, 1752 },
		{ 1807, 1806 },
		{ 2533, 2500 },
		{ 2718, 2687 },
		{ 1957, 1938 },
		{ 1862, 1861 },
		{ 3198, 3197 },
		{ 3023, 3022 },
		{ 2267, 2264 },
		{ 3025, 3024 },
		{ 1520, 1498 },
		{ 2858, 2834 },
		{ 1961, 1942 },
		{ 1521, 1499 },
		{ 2863, 2839 },
		{ 2727, 2697 },
		{ 3034, 3033 },
		{ 2728, 2698 },
		{ 1439, 1438 },
		{ 1965, 1946 },
		{ 3038, 3037 },
		{ 2542, 2509 },
		{ 3041, 3040 },
		{ 3231, 3222 },
		{ 1866, 1865 },
		{ 3045, 3044 },
		{ 1968, 1950 },
		{ 1757, 1756 },
		{ 2132, 2111 },
		{ 2743, 2712 },
		{ 2876, 2852 },
		{ 2133, 2112 },
		{ 2878, 2854 },
		{ 3262, 3256 },
		{ 3263, 3258 },
		{ 1397, 1396 },
		{ 1321, 1320 },
		{ 1404, 1403 },
		{ 2299, 2298 },
		{ 1980, 1964 },
		{ 1342, 1341 },
		{ 3064, 3062 },
		{ 2753, 2722 },
		{ 3281, 3277 },
		{ 1447, 1446 },
		{ 2150, 2129 },
		{ 3071, 3070 },
		{ 1720, 1719 },
		{ 2762, 2731 },
		{ 2645, 2610 },
		{ 3075, 3074 },
		{ 1822, 1821 },
		{ 1602, 1581 },
		{ 3299, 3298 },
		{ 1727, 1726 },
		{ 1904, 1884 },
		{ 1905, 1885 },
		{ 1906, 1885 },
		{ 2653, 2618 },
		{ 2315, 2314 },
		{ 2566, 2533 },
		{ 1603, 1582 },
		{ 3311, 3308 },
		{ 2317, 2316 },
		{ 2777, 2746 },
		{ 2159, 2141 },
		{ 3095, 3094 },
		{ 3320, 3317 },
		{ 2167, 2149 },
		{ 2782, 2752 },
		{ 2916, 2897 },
		{ 1831, 1830 },
		{ 2664, 2628 },
		{ 1777, 1776 },
		{ 2574, 2540 },
		{ 3110, 3109 },
		{ 1374, 1373 },
		{ 1528, 1506 },
		{ 2704, 2669 },
		{ 2139, 2118 },
		{ 2741, 2710 },
		{ 2111, 2088 },
		{ 3269, 3264 },
		{ 1446, 1445 },
		{ 130, 115 },
		{ 3161, 3160 },
		{ 1874, 1873 },
		{ 3317, 3314 },
		{ 1959, 1940 },
		{ 2606, 2574 },
		{ 2447, 2412 },
		{ 3042, 3041 },
		{ 3200, 3199 },
		{ 3065, 3063 },
		{ 2152, 2131 },
		{ 3237, 3235 },
		{ 3091, 3090 },
		{ 2854, 2830 },
		{ 1949, 1930 },
		{ 1969, 1951 },
		{ 3127, 3126 },
		{ 1739, 1738 },
		{ 1853, 1852 },
		{ 2676, 2640 },
		{ 1907, 1885 },
		{ 3216, 3215 },
		{ 3185, 3184 },
		{ 2895, 2873 },
		{ 3005, 3004 },
		{ 3313, 3310 },
		{ 3044, 3043 },
		{ 3316, 3313 },
		{ 1997, 1982 },
		{ 1379, 1378 },
		{ 2766, 2735 },
		{ 3168, 3167 },
		{ 2737, 2706 },
		{ 3050, 3049 },
		{ 3324, 3321 },
		{ 1449, 1448 },
		{ 653, 591 },
		{ 3144, 3143 },
		{ 3096, 3095 },
		{ 3331, 3329 },
		{ 1326, 1325 },
		{ 3019, 3018 },
		{ 2201, 2186 },
		{ 2757, 2726 },
		{ 1481, 1474 },
		{ 2761, 2730 },
		{ 654, 591 },
		{ 2092, 2086 },
		{ 2786, 2756 },
		{ 2816, 2787 },
		{ 1431, 1428 },
		{ 2820, 2791 },
		{ 2725, 2695 },
		{ 1901, 1897 },
		{ 2603, 2571 },
		{ 2692, 2657 },
		{ 2787, 2757 },
		{ 1810, 1809 },
		{ 1768, 1767 },
		{ 3069, 3068 },
		{ 2791, 2761 },
		{ 1662, 1646 },
		{ 655, 591 },
		{ 2110, 2087 },
		{ 2215, 2201 },
		{ 2913, 2894 },
		{ 1879, 1878 },
		{ 2915, 2896 },
		{ 2046, 2043 },
		{ 1791, 1790 },
		{ 2799, 2769 },
		{ 2857, 2833 },
		{ 2655, 2620 },
		{ 3018, 3017 },
		{ 1421, 1420 },
		{ 3167, 3166 },
		{ 1411, 1410 },
		{ 2804, 2774 },
		{ 1794, 1793 },
		{ 2806, 2776 },
		{ 1330, 1329 },
		{ 1716, 1715 },
		{ 3027, 3026 },
		{ 3176, 3175 },
		{ 2301, 2300 },
		{ 2756, 2725 },
		{ 1760, 1759 },
		{ 1706, 1705 },
		{ 3294, 3291 },
		{ 2010, 1997 },
		{ 2583, 2549 },
		{ 2254, 2247 },
		{ 1394, 1393 },
		{ 3301, 3300 },
		{ 2946, 2934 },
		{ 2821, 2793 },
		{ 3304, 3303 },
		{ 2879, 2855 },
		{ 2018, 2008 },
		{ 1869, 1868 },
		{ 2716, 2685 },
		{ 2311, 2310 },
		{ 2550, 2518 },
		{ 3199, 3198 },
		{ 3314, 3311 },
		{ 2675, 2639 },
		{ 2260, 2256 },
		{ 2313, 2312 },
		{ 3204, 3203 },
		{ 1734, 1733 },
		{ 3126, 3125 },
		{ 3321, 3318 },
		{ 3207, 3206 },
		{ 1784, 1783 },
		{ 1328, 1327 },
		{ 2779, 2749 },
		{ 1441, 1440 },
		{ 2187, 2172 },
		{ 2971, 2969 },
		{ 3330, 3328 },
		{ 2188, 2175 },
		{ 3135, 3134 },
		{ 1923, 1899 },
		{ 2107, 2083 },
		{ 718, 652 },
		{ 788, 727 },
		{ 908, 852 },
		{ 448, 403 },
		{ 1318, 11 },
		{ 1352, 15 },
		{ 1775, 31 },
		{ 69, 11 },
		{ 69, 15 },
		{ 69, 31 },
		{ 3103, 57 },
		{ 1339, 13 },
		{ 1402, 19 },
		{ 69, 57 },
		{ 69, 13 },
		{ 69, 19 },
		{ 1026, 979 },
		{ 1801, 33 },
		{ 787, 727 },
		{ 449, 403 },
		{ 69, 33 },
		{ 1827, 35 },
		{ 719, 652 },
		{ 2996, 47 },
		{ 69, 35 },
		{ 1028, 981 },
		{ 69, 47 },
		{ 1430, 21 },
		{ 909, 852 },
		{ 1725, 27 },
		{ 69, 21 },
		{ 1751, 29 },
		{ 69, 27 },
		{ 3193, 61 },
		{ 69, 29 },
		{ 1029, 982 },
		{ 69, 61 },
		{ 1045, 1002 },
		{ 1073, 1034 },
		{ 1076, 1037 },
		{ 1084, 1046 },
		{ 1094, 1056 },
		{ 1115, 1079 },
		{ 2166, 2148 },
		{ 1134, 1097 },
		{ 1144, 1109 },
		{ 1146, 1111 },
		{ 1159, 1128 },
		{ 1953, 1934 },
		{ 1954, 1935 },
		{ 1174, 1145 },
		{ 1180, 1152 },
		{ 1187, 1160 },
		{ 2182, 2165 },
		{ 1188, 1161 },
		{ 1202, 1183 },
		{ 1203, 1184 },
		{ 1221, 1204 },
		{ 1226, 1210 },
		{ 1234, 1222 },
		{ 1246, 1235 },
		{ 1252, 1241 },
		{ 1974, 1956 },
		{ 1287, 1286 },
		{ 272, 229 },
		{ 282, 238 },
		{ 290, 246 },
		{ 305, 258 },
		{ 314, 267 },
		{ 321, 273 },
		{ 331, 283 },
		{ 1989, 1973 },
		{ 349, 300 },
		{ 352, 303 },
		{ 361, 311 },
		{ 362, 312 },
		{ 367, 317 },
		{ 382, 335 },
		{ 408, 363 },
		{ 411, 366 },
		{ 419, 374 },
		{ 430, 386 },
		{ 437, 393 },
		{ 446, 401 },
		{ 236, 198 },
		{ 450, 404 },
		{ 1316, 11 },
		{ 1350, 15 },
		{ 1773, 31 },
		{ 463, 415 },
		{ 240, 202 },
		{ 485, 431 },
		{ 3102, 57 },
		{ 1337, 13 },
		{ 1400, 19 },
		{ 488, 435 },
		{ 498, 443 },
		{ 499, 444 },
		{ 502, 447 },
		{ 1799, 33 },
		{ 514, 459 },
		{ 523, 470 },
		{ 556, 494 },
		{ 1825, 35 },
		{ 567, 503 },
		{ 2994, 47 },
		{ 570, 506 },
		{ 571, 507 },
		{ 575, 511 },
		{ 1428, 21 },
		{ 588, 526 },
		{ 1723, 27 },
		{ 592, 530 },
		{ 1749, 29 },
		{ 596, 534 },
		{ 3191, 61 },
		{ 606, 544 },
		{ 622, 557 },
		{ 623, 559 },
		{ 628, 564 },
		{ 645, 581 },
		{ 241, 203 },
		{ 662, 595 },
		{ 675, 608 },
		{ 678, 611 },
		{ 689, 621 },
		{ 706, 637 },
		{ 707, 638 },
		{ 717, 651 },
		{ 248, 210 },
		{ 737, 670 },
		{ 249, 211 },
		{ 789, 728 },
		{ 801, 739 },
		{ 803, 741 },
		{ 805, 743 },
		{ 810, 748 },
		{ 813, 751 },
		{ 2121, 2100 },
		{ 832, 771 },
		{ 843, 781 },
		{ 847, 785 },
		{ 848, 786 },
		{ 881, 822 },
		{ 900, 844 },
		{ 271, 228 },
		{ 941, 887 },
		{ 951, 897 },
		{ 966, 912 },
		{ 980, 927 },
		{ 1006, 956 },
		{ 2143, 2122 },
		{ 2144, 2123 },
		{ 1010, 960 },
		{ 1933, 1914 },
		{ 2472, 2472 },
		{ 2472, 2472 },
		{ 69, 37 },
		{ 69, 43 },
		{ 69, 59 },
		{ 69, 17 },
		{ 69, 49 },
		{ 69, 25 },
		{ 455, 408 },
		{ 69, 53 },
		{ 456, 408 },
		{ 454, 408 },
		{ 69, 55 },
		{ 223, 189 },
		{ 2194, 2180 },
		{ 2196, 2181 },
		{ 3278, 3274 },
		{ 221, 189 },
		{ 3284, 3279 },
		{ 3250, 3249 },
		{ 618, 556 },
		{ 2195, 2180 },
		{ 2197, 2181 },
		{ 2472, 2472 },
		{ 619, 556 },
		{ 2255, 2248 },
		{ 858, 796 },
		{ 457, 408 },
		{ 490, 437 },
		{ 492, 437 },
		{ 533, 479 },
		{ 534, 479 },
		{ 224, 189 },
		{ 222, 189 },
		{ 429, 385 },
		{ 621, 556 },
		{ 458, 409 },
		{ 620, 556 },
		{ 535, 479 },
		{ 493, 437 },
		{ 763, 703 },
		{ 764, 704 },
		{ 413, 368 },
		{ 2192, 2178 },
		{ 491, 437 },
		{ 2001, 1987 },
		{ 525, 472 },
		{ 613, 551 },
		{ 345, 296 },
		{ 855, 793 },
		{ 2104, 2080 },
		{ 1177, 1148 },
		{ 546, 488 },
		{ 1183, 1155 },
		{ 216, 186 },
		{ 300, 253 },
		{ 716, 650 },
		{ 215, 186 },
		{ 2103, 2080 },
		{ 2125, 2104 },
		{ 688, 620 },
		{ 548, 488 },
		{ 465, 417 },
		{ 217, 186 },
		{ 2738, 2738 },
		{ 2738, 2738 },
		{ 547, 488 },
		{ 659, 592 },
		{ 958, 904 },
		{ 375, 325 },
		{ 2102, 2080 },
		{ 658, 592 },
		{ 209, 183 },
		{ 957, 903 },
		{ 211, 183 },
		{ 440, 395 },
		{ 439, 395 },
		{ 210, 183 },
		{ 2504, 2472 },
		{ 657, 592 },
		{ 656, 592 },
		{ 1859, 37 },
		{ 2296, 43 },
		{ 3150, 59 },
		{ 1385, 17 },
		{ 3012, 49 },
		{ 1696, 25 },
		{ 2738, 2738 },
		{ 3058, 53 },
		{ 540, 483 },
		{ 231, 193 },
		{ 3085, 55 },
		{ 541, 483 },
		{ 537, 481 },
		{ 838, 776 },
		{ 860, 798 },
		{ 1192, 1165 },
		{ 605, 543 },
		{ 227, 190 },
		{ 310, 263 },
		{ 311, 264 },
		{ 225, 190 },
		{ 230, 193 },
		{ 837, 776 },
		{ 226, 190 },
		{ 2126, 2105 },
		{ 748, 685 },
		{ 953, 899 },
		{ 265, 223 },
		{ 276, 232 },
		{ 859, 797 },
		{ 1936, 1917 },
		{ 291, 247 },
		{ 538, 481 },
		{ 1191, 1165 },
		{ 553, 492 },
		{ 1119, 1083 },
		{ 641, 576 },
		{ 510, 455 },
		{ 3248, 3246 },
		{ 1145, 1110 },
		{ 731, 664 },
		{ 594, 532 },
		{ 572, 508 },
		{ 3261, 3255 },
		{ 1916, 1892 },
		{ 1017, 969 },
		{ 292, 247 },
		{ 1021, 974 },
		{ 864, 802 },
		{ 554, 492 },
		{ 750, 687 },
		{ 753, 690 },
		{ 3270, 3265 },
		{ 242, 204 },
		{ 2769, 2738 },
		{ 2579, 2545 },
		{ 3275, 3271 },
		{ 1340, 1337 },
		{ 918, 863 },
		{ 442, 397 },
		{ 1217, 1200 },
		{ 1346, 1345 },
		{ 839, 777 },
		{ 1224, 1207 },
		{ 2997, 2994 },
		{ 785, 725 },
		{ 1387, 1385 },
		{ 786, 726 },
		{ 1319, 1316 },
		{ 1242, 1231 },
		{ 1243, 1232 },
		{ 1244, 1233 },
		{ 585, 523 },
		{ 212, 184 },
		{ 366, 316 },
		{ 287, 243 },
		{ 734, 667 },
		{ 438, 394 },
		{ 213, 184 },
		{ 199, 180 },
		{ 600, 538 },
		{ 967, 913 },
		{ 973, 919 },
		{ 201, 180 },
		{ 751, 688 },
		{ 981, 928 },
		{ 584, 523 },
		{ 987, 934 },
		{ 992, 941 },
		{ 998, 949 },
		{ 200, 180 },
		{ 1005, 955 },
		{ 604, 542 },
		{ 754, 691 },
		{ 515, 460 },
		{ 380, 332 },
		{ 765, 705 },
		{ 611, 549 },
		{ 317, 270 },
		{ 1030, 983 },
		{ 1042, 999 },
		{ 3249, 3248 },
		{ 384, 338 },
		{ 1050, 1007 },
		{ 1051, 1008 },
		{ 1055, 1012 },
		{ 1056, 1013 },
		{ 1062, 1019 },
		{ 386, 340 },
		{ 403, 359 },
		{ 3266, 3261 },
		{ 544, 486 },
		{ 355, 305 },
		{ 1096, 1058 },
		{ 1098, 1060 },
		{ 1102, 1064 },
		{ 1112, 1076 },
		{ 3274, 3270 },
		{ 1114, 1078 },
		{ 643, 578 },
		{ 550, 490 },
		{ 354, 305 },
		{ 353, 305 },
		{ 1127, 1090 },
		{ 3279, 3275 },
		{ 648, 584 },
		{ 834, 773 },
		{ 835, 774 },
		{ 652, 590 },
		{ 1148, 1113 },
		{ 1935, 1916 },
		{ 1156, 1122 },
		{ 461, 412 },
		{ 1173, 1144 },
		{ 842, 780 },
		{ 1175, 1146 },
		{ 356, 306 },
		{ 844, 782 },
		{ 2123, 2102 },
		{ 307, 260 },
		{ 415, 370 },
		{ 853, 791 },
		{ 854, 792 },
		{ 1196, 1174 },
		{ 1201, 1181 },
		{ 418, 373 },
		{ 273, 230 },
		{ 1210, 1192 },
		{ 1211, 1194 },
		{ 422, 377 },
		{ 700, 630 },
		{ 1222, 1205 },
		{ 702, 633 },
		{ 878, 818 },
		{ 879, 819 },
		{ 1235, 1223 },
		{ 1237, 1225 },
		{ 579, 516 },
		{ 889, 832 },
		{ 897, 841 },
		{ 580, 517 },
		{ 1250, 1239 },
		{ 1251, 1240 },
		{ 363, 313 },
		{ 1256, 1247 },
		{ 1271, 1262 },
		{ 916, 861 },
		{ 1298, 1297 },
		{ 586, 524 },
		{ 923, 868 },
		{ 936, 882 },
		{ 220, 188 },
		{ 262, 221 },
		{ 1971, 1953 },
		{ 219, 188 },
		{ 318, 271 },
		{ 903, 847 },
		{ 904, 848 },
		{ 263, 221 },
		{ 1065, 1023 },
		{ 1066, 1024 },
		{ 1069, 1028 },
		{ 234, 196 },
		{ 1625, 1625 },
		{ 1628, 1628 },
		{ 1631, 1631 },
		{ 1634, 1634 },
		{ 1637, 1637 },
		{ 1640, 1640 },
		{ 235, 197 },
		{ 1079, 1040 },
		{ 669, 602 },
		{ 598, 536 },
		{ 1095, 1057 },
		{ 412, 367 },
		{ 685, 618 },
		{ 1099, 1061 },
		{ 948, 894 },
		{ 1658, 1658 },
		{ 1281, 1275 },
		{ 950, 896 },
		{ 376, 328 },
		{ 565, 501 },
		{ 339, 290 },
		{ 1123, 1086 },
		{ 1667, 1667 },
		{ 1625, 1625 },
		{ 1628, 1628 },
		{ 1631, 1631 },
		{ 1634, 1634 },
		{ 1637, 1637 },
		{ 1640, 1640 },
		{ 609, 547 },
		{ 251, 213 },
		{ 1139, 1103 },
		{ 1588, 1588 },
		{ 2161, 2143 },
		{ 383, 336 },
		{ 712, 643 },
		{ 615, 553 },
		{ 247, 209 },
		{ 1658, 1658 },
		{ 1151, 1116 },
		{ 460, 411 },
		{ 576, 512 },
		{ 997, 948 },
		{ 426, 382 },
		{ 736, 669 },
		{ 1667, 1667 },
		{ 1692, 1692 },
		{ 634, 570 },
		{ 635, 571 },
		{ 527, 475 },
		{ 583, 522 },
		{ 1023, 976 },
		{ 752, 689 },
		{ 862, 800 },
		{ 644, 579 },
		{ 1588, 1588 },
		{ 364, 314 },
		{ 391, 345 },
		{ 650, 588 },
		{ 433, 389 },
		{ 1215, 1198 },
		{ 774, 716 },
		{ 1498, 1625 },
		{ 1498, 1628 },
		{ 1498, 1631 },
		{ 1498, 1634 },
		{ 1498, 1637 },
		{ 1498, 1640 },
		{ 627, 563 },
		{ 1692, 1692 },
		{ 348, 299 },
		{ 1024, 977 },
		{ 720, 653 },
		{ 633, 569 },
		{ 313, 266 },
		{ 416, 371 },
		{ 1033, 989 },
		{ 1498, 1658 },
		{ 1039, 996 },
		{ 857, 795 },
		{ 3245, 3241 },
		{ 1043, 1000 },
		{ 521, 467 },
		{ 2212, 2198 },
		{ 1498, 1667 },
		{ 2213, 2199 },
		{ 743, 676 },
		{ 747, 682 },
		{ 1052, 1009 },
		{ 323, 275 },
		{ 261, 220 },
		{ 865, 803 },
		{ 875, 814 },
		{ 1284, 1282 },
		{ 1498, 1588 },
		{ 877, 816 },
		{ 1288, 1287 },
		{ 388, 342 },
		{ 1071, 1031 },
		{ 423, 378 },
		{ 880, 820 },
		{ 374, 324 },
		{ 651, 589 },
		{ 1086, 1048 },
		{ 756, 693 },
		{ 757, 695 },
		{ 759, 698 },
		{ 396, 350 },
		{ 1498, 1692 },
		{ 397, 352 },
		{ 545, 487 },
		{ 769, 708 },
		{ 772, 713 },
		{ 773, 714 },
		{ 289, 245 },
		{ 1973, 1955 },
		{ 783, 723 },
		{ 404, 360 },
		{ 673, 606 },
		{ 340, 291 },
		{ 1142, 1106 },
		{ 677, 610 },
		{ 264, 222 },
		{ 962, 908 },
		{ 965, 911 },
		{ 1988, 1972 },
		{ 679, 612 },
		{ 683, 616 },
		{ 968, 914 },
		{ 1163, 1132 },
		{ 1166, 1135 },
		{ 1167, 1136 },
		{ 971, 917 },
		{ 558, 496 },
		{ 381, 333 },
		{ 817, 755 },
		{ 984, 931 },
		{ 1182, 1154 },
		{ 985, 932 },
		{ 2165, 2147 },
		{ 2015, 2003 },
		{ 2016, 2004 },
		{ 1184, 1156 },
		{ 986, 933 },
		{ 823, 762 },
		{ 443, 398 },
		{ 614, 552 },
		{ 568, 504 },
		{ 503, 448 },
		{ 506, 451 },
		{ 2179, 2162 },
		{ 509, 454 },
		{ 625, 561 },
		{ 640, 575 },
		{ 208, 182 },
		{ 529, 477 },
		{ 639, 575 },
		{ 406, 361 },
		{ 405, 361 },
		{ 294, 248 },
		{ 293, 248 },
		{ 560, 498 },
		{ 561, 498 },
		{ 531, 478 },
		{ 207, 182 },
		{ 1231, 1219 },
		{ 530, 477 },
		{ 697, 628 },
		{ 698, 628 },
		{ 562, 499 },
		{ 563, 499 },
		{ 254, 215 },
		{ 257, 218 },
		{ 532, 478 },
		{ 253, 215 },
		{ 258, 218 },
		{ 269, 226 },
		{ 319, 272 },
		{ 299, 252 },
		{ 798, 737 },
		{ 2651, 2651 },
		{ 2651, 2651 },
		{ 268, 226 },
		{ 3271, 3266 },
		{ 259, 218 },
		{ 320, 272 },
		{ 1232, 1220 },
		{ 467, 419 },
		{ 298, 252 },
		{ 799, 737 },
		{ 508, 453 },
		{ 861, 799 },
		{ 1147, 1112 },
		{ 542, 484 },
		{ 371, 321 },
		{ 244, 206 },
		{ 2162, 2144 },
		{ 804, 742 },
		{ 2098, 2075 },
		{ 1058, 1015 },
		{ 608, 546 },
		{ 1990, 1974 },
		{ 1934, 1915 },
		{ 2651, 2651 },
		{ 807, 745 },
		{ 983, 930 },
		{ 1937, 1918 },
		{ 2545, 2513 },
		{ 487, 433 },
		{ 812, 750 },
		{ 281, 237 },
		{ 407, 362 },
		{ 2183, 2166 },
		{ 2003, 1989 },
		{ 1290, 1289 },
		{ 267, 225 },
		{ 701, 631 },
		{ 301, 254 },
		{ 2122, 2101 },
		{ 284, 240 },
		{ 758, 697 },
		{ 370, 320 },
		{ 2198, 2182 },
		{ 2127, 2106 },
		{ 1194, 1171 },
		{ 1195, 1173 },
		{ 504, 449 },
		{ 3246, 3244 },
		{ 1197, 1175 },
		{ 713, 645 },
		{ 624, 560 },
		{ 566, 502 },
		{ 943, 889 },
		{ 665, 598 },
		{ 1213, 1196 },
		{ 626, 562 },
		{ 724, 657 },
		{ 725, 658 },
		{ 3265, 3260 },
		{ 1036, 992 },
		{ 726, 659 },
		{ 505, 450 },
		{ 1972, 1954 },
		{ 275, 231 },
		{ 2686, 2651 },
		{ 686, 619 },
		{ 797, 736 },
		{ 911, 854 },
		{ 1185, 1158 },
		{ 988, 935 },
		{ 274, 231 },
		{ 989, 936 },
		{ 1080, 1041 },
		{ 453, 407 },
		{ 687, 619 },
		{ 464, 416 },
		{ 921, 866 },
		{ 616, 554 },
		{ 1966, 1947 },
		{ 928, 873 },
		{ 932, 878 },
		{ 1015, 966 },
		{ 933, 879 },
		{ 573, 509 },
		{ 939, 885 },
		{ 2101, 2079 },
		{ 520, 466 },
		{ 2105, 2081 },
		{ 1117, 1081 },
		{ 232, 194 },
		{ 602, 540 },
		{ 811, 749 },
		{ 2193, 2179 },
		{ 1130, 1093 },
		{ 1230, 1218 },
		{ 1915, 1891 },
		{ 496, 440 },
		{ 1917, 1893 },
		{ 1135, 1098 },
		{ 1136, 1099 },
		{ 1137, 1100 },
		{ 1031, 987 },
		{ 952, 898 },
		{ 306, 259 },
		{ 816, 754 },
		{ 582, 521 },
		{ 742, 675 },
		{ 2231, 2212 },
		{ 270, 227 },
		{ 2002, 1988 },
		{ 833, 772 },
		{ 745, 678 },
		{ 1257, 1248 },
		{ 1268, 1259 },
		{ 1269, 1260 },
		{ 631, 567 },
		{ 1274, 1265 },
		{ 255, 216 },
		{ 1282, 1277 },
		{ 1283, 1278 },
		{ 1164, 1133 },
		{ 2026, 2015 },
		{ 883, 826 },
		{ 569, 505 },
		{ 1169, 1138 },
		{ 1059, 1016 },
		{ 612, 550 },
		{ 1063, 1020 },
		{ 511, 456 },
		{ 1178, 1149 },
		{ 795, 734 },
		{ 846, 784 },
		{ 767, 707 },
		{ 852, 790 },
		{ 1116, 1080 },
		{ 1285, 1283 },
		{ 906, 850 },
		{ 1190, 1164 },
		{ 768, 707 },
		{ 972, 918 },
		{ 1297, 1296 },
		{ 1193, 1169 },
		{ 252, 214 },
		{ 402, 358 },
		{ 1128, 1091 },
		{ 1129, 1092 },
		{ 1198, 1178 },
		{ 601, 539 },
		{ 982, 929 },
		{ 760, 699 },
		{ 1204, 1185 },
		{ 1209, 1191 },
		{ 727, 660 },
		{ 2177, 2160 },
		{ 730, 663 },
		{ 926, 871 },
		{ 649, 585 },
		{ 468, 420 },
		{ 473, 428 },
		{ 1986, 1970 },
		{ 581, 520 },
		{ 937, 883 },
		{ 316, 269 },
		{ 1228, 1216 },
		{ 1229, 1217 },
		{ 1002, 952 },
		{ 1152, 1117 },
		{ 1153, 1119 },
		{ 445, 400 },
		{ 661, 594 },
		{ 1236, 1224 },
		{ 2124, 2103 },
		{ 1161, 1130 },
		{ 1241, 1230 },
		{ 486, 432 },
		{ 392, 346 },
		{ 749, 686 },
		{ 549, 489 },
		{ 1168, 1137 },
		{ 591, 529 },
		{ 885, 828 },
		{ 1253, 1244 },
		{ 674, 607 },
		{ 3244, 3240 },
		{ 1097, 1059 },
		{ 1266, 1257 },
		{ 959, 905 },
		{ 895, 839 },
		{ 1101, 1063 },
		{ 302, 255 },
		{ 1275, 1268 },
		{ 1276, 1269 },
		{ 3260, 3254 },
		{ 1280, 1274 },
		{ 528, 476 },
		{ 414, 369 },
		{ 1018, 970 },
		{ 2163, 2145 },
		{ 2164, 2146 },
		{ 872, 811 },
		{ 711, 642 },
		{ 738, 671 },
		{ 237, 199 },
		{ 637, 573 },
		{ 806, 744 },
		{ 935, 881 },
		{ 347, 298 },
		{ 436, 392 },
		{ 459, 410 },
		{ 887, 830 },
		{ 471, 424 },
		{ 945, 891 },
		{ 721, 654 },
		{ 694, 625 },
		{ 1295, 1294 },
		{ 990, 939 },
		{ 696, 627 },
		{ 901, 845 },
		{ 821, 759 },
		{ 239, 201 },
		{ 647, 583 },
		{ 617, 555 },
		{ 632, 568 },
		{ 427, 383 },
		{ 372, 322 },
		{ 1085, 1047 },
		{ 771, 710 },
		{ 808, 746 },
		{ 497, 442 },
		{ 978, 925 },
		{ 390, 344 },
		{ 636, 572 },
		{ 884, 827 },
		{ 238, 200 },
		{ 732, 665 },
		{ 1176, 1147 },
		{ 1105, 1068 },
		{ 1108, 1072 },
		{ 277, 233 },
		{ 856, 794 },
		{ 735, 668 },
		{ 1046, 1003 },
		{ 1258, 1249 },
		{ 1263, 1254 },
		{ 1047, 1004 },
		{ 1267, 1258 },
		{ 1118, 1082 },
		{ 946, 892 },
		{ 1189, 1162 },
		{ 1272, 1263 },
		{ 1273, 1264 },
		{ 1121, 1084 },
		{ 329, 281 },
		{ 365, 315 },
		{ 1279, 1273 },
		{ 1122, 1084 },
		{ 664, 597 },
		{ 1120, 1084 },
		{ 993, 942 },
		{ 2148, 2127 },
		{ 994, 943 },
		{ 739, 672 },
		{ 296, 250 },
		{ 1956, 1937 },
		{ 629, 565 },
		{ 334, 286 },
		{ 840, 778 },
		{ 1140, 1104 },
		{ 1291, 1290 },
		{ 1208, 1189 },
		{ 873, 812 },
		{ 1143, 1107 },
		{ 1014, 964 },
		{ 1212, 1195 },
		{ 379, 331 },
		{ 1214, 1197 },
		{ 1074, 1035 },
		{ 1075, 1036 },
		{ 922, 867 },
		{ 1077, 1038 },
		{ 564, 500 },
		{ 924, 869 },
		{ 1227, 1213 },
		{ 1082, 1044 },
		{ 1157, 1124 },
		{ 969, 915 },
		{ 599, 537 },
		{ 1486, 1486 },
		{ 770, 709 },
		{ 1054, 1011 },
		{ 308, 261 },
		{ 886, 829 },
		{ 335, 287 },
		{ 1107, 1070 },
		{ 870, 810 },
		{ 385, 339 },
		{ 336, 287 },
		{ 871, 810 },
		{ 1344, 1344 },
		{ 1344, 1344 },
		{ 552, 491 },
		{ 1110, 1073 },
		{ 776, 717 },
		{ 775, 717 },
		{ 551, 491 },
		{ 777, 717 },
		{ 1109, 1073 },
		{ 1016, 967 },
		{ 424, 379 },
		{ 1113, 1077 },
		{ 1486, 1486 },
		{ 517, 463 },
		{ 1020, 973 },
		{ 519, 465 },
		{ 791, 730 },
		{ 1216, 1199 },
		{ 794, 733 },
		{ 1218, 1201 },
		{ 288, 244 },
		{ 942, 888 },
		{ 796, 735 },
		{ 1344, 1344 },
		{ 681, 614 },
		{ 322, 274 },
		{ 800, 738 },
		{ 1035, 991 },
		{ 1133, 1096 },
		{ 522, 468 },
		{ 1038, 995 },
		{ 428, 384 },
		{ 1041, 998 },
		{ 1138, 1102 },
		{ 524, 471 },
		{ 1239, 1228 },
		{ 409, 364 },
		{ 260, 219 },
		{ 642, 577 },
		{ 607, 545 },
		{ 809, 747 },
		{ 1248, 1237 },
		{ 378, 330 },
		{ 243, 205 },
		{ 1987, 1971 },
		{ 501, 446 },
		{ 1149, 1114 },
		{ 703, 634 },
		{ 755, 692 },
		{ 1057, 1014 },
		{ 2145, 2124 },
		{ 1498, 1486 },
		{ 2146, 2125 },
		{ 1259, 1250 },
		{ 1260, 1251 },
		{ 970, 916 },
		{ 1265, 1256 },
		{ 704, 635 },
		{ 1345, 1344 },
		{ 1158, 1127 },
		{ 1061, 1018 },
		{ 344, 295 },
		{ 536, 480 },
		{ 826, 765 },
		{ 896, 840 },
		{ 1068, 1027 },
		{ 827, 766 },
		{ 898, 842 },
		{ 1278, 1271 },
		{ 1171, 1142 },
		{ 368, 318 },
		{ 539, 482 },
		{ 466, 418 },
		{ 400, 355 },
		{ 358, 308 },
		{ 660, 593 },
		{ 910, 853 },
		{ 1081, 1042 },
		{ 3194, 3191 },
		{ 297, 251 },
		{ 472, 426 },
		{ 663, 596 },
		{ 420, 375 },
		{ 2178, 2161 },
		{ 1092, 1054 },
		{ 2180, 2163 },
		{ 2181, 2164 },
		{ 1093, 1055 },
		{ 333, 285 },
		{ 590, 528 },
		{ 671, 604 },
		{ 925, 870 },
		{ 850, 788 },
		{ 1007, 957 },
		{ 1100, 1062 },
		{ 1008, 958 },
		{ 512, 457 },
		{ 1104, 1067 },
		{ 930, 875 },
		{ 1125, 1088 },
		{ 1126, 1089 },
		{ 761, 700 },
		{ 907, 851 },
		{ 425, 380 },
		{ 874, 813 },
		{ 447, 402 },
		{ 2246, 2233 },
		{ 1952, 1933 },
		{ 1861, 1859 },
		{ 593, 531 },
		{ 741, 674 },
		{ 849, 787 },
		{ 2011, 1998 },
		{ 2142, 2121 },
		{ 1726, 1723 },
		{ 729, 662 },
		{ 1053, 1010 },
		{ 1403, 1400 },
		{ 1752, 1749 },
		{ 2204, 2189 },
		{ 1776, 1773 },
		{ 526, 473 },
		{ 1802, 1799 },
		{ 2029, 2020 },
		{ 1160, 1129 },
		{ 960, 906 },
		{ 1219, 1202 },
		{ 1220, 1203 },
		{ 1698, 1696 },
		{ 346, 297 },
		{ 999, 950 },
		{ 343, 294 },
		{ 1070, 1029 },
		{ 991, 940 },
		{ 1000, 950 },
		{ 1186, 1159 },
		{ 246, 208 },
		{ 779, 719 },
		{ 815, 753 },
		{ 876, 815 },
		{ 1027, 980 },
		{ 389, 343 },
		{ 1165, 1134 },
		{ 452, 406 },
		{ 766, 706 },
		{ 444, 399 },
		{ 1083, 1045 },
		{ 1199, 1179 },
		{ 1200, 1180 },
		{ 733, 666 },
		{ 1233, 1221 },
		{ 722, 655 },
		{ 863, 801 },
		{ 295, 249 },
		{ 1277, 1270 },
		{ 1206, 1187 },
		{ 1207, 1188 },
		{ 963, 909 },
		{ 399, 354 },
		{ 507, 452 },
		{ 1067, 1026 },
		{ 888, 831 },
		{ 1091, 1053 },
		{ 1289, 1288 },
		{ 324, 276 },
		{ 917, 862 },
		{ 1294, 1293 },
		{ 326, 278 },
		{ 1296, 1295 },
		{ 920, 865 },
		{ 283, 239 },
		{ 781, 721 },
		{ 2147, 2126 },
		{ 218, 187 },
		{ 676, 609 },
		{ 1009, 959 },
		{ 587, 525 },
		{ 543, 485 },
		{ 1103, 1065 },
		{ 927, 872 },
		{ 589, 527 },
		{ 929, 874 },
		{ 790, 729 },
		{ 931, 876 },
		{ 1111, 1074 },
		{ 2160, 2142 },
		{ 680, 613 },
		{ 1022, 975 },
		{ 2004, 1990 },
		{ 214, 185 },
		{ 682, 615 },
		{ 1025, 978 },
		{ 462, 413 },
		{ 740, 673 },
		{ 2013, 2001 },
		{ 938, 884 },
		{ 684, 617 },
		{ 940, 886 },
		{ 432, 388 },
		{ 410, 365 },
		{ 1034, 990 },
		{ 434, 390 },
		{ 435, 391 },
		{ 1037, 993 },
		{ 1132, 1095 },
		{ 868, 806 },
		{ 1918, 1894 },
		{ 693, 624 },
		{ 1040, 997 },
		{ 369, 319 },
		{ 695, 626 },
		{ 351, 302 },
		{ 513, 458 },
		{ 955, 901 },
		{ 1141, 1105 },
		{ 956, 902 },
		{ 1048, 1005 },
		{ 559, 497 },
		{ 470, 422 },
		{ 2199, 2183 },
		{ 279, 235 },
		{ 280, 236 },
		{ 961, 907 },
		{ 441, 396 },
		{ 474, 429 },
		{ 2206, 2192 },
		{ 2208, 2194 },
		{ 2209, 2195 },
		{ 2210, 2196 },
		{ 2211, 2197 },
		{ 309, 262 },
		{ 820, 758 },
		{ 393, 347 },
		{ 357, 307 },
		{ 824, 763 },
		{ 825, 764 },
		{ 762, 701 },
		{ 891, 835 },
		{ 245, 207 },
		{ 2106, 2082 },
		{ 977, 923 },
		{ 828, 767 },
		{ 831, 770 },
		{ 360, 310 },
		{ 1072, 1032 },
		{ 899, 843 },
		{ 1270, 1261 },
		{ 1170, 1140 },
		{ 1955, 1936 },
		{ 489, 436 },
		{ 1172, 1143 },
		{ 421, 376 },
		{ 494, 438 },
		{ 836, 775 },
		{ 905, 849 },
		{ 341, 292 },
		{ 401, 356 },
		{ 1179, 1150 },
		{ 196, 176 },
		{ 666, 599 },
		{ 312, 265 },
		{ 912, 856 },
		{ 1970, 1952 },
		{ 914, 859 },
		{ 1286, 1284 },
		{ 1087, 1049 },
		{ 690, 622 },
		{ 691, 622 },
		{ 2034, 2034 },
		{ 2034, 2034 },
		{ 2244, 2244 },
		{ 2244, 2244 },
		{ 1693, 1693 },
		{ 1693, 1693 },
		{ 2039, 2039 },
		{ 2039, 2039 },
		{ 2190, 2190 },
		{ 2190, 2190 },
		{ 2250, 2250 },
		{ 2250, 2250 },
		{ 1638, 1638 },
		{ 1638, 1638 },
		{ 1629, 1629 },
		{ 1629, 1629 },
		{ 1641, 1641 },
		{ 1641, 1641 },
		{ 1589, 1589 },
		{ 1589, 1589 },
		{ 1668, 1668 },
		{ 1668, 1668 },
		{ 603, 541 },
		{ 2034, 2034 },
		{ 2298, 2296 },
		{ 2244, 2244 },
		{ 285, 241 },
		{ 1693, 1693 },
		{ 893, 837 },
		{ 2039, 2039 },
		{ 2717, 2686 },
		{ 2190, 2190 },
		{ 256, 217 },
		{ 2250, 2250 },
		{ 1150, 1115 },
		{ 1638, 1638 },
		{ 337, 288 },
		{ 1629, 1629 },
		{ 330, 282 },
		{ 1641, 1641 },
		{ 595, 533 },
		{ 1589, 1589 },
		{ 1154, 1120 },
		{ 1668, 1668 },
		{ 1632, 1632 },
		{ 1632, 1632 },
		{ 2053, 2053 },
		{ 2053, 2053 },
		{ 1659, 1659 },
		{ 1659, 1659 },
		{ 1626, 1626 },
		{ 1626, 1626 },
		{ 2035, 2034 },
		{ 1155, 1121 },
		{ 2245, 2244 },
		{ 431, 387 },
		{ 1694, 1693 },
		{ 574, 510 },
		{ 2040, 2039 },
		{ 304, 257 },
		{ 2191, 2190 },
		{ 1347, 1346 },
		{ 2251, 2250 },
		{ 902, 846 },
		{ 1639, 1638 },
		{ 1049, 1006 },
		{ 1630, 1629 },
		{ 1632, 1632 },
		{ 1642, 1641 },
		{ 2053, 2053 },
		{ 1590, 1589 },
		{ 1659, 1659 },
		{ 1669, 1668 },
		{ 1626, 1626 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2022, 2022 },
		{ 2022, 2022 },
		{ 2024, 2024 },
		{ 2024, 2024 },
		{ 1635, 1635 },
		{ 1635, 1635 },
		{ 2219, 2219 },
		{ 2219, 2219 },
		{ 2221, 2221 },
		{ 2221, 2221 },
		{ 2223, 2223 },
		{ 2223, 2223 },
		{ 2225, 2225 },
		{ 2225, 2225 },
		{ 2227, 2227 },
		{ 2227, 2227 },
		{ 2229, 2229 },
		{ 2229, 2229 },
		{ 1999, 1999 },
		{ 1999, 1999 },
		{ 1633, 1632 },
		{ 2273, 2273 },
		{ 2054, 2053 },
		{ 2022, 2022 },
		{ 1660, 1659 },
		{ 2024, 2024 },
		{ 1627, 1626 },
		{ 1635, 1635 },
		{ 325, 277 },
		{ 2219, 2219 },
		{ 728, 661 },
		{ 2221, 2221 },
		{ 802, 740 },
		{ 2223, 2223 },
		{ 822, 760 },
		{ 2225, 2225 },
		{ 974, 920 },
		{ 2227, 2227 },
		{ 975, 921 },
		{ 2229, 2229 },
		{ 692, 623 },
		{ 1999, 1999 },
		{ 1032, 988 },
		{ 577, 514 },
		{ 979, 926 },
		{ 495, 439 },
		{ 715, 649 },
		{ 1011, 961 },
		{ 1013, 963 },
		{ 2168, 2151 },
		{ 2274, 2273 },
		{ 1975, 1958 },
		{ 2023, 2022 },
		{ 1124, 1087 },
		{ 2025, 2024 },
		{ 851, 789 },
		{ 1636, 1635 },
		{ 395, 349 },
		{ 2220, 2219 },
		{ 1704, 1703 },
		{ 2222, 2221 },
		{ 3286, 3281 },
		{ 2224, 2223 },
		{ 2232, 2213 },
		{ 2226, 2225 },
		{ 1205, 1186 },
		{ 2228, 2227 },
		{ 1808, 1807 },
		{ 2230, 2229 },
		{ 1620, 1602 },
		{ 2000, 1999 },
		{ 228, 191 },
		{ 1758, 1757 },
		{ 782, 722 },
		{ 1867, 1866 },
		{ 359, 309 },
		{ 1238, 1226 },
		{ 1409, 1408 },
		{ 1181, 1153 },
		{ 1240, 1229 },
		{ 1019, 972 },
		{ 1995, 1980 },
		{ 1131, 1094 },
		{ 919, 864 },
		{ 949, 895 },
		{ 1245, 1234 },
		{ 3247, 3245 },
		{ 1106, 1069 },
		{ 1247, 1236 },
		{ 784, 724 },
		{ 1001, 951 },
		{ 1064, 1021 },
		{ 892, 836 },
		{ 866, 804 },
		{ 1782, 1781 },
		{ 1293, 1292 },
		{ 1732, 1731 },
		{ 1255, 1246 },
		{ 1088, 1050 },
		{ 1089, 1051 },
		{ 867, 805 },
		{ 1223, 1206 },
		{ 954, 900 },
		{ 1262, 1253 },
		{ 1225, 1209 },
		{ 198, 178 },
		{ 814, 752 },
		{ 2027, 2016 },
		{ 841, 779 },
		{ 2243, 2231 },
		{ 890, 833 },
		{ 995, 944 },
		{ 2112, 2089 },
		{ 646, 582 },
		{ 2100, 2078 },
		{ 315, 268 },
		{ 2033, 2026 },
		{ 518, 464 },
		{ 1261, 1252 },
		{ 1914, 1890 },
		{ 1353, 1350 },
		{ 2014, 2002 },
		{ 3087, 3085 },
		{ 2207, 2193 },
		{ 229, 192 },
		{ 2898, 2898 },
		{ 2898, 2898 },
		{ 2781, 2781 },
		{ 2781, 2781 },
		{ 2970, 2970 },
		{ 2970, 2970 },
		{ 2782, 2782 },
		{ 2782, 2782 },
		{ 1823, 1823 },
		{ 1823, 1823 },
		{ 3189, 3189 },
		{ 3189, 3189 },
		{ 2902, 2902 },
		{ 2902, 2902 },
		{ 2904, 2904 },
		{ 2904, 2904 },
		{ 3054, 3054 },
		{ 3054, 3054 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2977, 2977 },
		{ 2977, 2977 },
		{ 1003, 953 },
		{ 2898, 2898 },
		{ 1004, 954 },
		{ 2781, 2781 },
		{ 1292, 1291 },
		{ 2970, 2970 },
		{ 610, 548 },
		{ 2782, 2782 },
		{ 328, 280 },
		{ 1823, 1823 },
		{ 882, 823 },
		{ 3189, 3189 },
		{ 555, 493 },
		{ 2902, 2902 },
		{ 278, 234 },
		{ 2904, 2904 },
		{ 1828, 1825 },
		{ 3054, 3054 },
		{ 1078, 1039 },
		{ 2905, 2905 },
		{ 944, 890 },
		{ 2977, 2977 },
		{ 829, 768 },
		{ 2648, 2648 },
		{ 2648, 2648 },
		{ 2907, 2907 },
		{ 2907, 2907 },
		{ 2917, 2898 },
		{ 1012, 962 },
		{ 2810, 2781 },
		{ 830, 769 },
		{ 2972, 2970 },
		{ 947, 893 },
		{ 2811, 2782 },
		{ 778, 718 },
		{ 1824, 1823 },
		{ 557, 495 },
		{ 3190, 3189 },
		{ 3152, 3150 },
		{ 2921, 2902 },
		{ 780, 720 },
		{ 2922, 2904 },
		{ 500, 445 },
		{ 3055, 3054 },
		{ 342, 293 },
		{ 2923, 2905 },
		{ 2648, 2648 },
		{ 2978, 2977 },
		{ 2907, 2907 },
		{ 1162, 1131 },
		{ 2908, 2908 },
		{ 2908, 2908 },
		{ 1383, 1383 },
		{ 1383, 1383 },
		{ 1747, 1747 },
		{ 1747, 1747 },
		{ 2859, 2859 },
		{ 2859, 2859 },
		{ 1426, 1426 },
		{ 1426, 1426 },
		{ 2611, 2611 },
		{ 2611, 2611 },
		{ 2654, 2654 },
		{ 2654, 2654 },
		{ 2790, 2790 },
		{ 2790, 2790 },
		{ 1398, 1398 },
		{ 1398, 1398 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2919, 2919 },
		{ 2919, 2919 },
		{ 2684, 2648 },
		{ 2908, 2908 },
		{ 2925, 2907 },
		{ 1383, 1383 },
		{ 197, 177 },
		{ 1747, 1747 },
		{ 1090, 1052 },
		{ 2859, 2859 },
		{ 469, 421 },
		{ 1426, 1426 },
		{ 894, 838 },
		{ 2611, 2611 },
		{ 387, 341 },
		{ 2654, 2654 },
		{ 233, 195 },
		{ 2790, 2790 },
		{ 373, 323 },
		{ 1398, 1398 },
		{ 332, 284 },
		{ 2677, 2677 },
		{ 699, 629 },
		{ 2919, 2919 },
		{ 286, 242 },
		{ 2762, 2762 },
		{ 2762, 2762 },
		{ 2924, 2924 },
		{ 2924, 2924 },
		{ 2926, 2908 },
		{ 845, 783 },
		{ 1384, 1383 },
		{ 792, 731 },
		{ 1748, 1747 },
		{ 964, 910 },
		{ 2883, 2859 },
		{ 793, 732 },
		{ 1427, 1426 },
		{ 1249, 1238 },
		{ 2646, 2611 },
		{ 451, 405 },
		{ 2689, 2654 },
		{ 744, 677 },
		{ 2819, 2790 },
		{ 597, 535 },
		{ 1399, 1398 },
		{ 746, 681 },
		{ 2678, 2677 },
		{ 2762, 2762 },
		{ 2936, 2919 },
		{ 2924, 2924 },
		{ 1254, 1245 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 2927, 2927 },
		{ 2927, 2927 },
		{ 1721, 1721 },
		{ 1721, 1721 },
		{ 2614, 2614 },
		{ 2614, 2614 },
		{ 2930, 2930 },
		{ 2930, 2930 },
		{ 3148, 3148 },
		{ 3148, 3148 },
		{ 2542, 2542 },
		{ 2542, 2542 },
		{ 2836, 2836 },
		{ 2836, 2836 },
		{ 2837, 2837 },
		{ 2837, 2837 },
		{ 3083, 3083 },
		{ 3083, 3083 },
		{ 1880, 1880 },
		{ 1880, 1880 },
		{ 2792, 2762 },
		{ 3010, 3010 },
		{ 2938, 2924 },
		{ 2927, 2927 },
		{ 266, 224 },
		{ 1721, 1721 },
		{ 630, 566 },
		{ 2614, 2614 },
		{ 705, 636 },
		{ 2930, 2930 },
		{ 667, 600 },
		{ 3148, 3148 },
		{ 913, 858 },
		{ 2542, 2542 },
		{ 2265, 2262 },
		{ 2836, 2836 },
		{ 668, 601 },
		{ 2837, 2837 },
		{ 976, 922 },
		{ 3083, 3083 },
		{ 1044, 1001 },
		{ 1880, 1880 },
		{ 915, 860 },
		{ 2935, 2935 },
		{ 2935, 2935 },
		{ 2485, 2485 },
		{ 2485, 2485 },
		{ 3011, 3010 },
		{ 1264, 1255 },
		{ 2939, 2927 },
		{ 708, 639 },
		{ 1722, 1721 },
		{ 1348, 1347 },
		{ 2649, 2614 },
		{ 709, 640 },
		{ 2942, 2930 },
		{ 710, 641 },
		{ 3149, 3148 },
		{ 377, 329 },
		{ 2576, 2542 },
		{ 3105, 3102 },
		{ 2860, 2836 },
		{ 3014, 3012 },
		{ 2861, 2837 },
		{ 670, 603 },
		{ 3084, 3083 },
		{ 2935, 2935 },
		{ 1881, 1880 },
		{ 2485, 2485 },
		{ 394, 348 },
		{ 2319, 2319 },
		{ 2319, 2319 },
		{ 1797, 1797 },
		{ 1797, 1797 },
		{ 2717, 2717 },
		{ 2717, 2717 },
		{ 1348, 1348 },
		{ 1348, 1348 },
		{ 2884, 2884 },
		{ 2884, 2884 },
		{ 2949, 2949 },
		{ 2949, 2949 },
		{ 1857, 1857 },
		{ 1857, 1857 },
		{ 1335, 1335 },
		{ 1335, 1335 },
		{ 2952, 2952 },
		{ 2952, 2952 },
		{ 2953, 2953 },
		{ 2953, 2953 },
		{ 2747, 2747 },
		{ 2747, 2747 },
		{ 2947, 2935 },
		{ 2319, 2319 },
		{ 2486, 2485 },
		{ 1797, 1797 },
		{ 714, 648 },
		{ 2717, 2717 },
		{ 672, 605 },
		{ 1348, 1348 },
		{ 303, 256 },
		{ 2884, 2884 },
		{ 250, 212 },
		{ 2949, 2949 },
		{ 350, 301 },
		{ 1857, 1857 },
		{ 516, 461 },
		{ 1335, 1335 },
		{ 869, 809 },
		{ 2952, 2952 },
		{ 417, 372 },
		{ 2953, 2953 },
		{ 1060, 1017 },
		{ 2747, 2747 },
		{ 638, 574 },
		{ 3100, 3100 },
		{ 3100, 3100 },
		{ 1771, 1771 },
		{ 1771, 1771 },
		{ 2320, 2319 },
		{ 818, 756 },
		{ 1798, 1797 },
		{ 819, 757 },
		{ 2748, 2717 },
		{ 723, 656 },
		{ 1349, 1348 },
		{ 996, 946 },
		{ 2903, 2884 },
		{ 934, 880 },
		{ 2956, 2949 },
		{ 578, 515 },
		{ 1858, 1857 },
		{ 398, 353 },
		{ 1336, 1335 },
		{ 338, 289 },
		{ 2959, 2952 },
		{ 327, 279 },
		{ 2960, 2953 },
		{ 3100, 3100 },
		{ 2778, 2747 },
		{ 1771, 1771 },
		{ 2032, 2025 },
		{ 2957, 2957 },
		{ 2957, 2957 },
		{ 2777, 2777 },
		{ 2777, 2777 },
		{ 2644, 2644 },
		{ 2644, 2644 },
		{ 2850, 2850 },
		{ 2850, 2850 },
		{ 2851, 2851 },
		{ 2851, 2851 },
		{ 2965, 2965 },
		{ 2965, 2965 },
		{ 2551, 2551 },
		{ 2551, 2551 },
		{ 3323, 3323 },
		{ 3331, 3331 },
		{ 3335, 3335 },
		{ 2238, 2222 },
		{ 1654, 1636 },
		{ 2239, 2224 },
		{ 1651, 1627 },
		{ 2240, 2226 },
		{ 3101, 3100 },
		{ 2957, 2957 },
		{ 1772, 1771 },
		{ 2777, 2777 },
		{ 1653, 1633 },
		{ 2644, 2644 },
		{ 2241, 2228 },
		{ 2850, 2850 },
		{ 1655, 1639 },
		{ 2851, 2851 },
		{ 2242, 2230 },
		{ 2965, 2965 },
		{ 2012, 2000 },
		{ 2551, 2551 },
		{ 1695, 1694 },
		{ 3323, 3323 },
		{ 3331, 3331 },
		{ 3335, 3335 },
		{ 2205, 2191 },
		{ 1670, 1660 },
		{ 2055, 2054 },
		{ 2041, 2035 },
		{ 1676, 1669 },
		{ 1652, 1630 },
		{ 2276, 2274 },
		{ 2252, 2245 },
		{ 1608, 1590 },
		{ 2963, 2957 },
		{ 2044, 2040 },
		{ 2807, 2777 },
		{ 2031, 2023 },
		{ 2681, 2644 },
		{ 1656, 1642 },
		{ 2874, 2850 },
		{ 2237, 2220 },
		{ 2875, 2851 },
		{ 2257, 2251 },
		{ 2968, 2965 },
		{ 1814, 1813 },
		{ 2552, 2551 },
		{ 1709, 1708 },
		{ 3326, 3323 },
		{ 3333, 3331 },
		{ 3336, 3335 },
		{ 1710, 1709 },
		{ 1737, 1736 },
		{ 3297, 3296 },
		{ 3298, 3297 },
		{ 1872, 1871 },
		{ 1873, 1872 },
		{ 1738, 1737 },
		{ 3221, 3220 },
		{ 1787, 1786 },
		{ 1788, 1787 },
		{ 1415, 1414 },
		{ 1686, 1682 },
		{ 1763, 1762 },
		{ 1764, 1763 },
		{ 1414, 1413 },
		{ 1682, 1677 },
		{ 1683, 1678 },
		{ 1458, 1457 },
		{ 1813, 1812 },
		{ 2984, 2984 },
		{ 2981, 2984 },
		{ 165, 165 },
		{ 162, 165 },
		{ 1984, 1969 },
		{ 1985, 1969 },
		{ 2006, 1994 },
		{ 2007, 1994 },
		{ 2286, 2286 },
		{ 2060, 2060 },
		{ 164, 160 },
		{ 2989, 2985 },
		{ 170, 166 },
		{ 2346, 2321 },
		{ 90, 72 },
		{ 163, 160 },
		{ 2988, 2985 },
		{ 169, 166 },
		{ 2345, 2321 },
		{ 89, 72 },
		{ 2137, 2116 },
		{ 2291, 2287 },
		{ 2983, 2979 },
		{ 2984, 2984 },
		{ 2065, 2061 },
		{ 165, 165 },
		{ 2290, 2287 },
		{ 2982, 2979 },
		{ 2478, 2446 },
		{ 2064, 2061 },
		{ 171, 168 },
		{ 2286, 2286 },
		{ 2060, 2060 },
		{ 173, 172 },
		{ 2990, 2987 },
		{ 2992, 2991 },
		{ 121, 105 },
		{ 2413, 2378 },
		{ 2985, 2984 },
		{ 1944, 1925 },
		{ 166, 165 },
		{ 3232, 3224 },
		{ 2066, 2063 },
		{ 2068, 2067 },
		{ 2292, 2289 },
		{ 2294, 2293 },
		{ 2287, 2286 },
		{ 2061, 2060 },
		{ 2822, 2794 },
		{ 2281, 2280 },
		{ 2735, 2704 },
		{ 2135, 2114 },
		{ 2019, 2009 },
		{ 2293, 2291 },
		{ 2063, 2059 },
		{ 2378, 2346 },
		{ 105, 90 },
		{ 1925, 1903 },
		{ 172, 170 },
		{ 2987, 2983 },
		{ 2289, 2285 },
		{ 2067, 2065 },
		{ 2116, 2094 },
		{ 2991, 2989 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 0, 1322 },
		{ 0, 3211 },
		{ 2678, 2678 },
		{ 2678, 2678 },
		{ 0, 3106 },
		{ 2649, 2649 },
		{ 2649, 2649 },
		{ 2923, 2923 },
		{ 2923, 2923 },
		{ 0, 2621 },
		{ 0, 3110 },
		{ 0, 3015 },
		{ 0, 2760 },
		{ 0, 2308 },
		{ 2807, 2807 },
		{ 2807, 2807 },
		{ 2861, 2861 },
		{ 2861, 2861 },
		{ 0, 3020 },
		{ 2068, 2068 },
		{ 2069, 2068 },
		{ 2917, 2917 },
		{ 2684, 2684 },
		{ 2684, 2684 },
		{ 0, 3120 },
		{ 2678, 2678 },
		{ 2810, 2810 },
		{ 2810, 2810 },
		{ 2649, 2649 },
		{ 0, 3123 },
		{ 2923, 2923 },
		{ 2811, 2811 },
		{ 2811, 2811 },
		{ 2936, 2936 },
		{ 2936, 2936 },
		{ 0, 2937 },
		{ 0, 3128 },
		{ 2807, 2807 },
		{ 0, 1343 },
		{ 2861, 2861 },
		{ 2938, 2938 },
		{ 2938, 2938 },
		{ 2068, 2068 },
		{ 2939, 2939 },
		{ 2939, 2939 },
		{ 2684, 2684 },
		{ 0, 3031 },
		{ 2576, 2576 },
		{ 2576, 2576 },
		{ 2810, 2810 },
		{ 2942, 2942 },
		{ 2942, 2942 },
		{ 0, 2943 },
		{ 0, 2869 },
		{ 2811, 2811 },
		{ 0, 3038 },
		{ 2936, 2936 },
		{ 2294, 2294 },
		{ 2295, 2294 },
		{ 2689, 2689 },
		{ 2689, 2689 },
		{ 2947, 2947 },
		{ 2947, 2947 },
		{ 2938, 2938 },
		{ 0, 3139 },
		{ 0, 2727 },
		{ 2939, 2939 },
		{ 0, 2770 },
		{ 2874, 2874 },
		{ 2874, 2874 },
		{ 2576, 2576 },
		{ 2875, 2875 },
		{ 2875, 2875 },
		{ 2942, 2942 },
		{ 0, 2728 },
		{ 0, 3045 },
		{ 0, 1374 },
		{ 0, 2578 },
		{ 0, 2823 },
		{ 0, 2824 },
		{ 2294, 2294 },
		{ 0, 2825 },
		{ 2689, 2689 },
		{ 0, 3153 },
		{ 2947, 2947 },
		{ 0, 2578 },
		{ 2956, 2956 },
		{ 2956, 2956 },
		{ 2959, 2959 },
		{ 2959, 2959 },
		{ 0, 2882 },
		{ 2874, 2874 },
		{ 2960, 2960 },
		{ 2960, 2960 },
		{ 2875, 2875 },
		{ 0, 3157 },
		{ 2883, 2883 },
		{ 2883, 2883 },
		{ 2963, 2963 },
		{ 2963, 2963 },
		{ 0, 1832 },
		{ 0, 3161 },
		{ 0, 2885 },
		{ 0, 1390 },
		{ 0, 3164 },
		{ 0, 2694 },
		{ 2968, 2968 },
		{ 2968, 2968 },
		{ 0, 2888 },
		{ 2956, 2956 },
		{ 0, 2889 },
		{ 2959, 2959 },
		{ 0, 3169 },
		{ 0, 2829 },
		{ 0, 3066 },
		{ 2960, 2960 },
		{ 2972, 2972 },
		{ 2972, 2972 },
		{ 0, 2449 },
		{ 2883, 2883 },
		{ 0, 2892 },
		{ 2963, 2963 },
		{ 2778, 2778 },
		{ 2778, 2778 },
		{ 0, 1843 },
		{ 0, 1358 },
		{ 0, 2635 },
		{ 2978, 2978 },
		{ 2978, 2978 },
		{ 2968, 2968 },
		{ 0, 3180 },
		{ 0, 3075 },
		{ 0, 2609 },
		{ 2486, 2486 },
		{ 2486, 2486 },
		{ 0, 1369 },
		{ 0, 2741 },
		{ 0, 2702 },
		{ 0, 2564 },
		{ 2972, 2972 },
		{ 173, 173 },
		{ 174, 173 },
		{ 2903, 2903 },
		{ 2903, 2903 },
		{ 0, 2565 },
		{ 2778, 2778 },
		{ 0, 3066 },
		{ 2992, 2992 },
		{ 2993, 2992 },
		{ 0, 2449 },
		{ 2978, 2978 },
		{ 0, 2566 },
		{ 0, 2707 },
		{ 2792, 2792 },
		{ 2792, 2792 },
		{ 0, 3091 },
		{ 2486, 2486 },
		{ 2748, 2748 },
		{ 2748, 2748 },
		{ 0, 3200 },
		{ 0, 3000 },
		{ 0, 1848 },
		{ 0, 2711 },
		{ 173, 173 },
		{ 0, 2528 },
		{ 2903, 2903 },
		{ 2646, 2646 },
		{ 2646, 2646 },
		{ 0, 2714 },
		{ 1603, 1603 },
		{ 2992, 2992 },
		{ 1301, 1301 },
		{ 1445, 1444 },
		{ 1436, 1435 },
		{ 167, 169 },
		{ 0, 2345 },
		{ 2792, 2792 },
		{ 2986, 2982 },
		{ 3236, 3232 },
		{ 1963, 1944 },
		{ 2748, 2748 },
		{ 0, 89 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2646, 2646 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1603, 1603 },
		{ 0, 0 },
		{ 1301, 1301 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -70, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 46, 433, 0 },
		{ -179, 3175, 0 },
		{ 5, 0, 0 },
		{ -1300, 1095, -31 },
		{ 7, 0, -31 },
		{ -1304, 2088, -33 },
		{ 9, 0, -33 },
		{ -1317, 3510, 160 },
		{ 11, 0, 160 },
		{ -1338, 3517, 164 },
		{ 13, 0, 164 },
		{ -1351, 3511, 172 },
		{ 15, 0, 172 },
		{ -1386, 3663, 0 },
		{ 17, 0, 0 },
		{ -1401, 3518, 156 },
		{ 19, 0, 156 },
		{ -1429, 3533, 23 },
		{ 21, 0, 23 },
		{ -1471, 547, 0 },
		{ 23, 0, 0 },
		{ -1697, 3665, 0 },
		{ 25, 0, 0 },
		{ -1724, 3535, 0 },
		{ 27, 0, 0 },
		{ -1750, 3537, 0 },
		{ 29, 0, 0 },
		{ -1774, 3512, 0 },
		{ 31, 0, 0 },
		{ -1800, 3523, 0 },
		{ 33, 0, 0 },
		{ -1826, 3527, 176 },
		{ 35, 0, 176 },
		{ -1860, 3660, 283 },
		{ 37, 0, 283 },
		{ 40, 218, 0 },
		{ -1896, 661, 0 },
		{ 42, 216, 0 },
		{ -2085, 205, 0 },
		{ -2297, 3661, 0 },
		{ 43, 0, 0 },
		{ 46, 14, 0 },
		{ -88, 319, 0 },
		{ -2995, 3529, 168 },
		{ 47, 0, 168 },
		{ -3013, 3664, 191 },
		{ 49, 0, 191 },
		{ 3057, 1615, 0 },
		{ 51, 0, 0 },
		{ -3059, 3667, 289 },
		{ 53, 0, 289 },
		{ -3086, 3670, 194 },
		{ 55, 0, 194 },
		{ -3104, 3516, 187 },
		{ 57, 0, 187 },
		{ -3151, 3662, 180 },
		{ 59, 0, 180 },
		{ -3192, 3539, 186 },
		{ 61, 0, 186 },
		{ 46, 1, 0 },
		{ 63, 0, 0 },
		{ -3243, 2054, 0 },
		{ 65, 0, 0 },
		{ -3253, 2108, 45 },
		{ 67, 0, 45 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 450 },
		{ 3224, 5227, 457 },
		{ 0, 0, 261 },
		{ 0, 0, 263 },
		{ 159, 1301, 280 },
		{ 159, 1419, 280 },
		{ 159, 1384, 280 },
		{ 159, 1406, 280 },
		{ 159, 1410, 280 },
		{ 159, 1441, 280 },
		{ 159, 1447, 280 },
		{ 159, 1469, 280 },
		{ 3317, 3206, 457 },
		{ 159, 1480, 280 },
		{ 3317, 1904, 279 },
		{ 104, 2970, 457 },
		{ 159, 0, 280 },
		{ 0, 0, 457 },
		{ -89, 5492, 257 },
		{ -90, 5269, 0 },
		{ 159, 1471, 280 },
		{ 159, 1515, 280 },
		{ 159, 1484, 280 },
		{ 159, 1498, 280 },
		{ 159, 1480, 280 },
		{ 159, 1480, 280 },
		{ 159, 1487, 280 },
		{ 159, 1502, 280 },
		{ 159, 1521, 280 },
		{ 3276, 2612, 0 },
		{ 159, 833, 280 },
		{ 3317, 2057, 276 },
		{ 119, 1618, 0 },
		{ 3317, 2034, 277 },
		{ 3224, 5244, 0 },
		{ 159, 842, 280 },
		{ 159, 840, 280 },
		{ 159, 841, 280 },
		{ 159, 837, 280 },
		{ 159, 0, 268 },
		{ 159, 882, 280 },
		{ 159, 884, 280 },
		{ 159, 868, 280 },
		{ 159, 873, 280 },
		{ 3314, 3292, 0 },
		{ 159, 881, 280 },
		{ 133, 1709, 0 },
		{ 119, 0, 0 },
		{ 3226, 2978, 278 },
		{ 135, 1617, 0 },
		{ 0, 0, 259 },
		{ 159, 913, 264 },
		{ 159, 915, 280 },
		{ 159, 907, 280 },
		{ 159, 912, 280 },
		{ 159, 910, 280 },
		{ 159, 903, 280 },
		{ 159, 0, 271 },
		{ 159, 904, 280 },
		{ 0, 0, 273 },
		{ 159, 910, 280 },
		{ 133, 0, 0 },
		{ 3226, 3040, 276 },
		{ 135, 0, 0 },
		{ 3226, 2948, 277 },
		{ 159, 925, 280 },
		{ 159, 922, 280 },
		{ 159, 923, 280 },
		{ 159, 958, 280 },
		{ 159, 1054, 280 },
		{ 159, 0, 270 },
		{ 159, 1163, 280 },
		{ 159, 1236, 280 },
		{ 159, 1228, 280 },
		{ 159, 0, 266 },
		{ 159, 1308, 280 },
		{ 159, 0, 267 },
		{ 159, 0, 269 },
		{ 159, 1276, 280 },
		{ 159, 1276, 280 },
		{ 159, 0, 265 },
		{ 159, 1297, 280 },
		{ 159, 0, 272 },
		{ 159, 890, 280 },
		{ 159, 1323, 280 },
		{ 0, 0, 275 },
		{ 159, 1306, 280 },
		{ 159, 1308, 280 },
		{ 3334, 1376, 274 },
		{ 3224, 5223, 457 },
		{ 165, 0, 261 },
		{ 0, 0, 262 },
		{ -163, 20, 257 },
		{ -164, 1, 0 },
		{ 3280, 5248, 0 },
		{ 3224, 5225, 0 },
		{ 0, 0, 258 },
		{ 3224, 5238, 0 },
		{ -169, 5485, 0 },
		{ -170, 5271, 0 },
		{ 173, 0, 259 },
		{ 3224, 5241, 0 },
		{ 3280, 5452, 0 },
		{ 0, 0, 260 },
		{ 3281, 1861, 154 },
		{ 2197, 4602, 154 },
		{ 3150, 4915, 154 },
		{ 3281, 4795, 154 },
		{ 0, 0, 154 },
		{ 3266, 3762, 0 },
		{ 2212, 3151, 0 },
		{ 3266, 4019, 0 },
		{ 3266, 3665, 0 },
		{ 3254, 3747, 0 },
		{ 2197, 4533, 0 },
		{ 3271, 3649, 0 },
		{ 2197, 4517, 0 },
		{ 3241, 3844, 0 },
		{ 3275, 3619, 0 },
		{ 3241, 3693, 0 },
		{ 3281, 4761, 0 },
		{ 3085, 4815, 0 },
		{ 3271, 3688, 0 },
		{ 2212, 4125, 0 },
		{ 3150, 4925, 0 },
		{ 2143, 3851, 0 },
		{ 2143, 3858, 0 },
		{ 2165, 3505, 0 },
		{ 2146, 4240, 0 },
		{ 2127, 4272, 0 },
		{ 2146, 4257, 0 },
		{ 2165, 3511, 0 },
		{ 2165, 3542, 0 },
		{ 3271, 3720, 0 },
		{ 3191, 4383, 0 },
		{ 3266, 4050, 0 },
		{ 2197, 4582, 0 },
		{ 1270, 4479, 0 },
		{ 2143, 3889, 0 },
		{ 2165, 3550, 0 },
		{ 2165, 3552, 0 },
		{ 3150, 5065, 0 },
		{ 2143, 3882, 0 },
		{ 3254, 4179, 0 },
		{ 3266, 4029, 0 },
		{ 2212, 4153, 0 },
		{ 2686, 4645, 0 },
		{ 3150, 4042, 0 },
		{ 3191, 4377, 0 },
		{ 3241, 3943, 0 },
		{ 3241, 3848, 0 },
		{ 3241, 3975, 0 },
		{ 2105, 3693, 0 },
		{ 3150, 4987, 0 },
		{ 3266, 4070, 0 },
		{ 3191, 4038, 0 },
		{ 2212, 4144, 0 },
		{ 2165, 3566, 0 },
		{ 2165, 3485, 0 },
		{ 3275, 3814, 0 },
		{ 3254, 4107, 0 },
		{ 2105, 3694, 0 },
		{ 2127, 4277, 0 },
		{ 3150, 4853, 0 },
		{ 2197, 4564, 0 },
		{ 2197, 4565, 0 },
		{ 3266, 4065, 0 },
		{ 2165, 3486, 0 },
		{ 2197, 4514, 0 },
		{ 3266, 4074, 0 },
		{ 2686, 4639, 0 },
		{ 3150, 4933, 0 },
		{ 3275, 3743, 0 },
		{ 3191, 4360, 0 },
		{ 3241, 3967, 0 },
		{ 2165, 3487, 0 },
		{ 3275, 3714, 0 },
		{ 3266, 4015, 0 },
		{ 1270, 4496, 0 },
		{ 2127, 4301, 0 },
		{ 3191, 4419, 0 },
		{ 2212, 4044, 0 },
		{ 1155, 3637, 0 },
		{ 3266, 4072, 0 },
		{ 3254, 4226, 0 },
		{ 3150, 5063, 0 },
		{ 2686, 4672, 0 },
		{ 2165, 3488, 0 },
		{ 2212, 4139, 0 },
		{ 3275, 3807, 0 },
		{ 1011, 4330, 0 },
		{ 2197, 4574, 0 },
		{ 2105, 3684, 0 },
		{ 2105, 3685, 0 },
		{ 2197, 4604, 0 },
		{ 3241, 3927, 0 },
		{ 2165, 3489, 0 },
		{ 3085, 4806, 0 },
		{ 3254, 4199, 0 },
		{ 3275, 3766, 0 },
		{ 2143, 3844, 0 },
		{ 2233, 4041, 0 },
		{ 2165, 3490, 0 },
		{ 3191, 4365, 0 },
		{ 3241, 3942, 0 },
		{ 2197, 4508, 0 },
		{ 2686, 4717, 0 },
		{ 2197, 4511, 0 },
		{ 3150, 5099, 0 },
		{ 3150, 4847, 0 },
		{ 2127, 4291, 0 },
		{ 2686, 4651, 0 },
		{ 2165, 3491, 0 },
		{ 3150, 4929, 0 },
		{ 3191, 4428, 0 },
		{ 2127, 4304, 0 },
		{ 3191, 4338, 0 },
		{ 2686, 4649, 0 },
		{ 3150, 5097, 0 },
		{ 2143, 3872, 0 },
		{ 3241, 3972, 0 },
		{ 2197, 4599, 0 },
		{ 3150, 4883, 0 },
		{ 1270, 4474, 0 },
		{ 3191, 4401, 0 },
		{ 1155, 3630, 0 },
		{ 2233, 4471, 0 },
		{ 2146, 4244, 0 },
		{ 3241, 3923, 0 },
		{ 2165, 3493, 0 },
		{ 3150, 5067, 0 },
		{ 2197, 4555, 0 },
		{ 2165, 3494, 0 },
		{ 0, 0, 78 },
		{ 3266, 3794, 0 },
		{ 3275, 3804, 0 },
		{ 2197, 4577, 0 },
		{ 3191, 4414, 0 },
		{ 3281, 4765, 0 },
		{ 2197, 4587, 0 },
		{ 2165, 3495, 0 },
		{ 2165, 3496, 0 },
		{ 3275, 3831, 0 },
		{ 2143, 3908, 0 },
		{ 2127, 4292, 0 },
		{ 3275, 3742, 0 },
		{ 2165, 3497, 0 },
		{ 3191, 4410, 0 },
		{ 2197, 4553, 0 },
		{ 3266, 4076, 0 },
		{ 3266, 4049, 0 },
		{ 2146, 4262, 0 },
		{ 3150, 4927, 0 },
		{ 3241, 3954, 0 },
		{ 904, 3653, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2143, 3870, 0 },
		{ 3150, 5021, 0 },
		{ 3191, 4382, 0 },
		{ 2127, 4313, 0 },
		{ 3275, 3763, 0 },
		{ 3241, 3987, 0 },
		{ 0, 0, 76 },
		{ 2165, 3498, 0 },
		{ 2143, 3886, 0 },
		{ 0, 0, 133 },
		{ 3275, 3770, 0 },
		{ 3191, 4337, 0 },
		{ 3275, 3776, 0 },
		{ 3150, 4923, 0 },
		{ 3241, 3950, 0 },
		{ 1270, 4484, 0 },
		{ 2127, 4269, 0 },
		{ 2143, 3909, 0 },
		{ 3254, 4212, 0 },
		{ 2197, 4576, 0 },
		{ 3150, 5032, 0 },
		{ 3281, 4747, 0 },
		{ 3241, 3960, 0 },
		{ 0, 0, 70 },
		{ 3241, 3962, 0 },
		{ 3150, 5095, 0 },
		{ 1270, 4501, 0 },
		{ 3191, 4413, 0 },
		{ 2197, 4600, 0 },
		{ 0, 0, 81 },
		{ 3254, 4180, 0 },
		{ 3275, 3777, 0 },
		{ 3241, 3970, 0 },
		{ 3266, 4013, 0 },
		{ 3266, 4066, 0 },
		{ 2165, 3499, 0 },
		{ 3191, 4376, 0 },
		{ 2197, 4543, 0 },
		{ 2165, 3500, 0 },
		{ 2143, 3863, 0 },
		{ 2178, 3623, 0 },
		{ 3254, 4232, 0 },
		{ 3275, 3808, 0 },
		{ 3241, 3928, 0 },
		{ 3150, 5073, 0 },
		{ 3275, 3813, 0 },
		{ 2165, 3501, 0 },
		{ 3191, 4422, 0 },
		{ 2197, 4595, 0 },
		{ 3275, 3817, 0 },
		{ 3241, 3952, 0 },
		{ 3191, 4350, 0 },
		{ 1089, 4444, 0 },
		{ 0, 0, 8 },
		{ 2143, 3895, 0 },
		{ 2146, 4261, 0 },
		{ 3191, 4371, 0 },
		{ 2178, 3615, 0 },
		{ 2165, 3502, 0 },
		{ 2686, 4668, 0 },
		{ 2197, 4542, 0 },
		{ 2143, 3911, 0 },
		{ 2197, 4545, 0 },
		{ 2197, 4546, 0 },
		{ 2146, 4245, 0 },
		{ 2165, 3503, 0 },
		{ 3275, 3745, 0 },
		{ 3275, 3662, 0 },
		{ 2197, 4567, 0 },
		{ 3271, 3726, 0 },
		{ 3241, 3998, 0 },
		{ 1270, 4488, 0 },
		{ 3254, 4205, 0 },
		{ 2165, 3504, 0 },
		{ 2233, 4447, 0 },
		{ 2212, 3440, 0 },
		{ 2165, 3506, 0 },
		{ 3150, 4949, 0 },
		{ 1270, 4486, 0 },
		{ 2212, 4109, 0 },
		{ 3281, 3612, 0 },
		{ 2178, 3617, 0 },
		{ 2146, 4246, 0 },
		{ 2143, 3892, 0 },
		{ 3275, 3800, 0 },
		{ 2197, 4536, 0 },
		{ 0, 0, 123 },
		{ 2165, 3510, 0 },
		{ 2212, 4111, 0 },
		{ 2104, 3645, 0 },
		{ 3191, 4412, 0 },
		{ 3266, 4042, 0 },
		{ 3254, 4194, 0 },
		{ 3150, 4919, 0 },
		{ 2197, 4562, 0 },
		{ 0, 0, 7 },
		{ 2146, 4248, 0 },
		{ 0, 0, 6 },
		{ 3191, 4420, 0 },
		{ 0, 0, 128 },
		{ 3254, 4195, 0 },
		{ 2197, 4568, 0 },
		{ 3281, 2103, 0 },
		{ 2165, 3512, 0 },
		{ 3254, 4211, 0 },
		{ 3266, 4063, 0 },
		{ 0, 0, 132 },
		{ 2165, 3516, 0 },
		{ 2197, 4593, 0 },
		{ 3281, 3632, 0 },
		{ 2197, 4596, 0 },
		{ 2686, 4734, 0 },
		{ 2212, 4132, 0 },
		{ 0, 0, 77 },
		{ 2127, 4267, 0 },
		{ 2165, 3517, 115 },
		{ 2165, 3518, 116 },
		{ 3150, 4881, 0 },
		{ 3191, 4385, 0 },
		{ 2165, 3519, 0 },
		{ 3241, 4001, 0 },
		{ 3266, 4081, 0 },
		{ 3266, 4096, 0 },
		{ 3241, 4002, 0 },
		{ 1270, 4502, 0 },
		{ 3266, 4045, 0 },
		{ 3241, 4004, 0 },
		{ 3271, 3704, 0 },
		{ 2212, 4164, 0 },
		{ 3191, 4436, 0 },
		{ 2197, 4556, 0 },
		{ 2165, 3521, 0 },
		{ 3275, 3762, 0 },
		{ 3150, 5069, 0 },
		{ 0, 0, 114 },
		{ 3191, 4353, 0 },
		{ 3085, 4808, 0 },
		{ 3191, 4355, 0 },
		{ 2212, 4122, 0 },
		{ 3241, 3935, 0 },
		{ 3191, 4369, 0 },
		{ 0, 0, 9 },
		{ 2165, 3522, 0 },
		{ 3191, 4374, 0 },
		{ 2178, 3627, 0 },
		{ 2233, 4463, 0 },
		{ 0, 0, 112 },
		{ 2143, 3901, 0 },
		{ 3254, 4231, 0 },
		{ 3266, 4021, 0 },
		{ 2212, 4029, 0 },
		{ 3254, 3623, 0 },
		{ 3191, 4402, 0 },
		{ 3271, 3699, 0 },
		{ 3191, 4411, 0 },
		{ 3271, 3678, 0 },
		{ 3266, 4048, 0 },
		{ 2197, 4521, 0 },
		{ 3275, 3779, 0 },
		{ 3241, 3963, 0 },
		{ 3271, 3652, 0 },
		{ 3254, 4214, 0 },
		{ 3275, 3788, 0 },
		{ 3191, 4346, 0 },
		{ 3275, 3717, 0 },
		{ 3150, 4851, 0 },
		{ 2165, 3523, 0 },
		{ 3150, 4875, 0 },
		{ 3241, 3986, 0 },
		{ 2197, 4561, 0 },
		{ 3266, 4017, 0 },
		{ 3266, 4025, 0 },
		{ 2127, 4319, 0 },
		{ 2143, 3871, 0 },
		{ 3266, 4086, 0 },
		{ 2165, 3525, 102 },
		{ 3241, 4000, 0 },
		{ 2212, 4159, 0 },
		{ 2165, 3527, 0 },
		{ 2165, 3528, 0 },
		{ 3271, 3709, 0 },
		{ 2212, 4119, 0 },
		{ 2686, 4670, 0 },
		{ 2165, 3529, 0 },
		{ 2143, 3893, 0 },
		{ 0, 0, 111 },
		{ 2686, 4732, 0 },
		{ 3150, 5093, 0 },
		{ 3275, 3825, 0 },
		{ 3275, 3828, 0 },
		{ 0, 0, 125 },
		{ 0, 0, 127 },
		{ 3254, 4197, 0 },
		{ 2212, 4141, 0 },
		{ 2143, 3902, 0 },
		{ 2197, 3755, 0 },
		{ 3275, 3836, 0 },
		{ 2197, 4520, 0 },
		{ 2165, 3531, 0 },
		{ 2197, 4524, 0 },
		{ 3191, 4429, 0 },
		{ 3254, 4216, 0 },
		{ 2165, 3533, 0 },
		{ 2233, 4451, 0 },
		{ 3271, 3708, 0 },
		{ 2686, 4653, 0 },
		{ 2165, 3535, 0 },
		{ 3150, 4953, 0 },
		{ 2143, 3861, 0 },
		{ 1011, 4326, 0 },
		{ 3275, 3748, 0 },
		{ 3254, 4184, 0 },
		{ 2212, 4126, 0 },
		{ 2686, 4635, 0 },
		{ 3275, 3760, 0 },
		{ 2105, 3682, 0 },
		{ 2165, 3537, 0 },
		{ 3191, 4379, 0 },
		{ 3266, 4055, 0 },
		{ 2143, 3881, 0 },
		{ 3150, 4845, 0 },
		{ 3275, 3765, 0 },
		{ 2212, 4162, 0 },
		{ 2178, 3628, 0 },
		{ 3241, 3999, 0 },
		{ 2143, 3888, 0 },
		{ 2212, 4113, 0 },
		{ 2146, 4259, 0 },
		{ 3281, 3624, 0 },
		{ 2165, 3538, 0 },
		{ 0, 0, 71 },
		{ 2165, 3539, 0 },
		{ 3266, 4085, 0 },
		{ 3241, 4005, 0 },
		{ 3266, 4090, 0 },
		{ 3241, 3921, 0 },
		{ 2165, 3540, 117 },
		{ 2127, 4303, 0 },
		{ 3150, 4989, 0 },
		{ 2212, 4151, 0 },
		{ 2146, 4260, 0 },
		{ 3241, 3926, 0 },
		{ 2143, 3899, 0 },
		{ 2143, 3900, 0 },
		{ 2127, 4270, 0 },
		{ 2146, 4241, 0 },
		{ 3150, 5077, 0 },
		{ 3266, 4011, 0 },
		{ 3271, 3703, 0 },
		{ 3191, 4378, 0 },
		{ 3275, 3787, 0 },
		{ 2143, 3906, 0 },
		{ 0, 0, 129 },
		{ 2165, 3541, 0 },
		{ 3085, 4804, 0 },
		{ 2146, 4258, 0 },
		{ 3275, 3793, 0 },
		{ 3254, 4193, 0 },
		{ 0, 0, 124 },
		{ 0, 0, 113 },
		{ 2143, 3910, 0 },
		{ 3241, 3955, 0 },
		{ 3275, 3796, 0 },
		{ 2212, 3359, 0 },
		{ 3281, 3671, 0 },
		{ 3191, 4415, 0 },
		{ 3254, 4206, 0 },
		{ 2165, 3543, 0 },
		{ 3191, 4421, 0 },
		{ 2127, 4295, 0 },
		{ 3266, 4088, 0 },
		{ 2197, 4603, 0 },
		{ 3150, 4993, 0 },
		{ 3150, 4999, 0 },
		{ 2143, 3860, 0 },
		{ 3150, 5027, 0 },
		{ 3191, 4430, 0 },
		{ 3150, 5061, 0 },
		{ 3241, 3971, 0 },
		{ 3254, 4219, 0 },
		{ 2165, 3544, 0 },
		{ 2197, 4518, 0 },
		{ 3241, 3974, 0 },
		{ 2165, 3545, 0 },
		{ 3241, 3979, 0 },
		{ 2197, 4530, 0 },
		{ 3191, 4364, 0 },
		{ 2197, 4534, 0 },
		{ 3241, 3980, 0 },
		{ 2197, 4540, 0 },
		{ 2143, 3864, 0 },
		{ 3254, 4111, 0 },
		{ 2104, 3643, 0 },
		{ 2165, 3546, 0 },
		{ 3281, 4613, 0 },
		{ 2686, 4729, 0 },
		{ 2197, 4551, 0 },
		{ 2146, 4251, 0 },
		{ 2197, 4554, 0 },
		{ 2146, 4254, 0 },
		{ 3266, 4023, 0 },
		{ 3150, 4931, 0 },
		{ 3275, 3818, 0 },
		{ 3266, 4071, 0 },
		{ 0, 0, 104 },
		{ 3275, 3820, 0 },
		{ 3191, 4387, 0 },
		{ 3191, 4397, 0 },
		{ 3150, 4991, 0 },
		{ 2165, 3547, 0 },
		{ 2165, 3548, 0 },
		{ 3150, 5013, 0 },
		{ 3150, 5017, 0 },
		{ 3150, 5019, 0 },
		{ 2146, 4238, 0 },
		{ 2143, 3887, 0 },
		{ 0, 0, 138 },
		{ 3266, 4084, 0 },
		{ 0, 0, 126 },
		{ 0, 0, 130 },
		{ 3150, 5059, 0 },
		{ 2686, 4735, 0 },
		{ 1155, 3638, 0 },
		{ 2165, 3549, 0 },
		{ 3191, 3443, 0 },
		{ 3241, 3925, 0 },
		{ 2146, 4250, 0 },
		{ 1270, 4494, 0 },
		{ 3150, 5087, 0 },
		{ 3266, 4091, 0 },
		{ 3266, 4092, 0 },
		{ 3266, 4095, 0 },
		{ 3254, 4189, 0 },
		{ 2686, 4719, 0 },
		{ 2233, 4457, 0 },
		{ 3254, 4191, 0 },
		{ 3271, 3707, 0 },
		{ 2127, 4273, 0 },
		{ 1270, 4492, 0 },
		{ 3275, 3744, 0 },
		{ 2127, 4279, 0 },
		{ 2143, 3896, 0 },
		{ 2165, 3551, 0 },
		{ 2146, 4239, 0 },
		{ 2127, 4300, 0 },
		{ 2197, 4537, 0 },
		{ 2233, 4452, 0 },
		{ 2212, 4142, 0 },
		{ 3241, 3939, 0 },
		{ 3150, 4951, 0 },
		{ 2212, 4147, 0 },
		{ 0, 0, 65 },
		{ 0, 0, 66 },
		{ 3150, 4955, 0 },
		{ 3241, 3940, 0 },
		{ 0, 0, 75 },
		{ 0, 0, 121 },
		{ 2105, 3691, 0 },
		{ 3254, 4213, 0 },
		{ 3271, 3717, 0 },
		{ 3275, 3752, 0 },
		{ 2143, 3904, 0 },
		{ 3271, 3718, 0 },
		{ 3275, 3761, 0 },
		{ 3191, 4388, 0 },
		{ 3241, 3957, 0 },
		{ 0, 0, 106 },
		{ 3241, 3958, 0 },
		{ 0, 0, 108 },
		{ 3266, 4075, 0 },
		{ 3241, 3959, 0 },
		{ 3254, 4186, 0 },
		{ 1089, 4442, 0 },
		{ 2197, 4580, 0 },
		{ 0, 0, 136 },
		{ 2178, 3621, 0 },
		{ 2178, 3622, 0 },
		{ 3275, 3764, 0 },
		{ 1270, 4487, 0 },
		{ 2233, 4176, 0 },
		{ 3241, 3964, 0 },
		{ 1011, 4328, 0 },
		{ 2127, 4265, 0 },
		{ 0, 0, 122 },
		{ 0, 0, 137 },
		{ 3241, 3965, 0 },
		{ 3241, 3966, 0 },
		{ 0, 0, 153 },
		{ 2143, 3913, 0 },
		{ 3281, 4350, 0 },
		{ 3150, 4873, 0 },
		{ 1270, 4480, 0 },
		{ 3150, 4879, 0 },
		{ 2197, 4515, 0 },
		{ 3281, 4763, 0 },
		{ 3241, 3969, 0 },
		{ 3281, 4779, 0 },
		{ 3271, 3732, 0 },
		{ 3271, 3734, 0 },
		{ 3254, 3439, 0 },
		{ 2165, 3553, 0 },
		{ 2197, 4526, 0 },
		{ 3191, 4356, 0 },
		{ 3150, 4941, 0 },
		{ 3150, 4945, 0 },
		{ 3191, 4358, 0 },
		{ 2212, 4166, 0 },
		{ 3191, 4362, 0 },
		{ 2212, 4102, 0 },
		{ 2212, 4045, 0 },
		{ 3191, 4366, 0 },
		{ 2165, 3554, 0 },
		{ 2686, 4721, 0 },
		{ 2165, 3555, 0 },
		{ 3266, 4052, 0 },
		{ 2165, 3556, 0 },
		{ 2146, 4242, 0 },
		{ 3266, 4059, 0 },
		{ 2127, 4266, 0 },
		{ 3191, 4380, 0 },
		{ 2165, 3557, 0 },
		{ 2212, 4127, 0 },
		{ 3266, 4064, 0 },
		{ 2165, 3558, 0 },
		{ 3281, 4796, 0 },
		{ 1270, 4481, 0 },
		{ 2212, 4140, 0 },
		{ 3241, 3988, 0 },
		{ 3150, 5083, 0 },
		{ 3150, 5085, 0 },
		{ 2197, 4575, 0 },
		{ 2146, 4256, 0 },
		{ 2686, 4723, 0 },
		{ 0, 0, 134 },
		{ 3241, 3997, 0 },
		{ 2197, 4578, 0 },
		{ 2197, 4579, 0 },
		{ 3191, 4403, 0 },
		{ 3191, 4406, 0 },
		{ 2197, 4585, 0 },
		{ 3150, 4861, 0 },
		{ 3150, 4869, 0 },
		{ 2197, 4586, 0 },
		{ 2165, 3560, 0 },
		{ 2212, 4146, 0 },
		{ 3275, 3794, 0 },
		{ 3275, 3795, 0 },
		{ 2197, 4597, 0 },
		{ 3271, 3689, 0 },
		{ 3271, 3729, 0 },
		{ 2127, 4305, 0 },
		{ 3281, 4798, 0 },
		{ 3275, 3802, 0 },
		{ 2165, 3561, 68 },
		{ 3275, 3805, 0 },
		{ 3150, 4939, 0 },
		{ 2212, 4167, 0 },
		{ 2165, 3562, 0 },
		{ 2165, 3563, 0 },
		{ 2233, 4453, 0 },
		{ 3191, 4432, 0 },
		{ 3281, 4745, 0 },
		{ 3254, 4170, 0 },
		{ 3275, 3809, 0 },
		{ 3275, 3810, 0 },
		{ 1155, 3631, 0 },
		{ 2127, 4278, 0 },
		{ 3241, 3932, 0 },
		{ 2178, 3607, 0 },
		{ 2105, 3695, 0 },
		{ 2105, 3680, 0 },
		{ 3266, 4046, 0 },
		{ 2143, 3905, 0 },
		{ 1270, 4495, 0 },
		{ 3271, 3715, 0 },
		{ 3241, 3944, 0 },
		{ 3281, 4783, 0 },
		{ 3281, 4790, 0 },
		{ 2197, 4549, 0 },
		{ 0, 0, 69 },
		{ 0, 0, 72 },
		{ 3150, 5071, 0 },
		{ 3191, 4339, 0 },
		{ 2146, 4237, 0 },
		{ 2127, 4309, 0 },
		{ 2233, 4446, 0 },
		{ 3241, 3945, 0 },
		{ 1270, 4482, 0 },
		{ 3241, 3948, 0 },
		{ 0, 0, 118 },
		{ 3275, 3821, 0 },
		{ 3275, 3822, 0 },
		{ 3241, 3953, 0 },
		{ 0, 0, 110 },
		{ 2165, 3564, 0 },
		{ 3150, 4849, 0 },
		{ 0, 0, 119 },
		{ 0, 0, 120 },
		{ 2212, 4158, 0 },
		{ 2127, 4271, 0 },
		{ 3254, 4217, 0 },
		{ 1011, 4331, 0 },
		{ 2146, 4247, 0 },
		{ 1270, 4504, 0 },
		{ 3275, 3826, 0 },
		{ 3085, 4801, 0 },
		{ 0, 0, 3 },
		{ 2197, 4581, 0 },
		{ 3281, 4782, 0 },
		{ 2686, 4641, 0 },
		{ 3150, 4921, 0 },
		{ 3254, 4224, 0 },
		{ 3191, 4404, 0 },
		{ 3275, 3827, 0 },
		{ 3191, 4407, 0 },
		{ 2197, 4589, 0 },
		{ 2165, 3565, 0 },
		{ 2146, 4255, 0 },
		{ 2686, 4676, 0 },
		{ 2143, 3845, 0 },
		{ 2143, 3846, 0 },
		{ 2197, 4598, 0 },
		{ 3254, 4173, 0 },
		{ 1089, 4443, 0 },
		{ 2197, 3449, 0 },
		{ 3191, 4416, 0 },
		{ 2212, 4103, 0 },
		{ 0, 0, 79 },
		{ 2197, 4605, 0 },
		{ 0, 0, 87 },
		{ 3150, 4995, 0 },
		{ 2197, 4607, 0 },
		{ 3150, 5005, 0 },
		{ 3275, 3834, 0 },
		{ 2197, 4509, 0 },
		{ 3271, 3725, 0 },
		{ 3281, 4773, 0 },
		{ 2197, 4513, 0 },
		{ 2212, 4112, 0 },
		{ 2127, 4317, 0 },
		{ 3275, 3837, 0 },
		{ 2127, 4320, 0 },
		{ 3191, 4431, 0 },
		{ 3254, 4192, 0 },
		{ 2197, 4523, 0 },
		{ 2212, 4115, 0 },
		{ 2197, 4525, 0 },
		{ 3191, 4438, 0 },
		{ 2197, 4527, 0 },
		{ 0, 0, 74 },
		{ 2212, 4116, 0 },
		{ 2212, 4118, 0 },
		{ 3150, 5091, 0 },
		{ 2146, 4243, 0 },
		{ 3275, 3838, 0 },
		{ 3254, 4198, 0 },
		{ 2197, 4539, 0 },
		{ 2212, 4120, 0 },
		{ 2197, 4541, 0 },
		{ 2165, 3567, 0 },
		{ 3191, 4361, 0 },
		{ 3266, 4087, 0 },
		{ 3150, 4859, 0 },
		{ 2146, 4249, 0 },
		{ 2127, 4286, 0 },
		{ 3150, 4871, 0 },
		{ 2143, 3866, 0 },
		{ 3281, 4774, 0 },
		{ 2143, 3869, 0 },
		{ 2165, 3568, 0 },
		{ 2212, 4138, 0 },
		{ 2105, 3692, 0 },
		{ 3281, 4792, 0 },
		{ 2197, 4557, 0 },
		{ 2197, 4559, 0 },
		{ 904, 3657, 0 },
		{ 0, 3652, 0 },
		{ 3254, 4223, 0 },
		{ 2233, 4467, 0 },
		{ 2197, 4566, 0 },
		{ 3241, 3976, 0 },
		{ 1270, 4500, 0 },
		{ 3150, 4943, 0 },
		{ 3241, 3977, 0 },
		{ 2165, 3569, 0 },
		{ 3275, 3749, 0 },
		{ 3241, 3981, 0 },
		{ 2127, 4324, 0 },
		{ 3191, 4395, 0 },
		{ 3241, 3985, 0 },
		{ 3254, 4176, 0 },
		{ 3275, 3750, 0 },
		{ 2686, 4725, 0 },
		{ 2686, 4727, 0 },
		{ 3150, 5001, 0 },
		{ 2197, 4584, 0 },
		{ 0, 0, 73 },
		{ 2127, 4268, 0 },
		{ 2686, 4733, 0 },
		{ 2165, 3570, 0 },
		{ 3275, 3753, 0 },
		{ 3254, 4185, 0 },
		{ 3266, 4060, 0 },
		{ 3241, 3989, 0 },
		{ 3241, 3991, 0 },
		{ 3241, 3996, 0 },
		{ 3275, 3755, 0 },
		{ 2212, 4105, 0 },
		{ 2212, 4107, 0 },
		{ 0, 0, 145 },
		{ 0, 0, 146 },
		{ 2146, 4253, 0 },
		{ 1270, 4476, 0 },
		{ 3275, 3756, 0 },
		{ 2127, 4297, 0 },
		{ 2127, 4299, 0 },
		{ 3085, 4802, 0 },
		{ 0, 0, 10 },
		{ 3150, 5089, 0 },
		{ 0, 0, 12 },
		{ 2143, 3894, 0 },
		{ 3275, 3757, 0 },
		{ 3150, 4478, 0 },
		{ 3281, 4780, 0 },
		{ 3254, 4202, 0 },
		{ 3150, 4839, 0 },
		{ 3150, 4841, 0 },
		{ 3275, 3759, 0 },
		{ 2165, 3571, 0 },
		{ 3191, 4433, 0 },
		{ 3191, 4435, 0 },
		{ 2197, 4519, 0 },
		{ 2165, 3574, 0 },
		{ 3281, 4737, 0 },
		{ 3150, 4867, 0 },
		{ 3281, 4738, 0 },
		{ 2127, 4311, 0 },
		{ 0, 0, 88 },
		{ 2212, 4117, 0 },
		{ 3191, 4349, 0 },
		{ 0, 0, 86 },
		{ 3271, 3712, 0 },
		{ 2146, 4234, 0 },
		{ 0, 0, 89 },
		{ 3281, 4770, 0 },
		{ 3191, 4354, 0 },
		{ 3271, 3714, 0 },
		{ 2197, 4531, 0 },
		{ 2143, 3903, 0 },
		{ 3241, 3924, 0 },
		{ 2197, 4535, 0 },
		{ 2165, 3437, 0 },
		{ 1270, 4483, 0 },
		{ 2165, 3446, 0 },
		{ 2165, 3456, 0 },
		{ 3275, 3767, 0 },
		{ 0, 0, 67 },
		{ 0, 0, 105 },
		{ 0, 0, 107 },
		{ 2212, 4137, 0 },
		{ 2686, 4731, 0 },
		{ 3241, 3929, 0 },
		{ 2197, 4544, 0 },
		{ 3191, 4367, 0 },
		{ 3266, 4094, 0 },
		{ 2197, 4547, 0 },
		{ 0, 0, 151 },
		{ 3191, 4370, 0 },
		{ 3241, 3931, 0 },
		{ 2197, 4552, 0 },
		{ 3191, 4372, 0 },
		{ 3275, 3768, 0 },
		{ 3241, 3934, 0 },
		{ 3150, 5003, 0 },
		{ 2165, 3458, 0 },
		{ 2127, 4280, 0 },
		{ 2127, 4283, 0 },
		{ 2197, 4560, 0 },
		{ 2686, 4678, 0 },
		{ 3275, 3771, 0 },
		{ 3275, 3772, 0 },
		{ 3241, 3941, 0 },
		{ 2233, 4458, 0 },
		{ 0, 4329, 0 },
		{ 3275, 3773, 0 },
		{ 3275, 3774, 0 },
		{ 3191, 4389, 0 },
		{ 3266, 4054, 0 },
		{ 2212, 4161, 0 },
		{ 3150, 5075, 0 },
		{ 3191, 4400, 0 },
		{ 3275, 3775, 0 },
		{ 2212, 4163, 0 },
		{ 3281, 4781, 0 },
		{ 0, 0, 20 },
		{ 2143, 3848, 0 },
		{ 2143, 3849, 0 },
		{ 0, 0, 139 },
		{ 1270, 4503, 0 },
		{ 3191, 4405, 0 },
		{ 2143, 3850, 0 },
		{ 1270, 4475, 0 },
		{ 0, 0, 144 },
		{ 3241, 3951, 0 },
		{ 2197, 4588, 0 },
		{ 0, 0, 103 },
		{ 2165, 3459, 0 },
		{ 2127, 4315, 0 },
		{ 2127, 4316, 0 },
		{ 2165, 3460, 0 },
		{ 2127, 4318, 0 },
		{ 3150, 4857, 0 },
		{ 2143, 3859, 0 },
		{ 2212, 4108, 0 },
		{ 3191, 4417, 0 },
		{ 0, 0, 84 },
		{ 2127, 4322, 0 },
		{ 1270, 4489, 0 },
		{ 2165, 3461, 0 },
		{ 2127, 4264, 0 },
		{ 3241, 3956, 0 },
		{ 2197, 4609, 0 },
		{ 3281, 4788, 0 },
		{ 3281, 4789, 0 },
		{ 3150, 4917, 0 },
		{ 2197, 4506, 0 },
		{ 3191, 4424, 0 },
		{ 3191, 4427, 0 },
		{ 2165, 3462, 0 },
		{ 2143, 3862, 0 },
		{ 3275, 3781, 0 },
		{ 3254, 4221, 0 },
		{ 3275, 3782, 0 },
		{ 2143, 3865, 0 },
		{ 3191, 4434, 0 },
		{ 3254, 4225, 0 },
		{ 3275, 3783, 0 },
		{ 2197, 4522, 0 },
		{ 0, 0, 92 },
		{ 3191, 4437, 0 },
		{ 2127, 4275, 0 },
		{ 3281, 4777, 0 },
		{ 3191, 4335, 0 },
		{ 0, 0, 109 },
		{ 2127, 4276, 0 },
		{ 3281, 4349, 0 },
		{ 2197, 4528, 0 },
		{ 0, 0, 149 },
		{ 3275, 3784, 0 },
		{ 3191, 4351, 0 },
		{ 3275, 3786, 0 },
		{ 2165, 3463, 64 },
		{ 3254, 4171, 0 },
		{ 2212, 4124, 0 },
		{ 2127, 4285, 0 },
		{ 3271, 3702, 0 },
		{ 3085, 4300, 0 },
		{ 0, 0, 93 },
		{ 2143, 3873, 0 },
		{ 3281, 4743, 0 },
		{ 1089, 4440, 0 },
		{ 0, 4441, 0 },
		{ 3275, 3791, 0 },
		{ 3254, 4181, 0 },
		{ 3254, 4182, 0 },
		{ 2212, 4129, 0 },
		{ 3281, 4772, 0 },
		{ 2197, 4548, 0 },
		{ 3191, 4368, 0 },
		{ 2165, 3465, 0 },
		{ 2212, 4134, 0 },
		{ 2212, 4135, 0 },
		{ 2212, 4136, 0 },
		{ 0, 0, 97 },
		{ 3191, 4373, 0 },
		{ 2143, 3883, 0 },
		{ 2127, 4306, 0 },
		{ 2197, 4558, 0 },
		{ 3241, 3973, 0 },
		{ 2127, 4310, 0 },
		{ 0, 0, 131 },
		{ 2165, 3466, 0 },
		{ 3271, 3706, 0 },
		{ 2165, 3467, 0 },
		{ 3266, 4047, 0 },
		{ 3275, 3797, 0 },
		{ 3191, 4386, 0 },
		{ 2686, 4647, 0 },
		{ 2143, 3891, 0 },
		{ 3254, 4203, 0 },
		{ 0, 0, 99 },
		{ 3254, 4204, 0 },
		{ 2686, 4655, 0 },
		{ 2686, 4666, 0 },
		{ 3275, 3799, 0 },
		{ 0, 0, 16 },
		{ 2127, 4323, 0 },
		{ 0, 0, 18 },
		{ 0, 0, 19 },
		{ 3191, 4399, 0 },
		{ 2165, 3468, 0 },
		{ 2233, 4466, 0 },
		{ 3254, 4209, 0 },
		{ 3150, 4888, 0 },
		{ 3241, 3982, 0 },
		{ 2212, 4156, 0 },
		{ 1270, 4485, 0 },
		{ 3241, 3983, 0 },
		{ 3241, 3984, 0 },
		{ 3254, 4215, 0 },
		{ 2212, 4160, 0 },
		{ 0, 0, 63 },
		{ 2197, 4591, 0 },
		{ 0, 0, 140 },
		{ 3191, 4409, 0 },
		{ 2197, 4594, 0 },
		{ 3275, 3801, 0 },
		{ 2165, 3471, 0 },
		{ 3275, 3803, 0 },
		{ 2127, 4274, 0 },
		{ 1155, 3633, 0 },
		{ 2212, 4165, 0 },
		{ 2197, 4601, 0 },
		{ 0, 0, 82 },
		{ 2165, 3472, 0 },
		{ 3281, 4768, 0 },
		{ 3241, 3990, 0 },
		{ 0, 3635, 0 },
		{ 3241, 3995, 0 },
		{ 0, 0, 17 },
		{ 2212, 4104, 0 },
		{ 1270, 4478, 0 },
		{ 2165, 3473, 62 },
		{ 2165, 3475, 0 },
		{ 2127, 4287, 0 },
		{ 0, 0, 83 },
		{ 3254, 4174, 0 },
		{ 3281, 3700, 0 },
		{ 0, 0, 90 },
		{ 0, 0, 91 },
		{ 0, 0, 60 },
		{ 3254, 4178, 0 },
		{ 0, 0, 141 },
		{ 3266, 4079, 0 },
		{ 0, 0, 142 },
		{ 3266, 4080, 0 },
		{ 3275, 3811, 0 },
		{ 3266, 4083, 0 },
		{ 0, 0, 150 },
		{ 0, 0, 135 },
		{ 3254, 4183, 0 },
		{ 1270, 4490, 0 },
		{ 1270, 4491, 0 },
		{ 3275, 3812, 0 },
		{ 0, 0, 40 },
		{ 2165, 3476, 41 },
		{ 2165, 3477, 43 },
		{ 3254, 4187, 0 },
		{ 3281, 4755, 0 },
		{ 1270, 4498, 0 },
		{ 1270, 4499, 0 },
		{ 2127, 4308, 0 },
		{ 0, 0, 80 },
		{ 3254, 4188, 0 },
		{ 3275, 3815, 0 },
		{ 0, 0, 98 },
		{ 3275, 3816, 0 },
		{ 2127, 4312, 0 },
		{ 3266, 4089, 0 },
		{ 2127, 4314, 0 },
		{ 2143, 3912, 0 },
		{ 3191, 4357, 0 },
		{ 3271, 3727, 0 },
		{ 3191, 4359, 0 },
		{ 2233, 4468, 0 },
		{ 2233, 4469, 0 },
		{ 2165, 3478, 0 },
		{ 3275, 3819, 0 },
		{ 3281, 4791, 0 },
		{ 3271, 3730, 0 },
		{ 0, 0, 94 },
		{ 3281, 4794, 0 },
		{ 2165, 3479, 0 },
		{ 0, 0, 143 },
		{ 0, 0, 147 },
		{ 2127, 4321, 0 },
		{ 0, 0, 152 },
		{ 0, 0, 11 },
		{ 3254, 4200, 0 },
		{ 3254, 4201, 0 },
		{ 2212, 4130, 0 },
		{ 3266, 4020, 0 },
		{ 3266, 4041, 0 },
		{ 1270, 4493, 0 },
		{ 2165, 3480, 0 },
		{ 3275, 3823, 0 },
		{ 3254, 4207, 0 },
		{ 3275, 3824, 0 },
		{ 3281, 4766, 0 },
		{ 0, 0, 148 },
		{ 3191, 4375, 0 },
		{ 3281, 4769, 0 },
		{ 3254, 4210, 0 },
		{ 3271, 3736, 0 },
		{ 3271, 3737, 0 },
		{ 3271, 3738, 0 },
		{ 3281, 4775, 0 },
		{ 2165, 3481, 0 },
		{ 3281, 4778, 0 },
		{ 3191, 4381, 0 },
		{ 3150, 4947, 0 },
		{ 3275, 3829, 0 },
		{ 3275, 3830, 0 },
		{ 2165, 3482, 0 },
		{ 0, 0, 42 },
		{ 0, 0, 44 },
		{ 3254, 4218, 0 },
		{ 3150, 4960, 0 },
		{ 3281, 4787, 0 },
		{ 3275, 3832, 0 },
		{ 2212, 4148, 0 },
		{ 2127, 4281, 0 },
		{ 3191, 4393, 0 },
		{ 3191, 4394, 0 },
		{ 3085, 4809, 0 },
		{ 3281, 4793, 0 },
		{ 2127, 4282, 0 },
		{ 3150, 5011, 0 },
		{ 3191, 4396, 0 },
		{ 3254, 4222, 0 },
		{ 2127, 4284, 0 },
		{ 2212, 4149, 0 },
		{ 2212, 4150, 0 },
		{ 2197, 4590, 0 },
		{ 3275, 3833, 0 },
		{ 2127, 4288, 0 },
		{ 2127, 4289, 0 },
		{ 2212, 4152, 0 },
		{ 0, 0, 85 },
		{ 0, 0, 100 },
		{ 3254, 4227, 0 },
		{ 3254, 4228, 0 },
		{ 0, 4497, 0 },
		{ 3191, 4408, 0 },
		{ 0, 0, 95 },
		{ 2127, 4293, 0 },
		{ 3254, 4230, 0 },
		{ 2143, 3868, 0 },
		{ 0, 0, 13 },
		{ 2212, 4154, 0 },
		{ 2212, 4155, 0 },
		{ 0, 0, 96 },
		{ 0, 0, 61 },
		{ 0, 0, 101 },
		{ 3241, 3946, 0 },
		{ 3254, 4172, 0 },
		{ 2197, 4608, 0 },
		{ 0, 0, 15 },
		{ 2165, 3484, 0 },
		{ 3241, 3949, 0 },
		{ 2197, 4507, 0 },
		{ 3266, 4069, 0 },
		{ 2127, 4307, 0 },
		{ 3150, 4843, 0 },
		{ 3281, 4785, 0 },
		{ 2197, 4510, 0 },
		{ 2146, 4252, 0 },
		{ 2197, 4512, 0 },
		{ 3254, 4177, 0 },
		{ 3275, 3835, 0 },
		{ 0, 0, 14 },
		{ 3334, 1457, 249 },
		{ 0, 0, 250 },
		{ 3280, 5483, 251 },
		{ 3317, 1928, 255 },
		{ 1307, 2968, 256 },
		{ 0, 0, 256 },
		{ 3317, 1968, 252 },
		{ 1310, 1633, 0 },
		{ 3317, 2024, 253 },
		{ 1313, 1710, 0 },
		{ 1310, 0, 0 },
		{ 3226, 2925, 254 },
		{ 1315, 1584, 0 },
		{ 1313, 0, 0 },
		{ 3226, 2958, 252 },
		{ 1315, 0, 0 },
		{ 3226, 2968, 253 },
		{ 3271, 3735, 161 },
		{ 0, 0, 161 },
		{ 0, 0, 162 },
		{ 3295, 2353, 0 },
		{ 3317, 3243, 0 },
		{ 3334, 2390, 0 },
		{ 1323, 5221, 0 },
		{ 3314, 2940, 0 },
		{ 3317, 3190, 0 },
		{ 3329, 3333, 0 },
		{ 3325, 2831, 0 },
		{ 3328, 3409, 0 },
		{ 3334, 2375, 0 },
		{ 3328, 3375, 0 },
		{ 3330, 2259, 0 },
		{ 3217, 3005, 0 },
		{ 3332, 2506, 0 },
		{ 3276, 2672, 0 },
		{ 3295, 2285, 0 },
		{ 3335, 5103, 0 },
		{ 0, 0, 159 },
		{ 3271, 3724, 165 },
		{ 0, 0, 165 },
		{ 0, 0, 166 },
		{ 3295, 2341, 0 },
		{ 3317, 3247, 0 },
		{ 3334, 2407, 0 },
		{ 1344, 5292, 0 },
		{ 3281, 4404, 0 },
		{ 3271, 3728, 0 },
		{ 2686, 4674, 0 },
		{ 3150, 5015, 0 },
		{ 3335, 5095, 0 },
		{ 0, 0, 163 },
		{ 3085, 4811, 173 },
		{ 0, 0, 173 },
		{ 0, 0, 174 },
		{ 3317, 3196, 0 },
		{ 3163, 3118, 0 },
		{ 3332, 2509, 0 },
		{ 3334, 2432, 0 },
		{ 3317, 3184, 0 },
		{ 1359, 5345, 0 },
		{ 3317, 2842, 0 },
		{ 3299, 1713, 0 },
		{ 3317, 3198, 0 },
		{ 3334, 2387, 0 },
		{ 2909, 1620, 0 },
		{ 3330, 2144, 0 },
		{ 3324, 3080, 0 },
		{ 3217, 3037, 0 },
		{ 3276, 2629, 0 },
		{ 3119, 3100, 0 },
		{ 1370, 5355, 0 },
		{ 3317, 2850, 0 },
		{ 3325, 2813, 0 },
		{ 3295, 2352, 0 },
		{ 3317, 3283, 0 },
		{ 1375, 5295, 0 },
		{ 3327, 2783, 0 },
		{ 3322, 1886, 0 },
		{ 3276, 2619, 0 },
		{ 3329, 3322, 0 },
		{ 3330, 2000, 0 },
		{ 3217, 3074, 0 },
		{ 3332, 2483, 0 },
		{ 3276, 2574, 0 },
		{ 3335, 4947, 0 },
		{ 0, 0, 171 },
		{ 3271, 3733, 197 },
		{ 0, 0, 197 },
		{ 3295, 2290, 0 },
		{ 3317, 3158, 0 },
		{ 3334, 2398, 0 },
		{ 1391, 5325, 0 },
		{ 3329, 3042, 0 },
		{ 3325, 2796, 0 },
		{ 3328, 3387, 0 },
		{ 3295, 2296, 0 },
		{ 3295, 2311, 0 },
		{ 3317, 3242, 0 },
		{ 3295, 2316, 0 },
		{ 3335, 4961, 0 },
		{ 0, 0, 196 },
		{ 2233, 4459, 157 },
		{ 0, 0, 157 },
		{ 0, 0, 158 },
		{ 3317, 3244, 0 },
		{ 3276, 2575, 0 },
		{ 3332, 2522, 0 },
		{ 3319, 2699, 0 },
		{ 3317, 3175, 0 },
		{ 3281, 4767, 0 },
		{ 3325, 2801, 0 },
		{ 3328, 3371, 0 },
		{ 3295, 2323, 0 },
		{ 3295, 2333, 0 },
		{ 3297, 5192, 0 },
		{ 3297, 5188, 0 },
		{ 3217, 2963, 0 },
		{ 3276, 2638, 0 },
		{ 3217, 3023, 0 },
		{ 3330, 2017, 0 },
		{ 3217, 3048, 0 },
		{ 3328, 3369, 0 },
		{ 3325, 2811, 0 },
		{ 3217, 2864, 0 },
		{ 3295, 1820, 0 },
		{ 3317, 3168, 0 },
		{ 3334, 2414, 0 },
		{ 3335, 4953, 0 },
		{ 0, 0, 155 },
		{ 2791, 3344, 24 },
		{ 0, 0, 24 },
		{ 0, 0, 25 },
		{ 3317, 3181, 0 },
		{ 3119, 3102, 0 },
		{ 3217, 3016, 0 },
		{ 3276, 2609, 0 },
		{ 3280, 5462, 0 },
		{ 3332, 2503, 0 },
		{ 3077, 2467, 0 },
		{ 3317, 3225, 0 },
		{ 3334, 2429, 0 },
		{ 3328, 3411, 0 },
		{ 3330, 2204, 0 },
		{ 3332, 2478, 0 },
		{ 3334, 2371, 0 },
		{ 3280, 5461, 0 },
		{ 3314, 3291, 0 },
		{ 3317, 3251, 0 },
		{ 3295, 2354, 0 },
		{ 3329, 3328, 0 },
		{ 3334, 2381, 0 },
		{ 3217, 3030, 0 },
		{ 3077, 2464, 0 },
		{ 3330, 2248, 0 },
		{ 3217, 3062, 0 },
		{ 3332, 2548, 0 },
		{ 3276, 2636, 0 },
		{ 3280, 2, 0 },
		{ 3297, 5195, 0 },
		{ 0, 0, 21 },
		{ 1474, 0, 1 },
		{ 1474, 0, 198 },
		{ 1474, 3074, 248 },
		{ 1689, 218, 248 },
		{ 1689, 400, 248 },
		{ 1689, 389, 248 },
		{ 1689, 520, 248 },
		{ 1689, 391, 248 },
		{ 1689, 408, 248 },
		{ 1689, 396, 248 },
		{ 1689, 478, 248 },
		{ 1689, 495, 248 },
		{ 1474, 0, 248 },
		{ 1486, 2860, 248 },
		{ 1474, 3154, 248 },
		{ 2791, 3338, 244 },
		{ 1689, 513, 248 },
		{ 1689, 537, 248 },
		{ 1689, 590, 248 },
		{ 1689, 0, 248 },
		{ 1689, 625, 248 },
		{ 1689, 612, 248 },
		{ 3332, 2484, 0 },
		{ 0, 0, 199 },
		{ 3276, 2632, 0 },
		{ 1689, 582, 0 },
		{ 1689, 0, 0 },
		{ 3280, 4393, 0 },
		{ 1689, 602, 0 },
		{ 1689, 619, 0 },
		{ 1689, 617, 0 },
		{ 1689, 625, 0 },
		{ 1689, 617, 0 },
		{ 1689, 621, 0 },
		{ 1689, 639, 0 },
		{ 1689, 621, 0 },
		{ 1689, 614, 0 },
		{ 1689, 606, 0 },
		{ 1689, 637, 0 },
		{ 3317, 3217, 0 },
		{ 3317, 3220, 0 },
		{ 1690, 644, 0 },
		{ 1690, 645, 0 },
		{ 1689, 656, 0 },
		{ 1689, 693, 0 },
		{ 1689, 684, 0 },
		{ 3332, 2507, 0 },
		{ 3314, 3285, 0 },
		{ 1689, 682, 0 },
		{ 1689, 726, 0 },
		{ 1689, 703, 0 },
		{ 1689, 704, 0 },
		{ 1689, 727, 0 },
		{ 1689, 754, 0 },
		{ 1689, 759, 0 },
		{ 1689, 753, 0 },
		{ 1689, 729, 0 },
		{ 1689, 721, 0 },
		{ 1689, 757, 0 },
		{ 1689, 772, 0 },
		{ 1689, 759, 0 },
		{ 3276, 2655, 0 },
		{ 3163, 3146, 0 },
		{ 1689, 775, 0 },
		{ 1689, 765, 0 },
		{ 1690, 767, 0 },
		{ 1689, 763, 0 },
		{ 1689, 793, 0 },
		{ 3325, 2822, 0 },
		{ 0, 0, 247 },
		{ 1689, 805, 0 },
		{ 1689, 838, 0 },
		{ 1689, 825, 0 },
		{ 1689, 830, 0 },
		{ 1689, 20, 0 },
		{ 1689, 73, 0 },
		{ 1689, 69, 0 },
		{ 1689, 58, 0 },
		{ 1689, 36, 0 },
		{ 1689, 40, 0 },
		{ 1689, 82, 0 },
		{ 1689, 0, 233 },
		{ 1689, 118, 0 },
		{ 3332, 2482, 0 },
		{ 3217, 3042, 0 },
		{ 1689, 76, 0 },
		{ 1689, 108, 0 },
		{ 1689, 110, 0 },
		{ 1689, 114, 0 },
		{ 1689, 114, 0 },
		{ -1569, 1535, 0 },
		{ 1690, 122, 0 },
		{ 1689, 155, 0 },
		{ 1689, 161, 0 },
		{ 1689, 153, 0 },
		{ 1689, 163, 0 },
		{ 1689, 167, 0 },
		{ 1689, 146, 0 },
		{ 1689, 163, 0 },
		{ 1689, 140, 0 },
		{ 1689, 131, 0 },
		{ 1689, 0, 232 },
		{ 1689, 138, 0 },
		{ 3319, 2719, 0 },
		{ 3276, 2610, 0 },
		{ 1689, 157, 0 },
		{ 1689, 168, 0 },
		{ 1689, 191, 0 },
		{ 1689, 0, 246 },
		{ 1689, 191, 0 },
		{ 0, 0, 234 },
		{ 1689, 185, 0 },
		{ 1691, 4, -4 },
		{ 1689, 246, 0 },
		{ 1689, 258, 0 },
		{ 1689, 255, 0 },
		{ 1689, 260, 0 },
		{ 1689, 264, 0 },
		{ 1689, 277, 0 },
		{ 1689, 248, 0 },
		{ 1689, 253, 0 },
		{ 1689, 244, 0 },
		{ 3317, 3259, 0 },
		{ 3317, 3268, 0 },
		{ 1689, 0, 236 },
		{ 1689, 285, 237 },
		{ 1689, 258, 0 },
		{ 1689, 266, 0 },
		{ 1689, 307, 0 },
		{ 1589, 3958, 0 },
		{ 3280, 4690, 0 },
		{ 2274, 5157, 223 },
		{ 1689, 310, 0 },
		{ 1689, 314, 0 },
		{ 1689, 344, 0 },
		{ 1689, 346, 0 },
		{ 1689, 381, 0 },
		{ 1689, 382, 0 },
		{ 1689, 370, 0 },
		{ 1689, 375, 0 },
		{ 1689, 357, 0 },
		{ 1689, 364, 0 },
		{ 1690, 350, 0 },
		{ 3281, 4759, 0 },
		{ 3280, 5481, 239 },
		{ 1689, 353, 0 },
		{ 1689, 365, 0 },
		{ 1689, 347, 0 },
		{ 1689, 363, 0 },
		{ 0, 0, 203 },
		{ 1691, 31, -7 },
		{ 1691, 206, -10 },
		{ 1691, 321, -13 },
		{ 1691, 349, -16 },
		{ 1691, 351, -19 },
		{ 1691, 435, -22 },
		{ 1689, 413, 0 },
		{ 1689, 426, 0 },
		{ 1689, 399, 0 },
		{ 1689, 0, 221 },
		{ 1689, 0, 235 },
		{ 3325, 2805, 0 },
		{ 1689, 430, 0 },
		{ 1689, 421, 0 },
		{ 1689, 454, 0 },
		{ 1690, 463, 0 },
		{ 1626, 3926, 0 },
		{ 3280, 4722, 0 },
		{ 2274, 5129, 224 },
		{ 1629, 3927, 0 },
		{ 3280, 4686, 0 },
		{ 2274, 5154, 225 },
		{ 1632, 3928, 0 },
		{ 3280, 4716, 0 },
		{ 2274, 5135, 228 },
		{ 1635, 3929, 0 },
		{ 3280, 4752, 0 },
		{ 2274, 5127, 229 },
		{ 1638, 3930, 0 },
		{ 3280, 4684, 0 },
		{ 2274, 5139, 230 },
		{ 1641, 3931, 0 },
		{ 3280, 4688, 0 },
		{ 2274, 5163, 231 },
		{ 1689, 508, 0 },
		{ 1691, 463, -25 },
		{ 1689, 481, 0 },
		{ 3328, 3356, 0 },
		{ 1689, 462, 0 },
		{ 1689, 507, 0 },
		{ 1689, 472, 0 },
		{ 1689, 486, 0 },
		{ 0, 0, 205 },
		{ 0, 0, 207 },
		{ 0, 0, 213 },
		{ 0, 0, 215 },
		{ 0, 0, 217 },
		{ 0, 0, 219 },
		{ 1691, 465, -28 },
		{ 1659, 3941, 0 },
		{ 3280, 4720, 0 },
		{ 2274, 5150, 227 },
		{ 1689, 0, 220 },
		{ 3295, 2288, 0 },
		{ 1689, 474, 0 },
		{ 1689, 490, 0 },
		{ 1690, 497, 0 },
		{ 1689, 494, 0 },
		{ 1668, 3948, 0 },
		{ 3280, 4692, 0 },
		{ 2274, 5153, 226 },
		{ 0, 0, 211 },
		{ 3295, 2318, 0 },
		{ 1689, 4, 242 },
		{ 1690, 501, 0 },
		{ 1689, 4, 245 },
		{ 1689, 544, 0 },
		{ 0, 0, 209 },
		{ 3297, 5193, 0 },
		{ 3297, 5194, 0 },
		{ 1689, 532, 0 },
		{ 0, 0, 243 },
		{ 1689, 557, 0 },
		{ 3297, 5189, 0 },
		{ 0, 0, 241 },
		{ 1689, 576, 0 },
		{ 1689, 581, 0 },
		{ 0, 0, 240 },
		{ 1689, 586, 0 },
		{ 1689, 577, 0 },
		{ 1690, 579, 238 },
		{ 1691, 1003, 0 },
		{ 1692, 736, -1 },
		{ 1693, 3972, 0 },
		{ 3280, 4676, 0 },
		{ 2274, 5145, 222 },
		{ 0, 0, 201 },
		{ 2233, 4470, 291 },
		{ 0, 0, 291 },
		{ 3317, 3176, 0 },
		{ 3276, 2661, 0 },
		{ 3332, 2541, 0 },
		{ 3319, 2736, 0 },
		{ 3317, 3193, 0 },
		{ 3281, 4749, 0 },
		{ 3325, 2809, 0 },
		{ 3328, 3382, 0 },
		{ 3295, 2319, 0 },
		{ 3295, 2322, 0 },
		{ 3297, 5174, 0 },
		{ 3297, 5178, 0 },
		{ 3217, 3032, 0 },
		{ 3276, 2603, 0 },
		{ 3217, 3038, 0 },
		{ 3330, 2156, 0 },
		{ 3217, 3046, 0 },
		{ 3328, 3376, 0 },
		{ 3325, 2788, 0 },
		{ 3217, 3054, 0 },
		{ 3295, 1814, 0 },
		{ 3317, 3254, 0 },
		{ 3334, 2369, 0 },
		{ 3335, 5021, 0 },
		{ 0, 0, 290 },
		{ 2233, 4456, 293 },
		{ 0, 0, 293 },
		{ 0, 0, 294 },
		{ 3317, 3261, 0 },
		{ 3276, 2616, 0 },
		{ 3332, 2488, 0 },
		{ 3319, 2729, 0 },
		{ 3317, 3163, 0 },
		{ 3281, 4786, 0 },
		{ 3325, 2812, 0 },
		{ 3328, 3404, 0 },
		{ 3295, 2334, 0 },
		{ 3295, 2335, 0 },
		{ 3297, 5179, 0 },
		{ 3297, 5184, 0 },
		{ 3329, 3310, 0 },
		{ 3334, 2380, 0 },
		{ 3332, 2508, 0 },
		{ 3295, 2336, 0 },
		{ 3295, 2339, 0 },
		{ 3332, 2531, 0 },
		{ 3299, 1714, 0 },
		{ 3317, 3203, 0 },
		{ 3334, 2395, 0 },
		{ 3335, 4949, 0 },
		{ 0, 0, 292 },
		{ 2233, 4460, 296 },
		{ 0, 0, 296 },
		{ 0, 0, 297 },
		{ 3317, 3207, 0 },
		{ 3276, 2581, 0 },
		{ 3332, 2560, 0 },
		{ 3319, 2727, 0 },
		{ 3317, 3234, 0 },
		{ 3281, 4762, 0 },
		{ 3325, 2828, 0 },
		{ 3328, 3381, 0 },
		{ 3295, 2349, 0 },
		{ 3295, 2351, 0 },
		{ 3297, 5190, 0 },
		{ 3297, 5191, 0 },
		{ 3319, 2737, 0 },
		{ 3322, 1883, 0 },
		{ 3330, 2260, 0 },
		{ 3328, 3353, 0 },
		{ 3330, 2266, 0 },
		{ 3332, 2495, 0 },
		{ 3334, 2368, 0 },
		{ 3335, 5136, 0 },
		{ 0, 0, 295 },
		{ 2233, 4462, 299 },
		{ 0, 0, 299 },
		{ 0, 0, 300 },
		{ 3317, 3280, 0 },
		{ 3276, 2634, 0 },
		{ 3332, 2505, 0 },
		{ 3319, 2703, 0 },
		{ 3317, 3167, 0 },
		{ 3281, 4784, 0 },
		{ 3325, 2830, 0 },
		{ 3328, 3408, 0 },
		{ 3295, 2357, 0 },
		{ 3295, 2362, 0 },
		{ 3297, 5186, 0 },
		{ 3297, 5187, 0 },
		{ 3317, 3180, 0 },
		{ 3299, 1709, 0 },
		{ 3328, 3364, 0 },
		{ 3325, 2800, 0 },
		{ 3322, 1855, 0 },
		{ 3328, 3373, 0 },
		{ 3330, 2128, 0 },
		{ 3332, 2529, 0 },
		{ 3334, 2383, 0 },
		{ 3335, 5091, 0 },
		{ 0, 0, 298 },
		{ 2233, 4464, 302 },
		{ 0, 0, 302 },
		{ 0, 0, 303 },
		{ 3317, 3202, 0 },
		{ 3276, 2584, 0 },
		{ 3332, 2538, 0 },
		{ 3319, 2728, 0 },
		{ 3317, 3208, 0 },
		{ 3281, 4757, 0 },
		{ 3325, 2825, 0 },
		{ 3328, 3352, 0 },
		{ 3295, 2294, 0 },
		{ 3295, 2295, 0 },
		{ 3297, 5196, 0 },
		{ 3297, 5172, 0 },
		{ 3332, 2552, 0 },
		{ 3077, 2457, 0 },
		{ 3330, 2132, 0 },
		{ 3217, 3058, 0 },
		{ 3319, 2705, 0 },
		{ 3217, 3072, 0 },
		{ 3295, 2298, 0 },
		{ 3317, 3258, 0 },
		{ 3334, 2404, 0 },
		{ 3335, 4881, 0 },
		{ 0, 0, 301 },
		{ 3150, 4855, 177 },
		{ 0, 0, 177 },
		{ 0, 0, 178 },
		{ 3163, 3121, 0 },
		{ 3330, 2139, 0 },
		{ 3317, 3278, 0 },
		{ 3334, 2413, 0 },
		{ 1833, 5320, 0 },
		{ 3317, 2866, 0 },
		{ 3299, 1711, 0 },
		{ 3317, 3159, 0 },
		{ 3334, 2426, 0 },
		{ 2909, 1635, 0 },
		{ 3330, 2162, 0 },
		{ 3324, 3086, 0 },
		{ 3217, 3033, 0 },
		{ 3276, 2573, 0 },
		{ 3119, 3103, 0 },
		{ 1844, 5344, 0 },
		{ 3317, 2856, 0 },
		{ 3325, 2763, 0 },
		{ 3295, 2321, 0 },
		{ 3317, 3194, 0 },
		{ 1849, 5380, 0 },
		{ 3327, 2781, 0 },
		{ 3322, 1849, 0 },
		{ 3276, 2577, 0 },
		{ 3329, 3311, 0 },
		{ 3330, 2206, 0 },
		{ 3217, 3065, 0 },
		{ 3332, 2519, 0 },
		{ 3276, 2598, 0 },
		{ 3335, 5101, 0 },
		{ 0, 0, 175 },
		{ 2233, 4450, 284 },
		{ 0, 0, 284 },
		{ 3317, 3212, 0 },
		{ 3276, 2599, 0 },
		{ 3332, 2520, 0 },
		{ 3319, 2709, 0 },
		{ 3317, 3231, 0 },
		{ 3281, 4764, 0 },
		{ 3325, 2824, 0 },
		{ 3328, 3394, 0 },
		{ 3295, 2324, 0 },
		{ 3295, 2325, 0 },
		{ 3297, 5182, 0 },
		{ 3297, 5183, 0 },
		{ 3314, 3294, 0 },
		{ 3217, 3028, 0 },
		{ 3295, 2332, 0 },
		{ 3077, 2458, 0 },
		{ 3325, 2748, 0 },
		{ 3328, 3361, 0 },
		{ 2909, 1664, 0 },
		{ 3335, 5037, 0 },
		{ 0, 0, 282 },
		{ 1897, 0, 1 },
		{ 2056, 3195, 399 },
		{ 3317, 3262, 399 },
		{ 3328, 3315, 399 },
		{ 3314, 2488, 399 },
		{ 1897, 0, 366 },
		{ 1897, 3115, 399 },
		{ 3324, 1847, 399 },
		{ 3085, 4810, 399 },
		{ 2212, 4131, 399 },
		{ 3271, 3711, 399 },
		{ 2212, 4133, 399 },
		{ 2197, 4550, 399 },
		{ 3334, 2283, 399 },
		{ 1897, 0, 399 },
		{ 2791, 3347, 397 },
		{ 3328, 3117, 399 },
		{ 3328, 3417, 399 },
		{ 0, 0, 399 },
		{ 3332, 2481, 0 },
		{ -1902, 7, 356 },
		{ -1903, 5270, 0 },
		{ 3276, 2645, 0 },
		{ 0, 0, 362 },
		{ 0, 0, 363 },
		{ 3325, 2810, 0 },
		{ 3217, 3063, 0 },
		{ 3317, 3174, 0 },
		{ 0, 0, 367 },
		{ 3276, 2649, 0 },
		{ 3334, 2396, 0 },
		{ 3217, 3073, 0 },
		{ 2165, 3575, 0 },
		{ 3266, 4057, 0 },
		{ 3275, 3798, 0 },
		{ 2105, 3696, 0 },
		{ 3266, 4061, 0 },
		{ 3322, 1796, 0 },
		{ 3295, 2340, 0 },
		{ 3276, 2674, 0 },
		{ 3330, 1918, 0 },
		{ 3334, 2408, 0 },
		{ 3332, 2496, 0 },
		{ 3224, 5247, 0 },
		{ 3332, 2497, 0 },
		{ 3295, 2344, 0 },
		{ 3330, 1980, 0 },
		{ 3276, 2597, 0 },
		{ 3314, 3306, 0 },
		{ 3334, 2417, 0 },
		{ 3325, 2798, 0 },
		{ 2233, 4449, 0 },
		{ 2165, 3469, 0 },
		{ 2165, 3470, 0 },
		{ 2197, 4592, 0 },
		{ 2127, 4302, 0 },
		{ 3317, 3211, 0 },
		{ 3295, 2350, 0 },
		{ 3314, 3296, 0 },
		{ 3322, 1845, 0 },
		{ 3317, 3219, 0 },
		{ 3325, 2804, 0 },
		{ 0, 5490, 359 },
		{ 3319, 2738, 0 },
		{ 3317, 3226, 0 },
		{ 2212, 4114, 0 },
		{ 3330, 2014, 0 },
		{ 0, 0, 398 },
		{ 3317, 3233, 0 },
		{ 3314, 3307, 0 },
		{ 2197, 4606, 0 },
		{ 2143, 3842, 0 },
		{ 3266, 4097, 0 },
		{ 3241, 3968, 0 },
		{ 2165, 3483, 0 },
		{ 0, 0, 387 },
		{ 3281, 4741, 0 },
		{ 3332, 2518, 0 },
		{ 3334, 2436, 0 },
		{ 3276, 2613, 0 },
		{ -1979, 1170, 0 },
		{ 0, 0, 358 },
		{ 3317, 3246, 0 },
		{ 0, 0, 386 },
		{ 3077, 2465, 0 },
		{ 3217, 3071, 0 },
		{ 3276, 2622, 0 },
		{ 1994, 5210, 0 },
		{ 3254, 4196, 0 },
		{ 3191, 4384, 0 },
		{ 3241, 3978, 0 },
		{ 2165, 3492, 0 },
		{ 3266, 4056, 0 },
		{ 3332, 2527, 0 },
		{ 3319, 2730, 0 },
		{ 3276, 2631, 0 },
		{ 3330, 2040, 0 },
		{ 0, 0, 388 },
		{ 3281, 4771, 365 },
		{ 3330, 2098, 0 },
		{ 3329, 3321, 0 },
		{ 3330, 2100, 0 },
		{ 0, 0, 391 },
		{ 0, 0, 392 },
		{ 1999, 0, -41 },
		{ 2178, 3626, 0 },
		{ 2212, 4145, 0 },
		{ 3266, 4068, 0 },
		{ 2197, 4532, 0 },
		{ 3217, 3008, 0 },
		{ 0, 0, 390 },
		{ 0, 0, 396 },
		{ 0, 5212, 0 },
		{ 3325, 2786, 0 },
		{ 3295, 2280, 0 },
		{ 3328, 3384, 0 },
		{ 2233, 4454, 0 },
		{ 3280, 4766, 0 },
		{ 2274, 5143, 381 },
		{ 2197, 4538, 0 },
		{ 3085, 4812, 0 },
		{ 3241, 3993, 0 },
		{ 3241, 3994, 0 },
		{ 3276, 2639, 0 },
		{ 0, 0, 393 },
		{ 0, 0, 394 },
		{ 3328, 3393, 0 },
		{ 2794, 5261, 0 },
		{ 3325, 2797, 0 },
		{ 3317, 3162, 0 },
		{ 0, 0, 371 },
		{ 2022, 0, -44 },
		{ 2024, 0, -47 },
		{ 2212, 4157, 0 },
		{ 3281, 4797, 0 },
		{ 0, 0, 389 },
		{ 3295, 2283, 0 },
		{ 0, 0, 364 },
		{ 2233, 4465, 0 },
		{ 3276, 2646, 0 },
		{ 3280, 4748, 0 },
		{ 2274, 5161, 382 },
		{ 3280, 4750, 0 },
		{ 2274, 5108, 383 },
		{ 3085, 4807, 0 },
		{ 2034, 0, -77 },
		{ 3295, 2284, 0 },
		{ 3317, 3170, 0 },
		{ 3317, 3172, 0 },
		{ 0, 0, 373 },
		{ 0, 0, 375 },
		{ 2039, 0, -35 },
		{ 3280, 4672, 0 },
		{ 2274, 5152, 385 },
		{ 0, 0, 361 },
		{ 3276, 2651, 0 },
		{ 3334, 2385, 0 },
		{ 3280, 4678, 0 },
		{ 2274, 5159, 384 },
		{ 0, 0, 379 },
		{ 3332, 2476, 0 },
		{ 3328, 3363, 0 },
		{ 0, 0, 377 },
		{ 3319, 2735, 0 },
		{ 3330, 2118, 0 },
		{ 3317, 3182, 0 },
		{ 3217, 3052, 0 },
		{ 0, 0, 395 },
		{ 3332, 2479, 0 },
		{ 3276, 2572, 0 },
		{ 2053, 0, -50 },
		{ 3280, 4718, 0 },
		{ 2274, 5151, 380 },
		{ 0, 0, 369 },
		{ 1897, 3184, 399 },
		{ 2060, 2856, 399 },
		{ -2058, 21, 356 },
		{ -2059, 5267, 0 },
		{ 3280, 5255, 0 },
		{ 3224, 5237, 0 },
		{ 0, 0, 357 },
		{ 3224, 5250, 0 },
		{ -2064, 19, 0 },
		{ -2065, 5274, 0 },
		{ 2068, 2, 359 },
		{ 3224, 5251, 0 },
		{ 3280, 5331, 0 },
		{ 0, 0, 360 },
		{ 2086, 0, 1 },
		{ 2282, 3177, 355 },
		{ 3317, 3204, 355 },
		{ 2086, 0, 309 },
		{ 2086, 3137, 355 },
		{ 3266, 4053, 355 },
		{ 2086, 0, 312 },
		{ 3322, 1884, 355 },
		{ 3085, 4805, 355 },
		{ 2212, 4121, 355 },
		{ 3271, 3656, 355 },
		{ 2212, 4123, 355 },
		{ 2197, 4583, 355 },
		{ 3328, 3418, 355 },
		{ 3334, 2282, 355 },
		{ 2086, 0, 355 },
		{ 2791, 3341, 352 },
		{ 3328, 3358, 355 },
		{ 3314, 3289, 355 },
		{ 3085, 4803, 355 },
		{ 3328, 1828, 355 },
		{ 0, 0, 355 },
		{ 3332, 2490, 0 },
		{ -2093, 17, 304 },
		{ -2094, 5275, 0 },
		{ 3276, 2586, 0 },
		{ 0, 0, 310 },
		{ 3276, 2595, 0 },
		{ 3332, 2491, 0 },
		{ 3334, 2405, 0 },
		{ 2165, 3559, 0 },
		{ 3266, 4073, 0 },
		{ 3275, 3806, 0 },
		{ 3254, 4208, 0 },
		{ 0, 3642, 0 },
		{ 0, 3690, 0 },
		{ 3266, 4078, 0 },
		{ 3325, 2794, 0 },
		{ 3322, 1744, 0 },
		{ 3295, 2300, 0 },
		{ 3276, 2607, 0 },
		{ 3317, 3235, 0 },
		{ 3317, 3238, 0 },
		{ 3332, 2498, 0 },
		{ 2794, 5260, 0 },
		{ 3332, 2499, 0 },
		{ 3224, 5228, 0 },
		{ 3332, 2500, 0 },
		{ 3314, 3287, 0 },
		{ 3077, 2453, 0 },
		{ 3334, 2409, 0 },
		{ 2233, 4455, 0 },
		{ 2165, 3572, 0 },
		{ 2165, 3573, 0 },
		{ 3191, 4390, 0 },
		{ 3191, 4392, 0 },
		{ 2197, 4516, 0 },
		{ 0, 4298, 0 },
		{ 3295, 2304, 0 },
		{ 3317, 3252, 0 },
		{ 3295, 2305, 0 },
		{ 3314, 3302, 0 },
		{ 3276, 2623, 0 },
		{ 3295, 2309, 0 },
		{ 3325, 2818, 0 },
		{ 0, 0, 354 },
		{ 3325, 2820, 0 },
		{ 0, 0, 306 },
		{ 3319, 2732, 0 },
		{ 0, 0, 351 },
		{ 3322, 1746, 0 },
		{ 3317, 3272, 0 },
		{ 2197, 4529, 0 },
		{ 0, 3885, 0 },
		{ 3266, 4051, 0 },
		{ 2146, 4235, 0 },
		{ 0, 4236, 0 },
		{ 3241, 3992, 0 },
		{ 2165, 3464, 0 },
		{ 3317, 3275, 0 },
		{ 0, 0, 344 },
		{ 3281, 4739, 0 },
		{ 3332, 2512, 0 },
		{ 3330, 2166, 0 },
		{ 3330, 2184, 0 },
		{ 3322, 1748, 0 },
		{ -2173, 1245, 0 },
		{ 3317, 3160, 0 },
		{ 3325, 2750, 0 },
		{ 3276, 2641, 0 },
		{ 3254, 4190, 0 },
		{ 3191, 4423, 0 },
		{ 3241, 4003, 0 },
		{ 3191, 4425, 0 },
		{ 3191, 4426, 0 },
		{ 0, 3474, 0 },
		{ 3266, 4067, 0 },
		{ 0, 0, 343 },
		{ 3332, 2523, 0 },
		{ 3319, 2713, 0 },
		{ 3217, 2968, 0 },
		{ 0, 0, 350 },
		{ 3328, 3412, 0 },
		{ 0, 0, 345 },
		{ 0, 0, 308 },
		{ 3328, 3415, 0 },
		{ 3330, 2227, 0 },
		{ 2190, 0, -74 },
		{ 0, 3624, 0 },
		{ 2212, 4128, 0 },
		{ 2181, 3602, 0 },
		{ 2178, 3603, 0 },
		{ 3266, 4077, 0 },
		{ 2197, 4563, 0 },
		{ 3217, 3004, 0 },
		{ 0, 0, 347 },
		{ 3329, 3335, 0 },
		{ 3330, 2232, 0 },
		{ 3330, 2234, 0 },
		{ 2233, 4461, 0 },
		{ 3280, 4680, 0 },
		{ 2274, 5149, 334 },
		{ 2197, 4569, 0 },
		{ 3085, 4814, 0 },
		{ 2197, 4570, 0 },
		{ 2197, 4571, 0 },
		{ 2197, 4572, 0 },
		{ 0, 4573, 0 },
		{ 3241, 3936, 0 },
		{ 3241, 3938, 0 },
		{ 3276, 2654, 0 },
		{ 3328, 3359, 0 },
		{ 3217, 3018, 0 },
		{ 3217, 3021, 0 },
		{ 3317, 3183, 0 },
		{ 0, 0, 316 },
		{ 2219, 0, -53 },
		{ 2221, 0, -56 },
		{ 2223, 0, -62 },
		{ 2225, 0, -65 },
		{ 2227, 0, -68 },
		{ 2229, 0, -71 },
		{ 0, 4143, 0 },
		{ 3281, 4753, 0 },
		{ 0, 0, 346 },
		{ 3325, 2799, 0 },
		{ 3332, 2535, 0 },
		{ 3332, 2537, 0 },
		{ 3276, 2668, 0 },
		{ 3280, 4754, 0 },
		{ 2274, 5165, 335 },
		{ 3280, 4756, 0 },
		{ 2274, 5126, 336 },
		{ 3280, 4758, 0 },
		{ 2274, 5128, 339 },
		{ 3280, 4760, 0 },
		{ 2274, 5130, 340 },
		{ 3280, 4762, 0 },
		{ 2274, 5137, 341 },
		{ 3280, 4764, 0 },
		{ 2274, 5141, 342 },
		{ 3085, 4800, 0 },
		{ 2244, 0, -80 },
		{ 0, 4448, 0 },
		{ 3276, 2669, 0 },
		{ 3276, 2671, 0 },
		{ 3317, 3200, 0 },
		{ 0, 0, 318 },
		{ 0, 0, 320 },
		{ 0, 0, 326 },
		{ 0, 0, 328 },
		{ 0, 0, 330 },
		{ 0, 0, 332 },
		{ 2250, 0, -38 },
		{ 3280, 4674, 0 },
		{ 2274, 5156, 338 },
		{ 3317, 3201, 0 },
		{ 3328, 3386, 0 },
		{ 3279, 3605, 349 },
		{ 3334, 2378, 0 },
		{ 3280, 4682, 0 },
		{ 2274, 5167, 337 },
		{ 0, 0, 324 },
		{ 3276, 2673, 0 },
		{ 3334, 2379, 0 },
		{ 0, 0, 311 },
		{ 3328, 3401, 0 },
		{ 0, 0, 322 },
		{ 3332, 2542, 0 },
		{ 2909, 1636, 0 },
		{ 3330, 2251, 0 },
		{ 3319, 2715, 0 },
		{ 3150, 4997, 0 },
		{ 3217, 3057, 0 },
		{ 3317, 3215, 0 },
		{ 3325, 2823, 0 },
		{ 3332, 2555, 0 },
		{ 0, 0, 348 },
		{ 3119, 3109, 0 },
		{ 3276, 2576, 0 },
		{ 3332, 2558, 0 },
		{ 2273, 0, -59 },
		{ 3334, 2382, 0 },
		{ 3280, 4746, 0 },
		{ 0, 5155, 333 },
		{ 3217, 3067, 0 },
		{ 0, 0, 314 },
		{ 3330, 2254, 0 },
		{ 3324, 3081, 0 },
		{ 3319, 2731, 0 },
		{ 2794, 5258, 0 },
		{ 0, 0, 353 },
		{ 2086, 3205, 355 },
		{ 2286, 2858, 355 },
		{ -2284, 8, 304 },
		{ -2285, 5273, 0 },
		{ 3280, 5254, 0 },
		{ 3224, 5234, 0 },
		{ 0, 0, 305 },
		{ 3224, 5252, 0 },
		{ -2290, 22, 0 },
		{ -2291, 5266, 0 },
		{ 2294, 0, 306 },
		{ 3224, 5253, 0 },
		{ 3280, 5369, 0 },
		{ 0, 0, 307 },
		{ 2686, 4637, 401 },
		{ 0, 0, 401 },
		{ 3317, 3245, 0 },
		{ 3163, 3151, 0 },
		{ 3328, 3379, 0 },
		{ 3322, 1850, 0 },
		{ 3325, 2769, 0 },
		{ 3330, 2264, 0 },
		{ 3280, 7, 0 },
		{ 3334, 2391, 0 },
		{ 3322, 1852, 0 },
		{ 3276, 2601, 0 },
		{ 2309, 5222, 0 },
		{ 3280, 2276, 0 },
		{ 3328, 3396, 0 },
		{ 3334, 2397, 0 },
		{ 3328, 3402, 0 },
		{ 3319, 2707, 0 },
		{ 3317, 3266, 0 },
		{ 3330, 1922, 0 },
		{ 3317, 3270, 0 },
		{ 3334, 2401, 0 },
		{ 3295, 2343, 0 },
		{ 3335, 5089, 0 },
		{ 0, 0, 400 },
		{ 3224, 5226, 457 },
		{ 0, 0, 406 },
		{ 0, 0, 408 },
		{ 2342, 908, 447 },
		{ 2541, 922, 447 },
		{ 2567, 920, 447 },
		{ 2501, 921, 447 },
		{ 2343, 935, 447 },
		{ 2341, 925, 447 },
		{ 2567, 919, 447 },
		{ 2365, 935, 447 },
		{ 2567, 939, 447 },
		{ 2536, 937, 447 },
		{ 2541, 938, 447 },
		{ 2471, 947, 447 },
		{ 2339, 965, 447 },
		{ 3317, 1958, 446 },
		{ 2374, 2997, 457 },
		{ 2607, 937, 447 },
		{ 2365, 945, 447 },
		{ 2541, 951, 447 },
		{ 2379, 952, 447 },
		{ 2541, 949, 447 },
		{ 3317, 3192, 457 },
		{ -2345, 5486, 402 },
		{ -2346, 5268, 0 },
		{ 2607, 946, 447 },
		{ 2616, 446, 447 },
		{ 2607, 949, 447 },
		{ 2434, 947, 447 },
		{ 2541, 955, 447 },
		{ 2548, 950, 447 },
		{ 2541, 957, 447 },
		{ 2471, 966, 447 },
		{ 2434, 963, 447 },
		{ 2567, 1045, 447 },
		{ 2501, 958, 447 },
		{ 2471, 980, 447 },
		{ 2567, 964, 447 },
		{ 2339, 957, 447 },
		{ 2586, 964, 447 },
		{ 2376, 978, 447 },
		{ 2339, 989, 447 },
		{ 2548, 999, 447 },
		{ 2339, 1009, 447 },
		{ 2582, 1030, 447 },
		{ 2541, 1036, 447 },
		{ 2607, 1268, 447 },
		{ 2582, 1032, 447 },
		{ 2439, 1046, 447 },
		{ 2616, 662, 447 },
		{ 3317, 1991, 443 },
		{ 2409, 1632, 0 },
		{ 3317, 2001, 444 },
		{ 2582, 1046, 447 },
		{ 2582, 1047, 447 },
		{ 3276, 2624, 0 },
		{ 3224, 5245, 0 },
		{ 2339, 1060, 447 },
		{ 3332, 2301, 0 },
		{ 2536, 1085, 447 },
		{ 2421, 1070, 447 },
		{ 2582, 1078, 447 },
		{ 2586, 1073, 447 },
		{ 2586, 1075, 447 },
		{ 2439, 1120, 447 },
		{ 2536, 1128, 447 },
		{ 2536, 1129, 447 },
		{ 2501, 1113, 447 },
		{ 2536, 1131, 447 },
		{ 2567, 1119, 447 },
		{ 2501, 1116, 447 },
		{ 2536, 1160, 447 },
		{ 2471, 1165, 447 },
		{ 2567, 1149, 447 },
		{ 2616, 668, 447 },
		{ 2616, 670, 447 },
		{ 2575, 1151, 447 },
		{ 2575, 1188, 447 },
		{ 2536, 1203, 447 },
		{ 2523, 1205, 447 },
		{ 2421, 1189, 447 },
		{ 2548, 1196, 447 },
		{ 2471, 1211, 447 },
		{ 2523, 1209, 447 },
		{ 3327, 2795, 0 },
		{ 2443, 1665, 0 },
		{ 2409, 0, 0 },
		{ 3226, 2938, 445 },
		{ 2445, 1666, 0 },
		{ 2339, 889, 447 },
		{ 3314, 3298, 0 },
		{ 0, 0, 404 },
		{ 2536, 1236, 447 },
		{ 3163, 3147, 0 },
		{ 2616, 672, 447 },
		{ 2439, 1231, 447 },
		{ 2586, 1225, 447 },
		{ 2616, 674, 447 },
		{ 2541, 1268, 447 },
		{ 2339, 1253, 447 },
		{ 2567, 1257, 447 },
		{ 2437, 1273, 447 },
		{ 2536, 1273, 447 },
		{ 2616, 10, 447 },
		{ 2586, 1260, 447 },
		{ 2541, 1273, 447 },
		{ 2616, 117, 447 },
		{ 2586, 1263, 447 },
		{ 2471, 1283, 447 },
		{ 3280, 2552, 0 },
		{ 3330, 2085, 0 },
		{ 2575, 1266, 447 },
		{ 2339, 1270, 447 },
		{ 2567, 1269, 447 },
		{ 2616, 120, 447 },
		{ 2339, 1285, 447 },
		{ 2586, 1269, 447 },
		{ 2339, 1278, 447 },
		{ 2339, 1268, 447 },
		{ 3217, 3049, 0 },
		{ 2443, 0, 0 },
		{ 3226, 3008, 443 },
		{ 2445, 0, 0 },
		{ 3226, 3030, 444 },
		{ 3224, 5236, 0 },
		{ 0, 0, 449 },
		{ 2567, 1274, 447 },
		{ 2480, 5368, 0 },
		{ 3325, 2382, 0 },
		{ 2471, 1292, 447 },
		{ 2616, 212, 447 },
		{ 3295, 2227, 0 },
		{ 2641, 6, 447 },
		{ 2575, 1277, 447 },
		{ 2471, 1296, 447 },
		{ 2541, 1289, 447 },
		{ 2586, 1279, 447 },
		{ 2567, 1282, 447 },
		{ 3280, 2259, 0 },
		{ 2616, 214, 447 },
		{ 2501, 1279, 447 },
		{ 3332, 2321, 0 },
		{ 2541, 1293, 447 },
		{ 2586, 1283, 447 },
		{ 3276, 2579, 0 },
		{ 3276, 2580, 0 },
		{ 3334, 2399, 0 },
		{ 2548, 1289, 447 },
		{ 2567, 1287, 447 },
		{ 2339, 1305, 447 },
		{ 3280, 3662, 0 },
		{ 2536, 1302, 447 },
		{ 2536, 1303, 447 },
		{ 2616, 219, 447 },
		{ 2541, 1301, 447 },
		{ 3325, 2770, 0 },
		{ 2511, 1867, 0 },
		{ 2616, 324, 447 },
		{ 3327, 2526, 0 },
		{ 3217, 3051, 0 },
		{ 2586, 1291, 447 },
		{ 3295, 2219, 0 },
		{ 3330, 2146, 0 },
		{ 3335, 5064, 0 },
		{ 3280, 5445, 412 },
		{ 2607, 1299, 447 },
		{ 2586, 1293, 447 },
		{ 2607, 1301, 447 },
		{ 2541, 1310, 447 },
		{ 2616, 330, 447 },
		{ 3332, 2530, 0 },
		{ 3327, 2785, 0 },
		{ 2541, 1308, 447 },
		{ 3163, 3119, 0 },
		{ 2548, 1303, 447 },
		{ 2541, 1311, 447 },
		{ 3217, 2860, 0 },
		{ 3217, 2861, 0 },
		{ 3317, 3209, 0 },
		{ 2339, 1299, 447 },
		{ 2541, 1313, 447 },
		{ 2586, 1303, 447 },
		{ 3332, 2545, 0 },
		{ 2616, 332, 447 },
		{ 2616, 345, 447 },
		{ 3334, 2231, 0 },
		{ 2582, 1312, 447 },
		{ 3317, 3228, 0 },
		{ 2511, 33, 448 },
		{ 2577, 36, 448 },
		{ 3332, 2349, 0 },
		{ 3266, 4062, 0 },
		{ 3217, 3026, 0 },
		{ 3319, 2717, 0 },
		{ 2541, 1329, 447 },
		{ 3330, 1988, 0 },
		{ 3328, 3397, 0 },
		{ 2641, 210, 447 },
		{ 2548, 1324, 447 },
		{ 2536, 1335, 447 },
		{ 2548, 1326, 447 },
		{ 2339, 1338, 447 },
		{ 3280, 2257, 0 },
		{ 3077, 2452, 0 },
		{ 3334, 2412, 0 },
		{ 2582, 1329, 447 },
		{ 2561, 5373, 0 },
		{ 2582, 1330, 447 },
		{ 2548, 1356, 447 },
		{ 3330, 2136, 0 },
		{ 3330, 2138, 0 },
		{ 3317, 3267, 0 },
		{ 2536, 1367, 447 },
		{ 2582, 1359, 447 },
		{ 2339, 1369, 447 },
		{ 3334, 2438, 0 },
		{ 3332, 2183, 0 },
		{ 3280, 2550, 0 },
		{ 3317, 3281, 0 },
		{ 2339, 1367, 447 },
		{ 3335, 5029, 0 },
		{ 2511, 27, 448 },
		{ 3163, 3148, 0 },
		{ 3271, 3722, 0 },
		{ 3330, 2180, 0 },
		{ 3217, 2855, 0 },
		{ 2339, 1390, 447 },
		{ 3328, 3385, 0 },
		{ 3330, 2191, 0 },
		{ 3335, 5173, 0 },
		{ 0, 0, 433 },
		{ 2567, 1388, 447 },
		{ 2582, 1393, 447 },
		{ 2582, 1394, 447 },
		{ 2616, 437, 447 },
		{ 3332, 2532, 0 },
		{ 3322, 1909, 0 },
		{ 3332, 2536, 0 },
		{ 2568, 1412, 447 },
		{ 3280, 2289, 0 },
		{ 2616, 439, 447 },
		{ 2582, 1407, 447 },
		{ 2596, 5343, 0 },
		{ 2597, 5349, 0 },
		{ 2598, 5372, 0 },
		{ 2339, 1404, 447 },
		{ 2339, 1416, 447 },
		{ 2616, 444, 447 },
		{ 3096, 2835, 0 },
		{ 3328, 3349, 0 },
		{ 3163, 3126, 0 },
		{ 3295, 2308, 0 },
		{ 3314, 3297, 0 },
		{ 2339, 1433, 447 },
		{ 3280, 5359, 438 },
		{ 2411, 814, 448 },
		{ 2608, 5295, 0 },
		{ 3295, 2314, 0 },
		{ 3276, 2650, 0 },
		{ 3330, 1955, 0 },
		{ 2339, 1440, 447 },
		{ 3330, 1982, 0 },
		{ 3295, 2320, 0 },
		{ 2616, 548, 447 },
		{ 2339, 1437, 447 },
		{ 2616, 552, 447 },
		{ 3280, 2669, 0 },
		{ 3334, 2419, 0 },
		{ 3325, 2826, 0 },
		{ 3319, 2711, 0 },
		{ 2616, 554, 447 },
		{ 3334, 2425, 0 },
		{ 3280, 2274, 0 },
		{ 2616, 556, 447 },
		{ 3330, 2185, 0 },
		{ 3330, 2187, 0 },
		{ 3314, 2953, 0 },
		{ 2616, 558, 447 },
		{ 2616, 560, 447 },
		{ 3329, 2174, 0 },
		{ 3217, 2866, 0 },
		{ 3334, 2370, 0 },
		{ 3163, 3144, 0 },
		{ 3325, 2795, 0 },
		{ 3322, 1853, 0 },
		{ 2616, 1649, 447 },
		{ 3332, 2201, 0 },
		{ 2644, 5395, 0 },
		{ 3317, 3256, 0 },
		{ 3335, 4955, 0 },
		{ 2616, 562, 447 },
		{ 3295, 2348, 0 },
		{ 3335, 5023, 0 },
		{ 3280, 2667, 0 },
		{ 2641, 666, 447 },
		{ 3332, 2307, 0 },
		{ 3317, 3265, 0 },
		{ 3330, 2163, 0 },
		{ 3328, 3367, 0 },
		{ 2656, 5228, 0 },
		{ 3332, 2176, 0 },
		{ 3332, 2526, 0 },
		{ 3334, 2386, 0 },
		{ 3280, 2261, 0 },
		{ 3334, 2388, 0 },
		{ 3334, 2389, 0 },
		{ 3317, 3279, 0 },
		{ 3280, 2265, 0 },
		{ 3295, 2221, 0 },
		{ 3295, 2360, 0 },
		{ 3317, 3155, 0 },
		{ 3276, 2625, 0 },
		{ 3276, 2626, 0 },
		{ 2671, 5332, 0 },
		{ 3276, 2627, 0 },
		{ 3317, 3161, 0 },
		{ 3295, 2361, 0 },
		{ 3328, 3400, 0 },
		{ 3329, 3312, 0 },
		{ 2677, 889, 447 },
		{ 3317, 3165, 0 },
		{ 3077, 2468, 0 },
		{ 3335, 5165, 0 },
		{ 3295, 2364, 0 },
		{ 3280, 5478, 410 },
		{ 3295, 2229, 0 },
		{ 3335, 4918, 0 },
		{ 3280, 5317, 419 },
		{ 3332, 2547, 0 },
		{ 3280, 4106, 0 },
		{ 3077, 2454, 0 },
		{ 3276, 2640, 0 },
		{ 3335, 4957, 0 },
		{ 3330, 2219, 0 },
		{ 3327, 2791, 0 },
		{ 3328, 3350, 0 },
		{ 3163, 3115, 0 },
		{ 3119, 3107, 0 },
		{ 3332, 2554, 0 },
		{ 3334, 2402, 0 },
		{ 3317, 3186, 0 },
		{ 3317, 3188, 0 },
		{ 3077, 2459, 0 },
		{ 3334, 2403, 0 },
		{ 3217, 2859, 0 },
		{ 3299, 1710, 0 },
		{ 3276, 2653, 0 },
		{ 3314, 3286, 0 },
		{ 3322, 1851, 0 },
		{ 3295, 2090, 0 },
		{ 3119, 3106, 0 },
		{ 3276, 2659, 0 },
		{ 3077, 2469, 0 },
		{ 3276, 2663, 0 },
		{ 3317, 3205, 0 },
		{ 3335, 4963, 0 },
		{ 3280, 5314, 436 },
		{ 3276, 2666, 0 },
		{ 3330, 2238, 0 },
		{ 0, 0, 454 },
		{ 3295, 2291, 0 },
		{ 3217, 3009, 0 },
		{ 3280, 5334, 418 },
		{ 3328, 3395, 0 },
		{ 0, 4643, 0 },
		{ 3317, 3210, 0 },
		{ 3217, 3010, 0 },
		{ 3280, 5371, 442 },
		{ 3217, 3012, 0 },
		{ 3217, 3015, 0 },
		{ 3334, 2411, 0 },
		{ 3163, 3131, 0 },
		{ 2724, 5328, 0 },
		{ 2791, 3346, 0 },
		{ 3332, 2485, 0 },
		{ 3317, 3222, 0 },
		{ 3317, 3224, 0 },
		{ 3330, 2244, 0 },
		{ 3332, 2489, 0 },
		{ 2715, 1557, 0 },
		{ 2732, 5343, 0 },
		{ 3332, 2511, 0 },
		{ 2794, 5259, 0 },
		{ 3077, 2461, 0 },
		{ 3329, 3325, 0 },
		{ 2738, 5393, 0 },
		{ 3330, 2249, 0 },
		{ 3334, 2416, 0 },
		{ 3314, 3288, 0 },
		{ 2742, 5372, 0 },
		{ 3317, 3236, 0 },
		{ 3217, 3035, 0 },
		{ 2745, 5390, 0 },
		{ 0, 1542, 0 },
		{ 3325, 2832, 0 },
		{ 3335, 5093, 0 },
		{ 3334, 2418, 0 },
		{ 3330, 2252, 0 },
		{ 3332, 2502, 0 },
		{ 3325, 2765, 0 },
		{ 3317, 3249, 0 },
		{ 3295, 2302, 0 },
		{ 3280, 3095, 0 },
		{ 3328, 3380, 0 },
		{ 2791, 3337, 0 },
		{ 2758, 5286, 0 },
		{ 2759, 5295, 0 },
		{ 3324, 3079, 0 },
		{ 2791, 3339, 0 },
		{ 3317, 3255, 0 },
		{ 3295, 2203, 0 },
		{ 3319, 2733, 0 },
		{ 3330, 2256, 0 },
		{ 3329, 3323, 0 },
		{ 3325, 2793, 0 },
		{ 3334, 2428, 0 },
		{ 3280, 3726, 0 },
		{ 3295, 2306, 0 },
		{ 3217, 3064, 0 },
		{ 2772, 5348, 0 },
		{ 3332, 2314, 0 },
		{ 3334, 2434, 0 },
		{ 3319, 2701, 0 },
		{ 3329, 3045, 0 },
		{ 3317, 3271, 0 },
		{ 3335, 5109, 0 },
		{ 3280, 5469, 439 },
		{ 3328, 3410, 0 },
		{ 3332, 2515, 0 },
		{ 3276, 2604, 0 },
		{ 3317, 3276, 0 },
		{ 3276, 2606, 0 },
		{ 3077, 2463, 0 },
		{ 3322, 1885, 0 },
		{ 2791, 3342, 0 },
		{ 3328, 3351, 0 },
		{ 3314, 2936, 0 },
		{ 3314, 2938, 0 },
		{ 2790, 5275, 0 },
		{ 3328, 3355, 0 },
		{ 3335, 4990, 0 },
		{ 3330, 2263, 0 },
		{ 3317, 3156, 0 },
		{ 2846, 1669, 0 },
		{ 3295, 2315, 0 },
		{ 3332, 2525, 0 },
		{ 3217, 2970, 0 },
		{ 3328, 3365, 0 },
		{ 2800, 5277, 0 },
		{ 3276, 2618, 0 },
		{ 3319, 2352, 0 },
		{ 3077, 2471, 0 },
		{ 3328, 3372, 0 },
		{ 3217, 3006, 0 },
		{ 3328, 3374, 0 },
		{ 3335, 5163, 0 },
		{ 3280, 5434, 434 },
		{ 3330, 2271, 0 },
		{ 3334, 2373, 0 },
		{ 3335, 4875, 0 },
		{ 3335, 4879, 0 },
		{ 3330, 2273, 0 },
		{ 3334, 2377, 0 },
		{ 3163, 3134, 0 },
		{ 3217, 3014, 0 },
		{ 2791, 3343, 0 },
		{ 3317, 3177, 0 },
		{ 3317, 3178, 0 },
		{ 3335, 4959, 0 },
		{ 0, 3345, 0 },
		{ 3280, 5465, 417 },
		{ 3328, 3390, 0 },
		{ 0, 5257, 0 },
		{ 3330, 1917, 0 },
		{ 3299, 1712, 0 },
		{ 3330, 1919, 0 },
		{ 3077, 2462, 0 },
		{ 3330, 1920, 0 },
		{ 3332, 2342, 0 },
		{ 3119, 3108, 0 },
		{ 3332, 2543, 0 },
		{ 3317, 3191, 0 },
		{ 3330, 1945, 0 },
		{ 3295, 2326, 0 },
		{ 3295, 2328, 0 },
		{ 3280, 5326, 411 },
		{ 3332, 2551, 0 },
		{ 3295, 2330, 0 },
		{ 3280, 5338, 426 },
		{ 3280, 5343, 427 },
		{ 3295, 2331, 0 },
		{ 3217, 3041, 0 },
		{ 3163, 3132, 0 },
		{ 3325, 2792, 0 },
		{ 3217, 3043, 0 },
		{ 3077, 2474, 0 },
		{ 3077, 2451, 0 },
		{ 0, 0, 453 },
		{ 3217, 3047, 0 },
		{ 3330, 1946, 0 },
		{ 3330, 1947, 0 },
		{ 2847, 5297, 0 },
		{ 2848, 5285, 0 },
		{ 2849, 5287, 0 },
		{ 3330, 1949, 0 },
		{ 3324, 3077, 0 },
		{ 3077, 2456, 0 },
		{ 2853, 5323, 0 },
		{ 3314, 3305, 0 },
		{ 3334, 2392, 0 },
		{ 3217, 3056, 0 },
		{ 3328, 3366, 0 },
		{ 3317, 3218, 0 },
		{ 3334, 2394, 0 },
		{ 3335, 5031, 0 },
		{ 3335, 5033, 0 },
		{ 3276, 2665, 0 },
		{ 3317, 3221, 0 },
		{ 3217, 3059, 0 },
		{ 3325, 2803, 0 },
		{ 3330, 1950, 0 },
		{ 3330, 1952, 0 },
		{ 3325, 2807, 0 },
		{ 3295, 2337, 0 },
		{ 0, 1668, 0 },
		{ 3327, 2821, 0 },
		{ 3295, 2214, 0 },
		{ 3295, 2217, 0 },
		{ 3335, 5167, 0 },
		{ 3335, 5169, 0 },
		{ 3317, 3237, 0 },
		{ 3332, 2305, 0 },
		{ 3317, 3239, 0 },
		{ 3328, 3392, 0 },
		{ 3332, 2493, 0 },
		{ 3330, 1999, 0 },
		{ 3295, 2347, 0 },
		{ 3335, 4951, 0 },
		{ 3334, 1017, 414 },
		{ 3280, 5328, 429 },
		{ 3119, 3101, 0 },
		{ 3334, 2406, 0 },
		{ 3330, 2003, 0 },
		{ 3217, 2878, 0 },
		{ 3324, 3093, 0 },
		{ 3324, 3095, 0 },
		{ 3217, 2962, 0 },
		{ 2891, 5272, 0 },
		{ 3330, 2008, 0 },
		{ 3217, 2967, 0 },
		{ 3330, 2012, 0 },
		{ 3329, 3316, 0 },
		{ 3280, 5380, 425 },
		{ 3280, 5383, 440 },
		{ 3334, 2410, 0 },
		{ 3077, 2455, 0 },
		{ 3325, 2746, 0 },
		{ 3330, 2013, 0 },
		{ 3276, 2594, 0 },
		{ 3217, 3007, 0 },
		{ 2902, 5353, 0 },
		{ 3280, 5408, 413 },
		{ 3335, 5097, 0 },
		{ 2904, 5365, 0 },
		{ 2909, 1665, 0 },
		{ 3330, 2015, 0 },
		{ 2907, 5371, 0 },
		{ 2908, 5373, 0 },
		{ 3330, 2016, 0 },
		{ 3327, 2779, 0 },
		{ 2911, 5339, 0 },
		{ 3325, 2772, 0 },
		{ 3328, 3360, 0 },
		{ 3334, 2415, 0 },
		{ 3328, 3362, 0 },
		{ 3317, 3277, 0 },
		{ 3335, 4873, 0 },
		{ 3332, 2513, 0 },
		{ 3295, 2359, 0 },
		{ 3332, 2516, 0 },
		{ 3335, 4885, 0 },
		{ 3280, 5454, 431 },
		{ 3335, 4887, 0 },
		{ 3335, 4891, 0 },
		{ 2909, 1559, 0 },
		{ 3335, 4920, 0 },
		{ 3335, 4945, 0 },
		{ 0, 1592, 0 },
		{ 3217, 3027, 0 },
		{ 3327, 2777, 0 },
		{ 3317, 3157, 0 },
		{ 3330, 2041, 0 },
		{ 3217, 3031, 0 },
		{ 3330, 2059, 0 },
		{ 3334, 2423, 0 },
		{ 3280, 5310, 437 },
		{ 3334, 2424, 0 },
		{ 3335, 4965, 0 },
		{ 3276, 2614, 0 },
		{ 0, 0, 456 },
		{ 0, 0, 455 },
		{ 3280, 5319, 415 },
		{ 3335, 4992, 0 },
		{ 0, 0, 452 },
		{ 0, 0, 451 },
		{ 3335, 5019, 0 },
		{ 3325, 2802, 0 },
		{ 3217, 3040, 0 },
		{ 3335, 5025, 0 },
		{ 3295, 2366, 0 },
		{ 3077, 2473, 0 },
		{ 3332, 2528, 0 },
		{ 3328, 3389, 0 },
		{ 3335, 5062, 0 },
		{ 3280, 5345, 409 },
		{ 2948, 5251, 0 },
		{ 3280, 5352, 441 },
		{ 3280, 5355, 416 },
		{ 3317, 3173, 0 },
		{ 3325, 2806, 0 },
		{ 3280, 5362, 423 },
		{ 2951, 5271, 0 },
		{ 3330, 2062, 0 },
		{ 3334, 2427, 0 },
		{ 3330, 2073, 0 },
		{ 3280, 5373, 432 },
		{ 3280, 2546, 0 },
		{ 3335, 5099, 0 },
		{ 3317, 3179, 0 },
		{ 3327, 2787, 0 },
		{ 3335, 5105, 0 },
		{ 3335, 5107, 0 },
		{ 3332, 2533, 0 },
		{ 3330, 2091, 0 },
		{ 3280, 5398, 420 },
		{ 3335, 5161, 0 },
		{ 3217, 3055, 0 },
		{ 3280, 5400, 424 },
		{ 3280, 5404, 428 },
		{ 3334, 2430, 0 },
		{ 3317, 3185, 0 },
		{ 3280, 5410, 421 },
		{ 3325, 2821, 0 },
		{ 3335, 5171, 0 },
		{ 3334, 2431, 0 },
		{ 3317, 3189, 0 },
		{ 3280, 5418, 430 },
		{ 3328, 3413, 0 },
		{ 3335, 4877, 0 },
		{ 3330, 2093, 0 },
		{ 3280, 5428, 422 },
		{ 3217, 3060, 0 },
		{ 3332, 2539, 0 },
		{ 3276, 2635, 0 },
		{ 3295, 2287, 0 },
		{ 3335, 4893, 0 },
		{ 3280, 5439, 435 },
		{ 3224, 5235, 457 },
		{ 2984, 0, 406 },
		{ 0, 0, 407 },
		{ -2982, 5488, 402 },
		{ -2983, 5272, 0 },
		{ 3280, 5246, 0 },
		{ 3224, 5224, 0 },
		{ 0, 0, 403 },
		{ 3224, 5242, 0 },
		{ -2988, 16, 0 },
		{ -2989, 5276, 0 },
		{ 2992, 0, 404 },
		{ 3224, 5243, 0 },
		{ 3280, 5459, 0 },
		{ 0, 0, 405 },
		{ 3271, 3731, 169 },
		{ 0, 0, 169 },
		{ 0, 0, 170 },
		{ 3295, 2289, 0 },
		{ 3317, 3199, 0 },
		{ 3334, 2439, 0 },
		{ 3001, 5379, 0 },
		{ 3327, 2797, 0 },
		{ 3322, 1854, 0 },
		{ 3276, 2643, 0 },
		{ 3329, 3317, 0 },
		{ 3330, 2102, 0 },
		{ 3217, 3075, 0 },
		{ 3332, 2549, 0 },
		{ 3276, 2647, 0 },
		{ 3295, 2292, 0 },
		{ 3335, 5017, 0 },
		{ 0, 0, 167 },
		{ 3150, 5025, 192 },
		{ 0, 0, 192 },
		{ 3330, 2104, 0 },
		{ 3016, 5231, 0 },
		{ 3317, 2860, 0 },
		{ 3328, 3368, 0 },
		{ 3329, 3334, 0 },
		{ 3324, 3096, 0 },
		{ 3021, 5236, 0 },
		{ 3280, 2663, 0 },
		{ 3317, 3214, 0 },
		{ 3276, 2652, 0 },
		{ 3317, 3216, 0 },
		{ 3334, 2372, 0 },
		{ 3328, 3377, 0 },
		{ 3330, 2116, 0 },
		{ 3217, 2965, 0 },
		{ 3332, 2556, 0 },
		{ 3276, 2657, 0 },
		{ 3032, 5269, 0 },
		{ 3280, 3097, 0 },
		{ 3317, 3223, 0 },
		{ 3163, 3133, 0 },
		{ 3332, 2557, 0 },
		{ 3334, 2374, 0 },
		{ 3317, 3227, 0 },
		{ 3039, 5262, 0 },
		{ 3334, 2233, 0 },
		{ 3317, 3229, 0 },
		{ 3314, 3299, 0 },
		{ 3322, 1881, 0 },
		{ 3329, 3319, 0 },
		{ 3317, 3232, 0 },
		{ 3046, 5294, 0 },
		{ 3327, 2789, 0 },
		{ 3322, 1882, 0 },
		{ 3276, 2667, 0 },
		{ 3329, 3326, 0 },
		{ 3330, 2134, 0 },
		{ 3217, 3011, 0 },
		{ 3332, 2480, 0 },
		{ 3276, 2670, 0 },
		{ 3335, 4889, 0 },
		{ 0, 0, 190 },
		{ 3057, 0, 1 },
		{ -3057, 1506, 281 },
		{ 3317, 3128, 287 },
		{ 0, 0, 287 },
		{ 3295, 2307, 0 },
		{ 3276, 2675, 0 },
		{ 3317, 3248, 0 },
		{ 3314, 3301, 0 },
		{ 3334, 2384, 0 },
		{ 0, 0, 286 },
		{ 3067, 5358, 0 },
		{ 3319, 2248, 0 },
		{ 3328, 3354, 0 },
		{ 3096, 2837, 0 },
		{ 3317, 3253, 0 },
		{ 3163, 3141, 0 },
		{ 3217, 3029, 0 },
		{ 3325, 2816, 0 },
		{ 3317, 3257, 0 },
		{ 3076, 5341, 0 },
		{ 3332, 2317, 0 },
		{ 0, 2460, 0 },
		{ 3330, 2148, 0 },
		{ 3217, 3034, 0 },
		{ 3332, 2492, 0 },
		{ 3276, 2578, 0 },
		{ 3295, 2312, 0 },
		{ 3335, 5035, 0 },
		{ 0, 0, 285 },
		{ 0, 4813, 195 },
		{ 0, 0, 195 },
		{ 3332, 2494, 0 },
		{ 3322, 1743, 0 },
		{ 3276, 2582, 0 },
		{ 3314, 3304, 0 },
		{ 3092, 5377, 0 },
		{ 3329, 3090, 0 },
		{ 3324, 3082, 0 },
		{ 3317, 3273, 0 },
		{ 3329, 3331, 0 },
		{ 0, 2845, 0 },
		{ 3217, 3045, 0 },
		{ 3276, 2583, 0 },
		{ 3119, 3104, 0 },
		{ 3335, 5134, 0 },
		{ 0, 0, 193 },
		{ 3150, 5023, 189 },
		{ 0, 0, 188 },
		{ 0, 0, 189 },
		{ 3330, 2160, 0 },
		{ 3107, 5209, 0 },
		{ 3330, 2189, 0 },
		{ 3324, 3097, 0 },
		{ 3317, 3282, 0 },
		{ 3111, 5233, 0 },
		{ 3280, 3093, 0 },
		{ 3317, 3154, 0 },
		{ 3119, 3099, 0 },
		{ 3217, 3050, 0 },
		{ 3276, 2588, 0 },
		{ 3276, 2590, 0 },
		{ 3217, 3053, 0 },
		{ 3276, 2592, 0 },
		{ 0, 3105, 0 },
		{ 3121, 5234, 0 },
		{ 3332, 2331, 0 },
		{ 3163, 3120, 0 },
		{ 3124, 5249, 0 },
		{ 3317, 2858, 0 },
		{ 3328, 3405, 0 },
		{ 3329, 3309, 0 },
		{ 3324, 3094, 0 },
		{ 3129, 5254, 0 },
		{ 3280, 2671, 0 },
		{ 3317, 3169, 0 },
		{ 3276, 2596, 0 },
		{ 3317, 3171, 0 },
		{ 3334, 2393, 0 },
		{ 3328, 3416, 0 },
		{ 3330, 2165, 0 },
		{ 3217, 3061, 0 },
		{ 3332, 2501, 0 },
		{ 3276, 2600, 0 },
		{ 3140, 5283, 0 },
		{ 3327, 2793, 0 },
		{ 3322, 1747, 0 },
		{ 3276, 2602, 0 },
		{ 3329, 3330, 0 },
		{ 3330, 2174, 0 },
		{ 3217, 3069, 0 },
		{ 3332, 2504, 0 },
		{ 3276, 2605, 0 },
		{ 3335, 5027, 0 },
		{ 0, 0, 182 },
		{ 0, 4877, 181 },
		{ 0, 0, 181 },
		{ 3330, 2176, 0 },
		{ 3154, 5288, 0 },
		{ 3330, 2193, 0 },
		{ 3324, 3078, 0 },
		{ 3317, 3187, 0 },
		{ 3158, 5315, 0 },
		{ 3317, 2840, 0 },
		{ 3276, 2608, 0 },
		{ 3314, 3293, 0 },
		{ 3162, 5311, 0 },
		{ 3332, 2346, 0 },
		{ 0, 3123, 0 },
		{ 3165, 5324, 0 },
		{ 3317, 2854, 0 },
		{ 3328, 3370, 0 },
		{ 3329, 3324, 0 },
		{ 3324, 3083, 0 },
		{ 3170, 5330, 0 },
		{ 3280, 2665, 0 },
		{ 3317, 3195, 0 },
		{ 3276, 2611, 0 },
		{ 3317, 3197, 0 },
		{ 3334, 2400, 0 },
		{ 3328, 3378, 0 },
		{ 3330, 2186, 0 },
		{ 3217, 2879, 0 },
		{ 3332, 2510, 0 },
		{ 3276, 2615, 0 },
		{ 3181, 5349, 0 },
		{ 3327, 2823, 0 },
		{ 3322, 1816, 0 },
		{ 3276, 2617, 0 },
		{ 3329, 3315, 0 },
		{ 3330, 2197, 0 },
		{ 3217, 2971, 0 },
		{ 3332, 2514, 0 },
		{ 3276, 2620, 0 },
		{ 3335, 4883, 0 },
		{ 0, 0, 179 },
		{ 0, 4418, 184 },
		{ 0, 0, 184 },
		{ 0, 0, 185 },
		{ 3276, 2621, 0 },
		{ 3295, 2329, 0 },
		{ 3330, 2203, 0 },
		{ 3317, 3213, 0 },
		{ 3328, 3398, 0 },
		{ 3314, 3300, 0 },
		{ 3201, 5379, 0 },
		{ 3317, 2852, 0 },
		{ 3299, 1672, 0 },
		{ 3328, 3403, 0 },
		{ 3325, 2827, 0 },
		{ 3322, 1846, 0 },
		{ 3328, 3407, 0 },
		{ 3330, 2212, 0 },
		{ 3217, 3013, 0 },
		{ 3332, 2521, 0 },
		{ 3276, 2628, 0 },
		{ 3212, 5220, 0 },
		{ 3327, 2825, 0 },
		{ 3322, 1848, 0 },
		{ 3276, 2630, 0 },
		{ 3329, 3314, 0 },
		{ 3330, 2222, 0 },
		{ 0, 3025, 0 },
		{ 3332, 2524, 0 },
		{ 3276, 2633, 0 },
		{ 3297, 5185, 0 },
		{ 0, 0, 183 },
		{ 3317, 3230, 457 },
		{ 3334, 1617, 26 },
		{ 0, 5249, 457 },
		{ 3233, 0, 457 },
		{ 2338, 2993, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 3276, 2637, 0 },
		{ -3232, 5489, 0 },
		{ 3334, 91, 0 },
		{ 0, 0, 28 },
		{ 3314, 3303, 0 },
		{ 0, 0, 27 },
		{ 0, 0, 22 },
		{ 0, 0, 34 },
		{ 0, 0, 35 },
		{ 3254, 4220, 39 },
		{ 0, 3933, 39 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 3266, 4082, 0 },
		{ 3281, 4776, 0 },
		{ 3271, 3705, 0 },
		{ 0, 0, 37 },
		{ 3275, 3769, 0 },
		{ 3279, 3599, 0 },
		{ 3289, 2111, 0 },
		{ 0, 0, 36 },
		{ 3317, 3144, 50 },
		{ 0, 0, 50 },
		{ 0, 4229, 50 },
		{ 3271, 3710, 50 },
		{ 3317, 3240, 50 },
		{ 0, 0, 58 },
		{ 3317, 3241, 0 },
		{ 3276, 2642, 0 },
		{ 3266, 4093, 0 },
		{ 3275, 3778, 0 },
		{ 3330, 2250, 0 },
		{ 3276, 2644, 0 },
		{ 3314, 3290, 0 },
		{ 3271, 3719, 0 },
		{ 0, 4038, 0 },
		{ 3322, 1879, 0 },
		{ 3332, 2534, 0 },
		{ 0, 0, 49 },
		{ 3275, 3785, 0 },
		{ 0, 3723, 0 },
		{ 3334, 2420, 0 },
		{ 3319, 2739, 0 },
		{ 3279, 3596, 54 },
		{ 0, 3792, 0 },
		{ 0, 2648, 0 },
		{ 3317, 3250, 0 },
		{ 3283, 1854, 0 },
		{ 0, 3598, 51 },
		{ 0, 5, 59 },
		{ 0, 4751, 0 },
		{ 3289, 1938, 0 },
		{ 3334, 1694, 0 },
		{ 3290, 1844, 0 },
		{ 0, 0, 57 },
		{ 3325, 2808, 0 },
		{ 0, 0, 55 },
		{ 0, 0, 56 },
		{ 3226, 2098, 0 },
		{ 3334, 1769, 0 },
		{ 3328, 3383, 0 },
		{ 0, 0, 52 },
		{ 0, 0, 53 },
		{ 3295, 2355, 0 },
		{ 0, 2356, 0 },
		{ 3297, 5180, 0 },
		{ 0, 5181, 0 },
		{ 3317, 3260, 0 },
		{ 0, 1715, 0 },
		{ 3328, 3388, 0 },
		{ 3325, 2814, 0 },
		{ 3322, 1911, 0 },
		{ 3328, 3391, 0 },
		{ 3330, 2267, 0 },
		{ 3332, 2550, 0 },
		{ 3334, 2433, 0 },
		{ 3328, 2376, 0 },
		{ 3317, 3269, 0 },
		{ 3332, 2553, 0 },
		{ 3329, 3318, 0 },
		{ 3328, 3399, 0 },
		{ 3334, 2435, 0 },
		{ 3329, 3320, 0 },
		{ 0, 3295, 0 },
		{ 3317, 3164, 0 },
		{ 3322, 1913, 0 },
		{ 0, 3274, 0 },
		{ 3328, 3406, 0 },
		{ 0, 2734, 0 },
		{ 3334, 2437, 0 },
		{ 3329, 3327, 0 },
		{ 0, 1915, 0 },
		{ 3335, 5175, 0 },
		{ 0, 3088, 0 },
		{ 0, 2829, 0 },
		{ 0, 0, 46 },
		{ 3280, 3092, 0 },
		{ 0, 3414, 0 },
		{ 0, 3332, 0 },
		{ 0, 2274, 0 },
		{ 3335, 5176, 0 },
		{ 0, 2559, 0 },
		{ 0, 0, 47 },
		{ 0, 2440, 0 },
		{ 3297, 5177, 0 },
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
		0,
		0,
		0,
		0
	};
	yybackup = backup;
}
