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
fprintf(fpOutputFile,"%d",bGENERATE_CSV_IND_FILE);
#line 2004 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1297 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_TXT_GEN_FILE);
#line 2011 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1299 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 2018 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1300 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 2025 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1302 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 2032 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1303 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 2039 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1305 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 2053 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1313 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  if( TARGET==CUDA )
    strcat(sFileName,"Individual.cu");
  else if( TARGET==STD )
    strcat(sFileName,"Individual.cpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 2070 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1324 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2084 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1332 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2098 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1341 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 2112 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1350 "EaseaLex.l"

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

#line 2175 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1407 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded without warnings.\n");
  printf ("You can now type \"make\" to compile your project.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 2192 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1419 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2199 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1425 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2211 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1431 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2224 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1438 "EaseaLex.l"

#line 2231 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1439 "EaseaLex.l"
lineCounter++;
#line 2238 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1441 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2250 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1447 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2263 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1455 "EaseaLex.l"

#line 2270 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1456 "EaseaLex.l"

  lineCounter++;
 
#line 2279 "EaseaLex.cpp"
		}
		break;
#line 1459 "EaseaLex.l"
               
#line 2284 "EaseaLex.cpp"
	case 162:
		{
#line 1460 "EaseaLex.l"

  fprintf (fpOutputFile,"// User CUDA\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2294 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1466 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user CUDA were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2307 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1474 "EaseaLex.l"

#line 2314 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1475 "EaseaLex.l"

  lineCounter++;
 
#line 2323 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1479 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2335 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1485 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2349 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1493 "EaseaLex.l"

#line 2356 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1494 "EaseaLex.l"

  lineCounter++;
 
#line 2365 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1498 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
      
  BEGIN COPY;
 
#line 2379 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1506 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2394 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1515 "EaseaLex.l"

#line 2401 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1516 "EaseaLex.l"
lineCounter++;
#line 2408 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1521 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2422 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1530 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2436 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1538 "EaseaLex.l"

#line 2443 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1539 "EaseaLex.l"
lineCounter++;
#line 2450 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1542 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2466 "EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 1553 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2482 "EaseaLex.cpp"
		}
		break;
	case 180:
		{
#line 1563 "EaseaLex.l"

#line 2489 "EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 1566 "EaseaLex.l"

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
 
#line 2507 "EaseaLex.cpp"
		}
		break;
	case 182:
		{
#line 1579 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2524 "EaseaLex.cpp"
		}
		break;
	case 183:
		{
#line 1591 "EaseaLex.l"

#line 2531 "EaseaLex.cpp"
		}
		break;
	case 184:
		{
#line 1592 "EaseaLex.l"
lineCounter++;
#line 2538 "EaseaLex.cpp"
		}
		break;
	case 185:
		{
#line 1594 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2554 "EaseaLex.cpp"
		}
		break;
	case 186:
		{
#line 1606 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2570 "EaseaLex.cpp"
		}
		break;
	case 187:
		{
#line 1616 "EaseaLex.l"
lineCounter++;
#line 2577 "EaseaLex.cpp"
		}
		break;
	case 188:
		{
#line 1617 "EaseaLex.l"

#line 2584 "EaseaLex.cpp"
		}
		break;
	case 189:
		{
#line 1621 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2599 "EaseaLex.cpp"
		}
		break;
	case 190:
		{
#line 1631 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2614 "EaseaLex.cpp"
		}
		break;
	case 191:
		{
#line 1640 "EaseaLex.l"

#line 2621 "EaseaLex.cpp"
		}
		break;
	case 192:
		{
#line 1643 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    //fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    fprintf (fpOutputFile,"{\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2635 "EaseaLex.cpp"
		}
		break;
	case 193:
		{
#line 1651 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2649 "EaseaLex.cpp"
		}
		break;
	case 194:
		{
#line 1659 "EaseaLex.l"

#line 2656 "EaseaLex.cpp"
		}
		break;
	case 195:
		{
#line 1663 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2664 "EaseaLex.cpp"
		}
		break;
	case 196:
		{
#line 1665 "EaseaLex.l"

#line 2671 "EaseaLex.cpp"
		}
		break;
	case 197:
		{
#line 1671 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2678 "EaseaLex.cpp"
		}
		break;
	case 198:
		{
#line 1672 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2685 "EaseaLex.cpp"
		}
		break;
	case 199:
	case 200:
		{
#line 1675 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2696 "EaseaLex.cpp"
		}
		break;
	case 201:
	case 202:
		{
#line 1680 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2705 "EaseaLex.cpp"
		}
		break;
	case 203:
	case 204:
		{
#line 1683 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2714 "EaseaLex.cpp"
		}
		break;
	case 205:
	case 206:
		{
#line 1686 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2731 "EaseaLex.cpp"
		}
		break;
	case 207:
	case 208:
		{
#line 1697 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2745 "EaseaLex.cpp"
		}
		break;
	case 209:
	case 210:
		{
#line 1705 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2754 "EaseaLex.cpp"
		}
		break;
	case 211:
	case 212:
		{
#line 1708 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2763 "EaseaLex.cpp"
		}
		break;
	case 213:
	case 214:
		{
#line 1711 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2772 "EaseaLex.cpp"
		}
		break;
	case 215:
	case 216:
		{
#line 1714 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2781 "EaseaLex.cpp"
		}
		break;
	case 217:
	case 218:
		{
#line 1717 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2790 "EaseaLex.cpp"
		}
		break;
	case 219:
	case 220:
		{
#line 1721 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2802 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1727 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2809 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1728 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2816 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1729 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2823 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1730 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2833 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1735 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2840 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1736 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2847 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1737 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
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
fprintf(fpOutputFile,"false");
#line 2882 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1742 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2889 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1743 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2897 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1745 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2905 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1747 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2913 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1749 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2923 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1753 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2930 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1754 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2937 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1755 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2948 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1760 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2955 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1761 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2964 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1764 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2976 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1770 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2985 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1773 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2997 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1779 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 3008 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1784 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 3024 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1794 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3031 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1797 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 3040 "EaseaLex.cpp"
		}
		break;
	case 249:
		{
#line 1800 "EaseaLex.l"
BEGIN COPY;
#line 3047 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1802 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 3054 "EaseaLex.cpp"
		}
		break;
	case 251:
	case 252:
	case 253:
		{
#line 1805 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 3067 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1810 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 3078 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1815 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 3087 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1824 "EaseaLex.l"
;
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
#line 1830 "EaseaLex.l"
 /* do nothing */ 
#line 3122 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1831 "EaseaLex.l"
 /*return '\n';*/ 
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
#line 1835 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 3145 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1838 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 3155 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1842 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
//  printf("match gpnode\n");
  return GPNODE;
 
#line 3167 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1849 "EaseaLex.l"
return STATIC;
#line 3174 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1850 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 3181 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1851 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 3188 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1852 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 3195 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1853 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 3202 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1854 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 3209 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1856 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 3216 "EaseaLex.cpp"
		}
		break;
#line 1857 "EaseaLex.l"
  
#line 3221 "EaseaLex.cpp"
	case 273:
		{
#line 1858 "EaseaLex.l"
return GENOME; 
#line 3226 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1860 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 3236 "EaseaLex.cpp"
		}
		break;
	case 275:
	case 276:
	case 277:
		{
#line 1867 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 3245 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1868 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 3252 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1871 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 3260 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1873 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 3267 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1879 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 3279 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1885 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 3292 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1892 "EaseaLex.l"

#line 3299 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1894 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3310 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1905 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3325 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1915 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3336 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1921 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3345 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1925 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3360 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1938 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3372 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1944 "EaseaLex.l"

#line 3379 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1945 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3392 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1952 "EaseaLex.l"

#line 3399 "EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1953 "EaseaLex.l"
lineCounter++;
#line 3406 "EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1954 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3419 "EaseaLex.cpp"
		}
		break;
	case 295:
		{
#line 1961 "EaseaLex.l"

#line 3426 "EaseaLex.cpp"
		}
		break;
	case 296:
		{
#line 1962 "EaseaLex.l"
lineCounter++;
#line 3433 "EaseaLex.cpp"
		}
		break;
	case 297:
		{
#line 1964 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3446 "EaseaLex.cpp"
		}
		break;
	case 298:
		{
#line 1971 "EaseaLex.l"

#line 3453 "EaseaLex.cpp"
		}
		break;
	case 299:
		{
#line 1972 "EaseaLex.l"
lineCounter++;
#line 3460 "EaseaLex.cpp"
		}
		break;
	case 300:
		{
#line 1975 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3473 "EaseaLex.cpp"
		}
		break;
	case 301:
		{
#line 1982 "EaseaLex.l"

#line 3480 "EaseaLex.cpp"
		}
		break;
	case 302:
		{
#line 1983 "EaseaLex.l"
lineCounter++;
#line 3487 "EaseaLex.cpp"
		}
		break;
	case 303:
		{
#line 1989 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
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
fprintf(fpOutputFile,"genome");
#line 3522 "EaseaLex.cpp"
		}
		break;
	case 308:
		{
#line 1994 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3529 "EaseaLex.cpp"
		}
		break;
	case 309:
		{
#line 1995 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3536 "EaseaLex.cpp"
		}
		break;
	case 310:
		{
#line 1997 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3545 "EaseaLex.cpp"
		}
		break;
	case 311:
		{
#line 2000 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3558 "EaseaLex.cpp"
		}
		break;
	case 312:
	case 313:
		{
#line 2009 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3569 "EaseaLex.cpp"
		}
		break;
	case 314:
	case 315:
		{
#line 2014 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3578 "EaseaLex.cpp"
		}
		break;
	case 316:
	case 317:
		{
#line 2017 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3587 "EaseaLex.cpp"
		}
		break;
	case 318:
	case 319:
		{
#line 2020 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3599 "EaseaLex.cpp"
		}
		break;
	case 320:
	case 321:
		{
#line 2026 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3612 "EaseaLex.cpp"
		}
		break;
	case 322:
	case 323:
		{
#line 2033 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3621 "EaseaLex.cpp"
		}
		break;
	case 324:
	case 325:
		{
#line 2036 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3630 "EaseaLex.cpp"
		}
		break;
	case 326:
	case 327:
		{
#line 2039 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3639 "EaseaLex.cpp"
		}
		break;
	case 328:
	case 329:
		{
#line 2042 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3648 "EaseaLex.cpp"
		}
		break;
	case 330:
	case 331:
		{
#line 2045 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3657 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 2048 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3666 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 2051 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3676 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 2055 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3684 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 2057 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3695 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 2062 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3706 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 2067 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3714 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 2069 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3722 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 2071 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3730 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 2073 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3738 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 2075 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3746 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 2077 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3753 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 2078 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3760 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 2079 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3768 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 2081 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3776 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 2083 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3784 "EaseaLex.cpp"
		}
		break;
	case 347:
		{
#line 2085 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3791 "EaseaLex.cpp"
		}
		break;
	case 348:
		{
#line 2086 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3803 "EaseaLex.cpp"
		}
		break;
	case 349:
		{
#line 2092 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3812 "EaseaLex.cpp"
		}
		break;
	case 350:
		{
#line 2095 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3822 "EaseaLex.cpp"
		}
		break;
	case 351:
		{
#line 2099 "EaseaLex.l"
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
#line 3839 "EaseaLex.cpp"
		}
		break;
	case 352:
	case 353:
		{
#line 2111 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3849 "EaseaLex.cpp"
		}
		break;
	case 354:
		{
#line 2114 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3856 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 2121 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
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
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3877 "EaseaLex.cpp"
		}
		break;
	case 358:
		{
#line 2124 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
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
#line 2127 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3900 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 2131 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3913 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 2139 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3926 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 2148 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3939 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 2157 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3954 "EaseaLex.cpp"
		}
		break;
	case 365:
		{
#line 2167 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3961 "EaseaLex.cpp"
		}
		break;
	case 366:
		{
#line 2168 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3968 "EaseaLex.cpp"
		}
		break;
	case 367:
	case 368:
		{
#line 2171 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3979 "EaseaLex.cpp"
		}
		break;
	case 369:
	case 370:
		{
#line 2176 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3988 "EaseaLex.cpp"
		}
		break;
	case 371:
	case 372:
		{
#line 2179 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3997 "EaseaLex.cpp"
		}
		break;
	case 373:
	case 374:
		{
#line 2182 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 4010 "EaseaLex.cpp"
		}
		break;
	case 375:
	case 376:
		{
#line 2189 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 4023 "EaseaLex.cpp"
		}
		break;
	case 377:
	case 378:
		{
#line 2196 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 4032 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2199 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 4039 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2200 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4046 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2201 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4053 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2202 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 4063 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2207 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4070 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2208 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 4077 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2209 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 4084 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2210 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 4091 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2211 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 4099 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2213 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 4107 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2215 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 4115 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2217 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 4123 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2219 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 4131 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2221 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 4139 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2223 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 4147 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2225 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 4154 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2226 "EaseaLex.l"
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
#line 4177 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2243 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 4188 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2248 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 4202 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2256 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 4209 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2262 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 4219 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2266 "EaseaLex.l"

#line 4226 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2269 "EaseaLex.l"
;
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
#line 2274 "EaseaLex.l"
 /* do nothing */ 
#line 4261 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2275 "EaseaLex.l"
 /*return '\n';*/ 
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
#line 2278 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 4282 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2279 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 4289 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2280 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 4296 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2281 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4303 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2282 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4310 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2283 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4317 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2284 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4324 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2285 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4331 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2286 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4338 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2288 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4345 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2289 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4352 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2290 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4359 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2292 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to Ind.csv File...\n");return GENERATE_CSV_IND_FILE;
#line 4366 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2294 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Gen to Gen.txt File...\n");return GENERATE_TXT_GEN_FILE;
#line 4373 "EaseaLex.cpp"
		}
		break;
	case 422:
		{
#line 2296 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4380 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2297 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4387 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2299 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4394 "EaseaLex.cpp"
		}
		break;
	case 425:
		{
#line 2300 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4401 "EaseaLex.cpp"
		}
		break;
	case 426:
		{
#line 2302 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4412 "EaseaLex.cpp"
		}
		break;
	case 427:
		{
#line 2307 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4419 "EaseaLex.cpp"
		}
		break;
	case 428:
		{
#line 2309 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4430 "EaseaLex.cpp"
		}
		break;
	case 429:
		{
#line 2314 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4437 "EaseaLex.cpp"
		}
		break;
	case 430:
		{
#line 2317 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4444 "EaseaLex.cpp"
		}
		break;
	case 431:
		{
#line 2318 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4451 "EaseaLex.cpp"
		}
		break;
	case 432:
		{
#line 2319 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4458 "EaseaLex.cpp"
		}
		break;
	case 433:
		{
#line 2320 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4465 "EaseaLex.cpp"
		}
		break;
	case 434:
		{
#line 2321 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4472 "EaseaLex.cpp"
		}
		break;
#line 2322 "EaseaLex.l"
 
#line 4477 "EaseaLex.cpp"
	case 435:
		{
#line 2323 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4482 "EaseaLex.cpp"
		}
		break;
	case 436:
		{
#line 2324 "EaseaLex.l"
if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;
#line 4489 "EaseaLex.cpp"
		}
		break;
	case 437:
		{
#line 2325 "EaseaLex.l"
if(bVERBOSE) printf("\tExperiment Id...\n"); return EXPID;
#line 4496 "EaseaLex.cpp"
		}
		break;
	case 438:
		{
#line 2326 "EaseaLex.l"
if(bVERBOSE) printf("\tGrid Working Path...\n"); return WORKING_PATH;
#line 4503 "EaseaLex.cpp"
		}
		break;
	case 439:
		{
#line 2327 "EaseaLex.l"
if(bVERBOSE) printf("\tMigration Probability...\n"); return MIGRATION_PROBABILITY;
#line 4510 "EaseaLex.cpp"
		}
		break;
	case 440:
		{
#line 2328 "EaseaLex.l"
if(bVERBOSE) printf("\tServer port...\n"); return SERVER_PORT;
#line 4517 "EaseaLex.cpp"
		}
		break;
#line 2330 "EaseaLex.l"
 
#line 4522 "EaseaLex.cpp"
	case 441:
	case 442:
	case 443:
		{
#line 2334 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4529 "EaseaLex.cpp"
		}
		break;
	case 444:
		{
#line 2335 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4536 "EaseaLex.cpp"
		}
		break;
	case 445:
		{
#line 2338 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4544 "EaseaLex.cpp"
		}
		break;
	case 446:
		{
#line 2341 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return PATH_IDENTIFIER; 
#line 4552 "EaseaLex.cpp"
		}
		break;
	case 447:
		{
#line 2346 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4559 "EaseaLex.cpp"
		}
		break;
	case 448:
		{
#line 2348 "EaseaLex.l"

  lineCounter++;

#line 4568 "EaseaLex.cpp"
		}
		break;
	case 449:
		{
#line 2351 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4578 "EaseaLex.cpp"
		}
		break;
	case 450:
		{
#line 2356 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4588 "EaseaLex.cpp"
		}
		break;
	case 451:
		{
#line 2361 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4598 "EaseaLex.cpp"
		}
		break;
	case 452:
		{
#line 2366 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4608 "EaseaLex.cpp"
		}
		break;
	case 453:
		{
#line 2371 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4618 "EaseaLex.cpp"
		}
		break;
	case 454:
		{
#line 2376 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4628 "EaseaLex.cpp"
		}
		break;
	case 455:
		{
#line 2385 "EaseaLex.l"
return  (char)yytext[0];
#line 4635 "EaseaLex.cpp"
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
#line 2387 "EaseaLex.l"


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

#line 4835 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
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
		205,
		-206,
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
		217,
		-218,
		0,
		209,
		-210,
		0,
		207,
		-208,
		0,
		-249,
		0,
		-255,
		0,
		377,
		-378,
		0,
		322,
		-323,
		0,
		375,
		-376,
		0,
		320,
		-321,
		0,
		369,
		-370,
		0,
		371,
		-372,
		0,
		373,
		-374,
		0,
		367,
		-368,
		0,
		316,
		-317,
		0,
		318,
		-319,
		0,
		312,
		-313,
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
		330,
		-331,
		0,
		314,
		-315,
		0
	};
	yymatch = match;

	yytransitionmax = 5483;
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
		{ 3204, 63 },
		{ 3204, 63 },
		{ 1953, 2056 },
		{ 1578, 1561 },
		{ 1579, 1561 },
		{ 2475, 2444 },
		{ 2475, 2444 },
		{ 1953, 1934 },
		{ 0, 89 },
		{ 2450, 2415 },
		{ 2450, 2415 },
		{ 73, 3 },
		{ 74, 3 },
		{ 2312, 45 },
		{ 2313, 45 },
		{ 71, 1 },
		{ 2963, 2965 },
		{ 0, 1892 },
		{ 69, 1 },
		{ 2278, 2274 },
		{ 0, 2083 },
		{ 2052, 2054 },
		{ 167, 163 },
		{ 3204, 63 },
		{ 1426, 1425 },
		{ 3202, 63 },
		{ 1578, 1561 },
		{ 3262, 3257 },
		{ 2475, 2444 },
		{ 1447, 1446 },
		{ 1615, 1599 },
		{ 1616, 1599 },
		{ 2450, 2415 },
		{ 2106, 2084 },
		{ 73, 3 },
		{ 3206, 63 },
		{ 2312, 45 },
		{ 88, 63 },
		{ 3201, 63 },
		{ 3203, 63 },
		{ 3203, 63 },
		{ 3203, 63 },
		{ 3203, 63 },
		{ 3203, 63 },
		{ 3203, 63 },
		{ 3203, 63 },
		{ 3203, 63 },
		{ 3203, 63 },
		{ 3203, 63 },
		{ 72, 3 },
		{ 3205, 63 },
		{ 2311, 45 },
		{ 1668, 1662 },
		{ 1615, 1599 },
		{ 2476, 2444 },
		{ 1580, 1561 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 1617, 1599 },
		{ 3199, 63 },
		{ 0, 2501 },
		{ 1670, 1664 },
		{ 3200, 63 },
		{ 1544, 1523 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3200, 63 },
		{ 3207, 63 },
		{ 3211, 3210 },
		{ 2453, 2418 },
		{ 2453, 2418 },
		{ 3210, 3210 },
		{ 2462, 2426 },
		{ 2462, 2426 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 2533, 2500 },
		{ 3210, 3210 },
		{ 2567, 2533 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 2453, 2418 },
		{ 1545, 1524 },
		{ 1546, 1525 },
		{ 2462, 2426 },
		{ 1547, 1526 },
		{ 1548, 1527 },
		{ 1549, 1528 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 1550, 1529 },
		{ 3210, 3210 },
		{ 1551, 1531 },
		{ 3210, 3210 },
		{ 1554, 1534 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 3210, 3210 },
		{ 2273, 42 },
		{ 1618, 1600 },
		{ 1619, 1600 },
		{ 1555, 1535 },
		{ 2060, 42 },
		{ 2541, 2509 },
		{ 2541, 2509 },
		{ 2473, 2442 },
		{ 2473, 2442 },
		{ 2483, 2451 },
		{ 2483, 2451 },
		{ 2075, 41 },
		{ 1556, 1536 },
		{ 1886, 39 },
		{ 2497, 2465 },
		{ 2497, 2465 },
		{ 1557, 1537 },
		{ 1558, 1538 },
		{ 1560, 1540 },
		{ 1561, 1541 },
		{ 1562, 1542 },
		{ 1563, 1543 },
		{ 1564, 1544 },
		{ 2273, 42 },
		{ 1618, 1600 },
		{ 2063, 42 },
		{ 1565, 1545 },
		{ 1566, 1546 },
		{ 2541, 2509 },
		{ 1567, 1547 },
		{ 2473, 2442 },
		{ 1568, 1548 },
		{ 2483, 2451 },
		{ 1569, 1549 },
		{ 2075, 41 },
		{ 1570, 1551 },
		{ 1886, 39 },
		{ 2497, 2465 },
		{ 2272, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2061, 41 },
		{ 2076, 42 },
		{ 1873, 39 },
		{ 1573, 1554 },
		{ 1620, 1600 },
		{ 2542, 2509 },
		{ 1574, 1555 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2062, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2070, 42 },
		{ 2068, 42 },
		{ 2081, 42 },
		{ 2069, 42 },
		{ 2081, 42 },
		{ 2072, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2071, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 1575, 1556 },
		{ 2064, 42 },
		{ 2066, 42 },
		{ 1576, 1558 },
		{ 2081, 42 },
		{ 1577, 1560 },
		{ 2081, 42 },
		{ 2079, 42 },
		{ 2067, 42 },
		{ 2081, 42 },
		{ 2080, 42 },
		{ 2073, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2078, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2065, 42 },
		{ 2081, 42 },
		{ 2077, 42 },
		{ 2081, 42 },
		{ 2074, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2081, 42 },
		{ 2957, 46 },
		{ 2958, 46 },
		{ 1621, 1601 },
		{ 1622, 1601 },
		{ 69, 46 },
		{ 2502, 2469 },
		{ 2502, 2469 },
		{ 1474, 1452 },
		{ 1581, 1562 },
		{ 1582, 1563 },
		{ 1583, 1564 },
		{ 2514, 2481 },
		{ 2514, 2481 },
		{ 2528, 2495 },
		{ 2528, 2495 },
		{ 1585, 1565 },
		{ 1587, 1566 },
		{ 1584, 1564 },
		{ 1588, 1567 },
		{ 1589, 1568 },
		{ 1590, 1569 },
		{ 1591, 1570 },
		{ 1586, 1565 },
		{ 2957, 46 },
		{ 1594, 1574 },
		{ 1621, 1601 },
		{ 2529, 2496 },
		{ 2529, 2496 },
		{ 2502, 2469 },
		{ 1595, 1575 },
		{ 1624, 1602 },
		{ 1625, 1602 },
		{ 1627, 1603 },
		{ 1628, 1603 },
		{ 2514, 2481 },
		{ 1596, 1576 },
		{ 2528, 2495 },
		{ 2328, 46 },
		{ 2956, 46 },
		{ 2327, 46 },
		{ 2327, 46 },
		{ 2327, 46 },
		{ 2327, 46 },
		{ 2327, 46 },
		{ 2327, 46 },
		{ 2327, 46 },
		{ 2327, 46 },
		{ 2327, 46 },
		{ 2327, 46 },
		{ 2529, 2496 },
		{ 1597, 1577 },
		{ 1599, 1581 },
		{ 1600, 1582 },
		{ 1624, 1602 },
		{ 1623, 1601 },
		{ 1627, 1603 },
		{ 2329, 46 },
		{ 2325, 46 },
		{ 2320, 46 },
		{ 2329, 46 },
		{ 2317, 46 },
		{ 2324, 46 },
		{ 2322, 46 },
		{ 2329, 46 },
		{ 2326, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2319, 46 },
		{ 2314, 46 },
		{ 2321, 46 },
		{ 2316, 46 },
		{ 2329, 46 },
		{ 2323, 46 },
		{ 2318, 46 },
		{ 2315, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 1626, 1602 },
		{ 2334, 46 },
		{ 1629, 1603 },
		{ 1601, 1583 },
		{ 2329, 46 },
		{ 1602, 1584 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2330, 46 },
		{ 2331, 46 },
		{ 2332, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2333, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 2329, 46 },
		{ 161, 4 },
		{ 162, 4 },
		{ 1630, 1604 },
		{ 1631, 1604 },
		{ 2578, 2546 },
		{ 2578, 2546 },
		{ 2584, 2552 },
		{ 2584, 2552 },
		{ 1603, 1585 },
		{ 1604, 1586 },
		{ 1605, 1587 },
		{ 2591, 2559 },
		{ 2591, 2559 },
		{ 2370, 2338 },
		{ 2370, 2338 },
		{ 1606, 1588 },
		{ 1607, 1589 },
		{ 1608, 1590 },
		{ 1609, 1591 },
		{ 1611, 1594 },
		{ 1612, 1595 },
		{ 1613, 1596 },
		{ 1614, 1597 },
		{ 161, 4 },
		{ 1477, 1453 },
		{ 1630, 1604 },
		{ 1478, 1454 },
		{ 2578, 2546 },
		{ 1482, 1456 },
		{ 2584, 2552 },
		{ 1648, 1634 },
		{ 1649, 1634 },
		{ 1657, 1647 },
		{ 1658, 1647 },
		{ 2591, 2559 },
		{ 1483, 1457 },
		{ 2370, 2338 },
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
		{ 1484, 1458 },
		{ 1633, 1605 },
		{ 1634, 1606 },
		{ 1635, 1607 },
		{ 1648, 1634 },
		{ 1632, 1604 },
		{ 1657, 1647 },
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
		{ 1650, 1634 },
		{ 83, 4 },
		{ 1659, 1647 },
		{ 1637, 1611 },
		{ 87, 4 },
		{ 1638, 1612 },
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
		{ 1462, 23 },
		{ 2605, 2575 },
		{ 2605, 2575 },
		{ 1639, 1613 },
		{ 1449, 23 },
		{ 2607, 2577 },
		{ 2607, 2577 },
		{ 2612, 2582 },
		{ 2612, 2582 },
		{ 2615, 2585 },
		{ 2615, 2585 },
		{ 2619, 2589 },
		{ 2619, 2589 },
		{ 2620, 2590 },
		{ 2620, 2590 },
		{ 2637, 2602 },
		{ 2637, 2602 },
		{ 1640, 1614 },
		{ 1647, 1633 },
		{ 1485, 1459 },
		{ 1651, 1635 },
		{ 1653, 1637 },
		{ 1654, 1638 },
		{ 1462, 23 },
		{ 2605, 2575 },
		{ 1450, 23 },
		{ 1463, 23 },
		{ 1655, 1639 },
		{ 2607, 2577 },
		{ 1481, 1455 },
		{ 2612, 2582 },
		{ 1656, 1640 },
		{ 2615, 2585 },
		{ 1662, 1653 },
		{ 2619, 2589 },
		{ 1663, 1654 },
		{ 2620, 2590 },
		{ 1480, 1455 },
		{ 2637, 2602 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1479, 1455 },
		{ 1487, 1460 },
		{ 1664, 1655 },
		{ 1665, 1656 },
		{ 1486, 1460 },
		{ 1490, 1465 },
		{ 1669, 1663 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1466, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1455, 23 },
		{ 1453, 23 },
		{ 1468, 23 },
		{ 1454, 23 },
		{ 1468, 23 },
		{ 1457, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1456, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1491, 1466 },
		{ 1451, 23 },
		{ 1464, 23 },
		{ 1671, 1665 },
		{ 1458, 23 },
		{ 1674, 1669 },
		{ 1468, 23 },
		{ 1469, 23 },
		{ 1452, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1459, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1467, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1470, 23 },
		{ 1468, 23 },
		{ 1465, 23 },
		{ 1468, 23 },
		{ 1460, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 1468, 23 },
		{ 2047, 40 },
		{ 2396, 2361 },
		{ 2396, 2361 },
		{ 1675, 1671 },
		{ 1872, 40 },
		{ 2641, 2606 },
		{ 2641, 2606 },
		{ 2421, 2386 },
		{ 2421, 2386 },
		{ 2422, 2387 },
		{ 2422, 2387 },
		{ 2440, 2406 },
		{ 2440, 2406 },
		{ 2443, 2409 },
		{ 2443, 2409 },
		{ 1677, 1674 },
		{ 1678, 1675 },
		{ 1679, 1677 },
		{ 1680, 1678 },
		{ 1475, 1679 },
		{ 1492, 1467 },
		{ 1493, 1469 },
		{ 1494, 1470 },
		{ 2047, 40 },
		{ 2396, 2361 },
		{ 1877, 40 },
		{ 1497, 1474 },
		{ 1498, 1477 },
		{ 2641, 2606 },
		{ 1499, 1478 },
		{ 2421, 2386 },
		{ 1500, 1479 },
		{ 2422, 2387 },
		{ 1501, 1480 },
		{ 2440, 2406 },
		{ 1502, 1481 },
		{ 2443, 2409 },
		{ 1503, 1482 },
		{ 2046, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1504, 1483 },
		{ 1887, 40 },
		{ 1505, 1484 },
		{ 1507, 1485 },
		{ 1508, 1486 },
		{ 0, 2606 },
		{ 1506, 1484 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1874, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1882, 40 },
		{ 1880, 40 },
		{ 1890, 40 },
		{ 1881, 40 },
		{ 1890, 40 },
		{ 1884, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1883, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1509, 1487 },
		{ 1878, 40 },
		{ 1512, 1490 },
		{ 1513, 1491 },
		{ 1890, 40 },
		{ 1514, 1492 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1879, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1875, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1876, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1889, 40 },
		{ 1890, 40 },
		{ 1888, 40 },
		{ 1890, 40 },
		{ 1885, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1890, 40 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1515, 1493 },
		{ 1516, 1494 },
		{ 1519, 1497 },
		{ 1520, 1498 },
		{ 1521, 1499 },
		{ 1522, 1500 },
		{ 1523, 1501 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1524, 1502 },
		{ 1525, 1503 },
		{ 1526, 1504 },
		{ 1527, 1505 },
		{ 1475, 1681 },
		{ 1528, 1506 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 1475, 1681 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 1529, 1507 },
		{ 1530, 1508 },
		{ 1531, 1509 },
		{ 1534, 1512 },
		{ 1535, 1513 },
		{ 1536, 1514 },
		{ 1537, 1515 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 1538, 1516 },
		{ 1540, 1519 },
		{ 1541, 1520 },
		{ 1542, 1521 },
		{ 2501, 2567 },
		{ 1543, 1522 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2501, 2567 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2436, 2401 },
		{ 156, 154 },
		{ 116, 101 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 108 },
		{ 125, 109 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 126, 111 },
		{ 127, 112 },
		{ 128, 113 },
		{ 129, 114 },
		{ 2329, 2631 },
		{ 131, 116 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 2329, 2631 },
		{ 1476, 1680 },
		{ 0, 1680 },
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
		{ 2337, 2314 },
		{ 2845, 2845 },
		{ 2339, 2315 },
		{ 2342, 2316 },
		{ 2343, 2317 },
		{ 2352, 2319 },
		{ 2340, 2316 },
		{ 2347, 2318 },
		{ 2354, 2320 },
		{ 2341, 2316 },
		{ 1476, 1680 },
		{ 2346, 2318 },
		{ 2355, 2321 },
		{ 2358, 2323 },
		{ 2344, 2317 },
		{ 2356, 2322 },
		{ 2345, 2317 },
		{ 2351, 2319 },
		{ 2359, 2324 },
		{ 2360, 2325 },
		{ 2361, 2326 },
		{ 2329, 2329 },
		{ 2365, 2330 },
		{ 2353, 2331 },
		{ 2845, 2845 },
		{ 2338, 2332 },
		{ 2348, 2318 },
		{ 2349, 2318 },
		{ 2357, 2322 },
		{ 2350, 2333 },
		{ 2369, 2337 },
		{ 2366, 2331 },
		{ 147, 139 },
		{ 2371, 2339 },
		{ 2372, 2340 },
		{ 2373, 2341 },
		{ 2374, 2342 },
		{ 2375, 2343 },
		{ 2376, 2344 },
		{ 0, 1680 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2377, 2345 },
		{ 2380, 2347 },
		{ 2381, 2348 },
		{ 2382, 2349 },
		{ 2383, 2350 },
		{ 2384, 2351 },
		{ 2385, 2352 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 69, 7 },
		{ 2387, 2353 },
		{ 2388, 2354 },
		{ 2389, 2355 },
		{ 2845, 2845 },
		{ 1681, 1680 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2390, 2356 },
		{ 2391, 2357 },
		{ 2394, 2359 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 2378, 2346 },
		{ 2395, 2360 },
		{ 148, 140 },
		{ 2401, 2365 },
		{ 2386, 2366 },
		{ 2404, 2369 },
		{ 2379, 2346 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 2406, 2371 },
		{ 2407, 2372 },
		{ 2408, 2373 },
		{ 2409, 2374 },
		{ 1289, 7 },
		{ 2410, 2375 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 1289, 7 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 2411, 2376 },
		{ 2412, 2377 },
		{ 2413, 2378 },
		{ 2414, 2379 },
		{ 2415, 2380 },
		{ 2416, 2381 },
		{ 2417, 2382 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 2418, 2383 },
		{ 2419, 2384 },
		{ 2420, 2385 },
		{ 149, 142 },
		{ 0, 1952 },
		{ 2423, 2388 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 1952 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 2424, 2389 },
		{ 2425, 2390 },
		{ 2426, 2391 },
		{ 2427, 2392 },
		{ 2428, 2393 },
		{ 2429, 2394 },
		{ 2430, 2395 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 150, 143 },
		{ 2438, 2404 },
		{ 151, 144 },
		{ 2441, 2407 },
		{ 0, 2146 },
		{ 2442, 2408 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 0, 2146 },
		{ 2392, 2358 },
		{ 2444, 2410 },
		{ 2446, 2411 },
		{ 2447, 2412 },
		{ 2448, 2413 },
		{ 2445, 2410 },
		{ 2449, 2414 },
		{ 152, 146 },
		{ 2451, 2416 },
		{ 2393, 2358 },
		{ 2452, 2417 },
		{ 2454, 2419 },
		{ 2455, 2420 },
		{ 2459, 2423 },
		{ 2460, 2424 },
		{ 2461, 2425 },
		{ 2463, 2427 },
		{ 2464, 2428 },
		{ 2465, 2429 },
		{ 2466, 2430 },
		{ 2469, 2438 },
		{ 2472, 2441 },
		{ 153, 149 },
		{ 154, 150 },
		{ 2477, 2445 },
		{ 2478, 2446 },
		{ 2479, 2447 },
		{ 2480, 2448 },
		{ 2481, 2449 },
		{ 2484, 2452 },
		{ 2486, 2454 },
		{ 2487, 2455 },
		{ 2491, 2459 },
		{ 2492, 2460 },
		{ 2493, 2461 },
		{ 2495, 2463 },
		{ 2496, 2464 },
		{ 155, 152 },
		{ 2498, 2466 },
		{ 2506, 2472 },
		{ 2509, 2477 },
		{ 2510, 2478 },
		{ 2511, 2479 },
		{ 2513, 2480 },
		{ 91, 75 },
		{ 2517, 2484 },
		{ 2519, 2486 },
		{ 2512, 2480 },
		{ 2520, 2487 },
		{ 2524, 2491 },
		{ 2525, 2492 },
		{ 2526, 2493 },
		{ 157, 155 },
		{ 158, 157 },
		{ 2531, 2498 },
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
		{ 2538, 2506 },
		{ 2543, 2510 },
		{ 2544, 2511 },
		{ 2545, 2512 },
		{ 2546, 2513 },
		{ 2550, 2517 },
		{ 2552, 2519 },
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
		{ 2553, 2520 },
		{ 2557, 2524 },
		{ 2558, 2525 },
		{ 2559, 2526 },
		{ 87, 159 },
		{ 2565, 2531 },
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
		{ 2572, 2538 },
		{ 2575, 2543 },
		{ 2576, 2544 },
		{ 2577, 2545 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 2582, 2550 },
		{ 95, 78 },
		{ 2585, 2553 },
		{ 2589, 2557 },
		{ 2590, 2558 },
		{ 93, 76 },
		{ 96, 79 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 2597, 2565 },
		{ 97, 80 },
		{ 2602, 2572 },
		{ 98, 81 },
		{ 1289, 1289 },
		{ 2606, 2576 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 99, 82 },
		{ 101, 84 },
		{ 106, 91 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 107, 92 },
		{ 108, 93 },
		{ 109, 94 },
		{ 110, 95 },
		{ 111, 96 },
		{ 112, 97 },
		{ 113, 98 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 114, 99 },
		{ 1304, 1301 },
		{ 3033, 51 },
		{ 1304, 1301 },
		{ 0, 1539 },
		{ 0, 3034 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 1539 },
		{ 0, 2597 },
		{ 0, 2597 },
		{ 134, 120 },
		{ 118, 103 },
		{ 134, 120 },
		{ 118, 103 },
		{ 2735, 2705 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 2398, 2363 },
		{ 1299, 1296 },
		{ 2398, 2363 },
		{ 1299, 1296 },
		{ 2721, 2691 },
		{ 2906, 2889 },
		{ 0, 2597 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 2432, 2397 },
		{ 2434, 2400 },
		{ 2432, 2397 },
		{ 2434, 2400 },
		{ 3200, 3200 },
		{ 2909, 2892 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 3200, 3200 },
		{ 88, 51 },
		{ 1354, 1353 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 132, 117 },
		{ 1302, 1298 },
		{ 132, 117 },
		{ 1302, 1298 },
		{ 1828, 1827 },
		{ 2252, 2249 },
		{ 2631, 2597 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 1870, 1869 },
		{ 2888, 2870 },
		{ 3265, 3260 },
		{ 2783, 2754 },
		{ 3180, 3179 },
		{ 1735, 1734 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3260, 3260 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 1780, 1779 },
		{ 2692, 2657 },
		{ 1825, 1824 },
		{ 2809, 2782 },
		{ 1351, 1350 },
		{ 3277, 3276 },
		{ 3066, 3065 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 2580, 2548 },
		{ 2119, 2098 },
		{ 3270, 3267 },
		{ 3119, 3118 },
		{ 2148, 2130 },
		{ 2162, 2145 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3267, 3267 },
		{ 3266, 3261 },
		{ 3266, 3261 },
		{ 3266, 3261 },
		{ 3266, 3261 },
		{ 3266, 3261 },
		{ 3266, 3261 },
		{ 3266, 3261 },
		{ 3266, 3261 },
		{ 3266, 3261 },
		{ 3266, 3261 },
		{ 3259, 3255 },
		{ 3259, 3255 },
		{ 3259, 3255 },
		{ 3259, 3255 },
		{ 3259, 3255 },
		{ 3259, 3255 },
		{ 3259, 3255 },
		{ 3259, 3255 },
		{ 3259, 3255 },
		{ 3259, 3255 },
		{ 0, 2468 },
		{ 3160, 3159 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 0, 2468 },
		{ 2104, 2080 },
		{ 184, 175 },
		{ 195, 175 },
		{ 186, 175 },
		{ 1709, 1708 },
		{ 181, 175 },
		{ 191, 175 },
		{ 185, 175 },
		{ 1928, 1909 },
		{ 183, 175 },
		{ 1414, 1413 },
		{ 1709, 1708 },
		{ 0, 3261 },
		{ 193, 175 },
		{ 192, 175 },
		{ 182, 175 },
		{ 190, 175 },
		{ 1414, 1413 },
		{ 187, 175 },
		{ 189, 175 },
		{ 180, 175 },
		{ 2103, 2080 },
		{ 0, 3255 },
		{ 188, 175 },
		{ 194, 175 },
		{ 102, 85 },
		{ 1903, 1879 },
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
		{ 3183, 3182 },
		{ 3191, 3190 },
		{ 1902, 1879 },
		{ 1950, 1931 },
		{ 1841, 1840 },
		{ 2292, 2291 },
		{ 2695, 2660 },
		{ 2297, 2296 },
		{ 2980, 2979 },
		{ 2630, 2596 },
		{ 3249, 3244 },
		{ 103, 85 },
		{ 1295, 1292 },
		{ 2500, 2468 },
		{ 1292, 1292 },
		{ 1292, 1292 },
		{ 1292, 1292 },
		{ 1292, 1292 },
		{ 1292, 1292 },
		{ 1292, 1292 },
		{ 1292, 1292 },
		{ 1292, 1292 },
		{ 1292, 1292 },
		{ 1292, 1292 },
		{ 3259, 3259 },
		{ 3259, 3259 },
		{ 3259, 3259 },
		{ 3259, 3259 },
		{ 3259, 3259 },
		{ 3259, 3259 },
		{ 3259, 3259 },
		{ 3259, 3259 },
		{ 3259, 3259 },
		{ 3259, 3259 },
		{ 3020, 3019 },
		{ 1296, 1292 },
		{ 3025, 3024 },
		{ 1783, 1782 },
		{ 1756, 1755 },
		{ 2089, 2067 },
		{ 2773, 2744 },
		{ 1367, 1366 },
		{ 2362, 2327 },
		{ 103, 85 },
		{ 2327, 2327 },
		{ 2327, 2327 },
		{ 2327, 2327 },
		{ 2327, 2327 },
		{ 2327, 2327 },
		{ 2327, 2327 },
		{ 2327, 2327 },
		{ 2327, 2327 },
		{ 2327, 2327 },
		{ 2327, 2327 },
		{ 1295, 1295 },
		{ 1295, 1295 },
		{ 1295, 1295 },
		{ 1295, 1295 },
		{ 1295, 1295 },
		{ 1295, 1295 },
		{ 1295, 1295 },
		{ 1295, 1295 },
		{ 1295, 1295 },
		{ 1295, 1295 },
		{ 3280, 3279 },
		{ 2363, 2327 },
		{ 3296, 3293 },
		{ 1296, 1292 },
		{ 3302, 3299 },
		{ 3264, 3259 },
		{ 2799, 2771 },
		{ 1931, 1912 },
		{ 2810, 2783 },
		{ 2811, 2784 },
		{ 2813, 2786 },
		{ 1298, 1295 },
		{ 2306, 2305 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2362, 2362 },
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
		{ 2363, 2327 },
		{ 2397, 2362 },
		{ 2818, 2791 },
		{ 2831, 2808 },
		{ 2835, 2812 },
		{ 3219, 65 },
		{ 2851, 2828 },
		{ 2852, 2829 },
		{ 69, 65 },
		{ 2601, 2571 },
		{ 1298, 1295 },
		{ 2400, 2364 },
		{ 1937, 1918 },
		{ 1297, 1297 },
		{ 1297, 1297 },
		{ 1297, 1297 },
		{ 1297, 1297 },
		{ 1297, 1297 },
		{ 1297, 1297 },
		{ 1297, 1297 },
		{ 1297, 1297 },
		{ 1297, 1297 },
		{ 1297, 1297 },
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
		{ 2397, 2362 },
		{ 1301, 1297 },
		{ 2422, 2422 },
		{ 2422, 2422 },
		{ 2603, 2573 },
		{ 1291, 9 },
		{ 2865, 2842 },
		{ 2661, 2661 },
		{ 2661, 2661 },
		{ 69, 9 },
		{ 2400, 2364 },
		{ 120, 104 },
		{ 2539, 2507 },
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
		{ 2870, 2849 },
		{ 2876, 2855 },
		{ 2422, 2422 },
		{ 3234, 67 },
		{ 2882, 2863 },
		{ 1291, 9 },
		{ 69, 67 },
		{ 2661, 2661 },
		{ 3218, 65 },
		{ 1370, 1369 },
		{ 1301, 1297 },
		{ 117, 102 },
		{ 3217, 65 },
		{ 2889, 2871 },
		{ 2892, 2874 },
		{ 1957, 1938 },
		{ 1409, 1408 },
		{ 2911, 2894 },
		{ 2914, 2897 },
		{ 1293, 9 },
		{ 120, 104 },
		{ 1292, 9 },
		{ 1292, 9 },
		{ 1292, 9 },
		{ 1292, 9 },
		{ 1292, 9 },
		{ 1292, 9 },
		{ 1292, 9 },
		{ 1292, 9 },
		{ 1292, 9 },
		{ 1292, 9 },
		{ 3266, 3266 },
		{ 3266, 3266 },
		{ 3266, 3266 },
		{ 3266, 3266 },
		{ 3266, 3266 },
		{ 3266, 3266 },
		{ 3266, 3266 },
		{ 3266, 3266 },
		{ 3266, 3266 },
		{ 3266, 3266 },
		{ 2931, 2924 },
		{ 2933, 2926 },
		{ 117, 102 },
		{ 3227, 3227 },
		{ 3227, 3227 },
		{ 3227, 3227 },
		{ 3227, 3227 },
		{ 3227, 3227 },
		{ 3227, 3227 },
		{ 3227, 3227 },
		{ 3227, 3227 },
		{ 3227, 3227 },
		{ 3227, 3227 },
		{ 478, 430 },
		{ 483, 430 },
		{ 480, 430 },
		{ 479, 430 },
		{ 482, 430 },
		{ 477, 430 },
		{ 2940, 2934 },
		{ 476, 430 },
		{ 3215, 65 },
		{ 2950, 2948 },
		{ 3216, 65 },
		{ 481, 430 },
		{ 3231, 67 },
		{ 484, 430 },
		{ 2591, 2591 },
		{ 2591, 2591 },
		{ 2612, 2612 },
		{ 2612, 2612 },
		{ 3232, 67 },
		{ 475, 430 },
		{ 1984, 1968 },
		{ 2457, 2422 },
		{ 3269, 3266 },
		{ 2528, 2528 },
		{ 2528, 2528 },
		{ 2586, 2586 },
		{ 2586, 2586 },
		{ 2587, 2587 },
		{ 2587, 2587 },
		{ 3084, 3084 },
		{ 3084, 3084 },
		{ 3229, 67 },
		{ 2458, 2422 },
		{ 3131, 3131 },
		{ 3131, 3131 },
		{ 3228, 3227 },
		{ 2696, 2661 },
		{ 2591, 2591 },
		{ 1986, 1971 },
		{ 2612, 2612 },
		{ 2983, 2982 },
		{ 2598, 2598 },
		{ 2598, 2598 },
		{ 2722, 2722 },
		{ 2722, 2722 },
		{ 2992, 2991 },
		{ 2528, 2528 },
		{ 3005, 3004 },
		{ 2586, 2586 },
		{ 1988, 1973 },
		{ 2587, 2587 },
		{ 2038, 2036 },
		{ 3084, 3084 },
		{ 3233, 67 },
		{ 2832, 2832 },
		{ 2832, 2832 },
		{ 3131, 3131 },
		{ 2834, 2834 },
		{ 2834, 2834 },
		{ 2473, 2473 },
		{ 2473, 2473 },
		{ 2620, 2620 },
		{ 2620, 2620 },
		{ 1785, 1784 },
		{ 2598, 2598 },
		{ 3028, 3027 },
		{ 2722, 2722 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2637, 2637 },
		{ 2637, 2637 },
		{ 2497, 2497 },
		{ 2497, 2497 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 1807, 1806 },
		{ 2623, 2591 },
		{ 2832, 2832 },
		{ 2622, 2591 },
		{ 2554, 2521 },
		{ 2834, 2834 },
		{ 2555, 2522 },
		{ 2473, 2473 },
		{ 1820, 1819 },
		{ 2620, 2620 },
		{ 1355, 1354 },
		{ 3056, 3055 },
		{ 2625, 2591 },
		{ 3044, 3044 },
		{ 3044, 3044 },
		{ 2443, 2443 },
		{ 2508, 2474 },
		{ 2637, 2637 },
		{ 3083, 3082 },
		{ 2497, 2497 },
		{ 1704, 1703 },
		{ 3016, 3016 },
		{ 2551, 2551 },
		{ 2551, 2551 },
		{ 2514, 2514 },
		{ 2514, 2514 },
		{ 2450, 2450 },
		{ 2450, 2450 },
		{ 1829, 1828 },
		{ 2621, 2591 },
		{ 2615, 2615 },
		{ 2615, 2615 },
		{ 3113, 3112 },
		{ 2624, 2591 },
		{ 2644, 2609 },
		{ 3122, 3121 },
		{ 3044, 3044 },
		{ 3130, 3129 },
		{ 2160, 2143 },
		{ 2619, 2619 },
		{ 2619, 2619 },
		{ 2584, 2584 },
		{ 2584, 2584 },
		{ 2648, 2612 },
		{ 2647, 2612 },
		{ 2551, 2551 },
		{ 2570, 2536 },
		{ 2514, 2514 },
		{ 3154, 3153 },
		{ 2450, 2450 },
		{ 2562, 2528 },
		{ 2561, 2528 },
		{ 2161, 2144 },
		{ 2615, 2615 },
		{ 2299, 2299 },
		{ 2299, 2299 },
		{ 3163, 3162 },
		{ 2616, 2586 },
		{ 3174, 3173 },
		{ 2617, 2587 },
		{ 2632, 2598 },
		{ 3085, 3084 },
		{ 2619, 2619 },
		{ 2574, 2540 },
		{ 2584, 2584 },
		{ 3132, 3131 },
		{ 2838, 2838 },
		{ 2838, 2838 },
		{ 2633, 2598 },
		{ 1432, 1431 },
		{ 2370, 2370 },
		{ 2370, 2370 },
		{ 2607, 2607 },
		{ 2607, 2607 },
		{ 2752, 2722 },
		{ 3185, 3184 },
		{ 1844, 1843 },
		{ 2299, 2299 },
		{ 3194, 3193 },
		{ 3053, 3053 },
		{ 3053, 3053 },
		{ 3045, 3044 },
		{ 2731, 2731 },
		{ 2731, 2731 },
		{ 2680, 2645 },
		{ 2855, 2832 },
		{ 3098, 3098 },
		{ 3098, 3098 },
		{ 2857, 2834 },
		{ 2838, 2838 },
		{ 2507, 2473 },
		{ 2179, 2166 },
		{ 2656, 2620 },
		{ 2370, 2370 },
		{ 2192, 2177 },
		{ 2607, 2607 },
		{ 2453, 2453 },
		{ 2453, 2453 },
		{ 2474, 2443 },
		{ 2193, 2178 },
		{ 2673, 2637 },
		{ 2530, 2497 },
		{ 3053, 3053 },
		{ 3017, 3016 },
		{ 2703, 2670 },
		{ 2731, 2731 },
		{ 2719, 2689 },
		{ 3139, 3139 },
		{ 3139, 3139 },
		{ 3098, 3098 },
		{ 1443, 1442 },
		{ 2787, 2787 },
		{ 2787, 2787 },
		{ 3045, 3044 },
		{ 2502, 2502 },
		{ 2502, 2502 },
		{ 3244, 3239 },
		{ 2760, 2760 },
		{ 2760, 2760 },
		{ 2453, 2453 },
		{ 2728, 2698 },
		{ 2253, 2250 },
		{ 2739, 2709 },
		{ 2268, 2267 },
		{ 2754, 2724 },
		{ 2583, 2551 },
		{ 1321, 1320 },
		{ 2547, 2514 },
		{ 1757, 1756 },
		{ 2482, 2450 },
		{ 3139, 3139 },
		{ 2781, 2752 },
		{ 2294, 2293 },
		{ 2651, 2615 },
		{ 2787, 2787 },
		{ 3282, 3281 },
		{ 3284, 3284 },
		{ 2502, 2502 },
		{ 1759, 1758 },
		{ 2795, 2767 },
		{ 2760, 2760 },
		{ 3309, 3307 },
		{ 2655, 2619 },
		{ 2664, 2628 },
		{ 2614, 2584 },
		{ 2099, 2074 },
		{ 2440, 2440 },
		{ 2440, 2440 },
		{ 2098, 2074 },
		{ 1910, 1885 },
		{ 1776, 1775 },
		{ 2672, 2635 },
		{ 1909, 1885 },
		{ 1999, 1986 },
		{ 2954, 2953 },
		{ 2018, 2008 },
		{ 2975, 2974 },
		{ 2300, 2299 },
		{ 2026, 2018 },
		{ 3284, 3284 },
		{ 1325, 1324 },
		{ 2987, 2986 },
		{ 1661, 1652 },
		{ 1378, 1377 },
		{ 2704, 2672 },
		{ 1801, 1800 },
		{ 1802, 1801 },
		{ 1385, 1384 },
		{ 2861, 2838 },
		{ 2440, 2440 },
		{ 1811, 1810 },
		{ 2120, 2099 },
		{ 2405, 2370 },
		{ 2743, 2713 },
		{ 2642, 2607 },
		{ 3039, 3037 },
		{ 2139, 2118 },
		{ 2141, 2120 },
		{ 2758, 2728 },
		{ 3060, 3059 },
		{ 2595, 2563 },
		{ 3054, 3053 },
		{ 2144, 2123 },
		{ 1386, 1385 },
		{ 2761, 2731 },
		{ 2782, 2753 },
		{ 2599, 2569 },
		{ 1388, 1387 },
		{ 3099, 3098 },
		{ 1667, 1661 },
		{ 1697, 1696 },
		{ 2604, 2574 },
		{ 1837, 1836 },
		{ 1698, 1697 },
		{ 1402, 1401 },
		{ 1860, 1859 },
		{ 3173, 3172 },
		{ 1861, 1860 },
		{ 2485, 2453 },
		{ 2819, 2792 },
		{ 2820, 2793 },
		{ 2822, 2796 },
		{ 2823, 2799 },
		{ 1866, 1865 },
		{ 1403, 1402 },
		{ 1725, 1724 },
		{ 1726, 1725 },
		{ 2854, 2831 },
		{ 1732, 1731 },
		{ 3140, 3139 },
		{ 1733, 1732 },
		{ 1929, 1910 },
		{ 1331, 1330 },
		{ 2814, 2787 },
		{ 2866, 2843 },
		{ 2309, 2308 },
		{ 2534, 2502 },
		{ 2789, 2760 },
		{ 1936, 1917 },
		{ 2638, 2603 },
		{ 1751, 1750 },
		{ 1948, 1929 },
		{ 1752, 1751 },
		{ 3272, 3271 },
		{ 3287, 3284 },
		{ 3273, 3272 },
		{ 1363, 1362 },
		{ 2901, 2883 },
		{ 1310, 1309 },
		{ 3286, 3284 },
		{ 1438, 1437 },
		{ 3285, 3284 },
		{ 1775, 1774 },
		{ 2657, 2621 },
		{ 2922, 2911 },
		{ 2762, 2732 },
		{ 1967, 1950 },
		{ 2560, 2527 },
		{ 1761, 1760 },
		{ 3003, 3002 },
		{ 1711, 1710 },
		{ 3014, 3013 },
		{ 2627, 2593 },
		{ 1434, 1433 },
		{ 2796, 2768 },
		{ 2471, 2440 },
		{ 1319, 1318 },
		{ 2800, 2772 },
		{ 2246, 2239 },
		{ 2249, 2244 },
		{ 1730, 1729 },
		{ 3043, 3041 },
		{ 1440, 1439 },
		{ 2265, 2262 },
		{ 1787, 1786 },
		{ 2033, 2028 },
		{ 2650, 2614 },
		{ 1353, 1352 },
		{ 2652, 2616 },
		{ 2653, 2617 },
		{ 3111, 3110 },
		{ 1312, 1311 },
		{ 2840, 2817 },
		{ 2844, 2821 },
		{ 2296, 2295 },
		{ 1737, 1736 },
		{ 1921, 1902 },
		{ 3152, 3151 },
		{ 2302, 2301 },
		{ 1380, 1379 },
		{ 2490, 2458 },
		{ 2308, 2307 },
		{ 2686, 2651 },
		{ 2690, 2655 },
		{ 2869, 2848 },
		{ 1813, 1812 },
		{ 2110, 2089 },
		{ 2879, 2860 },
		{ 1333, 1332 },
		{ 1932, 1913 },
		{ 2131, 2110 },
		{ 2712, 2682 },
		{ 2896, 2878 },
		{ 2549, 2516 },
		{ 1822, 1821 },
		{ 1416, 1415 },
		{ 2729, 2699 },
		{ 3253, 3249 },
		{ 1940, 1921 },
		{ 2915, 2898 },
		{ 2916, 2900 },
		{ 2738, 2708 },
		{ 2609, 2579 },
		{ 2932, 2925 },
		{ 2470, 2470 },
		{ 2470, 2470 },
		{ 2613, 2583 },
		{ 2942, 2939 },
		{ 2946, 2943 },
		{ 1827, 1826 },
		{ 3284, 3283 },
		{ 2756, 2726 },
		{ 3292, 3289 },
		{ 1430, 1429 },
		{ 3300, 3297 },
		{ 1347, 1346 },
		{ 2977, 2976 },
		{ 3312, 3311 },
		{ 2529, 2529 },
		{ 2529, 2529 },
		{ 2928, 2928 },
		{ 2928, 2928 },
		{ 2421, 2421 },
		{ 2421, 2421 },
		{ 2670, 2633 },
		{ 2790, 2761 },
		{ 2828, 2804 },
		{ 2470, 2470 },
		{ 2829, 2805 },
		{ 2880, 2861 },
		{ 2699, 2664 },
		{ 2548, 2515 },
		{ 2837, 2814 },
		{ 2130, 2109 },
		{ 3055, 3054 },
		{ 2677, 2642 },
		{ 1806, 1805 },
		{ 1867, 1866 },
		{ 2725, 2695 },
		{ 2812, 2785 },
		{ 2772, 2743 },
		{ 2529, 2529 },
		{ 2689, 2654 },
		{ 2928, 2928 },
		{ 1442, 1441 },
		{ 2421, 2421 },
		{ 1971, 1956 },
		{ 2924, 2913 },
		{ 1428, 1427 },
		{ 1899, 1876 },
		{ 2685, 2650 },
		{ 2259, 2256 },
		{ 2262, 2260 },
		{ 3030, 3029 },
		{ 1745, 1744 },
		{ 2035, 2032 },
		{ 1433, 1432 },
		{ 2041, 2040 },
		{ 1914, 1891 },
		{ 1898, 1876 },
		{ 1552, 1532 },
		{ 1372, 1371 },
		{ 1495, 1471 },
		{ 2716, 2686 },
		{ 3058, 3057 },
		{ 1719, 1718 },
		{ 2503, 2470 },
		{ 3065, 3064 },
		{ 2864, 2841 },
		{ 2720, 2690 },
		{ 2105, 2082 },
		{ 2109, 2088 },
		{ 1760, 1759 },
		{ 1933, 1914 },
		{ 3115, 3114 },
		{ 1935, 1916 },
		{ 2124, 2103 },
		{ 3124, 3123 },
		{ 2126, 2105 },
		{ 2740, 2710 },
		{ 2128, 2107 },
		{ 1427, 1426 },
		{ 1769, 1768 },
		{ 3156, 3155 },
		{ 1323, 1322 },
		{ 2900, 2882 },
		{ 3165, 3164 },
		{ 1517, 1495 },
		{ 2902, 2884 },
		{ 1731, 1730 },
		{ 1346, 1345 },
		{ 2768, 2739 },
		{ 3187, 3186 },
		{ 2159, 2142 },
		{ 1966, 1949 },
		{ 3196, 3195 },
		{ 1846, 1845 },
		{ 2504, 2470 },
		{ 1854, 1853 },
		{ 1396, 1395 },
		{ 2925, 2914 },
		{ 2784, 2755 },
		{ 2174, 2158 },
		{ 2649, 2613 },
		{ 1981, 1965 },
		{ 2939, 2933 },
		{ 3250, 3245 },
		{ 1786, 1785 },
		{ 2515, 2482 },
		{ 1734, 1733 },
		{ 2579, 2547 },
		{ 2952, 2951 },
		{ 2224, 2206 },
		{ 2581, 2549 },
		{ 2563, 2529 },
		{ 2225, 2207 },
		{ 2934, 2928 },
		{ 1795, 1794 },
		{ 2456, 2421 },
		{ 1691, 1690 },
		{ 2816, 2789 },
		{ 2985, 2984 },
		{ 3283, 3282 },
		{ 2251, 2248 },
		{ 2527, 2494 },
		{ 3289, 3286 },
		{ 2675, 2640 },
		{ 2821, 2795 },
		{ 3007, 3006 },
		{ 3013, 3012 },
		{ 1445, 1444 },
		{ 3311, 3309 },
		{ 1805, 1804 },
		{ 2998, 2998 },
		{ 2998, 2998 },
		{ 3147, 3147 },
		{ 3147, 3147 },
		{ 2605, 2605 },
		{ 2605, 2605 },
		{ 2578, 2578 },
		{ 2578, 2578 },
		{ 3106, 3106 },
		{ 3106, 3106 },
		{ 2238, 2225 },
		{ 1324, 1323 },
		{ 2248, 2243 },
		{ 1930, 1911 },
		{ 2042, 2041 },
		{ 1832, 1831 },
		{ 3059, 3058 },
		{ 1373, 1372 },
		{ 1395, 1394 },
		{ 2261, 2259 },
		{ 3067, 3066 },
		{ 3076, 3075 },
		{ 1842, 1841 },
		{ 2998, 2998 },
		{ 2488, 2456 },
		{ 3147, 3147 },
		{ 3093, 3092 },
		{ 2605, 2605 },
		{ 3094, 3093 },
		{ 2578, 2578 },
		{ 3096, 3095 },
		{ 3106, 3106 },
		{ 2883, 2864 },
		{ 2489, 2457 },
		{ 3109, 3108 },
		{ 1744, 1743 },
		{ 1794, 1793 },
		{ 2107, 2085 },
		{ 3116, 3115 },
		{ 2108, 2087 },
		{ 3120, 3119 },
		{ 1938, 1919 },
		{ 1847, 1846 },
		{ 3125, 3124 },
		{ 1853, 1852 },
		{ 2298, 2297 },
		{ 3137, 3136 },
		{ 2769, 2740 },
		{ 2771, 2742 },
		{ 3150, 3149 },
		{ 1702, 1701 },
		{ 2121, 2100 },
		{ 2918, 2902 },
		{ 3157, 3156 },
		{ 1425, 1424 },
		{ 3161, 3160 },
		{ 1572, 1553 },
		{ 115, 100 },
		{ 3166, 3165 },
		{ 3172, 3171 },
		{ 1968, 1951 },
		{ 2788, 2759 },
		{ 1718, 1717 },
		{ 1368, 1367 },
		{ 1973, 1958 },
		{ 2143, 2122 },
		{ 3188, 3187 },
		{ 2402, 2367 },
		{ 3192, 3191 },
		{ 2658, 2622 },
		{ 2660, 2624 },
		{ 3197, 3196 },
		{ 2662, 2626 },
		{ 2953, 2952 },
		{ 1358, 1357 },
		{ 3212, 3208 },
		{ 1983, 1967 },
		{ 1496, 1473 },
		{ 1768, 1767 },
		{ 1446, 1445 },
		{ 3241, 3236 },
		{ 2981, 2980 },
		{ 3245, 3240 },
		{ 1407, 1406 },
		{ 2007, 1995 },
		{ 2986, 2985 },
		{ 3257, 3253 },
		{ 2678, 2643 },
		{ 2166, 2149 },
		{ 1916, 1894 },
		{ 3001, 3000 },
		{ 2020, 2011 },
		{ 1920, 1901 },
		{ 2600, 2570 },
		{ 2999, 2998 },
		{ 3008, 3007 },
		{ 3148, 3147 },
		{ 2032, 2027 },
		{ 2640, 2605 },
		{ 2694, 2659 },
		{ 2608, 2578 },
		{ 2204, 2190 },
		{ 3107, 3106 },
		{ 1532, 1510 },
		{ 2698, 2663 },
		{ 3026, 3025 },
		{ 2847, 2824 },
		{ 1690, 1689 },
		{ 3031, 3030 },
		{ 2700, 2665 },
		{ 2702, 2669 },
		{ 2226, 2208 },
		{ 2237, 2224 },
		{ 3040, 3038 },
		{ 2833, 2833 },
		{ 2833, 2833 },
		{ 2646, 2646 },
		{ 2646, 2646 },
		{ 3117, 3117 },
		{ 3117, 3117 },
		{ 2396, 2396 },
		{ 2396, 2396 },
		{ 2930, 2930 },
		{ 2930, 2930 },
		{ 1839, 1839 },
		{ 1839, 1839 },
		{ 1365, 1365 },
		{ 1365, 1365 },
		{ 3158, 3158 },
		{ 3158, 3158 },
		{ 2875, 2875 },
		{ 2875, 2875 },
		{ 3189, 3189 },
		{ 3189, 3189 },
		{ 3023, 3023 },
		{ 3023, 3023 },
		{ 1855, 1854 },
		{ 2833, 2833 },
		{ 2611, 2581 },
		{ 2646, 2646 },
		{ 2175, 2159 },
		{ 3117, 3117 },
		{ 2254, 2251 },
		{ 2396, 2396 },
		{ 2537, 2505 },
		{ 2930, 2930 },
		{ 1571, 1552 },
		{ 1839, 1839 },
		{ 1746, 1745 },
		{ 1365, 1365 },
		{ 1796, 1795 },
		{ 3158, 3158 },
		{ 1720, 1719 },
		{ 2875, 2875 },
		{ 1982, 1966 },
		{ 3189, 3189 },
		{ 2270, 2269 },
		{ 3023, 3023 },
		{ 2978, 2978 },
		{ 2978, 2978 },
		{ 2483, 2483 },
		{ 2483, 2483 },
		{ 2147, 2128 },
		{ 2037, 2035 },
		{ 1692, 1691 },
		{ 1755, 1754 },
		{ 1954, 1935 },
		{ 2763, 2733 },
		{ 3299, 3296 },
		{ 1397, 1396 },
		{ 1770, 1769 },
		{ 1809, 1808 },
		{ 3254, 3250 },
		{ 2304, 2303 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 1834, 1834 },
		{ 1834, 1834 },
		{ 1823, 1823 },
		{ 1823, 1823 },
		{ 1749, 1748 },
		{ 2978, 2978 },
		{ 1773, 1772 },
		{ 2483, 2483 },
		{ 2736, 2706 },
		{ 3178, 3178 },
		{ 3178, 3178 },
		{ 3142, 3142 },
		{ 3142, 3142 },
		{ 1349, 1349 },
		{ 1349, 1349 },
		{ 3101, 3101 },
		{ 3101, 3101 },
		{ 2993, 2993 },
		{ 2993, 2993 },
		{ 1360, 1360 },
		{ 1360, 1360 },
		{ 3135, 3135 },
		{ 1317, 1316 },
		{ 1834, 1834 },
		{ 1868, 1867 },
		{ 1823, 1823 },
		{ 2895, 2877 },
		{ 2165, 2148 },
		{ 2741, 2711 },
		{ 1836, 1835 },
		{ 2293, 2292 },
		{ 2856, 2833 },
		{ 3178, 3178 },
		{ 2681, 2646 },
		{ 3142, 3142 },
		{ 3118, 3117 },
		{ 1349, 1349 },
		{ 2431, 2396 },
		{ 3101, 3101 },
		{ 2936, 2930 },
		{ 2993, 2993 },
		{ 1840, 1839 },
		{ 1360, 1360 },
		{ 1366, 1365 },
		{ 2499, 2467 },
		{ 3159, 3158 },
		{ 2826, 2802 },
		{ 2893, 2875 },
		{ 2755, 2725 },
		{ 3190, 3189 },
		{ 1998, 1985 },
		{ 3024, 3023 },
		{ 1707, 1706 },
		{ 2118, 2097 },
		{ 2629, 2595 },
		{ 2921, 2910 },
		{ 1383, 1382 },
		{ 2010, 2000 },
		{ 1941, 1922 },
		{ 2850, 2827 },
		{ 2223, 2205 },
		{ 1782, 1781 },
		{ 2853, 2830 },
		{ 3268, 3263 },
		{ 1400, 1399 },
		{ 1952, 1933 },
		{ 1636, 1610 },
		{ 3051, 3050 },
		{ 1695, 1694 },
		{ 3279, 3278 },
		{ 2944, 2941 },
		{ 1917, 1897 },
		{ 1412, 1411 },
		{ 1723, 1722 },
		{ 1362, 1361 },
		{ 2979, 2978 },
		{ 2145, 2124 },
		{ 2516, 2483 },
		{ 2146, 2126 },
		{ 1539, 1517 },
		{ 3182, 3181 },
		{ 2258, 2255 },
		{ 1858, 1857 },
		{ 3305, 3302 },
		{ 1799, 1798 },
		{ 2881, 2862 },
		{ 2610, 2580 },
		{ 2588, 2588 },
		{ 2588, 2588 },
		{ 2592, 2560 },
		{ 3136, 3135 },
		{ 3047, 3046 },
		{ 1835, 1834 },
		{ 3074, 3073 },
		{ 1824, 1823 },
		{ 2747, 2747 },
		{ 2747, 2747 },
		{ 2748, 2748 },
		{ 2748, 2748 },
		{ 1313, 1313 },
		{ 1313, 1313 },
		{ 3179, 3178 },
		{ 1891, 2047 },
		{ 3143, 3142 },
		{ 2082, 2273 },
		{ 1350, 1349 },
		{ 1471, 1462 },
		{ 3102, 3101 },
		{ 2267, 2265 },
		{ 2994, 2993 },
		{ 2588, 2588 },
		{ 1361, 1360 },
		{ 0, 2328 },
		{ 1972, 1957 },
		{ 1810, 1809 },
		{ 1922, 1903 },
		{ 1371, 1370 },
		{ 2571, 2537 },
		{ 2747, 2747 },
		{ 3155, 3154 },
		{ 2748, 2748 },
		{ 2691, 2656 },
		{ 1313, 1313 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 2435, 2435 },
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
		{ 0, 1293 },
		{ 0, 2328 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 3203, 3203 },
		{ 3203, 3203 },
		{ 3203, 3203 },
		{ 3203, 3203 },
		{ 3203, 3203 },
		{ 3203, 3203 },
		{ 3203, 3203 },
		{ 3203, 3203 },
		{ 3203, 3203 },
		{ 3203, 3203 },
		{ 0, 1293 },
		{ 3006, 3005 },
		{ 2871, 2850 },
		{ 2618, 2588 },
		{ 2874, 2853 },
		{ 2521, 2488 },
		{ 2877, 2856 },
		{ 3164, 3163 },
		{ 2522, 2489 },
		{ 1413, 1412 },
		{ 2626, 2592 },
		{ 2776, 2747 },
		{ 2785, 2756 },
		{ 2777, 2748 },
		{ 1406, 1405 },
		{ 1314, 1313 },
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
		{ 1303, 1303 },
		{ 1303, 1303 },
		{ 1303, 1303 },
		{ 1303, 1303 },
		{ 1303, 1303 },
		{ 1303, 1303 },
		{ 1303, 1303 },
		{ 1303, 1303 },
		{ 1303, 1303 },
		{ 1303, 1303 },
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
		{ 1381, 1381 },
		{ 1381, 1381 },
		{ 2734, 2734 },
		{ 2734, 2734 },
		{ 0, 86 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 3069, 3069 },
		{ 3069, 3069 },
		{ 3304, 3304 },
		{ 3088, 3088 },
		{ 3088, 3088 },
		{ 3009, 3009 },
		{ 3009, 3009 },
		{ 2884, 2865 },
		{ 1381, 1381 },
		{ 2176, 2160 },
		{ 2734, 2734 },
		{ 2714, 2714 },
		{ 2714, 2714 },
		{ 3029, 3028 },
		{ 1472, 1451 },
		{ 2792, 2763 },
		{ 3186, 3185 },
		{ 2190, 2174 },
		{ 1322, 1321 },
		{ 1995, 1981 },
		{ 2705, 2673 },
		{ 0, 86 },
		{ 2802, 2774 },
		{ 3069, 3069 },
		{ 2709, 2678 },
		{ 3304, 3304 },
		{ 3088, 3088 },
		{ 3195, 3194 },
		{ 3009, 3009 },
		{ 2710, 2680 },
		{ 2711, 2681 },
		{ 3050, 3049 },
		{ 2910, 2893 },
		{ 1424, 1423 },
		{ 2714, 2714 },
		{ 2913, 2896 },
		{ 2206, 2192 },
		{ 3057, 3056 },
		{ 2207, 2193 },
		{ 1408, 1407 },
		{ 2536, 2504 },
		{ 1865, 1864 },
		{ 1441, 1440 },
		{ 1701, 1700 },
		{ 2733, 2703 },
		{ 1831, 1830 },
		{ 3075, 3074 },
		{ 2824, 2800 },
		{ 1357, 1356 },
		{ 2827, 2803 },
		{ 1703, 1702 },
		{ 3092, 3091 },
		{ 1553, 1533 },
		{ 2830, 2807 },
		{ 3095, 3094 },
		{ 2941, 2936 },
		{ 1900, 1878 },
		{ 1705, 1704 },
		{ 1410, 1409 },
		{ 2467, 2431 },
		{ 2951, 2950 },
		{ 2841, 2818 },
		{ 3114, 3113 },
		{ 2505, 2471 },
		{ 2040, 2038 },
		{ 2849, 2826 },
		{ 1708, 1707 },
		{ 2256, 2253 },
		{ 3292, 3292 },
		{ 1808, 1807 },
		{ 3123, 3122 },
		{ 2759, 2729 },
		{ 1444, 1443 },
		{ 1918, 1898 },
		{ 2984, 2983 },
		{ 1382, 1381 },
		{ 1845, 1844 },
		{ 2764, 2734 },
		{ 2086, 2064 },
		{ 2749, 2719 },
		{ 3133, 3132 },
		{ 1473, 1451 },
		{ 1830, 1829 },
		{ 2872, 2851 },
		{ 2873, 2852 },
		{ 3071, 3070 },
		{ 3146, 3145 },
		{ 1356, 1355 },
		{ 2836, 2813 },
		{ 3304, 3301 },
		{ 3070, 3069 },
		{ 3292, 3292 },
		{ 3089, 3088 },
		{ 2269, 2268 },
		{ 3010, 3009 },
		{ 3306, 3304 },
		{ 3105, 3104 },
		{ 2997, 2996 },
		{ 69, 5 },
		{ 3086, 3085 },
		{ 2744, 2714 },
		{ 3091, 3090 },
		{ 2260, 2258 },
		{ 1359, 1358 },
		{ 1423, 1422 },
		{ 1833, 1832 },
		{ 3077, 3076 },
		{ 3097, 3096 },
		{ 2815, 2788 },
		{ 2868, 2847 },
		{ 2697, 2662 },
		{ 2684, 2649 },
		{ 3037, 3035 },
		{ 3235, 3229 },
		{ 1911, 1888 },
		{ 2713, 2683 },
		{ 1893, 1873 },
		{ 2275, 2272 },
		{ 1912, 1888 },
		{ 2628, 2594 },
		{ 3100, 3099 },
		{ 1892, 1873 },
		{ 2274, 2272 },
		{ 1901, 1878 },
		{ 1533, 1511 },
		{ 3141, 3140 },
		{ 2439, 2405 },
		{ 2568, 2534 },
		{ 3038, 3035 },
		{ 2290, 2289 },
		{ 2683, 2648 },
		{ 2084, 2061 },
		{ 1345, 1344 },
		{ 3012, 3011 },
		{ 2518, 2485 },
		{ 3049, 3048 },
		{ 2083, 2061 },
		{ 1819, 1818 },
		{ 2594, 2562 },
		{ 205, 181 },
		{ 2825, 2801 },
		{ 2049, 2046 },
		{ 203, 181 },
		{ 3236, 3229 },
		{ 204, 181 },
		{ 2087, 2064 },
		{ 2048, 2046 },
		{ 2801, 2773 },
		{ 1489, 1463 },
		{ 2157, 2139 },
		{ 176, 5 },
		{ 206, 181 },
		{ 1821, 1820 },
		{ 202, 181 },
		{ 177, 5 },
		{ 2654, 2618 },
		{ 2912, 2895 },
		{ 1767, 1766 },
		{ 2564, 2530 },
		{ 1364, 1363 },
		{ 3295, 3292 },
		{ 178, 5 },
		{ 1379, 1378 },
		{ 2659, 2623 },
		{ 1826, 1825 },
		{ 2164, 2147 },
		{ 3108, 3107 },
		{ 2663, 2627 },
		{ 3110, 3109 },
		{ 2011, 2001 },
		{ 2929, 2921 },
		{ 1721, 1720 },
		{ 2669, 2632 },
		{ 1771, 1770 },
		{ 1415, 1414 },
		{ 2027, 2019 },
		{ 2028, 2020 },
		{ 175, 5 },
		{ 1919, 1899 },
		{ 2804, 2776 },
		{ 2805, 2777 },
		{ 2943, 2940 },
		{ 1398, 1397 },
		{ 1689, 1688 },
		{ 2947, 2944 },
		{ 3134, 3133 },
		{ 1779, 1778 },
		{ 1422, 1421 },
		{ 2039, 2037 },
		{ 2208, 2194 },
		{ 1348, 1347 },
		{ 2687, 2652 },
		{ 2817, 2790 },
		{ 3149, 3148 },
		{ 2976, 2975 },
		{ 3151, 3150 },
		{ 2688, 2653 },
		{ 1315, 1314 },
		{ 2367, 2334 },
		{ 1693, 1692 },
		{ 1838, 1837 },
		{ 1344, 1343 },
		{ 1352, 1351 },
		{ 2239, 2226 },
		{ 2243, 2236 },
		{ 1793, 1792 },
		{ 1736, 1735 },
		{ 2085, 2062 },
		{ 2701, 2666 },
		{ 3000, 2999 },
		{ 100, 83 },
		{ 3175, 3174 },
		{ 3002, 3001 },
		{ 1743, 1742 },
		{ 1797, 1796 },
		{ 2523, 2490 },
		{ 2708, 2677 },
		{ 1947, 1928 },
		{ 2843, 2820 },
		{ 3011, 3010 },
		{ 1852, 1851 },
		{ 2257, 2254 },
		{ 2848, 2825 },
		{ 3015, 3014 },
		{ 1510, 1488 },
		{ 3018, 3017 },
		{ 1951, 1932 },
		{ 3022, 3021 },
		{ 3208, 3199 },
		{ 1511, 1489 },
		{ 2717, 2687 },
		{ 2718, 2688 },
		{ 1429, 1428 },
		{ 1955, 1936 },
		{ 2532, 2499 },
		{ 1856, 1855 },
		{ 2860, 2837 },
		{ 1958, 1940 },
		{ 3239, 3233 },
		{ 3240, 3235 },
		{ 2862, 2839 },
		{ 1747, 1746 },
		{ 2732, 2702 },
		{ 2122, 2101 },
		{ 3041, 3039 },
		{ 2123, 2102 },
		{ 1387, 1386 },
		{ 1311, 1310 },
		{ 3258, 3254 },
		{ 3048, 3047 },
		{ 1394, 1393 },
		{ 2289, 2288 },
		{ 1970, 1954 },
		{ 3052, 3051 },
		{ 2742, 2712 },
		{ 1332, 1331 },
		{ 1437, 1436 },
		{ 2140, 2119 },
		{ 3276, 3275 },
		{ 2751, 2721 },
		{ 1710, 1709 },
		{ 2635, 2600 },
		{ 1812, 1811 },
		{ 1592, 1571 },
		{ 1717, 1716 },
		{ 1894, 1874 },
		{ 3288, 3285 },
		{ 1895, 1875 },
		{ 1896, 1875 },
		{ 3072, 3071 },
		{ 2643, 2608 },
		{ 2305, 2304 },
		{ 3297, 3294 },
		{ 2556, 2523 },
		{ 2765, 2735 },
		{ 2898, 2880 },
		{ 1593, 1572 },
		{ 2307, 2306 },
		{ 3087, 3086 },
		{ 2770, 2741 },
		{ 3090, 3089 },
		{ 2149, 2131 },
		{ 1436, 1435 },
		{ 130, 115 },
		{ 1864, 1863 },
		{ 2839, 2816 },
		{ 1949, 1930 },
		{ 3246, 3241 },
		{ 2596, 2564 },
		{ 2437, 2402 },
		{ 3138, 3137 },
		{ 2142, 2121 },
		{ 3294, 3291 },
		{ 1939, 1920 },
		{ 1959, 1941 },
		{ 2693, 2658 },
		{ 3019, 3018 },
		{ 3177, 3176 },
		{ 3042, 3040 },
		{ 1518, 1496 },
		{ 3214, 3212 },
		{ 3068, 3067 },
		{ 2730, 2700 },
		{ 2129, 2108 },
		{ 2101, 2078 },
		{ 3104, 3103 },
		{ 1843, 1842 },
		{ 2753, 2723 },
		{ 2666, 2630 },
		{ 3193, 3192 },
		{ 3162, 3161 },
		{ 1987, 1972 },
		{ 2982, 2981 },
		{ 3290, 3287 },
		{ 3021, 3020 },
		{ 1897, 1875 },
		{ 3293, 3290 },
		{ 2726, 2696 },
		{ 1369, 1368 },
		{ 1439, 1438 },
		{ 3145, 3144 },
		{ 2878, 2857 },
		{ 3027, 3026 },
		{ 3301, 3298 },
		{ 653, 591 },
		{ 1316, 1315 },
		{ 3121, 3120 },
		{ 3073, 3072 },
		{ 3308, 3306 },
		{ 2191, 2176 },
		{ 2996, 2995 },
		{ 1729, 1728 },
		{ 2803, 2775 },
		{ 2774, 2745 },
		{ 654, 591 },
		{ 2082, 2076 },
		{ 2807, 2779 },
		{ 1421, 1418 },
		{ 2715, 2685 },
		{ 1891, 1887 },
		{ 2746, 2716 },
		{ 2750, 2720 },
		{ 1471, 1464 },
		{ 2303, 2302 },
		{ 1724, 1723 },
		{ 2767, 2738 },
		{ 1774, 1773 },
		{ 1318, 1317 },
		{ 3046, 3045 },
		{ 1431, 1430 },
		{ 655, 591 },
		{ 2177, 2162 },
		{ 2178, 2165 },
		{ 1913, 1889 },
		{ 2097, 2073 },
		{ 2775, 2746 },
		{ 2593, 2561 },
		{ 2894, 2876 },
		{ 2682, 2647 },
		{ 2779, 2750 },
		{ 2897, 2879 },
		{ 1800, 1799 },
		{ 2995, 2994 },
		{ 1758, 1757 },
		{ 3144, 3143 },
		{ 1652, 1636 },
		{ 2100, 2077 },
		{ 2205, 2191 },
		{ 2842, 2819 },
		{ 2786, 2757 },
		{ 1869, 1868 },
		{ 3004, 3003 },
		{ 3153, 3152 },
		{ 2036, 2033 },
		{ 1781, 1780 },
		{ 2791, 2762 },
		{ 2645, 2610 },
		{ 3271, 3268 },
		{ 2793, 2764 },
		{ 1411, 1410 },
		{ 1401, 1400 },
		{ 1784, 1783 },
		{ 3278, 3277 },
		{ 1320, 1319 },
		{ 1706, 1705 },
		{ 3281, 3280 },
		{ 2926, 2915 },
		{ 2745, 2715 },
		{ 2291, 2290 },
		{ 1750, 1749 },
		{ 1696, 1695 },
		{ 2863, 2840 },
		{ 3176, 3175 },
		{ 3291, 3288 },
		{ 2000, 1987 },
		{ 2808, 2781 },
		{ 2573, 2539 },
		{ 3181, 3180 },
		{ 2244, 2237 },
		{ 3103, 3102 },
		{ 3298, 3295 },
		{ 3184, 3183 },
		{ 1384, 1383 },
		{ 2008, 1998 },
		{ 1859, 1858 },
		{ 2706, 2675 },
		{ 2301, 2300 },
		{ 2948, 2946 },
		{ 3307, 3305 },
		{ 2540, 2508 },
		{ 3112, 3111 },
		{ 2665, 2629 },
		{ 2250, 2246 },
		{ 904, 849 },
		{ 448, 403 },
		{ 717, 652 },
		{ 786, 726 },
		{ 1308, 11 },
		{ 1342, 15 },
		{ 1765, 31 },
		{ 69, 11 },
		{ 69, 15 },
		{ 69, 31 },
		{ 3080, 57 },
		{ 1329, 13 },
		{ 1392, 19 },
		{ 69, 57 },
		{ 69, 13 },
		{ 69, 19 },
		{ 1791, 33 },
		{ 449, 403 },
		{ 1817, 35 },
		{ 69, 33 },
		{ 785, 726 },
		{ 69, 35 },
		{ 2973, 47 },
		{ 1923, 1904 },
		{ 718, 652 },
		{ 69, 47 },
		{ 905, 849 },
		{ 1420, 21 },
		{ 1715, 27 },
		{ 1741, 29 },
		{ 69, 21 },
		{ 69, 27 },
		{ 69, 29 },
		{ 3170, 61 },
		{ 1068, 1030 },
		{ 1076, 1039 },
		{ 69, 61 },
		{ 1086, 1049 },
		{ 1106, 1071 },
		{ 1125, 1089 },
		{ 1134, 1100 },
		{ 1136, 1102 },
		{ 1149, 1119 },
		{ 1164, 1135 },
		{ 1170, 1142 },
		{ 2156, 2138 },
		{ 1177, 1150 },
		{ 1178, 1151 },
		{ 1192, 1173 },
		{ 1943, 1924 },
		{ 1944, 1925 },
		{ 1193, 1174 },
		{ 1211, 1194 },
		{ 1216, 1200 },
		{ 1224, 1212 },
		{ 2172, 2155 },
		{ 1236, 1225 },
		{ 1242, 1231 },
		{ 1277, 1276 },
		{ 272, 229 },
		{ 282, 238 },
		{ 290, 246 },
		{ 305, 258 },
		{ 1964, 1946 },
		{ 314, 267 },
		{ 321, 273 },
		{ 331, 283 },
		{ 349, 300 },
		{ 352, 303 },
		{ 361, 311 },
		{ 362, 312 },
		{ 367, 317 },
		{ 1979, 1963 },
		{ 382, 335 },
		{ 408, 363 },
		{ 411, 366 },
		{ 419, 374 },
		{ 430, 386 },
		{ 437, 393 },
		{ 446, 401 },
		{ 236, 198 },
		{ 450, 404 },
		{ 463, 415 },
		{ 240, 202 },
		{ 485, 431 },
		{ 488, 435 },
		{ 1306, 11 },
		{ 1340, 15 },
		{ 1763, 31 },
		{ 498, 443 },
		{ 499, 444 },
		{ 502, 447 },
		{ 3079, 57 },
		{ 1327, 13 },
		{ 1390, 19 },
		{ 514, 459 },
		{ 523, 470 },
		{ 556, 494 },
		{ 1789, 33 },
		{ 567, 503 },
		{ 1815, 35 },
		{ 570, 506 },
		{ 571, 507 },
		{ 575, 511 },
		{ 2971, 47 },
		{ 588, 526 },
		{ 592, 530 },
		{ 596, 534 },
		{ 606, 544 },
		{ 1418, 21 },
		{ 1713, 27 },
		{ 1739, 29 },
		{ 622, 557 },
		{ 623, 559 },
		{ 628, 564 },
		{ 3168, 61 },
		{ 645, 581 },
		{ 241, 203 },
		{ 662, 595 },
		{ 675, 608 },
		{ 678, 611 },
		{ 688, 621 },
		{ 705, 637 },
		{ 706, 638 },
		{ 716, 651 },
		{ 248, 210 },
		{ 736, 670 },
		{ 249, 211 },
		{ 787, 727 },
		{ 799, 738 },
		{ 801, 740 },
		{ 803, 742 },
		{ 808, 747 },
		{ 809, 748 },
		{ 829, 769 },
		{ 840, 779 },
		{ 844, 783 },
		{ 845, 784 },
		{ 877, 819 },
		{ 2111, 2090 },
		{ 896, 841 },
		{ 271, 228 },
		{ 936, 883 },
		{ 946, 893 },
		{ 961, 908 },
		{ 1000, 951 },
		{ 1004, 955 },
		{ 1020, 974 },
		{ 1021, 975 },
		{ 1022, 976 },
		{ 1038, 996 },
		{ 1065, 1027 },
		{ 2133, 2112 },
		{ 2134, 2113 },
		{ 2462, 2462 },
		{ 2462, 2462 },
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
		{ 618, 556 },
		{ 2184, 2170 },
		{ 3255, 3251 },
		{ 221, 189 },
		{ 620, 556 },
		{ 2186, 2171 },
		{ 3261, 3256 },
		{ 3227, 3226 },
		{ 2185, 2170 },
		{ 2462, 2462 },
		{ 2245, 2238 },
		{ 855, 794 },
		{ 2187, 2171 },
		{ 457, 408 },
		{ 429, 385 },
		{ 621, 556 },
		{ 458, 409 },
		{ 619, 556 },
		{ 224, 189 },
		{ 222, 189 },
		{ 490, 437 },
		{ 492, 437 },
		{ 533, 479 },
		{ 534, 479 },
		{ 761, 702 },
		{ 762, 703 },
		{ 2182, 2168 },
		{ 1991, 1977 },
		{ 413, 368 },
		{ 525, 472 },
		{ 535, 479 },
		{ 493, 437 },
		{ 613, 551 },
		{ 300, 253 },
		{ 715, 650 },
		{ 345, 296 },
		{ 491, 437 },
		{ 2094, 2070 },
		{ 1167, 1138 },
		{ 852, 791 },
		{ 1173, 1145 },
		{ 546, 488 },
		{ 686, 619 },
		{ 216, 186 },
		{ 2115, 2094 },
		{ 2093, 2070 },
		{ 215, 186 },
		{ 465, 417 },
		{ 375, 325 },
		{ 952, 899 },
		{ 548, 488 },
		{ 953, 900 },
		{ 217, 186 },
		{ 2727, 2727 },
		{ 2727, 2727 },
		{ 547, 488 },
		{ 209, 183 },
		{ 2092, 2070 },
		{ 211, 183 },
		{ 440, 395 },
		{ 439, 395 },
		{ 210, 183 },
		{ 537, 481 },
		{ 1182, 1155 },
		{ 2494, 2462 },
		{ 605, 543 },
		{ 231, 193 },
		{ 1849, 37 },
		{ 2286, 43 },
		{ 3127, 59 },
		{ 1375, 17 },
		{ 2989, 49 },
		{ 1686, 25 },
		{ 310, 263 },
		{ 3035, 53 },
		{ 659, 592 },
		{ 2727, 2727 },
		{ 3062, 55 },
		{ 230, 193 },
		{ 658, 592 },
		{ 835, 774 },
		{ 1181, 1155 },
		{ 538, 481 },
		{ 540, 483 },
		{ 2116, 2095 },
		{ 311, 264 },
		{ 541, 483 },
		{ 657, 592 },
		{ 656, 592 },
		{ 834, 774 },
		{ 227, 190 },
		{ 747, 685 },
		{ 948, 895 },
		{ 225, 190 },
		{ 265, 223 },
		{ 276, 232 },
		{ 226, 190 },
		{ 1926, 1907 },
		{ 856, 795 },
		{ 857, 796 },
		{ 553, 492 },
		{ 291, 247 },
		{ 1336, 1335 },
		{ 442, 397 },
		{ 1214, 1197 },
		{ 3225, 3223 },
		{ 836, 775 },
		{ 1377, 1375 },
		{ 783, 724 },
		{ 1309, 1306 },
		{ 3238, 3232 },
		{ 1232, 1221 },
		{ 1233, 1222 },
		{ 1234, 1223 },
		{ 1110, 1075 },
		{ 554, 492 },
		{ 292, 247 },
		{ 784, 725 },
		{ 641, 576 },
		{ 3247, 3242 },
		{ 1135, 1101 },
		{ 510, 455 },
		{ 3252, 3248 },
		{ 730, 664 },
		{ 2757, 2727 },
		{ 594, 532 },
		{ 1906, 1882 },
		{ 1011, 964 },
		{ 1015, 969 },
		{ 572, 508 },
		{ 861, 800 },
		{ 2974, 2971 },
		{ 749, 687 },
		{ 751, 689 },
		{ 2569, 2535 },
		{ 1330, 1327 },
		{ 242, 204 },
		{ 914, 860 },
		{ 1207, 1190 },
		{ 212, 184 },
		{ 585, 523 },
		{ 1240, 1229 },
		{ 1241, 1230 },
		{ 580, 517 },
		{ 213, 184 },
		{ 1246, 1237 },
		{ 1261, 1252 },
		{ 363, 313 },
		{ 1288, 1287 },
		{ 912, 858 },
		{ 586, 524 },
		{ 919, 865 },
		{ 931, 878 },
		{ 366, 316 },
		{ 584, 523 },
		{ 287, 243 },
		{ 733, 667 },
		{ 438, 394 },
		{ 199, 180 },
		{ 748, 686 },
		{ 962, 909 },
		{ 968, 915 },
		{ 201, 180 },
		{ 976, 924 },
		{ 981, 929 },
		{ 986, 936 },
		{ 992, 944 },
		{ 3226, 3225 },
		{ 999, 950 },
		{ 200, 180 },
		{ 600, 538 },
		{ 604, 542 },
		{ 752, 690 },
		{ 515, 460 },
		{ 380, 332 },
		{ 763, 704 },
		{ 611, 549 },
		{ 3243, 3238 },
		{ 1023, 977 },
		{ 1035, 993 },
		{ 317, 270 },
		{ 1043, 1001 },
		{ 1044, 1002 },
		{ 1048, 1006 },
		{ 3251, 3247 },
		{ 1049, 1007 },
		{ 1055, 1013 },
		{ 384, 338 },
		{ 386, 340 },
		{ 3256, 3252 },
		{ 403, 359 },
		{ 544, 486 },
		{ 1088, 1051 },
		{ 1090, 1053 },
		{ 1094, 1057 },
		{ 1103, 1068 },
		{ 1105, 1070 },
		{ 355, 305 },
		{ 643, 578 },
		{ 1118, 1082 },
		{ 550, 490 },
		{ 648, 584 },
		{ 831, 771 },
		{ 832, 772 },
		{ 1138, 1104 },
		{ 1925, 1906 },
		{ 354, 305 },
		{ 353, 305 },
		{ 1146, 1113 },
		{ 652, 590 },
		{ 1163, 1134 },
		{ 461, 412 },
		{ 1165, 1136 },
		{ 839, 778 },
		{ 356, 306 },
		{ 2113, 2092 },
		{ 841, 780 },
		{ 307, 260 },
		{ 415, 370 },
		{ 850, 789 },
		{ 1186, 1164 },
		{ 1191, 1171 },
		{ 851, 790 },
		{ 418, 373 },
		{ 1200, 1182 },
		{ 1201, 1184 },
		{ 273, 230 },
		{ 422, 377 },
		{ 1212, 1195 },
		{ 699, 630 },
		{ 701, 633 },
		{ 874, 815 },
		{ 1225, 1213 },
		{ 1227, 1215 },
		{ 875, 816 },
		{ 579, 516 },
		{ 885, 829 },
		{ 893, 838 },
		{ 262, 221 },
		{ 220, 188 },
		{ 1017, 971 },
		{ 583, 522 },
		{ 219, 188 },
		{ 750, 688 },
		{ 263, 221 },
		{ 859, 798 },
		{ 644, 579 },
		{ 364, 314 },
		{ 391, 345 },
		{ 650, 588 },
		{ 1205, 1188 },
		{ 433, 389 },
		{ 1961, 1943 },
		{ 772, 715 },
		{ 318, 271 },
		{ 899, 844 },
		{ 1058, 1017 },
		{ 1059, 1018 },
		{ 1062, 1022 },
		{ 900, 845 },
		{ 1615, 1615 },
		{ 1618, 1618 },
		{ 1621, 1621 },
		{ 1624, 1624 },
		{ 1627, 1627 },
		{ 1630, 1630 },
		{ 234, 196 },
		{ 1071, 1033 },
		{ 235, 197 },
		{ 669, 602 },
		{ 1087, 1050 },
		{ 598, 536 },
		{ 412, 367 },
		{ 1091, 1054 },
		{ 685, 618 },
		{ 1648, 1648 },
		{ 1271, 1265 },
		{ 943, 890 },
		{ 945, 892 },
		{ 376, 328 },
		{ 565, 501 },
		{ 1114, 1078 },
		{ 1657, 1657 },
		{ 1615, 1615 },
		{ 1618, 1618 },
		{ 1621, 1621 },
		{ 1624, 1624 },
		{ 1627, 1627 },
		{ 1630, 1630 },
		{ 339, 290 },
		{ 609, 547 },
		{ 1130, 1095 },
		{ 1578, 1578 },
		{ 2151, 2133 },
		{ 251, 213 },
		{ 383, 336 },
		{ 711, 643 },
		{ 615, 553 },
		{ 1648, 1648 },
		{ 1141, 1107 },
		{ 247, 209 },
		{ 460, 411 },
		{ 991, 943 },
		{ 576, 512 },
		{ 426, 382 },
		{ 1657, 1657 },
		{ 1682, 1682 },
		{ 735, 669 },
		{ 634, 570 },
		{ 635, 571 },
		{ 527, 475 },
		{ 980, 928 },
		{ 814, 753 },
		{ 820, 760 },
		{ 443, 398 },
		{ 1578, 1578 },
		{ 614, 552 },
		{ 568, 504 },
		{ 503, 448 },
		{ 2169, 2152 },
		{ 506, 451 },
		{ 509, 454 },
		{ 1488, 1615 },
		{ 1488, 1618 },
		{ 1488, 1621 },
		{ 1488, 1624 },
		{ 1488, 1627 },
		{ 1488, 1630 },
		{ 3222, 3218 },
		{ 1682, 1682 },
		{ 625, 561 },
		{ 627, 563 },
		{ 1018, 972 },
		{ 348, 299 },
		{ 719, 653 },
		{ 633, 569 },
		{ 313, 266 },
		{ 1488, 1648 },
		{ 1026, 983 },
		{ 1032, 990 },
		{ 416, 371 },
		{ 1036, 994 },
		{ 854, 793 },
		{ 2202, 2188 },
		{ 1488, 1657 },
		{ 2203, 2189 },
		{ 521, 467 },
		{ 742, 676 },
		{ 1045, 1003 },
		{ 746, 682 },
		{ 323, 275 },
		{ 261, 220 },
		{ 862, 801 },
		{ 1274, 1272 },
		{ 1488, 1578 },
		{ 871, 811 },
		{ 1278, 1277 },
		{ 873, 813 },
		{ 1063, 1024 },
		{ 388, 342 },
		{ 423, 378 },
		{ 876, 817 },
		{ 374, 324 },
		{ 1078, 1041 },
		{ 651, 589 },
		{ 754, 692 },
		{ 755, 694 },
		{ 757, 697 },
		{ 1488, 1682 },
		{ 396, 350 },
		{ 397, 352 },
		{ 545, 487 },
		{ 767, 707 },
		{ 770, 712 },
		{ 771, 713 },
		{ 1963, 1945 },
		{ 289, 245 },
		{ 781, 722 },
		{ 404, 360 },
		{ 673, 606 },
		{ 1133, 1098 },
		{ 340, 291 },
		{ 677, 610 },
		{ 264, 222 },
		{ 957, 904 },
		{ 1978, 1962 },
		{ 960, 907 },
		{ 679, 612 },
		{ 683, 616 },
		{ 1153, 1123 },
		{ 1156, 1126 },
		{ 1157, 1127 },
		{ 963, 910 },
		{ 966, 913 },
		{ 558, 496 },
		{ 381, 333 },
		{ 978, 926 },
		{ 1172, 1144 },
		{ 979, 927 },
		{ 2155, 2137 },
		{ 2005, 1993 },
		{ 2006, 1994 },
		{ 1174, 1146 },
		{ 562, 499 },
		{ 563, 499 },
		{ 254, 215 },
		{ 640, 575 },
		{ 208, 182 },
		{ 253, 215 },
		{ 639, 575 },
		{ 406, 361 },
		{ 405, 361 },
		{ 294, 248 },
		{ 293, 248 },
		{ 529, 477 },
		{ 560, 498 },
		{ 561, 498 },
		{ 207, 182 },
		{ 696, 628 },
		{ 697, 628 },
		{ 257, 218 },
		{ 269, 226 },
		{ 319, 272 },
		{ 258, 218 },
		{ 299, 252 },
		{ 530, 477 },
		{ 796, 736 },
		{ 268, 226 },
		{ 2641, 2641 },
		{ 2641, 2641 },
		{ 320, 272 },
		{ 531, 478 },
		{ 259, 218 },
		{ 566, 502 },
		{ 298, 252 },
		{ 3248, 3243 },
		{ 797, 736 },
		{ 938, 885 },
		{ 1203, 1186 },
		{ 665, 598 },
		{ 626, 562 },
		{ 532, 478 },
		{ 723, 657 },
		{ 1029, 986 },
		{ 724, 658 },
		{ 725, 659 },
		{ 1962, 1944 },
		{ 1221, 1209 },
		{ 1222, 1210 },
		{ 505, 450 },
		{ 467, 419 },
		{ 2641, 2641 },
		{ 508, 453 },
		{ 1137, 1103 },
		{ 858, 797 },
		{ 542, 484 },
		{ 371, 321 },
		{ 2152, 2134 },
		{ 244, 206 },
		{ 2088, 2065 },
		{ 1051, 1009 },
		{ 802, 741 },
		{ 1980, 1964 },
		{ 1924, 1905 },
		{ 608, 546 },
		{ 977, 925 },
		{ 1927, 1908 },
		{ 2535, 2503 },
		{ 805, 744 },
		{ 487, 433 },
		{ 281, 237 },
		{ 810, 749 },
		{ 2173, 2156 },
		{ 1993, 1979 },
		{ 1280, 1279 },
		{ 407, 362 },
		{ 267, 225 },
		{ 3223, 3221 },
		{ 700, 631 },
		{ 2112, 2091 },
		{ 301, 254 },
		{ 284, 240 },
		{ 756, 696 },
		{ 2188, 2172 },
		{ 2117, 2096 },
		{ 1184, 1162 },
		{ 1185, 1163 },
		{ 370, 320 },
		{ 3242, 3237 },
		{ 1187, 1165 },
		{ 504, 449 },
		{ 712, 645 },
		{ 2676, 2641 },
		{ 624, 560 },
		{ 275, 231 },
		{ 1108, 1073 },
		{ 602, 540 },
		{ 496, 440 },
		{ 813, 752 },
		{ 1956, 1937 },
		{ 1121, 1085 },
		{ 274, 231 },
		{ 1024, 981 },
		{ 1220, 1208 },
		{ 1126, 1090 },
		{ 1127, 1091 },
		{ 1128, 1092 },
		{ 947, 894 },
		{ 2091, 2069 },
		{ 306, 259 },
		{ 2095, 2071 },
		{ 741, 675 },
		{ 582, 521 },
		{ 830, 770 },
		{ 744, 678 },
		{ 1905, 1881 },
		{ 2183, 2169 },
		{ 270, 227 },
		{ 1907, 1883 },
		{ 631, 567 },
		{ 255, 216 },
		{ 879, 823 },
		{ 1247, 1238 },
		{ 1258, 1249 },
		{ 1259, 1250 },
		{ 569, 505 },
		{ 1264, 1255 },
		{ 1154, 1124 },
		{ 1272, 1267 },
		{ 1273, 1268 },
		{ 1992, 1978 },
		{ 2221, 2202 },
		{ 612, 550 },
		{ 1052, 1010 },
		{ 1159, 1129 },
		{ 793, 733 },
		{ 1056, 1014 },
		{ 843, 782 },
		{ 795, 735 },
		{ 1168, 1139 },
		{ 511, 456 },
		{ 907, 851 },
		{ 982, 930 },
		{ 2016, 2005 },
		{ 983, 931 },
		{ 1175, 1148 },
		{ 453, 407 },
		{ 1072, 1034 },
		{ 464, 416 },
		{ 917, 863 },
		{ 616, 554 },
		{ 924, 870 },
		{ 927, 874 },
		{ 1009, 961 },
		{ 928, 875 },
		{ 573, 509 },
		{ 934, 881 },
		{ 520, 466 },
		{ 232, 194 },
		{ 765, 706 },
		{ 1256, 1247 },
		{ 392, 346 },
		{ 954, 901 },
		{ 1093, 1056 },
		{ 891, 836 },
		{ 766, 706 },
		{ 1265, 1258 },
		{ 1266, 1259 },
		{ 1270, 1264 },
		{ 549, 489 },
		{ 591, 529 },
		{ 674, 607 },
		{ 1107, 1072 },
		{ 1275, 1273 },
		{ 849, 788 },
		{ 1180, 1154 },
		{ 902, 847 },
		{ 1287, 1286 },
		{ 1183, 1159 },
		{ 967, 914 },
		{ 302, 255 },
		{ 1119, 1083 },
		{ 1120, 1084 },
		{ 1188, 1168 },
		{ 975, 923 },
		{ 528, 476 },
		{ 414, 369 },
		{ 1194, 1175 },
		{ 1199, 1181 },
		{ 252, 214 },
		{ 2167, 2150 },
		{ 758, 698 },
		{ 402, 358 },
		{ 922, 868 },
		{ 726, 660 },
		{ 729, 663 },
		{ 1976, 1960 },
		{ 601, 539 },
		{ 649, 585 },
		{ 932, 879 },
		{ 1218, 1206 },
		{ 1219, 1207 },
		{ 996, 947 },
		{ 1142, 1108 },
		{ 1143, 1110 },
		{ 687, 620 },
		{ 468, 420 },
		{ 1226, 1214 },
		{ 2114, 2093 },
		{ 1151, 1121 },
		{ 3221, 3217 },
		{ 1231, 1220 },
		{ 473, 428 },
		{ 581, 520 },
		{ 316, 269 },
		{ 445, 400 },
		{ 1158, 1128 },
		{ 661, 594 },
		{ 486, 432 },
		{ 3237, 3231 },
		{ 1243, 1234 },
		{ 881, 825 },
		{ 1089, 1052 },
		{ 647, 583 },
		{ 617, 555 },
		{ 632, 568 },
		{ 427, 383 },
		{ 1012, 965 },
		{ 372, 322 },
		{ 710, 642 },
		{ 737, 671 },
		{ 2153, 2135 },
		{ 2154, 2136 },
		{ 237, 199 },
		{ 804, 743 },
		{ 930, 877 },
		{ 637, 573 },
		{ 347, 298 },
		{ 883, 827 },
		{ 436, 392 },
		{ 459, 410 },
		{ 940, 887 },
		{ 471, 424 },
		{ 720, 654 },
		{ 984, 934 },
		{ 897, 842 },
		{ 818, 757 },
		{ 693, 625 },
		{ 695, 627 },
		{ 239, 201 },
		{ 1285, 1284 },
		{ 1248, 1239 },
		{ 1253, 1244 },
		{ 853, 792 },
		{ 1257, 1248 },
		{ 1112, 1076 },
		{ 1179, 1152 },
		{ 941, 888 },
		{ 1262, 1253 },
		{ 1113, 1076 },
		{ 1263, 1254 },
		{ 1111, 1076 },
		{ 731, 665 },
		{ 277, 233 },
		{ 987, 937 },
		{ 1269, 1263 },
		{ 988, 938 },
		{ 734, 668 },
		{ 329, 281 },
		{ 365, 315 },
		{ 664, 597 },
		{ 738, 672 },
		{ 1131, 1096 },
		{ 1132, 1097 },
		{ 1198, 1179 },
		{ 1281, 1280 },
		{ 296, 250 },
		{ 1008, 959 },
		{ 870, 810 },
		{ 2138, 2117 },
		{ 1202, 1185 },
		{ 1946, 1927 },
		{ 1066, 1028 },
		{ 1204, 1187 },
		{ 1067, 1029 },
		{ 837, 776 },
		{ 1069, 1031 },
		{ 918, 864 },
		{ 629, 565 },
		{ 1074, 1037 },
		{ 1217, 1203 },
		{ 1147, 1115 },
		{ 920, 866 },
		{ 1077, 1040 },
		{ 964, 911 },
		{ 334, 286 },
		{ 379, 331 },
		{ 769, 709 },
		{ 973, 921 },
		{ 806, 745 },
		{ 564, 500 },
		{ 880, 824 },
		{ 497, 442 },
		{ 1166, 1137 },
		{ 1099, 1064 },
		{ 390, 344 },
		{ 636, 572 },
		{ 238, 200 },
		{ 1039, 997 },
		{ 1040, 998 },
		{ 1109, 1074 },
		{ 882, 826 },
		{ 768, 708 },
		{ 1047, 1005 },
		{ 599, 537 },
		{ 308, 261 },
		{ 1476, 1476 },
		{ 1334, 1334 },
		{ 1334, 1334 },
		{ 552, 491 },
		{ 335, 287 },
		{ 1206, 1189 },
		{ 867, 808 },
		{ 551, 491 },
		{ 336, 287 },
		{ 868, 808 },
		{ 774, 716 },
		{ 773, 716 },
		{ 1101, 1065 },
		{ 775, 716 },
		{ 424, 379 },
		{ 1208, 1191 },
		{ 517, 463 },
		{ 1100, 1065 },
		{ 789, 729 },
		{ 937, 884 },
		{ 792, 732 },
		{ 519, 465 },
		{ 794, 734 },
		{ 1476, 1476 },
		{ 1334, 1334 },
		{ 1124, 1088 },
		{ 1028, 985 },
		{ 288, 244 },
		{ 1031, 989 },
		{ 681, 614 },
		{ 1129, 1094 },
		{ 1034, 992 },
		{ 798, 737 },
		{ 1229, 1218 },
		{ 322, 274 },
		{ 522, 468 },
		{ 428, 384 },
		{ 524, 471 },
		{ 409, 364 },
		{ 1238, 1227 },
		{ 260, 219 },
		{ 642, 577 },
		{ 1139, 1105 },
		{ 807, 746 },
		{ 607, 545 },
		{ 378, 330 },
		{ 1050, 1008 },
		{ 1249, 1240 },
		{ 1250, 1241 },
		{ 1977, 1961 },
		{ 243, 205 },
		{ 1255, 1246 },
		{ 1148, 1118 },
		{ 965, 912 },
		{ 1054, 1012 },
		{ 501, 446 },
		{ 2135, 2114 },
		{ 2136, 2115 },
		{ 753, 691 },
		{ 1335, 1334 },
		{ 702, 634 },
		{ 703, 635 },
		{ 1488, 1476 },
		{ 892, 837 },
		{ 823, 763 },
		{ 1162, 1133 },
		{ 1268, 1261 },
		{ 894, 839 },
		{ 824, 764 },
		{ 344, 295 },
		{ 536, 480 },
		{ 368, 318 },
		{ 539, 482 },
		{ 466, 418 },
		{ 1073, 1035 },
		{ 906, 850 },
		{ 400, 355 },
		{ 358, 308 },
		{ 660, 593 },
		{ 1084, 1047 },
		{ 1085, 1048 },
		{ 297, 251 },
		{ 472, 426 },
		{ 3171, 3168 },
		{ 663, 596 },
		{ 420, 375 },
		{ 921, 867 },
		{ 1001, 952 },
		{ 1092, 1055 },
		{ 2168, 2151 },
		{ 1002, 953 },
		{ 2170, 2153 },
		{ 2171, 2154 },
		{ 333, 285 },
		{ 1096, 1060 },
		{ 1097, 1061 },
		{ 847, 786 },
		{ 925, 871 },
		{ 1010, 962 },
		{ 1104, 1069 },
		{ 590, 528 },
		{ 671, 604 },
		{ 1014, 968 },
		{ 512, 457 },
		{ 385, 339 },
		{ 1116, 1080 },
		{ 1117, 1081 },
		{ 759, 699 },
		{ 903, 848 },
		{ 425, 380 },
		{ 1766, 1763 },
		{ 447, 402 },
		{ 1792, 1789 },
		{ 2019, 2010 },
		{ 1150, 1120 },
		{ 1046, 1004 },
		{ 1209, 1192 },
		{ 1210, 1193 },
		{ 1688, 1686 },
		{ 846, 785 },
		{ 593, 531 },
		{ 955, 902 },
		{ 2236, 2223 },
		{ 1942, 1923 },
		{ 1851, 1849 },
		{ 869, 809 },
		{ 740, 674 },
		{ 728, 662 },
		{ 2001, 1988 },
		{ 2132, 2111 },
		{ 1716, 1713 },
		{ 526, 473 },
		{ 346, 297 },
		{ 1393, 1390 },
		{ 1742, 1739 },
		{ 2194, 2179 },
		{ 993, 945 },
		{ 1190, 1170 },
		{ 860, 799 },
		{ 958, 905 },
		{ 994, 945 },
		{ 1223, 1211 },
		{ 732, 666 },
		{ 721, 655 },
		{ 1196, 1177 },
		{ 1197, 1178 },
		{ 1267, 1260 },
		{ 1060, 1020 },
		{ 1061, 1021 },
		{ 884, 828 },
		{ 985, 935 },
		{ 295, 249 },
		{ 399, 354 },
		{ 507, 452 },
		{ 1176, 1149 },
		{ 343, 294 },
		{ 872, 812 },
		{ 812, 751 },
		{ 777, 718 },
		{ 246, 208 },
		{ 1155, 1125 },
		{ 389, 343 },
		{ 1075, 1038 },
		{ 764, 705 },
		{ 452, 406 },
		{ 444, 399 },
		{ 1189, 1169 },
		{ 1016, 970 },
		{ 788, 728 },
		{ 2137, 2116 },
		{ 589, 527 },
		{ 1019, 973 },
		{ 680, 613 },
		{ 214, 185 },
		{ 933, 880 },
		{ 682, 615 },
		{ 935, 882 },
		{ 462, 413 },
		{ 1027, 984 },
		{ 739, 673 },
		{ 684, 617 },
		{ 1123, 1087 },
		{ 2150, 2132 },
		{ 1994, 1980 },
		{ 1030, 987 },
		{ 432, 388 },
		{ 410, 365 },
		{ 1033, 991 },
		{ 865, 804 },
		{ 2003, 1991 },
		{ 434, 390 },
		{ 435, 391 },
		{ 692, 624 },
		{ 369, 319 },
		{ 950, 897 },
		{ 1041, 999 },
		{ 951, 898 },
		{ 694, 626 },
		{ 351, 302 },
		{ 1908, 1884 },
		{ 513, 458 },
		{ 559, 497 },
		{ 956, 903 },
		{ 470, 422 },
		{ 279, 235 },
		{ 280, 236 },
		{ 441, 396 },
		{ 817, 756 },
		{ 474, 429 },
		{ 309, 262 },
		{ 821, 761 },
		{ 822, 762 },
		{ 887, 832 },
		{ 393, 347 },
		{ 972, 919 },
		{ 760, 700 },
		{ 2189, 2173 },
		{ 1064, 1025 },
		{ 825, 765 },
		{ 1160, 1131 },
		{ 1161, 1132 },
		{ 1260, 1251 },
		{ 2196, 2182 },
		{ 2198, 2184 },
		{ 2199, 2185 },
		{ 2200, 2186 },
		{ 2201, 2187 },
		{ 828, 768 },
		{ 895, 840 },
		{ 357, 307 },
		{ 245, 207 },
		{ 360, 310 },
		{ 489, 436 },
		{ 901, 846 },
		{ 2096, 2072 },
		{ 1169, 1140 },
		{ 833, 773 },
		{ 421, 376 },
		{ 494, 438 },
		{ 341, 292 },
		{ 401, 356 },
		{ 1945, 1926 },
		{ 1079, 1042 },
		{ 1083, 1046 },
		{ 1276, 1274 },
		{ 908, 853 },
		{ 910, 856 },
		{ 1279, 1278 },
		{ 196, 176 },
		{ 913, 859 },
		{ 1284, 1283 },
		{ 666, 599 },
		{ 1286, 1285 },
		{ 916, 862 },
		{ 312, 265 },
		{ 1960, 1942 },
		{ 324, 276 },
		{ 326, 278 },
		{ 1003, 954 },
		{ 779, 720 },
		{ 1095, 1058 },
		{ 283, 239 },
		{ 218, 187 },
		{ 923, 869 },
		{ 676, 609 },
		{ 1102, 1066 },
		{ 587, 525 },
		{ 926, 872 },
		{ 543, 485 },
		{ 1625, 1625 },
		{ 1625, 1625 },
		{ 2209, 2209 },
		{ 2209, 2209 },
		{ 2211, 2211 },
		{ 2211, 2211 },
		{ 2213, 2213 },
		{ 2213, 2213 },
		{ 2215, 2215 },
		{ 2215, 2215 },
		{ 2217, 2217 },
		{ 2217, 2217 },
		{ 2219, 2219 },
		{ 2219, 2219 },
		{ 1989, 1989 },
		{ 1989, 1989 },
		{ 689, 622 },
		{ 690, 622 },
		{ 2024, 2024 },
		{ 2024, 2024 },
		{ 2234, 2234 },
		{ 2234, 2234 },
		{ 1042, 1000 },
		{ 1625, 1625 },
		{ 330, 282 },
		{ 2209, 2209 },
		{ 595, 533 },
		{ 2211, 2211 },
		{ 898, 843 },
		{ 2213, 2213 },
		{ 431, 387 },
		{ 2215, 2215 },
		{ 574, 510 },
		{ 2217, 2217 },
		{ 304, 257 },
		{ 2219, 2219 },
		{ 969, 916 },
		{ 1989, 1989 },
		{ 1683, 1683 },
		{ 1683, 1683 },
		{ 1025, 982 },
		{ 2024, 2024 },
		{ 970, 917 },
		{ 2234, 2234 },
		{ 2029, 2029 },
		{ 2029, 2029 },
		{ 2180, 2180 },
		{ 2180, 2180 },
		{ 2240, 2240 },
		{ 2240, 2240 },
		{ 1628, 1628 },
		{ 1628, 1628 },
		{ 1626, 1625 },
		{ 325, 277 },
		{ 2210, 2209 },
		{ 2288, 2286 },
		{ 2212, 2211 },
		{ 819, 758 },
		{ 2214, 2213 },
		{ 974, 922 },
		{ 2216, 2215 },
		{ 1683, 1683 },
		{ 2218, 2217 },
		{ 2707, 2676 },
		{ 2220, 2219 },
		{ 800, 739 },
		{ 1990, 1989 },
		{ 2029, 2029 },
		{ 1140, 1106 },
		{ 2180, 2180 },
		{ 2025, 2024 },
		{ 2240, 2240 },
		{ 2235, 2234 },
		{ 1628, 1628 },
		{ 1619, 1619 },
		{ 1619, 1619 },
		{ 1631, 1631 },
		{ 1631, 1631 },
		{ 1579, 1579 },
		{ 1579, 1579 },
		{ 1658, 1658 },
		{ 1658, 1658 },
		{ 1622, 1622 },
		{ 1622, 1622 },
		{ 2043, 2043 },
		{ 2043, 2043 },
		{ 1649, 1649 },
		{ 1649, 1649 },
		{ 1616, 1616 },
		{ 1616, 1616 },
		{ 1684, 1683 },
		{ 2263, 2263 },
		{ 2263, 2263 },
		{ 2012, 2012 },
		{ 2012, 2012 },
		{ 727, 661 },
		{ 2030, 2029 },
		{ 1619, 1619 },
		{ 2181, 2180 },
		{ 1631, 1631 },
		{ 2241, 2240 },
		{ 1579, 1579 },
		{ 1629, 1628 },
		{ 1658, 1658 },
		{ 691, 623 },
		{ 1622, 1622 },
		{ 577, 514 },
		{ 2043, 2043 },
		{ 1144, 1111 },
		{ 1649, 1649 },
		{ 1145, 1112 },
		{ 1616, 1616 },
		{ 2014, 2014 },
		{ 2014, 2014 },
		{ 2263, 2263 },
		{ 495, 439 },
		{ 2012, 2012 },
		{ 889, 834 },
		{ 714, 649 },
		{ 1337, 1336 },
		{ 603, 541 },
		{ 285, 241 },
		{ 256, 217 },
		{ 337, 288 },
		{ 995, 946 },
		{ 1080, 1043 },
		{ 1620, 1619 },
		{ 1213, 1196 },
		{ 1632, 1631 },
		{ 1081, 1044 },
		{ 1580, 1579 },
		{ 1252, 1243 },
		{ 1659, 1658 },
		{ 1215, 1199 },
		{ 1623, 1622 },
		{ 2014, 2014 },
		{ 2044, 2043 },
		{ 915, 861 },
		{ 1650, 1649 },
		{ 780, 721 },
		{ 1617, 1616 },
		{ 3263, 3258 },
		{ 2017, 2006 },
		{ 2264, 2263 },
		{ 888, 833 },
		{ 2013, 2012 },
		{ 228, 191 },
		{ 949, 896 },
		{ 2158, 2141 },
		{ 1965, 1948 },
		{ 863, 802 },
		{ 1115, 1079 },
		{ 864, 803 },
		{ 1694, 1693 },
		{ 2222, 2203 },
		{ 1195, 1176 },
		{ 1798, 1797 },
		{ 1610, 1592 },
		{ 1005, 956 },
		{ 1748, 1747 },
		{ 1007, 958 },
		{ 1857, 1856 },
		{ 782, 723 },
		{ 3224, 3222 },
		{ 2015, 2014 },
		{ 1228, 1216 },
		{ 1399, 1398 },
		{ 1171, 1143 },
		{ 1230, 1219 },
		{ 811, 750 },
		{ 1985, 1970 },
		{ 838, 777 },
		{ 1122, 1086 },
		{ 848, 787 },
		{ 1235, 1224 },
		{ 359, 309 },
		{ 1237, 1226 },
		{ 1013, 967 },
		{ 1098, 1062 },
		{ 198, 178 },
		{ 395, 349 },
		{ 944, 891 },
		{ 1772, 1771 },
		{ 1283, 1282 },
		{ 1722, 1721 },
		{ 1245, 1236 },
		{ 1057, 1015 },
		{ 2090, 2068 },
		{ 518, 464 },
		{ 2023, 2016 },
		{ 229, 192 },
		{ 1251, 1242 },
		{ 1904, 1880 },
		{ 1343, 1340 },
		{ 2004, 1992 },
		{ 2197, 2183 },
		{ 989, 939 },
		{ 2233, 2221 },
		{ 646, 582 },
		{ 886, 830 },
		{ 3064, 3062 },
		{ 2102, 2079 },
		{ 315, 268 },
		{ 2881, 2881 },
		{ 2881, 2881 },
		{ 2765, 2765 },
		{ 2765, 2765 },
		{ 1847, 1847 },
		{ 1847, 1847 },
		{ 3166, 3166 },
		{ 3166, 3166 },
		{ 3031, 3031 },
		{ 3031, 3031 },
		{ 2954, 2954 },
		{ 2954, 2954 },
		{ 1325, 1325 },
		{ 1325, 1325 },
		{ 2885, 2885 },
		{ 2885, 2885 },
		{ 2887, 2887 },
		{ 2887, 2887 },
		{ 2888, 2888 },
		{ 2888, 2888 },
		{ 2769, 2769 },
		{ 2769, 2769 },
		{ 394, 348 },
		{ 2881, 2881 },
		{ 713, 648 },
		{ 2765, 2765 },
		{ 672, 605 },
		{ 1847, 1847 },
		{ 1053, 1011 },
		{ 3166, 3166 },
		{ 303, 256 },
		{ 3031, 3031 },
		{ 250, 212 },
		{ 2954, 2954 },
		{ 866, 807 },
		{ 1325, 1325 },
		{ 350, 301 },
		{ 2885, 2885 },
		{ 516, 461 },
		{ 2887, 2887 },
		{ 990, 941 },
		{ 2888, 2888 },
		{ 815, 754 },
		{ 2769, 2769 },
		{ 929, 876 },
		{ 2890, 2890 },
		{ 2890, 2890 },
		{ 2891, 2891 },
		{ 2891, 2891 },
		{ 2899, 2881 },
		{ 816, 755 },
		{ 2794, 2765 },
		{ 1282, 1281 },
		{ 1848, 1847 },
		{ 417, 372 },
		{ 3167, 3166 },
		{ 638, 574 },
		{ 3032, 3031 },
		{ 997, 948 },
		{ 2955, 2954 },
		{ 3129, 3127 },
		{ 1326, 1325 },
		{ 998, 949 },
		{ 2903, 2885 },
		{ 722, 656 },
		{ 2904, 2887 },
		{ 1818, 1815 },
		{ 2905, 2888 },
		{ 2890, 2890 },
		{ 2797, 2769 },
		{ 2891, 2891 },
		{ 578, 515 },
		{ 2770, 2770 },
		{ 2770, 2770 },
		{ 1761, 1761 },
		{ 1761, 1761 },
		{ 2634, 2634 },
		{ 2634, 2634 },
		{ 2844, 2844 },
		{ 2844, 2844 },
		{ 2541, 2541 },
		{ 2541, 2541 },
		{ 1813, 1813 },
		{ 1813, 1813 },
		{ 2638, 2638 },
		{ 2638, 2638 },
		{ 1373, 1373 },
		{ 1373, 1373 },
		{ 2987, 2987 },
		{ 2987, 2987 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 1737, 1737 },
		{ 1737, 1737 },
		{ 2907, 2890 },
		{ 2770, 2770 },
		{ 2908, 2891 },
		{ 1761, 1761 },
		{ 398, 353 },
		{ 2634, 2634 },
		{ 1070, 1032 },
		{ 2844, 2844 },
		{ 338, 289 },
		{ 2541, 2541 },
		{ 878, 820 },
		{ 1813, 1813 },
		{ 327, 279 },
		{ 2638, 2638 },
		{ 939, 886 },
		{ 1373, 1373 },
		{ 1006, 957 },
		{ 2987, 2987 },
		{ 610, 548 },
		{ 2901, 2901 },
		{ 328, 280 },
		{ 1737, 1737 },
		{ 942, 889 },
		{ 2906, 2906 },
		{ 2906, 2906 },
		{ 2909, 2909 },
		{ 2909, 2909 },
		{ 2798, 2770 },
		{ 1152, 1122 },
		{ 1762, 1761 },
		{ 826, 766 },
		{ 2671, 2634 },
		{ 827, 767 },
		{ 2867, 2844 },
		{ 555, 493 },
		{ 2542, 2541 },
		{ 1082, 1045 },
		{ 1814, 1813 },
		{ 776, 717 },
		{ 2674, 2638 },
		{ 278, 234 },
		{ 1374, 1373 },
		{ 778, 719 },
		{ 2988, 2987 },
		{ 557, 495 },
		{ 2917, 2901 },
		{ 2906, 2906 },
		{ 1738, 1737 },
		{ 2909, 2909 },
		{ 500, 445 },
		{ 2778, 2778 },
		{ 2778, 2778 },
		{ 1416, 1416 },
		{ 1416, 1416 },
		{ 3125, 3125 },
		{ 3125, 3125 },
		{ 2912, 2912 },
		{ 2912, 2912 },
		{ 2601, 2601 },
		{ 2601, 2601 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 2644, 2644 },
		{ 2644, 2644 },
		{ 2751, 2751 },
		{ 2751, 2751 },
		{ 2916, 2916 },
		{ 2916, 2916 },
		{ 1388, 1388 },
		{ 1388, 1388 },
		{ 2822, 2822 },
		{ 2822, 2822 },
		{ 2919, 2906 },
		{ 2778, 2778 },
		{ 2920, 2909 },
		{ 1416, 1416 },
		{ 890, 835 },
		{ 3125, 3125 },
		{ 342, 293 },
		{ 2912, 2912 },
		{ 197, 177 },
		{ 2601, 2601 },
		{ 469, 421 },
		{ 3060, 3060 },
		{ 387, 341 },
		{ 2644, 2644 },
		{ 1239, 1228 },
		{ 2751, 2751 },
		{ 233, 195 },
		{ 2916, 2916 },
		{ 373, 323 },
		{ 1388, 1388 },
		{ 332, 284 },
		{ 2822, 2822 },
		{ 959, 906 },
		{ 2823, 2823 },
		{ 2823, 2823 },
		{ 2667, 2667 },
		{ 2667, 2667 },
		{ 2806, 2778 },
		{ 1244, 1235 },
		{ 1417, 1416 },
		{ 842, 781 },
		{ 3126, 3125 },
		{ 698, 629 },
		{ 2923, 2912 },
		{ 790, 730 },
		{ 2636, 2601 },
		{ 791, 731 },
		{ 3061, 3060 },
		{ 286, 242 },
		{ 2679, 2644 },
		{ 3082, 3079 },
		{ 2780, 2751 },
		{ 2991, 2989 },
		{ 2927, 2916 },
		{ 2255, 2252 },
		{ 1389, 1388 },
		{ 2823, 2823 },
		{ 2845, 2822 },
		{ 2667, 2667 },
		{ 451, 405 },
		{ 1711, 1711 },
		{ 1711, 1711 },
		{ 2929, 2929 },
		{ 2929, 2929 },
		{ 2604, 2604 },
		{ 2604, 2604 },
		{ 2931, 2931 },
		{ 2931, 2931 },
		{ 2932, 2932 },
		{ 2932, 2932 },
		{ 2532, 2532 },
		{ 2532, 2532 },
		{ 2868, 2868 },
		{ 2868, 2868 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 1870, 1870 },
		{ 1870, 1870 },
		{ 2475, 2475 },
		{ 2475, 2475 },
		{ 2309, 2309 },
		{ 2309, 2309 },
		{ 2846, 2823 },
		{ 1711, 1711 },
		{ 2668, 2667 },
		{ 2929, 2929 },
		{ 743, 677 },
		{ 2604, 2604 },
		{ 597, 535 },
		{ 2931, 2931 },
		{ 745, 681 },
		{ 2932, 2932 },
		{ 1254, 1245 },
		{ 2532, 2532 },
		{ 266, 224 },
		{ 2868, 2868 },
		{ 1338, 1337 },
		{ 3077, 3077 },
		{ 1037, 995 },
		{ 1870, 1870 },
		{ 909, 855 },
		{ 2475, 2475 },
		{ 971, 918 },
		{ 2309, 2309 },
		{ 630, 566 },
		{ 2707, 2707 },
		{ 2707, 2707 },
		{ 2942, 2942 },
		{ 2942, 2942 },
		{ 1712, 1711 },
		{ 911, 857 },
		{ 2935, 2929 },
		{ 704, 636 },
		{ 2639, 2604 },
		{ 667, 600 },
		{ 2937, 2931 },
		{ 668, 601 },
		{ 2938, 2932 },
		{ 707, 639 },
		{ 2566, 2532 },
		{ 708, 640 },
		{ 2886, 2868 },
		{ 709, 641 },
		{ 3078, 3077 },
		{ 377, 329 },
		{ 1871, 1870 },
		{ 670, 603 },
		{ 2476, 2475 },
		{ 2707, 2707 },
		{ 2310, 2309 },
		{ 2942, 2942 },
		{ 2031, 2025 },
		{ 1787, 1787 },
		{ 1787, 1787 },
		{ 1338, 1338 },
		{ 1338, 1338 },
		{ 2736, 2736 },
		{ 2736, 2736 },
		{ 2947, 2947 },
		{ 2947, 2947 },
		{ 2835, 2835 },
		{ 2835, 2835 },
		{ 2836, 2836 },
		{ 2836, 2836 },
		{ 3300, 3300 },
		{ 3308, 3308 },
		{ 3312, 3312 },
		{ 1666, 1659 },
		{ 1642, 1620 },
		{ 2266, 2264 },
		{ 2242, 2235 },
		{ 1598, 1580 },
		{ 2034, 2030 },
		{ 2021, 2013 },
		{ 2737, 2707 },
		{ 1787, 1787 },
		{ 2945, 2942 },
		{ 1338, 1338 },
		{ 1646, 1632 },
		{ 2736, 2736 },
		{ 2227, 2210 },
		{ 2947, 2947 },
		{ 2247, 2241 },
		{ 2835, 2835 },
		{ 2022, 2015 },
		{ 2836, 2836 },
		{ 2228, 2212 },
		{ 3300, 3300 },
		{ 3308, 3308 },
		{ 3312, 3312 },
		{ 1644, 1626 },
		{ 2229, 2214 },
		{ 1641, 1617 },
		{ 2230, 2216 },
		{ 1643, 1623 },
		{ 2231, 2218 },
		{ 1645, 1629 },
		{ 2232, 2220 },
		{ 2002, 1990 },
		{ 1685, 1684 },
		{ 2195, 2181 },
		{ 1788, 1787 },
		{ 1660, 1650 },
		{ 1339, 1338 },
		{ 2045, 2044 },
		{ 2766, 2736 },
		{ 1777, 1776 },
		{ 2949, 2947 },
		{ 1778, 1777 },
		{ 2858, 2835 },
		{ 1405, 1404 },
		{ 2859, 2836 },
		{ 1676, 1672 },
		{ 3303, 3300 },
		{ 3310, 3308 },
		{ 3313, 3312 },
		{ 3274, 3273 },
		{ 3275, 3274 },
		{ 1753, 1752 },
		{ 1754, 1753 },
		{ 1404, 1403 },
		{ 3198, 3197 },
		{ 1672, 1667 },
		{ 1673, 1668 },
		{ 1448, 1447 },
		{ 1803, 1802 },
		{ 1804, 1803 },
		{ 1699, 1698 },
		{ 1700, 1699 },
		{ 1727, 1726 },
		{ 1862, 1861 },
		{ 1863, 1862 },
		{ 1728, 1727 },
		{ 2961, 2961 },
		{ 2958, 2961 },
		{ 165, 165 },
		{ 162, 165 },
		{ 1996, 1984 },
		{ 1997, 1984 },
		{ 1974, 1959 },
		{ 1975, 1959 },
		{ 2050, 2050 },
		{ 2276, 2276 },
		{ 170, 166 },
		{ 2966, 2962 },
		{ 2336, 2311 },
		{ 90, 72 },
		{ 2281, 2277 },
		{ 169, 166 },
		{ 2965, 2962 },
		{ 2335, 2311 },
		{ 89, 72 },
		{ 2280, 2277 },
		{ 2127, 2106 },
		{ 2055, 2051 },
		{ 2960, 2956 },
		{ 2961, 2961 },
		{ 164, 160 },
		{ 165, 165 },
		{ 2054, 2051 },
		{ 2959, 2956 },
		{ 2468, 2436 },
		{ 163, 160 },
		{ 171, 168 },
		{ 2050, 2050 },
		{ 2276, 2276 },
		{ 173, 172 },
		{ 2967, 2964 },
		{ 2969, 2968 },
		{ 121, 105 },
		{ 2403, 2368 },
		{ 2962, 2961 },
		{ 1934, 1915 },
		{ 166, 165 },
		{ 3209, 3201 },
		{ 2056, 2053 },
		{ 2058, 2057 },
		{ 2282, 2279 },
		{ 2284, 2283 },
		{ 2051, 2050 },
		{ 2277, 2276 },
		{ 2009, 1999 },
		{ 2723, 2693 },
		{ 2724, 2694 },
		{ 2271, 2270 },
		{ 2125, 2104 },
		{ 168, 164 },
		{ 2283, 2281 },
		{ 2053, 2049 },
		{ 2368, 2336 },
		{ 105, 90 },
		{ 1915, 1893 },
		{ 2964, 2960 },
		{ 172, 170 },
		{ 2279, 2275 },
		{ 2057, 2055 },
		{ 2968, 2966 },
		{ 0, 3188 },
		{ 0, 1838 },
		{ 0, 2992 },
		{ 0, 3087 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 0, 2701 },
		{ 2636, 2636 },
		{ 2636, 2636 },
		{ 0, 2997 },
		{ 0, 2518 },
		{ 0, 2704 },
		{ 0, 1312 },
		{ 2668, 2668 },
		{ 2668, 2668 },
		{ 2846, 2846 },
		{ 2846, 2846 },
		{ 0, 3097 },
		{ 2639, 2639 },
		{ 2639, 2639 },
		{ 2794, 2794 },
		{ 2794, 2794 },
		{ 0, 3100 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 0, 2918 },
		{ 0, 3008 },
		{ 2905, 2905 },
		{ 2919, 2919 },
		{ 2919, 2919 },
		{ 2636, 2636 },
		{ 0, 3105 },
		{ 2920, 2920 },
		{ 2920, 2920 },
		{ 0, 2749 },
		{ 0, 2922 },
		{ 2668, 2668 },
		{ 0, 2611 },
		{ 2846, 2846 },
		{ 2923, 2923 },
		{ 2923, 2923 },
		{ 2639, 2639 },
		{ 0, 3015 },
		{ 2794, 2794 },
		{ 2797, 2797 },
		{ 2797, 2797 },
		{ 2917, 2917 },
		{ 2798, 2798 },
		{ 2798, 2798 },
		{ 2927, 2927 },
		{ 2927, 2927 },
		{ 2919, 2919 },
		{ 0, 2298 },
		{ 0, 3116 },
		{ 0, 2854 },
		{ 2920, 2920 },
		{ 2058, 2058 },
		{ 2059, 2058 },
		{ 2674, 2674 },
		{ 2674, 2674 },
		{ 0, 3022 },
		{ 0, 1333 },
		{ 2923, 2923 },
		{ 2858, 2858 },
		{ 2858, 2858 },
		{ 2859, 2859 },
		{ 2859, 2859 },
		{ 2797, 2797 },
		{ 2935, 2935 },
		{ 2935, 2935 },
		{ 2798, 2798 },
		{ 0, 3130 },
		{ 2927, 2927 },
		{ 2937, 2937 },
		{ 2937, 2937 },
		{ 2938, 2938 },
		{ 2938, 2938 },
		{ 2566, 2566 },
		{ 2566, 2566 },
		{ 2058, 2058 },
		{ 0, 2758 },
		{ 2674, 2674 },
		{ 2284, 2284 },
		{ 2285, 2284 },
		{ 2679, 2679 },
		{ 2679, 2679 },
		{ 2858, 2858 },
		{ 0, 3134 },
		{ 2859, 2859 },
		{ 0, 2809 },
		{ 0, 2866 },
		{ 2935, 2935 },
		{ 2945, 2945 },
		{ 2945, 2945 },
		{ 0, 3138 },
		{ 0, 2810 },
		{ 2937, 2937 },
		{ 0, 3141 },
		{ 2938, 2938 },
		{ 0, 2869 },
		{ 2566, 2566 },
		{ 2867, 2867 },
		{ 2867, 2867 },
		{ 2949, 2949 },
		{ 2949, 2949 },
		{ 2284, 2284 },
		{ 0, 2811 },
		{ 2679, 2679 },
		{ 0, 3043 },
		{ 0, 3146 },
		{ 0, 2717 },
		{ 0, 2872 },
		{ 0, 2873 },
		{ 0, 2718 },
		{ 0, 2568 },
		{ 2945, 2945 },
		{ 2955, 2955 },
		{ 2955, 2955 },
		{ 0, 2815 },
		{ 0, 1364 },
		{ 0, 3052 },
		{ 0, 1822 },
		{ 0, 2568 },
		{ 0, 3157 },
		{ 2867, 2867 },
		{ 0, 1380 },
		{ 2949, 2949 },
		{ 2766, 2766 },
		{ 2766, 2766 },
		{ 0, 2684 },
		{ 0, 2439 },
		{ 0, 1833 },
		{ 2969, 2969 },
		{ 2970, 2969 },
		{ 0, 1348 },
		{ 0, 2625 },
		{ 2886, 2886 },
		{ 2886, 2886 },
		{ 0, 2599 },
		{ 2955, 2955 },
		{ 0, 3043 },
		{ 2476, 2476 },
		{ 2476, 2476 },
		{ 0, 2977 },
		{ 0, 3068 },
		{ 0, 2730 },
		{ 0, 1359 },
		{ 0, 2692 },
		{ 0, 3177 },
		{ 0, 2554 },
		{ 2766, 2766 },
		{ 0, 2555 },
		{ 2780, 2780 },
		{ 2780, 2780 },
		{ 0, 2556 },
		{ 2969, 2969 },
		{ 2737, 2737 },
		{ 2737, 2737 },
		{ 0, 2697 },
		{ 2886, 2886 },
		{ 0, 3083 },
		{ 0, 2439 },
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 2476, 2476 },
		{ 173, 173 },
		{ 174, 173 },
		{ 1291, 1291 },
		{ 1593, 1593 },
		{ 2295, 2294 },
		{ 1435, 1434 },
		{ 2052, 2048 },
		{ 2278, 2280 },
		{ 2963, 2959 },
		{ 3213, 3209 },
		{ 2780, 2780 },
		{ 167, 169 },
		{ 0, 2335 },
		{ 0, 0 },
		{ 2737, 2737 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2899, 2899 },
		{ 0, 0 },
		{ 0, 0 },
		{ 173, 173 },
		{ 0, 0 },
		{ 1291, 1291 },
		{ 1593, 1593 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -70, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 46, 433, 0 },
		{ -179, 3174, 0 },
		{ 5, 0, 0 },
		{ -1290, 1095, -31 },
		{ 7, 0, -31 },
		{ -1294, 2088, -33 },
		{ 9, 0, -33 },
		{ -1307, 3496, 159 },
		{ 11, 0, 159 },
		{ -1328, 3503, 163 },
		{ 13, 0, 163 },
		{ -1341, 3497, 171 },
		{ 15, 0, 171 },
		{ -1376, 3648, 0 },
		{ 17, 0, 0 },
		{ -1391, 3504, 155 },
		{ 19, 0, 155 },
		{ -1419, 3519, 23 },
		{ 21, 0, 23 },
		{ -1461, 547, 0 },
		{ 23, 0, 0 },
		{ -1687, 3650, 0 },
		{ 25, 0, 0 },
		{ -1714, 3520, 0 },
		{ 27, 0, 0 },
		{ -1740, 3521, 0 },
		{ 29, 0, 0 },
		{ -1764, 3498, 0 },
		{ 31, 0, 0 },
		{ -1790, 3508, 0 },
		{ 33, 0, 0 },
		{ -1816, 3510, 175 },
		{ 35, 0, 175 },
		{ -1850, 3645, 282 },
		{ 37, 0, 282 },
		{ 40, 218, 0 },
		{ -1886, 661, 0 },
		{ 42, 216, 0 },
		{ -2075, 205, 0 },
		{ -2287, 3646, 0 },
		{ 43, 0, 0 },
		{ 46, 14, 0 },
		{ -88, 319, 0 },
		{ -2972, 3514, 167 },
		{ 47, 0, 167 },
		{ -2990, 3649, 190 },
		{ 49, 0, 190 },
		{ 3034, 1615, 0 },
		{ 51, 0, 0 },
		{ -3036, 3652, 288 },
		{ 53, 0, 288 },
		{ -3063, 3655, 193 },
		{ 55, 0, 193 },
		{ -3081, 3502, 186 },
		{ 57, 0, 186 },
		{ -3128, 3647, 179 },
		{ 59, 0, 179 },
		{ -3169, 3525, 185 },
		{ 61, 0, 185 },
		{ 46, 1, 0 },
		{ 63, 0, 0 },
		{ -3220, 2054, 0 },
		{ 65, 0, 0 },
		{ -3230, 2108, 45 },
		{ 67, 0, 45 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 448 },
		{ 3201, 5198, 455 },
		{ 0, 0, 260 },
		{ 0, 0, 262 },
		{ 159, 1301, 279 },
		{ 159, 1419, 279 },
		{ 159, 1384, 279 },
		{ 159, 1406, 279 },
		{ 159, 1410, 279 },
		{ 159, 1441, 279 },
		{ 159, 1447, 279 },
		{ 159, 1469, 279 },
		{ 3294, 3198, 455 },
		{ 159, 1480, 279 },
		{ 3294, 1904, 278 },
		{ 104, 3009, 455 },
		{ 159, 0, 279 },
		{ 0, 0, 455 },
		{ -89, 8, 256 },
		{ -90, 5242, 0 },
		{ 159, 1471, 279 },
		{ 159, 1515, 279 },
		{ 159, 1484, 279 },
		{ 159, 1498, 279 },
		{ 159, 1480, 279 },
		{ 159, 1480, 279 },
		{ 159, 1487, 279 },
		{ 159, 1502, 279 },
		{ 159, 1521, 279 },
		{ 3253, 2614, 0 },
		{ 159, 833, 279 },
		{ 3294, 2057, 275 },
		{ 119, 1618, 0 },
		{ 3294, 2034, 276 },
		{ 3201, 5216, 0 },
		{ 159, 842, 279 },
		{ 159, 840, 279 },
		{ 159, 841, 279 },
		{ 159, 837, 279 },
		{ 159, 0, 267 },
		{ 159, 882, 279 },
		{ 159, 884, 279 },
		{ 159, 868, 279 },
		{ 159, 873, 279 },
		{ 3291, 3272, 0 },
		{ 159, 881, 279 },
		{ 133, 1709, 0 },
		{ 119, 0, 0 },
		{ 3203, 3016, 277 },
		{ 135, 1617, 0 },
		{ 0, 0, 258 },
		{ 159, 913, 263 },
		{ 159, 915, 279 },
		{ 159, 907, 279 },
		{ 159, 912, 279 },
		{ 159, 910, 279 },
		{ 159, 903, 279 },
		{ 159, 0, 270 },
		{ 159, 904, 279 },
		{ 0, 0, 272 },
		{ 159, 910, 279 },
		{ 133, 0, 0 },
		{ 3203, 2928, 275 },
		{ 135, 0, 0 },
		{ 3203, 2986, 276 },
		{ 159, 925, 279 },
		{ 159, 922, 279 },
		{ 159, 923, 279 },
		{ 159, 958, 279 },
		{ 159, 1054, 279 },
		{ 159, 0, 269 },
		{ 159, 1163, 279 },
		{ 159, 1236, 279 },
		{ 159, 1228, 279 },
		{ 159, 0, 265 },
		{ 159, 1308, 279 },
		{ 159, 0, 266 },
		{ 159, 0, 268 },
		{ 159, 1276, 279 },
		{ 159, 1276, 279 },
		{ 159, 0, 264 },
		{ 159, 1297, 279 },
		{ 159, 0, 271 },
		{ 159, 890, 279 },
		{ 159, 1323, 279 },
		{ 0, 0, 274 },
		{ 159, 1306, 279 },
		{ 159, 1308, 279 },
		{ 3311, 1376, 273 },
		{ 3201, 5209, 455 },
		{ 165, 0, 260 },
		{ 0, 0, 261 },
		{ -163, 22, 256 },
		{ -164, 5238, 0 },
		{ 3257, 5220, 0 },
		{ 3201, 5195, 0 },
		{ 0, 0, 257 },
		{ 3201, 5210, 0 },
		{ -169, 5457, 0 },
		{ -170, 5245, 0 },
		{ 173, 0, 258 },
		{ 3201, 5213, 0 },
		{ 3257, 5447, 0 },
		{ 0, 0, 259 },
		{ 3258, 1861, 153 },
		{ 2187, 4565, 153 },
		{ 3127, 4967, 153 },
		{ 3258, 4767, 153 },
		{ 0, 0, 153 },
		{ 3243, 3759, 0 },
		{ 2202, 3152, 0 },
		{ 3243, 4007, 0 },
		{ 3243, 3648, 0 },
		{ 3231, 3731, 0 },
		{ 2187, 4490, 0 },
		{ 3248, 3637, 0 },
		{ 2187, 4579, 0 },
		{ 3218, 3830, 0 },
		{ 3252, 3604, 0 },
		{ 3218, 3684, 0 },
		{ 3258, 4734, 0 },
		{ 3062, 4779, 0 },
		{ 3248, 3663, 0 },
		{ 2202, 4149, 0 },
		{ 3127, 4975, 0 },
		{ 2133, 3853, 0 },
		{ 2133, 3855, 0 },
		{ 2155, 3487, 0 },
		{ 2136, 4226, 0 },
		{ 2117, 4301, 0 },
		{ 2136, 4242, 0 },
		{ 2155, 3490, 0 },
		{ 2155, 3524, 0 },
		{ 3248, 3721, 0 },
		{ 3168, 4363, 0 },
		{ 3243, 4048, 0 },
		{ 2187, 4547, 0 },
		{ 1260, 4475, 0 },
		{ 2133, 3887, 0 },
		{ 2155, 3532, 0 },
		{ 2155, 3534, 0 },
		{ 3127, 4825, 0 },
		{ 2133, 3881, 0 },
		{ 3231, 4181, 0 },
		{ 3243, 3998, 0 },
		{ 2202, 4111, 0 },
		{ 2676, 4709, 0 },
		{ 3127, 4025, 0 },
		{ 3168, 4353, 0 },
		{ 3218, 3939, 0 },
		{ 3218, 3832, 0 },
		{ 3218, 3971, 0 },
		{ 2095, 3678, 0 },
		{ 3127, 5043, 0 },
		{ 3243, 4066, 0 },
		{ 3168, 4018, 0 },
		{ 2202, 4108, 0 },
		{ 2155, 3548, 0 },
		{ 2155, 3466, 0 },
		{ 3252, 3812, 0 },
		{ 3231, 4093, 0 },
		{ 2095, 3679, 0 },
		{ 2117, 4257, 0 },
		{ 3127, 4927, 0 },
		{ 2187, 4521, 0 },
		{ 2187, 4522, 0 },
		{ 3243, 4060, 0 },
		{ 2155, 3467, 0 },
		{ 2187, 4578, 0 },
		{ 3243, 4071, 0 },
		{ 2676, 4708, 0 },
		{ 3127, 4997, 0 },
		{ 3252, 3741, 0 },
		{ 3168, 4340, 0 },
		{ 3218, 3964, 0 },
		{ 2155, 3468, 0 },
		{ 3252, 3702, 0 },
		{ 3243, 4003, 0 },
		{ 1260, 4467, 0 },
		{ 2117, 4270, 0 },
		{ 3168, 4394, 0 },
		{ 2202, 4025, 0 },
		{ 1145, 3614, 0 },
		{ 3243, 4070, 0 },
		{ 3231, 4172, 0 },
		{ 3127, 4823, 0 },
		{ 2676, 4621, 0 },
		{ 2155, 3469, 0 },
		{ 2202, 4100, 0 },
		{ 3252, 3803, 0 },
		{ 1005, 4310, 0 },
		{ 2187, 4526, 0 },
		{ 2095, 3657, 0 },
		{ 2095, 3669, 0 },
		{ 2187, 4571, 0 },
		{ 3218, 3924, 0 },
		{ 2155, 3471, 0 },
		{ 3062, 4791, 0 },
		{ 3231, 4206, 0 },
		{ 3252, 3766, 0 },
		{ 2133, 3841, 0 },
		{ 2223, 4021, 0 },
		{ 2155, 3472, 0 },
		{ 3168, 4347, 0 },
		{ 3218, 3938, 0 },
		{ 2187, 4573, 0 },
		{ 2676, 4640, 0 },
		{ 2187, 4574, 0 },
		{ 3127, 4899, 0 },
		{ 3127, 4907, 0 },
		{ 2117, 4262, 0 },
		{ 2676, 4611, 0 },
		{ 2155, 3473, 0 },
		{ 3127, 4979, 0 },
		{ 3168, 4406, 0 },
		{ 2117, 4289, 0 },
		{ 3168, 4321, 0 },
		{ 2676, 4710, 0 },
		{ 3127, 4895, 0 },
		{ 2133, 3876, 0 },
		{ 3218, 3969, 0 },
		{ 2187, 4556, 0 },
		{ 3127, 4965, 0 },
		{ 1260, 4471, 0 },
		{ 3168, 4382, 0 },
		{ 1145, 3616, 0 },
		{ 2223, 4447, 0 },
		{ 2136, 4230, 0 },
		{ 3218, 3921, 0 },
		{ 2155, 3474, 0 },
		{ 3127, 4829, 0 },
		{ 2187, 4515, 0 },
		{ 2155, 3475, 0 },
		{ 0, 0, 78 },
		{ 3243, 3797, 0 },
		{ 3252, 3800, 0 },
		{ 2187, 4546, 0 },
		{ 3168, 4390, 0 },
		{ 3258, 4763, 0 },
		{ 2187, 4548, 0 },
		{ 2155, 3476, 0 },
		{ 2155, 3477, 0 },
		{ 3252, 3733, 0 },
		{ 2133, 3834, 0 },
		{ 2117, 4263, 0 },
		{ 3252, 3739, 0 },
		{ 2155, 3478, 0 },
		{ 3168, 4384, 0 },
		{ 2187, 4510, 0 },
		{ 3243, 4077, 0 },
		{ 3243, 4046, 0 },
		{ 2136, 4221, 0 },
		{ 3127, 4977, 0 },
		{ 3218, 3950, 0 },
		{ 900, 3631, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2133, 3866, 0 },
		{ 3127, 5073, 0 },
		{ 3168, 4358, 0 },
		{ 2117, 4290, 0 },
		{ 3252, 3760, 0 },
		{ 3218, 3983, 0 },
		{ 0, 0, 76 },
		{ 2155, 3480, 0 },
		{ 2133, 3882, 0 },
		{ 0, 0, 133 },
		{ 3252, 3773, 0 },
		{ 3168, 4417, 0 },
		{ 3252, 3774, 0 },
		{ 3127, 4971, 0 },
		{ 3218, 3947, 0 },
		{ 1260, 4477, 0 },
		{ 2117, 4299, 0 },
		{ 2133, 3835, 0 },
		{ 3231, 4153, 0 },
		{ 2187, 4530, 0 },
		{ 3127, 4815, 0 },
		{ 3258, 4768, 0 },
		{ 3218, 3957, 0 },
		{ 0, 0, 70 },
		{ 3218, 3958, 0 },
		{ 3127, 4891, 0 },
		{ 1260, 4468, 0 },
		{ 3168, 4389, 0 },
		{ 2187, 4557, 0 },
		{ 0, 0, 81 },
		{ 3231, 4184, 0 },
		{ 3252, 3776, 0 },
		{ 3218, 3966, 0 },
		{ 3243, 4001, 0 },
		{ 3243, 4065, 0 },
		{ 2155, 3481, 0 },
		{ 3168, 4351, 0 },
		{ 2187, 4503, 0 },
		{ 2155, 3482, 0 },
		{ 2133, 3859, 0 },
		{ 2168, 3608, 0 },
		{ 3231, 4178, 0 },
		{ 3252, 3804, 0 },
		{ 3218, 3928, 0 },
		{ 3127, 4847, 0 },
		{ 3252, 3809, 0 },
		{ 2155, 3483, 0 },
		{ 3168, 4398, 0 },
		{ 2187, 4554, 0 },
		{ 3252, 3813, 0 },
		{ 3218, 3948, 0 },
		{ 3168, 4327, 0 },
		{ 1081, 4423, 0 },
		{ 0, 0, 8 },
		{ 2133, 3891, 0 },
		{ 2136, 4219, 0 },
		{ 3168, 4349, 0 },
		{ 2168, 3594, 0 },
		{ 2155, 3484, 0 },
		{ 2676, 4617, 0 },
		{ 2187, 4502, 0 },
		{ 2133, 3838, 0 },
		{ 2187, 4507, 0 },
		{ 2187, 4508, 0 },
		{ 2136, 4232, 0 },
		{ 2155, 3485, 0 },
		{ 3252, 3743, 0 },
		{ 3252, 3645, 0 },
		{ 2187, 4523, 0 },
		{ 3248, 3688, 0 },
		{ 3218, 3902, 0 },
		{ 1260, 4481, 0 },
		{ 3231, 4207, 0 },
		{ 2155, 3486, 0 },
		{ 2223, 4426, 0 },
		{ 2202, 3424, 0 },
		{ 2155, 3488, 0 },
		{ 3127, 5008, 0 },
		{ 1260, 4480, 0 },
		{ 2202, 4137, 0 },
		{ 3258, 3597, 0 },
		{ 2168, 3596, 0 },
		{ 2136, 4233, 0 },
		{ 2133, 3888, 0 },
		{ 3252, 3797, 0 },
		{ 2187, 4494, 0 },
		{ 0, 0, 123 },
		{ 2155, 3489, 0 },
		{ 2202, 4139, 0 },
		{ 2094, 3629, 0 },
		{ 3168, 4386, 0 },
		{ 3243, 4040, 0 },
		{ 3231, 4198, 0 },
		{ 3127, 4969, 0 },
		{ 2187, 4520, 0 },
		{ 0, 0, 7 },
		{ 2136, 4235, 0 },
		{ 0, 0, 6 },
		{ 3168, 4395, 0 },
		{ 0, 0, 128 },
		{ 3231, 4204, 0 },
		{ 2187, 4525, 0 },
		{ 3258, 2103, 0 },
		{ 2155, 3491, 0 },
		{ 3231, 4210, 0 },
		{ 3243, 4059, 0 },
		{ 0, 0, 132 },
		{ 2155, 3492, 0 },
		{ 2187, 4549, 0 },
		{ 3258, 3623, 0 },
		{ 2187, 4555, 0 },
		{ 2676, 4702, 0 },
		{ 2202, 4088, 0 },
		{ 0, 0, 77 },
		{ 2117, 4296, 0 },
		{ 2155, 3496, 115 },
		{ 2155, 3497, 116 },
		{ 3127, 4936, 0 },
		{ 3168, 4368, 0 },
		{ 2155, 3498, 0 },
		{ 3218, 3906, 0 },
		{ 3243, 4080, 0 },
		{ 3243, 4039, 0 },
		{ 3218, 3908, 0 },
		{ 1260, 4469, 0 },
		{ 3243, 4042, 0 },
		{ 3218, 3909, 0 },
		{ 3248, 3706, 0 },
		{ 2202, 4131, 0 },
		{ 3168, 4416, 0 },
		{ 2187, 4517, 0 },
		{ 2155, 3502, 0 },
		{ 3252, 3759, 0 },
		{ 3127, 4831, 0 },
		{ 0, 0, 114 },
		{ 3168, 4329, 0 },
		{ 3062, 4777, 0 },
		{ 3168, 4334, 0 },
		{ 2202, 4148, 0 },
		{ 3218, 3934, 0 },
		{ 3168, 4348, 0 },
		{ 0, 0, 9 },
		{ 2155, 3503, 0 },
		{ 3168, 4350, 0 },
		{ 2168, 3609, 0 },
		{ 2223, 4446, 0 },
		{ 0, 0, 112 },
		{ 2133, 3897, 0 },
		{ 3231, 4177, 0 },
		{ 3243, 4015, 0 },
		{ 2202, 4032, 0 },
		{ 3231, 3614, 0 },
		{ 3168, 4383, 0 },
		{ 3248, 3667, 0 },
		{ 3168, 4385, 0 },
		{ 3248, 3671, 0 },
		{ 3243, 4045, 0 },
		{ 2187, 4585, 0 },
		{ 3252, 3777, 0 },
		{ 3218, 3959, 0 },
		{ 3248, 3640, 0 },
		{ 3231, 4161, 0 },
		{ 3252, 3786, 0 },
		{ 3168, 4320, 0 },
		{ 3252, 3701, 0 },
		{ 3127, 4921, 0 },
		{ 2155, 3504, 0 },
		{ 3127, 4931, 0 },
		{ 3218, 3982, 0 },
		{ 2187, 4518, 0 },
		{ 3243, 4006, 0 },
		{ 3243, 3994, 0 },
		{ 2117, 4294, 0 },
		{ 2133, 3867, 0 },
		{ 3243, 4023, 0 },
		{ 2155, 3506, 102 },
		{ 3218, 3905, 0 },
		{ 2202, 4116, 0 },
		{ 2155, 3508, 0 },
		{ 2155, 3509, 0 },
		{ 3248, 3714, 0 },
		{ 2202, 4146, 0 },
		{ 2676, 4619, 0 },
		{ 2155, 3510, 0 },
		{ 2133, 3890, 0 },
		{ 0, 0, 111 },
		{ 2676, 4693, 0 },
		{ 3127, 4864, 0 },
		{ 3252, 3821, 0 },
		{ 3252, 3729, 0 },
		{ 0, 0, 125 },
		{ 0, 0, 127 },
		{ 3231, 4205, 0 },
		{ 2202, 4103, 0 },
		{ 2133, 3828, 0 },
		{ 2187, 3741, 0 },
		{ 3252, 3736, 0 },
		{ 2187, 4583, 0 },
		{ 2155, 3512, 0 },
		{ 2187, 4487, 0 },
		{ 3168, 4413, 0 },
		{ 3231, 4162, 0 },
		{ 2155, 3513, 0 },
		{ 2223, 4435, 0 },
		{ 3248, 3710, 0 },
		{ 2676, 4613, 0 },
		{ 2155, 3514, 0 },
		{ 3127, 5037, 0 },
		{ 2133, 3858, 0 },
		{ 1005, 4309, 0 },
		{ 3252, 3756, 0 },
		{ 3231, 4189, 0 },
		{ 2202, 4087, 0 },
		{ 2676, 4707, 0 },
		{ 3252, 3757, 0 },
		{ 2095, 3649, 0 },
		{ 2155, 3515, 0 },
		{ 3168, 4357, 0 },
		{ 3243, 4054, 0 },
		{ 2133, 3877, 0 },
		{ 3127, 4905, 0 },
		{ 3252, 3762, 0 },
		{ 2202, 4123, 0 },
		{ 2168, 3612, 0 },
		{ 3218, 3904, 0 },
		{ 2133, 3884, 0 },
		{ 2202, 4141, 0 },
		{ 2136, 4217, 0 },
		{ 3258, 3603, 0 },
		{ 2155, 3519, 0 },
		{ 0, 0, 71 },
		{ 2155, 3520, 0 },
		{ 3243, 4083, 0 },
		{ 3218, 3918, 0 },
		{ 3243, 4030, 0 },
		{ 3218, 3919, 0 },
		{ 2155, 3521, 117 },
		{ 2117, 4282, 0 },
		{ 3127, 5053, 0 },
		{ 2202, 4110, 0 },
		{ 2136, 4218, 0 },
		{ 3218, 3923, 0 },
		{ 2133, 3895, 0 },
		{ 2133, 3896, 0 },
		{ 2117, 4300, 0 },
		{ 2136, 4229, 0 },
		{ 3127, 4849, 0 },
		{ 3243, 3999, 0 },
		{ 3248, 3703, 0 },
		{ 3168, 4354, 0 },
		{ 3252, 3784, 0 },
		{ 2133, 3833, 0 },
		{ 0, 0, 129 },
		{ 2155, 3523, 0 },
		{ 3062, 4787, 0 },
		{ 2136, 4216, 0 },
		{ 3252, 3787, 0 },
		{ 3231, 4190, 0 },
		{ 0, 0, 124 },
		{ 0, 0, 113 },
		{ 2133, 3836, 0 },
		{ 3218, 3952, 0 },
		{ 3252, 3795, 0 },
		{ 2202, 3344, 0 },
		{ 3258, 3678, 0 },
		{ 3168, 4391, 0 },
		{ 3231, 4209, 0 },
		{ 2155, 3525, 0 },
		{ 3168, 4397, 0 },
		{ 2117, 4264, 0 },
		{ 3243, 4029, 0 },
		{ 2187, 4568, 0 },
		{ 3127, 5063, 0 },
		{ 3127, 5065, 0 },
		{ 2133, 3856, 0 },
		{ 3127, 5075, 0 },
		{ 3168, 4414, 0 },
		{ 3127, 4819, 0 },
		{ 3218, 3967, 0 },
		{ 3231, 4163, 0 },
		{ 2155, 3526, 0 },
		{ 2187, 4581, 0 },
		{ 3218, 3970, 0 },
		{ 2155, 3527, 0 },
		{ 3218, 3975, 0 },
		{ 2187, 4489, 0 },
		{ 3168, 4342, 0 },
		{ 2187, 4492, 0 },
		{ 3218, 3976, 0 },
		{ 2187, 4497, 0 },
		{ 2133, 3861, 0 },
		{ 2094, 3624, 0 },
		{ 3231, 4197, 0 },
		{ 2155, 3528, 0 },
		{ 3258, 4605, 0 },
		{ 2676, 4691, 0 },
		{ 2187, 4509, 0 },
		{ 2136, 4240, 0 },
		{ 2187, 4514, 0 },
		{ 2136, 4241, 0 },
		{ 3243, 4009, 0 },
		{ 3127, 4991, 0 },
		{ 3252, 3815, 0 },
		{ 3243, 4068, 0 },
		{ 0, 0, 104 },
		{ 3252, 3816, 0 },
		{ 3168, 4373, 0 },
		{ 3168, 4374, 0 },
		{ 3127, 5061, 0 },
		{ 2155, 3529, 0 },
		{ 2155, 3530, 0 },
		{ 3127, 5067, 0 },
		{ 3127, 5069, 0 },
		{ 3127, 5071, 0 },
		{ 2136, 4222, 0 },
		{ 2133, 3883, 0 },
		{ 0, 0, 138 },
		{ 3243, 4081, 0 },
		{ 0, 0, 126 },
		{ 0, 0, 130 },
		{ 3127, 4817, 0 },
		{ 2676, 4705, 0 },
		{ 1145, 3615, 0 },
		{ 2155, 3531, 0 },
		{ 3168, 3431, 0 },
		{ 3218, 3922, 0 },
		{ 2136, 4236, 0 },
		{ 1260, 4459, 0 },
		{ 3127, 4857, 0 },
		{ 3243, 4032, 0 },
		{ 3243, 4034, 0 },
		{ 3243, 4035, 0 },
		{ 3231, 4186, 0 },
		{ 2676, 4682, 0 },
		{ 2223, 4442, 0 },
		{ 3231, 4187, 0 },
		{ 3248, 3708, 0 },
		{ 2117, 4256, 0 },
		{ 1260, 4458, 0 },
		{ 3252, 3742, 0 },
		{ 2117, 4261, 0 },
		{ 2133, 3894, 0 },
		{ 2155, 3533, 0 },
		{ 2136, 4223, 0 },
		{ 2117, 4265, 0 },
		{ 2187, 4496, 0 },
		{ 2223, 4441, 0 },
		{ 2202, 4102, 0 },
		{ 3218, 3935, 0 },
		{ 3127, 5035, 0 },
		{ 2202, 4105, 0 },
		{ 0, 0, 65 },
		{ 0, 0, 66 },
		{ 3127, 5039, 0 },
		{ 3218, 3937, 0 },
		{ 0, 0, 75 },
		{ 0, 0, 121 },
		{ 2095, 3675, 0 },
		{ 3252, 3745, 0 },
		{ 3248, 3717, 0 },
		{ 2133, 3830, 0 },
		{ 3248, 3718, 0 },
		{ 3252, 3758, 0 },
		{ 3168, 4371, 0 },
		{ 3218, 3953, 0 },
		{ 0, 0, 106 },
		{ 3218, 3954, 0 },
		{ 0, 0, 108 },
		{ 3243, 4072, 0 },
		{ 3218, 3955, 0 },
		{ 3231, 4183, 0 },
		{ 1081, 4421, 0 },
		{ 2187, 4532, 0 },
		{ 0, 0, 136 },
		{ 2168, 3604, 0 },
		{ 2168, 3605, 0 },
		{ 3252, 3761, 0 },
		{ 1260, 4479, 0 },
		{ 2223, 4158, 0 },
		{ 3218, 3960, 0 },
		{ 1005, 4307, 0 },
		{ 2117, 4291, 0 },
		{ 0, 0, 122 },
		{ 0, 0, 137 },
		{ 3218, 3961, 0 },
		{ 3218, 3962, 0 },
		{ 0, 0, 152 },
		{ 2133, 3840, 0 },
		{ 3258, 4329, 0 },
		{ 3127, 4925, 0 },
		{ 1260, 4474, 0 },
		{ 3127, 4929, 0 },
		{ 2187, 4576, 0 },
		{ 3258, 4727, 0 },
		{ 3218, 3965, 0 },
		{ 3258, 4750, 0 },
		{ 3248, 3693, 0 },
		{ 3248, 3702, 0 },
		{ 3231, 3427, 0 },
		{ 2155, 3535, 0 },
		{ 2187, 4485, 0 },
		{ 3168, 4331, 0 },
		{ 3127, 4993, 0 },
		{ 3127, 4995, 0 },
		{ 3168, 4333, 0 },
		{ 2202, 4126, 0 },
		{ 3168, 4335, 0 },
		{ 2202, 4129, 0 },
		{ 2202, 4027, 0 },
		{ 3168, 4345, 0 },
		{ 2155, 3536, 0 },
		{ 2676, 4652, 0 },
		{ 2155, 3537, 0 },
		{ 3243, 4051, 0 },
		{ 2155, 3538, 0 },
		{ 2136, 4227, 0 },
		{ 3243, 4058, 0 },
		{ 2117, 4293, 0 },
		{ 3168, 4356, 0 },
		{ 2155, 3539, 0 },
		{ 2155, 3540, 0 },
		{ 3243, 4061, 0 },
		{ 3258, 4757, 0 },
		{ 1260, 4473, 0 },
		{ 2202, 4089, 0 },
		{ 3218, 3900, 0 },
		{ 3127, 4835, 0 },
		{ 3127, 4843, 0 },
		{ 2187, 4524, 0 },
		{ 2136, 4239, 0 },
		{ 2676, 4644, 0 },
		{ 0, 0, 134 },
		{ 3218, 3901, 0 },
		{ 2187, 4527, 0 },
		{ 2187, 4528, 0 },
		{ 3168, 4377, 0 },
		{ 3168, 4381, 0 },
		{ 2187, 4535, 0 },
		{ 3127, 4917, 0 },
		{ 3127, 4919, 0 },
		{ 2187, 4544, 0 },
		{ 2155, 3541, 0 },
		{ 2202, 4104, 0 },
		{ 3252, 3788, 0 },
		{ 3252, 3789, 0 },
		{ 2187, 4553, 0 },
		{ 3248, 3674, 0 },
		{ 3248, 3691, 0 },
		{ 2117, 4279, 0 },
		{ 3258, 4759, 0 },
		{ 3252, 3799, 0 },
		{ 2155, 3542, 68 },
		{ 3252, 3802, 0 },
		{ 3127, 4989, 0 },
		{ 2202, 4128, 0 },
		{ 2155, 3543, 0 },
		{ 2155, 3544, 0 },
		{ 2223, 4434, 0 },
		{ 3168, 4409, 0 },
		{ 3258, 4761, 0 },
		{ 3231, 4166, 0 },
		{ 3252, 3805, 0 },
		{ 3252, 3808, 0 },
		{ 1145, 3620, 0 },
		{ 2117, 4247, 0 },
		{ 3218, 3930, 0 },
		{ 2168, 3591, 0 },
		{ 2095, 3682, 0 },
		{ 2095, 3683, 0 },
		{ 3243, 4044, 0 },
		{ 2133, 3832, 0 },
		{ 1260, 4454, 0 },
		{ 3248, 3715, 0 },
		{ 3218, 3940, 0 },
		{ 3258, 4738, 0 },
		{ 3258, 4740, 0 },
		{ 2187, 4505, 0 },
		{ 0, 0, 69 },
		{ 0, 0, 72 },
		{ 3127, 4827, 0 },
		{ 3168, 4322, 0 },
		{ 2223, 4440, 0 },
		{ 2117, 4272, 0 },
		{ 3218, 3943, 0 },
		{ 1260, 4472, 0 },
		{ 3218, 3945, 0 },
		{ 0, 0, 118 },
		{ 3252, 3817, 0 },
		{ 3252, 3820, 0 },
		{ 3218, 3949, 0 },
		{ 0, 0, 110 },
		{ 2155, 3545, 0 },
		{ 3127, 4897, 0 },
		{ 0, 0, 119 },
		{ 0, 0, 120 },
		{ 2202, 4112, 0 },
		{ 2117, 4295, 0 },
		{ 3231, 4213, 0 },
		{ 1005, 4306, 0 },
		{ 2136, 4231, 0 },
		{ 1260, 4465, 0 },
		{ 3252, 3822, 0 },
		{ 3062, 4788, 0 },
		{ 0, 0, 3 },
		{ 2187, 4529, 0 },
		{ 3258, 4732, 0 },
		{ 2676, 4704, 0 },
		{ 3127, 4963, 0 },
		{ 3231, 4156, 0 },
		{ 3168, 4376, 0 },
		{ 3252, 3823, 0 },
		{ 3168, 4380, 0 },
		{ 2187, 4545, 0 },
		{ 2155, 3547, 0 },
		{ 2136, 4238, 0 },
		{ 2676, 4615, 0 },
		{ 2133, 3842, 0 },
		{ 2133, 3846, 0 },
		{ 2187, 4550, 0 },
		{ 3231, 4168, 0 },
		{ 1081, 4422, 0 },
		{ 2187, 3433, 0 },
		{ 3168, 4388, 0 },
		{ 2202, 4132, 0 },
		{ 0, 0, 79 },
		{ 2187, 4562, 0 },
		{ 0, 0, 87 },
		{ 3127, 5049, 0 },
		{ 2187, 4563, 0 },
		{ 3127, 5059, 0 },
		{ 3252, 3735, 0 },
		{ 2187, 4566, 0 },
		{ 3248, 3722, 0 },
		{ 3258, 4725, 0 },
		{ 2187, 4570, 0 },
		{ 2202, 4140, 0 },
		{ 2117, 4281, 0 },
		{ 3252, 3737, 0 },
		{ 2117, 4286, 0 },
		{ 3168, 4399, 0 },
		{ 3231, 4185, 0 },
		{ 2187, 4580, 0 },
		{ 2202, 4142, 0 },
		{ 3168, 4410, 0 },
		{ 2187, 4584, 0 },
		{ 0, 0, 74 },
		{ 2202, 4143, 0 },
		{ 2202, 4145, 0 },
		{ 3127, 4837, 0 },
		{ 2136, 4228, 0 },
		{ 3252, 3738, 0 },
		{ 3231, 4191, 0 },
		{ 2187, 4491, 0 },
		{ 2202, 4147, 0 },
		{ 2187, 4493, 0 },
		{ 2155, 3549, 0 },
		{ 3168, 4332, 0 },
		{ 3243, 4027, 0 },
		{ 3127, 4901, 0 },
		{ 2136, 4234, 0 },
		{ 2117, 4251, 0 },
		{ 3127, 4909, 0 },
		{ 2133, 3864, 0 },
		{ 3258, 4769, 0 },
		{ 2133, 3865, 0 },
		{ 2155, 3550, 0 },
		{ 2202, 4098, 0 },
		{ 2095, 3676, 0 },
		{ 3258, 4735, 0 },
		{ 2187, 4511, 0 },
		{ 2187, 4513, 0 },
		{ 900, 3632, 0 },
		{ 0, 3634, 0 },
		{ 3231, 4154, 0 },
		{ 2223, 4436, 0 },
		{ 2187, 4519, 0 },
		{ 3218, 3972, 0 },
		{ 1260, 4455, 0 },
		{ 3127, 4981, 0 },
		{ 3218, 3974, 0 },
		{ 2155, 3551, 0 },
		{ 3252, 3746, 0 },
		{ 3218, 3980, 0 },
		{ 2117, 4288, 0 },
		{ 3168, 4366, 0 },
		{ 3218, 3981, 0 },
		{ 3231, 4171, 0 },
		{ 3252, 3747, 0 },
		{ 2676, 4623, 0 },
		{ 2676, 4629, 0 },
		{ 3127, 5051, 0 },
		{ 2187, 4531, 0 },
		{ 0, 0, 73 },
		{ 2117, 4292, 0 },
		{ 2676, 4646, 0 },
		{ 3231, 4176, 0 },
		{ 3252, 3749, 0 },
		{ 3243, 4055, 0 },
		{ 3218, 3984, 0 },
		{ 3218, 3986, 0 },
		{ 3218, 3899, 0 },
		{ 3252, 3750, 0 },
		{ 2202, 4133, 0 },
		{ 2202, 4135, 0 },
		{ 0, 0, 144 },
		{ 0, 0, 145 },
		{ 2136, 4237, 0 },
		{ 1260, 4466, 0 },
		{ 3252, 3751, 0 },
		{ 2117, 4258, 0 },
		{ 2117, 4260, 0 },
		{ 3062, 4785, 0 },
		{ 0, 0, 10 },
		{ 3127, 4833, 0 },
		{ 0, 0, 12 },
		{ 2133, 3889, 0 },
		{ 3252, 3752, 0 },
		{ 3127, 4457, 0 },
		{ 3258, 4712, 0 },
		{ 3231, 4194, 0 },
		{ 3127, 4851, 0 },
		{ 3127, 4855, 0 },
		{ 3252, 3754, 0 },
		{ 2155, 3552, 0 },
		{ 3168, 4400, 0 },
		{ 3168, 4403, 0 },
		{ 2187, 4575, 0 },
		{ 2155, 3553, 0 },
		{ 3258, 4746, 0 },
		{ 3127, 4903, 0 },
		{ 3258, 4748, 0 },
		{ 2117, 4271, 0 },
		{ 0, 0, 88 },
		{ 2202, 4144, 0 },
		{ 3168, 4411, 0 },
		{ 0, 0, 86 },
		{ 3248, 3712, 0 },
		{ 2136, 4220, 0 },
		{ 0, 0, 89 },
		{ 3258, 4765, 0 },
		{ 3168, 4415, 0 },
		{ 3248, 3713, 0 },
		{ 2187, 4484, 0 },
		{ 2133, 3827, 0 },
		{ 3218, 3920, 0 },
		{ 2187, 4488, 0 },
		{ 2155, 3554, 0 },
		{ 2155, 3555, 0 },
		{ 2155, 3556, 0 },
		{ 3252, 3764, 0 },
		{ 0, 0, 67 },
		{ 0, 0, 105 },
		{ 0, 0, 107 },
		{ 2202, 4093, 0 },
		{ 2676, 4627, 0 },
		{ 3218, 3926, 0 },
		{ 2187, 4495, 0 },
		{ 3168, 4339, 0 },
		{ 3243, 4033, 0 },
		{ 2187, 4501, 0 },
		{ 0, 0, 150 },
		{ 3168, 4341, 0 },
		{ 3218, 3927, 0 },
		{ 2187, 4504, 0 },
		{ 3168, 4344, 0 },
		{ 3252, 3765, 0 },
		{ 3218, 3929, 0 },
		{ 3127, 5047, 0 },
		{ 2155, 3557, 0 },
		{ 2117, 4302, 0 },
		{ 2117, 4303, 0 },
		{ 2187, 4512, 0 },
		{ 2676, 4609, 0 },
		{ 3252, 3767, 0 },
		{ 3252, 3768, 0 },
		{ 3218, 3936, 0 },
		{ 2223, 4430, 0 },
		{ 0, 4308, 0 },
		{ 3252, 3769, 0 },
		{ 3252, 3771, 0 },
		{ 3168, 4359, 0 },
		{ 3243, 4050, 0 },
		{ 2202, 4124, 0 },
		{ 3127, 4821, 0 },
		{ 3168, 4367, 0 },
		{ 3252, 3772, 0 },
		{ 2202, 4127, 0 },
		{ 3258, 4774, 0 },
		{ 0, 0, 20 },
		{ 2133, 3843, 0 },
		{ 2133, 3844, 0 },
		{ 0, 0, 139 },
		{ 1260, 4463, 0 },
		{ 1260, 4464, 0 },
		{ 2133, 3845, 0 },
		{ 0, 0, 143 },
		{ 3218, 3946, 0 },
		{ 2187, 4534, 0 },
		{ 0, 0, 103 },
		{ 2155, 3558, 0 },
		{ 2117, 4276, 0 },
		{ 2117, 4278, 0 },
		{ 2155, 3441, 0 },
		{ 2117, 4280, 0 },
		{ 3127, 4893, 0 },
		{ 2133, 3854, 0 },
		{ 2202, 4138, 0 },
		{ 3168, 4387, 0 },
		{ 0, 0, 84 },
		{ 2117, 4283, 0 },
		{ 1260, 4478, 0 },
		{ 2155, 3442, 0 },
		{ 2117, 4287, 0 },
		{ 3218, 3951, 0 },
		{ 2187, 4559, 0 },
		{ 3258, 4713, 0 },
		{ 3258, 4717, 0 },
		{ 3127, 4923, 0 },
		{ 2187, 4560, 0 },
		{ 3168, 4392, 0 },
		{ 3168, 4393, 0 },
		{ 2155, 3444, 0 },
		{ 2133, 3857, 0 },
		{ 3252, 3778, 0 },
		{ 3231, 4214, 0 },
		{ 3252, 3779, 0 },
		{ 2133, 3860, 0 },
		{ 3168, 4401, 0 },
		{ 3231, 4155, 0 },
		{ 3252, 3780, 0 },
		{ 2187, 4577, 0 },
		{ 0, 0, 92 },
		{ 3168, 4407, 0 },
		{ 3168, 4408, 0 },
		{ 3258, 4766, 0 },
		{ 0, 0, 109 },
		{ 2117, 4298, 0 },
		{ 3258, 4331, 0 },
		{ 2187, 4582, 0 },
		{ 0, 0, 148 },
		{ 3252, 3781, 0 },
		{ 3168, 4412, 0 },
		{ 3252, 3782, 0 },
		{ 2155, 3445, 64 },
		{ 3231, 4164, 0 },
		{ 2202, 4086, 0 },
		{ 2117, 4304, 0 },
		{ 3248, 3699, 0 },
		{ 3062, 4259, 0 },
		{ 0, 0, 93 },
		{ 2133, 3868, 0 },
		{ 3258, 4739, 0 },
		{ 1081, 4419, 0 },
		{ 0, 4420, 0 },
		{ 3252, 3785, 0 },
		{ 3231, 4173, 0 },
		{ 3231, 4174, 0 },
		{ 2202, 4091, 0 },
		{ 3258, 4760, 0 },
		{ 2187, 4498, 0 },
		{ 3168, 4338, 0 },
		{ 2155, 3446, 0 },
		{ 2202, 4095, 0 },
		{ 2202, 4096, 0 },
		{ 2202, 4097, 0 },
		{ 0, 0, 97 },
		{ 3168, 4343, 0 },
		{ 2133, 3878, 0 },
		{ 2117, 4266, 0 },
		{ 2117, 4267, 0 },
		{ 3218, 3968, 0 },
		{ 0, 0, 131 },
		{ 2155, 3447, 0 },
		{ 3248, 3705, 0 },
		{ 2155, 3448, 0 },
		{ 3243, 4043, 0 },
		{ 3252, 3790, 0 },
		{ 3168, 4355, 0 },
		{ 2676, 4655, 0 },
		{ 2133, 3886, 0 },
		{ 3231, 4195, 0 },
		{ 0, 0, 99 },
		{ 3231, 4196, 0 },
		{ 2676, 4695, 0 },
		{ 2676, 4697, 0 },
		{ 3252, 3794, 0 },
		{ 0, 0, 16 },
		{ 2117, 4285, 0 },
		{ 0, 0, 18 },
		{ 0, 0, 19 },
		{ 3168, 4365, 0 },
		{ 2155, 3449, 0 },
		{ 2223, 4429, 0 },
		{ 3231, 4201, 0 },
		{ 3127, 4915, 0 },
		{ 3218, 3977, 0 },
		{ 2202, 4118, 0 },
		{ 1260, 4476, 0 },
		{ 3218, 3978, 0 },
		{ 3218, 3979, 0 },
		{ 3231, 4208, 0 },
		{ 2202, 4125, 0 },
		{ 0, 0, 63 },
		{ 2187, 4536, 0 },
		{ 2187, 4537, 0 },
		{ 3168, 4378, 0 },
		{ 3252, 3796, 0 },
		{ 2155, 3450, 0 },
		{ 3252, 3798, 0 },
		{ 2117, 4297, 0 },
		{ 1145, 3619, 0 },
		{ 2202, 4130, 0 },
		{ 2187, 4552, 0 },
		{ 0, 0, 82 },
		{ 2155, 3451, 0 },
		{ 3258, 4755, 0 },
		{ 3218, 3985, 0 },
		{ 0, 3621, 0 },
		{ 3218, 3990, 0 },
		{ 0, 0, 17 },
		{ 2202, 4136, 0 },
		{ 1260, 4470, 0 },
		{ 2155, 3453, 62 },
		{ 2155, 3454, 0 },
		{ 2117, 4250, 0 },
		{ 0, 0, 83 },
		{ 3231, 4167, 0 },
		{ 3258, 3666, 0 },
		{ 0, 0, 90 },
		{ 0, 0, 91 },
		{ 0, 0, 60 },
		{ 3231, 4170, 0 },
		{ 0, 0, 140 },
		{ 0, 0, 141 },
		{ 3243, 4075, 0 },
		{ 3243, 4076, 0 },
		{ 3252, 3806, 0 },
		{ 3243, 4079, 0 },
		{ 0, 0, 149 },
		{ 0, 0, 135 },
		{ 3231, 4175, 0 },
		{ 1260, 4482, 0 },
		{ 1260, 4453, 0 },
		{ 3252, 3807, 0 },
		{ 0, 0, 40 },
		{ 2155, 3455, 41 },
		{ 2155, 3458, 43 },
		{ 3231, 4179, 0 },
		{ 3258, 4743, 0 },
		{ 1260, 4460, 0 },
		{ 1260, 4461, 0 },
		{ 2117, 4268, 0 },
		{ 0, 0, 80 },
		{ 3231, 4180, 0 },
		{ 3252, 3810, 0 },
		{ 0, 0, 98 },
		{ 3252, 3811, 0 },
		{ 2117, 4274, 0 },
		{ 3243, 4028, 0 },
		{ 2117, 4277, 0 },
		{ 2133, 3837, 0 },
		{ 3168, 4318, 0 },
		{ 3248, 3723, 0 },
		{ 3168, 4328, 0 },
		{ 2223, 4431, 0 },
		{ 2223, 4432, 0 },
		{ 2155, 3459, 0 },
		{ 3252, 3814, 0 },
		{ 3258, 4715, 0 },
		{ 3248, 3689, 0 },
		{ 0, 0, 94 },
		{ 3258, 4721, 0 },
		{ 2155, 3460, 0 },
		{ 0, 0, 142 },
		{ 0, 0, 146 },
		{ 2117, 4284, 0 },
		{ 0, 0, 151 },
		{ 0, 0, 11 },
		{ 3231, 4192, 0 },
		{ 3231, 4193, 0 },
		{ 2202, 4094, 0 },
		{ 3243, 4037, 0 },
		{ 3243, 4038, 0 },
		{ 1260, 4457, 0 },
		{ 2155, 3461, 0 },
		{ 3252, 3818, 0 },
		{ 3231, 4199, 0 },
		{ 3252, 3819, 0 },
		{ 3258, 4753, 0 },
		{ 0, 0, 147 },
		{ 3168, 4346, 0 },
		{ 3258, 4756, 0 },
		{ 3231, 4203, 0 },
		{ 3248, 3696, 0 },
		{ 3248, 3697, 0 },
		{ 3248, 3698, 0 },
		{ 3258, 4762, 0 },
		{ 2155, 3463, 0 },
		{ 3258, 4764, 0 },
		{ 3168, 4352, 0 },
		{ 3127, 4973, 0 },
		{ 3252, 3727, 0 },
		{ 3252, 3728, 0 },
		{ 2155, 3464, 0 },
		{ 0, 0, 42 },
		{ 0, 0, 44 },
		{ 3231, 4212, 0 },
		{ 3127, 4987, 0 },
		{ 3258, 4773, 0 },
		{ 3252, 3731, 0 },
		{ 2202, 4113, 0 },
		{ 2117, 4245, 0 },
		{ 3168, 4360, 0 },
		{ 3168, 4361, 0 },
		{ 3062, 4780, 0 },
		{ 3258, 4719, 0 },
		{ 2117, 4246, 0 },
		{ 3127, 5041, 0 },
		{ 3168, 4364, 0 },
		{ 3231, 4152, 0 },
		{ 2117, 4248, 0 },
		{ 2202, 4114, 0 },
		{ 2202, 4115, 0 },
		{ 2187, 4538, 0 },
		{ 3252, 3732, 0 },
		{ 2117, 4252, 0 },
		{ 2117, 4254, 0 },
		{ 2202, 4117, 0 },
		{ 0, 0, 85 },
		{ 0, 0, 100 },
		{ 3231, 4158, 0 },
		{ 3231, 4159, 0 },
		{ 0, 4462, 0 },
		{ 3168, 4379, 0 },
		{ 0, 0, 95 },
		{ 2117, 4259, 0 },
		{ 3231, 4160, 0 },
		{ 2133, 3863, 0 },
		{ 0, 0, 13 },
		{ 2202, 4119, 0 },
		{ 2202, 4120, 0 },
		{ 0, 0, 96 },
		{ 0, 0, 61 },
		{ 0, 0, 101 },
		{ 3218, 3941, 0 },
		{ 3231, 4165, 0 },
		{ 2187, 4561, 0 },
		{ 0, 0, 15 },
		{ 2155, 3465, 0 },
		{ 3218, 3944, 0 },
		{ 2187, 4564, 0 },
		{ 3243, 4064, 0 },
		{ 2117, 4269, 0 },
		{ 3127, 4845, 0 },
		{ 3258, 4771, 0 },
		{ 2187, 4567, 0 },
		{ 2136, 4243, 0 },
		{ 2187, 4569, 0 },
		{ 3231, 4169, 0 },
		{ 3252, 3734, 0 },
		{ 0, 0, 14 },
		{ 3311, 1457, 248 },
		{ 0, 0, 249 },
		{ 3257, 5449, 250 },
		{ 3294, 1928, 254 },
		{ 1297, 2917, 255 },
		{ 0, 0, 255 },
		{ 3294, 1968, 251 },
		{ 1300, 1633, 0 },
		{ 3294, 2024, 252 },
		{ 1303, 1710, 0 },
		{ 1300, 0, 0 },
		{ 3203, 2940, 253 },
		{ 1305, 1584, 0 },
		{ 1303, 0, 0 },
		{ 3203, 2996, 251 },
		{ 1305, 0, 0 },
		{ 3203, 3006, 252 },
		{ 3248, 3694, 160 },
		{ 0, 0, 160 },
		{ 0, 0, 161 },
		{ 3272, 2357, 0 },
		{ 3294, 3235, 0 },
		{ 3311, 2391, 0 },
		{ 1313, 5203, 0 },
		{ 3291, 2933, 0 },
		{ 3294, 3185, 0 },
		{ 3306, 3315, 0 },
		{ 3302, 2757, 0 },
		{ 3305, 3339, 0 },
		{ 3311, 2376, 0 },
		{ 3305, 3375, 0 },
		{ 3307, 2254, 0 },
		{ 3194, 3002, 0 },
		{ 3309, 2507, 0 },
		{ 3253, 2568, 0 },
		{ 3272, 2289, 0 },
		{ 3312, 4861, 0 },
		{ 0, 0, 158 },
		{ 3248, 3720, 164 },
		{ 0, 0, 164 },
		{ 0, 0, 165 },
		{ 3272, 2341, 0 },
		{ 3294, 3243, 0 },
		{ 3311, 2408, 0 },
		{ 1334, 5285, 0 },
		{ 3258, 4378, 0 },
		{ 3248, 3687, 0 },
		{ 2676, 4706, 0 },
		{ 3127, 5045, 0 },
		{ 3312, 5139, 0 },
		{ 0, 0, 162 },
		{ 3062, 4782, 172 },
		{ 0, 0, 172 },
		{ 0, 0, 173 },
		{ 3294, 3189, 0 },
		{ 3140, 3119, 0 },
		{ 3309, 2513, 0 },
		{ 3311, 2435, 0 },
		{ 3294, 3178, 0 },
		{ 1349, 5324, 0 },
		{ 3294, 2847, 0 },
		{ 3276, 1713, 0 },
		{ 3294, 3190, 0 },
		{ 3311, 2387, 0 },
		{ 2892, 1620, 0 },
		{ 3307, 2138, 0 },
		{ 3301, 3072, 0 },
		{ 3194, 3032, 0 },
		{ 3253, 2631, 0 },
		{ 3096, 3089, 0 },
		{ 1360, 5336, 0 },
		{ 3294, 2853, 0 },
		{ 3302, 2809, 0 },
		{ 3272, 2355, 0 },
		{ 3294, 3147, 0 },
		{ 1365, 5308, 0 },
		{ 3304, 2784, 0 },
		{ 3299, 1886, 0 },
		{ 3253, 2620, 0 },
		{ 3306, 3308, 0 },
		{ 3307, 2008, 0 },
		{ 3194, 2854, 0 },
		{ 3309, 2484, 0 },
		{ 3253, 2574, 0 },
		{ 3312, 4935, 0 },
		{ 0, 0, 170 },
		{ 3248, 3692, 196 },
		{ 0, 0, 196 },
		{ 3272, 2292, 0 },
		{ 3294, 3150, 0 },
		{ 3311, 2399, 0 },
		{ 1381, 5317, 0 },
		{ 3306, 3065, 0 },
		{ 3302, 2791, 0 },
		{ 3305, 3394, 0 },
		{ 3272, 2296, 0 },
		{ 3272, 2312, 0 },
		{ 3294, 3234, 0 },
		{ 3272, 2316, 0 },
		{ 3312, 5011, 0 },
		{ 0, 0, 195 },
		{ 2223, 4448, 156 },
		{ 0, 0, 156 },
		{ 0, 0, 157 },
		{ 3294, 3238, 0 },
		{ 3253, 2575, 0 },
		{ 3309, 2522, 0 },
		{ 3296, 2727, 0 },
		{ 3294, 3170, 0 },
		{ 3258, 4754, 0 },
		{ 3302, 2799, 0 },
		{ 3305, 3372, 0 },
		{ 3272, 2323, 0 },
		{ 3272, 2333, 0 },
		{ 3274, 5156, 0 },
		{ 3274, 5146, 0 },
		{ 3194, 2927, 0 },
		{ 3253, 2640, 0 },
		{ 3194, 3023, 0 },
		{ 3307, 2015, 0 },
		{ 3194, 3042, 0 },
		{ 3305, 3371, 0 },
		{ 3302, 2807, 0 },
		{ 3194, 2922, 0 },
		{ 3272, 1820, 0 },
		{ 3294, 3162, 0 },
		{ 3311, 2415, 0 },
		{ 3312, 4995, 0 },
		{ 0, 0, 154 },
		{ 2779, 3328, 24 },
		{ 0, 0, 24 },
		{ 0, 0, 25 },
		{ 3294, 3175, 0 },
		{ 3096, 3090, 0 },
		{ 3194, 3017, 0 },
		{ 3253, 2611, 0 },
		{ 3257, 2, 0 },
		{ 3309, 2504, 0 },
		{ 3054, 2470, 0 },
		{ 3294, 3220, 0 },
		{ 3311, 2433, 0 },
		{ 3305, 3341, 0 },
		{ 3307, 2197, 0 },
		{ 3309, 2479, 0 },
		{ 3311, 2373, 0 },
		{ 3257, 5429, 0 },
		{ 3291, 3271, 0 },
		{ 3294, 3244, 0 },
		{ 3272, 2359, 0 },
		{ 3306, 3309, 0 },
		{ 3311, 2382, 0 },
		{ 3194, 3026, 0 },
		{ 3054, 2466, 0 },
		{ 3307, 2238, 0 },
		{ 3194, 3056, 0 },
		{ 3309, 2553, 0 },
		{ 3253, 2636, 0 },
		{ 3257, 7, 0 },
		{ 3274, 5160, 0 },
		{ 0, 0, 21 },
		{ 1464, 0, 1 },
		{ 1464, 0, 197 },
		{ 1464, 3069, 247 },
		{ 1679, 218, 247 },
		{ 1679, 400, 247 },
		{ 1679, 389, 247 },
		{ 1679, 520, 247 },
		{ 1679, 391, 247 },
		{ 1679, 408, 247 },
		{ 1679, 396, 247 },
		{ 1679, 478, 247 },
		{ 1679, 495, 247 },
		{ 1464, 0, 247 },
		{ 1476, 2851, 247 },
		{ 1464, 3137, 247 },
		{ 2779, 3333, 243 },
		{ 1679, 513, 247 },
		{ 1679, 537, 247 },
		{ 1679, 590, 247 },
		{ 1679, 0, 247 },
		{ 1679, 625, 247 },
		{ 1679, 612, 247 },
		{ 3309, 2485, 0 },
		{ 0, 0, 198 },
		{ 3253, 2634, 0 },
		{ 1679, 582, 0 },
		{ 1679, 0, 0 },
		{ 3257, 4377, 0 },
		{ 1679, 602, 0 },
		{ 1679, 619, 0 },
		{ 1679, 617, 0 },
		{ 1679, 625, 0 },
		{ 1679, 617, 0 },
		{ 1679, 621, 0 },
		{ 1679, 639, 0 },
		{ 1679, 621, 0 },
		{ 1679, 614, 0 },
		{ 1679, 606, 0 },
		{ 1679, 637, 0 },
		{ 3294, 3212, 0 },
		{ 3294, 3217, 0 },
		{ 1680, 644, 0 },
		{ 1680, 645, 0 },
		{ 1679, 656, 0 },
		{ 1679, 693, 0 },
		{ 1679, 684, 0 },
		{ 3309, 2510, 0 },
		{ 3291, 3288, 0 },
		{ 1679, 682, 0 },
		{ 1679, 726, 0 },
		{ 1679, 703, 0 },
		{ 1679, 704, 0 },
		{ 1679, 727, 0 },
		{ 1679, 754, 0 },
		{ 1679, 759, 0 },
		{ 1679, 753, 0 },
		{ 1679, 729, 0 },
		{ 1679, 721, 0 },
		{ 1679, 757, 0 },
		{ 1679, 772, 0 },
		{ 1679, 759, 0 },
		{ 3253, 2660, 0 },
		{ 3140, 3111, 0 },
		{ 1679, 775, 0 },
		{ 1679, 765, 0 },
		{ 1680, 767, 0 },
		{ 1679, 763, 0 },
		{ 1679, 793, 0 },
		{ 3302, 2814, 0 },
		{ 0, 0, 246 },
		{ 1679, 805, 0 },
		{ 1679, 838, 0 },
		{ 1679, 825, 0 },
		{ 1679, 830, 0 },
		{ 1679, 20, 0 },
		{ 1679, 73, 0 },
		{ 1679, 69, 0 },
		{ 1679, 58, 0 },
		{ 1679, 36, 0 },
		{ 1679, 40, 0 },
		{ 1679, 82, 0 },
		{ 1679, 0, 232 },
		{ 1679, 118, 0 },
		{ 3309, 2483, 0 },
		{ 3194, 3036, 0 },
		{ 1679, 76, 0 },
		{ 1679, 108, 0 },
		{ 1679, 110, 0 },
		{ 1679, 114, 0 },
		{ 1679, 114, 0 },
		{ -1559, 1535, 0 },
		{ 1680, 122, 0 },
		{ 1679, 155, 0 },
		{ 1679, 161, 0 },
		{ 1679, 153, 0 },
		{ 1679, 163, 0 },
		{ 1679, 167, 0 },
		{ 1679, 146, 0 },
		{ 1679, 163, 0 },
		{ 1679, 140, 0 },
		{ 1679, 131, 0 },
		{ 1679, 0, 231 },
		{ 1679, 138, 0 },
		{ 3296, 2704, 0 },
		{ 3253, 2613, 0 },
		{ 1679, 157, 0 },
		{ 1679, 168, 0 },
		{ 1679, 191, 0 },
		{ 1679, 0, 245 },
		{ 1679, 191, 0 },
		{ 0, 0, 233 },
		{ 1679, 185, 0 },
		{ 1681, 4, -4 },
		{ 1679, 246, 0 },
		{ 1679, 258, 0 },
		{ 1679, 255, 0 },
		{ 1679, 260, 0 },
		{ 1679, 264, 0 },
		{ 1679, 277, 0 },
		{ 1679, 248, 0 },
		{ 1679, 253, 0 },
		{ 1679, 244, 0 },
		{ 3294, 3251, 0 },
		{ 3294, 3264, 0 },
		{ 1679, 0, 235 },
		{ 1679, 285, 236 },
		{ 1679, 258, 0 },
		{ 1679, 266, 0 },
		{ 1679, 307, 0 },
		{ 1579, 3953, 0 },
		{ 3257, 4724, 0 },
		{ 2264, 5104, 222 },
		{ 1679, 310, 0 },
		{ 1679, 314, 0 },
		{ 1679, 344, 0 },
		{ 1679, 346, 0 },
		{ 1679, 381, 0 },
		{ 1679, 382, 0 },
		{ 1679, 370, 0 },
		{ 1679, 375, 0 },
		{ 1679, 357, 0 },
		{ 1679, 364, 0 },
		{ 1680, 350, 0 },
		{ 3258, 4745, 0 },
		{ 3257, 5450, 238 },
		{ 1679, 353, 0 },
		{ 1679, 365, 0 },
		{ 1679, 347, 0 },
		{ 1679, 363, 0 },
		{ 0, 0, 202 },
		{ 1681, 31, -7 },
		{ 1681, 206, -10 },
		{ 1681, 321, -13 },
		{ 1681, 349, -16 },
		{ 1681, 351, -19 },
		{ 1681, 435, -22 },
		{ 1679, 413, 0 },
		{ 1679, 426, 0 },
		{ 1679, 399, 0 },
		{ 1679, 0, 220 },
		{ 1679, 0, 234 },
		{ 3302, 2801, 0 },
		{ 1679, 430, 0 },
		{ 1679, 421, 0 },
		{ 1679, 454, 0 },
		{ 1680, 463, 0 },
		{ 1616, 3921, 0 },
		{ 3257, 4734, 0 },
		{ 2264, 5125, 223 },
		{ 1619, 3922, 0 },
		{ 3257, 4720, 0 },
		{ 2264, 5101, 224 },
		{ 1622, 3923, 0 },
		{ 3257, 4728, 0 },
		{ 2264, 5127, 227 },
		{ 1625, 3924, 0 },
		{ 3257, 4646, 0 },
		{ 2264, 5123, 228 },
		{ 1628, 3925, 0 },
		{ 3257, 4696, 0 },
		{ 2264, 5129, 229 },
		{ 1631, 3926, 0 },
		{ 3257, 4722, 0 },
		{ 2264, 5111, 230 },
		{ 1679, 508, 0 },
		{ 1681, 463, -25 },
		{ 1679, 481, 0 },
		{ 3305, 3357, 0 },
		{ 1679, 462, 0 },
		{ 1679, 507, 0 },
		{ 1679, 472, 0 },
		{ 1679, 486, 0 },
		{ 0, 0, 204 },
		{ 0, 0, 206 },
		{ 0, 0, 212 },
		{ 0, 0, 214 },
		{ 0, 0, 216 },
		{ 0, 0, 218 },
		{ 1681, 465, -28 },
		{ 1649, 3936, 0 },
		{ 3257, 4732, 0 },
		{ 2264, 5135, 226 },
		{ 1679, 0, 219 },
		{ 3272, 2291, 0 },
		{ 1679, 474, 0 },
		{ 1679, 490, 0 },
		{ 1680, 497, 0 },
		{ 1679, 494, 0 },
		{ 1658, 3943, 0 },
		{ 3257, 4726, 0 },
		{ 2264, 5100, 225 },
		{ 0, 0, 210 },
		{ 3272, 2318, 0 },
		{ 1679, 4, 241 },
		{ 1680, 501, 0 },
		{ 1679, 4, 244 },
		{ 1679, 544, 0 },
		{ 0, 0, 208 },
		{ 3274, 5158, 0 },
		{ 3274, 5159, 0 },
		{ 1679, 532, 0 },
		{ 0, 0, 242 },
		{ 1679, 557, 0 },
		{ 3274, 5148, 0 },
		{ 0, 0, 240 },
		{ 1679, 576, 0 },
		{ 1679, 581, 0 },
		{ 0, 0, 239 },
		{ 1679, 586, 0 },
		{ 1679, 577, 0 },
		{ 1680, 579, 237 },
		{ 1681, 1003, 0 },
		{ 1682, 736, -1 },
		{ 1683, 3967, 0 },
		{ 3257, 4684, 0 },
		{ 2264, 5132, 221 },
		{ 0, 0, 200 },
		{ 2223, 4433, 290 },
		{ 0, 0, 290 },
		{ 3294, 3171, 0 },
		{ 3253, 2664, 0 },
		{ 3309, 2542, 0 },
		{ 3296, 2722, 0 },
		{ 3294, 3187, 0 },
		{ 3258, 4741, 0 },
		{ 3302, 2803, 0 },
		{ 3305, 3382, 0 },
		{ 3272, 2319, 0 },
		{ 3272, 2322, 0 },
		{ 3274, 5163, 0 },
		{ 3274, 5164, 0 },
		{ 3194, 3027, 0 },
		{ 3253, 2607, 0 },
		{ 3194, 3034, 0 },
		{ 3307, 2148, 0 },
		{ 3194, 3041, 0 },
		{ 3305, 3376, 0 },
		{ 3302, 2787, 0 },
		{ 3194, 3050, 0 },
		{ 3272, 1814, 0 },
		{ 3294, 3248, 0 },
		{ 3311, 2370, 0 },
		{ 3312, 5065, 0 },
		{ 0, 0, 289 },
		{ 2223, 4445, 292 },
		{ 0, 0, 292 },
		{ 0, 0, 293 },
		{ 3294, 3252, 0 },
		{ 3253, 2619, 0 },
		{ 3309, 2488, 0 },
		{ 3296, 2710, 0 },
		{ 3294, 3159, 0 },
		{ 3258, 4772, 0 },
		{ 3302, 2808, 0 },
		{ 3305, 3336, 0 },
		{ 3272, 2334, 0 },
		{ 3272, 2335, 0 },
		{ 3274, 5165, 0 },
		{ 3274, 5168, 0 },
		{ 3306, 3321, 0 },
		{ 3311, 2380, 0 },
		{ 3309, 2512, 0 },
		{ 3272, 2337, 0 },
		{ 3272, 2339, 0 },
		{ 3309, 2532, 0 },
		{ 3276, 1672, 0 },
		{ 3294, 3194, 0 },
		{ 3311, 2395, 0 },
		{ 3312, 4941, 0 },
		{ 0, 0, 291 },
		{ 2223, 4449, 295 },
		{ 0, 0, 295 },
		{ 0, 0, 296 },
		{ 3294, 3201, 0 },
		{ 3253, 2592, 0 },
		{ 3309, 2477, 0 },
		{ 3296, 2706, 0 },
		{ 3294, 3229, 0 },
		{ 3258, 4747, 0 },
		{ 3302, 2739, 0 },
		{ 3305, 3381, 0 },
		{ 3272, 2349, 0 },
		{ 3272, 2351, 0 },
		{ 3274, 5154, 0 },
		{ 3274, 5155, 0 },
		{ 3296, 2723, 0 },
		{ 3299, 1883, 0 },
		{ 3307, 2256, 0 },
		{ 3305, 3355, 0 },
		{ 3307, 2266, 0 },
		{ 3309, 2495, 0 },
		{ 3311, 2368, 0 },
		{ 3312, 4923, 0 },
		{ 0, 0, 294 },
		{ 2223, 4425, 298 },
		{ 0, 0, 298 },
		{ 0, 0, 299 },
		{ 3294, 3145, 0 },
		{ 3253, 2635, 0 },
		{ 3309, 2505, 0 },
		{ 3296, 2728, 0 },
		{ 3294, 3161, 0 },
		{ 3258, 4770, 0 },
		{ 3302, 2741, 0 },
		{ 3305, 3338, 0 },
		{ 3272, 2361, 0 },
		{ 3272, 2279, 0 },
		{ 3274, 5142, 0 },
		{ 3274, 5144, 0 },
		{ 3294, 3174, 0 },
		{ 3276, 1709, 0 },
		{ 3305, 3366, 0 },
		{ 3302, 2796, 0 },
		{ 3299, 1882, 0 },
		{ 3305, 3373, 0 },
		{ 3307, 2116, 0 },
		{ 3309, 2530, 0 },
		{ 3311, 2384, 0 },
		{ 3312, 5137, 0 },
		{ 0, 0, 297 },
		{ 2223, 4427, 301 },
		{ 0, 0, 301 },
		{ 0, 0, 302 },
		{ 3294, 3193, 0 },
		{ 3253, 2593, 0 },
		{ 3309, 2540, 0 },
		{ 3296, 2708, 0 },
		{ 3294, 3202, 0 },
		{ 3258, 4744, 0 },
		{ 3302, 2819, 0 },
		{ 3305, 3353, 0 },
		{ 3272, 2294, 0 },
		{ 3272, 2295, 0 },
		{ 3274, 5161, 0 },
		{ 3274, 5162, 0 },
		{ 3309, 2555, 0 },
		{ 3054, 2458, 0 },
		{ 3307, 2128, 0 },
		{ 3194, 3053, 0 },
		{ 3296, 2729, 0 },
		{ 3194, 2852, 0 },
		{ 3272, 2299, 0 },
		{ 3294, 3250, 0 },
		{ 3311, 2405, 0 },
		{ 3312, 4931, 0 },
		{ 0, 0, 300 },
		{ 3127, 4859, 176 },
		{ 0, 0, 176 },
		{ 0, 0, 177 },
		{ 3140, 3124, 0 },
		{ 3307, 2136, 0 },
		{ 3294, 3140, 0 },
		{ 3311, 2414, 0 },
		{ 1823, 5311, 0 },
		{ 3294, 2836, 0 },
		{ 3276, 1711, 0 },
		{ 3294, 3152, 0 },
		{ 3311, 2429, 0 },
		{ 2892, 1635, 0 },
		{ 3307, 2156, 0 },
		{ 3301, 3067, 0 },
		{ 3194, 3029, 0 },
		{ 3253, 2572, 0 },
		{ 3096, 3091, 0 },
		{ 1834, 5321, 0 },
		{ 3294, 2834, 0 },
		{ 3302, 2764, 0 },
		{ 3272, 2321, 0 },
		{ 3294, 3188, 0 },
		{ 1839, 5190, 0 },
		{ 3304, 2782, 0 },
		{ 3299, 1849, 0 },
		{ 3253, 2579, 0 },
		{ 3306, 3296, 0 },
		{ 3307, 2204, 0 },
		{ 3194, 3060, 0 },
		{ 3309, 2519, 0 },
		{ 3253, 2599, 0 },
		{ 3312, 4853, 0 },
		{ 0, 0, 174 },
		{ 2223, 4439, 283 },
		{ 0, 0, 283 },
		{ 3294, 3208, 0 },
		{ 3253, 2601, 0 },
		{ 3309, 2521, 0 },
		{ 3296, 2694, 0 },
		{ 3294, 3223, 0 },
		{ 3258, 4749, 0 },
		{ 3302, 2817, 0 },
		{ 3305, 3396, 0 },
		{ 3272, 2324, 0 },
		{ 3272, 2326, 0 },
		{ 3274, 5166, 0 },
		{ 3274, 5167, 0 },
		{ 3291, 3273, 0 },
		{ 3194, 3025, 0 },
		{ 3272, 2332, 0 },
		{ 3054, 2459, 0 },
		{ 3302, 2759, 0 },
		{ 3305, 3362, 0 },
		{ 2892, 1664, 0 },
		{ 3312, 5081, 0 },
		{ 0, 0, 281 },
		{ 1887, 0, 1 },
		{ 2046, 3163, 398 },
		{ 3294, 3253, 398 },
		{ 3305, 3307, 398 },
		{ 3291, 2483, 398 },
		{ 1887, 0, 365 },
		{ 1887, 3111, 398 },
		{ 3301, 1847, 398 },
		{ 3062, 4781, 398 },
		{ 2202, 4106, 398 },
		{ 3248, 3711, 398 },
		{ 2202, 4109, 398 },
		{ 2187, 4516, 398 },
		{ 3311, 2282, 398 },
		{ 1887, 0, 398 },
		{ 2779, 3330, 396 },
		{ 3305, 3106, 398 },
		{ 3305, 3345, 398 },
		{ 0, 0, 398 },
		{ 3309, 2481, 0 },
		{ -1892, 17, 355 },
		{ -1893, 5243, 0 },
		{ 3253, 2646, 0 },
		{ 0, 0, 361 },
		{ 0, 0, 362 },
		{ 3302, 2806, 0 },
		{ 3194, 3057, 0 },
		{ 3294, 3166, 0 },
		{ 0, 0, 366 },
		{ 3253, 2649, 0 },
		{ 3311, 2396, 0 },
		{ 3194, 2853, 0 },
		{ 2155, 3430, 0 },
		{ 3243, 4053, 0 },
		{ 3252, 3791, 0 },
		{ 2095, 3681, 0 },
		{ 3243, 4056, 0 },
		{ 3299, 1816, 0 },
		{ 3272, 2340, 0 },
		{ 3253, 2570, 0 },
		{ 3307, 1917, 0 },
		{ 3311, 2409, 0 },
		{ 3309, 2496, 0 },
		{ 3201, 5219, 0 },
		{ 3309, 2498, 0 },
		{ 3272, 2347, 0 },
		{ 3307, 1955, 0 },
		{ 3253, 2598, 0 },
		{ 3291, 3282, 0 },
		{ 3311, 2418, 0 },
		{ 3302, 2793, 0 },
		{ 2223, 4438, 0 },
		{ 2155, 3456, 0 },
		{ 2155, 3457, 0 },
		{ 2187, 4558, 0 },
		{ 2117, 4275, 0 },
		{ 3294, 3205, 0 },
		{ 3272, 2350, 0 },
		{ 3291, 3275, 0 },
		{ 3299, 1848, 0 },
		{ 3294, 3214, 0 },
		{ 3302, 2800, 0 },
		{ 0, 7, 358 },
		{ 3296, 2724, 0 },
		{ 3294, 3221, 0 },
		{ 2202, 4090, 0 },
		{ 3307, 2014, 0 },
		{ 0, 0, 397 },
		{ 3294, 3225, 0 },
		{ 3291, 3283, 0 },
		{ 2187, 4572, 0 },
		{ 2133, 3839, 0 },
		{ 3243, 4036, 0 },
		{ 3218, 3963, 0 },
		{ 2155, 3470, 0 },
		{ 0, 0, 386 },
		{ 3258, 4737, 0 },
		{ 3309, 2517, 0 },
		{ 3311, 2366, 0 },
		{ 3253, 2617, 0 },
		{ -1969, 1170, 0 },
		{ 0, 0, 357 },
		{ 3294, 3240, 0 },
		{ 0, 0, 385 },
		{ 3054, 2468, 0 },
		{ 3194, 2851, 0 },
		{ 3253, 2621, 0 },
		{ 1984, 5184, 0 },
		{ 3231, 4188, 0 },
		{ 3168, 4362, 0 },
		{ 3218, 3973, 0 },
		{ 2155, 3479, 0 },
		{ 3243, 4052, 0 },
		{ 3309, 2527, 0 },
		{ 3296, 2712, 0 },
		{ 3253, 2633, 0 },
		{ 3307, 2073, 0 },
		{ 0, 0, 387 },
		{ 3258, 4758, 364 },
		{ 3307, 2091, 0 },
		{ 3306, 3301, 0 },
		{ 3307, 2102, 0 },
		{ 0, 0, 390 },
		{ 0, 0, 391 },
		{ 1989, 0, -47 },
		{ 2168, 3607, 0 },
		{ 2202, 4121, 0 },
		{ 3243, 4063, 0 },
		{ 2187, 4500, 0 },
		{ 3194, 3003, 0 },
		{ 0, 0, 389 },
		{ 0, 0, 395 },
		{ 0, 5182, 0 },
		{ 3302, 2785, 0 },
		{ 3272, 2282, 0 },
		{ 3305, 3386, 0 },
		{ 2223, 4443, 0 },
		{ 3257, 4660, 0 },
		{ 2264, 5131, 380 },
		{ 2187, 4506, 0 },
		{ 3062, 4783, 0 },
		{ 3218, 3988, 0 },
		{ 3218, 3989, 0 },
		{ 3253, 2641, 0 },
		{ 0, 0, 392 },
		{ 0, 0, 393 },
		{ 3305, 3395, 0 },
		{ 2694, 5229, 0 },
		{ 3302, 2792, 0 },
		{ 3294, 3157, 0 },
		{ 0, 0, 370 },
		{ 2012, 0, -50 },
		{ 2014, 0, -53 },
		{ 2202, 4134, 0 },
		{ 3258, 4730, 0 },
		{ 0, 0, 388 },
		{ 3272, 2284, 0 },
		{ 0, 0, 363 },
		{ 2223, 4428, 0 },
		{ 3253, 2648, 0 },
		{ 3257, 4739, 0 },
		{ 2264, 5106, 381 },
		{ 3257, 4758, 0 },
		{ 2264, 5117, 382 },
		{ 3062, 4778, 0 },
		{ 2024, 0, -35 },
		{ 3272, 2287, 0 },
		{ 3294, 3163, 0 },
		{ 3294, 3164, 0 },
		{ 0, 0, 372 },
		{ 0, 0, 374 },
		{ 2029, 0, -41 },
		{ 3257, 4664, 0 },
		{ 2264, 5084, 384 },
		{ 0, 0, 360 },
		{ 3253, 2654, 0 },
		{ 3311, 2385, 0 },
		{ 3257, 4690, 0 },
		{ 2264, 5105, 383 },
		{ 0, 0, 378 },
		{ 3309, 2478, 0 },
		{ 3305, 3365, 0 },
		{ 0, 0, 376 },
		{ 3296, 2721, 0 },
		{ 3307, 2104, 0 },
		{ 3294, 3176, 0 },
		{ 3194, 3048, 0 },
		{ 0, 0, 394 },
		{ 3309, 2480, 0 },
		{ 3253, 2571, 0 },
		{ 2043, 0, -56 },
		{ 3257, 4730, 0 },
		{ 2264, 5137, 379 },
		{ 0, 0, 368 },
		{ 1887, 3188, 398 },
		{ 2050, 2847, 398 },
		{ -2048, 5452, 355 },
		{ -2049, 5240, 0 },
		{ 3257, 5226, 0 },
		{ 3201, 5206, 0 },
		{ 0, 0, 356 },
		{ 3201, 5222, 0 },
		{ -2054, 21, 0 },
		{ -2055, 5247, 0 },
		{ 2058, 2, 358 },
		{ 3201, 5223, 0 },
		{ 3257, 5338, 0 },
		{ 0, 0, 359 },
		{ 2076, 0, 1 },
		{ 2272, 3178, 354 },
		{ 3294, 3195, 354 },
		{ 2076, 0, 308 },
		{ 2076, 3133, 354 },
		{ 3243, 4049, 354 },
		{ 2076, 0, 311 },
		{ 3299, 1884, 354 },
		{ 3062, 4776, 354 },
		{ 2202, 4099, 354 },
		{ 3248, 3642, 354 },
		{ 2202, 4101, 354 },
		{ 2187, 4551, 354 },
		{ 3305, 3346, 354 },
		{ 3311, 2278, 354 },
		{ 2076, 0, 354 },
		{ 2779, 3326, 351 },
		{ 3305, 3358, 354 },
		{ 3291, 3293, 354 },
		{ 3062, 4790, 354 },
		{ 3305, 1828, 354 },
		{ 0, 0, 354 },
		{ 3309, 2493, 0 },
		{ -2083, 20, 303 },
		{ -2084, 1, 0 },
		{ 3253, 2594, 0 },
		{ 0, 0, 309 },
		{ 3253, 2596, 0 },
		{ 3309, 2494, 0 },
		{ 3311, 2406, 0 },
		{ 2155, 3546, 0 },
		{ 3243, 4069, 0 },
		{ 3252, 3801, 0 },
		{ 3231, 4200, 0 },
		{ 0, 3626, 0 },
		{ 0, 3668, 0 },
		{ 3243, 4074, 0 },
		{ 3302, 2788, 0 },
		{ 3299, 1744, 0 },
		{ 3272, 2300, 0 },
		{ 3253, 2608, 0 },
		{ 3294, 3231, 0 },
		{ 3294, 3233, 0 },
		{ 3309, 2499, 0 },
		{ 2694, 5233, 0 },
		{ 3309, 2501, 0 },
		{ 3201, 5200, 0 },
		{ 3309, 2503, 0 },
		{ 3291, 3292, 0 },
		{ 3054, 2455, 0 },
		{ 3311, 2410, 0 },
		{ 2223, 4444, 0 },
		{ 2155, 3559, 0 },
		{ 2155, 3560, 0 },
		{ 3168, 4369, 0 },
		{ 3168, 4370, 0 },
		{ 2187, 4486, 0 },
		{ 0, 4273, 0 },
		{ 3272, 2305, 0 },
		{ 3294, 3245, 0 },
		{ 3272, 2306, 0 },
		{ 3291, 3280, 0 },
		{ 3253, 2622, 0 },
		{ 3272, 2311, 0 },
		{ 3302, 2811, 0 },
		{ 0, 0, 353 },
		{ 3302, 2813, 0 },
		{ 0, 0, 305 },
		{ 3296, 2720, 0 },
		{ 0, 0, 350 },
		{ 3299, 1747, 0 },
		{ 3294, 3269, 0 },
		{ 2187, 4499, 0 },
		{ 0, 3880, 0 },
		{ 3243, 4047, 0 },
		{ 2136, 4224, 0 },
		{ 0, 4225, 0 },
		{ 3218, 3987, 0 },
		{ 2155, 3452, 0 },
		{ 3294, 3137, 0 },
		{ 0, 0, 343 },
		{ 3258, 4736, 0 },
		{ 3309, 2516, 0 },
		{ 3307, 2166, 0 },
		{ 3307, 2180, 0 },
		{ 3299, 1748, 0 },
		{ -2163, 1245, 0 },
		{ 3294, 3153, 0 },
		{ 3302, 2762, 0 },
		{ 3253, 2645, 0 },
		{ 3231, 4182, 0 },
		{ 3168, 4402, 0 },
		{ 3218, 3907, 0 },
		{ 3168, 4404, 0 },
		{ 3168, 4405, 0 },
		{ 0, 3462, 0 },
		{ 3243, 4062, 0 },
		{ 0, 0, 342 },
		{ 3309, 2525, 0 },
		{ 3296, 2698, 0 },
		{ 3194, 2993, 0 },
		{ 0, 0, 349 },
		{ 3305, 3343, 0 },
		{ 0, 0, 344 },
		{ 0, 0, 307 },
		{ 3305, 3344, 0 },
		{ 3307, 2219, 0 },
		{ 2180, 0, -80 },
		{ 0, 3606, 0 },
		{ 2202, 4107, 0 },
		{ 2171, 3588, 0 },
		{ 2168, 3592, 0 },
		{ 3243, 4073, 0 },
		{ 2187, 4533, 0 },
		{ 3194, 3001, 0 },
		{ 0, 0, 346 },
		{ 3306, 3319, 0 },
		{ 3307, 2222, 0 },
		{ 3307, 2227, 0 },
		{ 2223, 4450, 0 },
		{ 3257, 4692, 0 },
		{ 2264, 5133, 333 },
		{ 2187, 4539, 0 },
		{ 3062, 4784, 0 },
		{ 2187, 4540, 0 },
		{ 2187, 4541, 0 },
		{ 2187, 4542, 0 },
		{ 0, 4543, 0 },
		{ 3218, 3931, 0 },
		{ 3218, 3933, 0 },
		{ 3253, 2658, 0 },
		{ 3305, 3359, 0 },
		{ 3194, 3020, 0 },
		{ 3194, 3022, 0 },
		{ 3294, 3177, 0 },
		{ 0, 0, 315 },
		{ 2209, 0, -59 },
		{ 2211, 0, -62 },
		{ 2213, 0, -68 },
		{ 2215, 0, -71 },
		{ 2217, 0, -74 },
		{ 2219, 0, -77 },
		{ 0, 4122, 0 },
		{ 3258, 4742, 0 },
		{ 0, 0, 345 },
		{ 3302, 2795, 0 },
		{ 3309, 2535, 0 },
		{ 3309, 2538, 0 },
		{ 3253, 2668, 0 },
		{ 3257, 4648, 0 },
		{ 2264, 5113, 334 },
		{ 3257, 4650, 0 },
		{ 2264, 5119, 335 },
		{ 3257, 4652, 0 },
		{ 2264, 5124, 338 },
		{ 3257, 4654, 0 },
		{ 2264, 5126, 339 },
		{ 3257, 4656, 0 },
		{ 2264, 5128, 340 },
		{ 3257, 4658, 0 },
		{ 2264, 5130, 341 },
		{ 3062, 4786, 0 },
		{ 2234, 0, -38 },
		{ 0, 4437, 0 },
		{ 3253, 2669, 0 },
		{ 3253, 2567, 0 },
		{ 3294, 3191, 0 },
		{ 0, 0, 317 },
		{ 0, 0, 319 },
		{ 0, 0, 325 },
		{ 0, 0, 327 },
		{ 0, 0, 329 },
		{ 0, 0, 331 },
		{ 2240, 0, -44 },
		{ 3257, 4666, 0 },
		{ 2264, 5103, 337 },
		{ 3294, 3192, 0 },
		{ 3305, 3390, 0 },
		{ 3256, 3589, 348 },
		{ 3311, 2378, 0 },
		{ 3257, 4694, 0 },
		{ 2264, 5115, 336 },
		{ 0, 0, 323 },
		{ 3253, 2569, 0 },
		{ 3311, 2379, 0 },
		{ 0, 0, 310 },
		{ 3305, 3404, 0 },
		{ 0, 0, 321 },
		{ 3309, 2546, 0 },
		{ 2892, 1636, 0 },
		{ 3307, 2249, 0 },
		{ 3296, 2700, 0 },
		{ 3127, 5003, 0 },
		{ 3194, 3051, 0 },
		{ 3294, 3209, 0 },
		{ 3302, 2816, 0 },
		{ 3309, 2474, 0 },
		{ 0, 0, 347 },
		{ 3096, 3088, 0 },
		{ 3253, 2576, 0 },
		{ 3309, 2475, 0 },
		{ 2263, 0, -65 },
		{ 3311, 2383, 0 },
		{ 3257, 4737, 0 },
		{ 0, 5102, 332 },
		{ 3194, 2846, 0 },
		{ 0, 0, 313 },
		{ 3307, 2251, 0 },
		{ 3301, 3078, 0 },
		{ 3296, 2714, 0 },
		{ 2694, 5232, 0 },
		{ 0, 0, 352 },
		{ 2076, 3164, 354 },
		{ 2276, 2849, 354 },
		{ -2274, 19, 303 },
		{ -2275, 5246, 0 },
		{ 3257, 5227, 0 },
		{ 3201, 5199, 0 },
		{ 0, 0, 304 },
		{ 3201, 5224, 0 },
		{ -2280, 5453, 0 },
		{ -2281, 5239, 0 },
		{ 2284, 0, 305 },
		{ 3201, 5225, 0 },
		{ 3257, 5364, 0 },
		{ 0, 0, 306 },
		{ 2676, 4642, 400 },
		{ 0, 0, 400 },
		{ 3294, 3239, 0 },
		{ 3140, 3116, 0 },
		{ 3305, 3380, 0 },
		{ 3299, 1850, 0 },
		{ 3302, 2765, 0 },
		{ 3307, 2260, 0 },
		{ 3257, 5428, 0 },
		{ 3311, 2394, 0 },
		{ 3299, 1852, 0 },
		{ 3253, 2602, 0 },
		{ 2299, 5231, 0 },
		{ 3257, 2289, 0 },
		{ 3305, 3398, 0 },
		{ 3311, 2398, 0 },
		{ 3305, 3335, 0 },
		{ 3296, 2731, 0 },
		{ 3294, 3259, 0 },
		{ 3307, 1922, 0 },
		{ 3294, 3265, 0 },
		{ 3311, 2401, 0 },
		{ 3272, 2344, 0 },
		{ 3312, 5085, 0 },
		{ 0, 0, 399 },
		{ 3201, 5197, 455 },
		{ 0, 0, 405 },
		{ 0, 0, 407 },
		{ 2332, 908, 445 },
		{ 2531, 922, 445 },
		{ 2557, 920, 445 },
		{ 2491, 921, 445 },
		{ 2333, 935, 445 },
		{ 2331, 925, 445 },
		{ 2557, 919, 445 },
		{ 2355, 935, 445 },
		{ 2557, 939, 445 },
		{ 2526, 937, 445 },
		{ 2531, 938, 445 },
		{ 2461, 947, 445 },
		{ 2329, 965, 445 },
		{ 3294, 1958, 444 },
		{ 2364, 2886, 455 },
		{ 2597, 937, 445 },
		{ 2355, 945, 445 },
		{ 2531, 951, 445 },
		{ 2369, 952, 445 },
		{ 2531, 949, 445 },
		{ 3294, 3186, 455 },
		{ -2335, 5458, 401 },
		{ -2336, 5241, 0 },
		{ 2597, 946, 445 },
		{ 2606, 446, 445 },
		{ 2597, 949, 445 },
		{ 2424, 947, 445 },
		{ 2531, 955, 445 },
		{ 2538, 950, 445 },
		{ 2531, 957, 445 },
		{ 2461, 966, 445 },
		{ 2424, 963, 445 },
		{ 2557, 1045, 445 },
		{ 2491, 958, 445 },
		{ 2461, 980, 445 },
		{ 2557, 964, 445 },
		{ 2329, 957, 445 },
		{ 2576, 964, 445 },
		{ 2366, 978, 445 },
		{ 2329, 989, 445 },
		{ 2538, 999, 445 },
		{ 2329, 1009, 445 },
		{ 2572, 1030, 445 },
		{ 2531, 1036, 445 },
		{ 2597, 1268, 445 },
		{ 2572, 1032, 445 },
		{ 2429, 1046, 445 },
		{ 2606, 662, 445 },
		{ 3294, 1991, 441 },
		{ 2399, 1632, 0 },
		{ 3294, 2001, 442 },
		{ 2572, 1046, 445 },
		{ 2572, 1047, 445 },
		{ 3253, 2624, 0 },
		{ 3201, 5217, 0 },
		{ 2329, 1060, 445 },
		{ 3309, 2305, 0 },
		{ 2526, 1085, 445 },
		{ 2411, 1070, 445 },
		{ 2572, 1078, 445 },
		{ 2576, 1073, 445 },
		{ 2576, 1075, 445 },
		{ 2429, 1120, 445 },
		{ 2526, 1128, 445 },
		{ 2526, 1129, 445 },
		{ 2491, 1113, 445 },
		{ 2526, 1131, 445 },
		{ 2557, 1119, 445 },
		{ 2491, 1116, 445 },
		{ 2526, 1160, 445 },
		{ 2461, 1165, 445 },
		{ 2557, 1149, 445 },
		{ 2606, 668, 445 },
		{ 2606, 670, 445 },
		{ 2565, 1151, 445 },
		{ 2565, 1188, 445 },
		{ 2526, 1203, 445 },
		{ 2513, 1205, 445 },
		{ 2411, 1189, 445 },
		{ 2538, 1196, 445 },
		{ 2461, 1211, 445 },
		{ 2513, 1209, 445 },
		{ 3304, 2778, 0 },
		{ 2433, 1665, 0 },
		{ 2399, 0, 0 },
		{ 3203, 2950, 443 },
		{ 2435, 1666, 0 },
		{ 2329, 889, 445 },
		{ 3291, 3278, 0 },
		{ 0, 0, 403 },
		{ 2526, 1236, 445 },
		{ 3140, 3113, 0 },
		{ 2606, 672, 445 },
		{ 2429, 1231, 445 },
		{ 2576, 1225, 445 },
		{ 2606, 674, 445 },
		{ 2531, 1268, 445 },
		{ 2329, 1253, 445 },
		{ 2557, 1257, 445 },
		{ 2427, 1273, 445 },
		{ 2526, 1273, 445 },
		{ 2606, 10, 445 },
		{ 2576, 1260, 445 },
		{ 2531, 1273, 445 },
		{ 2606, 117, 445 },
		{ 2576, 1263, 445 },
		{ 2461, 1283, 445 },
		{ 3257, 2547, 0 },
		{ 3307, 2085, 0 },
		{ 2565, 1266, 445 },
		{ 2329, 1270, 445 },
		{ 2557, 1269, 445 },
		{ 2606, 120, 445 },
		{ 2329, 1285, 445 },
		{ 2576, 1269, 445 },
		{ 2329, 1278, 445 },
		{ 2329, 1268, 445 },
		{ 3194, 3043, 0 },
		{ 2433, 0, 0 },
		{ 3203, 3031, 441 },
		{ 2435, 0, 0 },
		{ 3203, 2918, 442 },
		{ 3201, 5208, 0 },
		{ 0, 0, 447 },
		{ 2557, 1274, 445 },
		{ 2470, 5350, 0 },
		{ 3302, 2381, 0 },
		{ 2461, 1292, 445 },
		{ 2606, 212, 445 },
		{ 3272, 2227, 0 },
		{ 2631, 6, 445 },
		{ 2565, 1277, 445 },
		{ 2461, 1296, 445 },
		{ 2531, 1289, 445 },
		{ 2576, 1279, 445 },
		{ 2557, 1282, 445 },
		{ 3257, 2261, 0 },
		{ 2606, 214, 445 },
		{ 2491, 1279, 445 },
		{ 3309, 2331, 0 },
		{ 2531, 1293, 445 },
		{ 2576, 1283, 445 },
		{ 3253, 2581, 0 },
		{ 3253, 2590, 0 },
		{ 3311, 2400, 0 },
		{ 2538, 1289, 445 },
		{ 2557, 1287, 445 },
		{ 2329, 1305, 445 },
		{ 3257, 3647, 0 },
		{ 2526, 1302, 445 },
		{ 2526, 1303, 445 },
		{ 2606, 219, 445 },
		{ 2531, 1301, 445 },
		{ 3302, 2779, 0 },
		{ 2501, 1867, 0 },
		{ 2606, 324, 445 },
		{ 3304, 2529, 0 },
		{ 3194, 3047, 0 },
		{ 2576, 1291, 445 },
		{ 3272, 2219, 0 },
		{ 3307, 2144, 0 },
		{ 3312, 5083, 0 },
		{ 3257, 5423, 411 },
		{ 2597, 1299, 445 },
		{ 2576, 1293, 445 },
		{ 2597, 1301, 445 },
		{ 2531, 1310, 445 },
		{ 2606, 330, 445 },
		{ 3309, 2531, 0 },
		{ 3304, 2818, 0 },
		{ 2531, 1308, 445 },
		{ 3140, 3121, 0 },
		{ 2538, 1303, 445 },
		{ 2531, 1311, 445 },
		{ 3194, 2918, 0 },
		{ 3194, 2921, 0 },
		{ 3294, 3203, 0 },
		{ 2329, 1299, 445 },
		{ 2531, 1313, 445 },
		{ 2576, 1303, 445 },
		{ 3309, 2547, 0 },
		{ 2606, 332, 445 },
		{ 2606, 345, 445 },
		{ 3311, 2231, 0 },
		{ 2572, 1312, 445 },
		{ 3294, 3222, 0 },
		{ 2501, 33, 446 },
		{ 2567, 36, 446 },
		{ 3309, 2349, 0 },
		{ 3243, 4057, 0 },
		{ 3194, 3024, 0 },
		{ 3296, 2702, 0 },
		{ 2531, 1329, 445 },
		{ 3307, 1988, 0 },
		{ 3305, 3401, 0 },
		{ 2631, 210, 445 },
		{ 2538, 1324, 445 },
		{ 2526, 1335, 445 },
		{ 2538, 1326, 445 },
		{ 2329, 1338, 445 },
		{ 3257, 2259, 0 },
		{ 3054, 2453, 0 },
		{ 3311, 2413, 0 },
		{ 2572, 1329, 445 },
		{ 2551, 5189, 0 },
		{ 2572, 1330, 445 },
		{ 2538, 1356, 445 },
		{ 3307, 2132, 0 },
		{ 3307, 2134, 0 },
		{ 3294, 3261, 0 },
		{ 2526, 1367, 445 },
		{ 2572, 1359, 445 },
		{ 2329, 1369, 445 },
		{ 3311, 2367, 0 },
		{ 3309, 2183, 0 },
		{ 3257, 2543, 0 },
		{ 3294, 3146, 0 },
		{ 2329, 1367, 445 },
		{ 3312, 5075, 0 },
		{ 2501, 27, 446 },
		{ 3140, 3114, 0 },
		{ 3248, 3719, 0 },
		{ 3307, 2174, 0 },
		{ 3194, 2855, 0 },
		{ 2329, 1390, 445 },
		{ 3305, 3388, 0 },
		{ 3307, 2191, 0 },
		{ 3312, 4929, 0 },
		{ 0, 0, 431 },
		{ 2557, 1388, 445 },
		{ 2572, 1393, 445 },
		{ 2572, 1394, 445 },
		{ 2606, 437, 445 },
		{ 3309, 2533, 0 },
		{ 3299, 1743, 0 },
		{ 3309, 2536, 0 },
		{ 2558, 1412, 445 },
		{ 3257, 2257, 0 },
		{ 2606, 439, 445 },
		{ 2572, 1407, 445 },
		{ 2586, 5324, 0 },
		{ 2587, 5326, 0 },
		{ 2588, 5345, 0 },
		{ 2329, 1404, 445 },
		{ 2329, 1416, 445 },
		{ 2606, 444, 445 },
		{ 3073, 2825, 0 },
		{ 3305, 3348, 0 },
		{ 3140, 3125, 0 },
		{ 3272, 2309, 0 },
		{ 3291, 3277, 0 },
		{ 2329, 1433, 445 },
		{ 3257, 5359, 436 },
		{ 2401, 814, 446 },
		{ 2598, 5302, 0 },
		{ 3272, 2315, 0 },
		{ 3253, 2650, 0 },
		{ 3307, 1952, 0 },
		{ 2329, 1440, 445 },
		{ 3307, 1980, 0 },
		{ 3272, 2320, 0 },
		{ 2606, 548, 445 },
		{ 2329, 1437, 445 },
		{ 2606, 552, 445 },
		{ 3257, 2664, 0 },
		{ 3311, 2422, 0 },
		{ 3302, 2821, 0 },
		{ 3296, 2696, 0 },
		{ 2606, 554, 445 },
		{ 3311, 2426, 0 },
		{ 3257, 2276, 0 },
		{ 2606, 556, 445 },
		{ 3307, 2185, 0 },
		{ 3307, 2187, 0 },
		{ 3291, 2921, 0 },
		{ 2606, 558, 445 },
		{ 2606, 560, 445 },
		{ 3306, 2174, 0 },
		{ 3194, 2923, 0 },
		{ 3311, 2372, 0 },
		{ 3140, 3106, 0 },
		{ 3302, 2789, 0 },
		{ 3299, 1854, 0 },
		{ 2606, 1649, 445 },
		{ 3309, 2201, 0 },
		{ 2634, 5371, 0 },
		{ 3294, 3249, 0 },
		{ 3312, 5001, 0 },
		{ 2606, 562, 445 },
		{ 3272, 2348, 0 },
		{ 3312, 5069, 0 },
		{ 3257, 2662, 0 },
		{ 2631, 666, 445 },
		{ 3309, 2307, 0 },
		{ 3294, 3258, 0 },
		{ 3307, 2162, 0 },
		{ 3305, 3368, 0 },
		{ 2646, 5226, 0 },
		{ 3309, 2176, 0 },
		{ 3309, 2526, 0 },
		{ 3311, 2386, 0 },
		{ 3257, 2265, 0 },
		{ 3311, 2388, 0 },
		{ 3311, 2389, 0 },
		{ 3294, 3143, 0 },
		{ 3257, 2274, 0 },
		{ 3272, 2221, 0 },
		{ 3272, 2362, 0 },
		{ 3253, 2626, 0 },
		{ 3294, 3151, 0 },
		{ 3253, 2627, 0 },
		{ 2661, 5311, 0 },
		{ 3253, 2629, 0 },
		{ 3294, 3155, 0 },
		{ 3272, 2272, 0 },
		{ 3305, 3403, 0 },
		{ 3306, 3298, 0 },
		{ 2667, 889, 445 },
		{ 3294, 3160, 0 },
		{ 3054, 2446, 0 },
		{ 3312, 4925, 0 },
		{ 3272, 2280, 0 },
		{ 3257, 5289, 409 },
		{ 3272, 2229, 0 },
		{ 3312, 4933, 0 },
		{ 3257, 5300, 418 },
		{ 3309, 2549, 0 },
		{ 3257, 4089, 0 },
		{ 3054, 2457, 0 },
		{ 3253, 2644, 0 },
		{ 3312, 5005, 0 },
		{ 3307, 2212, 0 },
		{ 3304, 2774, 0 },
		{ 3305, 3350, 0 },
		{ 3140, 3117, 0 },
		{ 3096, 3097, 0 },
		{ 3309, 2473, 0 },
		{ 3311, 2402, 0 },
		{ 3294, 3179, 0 },
		{ 3294, 3184, 0 },
		{ 3054, 2464, 0 },
		{ 3311, 2403, 0 },
		{ 3194, 2859, 0 },
		{ 3276, 1710, 0 },
		{ 3291, 3284, 0 },
		{ 3253, 2656, 0 },
		{ 3299, 1851, 0 },
		{ 3272, 2090, 0 },
		{ 3096, 3096, 0 },
		{ 3253, 2661, 0 },
		{ 3054, 2452, 0 },
		{ 3253, 2666, 0 },
		{ 3294, 3196, 0 },
		{ 3312, 5040, 0 },
		{ 3257, 5295, 434 },
		{ 3253, 2667, 0 },
		{ 3307, 2232, 0 },
		{ 0, 0, 452 },
		{ 3272, 2293, 0 },
		{ 3194, 3004, 0 },
		{ 3257, 5340, 417 },
		{ 3305, 3397, 0 },
		{ 0, 4650, 0 },
		{ 3294, 3204, 0 },
		{ 3194, 3008, 0 },
		{ 3257, 5366, 440 },
		{ 3194, 3013, 0 },
		{ 3194, 3014, 0 },
		{ 3311, 2411, 0 },
		{ 3140, 3102, 0 },
		{ 2714, 5322, 0 },
		{ 2779, 3329, 0 },
		{ 3309, 2486, 0 },
		{ 3294, 3218, 0 },
		{ 3294, 3219, 0 },
		{ 3307, 2234, 0 },
		{ 3309, 2492, 0 },
		{ 2705, 1557, 0 },
		{ 2722, 5323, 0 },
		{ 2694, 5230, 0 },
		{ 0, 5231, 0 },
		{ 3054, 2460, 0 },
		{ 3306, 3307, 0 },
		{ 2727, 5369, 0 },
		{ 3307, 2248, 0 },
		{ 3311, 2416, 0 },
		{ 3291, 3291, 0 },
		{ 2731, 5186, 0 },
		{ 3294, 3230, 0 },
		{ 3194, 3028, 0 },
		{ 2734, 5203, 0 },
		{ 0, 1542, 0 },
		{ 3302, 2743, 0 },
		{ 3312, 5110, 0 },
		{ 3311, 2421, 0 },
		{ 3307, 2250, 0 },
		{ 3309, 2502, 0 },
		{ 3302, 2763, 0 },
		{ 3294, 3242, 0 },
		{ 3272, 2302, 0 },
		{ 3257, 3091, 0 },
		{ 3305, 3379, 0 },
		{ 2779, 3331, 0 },
		{ 2747, 5301, 0 },
		{ 2748, 5304, 0 },
		{ 3301, 3064, 0 },
		{ 2779, 3332, 0 },
		{ 3294, 3247, 0 },
		{ 3272, 2203, 0 },
		{ 3306, 3297, 0 },
		{ 3307, 2252, 0 },
		{ 3302, 2783, 0 },
		{ 3311, 2431, 0 },
		{ 3257, 3714, 0 },
		{ 3272, 2307, 0 },
		{ 3194, 3055, 0 },
		{ 2760, 5327, 0 },
		{ 3309, 2317, 0 },
		{ 3311, 2365, 0 },
		{ 3296, 2725, 0 },
		{ 3306, 3067, 0 },
		{ 3294, 3262, 0 },
		{ 3312, 5141, 0 },
		{ 3257, 5438, 437 },
		{ 3305, 3337, 0 },
		{ 3309, 2514, 0 },
		{ 3253, 2604, 0 },
		{ 3294, 3267, 0 },
		{ 3253, 2605, 0 },
		{ 3054, 2462, 0 },
		{ 3299, 1885, 0 },
		{ 2779, 3324, 0 },
		{ 3305, 3347, 0 },
		{ 3291, 2929, 0 },
		{ 3291, 2931, 0 },
		{ 2778, 5267, 0 },
		{ 3305, 3351, 0 },
		{ 3312, 5007, 0 },
		{ 3307, 2259, 0 },
		{ 3272, 2314, 0 },
		{ 0, 1668, 0 },
		{ 3309, 2524, 0 },
		{ 3194, 2925, 0 },
		{ 3305, 3361, 0 },
		{ 2787, 5260, 0 },
		{ 3253, 2618, 0 },
		{ 3296, 2352, 0 },
		{ 3054, 2447, 0 },
		{ 3305, 3367, 0 },
		{ 3194, 2999, 0 },
		{ 3305, 3370, 0 },
		{ 3312, 4851, 0 },
		{ 3257, 5409, 432 },
		{ 3307, 2267, 0 },
		{ 3311, 2374, 0 },
		{ 3312, 4869, 0 },
		{ 3312, 4921, 0 },
		{ 3307, 1916, 0 },
		{ 3311, 2377, 0 },
		{ 3140, 3134, 0 },
		{ 3194, 3006, 0 },
		{ 2779, 3323, 0 },
		{ 3294, 3167, 0 },
		{ 3294, 3168, 0 },
		{ 3312, 4993, 0 },
		{ 0, 3327, 0 },
		{ 3257, 5434, 416 },
		{ 3305, 3387, 0 },
		{ 3276, 1712, 0 },
		{ 3307, 1918, 0 },
		{ 3307, 1919, 0 },
		{ 3054, 2461, 0 },
		{ 3307, 1920, 0 },
		{ 3309, 2346, 0 },
		{ 3096, 3094, 0 },
		{ 3309, 2543, 0 },
		{ 3294, 3180, 0 },
		{ 3307, 1945, 0 },
		{ 3272, 2328, 0 },
		{ 3272, 2329, 0 },
		{ 3257, 5302, 410 },
		{ 3309, 2550, 0 },
		{ 3272, 2330, 0 },
		{ 3257, 5326, 424 },
		{ 3257, 5329, 425 },
		{ 3272, 2331, 0 },
		{ 3194, 3031, 0 },
		{ 3140, 3127, 0 },
		{ 3302, 2781, 0 },
		{ 3194, 3033, 0 },
		{ 3054, 2448, 0 },
		{ 3054, 2450, 0 },
		{ 0, 0, 451 },
		{ 3194, 3037, 0 },
		{ 3307, 1946, 0 },
		{ 2832, 5265, 0 },
		{ 2833, 5284, 0 },
		{ 2834, 5282, 0 },
		{ 3307, 1947, 0 },
		{ 3301, 3073, 0 },
		{ 3054, 2454, 0 },
		{ 2838, 5298, 0 },
		{ 3291, 3274, 0 },
		{ 3311, 2392, 0 },
		{ 3194, 3045, 0 },
		{ 3305, 3360, 0 },
		{ 3294, 3206, 0 },
		{ 3311, 2393, 0 },
		{ 3312, 5013, 0 },
		{ 3312, 5038, 0 },
		{ 3253, 2663, 0 },
		{ 3294, 3210, 0 },
		{ 3194, 3049, 0 },
		{ 3302, 2794, 0 },
		{ 3307, 1949, 0 },
		{ 3307, 1950, 0 },
		{ 3302, 2797, 0 },
		{ 3272, 2336, 0 },
		{ 3272, 2214, 0 },
		{ 3304, 2772, 0 },
		{ 3272, 2217, 0 },
		{ 3312, 5145, 0 },
		{ 3312, 5147, 0 },
		{ 3294, 3224, 0 },
		{ 3309, 2301, 0 },
		{ 3294, 3228, 0 },
		{ 3305, 3383, 0 },
		{ 3309, 2491, 0 },
		{ 3307, 1982, 0 },
		{ 3272, 2343, 0 },
		{ 3312, 4927, 0 },
		{ 3311, 1017, 413 },
		{ 3257, 5297, 427 },
		{ 3096, 3095, 0 },
		{ 3311, 2404, 0 },
		{ 3307, 1999, 0 },
		{ 3194, 2915, 0 },
		{ 3301, 3068, 0 },
		{ 3301, 3069, 0 },
		{ 3194, 2917, 0 },
		{ 2875, 5243, 0 },
		{ 3307, 2000, 0 },
		{ 3194, 2919, 0 },
		{ 3306, 3311, 0 },
		{ 3257, 5345, 423 },
		{ 3257, 5347, 438 },
		{ 3311, 2407, 0 },
		{ 3054, 2451, 0 },
		{ 3302, 2820, 0 },
		{ 3307, 2003, 0 },
		{ 3253, 2589, 0 },
		{ 3194, 2991, 0 },
		{ 2885, 5323, 0 },
		{ 3257, 5383, 412 },
		{ 3312, 5077, 0 },
		{ 2887, 5332, 0 },
		{ 2892, 1665, 0 },
		{ 3307, 2012, 0 },
		{ 2890, 5344, 0 },
		{ 2891, 5345, 0 },
		{ 3307, 2013, 0 },
		{ 3304, 2788, 0 },
		{ 3305, 3349, 0 },
		{ 3302, 2761, 0 },
		{ 3311, 2412, 0 },
		{ 3305, 3352, 0 },
		{ 3294, 3263, 0 },
		{ 3312, 4849, 0 },
		{ 3309, 2508, 0 },
		{ 3272, 2356, 0 },
		{ 3309, 2511, 0 },
		{ 3312, 4863, 0 },
		{ 3257, 5418, 429 },
		{ 3312, 4865, 0 },
		{ 3312, 4867, 0 },
		{ 2892, 1559, 0 },
		{ 3312, 4894, 0 },
		{ 3312, 4896, 0 },
		{ 0, 1592, 0 },
		{ 3194, 3016, 0 },
		{ 3307, 2016, 0 },
		{ 3294, 3144, 0 },
		{ 3194, 3019, 0 },
		{ 3307, 2017, 0 },
		{ 3311, 2419, 0 },
		{ 3257, 5444, 435 },
		{ 3311, 2420, 0 },
		{ 3312, 4939, 0 },
		{ 3253, 2609, 0 },
		{ 0, 0, 454 },
		{ 0, 0, 453 },
		{ 3257, 5286, 414 },
		{ 3312, 4966, 0 },
		{ 0, 0, 450 },
		{ 0, 0, 449 },
		{ 3312, 4968, 0 },
		{ 3302, 2790, 0 },
		{ 3272, 2363, 0 },
		{ 3312, 4999, 0 },
		{ 3054, 2469, 0 },
		{ 3309, 2523, 0 },
		{ 3305, 3378, 0 },
		{ 3312, 5009, 0 },
		{ 3257, 5305, 408 },
		{ 2928, 5211, 0 },
		{ 3257, 5310, 439 },
		{ 3257, 5314, 415 },
		{ 3294, 3158, 0 },
		{ 2930, 5224, 0 },
		{ 3257, 5321, 421 },
		{ 3307, 2040, 0 },
		{ 3311, 2423, 0 },
		{ 3307, 2041, 0 },
		{ 3257, 5331, 430 },
		{ 3257, 2545, 0 },
		{ 3312, 5067, 0 },
		{ 3304, 2780, 0 },
		{ 3312, 5071, 0 },
		{ 3312, 5073, 0 },
		{ 3309, 2528, 0 },
		{ 3307, 2059, 0 },
		{ 3257, 5350, 419 },
		{ 3194, 3039, 0 },
		{ 3257, 5355, 422 },
		{ 3257, 5357, 426 },
		{ 3311, 2427, 0 },
		{ 3294, 3169, 0 },
		{ 3302, 2805, 0 },
		{ 3312, 5112, 0 },
		{ 3311, 2428, 0 },
		{ 3294, 3172, 0 },
		{ 3257, 5374, 428 },
		{ 3305, 3399, 0 },
		{ 3312, 5143, 0 },
		{ 3307, 2062, 0 },
		{ 3257, 5385, 420 },
		{ 3194, 3044, 0 },
		{ 3309, 2534, 0 },
		{ 3253, 2630, 0 },
		{ 3272, 2283, 0 },
		{ 3312, 4859, 0 },
		{ 3257, 5398, 433 },
		{ 3201, 5207, 455 },
		{ 2961, 0, 405 },
		{ 0, 0, 406 },
		{ -2959, 5454, 401 },
		{ -2960, 5244, 0 },
		{ 3257, 5218, 0 },
		{ 3201, 5196, 0 },
		{ 0, 0, 402 },
		{ 3201, 5214, 0 },
		{ -2965, 16, 0 },
		{ -2966, 5248, 0 },
		{ 2969, 0, 403 },
		{ 3201, 5215, 0 },
		{ 3257, 5414, 0 },
		{ 0, 0, 404 },
		{ 3248, 3716, 168 },
		{ 0, 0, 168 },
		{ 0, 0, 169 },
		{ 3272, 2285, 0 },
		{ 3294, 3182, 0 },
		{ 3311, 2436, 0 },
		{ 2978, 5332, 0 },
		{ 3304, 2816, 0 },
		{ 3299, 1853, 0 },
		{ 3253, 2638, 0 },
		{ 3306, 3302, 0 },
		{ 3307, 2093, 0 },
		{ 3194, 3058, 0 },
		{ 3309, 2544, 0 },
		{ 3253, 2642, 0 },
		{ 3272, 2290, 0 },
		{ 3312, 4937, 0 },
		{ 0, 0, 166 },
		{ 3127, 5001, 191 },
		{ 0, 0, 191 },
		{ 3307, 2098, 0 },
		{ 2993, 5192, 0 },
		{ 3294, 2851, 0 },
		{ 3305, 3354, 0 },
		{ 3306, 3320, 0 },
		{ 3301, 3082, 0 },
		{ 2998, 5197, 0 },
		{ 3257, 2658, 0 },
		{ 3294, 3197, 0 },
		{ 3253, 2647, 0 },
		{ 3294, 3200, 0 },
		{ 3311, 2369, 0 },
		{ 3305, 3363, 0 },
		{ 3307, 2100, 0 },
		{ 3194, 2914, 0 },
		{ 3309, 2551, 0 },
		{ 3253, 2652, 0 },
		{ 3009, 5219, 0 },
		{ 3257, 3085, 0 },
		{ 3294, 3207, 0 },
		{ 3140, 3120, 0 },
		{ 3309, 2552, 0 },
		{ 3311, 2371, 0 },
		{ 3294, 3211, 0 },
		{ 3016, 5219, 0 },
		{ 3311, 2233, 0 },
		{ 3294, 3213, 0 },
		{ 3291, 3285, 0 },
		{ 3299, 1879, 0 },
		{ 3306, 3304, 0 },
		{ 3294, 3215, 0 },
		{ 3023, 5249, 0 },
		{ 3304, 2792, 0 },
		{ 3299, 1881, 0 },
		{ 3253, 2662, 0 },
		{ 3306, 3312, 0 },
		{ 3307, 2118, 0 },
		{ 3194, 2997, 0 },
		{ 3309, 2476, 0 },
		{ 3253, 2665, 0 },
		{ 3312, 4857, 0 },
		{ 0, 0, 189 },
		{ 3034, 0, 1 },
		{ -3034, 1506, 280 },
		{ 3294, 3116, 286 },
		{ 0, 0, 286 },
		{ 3272, 2304, 0 },
		{ 3253, 2670, 0 },
		{ 3294, 3232, 0 },
		{ 3291, 3287, 0 },
		{ 3311, 2381, 0 },
		{ 0, 0, 285 },
		{ 3044, 5322, 0 },
		{ 3296, 2248, 0 },
		{ 3305, 3340, 0 },
		{ 3073, 2827, 0 },
		{ 3294, 3237, 0 },
		{ 3140, 3122, 0 },
		{ 3194, 3015, 0 },
		{ 3302, 2802, 0 },
		{ 3294, 3241, 0 },
		{ 3053, 5300, 0 },
		{ 3309, 2314, 0 },
		{ 0, 2456, 0 },
		{ 3307, 2139, 0 },
		{ 3194, 3021, 0 },
		{ 3309, 2487, 0 },
		{ 3253, 2573, 0 },
		{ 3272, 2308, 0 },
		{ 3312, 5003, 0 },
		{ 0, 0, 284 },
		{ 0, 4789, 194 },
		{ 0, 0, 194 },
		{ 3309, 2490, 0 },
		{ 3299, 1716, 0 },
		{ 3253, 2577, 0 },
		{ 3291, 3290, 0 },
		{ 3069, 5336, 0 },
		{ 3306, 3080, 0 },
		{ 3301, 3070, 0 },
		{ 3294, 3257, 0 },
		{ 3306, 3317, 0 },
		{ 0, 2829, 0 },
		{ 3194, 3030, 0 },
		{ 3253, 2578, 0 },
		{ 3096, 3092, 0 },
		{ 3312, 5079, 0 },
		{ 0, 0, 192 },
		{ 3127, 4999, 188 },
		{ 0, 0, 187 },
		{ 0, 0, 188 },
		{ 3307, 2146, 0 },
		{ 3084, 5335, 0 },
		{ 3307, 2189, 0 },
		{ 3301, 3084, 0 },
		{ 3294, 3266, 0 },
		{ 3088, 5196, 0 },
		{ 3257, 3083, 0 },
		{ 3294, 3268, 0 },
		{ 3096, 3087, 0 },
		{ 3194, 3035, 0 },
		{ 3253, 2583, 0 },
		{ 3253, 2585, 0 },
		{ 3194, 3038, 0 },
		{ 3253, 2587, 0 },
		{ 0, 3093, 0 },
		{ 3098, 5197, 0 },
		{ 3309, 2321, 0 },
		{ 3140, 3107, 0 },
		{ 3101, 5212, 0 },
		{ 3294, 2849, 0 },
		{ 3305, 3391, 0 },
		{ 3306, 3295, 0 },
		{ 3301, 3081, 0 },
		{ 3106, 5219, 0 },
		{ 3257, 2666, 0 },
		{ 3294, 3154, 0 },
		{ 3253, 2591, 0 },
		{ 3294, 3156, 0 },
		{ 3311, 2390, 0 },
		{ 3305, 3402, 0 },
		{ 3307, 2160, 0 },
		{ 3194, 3046, 0 },
		{ 3309, 2497, 0 },
		{ 3253, 2595, 0 },
		{ 3117, 5242, 0 },
		{ 3304, 2776, 0 },
		{ 3299, 1746, 0 },
		{ 3253, 2597, 0 },
		{ 3306, 3316, 0 },
		{ 3307, 2163, 0 },
		{ 3194, 3054, 0 },
		{ 3309, 2500, 0 },
		{ 3253, 2600, 0 },
		{ 3312, 4997, 0 },
		{ 0, 0, 181 },
		{ 0, 4853, 180 },
		{ 0, 0, 180 },
		{ 3307, 2165, 0 },
		{ 3131, 5246, 0 },
		{ 3307, 2193, 0 },
		{ 3301, 3065, 0 },
		{ 3294, 3173, 0 },
		{ 3135, 5277, 0 },
		{ 3294, 2832, 0 },
		{ 3253, 2603, 0 },
		{ 3291, 3279, 0 },
		{ 3139, 5274, 0 },
		{ 3309, 2342, 0 },
		{ 0, 3112, 0 },
		{ 3142, 5287, 0 },
		{ 3294, 2845, 0 },
		{ 3305, 3356, 0 },
		{ 3306, 3310, 0 },
		{ 3301, 3071, 0 },
		{ 3147, 5297, 0 },
		{ 3257, 2660, 0 },
		{ 3294, 3181, 0 },
		{ 3253, 2606, 0 },
		{ 3294, 3183, 0 },
		{ 3311, 2397, 0 },
		{ 3305, 3364, 0 },
		{ 3307, 2176, 0 },
		{ 3194, 2857, 0 },
		{ 3309, 2506, 0 },
		{ 3253, 2610, 0 },
		{ 3158, 5312, 0 },
		{ 3304, 2786, 0 },
		{ 3299, 1796, 0 },
		{ 3253, 2612, 0 },
		{ 3306, 3300, 0 },
		{ 3307, 2184, 0 },
		{ 3194, 2920, 0 },
		{ 3309, 2509, 0 },
		{ 3253, 2615, 0 },
		{ 3312, 4855, 0 },
		{ 0, 0, 178 },
		{ 0, 4396, 183 },
		{ 0, 0, 183 },
		{ 0, 0, 184 },
		{ 3253, 2616, 0 },
		{ 3272, 2325, 0 },
		{ 3307, 2186, 0 },
		{ 3294, 3199, 0 },
		{ 3305, 3384, 0 },
		{ 3291, 3286, 0 },
		{ 3178, 5338, 0 },
		{ 3294, 2843, 0 },
		{ 3276, 1671, 0 },
		{ 3305, 3389, 0 },
		{ 3302, 2815, 0 },
		{ 3299, 1845, 0 },
		{ 3305, 3393, 0 },
		{ 3307, 2203, 0 },
		{ 3194, 3000, 0 },
		{ 3309, 2515, 0 },
		{ 3253, 2623, 0 },
		{ 3189, 5189, 0 },
		{ 3304, 2790, 0 },
		{ 3299, 1846, 0 },
		{ 3253, 2625, 0 },
		{ 3306, 3299, 0 },
		{ 3307, 2206, 0 },
		{ 0, 3011, 0 },
		{ 3309, 2518, 0 },
		{ 3253, 2628, 0 },
		{ 3274, 5157, 0 },
		{ 0, 0, 182 },
		{ 3294, 3216, 455 },
		{ 3311, 1617, 26 },
		{ 0, 5221, 455 },
		{ 3210, 0, 455 },
		{ 2328, 2960, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 3253, 2632, 0 },
		{ -3209, 5455, 0 },
		{ 3311, 91, 0 },
		{ 0, 0, 28 },
		{ 3291, 3289, 0 },
		{ 0, 0, 27 },
		{ 0, 0, 22 },
		{ 0, 0, 34 },
		{ 0, 0, 35 },
		{ 3231, 4202, 39 },
		{ 0, 3916, 39 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 3243, 4067, 0 },
		{ 3258, 4751, 0 },
		{ 3248, 3690, 0 },
		{ 0, 0, 37 },
		{ 3252, 3753, 0 },
		{ 3256, 3586, 0 },
		{ 3266, 2111, 0 },
		{ 0, 0, 36 },
		{ 3294, 3131, 50 },
		{ 0, 0, 50 },
		{ 0, 4211, 50 },
		{ 3248, 3695, 50 },
		{ 3294, 3226, 50 },
		{ 0, 0, 58 },
		{ 3294, 3227, 0 },
		{ 3253, 2637, 0 },
		{ 3243, 4078, 0 },
		{ 3252, 3763, 0 },
		{ 3307, 2244, 0 },
		{ 3253, 2639, 0 },
		{ 3291, 3276, 0 },
		{ 3248, 3704, 0 },
		{ 0, 4025, 0 },
		{ 3299, 1855, 0 },
		{ 3309, 2529, 0 },
		{ 0, 0, 49 },
		{ 3252, 3770, 0 },
		{ 0, 3707, 0 },
		{ 3311, 2417, 0 },
		{ 3296, 2730, 0 },
		{ 3256, 3581, 54 },
		{ 0, 3775, 0 },
		{ 0, 2643, 0 },
		{ 3294, 3236, 0 },
		{ 3260, 1854, 0 },
		{ 0, 3585, 51 },
		{ 0, 5, 59 },
		{ 0, 4729, 0 },
		{ 3266, 1938, 0 },
		{ 3311, 1694, 0 },
		{ 3267, 1844, 0 },
		{ 0, 0, 57 },
		{ 3302, 2798, 0 },
		{ 0, 0, 55 },
		{ 0, 0, 56 },
		{ 3203, 2098, 0 },
		{ 3311, 1769, 0 },
		{ 3305, 3369, 0 },
		{ 0, 0, 52 },
		{ 0, 0, 53 },
		{ 3272, 2352, 0 },
		{ 0, 2354, 0 },
		{ 3274, 5152, 0 },
		{ 0, 5153, 0 },
		{ 3294, 3246, 0 },
		{ 0, 1714, 0 },
		{ 3305, 3374, 0 },
		{ 3302, 2804, 0 },
		{ 3299, 1909, 0 },
		{ 3305, 3377, 0 },
		{ 3307, 2263, 0 },
		{ 3309, 2545, 0 },
		{ 3311, 2430, 0 },
		{ 3305, 2371, 0 },
		{ 3294, 3254, 0 },
		{ 3309, 2548, 0 },
		{ 3306, 3303, 0 },
		{ 3305, 3385, 0 },
		{ 3311, 2432, 0 },
		{ 3306, 3306, 0 },
		{ 0, 3281, 0 },
		{ 3294, 3148, 0 },
		{ 3299, 1911, 0 },
		{ 0, 3260, 0 },
		{ 3305, 3392, 0 },
		{ 0, 2726, 0 },
		{ 3311, 2434, 0 },
		{ 3306, 3313, 0 },
		{ 0, 1913, 0 },
		{ 3312, 5149, 0 },
		{ 0, 3074, 0 },
		{ 0, 2818, 0 },
		{ 0, 0, 46 },
		{ 3257, 3082, 0 },
		{ 0, 3400, 0 },
		{ 0, 3318, 0 },
		{ 0, 2269, 0 },
		{ 3312, 5150, 0 },
		{ 0, 2554, 0 },
		{ 0, 0, 47 },
		{ 0, 2437, 0 },
		{ 3274, 5151, 0 },
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
		0
	};
	yybackup = backup;
}
