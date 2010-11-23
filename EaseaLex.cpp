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
#include "Easea.h"
#include "EaseaParse.h"
#ifdef WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif
#include "debug.h"
#include <iostream>
#include <sstream>

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
  
#line 69 "EaseaLex.cpp"
// repeated because of possible precompiled header
#include <clex.h>

#include "EaseaLex.h"

#line 73 "EaseaLex.l"
                                         
#line 81 "EaseaLex.l"
 // lexical analyser name and class definition
#line 79 "EaseaLex.cpp"
/////////////////////////////////////////////////////////////////////////////
// constructor

YYLEXNAME::YYLEXNAME()
{
	yytables();
#line 110 "EaseaLex.l"
                                
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

#line 99 "EaseaLex.cpp"
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
#line 130 "EaseaLex.l"

  // extract yylval for use later on in actions
  YYSTYPE& yylval = *(YYSTYPE*)yyparserptr->yylvalptr;
  
#line 150 "EaseaLex.cpp"
	yyreturnflg = 1;
	switch (action) {
	case 1:
		{
#line 136 "EaseaLex.l"

#line 157 "EaseaLex.cpp"
		}
		break;
	case 2:
		{
#line 139 "EaseaLex.l"

  BEGIN TEMPLATE_ANALYSIS; yyless(yyleng-1);
 
#line 166 "EaseaLex.cpp"
		}
		break;
	case 3:
		{
#line 147 "EaseaLex.l"
             
  char sFileName[1000];
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".cpp"); 
  fpOutputFile=fopen(sFileName,"w");
 
#line 178 "EaseaLex.cpp"
		}
		break;
	case 4:
		{
#line 153 "EaseaLex.l"
fprintf(fpOutputFile,"EASEA");
#line 185 "EaseaLex.cpp"
		}
		break;
	case 5:
		{
#line 154 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sPROJECT_NAME);
#line 192 "EaseaLex.cpp"
		}
		break;
	case 6:
		{
#line 155 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sEZ_PATH);
#line 199 "EaseaLex.cpp"
		}
		break;
	case 7:
		{
#line 156 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sTPL_DIR);
#line 206 "EaseaLex.cpp"
		}
		break;
	case 8:
		{
#line 157 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sEO_DIR);
#line 213 "EaseaLex.cpp"
		}
		break;
	case 9:
		{
#line 158 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sLOWER_CASE_PROJECT_NAME);
#line 220 "EaseaLex.cpp"
		}
		break;
	case 10:
		{
#line 159 "EaseaLex.l"
switch (OPERATING_SYSTEM) {
  case UNIX : fprintf(fpOutputFile,"UNIX_OS"); break;
  case WINDOWS : fprintf(fpOutputFile,"WINDOWS_OS"); break;
  case UNKNOWN_OS : fprintf(fpOutputFile,"UNKNOWN_OS"); break;
  }
 
#line 232 "EaseaLex.cpp"
		}
		break;
	case 11:
		{
#line 165 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user declarations.\n");
  yyreset();
  yyin = fpGenomeFile;                                                    // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_USER_DECLARATIONS;
 
#line 245 "EaseaLex.cpp"
		}
		break;
	case 12:
		{
#line 173 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user cuda.\n");
  yyreset();
  yyin = fpGenomeFile;                                                    // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_USER_CUDA;
 
#line 258 "EaseaLex.cpp"
		}
		break;
	case 13:
		{
#line 181 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting initialisation function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISATION_FUNCTION;
 
#line 271 "EaseaLex.cpp"
		}
		break;
	case 14:
		{
#line 189 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting generation before reduce function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bGenerationFunctionBeforeReplacement = true;
  BEGIN COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT;
 
#line 284 "EaseaLex.cpp"
		}
		break;
	case 15:
		{
#line 198 "EaseaLex.l"

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
 
#line 303 "EaseaLex.cpp"
		}
		break;
	case 16:
		{
#line 217 "EaseaLex.l"

  if( bVERBOSE )printf("inserting gp parameters\n");
  //  fprintf(fpOutputFile,"#define MAX_XOVER_DEPTH",%d
  fprintf(fpOutputFile,"#define TREE_DEPTH_MAX %d\n",iMAX_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MAX %d\n",iMAX_INIT_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MIN %d\n",iMIN_INIT_TREE_D);

  fprintf(fpOutputFile,"#define MAX_PROGS_SIZE %d\n",iPRG_BUF_SIZE);
  fprintf(fpOutputFile,"#define NB_GPU %d\n",iNB_GPU);

  fprintf(fpOutputFile,"#define NO_FITNESS_CASES %d\n",iNO_FITNESS_CASES);

#line 321 "EaseaLex.cpp"
		}
		break;
	case 17:
		{
#line 235 "EaseaLex.l"

  
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
 
#line 358 "EaseaLex.cpp"
		}
		break;
	case 18:
		{
#line 267 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"    case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"      %s",opDesc[i]->gpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"      break;\n");

  }
 
#line 372 "EaseaLex.cpp"
		}
		break;
	case 19:
		{
#line 276 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"  case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"    %s\n",opDesc[i]->cpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"    break;\n");
  }
 
#line 385 "EaseaLex.cpp"
		}
		break;
	case 20:
		{
#line 285 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Analysing GP OP code from ez file\n");
  BEGIN COPY_GP_OPCODE;
 
#line 398 "EaseaLex.cpp"
		}
		break;
	case 21:
		{
#line 294 "EaseaLex.l"

  if (bVERBOSE) printf ("found begin section\n");
  bGPOPCODE_ANALYSIS = true;
  BEGIN GP_RULE_ANALYSIS;
 
#line 409 "EaseaLex.cpp"
		}
		break;
	case 22:
		{
#line 300 "EaseaLex.l"
 
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
 
#line 429 "EaseaLex.cpp"
		}
		break;
	case 23:
		{
#line 315 "EaseaLex.l"

  if (bVERBOSE) printf("*** No GP OP codes were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
 
#line 441 "EaseaLex.cpp"
		}
		break;
	case 24:
		{
#line 321 "EaseaLex.l"

#line 448 "EaseaLex.cpp"
		}
		break;
	case 25:
		{
#line 322 "EaseaLex.l"
if( bGPOPCODE_ANALYSIS )printf("\n");lineCounter++;
#line 455 "EaseaLex.cpp"
		}
		break;
	case 26:
		{
#line 329 "EaseaLex.l"

  // this rule match the OP_NAME
  if( iGP_OPCODE_FIELD != 0 ){
    fprintf(stderr,"Error, OP_CODE name must be given first\n");
    exit(-1);
  }
  opDesc[iNoOp] = new OPCodeDesc();
  opDesc[iNoOp]->opcode = new string(yytext);
 
#line 470 "EaseaLex.cpp"
		}
		break;
	case 27:
		{
#line 339 "EaseaLex.l"

#line 477 "EaseaLex.cpp"
		}
		break;
	case 28:
		{
#line 341 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 1 ){
    fprintf(stderr,"Error, op code real name must be given at the second place\n");
    exit(-1);
  }
  opDesc[iNoOp]->realName = new string(yytext);
 
#line 490 "EaseaLex.cpp"
		}
		break;
	case 29:
		{
#line 350 "EaseaLex.l"

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
 
#line 509 "EaseaLex.cpp"
		}
		break;
	case 30:
		{
#line 364 "EaseaLex.l"

#line 516 "EaseaLex.cpp"
		}
		break;
	case 31:
		{
#line 365 "EaseaLex.l"

  iGP_OPCODE_FIELD = 0;
  iNoOp++;
 
#line 526 "EaseaLex.cpp"
		}
		break;
	case 32:
		{
#line 370 "EaseaLex.l"

  if( bGPOPCODE_ANALYSIS ) iGP_OPCODE_FIELD++;
 
#line 535 "EaseaLex.cpp"
		}
		break;
	case 33:
		{
#line 375 "EaseaLex.l"

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
 
#line 557 "EaseaLex.cpp"
		}
		break;
	case 34:
		{
#line 396 "EaseaLex.l"

  accolade_counter++;
  opDesc[iNoOp]->cpuCodeStream << "{";
  opDesc[iNoOp]->gpuCodeStream << "{";
 
#line 568 "EaseaLex.cpp"
		}
		break;
	case 35:
		{
#line 402 "EaseaLex.l"

  accolade_counter--;
  if( accolade_counter==0 ){
    opDesc[iNoOp]->gpuCodeStream << "\n      stack[sp++] = RESULT;\n";

    BEGIN GP_RULE_ANALYSIS;
  }
  else{
    opDesc[iNoOp]->cpuCodeStream << "}";
    opDesc[iNoOp]->gpuCodeStream << "}";
  }
 
#line 586 "EaseaLex.cpp"
		}
		break;
	case 36:
		{
#line 415 "EaseaLex.l"

  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
//  printf("input no : %d\n",no_input);
  opDesc[iNoOp]->cpuCodeStream << "input["<< no_input <<"]" ;
  opDesc[iNoOp]->gpuCodeStream << "input["<< no_input << "]";  
 
#line 599 "EaseaLex.cpp"
		}
		break;
	case 37:
		{
#line 423 "EaseaLex.l"

  opDesc[iNoOp]->isERC = true;
  opDesc[iNoOp]->cpuCodeStream << "root->erc_value" ;
  opDesc[iNoOp]->gpuCodeStream << "k_progs[start_prog++];" ;
//  printf("ERC matched\n");

#line 611 "EaseaLex.cpp"
		}
		break;
	case 38:
		{
#line 430 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << "\n  ";
  opDesc[iNoOp]->gpuCodeStream << "\n    ";
 
#line 621 "EaseaLex.cpp"
		}
		break;
	case 39:
		{
#line 436 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << yytext;
  opDesc[iNoOp]->gpuCodeStream << yytext;
 
#line 631 "EaseaLex.cpp"
		}
		break;
	case 40:
		{
#line 441 "EaseaLex.l"
 
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  if( bVERBOSE ) printf("Insert GP eval header\n");
  iCOPY_GP_EVAL_STATUS = EVAL_HDR;
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 647 "EaseaLex.cpp"
		}
		break;
	case 41:
		{
#line 452 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = false;
  BEGIN COPY_GP_EVAL;
 
#line 664 "EaseaLex.cpp"
		}
		break;
	case 42:
		{
#line 466 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 681 "EaseaLex.cpp"
		}
		break;
	case 43:
		{
#line 480 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 697 "EaseaLex.cpp"
		}
		break;
	case 44:
		{
#line 491 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 714 "EaseaLex.cpp"
		}
		break;
	case 45:
		{
#line 504 "EaseaLex.l"

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
 
#line 733 "EaseaLex.cpp"
		}
		break;
	case 46:
		{
#line 519 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_HDR){
    bIsCopyingGPEval = true;
  }
 
#line 744 "EaseaLex.cpp"
		}
		break;
	case 47:
		{
#line 525 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_BDY){
    bIsCopyingGPEval = true;
  }
 
#line 755 "EaseaLex.cpp"
		}
		break;
	case 48:
		{
#line 533 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_FTR){
    bIsCopyingGPEval = true;
  }
 
#line 766 "EaseaLex.cpp"
		}
		break;
	case 49:
		{
#line 539 "EaseaLex.l"

  if( bIsCopyingGPEval ){
    bIsCopyingGPEval = false;
    bCOPY_GP_EVAL_GPU = false;
    rewind(fpGenomeFile);
    yyin = fpTemplateFile;
    BEGIN TEMPLATE_ANALYSIS;
  }
 
#line 781 "EaseaLex.cpp"
		}
		break;
	case 50:
		{
#line 549 "EaseaLex.l"

  if( bIsCopyingGPEval ) fprintf(fpOutputFile,"%s",yytext);
 
#line 790 "EaseaLex.cpp"
		}
		break;
	case 51:
		{
#line 553 "EaseaLex.l"

  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[i*NUMTHREAD2+tid]" );
    else fprintf(fpOutputFile, "outputs[i]" );
  
 
#line 803 "EaseaLex.cpp"
		}
		break;
	case 52:
		{
#line 561 "EaseaLex.l"

  char* endptr;
  unsigned no_output = strtol(yytext+strlen("OUTPUT["),&endptr,10);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[(i+%d)*NUMTHREAD2+tid]", no_output);
    else fprintf(fpOutputFile, "outputs[i+%d]", no_output );
  
 
#line 818 "EaseaLex.cpp"
		}
		break;
	case 53:
		{
#line 571 "EaseaLex.l"

	char *var;
	var = strndup(yytext+strlen("OUTPUT["), strlen(yytext) - strlen("OUTPUT[") - 1);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[(i+%s)*NUMTHREAD2+tid]", var);
    else fprintf(fpOutputFile, "outputs[i+%s]", var);
  
 
#line 833 "EaseaLex.cpp"
		}
		break;
	case 54:
		{
#line 581 "EaseaLex.l"

  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[i*NUMTHREAD2+tid]" );
    else fprintf(fpOutputFile, "inputs[i][0]" );
  
 
#line 846 "EaseaLex.cpp"
		}
		break;
	case 55:
		{
#line 589 "EaseaLex.l"

  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[(i+%d)*NUMTHREAD2+tid]", no_input);
    else fprintf(fpOutputFile, "inputs[i+%d][0]", no_input );
  
 
#line 861 "EaseaLex.cpp"
		}
		break;
	case 56:
		{
#line 599 "EaseaLex.l"

	char *var;
	var = strndup(yytext+strlen("INPUT["), strlen(yytext) - strlen("INPUT[") - 1);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[(i+%s)*NUMTHREAD2+tid]", var);
    else fprintf(fpOutputFile, "inputs[i+%s][0]", var);
  
 
#line 876 "EaseaLex.cpp"
		}
		break;
	case 57:
		{
#line 609 "EaseaLex.l"

  if( bIsCopyingGPEval )
    if( iCOPY_GP_EVAL_STATUS==EVAL_FTR )
      if( bCOPY_GP_EVAL_GPU ){
	fprintf(fpOutputFile,"k_results[index] =");
      }
      else fprintf(fpOutputFile,"%s",yytext);
 
#line 890 "EaseaLex.cpp"
		}
		break;
	case 58:
		{
#line 621 "EaseaLex.l"

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
 
#line 908 "EaseaLex.cpp"
		}
		break;
	case 59:
		{
#line 635 "EaseaLex.l"

  if( bIsCopyingGPEval )
    fprintf(fpOutputFile,"return fitness = "); 
 
#line 918 "EaseaLex.cpp"
		}
		break;
	case 60:
		{
#line 642 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  yyreset();
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Evaluation population in a single function!!.\n");
  lineCounter = 1;
  BEGIN COPY_INSTEAD_EVAL;
 
#line 932 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 651 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting at the end of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bEndGeneration = true;
  bBeginGeneration = false;
  BEGIN COPY_END_GENERATION_FUNCTION;
 
#line 946 "EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 660 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Bound Checking function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_BOUND_CHECKING_FUNCTION;
 
#line 958 "EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 667 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 
#line 970 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 674 "EaseaLex.l"

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
 
#line 999 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 697 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 
#line 1016 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 709 "EaseaLex.l"

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
 
#line 1042 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 730 "EaseaLex.l"

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
  
 
#line 1063 "EaseaLex.cpp"
		}
		break;
#line 748 "EaseaLex.l"
  
#line 762 "EaseaLex.l"
      
#line 1070 "EaseaLex.cpp"
	case 68:
		{
#line 770 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 1083 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 779 "EaseaLex.l"

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
 
#line 1106 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 796 "EaseaLex.l"

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
 
#line 1129 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 814 "EaseaLex.l"

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
 
#line 1161 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 841 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  fprintf (fpOutputFile,"// Memberwise serialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->serializeIndividual(fpOutputFile, "this");
  //fprintf(fpOutputFile,"\tEASEA_Line << endl;\n");
 
#line 1175 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 850 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome deserializer.\n");
  fprintf (fpOutputFile,"// Memberwise deserialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->deserializeIndividual(fpOutputFile, "this");
 
#line 1188 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 858 "EaseaLex.l"

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
 
#line 1209 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 874 "EaseaLex.l"

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
 
#line 1231 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 891 "EaseaLex.l"
       
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
 
#line 1259 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 913 "EaseaLex.l"

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
 
#line 1281 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 929 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 1296 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 938 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 1308 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 946 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 1320 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 953 "EaseaLex.l"

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
 
#line 1351 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 978 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 1364 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 985 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 1378 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 994 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISER;   
 
#line 1390 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 1001 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 1403 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 1009 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 1415 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 1015 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 1427 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 1021 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 1439 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 1027 "EaseaLex.l"
      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 1452 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 1034 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 1465 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 1041 "EaseaLex.l"
 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 1479 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 1050 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 1490 "EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 1055 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 1504 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 1064 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 1518 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 1073 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 1532 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 1083 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 1545 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 1091 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 1554 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 1095 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 1563 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 1099 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 1572 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 1103 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 1581 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 1107 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 1591 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 1112 "EaseaLex.l"

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

#line 1610 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 1125 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 1617 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 1126 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 1624 "EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 1127 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 1631 "EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 1128 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 1638 "EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 1129 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 1645 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 1130 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1652 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 1131 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1659 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 1132 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1666 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 1133 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1673 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 1134 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1680 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 1135 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",nELITE);
  ////DEBUG_PRT_PRT("elitism is %d, elite size is %d",bELITISM, nELITE);
 
#line 1690 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 1140 "EaseaLex.l"

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
 
#line 1709 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 1153 "EaseaLex.l"

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
 
#line 1728 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 1166 "EaseaLex.l"

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
 
#line 1747 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 1179 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1757 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 1183 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1764 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 1184 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1771 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 1185 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1778 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 1186 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1785 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1187 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1792 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1188 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1799 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1189 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1806 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1190 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1813 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1191 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1820 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1192 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1827 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1194 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1834 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1195 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1841 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1197 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bREMOTE_ISLAND_MODEL);
#line 1848 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1198 "EaseaLex.l"
if(strlen(sIP_FILE)>0)fprintf(fpOutputFile,"%s",sIP_FILE); else fprintf(fpOutputFile,"NULL");
#line 1855 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1199 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMIGRATION_PROBABILITY);
#line 1862 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1201 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1869 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1202 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1876 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1203 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1883 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1204 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1890 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1205 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1897 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1207 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 1904 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1208 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 1911 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1210 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1925 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1218 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  if( TARGET==CUDA )
    strcat(sFileName,"Individual.cu");
  else if( TARGET==STD )
    if( TARGET_FLAVOR==CUDA_FLAVOR_GP )
      strcat(sFileName,"Individual.cu");
    else
      strcat(sFileName,"Individual.cpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1945 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1232 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1959 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1240 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1973 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1249 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1987 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1258 "EaseaLex.l"

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

#line 2050 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1315 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded without warnings.\n");
  printf ("You can now type \"make\" to compile your project.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 2067 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1327 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2074 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1333 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2086 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1339 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2099 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1346 "EaseaLex.l"

#line 2106 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1347 "EaseaLex.l"
lineCounter++;
#line 2113 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1349 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2125 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1355 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2138 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1363 "EaseaLex.l"

#line 2145 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1364 "EaseaLex.l"

  lineCounter++;
 
#line 2154 "EaseaLex.cpp"
		}
		break;
#line 1367 "EaseaLex.l"
               
#line 2159 "EaseaLex.cpp"
	case 156:
		{
#line 1368 "EaseaLex.l"

  fprintf (fpOutputFile,"// User CUDA\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2169 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1374 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user CUDA were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2182 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1382 "EaseaLex.l"

#line 2189 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1383 "EaseaLex.l"

  lineCounter++;
 
#line 2198 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1387 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2210 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1393 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2224 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1401 "EaseaLex.l"

#line 2231 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1402 "EaseaLex.l"

  lineCounter++;
 
#line 2240 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1406 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
      
  BEGIN COPY;
 
#line 2254 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1414 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2269 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1423 "EaseaLex.l"

#line 2276 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1424 "EaseaLex.l"
lineCounter++;
#line 2283 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1429 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2297 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1438 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2311 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1446 "EaseaLex.l"

#line 2318 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1447 "EaseaLex.l"
lineCounter++;
#line 2325 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1450 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2341 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1461 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2357 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1471 "EaseaLex.l"

#line 2364 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1474 "EaseaLex.l"

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
 
#line 2382 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1487 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2399 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1499 "EaseaLex.l"

#line 2406 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1500 "EaseaLex.l"
lineCounter++;
#line 2413 "EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 1502 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2429 "EaseaLex.cpp"
		}
		break;
	case 180:
		{
#line 1514 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2445 "EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 1524 "EaseaLex.l"
lineCounter++;
#line 2452 "EaseaLex.cpp"
		}
		break;
	case 182:
		{
#line 1525 "EaseaLex.l"

#line 2459 "EaseaLex.cpp"
		}
		break;
	case 183:
		{
#line 1529 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2474 "EaseaLex.cpp"
		}
		break;
	case 184:
		{
#line 1539 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2489 "EaseaLex.cpp"
		}
		break;
	case 185:
		{
#line 1548 "EaseaLex.l"

#line 2496 "EaseaLex.cpp"
		}
		break;
	case 186:
		{
#line 1551 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2509 "EaseaLex.cpp"
		}
		break;
	case 187:
		{
#line 1558 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2523 "EaseaLex.cpp"
		}
		break;
	case 188:
		{
#line 1566 "EaseaLex.l"

#line 2530 "EaseaLex.cpp"
		}
		break;
	case 189:
		{
#line 1570 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2538 "EaseaLex.cpp"
		}
		break;
	case 190:
		{
#line 1572 "EaseaLex.l"

#line 2545 "EaseaLex.cpp"
		}
		break;
	case 191:
		{
#line 1578 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2552 "EaseaLex.cpp"
		}
		break;
	case 192:
		{
#line 1579 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2559 "EaseaLex.cpp"
		}
		break;
	case 193:
	case 194:
		{
#line 1582 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2570 "EaseaLex.cpp"
		}
		break;
	case 195:
	case 196:
		{
#line 1587 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2579 "EaseaLex.cpp"
		}
		break;
	case 197:
	case 198:
		{
#line 1590 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2588 "EaseaLex.cpp"
		}
		break;
	case 199:
	case 200:
		{
#line 1593 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2605 "EaseaLex.cpp"
		}
		break;
	case 201:
	case 202:
		{
#line 1604 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2619 "EaseaLex.cpp"
		}
		break;
	case 203:
	case 204:
		{
#line 1612 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2628 "EaseaLex.cpp"
		}
		break;
	case 205:
	case 206:
		{
#line 1615 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2637 "EaseaLex.cpp"
		}
		break;
	case 207:
	case 208:
		{
#line 1618 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2646 "EaseaLex.cpp"
		}
		break;
	case 209:
	case 210:
		{
#line 1621 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2655 "EaseaLex.cpp"
		}
		break;
	case 211:
	case 212:
		{
#line 1624 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2664 "EaseaLex.cpp"
		}
		break;
	case 213:
	case 214:
		{
#line 1628 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2676 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1634 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2683 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1635 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2690 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1636 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2697 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1637 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2707 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1642 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2714 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1643 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2721 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1644 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2728 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1645 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2735 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1646 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2742 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1647 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2749 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1648 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2756 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1649 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2763 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1650 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2771 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1652 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2779 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1654 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2787 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1656 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2797 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1660 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2804 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1661 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2811 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1662 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2822 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1667 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2829 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1668 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2838 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1671 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2850 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1677 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2859 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1680 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2871 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1686 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2882 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1691 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2898 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1701 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2905 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1704 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2914 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1707 "EaseaLex.l"
BEGIN COPY;
#line 2921 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1709 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2928 "EaseaLex.cpp"
		}
		break;
	case 245:
	case 246:
	case 247:
		{
#line 1712 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2941 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1717 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2952 "EaseaLex.cpp"
		}
		break;
	case 249:
		{
#line 1722 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 2961 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1731 "EaseaLex.l"
;
#line 2968 "EaseaLex.cpp"
		}
		break;
	case 251:
		{
#line 1732 "EaseaLex.l"
;
#line 2975 "EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1733 "EaseaLex.l"
;
#line 2982 "EaseaLex.cpp"
		}
		break;
	case 253:
		{
#line 1734 "EaseaLex.l"
;
#line 2989 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1737 "EaseaLex.l"
 /* do nothing */ 
#line 2996 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1738 "EaseaLex.l"
 /*return '\n';*/ 
#line 3003 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1739 "EaseaLex.l"
 /*return '\n';*/ 
#line 3010 "EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1742 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 3019 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1745 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 3029 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1749 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
//  printf("match gpnode\n");
  return GPNODE;
 
#line 3041 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1756 "EaseaLex.l"
return STATIC;
#line 3048 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1757 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 3055 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1758 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 3062 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1759 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 3069 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1760 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 3076 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1761 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 3083 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1763 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 3090 "EaseaLex.cpp"
		}
		break;
#line 1764 "EaseaLex.l"
  
#line 3095 "EaseaLex.cpp"
	case 267:
		{
#line 1765 "EaseaLex.l"
return GENOME; 
#line 3100 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1767 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 3110 "EaseaLex.cpp"
		}
		break;
	case 269:
	case 270:
	case 271:
		{
#line 1774 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 3119 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1775 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 3126 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1778 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 3134 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1780 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 3141 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1786 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 3153 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1792 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 3166 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1799 "EaseaLex.l"

#line 3173 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1801 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3184 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1812 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3199 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1822 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3210 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1828 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3219 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1832 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3234 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1845 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3246 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1851 "EaseaLex.l"

#line 3253 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1852 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3266 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1859 "EaseaLex.l"

#line 3273 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1860 "EaseaLex.l"
lineCounter++;
#line 3280 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1861 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3293 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1868 "EaseaLex.l"

#line 3300 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1869 "EaseaLex.l"
lineCounter++;
#line 3307 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1871 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3320 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1878 "EaseaLex.l"

#line 3327 "EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1879 "EaseaLex.l"
lineCounter++;
#line 3334 "EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1881 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3347 "EaseaLex.cpp"
		}
		break;
	case 295:
		{
#line 1888 "EaseaLex.l"

#line 3354 "EaseaLex.cpp"
		}
		break;
	case 296:
		{
#line 1889 "EaseaLex.l"
lineCounter++;
#line 3361 "EaseaLex.cpp"
		}
		break;
	case 297:
		{
#line 1895 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3368 "EaseaLex.cpp"
		}
		break;
	case 298:
		{
#line 1896 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3375 "EaseaLex.cpp"
		}
		break;
	case 299:
		{
#line 1897 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3382 "EaseaLex.cpp"
		}
		break;
	case 300:
		{
#line 1898 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3389 "EaseaLex.cpp"
		}
		break;
	case 301:
		{
#line 1899 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3396 "EaseaLex.cpp"
		}
		break;
	case 302:
		{
#line 1900 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3403 "EaseaLex.cpp"
		}
		break;
	case 303:
		{
#line 1901 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3410 "EaseaLex.cpp"
		}
		break;
	case 304:
		{
#line 1903 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3419 "EaseaLex.cpp"
		}
		break;
	case 305:
		{
#line 1906 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3432 "EaseaLex.cpp"
		}
		break;
	case 306:
	case 307:
		{
#line 1915 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3443 "EaseaLex.cpp"
		}
		break;
	case 308:
	case 309:
		{
#line 1920 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3452 "EaseaLex.cpp"
		}
		break;
	case 310:
	case 311:
		{
#line 1923 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3461 "EaseaLex.cpp"
		}
		break;
	case 312:
	case 313:
		{
#line 1926 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3473 "EaseaLex.cpp"
		}
		break;
	case 314:
	case 315:
		{
#line 1932 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3486 "EaseaLex.cpp"
		}
		break;
	case 316:
	case 317:
		{
#line 1939 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3495 "EaseaLex.cpp"
		}
		break;
	case 318:
	case 319:
		{
#line 1942 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3504 "EaseaLex.cpp"
		}
		break;
	case 320:
	case 321:
		{
#line 1945 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3513 "EaseaLex.cpp"
		}
		break;
	case 322:
	case 323:
		{
#line 1948 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3522 "EaseaLex.cpp"
		}
		break;
	case 324:
	case 325:
		{
#line 1951 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3531 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1954 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3540 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1957 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3550 "EaseaLex.cpp"
		}
		break;
	case 328:
		{
#line 1961 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3558 "EaseaLex.cpp"
		}
		break;
	case 329:
		{
#line 1963 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3569 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1968 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3580 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1973 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3588 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 1975 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3596 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 1977 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3604 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 1979 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3612 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 1981 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3620 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 1983 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3627 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 1984 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3634 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1985 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3642 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 1987 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3650 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 1989 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3658 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 1991 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3665 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 1992 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3677 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 1998 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3686 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 2001 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3696 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 2005 "EaseaLex.l"
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
#line 3713 "EaseaLex.cpp"
		}
		break;
	case 346:
	case 347:
		{
#line 2017 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3723 "EaseaLex.cpp"
		}
		break;
	case 348:
		{
#line 2020 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3730 "EaseaLex.cpp"
		}
		break;
	case 349:
		{
#line 2027 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3737 "EaseaLex.cpp"
		}
		break;
	case 350:
		{
#line 2028 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3744 "EaseaLex.cpp"
		}
		break;
	case 351:
		{
#line 2029 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3751 "EaseaLex.cpp"
		}
		break;
	case 352:
		{
#line 2030 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3758 "EaseaLex.cpp"
		}
		break;
	case 353:
		{
#line 2031 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3765 "EaseaLex.cpp"
		}
		break;
	case 354:
		{
#line 2033 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3774 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 2037 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3787 "EaseaLex.cpp"
		}
		break;
	case 356:
		{
#line 2045 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3800 "EaseaLex.cpp"
		}
		break;
	case 357:
		{
#line 2054 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3813 "EaseaLex.cpp"
		}
		break;
	case 358:
		{
#line 2063 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3828 "EaseaLex.cpp"
		}
		break;
	case 359:
		{
#line 2073 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3835 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 2074 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3842 "EaseaLex.cpp"
		}
		break;
	case 361:
	case 362:
		{
#line 2077 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3853 "EaseaLex.cpp"
		}
		break;
	case 363:
	case 364:
		{
#line 2082 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3862 "EaseaLex.cpp"
		}
		break;
	case 365:
	case 366:
		{
#line 2085 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3871 "EaseaLex.cpp"
		}
		break;
	case 367:
	case 368:
		{
#line 2088 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3884 "EaseaLex.cpp"
		}
		break;
	case 369:
	case 370:
		{
#line 2095 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3897 "EaseaLex.cpp"
		}
		break;
	case 371:
	case 372:
		{
#line 2102 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3906 "EaseaLex.cpp"
		}
		break;
	case 373:
		{
#line 2105 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3913 "EaseaLex.cpp"
		}
		break;
	case 374:
		{
#line 2106 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3920 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2107 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3927 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2108 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3937 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2113 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3944 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2114 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3951 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2115 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3958 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2116 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3965 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2117 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3973 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2119 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3981 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2121 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3989 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2123 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3997 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2125 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 4005 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2127 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 4013 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2129 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 4021 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2131 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 4028 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2132 "EaseaLex.l"
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
#line 4051 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2149 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 4062 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2154 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 4076 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2162 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 4083 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2168 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 4093 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2172 "EaseaLex.l"

#line 4100 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2175 "EaseaLex.l"
;
#line 4107 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2176 "EaseaLex.l"
;
#line 4114 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2177 "EaseaLex.l"
;
#line 4121 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2178 "EaseaLex.l"
;
#line 4128 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2180 "EaseaLex.l"
 /* do nothing */ 
#line 4135 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2181 "EaseaLex.l"
 /*return '\n';*/ 
#line 4142 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2182 "EaseaLex.l"
 /*return '\n';*/ 
#line 4149 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2184 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 4156 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2185 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 4163 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2186 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 4170 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2187 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4177 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2188 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4184 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2189 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4191 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2190 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4198 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2191 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4205 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2192 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4212 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2194 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4219 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2195 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4226 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2196 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4233 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2197 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4240 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2198 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4247 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2200 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4254 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2201 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4261 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2203 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4272 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2208 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4279 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2210 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4290 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2215 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4297 "EaseaLex.cpp"
		}
		break;
	case 422:
		{
#line 2218 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4304 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2219 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4311 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2220 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4318 "EaseaLex.cpp"
		}
		break;
	case 425:
		{
#line 2221 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4325 "EaseaLex.cpp"
		}
		break;
	case 426:
		{
#line 2222 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4332 "EaseaLex.cpp"
		}
		break;
#line 2223 "EaseaLex.l"
 
#line 4337 "EaseaLex.cpp"
	case 427:
		{
#line 2224 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4342 "EaseaLex.cpp"
		}
		break;
	case 428:
		{
#line 2225 "EaseaLex.l"
if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;
#line 4349 "EaseaLex.cpp"
		}
		break;
	case 429:
		{
#line 2226 "EaseaLex.l"
if(bVERBOSE) printf("\tMigration Probability...\n"); return MIGRATION_PROBABILITY;
#line 4356 "EaseaLex.cpp"
		}
		break;
#line 2228 "EaseaLex.l"
 
#line 4361 "EaseaLex.cpp"
	case 430:
	case 431:
	case 432:
		{
#line 2232 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4368 "EaseaLex.cpp"
		}
		break;
	case 433:
		{
#line 2233 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4375 "EaseaLex.cpp"
		}
		break;
	case 434:
		{
#line 2236 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4383 "EaseaLex.cpp"
		}
		break;
	case 435:
		{
#line 2239 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4390 "EaseaLex.cpp"
		}
		break;
	case 436:
		{
#line 2241 "EaseaLex.l"

  lineCounter++;

#line 4399 "EaseaLex.cpp"
		}
		break;
	case 437:
		{
#line 2244 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4409 "EaseaLex.cpp"
		}
		break;
	case 438:
		{
#line 2249 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4419 "EaseaLex.cpp"
		}
		break;
	case 439:
		{
#line 2254 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4429 "EaseaLex.cpp"
		}
		break;
	case 440:
		{
#line 2259 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4439 "EaseaLex.cpp"
		}
		break;
	case 441:
		{
#line 2264 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4449 "EaseaLex.cpp"
		}
		break;
	case 442:
		{
#line 2269 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4459 "EaseaLex.cpp"
		}
		break;
	case 443:
		{
#line 2278 "EaseaLex.l"
return  (char)yytext[0];
#line 4466 "EaseaLex.cpp"
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
#line 2280 "EaseaLex.l"


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

#line 4663 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		193,
		-194,
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
		205,
		-206,
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
		203,
		-204,
		0,
		201,
		-202,
		0,
		-243,
		0,
		-249,
		0,
		365,
		-366,
		0,
		367,
		-368,
		0,
		361,
		-362,
		0,
		310,
		-311,
		0,
		312,
		-313,
		0,
		306,
		-307,
		0,
		318,
		-319,
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
		308,
		-309,
		0,
		371,
		-372,
		0,
		316,
		-317,
		0,
		369,
		-370,
		0,
		314,
		-315,
		0,
		363,
		-364,
		0
	};
	yymatch = match;

	yytransitionmax = 5211;
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
		{ 3072, 63 },
		{ 3072, 63 },
		{ 1902, 2005 },
		{ 1527, 1510 },
		{ 1528, 1510 },
		{ 2408, 2381 },
		{ 2408, 2381 },
		{ 2001, 1997 },
		{ 2227, 2229 },
		{ 2380, 2350 },
		{ 2380, 2350 },
		{ 73, 3 },
		{ 74, 3 },
		{ 2261, 45 },
		{ 2262, 45 },
		{ 71, 1 },
		{ 2831, 2833 },
		{ 0, 2283 },
		{ 69, 1 },
		{ 167, 169 },
		{ 1902, 1883 },
		{ 0, 89 },
		{ 0, 1841 },
		{ 3072, 63 },
		{ 1384, 1383 },
		{ 3070, 63 },
		{ 1527, 1510 },
		{ 3130, 3125 },
		{ 2408, 2381 },
		{ 1375, 1374 },
		{ 1564, 1548 },
		{ 1565, 1548 },
		{ 2380, 2350 },
		{ 2228, 2224 },
		{ 73, 3 },
		{ 3074, 63 },
		{ 2261, 45 },
		{ 88, 63 },
		{ 3069, 63 },
		{ 3071, 63 },
		{ 3071, 63 },
		{ 3071, 63 },
		{ 3071, 63 },
		{ 3071, 63 },
		{ 3071, 63 },
		{ 3071, 63 },
		{ 3071, 63 },
		{ 3071, 63 },
		{ 3071, 63 },
		{ 72, 3 },
		{ 3073, 63 },
		{ 2260, 45 },
		{ 1617, 1611 },
		{ 1564, 1548 },
		{ 2409, 2381 },
		{ 1529, 1510 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 1566, 1548 },
		{ 3067, 63 },
		{ 1619, 1613 },
		{ 1490, 1469 },
		{ 3068, 63 },
		{ 1491, 1470 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3068, 63 },
		{ 3075, 63 },
		{ 2222, 42 },
		{ 1567, 1549 },
		{ 1568, 1549 },
		{ 1485, 1463 },
		{ 2009, 42 },
		{ 2465, 2437 },
		{ 2465, 2437 },
		{ 2385, 2354 },
		{ 2385, 2354 },
		{ 2388, 2357 },
		{ 2388, 2357 },
		{ 2024, 41 },
		{ 1486, 1464 },
		{ 1835, 39 },
		{ 2406, 2379 },
		{ 2406, 2379 },
		{ 1487, 1465 },
		{ 1489, 1468 },
		{ 1492, 1471 },
		{ 1493, 1472 },
		{ 1494, 1473 },
		{ 1495, 1474 },
		{ 1496, 1475 },
		{ 2222, 42 },
		{ 1567, 1549 },
		{ 2012, 42 },
		{ 1497, 1476 },
		{ 1498, 1477 },
		{ 2465, 2437 },
		{ 1499, 1478 },
		{ 2385, 2354 },
		{ 1500, 1480 },
		{ 2388, 2357 },
		{ 1503, 1483 },
		{ 2024, 41 },
		{ 1504, 1484 },
		{ 1835, 39 },
		{ 2406, 2379 },
		{ 2221, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2010, 41 },
		{ 2025, 42 },
		{ 1822, 39 },
		{ 1505, 1485 },
		{ 1569, 1549 },
		{ 2466, 2437 },
		{ 1506, 1486 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2011, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2019, 42 },
		{ 2017, 42 },
		{ 2030, 42 },
		{ 2018, 42 },
		{ 2030, 42 },
		{ 2021, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2020, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 1507, 1487 },
		{ 2013, 42 },
		{ 2015, 42 },
		{ 1509, 1489 },
		{ 2030, 42 },
		{ 1510, 1490 },
		{ 2030, 42 },
		{ 2028, 42 },
		{ 2016, 42 },
		{ 2030, 42 },
		{ 2029, 42 },
		{ 2022, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2027, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2014, 42 },
		{ 2030, 42 },
		{ 2026, 42 },
		{ 2030, 42 },
		{ 2023, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 2030, 42 },
		{ 1411, 23 },
		{ 1570, 1550 },
		{ 1571, 1550 },
		{ 1511, 1491 },
		{ 1398, 23 },
		{ 2414, 2386 },
		{ 2414, 2386 },
		{ 2427, 2399 },
		{ 2427, 2399 },
		{ 2430, 2402 },
		{ 2430, 2402 },
		{ 2453, 2425 },
		{ 2453, 2425 },
		{ 2454, 2426 },
		{ 2454, 2426 },
		{ 2497, 2469 },
		{ 2497, 2469 },
		{ 1512, 1492 },
		{ 1513, 1493 },
		{ 1514, 1494 },
		{ 1515, 1495 },
		{ 1516, 1496 },
		{ 1517, 1497 },
		{ 1411, 23 },
		{ 1570, 1550 },
		{ 1399, 23 },
		{ 1412, 23 },
		{ 1518, 1498 },
		{ 2414, 2386 },
		{ 1519, 1500 },
		{ 2427, 2399 },
		{ 1522, 1503 },
		{ 2430, 2402 },
		{ 1523, 1504 },
		{ 2453, 2425 },
		{ 1524, 1505 },
		{ 2454, 2426 },
		{ 1525, 1507 },
		{ 2497, 2469 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1526, 1509 },
		{ 1423, 1401 },
		{ 1530, 1511 },
		{ 1531, 1512 },
		{ 1572, 1550 },
		{ 1536, 1515 },
		{ 1537, 1516 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1415, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1404, 23 },
		{ 1402, 23 },
		{ 1417, 23 },
		{ 1403, 23 },
		{ 1417, 23 },
		{ 1406, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1405, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1538, 1517 },
		{ 1400, 23 },
		{ 1413, 23 },
		{ 1539, 1518 },
		{ 1407, 23 },
		{ 1540, 1519 },
		{ 1417, 23 },
		{ 1418, 23 },
		{ 1401, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1408, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1416, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1419, 23 },
		{ 1417, 23 },
		{ 1414, 23 },
		{ 1417, 23 },
		{ 1409, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1417, 23 },
		{ 1996, 40 },
		{ 1573, 1551 },
		{ 1574, 1551 },
		{ 1532, 1513 },
		{ 1821, 40 },
		{ 2502, 2474 },
		{ 2502, 2474 },
		{ 2509, 2481 },
		{ 2509, 2481 },
		{ 1534, 1514 },
		{ 1533, 1513 },
		{ 2522, 2495 },
		{ 2522, 2495 },
		{ 2523, 2496 },
		{ 2523, 2496 },
		{ 1543, 1523 },
		{ 1535, 1514 },
		{ 1544, 1524 },
		{ 1545, 1525 },
		{ 1546, 1526 },
		{ 1548, 1530 },
		{ 1549, 1531 },
		{ 1550, 1532 },
		{ 1996, 40 },
		{ 1573, 1551 },
		{ 1826, 40 },
		{ 2527, 2500 },
		{ 2527, 2500 },
		{ 2502, 2474 },
		{ 1551, 1533 },
		{ 2509, 2481 },
		{ 1552, 1534 },
		{ 1576, 1552 },
		{ 1577, 1552 },
		{ 2522, 2495 },
		{ 1553, 1535 },
		{ 2523, 2496 },
		{ 1554, 1536 },
		{ 1995, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 2527, 2500 },
		{ 1836, 40 },
		{ 1555, 1537 },
		{ 1556, 1538 },
		{ 1575, 1551 },
		{ 1557, 1539 },
		{ 1576, 1552 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1823, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1831, 40 },
		{ 1829, 40 },
		{ 1839, 40 },
		{ 1830, 40 },
		{ 1839, 40 },
		{ 1833, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1832, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1558, 1540 },
		{ 1827, 40 },
		{ 1578, 1552 },
		{ 1560, 1543 },
		{ 1839, 40 },
		{ 1561, 1544 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1828, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1824, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1825, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1838, 40 },
		{ 1839, 40 },
		{ 1837, 40 },
		{ 1839, 40 },
		{ 1834, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 1839, 40 },
		{ 2825, 46 },
		{ 2826, 46 },
		{ 1579, 1553 },
		{ 1580, 1553 },
		{ 69, 46 },
		{ 2530, 2503 },
		{ 2530, 2503 },
		{ 1562, 1545 },
		{ 1563, 1546 },
		{ 1426, 1402 },
		{ 1427, 1403 },
		{ 2534, 2507 },
		{ 2534, 2507 },
		{ 2535, 2508 },
		{ 2535, 2508 },
		{ 1431, 1405 },
		{ 1432, 1406 },
		{ 1433, 1407 },
		{ 1582, 1554 },
		{ 1583, 1555 },
		{ 1584, 1556 },
		{ 1586, 1560 },
		{ 1587, 1561 },
		{ 2825, 46 },
		{ 1588, 1562 },
		{ 1579, 1553 },
		{ 2315, 2286 },
		{ 2315, 2286 },
		{ 2530, 2503 },
		{ 1589, 1563 },
		{ 1597, 1583 },
		{ 1598, 1583 },
		{ 1606, 1596 },
		{ 1607, 1596 },
		{ 2534, 2507 },
		{ 1596, 1582 },
		{ 2535, 2508 },
		{ 2277, 46 },
		{ 2824, 46 },
		{ 2276, 46 },
		{ 2276, 46 },
		{ 2276, 46 },
		{ 2276, 46 },
		{ 2276, 46 },
		{ 2276, 46 },
		{ 2276, 46 },
		{ 2276, 46 },
		{ 2276, 46 },
		{ 2276, 46 },
		{ 2315, 2286 },
		{ 1434, 1408 },
		{ 1600, 1584 },
		{ 1602, 1586 },
		{ 1597, 1583 },
		{ 1581, 1553 },
		{ 1606, 1596 },
		{ 2278, 46 },
		{ 2274, 46 },
		{ 2269, 46 },
		{ 2278, 46 },
		{ 2266, 46 },
		{ 2273, 46 },
		{ 2271, 46 },
		{ 2278, 46 },
		{ 2275, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2268, 46 },
		{ 2263, 46 },
		{ 2270, 46 },
		{ 2265, 46 },
		{ 2278, 46 },
		{ 2272, 46 },
		{ 2267, 46 },
		{ 2264, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 1599, 1583 },
		{ 2282, 46 },
		{ 1608, 1596 },
		{ 1603, 1587 },
		{ 2278, 46 },
		{ 1604, 1588 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2279, 46 },
		{ 2280, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2281, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 2278, 46 },
		{ 161, 4 },
		{ 162, 4 },
		{ 2549, 2519 },
		{ 2549, 2519 },
		{ 2338, 2307 },
		{ 2338, 2307 },
		{ 2360, 2329 },
		{ 2360, 2329 },
		{ 2361, 2330 },
		{ 2361, 2330 },
		{ 2377, 2347 },
		{ 2377, 2347 },
		{ 1430, 1404 },
		{ 1605, 1589 },
		{ 1436, 1409 },
		{ 1611, 1602 },
		{ 1612, 1603 },
		{ 1435, 1409 },
		{ 1613, 1604 },
		{ 1614, 1605 },
		{ 1429, 1404 },
		{ 1439, 1414 },
		{ 1618, 1612 },
		{ 161, 4 },
		{ 1440, 1415 },
		{ 2549, 2519 },
		{ 1620, 1614 },
		{ 2338, 2307 },
		{ 1623, 1618 },
		{ 2360, 2329 },
		{ 1624, 1620 },
		{ 2361, 2330 },
		{ 1428, 1404 },
		{ 2377, 2347 },
		{ 1626, 1623 },
		{ 1627, 1624 },
		{ 1628, 1626 },
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
		{ 1629, 1627 },
		{ 1424, 1628 },
		{ 0, 2519 },
		{ 1441, 1416 },
		{ 1442, 1418 },
		{ 1443, 1419 },
		{ 1446, 1423 },
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
		{ 1447, 1426 },
		{ 83, 4 },
		{ 1448, 1427 },
		{ 1449, 1428 },
		{ 87, 4 },
		{ 1450, 1429 },
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
		{ 3079, 3078 },
		{ 1451, 1430 },
		{ 1452, 1431 },
		{ 3078, 3078 },
		{ 1453, 1432 },
		{ 1456, 1434 },
		{ 1454, 1433 },
		{ 1457, 1435 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 1455, 1433 },
		{ 3078, 3078 },
		{ 1458, 1436 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 1461, 1439 },
		{ 1462, 1440 },
		{ 1463, 1441 },
		{ 1464, 1442 },
		{ 1465, 1443 },
		{ 1468, 1446 },
		{ 1469, 1447 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 1470, 1448 },
		{ 1471, 1449 },
		{ 1472, 1450 },
		{ 1473, 1451 },
		{ 1474, 1452 },
		{ 1475, 1453 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 3078, 3078 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1476, 1454 },
		{ 1477, 1455 },
		{ 1478, 1456 },
		{ 1479, 1457 },
		{ 1480, 1458 },
		{ 1483, 1461 },
		{ 1484, 1462 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 156, 154 },
		{ 107, 92 },
		{ 108, 93 },
		{ 109, 94 },
		{ 1424, 1630 },
		{ 110, 95 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 1424, 1630 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 111, 96 },
		{ 112, 97 },
		{ 113, 98 },
		{ 114, 99 },
		{ 116, 101 },
		{ 122, 106 },
		{ 123, 107 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 124, 108 },
		{ 125, 109 },
		{ 126, 111 },
		{ 127, 112 },
		{ 2278, 2543 },
		{ 128, 113 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 2278, 2543 },
		{ 1425, 1629 },
		{ 0, 1629 },
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
		{ 144, 136 },
		{ 145, 137 },
		{ 2730, 2730 },
		{ 146, 138 },
		{ 2285, 2263 },
		{ 2287, 2264 },
		{ 2290, 2265 },
		{ 2291, 2266 },
		{ 2299, 2268 },
		{ 2288, 2265 },
		{ 2294, 2267 },
		{ 1425, 1629 },
		{ 2289, 2265 },
		{ 2301, 2269 },
		{ 2293, 2267 },
		{ 2302, 2270 },
		{ 2303, 2271 },
		{ 2292, 2266 },
		{ 2304, 2272 },
		{ 2305, 2273 },
		{ 2298, 2268 },
		{ 2306, 2274 },
		{ 2307, 2275 },
		{ 2278, 2278 },
		{ 2300, 2279 },
		{ 2730, 2730 },
		{ 2286, 2280 },
		{ 2297, 2281 },
		{ 2314, 2285 },
		{ 2295, 2267 },
		{ 2296, 2267 },
		{ 147, 139 },
		{ 2311, 2279 },
		{ 2316, 2287 },
		{ 2317, 2288 },
		{ 2318, 2289 },
		{ 2319, 2290 },
		{ 2320, 2291 },
		{ 2321, 2292 },
		{ 2322, 2293 },
		{ 0, 1629 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2323, 2294 },
		{ 2324, 2295 },
		{ 2325, 2296 },
		{ 2326, 2297 },
		{ 2327, 2298 },
		{ 2328, 2299 },
		{ 2330, 2300 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 69, 7 },
		{ 2331, 2301 },
		{ 2332, 2302 },
		{ 2333, 2303 },
		{ 2730, 2730 },
		{ 1630, 1629 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2336, 2305 },
		{ 2337, 2306 },
		{ 148, 140 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 2329, 2311 },
		{ 2345, 2314 },
		{ 2347, 2316 },
		{ 2348, 2317 },
		{ 2349, 2318 },
		{ 2350, 2319 },
		{ 2351, 2320 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 2352, 2321 },
		{ 2353, 2322 },
		{ 2354, 2323 },
		{ 2355, 2324 },
		{ 1238, 7 },
		{ 2356, 2325 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 1238, 7 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 2357, 2326 },
		{ 2358, 2327 },
		{ 2359, 2328 },
		{ 149, 142 },
		{ 2362, 2331 },
		{ 2363, 2332 },
		{ 2364, 2333 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 2365, 2334 },
		{ 2366, 2335 },
		{ 2367, 2336 },
		{ 2368, 2337 },
		{ 0, 1901 },
		{ 2375, 2345 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 1901 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 150, 143 },
		{ 2378, 2348 },
		{ 2379, 2349 },
		{ 151, 144 },
		{ 2383, 2352 },
		{ 2384, 2353 },
		{ 2386, 2355 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 2387, 2356 },
		{ 2381, 2351 },
		{ 152, 146 },
		{ 2389, 2358 },
		{ 0, 2095 },
		{ 2382, 2351 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 0, 2095 },
		{ 2334, 2304 },
		{ 2390, 2359 },
		{ 2394, 2362 },
		{ 2395, 2363 },
		{ 2396, 2364 },
		{ 2397, 2365 },
		{ 2398, 2366 },
		{ 2399, 2367 },
		{ 2400, 2368 },
		{ 2335, 2304 },
		{ 2402, 2375 },
		{ 2405, 2378 },
		{ 2410, 2382 },
		{ 2411, 2383 },
		{ 2412, 2384 },
		{ 153, 149 },
		{ 2415, 2387 },
		{ 2417, 2389 },
		{ 2418, 2390 },
		{ 2422, 2394 },
		{ 2423, 2395 },
		{ 2424, 2396 },
		{ 2425, 2397 },
		{ 2426, 2398 },
		{ 154, 150 },
		{ 2428, 2400 },
		{ 2434, 2405 },
		{ 2437, 2410 },
		{ 2438, 2411 },
		{ 2440, 2412 },
		{ 2443, 2415 },
		{ 2445, 2417 },
		{ 2446, 2418 },
		{ 2439, 2412 },
		{ 2450, 2422 },
		{ 2451, 2423 },
		{ 2452, 2424 },
		{ 155, 152 },
		{ 2456, 2428 },
		{ 2462, 2434 },
		{ 91, 75 },
		{ 2467, 2438 },
		{ 2468, 2439 },
		{ 2469, 2440 },
		{ 2472, 2443 },
		{ 2474, 2445 },
		{ 2475, 2446 },
		{ 2479, 2450 },
		{ 2480, 2451 },
		{ 2481, 2452 },
		{ 2486, 2456 },
		{ 2492, 2462 },
		{ 2495, 2467 },
		{ 2496, 2468 },
		{ 157, 155 },
		{ 2500, 2472 },
		{ 158, 157 },
		{ 2503, 2475 },
		{ 2507, 2479 },
		{ 2508, 2480 },
		{ 159, 158 },
		{ 2514, 2486 },
		{ 2519, 2492 },
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
		{ 94, 77 },
		{ 95, 78 },
		{ 96, 79 },
		{ 97, 80 },
		{ 98, 81 },
		{ 99, 82 },
		{ 101, 84 },
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
		{ 106, 91 },
		{ 2340, 2309 },
		{ 0, 2902 },
		{ 2340, 2309 },
		{ 87, 159 },
		{ 2622, 2596 },
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
		{ 1253, 1250 },
		{ 134, 120 },
		{ 1253, 1250 },
		{ 134, 120 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 118, 103 },
		{ 1248, 1245 },
		{ 118, 103 },
		{ 1248, 1245 },
		{ 2901, 51 },
		{ 2633, 2607 },
		{ 93, 76 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 2370, 2339 },
		{ 2372, 2342 },
		{ 2370, 2339 },
		{ 2372, 2342 },
		{ 1238, 1238 },
		{ 2784, 2769 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 0, 2514 },
		{ 0, 2514 },
		{ 132, 117 },
		{ 1251, 1247 },
		{ 132, 117 },
		{ 1251, 1247 },
		{ 2787, 2772 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 2201, 2198 },
		{ 1819, 1818 },
		{ 1303, 1302 },
		{ 1777, 1776 },
		{ 2768, 2752 },
		{ 2597, 2567 },
		{ 0, 2514 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 1729, 1728 },
		{ 88, 51 },
		{ 1774, 1773 },
		{ 1300, 1299 },
		{ 3068, 3068 },
		{ 3048, 3047 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 3068, 3068 },
		{ 1852, 1828 },
		{ 1684, 1683 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 3145, 3144 },
		{ 1851, 1828 },
		{ 2934, 2933 },
		{ 1705, 1704 },
		{ 2667, 2641 },
		{ 2987, 2986 },
		{ 2543, 2514 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 2038, 2016 },
		{ 1316, 1315 },
		{ 3028, 3027 },
		{ 2068, 2047 },
		{ 0, 1488 },
		{ 3051, 3050 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 0, 1488 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3059, 3058 },
		{ 2097, 2079 },
		{ 2498, 2470 },
		{ 2111, 2094 },
		{ 1877, 1858 },
		{ 1899, 1880 },
		{ 2848, 2847 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 1790, 1789 },
		{ 3117, 3112 },
		{ 3133, 3128 },
		{ 2888, 2887 },
		{ 2893, 2892 },
		{ 2598, 2568 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3128, 3128 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 2241, 2240 },
		{ 2246, 2245 },
		{ 2542, 2513 },
		{ 1732, 1731 },
		{ 3148, 3147 },
		{ 3164, 3161 },
		{ 3170, 3167 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 2690, 2665 },
		{ 2700, 2676 },
		{ 3138, 3135 },
		{ 1906, 1887 },
		{ 2706, 2682 },
		{ 2719, 2699 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3135, 3135 },
		{ 3134, 3129 },
		{ 3134, 3129 },
		{ 3134, 3129 },
		{ 3134, 3129 },
		{ 3134, 3129 },
		{ 3134, 3129 },
		{ 3134, 3129 },
		{ 3134, 3129 },
		{ 3134, 3129 },
		{ 3134, 3129 },
		{ 3127, 3123 },
		{ 3127, 3123 },
		{ 3127, 3123 },
		{ 3127, 3123 },
		{ 3127, 3123 },
		{ 3127, 3123 },
		{ 3127, 3123 },
		{ 3127, 3123 },
		{ 3127, 3123 },
		{ 3127, 3123 },
		{ 184, 175 },
		{ 194, 175 },
		{ 186, 175 },
		{ 1363, 1362 },
		{ 181, 175 },
		{ 2053, 2029 },
		{ 185, 175 },
		{ 2721, 2701 },
		{ 183, 175 },
		{ 1658, 1657 },
		{ 1363, 1362 },
		{ 2736, 2716 },
		{ 192, 175 },
		{ 191, 175 },
		{ 182, 175 },
		{ 190, 175 },
		{ 1658, 1657 },
		{ 187, 175 },
		{ 189, 175 },
		{ 180, 175 },
		{ 2737, 2717 },
		{ 1358, 1357 },
		{ 188, 175 },
		{ 193, 175 },
		{ 3087, 65 },
		{ 0, 3129 },
		{ 2052, 2029 },
		{ 69, 65 },
		{ 1933, 1917 },
		{ 2747, 2727 },
		{ 1935, 1920 },
		{ 2752, 2734 },
		{ 2762, 2745 },
		{ 1937, 1922 },
		{ 102, 85 },
		{ 0, 3123 },
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
		{ 3127, 3127 },
		{ 3127, 3127 },
		{ 3127, 3127 },
		{ 3127, 3127 },
		{ 3127, 3127 },
		{ 3127, 3127 },
		{ 3127, 3127 },
		{ 3127, 3127 },
		{ 3127, 3127 },
		{ 3127, 3127 },
		{ 1244, 1241 },
		{ 103, 85 },
		{ 1241, 1241 },
		{ 1241, 1241 },
		{ 1241, 1241 },
		{ 1241, 1241 },
		{ 1241, 1241 },
		{ 1241, 1241 },
		{ 1241, 1241 },
		{ 1241, 1241 },
		{ 1241, 1241 },
		{ 1241, 1241 },
		{ 2769, 2753 },
		{ 456, 411 },
		{ 461, 411 },
		{ 458, 411 },
		{ 457, 411 },
		{ 460, 411 },
		{ 455, 411 },
		{ 2772, 2756 },
		{ 454, 411 },
		{ 2518, 2491 },
		{ 1987, 1985 },
		{ 1245, 1241 },
		{ 459, 411 },
		{ 2790, 2775 },
		{ 462, 411 },
		{ 3086, 65 },
		{ 2804, 2798 },
		{ 2806, 2800 },
		{ 2812, 2807 },
		{ 3085, 65 },
		{ 453, 411 },
		{ 103, 85 },
		{ 2308, 2276 },
		{ 3132, 3127 },
		{ 2276, 2276 },
		{ 2276, 2276 },
		{ 2276, 2276 },
		{ 2276, 2276 },
		{ 2276, 2276 },
		{ 2276, 2276 },
		{ 2276, 2276 },
		{ 2276, 2276 },
		{ 2276, 2276 },
		{ 2276, 2276 },
		{ 2818, 2817 },
		{ 2520, 2493 },
		{ 1734, 1733 },
		{ 2851, 2850 },
		{ 2860, 2859 },
		{ 2873, 2872 },
		{ 1756, 1755 },
		{ 1769, 1768 },
		{ 1304, 1303 },
		{ 1245, 1241 },
		{ 2896, 2895 },
		{ 2309, 2276 },
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
		{ 1653, 1652 },
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
		{ 120, 104 },
		{ 2463, 2435 },
		{ 3083, 65 },
		{ 1778, 1777 },
		{ 3084, 65 },
		{ 2109, 2092 },
		{ 2110, 2093 },
		{ 2924, 2923 },
		{ 1381, 1380 },
		{ 2951, 2950 },
		{ 2309, 2276 },
		{ 117, 102 },
		{ 1244, 1244 },
		{ 1244, 1244 },
		{ 1244, 1244 },
		{ 1244, 1244 },
		{ 1244, 1244 },
		{ 1244, 1244 },
		{ 1244, 1244 },
		{ 1244, 1244 },
		{ 1244, 1244 },
		{ 1244, 1244 },
		{ 3134, 3134 },
		{ 3134, 3134 },
		{ 3134, 3134 },
		{ 3134, 3134 },
		{ 3134, 3134 },
		{ 3134, 3134 },
		{ 3134, 3134 },
		{ 3134, 3134 },
		{ 3134, 3134 },
		{ 3134, 3134 },
		{ 120, 104 },
		{ 1247, 1244 },
		{ 1793, 1792 },
		{ 2128, 2115 },
		{ 2981, 2980 },
		{ 2141, 2126 },
		{ 2990, 2989 },
		{ 2998, 2997 },
		{ 2476, 2447 },
		{ 2477, 2448 },
		{ 3022, 3021 },
		{ 117, 102 },
		{ 2142, 2127 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 1247, 1244 },
		{ 1250, 1246 },
		{ 3137, 3134 },
		{ 3031, 3030 },
		{ 3102, 67 },
		{ 1240, 9 },
		{ 3042, 3041 },
		{ 69, 67 },
		{ 1392, 1391 },
		{ 69, 9 },
		{ 2202, 2199 },
		{ 2339, 2308 },
		{ 3053, 3052 },
		{ 2310, 2310 },
		{ 2310, 2310 },
		{ 2310, 2310 },
		{ 2310, 2310 },
		{ 2310, 2310 },
		{ 2310, 2310 },
		{ 2310, 2310 },
		{ 2310, 2310 },
		{ 2310, 2310 },
		{ 2310, 2310 },
		{ 2217, 2216 },
		{ 3062, 3061 },
		{ 1270, 1269 },
		{ 2490, 2460 },
		{ 2585, 2555 },
		{ 1240, 9 },
		{ 1706, 1705 },
		{ 2494, 2464 },
		{ 2243, 2242 },
		{ 2605, 2577 },
		{ 1250, 1246 },
		{ 2342, 2310 },
		{ 3112, 3107 },
		{ 2620, 2594 },
		{ 1708, 1707 },
		{ 2626, 2600 },
		{ 2436, 2407 },
		{ 2636, 2610 },
		{ 1880, 1861 },
		{ 1242, 9 },
		{ 2339, 2308 },
		{ 1241, 9 },
		{ 1241, 9 },
		{ 1241, 9 },
		{ 1241, 9 },
		{ 1241, 9 },
		{ 1241, 9 },
		{ 1241, 9 },
		{ 1241, 9 },
		{ 1241, 9 },
		{ 1241, 9 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 3095, 3095 },
		{ 2361, 2361 },
		{ 2361, 2361 },
		{ 2342, 2310 },
		{ 3099, 67 },
		{ 2504, 2504 },
		{ 2504, 2504 },
		{ 2527, 2527 },
		{ 2527, 2527 },
		{ 2255, 2254 },
		{ 3100, 67 },
		{ 2453, 2453 },
		{ 2453, 2453 },
		{ 1886, 1867 },
		{ 2952, 2952 },
		{ 2952, 2952 },
		{ 2999, 2999 },
		{ 2999, 2999 },
		{ 2505, 2505 },
		{ 2505, 2505 },
		{ 2720, 2720 },
		{ 2720, 2720 },
		{ 2675, 2649 },
		{ 3097, 67 },
		{ 2361, 2361 },
		{ 2515, 2515 },
		{ 2515, 2515 },
		{ 3150, 3149 },
		{ 2504, 2504 },
		{ 1319, 1318 },
		{ 2527, 2527 },
		{ 2380, 2380 },
		{ 2380, 2380 },
		{ 2686, 2661 },
		{ 2453, 2453 },
		{ 3177, 3175 },
		{ 3096, 3095 },
		{ 2952, 2952 },
		{ 1760, 1759 },
		{ 2999, 2999 },
		{ 2069, 2048 },
		{ 2505, 2505 },
		{ 2822, 2821 },
		{ 2720, 2720 },
		{ 2088, 2067 },
		{ 3101, 67 },
		{ 2535, 2535 },
		{ 2535, 2535 },
		{ 2515, 2515 },
		{ 2406, 2406 },
		{ 2406, 2406 },
		{ 2549, 2549 },
		{ 2549, 2549 },
		{ 2843, 2842 },
		{ 2380, 2380 },
		{ 2569, 2569 },
		{ 2569, 2569 },
		{ 2623, 2623 },
		{ 2623, 2623 },
		{ 2427, 2427 },
		{ 2427, 2427 },
		{ 2884, 2884 },
		{ 2884, 2884 },
		{ 2090, 2069 },
		{ 2093, 2072 },
		{ 2912, 2912 },
		{ 2912, 2912 },
		{ 2530, 2530 },
		{ 2530, 2530 },
		{ 2535, 2535 },
		{ 2534, 2534 },
		{ 2534, 2534 },
		{ 2406, 2406 },
		{ 2855, 2854 },
		{ 2549, 2549 },
		{ 2385, 2385 },
		{ 2385, 2385 },
		{ 2567, 2536 },
		{ 2569, 2569 },
		{ 2571, 2540 },
		{ 2623, 2623 },
		{ 1335, 1334 },
		{ 2427, 2427 },
		{ 1859, 1834 },
		{ 2884, 2884 },
		{ 2579, 2547 },
		{ 1858, 1834 },
		{ 1337, 1336 },
		{ 2912, 2912 },
		{ 2048, 2023 },
		{ 2530, 2530 },
		{ 1616, 1610 },
		{ 2047, 2023 },
		{ 2534, 2534 },
		{ 2509, 2509 },
		{ 2509, 2509 },
		{ 1646, 1645 },
		{ 2392, 2361 },
		{ 2385, 2385 },
		{ 2473, 2473 },
		{ 2473, 2473 },
		{ 2248, 2248 },
		{ 2248, 2248 },
		{ 1786, 1785 },
		{ 1647, 1646 },
		{ 2502, 2502 },
		{ 2502, 2502 },
		{ 2907, 2905 },
		{ 2393, 2361 },
		{ 2558, 2527 },
		{ 2557, 2527 },
		{ 1351, 1350 },
		{ 2531, 2504 },
		{ 2483, 2453 },
		{ 2482, 2453 },
		{ 2430, 2430 },
		{ 2430, 2430 },
		{ 2509, 2509 },
		{ 3152, 3152 },
		{ 2544, 2515 },
		{ 2606, 2579 },
		{ 2953, 2952 },
		{ 2473, 2473 },
		{ 3000, 2999 },
		{ 2248, 2248 },
		{ 2532, 2505 },
		{ 2740, 2720 },
		{ 2545, 2515 },
		{ 2502, 2502 },
		{ 1809, 1808 },
		{ 2723, 2723 },
		{ 2723, 2723 },
		{ 2928, 2927 },
		{ 2913, 2912 },
		{ 1810, 1809 },
		{ 2388, 2388 },
		{ 2388, 2388 },
		{ 2407, 2380 },
		{ 2430, 2430 },
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 3152, 3152 },
		{ 1815, 1814 },
		{ 1352, 1351 },
		{ 1674, 1673 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2640, 2614 },
		{ 2966, 2966 },
		{ 2966, 2966 },
		{ 1675, 1674 },
		{ 2652, 2626 },
		{ 2566, 2535 },
		{ 2723, 2723 },
		{ 1681, 1680 },
		{ 2435, 2406 },
		{ 1682, 1681 },
		{ 2580, 2549 },
		{ 2388, 2388 },
		{ 2523, 2523 },
		{ 2523, 2523 },
		{ 2599, 2569 },
		{ 2921, 2921 },
		{ 2649, 2623 },
		{ 2455, 2427 },
		{ 2913, 2912 },
		{ 2885, 2884 },
		{ 2538, 2509 },
		{ 2315, 2315 },
		{ 1878, 1859 },
		{ 2561, 2530 },
		{ 2966, 2966 },
		{ 1280, 1279 },
		{ 2565, 2534 },
		{ 3007, 3007 },
		{ 3007, 3007 },
		{ 2629, 2629 },
		{ 2629, 2629 },
		{ 2413, 2385 },
		{ 2258, 2257 },
		{ 2678, 2678 },
		{ 2678, 2678 },
		{ 2523, 2523 },
		{ 2512, 2484 },
		{ 2536, 2509 },
		{ 3041, 3040 },
		{ 2654, 2654 },
		{ 2654, 2654 },
		{ 2537, 2509 },
		{ 2377, 2377 },
		{ 2377, 2377 },
		{ 1885, 1866 },
		{ 2516, 2489 },
		{ 1700, 1699 },
		{ 2707, 2683 },
		{ 2708, 2684 },
		{ 2710, 2687 },
		{ 3007, 3007 },
		{ 2711, 2690 },
		{ 2629, 2629 },
		{ 1897, 1878 },
		{ 1701, 1700 },
		{ 2501, 2473 },
		{ 2678, 2678 },
		{ 2249, 2248 },
		{ 2521, 2494 },
		{ 3155, 3152 },
		{ 1312, 1311 },
		{ 2529, 2502 },
		{ 2654, 2654 },
		{ 2739, 2719 },
		{ 3154, 3152 },
		{ 2377, 2377 },
		{ 3153, 3152 },
		{ 1259, 1258 },
		{ 1387, 1386 },
		{ 1724, 1723 },
		{ 2458, 2430 },
		{ 2748, 2728 },
		{ 1725, 1724 },
		{ 1948, 1935 },
		{ 1967, 1957 },
		{ 1975, 1967 },
		{ 1274, 1273 },
		{ 3140, 3139 },
		{ 3141, 3140 },
		{ 1610, 1601 },
		{ 2779, 2763 },
		{ 1327, 1326 },
		{ 1750, 1749 },
		{ 1751, 1750 },
		{ 1334, 1333 },
		{ 2743, 2723 },
		{ 2550, 2520 },
		{ 2471, 2442 },
		{ 2528, 2501 },
		{ 2871, 2870 },
		{ 2416, 2388 },
		{ 2651, 2625 },
		{ 2882, 2881 },
		{ 2421, 2393 },
		{ 2922, 2921 },
		{ 2198, 2193 },
		{ 2656, 2630 },
		{ 1679, 1678 },
		{ 1389, 1388 },
		{ 2214, 2211 },
		{ 2346, 2315 },
		{ 1736, 1735 },
		{ 2687, 2662 },
		{ 2967, 2966 },
		{ 2911, 2909 },
		{ 1982, 1977 },
		{ 2691, 2666 },
		{ 2539, 2510 },
		{ 1302, 1301 },
		{ 1261, 1260 },
		{ 2403, 2403 },
		{ 2403, 2403 },
		{ 2245, 2244 },
		{ 1686, 1685 },
		{ 2553, 2523 },
		{ 2979, 2978 },
		{ 1870, 1851 },
		{ 2251, 2250 },
		{ 2560, 2529 },
		{ 2725, 2705 },
		{ 2729, 2709 },
		{ 1329, 1328 },
		{ 3020, 3019 },
		{ 2562, 2531 },
		{ 2563, 2532 },
		{ 2257, 2256 },
		{ 1762, 1761 },
		{ 2059, 2038 },
		{ 1282, 1281 },
		{ 3008, 3007 },
		{ 1881, 1862 },
		{ 2655, 2629 },
		{ 2751, 2733 },
		{ 2403, 2403 },
		{ 2080, 2059 },
		{ 2702, 2678 },
		{ 2759, 2742 },
		{ 1771, 1770 },
		{ 1365, 1364 },
		{ 2680, 2654 },
		{ 2591, 2561 },
		{ 2404, 2377 },
		{ 2595, 2565 },
		{ 2774, 2758 },
		{ 1889, 1870 },
		{ 1776, 1775 },
		{ 1379, 1378 },
		{ 1296, 1295 },
		{ 3121, 3117 },
		{ 2791, 2776 },
		{ 2792, 2778 },
		{ 1916, 1899 },
		{ 2805, 2799 },
		{ 2613, 2587 },
		{ 1710, 1709 },
		{ 2813, 2811 },
		{ 2816, 2814 },
		{ 1660, 1659 },
		{ 1383, 1382 },
		{ 2627, 2601 },
		{ 3152, 3151 },
		{ 1268, 1267 },
		{ 3160, 3157 },
		{ 2845, 2844 },
		{ 3168, 3165 },
		{ 2635, 2609 },
		{ 2195, 2188 },
		{ 3180, 3179 },
		{ 2360, 2360 },
		{ 2360, 2360 },
		{ 2802, 2802 },
		{ 2802, 2802 },
		{ 2431, 2403 },
		{ 2454, 2454 },
		{ 2454, 2454 },
		{ 2470, 2441 },
		{ 2583, 2553 },
		{ 1920, 1905 },
		{ 1377, 1376 },
		{ 2666, 2640 },
		{ 2624, 2598 },
		{ 2760, 2743 },
		{ 2716, 2695 },
		{ 2717, 2696 },
		{ 2923, 2922 },
		{ 2594, 2564 },
		{ 2079, 2058 },
		{ 2722, 2702 },
		{ 2681, 2655 },
		{ 1755, 1754 },
		{ 1816, 1815 },
		{ 2360, 2360 },
		{ 2577, 2545 },
		{ 2802, 2802 },
		{ 2601, 2571 },
		{ 1391, 1390 },
		{ 2454, 2454 },
		{ 2798, 2789 },
		{ 2701, 2677 },
		{ 1848, 1825 },
		{ 1882, 1863 },
		{ 2898, 2897 },
		{ 2582, 2552 },
		{ 2432, 2403 },
		{ 1884, 1865 },
		{ 2073, 2052 },
		{ 2590, 2560 },
		{ 2441, 2413 },
		{ 2075, 2054 },
		{ 1847, 1825 },
		{ 2077, 2056 },
		{ 1376, 1375 },
		{ 1718, 1717 },
		{ 2926, 2925 },
		{ 1272, 1271 },
		{ 2933, 2932 },
		{ 1466, 1444 },
		{ 2746, 2726 },
		{ 1680, 1679 },
		{ 1295, 1294 },
		{ 2108, 2091 },
		{ 2617, 2591 },
		{ 2983, 2982 },
		{ 1915, 1898 },
		{ 2621, 2595 },
		{ 2992, 2991 },
		{ 1795, 1794 },
		{ 1803, 1802 },
		{ 1345, 1344 },
		{ 2123, 2107 },
		{ 1930, 1914 },
		{ 3024, 3023 },
		{ 1735, 1734 },
		{ 2778, 2762 },
		{ 3033, 3032 },
		{ 1683, 1682 },
		{ 2780, 2764 },
		{ 2637, 2611 },
		{ 2173, 2155 },
		{ 2174, 2156 },
		{ 3055, 3054 },
		{ 1744, 1743 },
		{ 1640, 1639 },
		{ 3064, 3063 },
		{ 2200, 2197 },
		{ 2799, 2790 },
		{ 1394, 1393 },
		{ 2662, 2636 },
		{ 1754, 1753 },
		{ 2208, 2205 },
		{ 2811, 2806 },
		{ 2211, 2209 },
		{ 2676, 2650 },
		{ 3118, 3113 },
		{ 1694, 1693 },
		{ 1984, 1981 },
		{ 2820, 2819 },
		{ 2559, 2528 },
		{ 1382, 1381 },
		{ 1990, 1989 },
		{ 1863, 1840 },
		{ 1501, 1481 },
		{ 1321, 1320 },
		{ 2853, 2852 },
		{ 2391, 2360 },
		{ 1444, 1420 },
		{ 2807, 2802 },
		{ 2704, 2680 },
		{ 3151, 3150 },
		{ 2484, 2454 },
		{ 1668, 1667 },
		{ 2054, 2031 },
		{ 3157, 3154 },
		{ 2875, 2874 },
		{ 2881, 2880 },
		{ 2058, 2037 },
		{ 2709, 2686 },
		{ 2499, 2471 },
		{ 3179, 3177 },
		{ 1709, 1708 },
		{ 2866, 2866 },
		{ 2866, 2866 },
		{ 3015, 3015 },
		{ 3015, 3015 },
		{ 2522, 2522 },
		{ 2522, 2522 },
		{ 2497, 2497 },
		{ 2497, 2497 },
		{ 2974, 2974 },
		{ 2974, 2974 },
		{ 1796, 1795 },
		{ 1802, 1801 },
		{ 2247, 2246 },
		{ 2927, 2926 },
		{ 1651, 1650 },
		{ 2070, 2049 },
		{ 1374, 1373 },
		{ 2935, 2934 },
		{ 2944, 2943 },
		{ 1521, 1502 },
		{ 2763, 2746 },
		{ 2961, 2960 },
		{ 2962, 2961 },
		{ 2866, 2866 },
		{ 2964, 2963 },
		{ 3015, 3015 },
		{ 115, 100 },
		{ 2522, 2522 },
		{ 1917, 1900 },
		{ 2497, 2497 },
		{ 2977, 2976 },
		{ 2974, 2974 },
		{ 2343, 2312 },
		{ 1667, 1666 },
		{ 1317, 1316 },
		{ 2984, 2983 },
		{ 1922, 1907 },
		{ 2988, 2987 },
		{ 2092, 2071 },
		{ 1307, 1306 },
		{ 2993, 2992 },
		{ 1932, 1916 },
		{ 1445, 1422 },
		{ 3005, 3004 },
		{ 1717, 1716 },
		{ 1395, 1394 },
		{ 3018, 3017 },
		{ 2794, 2780 },
		{ 1356, 1355 },
		{ 2663, 2637 },
		{ 3025, 3024 },
		{ 2665, 2639 },
		{ 3029, 3028 },
		{ 1956, 1944 },
		{ 2115, 2098 },
		{ 3034, 3033 },
		{ 3040, 3039 },
		{ 1865, 1843 },
		{ 1969, 1960 },
		{ 2679, 2653 },
		{ 1869, 1850 },
		{ 1981, 1976 },
		{ 2153, 2139 },
		{ 3056, 3055 },
		{ 1481, 1459 },
		{ 3060, 3059 },
		{ 2821, 2820 },
		{ 2419, 2391 },
		{ 3065, 3064 },
		{ 2420, 2392 },
		{ 1639, 1638 },
		{ 2175, 2157 },
		{ 3080, 3076 },
		{ 2568, 2537 },
		{ 2849, 2848 },
		{ 2186, 2173 },
		{ 2187, 2174 },
		{ 3109, 3104 },
		{ 2854, 2853 },
		{ 3113, 3108 },
		{ 1273, 1272 },
		{ 2197, 2192 },
		{ 1879, 1860 },
		{ 3125, 3121 },
		{ 2869, 2868 },
		{ 1991, 1990 },
		{ 1781, 1780 },
		{ 2584, 2554 },
		{ 2876, 2875 },
		{ 1322, 1321 },
		{ 1344, 1343 },
		{ 2210, 2208 },
		{ 1791, 1790 },
		{ 1693, 1692 },
		{ 2867, 2866 },
		{ 2894, 2893 },
		{ 3016, 3015 },
		{ 1743, 1742 },
		{ 2552, 2522 },
		{ 2732, 2712 },
		{ 2524, 2497 },
		{ 2899, 2898 },
		{ 2975, 2974 },
		{ 2056, 2034 },
		{ 2057, 2036 },
		{ 2600, 2570 },
		{ 2517, 2490 },
		{ 2908, 2906 },
		{ 2602, 2572 },
		{ 2604, 2576 },
		{ 1887, 1868 },
		{ 2985, 2985 },
		{ 2985, 2985 },
		{ 3057, 3057 },
		{ 3057, 3057 },
		{ 2414, 2414 },
		{ 2414, 2414 },
		{ 2846, 2846 },
		{ 2846, 2846 },
		{ 2556, 2556 },
		{ 2556, 2556 },
		{ 2338, 2338 },
		{ 2338, 2338 },
		{ 2757, 2757 },
		{ 2757, 2757 },
		{ 2891, 2891 },
		{ 2891, 2891 },
		{ 3026, 3026 },
		{ 3026, 3026 },
		{ 1314, 1314 },
		{ 1314, 1314 },
		{ 1788, 1788 },
		{ 1788, 1788 },
		{ 1641, 1640 },
		{ 2985, 2985 },
		{ 1704, 1703 },
		{ 3057, 3057 },
		{ 1903, 1884 },
		{ 2414, 2414 },
		{ 1346, 1345 },
		{ 2846, 2846 },
		{ 1719, 1718 },
		{ 2556, 2556 },
		{ 1758, 1757 },
		{ 2338, 2338 },
		{ 2657, 2631 },
		{ 2757, 2757 },
		{ 2253, 2252 },
		{ 2891, 2891 },
		{ 1804, 1803 },
		{ 3026, 3026 },
		{ 2124, 2108 },
		{ 1314, 1314 },
		{ 2461, 2433 },
		{ 1788, 1788 },
		{ 2203, 2200 },
		{ 1520, 1501 },
		{ 1695, 1694 },
		{ 1745, 1744 },
		{ 1669, 1668 },
		{ 1931, 1915 },
		{ 3167, 3164 },
		{ 2219, 2218 },
		{ 2526, 2499 },
		{ 2096, 2077 },
		{ 3122, 3118 },
		{ 1986, 1984 },
		{ 3003, 3003 },
		{ 3003, 3003 },
		{ 1309, 1309 },
		{ 1309, 1309 },
		{ 1783, 1783 },
		{ 1783, 1783 },
		{ 3046, 3046 },
		{ 3046, 3046 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 1772, 1772 },
		{ 1772, 1772 },
		{ 2969, 2969 },
		{ 2969, 2969 },
		{ 1672, 1671 },
		{ 2861, 2861 },
		{ 2861, 2861 },
		{ 1298, 1298 },
		{ 1298, 1298 },
		{ 1311, 1310 },
		{ 2429, 2401 },
		{ 2094, 2073 },
		{ 2095, 2075 },
		{ 3003, 3003 },
		{ 1488, 1466 },
		{ 1309, 1309 },
		{ 2207, 2204 },
		{ 1783, 1783 },
		{ 1807, 1806 },
		{ 3046, 3046 },
		{ 2634, 2608 },
		{ 3010, 3010 },
		{ 1748, 1747 },
		{ 1772, 1772 },
		{ 2714, 2693 },
		{ 2969, 2969 },
		{ 1698, 1697 },
		{ 2986, 2985 },
		{ 2861, 2861 },
		{ 3058, 3057 },
		{ 1298, 1298 },
		{ 2442, 2414 },
		{ 2797, 2788 },
		{ 2847, 2846 },
		{ 2525, 2498 },
		{ 2586, 2556 },
		{ 2638, 2612 },
		{ 2369, 2338 },
		{ 1722, 1721 },
		{ 2773, 2757 },
		{ 1266, 1265 },
		{ 2892, 2891 },
		{ 2650, 2624 },
		{ 3027, 3026 },
		{ 1817, 1816 },
		{ 1315, 1314 },
		{ 2114, 2097 },
		{ 1789, 1788 },
		{ 2735, 2715 },
		{ 1785, 1784 },
		{ 2919, 2918 },
		{ 3136, 3131 },
		{ 2242, 2241 },
		{ 2738, 2718 },
		{ 1947, 1934 },
		{ 1656, 1655 },
		{ 2067, 2046 },
		{ 3147, 3146 },
		{ 1332, 1331 },
		{ 1959, 1949 },
		{ 2541, 2512 },
		{ 1890, 1871 },
		{ 2172, 2154 },
		{ 1731, 1730 },
		{ 3050, 3049 },
		{ 1349, 1348 },
		{ 1901, 1882 },
		{ 2761, 2744 },
		{ 1585, 1559 },
		{ 3173, 3170 },
		{ 1644, 1643 },
		{ 1866, 1846 },
		{ 1361, 1360 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 2506, 2506 },
		{ 2506, 2506 },
		{ 2644, 2644 },
		{ 2644, 2644 },
		{ 2645, 2645 },
		{ 2645, 2645 },
		{ 2915, 2914 },
		{ 3004, 3003 },
		{ 2942, 2941 },
		{ 1310, 1309 },
		{ 0, 1242 },
		{ 1784, 1783 },
		{ 2031, 2222 },
		{ 3047, 3046 },
		{ 1420, 1411 },
		{ 3011, 3010 },
		{ 1840, 1996 },
		{ 1773, 1772 },
		{ 2653, 2627 },
		{ 2970, 2969 },
		{ 2874, 2873 },
		{ 1262, 1262 },
		{ 2862, 2861 },
		{ 2506, 2506 },
		{ 1299, 1298 },
		{ 2644, 2644 },
		{ 0, 86 },
		{ 2645, 2645 },
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
		{ 1330, 1330 },
		{ 1330, 1330 },
		{ 0, 2277 },
		{ 1794, 1793 },
		{ 0, 1242 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2632, 2632 },
		{ 2632, 2632 },
		{ 3023, 3022 },
		{ 2216, 2214 },
		{ 1921, 1906 },
		{ 0, 86 },
		{ 2035, 2013 },
		{ 1759, 1758 },
		{ 1330, 1330 },
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
		{ 0, 2277 },
		{ 2753, 2735 },
		{ 2756, 2738 },
		{ 3032, 3031 },
		{ 2632, 2632 },
		{ 3071, 3071 },
		{ 3071, 3071 },
		{ 3071, 3071 },
		{ 3071, 3071 },
		{ 3071, 3071 },
		{ 3071, 3071 },
		{ 3071, 3071 },
		{ 3071, 3071 },
		{ 3071, 3071 },
		{ 3071, 3071 },
		{ 1871, 1852 },
		{ 1320, 1319 },
		{ 1263, 1262 },
		{ 1362, 1361 },
		{ 2533, 2506 },
		{ 1355, 1354 },
		{ 2670, 2644 },
		{ 2897, 2896 },
		{ 2671, 2645 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 1249, 1249 },
		{ 1249, 1249 },
		{ 1249, 1249 },
		{ 1249, 1249 },
		{ 1249, 1249 },
		{ 1249, 1249 },
		{ 1249, 1249 },
		{ 1249, 1249 },
		{ 1249, 1249 },
		{ 1249, 1249 },
		{ 2036, 2013 },
		{ 2125, 2109 },
		{ 1331, 1330 },
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
		{ 2764, 2747 },
		{ 1421, 1400 },
		{ 2937, 2937 },
		{ 2937, 2937 },
		{ 2658, 2632 },
		{ 1252, 1252 },
		{ 1252, 1252 },
		{ 1252, 1252 },
		{ 1252, 1252 },
		{ 1252, 1252 },
		{ 1252, 1252 },
		{ 1252, 1252 },
		{ 1252, 1252 },
		{ 1252, 1252 },
		{ 1252, 1252 },
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
		{ 2937, 2937 },
		{ 3172, 3172 },
		{ 2956, 2956 },
		{ 2956, 2956 },
		{ 2615, 2615 },
		{ 2615, 2615 },
		{ 2877, 2877 },
		{ 2877, 2877 },
		{ 2596, 2566 },
		{ 2139, 2123 },
		{ 3054, 3053 },
		{ 2677, 2651 },
		{ 1271, 1270 },
		{ 1944, 1930 },
		{ 1373, 1372 },
		{ 2683, 2657 },
		{ 2918, 2917 },
		{ 3063, 3062 },
		{ 2433, 2404 },
		{ 2155, 2141 },
		{ 2788, 2773 },
		{ 2789, 2774 },
		{ 2925, 2924 },
		{ 2156, 2142 },
		{ 3172, 3172 },
		{ 2956, 2956 },
		{ 2401, 2369 },
		{ 2615, 2615 },
		{ 2693, 2668 },
		{ 2877, 2877 },
		{ 1357, 1356 },
		{ 2607, 2580 },
		{ 2610, 2584 },
		{ 2611, 2585 },
		{ 2612, 2586 },
		{ 2943, 2942 },
		{ 2491, 2461 },
		{ 1814, 1813 },
		{ 1390, 1389 },
		{ 1650, 1649 },
		{ 2960, 2959 },
		{ 1780, 1779 },
		{ 2712, 2691 },
		{ 1422, 1400 },
		{ 2963, 2962 },
		{ 1306, 1305 },
		{ 2715, 2694 },
		{ 2819, 2818 },
		{ 1652, 1651 },
		{ 1502, 1482 },
		{ 2718, 2698 },
		{ 2631, 2605 },
		{ 2982, 2981 },
		{ 2447, 2419 },
		{ 2448, 2420 },
		{ 1849, 1827 },
		{ 2726, 2706 },
		{ 1654, 1653 },
		{ 1359, 1358 },
		{ 2991, 2990 },
		{ 3160, 3160 },
		{ 2852, 2851 },
		{ 2734, 2714 },
		{ 1989, 1987 },
		{ 1657, 1656 },
		{ 2205, 2202 },
		{ 1757, 1756 },
		{ 2938, 2937 },
		{ 1393, 1392 },
		{ 2460, 2432 },
		{ 1867, 1847 },
		{ 3014, 3013 },
		{ 2755, 2737 },
		{ 2973, 2972 },
		{ 2218, 2217 },
		{ 2954, 2953 },
		{ 69, 5 },
		{ 1779, 1778 },
		{ 2865, 2864 },
		{ 3172, 3169 },
		{ 3001, 3000 },
		{ 2939, 2938 },
		{ 2646, 2620 },
		{ 3160, 3160 },
		{ 1305, 1304 },
		{ 2754, 2736 },
		{ 2959, 2958 },
		{ 1372, 1371 },
		{ 1782, 1781 },
		{ 2703, 2679 },
		{ 2945, 2944 },
		{ 2957, 2956 },
		{ 2209, 2207 },
		{ 2641, 2615 },
		{ 3174, 3172 },
		{ 2878, 2877 },
		{ 2965, 2964 },
		{ 1308, 1307 },
		{ 2750, 2732 },
		{ 2589, 2559 },
		{ 2905, 2903 },
		{ 3103, 3097 },
		{ 1860, 1837 },
		{ 2692, 2667 },
		{ 2614, 2588 },
		{ 3009, 3008 },
		{ 1861, 1837 },
		{ 1482, 1460 },
		{ 2376, 2346 },
		{ 2880, 2879 },
		{ 2033, 2010 },
		{ 2511, 2483 },
		{ 2917, 2916 },
		{ 2540, 2511 },
		{ 1294, 1293 },
		{ 2032, 2010 },
		{ 2906, 2903 },
		{ 2444, 2416 },
		{ 2588, 2558 },
		{ 1768, 1767 },
		{ 2224, 2221 },
		{ 1998, 1995 },
		{ 1850, 1827 },
		{ 2488, 2458 },
		{ 2713, 2692 },
		{ 2223, 2221 },
		{ 1997, 1995 },
		{ 2239, 2238 },
		{ 176, 5 },
		{ 1842, 1822 },
		{ 2968, 2967 },
		{ 3104, 3097 },
		{ 177, 5 },
		{ 1438, 1412 },
		{ 1841, 1822 },
		{ 1787, 1786 },
		{ 1293, 1292 },
		{ 2457, 2429 },
		{ 1301, 1300 },
		{ 178, 5 },
		{ 2547, 2517 },
		{ 2188, 2175 },
		{ 2659, 2633 },
		{ 2192, 2185 },
		{ 2976, 2975 },
		{ 1742, 1741 },
		{ 2978, 2977 },
		{ 2664, 2638 },
		{ 1685, 1684 },
		{ 2803, 2797 },
		{ 2554, 2524 },
		{ 2034, 2011 },
		{ 3163, 3160 },
		{ 100, 83 },
		{ 1692, 1691 },
		{ 175, 5 },
		{ 1746, 1745 },
		{ 1896, 1877 },
		{ 1801, 1800 },
		{ 2814, 2812 },
		{ 2564, 2533 },
		{ 2206, 2203 },
		{ 1459, 1437 },
		{ 1900, 1881 },
		{ 3002, 3001 },
		{ 1460, 1438 },
		{ 2570, 2539 },
		{ 1378, 1377 },
		{ 2478, 2449 },
		{ 2844, 2843 },
		{ 2576, 2544 },
		{ 1904, 1885 },
		{ 3017, 3016 },
		{ 2695, 2670 },
		{ 3019, 3018 },
		{ 2696, 2671 },
		{ 1805, 1804 },
		{ 1907, 1889 },
		{ 1696, 1695 },
		{ 2071, 2050 },
		{ 2485, 2455 },
		{ 2705, 2681 },
		{ 2072, 2051 },
		{ 1336, 1335 },
		{ 1260, 1259 },
		{ 2868, 2867 },
		{ 1343, 1342 },
		{ 2870, 2869 },
		{ 2238, 2237 },
		{ 1919, 1903 },
		{ 3043, 3042 },
		{ 2592, 2562 },
		{ 2593, 2563 },
		{ 1281, 1280 },
		{ 1386, 1385 },
		{ 2879, 2878 },
		{ 2089, 2068 },
		{ 1659, 1658 },
		{ 1761, 1760 },
		{ 2883, 2882 },
		{ 1541, 1520 },
		{ 2886, 2885 },
		{ 1666, 1665 },
		{ 2890, 2889 },
		{ 1843, 1823 },
		{ 1844, 1824 },
		{ 1845, 1824 },
		{ 2603, 2573 },
		{ 3076, 3067 },
		{ 2728, 2708 },
		{ 2254, 2253 },
		{ 1542, 1521 },
		{ 2733, 2713 },
		{ 2256, 2255 },
		{ 2098, 2080 },
		{ 2609, 2583 },
		{ 2106, 2088 },
		{ 1770, 1769 },
		{ 3107, 3101 },
		{ 3108, 3103 },
		{ 1716, 1715 },
		{ 2909, 2907 },
		{ 1313, 1312 },
		{ 2742, 2722 },
		{ 1328, 1327 },
		{ 2916, 2915 },
		{ 2744, 2724 },
		{ 1775, 1774 },
		{ 3126, 3122 },
		{ 2618, 2592 },
		{ 2920, 2919 },
		{ 2619, 2593 },
		{ 2113, 2096 },
		{ 1960, 1950 },
		{ 1670, 1669 },
		{ 1720, 1719 },
		{ 1364, 1363 },
		{ 1976, 1968 },
		{ 3144, 3143 },
		{ 2630, 2604 },
		{ 1977, 1969 },
		{ 1868, 1848 },
		{ 1347, 1346 },
		{ 1638, 1637 },
		{ 2940, 2939 },
		{ 1728, 1727 },
		{ 3156, 3153 },
		{ 1371, 1370 },
		{ 1988, 1986 },
		{ 2639, 2613 },
		{ 2449, 2421 },
		{ 3165, 3162 },
		{ 2157, 2143 },
		{ 1846, 1824 },
		{ 1297, 1296 },
		{ 2955, 2954 },
		{ 2648, 2622 },
		{ 2958, 2957 },
		{ 2776, 2760 },
		{ 2312, 2282 },
		{ 1264, 1263 },
		{ 1642, 1641 },
		{ 2050, 2027 },
		{ 2887, 2886 },
		{ 3006, 3005 },
		{ 3114, 3109 },
		{ 1385, 1384 },
		{ 2910, 2908 },
		{ 130, 115 },
		{ 1813, 1812 },
		{ 2936, 2935 },
		{ 3162, 3159 },
		{ 1898, 1879 },
		{ 3045, 3044 },
		{ 2091, 2070 },
		{ 1888, 1869 },
		{ 2374, 2343 },
		{ 3082, 3080 },
		{ 2628, 2602 },
		{ 1908, 1890 },
		{ 1467, 1445 },
		{ 2078, 2057 },
		{ 2724, 2704 },
		{ 2513, 2485 },
		{ 3030, 3029 },
		{ 3061, 3060 },
		{ 624, 565 },
		{ 2895, 2894 },
		{ 2758, 2740 },
		{ 2864, 2863 },
		{ 2625, 2599 },
		{ 3158, 3155 },
		{ 3013, 3012 },
		{ 3161, 3158 },
		{ 1265, 1264 },
		{ 2941, 2940 },
		{ 625, 565 },
		{ 2573, 2542 },
		{ 2989, 2988 },
		{ 2140, 2125 },
		{ 2850, 2849 },
		{ 3169, 3166 },
		{ 1678, 1677 },
		{ 2972, 2971 },
		{ 1792, 1791 },
		{ 2889, 2888 },
		{ 3176, 3174 },
		{ 1936, 1921 },
		{ 1318, 1317 },
		{ 1388, 1387 },
		{ 2698, 2673 },
		{ 2668, 2642 },
		{ 626, 565 },
		{ 1370, 1367 },
		{ 1840, 1836 },
		{ 2616, 2590 },
		{ 1420, 1413 },
		{ 2031, 2025 },
		{ 2643, 2617 },
		{ 2694, 2669 },
		{ 2647, 2621 },
		{ 204, 181 },
		{ 1733, 1732 },
		{ 1269, 1268 },
		{ 202, 181 },
		{ 1655, 1654 },
		{ 203, 181 },
		{ 2240, 2239 },
		{ 1699, 1698 },
		{ 1645, 1644 },
		{ 2608, 2582 },
		{ 1949, 1936 },
		{ 2193, 2186 },
		{ 1333, 1332 },
		{ 2863, 2862 },
		{ 201, 181 },
		{ 1957, 1947 },
		{ 2661, 2635 },
		{ 1808, 1807 },
		{ 2775, 2759 },
		{ 2572, 2541 },
		{ 3012, 3011 },
		{ 2250, 2249 },
		{ 2199, 2195 },
		{ 2872, 2871 },
		{ 2252, 2251 },
		{ 1673, 1672 },
		{ 1723, 1722 },
		{ 2669, 2643 },
		{ 3021, 3020 },
		{ 2493, 2463 },
		{ 1267, 1266 },
		{ 2727, 2707 },
		{ 2673, 2647 },
		{ 1380, 1379 },
		{ 2126, 2111 },
		{ 3139, 3136 },
		{ 2127, 2114 },
		{ 1862, 1838 },
		{ 2800, 2791 },
		{ 2587, 2557 },
		{ 3146, 3145 },
		{ 2046, 2022 },
		{ 2682, 2656 },
		{ 3149, 3148 },
		{ 1749, 1748 },
		{ 2684, 2658 },
		{ 1707, 1706 },
		{ 1601, 1585 },
		{ 3044, 3043 },
		{ 2464, 2436 },
		{ 2049, 2026 },
		{ 3159, 3156 },
		{ 2971, 2970 },
		{ 3049, 3048 },
		{ 2745, 2725 },
		{ 2817, 2816 },
		{ 3052, 3051 },
		{ 2154, 2140 },
		{ 3166, 3163 },
		{ 1818, 1817 },
		{ 2510, 2482 },
		{ 1985, 1982 },
		{ 1730, 1729 },
		{ 2980, 2979 },
		{ 2642, 2616 },
		{ 2699, 2675 },
		{ 3175, 3173 },
		{ 2555, 2525 },
		{ 1360, 1359 },
		{ 1350, 1349 },
		{ 2914, 2913 },
		{ 867, 814 },
		{ 428, 386 },
		{ 685, 623 },
		{ 752, 694 },
		{ 1257, 11 },
		{ 1291, 15 },
		{ 1714, 31 },
		{ 69, 11 },
		{ 69, 15 },
		{ 69, 31 },
		{ 2948, 57 },
		{ 1278, 13 },
		{ 1341, 19 },
		{ 69, 57 },
		{ 69, 13 },
		{ 69, 19 },
		{ 1740, 33 },
		{ 429, 386 },
		{ 1766, 35 },
		{ 69, 33 },
		{ 751, 694 },
		{ 69, 35 },
		{ 2841, 47 },
		{ 336, 291 },
		{ 686, 623 },
		{ 69, 47 },
		{ 868, 814 },
		{ 1369, 21 },
		{ 1664, 27 },
		{ 1690, 29 },
		{ 69, 21 },
		{ 69, 27 },
		{ 69, 29 },
		{ 3038, 61 },
		{ 339, 294 },
		{ 346, 300 },
		{ 69, 61 },
		{ 347, 301 },
		{ 1872, 1853 },
		{ 352, 306 },
		{ 366, 323 },
		{ 390, 348 },
		{ 393, 351 },
		{ 400, 358 },
		{ 2082, 2061 },
		{ 2083, 2062 },
		{ 411, 370 },
		{ 418, 377 },
		{ 427, 385 },
		{ 232, 197 },
		{ 441, 396 },
		{ 236, 201 },
		{ 463, 412 },
		{ 466, 416 },
		{ 1892, 1873 },
		{ 1893, 1874 },
		{ 476, 424 },
		{ 2105, 2087 },
		{ 477, 425 },
		{ 490, 438 },
		{ 499, 448 },
		{ 532, 472 },
		{ 542, 480 },
		{ 544, 482 },
		{ 545, 483 },
		{ 549, 487 },
		{ 562, 502 },
		{ 2121, 2104 },
		{ 566, 506 },
		{ 1913, 1895 },
		{ 570, 510 },
		{ 580, 520 },
		{ 595, 533 },
		{ 596, 535 },
		{ 601, 540 },
		{ 616, 555 },
		{ 237, 202 },
		{ 633, 569 },
		{ 1928, 1912 },
		{ 646, 582 },
		{ 649, 585 },
		{ 658, 594 },
		{ 673, 608 },
		{ 674, 609 },
		{ 684, 622 },
		{ 243, 208 },
		{ 1255, 11 },
		{ 1289, 15 },
		{ 1712, 31 },
		{ 704, 641 },
		{ 244, 209 },
		{ 753, 695 },
		{ 2947, 57 },
		{ 1276, 13 },
		{ 1339, 19 },
		{ 765, 706 },
		{ 767, 708 },
		{ 769, 710 },
		{ 1738, 33 },
		{ 774, 715 },
		{ 1764, 35 },
		{ 794, 735 },
		{ 805, 745 },
		{ 809, 749 },
		{ 2839, 47 },
		{ 810, 750 },
		{ 840, 784 },
		{ 859, 806 },
		{ 264, 225 },
		{ 1367, 21 },
		{ 1662, 27 },
		{ 1688, 29 },
		{ 897, 846 },
		{ 907, 856 },
		{ 922, 871 },
		{ 3036, 61 },
		{ 959, 912 },
		{ 963, 916 },
		{ 979, 935 },
		{ 995, 955 },
		{ 1020, 984 },
		{ 1023, 987 },
		{ 1031, 996 },
		{ 1041, 1006 },
		{ 1059, 1026 },
		{ 1078, 1044 },
		{ 1085, 1053 },
		{ 1087, 1055 },
		{ 1100, 1072 },
		{ 1113, 1086 },
		{ 1119, 1093 },
		{ 1126, 1101 },
		{ 1127, 1102 },
		{ 1141, 1122 },
		{ 1142, 1123 },
		{ 1160, 1143 },
		{ 1165, 1149 },
		{ 1173, 1161 },
		{ 1185, 1174 },
		{ 1191, 1180 },
		{ 1226, 1225 },
		{ 274, 234 },
		{ 281, 241 },
		{ 296, 253 },
		{ 303, 260 },
		{ 2060, 2039 },
		{ 309, 265 },
		{ 319, 275 },
		{ 434, 390 },
		{ 69, 37 },
		{ 435, 390 },
		{ 433, 390 },
		{ 69, 59 },
		{ 69, 43 },
		{ 69, 49 },
		{ 69, 17 },
		{ 69, 25 },
		{ 69, 53 },
		{ 69, 55 },
		{ 220, 189 },
		{ 3123, 3119 },
		{ 2135, 2120 },
		{ 2133, 2119 },
		{ 218, 189 },
		{ 3129, 3124 },
		{ 3095, 3094 },
		{ 2194, 2187 },
		{ 436, 390 },
		{ 2136, 2120 },
		{ 2134, 2119 },
		{ 468, 418 },
		{ 470, 418 },
		{ 2131, 2117 },
		{ 1940, 1926 },
		{ 820, 760 },
		{ 509, 457 },
		{ 510, 457 },
		{ 437, 391 },
		{ 221, 189 },
		{ 219, 189 },
		{ 727, 670 },
		{ 471, 418 },
		{ 728, 671 },
		{ 511, 457 },
		{ 587, 527 },
		{ 410, 369 },
		{ 469, 418 },
		{ 501, 450 },
		{ 522, 466 },
		{ 1116, 1089 },
		{ 2043, 2019 },
		{ 214, 186 },
		{ 1122, 1096 },
		{ 683, 621 },
		{ 213, 186 },
		{ 817, 757 },
		{ 291, 248 },
		{ 524, 466 },
		{ 2042, 2019 },
		{ 332, 287 },
		{ 215, 186 },
		{ 443, 398 },
		{ 523, 466 },
		{ 2064, 2043 },
		{ 630, 566 },
		{ 359, 313 },
		{ 913, 862 },
		{ 207, 183 },
		{ 629, 566 },
		{ 209, 183 },
		{ 2041, 2019 },
		{ 914, 863 },
		{ 208, 183 },
		{ 227, 192 },
		{ 800, 740 },
		{ 715, 656 },
		{ 628, 566 },
		{ 627, 566 },
		{ 513, 459 },
		{ 516, 461 },
		{ 421, 379 },
		{ 420, 379 },
		{ 517, 461 },
		{ 799, 740 },
		{ 1131, 1106 },
		{ 226, 192 },
		{ 258, 220 },
		{ 821, 761 },
		{ 1798, 37 },
		{ 822, 762 },
		{ 2065, 2044 },
		{ 2995, 59 },
		{ 2235, 43 },
		{ 2857, 49 },
		{ 1324, 17 },
		{ 1635, 25 },
		{ 2903, 53 },
		{ 2930, 55 },
		{ 514, 459 },
		{ 592, 532 },
		{ 300, 257 },
		{ 224, 190 },
		{ 1130, 1106 },
		{ 593, 532 },
		{ 222, 190 },
		{ 579, 519 },
		{ 268, 228 },
		{ 223, 190 },
		{ 909, 858 },
		{ 1875, 1856 },
		{ 529, 470 },
		{ 282, 242 },
		{ 1279, 1276 },
		{ 568, 508 },
		{ 594, 532 },
		{ 716, 657 },
		{ 3093, 3091 },
		{ 1156, 1139 },
		{ 2489, 2459 },
		{ 1285, 1284 },
		{ 718, 659 },
		{ 3106, 3100 },
		{ 1163, 1146 },
		{ 877, 825 },
		{ 1326, 1324 },
		{ 530, 470 },
		{ 283, 242 },
		{ 546, 484 },
		{ 1258, 1255 },
		{ 1181, 1170 },
		{ 3115, 3110 },
		{ 1182, 1171 },
		{ 1183, 1172 },
		{ 3120, 3116 },
		{ 1063, 1030 },
		{ 801, 741 },
		{ 238, 203 },
		{ 1086, 1054 },
		{ 749, 692 },
		{ 750, 693 },
		{ 423, 381 },
		{ 2842, 2839 },
		{ 1855, 1831 },
		{ 612, 550 },
		{ 698, 635 },
		{ 970, 925 },
		{ 974, 930 },
		{ 486, 434 },
		{ 826, 766 },
		{ 210, 184 },
		{ 559, 499 },
		{ 1161, 1144 },
		{ 399, 357 },
		{ 837, 780 },
		{ 211, 184 },
		{ 838, 781 },
		{ 1174, 1162 },
		{ 1176, 1164 },
		{ 553, 492 },
		{ 848, 794 },
		{ 856, 803 },
		{ 2062, 2041 },
		{ 554, 493 },
		{ 1189, 1178 },
		{ 558, 499 },
		{ 1190, 1179 },
		{ 198, 180 },
		{ 1195, 1186 },
		{ 1210, 1201 },
		{ 875, 823 },
		{ 200, 180 },
		{ 1237, 1236 },
		{ 560, 500 },
		{ 882, 830 },
		{ 3094, 3093 },
		{ 892, 841 },
		{ 403, 361 },
		{ 199, 180 },
		{ 701, 638 },
		{ 298, 255 },
		{ 348, 302 },
		{ 351, 305 },
		{ 574, 514 },
		{ 923, 872 },
		{ 3111, 3106 },
		{ 929, 878 },
		{ 935, 885 },
		{ 940, 890 },
		{ 945, 897 },
		{ 951, 905 },
		{ 958, 911 },
		{ 3119, 3115 },
		{ 719, 660 },
		{ 578, 518 },
		{ 491, 439 },
		{ 729, 672 },
		{ 3124, 3120 },
		{ 419, 378 },
		{ 980, 936 },
		{ 992, 952 },
		{ 585, 525 },
		{ 1000, 960 },
		{ 1001, 961 },
		{ 1005, 965 },
		{ 1006, 966 },
		{ 1012, 972 },
		{ 265, 226 },
		{ 279, 239 },
		{ 364, 320 },
		{ 306, 263 },
		{ 1043, 1008 },
		{ 1045, 1010 },
		{ 1049, 1014 },
		{ 1056, 1023 },
		{ 1058, 1025 },
		{ 520, 464 },
		{ 368, 325 },
		{ 1071, 1037 },
		{ 614, 552 },
		{ 796, 737 },
		{ 797, 738 },
		{ 526, 468 },
		{ 1089, 1057 },
		{ 1097, 1066 },
		{ 619, 558 },
		{ 1112, 1085 },
		{ 804, 744 },
		{ 1114, 1087 },
		{ 623, 564 },
		{ 806, 746 },
		{ 370, 327 },
		{ 440, 394 },
		{ 815, 755 },
		{ 816, 756 },
		{ 1135, 1113 },
		{ 1140, 1120 },
		{ 386, 345 },
		{ 342, 296 },
		{ 1149, 1131 },
		{ 1150, 1133 },
		{ 343, 297 },
		{ 1874, 1855 },
		{ 396, 354 },
		{ 1564, 1564 },
		{ 1567, 1567 },
		{ 1570, 1570 },
		{ 341, 296 },
		{ 340, 296 },
		{ 1573, 1573 },
		{ 1576, 1576 },
		{ 1579, 1579 },
		{ 1046, 1011 },
		{ 572, 512 },
		{ 640, 576 },
		{ 1220, 1214 },
		{ 904, 853 },
		{ 906, 855 },
		{ 230, 195 },
		{ 1067, 1033 },
		{ 541, 479 },
		{ 1597, 1597 },
		{ 656, 592 },
		{ 1083, 1050 },
		{ 231, 196 },
		{ 367, 324 },
		{ 583, 523 },
		{ 1564, 1564 },
		{ 1567, 1567 },
		{ 1570, 1570 },
		{ 679, 614 },
		{ 1606, 1606 },
		{ 1573, 1573 },
		{ 1576, 1576 },
		{ 1579, 1579 },
		{ 1092, 1060 },
		{ 349, 303 },
		{ 1527, 1527 },
		{ 439, 393 },
		{ 589, 529 },
		{ 950, 904 },
		{ 407, 366 },
		{ 550, 488 },
		{ 703, 640 },
		{ 1597, 1597 },
		{ 2100, 2082 },
		{ 503, 453 },
		{ 246, 211 },
		{ 605, 544 },
		{ 976, 932 },
		{ 717, 658 },
		{ 1631, 1631 },
		{ 824, 764 },
		{ 606, 545 },
		{ 1606, 1606 },
		{ 557, 498 },
		{ 375, 332 },
		{ 615, 553 },
		{ 1154, 1137 },
		{ 414, 373 },
		{ 1527, 1527 },
		{ 738, 683 },
		{ 326, 281 },
		{ 1015, 976 },
		{ 1016, 977 },
		{ 1017, 979 },
		{ 1437, 1564 },
		{ 1437, 1567 },
		{ 1437, 1570 },
		{ 862, 809 },
		{ 863, 810 },
		{ 1437, 1573 },
		{ 1437, 1576 },
		{ 1437, 1579 },
		{ 1631, 1631 },
		{ 1026, 990 },
		{ 621, 562 },
		{ 242, 207 },
		{ 1910, 1892 },
		{ 1042, 1007 },
		{ 360, 316 },
		{ 394, 352 },
		{ 534, 474 },
		{ 1437, 1597 },
		{ 648, 584 },
		{ 918, 867 },
		{ 921, 870 },
		{ 365, 321 },
		{ 650, 586 },
		{ 1104, 1076 },
		{ 1107, 1079 },
		{ 3090, 3086 },
		{ 1108, 1080 },
		{ 1437, 1606 },
		{ 924, 873 },
		{ 1912, 1894 },
		{ 927, 876 },
		{ 779, 720 },
		{ 785, 726 },
		{ 1437, 1527 },
		{ 937, 887 },
		{ 1121, 1095 },
		{ 938, 888 },
		{ 1123, 1097 },
		{ 939, 889 },
		{ 654, 590 },
		{ 1927, 1911 },
		{ 480, 428 },
		{ 543, 481 },
		{ 588, 528 },
		{ 482, 430 },
		{ 485, 433 },
		{ 424, 382 },
		{ 1437, 1631 },
		{ 335, 290 },
		{ 598, 537 },
		{ 687, 624 },
		{ 977, 933 },
		{ 600, 539 },
		{ 397, 355 },
		{ 983, 942 },
		{ 1954, 1942 },
		{ 1955, 1943 },
		{ 2104, 2086 },
		{ 989, 949 },
		{ 497, 445 },
		{ 993, 953 },
		{ 819, 759 },
		{ 280, 240 },
		{ 710, 647 },
		{ 1002, 962 },
		{ 714, 653 },
		{ 311, 267 },
		{ 302, 259 },
		{ 2118, 2101 },
		{ 827, 767 },
		{ 834, 776 },
		{ 836, 778 },
		{ 1223, 1221 },
		{ 1018, 981 },
		{ 1227, 1226 },
		{ 404, 362 },
		{ 372, 329 },
		{ 839, 782 },
		{ 358, 312 },
		{ 1033, 998 },
		{ 721, 662 },
		{ 722, 664 },
		{ 724, 667 },
		{ 2151, 2137 },
		{ 2152, 2138 },
		{ 380, 337 },
		{ 622, 563 },
		{ 521, 465 },
		{ 733, 675 },
		{ 736, 680 },
		{ 737, 681 },
		{ 381, 339 },
		{ 747, 690 },
		{ 217, 188 },
		{ 327, 282 },
		{ 257, 219 },
		{ 1084, 1051 },
		{ 644, 580 },
		{ 249, 213 },
		{ 388, 346 },
		{ 387, 346 },
		{ 248, 213 },
		{ 285, 243 },
		{ 284, 243 },
		{ 206, 182 },
		{ 666, 601 },
		{ 667, 601 },
		{ 505, 455 },
		{ 536, 476 },
		{ 537, 476 },
		{ 538, 477 },
		{ 539, 477 },
		{ 611, 549 },
		{ 252, 216 },
		{ 205, 182 },
		{ 610, 549 },
		{ 253, 216 },
		{ 290, 247 },
		{ 506, 455 },
		{ 762, 704 },
		{ 262, 223 },
		{ 507, 456 },
		{ 307, 264 },
		{ 2061, 2040 },
		{ 273, 233 },
		{ 254, 216 },
		{ 261, 223 },
		{ 289, 247 },
		{ 389, 347 },
		{ 763, 704 },
		{ 308, 264 },
		{ 508, 456 },
		{ 986, 945 },
		{ 2066, 2045 },
		{ 518, 462 },
		{ 823, 763 },
		{ 2137, 2121 },
		{ 1152, 1135 },
		{ 465, 414 },
		{ 768, 709 },
		{ 292, 249 },
		{ 771, 712 },
		{ 669, 603 },
		{ 775, 716 },
		{ 1170, 1158 },
		{ 1911, 1893 },
		{ 1171, 1159 },
		{ 1088, 1056 },
		{ 354, 308 },
		{ 1008, 968 },
		{ 597, 536 },
		{ 936, 886 },
		{ 355, 309 },
		{ 723, 666 },
		{ 680, 616 },
		{ 1873, 1854 },
		{ 599, 538 },
		{ 260, 222 },
		{ 1929, 1913 },
		{ 2037, 2014 },
		{ 1876, 1857 },
		{ 2101, 2083 },
		{ 636, 572 },
		{ 3091, 3089 },
		{ 276, 236 },
		{ 691, 628 },
		{ 692, 629 },
		{ 2459, 2431 },
		{ 693, 630 },
		{ 481, 429 },
		{ 1229, 1228 },
		{ 1942, 1928 },
		{ 582, 522 },
		{ 899, 848 },
		{ 3110, 3105 },
		{ 445, 400 },
		{ 1133, 1111 },
		{ 1134, 1112 },
		{ 2122, 2105 },
		{ 484, 432 },
		{ 3116, 3111 },
		{ 1136, 1114 },
		{ 267, 227 },
		{ 2044, 2020 },
		{ 1105, 1077 },
		{ 1213, 1204 },
		{ 1854, 1830 },
		{ 1009, 969 },
		{ 1856, 1832 },
		{ 266, 227 },
		{ 1221, 1216 },
		{ 2132, 2118 },
		{ 1222, 1217 },
		{ 808, 748 },
		{ 1110, 1082 },
		{ 1013, 973 },
		{ 761, 703 },
		{ 297, 254 },
		{ 547, 485 },
		{ 1117, 1090 },
		{ 870, 816 },
		{ 228, 193 },
		{ 941, 891 },
		{ 1941, 1927 },
		{ 942, 892 },
		{ 1124, 1099 },
		{ 2170, 2151 },
		{ 1027, 991 },
		{ 590, 530 },
		{ 880, 828 },
		{ 496, 444 },
		{ 885, 833 },
		{ 888, 837 },
		{ 889, 838 },
		{ 968, 922 },
		{ 432, 389 },
		{ 1965, 1954 },
		{ 895, 844 },
		{ 576, 516 },
		{ 442, 397 },
		{ 1061, 1028 },
		{ 778, 719 },
		{ 556, 497 },
		{ 981, 940 },
		{ 1074, 1040 },
		{ 709, 646 },
		{ 1079, 1045 },
		{ 1169, 1157 },
		{ 1080, 1046 },
		{ 1081, 1047 },
		{ 908, 857 },
		{ 263, 224 },
		{ 795, 736 },
		{ 712, 649 },
		{ 474, 421 },
		{ 1905, 1886 },
		{ 250, 214 },
		{ 603, 542 },
		{ 487, 435 },
		{ 842, 788 },
		{ 586, 526 },
		{ 759, 701 },
		{ 1196, 1187 },
		{ 1207, 1198 },
		{ 2040, 2018 },
		{ 1208, 1199 },
		{ 731, 674 },
		{ 1236, 1235 },
		{ 575, 515 },
		{ 426, 384 },
		{ 1137, 1117 },
		{ 620, 559 },
		{ 732, 674 },
		{ 2063, 2042 },
		{ 657, 593 },
		{ 893, 842 },
		{ 1143, 1124 },
		{ 1148, 1130 },
		{ 555, 496 },
		{ 305, 262 },
		{ 955, 908 },
		{ 464, 413 },
		{ 247, 212 },
		{ 293, 250 },
		{ 632, 568 },
		{ 1093, 1061 },
		{ 1094, 1063 },
		{ 1167, 1155 },
		{ 1168, 1156 },
		{ 525, 467 },
		{ 844, 790 },
		{ 1102, 1074 },
		{ 565, 505 },
		{ 854, 801 },
		{ 1175, 1163 },
		{ 915, 864 },
		{ 1180, 1169 },
		{ 504, 454 },
		{ 1109, 1081 },
		{ 1044, 1009 },
		{ 376, 333 },
		{ 814, 754 },
		{ 1048, 1013 },
		{ 645, 581 },
		{ 1192, 1183 },
		{ 865, 812 },
		{ 395, 353 },
		{ 1205, 1196 },
		{ 928, 877 },
		{ 1060, 1027 },
		{ 446, 401 },
		{ 3089, 3085 },
		{ 725, 668 },
		{ 1214, 1207 },
		{ 1215, 1208 },
		{ 1925, 1909 },
		{ 2116, 2099 },
		{ 1219, 1213 },
		{ 451, 409 },
		{ 694, 631 },
		{ 3105, 3099 },
		{ 1129, 1105 },
		{ 1072, 1038 },
		{ 1224, 1222 },
		{ 1132, 1110 },
		{ 1073, 1039 },
		{ 697, 634 },
		{ 901, 850 },
		{ 235, 200 },
		{ 943, 895 },
		{ 438, 392 },
		{ 608, 547 },
		{ 860, 807 },
		{ 1234, 1233 },
		{ 783, 724 },
		{ 449, 405 },
		{ 688, 625 },
		{ 417, 376 },
		{ 356, 310 },
		{ 663, 598 },
		{ 971, 926 },
		{ 665, 600 },
		{ 2102, 2084 },
		{ 2103, 2085 },
		{ 408, 367 },
		{ 233, 198 },
		{ 618, 557 },
		{ 334, 289 },
		{ 678, 613 },
		{ 891, 840 },
		{ 705, 642 },
		{ 770, 711 },
		{ 591, 531 },
		{ 846, 792 },
		{ 604, 543 },
		{ 818, 758 },
		{ 1230, 1229 },
		{ 350, 304 },
		{ 635, 571 },
		{ 1151, 1134 },
		{ 1021, 985 },
		{ 1153, 1136 },
		{ 1022, 986 },
		{ 256, 218 },
		{ 1024, 988 },
		{ 967, 920 },
		{ 374, 331 },
		{ 1098, 1068 },
		{ 1166, 1152 },
		{ 1029, 994 },
		{ 1895, 1876 },
		{ 2087, 2066 },
		{ 540, 478 },
		{ 1032, 997 },
		{ 363, 319 },
		{ 699, 636 },
		{ 602, 541 },
		{ 833, 775 },
		{ 881, 829 },
		{ 925, 874 },
		{ 702, 639 },
		{ 883, 831 },
		{ 1115, 1088 },
		{ 802, 742 },
		{ 1052, 1019 },
		{ 934, 884 },
		{ 234, 199 },
		{ 317, 273 },
		{ 287, 245 },
		{ 996, 956 },
		{ 1062, 1029 },
		{ 1197, 1188 },
		{ 1202, 1193 },
		{ 997, 957 },
		{ 1206, 1197 },
		{ 1128, 1103 },
		{ 1065, 1031 },
		{ 706, 643 },
		{ 1211, 1202 },
		{ 1212, 1203 },
		{ 1066, 1031 },
		{ 322, 278 },
		{ 1064, 1031 },
		{ 843, 789 },
		{ 607, 546 },
		{ 1218, 1212 },
		{ 772, 713 },
		{ 735, 677 },
		{ 902, 851 },
		{ 946, 898 },
		{ 947, 899 },
		{ 269, 229 },
		{ 475, 423 },
		{ 1147, 1128 },
		{ 845, 791 },
		{ 734, 676 },
		{ 1004, 964 },
		{ 573, 513 },
		{ 1425, 1425 },
		{ 323, 279 },
		{ 1283, 1283 },
		{ 1283, 1283 },
		{ 528, 469 },
		{ 324, 279 },
		{ 740, 684 },
		{ 739, 684 },
		{ 527, 469 },
		{ 741, 684 },
		{ 1054, 1020 },
		{ 2120, 2103 },
		{ 855, 802 },
		{ 1204, 1195 },
		{ 310, 266 },
		{ 1053, 1020 },
		{ 857, 804 },
		{ 493, 441 },
		{ 564, 504 },
		{ 1111, 1084 },
		{ 631, 567 },
		{ 495, 443 },
		{ 384, 342 },
		{ 1425, 1425 },
		{ 634, 570 },
		{ 1283, 1283 },
		{ 869, 815 },
		{ 1217, 1210 },
		{ 369, 326 },
		{ 1028, 992 },
		{ 498, 446 },
		{ 362, 318 },
		{ 642, 578 },
		{ 500, 449 },
		{ 1039, 1004 },
		{ 1040, 1005 },
		{ 401, 359 },
		{ 812, 752 },
		{ 884, 832 },
		{ 288, 246 },
		{ 960, 913 },
		{ 961, 914 },
		{ 1047, 1012 },
		{ 886, 834 },
		{ 444, 399 },
		{ 479, 427 },
		{ 969, 923 },
		{ 239, 204 },
		{ 1057, 1024 },
		{ 755, 697 },
		{ 1926, 1910 },
		{ 973, 929 },
		{ 758, 700 },
		{ 652, 588 },
		{ 760, 702 },
		{ 898, 847 },
		{ 405, 363 },
		{ 581, 521 },
		{ 1155, 1138 },
		{ 764, 705 },
		{ 1284, 1283 },
		{ 1157, 1140 },
		{ 1437, 1425 },
		{ 985, 944 },
		{ 2084, 2063 },
		{ 2085, 2064 },
		{ 512, 458 },
		{ 988, 948 },
		{ 1077, 1043 },
		{ 613, 551 },
		{ 991, 951 },
		{ 3039, 3036 },
		{ 391, 349 },
		{ 515, 460 },
		{ 1082, 1049 },
		{ 450, 407 },
		{ 255, 217 },
		{ 670, 605 },
		{ 773, 714 },
		{ 671, 606 },
		{ 1178, 1167 },
		{ 720, 661 },
		{ 409, 368 },
		{ 1090, 1058 },
		{ 331, 286 },
		{ 488, 436 },
		{ 1187, 1176 },
		{ 1007, 967 },
		{ 321, 277 },
		{ 926, 875 },
		{ 1099, 1071 },
		{ 1011, 971 },
		{ 788, 729 },
		{ 789, 730 },
		{ 1198, 1189 },
		{ 2117, 2100 },
		{ 1199, 1190 },
		{ 2119, 2102 },
		{ 866, 813 },
		{ 1069, 1035 },
		{ 1070, 1036 },
		{ 406, 364 },
		{ 811, 751 },
		{ 1101, 1073 },
		{ 1950, 1937 },
		{ 1637, 1635 },
		{ 1691, 1688 },
		{ 1158, 1141 },
		{ 1159, 1142 },
		{ 1800, 1798 },
		{ 1715, 1712 },
		{ 2185, 2172 },
		{ 2081, 2060 },
		{ 696, 633 },
		{ 502, 451 },
		{ 1968, 1959 },
		{ 916, 865 },
		{ 1741, 1738 },
		{ 1891, 1872 },
		{ 708, 645 },
		{ 1003, 963 },
		{ 567, 507 },
		{ 2143, 2128 },
		{ 1665, 1662 },
		{ 1342, 1339 },
		{ 333, 288 },
		{ 952, 906 },
		{ 730, 673 },
		{ 241, 206 },
		{ 1030, 995 },
		{ 953, 906 },
		{ 825, 765 },
		{ 847, 793 },
		{ 373, 330 },
		{ 1138, 1118 },
		{ 1139, 1119 },
		{ 431, 388 },
		{ 832, 774 },
		{ 330, 285 },
		{ 1172, 1160 },
		{ 700, 637 },
		{ 1145, 1126 },
		{ 1146, 1127 },
		{ 835, 777 },
		{ 689, 626 },
		{ 1216, 1209 },
		{ 483, 431 },
		{ 777, 718 },
		{ 383, 341 },
		{ 919, 868 },
		{ 1125, 1100 },
		{ 743, 686 },
		{ 944, 896 },
		{ 425, 383 },
		{ 286, 244 },
		{ 1106, 1078 },
		{ 664, 599 },
		{ 911, 860 },
		{ 998, 958 },
		{ 912, 861 },
		{ 535, 475 },
		{ 195, 176 },
		{ 338, 293 },
		{ 452, 410 },
		{ 917, 866 },
		{ 422, 380 },
		{ 216, 187 },
		{ 782, 723 },
		{ 275, 235 },
		{ 377, 334 },
		{ 786, 727 },
		{ 787, 728 },
		{ 726, 669 },
		{ 850, 797 },
		{ 240, 205 },
		{ 790, 731 },
		{ 2086, 2065 },
		{ 1019, 982 },
		{ 933, 882 },
		{ 793, 734 },
		{ 467, 417 },
		{ 858, 805 },
		{ 402, 360 },
		{ 1209, 1200 },
		{ 1943, 1929 },
		{ 472, 419 },
		{ 344, 298 },
		{ 1118, 1091 },
		{ 798, 739 },
		{ 2099, 2081 },
		{ 864, 811 },
		{ 1952, 1940 },
		{ 212, 185 },
		{ 328, 283 },
		{ 385, 343 },
		{ 299, 256 },
		{ 1857, 1833 },
		{ 1034, 999 },
		{ 1038, 1003 },
		{ 271, 231 },
		{ 871, 818 },
		{ 873, 821 },
		{ 637, 573 },
		{ 1225, 1223 },
		{ 876, 824 },
		{ 312, 268 },
		{ 1228, 1227 },
		{ 879, 827 },
		{ 314, 270 },
		{ 1233, 1232 },
		{ 745, 688 },
		{ 1235, 1234 },
		{ 561, 501 },
		{ 962, 915 },
		{ 1050, 1015 },
		{ 519, 463 },
		{ 563, 503 },
		{ 1055, 1021 },
		{ 647, 583 },
		{ 301, 258 },
		{ 887, 835 },
		{ 754, 696 },
		{ 413, 372 },
		{ 2138, 2122 },
		{ 392, 350 },
		{ 975, 931 },
		{ 651, 587 },
		{ 415, 374 },
		{ 978, 934 },
		{ 2145, 2131 },
		{ 2147, 2133 },
		{ 2148, 2134 },
		{ 2149, 2135 },
		{ 2150, 2136 },
		{ 894, 843 },
		{ 653, 589 },
		{ 2045, 2021 },
		{ 896, 845 },
		{ 1894, 1875 },
		{ 707, 644 },
		{ 984, 943 },
		{ 416, 375 },
		{ 1076, 1042 },
		{ 655, 591 },
		{ 987, 946 },
		{ 353, 307 },
		{ 489, 437 },
		{ 990, 950 },
		{ 830, 770 },
		{ 448, 403 },
		{ 662, 597 },
		{ 272, 232 },
		{ 1909, 1891 },
		{ 1528, 1528 },
		{ 1528, 1528 },
		{ 1607, 1607 },
		{ 1607, 1607 },
		{ 1571, 1571 },
		{ 1571, 1571 },
		{ 1992, 1992 },
		{ 1992, 1992 },
		{ 1598, 1598 },
		{ 1598, 1598 },
		{ 1565, 1565 },
		{ 1565, 1565 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 1961, 1961 },
		{ 1961, 1961 },
		{ 1963, 1963 },
		{ 1963, 1963 },
		{ 1574, 1574 },
		{ 1574, 1574 },
		{ 2158, 2158 },
		{ 2158, 2158 },
		{ 313, 269 },
		{ 1528, 1528 },
		{ 577, 517 },
		{ 1607, 1607 },
		{ 473, 420 },
		{ 1571, 1571 },
		{ 2237, 2235 },
		{ 1992, 1992 },
		{ 695, 632 },
		{ 1598, 1598 },
		{ 861, 808 },
		{ 1565, 1565 },
		{ 999, 959 },
		{ 2212, 2212 },
		{ 661, 596 },
		{ 1961, 1961 },
		{ 784, 725 },
		{ 1963, 1963 },
		{ 766, 707 },
		{ 1574, 1574 },
		{ 412, 371 },
		{ 2158, 2158 },
		{ 2160, 2160 },
		{ 2160, 2160 },
		{ 2162, 2162 },
		{ 2162, 2162 },
		{ 2164, 2164 },
		{ 2164, 2164 },
		{ 2166, 2166 },
		{ 2166, 2166 },
		{ 1529, 1528 },
		{ 930, 879 },
		{ 1608, 1607 },
		{ 982, 941 },
		{ 1572, 1571 },
		{ 931, 880 },
		{ 1993, 1992 },
		{ 1286, 1285 },
		{ 1599, 1598 },
		{ 682, 620 },
		{ 1566, 1565 },
		{ 548, 486 },
		{ 2213, 2212 },
		{ 569, 509 },
		{ 1962, 1961 },
		{ 2160, 2160 },
		{ 1964, 1963 },
		{ 2162, 2162 },
		{ 1575, 1574 },
		{ 2164, 2164 },
		{ 2159, 2158 },
		{ 2166, 2166 },
		{ 2168, 2168 },
		{ 2168, 2168 },
		{ 1938, 1938 },
		{ 1938, 1938 },
		{ 659, 595 },
		{ 660, 595 },
		{ 1973, 1973 },
		{ 1973, 1973 },
		{ 2183, 2183 },
		{ 2183, 2183 },
		{ 1632, 1632 },
		{ 1632, 1632 },
		{ 1978, 1978 },
		{ 1978, 1978 },
		{ 2129, 2129 },
		{ 2129, 2129 },
		{ 2189, 2189 },
		{ 2189, 2189 },
		{ 1577, 1577 },
		{ 1577, 1577 },
		{ 1568, 1568 },
		{ 1568, 1568 },
		{ 2161, 2160 },
		{ 2168, 2168 },
		{ 2163, 2162 },
		{ 1938, 1938 },
		{ 2165, 2164 },
		{ 251, 215 },
		{ 2167, 2166 },
		{ 1973, 1973 },
		{ 277, 237 },
		{ 2183, 2183 },
		{ 852, 799 },
		{ 1632, 1632 },
		{ 1091, 1059 },
		{ 1978, 1978 },
		{ 551, 490 },
		{ 2129, 2129 },
		{ 318, 274 },
		{ 2189, 2189 },
		{ 295, 252 },
		{ 1577, 1577 },
		{ 1095, 1064 },
		{ 1568, 1568 },
		{ 1580, 1580 },
		{ 1580, 1580 },
		{ 1096, 1065 },
		{ 748, 691 },
		{ 1348, 1347 },
		{ 1559, 1541 },
		{ 1966, 1955 },
		{ 1643, 1642 },
		{ 2169, 2168 },
		{ 1075, 1041 },
		{ 1939, 1938 },
		{ 1914, 1897 },
		{ 3131, 3126 },
		{ 379, 336 },
		{ 1974, 1973 },
		{ 1184, 1173 },
		{ 2184, 2183 },
		{ 2107, 2090 },
		{ 1633, 1632 },
		{ 1747, 1746 },
		{ 1979, 1978 },
		{ 803, 743 },
		{ 2130, 2129 },
		{ 1580, 1580 },
		{ 2190, 2189 },
		{ 1697, 1696 },
		{ 1578, 1577 },
		{ 1186, 1175 },
		{ 1569, 1568 },
		{ 1051, 1017 },
		{ 851, 798 },
		{ 2171, 2152 },
		{ 1806, 1805 },
		{ 972, 928 },
		{ 905, 854 },
		{ 878, 826 },
		{ 1194, 1185 },
		{ 1232, 1231 },
		{ 954, 907 },
		{ 1014, 974 },
		{ 3092, 3090 },
		{ 1162, 1145 },
		{ 1934, 1919 },
		{ 1035, 1000 },
		{ 1164, 1148 },
		{ 1201, 1192 },
		{ 1036, 1001 },
		{ 828, 768 },
		{ 829, 769 },
		{ 345, 299 },
		{ 1721, 1720 },
		{ 1671, 1670 },
		{ 1581, 1580 },
		{ 910, 859 },
		{ 813, 753 },
		{ 1068, 1034 },
		{ 1144, 1125 },
		{ 746, 689 },
		{ 776, 717 },
		{ 1120, 1094 },
		{ 964, 917 },
		{ 1177, 1165 },
		{ 966, 919 },
		{ 1179, 1168 },
		{ 197, 178 },
		{ 304, 261 },
		{ 617, 556 },
		{ 2051, 2028 },
		{ 225, 191 },
		{ 2039, 2017 },
		{ 948, 900 },
		{ 1972, 1965 },
		{ 849, 795 },
		{ 1200, 1191 },
		{ 1853, 1829 },
		{ 1292, 1289 },
		{ 1953, 1941 },
		{ 2146, 2132 },
		{ 2932, 2930 },
		{ 494, 442 },
		{ 2182, 2170 },
		{ 2521, 2521 },
		{ 2521, 2521 },
		{ 2784, 2784 },
		{ 2784, 2784 },
		{ 2787, 2787 },
		{ 2787, 2787 },
		{ 2465, 2465 },
		{ 2465, 2465 },
		{ 1365, 1365 },
		{ 1365, 1365 },
		{ 1337, 1337 },
		{ 1337, 1337 },
		{ 2408, 2408 },
		{ 2408, 2408 },
		{ 2792, 2792 },
		{ 2792, 2792 },
		{ 2546, 2546 },
		{ 2546, 2546 },
		{ 2672, 2672 },
		{ 2672, 2672 },
		{ 2928, 2928 },
		{ 2928, 2928 },
		{ 609, 548 },
		{ 2521, 2521 },
		{ 1103, 1075 },
		{ 2784, 2784 },
		{ 357, 311 },
		{ 2787, 2787 },
		{ 841, 785 },
		{ 2465, 2465 },
		{ 965, 918 },
		{ 1365, 1365 },
		{ 900, 849 },
		{ 1337, 1337 },
		{ 552, 491 },
		{ 2408, 2408 },
		{ 492, 440 },
		{ 2792, 2792 },
		{ 1037, 1002 },
		{ 2546, 2546 },
		{ 903, 852 },
		{ 2672, 2672 },
		{ 329, 284 },
		{ 2928, 2928 },
		{ 791, 732 },
		{ 1660, 1660 },
		{ 1660, 1660 },
		{ 1819, 1819 },
		{ 1819, 1819 },
		{ 2551, 2521 },
		{ 792, 733 },
		{ 2795, 2784 },
		{ 278, 238 },
		{ 2796, 2787 },
		{ 742, 685 },
		{ 2466, 2465 },
		{ 2997, 2995 },
		{ 1366, 1365 },
		{ 1188, 1177 },
		{ 1338, 1337 },
		{ 584, 524 },
		{ 2409, 2408 },
		{ 744, 687 },
		{ 2801, 2792 },
		{ 294, 251 },
		{ 2578, 2546 },
		{ 361, 317 },
		{ 2697, 2672 },
		{ 1660, 1660 },
		{ 2929, 2928 },
		{ 1819, 1819 },
		{ 1193, 1184 },
		{ 2993, 2993 },
		{ 2993, 2993 },
		{ 2550, 2550 },
		{ 2550, 2550 },
		{ 2803, 2803 },
		{ 2803, 2803 },
		{ 2804, 2804 },
		{ 2804, 2804 },
		{ 2805, 2805 },
		{ 2805, 2805 },
		{ 2710, 2710 },
		{ 2710, 2710 },
		{ 2711, 2711 },
		{ 2711, 2711 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 2648, 2648 },
		{ 2648, 2648 },
		{ 2813, 2813 },
		{ 2813, 2813 },
		{ 2750, 2750 },
		{ 2750, 2750 },
		{ 1661, 1660 },
		{ 2993, 2993 },
		{ 1820, 1819 },
		{ 2550, 2550 },
		{ 853, 800 },
		{ 2803, 2803 },
		{ 378, 335 },
		{ 2804, 2804 },
		{ 531, 471 },
		{ 2805, 2805 },
		{ 398, 356 },
		{ 2710, 2710 },
		{ 533, 473 },
		{ 2711, 2711 },
		{ 229, 194 },
		{ 2258, 2258 },
		{ 320, 276 },
		{ 2648, 2648 },
		{ 920, 869 },
		{ 2813, 2813 },
		{ 478, 426 },
		{ 2750, 2750 },
		{ 1203, 1194 },
		{ 1736, 1736 },
		{ 1736, 1736 },
		{ 2574, 2574 },
		{ 2574, 2574 },
		{ 2994, 2993 },
		{ 807, 747 },
		{ 2581, 2550 },
		{ 259, 221 },
		{ 2808, 2803 },
		{ 1287, 1286 },
		{ 2809, 2804 },
		{ 756, 698 },
		{ 2810, 2805 },
		{ 757, 699 },
		{ 2730, 2710 },
		{ 382, 340 },
		{ 2731, 2711 },
		{ 2204, 2201 },
		{ 2259, 2258 },
		{ 668, 602 },
		{ 2674, 2648 },
		{ 994, 954 },
		{ 2815, 2813 },
		{ 1736, 1736 },
		{ 2766, 2750 },
		{ 2574, 2574 },
		{ 447, 402 },
		{ 2945, 2945 },
		{ 2945, 2945 },
		{ 2457, 2457 },
		{ 2457, 2457 },
		{ 1287, 1287 },
		{ 1287, 1287 },
		{ 1796, 1796 },
		{ 1796, 1796 },
		{ 1274, 1274 },
		{ 1274, 1274 },
		{ 2822, 2822 },
		{ 2822, 2822 },
		{ 1710, 1710 },
		{ 1710, 1710 },
		{ 2761, 2761 },
		{ 2761, 2761 },
		{ 2721, 2721 },
		{ 2721, 2721 },
		{ 1762, 1762 },
		{ 1762, 1762 },
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 1737, 1736 },
		{ 2945, 2945 },
		{ 2575, 2574 },
		{ 2457, 2457 },
		{ 711, 648 },
		{ 1287, 1287 },
		{ 270, 230 },
		{ 1796, 1796 },
		{ 713, 652 },
		{ 1274, 1274 },
		{ 932, 881 },
		{ 2822, 2822 },
		{ 872, 820 },
		{ 1710, 1710 },
		{ 196, 177 },
		{ 2761, 2761 },
		{ 874, 822 },
		{ 2721, 2721 },
		{ 672, 607 },
		{ 1762, 1762 },
		{ 571, 511 },
		{ 2899, 2899 },
		{ 337, 292 },
		{ 2518, 2518 },
		{ 2518, 2518 },
		{ 2765, 2765 },
		{ 2765, 2765 },
		{ 2946, 2945 },
		{ 675, 610 },
		{ 2487, 2457 },
		{ 676, 611 },
		{ 1288, 1287 },
		{ 2859, 2857 },
		{ 1797, 1796 },
		{ 2950, 2947 },
		{ 1275, 1274 },
		{ 677, 612 },
		{ 2823, 2822 },
		{ 638, 574 },
		{ 1711, 1710 },
		{ 1010, 970 },
		{ 2777, 2761 },
		{ 639, 575 },
		{ 2741, 2721 },
		{ 325, 280 },
		{ 1763, 1762 },
		{ 2518, 2518 },
		{ 2900, 2899 },
		{ 2765, 2765 },
		{ 681, 619 },
		{ 2767, 2767 },
		{ 2767, 2767 },
		{ 2768, 2768 },
		{ 2768, 2768 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2770, 2770 },
		{ 2770, 2770 },
		{ 2771, 2771 },
		{ 2771, 2771 },
		{ 2634, 2634 },
		{ 2634, 2634 },
		{ 1322, 1322 },
		{ 1322, 1322 },
		{ 3034, 3034 },
		{ 3034, 3034 },
		{ 2663, 2663 },
		{ 2663, 2663 },
		{ 2664, 2664 },
		{ 2664, 2664 },
		{ 2855, 2855 },
		{ 2855, 2855 },
		{ 2548, 2518 },
		{ 2767, 2767 },
		{ 2781, 2765 },
		{ 2768, 2768 },
		{ 641, 577 },
		{ 2659, 2659 },
		{ 1231, 1230 },
		{ 2770, 2770 },
		{ 245, 210 },
		{ 2771, 2771 },
		{ 643, 579 },
		{ 2634, 2634 },
		{ 949, 902 },
		{ 1322, 1322 },
		{ 831, 773 },
		{ 3034, 3034 },
		{ 315, 271 },
		{ 2663, 2663 },
		{ 890, 839 },
		{ 2664, 2664 },
		{ 780, 721 },
		{ 2855, 2855 },
		{ 781, 722 },
		{ 2729, 2729 },
		{ 2729, 2729 },
		{ 1686, 1686 },
		{ 1686, 1686 },
		{ 2782, 2767 },
		{ 956, 909 },
		{ 2783, 2768 },
		{ 957, 910 },
		{ 2685, 2659 },
		{ 1025, 989 },
		{ 2785, 2770 },
		{ 1767, 1764 },
		{ 2786, 2771 },
		{ 430, 387 },
		{ 2660, 2634 },
		{ 371, 328 },
		{ 1323, 1322 },
		{ 316, 272 },
		{ 3035, 3034 },
		{ 690, 627 },
		{ 2688, 2663 },
		{ 3168, 3168 },
		{ 2689, 2664 },
		{ 2729, 2729 },
		{ 2856, 2855 },
		{ 1686, 1686 },
		{ 2176, 2159 },
		{ 2779, 2779 },
		{ 2779, 2779 },
		{ 3176, 3176 },
		{ 3180, 3180 },
		{ 2196, 2190 },
		{ 1971, 1964 },
		{ 2177, 2161 },
		{ 1593, 1575 },
		{ 2178, 2163 },
		{ 1590, 1566 },
		{ 2179, 2165 },
		{ 1592, 1572 },
		{ 2180, 2167 },
		{ 1594, 1578 },
		{ 2181, 2169 },
		{ 1951, 1939 },
		{ 1634, 1633 },
		{ 3168, 3168 },
		{ 2144, 2130 },
		{ 1609, 1599 },
		{ 1994, 1993 },
		{ 1980, 1974 },
		{ 2749, 2729 },
		{ 2779, 2779 },
		{ 1687, 1686 },
		{ 3176, 3176 },
		{ 3180, 3180 },
		{ 1615, 1608 },
		{ 1591, 1569 },
		{ 2215, 2213 },
		{ 2191, 2184 },
		{ 1547, 1529 },
		{ 1983, 1979 },
		{ 1970, 1962 },
		{ 1595, 1581 },
		{ 1625, 1621 },
		{ 1702, 1701 },
		{ 1703, 1702 },
		{ 1353, 1352 },
		{ 3142, 3141 },
		{ 3143, 3142 },
		{ 1621, 1616 },
		{ 1622, 1617 },
		{ 3171, 3168 },
		{ 1397, 1396 },
		{ 3066, 3065 },
		{ 1752, 1751 },
		{ 1753, 1752 },
		{ 1648, 1647 },
		{ 2793, 2779 },
		{ 1649, 1648 },
		{ 3178, 3176 },
		{ 3181, 3180 },
		{ 1676, 1675 },
		{ 1811, 1810 },
		{ 1812, 1811 },
		{ 1677, 1676 },
		{ 1726, 1725 },
		{ 1727, 1726 },
		{ 1354, 1353 },
		{ 2829, 2829 },
		{ 2826, 2829 },
		{ 165, 165 },
		{ 162, 165 },
		{ 1923, 1908 },
		{ 1924, 1908 },
		{ 1945, 1933 },
		{ 1946, 1933 },
		{ 2225, 2225 },
		{ 1999, 1999 },
		{ 2230, 2226 },
		{ 2834, 2830 },
		{ 2004, 2000 },
		{ 164, 160 },
		{ 170, 166 },
		{ 2229, 2226 },
		{ 2833, 2830 },
		{ 2003, 2000 },
		{ 163, 160 },
		{ 169, 166 },
		{ 2233, 2232 },
		{ 2284, 2260 },
		{ 2828, 2824 },
		{ 2829, 2829 },
		{ 90, 72 },
		{ 165, 165 },
		{ 2283, 2260 },
		{ 2827, 2824 },
		{ 2344, 2313 },
		{ 89, 72 },
		{ 3077, 3069 },
		{ 2225, 2225 },
		{ 1999, 1999 },
		{ 2076, 2055 },
		{ 171, 168 },
		{ 173, 172 },
		{ 121, 105 },
		{ 2835, 2832 },
		{ 2830, 2829 },
		{ 2837, 2836 },
		{ 166, 165 },
		{ 1883, 1864 },
		{ 2005, 2002 },
		{ 2007, 2006 },
		{ 2231, 2228 },
		{ 1958, 1948 },
		{ 2226, 2225 },
		{ 2000, 1999 },
		{ 2220, 2219 },
		{ 2074, 2053 },
		{ 2006, 2004 },
		{ 2055, 2033 },
		{ 168, 164 },
		{ 2232, 2230 },
		{ 2313, 2284 },
		{ 2002, 1998 },
		{ 2832, 2828 },
		{ 105, 90 },
		{ 1864, 1842 },
		{ 172, 170 },
		{ 2836, 2834 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 0, 2883 },
		{ 0, 2606 },
		{ 2688, 2688 },
		{ 2688, 2688 },
		{ 0, 2973 },
		{ 2689, 2689 },
		{ 2689, 2689 },
		{ 0, 2739 },
		{ 173, 173 },
		{ 174, 173 },
		{ 2808, 2808 },
		{ 2808, 2808 },
		{ 0, 2890 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2810, 2810 },
		{ 2810, 2810 },
		{ 2741, 2741 },
		{ 2741, 2741 },
		{ 0, 2646 },
		{ 0, 2516 },
		{ 2409, 2409 },
		{ 0, 2984 },
		{ 0, 2376 },
		{ 0, 1787 },
		{ 2688, 2688 },
		{ 2815, 2815 },
		{ 2815, 2815 },
		{ 2689, 2689 },
		{ 2575, 2575 },
		{ 2575, 2575 },
		{ 173, 173 },
		{ 0, 2748 },
		{ 2808, 2808 },
		{ 2749, 2749 },
		{ 2749, 2749 },
		{ 2809, 2809 },
		{ 0, 2652 },
		{ 2810, 2810 },
		{ 0, 2751 },
		{ 2741, 2741 },
		{ 2823, 2823 },
		{ 2823, 2823 },
		{ 0, 2700 },
		{ 0, 2998 },
		{ 0, 2754 },
		{ 0, 2755 },
		{ 0, 2911 },
		{ 0, 3002 },
		{ 2815, 2815 },
		{ 2548, 2548 },
		{ 2548, 2548 },
		{ 2575, 2575 },
		{ 0, 1261 },
		{ 0, 2376 },
		{ 0, 2703 },
		{ 0, 3006 },
		{ 2749, 2749 },
		{ 2581, 2581 },
		{ 2581, 2581 },
		{ 0, 3009 },
		{ 0, 2247 },
		{ 2837, 2837 },
		{ 2838, 2837 },
		{ 2823, 2823 },
		{ 0, 2618 },
		{ 0, 2920 },
		{ 0, 3014 },
		{ 0, 2619 },
		{ 2660, 2660 },
		{ 2660, 2660 },
		{ 2551, 2551 },
		{ 2551, 2551 },
		{ 2548, 2548 },
		{ 2007, 2007 },
		{ 2008, 2007 },
		{ 0, 2845 },
		{ 2766, 2766 },
		{ 2766, 2766 },
		{ 0, 2911 },
		{ 0, 2476 },
		{ 2581, 2581 },
		{ 0, 2526 },
		{ 0, 2477 },
		{ 0, 3025 },
		{ 2837, 2837 },
		{ 0, 2589 },
		{ 0, 2478 },
		{ 0, 2628 },
		{ 0, 1282 },
		{ 0, 2936 },
		{ 0, 1313 },
		{ 2660, 2660 },
		{ 0, 1771 },
		{ 2551, 2551 },
		{ 2233, 2233 },
		{ 2234, 2233 },
		{ 2007, 2007 },
		{ 2777, 2777 },
		{ 2777, 2777 },
		{ 2766, 2766 },
		{ 0, 2860 },
		{ 2674, 2674 },
		{ 2674, 2674 },
		{ 0, 1329 },
		{ 2783, 2783 },
		{ 2783, 2783 },
		{ 0, 1782 },
		{ 0, 2865 },
		{ 0, 2597 },
		{ 0, 1297 },
		{ 0, 3045 },
		{ 0, 2951 },
		{ 0, 2444 },
		{ 2487, 2487 },
		{ 2487, 2487 },
		{ 0, 2538 },
		{ 0, 2955 },
		{ 2233, 2233 },
		{ 0, 2488 },
		{ 0, 2794 },
		{ 2777, 2777 },
		{ 2793, 2793 },
		{ 2793, 2793 },
		{ 0, 3056 },
		{ 2674, 2674 },
		{ 0, 2876 },
		{ 0, 2488 },
		{ 2783, 2783 },
		{ 2795, 2795 },
		{ 2795, 2795 },
		{ 2796, 2796 },
		{ 2796, 2796 },
		{ 2731, 2731 },
		{ 2731, 2731 },
		{ 0, 2603 },
		{ 0, 1308 },
		{ 2487, 2487 },
		{ 0, 2965 },
		{ 2685, 2685 },
		{ 2685, 2685 },
		{ 2801, 2801 },
		{ 2801, 2801 },
		{ 0, 2968 },
		{ 1542, 1542 },
		{ 2793, 2793 },
		{ 1240, 1240 },
		{ 1396, 1395 },
		{ 2244, 2243 },
		{ 2227, 2223 },
		{ 0, 2032 },
		{ 2831, 2827 },
		{ 2795, 2795 },
		{ 3081, 3077 },
		{ 2796, 2796 },
		{ 2001, 2003 },
		{ 2731, 2731 },
		{ 167, 163 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2685, 2685 },
		{ 0, 0 },
		{ 2801, 2801 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1542, 1542 },
		{ 0, 0 },
		{ 1240, 1240 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -70, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 46, 572, 0 },
		{ -179, 3006, 0 },
		{ 5, 0, 0 },
		{ -1239, 1017, -31 },
		{ 7, 0, -31 },
		{ -1243, 2033, -33 },
		{ 9, 0, -33 },
		{ -1256, 3329, 153 },
		{ 11, 0, 153 },
		{ -1277, 3336, 157 },
		{ 13, 0, 157 },
		{ -1290, 3330, 165 },
		{ 15, 0, 165 },
		{ -1325, 3477, 0 },
		{ 17, 0, 0 },
		{ -1340, 3337, 149 },
		{ 19, 0, 149 },
		{ -1368, 3352, 23 },
		{ 21, 0, 23 },
		{ -1410, 230, 0 },
		{ 23, 0, 0 },
		{ -1636, 3478, 0 },
		{ 25, 0, 0 },
		{ -1663, 3353, 0 },
		{ 27, 0, 0 },
		{ -1689, 3354, 0 },
		{ 29, 0, 0 },
		{ -1713, 3331, 0 },
		{ 31, 0, 0 },
		{ -1739, 3341, 0 },
		{ 33, 0, 0 },
		{ -1765, 3343, 169 },
		{ 35, 0, 169 },
		{ -1799, 3471, 276 },
		{ 37, 0, 276 },
		{ 40, 129, 0 },
		{ -1835, 344, 0 },
		{ 42, 127, 0 },
		{ -2024, 116, 0 },
		{ -2236, 3475, 0 },
		{ 43, 0, 0 },
		{ 46, 14, 0 },
		{ -88, 458, 0 },
		{ -2840, 3347, 161 },
		{ 47, 0, 161 },
		{ -2858, 3476, 184 },
		{ 49, 0, 184 },
		{ 2902, 1434, 0 },
		{ 51, 0, 0 },
		{ -2904, 3479, 282 },
		{ 53, 0, 282 },
		{ -2931, 3480, 187 },
		{ 55, 0, 187 },
		{ -2949, 3335, 180 },
		{ 57, 0, 180 },
		{ -2996, 3474, 173 },
		{ 59, 0, 173 },
		{ -3037, 3358, 179 },
		{ 61, 0, 179 },
		{ 46, 1, 0 },
		{ 63, 0, 0 },
		{ -3088, 1851, 0 },
		{ 65, 0, 0 },
		{ -3098, 2031, 45 },
		{ 67, 0, 45 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 436 },
		{ 3069, 4960, 443 },
		{ 0, 0, 254 },
		{ 0, 0, 256 },
		{ 159, 1219, 273 },
		{ 159, 1348, 273 },
		{ 159, 1247, 273 },
		{ 159, 1254, 273 },
		{ 159, 1254, 273 },
		{ 159, 1258, 273 },
		{ 159, 1263, 273 },
		{ 159, 1257, 273 },
		{ 3162, 2995, 443 },
		{ 159, 1268, 273 },
		{ 3162, 1825, 272 },
		{ 104, 2736, 443 },
		{ 159, 0, 273 },
		{ 0, 0, 443 },
		{ -89, 21, 250 },
		{ -90, 4993, 0 },
		{ 159, 1285, 273 },
		{ 159, 750, 273 },
		{ 159, 719, 273 },
		{ 159, 733, 273 },
		{ 159, 716, 273 },
		{ 159, 752, 273 },
		{ 159, 759, 273 },
		{ 159, 774, 273 },
		{ 159, 767, 273 },
		{ 3121, 2443, 0 },
		{ 159, 757, 273 },
		{ 3162, 1914, 269 },
		{ 119, 1400, 0 },
		{ 3162, 1903, 270 },
		{ 3069, 4967, 0 },
		{ 159, 766, 273 },
		{ 159, 764, 273 },
		{ 159, 791, 273 },
		{ 159, 787, 273 },
		{ 159, 0, 261 },
		{ 159, 806, 273 },
		{ 159, 808, 273 },
		{ 159, 793, 273 },
		{ 159, 826, 273 },
		{ 3159, 3111, 0 },
		{ 159, 833, 273 },
		{ 133, 1467, 0 },
		{ 119, 0, 0 },
		{ 3071, 2759, 271 },
		{ 135, 1387, 0 },
		{ 0, 0, 252 },
		{ 159, 837, 257 },
		{ 159, 839, 273 },
		{ 159, 831, 273 },
		{ 159, 836, 273 },
		{ 159, 834, 273 },
		{ 159, 827, 273 },
		{ 159, 0, 264 },
		{ 159, 828, 273 },
		{ 0, 0, 266 },
		{ 159, 834, 273 },
		{ 133, 0, 0 },
		{ 3071, 2793, 269 },
		{ 135, 0, 0 },
		{ 3071, 2860, 270 },
		{ 159, 849, 273 },
		{ 159, 846, 273 },
		{ 159, 848, 273 },
		{ 159, 878, 273 },
		{ 159, 963, 273 },
		{ 159, 0, 263 },
		{ 159, 1052, 273 },
		{ 159, 1125, 273 },
		{ 159, 1118, 273 },
		{ 159, 0, 259 },
		{ 159, 1193, 273 },
		{ 159, 0, 260 },
		{ 159, 0, 262 },
		{ 159, 1191, 273 },
		{ 159, 1199, 273 },
		{ 159, 0, 258 },
		{ 159, 1219, 273 },
		{ 159, 0, 265 },
		{ 159, 769, 273 },
		{ 159, 1247, 273 },
		{ 0, 0, 268 },
		{ 159, 1231, 273 },
		{ 159, 1235, 273 },
		{ 3179, 1305, 267 },
		{ 3069, 4949, 443 },
		{ 165, 0, 254 },
		{ 0, 0, 255 },
		{ -163, 5188, 250 },
		{ -164, 4988, 0 },
		{ 3125, 4971, 0 },
		{ 3069, 4950, 0 },
		{ 0, 0, 251 },
		{ 3069, 4965, 0 },
		{ -169, 19, 0 },
		{ -170, 4995, 0 },
		{ 173, 0, 252 },
		{ 3069, 4966, 0 },
		{ 3125, 5040, 0 },
		{ 0, 0, 253 },
		{ 3126, 1772, 147 },
		{ 2136, 4270, 147 },
		{ 2995, 4745, 147 },
		{ 3126, 4546, 147 },
		{ 0, 0, 147 },
		{ 3111, 3572, 0 },
		{ 2151, 3188, 0 },
		{ 3111, 3814, 0 },
		{ 3111, 3464, 0 },
		{ 3099, 3546, 0 },
		{ 2136, 4301, 0 },
		{ 3116, 3450, 0 },
		{ 2136, 4275, 0 },
		{ 3086, 3791, 0 },
		{ 3120, 3429, 0 },
		{ 3086, 3500, 0 },
		{ 2930, 4551, 0 },
		{ 3116, 3475, 0 },
		{ 2151, 3902, 0 },
		{ 2995, 4673, 0 },
		{ 2082, 3649, 0 },
		{ 2082, 3655, 0 },
		{ 2104, 3289, 0 },
		{ 2085, 4028, 0 },
		{ 2066, 4070, 0 },
		{ 2085, 4011, 0 },
		{ 2104, 3291, 0 },
		{ 2104, 3316, 0 },
		{ 3116, 3526, 0 },
		{ 3036, 4152, 0 },
		{ 2136, 4283, 0 },
		{ 1209, 4236, 0 },
		{ 2082, 3708, 0 },
		{ 2104, 3325, 0 },
		{ 2104, 3330, 0 },
		{ 2995, 4811, 0 },
		{ 2082, 3678, 0 },
		{ 3099, 3964, 0 },
		{ 3111, 3801, 0 },
		{ 2151, 3937, 0 },
		{ 2235, 4464, 0 },
		{ 2995, 3828, 0 },
		{ 3036, 4181, 0 },
		{ 2066, 4047, 0 },
		{ 3086, 3793, 0 },
		{ 2044, 3475, 0 },
		{ 2995, 4689, 0 },
		{ 3111, 3857, 0 },
		{ 3036, 3827, 0 },
		{ 2151, 3932, 0 },
		{ 2104, 3348, 0 },
		{ 3120, 3597, 0 },
		{ 3099, 3891, 0 },
		{ 2044, 3495, 0 },
		{ 2066, 4095, 0 },
		{ 2995, 4737, 0 },
		{ 2136, 4308, 0 },
		{ 2136, 4360, 0 },
		{ 3111, 3824, 0 },
		{ 2104, 3381, 0 },
		{ 2136, 4277, 0 },
		{ 3111, 3864, 0 },
		{ 2235, 4467, 0 },
		{ 2995, 4617, 0 },
		{ 3120, 3598, 0 },
		{ 3086, 3760, 0 },
		{ 2104, 3382, 0 },
		{ 3120, 3517, 0 },
		{ 3111, 3803, 0 },
		{ 1209, 4262, 0 },
		{ 2066, 4072, 0 },
		{ 3036, 4144, 0 },
		{ 2151, 3828, 0 },
		{ 1096, 3442, 0 },
		{ 3111, 3840, 0 },
		{ 3099, 3965, 0 },
		{ 2995, 4629, 0 },
		{ 2235, 4477, 0 },
		{ 2104, 3383, 0 },
		{ 2151, 3898, 0 },
		{ 3120, 3570, 0 },
		{ 2136, 4304, 0 },
		{ 2044, 3489, 0 },
		{ 2136, 4328, 0 },
		{ 3086, 3765, 0 },
		{ 2104, 3384, 0 },
		{ 2930, 4548, 0 },
		{ 3099, 3961, 0 },
		{ 3120, 3600, 0 },
		{ 2172, 3831, 0 },
		{ 2104, 3386, 0 },
		{ 3036, 4119, 0 },
		{ 3086, 3764, 0 },
		{ 2136, 4314, 0 },
		{ 2235, 4385, 0 },
		{ 2136, 4317, 0 },
		{ 2995, 4819, 0 },
		{ 2995, 4843, 0 },
		{ 2066, 4071, 0 },
		{ 2235, 4475, 0 },
		{ 2104, 3387, 0 },
		{ 2995, 4675, 0 },
		{ 3036, 4193, 0 },
		{ 2066, 4085, 0 },
		{ 3036, 4110, 0 },
		{ 2995, 4775, 0 },
		{ 2082, 3693, 0 },
		{ 3086, 3792, 0 },
		{ 2136, 4302, 0 },
		{ 2995, 4607, 0 },
		{ 1209, 4246, 0 },
		{ 3036, 4189, 0 },
		{ 1096, 3445, 0 },
		{ 2172, 4232, 0 },
		{ 2085, 4030, 0 },
		{ 3086, 3746, 0 },
		{ 2104, 3263, 0 },
		{ 2995, 4753, 0 },
		{ 2136, 4271, 0 },
		{ 2104, 3274, 0 },
		{ 0, 0, 78 },
		{ 3111, 3642, 0 },
		{ 3120, 3631, 0 },
		{ 2136, 4295, 0 },
		{ 3126, 4531, 0 },
		{ 2104, 3275, 0 },
		{ 2104, 3277, 0 },
		{ 3120, 3571, 0 },
		{ 2082, 3667, 0 },
		{ 2066, 4041, 0 },
		{ 3120, 3572, 0 },
		{ 2104, 3279, 0 },
		{ 2136, 4354, 0 },
		{ 3111, 3848, 0 },
		{ 3111, 3852, 0 },
		{ 2085, 4021, 0 },
		{ 2995, 4591, 0 },
		{ 3086, 3776, 0 },
		{ 863, 3453, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2082, 3711, 0 },
		{ 2995, 4631, 0 },
		{ 3036, 4136, 0 },
		{ 2066, 4058, 0 },
		{ 3120, 3599, 0 },
		{ 3086, 3719, 0 },
		{ 0, 0, 76 },
		{ 2104, 3280, 0 },
		{ 2082, 3656, 0 },
		{ 3120, 3607, 0 },
		{ 3036, 4133, 0 },
		{ 3120, 3621, 0 },
		{ 2995, 4841, 0 },
		{ 3086, 3774, 0 },
		{ 1209, 4241, 0 },
		{ 2066, 4050, 0 },
		{ 2082, 3687, 0 },
		{ 3099, 3982, 0 },
		{ 2136, 4278, 0 },
		{ 2995, 4665, 0 },
		{ 3126, 4495, 0 },
		{ 3086, 3783, 0 },
		{ 0, 0, 70 },
		{ 3086, 3789, 0 },
		{ 2995, 4697, 0 },
		{ 1209, 4256, 0 },
		{ 3036, 4127, 0 },
		{ 2136, 4303, 0 },
		{ 0, 0, 81 },
		{ 3120, 3627, 0 },
		{ 3111, 3800, 0 },
		{ 3111, 3828, 0 },
		{ 2104, 3281, 0 },
		{ 3036, 4177, 0 },
		{ 2136, 4333, 0 },
		{ 2104, 3282, 0 },
		{ 2082, 3712, 0 },
		{ 3099, 3988, 0 },
		{ 3120, 3633, 0 },
		{ 3086, 3751, 0 },
		{ 2995, 4669, 0 },
		{ 3120, 3543, 0 },
		{ 2104, 3283, 0 },
		{ 3036, 4141, 0 },
		{ 2136, 4291, 0 },
		{ 3120, 3567, 0 },
		{ 3086, 3773, 0 },
		{ 3036, 4161, 0 },
		{ 1036, 4207, 0 },
		{ 0, 0, 8 },
		{ 2082, 3672, 0 },
		{ 2085, 4027, 0 },
		{ 3036, 4187, 0 },
		{ 2117, 3430, 0 },
		{ 2104, 3286, 0 },
		{ 2235, 4405, 0 },
		{ 2136, 4331, 0 },
		{ 2082, 3690, 0 },
		{ 2136, 4336, 0 },
		{ 2136, 4350, 0 },
		{ 2085, 4020, 0 },
		{ 2104, 3287, 0 },
		{ 3120, 3588, 0 },
		{ 3120, 3471, 0 },
		{ 2136, 4274, 0 },
		{ 3116, 3530, 0 },
		{ 3086, 3744, 0 },
		{ 1209, 4261, 0 },
		{ 3099, 3951, 0 },
		{ 2104, 3288, 0 },
		{ 2151, 3257, 0 },
		{ 2995, 4839, 0 },
		{ 1209, 4244, 0 },
		{ 2151, 3916, 0 },
		{ 3126, 3416, 0 },
		{ 2117, 3422, 0 },
		{ 2085, 4013, 0 },
		{ 2082, 3669, 0 },
		{ 3120, 3622, 0 },
		{ 0, 0, 122 },
		{ 2104, 3290, 0 },
		{ 2151, 3920, 0 },
		{ 2043, 3448, 0 },
		{ 3036, 4149, 0 },
		{ 3111, 3875, 0 },
		{ 3099, 3992, 0 },
		{ 2995, 4708, 0 },
		{ 2136, 4358, 0 },
		{ 0, 0, 7 },
		{ 2085, 4018, 0 },
		{ 0, 0, 6 },
		{ 3036, 4180, 0 },
		{ 0, 0, 127 },
		{ 3099, 4000, 0 },
		{ 2136, 4272, 0 },
		{ 3126, 1840, 0 },
		{ 2104, 3292, 0 },
		{ 3099, 3963, 0 },
		{ 3111, 3838, 0 },
		{ 0, 0, 131 },
		{ 2104, 3293, 0 },
		{ 2136, 4289, 0 },
		{ 3126, 3438, 0 },
		{ 2136, 4294, 0 },
		{ 2235, 4389, 0 },
		{ 2151, 3935, 0 },
		{ 0, 0, 77 },
		{ 2066, 4096, 0 },
		{ 2104, 3296, 114 },
		{ 2104, 3298, 115 },
		{ 2995, 4679, 0 },
		{ 3036, 4150, 0 },
		{ 3086, 3739, 0 },
		{ 3111, 3869, 0 },
		{ 3086, 3742, 0 },
		{ 1209, 4254, 0 },
		{ 3111, 3879, 0 },
		{ 3086, 3743, 0 },
		{ 3116, 3537, 0 },
		{ 2151, 3939, 0 },
		{ 3036, 4190, 0 },
		{ 2136, 4355, 0 },
		{ 2104, 3299, 0 },
		{ 3120, 3585, 0 },
		{ 2995, 4601, 0 },
		{ 3036, 4122, 0 },
		{ 2930, 4562, 0 },
		{ 3036, 4126, 0 },
		{ 2151, 3911, 0 },
		{ 3086, 3757, 0 },
		{ 3036, 4135, 0 },
		{ 0, 0, 9 },
		{ 2104, 3300, 0 },
		{ 3036, 4138, 0 },
		{ 2117, 3432, 0 },
		{ 2172, 4221, 0 },
		{ 0, 0, 112 },
		{ 2082, 3677, 0 },
		{ 3099, 3979, 0 },
		{ 3111, 3818, 0 },
		{ 2151, 3832, 0 },
		{ 3099, 3432, 0 },
		{ 3036, 4171, 0 },
		{ 3116, 3488, 0 },
		{ 3036, 4178, 0 },
		{ 3116, 3472, 0 },
		{ 3111, 3834, 0 },
		{ 2136, 4324, 0 },
		{ 3120, 3606, 0 },
		{ 3086, 3785, 0 },
		{ 3116, 3452, 0 },
		{ 3099, 3971, 0 },
		{ 3120, 3612, 0 },
		{ 3036, 4113, 0 },
		{ 3120, 3516, 0 },
		{ 2995, 4667, 0 },
		{ 2104, 3301, 0 },
		{ 2995, 4671, 0 },
		{ 3086, 3714, 0 },
		{ 2136, 4269, 0 },
		{ 3111, 3809, 0 },
		{ 3111, 3811, 0 },
		{ 2066, 4056, 0 },
		{ 2082, 3651, 0 },
		{ 2104, 3302, 102 },
		{ 3086, 3740, 0 },
		{ 2104, 3303, 0 },
		{ 2104, 3304, 0 },
		{ 3116, 3517, 0 },
		{ 2151, 3899, 0 },
		{ 2235, 4426, 0 },
		{ 2104, 3305, 0 },
		{ 2082, 3673, 0 },
		{ 0, 0, 111 },
		{ 2235, 4473, 0 },
		{ 2995, 4599, 0 },
		{ 3120, 3549, 0 },
		{ 3120, 3553, 0 },
		{ 0, 0, 124 },
		{ 0, 0, 126 },
		{ 3099, 3960, 0 },
		{ 2151, 3923, 0 },
		{ 2082, 3686, 0 },
		{ 2136, 3556, 0 },
		{ 3120, 3563, 0 },
		{ 2136, 4321, 0 },
		{ 2104, 3306, 0 },
		{ 2136, 4325, 0 },
		{ 3036, 4123, 0 },
		{ 3099, 3974, 0 },
		{ 2104, 3308, 0 },
		{ 2172, 4228, 0 },
		{ 3116, 3503, 0 },
		{ 2235, 4428, 0 },
		{ 2104, 3310, 0 },
		{ 2995, 4751, 0 },
		{ 2082, 3644, 0 },
		{ 964, 4102, 0 },
		{ 3120, 3573, 0 },
		{ 3099, 3950, 0 },
		{ 2151, 3919, 0 },
		{ 2235, 4387, 0 },
		{ 3120, 3584, 0 },
		{ 2044, 3494, 0 },
		{ 2104, 3311, 0 },
		{ 3036, 4162, 0 },
		{ 3111, 3872, 0 },
		{ 2082, 3657, 0 },
		{ 2995, 4625, 0 },
		{ 3120, 3591, 0 },
		{ 2151, 3941, 0 },
		{ 2117, 3429, 0 },
		{ 3086, 3741, 0 },
		{ 2082, 3670, 0 },
		{ 2151, 3909, 0 },
		{ 2085, 4035, 0 },
		{ 3126, 3507, 0 },
		{ 2104, 3312, 0 },
		{ 0, 0, 71 },
		{ 2104, 3313, 0 },
		{ 3111, 3850, 0 },
		{ 3086, 3747, 0 },
		{ 3111, 3856, 0 },
		{ 3086, 3750, 0 },
		{ 2104, 3314, 116 },
		{ 2066, 4060, 0 },
		{ 2151, 3938, 0 },
		{ 2085, 4037, 0 },
		{ 2082, 3679, 0 },
		{ 2082, 3684, 0 },
		{ 2066, 4088, 0 },
		{ 2085, 4014, 0 },
		{ 2995, 4587, 0 },
		{ 3111, 3815, 0 },
		{ 3116, 3533, 0 },
		{ 3036, 4174, 0 },
		{ 3120, 3609, 0 },
		{ 2082, 3688, 0 },
		{ 0, 0, 128 },
		{ 2104, 3315, 0 },
		{ 2930, 4549, 0 },
		{ 2085, 4029, 0 },
		{ 3120, 3615, 0 },
		{ 3099, 3953, 0 },
		{ 0, 0, 123 },
		{ 0, 0, 113 },
		{ 2082, 3707, 0 },
		{ 3086, 3784, 0 },
		{ 3120, 3619, 0 },
		{ 2151, 3160, 0 },
		{ 3126, 3472, 0 },
		{ 3036, 4125, 0 },
		{ 3099, 3966, 0 },
		{ 2104, 3317, 0 },
		{ 3036, 4129, 0 },
		{ 2066, 4042, 0 },
		{ 3111, 3862, 0 },
		{ 2136, 4311, 0 },
		{ 2995, 4769, 0 },
		{ 2995, 4773, 0 },
		{ 2082, 3645, 0 },
		{ 2995, 4807, 0 },
		{ 3036, 4137, 0 },
		{ 2995, 4813, 0 },
		{ 3086, 3795, 0 },
		{ 3099, 3985, 0 },
		{ 2104, 3319, 0 },
		{ 2136, 4327, 0 },
		{ 3086, 3716, 0 },
		{ 2104, 3320, 0 },
		{ 3086, 3720, 0 },
		{ 2136, 4335, 0 },
		{ 3036, 4158, 0 },
		{ 2136, 4344, 0 },
		{ 3086, 3737, 0 },
		{ 2136, 4352, 0 },
		{ 2082, 3653, 0 },
		{ 3099, 3956, 0 },
		{ 2104, 3321, 0 },
		{ 3126, 4443, 0 },
		{ 2235, 4399, 0 },
		{ 2136, 4359, 0 },
		{ 2085, 4022, 0 },
		{ 2136, 4265, 0 },
		{ 2085, 4024, 0 },
		{ 3111, 3806, 0 },
		{ 2995, 4701, 0 },
		{ 3111, 3842, 0 },
		{ 0, 0, 104 },
		{ 3036, 4182, 0 },
		{ 3036, 4184, 0 },
		{ 2995, 4749, 0 },
		{ 2104, 3322, 0 },
		{ 2104, 3323, 0 },
		{ 2995, 4759, 0 },
		{ 2995, 4761, 0 },
		{ 2995, 4767, 0 },
		{ 2085, 4031, 0 },
		{ 2082, 3661, 0 },
		{ 0, 0, 134 },
		{ 3111, 3854, 0 },
		{ 0, 0, 125 },
		{ 0, 0, 129 },
		{ 2995, 4780, 0 },
		{ 2235, 4424, 0 },
		{ 1096, 3439, 0 },
		{ 2104, 3324, 0 },
		{ 3036, 3264, 0 },
		{ 3086, 3748, 0 },
		{ 2085, 4019, 0 },
		{ 1209, 4252, 0 },
		{ 2995, 4845, 0 },
		{ 3111, 3865, 0 },
		{ 3111, 3866, 0 },
		{ 3111, 3868, 0 },
		{ 3099, 4001, 0 },
		{ 2235, 4393, 0 },
		{ 2172, 4220, 0 },
		{ 3099, 4008, 0 },
		{ 3116, 3534, 0 },
		{ 2066, 4059, 0 },
		{ 1209, 4248, 0 },
		{ 3120, 3569, 0 },
		{ 2066, 4064, 0 },
		{ 2082, 3674, 0 },
		{ 2104, 3329, 0 },
		{ 2085, 4033, 0 },
		{ 2066, 4081, 0 },
		{ 2136, 4348, 0 },
		{ 2172, 4226, 0 },
		{ 2151, 3926, 0 },
		{ 3086, 3761, 0 },
		{ 2995, 4735, 0 },
		{ 2151, 3934, 0 },
		{ 0, 0, 65 },
		{ 0, 0, 66 },
		{ 2995, 4739, 0 },
		{ 3086, 3763, 0 },
		{ 0, 0, 75 },
		{ 0, 0, 120 },
		{ 2044, 3464, 0 },
		{ 3116, 3505, 0 },
		{ 2082, 3681, 0 },
		{ 3116, 3510, 0 },
		{ 3120, 3583, 0 },
		{ 3036, 4186, 0 },
		{ 3086, 3778, 0 },
		{ 0, 0, 106 },
		{ 3086, 3779, 0 },
		{ 0, 0, 108 },
		{ 3111, 3853, 0 },
		{ 3086, 3780, 0 },
		{ 3099, 3994, 0 },
		{ 2136, 4281, 0 },
		{ 2117, 3425, 0 },
		{ 2117, 3427, 0 },
		{ 3120, 3586, 0 },
		{ 1209, 4235, 0 },
		{ 2172, 3955, 0 },
		{ 3086, 3786, 0 },
		{ 964, 4100, 0 },
		{ 2066, 4091, 0 },
		{ 0, 0, 121 },
		{ 0, 0, 133 },
		{ 3086, 3787, 0 },
		{ 3086, 3788, 0 },
		{ 0, 0, 146 },
		{ 2082, 3692, 0 },
		{ 3126, 4117, 0 },
		{ 2995, 4619, 0 },
		{ 1209, 4259, 0 },
		{ 2995, 4627, 0 },
		{ 2136, 4319, 0 },
		{ 3126, 4539, 0 },
		{ 3086, 3790, 0 },
		{ 3126, 4485, 0 },
		{ 3116, 3528, 0 },
		{ 3116, 3529, 0 },
		{ 3099, 3260, 0 },
		{ 2104, 3331, 0 },
		{ 2136, 4330, 0 },
		{ 3036, 4154, 0 },
		{ 2995, 4693, 0 },
		{ 2995, 4695, 0 },
		{ 3036, 4157, 0 },
		{ 2151, 3942, 0 },
		{ 3036, 4159, 0 },
		{ 2151, 3897, 0 },
		{ 2151, 3830, 0 },
		{ 3036, 4164, 0 },
		{ 2104, 3335, 0 },
		{ 2235, 4403, 0 },
		{ 2104, 3336, 0 },
		{ 3111, 3839, 0 },
		{ 2104, 3337, 0 },
		{ 2085, 4034, 0 },
		{ 3111, 3841, 0 },
		{ 2066, 4090, 0 },
		{ 3036, 4183, 0 },
		{ 2104, 3339, 0 },
		{ 3111, 3843, 0 },
		{ 3126, 4540, 0 },
		{ 1209, 4255, 0 },
		{ 2151, 3922, 0 },
		{ 3086, 3729, 0 },
		{ 2995, 4823, 0 },
		{ 2995, 4825, 0 },
		{ 2136, 4276, 0 },
		{ 2085, 4017, 0 },
		{ 2235, 4401, 0 },
		{ 3086, 3730, 0 },
		{ 2136, 4279, 0 },
		{ 2136, 4280, 0 },
		{ 3036, 4197, 0 },
		{ 3036, 4198, 0 },
		{ 2136, 4284, 0 },
		{ 2995, 4609, 0 },
		{ 2995, 4615, 0 },
		{ 2136, 4288, 0 },
		{ 2104, 3341, 0 },
		{ 2151, 3933, 0 },
		{ 3120, 3610, 0 },
		{ 3120, 3611, 0 },
		{ 2136, 4297, 0 },
		{ 3116, 3473, 0 },
		{ 3116, 3525, 0 },
		{ 2066, 4067, 0 },
		{ 3126, 4503, 0 },
		{ 3120, 3617, 0 },
		{ 2104, 3342, 68 },
		{ 3120, 3620, 0 },
		{ 2995, 4687, 0 },
		{ 2151, 3894, 0 },
		{ 2104, 3343, 0 },
		{ 2104, 3345, 0 },
		{ 2172, 4209, 0 },
		{ 3036, 4142, 0 },
		{ 3126, 4536, 0 },
		{ 3099, 3983, 0 },
		{ 3120, 3623, 0 },
		{ 3120, 3624, 0 },
		{ 1096, 3441, 0 },
		{ 2066, 4039, 0 },
		{ 3086, 3759, 0 },
		{ 2117, 3419, 0 },
		{ 2044, 3476, 0 },
		{ 2044, 3478, 0 },
		{ 3111, 3835, 0 },
		{ 2082, 3683, 0 },
		{ 1209, 4239, 0 },
		{ 3116, 3538, 0 },
		{ 3086, 3767, 0 },
		{ 3126, 4529, 0 },
		{ 3126, 4530, 0 },
		{ 2136, 4357, 0 },
		{ 0, 0, 69 },
		{ 0, 0, 72 },
		{ 2995, 4817, 0 },
		{ 1209, 4245, 0 },
		{ 2066, 4061, 0 },
		{ 3086, 3768, 0 },
		{ 1209, 4251, 0 },
		{ 3086, 3769, 0 },
		{ 0, 0, 117 },
		{ 3120, 3544, 0 },
		{ 3120, 3546, 0 },
		{ 3086, 3775, 0 },
		{ 0, 0, 110 },
		{ 2104, 3346, 0 },
		{ 2995, 4593, 0 },
		{ 0, 0, 118 },
		{ 0, 0, 119 },
		{ 2151, 3940, 0 },
		{ 2066, 4087, 0 },
		{ 3099, 3972, 0 },
		{ 964, 4099, 0 },
		{ 2085, 4036, 0 },
		{ 1209, 4240, 0 },
		{ 3120, 3550, 0 },
		{ 2930, 4555, 0 },
		{ 0, 0, 3 },
		{ 2136, 4282, 0 },
		{ 3126, 4512, 0 },
		{ 2235, 4469, 0 },
		{ 2995, 4663, 0 },
		{ 3099, 3975, 0 },
		{ 3036, 4117, 0 },
		{ 3120, 3551, 0 },
		{ 3036, 4121, 0 },
		{ 2136, 4290, 0 },
		{ 2104, 3347, 0 },
		{ 2085, 4015, 0 },
		{ 2235, 4395, 0 },
		{ 2082, 3700, 0 },
		{ 2082, 3701, 0 },
		{ 2136, 4299, 0 },
		{ 3099, 3987, 0 },
		{ 1036, 4204, 0 },
		{ 2136, 3266, 0 },
		{ 3036, 4131, 0 },
		{ 2151, 3901, 0 },
		{ 0, 0, 79 },
		{ 2136, 4309, 0 },
		{ 0, 0, 87 },
		{ 2995, 4743, 0 },
		{ 2136, 4310, 0 },
		{ 2995, 4747, 0 },
		{ 3120, 3560, 0 },
		{ 2136, 4313, 0 },
		{ 3116, 3513, 0 },
		{ 3126, 4517, 0 },
		{ 2136, 4316, 0 },
		{ 2151, 3910, 0 },
		{ 2066, 4062, 0 },
		{ 3120, 3564, 0 },
		{ 2066, 4065, 0 },
		{ 3036, 4143, 0 },
		{ 2151, 3912, 0 },
		{ 3036, 4148, 0 },
		{ 2136, 4329, 0 },
		{ 0, 0, 74 },
		{ 2151, 3913, 0 },
		{ 2151, 3914, 0 },
		{ 2995, 4821, 0 },
		{ 2085, 4032, 0 },
		{ 3120, 3566, 0 },
		{ 3099, 3957, 0 },
		{ 2136, 4343, 0 },
		{ 2151, 3918, 0 },
		{ 2136, 4346, 0 },
		{ 2104, 3352, 0 },
		{ 3036, 4160, 0 },
		{ 3111, 3873, 0 },
		{ 2995, 4597, 0 },
		{ 2085, 4010, 0 },
		{ 2066, 4092, 0 },
		{ 2995, 4605, 0 },
		{ 2082, 3647, 0 },
		{ 3126, 4516, 0 },
		{ 2082, 3648, 0 },
		{ 2104, 3353, 0 },
		{ 2151, 3931, 0 },
		{ 2044, 3497, 0 },
		{ 3126, 4535, 0 },
		{ 2136, 4266, 0 },
		{ 2136, 4268, 0 },
		{ 863, 3454, 0 },
		{ 0, 3459, 0 },
		{ 3099, 3977, 0 },
		{ 2172, 4223, 0 },
		{ 2136, 4273, 0 },
		{ 3086, 3717, 0 },
		{ 1209, 4257, 0 },
		{ 2995, 4677, 0 },
		{ 3086, 3718, 0 },
		{ 2104, 3354, 0 },
		{ 3120, 3574, 0 },
		{ 3086, 3726, 0 },
		{ 2066, 4063, 0 },
		{ 3036, 4194, 0 },
		{ 3086, 3728, 0 },
		{ 3099, 3990, 0 },
		{ 3120, 3576, 0 },
		{ 2235, 4416, 0 },
		{ 2235, 4420, 0 },
		{ 2995, 4741, 0 },
		{ 2136, 4287, 0 },
		{ 0, 0, 73 },
		{ 2066, 4069, 0 },
		{ 3120, 3577, 0 },
		{ 3111, 3851, 0 },
		{ 3086, 3732, 0 },
		{ 3086, 3734, 0 },
		{ 3086, 3736, 0 },
		{ 3120, 3578, 0 },
		{ 2151, 3903, 0 },
		{ 2151, 3905, 0 },
		{ 0, 0, 138 },
		{ 0, 0, 139 },
		{ 2085, 4012, 0 },
		{ 1209, 4260, 0 },
		{ 3120, 3579, 0 },
		{ 2066, 4093, 0 },
		{ 2066, 4094, 0 },
		{ 2930, 4553, 0 },
		{ 0, 0, 10 },
		{ 2995, 4815, 0 },
		{ 0, 0, 12 },
		{ 2082, 3671, 0 },
		{ 3120, 3580, 0 },
		{ 2995, 4239, 0 },
		{ 3126, 4520, 0 },
		{ 3099, 3962, 0 },
		{ 2995, 4831, 0 },
		{ 2995, 4833, 0 },
		{ 3120, 3581, 0 },
		{ 2104, 3356, 0 },
		{ 3036, 4145, 0 },
		{ 3036, 4146, 0 },
		{ 2136, 4322, 0 },
		{ 2104, 3357, 0 },
		{ 3126, 4542, 0 },
		{ 2995, 4595, 0 },
		{ 3126, 4544, 0 },
		{ 2066, 4049, 0 },
		{ 0, 0, 88 },
		{ 2151, 3915, 0 },
		{ 3036, 4151, 0 },
		{ 0, 0, 86 },
		{ 3116, 3535, 0 },
		{ 2085, 4023, 0 },
		{ 0, 0, 89 },
		{ 3126, 4515, 0 },
		{ 3036, 4156, 0 },
		{ 3116, 3536, 0 },
		{ 2136, 4334, 0 },
		{ 2082, 3680, 0 },
		{ 3086, 3749, 0 },
		{ 2136, 4337, 0 },
		{ 2104, 3358, 0 },
		{ 3120, 3589, 0 },
		{ 0, 0, 67 },
		{ 0, 0, 105 },
		{ 0, 0, 107 },
		{ 2151, 3924, 0 },
		{ 2235, 4418, 0 },
		{ 3086, 3752, 0 },
		{ 2136, 4349, 0 },
		{ 3036, 4168, 0 },
		{ 3111, 3832, 0 },
		{ 2136, 4353, 0 },
		{ 0, 0, 144 },
		{ 3036, 4172, 0 },
		{ 3086, 3756, 0 },
		{ 2136, 4356, 0 },
		{ 3036, 4175, 0 },
		{ 3120, 3590, 0 },
		{ 3086, 3758, 0 },
		{ 2995, 4703, 0 },
		{ 2104, 3359, 0 },
		{ 2066, 4073, 0 },
		{ 2066, 4077, 0 },
		{ 2136, 4267, 0 },
		{ 2235, 4397, 0 },
		{ 3120, 3592, 0 },
		{ 3120, 3593, 0 },
		{ 3086, 3762, 0 },
		{ 2172, 4227, 0 },
		{ 0, 4101, 0 },
		{ 3120, 3594, 0 },
		{ 3120, 3595, 0 },
		{ 3036, 4192, 0 },
		{ 3111, 3849, 0 },
		{ 2151, 3888, 0 },
		{ 2995, 4771, 0 },
		{ 3036, 4196, 0 },
		{ 3120, 3596, 0 },
		{ 2151, 3896, 0 },
		{ 3126, 4521, 0 },
		{ 0, 0, 20 },
		{ 2082, 3694, 0 },
		{ 2082, 3695, 0 },
		{ 0, 0, 135 },
		{ 2082, 3696, 0 },
		{ 0, 0, 137 },
		{ 3086, 3771, 0 },
		{ 2136, 4286, 0 },
		{ 0, 0, 103 },
		{ 2104, 3360, 0 },
		{ 2066, 4044, 0 },
		{ 2066, 4046, 0 },
		{ 2104, 3361, 0 },
		{ 2066, 4048, 0 },
		{ 2995, 4835, 0 },
		{ 2082, 3706, 0 },
		{ 2151, 3908, 0 },
		{ 3036, 4134, 0 },
		{ 0, 0, 84 },
		{ 2066, 4053, 0 },
		{ 1209, 4237, 0 },
		{ 2104, 3362, 0 },
		{ 2066, 4057, 0 },
		{ 3086, 3777, 0 },
		{ 2136, 4306, 0 },
		{ 3126, 4525, 0 },
		{ 3126, 4528, 0 },
		{ 2995, 4603, 0 },
		{ 2136, 4307, 0 },
		{ 3036, 4139, 0 },
		{ 3036, 4140, 0 },
		{ 2104, 3363, 0 },
		{ 2082, 3710, 0 },
		{ 3120, 3601, 0 },
		{ 3099, 3981, 0 },
		{ 3120, 3602, 0 },
		{ 2082, 3643, 0 },
		{ 3036, 4147, 0 },
		{ 3099, 3984, 0 },
		{ 3120, 3603, 0 },
		{ 2136, 4323, 0 },
		{ 0, 0, 92 },
		{ 3126, 4511, 0 },
		{ 0, 0, 109 },
		{ 2066, 4068, 0 },
		{ 3126, 4121, 0 },
		{ 2136, 4326, 0 },
		{ 0, 0, 142 },
		{ 3120, 3604, 0 },
		{ 3036, 4153, 0 },
		{ 3120, 3605, 0 },
		{ 2104, 3364, 64 },
		{ 3099, 3991, 0 },
		{ 2151, 3921, 0 },
		{ 2066, 4074, 0 },
		{ 3116, 3524, 0 },
		{ 2930, 4090, 0 },
		{ 0, 0, 93 },
		{ 2082, 3650, 0 },
		{ 3126, 4537, 0 },
		{ 1036, 4205, 0 },
		{ 0, 4206, 0 },
		{ 3120, 3608, 0 },
		{ 3099, 4004, 0 },
		{ 3099, 4007, 0 },
		{ 2151, 3925, 0 },
		{ 3126, 4491, 0 },
		{ 2136, 4351, 0 },
		{ 3036, 4173, 0 },
		{ 2104, 3365, 0 },
		{ 2151, 3927, 0 },
		{ 2151, 3929, 0 },
		{ 2151, 3930, 0 },
		{ 0, 0, 97 },
		{ 3036, 4179, 0 },
		{ 2082, 3654, 0 },
		{ 3086, 3794, 0 },
		{ 0, 0, 130 },
		{ 2104, 3366, 0 },
		{ 3116, 3527, 0 },
		{ 2104, 3367, 0 },
		{ 3111, 3847, 0 },
		{ 3120, 3613, 0 },
		{ 3036, 4188, 0 },
		{ 2235, 4471, 0 },
		{ 2082, 3666, 0 },
		{ 3099, 3967, 0 },
		{ 0, 0, 99 },
		{ 3099, 3968, 0 },
		{ 2235, 4479, 0 },
		{ 2235, 4483, 0 },
		{ 3120, 3614, 0 },
		{ 0, 0, 16 },
		{ 2066, 4051, 0 },
		{ 0, 0, 18 },
		{ 0, 0, 19 },
		{ 3036, 4195, 0 },
		{ 2104, 3368, 0 },
		{ 2172, 4210, 0 },
		{ 3099, 3973, 0 },
		{ 2995, 4589, 0 },
		{ 3086, 3721, 0 },
		{ 2151, 3885, 0 },
		{ 1209, 4263, 0 },
		{ 3086, 3722, 0 },
		{ 3086, 3724, 0 },
		{ 3099, 3980, 0 },
		{ 2151, 3895, 0 },
		{ 0, 0, 63 },
		{ 3036, 4124, 0 },
		{ 3120, 3616, 0 },
		{ 2104, 3369, 0 },
		{ 3120, 3618, 0 },
		{ 2066, 4066, 0 },
		{ 1096, 3435, 0 },
		{ 2151, 3900, 0 },
		{ 2136, 4296, 0 },
		{ 0, 0, 82 },
		{ 2104, 3370, 0 },
		{ 3126, 4541, 0 },
		{ 3086, 3733, 0 },
		{ 0, 3438, 0 },
		{ 3086, 3735, 0 },
		{ 0, 0, 17 },
		{ 2151, 3906, 0 },
		{ 1209, 4258, 0 },
		{ 2104, 3371, 62 },
		{ 2104, 3372, 0 },
		{ 2066, 4079, 0 },
		{ 0, 0, 83 },
		{ 3099, 4003, 0 },
		{ 3126, 3492, 0 },
		{ 0, 0, 90 },
		{ 0, 0, 91 },
		{ 0, 0, 60 },
		{ 3099, 4006, 0 },
		{ 3111, 3876, 0 },
		{ 3111, 3877, 0 },
		{ 3120, 3625, 0 },
		{ 3111, 3881, 0 },
		{ 0, 0, 143 },
		{ 0, 0, 132 },
		{ 3099, 3952, 0 },
		{ 1209, 4242, 0 },
		{ 1209, 4243, 0 },
		{ 3120, 3626, 0 },
		{ 0, 0, 40 },
		{ 2104, 3373, 41 },
		{ 2104, 3374, 43 },
		{ 3099, 3958, 0 },
		{ 3126, 4538, 0 },
		{ 1209, 4249, 0 },
		{ 1209, 4250, 0 },
		{ 2066, 4097, 0 },
		{ 0, 0, 80 },
		{ 3099, 3959, 0 },
		{ 3120, 3629, 0 },
		{ 0, 0, 98 },
		{ 3120, 3630, 0 },
		{ 2066, 4043, 0 },
		{ 3111, 3837, 0 },
		{ 2066, 4045, 0 },
		{ 2082, 3689, 0 },
		{ 3036, 4163, 0 },
		{ 3116, 3507, 0 },
		{ 3036, 4166, 0 },
		{ 2172, 4214, 0 },
		{ 2172, 4215, 0 },
		{ 2104, 3375, 0 },
		{ 3120, 3542, 0 },
		{ 3126, 4523, 0 },
		{ 3116, 3512, 0 },
		{ 0, 0, 94 },
		{ 3126, 4526, 0 },
		{ 2104, 3376, 0 },
		{ 0, 0, 136 },
		{ 0, 0, 140 },
		{ 2066, 4052, 0 },
		{ 0, 0, 145 },
		{ 0, 0, 11 },
		{ 3099, 3969, 0 },
		{ 3099, 3970, 0 },
		{ 2151, 3928, 0 },
		{ 3111, 3844, 0 },
		{ 3111, 3846, 0 },
		{ 1209, 4247, 0 },
		{ 2104, 3377, 0 },
		{ 3120, 3547, 0 },
		{ 3099, 3976, 0 },
		{ 3120, 3548, 0 },
		{ 3126, 4543, 0 },
		{ 0, 0, 141 },
		{ 3036, 4185, 0 },
		{ 3126, 4545, 0 },
		{ 3099, 3978, 0 },
		{ 3116, 3519, 0 },
		{ 3116, 3521, 0 },
		{ 3116, 3522, 0 },
		{ 3126, 4497, 0 },
		{ 2104, 3378, 0 },
		{ 3126, 4509, 0 },
		{ 3036, 4191, 0 },
		{ 2995, 4623, 0 },
		{ 3120, 3554, 0 },
		{ 3120, 3556, 0 },
		{ 2104, 3379, 0 },
		{ 0, 0, 42 },
		{ 0, 0, 44 },
		{ 3099, 3986, 0 },
		{ 2995, 4636, 0 },
		{ 3126, 4518, 0 },
		{ 3120, 3558, 0 },
		{ 2151, 3943, 0 },
		{ 2066, 4075, 0 },
		{ 3036, 4199, 0 },
		{ 3036, 4201, 0 },
		{ 2930, 4556, 0 },
		{ 3126, 4527, 0 },
		{ 2066, 4076, 0 },
		{ 2995, 4681, 0 },
		{ 3036, 4118, 0 },
		{ 3099, 3989, 0 },
		{ 2066, 4078, 0 },
		{ 2151, 3944, 0 },
		{ 2151, 3946, 0 },
		{ 2136, 4292, 0 },
		{ 3120, 3559, 0 },
		{ 2066, 4082, 0 },
		{ 2066, 4083, 0 },
		{ 2151, 3886, 0 },
		{ 0, 0, 85 },
		{ 0, 0, 100 },
		{ 3099, 3995, 0 },
		{ 3099, 3996, 0 },
		{ 0, 4253, 0 },
		{ 3036, 4132, 0 },
		{ 0, 0, 95 },
		{ 2066, 4089, 0 },
		{ 3099, 3999, 0 },
		{ 2082, 3646, 0 },
		{ 0, 0, 13 },
		{ 2151, 3891, 0 },
		{ 2151, 3893, 0 },
		{ 0, 0, 96 },
		{ 0, 0, 61 },
		{ 0, 0, 101 },
		{ 3086, 3770, 0 },
		{ 3099, 4005, 0 },
		{ 2136, 4312, 0 },
		{ 0, 0, 15 },
		{ 2104, 3380, 0 },
		{ 3086, 3772, 0 },
		{ 2136, 4315, 0 },
		{ 3111, 3870, 0 },
		{ 2066, 4040, 0 },
		{ 2995, 4809, 0 },
		{ 3126, 4519, 0 },
		{ 2136, 4318, 0 },
		{ 2085, 4016, 0 },
		{ 2136, 4320, 0 },
		{ 3099, 3949, 0 },
		{ 3120, 3562, 0 },
		{ 0, 0, 14 },
		{ 3179, 1385, 242 },
		{ 0, 0, 243 },
		{ 3125, 5178, 244 },
		{ 3162, 1847, 248 },
		{ 1246, 2720, 249 },
		{ 0, 0, 249 },
		{ 3162, 1936, 245 },
		{ 1249, 1401, 0 },
		{ 3162, 1969, 246 },
		{ 1252, 1468, 0 },
		{ 1249, 0, 0 },
		{ 3071, 2847, 247 },
		{ 1254, 1386, 0 },
		{ 1252, 0, 0 },
		{ 3071, 2875, 245 },
		{ 1254, 0, 0 },
		{ 3071, 2885, 246 },
		{ 3116, 3518, 154 },
		{ 0, 0, 154 },
		{ 0, 0, 155 },
		{ 3140, 2199, 0 },
		{ 3162, 3026, 0 },
		{ 3179, 2242, 0 },
		{ 1262, 4994, 0 },
		{ 3159, 2768, 0 },
		{ 3162, 3102, 0 },
		{ 3174, 3138, 0 },
		{ 3170, 2636, 0 },
		{ 3173, 3197, 0 },
		{ 3179, 2294, 0 },
		{ 3173, 3169, 0 },
		{ 3175, 1946, 0 },
		{ 3062, 2850, 0 },
		{ 3177, 2350, 0 },
		{ 3121, 2497, 0 },
		{ 3140, 2208, 0 },
		{ 3180, 4773, 0 },
		{ 0, 0, 152 },
		{ 3116, 3502, 158 },
		{ 0, 0, 158 },
		{ 0, 0, 159 },
		{ 3140, 2157, 0 },
		{ 3162, 3035, 0 },
		{ 3179, 2261, 0 },
		{ 1283, 5063, 0 },
		{ 3126, 4171, 0 },
		{ 3116, 3509, 0 },
		{ 2235, 4422, 0 },
		{ 2995, 4691, 0 },
		{ 3180, 4769, 0 },
		{ 0, 0, 156 },
		{ 2930, 4558, 166 },
		{ 0, 0, 166 },
		{ 0, 0, 167 },
		{ 3162, 2978, 0 },
		{ 3008, 2955, 0 },
		{ 3177, 2355, 0 },
		{ 3179, 2280, 0 },
		{ 3162, 3096, 0 },
		{ 1298, 5050, 0 },
		{ 3162, 2702, 0 },
		{ 3144, 1443, 0 },
		{ 3162, 2980, 0 },
		{ 3179, 2241, 0 },
		{ 2772, 1406, 0 },
		{ 3175, 1831, 0 },
		{ 3169, 2923, 0 },
		{ 3062, 2883, 0 },
		{ 3121, 2456, 0 },
		{ 2964, 2937, 0 },
		{ 1309, 5076, 0 },
		{ 3162, 2687, 0 },
		{ 3170, 2605, 0 },
		{ 3140, 2192, 0 },
		{ 3162, 3064, 0 },
		{ 1314, 5030, 0 },
		{ 3172, 2647, 0 },
		{ 3167, 1519, 0 },
		{ 3121, 2451, 0 },
		{ 3174, 3152, 0 },
		{ 3175, 2013, 0 },
		{ 3062, 2762, 0 },
		{ 3177, 2398, 0 },
		{ 3121, 2506, 0 },
		{ 3180, 4849, 0 },
		{ 0, 0, 164 },
		{ 3116, 3514, 190 },
		{ 0, 0, 190 },
		{ 3140, 2213, 0 },
		{ 3162, 3066, 0 },
		{ 3179, 2254, 0 },
		{ 1330, 5046, 0 },
		{ 3174, 2808, 0 },
		{ 3170, 2654, 0 },
		{ 3173, 3179, 0 },
		{ 3140, 2216, 0 },
		{ 3140, 2066, 0 },
		{ 3162, 3025, 0 },
		{ 3140, 2072, 0 },
		{ 3180, 4631, 0 },
		{ 0, 0, 189 },
		{ 2172, 4231, 150 },
		{ 0, 0, 150 },
		{ 0, 0, 151 },
		{ 3162, 3028, 0 },
		{ 3121, 2507, 0 },
		{ 3177, 2364, 0 },
		{ 3164, 2557, 0 },
		{ 3162, 3084, 0 },
		{ 3126, 4486, 0 },
		{ 3170, 2661, 0 },
		{ 3173, 3236, 0 },
		{ 3140, 2096, 0 },
		{ 3140, 2128, 0 },
		{ 3142, 4898, 0 },
		{ 3142, 4919, 0 },
		{ 3062, 2766, 0 },
		{ 3121, 2465, 0 },
		{ 3062, 2868, 0 },
		{ 3175, 1742, 0 },
		{ 3062, 2896, 0 },
		{ 3173, 3235, 0 },
		{ 3170, 2668, 0 },
		{ 3062, 2764, 0 },
		{ 3140, 1725, 0 },
		{ 3162, 3078, 0 },
		{ 3179, 2271, 0 },
		{ 3180, 4629, 0 },
		{ 0, 0, 148 },
		{ 2673, 3158, 24 },
		{ 0, 0, 24 },
		{ 0, 0, 25 },
		{ 3162, 3089, 0 },
		{ 2964, 2927, 0 },
		{ 3062, 2852, 0 },
		{ 3121, 2433, 0 },
		{ 3125, 7, 0 },
		{ 3177, 2347, 0 },
		{ 2922, 2313, 0 },
		{ 3162, 3009, 0 },
		{ 3179, 2279, 0 },
		{ 3173, 3200, 0 },
		{ 3175, 1864, 0 },
		{ 3177, 2394, 0 },
		{ 3179, 2291, 0 },
		{ 3125, 2, 0 },
		{ 3159, 3109, 0 },
		{ 3162, 3036, 0 },
		{ 3140, 2200, 0 },
		{ 3174, 3153, 0 },
		{ 3179, 2231, 0 },
		{ 3062, 2876, 0 },
		{ 2922, 2330, 0 },
		{ 3175, 1929, 0 },
		{ 3062, 2906, 0 },
		{ 3177, 2382, 0 },
		{ 3121, 2462, 0 },
		{ 3125, 5156, 0 },
		{ 3142, 4904, 0 },
		{ 0, 0, 21 },
		{ 1413, 0, 1 },
		{ 1413, 0, 191 },
		{ 1413, 2885, 241 },
		{ 1628, 172, 241 },
		{ 1628, 410, 241 },
		{ 1628, 398, 241 },
		{ 1628, 528, 241 },
		{ 1628, 403, 241 },
		{ 1628, 414, 241 },
		{ 1628, 389, 241 },
		{ 1628, 420, 241 },
		{ 1628, 484, 241 },
		{ 1413, 0, 241 },
		{ 1425, 2695, 241 },
		{ 1413, 2976, 241 },
		{ 2673, 3161, 237 },
		{ 1628, 505, 241 },
		{ 1628, 504, 241 },
		{ 1628, 533, 241 },
		{ 1628, 0, 241 },
		{ 1628, 568, 241 },
		{ 1628, 555, 241 },
		{ 3177, 2401, 0 },
		{ 0, 0, 192 },
		{ 3121, 2459, 0 },
		{ 1628, 522, 0 },
		{ 1628, 0, 0 },
		{ 3125, 4169, 0 },
		{ 1628, 568, 0 },
		{ 1628, 585, 0 },
		{ 1628, 582, 0 },
		{ 1628, 590, 0 },
		{ 1628, 608, 0 },
		{ 1628, 611, 0 },
		{ 1628, 619, 0 },
		{ 1628, 601, 0 },
		{ 1628, 592, 0 },
		{ 1628, 585, 0 },
		{ 1628, 592, 0 },
		{ 3162, 3004, 0 },
		{ 3162, 3007, 0 },
		{ 1629, 609, 0 },
		{ 1629, 610, 0 },
		{ 1628, 620, 0 },
		{ 1628, 621, 0 },
		{ 1628, 612, 0 },
		{ 3177, 2352, 0 },
		{ 3159, 3123, 0 },
		{ 1628, 610, 0 },
		{ 1628, 654, 0 },
		{ 1628, 657, 0 },
		{ 1628, 658, 0 },
		{ 1628, 681, 0 },
		{ 1628, 682, 0 },
		{ 1628, 687, 0 },
		{ 1628, 681, 0 },
		{ 1628, 693, 0 },
		{ 1628, 684, 0 },
		{ 1628, 681, 0 },
		{ 1628, 696, 0 },
		{ 1628, 683, 0 },
		{ 3121, 2481, 0 },
		{ 3008, 2948, 0 },
		{ 1628, 699, 0 },
		{ 1628, 689, 0 },
		{ 1629, 18, 0 },
		{ 1628, 22, 0 },
		{ 1628, 29, 0 },
		{ 3170, 2610, 0 },
		{ 0, 0, 240 },
		{ 1628, 41, 0 },
		{ 1628, 26, 0 },
		{ 1628, 14, 0 },
		{ 1628, 63, 0 },
		{ 1628, 67, 0 },
		{ 1628, 68, 0 },
		{ 1628, 64, 0 },
		{ 1628, 52, 0 },
		{ 1628, 33, 0 },
		{ 1628, 37, 0 },
		{ 1628, 53, 0 },
		{ 1628, 0, 226 },
		{ 1628, 89, 0 },
		{ 3177, 2397, 0 },
		{ 3062, 2887, 0 },
		{ 1628, 47, 0 },
		{ 1628, 51, 0 },
		{ 1628, 61, 0 },
		{ 1628, 64, 0 },
		{ 1628, 90, 0 },
		{ -1508, 1544, 0 },
		{ 1629, 100, 0 },
		{ 1628, 134, 0 },
		{ 1628, 169, 0 },
		{ 1628, 174, 0 },
		{ 1628, 184, 0 },
		{ 1628, 185, 0 },
		{ 1628, 164, 0 },
		{ 1628, 180, 0 },
		{ 1628, 156, 0 },
		{ 1628, 150, 0 },
		{ 1628, 0, 225 },
		{ 1628, 157, 0 },
		{ 3164, 2574, 0 },
		{ 3121, 2436, 0 },
		{ 1628, 161, 0 },
		{ 1628, 171, 0 },
		{ 1628, 169, 0 },
		{ 1628, 0, 239 },
		{ 1628, 168, 0 },
		{ 0, 0, 227 },
		{ 1628, 172, 0 },
		{ 1630, 4, -4 },
		{ 1628, 200, 0 },
		{ 1628, 212, 0 },
		{ 1628, 273, 0 },
		{ 1628, 279, 0 },
		{ 1628, 213, 0 },
		{ 1628, 225, 0 },
		{ 1628, 222, 0 },
		{ 1628, 229, 0 },
		{ 1628, 221, 0 },
		{ 3162, 3042, 0 },
		{ 3162, 3053, 0 },
		{ 1628, 0, 229 },
		{ 1628, 301, 230 },
		{ 1628, 271, 0 },
		{ 1628, 274, 0 },
		{ 1628, 301, 0 },
		{ 1528, 3742, 0 },
		{ 3125, 4422, 0 },
		{ 2213, 4888, 216 },
		{ 1628, 304, 0 },
		{ 1628, 308, 0 },
		{ 1628, 306, 0 },
		{ 1628, 313, 0 },
		{ 1628, 315, 0 },
		{ 1628, 319, 0 },
		{ 1628, 308, 0 },
		{ 1628, 322, 0 },
		{ 1628, 304, 0 },
		{ 1628, 312, 0 },
		{ 1629, 325, 0 },
		{ 3126, 4487, 0 },
		{ 3125, 5176, 232 },
		{ 1628, 330, 0 },
		{ 1628, 343, 0 },
		{ 1628, 358, 0 },
		{ 1628, 374, 0 },
		{ 0, 0, 196 },
		{ 1630, 31, -7 },
		{ 1630, 117, -10 },
		{ 1630, 231, -13 },
		{ 1630, 345, -16 },
		{ 1630, 376, -19 },
		{ 1630, 460, -22 },
		{ 1628, 406, 0 },
		{ 1628, 419, 0 },
		{ 1628, 392, 0 },
		{ 1628, 0, 214 },
		{ 1628, 0, 228 },
		{ 3170, 2664, 0 },
		{ 1628, 391, 0 },
		{ 1628, 381, 0 },
		{ 1628, 386, 0 },
		{ 1629, 386, 0 },
		{ 1565, 3709, 0 },
		{ 3125, 4432, 0 },
		{ 2213, 4866, 217 },
		{ 1568, 3710, 0 },
		{ 3125, 4516, 0 },
		{ 2213, 4885, 218 },
		{ 1571, 3711, 0 },
		{ 3125, 4426, 0 },
		{ 2213, 4868, 221 },
		{ 1574, 3714, 0 },
		{ 3125, 4440, 0 },
		{ 2213, 4864, 222 },
		{ 1577, 3715, 0 },
		{ 3125, 4514, 0 },
		{ 2213, 4870, 223 },
		{ 1580, 3716, 0 },
		{ 3125, 4540, 0 },
		{ 2213, 4891, 224 },
		{ 1628, 436, 0 },
		{ 1630, 488, -25 },
		{ 1628, 423, 0 },
		{ 3173, 3214, 0 },
		{ 1628, 404, 0 },
		{ 1628, 481, 0 },
		{ 1628, 443, 0 },
		{ 1628, 493, 0 },
		{ 0, 0, 198 },
		{ 0, 0, 200 },
		{ 0, 0, 206 },
		{ 0, 0, 208 },
		{ 0, 0, 210 },
		{ 0, 0, 212 },
		{ 1630, 490, -28 },
		{ 1598, 3726, 0 },
		{ 3125, 4430, 0 },
		{ 2213, 4876, 220 },
		{ 1628, 0, 213 },
		{ 3140, 2211, 0 },
		{ 1628, 481, 0 },
		{ 1628, 496, 0 },
		{ 1629, 489, 0 },
		{ 1628, 486, 0 },
		{ 1607, 3736, 0 },
		{ 3125, 4424, 0 },
		{ 2213, 4884, 219 },
		{ 0, 0, 204 },
		{ 3140, 2076, 0 },
		{ 1628, 4, 235 },
		{ 1629, 493, 0 },
		{ 1628, 3, 238 },
		{ 1628, 510, 0 },
		{ 0, 0, 202 },
		{ 3142, 4901, 0 },
		{ 3142, 4902, 0 },
		{ 1628, 498, 0 },
		{ 0, 0, 236 },
		{ 1628, 495, 0 },
		{ 3142, 4895, 0 },
		{ 0, 0, 234 },
		{ 1628, 506, 0 },
		{ 1628, 511, 0 },
		{ 0, 0, 233 },
		{ 1628, 516, 0 },
		{ 1628, 519, 0 },
		{ 1629, 521, 231 },
		{ 1630, 925, 0 },
		{ 1631, 736, -1 },
		{ 1632, 3756, 0 },
		{ 3125, 4506, 0 },
		{ 2213, 4873, 215 },
		{ 0, 0, 194 },
		{ 2172, 4212, 284 },
		{ 0, 0, 284 },
		{ 3162, 3085, 0 },
		{ 3121, 2487, 0 },
		{ 3177, 2378, 0 },
		{ 3164, 2551, 0 },
		{ 3162, 3103, 0 },
		{ 3126, 4489, 0 },
		{ 3170, 2666, 0 },
		{ 3173, 3175, 0 },
		{ 3140, 2081, 0 },
		{ 3140, 2089, 0 },
		{ 3142, 4908, 0 },
		{ 3142, 4910, 0 },
		{ 3062, 2877, 0 },
		{ 3121, 2431, 0 },
		{ 3062, 2886, 0 },
		{ 3175, 1845, 0 },
		{ 3062, 2895, 0 },
		{ 3173, 3171, 0 },
		{ 3170, 2651, 0 },
		{ 3062, 2902, 0 },
		{ 3140, 1731, 0 },
		{ 3162, 3039, 0 },
		{ 3179, 2290, 0 },
		{ 3180, 4666, 0 },
		{ 0, 0, 283 },
		{ 2172, 4230, 286 },
		{ 0, 0, 286 },
		{ 0, 0, 287 },
		{ 3162, 3044, 0 },
		{ 3121, 2450, 0 },
		{ 3177, 2406, 0 },
		{ 3164, 2577, 0 },
		{ 3162, 3076, 0 },
		{ 3126, 4533, 0 },
		{ 3170, 2600, 0 },
		{ 3173, 3192, 0 },
		{ 3140, 2129, 0 },
		{ 3140, 2135, 0 },
		{ 3142, 4913, 0 },
		{ 3142, 4916, 0 },
		{ 3174, 3146, 0 },
		{ 3179, 2230, 0 },
		{ 3177, 2354, 0 },
		{ 3140, 2139, 0 },
		{ 3140, 2141, 0 },
		{ 3177, 2371, 0 },
		{ 3144, 1473, 0 },
		{ 3162, 2990, 0 },
		{ 3179, 2246, 0 },
		{ 3180, 4884, 0 },
		{ 0, 0, 285 },
		{ 2172, 4213, 289 },
		{ 0, 0, 289 },
		{ 0, 0, 290 },
		{ 3162, 2996, 0 },
		{ 3121, 2510, 0 },
		{ 3177, 2390, 0 },
		{ 3164, 2575, 0 },
		{ 3162, 3020, 0 },
		{ 3126, 4507, 0 },
		{ 3170, 2622, 0 },
		{ 3173, 3174, 0 },
		{ 3140, 2178, 0 },
		{ 3140, 2186, 0 },
		{ 3142, 4896, 0 },
		{ 3142, 4897, 0 },
		{ 3164, 2553, 0 },
		{ 3167, 1488, 0 },
		{ 3175, 1950, 0 },
		{ 3173, 3213, 0 },
		{ 3175, 1958, 0 },
		{ 3177, 2415, 0 },
		{ 3179, 2287, 0 },
		{ 3180, 4777, 0 },
		{ 0, 0, 288 },
		{ 2172, 4217, 292 },
		{ 0, 0, 292 },
		{ 0, 0, 293 },
		{ 3162, 3062, 0 },
		{ 3121, 2461, 0 },
		{ 3177, 2348, 0 },
		{ 3164, 2559, 0 },
		{ 3162, 3077, 0 },
		{ 3126, 4532, 0 },
		{ 3170, 2634, 0 },
		{ 3173, 3193, 0 },
		{ 3140, 2201, 0 },
		{ 3140, 2204, 0 },
		{ 3142, 4917, 0 },
		{ 3142, 4918, 0 },
		{ 3162, 3087, 0 },
		{ 3144, 1440, 0 },
		{ 3173, 3229, 0 },
		{ 3170, 2659, 0 },
		{ 3167, 1638, 0 },
		{ 3173, 3168, 0 },
		{ 3175, 1825, 0 },
		{ 3177, 2368, 0 },
		{ 3179, 2234, 0 },
		{ 3180, 4738, 0 },
		{ 0, 0, 291 },
		{ 2172, 4224, 295 },
		{ 0, 0, 295 },
		{ 0, 0, 296 },
		{ 3162, 2987, 0 },
		{ 3121, 2514, 0 },
		{ 3177, 2377, 0 },
		{ 3164, 2576, 0 },
		{ 3162, 2998, 0 },
		{ 3126, 4501, 0 },
		{ 3170, 2618, 0 },
		{ 3173, 3211, 0 },
		{ 3140, 2214, 0 },
		{ 3140, 2215, 0 },
		{ 3142, 4906, 0 },
		{ 3142, 4907, 0 },
		{ 3177, 2384, 0 },
		{ 2922, 2324, 0 },
		{ 3175, 1829, 0 },
		{ 3062, 2904, 0 },
		{ 3164, 2561, 0 },
		{ 3062, 2734, 0 },
		{ 3140, 2023, 0 },
		{ 3162, 3040, 0 },
		{ 3179, 2259, 0 },
		{ 3180, 4783, 0 },
		{ 0, 0, 294 },
		{ 2995, 4837, 170 },
		{ 0, 0, 170 },
		{ 0, 0, 171 },
		{ 3008, 2960, 0 },
		{ 3175, 1830, 0 },
		{ 3162, 3059, 0 },
		{ 3179, 2270, 0 },
		{ 1772, 5033, 0 },
		{ 3162, 2695, 0 },
		{ 3144, 1442, 0 },
		{ 3162, 3069, 0 },
		{ 3179, 2278, 0 },
		{ 2772, 1407, 0 },
		{ 3175, 1859, 0 },
		{ 3169, 2916, 0 },
		{ 3062, 2879, 0 },
		{ 3121, 2503, 0 },
		{ 2964, 2928, 0 },
		{ 1783, 5047, 0 },
		{ 3162, 2689, 0 },
		{ 3170, 2645, 0 },
		{ 3140, 2088, 0 },
		{ 3162, 2977, 0 },
		{ 1788, 4963, 0 },
		{ 3172, 2649, 0 },
		{ 3167, 1593, 0 },
		{ 3121, 2509, 0 },
		{ 3174, 3148, 0 },
		{ 3175, 1890, 0 },
		{ 3062, 2715, 0 },
		{ 3177, 2362, 0 },
		{ 3121, 2427, 0 },
		{ 3180, 4771, 0 },
		{ 0, 0, 168 },
		{ 2172, 4216, 277 },
		{ 0, 0, 277 },
		{ 3162, 3000, 0 },
		{ 3121, 2428, 0 },
		{ 3177, 2363, 0 },
		{ 3164, 2567, 0 },
		{ 3162, 3018, 0 },
		{ 3126, 4514, 0 },
		{ 3170, 2614, 0 },
		{ 3173, 3184, 0 },
		{ 3140, 2114, 0 },
		{ 3140, 2119, 0 },
		{ 3142, 4914, 0 },
		{ 3142, 4915, 0 },
		{ 3159, 3112, 0 },
		{ 3062, 2875, 0 },
		{ 3140, 2127, 0 },
		{ 2922, 2325, 0 },
		{ 3170, 2640, 0 },
		{ 3173, 3226, 0 },
		{ 2772, 1405, 0 },
		{ 3180, 4668, 0 },
		{ 0, 0, 275 },
		{ 1836, 0, 1 },
		{ 1995, 3030, 392 },
		{ 3162, 3046, 392 },
		{ 3173, 3099, 392 },
		{ 3159, 2346, 392 },
		{ 1836, 0, 359 },
		{ 1836, 2964, 392 },
		{ 3169, 1486, 392 },
		{ 2930, 4557, 392 },
		{ 2151, 3887, 392 },
		{ 3116, 3532, 392 },
		{ 2151, 3889, 392 },
		{ 2136, 4305, 392 },
		{ 3179, 2072, 392 },
		{ 1836, 0, 392 },
		{ 2673, 3159, 390 },
		{ 3173, 2948, 392 },
		{ 3173, 3204, 392 },
		{ 0, 0, 392 },
		{ 3177, 2396, 0 },
		{ -1841, 22, 349 },
		{ -1842, 4994, 0 },
		{ 3121, 2474, 0 },
		{ 0, 0, 355 },
		{ 0, 0, 356 },
		{ 3170, 2667, 0 },
		{ 3062, 2908, 0 },
		{ 3162, 3083, 0 },
		{ 0, 0, 360 },
		{ 3121, 2477, 0 },
		{ 3179, 2249, 0 },
		{ 3062, 2761, 0 },
		{ 2104, 3278, 0 },
		{ 3111, 3855, 0 },
		{ 3120, 3632, 0 },
		{ 2044, 3498, 0 },
		{ 3111, 3860, 0 },
		{ 3167, 1564, 0 },
		{ 3140, 2154, 0 },
		{ 3121, 2499, 0 },
		{ 3175, 1962, 0 },
		{ 3179, 2263, 0 },
		{ 3177, 2336, 0 },
		{ 3069, 4972, 0 },
		{ 3177, 2340, 0 },
		{ 3140, 2176, 0 },
		{ 3175, 1997, 0 },
		{ 3121, 2527, 0 },
		{ 3159, 3118, 0 },
		{ 3179, 2277, 0 },
		{ 3170, 2657, 0 },
		{ 2172, 4225, 0 },
		{ 2104, 3294, 0 },
		{ 2104, 3295, 0 },
		{ 2136, 4347, 0 },
		{ 2066, 4054, 0 },
		{ 3162, 2999, 0 },
		{ 3140, 2185, 0 },
		{ 3159, 3115, 0 },
		{ 3167, 1565, 0 },
		{ 3162, 3005, 0 },
		{ 3170, 2662, 0 },
		{ 0, 20, 352 },
		{ 3164, 2555, 0 },
		{ 3162, 3013, 0 },
		{ 2151, 3936, 0 },
		{ 3175, 1672, 0 },
		{ 0, 0, 391 },
		{ 3162, 3019, 0 },
		{ 3159, 3122, 0 },
		{ 2136, 4361, 0 },
		{ 2082, 3709, 0 },
		{ 3111, 3845, 0 },
		{ 3086, 3727, 0 },
		{ 2104, 3309, 0 },
		{ 0, 0, 380 },
		{ 3126, 4493, 0 },
		{ 3177, 2359, 0 },
		{ 3179, 2284, 0 },
		{ 3121, 2445, 0 },
		{ -1918, 1092, 0 },
		{ 0, 0, 351 },
		{ 3162, 3031, 0 },
		{ 0, 0, 379 },
		{ 2922, 2312, 0 },
		{ 3062, 2731, 0 },
		{ 3121, 2453, 0 },
		{ 1933, 4933, 0 },
		{ 3099, 3997, 0 },
		{ 3036, 4155, 0 },
		{ 3086, 3738, 0 },
		{ 2104, 3318, 0 },
		{ 3111, 3858, 0 },
		{ 3177, 2366, 0 },
		{ 3164, 2578, 0 },
		{ 3121, 2458, 0 },
		{ 3175, 1749, 0 },
		{ 0, 0, 381 },
		{ 3126, 4524, 358 },
		{ 3175, 1751, 0 },
		{ 3174, 3151, 0 },
		{ 3175, 1754, 0 },
		{ 0, 0, 384 },
		{ 0, 0, 385 },
		{ 1938, 0, -80 },
		{ 2117, 3418, 0 },
		{ 2151, 3904, 0 },
		{ 3111, 3871, 0 },
		{ 2136, 4293, 0 },
		{ 3062, 2851, 0 },
		{ 0, 0, 383 },
		{ 0, 0, 389 },
		{ 0, 4935, 0 },
		{ 3170, 2650, 0 },
		{ 3140, 2205, 0 },
		{ 3173, 3177, 0 },
		{ 2172, 4211, 0 },
		{ 3125, 4498, 0 },
		{ 2213, 4872, 374 },
		{ 2136, 4300, 0 },
		{ 2930, 4559, 0 },
		{ 3086, 3753, 0 },
		{ 3086, 3754, 0 },
		{ 3121, 2470, 0 },
		{ 0, 0, 386 },
		{ 0, 0, 387 },
		{ 3173, 3182, 0 },
		{ 2219, 4977, 0 },
		{ 3170, 2655, 0 },
		{ 3162, 3075, 0 },
		{ 0, 0, 364 },
		{ 1961, 0, -35 },
		{ 1963, 0, -38 },
		{ 2151, 3917, 0 },
		{ 3126, 4488, 0 },
		{ 0, 0, 382 },
		{ 3140, 2206, 0 },
		{ 0, 0, 357 },
		{ 2172, 4222, 0 },
		{ 3121, 2475, 0 },
		{ 3125, 4436, 0 },
		{ 2213, 4890, 375 },
		{ 3125, 4438, 0 },
		{ 2213, 4862, 376 },
		{ 2930, 4554, 0 },
		{ 1973, 0, -68 },
		{ 3140, 2207, 0 },
		{ 3162, 3079, 0 },
		{ 3162, 3082, 0 },
		{ 0, 0, 366 },
		{ 0, 0, 368 },
		{ 1978, 0, -74 },
		{ 3125, 4502, 0 },
		{ 2213, 4878, 378 },
		{ 0, 0, 354 },
		{ 3121, 2478, 0 },
		{ 3179, 2238, 0 },
		{ 3125, 4508, 0 },
		{ 2213, 4889, 377 },
		{ 0, 0, 372 },
		{ 3177, 2391, 0 },
		{ 3173, 3228, 0 },
		{ 0, 0, 370 },
		{ 3164, 2584, 0 },
		{ 3175, 1799, 0 },
		{ 3162, 3090, 0 },
		{ 3062, 2901, 0 },
		{ 0, 0, 388 },
		{ 3177, 2395, 0 },
		{ 3121, 2502, 0 },
		{ 1992, 0, -41 },
		{ 3125, 4428, 0 },
		{ 2213, 4877, 373 },
		{ 0, 0, 362 },
		{ 1836, 3022, 392 },
		{ 1999, 2697, 392 },
		{ -1997, 7, 349 },
		{ -1998, 4991, 0 },
		{ 3125, 4978, 0 },
		{ 3069, 4948, 0 },
		{ 0, 0, 350 },
		{ 3069, 4973, 0 },
		{ -2003, 5186, 0 },
		{ -2004, 4986, 0 },
		{ 2007, 2, 352 },
		{ 3069, 4974, 0 },
		{ 3125, 5106, 0 },
		{ 0, 0, 353 },
		{ 2025, 0, 1 },
		{ 2221, 3011, 348 },
		{ 3162, 2993, 348 },
		{ 2025, 0, 302 },
		{ 2025, 2804, 348 },
		{ 3111, 3859, 348 },
		{ 2025, 0, 305 },
		{ 3167, 1518, 348 },
		{ 2930, 4552, 348 },
		{ 2151, 3945, 348 },
		{ 3116, 3460, 348 },
		{ 2151, 3884, 348 },
		{ 2136, 4345, 348 },
		{ 3173, 3208, 348 },
		{ 3179, 2078, 348 },
		{ 2025, 0, 348 },
		{ 2673, 3162, 345 },
		{ 3173, 3217, 348 },
		{ 3159, 3105, 348 },
		{ 2930, 4550, 348 },
		{ 3173, 1745, 348 },
		{ 0, 0, 348 },
		{ 3177, 2407, 0 },
		{ -2032, 5181, 297 },
		{ -2033, 4987, 0 },
		{ 3121, 2520, 0 },
		{ 0, 0, 303 },
		{ 3121, 2521, 0 },
		{ 3177, 2411, 0 },
		{ 3179, 2260, 0 },
		{ 2104, 3385, 0 },
		{ 3111, 3823, 0 },
		{ 3120, 3552, 0 },
		{ 3099, 3955, 0 },
		{ 0, 3450, 0 },
		{ 0, 3479, 0 },
		{ 3111, 3833, 0 },
		{ 3170, 2652, 0 },
		{ 3167, 1521, 0 },
		{ 3140, 2025, 0 },
		{ 3121, 2432, 0 },
		{ 3162, 3021, 0 },
		{ 3162, 3024, 0 },
		{ 3177, 2341, 0 },
		{ 2219, 4981, 0 },
		{ 3177, 2344, 0 },
		{ 3069, 4964, 0 },
		{ 3177, 2346, 0 },
		{ 3159, 3124, 0 },
		{ 2922, 2321, 0 },
		{ 3179, 2267, 0 },
		{ 2172, 4219, 0 },
		{ 2104, 3284, 0 },
		{ 2104, 3285, 0 },
		{ 3036, 4169, 0 },
		{ 3036, 4170, 0 },
		{ 2136, 4285, 0 },
		{ 0, 4055, 0 },
		{ 3140, 2029, 0 },
		{ 3162, 3038, 0 },
		{ 3140, 2048, 0 },
		{ 3159, 3117, 0 },
		{ 3121, 2455, 0 },
		{ 3140, 2049, 0 },
		{ 3170, 2607, 0 },
		{ 0, 0, 347 },
		{ 3170, 2608, 0 },
		{ 0, 0, 299 },
		{ 3164, 2582, 0 },
		{ 0, 0, 344 },
		{ 3167, 1561, 0 },
		{ 3162, 3056, 0 },
		{ 2136, 4298, 0 },
		{ 0, 3676, 0 },
		{ 3111, 3861, 0 },
		{ 2085, 4025, 0 },
		{ 0, 4026, 0 },
		{ 3086, 3755, 0 },
		{ 2104, 3297, 0 },
		{ 3162, 3058, 0 },
		{ 0, 0, 337 },
		{ 3126, 4499, 0 },
		{ 3177, 2356, 0 },
		{ 3175, 1861, 0 },
		{ 3175, 1862, 0 },
		{ 3167, 1563, 0 },
		{ -2112, 1167, 0 },
		{ 3162, 3074, 0 },
		{ 3170, 2642, 0 },
		{ 3121, 2471, 0 },
		{ 3099, 3998, 0 },
		{ 3036, 4200, 0 },
		{ 3086, 3766, 0 },
		{ 3036, 4202, 0 },
		{ 3036, 4116, 0 },
		{ 0, 3307, 0 },
		{ 3111, 3878, 0 },
		{ 0, 0, 336 },
		{ 3177, 2365, 0 },
		{ 3164, 2569, 0 },
		{ 3062, 2801, 0 },
		{ 0, 0, 343 },
		{ 3173, 3201, 0 },
		{ 0, 0, 338 },
		{ 0, 0, 301 },
		{ 3173, 3203, 0 },
		{ 3175, 1891, 0 },
		{ 2129, 0, -65 },
		{ 0, 3417, 0 },
		{ 2151, 3892, 0 },
		{ 2120, 3414, 0 },
		{ 2117, 3413, 0 },
		{ 3111, 3836, 0 },
		{ 2136, 4332, 0 },
		{ 3062, 2847, 0 },
		{ 0, 0, 340 },
		{ 3174, 3143, 0 },
		{ 3175, 1893, 0 },
		{ 3175, 1900, 0 },
		{ 2172, 4229, 0 },
		{ 3125, 4510, 0 },
		{ 2213, 4875, 327 },
		{ 2136, 4338, 0 },
		{ 2930, 4560, 0 },
		{ 2136, 4339, 0 },
		{ 2136, 4340, 0 },
		{ 2136, 4341, 0 },
		{ 0, 4342, 0 },
		{ 3086, 3781, 0 },
		{ 3086, 3782, 0 },
		{ 3121, 2479, 0 },
		{ 3173, 3224, 0 },
		{ 3062, 2857, 0 },
		{ 3062, 2861, 0 },
		{ 3162, 3094, 0 },
		{ 0, 0, 309 },
		{ 2158, 0, -44 },
		{ 2160, 0, -47 },
		{ 2162, 0, -53 },
		{ 2164, 0, -56 },
		{ 2166, 0, -59 },
		{ 2168, 0, -62 },
		{ 0, 3907, 0 },
		{ 3126, 4513, 0 },
		{ 0, 0, 339 },
		{ 3170, 2658, 0 },
		{ 3177, 2374, 0 },
		{ 3177, 2375, 0 },
		{ 3121, 2488, 0 },
		{ 3125, 4442, 0 },
		{ 2213, 4856, 328 },
		{ 3125, 4466, 0 },
		{ 2213, 4863, 329 },
		{ 3125, 4468, 0 },
		{ 2213, 4865, 332 },
		{ 3125, 4470, 0 },
		{ 2213, 4867, 333 },
		{ 3125, 4472, 0 },
		{ 2213, 4869, 334 },
		{ 3125, 4496, 0 },
		{ 2213, 4871, 335 },
		{ 2930, 4563, 0 },
		{ 2183, 0, -71 },
		{ 0, 4218, 0 },
		{ 3121, 2492, 0 },
		{ 3121, 2493, 0 },
		{ 3162, 2983, 0 },
		{ 0, 0, 311 },
		{ 0, 0, 313 },
		{ 0, 0, 319 },
		{ 0, 0, 321 },
		{ 0, 0, 323 },
		{ 0, 0, 325 },
		{ 2189, 0, -77 },
		{ 3125, 4504, 0 },
		{ 2213, 4887, 331 },
		{ 3162, 2985, 0 },
		{ 3173, 3178, 0 },
		{ 3124, 3410, 342 },
		{ 3179, 2299, 0 },
		{ 3125, 4512, 0 },
		{ 2213, 4861, 330 },
		{ 0, 0, 317 },
		{ 3121, 2498, 0 },
		{ 3179, 2228, 0 },
		{ 0, 0, 304 },
		{ 3173, 3189, 0 },
		{ 0, 0, 315 },
		{ 3177, 2380, 0 },
		{ 2772, 1404, 0 },
		{ 3175, 1931, 0 },
		{ 3164, 2573, 0 },
		{ 2995, 4699, 0 },
		{ 3062, 2903, 0 },
		{ 3162, 3003, 0 },
		{ 3170, 2612, 0 },
		{ 3177, 2385, 0 },
		{ 0, 0, 341 },
		{ 2964, 2932, 0 },
		{ 3121, 2508, 0 },
		{ 3177, 2387, 0 },
		{ 2212, 0, -50 },
		{ 3179, 2232, 0 },
		{ 3125, 4434, 0 },
		{ 0, 4886, 326 },
		{ 3062, 2730, 0 },
		{ 0, 0, 307 },
		{ 3175, 1944, 0 },
		{ 3169, 2913, 0 },
		{ 3164, 2580, 0 },
		{ 0, 4980, 0 },
		{ 0, 0, 346 },
		{ 2025, 3021, 348 },
		{ 2225, 2693, 348 },
		{ -2223, 5180, 297 },
		{ -2224, 1, 0 },
		{ 3125, 4977, 0 },
		{ 3069, 4946, 0 },
		{ 0, 0, 298 },
		{ 3069, 4975, 0 },
		{ -2229, 8, 0 },
		{ -2230, 4989, 0 },
		{ 2233, 0, 299 },
		{ 3069, 4951, 0 },
		{ 3125, 5127, 0 },
		{ 0, 0, 300 },
		{ 0, 4391, 394 },
		{ 0, 0, 394 },
		{ 3162, 3030, 0 },
		{ 3008, 2968, 0 },
		{ 3173, 3173, 0 },
		{ 3167, 1635, 0 },
		{ 3170, 2648, 0 },
		{ 3175, 1952, 0 },
		{ 3125, 5157, 0 },
		{ 3179, 2245, 0 },
		{ 3167, 1636, 0 },
		{ 3121, 2429, 0 },
		{ 2248, 4990, 0 },
		{ 3125, 2192, 0 },
		{ 3173, 3188, 0 },
		{ 3179, 2250, 0 },
		{ 3173, 3191, 0 },
		{ 3164, 2565, 0 },
		{ 3162, 3052, 0 },
		{ 3175, 1993, 0 },
		{ 3162, 3055, 0 },
		{ 3179, 2258, 0 },
		{ 3140, 2164, 0 },
		{ 3180, 4707, 0 },
		{ 0, 0, 393 },
		{ 3069, 4957, 443 },
		{ 0, 0, 399 },
		{ 0, 0, 401 },
		{ 2280, 833, 434 },
		{ 2456, 846, 434 },
		{ 2479, 844, 434 },
		{ 2422, 845, 434 },
		{ 2281, 859, 434 },
		{ 2279, 849, 434 },
		{ 2479, 845, 434 },
		{ 2302, 859, 434 },
		{ 2452, 861, 434 },
		{ 2452, 863, 434 },
		{ 2456, 860, 434 },
		{ 2396, 870, 434 },
		{ 2278, 888, 434 },
		{ 3162, 1881, 433 },
		{ 2310, 2750, 443 },
		{ 2514, 860, 434 },
		{ 2456, 873, 434 },
		{ 2314, 874, 434 },
		{ 2456, 868, 434 },
		{ 3162, 3101, 443 },
		{ -2283, 17, 395 },
		{ -2284, 4990, 0 },
		{ 2514, 865, 434 },
		{ 2519, 484, 434 },
		{ 2514, 870, 434 },
		{ 2363, 868, 434 },
		{ 2456, 876, 434 },
		{ 2462, 871, 434 },
		{ 2456, 878, 434 },
		{ 2396, 887, 434 },
		{ 2367, 877, 434 },
		{ 2422, 879, 434 },
		{ 2396, 901, 434 },
		{ 2479, 885, 434 },
		{ 2278, 878, 434 },
		{ 2424, 885, 434 },
		{ 2311, 899, 434 },
		{ 2278, 883, 434 },
		{ 2462, 920, 434 },
		{ 2278, 930, 434 },
		{ 2492, 923, 434 },
		{ 2514, 1190, 434 },
		{ 2492, 952, 434 },
		{ 2367, 955, 434 },
		{ 2519, 576, 434 },
		{ 3162, 1979, 430 },
		{ 2341, 1354, 0 },
		{ 3162, 2002, 431 },
		{ 2492, 965, 434 },
		{ 3121, 2449, 0 },
		{ 3069, 4959, 0 },
		{ 2278, 978, 434 },
		{ 3177, 2236, 0 },
		{ 2452, 976, 434 },
		{ 2352, 961, 434 },
		{ 2492, 969, 434 },
		{ 2424, 964, 434 },
		{ 2424, 965, 434 },
		{ 2367, 1000, 434 },
		{ 2452, 1008, 434 },
		{ 2452, 1009, 434 },
		{ 2479, 997, 434 },
		{ 2422, 995, 434 },
		{ 2452, 1049, 434 },
		{ 2396, 1054, 434 },
		{ 2479, 1038, 434 },
		{ 2519, 578, 434 },
		{ 2519, 580, 434 },
		{ 2486, 1039, 434 },
		{ 2486, 1040, 434 },
		{ 2452, 1055, 434 },
		{ 2352, 1066, 434 },
		{ 2462, 1073, 434 },
		{ 2396, 1088, 434 },
		{ 2440, 1086, 434 },
		{ 3172, 2639, 0 },
		{ 2371, 1433, 0 },
		{ 2341, 0, 0 },
		{ 3071, 2774, 432 },
		{ 2373, 1434, 0 },
		{ 3159, 3119, 0 },
		{ 0, 0, 397 },
		{ 2452, 1087, 434 },
		{ 3008, 2949, 0 },
		{ 2519, 582, 434 },
		{ 2367, 1118, 434 },
		{ 2424, 1111, 434 },
		{ 2519, 10, 434 },
		{ 2456, 1158, 434 },
		{ 2278, 1112, 434 },
		{ 2365, 1131, 434 },
		{ 2519, 123, 434 },
		{ 2424, 1115, 434 },
		{ 2456, 1153, 434 },
		{ 2519, 125, 434 },
		{ 2424, 1145, 434 },
		{ 2396, 1194, 434 },
		{ 3125, 2406, 0 },
		{ 3175, 2092, 0 },
		{ 2486, 1177, 434 },
		{ 2278, 1181, 434 },
		{ 2479, 1180, 434 },
		{ 2278, 1196, 434 },
		{ 2424, 1180, 434 },
		{ 2278, 1189, 434 },
		{ 2278, 1179, 434 },
		{ 3062, 2864, 0 },
		{ 2371, 0, 0 },
		{ 3071, 2827, 430 },
		{ 2373, 0, 0 },
		{ 3071, 2837, 431 },
		{ 0, 0, 435 },
		{ 2479, 1186, 434 },
		{ 2403, 4993, 0 },
		{ 3170, 2280, 0 },
		{ 2396, 1204, 434 },
		{ 2519, 130, 434 },
		{ 3140, 2122, 0 },
		{ 2543, 6, 434 },
		{ 2486, 1187, 434 },
		{ 2396, 1206, 434 },
		{ 2424, 1188, 434 },
		{ 3125, 2166, 0 },
		{ 2519, 235, 434 },
		{ 2422, 1188, 434 },
		{ 3177, 2226, 0 },
		{ 2456, 1202, 434 },
		{ 2424, 1192, 434 },
		{ 3121, 2484, 0 },
		{ 3121, 2486, 0 },
		{ 3179, 2226, 0 },
		{ 2462, 1198, 434 },
		{ 2479, 1196, 434 },
		{ 2278, 1214, 434 },
		{ 2452, 1211, 434 },
		{ 2452, 1212, 434 },
		{ 2519, 237, 434 },
		{ 2456, 1210, 434 },
		{ 3170, 2606, 0 },
		{ 2519, 239, 434 },
		{ 3172, 2348, 0 },
		{ 3062, 2856, 0 },
		{ 2424, 1200, 434 },
		{ 3140, 2140, 0 },
		{ 3175, 1960, 0 },
		{ 3180, 4633, 0 },
		{ 3125, 5030, 405 },
		{ 2514, 1208, 434 },
		{ 2424, 1202, 434 },
		{ 2456, 1218, 434 },
		{ 3177, 2343, 0 },
		{ 3172, 2633, 0 },
		{ 2456, 1215, 434 },
		{ 3008, 2958, 0 },
		{ 2462, 1210, 434 },
		{ 2456, 1217, 434 },
		{ 3062, 2891, 0 },
		{ 3062, 2892, 0 },
		{ 3162, 3092, 0 },
		{ 2278, 1206, 434 },
		{ 2456, 1220, 434 },
		{ 2278, 1210, 434 },
		{ 2519, 241, 434 },
		{ 2519, 243, 434 },
		{ 3179, 2150, 0 },
		{ 2492, 1218, 434 },
		{ 3162, 2979, 0 },
		{ 3177, 2206, 0 },
		{ 3111, 3867, 0 },
		{ 3062, 2907, 0 },
		{ 3164, 2571, 0 },
		{ 2456, 1224, 434 },
		{ 3175, 1857, 0 },
		{ 3173, 3216, 0 },
		{ 2543, 121, 434 },
		{ 2462, 1220, 434 },
		{ 2462, 1221, 434 },
		{ 2278, 1233, 434 },
		{ 2922, 2310, 0 },
		{ 3179, 2220, 0 },
		{ 2492, 1224, 434 },
		{ 2473, 5042, 0 },
		{ 2492, 1225, 434 },
		{ 2462, 1225, 434 },
		{ 3175, 1896, 0 },
		{ 3175, 1897, 0 },
		{ 3162, 3010, 0 },
		{ 2452, 1236, 434 },
		{ 2492, 1228, 434 },
		{ 2278, 1238, 434 },
		{ 3177, 2102, 0 },
		{ 3125, 2411, 0 },
		{ 3162, 3022, 0 },
		{ 2278, 1235, 434 },
		{ 3180, 4767, 0 },
		{ 3008, 2964, 0 },
		{ 3116, 3508, 0 },
		{ 3175, 1947, 0 },
		{ 3062, 2874, 0 },
		{ 2278, 1230, 434 },
		{ 3173, 3196, 0 },
		{ 3175, 1951, 0 },
		{ 3180, 4627, 0 },
		{ 0, 0, 423 },
		{ 2479, 1228, 434 },
		{ 2492, 1233, 434 },
		{ 2519, 245, 434 },
		{ 3167, 1562, 0 },
		{ 3177, 2413, 0 },
		{ 2480, 1242, 434 },
		{ 3125, 2190, 0 },
		{ 2519, 349, 434 },
		{ 2492, 1237, 434 },
		{ 2504, 5005, 0 },
		{ 2505, 5008, 0 },
		{ 2506, 5028, 0 },
		{ 2278, 1234, 434 },
		{ 2278, 1246, 434 },
		{ 2519, 351, 434 },
		{ 3173, 3227, 0 },
		{ 3008, 2952, 0 },
		{ 3140, 2168, 0 },
		{ 3159, 3126, 0 },
		{ 2278, 1236, 434 },
		{ 3125, 5146, 428 },
		{ 2515, 5057, 0 },
		{ 3140, 2177, 0 },
		{ 3121, 2523, 0 },
		{ 3175, 1798, 0 },
		{ 2278, 1242, 434 },
		{ 3175, 1824, 0 },
		{ 3140, 2190, 0 },
		{ 2519, 355, 434 },
		{ 2519, 357, 434 },
		{ 3125, 2524, 0 },
		{ 3170, 2630, 0 },
		{ 3164, 2581, 0 },
		{ 2519, 370, 434 },
		{ 3179, 2221, 0 },
		{ 3125, 2196, 0 },
		{ 2519, 463, 434 },
		{ 3175, 2096, 0 },
		{ 3175, 2109, 0 },
		{ 3159, 2770, 0 },
		{ 2519, 469, 434 },
		{ 2519, 471, 434 },
		{ 3174, 2185, 0 },
		{ 3179, 2240, 0 },
		{ 3008, 2954, 0 },
		{ 3170, 2656, 0 },
		{ 3167, 1637, 0 },
		{ 2519, 1499, 434 },
		{ 3177, 2116, 0 },
		{ 2546, 5003, 0 },
		{ 3162, 2982, 0 },
		{ 3180, 4810, 0 },
		{ 2543, 574, 434 },
		{ 3140, 2218, 0 },
		{ 3180, 4621, 0 },
		{ 3125, 2522, 0 },
		{ 3177, 2250, 0 },
		{ 3162, 2992, 0 },
		{ 3173, 3234, 0 },
		{ 2556, 5021, 0 },
		{ 3177, 2098, 0 },
		{ 3177, 2393, 0 },
		{ 3179, 2251, 0 },
		{ 3125, 2158, 0 },
		{ 3179, 2256, 0 },
		{ 3179, 2257, 0 },
		{ 3162, 3002, 0 },
		{ 3125, 2161, 0 },
		{ 3140, 2137, 0 },
		{ 3140, 2062, 0 },
		{ 3121, 2490, 0 },
		{ 2569, 5042, 0 },
		{ 3162, 3008, 0 },
		{ 3140, 2064, 0 },
		{ 3173, 3186, 0 },
		{ 3174, 3141, 0 },
		{ 2574, 811, 434 },
		{ 3162, 3012, 0 },
		{ 2922, 2327, 0 },
		{ 3180, 4637, 0 },
		{ 3140, 2070, 0 },
		{ 3125, 5082, 403 },
		{ 3140, 2142, 0 },
		{ 3180, 4695, 0 },
		{ 3125, 5103, 412 },
		{ 3177, 2338, 0 },
		{ 2922, 2311, 0 },
		{ 3121, 2504, 0 },
		{ 3175, 1948, 0 },
		{ 3172, 2637, 0 },
		{ 3173, 3206, 0 },
		{ 3008, 2959, 0 },
		{ 2964, 2939, 0 },
		{ 3177, 2342, 0 },
		{ 3179, 2273, 0 },
		{ 3162, 3033, 0 },
		{ 3162, 3034, 0 },
		{ 2922, 2320, 0 },
		{ 3179, 2275, 0 },
		{ 3062, 2846, 0 },
		{ 3144, 1412, 0 },
		{ 3167, 1598, 0 },
		{ 3140, 2146, 0 },
		{ 3121, 2522, 0 },
		{ 2922, 2329, 0 },
		{ 3121, 2525, 0 },
		{ 3162, 3049, 0 },
		{ 3180, 4740, 0 },
		{ 3125, 5061, 426 },
		{ 3121, 2526, 0 },
		{ 3175, 1953, 0 },
		{ 0, 0, 440 },
		{ 3140, 2105, 0 },
		{ 3062, 2869, 0 },
		{ 3125, 5090, 411 },
		{ 3173, 3176, 0 },
		{ 3162, 3057, 0 },
		{ 3062, 2870, 0 },
		{ 3062, 2871, 0 },
		{ 3062, 2872, 0 },
		{ 3179, 2286, 0 },
		{ 3008, 2945, 0 },
		{ 2615, 5029, 0 },
		{ 2673, 3160, 0 },
		{ 3177, 2357, 0 },
		{ 3162, 3071, 0 },
		{ 3162, 3073, 0 },
		{ 3175, 1957, 0 },
		{ 3177, 2360, 0 },
		{ 2607, 1279, 0 },
		{ 2623, 5035, 0 },
		{ 2922, 2315, 0 },
		{ 3174, 3134, 0 },
		{ 3175, 1959, 0 },
		{ 3179, 2292, 0 },
		{ 3159, 3121, 0 },
		{ 2629, 5065, 0 },
		{ 3162, 3081, 0 },
		{ 3062, 2889, 0 },
		{ 2632, 4943, 0 },
		{ 0, 1326, 0 },
		{ 3170, 2616, 0 },
		{ 3179, 2298, 0 },
		{ 3175, 1961, 0 },
		{ 3177, 2373, 0 },
		{ 3170, 2632, 0 },
		{ 3162, 3091, 0 },
		{ 3140, 2132, 0 },
		{ 3125, 2938, 0 },
		{ 3173, 3231, 0 },
		{ 2673, 3163, 0 },
		{ 2644, 5006, 0 },
		{ 2645, 5009, 0 },
		{ 3169, 2921, 0 },
		{ 2673, 3165, 0 },
		{ 3162, 3098, 0 },
		{ 3140, 2148, 0 },
		{ 3170, 2638, 0 },
		{ 3179, 2224, 0 },
		{ 3140, 2136, 0 },
		{ 3062, 2692, 0 },
		{ 2654, 5020, 0 },
		{ 3177, 2267, 0 },
		{ 3179, 2229, 0 },
		{ 3164, 2563, 0 },
		{ 3174, 2823, 0 },
		{ 3162, 2984, 0 },
		{ 3180, 4847, 0 },
		{ 3173, 3183, 0 },
		{ 3177, 2383, 0 },
		{ 3121, 2466, 0 },
		{ 3162, 2989, 0 },
		{ 3121, 2468, 0 },
		{ 2922, 2314, 0 },
		{ 3167, 1489, 0 },
		{ 2673, 3156, 0 },
		{ 3173, 3194, 0 },
		{ 3159, 2772, 0 },
		{ 3159, 2774, 0 },
		{ 2672, 5002, 0 },
		{ 3173, 3199, 0 },
		{ 3180, 4709, 0 },
		{ 3175, 2006, 0 },
		{ 3177, 2388, 0 },
		{ 3062, 2849, 0 },
		{ 2678, 4967, 0 },
		{ 3121, 2476, 0 },
		{ 3164, 2277, 0 },
		{ 2922, 2323, 0 },
		{ 3173, 3209, 0 },
		{ 3062, 2853, 0 },
		{ 3173, 3212, 0 },
		{ 3180, 4841, 0 },
		{ 3125, 5101, 424 },
		{ 3175, 2017, 0 },
		{ 3179, 2235, 0 },
		{ 3180, 4853, 0 },
		{ 3180, 4855, 0 },
		{ 3175, 1669, 0 },
		{ 3179, 2239, 0 },
		{ 3008, 2944, 0 },
		{ 3062, 2866, 0 },
		{ 2673, 3164, 0 },
		{ 3162, 3015, 0 },
		{ 3162, 3017, 0 },
		{ 3180, 4639, 0 },
		{ 0, 3155, 0 },
		{ 3125, 5134, 410 },
		{ 3173, 3232, 0 },
		{ 3175, 1670, 0 },
		{ 2922, 2333, 0 },
		{ 3177, 2271, 0 },
		{ 2964, 2929, 0 },
		{ 3177, 2403, 0 },
		{ 3162, 3023, 0 },
		{ 3175, 1673, 0 },
		{ 3140, 2179, 0 },
		{ 3140, 2180, 0 },
		{ 3125, 5171, 404 },
		{ 3177, 2412, 0 },
		{ 3140, 2181, 0 },
		{ 3125, 5034, 416 },
		{ 3125, 5037, 417 },
		{ 3140, 2183, 0 },
		{ 3062, 2880, 0 },
		{ 3008, 2965, 0 },
		{ 3170, 2620, 0 },
		{ 3062, 2884, 0 },
		{ 2922, 2317, 0 },
		{ 2922, 2318, 0 },
		{ 0, 0, 439 },
		{ 3062, 2888, 0 },
		{ 3175, 1674, 0 },
		{ 2720, 4969, 0 },
		{ 3175, 1728, 0 },
		{ 2922, 2322, 0 },
		{ 2723, 4985, 0 },
		{ 3159, 3125, 0 },
		{ 3179, 2252, 0 },
		{ 3062, 2894, 0 },
		{ 3173, 3198, 0 },
		{ 3162, 3051, 0 },
		{ 3179, 2253, 0 },
		{ 3180, 4703, 0 },
		{ 3180, 4705, 0 },
		{ 3121, 2516, 0 },
		{ 3162, 3054, 0 },
		{ 3062, 2900, 0 },
		{ 3170, 2644, 0 },
		{ 3175, 1732, 0 },
		{ 3175, 1741, 0 },
		{ 3170, 2649, 0 },
		{ 3140, 2195, 0 },
		{ 3140, 2111, 0 },
		{ 3180, 4781, 0 },
		{ 3162, 3065, 0 },
		{ 3177, 2221, 0 },
		{ 3162, 3068, 0 },
		{ 3173, 3221, 0 },
		{ 3177, 2353, 0 },
		{ 3175, 1750, 0 },
		{ 3140, 2203, 0 },
		{ 3180, 4882, 0 },
		{ 3179, 939, 407 },
		{ 3125, 5165, 419 },
		{ 2964, 2938, 0 },
		{ 3179, 2265, 0 },
		{ 3175, 1752, 0 },
		{ 3062, 2747, 0 },
		{ 3169, 2924, 0 },
		{ 3169, 2911, 0 },
		{ 3062, 2748, 0 },
		{ 2757, 4946, 0 },
		{ 3174, 3132, 0 },
		{ 3125, 5049, 415 },
		{ 3179, 2269, 0 },
		{ 2922, 2316, 0 },
		{ 3170, 2663, 0 },
		{ 3175, 1753, 0 },
		{ 3121, 2437, 0 },
		{ 3062, 2813, 0 },
		{ 2765, 5015, 0 },
		{ 3125, 5066, 406 },
		{ 3180, 4713, 0 },
		{ 2767, 5022, 0 },
		{ 2772, 1408, 0 },
		{ 3175, 1789, 0 },
		{ 2770, 5028, 0 },
		{ 2771, 5029, 0 },
		{ 3175, 1796, 0 },
		{ 3172, 2641, 0 },
		{ 3179, 2276, 0 },
		{ 3173, 3185, 0 },
		{ 3162, 3100, 0 },
		{ 3180, 4779, 0 },
		{ 3177, 2369, 0 },
		{ 3140, 2212, 0 },
		{ 3177, 2372, 0 },
		{ 3180, 4812, 0 },
		{ 3125, 5109, 421 },
		{ 3180, 4837, 0 },
		{ 3180, 4839, 0 },
		{ 2772, 1360, 0 },
		{ 3180, 4843, 0 },
		{ 3180, 4845, 0 },
		{ 0, 1393, 0 },
		{ 3062, 2858, 0 },
		{ 3062, 2859, 0 },
		{ 3175, 1802, 0 },
		{ 3179, 2282, 0 },
		{ 3125, 5130, 427 },
		{ 3179, 2283, 0 },
		{ 3180, 4909, 0 },
		{ 3121, 2464, 0 },
		{ 0, 0, 442 },
		{ 0, 0, 441 },
		{ 3125, 5137, 408 },
		{ 3180, 4623, 0 },
		{ 0, 0, 438 },
		{ 0, 0, 437 },
		{ 3180, 4625, 0 },
		{ 3170, 2628, 0 },
		{ 2922, 2332, 0 },
		{ 3177, 2381, 0 },
		{ 3173, 3205, 0 },
		{ 3180, 4635, 0 },
		{ 3125, 5154, 402 },
		{ 2802, 5056, 0 },
		{ 3125, 5161, 429 },
		{ 3125, 5163, 409 },
		{ 3162, 2991, 0 },
		{ 3175, 1805, 0 },
		{ 3179, 2285, 0 },
		{ 3175, 1806, 0 },
		{ 3125, 5173, 422 },
		{ 3125, 2408, 0 },
		{ 3180, 4697, 0 },
		{ 3180, 4699, 0 },
		{ 3180, 4701, 0 },
		{ 3177, 2386, 0 },
		{ 3175, 1807, 0 },
		{ 3125, 5042, 413 },
		{ 3125, 5045, 414 },
		{ 3125, 5047, 418 },
		{ 3179, 2288, 0 },
		{ 3162, 3001, 0 },
		{ 3180, 4711, 0 },
		{ 3179, 2289, 0 },
		{ 3125, 5058, 420 },
		{ 3173, 3222, 0 },
		{ 3175, 1823, 0 },
		{ 3062, 2885, 0 },
		{ 3177, 2392, 0 },
		{ 3121, 2483, 0 },
		{ 3140, 2027, 0 },
		{ 3180, 4775, 0 },
		{ 3125, 5073, 425 },
		{ 3069, 4958, 443 },
		{ 2829, 0, 399 },
		{ 0, 0, 400 },
		{ -2827, 5182, 395 },
		{ -2828, 4992, 0 },
		{ 3125, 4969, 0 },
		{ 3069, 4947, 0 },
		{ 0, 0, 396 },
		{ 3069, 4968, 0 },
		{ -2833, 16, 0 },
		{ -2834, 4996, 0 },
		{ 2837, 0, 397 },
		{ 3069, 4970, 0 },
		{ 3125, 5094, 0 },
		{ 0, 0, 398 },
		{ 3116, 3531, 162 },
		{ 0, 0, 162 },
		{ 0, 0, 163 },
		{ 3140, 2038, 0 },
		{ 3162, 3011, 0 },
		{ 3179, 2296, 0 },
		{ 2846, 5015, 0 },
		{ 3172, 2635, 0 },
		{ 3167, 1566, 0 },
		{ 3121, 2491, 0 },
		{ 3174, 3144, 0 },
		{ 3175, 1826, 0 },
		{ 3062, 2899, 0 },
		{ 3177, 2399, 0 },
		{ 3121, 2495, 0 },
		{ 3140, 2058, 0 },
		{ 3180, 4857, 0 },
		{ 0, 0, 160 },
		{ 2995, 4763, 185 },
		{ 0, 0, 185 },
		{ 3175, 1827, 0 },
		{ 2861, 5041, 0 },
		{ 3162, 2700, 0 },
		{ 3173, 3180, 0 },
		{ 3174, 3133, 0 },
		{ 3169, 2917, 0 },
		{ 2866, 5046, 0 },
		{ 3125, 2518, 0 },
		{ 3162, 3027, 0 },
		{ 3121, 2501, 0 },
		{ 3162, 3029, 0 },
		{ 3179, 2222, 0 },
		{ 3173, 3190, 0 },
		{ 3175, 1828, 0 },
		{ 3062, 2694, 0 },
		{ 3177, 2409, 0 },
		{ 3121, 2505, 0 },
		{ 2877, 5069, 0 },
		{ 3125, 2940, 0 },
		{ 3162, 3037, 0 },
		{ 3008, 2950, 0 },
		{ 3177, 2410, 0 },
		{ 3179, 2225, 0 },
		{ 3162, 3041, 0 },
		{ 2884, 4927, 0 },
		{ 3179, 2152, 0 },
		{ 3162, 3043, 0 },
		{ 3159, 3106, 0 },
		{ 3167, 1596, 0 },
		{ 3174, 3149, 0 },
		{ 3162, 3045, 0 },
		{ 2891, 4951, 0 },
		{ 3172, 2643, 0 },
		{ 3167, 1597, 0 },
		{ 3121, 2512, 0 },
		{ 3174, 3131, 0 },
		{ 3175, 1833, 0 },
		{ 3062, 2768, 0 },
		{ 3177, 2337, 0 },
		{ 3121, 2518, 0 },
		{ 3180, 4785, 0 },
		{ 0, 0, 183 },
		{ 2902, 0, 1 },
		{ -2902, 1273, 274 },
		{ 3162, 2958, 280 },
		{ 0, 0, 280 },
		{ 3140, 2092, 0 },
		{ 3121, 2524, 0 },
		{ 3162, 3063, 0 },
		{ 3159, 3110, 0 },
		{ 3179, 2237, 0 },
		{ 0, 0, 279 },
		{ 2912, 5011, 0 },
		{ 3164, 2156, 0 },
		{ 3173, 3237, 0 },
		{ 2941, 2678, 0 },
		{ 3162, 3067, 0 },
		{ 3008, 2953, 0 },
		{ 3062, 2854, 0 },
		{ 3170, 2646, 0 },
		{ 3162, 3072, 0 },
		{ 2921, 4996, 0 },
		{ 3177, 2230, 0 },
		{ 0, 2319, 0 },
		{ 3175, 1863, 0 },
		{ 3062, 2860, 0 },
		{ 3177, 2349, 0 },
		{ 3121, 2430, 0 },
		{ 3140, 2117, 0 },
		{ 3180, 4641, 0 },
		{ 0, 0, 278 },
		{ 0, 4561, 188 },
		{ 0, 0, 188 },
		{ 3177, 2351, 0 },
		{ 3167, 1487, 0 },
		{ 3121, 2434, 0 },
		{ 3159, 3113, 0 },
		{ 2937, 5032, 0 },
		{ 3174, 2911, 0 },
		{ 3169, 2920, 0 },
		{ 3162, 3086, 0 },
		{ 3174, 3139, 0 },
		{ 0, 2680, 0 },
		{ 3062, 2873, 0 },
		{ 3121, 2435, 0 },
		{ 2964, 2930, 0 },
		{ 3180, 4765, 0 },
		{ 0, 0, 186 },
		{ 2995, 4765, 182 },
		{ 0, 0, 181 },
		{ 0, 0, 182 },
		{ 3175, 1865, 0 },
		{ 2952, 5037, 0 },
		{ 3175, 2105, 0 },
		{ 3169, 2914, 0 },
		{ 3162, 3097, 0 },
		{ 2956, 5060, 0 },
		{ 3125, 2936, 0 },
		{ 3162, 3099, 0 },
		{ 2964, 2926, 0 },
		{ 3062, 2878, 0 },
		{ 3121, 2438, 0 },
		{ 3121, 2439, 0 },
		{ 3062, 2882, 0 },
		{ 3121, 2441, 0 },
		{ 0, 2936, 0 },
		{ 2966, 5068, 0 },
		{ 3177, 2239, 0 },
		{ 3008, 2971, 0 },
		{ 2969, 5083, 0 },
		{ 3162, 2697, 0 },
		{ 3173, 3219, 0 },
		{ 3174, 3147, 0 },
		{ 3169, 2912, 0 },
		{ 2974, 4942, 0 },
		{ 3125, 2526, 0 },
		{ 3162, 2986, 0 },
		{ 3121, 2447, 0 },
		{ 3162, 2988, 0 },
		{ 3179, 2248, 0 },
		{ 3173, 3230, 0 },
		{ 3175, 1892, 0 },
		{ 3062, 2890, 0 },
		{ 3177, 2358, 0 },
		{ 3121, 2452, 0 },
		{ 2985, 4961, 0 },
		{ 3172, 2629, 0 },
		{ 3167, 1490, 0 },
		{ 3121, 2454, 0 },
		{ 3174, 3142, 0 },
		{ 3175, 1894, 0 },
		{ 3062, 2897, 0 },
		{ 3177, 2361, 0 },
		{ 3121, 2457, 0 },
		{ 3180, 4693, 0 },
		{ 0, 0, 175 },
		{ 0, 4621, 174 },
		{ 0, 0, 174 },
		{ 3175, 1895, 0 },
		{ 2999, 4969, 0 },
		{ 3175, 2107, 0 },
		{ 3169, 2919, 0 },
		{ 3162, 3006, 0 },
		{ 3003, 4988, 0 },
		{ 3162, 2685, 0 },
		{ 3121, 2460, 0 },
		{ 3159, 3107, 0 },
		{ 3007, 4986, 0 },
		{ 3177, 2265, 0 },
		{ 0, 2946, 0 },
		{ 3010, 5000, 0 },
		{ 3162, 2693, 0 },
		{ 3173, 3187, 0 },
		{ 3174, 3136, 0 },
		{ 3169, 2910, 0 },
		{ 3015, 5005, 0 },
		{ 3125, 2520, 0 },
		{ 3162, 3014, 0 },
		{ 3121, 2463, 0 },
		{ 3162, 3016, 0 },
		{ 3179, 2255, 0 },
		{ 3173, 3195, 0 },
		{ 3175, 1898, 0 },
		{ 3062, 2729, 0 },
		{ 3177, 2367, 0 },
		{ 3121, 2467, 0 },
		{ 3026, 5023, 0 },
		{ 3172, 2645, 0 },
		{ 3167, 1520, 0 },
		{ 3121, 2469, 0 },
		{ 3174, 3128, 0 },
		{ 3175, 1924, 0 },
		{ 3062, 2749, 0 },
		{ 3177, 2370, 0 },
		{ 3121, 2472, 0 },
		{ 3180, 4851, 0 },
		{ 0, 0, 172 },
		{ 0, 4176, 177 },
		{ 0, 0, 177 },
		{ 0, 0, 178 },
		{ 3121, 2473, 0 },
		{ 3140, 2170, 0 },
		{ 3175, 1927, 0 },
		{ 3162, 3032, 0 },
		{ 3173, 3215, 0 },
		{ 3159, 3116, 0 },
		{ 3046, 5051, 0 },
		{ 3162, 2691, 0 },
		{ 3144, 1445, 0 },
		{ 3173, 3220, 0 },
		{ 3170, 2660, 0 },
		{ 3167, 1523, 0 },
		{ 3173, 3223, 0 },
		{ 3175, 1933, 0 },
		{ 3062, 2848, 0 },
		{ 3177, 2376, 0 },
		{ 3121, 2480, 0 },
		{ 3057, 5063, 0 },
		{ 3172, 2631, 0 },
		{ 3167, 1560, 0 },
		{ 3121, 2482, 0 },
		{ 3174, 3129, 0 },
		{ 3175, 1945, 0 },
		{ 0, 2855, 0 },
		{ 3177, 2379, 0 },
		{ 3121, 2485, 0 },
		{ 3142, 4905, 0 },
		{ 0, 0, 176 },
		{ 3162, 3050, 443 },
		{ 3179, 1467, 26 },
		{ 0, 4961, 443 },
		{ 3078, 0, 443 },
		{ 2277, 2808, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 3121, 2489, 0 },
		{ -3077, 5184, 0 },
		{ 3179, 661, 0 },
		{ 0, 0, 28 },
		{ 3159, 3120, 0 },
		{ 0, 0, 27 },
		{ 0, 0, 22 },
		{ 0, 0, 34 },
		{ 0, 0, 35 },
		{ 3099, 3993, 39 },
		{ 0, 3723, 39 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 3111, 3863, 0 },
		{ 3126, 4522, 0 },
		{ 3116, 3506, 0 },
		{ 0, 0, 37 },
		{ 3120, 3565, 0 },
		{ 3124, 3409, 0 },
		{ 3134, 2043, 0 },
		{ 0, 0, 36 },
		{ 3162, 2973, 50 },
		{ 0, 0, 50 },
		{ 0, 4002, 50 },
		{ 3116, 3511, 50 },
		{ 3162, 3060, 50 },
		{ 0, 0, 58 },
		{ 3162, 3061, 0 },
		{ 3121, 2494, 0 },
		{ 3111, 3874, 0 },
		{ 3120, 3575, 0 },
		{ 3175, 1956, 0 },
		{ 3121, 2496, 0 },
		{ 3159, 3108, 0 },
		{ 3116, 3520, 0 },
		{ 0, 3880, 0 },
		{ 3167, 1594, 0 },
		{ 3177, 2389, 0 },
		{ 0, 0, 49 },
		{ 3120, 3582, 0 },
		{ 0, 3523, 0 },
		{ 3179, 2281, 0 },
		{ 3164, 2583, 0 },
		{ 3124, 3404, 54 },
		{ 0, 3587, 0 },
		{ 0, 2500, 0 },
		{ 3162, 3070, 0 },
		{ 3128, 1779, 0 },
		{ 0, 3408, 51 },
		{ 0, 5, 59 },
		{ 0, 4494, 0 },
		{ 3134, 1835, 0 },
		{ 3179, 1619, 0 },
		{ 3135, 1769, 0 },
		{ 0, 0, 57 },
		{ 3170, 2647, 0 },
		{ 0, 0, 55 },
		{ 0, 0, 56 },
		{ 3071, 1946, 0 },
		{ 3179, 1694, 0 },
		{ 3173, 3202, 0 },
		{ 0, 0, 52 },
		{ 0, 0, 53 },
		{ 3140, 2209, 0 },
		{ 0, 2210, 0 },
		{ 3142, 4899, 0 },
		{ 0, 4900, 0 },
		{ 3162, 3080, 0 },
		{ 0, 1484, 0 },
		{ 3173, 3207, 0 },
		{ 3170, 2653, 0 },
		{ 3167, 1639, 0 },
		{ 3173, 3210, 0 },
		{ 3175, 2011, 0 },
		{ 3177, 2404, 0 },
		{ 3179, 2293, 0 },
		{ 3173, 2209, 0 },
		{ 3162, 3088, 0 },
		{ 3177, 2408, 0 },
		{ 3174, 3135, 0 },
		{ 3173, 3218, 0 },
		{ 3179, 2295, 0 },
		{ 3174, 3137, 0 },
		{ 0, 3114, 0 },
		{ 3162, 2994, 0 },
		{ 3167, 1640, 0 },
		{ 0, 3093, 0 },
		{ 3173, 3225, 0 },
		{ 0, 2579, 0 },
		{ 3179, 2297, 0 },
		{ 3174, 3145, 0 },
		{ 0, 1641, 0 },
		{ 3180, 4903, 0 },
		{ 0, 2918, 0 },
		{ 0, 2665, 0 },
		{ 0, 0, 46 },
		{ 3125, 2935, 0 },
		{ 0, 3233, 0 },
		{ 0, 3150, 0 },
		{ 0, 2019, 0 },
		{ 3180, 4911, 0 },
		{ 0, 2414, 0 },
		{ 0, 0, 47 },
		{ 0, 2300, 0 },
		{ 3142, 4912, 0 },
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
		0
	};
	yybackup = backup;
}
