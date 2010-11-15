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

  if( bIsCopyingGPEval )
    if( iCOPY_GP_EVAL_STATUS==EVAL_FTR )
      if( bCOPY_GP_EVAL_GPU ){
	fprintf(fpOutputFile,"k_results[index] =");
      }
      else fprintf(fpOutputFile,"%s",yytext);
 
#line 817 "EaseaLex.cpp"
		}
		break;
	case 53:
		{
#line 573 "EaseaLex.l"

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
 
#line 835 "EaseaLex.cpp"
		}
		break;
	case 54:
		{
#line 587 "EaseaLex.l"

  if( bIsCopyingGPEval )
    fprintf(fpOutputFile,"return fitness = "); 
 
#line 845 "EaseaLex.cpp"
		}
		break;
	case 55:
		{
#line 594 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  yyreset();
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Evaluation population in a single function!!.\n");
  lineCounter = 1;
  BEGIN COPY_INSTEAD_EVAL;
 
#line 859 "EaseaLex.cpp"
		}
		break;
	case 56:
		{
#line 603 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting at the end of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bEndGeneration = true;
  bBeginGeneration = false;
  BEGIN COPY_END_GENERATION_FUNCTION;
 
#line 873 "EaseaLex.cpp"
		}
		break;
	case 57:
		{
#line 612 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Bound Checking function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_BOUND_CHECKING_FUNCTION;
 
#line 885 "EaseaLex.cpp"
		}
		break;
	case 58:
		{
#line 619 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 
#line 897 "EaseaLex.cpp"
		}
		break;
	case 59:
		{
#line 626 "EaseaLex.l"

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
 
#line 926 "EaseaLex.cpp"
		}
		break;
	case 60:
		{
#line 649 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 
#line 943 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 661 "EaseaLex.l"

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
 
#line 969 "EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 682 "EaseaLex.l"

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
  
 
#line 990 "EaseaLex.cpp"
		}
		break;
#line 700 "EaseaLex.l"
  
#line 714 "EaseaLex.l"
      
#line 997 "EaseaLex.cpp"
	case 63:
		{
#line 722 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 1010 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 731 "EaseaLex.l"

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
 
#line 1033 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 748 "EaseaLex.l"

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
 
#line 1056 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 766 "EaseaLex.l"

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
 
#line 1088 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 793 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  fprintf (fpOutputFile,"// Memberwise serialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->serializeIndividual(fpOutputFile, "this");
  //fprintf(fpOutputFile,"\tEASEA_Line << endl;\n");
 
#line 1102 "EaseaLex.cpp"
		}
		break;
	case 68:
		{
#line 802 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome deserializer.\n");
  fprintf (fpOutputFile,"// Memberwise deserialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->deserializeIndividual(fpOutputFile, "this");
 
#line 1115 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 810 "EaseaLex.l"

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
 
#line 1136 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 826 "EaseaLex.l"

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
 
#line 1158 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 843 "EaseaLex.l"
       
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
 
#line 1186 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 865 "EaseaLex.l"

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
 
#line 1208 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 881 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 1223 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 890 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 1235 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 898 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 1247 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 905 "EaseaLex.l"

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
 
#line 1278 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 930 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 1291 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 937 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 1305 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 946 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISER;   
 
#line 1317 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 953 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 1330 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 961 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 1342 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 967 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 1354 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 973 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 1366 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 979 "EaseaLex.l"
      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 1379 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 986 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 1392 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 993 "EaseaLex.l"
 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 1406 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 1002 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 1417 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 1007 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 1431 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 1016 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 1445 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 1025 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 1459 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 1035 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 1472 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 1043 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 1481 "EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 1047 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 1490 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 1051 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 1499 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 1055 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 1508 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 1059 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 1518 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 1064 "EaseaLex.l"

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

#line 1537 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 1077 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 1544 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 1078 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 1551 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 1079 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 1558 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 1080 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 1565 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 1081 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 1572 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 1082 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1579 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 1083 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1586 "EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 1084 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1593 "EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 1085 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1600 "EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 1086 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1607 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 1087 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",nELITE);
  ////DEBUG_PRT_PRT("elitism is %d, elite size is %d",bELITISM, nELITE);
 
#line 1617 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 1092 "EaseaLex.l"

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
 
#line 1636 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 1105 "EaseaLex.l"

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
 
#line 1655 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 1118 "EaseaLex.l"

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
 
#line 1674 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 1131 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1684 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 1135 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1691 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 1136 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1698 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 1137 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1705 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 1138 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1712 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 1139 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1719 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 1140 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1726 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 1141 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1733 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 1142 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1740 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 1143 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1747 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1144 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1754 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1146 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1761 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1147 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1768 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1149 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bREMOTE_ISLAND_MODEL);
#line 1775 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1150 "EaseaLex.l"
if(strlen(sIP_FILE)>0)fprintf(fpOutputFile,"%s",sIP_FILE); else fprintf(fpOutputFile,"NULL");
#line 1782 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1151 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMIGRATION_PROBABILITY);
#line 1789 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1153 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1796 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1154 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1803 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1155 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1810 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1156 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1817 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1157 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1824 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1159 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 1831 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1160 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 1838 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1162 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1852 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1170 "EaseaLex.l"

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
 
#line 1872 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1184 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1886 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1192 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1900 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1201 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1914 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1210 "EaseaLex.l"

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

#line 1977 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1267 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded without warnings.\n");
  printf ("You can now type \"make\" to compile your project.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 1994 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1279 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2001 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1285 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2013 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1291 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2026 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1298 "EaseaLex.l"

#line 2033 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1299 "EaseaLex.l"
lineCounter++;
#line 2040 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1301 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2052 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1307 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2065 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1315 "EaseaLex.l"

#line 2072 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1316 "EaseaLex.l"

  lineCounter++;
 
#line 2081 "EaseaLex.cpp"
		}
		break;
#line 1319 "EaseaLex.l"
               
#line 2086 "EaseaLex.cpp"
	case 151:
		{
#line 1320 "EaseaLex.l"

  fprintf (fpOutputFile,"// User CUDA\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2096 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1326 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user CUDA were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2109 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1334 "EaseaLex.l"

#line 2116 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1335 "EaseaLex.l"

  lineCounter++;
 
#line 2125 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1339 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2137 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1345 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2151 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1353 "EaseaLex.l"

#line 2158 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1354 "EaseaLex.l"

  lineCounter++;
 
#line 2167 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1358 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
      
  BEGIN COPY;
 
#line 2181 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1366 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2196 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1375 "EaseaLex.l"

#line 2203 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1376 "EaseaLex.l"
lineCounter++;
#line 2210 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1381 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2224 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1390 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2238 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1398 "EaseaLex.l"

#line 2245 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1399 "EaseaLex.l"
lineCounter++;
#line 2252 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1402 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2268 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1413 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2284 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1423 "EaseaLex.l"

#line 2291 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1426 "EaseaLex.l"

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
 
#line 2309 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1439 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2326 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1451 "EaseaLex.l"

#line 2333 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1452 "EaseaLex.l"
lineCounter++;
#line 2340 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1454 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2356 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1466 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2372 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1476 "EaseaLex.l"
lineCounter++;
#line 2379 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1477 "EaseaLex.l"

#line 2386 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1481 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2401 "EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 1491 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2416 "EaseaLex.cpp"
		}
		break;
	case 180:
		{
#line 1500 "EaseaLex.l"

#line 2423 "EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 1503 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2436 "EaseaLex.cpp"
		}
		break;
	case 182:
		{
#line 1510 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2450 "EaseaLex.cpp"
		}
		break;
	case 183:
		{
#line 1518 "EaseaLex.l"

#line 2457 "EaseaLex.cpp"
		}
		break;
	case 184:
		{
#line 1522 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2465 "EaseaLex.cpp"
		}
		break;
	case 185:
		{
#line 1524 "EaseaLex.l"

#line 2472 "EaseaLex.cpp"
		}
		break;
	case 186:
		{
#line 1530 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2479 "EaseaLex.cpp"
		}
		break;
	case 187:
		{
#line 1531 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2486 "EaseaLex.cpp"
		}
		break;
	case 188:
	case 189:
		{
#line 1534 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2497 "EaseaLex.cpp"
		}
		break;
	case 190:
	case 191:
		{
#line 1539 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2506 "EaseaLex.cpp"
		}
		break;
	case 192:
	case 193:
		{
#line 1542 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2515 "EaseaLex.cpp"
		}
		break;
	case 194:
	case 195:
		{
#line 1545 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2532 "EaseaLex.cpp"
		}
		break;
	case 196:
	case 197:
		{
#line 1556 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2546 "EaseaLex.cpp"
		}
		break;
	case 198:
	case 199:
		{
#line 1564 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2555 "EaseaLex.cpp"
		}
		break;
	case 200:
	case 201:
		{
#line 1567 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2564 "EaseaLex.cpp"
		}
		break;
	case 202:
	case 203:
		{
#line 1570 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2573 "EaseaLex.cpp"
		}
		break;
	case 204:
	case 205:
		{
#line 1573 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2582 "EaseaLex.cpp"
		}
		break;
	case 206:
	case 207:
		{
#line 1576 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2591 "EaseaLex.cpp"
		}
		break;
	case 208:
	case 209:
		{
#line 1580 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2603 "EaseaLex.cpp"
		}
		break;
	case 210:
		{
#line 1586 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2610 "EaseaLex.cpp"
		}
		break;
	case 211:
		{
#line 1587 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2617 "EaseaLex.cpp"
		}
		break;
	case 212:
		{
#line 1588 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2624 "EaseaLex.cpp"
		}
		break;
	case 213:
		{
#line 1589 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2634 "EaseaLex.cpp"
		}
		break;
	case 214:
		{
#line 1594 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2641 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1595 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2648 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1596 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2655 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1597 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2662 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1598 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2669 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1599 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2676 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1600 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2683 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1601 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2690 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1602 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2698 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1604 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2706 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1606 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2714 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1608 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2724 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1612 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2731 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1613 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2738 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1614 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2749 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1619 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2756 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1620 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2765 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1623 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2777 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1629 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2786 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1632 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2798 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1638 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2809 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1643 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2825 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1653 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2832 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1656 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2841 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1659 "EaseaLex.l"
BEGIN COPY;
#line 2848 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1661 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2855 "EaseaLex.cpp"
		}
		break;
	case 240:
	case 241:
	case 242:
		{
#line 1664 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2868 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1669 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2879 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1674 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 2888 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1683 "EaseaLex.l"
;
#line 2895 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1684 "EaseaLex.l"
;
#line 2902 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1685 "EaseaLex.l"
;
#line 2909 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1686 "EaseaLex.l"
;
#line 2916 "EaseaLex.cpp"
		}
		break;
	case 249:
		{
#line 1689 "EaseaLex.l"
 /* do nothing */ 
#line 2923 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1690 "EaseaLex.l"
 /*return '\n';*/ 
#line 2930 "EaseaLex.cpp"
		}
		break;
	case 251:
		{
#line 1691 "EaseaLex.l"
 /*return '\n';*/ 
#line 2937 "EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1694 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 2946 "EaseaLex.cpp"
		}
		break;
	case 253:
		{
#line 1697 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 2956 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1701 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
//  printf("match gpnode\n");
  return GPNODE;
 
#line 2968 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1708 "EaseaLex.l"
return STATIC;
#line 2975 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1709 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 2982 "EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1710 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 2989 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1711 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 2996 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1712 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 3003 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1713 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 3010 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1715 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 3017 "EaseaLex.cpp"
		}
		break;
#line 1716 "EaseaLex.l"
  
#line 3022 "EaseaLex.cpp"
	case 262:
		{
#line 1717 "EaseaLex.l"
return GENOME; 
#line 3027 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1719 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 3037 "EaseaLex.cpp"
		}
		break;
	case 264:
	case 265:
	case 266:
		{
#line 1726 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 3046 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1727 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 3053 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1730 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 3061 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1732 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 3068 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1738 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 3080 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1744 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 3093 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1751 "EaseaLex.l"

#line 3100 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1753 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3111 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1764 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3126 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1774 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3137 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1780 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3146 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1784 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3161 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1797 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3173 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1803 "EaseaLex.l"

#line 3180 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1804 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3193 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1811 "EaseaLex.l"

#line 3200 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1812 "EaseaLex.l"
lineCounter++;
#line 3207 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1813 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3220 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1820 "EaseaLex.l"

#line 3227 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1821 "EaseaLex.l"
lineCounter++;
#line 3234 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1823 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3247 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1830 "EaseaLex.l"

#line 3254 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1831 "EaseaLex.l"
lineCounter++;
#line 3261 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1833 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3274 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1840 "EaseaLex.l"

#line 3281 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1841 "EaseaLex.l"
lineCounter++;
#line 3288 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1847 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3295 "EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1848 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3302 "EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1849 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3309 "EaseaLex.cpp"
		}
		break;
	case 295:
		{
#line 1850 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3316 "EaseaLex.cpp"
		}
		break;
	case 296:
		{
#line 1851 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3323 "EaseaLex.cpp"
		}
		break;
	case 297:
		{
#line 1852 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3330 "EaseaLex.cpp"
		}
		break;
	case 298:
		{
#line 1853 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3337 "EaseaLex.cpp"
		}
		break;
	case 299:
		{
#line 1855 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3346 "EaseaLex.cpp"
		}
		break;
	case 300:
		{
#line 1858 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3359 "EaseaLex.cpp"
		}
		break;
	case 301:
	case 302:
		{
#line 1867 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3370 "EaseaLex.cpp"
		}
		break;
	case 303:
	case 304:
		{
#line 1872 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3379 "EaseaLex.cpp"
		}
		break;
	case 305:
	case 306:
		{
#line 1875 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3388 "EaseaLex.cpp"
		}
		break;
	case 307:
	case 308:
		{
#line 1878 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3400 "EaseaLex.cpp"
		}
		break;
	case 309:
	case 310:
		{
#line 1884 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3413 "EaseaLex.cpp"
		}
		break;
	case 311:
	case 312:
		{
#line 1891 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3422 "EaseaLex.cpp"
		}
		break;
	case 313:
	case 314:
		{
#line 1894 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3431 "EaseaLex.cpp"
		}
		break;
	case 315:
	case 316:
		{
#line 1897 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3440 "EaseaLex.cpp"
		}
		break;
	case 317:
	case 318:
		{
#line 1900 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3449 "EaseaLex.cpp"
		}
		break;
	case 319:
	case 320:
		{
#line 1903 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3458 "EaseaLex.cpp"
		}
		break;
	case 321:
		{
#line 1906 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3467 "EaseaLex.cpp"
		}
		break;
	case 322:
		{
#line 1909 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3477 "EaseaLex.cpp"
		}
		break;
	case 323:
		{
#line 1913 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3485 "EaseaLex.cpp"
		}
		break;
	case 324:
		{
#line 1915 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3496 "EaseaLex.cpp"
		}
		break;
	case 325:
		{
#line 1920 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3507 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1925 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3515 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1927 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3523 "EaseaLex.cpp"
		}
		break;
	case 328:
		{
#line 1929 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3531 "EaseaLex.cpp"
		}
		break;
	case 329:
		{
#line 1931 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3539 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1933 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3547 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1935 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3554 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 1936 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3561 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 1937 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3569 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 1939 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3577 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 1941 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3585 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 1943 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3592 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 1944 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3604 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1950 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3613 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 1953 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3623 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 1957 "EaseaLex.l"
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
#line 3640 "EaseaLex.cpp"
		}
		break;
	case 341:
	case 342:
		{
#line 1969 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3650 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 1972 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3657 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 1979 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3664 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 1980 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3671 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 1981 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3678 "EaseaLex.cpp"
		}
		break;
	case 347:
		{
#line 1982 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3685 "EaseaLex.cpp"
		}
		break;
	case 348:
		{
#line 1983 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3692 "EaseaLex.cpp"
		}
		break;
	case 349:
		{
#line 1985 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3701 "EaseaLex.cpp"
		}
		break;
	case 350:
		{
#line 1989 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3714 "EaseaLex.cpp"
		}
		break;
	case 351:
		{
#line 1997 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3727 "EaseaLex.cpp"
		}
		break;
	case 352:
		{
#line 2006 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3740 "EaseaLex.cpp"
		}
		break;
	case 353:
		{
#line 2015 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3755 "EaseaLex.cpp"
		}
		break;
	case 354:
		{
#line 2025 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3762 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 2026 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3769 "EaseaLex.cpp"
		}
		break;
	case 356:
	case 357:
		{
#line 2029 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3780 "EaseaLex.cpp"
		}
		break;
	case 358:
	case 359:
		{
#line 2034 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3789 "EaseaLex.cpp"
		}
		break;
	case 360:
	case 361:
		{
#line 2037 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3798 "EaseaLex.cpp"
		}
		break;
	case 362:
	case 363:
		{
#line 2040 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3811 "EaseaLex.cpp"
		}
		break;
	case 364:
	case 365:
		{
#line 2047 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3824 "EaseaLex.cpp"
		}
		break;
	case 366:
	case 367:
		{
#line 2054 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3833 "EaseaLex.cpp"
		}
		break;
	case 368:
		{
#line 2057 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3840 "EaseaLex.cpp"
		}
		break;
	case 369:
		{
#line 2058 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3847 "EaseaLex.cpp"
		}
		break;
	case 370:
		{
#line 2059 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3854 "EaseaLex.cpp"
		}
		break;
	case 371:
		{
#line 2060 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3864 "EaseaLex.cpp"
		}
		break;
	case 372:
		{
#line 2065 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3871 "EaseaLex.cpp"
		}
		break;
	case 373:
		{
#line 2066 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3878 "EaseaLex.cpp"
		}
		break;
	case 374:
		{
#line 2067 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3885 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2068 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3892 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2069 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3900 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2071 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3908 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2073 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3916 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2075 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3924 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2077 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 3932 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2079 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 3940 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2081 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 3948 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2083 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3955 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2084 "EaseaLex.l"
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
#line 3978 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2101 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 3989 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2106 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 4003 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2114 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 4010 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2120 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 4020 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2124 "EaseaLex.l"

#line 4027 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2127 "EaseaLex.l"
;
#line 4034 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2128 "EaseaLex.l"
;
#line 4041 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2129 "EaseaLex.l"
;
#line 4048 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2130 "EaseaLex.l"
;
#line 4055 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2132 "EaseaLex.l"
 /* do nothing */ 
#line 4062 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2133 "EaseaLex.l"
 /*return '\n';*/ 
#line 4069 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2134 "EaseaLex.l"
 /*return '\n';*/ 
#line 4076 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2136 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 4083 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2137 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 4090 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2138 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 4097 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2139 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4104 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2140 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4111 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2141 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4118 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2142 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4125 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2143 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4132 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2144 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4139 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2146 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4146 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2147 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4153 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2148 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4160 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2149 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4167 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2150 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4174 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2152 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4181 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2153 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4188 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2155 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4199 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2160 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4206 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2162 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4217 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2167 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4224 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2170 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4231 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2171 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4238 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2172 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4245 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2173 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4252 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2174 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4259 "EaseaLex.cpp"
		}
		break;
#line 2175 "EaseaLex.l"
 
#line 4264 "EaseaLex.cpp"
	case 422:
		{
#line 2176 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4269 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2177 "EaseaLex.l"
if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;
#line 4276 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2178 "EaseaLex.l"
if(bVERBOSE) printf("\tMigration Probability...\n"); return MIGRATION_PROBABILITY;
#line 4283 "EaseaLex.cpp"
		}
		break;
#line 2180 "EaseaLex.l"
 
#line 4288 "EaseaLex.cpp"
	case 425:
	case 426:
	case 427:
		{
#line 2184 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4295 "EaseaLex.cpp"
		}
		break;
	case 428:
		{
#line 2185 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4302 "EaseaLex.cpp"
		}
		break;
	case 429:
		{
#line 2188 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4310 "EaseaLex.cpp"
		}
		break;
	case 430:
		{
#line 2191 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4317 "EaseaLex.cpp"
		}
		break;
	case 431:
		{
#line 2193 "EaseaLex.l"

  lineCounter++;

#line 4326 "EaseaLex.cpp"
		}
		break;
	case 432:
		{
#line 2196 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4336 "EaseaLex.cpp"
		}
		break;
	case 433:
		{
#line 2201 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4346 "EaseaLex.cpp"
		}
		break;
	case 434:
		{
#line 2206 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4356 "EaseaLex.cpp"
		}
		break;
	case 435:
		{
#line 2211 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4366 "EaseaLex.cpp"
		}
		break;
	case 436:
		{
#line 2216 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4376 "EaseaLex.cpp"
		}
		break;
	case 437:
		{
#line 2221 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4386 "EaseaLex.cpp"
		}
		break;
	case 438:
		{
#line 2230 "EaseaLex.l"
return  (char)yytext[0];
#line 4393 "EaseaLex.cpp"
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
#line 2232 "EaseaLex.l"


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

#line 4590 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		188,
		-189,
		0,
		190,
		-191,
		0,
		192,
		-193,
		0,
		194,
		-195,
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
		198,
		-199,
		0,
		196,
		-197,
		0,
		-238,
		0,
		-244,
		0,
		364,
		-365,
		0,
		309,
		-310,
		0,
		358,
		-359,
		0,
		360,
		-361,
		0,
		362,
		-363,
		0,
		356,
		-357,
		0,
		305,
		-306,
		0,
		307,
		-308,
		0,
		301,
		-302,
		0,
		313,
		-314,
		0,
		315,
		-316,
		0,
		317,
		-318,
		0,
		319,
		-320,
		0,
		303,
		-304,
		0,
		366,
		-367,
		0,
		311,
		-312,
		0
	};
	yymatch = match;

	yytransitionmax = 5033;
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
		{ 2831, 2833 },
		{ 0, 2283 },
		{ 2380, 2350 },
		{ 2380, 2350 },
		{ 73, 3 },
		{ 74, 3 },
		{ 2261, 45 },
		{ 2262, 45 },
		{ 71, 1 },
		{ 167, 169 },
		{ 1902, 1883 },
		{ 69, 1 },
		{ 0, 89 },
		{ 0, 1841 },
		{ 2227, 2223 },
		{ 0, 2032 },
		{ 3072, 63 },
		{ 1375, 1374 },
		{ 3070, 63 },
		{ 1527, 1510 },
		{ 3121, 3119 },
		{ 2408, 2381 },
		{ 1396, 1395 },
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
		{ 134, 120 },
		{ 0, 2902 },
		{ 134, 120 },
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
		{ 118, 103 },
		{ 1248, 1245 },
		{ 118, 103 },
		{ 1248, 1245 },
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
		{ 132, 117 },
		{ 1251, 1247 },
		{ 132, 117 },
		{ 1251, 1247 },
		{ 1238, 1238 },
		{ 2787, 2772 },
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
		{ 2340, 2309 },
		{ 1253, 1250 },
		{ 2340, 2309 },
		{ 1253, 1250 },
		{ 2201, 2198 },
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
		{ 1819, 1818 },
		{ 1303, 1302 },
		{ 1777, 1776 },
		{ 2768, 2752 },
		{ 2784, 2769 },
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
		{ 3130, 3129 },
		{ 1851, 1828 },
		{ 2893, 2892 },
		{ 2598, 2568 },
		{ 2241, 2240 },
		{ 2246, 2245 },
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
		{ 2542, 2513 },
		{ 1732, 1731 },
		{ 2934, 2933 },
		{ 1705, 1704 },
		{ 0, 1488 },
		{ 2667, 2641 },
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
		{ 184, 175 },
		{ 194, 175 },
		{ 186, 175 },
		{ 1363, 1362 },
		{ 181, 175 },
		{ 2987, 2986 },
		{ 185, 175 },
		{ 2038, 2016 },
		{ 183, 175 },
		{ 1658, 1657 },
		{ 1363, 1362 },
		{ 1316, 1315 },
		{ 192, 175 },
		{ 191, 175 },
		{ 182, 175 },
		{ 190, 175 },
		{ 1658, 1657 },
		{ 187, 175 },
		{ 189, 175 },
		{ 180, 175 },
		{ 3028, 3027 },
		{ 2068, 2047 },
		{ 188, 175 },
		{ 193, 175 },
		{ 2308, 2276 },
		{ 3051, 3050 },
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
		{ 102, 85 },
		{ 2309, 2276 },
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
		{ 3059, 3058 },
		{ 456, 411 },
		{ 461, 411 },
		{ 458, 411 },
		{ 457, 411 },
		{ 460, 411 },
		{ 455, 411 },
		{ 3101, 67 },
		{ 454, 411 },
		{ 2097, 2079 },
		{ 69, 67 },
		{ 103, 85 },
		{ 459, 411 },
		{ 2053, 2029 },
		{ 462, 411 },
		{ 2498, 2470 },
		{ 2111, 2094 },
		{ 1877, 1858 },
		{ 1899, 1880 },
		{ 2848, 2847 },
		{ 453, 411 },
		{ 2309, 2276 },
		{ 1244, 1241 },
		{ 3096, 3095 },
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
		{ 2052, 2029 },
		{ 1790, 1789 },
		{ 3113, 3109 },
		{ 2888, 2887 },
		{ 3133, 3132 },
		{ 3149, 3146 },
		{ 3155, 3152 },
		{ 1880, 1861 },
		{ 2255, 2254 },
		{ 103, 85 },
		{ 1886, 1867 },
		{ 1245, 1241 },
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
		{ 2675, 2649 },
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
		{ 1247, 1244 },
		{ 2527, 2527 },
		{ 2527, 2527 },
		{ 2453, 2453 },
		{ 2453, 2453 },
		{ 1319, 1318 },
		{ 2686, 2661 },
		{ 2690, 2665 },
		{ 2700, 2676 },
		{ 3099, 67 },
		{ 1245, 1241 },
		{ 1250, 1246 },
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
		{ 3097, 67 },
		{ 1906, 1887 },
		{ 2527, 2527 },
		{ 2706, 2682 },
		{ 2453, 2453 },
		{ 2719, 2699 },
		{ 2721, 2701 },
		{ 2736, 2716 },
		{ 2737, 2717 },
		{ 3087, 65 },
		{ 1247, 1244 },
		{ 2339, 2308 },
		{ 69, 65 },
		{ 1358, 1357 },
		{ 1933, 1917 },
		{ 2747, 2727 },
		{ 1935, 1920 },
		{ 2752, 2734 },
		{ 2762, 2745 },
		{ 1937, 1922 },
		{ 2769, 2753 },
		{ 1250, 1246 },
		{ 3100, 67 },
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
		{ 2339, 2308 },
		{ 2342, 2310 },
		{ 2772, 2756 },
		{ 2361, 2361 },
		{ 2361, 2361 },
		{ 1240, 9 },
		{ 2518, 2491 },
		{ 2504, 2504 },
		{ 2504, 2504 },
		{ 69, 9 },
		{ 1987, 1985 },
		{ 120, 104 },
		{ 2790, 2775 },
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
		{ 2804, 2798 },
		{ 2806, 2800 },
		{ 3086, 65 },
		{ 2361, 2361 },
		{ 2812, 2807 },
		{ 1240, 9 },
		{ 3085, 65 },
		{ 2504, 2504 },
		{ 2952, 2952 },
		{ 2952, 2952 },
		{ 2342, 2310 },
		{ 117, 102 },
		{ 2999, 2999 },
		{ 2999, 2999 },
		{ 2818, 2817 },
		{ 2558, 2527 },
		{ 2557, 2527 },
		{ 2483, 2453 },
		{ 2482, 2453 },
		{ 1242, 9 },
		{ 120, 104 },
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
		{ 2952, 2952 },
		{ 2505, 2505 },
		{ 2505, 2505 },
		{ 2520, 2493 },
		{ 2999, 2999 },
		{ 2515, 2515 },
		{ 2515, 2515 },
		{ 2380, 2380 },
		{ 2380, 2380 },
		{ 2535, 2535 },
		{ 2535, 2535 },
		{ 1734, 1733 },
		{ 117, 102 },
		{ 2406, 2406 },
		{ 2406, 2406 },
		{ 2549, 2549 },
		{ 2549, 2549 },
		{ 2569, 2569 },
		{ 2569, 2569 },
		{ 2623, 2623 },
		{ 2623, 2623 },
		{ 2720, 2720 },
		{ 2720, 2720 },
		{ 2851, 2850 },
		{ 2505, 2505 },
		{ 3083, 65 },
		{ 2860, 2859 },
		{ 3084, 65 },
		{ 2515, 2515 },
		{ 2873, 2872 },
		{ 2380, 2380 },
		{ 1756, 1755 },
		{ 2535, 2535 },
		{ 1769, 1768 },
		{ 2427, 2427 },
		{ 2427, 2427 },
		{ 2406, 2406 },
		{ 1304, 1303 },
		{ 2549, 2549 },
		{ 2896, 2895 },
		{ 2569, 2569 },
		{ 1653, 1652 },
		{ 2623, 2623 },
		{ 2463, 2435 },
		{ 2720, 2720 },
		{ 2392, 2361 },
		{ 2884, 2884 },
		{ 2884, 2884 },
		{ 2912, 2912 },
		{ 2912, 2912 },
		{ 1778, 1777 },
		{ 2509, 2509 },
		{ 2509, 2509 },
		{ 2534, 2534 },
		{ 2534, 2534 },
		{ 2109, 2092 },
		{ 2393, 2361 },
		{ 2427, 2427 },
		{ 2385, 2385 },
		{ 2385, 2385 },
		{ 2531, 2504 },
		{ 2110, 2093 },
		{ 2473, 2473 },
		{ 2473, 2473 },
		{ 2248, 2248 },
		{ 2248, 2248 },
		{ 2502, 2502 },
		{ 2502, 2502 },
		{ 2924, 2923 },
		{ 2884, 2884 },
		{ 1381, 1380 },
		{ 2912, 2912 },
		{ 2530, 2530 },
		{ 2530, 2530 },
		{ 2509, 2509 },
		{ 2951, 2950 },
		{ 2534, 2534 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2966, 2966 },
		{ 2966, 2966 },
		{ 2385, 2385 },
		{ 2523, 2523 },
		{ 2523, 2523 },
		{ 2953, 2952 },
		{ 2473, 2473 },
		{ 1793, 1792 },
		{ 2248, 2248 },
		{ 3000, 2999 },
		{ 2502, 2502 },
		{ 2128, 2115 },
		{ 3007, 3007 },
		{ 3007, 3007 },
		{ 2981, 2980 },
		{ 2141, 2126 },
		{ 2530, 2530 },
		{ 2990, 2989 },
		{ 2629, 2629 },
		{ 2629, 2629 },
		{ 2544, 2515 },
		{ 2315, 2315 },
		{ 2998, 2997 },
		{ 2966, 2966 },
		{ 2678, 2678 },
		{ 2678, 2678 },
		{ 2523, 2523 },
		{ 2476, 2447 },
		{ 2545, 2515 },
		{ 2532, 2505 },
		{ 2430, 2430 },
		{ 2430, 2430 },
		{ 2723, 2723 },
		{ 2723, 2723 },
		{ 2407, 2380 },
		{ 3007, 3007 },
		{ 2566, 2535 },
		{ 2913, 2912 },
		{ 2388, 2388 },
		{ 2388, 2388 },
		{ 2435, 2406 },
		{ 2629, 2629 },
		{ 2580, 2549 },
		{ 2477, 2448 },
		{ 2599, 2569 },
		{ 2538, 2509 },
		{ 2649, 2623 },
		{ 2678, 2678 },
		{ 2740, 2720 },
		{ 3022, 3021 },
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 2142, 2127 },
		{ 2430, 2430 },
		{ 3031, 3030 },
		{ 2723, 2723 },
		{ 2654, 2654 },
		{ 2654, 2654 },
		{ 3042, 3041 },
		{ 1392, 1391 },
		{ 2455, 2427 },
		{ 2388, 2388 },
		{ 2536, 2509 },
		{ 2202, 2199 },
		{ 3053, 3052 },
		{ 2217, 2216 },
		{ 2537, 2509 },
		{ 3062, 3061 },
		{ 1270, 1269 },
		{ 2913, 2912 },
		{ 2490, 2460 },
		{ 2585, 2555 },
		{ 2885, 2884 },
		{ 2921, 2921 },
		{ 1706, 1705 },
		{ 2494, 2464 },
		{ 2243, 2242 },
		{ 2565, 2534 },
		{ 2605, 2577 },
		{ 2654, 2654 },
		{ 3109, 3105 },
		{ 2620, 2594 },
		{ 2413, 2385 },
		{ 1708, 1707 },
		{ 2626, 2600 },
		{ 3135, 3134 },
		{ 2501, 2473 },
		{ 3137, 3137 },
		{ 2249, 2248 },
		{ 2436, 2407 },
		{ 2529, 2502 },
		{ 2636, 2610 },
		{ 3162, 3160 },
		{ 2779, 2763 },
		{ 1859, 1834 },
		{ 2048, 2023 },
		{ 2561, 2530 },
		{ 1858, 1834 },
		{ 2047, 2023 },
		{ 1327, 1326 },
		{ 2346, 2315 },
		{ 1750, 1749 },
		{ 2967, 2966 },
		{ 1751, 1750 },
		{ 1334, 1333 },
		{ 2553, 2523 },
		{ 2377, 2377 },
		{ 2377, 2377 },
		{ 2550, 2520 },
		{ 1760, 1759 },
		{ 3137, 3137 },
		{ 2069, 2048 },
		{ 2822, 2821 },
		{ 2088, 2067 },
		{ 3008, 3007 },
		{ 2843, 2842 },
		{ 2090, 2069 },
		{ 2093, 2072 },
		{ 2855, 2854 },
		{ 2567, 2536 },
		{ 2655, 2629 },
		{ 2571, 2540 },
		{ 1335, 1334 },
		{ 2579, 2547 },
		{ 1337, 1336 },
		{ 1616, 1610 },
		{ 2702, 2678 },
		{ 1646, 1645 },
		{ 1786, 1785 },
		{ 2377, 2377 },
		{ 1647, 1646 },
		{ 2907, 2905 },
		{ 2458, 2430 },
		{ 1351, 1350 },
		{ 2743, 2723 },
		{ 2606, 2579 },
		{ 1809, 1808 },
		{ 2928, 2927 },
		{ 1810, 1809 },
		{ 1815, 1814 },
		{ 2416, 2388 },
		{ 1352, 1351 },
		{ 1674, 1673 },
		{ 2640, 2614 },
		{ 1675, 1674 },
		{ 2652, 2626 },
		{ 1681, 1680 },
		{ 1682, 1681 },
		{ 1878, 1859 },
		{ 1280, 1279 },
		{ 2258, 2257 },
		{ 2512, 2484 },
		{ 2922, 2921 },
		{ 3041, 3040 },
		{ 1885, 1866 },
		{ 2516, 2489 },
		{ 2680, 2654 },
		{ 1700, 1699 },
		{ 2707, 2683 },
		{ 2708, 2684 },
		{ 2710, 2687 },
		{ 2711, 2690 },
		{ 1897, 1878 },
		{ 1701, 1700 },
		{ 2521, 2494 },
		{ 1312, 1311 },
		{ 2739, 2719 },
		{ 1259, 1258 },
		{ 1387, 1386 },
		{ 1724, 1723 },
		{ 3125, 3124 },
		{ 3126, 3125 },
		{ 2748, 2728 },
		{ 1725, 1724 },
		{ 1948, 1935 },
		{ 3140, 3137 },
		{ 1967, 1957 },
		{ 1975, 1967 },
		{ 1274, 1273 },
		{ 1610, 1601 },
		{ 3139, 3137 },
		{ 1383, 1382 },
		{ 3138, 3137 },
		{ 2627, 2601 },
		{ 1268, 1267 },
		{ 2845, 2844 },
		{ 2635, 2609 },
		{ 2195, 2188 },
		{ 2471, 2442 },
		{ 2528, 2501 },
		{ 2871, 2870 },
		{ 2651, 2625 },
		{ 2882, 2881 },
		{ 2421, 2393 },
		{ 2198, 2193 },
		{ 2656, 2630 },
		{ 1679, 1678 },
		{ 1389, 1388 },
		{ 2214, 2211 },
		{ 1736, 1735 },
		{ 2687, 2662 },
		{ 2911, 2909 },
		{ 1982, 1977 },
		{ 2691, 2666 },
		{ 2539, 2510 },
		{ 2404, 2377 },
		{ 1302, 1301 },
		{ 1261, 1260 },
		{ 2403, 2403 },
		{ 2403, 2403 },
		{ 2245, 2244 },
		{ 1686, 1685 },
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
		{ 1881, 1862 },
		{ 2751, 2733 },
		{ 2080, 2059 },
		{ 2759, 2742 },
		{ 1771, 1770 },
		{ 2403, 2403 },
		{ 1365, 1364 },
		{ 2591, 2561 },
		{ 2595, 2565 },
		{ 2774, 2758 },
		{ 1889, 1870 },
		{ 1776, 1775 },
		{ 1379, 1378 },
		{ 1296, 1295 },
		{ 3116, 3113 },
		{ 2791, 2776 },
		{ 2792, 2778 },
		{ 1916, 1899 },
		{ 2805, 2799 },
		{ 2613, 2587 },
		{ 3137, 3136 },
		{ 1710, 1709 },
		{ 3145, 3142 },
		{ 2813, 2811 },
		{ 3153, 3150 },
		{ 2816, 2814 },
		{ 1660, 1659 },
		{ 3165, 3164 },
		{ 2360, 2360 },
		{ 2360, 2360 },
		{ 2802, 2802 },
		{ 2802, 2802 },
		{ 2454, 2454 },
		{ 2454, 2454 },
		{ 1816, 1815 },
		{ 2577, 2545 },
		{ 2601, 2571 },
		{ 1391, 1390 },
		{ 2798, 2789 },
		{ 2701, 2677 },
		{ 2470, 2441 },
		{ 2583, 2553 },
		{ 1920, 1905 },
		{ 1377, 1376 },
		{ 2431, 2403 },
		{ 2666, 2640 },
		{ 2624, 2598 },
		{ 2760, 2743 },
		{ 2716, 2695 },
		{ 2717, 2696 },
		{ 2923, 2922 },
		{ 2360, 2360 },
		{ 2594, 2564 },
		{ 2802, 2802 },
		{ 2079, 2058 },
		{ 2454, 2454 },
		{ 2722, 2702 },
		{ 2681, 2655 },
		{ 1755, 1754 },
		{ 1848, 1825 },
		{ 2875, 2874 },
		{ 2881, 2880 },
		{ 2058, 2037 },
		{ 2709, 2686 },
		{ 2499, 2471 },
		{ 1709, 1708 },
		{ 1882, 1863 },
		{ 2898, 2897 },
		{ 2582, 2552 },
		{ 1847, 1825 },
		{ 1884, 1865 },
		{ 2073, 2052 },
		{ 2590, 2560 },
		{ 2441, 2413 },
		{ 2075, 2054 },
		{ 2432, 2403 },
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
		{ 3114, 3110 },
		{ 1694, 1693 },
		{ 1984, 1981 },
		{ 2820, 2819 },
		{ 2559, 2528 },
		{ 2391, 2360 },
		{ 1382, 1381 },
		{ 2807, 2802 },
		{ 1990, 1989 },
		{ 2484, 2454 },
		{ 3136, 3135 },
		{ 1863, 1840 },
		{ 1501, 1481 },
		{ 3142, 3139 },
		{ 1321, 1320 },
		{ 2853, 2852 },
		{ 1444, 1420 },
		{ 2704, 2680 },
		{ 1668, 1667 },
		{ 3164, 3162 },
		{ 2054, 2031 },
		{ 2974, 2974 },
		{ 2974, 2974 },
		{ 2866, 2866 },
		{ 2866, 2866 },
		{ 3015, 3015 },
		{ 3015, 3015 },
		{ 2522, 2522 },
		{ 2522, 2522 },
		{ 2497, 2497 },
		{ 2497, 2497 },
		{ 2600, 2570 },
		{ 2517, 2490 },
		{ 2908, 2906 },
		{ 2602, 2572 },
		{ 2604, 2576 },
		{ 1887, 1868 },
		{ 1796, 1795 },
		{ 1802, 1801 },
		{ 2247, 2246 },
		{ 2927, 2926 },
		{ 1651, 1650 },
		{ 2070, 2049 },
		{ 1374, 1373 },
		{ 2974, 2974 },
		{ 2935, 2934 },
		{ 2866, 2866 },
		{ 2944, 2943 },
		{ 3015, 3015 },
		{ 1521, 1502 },
		{ 2522, 2522 },
		{ 2763, 2746 },
		{ 2497, 2497 },
		{ 2961, 2960 },
		{ 2962, 2961 },
		{ 2964, 2963 },
		{ 115, 100 },
		{ 1917, 1900 },
		{ 2977, 2976 },
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
		{ 3107, 3103 },
		{ 2854, 2853 },
		{ 3110, 3106 },
		{ 1273, 1272 },
		{ 2197, 2192 },
		{ 1879, 1860 },
		{ 3119, 3116 },
		{ 2869, 2868 },
		{ 1991, 1990 },
		{ 1781, 1780 },
		{ 2584, 2554 },
		{ 2975, 2974 },
		{ 2876, 2875 },
		{ 2867, 2866 },
		{ 1322, 1321 },
		{ 3016, 3015 },
		{ 1344, 1343 },
		{ 2552, 2522 },
		{ 2210, 2208 },
		{ 2524, 2497 },
		{ 1791, 1790 },
		{ 1693, 1692 },
		{ 2894, 2893 },
		{ 1743, 1742 },
		{ 2732, 2712 },
		{ 2899, 2898 },
		{ 2056, 2034 },
		{ 2057, 2036 },
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
		{ 1931, 1915 },
		{ 2338, 2338 },
		{ 2219, 2218 },
		{ 2757, 2757 },
		{ 2526, 2499 },
		{ 2891, 2891 },
		{ 2096, 2077 },
		{ 3026, 3026 },
		{ 3117, 3114 },
		{ 1314, 1314 },
		{ 1986, 1984 },
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
		{ 2657, 2631 },
		{ 2253, 2252 },
		{ 1804, 1803 },
		{ 2124, 2108 },
		{ 2461, 2433 },
		{ 3152, 3149 },
		{ 2203, 2200 },
		{ 1520, 1501 },
		{ 1695, 1694 },
		{ 1745, 1744 },
		{ 1669, 1668 },
		{ 3046, 3046 },
		{ 3046, 3046 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 1772, 1772 },
		{ 1772, 1772 },
		{ 2969, 2969 },
		{ 2969, 2969 },
		{ 2861, 2861 },
		{ 2861, 2861 },
		{ 1298, 1298 },
		{ 1298, 1298 },
		{ 3003, 3003 },
		{ 3003, 3003 },
		{ 1901, 1882 },
		{ 1309, 1309 },
		{ 1309, 1309 },
		{ 1783, 1783 },
		{ 1783, 1783 },
		{ 2761, 2744 },
		{ 1585, 1559 },
		{ 1644, 1643 },
		{ 1866, 1846 },
		{ 3046, 3046 },
		{ 1361, 1360 },
		{ 3010, 3010 },
		{ 1672, 1671 },
		{ 1772, 1772 },
		{ 1311, 1310 },
		{ 2969, 2969 },
		{ 2429, 2401 },
		{ 2861, 2861 },
		{ 2094, 2073 },
		{ 1298, 1298 },
		{ 2095, 2075 },
		{ 3003, 3003 },
		{ 1488, 1466 },
		{ 2369, 2338 },
		{ 1309, 1309 },
		{ 2773, 2757 },
		{ 1783, 1783 },
		{ 2892, 2891 },
		{ 2207, 2204 },
		{ 3027, 3026 },
		{ 1807, 1806 },
		{ 1315, 1314 },
		{ 2634, 2608 },
		{ 1789, 1788 },
		{ 1748, 1747 },
		{ 2986, 2985 },
		{ 2714, 2693 },
		{ 3058, 3057 },
		{ 1698, 1697 },
		{ 2442, 2414 },
		{ 2797, 2788 },
		{ 2847, 2846 },
		{ 2525, 2498 },
		{ 2586, 2556 },
		{ 2638, 2612 },
		{ 1722, 1721 },
		{ 1266, 1265 },
		{ 2650, 2624 },
		{ 1817, 1816 },
		{ 3123, 3122 },
		{ 2114, 2097 },
		{ 2735, 2715 },
		{ 1785, 1784 },
		{ 3132, 3131 },
		{ 2919, 2918 },
		{ 2242, 2241 },
		{ 2738, 2718 },
		{ 1947, 1934 },
		{ 1656, 1655 },
		{ 2067, 2046 },
		{ 1332, 1331 },
		{ 1959, 1949 },
		{ 2541, 2512 },
		{ 1890, 1871 },
		{ 2172, 2154 },
		{ 3158, 3155 },
		{ 1731, 1730 },
		{ 3050, 3049 },
		{ 1349, 1348 },
		{ 2644, 2644 },
		{ 2644, 2644 },
		{ 2645, 2645 },
		{ 2645, 2645 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 2506, 2506 },
		{ 2506, 2506 },
		{ 2915, 2914 },
		{ 3047, 3046 },
		{ 2942, 2941 },
		{ 3011, 3010 },
		{ 2031, 2222 },
		{ 1773, 1772 },
		{ 1420, 1411 },
		{ 2970, 2969 },
		{ 1840, 1996 },
		{ 2862, 2861 },
		{ 1657, 1656 },
		{ 1299, 1298 },
		{ 2205, 2202 },
		{ 3004, 3003 },
		{ 1757, 1756 },
		{ 2644, 2644 },
		{ 1310, 1309 },
		{ 2645, 2645 },
		{ 1784, 1783 },
		{ 1262, 1262 },
		{ 1393, 1392 },
		{ 2506, 2506 },
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
		{ 2937, 2937 },
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
		{ 0, 1242 },
		{ 1330, 1330 },
		{ 1330, 1330 },
		{ 2632, 2632 },
		{ 2632, 2632 },
		{ 2460, 2432 },
		{ 1867, 1847 },
		{ 2653, 2627 },
		{ 2874, 2873 },
		{ 1794, 1793 },
		{ 3023, 3022 },
		{ 2937, 2937 },
		{ 0, 86 },
		{ 2216, 2214 },
		{ 1921, 1906 },
		{ 1759, 1758 },
		{ 2753, 2735 },
		{ 2756, 2738 },
		{ 3032, 3031 },
		{ 2670, 2644 },
		{ 1871, 1852 },
		{ 2671, 2645 },
		{ 1320, 1319 },
		{ 1263, 1262 },
		{ 1330, 1330 },
		{ 2533, 2506 },
		{ 2632, 2632 },
		{ 1362, 1361 },
		{ 1355, 1354 },
		{ 2897, 2896 },
		{ 2125, 2109 },
		{ 2764, 2747 },
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
		{ 0, 2277 },
		{ 0, 86 },
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
		{ 0, 2277 },
		{ 2035, 2013 },
		{ 1421, 1400 },
		{ 2938, 2937 },
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
		{ 3157, 3157 },
		{ 2596, 2566 },
		{ 1331, 1330 },
		{ 2139, 2123 },
		{ 2658, 2632 },
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
		{ 2956, 2956 },
		{ 2956, 2956 },
		{ 2615, 2615 },
		{ 2615, 2615 },
		{ 2877, 2877 },
		{ 2877, 2877 },
		{ 3054, 3053 },
		{ 2677, 2651 },
		{ 3157, 3157 },
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
		{ 2401, 2369 },
		{ 2693, 2668 },
		{ 2956, 2956 },
		{ 1357, 1356 },
		{ 2615, 2615 },
		{ 2607, 2580 },
		{ 2877, 2877 },
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
		{ 2036, 2013 },
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
		{ 3145, 3145 },
		{ 2448, 2420 },
		{ 1849, 1827 },
		{ 2726, 2706 },
		{ 1654, 1653 },
		{ 1359, 1358 },
		{ 2991, 2990 },
		{ 2852, 2851 },
		{ 2734, 2714 },
		{ 1989, 1987 },
		{ 2865, 2864 },
		{ 3001, 3000 },
		{ 2939, 2938 },
		{ 2646, 2620 },
		{ 1305, 1304 },
		{ 2754, 2736 },
		{ 3014, 3013 },
		{ 2755, 2737 },
		{ 3157, 3154 },
		{ 2973, 2972 },
		{ 2218, 2217 },
		{ 2954, 2953 },
		{ 69, 5 },
		{ 3145, 3145 },
		{ 1779, 1778 },
		{ 2945, 2944 },
		{ 2209, 2207 },
		{ 3159, 3157 },
		{ 2965, 2964 },
		{ 1308, 1307 },
		{ 2750, 2732 },
		{ 2589, 2559 },
		{ 2959, 2958 },
		{ 1372, 1371 },
		{ 1782, 1781 },
		{ 2703, 2679 },
		{ 2905, 2903 },
		{ 3102, 3097 },
		{ 2957, 2956 },
		{ 1860, 1837 },
		{ 2641, 2615 },
		{ 1998, 1995 },
		{ 2878, 2877 },
		{ 1861, 1837 },
		{ 2488, 2458 },
		{ 2713, 2692 },
		{ 1997, 1995 },
		{ 2239, 2238 },
		{ 1842, 1822 },
		{ 2968, 2967 },
		{ 2692, 2667 },
		{ 2614, 2588 },
		{ 2906, 2903 },
		{ 1841, 1822 },
		{ 3009, 3008 },
		{ 1482, 1460 },
		{ 2376, 2346 },
		{ 2880, 2879 },
		{ 2033, 2010 },
		{ 2511, 2483 },
		{ 2917, 2916 },
		{ 2540, 2511 },
		{ 1294, 1293 },
		{ 2032, 2010 },
		{ 2444, 2416 },
		{ 2588, 2558 },
		{ 1768, 1767 },
		{ 3103, 3097 },
		{ 2224, 2221 },
		{ 1850, 1827 },
		{ 1438, 1412 },
		{ 2648, 2622 },
		{ 2958, 2957 },
		{ 2223, 2221 },
		{ 176, 5 },
		{ 2776, 2760 },
		{ 2312, 2282 },
		{ 1264, 1263 },
		{ 177, 5 },
		{ 1642, 1641 },
		{ 1787, 1786 },
		{ 1293, 1292 },
		{ 2457, 2429 },
		{ 1301, 1300 },
		{ 2547, 2517 },
		{ 178, 5 },
		{ 2188, 2175 },
		{ 2659, 2633 },
		{ 2192, 2185 },
		{ 2976, 2975 },
		{ 1742, 1741 },
		{ 2978, 2977 },
		{ 3148, 3145 },
		{ 2664, 2638 },
		{ 1685, 1684 },
		{ 2803, 2797 },
		{ 2554, 2524 },
		{ 2034, 2011 },
		{ 100, 83 },
		{ 1692, 1691 },
		{ 1746, 1745 },
		{ 175, 5 },
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
		{ 3105, 3100 },
		{ 3106, 3102 },
		{ 1716, 1715 },
		{ 2909, 2907 },
		{ 1313, 1312 },
		{ 2742, 2722 },
		{ 1328, 1327 },
		{ 2916, 2915 },
		{ 2744, 2724 },
		{ 1775, 1774 },
		{ 3120, 3117 },
		{ 2618, 2592 },
		{ 2920, 2919 },
		{ 2619, 2593 },
		{ 3129, 3128 },
		{ 2113, 2096 },
		{ 1960, 1950 },
		{ 1670, 1669 },
		{ 1720, 1719 },
		{ 1364, 1363 },
		{ 1976, 1968 },
		{ 2630, 2604 },
		{ 3141, 3138 },
		{ 1977, 1969 },
		{ 1868, 1848 },
		{ 1347, 1346 },
		{ 1638, 1637 },
		{ 3150, 3147 },
		{ 2940, 2939 },
		{ 1728, 1727 },
		{ 1371, 1370 },
		{ 1988, 1986 },
		{ 2639, 2613 },
		{ 2449, 2421 },
		{ 2157, 2143 },
		{ 1846, 1824 },
		{ 1297, 1296 },
		{ 2955, 2954 },
		{ 2628, 2602 },
		{ 1908, 1890 },
		{ 1467, 1445 },
		{ 2078, 2057 },
		{ 2724, 2704 },
		{ 2513, 2485 },
		{ 2050, 2027 },
		{ 2887, 2886 },
		{ 3006, 3005 },
		{ 3147, 3144 },
		{ 3111, 3107 },
		{ 1385, 1384 },
		{ 2910, 2908 },
		{ 130, 115 },
		{ 1813, 1812 },
		{ 2936, 2935 },
		{ 1898, 1879 },
		{ 3045, 3044 },
		{ 2091, 2070 },
		{ 1888, 1869 },
		{ 2374, 2343 },
		{ 3082, 3080 },
		{ 2972, 2971 },
		{ 1792, 1791 },
		{ 2889, 2888 },
		{ 1936, 1921 },
		{ 1318, 1317 },
		{ 1388, 1387 },
		{ 3030, 3029 },
		{ 3143, 3140 },
		{ 3061, 3060 },
		{ 3146, 3143 },
		{ 624, 565 },
		{ 2895, 2894 },
		{ 2758, 2740 },
		{ 2864, 2863 },
		{ 2625, 2599 },
		{ 3013, 3012 },
		{ 3154, 3151 },
		{ 1265, 1264 },
		{ 2941, 2940 },
		{ 2573, 2542 },
		{ 625, 565 },
		{ 2989, 2988 },
		{ 3161, 3159 },
		{ 2140, 2125 },
		{ 2850, 2849 },
		{ 1678, 1677 },
		{ 2031, 2025 },
		{ 2643, 2617 },
		{ 2694, 2669 },
		{ 2647, 2621 },
		{ 2698, 2673 },
		{ 2668, 2642 },
		{ 1370, 1367 },
		{ 1840, 1836 },
		{ 2616, 2590 },
		{ 1420, 1413 },
		{ 626, 565 },
		{ 204, 181 },
		{ 2642, 2616 },
		{ 2699, 2675 },
		{ 202, 181 },
		{ 2555, 2525 },
		{ 203, 181 },
		{ 1360, 1359 },
		{ 1350, 1349 },
		{ 2914, 2913 },
		{ 1733, 1732 },
		{ 1269, 1268 },
		{ 1655, 1654 },
		{ 2240, 2239 },
		{ 1699, 1698 },
		{ 201, 181 },
		{ 1645, 1644 },
		{ 2608, 2582 },
		{ 1949, 1936 },
		{ 2193, 2186 },
		{ 1333, 1332 },
		{ 2863, 2862 },
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
		{ 3124, 3123 },
		{ 2493, 2463 },
		{ 1267, 1266 },
		{ 2727, 2707 },
		{ 2673, 2647 },
		{ 3131, 3130 },
		{ 1380, 1379 },
		{ 2126, 2111 },
		{ 3134, 3133 },
		{ 2127, 2114 },
		{ 1862, 1838 },
		{ 2800, 2791 },
		{ 2587, 2557 },
		{ 2046, 2022 },
		{ 2682, 2656 },
		{ 1749, 1748 },
		{ 3144, 3141 },
		{ 2684, 2658 },
		{ 1707, 1706 },
		{ 1601, 1585 },
		{ 3044, 3043 },
		{ 2464, 2436 },
		{ 2049, 2026 },
		{ 3151, 3148 },
		{ 2971, 2970 },
		{ 3049, 3048 },
		{ 2745, 2725 },
		{ 2817, 2816 },
		{ 3052, 3051 },
		{ 2154, 2140 },
		{ 1818, 1817 },
		{ 3160, 3158 },
		{ 2510, 2482 },
		{ 1985, 1982 },
		{ 1730, 1729 },
		{ 2980, 2979 },
		{ 685, 623 },
		{ 752, 694 },
		{ 867, 814 },
		{ 428, 386 },
		{ 1766, 35 },
		{ 2841, 47 },
		{ 1369, 21 },
		{ 69, 35 },
		{ 69, 47 },
		{ 69, 21 },
		{ 1664, 27 },
		{ 1690, 29 },
		{ 3038, 61 },
		{ 69, 27 },
		{ 69, 29 },
		{ 69, 61 },
		{ 281, 241 },
		{ 1257, 11 },
		{ 751, 694 },
		{ 429, 386 },
		{ 69, 11 },
		{ 1291, 15 },
		{ 686, 623 },
		{ 1714, 31 },
		{ 69, 15 },
		{ 296, 253 },
		{ 69, 31 },
		{ 2948, 57 },
		{ 868, 814 },
		{ 1278, 13 },
		{ 69, 57 },
		{ 1341, 19 },
		{ 69, 13 },
		{ 1740, 33 },
		{ 69, 19 },
		{ 303, 260 },
		{ 69, 33 },
		{ 2060, 2039 },
		{ 309, 265 },
		{ 319, 275 },
		{ 336, 291 },
		{ 339, 294 },
		{ 346, 300 },
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
		{ 1764, 35 },
		{ 2839, 47 },
		{ 1367, 21 },
		{ 649, 585 },
		{ 658, 594 },
		{ 673, 608 },
		{ 1662, 27 },
		{ 1688, 29 },
		{ 3036, 61 },
		{ 674, 609 },
		{ 684, 622 },
		{ 243, 208 },
		{ 704, 641 },
		{ 1255, 11 },
		{ 244, 209 },
		{ 753, 695 },
		{ 765, 706 },
		{ 1289, 15 },
		{ 767, 708 },
		{ 1712, 31 },
		{ 769, 710 },
		{ 774, 715 },
		{ 794, 735 },
		{ 2947, 57 },
		{ 805, 745 },
		{ 1276, 13 },
		{ 809, 749 },
		{ 1339, 19 },
		{ 810, 750 },
		{ 1738, 33 },
		{ 840, 784 },
		{ 859, 806 },
		{ 264, 225 },
		{ 897, 846 },
		{ 907, 856 },
		{ 922, 871 },
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
		{ 69, 43 },
		{ 69, 49 },
		{ 69, 17 },
		{ 69, 25 },
		{ 69, 53 },
		{ 69, 55 },
		{ 434, 390 },
		{ 69, 37 },
		{ 435, 390 },
		{ 433, 390 },
		{ 69, 59 },
		{ 220, 189 },
		{ 2135, 2120 },
		{ 2133, 2119 },
		{ 2194, 2187 },
		{ 218, 189 },
		{ 3095, 3094 },
		{ 468, 418 },
		{ 470, 418 },
		{ 2136, 2120 },
		{ 2134, 2119 },
		{ 820, 760 },
		{ 437, 391 },
		{ 727, 670 },
		{ 728, 671 },
		{ 436, 390 },
		{ 509, 457 },
		{ 510, 457 },
		{ 471, 418 },
		{ 587, 527 },
		{ 221, 189 },
		{ 219, 189 },
		{ 410, 369 },
		{ 469, 418 },
		{ 511, 457 },
		{ 501, 450 },
		{ 2131, 2117 },
		{ 1940, 1926 },
		{ 2043, 2019 },
		{ 1116, 1089 },
		{ 214, 186 },
		{ 1122, 1096 },
		{ 683, 621 },
		{ 213, 186 },
		{ 817, 757 },
		{ 291, 248 },
		{ 2042, 2019 },
		{ 332, 287 },
		{ 522, 466 },
		{ 215, 186 },
		{ 443, 398 },
		{ 2064, 2043 },
		{ 630, 566 },
		{ 359, 313 },
		{ 913, 862 },
		{ 914, 863 },
		{ 629, 566 },
		{ 524, 466 },
		{ 2041, 2019 },
		{ 421, 379 },
		{ 420, 379 },
		{ 207, 183 },
		{ 523, 466 },
		{ 209, 183 },
		{ 628, 566 },
		{ 627, 566 },
		{ 208, 183 },
		{ 1131, 1106 },
		{ 513, 459 },
		{ 516, 461 },
		{ 2065, 2044 },
		{ 300, 257 },
		{ 517, 461 },
		{ 579, 519 },
		{ 268, 228 },
		{ 909, 858 },
		{ 1875, 1856 },
		{ 715, 656 },
		{ 800, 740 },
		{ 2235, 43 },
		{ 2857, 49 },
		{ 1324, 17 },
		{ 1635, 25 },
		{ 2903, 53 },
		{ 2930, 55 },
		{ 1130, 1106 },
		{ 1798, 37 },
		{ 799, 740 },
		{ 514, 459 },
		{ 2995, 59 },
		{ 227, 192 },
		{ 592, 532 },
		{ 258, 220 },
		{ 224, 190 },
		{ 821, 761 },
		{ 593, 532 },
		{ 222, 190 },
		{ 822, 762 },
		{ 529, 470 },
		{ 223, 190 },
		{ 282, 242 },
		{ 1855, 1831 },
		{ 226, 192 },
		{ 612, 550 },
		{ 698, 635 },
		{ 970, 925 },
		{ 594, 532 },
		{ 974, 930 },
		{ 486, 434 },
		{ 826, 766 },
		{ 1279, 1276 },
		{ 568, 508 },
		{ 716, 657 },
		{ 530, 470 },
		{ 3093, 3091 },
		{ 283, 242 },
		{ 1156, 1139 },
		{ 2489, 2459 },
		{ 1285, 1284 },
		{ 718, 659 },
		{ 3104, 3099 },
		{ 1163, 1146 },
		{ 877, 825 },
		{ 1326, 1324 },
		{ 546, 484 },
		{ 1258, 1255 },
		{ 1181, 1170 },
		{ 1182, 1171 },
		{ 1183, 1172 },
		{ 3115, 3112 },
		{ 1063, 1030 },
		{ 801, 741 },
		{ 238, 203 },
		{ 1086, 1054 },
		{ 749, 692 },
		{ 750, 693 },
		{ 423, 381 },
		{ 2842, 2839 },
		{ 559, 499 },
		{ 210, 184 },
		{ 1135, 1113 },
		{ 1140, 1120 },
		{ 386, 345 },
		{ 1149, 1131 },
		{ 211, 184 },
		{ 342, 296 },
		{ 1150, 1133 },
		{ 343, 297 },
		{ 1874, 1855 },
		{ 396, 354 },
		{ 1161, 1144 },
		{ 399, 357 },
		{ 558, 499 },
		{ 837, 780 },
		{ 341, 296 },
		{ 340, 296 },
		{ 838, 781 },
		{ 1174, 1162 },
		{ 1176, 1164 },
		{ 553, 492 },
		{ 848, 794 },
		{ 856, 803 },
		{ 2062, 2041 },
		{ 554, 493 },
		{ 1189, 1178 },
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
		{ 3108, 3104 },
		{ 929, 878 },
		{ 935, 885 },
		{ 940, 890 },
		{ 945, 897 },
		{ 951, 905 },
		{ 958, 911 },
		{ 719, 660 },
		{ 578, 518 },
		{ 3118, 3115 },
		{ 491, 439 },
		{ 729, 672 },
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
		{ 1017, 979 },
		{ 862, 809 },
		{ 863, 810 },
		{ 1026, 990 },
		{ 621, 562 },
		{ 242, 207 },
		{ 1910, 1892 },
		{ 1042, 1007 },
		{ 360, 316 },
		{ 394, 352 },
		{ 1564, 1564 },
		{ 1567, 1567 },
		{ 1570, 1570 },
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
		{ 679, 614 },
		{ 1606, 1606 },
		{ 1564, 1564 },
		{ 1567, 1567 },
		{ 1570, 1570 },
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
		{ 1606, 1606 },
		{ 1631, 1631 },
		{ 824, 764 },
		{ 606, 545 },
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
		{ 733, 675 },
		{ 736, 680 },
		{ 737, 681 },
		{ 1437, 1564 },
		{ 1437, 1567 },
		{ 1437, 1570 },
		{ 1437, 1573 },
		{ 1437, 1576 },
		{ 1437, 1579 },
		{ 381, 339 },
		{ 1631, 1631 },
		{ 747, 690 },
		{ 217, 188 },
		{ 327, 282 },
		{ 257, 219 },
		{ 1084, 1051 },
		{ 644, 580 },
		{ 534, 474 },
		{ 1437, 1597 },
		{ 648, 584 },
		{ 918, 867 },
		{ 921, 870 },
		{ 365, 321 },
		{ 650, 586 },
		{ 1104, 1076 },
		{ 1437, 1606 },
		{ 1107, 1079 },
		{ 3090, 3086 },
		{ 1108, 1080 },
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
		{ 335, 290 },
		{ 1437, 1631 },
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
		{ 611, 549 },
		{ 249, 213 },
		{ 206, 182 },
		{ 610, 549 },
		{ 248, 213 },
		{ 388, 346 },
		{ 387, 346 },
		{ 285, 243 },
		{ 284, 243 },
		{ 666, 601 },
		{ 667, 601 },
		{ 505, 455 },
		{ 205, 182 },
		{ 536, 476 },
		{ 537, 476 },
		{ 538, 477 },
		{ 539, 477 },
		{ 252, 216 },
		{ 290, 247 },
		{ 262, 223 },
		{ 253, 216 },
		{ 762, 704 },
		{ 506, 455 },
		{ 507, 456 },
		{ 307, 264 },
		{ 261, 223 },
		{ 582, 522 },
		{ 899, 848 },
		{ 289, 247 },
		{ 254, 216 },
		{ 445, 400 },
		{ 763, 704 },
		{ 308, 264 },
		{ 508, 456 },
		{ 1133, 1111 },
		{ 1134, 1112 },
		{ 2122, 2105 },
		{ 3112, 3108 },
		{ 484, 432 },
		{ 1136, 1114 },
		{ 2061, 2040 },
		{ 273, 233 },
		{ 389, 347 },
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
		{ 267, 227 },
		{ 1905, 1886 },
		{ 250, 214 },
		{ 603, 542 },
		{ 487, 435 },
		{ 842, 788 },
		{ 586, 526 },
		{ 266, 227 },
		{ 759, 701 },
		{ 1196, 1187 },
		{ 1207, 1198 },
		{ 2040, 2018 },
		{ 1208, 1199 },
		{ 2044, 2020 },
		{ 1105, 1077 },
		{ 1213, 1204 },
		{ 1854, 1830 },
		{ 1009, 969 },
		{ 1856, 1832 },
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
		{ 731, 674 },
		{ 2116, 2099 },
		{ 1219, 1213 },
		{ 451, 409 },
		{ 694, 631 },
		{ 1129, 1105 },
		{ 732, 674 },
		{ 1072, 1038 },
		{ 1224, 1222 },
		{ 1132, 1110 },
		{ 1073, 1039 },
		{ 697, 634 },
		{ 1236, 1235 },
		{ 575, 515 },
		{ 426, 384 },
		{ 1137, 1117 },
		{ 620, 559 },
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
		{ 322, 278 },
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
		{ 573, 513 },
		{ 1064, 1031 },
		{ 1425, 1425 },
		{ 845, 791 },
		{ 734, 676 },
		{ 1004, 964 },
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
		{ 1187, 1176 },
		{ 1007, 967 },
		{ 321, 277 },
		{ 926, 875 },
		{ 1053, 1020 },
		{ 1099, 1071 },
		{ 1011, 971 },
		{ 788, 729 },
		{ 789, 730 },
		{ 1425, 1425 },
		{ 1198, 1189 },
		{ 2117, 2100 },
		{ 1199, 1190 },
		{ 2119, 2102 },
		{ 1283, 1283 },
		{ 2120, 2103 },
		{ 855, 802 },
		{ 1204, 1195 },
		{ 310, 266 },
		{ 857, 804 },
		{ 493, 441 },
		{ 564, 504 },
		{ 1111, 1084 },
		{ 631, 567 },
		{ 495, 443 },
		{ 384, 342 },
		{ 634, 570 },
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
		{ 1437, 1425 },
		{ 1284, 1283 },
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
		{ 1157, 1140 },
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
		{ 866, 813 },
		{ 1069, 1035 },
		{ 1070, 1036 },
		{ 406, 364 },
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
		{ 952, 906 },
		{ 689, 626 },
		{ 1216, 1209 },
		{ 483, 431 },
		{ 953, 906 },
		{ 777, 718 },
		{ 383, 341 },
		{ 919, 868 },
		{ 1125, 1100 },
		{ 743, 686 },
		{ 944, 896 },
		{ 425, 383 },
		{ 286, 244 },
		{ 1106, 1078 },
		{ 730, 673 },
		{ 241, 206 },
		{ 1030, 995 },
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
		{ 1580, 1580 },
		{ 1580, 1580 },
		{ 1528, 1528 },
		{ 1528, 1528 },
		{ 1607, 1607 },
		{ 1607, 1607 },
		{ 682, 620 },
		{ 1973, 1973 },
		{ 548, 486 },
		{ 2183, 2183 },
		{ 569, 509 },
		{ 1632, 1632 },
		{ 251, 215 },
		{ 1978, 1978 },
		{ 277, 237 },
		{ 2129, 2129 },
		{ 852, 799 },
		{ 2189, 2189 },
		{ 1091, 1059 },
		{ 1577, 1577 },
		{ 551, 490 },
		{ 1568, 1568 },
		{ 318, 274 },
		{ 1580, 1580 },
		{ 295, 252 },
		{ 1528, 1528 },
		{ 1095, 1064 },
		{ 1607, 1607 },
		{ 1571, 1571 },
		{ 1571, 1571 },
		{ 1992, 1992 },
		{ 1992, 1992 },
		{ 1598, 1598 },
		{ 1598, 1598 },
		{ 1565, 1565 },
		{ 1565, 1565 },
		{ 1974, 1973 },
		{ 1096, 1065 },
		{ 2184, 2183 },
		{ 313, 269 },
		{ 1633, 1632 },
		{ 577, 517 },
		{ 1979, 1978 },
		{ 473, 420 },
		{ 2130, 2129 },
		{ 2237, 2235 },
		{ 2190, 2189 },
		{ 695, 632 },
		{ 1578, 1577 },
		{ 861, 808 },
		{ 1569, 1568 },
		{ 1571, 1571 },
		{ 1581, 1580 },
		{ 1992, 1992 },
		{ 1529, 1528 },
		{ 1598, 1598 },
		{ 1608, 1607 },
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
		{ 2160, 2160 },
		{ 2160, 2160 },
		{ 2162, 2162 },
		{ 2162, 2162 },
		{ 2164, 2164 },
		{ 2164, 2164 },
		{ 2166, 2166 },
		{ 2166, 2166 },
		{ 2168, 2168 },
		{ 2168, 2168 },
		{ 1938, 1938 },
		{ 1938, 1938 },
		{ 1572, 1571 },
		{ 2212, 2212 },
		{ 1993, 1992 },
		{ 1961, 1961 },
		{ 1599, 1598 },
		{ 1963, 1963 },
		{ 1566, 1565 },
		{ 1574, 1574 },
		{ 999, 959 },
		{ 2158, 2158 },
		{ 661, 596 },
		{ 2160, 2160 },
		{ 784, 725 },
		{ 2162, 2162 },
		{ 766, 707 },
		{ 2164, 2164 },
		{ 412, 371 },
		{ 2166, 2166 },
		{ 930, 879 },
		{ 2168, 2168 },
		{ 982, 941 },
		{ 1938, 1938 },
		{ 659, 595 },
		{ 660, 595 },
		{ 931, 880 },
		{ 1286, 1285 },
		{ 910, 859 },
		{ 813, 753 },
		{ 1068, 1034 },
		{ 1144, 1125 },
		{ 2213, 2212 },
		{ 746, 689 },
		{ 1962, 1961 },
		{ 776, 717 },
		{ 1964, 1963 },
		{ 1120, 1094 },
		{ 1575, 1574 },
		{ 964, 917 },
		{ 2159, 2158 },
		{ 1177, 1165 },
		{ 2161, 2160 },
		{ 966, 919 },
		{ 2163, 2162 },
		{ 1179, 1168 },
		{ 2165, 2164 },
		{ 197, 178 },
		{ 2167, 2166 },
		{ 3122, 3120 },
		{ 2169, 2168 },
		{ 748, 691 },
		{ 1939, 1938 },
		{ 1348, 1347 },
		{ 1559, 1541 },
		{ 1966, 1955 },
		{ 1643, 1642 },
		{ 1075, 1041 },
		{ 1914, 1897 },
		{ 379, 336 },
		{ 1184, 1173 },
		{ 2107, 2090 },
		{ 1747, 1746 },
		{ 803, 743 },
		{ 1697, 1696 },
		{ 1186, 1175 },
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
		{ 304, 261 },
		{ 617, 556 },
		{ 2051, 2028 },
		{ 225, 191 },
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
		{ 2729, 2729 },
		{ 2729, 2729 },
		{ 1686, 1686 },
		{ 1686, 1686 },
		{ 315, 271 },
		{ 2659, 2659 },
		{ 890, 839 },
		{ 2770, 2770 },
		{ 780, 721 },
		{ 2771, 2771 },
		{ 781, 722 },
		{ 2634, 2634 },
		{ 956, 909 },
		{ 1322, 1322 },
		{ 957, 910 },
		{ 3034, 3034 },
		{ 1025, 989 },
		{ 2663, 2663 },
		{ 1767, 1764 },
		{ 2664, 2664 },
		{ 430, 387 },
		{ 2855, 2855 },
		{ 371, 328 },
		{ 2729, 2729 },
		{ 316, 272 },
		{ 1686, 1686 },
		{ 690, 627 },
		{ 2779, 2779 },
		{ 2779, 2779 },
		{ 2521, 2521 },
		{ 2521, 2521 },
		{ 2685, 2659 },
		{ 609, 548 },
		{ 2785, 2770 },
		{ 1103, 1075 },
		{ 2786, 2771 },
		{ 357, 311 },
		{ 2660, 2634 },
		{ 841, 785 },
		{ 1323, 1322 },
		{ 965, 918 },
		{ 3035, 3034 },
		{ 900, 849 },
		{ 2688, 2663 },
		{ 552, 491 },
		{ 2689, 2664 },
		{ 492, 440 },
		{ 2856, 2855 },
		{ 1037, 1002 },
		{ 2749, 2729 },
		{ 2779, 2779 },
		{ 1687, 1686 },
		{ 2521, 2521 },
		{ 903, 852 },
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
		{ 1660, 1660 },
		{ 1660, 1660 },
		{ 2793, 2779 },
		{ 2784, 2784 },
		{ 2551, 2521 },
		{ 2787, 2787 },
		{ 329, 284 },
		{ 2465, 2465 },
		{ 791, 732 },
		{ 1365, 1365 },
		{ 792, 733 },
		{ 1337, 1337 },
		{ 278, 238 },
		{ 2408, 2408 },
		{ 742, 685 },
		{ 2792, 2792 },
		{ 2997, 2995 },
		{ 2546, 2546 },
		{ 1188, 1177 },
		{ 2672, 2672 },
		{ 584, 524 },
		{ 2928, 2928 },
		{ 744, 687 },
		{ 1660, 1660 },
		{ 294, 251 },
		{ 1819, 1819 },
		{ 1819, 1819 },
		{ 2993, 2993 },
		{ 2993, 2993 },
		{ 2795, 2784 },
		{ 361, 317 },
		{ 2796, 2787 },
		{ 1193, 1184 },
		{ 2466, 2465 },
		{ 853, 800 },
		{ 1366, 1365 },
		{ 378, 335 },
		{ 1338, 1337 },
		{ 531, 471 },
		{ 2409, 2408 },
		{ 398, 356 },
		{ 2801, 2792 },
		{ 533, 473 },
		{ 2578, 2546 },
		{ 229, 194 },
		{ 2697, 2672 },
		{ 320, 276 },
		{ 2929, 2928 },
		{ 1819, 1819 },
		{ 1661, 1660 },
		{ 2993, 2993 },
		{ 920, 869 },
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
		{ 1736, 1736 },
		{ 1736, 1736 },
		{ 1820, 1819 },
		{ 2550, 2550 },
		{ 2994, 2993 },
		{ 2803, 2803 },
		{ 478, 426 },
		{ 2804, 2804 },
		{ 1203, 1194 },
		{ 2805, 2805 },
		{ 807, 747 },
		{ 2710, 2710 },
		{ 259, 221 },
		{ 2711, 2711 },
		{ 1287, 1286 },
		{ 2258, 2258 },
		{ 756, 698 },
		{ 2648, 2648 },
		{ 757, 699 },
		{ 2813, 2813 },
		{ 382, 340 },
		{ 2750, 2750 },
		{ 2204, 2201 },
		{ 1736, 1736 },
		{ 668, 602 },
		{ 2574, 2574 },
		{ 2574, 2574 },
		{ 2945, 2945 },
		{ 2945, 2945 },
		{ 2581, 2550 },
		{ 994, 954 },
		{ 2808, 2803 },
		{ 447, 402 },
		{ 2809, 2804 },
		{ 711, 648 },
		{ 2810, 2805 },
		{ 270, 230 },
		{ 2730, 2710 },
		{ 713, 652 },
		{ 2731, 2711 },
		{ 932, 881 },
		{ 2259, 2258 },
		{ 872, 820 },
		{ 2674, 2648 },
		{ 196, 177 },
		{ 2815, 2813 },
		{ 874, 822 },
		{ 2766, 2750 },
		{ 2574, 2574 },
		{ 1737, 1736 },
		{ 2945, 2945 },
		{ 672, 607 },
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
		{ 2518, 2518 },
		{ 2518, 2518 },
		{ 2575, 2574 },
		{ 2457, 2457 },
		{ 2946, 2945 },
		{ 1287, 1287 },
		{ 571, 511 },
		{ 1796, 1796 },
		{ 337, 292 },
		{ 1274, 1274 },
		{ 675, 610 },
		{ 2822, 2822 },
		{ 676, 611 },
		{ 1710, 1710 },
		{ 2859, 2857 },
		{ 2761, 2761 },
		{ 2950, 2947 },
		{ 2721, 2721 },
		{ 677, 612 },
		{ 1762, 1762 },
		{ 638, 574 },
		{ 2899, 2899 },
		{ 1010, 970 },
		{ 2518, 2518 },
		{ 639, 575 },
		{ 2765, 2765 },
		{ 2765, 2765 },
		{ 2767, 2767 },
		{ 2767, 2767 },
		{ 2487, 2457 },
		{ 325, 280 },
		{ 1288, 1287 },
		{ 681, 619 },
		{ 1797, 1796 },
		{ 641, 577 },
		{ 1275, 1274 },
		{ 1231, 1230 },
		{ 2823, 2822 },
		{ 245, 210 },
		{ 1711, 1710 },
		{ 643, 579 },
		{ 2777, 2761 },
		{ 949, 902 },
		{ 2741, 2721 },
		{ 831, 773 },
		{ 1763, 1762 },
		{ 3153, 3153 },
		{ 2900, 2899 },
		{ 2765, 2765 },
		{ 2548, 2518 },
		{ 2767, 2767 },
		{ 2144, 2130 },
		{ 2768, 2768 },
		{ 2768, 2768 },
		{ 3161, 3161 },
		{ 3165, 3165 },
		{ 1609, 1599 },
		{ 1994, 1993 },
		{ 1980, 1974 },
		{ 1615, 1608 },
		{ 1591, 1569 },
		{ 2215, 2213 },
		{ 2191, 2184 },
		{ 1547, 1529 },
		{ 1983, 1979 },
		{ 1970, 1962 },
		{ 1595, 1581 },
		{ 2176, 2159 },
		{ 2196, 2190 },
		{ 3153, 3153 },
		{ 1971, 1964 },
		{ 2177, 2161 },
		{ 1593, 1575 },
		{ 2178, 2163 },
		{ 2781, 2765 },
		{ 2768, 2768 },
		{ 2782, 2767 },
		{ 3161, 3161 },
		{ 3165, 3165 },
		{ 1590, 1566 },
		{ 2179, 2165 },
		{ 1592, 1572 },
		{ 2180, 2167 },
		{ 1594, 1578 },
		{ 2181, 2169 },
		{ 1951, 1939 },
		{ 1634, 1633 },
		{ 3066, 3065 },
		{ 1752, 1751 },
		{ 1753, 1752 },
		{ 1648, 1647 },
		{ 3127, 3126 },
		{ 3128, 3127 },
		{ 1649, 1648 },
		{ 1676, 1675 },
		{ 3156, 3153 },
		{ 1811, 1810 },
		{ 1812, 1811 },
		{ 1677, 1676 },
		{ 1726, 1725 },
		{ 1727, 1726 },
		{ 2783, 2768 },
		{ 1354, 1353 },
		{ 3163, 3161 },
		{ 3166, 3165 },
		{ 1625, 1621 },
		{ 1702, 1701 },
		{ 1703, 1702 },
		{ 1353, 1352 },
		{ 1621, 1616 },
		{ 1622, 1617 },
		{ 1397, 1396 },
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
		{ 170, 166 },
		{ 2284, 2260 },
		{ 2828, 2824 },
		{ 90, 72 },
		{ 2230, 2226 },
		{ 169, 166 },
		{ 2283, 2260 },
		{ 2827, 2824 },
		{ 89, 72 },
		{ 2229, 2226 },
		{ 2344, 2313 },
		{ 2834, 2830 },
		{ 2004, 2000 },
		{ 2829, 2829 },
		{ 164, 160 },
		{ 165, 165 },
		{ 2833, 2830 },
		{ 2003, 2000 },
		{ 3077, 3069 },
		{ 163, 160 },
		{ 2076, 2055 },
		{ 2225, 2225 },
		{ 1999, 1999 },
		{ 171, 168 },
		{ 173, 172 },
		{ 121, 105 },
		{ 2835, 2832 },
		{ 2837, 2836 },
		{ 2830, 2829 },
		{ 1883, 1864 },
		{ 166, 165 },
		{ 2005, 2002 },
		{ 2007, 2006 },
		{ 2231, 2228 },
		{ 2233, 2232 },
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
		{ 0, 2794 },
		{ 2795, 2795 },
		{ 2795, 2795 },
		{ 2796, 2796 },
		{ 2796, 2796 },
		{ 0, 3056 },
		{ 0, 2876 },
		{ 2731, 2731 },
		{ 2731, 2731 },
		{ 0, 2603 },
		{ 0, 1308 },
		{ 0, 2965 },
		{ 2685, 2685 },
		{ 2685, 2685 },
		{ 2801, 2801 },
		{ 2801, 2801 },
		{ 0, 2968 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 0, 2883 },
		{ 0, 2606 },
		{ 2688, 2688 },
		{ 2688, 2688 },
		{ 0, 2973 },
		{ 2795, 2795 },
		{ 0, 2739 },
		{ 2796, 2796 },
		{ 2689, 2689 },
		{ 2689, 2689 },
		{ 0, 2890 },
		{ 2731, 2731 },
		{ 173, 173 },
		{ 174, 173 },
		{ 2808, 2808 },
		{ 2808, 2808 },
		{ 2685, 2685 },
		{ 0, 2646 },
		{ 2801, 2801 },
		{ 2809, 2809 },
		{ 2809, 2809 },
		{ 2409, 2409 },
		{ 2810, 2810 },
		{ 2810, 2810 },
		{ 0, 2516 },
		{ 2688, 2688 },
		{ 2741, 2741 },
		{ 2741, 2741 },
		{ 0, 2984 },
		{ 0, 2376 },
		{ 0, 1787 },
		{ 2689, 2689 },
		{ 2815, 2815 },
		{ 2815, 2815 },
		{ 0, 2748 },
		{ 173, 173 },
		{ 0, 2652 },
		{ 2808, 2808 },
		{ 2575, 2575 },
		{ 2575, 2575 },
		{ 2749, 2749 },
		{ 2749, 2749 },
		{ 2809, 2809 },
		{ 0, 2751 },
		{ 0, 2700 },
		{ 2810, 2810 },
		{ 2823, 2823 },
		{ 2823, 2823 },
		{ 0, 2998 },
		{ 2741, 2741 },
		{ 0, 2754 },
		{ 0, 2755 },
		{ 0, 2911 },
		{ 0, 3002 },
		{ 0, 1261 },
		{ 2815, 2815 },
		{ 2548, 2548 },
		{ 2548, 2548 },
		{ 0, 2703 },
		{ 0, 3006 },
		{ 0, 2376 },
		{ 2575, 2575 },
		{ 0, 3009 },
		{ 2749, 2749 },
		{ 2581, 2581 },
		{ 2581, 2581 },
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
		{ 0, 2845 },
		{ 2548, 2548 },
		{ 2007, 2007 },
		{ 2008, 2007 },
		{ 2766, 2766 },
		{ 2766, 2766 },
		{ 0, 2911 },
		{ 0, 2476 },
		{ 0, 2526 },
		{ 2581, 2581 },
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
		{ 0, 2860 },
		{ 2007, 2007 },
		{ 0, 1329 },
		{ 2766, 2766 },
		{ 2777, 2777 },
		{ 2777, 2777 },
		{ 2674, 2674 },
		{ 2674, 2674 },
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
		{ 2793, 2793 },
		{ 2793, 2793 },
		{ 1542, 1542 },
		{ 1240, 1240 },
		{ 2777, 2777 },
		{ 2244, 2243 },
		{ 2674, 2674 },
		{ 0, 2488 },
		{ 2783, 2783 },
		{ 1384, 1383 },
		{ 2831, 2827 },
		{ 3081, 3077 },
		{ 2001, 2003 },
		{ 167, 163 },
		{ 2001, 1997 },
		{ 2227, 2229 },
		{ 0, 0 },
		{ 2487, 2487 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2793, 2793 },
		{ 0, 0 },
		{ 1542, 1542 },
		{ 1240, 1240 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -70, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 46, 572, 0 },
		{ -179, 2842, 0 },
		{ 5, 0, 0 },
		{ -1239, 1017, -31 },
		{ 7, 0, -31 },
		{ -1243, 1853, -33 },
		{ 9, 0, -33 },
		{ -1256, 3171, 148 },
		{ 11, 0, 148 },
		{ -1277, 3183, 152 },
		{ 13, 0, 152 },
		{ -1290, 3175, 160 },
		{ 15, 0, 160 },
		{ -1325, 3301, 0 },
		{ 17, 0, 0 },
		{ -1340, 3185, 144 },
		{ 19, 0, 144 },
		{ -1368, 3160, 23 },
		{ 21, 0, 23 },
		{ -1410, 230, 0 },
		{ 23, 0, 0 },
		{ -1636, 3302, 0 },
		{ 25, 0, 0 },
		{ -1663, 3164, 0 },
		{ 27, 0, 0 },
		{ -1689, 3165, 0 },
		{ 29, 0, 0 },
		{ -1713, 3177, 0 },
		{ 31, 0, 0 },
		{ -1739, 3187, 0 },
		{ 33, 0, 0 },
		{ -1765, 3158, 164 },
		{ 35, 0, 164 },
		{ -1799, 3306, 271 },
		{ 37, 0, 271 },
		{ 40, 129, 0 },
		{ -1835, 344, 0 },
		{ 42, 127, 0 },
		{ -2024, 116, 0 },
		{ -2236, 3299, 0 },
		{ 43, 0, 0 },
		{ 46, 14, 0 },
		{ -88, 458, 0 },
		{ -2840, 3159, 156 },
		{ 47, 0, 156 },
		{ -2858, 3300, 179 },
		{ 49, 0, 179 },
		{ 2902, 1434, 0 },
		{ 51, 0, 0 },
		{ -2904, 3303, 277 },
		{ 53, 0, 277 },
		{ -2931, 3304, 182 },
		{ 55, 0, 182 },
		{ -2949, 3181, 175 },
		{ 57, 0, 175 },
		{ -2996, 3309, 168 },
		{ 59, 0, 168 },
		{ -3037, 3166, 174 },
		{ 61, 0, 174 },
		{ 46, 1, 0 },
		{ 63, 0, 0 },
		{ -3088, 1813, 0 },
		{ 65, 0, 0 },
		{ -3098, 1722, 45 },
		{ 67, 0, 45 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 431 },
		{ 3069, 4772, 438 },
		{ 0, 0, 249 },
		{ 0, 0, 251 },
		{ 159, 1219, 268 },
		{ 159, 1348, 268 },
		{ 159, 1247, 268 },
		{ 159, 1254, 268 },
		{ 159, 1254, 268 },
		{ 159, 1258, 268 },
		{ 159, 1263, 268 },
		{ 159, 1257, 268 },
		{ 3147, 2830, 438 },
		{ 159, 1268, 268 },
		{ 3147, 1667, 267 },
		{ 104, 2621, 438 },
		{ 159, 0, 268 },
		{ 0, 0, 438 },
		{ -89, 19, 245 },
		{ -90, 4816, 0 },
		{ 159, 1285, 268 },
		{ 159, 750, 268 },
		{ 159, 719, 268 },
		{ 159, 733, 268 },
		{ 159, 716, 268 },
		{ 159, 752, 268 },
		{ 159, 759, 268 },
		{ 159, 774, 268 },
		{ 159, 767, 268 },
		{ 3116, 2281, 0 },
		{ 159, 757, 268 },
		{ 3147, 1822, 264 },
		{ 119, 1386, 0 },
		{ 3147, 1799, 265 },
		{ 3069, 4789, 0 },
		{ 159, 766, 268 },
		{ 159, 764, 268 },
		{ 159, 791, 268 },
		{ 159, 787, 268 },
		{ 159, 0, 256 },
		{ 159, 806, 268 },
		{ 159, 808, 268 },
		{ 159, 793, 268 },
		{ 159, 826, 268 },
		{ 3144, 2947, 0 },
		{ 159, 833, 268 },
		{ 133, 1433, 0 },
		{ 119, 0, 0 },
		{ 3071, 2620, 266 },
		{ 135, 1354, 0 },
		{ 0, 0, 247 },
		{ 159, 837, 252 },
		{ 159, 839, 268 },
		{ 159, 831, 268 },
		{ 159, 836, 268 },
		{ 159, 834, 268 },
		{ 159, 827, 268 },
		{ 159, 0, 259 },
		{ 159, 828, 268 },
		{ 0, 0, 261 },
		{ 159, 834, 268 },
		{ 133, 0, 0 },
		{ 3071, 2675, 264 },
		{ 135, 0, 0 },
		{ 3071, 2588, 265 },
		{ 159, 849, 268 },
		{ 159, 846, 268 },
		{ 159, 848, 268 },
		{ 159, 878, 268 },
		{ 159, 963, 268 },
		{ 159, 0, 258 },
		{ 159, 1052, 268 },
		{ 159, 1125, 268 },
		{ 159, 1118, 268 },
		{ 159, 0, 254 },
		{ 159, 1193, 268 },
		{ 159, 0, 255 },
		{ 159, 0, 257 },
		{ 159, 1191, 268 },
		{ 159, 1199, 268 },
		{ 159, 0, 253 },
		{ 159, 1219, 268 },
		{ 159, 0, 260 },
		{ 159, 769, 268 },
		{ 159, 1247, 268 },
		{ 0, 0, 263 },
		{ 159, 1231, 268 },
		{ 159, 1235, 268 },
		{ 3164, 1305, 262 },
		{ 3069, 4783, 438 },
		{ 165, 0, 249 },
		{ 0, 0, 250 },
		{ -163, 5009, 245 },
		{ -164, 4811, 0 },
		{ 3119, 4794, 0 },
		{ 3069, 4769, 0 },
		{ 0, 0, 246 },
		{ 3069, 4787, 0 },
		{ -169, 16, 0 },
		{ -170, 4818, 0 },
		{ 173, 0, 247 },
		{ 3069, 4788, 0 },
		{ 3119, 4884, 0 },
		{ 0, 0, 248 },
		{ 3120, 1602, 142 },
		{ 2136, 4105, 142 },
		{ 2995, 4596, 142 },
		{ 3120, 4328, 142 },
		{ 0, 0, 142 },
		{ 3108, 3409, 0 },
		{ 2151, 3017, 0 },
		{ 3108, 3635, 0 },
		{ 3108, 3295, 0 },
		{ 3085, 3373, 0 },
		{ 2136, 4136, 0 },
		{ 3112, 3276, 0 },
		{ 2136, 4110, 0 },
		{ 3086, 3544, 0 },
		{ 3115, 3258, 0 },
		{ 3086, 3329, 0 },
		{ 2930, 4386, 0 },
		{ 3112, 3329, 0 },
		{ 2151, 3737, 0 },
		{ 2995, 4524, 0 },
		{ 2082, 3484, 0 },
		{ 2082, 3490, 0 },
		{ 2104, 3124, 0 },
		{ 2085, 3835, 0 },
		{ 2066, 3905, 0 },
		{ 2085, 3846, 0 },
		{ 2104, 3126, 0 },
		{ 2104, 3151, 0 },
		{ 3112, 3359, 0 },
		{ 3036, 3989, 0 },
		{ 2136, 4118, 0 },
		{ 1209, 4072, 0 },
		{ 2082, 3467, 0 },
		{ 2104, 3166, 0 },
		{ 2104, 3169, 0 },
		{ 2995, 4662, 0 },
		{ 2082, 3513, 0 },
		{ 3085, 3798, 0 },
		{ 3108, 3627, 0 },
		{ 2151, 3709, 0 },
		{ 2235, 4214, 0 },
		{ 2995, 3655, 0 },
		{ 3036, 4016, 0 },
		{ 2066, 3882, 0 },
		{ 3086, 3546, 0 },
		{ 2044, 3318, 0 },
		{ 2995, 4564, 0 },
		{ 3108, 3691, 0 },
		{ 3036, 3649, 0 },
		{ 2151, 3767, 0 },
		{ 2104, 3187, 0 },
		{ 3115, 3433, 0 },
		{ 3085, 3715, 0 },
		{ 2044, 3300, 0 },
		{ 2066, 3871, 0 },
		{ 2995, 4588, 0 },
		{ 2136, 4143, 0 },
		{ 2136, 4098, 0 },
		{ 3108, 3664, 0 },
		{ 2104, 3216, 0 },
		{ 2136, 4112, 0 },
		{ 3108, 3698, 0 },
		{ 2235, 4216, 0 },
		{ 2995, 4492, 0 },
		{ 3115, 3434, 0 },
		{ 3086, 3595, 0 },
		{ 2104, 3085, 0 },
		{ 3115, 3343, 0 },
		{ 3108, 3631, 0 },
		{ 1209, 4069, 0 },
		{ 2066, 3907, 0 },
		{ 3036, 3979, 0 },
		{ 2151, 3652, 0 },
		{ 1096, 3268, 0 },
		{ 3108, 3674, 0 },
		{ 3085, 3799, 0 },
		{ 2995, 4504, 0 },
		{ 2235, 4226, 0 },
		{ 2104, 3094, 0 },
		{ 2151, 3733, 0 },
		{ 3115, 3407, 0 },
		{ 2136, 4139, 0 },
		{ 2044, 3297, 0 },
		{ 2136, 4163, 0 },
		{ 3086, 3600, 0 },
		{ 2104, 3104, 0 },
		{ 2930, 4383, 0 },
		{ 3085, 3795, 0 },
		{ 3115, 3436, 0 },
		{ 2172, 3656, 0 },
		{ 2104, 3107, 0 },
		{ 3036, 3957, 0 },
		{ 3086, 3599, 0 },
		{ 2136, 4149, 0 },
		{ 2235, 4241, 0 },
		{ 2136, 4152, 0 },
		{ 2995, 4410, 0 },
		{ 2995, 4430, 0 },
		{ 2066, 3906, 0 },
		{ 2235, 4224, 0 },
		{ 2104, 3108, 0 },
		{ 2995, 4526, 0 },
		{ 3036, 3941, 0 },
		{ 2066, 3862, 0 },
		{ 3036, 3933, 0 },
		{ 2995, 4654, 0 },
		{ 2082, 3528, 0 },
		{ 3086, 3545, 0 },
		{ 2136, 4137, 0 },
		{ 2995, 4486, 0 },
		{ 1209, 4081, 0 },
		{ 3036, 4024, 0 },
		{ 1096, 3270, 0 },
		{ 2172, 4043, 0 },
		{ 2085, 3837, 0 },
		{ 3086, 3580, 0 },
		{ 2104, 3109, 0 },
		{ 2995, 4632, 0 },
		{ 2136, 4106, 0 },
		{ 2104, 3110, 0 },
		{ 0, 0, 73 },
		{ 3108, 3387, 0 },
		{ 3115, 3375, 0 },
		{ 2136, 4130, 0 },
		{ 3120, 4367, 0 },
		{ 2104, 3111, 0 },
		{ 2104, 3112, 0 },
		{ 3115, 3408, 0 },
		{ 2082, 3502, 0 },
		{ 2066, 3876, 0 },
		{ 3115, 3409, 0 },
		{ 2104, 3114, 0 },
		{ 2136, 4092, 0 },
		{ 3108, 3682, 0 },
		{ 3108, 3686, 0 },
		{ 2085, 3856, 0 },
		{ 2995, 4442, 0 },
		{ 3086, 3611, 0 },
		{ 863, 3278, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2082, 3470, 0 },
		{ 2995, 4510, 0 },
		{ 3036, 3971, 0 },
		{ 2066, 3893, 0 },
		{ 3115, 3435, 0 },
		{ 3086, 3554, 0 },
		{ 0, 0, 71 },
		{ 2104, 3115, 0 },
		{ 2082, 3491, 0 },
		{ 3115, 3443, 0 },
		{ 3036, 3968, 0 },
		{ 3115, 3457, 0 },
		{ 2995, 4428, 0 },
		{ 3086, 3609, 0 },
		{ 1209, 4076, 0 },
		{ 2066, 3885, 0 },
		{ 2082, 3522, 0 },
		{ 3085, 3816, 0 },
		{ 2136, 4113, 0 },
		{ 2995, 4516, 0 },
		{ 3120, 4340, 0 },
		{ 3086, 3618, 0 },
		{ 0, 0, 65 },
		{ 3086, 3541, 0 },
		{ 2995, 4572, 0 },
		{ 1209, 4063, 0 },
		{ 3036, 3964, 0 },
		{ 2136, 4138, 0 },
		{ 0, 0, 76 },
		{ 3115, 3370, 0 },
		{ 3108, 3629, 0 },
		{ 3108, 3665, 0 },
		{ 2104, 3116, 0 },
		{ 3036, 4012, 0 },
		{ 2136, 4168, 0 },
		{ 2104, 3117, 0 },
		{ 2082, 3471, 0 },
		{ 3085, 3822, 0 },
		{ 3115, 3377, 0 },
		{ 3086, 3586, 0 },
		{ 2995, 4520, 0 },
		{ 3115, 3379, 0 },
		{ 2104, 3118, 0 },
		{ 3036, 3976, 0 },
		{ 2136, 4126, 0 },
		{ 3115, 3404, 0 },
		{ 3086, 3608, 0 },
		{ 3036, 3998, 0 },
		{ 1036, 4030, 0 },
		{ 0, 0, 8 },
		{ 2082, 3507, 0 },
		{ 2085, 3834, 0 },
		{ 3036, 4022, 0 },
		{ 2117, 3254, 0 },
		{ 2104, 3121, 0 },
		{ 2235, 4298, 0 },
		{ 2136, 4166, 0 },
		{ 2082, 3525, 0 },
		{ 2136, 4171, 0 },
		{ 2136, 4088, 0 },
		{ 2085, 3855, 0 },
		{ 2104, 3122, 0 },
		{ 3115, 3424, 0 },
		{ 3115, 3287, 0 },
		{ 2136, 4109, 0 },
		{ 3112, 3363, 0 },
		{ 3086, 3579, 0 },
		{ 1209, 4068, 0 },
		{ 3085, 3786, 0 },
		{ 2104, 3123, 0 },
		{ 2151, 3088, 0 },
		{ 2995, 4426, 0 },
		{ 1209, 4079, 0 },
		{ 2151, 3751, 0 },
		{ 3120, 3251, 0 },
		{ 2117, 3244, 0 },
		{ 2085, 3848, 0 },
		{ 2082, 3504, 0 },
		{ 3115, 3458, 0 },
		{ 0, 0, 117 },
		{ 2104, 3125, 0 },
		{ 2151, 3755, 0 },
		{ 2043, 3274, 0 },
		{ 3036, 3984, 0 },
		{ 3108, 3653, 0 },
		{ 3085, 3826, 0 },
		{ 2995, 4584, 0 },
		{ 2136, 4096, 0 },
		{ 0, 0, 7 },
		{ 2085, 3853, 0 },
		{ 0, 0, 6 },
		{ 3036, 4015, 0 },
		{ 0, 0, 122 },
		{ 3085, 3775, 0 },
		{ 2136, 4107, 0 },
		{ 3120, 1660, 0 },
		{ 2104, 3127, 0 },
		{ 3085, 3797, 0 },
		{ 3108, 3672, 0 },
		{ 0, 0, 126 },
		{ 2104, 3128, 0 },
		{ 2136, 4124, 0 },
		{ 3120, 3262, 0 },
		{ 2136, 4129, 0 },
		{ 2235, 4245, 0 },
		{ 2151, 3770, 0 },
		{ 0, 0, 72 },
		{ 2066, 3872, 0 },
		{ 2104, 3131, 109 },
		{ 2104, 3133, 110 },
		{ 2995, 4558, 0 },
		{ 3036, 3985, 0 },
		{ 3086, 3574, 0 },
		{ 3108, 3703, 0 },
		{ 3086, 3577, 0 },
		{ 1209, 4060, 0 },
		{ 3108, 3661, 0 },
		{ 3086, 3578, 0 },
		{ 3112, 3335, 0 },
		{ 2151, 3711, 0 },
		{ 3036, 4025, 0 },
		{ 2136, 4093, 0 },
		{ 2104, 3134, 0 },
		{ 3115, 3422, 0 },
		{ 2995, 4452, 0 },
		{ 3036, 3959, 0 },
		{ 2930, 4381, 0 },
		{ 3036, 3963, 0 },
		{ 2151, 3746, 0 },
		{ 3086, 3592, 0 },
		{ 3036, 3970, 0 },
		{ 0, 0, 9 },
		{ 2104, 3135, 0 },
		{ 3036, 3973, 0 },
		{ 2117, 3257, 0 },
		{ 2172, 4032, 0 },
		{ 0, 0, 107 },
		{ 2082, 3512, 0 },
		{ 3085, 3813, 0 },
		{ 3108, 3645, 0 },
		{ 2151, 3657, 0 },
		{ 3085, 3260, 0 },
		{ 3036, 4006, 0 },
		{ 3112, 3315, 0 },
		{ 3036, 4013, 0 },
		{ 3112, 3299, 0 },
		{ 3108, 3668, 0 },
		{ 2136, 4159, 0 },
		{ 3115, 3442, 0 },
		{ 3086, 3620, 0 },
		{ 3112, 3289, 0 },
		{ 3085, 3805, 0 },
		{ 3115, 3448, 0 },
		{ 3036, 3936, 0 },
		{ 3115, 3341, 0 },
		{ 2995, 4518, 0 },
		{ 2104, 3136, 0 },
		{ 2995, 4522, 0 },
		{ 3086, 3549, 0 },
		{ 2136, 4104, 0 },
		{ 3108, 3637, 0 },
		{ 3108, 3639, 0 },
		{ 2066, 3891, 0 },
		{ 2082, 3486, 0 },
		{ 2104, 3137, 97 },
		{ 3086, 3575, 0 },
		{ 2104, 3138, 0 },
		{ 2104, 3139, 0 },
		{ 3112, 3351, 0 },
		{ 2151, 3734, 0 },
		{ 2235, 4210, 0 },
		{ 2104, 3140, 0 },
		{ 2082, 3508, 0 },
		{ 0, 0, 106 },
		{ 2235, 4222, 0 },
		{ 2995, 4450, 0 },
		{ 3115, 3387, 0 },
		{ 3115, 3391, 0 },
		{ 0, 0, 119 },
		{ 0, 0, 121 },
		{ 3085, 3794, 0 },
		{ 2151, 3758, 0 },
		{ 2082, 3521, 0 },
		{ 2136, 3381, 0 },
		{ 3115, 3400, 0 },
		{ 2136, 4156, 0 },
		{ 2104, 3141, 0 },
		{ 2136, 4160, 0 },
		{ 3036, 3960, 0 },
		{ 3085, 3808, 0 },
		{ 2104, 3143, 0 },
		{ 2172, 4039, 0 },
		{ 3112, 3338, 0 },
		{ 2235, 4212, 0 },
		{ 2104, 3145, 0 },
		{ 2995, 4630, 0 },
		{ 2082, 3479, 0 },
		{ 964, 3921, 0 },
		{ 3115, 3410, 0 },
		{ 3085, 3785, 0 },
		{ 2151, 3754, 0 },
		{ 2235, 4243, 0 },
		{ 3115, 3420, 0 },
		{ 2044, 3299, 0 },
		{ 2104, 3146, 0 },
		{ 3036, 3999, 0 },
		{ 3108, 3649, 0 },
		{ 2082, 3492, 0 },
		{ 2995, 4500, 0 },
		{ 3115, 3427, 0 },
		{ 2151, 3713, 0 },
		{ 2117, 3251, 0 },
		{ 3086, 3576, 0 },
		{ 2082, 3505, 0 },
		{ 2151, 3744, 0 },
		{ 2085, 3842, 0 },
		{ 3120, 3336, 0 },
		{ 2104, 3147, 0 },
		{ 0, 0, 66 },
		{ 2104, 3148, 0 },
		{ 3108, 3684, 0 },
		{ 3086, 3582, 0 },
		{ 3108, 3690, 0 },
		{ 3086, 3585, 0 },
		{ 2104, 3149, 111 },
		{ 2066, 3895, 0 },
		{ 2151, 3710, 0 },
		{ 2085, 3844, 0 },
		{ 2082, 3514, 0 },
		{ 2082, 3520, 0 },
		{ 2066, 3864, 0 },
		{ 2085, 3849, 0 },
		{ 2995, 4438, 0 },
		{ 3108, 3626, 0 },
		{ 3112, 3330, 0 },
		{ 3036, 4009, 0 },
		{ 3115, 3445, 0 },
		{ 2082, 3523, 0 },
		{ 0, 0, 123 },
		{ 2104, 3150, 0 },
		{ 2930, 4384, 0 },
		{ 2085, 3836, 0 },
		{ 3115, 3451, 0 },
		{ 3085, 3788, 0 },
		{ 0, 0, 118 },
		{ 0, 0, 108 },
		{ 2082, 3466, 0 },
		{ 3086, 3619, 0 },
		{ 3115, 3455, 0 },
		{ 2151, 2997, 0 },
		{ 3120, 3297, 0 },
		{ 3036, 3962, 0 },
		{ 3085, 3800, 0 },
		{ 2104, 3152, 0 },
		{ 3036, 3965, 0 },
		{ 2066, 3877, 0 },
		{ 3108, 3696, 0 },
		{ 2136, 4146, 0 },
		{ 2995, 4644, 0 },
		{ 2995, 4648, 0 },
		{ 2082, 3480, 0 },
		{ 2995, 4658, 0 },
		{ 3036, 3972, 0 },
		{ 2995, 4664, 0 },
		{ 3086, 3548, 0 },
		{ 3085, 3819, 0 },
		{ 2104, 3154, 0 },
		{ 2136, 4162, 0 },
		{ 3086, 3551, 0 },
		{ 2104, 3158, 0 },
		{ 3086, 3555, 0 },
		{ 2136, 4170, 0 },
		{ 3036, 3995, 0 },
		{ 2136, 4179, 0 },
		{ 3086, 3572, 0 },
		{ 2136, 4090, 0 },
		{ 2082, 3488, 0 },
		{ 3085, 3790, 0 },
		{ 2104, 3159, 0 },
		{ 3120, 4306, 0 },
		{ 2235, 4292, 0 },
		{ 2136, 4097, 0 },
		{ 2085, 3857, 0 },
		{ 2136, 4100, 0 },
		{ 2085, 3859, 0 },
		{ 3108, 3633, 0 },
		{ 2995, 4576, 0 },
		{ 3108, 3676, 0 },
		{ 0, 0, 99 },
		{ 3036, 4017, 0 },
		{ 3036, 4019, 0 },
		{ 2995, 4603, 0 },
		{ 2104, 3160, 0 },
		{ 2104, 3164, 0 },
		{ 2995, 4634, 0 },
		{ 2995, 4636, 0 },
		{ 2995, 4642, 0 },
		{ 2085, 3838, 0 },
		{ 2082, 3493, 0 },
		{ 0, 0, 129 },
		{ 3108, 3688, 0 },
		{ 0, 0, 120 },
		{ 0, 0, 124 },
		{ 2995, 4656, 0 },
		{ 2235, 4208, 0 },
		{ 1096, 3265, 0 },
		{ 2104, 3165, 0 },
		{ 3036, 3091, 0 },
		{ 3086, 3583, 0 },
		{ 2085, 3854, 0 },
		{ 1209, 4058, 0 },
		{ 2995, 4432, 0 },
		{ 3108, 3699, 0 },
		{ 3108, 3700, 0 },
		{ 3108, 3702, 0 },
		{ 3085, 3776, 0 },
		{ 2235, 4249, 0 },
		{ 2172, 4055, 0 },
		{ 3085, 3783, 0 },
		{ 3112, 3331, 0 },
		{ 2066, 3894, 0 },
		{ 1209, 4083, 0 },
		{ 3115, 3406, 0 },
		{ 2066, 3899, 0 },
		{ 2082, 3509, 0 },
		{ 2104, 3167, 0 },
		{ 2085, 3840, 0 },
		{ 2066, 3916, 0 },
		{ 2136, 4183, 0 },
		{ 2172, 4037, 0 },
		{ 2151, 3761, 0 },
		{ 3086, 3596, 0 },
		{ 2995, 4586, 0 },
		{ 2151, 3769, 0 },
		{ 0, 0, 60 },
		{ 0, 0, 61 },
		{ 2995, 4590, 0 },
		{ 3086, 3598, 0 },
		{ 0, 0, 70 },
		{ 0, 0, 115 },
		{ 2044, 3303, 0 },
		{ 3112, 3339, 0 },
		{ 2082, 3516, 0 },
		{ 3112, 3346, 0 },
		{ 3115, 3419, 0 },
		{ 3036, 4021, 0 },
		{ 3086, 3613, 0 },
		{ 0, 0, 101 },
		{ 3086, 3614, 0 },
		{ 0, 0, 103 },
		{ 3108, 3687, 0 },
		{ 3086, 3615, 0 },
		{ 3085, 3828, 0 },
		{ 2136, 4116, 0 },
		{ 2117, 3245, 0 },
		{ 2117, 3246, 0 },
		{ 3115, 3423, 0 },
		{ 1209, 4071, 0 },
		{ 2172, 3779, 0 },
		{ 3086, 3532, 0 },
		{ 964, 3925, 0 },
		{ 2066, 3867, 0 },
		{ 0, 0, 116 },
		{ 0, 0, 128 },
		{ 3086, 3533, 0 },
		{ 3086, 3534, 0 },
		{ 0, 0, 141 },
		{ 2082, 3527, 0 },
		{ 3120, 3940, 0 },
		{ 2995, 4494, 0 },
		{ 1209, 4066, 0 },
		{ 2995, 4502, 0 },
		{ 2136, 4154, 0 },
		{ 3120, 4314, 0 },
		{ 3086, 3543, 0 },
		{ 3120, 4332, 0 },
		{ 3112, 3361, 0 },
		{ 3112, 3362, 0 },
		{ 3085, 3087, 0 },
		{ 2104, 3170, 0 },
		{ 2136, 4165, 0 },
		{ 3036, 3991, 0 },
		{ 2995, 4568, 0 },
		{ 2995, 4570, 0 },
		{ 3036, 3994, 0 },
		{ 2151, 3715, 0 },
		{ 3036, 3996, 0 },
		{ 2151, 3732, 0 },
		{ 2151, 3655, 0 },
		{ 3036, 4001, 0 },
		{ 2104, 3171, 0 },
		{ 2235, 4296, 0 },
		{ 2104, 3173, 0 },
		{ 3108, 3673, 0 },
		{ 2104, 3175, 0 },
		{ 2085, 3841, 0 },
		{ 3108, 3675, 0 },
		{ 2066, 3866, 0 },
		{ 3036, 4018, 0 },
		{ 2104, 3176, 0 },
		{ 3108, 3677, 0 },
		{ 3120, 4316, 0 },
		{ 1209, 4062, 0 },
		{ 2151, 3757, 0 },
		{ 3086, 3564, 0 },
		{ 2995, 4414, 0 },
		{ 2995, 4416, 0 },
		{ 2136, 4111, 0 },
		{ 2085, 3852, 0 },
		{ 2235, 4294, 0 },
		{ 3086, 3565, 0 },
		{ 2136, 4114, 0 },
		{ 2136, 4115, 0 },
		{ 3036, 3946, 0 },
		{ 3036, 3947, 0 },
		{ 2136, 4119, 0 },
		{ 2995, 4488, 0 },
		{ 2995, 4490, 0 },
		{ 2136, 4123, 0 },
		{ 2104, 3177, 0 },
		{ 2151, 3768, 0 },
		{ 3115, 3446, 0 },
		{ 3115, 3447, 0 },
		{ 2136, 4132, 0 },
		{ 3112, 3314, 0 },
		{ 3112, 3358, 0 },
		{ 2066, 3902, 0 },
		{ 3120, 4344, 0 },
		{ 3115, 3453, 0 },
		{ 2104, 3179, 63 },
		{ 3115, 3456, 0 },
		{ 2995, 4562, 0 },
		{ 2151, 3729, 0 },
		{ 2104, 3181, 0 },
		{ 2104, 3183, 0 },
		{ 2172, 4044, 0 },
		{ 3036, 3977, 0 },
		{ 3120, 4310, 0 },
		{ 3085, 3817, 0 },
		{ 3115, 3459, 0 },
		{ 3115, 3460, 0 },
		{ 1096, 3267, 0 },
		{ 2066, 3874, 0 },
		{ 3086, 3594, 0 },
		{ 2117, 3243, 0 },
		{ 2044, 3320, 0 },
		{ 2044, 3323, 0 },
		{ 3108, 3669, 0 },
		{ 2082, 3519, 0 },
		{ 1209, 4074, 0 },
		{ 3112, 3336, 0 },
		{ 3086, 3602, 0 },
		{ 3120, 4365, 0 },
		{ 3120, 4366, 0 },
		{ 2136, 4095, 0 },
		{ 0, 0, 64 },
		{ 0, 0, 67 },
		{ 2995, 4668, 0 },
		{ 1209, 4080, 0 },
		{ 2066, 3896, 0 },
		{ 3086, 3603, 0 },
		{ 1209, 4086, 0 },
		{ 3086, 3604, 0 },
		{ 0, 0, 112 },
		{ 3115, 3381, 0 },
		{ 3115, 3384, 0 },
		{ 3086, 3610, 0 },
		{ 0, 0, 105 },
		{ 2104, 3185, 0 },
		{ 2995, 4444, 0 },
		{ 0, 0, 113 },
		{ 0, 0, 114 },
		{ 2151, 3712, 0 },
		{ 2066, 3863, 0 },
		{ 3085, 3806, 0 },
		{ 964, 3924, 0 },
		{ 2085, 3843, 0 },
		{ 1209, 4075, 0 },
		{ 3115, 3388, 0 },
		{ 2930, 4374, 0 },
		{ 0, 0, 3 },
		{ 2136, 4117, 0 },
		{ 3120, 4348, 0 },
		{ 2235, 4218, 0 },
		{ 2995, 4514, 0 },
		{ 3085, 3809, 0 },
		{ 3036, 3955, 0 },
		{ 3115, 3389, 0 },
		{ 3036, 3958, 0 },
		{ 2136, 4125, 0 },
		{ 2104, 3186, 0 },
		{ 2085, 3850, 0 },
		{ 2235, 4251, 0 },
		{ 2082, 3463, 0 },
		{ 2082, 3464, 0 },
		{ 2136, 4134, 0 },
		{ 3085, 3821, 0 },
		{ 1036, 4027, 0 },
		{ 2136, 3097, 0 },
		{ 3036, 3966, 0 },
		{ 2151, 3736, 0 },
		{ 0, 0, 74 },
		{ 2136, 4144, 0 },
		{ 0, 0, 82 },
		{ 2995, 4594, 0 },
		{ 2136, 4145, 0 },
		{ 2995, 4598, 0 },
		{ 3115, 3397, 0 },
		{ 2136, 4148, 0 },
		{ 3112, 3349, 0 },
		{ 3120, 4353, 0 },
		{ 2136, 4151, 0 },
		{ 2151, 3745, 0 },
		{ 2066, 3897, 0 },
		{ 3115, 3401, 0 },
		{ 2066, 3900, 0 },
		{ 3036, 3978, 0 },
		{ 2151, 3747, 0 },
		{ 3036, 3983, 0 },
		{ 2136, 4164, 0 },
		{ 0, 0, 69 },
		{ 2151, 3748, 0 },
		{ 2151, 3749, 0 },
		{ 2995, 4412, 0 },
		{ 2085, 3839, 0 },
		{ 3115, 3403, 0 },
		{ 3085, 3791, 0 },
		{ 2136, 4178, 0 },
		{ 2151, 3753, 0 },
		{ 2136, 4181, 0 },
		{ 2104, 3188, 0 },
		{ 3036, 3997, 0 },
		{ 3108, 3650, 0 },
		{ 2995, 4448, 0 },
		{ 2085, 3845, 0 },
		{ 2066, 3868, 0 },
		{ 2995, 4459, 0 },
		{ 2082, 3482, 0 },
		{ 3120, 4352, 0 },
		{ 2082, 3483, 0 },
		{ 2104, 3189, 0 },
		{ 2151, 3766, 0 },
		{ 2044, 3301, 0 },
		{ 3120, 4309, 0 },
		{ 2136, 4101, 0 },
		{ 2136, 4103, 0 },
		{ 863, 3279, 0 },
		{ 0, 3280, 0 },
		{ 3085, 3811, 0 },
		{ 2172, 4034, 0 },
		{ 2136, 4108, 0 },
		{ 3086, 3552, 0 },
		{ 1209, 4064, 0 },
		{ 2995, 4531, 0 },
		{ 3086, 3553, 0 },
		{ 2104, 3190, 0 },
		{ 3115, 3411, 0 },
		{ 3086, 3561, 0 },
		{ 2066, 3898, 0 },
		{ 3036, 3942, 0 },
		{ 3086, 3563, 0 },
		{ 3085, 3824, 0 },
		{ 3115, 3413, 0 },
		{ 2235, 4300, 0 },
		{ 2235, 4306, 0 },
		{ 2995, 4592, 0 },
		{ 2136, 4122, 0 },
		{ 0, 0, 68 },
		{ 2066, 3904, 0 },
		{ 3115, 3414, 0 },
		{ 3108, 3685, 0 },
		{ 3086, 3567, 0 },
		{ 3086, 3569, 0 },
		{ 3086, 3571, 0 },
		{ 3115, 3415, 0 },
		{ 2151, 3738, 0 },
		{ 2151, 3740, 0 },
		{ 0, 0, 133 },
		{ 0, 0, 134 },
		{ 2085, 3847, 0 },
		{ 1209, 4067, 0 },
		{ 3115, 3416, 0 },
		{ 2066, 3869, 0 },
		{ 2066, 3870, 0 },
		{ 2930, 4372, 0 },
		{ 0, 0, 10 },
		{ 2995, 4666, 0 },
		{ 0, 0, 12 },
		{ 2082, 3506, 0 },
		{ 3115, 3417, 0 },
		{ 2995, 4062, 0 },
		{ 3120, 4356, 0 },
		{ 3085, 3796, 0 },
		{ 2995, 4418, 0 },
		{ 2995, 4420, 0 },
		{ 3115, 3418, 0 },
		{ 2104, 3191, 0 },
		{ 3036, 3980, 0 },
		{ 3036, 3981, 0 },
		{ 2136, 4157, 0 },
		{ 2104, 3192, 0 },
		{ 3120, 4320, 0 },
		{ 2995, 4446, 0 },
		{ 3120, 4324, 0 },
		{ 2066, 3884, 0 },
		{ 0, 0, 83 },
		{ 2151, 3750, 0 },
		{ 3036, 3986, 0 },
		{ 0, 0, 81 },
		{ 3112, 3332, 0 },
		{ 2085, 3858, 0 },
		{ 0, 0, 84 },
		{ 3120, 4351, 0 },
		{ 3036, 3993, 0 },
		{ 3112, 3334, 0 },
		{ 2136, 4169, 0 },
		{ 2082, 3515, 0 },
		{ 3086, 3584, 0 },
		{ 2136, 4172, 0 },
		{ 2104, 3193, 0 },
		{ 3115, 3425, 0 },
		{ 0, 0, 62 },
		{ 0, 0, 100 },
		{ 0, 0, 102 },
		{ 2151, 3759, 0 },
		{ 2235, 4302, 0 },
		{ 3086, 3587, 0 },
		{ 2136, 4184, 0 },
		{ 3036, 4003, 0 },
		{ 3108, 3666, 0 },
		{ 2136, 4091, 0 },
		{ 0, 0, 139 },
		{ 3036, 4007, 0 },
		{ 3086, 3591, 0 },
		{ 2136, 4094, 0 },
		{ 3036, 4010, 0 },
		{ 3115, 3426, 0 },
		{ 3086, 3593, 0 },
		{ 2995, 4582, 0 },
		{ 2104, 3194, 0 },
		{ 2066, 3908, 0 },
		{ 2066, 3912, 0 },
		{ 2136, 4102, 0 },
		{ 2235, 4290, 0 },
		{ 3115, 3428, 0 },
		{ 3115, 3429, 0 },
		{ 3086, 3597, 0 },
		{ 2172, 4038, 0 },
		{ 0, 3926, 0 },
		{ 3115, 3430, 0 },
		{ 3115, 3431, 0 },
		{ 3036, 3940, 0 },
		{ 3108, 3683, 0 },
		{ 2151, 3724, 0 },
		{ 2995, 4646, 0 },
		{ 3036, 3945, 0 },
		{ 3115, 3432, 0 },
		{ 2151, 3731, 0 },
		{ 3120, 4357, 0 },
		{ 0, 0, 20 },
		{ 2082, 3529, 0 },
		{ 2082, 3530, 0 },
		{ 0, 0, 130 },
		{ 2082, 3462, 0 },
		{ 0, 0, 132 },
		{ 3086, 3606, 0 },
		{ 2136, 4121, 0 },
		{ 0, 0, 98 },
		{ 2104, 3195, 0 },
		{ 2066, 3879, 0 },
		{ 2066, 3881, 0 },
		{ 2104, 3196, 0 },
		{ 2066, 3883, 0 },
		{ 2995, 4422, 0 },
		{ 2082, 3465, 0 },
		{ 2151, 3743, 0 },
		{ 3036, 3969, 0 },
		{ 0, 0, 79 },
		{ 2066, 3888, 0 },
		{ 1209, 4073, 0 },
		{ 2104, 3197, 0 },
		{ 2066, 3892, 0 },
		{ 3086, 3612, 0 },
		{ 2136, 4141, 0 },
		{ 3120, 4361, 0 },
		{ 3120, 4364, 0 },
		{ 2995, 4454, 0 },
		{ 2136, 4142, 0 },
		{ 3036, 3974, 0 },
		{ 3036, 3975, 0 },
		{ 2104, 3198, 0 },
		{ 2082, 3469, 0 },
		{ 3115, 3437, 0 },
		{ 3085, 3815, 0 },
		{ 3115, 3438, 0 },
		{ 2082, 3478, 0 },
		{ 3036, 3982, 0 },
		{ 3085, 3818, 0 },
		{ 3115, 3439, 0 },
		{ 2136, 4158, 0 },
		{ 0, 0, 87 },
		{ 3120, 4347, 0 },
		{ 0, 0, 104 },
		{ 2066, 3903, 0 },
		{ 3120, 3944, 0 },
		{ 2136, 4161, 0 },
		{ 0, 0, 137 },
		{ 3115, 3440, 0 },
		{ 3036, 3990, 0 },
		{ 3115, 3441, 0 },
		{ 2104, 3199, 59 },
		{ 3085, 3825, 0 },
		{ 2151, 3756, 0 },
		{ 2066, 3909, 0 },
		{ 3112, 3357, 0 },
		{ 2930, 3925, 0 },
		{ 0, 0, 88 },
		{ 2082, 3485, 0 },
		{ 3120, 4311, 0 },
		{ 1036, 4028, 0 },
		{ 0, 4029, 0 },
		{ 3115, 3444, 0 },
		{ 3085, 3779, 0 },
		{ 3085, 3782, 0 },
		{ 2151, 3760, 0 },
		{ 3120, 4338, 0 },
		{ 2136, 4089, 0 },
		{ 3036, 4008, 0 },
		{ 2104, 3200, 0 },
		{ 2151, 3762, 0 },
		{ 2151, 3764, 0 },
		{ 2151, 3765, 0 },
		{ 0, 0, 92 },
		{ 3036, 4014, 0 },
		{ 2082, 3489, 0 },
		{ 3086, 3547, 0 },
		{ 0, 0, 125 },
		{ 2104, 3201, 0 },
		{ 3112, 3360, 0 },
		{ 2104, 3202, 0 },
		{ 3108, 3681, 0 },
		{ 3115, 3449, 0 },
		{ 3036, 4023, 0 },
		{ 2235, 4220, 0 },
		{ 2082, 3501, 0 },
		{ 3085, 3801, 0 },
		{ 0, 0, 94 },
		{ 3085, 3802, 0 },
		{ 2235, 4228, 0 },
		{ 2235, 4239, 0 },
		{ 3115, 3450, 0 },
		{ 0, 0, 16 },
		{ 2066, 3886, 0 },
		{ 0, 0, 18 },
		{ 0, 0, 19 },
		{ 3036, 3944, 0 },
		{ 2104, 3203, 0 },
		{ 2172, 4045, 0 },
		{ 3085, 3807, 0 },
		{ 2995, 4440, 0 },
		{ 3086, 3556, 0 },
		{ 2151, 3721, 0 },
		{ 1209, 4070, 0 },
		{ 3086, 3558, 0 },
		{ 3086, 3560, 0 },
		{ 3085, 3814, 0 },
		{ 2151, 3730, 0 },
		{ 0, 0, 58 },
		{ 3036, 3961, 0 },
		{ 3115, 3452, 0 },
		{ 2104, 3204, 0 },
		{ 3115, 3454, 0 },
		{ 2066, 3901, 0 },
		{ 1096, 3262, 0 },
		{ 2151, 3735, 0 },
		{ 2136, 4131, 0 },
		{ 0, 0, 77 },
		{ 2104, 3205, 0 },
		{ 3120, 4318, 0 },
		{ 3086, 3568, 0 },
		{ 0, 3264, 0 },
		{ 3086, 3570, 0 },
		{ 0, 0, 17 },
		{ 2151, 3741, 0 },
		{ 1209, 4065, 0 },
		{ 2104, 3206, 57 },
		{ 2104, 3207, 0 },
		{ 2066, 3914, 0 },
		{ 0, 0, 78 },
		{ 3085, 3777, 0 },
		{ 3120, 3312, 0 },
		{ 0, 0, 85 },
		{ 0, 0, 86 },
		{ 0, 0, 55 },
		{ 3085, 3781, 0 },
		{ 3108, 3657, 0 },
		{ 3108, 3658, 0 },
		{ 3115, 3368, 0 },
		{ 3108, 3662, 0 },
		{ 0, 0, 138 },
		{ 0, 0, 127 },
		{ 3085, 3787, 0 },
		{ 1209, 4077, 0 },
		{ 1209, 4078, 0 },
		{ 3115, 3369, 0 },
		{ 0, 0, 40 },
		{ 2104, 3208, 41 },
		{ 2104, 3209, 43 },
		{ 3085, 3792, 0 },
		{ 3120, 4312, 0 },
		{ 1209, 4084, 0 },
		{ 1209, 4085, 0 },
		{ 2066, 3873, 0 },
		{ 0, 0, 75 },
		{ 3085, 3793, 0 },
		{ 3115, 3371, 0 },
		{ 0, 0, 93 },
		{ 3115, 3374, 0 },
		{ 2066, 3878, 0 },
		{ 3108, 3671, 0 },
		{ 2066, 3880, 0 },
		{ 2082, 3524, 0 },
		{ 3036, 4000, 0 },
		{ 3112, 3343, 0 },
		{ 3036, 4002, 0 },
		{ 2172, 4049, 0 },
		{ 2172, 4050, 0 },
		{ 2104, 3210, 0 },
		{ 3115, 3378, 0 },
		{ 3120, 4359, 0 },
		{ 3112, 3348, 0 },
		{ 0, 0, 89 },
		{ 3120, 4362, 0 },
		{ 2104, 3211, 0 },
		{ 0, 0, 131 },
		{ 0, 0, 135 },
		{ 2066, 3887, 0 },
		{ 0, 0, 140 },
		{ 0, 0, 11 },
		{ 3085, 3803, 0 },
		{ 3085, 3804, 0 },
		{ 2151, 3763, 0 },
		{ 3108, 3678, 0 },
		{ 3108, 3680, 0 },
		{ 1209, 4082, 0 },
		{ 2104, 3212, 0 },
		{ 3115, 3385, 0 },
		{ 3085, 3810, 0 },
		{ 3115, 3386, 0 },
		{ 3120, 4322, 0 },
		{ 0, 0, 136 },
		{ 3036, 4020, 0 },
		{ 3120, 4326, 0 },
		{ 3085, 3812, 0 },
		{ 3112, 3353, 0 },
		{ 3112, 3354, 0 },
		{ 3112, 3355, 0 },
		{ 3120, 4341, 0 },
		{ 2104, 3213, 0 },
		{ 3120, 4346, 0 },
		{ 3036, 3939, 0 },
		{ 2995, 4498, 0 },
		{ 3115, 3392, 0 },
		{ 3115, 3393, 0 },
		{ 2104, 3214, 0 },
		{ 0, 0, 42 },
		{ 0, 0, 44 },
		{ 3085, 3820, 0 },
		{ 2995, 4512, 0 },
		{ 3120, 4354, 0 },
		{ 3115, 3395, 0 },
		{ 2151, 3716, 0 },
		{ 2066, 3910, 0 },
		{ 3036, 3949, 0 },
		{ 3036, 3951, 0 },
		{ 2930, 4375, 0 },
		{ 3120, 4363, 0 },
		{ 2066, 3911, 0 },
		{ 2995, 4560, 0 },
		{ 3036, 3956, 0 },
		{ 3085, 3823, 0 },
		{ 2066, 3913, 0 },
		{ 2151, 3717, 0 },
		{ 2151, 3719, 0 },
		{ 2136, 4127, 0 },
		{ 3115, 3396, 0 },
		{ 2066, 3917, 0 },
		{ 2066, 3918, 0 },
		{ 2151, 3722, 0 },
		{ 0, 0, 80 },
		{ 0, 0, 95 },
		{ 3085, 3829, 0 },
		{ 3085, 3830, 0 },
		{ 0, 4059, 0 },
		{ 3036, 3967, 0 },
		{ 0, 0, 90 },
		{ 2066, 3865, 0 },
		{ 3085, 3774, 0 },
		{ 2082, 3481, 0 },
		{ 0, 0, 13 },
		{ 2151, 3726, 0 },
		{ 2151, 3728, 0 },
		{ 0, 0, 91 },
		{ 0, 0, 56 },
		{ 0, 0, 96 },
		{ 3086, 3605, 0 },
		{ 3085, 3780, 0 },
		{ 2136, 4147, 0 },
		{ 0, 0, 15 },
		{ 2104, 3215, 0 },
		{ 3086, 3607, 0 },
		{ 2136, 4150, 0 },
		{ 3108, 3704, 0 },
		{ 2066, 3875, 0 },
		{ 2995, 4660, 0 },
		{ 3120, 4355, 0 },
		{ 2136, 4153, 0 },
		{ 2085, 3851, 0 },
		{ 2136, 4155, 0 },
		{ 3085, 3784, 0 },
		{ 3115, 3399, 0 },
		{ 0, 0, 14 },
		{ 3164, 1385, 237 },
		{ 0, 0, 238 },
		{ 3119, 5000, 239 },
		{ 3147, 1701, 243 },
		{ 1246, 2609, 244 },
		{ 0, 0, 244 },
		{ 3147, 1723, 240 },
		{ 1249, 1387, 0 },
		{ 3147, 1734, 241 },
		{ 1252, 1434, 0 },
		{ 1249, 0, 0 },
		{ 3071, 2724, 242 },
		{ 1254, 1468, 0 },
		{ 1252, 0, 0 },
		{ 3071, 2598, 240 },
		{ 1254, 0, 0 },
		{ 3071, 2608, 241 },
		{ 3112, 3352, 149 },
		{ 0, 0, 149 },
		{ 0, 0, 150 },
		{ 3125, 2042, 0 },
		{ 3147, 2861, 0 },
		{ 3164, 2083, 0 },
		{ 1262, 4835, 0 },
		{ 3144, 2601, 0 },
		{ 3147, 2809, 0 },
		{ 3159, 2974, 0 },
		{ 3155, 2475, 0 },
		{ 3158, 3033, 0 },
		{ 3164, 2060, 0 },
		{ 3158, 3006, 0 },
		{ 3160, 1942, 0 },
		{ 3062, 2686, 0 },
		{ 3162, 2185, 0 },
		{ 3116, 2332, 0 },
		{ 3125, 2053, 0 },
		{ 3165, 4666, 0 },
		{ 0, 0, 147 },
		{ 3112, 3337, 153 },
		{ 0, 0, 153 },
		{ 0, 0, 154 },
		{ 3125, 2024, 0 },
		{ 3147, 2870, 0 },
		{ 3164, 2101, 0 },
		{ 1283, 4908, 0 },
		{ 3120, 3994, 0 },
		{ 3112, 3345, 0 },
		{ 2235, 4307, 0 },
		{ 2995, 4566, 0 },
		{ 3165, 4662, 0 },
		{ 0, 0, 151 },
		{ 2930, 4377, 161 },
		{ 0, 0, 161 },
		{ 0, 0, 162 },
		{ 3147, 2813, 0 },
		{ 3008, 2793, 0 },
		{ 3162, 2190, 0 },
		{ 3164, 2115, 0 },
		{ 3147, 2931, 0 },
		{ 1298, 4895, 0 },
		{ 3147, 2524, 0 },
		{ 3129, 1443, 0 },
		{ 3147, 2815, 0 },
		{ 3164, 2082, 0 },
		{ 2772, 1405, 0 },
		{ 3160, 1832, 0 },
		{ 3154, 2743, 0 },
		{ 3062, 2719, 0 },
		{ 3116, 2291, 0 },
		{ 2964, 2759, 0 },
		{ 1309, 4771, 0 },
		{ 3147, 2529, 0 },
		{ 3155, 2443, 0 },
		{ 3125, 2040, 0 },
		{ 3147, 2899, 0 },
		{ 1314, 4875, 0 },
		{ 3157, 2466, 0 },
		{ 3152, 1561, 0 },
		{ 3116, 2286, 0 },
		{ 3159, 2961, 0 },
		{ 3160, 1681, 0 },
		{ 3062, 2595, 0 },
		{ 3162, 2238, 0 },
		{ 3116, 2343, 0 },
		{ 3165, 4452, 0 },
		{ 0, 0, 159 },
		{ 3112, 3350, 185 },
		{ 0, 0, 185 },
		{ 3125, 1974, 0 },
		{ 3147, 2901, 0 },
		{ 3164, 2094, 0 },
		{ 1330, 4886, 0 },
		{ 3159, 2670, 0 },
		{ 3155, 2489, 0 },
		{ 3158, 3015, 0 },
		{ 3125, 1979, 0 },
		{ 3125, 1997, 0 },
		{ 3147, 2860, 0 },
		{ 3125, 1999, 0 },
		{ 3165, 4524, 0 },
		{ 0, 0, 184 },
		{ 2172, 4042, 145 },
		{ 0, 0, 145 },
		{ 0, 0, 146 },
		{ 3147, 2863, 0 },
		{ 3116, 2345, 0 },
		{ 3162, 2199, 0 },
		{ 3149, 2398, 0 },
		{ 3147, 2920, 0 },
		{ 3120, 4334, 0 },
		{ 3155, 2497, 0 },
		{ 3158, 3003, 0 },
		{ 3125, 2008, 0 },
		{ 3125, 2016, 0 },
		{ 3127, 4739, 0 },
		{ 3127, 4733, 0 },
		{ 3062, 2601, 0 },
		{ 3116, 2300, 0 },
		{ 3062, 2701, 0 },
		{ 3160, 1711, 0 },
		{ 3062, 2733, 0 },
		{ 3158, 3002, 0 },
		{ 3155, 2439, 0 },
		{ 3062, 2600, 0 },
		{ 3125, 1555, 0 },
		{ 3147, 2914, 0 },
		{ 3164, 2108, 0 },
		{ 3165, 4522, 0 },
		{ 0, 0, 143 },
		{ 2673, 2990, 24 },
		{ 0, 0, 24 },
		{ 0, 0, 25 },
		{ 3147, 2925, 0 },
		{ 2964, 2763, 0 },
		{ 3062, 2688, 0 },
		{ 3116, 2268, 0 },
		{ 3119, 2, 0 },
		{ 3162, 2182, 0 },
		{ 2922, 2147, 0 },
		{ 3147, 2844, 0 },
		{ 3164, 2114, 0 },
		{ 3158, 3037, 0 },
		{ 3160, 1865, 0 },
		{ 3162, 2230, 0 },
		{ 3164, 2057, 0 },
		{ 3119, 4983, 0 },
		{ 3144, 2945, 0 },
		{ 3147, 2871, 0 },
		{ 3125, 2043, 0 },
		{ 3159, 2962, 0 },
		{ 3164, 2073, 0 },
		{ 3062, 2711, 0 },
		{ 2922, 2141, 0 },
		{ 3160, 1933, 0 },
		{ 3062, 2529, 0 },
		{ 3162, 2217, 0 },
		{ 3116, 2297, 0 },
		{ 3119, 7, 0 },
		{ 3127, 4742, 0 },
		{ 0, 0, 21 },
		{ 1413, 0, 1 },
		{ 1413, 0, 186 },
		{ 1413, 2721, 236 },
		{ 1628, 172, 236 },
		{ 1628, 410, 236 },
		{ 1628, 398, 236 },
		{ 1628, 528, 236 },
		{ 1628, 403, 236 },
		{ 1628, 414, 236 },
		{ 1628, 389, 236 },
		{ 1628, 420, 236 },
		{ 1628, 484, 236 },
		{ 1413, 0, 236 },
		{ 1425, 2522, 236 },
		{ 1413, 2803, 236 },
		{ 2673, 2993, 232 },
		{ 1628, 505, 236 },
		{ 1628, 504, 236 },
		{ 1628, 533, 236 },
		{ 1628, 0, 236 },
		{ 1628, 568, 236 },
		{ 1628, 555, 236 },
		{ 3162, 2240, 0 },
		{ 0, 0, 187 },
		{ 3116, 2294, 0 },
		{ 1628, 522, 0 },
		{ 1628, 0, 0 },
		{ 3119, 3989, 0 },
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
		{ 3147, 2839, 0 },
		{ 3147, 2842, 0 },
		{ 1629, 609, 0 },
		{ 1629, 610, 0 },
		{ 1628, 620, 0 },
		{ 1628, 621, 0 },
		{ 1628, 612, 0 },
		{ 3162, 2187, 0 },
		{ 3144, 2936, 0 },
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
		{ 3116, 2316, 0 },
		{ 3008, 2786, 0 },
		{ 1628, 699, 0 },
		{ 1628, 689, 0 },
		{ 1629, 18, 0 },
		{ 1628, 22, 0 },
		{ 1628, 29, 0 },
		{ 3155, 2451, 0 },
		{ 0, 0, 235 },
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
		{ 1628, 0, 221 },
		{ 1628, 89, 0 },
		{ 3162, 2236, 0 },
		{ 3062, 2723, 0 },
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
		{ 1628, 0, 220 },
		{ 1628, 157, 0 },
		{ 3149, 2410, 0 },
		{ 3116, 2274, 0 },
		{ 1628, 161, 0 },
		{ 1628, 171, 0 },
		{ 1628, 169, 0 },
		{ 1628, 0, 234 },
		{ 1628, 168, 0 },
		{ 0, 0, 222 },
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
		{ 3147, 2877, 0 },
		{ 3147, 2888, 0 },
		{ 1628, 0, 224 },
		{ 1628, 301, 225 },
		{ 1628, 271, 0 },
		{ 1628, 274, 0 },
		{ 1628, 301, 0 },
		{ 1528, 3577, 0 },
		{ 3119, 4263, 0 },
		{ 2213, 4691, 211 },
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
		{ 3120, 4335, 0 },
		{ 3119, 4999, 227 },
		{ 1628, 330, 0 },
		{ 1628, 343, 0 },
		{ 1628, 358, 0 },
		{ 1628, 374, 0 },
		{ 0, 0, 191 },
		{ 1630, 31, -7 },
		{ 1630, 117, -10 },
		{ 1630, 231, -13 },
		{ 1630, 345, -16 },
		{ 1630, 376, -19 },
		{ 1630, 460, -22 },
		{ 1628, 406, 0 },
		{ 1628, 419, 0 },
		{ 1628, 392, 0 },
		{ 1628, 0, 209 },
		{ 1628, 0, 223 },
		{ 3155, 2435, 0 },
		{ 1628, 391, 0 },
		{ 1628, 381, 0 },
		{ 1628, 386, 0 },
		{ 1629, 386, 0 },
		{ 1565, 3546, 0 },
		{ 3119, 4295, 0 },
		{ 2213, 4707, 212 },
		{ 1568, 3547, 0 },
		{ 3119, 4259, 0 },
		{ 2213, 4688, 213 },
		{ 1571, 3548, 0 },
		{ 3119, 4289, 0 },
		{ 2213, 4709, 216 },
		{ 1574, 3549, 0 },
		{ 3119, 4325, 0 },
		{ 2213, 4700, 217 },
		{ 1577, 3550, 0 },
		{ 3119, 4257, 0 },
		{ 2213, 4711, 218 },
		{ 1580, 3551, 0 },
		{ 3119, 4261, 0 },
		{ 2213, 4694, 219 },
		{ 1628, 436, 0 },
		{ 1630, 488, -25 },
		{ 1628, 423, 0 },
		{ 3158, 3050, 0 },
		{ 1628, 404, 0 },
		{ 1628, 481, 0 },
		{ 1628, 443, 0 },
		{ 1628, 493, 0 },
		{ 0, 0, 193 },
		{ 0, 0, 195 },
		{ 0, 0, 201 },
		{ 0, 0, 203 },
		{ 0, 0, 205 },
		{ 0, 0, 207 },
		{ 1630, 490, -28 },
		{ 1598, 3561, 0 },
		{ 3119, 4293, 0 },
		{ 2213, 4684, 215 },
		{ 1628, 0, 208 },
		{ 3125, 2054, 0 },
		{ 1628, 481, 0 },
		{ 1628, 496, 0 },
		{ 1629, 489, 0 },
		{ 1628, 486, 0 },
		{ 1607, 3568, 0 },
		{ 3119, 4265, 0 },
		{ 2213, 4687, 214 },
		{ 0, 0, 199 },
		{ 3125, 2000, 0 },
		{ 1628, 4, 230 },
		{ 1629, 493, 0 },
		{ 1628, 3, 233 },
		{ 1628, 510, 0 },
		{ 0, 0, 197 },
		{ 3127, 4740, 0 },
		{ 3127, 4741, 0 },
		{ 1628, 498, 0 },
		{ 0, 0, 231 },
		{ 1628, 495, 0 },
		{ 3127, 4736, 0 },
		{ 0, 0, 229 },
		{ 1628, 506, 0 },
		{ 1628, 511, 0 },
		{ 0, 0, 228 },
		{ 1628, 516, 0 },
		{ 1628, 519, 0 },
		{ 1629, 521, 226 },
		{ 1630, 925, 0 },
		{ 1631, 736, -1 },
		{ 1632, 3592, 0 },
		{ 3119, 4249, 0 },
		{ 2213, 4714, 210 },
		{ 0, 0, 189 },
		{ 2172, 4047, 279 },
		{ 0, 0, 279 },
		{ 3147, 2921, 0 },
		{ 3116, 2322, 0 },
		{ 3162, 2213, 0 },
		{ 3149, 2392, 0 },
		{ 3147, 2811, 0 },
		{ 3120, 4337, 0 },
		{ 3155, 2436, 0 },
		{ 3158, 3011, 0 },
		{ 3125, 2002, 0 },
		{ 3125, 2005, 0 },
		{ 3127, 4721, 0 },
		{ 3127, 4724, 0 },
		{ 3062, 2712, 0 },
		{ 3116, 2266, 0 },
		{ 3062, 2722, 0 },
		{ 3160, 1836, 0 },
		{ 3062, 2732, 0 },
		{ 3158, 3007, 0 },
		{ 3155, 2487, 0 },
		{ 3062, 2519, 0 },
		{ 3125, 1561, 0 },
		{ 3147, 2874, 0 },
		{ 3164, 2128, 0 },
		{ 3165, 4536, 0 },
		{ 0, 0, 278 },
		{ 2172, 4041, 281 },
		{ 0, 0, 281 },
		{ 0, 0, 282 },
		{ 3147, 2879, 0 },
		{ 3116, 2285, 0 },
		{ 3162, 2242, 0 },
		{ 3149, 2413, 0 },
		{ 3147, 2912, 0 },
		{ 3120, 4369, 0 },
		{ 3155, 2441, 0 },
		{ 3158, 3027, 0 },
		{ 3125, 2017, 0 },
		{ 3125, 2019, 0 },
		{ 3127, 4725, 0 },
		{ 3127, 4729, 0 },
		{ 3159, 2982, 0 },
		{ 3164, 2072, 0 },
		{ 3162, 2189, 0 },
		{ 3125, 2021, 0 },
		{ 3125, 2022, 0 },
		{ 3162, 2206, 0 },
		{ 3129, 1473, 0 },
		{ 3147, 2826, 0 },
		{ 3164, 2087, 0 },
		{ 3165, 4464, 0 },
		{ 0, 0, 280 },
		{ 2172, 4048, 284 },
		{ 0, 0, 284 },
		{ 0, 0, 285 },
		{ 3147, 2831, 0 },
		{ 3116, 2350, 0 },
		{ 3162, 2225, 0 },
		{ 3149, 2411, 0 },
		{ 3147, 2855, 0 },
		{ 3120, 4345, 0 },
		{ 3155, 2467, 0 },
		{ 3158, 3009, 0 },
		{ 3125, 2032, 0 },
		{ 3125, 2038, 0 },
		{ 3127, 4737, 0 },
		{ 3127, 4738, 0 },
		{ 3149, 2394, 0 },
		{ 3152, 1521, 0 },
		{ 3160, 1948, 0 },
		{ 3158, 3049, 0 },
		{ 3160, 1957, 0 },
		{ 3162, 2170, 0 },
		{ 3164, 2123, 0 },
		{ 3165, 4670, 0 },
		{ 0, 0, 283 },
		{ 2172, 4052, 287 },
		{ 0, 0, 287 },
		{ 0, 0, 288 },
		{ 3147, 2897, 0 },
		{ 3116, 2296, 0 },
		{ 3162, 2183, 0 },
		{ 3149, 2400, 0 },
		{ 3147, 2913, 0 },
		{ 3120, 4368, 0 },
		{ 3155, 2474, 0 },
		{ 3158, 3028, 0 },
		{ 3125, 2044, 0 },
		{ 3125, 2048, 0 },
		{ 3127, 4730, 0 },
		{ 3127, 4731, 0 },
		{ 3147, 2924, 0 },
		{ 3129, 1440, 0 },
		{ 3158, 3065, 0 },
		{ 3155, 2495, 0 },
		{ 3152, 1519, 0 },
		{ 3158, 3005, 0 },
		{ 3160, 1806, 0 },
		{ 3162, 2203, 0 },
		{ 3164, 2075, 0 },
		{ 3165, 4608, 0 },
		{ 0, 0, 286 },
		{ 2172, 4035, 290 },
		{ 0, 0, 290 },
		{ 0, 0, 291 },
		{ 3147, 2822, 0 },
		{ 3116, 2352, 0 },
		{ 3162, 2212, 0 },
		{ 3149, 2412, 0 },
		{ 3147, 2832, 0 },
		{ 3120, 4343, 0 },
		{ 3155, 2463, 0 },
		{ 3158, 3046, 0 },
		{ 3125, 1976, 0 },
		{ 3125, 1978, 0 },
		{ 3127, 4719, 0 },
		{ 3127, 4720, 0 },
		{ 3162, 2219, 0 },
		{ 2922, 2162, 0 },
		{ 3160, 1826, 0 },
		{ 3062, 2523, 0 },
		{ 3149, 2402, 0 },
		{ 3062, 2588, 0 },
		{ 3125, 1984, 0 },
		{ 3147, 2875, 0 },
		{ 3164, 2099, 0 },
		{ 3165, 4676, 0 },
		{ 0, 0, 289 },
		{ 2995, 4424, 165 },
		{ 0, 0, 165 },
		{ 0, 0, 166 },
		{ 3008, 2797, 0 },
		{ 3160, 1828, 0 },
		{ 3147, 2894, 0 },
		{ 3164, 2106, 0 },
		{ 1772, 4878, 0 },
		{ 3147, 2518, 0 },
		{ 3129, 1442, 0 },
		{ 3147, 2904, 0 },
		{ 3164, 2113, 0 },
		{ 2772, 1406, 0 },
		{ 3160, 1845, 0 },
		{ 3154, 2753, 0 },
		{ 3062, 2714, 0 },
		{ 3116, 2338, 0 },
		{ 2964, 2764, 0 },
		{ 1783, 4892, 0 },
		{ 3147, 2531, 0 },
		{ 3155, 2481, 0 },
		{ 3125, 2003, 0 },
		{ 3147, 2812, 0 },
		{ 1788, 4809, 0 },
		{ 3157, 2468, 0 },
		{ 3152, 1643, 0 },
		{ 3116, 2349, 0 },
		{ 3159, 2958, 0 },
		{ 3160, 1881, 0 },
		{ 3062, 2582, 0 },
		{ 3162, 2197, 0 },
		{ 3116, 2262, 0 },
		{ 3165, 4664, 0 },
		{ 0, 0, 163 },
		{ 2172, 4051, 272 },
		{ 0, 0, 272 },
		{ 3147, 2835, 0 },
		{ 3116, 2263, 0 },
		{ 3162, 2198, 0 },
		{ 3149, 2405, 0 },
		{ 3147, 2853, 0 },
		{ 3120, 4350, 0 },
		{ 3155, 2459, 0 },
		{ 3158, 3019, 0 },
		{ 3125, 2011, 0 },
		{ 3125, 2013, 0 },
		{ 3127, 4727, 0 },
		{ 3127, 4728, 0 },
		{ 3144, 2948, 0 },
		{ 3062, 2710, 0 },
		{ 3125, 2014, 0 },
		{ 2922, 2138, 0 },
		{ 3155, 2477, 0 },
		{ 3158, 3061, 0 },
		{ 2772, 1404, 0 },
		{ 3165, 4561, 0 },
		{ 0, 0, 270 },
		{ 1836, 0, 1 },
		{ 1995, 2839, 387 },
		{ 3147, 2881, 387 },
		{ 3158, 2934, 387 },
		{ 3144, 2175, 387 },
		{ 1836, 0, 354 },
		{ 1836, 2801, 387 },
		{ 3154, 1486, 387 },
		{ 2930, 4376, 387 },
		{ 2151, 3723, 387 },
		{ 3112, 3328, 387 },
		{ 2151, 3725, 387 },
		{ 2136, 4140, 387 },
		{ 3164, 1973, 387 },
		{ 1836, 0, 387 },
		{ 2673, 2991, 385 },
		{ 3158, 2775, 387 },
		{ 3158, 3041, 387 },
		{ 0, 0, 387 },
		{ 3162, 2235, 0 },
		{ -1841, 20, 344 },
		{ -1842, 4817, 0 },
		{ 3116, 2309, 0 },
		{ 0, 0, 350 },
		{ 0, 0, 351 },
		{ 3155, 2437, 0 },
		{ 3062, 2579, 0 },
		{ 3147, 2919, 0 },
		{ 0, 0, 355 },
		{ 3116, 2312, 0 },
		{ 3164, 2089, 0 },
		{ 3062, 2593, 0 },
		{ 2104, 3113, 0 },
		{ 3108, 3689, 0 },
		{ 3115, 3376, 0 },
		{ 2044, 3302, 0 },
		{ 3108, 3694, 0 },
		{ 3152, 1625, 0 },
		{ 3125, 2023, 0 },
		{ 3116, 2334, 0 },
		{ 3160, 1650, 0 },
		{ 3164, 2102, 0 },
		{ 3162, 2171, 0 },
		{ 3069, 4793, 0 },
		{ 3162, 2175, 0 },
		{ 3125, 2029, 0 },
		{ 3160, 1653, 0 },
		{ 3116, 2261, 0 },
		{ 3144, 2953, 0 },
		{ 3164, 2112, 0 },
		{ 3155, 2492, 0 },
		{ 2172, 4036, 0 },
		{ 2104, 3129, 0 },
		{ 2104, 3130, 0 },
		{ 2136, 4182, 0 },
		{ 2066, 3889, 0 },
		{ 3147, 2834, 0 },
		{ 3125, 2037, 0 },
		{ 3144, 2950, 0 },
		{ 3152, 1626, 0 },
		{ 3147, 2840, 0 },
		{ 3155, 2429, 0 },
		{ 0, 17, 347 },
		{ 3149, 2396, 0 },
		{ 3147, 2848, 0 },
		{ 2151, 3708, 0 },
		{ 3160, 1699, 0 },
		{ 0, 0, 386 },
		{ 3147, 2854, 0 },
		{ 3144, 2935, 0 },
		{ 2136, 4099, 0 },
		{ 2082, 3468, 0 },
		{ 3108, 3679, 0 },
		{ 3086, 3562, 0 },
		{ 2104, 3144, 0 },
		{ 0, 0, 375 },
		{ 3120, 4339, 0 },
		{ 3162, 2194, 0 },
		{ 3164, 2119, 0 },
		{ 3116, 2282, 0 },
		{ -1918, 1092, 0 },
		{ 0, 0, 346 },
		{ 3147, 2866, 0 },
		{ 0, 0, 374 },
		{ 2922, 2146, 0 },
		{ 3062, 2587, 0 },
		{ 3116, 2288, 0 },
		{ 1933, 4756, 0 },
		{ 3085, 3831, 0 },
		{ 3036, 3992, 0 },
		{ 3086, 3573, 0 },
		{ 2104, 3153, 0 },
		{ 3108, 3692, 0 },
		{ 3162, 2201, 0 },
		{ 3149, 2380, 0 },
		{ 3116, 2293, 0 },
		{ 3160, 1712, 0 },
		{ 0, 0, 376 },
		{ 3120, 4360, 353 },
		{ 3160, 1714, 0 },
		{ 3159, 2960, 0 },
		{ 3160, 1717, 0 },
		{ 0, 0, 379 },
		{ 0, 0, 380 },
		{ 1938, 0, -41 },
		{ 2117, 3259, 0 },
		{ 2151, 3739, 0 },
		{ 3108, 3705, 0 },
		{ 2136, 4128, 0 },
		{ 3062, 2687, 0 },
		{ 0, 0, 378 },
		{ 0, 0, 384 },
		{ 0, 4758, 0 },
		{ 3155, 2486, 0 },
		{ 3125, 2049, 0 },
		{ 3158, 3013, 0 },
		{ 2172, 4046, 0 },
		{ 3119, 4339, 0 },
		{ 2213, 4713, 369 },
		{ 2136, 4135, 0 },
		{ 2930, 4378, 0 },
		{ 3086, 3588, 0 },
		{ 3086, 3589, 0 },
		{ 3116, 2305, 0 },
		{ 0, 0, 381 },
		{ 0, 0, 382 },
		{ 3158, 3017, 0 },
		{ 2219, 4800, 0 },
		{ 3155, 2490, 0 },
		{ 3147, 2911, 0 },
		{ 0, 0, 359 },
		{ 1961, 0, -44 },
		{ 1963, 0, -47 },
		{ 2151, 3752, 0 },
		{ 3120, 4336, 0 },
		{ 0, 0, 377 },
		{ 3125, 2051, 0 },
		{ 0, 0, 352 },
		{ 2172, 4033, 0 },
		{ 3116, 2310, 0 },
		{ 3119, 4321, 0 },
		{ 2213, 4693, 370 },
		{ 3119, 4323, 0 },
		{ 2213, 4698, 371 },
		{ 2930, 4373, 0 },
		{ 1973, 0, -77 },
		{ 3125, 2052, 0 },
		{ 3147, 2915, 0 },
		{ 3147, 2918, 0 },
		{ 0, 0, 361 },
		{ 0, 0, 363 },
		{ 1978, 0, -35 },
		{ 3119, 4245, 0 },
		{ 2213, 4686, 373 },
		{ 0, 0, 349 },
		{ 3116, 2313, 0 },
		{ 3164, 2078, 0 },
		{ 3119, 4251, 0 },
		{ 2213, 4692, 372 },
		{ 0, 0, 367 },
		{ 3162, 2226, 0 },
		{ 3158, 3064, 0 },
		{ 0, 0, 365 },
		{ 3149, 2390, 0 },
		{ 3160, 1751, 0 },
		{ 3147, 2926, 0 },
		{ 3062, 2737, 0 },
		{ 0, 0, 383 },
		{ 3162, 2232, 0 },
		{ 3116, 2337, 0 },
		{ 1992, 0, -50 },
		{ 3119, 4291, 0 },
		{ 2213, 4685, 368 },
		{ 0, 0, 357 },
		{ 1836, 2832, 387 },
		{ 1999, 2524, 387 },
		{ -1997, 5010, 344 },
		{ -1998, 4814, 0 },
		{ 3119, 4801, 0 },
		{ 3069, 4781, 0 },
		{ 0, 0, 345 },
		{ 3069, 4795, 0 },
		{ -2003, 5008, 0 },
		{ -2004, 4809, 0 },
		{ 2007, 2, 347 },
		{ 3069, 4796, 0 },
		{ 3119, 4952, 0 },
		{ 0, 0, 348 },
		{ 2025, 0, 1 },
		{ 2221, 2849, 343 },
		{ 3147, 2829, 343 },
		{ 2025, 0, 297 },
		{ 2025, 2720, 343 },
		{ 3108, 3693, 343 },
		{ 2025, 0, 300 },
		{ 3152, 1557, 343 },
		{ 2930, 4371, 343 },
		{ 2151, 3718, 343 },
		{ 3112, 3285, 343 },
		{ 2151, 3720, 343 },
		{ 2136, 4180, 343 },
		{ 3158, 3044, 343 },
		{ 3164, 1974, 343 },
		{ 2025, 0, 343 },
		{ 2673, 2984, 340 },
		{ 3158, 3053, 343 },
		{ 3144, 2940, 343 },
		{ 2930, 4385, 343 },
		{ 3158, 1641, 343 },
		{ 0, 0, 343 },
		{ 3162, 2244, 0 },
		{ -2032, 22, 292 },
		{ -2033, 4810, 0 },
		{ 3116, 2355, 0 },
		{ 0, 0, 298 },
		{ 3116, 2356, 0 },
		{ 3162, 2167, 0 },
		{ 3164, 2100, 0 },
		{ 2104, 3106, 0 },
		{ 3108, 3663, 0 },
		{ 3115, 3390, 0 },
		{ 3085, 3789, 0 },
		{ 0, 3275, 0 },
		{ 0, 3296, 0 },
		{ 3108, 3667, 0 },
		{ 3155, 2488, 0 },
		{ 3152, 1571, 0 },
		{ 3125, 1986, 0 },
		{ 3116, 2267, 0 },
		{ 3147, 2856, 0 },
		{ 3147, 2859, 0 },
		{ 3162, 2176, 0 },
		{ 2219, 4804, 0 },
		{ 3162, 2179, 0 },
		{ 3069, 4784, 0 },
		{ 3162, 2181, 0 },
		{ 3144, 2937, 0 },
		{ 2922, 2158, 0 },
		{ 3164, 2104, 0 },
		{ 2172, 4054, 0 },
		{ 2104, 3119, 0 },
		{ 2104, 3120, 0 },
		{ 3036, 4004, 0 },
		{ 3036, 4005, 0 },
		{ 2136, 4120, 0 },
		{ 0, 3890, 0 },
		{ 3125, 1988, 0 },
		{ 3147, 2873, 0 },
		{ 3125, 1991, 0 },
		{ 3144, 2952, 0 },
		{ 3116, 2290, 0 },
		{ 3125, 1992, 0 },
		{ 3155, 2447, 0 },
		{ 0, 0, 342 },
		{ 3155, 2449, 0 },
		{ 0, 0, 294 },
		{ 3149, 2386, 0 },
		{ 0, 0, 339 },
		{ 3152, 1617, 0 },
		{ 3147, 2891, 0 },
		{ 2136, 4133, 0 },
		{ 0, 3511, 0 },
		{ 3108, 3695, 0 },
		{ 2085, 3860, 0 },
		{ 0, 3833, 0 },
		{ 3086, 3590, 0 },
		{ 2104, 3132, 0 },
		{ 3147, 2893, 0 },
		{ 0, 0, 332 },
		{ 3120, 4342, 0 },
		{ 3162, 2191, 0 },
		{ 3160, 1850, 0 },
		{ 3160, 1856, 0 },
		{ 3152, 1624, 0 },
		{ -2112, 1167, 0 },
		{ 3147, 2910, 0 },
		{ 3155, 2479, 0 },
		{ 3116, 2306, 0 },
		{ 3085, 3773, 0 },
		{ 3036, 3950, 0 },
		{ 3086, 3601, 0 },
		{ 3036, 3952, 0 },
		{ 3036, 3954, 0 },
		{ 0, 3142, 0 },
		{ 3108, 3659, 0 },
		{ 0, 0, 331 },
		{ 3162, 2200, 0 },
		{ 3149, 2406, 0 },
		{ 3062, 2603, 0 },
		{ 0, 0, 338 },
		{ 3158, 3038, 0 },
		{ 0, 0, 333 },
		{ 0, 0, 296 },
		{ 3158, 3040, 0 },
		{ 3160, 1885, 0 },
		{ 2129, 0, -74 },
		{ 0, 3258, 0 },
		{ 2151, 3727, 0 },
		{ 2120, 3242, 0 },
		{ 2117, 3241, 0 },
		{ 3108, 3670, 0 },
		{ 2136, 4167, 0 },
		{ 3062, 2665, 0 },
		{ 0, 0, 335 },
		{ 3159, 2980, 0 },
		{ 3160, 1889, 0 },
		{ 3160, 1926, 0 },
		{ 2172, 4040, 0 },
		{ 3119, 4253, 0 },
		{ 2213, 4679, 322 },
		{ 2136, 4173, 0 },
		{ 2930, 4379, 0 },
		{ 2136, 4174, 0 },
		{ 2136, 4175, 0 },
		{ 2136, 4176, 0 },
		{ 0, 4177, 0 },
		{ 3086, 3616, 0 },
		{ 3086, 3617, 0 },
		{ 3116, 2314, 0 },
		{ 3158, 3060, 0 },
		{ 3062, 2693, 0 },
		{ 3062, 2697, 0 },
		{ 3147, 2929, 0 },
		{ 0, 0, 304 },
		{ 2158, 0, -53 },
		{ 2160, 0, -56 },
		{ 2162, 0, -62 },
		{ 2164, 0, -65 },
		{ 2166, 0, -68 },
		{ 2168, 0, -71 },
		{ 0, 3742, 0 },
		{ 3120, 4349, 0 },
		{ 0, 0, 334 },
		{ 3155, 2493, 0 },
		{ 3162, 2209, 0 },
		{ 3162, 2210, 0 },
		{ 3116, 2323, 0 },
		{ 3119, 4327, 0 },
		{ 2213, 4695, 323 },
		{ 3119, 4329, 0 },
		{ 2213, 4699, 324 },
		{ 3119, 4331, 0 },
		{ 2213, 4701, 327 },
		{ 3119, 4333, 0 },
		{ 2213, 4708, 328 },
		{ 3119, 4335, 0 },
		{ 2213, 4710, 329 },
		{ 3119, 4337, 0 },
		{ 2213, 4712, 330 },
		{ 2930, 4382, 0 },
		{ 2183, 0, -80 },
		{ 0, 4053, 0 },
		{ 3116, 2327, 0 },
		{ 3116, 2328, 0 },
		{ 3147, 2818, 0 },
		{ 0, 0, 306 },
		{ 0, 0, 308 },
		{ 0, 0, 314 },
		{ 0, 0, 316 },
		{ 0, 0, 318 },
		{ 0, 0, 320 },
		{ 2189, 0, -38 },
		{ 3119, 4247, 0 },
		{ 2213, 4690, 326 },
		{ 3147, 2820, 0 },
		{ 3158, 3014, 0 },
		{ 3094, 3235, 337 },
		{ 3164, 2063, 0 },
		{ 3119, 4255, 0 },
		{ 2213, 4696, 325 },
		{ 0, 0, 312 },
		{ 3116, 2333, 0 },
		{ 3164, 2070, 0 },
		{ 0, 0, 299 },
		{ 3158, 3024, 0 },
		{ 0, 0, 310 },
		{ 3162, 2215, 0 },
		{ 2772, 1393, 0 },
		{ 3160, 1937, 0 },
		{ 3149, 2409, 0 },
		{ 2995, 4574, 0 },
		{ 3062, 2521, 0 },
		{ 3147, 2838, 0 },
		{ 3155, 2457, 0 },
		{ 3162, 2220, 0 },
		{ 0, 0, 336 },
		{ 2964, 2756, 0 },
		{ 3116, 2347, 0 },
		{ 3162, 2222, 0 },
		{ 2212, 0, -59 },
		{ 3164, 2074, 0 },
		{ 3119, 4319, 0 },
		{ 0, 4689, 321 },
		{ 3062, 2586, 0 },
		{ 0, 0, 302 },
		{ 3160, 1939, 0 },
		{ 3154, 2749, 0 },
		{ 3149, 2382, 0 },
		{ 0, 4803, 0 },
		{ 0, 0, 341 },
		{ 2025, 2859, 343 },
		{ 2225, 2520, 343 },
		{ -2223, 21, 292 },
		{ -2224, 1, 0 },
		{ 3119, 4800, 0 },
		{ 3069, 4773, 0 },
		{ 0, 0, 293 },
		{ 3069, 4797, 0 },
		{ -2229, 5011, 0 },
		{ -2230, 4812, 0 },
		{ 2233, 0, 294 },
		{ 3069, 4798, 0 },
		{ 3119, 4972, 0 },
		{ 0, 0, 295 },
		{ 0, 4247, 389 },
		{ 0, 0, 389 },
		{ 3147, 2865, 0 },
		{ 3008, 2778, 0 },
		{ 3158, 3008, 0 },
		{ 3152, 1489, 0 },
		{ 3155, 2484, 0 },
		{ 3160, 1950, 0 },
		{ 3119, 4979, 0 },
		{ 3164, 2086, 0 },
		{ 3152, 1490, 0 },
		{ 3116, 2264, 0 },
		{ 2248, 4835, 0 },
		{ 3119, 1966, 0 },
		{ 3158, 3023, 0 },
		{ 3164, 2090, 0 },
		{ 3158, 3026, 0 },
		{ 3149, 2404, 0 },
		{ 3147, 2887, 0 },
		{ 3160, 1651, 0 },
		{ 3147, 2890, 0 },
		{ 3164, 2098, 0 },
		{ 3125, 2025, 0 },
		{ 3165, 4600, 0 },
		{ 0, 0, 388 },
		{ 3069, 4770, 438 },
		{ 0, 0, 394 },
		{ 0, 0, 396 },
		{ 2280, 833, 429 },
		{ 2456, 846, 429 },
		{ 2479, 844, 429 },
		{ 2422, 845, 429 },
		{ 2281, 859, 429 },
		{ 2279, 849, 429 },
		{ 2479, 845, 429 },
		{ 2302, 859, 429 },
		{ 2452, 861, 429 },
		{ 2452, 863, 429 },
		{ 2456, 860, 429 },
		{ 2396, 870, 429 },
		{ 2278, 888, 429 },
		{ 3147, 1645, 428 },
		{ 2310, 2652, 438 },
		{ 2514, 860, 429 },
		{ 2456, 873, 429 },
		{ 2314, 874, 429 },
		{ 2456, 868, 429 },
		{ 3147, 2808, 438 },
		{ -2283, 8, 390 },
		{ -2284, 4813, 0 },
		{ 2514, 865, 429 },
		{ 2519, 484, 429 },
		{ 2514, 870, 429 },
		{ 2363, 868, 429 },
		{ 2456, 876, 429 },
		{ 2462, 871, 429 },
		{ 2456, 878, 429 },
		{ 2396, 887, 429 },
		{ 2367, 877, 429 },
		{ 2422, 879, 429 },
		{ 2396, 901, 429 },
		{ 2479, 885, 429 },
		{ 2278, 878, 429 },
		{ 2424, 885, 429 },
		{ 2311, 899, 429 },
		{ 2278, 883, 429 },
		{ 2462, 920, 429 },
		{ 2278, 930, 429 },
		{ 2492, 923, 429 },
		{ 2514, 1190, 429 },
		{ 2492, 952, 429 },
		{ 2367, 955, 429 },
		{ 2519, 576, 429 },
		{ 3147, 1756, 425 },
		{ 2341, 1467, 0 },
		{ 3147, 1789, 426 },
		{ 2492, 965, 429 },
		{ 3116, 2284, 0 },
		{ 3069, 4774, 0 },
		{ 2278, 978, 429 },
		{ 3162, 1979, 0 },
		{ 2452, 976, 429 },
		{ 2352, 961, 429 },
		{ 2492, 969, 429 },
		{ 2424, 964, 429 },
		{ 2424, 965, 429 },
		{ 2367, 1000, 429 },
		{ 2452, 1008, 429 },
		{ 2452, 1009, 429 },
		{ 2479, 997, 429 },
		{ 2422, 995, 429 },
		{ 2452, 1049, 429 },
		{ 2396, 1054, 429 },
		{ 2479, 1038, 429 },
		{ 2519, 578, 429 },
		{ 2519, 580, 429 },
		{ 2486, 1039, 429 },
		{ 2486, 1040, 429 },
		{ 2452, 1055, 429 },
		{ 2352, 1066, 429 },
		{ 2462, 1073, 429 },
		{ 2396, 1088, 429 },
		{ 2440, 1086, 429 },
		{ 3157, 2458, 0 },
		{ 2371, 1400, 0 },
		{ 2341, 0, 0 },
		{ 3071, 2663, 427 },
		{ 2373, 1401, 0 },
		{ 3144, 2954, 0 },
		{ 0, 0, 392 },
		{ 2452, 1087, 429 },
		{ 3008, 2787, 0 },
		{ 2519, 582, 429 },
		{ 2367, 1118, 429 },
		{ 2424, 1111, 429 },
		{ 2519, 10, 429 },
		{ 2456, 1158, 429 },
		{ 2278, 1112, 429 },
		{ 2365, 1131, 429 },
		{ 2519, 123, 429 },
		{ 2424, 1115, 429 },
		{ 2456, 1153, 429 },
		{ 2519, 125, 429 },
		{ 2424, 1145, 429 },
		{ 2396, 1194, 429 },
		{ 3119, 2235, 0 },
		{ 3160, 1851, 0 },
		{ 2486, 1177, 429 },
		{ 2278, 1181, 429 },
		{ 2479, 1180, 429 },
		{ 2278, 1196, 429 },
		{ 2424, 1180, 429 },
		{ 2278, 1189, 429 },
		{ 2278, 1179, 429 },
		{ 3062, 2698, 0 },
		{ 2371, 0, 0 },
		{ 3071, 2695, 425 },
		{ 2373, 0, 0 },
		{ 3071, 2709, 426 },
		{ 0, 0, 430 },
		{ 2479, 1186, 429 },
		{ 2403, 4839, 0 },
		{ 3155, 2087, 0 },
		{ 2396, 1204, 429 },
		{ 2519, 130, 429 },
		{ 3125, 1909, 0 },
		{ 2543, 6, 429 },
		{ 2486, 1187, 429 },
		{ 2396, 1206, 429 },
		{ 2424, 1188, 429 },
		{ 3119, 1960, 0 },
		{ 2519, 235, 429 },
		{ 2422, 1188, 429 },
		{ 3162, 2019, 0 },
		{ 2456, 1202, 429 },
		{ 2424, 1192, 429 },
		{ 3116, 2319, 0 },
		{ 3116, 2321, 0 },
		{ 3164, 2069, 0 },
		{ 2462, 1198, 429 },
		{ 2479, 1196, 429 },
		{ 2278, 1214, 429 },
		{ 2452, 1211, 429 },
		{ 2452, 1212, 429 },
		{ 2519, 237, 429 },
		{ 2456, 1210, 429 },
		{ 3155, 2445, 0 },
		{ 2519, 239, 429 },
		{ 3157, 2189, 0 },
		{ 3062, 2692, 0 },
		{ 2424, 1200, 429 },
		{ 3125, 1915, 0 },
		{ 3160, 1963, 0 },
		{ 3165, 4526, 0 },
		{ 3119, 4870, 400 },
		{ 2514, 1208, 429 },
		{ 2424, 1202, 429 },
		{ 2456, 1218, 429 },
		{ 3162, 2178, 0 },
		{ 3157, 2474, 0 },
		{ 2456, 1215, 429 },
		{ 3008, 2795, 0 },
		{ 2462, 1210, 429 },
		{ 2456, 1217, 429 },
		{ 3062, 2727, 0 },
		{ 3062, 2729, 0 },
		{ 3147, 2928, 0 },
		{ 2278, 1206, 429 },
		{ 2456, 1220, 429 },
		{ 2278, 1210, 429 },
		{ 2519, 241, 429 },
		{ 2519, 243, 429 },
		{ 3164, 1936, 0 },
		{ 2492, 1218, 429 },
		{ 3147, 2814, 0 },
		{ 3162, 2011, 0 },
		{ 3108, 3701, 0 },
		{ 3062, 2578, 0 },
		{ 3149, 2407, 0 },
		{ 2456, 1224, 429 },
		{ 3160, 1838, 0 },
		{ 3158, 3052, 0 },
		{ 2543, 121, 429 },
		{ 2462, 1220, 429 },
		{ 2462, 1221, 429 },
		{ 2278, 1233, 429 },
		{ 2922, 2144, 0 },
		{ 3164, 2064, 0 },
		{ 2492, 1224, 429 },
		{ 2473, 4887, 0 },
		{ 2492, 1225, 429 },
		{ 2462, 1225, 429 },
		{ 3160, 1901, 0 },
		{ 3160, 1917, 0 },
		{ 3147, 2845, 0 },
		{ 2452, 1236, 429 },
		{ 2492, 1228, 429 },
		{ 2278, 1238, 429 },
		{ 3162, 1786, 0 },
		{ 3119, 2239, 0 },
		{ 3147, 2857, 0 },
		{ 2278, 1235, 429 },
		{ 3165, 4660, 0 },
		{ 3008, 2775, 0 },
		{ 3112, 3344, 0 },
		{ 3160, 1944, 0 },
		{ 3062, 2709, 0 },
		{ 2278, 1230, 429 },
		{ 3158, 3032, 0 },
		{ 3160, 1949, 0 },
		{ 3165, 4520, 0 },
		{ 0, 0, 418 },
		{ 2479, 1228, 429 },
		{ 2492, 1233, 429 },
		{ 2519, 245, 429 },
		{ 3152, 1623, 0 },
		{ 3162, 2169, 0 },
		{ 2480, 1242, 429 },
		{ 3119, 1964, 0 },
		{ 2519, 349, 429 },
		{ 2492, 1237, 429 },
		{ 2504, 4850, 0 },
		{ 2505, 4853, 0 },
		{ 2506, 4873, 0 },
		{ 2278, 1234, 429 },
		{ 2278, 1246, 429 },
		{ 2519, 351, 429 },
		{ 3158, 3063, 0 },
		{ 3008, 2790, 0 },
		{ 3125, 2026, 0 },
		{ 3144, 2939, 0 },
		{ 2278, 1236, 429 },
		{ 3119, 4991, 423 },
		{ 2515, 4902, 0 },
		{ 3125, 2030, 0 },
		{ 3116, 2257, 0 },
		{ 3160, 1747, 0 },
		{ 2278, 1242, 429 },
		{ 3160, 1798, 0 },
		{ 3125, 2039, 0 },
		{ 2519, 355, 429 },
		{ 2519, 357, 429 },
		{ 3119, 2355, 0 },
		{ 3155, 2471, 0 },
		{ 3149, 2384, 0 },
		{ 2519, 370, 429 },
		{ 3164, 2065, 0 },
		{ 3119, 1968, 0 },
		{ 2519, 463, 429 },
		{ 3160, 1855, 0 },
		{ 3160, 1903, 0 },
		{ 3144, 2603, 0 },
		{ 2519, 469, 429 },
		{ 2519, 471, 429 },
		{ 3159, 1953, 0 },
		{ 3164, 2080, 0 },
		{ 3008, 2792, 0 },
		{ 3155, 2491, 0 },
		{ 3152, 1518, 0 },
		{ 2519, 1499, 429 },
		{ 3162, 1907, 0 },
		{ 2546, 4847, 0 },
		{ 3147, 2816, 0 },
		{ 3165, 4680, 0 },
		{ 2543, 574, 429 },
		{ 3125, 1983, 0 },
		{ 3165, 4491, 0 },
		{ 3119, 2353, 0 },
		{ 3162, 1984, 0 },
		{ 3147, 2828, 0 },
		{ 3158, 3000, 0 },
		{ 2556, 4865, 0 },
		{ 3162, 1784, 0 },
		{ 3162, 2228, 0 },
		{ 3164, 2091, 0 },
		{ 3119, 1974, 0 },
		{ 3164, 2096, 0 },
		{ 3164, 2097, 0 },
		{ 3147, 2837, 0 },
		{ 3119, 1955, 0 },
		{ 3125, 1911, 0 },
		{ 3125, 1994, 0 },
		{ 3116, 2325, 0 },
		{ 2569, 4887, 0 },
		{ 3147, 2843, 0 },
		{ 3125, 1996, 0 },
		{ 3158, 3021, 0 },
		{ 3159, 2976, 0 },
		{ 2574, 811, 429 },
		{ 3147, 2847, 0 },
		{ 2922, 2139, 0 },
		{ 3165, 4530, 0 },
		{ 3125, 1998, 0 },
		{ 3119, 4928, 398 },
		{ 3125, 1917, 0 },
		{ 3165, 4588, 0 },
		{ 3119, 4948, 407 },
		{ 3162, 2173, 0 },
		{ 2922, 2145, 0 },
		{ 3116, 2339, 0 },
		{ 3160, 1945, 0 },
		{ 3157, 2478, 0 },
		{ 3158, 3043, 0 },
		{ 3008, 2796, 0 },
		{ 2964, 2761, 0 },
		{ 3162, 2177, 0 },
		{ 3164, 2109, 0 },
		{ 3147, 2868, 0 },
		{ 3147, 2869, 0 },
		{ 2922, 2156, 0 },
		{ 3164, 2110, 0 },
		{ 3062, 2663, 0 },
		{ 3129, 1412, 0 },
		{ 3152, 1488, 0 },
		{ 3125, 1919, 0 },
		{ 3116, 2256, 0 },
		{ 2922, 2140, 0 },
		{ 3116, 2259, 0 },
		{ 3147, 2884, 0 },
		{ 3165, 4633, 0 },
		{ 3119, 4910, 421 },
		{ 3116, 2260, 0 },
		{ 3160, 1952, 0 },
		{ 0, 0, 435 },
		{ 3125, 2010, 0 },
		{ 3062, 2703, 0 },
		{ 3119, 4936, 406 },
		{ 3158, 3012, 0 },
		{ 3147, 2892, 0 },
		{ 3062, 2705, 0 },
		{ 3062, 2706, 0 },
		{ 3062, 2707, 0 },
		{ 3164, 2121, 0 },
		{ 3008, 2782, 0 },
		{ 2615, 4874, 0 },
		{ 2673, 2992, 0 },
		{ 3162, 2192, 0 },
		{ 3147, 2906, 0 },
		{ 3147, 2908, 0 },
		{ 3160, 1955, 0 },
		{ 3162, 2195, 0 },
		{ 2607, 1279, 0 },
		{ 2623, 4880, 0 },
		{ 2922, 2150, 0 },
		{ 3159, 2971, 0 },
		{ 3160, 1958, 0 },
		{ 3164, 2059, 0 },
		{ 3144, 2934, 0 },
		{ 2629, 4760, 0 },
		{ 3147, 2916, 0 },
		{ 3062, 2725, 0 },
		{ 2632, 4783, 0 },
		{ 0, 1326, 0 },
		{ 3155, 2461, 0 },
		{ 3164, 2062, 0 },
		{ 3160, 1965, 0 },
		{ 3162, 2208, 0 },
		{ 3155, 2473, 0 },
		{ 3147, 2927, 0 },
		{ 3125, 2018, 0 },
		{ 3119, 2775, 0 },
		{ 3158, 2997, 0 },
		{ 2673, 2985, 0 },
		{ 2644, 4851, 0 },
		{ 2645, 4854, 0 },
		{ 3154, 2742, 0 },
		{ 2673, 2987, 0 },
		{ 3147, 2803, 0 },
		{ 3125, 1921, 0 },
		{ 3155, 2476, 0 },
		{ 3164, 2067, 0 },
		{ 3125, 2020, 0 },
		{ 3062, 2580, 0 },
		{ 2654, 4865, 0 },
		{ 3162, 1999, 0 },
		{ 3164, 2071, 0 },
		{ 3149, 2403, 0 },
		{ 3159, 2672, 0 },
		{ 3147, 2819, 0 },
		{ 3165, 4450, 0 },
		{ 3158, 3018, 0 },
		{ 3162, 2218, 0 },
		{ 3116, 2301, 0 },
		{ 3147, 2825, 0 },
		{ 3116, 2303, 0 },
		{ 2922, 2149, 0 },
		{ 3152, 1523, 0 },
		{ 2673, 2989, 0 },
		{ 3158, 3029, 0 },
		{ 3144, 2597, 0 },
		{ 3144, 2599, 0 },
		{ 2672, 4840, 0 },
		{ 3158, 3035, 0 },
		{ 3165, 4602, 0 },
		{ 3160, 1665, 0 },
		{ 3162, 2223, 0 },
		{ 3062, 2684, 0 },
		{ 2678, 4806, 0 },
		{ 3116, 2311, 0 },
		{ 3149, 2037, 0 },
		{ 2922, 2161, 0 },
		{ 3158, 3045, 0 },
		{ 3062, 2689, 0 },
		{ 3158, 3048, 0 },
		{ 3165, 4444, 0 },
		{ 3119, 4946, 419 },
		{ 3160, 1682, 0 },
		{ 3164, 2076, 0 },
		{ 3165, 4456, 0 },
		{ 3165, 4458, 0 },
		{ 3160, 1683, 0 },
		{ 3164, 2079, 0 },
		{ 3008, 2781, 0 },
		{ 3062, 2699, 0 },
		{ 2673, 2986, 0 },
		{ 3147, 2850, 0 },
		{ 3147, 2852, 0 },
		{ 3165, 4532, 0 },
		{ 0, 2988, 0 },
		{ 3119, 4980, 405 },
		{ 3158, 2998, 0 },
		{ 3160, 1684, 0 },
		{ 2922, 2143, 0 },
		{ 3162, 2005, 0 },
		{ 2964, 2765, 0 },
		{ 3162, 2241, 0 },
		{ 3147, 2858, 0 },
		{ 3160, 1701, 0 },
		{ 3125, 2033, 0 },
		{ 3125, 2034, 0 },
		{ 3119, 4865, 399 },
		{ 3162, 2168, 0 },
		{ 3125, 2035, 0 },
		{ 3119, 4874, 411 },
		{ 3119, 4880, 412 },
		{ 3125, 2036, 0 },
		{ 3062, 2715, 0 },
		{ 3008, 2776, 0 },
		{ 3155, 2465, 0 },
		{ 3062, 2720, 0 },
		{ 2922, 2152, 0 },
		{ 2922, 2153, 0 },
		{ 0, 0, 434 },
		{ 3062, 2724, 0 },
		{ 3160, 1703, 0 },
		{ 2720, 4810, 0 },
		{ 3160, 1704, 0 },
		{ 2922, 2160, 0 },
		{ 2723, 4828, 0 },
		{ 3144, 2938, 0 },
		{ 3164, 2092, 0 },
		{ 3062, 2731, 0 },
		{ 3158, 3034, 0 },
		{ 3147, 2886, 0 },
		{ 3164, 2093, 0 },
		{ 3165, 4596, 0 },
		{ 3165, 4598, 0 },
		{ 3116, 2353, 0 },
		{ 3147, 2889, 0 },
		{ 3062, 2736, 0 },
		{ 3155, 2480, 0 },
		{ 3160, 1705, 0 },
		{ 3160, 1706, 0 },
		{ 3155, 2485, 0 },
		{ 3125, 2041, 0 },
		{ 3125, 1923, 0 },
		{ 3165, 4674, 0 },
		{ 3147, 2900, 0 },
		{ 3162, 2013, 0 },
		{ 3147, 2903, 0 },
		{ 3158, 3057, 0 },
		{ 3162, 2188, 0 },
		{ 3160, 1713, 0 },
		{ 3125, 2047, 0 },
		{ 3165, 4462, 0 },
		{ 3164, 939, 402 },
		{ 3119, 4860, 414 },
		{ 2964, 2760, 0 },
		{ 3164, 2103, 0 },
		{ 3160, 1715, 0 },
		{ 3062, 2589, 0 },
		{ 3154, 2744, 0 },
		{ 3154, 2746, 0 },
		{ 3062, 2590, 0 },
		{ 2757, 4785, 0 },
		{ 3159, 2969, 0 },
		{ 3119, 4898, 410 },
		{ 3164, 2105, 0 },
		{ 2922, 2151, 0 },
		{ 3155, 2434, 0 },
		{ 3160, 1716, 0 },
		{ 3116, 2276, 0 },
		{ 3062, 2604, 0 },
		{ 2765, 4857, 0 },
		{ 3119, 4912, 401 },
		{ 3165, 4606, 0 },
		{ 2767, 4866, 0 },
		{ 2772, 1407, 0 },
		{ 3160, 1718, 0 },
		{ 2770, 4873, 0 },
		{ 2771, 4874, 0 },
		{ 3160, 1743, 0 },
		{ 3157, 2460, 0 },
		{ 3164, 2111, 0 },
		{ 3158, 3020, 0 },
		{ 3147, 2807, 0 },
		{ 3165, 4672, 0 },
		{ 3162, 2204, 0 },
		{ 3125, 1968, 0 },
		{ 3162, 2207, 0 },
		{ 3165, 4705, 0 },
		{ 3119, 4954, 416 },
		{ 3165, 4707, 0 },
		{ 3165, 4732, 0 },
		{ 2772, 1408, 0 },
		{ 3165, 4446, 0 },
		{ 3165, 4448, 0 },
		{ 0, 1360, 0 },
		{ 3062, 2694, 0 },
		{ 3062, 2695, 0 },
		{ 3160, 1753, 0 },
		{ 3164, 2117, 0 },
		{ 3119, 4978, 422 },
		{ 3164, 2118, 0 },
		{ 3165, 4489, 0 },
		{ 3116, 2299, 0 },
		{ 0, 0, 437 },
		{ 0, 0, 436 },
		{ 3119, 4982, 403 },
		{ 3165, 4516, 0 },
		{ 0, 0, 433 },
		{ 0, 0, 432 },
		{ 3165, 4518, 0 },
		{ 3155, 2469, 0 },
		{ 2922, 2142, 0 },
		{ 3162, 2216, 0 },
		{ 3158, 3042, 0 },
		{ 3165, 4528, 0 },
		{ 3119, 4997, 397 },
		{ 2802, 4757, 0 },
		{ 3119, 4854, 424 },
		{ 3119, 4856, 404 },
		{ 3147, 2827, 0 },
		{ 3160, 1764, 0 },
		{ 3164, 2120, 0 },
		{ 3160, 1765, 0 },
		{ 3119, 4867, 417 },
		{ 3119, 2237, 0 },
		{ 3165, 4590, 0 },
		{ 3165, 4592, 0 },
		{ 3165, 4594, 0 },
		{ 3162, 2221, 0 },
		{ 3160, 1768, 0 },
		{ 3119, 4886, 408 },
		{ 3119, 4891, 409 },
		{ 3119, 4894, 413 },
		{ 3164, 2125, 0 },
		{ 3147, 2836, 0 },
		{ 3165, 4604, 0 },
		{ 3164, 2127, 0 },
		{ 3119, 4904, 415 },
		{ 3158, 3058, 0 },
		{ 3160, 1778, 0 },
		{ 3062, 2721, 0 },
		{ 3162, 2227, 0 },
		{ 3116, 2318, 0 },
		{ 3125, 1987, 0 },
		{ 3165, 4668, 0 },
		{ 3119, 4918, 420 },
		{ 3069, 4771, 438 },
		{ 2829, 0, 394 },
		{ 0, 0, 395 },
		{ -2827, 5006, 390 },
		{ -2828, 4815, 0 },
		{ 3119, 4792, 0 },
		{ 3069, 4780, 0 },
		{ 0, 0, 391 },
		{ 3069, 4790, 0 },
		{ -2833, 7, 0 },
		{ -2834, 4819, 0 },
		{ 2837, 0, 392 },
		{ 3069, 4791, 0 },
		{ 3119, 4939, 0 },
		{ 0, 0, 393 },
		{ 3112, 3364, 157 },
		{ 0, 0, 157 },
		{ 0, 0, 158 },
		{ 3125, 1990, 0 },
		{ 3147, 2846, 0 },
		{ 3164, 2061, 0 },
		{ 2846, 4857, 0 },
		{ 3157, 2476, 0 },
		{ 3152, 1627, 0 },
		{ 3116, 2326, 0 },
		{ 3159, 2981, 0 },
		{ 3160, 1818, 0 },
		{ 3062, 2735, 0 },
		{ 3162, 2239, 0 },
		{ 3116, 2330, 0 },
		{ 3125, 1993, 0 },
		{ 3165, 4460, 0 },
		{ 0, 0, 155 },
		{ 2995, 4638, 180 },
		{ 0, 0, 180 },
		{ 3160, 1821, 0 },
		{ 2861, 4882, 0 },
		{ 3147, 2522, 0 },
		{ 3158, 3016, 0 },
		{ 3159, 2970, 0 },
		{ 3154, 2739, 0 },
		{ 2866, 4891, 0 },
		{ 3119, 2349, 0 },
		{ 3147, 2862, 0 },
		{ 3116, 2336, 0 },
		{ 3147, 2864, 0 },
		{ 3164, 2066, 0 },
		{ 3158, 3025, 0 },
		{ 3160, 1824, 0 },
		{ 3062, 2581, 0 },
		{ 3162, 2165, 0 },
		{ 3116, 2341, 0 },
		{ 2877, 4770, 0 },
		{ 3119, 2777, 0 },
		{ 3147, 2872, 0 },
		{ 3008, 2788, 0 },
		{ 3162, 2166, 0 },
		{ 3164, 2068, 0 },
		{ 3147, 2876, 0 },
		{ 2884, 4767, 0 },
		{ 3164, 1948, 0 },
		{ 3147, 2878, 0 },
		{ 3144, 2941, 0 },
		{ 3152, 1645, 0 },
		{ 3159, 2959, 0 },
		{ 3147, 2880, 0 },
		{ 2891, 4789, 0 },
		{ 3157, 2462, 0 },
		{ 3152, 1487, 0 },
		{ 3116, 2351, 0 },
		{ 3159, 2968, 0 },
		{ 3160, 1834, 0 },
		{ 3062, 2602, 0 },
		{ 3162, 2172, 0 },
		{ 3116, 2354, 0 },
		{ 3165, 4678, 0 },
		{ 0, 0, 178 },
		{ 2902, 0, 1 },
		{ -2902, 1273, 269 },
		{ 3147, 2784, 275 },
		{ 0, 0, 275 },
		{ 3125, 2006, 0 },
		{ 3116, 2258, 0 },
		{ 3147, 2898, 0 },
		{ 3144, 2946, 0 },
		{ 3164, 2077, 0 },
		{ 0, 0, 274 },
		{ 2912, 4856, 0 },
		{ 3149, 1950, 0 },
		{ 3158, 3004, 0 },
		{ 2941, 2507, 0 },
		{ 3147, 2902, 0 },
		{ 3008, 2791, 0 },
		{ 3062, 2690, 0 },
		{ 3155, 2483, 0 },
		{ 3147, 2907, 0 },
		{ 2921, 4841, 0 },
		{ 3162, 2031, 0 },
		{ 0, 2154, 0 },
		{ 3160, 1863, 0 },
		{ 3062, 2696, 0 },
		{ 3162, 2184, 0 },
		{ 3116, 2265, 0 },
		{ 3125, 2012, 0 },
		{ 3165, 4534, 0 },
		{ 0, 0, 273 },
		{ 0, 4380, 183 },
		{ 0, 0, 183 },
		{ 3162, 2186, 0 },
		{ 3152, 1520, 0 },
		{ 3116, 2270, 0 },
		{ 3144, 2949, 0 },
		{ 2937, 4877, 0 },
		{ 3159, 2657, 0 },
		{ 3154, 2741, 0 },
		{ 3147, 2923, 0 },
		{ 3159, 2975, 0 },
		{ 0, 2509, 0 },
		{ 3062, 2708, 0 },
		{ 3116, 2272, 0 },
		{ 2964, 2755, 0 },
		{ 3165, 4635, 0 },
		{ 0, 0, 181 },
		{ 2995, 4640, 177 },
		{ 0, 0, 176 },
		{ 0, 0, 177 },
		{ 3160, 1870, 0 },
		{ 2952, 4882, 0 },
		{ 3160, 1879, 0 },
		{ 3154, 2750, 0 },
		{ 3147, 2932, 0 },
		{ 2956, 4905, 0 },
		{ 3119, 2773, 0 },
		{ 3147, 2804, 0 },
		{ 2964, 2762, 0 },
		{ 3062, 2713, 0 },
		{ 3116, 2278, 0 },
		{ 3116, 2279, 0 },
		{ 3062, 2718, 0 },
		{ 3116, 2280, 0 },
		{ 0, 2758, 0 },
		{ 2966, 4762, 0 },
		{ 3162, 1981, 0 },
		{ 3008, 2780, 0 },
		{ 2969, 4777, 0 },
		{ 3147, 2520, 0 },
		{ 3158, 3055, 0 },
		{ 3159, 2957, 0 },
		{ 3154, 2748, 0 },
		{ 2974, 4782, 0 },
		{ 3119, 2347, 0 },
		{ 3147, 2821, 0 },
		{ 3116, 2283, 0 },
		{ 3147, 2823, 0 },
		{ 3164, 2088, 0 },
		{ 3158, 3066, 0 },
		{ 3160, 1888, 0 },
		{ 3062, 2726, 0 },
		{ 3162, 2193, 0 },
		{ 3116, 2287, 0 },
		{ 2985, 4807, 0 },
		{ 3157, 2470, 0 },
		{ 3152, 1555, 0 },
		{ 3116, 2289, 0 },
		{ 3159, 2978, 0 },
		{ 3160, 1891, 0 },
		{ 3062, 2734, 0 },
		{ 3162, 2196, 0 },
		{ 3116, 2292, 0 },
		{ 3165, 4563, 0 },
		{ 0, 0, 170 },
		{ 0, 4496, 169 },
		{ 0, 0, 169 },
		{ 3160, 1896, 0 },
		{ 2999, 4813, 0 },
		{ 3160, 1883, 0 },
		{ 3154, 2740, 0 },
		{ 3147, 2841, 0 },
		{ 3003, 4833, 0 },
		{ 3147, 2526, 0 },
		{ 3116, 2295, 0 },
		{ 3144, 2942, 0 },
		{ 3007, 4829, 0 },
		{ 3162, 1993, 0 },
		{ 0, 2785, 0 },
		{ 3010, 4842, 0 },
		{ 3147, 2516, 0 },
		{ 3158, 3022, 0 },
		{ 3159, 2972, 0 },
		{ 3154, 2745, 0 },
		{ 3015, 4850, 0 },
		{ 3119, 2351, 0 },
		{ 3147, 2849, 0 },
		{ 3116, 2298, 0 },
		{ 3147, 2851, 0 },
		{ 3164, 2095, 0 },
		{ 3158, 3030, 0 },
		{ 3160, 1923, 0 },
		{ 3062, 2583, 0 },
		{ 3162, 2202, 0 },
		{ 3116, 2302, 0 },
		{ 3026, 4868, 0 },
		{ 3157, 2464, 0 },
		{ 3152, 1570, 0 },
		{ 3116, 2304, 0 },
		{ 3159, 2963, 0 },
		{ 3160, 1928, 0 },
		{ 3062, 2591, 0 },
		{ 3162, 2205, 0 },
		{ 3116, 2307, 0 },
		{ 3165, 4454, 0 },
		{ 0, 0, 167 },
		{ 0, 4011, 172 },
		{ 0, 0, 172 },
		{ 0, 0, 173 },
		{ 3116, 2308, 0 },
		{ 3125, 2028, 0 },
		{ 3160, 1932, 0 },
		{ 3147, 2867, 0 },
		{ 3158, 3051, 0 },
		{ 3144, 2951, 0 },
		{ 3046, 4896, 0 },
		{ 3147, 2514, 0 },
		{ 3129, 1445, 0 },
		{ 3158, 3056, 0 },
		{ 3155, 2496, 0 },
		{ 3152, 1575, 0 },
		{ 3158, 3059, 0 },
		{ 3160, 1938, 0 },
		{ 3062, 2683, 0 },
		{ 3162, 2211, 0 },
		{ 3116, 2315, 0 },
		{ 3057, 4765, 0 },
		{ 3157, 2472, 0 },
		{ 3152, 1608, 0 },
		{ 3116, 2317, 0 },
		{ 3159, 2965, 0 },
		{ 3160, 1941, 0 },
		{ 0, 2691, 0 },
		{ 3162, 2214, 0 },
		{ 3116, 2320, 0 },
		{ 3127, 4718, 0 },
		{ 0, 0, 171 },
		{ 3147, 2885, 438 },
		{ 3164, 1467, 26 },
		{ 0, 4782, 438 },
		{ 3078, 0, 438 },
		{ 2277, 2685, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 3116, 2324, 0 },
		{ -3077, 5007, 0 },
		{ 3164, 661, 0 },
		{ 0, 0, 28 },
		{ 3144, 2955, 0 },
		{ 0, 0, 27 },
		{ 0, 0, 22 },
		{ 0, 0, 34 },
		{ 0, 0, 35 },
		{ 0, 3827, 39 },
		{ 0, 3559, 39 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 3108, 3697, 0 },
		{ 3120, 4358, 0 },
		{ 3112, 3341, 0 },
		{ 0, 0, 37 },
		{ 3115, 3402, 0 },
		{ 0, 3237, 0 },
		{ 3071, 1655, 0 },
		{ 0, 0, 36 },
		{ 3147, 2799, 50 },
		{ 0, 0, 50 },
		{ 3112, 3347, 50 },
		{ 3147, 2895, 50 },
		{ 0, 0, 53 },
		{ 3147, 2896, 0 },
		{ 3116, 2329, 0 },
		{ 3115, 3412, 0 },
		{ 3160, 1954, 0 },
		{ 3116, 2331, 0 },
		{ 3144, 2944, 0 },
		{ 0, 3660, 0 },
		{ 3152, 1644, 0 },
		{ 3162, 2224, 0 },
		{ 0, 0, 49 },
		{ 0, 3356, 0 },
		{ 3164, 2116, 0 },
		{ 3149, 2388, 0 },
		{ 0, 3421, 0 },
		{ 0, 2335, 0 },
		{ 3147, 2905, 0 },
		{ 0, 0, 51 },
		{ 0, 5, 54 },
		{ 0, 4330, 0 },
		{ 0, 0, 52 },
		{ 3155, 2478, 0 },
		{ 3158, 3031, 0 },
		{ 3125, 2045, 0 },
		{ 0, 2046, 0 },
		{ 3127, 4722, 0 },
		{ 0, 4723, 0 },
		{ 3147, 2909, 0 },
		{ 0, 1484, 0 },
		{ 3158, 3036, 0 },
		{ 3155, 2482, 0 },
		{ 3152, 1646, 0 },
		{ 3158, 3039, 0 },
		{ 3160, 1959, 0 },
		{ 3162, 2234, 0 },
		{ 3164, 2122, 0 },
		{ 3158, 2068, 0 },
		{ 3147, 2917, 0 },
		{ 3162, 2237, 0 },
		{ 3159, 2964, 0 },
		{ 3158, 3047, 0 },
		{ 3164, 2124, 0 },
		{ 3159, 2966, 0 },
		{ 0, 2943, 0 },
		{ 3147, 2824, 0 },
		{ 3152, 1647, 0 },
		{ 0, 2922, 0 },
		{ 3158, 3054, 0 },
		{ 0, 2408, 0 },
		{ 3164, 2126, 0 },
		{ 3159, 2973, 0 },
		{ 0, 1648, 0 },
		{ 3165, 4726, 0 },
		{ 0, 2747, 0 },
		{ 0, 2494, 0 },
		{ 0, 0, 46 },
		{ 3119, 2758, 0 },
		{ 0, 3062, 0 },
		{ 0, 2979, 0 },
		{ 0, 1966, 0 },
		{ 3165, 4734, 0 },
		{ 0, 2243, 0 },
		{ 0, 0, 47 },
		{ 0, 2129, 0 },
		{ 3127, 4735, 0 },
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
