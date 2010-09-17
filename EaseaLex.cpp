#include <clex.h>

#line 1 "EaseaLex.l"

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
#line 172 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting initialisation function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_INITIALISATION_FUNCTION;
 
#line 257 "EaseaLex.cpp"
		}
		break;
	case 13:
		{
#line 179 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting generation before reduce function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bGenerationFunctionBeforeReplacement = true;
  BEGIN COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT;
 
#line 270 "EaseaLex.cpp"
		}
		break;
	case 14:
		{
#line 188 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  if (bVERBOSE) printf ("Inserting at the begining of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bBeginGeneration = true;
  bEndGeneration = false;
  BEGIN COPY_BEG_GENERATION_FUNCTION;
 
#line 285 "EaseaLex.cpp"
		}
		break;
	case 15:
		{
#line 203 "EaseaLex.l"

  if( bVERBOSE )printf("inserting gp parameters\n");
  //  fprintf(fpOutputFile,"#define MAX_XOVER_DEPTH",%d
  fprintf(fpOutputFile,"#define TREE_DEPTH_MAX %d\n",iMAX_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MAX %d\n",iMAX_INIT_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MIN %d\n",iMIN_INIT_TREE_D);

  fprintf(fpOutputFile,"#define MAX_PROGS_SIZE %d\n",iPRG_BUF_SIZE);
  fprintf(fpOutputFile,"#define NB_GPU %d\n",iNB_GPU);

  fprintf(fpOutputFile,"#define NO_FITNESS_CASES %d\n",iNO_FITNESS_CASES);

#line 303 "EaseaLex.cpp"
		}
		break;
	case 16:
		{
#line 221 "EaseaLex.l"

  
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
 
#line 340 "EaseaLex.cpp"
		}
		break;
	case 17:
		{
#line 253 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"    case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"      %s",opDesc[i]->gpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"      break;\n");

  }
 
#line 354 "EaseaLex.cpp"
		}
		break;
	case 18:
		{
#line 262 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"  case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"    %s\n",opDesc[i]->cpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"    break;\n");
  }
 
#line 367 "EaseaLex.cpp"
		}
		break;
	case 19:
		{
#line 271 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Analysing GP OP code from ez file\n");
  BEGIN COPY_GP_OPCODE;
 
#line 380 "EaseaLex.cpp"
		}
		break;
	case 20:
		{
#line 280 "EaseaLex.l"

  if (bVERBOSE) printf ("found begin section\n");
  bGPOPCODE_ANALYSIS = true;
  BEGIN GP_RULE_ANALYSIS;
 
#line 391 "EaseaLex.cpp"
		}
		break;
	case 21:
		{
#line 286 "EaseaLex.l"
 
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
 
#line 411 "EaseaLex.cpp"
		}
		break;
	case 22:
		{
#line 301 "EaseaLex.l"

  if (bVERBOSE) printf("*** No GP OP codes were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
 
#line 423 "EaseaLex.cpp"
		}
		break;
	case 23:
		{
#line 307 "EaseaLex.l"

#line 430 "EaseaLex.cpp"
		}
		break;
	case 24:
		{
#line 308 "EaseaLex.l"
if( bGPOPCODE_ANALYSIS )printf("\n");lineCounter++;
#line 437 "EaseaLex.cpp"
		}
		break;
	case 25:
		{
#line 315 "EaseaLex.l"

  // this rule match the OP_NAME
  if( iGP_OPCODE_FIELD != 0 ){
    fprintf(stderr,"Error, OP_CODE name must be given first\n");
    exit(-1);
  }
  opDesc[iNoOp] = new OPCodeDesc();
  opDesc[iNoOp]->opcode = new string(yytext);
 
#line 452 "EaseaLex.cpp"
		}
		break;
	case 26:
		{
#line 325 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 1 ){
    fprintf(stderr,"Error, op code real name must be given at the second place\n");
    exit(-1);
  }
  opDesc[iNoOp]->realName = new string(yytext);
 
#line 465 "EaseaLex.cpp"
		}
		break;
	case 27:
		{
#line 334 "EaseaLex.l"

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
 
#line 484 "EaseaLex.cpp"
		}
		break;
	case 28:
		{
#line 348 "EaseaLex.l"

#line 491 "EaseaLex.cpp"
		}
		break;
	case 29:
		{
#line 349 "EaseaLex.l"

  iGP_OPCODE_FIELD = 0;
  iNoOp++;
 
#line 501 "EaseaLex.cpp"
		}
		break;
	case 30:
		{
#line 354 "EaseaLex.l"

  if( bGPOPCODE_ANALYSIS ) iGP_OPCODE_FIELD++;
 
#line 510 "EaseaLex.cpp"
		}
		break;
	case 31:
		{
#line 359 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 3 ){
    fprintf(stderr,"Error, code must be given at the forth place\n");
    exit(-1);
  }
  if( bVERBOSE ) printf("begining of the code part\n");
  accolade_counter=1;

  printf("arity : %d\n",opDesc[iNoOp]->arity);
  if( opDesc[iNoOp]->arity>=2 )
    opDesc[iNoOp]->gpuCodeStream << "OP2 = stack[--sp];\n      ";
  if( opDesc[iNoOp]->arity>=1 )
    opDesc[iNoOp]->gpuCodeStream << "OP1 = stack[--sp];\n      ";

  BEGIN GP_COPY_OPCODE_CODE;
 
#line 532 "EaseaLex.cpp"
		}
		break;
	case 32:
		{
#line 380 "EaseaLex.l"

  accolade_counter++;
  opDesc[iNoOp]->cpuCodeStream << "{";
  opDesc[iNoOp]->gpuCodeStream << "{";
 
#line 543 "EaseaLex.cpp"
		}
		break;
	case 33:
		{
#line 386 "EaseaLex.l"

  accolade_counter--;
  if( accolade_counter==0 ){
    opDesc[iNoOp]->gpuCodeStream << "\n      stack[sp++] = RESULT;\n";

    BEGIN GP_RULE_ANALYSIS;
  }
  else{
    opDesc[iNoOp]->cpuCodeStream << "}";
    opDesc[iNoOp]->gpuCodeStream << "}";
  }
 
#line 561 "EaseaLex.cpp"
		}
		break;
	case 34:
		{
#line 399 "EaseaLex.l"

  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
  printf("input no : %d\n",no_input);
  opDesc[iNoOp]->cpuCodeStream << "input["<< no_input <<"]" ;
  opDesc[iNoOp]->gpuCodeStream << "input["<< no_input << "]";  
 
#line 574 "EaseaLex.cpp"
		}
		break;
	case 35:
		{
#line 407 "EaseaLex.l"

  opDesc[iNoOp]->isERC = true;
  opDesc[iNoOp]->cpuCodeStream << "root->erc_value" ;
  opDesc[iNoOp]->gpuCodeStream << "k_progs[start_prog++];" ;
  printf("ERC matched\n");

#line 586 "EaseaLex.cpp"
		}
		break;
	case 36:
		{
#line 414 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << "\n  ";
  opDesc[iNoOp]->gpuCodeStream << "\n    ";
 
#line 596 "EaseaLex.cpp"
		}
		break;
	case 37:
		{
#line 420 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << yytext;
  opDesc[iNoOp]->gpuCodeStream << yytext;
 
#line 606 "EaseaLex.cpp"
		}
		break;
	case 38:
		{
#line 425 "EaseaLex.l"
 
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  if( bVERBOSE ) printf("Insert GP eval header\n");
  iCOPY_GP_EVAL_STATUS = EVAL_HDR;
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 622 "EaseaLex.cpp"
		}
		break;
	case 39:
		{
#line 436 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  BEGIN COPY_GP_EVAL;
 
#line 638 "EaseaLex.cpp"
		}
		break;
	case 40:
		{
#line 447 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 654 "EaseaLex.cpp"
		}
		break;
	case 41:
		{
#line 458 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 671 "EaseaLex.cpp"
		}
		break;
	case 42:
		{
#line 471 "EaseaLex.l"

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
 
#line 690 "EaseaLex.cpp"
		}
		break;
	case 43:
		{
#line 486 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_HDR){
    bIsCopyingGPEval = true;
  }
 
#line 701 "EaseaLex.cpp"
		}
		break;
	case 44:
		{
#line 492 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_BDY){
    bIsCopyingGPEval = true;
  }
 
#line 712 "EaseaLex.cpp"
		}
		break;
	case 45:
		{
#line 500 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_FTR){
    bIsCopyingGPEval = true;
  }
 
#line 723 "EaseaLex.cpp"
		}
		break;
	case 46:
		{
#line 506 "EaseaLex.l"

  if( bIsCopyingGPEval ){
    bIsCopyingGPEval = false;
    bCOPY_GP_EVAL_GPU = false;
    rewind(fpGenomeFile);
    yyin = fpTemplateFile;
    BEGIN TEMPLATE_ANALYSIS;
  }
 
#line 738 "EaseaLex.cpp"
		}
		break;
	case 47:
		{
#line 516 "EaseaLex.l"

  if( bIsCopyingGPEval ) fprintf(fpOutputFile,"%s",yytext);
 
#line 747 "EaseaLex.cpp"
		}
		break;
	case 48:
		{
#line 520 "EaseaLex.l"

  if( bIsCopyingGPEval) fprintf(fpOutputFile, "outputs[i]" );
 
#line 756 "EaseaLex.cpp"
		}
		break;
	case 49:
		{
#line 524 "EaseaLex.l"

  if( bIsCopyingGPEval )
    if( iCOPY_GP_EVAL_STATUS==EVAL_FTR )
      if( bCOPY_GP_EVAL_GPU ){
	fprintf(fpOutputFile,"k_results[index] =");
      }
      else fprintf(fpOutputFile,"%s",yytext);
 
#line 770 "EaseaLex.cpp"
		}
		break;
	case 50:
		{
#line 536 "EaseaLex.l"

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
 
#line 788 "EaseaLex.cpp"
		}
		break;
	case 51:
		{
#line 550 "EaseaLex.l"

  if( bIsCopyingGPEval )
    fprintf(fpOutputFile,"return fitness = "); 
 
#line 798 "EaseaLex.cpp"
		}
		break;
	case 52:
		{
#line 557 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  yyreset();
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Evaluation population in a single function!!.\n");
  BEGIN COPY_INSTEAD_EVAL;
 
#line 811 "EaseaLex.cpp"
		}
		break;
	case 53:
		{
#line 565 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting at the end of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bEndGeneration = true;
  bBeginGeneration = false;
  BEGIN COPY_END_GENERATION_FUNCTION;
 
#line 825 "EaseaLex.cpp"
		}
		break;
	case 54:
		{
#line 574 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Bound Checking function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_BOUND_CHECKING_FUNCTION;
 
#line 837 "EaseaLex.cpp"
		}
		break;
	case 55:
		{
#line 581 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 
#line 849 "EaseaLex.cpp"
		}
		break;
	case 56:
		{
#line 588 "EaseaLex.l"

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
 
#line 878 "EaseaLex.cpp"
		}
		break;
	case 57:
		{
#line 611 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 
#line 895 "EaseaLex.cpp"
		}
		break;
	case 58:
		{
#line 623 "EaseaLex.l"

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
 
#line 921 "EaseaLex.cpp"
		}
		break;
	case 59:
		{
#line 644 "EaseaLex.l"

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
  
 
#line 942 "EaseaLex.cpp"
		}
		break;
#line 662 "EaseaLex.l"
  
#line 676 "EaseaLex.l"
      
#line 949 "EaseaLex.cpp"
	case 60:
		{
#line 684 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 962 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 693 "EaseaLex.l"

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
 
#line 985 "EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 710 "EaseaLex.l"

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
 
#line 1008 "EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 728 "EaseaLex.l"

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
 
#line 1040 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 755 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  fprintf (fpOutputFile,"// Memberwise serialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->serializeIndividual(fpOutputFile, "this");
  //fprintf(fpOutputFile,"\tEASEA_Line << endl;\n");
 
#line 1054 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 764 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome deserializer.\n");
  fprintf (fpOutputFile,"// Memberwise deserialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->deserializeIndividual(fpOutputFile, "this");
 
#line 1067 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 772 "EaseaLex.l"

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
 
#line 1088 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 788 "EaseaLex.l"

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
 
#line 1110 "EaseaLex.cpp"
		}
		break;
	case 68:
		{
#line 805 "EaseaLex.l"
       
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
 
#line 1138 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 827 "EaseaLex.l"

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
 
#line 1160 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 843 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 1175 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 852 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 1187 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 860 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 1199 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 867 "EaseaLex.l"

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
 
#line 1230 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 892 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 1243 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 899 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 1257 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 908 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_INITIALISER;   
 
#line 1268 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 914 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 1280 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 921 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 1292 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 927 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 1304 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 933 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 1316 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 939 "EaseaLex.l"
      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 1329 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 946 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 1342 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 953 "EaseaLex.l"
 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 1356 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 962 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 1367 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 967 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 1381 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 976 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 1395 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 985 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 1409 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 995 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 1422 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 1003 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 1431 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 1007 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 1440 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 1011 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 1449 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 1015 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 1458 "EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 1019 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 1468 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 1024 "EaseaLex.l"

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

#line 1487 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 1037 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 1494 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 1038 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 1501 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 1039 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 1508 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 1040 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 1515 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 1041 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 1522 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 1042 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1529 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 1043 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1536 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 1044 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1543 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 1045 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1550 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 1046 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1557 "EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 1047 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",nELITE);
  ////DEBUG_PRT_PRT("elitism is %d, elite size is %d",bELITISM, nELITE);
 
#line 1567 "EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 1052 "EaseaLex.l"

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
 
#line 1586 "EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 1065 "EaseaLex.l"

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
 
#line 1605 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 1078 "EaseaLex.l"

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
 
#line 1624 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 1091 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1634 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 1095 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1641 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 1096 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1648 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 1097 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1655 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 1098 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1662 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 1099 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1669 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 1100 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1676 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 1101 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1683 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 1102 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1690 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 1103 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1697 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 1104 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1704 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 1106 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1711 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 1107 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1718 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1109 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bREMOTE_ISLAND_MODEL);
#line 1725 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1111 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1732 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1112 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1739 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1113 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1746 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1114 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1753 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1115 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1760 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1117 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 1767 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1118 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 1774 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1120 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1788 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1128 "EaseaLex.l"

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
 
#line 1808 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1142 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1822 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1150 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1836 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1159 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1850 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1168 "EaseaLex.l"

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

#line 1913 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1225 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded with no warning.\n");
  printf ("Have a nice compile time.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 1930 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1237 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1937 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1243 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1949 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1249 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1962 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1256 "EaseaLex.l"

#line 1969 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1257 "EaseaLex.l"
lineCounter++;
#line 1976 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1259 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1988 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1265 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2001 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1273 "EaseaLex.l"

#line 2008 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1274 "EaseaLex.l"

  lineCounter++;
 
#line 2017 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1278 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2029 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1284 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2043 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1292 "EaseaLex.l"

#line 2050 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1293 "EaseaLex.l"

  lineCounter++;
 
#line 2059 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1297 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){");
  bFunction=1; bInitFunction=1;
      
  BEGIN COPY;
 
#line 2071 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1303 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2085 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1311 "EaseaLex.l"

#line 2092 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1316 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){");
  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2103 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1322 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2117 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1330 "EaseaLex.l"

#line 2124 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1333 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2140 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1344 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2156 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1354 "EaseaLex.l"

#line 2163 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1357 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    bBeginGeneration = 0;
    bBeginGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2179 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1368 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2193 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1377 "EaseaLex.l"

#line 2200 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1379 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2216 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1391 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2232 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1401 "EaseaLex.l"

#line 2239 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1405 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2254 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1415 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2269 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1424 "EaseaLex.l"

#line 2276 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1427 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2289 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1434 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2303 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1442 "EaseaLex.l"

#line 2310 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1446 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2318 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1448 "EaseaLex.l"

#line 2325 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1454 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2332 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1455 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2339 "EaseaLex.cpp"
		}
		break;
	case 175:
	case 176:
		{
#line 1458 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2350 "EaseaLex.cpp"
		}
		break;
	case 177:
	case 178:
		{
#line 1463 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2359 "EaseaLex.cpp"
		}
		break;
	case 179:
	case 180:
		{
#line 1466 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2368 "EaseaLex.cpp"
		}
		break;
	case 181:
	case 182:
		{
#line 1469 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2385 "EaseaLex.cpp"
		}
		break;
	case 183:
	case 184:
		{
#line 1480 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2399 "EaseaLex.cpp"
		}
		break;
	case 185:
	case 186:
		{
#line 1488 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2408 "EaseaLex.cpp"
		}
		break;
	case 187:
	case 188:
		{
#line 1491 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2417 "EaseaLex.cpp"
		}
		break;
	case 189:
	case 190:
		{
#line 1494 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2426 "EaseaLex.cpp"
		}
		break;
	case 191:
	case 192:
		{
#line 1497 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2435 "EaseaLex.cpp"
		}
		break;
	case 193:
	case 194:
		{
#line 1500 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2444 "EaseaLex.cpp"
		}
		break;
	case 195:
	case 196:
		{
#line 1504 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2456 "EaseaLex.cpp"
		}
		break;
	case 197:
		{
#line 1510 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2463 "EaseaLex.cpp"
		}
		break;
	case 198:
		{
#line 1511 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2470 "EaseaLex.cpp"
		}
		break;
	case 199:
		{
#line 1512 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2477 "EaseaLex.cpp"
		}
		break;
	case 200:
		{
#line 1513 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2487 "EaseaLex.cpp"
		}
		break;
	case 201:
		{
#line 1518 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2494 "EaseaLex.cpp"
		}
		break;
	case 202:
		{
#line 1519 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2501 "EaseaLex.cpp"
		}
		break;
	case 203:
		{
#line 1520 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2508 "EaseaLex.cpp"
		}
		break;
	case 204:
		{
#line 1521 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2515 "EaseaLex.cpp"
		}
		break;
	case 205:
		{
#line 1522 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2522 "EaseaLex.cpp"
		}
		break;
	case 206:
		{
#line 1523 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2529 "EaseaLex.cpp"
		}
		break;
	case 207:
		{
#line 1524 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2536 "EaseaLex.cpp"
		}
		break;
	case 208:
		{
#line 1525 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2543 "EaseaLex.cpp"
		}
		break;
	case 209:
		{
#line 1526 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2551 "EaseaLex.cpp"
		}
		break;
	case 210:
		{
#line 1528 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2559 "EaseaLex.cpp"
		}
		break;
	case 211:
		{
#line 1530 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2567 "EaseaLex.cpp"
		}
		break;
	case 212:
		{
#line 1532 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2577 "EaseaLex.cpp"
		}
		break;
	case 213:
		{
#line 1536 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2584 "EaseaLex.cpp"
		}
		break;
	case 214:
		{
#line 1537 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2591 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1538 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2602 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1543 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2609 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1544 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2618 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1547 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2630 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1553 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2639 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1556 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2651 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1562 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2662 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1567 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2678 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1577 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2685 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1580 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2694 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1583 "EaseaLex.l"
BEGIN COPY;
#line 2701 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1585 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2708 "EaseaLex.cpp"
		}
		break;
	case 227:
	case 228:
	case 229:
		{
#line 1588 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2721 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1593 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2732 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1598 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 2741 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1607 "EaseaLex.l"
;
#line 2748 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1608 "EaseaLex.l"
;
#line 2755 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1609 "EaseaLex.l"
;
#line 2762 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1610 "EaseaLex.l"
;
#line 2769 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1613 "EaseaLex.l"
 /* do nothing */ 
#line 2776 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1614 "EaseaLex.l"
 /*return '\n';*/ 
#line 2783 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1615 "EaseaLex.l"
 /*return '\n';*/ 
#line 2790 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1618 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 2799 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1621 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 2809 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1625 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
  printf("match gpnode\n");
  return GPNODE;
 
#line 2821 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1632 "EaseaLex.l"
return STATIC;
#line 2828 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1633 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 2835 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1634 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 2842 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1635 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 2849 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1636 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 2856 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1637 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 2863 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1639 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 2870 "EaseaLex.cpp"
		}
		break;
#line 1640 "EaseaLex.l"
  
#line 2875 "EaseaLex.cpp"
	case 249:
		{
#line 1641 "EaseaLex.l"
return GENOME; 
#line 2880 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1643 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 2890 "EaseaLex.cpp"
		}
		break;
	case 251:
	case 252:
	case 253:
		{
#line 1650 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 2899 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1651 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 2906 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1654 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 2914 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1656 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 2921 "EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1662 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 2933 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1668 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2946 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1675 "EaseaLex.l"

#line 2953 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1677 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 2964 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1688 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 2979 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1698 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 2990 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1704 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 2999 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1708 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3014 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1721 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3026 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1727 "EaseaLex.l"

#line 3033 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1728 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3046 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1735 "EaseaLex.l"

#line 3053 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1736 "EaseaLex.l"
lineCounter++;
#line 3060 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1737 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3073 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1744 "EaseaLex.l"

#line 3080 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1745 "EaseaLex.l"
lineCounter++;
#line 3087 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1747 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3100 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1754 "EaseaLex.l"

#line 3107 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1755 "EaseaLex.l"
lineCounter++;
#line 3114 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1757 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3127 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1764 "EaseaLex.l"

#line 3134 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1765 "EaseaLex.l"
lineCounter++;
#line 3141 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1771 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3148 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1772 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3155 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1773 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3162 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1774 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3169 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1775 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3176 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1776 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3183 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1777 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3190 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1779 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3199 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1782 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3212 "EaseaLex.cpp"
		}
		break;
	case 288:
	case 289:
		{
#line 1791 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3223 "EaseaLex.cpp"
		}
		break;
	case 290:
	case 291:
		{
#line 1796 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3232 "EaseaLex.cpp"
		}
		break;
	case 292:
	case 293:
		{
#line 1799 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3241 "EaseaLex.cpp"
		}
		break;
	case 294:
	case 295:
		{
#line 1802 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3253 "EaseaLex.cpp"
		}
		break;
	case 296:
	case 297:
		{
#line 1808 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3266 "EaseaLex.cpp"
		}
		break;
	case 298:
	case 299:
		{
#line 1815 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3275 "EaseaLex.cpp"
		}
		break;
	case 300:
	case 301:
		{
#line 1818 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3284 "EaseaLex.cpp"
		}
		break;
	case 302:
	case 303:
		{
#line 1821 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3293 "EaseaLex.cpp"
		}
		break;
	case 304:
	case 305:
		{
#line 1824 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3302 "EaseaLex.cpp"
		}
		break;
	case 306:
	case 307:
		{
#line 1827 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3311 "EaseaLex.cpp"
		}
		break;
	case 308:
		{
#line 1830 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3320 "EaseaLex.cpp"
		}
		break;
	case 309:
		{
#line 1833 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3330 "EaseaLex.cpp"
		}
		break;
	case 310:
		{
#line 1837 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3338 "EaseaLex.cpp"
		}
		break;
	case 311:
		{
#line 1839 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3349 "EaseaLex.cpp"
		}
		break;
	case 312:
		{
#line 1844 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3360 "EaseaLex.cpp"
		}
		break;
	case 313:
		{
#line 1849 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3368 "EaseaLex.cpp"
		}
		break;
	case 314:
		{
#line 1851 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3376 "EaseaLex.cpp"
		}
		break;
	case 315:
		{
#line 1853 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3384 "EaseaLex.cpp"
		}
		break;
	case 316:
		{
#line 1855 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3392 "EaseaLex.cpp"
		}
		break;
	case 317:
		{
#line 1857 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3400 "EaseaLex.cpp"
		}
		break;
	case 318:
		{
#line 1859 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3407 "EaseaLex.cpp"
		}
		break;
	case 319:
		{
#line 1860 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3414 "EaseaLex.cpp"
		}
		break;
	case 320:
		{
#line 1861 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3422 "EaseaLex.cpp"
		}
		break;
	case 321:
		{
#line 1863 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3430 "EaseaLex.cpp"
		}
		break;
	case 322:
		{
#line 1865 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3438 "EaseaLex.cpp"
		}
		break;
	case 323:
		{
#line 1867 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3445 "EaseaLex.cpp"
		}
		break;
	case 324:
		{
#line 1868 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3457 "EaseaLex.cpp"
		}
		break;
	case 325:
		{
#line 1874 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3466 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1877 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3476 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1881 "EaseaLex.l"
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
#line 3493 "EaseaLex.cpp"
		}
		break;
	case 328:
	case 329:
		{
#line 1893 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3503 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1896 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3510 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1903 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3517 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 1904 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3524 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 1905 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3531 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 1906 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3538 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 1907 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3545 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 1909 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3554 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 1913 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3567 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1921 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3580 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 1930 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3593 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 1939 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3608 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 1949 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3615 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 1950 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3622 "EaseaLex.cpp"
		}
		break;
	case 343:
	case 344:
		{
#line 1953 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3633 "EaseaLex.cpp"
		}
		break;
	case 345:
	case 346:
		{
#line 1958 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3642 "EaseaLex.cpp"
		}
		break;
	case 347:
	case 348:
		{
#line 1961 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3651 "EaseaLex.cpp"
		}
		break;
	case 349:
	case 350:
		{
#line 1964 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3664 "EaseaLex.cpp"
		}
		break;
	case 351:
	case 352:
		{
#line 1971 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3677 "EaseaLex.cpp"
		}
		break;
	case 353:
	case 354:
		{
#line 1978 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3686 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 1981 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3693 "EaseaLex.cpp"
		}
		break;
	case 356:
		{
#line 1982 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3700 "EaseaLex.cpp"
		}
		break;
	case 357:
		{
#line 1983 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3707 "EaseaLex.cpp"
		}
		break;
	case 358:
		{
#line 1984 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3717 "EaseaLex.cpp"
		}
		break;
	case 359:
		{
#line 1989 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3724 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 1990 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3731 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 1991 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3738 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 1992 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3745 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 1993 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3753 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 1995 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3761 "EaseaLex.cpp"
		}
		break;
	case 365:
		{
#line 1997 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3769 "EaseaLex.cpp"
		}
		break;
	case 366:
		{
#line 1999 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3777 "EaseaLex.cpp"
		}
		break;
	case 367:
		{
#line 2001 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 3785 "EaseaLex.cpp"
		}
		break;
	case 368:
		{
#line 2003 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 3793 "EaseaLex.cpp"
		}
		break;
	case 369:
		{
#line 2005 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 3801 "EaseaLex.cpp"
		}
		break;
	case 370:
		{
#line 2007 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3808 "EaseaLex.cpp"
		}
		break;
	case 371:
		{
#line 2008 "EaseaLex.l"
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
#line 3831 "EaseaLex.cpp"
		}
		break;
	case 372:
		{
#line 2025 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 3842 "EaseaLex.cpp"
		}
		break;
	case 373:
		{
#line 2030 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 3856 "EaseaLex.cpp"
		}
		break;
	case 374:
		{
#line 2038 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3863 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2044 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 3873 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2048 "EaseaLex.l"

#line 3880 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2051 "EaseaLex.l"
;
#line 3887 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2052 "EaseaLex.l"
;
#line 3894 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2053 "EaseaLex.l"
;
#line 3901 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2054 "EaseaLex.l"
;
#line 3908 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2056 "EaseaLex.l"
 /* do nothing */ 
#line 3915 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2057 "EaseaLex.l"
 /*return '\n';*/ 
#line 3922 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2058 "EaseaLex.l"
 /*return '\n';*/ 
#line 3929 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2060 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 3936 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2061 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 3943 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2062 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 3950 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2063 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 3957 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2064 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 3964 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2065 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 3971 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2066 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 3978 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2067 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 3985 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2068 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 3992 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2070 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 3999 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2071 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4006 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2072 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4013 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2073 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4020 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2074 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4027 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2076 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4034 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2077 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4041 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2079 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4052 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2084 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4059 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2086 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4070 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2091 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4077 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2094 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4084 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2095 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4091 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2096 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4098 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2097 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4105 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2098 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4112 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2100 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4119 "EaseaLex.cpp"
		}
		break;
#line 2102 "EaseaLex.l"
 
#line 4124 "EaseaLex.cpp"
	case 410:
	case 411:
	case 412:
		{
#line 2106 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4131 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2107 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4138 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2110 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4146 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2113 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4153 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2115 "EaseaLex.l"

  lineCounter++;

#line 4162 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2118 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4172 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2123 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4182 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2128 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4192 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2133 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4202 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2138 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4212 "EaseaLex.cpp"
		}
		break;
	case 422:
		{
#line 2143 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4222 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2152 "EaseaLex.l"
return  (char)yytext[0];
#line 4229 "EaseaLex.cpp"
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
#line 2154 "EaseaLex.l"


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
    else if( TARGET_FLAVOR == CUDA_FLAVOR_GP )
      strcat(sTemp,"CUDA_GP.tpl");
    else if(TARGET_FLAVOR == CUDA_FLAVOR_MEMETIC )
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

#line 4417 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		175,
		-176,
		0,
		177,
		-178,
		0,
		179,
		-180,
		0,
		181,
		-182,
		0,
		187,
		-188,
		0,
		189,
		-190,
		0,
		191,
		-192,
		0,
		193,
		-194,
		0,
		185,
		-186,
		0,
		183,
		-184,
		0,
		-225,
		0,
		-231,
		0,
		292,
		-293,
		0,
		294,
		-295,
		0,
		288,
		-289,
		0,
		300,
		-301,
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
		290,
		-291,
		0,
		353,
		-354,
		0,
		298,
		-299,
		0,
		351,
		-352,
		0,
		296,
		-297,
		0,
		345,
		-346,
		0,
		347,
		-348,
		0,
		349,
		-350,
		0,
		343,
		-344,
		0
	};
	yymatch = match;

	yytransitionmax = 4930;
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
		{ 2991, 61 },
		{ 2991, 61 },
		{ 1853, 1956 },
		{ 1494, 1494 },
		{ 67, 61 },
		{ 2350, 2325 },
		{ 2350, 2325 },
		{ 2329, 2300 },
		{ 2329, 2300 },
		{ 1952, 1954 },
		{ 0, 87 },
		{ 71, 3 },
		{ 72, 3 },
		{ 2212, 43 },
		{ 2213, 43 },
		{ 1975, 39 },
		{ 69, 1 },
		{ 1952, 1948 },
		{ 0, 2233 },
		{ 67, 1 },
		{ 2753, 2749 },
		{ 2178, 2180 },
		{ 165, 161 },
		{ 2991, 61 },
		{ 2195, 2194 },
		{ 2989, 61 },
		{ 1494, 1494 },
		{ 3038, 3036 },
		{ 2350, 2325 },
		{ 1336, 1335 },
		{ 2329, 2300 },
		{ 1327, 1326 },
		{ 1479, 1462 },
		{ 1480, 1462 },
		{ 71, 3 },
		{ 2993, 61 },
		{ 2212, 43 },
		{ 2025, 2004 },
		{ 1975, 39 },
		{ 2990, 61 },
		{ 2990, 61 },
		{ 2990, 61 },
		{ 2990, 61 },
		{ 2990, 61 },
		{ 2990, 61 },
		{ 2990, 61 },
		{ 2990, 61 },
		{ 2990, 61 },
		{ 2990, 61 },
		{ 70, 3 },
		{ 2992, 61 },
		{ 2211, 43 },
		{ 1569, 1563 },
		{ 1961, 39 },
		{ 2351, 2325 },
		{ 1479, 1462 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 1571, 1565 },
		{ 2987, 61 },
		{ 1481, 1462 },
		{ 1442, 1421 },
		{ 2988, 61 },
		{ 1443, 1422 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2988, 61 },
		{ 2994, 61 },
		{ 2173, 40 },
		{ 1516, 1500 },
		{ 1517, 1500 },
		{ 1435, 1413 },
		{ 1960, 40 },
		{ 2403, 2377 },
		{ 2403, 2377 },
		{ 2332, 2303 },
		{ 2332, 2303 },
		{ 2348, 2323 },
		{ 2348, 2323 },
		{ 1786, 37 },
		{ 2356, 2330 },
		{ 2356, 2330 },
		{ 2368, 2342 },
		{ 2368, 2342 },
		{ 1436, 1414 },
		{ 1437, 1415 },
		{ 1438, 1416 },
		{ 1439, 1417 },
		{ 1441, 1420 },
		{ 1444, 1423 },
		{ 1445, 1424 },
		{ 2173, 40 },
		{ 1516, 1500 },
		{ 1963, 40 },
		{ 1446, 1425 },
		{ 1447, 1426 },
		{ 2403, 2377 },
		{ 1448, 1427 },
		{ 2332, 2303 },
		{ 1449, 1428 },
		{ 2348, 2323 },
		{ 1450, 1429 },
		{ 1786, 37 },
		{ 2356, 2330 },
		{ 1451, 1430 },
		{ 2368, 2342 },
		{ 2172, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1773, 37 },
		{ 1976, 40 },
		{ 1452, 1432 },
		{ 1455, 1435 },
		{ 1518, 1500 },
		{ 2404, 2377 },
		{ 1456, 1436 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1962, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1970, 40 },
		{ 1968, 40 },
		{ 1981, 40 },
		{ 1969, 40 },
		{ 1981, 40 },
		{ 1972, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1971, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1457, 1437 },
		{ 1964, 40 },
		{ 1966, 40 },
		{ 1458, 1438 },
		{ 1981, 40 },
		{ 1459, 1439 },
		{ 1981, 40 },
		{ 1979, 40 },
		{ 1967, 40 },
		{ 1981, 40 },
		{ 1980, 40 },
		{ 1973, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1978, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1965, 40 },
		{ 1981, 40 },
		{ 1977, 40 },
		{ 1981, 40 },
		{ 1974, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1981, 40 },
		{ 1363, 21 },
		{ 1519, 1501 },
		{ 1520, 1501 },
		{ 1461, 1441 },
		{ 1350, 21 },
		{ 2370, 2344 },
		{ 2370, 2344 },
		{ 2392, 2366 },
		{ 2392, 2366 },
		{ 2393, 2367 },
		{ 2393, 2367 },
		{ 2433, 2407 },
		{ 2433, 2407 },
		{ 2438, 2412 },
		{ 2438, 2412 },
		{ 2444, 2418 },
		{ 2444, 2418 },
		{ 1462, 1442 },
		{ 1463, 1443 },
		{ 1464, 1444 },
		{ 1465, 1445 },
		{ 1466, 1446 },
		{ 1467, 1447 },
		{ 1363, 21 },
		{ 1519, 1501 },
		{ 1351, 21 },
		{ 1364, 21 },
		{ 1468, 1448 },
		{ 2370, 2344 },
		{ 1469, 1449 },
		{ 2392, 2366 },
		{ 1470, 1450 },
		{ 2393, 2367 },
		{ 1471, 1452 },
		{ 2433, 2407 },
		{ 1474, 1455 },
		{ 2438, 2412 },
		{ 1475, 1456 },
		{ 2444, 2418 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1476, 1457 },
		{ 1477, 1459 },
		{ 1478, 1461 },
		{ 1375, 1353 },
		{ 1521, 1501 },
		{ 1482, 1463 },
		{ 1483, 1464 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1367, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1356, 21 },
		{ 1354, 21 },
		{ 1369, 21 },
		{ 1355, 21 },
		{ 1369, 21 },
		{ 1358, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1357, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1488, 1467 },
		{ 1352, 21 },
		{ 1365, 21 },
		{ 1489, 1468 },
		{ 1359, 21 },
		{ 1490, 1469 },
		{ 1369, 21 },
		{ 1370, 21 },
		{ 1353, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1360, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1368, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1371, 21 },
		{ 1369, 21 },
		{ 1366, 21 },
		{ 1369, 21 },
		{ 1361, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1369, 21 },
		{ 1947, 38 },
		{ 1522, 1502 },
		{ 1523, 1502 },
		{ 1484, 1465 },
		{ 1772, 38 },
		{ 2457, 2431 },
		{ 2457, 2431 },
		{ 2458, 2432 },
		{ 2458, 2432 },
		{ 1486, 1466 },
		{ 1485, 1465 },
		{ 2462, 2436 },
		{ 2462, 2436 },
		{ 2468, 2442 },
		{ 2468, 2442 },
		{ 1491, 1470 },
		{ 1487, 1466 },
		{ 1492, 1471 },
		{ 1495, 1475 },
		{ 1496, 1476 },
		{ 1497, 1477 },
		{ 1498, 1478 },
		{ 1500, 1482 },
		{ 1947, 38 },
		{ 1522, 1502 },
		{ 1777, 38 },
		{ 2469, 2443 },
		{ 2469, 2443 },
		{ 2457, 2431 },
		{ 1501, 1483 },
		{ 2458, 2432 },
		{ 1502, 1484 },
		{ 1525, 1503 },
		{ 1526, 1503 },
		{ 2462, 2436 },
		{ 1503, 1485 },
		{ 2468, 2442 },
		{ 1504, 1486 },
		{ 1946, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 2469, 2443 },
		{ 1787, 38 },
		{ 1505, 1487 },
		{ 1506, 1488 },
		{ 1524, 1502 },
		{ 1507, 1489 },
		{ 1525, 1503 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1774, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1782, 38 },
		{ 1780, 38 },
		{ 1790, 38 },
		{ 1781, 38 },
		{ 1790, 38 },
		{ 1784, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1783, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1508, 1490 },
		{ 1778, 38 },
		{ 1527, 1503 },
		{ 1509, 1491 },
		{ 1790, 38 },
		{ 1510, 1492 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1779, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1775, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1776, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1789, 38 },
		{ 1790, 38 },
		{ 1788, 38 },
		{ 1790, 38 },
		{ 1785, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 1790, 38 },
		{ 2747, 44 },
		{ 2748, 44 },
		{ 1528, 1504 },
		{ 1529, 1504 },
		{ 67, 44 },
		{ 2263, 2236 },
		{ 2263, 2236 },
		{ 1512, 1495 },
		{ 1513, 1496 },
		{ 1514, 1497 },
		{ 1515, 1498 },
		{ 2305, 2276 },
		{ 2305, 2276 },
		{ 2306, 2277 },
		{ 2306, 2277 },
		{ 1378, 1354 },
		{ 1379, 1355 },
		{ 1383, 1357 },
		{ 1384, 1358 },
		{ 1385, 1359 },
		{ 1534, 1506 },
		{ 1535, 1507 },
		{ 1536, 1508 },
		{ 2747, 44 },
		{ 1538, 1512 },
		{ 1528, 1504 },
		{ 2321, 2293 },
		{ 2321, 2293 },
		{ 2263, 2236 },
		{ 1539, 1513 },
		{ 1531, 1505 },
		{ 1532, 1505 },
		{ 1549, 1535 },
		{ 1550, 1535 },
		{ 2305, 2276 },
		{ 1540, 1514 },
		{ 2306, 2277 },
		{ 2227, 44 },
		{ 2746, 44 },
		{ 2226, 44 },
		{ 2226, 44 },
		{ 2226, 44 },
		{ 2226, 44 },
		{ 2226, 44 },
		{ 2226, 44 },
		{ 2226, 44 },
		{ 2226, 44 },
		{ 2226, 44 },
		{ 2226, 44 },
		{ 2321, 2293 },
		{ 1541, 1515 },
		{ 1548, 1534 },
		{ 1386, 1360 },
		{ 1531, 1505 },
		{ 1530, 1504 },
		{ 1549, 1535 },
		{ 2228, 44 },
		{ 2225, 44 },
		{ 2220, 44 },
		{ 2228, 44 },
		{ 2217, 44 },
		{ 2224, 44 },
		{ 2222, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2219, 44 },
		{ 2214, 44 },
		{ 2221, 44 },
		{ 2216, 44 },
		{ 2228, 44 },
		{ 2223, 44 },
		{ 2218, 44 },
		{ 2215, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 1533, 1505 },
		{ 2232, 44 },
		{ 1551, 1535 },
		{ 1552, 1536 },
		{ 2228, 44 },
		{ 1554, 1538 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2229, 44 },
		{ 2230, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2231, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 2228, 44 },
		{ 159, 4 },
		{ 160, 4 },
		{ 1558, 1548 },
		{ 1559, 1548 },
		{ 2483, 2454 },
		{ 2483, 2454 },
		{ 2324, 2296 },
		{ 2324, 2296 },
		{ 1382, 1356 },
		{ 1555, 1539 },
		{ 1556, 1540 },
		{ 1557, 1541 },
		{ 1388, 1361 },
		{ 1563, 1554 },
		{ 1564, 1555 },
		{ 1387, 1361 },
		{ 1381, 1356 },
		{ 1565, 1556 },
		{ 1566, 1557 },
		{ 1391, 1366 },
		{ 1570, 1564 },
		{ 1392, 1367 },
		{ 1572, 1566 },
		{ 159, 4 },
		{ 1575, 1570 },
		{ 1558, 1548 },
		{ 1576, 1572 },
		{ 2483, 2454 },
		{ 1380, 1356 },
		{ 2324, 2296 },
		{ 1578, 1575 },
		{ 1579, 1576 },
		{ 1580, 1578 },
		{ 1581, 1579 },
		{ 1376, 1580 },
		{ 1393, 1368 },
		{ 1394, 1370 },
		{ 84, 4 },
		{ 158, 4 },
		{ 83, 4 },
		{ 83, 4 },
		{ 83, 4 },
		{ 83, 4 },
		{ 83, 4 },
		{ 83, 4 },
		{ 83, 4 },
		{ 83, 4 },
		{ 83, 4 },
		{ 83, 4 },
		{ 1395, 1371 },
		{ 1398, 1375 },
		{ 1399, 1378 },
		{ 1400, 1379 },
		{ 0, 2454 },
		{ 1560, 1548 },
		{ 1401, 1380 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 74, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 82, 4 },
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
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 1402, 1381 },
		{ 81, 4 },
		{ 1403, 1382 },
		{ 1404, 1383 },
		{ 85, 4 },
		{ 1405, 1384 },
		{ 85, 4 },
		{ 73, 4 },
		{ 79, 4 },
		{ 77, 4 },
		{ 85, 4 },
		{ 78, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 76, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 80, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 75, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 85, 4 },
		{ 2997, 2996 },
		{ 1406, 1385 },
		{ 1408, 1386 },
		{ 2996, 2996 },
		{ 1409, 1387 },
		{ 1407, 1385 },
		{ 1410, 1388 },
		{ 1413, 1391 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 1414, 1392 },
		{ 2996, 2996 },
		{ 1415, 1393 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 1416, 1394 },
		{ 1417, 1395 },
		{ 1420, 1398 },
		{ 1421, 1399 },
		{ 1422, 1400 },
		{ 1423, 1401 },
		{ 1424, 1402 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 1425, 1403 },
		{ 1426, 1404 },
		{ 1427, 1405 },
		{ 1428, 1406 },
		{ 1429, 1407 },
		{ 1430, 1408 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 2996, 2996 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1431, 1409 },
		{ 1432, 1410 },
		{ 154, 152 },
		{ 104, 89 },
		{ 105, 90 },
		{ 106, 91 },
		{ 107, 92 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 108, 93 },
		{ 109, 94 },
		{ 110, 95 },
		{ 111, 96 },
		{ 1376, 1582 },
		{ 112, 97 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 1376, 1582 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 114, 99 },
		{ 120, 104 },
		{ 121, 105 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 109 },
		{ 125, 110 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 126, 111 },
		{ 127, 112 },
		{ 129, 114 },
		{ 134, 120 },
		{ 2228, 2477 },
		{ 135, 121 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 2228, 2477 },
		{ 1377, 1581 },
		{ 0, 1581 },
		{ 136, 122 },
		{ 137, 123 },
		{ 138, 124 },
		{ 139, 125 },
		{ 140, 127 },
		{ 141, 129 },
		{ 142, 134 },
		{ 143, 135 },
		{ 144, 136 },
		{ 2235, 2214 },
		{ 2237, 2215 },
		{ 2240, 2216 },
		{ 2657, 2657 },
		{ 2241, 2217 },
		{ 2238, 2216 },
		{ 2248, 2219 },
		{ 2251, 2220 },
		{ 2239, 2216 },
		{ 2244, 2218 },
		{ 2252, 2221 },
		{ 2253, 2222 },
		{ 1377, 1581 },
		{ 2243, 2218 },
		{ 2242, 2217 },
		{ 2254, 2223 },
		{ 2255, 2224 },
		{ 2256, 2225 },
		{ 2228, 2228 },
		{ 2249, 2229 },
		{ 2236, 2230 },
		{ 2247, 2231 },
		{ 2262, 2235 },
		{ 145, 137 },
		{ 2264, 2237 },
		{ 2265, 2238 },
		{ 2657, 2657 },
		{ 2250, 2229 },
		{ 2245, 2218 },
		{ 2246, 2218 },
		{ 2266, 2239 },
		{ 2267, 2240 },
		{ 2268, 2241 },
		{ 2269, 2242 },
		{ 2270, 2243 },
		{ 2271, 2244 },
		{ 2272, 2245 },
		{ 2273, 2246 },
		{ 2274, 2247 },
		{ 2275, 2248 },
		{ 2276, 2249 },
		{ 0, 1581 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2277, 2250 },
		{ 2278, 2251 },
		{ 2279, 2252 },
		{ 2280, 2253 },
		{ 2283, 2255 },
		{ 2284, 2256 },
		{ 2291, 2262 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 67, 7 },
		{ 2293, 2264 },
		{ 2294, 2265 },
		{ 2295, 2266 },
		{ 2657, 2657 },
		{ 1582, 1581 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2657, 2657 },
		{ 2296, 2267 },
		{ 2297, 2268 },
		{ 2298, 2269 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 2299, 2270 },
		{ 2300, 2271 },
		{ 2301, 2272 },
		{ 2302, 2273 },
		{ 2303, 2274 },
		{ 2304, 2275 },
		{ 146, 138 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 2307, 2278 },
		{ 2308, 2279 },
		{ 2309, 2280 },
		{ 2310, 2281 },
		{ 1204, 7 },
		{ 2311, 2282 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 1204, 7 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 2312, 2283 },
		{ 2313, 2284 },
		{ 2319, 2291 },
		{ 147, 140 },
		{ 2322, 2294 },
		{ 2323, 2295 },
		{ 148, 141 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 2327, 2298 },
		{ 2325, 2297 },
		{ 2328, 2299 },
		{ 149, 142 },
		{ 0, 1440 },
		{ 2326, 2297 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1440 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 2330, 2301 },
		{ 2331, 2302 },
		{ 2333, 2304 },
		{ 2337, 2307 },
		{ 2338, 2308 },
		{ 2339, 2309 },
		{ 2340, 2310 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 2341, 2311 },
		{ 2342, 2312 },
		{ 2343, 2313 },
		{ 2344, 2319 },
		{ 0, 1852 },
		{ 2347, 2322 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 1852 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 150, 144 },
		{ 2352, 2326 },
		{ 2353, 2327 },
		{ 2354, 2328 },
		{ 2357, 2331 },
		{ 2359, 2333 },
		{ 2363, 2337 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 2364, 2338 },
		{ 2365, 2339 },
		{ 2366, 2340 },
		{ 2367, 2341 },
		{ 0, 2046 },
		{ 151, 147 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 0, 2046 },
		{ 2281, 2254 },
		{ 2369, 2343 },
		{ 152, 148 },
		{ 2374, 2347 },
		{ 2377, 2352 },
		{ 2378, 2353 },
		{ 2380, 2354 },
		{ 2383, 2357 },
		{ 2385, 2359 },
		{ 2282, 2254 },
		{ 2379, 2354 },
		{ 2389, 2363 },
		{ 2390, 2364 },
		{ 2391, 2365 },
		{ 153, 150 },
		{ 2395, 2369 },
		{ 2400, 2374 },
		{ 2405, 2378 },
		{ 2406, 2379 },
		{ 2407, 2380 },
		{ 2410, 2383 },
		{ 2412, 2385 },
		{ 2416, 2389 },
		{ 2417, 2390 },
		{ 2418, 2391 },
		{ 2423, 2395 },
		{ 2428, 2400 },
		{ 2431, 2405 },
		{ 2432, 2406 },
		{ 89, 73 },
		{ 2436, 2410 },
		{ 155, 153 },
		{ 2442, 2416 },
		{ 2443, 2417 },
		{ 156, 155 },
		{ 2449, 2423 },
		{ 2454, 2428 },
		{ 157, 156 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 92, 75 },
		{ 93, 76 },
		{ 94, 77 },
		{ 95, 78 },
		{ 96, 79 },
		{ 97, 80 },
		{ 99, 82 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 2314, 2285 },
		{ 2316, 2288 },
		{ 2314, 2285 },
		{ 2316, 2288 },
		{ 85, 157 },
		{ 2823, 49 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 85, 157 },
		{ 90, 74 },
		{ 1219, 1216 },
		{ 132, 118 },
		{ 1219, 1216 },
		{ 132, 118 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 116, 101 },
		{ 1214, 1211 },
		{ 116, 101 },
		{ 1214, 1211 },
		{ 0, 2824 },
		{ 2553, 2528 },
		{ 91, 74 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 2286, 2258 },
		{ 130, 115 },
		{ 2286, 2258 },
		{ 130, 115 },
		{ 1204, 1204 },
		{ 2564, 2539 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 0, 2449 },
		{ 0, 2449 },
		{ 1217, 1213 },
		{ 1728, 1727 },
		{ 1217, 1213 },
		{ 86, 49 },
		{ 2152, 2149 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 1770, 1769 },
		{ 2693, 2678 },
		{ 2710, 2696 },
		{ 1255, 1254 },
		{ 1681, 1680 },
		{ 2529, 2500 },
		{ 0, 2449 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 1725, 1724 },
		{ 1252, 1251 },
		{ 2968, 2967 },
		{ 1636, 1635 },
		{ 2988, 2988 },
		{ 3047, 3046 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 1315, 1314 },
		{ 2004, 1980 },
		{ 182, 173 },
		{ 192, 173 },
		{ 184, 173 },
		{ 1610, 1609 },
		{ 179, 173 },
		{ 1315, 1314 },
		{ 183, 173 },
		{ 2815, 2814 },
		{ 181, 173 },
		{ 2192, 2191 },
		{ 1610, 1609 },
		{ 2530, 2501 },
		{ 190, 173 },
		{ 189, 173 },
		{ 180, 173 },
		{ 188, 173 },
		{ 2477, 2449 },
		{ 185, 173 },
		{ 187, 173 },
		{ 178, 173 },
		{ 2003, 1980 },
		{ 2197, 2196 },
		{ 186, 173 },
		{ 191, 173 },
		{ 2257, 2226 },
		{ 1684, 1683 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 2226, 2226 },
		{ 3012, 3012 },
		{ 3012, 3012 },
		{ 3012, 3012 },
		{ 3012, 3012 },
		{ 3012, 3012 },
		{ 3012, 3012 },
		{ 3012, 3012 },
		{ 3012, 3012 },
		{ 3012, 3012 },
		{ 3012, 3012 },
		{ 1210, 1207 },
		{ 2258, 2226 },
		{ 1207, 1207 },
		{ 1207, 1207 },
		{ 1207, 1207 },
		{ 1207, 1207 },
		{ 1207, 1207 },
		{ 1207, 1207 },
		{ 1207, 1207 },
		{ 1207, 1207 },
		{ 1207, 1207 },
		{ 1207, 1207 },
		{ 1803, 1779 },
		{ 443, 400 },
		{ 448, 400 },
		{ 445, 400 },
		{ 444, 400 },
		{ 447, 400 },
		{ 442, 400 },
		{ 3018, 65 },
		{ 441, 400 },
		{ 1657, 1656 },
		{ 67, 65 },
		{ 1211, 1207 },
		{ 446, 400 },
		{ 1802, 1779 },
		{ 449, 400 },
		{ 2856, 2855 },
		{ 2476, 2448 },
		{ 1989, 1967 },
		{ 2908, 2907 },
		{ 2597, 2572 },
		{ 440, 400 },
		{ 2258, 2226 },
		{ 100, 83 },
		{ 3013, 3012 },
		{ 83, 83 },
		{ 83, 83 },
		{ 83, 83 },
		{ 83, 83 },
		{ 83, 83 },
		{ 83, 83 },
		{ 83, 83 },
		{ 83, 83 },
		{ 83, 83 },
		{ 83, 83 },
		{ 1268, 1267 },
		{ 2949, 2948 },
		{ 2019, 1998 },
		{ 2971, 2970 },
		{ 2979, 2978 },
		{ 2048, 2030 },
		{ 2062, 2045 },
		{ 1828, 1809 },
		{ 1850, 1831 },
		{ 1211, 1207 },
		{ 1741, 1740 },
		{ 101, 83 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 2770, 2769 },
		{ 1212, 1212 },
		{ 1212, 1212 },
		{ 1212, 1212 },
		{ 1212, 1212 },
		{ 1212, 1212 },
		{ 1212, 1212 },
		{ 1212, 1212 },
		{ 1212, 1212 },
		{ 1212, 1212 },
		{ 1212, 1212 },
		{ 1213, 1210 },
		{ 2462, 2462 },
		{ 2462, 2462 },
		{ 2392, 2392 },
		{ 2392, 2392 },
		{ 2434, 2408 },
		{ 3030, 3026 },
		{ 2810, 2809 },
		{ 3050, 3049 },
		{ 3016, 65 },
		{ 101, 83 },
		{ 1216, 1212 },
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
		{ 3014, 65 },
		{ 3066, 3063 },
		{ 2462, 2462 },
		{ 3072, 3069 },
		{ 2392, 2392 },
		{ 2567, 2542 },
		{ 2079, 2066 },
		{ 2092, 2077 },
		{ 2093, 2078 },
		{ 3004, 63 },
		{ 1213, 1210 },
		{ 118, 102 },
		{ 67, 63 },
		{ 2604, 2579 },
		{ 1344, 1343 },
		{ 2615, 2591 },
		{ 2619, 2595 },
		{ 2628, 2605 },
		{ 2376, 2349 },
		{ 2634, 2611 },
		{ 2646, 2627 },
		{ 1216, 1212 },
		{ 3017, 65 },
		{ 100, 100 },
		{ 100, 100 },
		{ 100, 100 },
		{ 100, 100 },
		{ 100, 100 },
		{ 100, 100 },
		{ 100, 100 },
		{ 100, 100 },
		{ 100, 100 },
		{ 100, 100 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 2257, 2257 },
		{ 118, 102 },
		{ 115, 100 },
		{ 2648, 2629 },
		{ 2305, 2305 },
		{ 2305, 2305 },
		{ 1206, 9 },
		{ 2662, 2643 },
		{ 2439, 2439 },
		{ 2439, 2439 },
		{ 67, 9 },
		{ 2663, 2644 },
		{ 2285, 2257 },
		{ 2153, 2150 },
		{ 2259, 2259 },
		{ 2259, 2259 },
		{ 2259, 2259 },
		{ 2259, 2259 },
		{ 2259, 2259 },
		{ 2259, 2259 },
		{ 2259, 2259 },
		{ 2259, 2259 },
		{ 2259, 2259 },
		{ 2259, 2259 },
		{ 2168, 2167 },
		{ 2673, 2654 },
		{ 3003, 63 },
		{ 2305, 2305 },
		{ 1236, 1235 },
		{ 1206, 9 },
		{ 3002, 63 },
		{ 2439, 2439 },
		{ 2873, 2873 },
		{ 2873, 2873 },
		{ 115, 100 },
		{ 2288, 2259 },
		{ 2920, 2920 },
		{ 2920, 2920 },
		{ 2678, 2661 },
		{ 2492, 2462 },
		{ 2491, 2462 },
		{ 2420, 2392 },
		{ 2419, 2392 },
		{ 1208, 9 },
		{ 2285, 2257 },
		{ 1207, 9 },
		{ 1207, 9 },
		{ 1207, 9 },
		{ 1207, 9 },
		{ 1207, 9 },
		{ 1207, 9 },
		{ 1207, 9 },
		{ 1207, 9 },
		{ 1207, 9 },
		{ 1207, 9 },
		{ 2873, 2873 },
		{ 2441, 2441 },
		{ 2441, 2441 },
		{ 2687, 2671 },
		{ 2920, 2920 },
		{ 2450, 2450 },
		{ 2450, 2450 },
		{ 2324, 2324 },
		{ 2324, 2324 },
		{ 2469, 2469 },
		{ 2469, 2469 },
		{ 1658, 1657 },
		{ 2288, 2259 },
		{ 2348, 2348 },
		{ 2348, 2348 },
		{ 2483, 2483 },
		{ 2483, 2483 },
		{ 2502, 2502 },
		{ 2502, 2502 },
		{ 2554, 2554 },
		{ 2554, 2554 },
		{ 2647, 2647 },
		{ 2647, 2647 },
		{ 2696, 2681 },
		{ 2441, 2441 },
		{ 3000, 63 },
		{ 2194, 2193 },
		{ 3001, 63 },
		{ 2450, 2450 },
		{ 2713, 2699 },
		{ 2324, 2324 },
		{ 2726, 2720 },
		{ 2469, 2469 },
		{ 2728, 2722 },
		{ 2368, 2368 },
		{ 2368, 2368 },
		{ 2348, 2348 },
		{ 2734, 2729 },
		{ 2483, 2483 },
		{ 2740, 2739 },
		{ 2502, 2502 },
		{ 1660, 1659 },
		{ 2554, 2554 },
		{ 2453, 2427 },
		{ 2647, 2647 },
		{ 2334, 2305 },
		{ 2806, 2806 },
		{ 2806, 2806 },
		{ 2834, 2834 },
		{ 2834, 2834 },
		{ 2773, 2772 },
		{ 2444, 2444 },
		{ 2444, 2444 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2782, 2781 },
		{ 2335, 2305 },
		{ 2368, 2368 },
		{ 2411, 2411 },
		{ 2411, 2411 },
		{ 2465, 2439 },
		{ 2795, 2794 },
		{ 2468, 2468 },
		{ 2468, 2468 },
		{ 2438, 2438 },
		{ 2438, 2438 },
		{ 2199, 2199 },
		{ 2199, 2199 },
		{ 1831, 1812 },
		{ 2806, 2806 },
		{ 2455, 2429 },
		{ 2834, 2834 },
		{ 2263, 2263 },
		{ 2263, 2263 },
		{ 2444, 2444 },
		{ 2206, 2205 },
		{ 2329, 2329 },
		{ 2887, 2887 },
		{ 2887, 2887 },
		{ 2928, 2928 },
		{ 2928, 2928 },
		{ 2411, 2411 },
		{ 2370, 2370 },
		{ 2370, 2370 },
		{ 2874, 2873 },
		{ 2468, 2468 },
		{ 2818, 2817 },
		{ 2438, 2438 },
		{ 2921, 2920 },
		{ 2199, 2199 },
		{ 1837, 1818 },
		{ 2560, 2560 },
		{ 2560, 2560 },
		{ 1271, 1270 },
		{ 1857, 1838 },
		{ 2263, 2263 },
		{ 1310, 1309 },
		{ 2607, 2607 },
		{ 2607, 2607 },
		{ 2478, 2450 },
		{ 2887, 2887 },
		{ 1884, 1868 },
		{ 2928, 2928 },
		{ 2458, 2458 },
		{ 2458, 2458 },
		{ 2370, 2370 },
		{ 2846, 2845 },
		{ 2479, 2450 },
		{ 2467, 2441 },
		{ 2650, 2650 },
		{ 2650, 2650 },
		{ 2332, 2332 },
		{ 2332, 2332 },
		{ 2349, 2324 },
		{ 2560, 2560 },
		{ 2499, 2469 },
		{ 2835, 2834 },
		{ 2843, 2843 },
		{ 2843, 2843 },
		{ 2375, 2348 },
		{ 2607, 2607 },
		{ 2513, 2483 },
		{ 1886, 1871 },
		{ 2531, 2502 },
		{ 2472, 2444 },
		{ 2579, 2554 },
		{ 2458, 2458 },
		{ 2666, 2647 },
		{ 2872, 2871 },
		{ 2584, 2584 },
		{ 2584, 2584 },
		{ 2401, 2375 },
		{ 2650, 2650 },
		{ 1888, 1873 },
		{ 2332, 2332 },
		{ 2902, 2901 },
		{ 1938, 1936 },
		{ 2911, 2910 },
		{ 2919, 2918 },
		{ 2394, 2368 },
		{ 2843, 2843 },
		{ 2470, 2444 },
		{ 1686, 1685 },
		{ 1708, 1707 },
		{ 2943, 2942 },
		{ 2471, 2444 },
		{ 1720, 1719 },
		{ 2952, 2951 },
		{ 2835, 2834 },
		{ 2962, 2961 },
		{ 1256, 1255 },
		{ 2807, 2806 },
		{ 2584, 2584 },
		{ 2413, 2386 },
		{ 2973, 2972 },
		{ 2415, 2388 },
		{ 2355, 2329 },
		{ 2982, 2981 },
		{ 1605, 1604 },
		{ 1729, 1728 },
		{ 2060, 2043 },
		{ 2437, 2411 },
		{ 2518, 2489 },
		{ 2061, 2044 },
		{ 1333, 1332 },
		{ 2498, 2468 },
		{ 2426, 2398 },
		{ 2464, 2438 },
		{ 3026, 3022 },
		{ 2200, 2199 },
		{ 2537, 2510 },
		{ 2550, 2525 },
		{ 1744, 1743 },
		{ 3052, 3051 },
		{ 2292, 2263 },
		{ 3054, 3054 },
		{ 2557, 2532 },
		{ 2430, 2402 },
		{ 3079, 3077 },
		{ 2888, 2887 },
		{ 1810, 1785 },
		{ 2929, 2928 },
		{ 2703, 2688 },
		{ 1809, 1785 },
		{ 2396, 2370 },
		{ 2321, 2321 },
		{ 2321, 2321 },
		{ 1999, 1974 },
		{ 1303, 1302 },
		{ 1899, 1886 },
		{ 1998, 1974 },
		{ 1918, 1908 },
		{ 1926, 1918 },
		{ 2585, 2560 },
		{ 1304, 1303 },
		{ 2484, 2455 },
		{ 2744, 2743 },
		{ 1264, 1263 },
		{ 3054, 3054 },
		{ 2630, 2607 },
		{ 2765, 2764 },
		{ 1702, 1701 },
		{ 1703, 1702 },
		{ 2777, 2776 },
		{ 1240, 1239 },
		{ 2487, 2458 },
		{ 2500, 2470 },
		{ 2504, 2474 },
		{ 2321, 2321 },
		{ 1712, 1711 },
		{ 2512, 2481 },
		{ 2669, 2650 },
		{ 1568, 1562 },
		{ 2358, 2332 },
		{ 2020, 1999 },
		{ 2039, 2018 },
		{ 2041, 2020 },
		{ 2829, 2827 },
		{ 2044, 2023 },
		{ 2844, 2843 },
		{ 1225, 1224 },
		{ 2538, 2512 },
		{ 2850, 2849 },
		{ 1598, 1597 },
		{ 1599, 1598 },
		{ 1737, 1736 },
		{ 1339, 1338 },
		{ 1279, 1278 },
		{ 2571, 2546 },
		{ 2609, 2584 },
		{ 1760, 1759 },
		{ 2582, 2557 },
		{ 1761, 1760 },
		{ 1766, 1765 },
		{ 1626, 1625 },
		{ 1627, 1626 },
		{ 1633, 1632 },
		{ 2961, 2960 },
		{ 1634, 1633 },
		{ 2447, 2421 },
		{ 1286, 1285 },
		{ 2451, 2425 },
		{ 2635, 2612 },
		{ 2636, 2613 },
		{ 2638, 2616 },
		{ 2639, 2619 },
		{ 1829, 1810 },
		{ 1652, 1651 },
		{ 1836, 1817 },
		{ 2456, 2430 },
		{ 2665, 2646 },
		{ 2209, 2208 },
		{ 1653, 1652 },
		{ 3042, 3041 },
		{ 3043, 3042 },
		{ 1848, 1829 },
		{ 2674, 2655 },
		{ 1287, 1286 },
		{ 1289, 1288 },
		{ 1562, 1553 },
		{ 1676, 1675 },
		{ 1677, 1676 },
		{ 1335, 1334 },
		{ 3057, 3054 },
		{ 2767, 2766 },
		{ 2566, 2541 },
		{ 1631, 1630 },
		{ 1248, 1247 },
		{ 3056, 3054 },
		{ 1341, 1340 },
		{ 3055, 3054 },
		{ 2793, 2792 },
		{ 2581, 2556 },
		{ 2804, 2803 },
		{ 2463, 2437 },
		{ 2409, 2382 },
		{ 2586, 2561 },
		{ 2361, 2335 },
		{ 1688, 1687 },
		{ 2146, 2139 },
		{ 2149, 2144 },
		{ 2616, 2592 },
		{ 2833, 2831 },
		{ 1281, 1280 },
		{ 2346, 2321 },
		{ 2620, 2596 },
		{ 2473, 2445 },
		{ 1638, 1637 },
		{ 2165, 2162 },
		{ 1933, 1928 },
		{ 1227, 1226 },
		{ 2345, 2345 },
		{ 2345, 2345 },
		{ 2900, 2899 },
		{ 1254, 1253 },
		{ 1821, 1802 },
		{ 2494, 2464 },
		{ 2652, 2633 },
		{ 2656, 2637 },
		{ 2495, 2465 },
		{ 2941, 2940 },
		{ 2497, 2467 },
		{ 2196, 2195 },
		{ 1714, 1713 },
		{ 1317, 1316 },
		{ 2202, 2201 },
		{ 1722, 1721 },
		{ 2208, 2207 },
		{ 2677, 2660 },
		{ 2010, 1989 },
		{ 2684, 2668 },
		{ 1832, 1813 },
		{ 2527, 2498 },
		{ 1331, 1330 },
		{ 2345, 2345 },
		{ 2698, 2683 },
		{ 2031, 2010 },
		{ 1727, 1726 },
		{ 1840, 1821 },
		{ 2714, 2700 },
		{ 2715, 2702 },
		{ 3033, 3030 },
		{ 1612, 1611 },
		{ 2727, 2721 },
		{ 2545, 2520 },
		{ 1662, 1661 },
		{ 2735, 2733 },
		{ 3054, 3053 },
		{ 2738, 2736 },
		{ 3062, 3059 },
		{ 1234, 1233 },
		{ 3070, 3067 },
		{ 1867, 1850 },
		{ 2558, 2533 },
		{ 3082, 3081 },
		{ 2306, 2306 },
		{ 2306, 2306 },
		{ 2724, 2724 },
		{ 2724, 2724 },
		{ 2393, 2393 },
		{ 2393, 2393 },
		{ 2610, 2585 },
		{ 1767, 1766 },
		{ 2408, 2381 },
		{ 1343, 1342 },
		{ 2720, 2712 },
		{ 2533, 2504 },
		{ 2510, 2479 },
		{ 2629, 2606 },
		{ 1871, 1856 },
		{ 1329, 1328 },
		{ 2516, 2487 },
		{ 2030, 2009 },
		{ 2371, 2345 },
		{ 2596, 2571 },
		{ 2555, 2530 },
		{ 2685, 2669 },
		{ 2845, 2844 },
		{ 2306, 2306 },
		{ 2643, 2623 },
		{ 2724, 2724 },
		{ 2644, 2625 },
		{ 2393, 2393 },
		{ 2525, 2496 },
		{ 1707, 1706 },
		{ 2649, 2630 },
		{ 1799, 1776 },
		{ 2803, 2802 },
		{ 1706, 1705 },
		{ 2159, 2156 },
		{ 2637, 2615 },
		{ 2162, 2160 },
		{ 1646, 1645 },
		{ 2820, 2819 },
		{ 1935, 1932 },
		{ 1334, 1333 },
		{ 1798, 1776 },
		{ 2515, 2486 },
		{ 2435, 2409 },
		{ 1941, 1940 },
		{ 2523, 2494 },
		{ 1814, 1791 },
		{ 1453, 1433 },
		{ 1273, 1272 },
		{ 2372, 2345 },
		{ 2848, 2847 },
		{ 1396, 1372 },
		{ 2855, 2854 },
		{ 1620, 1619 },
		{ 2381, 2355 },
		{ 2672, 2653 },
		{ 2005, 1982 },
		{ 2009, 1988 },
		{ 1661, 1660 },
		{ 2904, 2903 },
		{ 1833, 1814 },
		{ 2552, 2527 },
		{ 2913, 2912 },
		{ 1835, 1816 },
		{ 2024, 2003 },
		{ 2026, 2005 },
		{ 2028, 2007 },
		{ 1328, 1327 },
		{ 2945, 2944 },
		{ 1670, 1669 },
		{ 2702, 2687 },
		{ 2954, 2953 },
		{ 1238, 1237 },
		{ 2704, 2689 },
		{ 2568, 2543 },
		{ 1418, 1396 },
		{ 1632, 1631 },
		{ 2975, 2974 },
		{ 1247, 1246 },
		{ 2059, 2042 },
		{ 2984, 2983 },
		{ 2721, 2713 },
		{ 1866, 1849 },
		{ 1746, 1745 },
		{ 2592, 2567 },
		{ 1754, 1753 },
		{ 2733, 2728 },
		{ 1297, 1296 },
		{ 2074, 2058 },
		{ 2605, 2580 },
		{ 3031, 3027 },
		{ 1881, 1865 },
		{ 2742, 2741 },
		{ 1687, 1686 },
		{ 1635, 1634 },
		{ 2124, 2106 },
		{ 2336, 2306 },
		{ 2493, 2463 },
		{ 2729, 2724 },
		{ 3053, 3052 },
		{ 2421, 2393 },
		{ 2125, 2107 },
		{ 1696, 1695 },
		{ 3059, 3056 },
		{ 2775, 2774 },
		{ 1592, 1591 },
		{ 2151, 2148 },
		{ 2632, 2609 },
		{ 1346, 1345 },
		{ 3081, 3079 },
		{ 2797, 2796 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 2788, 2788 },
		{ 2788, 2788 },
		{ 2936, 2936 },
		{ 2936, 2936 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 2457, 2457 },
		{ 2457, 2457 },
		{ 2137, 2124 },
		{ 2138, 2125 },
		{ 2830, 2828 },
		{ 1239, 1238 },
		{ 2532, 2503 },
		{ 2148, 2143 },
		{ 2534, 2505 },
		{ 2536, 2509 },
		{ 1830, 1811 },
		{ 2849, 2848 },
		{ 1942, 1941 },
		{ 2452, 2426 },
		{ 1732, 1731 },
		{ 2895, 2895 },
		{ 2857, 2856 },
		{ 2788, 2788 },
		{ 2866, 2865 },
		{ 2936, 2936 },
		{ 1274, 1273 },
		{ 2433, 2433 },
		{ 1296, 1295 },
		{ 2457, 2457 },
		{ 2882, 2881 },
		{ 2883, 2882 },
		{ 2885, 2884 },
		{ 2161, 2159 },
		{ 2688, 2672 },
		{ 2898, 2897 },
		{ 1742, 1741 },
		{ 1645, 1644 },
		{ 1695, 1694 },
		{ 2905, 2904 },
		{ 2007, 1985 },
		{ 2909, 2908 },
		{ 2008, 1987 },
		{ 1838, 1819 },
		{ 2914, 2913 },
		{ 1747, 1746 },
		{ 1753, 1752 },
		{ 2926, 2925 },
		{ 2198, 2197 },
		{ 1603, 1602 },
		{ 2939, 2938 },
		{ 2717, 2704 },
		{ 2021, 2000 },
		{ 1326, 1325 },
		{ 2946, 2945 },
		{ 2289, 2260 },
		{ 2950, 2949 },
		{ 2593, 2568 },
		{ 2595, 2570 },
		{ 2955, 2954 },
		{ 2960, 2959 },
		{ 1473, 1454 },
		{ 113, 98 },
		{ 1868, 1851 },
		{ 1619, 1618 },
		{ 2608, 2583 },
		{ 1269, 1268 },
		{ 2976, 2975 },
		{ 1873, 1858 },
		{ 2980, 2979 },
		{ 2743, 2742 },
		{ 2043, 2022 },
		{ 2985, 2984 },
		{ 1259, 1258 },
		{ 1883, 1867 },
		{ 1397, 1374 },
		{ 2998, 2995 },
		{ 1669, 1668 },
		{ 2771, 2770 },
		{ 1347, 1346 },
		{ 1308, 1307 },
		{ 3024, 3020 },
		{ 2776, 2775 },
		{ 3027, 3023 },
		{ 1907, 1895 },
		{ 2501, 2471 },
		{ 2066, 2049 },
		{ 3036, 3033 },
		{ 2791, 2790 },
		{ 2360, 2334 },
		{ 1816, 1794 },
		{ 2362, 2336 },
		{ 2896, 2895 },
		{ 2798, 2797 },
		{ 2789, 2788 },
		{ 1920, 1911 },
		{ 2937, 2936 },
		{ 1820, 1801 },
		{ 2459, 2433 },
		{ 1932, 1927 },
		{ 2486, 2457 },
		{ 2517, 2488 },
		{ 2104, 2090 },
		{ 2816, 2815 },
		{ 1433, 1411 },
		{ 1591, 1590 },
		{ 2821, 2820 },
		{ 2126, 2108 },
		{ 2659, 2640 },
		{ 2813, 2813 },
		{ 2813, 2813 },
		{ 2768, 2768 },
		{ 2768, 2768 },
		{ 2490, 2490 },
		{ 2490, 2490 },
		{ 2977, 2977 },
		{ 2977, 2977 },
		{ 1739, 1739 },
		{ 1739, 1739 },
		{ 1266, 1266 },
		{ 1266, 1266 },
		{ 2682, 2682 },
		{ 2682, 2682 },
		{ 2356, 2356 },
		{ 2356, 2356 },
		{ 2947, 2947 },
		{ 2947, 2947 },
		{ 2906, 2906 },
		{ 2906, 2906 },
		{ 1937, 1935 },
		{ 1593, 1592 },
		{ 2399, 2373 },
		{ 2813, 2813 },
		{ 1656, 1655 },
		{ 2768, 2768 },
		{ 3034, 3031 },
		{ 2490, 2490 },
		{ 1854, 1835 },
		{ 2977, 2977 },
		{ 1298, 1297 },
		{ 1739, 1739 },
		{ 1671, 1670 },
		{ 1266, 1266 },
		{ 1710, 1709 },
		{ 2682, 2682 },
		{ 2461, 2435 },
		{ 2356, 2356 },
		{ 2204, 2203 },
		{ 2947, 2947 },
		{ 1755, 1754 },
		{ 2906, 2906 },
		{ 2075, 2059 },
		{ 2154, 2151 },
		{ 2587, 2562 },
		{ 1472, 1453 },
		{ 1647, 1646 },
		{ 3069, 3066 },
		{ 1697, 1696 },
		{ 1621, 1620 },
		{ 1882, 1866 },
		{ 2170, 2169 },
		{ 2047, 2028 },
		{ 2966, 2966 },
		{ 2966, 2966 },
		{ 2931, 2931 },
		{ 2931, 2931 },
		{ 1723, 1723 },
		{ 1723, 1723 },
		{ 2890, 2890 },
		{ 2890, 2890 },
		{ 2783, 2783 },
		{ 2783, 2783 },
		{ 1250, 1250 },
		{ 1250, 1250 },
		{ 2924, 2924 },
		{ 2924, 2924 },
		{ 1261, 1261 },
		{ 1261, 1261 },
		{ 1734, 1734 },
		{ 1734, 1734 },
		{ 1841, 1822 },
		{ 2123, 2105 },
		{ 1683, 1682 },
		{ 1301, 1300 },
		{ 1852, 1833 },
		{ 2966, 2966 },
		{ 1537, 1511 },
		{ 2931, 2931 },
		{ 2565, 2540 },
		{ 1723, 1723 },
		{ 1596, 1595 },
		{ 2890, 2890 },
		{ 1817, 1797 },
		{ 2783, 2783 },
		{ 2642, 2622 },
		{ 1250, 1250 },
		{ 2719, 2711 },
		{ 2924, 2924 },
		{ 1313, 1312 },
		{ 1261, 1261 },
		{ 2569, 2544 },
		{ 1734, 1734 },
		{ 2814, 2813 },
		{ 1624, 1623 },
		{ 2769, 2768 },
		{ 2460, 2434 },
		{ 2519, 2490 },
		{ 2580, 2555 },
		{ 2978, 2977 },
		{ 1263, 1262 },
		{ 1740, 1739 },
		{ 2045, 2024 },
		{ 1267, 1266 },
		{ 2046, 2026 },
		{ 2697, 2682 },
		{ 1440, 1418 },
		{ 2382, 2356 },
		{ 2158, 2155 },
		{ 2948, 2947 },
		{ 2664, 2645 },
		{ 2907, 2906 },
		{ 2841, 2840 },
		{ 3040, 3039 },
		{ 1758, 1757 },
		{ 1700, 1699 },
		{ 1650, 1649 },
		{ 3049, 3048 },
		{ 1674, 1673 },
		{ 1232, 1231 },
		{ 2475, 2447 },
		{ 1768, 1767 },
		{ 2065, 2048 },
		{ 1736, 1735 },
		{ 2193, 2192 },
		{ 1898, 1885 },
		{ 2686, 2670 },
		{ 1608, 1607 },
		{ 2970, 2969 },
		{ 3075, 3072 },
		{ 2018, 1997 },
		{ 1284, 1283 },
		{ 1910, 1900 },
		{ 2576, 2576 },
		{ 2576, 2576 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 2440, 2440 },
		{ 2440, 2440 },
		{ 2574, 2574 },
		{ 2574, 2574 },
		{ 2837, 2836 },
		{ 2864, 2863 },
		{ 1982, 2173 },
		{ 1372, 1363 },
		{ 2967, 2966 },
		{ 1791, 1947 },
		{ 2932, 2931 },
		{ 2583, 2558 },
		{ 1724, 1723 },
		{ 2796, 2795 },
		{ 2891, 2890 },
		{ 1454, 1434 },
		{ 2784, 2783 },
		{ 2398, 2372 },
		{ 1251, 1250 },
		{ 2576, 2576 },
		{ 2925, 2924 },
		{ 1228, 1228 },
		{ 1262, 1261 },
		{ 2440, 2440 },
		{ 1735, 1734 },
		{ 2574, 2574 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 1218, 1218 },
		{ 1218, 1218 },
		{ 1218, 1218 },
		{ 1218, 1218 },
		{ 1218, 1218 },
		{ 1218, 1218 },
		{ 1218, 1218 },
		{ 1218, 1218 },
		{ 1218, 1218 },
		{ 1218, 1218 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 2859, 2859 },
		{ 2859, 2859 },
		{ 1282, 1282 },
		{ 1282, 1282 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 2315, 2315 },
		{ 1606, 1605 },
		{ 2944, 2943 },
		{ 1800, 1778 },
		{ 1311, 1310 },
		{ 2681, 2664 },
		{ 1940, 1938 },
		{ 1609, 1608 },
		{ 2156, 2153 },
		{ 1709, 1708 },
		{ 2859, 2859 },
		{ 2953, 2952 },
		{ 1282, 1282 },
		{ 0, 1208 },
		{ 0, 84 },
		{ 1345, 1344 },
		{ 2819, 2818 },
		{ 2689, 2673 },
		{ 2601, 2576 },
		{ 1818, 1798 },
		{ 1229, 1228 },
		{ 2528, 2499 },
		{ 2466, 2440 },
		{ 1745, 1744 },
		{ 2599, 2574 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 117, 117 },
		{ 117, 117 },
		{ 117, 117 },
		{ 117, 117 },
		{ 117, 117 },
		{ 117, 117 },
		{ 117, 117 },
		{ 117, 117 },
		{ 117, 117 },
		{ 117, 117 },
		{ 0, 1208 },
		{ 0, 84 },
		{ 2990, 2990 },
		{ 2990, 2990 },
		{ 2990, 2990 },
		{ 2990, 2990 },
		{ 2990, 2990 },
		{ 2990, 2990 },
		{ 2990, 2990 },
		{ 2990, 2990 },
		{ 2990, 2990 },
		{ 2990, 2990 },
		{ 0, 2227 },
		{ 131, 131 },
		{ 131, 131 },
		{ 131, 131 },
		{ 131, 131 },
		{ 131, 131 },
		{ 131, 131 },
		{ 131, 131 },
		{ 131, 131 },
		{ 131, 131 },
		{ 131, 131 },
		{ 2563, 2563 },
		{ 2563, 2563 },
		{ 1801, 1778 },
		{ 2606, 2581 },
		{ 2167, 2165 },
		{ 1986, 1964 },
		{ 1872, 1857 },
		{ 2974, 2973 },
		{ 2612, 2587 },
		{ 2860, 2859 },
		{ 2711, 2697 },
		{ 1283, 1282 },
		{ 2840, 2839 },
		{ 2712, 2698 },
		{ 1711, 1710 },
		{ 1822, 1803 },
		{ 2983, 2982 },
		{ 1272, 1271 },
		{ 2847, 2846 },
		{ 2373, 2346 },
		{ 2622, 2598 },
		{ 0, 2227 },
		{ 3074, 3074 },
		{ 2563, 2563 },
		{ 1215, 1215 },
		{ 1215, 1215 },
		{ 1215, 1215 },
		{ 1215, 1215 },
		{ 1215, 1215 },
		{ 1215, 1215 },
		{ 1215, 1215 },
		{ 1215, 1215 },
		{ 1215, 1215 },
		{ 1215, 1215 },
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
		{ 1314, 1313 },
		{ 3074, 3074 },
		{ 2877, 2877 },
		{ 2877, 2877 },
		{ 2799, 2799 },
		{ 2799, 2799 },
		{ 2547, 2547 },
		{ 2547, 2547 },
		{ 2539, 2513 },
		{ 2542, 2517 },
		{ 2543, 2518 },
		{ 2544, 2519 },
		{ 1307, 1306 },
		{ 2865, 2864 },
		{ 2076, 2060 },
		{ 2427, 2399 },
		{ 1373, 1352 },
		{ 2090, 2074 },
		{ 2881, 2880 },
		{ 2640, 2620 },
		{ 2741, 2740 },
		{ 2884, 2883 },
		{ 1237, 1236 },
		{ 1895, 1881 },
		{ 2562, 2537 },
		{ 2877, 2877 },
		{ 2645, 2626 },
		{ 2799, 2799 },
		{ 1987, 1964 },
		{ 2547, 2547 },
		{ 1325, 1324 },
		{ 2106, 2092 },
		{ 2107, 2093 },
		{ 2903, 2902 },
		{ 2386, 2360 },
		{ 2653, 2634 },
		{ 2388, 2362 },
		{ 2774, 2773 },
		{ 3062, 3062 },
		{ 1309, 1308 },
		{ 2661, 2642 },
		{ 2912, 2911 },
		{ 1765, 1764 },
		{ 1342, 1341 },
		{ 1602, 1601 },
		{ 1731, 1730 },
		{ 2588, 2563 },
		{ 1258, 1257 },
		{ 1604, 1603 },
		{ 2787, 2786 },
		{ 2922, 2921 },
		{ 2861, 2860 },
		{ 2575, 2550 },
		{ 1257, 1256 },
		{ 2169, 2168 },
		{ 2935, 2934 },
		{ 67, 5 },
		{ 3074, 3071 },
		{ 2894, 2893 },
		{ 2679, 2662 },
		{ 2875, 2874 },
		{ 3062, 3062 },
		{ 2680, 2663 },
		{ 1730, 1729 },
		{ 2867, 2866 },
		{ 1260, 1259 },
		{ 2886, 2885 },
		{ 1324, 1323 },
		{ 1733, 1732 },
		{ 2522, 2493 },
		{ 2880, 2879 },
		{ 3076, 3074 },
		{ 2631, 2608 },
		{ 2676, 2659 },
		{ 2160, 2158 },
		{ 3019, 3014 },
		{ 2827, 2825 },
		{ 1811, 1788 },
		{ 2190, 2189 },
		{ 1949, 1946 },
		{ 2474, 2446 },
		{ 1812, 1788 },
		{ 2446, 2420 },
		{ 1374, 1352 },
		{ 1948, 1946 },
		{ 2521, 2492 },
		{ 2889, 2888 },
		{ 1434, 1412 },
		{ 2930, 2929 },
		{ 1793, 1773 },
		{ 2641, 2621 },
		{ 2878, 2877 },
		{ 2828, 2825 },
		{ 2800, 2799 },
		{ 1792, 1773 },
		{ 2572, 2547 },
		{ 1984, 1961 },
		{ 2802, 2801 },
		{ 1246, 1245 },
		{ 2621, 2597 },
		{ 2839, 2838 },
		{ 1983, 1961 },
		{ 2384, 2358 },
		{ 2546, 2521 },
		{ 1719, 1718 },
		{ 3020, 3014 },
		{ 2424, 2396 },
		{ 2320, 2292 },
		{ 174, 5 },
		{ 2175, 2172 },
		{ 1390, 1364 },
		{ 2387, 2361 },
		{ 175, 5 },
		{ 2578, 2553 },
		{ 2174, 2172 },
		{ 1928, 1920 },
		{ 1680, 1679 },
		{ 1338, 1337 },
		{ 1594, 1593 },
		{ 176, 5 },
		{ 1299, 1298 },
		{ 1316, 1315 },
		{ 1939, 1937 },
		{ 1230, 1229 },
		{ 2481, 2452 },
		{ 2589, 2564 },
		{ 2897, 2896 },
		{ 2108, 2094 },
		{ 2899, 2898 },
		{ 2725, 2719 },
		{ 3065, 3062 },
		{ 1738, 1737 },
		{ 2594, 2569 },
		{ 2260, 2232 },
		{ 1637, 1636 },
		{ 173, 5 },
		{ 2488, 2459 },
		{ 1694, 1693 },
		{ 1323, 1322 },
		{ 2736, 2734 },
		{ 1644, 1643 },
		{ 98, 81 },
		{ 2496, 2466 },
		{ 2139, 2126 },
		{ 2143, 2136 },
		{ 1698, 1697 },
		{ 2923, 2922 },
		{ 1985, 1962 },
		{ 1249, 1248 },
		{ 2766, 2765 },
		{ 2503, 2473 },
		{ 1752, 1751 },
		{ 2414, 2387 },
		{ 2509, 2478 },
		{ 2938, 2937 },
		{ 1847, 1828 },
		{ 2940, 2939 },
		{ 2623, 2599 },
		{ 2625, 2601 },
		{ 1245, 1244 },
		{ 1648, 1647 },
		{ 1851, 1832 },
		{ 2157, 2154 },
		{ 1493, 1472 },
		{ 2633, 2610 },
		{ 2790, 2789 },
		{ 2422, 2394 },
		{ 2792, 2791 },
		{ 1756, 1755 },
		{ 1855, 1836 },
		{ 1494, 1473 },
		{ 1858, 1840 },
		{ 2963, 2962 },
		{ 2524, 2495 },
		{ 1288, 1287 },
		{ 2801, 2800 },
		{ 2526, 2497 },
		{ 1265, 1264 },
		{ 2022, 2001 },
		{ 2805, 2804 },
		{ 2023, 2002 },
		{ 2808, 2807 },
		{ 1611, 1610 },
		{ 2812, 2811 },
		{ 1330, 1329 },
		{ 1870, 1854 },
		{ 1618, 1617 },
		{ 2189, 2188 },
		{ 2535, 2506 },
		{ 2995, 2987 },
		{ 2655, 2636 },
		{ 1713, 1712 },
		{ 1295, 1294 },
		{ 2660, 2641 },
		{ 2040, 2019 },
		{ 1226, 1225 },
		{ 2541, 2516 },
		{ 1668, 1667 },
		{ 2831, 2829 },
		{ 3022, 3017 },
		{ 3023, 3019 },
		{ 1794, 1774 },
		{ 1795, 1775 },
		{ 1796, 1775 },
		{ 2668, 2649 },
		{ 2838, 2837 },
		{ 1721, 1720 },
		{ 2670, 2651 },
		{ 1411, 1389 },
		{ 2842, 2841 },
		{ 3037, 3034 },
		{ 2549, 2524 },
		{ 1622, 1621 },
		{ 2551, 2526 },
		{ 3046, 3045 },
		{ 2205, 2204 },
		{ 2049, 2031 },
		{ 2207, 2206 },
		{ 2057, 2039 },
		{ 1412, 1390 },
		{ 2561, 2536 },
		{ 1726, 1725 },
		{ 3058, 3055 },
		{ 1672, 1671 },
		{ 1590, 1589 },
		{ 2862, 2861 },
		{ 1911, 1901 },
		{ 3067, 3064 },
		{ 2064, 2047 },
		{ 1280, 1279 },
		{ 1253, 1252 },
		{ 2570, 2545 },
		{ 1819, 1799 },
		{ 1927, 1919 },
		{ 2876, 2875 },
		{ 2700, 2685 },
		{ 2879, 2878 },
		{ 2448, 2422 },
		{ 1419, 1397 },
		{ 2029, 2008 },
		{ 2001, 1978 },
		{ 1337, 1336 },
		{ 2559, 2534 },
		{ 2318, 2289 },
		{ 2809, 2808 },
		{ 2927, 2926 },
		{ 3064, 3061 },
		{ 3028, 3024 },
		{ 128, 113 },
		{ 2832, 2830 },
		{ 1797, 1775 },
		{ 2651, 2632 },
		{ 1764, 1763 },
		{ 2858, 2857 },
		{ 1849, 1830 },
		{ 2965, 2964 },
		{ 2042, 2021 },
		{ 1839, 1820 },
		{ 1859, 1841 },
		{ 2999, 2998 },
		{ 2893, 2892 },
		{ 2556, 2531 },
		{ 2811, 2810 },
		{ 1231, 1230 },
		{ 608, 550 },
		{ 2091, 2076 },
		{ 2951, 2950 },
		{ 3060, 3057 },
		{ 2981, 2980 },
		{ 3063, 3060 },
		{ 2506, 2476 },
		{ 2817, 2816 },
		{ 1630, 1629 },
		{ 2786, 2785 },
		{ 609, 550 },
		{ 1743, 1742 },
		{ 2934, 2933 },
		{ 3071, 3068 },
		{ 2683, 2666 },
		{ 2863, 2862 },
		{ 1887, 1872 },
		{ 2910, 2909 },
		{ 3078, 3076 },
		{ 1270, 1269 },
		{ 2772, 2771 },
		{ 1340, 1339 },
		{ 1322, 1319 },
		{ 1791, 1787 },
		{ 2548, 2523 },
		{ 1372, 1365 },
		{ 610, 550 },
		{ 2577, 2552 },
		{ 2598, 2573 },
		{ 2626, 2602 },
		{ 1982, 1976 },
		{ 202, 179 },
		{ 2627, 2604 },
		{ 2402, 2376 },
		{ 200, 179 },
		{ 2836, 2835 },
		{ 201, 179 },
		{ 2573, 2548 },
		{ 1625, 1624 },
		{ 2445, 2419 },
		{ 1675, 1674 },
		{ 2489, 2460 },
		{ 1235, 1234 },
		{ 1332, 1331 },
		{ 2077, 2062 },
		{ 199, 179 },
		{ 2078, 2065 },
		{ 2785, 2784 },
		{ 1813, 1789 },
		{ 1997, 1973 },
		{ 2540, 2515 },
		{ 2699, 2684 },
		{ 1701, 1700 },
		{ 1659, 1658 },
		{ 2591, 2566 },
		{ 2933, 2932 },
		{ 2794, 2793 },
		{ 1553, 1537 },
		{ 2000, 1977 },
		{ 2105, 2091 },
		{ 1769, 1768 },
		{ 2505, 2475 },
		{ 1936, 1933 },
		{ 2942, 2941 },
		{ 1682, 1681 },
		{ 3041, 3040 },
		{ 1312, 1311 },
		{ 2654, 2635 },
		{ 1302, 1301 },
		{ 2602, 2577 },
		{ 3048, 3047 },
		{ 2722, 2714 },
		{ 1685, 1684 },
		{ 3051, 3050 },
		{ 1233, 1232 },
		{ 1607, 1606 },
		{ 2191, 2190 },
		{ 2429, 2401 },
		{ 1651, 1650 },
		{ 2611, 2586 },
		{ 1597, 1596 },
		{ 3061, 3058 },
		{ 2613, 2588 },
		{ 2520, 2491 },
		{ 2964, 2963 },
		{ 2739, 2738 },
		{ 2892, 2891 },
		{ 1900, 1887 },
		{ 3068, 3065 },
		{ 2969, 2968 },
		{ 2671, 2652 },
		{ 2144, 2137 },
		{ 2972, 2971 },
		{ 1285, 1284 },
		{ 1908, 1898 },
		{ 1759, 1758 },
		{ 3077, 3075 },
		{ 2201, 2200 },
		{ 2901, 2900 },
		{ 2150, 2146 },
		{ 2203, 2202 },
		{ 668, 607 },
		{ 734, 677 },
		{ 845, 794 },
		{ 416, 376 },
		{ 1616, 25 },
		{ 1321, 19 },
		{ 1223, 11 },
		{ 67, 25 },
		{ 67, 19 },
		{ 67, 11 },
		{ 2763, 45 },
		{ 1642, 27 },
		{ 1293, 17 },
		{ 67, 45 },
		{ 67, 27 },
		{ 67, 17 },
		{ 357, 316 },
		{ 1666, 29 },
		{ 733, 677 },
		{ 417, 376 },
		{ 67, 29 },
		{ 1692, 31 },
		{ 669, 607 },
		{ 380, 340 },
		{ 67, 31 },
		{ 383, 343 },
		{ 389, 349 },
		{ 400, 361 },
		{ 846, 794 },
		{ 406, 367 },
		{ 2011, 1990 },
		{ 415, 375 },
		{ 229, 195 },
		{ 429, 386 },
		{ 1823, 1804 },
		{ 233, 199 },
		{ 450, 401 },
		{ 453, 404 },
		{ 463, 412 },
		{ 464, 413 },
		{ 477, 426 },
		{ 485, 435 },
		{ 518, 459 },
		{ 2033, 2012 },
		{ 2034, 2013 },
		{ 528, 467 },
		{ 530, 469 },
		{ 531, 470 },
		{ 535, 474 },
		{ 547, 488 },
		{ 1843, 1824 },
		{ 1844, 1825 },
		{ 551, 492 },
		{ 555, 496 },
		{ 565, 506 },
		{ 580, 519 },
		{ 2056, 2038 },
		{ 581, 521 },
		{ 586, 526 },
		{ 234, 200 },
		{ 617, 554 },
		{ 630, 567 },
		{ 633, 570 },
		{ 642, 579 },
		{ 1864, 1846 },
		{ 657, 593 },
		{ 2072, 2055 },
		{ 658, 594 },
		{ 667, 606 },
		{ 240, 206 },
		{ 687, 625 },
		{ 260, 222 },
		{ 735, 678 },
		{ 747, 689 },
		{ 1879, 1863 },
		{ 749, 691 },
		{ 751, 693 },
		{ 756, 698 },
		{ 776, 718 },
		{ 785, 727 },
		{ 789, 731 },
		{ 790, 732 },
		{ 820, 766 },
		{ 837, 786 },
		{ 269, 231 },
		{ 875, 826 },
		{ 1614, 25 },
		{ 1319, 19 },
		{ 1221, 11 },
		{ 883, 834 },
		{ 898, 849 },
		{ 934, 888 },
		{ 2761, 45 },
		{ 1640, 27 },
		{ 1291, 17 },
		{ 938, 892 },
		{ 954, 911 },
		{ 969, 930 },
		{ 994, 959 },
		{ 1664, 29 },
		{ 997, 962 },
		{ 1004, 970 },
		{ 1014, 980 },
		{ 1690, 31 },
		{ 1031, 999 },
		{ 1050, 1017 },
		{ 1057, 1026 },
		{ 1059, 1028 },
		{ 1071, 1044 },
		{ 1084, 1058 },
		{ 1089, 1064 },
		{ 1096, 1072 },
		{ 1097, 1073 },
		{ 1111, 1093 },
		{ 1128, 1112 },
		{ 1133, 1118 },
		{ 1140, 1129 },
		{ 1151, 1141 },
		{ 1157, 1147 },
		{ 1192, 1191 },
		{ 276, 238 },
		{ 290, 249 },
		{ 297, 256 },
		{ 303, 261 },
		{ 312, 270 },
		{ 328, 285 },
		{ 331, 288 },
		{ 338, 294 },
		{ 339, 295 },
		{ 344, 300 },
		{ 67, 33 },
		{ 67, 53 },
		{ 422, 380 },
		{ 67, 35 },
		{ 423, 380 },
		{ 421, 380 },
		{ 67, 55 },
		{ 67, 59 },
		{ 67, 23 },
		{ 67, 15 },
		{ 67, 41 },
		{ 67, 51 },
		{ 67, 57 },
		{ 67, 13 },
		{ 67, 47 },
		{ 217, 187 },
		{ 2084, 2070 },
		{ 2086, 2071 },
		{ 2145, 2138 },
		{ 215, 187 },
		{ 3012, 3011 },
		{ 424, 380 },
		{ 800, 742 },
		{ 2085, 2070 },
		{ 2087, 2071 },
		{ 455, 406 },
		{ 457, 406 },
		{ 487, 437 },
		{ 2082, 2068 },
		{ 495, 444 },
		{ 496, 444 },
		{ 1891, 1877 },
		{ 710, 654 },
		{ 711, 655 },
		{ 218, 187 },
		{ 216, 187 },
		{ 458, 406 },
		{ 497, 444 },
		{ 425, 381 },
		{ 572, 513 },
		{ 399, 360 },
		{ 456, 406 },
		{ 797, 739 },
		{ 211, 184 },
		{ 285, 244 },
		{ 324, 281 },
		{ 210, 184 },
		{ 508, 453 },
		{ 1092, 1067 },
		{ 1994, 1970 },
		{ 666, 605 },
		{ 2015, 1994 },
		{ 212, 184 },
		{ 431, 388 },
		{ 614, 551 },
		{ 890, 841 },
		{ 510, 453 },
		{ 1993, 1970 },
		{ 613, 551 },
		{ 350, 306 },
		{ 889, 840 },
		{ 509, 453 },
		{ 502, 448 },
		{ 1101, 1077 },
		{ 499, 446 },
		{ 503, 448 },
		{ 612, 551 },
		{ 611, 551 },
		{ 294, 253 },
		{ 1992, 1970 },
		{ 2016, 1995 },
		{ 224, 190 },
		{ 409, 369 },
		{ 408, 369 },
		{ 885, 836 },
		{ 564, 505 },
		{ 698, 640 },
		{ 801, 743 },
		{ 802, 744 },
		{ 1716, 33 },
		{ 2852, 53 },
		{ 1100, 1077 },
		{ 1749, 35 },
		{ 223, 190 },
		{ 500, 446 },
		{ 2869, 55 },
		{ 2957, 59 },
		{ 1587, 23 },
		{ 1276, 15 },
		{ 2186, 41 },
		{ 2825, 51 },
		{ 2916, 57 },
		{ 1242, 13 },
		{ 2779, 47 },
		{ 221, 188 },
		{ 1826, 1807 },
		{ 263, 225 },
		{ 219, 188 },
		{ 254, 217 },
		{ 577, 518 },
		{ 220, 188 },
		{ 277, 239 },
		{ 515, 457 },
		{ 578, 518 },
		{ 699, 641 },
		{ 1224, 1221 },
		{ 2764, 2761 },
		{ 1148, 1138 },
		{ 3010, 3008 },
		{ 1149, 1139 },
		{ 855, 805 },
		{ 701, 643 },
		{ 1806, 1782 },
		{ 3021, 3016 },
		{ 579, 518 },
		{ 1035, 1003 },
		{ 278, 239 },
		{ 516, 457 },
		{ 781, 723 },
		{ 553, 494 },
		{ 1058, 1027 },
		{ 532, 471 },
		{ 731, 675 },
		{ 732, 676 },
		{ 235, 201 },
		{ 3032, 3029 },
		{ 411, 371 },
		{ 945, 901 },
		{ 949, 906 },
		{ 681, 619 },
		{ 806, 748 },
		{ 1125, 1109 },
		{ 597, 536 },
		{ 2425, 2397 },
		{ 1131, 1115 },
		{ 473, 422 },
		{ 1278, 1276 },
		{ 205, 181 },
		{ 207, 182 },
		{ 544, 485 },
		{ 1110, 1090 },
		{ 334, 290 },
		{ 206, 181 },
		{ 208, 182 },
		{ 1118, 1101 },
		{ 1119, 1103 },
		{ 335, 291 },
		{ 386, 346 },
		{ 1129, 1113 },
		{ 392, 352 },
		{ 333, 290 },
		{ 332, 290 },
		{ 196, 178 },
		{ 543, 485 },
		{ 817, 762 },
		{ 1825, 1806 },
		{ 198, 178 },
		{ 1141, 1130 },
		{ 1143, 1132 },
		{ 818, 763 },
		{ 539, 479 },
		{ 828, 776 },
		{ 1155, 1145 },
		{ 197, 178 },
		{ 1156, 1146 },
		{ 834, 783 },
		{ 1161, 1152 },
		{ 1176, 1167 },
		{ 540, 480 },
		{ 1203, 1202 },
		{ 2013, 1992 },
		{ 292, 251 },
		{ 853, 803 },
		{ 545, 486 },
		{ 3011, 3010 },
		{ 860, 810 },
		{ 870, 821 },
		{ 340, 296 },
		{ 684, 622 },
		{ 343, 299 },
		{ 407, 368 },
		{ 478, 427 },
		{ 559, 500 },
		{ 3025, 3021 },
		{ 899, 850 },
		{ 905, 856 },
		{ 911, 863 },
		{ 916, 868 },
		{ 921, 875 },
		{ 926, 881 },
		{ 933, 887 },
		{ 702, 644 },
		{ 3035, 3032 },
		{ 563, 504 },
		{ 261, 223 },
		{ 712, 656 },
		{ 274, 236 },
		{ 955, 912 },
		{ 966, 927 },
		{ 570, 511 },
		{ 974, 935 },
		{ 975, 936 },
		{ 979, 940 },
		{ 980, 941 },
		{ 986, 947 },
		{ 355, 313 },
		{ 300, 259 },
		{ 359, 318 },
		{ 506, 451 },
		{ 1016, 982 },
		{ 1018, 984 },
		{ 1022, 988 },
		{ 1029, 997 },
		{ 1030, 998 },
		{ 361, 320 },
		{ 512, 455 },
		{ 1043, 1010 },
		{ 599, 538 },
		{ 777, 719 },
		{ 778, 720 },
		{ 603, 543 },
		{ 1068, 1038 },
		{ 784, 726 },
		{ 1083, 1057 },
		{ 607, 549 },
		{ 1085, 1059 },
		{ 786, 728 },
		{ 428, 384 },
		{ 376, 337 },
		{ 795, 737 },
		{ 796, 738 },
		{ 1105, 1084 },
		{ 1479, 1479 },
		{ 1039, 1006 },
		{ 557, 498 },
		{ 384, 344 },
		{ 1055, 1023 },
		{ 640, 577 },
		{ 527, 466 },
		{ 2051, 2033 },
		{ 239, 205 },
		{ 1063, 1032 },
		{ 227, 193 },
		{ 663, 599 },
		{ 568, 509 },
		{ 1583, 1583 },
		{ 358, 317 },
		{ 925, 880 },
		{ 427, 383 },
		{ 574, 515 },
		{ 396, 357 },
		{ 686, 624 },
		{ 536, 475 },
		{ 489, 440 },
		{ 951, 908 },
		{ 1479, 1479 },
		{ 341, 297 },
		{ 804, 746 },
		{ 700, 642 },
		{ 1123, 1107 },
		{ 590, 530 },
		{ 591, 531 },
		{ 542, 484 },
		{ 242, 208 },
		{ 1861, 1843 },
		{ 600, 539 },
		{ 720, 666 },
		{ 989, 951 },
		{ 1583, 1583 },
		{ 1516, 1516 },
		{ 1519, 1519 },
		{ 990, 952 },
		{ 1522, 1522 },
		{ 1525, 1525 },
		{ 1528, 1528 },
		{ 1531, 1531 },
		{ 991, 954 },
		{ 840, 789 },
		{ 841, 790 },
		{ 999, 964 },
		{ 403, 364 },
		{ 605, 547 },
		{ 1015, 981 },
		{ 365, 324 },
		{ 1186, 1180 },
		{ 1549, 1549 },
		{ 319, 276 },
		{ 1019, 985 },
		{ 228, 194 },
		{ 351, 309 },
		{ 880, 831 },
		{ 1558, 1558 },
		{ 1516, 1516 },
		{ 1519, 1519 },
		{ 1389, 1479 },
		{ 1522, 1522 },
		{ 1525, 1525 },
		{ 1528, 1528 },
		{ 1531, 1531 },
		{ 882, 833 },
		{ 624, 561 },
		{ 2102, 2088 },
		{ 2103, 2089 },
		{ 469, 418 },
		{ 1056, 1024 },
		{ 634, 571 },
		{ 894, 845 },
		{ 1389, 1583 },
		{ 1549, 1549 },
		{ 897, 848 },
		{ 638, 575 },
		{ 761, 703 },
		{ 900, 851 },
		{ 1075, 1048 },
		{ 1558, 1558 },
		{ 1078, 1051 },
		{ 1079, 1052 },
		{ 903, 854 },
		{ 767, 709 },
		{ 529, 468 },
		{ 3007, 3003 },
		{ 913, 865 },
		{ 1091, 1066 },
		{ 914, 866 },
		{ 1093, 1068 },
		{ 915, 867 },
		{ 1863, 1845 },
		{ 472, 421 },
		{ 573, 514 },
		{ 214, 186 },
		{ 387, 347 },
		{ 1389, 1516 },
		{ 1389, 1519 },
		{ 320, 277 },
		{ 1389, 1522 },
		{ 1389, 1525 },
		{ 1389, 1528 },
		{ 1389, 1531 },
		{ 483, 432 },
		{ 583, 523 },
		{ 670, 608 },
		{ 585, 525 },
		{ 1878, 1862 },
		{ 349, 305 },
		{ 952, 909 },
		{ 393, 353 },
		{ 799, 741 },
		{ 1389, 1549 },
		{ 958, 918 },
		{ 963, 924 },
		{ 363, 322 },
		{ 967, 928 },
		{ 693, 631 },
		{ 1389, 1558 },
		{ 697, 637 },
		{ 253, 216 },
		{ 976, 937 },
		{ 370, 329 },
		{ 1905, 1893 },
		{ 1906, 1894 },
		{ 807, 749 },
		{ 814, 758 },
		{ 2055, 2037 },
		{ 816, 760 },
		{ 1189, 1187 },
		{ 371, 331 },
		{ 1193, 1192 },
		{ 327, 284 },
		{ 992, 956 },
		{ 819, 764 },
		{ 507, 452 },
		{ 704, 646 },
		{ 705, 648 },
		{ 2069, 2052 },
		{ 1006, 972 },
		{ 707, 651 },
		{ 606, 548 },
		{ 275, 237 },
		{ 356, 314 },
		{ 716, 659 },
		{ 719, 664 },
		{ 296, 255 },
		{ 729, 673 },
		{ 412, 372 },
		{ 520, 461 },
		{ 628, 565 },
		{ 467, 416 },
		{ 632, 569 },
		{ 522, 463 },
		{ 523, 463 },
		{ 524, 464 },
		{ 525, 464 },
		{ 596, 535 },
		{ 280, 240 },
		{ 279, 240 },
		{ 595, 535 },
		{ 204, 180 },
		{ 378, 338 },
		{ 377, 338 },
		{ 650, 586 },
		{ 651, 586 },
		{ 245, 210 },
		{ 491, 442 },
		{ 301, 260 },
		{ 244, 210 },
		{ 248, 213 },
		{ 203, 180 },
		{ 675, 613 },
		{ 249, 213 },
		{ 284, 243 },
		{ 493, 443 },
		{ 302, 260 },
		{ 744, 687 },
		{ 492, 442 },
		{ 676, 614 },
		{ 258, 220 },
		{ 286, 245 },
		{ 250, 213 },
		{ 2017, 1996 },
		{ 283, 243 },
		{ 494, 443 },
		{ 257, 220 },
		{ 745, 687 },
		{ 3029, 3025 },
		{ 468, 417 },
		{ 1195, 1194 },
		{ 567, 508 },
		{ 877, 828 },
		{ 2088, 2072 },
		{ 432, 389 },
		{ 471, 420 },
		{ 1862, 1844 },
		{ 1103, 1082 },
		{ 1104, 1083 },
		{ 256, 219 },
		{ 1106, 1085 },
		{ 271, 233 },
		{ 803, 745 },
		{ 961, 921 },
		{ 1824, 1805 },
		{ 504, 449 },
		{ 1121, 1105 },
		{ 1827, 1808 },
		{ 750, 692 },
		{ 1880, 1864 },
		{ 452, 403 },
		{ 753, 695 },
		{ 1988, 1965 },
		{ 653, 588 },
		{ 757, 699 },
		{ 379, 339 },
		{ 2052, 2034 },
		{ 1138, 1127 },
		{ 268, 230 },
		{ 582, 522 },
		{ 2397, 2371 },
		{ 1893, 1879 },
		{ 1060, 1029 },
		{ 3008, 3006 },
		{ 982, 943 },
		{ 912, 864 },
		{ 706, 650 },
		{ 346, 302 },
		{ 584, 524 },
		{ 620, 557 },
		{ 347, 303 },
		{ 674, 612 },
		{ 2073, 2056 },
		{ 2012, 1991 },
		{ 541, 483 },
		{ 474, 423 },
		{ 858, 808 },
		{ 1087, 1061 },
		{ 917, 869 },
		{ 1805, 1781 },
		{ 1991, 1969 },
		{ 918, 870 },
		{ 1995, 1971 },
		{ 1807, 1783 },
		{ 225, 191 },
		{ 863, 813 },
		{ 1094, 1070 },
		{ 1000, 965 },
		{ 866, 817 },
		{ 867, 818 },
		{ 259, 221 },
		{ 2083, 2069 },
		{ 873, 824 },
		{ 943, 898 },
		{ 760, 702 },
		{ 692, 630 },
		{ 420, 379 },
		{ 695, 633 },
		{ 533, 472 },
		{ 1892, 1878 },
		{ 884, 835 },
		{ 1033, 1001 },
		{ 956, 916 },
		{ 588, 528 },
		{ 571, 512 },
		{ 1046, 1013 },
		{ 2121, 2102 },
		{ 430, 387 },
		{ 1137, 1126 },
		{ 1051, 1018 },
		{ 1052, 1019 },
		{ 1053, 1020 },
		{ 1916, 1905 },
		{ 291, 250 },
		{ 741, 684 },
		{ 822, 770 },
		{ 788, 730 },
		{ 743, 686 },
		{ 461, 409 },
		{ 575, 516 },
		{ 246, 211 },
		{ 1162, 1153 },
		{ 1173, 1164 },
		{ 1174, 1165 },
		{ 561, 502 },
		{ 1179, 1170 },
		{ 983, 944 },
		{ 1187, 1182 },
		{ 1188, 1183 },
		{ 1856, 1837 },
		{ 1076, 1049 },
		{ 848, 796 },
		{ 987, 948 },
		{ 1081, 1054 },
		{ 714, 658 },
		{ 560, 501 },
		{ 930, 884 },
		{ 604, 544 },
		{ 641, 578 },
		{ 414, 374 },
		{ 715, 658 },
		{ 1135, 1124 },
		{ 1136, 1125 },
		{ 1064, 1033 },
		{ 1065, 1035 },
		{ 299, 258 },
		{ 243, 209 },
		{ 1142, 1131 },
		{ 1073, 1046 },
		{ 1147, 1137 },
		{ 824, 772 },
		{ 262, 224 },
		{ 832, 781 },
		{ 891, 842 },
		{ 1080, 1053 },
		{ 385, 345 },
		{ 1158, 1149 },
		{ 1017, 983 },
		{ 616, 553 },
		{ 1171, 1162 },
		{ 1876, 1860 },
		{ 2067, 2050 },
		{ 794, 736 },
		{ 1021, 987 },
		{ 433, 390 },
		{ 843, 792 },
		{ 1180, 1173 },
		{ 1181, 1174 },
		{ 1185, 1179 },
		{ 438, 398 },
		{ 904, 855 },
		{ 1032, 1000 },
		{ 550, 491 },
		{ 1190, 1188 },
		{ 287, 246 },
		{ 1099, 1076 },
		{ 2014, 1993 },
		{ 629, 566 },
		{ 1202, 1201 },
		{ 1102, 1081 },
		{ 511, 454 },
		{ 1044, 1011 },
		{ 1045, 1012 },
		{ 708, 652 },
		{ 1107, 1087 },
		{ 366, 325 },
		{ 677, 615 },
		{ 1112, 1094 },
		{ 1117, 1100 },
		{ 3006, 3002 },
		{ 680, 618 },
		{ 490, 441 },
		{ 871, 822 },
		{ 451, 402 },
		{ 838, 787 },
		{ 576, 517 },
		{ 589, 529 },
		{ 765, 707 },
		{ 348, 304 },
		{ 230, 196 },
		{ 671, 609 },
		{ 946, 902 },
		{ 593, 533 },
		{ 326, 283 },
		{ 426, 382 },
		{ 647, 583 },
		{ 649, 585 },
		{ 436, 394 },
		{ 869, 820 },
		{ 405, 366 },
		{ 602, 542 },
		{ 232, 198 },
		{ 826, 774 },
		{ 752, 694 },
		{ 878, 829 },
		{ 1200, 1199 },
		{ 688, 626 },
		{ 919, 873 },
		{ 2053, 2035 },
		{ 2054, 2036 },
		{ 662, 598 },
		{ 397, 358 },
		{ 970, 931 },
		{ 1034, 1002 },
		{ 1098, 1074 },
		{ 1177, 1168 },
		{ 1178, 1169 },
		{ 971, 932 },
		{ 1037, 1004 },
		{ 689, 627 },
		{ 1184, 1178 },
		{ 342, 298 },
		{ 1038, 1004 },
		{ 823, 771 },
		{ 1036, 1004 },
		{ 526, 465 },
		{ 264, 226 },
		{ 754, 696 },
		{ 879, 830 },
		{ 922, 876 },
		{ 923, 877 },
		{ 1116, 1098 },
		{ 1196, 1195 },
		{ 718, 661 },
		{ 619, 556 },
		{ 798, 740 },
		{ 1120, 1104 },
		{ 354, 312 },
		{ 1122, 1106 },
		{ 252, 215 },
		{ 995, 960 },
		{ 996, 961 },
		{ 462, 411 },
		{ 942, 896 },
		{ 587, 527 },
		{ 1134, 1121 },
		{ 1069, 1040 },
		{ 1002, 968 },
		{ 310, 268 },
		{ 1846, 1827 },
		{ 1005, 971 },
		{ 682, 620 },
		{ 2038, 2017 },
		{ 231, 197 },
		{ 780, 722 },
		{ 813, 757 },
		{ 859, 809 },
		{ 901, 852 },
		{ 685, 623 },
		{ 861, 811 },
		{ 1086, 1060 },
		{ 782, 724 },
		{ 1025, 993 },
		{ 910, 862 },
		{ 315, 273 },
		{ 281, 241 },
		{ 1163, 1154 },
		{ 1168, 1159 },
		{ 592, 532 },
		{ 1172, 1163 },
		{ 558, 499 },
		{ 1377, 1377 },
		{ 825, 773 },
		{ 717, 660 },
		{ 978, 939 },
		{ 514, 456 },
		{ 316, 274 },
		{ 948, 905 },
		{ 1027, 994 },
		{ 513, 456 },
		{ 317, 274 },
		{ 722, 667 },
		{ 721, 667 },
		{ 1026, 994 },
		{ 723, 667 },
		{ 746, 688 },
		{ 876, 827 },
		{ 390, 350 },
		{ 251, 214 },
		{ 480, 429 },
		{ 482, 431 },
		{ 654, 590 },
		{ 960, 920 },
		{ 1124, 1108 },
		{ 1377, 1377 },
		{ 655, 591 },
		{ 1126, 1110 },
		{ 304, 262 },
		{ 965, 926 },
		{ 1049, 1016 },
		{ 755, 697 },
		{ 703, 645 },
		{ 549, 490 },
		{ 484, 433 },
		{ 1054, 1022 },
		{ 394, 354 },
		{ 486, 436 },
		{ 615, 552 },
		{ 770, 712 },
		{ 771, 713 },
		{ 1877, 1861 },
		{ 1145, 1135 },
		{ 381, 341 },
		{ 1061, 1030 },
		{ 902, 853 },
		{ 981, 942 },
		{ 1153, 1143 },
		{ 833, 782 },
		{ 282, 242 },
		{ 985, 946 },
		{ 1070, 1043 },
		{ 835, 784 },
		{ 618, 555 },
		{ 466, 415 },
		{ 1164, 1155 },
		{ 1165, 1156 },
		{ 2035, 2014 },
		{ 2036, 2015 },
		{ 398, 359 },
		{ 1170, 1161 },
		{ 437, 396 },
		{ 626, 563 },
		{ 236, 202 },
		{ 1389, 1377 },
		{ 847, 795 },
		{ 1082, 1056 },
		{ 498, 445 },
		{ 353, 311 },
		{ 501, 447 },
		{ 566, 507 },
		{ 1001, 966 },
		{ 1183, 1176 },
		{ 792, 734 },
		{ 374, 334 },
		{ 636, 573 },
		{ 2959, 2957 },
		{ 862, 812 },
		{ 1012, 978 },
		{ 1013, 979 },
		{ 360, 319 },
		{ 864, 814 },
		{ 737, 680 },
		{ 935, 889 },
		{ 936, 890 },
		{ 740, 683 },
		{ 1020, 986 },
		{ 314, 272 },
		{ 742, 685 },
		{ 2068, 2051 },
		{ 944, 899 },
		{ 2070, 2053 },
		{ 2071, 2054 },
		{ 598, 537 },
		{ 475, 424 },
		{ 844, 793 },
		{ 395, 355 },
		{ 1041, 1008 },
		{ 1042, 1009 },
		{ 2094, 2079 },
		{ 1693, 1690 },
		{ 1294, 1291 },
		{ 1127, 1111 },
		{ 1589, 1587 },
		{ 977, 938 },
		{ 1072, 1045 },
		{ 679, 617 },
		{ 1842, 1823 },
		{ 1919, 1910 },
		{ 2032, 2011 },
		{ 1643, 1640 },
		{ 552, 493 },
		{ 2136, 2123 },
		{ 691, 629 },
		{ 791, 733 },
		{ 1667, 1664 },
		{ 488, 438 },
		{ 325, 282 },
		{ 892, 843 },
		{ 1751, 1749 },
		{ 1617, 1614 },
		{ 1901, 1888 },
		{ 927, 882 },
		{ 323, 280 },
		{ 1139, 1128 },
		{ 713, 657 },
		{ 928, 882 },
		{ 672, 610 },
		{ 1114, 1096 },
		{ 1115, 1097 },
		{ 683, 621 },
		{ 1182, 1175 },
		{ 470, 419 },
		{ 895, 846 },
		{ 373, 333 },
		{ 805, 747 },
		{ 920, 874 },
		{ 1095, 1071 },
		{ 419, 378 },
		{ 827, 775 },
		{ 238, 204 },
		{ 812, 756 },
		{ 1077, 1050 },
		{ 413, 373 },
		{ 1003, 969 },
		{ 725, 669 },
		{ 815, 759 },
		{ 759, 701 },
		{ 364, 323 },
		{ 1108, 1088 },
		{ 1109, 1089 },
		{ 736, 679 },
		{ 865, 815 },
		{ 305, 263 },
		{ 404, 365 },
		{ 635, 572 },
		{ 2096, 2082 },
		{ 2098, 2084 },
		{ 2099, 2085 },
		{ 2100, 2086 },
		{ 2101, 2087 },
		{ 382, 342 },
		{ 1996, 1972 },
		{ 1845, 1826 },
		{ 950, 907 },
		{ 637, 574 },
		{ 872, 823 },
		{ 953, 910 },
		{ 690, 628 },
		{ 874, 825 },
		{ 435, 392 },
		{ 639, 576 },
		{ 1048, 1015 },
		{ 959, 919 },
		{ 476, 425 },
		{ 307, 265 },
		{ 962, 922 },
		{ 1860, 1842 },
		{ 810, 752 },
		{ 964, 925 },
		{ 345, 301 },
		{ 646, 582 },
		{ 521, 462 },
		{ 648, 584 },
		{ 237, 203 },
		{ 887, 838 },
		{ 972, 933 },
		{ 888, 839 },
		{ 439, 399 },
		{ 410, 370 },
		{ 295, 254 },
		{ 330, 287 },
		{ 893, 844 },
		{ 266, 228 },
		{ 764, 706 },
		{ 367, 326 },
		{ 454, 405 },
		{ 768, 710 },
		{ 769, 711 },
		{ 2037, 2016 },
		{ 709, 653 },
		{ 829, 778 },
		{ 391, 351 },
		{ 772, 714 },
		{ 775, 717 },
		{ 1175, 1166 },
		{ 1894, 1880 },
		{ 993, 957 },
		{ 909, 860 },
		{ 459, 407 },
		{ 836, 785 },
		{ 267, 229 },
		{ 2050, 2032 },
		{ 1903, 1891 },
		{ 1088, 1062 },
		{ 209, 183 },
		{ 779, 721 },
		{ 1808, 1784 },
		{ 336, 292 },
		{ 842, 791 },
		{ 193, 174 },
		{ 375, 335 },
		{ 270, 232 },
		{ 321, 278 },
		{ 1007, 973 },
		{ 1191, 1189 },
		{ 1011, 977 },
		{ 621, 558 },
		{ 1194, 1193 },
		{ 849, 798 },
		{ 851, 801 },
		{ 1199, 1198 },
		{ 213, 185 },
		{ 1201, 1200 },
		{ 854, 804 },
		{ 546, 487 },
		{ 857, 807 },
		{ 727, 671 },
		{ 505, 450 },
		{ 548, 489 },
		{ 293, 252 },
		{ 1023, 989 },
		{ 937, 891 },
		{ 631, 568 },
		{ 1028, 995 },
		{ 402, 363 },
		{ 2089, 2073 },
		{ 1943, 1943 },
		{ 1943, 1943 },
		{ 1550, 1550 },
		{ 1550, 1550 },
		{ 1517, 1517 },
		{ 1517, 1517 },
		{ 2163, 2163 },
		{ 2163, 2163 },
		{ 1912, 1912 },
		{ 1912, 1912 },
		{ 1914, 1914 },
		{ 1914, 1914 },
		{ 1526, 1526 },
		{ 1526, 1526 },
		{ 2109, 2109 },
		{ 2109, 2109 },
		{ 2111, 2111 },
		{ 2111, 2111 },
		{ 2113, 2113 },
		{ 2113, 2113 },
		{ 2115, 2115 },
		{ 2115, 2115 },
		{ 311, 269 },
		{ 1943, 1943 },
		{ 1066, 1036 },
		{ 1550, 1550 },
		{ 1067, 1037 },
		{ 1517, 1517 },
		{ 678, 616 },
		{ 2163, 2163 },
		{ 766, 708 },
		{ 1912, 1912 },
		{ 748, 690 },
		{ 1914, 1914 },
		{ 562, 503 },
		{ 1526, 1526 },
		{ 973, 934 },
		{ 2109, 2109 },
		{ 645, 581 },
		{ 2111, 2111 },
		{ 2188, 2186 },
		{ 2113, 2113 },
		{ 665, 604 },
		{ 2115, 2115 },
		{ 2117, 2117 },
		{ 2117, 2117 },
		{ 2119, 2119 },
		{ 2119, 2119 },
		{ 1889, 1889 },
		{ 1889, 1889 },
		{ 643, 580 },
		{ 644, 580 },
		{ 1944, 1943 },
		{ 906, 857 },
		{ 1551, 1550 },
		{ 907, 858 },
		{ 1518, 1517 },
		{ 247, 212 },
		{ 2164, 2163 },
		{ 957, 917 },
		{ 1913, 1912 },
		{ 306, 264 },
		{ 1915, 1914 },
		{ 460, 408 },
		{ 1527, 1526 },
		{ 554, 495 },
		{ 2110, 2109 },
		{ 2117, 2117 },
		{ 2112, 2111 },
		{ 2119, 2119 },
		{ 2114, 2113 },
		{ 1889, 1889 },
		{ 2116, 2115 },
		{ 534, 473 },
		{ 1924, 1924 },
		{ 1924, 1924 },
		{ 2134, 2134 },
		{ 2134, 2134 },
		{ 1584, 1584 },
		{ 1584, 1584 },
		{ 1929, 1929 },
		{ 1929, 1929 },
		{ 2080, 2080 },
		{ 2080, 2080 },
		{ 2140, 2140 },
		{ 2140, 2140 },
		{ 1529, 1529 },
		{ 1529, 1529 },
		{ 1520, 1520 },
		{ 1520, 1520 },
		{ 1532, 1532 },
		{ 1532, 1532 },
		{ 1480, 1480 },
		{ 1480, 1480 },
		{ 1559, 1559 },
		{ 1559, 1559 },
		{ 2118, 2117 },
		{ 1924, 1924 },
		{ 2120, 2119 },
		{ 2134, 2134 },
		{ 1890, 1889 },
		{ 1584, 1584 },
		{ 289, 248 },
		{ 1929, 1929 },
		{ 272, 234 },
		{ 2080, 2080 },
		{ 537, 477 },
		{ 2140, 2140 },
		{ 1062, 1031 },
		{ 1529, 1529 },
		{ 401, 362 },
		{ 1520, 1520 },
		{ 839, 788 },
		{ 1532, 1532 },
		{ 808, 750 },
		{ 1480, 1480 },
		{ 1649, 1648 },
		{ 1559, 1559 },
		{ 1523, 1523 },
		{ 1523, 1523 },
		{ 1144, 1133 },
		{ 1040, 1007 },
		{ 1146, 1136 },
		{ 2122, 2103 },
		{ 1757, 1756 },
		{ 1090, 1065 },
		{ 1925, 1924 },
		{ 809, 751 },
		{ 2135, 2134 },
		{ 886, 837 },
		{ 1585, 1584 },
		{ 1150, 1140 },
		{ 1930, 1929 },
		{ 195, 176 },
		{ 2081, 2080 },
		{ 3039, 3037 },
		{ 2141, 2140 },
		{ 1152, 1142 },
		{ 1530, 1529 },
		{ 793, 735 },
		{ 1521, 1520 },
		{ 1523, 1523 },
		{ 1533, 1532 },
		{ 337, 293 },
		{ 1481, 1480 },
		{ 1885, 1870 },
		{ 1560, 1559 },
		{ 939, 893 },
		{ 1047, 1014 },
		{ 941, 895 },
		{ 1160, 1151 },
		{ 1198, 1197 },
		{ 728, 672 },
		{ 758, 700 },
		{ 1673, 1672 },
		{ 1623, 1622 },
		{ 1024, 991 },
		{ 369, 328 },
		{ 1130, 1114 },
		{ 1167, 1158 },
		{ 730, 674 },
		{ 1132, 1117 },
		{ 783, 725 },
		{ 947, 904 },
		{ 881, 832 },
		{ 856, 806 },
		{ 1008, 974 },
		{ 1009, 975 },
		{ 3009, 3007 },
		{ 988, 949 },
		{ 1524, 1523 },
		{ 1300, 1299 },
		{ 1511, 1493 },
		{ 1917, 1906 },
		{ 1595, 1594 },
		{ 929, 883 },
		{ 1865, 1848 },
		{ 1113, 1095 },
		{ 830, 779 },
		{ 2058, 2041 },
		{ 1699, 1698 },
		{ 1904, 1892 },
		{ 1244, 1242 },
		{ 1166, 1157 },
		{ 222, 189 },
		{ 1990, 1968 },
		{ 2097, 2083 },
		{ 2002, 1979 },
		{ 481, 430 },
		{ 298, 257 },
		{ 601, 541 },
		{ 1804, 1780 },
		{ 2133, 2121 },
		{ 1923, 1916 },
		{ 2854, 2852 },
		{ 2600, 2600 },
		{ 2600, 2600 },
		{ 2715, 2715 },
		{ 2715, 2715 },
		{ 1688, 1688 },
		{ 1688, 1688 },
		{ 1274, 1274 },
		{ 1274, 1274 },
		{ 2507, 2507 },
		{ 2507, 2507 },
		{ 2403, 2403 },
		{ 2403, 2403 },
		{ 2578, 2578 },
		{ 2578, 2578 },
		{ 2725, 2725 },
		{ 2725, 2725 },
		{ 2726, 2726 },
		{ 2726, 2726 },
		{ 2727, 2727 },
		{ 2727, 2727 },
		{ 2850, 2850 },
		{ 2850, 2850 },
		{ 556, 497 },
		{ 2600, 2600 },
		{ 850, 800 },
		{ 2715, 2715 },
		{ 908, 859 },
		{ 1688, 1688 },
		{ 696, 636 },
		{ 1274, 1274 },
		{ 852, 802 },
		{ 2507, 2507 },
		{ 318, 275 },
		{ 2403, 2403 },
		{ 656, 592 },
		{ 2578, 2578 },
		{ 265, 227 },
		{ 2725, 2725 },
		{ 622, 559 },
		{ 2726, 2726 },
		{ 659, 595 },
		{ 2727, 2727 },
		{ 660, 596 },
		{ 2850, 2850 },
		{ 661, 597 },
		{ 2638, 2638 },
		{ 2638, 2638 },
		{ 2639, 2639 },
		{ 2639, 2639 },
		{ 2624, 2600 },
		{ 623, 560 },
		{ 2723, 2715 },
		{ 418, 377 },
		{ 1689, 1688 },
		{ 984, 945 },
		{ 1275, 1274 },
		{ 664, 603 },
		{ 2508, 2507 },
		{ 625, 562 },
		{ 2404, 2403 },
		{ 362, 321 },
		{ 2603, 2578 },
		{ 627, 564 },
		{ 2730, 2725 },
		{ 2918, 2916 },
		{ 2731, 2726 },
		{ 924, 879 },
		{ 2732, 2727 },
		{ 2638, 2638 },
		{ 2851, 2850 },
		{ 2639, 2639 },
		{ 1197, 1196 },
		{ 1317, 1317 },
		{ 1317, 1317 },
		{ 2453, 2453 },
		{ 2453, 2453 },
		{ 2735, 2735 },
		{ 2735, 2735 },
		{ 2914, 2914 },
		{ 2914, 2914 },
		{ 2676, 2676 },
		{ 2676, 2676 },
		{ 2350, 2350 },
		{ 2350, 2350 },
		{ 2209, 2209 },
		{ 2209, 2209 },
		{ 2456, 2456 },
		{ 2456, 2456 },
		{ 1612, 1612 },
		{ 1612, 1612 },
		{ 1289, 1289 },
		{ 1289, 1289 },
		{ 1638, 1638 },
		{ 1638, 1638 },
		{ 2657, 2638 },
		{ 1317, 1317 },
		{ 2658, 2639 },
		{ 2453, 2453 },
		{ 308, 266 },
		{ 2735, 2735 },
		{ 811, 755 },
		{ 2914, 2914 },
		{ 868, 819 },
		{ 2676, 2676 },
		{ 309, 267 },
		{ 2350, 2350 },
		{ 762, 704 },
		{ 2209, 2209 },
		{ 931, 885 },
		{ 2456, 2456 },
		{ 932, 886 },
		{ 1612, 1612 },
		{ 763, 705 },
		{ 1289, 1289 },
		{ 998, 963 },
		{ 1638, 1638 },
		{ 322, 279 },
		{ 2744, 2744 },
		{ 2744, 2744 },
		{ 2686, 2686 },
		{ 2686, 2686 },
		{ 1318, 1317 },
		{ 479, 428 },
		{ 2482, 2453 },
		{ 2155, 2152 },
		{ 2737, 2735 },
		{ 673, 611 },
		{ 2915, 2914 },
		{ 594, 534 },
		{ 2691, 2676 },
		{ 538, 478 },
		{ 2351, 2350 },
		{ 226, 192 },
		{ 2210, 2209 },
		{ 1074, 1047 },
		{ 2485, 2456 },
		{ 940, 894 },
		{ 1613, 1612 },
		{ 821, 767 },
		{ 1290, 1289 },
		{ 2744, 2744 },
		{ 1639, 1638 },
		{ 2686, 2686 },
		{ 352, 310 },
		{ 2867, 2867 },
		{ 2867, 2867 },
		{ 2648, 2648 },
		{ 2648, 2648 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2589, 2589 },
		{ 2589, 2589 },
		{ 2690, 2690 },
		{ 2690, 2690 },
		{ 2692, 2692 },
		{ 2692, 2692 },
		{ 2693, 2693 },
		{ 2693, 2693 },
		{ 2821, 2821 },
		{ 2821, 2821 },
		{ 2694, 2694 },
		{ 2694, 2694 },
		{ 2695, 2695 },
		{ 2695, 2695 },
		{ 1714, 1714 },
		{ 1714, 1714 },
		{ 2745, 2744 },
		{ 2867, 2867 },
		{ 2701, 2686 },
		{ 2648, 2648 },
		{ 368, 327 },
		{ 2565, 2565 },
		{ 569, 510 },
		{ 2589, 2589 },
		{ 1010, 976 },
		{ 2690, 2690 },
		{ 773, 715 },
		{ 2692, 2692 },
		{ 774, 716 },
		{ 2693, 2693 },
		{ 724, 668 },
		{ 2821, 2821 },
		{ 1718, 1716 },
		{ 2694, 2694 },
		{ 241, 207 },
		{ 2695, 2695 },
		{ 2781, 2779 },
		{ 1714, 1714 },
		{ 726, 670 },
		{ 1770, 1770 },
		{ 1770, 1770 },
		{ 2593, 2593 },
		{ 2593, 2593 },
		{ 2868, 2867 },
		{ 1154, 1144 },
		{ 2667, 2648 },
		{ 388, 348 },
		{ 2590, 2565 },
		{ 831, 780 },
		{ 2614, 2589 },
		{ 2871, 2869 },
		{ 2705, 2690 },
		{ 517, 458 },
		{ 2706, 2692 },
		{ 288, 247 },
		{ 2707, 2693 },
		{ 1159, 1150 },
		{ 2822, 2821 },
		{ 519, 460 },
		{ 2708, 2694 },
		{ 313, 271 },
		{ 2709, 2695 },
		{ 1770, 1770 },
		{ 1715, 1714 },
		{ 2593, 2593 },
		{ 465, 414 },
		{ 2594, 2594 },
		{ 2594, 2594 },
		{ 2480, 2480 },
		{ 2480, 2480 },
		{ 2656, 2656 },
		{ 2656, 2656 },
		{ 2777, 2777 },
		{ 2777, 2777 },
		{ 2703, 2703 },
		{ 2703, 2703 },
		{ 1747, 1747 },
		{ 1747, 1747 },
		{ 2710, 2710 },
		{ 2710, 2710 },
		{ 1240, 1240 },
		{ 1240, 1240 },
		{ 2484, 2484 },
		{ 2484, 2484 },
		{ 1662, 1662 },
		{ 1662, 1662 },
		{ 2955, 2955 },
		{ 2955, 2955 },
		{ 1771, 1770 },
		{ 2594, 2594 },
		{ 2617, 2593 },
		{ 2480, 2480 },
		{ 372, 332 },
		{ 2656, 2656 },
		{ 434, 391 },
		{ 2777, 2777 },
		{ 896, 847 },
		{ 2703, 2703 },
		{ 273, 235 },
		{ 1747, 1747 },
		{ 787, 729 },
		{ 2710, 2710 },
		{ 194, 175 },
		{ 1240, 1240 },
		{ 1169, 1160 },
		{ 2484, 2484 },
		{ 738, 681 },
		{ 1662, 1662 },
		{ 739, 682 },
		{ 2955, 2955 },
		{ 329, 286 },
		{ 652, 587 },
		{ 255, 218 },
		{ 968, 929 },
		{ 694, 632 },
		{ 2618, 2594 },
		{ 3070, 3070 },
		{ 2511, 2480 },
		{ 3078, 3078 },
		{ 2675, 2656 },
		{ 3082, 3082 },
		{ 2778, 2777 },
		{ 1586, 1585 },
		{ 2716, 2703 },
		{ 2095, 2081 },
		{ 1748, 1747 },
		{ 1561, 1551 },
		{ 2718, 2710 },
		{ 1945, 1944 },
		{ 1241, 1240 },
		{ 1931, 1925 },
		{ 2514, 2484 },
		{ 1567, 1560 },
		{ 1663, 1662 },
		{ 1543, 1521 },
		{ 2956, 2955 },
		{ 2166, 2164 },
		{ 2142, 2135 },
		{ 1499, 1481 },
		{ 3070, 3070 },
		{ 1934, 1930 },
		{ 3078, 3078 },
		{ 1921, 1913 },
		{ 3082, 3082 },
		{ 1547, 1533 },
		{ 2127, 2110 },
		{ 2147, 2141 },
		{ 1922, 1915 },
		{ 2128, 2112 },
		{ 1545, 1527 },
		{ 2129, 2114 },
		{ 1542, 1518 },
		{ 2130, 2116 },
		{ 1544, 1524 },
		{ 2131, 2118 },
		{ 1546, 1530 },
		{ 2132, 2120 },
		{ 1902, 1890 },
		{ 2986, 2985 },
		{ 1679, 1678 },
		{ 1306, 1305 },
		{ 1577, 1573 },
		{ 3044, 3043 },
		{ 3045, 3044 },
		{ 1654, 1653 },
		{ 3073, 3070 },
		{ 1655, 1654 },
		{ 3080, 3078 },
		{ 1305, 1304 },
		{ 3083, 3082 },
		{ 1573, 1568 },
		{ 1574, 1569 },
		{ 1349, 1348 },
		{ 1704, 1703 },
		{ 1705, 1704 },
		{ 1600, 1599 },
		{ 1601, 1600 },
		{ 1628, 1627 },
		{ 1762, 1761 },
		{ 1763, 1762 },
		{ 1629, 1628 },
		{ 1678, 1677 },
		{ 2751, 2751 },
		{ 2748, 2751 },
		{ 163, 163 },
		{ 160, 163 },
		{ 1896, 1884 },
		{ 1897, 1884 },
		{ 1874, 1859 },
		{ 1875, 1859 },
		{ 1950, 1950 },
		{ 2176, 2176 },
		{ 2181, 2177 },
		{ 1955, 1951 },
		{ 2750, 2746 },
		{ 162, 158 },
		{ 168, 164 },
		{ 2180, 2177 },
		{ 1954, 1951 },
		{ 2749, 2746 },
		{ 161, 158 },
		{ 167, 164 },
		{ 2757, 2754 },
		{ 2756, 2752 },
		{ 2234, 2211 },
		{ 2751, 2751 },
		{ 88, 70 },
		{ 163, 163 },
		{ 2755, 2752 },
		{ 2233, 2211 },
		{ 2759, 2758 },
		{ 87, 70 },
		{ 1958, 1957 },
		{ 1950, 1950 },
		{ 2176, 2176 },
		{ 2182, 2179 },
		{ 2184, 2183 },
		{ 2027, 2006 },
		{ 169, 166 },
		{ 171, 170 },
		{ 2752, 2751 },
		{ 119, 103 },
		{ 164, 163 },
		{ 1834, 1815 },
		{ 1956, 1953 },
		{ 2290, 2261 },
		{ 1909, 1899 },
		{ 2171, 2170 },
		{ 1951, 1950 },
		{ 2177, 2176 },
		{ 2758, 2756 },
		{ 166, 162 },
		{ 2261, 2234 },
		{ 2183, 2181 },
		{ 1953, 1949 },
		{ 103, 88 },
		{ 1815, 1793 },
		{ 170, 168 },
		{ 2754, 2750 },
		{ 2179, 2175 },
		{ 1957, 1955 },
		{ 2006, 1984 },
		{ 0, 2538 },
		{ 0, 2976 },
		{ 0, 2575 },
		{ 0, 2886 },
		{ 0, 2805 },
		{ 2617, 2617 },
		{ 2617, 2617 },
		{ 0, 2889 },
		{ 0, 2665 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2731, 2731 },
		{ 2731, 2731 },
		{ 2732, 2732 },
		{ 2732, 2732 },
		{ 0, 2894 },
		{ 2618, 2618 },
		{ 2618, 2618 },
		{ 0, 2812 },
		{ 2667, 2667 },
		{ 2667, 2667 },
		{ 0, 1281 },
		{ 0, 1738 },
		{ 2737, 2737 },
		{ 2737, 2737 },
		{ 0, 1249 },
		{ 2508, 2508 },
		{ 2508, 2508 },
		{ 2617, 2617 },
		{ 0, 2198 },
		{ 2482, 2482 },
		{ 2482, 2482 },
		{ 2730, 2730 },
		{ 0, 2905 },
		{ 2731, 2731 },
		{ 0, 2674 },
		{ 2732, 2732 },
		{ 2675, 2675 },
		{ 2675, 2675 },
		{ 2618, 2618 },
		{ 0, 2582 },
		{ 0, 2677 },
		{ 2667, 2667 },
		{ 2745, 2745 },
		{ 2745, 2745 },
		{ 0, 2679 },
		{ 2737, 2737 },
		{ 0, 2680 },
		{ 0, 2628 },
		{ 2508, 2508 },
		{ 0, 2413 },
		{ 2514, 2514 },
		{ 2514, 2514 },
		{ 2482, 2482 },
		{ 0, 2833 },
		{ 0, 2919 },
		{ 0, 2631 },
		{ 2485, 2485 },
		{ 2485, 2485 },
		{ 0, 2923 },
		{ 2675, 2675 },
		{ 2759, 2759 },
		{ 2760, 2759 },
		{ 0, 2549 },
		{ 0, 2414 },
		{ 0, 2927 },
		{ 2745, 2745 },
		{ 0, 2551 },
		{ 2590, 2590 },
		{ 2590, 2590 },
		{ 0, 2930 },
		{ 0, 2842 },
		{ 0, 2415 },
		{ 0, 2767 },
		{ 2514, 2514 },
		{ 2691, 2691 },
		{ 2691, 2691 },
		{ 0, 2935 },
		{ 0, 2461 },
		{ 0, 1722 },
		{ 2485, 2485 },
		{ 2184, 2184 },
		{ 2185, 2184 },
		{ 0, 1227 },
		{ 2759, 2759 },
		{ 0, 2522 },
		{ 0, 2833 },
		{ 0, 2559 },
		{ 0, 1733 },
		{ 0, 1265 },
		{ 0, 2946 },
		{ 2590, 2590 },
		{ 2701, 2701 },
		{ 2701, 2701 },
		{ 0, 1260 },
		{ 0, 2858 },
		{ 0, 2782 },
		{ 0, 2424 },
		{ 2691, 2691 },
		{ 2603, 2603 },
		{ 2603, 2603 },
		{ 2707, 2707 },
		{ 2707, 2707 },
		{ 0, 2529 },
		{ 2184, 2184 },
		{ 0, 2424 },
		{ 0, 2787 },
		{ 0, 2384 },
		{ 0, 2472 },
		{ 171, 171 },
		{ 172, 171 },
		{ 2351, 2351 },
		{ 2351, 2351 },
		{ 2716, 2716 },
		{ 2716, 2716 },
		{ 2701, 2701 },
		{ 0, 2872 },
		{ 0, 2717 },
		{ 2718, 2718 },
		{ 2718, 2718 },
		{ 0, 2965 },
		{ 0, 2876 },
		{ 2603, 2603 },
		{ 0, 2535 },
		{ 2707, 2707 },
		{ 1958, 1958 },
		{ 1959, 1958 },
		{ 2658, 2658 },
		{ 2658, 2658 },
		{ 0, 2798 },
		{ 0, 2451 },
		{ 0, 2320 },
		{ 171, 171 },
		{ 1206, 1206 },
		{ 2351, 2351 },
		{ 1348, 1347 },
		{ 2716, 2716 },
		{ 2723, 2723 },
		{ 2723, 2723 },
		{ 1853, 1834 },
		{ 165, 167 },
		{ 2718, 2718 },
		{ 2614, 2614 },
		{ 2614, 2614 },
		{ 2753, 2755 },
		{ 0, 1792 },
		{ 2178, 2174 },
		{ 0, 1983 },
		{ 1958, 1958 },
		{ 0, 0 },
		{ 2658, 2658 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1206, 1206 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2723, 2723 },
		{ 0, 0 },
		{ 0, 2320 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2614, 2614 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -68, 16, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 44, 572, 0 },
		{ -177, 2804, 0 },
		{ 5, 0, 0 },
		{ -1205, 1017, -31 },
		{ 7, 0, -31 },
		{ -1209, 1828, -33 },
		{ 9, 0, -33 },
		{ -1222, 3122, 143 },
		{ 11, 0, 143 },
		{ -1243, 3256, 151 },
		{ 13, 0, 151 },
		{ -1277, 3252, 0 },
		{ 15, 0, 0 },
		{ -1292, 3128, 139 },
		{ 17, 0, 139 },
		{ -1320, 3121, 22 },
		{ 19, 0, 22 },
		{ -1362, 230, 0 },
		{ 21, 0, 0 },
		{ -1588, 3251, 0 },
		{ 23, 0, 0 },
		{ -1615, 3120, 0 },
		{ 25, 0, 0 },
		{ -1641, 3127, 0 },
		{ 27, 0, 0 },
		{ -1665, 3133, 0 },
		{ 29, 0, 0 },
		{ -1691, 3137, 0 },
		{ 31, 0, 0 },
		{ -1717, 3243, 154 },
		{ 33, 0, 154 },
		{ -1750, 3246, 258 },
		{ 35, 0, 258 },
		{ 38, 127, 0 },
		{ -1786, 344, 0 },
		{ 40, 16, 0 },
		{ -1975, 116, 0 },
		{ -2187, 3253, 0 },
		{ 41, 0, 0 },
		{ 44, 14, 0 },
		{ -86, 458, 0 },
		{ -2762, 3126, 147 },
		{ 45, 0, 147 },
		{ -2780, 3257, 166 },
		{ 47, 0, 166 },
		{ 2824, 1438, 0 },
		{ 49, 0, 0 },
		{ -2826, 3254, 264 },
		{ 51, 0, 264 },
		{ -2853, 3244, 169 },
		{ 53, 0, 169 },
		{ -2870, 3249, 163 },
		{ 55, 0, 163 },
		{ -2917, 3255, 157 },
		{ 57, 0, 157 },
		{ -2958, 3250, 162 },
		{ 59, 0, 162 },
		{ -86, 1, 0 },
		{ 61, 0, 0 },
		{ -3005, 1788, 0 },
		{ 63, 0, 0 },
		{ -3015, 1697, 42 },
		{ 65, 0, 42 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 416 },
		{ 2758, 4686, 423 },
		{ 0, 0, 236 },
		{ 0, 0, 238 },
		{ 157, 1283, 255 },
		{ 157, 1398, 255 },
		{ 157, 1297, 255 },
		{ 157, 1304, 255 },
		{ 157, 1304, 255 },
		{ 157, 1308, 255 },
		{ 157, 1313, 255 },
		{ 157, 1307, 255 },
		{ 3064, 2801, 423 },
		{ 157, 1318, 255 },
		{ 3064, 1676, 254 },
		{ 102, 2594, 423 },
		{ 157, 0, 255 },
		{ 0, 0, 423 },
		{ -87, 10, 232 },
		{ -88, 4715, 0 },
		{ 157, 686, 255 },
		{ 157, 720, 255 },
		{ 157, 689, 255 },
		{ 157, 703, 255 },
		{ 157, 711, 255 },
		{ 157, 711, 255 },
		{ 157, 718, 255 },
		{ 157, 733, 255 },
		{ 157, 727, 255 },
		{ 3033, 2286, 0 },
		{ 157, 753, 255 },
		{ 3064, 1764, 251 },
		{ 117, 1450, 0 },
		{ 3064, 1731, 252 },
		{ 2758, 4696, 0 },
		{ 157, 762, 255 },
		{ 157, 760, 255 },
		{ 157, 761, 255 },
		{ 157, 757, 255 },
		{ 157, 0, 243 },
		{ 157, 776, 255 },
		{ 157, 778, 255 },
		{ 157, 788, 255 },
		{ 157, 793, 255 },
		{ 3061, 2909, 0 },
		{ 157, 800, 255 },
		{ 131, 1484, 0 },
		{ 117, 0, 0 },
		{ 2990, 2636, 253 },
		{ 133, 1437, 0 },
		{ 0, 0, 234 },
		{ 157, 804, 239 },
		{ 157, 807, 255 },
		{ 157, 827, 255 },
		{ 157, 832, 255 },
		{ 157, 830, 255 },
		{ 157, 823, 255 },
		{ 157, 0, 246 },
		{ 157, 824, 255 },
		{ 0, 0, 248 },
		{ 157, 830, 255 },
		{ 131, 0, 0 },
		{ 2990, 2659, 251 },
		{ 133, 0, 0 },
		{ 2990, 2703, 252 },
		{ 157, 845, 255 },
		{ 157, 842, 255 },
		{ 157, 843, 255 },
		{ 157, 869, 255 },
		{ 157, 980, 255 },
		{ 157, 0, 245 },
		{ 157, 1052, 255 },
		{ 157, 1056, 255 },
		{ 157, 1076, 255 },
		{ 157, 0, 241 },
		{ 157, 1233, 255 },
		{ 157, 0, 242 },
		{ 157, 0, 244 },
		{ 157, 1224, 255 },
		{ 157, 1252, 255 },
		{ 157, 0, 240 },
		{ 157, 1271, 255 },
		{ 157, 0, 247 },
		{ 157, 738, 255 },
		{ 157, 1299, 255 },
		{ 0, 0, 250 },
		{ 157, 1284, 255 },
		{ 157, 1287, 255 },
		{ 3081, 1355, 249 },
		{ 2758, 4675, 423 },
		{ 163, 0, 236 },
		{ 0, 0, 237 },
		{ -161, 22, 232 },
		{ -162, 4711, 0 },
		{ 3036, 4697, 0 },
		{ 2758, 4676, 0 },
		{ 0, 0, 233 },
		{ 2758, 4693, 0 },
		{ -167, 4894, 0 },
		{ -168, 4717, 0 },
		{ 171, 0, 234 },
		{ 2758, 4694, 0 },
		{ 3036, 4864, 0 },
		{ 0, 0, 235 },
		{ 3037, 1577, 137 },
		{ 2087, 4081, 137 },
		{ 2916, 4559, 137 },
		{ 3037, 4243, 137 },
		{ 0, 0, 137 },
		{ 3025, 3339, 0 },
		{ 2102, 2980, 0 },
		{ 3025, 3582, 0 },
		{ 3002, 3315, 0 },
		{ 3002, 3316, 0 },
		{ 2087, 4076, 0 },
		{ 3029, 3223, 0 },
		{ 2087, 4093, 0 },
		{ 3003, 3503, 0 },
		{ 3032, 3206, 0 },
		{ 3003, 3274, 0 },
		{ 2852, 4295, 0 },
		{ 3029, 3254, 0 },
		{ 2102, 3656, 0 },
		{ 2916, 4439, 0 },
		{ 2033, 3415, 0 },
		{ 2033, 3461, 0 },
		{ 2055, 3063, 0 },
		{ 2036, 3773, 0 },
		{ 2017, 3838, 0 },
		{ 2036, 3785, 0 },
		{ 2055, 3066, 0 },
		{ 2055, 3090, 0 },
		{ 3029, 3295, 0 },
		{ 2957, 3920, 0 },
		{ 2087, 4045, 0 },
		{ 1175, 4000, 0 },
		{ 2033, 3413, 0 },
		{ 2055, 3100, 0 },
		{ 2916, 4491, 0 },
		{ 2033, 3436, 0 },
		{ 3002, 3719, 0 },
		{ 3025, 3580, 0 },
		{ 2102, 3692, 0 },
		{ 2186, 4166, 0 },
		{ 2916, 3596, 0 },
		{ 2957, 3876, 0 },
		{ 2017, 3824, 0 },
		{ 3003, 3529, 0 },
		{ 1995, 3268, 0 },
		{ 2916, 4569, 0 },
		{ 3025, 3610, 0 },
		{ 2957, 3598, 0 },
		{ 2102, 3662, 0 },
		{ 2055, 3102, 0 },
		{ 3032, 3366, 0 },
		{ 3002, 3724, 0 },
		{ 1995, 3266, 0 },
		{ 2017, 3811, 0 },
		{ 2916, 4343, 0 },
		{ 2087, 4054, 0 },
		{ 2087, 4072, 0 },
		{ 3025, 3629, 0 },
		{ 2055, 3115, 0 },
		{ 2087, 4083, 0 },
		{ 3025, 3612, 0 },
		{ 2186, 4213, 0 },
		{ 2916, 4555, 0 },
		{ 3032, 3368, 0 },
		{ 3003, 3551, 0 },
		{ 2055, 3151, 0 },
		{ 3032, 3288, 0 },
		{ 3025, 3570, 0 },
		{ 2017, 3850, 0 },
		{ 2957, 3906, 0 },
		{ 2102, 3596, 0 },
		{ 1067, 3211, 0 },
		{ 3025, 3592, 0 },
		{ 3002, 3747, 0 },
		{ 2916, 4511, 0 },
		{ 2186, 4211, 0 },
		{ 2055, 3152, 0 },
		{ 2102, 3685, 0 },
		{ 3032, 3343, 0 },
		{ 2087, 4101, 0 },
		{ 1995, 3238, 0 },
		{ 2087, 4051, 0 },
		{ 3003, 3555, 0 },
		{ 2055, 3153, 0 },
		{ 2852, 4300, 0 },
		{ 3002, 3718, 0 },
		{ 3032, 3378, 0 },
		{ 2123, 3588, 0 },
		{ 2055, 3154, 0 },
		{ 2957, 3885, 0 },
		{ 2087, 4014, 0 },
		{ 2186, 4170, 0 },
		{ 2087, 4036, 0 },
		{ 2916, 4405, 0 },
		{ 2916, 4411, 0 },
		{ 2017, 3833, 0 },
		{ 2186, 4131, 0 },
		{ 2055, 3155, 0 },
		{ 2916, 4517, 0 },
		{ 2957, 3944, 0 },
		{ 2017, 3849, 0 },
		{ 2957, 3868, 0 },
		{ 2916, 4339, 0 },
		{ 2033, 3459, 0 },
		{ 3003, 3507, 0 },
		{ 2087, 4084, 0 },
		{ 2916, 4423, 0 },
		{ 1175, 3983, 0 },
		{ 1067, 3212, 0 },
		{ 2123, 3976, 0 },
		{ 2036, 3777, 0 },
		{ 3003, 3541, 0 },
		{ 2055, 3156, 0 },
		{ 2916, 4567, 0 },
		{ 2087, 4052, 0 },
		{ 2055, 3157, 0 },
		{ 0, 0, 70 },
		{ 3025, 3327, 0 },
		{ 3032, 3318, 0 },
		{ 2087, 4079, 0 },
		{ 3037, 4253, 0 },
		{ 2055, 3158, 0 },
		{ 2055, 3159, 0 },
		{ 3032, 3349, 0 },
		{ 2033, 3429, 0 },
		{ 2017, 3806, 0 },
		{ 3032, 3351, 0 },
		{ 2055, 3160, 0 },
		{ 2087, 4041, 0 },
		{ 3025, 3638, 0 },
		{ 3025, 3641, 0 },
		{ 2036, 3772, 0 },
		{ 3003, 3517, 0 },
		{ 841, 3228, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2033, 3462, 0 },
		{ 2916, 4450, 0 },
		{ 2957, 3925, 0 },
		{ 2017, 3822, 0 },
		{ 3032, 3377, 0 },
		{ 3003, 3552, 0 },
		{ 0, 0, 68 },
		{ 2055, 3047, 0 },
		{ 2033, 3419, 0 },
		{ 3032, 3379, 0 },
		{ 2957, 3937, 0 },
		{ 3032, 3386, 0 },
		{ 2916, 4367, 0 },
		{ 3003, 3524, 0 },
		{ 1175, 4008, 0 },
		{ 2033, 3456, 0 },
		{ 3002, 3758, 0 },
		{ 2087, 4056, 0 },
		{ 2916, 4477, 0 },
		{ 3037, 4267, 0 },
		{ 3003, 3531, 0 },
		{ 0, 0, 62 },
		{ 3003, 3539, 0 },
		{ 2916, 4549, 0 },
		{ 1175, 3994, 0 },
		{ 2957, 3931, 0 },
		{ 2087, 4082, 0 },
		{ 0, 0, 73 },
		{ 3032, 3400, 0 },
		{ 3025, 3574, 0 },
		{ 3025, 3626, 0 },
		{ 2055, 3054, 0 },
		{ 2957, 3900, 0 },
		{ 2087, 4022, 0 },
		{ 2055, 3056, 0 },
		{ 2033, 3408, 0 },
		{ 3002, 3728, 0 },
		{ 3032, 3319, 0 },
		{ 3003, 3504, 0 },
		{ 2916, 4503, 0 },
		{ 2055, 3057, 0 },
		{ 2957, 3875, 0 },
		{ 2087, 4063, 0 },
		{ 3032, 3321, 0 },
		{ 3003, 3519, 0 },
		{ 2957, 3893, 0 },
		{ 1009, 3954, 0 },
		{ 0, 0, 8 },
		{ 2033, 3423, 0 },
		{ 2036, 3795, 0 },
		{ 2957, 3916, 0 },
		{ 2068, 3206, 0 },
		{ 2055, 3058, 0 },
		{ 2186, 4219, 0 },
		{ 2087, 4106, 0 },
		{ 2033, 3453, 0 },
		{ 2087, 4015, 0 },
		{ 2036, 3783, 0 },
		{ 2055, 3060, 0 },
		{ 3032, 3352, 0 },
		{ 3032, 3244, 0 },
		{ 2087, 4050, 0 },
		{ 3029, 3297, 0 },
		{ 3003, 3557, 0 },
		{ 1175, 4003, 0 },
		{ 3002, 3712, 0 },
		{ 2055, 3062, 0 },
		{ 2102, 3050, 0 },
		{ 2916, 4359, 0 },
		{ 1175, 3998, 0 },
		{ 2102, 3668, 0 },
		{ 3037, 3191, 0 },
		{ 2068, 3204, 0 },
		{ 2036, 3778, 0 },
		{ 2033, 3421, 0 },
		{ 3032, 3399, 0 },
		{ 0, 0, 114 },
		{ 2055, 3064, 0 },
		{ 2102, 3679, 0 },
		{ 1994, 3221, 0 },
		{ 3025, 3605, 0 },
		{ 3002, 3737, 0 },
		{ 2916, 4551, 0 },
		{ 2087, 4031, 0 },
		{ 0, 0, 7 },
		{ 2036, 3781, 0 },
		{ 0, 0, 6 },
		{ 2957, 3918, 0 },
		{ 0, 0, 119 },
		{ 3002, 3742, 0 },
		{ 2087, 4049, 0 },
		{ 3037, 1635, 0 },
		{ 2055, 3067, 0 },
		{ 3002, 3766, 0 },
		{ 3025, 3621, 0 },
		{ 2055, 3068, 0 },
		{ 2087, 4057, 0 },
		{ 3037, 3214, 0 },
		{ 2087, 4070, 0 },
		{ 2186, 4172, 0 },
		{ 2102, 3690, 0 },
		{ 0, 0, 69 },
		{ 2017, 3827, 0 },
		{ 2055, 3069, 106 },
		{ 2055, 3070, 107 },
		{ 2916, 4522, 0 },
		{ 2957, 3911, 0 },
		{ 3003, 3560, 0 },
		{ 3025, 3600, 0 },
		{ 3003, 3477, 0 },
		{ 1175, 3992, 0 },
		{ 3025, 3606, 0 },
		{ 3003, 3501, 0 },
		{ 3029, 3306, 0 },
		{ 2102, 3647, 0 },
		{ 2957, 3951, 0 },
		{ 2087, 4035, 0 },
		{ 2055, 3071, 0 },
		{ 3032, 3353, 0 },
		{ 2916, 4429, 0 },
		{ 2957, 3877, 0 },
		{ 2852, 4299, 0 },
		{ 2957, 3878, 0 },
		{ 3003, 3512, 0 },
		{ 2957, 3891, 0 },
		{ 0, 0, 9 },
		{ 2055, 3072, 0 },
		{ 2957, 3894, 0 },
		{ 2068, 3193, 0 },
		{ 2123, 3975, 0 },
		{ 0, 0, 104 },
		{ 2033, 3426, 0 },
		{ 3002, 3764, 0 },
		{ 3025, 3589, 0 },
		{ 2102, 3597, 0 },
		{ 3002, 3207, 0 },
		{ 2957, 3924, 0 },
		{ 3029, 3255, 0 },
		{ 2957, 3926, 0 },
		{ 3029, 3236, 0 },
		{ 3025, 3616, 0 },
		{ 2087, 4099, 0 },
		{ 3032, 3380, 0 },
		{ 3003, 3544, 0 },
		{ 3029, 3232, 0 },
		{ 3002, 3753, 0 },
		{ 3032, 3387, 0 },
		{ 2957, 3867, 0 },
		{ 3032, 3289, 0 },
		{ 2916, 4509, 0 },
		{ 2055, 3073, 0 },
		{ 2916, 4515, 0 },
		{ 3003, 3558, 0 },
		{ 2087, 4043, 0 },
		{ 3025, 3565, 0 },
		{ 3025, 3567, 0 },
		{ 2017, 3810, 0 },
		{ 2033, 3411, 0 },
		{ 2055, 3076, 94 },
		{ 3003, 3493, 0 },
		{ 2055, 3077, 0 },
		{ 2055, 3078, 0 },
		{ 3029, 3292, 0 },
		{ 2102, 3670, 0 },
		{ 2186, 4182, 0 },
		{ 2055, 3079, 0 },
		{ 2033, 3425, 0 },
		{ 0, 0, 103 },
		{ 2186, 4215, 0 },
		{ 2916, 4437, 0 },
		{ 3032, 3332, 0 },
		{ 3032, 3340, 0 },
		{ 0, 0, 116 },
		{ 0, 0, 118 },
		{ 2102, 3646, 0 },
		{ 2033, 3435, 0 },
		{ 2087, 3326, 0 },
		{ 3032, 3345, 0 },
		{ 2087, 4096, 0 },
		{ 2055, 3080, 0 },
		{ 2087, 4100, 0 },
		{ 2957, 3890, 0 },
		{ 3002, 3745, 0 },
		{ 2055, 3083, 0 },
		{ 2123, 3970, 0 },
		{ 3029, 3290, 0 },
		{ 2186, 4174, 0 },
		{ 2055, 3084, 0 },
		{ 2916, 4329, 0 },
		{ 2033, 3407, 0 },
		{ 939, 3856, 0 },
		{ 3032, 3354, 0 },
		{ 3002, 3708, 0 },
		{ 2102, 3696, 0 },
		{ 2186, 4143, 0 },
		{ 3032, 3365, 0 },
		{ 1995, 3245, 0 },
		{ 2055, 3085, 0 },
		{ 2957, 3927, 0 },
		{ 3025, 3602, 0 },
		{ 2033, 3417, 0 },
		{ 2916, 4479, 0 },
		{ 3032, 3371, 0 },
		{ 2102, 3676, 0 },
		{ 2068, 3205, 0 },
		{ 3003, 3502, 0 },
		{ 2033, 3422, 0 },
		{ 2102, 3691, 0 },
		{ 2036, 3769, 0 },
		{ 3037, 3288, 0 },
		{ 2055, 3086, 0 },
		{ 0, 0, 63 },
		{ 2055, 3088, 0 },
		{ 3025, 3630, 0 },
		{ 3003, 3513, 0 },
		{ 3025, 3639, 0 },
		{ 3003, 3515, 0 },
		{ 2055, 3089, 108 },
		{ 2017, 3829, 0 },
		{ 2102, 3675, 0 },
		{ 2036, 3770, 0 },
		{ 2033, 3433, 0 },
		{ 2033, 3434, 0 },
		{ 2017, 3853, 0 },
		{ 2036, 3776, 0 },
		{ 2916, 4435, 0 },
		{ 3025, 3571, 0 },
		{ 3029, 3303, 0 },
		{ 2957, 3950, 0 },
		{ 3032, 3389, 0 },
		{ 2033, 3438, 0 },
		{ 0, 0, 120 },
		{ 2852, 4301, 0 },
		{ 2036, 3784, 0 },
		{ 3032, 3392, 0 },
		{ 3002, 3710, 0 },
		{ 0, 0, 115 },
		{ 0, 0, 105 },
		{ 2033, 3454, 0 },
		{ 3003, 3550, 0 },
		{ 3032, 3396, 0 },
		{ 2102, 2956, 0 },
		{ 3037, 3243, 0 },
		{ 2957, 3895, 0 },
		{ 3002, 3731, 0 },
		{ 2055, 3091, 0 },
		{ 2957, 3910, 0 },
		{ 2017, 3819, 0 },
		{ 3025, 3640, 0 },
		{ 2087, 4088, 0 },
		{ 2916, 4345, 0 },
		{ 2916, 4357, 0 },
		{ 2033, 3473, 0 },
		{ 2916, 4365, 0 },
		{ 2957, 3919, 0 },
		{ 2916, 4369, 0 },
		{ 3003, 3559, 0 },
		{ 3002, 3750, 0 },
		{ 2055, 3092, 0 },
		{ 2087, 4104, 0 },
		{ 3003, 3561, 0 },
		{ 2055, 3093, 0 },
		{ 3003, 3479, 0 },
		{ 2087, 4016, 0 },
		{ 2957, 3932, 0 },
		{ 2087, 4026, 0 },
		{ 3003, 3484, 0 },
		{ 2087, 4032, 0 },
		{ 2033, 3410, 0 },
		{ 3002, 3711, 0 },
		{ 2055, 3094, 0 },
		{ 3037, 4161, 0 },
		{ 2186, 4147, 0 },
		{ 2087, 4042, 0 },
		{ 2036, 3779, 0 },
		{ 2087, 4044, 0 },
		{ 2036, 3780, 0 },
		{ 3025, 3576, 0 },
		{ 2916, 4568, 0 },
		{ 3025, 3624, 0 },
		{ 0, 0, 96 },
		{ 2957, 3879, 0 },
		{ 2957, 3883, 0 },
		{ 2916, 4341, 0 },
		{ 2055, 3096, 0 },
		{ 2055, 3098, 0 },
		{ 2916, 4347, 0 },
		{ 2916, 4349, 0 },
		{ 2916, 4351, 0 },
		{ 2036, 3794, 0 },
		{ 2033, 3416, 0 },
		{ 0, 0, 124 },
		{ 0, 0, 117 },
		{ 0, 0, 121 },
		{ 2916, 4363, 0 },
		{ 2186, 4151, 0 },
		{ 1067, 3217, 0 },
		{ 2055, 3099, 0 },
		{ 2957, 3053, 0 },
		{ 3003, 3514, 0 },
		{ 2036, 3774, 0 },
		{ 1175, 3987, 0 },
		{ 2916, 4433, 0 },
		{ 3025, 3642, 0 },
		{ 3025, 3583, 0 },
		{ 3025, 3590, 0 },
		{ 3002, 3759, 0 },
		{ 2186, 4137, 0 },
		{ 2123, 3965, 0 },
		{ 3002, 3763, 0 },
		{ 3029, 3300, 0 },
		{ 2017, 3836, 0 },
		{ 1175, 3990, 0 },
		{ 3032, 3350, 0 },
		{ 2017, 3843, 0 },
		{ 2033, 3424, 0 },
		{ 2055, 3101, 0 },
		{ 2036, 3790, 0 },
		{ 2017, 3804, 0 },
		{ 2087, 4029, 0 },
		{ 2123, 3972, 0 },
		{ 2102, 3667, 0 },
		{ 3003, 3526, 0 },
		{ 2916, 4571, 0 },
		{ 2102, 3669, 0 },
		{ 0, 0, 57 },
		{ 0, 0, 58 },
		{ 2916, 4335, 0 },
		{ 3003, 3528, 0 },
		{ 0, 0, 67 },
		{ 0, 0, 112 },
		{ 1995, 3246, 0 },
		{ 3029, 3275, 0 },
		{ 2033, 3431, 0 },
		{ 3029, 3282, 0 },
		{ 3032, 3363, 0 },
		{ 2957, 3889, 0 },
		{ 3003, 3545, 0 },
		{ 0, 0, 98 },
		{ 3003, 3546, 0 },
		{ 0, 0, 100 },
		{ 3025, 3637, 0 },
		{ 3003, 3549, 0 },
		{ 3002, 3756, 0 },
		{ 2087, 4061, 0 },
		{ 2068, 3198, 0 },
		{ 2068, 3199, 0 },
		{ 3032, 3367, 0 },
		{ 1175, 3985, 0 },
		{ 2123, 3714, 0 },
		{ 3003, 3553, 0 },
		{ 939, 3859, 0 },
		{ 2017, 3818, 0 },
		{ 0, 0, 113 },
		{ 0, 0, 123 },
		{ 3003, 3554, 0 },
		{ 0, 0, 136 },
		{ 2033, 3439, 0 },
		{ 3037, 3875, 0 },
		{ 2916, 4487, 0 },
		{ 1175, 4005, 0 },
		{ 2916, 4495, 0 },
		{ 2087, 4098, 0 },
		{ 3037, 4262, 0 },
		{ 3003, 3556, 0 },
		{ 3037, 4270, 0 },
		{ 3029, 3293, 0 },
		{ 3029, 3294, 0 },
		{ 3002, 3049, 0 },
		{ 2055, 3103, 0 },
		{ 2087, 4012, 0 },
		{ 2957, 3939, 0 },
		{ 2916, 4563, 0 },
		{ 2916, 4565, 0 },
		{ 2957, 3942, 0 },
		{ 2102, 3686, 0 },
		{ 2957, 3945, 0 },
		{ 2102, 3689, 0 },
		{ 2102, 3599, 0 },
		{ 2957, 3873, 0 },
		{ 2055, 3104, 0 },
		{ 2186, 4141, 0 },
		{ 2055, 3106, 0 },
		{ 3025, 3619, 0 },
		{ 2055, 3107, 0 },
		{ 2036, 3787, 0 },
		{ 3025, 3622, 0 },
		{ 2017, 3812, 0 },
		{ 2957, 3888, 0 },
		{ 2055, 3108, 0 },
		{ 3025, 3625, 0 },
		{ 3037, 4263, 0 },
		{ 1175, 4007, 0 },
		{ 2102, 3666, 0 },
		{ 3003, 3485, 0 },
		{ 2916, 4413, 0 },
		{ 2916, 4419, 0 },
		{ 2087, 4055, 0 },
		{ 2036, 3771, 0 },
		{ 2186, 4139, 0 },
		{ 3003, 3492, 0 },
		{ 2087, 4058, 0 },
		{ 2087, 4059, 0 },
		{ 2957, 3896, 0 },
		{ 2957, 3897, 0 },
		{ 2087, 4064, 0 },
		{ 2916, 4483, 0 },
		{ 2916, 4485, 0 },
		{ 2087, 4065, 0 },
		{ 2055, 3109, 0 },
		{ 3032, 3390, 0 },
		{ 3032, 3391, 0 },
		{ 2087, 4077, 0 },
		{ 2017, 3839, 0 },
		{ 3029, 3289, 0 },
		{ 2017, 3846, 0 },
		{ 3037, 4272, 0 },
		{ 3032, 3394, 0 },
		{ 2055, 3110, 60 },
		{ 3032, 3398, 0 },
		{ 2916, 4557, 0 },
		{ 2102, 3688, 0 },
		{ 2055, 3111, 0 },
		{ 2055, 3112, 0 },
		{ 2123, 3973, 0 },
		{ 2957, 3930, 0 },
		{ 3037, 4249, 0 },
		{ 3002, 3735, 0 },
		{ 3032, 3401, 0 },
		{ 3032, 3402, 0 },
		{ 1067, 3209, 0 },
		{ 2017, 3820, 0 },
		{ 3003, 3520, 0 },
		{ 2068, 3188, 0 },
		{ 1995, 3247, 0 },
		{ 1995, 3248, 0 },
		{ 3025, 3613, 0 },
		{ 2033, 3430, 0 },
		{ 1175, 3995, 0 },
		{ 3029, 3301, 0 },
		{ 3003, 3534, 0 },
		{ 3037, 4224, 0 },
		{ 3037, 4237, 0 },
		{ 2087, 4039, 0 },
		{ 0, 0, 61 },
		{ 0, 0, 64 },
		{ 2916, 4407, 0 },
		{ 1175, 4001, 0 },
		{ 2017, 3840, 0 },
		{ 3003, 3535, 0 },
		{ 1175, 4006, 0 },
		{ 3003, 3537, 0 },
		{ 0, 0, 109 },
		{ 3032, 3326, 0 },
		{ 3032, 3331, 0 },
		{ 3003, 3543, 0 },
		{ 0, 0, 102 },
		{ 2055, 3113, 0 },
		{ 2916, 4445, 0 },
		{ 0, 0, 110 },
		{ 0, 0, 111 },
		{ 2102, 3687, 0 },
		{ 2017, 3808, 0 },
		{ 3002, 3723, 0 },
		{ 939, 3858, 0 },
		{ 2036, 3786, 0 },
		{ 1175, 3999, 0 },
		{ 3032, 3333, 0 },
		{ 0, 0, 3 },
		{ 2087, 4062, 0 },
		{ 3037, 4288, 0 },
		{ 2916, 4505, 0 },
		{ 3002, 3725, 0 },
		{ 2957, 3905, 0 },
		{ 3032, 3337, 0 },
		{ 2957, 3909, 0 },
		{ 2087, 4071, 0 },
		{ 2055, 3114, 0 },
		{ 2036, 3768, 0 },
		{ 2186, 4221, 0 },
		{ 2033, 3450, 0 },
		{ 2033, 3451, 0 },
		{ 2087, 4080, 0 },
		{ 3002, 3738, 0 },
		{ 1009, 3953, 0 },
		{ 2087, 3059, 0 },
		{ 2957, 3922, 0 },
		{ 2102, 3703, 0 },
		{ 0, 0, 71 },
		{ 2087, 4090, 0 },
		{ 0, 0, 79 },
		{ 2916, 4331, 0 },
		{ 2087, 4091, 0 },
		{ 2916, 4337, 0 },
		{ 3032, 3344, 0 },
		{ 2087, 4095, 0 },
		{ 3029, 3281, 0 },
		{ 3037, 4275, 0 },
		{ 2087, 4097, 0 },
		{ 2102, 3648, 0 },
		{ 2017, 3841, 0 },
		{ 3032, 3347, 0 },
		{ 2017, 3844, 0 },
		{ 2957, 3934, 0 },
		{ 2102, 3657, 0 },
		{ 2957, 3938, 0 },
		{ 2087, 4013, 0 },
		{ 0, 0, 66 },
		{ 2102, 3660, 0 },
		{ 2102, 3661, 0 },
		{ 2916, 4409, 0 },
		{ 2036, 3782, 0 },
		{ 3032, 3348, 0 },
		{ 3002, 3765, 0 },
		{ 2087, 4027, 0 },
		{ 2102, 3664, 0 },
		{ 2087, 4030, 0 },
		{ 2055, 3116, 0 },
		{ 2957, 3874, 0 },
		{ 3025, 3603, 0 },
		{ 2036, 3788, 0 },
		{ 2017, 3813, 0 },
		{ 2033, 3463, 0 },
		{ 3037, 4274, 0 },
		{ 2033, 3472, 0 },
		{ 2055, 3120, 0 },
		{ 2102, 3672, 0 },
		{ 1995, 3244, 0 },
		{ 3037, 4239, 0 },
		{ 2087, 4046, 0 },
		{ 2087, 4048, 0 },
		{ 841, 3229, 0 },
		{ 0, 3224, 0 },
		{ 3002, 3726, 0 },
		{ 2123, 3977, 0 },
		{ 2087, 4053, 0 },
		{ 3003, 3480, 0 },
		{ 1175, 3993, 0 },
		{ 2916, 4553, 0 },
		{ 3003, 3483, 0 },
		{ 2055, 3121, 0 },
		{ 3032, 3356, 0 },
		{ 3003, 3486, 0 },
		{ 2017, 3842, 0 },
		{ 2957, 3902, 0 },
		{ 3003, 3491, 0 },
		{ 3002, 3743, 0 },
		{ 3032, 3357, 0 },
		{ 2186, 4162, 0 },
		{ 2186, 4164, 0 },
		{ 2916, 4333, 0 },
		{ 2087, 4069, 0 },
		{ 0, 0, 65 },
		{ 2017, 3848, 0 },
		{ 3032, 3358, 0 },
		{ 3025, 3636, 0 },
		{ 3003, 3495, 0 },
		{ 3003, 3497, 0 },
		{ 3003, 3499, 0 },
		{ 3032, 3359, 0 },
		{ 2102, 3650, 0 },
		{ 2102, 3653, 0 },
		{ 0, 0, 128 },
		{ 0, 0, 129 },
		{ 2036, 3791, 0 },
		{ 1175, 3996, 0 },
		{ 3032, 3360, 0 },
		{ 2017, 3814, 0 },
		{ 2017, 3815, 0 },
		{ 0, 0, 10 },
		{ 2916, 4373, 0 },
		{ 2033, 3420, 0 },
		{ 3032, 3361, 0 },
		{ 2916, 3987, 0 },
		{ 3037, 4285, 0 },
		{ 3002, 3709, 0 },
		{ 2916, 4415, 0 },
		{ 2916, 4417, 0 },
		{ 3032, 3362, 0 },
		{ 2055, 3122, 0 },
		{ 2957, 3940, 0 },
		{ 2957, 3941, 0 },
		{ 2087, 4103, 0 },
		{ 2055, 3126, 0 },
		{ 3037, 4257, 0 },
		{ 2916, 4443, 0 },
		{ 3037, 4259, 0 },
		{ 2017, 3828, 0 },
		{ 0, 0, 80 },
		{ 2102, 3665, 0 },
		{ 2957, 3947, 0 },
		{ 0, 0, 78 },
		{ 3029, 3298, 0 },
		{ 2036, 3775, 0 },
		{ 0, 0, 81 },
		{ 3037, 4273, 0 },
		{ 2957, 3865, 0 },
		{ 3029, 3299, 0 },
		{ 2087, 4025, 0 },
		{ 2033, 3427, 0 },
		{ 3003, 3518, 0 },
		{ 2087, 4028, 0 },
		{ 2055, 3127, 0 },
		{ 3032, 3369, 0 },
		{ 0, 0, 59 },
		{ 0, 0, 97 },
		{ 0, 0, 99 },
		{ 2102, 3674, 0 },
		{ 2186, 4168, 0 },
		{ 3003, 3522, 0 },
		{ 2087, 4034, 0 },
		{ 2957, 3880, 0 },
		{ 3025, 3614, 0 },
		{ 2087, 4037, 0 },
		{ 0, 0, 134 },
		{ 3003, 3523, 0 },
		{ 2087, 4040, 0 },
		{ 2957, 3886, 0 },
		{ 3032, 3370, 0 },
		{ 3003, 3525, 0 },
		{ 2916, 4570, 0 },
		{ 2055, 3128, 0 },
		{ 2017, 3797, 0 },
		{ 2017, 3802, 0 },
		{ 2087, 4047, 0 },
		{ 2186, 4145, 0 },
		{ 3032, 3372, 0 },
		{ 3032, 3373, 0 },
		{ 3003, 3530, 0 },
		{ 2123, 3963, 0 },
		{ 0, 3860, 0 },
		{ 3032, 3374, 0 },
		{ 3032, 3375, 0 },
		{ 2957, 3903, 0 },
		{ 3025, 3635, 0 },
		{ 2102, 3698, 0 },
		{ 2916, 4361, 0 },
		{ 2957, 3907, 0 },
		{ 3032, 3376, 0 },
		{ 2102, 3704, 0 },
		{ 3037, 4279, 0 },
		{ 0, 0, 19 },
		{ 2033, 3440, 0 },
		{ 2033, 3444, 0 },
		{ 0, 0, 125 },
		{ 2033, 3449, 0 },
		{ 0, 0, 127 },
		{ 3003, 3542, 0 },
		{ 2087, 4068, 0 },
		{ 0, 0, 95 },
		{ 2055, 3129, 0 },
		{ 2017, 3825, 0 },
		{ 2017, 3826, 0 },
		{ 2055, 3131, 0 },
		{ 2916, 4421, 0 },
		{ 2033, 3452, 0 },
		{ 2102, 3659, 0 },
		{ 2957, 3928, 0 },
		{ 0, 0, 76 },
		{ 2017, 3832, 0 },
		{ 1175, 4004, 0 },
		{ 2055, 3132, 0 },
		{ 2017, 3835, 0 },
		{ 3003, 3548, 0 },
		{ 2087, 4085, 0 },
		{ 3037, 4276, 0 },
		{ 3037, 4277, 0 },
		{ 2916, 4481, 0 },
		{ 2087, 4087, 0 },
		{ 2957, 3935, 0 },
		{ 2957, 3936, 0 },
		{ 2055, 3133, 0 },
		{ 2033, 3455, 0 },
		{ 3032, 3381, 0 },
		{ 3002, 3730, 0 },
		{ 3032, 3382, 0 },
		{ 2033, 3460, 0 },
		{ 2957, 3943, 0 },
		{ 3002, 3736, 0 },
		{ 3032, 3383, 0 },
		{ 2087, 4102, 0 },
		{ 0, 0, 84 },
		{ 3037, 4266, 0 },
		{ 0, 0, 101 },
		{ 2017, 3847, 0 },
		{ 3037, 3872, 0 },
		{ 2087, 4105, 0 },
		{ 0, 0, 132 },
		{ 3032, 3384, 0 },
		{ 3032, 3385, 0 },
		{ 2055, 3135, 56 },
		{ 3002, 3744, 0 },
		{ 2102, 3673, 0 },
		{ 2017, 3798, 0 },
		{ 3029, 3286, 0 },
		{ 2852, 3813, 0 },
		{ 0, 0, 85 },
		{ 2033, 3406, 0 },
		{ 3037, 4231, 0 },
		{ 1009, 3955, 0 },
		{ 0, 3956, 0 },
		{ 3032, 3388, 0 },
		{ 3002, 3754, 0 },
		{ 3002, 3755, 0 },
		{ 2102, 3677, 0 },
		{ 3037, 4258, 0 },
		{ 2087, 4033, 0 },
		{ 2957, 3887, 0 },
		{ 2055, 3136, 0 },
		{ 2102, 3681, 0 },
		{ 2102, 3682, 0 },
		{ 2102, 3683, 0 },
		{ 0, 0, 89 },
		{ 2957, 3892, 0 },
		{ 2033, 3409, 0 },
		{ 3003, 3478, 0 },
		{ 0, 0, 122 },
		{ 2055, 3137, 0 },
		{ 3029, 3291, 0 },
		{ 2055, 3138, 0 },
		{ 3025, 3633, 0 },
		{ 2957, 3901, 0 },
		{ 2186, 4217, 0 },
		{ 2033, 3414, 0 },
		{ 3002, 3716, 0 },
		{ 0, 0, 91 },
		{ 3002, 3717, 0 },
		{ 2186, 4133, 0 },
		{ 2186, 4135, 0 },
		{ 3032, 3393, 0 },
		{ 0, 0, 15 },
		{ 2017, 3831, 0 },
		{ 0, 0, 17 },
		{ 0, 0, 18 },
		{ 2957, 3908, 0 },
		{ 2055, 3139, 0 },
		{ 2123, 3964, 0 },
		{ 3002, 3721, 0 },
		{ 2916, 4441, 0 },
		{ 3003, 3487, 0 },
		{ 2102, 3702, 0 },
		{ 1175, 4002, 0 },
		{ 3003, 3489, 0 },
		{ 3003, 3490, 0 },
		{ 3002, 3727, 0 },
		{ 2102, 3705, 0 },
		{ 0, 0, 55 },
		{ 2957, 3923, 0 },
		{ 3032, 3395, 0 },
		{ 2055, 3140, 0 },
		{ 3032, 3397, 0 },
		{ 2017, 3845, 0 },
		{ 2102, 3649, 0 },
		{ 2087, 4075, 0 },
		{ 0, 0, 74 },
		{ 2055, 3141, 0 },
		{ 3037, 4235, 0 },
		{ 3003, 3496, 0 },
		{ 0, 3215, 0 },
		{ 3003, 3498, 0 },
		{ 0, 0, 16 },
		{ 2102, 3658, 0 },
		{ 1175, 3997, 0 },
		{ 2055, 3142, 54 },
		{ 2055, 3143, 0 },
		{ 2017, 3799, 0 },
		{ 0, 0, 75 },
		{ 3002, 3748, 0 },
		{ 3037, 3252, 0 },
		{ 0, 0, 82 },
		{ 0, 0, 83 },
		{ 0, 0, 52 },
		{ 3002, 3752, 0 },
		{ 3025, 3608, 0 },
		{ 3025, 3609, 0 },
		{ 3032, 3403, 0 },
		{ 3025, 3611, 0 },
		{ 0, 0, 133 },
		{ 3002, 3757, 0 },
		{ 1175, 4009, 0 },
		{ 1175, 4010, 0 },
		{ 3032, 3312, 0 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 2055, 3144, 40 },
		{ 3002, 3760, 0 },
		{ 3037, 4287, 0 },
		{ 1175, 3988, 0 },
		{ 1175, 3989, 0 },
		{ 2017, 3816, 0 },
		{ 0, 0, 72 },
		{ 3002, 3761, 0 },
		{ 3032, 3316, 0 },
		{ 0, 0, 90 },
		{ 3032, 3317, 0 },
		{ 2017, 3821, 0 },
		{ 3025, 3617, 0 },
		{ 2017, 3823, 0 },
		{ 2033, 3432, 0 },
		{ 2957, 3881, 0 },
		{ 3029, 3302, 0 },
		{ 2957, 3884, 0 },
		{ 2123, 3961, 0 },
		{ 2055, 3145, 0 },
		{ 3032, 3320, 0 },
		{ 3037, 4268, 0 },
		{ 3029, 3305, 0 },
		{ 0, 0, 86 },
		{ 3037, 4271, 0 },
		{ 2055, 3146, 0 },
		{ 0, 0, 126 },
		{ 0, 0, 130 },
		{ 2017, 3830, 0 },
		{ 0, 0, 135 },
		{ 0, 0, 11 },
		{ 3002, 3714, 0 },
		{ 3002, 3715, 0 },
		{ 2102, 3680, 0 },
		{ 3025, 3628, 0 },
		{ 1175, 3984, 0 },
		{ 2055, 3147, 0 },
		{ 3032, 3329, 0 },
		{ 3002, 3720, 0 },
		{ 3032, 3330, 0 },
		{ 3037, 4230, 0 },
		{ 0, 0, 131 },
		{ 2957, 3899, 0 },
		{ 3037, 4232, 0 },
		{ 3002, 3722, 0 },
		{ 3029, 3278, 0 },
		{ 3029, 3280, 0 },
		{ 3037, 4241, 0 },
		{ 2055, 3148, 0 },
		{ 3037, 4247, 0 },
		{ 2957, 3904, 0 },
		{ 2916, 4501, 0 },
		{ 3032, 3334, 0 },
		{ 3032, 3336, 0 },
		{ 2055, 3149, 0 },
		{ 0, 0, 41 },
		{ 3002, 3729, 0 },
		{ 2916, 4513, 0 },
		{ 3037, 4260, 0 },
		{ 3032, 3338, 0 },
		{ 2102, 3693, 0 },
		{ 2017, 3851, 0 },
		{ 2957, 3912, 0 },
		{ 2957, 3913, 0 },
		{ 2852, 4294, 0 },
		{ 3037, 4269, 0 },
		{ 2017, 3852, 0 },
		{ 2916, 4561, 0 },
		{ 2957, 3917, 0 },
		{ 3002, 3732, 0 },
		{ 2017, 3854, 0 },
		{ 2102, 3694, 0 },
		{ 2102, 3695, 0 },
		{ 2087, 4066, 0 },
		{ 3032, 3339, 0 },
		{ 2017, 3800, 0 },
		{ 2017, 3801, 0 },
		{ 2102, 3697, 0 },
		{ 0, 0, 77 },
		{ 0, 0, 92 },
		{ 3002, 3739, 0 },
		{ 3002, 3740, 0 },
		{ 0, 3991, 0 },
		{ 2957, 3929, 0 },
		{ 0, 0, 87 },
		{ 2017, 3805, 0 },
		{ 3002, 3741, 0 },
		{ 2033, 3457, 0 },
		{ 0, 0, 12 },
		{ 2102, 3699, 0 },
		{ 2102, 3700, 0 },
		{ 0, 0, 88 },
		{ 0, 0, 53 },
		{ 0, 0, 93 },
		{ 3003, 3538, 0 },
		{ 3002, 3746, 0 },
		{ 2087, 4086, 0 },
		{ 0, 0, 14 },
		{ 2055, 3150, 0 },
		{ 3003, 3540, 0 },
		{ 2087, 4089, 0 },
		{ 3025, 3601, 0 },
		{ 2017, 3817, 0 },
		{ 2916, 4378, 0 },
		{ 3037, 4261, 0 },
		{ 2087, 4092, 0 },
		{ 2036, 3789, 0 },
		{ 2087, 4094, 0 },
		{ 3002, 3751, 0 },
		{ 3032, 3341, 0 },
		{ 0, 0, 13 },
		{ 3081, 1435, 224 },
		{ 0, 0, 225 },
		{ 3036, 4888, 226 },
		{ 3064, 1642, 230 },
		{ 1212, 2593, 231 },
		{ 0, 0, 231 },
		{ 3064, 1698, 227 },
		{ 1215, 1451, 0 },
		{ 3064, 1709, 228 },
		{ 1218, 1517, 0 },
		{ 1215, 0, 0 },
		{ 2990, 2693, 229 },
		{ 1220, 1436, 0 },
		{ 1218, 0, 0 },
		{ 2990, 2568, 227 },
		{ 1220, 0, 0 },
		{ 2990, 2578, 228 },
		{ 3029, 3276, 144 },
		{ 0, 0, 144 },
		{ 0, 0, 145 },
		{ 3042, 1991, 0 },
		{ 3064, 2855, 0 },
		{ 3081, 2062, 0 },
		{ 1228, 4747, 0 },
		{ 3061, 2569, 0 },
		{ 3064, 2783, 0 },
		{ 3076, 2925, 0 },
		{ 3072, 2454, 0 },
		{ 3075, 3002, 0 },
		{ 3081, 2102, 0 },
		{ 3075, 2970, 0 },
		{ 3077, 1743, 0 },
		{ 2982, 2678, 0 },
		{ 3079, 2182, 0 },
		{ 3033, 2235, 0 },
		{ 3042, 1975, 0 },
		{ 3082, 4593, 0 },
		{ 0, 0, 142 },
		{ 2852, 4293, 152 },
		{ 0, 0, 152 },
		{ 3064, 2819, 0 },
		{ 2929, 2757, 0 },
		{ 3079, 2188, 0 },
		{ 3081, 2039, 0 },
		{ 3064, 2808, 0 },
		{ 1250, 4688, 0 },
		{ 3064, 2497, 0 },
		{ 3046, 1491, 0 },
		{ 3064, 2890, 0 },
		{ 3081, 2066, 0 },
		{ 2696, 1457, 0 },
		{ 3077, 1920, 0 },
		{ 3071, 2710, 0 },
		{ 2982, 2703, 0 },
		{ 3033, 2297, 0 },
		{ 2885, 2723, 0 },
		{ 1261, 4757, 0 },
		{ 3064, 2501, 0 },
		{ 3072, 2435, 0 },
		{ 3042, 1968, 0 },
		{ 3064, 2837, 0 },
		{ 1266, 4751, 0 },
		{ 3074, 2444, 0 },
		{ 3069, 1617, 0 },
		{ 3033, 2290, 0 },
		{ 3076, 2945, 0 },
		{ 3077, 1863, 0 },
		{ 2982, 2629, 0 },
		{ 3079, 2158, 0 },
		{ 3033, 2250, 0 },
		{ 3082, 4369, 0 },
		{ 0, 0, 150 },
		{ 3029, 3307, 172 },
		{ 0, 0, 172 },
		{ 3042, 1998, 0 },
		{ 3064, 2889, 0 },
		{ 3081, 2055, 0 },
		{ 1282, 4686, 0 },
		{ 3076, 2629, 0 },
		{ 3072, 2466, 0 },
		{ 3075, 3021, 0 },
		{ 3042, 2011, 0 },
		{ 3042, 2028, 0 },
		{ 3064, 2834, 0 },
		{ 3042, 2029, 0 },
		{ 3082, 4453, 0 },
		{ 0, 0, 171 },
		{ 2123, 3960, 140 },
		{ 0, 0, 140 },
		{ 0, 0, 141 },
		{ 3064, 2852, 0 },
		{ 3033, 2252, 0 },
		{ 3079, 2197, 0 },
		{ 3066, 2364, 0 },
		{ 3064, 2780, 0 },
		{ 3037, 4281, 0 },
		{ 3072, 2409, 0 },
		{ 3075, 2996, 0 },
		{ 3042, 1959, 0 },
		{ 3042, 1965, 0 },
		{ 3044, 4632, 0 },
		{ 3044, 4624, 0 },
		{ 2982, 2668, 0 },
		{ 3033, 2304, 0 },
		{ 2982, 2695, 0 },
		{ 3077, 1866, 0 },
		{ 2982, 2548, 0 },
		{ 3075, 2994, 0 },
		{ 3072, 2424, 0 },
		{ 2982, 2656, 0 },
		{ 3042, 1525, 0 },
		{ 3064, 2781, 0 },
		{ 3081, 2076, 0 },
		{ 3082, 4435, 0 },
		{ 0, 0, 138 },
		{ 2602, 2949, 23 },
		{ 0, 0, 23 },
		{ 0, 0, 24 },
		{ 3064, 2798, 0 },
		{ 2885, 2725, 0 },
		{ 2982, 2686, 0 },
		{ 3033, 2277, 0 },
		{ 3036, 9, 0 },
		{ 3079, 2177, 0 },
		{ 2844, 2124, 0 },
		{ 3064, 2844, 0 },
		{ 3081, 2085, 0 },
		{ 3075, 2971, 0 },
		{ 3077, 1934, 0 },
		{ 3079, 2150, 0 },
		{ 3081, 2034, 0 },
		{ 3036, 7, 0 },
		{ 3061, 2902, 0 },
		{ 3064, 2777, 0 },
		{ 3042, 1997, 0 },
		{ 3076, 2947, 0 },
		{ 3081, 2041, 0 },
		{ 2982, 2699, 0 },
		{ 2844, 2118, 0 },
		{ 3077, 1687, 0 },
		{ 2982, 2559, 0 },
		{ 3079, 2218, 0 },
		{ 3033, 2303, 0 },
		{ 3036, 4867, 0 },
		{ 3044, 4636, 0 },
		{ 0, 0, 20 },
		{ 1365, 0, 1 },
		{ 1365, 0, 173 },
		{ 1365, 2743, 223 },
		{ 1580, 174, 223 },
		{ 1580, 416, 223 },
		{ 1580, 404, 223 },
		{ 1580, 524, 223 },
		{ 1580, 405, 223 },
		{ 1580, 416, 223 },
		{ 1580, 391, 223 },
		{ 1580, 422, 223 },
		{ 1580, 482, 223 },
		{ 1365, 0, 223 },
		{ 1377, 2489, 223 },
		{ 1365, 2771, 223 },
		{ 2602, 2952, 219 },
		{ 1580, 503, 223 },
		{ 1580, 501, 223 },
		{ 1580, 516, 223 },
		{ 1580, 0, 223 },
		{ 1580, 551, 223 },
		{ 1580, 550, 223 },
		{ 3079, 2161, 0 },
		{ 0, 0, 174 },
		{ 3033, 2299, 0 },
		{ 1580, 517, 0 },
		{ 1580, 0, 0 },
		{ 3036, 3923, 0 },
		{ 1580, 537, 0 },
		{ 1580, 553, 0 },
		{ 1580, 552, 0 },
		{ 1580, 585, 0 },
		{ 1580, 577, 0 },
		{ 1580, 580, 0 },
		{ 1580, 588, 0 },
		{ 1580, 596, 0 },
		{ 1580, 589, 0 },
		{ 1580, 582, 0 },
		{ 1580, 586, 0 },
		{ 3064, 2868, 0 },
		{ 3064, 2879, 0 },
		{ 1581, 592, 0 },
		{ 1581, 595, 0 },
		{ 1580, 606, 0 },
		{ 1580, 618, 0 },
		{ 1580, 609, 0 },
		{ 3079, 2185, 0 },
		{ 3061, 2899, 0 },
		{ 1580, 607, 0 },
		{ 1580, 651, 0 },
		{ 1580, 628, 0 },
		{ 1580, 629, 0 },
		{ 1580, 652, 0 },
		{ 1580, 679, 0 },
		{ 1580, 684, 0 },
		{ 1580, 678, 0 },
		{ 1580, 654, 0 },
		{ 1580, 645, 0 },
		{ 1580, 642, 0 },
		{ 1580, 693, 0 },
		{ 1580, 680, 0 },
		{ 3033, 2328, 0 },
		{ 2929, 2746, 0 },
		{ 1580, 28, 0 },
		{ 1580, 30, 0 },
		{ 1581, 32, 0 },
		{ 1580, 28, 0 },
		{ 1580, 32, 0 },
		{ 3072, 2441, 0 },
		{ 0, 0, 222 },
		{ 1580, 44, 0 },
		{ 1580, 26, 0 },
		{ 1580, 14, 0 },
		{ 1580, 66, 0 },
		{ 1580, 70, 0 },
		{ 1580, 74, 0 },
		{ 1580, 70, 0 },
		{ 1580, 59, 0 },
		{ 1580, 38, 0 },
		{ 1580, 43, 0 },
		{ 1580, 60, 0 },
		{ 1580, 0, 208 },
		{ 1580, 109, 0 },
		{ 3079, 2157, 0 },
		{ 2982, 2490, 0 },
		{ 1580, 66, 0 },
		{ 1580, 71, 0 },
		{ 1580, 91, 0 },
		{ 1580, 94, 0 },
		{ 1580, 95, 0 },
		{ -1460, 1092, 0 },
		{ 1581, 132, 0 },
		{ 1580, 178, 0 },
		{ 1580, 184, 0 },
		{ 1580, 176, 0 },
		{ 1580, 186, 0 },
		{ 1580, 187, 0 },
		{ 1580, 166, 0 },
		{ 1580, 186, 0 },
		{ 1580, 163, 0 },
		{ 1580, 154, 0 },
		{ 1580, 0, 207 },
		{ 1580, 161, 0 },
		{ 3066, 2379, 0 },
		{ 3033, 2285, 0 },
		{ 1580, 165, 0 },
		{ 1580, 175, 0 },
		{ 1580, 183, 0 },
		{ 1580, 0, 221 },
		{ 1580, 181, 0 },
		{ 0, 0, 209 },
		{ 1580, 174, 0 },
		{ 1582, 33, -4 },
		{ 1580, 203, 0 },
		{ 1580, 215, 0 },
		{ 1580, 273, 0 },
		{ 1580, 279, 0 },
		{ 1580, 241, 0 },
		{ 1580, 255, 0 },
		{ 1580, 227, 0 },
		{ 1580, 273, 0 },
		{ 1580, 265, 0 },
		{ 3064, 2823, 0 },
		{ 3064, 2830, 0 },
		{ 1580, 0, 211 },
		{ 1580, 304, 212 },
		{ 1580, 273, 0 },
		{ 1580, 276, 0 },
		{ 1580, 303, 0 },
		{ 1480, 3479, 0 },
		{ 3036, 4260, 0 },
		{ 2164, 4599, 198 },
		{ 1580, 306, 0 },
		{ 1580, 316, 0 },
		{ 1580, 315, 0 },
		{ 1580, 319, 0 },
		{ 1580, 321, 0 },
		{ 1580, 335, 0 },
		{ 1580, 323, 0 },
		{ 1580, 325, 0 },
		{ 1580, 334, 0 },
		{ 1580, 343, 0 },
		{ 1581, 330, 0 },
		{ 3037, 4282, 0 },
		{ 3036, 4, 214 },
		{ 1580, 366, 0 },
		{ 1580, 378, 0 },
		{ 1580, 360, 0 },
		{ 1580, 376, 0 },
		{ 0, 0, 178 },
		{ 1582, 117, -7 },
		{ 1582, 231, -10 },
		{ 1582, 345, -13 },
		{ 1582, 376, -16 },
		{ 1582, 460, -19 },
		{ 1582, 488, -22 },
		{ 1580, 408, 0 },
		{ 1580, 421, 0 },
		{ 1580, 394, 0 },
		{ 1580, 0, 196 },
		{ 1580, 0, 210 },
		{ 3072, 2412, 0 },
		{ 1580, 394, 0 },
		{ 1580, 388, 0 },
		{ 1580, 397, 0 },
		{ 1581, 407, 0 },
		{ 1517, 3516, 0 },
		{ 3036, 4172, 0 },
		{ 2164, 4612, 199 },
		{ 1520, 3517, 0 },
		{ 3036, 4256, 0 },
		{ 2164, 4595, 200 },
		{ 1523, 3519, 0 },
		{ 3036, 4286, 0 },
		{ 2164, 4614, 203 },
		{ 1526, 3520, 0 },
		{ 3036, 4180, 0 },
		{ 2164, 4610, 204 },
		{ 1529, 3521, 0 },
		{ 3036, 4254, 0 },
		{ 2164, 4616, 205 },
		{ 1532, 3522, 0 },
		{ 3036, 4258, 0 },
		{ 2164, 4605, 206 },
		{ 1580, 452, 0 },
		{ 1582, 490, -25 },
		{ 1580, 457, 0 },
		{ 3075, 2985, 0 },
		{ 1580, 439, 0 },
		{ 1580, 519, 0 },
		{ 1580, 480, 0 },
		{ 1580, 491, 0 },
		{ 0, 0, 180 },
		{ 0, 0, 182 },
		{ 0, 0, 188 },
		{ 0, 0, 190 },
		{ 0, 0, 192 },
		{ 0, 0, 194 },
		{ 1582, 574, -28 },
		{ 1550, 3532, 0 },
		{ 3036, 4170, 0 },
		{ 2164, 4587, 202 },
		{ 1580, 0, 195 },
		{ 3042, 2030, 0 },
		{ 1580, 479, 0 },
		{ 1580, 494, 0 },
		{ 1581, 488, 0 },
		{ 1580, 485, 0 },
		{ 1559, 3538, 0 },
		{ 3036, 4262, 0 },
		{ 2164, 4593, 201 },
		{ 0, 0, 186 },
		{ 3042, 1983, 0 },
		{ 1580, 4, 217 },
		{ 1581, 491, 0 },
		{ 1580, 1, 220 },
		{ 1580, 506, 0 },
		{ 0, 0, 184 },
		{ 3044, 4634, 0 },
		{ 3044, 4635, 0 },
		{ 1580, 494, 0 },
		{ 0, 0, 218 },
		{ 1580, 491, 0 },
		{ 3044, 4625, 0 },
		{ 0, 0, 216 },
		{ 1580, 502, 0 },
		{ 1580, 507, 0 },
		{ 0, 0, 215 },
		{ 1580, 512, 0 },
		{ 1580, 503, 0 },
		{ 1581, 505, 213 },
		{ 1582, 925, 0 },
		{ 1583, 736, -1 },
		{ 1584, 3492, 0 },
		{ 3036, 4246, 0 },
		{ 2164, 4583, 197 },
		{ 0, 0, 176 },
		{ 2123, 3962, 266 },
		{ 0, 0, 266 },
		{ 3064, 2884, 0 },
		{ 3033, 2329, 0 },
		{ 3079, 2215, 0 },
		{ 3066, 2355, 0 },
		{ 3064, 2778, 0 },
		{ 3037, 4284, 0 },
		{ 3072, 2416, 0 },
		{ 3075, 3008, 0 },
		{ 3042, 1994, 0 },
		{ 3042, 1995, 0 },
		{ 3044, 4639, 0 },
		{ 3044, 4640, 0 },
		{ 2982, 2700, 0 },
		{ 3033, 2273, 0 },
		{ 2982, 2704, 0 },
		{ 3077, 1928, 0 },
		{ 2982, 2545, 0 },
		{ 3075, 3003, 0 },
		{ 3072, 2462, 0 },
		{ 2982, 2551, 0 },
		{ 3042, 1530, 0 },
		{ 3064, 2842, 0 },
		{ 3081, 2094, 0 },
		{ 3082, 4451, 0 },
		{ 0, 0, 265 },
		{ 2123, 3979, 268 },
		{ 0, 0, 268 },
		{ 0, 0, 269 },
		{ 3064, 2846, 0 },
		{ 3033, 2288, 0 },
		{ 3079, 2163, 0 },
		{ 3066, 2383, 0 },
		{ 3064, 2872, 0 },
		{ 3037, 4265, 0 },
		{ 3072, 2429, 0 },
		{ 3075, 2966, 0 },
		{ 3042, 2005, 0 },
		{ 3042, 2006, 0 },
		{ 3044, 4641, 0 },
		{ 3044, 4644, 0 },
		{ 3076, 2934, 0 },
		{ 3081, 2038, 0 },
		{ 3079, 2186, 0 },
		{ 3042, 2007, 0 },
		{ 3042, 2009, 0 },
		{ 3079, 2204, 0 },
		{ 3046, 1493, 0 },
		{ 3064, 2794, 0 },
		{ 3081, 2059, 0 },
		{ 3082, 4455, 0 },
		{ 0, 0, 267 },
		{ 2123, 3969, 271 },
		{ 0, 0, 271 },
		{ 0, 0, 272 },
		{ 3064, 2800, 0 },
		{ 3033, 2261, 0 },
		{ 3079, 2147, 0 },
		{ 3066, 2380, 0 },
		{ 3064, 2820, 0 },
		{ 3037, 4226, 0 },
		{ 3072, 2451, 0 },
		{ 3075, 3006, 0 },
		{ 3042, 2018, 0 },
		{ 3042, 2023, 0 },
		{ 3044, 4628, 0 },
		{ 3044, 4630, 0 },
		{ 3066, 2358, 0 },
		{ 3069, 1592, 0 },
		{ 3077, 1781, 0 },
		{ 3075, 2981, 0 },
		{ 3077, 1811, 0 },
		{ 3079, 2168, 0 },
		{ 3081, 2097, 0 },
		{ 3082, 4597, 0 },
		{ 0, 0, 270 },
		{ 2123, 3974, 274 },
		{ 0, 0, 274 },
		{ 0, 0, 275 },
		{ 3064, 2857, 0 },
		{ 3033, 2301, 0 },
		{ 3079, 2179, 0 },
		{ 3066, 2366, 0 },
		{ 3064, 2883, 0 },
		{ 3037, 4264, 0 },
		{ 3072, 2453, 0 },
		{ 3075, 2968, 0 },
		{ 3042, 2031, 0 },
		{ 3042, 2032, 0 },
		{ 3044, 4645, 0 },
		{ 3044, 4623, 0 },
		{ 3064, 2776, 0 },
		{ 3046, 1461, 0 },
		{ 3075, 2992, 0 },
		{ 3072, 2408, 0 },
		{ 3069, 1550, 0 },
		{ 3075, 3000, 0 },
		{ 3077, 1912, 0 },
		{ 3079, 2203, 0 },
		{ 3081, 2050, 0 },
		{ 3082, 4367, 0 },
		{ 0, 0, 273 },
		{ 2123, 3959, 277 },
		{ 0, 0, 277 },
		{ 0, 0, 278 },
		{ 3064, 2797, 0 },
		{ 3033, 2262, 0 },
		{ 3079, 2212, 0 },
		{ 3066, 2382, 0 },
		{ 3064, 2805, 0 },
		{ 3037, 4290, 0 },
		{ 3072, 2450, 0 },
		{ 3075, 2980, 0 },
		{ 3042, 1972, 0 },
		{ 3042, 1973, 0 },
		{ 3044, 4637, 0 },
		{ 3044, 4638, 0 },
		{ 3079, 2143, 0 },
		{ 2844, 2138, 0 },
		{ 3077, 1913, 0 },
		{ 2982, 2553, 0 },
		{ 3066, 2368, 0 },
		{ 2982, 2626, 0 },
		{ 3042, 1980, 0 },
		{ 3064, 2851, 0 },
		{ 3081, 2075, 0 },
		{ 3082, 4527, 0 },
		{ 0, 0, 276 },
		{ 2916, 4489, 155 },
		{ 0, 0, 155 },
		{ 2929, 2763, 0 },
		{ 3077, 1916, 0 },
		{ 3064, 2866, 0 },
		{ 3081, 2078, 0 },
		{ 1723, 4742, 0 },
		{ 3064, 2491, 0 },
		{ 3046, 1490, 0 },
		{ 3064, 2881, 0 },
		{ 3081, 2089, 0 },
		{ 2696, 1440, 0 },
		{ 3077, 1929, 0 },
		{ 3071, 2720, 0 },
		{ 2982, 2701, 0 },
		{ 3033, 2244, 0 },
		{ 2885, 2726, 0 },
		{ 1734, 4751, 0 },
		{ 3064, 2503, 0 },
		{ 3072, 2458, 0 },
		{ 3042, 1996, 0 },
		{ 3064, 2791, 0 },
		{ 1739, 4684, 0 },
		{ 3074, 2442, 0 },
		{ 3069, 1627, 0 },
		{ 3033, 2260, 0 },
		{ 3076, 2937, 0 },
		{ 3077, 1942, 0 },
		{ 2982, 2567, 0 },
		{ 3079, 2193, 0 },
		{ 3033, 2269, 0 },
		{ 3082, 4589, 0 },
		{ 0, 0, 153 },
		{ 2123, 3978, 259 },
		{ 0, 0, 259 },
		{ 3064, 2811, 0 },
		{ 3033, 2270, 0 },
		{ 3079, 2195, 0 },
		{ 3066, 2374, 0 },
		{ 3064, 2828, 0 },
		{ 3037, 4234, 0 },
		{ 3072, 2449, 0 },
		{ 3075, 3023, 0 },
		{ 3042, 2001, 0 },
		{ 3042, 2003, 0 },
		{ 3044, 4642, 0 },
		{ 3044, 4643, 0 },
		{ 3061, 2913, 0 },
		{ 2982, 2698, 0 },
		{ 3042, 2004, 0 },
		{ 2844, 2116, 0 },
		{ 3072, 2456, 0 },
		{ 3075, 2988, 0 },
		{ 2696, 1454, 0 },
		{ 3082, 4552, 0 },
		{ 0, 0, 257 },
		{ 1787, 0, 1 },
		{ 1946, 2808, 374 },
		{ 3064, 2861, 374 },
		{ 3075, 2914, 374 },
		{ 3061, 2152, 374 },
		{ 1787, 0, 341 },
		{ 1787, 2618, 374 },
		{ 3071, 1596, 374 },
		{ 2852, 4302, 374 },
		{ 2102, 3651, 374 },
		{ 3029, 3283, 374 },
		{ 2102, 3655, 374 },
		{ 2087, 4078, 374 },
		{ 3081, 1955, 374 },
		{ 1787, 0, 374 },
		{ 2602, 2950, 372 },
		{ 3075, 2741, 374 },
		{ 3075, 2976, 374 },
		{ 0, 0, 374 },
		{ 3079, 2156, 0 },
		{ -1792, 4899, 331 },
		{ -1793, 4716, 0 },
		{ 3033, 2314, 0 },
		{ 0, 0, 337 },
		{ 0, 0, 338 },
		{ 3072, 2418, 0 },
		{ 2982, 2563, 0 },
		{ 3064, 2892, 0 },
		{ 0, 0, 342 },
		{ 3033, 2321, 0 },
		{ 3081, 2067, 0 },
		{ 2982, 2627, 0 },
		{ 2055, 3065, 0 },
		{ 3025, 3615, 0 },
		{ 3032, 3327, 0 },
		{ 1995, 3265, 0 },
		{ 3025, 3618, 0 },
		{ 3069, 1624, 0 },
		{ 3042, 2017, 0 },
		{ 3033, 2240, 0 },
		{ 3077, 1838, 0 },
		{ 3081, 2083, 0 },
		{ 3079, 2170, 0 },
		{ 2758, 4698, 0 },
		{ 3079, 2173, 0 },
		{ 3042, 2019, 0 },
		{ 3077, 1860, 0 },
		{ 3033, 2267, 0 },
		{ 3061, 2918, 0 },
		{ 3081, 2090, 0 },
		{ 3072, 2406, 0 },
		{ 2123, 3966, 0 },
		{ 2055, 3081, 0 },
		{ 2055, 3082, 0 },
		{ 2087, 4024, 0 },
		{ 2017, 3834, 0 },
		{ 3064, 2815, 0 },
		{ 3042, 2026, 0 },
		{ 3061, 2915, 0 },
		{ 3069, 1625, 0 },
		{ 3064, 2821, 0 },
		{ 3072, 2410, 0 },
		{ 0, 4893, 334 },
		{ 3066, 2362, 0 },
		{ 3064, 2829, 0 },
		{ 2102, 3701, 0 },
		{ 3077, 1864, 0 },
		{ 0, 0, 373 },
		{ 3064, 2831, 0 },
		{ 3061, 2919, 0 },
		{ 2087, 4038, 0 },
		{ 2033, 3437, 0 },
		{ 3025, 3607, 0 },
		{ 3003, 3500, 0 },
		{ 2055, 3095, 0 },
		{ 0, 0, 362 },
		{ 3037, 4286, 0 },
		{ 3079, 2192, 0 },
		{ 3081, 2104, 0 },
		{ 3033, 2287, 0 },
		{ -1869, 1167, 0 },
		{ 0, 0, 333 },
		{ 3064, 2845, 0 },
		{ 0, 0, 361 },
		{ 2844, 2123, 0 },
		{ 2982, 2618, 0 },
		{ 3033, 2292, 0 },
		{ 1884, 4661, 0 },
		{ 3002, 3733, 0 },
		{ 2957, 3898, 0 },
		{ 3003, 3516, 0 },
		{ 2055, 3105, 0 },
		{ 3025, 3620, 0 },
		{ 3079, 2201, 0 },
		{ 3066, 2384, 0 },
		{ 3033, 2298, 0 },
		{ 3077, 1871, 0 },
		{ 0, 0, 363 },
		{ 3037, 4255, 340 },
		{ 3077, 1892, 0 },
		{ 3076, 2942, 0 },
		{ 3077, 1903, 0 },
		{ 0, 0, 366 },
		{ 0, 0, 367 },
		{ 1889, 0, -71 },
		{ 2068, 3197, 0 },
		{ 2102, 3671, 0 },
		{ 3025, 3632, 0 },
		{ 2087, 4067, 0 },
		{ 2982, 2679, 0 },
		{ 0, 0, 365 },
		{ 0, 0, 371 },
		{ 0, 4659, 0 },
		{ 3072, 2460, 0 },
		{ 3042, 1960, 0 },
		{ 3075, 3015, 0 },
		{ 2123, 3980, 0 },
		{ 3036, 4216, 0 },
		{ 2164, 4618, 356 },
		{ 2087, 4074, 0 },
		{ 2852, 4292, 0 },
		{ 3003, 3532, 0 },
		{ 3003, 3533, 0 },
		{ 3033, 2308, 0 },
		{ 0, 0, 368 },
		{ 0, 0, 369 },
		{ 3075, 3022, 0 },
		{ 2170, 4702, 0 },
		{ 3072, 2467, 0 },
		{ 3064, 2886, 0 },
		{ 0, 0, 346 },
		{ 1912, 0, -74 },
		{ 1914, 0, -77 },
		{ 2102, 3684, 0 },
		{ 3037, 4283, 0 },
		{ 0, 0, 364 },
		{ 3042, 1962, 0 },
		{ 0, 0, 339 },
		{ 2123, 3967, 0 },
		{ 3033, 2319, 0 },
		{ 3036, 4176, 0 },
		{ 2164, 4603, 357 },
		{ 3036, 4178, 0 },
		{ 2164, 4608, 358 },
		{ 2852, 4304, 0 },
		{ 1924, 0, -59 },
		{ 3042, 1963, 0 },
		{ 3064, 2893, 0 },
		{ 3064, 2775, 0 },
		{ 0, 0, 348 },
		{ 0, 0, 350 },
		{ 1929, 0, -65 },
		{ 3036, 4242, 0 },
		{ 2164, 4591, 360 },
		{ 0, 0, 336 },
		{ 3033, 2323, 0 },
		{ 3081, 2061, 0 },
		{ 3036, 4248, 0 },
		{ 2164, 4601, 359 },
		{ 0, 0, 354 },
		{ 3079, 2149, 0 },
		{ 3075, 2990, 0 },
		{ 0, 0, 352 },
		{ 3066, 2354, 0 },
		{ 3077, 1906, 0 },
		{ 3064, 2782, 0 },
		{ 2982, 2550, 0 },
		{ 0, 0, 370 },
		{ 3079, 2154, 0 },
		{ 3033, 2242, 0 },
		{ 1943, 0, -80 },
		{ 3036, 4168, 0 },
		{ 2164, 4589, 355 },
		{ 0, 0, 344 },
		{ 1787, 2798, 374 },
		{ 1950, 2491, 374 },
		{ -1948, 17, 331 },
		{ -1949, 4714, 0 },
		{ 3036, 4703, 0 },
		{ 2758, 4673, 0 },
		{ 0, 0, 332 },
		{ 2758, 4699, 0 },
		{ -1954, 9, 0 },
		{ -1955, 4720, 0 },
		{ 1958, 2, 334 },
		{ 2758, 4687, 0 },
		{ 3036, 4880, 0 },
		{ 0, 0, 335 },
		{ 1976, 0, 1 },
		{ 2172, 2815, 330 },
		{ 3064, 2807, 330 },
		{ 1976, 0, 284 },
		{ 1976, 2688, 330 },
		{ 3025, 3623, 330 },
		{ 1976, 0, 287 },
		{ 3069, 1600, 330 },
		{ 2852, 4296, 330 },
		{ 2102, 3652, 330 },
		{ 3029, 3240, 330 },
		{ 2102, 3654, 330 },
		{ 2087, 4023, 330 },
		{ 3075, 2977, 330 },
		{ 3081, 1962, 330 },
		{ 1976, 0, 330 },
		{ 2602, 2957, 327 },
		{ 3075, 2986, 330 },
		{ 3061, 2901, 330 },
		{ 2852, 4298, 330 },
		{ 3075, 1544, 330 },
		{ 0, 0, 330 },
		{ 3079, 2166, 0 },
		{ -1983, 4901, 279 },
		{ -1984, 4721, 0 },
		{ 3033, 2264, 0 },
		{ 0, 0, 285 },
		{ 3033, 2266, 0 },
		{ 3079, 2167, 0 },
		{ 3081, 2081, 0 },
		{ 2055, 3061, 0 },
		{ 3025, 3644, 0 },
		{ 3032, 3342, 0 },
		{ 3002, 3749, 0 },
		{ 0, 3219, 0 },
		{ 0, 3240, 0 },
		{ 3025, 3594, 0 },
		{ 3072, 2465, 0 },
		{ 3069, 1619, 0 },
		{ 3042, 1985, 0 },
		{ 3033, 2276, 0 },
		{ 3064, 2838, 0 },
		{ 3064, 2840, 0 },
		{ 3079, 2174, 0 },
		{ 2170, 1, 0 },
		{ 3079, 2175, 0 },
		{ 2758, 4692, 0 },
		{ 3079, 2176, 0 },
		{ 3061, 2900, 0 },
		{ 2844, 2126, 0 },
		{ 3081, 2088, 0 },
		{ 2123, 3968, 0 },
		{ 2055, 3074, 0 },
		{ 2055, 3075, 0 },
		{ 2957, 3914, 0 },
		{ 2957, 3915, 0 },
		{ 2087, 4060, 0 },
		{ 0, 3837, 0 },
		{ 3042, 1986, 0 },
		{ 3064, 2854, 0 },
		{ 3042, 1987, 0 },
		{ 3061, 2917, 0 },
		{ 3033, 2295, 0 },
		{ 3042, 1989, 0 },
		{ 3072, 2437, 0 },
		{ 0, 0, 329 },
		{ 3072, 2439, 0 },
		{ 0, 0, 281 },
		{ 3066, 2386, 0 },
		{ 0, 0, 326 },
		{ 3069, 1622, 0 },
		{ 3064, 2876, 0 },
		{ 2087, 4073, 0 },
		{ 0, 3412, 0 },
		{ 3025, 3627, 0 },
		{ 2036, 3792, 0 },
		{ 0, 3793, 0 },
		{ 3003, 3536, 0 },
		{ 2055, 3087, 0 },
		{ 3064, 2878, 0 },
		{ 0, 0, 319 },
		{ 3037, 4289, 0 },
		{ 3079, 2189, 0 },
		{ 3077, 1930, 0 },
		{ 3077, 1933, 0 },
		{ 3069, 1623, 0 },
		{ -2063, 1242, 0 },
		{ 3064, 2888, 0 },
		{ 3072, 2457, 0 },
		{ 3033, 2310, 0 },
		{ 3002, 3734, 0 },
		{ 2957, 3946, 0 },
		{ 3003, 3547, 0 },
		{ 2957, 3948, 0 },
		{ 2957, 3949, 0 },
		{ 0, 3097, 0 },
		{ 3025, 3643, 0 },
		{ 0, 0, 318 },
		{ 3079, 2198, 0 },
		{ 3066, 2376, 0 },
		{ 2982, 2670, 0 },
		{ 0, 0, 325 },
		{ 3075, 2972, 0 },
		{ 0, 0, 320 },
		{ 0, 0, 283 },
		{ 3075, 2974, 0 },
		{ 3077, 1679, 0 },
		{ 2080, 0, -56 },
		{ 0, 3194, 0 },
		{ 2102, 3663, 0 },
		{ 2071, 3189, 0 },
		{ 2068, 3190, 0 },
		{ 3025, 3604, 0 },
		{ 2087, 4107, 0 },
		{ 2982, 2673, 0 },
		{ 0, 0, 322 },
		{ 3076, 2927, 0 },
		{ 3077, 1680, 0 },
		{ 3077, 1681, 0 },
		{ 2123, 3958, 0 },
		{ 3036, 4250, 0 },
		{ 2164, 4585, 309 },
		{ 2087, 4017, 0 },
		{ 2852, 4297, 0 },
		{ 2087, 4018, 0 },
		{ 2087, 4019, 0 },
		{ 2087, 4020, 0 },
		{ 0, 4021, 0 },
		{ 3003, 3475, 0 },
		{ 3003, 3476, 0 },
		{ 3033, 2326, 0 },
		{ 3075, 2987, 0 },
		{ 2982, 2687, 0 },
		{ 2982, 2688, 0 },
		{ 3064, 2787, 0 },
		{ 0, 0, 291 },
		{ 2109, 0, -35 },
		{ 2111, 0, -38 },
		{ 2113, 0, -44 },
		{ 2115, 0, -47 },
		{ 2117, 0, -50 },
		{ 2119, 0, -53 },
		{ 0, 3678, 0 },
		{ 3037, 4233, 0 },
		{ 0, 0, 321 },
		{ 3072, 2407, 0 },
		{ 3079, 2205, 0 },
		{ 3079, 2211, 0 },
		{ 3033, 2331, 0 },
		{ 3036, 4182, 0 },
		{ 2164, 4606, 310 },
		{ 3036, 4184, 0 },
		{ 2164, 4609, 311 },
		{ 3036, 4186, 0 },
		{ 2164, 4611, 314 },
		{ 3036, 4188, 0 },
		{ 2164, 4613, 315 },
		{ 3036, 4212, 0 },
		{ 2164, 4615, 316 },
		{ 3036, 4214, 0 },
		{ 2164, 4617, 317 },
		{ 2852, 4303, 0 },
		{ 2134, 0, -62 },
		{ 0, 3971, 0 },
		{ 3033, 2232, 0 },
		{ 3033, 2233, 0 },
		{ 3064, 2803, 0 },
		{ 0, 0, 293 },
		{ 0, 0, 295 },
		{ 0, 0, 301 },
		{ 0, 0, 303 },
		{ 0, 0, 305 },
		{ 0, 0, 307 },
		{ 2140, 0, -68 },
		{ 3036, 4244, 0 },
		{ 2164, 4598, 313 },
		{ 3064, 2804, 0 },
		{ 3075, 3019, 0 },
		{ 3011, 3183, 324 },
		{ 3081, 2051, 0 },
		{ 3036, 4252, 0 },
		{ 2164, 4607, 312 },
		{ 0, 0, 299 },
		{ 3033, 2237, 0 },
		{ 3081, 2052, 0 },
		{ 0, 0, 286 },
		{ 3075, 3027, 0 },
		{ 0, 0, 297 },
		{ 3079, 2216, 0 },
		{ 2696, 1443, 0 },
		{ 3077, 1728, 0 },
		{ 3066, 2377, 0 },
		{ 2916, 4431, 0 },
		{ 2982, 2552, 0 },
		{ 3064, 2822, 0 },
		{ 3072, 2443, 0 },
		{ 3079, 2144, 0 },
		{ 0, 0, 323 },
		{ 2885, 2732, 0 },
		{ 3033, 2257, 0 },
		{ 3079, 2146, 0 },
		{ 2163, 0, -41 },
		{ 3081, 2060, 0 },
		{ 3036, 4174, 0 },
		{ 0, 4597, 308 },
		{ 2982, 2616, 0 },
		{ 0, 0, 289 },
		{ 3077, 1739, 0 },
		{ 3071, 2711, 0 },
		{ 3066, 2385, 0 },
		{ 0, 4703, 0 },
		{ 0, 0, 328 },
		{ 1976, 2828, 330 },
		{ 2176, 2488, 330 },
		{ -2174, 4900, 279 },
		{ -2175, 4719, 0 },
		{ 3036, 4704, 0 },
		{ 2758, 4672, 0 },
		{ 0, 0, 280 },
		{ 2758, 4690, 0 },
		{ -2180, 21, 0 },
		{ -2181, 4713, 0 },
		{ 2184, 0, 281 },
		{ 2758, 4691, 0 },
		{ 3036, 4836, 0 },
		{ 0, 0, 282 },
		{ 0, 4149, 376 },
		{ 0, 0, 376 },
		{ 3064, 2847, 0 },
		{ 2929, 2737, 0 },
		{ 3075, 3004, 0 },
		{ 3069, 1534, 0 },
		{ 3072, 2459, 0 },
		{ 3077, 1796, 0 },
		{ 3036, 2, 0 },
		{ 3081, 2074, 0 },
		{ 3069, 1546, 0 },
		{ 3033, 2272, 0 },
		{ 2199, 4681, 0 },
		{ 3036, 1943, 0 },
		{ 3075, 3025, 0 },
		{ 3081, 2077, 0 },
		{ 3075, 3028, 0 },
		{ 3066, 2372, 0 },
		{ 3064, 2875, 0 },
		{ 3077, 1845, 0 },
		{ 3064, 2877, 0 },
		{ 3081, 2079, 0 },
		{ 3042, 2022, 0 },
		{ 3082, 4447, 0 },
		{ 0, 0, 375 },
		{ 2758, 4684, 423 },
		{ 0, 0, 381 },
		{ 0, 0, 383 },
		{ 2230, 828, 414 },
		{ 2395, 841, 414 },
		{ 2416, 839, 414 },
		{ 2363, 841, 414 },
		{ 2231, 857, 414 },
		{ 2229, 834, 414 },
		{ 2416, 838, 414 },
		{ 2252, 853, 414 },
		{ 2391, 855, 414 },
		{ 2391, 859, 414 },
		{ 2395, 856, 414 },
		{ 2339, 865, 414 },
		{ 3064, 1620, 413 },
		{ 2259, 2637, 423 },
		{ 2449, 854, 414 },
		{ 2395, 867, 414 },
		{ 2262, 867, 414 },
		{ 2395, 861, 414 },
		{ 3064, 2793, 423 },
		{ -2233, 18, 377 },
		{ -2234, 4712, 0 },
		{ 2449, 858, 414 },
		{ 2454, 463, 414 },
		{ 2449, 860, 414 },
		{ 2308, 858, 414 },
		{ 2395, 870, 414 },
		{ 2400, 865, 414 },
		{ 2395, 872, 414 },
		{ 2339, 881, 414 },
		{ 2312, 871, 414 },
		{ 2363, 862, 414 },
		{ 2339, 884, 414 },
		{ 2416, 868, 414 },
		{ 2228, 861, 414 },
		{ 2365, 868, 414 },
		{ 2228, 865, 414 },
		{ 2428, 887, 414 },
		{ 2400, 887, 414 },
		{ 2228, 897, 414 },
		{ 2428, 890, 414 },
		{ 2449, 1265, 414 },
		{ 2428, 891, 414 },
		{ 2312, 894, 414 },
		{ 3064, 1774, 410 },
		{ 2287, 1483, 0 },
		{ 3064, 1797, 411 },
		{ 3033, 2279, 0 },
		{ 2758, 4700, 0 },
		{ 2228, 905, 414 },
		{ 3079, 1949, 0 },
		{ 2391, 930, 414 },
		{ 2298, 915, 414 },
		{ 2428, 923, 414 },
		{ 2365, 946, 414 },
		{ 2365, 947, 414 },
		{ 2312, 956, 414 },
		{ 2391, 974, 414 },
		{ 2391, 975, 414 },
		{ 2416, 963, 414 },
		{ 2363, 960, 414 },
		{ 2391, 978, 414 },
		{ 2339, 983, 414 },
		{ 2454, 469, 414 },
		{ 2454, 471, 414 },
		{ 2423, 993, 414 },
		{ 2423, 994, 414 },
		{ 2391, 1009, 414 },
		{ 2298, 994, 414 },
		{ 2400, 1002, 414 },
		{ 2339, 1053, 414 },
		{ 2380, 1051, 414 },
		{ 2315, 1403, 0 },
		{ 2287, 0, 0 },
		{ 2990, 2558, 412 },
		{ 2317, 1404, 0 },
		{ 3061, 2904, 0 },
		{ 0, 0, 379 },
		{ 2391, 1051, 414 },
		{ 2929, 2766, 0 },
		{ 2454, 484, 414 },
		{ 2312, 1046, 414 },
		{ 2365, 1039, 414 },
		{ 2454, 578, 414 },
		{ 2395, 1083, 414 },
		{ 2228, 1066, 414 },
		{ 2310, 1086, 414 },
		{ 2454, 8, 414 },
		{ 2365, 1109, 414 },
		{ 2395, 1121, 414 },
		{ 2454, 123, 414 },
		{ 2365, 1111, 414 },
		{ 3077, 1826, 0 },
		{ 3036, 2212, 0 },
		{ 2423, 1113, 414 },
		{ 2228, 1117, 414 },
		{ 2416, 1116, 414 },
		{ 2228, 1132, 414 },
		{ 2365, 1142, 414 },
		{ 2228, 1151, 414 },
		{ 2228, 1141, 414 },
		{ 2315, 0, 0 },
		{ 2990, 2592, 410 },
		{ 2317, 0, 0 },
		{ 2990, 2626, 411 },
		{ 0, 0, 415 },
		{ 2416, 1147, 414 },
		{ 2345, 4824, 0 },
		{ 3072, 2062, 0 },
		{ 2339, 1166, 414 },
		{ 2454, 125, 414 },
		{ 3042, 1884, 0 },
		{ 2477, 6, 414 },
		{ 2423, 1186, 414 },
		{ 2339, 1205, 414 },
		{ 2365, 1187, 414 },
		{ 3036, 1930, 0 },
		{ 2454, 128, 414 },
		{ 2363, 1186, 414 },
		{ 3079, 1988, 0 },
		{ 2395, 1200, 414 },
		{ 3033, 2313, 0 },
		{ 3081, 2049, 0 },
		{ 3033, 2315, 0 },
		{ 2400, 1195, 414 },
		{ 2416, 1219, 414 },
		{ 2228, 1237, 414 },
		{ 2391, 1234, 414 },
		{ 2391, 1235, 414 },
		{ 2454, 130, 414 },
		{ 2395, 1261, 414 },
		{ 2454, 235, 414 },
		{ 3074, 2168, 0 },
		{ 2982, 2631, 0 },
		{ 2365, 1252, 414 },
		{ 3042, 1890, 0 },
		{ 3077, 1691, 0 },
		{ 3082, 4445, 0 },
		{ 3036, 4866, 387 },
		{ 2449, 1260, 414 },
		{ 2365, 1254, 414 },
		{ 2395, 1270, 414 },
		{ 3079, 2164, 0 },
		{ 3074, 2448, 0 },
		{ 2395, 1267, 414 },
		{ 2929, 2761, 0 },
		{ 2400, 1262, 414 },
		{ 2982, 2690, 0 },
		{ 3064, 2771, 0 },
		{ 2982, 2692, 0 },
		{ 2228, 1258, 414 },
		{ 2395, 1272, 414 },
		{ 2228, 1262, 414 },
		{ 2454, 237, 414 },
		{ 2454, 239, 414 },
		{ 3081, 1911, 0 },
		{ 2428, 1270, 414 },
		{ 3079, 1959, 0 },
		{ 3025, 3631, 0 },
		{ 2982, 2492, 0 },
		{ 3066, 2356, 0 },
		{ 2395, 1276, 414 },
		{ 3077, 1901, 0 },
		{ 3075, 2961, 0 },
		{ 2477, 121, 414 },
		{ 2400, 1271, 414 },
		{ 2400, 1272, 414 },
		{ 2228, 1284, 414 },
		{ 2844, 2117, 0 },
		{ 3081, 2047, 0 },
		{ 2428, 1275, 414 },
		{ 2411, 4759, 0 },
		{ 2428, 1276, 414 },
		{ 3077, 1923, 0 },
		{ 3064, 2812, 0 },
		{ 3077, 1925, 0 },
		{ 2391, 1286, 414 },
		{ 2428, 1278, 414 },
		{ 2228, 1288, 414 },
		{ 3079, 1761, 0 },
		{ 3036, 2216, 0 },
		{ 3064, 2826, 0 },
		{ 2228, 1285, 414 },
		{ 2929, 2765, 0 },
		{ 3029, 3304, 0 },
		{ 3077, 1936, 0 },
		{ 2982, 2671, 0 },
		{ 2228, 1280, 414 },
		{ 3075, 3005, 0 },
		{ 3077, 1947, 0 },
		{ 3082, 4373, 0 },
		{ 0, 0, 405 },
		{ 2416, 1278, 414 },
		{ 2428, 1283, 414 },
		{ 2454, 241, 414 },
		{ 3069, 1655, 0 },
		{ 3079, 2153, 0 },
		{ 2417, 1292, 414 },
		{ 3036, 1935, 0 },
		{ 2454, 243, 414 },
		{ 2439, 4698, 0 },
		{ 2440, 4728, 0 },
		{ 2441, 4720, 0 },
		{ 2228, 1283, 414 },
		{ 2228, 1295, 414 },
		{ 2454, 245, 414 },
		{ 3075, 2967, 0 },
		{ 2929, 2741, 0 },
		{ 3042, 2010, 0 },
		{ 3061, 2898, 0 },
		{ 2228, 1285, 414 },
		{ 2450, 4758, 0 },
		{ 3042, 2012, 0 },
		{ 3033, 2243, 0 },
		{ 3077, 1813, 0 },
		{ 2228, 1291, 414 },
		{ 3077, 1840, 0 },
		{ 3042, 2020, 0 },
		{ 2454, 349, 414 },
		{ 2454, 351, 414 },
		{ 3036, 2329, 0 },
		{ 3072, 2431, 0 },
		{ 3066, 2370, 0 },
		{ 2454, 355, 414 },
		{ 3081, 2046, 0 },
		{ 3036, 1941, 0 },
		{ 3077, 1830, 0 },
		{ 3061, 2571, 0 },
		{ 3077, 1878, 0 },
		{ 2454, 357, 414 },
		{ 2454, 370, 414 },
		{ 3076, 1928, 0 },
		{ 3081, 2058, 0 },
		{ 2929, 2739, 0 },
		{ 3072, 2455, 0 },
		{ 3069, 1599, 0 },
		{ 2454, 1549, 414 },
		{ 3079, 1882, 0 },
		{ 2480, 4836, 0 },
		{ 3064, 2784, 0 },
		{ 3082, 4437, 0 },
		{ 2477, 576, 414 },
		{ 3042, 1966, 0 },
		{ 3082, 4449, 0 },
		{ 3036, 2331, 0 },
		{ 3079, 1980, 0 },
		{ 3064, 2796, 0 },
		{ 3075, 2969, 0 },
		{ 2490, 4740, 0 },
		{ 3079, 1759, 0 },
		{ 3079, 2207, 0 },
		{ 3081, 2068, 0 },
		{ 3081, 2071, 0 },
		{ 3064, 2802, 0 },
		{ 3081, 2073, 0 },
		{ 3036, 1939, 0 },
		{ 3042, 1886, 0 },
		{ 3042, 1977, 0 },
		{ 3033, 2309, 0 },
		{ 2502, 4757, 0 },
		{ 3064, 2810, 0 },
		{ 3042, 1978, 0 },
		{ 3075, 2989, 0 },
		{ 3076, 2932, 0 },
		{ 2507, 811, 414 },
		{ 3064, 2813, 0 },
		{ 2844, 2121, 0 },
		{ 3082, 4581, 0 },
		{ 3042, 1981, 0 },
		{ 3036, 4785, 385 },
		{ 3042, 1892, 0 },
		{ 3082, 4595, 0 },
		{ 3036, 4812, 394 },
		{ 3079, 2152, 0 },
		{ 2844, 2125, 0 },
		{ 3033, 2325, 0 },
		{ 3077, 1932, 0 },
		{ 3074, 2438, 0 },
		{ 3075, 3011, 0 },
		{ 2929, 2744, 0 },
		{ 2885, 2727, 0 },
		{ 3079, 2155, 0 },
		{ 3064, 2833, 0 },
		{ 2844, 2137, 0 },
		{ 3064, 2836, 0 },
		{ 3081, 2084, 0 },
		{ 2982, 2565, 0 },
		{ 3046, 1462, 0 },
		{ 3069, 1536, 0 },
		{ 3042, 1894, 0 },
		{ 3033, 2236, 0 },
		{ 2844, 2120, 0 },
		{ 3033, 2238, 0 },
		{ 3064, 2848, 0 },
		{ 3082, 4371, 0 },
		{ 3036, 4781, 408 },
		{ 3033, 2239, 0 },
		{ 3077, 1940, 0 },
		{ 0, 0, 420 },
		{ 3042, 1992, 0 },
		{ 2982, 2664, 0 },
		{ 3036, 4806, 393 },
		{ 3075, 2978, 0 },
		{ 3064, 2856, 0 },
		{ 2982, 2665, 0 },
		{ 2982, 2666, 0 },
		{ 2982, 2667, 0 },
		{ 3081, 2096, 0 },
		{ 2929, 2762, 0 },
		{ 2547, 4751, 0 },
		{ 2602, 2951, 0 },
		{ 3064, 2871, 0 },
		{ 3077, 1941, 0 },
		{ 3064, 2873, 0 },
		{ 3079, 2171, 0 },
		{ 2539, 1376, 0 },
		{ 2554, 4752, 0 },
		{ 2844, 2129, 0 },
		{ 3076, 2923, 0 },
		{ 3077, 1946, 0 },
		{ 3081, 2105, 0 },
		{ 3061, 2903, 0 },
		{ 2560, 4776, 0 },
		{ 3064, 2880, 0 },
		{ 2982, 2680, 0 },
		{ 2563, 4665, 0 },
		{ 0, 1409, 0 },
		{ 3072, 2414, 0 },
		{ 3081, 2037, 0 },
		{ 3077, 1678, 0 },
		{ 3079, 2184, 0 },
		{ 3072, 2426, 0 },
		{ 3064, 2891, 0 },
		{ 3042, 1999, 0 },
		{ 3036, 2758, 0 },
		{ 3075, 2965, 0 },
		{ 2574, 4727, 0 },
		{ 3071, 2709, 0 },
		{ 2576, 4731, 0 },
		{ 2602, 2954, 0 },
		{ 3064, 2773, 0 },
		{ 3042, 1896, 0 },
		{ 3072, 2433, 0 },
		{ 3081, 2044, 0 },
		{ 3042, 2002, 0 },
		{ 2982, 2486, 0 },
		{ 2584, 4742, 0 },
		{ 3079, 1968, 0 },
		{ 3081, 2048, 0 },
		{ 3066, 2378, 0 },
		{ 3076, 2708, 0 },
		{ 3064, 2785, 0 },
		{ 3082, 4511, 0 },
		{ 3075, 2982, 0 },
		{ 3079, 2194, 0 },
		{ 3033, 2281, 0 },
		{ 3064, 2792, 0 },
		{ 3033, 2282, 0 },
		{ 2844, 2128, 0 },
		{ 3069, 1602, 0 },
		{ 2602, 2955, 0 },
		{ 3061, 2573, 0 },
		{ 2600, 4708, 0 },
		{ 3061, 2567, 0 },
		{ 3075, 2997, 0 },
		{ 3082, 4375, 0 },
		{ 3077, 1686, 0 },
		{ 3079, 2199, 0 },
		{ 2982, 2615, 0 },
		{ 2607, 4693, 0 },
		{ 3033, 2289, 0 },
		{ 3066, 2006, 0 },
		{ 2844, 2115, 0 },
		{ 3075, 3007, 0 },
		{ 2982, 2620, 0 },
		{ 3075, 3010, 0 },
		{ 3082, 4513, 0 },
		{ 3036, 4823, 406 },
		{ 3077, 1688, 0 },
		{ 3081, 2053, 0 },
		{ 3082, 4554, 0 },
		{ 3082, 4579, 0 },
		{ 3077, 1689, 0 },
		{ 3081, 2057, 0 },
		{ 2929, 2758, 0 },
		{ 2982, 2632, 0 },
		{ 3064, 2817, 0 },
		{ 3082, 4363, 0 },
		{ 3064, 2818, 0 },
		{ 0, 2956, 0 },
		{ 3036, 4854, 392 },
		{ 3075, 2960, 0 },
		{ 3077, 1690, 0 },
		{ 2844, 2122, 0 },
		{ 3079, 1974, 0 },
		{ 2885, 2730, 0 },
		{ 3079, 2217, 0 },
		{ 3064, 2824, 0 },
		{ 3077, 1692, 0 },
		{ 3042, 2013, 0 },
		{ 3042, 2014, 0 },
		{ 3036, 4897, 386 },
		{ 3079, 2145, 0 },
		{ 3042, 2015, 0 },
		{ 3036, 4760, 398 },
		{ 3036, 4771, 399 },
		{ 3042, 2016, 0 },
		{ 2982, 2675, 0 },
		{ 2929, 2749, 0 },
		{ 3072, 2420, 0 },
		{ 2844, 2133, 0 },
		{ 0, 0, 419 },
		{ 2844, 2135, 0 },
		{ 2982, 2682, 0 },
		{ 3077, 1693, 0 },
		{ 2647, 4697, 0 },
		{ 3077, 1718, 0 },
		{ 2844, 2139, 0 },
		{ 2650, 4709, 0 },
		{ 3061, 2912, 0 },
		{ 3081, 2069, 0 },
		{ 2982, 2691, 0 },
		{ 3075, 2995, 0 },
		{ 3064, 2850, 0 },
		{ 3081, 2070, 0 },
		{ 3082, 4408, 0 },
		{ 3082, 4410, 0 },
		{ 3033, 2332, 0 },
		{ 3064, 2853, 0 },
		{ 2982, 2696, 0 },
		{ 3077, 1722, 0 },
		{ 3077, 1726, 0 },
		{ 3072, 2445, 0 },
		{ 3042, 2021, 0 },
		{ 3042, 1898, 0 },
		{ 3082, 4509, 0 },
		{ 3064, 2864, 0 },
		{ 3079, 1986, 0 },
		{ 3064, 2867, 0 },
		{ 3075, 3018, 0 },
		{ 3079, 2165, 0 },
		{ 3077, 1740, 0 },
		{ 3042, 2027, 0 },
		{ 3082, 4583, 0 },
		{ 3081, 939, 389 },
		{ 3036, 4882, 401 },
		{ 2885, 2731, 0 },
		{ 3081, 2080, 0 },
		{ 3077, 1753, 0 },
		{ 3071, 2716, 0 },
		{ 3071, 2719, 0 },
		{ 2982, 2549, 0 },
		{ 2682, 4670, 0 },
		{ 3076, 2940, 0 },
		{ 3036, 4774, 397 },
		{ 3081, 2082, 0 },
		{ 2844, 2130, 0 },
		{ 3072, 2461, 0 },
		{ 3077, 1773, 0 },
		{ 3033, 2258, 0 },
		{ 2982, 2561, 0 },
		{ 2690, 4741, 0 },
		{ 3036, 4792, 388 },
		{ 3082, 4443, 0 },
		{ 2692, 4747, 0 },
		{ 2696, 1455, 0 },
		{ 2694, 4751, 0 },
		{ 2695, 4753, 0 },
		{ 3077, 1793, 0 },
		{ 3074, 2446, 0 },
		{ 3081, 2087, 0 },
		{ 3075, 2979, 0 },
		{ 3064, 2895, 0 },
		{ 3082, 4482, 0 },
		{ 3079, 2180, 0 },
		{ 3042, 1953, 0 },
		{ 3079, 2183, 0 },
		{ 3082, 4515, 0 },
		{ 3036, 4830, 403 },
		{ 3082, 4517, 0 },
		{ 3082, 4519, 0 },
		{ 3082, 4523, 0 },
		{ 3082, 4525, 0 },
		{ 0, 1456, 0 },
		{ 2982, 2622, 0 },
		{ 2982, 2625, 0 },
		{ 3077, 1799, 0 },
		{ 3081, 2091, 0 },
		{ 3036, 4847, 409 },
		{ 3081, 2092, 0 },
		{ 3082, 4587, 0 },
		{ 3033, 2275, 0 },
		{ 0, 0, 422 },
		{ 0, 0, 421 },
		{ 3036, 4856, 390 },
		{ 0, 0, 417 },
		{ 0, 0, 418 },
		{ 3082, 4591, 0 },
		{ 3072, 2422, 0 },
		{ 2844, 2119, 0 },
		{ 3079, 2191, 0 },
		{ 3075, 2999, 0 },
		{ 3082, 4365, 0 },
		{ 3036, 4868, 384 },
		{ 2724, 4776, 0 },
		{ 3036, 4873, 391 },
		{ 3064, 2789, 0 },
		{ 3077, 1801, 0 },
		{ 3081, 2095, 0 },
		{ 3077, 1803, 0 },
		{ 3036, 4892, 404 },
		{ 3036, 2214, 0 },
		{ 3082, 4377, 0 },
		{ 3082, 4379, 0 },
		{ 3082, 4381, 0 },
		{ 3079, 2196, 0 },
		{ 3077, 1807, 0 },
		{ 3036, 4764, 395 },
		{ 3036, 4766, 396 },
		{ 3036, 4768, 400 },
		{ 3081, 2098, 0 },
		{ 3064, 2799, 0 },
		{ 3082, 4439, 0 },
		{ 3081, 2100, 0 },
		{ 3036, 4778, 402 },
		{ 3075, 3013, 0 },
		{ 3077, 1809, 0 },
		{ 2982, 2676, 0 },
		{ 3079, 2202, 0 },
		{ 3033, 2294, 0 },
		{ 3042, 1967, 0 },
		{ 3082, 4480, 0 },
		{ 3036, 4798, 407 },
		{ 2758, 4674, 423 },
		{ 2751, 0, 381 },
		{ 0, 0, 382 },
		{ -2749, 20, 377 },
		{ -2750, 4718, 0 },
		{ 3036, 4695, 0 },
		{ 2758, 4683, 0 },
		{ 0, 0, 378 },
		{ 2758, 4677, 0 },
		{ -2755, 4898, 0 },
		{ -2756, 4710, 0 },
		{ 2759, 0, 379 },
		{ 0, 4685, 0 },
		{ 3036, 4816, 0 },
		{ 0, 0, 380 },
		{ 3029, 3277, 148 },
		{ 0, 0, 148 },
		{ 0, 0, 149 },
		{ 3042, 1971, 0 },
		{ 3064, 2809, 0 },
		{ 3081, 2036, 0 },
		{ 2768, 4735, 0 },
		{ 3074, 2436, 0 },
		{ 3069, 1639, 0 },
		{ 3033, 2302, 0 },
		{ 3076, 2946, 0 },
		{ 3077, 1820, 0 },
		{ 2982, 2693, 0 },
		{ 3079, 2214, 0 },
		{ 3033, 2306, 0 },
		{ 3042, 1974, 0 },
		{ 3082, 4585, 0 },
		{ 0, 0, 146 },
		{ 2916, 4493, 167 },
		{ 0, 0, 167 },
		{ 3077, 1825, 0 },
		{ 2783, 4759, 0 },
		{ 3064, 2495, 0 },
		{ 3075, 2975, 0 },
		{ 3076, 2935, 0 },
		{ 3071, 2706, 0 },
		{ 2788, 4767, 0 },
		{ 3036, 2325, 0 },
		{ 3064, 2825, 0 },
		{ 3033, 2312, 0 },
		{ 3064, 2827, 0 },
		{ 3081, 2043, 0 },
		{ 3075, 2984, 0 },
		{ 3077, 1831, 0 },
		{ 2982, 2488, 0 },
		{ 3079, 2220, 0 },
		{ 3033, 2317, 0 },
		{ 2799, 4795, 0 },
		{ 3036, 2756, 0 },
		{ 3064, 2835, 0 },
		{ 2929, 2756, 0 },
		{ 3079, 2142, 0 },
		{ 3081, 2045, 0 },
		{ 3064, 2839, 0 },
		{ 2806, 4654, 0 },
		{ 3081, 1923, 0 },
		{ 3064, 2841, 0 },
		{ 3061, 2905, 0 },
		{ 3069, 1657, 0 },
		{ 3076, 2924, 0 },
		{ 3064, 2843, 0 },
		{ 2813, 4680, 0 },
		{ 3074, 2434, 0 },
		{ 3069, 1532, 0 },
		{ 3033, 2327, 0 },
		{ 3076, 2933, 0 },
		{ 3077, 1856, 0 },
		{ 2982, 2560, 0 },
		{ 3079, 2148, 0 },
		{ 3033, 2330, 0 },
		{ 3082, 4521, 0 },
		{ 0, 0, 165 },
		{ 2824, 0, 1 },
		{ -2824, 1372, 256 },
		{ 3064, 2752, 262 },
		{ 0, 0, 262 },
		{ 3042, 1988, 0 },
		{ 3033, 2234, 0 },
		{ 3064, 2858, 0 },
		{ 3061, 2910, 0 },
		{ 3081, 2054, 0 },
		{ 0, 0, 261 },
		{ 2834, 4741, 0 },
		{ 3066, 1925, 0 },
		{ 3075, 2963, 0 },
		{ 2863, 2477, 0 },
		{ 3064, 2865, 0 },
		{ 2929, 2759, 0 },
		{ 2982, 2624, 0 },
		{ 3072, 2447, 0 },
		{ 3064, 2869, 0 },
		{ 2843, 4724, 0 },
		{ 3079, 1994, 0 },
		{ 0, 2131, 0 },
		{ 3077, 1876, 0 },
		{ 2982, 2630, 0 },
		{ 3079, 2160, 0 },
		{ 3033, 2241, 0 },
		{ 3042, 1993, 0 },
		{ 3082, 4383, 0 },
		{ 0, 0, 260 },
		{ 0, 4305, 170 },
		{ 0, 0, 170 },
		{ 3079, 2162, 0 },
		{ 3069, 1598, 0 },
		{ 3033, 2246, 0 },
		{ 3061, 2914, 0 },
		{ 2859, 4760, 0 },
		{ 3076, 2627, 0 },
		{ 3071, 2708, 0 },
		{ 3064, 2885, 0 },
		{ 3076, 2941, 0 },
		{ 0, 2478, 0 },
		{ 2982, 2669, 0 },
		{ 3033, 2248, 0 },
		{ 2885, 2722, 0 },
		{ 3082, 4507, 0 },
		{ 0, 0, 168 },
		{ 2916, 4507, 164 },
		{ 0, 0, 164 },
		{ 3077, 1898, 0 },
		{ 2873, 4764, 0 },
		{ 3077, 1854, 0 },
		{ 3071, 2717, 0 },
		{ 3064, 2894, 0 },
		{ 2877, 4787, 0 },
		{ 3036, 2754, 0 },
		{ 3064, 2896, 0 },
		{ 2885, 2728, 0 },
		{ 2982, 2674, 0 },
		{ 3033, 2254, 0 },
		{ 3033, 2255, 0 },
		{ 2982, 2677, 0 },
		{ 3033, 2256, 0 },
		{ 0, 2724, 0 },
		{ 2887, 4656, 0 },
		{ 3079, 1954, 0 },
		{ 2929, 2745, 0 },
		{ 2890, 4670, 0 },
		{ 3064, 2493, 0 },
		{ 3075, 3014, 0 },
		{ 3076, 2922, 0 },
		{ 3071, 2715, 0 },
		{ 2895, 4676, 0 },
		{ 3036, 2323, 0 },
		{ 3064, 2786, 0 },
		{ 3033, 2259, 0 },
		{ 3064, 2788, 0 },
		{ 3081, 2065, 0 },
		{ 3075, 3026, 0 },
		{ 3077, 1905, 0 },
		{ 2982, 2689, 0 },
		{ 3079, 2169, 0 },
		{ 3033, 2263, 0 },
		{ 2906, 4695, 0 },
		{ 3074, 2452, 0 },
		{ 3069, 1601, 0 },
		{ 3033, 2265, 0 },
		{ 3076, 2943, 0 },
		{ 3077, 1907, 0 },
		{ 2982, 2697, 0 },
		{ 3079, 2172, 0 },
		{ 3033, 2268, 0 },
		{ 3082, 4441, 0 },
		{ 0, 0, 159 },
		{ 0, 4371, 158 },
		{ 0, 0, 158 },
		{ 3077, 1908, 0 },
		{ 2920, 4703, 0 },
		{ 3077, 1858, 0 },
		{ 3071, 2707, 0 },
		{ 3064, 2806, 0 },
		{ 2924, 4722, 0 },
		{ 3064, 2499, 0 },
		{ 3033, 2271, 0 },
		{ 3061, 2906, 0 },
		{ 2928, 4718, 0 },
		{ 3079, 1956, 0 },
		{ 0, 2747, 0 },
		{ 2931, 4733, 0 },
		{ 3064, 2489, 0 },
		{ 3075, 2983, 0 },
		{ 3076, 2938, 0 },
		{ 3071, 2712, 0 },
		{ 2936, 4738, 0 },
		{ 3036, 2327, 0 },
		{ 3064, 2814, 0 },
		{ 3033, 2274, 0 },
		{ 3064, 2816, 0 },
		{ 3081, 2072, 0 },
		{ 3075, 2991, 0 },
		{ 3077, 1914, 0 },
		{ 2982, 2546, 0 },
		{ 3079, 2178, 0 },
		{ 3033, 2278, 0 },
		{ 2947, 4752, 0 },
		{ 3074, 2450, 0 },
		{ 3069, 1618, 0 },
		{ 3033, 2280, 0 },
		{ 3076, 2928, 0 },
		{ 3077, 1917, 0 },
		{ 2982, 2555, 0 },
		{ 3079, 2181, 0 },
		{ 3033, 2283, 0 },
		{ 3082, 4599, 0 },
		{ 0, 0, 156 },
		{ 0, 3933, 161 },
		{ 0, 0, 161 },
		{ 3033, 2284, 0 },
		{ 3042, 2008, 0 },
		{ 3077, 1919, 0 },
		{ 3064, 2832, 0 },
		{ 3075, 3012, 0 },
		{ 3061, 2916, 0 },
		{ 2966, 4783, 0 },
		{ 3064, 2487, 0 },
		{ 3046, 1492, 0 },
		{ 3075, 3017, 0 },
		{ 3072, 2463, 0 },
		{ 3069, 1620, 0 },
		{ 3075, 3020, 0 },
		{ 3077, 1924, 0 },
		{ 2982, 2619, 0 },
		{ 3079, 2187, 0 },
		{ 3033, 2291, 0 },
		{ 2977, 4663, 0 },
		{ 3074, 2440, 0 },
		{ 3069, 1621, 0 },
		{ 3033, 2293, 0 },
		{ 3076, 2930, 0 },
		{ 3077, 1927, 0 },
		{ 0, 2628, 0 },
		{ 3079, 2190, 0 },
		{ 3033, 2296, 0 },
		{ 3044, 4622, 0 },
		{ 0, 0, 160 },
		{ 3064, 2849, 423 },
		{ 3081, 1517, 25 },
		{ 2996, 0, 423 },
		{ 2227, 2648, 27 },
		{ 0, 0, 28 },
		{ 0, 0, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 3033, 2300, 0 },
		{ 3081, 661, 0 },
		{ 0, 0, 26 },
		{ 3061, 2920, 0 },
		{ 0, 0, 21 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 0, 3762, 37 },
		{ 0, 3494, 37 },
		{ 0, 0, 36 },
		{ 0, 0, 37 },
		{ 3025, 3634, 0 },
		{ 3037, 4278, 0 },
		{ 3029, 3279, 0 },
		{ 0, 0, 35 },
		{ 3032, 3346, 0 },
		{ 0, 3185, 0 },
		{ 2990, 1630, 0 },
		{ 0, 0, 34 },
		{ 3064, 2765, 47 },
		{ 0, 0, 47 },
		{ 3029, 3284, 47 },
		{ 3064, 2859, 47 },
		{ 0, 0, 50 },
		{ 3064, 2860, 0 },
		{ 3033, 2305, 0 },
		{ 3032, 3355, 0 },
		{ 3077, 1938, 0 },
		{ 3033, 2307, 0 },
		{ 3061, 2908, 0 },
		{ 0, 3599, 0 },
		{ 3069, 1656, 0 },
		{ 3079, 2200, 0 },
		{ 0, 0, 46 },
		{ 0, 3296, 0 },
		{ 3081, 2093, 0 },
		{ 3066, 2360, 0 },
		{ 0, 3364, 0 },
		{ 0, 2311, 0 },
		{ 3064, 2870, 0 },
		{ 0, 0, 48 },
		{ 0, 5, 51 },
		{ 0, 4245, 0 },
		{ 0, 0, 49 },
		{ 3072, 2448, 0 },
		{ 3075, 2993, 0 },
		{ 3042, 2024, 0 },
		{ 0, 2025, 0 },
		{ 3044, 4626, 0 },
		{ 0, 4627, 0 },
		{ 3064, 2874, 0 },
		{ 0, 1495, 0 },
		{ 3075, 2998, 0 },
		{ 3072, 2452, 0 },
		{ 3069, 1658, 0 },
		{ 3075, 3001, 0 },
		{ 3077, 1943, 0 },
		{ 3079, 2209, 0 },
		{ 3081, 2099, 0 },
		{ 3075, 2052, 0 },
		{ 3064, 2882, 0 },
		{ 3079, 2213, 0 },
		{ 3076, 2929, 0 },
		{ 3075, 3009, 0 },
		{ 3081, 2101, 0 },
		{ 3076, 2931, 0 },
		{ 0, 2907, 0 },
		{ 3064, 2790, 0 },
		{ 3069, 1673, 0 },
		{ 0, 2887, 0 },
		{ 3075, 3016, 0 },
		{ 0, 2381, 0 },
		{ 3081, 2103, 0 },
		{ 3076, 2939, 0 },
		{ 0, 1675, 0 },
		{ 3082, 4629, 0 },
		{ 0, 2714, 0 },
		{ 0, 2464, 0 },
		{ 0, 0, 43 },
		{ 3036, 2730, 0 },
		{ 0, 3024, 0 },
		{ 0, 2944, 0 },
		{ 0, 1948, 0 },
		{ 3082, 4631, 0 },
		{ 0, 2219, 0 },
		{ 0, 0, 44 },
		{ 0, 2106, 0 },
		{ 3044, 4633, 0 },
		{ 0, 0, 45 }
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
		0
	};
	yybackup = backup;
}
