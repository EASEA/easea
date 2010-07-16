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
#line 683 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    //DEBUG_PRT_PRT("found new symbol %s",pSym->Object->sName);
    fprintf(fpOutputFile," ar & %s;\n",pSym->Object->sName);
  }
 
#line 962 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 693 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 977 "EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 702 "EaseaLex.l"

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
 
#line 1000 "EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 719 "EaseaLex.l"

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
 
#line 1023 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 737 "EaseaLex.l"

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
 
#line 1055 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 764 "EaseaLex.l"

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
 
#line 1076 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 780 "EaseaLex.l"

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
 
#line 1098 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 797 "EaseaLex.l"
       
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
 
#line 1126 "EaseaLex.cpp"
		}
		break;
	case 68:
		{
#line 819 "EaseaLex.l"

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
 
#line 1148 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 835 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 1163 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 844 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 1175 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 852 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 1187 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 859 "EaseaLex.l"

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
 
#line 1218 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 884 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 1231 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 891 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 1245 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 900 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_INITIALISER;   
 
#line 1256 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 906 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 1268 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 913 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 1280 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 919 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 1292 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 925 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 1304 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 931 "EaseaLex.l"
      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 1317 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 938 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 1330 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 945 "EaseaLex.l"
 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 1344 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 954 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 1355 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 959 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 1369 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 968 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 1383 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 977 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 1397 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 987 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 1410 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 995 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 1419 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 999 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 1428 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 1003 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 1437 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 1007 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 1446 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 1011 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 1456 "EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 1016 "EaseaLex.l"

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

#line 1475 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 1029 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 1482 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 1030 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 1489 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 1031 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 1496 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 1032 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 1503 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 1033 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 1510 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 1034 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1517 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 1035 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1524 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 1036 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1531 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 1037 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1538 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 1038 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1545 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 1039 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",nELITE);
  ////DEBUG_PRT_PRT("elitism is %d, elite size is %d",bELITISM, nELITE);
 
#line 1555 "EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 1044 "EaseaLex.l"

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
 
#line 1574 "EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 1057 "EaseaLex.l"

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
 
#line 1593 "EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 1070 "EaseaLex.l"

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
 
#line 1612 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 1083 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1622 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 1087 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1629 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 1088 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1636 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 1089 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1643 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 1090 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1650 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 1091 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1657 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 1092 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1664 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 1093 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1671 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 1094 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1678 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 1095 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1685 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 1096 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1692 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 1097 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1699 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 1098 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1706 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 1100 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1713 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1101 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1720 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1102 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1727 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1103 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1734 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1104 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1741 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1106 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1755 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1114 "EaseaLex.l"

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
 
#line 1775 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1128 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1789 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1136 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1803 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1145 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1817 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1154 "EaseaLex.l"

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

#line 1880 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1211 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded with no warning.\n");
  printf ("Have a nice compile time.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 1897 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1223 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1904 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1229 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1916 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1235 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1929 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1242 "EaseaLex.l"

#line 1936 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1243 "EaseaLex.l"
lineCounter++;
#line 1943 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1245 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1955 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1251 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1968 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1259 "EaseaLex.l"

#line 1975 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1260 "EaseaLex.l"

  lineCounter++;
 
#line 1984 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1264 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1996 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1270 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2010 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1278 "EaseaLex.l"

#line 2017 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1279 "EaseaLex.l"

  lineCounter++;
 
#line 2026 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1283 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){");
  bFunction=1; bInitFunction=1;
      
  BEGIN COPY;
 
#line 2038 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1289 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2052 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1297 "EaseaLex.l"

#line 2059 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1302 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){");
  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2070 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1308 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2084 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1316 "EaseaLex.l"

#line 2091 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1319 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2107 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1330 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2123 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1340 "EaseaLex.l"

#line 2130 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1343 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    bBeginGeneration = 0;
    bBeginGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2146 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1354 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2160 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1363 "EaseaLex.l"

#line 2167 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1365 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2183 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1377 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2199 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1387 "EaseaLex.l"

#line 2206 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1391 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2221 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1401 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2236 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1410 "EaseaLex.l"

#line 2243 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1413 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2256 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1420 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2270 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1428 "EaseaLex.l"

#line 2277 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1432 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2285 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1434 "EaseaLex.l"

#line 2292 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1440 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2299 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1441 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2306 "EaseaLex.cpp"
		}
		break;
	case 171:
	case 172:
		{
#line 1444 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2317 "EaseaLex.cpp"
		}
		break;
	case 173:
	case 174:
		{
#line 1449 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2326 "EaseaLex.cpp"
		}
		break;
	case 175:
	case 176:
		{
#line 1452 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2335 "EaseaLex.cpp"
		}
		break;
	case 177:
	case 178:
		{
#line 1455 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2352 "EaseaLex.cpp"
		}
		break;
	case 179:
	case 180:
		{
#line 1466 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2366 "EaseaLex.cpp"
		}
		break;
	case 181:
	case 182:
		{
#line 1474 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2375 "EaseaLex.cpp"
		}
		break;
	case 183:
	case 184:
		{
#line 1477 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2384 "EaseaLex.cpp"
		}
		break;
	case 185:
	case 186:
		{
#line 1480 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2393 "EaseaLex.cpp"
		}
		break;
	case 187:
	case 188:
		{
#line 1483 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2402 "EaseaLex.cpp"
		}
		break;
	case 189:
	case 190:
		{
#line 1486 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2411 "EaseaLex.cpp"
		}
		break;
	case 191:
	case 192:
		{
#line 1490 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2423 "EaseaLex.cpp"
		}
		break;
	case 193:
		{
#line 1496 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2430 "EaseaLex.cpp"
		}
		break;
	case 194:
		{
#line 1497 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2437 "EaseaLex.cpp"
		}
		break;
	case 195:
		{
#line 1498 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2444 "EaseaLex.cpp"
		}
		break;
	case 196:
		{
#line 1499 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2454 "EaseaLex.cpp"
		}
		break;
	case 197:
		{
#line 1504 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2461 "EaseaLex.cpp"
		}
		break;
	case 198:
		{
#line 1505 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2468 "EaseaLex.cpp"
		}
		break;
	case 199:
		{
#line 1506 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2475 "EaseaLex.cpp"
		}
		break;
	case 200:
		{
#line 1507 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2482 "EaseaLex.cpp"
		}
		break;
	case 201:
		{
#line 1508 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2489 "EaseaLex.cpp"
		}
		break;
	case 202:
		{
#line 1509 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2496 "EaseaLex.cpp"
		}
		break;
	case 203:
		{
#line 1510 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2503 "EaseaLex.cpp"
		}
		break;
	case 204:
		{
#line 1511 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2510 "EaseaLex.cpp"
		}
		break;
	case 205:
		{
#line 1512 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2518 "EaseaLex.cpp"
		}
		break;
	case 206:
		{
#line 1514 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2526 "EaseaLex.cpp"
		}
		break;
	case 207:
		{
#line 1516 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2534 "EaseaLex.cpp"
		}
		break;
	case 208:
		{
#line 1518 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2544 "EaseaLex.cpp"
		}
		break;
	case 209:
		{
#line 1522 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2551 "EaseaLex.cpp"
		}
		break;
	case 210:
		{
#line 1523 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2558 "EaseaLex.cpp"
		}
		break;
	case 211:
		{
#line 1524 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2569 "EaseaLex.cpp"
		}
		break;
	case 212:
		{
#line 1529 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2576 "EaseaLex.cpp"
		}
		break;
	case 213:
		{
#line 1530 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"Individual");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2585 "EaseaLex.cpp"
		}
		break;
	case 214:
		{
#line 1533 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2597 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1539 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2606 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1542 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2618 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1548 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2629 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1553 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2645 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1563 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2652 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1566 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2661 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1569 "EaseaLex.l"
BEGIN COPY;
#line 2668 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1571 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2675 "EaseaLex.cpp"
		}
		break;
	case 223:
	case 224:
	case 225:
		{
#line 1574 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2688 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1579 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2699 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1584 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 2708 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1593 "EaseaLex.l"
;
#line 2715 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1594 "EaseaLex.l"
;
#line 2722 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1595 "EaseaLex.l"
;
#line 2729 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1596 "EaseaLex.l"
;
#line 2736 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1599 "EaseaLex.l"
 /* do nothing */ 
#line 2743 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1600 "EaseaLex.l"
 /*return '\n';*/ 
#line 2750 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1601 "EaseaLex.l"
 /*return '\n';*/ 
#line 2757 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1604 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("boolean");
  return BOOL;
#line 2766 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1607 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 2776 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1611 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
  printf("match gpnode\n");
  return GPNODE;
 
#line 2788 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1618 "EaseaLex.l"
return STATIC;
#line 2795 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1619 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 2802 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1620 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 2809 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1621 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 2816 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1622 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 2823 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1623 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 2830 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1625 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 2837 "EaseaLex.cpp"
		}
		break;
#line 1626 "EaseaLex.l"
  
#line 2842 "EaseaLex.cpp"
	case 245:
		{
#line 1627 "EaseaLex.l"
return GENOME; 
#line 2847 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1629 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 2857 "EaseaLex.cpp"
		}
		break;
	case 247:
	case 248:
	case 249:
		{
#line 1636 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 2866 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1637 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 2873 "EaseaLex.cpp"
		}
		break;
	case 251:
		{
#line 1640 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 2881 "EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1642 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 2888 "EaseaLex.cpp"
		}
		break;
	case 253:
		{
#line 1648 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 2900 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1654 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2913 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1661 "EaseaLex.l"

#line 2920 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1663 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 2931 "EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1674 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 2946 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1684 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 2957 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1690 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 2966 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1694 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 2981 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1707 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 2993 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1713 "EaseaLex.l"

#line 3000 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1714 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3013 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1721 "EaseaLex.l"

#line 3020 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1722 "EaseaLex.l"
lineCounter++;
#line 3027 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1723 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3040 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1730 "EaseaLex.l"

#line 3047 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1731 "EaseaLex.l"
lineCounter++;
#line 3054 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1733 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3067 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1740 "EaseaLex.l"

#line 3074 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1741 "EaseaLex.l"
lineCounter++;
#line 3081 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1743 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3094 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1750 "EaseaLex.l"

#line 3101 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1751 "EaseaLex.l"
lineCounter++;
#line 3108 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1757 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3115 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1758 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3122 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1759 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3129 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1760 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3136 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1761 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3143 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1762 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3150 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1763 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3157 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1765 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3166 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1768 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3179 "EaseaLex.cpp"
		}
		break;
	case 284:
	case 285:
		{
#line 1777 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3190 "EaseaLex.cpp"
		}
		break;
	case 286:
	case 287:
		{
#line 1782 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3199 "EaseaLex.cpp"
		}
		break;
	case 288:
	case 289:
		{
#line 1785 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3208 "EaseaLex.cpp"
		}
		break;
	case 290:
	case 291:
		{
#line 1788 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3220 "EaseaLex.cpp"
		}
		break;
	case 292:
	case 293:
		{
#line 1794 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3233 "EaseaLex.cpp"
		}
		break;
	case 294:
	case 295:
		{
#line 1801 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3242 "EaseaLex.cpp"
		}
		break;
	case 296:
	case 297:
		{
#line 1804 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3251 "EaseaLex.cpp"
		}
		break;
	case 298:
	case 299:
		{
#line 1807 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3260 "EaseaLex.cpp"
		}
		break;
	case 300:
	case 301:
		{
#line 1810 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3269 "EaseaLex.cpp"
		}
		break;
	case 302:
	case 303:
		{
#line 1813 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3278 "EaseaLex.cpp"
		}
		break;
	case 304:
		{
#line 1816 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3287 "EaseaLex.cpp"
		}
		break;
	case 305:
		{
#line 1819 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3297 "EaseaLex.cpp"
		}
		break;
	case 306:
		{
#line 1823 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3305 "EaseaLex.cpp"
		}
		break;
	case 307:
		{
#line 1825 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3316 "EaseaLex.cpp"
		}
		break;
	case 308:
		{
#line 1830 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3327 "EaseaLex.cpp"
		}
		break;
	case 309:
		{
#line 1835 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3335 "EaseaLex.cpp"
		}
		break;
	case 310:
		{
#line 1837 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3343 "EaseaLex.cpp"
		}
		break;
	case 311:
		{
#line 1839 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3351 "EaseaLex.cpp"
		}
		break;
	case 312:
		{
#line 1841 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3359 "EaseaLex.cpp"
		}
		break;
	case 313:
		{
#line 1843 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3367 "EaseaLex.cpp"
		}
		break;
	case 314:
		{
#line 1845 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3374 "EaseaLex.cpp"
		}
		break;
	case 315:
		{
#line 1846 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3381 "EaseaLex.cpp"
		}
		break;
	case 316:
		{
#line 1847 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3389 "EaseaLex.cpp"
		}
		break;
	case 317:
		{
#line 1849 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3397 "EaseaLex.cpp"
		}
		break;
	case 318:
		{
#line 1851 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3405 "EaseaLex.cpp"
		}
		break;
	case 319:
		{
#line 1853 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3412 "EaseaLex.cpp"
		}
		break;
	case 320:
		{
#line 1854 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3424 "EaseaLex.cpp"
		}
		break;
	case 321:
		{
#line 1860 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3433 "EaseaLex.cpp"
		}
		break;
	case 322:
		{
#line 1863 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3443 "EaseaLex.cpp"
		}
		break;
	case 323:
		{
#line 1867 "EaseaLex.l"
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
#line 3460 "EaseaLex.cpp"
		}
		break;
	case 324:
	case 325:
		{
#line 1879 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3470 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1882 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3477 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1889 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3484 "EaseaLex.cpp"
		}
		break;
	case 328:
		{
#line 1890 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3491 "EaseaLex.cpp"
		}
		break;
	case 329:
		{
#line 1891 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3498 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1892 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3505 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1893 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3512 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 1895 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3521 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 1899 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3534 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 1907 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3547 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 1916 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3560 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 1925 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3575 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 1935 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3582 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1936 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3589 "EaseaLex.cpp"
		}
		break;
	case 339:
	case 340:
		{
#line 1939 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3600 "EaseaLex.cpp"
		}
		break;
	case 341:
	case 342:
		{
#line 1944 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3609 "EaseaLex.cpp"
		}
		break;
	case 343:
	case 344:
		{
#line 1947 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3618 "EaseaLex.cpp"
		}
		break;
	case 345:
	case 346:
		{
#line 1950 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3631 "EaseaLex.cpp"
		}
		break;
	case 347:
	case 348:
		{
#line 1957 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3644 "EaseaLex.cpp"
		}
		break;
	case 349:
	case 350:
		{
#line 1964 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3653 "EaseaLex.cpp"
		}
		break;
	case 351:
		{
#line 1967 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3660 "EaseaLex.cpp"
		}
		break;
	case 352:
		{
#line 1968 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3667 "EaseaLex.cpp"
		}
		break;
	case 353:
		{
#line 1969 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3674 "EaseaLex.cpp"
		}
		break;
	case 354:
		{
#line 1970 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3684 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 1975 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3691 "EaseaLex.cpp"
		}
		break;
	case 356:
		{
#line 1976 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3698 "EaseaLex.cpp"
		}
		break;
	case 357:
		{
#line 1977 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3705 "EaseaLex.cpp"
		}
		break;
	case 358:
		{
#line 1978 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3712 "EaseaLex.cpp"
		}
		break;
	case 359:
		{
#line 1979 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3720 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 1981 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3728 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 1983 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3736 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 1985 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3744 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 1987 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 3752 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 1989 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 3760 "EaseaLex.cpp"
		}
		break;
	case 365:
		{
#line 1991 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 3768 "EaseaLex.cpp"
		}
		break;
	case 366:
		{
#line 1993 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3775 "EaseaLex.cpp"
		}
		break;
	case 367:
		{
#line 1994 "EaseaLex.l"
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
#line 3798 "EaseaLex.cpp"
		}
		break;
	case 368:
		{
#line 2011 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 3809 "EaseaLex.cpp"
		}
		break;
	case 369:
		{
#line 2016 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 3823 "EaseaLex.cpp"
		}
		break;
	case 370:
		{
#line 2024 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3830 "EaseaLex.cpp"
		}
		break;
	case 371:
		{
#line 2030 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 3840 "EaseaLex.cpp"
		}
		break;
	case 372:
		{
#line 2034 "EaseaLex.l"

#line 3847 "EaseaLex.cpp"
		}
		break;
	case 373:
		{
#line 2037 "EaseaLex.l"
;
#line 3854 "EaseaLex.cpp"
		}
		break;
	case 374:
		{
#line 2038 "EaseaLex.l"
;
#line 3861 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2039 "EaseaLex.l"
;
#line 3868 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2040 "EaseaLex.l"
;
#line 3875 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2042 "EaseaLex.l"
 /* do nothing */ 
#line 3882 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2043 "EaseaLex.l"
 /*return '\n';*/ 
#line 3889 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2044 "EaseaLex.l"
 /*return '\n';*/ 
#line 3896 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2046 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 3903 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2047 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 3910 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2048 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 3917 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2049 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 3924 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2050 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 3931 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2051 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 3938 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2052 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 3945 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2053 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 3952 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2054 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 3959 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2056 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 3966 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2057 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 3973 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2058 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 3980 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2059 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 3987 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2060 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 3994 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2062 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4005 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2067 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4012 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2069 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4023 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2074 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4030 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2077 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4037 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2078 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4044 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2079 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4051 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2080 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4058 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2081 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4065 "EaseaLex.cpp"
		}
		break;
#line 2082 "EaseaLex.l"
 
#line 4070 "EaseaLex.cpp"
	case 403:
	case 404:
	case 405:
		{
#line 2086 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4077 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2087 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4084 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2090 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4092 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2093 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4099 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2095 "EaseaLex.l"

  lineCounter++;

#line 4108 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2098 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4118 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2103 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4128 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2108 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4138 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2113 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4148 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2118 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4158 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2123 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4168 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2132 "EaseaLex.l"
return  (char)yytext[0];
#line 4175 "EaseaLex.cpp"
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
#line 2134 "EaseaLex.l"


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

#line 4363 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		171,
		-172,
		0,
		173,
		-174,
		0,
		175,
		-176,
		0,
		177,
		-178,
		0,
		183,
		-184,
		0,
		185,
		-186,
		0,
		187,
		-188,
		0,
		189,
		-190,
		0,
		181,
		-182,
		0,
		179,
		-180,
		0,
		-221,
		0,
		-227,
		0,
		288,
		-289,
		0,
		290,
		-291,
		0,
		284,
		-285,
		0,
		296,
		-297,
		0,
		298,
		-299,
		0,
		300,
		-301,
		0,
		302,
		-303,
		0,
		286,
		-287,
		0,
		349,
		-350,
		0,
		294,
		-295,
		0,
		347,
		-348,
		0,
		292,
		-293,
		0,
		341,
		-342,
		0,
		343,
		-344,
		0,
		345,
		-346,
		0,
		339,
		-340,
		0
	};
	yymatch = match;

	yytransitionmax = 4790;
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
		{ 2896, 61 },
		{ 2896, 61 },
		{ 1806, 1909 },
		{ 1447, 1447 },
		{ 67, 61 },
		{ 2292, 2270 },
		{ 2292, 2270 },
		{ 2275, 2249 },
		{ 2275, 2249 },
		{ 2412, 2412 },
		{ 2412, 2412 },
		{ 71, 3 },
		{ 72, 3 },
		{ 2165, 43 },
		{ 2166, 43 },
		{ 1928, 39 },
		{ 69, 1 },
		{ 2438, 2438 },
		{ 2438, 2438 },
		{ 67, 1 },
		{ 165, 161 },
		{ 0, 1745 },
		{ 2131, 2127 },
		{ 2896, 61 },
		{ 1280, 1279 },
		{ 2894, 61 },
		{ 1447, 1447 },
		{ 2943, 2941 },
		{ 2292, 2270 },
		{ 1301, 1300 },
		{ 2275, 2249 },
		{ 2148, 2147 },
		{ 1432, 1415 },
		{ 1433, 1415 },
		{ 71, 3 },
		{ 2898, 61 },
		{ 2165, 43 },
		{ 2124, 2123 },
		{ 1928, 39 },
		{ 2895, 61 },
		{ 2895, 61 },
		{ 2895, 61 },
		{ 2895, 61 },
		{ 2895, 61 },
		{ 2895, 61 },
		{ 2895, 61 },
		{ 2895, 61 },
		{ 2895, 61 },
		{ 2895, 61 },
		{ 70, 3 },
		{ 2897, 61 },
		{ 2164, 43 },
		{ 1522, 1516 },
		{ 1914, 39 },
		{ 2293, 2270 },
		{ 1432, 1415 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 1524, 1518 },
		{ 2892, 61 },
		{ 1434, 1415 },
		{ 1395, 1374 },
		{ 2893, 61 },
		{ 1396, 1375 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2893, 61 },
		{ 2899, 61 },
		{ 2126, 40 },
		{ 1469, 1453 },
		{ 1470, 1453 },
		{ 1383, 1361 },
		{ 1913, 40 },
		{ 2339, 2316 },
		{ 2339, 2316 },
		{ 2290, 2268 },
		{ 2290, 2268 },
		{ 2307, 2284 },
		{ 2307, 2284 },
		{ 1739, 37 },
		{ 2309, 2286 },
		{ 2309, 2286 },
		{ 2329, 2306 },
		{ 2329, 2306 },
		{ 1384, 1362 },
		{ 1385, 1363 },
		{ 1388, 1366 },
		{ 1389, 1367 },
		{ 1390, 1368 },
		{ 1391, 1369 },
		{ 1392, 1370 },
		{ 2126, 40 },
		{ 1469, 1453 },
		{ 1916, 40 },
		{ 1394, 1373 },
		{ 1397, 1376 },
		{ 2339, 2316 },
		{ 1398, 1377 },
		{ 2290, 2268 },
		{ 1399, 1378 },
		{ 2307, 2284 },
		{ 1400, 1379 },
		{ 1739, 37 },
		{ 2309, 2286 },
		{ 1401, 1380 },
		{ 2329, 2306 },
		{ 2125, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1726, 37 },
		{ 1929, 40 },
		{ 1402, 1381 },
		{ 1403, 1382 },
		{ 1471, 1453 },
		{ 2340, 2316 },
		{ 1404, 1383 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1915, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1923, 40 },
		{ 1921, 40 },
		{ 1934, 40 },
		{ 1922, 40 },
		{ 1934, 40 },
		{ 1925, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1924, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1405, 1385 },
		{ 1917, 40 },
		{ 1919, 40 },
		{ 1408, 1388 },
		{ 1934, 40 },
		{ 1409, 1389 },
		{ 1934, 40 },
		{ 1932, 40 },
		{ 1920, 40 },
		{ 1934, 40 },
		{ 1933, 40 },
		{ 1926, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1931, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1918, 40 },
		{ 1934, 40 },
		{ 1930, 40 },
		{ 1934, 40 },
		{ 1927, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1934, 40 },
		{ 1316, 21 },
		{ 1472, 1454 },
		{ 1473, 1454 },
		{ 1410, 1390 },
		{ 1303, 21 },
		{ 2366, 2343 },
		{ 2366, 2343 },
		{ 2369, 2346 },
		{ 2369, 2346 },
		{ 2375, 2352 },
		{ 2375, 2352 },
		{ 2387, 2364 },
		{ 2387, 2364 },
		{ 2388, 2365 },
		{ 2388, 2365 },
		{ 2390, 2367 },
		{ 2390, 2367 },
		{ 1411, 1391 },
		{ 1412, 1392 },
		{ 1414, 1394 },
		{ 1415, 1395 },
		{ 1416, 1396 },
		{ 1417, 1397 },
		{ 1316, 21 },
		{ 1472, 1454 },
		{ 1304, 21 },
		{ 1317, 21 },
		{ 1418, 1398 },
		{ 2366, 2343 },
		{ 1419, 1399 },
		{ 2369, 2346 },
		{ 1420, 1400 },
		{ 2375, 2352 },
		{ 1421, 1401 },
		{ 2387, 2364 },
		{ 1422, 1402 },
		{ 2388, 2365 },
		{ 1423, 1403 },
		{ 2390, 2367 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1424, 1405 },
		{ 1427, 1408 },
		{ 1428, 1409 },
		{ 1429, 1410 },
		{ 1474, 1454 },
		{ 1430, 1412 },
		{ 1431, 1414 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1320, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1309, 21 },
		{ 1307, 21 },
		{ 1322, 21 },
		{ 1308, 21 },
		{ 1322, 21 },
		{ 1311, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1310, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1328, 1306 },
		{ 1305, 21 },
		{ 1318, 21 },
		{ 1435, 1416 },
		{ 1312, 21 },
		{ 1436, 1417 },
		{ 1322, 21 },
		{ 1323, 21 },
		{ 1306, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1313, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1321, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1324, 21 },
		{ 1322, 21 },
		{ 1319, 21 },
		{ 1322, 21 },
		{ 1314, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1322, 21 },
		{ 1900, 38 },
		{ 1475, 1455 },
		{ 1476, 1455 },
		{ 1437, 1418 },
		{ 1725, 38 },
		{ 2396, 2373 },
		{ 2396, 2373 },
		{ 2397, 2374 },
		{ 2397, 2374 },
		{ 1439, 1419 },
		{ 1438, 1418 },
		{ 2214, 2189 },
		{ 2214, 2189 },
		{ 2251, 2225 },
		{ 2251, 2225 },
		{ 1441, 1420 },
		{ 1440, 1419 },
		{ 1442, 1421 },
		{ 1443, 1422 },
		{ 1444, 1423 },
		{ 1445, 1424 },
		{ 1448, 1428 },
		{ 1449, 1429 },
		{ 1900, 38 },
		{ 1475, 1455 },
		{ 1730, 38 },
		{ 2252, 2226 },
		{ 2252, 2226 },
		{ 2396, 2373 },
		{ 1450, 1430 },
		{ 2397, 2374 },
		{ 1451, 1431 },
		{ 1478, 1456 },
		{ 1479, 1456 },
		{ 2214, 2189 },
		{ 1453, 1435 },
		{ 2251, 2225 },
		{ 1454, 1436 },
		{ 1899, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 2252, 2226 },
		{ 1740, 38 },
		{ 1455, 1437 },
		{ 1456, 1438 },
		{ 1477, 1455 },
		{ 1457, 1439 },
		{ 1478, 1456 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1727, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1735, 38 },
		{ 1733, 38 },
		{ 1743, 38 },
		{ 1734, 38 },
		{ 1743, 38 },
		{ 1737, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1736, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1458, 1440 },
		{ 1731, 38 },
		{ 1480, 1456 },
		{ 1459, 1441 },
		{ 1743, 38 },
		{ 1460, 1442 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1732, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1728, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1729, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1742, 38 },
		{ 1743, 38 },
		{ 1741, 38 },
		{ 1743, 38 },
		{ 1738, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 1743, 38 },
		{ 2652, 44 },
		{ 2653, 44 },
		{ 1481, 1457 },
		{ 1482, 1457 },
		{ 67, 44 },
		{ 2266, 2241 },
		{ 2266, 2241 },
		{ 1461, 1443 },
		{ 1462, 1444 },
		{ 1463, 1445 },
		{ 1465, 1448 },
		{ 2269, 2244 },
		{ 2269, 2244 },
		{ 1466, 1449 },
		{ 1467, 1450 },
		{ 1468, 1451 },
		{ 1331, 1307 },
		{ 1332, 1308 },
		{ 1336, 1310 },
		{ 1337, 1311 },
		{ 1338, 1312 },
		{ 1487, 1459 },
		{ 1488, 1460 },
		{ 2652, 44 },
		{ 1489, 1461 },
		{ 1481, 1457 },
		{ 1491, 1465 },
		{ 1492, 1466 },
		{ 2266, 2241 },
		{ 1493, 1467 },
		{ 1484, 1458 },
		{ 1485, 1458 },
		{ 1502, 1488 },
		{ 1503, 1488 },
		{ 2269, 2244 },
		{ 1494, 1468 },
		{ 1501, 1487 },
		{ 2180, 44 },
		{ 2651, 44 },
		{ 2179, 44 },
		{ 2179, 44 },
		{ 2179, 44 },
		{ 2179, 44 },
		{ 2179, 44 },
		{ 2179, 44 },
		{ 2179, 44 },
		{ 2179, 44 },
		{ 2179, 44 },
		{ 2179, 44 },
		{ 1339, 1313 },
		{ 1505, 1489 },
		{ 1507, 1491 },
		{ 1508, 1492 },
		{ 1484, 1458 },
		{ 1483, 1457 },
		{ 1502, 1488 },
		{ 2181, 44 },
		{ 2178, 44 },
		{ 2173, 44 },
		{ 2181, 44 },
		{ 2170, 44 },
		{ 2177, 44 },
		{ 2175, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2172, 44 },
		{ 2167, 44 },
		{ 2174, 44 },
		{ 2169, 44 },
		{ 2181, 44 },
		{ 2176, 44 },
		{ 2171, 44 },
		{ 2168, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 1486, 1458 },
		{ 2185, 44 },
		{ 1504, 1488 },
		{ 1509, 1493 },
		{ 2181, 44 },
		{ 1510, 1494 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2182, 44 },
		{ 2183, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2184, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 2181, 44 },
		{ 159, 4 },
		{ 160, 4 },
		{ 1511, 1501 },
		{ 1512, 1501 },
		{ 2410, 2384 },
		{ 2410, 2384 },
		{ 1335, 1309 },
		{ 1341, 1314 },
		{ 1516, 1507 },
		{ 1517, 1508 },
		{ 1340, 1314 },
		{ 1518, 1509 },
		{ 1519, 1510 },
		{ 1344, 1319 },
		{ 1334, 1309 },
		{ 1523, 1517 },
		{ 1345, 1320 },
		{ 1525, 1519 },
		{ 1528, 1523 },
		{ 1529, 1525 },
		{ 1531, 1528 },
		{ 1532, 1529 },
		{ 1533, 1531 },
		{ 159, 4 },
		{ 1534, 1532 },
		{ 1511, 1501 },
		{ 1333, 1309 },
		{ 2410, 2384 },
		{ 1329, 1533 },
		{ 1346, 1321 },
		{ 1347, 1323 },
		{ 1348, 1324 },
		{ 1351, 1328 },
		{ 1352, 1331 },
		{ 1353, 1332 },
		{ 1354, 1333 },
		{ 1355, 1334 },
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
		{ 1356, 1335 },
		{ 1357, 1336 },
		{ 1358, 1337 },
		{ 1361, 1339 },
		{ 0, 2384 },
		{ 1513, 1501 },
		{ 1362, 1340 },
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
		{ 1363, 1341 },
		{ 81, 4 },
		{ 1366, 1344 },
		{ 1367, 1345 },
		{ 85, 4 },
		{ 1368, 1346 },
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
		{ 2902, 2901 },
		{ 1359, 1338 },
		{ 1369, 1347 },
		{ 2901, 2901 },
		{ 1370, 1348 },
		{ 1360, 1338 },
		{ 1373, 1351 },
		{ 1374, 1352 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 1375, 1353 },
		{ 2901, 2901 },
		{ 1376, 1354 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 1377, 1355 },
		{ 1378, 1356 },
		{ 1379, 1357 },
		{ 1380, 1358 },
		{ 1381, 1359 },
		{ 1382, 1360 },
		{ 154, 152 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 97, 80 },
		{ 99, 82 },
		{ 104, 89 },
		{ 105, 90 },
		{ 106, 91 },
		{ 107, 92 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 2901, 2901 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 108, 93 },
		{ 109, 94 },
		{ 110, 95 },
		{ 111, 96 },
		{ 112, 97 },
		{ 114, 99 },
		{ 120, 104 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 121, 105 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 109 },
		{ 1329, 1535 },
		{ 125, 110 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 1329, 1535 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 126, 111 },
		{ 127, 112 },
		{ 129, 114 },
		{ 134, 120 },
		{ 135, 121 },
		{ 136, 122 },
		{ 137, 123 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 138, 124 },
		{ 139, 125 },
		{ 140, 127 },
		{ 141, 129 },
		{ 2181, 2404 },
		{ 142, 134 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 2181, 2404 },
		{ 1330, 1534 },
		{ 0, 1534 },
		{ 143, 135 },
		{ 144, 136 },
		{ 2188, 2167 },
		{ 2190, 2168 },
		{ 2193, 2169 },
		{ 2194, 2170 },
		{ 2196, 2171 },
		{ 2191, 2169 },
		{ 2199, 2172 },
		{ 2202, 2173 },
		{ 2192, 2169 },
		{ 2203, 2174 },
		{ 2565, 2565 },
		{ 2204, 2175 },
		{ 2205, 2176 },
		{ 2195, 2170 },
		{ 2206, 2177 },
		{ 2207, 2178 },
		{ 2181, 2181 },
		{ 2200, 2182 },
		{ 2189, 2183 },
		{ 1330, 1534 },
		{ 2197, 2171 },
		{ 2198, 2184 },
		{ 2213, 2188 },
		{ 145, 137 },
		{ 2215, 2190 },
		{ 2201, 2182 },
		{ 2216, 2191 },
		{ 2217, 2192 },
		{ 2218, 2193 },
		{ 2219, 2194 },
		{ 2220, 2195 },
		{ 2221, 2196 },
		{ 2222, 2197 },
		{ 2565, 2565 },
		{ 2223, 2198 },
		{ 2224, 2199 },
		{ 2225, 2200 },
		{ 2226, 2201 },
		{ 2227, 2202 },
		{ 2228, 2203 },
		{ 2229, 2204 },
		{ 2230, 2205 },
		{ 2231, 2206 },
		{ 2232, 2207 },
		{ 2239, 2213 },
		{ 2241, 2215 },
		{ 2242, 2216 },
		{ 2243, 2217 },
		{ 0, 1534 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2244, 2218 },
		{ 2245, 2219 },
		{ 2246, 2220 },
		{ 2247, 2221 },
		{ 2248, 2222 },
		{ 2249, 2223 },
		{ 2250, 2224 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 67, 7 },
		{ 146, 138 },
		{ 2253, 2227 },
		{ 2254, 2228 },
		{ 2565, 2565 },
		{ 1535, 1534 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 2255, 2229 },
		{ 2256, 2230 },
		{ 2257, 2231 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 2258, 2232 },
		{ 2264, 2239 },
		{ 147, 140 },
		{ 2267, 2242 },
		{ 2268, 2243 },
		{ 148, 141 },
		{ 2272, 2246 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 2273, 2247 },
		{ 2270, 2245 },
		{ 2274, 2248 },
		{ 149, 142 },
		{ 1157, 7 },
		{ 2271, 2245 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 1157, 7 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 2276, 2250 },
		{ 2280, 2253 },
		{ 2281, 2254 },
		{ 2282, 2255 },
		{ 2283, 2256 },
		{ 2284, 2257 },
		{ 2285, 2258 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 2286, 2264 },
		{ 2289, 2267 },
		{ 150, 144 },
		{ 2294, 2271 },
		{ 0, 1393 },
		{ 2295, 2272 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1393 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 2296, 2273 },
		{ 2297, 2274 },
		{ 2299, 2276 },
		{ 2303, 2280 },
		{ 2304, 2281 },
		{ 2305, 2282 },
		{ 2306, 2283 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 2308, 2285 },
		{ 2313, 2289 },
		{ 2316, 2294 },
		{ 2317, 2295 },
		{ 0, 1805 },
		{ 2320, 2297 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1805 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 2319, 2296 },
		{ 2322, 2299 },
		{ 2326, 2303 },
		{ 2327, 2304 },
		{ 2318, 2296 },
		{ 2328, 2305 },
		{ 151, 147 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 2331, 2308 },
		{ 2336, 2313 },
		{ 152, 148 },
		{ 2341, 2317 },
		{ 0, 1999 },
		{ 2342, 2318 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 0, 1999 },
		{ 2343, 2319 },
		{ 2344, 2320 },
		{ 2346, 2322 },
		{ 2350, 2326 },
		{ 2351, 2327 },
		{ 2352, 2328 },
		{ 2356, 2331 },
		{ 2361, 2336 },
		{ 2364, 2341 },
		{ 2365, 2342 },
		{ 2367, 2344 },
		{ 153, 150 },
		{ 2373, 2350 },
		{ 2374, 2351 },
		{ 2379, 2356 },
		{ 2384, 2361 },
		{ 89, 73 },
		{ 155, 153 },
		{ 156, 155 },
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
		{ 0, 2729 },
		{ 2472, 2450 },
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
		{ 116, 101 },
		{ 1167, 1164 },
		{ 116, 101 },
		{ 1167, 1164 },
		{ 85, 157 },
		{ 2482, 2460 },
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
		{ 0, 2379 },
		{ 0, 2379 },
		{ 130, 115 },
		{ 90, 74 },
		{ 130, 115 },
		{ 1170, 1166 },
		{ 2105, 2102 },
		{ 1170, 1166 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1172, 1169 },
		{ 132, 118 },
		{ 1172, 1169 },
		{ 132, 118 },
		{ 1723, 1722 },
		{ 0, 2379 },
		{ 91, 74 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 2259, 2233 },
		{ 2261, 2236 },
		{ 2259, 2233 },
		{ 2261, 2236 },
		{ 1157, 1157 },
		{ 2728, 49 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1157, 1157 },
		{ 1208, 1207 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2234, 2209 },
		{ 2599, 2585 },
		{ 2234, 2209 },
		{ 2615, 2602 },
		{ 1681, 1680 },
		{ 1634, 1633 },
		{ 2404, 2379 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2451, 2425 },
		{ 1678, 1677 },
		{ 1205, 1204 },
		{ 2873, 2872 },
		{ 2893, 2893 },
		{ 1589, 1588 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 182, 173 },
		{ 192, 173 },
		{ 184, 173 },
		{ 1563, 1562 },
		{ 179, 173 },
		{ 2952, 2951 },
		{ 183, 173 },
		{ 1756, 1732 },
		{ 181, 173 },
		{ 86, 49 },
		{ 1563, 1562 },
		{ 2452, 2426 },
		{ 190, 173 },
		{ 189, 173 },
		{ 180, 173 },
		{ 188, 173 },
		{ 1610, 1609 },
		{ 185, 173 },
		{ 187, 173 },
		{ 178, 173 },
		{ 1755, 1732 },
		{ 1942, 1920 },
		{ 186, 173 },
		{ 191, 173 },
		{ 1163, 1160 },
		{ 2403, 2378 },
		{ 1160, 1160 },
		{ 1160, 1160 },
		{ 1160, 1160 },
		{ 1160, 1160 },
		{ 1160, 1160 },
		{ 1160, 1160 },
		{ 1160, 1160 },
		{ 1160, 1160 },
		{ 1160, 1160 },
		{ 1160, 1160 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 100, 83 },
		{ 1164, 1160 },
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
		{ 1221, 1220 },
		{ 431, 391 },
		{ 436, 391 },
		{ 433, 391 },
		{ 432, 391 },
		{ 435, 391 },
		{ 430, 391 },
		{ 2923, 65 },
		{ 429, 391 },
		{ 1268, 1267 },
		{ 67, 65 },
		{ 101, 83 },
		{ 434, 391 },
		{ 1957, 1933 },
		{ 437, 391 },
		{ 2761, 2760 },
		{ 1268, 1267 },
		{ 1972, 1951 },
		{ 2510, 2488 },
		{ 2813, 2812 },
		{ 428, 391 },
		{ 1164, 1160 },
		{ 2208, 2179 },
		{ 2918, 2917 },
		{ 2179, 2179 },
		{ 2179, 2179 },
		{ 2179, 2179 },
		{ 2179, 2179 },
		{ 2179, 2179 },
		{ 2179, 2179 },
		{ 2179, 2179 },
		{ 2179, 2179 },
		{ 2179, 2179 },
		{ 2179, 2179 },
		{ 1956, 1933 },
		{ 2001, 1983 },
		{ 2015, 1998 },
		{ 2854, 2853 },
		{ 1781, 1762 },
		{ 2876, 2875 },
		{ 2884, 2883 },
		{ 1803, 1784 },
		{ 1694, 1693 },
		{ 101, 83 },
		{ 2145, 2144 },
		{ 2209, 2179 },
		{ 1165, 1165 },
		{ 1165, 1165 },
		{ 1165, 1165 },
		{ 1165, 1165 },
		{ 1165, 1165 },
		{ 1165, 1165 },
		{ 1165, 1165 },
		{ 1165, 1165 },
		{ 1165, 1165 },
		{ 1165, 1165 },
		{ 2150, 2149 },
		{ 2208, 2208 },
		{ 2208, 2208 },
		{ 2208, 2208 },
		{ 2208, 2208 },
		{ 2208, 2208 },
		{ 2208, 2208 },
		{ 2208, 2208 },
		{ 2208, 2208 },
		{ 2208, 2208 },
		{ 2208, 2208 },
		{ 1169, 1165 },
		{ 2390, 2390 },
		{ 2390, 2390 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2675, 2674 },
		{ 1637, 1636 },
		{ 2715, 2714 },
		{ 2935, 2931 },
		{ 2921, 65 },
		{ 2209, 2179 },
		{ 2233, 2208 },
		{ 2210, 2210 },
		{ 2210, 2210 },
		{ 2210, 2210 },
		{ 2210, 2210 },
		{ 2210, 2210 },
		{ 2210, 2210 },
		{ 2210, 2210 },
		{ 2210, 2210 },
		{ 2210, 2210 },
		{ 2210, 2210 },
		{ 2919, 65 },
		{ 2720, 2719 },
		{ 2390, 2390 },
		{ 2955, 2954 },
		{ 2329, 2329 },
		{ 2971, 2968 },
		{ 2977, 2974 },
		{ 2485, 2463 },
		{ 2363, 2338 },
		{ 2909, 63 },
		{ 1169, 1165 },
		{ 2236, 2210 },
		{ 67, 63 },
		{ 1841, 1826 },
		{ 1891, 1889 },
		{ 2517, 2495 },
		{ 2527, 2506 },
		{ 2529, 2508 },
		{ 2538, 2518 },
		{ 1639, 1638 },
		{ 2543, 2523 },
		{ 2233, 2208 },
		{ 2922, 65 },
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
		{ 2236, 2210 },
		{ 118, 102 },
		{ 2555, 2537 },
		{ 2251, 2251 },
		{ 2251, 2251 },
		{ 1159, 9 },
		{ 2557, 2539 },
		{ 2370, 2370 },
		{ 2370, 2370 },
		{ 67, 9 },
		{ 2570, 2552 },
		{ 115, 100 },
		{ 2571, 2553 },
		{ 1163, 1163 },
		{ 1163, 1163 },
		{ 1163, 1163 },
		{ 1163, 1163 },
		{ 1163, 1163 },
		{ 1163, 1163 },
		{ 1163, 1163 },
		{ 1163, 1163 },
		{ 1163, 1163 },
		{ 1163, 1163 },
		{ 1661, 1660 },
		{ 1673, 1672 },
		{ 2908, 63 },
		{ 2251, 2251 },
		{ 2580, 2562 },
		{ 1159, 9 },
		{ 2907, 63 },
		{ 2370, 2370 },
		{ 2778, 2778 },
		{ 2778, 2778 },
		{ 118, 102 },
		{ 1166, 1163 },
		{ 2825, 2825 },
		{ 2825, 2825 },
		{ 2315, 2291 },
		{ 2417, 2390 },
		{ 2416, 2390 },
		{ 2354, 2329 },
		{ 2353, 2329 },
		{ 1161, 9 },
		{ 115, 100 },
		{ 1160, 9 },
		{ 1160, 9 },
		{ 1160, 9 },
		{ 1160, 9 },
		{ 1160, 9 },
		{ 1160, 9 },
		{ 1160, 9 },
		{ 1160, 9 },
		{ 1160, 9 },
		{ 1160, 9 },
		{ 2778, 2778 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2585, 2569 },
		{ 2825, 2825 },
		{ 2380, 2380 },
		{ 2380, 2380 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 2397, 2397 },
		{ 2397, 2397 },
		{ 2593, 2578 },
		{ 1166, 1163 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 2410, 2410 },
		{ 2410, 2410 },
		{ 2427, 2427 },
		{ 2427, 2427 },
		{ 2473, 2473 },
		{ 2473, 2473 },
		{ 2556, 2556 },
		{ 2556, 2556 },
		{ 1209, 1208 },
		{ 2372, 2372 },
		{ 2905, 63 },
		{ 2602, 2588 },
		{ 2906, 63 },
		{ 2380, 2380 },
		{ 1558, 1557 },
		{ 2290, 2290 },
		{ 2618, 2605 },
		{ 2397, 2397 },
		{ 2631, 2625 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 2269, 2269 },
		{ 2633, 2627 },
		{ 2410, 2410 },
		{ 2639, 2634 },
		{ 2427, 2427 },
		{ 2645, 2644 },
		{ 2473, 2473 },
		{ 1682, 1681 },
		{ 2556, 2556 },
		{ 2277, 2251 },
		{ 2711, 2711 },
		{ 2711, 2711 },
		{ 2739, 2739 },
		{ 2739, 2739 },
		{ 2013, 1996 },
		{ 2375, 2375 },
		{ 2375, 2375 },
		{ 2345, 2345 },
		{ 2345, 2345 },
		{ 2678, 2677 },
		{ 2278, 2251 },
		{ 2307, 2307 },
		{ 2369, 2369 },
		{ 2369, 2369 },
		{ 2393, 2370 },
		{ 2687, 2686 },
		{ 2152, 2152 },
		{ 2152, 2152 },
		{ 2396, 2396 },
		{ 2396, 2396 },
		{ 2792, 2792 },
		{ 2792, 2792 },
		{ 2700, 2699 },
		{ 2711, 2711 },
		{ 2383, 2360 },
		{ 2739, 2739 },
		{ 2275, 2275 },
		{ 2275, 2275 },
		{ 2375, 2375 },
		{ 2014, 1997 },
		{ 2345, 2345 },
		{ 2833, 2833 },
		{ 2833, 2833 },
		{ 2309, 2309 },
		{ 2309, 2309 },
		{ 2369, 2369 },
		{ 2478, 2478 },
		{ 2478, 2478 },
		{ 2779, 2778 },
		{ 2152, 2152 },
		{ 2385, 2362 },
		{ 2396, 2396 },
		{ 2826, 2825 },
		{ 2792, 2792 },
		{ 2723, 2722 },
		{ 2520, 2520 },
		{ 2520, 2520 },
		{ 1286, 1285 },
		{ 1697, 1696 },
		{ 2275, 2275 },
		{ 2032, 2019 },
		{ 2388, 2388 },
		{ 2388, 2388 },
		{ 2405, 2380 },
		{ 2833, 2833 },
		{ 2045, 2030 },
		{ 2309, 2309 },
		{ 2559, 2559 },
		{ 2559, 2559 },
		{ 2478, 2478 },
		{ 2046, 2031 },
		{ 2406, 2380 },
		{ 2395, 2372 },
		{ 2214, 2214 },
		{ 2214, 2214 },
		{ 2748, 2748 },
		{ 2748, 2748 },
		{ 2314, 2290 },
		{ 2520, 2520 },
		{ 2424, 2397 },
		{ 2740, 2739 },
		{ 2751, 2750 },
		{ 1297, 1296 },
		{ 2291, 2269 },
		{ 2388, 2388 },
		{ 2437, 2410 },
		{ 2777, 2776 },
		{ 2453, 2427 },
		{ 2400, 2375 },
		{ 2495, 2473 },
		{ 2559, 2559 },
		{ 2574, 2556 },
		{ 2106, 2103 },
		{ 2121, 2120 },
		{ 2807, 2806 },
		{ 2337, 2314 },
		{ 2214, 2214 },
		{ 2816, 2815 },
		{ 2748, 2748 },
		{ 2824, 2823 },
		{ 1189, 1188 },
		{ 1611, 1610 },
		{ 2848, 2847 },
		{ 2330, 2307 },
		{ 2147, 2146 },
		{ 2398, 2375 },
		{ 2857, 2856 },
		{ 2867, 2866 },
		{ 1613, 1612 },
		{ 2399, 2375 },
		{ 1784, 1765 },
		{ 2878, 2877 },
		{ 2740, 2739 },
		{ 2159, 2158 },
		{ 2887, 2886 },
		{ 2712, 2711 },
		{ 2347, 2323 },
		{ 2349, 2325 },
		{ 1790, 1771 },
		{ 1224, 1223 },
		{ 2368, 2345 },
		{ 1810, 1791 },
		{ 1263, 1262 },
		{ 1837, 1821 },
		{ 2931, 2927 },
		{ 2392, 2369 },
		{ 2458, 2434 },
		{ 2469, 2447 },
		{ 2359, 2334 },
		{ 2153, 2152 },
		{ 2957, 2956 },
		{ 2423, 2396 },
		{ 2793, 2792 },
		{ 2959, 2959 },
		{ 2476, 2454 },
		{ 1839, 1824 },
		{ 2984, 2982 },
		{ 1952, 1927 },
		{ 2298, 2275 },
		{ 1763, 1738 },
		{ 1951, 1927 },
		{ 1586, 1585 },
		{ 1762, 1738 },
		{ 2834, 2833 },
		{ 1587, 1586 },
		{ 2332, 2309 },
		{ 1239, 1238 },
		{ 1782, 1763 },
		{ 2500, 2478 },
		{ 2266, 2266 },
		{ 2266, 2266 },
		{ 1605, 1604 },
		{ 1789, 1770 },
		{ 2649, 2648 },
		{ 2411, 2385 },
		{ 2670, 2669 },
		{ 2959, 2959 },
		{ 2540, 2520 },
		{ 2162, 2161 },
		{ 1606, 1605 },
		{ 2682, 2681 },
		{ 1801, 1782 },
		{ 1240, 1239 },
		{ 2414, 2388 },
		{ 2425, 2398 },
		{ 2429, 2402 },
		{ 1242, 1241 },
		{ 2436, 2408 },
		{ 1515, 1506 },
		{ 2577, 2559 },
		{ 1629, 1628 },
		{ 1630, 1629 },
		{ 2266, 2266 },
		{ 2734, 2732 },
		{ 1256, 1255 },
		{ 2240, 2214 },
		{ 1852, 1839 },
		{ 2749, 2748 },
		{ 2459, 2436 },
		{ 2755, 2754 },
		{ 1871, 1861 },
		{ 1879, 1871 },
		{ 1257, 1256 },
		{ 1217, 1216 },
		{ 1655, 1654 },
		{ 2487, 2465 },
		{ 1656, 1655 },
		{ 2498, 2476 },
		{ 1193, 1192 },
		{ 1665, 1664 },
		{ 1521, 1515 },
		{ 1973, 1952 },
		{ 1992, 1971 },
		{ 2866, 2865 },
		{ 1994, 1973 },
		{ 1997, 1976 },
		{ 1178, 1177 },
		{ 2544, 2524 },
		{ 2545, 2525 },
		{ 2547, 2528 },
		{ 2548, 2529 },
		{ 2381, 2358 },
		{ 1551, 1550 },
		{ 1552, 1551 },
		{ 1690, 1689 },
		{ 2573, 2555 },
		{ 2386, 2363 },
		{ 1292, 1291 },
		{ 1232, 1231 },
		{ 2947, 2946 },
		{ 2948, 2947 },
		{ 2581, 2563 },
		{ 1713, 1712 },
		{ 1714, 1713 },
		{ 1719, 1718 },
		{ 1579, 1578 },
		{ 1580, 1579 },
		{ 2608, 2594 },
		{ 2672, 2671 },
		{ 2484, 2462 },
		{ 2155, 2154 },
		{ 2962, 2959 },
		{ 1675, 1674 },
		{ 2161, 2160 },
		{ 2698, 2697 },
		{ 2497, 2475 },
		{ 2961, 2959 },
		{ 2709, 2708 },
		{ 2960, 2959 },
		{ 2391, 2368 },
		{ 1963, 1942 },
		{ 2501, 2479 },
		{ 1785, 1766 },
		{ 1284, 1283 },
		{ 1984, 1963 },
		{ 2528, 2507 },
		{ 2301, 2278 },
		{ 2738, 2736 },
		{ 2530, 2509 },
		{ 1680, 1679 },
		{ 2401, 2376 },
		{ 1793, 1774 },
		{ 1565, 1564 },
		{ 1615, 1614 },
		{ 1187, 1186 },
		{ 1820, 1803 },
		{ 2805, 2804 },
		{ 1288, 1287 },
		{ 2288, 2266 },
		{ 2287, 2287 },
		{ 2287, 2287 },
		{ 2560, 2542 },
		{ 2564, 2546 },
		{ 2419, 2392 },
		{ 2420, 2393 },
		{ 2846, 2845 },
		{ 2422, 2395 },
		{ 1584, 1583 },
		{ 1201, 1200 },
		{ 1294, 1293 },
		{ 1641, 1640 },
		{ 2099, 2092 },
		{ 2584, 2568 },
		{ 2102, 2097 },
		{ 2591, 2576 },
		{ 1234, 1233 },
		{ 2449, 2423 },
		{ 1591, 1590 },
		{ 2604, 2590 },
		{ 2118, 2115 },
		{ 1886, 1881 },
		{ 1180, 1179 },
		{ 2287, 2287 },
		{ 2619, 2606 },
		{ 2620, 2607 },
		{ 1207, 1206 },
		{ 2938, 2935 },
		{ 2632, 2626 },
		{ 2464, 2442 },
		{ 1774, 1755 },
		{ 2640, 2638 },
		{ 2643, 2641 },
		{ 2959, 2958 },
		{ 2149, 2148 },
		{ 2967, 2964 },
		{ 1667, 1666 },
		{ 2975, 2972 },
		{ 2477, 2455 },
		{ 1270, 1269 },
		{ 2987, 2986 },
		{ 2629, 2629 },
		{ 2629, 2629 },
		{ 2252, 2252 },
		{ 2252, 2252 },
		{ 2625, 2617 },
		{ 2539, 2519 },
		{ 1296, 1295 },
		{ 2440, 2414 },
		{ 2447, 2421 },
		{ 2474, 2452 },
		{ 2509, 2487 },
		{ 1824, 1809 },
		{ 2552, 2533 },
		{ 2592, 2577 },
		{ 2553, 2535 },
		{ 1282, 1281 },
		{ 2750, 2749 },
		{ 2522, 2500 },
		{ 2558, 2540 },
		{ 1983, 1962 },
		{ 1660, 1659 },
		{ 2310, 2287 },
		{ 2455, 2429 },
		{ 2629, 2629 },
		{ 2434, 2406 },
		{ 2252, 2252 },
		{ 1720, 1719 },
		{ 1752, 1729 },
		{ 2439, 2413 },
		{ 2027, 2011 },
		{ 2725, 2724 },
		{ 2445, 2419 },
		{ 1834, 1818 },
		{ 1640, 1639 },
		{ 1588, 1587 },
		{ 2077, 2059 },
		{ 2078, 2060 },
		{ 1751, 1729 },
		{ 1649, 1648 },
		{ 1545, 1544 },
		{ 2579, 2561 },
		{ 2753, 2752 },
		{ 2104, 2101 },
		{ 2760, 2759 },
		{ 1299, 1298 },
		{ 1659, 1658 },
		{ 2471, 2449 },
		{ 2112, 2109 },
		{ 2115, 2113 },
		{ 1599, 1598 },
		{ 2809, 2808 },
		{ 1888, 1885 },
		{ 2311, 2287 },
		{ 1287, 1286 },
		{ 2818, 2817 },
		{ 1894, 1893 },
		{ 1767, 1744 },
		{ 2607, 2593 },
		{ 1406, 1386 },
		{ 2609, 2595 },
		{ 2850, 2849 },
		{ 1226, 1225 },
		{ 1349, 1325 },
		{ 2859, 2858 },
		{ 1573, 1572 },
		{ 1958, 1935 },
		{ 1962, 1941 },
		{ 2626, 2618 },
		{ 2507, 2485 },
		{ 2880, 2879 },
		{ 1614, 1613 },
		{ 1786, 1767 },
		{ 2889, 2888 },
		{ 1788, 1769 },
		{ 2638, 2633 },
		{ 2518, 2496 },
		{ 1977, 1956 },
		{ 1979, 1958 },
		{ 1981, 1960 },
		{ 2647, 2646 },
		{ 2418, 2391 },
		{ 1281, 1280 },
		{ 2936, 2932 },
		{ 1623, 1622 },
		{ 1191, 1190 },
		{ 1371, 1349 },
		{ 1585, 1584 },
		{ 2680, 2679 },
		{ 1200, 1199 },
		{ 2958, 2957 },
		{ 2012, 1995 },
		{ 2546, 2527 },
		{ 2964, 2961 },
		{ 1819, 1802 },
		{ 2702, 2701 },
		{ 2708, 2707 },
		{ 2634, 2629 },
		{ 1699, 1698 },
		{ 2279, 2252 },
		{ 1707, 1706 },
		{ 2986, 2984 },
		{ 1250, 1249 },
		{ 2800, 2800 },
		{ 2800, 2800 },
		{ 2693, 2693 },
		{ 2693, 2693 },
		{ 2841, 2841 },
		{ 2841, 2841 },
		{ 2366, 2366 },
		{ 2366, 2366 },
		{ 2387, 2387 },
		{ 2387, 2387 },
		{ 2735, 2733 },
		{ 1826, 1811 },
		{ 2302, 2279 },
		{ 2454, 2428 },
		{ 1996, 1975 },
		{ 2457, 2433 },
		{ 1212, 1211 },
		{ 2754, 2753 },
		{ 1836, 1820 },
		{ 1350, 1327 },
		{ 2382, 2359 },
		{ 2762, 2761 },
		{ 2771, 2770 },
		{ 2800, 2800 },
		{ 1622, 1621 },
		{ 2693, 2693 },
		{ 2594, 2579 },
		{ 2841, 2841 },
		{ 2787, 2786 },
		{ 2366, 2366 },
		{ 2788, 2787 },
		{ 2387, 2387 },
		{ 2790, 2789 },
		{ 1300, 1299 },
		{ 1261, 1260 },
		{ 2803, 2802 },
		{ 1860, 1848 },
		{ 2019, 2002 },
		{ 1769, 1747 },
		{ 2810, 2809 },
		{ 1873, 1864 },
		{ 2814, 2813 },
		{ 1773, 1754 },
		{ 1885, 1880 },
		{ 2819, 2818 },
		{ 2057, 2043 },
		{ 1386, 1364 },
		{ 2831, 2830 },
		{ 2622, 2609 },
		{ 1544, 1543 },
		{ 2844, 2843 },
		{ 2079, 2061 },
		{ 2090, 2077 },
		{ 2091, 2078 },
		{ 2851, 2850 },
		{ 2508, 2486 },
		{ 2855, 2854 },
		{ 1192, 1191 },
		{ 2101, 2096 },
		{ 2860, 2859 },
		{ 2865, 2864 },
		{ 1783, 1764 },
		{ 1895, 1894 },
		{ 2521, 2499 },
		{ 1685, 1684 },
		{ 1227, 1226 },
		{ 2648, 2647 },
		{ 2881, 2880 },
		{ 1249, 1248 },
		{ 2885, 2884 },
		{ 2114, 2112 },
		{ 1695, 1694 },
		{ 2890, 2889 },
		{ 1598, 1597 },
		{ 1648, 1647 },
		{ 2676, 2675 },
		{ 2903, 2900 },
		{ 1960, 1938 },
		{ 1961, 1940 },
		{ 2681, 2680 },
		{ 1791, 1772 },
		{ 2929, 2925 },
		{ 1700, 1699 },
		{ 2932, 2928 },
		{ 1706, 1705 },
		{ 2696, 2695 },
		{ 2426, 2399 },
		{ 2941, 2938 },
		{ 2151, 2150 },
		{ 1556, 1555 },
		{ 2703, 2702 },
		{ 2237, 2211 },
		{ 1974, 1953 },
		{ 1279, 1278 },
		{ 2801, 2800 },
		{ 1426, 1407 },
		{ 2694, 2693 },
		{ 113, 98 },
		{ 2842, 2841 },
		{ 2721, 2720 },
		{ 2389, 2366 },
		{ 2441, 2415 },
		{ 2413, 2387 },
		{ 2567, 2549 },
		{ 2726, 2725 },
		{ 1821, 1804 },
		{ 1572, 1571 },
		{ 1222, 1221 },
		{ 2300, 2277 },
		{ 2718, 2718 },
		{ 2718, 2718 },
		{ 1692, 1692 },
		{ 1692, 1692 },
		{ 1219, 1219 },
		{ 1219, 1219 },
		{ 2882, 2882 },
		{ 2882, 2882 },
		{ 2852, 2852 },
		{ 2852, 2852 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2589, 2589 },
		{ 2589, 2589 },
		{ 2811, 2811 },
		{ 2811, 2811 },
		{ 1807, 1788 },
		{ 1251, 1250 },
		{ 1624, 1623 },
		{ 2939, 2936 },
		{ 2335, 2312 },
		{ 1663, 1662 },
		{ 2157, 2156 },
		{ 2718, 2718 },
		{ 2502, 2480 },
		{ 1692, 1692 },
		{ 1708, 1707 },
		{ 1219, 1219 },
		{ 2028, 2012 },
		{ 2882, 2882 },
		{ 2107, 2104 },
		{ 2852, 2852 },
		{ 1425, 1406 },
		{ 2673, 2673 },
		{ 1600, 1599 },
		{ 2589, 2589 },
		{ 1650, 1649 },
		{ 2811, 2811 },
		{ 1574, 1573 },
		{ 1835, 1819 },
		{ 2974, 2971 },
		{ 2123, 2122 },
		{ 2000, 1981 },
		{ 1890, 1888 },
		{ 1546, 1545 },
		{ 1609, 1608 },
		{ 2871, 2871 },
		{ 2871, 2871 },
		{ 2836, 2836 },
		{ 2836, 2836 },
		{ 1676, 1676 },
		{ 1676, 1676 },
		{ 2795, 2795 },
		{ 2795, 2795 },
		{ 2688, 2688 },
		{ 2688, 2688 },
		{ 1203, 1203 },
		{ 1203, 1203 },
		{ 2829, 2829 },
		{ 2829, 2829 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1687, 1687 },
		{ 1687, 1687 },
		{ 1721, 1720 },
		{ 2018, 2001 },
		{ 2572, 2554 },
		{ 1689, 1688 },
		{ 2146, 2145 },
		{ 2871, 2871 },
		{ 1851, 1838 },
		{ 2836, 2836 },
		{ 1561, 1560 },
		{ 1676, 1676 },
		{ 1971, 1950 },
		{ 2795, 2795 },
		{ 1237, 1236 },
		{ 2688, 2688 },
		{ 1863, 1853 },
		{ 1203, 1203 },
		{ 2746, 2745 },
		{ 2829, 2829 },
		{ 1794, 1775 },
		{ 1214, 1214 },
		{ 2076, 2058 },
		{ 1687, 1687 },
		{ 1636, 1635 },
		{ 1254, 1253 },
		{ 1805, 1786 },
		{ 1490, 1464 },
		{ 1549, 1548 },
		{ 1770, 1750 },
		{ 2945, 2944 },
		{ 2719, 2718 },
		{ 1266, 1265 },
		{ 1693, 1692 },
		{ 1577, 1576 },
		{ 1220, 1219 },
		{ 1216, 1215 },
		{ 2883, 2882 },
		{ 2954, 2953 },
		{ 2853, 2852 },
		{ 1998, 1977 },
		{ 2674, 2673 },
		{ 1999, 1979 },
		{ 2603, 2589 },
		{ 2483, 2461 },
		{ 2812, 2811 },
		{ 2551, 2532 },
		{ 1393, 1371 },
		{ 2111, 2108 },
		{ 2875, 2874 },
		{ 1711, 1710 },
		{ 2624, 2616 },
		{ 1653, 1652 },
		{ 2496, 2474 },
		{ 2980, 2977 },
		{ 1603, 1602 },
		{ 1627, 1626 },
		{ 1185, 1184 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 1181, 1181 },
		{ 1181, 1181 },
		{ 2490, 2490 },
		{ 2490, 2490 },
		{ 2492, 2492 },
		{ 2492, 2492 },
		{ 2742, 2741 },
		{ 2769, 2768 },
		{ 0, 1161 },
		{ 0, 84 },
		{ 0, 2180 },
		{ 2764, 2764 },
		{ 2764, 2764 },
		{ 1744, 1900 },
		{ 1935, 2126 },
		{ 1325, 1316 },
		{ 2872, 2871 },
		{ 1698, 1697 },
		{ 2837, 2836 },
		{ 1939, 1917 },
		{ 1677, 1676 },
		{ 2371, 2371 },
		{ 2796, 2795 },
		{ 1181, 1181 },
		{ 2689, 2688 },
		{ 2490, 2490 },
		{ 1204, 1203 },
		{ 2492, 2492 },
		{ 2830, 2829 },
		{ 2588, 2572 },
		{ 1215, 1214 },
		{ 2120, 2118 },
		{ 1688, 1687 },
		{ 2849, 2848 },
		{ 2764, 2764 },
		{ 1825, 1810 },
		{ 2323, 2300 },
		{ 2325, 2302 },
		{ 2595, 2580 },
		{ 1664, 1663 },
		{ 0, 1161 },
		{ 0, 84 },
		{ 0, 2180 },
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
		{ 2235, 2235 },
		{ 2235, 2235 },
		{ 2235, 2235 },
		{ 2235, 2235 },
		{ 2235, 2235 },
		{ 2235, 2235 },
		{ 2235, 2235 },
		{ 2235, 2235 },
		{ 2235, 2235 },
		{ 2235, 2235 },
		{ 2260, 2260 },
		{ 2260, 2260 },
		{ 2260, 2260 },
		{ 2260, 2260 },
		{ 2260, 2260 },
		{ 2260, 2260 },
		{ 2260, 2260 },
		{ 2260, 2260 },
		{ 2260, 2260 },
		{ 2260, 2260 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 2262, 2262 },
		{ 2724, 2723 },
		{ 2858, 2857 },
		{ 1775, 1756 },
		{ 1940, 1917 },
		{ 1225, 1224 },
		{ 1267, 1266 },
		{ 2394, 2371 },
		{ 2519, 2497 },
		{ 1182, 1181 },
		{ 1260, 1259 },
		{ 2512, 2490 },
		{ 2029, 2013 },
		{ 2514, 2492 },
		{ 2524, 2502 },
		{ 2616, 2603 },
		{ 2617, 2604 },
		{ 2450, 2424 },
		{ 2879, 2878 },
		{ 2765, 2764 },
		{ 1168, 1168 },
		{ 1168, 1168 },
		{ 1168, 1168 },
		{ 1168, 1168 },
		{ 1168, 1168 },
		{ 1168, 1168 },
		{ 1168, 1168 },
		{ 1168, 1168 },
		{ 1168, 1168 },
		{ 1168, 1168 },
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
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 2895, 2895 },
		{ 1171, 1171 },
		{ 1171, 1171 },
		{ 1171, 1171 },
		{ 1171, 1171 },
		{ 1171, 1171 },
		{ 1171, 1171 },
		{ 1171, 1171 },
		{ 1171, 1171 },
		{ 1171, 1171 },
		{ 1171, 1171 },
		{ 1173, 1173 },
		{ 1173, 1173 },
		{ 1173, 1173 },
		{ 1173, 1173 },
		{ 1173, 1173 },
		{ 1173, 1173 },
		{ 1173, 1173 },
		{ 1173, 1173 },
		{ 1173, 1173 },
		{ 1173, 1173 },
		{ 1235, 1235 },
		{ 1235, 1235 },
		{ 2481, 2481 },
		{ 2481, 2481 },
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
		{ 2979, 2979 },
		{ 2782, 2782 },
		{ 2782, 2782 },
		{ 2704, 2704 },
		{ 2704, 2704 },
		{ 2466, 2466 },
		{ 2466, 2466 },
		{ 2745, 2744 },
		{ 1326, 1305 },
		{ 1235, 1235 },
		{ 2334, 2311 },
		{ 2481, 2481 },
		{ 2043, 2027 },
		{ 2532, 2511 },
		{ 2752, 2751 },
		{ 2888, 2887 },
		{ 1190, 1189 },
		{ 1848, 1834 },
		{ 1278, 1277 },
		{ 2059, 2045 },
		{ 2060, 2046 },
		{ 2460, 2437 },
		{ 2463, 2441 },
		{ 2979, 2979 },
		{ 2782, 2782 },
		{ 1262, 1261 },
		{ 2704, 2704 },
		{ 2770, 2769 },
		{ 2466, 2466 },
		{ 1718, 1717 },
		{ 2549, 2530 },
		{ 1295, 1294 },
		{ 2646, 2645 },
		{ 2786, 2785 },
		{ 1555, 1554 },
		{ 1684, 1683 },
		{ 2789, 2788 },
		{ 2554, 2536 },
		{ 1211, 1210 },
		{ 1557, 1556 },
		{ 2480, 2458 },
		{ 1407, 1387 },
		{ 2561, 2543 },
		{ 1753, 1731 },
		{ 2808, 2807 },
		{ 1559, 1558 },
		{ 2679, 2678 },
		{ 2569, 2551 },
		{ 1264, 1263 },
		{ 2360, 2335 },
		{ 1893, 1891 },
		{ 2967, 2967 },
		{ 2817, 2816 },
		{ 1562, 1561 },
		{ 2312, 2288 },
		{ 2109, 2106 },
		{ 1662, 1661 },
		{ 2499, 2477 },
		{ 2701, 2700 },
		{ 1298, 1297 },
		{ 1771, 1751 },
		{ 2692, 2691 },
		{ 2827, 2826 },
		{ 2766, 2765 },
		{ 1683, 1682 },
		{ 67, 5 },
		{ 1210, 1209 },
		{ 2840, 2839 },
		{ 2122, 2121 },
		{ 2979, 2976 },
		{ 2799, 2798 },
		{ 2491, 2469 },
		{ 2780, 2779 },
		{ 2586, 2570 },
		{ 2967, 2967 },
		{ 1327, 1305 },
		{ 1236, 1235 },
		{ 2587, 2571 },
		{ 2503, 2481 },
		{ 2772, 2771 },
		{ 2541, 2521 },
		{ 2791, 2790 },
		{ 1277, 1276 },
		{ 2444, 2418 },
		{ 1686, 1685 },
		{ 2785, 2784 },
		{ 2113, 2111 },
		{ 1213, 1212 },
		{ 2583, 2567 },
		{ 2732, 2730 },
		{ 2783, 2782 },
		{ 2402, 2377 },
		{ 2705, 2704 },
		{ 2981, 2979 },
		{ 2488, 2466 },
		{ 2924, 2919 },
		{ 1764, 1741 },
		{ 2465, 2443 },
		{ 2357, 2332 },
		{ 1902, 1899 },
		{ 1765, 1741 },
		{ 2143, 2142 },
		{ 2794, 2793 },
		{ 2377, 2354 },
		{ 1901, 1899 },
		{ 2733, 2730 },
		{ 2321, 2298 },
		{ 2835, 2834 },
		{ 2265, 2240 },
		{ 2550, 2531 },
		{ 1754, 1731 },
		{ 2707, 2706 },
		{ 1387, 1365 },
		{ 1746, 1726 },
		{ 2744, 2743 },
		{ 1937, 1914 },
		{ 2531, 2510 },
		{ 174, 5 },
		{ 1745, 1726 },
		{ 1199, 1198 },
		{ 1936, 1914 },
		{ 175, 5 },
		{ 1672, 1671 },
		{ 2443, 2417 },
		{ 2128, 2125 },
		{ 2925, 2919 },
		{ 1343, 1317 },
		{ 1976, 1955 },
		{ 176, 5 },
		{ 2127, 2125 },
		{ 2494, 2472 },
		{ 1564, 1563 },
		{ 1283, 1282 },
		{ 1823, 1807 },
		{ 1571, 1570 },
		{ 2142, 2141 },
		{ 1666, 1665 },
		{ 1248, 1247 },
		{ 2630, 2624 },
		{ 2802, 2801 },
		{ 1993, 1972 },
		{ 2804, 2803 },
		{ 1179, 1178 },
		{ 2970, 2967 },
		{ 173, 5 },
		{ 2504, 2482 },
		{ 1621, 1620 },
		{ 1747, 1727 },
		{ 1748, 1728 },
		{ 1749, 1728 },
		{ 1674, 1673 },
		{ 2641, 2639 },
		{ 1364, 1342 },
		{ 1575, 1574 },
		{ 2408, 2382 },
		{ 2158, 2157 },
		{ 2324, 2301 },
		{ 2002, 1984 },
		{ 2160, 2159 },
		{ 2415, 2389 },
		{ 2828, 2827 },
		{ 2671, 2670 },
		{ 2010, 1992 },
		{ 1365, 1343 },
		{ 1679, 1678 },
		{ 1625, 1624 },
		{ 2421, 2394 },
		{ 1543, 1542 },
		{ 2843, 2842 },
		{ 2533, 2512 },
		{ 2845, 2844 },
		{ 2535, 2514 },
		{ 1864, 1854 },
		{ 2017, 2000 },
		{ 1233, 1232 },
		{ 1206, 1205 },
		{ 2542, 2522 },
		{ 2695, 2694 },
		{ 2428, 2401 },
		{ 2697, 2696 },
		{ 1772, 1752 },
		{ 1880, 1872 },
		{ 2433, 2405 },
		{ 1881, 1873 },
		{ 1633, 1632 },
		{ 1291, 1290 },
		{ 2868, 2867 },
		{ 2706, 2705 },
		{ 1547, 1546 },
		{ 1252, 1251 },
		{ 1269, 1268 },
		{ 2710, 2709 },
		{ 2348, 2324 },
		{ 2713, 2712 },
		{ 1892, 1890 },
		{ 2717, 2716 },
		{ 1750, 1728 },
		{ 1183, 1182 },
		{ 2446, 2420 },
		{ 2061, 2047 },
		{ 2448, 2422 },
		{ 1691, 1690 },
		{ 2563, 2545 },
		{ 2211, 2185 },
		{ 2900, 2892 },
		{ 2355, 2330 },
		{ 2568, 2550 },
		{ 1590, 1589 },
		{ 1647, 1646 },
		{ 1276, 1275 },
		{ 1597, 1596 },
		{ 2736, 2734 },
		{ 2456, 2430 },
		{ 98, 81 },
		{ 2927, 2922 },
		{ 2928, 2924 },
		{ 2576, 2558 },
		{ 2743, 2742 },
		{ 2092, 2079 },
		{ 2096, 2089 },
		{ 1651, 1650 },
		{ 2747, 2746 },
		{ 2462, 2440 },
		{ 1938, 1915 },
		{ 2942, 2939 },
		{ 1202, 1201 },
		{ 1705, 1704 },
		{ 2468, 2446 },
		{ 2951, 2950 },
		{ 1800, 1781 },
		{ 2470, 2448 },
		{ 1198, 1197 },
		{ 1601, 1600 },
		{ 1804, 1785 },
		{ 2110, 2107 },
		{ 1446, 1425 },
		{ 2963, 2960 },
		{ 2767, 2766 },
		{ 2479, 2457 },
		{ 1709, 1708 },
		{ 1808, 1789 },
		{ 2972, 2969 },
		{ 1447, 1426 },
		{ 1811, 1793 },
		{ 2606, 2592 },
		{ 1241, 1240 },
		{ 2781, 2780 },
		{ 2486, 2464 },
		{ 2784, 2783 },
		{ 1218, 1217 },
		{ 1975, 1954 },
		{ 1717, 1716 },
		{ 1802, 1783 },
		{ 1995, 1974 },
		{ 2714, 2713 },
		{ 1792, 1773 },
		{ 2933, 2929 },
		{ 2263, 2237 },
		{ 2969, 2966 },
		{ 1812, 1794 },
		{ 2870, 2869 },
		{ 2737, 2735 },
		{ 1372, 1350 },
		{ 2378, 2355 },
		{ 1982, 1961 },
		{ 1954, 1931 },
		{ 2904, 2903 },
		{ 2763, 2762 },
		{ 1290, 1289 },
		{ 2832, 2831 },
		{ 128, 113 },
		{ 2798, 2797 },
		{ 2475, 2453 },
		{ 2716, 2715 },
		{ 1696, 1695 },
		{ 1840, 1825 },
		{ 1223, 1222 },
		{ 2856, 2855 },
		{ 2965, 2962 },
		{ 2886, 2885 },
		{ 2968, 2965 },
		{ 1293, 1292 },
		{ 2722, 2721 },
		{ 1184, 1183 },
		{ 2691, 2690 },
		{ 585, 531 },
		{ 2839, 2838 },
		{ 2976, 2973 },
		{ 2044, 2029 },
		{ 2768, 2767 },
		{ 2590, 2574 },
		{ 2815, 2814 },
		{ 2983, 2981 },
		{ 1583, 1582 },
		{ 2677, 2676 },
		{ 586, 531 },
		{ 2430, 2403 },
		{ 2536, 2515 },
		{ 2511, 2489 },
		{ 2493, 2471 },
		{ 1325, 1318 },
		{ 1935, 1929 },
		{ 1275, 1272 },
		{ 1744, 1740 },
		{ 2467, 2445 },
		{ 202, 179 },
		{ 1950, 1926 },
		{ 1654, 1653 },
		{ 200, 179 },
		{ 1612, 1611 },
		{ 201, 179 },
		{ 587, 531 },
		{ 1506, 1490 },
		{ 2690, 2689 },
		{ 1953, 1930 },
		{ 2605, 2591 },
		{ 2058, 2044 },
		{ 1722, 1721 },
		{ 1889, 1886 },
		{ 199, 179 },
		{ 2376, 2353 },
		{ 1635, 1634 },
		{ 2699, 2698 },
		{ 2506, 2484 },
		{ 2461, 2439 },
		{ 1265, 1264 },
		{ 2838, 2837 },
		{ 1255, 1254 },
		{ 2338, 2315 },
		{ 1638, 1637 },
		{ 2562, 2544 },
		{ 1186, 1185 },
		{ 1560, 1559 },
		{ 2627, 2619 },
		{ 2847, 2846 },
		{ 2515, 2493 },
		{ 2144, 2143 },
		{ 1604, 1603 },
		{ 2946, 2945 },
		{ 1550, 1549 },
		{ 1853, 1840 },
		{ 2097, 2090 },
		{ 2523, 2501 },
		{ 2953, 2952 },
		{ 1238, 1237 },
		{ 2525, 2503 },
		{ 2956, 2955 },
		{ 1861, 1851 },
		{ 2644, 2643 },
		{ 2578, 2560 },
		{ 1712, 1711 },
		{ 2154, 2153 },
		{ 2103, 2099 },
		{ 2797, 2796 },
		{ 2966, 2963 },
		{ 2869, 2868 },
		{ 2156, 2155 },
		{ 2442, 2416 },
		{ 1578, 1577 },
		{ 2874, 2873 },
		{ 1628, 1627 },
		{ 2973, 2970 },
		{ 1188, 1187 },
		{ 2877, 2876 },
		{ 2537, 2517 },
		{ 1285, 1284 },
		{ 2806, 2805 },
		{ 2030, 2015 },
		{ 2489, 2467 },
		{ 2982, 2980 },
		{ 2741, 2740 },
		{ 2362, 2337 },
		{ 2031, 2018 },
		{ 1766, 1742 },
		{ 641, 584 },
		{ 703, 650 },
		{ 806, 759 },
		{ 406, 368 },
		{ 1176, 11 },
		{ 1595, 27 },
		{ 1246, 17 },
		{ 67, 11 },
		{ 67, 27 },
		{ 67, 17 },
		{ 2668, 45 },
		{ 1619, 29 },
		{ 1645, 31 },
		{ 67, 45 },
		{ 67, 29 },
		{ 67, 31 },
		{ 513, 456 },
		{ 1569, 25 },
		{ 702, 650 },
		{ 407, 368 },
		{ 67, 25 },
		{ 1274, 19 },
		{ 642, 584 },
		{ 514, 457 },
		{ 67, 19 },
		{ 516, 459 },
		{ 528, 473 },
		{ 532, 477 },
		{ 807, 759 },
		{ 536, 481 },
		{ 546, 491 },
		{ 560, 503 },
		{ 561, 505 },
		{ 566, 510 },
		{ 233, 200 },
		{ 594, 535 },
		{ 607, 548 },
		{ 1776, 1757 },
		{ 1964, 1943 },
		{ 610, 551 },
		{ 618, 559 },
		{ 631, 571 },
		{ 640, 583 },
		{ 239, 206 },
		{ 660, 602 },
		{ 257, 221 },
		{ 704, 651 },
		{ 716, 662 },
		{ 718, 664 },
		{ 720, 666 },
		{ 724, 670 },
		{ 1986, 1965 },
		{ 1987, 1966 },
		{ 741, 687 },
		{ 1796, 1777 },
		{ 1797, 1778 },
		{ 750, 696 },
		{ 754, 700 },
		{ 755, 701 },
		{ 798, 751 },
		{ 266, 230 },
		{ 832, 787 },
		{ 840, 795 },
		{ 855, 810 },
		{ 2009, 1991 },
		{ 890, 845 },
		{ 894, 849 },
		{ 910, 868 },
		{ 1817, 1799 },
		{ 924, 886 },
		{ 948, 914 },
		{ 951, 917 },
		{ 958, 925 },
		{ 968, 935 },
		{ 2025, 2008 },
		{ 984, 953 },
		{ 1003, 971 },
		{ 1010, 979 },
		{ 1832, 1816 },
		{ 1012, 981 },
		{ 1024, 997 },
		{ 1037, 1011 },
		{ 1042, 1017 },
		{ 1049, 1025 },
		{ 1050, 1026 },
		{ 1064, 1046 },
		{ 1174, 11 },
		{ 1593, 27 },
		{ 1244, 17 },
		{ 1081, 1065 },
		{ 1086, 1071 },
		{ 1093, 1082 },
		{ 2666, 45 },
		{ 1617, 29 },
		{ 1643, 31 },
		{ 1104, 1094 },
		{ 1110, 1100 },
		{ 1145, 1144 },
		{ 273, 237 },
		{ 1567, 25 },
		{ 287, 248 },
		{ 292, 253 },
		{ 298, 258 },
		{ 1272, 19 },
		{ 307, 267 },
		{ 323, 282 },
		{ 326, 285 },
		{ 332, 290 },
		{ 337, 295 },
		{ 350, 311 },
		{ 371, 333 },
		{ 374, 336 },
		{ 380, 342 },
		{ 391, 354 },
		{ 397, 360 },
		{ 228, 195 },
		{ 417, 377 },
		{ 232, 199 },
		{ 438, 392 },
		{ 441, 395 },
		{ 451, 403 },
		{ 452, 404 },
		{ 462, 414 },
		{ 470, 423 },
		{ 502, 447 },
		{ 511, 454 },
		{ 67, 57 },
		{ 67, 15 },
		{ 67, 51 },
		{ 67, 55 },
		{ 67, 23 },
		{ 67, 35 },
		{ 67, 13 },
		{ 67, 47 },
		{ 67, 33 },
		{ 67, 53 },
		{ 67, 59 },
		{ 67, 41 },
		{ 2037, 2023 },
		{ 2039, 2024 },
		{ 2098, 2091 },
		{ 2917, 2916 },
		{ 443, 397 },
		{ 445, 397 },
		{ 390, 353 },
		{ 2038, 2023 },
		{ 2040, 2024 },
		{ 765, 711 },
		{ 410, 371 },
		{ 413, 372 },
		{ 411, 371 },
		{ 472, 425 },
		{ 553, 498 },
		{ 446, 397 },
		{ 681, 629 },
		{ 682, 630 },
		{ 2035, 2021 },
		{ 1844, 1830 },
		{ 444, 397 },
		{ 762, 708 },
		{ 480, 432 },
		{ 481, 432 },
		{ 1045, 1020 },
		{ 1947, 1923 },
		{ 319, 278 },
		{ 639, 582 },
		{ 211, 184 },
		{ 412, 371 },
		{ 482, 432 },
		{ 210, 184 },
		{ 493, 441 },
		{ 1946, 1923 },
		{ 282, 243 },
		{ 215, 187 },
		{ 1968, 1947 },
		{ 212, 184 },
		{ 419, 379 },
		{ 846, 801 },
		{ 591, 532 },
		{ 495, 441 },
		{ 847, 802 },
		{ 343, 301 },
		{ 590, 532 },
		{ 1945, 1923 },
		{ 494, 441 },
		{ 1054, 1030 },
		{ 400, 362 },
		{ 399, 362 },
		{ 217, 187 },
		{ 216, 187 },
		{ 589, 532 },
		{ 588, 532 },
		{ 223, 190 },
		{ 487, 436 },
		{ 767, 713 },
		{ 545, 490 },
		{ 488, 436 },
		{ 484, 434 },
		{ 260, 224 },
		{ 290, 251 },
		{ 670, 616 },
		{ 1779, 1760 },
		{ 842, 797 },
		{ 1053, 1030 },
		{ 222, 190 },
		{ 2821, 57 },
		{ 1229, 15 },
		{ 2730, 51 },
		{ 2774, 55 },
		{ 1540, 23 },
		{ 1702, 35 },
		{ 1195, 13 },
		{ 2684, 47 },
		{ 1669, 33 },
		{ 2757, 53 },
		{ 2862, 59 },
		{ 2139, 41 },
		{ 485, 434 },
		{ 557, 502 },
		{ 1969, 1948 },
		{ 220, 188 },
		{ 766, 712 },
		{ 558, 502 },
		{ 218, 188 },
		{ 274, 238 },
		{ 905, 863 },
		{ 219, 188 },
		{ 2358, 2333 },
		{ 2915, 2913 },
		{ 402, 364 },
		{ 771, 717 },
		{ 671, 617 },
		{ 1078, 1062 },
		{ 559, 502 },
		{ 2926, 2921 },
		{ 673, 619 },
		{ 1084, 1068 },
		{ 816, 770 },
		{ 234, 201 },
		{ 275, 238 },
		{ 1101, 1091 },
		{ 1177, 1174 },
		{ 1759, 1735 },
		{ 1231, 1229 },
		{ 2937, 2934 },
		{ 1102, 1092 },
		{ 746, 692 },
		{ 988, 957 },
		{ 534, 479 },
		{ 700, 648 },
		{ 2669, 2666 },
		{ 1011, 980 },
		{ 701, 649 },
		{ 574, 517 },
		{ 654, 596 },
		{ 458, 410 },
		{ 901, 858 },
		{ 207, 182 },
		{ 205, 181 },
		{ 525, 470 },
		{ 970, 937 },
		{ 972, 939 },
		{ 208, 182 },
		{ 206, 181 },
		{ 976, 943 },
		{ 982, 951 },
		{ 983, 952 },
		{ 354, 315 },
		{ 491, 439 },
		{ 996, 964 },
		{ 416, 375 },
		{ 742, 688 },
		{ 743, 689 },
		{ 524, 470 },
		{ 576, 519 },
		{ 1021, 991 },
		{ 749, 695 },
		{ 1036, 1010 },
		{ 580, 524 },
		{ 1038, 1012 },
		{ 751, 697 },
		{ 584, 530 },
		{ 497, 443 },
		{ 760, 706 },
		{ 761, 707 },
		{ 1058, 1037 },
		{ 1063, 1043 },
		{ 500, 445 },
		{ 1071, 1054 },
		{ 2916, 2915 },
		{ 1072, 1056 },
		{ 368, 331 },
		{ 295, 256 },
		{ 1082, 1066 },
		{ 329, 287 },
		{ 377, 339 },
		{ 781, 730 },
		{ 1094, 1083 },
		{ 2930, 2926 },
		{ 1096, 1085 },
		{ 782, 731 },
		{ 789, 741 },
		{ 795, 748 },
		{ 328, 287 },
		{ 327, 287 },
		{ 1108, 1098 },
		{ 1778, 1759 },
		{ 1109, 1099 },
		{ 196, 178 },
		{ 2940, 2937 },
		{ 1114, 1105 },
		{ 1129, 1120 },
		{ 198, 178 },
		{ 520, 464 },
		{ 1156, 1155 },
		{ 814, 768 },
		{ 521, 465 },
		{ 821, 775 },
		{ 829, 784 },
		{ 197, 178 },
		{ 383, 345 },
		{ 526, 471 },
		{ 657, 599 },
		{ 1966, 1945 },
		{ 333, 291 },
		{ 336, 294 },
		{ 288, 249 },
		{ 856, 811 },
		{ 862, 817 },
		{ 868, 823 },
		{ 873, 828 },
		{ 877, 832 },
		{ 882, 838 },
		{ 889, 844 },
		{ 398, 361 },
		{ 674, 620 },
		{ 540, 485 },
		{ 544, 489 },
		{ 463, 415 },
		{ 911, 869 },
		{ 921, 883 },
		{ 258, 222 },
		{ 929, 891 },
		{ 930, 892 },
		{ 934, 896 },
		{ 935, 897 },
		{ 941, 903 },
		{ 551, 496 },
		{ 348, 308 },
		{ 271, 235 },
		{ 352, 313 },
		{ 351, 312 },
		{ 1511, 1511 },
		{ 1432, 1432 },
		{ 659, 601 },
		{ 474, 428 },
		{ 523, 469 },
		{ 569, 513 },
		{ 672, 618 },
		{ 907, 865 },
		{ 769, 715 },
		{ 570, 514 },
		{ 387, 350 },
		{ 334, 292 },
		{ 577, 520 },
		{ 1076, 1060 },
		{ 689, 639 },
		{ 1536, 1536 },
		{ 227, 194 },
		{ 801, 754 },
		{ 802, 755 },
		{ 944, 907 },
		{ 2004, 1986 },
		{ 945, 908 },
		{ 946, 910 },
		{ 1511, 1511 },
		{ 1432, 1432 },
		{ 582, 528 },
		{ 394, 357 },
		{ 953, 919 },
		{ 358, 319 },
		{ 238, 205 },
		{ 969, 936 },
		{ 538, 483 },
		{ 601, 542 },
		{ 973, 940 },
		{ 1139, 1133 },
		{ 837, 792 },
		{ 839, 794 },
		{ 1469, 1469 },
		{ 1536, 1536 },
		{ 1814, 1796 },
		{ 226, 193 },
		{ 1472, 1472 },
		{ 1475, 1475 },
		{ 1478, 1478 },
		{ 1481, 1481 },
		{ 1484, 1484 },
		{ 344, 304 },
		{ 616, 557 },
		{ 992, 960 },
		{ 375, 337 },
		{ 314, 273 },
		{ 1008, 977 },
		{ 636, 576 },
		{ 549, 494 },
		{ 241, 208 },
		{ 1502, 1502 },
		{ 1016, 985 },
		{ 415, 374 },
		{ 517, 460 },
		{ 881, 837 },
		{ 1469, 1469 },
		{ 1009, 978 },
		{ 1342, 1511 },
		{ 1342, 1432 },
		{ 1472, 1472 },
		{ 1475, 1475 },
		{ 1478, 1478 },
		{ 1481, 1481 },
		{ 1484, 1484 },
		{ 854, 809 },
		{ 729, 675 },
		{ 734, 680 },
		{ 857, 812 },
		{ 2022, 2005 },
		{ 860, 815 },
		{ 614, 555 },
		{ 1028, 1001 },
		{ 1342, 1536 },
		{ 1502, 1502 },
		{ 1031, 1004 },
		{ 1032, 1005 },
		{ 512, 455 },
		{ 870, 825 },
		{ 871, 826 },
		{ 2912, 2908 },
		{ 872, 827 },
		{ 1044, 1019 },
		{ 456, 408 },
		{ 1046, 1021 },
		{ 554, 499 },
		{ 403, 365 },
		{ 2055, 2041 },
		{ 2056, 2042 },
		{ 272, 236 },
		{ 378, 340 },
		{ 563, 507 },
		{ 643, 585 },
		{ 565, 509 },
		{ 468, 420 },
		{ 1342, 1469 },
		{ 291, 252 },
		{ 908, 866 },
		{ 764, 710 },
		{ 1342, 1472 },
		{ 1342, 1475 },
		{ 1342, 1478 },
		{ 1342, 1481 },
		{ 1342, 1484 },
		{ 342, 300 },
		{ 913, 874 },
		{ 918, 880 },
		{ 666, 608 },
		{ 922, 884 },
		{ 384, 346 },
		{ 1816, 1798 },
		{ 356, 317 },
		{ 251, 216 },
		{ 1342, 1502 },
		{ 931, 893 },
		{ 772, 718 },
		{ 778, 726 },
		{ 780, 728 },
		{ 363, 324 },
		{ 364, 326 },
		{ 783, 732 },
		{ 1831, 1815 },
		{ 1142, 1140 },
		{ 947, 912 },
		{ 1146, 1145 },
		{ 676, 622 },
		{ 677, 624 },
		{ 679, 627 },
		{ 583, 529 },
		{ 960, 927 },
		{ 492, 440 },
		{ 685, 632 },
		{ 688, 637 },
		{ 315, 274 },
		{ 698, 646 },
		{ 1858, 1846 },
		{ 1859, 1847 },
		{ 214, 186 },
		{ 349, 309 },
		{ 322, 281 },
		{ 605, 546 },
		{ 2008, 1990 },
		{ 504, 449 },
		{ 609, 550 },
		{ 454, 406 },
		{ 611, 552 },
		{ 851, 806 },
		{ 204, 180 },
		{ 573, 516 },
		{ 476, 430 },
		{ 255, 219 },
		{ 572, 516 },
		{ 506, 451 },
		{ 507, 451 },
		{ 508, 452 },
		{ 509, 452 },
		{ 254, 219 },
		{ 203, 180 },
		{ 626, 566 },
		{ 627, 566 },
		{ 477, 430 },
		{ 370, 332 },
		{ 369, 332 },
		{ 277, 239 },
		{ 276, 239 },
		{ 244, 210 },
		{ 296, 257 },
		{ 478, 431 },
		{ 243, 210 },
		{ 281, 242 },
		{ 713, 660 },
		{ 420, 380 },
		{ 455, 407 },
		{ 2934, 2930 },
		{ 297, 257 },
		{ 253, 218 },
		{ 834, 789 },
		{ 479, 431 },
		{ 1965, 1944 },
		{ 280, 242 },
		{ 714, 660 },
		{ 562, 506 },
		{ 2026, 2009 },
		{ 339, 297 },
		{ 719, 665 },
		{ 1970, 1949 },
		{ 768, 714 },
		{ 628, 567 },
		{ 1815, 1797 },
		{ 722, 668 },
		{ 1148, 1147 },
		{ 564, 508 },
		{ 2041, 2025 },
		{ 916, 877 },
		{ 1056, 1035 },
		{ 1777, 1758 },
		{ 1057, 1036 },
		{ 725, 671 },
		{ 1780, 1761 },
		{ 1059, 1038 },
		{ 440, 394 },
		{ 340, 298 },
		{ 1833, 1817 },
		{ 678, 626 },
		{ 597, 538 },
		{ 1074, 1058 },
		{ 548, 493 },
		{ 1941, 1918 },
		{ 489, 437 },
		{ 869, 824 },
		{ 2913, 2911 },
		{ 647, 589 },
		{ 937, 899 },
		{ 1846, 1832 },
		{ 2333, 2310 },
		{ 2005, 1987 },
		{ 648, 590 },
		{ 1091, 1080 },
		{ 1013, 982 },
		{ 649, 591 },
		{ 268, 232 },
		{ 265, 229 },
		{ 283, 244 },
		{ 712, 659 },
		{ 1115, 1106 },
		{ 1869, 1858 },
		{ 1126, 1117 },
		{ 1127, 1118 },
		{ 1034, 1007 },
		{ 1132, 1123 },
		{ 2074, 2055 },
		{ 555, 500 },
		{ 1140, 1135 },
		{ 1141, 1136 },
		{ 899, 855 },
		{ 954, 920 },
		{ 1040, 1014 },
		{ 224, 191 },
		{ 1809, 1790 },
		{ 841, 796 },
		{ 753, 699 },
		{ 256, 220 },
		{ 1047, 1023 },
		{ 1758, 1734 },
		{ 449, 400 },
		{ 1760, 1736 },
		{ 665, 607 },
		{ 912, 873 },
		{ 418, 378 },
		{ 668, 610 },
		{ 522, 468 },
		{ 728, 674 },
		{ 986, 955 },
		{ 809, 761 },
		{ 1944, 1922 },
		{ 459, 411 },
		{ 1948, 1924 },
		{ 552, 497 },
		{ 999, 967 },
		{ 819, 773 },
		{ 1004, 972 },
		{ 1005, 973 },
		{ 1006, 974 },
		{ 245, 211 },
		{ 823, 777 },
		{ 1090, 1079 },
		{ 826, 781 },
		{ 1845, 1831 },
		{ 827, 782 },
		{ 938, 900 },
		{ 2036, 2022 },
		{ 874, 829 },
		{ 942, 904 },
		{ 567, 511 },
		{ 830, 785 },
		{ 542, 487 },
		{ 1029, 1002 },
		{ 710, 657 },
		{ 683, 631 },
		{ 759, 705 },
		{ 1100, 1090 },
		{ 804, 757 },
		{ 971, 938 },
		{ 1033, 1006 },
		{ 684, 631 },
		{ 376, 338 },
		{ 593, 534 },
		{ 975, 942 },
		{ 1111, 1102 },
		{ 861, 816 },
		{ 439, 393 },
		{ 1124, 1115 },
		{ 294, 255 },
		{ 284, 245 },
		{ 985, 954 },
		{ 531, 476 },
		{ 1133, 1126 },
		{ 1134, 1127 },
		{ 1138, 1132 },
		{ 606, 547 },
		{ 405, 367 },
		{ 650, 592 },
		{ 1052, 1029 },
		{ 1143, 1141 },
		{ 997, 965 },
		{ 1055, 1034 },
		{ 998, 966 },
		{ 1155, 1154 },
		{ 653, 595 },
		{ 359, 320 },
		{ 242, 209 },
		{ 1060, 1040 },
		{ 1829, 1813 },
		{ 421, 381 },
		{ 886, 841 },
		{ 1065, 1047 },
		{ 1070, 1053 },
		{ 2020, 2003 },
		{ 496, 442 },
		{ 581, 525 },
		{ 617, 558 },
		{ 785, 737 },
		{ 541, 486 },
		{ 793, 746 },
		{ 426, 389 },
		{ 1017, 986 },
		{ 1018, 988 },
		{ 1088, 1077 },
		{ 1089, 1078 },
		{ 2911, 2907 },
		{ 848, 803 },
		{ 1967, 1946 },
		{ 475, 429 },
		{ 1026, 999 },
		{ 259, 223 },
		{ 1095, 1084 },
		{ 644, 586 },
		{ 341, 299 },
		{ 902, 859 },
		{ 1153, 1152 },
		{ 321, 280 },
		{ 396, 359 },
		{ 2006, 1988 },
		{ 2007, 1989 },
		{ 623, 563 },
		{ 625, 565 },
		{ 579, 523 },
		{ 229, 196 },
		{ 388, 351 },
		{ 787, 739 },
		{ 721, 667 },
		{ 835, 790 },
		{ 635, 575 },
		{ 661, 603 },
		{ 875, 830 },
		{ 556, 501 },
		{ 799, 752 },
		{ 568, 512 },
		{ 231, 198 },
		{ 733, 679 },
		{ 414, 373 },
		{ 424, 385 },
		{ 879, 834 },
		{ 310, 270 },
		{ 658, 600 },
		{ 1991, 1970 },
		{ 1051, 1027 },
		{ 347, 307 },
		{ 1116, 1107 },
		{ 1121, 1112 },
		{ 596, 537 },
		{ 1125, 1116 },
		{ 745, 691 },
		{ 250, 215 },
		{ 898, 853 },
		{ 1130, 1121 },
		{ 1131, 1122 },
		{ 747, 693 },
		{ 777, 725 },
		{ 949, 915 },
		{ 1137, 1131 },
		{ 950, 916 },
		{ 820, 774 },
		{ 662, 604 },
		{ 1069, 1051 },
		{ 858, 813 },
		{ 956, 923 },
		{ 510, 453 },
		{ 1073, 1057 },
		{ 959, 926 },
		{ 1149, 1148 },
		{ 1075, 1059 },
		{ 278, 240 },
		{ 1022, 993 },
		{ 687, 634 },
		{ 867, 822 },
		{ 261, 225 },
		{ 723, 669 },
		{ 1087, 1074 },
		{ 305, 265 },
		{ 230, 197 },
		{ 335, 293 },
		{ 836, 791 },
		{ 1799, 1780 },
		{ 925, 887 },
		{ 926, 888 },
		{ 450, 402 },
		{ 1039, 1013 },
		{ 763, 709 },
		{ 655, 597 },
		{ 987, 956 },
		{ 878, 833 },
		{ 990, 958 },
		{ 933, 895 },
		{ 1330, 1330 },
		{ 539, 484 },
		{ 991, 958 },
		{ 686, 633 },
		{ 989, 958 },
		{ 786, 738 },
		{ 311, 271 },
		{ 499, 444 },
		{ 822, 776 },
		{ 980, 948 },
		{ 312, 271 },
		{ 498, 444 },
		{ 691, 640 },
		{ 690, 640 },
		{ 979, 948 },
		{ 692, 640 },
		{ 1123, 1114 },
		{ 1035, 1009 },
		{ 706, 653 },
		{ 955, 921 },
		{ 824, 778 },
		{ 709, 656 },
		{ 613, 554 },
		{ 1330, 1330 },
		{ 891, 846 },
		{ 966, 933 },
		{ 967, 934 },
		{ 892, 847 },
		{ 1136, 1129 },
		{ 711, 658 },
		{ 547, 492 },
		{ 575, 518 },
		{ 900, 856 },
		{ 1830, 1814 },
		{ 833, 788 },
		{ 974, 941 },
		{ 715, 661 },
		{ 904, 862 },
		{ 486, 435 },
		{ 389, 352 },
		{ 249, 214 },
		{ 460, 412 },
		{ 299, 259 },
		{ 346, 306 },
		{ 629, 569 },
		{ 915, 876 },
		{ 630, 570 },
		{ 675, 621 },
		{ 920, 882 },
		{ 465, 417 },
		{ 467, 419 },
		{ 530, 475 },
		{ 1002, 970 },
		{ 1988, 1967 },
		{ 1989, 1968 },
		{ 1077, 1061 },
		{ 367, 329 },
		{ 1079, 1063 },
		{ 592, 533 },
		{ 469, 421 },
		{ 794, 747 },
		{ 1007, 976 },
		{ 1342, 1330 },
		{ 859, 814 },
		{ 353, 314 },
		{ 796, 749 },
		{ 595, 536 },
		{ 936, 898 },
		{ 2864, 2862 },
		{ 471, 424 },
		{ 1014, 983 },
		{ 381, 343 },
		{ 940, 902 },
		{ 1098, 1088 },
		{ 235, 202 },
		{ 603, 544 },
		{ 279, 241 },
		{ 1023, 996 },
		{ 1106, 1096 },
		{ 808, 760 },
		{ 385, 347 },
		{ 372, 334 },
		{ 425, 387 },
		{ 757, 703 },
		{ 483, 433 },
		{ 2021, 2004 },
		{ 309, 269 },
		{ 2023, 2006 },
		{ 2024, 2007 },
		{ 1117, 1108 },
		{ 1118, 1109 },
		{ 386, 348 },
		{ 805, 758 },
		{ 994, 962 },
		{ 995, 963 },
		{ 1025, 998 },
		{ 932, 894 },
		{ 1795, 1776 },
		{ 1872, 1863 },
		{ 1985, 1964 },
		{ 1596, 1593 },
		{ 849, 804 },
		{ 2089, 2076 },
		{ 533, 478 },
		{ 652, 594 },
		{ 1620, 1617 },
		{ 320, 279 },
		{ 756, 702 },
		{ 664, 606 },
		{ 1704, 1702 },
		{ 1570, 1567 },
		{ 1854, 1841 },
		{ 2047, 2032 },
		{ 1646, 1643 },
		{ 1247, 1244 },
		{ 1080, 1064 },
		{ 1542, 1540 },
		{ 473, 426 },
		{ 883, 839 },
		{ 1030, 1003 },
		{ 779, 727 },
		{ 457, 409 },
		{ 884, 839 },
		{ 404, 366 },
		{ 237, 204 },
		{ 357, 318 },
		{ 852, 807 },
		{ 645, 587 },
		{ 1061, 1041 },
		{ 1062, 1042 },
		{ 656, 598 },
		{ 876, 831 },
		{ 1092, 1081 },
		{ 409, 370 },
		{ 1067, 1049 },
		{ 1068, 1050 },
		{ 788, 740 },
		{ 770, 716 },
		{ 318, 277 },
		{ 1135, 1128 },
		{ 694, 642 },
		{ 776, 724 },
		{ 1048, 1024 },
		{ 366, 328 },
		{ 727, 673 },
		{ 957, 924 },
		{ 247, 213 },
		{ 289, 250 },
		{ 1949, 1925 },
		{ 248, 213 },
		{ 1128, 1119 },
		{ 1041, 1015 },
		{ 300, 260 },
		{ 2049, 2035 },
		{ 2051, 2037 },
		{ 2052, 2038 },
		{ 2053, 2039 },
		{ 2054, 2040 },
		{ 598, 539 },
		{ 810, 763 },
		{ 812, 766 },
		{ 961, 928 },
		{ 965, 932 },
		{ 302, 262 },
		{ 1813, 1795 },
		{ 815, 769 },
		{ 527, 472 },
		{ 818, 772 },
		{ 696, 644 },
		{ 373, 335 },
		{ 529, 474 },
		{ 893, 848 },
		{ 1144, 1142 },
		{ 490, 438 },
		{ 608, 549 },
		{ 1147, 1146 },
		{ 393, 356 },
		{ 977, 944 },
		{ 1152, 1151 },
		{ 825, 779 },
		{ 1154, 1153 },
		{ 981, 949 },
		{ 705, 652 },
		{ 338, 296 },
		{ 395, 358 },
		{ 612, 553 },
		{ 906, 864 },
		{ 831, 786 },
		{ 193, 174 },
		{ 909, 867 },
		{ 461, 413 },
		{ 1990, 1969 },
		{ 615, 556 },
		{ 1847, 1833 },
		{ 663, 605 },
		{ 423, 383 },
		{ 914, 875 },
		{ 1761, 1737 },
		{ 213, 185 },
		{ 1856, 1844 },
		{ 263, 227 },
		{ 1001, 969 },
		{ 917, 878 },
		{ 775, 721 },
		{ 2003, 1985 },
		{ 919, 881 },
		{ 622, 562 },
		{ 325, 284 },
		{ 844, 799 },
		{ 845, 800 },
		{ 624, 564 },
		{ 505, 450 },
		{ 927, 889 },
		{ 427, 390 },
		{ 401, 363 },
		{ 850, 805 },
		{ 264, 228 },
		{ 360, 321 },
		{ 382, 344 },
		{ 732, 678 },
		{ 236, 203 },
		{ 442, 396 },
		{ 735, 681 },
		{ 790, 743 },
		{ 736, 682 },
		{ 737, 683 },
		{ 740, 686 },
		{ 866, 821 },
		{ 680, 628 },
		{ 797, 750 },
		{ 330, 288 },
		{ 447, 398 },
		{ 744, 690 },
		{ 209, 183 },
		{ 1798, 1779 },
		{ 803, 756 },
		{ 267, 231 },
		{ 316, 275 },
		{ 2042, 2026 },
		{ 2116, 2116 },
		{ 2116, 2116 },
		{ 1865, 1865 },
		{ 1865, 1865 },
		{ 1867, 1867 },
		{ 1867, 1867 },
		{ 1479, 1479 },
		{ 1479, 1479 },
		{ 2062, 2062 },
		{ 2062, 2062 },
		{ 2064, 2064 },
		{ 2064, 2064 },
		{ 2066, 2066 },
		{ 2066, 2066 },
		{ 2068, 2068 },
		{ 2068, 2068 },
		{ 2070, 2070 },
		{ 2070, 2070 },
		{ 2072, 2072 },
		{ 2072, 2072 },
		{ 1842, 1842 },
		{ 1842, 1842 },
		{ 864, 819 },
		{ 2116, 2116 },
		{ 448, 399 },
		{ 1865, 1865 },
		{ 301, 261 },
		{ 1867, 1867 },
		{ 535, 480 },
		{ 1479, 1479 },
		{ 651, 593 },
		{ 2062, 2062 },
		{ 515, 458 },
		{ 2064, 2064 },
		{ 269, 233 },
		{ 2066, 2066 },
		{ 638, 581 },
		{ 2068, 2068 },
		{ 621, 561 },
		{ 2070, 2070 },
		{ 246, 212 },
		{ 2072, 2072 },
		{ 800, 753 },
		{ 1842, 1842 },
		{ 619, 560 },
		{ 620, 560 },
		{ 1877, 1877 },
		{ 1877, 1877 },
		{ 2087, 2087 },
		{ 2087, 2087 },
		{ 1537, 1537 },
		{ 1537, 1537 },
		{ 2117, 2116 },
		{ 518, 462 },
		{ 1866, 1865 },
		{ 306, 266 },
		{ 1868, 1867 },
		{ 392, 355 },
		{ 1480, 1479 },
		{ 2141, 2139 },
		{ 2063, 2062 },
		{ 543, 488 },
		{ 2065, 2064 },
		{ 286, 247 },
		{ 2067, 2066 },
		{ 1015, 984 },
		{ 2069, 2068 },
		{ 717, 663 },
		{ 2071, 2070 },
		{ 1877, 1877 },
		{ 2073, 2072 },
		{ 2087, 2087 },
		{ 1843, 1842 },
		{ 1537, 1537 },
		{ 1882, 1882 },
		{ 1882, 1882 },
		{ 2033, 2033 },
		{ 2033, 2033 },
		{ 2093, 2093 },
		{ 2093, 2093 },
		{ 1482, 1482 },
		{ 1482, 1482 },
		{ 1473, 1473 },
		{ 1473, 1473 },
		{ 1485, 1485 },
		{ 1485, 1485 },
		{ 1433, 1433 },
		{ 1433, 1433 },
		{ 1512, 1512 },
		{ 1512, 1512 },
		{ 1476, 1476 },
		{ 1476, 1476 },
		{ 1896, 1896 },
		{ 1896, 1896 },
		{ 1503, 1503 },
		{ 1503, 1503 },
		{ 928, 890 },
		{ 1882, 1882 },
		{ 1878, 1877 },
		{ 2033, 2033 },
		{ 2088, 2087 },
		{ 2093, 2093 },
		{ 1538, 1537 },
		{ 1482, 1482 },
		{ 863, 818 },
		{ 1473, 1473 },
		{ 1019, 989 },
		{ 1485, 1485 },
		{ 1020, 990 },
		{ 1433, 1433 },
		{ 885, 840 },
		{ 1512, 1512 },
		{ 758, 704 },
		{ 1476, 1476 },
		{ 1103, 1093 },
		{ 1896, 1896 },
		{ 697, 645 },
		{ 1503, 1503 },
		{ 1470, 1470 },
		{ 1470, 1470 },
		{ 1105, 1095 },
		{ 331, 289 },
		{ 699, 647 },
		{ 1838, 1823 },
		{ 843, 798 },
		{ 1000, 968 },
		{ 1883, 1882 },
		{ 817, 771 },
		{ 2034, 2033 },
		{ 1113, 1104 },
		{ 2094, 2093 },
		{ 2944, 2942 },
		{ 1483, 1482 },
		{ 1151, 1150 },
		{ 1474, 1473 },
		{ 791, 744 },
		{ 1486, 1485 },
		{ 895, 850 },
		{ 1434, 1433 },
		{ 1626, 1625 },
		{ 1513, 1512 },
		{ 1470, 1470 },
		{ 1477, 1476 },
		{ 1576, 1575 },
		{ 1897, 1896 },
		{ 897, 852 },
		{ 1504, 1503 },
		{ 978, 946 },
		{ 1083, 1067 },
		{ 1120, 1111 },
		{ 195, 176 },
		{ 1085, 1070 },
		{ 362, 323 },
		{ 748, 694 },
		{ 773, 719 },
		{ 774, 720 },
		{ 903, 861 },
		{ 962, 929 },
		{ 963, 930 },
		{ 1253, 1252 },
		{ 1464, 1446 },
		{ 1870, 1859 },
		{ 1548, 1547 },
		{ 726, 672 },
		{ 1818, 1801 },
		{ 1066, 1048 },
		{ 2914, 2912 },
		{ 943, 905 },
		{ 2011, 1994 },
		{ 1652, 1651 },
		{ 1471, 1470 },
		{ 838, 793 },
		{ 1602, 1601 },
		{ 1097, 1086 },
		{ 993, 961 },
		{ 1099, 1089 },
		{ 2075, 2056 },
		{ 1710, 1709 },
		{ 1043, 1018 },
		{ 1876, 1869 },
		{ 1857, 1845 },
		{ 1197, 1195 },
		{ 1119, 1110 },
		{ 578, 522 },
		{ 1943, 1921 },
		{ 2050, 2036 },
		{ 1955, 1932 },
		{ 221, 189 },
		{ 293, 254 },
		{ 466, 418 },
		{ 1757, 1733 },
		{ 2086, 2074 },
		{ 2759, 2757 },
		{ 2411, 2411 },
		{ 2411, 2411 },
		{ 2431, 2431 },
		{ 2431, 2431 },
		{ 1615, 1615 },
		{ 1615, 1615 },
		{ 2557, 2557 },
		{ 2557, 2557 },
		{ 1641, 1641 },
		{ 1641, 1641 },
		{ 2649, 2649 },
		{ 2649, 2649 },
		{ 2504, 2504 },
		{ 2504, 2504 },
		{ 2819, 2819 },
		{ 2819, 2819 },
		{ 2596, 2596 },
		{ 2596, 2596 },
		{ 2598, 2598 },
		{ 2598, 2598 },
		{ 2599, 2599 },
		{ 2599, 2599 },
		{ 731, 677 },
		{ 2411, 2411 },
		{ 604, 545 },
		{ 2431, 2431 },
		{ 361, 322 },
		{ 1615, 1615 },
		{ 422, 382 },
		{ 2557, 2557 },
		{ 896, 851 },
		{ 1641, 1641 },
		{ 646, 588 },
		{ 2649, 2649 },
		{ 784, 734 },
		{ 2504, 2504 },
		{ 1027, 1000 },
		{ 2819, 2819 },
		{ 317, 276 },
		{ 2596, 2596 },
		{ 252, 217 },
		{ 2598, 2598 },
		{ 738, 684 },
		{ 2599, 2599 },
		{ 964, 931 },
		{ 2600, 2600 },
		{ 2600, 2600 },
		{ 2601, 2601 },
		{ 2601, 2601 },
		{ 2438, 2411 },
		{ 739, 685 },
		{ 2432, 2431 },
		{ 2823, 2821 },
		{ 1616, 1615 },
		{ 571, 515 },
		{ 2575, 2557 },
		{ 262, 226 },
		{ 1642, 1641 },
		{ 693, 641 },
		{ 2650, 2649 },
		{ 792, 745 },
		{ 2526, 2504 },
		{ 365, 327 },
		{ 2820, 2819 },
		{ 695, 643 },
		{ 2610, 2596 },
		{ 194, 175 },
		{ 2611, 2598 },
		{ 2600, 2600 },
		{ 2612, 2599 },
		{ 2601, 2601 },
		{ 308, 268 },
		{ 1227, 1227 },
		{ 1227, 1227 },
		{ 2772, 2772 },
		{ 2772, 2772 },
		{ 1270, 1270 },
		{ 1270, 1270 },
		{ 2483, 2483 },
		{ 2483, 2483 },
		{ 2162, 2162 },
		{ 2162, 2162 },
		{ 2726, 2726 },
		{ 2726, 2726 },
		{ 2564, 2564 },
		{ 2564, 2564 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2608, 2608 },
		{ 2608, 2608 },
		{ 1565, 1565 },
		{ 1565, 1565 },
		{ 2615, 2615 },
		{ 2615, 2615 },
		{ 2613, 2600 },
		{ 1227, 1227 },
		{ 2614, 2601 },
		{ 2772, 2772 },
		{ 519, 463 },
		{ 1270, 1270 },
		{ 1107, 1097 },
		{ 2483, 2483 },
		{ 225, 192 },
		{ 2162, 2162 },
		{ 853, 808 },
		{ 2726, 2726 },
		{ 408, 369 },
		{ 2564, 2564 },
		{ 464, 416 },
		{ 2292, 2292 },
		{ 1112, 1103 },
		{ 2608, 2608 },
		{ 550, 495 },
		{ 1565, 1565 },
		{ 752, 698 },
		{ 2615, 2615 },
		{ 285, 246 },
		{ 2682, 2682 },
		{ 2682, 2682 },
		{ 2383, 2383 },
		{ 2383, 2383 },
		{ 1228, 1227 },
		{ 324, 283 },
		{ 2773, 2772 },
		{ 270, 234 },
		{ 1271, 1270 },
		{ 707, 654 },
		{ 2505, 2483 },
		{ 708, 655 },
		{ 2163, 2162 },
		{ 923, 885 },
		{ 2727, 2726 },
		{ 355, 316 },
		{ 2582, 2564 },
		{ 2686, 2684 },
		{ 2293, 2292 },
		{ 1122, 1113 },
		{ 2621, 2608 },
		{ 313, 272 },
		{ 1566, 1565 },
		{ 2682, 2682 },
		{ 2623, 2615 },
		{ 2383, 2383 },
		{ 865, 820 },
		{ 2513, 2513 },
		{ 2513, 2513 },
		{ 1242, 1242 },
		{ 1242, 1242 },
		{ 1591, 1591 },
		{ 1591, 1591 },
		{ 2620, 2620 },
		{ 2620, 2620 },
		{ 2386, 2386 },
		{ 2386, 2386 },
		{ 1667, 1667 },
		{ 1667, 1667 },
		{ 1723, 1723 },
		{ 1723, 1723 },
		{ 2547, 2547 },
		{ 2547, 2547 },
		{ 2548, 2548 },
		{ 2548, 2548 },
		{ 2630, 2630 },
		{ 2630, 2630 },
		{ 2631, 2631 },
		{ 2631, 2631 },
		{ 2683, 2682 },
		{ 2513, 2513 },
		{ 2409, 2383 },
		{ 1242, 1242 },
		{ 811, 765 },
		{ 1591, 1591 },
		{ 1671, 1669 },
		{ 2620, 2620 },
		{ 2108, 2105 },
		{ 2386, 2386 },
		{ 501, 446 },
		{ 1667, 1667 },
		{ 2776, 2774 },
		{ 1723, 1723 },
		{ 813, 767 },
		{ 2547, 2547 },
		{ 240, 207 },
		{ 2548, 2548 },
		{ 503, 448 },
		{ 2630, 2630 },
		{ 667, 609 },
		{ 2631, 2631 },
		{ 303, 263 },
		{ 2632, 2632 },
		{ 2632, 2632 },
		{ 2494, 2494 },
		{ 2494, 2494 },
		{ 2534, 2513 },
		{ 669, 613 },
		{ 1243, 1242 },
		{ 304, 264 },
		{ 1592, 1591 },
		{ 345, 305 },
		{ 2628, 2620 },
		{ 632, 572 },
		{ 2412, 2386 },
		{ 633, 573 },
		{ 1668, 1667 },
		{ 939, 901 },
		{ 1724, 1723 },
		{ 634, 574 },
		{ 2565, 2547 },
		{ 379, 341 },
		{ 2566, 2548 },
		{ 880, 836 },
		{ 2635, 2630 },
		{ 2632, 2632 },
		{ 2636, 2631 },
		{ 2494, 2494 },
		{ 599, 540 },
		{ 2339, 2339 },
		{ 2339, 2339 },
		{ 2860, 2860 },
		{ 2860, 2860 },
		{ 2583, 2583 },
		{ 2583, 2583 },
		{ 2407, 2407 },
		{ 2407, 2407 },
		{ 2640, 2640 },
		{ 2640, 2640 },
		{ 1700, 1700 },
		{ 1700, 1700 },
		{ 2755, 2755 },
		{ 2755, 2755 },
		{ 1193, 1193 },
		{ 1193, 1193 },
		{ 637, 580 },
		{ 600, 541 },
		{ 828, 783 },
		{ 453, 405 },
		{ 887, 842 },
		{ 888, 843 },
		{ 2637, 2632 },
		{ 2339, 2339 },
		{ 2516, 2494 },
		{ 2860, 2860 },
		{ 602, 543 },
		{ 2583, 2583 },
		{ 537, 482 },
		{ 2407, 2407 },
		{ 1150, 1149 },
		{ 2640, 2640 },
		{ 952, 918 },
		{ 1700, 1700 },
		{ 730, 676 },
		{ 2755, 2755 },
		{ 1452, 1434 },
		{ 1193, 1193 },
		{ 1887, 1883 },
		{ 2975, 2975 },
		{ 1874, 1866 },
		{ 2983, 2983 },
		{ 2987, 2987 },
		{ 1500, 1486 },
		{ 2080, 2063 },
		{ 2100, 2094 },
		{ 1875, 1868 },
		{ 2081, 2065 },
		{ 1498, 1480 },
		{ 2340, 2339 },
		{ 2082, 2067 },
		{ 2861, 2860 },
		{ 1495, 1471 },
		{ 2597, 2583 },
		{ 2083, 2069 },
		{ 2435, 2407 },
		{ 1497, 1477 },
		{ 2642, 2640 },
		{ 2084, 2071 },
		{ 1701, 1700 },
		{ 1499, 1483 },
		{ 2756, 2755 },
		{ 2975, 2975 },
		{ 1194, 1193 },
		{ 2983, 2983 },
		{ 2987, 2987 },
		{ 2085, 2073 },
		{ 1855, 1843 },
		{ 1539, 1538 },
		{ 2048, 2034 },
		{ 1514, 1504 },
		{ 1898, 1897 },
		{ 1884, 1878 },
		{ 1520, 1513 },
		{ 1496, 1474 },
		{ 2119, 2117 },
		{ 2095, 2088 },
		{ 2891, 2890 },
		{ 1553, 1552 },
		{ 1554, 1553 },
		{ 1581, 1580 },
		{ 2949, 2948 },
		{ 2950, 2949 },
		{ 1715, 1714 },
		{ 1716, 1715 },
		{ 1582, 1581 },
		{ 1631, 1630 },
		{ 1632, 1631 },
		{ 2978, 2975 },
		{ 1259, 1258 },
		{ 2985, 2983 },
		{ 2988, 2987 },
		{ 1530, 1526 },
		{ 1607, 1606 },
		{ 1608, 1607 },
		{ 1258, 1257 },
		{ 1526, 1521 },
		{ 1527, 1522 },
		{ 1302, 1301 },
		{ 1657, 1656 },
		{ 1658, 1657 },
		{ 2656, 2656 },
		{ 2653, 2656 },
		{ 163, 163 },
		{ 160, 163 },
		{ 1827, 1812 },
		{ 1828, 1812 },
		{ 1849, 1837 },
		{ 1850, 1837 },
		{ 2129, 2129 },
		{ 1903, 1903 },
		{ 168, 164 },
		{ 2134, 2130 },
		{ 2655, 2651 },
		{ 1908, 1904 },
		{ 88, 70 },
		{ 167, 164 },
		{ 2133, 2130 },
		{ 2654, 2651 },
		{ 1907, 1904 },
		{ 87, 70 },
		{ 2662, 2659 },
		{ 2661, 2657 },
		{ 162, 158 },
		{ 2656, 2656 },
		{ 2187, 2164 },
		{ 163, 163 },
		{ 2660, 2657 },
		{ 161, 158 },
		{ 2664, 2663 },
		{ 2186, 2164 },
		{ 2238, 2212 },
		{ 2129, 2129 },
		{ 1903, 1903 },
		{ 1787, 1768 },
		{ 1909, 1906 },
		{ 1911, 1910 },
		{ 2135, 2132 },
		{ 2137, 2136 },
		{ 2657, 2656 },
		{ 1980, 1959 },
		{ 164, 163 },
		{ 119, 103 },
		{ 169, 166 },
		{ 171, 170 },
		{ 1978, 1957 },
		{ 1862, 1852 },
		{ 2130, 2129 },
		{ 1904, 1903 },
		{ 2663, 2661 },
		{ 2132, 2128 },
		{ 1910, 1908 },
		{ 1959, 1937 },
		{ 170, 168 },
		{ 2212, 2187 },
		{ 2136, 2134 },
		{ 1906, 1902 },
		{ 2659, 2655 },
		{ 166, 162 },
		{ 1768, 1746 },
		{ 103, 88 },
		{ 0, 2470 },
		{ 2505, 2505 },
		{ 2505, 2505 },
		{ 0, 2321 },
		{ 1911, 1911 },
		{ 1912, 1911 },
		{ 2597, 2597 },
		{ 2597, 2597 },
		{ 0, 2810 },
		{ 2664, 2664 },
		{ 2665, 2664 },
		{ 0, 1234 },
		{ 0, 2444 },
		{ 0, 1691 },
		{ 0, 2738 },
		{ 0, 1202 },
		{ 0, 2347 },
		{ 0, 2672 },
		{ 0, 2348 },
		{ 2516, 2516 },
		{ 2516, 2516 },
		{ 0, 2349 },
		{ 2293, 2293 },
		{ 2293, 2293 },
		{ 2505, 2505 },
		{ 0, 2824 },
		{ 0, 2451 },
		{ 1911, 1911 },
		{ 0, 2747 },
		{ 2597, 2597 },
		{ 0, 2151 },
		{ 0, 2828 },
		{ 2664, 2664 },
		{ 2612, 2612 },
		{ 2612, 2612 },
		{ 0, 1675 },
		{ 0, 2832 },
		{ 2137, 2137 },
		{ 2138, 2137 },
		{ 0, 2456 },
		{ 0, 2835 },
		{ 0, 2400 },
		{ 2516, 2516 },
		{ 2566, 2566 },
		{ 2566, 2566 },
		{ 2293, 2293 },
		{ 0, 2738 },
		{ 0, 2687 },
		{ 2526, 2526 },
		{ 2526, 2526 },
		{ 0, 2840 },
		{ 2621, 2621 },
		{ 2621, 2621 },
		{ 0, 2622 },
		{ 2623, 2623 },
		{ 2623, 2623 },
		{ 2612, 2612 },
		{ 0, 2692 },
		{ 0, 2763 },
		{ 0, 2265 },
		{ 2137, 2137 },
		{ 0, 2491 },
		{ 0, 2459 },
		{ 0, 1180 },
		{ 2628, 2628 },
		{ 2628, 2628 },
		{ 2566, 2566 },
		{ 0, 2851 },
		{ 0, 2357 },
		{ 0, 2573 },
		{ 0, 2381 },
		{ 2526, 2526 },
		{ 171, 171 },
		{ 172, 171 },
		{ 2621, 2621 },
		{ 0, 2703 },
		{ 0, 2357 },
		{ 2623, 2623 },
		{ 2575, 2575 },
		{ 2575, 2575 },
		{ 2432, 2432 },
		{ 2432, 2432 },
		{ 0, 2777 },
		{ 2635, 2635 },
		{ 2635, 2635 },
		{ 2636, 2636 },
		{ 2636, 2636 },
		{ 2628, 2628 },
		{ 2637, 2637 },
		{ 2637, 2637 },
		{ 0, 2265 },
		{ 0, 2781 },
		{ 0, 2498 },
		{ 0, 1686 },
		{ 0, 2710 },
		{ 171, 171 },
		{ 0, 2538 },
		{ 0, 2581 },
		{ 0, 2870 },
		{ 2642, 2642 },
		{ 2642, 2642 },
		{ 2575, 2575 },
		{ 0, 1218 },
		{ 2432, 2432 },
		{ 2582, 2582 },
		{ 2582, 2582 },
		{ 2635, 2635 },
		{ 0, 2584 },
		{ 2636, 2636 },
		{ 0, 2791 },
		{ 0, 2717 },
		{ 2637, 2637 },
		{ 0, 1213 },
		{ 0, 2794 },
		{ 0, 2586 },
		{ 0, 2587 },
		{ 0, 2881 },
		{ 0, 2541 },
		{ 2650, 2650 },
		{ 2650, 2650 },
		{ 0, 2799 },
		{ 0, 2468 },
		{ 2642, 2642 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 1159, 1159 },
		{ 1289, 1288 },
		{ 2582, 2582 },
		{ 0, 1936 },
		{ 2658, 2654 },
		{ 1905, 1907 },
		{ 165, 167 },
		{ 1905, 1901 },
		{ 0, 2186 },
		{ 2658, 2660 },
		{ 2131, 2133 },
		{ 0, 87 },
		{ 1806, 1787 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2650, 2650 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2409, 2409 },
		{ 0, 0 },
		{ 1159, 1159 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -68, 16, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 44, 572, 0 },
		{ -177, 2740, 0 },
		{ 5, 0, 0 },
		{ -1158, 1017, -31 },
		{ 7, 0, -31 },
		{ -1162, 1805, -33 },
		{ 9, 0, -33 },
		{ -1175, 3056, 139 },
		{ 11, 0, 139 },
		{ -1196, 3181, 147 },
		{ 13, 0, 147 },
		{ -1230, 3176, 0 },
		{ 15, 0, 0 },
		{ -1245, 3058, 135 },
		{ 17, 0, 135 },
		{ -1273, 3073, 22 },
		{ 19, 0, 22 },
		{ -1315, 230, 0 },
		{ 21, 0, 0 },
		{ -1541, 3179, 0 },
		{ 23, 0, 0 },
		{ -1568, 3069, 0 },
		{ 25, 0, 0 },
		{ -1594, 3057, 0 },
		{ 27, 0, 0 },
		{ -1618, 3063, 0 },
		{ 29, 0, 0 },
		{ -1644, 3064, 0 },
		{ 31, 0, 0 },
		{ -1670, 3183, 150 },
		{ 33, 0, 150 },
		{ -1703, 3180, 254 },
		{ 35, 0, 254 },
		{ 38, 127, 0 },
		{ -1739, 344, 0 },
		{ 40, 16, 0 },
		{ -1928, 116, 0 },
		{ -2140, 3186, 0 },
		{ 41, 0, 0 },
		{ 44, 14, 0 },
		{ -86, 458, 0 },
		{ -2667, 3062, 143 },
		{ 45, 0, 143 },
		{ -2685, 3182, 162 },
		{ 47, 0, 162 },
		{ 2729, 1503, 0 },
		{ 49, 0, 0 },
		{ -2731, 3177, 260 },
		{ 51, 0, 260 },
		{ -2758, 3184, 165 },
		{ 53, 0, 165 },
		{ -2775, 3178, 159 },
		{ 55, 0, 159 },
		{ -2822, 3175, 153 },
		{ 57, 0, 153 },
		{ -2863, 3185, 158 },
		{ 59, 0, 158 },
		{ -86, 1, 0 },
		{ 61, 0, 0 },
		{ -2910, 1765, 0 },
		{ 63, 0, 0 },
		{ -2920, 1674, 42 },
		{ 65, 0, 42 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 409 },
		{ 2663, 4553, 416 },
		{ 0, 0, 232 },
		{ 0, 0, 234 },
		{ 157, 1270, 251 },
		{ 157, 1383, 251 },
		{ 157, 1279, 251 },
		{ 157, 1286, 251 },
		{ 157, 1286, 251 },
		{ 157, 1290, 251 },
		{ 157, 1295, 251 },
		{ 157, 641, 251 },
		{ 2969, 2800, 416 },
		{ 157, 652, 251 },
		{ 2969, 1619, 250 },
		{ 102, 2462, 416 },
		{ 157, 0, 251 },
		{ 0, 0, 416 },
		{ -87, 4767, 228 },
		{ -88, 4598, 0 },
		{ 157, 643, 251 },
		{ 157, 677, 251 },
		{ 157, 646, 251 },
		{ 157, 660, 251 },
		{ 157, 678, 251 },
		{ 157, 678, 251 },
		{ 157, 685, 251 },
		{ 157, 700, 251 },
		{ 157, 693, 251 },
		{ 2938, 2278, 0 },
		{ 157, 683, 251 },
		{ 2969, 1751, 247 },
		{ 117, 1385, 0 },
		{ 2969, 1741, 248 },
		{ 2663, 4575, 0 },
		{ 157, 692, 251 },
		{ 157, 716, 251 },
		{ 157, 717, 251 },
		{ 157, 713, 251 },
		{ 157, 0, 239 },
		{ 157, 732, 251 },
		{ 157, 735, 251 },
		{ 157, 755, 251 },
		{ 157, 760, 251 },
		{ 2966, 2858, 0 },
		{ 157, 767, 251 },
		{ 131, 1419, 0 },
		{ 117, 0, 0 },
		{ 2895, 2630, 249 },
		{ 133, 1436, 0 },
		{ 0, 0, 230 },
		{ 157, 771, 235 },
		{ 157, 773, 251 },
		{ 157, 765, 251 },
		{ 157, 770, 251 },
		{ 157, 794, 251 },
		{ 157, 787, 251 },
		{ 157, 0, 242 },
		{ 157, 788, 251 },
		{ 0, 0, 244 },
		{ 157, 794, 251 },
		{ 131, 0, 0 },
		{ 2895, 2517, 247 },
		{ 133, 0, 0 },
		{ 2895, 2586, 248 },
		{ 157, 810, 251 },
		{ 157, 835, 251 },
		{ 157, 836, 251 },
		{ 157, 862, 251 },
		{ 157, 930, 251 },
		{ 157, 0, 241 },
		{ 157, 976, 251 },
		{ 157, 980, 251 },
		{ 157, 1001, 251 },
		{ 157, 0, 237 },
		{ 157, 1118, 251 },
		{ 157, 0, 238 },
		{ 157, 0, 240 },
		{ 157, 1192, 251 },
		{ 157, 1220, 251 },
		{ 157, 0, 236 },
		{ 157, 1268, 251 },
		{ 157, 0, 243 },
		{ 157, 667, 251 },
		{ 157, 1285, 251 },
		{ 0, 0, 246 },
		{ 157, 1268, 251 },
		{ 157, 1269, 251 },
		{ 2986, 1337, 245 },
		{ 2663, 4561, 416 },
		{ 163, 0, 232 },
		{ 0, 0, 233 },
		{ -161, 20, 228 },
		{ -162, 4596, 0 },
		{ 2941, 4574, 0 },
		{ 2663, 4549, 0 },
		{ 0, 0, 229 },
		{ 2663, 4576, 0 },
		{ -167, 4762, 0 },
		{ -168, 4591, 0 },
		{ 171, 0, 230 },
		{ 2663, 4577, 0 },
		{ 2941, 4704, 0 },
		{ 0, 0, 231 },
		{ 2942, 1554, 133 },
		{ 2040, 3951, 133 },
		{ 2821, 4265, 133 },
		{ 2942, 4154, 133 },
		{ 0, 0, 133 },
		{ 2930, 3301, 0 },
		{ 2055, 2917, 0 },
		{ 2930, 3495, 0 },
		{ 2907, 3242, 0 },
		{ 2907, 3241, 0 },
		{ 2040, 3996, 0 },
		{ 2934, 3152, 0 },
		{ 2040, 3961, 0 },
		{ 2908, 3473, 0 },
		{ 2937, 3166, 0 },
		{ 2908, 3206, 0 },
		{ 2757, 4192, 0 },
		{ 2934, 3181, 0 },
		{ 2055, 3576, 0 },
		{ 2821, 4301, 0 },
		{ 1986, 3371, 0 },
		{ 1986, 3347, 0 },
		{ 2008, 3082, 0 },
		{ 1989, 3688, 0 },
		{ 1970, 3742, 0 },
		{ 1989, 3699, 0 },
		{ 2008, 3084, 0 },
		{ 2008, 3001, 0 },
		{ 2934, 3215, 0 },
		{ 2862, 3833, 0 },
		{ 2040, 3983, 0 },
		{ 1128, 3886, 0 },
		{ 1986, 3360, 0 },
		{ 2008, 3010, 0 },
		{ 2821, 4381, 0 },
		{ 1986, 3385, 0 },
		{ 2907, 3650, 0 },
		{ 2930, 3506, 0 },
		{ 2055, 3602, 0 },
		{ 2139, 4043, 0 },
		{ 2821, 3913, 0 },
		{ 2862, 3799, 0 },
		{ 1970, 3715, 0 },
		{ 2908, 3448, 0 },
		{ 2821, 4239, 0 },
		{ 2930, 3513, 0 },
		{ 2862, 3495, 0 },
		{ 2055, 3580, 0 },
		{ 2008, 3012, 0 },
		{ 2937, 3319, 0 },
		{ 2907, 3674, 0 },
		{ 1948, 3174, 0 },
		{ 1970, 3738, 0 },
		{ 2821, 4255, 0 },
		{ 2040, 3963, 0 },
		{ 2040, 3979, 0 },
		{ 2930, 3559, 0 },
		{ 2008, 3027, 0 },
		{ 2040, 3999, 0 },
		{ 2930, 3558, 0 },
		{ 2139, 4037, 0 },
		{ 2821, 4323, 0 },
		{ 2937, 3327, 0 },
		{ 2908, 3425, 0 },
		{ 2008, 3065, 0 },
		{ 2937, 3217, 0 },
		{ 2930, 3502, 0 },
		{ 1970, 3734, 0 },
		{ 2862, 3835, 0 },
		{ 2055, 3518, 0 },
		{ 1020, 3145, 0 },
		{ 2930, 3560, 0 },
		{ 2907, 3633, 0 },
		{ 2821, 4315, 0 },
		{ 2139, 4066, 0 },
		{ 2008, 3067, 0 },
		{ 2937, 3304, 0 },
		{ 2040, 3910, 0 },
		{ 1948, 3175, 0 },
		{ 2908, 3432, 0 },
		{ 2008, 3068, 0 },
		{ 2757, 4193, 0 },
		{ 2907, 3632, 0 },
		{ 2937, 3270, 0 },
		{ 2076, 3513, 0 },
		{ 2008, 3069, 0 },
		{ 2862, 3801, 0 },
		{ 2040, 3915, 0 },
		{ 2139, 4029, 0 },
		{ 2040, 3926, 0 },
		{ 2821, 4387, 0 },
		{ 2821, 4395, 0 },
		{ 1970, 3741, 0 },
		{ 2139, 4058, 0 },
		{ 2008, 3071, 0 },
		{ 2821, 4270, 0 },
		{ 2862, 3845, 0 },
		{ 1970, 3705, 0 },
		{ 2862, 3769, 0 },
		{ 2821, 4337, 0 },
		{ 1986, 3381, 0 },
		{ 2908, 3469, 0 },
		{ 2040, 4000, 0 },
		{ 2821, 4237, 0 },
		{ 1128, 3900, 0 },
		{ 1020, 3137, 0 },
		{ 2076, 3867, 0 },
		{ 1989, 3681, 0 },
		{ 2908, 3475, 0 },
		{ 2008, 3072, 0 },
		{ 2821, 4321, 0 },
		{ 2040, 3970, 0 },
		{ 2008, 3073, 0 },
		{ 0, 0, 69 },
		{ 2930, 3286, 0 },
		{ 2040, 3993, 0 },
		{ 2942, 4125, 0 },
		{ 2008, 3074, 0 },
		{ 2937, 3302, 0 },
		{ 1986, 3342, 0 },
		{ 1970, 3743, 0 },
		{ 2937, 3303, 0 },
		{ 2008, 3075, 0 },
		{ 2040, 3946, 0 },
		{ 2930, 3521, 0 },
		{ 2930, 3539, 0 },
		{ 1989, 3678, 0 },
		{ 2908, 3440, 0 },
		{ 802, 3156, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 1986, 3377, 0 },
		{ 2821, 4397, 0 },
		{ 2862, 3802, 0 },
		{ 1970, 3709, 0 },
		{ 2937, 3326, 0 },
		{ 2908, 3474, 0 },
		{ 0, 0, 67 },
		{ 2008, 3076, 0 },
		{ 1986, 3330, 0 },
		{ 2937, 3328, 0 },
		{ 2862, 3823, 0 },
		{ 2937, 3245, 0 },
		{ 2821, 4331, 0 },
		{ 2908, 3447, 0 },
		{ 1128, 3887, 0 },
		{ 1986, 3359, 0 },
		{ 2907, 3649, 0 },
		{ 2040, 3980, 0 },
		{ 2821, 4225, 0 },
		{ 2942, 4156, 0 },
		{ 2908, 3454, 0 },
		{ 0, 0, 63 },
		{ 2908, 3455, 0 },
		{ 2821, 4261, 0 },
		{ 1128, 3905, 0 },
		{ 2862, 3815, 0 },
		{ 0, 0, 72 },
		{ 2937, 3269, 0 },
		{ 2930, 3500, 0 },
		{ 2008, 3077, 0 },
		{ 2862, 3840, 0 },
		{ 2040, 3932, 0 },
		{ 2008, 3078, 0 },
		{ 1986, 3380, 0 },
		{ 2907, 3625, 0 },
		{ 2937, 3273, 0 },
		{ 2908, 3426, 0 },
		{ 2821, 4407, 0 },
		{ 2008, 3079, 0 },
		{ 2862, 3830, 0 },
		{ 2040, 3981, 0 },
		{ 2937, 3298, 0 },
		{ 2908, 3445, 0 },
		{ 2862, 3839, 0 },
		{ 963, 3851, 0 },
		{ 0, 0, 8 },
		{ 1986, 3341, 0 },
		{ 1989, 3689, 0 },
		{ 2862, 3798, 0 },
		{ 2021, 3116, 0 },
		{ 2008, 3080, 0 },
		{ 2139, 4060, 0 },
		{ 2040, 3939, 0 },
		{ 1986, 3357, 0 },
		{ 2040, 3947, 0 },
		{ 1989, 3682, 0 },
		{ 2008, 3081, 0 },
		{ 2937, 3312, 0 },
		{ 2937, 3164, 0 },
		{ 2040, 3977, 0 },
		{ 2934, 3206, 0 },
		{ 2908, 3422, 0 },
		{ 1128, 3885, 0 },
		{ 2907, 3640, 0 },
		{ 2055, 2986, 0 },
		{ 2821, 4305, 0 },
		{ 1128, 3895, 0 },
		{ 2942, 3143, 0 },
		{ 2021, 3121, 0 },
		{ 1989, 3701, 0 },
		{ 1986, 3388, 0 },
		{ 2937, 3248, 0 },
		{ 0, 0, 113 },
		{ 2008, 3083, 0 },
		{ 2055, 3587, 0 },
		{ 1947, 3150, 0 },
		{ 2930, 3509, 0 },
		{ 2907, 3653, 0 },
		{ 2821, 4227, 0 },
		{ 2040, 3958, 0 },
		{ 0, 0, 7 },
		{ 1989, 3702, 0 },
		{ 0, 0, 6 },
		{ 2862, 3841, 0 },
		{ 0, 0, 118 },
		{ 2907, 3664, 0 },
		{ 2040, 3976, 0 },
		{ 2942, 1612, 0 },
		{ 2008, 3085, 0 },
		{ 2907, 3630, 0 },
		{ 2930, 3538, 0 },
		{ 2008, 3086, 0 },
		{ 2040, 3984, 0 },
		{ 2942, 3137, 0 },
		{ 2040, 3994, 0 },
		{ 2139, 4027, 0 },
		{ 2055, 3583, 0 },
		{ 0, 0, 68 },
		{ 1970, 3748, 0 },
		{ 2008, 3087, 105 },
		{ 2008, 3088, 106 },
		{ 2821, 4434, 0 },
		{ 2908, 3480, 0 },
		{ 2930, 3510, 0 },
		{ 2908, 3419, 0 },
		{ 1128, 3883, 0 },
		{ 2934, 3232, 0 },
		{ 2055, 3594, 0 },
		{ 2862, 3800, 0 },
		{ 2040, 3953, 0 },
		{ 2008, 3089, 0 },
		{ 2937, 3316, 0 },
		{ 2821, 4307, 0 },
		{ 2862, 3808, 0 },
		{ 2757, 4194, 0 },
		{ 2862, 3809, 0 },
		{ 2908, 3430, 0 },
		{ 2862, 3818, 0 },
		{ 0, 0, 9 },
		{ 2008, 3090, 0 },
		{ 2862, 3828, 0 },
		{ 2021, 3123, 0 },
		{ 2076, 3878, 0 },
		{ 0, 0, 103 },
		{ 1986, 3334, 0 },
		{ 2907, 3672, 0 },
		{ 2930, 3498, 0 },
		{ 2055, 3516, 0 },
		{ 2907, 3144, 0 },
		{ 2862, 3843, 0 },
		{ 2934, 3194, 0 },
		{ 2862, 3797, 0 },
		{ 2934, 3173, 0 },
		{ 2930, 3546, 0 },
		{ 2040, 3936, 0 },
		{ 2937, 3246, 0 },
		{ 2908, 3466, 0 },
		{ 2934, 3161, 0 },
		{ 2907, 3658, 0 },
		{ 2937, 3260, 0 },
		{ 2862, 3770, 0 },
		{ 2937, 3265, 0 },
		{ 2821, 4375, 0 },
		{ 2008, 3091, 0 },
		{ 2821, 4383, 0 },
		{ 2908, 3478, 0 },
		{ 2040, 3974, 0 },
		{ 2930, 3491, 0 },
		{ 2930, 3493, 0 },
		{ 1970, 3729, 0 },
		{ 2008, 3092, 93 },
		{ 2908, 3413, 0 },
		{ 2008, 2983, 0 },
		{ 2008, 2990, 0 },
		{ 2139, 4035, 0 },
		{ 2008, 2992, 0 },
		{ 1986, 3389, 0 },
		{ 0, 0, 102 },
		{ 2139, 4056, 0 },
		{ 2821, 4297, 0 },
		{ 2937, 3291, 0 },
		{ 2937, 3294, 0 },
		{ 0, 0, 115 },
		{ 0, 0, 117 },
		{ 2055, 3589, 0 },
		{ 1986, 3335, 0 },
		{ 2040, 3252, 0 },
		{ 2937, 3299, 0 },
		{ 2040, 3929, 0 },
		{ 2008, 2993, 0 },
		{ 2040, 3933, 0 },
		{ 2862, 3810, 0 },
		{ 2907, 3635, 0 },
		{ 2008, 2994, 0 },
		{ 2076, 3864, 0 },
		{ 2934, 3225, 0 },
		{ 2139, 4031, 0 },
		{ 2008, 2996, 0 },
		{ 2821, 4443, 0 },
		{ 1986, 3362, 0 },
		{ 895, 3758, 0 },
		{ 2937, 3314, 0 },
		{ 2907, 3662, 0 },
		{ 2055, 3614, 0 },
		{ 2139, 4064, 0 },
		{ 2937, 3315, 0 },
		{ 1948, 3171, 0 },
		{ 2008, 2997, 0 },
		{ 2862, 3789, 0 },
		{ 2930, 3544, 0 },
		{ 1986, 3384, 0 },
		{ 2821, 4311, 0 },
		{ 2937, 3325, 0 },
		{ 2055, 3596, 0 },
		{ 2021, 3124, 0 },
		{ 2908, 3421, 0 },
		{ 2055, 3570, 0 },
		{ 1989, 3696, 0 },
		{ 2942, 3213, 0 },
		{ 2008, 2998, 0 },
		{ 0, 0, 64 },
		{ 2008, 2999, 0 },
		{ 2930, 3519, 0 },
		{ 2908, 3427, 0 },
		{ 2930, 3529, 0 },
		{ 2908, 3429, 0 },
		{ 2008, 3000, 107 },
		{ 2055, 3612, 0 },
		{ 1989, 3698, 0 },
		{ 1986, 3336, 0 },
		{ 1986, 3340, 0 },
		{ 2821, 4253, 0 },
		{ 2930, 3489, 0 },
		{ 2934, 3230, 0 },
		{ 2862, 3790, 0 },
		{ 2937, 3252, 0 },
		{ 1986, 3343, 0 },
		{ 0, 0, 119 },
		{ 2757, 4188, 0 },
		{ 1989, 3687, 0 },
		{ 2937, 3256, 0 },
		{ 2907, 3659, 0 },
		{ 0, 0, 114 },
		{ 0, 0, 104 },
		{ 1986, 3356, 0 },
		{ 2908, 3464, 0 },
		{ 2937, 3259, 0 },
		{ 2055, 2904, 0 },
		{ 2942, 3173, 0 },
		{ 2862, 3817, 0 },
		{ 2907, 3626, 0 },
		{ 2008, 3002, 0 },
		{ 2862, 3825, 0 },
		{ 1970, 3712, 0 },
		{ 2930, 3542, 0 },
		{ 2040, 3921, 0 },
		{ 2821, 4414, 0 },
		{ 2821, 4432, 0 },
		{ 1986, 3363, 0 },
		{ 2821, 4441, 0 },
		{ 2862, 3834, 0 },
		{ 2821, 4223, 0 },
		{ 2908, 3476, 0 },
		{ 2907, 3639, 0 },
		{ 2008, 3003, 0 },
		{ 2040, 3937, 0 },
		{ 2908, 3479, 0 },
		{ 2008, 3006, 0 },
		{ 2908, 3481, 0 },
		{ 2040, 3948, 0 },
		{ 2862, 3781, 0 },
		{ 2908, 3407, 0 },
		{ 2040, 3955, 0 },
		{ 1986, 3378, 0 },
		{ 2907, 3660, 0 },
		{ 2008, 3007, 0 },
		{ 2942, 4049, 0 },
		{ 2139, 4041, 0 },
		{ 2040, 3969, 0 },
		{ 1989, 3685, 0 },
		{ 2040, 3973, 0 },
		{ 1989, 3686, 0 },
		{ 2930, 3497, 0 },
		{ 2930, 3525, 0 },
		{ 0, 0, 95 },
		{ 2862, 3803, 0 },
		{ 2862, 3805, 0 },
		{ 2008, 3008, 0 },
		{ 2821, 4399, 0 },
		{ 2821, 4401, 0 },
		{ 2821, 4405, 0 },
		{ 1989, 3693, 0 },
		{ 1986, 3383, 0 },
		{ 0, 0, 122 },
		{ 0, 0, 116 },
		{ 0, 0, 120 },
		{ 2821, 4431, 0 },
		{ 2139, 4039, 0 },
		{ 1020, 3138, 0 },
		{ 2008, 3009, 0 },
		{ 2862, 2989, 0 },
		{ 2908, 3428, 0 },
		{ 1989, 3677, 0 },
		{ 1128, 3889, 0 },
		{ 2821, 4231, 0 },
		{ 2930, 3549, 0 },
		{ 2930, 3554, 0 },
		{ 2930, 3557, 0 },
		{ 2907, 3641, 0 },
		{ 2139, 4033, 0 },
		{ 2076, 3865, 0 },
		{ 2907, 3648, 0 },
		{ 2934, 3231, 0 },
		{ 1970, 3751, 0 },
		{ 1128, 3892, 0 },
		{ 2937, 3300, 0 },
		{ 1970, 3706, 0 },
		{ 1986, 3333, 0 },
		{ 2008, 3011, 0 },
		{ 1989, 3694, 0 },
		{ 1970, 3725, 0 },
		{ 2040, 3957, 0 },
		{ 2076, 3869, 0 },
		{ 2055, 3585, 0 },
		{ 2908, 3443, 0 },
		{ 2821, 4385, 0 },
		{ 2055, 3588, 0 },
		{ 0, 0, 57 },
		{ 0, 0, 58 },
		{ 2821, 4393, 0 },
		{ 0, 0, 66 },
		{ 0, 0, 111 },
		{ 1948, 3176, 0 },
		{ 2934, 3208, 0 },
		{ 1986, 3337, 0 },
		{ 2934, 3212, 0 },
		{ 2937, 3313, 0 },
		{ 2862, 3806, 0 },
		{ 2908, 3461, 0 },
		{ 0, 0, 97 },
		{ 2908, 3462, 0 },
		{ 0, 0, 99 },
		{ 2930, 3541, 0 },
		{ 2908, 3463, 0 },
		{ 2040, 3991, 0 },
		{ 2021, 3126, 0 },
		{ 2021, 3127, 0 },
		{ 2076, 3625, 0 },
		{ 2908, 3467, 0 },
		{ 895, 3760, 0 },
		{ 1970, 3736, 0 },
		{ 0, 0, 112 },
		{ 0, 0, 121 },
		{ 2908, 3468, 0 },
		{ 0, 0, 132 },
		{ 1986, 3345, 0 },
		{ 2942, 3777, 0 },
		{ 2821, 4257, 0 },
		{ 1128, 3902, 0 },
		{ 2821, 4263, 0 },
		{ 2040, 3931, 0 },
		{ 2942, 4120, 0 },
		{ 2908, 3470, 0 },
		{ 2942, 4126, 0 },
		{ 2934, 3226, 0 },
		{ 2934, 3229, 0 },
		{ 2907, 2985, 0 },
		{ 2008, 3013, 0 },
		{ 2040, 3945, 0 },
		{ 2862, 3777, 0 },
		{ 2821, 4325, 0 },
		{ 2821, 4327, 0 },
		{ 2862, 3780, 0 },
		{ 2055, 3616, 0 },
		{ 2862, 3788, 0 },
		{ 2055, 3562, 0 },
		{ 2055, 3519, 0 },
		{ 2862, 3795, 0 },
		{ 2008, 3014, 0 },
		{ 2139, 4070, 0 },
		{ 2008, 3015, 0 },
		{ 2930, 3522, 0 },
		{ 2008, 3016, 0 },
		{ 1989, 3691, 0 },
		{ 2930, 3527, 0 },
		{ 1970, 3739, 0 },
		{ 2008, 3017, 0 },
		{ 2930, 3535, 0 },
		{ 2942, 4167, 0 },
		{ 1128, 3906, 0 },
		{ 2055, 3590, 0 },
		{ 2908, 3402, 0 },
		{ 2821, 4449, 0 },
		{ 2821, 4221, 0 },
		{ 2040, 3982, 0 },
		{ 1989, 3700, 0 },
		{ 2908, 3403, 0 },
		{ 2040, 3985, 0 },
		{ 2040, 3987, 0 },
		{ 2040, 3988, 0 },
		{ 2821, 4241, 0 },
		{ 2821, 4249, 0 },
		{ 2040, 3989, 0 },
		{ 2008, 3020, 0 },
		{ 2937, 3249, 0 },
		{ 2937, 3250, 0 },
		{ 2040, 3995, 0 },
		{ 1970, 3714, 0 },
		{ 2934, 3223, 0 },
		{ 1970, 3719, 0 },
		{ 2942, 4157, 0 },
		{ 2937, 3254, 0 },
		{ 2008, 3023, 61 },
		{ 2937, 3258, 0 },
		{ 2821, 4313, 0 },
		{ 2055, 3579, 0 },
		{ 2008, 3024, 0 },
		{ 2008, 3025, 0 },
		{ 2076, 3868, 0 },
		{ 2862, 3842, 0 },
		{ 2942, 4116, 0 },
		{ 2907, 3619, 0 },
		{ 2937, 3261, 0 },
		{ 2937, 3262, 0 },
		{ 1020, 3132, 0 },
		{ 1970, 3750, 0 },
		{ 2908, 3434, 0 },
		{ 2021, 3119, 0 },
		{ 1948, 3197, 0 },
		{ 1948, 3170, 0 },
		{ 2930, 3524, 0 },
		{ 1986, 3339, 0 },
		{ 1128, 3899, 0 },
		{ 2934, 3207, 0 },
		{ 2908, 3451, 0 },
		{ 2942, 4158, 0 },
		{ 2942, 4159, 0 },
		{ 2040, 3966, 0 },
		{ 0, 0, 62 },
		{ 0, 0, 60 },
		{ 1128, 3903, 0 },
		{ 1970, 3720, 0 },
		{ 2908, 3452, 0 },
		{ 1128, 3882, 0 },
		{ 2908, 3453, 0 },
		{ 0, 0, 108 },
		{ 2937, 3274, 0 },
		{ 2937, 3278, 0 },
		{ 2908, 3456, 0 },
		{ 0, 0, 101 },
		{ 2821, 4233, 0 },
		{ 0, 0, 109 },
		{ 0, 0, 110 },
		{ 2907, 3661, 0 },
		{ 895, 3762, 0 },
		{ 1989, 3690, 0 },
		{ 1128, 3898, 0 },
		{ 2937, 3279, 0 },
		{ 0, 0, 3 },
		{ 2040, 3986, 0 },
		{ 2942, 4139, 0 },
		{ 2821, 4259, 0 },
		{ 2907, 3663, 0 },
		{ 2862, 3819, 0 },
		{ 2937, 3280, 0 },
		{ 2862, 3824, 0 },
		{ 2040, 3992, 0 },
		{ 2008, 3026, 0 },
		{ 1989, 3697, 0 },
		{ 2139, 4045, 0 },
		{ 1986, 3348, 0 },
		{ 1986, 3349, 0 },
		{ 2040, 3998, 0 },
		{ 2907, 3621, 0 },
		{ 963, 3852, 0 },
		{ 2040, 2995, 0 },
		{ 2862, 3838, 0 },
		{ 2055, 3592, 0 },
		{ 0, 0, 70 },
		{ 2040, 3922, 0 },
		{ 0, 0, 78 },
		{ 2821, 4369, 0 },
		{ 2040, 3923, 0 },
		{ 2821, 4379, 0 },
		{ 2937, 3293, 0 },
		{ 2040, 3928, 0 },
		{ 2934, 3214, 0 },
		{ 2942, 4131, 0 },
		{ 2040, 3930, 0 },
		{ 2055, 3598, 0 },
		{ 1970, 3724, 0 },
		{ 2937, 3295, 0 },
		{ 2862, 3767, 0 },
		{ 2055, 3603, 0 },
		{ 2862, 3779, 0 },
		{ 2040, 3942, 0 },
		{ 0, 0, 65 },
		{ 2055, 3605, 0 },
		{ 2055, 3607, 0 },
		{ 2821, 4433, 0 },
		{ 2937, 3296, 0 },
		{ 2055, 3613, 0 },
		{ 2040, 3950, 0 },
		{ 2008, 3028, 0 },
		{ 2862, 3793, 0 },
		{ 2930, 3514, 0 },
		{ 1989, 3692, 0 },
		{ 1970, 3744, 0 },
		{ 1986, 3366, 0 },
		{ 2942, 4175, 0 },
		{ 1986, 3367, 0 },
		{ 2008, 3029, 0 },
		{ 2055, 3578, 0 },
		{ 1948, 3178, 0 },
		{ 2942, 4128, 0 },
		{ 2040, 3971, 0 },
		{ 2040, 3972, 0 },
		{ 802, 3152, 0 },
		{ 0, 3155, 0 },
		{ 2907, 3670, 0 },
		{ 2076, 3862, 0 },
		{ 2040, 3978, 0 },
		{ 2908, 3482, 0 },
		{ 1128, 3888, 0 },
		{ 2821, 4303, 0 },
		{ 2908, 3401, 0 },
		{ 2008, 3030, 0 },
		{ 2937, 3305, 0 },
		{ 2908, 3404, 0 },
		{ 1970, 3727, 0 },
		{ 2862, 3822, 0 },
		{ 2908, 3406, 0 },
		{ 2907, 3629, 0 },
		{ 2937, 3306, 0 },
		{ 2139, 4107, 0 },
		{ 2139, 4025, 0 },
		{ 2821, 4342, 0 },
		{ 2040, 3990, 0 },
		{ 1970, 3737, 0 },
		{ 2937, 3307, 0 },
		{ 2930, 3547, 0 },
		{ 2908, 3414, 0 },
		{ 2908, 3415, 0 },
		{ 2908, 3417, 0 },
		{ 2937, 3308, 0 },
		{ 2055, 3610, 0 },
		{ 1989, 3695, 0 },
		{ 1128, 3893, 0 },
		{ 2937, 3309, 0 },
		{ 1970, 3753, 0 },
		{ 1970, 3704, 0 },
		{ 0, 0, 10 },
		{ 2821, 4409, 0 },
		{ 1986, 3390, 0 },
		{ 2937, 3310, 0 },
		{ 2821, 3885, 0 },
		{ 2942, 4114, 0 },
		{ 2907, 3654, 0 },
		{ 2821, 4435, 0 },
		{ 2821, 4436, 0 },
		{ 2937, 3311, 0 },
		{ 2008, 3032, 0 },
		{ 2862, 3783, 0 },
		{ 2862, 3786, 0 },
		{ 2040, 3934, 0 },
		{ 2008, 3033, 0 },
		{ 2942, 4141, 0 },
		{ 2821, 4229, 0 },
		{ 2942, 4149, 0 },
		{ 1970, 3716, 0 },
		{ 0, 0, 79 },
		{ 2055, 3573, 0 },
		{ 2862, 3791, 0 },
		{ 0, 0, 77 },
		{ 2934, 3233, 0 },
		{ 1989, 3679, 0 },
		{ 0, 0, 80 },
		{ 2942, 4160, 0 },
		{ 2862, 3796, 0 },
		{ 2934, 3202, 0 },
		{ 2040, 3949, 0 },
		{ 1986, 3338, 0 },
		{ 2908, 3433, 0 },
		{ 2040, 3952, 0 },
		{ 2008, 3034, 0 },
		{ 2937, 3317, 0 },
		{ 0, 0, 59 },
		{ 0, 0, 96 },
		{ 0, 0, 98 },
		{ 2055, 3586, 0 },
		{ 2908, 3441, 0 },
		{ 2040, 3959, 0 },
		{ 2862, 3804, 0 },
		{ 2930, 3531, 0 },
		{ 2040, 3965, 0 },
		{ 0, 0, 130 },
		{ 2908, 3442, 0 },
		{ 2040, 3968, 0 },
		{ 2862, 3807, 0 },
		{ 2937, 3318, 0 },
		{ 2908, 3444, 0 },
		{ 2821, 4329, 0 },
		{ 2008, 3036, 0 },
		{ 1970, 3746, 0 },
		{ 1970, 3747, 0 },
		{ 2040, 3975, 0 },
		{ 2139, 4099, 0 },
		{ 2937, 3320, 0 },
		{ 2937, 3321, 0 },
		{ 2908, 3450, 0 },
		{ 2076, 3857, 0 },
		{ 0, 3756, 0 },
		{ 2937, 3322, 0 },
		{ 2937, 3323, 0 },
		{ 2862, 3826, 0 },
		{ 2930, 3550, 0 },
		{ 2055, 3608, 0 },
		{ 2821, 4403, 0 },
		{ 2862, 3831, 0 },
		{ 2937, 3324, 0 },
		{ 2055, 3611, 0 },
		{ 2942, 4171, 0 },
		{ 0, 0, 19 },
		{ 1986, 3350, 0 },
		{ 1986, 3352, 0 },
		{ 0, 0, 123 },
		{ 1986, 3353, 0 },
		{ 0, 0, 125 },
		{ 2908, 3459, 0 },
		{ 0, 0, 94 },
		{ 2008, 3037, 0 },
		{ 1970, 3721, 0 },
		{ 1970, 3723, 0 },
		{ 2008, 3038, 0 },
		{ 2821, 4447, 0 },
		{ 1986, 3358, 0 },
		{ 2055, 3574, 0 },
		{ 2862, 3778, 0 },
		{ 0, 0, 75 },
		{ 1970, 3728, 0 },
		{ 1128, 3907, 0 },
		{ 2008, 3039, 0 },
		{ 1970, 3731, 0 },
		{ 2908, 3465, 0 },
		{ 2040, 3924, 0 },
		{ 2942, 4161, 0 },
		{ 2942, 4162, 0 },
		{ 2821, 4243, 0 },
		{ 2040, 3925, 0 },
		{ 2862, 3784, 0 },
		{ 2862, 3785, 0 },
		{ 2008, 3040, 0 },
		{ 1986, 3361, 0 },
		{ 2937, 3238, 0 },
		{ 2907, 3622, 0 },
		{ 2937, 3239, 0 },
		{ 1986, 3364, 0 },
		{ 2862, 3794, 0 },
		{ 2907, 3627, 0 },
		{ 2937, 3242, 0 },
		{ 2040, 3940, 0 },
		{ 0, 0, 83 },
		{ 2942, 4151, 0 },
		{ 0, 0, 100 },
		{ 2942, 3774, 0 },
		{ 2040, 3944, 0 },
		{ 0, 0, 128 },
		{ 2937, 3243, 0 },
		{ 2937, 3244, 0 },
		{ 2008, 3042, 56 },
		{ 2907, 3634, 0 },
		{ 2055, 3591, 0 },
		{ 1970, 3752, 0 },
		{ 2934, 3224, 0 },
		{ 2757, 3764, 0 },
		{ 0, 0, 84 },
		{ 1986, 3379, 0 },
		{ 2942, 4178, 0 },
		{ 963, 3853, 0 },
		{ 0, 3854, 0 },
		{ 2937, 3247, 0 },
		{ 2907, 3644, 0 },
		{ 2907, 3646, 0 },
		{ 2055, 3597, 0 },
		{ 2942, 4129, 0 },
		{ 2040, 3964, 0 },
		{ 2862, 3811, 0 },
		{ 2008, 3043, 0 },
		{ 2055, 3599, 0 },
		{ 2055, 3600, 0 },
		{ 2055, 3601, 0 },
		{ 0, 0, 88 },
		{ 2862, 3820, 0 },
		{ 1986, 3382, 0 },
		{ 2908, 3393, 0 },
		{ 2008, 3044, 0 },
		{ 2934, 3228, 0 },
		{ 2008, 3046, 0 },
		{ 2930, 3556, 0 },
		{ 2862, 3829, 0 },
		{ 2139, 4068, 0 },
		{ 1986, 3387, 0 },
		{ 2907, 3665, 0 },
		{ 0, 0, 90 },
		{ 2907, 3666, 0 },
		{ 2139, 4109, 0 },
		{ 2139, 4111, 0 },
		{ 2937, 3253, 0 },
		{ 0, 0, 15 },
		{ 1970, 3735, 0 },
		{ 0, 0, 17 },
		{ 0, 0, 18 },
		{ 2862, 3836, 0 },
		{ 2008, 3047, 0 },
		{ 2076, 3856, 0 },
		{ 2907, 3673, 0 },
		{ 2821, 4235, 0 },
		{ 2908, 3408, 0 },
		{ 2055, 3615, 0 },
		{ 1128, 3881, 0 },
		{ 2908, 3411, 0 },
		{ 2908, 3412, 0 },
		{ 2907, 3623, 0 },
		{ 2055, 3567, 0 },
		{ 0, 0, 55 },
		{ 2862, 3776, 0 },
		{ 2937, 3255, 0 },
		{ 2008, 3048, 0 },
		{ 2937, 3257, 0 },
		{ 1970, 3749, 0 },
		{ 2055, 3575, 0 },
		{ 2040, 3914, 0 },
		{ 0, 0, 73 },
		{ 2008, 3049, 0 },
		{ 2942, 4182, 0 },
		{ 2908, 3418, 0 },
		{ 0, 3135, 0 },
		{ 2908, 3420, 0 },
		{ 0, 0, 16 },
		{ 2055, 3581, 0 },
		{ 1128, 3904, 0 },
		{ 2008, 3050, 54 },
		{ 2008, 3051, 0 },
		{ 1970, 3708, 0 },
		{ 0, 0, 74 },
		{ 2907, 3642, 0 },
		{ 2942, 3180, 0 },
		{ 0, 0, 81 },
		{ 0, 0, 82 },
		{ 0, 0, 52 },
		{ 2907, 3645, 0 },
		{ 2930, 3532, 0 },
		{ 2930, 3534, 0 },
		{ 2937, 3263, 0 },
		{ 2930, 3537, 0 },
		{ 0, 0, 129 },
		{ 2907, 3651, 0 },
		{ 1128, 3890, 0 },
		{ 1128, 3891, 0 },
		{ 2937, 3264, 0 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 2008, 3052, 40 },
		{ 2907, 3655, 0 },
		{ 2942, 4169, 0 },
		{ 1128, 3896, 0 },
		{ 1128, 3897, 0 },
		{ 1970, 3726, 0 },
		{ 0, 0, 71 },
		{ 2907, 3656, 0 },
		{ 2937, 3266, 0 },
		{ 0, 0, 89 },
		{ 2937, 3268, 0 },
		{ 1970, 3730, 0 },
		{ 2930, 3543, 0 },
		{ 1970, 3733, 0 },
		{ 1986, 3344, 0 },
		{ 2862, 3814, 0 },
		{ 2934, 3209, 0 },
		{ 2862, 3816, 0 },
		{ 2076, 3876, 0 },
		{ 2008, 3056, 0 },
		{ 2937, 3271, 0 },
		{ 2942, 4152, 0 },
		{ 2934, 3213, 0 },
		{ 0, 0, 85 },
		{ 2942, 4155, 0 },
		{ 2008, 3057, 0 },
		{ 0, 0, 124 },
		{ 0, 0, 126 },
		{ 1970, 3740, 0 },
		{ 0, 0, 131 },
		{ 0, 0, 11 },
		{ 2907, 3667, 0 },
		{ 2907, 3668, 0 },
		{ 2055, 3604, 0 },
		{ 2930, 3555, 0 },
		{ 1128, 3894, 0 },
		{ 2008, 3058, 0 },
		{ 2937, 3275, 0 },
		{ 2907, 3675, 0 },
		{ 2937, 3277, 0 },
		{ 2942, 4177, 0 },
		{ 0, 0, 127 },
		{ 2862, 3832, 0 },
		{ 2942, 4179, 0 },
		{ 2907, 3620, 0 },
		{ 2934, 3217, 0 },
		{ 2934, 3222, 0 },
		{ 2942, 4118, 0 },
		{ 2008, 3062, 0 },
		{ 2942, 4124, 0 },
		{ 2862, 3837, 0 },
		{ 2821, 4299, 0 },
		{ 2937, 3283, 0 },
		{ 2937, 3285, 0 },
		{ 2008, 3063, 0 },
		{ 0, 0, 41 },
		{ 2907, 3628, 0 },
		{ 2821, 4309, 0 },
		{ 2942, 4133, 0 },
		{ 2937, 3288, 0 },
		{ 2055, 3563, 0 },
		{ 1970, 3710, 0 },
		{ 2862, 3848, 0 },
		{ 2862, 3849, 0 },
		{ 2757, 4187, 0 },
		{ 2942, 4153, 0 },
		{ 1970, 3711, 0 },
		{ 2821, 4335, 0 },
		{ 2862, 3775, 0 },
		{ 2907, 3631, 0 },
		{ 1970, 3713, 0 },
		{ 2055, 3565, 0 },
		{ 2055, 3566, 0 },
		{ 2040, 3913, 0 },
		{ 2937, 3289, 0 },
		{ 1970, 3717, 0 },
		{ 1970, 3718, 0 },
		{ 2055, 3568, 0 },
		{ 0, 0, 76 },
		{ 0, 0, 91 },
		{ 2907, 3636, 0 },
		{ 2907, 3637, 0 },
		{ 0, 3901, 0 },
		{ 2862, 3787, 0 },
		{ 0, 0, 86 },
		{ 1970, 3722, 0 },
		{ 2907, 3638, 0 },
		{ 1986, 3365, 0 },
		{ 0, 0, 12 },
		{ 2055, 3571, 0 },
		{ 2055, 3572, 0 },
		{ 0, 0, 87 },
		{ 0, 0, 53 },
		{ 0, 0, 92 },
		{ 2908, 3458, 0 },
		{ 2907, 3643, 0 },
		{ 2040, 3935, 0 },
		{ 0, 0, 14 },
		{ 2008, 3064, 0 },
		{ 2908, 3460, 0 },
		{ 2040, 3938, 0 },
		{ 2930, 3528, 0 },
		{ 1970, 3732, 0 },
		{ 2821, 4445, 0 },
		{ 2942, 4137, 0 },
		{ 2040, 3941, 0 },
		{ 1989, 3680, 0 },
		{ 2040, 3943, 0 },
		{ 2907, 3647, 0 },
		{ 2937, 3292, 0 },
		{ 0, 0, 13 },
		{ 2986, 1420, 220 },
		{ 0, 0, 221 },
		{ 2941, 4757, 222 },
		{ 2969, 1597, 226 },
		{ 1165, 2461, 227 },
		{ 0, 0, 227 },
		{ 2969, 1774, 223 },
		{ 1168, 1386, 0 },
		{ 2969, 1675, 224 },
		{ 1171, 1422, 0 },
		{ 1168, 0, 0 },
		{ 2895, 2576, 225 },
		{ 1173, 1435, 0 },
		{ 1171, 0, 0 },
		{ 2895, 2606, 223 },
		{ 1173, 0, 0 },
		{ 2895, 2616, 224 },
		{ 2934, 3218, 140 },
		{ 0, 0, 140 },
		{ 0, 0, 141 },
		{ 2947, 1980, 0 },
		{ 2969, 2729, 0 },
		{ 2986, 2056, 0 },
		{ 1181, 4604, 0 },
		{ 2966, 2513, 0 },
		{ 2969, 2784, 0 },
		{ 2981, 2872, 0 },
		{ 2977, 2411, 0 },
		{ 2980, 2922, 0 },
		{ 2986, 2029, 0 },
		{ 2980, 2953, 0 },
		{ 2982, 1883, 0 },
		{ 2887, 2599, 0 },
		{ 2984, 2162, 0 },
		{ 2938, 2238, 0 },
		{ 2947, 1972, 0 },
		{ 2987, 4485, 0 },
		{ 0, 0, 138 },
		{ 2757, 4186, 148 },
		{ 0, 0, 148 },
		{ 2969, 2818, 0 },
		{ 2834, 2705, 0 },
		{ 2984, 2166, 0 },
		{ 2986, 2043, 0 },
		{ 2969, 2812, 0 },
		{ 1203, 4555, 0 },
		{ 2969, 2447, 0 },
		{ 2951, 1471, 0 },
		{ 2969, 2762, 0 },
		{ 2986, 2060, 0 },
		{ 2602, 1422, 0 },
		{ 2982, 1770, 0 },
		{ 2976, 2650, 0 },
		{ 2887, 2621, 0 },
		{ 2938, 2197, 0 },
		{ 2790, 2672, 0 },
		{ 1214, 4652, 0 },
		{ 2969, 2451, 0 },
		{ 2977, 2390, 0 },
		{ 2947, 1967, 0 },
		{ 2969, 2836, 0 },
		{ 1219, 4641, 0 },
		{ 2979, 2395, 0 },
		{ 2974, 1560, 0 },
		{ 2938, 2288, 0 },
		{ 2981, 2865, 0 },
		{ 2982, 1902, 0 },
		{ 2887, 2504, 0 },
		{ 2984, 2139, 0 },
		{ 2938, 2246, 0 },
		{ 2987, 4327, 0 },
		{ 0, 0, 146 },
		{ 2934, 3220, 168 },
		{ 0, 0, 168 },
		{ 2947, 1992, 0 },
		{ 2969, 2761, 0 },
		{ 2986, 2050, 0 },
		{ 1235, 4553, 0 },
		{ 2981, 2665, 0 },
		{ 2977, 2368, 0 },
		{ 2980, 2935, 0 },
		{ 2947, 1930, 0 },
		{ 2947, 1946, 0 },
		{ 2969, 2832, 0 },
		{ 2947, 1950, 0 },
		{ 2987, 4401, 0 },
		{ 0, 0, 167 },
		{ 2076, 3875, 136 },
		{ 0, 0, 136 },
		{ 0, 0, 137 },
		{ 2969, 2724, 0 },
		{ 2938, 2249, 0 },
		{ 2984, 2179, 0 },
		{ 2971, 2308, 0 },
		{ 2969, 2776, 0 },
		{ 2942, 4163, 0 },
		{ 2977, 2379, 0 },
		{ 2980, 2918, 0 },
		{ 2947, 1958, 0 },
		{ 2947, 1966, 0 },
		{ 2949, 4517, 0 },
		{ 2949, 4511, 0 },
		{ 2887, 2509, 0 },
		{ 2938, 2215, 0 },
		{ 2887, 2608, 0 },
		{ 2982, 1905, 0 },
		{ 2887, 2631, 0 },
		{ 2980, 2916, 0 },
		{ 2977, 2386, 0 },
		{ 2887, 2505, 0 },
		{ 2947, 1571, 0 },
		{ 2969, 2777, 0 },
		{ 2986, 2073, 0 },
		{ 2987, 4331, 0 },
		{ 0, 0, 134 },
		{ 2515, 2892, 23 },
		{ 0, 0, 23 },
		{ 0, 0, 24 },
		{ 2969, 2796, 0 },
		{ 2790, 2667, 0 },
		{ 2887, 2601, 0 },
		{ 2938, 2274, 0 },
		{ 2941, 2, 0 },
		{ 2984, 2159, 0 },
		{ 2749, 2092, 0 },
		{ 2969, 2719, 0 },
		{ 2986, 2018, 0 },
		{ 2980, 2956, 0 },
		{ 2982, 1840, 0 },
		{ 2984, 2131, 0 },
		{ 2986, 2032, 0 },
		{ 2941, 4735, 0 },
		{ 2966, 2856, 0 },
		{ 2969, 2772, 0 },
		{ 2947, 1991, 0 },
		{ 2981, 2870, 0 },
		{ 2986, 2044, 0 },
		{ 2887, 2614, 0 },
		{ 2749, 2083, 0 },
		{ 2982, 1865, 0 },
		{ 2887, 2642, 0 },
		{ 2984, 2122, 0 },
		{ 2938, 2214, 0 },
		{ 2941, 7, 0 },
		{ 2949, 4520, 0 },
		{ 0, 0, 20 },
		{ 1318, 0, 1 },
		{ 1318, 0, 169 },
		{ 1318, 2662, 219 },
		{ 1533, 204, 219 },
		{ 1533, 417, 219 },
		{ 1533, 405, 219 },
		{ 1533, 522, 219 },
		{ 1533, 406, 219 },
		{ 1533, 417, 219 },
		{ 1533, 392, 219 },
		{ 1533, 419, 219 },
		{ 1533, 477, 219 },
		{ 1318, 0, 219 },
		{ 1330, 2439, 219 },
		{ 1318, 2714, 219 },
		{ 2515, 2890, 215 },
		{ 1533, 497, 219 },
		{ 1533, 496, 219 },
		{ 1533, 510, 219 },
		{ 1533, 0, 219 },
		{ 1533, 545, 219 },
		{ 1533, 532, 219 },
		{ 2984, 2140, 0 },
		{ 0, 0, 170 },
		{ 2938, 2200, 0 },
		{ 1533, 499, 0 },
		{ 1533, 0, 0 },
		{ 2941, 3823, 0 },
		{ 1533, 519, 0 },
		{ 1533, 535, 0 },
		{ 1533, 532, 0 },
		{ 1533, 539, 0 },
		{ 1533, 542, 0 },
		{ 1533, 545, 0 },
		{ 1533, 552, 0 },
		{ 1533, 596, 0 },
		{ 1533, 525, 0 },
		{ 1533, 519, 0 },
		{ 1533, 548, 0 },
		{ 2969, 2739, 0 },
		{ 2969, 2750, 0 },
		{ 1534, 555, 0 },
		{ 1534, 556, 0 },
		{ 1533, 567, 0 },
		{ 1533, 596, 0 },
		{ 1533, 588, 0 },
		{ 2984, 2163, 0 },
		{ 2966, 2850, 0 },
		{ 1533, 587, 0 },
		{ 1533, 631, 0 },
		{ 1533, 610, 0 },
		{ 1533, 612, 0 },
		{ 1533, 646, 0 },
		{ 1533, 647, 0 },
		{ 1533, 652, 0 },
		{ 1533, 646, 0 },
		{ 1533, 622, 0 },
		{ 1533, 613, 0 },
		{ 1533, 13, 0 },
		{ 1533, 40, 0 },
		{ 1533, 27, 0 },
		{ 2938, 2227, 0 },
		{ 2834, 2698, 0 },
		{ 1533, 43, 0 },
		{ 1533, 33, 0 },
		{ 1534, 35, 0 },
		{ 1533, 31, 0 },
		{ 1533, 35, 0 },
		{ 2977, 2401, 0 },
		{ 0, 0, 218 },
		{ 1533, 50, 0 },
		{ 1533, 26, 0 },
		{ 1533, 14, 0 },
		{ 1533, 72, 0 },
		{ 1533, 77, 0 },
		{ 1533, 79, 0 },
		{ 1533, 76, 0 },
		{ 1533, 66, 0 },
		{ 1533, 58, 0 },
		{ 1533, 62, 0 },
		{ 1533, 79, 0 },
		{ 1533, 0, 204 },
		{ 1533, 140, 0 },
		{ 2984, 2136, 0 },
		{ 2887, 2624, 0 },
		{ 1533, 99, 0 },
		{ 1533, 103, 0 },
		{ 1533, 126, 0 },
		{ 1533, 140, 0 },
		{ 1533, 140, 0 },
		{ -1413, 1092, 0 },
		{ 1534, 148, 0 },
		{ 1533, 181, 0 },
		{ 1533, 187, 0 },
		{ 1533, 179, 0 },
		{ 1533, 193, 0 },
		{ 1533, 195, 0 },
		{ 1533, 175, 0 },
		{ 1533, 192, 0 },
		{ 1533, 169, 0 },
		{ 1533, 160, 0 },
		{ 1533, 0, 203 },
		{ 1533, 177, 0 },
		{ 2971, 2323, 0 },
		{ 2938, 2276, 0 },
		{ 1533, 180, 0 },
		{ 1533, 189, 0 },
		{ 1533, 186, 0 },
		{ 1533, 0, 217 },
		{ 1533, 185, 0 },
		{ 0, 0, 205 },
		{ 1533, 178, 0 },
		{ 1535, 33, -4 },
		{ 1533, 234, 0 },
		{ 1533, 247, 0 },
		{ 1533, 273, 0 },
		{ 1533, 279, 0 },
		{ 1533, 288, 0 },
		{ 1533, 301, 0 },
		{ 1533, 272, 0 },
		{ 1533, 277, 0 },
		{ 1533, 268, 0 },
		{ 2969, 2822, 0 },
		{ 2969, 2829, 0 },
		{ 1533, 0, 207 },
		{ 1533, 307, 208 },
		{ 1533, 276, 0 },
		{ 1533, 285, 0 },
		{ 1533, 313, 0 },
		{ 1433, 3406, 0 },
		{ 2941, 4148, 0 },
		{ 2117, 4455, 194 },
		{ 1533, 319, 0 },
		{ 1533, 324, 0 },
		{ 1533, 335, 0 },
		{ 1533, 336, 0 },
		{ 1533, 338, 0 },
		{ 1533, 366, 0 },
		{ 1533, 356, 0 },
		{ 1533, 358, 0 },
		{ 1533, 373, 0 },
		{ 1533, 380, 0 },
		{ 1534, 366, 0 },
		{ 2942, 4164, 0 },
		{ 2941, 4, 210 },
		{ 1533, 369, 0 },
		{ 1533, 383, 0 },
		{ 1533, 365, 0 },
		{ 1533, 381, 0 },
		{ 0, 0, 174 },
		{ 1535, 117, -7 },
		{ 1535, 231, -10 },
		{ 1535, 345, -13 },
		{ 1535, 376, -16 },
		{ 1535, 460, -19 },
		{ 1535, 488, -22 },
		{ 1533, 409, 0 },
		{ 1533, 422, 0 },
		{ 1533, 396, 0 },
		{ 1533, 0, 192 },
		{ 1533, 0, 206 },
		{ 2977, 2381, 0 },
		{ 1533, 396, 0 },
		{ 1533, 386, 0 },
		{ 1533, 391, 0 },
		{ 1534, 392, 0 },
		{ 1470, 3442, 0 },
		{ 2941, 4180, 0 },
		{ 2117, 4471, 195 },
		{ 1473, 3446, 0 },
		{ 2941, 4144, 0 },
		{ 2117, 4493, 196 },
		{ 1476, 3447, 0 },
		{ 2941, 4152, 0 },
		{ 2117, 4475, 199 },
		{ 1479, 3448, 0 },
		{ 2941, 4068, 0 },
		{ 2117, 4467, 200 },
		{ 1482, 3449, 0 },
		{ 2941, 4142, 0 },
		{ 2117, 4479, 201 },
		{ 1485, 3450, 0 },
		{ 2941, 4146, 0 },
		{ 2117, 4462, 202 },
		{ 1533, 437, 0 },
		{ 1535, 490, -25 },
		{ 1533, 422, 0 },
		{ 2980, 2903, 0 },
		{ 1533, 403, 0 },
		{ 1533, 448, 0 },
		{ 1533, 441, 0 },
		{ 1533, 453, 0 },
		{ 0, 0, 176 },
		{ 0, 0, 178 },
		{ 0, 0, 184 },
		{ 0, 0, 186 },
		{ 0, 0, 188 },
		{ 0, 0, 190 },
		{ 1535, 574, -28 },
		{ 1503, 3460, 0 },
		{ 2941, 4156, 0 },
		{ 2117, 4489, 198 },
		{ 1533, 0, 191 },
		{ 2947, 1952, 0 },
		{ 1533, 474, 0 },
		{ 1533, 489, 0 },
		{ 1534, 482, 0 },
		{ 1533, 479, 0 },
		{ 1512, 3405, 0 },
		{ 2941, 4150, 0 },
		{ 2117, 4492, 197 },
		{ 0, 0, 182 },
		{ 2947, 1974, 0 },
		{ 1533, 4, 213 },
		{ 1534, 486, 0 },
		{ 1533, 1, 216 },
		{ 1533, 501, 0 },
		{ 0, 0, 180 },
		{ 2949, 4518, 0 },
		{ 2949, 4519, 0 },
		{ 1533, 488, 0 },
		{ 0, 0, 214 },
		{ 1533, 484, 0 },
		{ 2949, 4514, 0 },
		{ 0, 0, 212 },
		{ 1533, 492, 0 },
		{ 1533, 497, 0 },
		{ 0, 0, 211 },
		{ 1533, 502, 0 },
		{ 1533, 494, 0 },
		{ 1534, 499, 209 },
		{ 1535, 925, 0 },
		{ 1536, 736, -1 },
		{ 1537, 3420, 0 },
		{ 2941, 4112, 0 },
		{ 2117, 4487, 193 },
		{ 0, 0, 172 },
		{ 2076, 3877, 262 },
		{ 0, 0, 262 },
		{ 2969, 2754, 0 },
		{ 2938, 2230, 0 },
		{ 2984, 2117, 0 },
		{ 2971, 2335, 0 },
		{ 2969, 2775, 0 },
		{ 2942, 4166, 0 },
		{ 2977, 2382, 0 },
		{ 2980, 2930, 0 },
		{ 2947, 1986, 0 },
		{ 2947, 1987, 0 },
		{ 2949, 4500, 0 },
		{ 2949, 4501, 0 },
		{ 2887, 2617, 0 },
		{ 2938, 2270, 0 },
		{ 2887, 2622, 0 },
		{ 2982, 1776, 0 },
		{ 2887, 2628, 0 },
		{ 2980, 2923, 0 },
		{ 2977, 2364, 0 },
		{ 2887, 2636, 0 },
		{ 2947, 1507, 0 },
		{ 2969, 2718, 0 },
		{ 2986, 2027, 0 },
		{ 2987, 4345, 0 },
		{ 0, 0, 261 },
		{ 2076, 3871, 264 },
		{ 0, 0, 264 },
		{ 0, 0, 265 },
		{ 2969, 2721, 0 },
		{ 2938, 2287, 0 },
		{ 2984, 2142, 0 },
		{ 2971, 2329, 0 },
		{ 2969, 2740, 0 },
		{ 2942, 4147, 0 },
		{ 2977, 2388, 0 },
		{ 2980, 2949, 0 },
		{ 2947, 1999, 0 },
		{ 2947, 2000, 0 },
		{ 2949, 4502, 0 },
		{ 2949, 4507, 0 },
		{ 2981, 2882, 0 },
		{ 2986, 2042, 0 },
		{ 2984, 2164, 0 },
		{ 2947, 1925, 0 },
		{ 2947, 1928, 0 },
		{ 2984, 2112, 0 },
		{ 2951, 1474, 0 },
		{ 2969, 2794, 0 },
		{ 2986, 2052, 0 },
		{ 2987, 4403, 0 },
		{ 0, 0, 263 },
		{ 2076, 3861, 267 },
		{ 0, 0, 267 },
		{ 0, 0, 268 },
		{ 2969, 2797, 0 },
		{ 2938, 2254, 0 },
		{ 2984, 2127, 0 },
		{ 2971, 2325, 0 },
		{ 2969, 2819, 0 },
		{ 2942, 4176, 0 },
		{ 2977, 2409, 0 },
		{ 2980, 2928, 0 },
		{ 2947, 1935, 0 },
		{ 2947, 1943, 0 },
		{ 2949, 4515, 0 },
		{ 2949, 4516, 0 },
		{ 2971, 2336, 0 },
		{ 2974, 1518, 0 },
		{ 2982, 1884, 0 },
		{ 2980, 2900, 0 },
		{ 2982, 1891, 0 },
		{ 2984, 2148, 0 },
		{ 2986, 2028, 0 },
		{ 2987, 4259, 0 },
		{ 0, 0, 266 },
		{ 2076, 3866, 270 },
		{ 0, 0, 270 },
		{ 0, 0, 271 },
		{ 2969, 2733, 0 },
		{ 2938, 2205, 0 },
		{ 2984, 2161, 0 },
		{ 2971, 2309, 0 },
		{ 2969, 2752, 0 },
		{ 2942, 4143, 0 },
		{ 2977, 2410, 0 },
		{ 2980, 2951, 0 },
		{ 2947, 1954, 0 },
		{ 2947, 1955, 0 },
		{ 2949, 4508, 0 },
		{ 2949, 4509, 0 },
		{ 2969, 2771, 0 },
		{ 2951, 1441, 0 },
		{ 2980, 2912, 0 },
		{ 2977, 2378, 0 },
		{ 2974, 1633, 0 },
		{ 2980, 2920, 0 },
		{ 2982, 1669, 0 },
		{ 2984, 2111, 0 },
		{ 2986, 2045, 0 },
		{ 2987, 4263, 0 },
		{ 0, 0, 269 },
		{ 2076, 3874, 273 },
		{ 0, 0, 273 },
		{ 0, 0, 274 },
		{ 2969, 2795, 0 },
		{ 2938, 2255, 0 },
		{ 2984, 2116, 0 },
		{ 2971, 2327, 0 },
		{ 2969, 2807, 0 },
		{ 2942, 4173, 0 },
		{ 2977, 2406, 0 },
		{ 2980, 2898, 0 },
		{ 2947, 1968, 0 },
		{ 2947, 1970, 0 },
		{ 2949, 4521, 0 },
		{ 2949, 4522, 0 },
		{ 2984, 2123, 0 },
		{ 2749, 2097, 0 },
		{ 2982, 1716, 0 },
		{ 2887, 2639, 0 },
		{ 2971, 2312, 0 },
		{ 2887, 2456, 0 },
		{ 2947, 1973, 0 },
		{ 2969, 2723, 0 },
		{ 2986, 2070, 0 },
		{ 2987, 4409, 0 },
		{ 0, 0, 272 },
		{ 2821, 4371, 151 },
		{ 0, 0, 151 },
		{ 2834, 2708, 0 },
		{ 2982, 1717, 0 },
		{ 2969, 2737, 0 },
		{ 2986, 2007, 0 },
		{ 1676, 4575, 0 },
		{ 2969, 2441, 0 },
		{ 2951, 1470, 0 },
		{ 2969, 2751, 0 },
		{ 2986, 2024, 0 },
		{ 2602, 1437, 0 },
		{ 2982, 1790, 0 },
		{ 2976, 2648, 0 },
		{ 2887, 2618, 0 },
		{ 2938, 2245, 0 },
		{ 2790, 2669, 0 },
		{ 1687, 4633, 0 },
		{ 2969, 2453, 0 },
		{ 2977, 2359, 0 },
		{ 2947, 1988, 0 },
		{ 2969, 2788, 0 },
		{ 1692, 4552, 0 },
		{ 2979, 2393, 0 },
		{ 2974, 1602, 0 },
		{ 2938, 2252, 0 },
		{ 2981, 2863, 0 },
		{ 2982, 1841, 0 },
		{ 2887, 2434, 0 },
		{ 2984, 2175, 0 },
		{ 2938, 2263, 0 },
		{ 2987, 4481, 0 },
		{ 0, 0, 149 },
		{ 2076, 3870, 255 },
		{ 0, 0, 255 },
		{ 2969, 2813, 0 },
		{ 2938, 2265, 0 },
		{ 2984, 2177, 0 },
		{ 2971, 2317, 0 },
		{ 2969, 2826, 0 },
		{ 2942, 4181, 0 },
		{ 2977, 2404, 0 },
		{ 2980, 2941, 0 },
		{ 2947, 1996, 0 },
		{ 2947, 1997, 0 },
		{ 2949, 4505, 0 },
		{ 2949, 4506, 0 },
		{ 2966, 2839, 0 },
		{ 2887, 2612, 0 },
		{ 2947, 1998, 0 },
		{ 2749, 2103, 0 },
		{ 2977, 2356, 0 },
		{ 2980, 2908, 0 },
		{ 2602, 1361, 0 },
		{ 2987, 4411, 0 },
		{ 0, 0, 253 },
		{ 1740, 0, 1 },
		{ 1899, 2759, 370 },
		{ 2969, 2734, 370 },
		{ 2980, 2787, 370 },
		{ 2966, 2116, 370 },
		{ 1740, 0, 337 },
		{ 1740, 2697, 370 },
		{ 2976, 1522, 370 },
		{ 2757, 4195, 370 },
		{ 2055, 3582, 370 },
		{ 2934, 3219, 370 },
		{ 2055, 3584, 370 },
		{ 2040, 3960, 370 },
		{ 2986, 1927, 370 },
		{ 1740, 0, 370 },
		{ 2515, 2893, 368 },
		{ 2980, 2687, 370 },
		{ 2980, 2964, 370 },
		{ 0, 0, 370 },
		{ 2984, 2134, 0 },
		{ -1745, 21, 327 },
		{ -1746, 4597, 0 },
		{ 2938, 2219, 0 },
		{ 0, 0, 333 },
		{ 0, 0, 334 },
		{ 2977, 2383, 0 },
		{ 2887, 2643, 0 },
		{ 2969, 2767, 0 },
		{ 0, 0, 338 },
		{ 2938, 2223, 0 },
		{ 2986, 2064, 0 },
		{ 2887, 2502, 0 },
		{ 2008, 3004, 0 },
		{ 2930, 3533, 0 },
		{ 2937, 3284, 0 },
		{ 1948, 3177, 0 },
		{ 2930, 3536, 0 },
		{ 2974, 1598, 0 },
		{ 2947, 1931, 0 },
		{ 2938, 2242, 0 },
		{ 2982, 1893, 0 },
		{ 2986, 2017, 0 },
		{ 2984, 2149, 0 },
		{ 2663, 4567, 0 },
		{ 2984, 2151, 0 },
		{ 2947, 1936, 0 },
		{ 2982, 1901, 0 },
		{ 2938, 2261, 0 },
		{ 2966, 2843, 0 },
		{ 2986, 2026, 0 },
		{ 2977, 2374, 0 },
		{ 2076, 3858, 0 },
		{ 2008, 3021, 0 },
		{ 2008, 3022, 0 },
		{ 2040, 3997, 0 },
		{ 1970, 3745, 0 },
		{ 2969, 2816, 0 },
		{ 2947, 1945, 0 },
		{ 2966, 2840, 0 },
		{ 2974, 1601, 0 },
		{ 2969, 2820, 0 },
		{ 2977, 2380, 0 },
		{ 0, 4768, 330 },
		{ 2971, 2307, 0 },
		{ 2969, 2827, 0 },
		{ 2055, 3577, 0 },
		{ 2982, 1904, 0 },
		{ 0, 0, 369 },
		{ 2969, 2830, 0 },
		{ 2966, 2847, 0 },
		{ 2040, 3927, 0 },
		{ 1986, 3370, 0 },
		{ 2930, 3526, 0 },
		{ 2908, 3446, 0 },
		{ 2008, 3035, 0 },
		{ 0, 0, 358 },
		{ 2942, 4168, 0 },
		{ 2984, 2171, 0 },
		{ 2986, 2030, 0 },
		{ 2938, 2286, 0 },
		{ -1822, 1167, 0 },
		{ 0, 0, 329 },
		{ 2969, 2720, 0 },
		{ 0, 0, 357 },
		{ 2749, 2088, 0 },
		{ 2887, 2452, 0 },
		{ 2938, 2192, 0 },
		{ 1837, 4536, 0 },
		{ 2907, 3652, 0 },
		{ 2862, 3792, 0 },
		{ 2908, 3457, 0 },
		{ 2008, 3045, 0 },
		{ 2930, 3540, 0 },
		{ 2984, 2110, 0 },
		{ 2971, 2330, 0 },
		{ 2938, 2199, 0 },
		{ 2982, 1906, 0 },
		{ 0, 0, 359 },
		{ 2942, 4127, 336 },
		{ 2982, 1918, 0 },
		{ 2981, 2864, 0 },
		{ 2982, 1663, 0 },
		{ 0, 0, 362 },
		{ 0, 0, 363 },
		{ 1842, 0, -71 },
		{ 2021, 3129, 0 },
		{ 2055, 3606, 0 },
		{ 2930, 3551, 0 },
		{ 2040, 3956, 0 },
		{ 2887, 2600, 0 },
		{ 0, 0, 361 },
		{ 0, 0, 367 },
		{ 0, 4538, 0 },
		{ 2977, 2362, 0 },
		{ 2947, 1960, 0 },
		{ 2980, 2931, 0 },
		{ 2076, 3872, 0 },
		{ 2941, 4082, 0 },
		{ 2117, 4486, 352 },
		{ 2040, 3962, 0 },
		{ 2757, 4185, 0 },
		{ 2908, 3471, 0 },
		{ 2908, 3472, 0 },
		{ 2938, 2217, 0 },
		{ 0, 0, 364 },
		{ 0, 0, 365 },
		{ 2980, 2938, 0 },
		{ 2123, 4580, 0 },
		{ 2977, 2370, 0 },
		{ 2969, 2759, 0 },
		{ 0, 0, 342 },
		{ 1865, 0, -74 },
		{ 1867, 0, -77 },
		{ 2055, 3564, 0 },
		{ 2942, 4165, 0 },
		{ 0, 0, 360 },
		{ 2947, 1964, 0 },
		{ 0, 0, 335 },
		{ 2076, 3859, 0 },
		{ 2938, 2221, 0 },
		{ 2941, 4064, 0 },
		{ 2117, 4459, 353 },
		{ 2941, 4066, 0 },
		{ 2117, 4465, 354 },
		{ 2757, 4184, 0 },
		{ 1877, 0, -59 },
		{ 2947, 1965, 0 },
		{ 2969, 2768, 0 },
		{ 2969, 2770, 0 },
		{ 0, 0, 344 },
		{ 0, 0, 346 },
		{ 1882, 0, -65 },
		{ 2941, 4108, 0 },
		{ 2117, 4491, 356 },
		{ 0, 0, 332 },
		{ 2938, 2224, 0 },
		{ 2986, 2055, 0 },
		{ 2941, 4136, 0 },
		{ 2117, 4457, 355 },
		{ 0, 0, 350 },
		{ 2984, 2129, 0 },
		{ 2980, 2909, 0 },
		{ 0, 0, 348 },
		{ 2971, 2334, 0 },
		{ 2982, 1664, 0 },
		{ 2969, 2781, 0 },
		{ 2887, 2633, 0 },
		{ 0, 0, 366 },
		{ 2984, 2133, 0 },
		{ 2938, 2243, 0 },
		{ 1896, 0, -80 },
		{ 2941, 4154, 0 },
		{ 2117, 4490, 351 },
		{ 0, 0, 340 },
		{ 1740, 2745, 370 },
		{ 1903, 2437, 370 },
		{ -1901, 4763, 327 },
		{ -1902, 4594, 0 },
		{ 2941, 4581, 0 },
		{ 2663, 4552, 0 },
		{ 0, 0, 328 },
		{ 2663, 4568, 0 },
		{ -1907, 4761, 0 },
		{ -1908, 4589, 0 },
		{ 1911, 2, 330 },
		{ 2663, 4569, 0 },
		{ 2941, 4636, 0 },
		{ 0, 0, 331 },
		{ 1929, 0, 1 },
		{ 2125, 2761, 326 },
		{ 2969, 2810, 326 },
		{ 1929, 0, 280 },
		{ 1929, 2507, 326 },
		{ 2930, 3545, 326 },
		{ 1929, 0, 283 },
		{ 2974, 1523, 326 },
		{ 2757, 4189, 326 },
		{ 2055, 3593, 326 },
		{ 2934, 3160, 326 },
		{ 2055, 3595, 326 },
		{ 2040, 3911, 326 },
		{ 2980, 2897, 326 },
		{ 2986, 1925, 326 },
		{ 1929, 0, 326 },
		{ 2515, 2891, 323 },
		{ 2980, 2905, 326 },
		{ 2966, 2853, 326 },
		{ 2757, 4191, 326 },
		{ 2980, 1593, 326 },
		{ 0, 0, 326 },
		{ 2984, 2143, 0 },
		{ -1936, 4759, 275 },
		{ -1937, 4590, 0 },
		{ 2938, 2258, 0 },
		{ 0, 0, 281 },
		{ 2938, 2259, 0 },
		{ 2984, 2144, 0 },
		{ 2986, 2015, 0 },
		{ 2008, 3005, 0 },
		{ 2930, 3516, 0 },
		{ 2937, 3301, 0 },
		{ 2907, 3671, 0 },
		{ 0, 3148, 0 },
		{ 0, 3195, 0 },
		{ 2930, 3523, 0 },
		{ 2977, 2366, 0 },
		{ 2974, 1577, 0 },
		{ 2947, 1975, 0 },
		{ 2938, 2273, 0 },
		{ 2969, 2837, 0 },
		{ 2969, 2714, 0 },
		{ 2984, 2154, 0 },
		{ 2123, 4579, 0 },
		{ 2984, 2155, 0 },
		{ 2663, 4573, 0 },
		{ 2984, 2156, 0 },
		{ 2966, 2852, 0 },
		{ 2749, 2096, 0 },
		{ 2986, 2019, 0 },
		{ 2076, 3860, 0 },
		{ 2008, 3018, 0 },
		{ 2008, 3019, 0 },
		{ 2862, 3812, 0 },
		{ 2862, 3813, 0 },
		{ 2040, 3954, 0 },
		{ 0, 3707, 0 },
		{ 2947, 1976, 0 },
		{ 2969, 2727, 0 },
		{ 2947, 1978, 0 },
		{ 2966, 2841, 0 },
		{ 2938, 2195, 0 },
		{ 2947, 1979, 0 },
		{ 2977, 2394, 0 },
		{ 0, 0, 325 },
		{ 2977, 2396, 0 },
		{ 0, 0, 277 },
		{ 2971, 2333, 0 },
		{ 0, 0, 322 },
		{ 2974, 1595, 0 },
		{ 2969, 2744, 0 },
		{ 2040, 3967, 0 },
		{ 0, 3351, 0 },
		{ 2930, 3553, 0 },
		{ 1989, 3683, 0 },
		{ 0, 3684, 0 },
		{ 2908, 3477, 0 },
		{ 2008, 3031, 0 },
		{ 2969, 2749, 0 },
		{ 0, 0, 315 },
		{ 2942, 4172, 0 },
		{ 2984, 2168, 0 },
		{ 2982, 1797, 0 },
		{ 2982, 1822, 0 },
		{ 2974, 1596, 0 },
		{ -2016, 1242, 0 },
		{ 2969, 2760, 0 },
		{ 2977, 2357, 0 },
		{ 2938, 2218, 0 },
		{ 2907, 3657, 0 },
		{ 2862, 3844, 0 },
		{ 2908, 3405, 0 },
		{ 2862, 3846, 0 },
		{ 2862, 3847, 0 },
		{ 0, 3041, 0 },
		{ 2930, 3520, 0 },
		{ 0, 0, 314 },
		{ 2984, 2107, 0 },
		{ 2971, 2319, 0 },
		{ 2887, 2511, 0 },
		{ 0, 0, 321 },
		{ 2980, 2958, 0 },
		{ 0, 0, 316 },
		{ 0, 0, 279 },
		{ 2980, 2963, 0 },
		{ 2982, 1843, 0 },
		{ 2033, 0, -56 },
		{ 0, 3128, 0 },
		{ 2055, 3609, 0 },
		{ 2024, 3117, 0 },
		{ 2021, 3118, 0 },
		{ 2930, 3530, 0 },
		{ 2040, 4001, 0 },
		{ 2887, 2595, 0 },
		{ 0, 0, 318 },
		{ 2981, 2877, 0 },
		{ 2982, 1848, 0 },
		{ 2982, 1853, 0 },
		{ 2076, 3873, 0 },
		{ 2941, 4138, 0 },
		{ 2117, 4488, 305 },
		{ 2040, 3916, 0 },
		{ 2757, 4190, 0 },
		{ 2040, 3917, 0 },
		{ 2040, 3918, 0 },
		{ 2040, 3919, 0 },
		{ 0, 3920, 0 },
		{ 2908, 3423, 0 },
		{ 2908, 3424, 0 },
		{ 2938, 2226, 0 },
		{ 2980, 2907, 0 },
		{ 2887, 2602, 0 },
		{ 2887, 2603, 0 },
		{ 2969, 2786, 0 },
		{ 0, 0, 287 },
		{ 2062, 0, -35 },
		{ 2064, 0, -38 },
		{ 2066, 0, -44 },
		{ 2068, 0, -47 },
		{ 2070, 0, -50 },
		{ 2072, 0, -53 },
		{ 0, 3569, 0 },
		{ 2942, 4180, 0 },
		{ 0, 0, 317 },
		{ 2977, 2376, 0 },
		{ 2984, 2113, 0 },
		{ 2984, 2114, 0 },
		{ 2938, 2232, 0 },
		{ 2941, 4070, 0 },
		{ 2117, 4463, 306 },
		{ 2941, 4072, 0 },
		{ 2117, 4466, 307 },
		{ 2941, 4074, 0 },
		{ 2117, 4469, 310 },
		{ 2941, 4076, 0 },
		{ 2117, 4473, 311 },
		{ 2941, 4078, 0 },
		{ 2117, 4477, 312 },
		{ 2941, 4080, 0 },
		{ 2117, 4485, 313 },
		{ 2757, 4196, 0 },
		{ 2087, 0, -62 },
		{ 0, 3863, 0 },
		{ 2938, 2233, 0 },
		{ 2938, 2234, 0 },
		{ 2969, 2805, 0 },
		{ 0, 0, 289 },
		{ 0, 0, 291 },
		{ 0, 0, 297 },
		{ 0, 0, 299 },
		{ 0, 0, 301 },
		{ 0, 0, 303 },
		{ 2093, 0, -68 },
		{ 2941, 4110, 0 },
		{ 2117, 4495, 309 },
		{ 2969, 2806, 0 },
		{ 2980, 2932, 0 },
		{ 2916, 3111, 320 },
		{ 2986, 2046, 0 },
		{ 2941, 4140, 0 },
		{ 2117, 4464, 308 },
		{ 0, 0, 295 },
		{ 2938, 2239, 0 },
		{ 2986, 2048, 0 },
		{ 0, 0, 282 },
		{ 2980, 2943, 0 },
		{ 0, 0, 293 },
		{ 2984, 2120, 0 },
		{ 2602, 1345, 0 },
		{ 2982, 1875, 0 },
		{ 2971, 2321, 0 },
		{ 2821, 4373, 0 },
		{ 2887, 2638, 0 },
		{ 2969, 2821, 0 },
		{ 2977, 2402, 0 },
		{ 2984, 2125, 0 },
		{ 0, 0, 319 },
		{ 2790, 2671, 0 },
		{ 2938, 2251, 0 },
		{ 2984, 2126, 0 },
		{ 2116, 0, -41 },
		{ 2986, 2054, 0 },
		{ 2941, 4062, 0 },
		{ 0, 4494, 304 },
		{ 2887, 2448, 0 },
		{ 0, 0, 285 },
		{ 2982, 1876, 0 },
		{ 2976, 2652, 0 },
		{ 2971, 2332, 0 },
		{ 0, 1, 0 },
		{ 0, 0, 324 },
		{ 1929, 2770, 326 },
		{ 2129, 2438, 326 },
		{ -2127, 22, 275 },
		{ -2128, 4588, 0 },
		{ 2941, 4580, 0 },
		{ 2663, 4550, 0 },
		{ 0, 0, 276 },
		{ 2663, 4570, 0 },
		{ -2133, 4766, 0 },
		{ -2134, 4593, 0 },
		{ 2137, 0, 277 },
		{ 2663, 4571, 0 },
		{ 2941, 4669, 0 },
		{ 0, 0, 278 },
		{ 0, 4062, 372 },
		{ 0, 0, 372 },
		{ 2969, 2722, 0 },
		{ 2834, 2687, 0 },
		{ 2980, 2927, 0 },
		{ 2974, 1604, 0 },
		{ 2977, 2360, 0 },
		{ 2982, 1887, 0 },
		{ 2941, 9, 0 },
		{ 2986, 2068, 0 },
		{ 2974, 1616, 0 },
		{ 2938, 2269, 0 },
		{ 2152, 4559, 0 },
		{ 2941, 1916, 0 },
		{ 2980, 2942, 0 },
		{ 2986, 2005, 0 },
		{ 2980, 2947, 0 },
		{ 2971, 2313, 0 },
		{ 2969, 2742, 0 },
		{ 2982, 1896, 0 },
		{ 2969, 2745, 0 },
		{ 2986, 2008, 0 },
		{ 2947, 1942, 0 },
		{ 2987, 4335, 0 },
		{ 0, 0, 371 },
		{ 2663, 4563, 416 },
		{ 0, 0, 377 },
		{ 0, 0, 379 },
		{ 2183, 821, 407 },
		{ 2331, 834, 407 },
		{ 2350, 832, 407 },
		{ 2303, 833, 407 },
		{ 2184, 841, 407 },
		{ 2182, 827, 407 },
		{ 2350, 831, 407 },
		{ 2203, 845, 407 },
		{ 2328, 848, 407 },
		{ 2328, 849, 407 },
		{ 2331, 847, 407 },
		{ 2282, 856, 407 },
		{ 2969, 1653, 406 },
		{ 2210, 2463, 416 },
		{ 2379, 845, 407 },
		{ 2331, 858, 407 },
		{ 2213, 858, 407 },
		{ 2331, 854, 407 },
		{ 2969, 2790, 416 },
		{ -2186, 4764, 373 },
		{ -2187, 4592, 0 },
		{ 2379, 851, 407 },
		{ 2384, 355, 407 },
		{ 2379, 853, 407 },
		{ 2254, 852, 407 },
		{ 2331, 860, 407 },
		{ 2336, 855, 407 },
		{ 2331, 862, 407 },
		{ 2282, 871, 407 },
		{ 2257, 861, 407 },
		{ 2350, 856, 407 },
		{ 2181, 850, 407 },
		{ 2305, 857, 407 },
		{ 2181, 854, 407 },
		{ 2361, 865, 407 },
		{ 2336, 865, 407 },
		{ 2181, 875, 407 },
		{ 2361, 868, 407 },
		{ 2319, 879, 407 },
		{ 2361, 870, 407 },
		{ 2257, 873, 407 },
		{ 2969, 1686, 403 },
		{ 2235, 1511, 0 },
		{ 2969, 1708, 404 },
		{ 2938, 2272, 0 },
		{ 2663, 4564, 0 },
		{ 2181, 884, 407 },
		{ 2984, 1963, 0 },
		{ 2328, 882, 407 },
		{ 2246, 867, 407 },
		{ 2361, 875, 407 },
		{ 2305, 881, 407 },
		{ 2305, 882, 407 },
		{ 2257, 891, 407 },
		{ 2328, 899, 407 },
		{ 2303, 883, 407 },
		{ 2328, 901, 407 },
		{ 2282, 906, 407 },
		{ 2384, 357, 407 },
		{ 2384, 370, 407 },
		{ 2356, 917, 407 },
		{ 2356, 918, 407 },
		{ 2328, 961, 407 },
		{ 2246, 946, 407 },
		{ 2282, 967, 407 },
		{ 2319, 975, 407 },
		{ 2260, 1468, 0 },
		{ 2235, 0, 0 },
		{ 2895, 2527, 405 },
		{ 2262, 1469, 0 },
		{ 2966, 2845, 0 },
		{ 0, 0, 375 },
		{ 2328, 975, 407 },
		{ 2834, 2694, 0 },
		{ 2384, 463, 407 },
		{ 2257, 970, 407 },
		{ 2305, 963, 407 },
		{ 2384, 469, 407 },
		{ 2331, 1008, 407 },
		{ 2181, 964, 407 },
		{ 2256, 1009, 407 },
		{ 2331, 1005, 407 },
		{ 2384, 8, 407 },
		{ 2305, 1034, 407 },
		{ 2982, 1803, 0 },
		{ 2941, 2182, 0 },
		{ 2356, 1036, 407 },
		{ 2181, 1040, 407 },
		{ 2350, 1039, 407 },
		{ 2181, 1055, 407 },
		{ 2181, 1047, 407 },
		{ 2181, 1037, 407 },
		{ 2260, 0, 0 },
		{ 2895, 2537, 403 },
		{ 2262, 0, 0 },
		{ 2895, 2547, 404 },
		{ 0, 0, 408 },
		{ 2350, 1069, 407 },
		{ 2287, 4629, 0 },
		{ 2977, 2039, 0 },
		{ 2282, 1087, 407 },
		{ 2384, 123, 407 },
		{ 2947, 1867, 0 },
		{ 2404, 6, 407 },
		{ 2356, 1071, 407 },
		{ 2282, 1091, 407 },
		{ 2305, 1109, 407 },
		{ 2303, 1108, 407 },
		{ 2984, 1926, 0 },
		{ 2331, 1122, 407 },
		{ 2938, 2289, 0 },
		{ 2986, 2021, 0 },
		{ 2938, 2193, 0 },
		{ 2336, 1117, 407 },
		{ 2350, 1115, 407 },
		{ 2181, 1133, 407 },
		{ 2328, 1130, 407 },
		{ 2384, 125, 407 },
		{ 2331, 1153, 407 },
		{ 2384, 128, 407 },
		{ 2979, 2139, 0 },
		{ 2887, 2637, 0 },
		{ 2305, 1143, 407 },
		{ 2947, 1861, 0 },
		{ 2982, 1730, 0 },
		{ 2987, 4341, 0 },
		{ 2941, 4654, 383 },
		{ 2379, 1151, 407 },
		{ 2305, 1145, 407 },
		{ 2331, 1199, 407 },
		{ 2331, 1158, 407 },
		{ 2834, 2692, 0 },
		{ 2336, 1190, 407 },
		{ 2887, 2453, 0 },
		{ 2969, 2743, 0 },
		{ 2887, 2454, 0 },
		{ 2181, 1184, 407 },
		{ 2331, 1198, 407 },
		{ 2181, 1189, 407 },
		{ 2384, 130, 407 },
		{ 2986, 1888, 0 },
		{ 2361, 1223, 407 },
		{ 2984, 1933, 0 },
		{ 2930, 3552, 0 },
		{ 2887, 2593, 0 },
		{ 2971, 2311, 0 },
		{ 2331, 1229, 407 },
		{ 2982, 1878, 0 },
		{ 2980, 2919, 0 },
		{ 2404, 121, 407 },
		{ 2336, 1225, 407 },
		{ 2336, 1227, 407 },
		{ 2181, 1265, 407 },
		{ 2361, 1256, 407 },
		{ 2345, 4532, 0 },
		{ 2361, 1257, 407 },
		{ 2982, 1899, 0 },
		{ 2969, 2779, 0 },
		{ 2982, 1900, 0 },
		{ 2328, 1267, 407 },
		{ 2361, 1259, 407 },
		{ 2181, 1269, 407 },
		{ 2984, 1738, 0 },
		{ 2969, 2792, 0 },
		{ 2181, 1266, 407 },
		{ 2834, 2684, 0 },
		{ 2934, 3204, 0 },
		{ 2982, 1911, 0 },
		{ 2887, 2632, 0 },
		{ 2181, 1261, 407 },
		{ 2980, 2962, 0 },
		{ 2982, 1658, 0 },
		{ 2987, 4471, 0 },
		{ 0, 0, 399 },
		{ 2350, 1259, 407 },
		{ 2361, 1264, 407 },
		{ 2384, 235, 407 },
		{ 2351, 1272, 407 },
		{ 2941, 1907, 0 },
		{ 2384, 237, 407 },
		{ 2370, 4541, 0 },
		{ 2371, 4559, 0 },
		{ 2372, 4546, 0 },
		{ 2181, 1263, 407 },
		{ 2181, 1275, 407 },
		{ 2384, 239, 407 },
		{ 2980, 2911, 0 },
		{ 2834, 2689, 0 },
		{ 2966, 2851, 0 },
		{ 2181, 1264, 407 },
		{ 2380, 4606, 0 },
		{ 2947, 1985, 0 },
		{ 2938, 2201, 0 },
		{ 2982, 1817, 0 },
		{ 2181, 1270, 407 },
		{ 2982, 1833, 0 },
		{ 2947, 1990, 0 },
		{ 2384, 241, 407 },
		{ 2384, 243, 407 },
		{ 2941, 2288, 0 },
		{ 2384, 245, 407 },
		{ 2986, 2014, 0 },
		{ 2941, 1912, 0 },
		{ 2982, 1807, 0 },
		{ 2966, 2511, 0 },
		{ 2982, 1855, 0 },
		{ 2384, 349, 407 },
		{ 2384, 351, 407 },
		{ 2981, 1905, 0 },
		{ 2986, 2025, 0 },
		{ 2834, 2677, 0 },
		{ 2974, 1527, 0 },
		{ 2384, 1451, 407 },
		{ 2984, 1859, 0 },
		{ 2407, 4653, 0 },
		{ 2969, 2741, 0 },
		{ 2987, 4374, 0 },
		{ 2404, 576, 407 },
		{ 2947, 1938, 0 },
		{ 2987, 4407, 0 },
		{ 2941, 2290, 0 },
		{ 2984, 1951, 0 },
		{ 2969, 2746, 0 },
		{ 2984, 1736, 0 },
		{ 2984, 2158, 0 },
		{ 2986, 2038, 0 },
		{ 2986, 2039, 0 },
		{ 2969, 2753, 0 },
		{ 2986, 2041, 0 },
		{ 2941, 1918, 0 },
		{ 2947, 1863, 0 },
		{ 2947, 1948, 0 },
		{ 2938, 2267, 0 },
		{ 2427, 4567, 0 },
		{ 2969, 2765, 0 },
		{ 2947, 1949, 0 },
		{ 2981, 2885, 0 },
		{ 2431, 811, 407 },
		{ 2969, 2769, 0 },
		{ 2749, 2101, 0 },
		{ 2987, 4477, 0 },
		{ 2947, 1951, 0 },
		{ 2941, 4755, 381 },
		{ 2947, 1869, 0 },
		{ 2987, 4255, 0 },
		{ 2438, 10, 390 },
		{ 2984, 2106, 0 },
		{ 2749, 2084, 0 },
		{ 2938, 2282, 0 },
		{ 2980, 2948, 0 },
		{ 2834, 2709, 0 },
		{ 2790, 2668, 0 },
		{ 2984, 2109, 0 },
		{ 2969, 2785, 0 },
		{ 2749, 2085, 0 },
		{ 2969, 2787, 0 },
		{ 2986, 2051, 0 },
		{ 2887, 2516, 0 },
		{ 2951, 1469, 0 },
		{ 2974, 1513, 0 },
		{ 2947, 1871, 0 },
		{ 2938, 2194, 0 },
		{ 2749, 2099, 0 },
		{ 2969, 2799, 0 },
		{ 2987, 4257, 0 },
		{ 2941, 4712, 402 },
		{ 2938, 2196, 0 },
		{ 2982, 1909, 0 },
		{ 0, 0, 413 },
		{ 2947, 1962, 0 },
		{ 2887, 2604, 0 },
		{ 1787, 18, 389 },
		{ 2980, 2915, 0 },
		{ 2969, 2809, 0 },
		{ 2887, 2605, 0 },
		{ 2986, 2063, 0 },
		{ 2834, 2683, 0 },
		{ 2466, 4555, 0 },
		{ 2515, 2894, 0 },
		{ 2969, 2814, 0 },
		{ 2982, 1910, 0 },
		{ 2969, 2817, 0 },
		{ 2984, 2124, 0 },
		{ 2460, 1279, 0 },
		{ 2473, 4552, 0 },
		{ 2749, 2086, 0 },
		{ 2981, 2861, 0 },
		{ 2982, 1917, 0 },
		{ 2986, 2072, 0 },
		{ 2478, 4569, 0 },
		{ 2969, 2825, 0 },
		{ 2887, 2623, 0 },
		{ 2481, 4604, 0 },
		{ 0, 1311, 0 },
		{ 2977, 2398, 0 },
		{ 2986, 2004, 0 },
		{ 2982, 1657, 0 },
		{ 2969, 2834, 0 },
		{ 2947, 1969, 0 },
		{ 2941, 2684, 0 },
		{ 2980, 2959, 0 },
		{ 2490, 4662, 0 },
		{ 2976, 2655, 0 },
		{ 2492, 4541, 0 },
		{ 2515, 2889, 0 },
		{ 2969, 2717, 0 },
		{ 2947, 1873, 0 },
		{ 2977, 2407, 0 },
		{ 2986, 2010, 0 },
		{ 2947, 1971, 0 },
		{ 2887, 2640, 0 },
		{ 2984, 1936, 0 },
		{ 2986, 2016, 0 },
		{ 2971, 2315, 0 },
		{ 2981, 2667, 0 },
		{ 2969, 2732, 0 },
		{ 2987, 4333, 0 },
		{ 2980, 2914, 0 },
		{ 2984, 2146, 0 },
		{ 2938, 2236, 0 },
		{ 2749, 2087, 0 },
		{ 2974, 1578, 0 },
		{ 2515, 2888, 0 },
		{ 2966, 2515, 0 },
		{ 2513, 4644, 0 },
		{ 2966, 2517, 0 },
		{ 2980, 2926, 0 },
		{ 2987, 4446, 0 },
		{ 2982, 1665, 0 },
		{ 2984, 2153, 0 },
		{ 2887, 2507, 0 },
		{ 2520, 4622, 0 },
		{ 2938, 2244, 0 },
		{ 2749, 2094, 0 },
		{ 2980, 2933, 0 },
		{ 2887, 2513, 0 },
		{ 2980, 2936, 0 },
		{ 2987, 4267, 0 },
		{ 2941, 4633, 400 },
		{ 2982, 1666, 0 },
		{ 2986, 2020, 0 },
		{ 2982, 1667, 0 },
		{ 2986, 2023, 0 },
		{ 2834, 2702, 0 },
		{ 2887, 2596, 0 },
		{ 2969, 2756, 0 },
		{ 2987, 4399, 0 },
		{ 2969, 2758, 0 },
		{ 0, 2887, 0 },
		{ 2941, 4651, 388 },
		{ 2980, 2955, 0 },
		{ 2982, 1668, 0 },
		{ 2749, 2082, 0 },
		{ 2984, 1945, 0 },
		{ 2790, 2665, 0 },
		{ 2969, 2763, 0 },
		{ 2982, 1670, 0 },
		{ 2947, 1981, 0 },
		{ 2947, 1982, 0 },
		{ 2941, 4680, 382 },
		{ 2984, 2169, 0 },
		{ 2947, 1983, 0 },
		{ 2947, 1984, 0 },
		{ 2887, 2613, 0 },
		{ 2834, 2695, 0 },
		{ 2977, 2400, 0 },
		{ 2749, 2089, 0 },
		{ 0, 0, 412 },
		{ 2749, 2091, 0 },
		{ 2887, 2620, 0 },
		{ 2982, 1695, 0 },
		{ 2556, 4622, 0 },
		{ 2982, 1699, 0 },
		{ 2749, 2095, 0 },
		{ 2559, 4647, 0 },
		{ 2986, 2036, 0 },
		{ 2887, 2625, 0 },
		{ 2980, 2921, 0 },
		{ 2969, 2789, 0 },
		{ 2986, 2037, 0 },
		{ 2987, 4413, 0 },
		{ 2987, 4415, 0 },
		{ 2938, 2284, 0 },
		{ 2969, 2793, 0 },
		{ 2887, 2630, 0 },
		{ 2982, 1703, 0 },
		{ 2982, 1705, 0 },
		{ 2977, 2358, 0 },
		{ 2947, 1989, 0 },
		{ 2947, 1875, 0 },
		{ 2987, 4261, 0 },
		{ 2969, 2803, 0 },
		{ 2984, 1957, 0 },
		{ 2980, 2940, 0 },
		{ 2984, 2118, 0 },
		{ 2982, 1720, 0 },
		{ 2947, 1995, 0 },
		{ 2987, 4339, 0 },
		{ 2986, 939, 385 },
		{ 2941, 4675, 395 },
		{ 2790, 2673, 0 },
		{ 2986, 2047, 0 },
		{ 2982, 1750, 0 },
		{ 2976, 2657, 0 },
		{ 2976, 2661, 0 },
		{ 2887, 2446, 0 },
		{ 2589, 4608, 0 },
		{ 2981, 2879, 0 },
		{ 2941, 4710, 393 },
		{ 2986, 2049, 0 },
		{ 2749, 2090, 0 },
		{ 2982, 1758, 0 },
		{ 2938, 2207, 0 },
		{ 2887, 2455, 0 },
		{ 2596, 4680, 0 },
		{ 2941, 4736, 384 },
		{ 2987, 4475, 0 },
		{ 2598, 4690, 0 },
		{ 2602, 1434, 0 },
		{ 2600, 4697, 0 },
		{ 2601, 4698, 0 },
		{ 2982, 1773, 0 },
		{ 2979, 2403, 0 },
		{ 2986, 2053, 0 },
		{ 2980, 2906, 0 },
		{ 2969, 2831, 0 },
		{ 2984, 2135, 0 },
		{ 2947, 2001, 0 },
		{ 2984, 2137, 0 },
		{ 2987, 4271, 0 },
		{ 2941, 4638, 397 },
		{ 2987, 4273, 0 },
		{ 2987, 4275, 0 },
		{ 2987, 4300, 0 },
		{ 2987, 4302, 0 },
		{ 0, 1436, 0 },
		{ 2887, 2514, 0 },
		{ 2887, 2515, 0 },
		{ 2982, 1778, 0 },
		{ 2986, 2058, 0 },
		{ 2986, 2059, 0 },
		{ 2987, 4343, 0 },
		{ 2938, 2229, 0 },
		{ 0, 0, 415 },
		{ 0, 0, 414 },
		{ 2941, 4665, 386 },
		{ 0, 0, 410 },
		{ 0, 0, 411 },
		{ 2987, 4347, 0 },
		{ 2977, 2405, 0 },
		{ 2749, 2081, 0 },
		{ 2984, 2145, 0 },
		{ 2980, 2924, 0 },
		{ 2987, 4405, 0 },
		{ 2941, 4683, 380 },
		{ 2629, 4589, 0 },
		{ 2941, 4686, 387 },
		{ 2969, 2725, 0 },
		{ 2982, 1780, 0 },
		{ 2986, 2062, 0 },
		{ 2982, 1784, 0 },
		{ 2941, 4696, 398 },
		{ 2941, 2180, 0 },
		{ 2987, 4417, 0 },
		{ 2987, 4419, 0 },
		{ 2987, 4444, 0 },
		{ 2984, 2152, 0 },
		{ 2982, 1786, 0 },
		{ 2941, 4715, 391 },
		{ 2941, 4717, 392 },
		{ 2941, 4720, 394 },
		{ 2986, 2065, 0 },
		{ 2969, 2738, 0 },
		{ 2987, 4479, 0 },
		{ 2986, 2066, 0 },
		{ 2941, 4731, 396 },
		{ 2980, 2939, 0 },
		{ 2982, 1788, 0 },
		{ 2887, 2615, 0 },
		{ 2984, 2157, 0 },
		{ 2938, 2247, 0 },
		{ 2947, 1937, 0 },
		{ 2987, 4265, 0 },
		{ 2941, 4750, 401 },
		{ 2663, 4551, 416 },
		{ 2656, 0, 377 },
		{ 0, 0, 378 },
		{ -2654, 4760, 373 },
		{ -2655, 4595, 0 },
		{ 2941, 4572, 0 },
		{ 2663, 4560, 0 },
		{ 0, 0, 374 },
		{ 2663, 4554, 0 },
		{ -2660, 4765, 0 },
		{ -2661, 4587, 0 },
		{ 2664, 0, 375 },
		{ 0, 4562, 0 },
		{ 2941, 4641, 0 },
		{ 0, 0, 376 },
		{ 2934, 3227, 144 },
		{ 0, 0, 144 },
		{ 0, 0, 145 },
		{ 2947, 1939, 0 },
		{ 2969, 2748, 0 },
		{ 2986, 2003, 0 },
		{ 2673, 4556, 0 },
		{ 2979, 2401, 0 },
		{ 2974, 1632, 0 },
		{ 2938, 2256, 0 },
		{ 2981, 2883, 0 },
		{ 2982, 1802, 0 },
		{ 2887, 2629, 0 },
		{ 2984, 2165, 0 },
		{ 2938, 2260, 0 },
		{ 2947, 1944, 0 },
		{ 2987, 4372, 0 },
		{ 0, 0, 142 },
		{ 2821, 4333, 163 },
		{ 0, 0, 163 },
		{ 2982, 1808, 0 },
		{ 2688, 4587, 0 },
		{ 2969, 2445, 0 },
		{ 2980, 2904, 0 },
		{ 2981, 2873, 0 },
		{ 2976, 2645, 0 },
		{ 2693, 4595, 0 },
		{ 2941, 2284, 0 },
		{ 2969, 2764, 0 },
		{ 2938, 2266, 0 },
		{ 2969, 2766, 0 },
		{ 2986, 2009, 0 },
		{ 2980, 2913, 0 },
		{ 2982, 1815, 0 },
		{ 2887, 2641, 0 },
		{ 2984, 2172, 0 },
		{ 2938, 2271, 0 },
		{ 2704, 4618, 0 },
		{ 2941, 2682, 0 },
		{ 2969, 2774, 0 },
		{ 2834, 2697, 0 },
		{ 2984, 2173, 0 },
		{ 2986, 2012, 0 },
		{ 2969, 2778, 0 },
		{ 2711, 4621, 0 },
		{ 2986, 1900, 0 },
		{ 2969, 2780, 0 },
		{ 2966, 2842, 0 },
		{ 2974, 1634, 0 },
		{ 2981, 2862, 0 },
		{ 2969, 2782, 0 },
		{ 2718, 4649, 0 },
		{ 2979, 2391, 0 },
		{ 2974, 1650, 0 },
		{ 2938, 2280, 0 },
		{ 2981, 2871, 0 },
		{ 2982, 1837, 0 },
		{ 2887, 2500, 0 },
		{ 2984, 2108, 0 },
		{ 2938, 2285, 0 },
		{ 2987, 4337, 0 },
		{ 0, 0, 161 },
		{ 2729, 0, 1 },
		{ -2729, 1275, 252 },
		{ 2969, 2692, 258 },
		{ 0, 0, 258 },
		{ 2947, 1957, 0 },
		{ 2938, 2191, 0 },
		{ 2969, 2798, 0 },
		{ 2966, 2849, 0 },
		{ 2986, 2022, 0 },
		{ 0, 0, 257 },
		{ 2739, 4578, 0 },
		{ 2971, 1902, 0 },
		{ 2980, 2961, 0 },
		{ 2768, 2421, 0 },
		{ 2969, 2804, 0 },
		{ 2834, 2700, 0 },
		{ 2887, 2590, 0 },
		{ 2977, 2372, 0 },
		{ 2969, 2808, 0 },
		{ 2748, 4558, 0 },
		{ 2984, 1965, 0 },
		{ 0, 2093, 0 },
		{ 2982, 1864, 0 },
		{ 2887, 2597, 0 },
		{ 2984, 2119, 0 },
		{ 2938, 2198, 0 },
		{ 2947, 1963, 0 },
		{ 2987, 4483, 0 },
		{ 0, 0, 256 },
		{ 0, 4197, 166 },
		{ 0, 0, 166 },
		{ 2984, 2121, 0 },
		{ 2974, 1575, 0 },
		{ 2938, 2202, 0 },
		{ 2966, 2855, 0 },
		{ 2764, 4600, 0 },
		{ 2981, 2524, 0 },
		{ 2976, 2647, 0 },
		{ 2969, 2824, 0 },
		{ 2981, 2878, 0 },
		{ 0, 2422, 0 },
		{ 2887, 2610, 0 },
		{ 2938, 2203, 0 },
		{ 2790, 2664, 0 },
		{ 2987, 4329, 0 },
		{ 0, 0, 164 },
		{ 2821, 4377, 160 },
		{ 0, 0, 160 },
		{ 2982, 1869, 0 },
		{ 2778, 4607, 0 },
		{ 2982, 1831, 0 },
		{ 2976, 2656, 0 },
		{ 2969, 2833, 0 },
		{ 2782, 4634, 0 },
		{ 2941, 2680, 0 },
		{ 2969, 2835, 0 },
		{ 2790, 2670, 0 },
		{ 2887, 2616, 0 },
		{ 2938, 2209, 0 },
		{ 2938, 2211, 0 },
		{ 2887, 2619, 0 },
		{ 2938, 2213, 0 },
		{ 0, 2666, 0 },
		{ 2792, 4639, 0 },
		{ 2984, 1920, 0 },
		{ 2834, 2688, 0 },
		{ 2795, 4653, 0 },
		{ 2969, 2443, 0 },
		{ 2980, 2944, 0 },
		{ 2981, 2860, 0 },
		{ 2976, 2654, 0 },
		{ 2800, 4658, 0 },
		{ 2941, 2282, 0 },
		{ 2969, 2726, 0 },
		{ 2938, 2216, 0 },
		{ 2969, 2728, 0 },
		{ 2986, 2031, 0 },
		{ 2980, 2957, 0 },
		{ 2982, 1877, 0 },
		{ 2887, 2627, 0 },
		{ 2984, 2128, 0 },
		{ 2938, 2220, 0 },
		{ 2811, 4547, 0 },
		{ 2979, 2405, 0 },
		{ 2974, 1579, 0 },
		{ 2938, 2222, 0 },
		{ 2981, 2880, 0 },
		{ 2982, 1880, 0 },
		{ 2887, 2635, 0 },
		{ 2984, 2132, 0 },
		{ 2938, 2225, 0 },
		{ 2987, 4269, 0 },
		{ 0, 0, 155 },
		{ 0, 4251, 154 },
		{ 0, 0, 154 },
		{ 2982, 1882, 0 },
		{ 2825, 4550, 0 },
		{ 2982, 1835, 0 },
		{ 2976, 2646, 0 },
		{ 2969, 2747, 0 },
		{ 2829, 4571, 0 },
		{ 2969, 2449, 0 },
		{ 2938, 2228, 0 },
		{ 2966, 2857, 0 },
		{ 2833, 4566, 0 },
		{ 2984, 1931, 0 },
		{ 0, 2693, 0 },
		{ 2836, 4580, 0 },
		{ 2969, 2439, 0 },
		{ 2980, 2917, 0 },
		{ 2981, 2875, 0 },
		{ 2976, 2651, 0 },
		{ 2841, 4588, 0 },
		{ 2941, 2286, 0 },
		{ 2969, 2755, 0 },
		{ 2938, 2231, 0 },
		{ 2969, 2757, 0 },
		{ 2986, 2040, 0 },
		{ 2980, 2925, 0 },
		{ 2982, 1885, 0 },
		{ 2887, 2450, 0 },
		{ 2984, 2138, 0 },
		{ 2938, 2235, 0 },
		{ 2852, 4606, 0 },
		{ 2979, 2399, 0 },
		{ 2974, 1597, 0 },
		{ 2938, 2237, 0 },
		{ 2981, 2866, 0 },
		{ 2982, 1889, 0 },
		{ 2887, 2501, 0 },
		{ 2984, 2141, 0 },
		{ 2938, 2240, 0 },
		{ 2987, 4473, 0 },
		{ 0, 0, 152 },
		{ 0, 3827, 157 },
		{ 0, 0, 157 },
		{ 2938, 2241, 0 },
		{ 2947, 1977, 0 },
		{ 2982, 1890, 0 },
		{ 2969, 2773, 0 },
		{ 2980, 2946, 0 },
		{ 2966, 2848, 0 },
		{ 2871, 4638, 0 },
		{ 2969, 2437, 0 },
		{ 2951, 1472, 0 },
		{ 2980, 2950, 0 },
		{ 2977, 2403, 0 },
		{ 2974, 1599, 0 },
		{ 2980, 2954, 0 },
		{ 2982, 1894, 0 },
		{ 2887, 2517, 0 },
		{ 2984, 2147, 0 },
		{ 2938, 2248, 0 },
		{ 2882, 4655, 0 },
		{ 2979, 2397, 0 },
		{ 2974, 1600, 0 },
		{ 2938, 2250, 0 },
		{ 2981, 2868, 0 },
		{ 2982, 1897, 0 },
		{ 0, 2598, 0 },
		{ 2984, 2150, 0 },
		{ 2938, 2253, 0 },
		{ 2949, 4499, 0 },
		{ 0, 0, 156 },
		{ 2969, 2791, 416 },
		{ 2986, 1496, 25 },
		{ 2901, 0, 416 },
		{ 2180, 2596, 27 },
		{ 0, 0, 28 },
		{ 0, 0, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 2938, 2257, 0 },
		{ 2986, 661, 0 },
		{ 0, 0, 26 },
		{ 2966, 2854, 0 },
		{ 0, 0, 21 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 0, 3669, 37 },
		{ 0, 3416, 37 },
		{ 0, 0, 36 },
		{ 0, 0, 37 },
		{ 2930, 3548, 0 },
		{ 2942, 4170, 0 },
		{ 2934, 3205, 0 },
		{ 0, 0, 35 },
		{ 2937, 3267, 0 },
		{ 0, 3112, 0 },
		{ 2895, 1607, 0 },
		{ 0, 0, 34 },
		{ 2969, 2712, 47 },
		{ 0, 0, 47 },
		{ 2934, 3211, 47 },
		{ 2969, 2801, 47 },
		{ 0, 0, 50 },
		{ 2969, 2802, 0 },
		{ 2938, 2262, 0 },
		{ 2937, 3276, 0 },
		{ 2982, 1907, 0 },
		{ 2938, 2264, 0 },
		{ 2966, 2844, 0 },
		{ 0, 3511, 0 },
		{ 2974, 1635, 0 },
		{ 2984, 2160, 0 },
		{ 0, 0, 46 },
		{ 0, 3221, 0 },
		{ 2986, 2061, 0 },
		{ 2971, 2310, 0 },
		{ 0, 3287, 0 },
		{ 0, 2268, 0 },
		{ 2969, 2811, 0 },
		{ 0, 0, 48 },
		{ 0, 5, 51 },
		{ 0, 4135, 0 },
		{ 0, 0, 49 },
		{ 2977, 2384, 0 },
		{ 2980, 2929, 0 },
		{ 2947, 1993, 0 },
		{ 0, 1994, 0 },
		{ 2949, 4503, 0 },
		{ 0, 4504, 0 },
		{ 2969, 2815, 0 },
		{ 0, 1506, 0 },
		{ 2980, 2934, 0 },
		{ 2977, 2392, 0 },
		{ 2974, 1652, 0 },
		{ 2980, 2937, 0 },
		{ 2982, 1913, 0 },
		{ 2984, 2167, 0 },
		{ 2986, 2067, 0 },
		{ 2980, 2023, 0 },
		{ 2969, 2823, 0 },
		{ 2984, 2170, 0 },
		{ 2981, 2867, 0 },
		{ 2980, 2945, 0 },
		{ 2986, 2069, 0 },
		{ 2981, 2869, 0 },
		{ 0, 2846, 0 },
		{ 2969, 2730, 0 },
		{ 2974, 1654, 0 },
		{ 0, 2828, 0 },
		{ 2980, 2952, 0 },
		{ 0, 2331, 0 },
		{ 2986, 2071, 0 },
		{ 2981, 2876, 0 },
		{ 0, 1655, 0 },
		{ 2987, 4510, 0 },
		{ 0, 2653, 0 },
		{ 0, 2408, 0 },
		{ 0, 0, 43 },
		{ 2941, 2679, 0 },
		{ 0, 2960, 0 },
		{ 0, 2881, 0 },
		{ 0, 1919, 0 },
		{ 2987, 4512, 0 },
		{ 0, 2178, 0 },
		{ 0, 0, 44 },
		{ 0, 2074, 0 },
		{ 2949, 4513, 0 },
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
		0
	};
	yybackup = backup;
}
