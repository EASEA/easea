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
  lineCounter = 1;
  BEGIN COPY_INITIALISATION_FUNCTION;
 
#line 258 "EaseaLex.cpp"
		}
		break;
	case 13:
		{
#line 180 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting generation before reduce function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bGenerationFunctionBeforeReplacement = true;
  BEGIN COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT;
 
#line 271 "EaseaLex.cpp"
		}
		break;
	case 14:
		{
#line 189 "EaseaLex.l"

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
 
#line 290 "EaseaLex.cpp"
		}
		break;
	case 15:
		{
#line 208 "EaseaLex.l"

  if( bVERBOSE )printf("inserting gp parameters\n");
  //  fprintf(fpOutputFile,"#define MAX_XOVER_DEPTH",%d
  fprintf(fpOutputFile,"#define TREE_DEPTH_MAX %d\n",iMAX_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MAX %d\n",iMAX_INIT_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MIN %d\n",iMIN_INIT_TREE_D);

  fprintf(fpOutputFile,"#define MAX_PROGS_SIZE %d\n",iPRG_BUF_SIZE);
  fprintf(fpOutputFile,"#define NB_GPU %d\n",iNB_GPU);

  fprintf(fpOutputFile,"#define NO_FITNESS_CASES %d\n",iNO_FITNESS_CASES);

#line 308 "EaseaLex.cpp"
		}
		break;
	case 16:
		{
#line 226 "EaseaLex.l"

  
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
 
#line 345 "EaseaLex.cpp"
		}
		break;
	case 17:
		{
#line 258 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"    case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"      %s",opDesc[i]->gpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"      break;\n");

  }
 
#line 359 "EaseaLex.cpp"
		}
		break;
	case 18:
		{
#line 267 "EaseaLex.l"

  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"  case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"    %s\n",opDesc[i]->cpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"    break;\n");
  }
 
#line 372 "EaseaLex.cpp"
		}
		break;
	case 19:
		{
#line 276 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Analysing GP OP code from ez file\n");
  BEGIN COPY_GP_OPCODE;
 
#line 385 "EaseaLex.cpp"
		}
		break;
	case 20:
		{
#line 285 "EaseaLex.l"

  if (bVERBOSE) printf ("found begin section\n");
  bGPOPCODE_ANALYSIS = true;
  BEGIN GP_RULE_ANALYSIS;
 
#line 396 "EaseaLex.cpp"
		}
		break;
	case 21:
		{
#line 291 "EaseaLex.l"
 
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
 
#line 416 "EaseaLex.cpp"
		}
		break;
	case 22:
		{
#line 306 "EaseaLex.l"

  if (bVERBOSE) printf("*** No GP OP codes were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
 
#line 428 "EaseaLex.cpp"
		}
		break;
	case 23:
		{
#line 312 "EaseaLex.l"

#line 435 "EaseaLex.cpp"
		}
		break;
	case 24:
		{
#line 313 "EaseaLex.l"
if( bGPOPCODE_ANALYSIS )printf("\n");lineCounter++;
#line 442 "EaseaLex.cpp"
		}
		break;
	case 25:
		{
#line 320 "EaseaLex.l"

  // this rule match the OP_NAME
  if( iGP_OPCODE_FIELD != 0 ){
    fprintf(stderr,"Error, OP_CODE name must be given first\n");
    exit(-1);
  }
  opDesc[iNoOp] = new OPCodeDesc();
  opDesc[iNoOp]->opcode = new string(yytext);
 
#line 457 "EaseaLex.cpp"
		}
		break;
	case 26:
		{
#line 330 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 1 ){
    fprintf(stderr,"Error, op code real name must be given at the second place\n");
    exit(-1);
  }
  opDesc[iNoOp]->realName = new string(yytext);
 
#line 470 "EaseaLex.cpp"
		}
		break;
	case 27:
		{
#line 339 "EaseaLex.l"

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
 
#line 489 "EaseaLex.cpp"
		}
		break;
	case 28:
		{
#line 353 "EaseaLex.l"

#line 496 "EaseaLex.cpp"
		}
		break;
	case 29:
		{
#line 354 "EaseaLex.l"

  iGP_OPCODE_FIELD = 0;
  iNoOp++;
 
#line 506 "EaseaLex.cpp"
		}
		break;
	case 30:
		{
#line 359 "EaseaLex.l"

  if( bGPOPCODE_ANALYSIS ) iGP_OPCODE_FIELD++;
 
#line 515 "EaseaLex.cpp"
		}
		break;
	case 31:
		{
#line 364 "EaseaLex.l"

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
 
#line 537 "EaseaLex.cpp"
		}
		break;
	case 32:
		{
#line 385 "EaseaLex.l"

  accolade_counter++;
  opDesc[iNoOp]->cpuCodeStream << "{";
  opDesc[iNoOp]->gpuCodeStream << "{";
 
#line 548 "EaseaLex.cpp"
		}
		break;
	case 33:
		{
#line 391 "EaseaLex.l"

  accolade_counter--;
  if( accolade_counter==0 ){
    opDesc[iNoOp]->gpuCodeStream << "\n      stack[sp++] = RESULT;\n";

    BEGIN GP_RULE_ANALYSIS;
  }
  else{
    opDesc[iNoOp]->cpuCodeStream << "}";
    opDesc[iNoOp]->gpuCodeStream << "}";
  }
 
#line 566 "EaseaLex.cpp"
		}
		break;
	case 34:
		{
#line 404 "EaseaLex.l"

  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
  printf("input no : %d\n",no_input);
  opDesc[iNoOp]->cpuCodeStream << "input["<< no_input <<"]" ;
  opDesc[iNoOp]->gpuCodeStream << "input["<< no_input << "]";  
 
#line 579 "EaseaLex.cpp"
		}
		break;
	case 35:
		{
#line 412 "EaseaLex.l"

  opDesc[iNoOp]->isERC = true;
  opDesc[iNoOp]->cpuCodeStream << "root->erc_value" ;
  opDesc[iNoOp]->gpuCodeStream << "k_progs[start_prog++];" ;
  printf("ERC matched\n");

#line 591 "EaseaLex.cpp"
		}
		break;
	case 36:
		{
#line 419 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << "\n  ";
  opDesc[iNoOp]->gpuCodeStream << "\n    ";
 
#line 601 "EaseaLex.cpp"
		}
		break;
	case 37:
		{
#line 425 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << yytext;
  opDesc[iNoOp]->gpuCodeStream << yytext;
 
#line 611 "EaseaLex.cpp"
		}
		break;
	case 38:
		{
#line 430 "EaseaLex.l"
 
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  if( bVERBOSE ) printf("Insert GP eval header\n");
  iCOPY_GP_EVAL_STATUS = EVAL_HDR;
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 627 "EaseaLex.cpp"
		}
		break;
	case 39:
		{
#line 441 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  BEGIN COPY_GP_EVAL;
 
#line 643 "EaseaLex.cpp"
		}
		break;
	case 40:
		{
#line 452 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 659 "EaseaLex.cpp"
		}
		break;
	case 41:
		{
#line 463 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 676 "EaseaLex.cpp"
		}
		break;
	case 42:
		{
#line 476 "EaseaLex.l"

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
 
#line 695 "EaseaLex.cpp"
		}
		break;
	case 43:
		{
#line 491 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_HDR){
    bIsCopyingGPEval = true;
  }
 
#line 706 "EaseaLex.cpp"
		}
		break;
	case 44:
		{
#line 497 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_BDY){
    bIsCopyingGPEval = true;
  }
 
#line 717 "EaseaLex.cpp"
		}
		break;
	case 45:
		{
#line 505 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_FTR){
    bIsCopyingGPEval = true;
  }
 
#line 728 "EaseaLex.cpp"
		}
		break;
	case 46:
		{
#line 511 "EaseaLex.l"

  if( bIsCopyingGPEval ){
    bIsCopyingGPEval = false;
    bCOPY_GP_EVAL_GPU = false;
    rewind(fpGenomeFile);
    yyin = fpTemplateFile;
    BEGIN TEMPLATE_ANALYSIS;
  }
 
#line 743 "EaseaLex.cpp"
		}
		break;
	case 47:
		{
#line 521 "EaseaLex.l"

  if( bIsCopyingGPEval ) fprintf(fpOutputFile,"%s",yytext);
 
#line 752 "EaseaLex.cpp"
		}
		break;
	case 48:
		{
#line 525 "EaseaLex.l"

  if( bIsCopyingGPEval) fprintf(fpOutputFile, "outputs[i]" );
 
#line 761 "EaseaLex.cpp"
		}
		break;
	case 49:
		{
#line 529 "EaseaLex.l"

  if( bIsCopyingGPEval )
    if( iCOPY_GP_EVAL_STATUS==EVAL_FTR )
      if( bCOPY_GP_EVAL_GPU ){
	fprintf(fpOutputFile,"k_results[index] =");
      }
      else fprintf(fpOutputFile,"%s",yytext);
 
#line 775 "EaseaLex.cpp"
		}
		break;
	case 50:
		{
#line 541 "EaseaLex.l"

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
 
#line 793 "EaseaLex.cpp"
		}
		break;
	case 51:
		{
#line 555 "EaseaLex.l"

  if( bIsCopyingGPEval )
    fprintf(fpOutputFile,"return fitness = "); 
 
#line 803 "EaseaLex.cpp"
		}
		break;
	case 52:
		{
#line 562 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  yyreset();
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Evaluation population in a single function!!.\n");
  lineCounter = 1;
  BEGIN COPY_INSTEAD_EVAL;
 
#line 817 "EaseaLex.cpp"
		}
		break;
	case 53:
		{
#line 571 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting at the end of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bEndGeneration = true;
  bBeginGeneration = false;
  BEGIN COPY_END_GENERATION_FUNCTION;
 
#line 831 "EaseaLex.cpp"
		}
		break;
	case 54:
		{
#line 580 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Bound Checking function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_BOUND_CHECKING_FUNCTION;
 
#line 843 "EaseaLex.cpp"
		}
		break;
	case 55:
		{
#line 587 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 
#line 855 "EaseaLex.cpp"
		}
		break;
	case 56:
		{
#line 594 "EaseaLex.l"

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
 
#line 884 "EaseaLex.cpp"
		}
		break;
	case 57:
		{
#line 617 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 
#line 901 "EaseaLex.cpp"
		}
		break;
	case 58:
		{
#line 629 "EaseaLex.l"

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
 
#line 927 "EaseaLex.cpp"
		}
		break;
	case 59:
		{
#line 650 "EaseaLex.l"

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
  
 
#line 948 "EaseaLex.cpp"
		}
		break;
#line 668 "EaseaLex.l"
  
#line 682 "EaseaLex.l"
      
#line 955 "EaseaLex.cpp"
	case 60:
		{
#line 690 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 968 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 699 "EaseaLex.l"

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
 
#line 991 "EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 716 "EaseaLex.l"

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
 
#line 1014 "EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 734 "EaseaLex.l"

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
 
#line 1046 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 761 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  fprintf (fpOutputFile,"// Memberwise serialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->serializeIndividual(fpOutputFile, "this");
  //fprintf(fpOutputFile,"\tEASEA_Line << endl;\n");
 
#line 1060 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 770 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome deserializer.\n");
  fprintf (fpOutputFile,"// Memberwise deserialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->deserializeIndividual(fpOutputFile, "this");
 
#line 1073 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 778 "EaseaLex.l"

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
 
#line 1094 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 794 "EaseaLex.l"

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
 
#line 1116 "EaseaLex.cpp"
		}
		break;
	case 68:
		{
#line 811 "EaseaLex.l"
       
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
 
#line 1144 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 833 "EaseaLex.l"

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
 
#line 1166 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 849 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 1181 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 858 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 1193 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 866 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 1205 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 873 "EaseaLex.l"

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
 
#line 1236 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 898 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 1249 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 905 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 1263 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 914 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISER;   
 
#line 1275 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 921 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 1288 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 929 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 1300 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 935 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 1312 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 941 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 1324 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 947 "EaseaLex.l"
      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 1337 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 954 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 1350 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 961 "EaseaLex.l"
 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 1364 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 970 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 1375 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 975 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 1389 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 984 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 1403 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 993 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 1417 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 1003 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 1430 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 1011 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 1439 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 1015 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 1448 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 1019 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 1457 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 1023 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 1466 "EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 1027 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 1476 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 1032 "EaseaLex.l"

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

#line 1495 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 1045 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 1502 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 1046 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 1509 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 1047 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 1516 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 1048 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 1523 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 1049 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 1530 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 1050 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1537 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 1051 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1544 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 1052 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1551 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 1053 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1558 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 1054 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1565 "EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 1055 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",nELITE);
  ////DEBUG_PRT_PRT("elitism is %d, elite size is %d",bELITISM, nELITE);
 
#line 1575 "EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 1060 "EaseaLex.l"

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
 
#line 1594 "EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 1073 "EaseaLex.l"

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
 
#line 1613 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 1086 "EaseaLex.l"

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
 
#line 1632 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 1099 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1642 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 1103 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1649 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 1104 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1656 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 1105 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1663 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 1106 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1670 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 1107 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1677 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 1108 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1684 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 1109 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1691 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 1110 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1698 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 1111 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1705 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 1112 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1712 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 1114 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1719 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 1115 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1726 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1117 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bREMOTE_ISLAND_MODEL);
#line 1733 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1119 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1740 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1120 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1747 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1121 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1754 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1122 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1761 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1123 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1768 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1125 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 1775 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1126 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 1782 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1128 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1796 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1136 "EaseaLex.l"

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
 
#line 1816 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1150 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1830 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1158 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1844 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1167 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1858 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1176 "EaseaLex.l"

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

#line 1921 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1233 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded with no warning.\n");
  printf ("Have a nice compile time.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 1938 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1245 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1945 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1251 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1957 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1257 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1970 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1264 "EaseaLex.l"

#line 1977 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1265 "EaseaLex.l"
lineCounter++;
#line 1984 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1267 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1996 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1273 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2009 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1281 "EaseaLex.l"

#line 2016 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1282 "EaseaLex.l"

  lineCounter++;
 
#line 2025 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1286 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2037 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1292 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2051 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1300 "EaseaLex.l"

#line 2058 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1301 "EaseaLex.l"

  lineCounter++;
 
#line 2067 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1305 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  
  BEGIN COPY;
 
#line 2081 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1313 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2096 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1322 "EaseaLex.l"

#line 2103 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1323 "EaseaLex.l"
lineCounter++;
#line 2110 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1328 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2124 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1337 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2138 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1345 "EaseaLex.l"

#line 2145 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1346 "EaseaLex.l"
lineCounter++;
#line 2152 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1349 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2168 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1360 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2184 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1370 "EaseaLex.l"

#line 2191 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1373 "EaseaLex.l"

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
 
#line 2209 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1386 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2226 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1398 "EaseaLex.l"

#line 2233 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1399 "EaseaLex.l"
lineCounter++;
#line 2240 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1401 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2256 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1413 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2272 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1423 "EaseaLex.l"
lineCounter++;
#line 2279 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1424 "EaseaLex.l"

#line 2286 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1428 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2301 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1438 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2316 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1447 "EaseaLex.l"

#line 2323 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1450 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2336 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1457 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2350 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1465 "EaseaLex.l"

#line 2357 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1469 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2365 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1471 "EaseaLex.l"

#line 2372 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1477 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2379 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1478 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2386 "EaseaLex.cpp"
		}
		break;
	case 179:
	case 180:
		{
#line 1481 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2397 "EaseaLex.cpp"
		}
		break;
	case 181:
	case 182:
		{
#line 1486 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2406 "EaseaLex.cpp"
		}
		break;
	case 183:
	case 184:
		{
#line 1489 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2415 "EaseaLex.cpp"
		}
		break;
	case 185:
	case 186:
		{
#line 1492 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2432 "EaseaLex.cpp"
		}
		break;
	case 187:
	case 188:
		{
#line 1503 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2446 "EaseaLex.cpp"
		}
		break;
	case 189:
	case 190:
		{
#line 1511 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2455 "EaseaLex.cpp"
		}
		break;
	case 191:
	case 192:
		{
#line 1514 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2464 "EaseaLex.cpp"
		}
		break;
	case 193:
	case 194:
		{
#line 1517 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2473 "EaseaLex.cpp"
		}
		break;
	case 195:
	case 196:
		{
#line 1520 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2482 "EaseaLex.cpp"
		}
		break;
	case 197:
	case 198:
		{
#line 1523 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2491 "EaseaLex.cpp"
		}
		break;
	case 199:
	case 200:
		{
#line 1527 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2503 "EaseaLex.cpp"
		}
		break;
	case 201:
		{
#line 1533 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2510 "EaseaLex.cpp"
		}
		break;
	case 202:
		{
#line 1534 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2517 "EaseaLex.cpp"
		}
		break;
	case 203:
		{
#line 1535 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2524 "EaseaLex.cpp"
		}
		break;
	case 204:
		{
#line 1536 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2534 "EaseaLex.cpp"
		}
		break;
	case 205:
		{
#line 1541 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2541 "EaseaLex.cpp"
		}
		break;
	case 206:
		{
#line 1542 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2548 "EaseaLex.cpp"
		}
		break;
	case 207:
		{
#line 1543 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2555 "EaseaLex.cpp"
		}
		break;
	case 208:
		{
#line 1544 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2562 "EaseaLex.cpp"
		}
		break;
	case 209:
		{
#line 1545 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2569 "EaseaLex.cpp"
		}
		break;
	case 210:
		{
#line 1546 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2576 "EaseaLex.cpp"
		}
		break;
	case 211:
		{
#line 1547 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2583 "EaseaLex.cpp"
		}
		break;
	case 212:
		{
#line 1548 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2590 "EaseaLex.cpp"
		}
		break;
	case 213:
		{
#line 1549 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2598 "EaseaLex.cpp"
		}
		break;
	case 214:
		{
#line 1551 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2606 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1553 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2614 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1555 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2624 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1559 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2631 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1560 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2638 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1561 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2649 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1566 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2656 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1567 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2665 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1570 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2677 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1576 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2686 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1579 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2698 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1585 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2709 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1590 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2725 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1600 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2732 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1603 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2741 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1606 "EaseaLex.l"
BEGIN COPY;
#line 2748 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1608 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2755 "EaseaLex.cpp"
		}
		break;
	case 231:
	case 232:
	case 233:
		{
#line 1611 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2768 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1616 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2779 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1621 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 2788 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1630 "EaseaLex.l"
;
#line 2795 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1631 "EaseaLex.l"
;
#line 2802 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1632 "EaseaLex.l"
;
#line 2809 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1633 "EaseaLex.l"
;
#line 2816 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1636 "EaseaLex.l"
 /* do nothing */ 
#line 2823 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1637 "EaseaLex.l"
 /*return '\n';*/ 
#line 2830 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1638 "EaseaLex.l"
 /*return '\n';*/ 
#line 2837 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1641 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 2846 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1644 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 2856 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1648 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
  printf("match gpnode\n");
  return GPNODE;
 
#line 2868 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1655 "EaseaLex.l"
return STATIC;
#line 2875 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1656 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 2882 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1657 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 2889 "EaseaLex.cpp"
		}
		break;
	case 249:
		{
#line 1658 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 2896 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1659 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 2903 "EaseaLex.cpp"
		}
		break;
	case 251:
		{
#line 1660 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 2910 "EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1662 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 2917 "EaseaLex.cpp"
		}
		break;
#line 1663 "EaseaLex.l"
  
#line 2922 "EaseaLex.cpp"
	case 253:
		{
#line 1664 "EaseaLex.l"
return GENOME; 
#line 2927 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1666 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 2937 "EaseaLex.cpp"
		}
		break;
	case 255:
	case 256:
	case 257:
		{
#line 1673 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 2946 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1674 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 2953 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1677 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 2961 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1679 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 2968 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1685 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 2980 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1691 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2993 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1698 "EaseaLex.l"

#line 3000 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1700 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3011 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1711 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3026 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1721 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3037 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1727 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3046 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1731 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3061 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1744 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3073 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1750 "EaseaLex.l"

#line 3080 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1751 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3093 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1758 "EaseaLex.l"

#line 3100 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1759 "EaseaLex.l"
lineCounter++;
#line 3107 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1760 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3120 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1767 "EaseaLex.l"

#line 3127 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1768 "EaseaLex.l"
lineCounter++;
#line 3134 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1770 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3147 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1777 "EaseaLex.l"

#line 3154 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1778 "EaseaLex.l"
lineCounter++;
#line 3161 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1780 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3174 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1787 "EaseaLex.l"

#line 3181 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1788 "EaseaLex.l"
lineCounter++;
#line 3188 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1794 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3195 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1795 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3202 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1796 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3209 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1797 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3216 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1798 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3223 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1799 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3230 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1800 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3237 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1802 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3246 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1805 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3259 "EaseaLex.cpp"
		}
		break;
	case 292:
	case 293:
		{
#line 1814 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3270 "EaseaLex.cpp"
		}
		break;
	case 294:
	case 295:
		{
#line 1819 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3279 "EaseaLex.cpp"
		}
		break;
	case 296:
	case 297:
		{
#line 1822 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3288 "EaseaLex.cpp"
		}
		break;
	case 298:
	case 299:
		{
#line 1825 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3300 "EaseaLex.cpp"
		}
		break;
	case 300:
	case 301:
		{
#line 1831 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3313 "EaseaLex.cpp"
		}
		break;
	case 302:
	case 303:
		{
#line 1838 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3322 "EaseaLex.cpp"
		}
		break;
	case 304:
	case 305:
		{
#line 1841 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3331 "EaseaLex.cpp"
		}
		break;
	case 306:
	case 307:
		{
#line 1844 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3340 "EaseaLex.cpp"
		}
		break;
	case 308:
	case 309:
		{
#line 1847 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3349 "EaseaLex.cpp"
		}
		break;
	case 310:
	case 311:
		{
#line 1850 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3358 "EaseaLex.cpp"
		}
		break;
	case 312:
		{
#line 1853 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3367 "EaseaLex.cpp"
		}
		break;
	case 313:
		{
#line 1856 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3377 "EaseaLex.cpp"
		}
		break;
	case 314:
		{
#line 1860 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3385 "EaseaLex.cpp"
		}
		break;
	case 315:
		{
#line 1862 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3396 "EaseaLex.cpp"
		}
		break;
	case 316:
		{
#line 1867 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3407 "EaseaLex.cpp"
		}
		break;
	case 317:
		{
#line 1872 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3415 "EaseaLex.cpp"
		}
		break;
	case 318:
		{
#line 1874 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3423 "EaseaLex.cpp"
		}
		break;
	case 319:
		{
#line 1876 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3431 "EaseaLex.cpp"
		}
		break;
	case 320:
		{
#line 1878 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3439 "EaseaLex.cpp"
		}
		break;
	case 321:
		{
#line 1880 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3447 "EaseaLex.cpp"
		}
		break;
	case 322:
		{
#line 1882 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3454 "EaseaLex.cpp"
		}
		break;
	case 323:
		{
#line 1883 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3461 "EaseaLex.cpp"
		}
		break;
	case 324:
		{
#line 1884 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3469 "EaseaLex.cpp"
		}
		break;
	case 325:
		{
#line 1886 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3477 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1888 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3485 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1890 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3492 "EaseaLex.cpp"
		}
		break;
	case 328:
		{
#line 1891 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3504 "EaseaLex.cpp"
		}
		break;
	case 329:
		{
#line 1897 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3513 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1900 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3523 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1904 "EaseaLex.l"
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
#line 3540 "EaseaLex.cpp"
		}
		break;
	case 332:
	case 333:
		{
#line 1916 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3550 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 1919 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3557 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 1926 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3564 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 1927 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3571 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 1928 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3578 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1929 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3585 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 1930 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3592 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 1932 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3601 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 1936 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3614 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 1944 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3627 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 1953 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3640 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 1962 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3655 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 1972 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3662 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 1973 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3669 "EaseaLex.cpp"
		}
		break;
	case 347:
	case 348:
		{
#line 1976 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3680 "EaseaLex.cpp"
		}
		break;
	case 349:
	case 350:
		{
#line 1981 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3689 "EaseaLex.cpp"
		}
		break;
	case 351:
	case 352:
		{
#line 1984 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3698 "EaseaLex.cpp"
		}
		break;
	case 353:
	case 354:
		{
#line 1987 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3711 "EaseaLex.cpp"
		}
		break;
	case 355:
	case 356:
		{
#line 1994 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3724 "EaseaLex.cpp"
		}
		break;
	case 357:
	case 358:
		{
#line 2001 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3733 "EaseaLex.cpp"
		}
		break;
	case 359:
		{
#line 2004 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3740 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 2005 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3747 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 2006 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3754 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 2007 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3764 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 2012 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3771 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 2013 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3778 "EaseaLex.cpp"
		}
		break;
	case 365:
		{
#line 2014 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3785 "EaseaLex.cpp"
		}
		break;
	case 366:
		{
#line 2015 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3792 "EaseaLex.cpp"
		}
		break;
	case 367:
		{
#line 2016 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3800 "EaseaLex.cpp"
		}
		break;
	case 368:
		{
#line 2018 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3808 "EaseaLex.cpp"
		}
		break;
	case 369:
		{
#line 2020 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3816 "EaseaLex.cpp"
		}
		break;
	case 370:
		{
#line 2022 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3824 "EaseaLex.cpp"
		}
		break;
	case 371:
		{
#line 2024 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 3832 "EaseaLex.cpp"
		}
		break;
	case 372:
		{
#line 2026 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 3840 "EaseaLex.cpp"
		}
		break;
	case 373:
		{
#line 2028 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 3848 "EaseaLex.cpp"
		}
		break;
	case 374:
		{
#line 2030 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3855 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2031 "EaseaLex.l"
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
#line 3878 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2048 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 3889 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2053 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 3903 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2061 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3910 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2067 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 3920 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2071 "EaseaLex.l"

#line 3927 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2074 "EaseaLex.l"
;
#line 3934 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2075 "EaseaLex.l"
;
#line 3941 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2076 "EaseaLex.l"
;
#line 3948 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2077 "EaseaLex.l"
;
#line 3955 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2079 "EaseaLex.l"
 /* do nothing */ 
#line 3962 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2080 "EaseaLex.l"
 /*return '\n';*/ 
#line 3969 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2081 "EaseaLex.l"
 /*return '\n';*/ 
#line 3976 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2083 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 3983 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2084 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 3990 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2085 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 3997 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2086 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4004 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2087 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4011 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2088 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4018 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2089 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4025 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2090 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4032 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2091 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4039 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2093 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4046 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2094 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4053 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2095 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4060 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2096 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4067 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2097 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4074 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2099 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4081 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2100 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4088 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2102 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4099 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2107 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4106 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2109 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4117 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2114 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4124 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2117 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4131 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2118 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4138 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2119 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4145 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2120 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4152 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2121 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4159 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2123 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4166 "EaseaLex.cpp"
		}
		break;
#line 2125 "EaseaLex.l"
 
#line 4171 "EaseaLex.cpp"
	case 414:
	case 415:
	case 416:
		{
#line 2129 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4178 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2130 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4185 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2133 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4193 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2136 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4200 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2138 "EaseaLex.l"

  lineCounter++;

#line 4209 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2141 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4219 "EaseaLex.cpp"
		}
		break;
	case 422:
		{
#line 2146 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4229 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2151 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4239 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2156 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4249 "EaseaLex.cpp"
		}
		break;
	case 425:
		{
#line 2161 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4259 "EaseaLex.cpp"
		}
		break;
	case 426:
		{
#line 2166 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4269 "EaseaLex.cpp"
		}
		break;
	case 427:
		{
#line 2175 "EaseaLex.l"
return  (char)yytext[0];
#line 4276 "EaseaLex.cpp"
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
#line 2177 "EaseaLex.l"


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

#line 4464 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		179,
		-180,
		0,
		181,
		-182,
		0,
		183,
		-184,
		0,
		185,
		-186,
		0,
		191,
		-192,
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
		189,
		-190,
		0,
		187,
		-188,
		0,
		-229,
		0,
		-235,
		0,
		296,
		-297,
		0,
		298,
		-299,
		0,
		292,
		-293,
		0,
		304,
		-305,
		0,
		306,
		-307,
		0,
		308,
		-309,
		0,
		310,
		-311,
		0,
		294,
		-295,
		0,
		357,
		-358,
		0,
		302,
		-303,
		0,
		355,
		-356,
		0,
		300,
		-301,
		0,
		349,
		-350,
		0,
		351,
		-352,
		0,
		353,
		-354,
		0,
		347,
		-348,
		0
	};
	yymatch = match;

	yytransitionmax = 4934;
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
		{ 2995, 61 },
		{ 2995, 61 },
		{ 1855, 1958 },
		{ 1495, 1495 },
		{ 67, 61 },
		{ 2352, 2327 },
		{ 2352, 2327 },
		{ 2331, 2302 },
		{ 2331, 2302 },
		{ 1954, 1956 },
		{ 0, 87 },
		{ 71, 3 },
		{ 72, 3 },
		{ 2214, 43 },
		{ 2215, 43 },
		{ 1977, 39 },
		{ 69, 1 },
		{ 1954, 1950 },
		{ 0, 2235 },
		{ 67, 1 },
		{ 2755, 2751 },
		{ 2180, 2182 },
		{ 165, 161 },
		{ 2995, 61 },
		{ 2197, 2196 },
		{ 2993, 61 },
		{ 1495, 1495 },
		{ 3042, 3040 },
		{ 2352, 2327 },
		{ 1337, 1336 },
		{ 2331, 2302 },
		{ 1328, 1327 },
		{ 1480, 1463 },
		{ 1481, 1463 },
		{ 71, 3 },
		{ 2997, 61 },
		{ 2214, 43 },
		{ 2027, 2006 },
		{ 1977, 39 },
		{ 2994, 61 },
		{ 2994, 61 },
		{ 2994, 61 },
		{ 2994, 61 },
		{ 2994, 61 },
		{ 2994, 61 },
		{ 2994, 61 },
		{ 2994, 61 },
		{ 2994, 61 },
		{ 2994, 61 },
		{ 70, 3 },
		{ 2996, 61 },
		{ 2213, 43 },
		{ 1570, 1564 },
		{ 1963, 39 },
		{ 2353, 2327 },
		{ 1480, 1463 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 1572, 1566 },
		{ 2991, 61 },
		{ 1482, 1463 },
		{ 1443, 1422 },
		{ 2992, 61 },
		{ 1444, 1423 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2992, 61 },
		{ 2998, 61 },
		{ 2175, 40 },
		{ 1517, 1501 },
		{ 1518, 1501 },
		{ 1436, 1414 },
		{ 1962, 40 },
		{ 2405, 2379 },
		{ 2405, 2379 },
		{ 2334, 2305 },
		{ 2334, 2305 },
		{ 2350, 2325 },
		{ 2350, 2325 },
		{ 1788, 37 },
		{ 2358, 2332 },
		{ 2358, 2332 },
		{ 2370, 2344 },
		{ 2370, 2344 },
		{ 1437, 1415 },
		{ 1438, 1416 },
		{ 1439, 1417 },
		{ 1440, 1418 },
		{ 1442, 1421 },
		{ 1445, 1424 },
		{ 1446, 1425 },
		{ 2175, 40 },
		{ 1517, 1501 },
		{ 1965, 40 },
		{ 1447, 1426 },
		{ 1448, 1427 },
		{ 2405, 2379 },
		{ 1449, 1428 },
		{ 2334, 2305 },
		{ 1450, 1429 },
		{ 2350, 2325 },
		{ 1451, 1430 },
		{ 1788, 37 },
		{ 2358, 2332 },
		{ 1452, 1431 },
		{ 2370, 2344 },
		{ 2174, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1775, 37 },
		{ 1978, 40 },
		{ 1453, 1433 },
		{ 1456, 1436 },
		{ 1519, 1501 },
		{ 2406, 2379 },
		{ 1457, 1437 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1964, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1972, 40 },
		{ 1970, 40 },
		{ 1983, 40 },
		{ 1971, 40 },
		{ 1983, 40 },
		{ 1974, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1973, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1458, 1438 },
		{ 1966, 40 },
		{ 1968, 40 },
		{ 1459, 1439 },
		{ 1983, 40 },
		{ 1460, 1440 },
		{ 1983, 40 },
		{ 1981, 40 },
		{ 1969, 40 },
		{ 1983, 40 },
		{ 1982, 40 },
		{ 1975, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1980, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1967, 40 },
		{ 1983, 40 },
		{ 1979, 40 },
		{ 1983, 40 },
		{ 1976, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1983, 40 },
		{ 1364, 21 },
		{ 1520, 1502 },
		{ 1521, 1502 },
		{ 1462, 1442 },
		{ 1351, 21 },
		{ 2372, 2346 },
		{ 2372, 2346 },
		{ 2394, 2368 },
		{ 2394, 2368 },
		{ 2395, 2369 },
		{ 2395, 2369 },
		{ 2435, 2409 },
		{ 2435, 2409 },
		{ 2440, 2414 },
		{ 2440, 2414 },
		{ 2446, 2420 },
		{ 2446, 2420 },
		{ 1463, 1443 },
		{ 1464, 1444 },
		{ 1465, 1445 },
		{ 1466, 1446 },
		{ 1467, 1447 },
		{ 1468, 1448 },
		{ 1364, 21 },
		{ 1520, 1502 },
		{ 1352, 21 },
		{ 1365, 21 },
		{ 1469, 1449 },
		{ 2372, 2346 },
		{ 1470, 1450 },
		{ 2394, 2368 },
		{ 1471, 1451 },
		{ 2395, 2369 },
		{ 1472, 1453 },
		{ 2435, 2409 },
		{ 1475, 1456 },
		{ 2440, 2414 },
		{ 1476, 1457 },
		{ 2446, 2420 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1477, 1458 },
		{ 1478, 1460 },
		{ 1479, 1462 },
		{ 1376, 1354 },
		{ 1522, 1502 },
		{ 1483, 1464 },
		{ 1484, 1465 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1368, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1357, 21 },
		{ 1355, 21 },
		{ 1370, 21 },
		{ 1356, 21 },
		{ 1370, 21 },
		{ 1359, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1358, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1489, 1468 },
		{ 1353, 21 },
		{ 1366, 21 },
		{ 1490, 1469 },
		{ 1360, 21 },
		{ 1491, 1470 },
		{ 1370, 21 },
		{ 1371, 21 },
		{ 1354, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1361, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1369, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1372, 21 },
		{ 1370, 21 },
		{ 1367, 21 },
		{ 1370, 21 },
		{ 1362, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1370, 21 },
		{ 1949, 38 },
		{ 1523, 1503 },
		{ 1524, 1503 },
		{ 1485, 1466 },
		{ 1774, 38 },
		{ 2459, 2433 },
		{ 2459, 2433 },
		{ 2460, 2434 },
		{ 2460, 2434 },
		{ 1487, 1467 },
		{ 1486, 1466 },
		{ 2464, 2438 },
		{ 2464, 2438 },
		{ 2470, 2444 },
		{ 2470, 2444 },
		{ 1492, 1471 },
		{ 1488, 1467 },
		{ 1493, 1472 },
		{ 1496, 1476 },
		{ 1497, 1477 },
		{ 1498, 1478 },
		{ 1499, 1479 },
		{ 1501, 1483 },
		{ 1949, 38 },
		{ 1523, 1503 },
		{ 1779, 38 },
		{ 2471, 2445 },
		{ 2471, 2445 },
		{ 2459, 2433 },
		{ 1502, 1484 },
		{ 2460, 2434 },
		{ 1503, 1485 },
		{ 1526, 1504 },
		{ 1527, 1504 },
		{ 2464, 2438 },
		{ 1504, 1486 },
		{ 2470, 2444 },
		{ 1505, 1487 },
		{ 1948, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 2471, 2445 },
		{ 1789, 38 },
		{ 1506, 1488 },
		{ 1507, 1489 },
		{ 1525, 1503 },
		{ 1508, 1490 },
		{ 1526, 1504 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1776, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1784, 38 },
		{ 1782, 38 },
		{ 1792, 38 },
		{ 1783, 38 },
		{ 1792, 38 },
		{ 1786, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1785, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1509, 1491 },
		{ 1780, 38 },
		{ 1528, 1504 },
		{ 1510, 1492 },
		{ 1792, 38 },
		{ 1511, 1493 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1781, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1777, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1778, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1791, 38 },
		{ 1792, 38 },
		{ 1790, 38 },
		{ 1792, 38 },
		{ 1787, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 1792, 38 },
		{ 2749, 44 },
		{ 2750, 44 },
		{ 1529, 1505 },
		{ 1530, 1505 },
		{ 67, 44 },
		{ 2265, 2238 },
		{ 2265, 2238 },
		{ 1513, 1496 },
		{ 1514, 1497 },
		{ 1515, 1498 },
		{ 1516, 1499 },
		{ 2307, 2278 },
		{ 2307, 2278 },
		{ 2308, 2279 },
		{ 2308, 2279 },
		{ 1379, 1355 },
		{ 1380, 1356 },
		{ 1384, 1358 },
		{ 1385, 1359 },
		{ 1386, 1360 },
		{ 1535, 1507 },
		{ 1536, 1508 },
		{ 1537, 1509 },
		{ 2749, 44 },
		{ 1539, 1513 },
		{ 1529, 1505 },
		{ 2323, 2295 },
		{ 2323, 2295 },
		{ 2265, 2238 },
		{ 1540, 1514 },
		{ 1532, 1506 },
		{ 1533, 1506 },
		{ 1550, 1536 },
		{ 1551, 1536 },
		{ 2307, 2278 },
		{ 1541, 1515 },
		{ 2308, 2279 },
		{ 2229, 44 },
		{ 2748, 44 },
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
		{ 2323, 2295 },
		{ 1542, 1516 },
		{ 1549, 1535 },
		{ 1387, 1361 },
		{ 1532, 1506 },
		{ 1531, 1505 },
		{ 1550, 1536 },
		{ 2230, 44 },
		{ 2227, 44 },
		{ 2222, 44 },
		{ 2230, 44 },
		{ 2219, 44 },
		{ 2226, 44 },
		{ 2224, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2221, 44 },
		{ 2216, 44 },
		{ 2223, 44 },
		{ 2218, 44 },
		{ 2230, 44 },
		{ 2225, 44 },
		{ 2220, 44 },
		{ 2217, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 1534, 1506 },
		{ 2234, 44 },
		{ 1552, 1536 },
		{ 1553, 1537 },
		{ 2230, 44 },
		{ 1555, 1539 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2231, 44 },
		{ 2232, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2233, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 2230, 44 },
		{ 159, 4 },
		{ 160, 4 },
		{ 1559, 1549 },
		{ 1560, 1549 },
		{ 2485, 2456 },
		{ 2485, 2456 },
		{ 2326, 2298 },
		{ 2326, 2298 },
		{ 1383, 1357 },
		{ 1556, 1540 },
		{ 1557, 1541 },
		{ 1558, 1542 },
		{ 1389, 1362 },
		{ 1564, 1555 },
		{ 1565, 1556 },
		{ 1388, 1362 },
		{ 1382, 1357 },
		{ 1566, 1557 },
		{ 1567, 1558 },
		{ 1392, 1367 },
		{ 1571, 1565 },
		{ 1393, 1368 },
		{ 1573, 1567 },
		{ 159, 4 },
		{ 1576, 1571 },
		{ 1559, 1549 },
		{ 1577, 1573 },
		{ 2485, 2456 },
		{ 1381, 1357 },
		{ 2326, 2298 },
		{ 1579, 1576 },
		{ 1580, 1577 },
		{ 1581, 1579 },
		{ 1582, 1580 },
		{ 1377, 1581 },
		{ 1394, 1369 },
		{ 1395, 1371 },
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
		{ 1396, 1372 },
		{ 1399, 1376 },
		{ 1400, 1379 },
		{ 1401, 1380 },
		{ 0, 2456 },
		{ 1561, 1549 },
		{ 1402, 1381 },
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
		{ 1403, 1382 },
		{ 81, 4 },
		{ 1404, 1383 },
		{ 1405, 1384 },
		{ 85, 4 },
		{ 1406, 1385 },
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
		{ 3001, 3000 },
		{ 1407, 1386 },
		{ 1409, 1387 },
		{ 3000, 3000 },
		{ 1410, 1388 },
		{ 1408, 1386 },
		{ 1411, 1389 },
		{ 1414, 1392 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 1415, 1393 },
		{ 3000, 3000 },
		{ 1416, 1394 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 1417, 1395 },
		{ 1418, 1396 },
		{ 1421, 1399 },
		{ 1422, 1400 },
		{ 1423, 1401 },
		{ 1424, 1402 },
		{ 1425, 1403 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 1426, 1404 },
		{ 1427, 1405 },
		{ 1428, 1406 },
		{ 1429, 1407 },
		{ 1430, 1408 },
		{ 1431, 1409 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 3000, 3000 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1432, 1410 },
		{ 1433, 1411 },
		{ 154, 152 },
		{ 104, 89 },
		{ 105, 90 },
		{ 106, 91 },
		{ 107, 92 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 108, 93 },
		{ 109, 94 },
		{ 110, 95 },
		{ 111, 96 },
		{ 1377, 1583 },
		{ 112, 97 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 1377, 1583 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 114, 99 },
		{ 120, 104 },
		{ 121, 105 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 109 },
		{ 125, 110 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 126, 111 },
		{ 127, 112 },
		{ 129, 114 },
		{ 134, 120 },
		{ 2230, 2479 },
		{ 135, 121 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 2230, 2479 },
		{ 1378, 1582 },
		{ 0, 1582 },
		{ 136, 122 },
		{ 137, 123 },
		{ 138, 124 },
		{ 139, 125 },
		{ 140, 127 },
		{ 141, 129 },
		{ 142, 134 },
		{ 143, 135 },
		{ 144, 136 },
		{ 2237, 2216 },
		{ 2239, 2217 },
		{ 2242, 2218 },
		{ 2659, 2659 },
		{ 2243, 2219 },
		{ 2240, 2218 },
		{ 2250, 2221 },
		{ 2253, 2222 },
		{ 2241, 2218 },
		{ 2246, 2220 },
		{ 2254, 2223 },
		{ 2255, 2224 },
		{ 1378, 1582 },
		{ 2245, 2220 },
		{ 2244, 2219 },
		{ 2256, 2225 },
		{ 2257, 2226 },
		{ 2258, 2227 },
		{ 2230, 2230 },
		{ 2251, 2231 },
		{ 2238, 2232 },
		{ 2249, 2233 },
		{ 2264, 2237 },
		{ 145, 137 },
		{ 2266, 2239 },
		{ 2267, 2240 },
		{ 2659, 2659 },
		{ 2252, 2231 },
		{ 2247, 2220 },
		{ 2248, 2220 },
		{ 2268, 2241 },
		{ 2269, 2242 },
		{ 2270, 2243 },
		{ 2271, 2244 },
		{ 2272, 2245 },
		{ 2273, 2246 },
		{ 2274, 2247 },
		{ 2275, 2248 },
		{ 2276, 2249 },
		{ 2277, 2250 },
		{ 2278, 2251 },
		{ 0, 1582 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2279, 2252 },
		{ 2280, 2253 },
		{ 2281, 2254 },
		{ 2282, 2255 },
		{ 2285, 2257 },
		{ 2286, 2258 },
		{ 2293, 2264 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 67, 7 },
		{ 2295, 2266 },
		{ 2296, 2267 },
		{ 2297, 2268 },
		{ 2659, 2659 },
		{ 1583, 1582 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2298, 2269 },
		{ 2299, 2270 },
		{ 2300, 2271 },
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
		{ 2301, 2272 },
		{ 2302, 2273 },
		{ 2303, 2274 },
		{ 2304, 2275 },
		{ 2305, 2276 },
		{ 2306, 2277 },
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
		{ 2309, 2280 },
		{ 2310, 2281 },
		{ 2311, 2282 },
		{ 2312, 2283 },
		{ 1204, 7 },
		{ 2313, 2284 },
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
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 2314, 2285 },
		{ 2315, 2286 },
		{ 2321, 2293 },
		{ 147, 140 },
		{ 2324, 2296 },
		{ 2325, 2297 },
		{ 148, 141 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 2329, 2300 },
		{ 2327, 2299 },
		{ 2330, 2301 },
		{ 149, 142 },
		{ 0, 1441 },
		{ 2328, 2299 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1441 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 2332, 2303 },
		{ 2333, 2304 },
		{ 2335, 2306 },
		{ 2339, 2309 },
		{ 2340, 2310 },
		{ 2341, 2311 },
		{ 2342, 2312 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 2343, 2313 },
		{ 2344, 2314 },
		{ 2345, 2315 },
		{ 2346, 2321 },
		{ 0, 1854 },
		{ 2349, 2324 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 1854 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 150, 144 },
		{ 2354, 2328 },
		{ 2355, 2329 },
		{ 2356, 2330 },
		{ 2359, 2333 },
		{ 2361, 2335 },
		{ 2365, 2339 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 2366, 2340 },
		{ 2367, 2341 },
		{ 2368, 2342 },
		{ 2369, 2343 },
		{ 0, 2048 },
		{ 151, 147 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 0, 2048 },
		{ 2283, 2256 },
		{ 2371, 2345 },
		{ 152, 148 },
		{ 2376, 2349 },
		{ 2379, 2354 },
		{ 2380, 2355 },
		{ 2382, 2356 },
		{ 2385, 2359 },
		{ 2387, 2361 },
		{ 2284, 2256 },
		{ 2381, 2356 },
		{ 2391, 2365 },
		{ 2392, 2366 },
		{ 2393, 2367 },
		{ 153, 150 },
		{ 2397, 2371 },
		{ 2402, 2376 },
		{ 2407, 2380 },
		{ 2408, 2381 },
		{ 2409, 2382 },
		{ 2412, 2385 },
		{ 2414, 2387 },
		{ 2418, 2391 },
		{ 2419, 2392 },
		{ 2420, 2393 },
		{ 2425, 2397 },
		{ 2430, 2402 },
		{ 2433, 2407 },
		{ 2434, 2408 },
		{ 89, 73 },
		{ 2438, 2412 },
		{ 155, 153 },
		{ 2444, 2418 },
		{ 2445, 2419 },
		{ 156, 155 },
		{ 2451, 2425 },
		{ 2456, 2430 },
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
		{ 2316, 2287 },
		{ 2318, 2290 },
		{ 2316, 2287 },
		{ 2318, 2290 },
		{ 85, 157 },
		{ 2825, 49 },
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
		{ 0, 2826 },
		{ 2555, 2530 },
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
		{ 2288, 2260 },
		{ 130, 115 },
		{ 2288, 2260 },
		{ 130, 115 },
		{ 1204, 1204 },
		{ 2566, 2541 },
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
		{ 0, 2451 },
		{ 0, 2451 },
		{ 1217, 1213 },
		{ 1730, 1729 },
		{ 1217, 1213 },
		{ 86, 49 },
		{ 2154, 2151 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 1772, 1771 },
		{ 2695, 2680 },
		{ 2712, 2698 },
		{ 1256, 1255 },
		{ 1682, 1681 },
		{ 2531, 2502 },
		{ 0, 2451 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 1727, 1726 },
		{ 1253, 1252 },
		{ 2972, 2971 },
		{ 1637, 1636 },
		{ 2992, 2992 },
		{ 3051, 3050 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 1316, 1315 },
		{ 2006, 1982 },
		{ 182, 173 },
		{ 192, 173 },
		{ 184, 173 },
		{ 1611, 1610 },
		{ 179, 173 },
		{ 1316, 1315 },
		{ 183, 173 },
		{ 2817, 2816 },
		{ 181, 173 },
		{ 2194, 2193 },
		{ 1611, 1610 },
		{ 2532, 2503 },
		{ 190, 173 },
		{ 189, 173 },
		{ 180, 173 },
		{ 188, 173 },
		{ 2479, 2451 },
		{ 185, 173 },
		{ 187, 173 },
		{ 178, 173 },
		{ 2005, 1982 },
		{ 2199, 2198 },
		{ 186, 173 },
		{ 191, 173 },
		{ 2259, 2228 },
		{ 1685, 1684 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 2228, 2228 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 1210, 1207 },
		{ 2260, 2228 },
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
		{ 1805, 1781 },
		{ 443, 400 },
		{ 448, 400 },
		{ 445, 400 },
		{ 444, 400 },
		{ 447, 400 },
		{ 442, 400 },
		{ 3022, 65 },
		{ 441, 400 },
		{ 1658, 1657 },
		{ 67, 65 },
		{ 1211, 1207 },
		{ 446, 400 },
		{ 1804, 1781 },
		{ 449, 400 },
		{ 2858, 2857 },
		{ 2478, 2450 },
		{ 1991, 1969 },
		{ 2911, 2910 },
		{ 2599, 2574 },
		{ 440, 400 },
		{ 2260, 2228 },
		{ 100, 83 },
		{ 3017, 3016 },
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
		{ 1269, 1268 },
		{ 2952, 2951 },
		{ 2021, 2000 },
		{ 2975, 2974 },
		{ 2983, 2982 },
		{ 2050, 2032 },
		{ 2064, 2047 },
		{ 1830, 1811 },
		{ 1852, 1833 },
		{ 1211, 1207 },
		{ 1743, 1742 },
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
		{ 2772, 2771 },
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
		{ 2464, 2464 },
		{ 2464, 2464 },
		{ 2394, 2394 },
		{ 2394, 2394 },
		{ 2436, 2410 },
		{ 3034, 3030 },
		{ 2812, 2811 },
		{ 3054, 3053 },
		{ 3020, 65 },
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
		{ 3018, 65 },
		{ 3070, 3067 },
		{ 2464, 2464 },
		{ 3076, 3073 },
		{ 2394, 2394 },
		{ 2569, 2544 },
		{ 2081, 2068 },
		{ 2094, 2079 },
		{ 2095, 2080 },
		{ 3008, 63 },
		{ 1213, 1210 },
		{ 118, 102 },
		{ 67, 63 },
		{ 2606, 2581 },
		{ 1345, 1344 },
		{ 2617, 2593 },
		{ 2621, 2597 },
		{ 2630, 2607 },
		{ 2378, 2351 },
		{ 2636, 2613 },
		{ 2648, 2629 },
		{ 1216, 1212 },
		{ 3021, 65 },
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
		{ 118, 102 },
		{ 115, 100 },
		{ 2650, 2631 },
		{ 2307, 2307 },
		{ 2307, 2307 },
		{ 1206, 9 },
		{ 2664, 2645 },
		{ 2441, 2441 },
		{ 2441, 2441 },
		{ 67, 9 },
		{ 2665, 2646 },
		{ 2287, 2259 },
		{ 2155, 2152 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 2261, 2261 },
		{ 2170, 2169 },
		{ 2675, 2656 },
		{ 3007, 63 },
		{ 2307, 2307 },
		{ 1236, 1235 },
		{ 1206, 9 },
		{ 3006, 63 },
		{ 2441, 2441 },
		{ 2876, 2876 },
		{ 2876, 2876 },
		{ 115, 100 },
		{ 2290, 2261 },
		{ 2923, 2923 },
		{ 2923, 2923 },
		{ 2680, 2663 },
		{ 2494, 2464 },
		{ 2493, 2464 },
		{ 2422, 2394 },
		{ 2421, 2394 },
		{ 1208, 9 },
		{ 2287, 2259 },
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
		{ 2876, 2876 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2689, 2673 },
		{ 2923, 2923 },
		{ 2452, 2452 },
		{ 2452, 2452 },
		{ 2326, 2326 },
		{ 2326, 2326 },
		{ 2471, 2471 },
		{ 2471, 2471 },
		{ 1659, 1658 },
		{ 2290, 2261 },
		{ 2350, 2350 },
		{ 2350, 2350 },
		{ 2485, 2485 },
		{ 2485, 2485 },
		{ 2504, 2504 },
		{ 2504, 2504 },
		{ 2556, 2556 },
		{ 2556, 2556 },
		{ 2649, 2649 },
		{ 2649, 2649 },
		{ 2698, 2683 },
		{ 2443, 2443 },
		{ 3004, 63 },
		{ 2196, 2195 },
		{ 3005, 63 },
		{ 2452, 2452 },
		{ 2715, 2701 },
		{ 2326, 2326 },
		{ 2728, 2722 },
		{ 2471, 2471 },
		{ 2730, 2724 },
		{ 2370, 2370 },
		{ 2370, 2370 },
		{ 2350, 2350 },
		{ 2736, 2731 },
		{ 2485, 2485 },
		{ 2742, 2741 },
		{ 2504, 2504 },
		{ 1661, 1660 },
		{ 2556, 2556 },
		{ 2455, 2429 },
		{ 2649, 2649 },
		{ 2336, 2307 },
		{ 2808, 2808 },
		{ 2808, 2808 },
		{ 2836, 2836 },
		{ 2836, 2836 },
		{ 2775, 2774 },
		{ 2446, 2446 },
		{ 2446, 2446 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2784, 2783 },
		{ 2337, 2307 },
		{ 2370, 2370 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2467, 2441 },
		{ 2797, 2796 },
		{ 2470, 2470 },
		{ 2470, 2470 },
		{ 2440, 2440 },
		{ 2440, 2440 },
		{ 2201, 2201 },
		{ 2201, 2201 },
		{ 1833, 1814 },
		{ 2808, 2808 },
		{ 2457, 2431 },
		{ 2836, 2836 },
		{ 2265, 2265 },
		{ 2265, 2265 },
		{ 2446, 2446 },
		{ 2208, 2207 },
		{ 2331, 2331 },
		{ 2890, 2890 },
		{ 2890, 2890 },
		{ 2931, 2931 },
		{ 2931, 2931 },
		{ 2413, 2413 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 2877, 2876 },
		{ 2470, 2470 },
		{ 2820, 2819 },
		{ 2440, 2440 },
		{ 2924, 2923 },
		{ 2201, 2201 },
		{ 1839, 1820 },
		{ 2562, 2562 },
		{ 2562, 2562 },
		{ 1272, 1271 },
		{ 1859, 1840 },
		{ 2265, 2265 },
		{ 1311, 1310 },
		{ 2609, 2609 },
		{ 2609, 2609 },
		{ 2480, 2452 },
		{ 2890, 2890 },
		{ 1886, 1870 },
		{ 2931, 2931 },
		{ 2460, 2460 },
		{ 2460, 2460 },
		{ 2372, 2372 },
		{ 2848, 2847 },
		{ 2481, 2452 },
		{ 2469, 2443 },
		{ 2652, 2652 },
		{ 2652, 2652 },
		{ 2334, 2334 },
		{ 2334, 2334 },
		{ 2351, 2326 },
		{ 2562, 2562 },
		{ 2501, 2471 },
		{ 2837, 2836 },
		{ 2845, 2845 },
		{ 2845, 2845 },
		{ 2377, 2350 },
		{ 2609, 2609 },
		{ 2515, 2485 },
		{ 1888, 1873 },
		{ 2533, 2504 },
		{ 2474, 2446 },
		{ 2581, 2556 },
		{ 2460, 2460 },
		{ 2668, 2649 },
		{ 2875, 2874 },
		{ 2586, 2586 },
		{ 2586, 2586 },
		{ 2403, 2377 },
		{ 2652, 2652 },
		{ 1890, 1875 },
		{ 2334, 2334 },
		{ 2905, 2904 },
		{ 1940, 1938 },
		{ 2914, 2913 },
		{ 2922, 2921 },
		{ 2396, 2370 },
		{ 2845, 2845 },
		{ 2472, 2446 },
		{ 1687, 1686 },
		{ 1709, 1708 },
		{ 2946, 2945 },
		{ 2473, 2446 },
		{ 1722, 1721 },
		{ 2955, 2954 },
		{ 2837, 2836 },
		{ 2966, 2965 },
		{ 1257, 1256 },
		{ 2809, 2808 },
		{ 2586, 2586 },
		{ 2415, 2388 },
		{ 2977, 2976 },
		{ 2417, 2390 },
		{ 2357, 2331 },
		{ 2986, 2985 },
		{ 1606, 1605 },
		{ 1731, 1730 },
		{ 2062, 2045 },
		{ 2439, 2413 },
		{ 2520, 2491 },
		{ 2063, 2046 },
		{ 1334, 1333 },
		{ 2500, 2470 },
		{ 2428, 2400 },
		{ 2466, 2440 },
		{ 3030, 3026 },
		{ 2202, 2201 },
		{ 2539, 2512 },
		{ 2552, 2527 },
		{ 1746, 1745 },
		{ 3056, 3055 },
		{ 2294, 2265 },
		{ 3058, 3058 },
		{ 2559, 2534 },
		{ 2432, 2404 },
		{ 3083, 3081 },
		{ 2891, 2890 },
		{ 1812, 1787 },
		{ 2932, 2931 },
		{ 2705, 2690 },
		{ 1811, 1787 },
		{ 2398, 2372 },
		{ 2323, 2323 },
		{ 2323, 2323 },
		{ 2001, 1976 },
		{ 1304, 1303 },
		{ 1901, 1888 },
		{ 2000, 1976 },
		{ 1920, 1910 },
		{ 1928, 1920 },
		{ 2587, 2562 },
		{ 1305, 1304 },
		{ 2486, 2457 },
		{ 2746, 2745 },
		{ 1265, 1264 },
		{ 3058, 3058 },
		{ 2632, 2609 },
		{ 2767, 2766 },
		{ 1703, 1702 },
		{ 1704, 1703 },
		{ 2779, 2778 },
		{ 1240, 1239 },
		{ 2489, 2460 },
		{ 2502, 2472 },
		{ 2506, 2476 },
		{ 2323, 2323 },
		{ 1713, 1712 },
		{ 2514, 2483 },
		{ 2671, 2652 },
		{ 1569, 1563 },
		{ 2360, 2334 },
		{ 2022, 2001 },
		{ 2041, 2020 },
		{ 2043, 2022 },
		{ 2831, 2829 },
		{ 2046, 2025 },
		{ 2846, 2845 },
		{ 1225, 1224 },
		{ 2540, 2514 },
		{ 2852, 2851 },
		{ 1599, 1598 },
		{ 1600, 1599 },
		{ 1739, 1738 },
		{ 1340, 1339 },
		{ 1280, 1279 },
		{ 2573, 2548 },
		{ 2611, 2586 },
		{ 1762, 1761 },
		{ 2584, 2559 },
		{ 1763, 1762 },
		{ 1768, 1767 },
		{ 1627, 1626 },
		{ 1628, 1627 },
		{ 1634, 1633 },
		{ 2965, 2964 },
		{ 1635, 1634 },
		{ 2449, 2423 },
		{ 1287, 1286 },
		{ 2453, 2427 },
		{ 2637, 2614 },
		{ 2638, 2615 },
		{ 2640, 2618 },
		{ 2641, 2621 },
		{ 1831, 1812 },
		{ 1653, 1652 },
		{ 1838, 1819 },
		{ 2458, 2432 },
		{ 2667, 2648 },
		{ 2211, 2210 },
		{ 1654, 1653 },
		{ 3046, 3045 },
		{ 3047, 3046 },
		{ 1850, 1831 },
		{ 2676, 2657 },
		{ 1288, 1287 },
		{ 1290, 1289 },
		{ 1563, 1554 },
		{ 1677, 1676 },
		{ 1678, 1677 },
		{ 1336, 1335 },
		{ 3061, 3058 },
		{ 2769, 2768 },
		{ 2568, 2543 },
		{ 1632, 1631 },
		{ 1249, 1248 },
		{ 3060, 3058 },
		{ 1342, 1341 },
		{ 3059, 3058 },
		{ 2795, 2794 },
		{ 2583, 2558 },
		{ 2806, 2805 },
		{ 2465, 2439 },
		{ 2411, 2384 },
		{ 2588, 2563 },
		{ 2363, 2337 },
		{ 1689, 1688 },
		{ 2148, 2141 },
		{ 2151, 2146 },
		{ 2618, 2594 },
		{ 2835, 2833 },
		{ 1282, 1281 },
		{ 2348, 2323 },
		{ 2622, 2598 },
		{ 2475, 2447 },
		{ 1639, 1638 },
		{ 2167, 2164 },
		{ 1935, 1930 },
		{ 1227, 1226 },
		{ 2347, 2347 },
		{ 2347, 2347 },
		{ 2903, 2902 },
		{ 1255, 1254 },
		{ 1823, 1804 },
		{ 2496, 2466 },
		{ 2654, 2635 },
		{ 2658, 2639 },
		{ 2497, 2467 },
		{ 2944, 2943 },
		{ 2499, 2469 },
		{ 2198, 2197 },
		{ 1715, 1714 },
		{ 1318, 1317 },
		{ 2204, 2203 },
		{ 1724, 1723 },
		{ 2210, 2209 },
		{ 2679, 2662 },
		{ 2012, 1991 },
		{ 2686, 2670 },
		{ 1834, 1815 },
		{ 2529, 2500 },
		{ 1332, 1331 },
		{ 2347, 2347 },
		{ 2700, 2685 },
		{ 2033, 2012 },
		{ 1729, 1728 },
		{ 1842, 1823 },
		{ 2716, 2702 },
		{ 2717, 2704 },
		{ 3037, 3034 },
		{ 1613, 1612 },
		{ 2729, 2723 },
		{ 2547, 2522 },
		{ 1663, 1662 },
		{ 2737, 2735 },
		{ 3058, 3057 },
		{ 2740, 2738 },
		{ 3066, 3063 },
		{ 1234, 1233 },
		{ 3074, 3071 },
		{ 1869, 1852 },
		{ 2560, 2535 },
		{ 3086, 3085 },
		{ 2308, 2308 },
		{ 2308, 2308 },
		{ 2726, 2726 },
		{ 2726, 2726 },
		{ 2395, 2395 },
		{ 2395, 2395 },
		{ 2612, 2587 },
		{ 1769, 1768 },
		{ 2410, 2383 },
		{ 1344, 1343 },
		{ 2722, 2714 },
		{ 2535, 2506 },
		{ 2512, 2481 },
		{ 2631, 2608 },
		{ 1873, 1858 },
		{ 1330, 1329 },
		{ 2518, 2489 },
		{ 2032, 2011 },
		{ 2373, 2347 },
		{ 2598, 2573 },
		{ 2557, 2532 },
		{ 2687, 2671 },
		{ 2847, 2846 },
		{ 2308, 2308 },
		{ 2645, 2625 },
		{ 2726, 2726 },
		{ 2646, 2627 },
		{ 2395, 2395 },
		{ 2527, 2498 },
		{ 1708, 1707 },
		{ 2651, 2632 },
		{ 1801, 1778 },
		{ 2805, 2804 },
		{ 1707, 1706 },
		{ 2161, 2158 },
		{ 2639, 2617 },
		{ 2164, 2162 },
		{ 1647, 1646 },
		{ 2822, 2821 },
		{ 1937, 1934 },
		{ 1335, 1334 },
		{ 1800, 1778 },
		{ 2517, 2488 },
		{ 2437, 2411 },
		{ 1943, 1942 },
		{ 2525, 2496 },
		{ 1816, 1793 },
		{ 1454, 1434 },
		{ 1274, 1273 },
		{ 2374, 2347 },
		{ 2850, 2849 },
		{ 1397, 1373 },
		{ 2857, 2856 },
		{ 1621, 1620 },
		{ 2383, 2357 },
		{ 2674, 2655 },
		{ 2007, 1984 },
		{ 2011, 1990 },
		{ 1662, 1661 },
		{ 2907, 2906 },
		{ 1835, 1816 },
		{ 2554, 2529 },
		{ 2916, 2915 },
		{ 1837, 1818 },
		{ 2026, 2005 },
		{ 2028, 2007 },
		{ 2030, 2009 },
		{ 1329, 1328 },
		{ 2948, 2947 },
		{ 1671, 1670 },
		{ 2704, 2689 },
		{ 2957, 2956 },
		{ 1238, 1237 },
		{ 2706, 2691 },
		{ 2570, 2545 },
		{ 1419, 1397 },
		{ 1633, 1632 },
		{ 2979, 2978 },
		{ 1248, 1247 },
		{ 2061, 2044 },
		{ 2988, 2987 },
		{ 2723, 2715 },
		{ 1868, 1851 },
		{ 1748, 1747 },
		{ 2594, 2569 },
		{ 1756, 1755 },
		{ 2735, 2730 },
		{ 1298, 1297 },
		{ 2076, 2060 },
		{ 2607, 2582 },
		{ 3035, 3031 },
		{ 1883, 1867 },
		{ 2744, 2743 },
		{ 1688, 1687 },
		{ 1636, 1635 },
		{ 2126, 2108 },
		{ 2338, 2308 },
		{ 2495, 2465 },
		{ 2731, 2726 },
		{ 3057, 3056 },
		{ 2423, 2395 },
		{ 2127, 2109 },
		{ 1697, 1696 },
		{ 3063, 3060 },
		{ 2777, 2776 },
		{ 1593, 1592 },
		{ 2153, 2150 },
		{ 2634, 2611 },
		{ 1347, 1346 },
		{ 3085, 3083 },
		{ 2799, 2798 },
		{ 2898, 2898 },
		{ 2898, 2898 },
		{ 2790, 2790 },
		{ 2790, 2790 },
		{ 2939, 2939 },
		{ 2939, 2939 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 2459, 2459 },
		{ 2459, 2459 },
		{ 2139, 2126 },
		{ 2140, 2127 },
		{ 2832, 2830 },
		{ 1239, 1238 },
		{ 2534, 2505 },
		{ 2150, 2145 },
		{ 2536, 2507 },
		{ 2538, 2511 },
		{ 1832, 1813 },
		{ 2851, 2850 },
		{ 1944, 1943 },
		{ 2454, 2428 },
		{ 1734, 1733 },
		{ 2898, 2898 },
		{ 2859, 2858 },
		{ 2790, 2790 },
		{ 2868, 2867 },
		{ 2939, 2939 },
		{ 1275, 1274 },
		{ 2435, 2435 },
		{ 1297, 1296 },
		{ 2459, 2459 },
		{ 2885, 2884 },
		{ 2886, 2885 },
		{ 2888, 2887 },
		{ 2163, 2161 },
		{ 2690, 2674 },
		{ 2901, 2900 },
		{ 1744, 1743 },
		{ 1646, 1645 },
		{ 1696, 1695 },
		{ 2908, 2907 },
		{ 2009, 1987 },
		{ 2912, 2911 },
		{ 2010, 1989 },
		{ 1840, 1821 },
		{ 2917, 2916 },
		{ 1749, 1748 },
		{ 1755, 1754 },
		{ 2929, 2928 },
		{ 2200, 2199 },
		{ 1604, 1603 },
		{ 2942, 2941 },
		{ 2719, 2706 },
		{ 2023, 2002 },
		{ 1327, 1326 },
		{ 2949, 2948 },
		{ 2291, 2262 },
		{ 2953, 2952 },
		{ 2595, 2570 },
		{ 2597, 2572 },
		{ 2958, 2957 },
		{ 2964, 2963 },
		{ 1474, 1455 },
		{ 113, 98 },
		{ 1870, 1853 },
		{ 1620, 1619 },
		{ 2610, 2585 },
		{ 1270, 1269 },
		{ 2980, 2979 },
		{ 1875, 1860 },
		{ 2984, 2983 },
		{ 2745, 2744 },
		{ 2045, 2024 },
		{ 2989, 2988 },
		{ 1260, 1259 },
		{ 1885, 1869 },
		{ 1398, 1375 },
		{ 3002, 2999 },
		{ 1670, 1669 },
		{ 2773, 2772 },
		{ 1348, 1347 },
		{ 1309, 1308 },
		{ 3028, 3024 },
		{ 2778, 2777 },
		{ 3031, 3027 },
		{ 1909, 1897 },
		{ 2503, 2473 },
		{ 2068, 2051 },
		{ 3040, 3037 },
		{ 2793, 2792 },
		{ 2362, 2336 },
		{ 1818, 1796 },
		{ 2364, 2338 },
		{ 2899, 2898 },
		{ 2800, 2799 },
		{ 2791, 2790 },
		{ 1922, 1913 },
		{ 2940, 2939 },
		{ 1822, 1803 },
		{ 2461, 2435 },
		{ 1934, 1929 },
		{ 2488, 2459 },
		{ 2519, 2490 },
		{ 2106, 2092 },
		{ 2818, 2817 },
		{ 1434, 1412 },
		{ 1592, 1591 },
		{ 2823, 2822 },
		{ 2128, 2110 },
		{ 2661, 2642 },
		{ 2815, 2815 },
		{ 2815, 2815 },
		{ 2770, 2770 },
		{ 2770, 2770 },
		{ 2492, 2492 },
		{ 2492, 2492 },
		{ 2981, 2981 },
		{ 2981, 2981 },
		{ 1741, 1741 },
		{ 1741, 1741 },
		{ 1267, 1267 },
		{ 1267, 1267 },
		{ 2684, 2684 },
		{ 2684, 2684 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2950, 2950 },
		{ 2950, 2950 },
		{ 2909, 2909 },
		{ 2909, 2909 },
		{ 1939, 1937 },
		{ 1594, 1593 },
		{ 2401, 2375 },
		{ 2815, 2815 },
		{ 1657, 1656 },
		{ 2770, 2770 },
		{ 3038, 3035 },
		{ 2492, 2492 },
		{ 1856, 1837 },
		{ 2981, 2981 },
		{ 1299, 1298 },
		{ 1741, 1741 },
		{ 1672, 1671 },
		{ 1267, 1267 },
		{ 1711, 1710 },
		{ 2684, 2684 },
		{ 2463, 2437 },
		{ 2358, 2358 },
		{ 2206, 2205 },
		{ 2950, 2950 },
		{ 1757, 1756 },
		{ 2909, 2909 },
		{ 2077, 2061 },
		{ 2156, 2153 },
		{ 2589, 2564 },
		{ 1473, 1454 },
		{ 1648, 1647 },
		{ 3073, 3070 },
		{ 1698, 1697 },
		{ 1622, 1621 },
		{ 1884, 1868 },
		{ 2172, 2171 },
		{ 2049, 2030 },
		{ 2970, 2970 },
		{ 2970, 2970 },
		{ 2934, 2934 },
		{ 2934, 2934 },
		{ 1725, 1725 },
		{ 1725, 1725 },
		{ 2893, 2893 },
		{ 2893, 2893 },
		{ 2785, 2785 },
		{ 2785, 2785 },
		{ 1251, 1251 },
		{ 1251, 1251 },
		{ 2927, 2927 },
		{ 2927, 2927 },
		{ 1262, 1262 },
		{ 1262, 1262 },
		{ 1736, 1736 },
		{ 1736, 1736 },
		{ 1843, 1824 },
		{ 2125, 2107 },
		{ 1684, 1683 },
		{ 1302, 1301 },
		{ 1854, 1835 },
		{ 2970, 2970 },
		{ 1538, 1512 },
		{ 2934, 2934 },
		{ 2567, 2542 },
		{ 1725, 1725 },
		{ 1597, 1596 },
		{ 2893, 2893 },
		{ 1819, 1799 },
		{ 2785, 2785 },
		{ 2644, 2624 },
		{ 1251, 1251 },
		{ 2721, 2713 },
		{ 2927, 2927 },
		{ 1314, 1313 },
		{ 1262, 1262 },
		{ 2571, 2546 },
		{ 1736, 1736 },
		{ 2816, 2815 },
		{ 1625, 1624 },
		{ 2771, 2770 },
		{ 2462, 2436 },
		{ 2521, 2492 },
		{ 2582, 2557 },
		{ 2982, 2981 },
		{ 1264, 1263 },
		{ 1742, 1741 },
		{ 2047, 2026 },
		{ 1268, 1267 },
		{ 2048, 2028 },
		{ 2699, 2684 },
		{ 1441, 1419 },
		{ 2384, 2358 },
		{ 2160, 2157 },
		{ 2951, 2950 },
		{ 2666, 2647 },
		{ 2910, 2909 },
		{ 2843, 2842 },
		{ 3044, 3043 },
		{ 1760, 1759 },
		{ 1701, 1700 },
		{ 1651, 1650 },
		{ 3053, 3052 },
		{ 1675, 1674 },
		{ 1232, 1231 },
		{ 2477, 2449 },
		{ 1770, 1769 },
		{ 2067, 2050 },
		{ 1738, 1737 },
		{ 2195, 2194 },
		{ 1900, 1887 },
		{ 2688, 2672 },
		{ 1609, 1608 },
		{ 2974, 2973 },
		{ 3079, 3076 },
		{ 2020, 1999 },
		{ 1285, 1284 },
		{ 1912, 1902 },
		{ 2578, 2578 },
		{ 2578, 2578 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 2442, 2442 },
		{ 2442, 2442 },
		{ 2576, 2576 },
		{ 2576, 2576 },
		{ 2839, 2838 },
		{ 2866, 2865 },
		{ 1984, 2175 },
		{ 1373, 1364 },
		{ 2971, 2970 },
		{ 1793, 1949 },
		{ 2935, 2934 },
		{ 2585, 2560 },
		{ 1726, 1725 },
		{ 2798, 2797 },
		{ 2894, 2893 },
		{ 1455, 1435 },
		{ 2786, 2785 },
		{ 2400, 2374 },
		{ 1252, 1251 },
		{ 2578, 2578 },
		{ 2928, 2927 },
		{ 1228, 1228 },
		{ 1263, 1262 },
		{ 2442, 2442 },
		{ 1737, 1736 },
		{ 2576, 2576 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 2289, 2289 },
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
		{ 2861, 2861 },
		{ 2861, 2861 },
		{ 1283, 1283 },
		{ 1283, 1283 },
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
		{ 1607, 1606 },
		{ 2947, 2946 },
		{ 1802, 1780 },
		{ 1312, 1311 },
		{ 2683, 2666 },
		{ 1942, 1940 },
		{ 1610, 1609 },
		{ 2158, 2155 },
		{ 1710, 1709 },
		{ 2861, 2861 },
		{ 2956, 2955 },
		{ 1283, 1283 },
		{ 0, 1208 },
		{ 0, 84 },
		{ 1346, 1345 },
		{ 2821, 2820 },
		{ 2691, 2675 },
		{ 2603, 2578 },
		{ 1820, 1800 },
		{ 1229, 1228 },
		{ 2530, 2501 },
		{ 2468, 2442 },
		{ 1747, 1746 },
		{ 2601, 2576 },
		{ 2319, 2319 },
		{ 2319, 2319 },
		{ 2319, 2319 },
		{ 2319, 2319 },
		{ 2319, 2319 },
		{ 2319, 2319 },
		{ 2319, 2319 },
		{ 2319, 2319 },
		{ 2319, 2319 },
		{ 2319, 2319 },
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
		{ 2994, 2994 },
		{ 2994, 2994 },
		{ 2994, 2994 },
		{ 2994, 2994 },
		{ 2994, 2994 },
		{ 2994, 2994 },
		{ 2994, 2994 },
		{ 2994, 2994 },
		{ 2994, 2994 },
		{ 2994, 2994 },
		{ 0, 2229 },
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
		{ 2565, 2565 },
		{ 2565, 2565 },
		{ 1803, 1780 },
		{ 2608, 2583 },
		{ 2169, 2167 },
		{ 1988, 1966 },
		{ 1874, 1859 },
		{ 2978, 2977 },
		{ 2614, 2589 },
		{ 2862, 2861 },
		{ 2713, 2699 },
		{ 1284, 1283 },
		{ 2842, 2841 },
		{ 2714, 2700 },
		{ 1712, 1711 },
		{ 1824, 1805 },
		{ 2987, 2986 },
		{ 1273, 1272 },
		{ 2849, 2848 },
		{ 2375, 2348 },
		{ 2624, 2600 },
		{ 0, 2229 },
		{ 3078, 3078 },
		{ 2565, 2565 },
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
		{ 1315, 1314 },
		{ 3078, 3078 },
		{ 2880, 2880 },
		{ 2880, 2880 },
		{ 2801, 2801 },
		{ 2801, 2801 },
		{ 2549, 2549 },
		{ 2549, 2549 },
		{ 2541, 2515 },
		{ 2544, 2519 },
		{ 2545, 2520 },
		{ 2546, 2521 },
		{ 1308, 1307 },
		{ 2867, 2866 },
		{ 2078, 2062 },
		{ 2429, 2401 },
		{ 1374, 1353 },
		{ 2092, 2076 },
		{ 2884, 2883 },
		{ 2642, 2622 },
		{ 2743, 2742 },
		{ 2887, 2886 },
		{ 1237, 1236 },
		{ 1897, 1883 },
		{ 2564, 2539 },
		{ 2880, 2880 },
		{ 2647, 2628 },
		{ 2801, 2801 },
		{ 1989, 1966 },
		{ 2549, 2549 },
		{ 1326, 1325 },
		{ 2108, 2094 },
		{ 2109, 2095 },
		{ 2906, 2905 },
		{ 2388, 2362 },
		{ 2655, 2636 },
		{ 2390, 2364 },
		{ 2776, 2775 },
		{ 3066, 3066 },
		{ 1310, 1309 },
		{ 2663, 2644 },
		{ 2915, 2914 },
		{ 1767, 1766 },
		{ 1343, 1342 },
		{ 1603, 1602 },
		{ 1733, 1732 },
		{ 2590, 2565 },
		{ 1259, 1258 },
		{ 1605, 1604 },
		{ 2789, 2788 },
		{ 2925, 2924 },
		{ 2863, 2862 },
		{ 2577, 2552 },
		{ 1258, 1257 },
		{ 2171, 2170 },
		{ 2938, 2937 },
		{ 67, 5 },
		{ 3078, 3075 },
		{ 2897, 2896 },
		{ 2681, 2664 },
		{ 2878, 2877 },
		{ 3066, 3066 },
		{ 2682, 2665 },
		{ 1732, 1731 },
		{ 2869, 2868 },
		{ 1261, 1260 },
		{ 2889, 2888 },
		{ 1325, 1324 },
		{ 1735, 1734 },
		{ 2524, 2495 },
		{ 2883, 2882 },
		{ 3080, 3078 },
		{ 2633, 2610 },
		{ 2678, 2661 },
		{ 2162, 2160 },
		{ 3023, 3018 },
		{ 2829, 2827 },
		{ 1813, 1790 },
		{ 2192, 2191 },
		{ 1951, 1948 },
		{ 2476, 2448 },
		{ 1814, 1790 },
		{ 2448, 2422 },
		{ 1375, 1353 },
		{ 1950, 1948 },
		{ 2523, 2494 },
		{ 2892, 2891 },
		{ 1435, 1413 },
		{ 2933, 2932 },
		{ 1795, 1775 },
		{ 2643, 2623 },
		{ 2881, 2880 },
		{ 2830, 2827 },
		{ 2802, 2801 },
		{ 1794, 1775 },
		{ 2574, 2549 },
		{ 1986, 1963 },
		{ 2804, 2803 },
		{ 1247, 1246 },
		{ 2623, 2599 },
		{ 2841, 2840 },
		{ 1985, 1963 },
		{ 2386, 2360 },
		{ 2548, 2523 },
		{ 1721, 1720 },
		{ 3024, 3018 },
		{ 2426, 2398 },
		{ 2322, 2294 },
		{ 174, 5 },
		{ 2177, 2174 },
		{ 1391, 1365 },
		{ 2389, 2363 },
		{ 175, 5 },
		{ 2580, 2555 },
		{ 2176, 2174 },
		{ 1930, 1922 },
		{ 1681, 1680 },
		{ 1339, 1338 },
		{ 1595, 1594 },
		{ 176, 5 },
		{ 1300, 1299 },
		{ 1317, 1316 },
		{ 1941, 1939 },
		{ 1230, 1229 },
		{ 2483, 2454 },
		{ 2591, 2566 },
		{ 2900, 2899 },
		{ 2110, 2096 },
		{ 2902, 2901 },
		{ 2727, 2721 },
		{ 3069, 3066 },
		{ 1740, 1739 },
		{ 2596, 2571 },
		{ 2262, 2234 },
		{ 1638, 1637 },
		{ 173, 5 },
		{ 2490, 2461 },
		{ 1695, 1694 },
		{ 1324, 1323 },
		{ 2738, 2736 },
		{ 1645, 1644 },
		{ 98, 81 },
		{ 2498, 2468 },
		{ 2141, 2128 },
		{ 2145, 2138 },
		{ 1699, 1698 },
		{ 2926, 2925 },
		{ 1987, 1964 },
		{ 1250, 1249 },
		{ 2768, 2767 },
		{ 2505, 2475 },
		{ 1754, 1753 },
		{ 2416, 2389 },
		{ 2511, 2480 },
		{ 2941, 2940 },
		{ 1849, 1830 },
		{ 2943, 2942 },
		{ 2625, 2601 },
		{ 2627, 2603 },
		{ 1246, 1245 },
		{ 1649, 1648 },
		{ 1853, 1834 },
		{ 2159, 2156 },
		{ 1494, 1473 },
		{ 2635, 2612 },
		{ 2792, 2791 },
		{ 2424, 2396 },
		{ 2794, 2793 },
		{ 1758, 1757 },
		{ 1857, 1838 },
		{ 1495, 1474 },
		{ 1860, 1842 },
		{ 2967, 2966 },
		{ 2526, 2497 },
		{ 1289, 1288 },
		{ 2803, 2802 },
		{ 2528, 2499 },
		{ 1266, 1265 },
		{ 2024, 2003 },
		{ 2807, 2806 },
		{ 2025, 2004 },
		{ 2810, 2809 },
		{ 1612, 1611 },
		{ 2814, 2813 },
		{ 1331, 1330 },
		{ 1872, 1856 },
		{ 1619, 1618 },
		{ 2191, 2190 },
		{ 2537, 2508 },
		{ 2999, 2991 },
		{ 2657, 2638 },
		{ 1714, 1713 },
		{ 1296, 1295 },
		{ 2662, 2643 },
		{ 2042, 2021 },
		{ 1226, 1225 },
		{ 2543, 2518 },
		{ 1669, 1668 },
		{ 2833, 2831 },
		{ 3026, 3021 },
		{ 3027, 3023 },
		{ 1796, 1776 },
		{ 1797, 1777 },
		{ 1798, 1777 },
		{ 2670, 2651 },
		{ 2840, 2839 },
		{ 1723, 1722 },
		{ 2672, 2653 },
		{ 1412, 1390 },
		{ 2844, 2843 },
		{ 3041, 3038 },
		{ 2551, 2526 },
		{ 1623, 1622 },
		{ 2553, 2528 },
		{ 3050, 3049 },
		{ 2207, 2206 },
		{ 2051, 2033 },
		{ 2209, 2208 },
		{ 2059, 2041 },
		{ 1413, 1391 },
		{ 2563, 2538 },
		{ 1728, 1727 },
		{ 3062, 3059 },
		{ 1673, 1672 },
		{ 1591, 1590 },
		{ 2864, 2863 },
		{ 1913, 1903 },
		{ 3071, 3068 },
		{ 2066, 2049 },
		{ 1281, 1280 },
		{ 1254, 1253 },
		{ 2572, 2547 },
		{ 1821, 1801 },
		{ 1929, 1921 },
		{ 2879, 2878 },
		{ 2702, 2687 },
		{ 2882, 2881 },
		{ 2450, 2424 },
		{ 1420, 1398 },
		{ 2031, 2010 },
		{ 2003, 1980 },
		{ 1338, 1337 },
		{ 2561, 2536 },
		{ 2320, 2291 },
		{ 2811, 2810 },
		{ 2930, 2929 },
		{ 3068, 3065 },
		{ 3032, 3028 },
		{ 128, 113 },
		{ 2834, 2832 },
		{ 1799, 1777 },
		{ 2653, 2634 },
		{ 1766, 1765 },
		{ 2860, 2859 },
		{ 1851, 1832 },
		{ 2969, 2968 },
		{ 2044, 2023 },
		{ 1841, 1822 },
		{ 1861, 1843 },
		{ 3003, 3002 },
		{ 2896, 2895 },
		{ 2558, 2533 },
		{ 2813, 2812 },
		{ 1231, 1230 },
		{ 608, 550 },
		{ 2093, 2078 },
		{ 2954, 2953 },
		{ 3064, 3061 },
		{ 2985, 2984 },
		{ 3067, 3064 },
		{ 2508, 2478 },
		{ 2819, 2818 },
		{ 1631, 1630 },
		{ 2788, 2787 },
		{ 609, 550 },
		{ 1745, 1744 },
		{ 2937, 2936 },
		{ 3075, 3072 },
		{ 2685, 2668 },
		{ 2865, 2864 },
		{ 1889, 1874 },
		{ 2913, 2912 },
		{ 3082, 3080 },
		{ 1271, 1270 },
		{ 2774, 2773 },
		{ 1341, 1340 },
		{ 1323, 1320 },
		{ 1793, 1789 },
		{ 2550, 2525 },
		{ 1373, 1366 },
		{ 610, 550 },
		{ 2579, 2554 },
		{ 2600, 2575 },
		{ 2628, 2604 },
		{ 1984, 1978 },
		{ 202, 179 },
		{ 2629, 2606 },
		{ 2404, 2378 },
		{ 200, 179 },
		{ 2838, 2837 },
		{ 201, 179 },
		{ 2575, 2550 },
		{ 1626, 1625 },
		{ 2447, 2421 },
		{ 1676, 1675 },
		{ 2491, 2462 },
		{ 1235, 1234 },
		{ 1333, 1332 },
		{ 2079, 2064 },
		{ 199, 179 },
		{ 2080, 2067 },
		{ 2787, 2786 },
		{ 1815, 1791 },
		{ 1999, 1975 },
		{ 2542, 2517 },
		{ 2701, 2686 },
		{ 1702, 1701 },
		{ 1660, 1659 },
		{ 2593, 2568 },
		{ 2936, 2935 },
		{ 2796, 2795 },
		{ 1554, 1538 },
		{ 2002, 1979 },
		{ 2107, 2093 },
		{ 1771, 1770 },
		{ 2507, 2477 },
		{ 1938, 1935 },
		{ 2945, 2944 },
		{ 1683, 1682 },
		{ 3045, 3044 },
		{ 1313, 1312 },
		{ 2656, 2637 },
		{ 1303, 1302 },
		{ 2604, 2579 },
		{ 3052, 3051 },
		{ 2724, 2716 },
		{ 1686, 1685 },
		{ 3055, 3054 },
		{ 1233, 1232 },
		{ 1608, 1607 },
		{ 2193, 2192 },
		{ 2431, 2403 },
		{ 1652, 1651 },
		{ 2613, 2588 },
		{ 1598, 1597 },
		{ 3065, 3062 },
		{ 2615, 2590 },
		{ 2522, 2493 },
		{ 2968, 2967 },
		{ 2741, 2740 },
		{ 2895, 2894 },
		{ 1902, 1889 },
		{ 3072, 3069 },
		{ 2973, 2972 },
		{ 2673, 2654 },
		{ 2146, 2139 },
		{ 2976, 2975 },
		{ 1286, 1285 },
		{ 1910, 1900 },
		{ 1761, 1760 },
		{ 3081, 3079 },
		{ 2203, 2202 },
		{ 2904, 2903 },
		{ 2152, 2148 },
		{ 2205, 2204 },
		{ 668, 607 },
		{ 734, 677 },
		{ 845, 794 },
		{ 416, 376 },
		{ 1322, 19 },
		{ 1719, 33 },
		{ 1667, 29 },
		{ 67, 19 },
		{ 67, 33 },
		{ 67, 29 },
		{ 2765, 45 },
		{ 1294, 17 },
		{ 1643, 27 },
		{ 67, 45 },
		{ 67, 17 },
		{ 67, 27 },
		{ 400, 361 },
		{ 1244, 13 },
		{ 733, 677 },
		{ 417, 376 },
		{ 67, 13 },
		{ 2872, 55 },
		{ 669, 607 },
		{ 1617, 25 },
		{ 67, 55 },
		{ 406, 367 },
		{ 67, 25 },
		{ 1693, 31 },
		{ 846, 794 },
		{ 2962, 59 },
		{ 67, 31 },
		{ 1223, 11 },
		{ 67, 59 },
		{ 415, 375 },
		{ 67, 11 },
		{ 229, 195 },
		{ 2013, 1992 },
		{ 429, 386 },
		{ 233, 199 },
		{ 450, 401 },
		{ 1825, 1806 },
		{ 453, 404 },
		{ 463, 412 },
		{ 464, 413 },
		{ 477, 426 },
		{ 485, 435 },
		{ 518, 459 },
		{ 528, 467 },
		{ 530, 469 },
		{ 2035, 2014 },
		{ 2036, 2015 },
		{ 531, 470 },
		{ 535, 474 },
		{ 547, 488 },
		{ 551, 492 },
		{ 555, 496 },
		{ 1845, 1826 },
		{ 1846, 1827 },
		{ 565, 506 },
		{ 580, 519 },
		{ 581, 521 },
		{ 586, 526 },
		{ 2058, 2040 },
		{ 234, 200 },
		{ 617, 554 },
		{ 630, 567 },
		{ 633, 570 },
		{ 642, 579 },
		{ 657, 593 },
		{ 658, 594 },
		{ 1866, 1848 },
		{ 667, 606 },
		{ 2074, 2057 },
		{ 240, 206 },
		{ 687, 625 },
		{ 260, 222 },
		{ 735, 678 },
		{ 747, 689 },
		{ 749, 691 },
		{ 751, 693 },
		{ 1881, 1865 },
		{ 756, 698 },
		{ 776, 718 },
		{ 785, 727 },
		{ 789, 731 },
		{ 790, 732 },
		{ 1320, 19 },
		{ 1717, 33 },
		{ 1665, 29 },
		{ 820, 766 },
		{ 837, 786 },
		{ 269, 231 },
		{ 2763, 45 },
		{ 1292, 17 },
		{ 1641, 27 },
		{ 875, 826 },
		{ 883, 834 },
		{ 898, 849 },
		{ 934, 888 },
		{ 1242, 13 },
		{ 938, 892 },
		{ 954, 911 },
		{ 969, 930 },
		{ 2871, 55 },
		{ 994, 959 },
		{ 1615, 25 },
		{ 997, 962 },
		{ 1004, 970 },
		{ 1014, 980 },
		{ 1691, 31 },
		{ 1031, 999 },
		{ 2960, 59 },
		{ 1050, 1017 },
		{ 1221, 11 },
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
		{ 357, 316 },
		{ 380, 340 },
		{ 383, 343 },
		{ 389, 349 },
		{ 67, 41 },
		{ 67, 47 },
		{ 67, 15 },
		{ 67, 23 },
		{ 67, 51 },
		{ 67, 53 },
		{ 422, 380 },
		{ 67, 35 },
		{ 423, 380 },
		{ 421, 380 },
		{ 67, 57 },
		{ 217, 187 },
		{ 2086, 2072 },
		{ 2088, 2073 },
		{ 2147, 2140 },
		{ 215, 187 },
		{ 3016, 3015 },
		{ 455, 406 },
		{ 457, 406 },
		{ 2087, 2072 },
		{ 2089, 2073 },
		{ 800, 742 },
		{ 487, 437 },
		{ 2084, 2070 },
		{ 1893, 1879 },
		{ 424, 380 },
		{ 495, 444 },
		{ 496, 444 },
		{ 458, 406 },
		{ 710, 654 },
		{ 218, 187 },
		{ 216, 187 },
		{ 711, 655 },
		{ 456, 406 },
		{ 497, 444 },
		{ 425, 381 },
		{ 572, 513 },
		{ 399, 360 },
		{ 797, 739 },
		{ 211, 184 },
		{ 285, 244 },
		{ 324, 281 },
		{ 210, 184 },
		{ 508, 453 },
		{ 1092, 1067 },
		{ 1996, 1972 },
		{ 666, 605 },
		{ 2017, 1996 },
		{ 212, 184 },
		{ 431, 388 },
		{ 614, 551 },
		{ 890, 841 },
		{ 510, 453 },
		{ 1995, 1972 },
		{ 613, 551 },
		{ 350, 306 },
		{ 889, 840 },
		{ 509, 453 },
		{ 1101, 1077 },
		{ 224, 190 },
		{ 294, 253 },
		{ 502, 448 },
		{ 612, 551 },
		{ 611, 551 },
		{ 503, 448 },
		{ 1994, 1972 },
		{ 2018, 1997 },
		{ 499, 446 },
		{ 409, 369 },
		{ 408, 369 },
		{ 885, 836 },
		{ 223, 190 },
		{ 221, 188 },
		{ 564, 505 },
		{ 698, 640 },
		{ 219, 188 },
		{ 1100, 1077 },
		{ 801, 743 },
		{ 220, 188 },
		{ 2188, 41 },
		{ 2781, 47 },
		{ 1277, 15 },
		{ 1588, 23 },
		{ 2827, 51 },
		{ 2854, 53 },
		{ 802, 744 },
		{ 1751, 35 },
		{ 500, 446 },
		{ 1828, 1809 },
		{ 2919, 57 },
		{ 263, 225 },
		{ 254, 217 },
		{ 577, 518 },
		{ 277, 239 },
		{ 515, 457 },
		{ 699, 641 },
		{ 578, 518 },
		{ 1224, 1221 },
		{ 2766, 2763 },
		{ 1148, 1138 },
		{ 3014, 3012 },
		{ 1149, 1139 },
		{ 855, 805 },
		{ 701, 643 },
		{ 1808, 1784 },
		{ 3025, 3020 },
		{ 1035, 1003 },
		{ 579, 518 },
		{ 278, 239 },
		{ 516, 457 },
		{ 781, 723 },
		{ 553, 494 },
		{ 1058, 1027 },
		{ 532, 471 },
		{ 731, 675 },
		{ 732, 676 },
		{ 235, 201 },
		{ 3036, 3033 },
		{ 411, 371 },
		{ 945, 901 },
		{ 949, 906 },
		{ 681, 619 },
		{ 806, 748 },
		{ 1125, 1109 },
		{ 597, 536 },
		{ 2427, 2399 },
		{ 1131, 1115 },
		{ 473, 422 },
		{ 1279, 1277 },
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
		{ 1827, 1808 },
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
		{ 2015, 1994 },
		{ 292, 251 },
		{ 853, 803 },
		{ 545, 486 },
		{ 3015, 3014 },
		{ 860, 810 },
		{ 870, 821 },
		{ 340, 296 },
		{ 684, 622 },
		{ 343, 299 },
		{ 407, 368 },
		{ 478, 427 },
		{ 559, 500 },
		{ 3029, 3025 },
		{ 899, 850 },
		{ 905, 856 },
		{ 911, 863 },
		{ 916, 868 },
		{ 921, 875 },
		{ 926, 881 },
		{ 933, 887 },
		{ 702, 644 },
		{ 3039, 3036 },
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
		{ 1480, 1480 },
		{ 1039, 1006 },
		{ 557, 498 },
		{ 384, 344 },
		{ 1055, 1023 },
		{ 640, 577 },
		{ 527, 466 },
		{ 2053, 2035 },
		{ 239, 205 },
		{ 1063, 1032 },
		{ 227, 193 },
		{ 663, 599 },
		{ 568, 509 },
		{ 1584, 1584 },
		{ 358, 317 },
		{ 925, 880 },
		{ 427, 383 },
		{ 574, 515 },
		{ 396, 357 },
		{ 686, 624 },
		{ 536, 475 },
		{ 489, 440 },
		{ 951, 908 },
		{ 1480, 1480 },
		{ 341, 297 },
		{ 804, 746 },
		{ 700, 642 },
		{ 1123, 1107 },
		{ 590, 530 },
		{ 591, 531 },
		{ 542, 484 },
		{ 242, 208 },
		{ 1863, 1845 },
		{ 600, 539 },
		{ 720, 666 },
		{ 989, 951 },
		{ 1584, 1584 },
		{ 1517, 1517 },
		{ 1520, 1520 },
		{ 990, 952 },
		{ 1523, 1523 },
		{ 1526, 1526 },
		{ 1529, 1529 },
		{ 1532, 1532 },
		{ 991, 954 },
		{ 840, 789 },
		{ 841, 790 },
		{ 999, 964 },
		{ 403, 364 },
		{ 605, 547 },
		{ 1015, 981 },
		{ 365, 324 },
		{ 1186, 1180 },
		{ 1550, 1550 },
		{ 319, 276 },
		{ 1019, 985 },
		{ 228, 194 },
		{ 351, 309 },
		{ 880, 831 },
		{ 1559, 1559 },
		{ 1517, 1517 },
		{ 1520, 1520 },
		{ 1390, 1480 },
		{ 1523, 1523 },
		{ 1526, 1526 },
		{ 1529, 1529 },
		{ 1532, 1532 },
		{ 882, 833 },
		{ 624, 561 },
		{ 2104, 2090 },
		{ 2105, 2091 },
		{ 469, 418 },
		{ 1056, 1024 },
		{ 634, 571 },
		{ 894, 845 },
		{ 1390, 1584 },
		{ 1550, 1550 },
		{ 897, 848 },
		{ 638, 575 },
		{ 761, 703 },
		{ 900, 851 },
		{ 1075, 1048 },
		{ 1559, 1559 },
		{ 1078, 1051 },
		{ 1079, 1052 },
		{ 903, 854 },
		{ 767, 709 },
		{ 529, 468 },
		{ 3011, 3007 },
		{ 913, 865 },
		{ 1091, 1066 },
		{ 914, 866 },
		{ 1093, 1068 },
		{ 915, 867 },
		{ 1865, 1847 },
		{ 472, 421 },
		{ 573, 514 },
		{ 214, 186 },
		{ 387, 347 },
		{ 1390, 1517 },
		{ 1390, 1520 },
		{ 320, 277 },
		{ 1390, 1523 },
		{ 1390, 1526 },
		{ 1390, 1529 },
		{ 1390, 1532 },
		{ 483, 432 },
		{ 583, 523 },
		{ 670, 608 },
		{ 585, 525 },
		{ 1880, 1864 },
		{ 349, 305 },
		{ 952, 909 },
		{ 393, 353 },
		{ 799, 741 },
		{ 1390, 1550 },
		{ 958, 918 },
		{ 963, 924 },
		{ 363, 322 },
		{ 967, 928 },
		{ 693, 631 },
		{ 1390, 1559 },
		{ 697, 637 },
		{ 253, 216 },
		{ 976, 937 },
		{ 370, 329 },
		{ 1907, 1895 },
		{ 1908, 1896 },
		{ 807, 749 },
		{ 814, 758 },
		{ 2057, 2039 },
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
		{ 2071, 2054 },
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
		{ 2019, 1998 },
		{ 283, 243 },
		{ 494, 443 },
		{ 257, 220 },
		{ 745, 687 },
		{ 3033, 3029 },
		{ 468, 417 },
		{ 1195, 1194 },
		{ 567, 508 },
		{ 877, 828 },
		{ 2090, 2074 },
		{ 432, 389 },
		{ 471, 420 },
		{ 1864, 1846 },
		{ 1103, 1082 },
		{ 1104, 1083 },
		{ 256, 219 },
		{ 1106, 1085 },
		{ 271, 233 },
		{ 803, 745 },
		{ 961, 921 },
		{ 1826, 1807 },
		{ 504, 449 },
		{ 1121, 1105 },
		{ 1829, 1810 },
		{ 750, 692 },
		{ 1882, 1866 },
		{ 452, 403 },
		{ 753, 695 },
		{ 1990, 1967 },
		{ 653, 588 },
		{ 757, 699 },
		{ 379, 339 },
		{ 2054, 2036 },
		{ 1138, 1127 },
		{ 268, 230 },
		{ 582, 522 },
		{ 2399, 2373 },
		{ 1895, 1881 },
		{ 1060, 1029 },
		{ 3012, 3010 },
		{ 982, 943 },
		{ 912, 864 },
		{ 706, 650 },
		{ 346, 302 },
		{ 584, 524 },
		{ 620, 557 },
		{ 347, 303 },
		{ 674, 612 },
		{ 2075, 2058 },
		{ 2014, 1993 },
		{ 541, 483 },
		{ 474, 423 },
		{ 858, 808 },
		{ 1087, 1061 },
		{ 917, 869 },
		{ 1807, 1783 },
		{ 1993, 1971 },
		{ 918, 870 },
		{ 1997, 1973 },
		{ 1809, 1785 },
		{ 225, 191 },
		{ 863, 813 },
		{ 1094, 1070 },
		{ 1000, 965 },
		{ 866, 817 },
		{ 867, 818 },
		{ 259, 221 },
		{ 2085, 2071 },
		{ 873, 824 },
		{ 943, 898 },
		{ 760, 702 },
		{ 692, 630 },
		{ 420, 379 },
		{ 695, 633 },
		{ 533, 472 },
		{ 1894, 1880 },
		{ 884, 835 },
		{ 1033, 1001 },
		{ 956, 916 },
		{ 588, 528 },
		{ 571, 512 },
		{ 1046, 1013 },
		{ 2123, 2104 },
		{ 430, 387 },
		{ 1137, 1126 },
		{ 1051, 1018 },
		{ 1052, 1019 },
		{ 1053, 1020 },
		{ 1918, 1907 },
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
		{ 1858, 1839 },
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
		{ 1878, 1862 },
		{ 2069, 2052 },
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
		{ 2016, 1995 },
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
		{ 3010, 3006 },
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
		{ 2055, 2037 },
		{ 2056, 2038 },
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
		{ 1848, 1829 },
		{ 1005, 971 },
		{ 682, 620 },
		{ 2040, 2019 },
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
		{ 1378, 1378 },
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
		{ 1378, 1378 },
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
		{ 1879, 1863 },
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
		{ 2037, 2016 },
		{ 2038, 2017 },
		{ 398, 359 },
		{ 1170, 1161 },
		{ 437, 396 },
		{ 626, 563 },
		{ 236, 202 },
		{ 1390, 1378 },
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
		{ 2963, 2960 },
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
		{ 2070, 2053 },
		{ 944, 899 },
		{ 2072, 2055 },
		{ 2073, 2056 },
		{ 598, 537 },
		{ 475, 424 },
		{ 844, 793 },
		{ 395, 355 },
		{ 1041, 1008 },
		{ 1042, 1009 },
		{ 2096, 2081 },
		{ 1694, 1691 },
		{ 1295, 1292 },
		{ 1127, 1111 },
		{ 1590, 1588 },
		{ 977, 938 },
		{ 1072, 1045 },
		{ 679, 617 },
		{ 1844, 1825 },
		{ 1921, 1912 },
		{ 2034, 2013 },
		{ 1644, 1641 },
		{ 552, 493 },
		{ 2138, 2125 },
		{ 691, 629 },
		{ 791, 733 },
		{ 1668, 1665 },
		{ 488, 438 },
		{ 325, 282 },
		{ 892, 843 },
		{ 1753, 1751 },
		{ 1618, 1615 },
		{ 1903, 1890 },
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
		{ 2098, 2084 },
		{ 2100, 2086 },
		{ 2101, 2087 },
		{ 2102, 2088 },
		{ 2103, 2089 },
		{ 382, 342 },
		{ 1998, 1974 },
		{ 1847, 1828 },
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
		{ 1862, 1844 },
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
		{ 2039, 2018 },
		{ 709, 653 },
		{ 829, 778 },
		{ 391, 351 },
		{ 772, 714 },
		{ 775, 717 },
		{ 1175, 1166 },
		{ 1896, 1882 },
		{ 993, 957 },
		{ 909, 860 },
		{ 459, 407 },
		{ 836, 785 },
		{ 267, 229 },
		{ 2052, 2034 },
		{ 1905, 1893 },
		{ 1088, 1062 },
		{ 209, 183 },
		{ 779, 721 },
		{ 1810, 1786 },
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
		{ 2091, 2075 },
		{ 1945, 1945 },
		{ 1945, 1945 },
		{ 1551, 1551 },
		{ 1551, 1551 },
		{ 1518, 1518 },
		{ 1518, 1518 },
		{ 2165, 2165 },
		{ 2165, 2165 },
		{ 1914, 1914 },
		{ 1914, 1914 },
		{ 1916, 1916 },
		{ 1916, 1916 },
		{ 1527, 1527 },
		{ 1527, 1527 },
		{ 2111, 2111 },
		{ 2111, 2111 },
		{ 2113, 2113 },
		{ 2113, 2113 },
		{ 2115, 2115 },
		{ 2115, 2115 },
		{ 2117, 2117 },
		{ 2117, 2117 },
		{ 311, 269 },
		{ 1945, 1945 },
		{ 1066, 1036 },
		{ 1551, 1551 },
		{ 1067, 1037 },
		{ 1518, 1518 },
		{ 678, 616 },
		{ 2165, 2165 },
		{ 766, 708 },
		{ 1914, 1914 },
		{ 748, 690 },
		{ 1916, 1916 },
		{ 562, 503 },
		{ 1527, 1527 },
		{ 973, 934 },
		{ 2111, 2111 },
		{ 645, 581 },
		{ 2113, 2113 },
		{ 2190, 2188 },
		{ 2115, 2115 },
		{ 665, 604 },
		{ 2117, 2117 },
		{ 2119, 2119 },
		{ 2119, 2119 },
		{ 2121, 2121 },
		{ 2121, 2121 },
		{ 1891, 1891 },
		{ 1891, 1891 },
		{ 643, 580 },
		{ 644, 580 },
		{ 1946, 1945 },
		{ 906, 857 },
		{ 1552, 1551 },
		{ 907, 858 },
		{ 1519, 1518 },
		{ 247, 212 },
		{ 2166, 2165 },
		{ 957, 917 },
		{ 1915, 1914 },
		{ 306, 264 },
		{ 1917, 1916 },
		{ 460, 408 },
		{ 1528, 1527 },
		{ 554, 495 },
		{ 2112, 2111 },
		{ 2119, 2119 },
		{ 2114, 2113 },
		{ 2121, 2121 },
		{ 2116, 2115 },
		{ 1891, 1891 },
		{ 2118, 2117 },
		{ 534, 473 },
		{ 1926, 1926 },
		{ 1926, 1926 },
		{ 2136, 2136 },
		{ 2136, 2136 },
		{ 1585, 1585 },
		{ 1585, 1585 },
		{ 1931, 1931 },
		{ 1931, 1931 },
		{ 2082, 2082 },
		{ 2082, 2082 },
		{ 2142, 2142 },
		{ 2142, 2142 },
		{ 1530, 1530 },
		{ 1530, 1530 },
		{ 1521, 1521 },
		{ 1521, 1521 },
		{ 1533, 1533 },
		{ 1533, 1533 },
		{ 1481, 1481 },
		{ 1481, 1481 },
		{ 1560, 1560 },
		{ 1560, 1560 },
		{ 2120, 2119 },
		{ 1926, 1926 },
		{ 2122, 2121 },
		{ 2136, 2136 },
		{ 1892, 1891 },
		{ 1585, 1585 },
		{ 289, 248 },
		{ 1931, 1931 },
		{ 272, 234 },
		{ 2082, 2082 },
		{ 537, 477 },
		{ 2142, 2142 },
		{ 1062, 1031 },
		{ 1530, 1530 },
		{ 401, 362 },
		{ 1521, 1521 },
		{ 839, 788 },
		{ 1533, 1533 },
		{ 808, 750 },
		{ 1481, 1481 },
		{ 1650, 1649 },
		{ 1560, 1560 },
		{ 1524, 1524 },
		{ 1524, 1524 },
		{ 1144, 1133 },
		{ 1040, 1007 },
		{ 1146, 1136 },
		{ 2124, 2105 },
		{ 1759, 1758 },
		{ 1090, 1065 },
		{ 1927, 1926 },
		{ 809, 751 },
		{ 2137, 2136 },
		{ 886, 837 },
		{ 1586, 1585 },
		{ 1150, 1140 },
		{ 1932, 1931 },
		{ 195, 176 },
		{ 2083, 2082 },
		{ 3043, 3041 },
		{ 2143, 2142 },
		{ 1152, 1142 },
		{ 1531, 1530 },
		{ 793, 735 },
		{ 1522, 1521 },
		{ 1524, 1524 },
		{ 1534, 1533 },
		{ 337, 293 },
		{ 1482, 1481 },
		{ 1887, 1872 },
		{ 1561, 1560 },
		{ 939, 893 },
		{ 1047, 1014 },
		{ 941, 895 },
		{ 1160, 1151 },
		{ 1198, 1197 },
		{ 728, 672 },
		{ 758, 700 },
		{ 1674, 1673 },
		{ 1624, 1623 },
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
		{ 3013, 3011 },
		{ 988, 949 },
		{ 1525, 1524 },
		{ 1301, 1300 },
		{ 1512, 1494 },
		{ 1919, 1908 },
		{ 1596, 1595 },
		{ 929, 883 },
		{ 1867, 1850 },
		{ 1113, 1095 },
		{ 830, 779 },
		{ 2060, 2043 },
		{ 1700, 1699 },
		{ 1906, 1894 },
		{ 1245, 1242 },
		{ 1166, 1157 },
		{ 222, 189 },
		{ 1992, 1970 },
		{ 2099, 2085 },
		{ 2004, 1981 },
		{ 481, 430 },
		{ 298, 257 },
		{ 601, 541 },
		{ 1806, 1782 },
		{ 2135, 2123 },
		{ 1925, 1918 },
		{ 2856, 2854 },
		{ 2602, 2602 },
		{ 2602, 2602 },
		{ 2717, 2717 },
		{ 2717, 2717 },
		{ 1689, 1689 },
		{ 1689, 1689 },
		{ 1275, 1275 },
		{ 1275, 1275 },
		{ 2509, 2509 },
		{ 2509, 2509 },
		{ 2405, 2405 },
		{ 2405, 2405 },
		{ 2580, 2580 },
		{ 2580, 2580 },
		{ 2727, 2727 },
		{ 2727, 2727 },
		{ 2728, 2728 },
		{ 2728, 2728 },
		{ 2729, 2729 },
		{ 2729, 2729 },
		{ 2852, 2852 },
		{ 2852, 2852 },
		{ 556, 497 },
		{ 2602, 2602 },
		{ 850, 800 },
		{ 2717, 2717 },
		{ 908, 859 },
		{ 1689, 1689 },
		{ 696, 636 },
		{ 1275, 1275 },
		{ 852, 802 },
		{ 2509, 2509 },
		{ 318, 275 },
		{ 2405, 2405 },
		{ 656, 592 },
		{ 2580, 2580 },
		{ 265, 227 },
		{ 2727, 2727 },
		{ 622, 559 },
		{ 2728, 2728 },
		{ 659, 595 },
		{ 2729, 2729 },
		{ 660, 596 },
		{ 2852, 2852 },
		{ 661, 597 },
		{ 2640, 2640 },
		{ 2640, 2640 },
		{ 2641, 2641 },
		{ 2641, 2641 },
		{ 2626, 2602 },
		{ 623, 560 },
		{ 2725, 2717 },
		{ 418, 377 },
		{ 1690, 1689 },
		{ 984, 945 },
		{ 1276, 1275 },
		{ 664, 603 },
		{ 2510, 2509 },
		{ 625, 562 },
		{ 2406, 2405 },
		{ 362, 321 },
		{ 2605, 2580 },
		{ 627, 564 },
		{ 2732, 2727 },
		{ 2921, 2919 },
		{ 2733, 2728 },
		{ 924, 879 },
		{ 2734, 2729 },
		{ 2640, 2640 },
		{ 2853, 2852 },
		{ 2641, 2641 },
		{ 1197, 1196 },
		{ 1318, 1318 },
		{ 1318, 1318 },
		{ 2455, 2455 },
		{ 2455, 2455 },
		{ 2737, 2737 },
		{ 2737, 2737 },
		{ 2917, 2917 },
		{ 2917, 2917 },
		{ 2678, 2678 },
		{ 2678, 2678 },
		{ 2352, 2352 },
		{ 2352, 2352 },
		{ 2211, 2211 },
		{ 2211, 2211 },
		{ 2458, 2458 },
		{ 2458, 2458 },
		{ 1613, 1613 },
		{ 1613, 1613 },
		{ 1290, 1290 },
		{ 1290, 1290 },
		{ 1639, 1639 },
		{ 1639, 1639 },
		{ 2659, 2640 },
		{ 1318, 1318 },
		{ 2660, 2641 },
		{ 2455, 2455 },
		{ 308, 266 },
		{ 2737, 2737 },
		{ 811, 755 },
		{ 2917, 2917 },
		{ 868, 819 },
		{ 2678, 2678 },
		{ 309, 267 },
		{ 2352, 2352 },
		{ 762, 704 },
		{ 2211, 2211 },
		{ 931, 885 },
		{ 2458, 2458 },
		{ 932, 886 },
		{ 1613, 1613 },
		{ 763, 705 },
		{ 1290, 1290 },
		{ 998, 963 },
		{ 1639, 1639 },
		{ 322, 279 },
		{ 2746, 2746 },
		{ 2746, 2746 },
		{ 2688, 2688 },
		{ 2688, 2688 },
		{ 1319, 1318 },
		{ 479, 428 },
		{ 2484, 2455 },
		{ 2157, 2154 },
		{ 2739, 2737 },
		{ 673, 611 },
		{ 2918, 2917 },
		{ 594, 534 },
		{ 2693, 2678 },
		{ 538, 478 },
		{ 2353, 2352 },
		{ 226, 192 },
		{ 2212, 2211 },
		{ 1074, 1047 },
		{ 2487, 2458 },
		{ 940, 894 },
		{ 1614, 1613 },
		{ 821, 767 },
		{ 1291, 1290 },
		{ 2746, 2746 },
		{ 1640, 1639 },
		{ 2688, 2688 },
		{ 352, 310 },
		{ 2869, 2869 },
		{ 2869, 2869 },
		{ 2650, 2650 },
		{ 2650, 2650 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2591, 2591 },
		{ 2591, 2591 },
		{ 2692, 2692 },
		{ 2692, 2692 },
		{ 2694, 2694 },
		{ 2694, 2694 },
		{ 2695, 2695 },
		{ 2695, 2695 },
		{ 2823, 2823 },
		{ 2823, 2823 },
		{ 2696, 2696 },
		{ 2696, 2696 },
		{ 2697, 2697 },
		{ 2697, 2697 },
		{ 1715, 1715 },
		{ 1715, 1715 },
		{ 2747, 2746 },
		{ 2869, 2869 },
		{ 2703, 2688 },
		{ 2650, 2650 },
		{ 368, 327 },
		{ 2567, 2567 },
		{ 569, 510 },
		{ 2591, 2591 },
		{ 1010, 976 },
		{ 2692, 2692 },
		{ 773, 715 },
		{ 2694, 2694 },
		{ 774, 716 },
		{ 2695, 2695 },
		{ 724, 668 },
		{ 2823, 2823 },
		{ 1720, 1717 },
		{ 2696, 2696 },
		{ 241, 207 },
		{ 2697, 2697 },
		{ 2783, 2781 },
		{ 1715, 1715 },
		{ 726, 670 },
		{ 1772, 1772 },
		{ 1772, 1772 },
		{ 2595, 2595 },
		{ 2595, 2595 },
		{ 2870, 2869 },
		{ 1154, 1144 },
		{ 2669, 2650 },
		{ 388, 348 },
		{ 2592, 2567 },
		{ 831, 780 },
		{ 2616, 2591 },
		{ 2874, 2871 },
		{ 2707, 2692 },
		{ 517, 458 },
		{ 2708, 2694 },
		{ 288, 247 },
		{ 2709, 2695 },
		{ 1159, 1150 },
		{ 2824, 2823 },
		{ 519, 460 },
		{ 2710, 2696 },
		{ 313, 271 },
		{ 2711, 2697 },
		{ 1772, 1772 },
		{ 1716, 1715 },
		{ 2595, 2595 },
		{ 465, 414 },
		{ 2596, 2596 },
		{ 2596, 2596 },
		{ 2482, 2482 },
		{ 2482, 2482 },
		{ 2658, 2658 },
		{ 2658, 2658 },
		{ 2779, 2779 },
		{ 2779, 2779 },
		{ 2705, 2705 },
		{ 2705, 2705 },
		{ 1749, 1749 },
		{ 1749, 1749 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 1240, 1240 },
		{ 1240, 1240 },
		{ 2486, 2486 },
		{ 2486, 2486 },
		{ 1663, 1663 },
		{ 1663, 1663 },
		{ 2958, 2958 },
		{ 2958, 2958 },
		{ 1773, 1772 },
		{ 2596, 2596 },
		{ 2619, 2595 },
		{ 2482, 2482 },
		{ 372, 332 },
		{ 2658, 2658 },
		{ 434, 391 },
		{ 2779, 2779 },
		{ 896, 847 },
		{ 2705, 2705 },
		{ 273, 235 },
		{ 1749, 1749 },
		{ 787, 729 },
		{ 2712, 2712 },
		{ 194, 175 },
		{ 1240, 1240 },
		{ 1169, 1160 },
		{ 2486, 2486 },
		{ 738, 681 },
		{ 1663, 1663 },
		{ 739, 682 },
		{ 2958, 2958 },
		{ 329, 286 },
		{ 652, 587 },
		{ 255, 218 },
		{ 968, 929 },
		{ 694, 632 },
		{ 2620, 2596 },
		{ 3074, 3074 },
		{ 2513, 2482 },
		{ 3082, 3082 },
		{ 2677, 2658 },
		{ 3086, 3086 },
		{ 2780, 2779 },
		{ 1587, 1586 },
		{ 2718, 2705 },
		{ 2097, 2083 },
		{ 1750, 1749 },
		{ 1562, 1552 },
		{ 2720, 2712 },
		{ 1947, 1946 },
		{ 1241, 1240 },
		{ 1933, 1927 },
		{ 2516, 2486 },
		{ 1568, 1561 },
		{ 1664, 1663 },
		{ 1544, 1522 },
		{ 2959, 2958 },
		{ 2168, 2166 },
		{ 2144, 2137 },
		{ 1500, 1482 },
		{ 3074, 3074 },
		{ 1936, 1932 },
		{ 3082, 3082 },
		{ 1923, 1915 },
		{ 3086, 3086 },
		{ 1548, 1534 },
		{ 2129, 2112 },
		{ 2149, 2143 },
		{ 1924, 1917 },
		{ 2130, 2114 },
		{ 1546, 1528 },
		{ 2131, 2116 },
		{ 1543, 1519 },
		{ 2132, 2118 },
		{ 1545, 1525 },
		{ 2133, 2120 },
		{ 1547, 1531 },
		{ 2134, 2122 },
		{ 1904, 1892 },
		{ 2990, 2989 },
		{ 1680, 1679 },
		{ 1307, 1306 },
		{ 1578, 1574 },
		{ 3048, 3047 },
		{ 3049, 3048 },
		{ 1655, 1654 },
		{ 3077, 3074 },
		{ 1656, 1655 },
		{ 3084, 3082 },
		{ 1306, 1305 },
		{ 3087, 3086 },
		{ 1574, 1569 },
		{ 1575, 1570 },
		{ 1350, 1349 },
		{ 1705, 1704 },
		{ 1706, 1705 },
		{ 1601, 1600 },
		{ 1602, 1601 },
		{ 1629, 1628 },
		{ 1764, 1763 },
		{ 1765, 1764 },
		{ 1630, 1629 },
		{ 1679, 1678 },
		{ 2753, 2753 },
		{ 2750, 2753 },
		{ 163, 163 },
		{ 160, 163 },
		{ 1898, 1886 },
		{ 1899, 1886 },
		{ 1876, 1861 },
		{ 1877, 1861 },
		{ 1952, 1952 },
		{ 2178, 2178 },
		{ 2183, 2179 },
		{ 1957, 1953 },
		{ 2752, 2748 },
		{ 162, 158 },
		{ 168, 164 },
		{ 2182, 2179 },
		{ 1956, 1953 },
		{ 2751, 2748 },
		{ 161, 158 },
		{ 167, 164 },
		{ 2759, 2756 },
		{ 2758, 2754 },
		{ 2236, 2213 },
		{ 2753, 2753 },
		{ 88, 70 },
		{ 163, 163 },
		{ 2757, 2754 },
		{ 2235, 2213 },
		{ 2761, 2760 },
		{ 87, 70 },
		{ 1960, 1959 },
		{ 1952, 1952 },
		{ 2178, 2178 },
		{ 2184, 2181 },
		{ 2186, 2185 },
		{ 2029, 2008 },
		{ 169, 166 },
		{ 171, 170 },
		{ 2754, 2753 },
		{ 119, 103 },
		{ 164, 163 },
		{ 1836, 1817 },
		{ 1958, 1955 },
		{ 2292, 2263 },
		{ 1911, 1901 },
		{ 2173, 2172 },
		{ 1953, 1952 },
		{ 2179, 2178 },
		{ 2760, 2758 },
		{ 166, 162 },
		{ 2263, 2236 },
		{ 2185, 2183 },
		{ 1955, 1951 },
		{ 103, 88 },
		{ 1817, 1795 },
		{ 170, 168 },
		{ 2756, 2752 },
		{ 2181, 2177 },
		{ 1959, 1957 },
		{ 2008, 1986 },
		{ 0, 2540 },
		{ 0, 2980 },
		{ 0, 2577 },
		{ 0, 2889 },
		{ 0, 2807 },
		{ 2619, 2619 },
		{ 2619, 2619 },
		{ 0, 2892 },
		{ 0, 2667 },
		{ 2732, 2732 },
		{ 2732, 2732 },
		{ 2733, 2733 },
		{ 2733, 2733 },
		{ 2734, 2734 },
		{ 2734, 2734 },
		{ 0, 2897 },
		{ 2620, 2620 },
		{ 2620, 2620 },
		{ 0, 2814 },
		{ 2669, 2669 },
		{ 2669, 2669 },
		{ 0, 1282 },
		{ 0, 1740 },
		{ 2739, 2739 },
		{ 2739, 2739 },
		{ 0, 1250 },
		{ 2510, 2510 },
		{ 2510, 2510 },
		{ 2619, 2619 },
		{ 0, 2200 },
		{ 2484, 2484 },
		{ 2484, 2484 },
		{ 2732, 2732 },
		{ 0, 2908 },
		{ 2733, 2733 },
		{ 0, 2676 },
		{ 2734, 2734 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2620, 2620 },
		{ 0, 2584 },
		{ 0, 2679 },
		{ 2669, 2669 },
		{ 2747, 2747 },
		{ 2747, 2747 },
		{ 0, 2681 },
		{ 2739, 2739 },
		{ 0, 2682 },
		{ 0, 2630 },
		{ 2510, 2510 },
		{ 0, 2415 },
		{ 2516, 2516 },
		{ 2516, 2516 },
		{ 2484, 2484 },
		{ 0, 2835 },
		{ 0, 2922 },
		{ 0, 2633 },
		{ 2487, 2487 },
		{ 2487, 2487 },
		{ 0, 2926 },
		{ 2677, 2677 },
		{ 2761, 2761 },
		{ 2762, 2761 },
		{ 0, 2551 },
		{ 0, 2416 },
		{ 0, 2930 },
		{ 2747, 2747 },
		{ 0, 2553 },
		{ 2592, 2592 },
		{ 2592, 2592 },
		{ 0, 2933 },
		{ 0, 2844 },
		{ 0, 2417 },
		{ 0, 2769 },
		{ 2516, 2516 },
		{ 2693, 2693 },
		{ 2693, 2693 },
		{ 0, 2938 },
		{ 0, 2463 },
		{ 0, 1724 },
		{ 2487, 2487 },
		{ 2186, 2186 },
		{ 2187, 2186 },
		{ 0, 1227 },
		{ 2761, 2761 },
		{ 0, 2524 },
		{ 0, 2835 },
		{ 0, 2561 },
		{ 0, 1735 },
		{ 0, 1266 },
		{ 0, 2949 },
		{ 2592, 2592 },
		{ 2703, 2703 },
		{ 2703, 2703 },
		{ 0, 1261 },
		{ 0, 2860 },
		{ 0, 2784 },
		{ 0, 2426 },
		{ 2693, 2693 },
		{ 2605, 2605 },
		{ 2605, 2605 },
		{ 2709, 2709 },
		{ 2709, 2709 },
		{ 0, 2531 },
		{ 2186, 2186 },
		{ 0, 2426 },
		{ 0, 2789 },
		{ 0, 2386 },
		{ 0, 2474 },
		{ 171, 171 },
		{ 172, 171 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 2718, 2718 },
		{ 2718, 2718 },
		{ 2703, 2703 },
		{ 0, 2875 },
		{ 0, 2719 },
		{ 2720, 2720 },
		{ 2720, 2720 },
		{ 0, 2969 },
		{ 0, 2879 },
		{ 2605, 2605 },
		{ 0, 2537 },
		{ 2709, 2709 },
		{ 1960, 1960 },
		{ 1961, 1960 },
		{ 2660, 2660 },
		{ 2660, 2660 },
		{ 0, 2800 },
		{ 0, 2453 },
		{ 0, 2322 },
		{ 171, 171 },
		{ 1206, 1206 },
		{ 2353, 2353 },
		{ 1349, 1348 },
		{ 2718, 2718 },
		{ 2725, 2725 },
		{ 2725, 2725 },
		{ 1855, 1836 },
		{ 165, 167 },
		{ 2720, 2720 },
		{ 2616, 2616 },
		{ 2616, 2616 },
		{ 2755, 2757 },
		{ 0, 1794 },
		{ 2180, 2176 },
		{ 0, 1985 },
		{ 1960, 1960 },
		{ 0, 0 },
		{ 2660, 2660 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1206, 1206 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2725, 2725 },
		{ 0, 0 },
		{ 0, 2322 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2616, 2616 }
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
		{ -1222, 3147, 143 },
		{ 11, 0, 143 },
		{ -1243, 3133, 151 },
		{ 13, 0, 151 },
		{ -1278, 3257, 0 },
		{ 15, 0, 0 },
		{ -1293, 3127, 139 },
		{ 17, 0, 139 },
		{ -1321, 3120, 22 },
		{ 19, 0, 22 },
		{ -1363, 230, 0 },
		{ 21, 0, 0 },
		{ -1589, 3258, 0 },
		{ 23, 0, 0 },
		{ -1616, 3139, 0 },
		{ 25, 0, 0 },
		{ -1642, 3128, 0 },
		{ 27, 0, 0 },
		{ -1666, 3122, 0 },
		{ 29, 0, 0 },
		{ -1692, 3143, 0 },
		{ 31, 0, 0 },
		{ -1718, 3121, 155 },
		{ 33, 0, 155 },
		{ -1752, 3262, 262 },
		{ 35, 0, 262 },
		{ 38, 127, 0 },
		{ -1788, 344, 0 },
		{ 40, 16, 0 },
		{ -1977, 116, 0 },
		{ -2189, 3255, 0 },
		{ 41, 0, 0 },
		{ 44, 14, 0 },
		{ -86, 458, 0 },
		{ -2764, 3126, 147 },
		{ 45, 0, 147 },
		{ -2782, 3256, 170 },
		{ 47, 0, 170 },
		{ 2826, 1438, 0 },
		{ 49, 0, 0 },
		{ -2828, 3259, 268 },
		{ 51, 0, 268 },
		{ -2855, 3260, 173 },
		{ 53, 0, 173 },
		{ -2873, 3137, 166 },
		{ 55, 0, 166 },
		{ -2920, 3265, 159 },
		{ 57, 0, 159 },
		{ -2961, 3145, 165 },
		{ 59, 0, 165 },
		{ -86, 1, 0 },
		{ 61, 0, 0 },
		{ -3009, 1788, 0 },
		{ 63, 0, 0 },
		{ -3019, 1697, 42 },
		{ 65, 0, 42 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 420 },
		{ 2760, 4690, 427 },
		{ 0, 0, 240 },
		{ 0, 0, 242 },
		{ 157, 1283, 259 },
		{ 157, 1398, 259 },
		{ 157, 1297, 259 },
		{ 157, 1304, 259 },
		{ 157, 1304, 259 },
		{ 157, 1308, 259 },
		{ 157, 1313, 259 },
		{ 157, 1307, 259 },
		{ 3068, 2801, 427 },
		{ 157, 1318, 259 },
		{ 3068, 1676, 258 },
		{ 102, 2594, 427 },
		{ 157, 0, 259 },
		{ 0, 0, 427 },
		{ -87, 10, 236 },
		{ -88, 4719, 0 },
		{ 157, 686, 259 },
		{ 157, 720, 259 },
		{ 157, 689, 259 },
		{ 157, 703, 259 },
		{ 157, 711, 259 },
		{ 157, 711, 259 },
		{ 157, 718, 259 },
		{ 157, 733, 259 },
		{ 157, 727, 259 },
		{ 3037, 2286, 0 },
		{ 157, 753, 259 },
		{ 3068, 1764, 255 },
		{ 117, 1450, 0 },
		{ 3068, 1731, 256 },
		{ 2760, 4700, 0 },
		{ 157, 762, 259 },
		{ 157, 760, 259 },
		{ 157, 761, 259 },
		{ 157, 757, 259 },
		{ 157, 0, 247 },
		{ 157, 776, 259 },
		{ 157, 778, 259 },
		{ 157, 788, 259 },
		{ 157, 793, 259 },
		{ 3065, 2909, 0 },
		{ 157, 800, 259 },
		{ 131, 1484, 0 },
		{ 117, 0, 0 },
		{ 2994, 2636, 257 },
		{ 133, 1437, 0 },
		{ 0, 0, 238 },
		{ 157, 804, 243 },
		{ 157, 807, 259 },
		{ 157, 827, 259 },
		{ 157, 832, 259 },
		{ 157, 830, 259 },
		{ 157, 823, 259 },
		{ 157, 0, 250 },
		{ 157, 824, 259 },
		{ 0, 0, 252 },
		{ 157, 830, 259 },
		{ 131, 0, 0 },
		{ 2994, 2659, 255 },
		{ 133, 0, 0 },
		{ 2994, 2703, 256 },
		{ 157, 845, 259 },
		{ 157, 842, 259 },
		{ 157, 843, 259 },
		{ 157, 869, 259 },
		{ 157, 980, 259 },
		{ 157, 0, 249 },
		{ 157, 1052, 259 },
		{ 157, 1056, 259 },
		{ 157, 1076, 259 },
		{ 157, 0, 245 },
		{ 157, 1233, 259 },
		{ 157, 0, 246 },
		{ 157, 0, 248 },
		{ 157, 1224, 259 },
		{ 157, 1252, 259 },
		{ 157, 0, 244 },
		{ 157, 1271, 259 },
		{ 157, 0, 251 },
		{ 157, 738, 259 },
		{ 157, 1299, 259 },
		{ 0, 0, 254 },
		{ 157, 1284, 259 },
		{ 157, 1287, 259 },
		{ 3085, 1355, 253 },
		{ 2760, 4679, 427 },
		{ 163, 0, 240 },
		{ 0, 0, 241 },
		{ -161, 22, 236 },
		{ -162, 4715, 0 },
		{ 3040, 4701, 0 },
		{ 2760, 4680, 0 },
		{ 0, 0, 237 },
		{ 2760, 4697, 0 },
		{ -167, 4898, 0 },
		{ -168, 4721, 0 },
		{ 171, 0, 238 },
		{ 2760, 4698, 0 },
		{ 3040, 4868, 0 },
		{ 0, 0, 239 },
		{ 3041, 1577, 137 },
		{ 2089, 4085, 137 },
		{ 2919, 4563, 137 },
		{ 3041, 4247, 137 },
		{ 0, 0, 137 },
		{ 3029, 3343, 0 },
		{ 2104, 2980, 0 },
		{ 3029, 3586, 0 },
		{ 3006, 3319, 0 },
		{ 3006, 3320, 0 },
		{ 2089, 4080, 0 },
		{ 3033, 3231, 0 },
		{ 2089, 4097, 0 },
		{ 3007, 3507, 0 },
		{ 3036, 3214, 0 },
		{ 3007, 3264, 0 },
		{ 2854, 4299, 0 },
		{ 3033, 3254, 0 },
		{ 2104, 3660, 0 },
		{ 2919, 4443, 0 },
		{ 2035, 3419, 0 },
		{ 2035, 3465, 0 },
		{ 2057, 3066, 0 },
		{ 2038, 3777, 0 },
		{ 2019, 3842, 0 },
		{ 2038, 3789, 0 },
		{ 2057, 3069, 0 },
		{ 2057, 3094, 0 },
		{ 3033, 3299, 0 },
		{ 2960, 3924, 0 },
		{ 2089, 4049, 0 },
		{ 1175, 4004, 0 },
		{ 2035, 3417, 0 },
		{ 2057, 3104, 0 },
		{ 2919, 4495, 0 },
		{ 2035, 3440, 0 },
		{ 3006, 3723, 0 },
		{ 3029, 3584, 0 },
		{ 2104, 3696, 0 },
		{ 2188, 4170, 0 },
		{ 2919, 3600, 0 },
		{ 2960, 3880, 0 },
		{ 2019, 3828, 0 },
		{ 3007, 3533, 0 },
		{ 1997, 3273, 0 },
		{ 2919, 4573, 0 },
		{ 3029, 3614, 0 },
		{ 2960, 3602, 0 },
		{ 2104, 3666, 0 },
		{ 2057, 3106, 0 },
		{ 3036, 3370, 0 },
		{ 3006, 3728, 0 },
		{ 1997, 3272, 0 },
		{ 2019, 3815, 0 },
		{ 2919, 4347, 0 },
		{ 2089, 4058, 0 },
		{ 2089, 4076, 0 },
		{ 3029, 3633, 0 },
		{ 2057, 3122, 0 },
		{ 2089, 4087, 0 },
		{ 3029, 3616, 0 },
		{ 2188, 4217, 0 },
		{ 2919, 4559, 0 },
		{ 3036, 3372, 0 },
		{ 3007, 3555, 0 },
		{ 2057, 3159, 0 },
		{ 3036, 3292, 0 },
		{ 3029, 3574, 0 },
		{ 2019, 3854, 0 },
		{ 2960, 3910, 0 },
		{ 2104, 3600, 0 },
		{ 1067, 3219, 0 },
		{ 3029, 3596, 0 },
		{ 3006, 3751, 0 },
		{ 2919, 4515, 0 },
		{ 2188, 4215, 0 },
		{ 2057, 3160, 0 },
		{ 2104, 3689, 0 },
		{ 3036, 3347, 0 },
		{ 2089, 4105, 0 },
		{ 1997, 3242, 0 },
		{ 2089, 4055, 0 },
		{ 3007, 3559, 0 },
		{ 2057, 3161, 0 },
		{ 2854, 4304, 0 },
		{ 3006, 3722, 0 },
		{ 3036, 3382, 0 },
		{ 2125, 3592, 0 },
		{ 2057, 3162, 0 },
		{ 2960, 3889, 0 },
		{ 2089, 4018, 0 },
		{ 2188, 4174, 0 },
		{ 2089, 4040, 0 },
		{ 2919, 4409, 0 },
		{ 2919, 4415, 0 },
		{ 2019, 3837, 0 },
		{ 2188, 4135, 0 },
		{ 2057, 3163, 0 },
		{ 2919, 4521, 0 },
		{ 2960, 3948, 0 },
		{ 2019, 3853, 0 },
		{ 2960, 3872, 0 },
		{ 2919, 4343, 0 },
		{ 2035, 3463, 0 },
		{ 3007, 3511, 0 },
		{ 2089, 4088, 0 },
		{ 2919, 4427, 0 },
		{ 1175, 3987, 0 },
		{ 1067, 3220, 0 },
		{ 2125, 3980, 0 },
		{ 2038, 3781, 0 },
		{ 3007, 3545, 0 },
		{ 2057, 3164, 0 },
		{ 2919, 4571, 0 },
		{ 2089, 4056, 0 },
		{ 2057, 3165, 0 },
		{ 0, 0, 70 },
		{ 3029, 3331, 0 },
		{ 3036, 3322, 0 },
		{ 2089, 4083, 0 },
		{ 3041, 4257, 0 },
		{ 2057, 3166, 0 },
		{ 2057, 3167, 0 },
		{ 3036, 3353, 0 },
		{ 2035, 3433, 0 },
		{ 2019, 3810, 0 },
		{ 3036, 3355, 0 },
		{ 2057, 3168, 0 },
		{ 2089, 4045, 0 },
		{ 3029, 3642, 0 },
		{ 3029, 3645, 0 },
		{ 2038, 3776, 0 },
		{ 3007, 3521, 0 },
		{ 841, 3236, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2035, 3466, 0 },
		{ 2919, 4454, 0 },
		{ 2960, 3929, 0 },
		{ 2019, 3826, 0 },
		{ 3036, 3381, 0 },
		{ 3007, 3556, 0 },
		{ 0, 0, 68 },
		{ 2057, 3169, 0 },
		{ 2035, 3423, 0 },
		{ 3036, 3383, 0 },
		{ 2960, 3941, 0 },
		{ 3036, 3390, 0 },
		{ 2919, 4371, 0 },
		{ 3007, 3528, 0 },
		{ 1175, 4012, 0 },
		{ 2035, 3460, 0 },
		{ 3006, 3762, 0 },
		{ 2089, 4060, 0 },
		{ 2919, 4481, 0 },
		{ 3041, 4271, 0 },
		{ 3007, 3535, 0 },
		{ 0, 0, 62 },
		{ 3007, 3543, 0 },
		{ 2919, 4553, 0 },
		{ 1175, 3998, 0 },
		{ 2960, 3935, 0 },
		{ 2089, 4086, 0 },
		{ 0, 0, 73 },
		{ 3036, 3404, 0 },
		{ 3029, 3578, 0 },
		{ 3029, 3630, 0 },
		{ 2057, 3170, 0 },
		{ 2960, 3904, 0 },
		{ 2089, 4026, 0 },
		{ 2057, 3171, 0 },
		{ 2035, 3412, 0 },
		{ 3006, 3732, 0 },
		{ 3036, 3323, 0 },
		{ 3007, 3508, 0 },
		{ 2919, 4507, 0 },
		{ 2057, 3172, 0 },
		{ 2960, 3879, 0 },
		{ 2089, 4067, 0 },
		{ 3036, 3325, 0 },
		{ 3007, 3523, 0 },
		{ 2960, 3897, 0 },
		{ 1009, 3958, 0 },
		{ 0, 0, 8 },
		{ 2035, 3427, 0 },
		{ 2038, 3799, 0 },
		{ 2960, 3920, 0 },
		{ 2070, 3215, 0 },
		{ 2057, 3047, 0 },
		{ 2188, 4223, 0 },
		{ 2089, 4110, 0 },
		{ 2035, 3457, 0 },
		{ 2089, 4019, 0 },
		{ 2038, 3787, 0 },
		{ 2057, 3056, 0 },
		{ 3036, 3356, 0 },
		{ 3036, 3252, 0 },
		{ 2089, 4054, 0 },
		{ 3033, 3301, 0 },
		{ 3007, 3561, 0 },
		{ 1175, 4007, 0 },
		{ 3006, 3716, 0 },
		{ 2057, 3064, 0 },
		{ 2104, 3050, 0 },
		{ 2919, 4363, 0 },
		{ 1175, 4002, 0 },
		{ 2104, 3672, 0 },
		{ 3041, 3207, 0 },
		{ 2070, 3213, 0 },
		{ 2038, 3782, 0 },
		{ 2035, 3425, 0 },
		{ 3036, 3403, 0 },
		{ 0, 0, 114 },
		{ 2057, 3068, 0 },
		{ 2104, 3683, 0 },
		{ 1996, 3229, 0 },
		{ 3029, 3609, 0 },
		{ 3006, 3741, 0 },
		{ 2919, 4555, 0 },
		{ 2089, 4035, 0 },
		{ 0, 0, 7 },
		{ 2038, 3785, 0 },
		{ 0, 0, 6 },
		{ 2960, 3922, 0 },
		{ 0, 0, 119 },
		{ 3006, 3746, 0 },
		{ 2089, 4053, 0 },
		{ 3041, 1635, 0 },
		{ 2057, 3070, 0 },
		{ 3006, 3770, 0 },
		{ 3029, 3625, 0 },
		{ 2057, 3072, 0 },
		{ 2089, 4061, 0 },
		{ 3041, 3218, 0 },
		{ 2089, 4074, 0 },
		{ 2188, 4176, 0 },
		{ 2104, 3694, 0 },
		{ 0, 0, 69 },
		{ 2019, 3831, 0 },
		{ 2057, 3073, 106 },
		{ 2057, 3074, 107 },
		{ 2919, 4526, 0 },
		{ 2960, 3915, 0 },
		{ 3007, 3564, 0 },
		{ 3029, 3604, 0 },
		{ 3007, 3481, 0 },
		{ 1175, 3996, 0 },
		{ 3029, 3610, 0 },
		{ 3007, 3505, 0 },
		{ 3033, 3310, 0 },
		{ 2104, 3651, 0 },
		{ 2960, 3955, 0 },
		{ 2089, 4039, 0 },
		{ 2057, 3075, 0 },
		{ 3036, 3357, 0 },
		{ 2919, 4433, 0 },
		{ 2960, 3881, 0 },
		{ 2854, 4303, 0 },
		{ 2960, 3882, 0 },
		{ 3007, 3516, 0 },
		{ 2960, 3895, 0 },
		{ 0, 0, 9 },
		{ 2057, 3076, 0 },
		{ 2960, 3898, 0 },
		{ 2070, 3200, 0 },
		{ 2125, 3979, 0 },
		{ 0, 0, 104 },
		{ 2035, 3430, 0 },
		{ 3006, 3768, 0 },
		{ 3029, 3593, 0 },
		{ 2104, 3601, 0 },
		{ 3006, 3216, 0 },
		{ 2960, 3928, 0 },
		{ 3033, 3270, 0 },
		{ 2960, 3930, 0 },
		{ 3033, 3247, 0 },
		{ 3029, 3620, 0 },
		{ 2089, 4103, 0 },
		{ 3036, 3384, 0 },
		{ 3007, 3548, 0 },
		{ 3033, 3240, 0 },
		{ 3006, 3757, 0 },
		{ 3036, 3391, 0 },
		{ 2960, 3871, 0 },
		{ 3036, 3293, 0 },
		{ 2919, 4513, 0 },
		{ 2057, 3077, 0 },
		{ 2919, 4519, 0 },
		{ 3007, 3562, 0 },
		{ 2089, 4047, 0 },
		{ 3029, 3569, 0 },
		{ 3029, 3571, 0 },
		{ 2019, 3814, 0 },
		{ 2035, 3415, 0 },
		{ 2057, 3078, 94 },
		{ 3007, 3497, 0 },
		{ 2057, 3079, 0 },
		{ 2057, 3082, 0 },
		{ 3033, 3296, 0 },
		{ 2104, 3674, 0 },
		{ 2188, 4186, 0 },
		{ 2057, 3083, 0 },
		{ 2035, 3429, 0 },
		{ 0, 0, 103 },
		{ 2188, 4219, 0 },
		{ 2919, 4441, 0 },
		{ 3036, 3336, 0 },
		{ 3036, 3344, 0 },
		{ 0, 0, 116 },
		{ 0, 0, 118 },
		{ 2104, 3650, 0 },
		{ 2035, 3439, 0 },
		{ 2089, 3330, 0 },
		{ 3036, 3349, 0 },
		{ 2089, 4100, 0 },
		{ 2057, 3084, 0 },
		{ 2089, 4104, 0 },
		{ 2960, 3894, 0 },
		{ 3006, 3749, 0 },
		{ 2057, 3085, 0 },
		{ 2125, 3974, 0 },
		{ 3033, 3294, 0 },
		{ 2188, 4178, 0 },
		{ 2057, 3086, 0 },
		{ 2919, 4333, 0 },
		{ 2035, 3411, 0 },
		{ 939, 3860, 0 },
		{ 3036, 3358, 0 },
		{ 3006, 3712, 0 },
		{ 2104, 3700, 0 },
		{ 2188, 4147, 0 },
		{ 3036, 3369, 0 },
		{ 1997, 3255, 0 },
		{ 2057, 3089, 0 },
		{ 2960, 3931, 0 },
		{ 3029, 3606, 0 },
		{ 2035, 3421, 0 },
		{ 2919, 4483, 0 },
		{ 3036, 3375, 0 },
		{ 2104, 3680, 0 },
		{ 2070, 3214, 0 },
		{ 3007, 3506, 0 },
		{ 2035, 3426, 0 },
		{ 2104, 3695, 0 },
		{ 2038, 3773, 0 },
		{ 3041, 3293, 0 },
		{ 2057, 3090, 0 },
		{ 0, 0, 63 },
		{ 2057, 3091, 0 },
		{ 3029, 3634, 0 },
		{ 3007, 3517, 0 },
		{ 3029, 3643, 0 },
		{ 3007, 3519, 0 },
		{ 2057, 3092, 108 },
		{ 2019, 3833, 0 },
		{ 2104, 3679, 0 },
		{ 2038, 3774, 0 },
		{ 2035, 3437, 0 },
		{ 2035, 3438, 0 },
		{ 2019, 3857, 0 },
		{ 2038, 3780, 0 },
		{ 2919, 4439, 0 },
		{ 3029, 3575, 0 },
		{ 3033, 3307, 0 },
		{ 2960, 3954, 0 },
		{ 3036, 3393, 0 },
		{ 2035, 3442, 0 },
		{ 0, 0, 120 },
		{ 2854, 4305, 0 },
		{ 2038, 3788, 0 },
		{ 3036, 3396, 0 },
		{ 3006, 3714, 0 },
		{ 0, 0, 115 },
		{ 0, 0, 105 },
		{ 2035, 3458, 0 },
		{ 3007, 3554, 0 },
		{ 3036, 3400, 0 },
		{ 2104, 2956, 0 },
		{ 3041, 3251, 0 },
		{ 2960, 3899, 0 },
		{ 3006, 3735, 0 },
		{ 2057, 3095, 0 },
		{ 2960, 3914, 0 },
		{ 2019, 3823, 0 },
		{ 3029, 3644, 0 },
		{ 2089, 4092, 0 },
		{ 2919, 4349, 0 },
		{ 2919, 4361, 0 },
		{ 2035, 3477, 0 },
		{ 2919, 4369, 0 },
		{ 2960, 3923, 0 },
		{ 2919, 4373, 0 },
		{ 3007, 3563, 0 },
		{ 3006, 3754, 0 },
		{ 2057, 3096, 0 },
		{ 2089, 4108, 0 },
		{ 3007, 3565, 0 },
		{ 2057, 3097, 0 },
		{ 3007, 3483, 0 },
		{ 2089, 4020, 0 },
		{ 2960, 3936, 0 },
		{ 2089, 4030, 0 },
		{ 3007, 3488, 0 },
		{ 2089, 4036, 0 },
		{ 2035, 3414, 0 },
		{ 3006, 3715, 0 },
		{ 2057, 3098, 0 },
		{ 3041, 4165, 0 },
		{ 2188, 4151, 0 },
		{ 2089, 4046, 0 },
		{ 2038, 3783, 0 },
		{ 2089, 4048, 0 },
		{ 2038, 3784, 0 },
		{ 3029, 3580, 0 },
		{ 2919, 4572, 0 },
		{ 3029, 3628, 0 },
		{ 0, 0, 96 },
		{ 2960, 3883, 0 },
		{ 2960, 3887, 0 },
		{ 2919, 4345, 0 },
		{ 2057, 3099, 0 },
		{ 2057, 3100, 0 },
		{ 2919, 4351, 0 },
		{ 2919, 4353, 0 },
		{ 2919, 4355, 0 },
		{ 2038, 3798, 0 },
		{ 2035, 3420, 0 },
		{ 0, 0, 124 },
		{ 0, 0, 117 },
		{ 0, 0, 121 },
		{ 2919, 4367, 0 },
		{ 2188, 4155, 0 },
		{ 1067, 3225, 0 },
		{ 2057, 3102, 0 },
		{ 2960, 3053, 0 },
		{ 3007, 3518, 0 },
		{ 2038, 3778, 0 },
		{ 1175, 3991, 0 },
		{ 2919, 4437, 0 },
		{ 3029, 3646, 0 },
		{ 3029, 3587, 0 },
		{ 3029, 3594, 0 },
		{ 3006, 3763, 0 },
		{ 2188, 4141, 0 },
		{ 2125, 3969, 0 },
		{ 3006, 3767, 0 },
		{ 3033, 3304, 0 },
		{ 2019, 3840, 0 },
		{ 1175, 3994, 0 },
		{ 3036, 3354, 0 },
		{ 2019, 3847, 0 },
		{ 2035, 3428, 0 },
		{ 2057, 3105, 0 },
		{ 2038, 3794, 0 },
		{ 2019, 3808, 0 },
		{ 2089, 4033, 0 },
		{ 2125, 3976, 0 },
		{ 2104, 3671, 0 },
		{ 3007, 3530, 0 },
		{ 2919, 4575, 0 },
		{ 2104, 3673, 0 },
		{ 0, 0, 57 },
		{ 0, 0, 58 },
		{ 2919, 4339, 0 },
		{ 3007, 3532, 0 },
		{ 0, 0, 67 },
		{ 0, 0, 112 },
		{ 1997, 3256, 0 },
		{ 3033, 3278, 0 },
		{ 2035, 3435, 0 },
		{ 3033, 3286, 0 },
		{ 3036, 3367, 0 },
		{ 2960, 3893, 0 },
		{ 3007, 3549, 0 },
		{ 0, 0, 98 },
		{ 3007, 3550, 0 },
		{ 0, 0, 100 },
		{ 3029, 3641, 0 },
		{ 3007, 3553, 0 },
		{ 3006, 3760, 0 },
		{ 2089, 4065, 0 },
		{ 2070, 3207, 0 },
		{ 2070, 3210, 0 },
		{ 3036, 3371, 0 },
		{ 1175, 3989, 0 },
		{ 2125, 3718, 0 },
		{ 3007, 3557, 0 },
		{ 939, 3863, 0 },
		{ 2019, 3822, 0 },
		{ 0, 0, 113 },
		{ 0, 0, 123 },
		{ 3007, 3558, 0 },
		{ 0, 0, 136 },
		{ 2035, 3443, 0 },
		{ 3041, 3879, 0 },
		{ 2919, 4491, 0 },
		{ 1175, 4009, 0 },
		{ 2919, 4499, 0 },
		{ 2089, 4102, 0 },
		{ 3041, 4266, 0 },
		{ 3007, 3560, 0 },
		{ 3041, 4274, 0 },
		{ 3033, 3297, 0 },
		{ 3033, 3298, 0 },
		{ 3006, 3049, 0 },
		{ 2057, 3107, 0 },
		{ 2089, 4016, 0 },
		{ 2960, 3943, 0 },
		{ 2919, 4567, 0 },
		{ 2919, 4569, 0 },
		{ 2960, 3946, 0 },
		{ 2104, 3690, 0 },
		{ 2960, 3949, 0 },
		{ 2104, 3693, 0 },
		{ 2104, 3603, 0 },
		{ 2960, 3877, 0 },
		{ 2057, 3108, 0 },
		{ 2188, 4145, 0 },
		{ 2057, 3109, 0 },
		{ 3029, 3623, 0 },
		{ 2057, 3110, 0 },
		{ 2038, 3791, 0 },
		{ 3029, 3626, 0 },
		{ 2019, 3816, 0 },
		{ 2960, 3892, 0 },
		{ 2057, 3112, 0 },
		{ 3029, 3629, 0 },
		{ 3041, 4267, 0 },
		{ 1175, 4011, 0 },
		{ 2104, 3670, 0 },
		{ 3007, 3489, 0 },
		{ 2919, 4417, 0 },
		{ 2919, 4423, 0 },
		{ 2089, 4059, 0 },
		{ 2038, 3775, 0 },
		{ 2188, 4143, 0 },
		{ 3007, 3496, 0 },
		{ 2089, 4062, 0 },
		{ 2089, 4063, 0 },
		{ 2960, 3900, 0 },
		{ 2960, 3901, 0 },
		{ 2089, 4068, 0 },
		{ 2919, 4487, 0 },
		{ 2919, 4489, 0 },
		{ 2089, 4069, 0 },
		{ 2057, 3113, 0 },
		{ 3036, 3394, 0 },
		{ 3036, 3395, 0 },
		{ 2089, 4081, 0 },
		{ 2019, 3843, 0 },
		{ 3033, 3293, 0 },
		{ 2019, 3850, 0 },
		{ 3041, 4276, 0 },
		{ 3036, 3398, 0 },
		{ 2057, 3114, 60 },
		{ 3036, 3402, 0 },
		{ 2919, 4561, 0 },
		{ 2104, 3692, 0 },
		{ 2057, 3115, 0 },
		{ 2057, 3116, 0 },
		{ 2125, 3977, 0 },
		{ 2960, 3934, 0 },
		{ 3041, 4253, 0 },
		{ 3006, 3739, 0 },
		{ 3036, 3405, 0 },
		{ 3036, 3406, 0 },
		{ 1067, 3217, 0 },
		{ 2019, 3824, 0 },
		{ 3007, 3524, 0 },
		{ 2070, 3199, 0 },
		{ 1997, 3259, 0 },
		{ 1997, 3267, 0 },
		{ 3029, 3617, 0 },
		{ 2035, 3434, 0 },
		{ 1175, 3999, 0 },
		{ 3033, 3305, 0 },
		{ 3007, 3538, 0 },
		{ 3041, 4228, 0 },
		{ 3041, 4241, 0 },
		{ 2089, 4043, 0 },
		{ 0, 0, 61 },
		{ 0, 0, 64 },
		{ 2919, 4411, 0 },
		{ 1175, 4005, 0 },
		{ 2019, 3844, 0 },
		{ 3007, 3539, 0 },
		{ 1175, 4010, 0 },
		{ 3007, 3541, 0 },
		{ 0, 0, 109 },
		{ 3036, 3330, 0 },
		{ 3036, 3335, 0 },
		{ 3007, 3547, 0 },
		{ 0, 0, 102 },
		{ 2057, 3120, 0 },
		{ 2919, 4449, 0 },
		{ 0, 0, 110 },
		{ 0, 0, 111 },
		{ 2104, 3691, 0 },
		{ 2019, 3812, 0 },
		{ 3006, 3727, 0 },
		{ 939, 3862, 0 },
		{ 2038, 3790, 0 },
		{ 1175, 4003, 0 },
		{ 3036, 3337, 0 },
		{ 0, 0, 3 },
		{ 2089, 4066, 0 },
		{ 3041, 4292, 0 },
		{ 2919, 4509, 0 },
		{ 3006, 3729, 0 },
		{ 2960, 3909, 0 },
		{ 3036, 3341, 0 },
		{ 2960, 3913, 0 },
		{ 2089, 4075, 0 },
		{ 2057, 3121, 0 },
		{ 2038, 3772, 0 },
		{ 2188, 4225, 0 },
		{ 2035, 3454, 0 },
		{ 2035, 3455, 0 },
		{ 2089, 4084, 0 },
		{ 3006, 3742, 0 },
		{ 1009, 3957, 0 },
		{ 2089, 3059, 0 },
		{ 2960, 3926, 0 },
		{ 2104, 3707, 0 },
		{ 0, 0, 71 },
		{ 2089, 4094, 0 },
		{ 0, 0, 79 },
		{ 2919, 4335, 0 },
		{ 2089, 4095, 0 },
		{ 2919, 4341, 0 },
		{ 3036, 3348, 0 },
		{ 2089, 4099, 0 },
		{ 3033, 3285, 0 },
		{ 3041, 4279, 0 },
		{ 2089, 4101, 0 },
		{ 2104, 3652, 0 },
		{ 2019, 3845, 0 },
		{ 3036, 3351, 0 },
		{ 2019, 3848, 0 },
		{ 2960, 3938, 0 },
		{ 2104, 3661, 0 },
		{ 2960, 3942, 0 },
		{ 2089, 4017, 0 },
		{ 0, 0, 66 },
		{ 2104, 3664, 0 },
		{ 2104, 3665, 0 },
		{ 2919, 4413, 0 },
		{ 2038, 3786, 0 },
		{ 3036, 3352, 0 },
		{ 3006, 3769, 0 },
		{ 2089, 4031, 0 },
		{ 2104, 3668, 0 },
		{ 2089, 4034, 0 },
		{ 2057, 3126, 0 },
		{ 2960, 3878, 0 },
		{ 3029, 3607, 0 },
		{ 2038, 3792, 0 },
		{ 2019, 3817, 0 },
		{ 2035, 3467, 0 },
		{ 3041, 4278, 0 },
		{ 2035, 3476, 0 },
		{ 2057, 3127, 0 },
		{ 2104, 3676, 0 },
		{ 1997, 3252, 0 },
		{ 3041, 4243, 0 },
		{ 2089, 4050, 0 },
		{ 2089, 4052, 0 },
		{ 841, 3237, 0 },
		{ 0, 3232, 0 },
		{ 3006, 3730, 0 },
		{ 2125, 3981, 0 },
		{ 2089, 4057, 0 },
		{ 3007, 3484, 0 },
		{ 1175, 3997, 0 },
		{ 2919, 4557, 0 },
		{ 3007, 3487, 0 },
		{ 2057, 3128, 0 },
		{ 3036, 3360, 0 },
		{ 3007, 3490, 0 },
		{ 2019, 3846, 0 },
		{ 2960, 3906, 0 },
		{ 3007, 3495, 0 },
		{ 3006, 3747, 0 },
		{ 3036, 3361, 0 },
		{ 2188, 4166, 0 },
		{ 2188, 4168, 0 },
		{ 2919, 4337, 0 },
		{ 2089, 4073, 0 },
		{ 0, 0, 65 },
		{ 2019, 3852, 0 },
		{ 3036, 3362, 0 },
		{ 3029, 3640, 0 },
		{ 3007, 3499, 0 },
		{ 3007, 3501, 0 },
		{ 3007, 3503, 0 },
		{ 3036, 3363, 0 },
		{ 2104, 3654, 0 },
		{ 2104, 3657, 0 },
		{ 0, 0, 128 },
		{ 0, 0, 129 },
		{ 2038, 3795, 0 },
		{ 1175, 4000, 0 },
		{ 3036, 3364, 0 },
		{ 2019, 3818, 0 },
		{ 2019, 3819, 0 },
		{ 0, 0, 10 },
		{ 2919, 4377, 0 },
		{ 2035, 3424, 0 },
		{ 3036, 3365, 0 },
		{ 2919, 3991, 0 },
		{ 3041, 4289, 0 },
		{ 3006, 3713, 0 },
		{ 2919, 4419, 0 },
		{ 2919, 4421, 0 },
		{ 3036, 3366, 0 },
		{ 2057, 3129, 0 },
		{ 2960, 3944, 0 },
		{ 2960, 3945, 0 },
		{ 2089, 4107, 0 },
		{ 2057, 3131, 0 },
		{ 3041, 4261, 0 },
		{ 2919, 4447, 0 },
		{ 3041, 4263, 0 },
		{ 2019, 3832, 0 },
		{ 0, 0, 80 },
		{ 2104, 3669, 0 },
		{ 2960, 3951, 0 },
		{ 0, 0, 78 },
		{ 3033, 3302, 0 },
		{ 2038, 3779, 0 },
		{ 0, 0, 81 },
		{ 3041, 4277, 0 },
		{ 2960, 3869, 0 },
		{ 3033, 3303, 0 },
		{ 2089, 4029, 0 },
		{ 2035, 3431, 0 },
		{ 3007, 3522, 0 },
		{ 2089, 4032, 0 },
		{ 2057, 3132, 0 },
		{ 3036, 3373, 0 },
		{ 0, 0, 59 },
		{ 0, 0, 97 },
		{ 0, 0, 99 },
		{ 2104, 3678, 0 },
		{ 2188, 4172, 0 },
		{ 3007, 3526, 0 },
		{ 2089, 4038, 0 },
		{ 2960, 3884, 0 },
		{ 3029, 3618, 0 },
		{ 2089, 4041, 0 },
		{ 0, 0, 134 },
		{ 3007, 3527, 0 },
		{ 2089, 4044, 0 },
		{ 2960, 3890, 0 },
		{ 3036, 3374, 0 },
		{ 3007, 3529, 0 },
		{ 2919, 4574, 0 },
		{ 2057, 3133, 0 },
		{ 2019, 3801, 0 },
		{ 2019, 3806, 0 },
		{ 2089, 4051, 0 },
		{ 2188, 4149, 0 },
		{ 3036, 3376, 0 },
		{ 3036, 3377, 0 },
		{ 3007, 3534, 0 },
		{ 2125, 3967, 0 },
		{ 0, 3864, 0 },
		{ 3036, 3378, 0 },
		{ 3036, 3379, 0 },
		{ 2960, 3907, 0 },
		{ 3029, 3639, 0 },
		{ 2104, 3702, 0 },
		{ 2919, 4365, 0 },
		{ 2960, 3911, 0 },
		{ 3036, 3380, 0 },
		{ 2104, 3708, 0 },
		{ 3041, 4283, 0 },
		{ 0, 0, 19 },
		{ 2035, 3444, 0 },
		{ 2035, 3448, 0 },
		{ 0, 0, 125 },
		{ 2035, 3453, 0 },
		{ 0, 0, 127 },
		{ 3007, 3546, 0 },
		{ 2089, 4072, 0 },
		{ 0, 0, 95 },
		{ 2057, 3135, 0 },
		{ 2019, 3829, 0 },
		{ 2019, 3830, 0 },
		{ 2057, 3137, 0 },
		{ 2919, 4425, 0 },
		{ 2035, 3456, 0 },
		{ 2104, 3663, 0 },
		{ 2960, 3932, 0 },
		{ 0, 0, 76 },
		{ 2019, 3836, 0 },
		{ 1175, 4008, 0 },
		{ 2057, 3138, 0 },
		{ 2019, 3839, 0 },
		{ 3007, 3552, 0 },
		{ 2089, 4089, 0 },
		{ 3041, 4280, 0 },
		{ 3041, 4281, 0 },
		{ 2919, 4485, 0 },
		{ 2089, 4091, 0 },
		{ 2960, 3939, 0 },
		{ 2960, 3940, 0 },
		{ 2057, 3139, 0 },
		{ 2035, 3459, 0 },
		{ 3036, 3385, 0 },
		{ 3006, 3734, 0 },
		{ 3036, 3386, 0 },
		{ 2035, 3464, 0 },
		{ 2960, 3947, 0 },
		{ 3006, 3740, 0 },
		{ 3036, 3387, 0 },
		{ 2089, 4106, 0 },
		{ 0, 0, 84 },
		{ 3041, 4270, 0 },
		{ 0, 0, 101 },
		{ 2019, 3851, 0 },
		{ 3041, 3876, 0 },
		{ 2089, 4109, 0 },
		{ 0, 0, 132 },
		{ 3036, 3388, 0 },
		{ 3036, 3389, 0 },
		{ 2057, 3141, 56 },
		{ 3006, 3748, 0 },
		{ 2104, 3677, 0 },
		{ 2019, 3802, 0 },
		{ 3033, 3289, 0 },
		{ 2854, 3817, 0 },
		{ 0, 0, 85 },
		{ 2035, 3410, 0 },
		{ 3041, 4235, 0 },
		{ 1009, 3959, 0 },
		{ 0, 3960, 0 },
		{ 3036, 3392, 0 },
		{ 3006, 3758, 0 },
		{ 3006, 3759, 0 },
		{ 2104, 3681, 0 },
		{ 3041, 4262, 0 },
		{ 2089, 4037, 0 },
		{ 2960, 3891, 0 },
		{ 2057, 3143, 0 },
		{ 2104, 3685, 0 },
		{ 2104, 3686, 0 },
		{ 2104, 3687, 0 },
		{ 0, 0, 89 },
		{ 2960, 3896, 0 },
		{ 2035, 3413, 0 },
		{ 3007, 3482, 0 },
		{ 0, 0, 122 },
		{ 2057, 3145, 0 },
		{ 3033, 3295, 0 },
		{ 2057, 3146, 0 },
		{ 3029, 3637, 0 },
		{ 2960, 3905, 0 },
		{ 2188, 4221, 0 },
		{ 2035, 3418, 0 },
		{ 3006, 3720, 0 },
		{ 0, 0, 91 },
		{ 3006, 3721, 0 },
		{ 2188, 4137, 0 },
		{ 2188, 4139, 0 },
		{ 3036, 3397, 0 },
		{ 0, 0, 15 },
		{ 2019, 3835, 0 },
		{ 0, 0, 17 },
		{ 0, 0, 18 },
		{ 2960, 3912, 0 },
		{ 2057, 3147, 0 },
		{ 2125, 3968, 0 },
		{ 3006, 3725, 0 },
		{ 2919, 4445, 0 },
		{ 3007, 3491, 0 },
		{ 2104, 3706, 0 },
		{ 1175, 4006, 0 },
		{ 3007, 3493, 0 },
		{ 3007, 3494, 0 },
		{ 3006, 3731, 0 },
		{ 2104, 3709, 0 },
		{ 0, 0, 55 },
		{ 2960, 3927, 0 },
		{ 3036, 3399, 0 },
		{ 2057, 3148, 0 },
		{ 3036, 3401, 0 },
		{ 2019, 3849, 0 },
		{ 2104, 3653, 0 },
		{ 2089, 4079, 0 },
		{ 0, 0, 74 },
		{ 2057, 3149, 0 },
		{ 3041, 4239, 0 },
		{ 3007, 3500, 0 },
		{ 0, 3223, 0 },
		{ 3007, 3502, 0 },
		{ 0, 0, 16 },
		{ 2104, 3662, 0 },
		{ 1175, 4001, 0 },
		{ 2057, 3150, 54 },
		{ 2057, 3151, 0 },
		{ 2019, 3803, 0 },
		{ 0, 0, 75 },
		{ 3006, 3752, 0 },
		{ 3041, 3259, 0 },
		{ 0, 0, 82 },
		{ 0, 0, 83 },
		{ 0, 0, 52 },
		{ 3006, 3756, 0 },
		{ 3029, 3612, 0 },
		{ 3029, 3613, 0 },
		{ 3036, 3407, 0 },
		{ 3029, 3615, 0 },
		{ 0, 0, 133 },
		{ 3006, 3761, 0 },
		{ 1175, 4013, 0 },
		{ 1175, 4014, 0 },
		{ 3036, 3316, 0 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 2057, 3152, 40 },
		{ 3006, 3764, 0 },
		{ 3041, 4291, 0 },
		{ 1175, 3992, 0 },
		{ 1175, 3993, 0 },
		{ 2019, 3820, 0 },
		{ 0, 0, 72 },
		{ 3006, 3765, 0 },
		{ 3036, 3320, 0 },
		{ 0, 0, 90 },
		{ 3036, 3321, 0 },
		{ 2019, 3825, 0 },
		{ 3029, 3621, 0 },
		{ 2019, 3827, 0 },
		{ 2035, 3436, 0 },
		{ 2960, 3885, 0 },
		{ 3033, 3306, 0 },
		{ 2960, 3888, 0 },
		{ 2125, 3965, 0 },
		{ 2057, 3153, 0 },
		{ 3036, 3324, 0 },
		{ 3041, 4272, 0 },
		{ 3033, 3309, 0 },
		{ 0, 0, 86 },
		{ 3041, 4275, 0 },
		{ 2057, 3154, 0 },
		{ 0, 0, 126 },
		{ 0, 0, 130 },
		{ 2019, 3834, 0 },
		{ 0, 0, 135 },
		{ 0, 0, 11 },
		{ 3006, 3718, 0 },
		{ 3006, 3719, 0 },
		{ 2104, 3684, 0 },
		{ 3029, 3632, 0 },
		{ 1175, 3988, 0 },
		{ 2057, 3155, 0 },
		{ 3036, 3333, 0 },
		{ 3006, 3724, 0 },
		{ 3036, 3334, 0 },
		{ 3041, 4234, 0 },
		{ 0, 0, 131 },
		{ 2960, 3903, 0 },
		{ 3041, 4236, 0 },
		{ 3006, 3726, 0 },
		{ 3033, 3282, 0 },
		{ 3033, 3284, 0 },
		{ 3041, 4245, 0 },
		{ 2057, 3156, 0 },
		{ 3041, 4251, 0 },
		{ 2960, 3908, 0 },
		{ 2919, 4505, 0 },
		{ 3036, 3338, 0 },
		{ 3036, 3340, 0 },
		{ 2057, 3157, 0 },
		{ 0, 0, 41 },
		{ 3006, 3733, 0 },
		{ 2919, 4517, 0 },
		{ 3041, 4264, 0 },
		{ 3036, 3342, 0 },
		{ 2104, 3697, 0 },
		{ 2019, 3855, 0 },
		{ 2960, 3916, 0 },
		{ 2960, 3917, 0 },
		{ 2854, 4298, 0 },
		{ 3041, 4273, 0 },
		{ 2019, 3856, 0 },
		{ 2919, 4565, 0 },
		{ 2960, 3921, 0 },
		{ 3006, 3736, 0 },
		{ 2019, 3858, 0 },
		{ 2104, 3698, 0 },
		{ 2104, 3699, 0 },
		{ 2089, 4070, 0 },
		{ 3036, 3343, 0 },
		{ 2019, 3804, 0 },
		{ 2019, 3805, 0 },
		{ 2104, 3701, 0 },
		{ 0, 0, 77 },
		{ 0, 0, 92 },
		{ 3006, 3743, 0 },
		{ 3006, 3744, 0 },
		{ 0, 3995, 0 },
		{ 2960, 3933, 0 },
		{ 0, 0, 87 },
		{ 2019, 3809, 0 },
		{ 3006, 3745, 0 },
		{ 2035, 3461, 0 },
		{ 0, 0, 12 },
		{ 2104, 3703, 0 },
		{ 2104, 3704, 0 },
		{ 0, 0, 88 },
		{ 0, 0, 53 },
		{ 0, 0, 93 },
		{ 3007, 3542, 0 },
		{ 3006, 3750, 0 },
		{ 2089, 4090, 0 },
		{ 0, 0, 14 },
		{ 2057, 3158, 0 },
		{ 3007, 3544, 0 },
		{ 2089, 4093, 0 },
		{ 3029, 3605, 0 },
		{ 2019, 3821, 0 },
		{ 2919, 4382, 0 },
		{ 3041, 4265, 0 },
		{ 2089, 4096, 0 },
		{ 2038, 3793, 0 },
		{ 2089, 4098, 0 },
		{ 3006, 3755, 0 },
		{ 3036, 3345, 0 },
		{ 0, 0, 13 },
		{ 3085, 1435, 228 },
		{ 0, 0, 229 },
		{ 3040, 4892, 230 },
		{ 3068, 1642, 234 },
		{ 1212, 2593, 235 },
		{ 0, 0, 235 },
		{ 3068, 1698, 231 },
		{ 1215, 1451, 0 },
		{ 3068, 1709, 232 },
		{ 1218, 1517, 0 },
		{ 1215, 0, 0 },
		{ 2994, 2693, 233 },
		{ 1220, 1436, 0 },
		{ 1218, 0, 0 },
		{ 2994, 2568, 231 },
		{ 1220, 0, 0 },
		{ 2994, 2578, 232 },
		{ 3033, 3280, 144 },
		{ 0, 0, 144 },
		{ 0, 0, 145 },
		{ 3046, 1991, 0 },
		{ 3068, 2855, 0 },
		{ 3085, 2062, 0 },
		{ 1228, 4751, 0 },
		{ 3065, 2569, 0 },
		{ 3068, 2783, 0 },
		{ 3080, 2925, 0 },
		{ 3076, 2454, 0 },
		{ 3079, 3002, 0 },
		{ 3085, 2102, 0 },
		{ 3079, 2970, 0 },
		{ 3081, 1743, 0 },
		{ 2986, 2678, 0 },
		{ 3083, 2182, 0 },
		{ 3037, 2235, 0 },
		{ 3046, 1975, 0 },
		{ 3086, 4597, 0 },
		{ 0, 0, 142 },
		{ 2854, 4297, 152 },
		{ 0, 0, 152 },
		{ 0, 0, 153 },
		{ 3068, 2819, 0 },
		{ 2932, 2757, 0 },
		{ 3083, 2188, 0 },
		{ 3085, 2039, 0 },
		{ 3068, 2808, 0 },
		{ 1251, 4692, 0 },
		{ 3068, 2497, 0 },
		{ 3050, 1491, 0 },
		{ 3068, 2890, 0 },
		{ 3085, 2066, 0 },
		{ 2698, 1457, 0 },
		{ 3081, 1920, 0 },
		{ 3075, 2710, 0 },
		{ 2986, 2703, 0 },
		{ 3037, 2297, 0 },
		{ 2888, 2723, 0 },
		{ 1262, 4761, 0 },
		{ 3068, 2501, 0 },
		{ 3076, 2435, 0 },
		{ 3046, 1968, 0 },
		{ 3068, 2837, 0 },
		{ 1267, 4755, 0 },
		{ 3078, 2444, 0 },
		{ 3073, 1617, 0 },
		{ 3037, 2290, 0 },
		{ 3080, 2945, 0 },
		{ 3081, 1863, 0 },
		{ 2986, 2629, 0 },
		{ 3083, 2158, 0 },
		{ 3037, 2250, 0 },
		{ 3086, 4373, 0 },
		{ 0, 0, 150 },
		{ 3033, 3311, 176 },
		{ 0, 0, 176 },
		{ 3046, 1998, 0 },
		{ 3068, 2889, 0 },
		{ 3085, 2055, 0 },
		{ 1283, 4690, 0 },
		{ 3080, 2629, 0 },
		{ 3076, 2466, 0 },
		{ 3079, 3021, 0 },
		{ 3046, 2011, 0 },
		{ 3046, 2028, 0 },
		{ 3068, 2834, 0 },
		{ 3046, 2029, 0 },
		{ 3086, 4457, 0 },
		{ 0, 0, 175 },
		{ 2125, 3964, 140 },
		{ 0, 0, 140 },
		{ 0, 0, 141 },
		{ 3068, 2852, 0 },
		{ 3037, 2252, 0 },
		{ 3083, 2197, 0 },
		{ 3070, 2364, 0 },
		{ 3068, 2780, 0 },
		{ 3041, 4285, 0 },
		{ 3076, 2409, 0 },
		{ 3079, 2996, 0 },
		{ 3046, 1959, 0 },
		{ 3046, 1965, 0 },
		{ 3048, 4636, 0 },
		{ 3048, 4628, 0 },
		{ 2986, 2668, 0 },
		{ 3037, 2304, 0 },
		{ 2986, 2695, 0 },
		{ 3081, 1866, 0 },
		{ 2986, 2548, 0 },
		{ 3079, 2994, 0 },
		{ 3076, 2424, 0 },
		{ 2986, 2656, 0 },
		{ 3046, 1525, 0 },
		{ 3068, 2781, 0 },
		{ 3085, 2076, 0 },
		{ 3086, 4439, 0 },
		{ 0, 0, 138 },
		{ 2604, 2949, 23 },
		{ 0, 0, 23 },
		{ 0, 0, 24 },
		{ 3068, 2798, 0 },
		{ 2888, 2725, 0 },
		{ 2986, 2686, 0 },
		{ 3037, 2277, 0 },
		{ 3040, 9, 0 },
		{ 3083, 2177, 0 },
		{ 2846, 2124, 0 },
		{ 3068, 2844, 0 },
		{ 3085, 2085, 0 },
		{ 3079, 2971, 0 },
		{ 3081, 1934, 0 },
		{ 3083, 2150, 0 },
		{ 3085, 2034, 0 },
		{ 3040, 7, 0 },
		{ 3065, 2902, 0 },
		{ 3068, 2777, 0 },
		{ 3046, 1997, 0 },
		{ 3080, 2947, 0 },
		{ 3085, 2041, 0 },
		{ 2986, 2699, 0 },
		{ 2846, 2118, 0 },
		{ 3081, 1687, 0 },
		{ 2986, 2559, 0 },
		{ 3083, 2218, 0 },
		{ 3037, 2303, 0 },
		{ 3040, 4871, 0 },
		{ 3048, 4640, 0 },
		{ 0, 0, 20 },
		{ 1366, 0, 1 },
		{ 1366, 0, 177 },
		{ 1366, 2743, 227 },
		{ 1581, 174, 227 },
		{ 1581, 416, 227 },
		{ 1581, 404, 227 },
		{ 1581, 524, 227 },
		{ 1581, 405, 227 },
		{ 1581, 416, 227 },
		{ 1581, 391, 227 },
		{ 1581, 422, 227 },
		{ 1581, 482, 227 },
		{ 1366, 0, 227 },
		{ 1378, 2489, 227 },
		{ 1366, 2771, 227 },
		{ 2604, 2952, 223 },
		{ 1581, 503, 227 },
		{ 1581, 501, 227 },
		{ 1581, 516, 227 },
		{ 1581, 0, 227 },
		{ 1581, 551, 227 },
		{ 1581, 550, 227 },
		{ 3083, 2161, 0 },
		{ 0, 0, 178 },
		{ 3037, 2299, 0 },
		{ 1581, 517, 0 },
		{ 1581, 0, 0 },
		{ 3040, 3927, 0 },
		{ 1581, 537, 0 },
		{ 1581, 553, 0 },
		{ 1581, 552, 0 },
		{ 1581, 585, 0 },
		{ 1581, 577, 0 },
		{ 1581, 580, 0 },
		{ 1581, 588, 0 },
		{ 1581, 596, 0 },
		{ 1581, 589, 0 },
		{ 1581, 582, 0 },
		{ 1581, 586, 0 },
		{ 3068, 2868, 0 },
		{ 3068, 2879, 0 },
		{ 1582, 592, 0 },
		{ 1582, 595, 0 },
		{ 1581, 606, 0 },
		{ 1581, 618, 0 },
		{ 1581, 609, 0 },
		{ 3083, 2185, 0 },
		{ 3065, 2899, 0 },
		{ 1581, 607, 0 },
		{ 1581, 651, 0 },
		{ 1581, 628, 0 },
		{ 1581, 629, 0 },
		{ 1581, 652, 0 },
		{ 1581, 679, 0 },
		{ 1581, 684, 0 },
		{ 1581, 678, 0 },
		{ 1581, 654, 0 },
		{ 1581, 645, 0 },
		{ 1581, 642, 0 },
		{ 1581, 693, 0 },
		{ 1581, 680, 0 },
		{ 3037, 2328, 0 },
		{ 2932, 2746, 0 },
		{ 1581, 28, 0 },
		{ 1581, 30, 0 },
		{ 1582, 32, 0 },
		{ 1581, 28, 0 },
		{ 1581, 32, 0 },
		{ 3076, 2441, 0 },
		{ 0, 0, 226 },
		{ 1581, 44, 0 },
		{ 1581, 26, 0 },
		{ 1581, 14, 0 },
		{ 1581, 66, 0 },
		{ 1581, 70, 0 },
		{ 1581, 74, 0 },
		{ 1581, 70, 0 },
		{ 1581, 59, 0 },
		{ 1581, 38, 0 },
		{ 1581, 43, 0 },
		{ 1581, 60, 0 },
		{ 1581, 0, 212 },
		{ 1581, 109, 0 },
		{ 3083, 2157, 0 },
		{ 2986, 2490, 0 },
		{ 1581, 66, 0 },
		{ 1581, 71, 0 },
		{ 1581, 91, 0 },
		{ 1581, 94, 0 },
		{ 1581, 95, 0 },
		{ -1461, 1092, 0 },
		{ 1582, 132, 0 },
		{ 1581, 178, 0 },
		{ 1581, 184, 0 },
		{ 1581, 176, 0 },
		{ 1581, 186, 0 },
		{ 1581, 187, 0 },
		{ 1581, 166, 0 },
		{ 1581, 186, 0 },
		{ 1581, 163, 0 },
		{ 1581, 154, 0 },
		{ 1581, 0, 211 },
		{ 1581, 161, 0 },
		{ 3070, 2379, 0 },
		{ 3037, 2285, 0 },
		{ 1581, 165, 0 },
		{ 1581, 175, 0 },
		{ 1581, 183, 0 },
		{ 1581, 0, 225 },
		{ 1581, 181, 0 },
		{ 0, 0, 213 },
		{ 1581, 174, 0 },
		{ 1583, 33, -4 },
		{ 1581, 203, 0 },
		{ 1581, 215, 0 },
		{ 1581, 273, 0 },
		{ 1581, 279, 0 },
		{ 1581, 241, 0 },
		{ 1581, 255, 0 },
		{ 1581, 227, 0 },
		{ 1581, 273, 0 },
		{ 1581, 265, 0 },
		{ 3068, 2823, 0 },
		{ 3068, 2830, 0 },
		{ 1581, 0, 215 },
		{ 1581, 304, 216 },
		{ 1581, 273, 0 },
		{ 1581, 276, 0 },
		{ 1581, 303, 0 },
		{ 1481, 3483, 0 },
		{ 3040, 4264, 0 },
		{ 2166, 4603, 202 },
		{ 1581, 306, 0 },
		{ 1581, 316, 0 },
		{ 1581, 315, 0 },
		{ 1581, 319, 0 },
		{ 1581, 321, 0 },
		{ 1581, 335, 0 },
		{ 1581, 323, 0 },
		{ 1581, 325, 0 },
		{ 1581, 334, 0 },
		{ 1581, 343, 0 },
		{ 1582, 330, 0 },
		{ 3041, 4286, 0 },
		{ 3040, 4, 218 },
		{ 1581, 366, 0 },
		{ 1581, 378, 0 },
		{ 1581, 360, 0 },
		{ 1581, 376, 0 },
		{ 0, 0, 182 },
		{ 1583, 117, -7 },
		{ 1583, 231, -10 },
		{ 1583, 345, -13 },
		{ 1583, 376, -16 },
		{ 1583, 460, -19 },
		{ 1583, 488, -22 },
		{ 1581, 408, 0 },
		{ 1581, 421, 0 },
		{ 1581, 394, 0 },
		{ 1581, 0, 200 },
		{ 1581, 0, 214 },
		{ 3076, 2412, 0 },
		{ 1581, 394, 0 },
		{ 1581, 388, 0 },
		{ 1581, 397, 0 },
		{ 1582, 407, 0 },
		{ 1518, 3520, 0 },
		{ 3040, 4176, 0 },
		{ 2166, 4616, 203 },
		{ 1521, 3521, 0 },
		{ 3040, 4260, 0 },
		{ 2166, 4599, 204 },
		{ 1524, 3523, 0 },
		{ 3040, 4290, 0 },
		{ 2166, 4618, 207 },
		{ 1527, 3524, 0 },
		{ 3040, 4184, 0 },
		{ 2166, 4614, 208 },
		{ 1530, 3525, 0 },
		{ 3040, 4258, 0 },
		{ 2166, 4620, 209 },
		{ 1533, 3526, 0 },
		{ 3040, 4262, 0 },
		{ 2166, 4609, 210 },
		{ 1581, 452, 0 },
		{ 1583, 490, -25 },
		{ 1581, 457, 0 },
		{ 3079, 2985, 0 },
		{ 1581, 439, 0 },
		{ 1581, 519, 0 },
		{ 1581, 480, 0 },
		{ 1581, 491, 0 },
		{ 0, 0, 184 },
		{ 0, 0, 186 },
		{ 0, 0, 192 },
		{ 0, 0, 194 },
		{ 0, 0, 196 },
		{ 0, 0, 198 },
		{ 1583, 574, -28 },
		{ 1551, 3536, 0 },
		{ 3040, 4174, 0 },
		{ 2166, 4591, 206 },
		{ 1581, 0, 199 },
		{ 3046, 2030, 0 },
		{ 1581, 479, 0 },
		{ 1581, 494, 0 },
		{ 1582, 488, 0 },
		{ 1581, 485, 0 },
		{ 1560, 3542, 0 },
		{ 3040, 4266, 0 },
		{ 2166, 4597, 205 },
		{ 0, 0, 190 },
		{ 3046, 1983, 0 },
		{ 1581, 4, 221 },
		{ 1582, 491, 0 },
		{ 1581, 1, 224 },
		{ 1581, 506, 0 },
		{ 0, 0, 188 },
		{ 3048, 4638, 0 },
		{ 3048, 4639, 0 },
		{ 1581, 494, 0 },
		{ 0, 0, 222 },
		{ 1581, 491, 0 },
		{ 3048, 4629, 0 },
		{ 0, 0, 220 },
		{ 1581, 502, 0 },
		{ 1581, 507, 0 },
		{ 0, 0, 219 },
		{ 1581, 512, 0 },
		{ 1581, 503, 0 },
		{ 1582, 505, 217 },
		{ 1583, 925, 0 },
		{ 1584, 736, -1 },
		{ 1585, 3496, 0 },
		{ 3040, 4250, 0 },
		{ 2166, 4587, 201 },
		{ 0, 0, 180 },
		{ 2125, 3966, 270 },
		{ 0, 0, 270 },
		{ 3068, 2884, 0 },
		{ 3037, 2329, 0 },
		{ 3083, 2215, 0 },
		{ 3070, 2355, 0 },
		{ 3068, 2778, 0 },
		{ 3041, 4288, 0 },
		{ 3076, 2416, 0 },
		{ 3079, 3008, 0 },
		{ 3046, 1994, 0 },
		{ 3046, 1995, 0 },
		{ 3048, 4643, 0 },
		{ 3048, 4644, 0 },
		{ 2986, 2700, 0 },
		{ 3037, 2273, 0 },
		{ 2986, 2704, 0 },
		{ 3081, 1928, 0 },
		{ 2986, 2545, 0 },
		{ 3079, 3003, 0 },
		{ 3076, 2462, 0 },
		{ 2986, 2551, 0 },
		{ 3046, 1530, 0 },
		{ 3068, 2842, 0 },
		{ 3085, 2094, 0 },
		{ 3086, 4455, 0 },
		{ 0, 0, 269 },
		{ 2125, 3983, 272 },
		{ 0, 0, 272 },
		{ 0, 0, 273 },
		{ 3068, 2846, 0 },
		{ 3037, 2288, 0 },
		{ 3083, 2163, 0 },
		{ 3070, 2383, 0 },
		{ 3068, 2872, 0 },
		{ 3041, 4269, 0 },
		{ 3076, 2429, 0 },
		{ 3079, 2966, 0 },
		{ 3046, 2005, 0 },
		{ 3046, 2006, 0 },
		{ 3048, 4645, 0 },
		{ 3048, 4648, 0 },
		{ 3080, 2934, 0 },
		{ 3085, 2038, 0 },
		{ 3083, 2186, 0 },
		{ 3046, 2007, 0 },
		{ 3046, 2009, 0 },
		{ 3083, 2204, 0 },
		{ 3050, 1493, 0 },
		{ 3068, 2794, 0 },
		{ 3085, 2059, 0 },
		{ 3086, 4459, 0 },
		{ 0, 0, 271 },
		{ 2125, 3973, 275 },
		{ 0, 0, 275 },
		{ 0, 0, 276 },
		{ 3068, 2800, 0 },
		{ 3037, 2261, 0 },
		{ 3083, 2147, 0 },
		{ 3070, 2380, 0 },
		{ 3068, 2820, 0 },
		{ 3041, 4230, 0 },
		{ 3076, 2451, 0 },
		{ 3079, 3006, 0 },
		{ 3046, 2018, 0 },
		{ 3046, 2023, 0 },
		{ 3048, 4632, 0 },
		{ 3048, 4634, 0 },
		{ 3070, 2358, 0 },
		{ 3073, 1592, 0 },
		{ 3081, 1781, 0 },
		{ 3079, 2981, 0 },
		{ 3081, 1811, 0 },
		{ 3083, 2168, 0 },
		{ 3085, 2097, 0 },
		{ 3086, 4601, 0 },
		{ 0, 0, 274 },
		{ 2125, 3978, 278 },
		{ 0, 0, 278 },
		{ 0, 0, 279 },
		{ 3068, 2857, 0 },
		{ 3037, 2301, 0 },
		{ 3083, 2179, 0 },
		{ 3070, 2366, 0 },
		{ 3068, 2883, 0 },
		{ 3041, 4268, 0 },
		{ 3076, 2453, 0 },
		{ 3079, 2968, 0 },
		{ 3046, 2031, 0 },
		{ 3046, 2032, 0 },
		{ 3048, 4649, 0 },
		{ 3048, 4627, 0 },
		{ 3068, 2776, 0 },
		{ 3050, 1461, 0 },
		{ 3079, 2992, 0 },
		{ 3076, 2408, 0 },
		{ 3073, 1550, 0 },
		{ 3079, 3000, 0 },
		{ 3081, 1912, 0 },
		{ 3083, 2203, 0 },
		{ 3085, 2050, 0 },
		{ 3086, 4371, 0 },
		{ 0, 0, 277 },
		{ 2125, 3963, 281 },
		{ 0, 0, 281 },
		{ 0, 0, 282 },
		{ 3068, 2797, 0 },
		{ 3037, 2262, 0 },
		{ 3083, 2212, 0 },
		{ 3070, 2382, 0 },
		{ 3068, 2805, 0 },
		{ 3041, 4294, 0 },
		{ 3076, 2450, 0 },
		{ 3079, 2980, 0 },
		{ 3046, 1972, 0 },
		{ 3046, 1973, 0 },
		{ 3048, 4641, 0 },
		{ 3048, 4642, 0 },
		{ 3083, 2143, 0 },
		{ 2846, 2138, 0 },
		{ 3081, 1913, 0 },
		{ 2986, 2553, 0 },
		{ 3070, 2368, 0 },
		{ 2986, 2626, 0 },
		{ 3046, 1980, 0 },
		{ 3068, 2851, 0 },
		{ 3085, 2075, 0 },
		{ 3086, 4531, 0 },
		{ 0, 0, 280 },
		{ 2919, 4493, 156 },
		{ 0, 0, 156 },
		{ 0, 0, 157 },
		{ 2932, 2763, 0 },
		{ 3081, 1916, 0 },
		{ 3068, 2866, 0 },
		{ 3085, 2078, 0 },
		{ 1725, 4746, 0 },
		{ 3068, 2491, 0 },
		{ 3050, 1490, 0 },
		{ 3068, 2881, 0 },
		{ 3085, 2089, 0 },
		{ 2698, 1440, 0 },
		{ 3081, 1929, 0 },
		{ 3075, 2720, 0 },
		{ 2986, 2701, 0 },
		{ 3037, 2244, 0 },
		{ 2888, 2726, 0 },
		{ 1736, 4755, 0 },
		{ 3068, 2503, 0 },
		{ 3076, 2458, 0 },
		{ 3046, 1996, 0 },
		{ 3068, 2791, 0 },
		{ 1741, 4688, 0 },
		{ 3078, 2442, 0 },
		{ 3073, 1627, 0 },
		{ 3037, 2260, 0 },
		{ 3080, 2937, 0 },
		{ 3081, 1942, 0 },
		{ 2986, 2567, 0 },
		{ 3083, 2193, 0 },
		{ 3037, 2269, 0 },
		{ 3086, 4593, 0 },
		{ 0, 0, 154 },
		{ 2125, 3982, 263 },
		{ 0, 0, 263 },
		{ 3068, 2811, 0 },
		{ 3037, 2270, 0 },
		{ 3083, 2195, 0 },
		{ 3070, 2374, 0 },
		{ 3068, 2828, 0 },
		{ 3041, 4238, 0 },
		{ 3076, 2449, 0 },
		{ 3079, 3023, 0 },
		{ 3046, 2001, 0 },
		{ 3046, 2003, 0 },
		{ 3048, 4646, 0 },
		{ 3048, 4647, 0 },
		{ 3065, 2913, 0 },
		{ 2986, 2698, 0 },
		{ 3046, 2004, 0 },
		{ 2846, 2116, 0 },
		{ 3076, 2456, 0 },
		{ 3079, 2988, 0 },
		{ 2698, 1454, 0 },
		{ 3086, 4556, 0 },
		{ 0, 0, 261 },
		{ 1789, 0, 1 },
		{ 1948, 2808, 378 },
		{ 3068, 2861, 378 },
		{ 3079, 2914, 378 },
		{ 3065, 2152, 378 },
		{ 1789, 0, 345 },
		{ 1789, 2618, 378 },
		{ 3075, 1596, 378 },
		{ 2854, 4306, 378 },
		{ 2104, 3655, 378 },
		{ 3033, 3287, 378 },
		{ 2104, 3659, 378 },
		{ 2089, 4082, 378 },
		{ 3085, 1955, 378 },
		{ 1789, 0, 378 },
		{ 2604, 2950, 376 },
		{ 3079, 2741, 378 },
		{ 3079, 2976, 378 },
		{ 0, 0, 378 },
		{ 3083, 2156, 0 },
		{ -1794, 4903, 335 },
		{ -1795, 4720, 0 },
		{ 3037, 2314, 0 },
		{ 0, 0, 341 },
		{ 0, 0, 342 },
		{ 3076, 2418, 0 },
		{ 2986, 2563, 0 },
		{ 3068, 2892, 0 },
		{ 0, 0, 346 },
		{ 3037, 2321, 0 },
		{ 3085, 2067, 0 },
		{ 2986, 2627, 0 },
		{ 2057, 3071, 0 },
		{ 3029, 3619, 0 },
		{ 3036, 3331, 0 },
		{ 1997, 3270, 0 },
		{ 3029, 3622, 0 },
		{ 3073, 1624, 0 },
		{ 3046, 2017, 0 },
		{ 3037, 2240, 0 },
		{ 3081, 1838, 0 },
		{ 3085, 2083, 0 },
		{ 3083, 2170, 0 },
		{ 2760, 4702, 0 },
		{ 3083, 2173, 0 },
		{ 3046, 2019, 0 },
		{ 3081, 1860, 0 },
		{ 3037, 2267, 0 },
		{ 3065, 2918, 0 },
		{ 3085, 2090, 0 },
		{ 3076, 2406, 0 },
		{ 2125, 3970, 0 },
		{ 2057, 3087, 0 },
		{ 2057, 3088, 0 },
		{ 2089, 4028, 0 },
		{ 2019, 3838, 0 },
		{ 3068, 2815, 0 },
		{ 3046, 2026, 0 },
		{ 3065, 2915, 0 },
		{ 3073, 1625, 0 },
		{ 3068, 2821, 0 },
		{ 3076, 2410, 0 },
		{ 0, 4897, 338 },
		{ 3070, 2362, 0 },
		{ 3068, 2829, 0 },
		{ 2104, 3705, 0 },
		{ 3081, 1864, 0 },
		{ 0, 0, 377 },
		{ 3068, 2831, 0 },
		{ 3065, 2919, 0 },
		{ 2089, 4042, 0 },
		{ 2035, 3441, 0 },
		{ 3029, 3611, 0 },
		{ 3007, 3504, 0 },
		{ 2057, 3101, 0 },
		{ 0, 0, 366 },
		{ 3041, 4290, 0 },
		{ 3083, 2192, 0 },
		{ 3085, 2104, 0 },
		{ 3037, 2287, 0 },
		{ -1871, 1167, 0 },
		{ 0, 0, 337 },
		{ 3068, 2845, 0 },
		{ 0, 0, 365 },
		{ 2846, 2123, 0 },
		{ 2986, 2618, 0 },
		{ 3037, 2292, 0 },
		{ 1886, 4665, 0 },
		{ 3006, 3737, 0 },
		{ 2960, 3902, 0 },
		{ 3007, 3520, 0 },
		{ 2057, 3111, 0 },
		{ 3029, 3624, 0 },
		{ 3083, 2201, 0 },
		{ 3070, 2384, 0 },
		{ 3037, 2298, 0 },
		{ 3081, 1871, 0 },
		{ 0, 0, 367 },
		{ 3041, 4259, 344 },
		{ 3081, 1892, 0 },
		{ 3080, 2942, 0 },
		{ 3081, 1903, 0 },
		{ 0, 0, 370 },
		{ 0, 0, 371 },
		{ 1891, 0, -71 },
		{ 2070, 3202, 0 },
		{ 2104, 3675, 0 },
		{ 3029, 3636, 0 },
		{ 2089, 4071, 0 },
		{ 2986, 2679, 0 },
		{ 0, 0, 369 },
		{ 0, 0, 375 },
		{ 0, 4663, 0 },
		{ 3076, 2460, 0 },
		{ 3046, 1960, 0 },
		{ 3079, 3015, 0 },
		{ 2125, 3984, 0 },
		{ 3040, 4220, 0 },
		{ 2166, 4622, 360 },
		{ 2089, 4078, 0 },
		{ 2854, 4296, 0 },
		{ 3007, 3536, 0 },
		{ 3007, 3537, 0 },
		{ 3037, 2308, 0 },
		{ 0, 0, 372 },
		{ 0, 0, 373 },
		{ 3079, 3022, 0 },
		{ 2172, 4706, 0 },
		{ 3076, 2467, 0 },
		{ 3068, 2886, 0 },
		{ 0, 0, 350 },
		{ 1914, 0, -74 },
		{ 1916, 0, -77 },
		{ 2104, 3688, 0 },
		{ 3041, 4287, 0 },
		{ 0, 0, 368 },
		{ 3046, 1962, 0 },
		{ 0, 0, 343 },
		{ 2125, 3971, 0 },
		{ 3037, 2319, 0 },
		{ 3040, 4180, 0 },
		{ 2166, 4607, 361 },
		{ 3040, 4182, 0 },
		{ 2166, 4612, 362 },
		{ 2854, 4308, 0 },
		{ 1926, 0, -59 },
		{ 3046, 1963, 0 },
		{ 3068, 2893, 0 },
		{ 3068, 2775, 0 },
		{ 0, 0, 352 },
		{ 0, 0, 354 },
		{ 1931, 0, -65 },
		{ 3040, 4246, 0 },
		{ 2166, 4595, 364 },
		{ 0, 0, 340 },
		{ 3037, 2323, 0 },
		{ 3085, 2061, 0 },
		{ 3040, 4252, 0 },
		{ 2166, 4605, 363 },
		{ 0, 0, 358 },
		{ 3083, 2149, 0 },
		{ 3079, 2990, 0 },
		{ 0, 0, 356 },
		{ 3070, 2354, 0 },
		{ 3081, 1906, 0 },
		{ 3068, 2782, 0 },
		{ 2986, 2550, 0 },
		{ 0, 0, 374 },
		{ 3083, 2154, 0 },
		{ 3037, 2242, 0 },
		{ 1945, 0, -80 },
		{ 3040, 4172, 0 },
		{ 2166, 4593, 359 },
		{ 0, 0, 348 },
		{ 1789, 2798, 378 },
		{ 1952, 2491, 378 },
		{ -1950, 17, 335 },
		{ -1951, 4718, 0 },
		{ 3040, 4707, 0 },
		{ 2760, 4677, 0 },
		{ 0, 0, 336 },
		{ 2760, 4703, 0 },
		{ -1956, 9, 0 },
		{ -1957, 4724, 0 },
		{ 1960, 2, 338 },
		{ 2760, 4691, 0 },
		{ 3040, 4884, 0 },
		{ 0, 0, 339 },
		{ 1978, 0, 1 },
		{ 2174, 2815, 334 },
		{ 3068, 2807, 334 },
		{ 1978, 0, 288 },
		{ 1978, 2688, 334 },
		{ 3029, 3627, 334 },
		{ 1978, 0, 291 },
		{ 3073, 1600, 334 },
		{ 2854, 4300, 334 },
		{ 2104, 3656, 334 },
		{ 3033, 3248, 334 },
		{ 2104, 3658, 334 },
		{ 2089, 4027, 334 },
		{ 3079, 2977, 334 },
		{ 3085, 1962, 334 },
		{ 1978, 0, 334 },
		{ 2604, 2957, 331 },
		{ 3079, 2986, 334 },
		{ 3065, 2901, 334 },
		{ 2854, 4302, 334 },
		{ 3079, 1544, 334 },
		{ 0, 0, 334 },
		{ 3083, 2166, 0 },
		{ -1985, 4905, 283 },
		{ -1986, 4725, 0 },
		{ 3037, 2264, 0 },
		{ 0, 0, 289 },
		{ 3037, 2266, 0 },
		{ 3083, 2167, 0 },
		{ 3085, 2081, 0 },
		{ 2057, 3067, 0 },
		{ 3029, 3648, 0 },
		{ 3036, 3346, 0 },
		{ 3006, 3753, 0 },
		{ 0, 3227, 0 },
		{ 0, 3248, 0 },
		{ 3029, 3598, 0 },
		{ 3076, 2465, 0 },
		{ 3073, 1619, 0 },
		{ 3046, 1985, 0 },
		{ 3037, 2276, 0 },
		{ 3068, 2838, 0 },
		{ 3068, 2840, 0 },
		{ 3083, 2174, 0 },
		{ 2172, 1, 0 },
		{ 3083, 2175, 0 },
		{ 2760, 4696, 0 },
		{ 3083, 2176, 0 },
		{ 3065, 2900, 0 },
		{ 2846, 2126, 0 },
		{ 3085, 2088, 0 },
		{ 2125, 3972, 0 },
		{ 2057, 3080, 0 },
		{ 2057, 3081, 0 },
		{ 2960, 3918, 0 },
		{ 2960, 3919, 0 },
		{ 2089, 4064, 0 },
		{ 0, 3841, 0 },
		{ 3046, 1986, 0 },
		{ 3068, 2854, 0 },
		{ 3046, 1987, 0 },
		{ 3065, 2917, 0 },
		{ 3037, 2295, 0 },
		{ 3046, 1989, 0 },
		{ 3076, 2437, 0 },
		{ 0, 0, 333 },
		{ 3076, 2439, 0 },
		{ 0, 0, 285 },
		{ 3070, 2386, 0 },
		{ 0, 0, 330 },
		{ 3073, 1622, 0 },
		{ 3068, 2876, 0 },
		{ 2089, 4077, 0 },
		{ 0, 3416, 0 },
		{ 3029, 3631, 0 },
		{ 2038, 3796, 0 },
		{ 0, 3797, 0 },
		{ 3007, 3540, 0 },
		{ 2057, 3093, 0 },
		{ 3068, 2878, 0 },
		{ 0, 0, 323 },
		{ 3041, 4293, 0 },
		{ 3083, 2189, 0 },
		{ 3081, 1930, 0 },
		{ 3081, 1933, 0 },
		{ 3073, 1623, 0 },
		{ -2065, 1242, 0 },
		{ 3068, 2888, 0 },
		{ 3076, 2457, 0 },
		{ 3037, 2310, 0 },
		{ 3006, 3738, 0 },
		{ 2960, 3950, 0 },
		{ 3007, 3551, 0 },
		{ 2960, 3952, 0 },
		{ 2960, 3953, 0 },
		{ 0, 3103, 0 },
		{ 3029, 3647, 0 },
		{ 0, 0, 322 },
		{ 3083, 2198, 0 },
		{ 3070, 2376, 0 },
		{ 2986, 2670, 0 },
		{ 0, 0, 329 },
		{ 3079, 2972, 0 },
		{ 0, 0, 324 },
		{ 0, 0, 287 },
		{ 3079, 2974, 0 },
		{ 3081, 1679, 0 },
		{ 2082, 0, -56 },
		{ 0, 3201, 0 },
		{ 2104, 3667, 0 },
		{ 2073, 3197, 0 },
		{ 2070, 3198, 0 },
		{ 3029, 3608, 0 },
		{ 2089, 4111, 0 },
		{ 2986, 2673, 0 },
		{ 0, 0, 326 },
		{ 3080, 2927, 0 },
		{ 3081, 1680, 0 },
		{ 3081, 1681, 0 },
		{ 2125, 3962, 0 },
		{ 3040, 4254, 0 },
		{ 2166, 4589, 313 },
		{ 2089, 4021, 0 },
		{ 2854, 4301, 0 },
		{ 2089, 4022, 0 },
		{ 2089, 4023, 0 },
		{ 2089, 4024, 0 },
		{ 0, 4025, 0 },
		{ 3007, 3479, 0 },
		{ 3007, 3480, 0 },
		{ 3037, 2326, 0 },
		{ 3079, 2987, 0 },
		{ 2986, 2687, 0 },
		{ 2986, 2688, 0 },
		{ 3068, 2787, 0 },
		{ 0, 0, 295 },
		{ 2111, 0, -35 },
		{ 2113, 0, -38 },
		{ 2115, 0, -44 },
		{ 2117, 0, -47 },
		{ 2119, 0, -50 },
		{ 2121, 0, -53 },
		{ 0, 3682, 0 },
		{ 3041, 4237, 0 },
		{ 0, 0, 325 },
		{ 3076, 2407, 0 },
		{ 3083, 2205, 0 },
		{ 3083, 2211, 0 },
		{ 3037, 2331, 0 },
		{ 3040, 4186, 0 },
		{ 2166, 4610, 314 },
		{ 3040, 4188, 0 },
		{ 2166, 4613, 315 },
		{ 3040, 4190, 0 },
		{ 2166, 4615, 318 },
		{ 3040, 4192, 0 },
		{ 2166, 4617, 319 },
		{ 3040, 4216, 0 },
		{ 2166, 4619, 320 },
		{ 3040, 4218, 0 },
		{ 2166, 4621, 321 },
		{ 2854, 4307, 0 },
		{ 2136, 0, -62 },
		{ 0, 3975, 0 },
		{ 3037, 2232, 0 },
		{ 3037, 2233, 0 },
		{ 3068, 2803, 0 },
		{ 0, 0, 297 },
		{ 0, 0, 299 },
		{ 0, 0, 305 },
		{ 0, 0, 307 },
		{ 0, 0, 309 },
		{ 0, 0, 311 },
		{ 2142, 0, -68 },
		{ 3040, 4248, 0 },
		{ 2166, 4602, 317 },
		{ 3068, 2804, 0 },
		{ 3079, 3019, 0 },
		{ 3015, 3191, 328 },
		{ 3085, 2051, 0 },
		{ 3040, 4256, 0 },
		{ 2166, 4611, 316 },
		{ 0, 0, 303 },
		{ 3037, 2237, 0 },
		{ 3085, 2052, 0 },
		{ 0, 0, 290 },
		{ 3079, 3027, 0 },
		{ 0, 0, 301 },
		{ 3083, 2216, 0 },
		{ 2698, 1443, 0 },
		{ 3081, 1728, 0 },
		{ 3070, 2377, 0 },
		{ 2919, 4435, 0 },
		{ 2986, 2552, 0 },
		{ 3068, 2822, 0 },
		{ 3076, 2443, 0 },
		{ 3083, 2144, 0 },
		{ 0, 0, 327 },
		{ 2888, 2732, 0 },
		{ 3037, 2257, 0 },
		{ 3083, 2146, 0 },
		{ 2165, 0, -41 },
		{ 3085, 2060, 0 },
		{ 3040, 4178, 0 },
		{ 0, 4601, 312 },
		{ 2986, 2616, 0 },
		{ 0, 0, 293 },
		{ 3081, 1739, 0 },
		{ 3075, 2711, 0 },
		{ 3070, 2385, 0 },
		{ 0, 4707, 0 },
		{ 0, 0, 332 },
		{ 1978, 2828, 334 },
		{ 2178, 2488, 334 },
		{ -2176, 4904, 283 },
		{ -2177, 4723, 0 },
		{ 3040, 4708, 0 },
		{ 2760, 4676, 0 },
		{ 0, 0, 284 },
		{ 2760, 4694, 0 },
		{ -2182, 21, 0 },
		{ -2183, 4717, 0 },
		{ 2186, 0, 285 },
		{ 2760, 4695, 0 },
		{ 3040, 4840, 0 },
		{ 0, 0, 286 },
		{ 0, 4153, 380 },
		{ 0, 0, 380 },
		{ 3068, 2847, 0 },
		{ 2932, 2737, 0 },
		{ 3079, 3004, 0 },
		{ 3073, 1534, 0 },
		{ 3076, 2459, 0 },
		{ 3081, 1796, 0 },
		{ 3040, 2, 0 },
		{ 3085, 2074, 0 },
		{ 3073, 1546, 0 },
		{ 3037, 2272, 0 },
		{ 2201, 4685, 0 },
		{ 3040, 1943, 0 },
		{ 3079, 3025, 0 },
		{ 3085, 2077, 0 },
		{ 3079, 3028, 0 },
		{ 3070, 2372, 0 },
		{ 3068, 2875, 0 },
		{ 3081, 1845, 0 },
		{ 3068, 2877, 0 },
		{ 3085, 2079, 0 },
		{ 3046, 2022, 0 },
		{ 3086, 4451, 0 },
		{ 0, 0, 379 },
		{ 2760, 4688, 427 },
		{ 0, 0, 385 },
		{ 0, 0, 387 },
		{ 2232, 828, 418 },
		{ 2397, 841, 418 },
		{ 2418, 839, 418 },
		{ 2365, 841, 418 },
		{ 2233, 857, 418 },
		{ 2231, 834, 418 },
		{ 2418, 838, 418 },
		{ 2254, 853, 418 },
		{ 2393, 855, 418 },
		{ 2393, 859, 418 },
		{ 2397, 856, 418 },
		{ 2341, 865, 418 },
		{ 3068, 1620, 417 },
		{ 2261, 2637, 427 },
		{ 2451, 854, 418 },
		{ 2397, 867, 418 },
		{ 2264, 867, 418 },
		{ 2397, 861, 418 },
		{ 3068, 2793, 427 },
		{ -2235, 18, 381 },
		{ -2236, 4716, 0 },
		{ 2451, 858, 418 },
		{ 2456, 463, 418 },
		{ 2451, 860, 418 },
		{ 2310, 858, 418 },
		{ 2397, 870, 418 },
		{ 2402, 865, 418 },
		{ 2397, 872, 418 },
		{ 2341, 881, 418 },
		{ 2314, 871, 418 },
		{ 2365, 862, 418 },
		{ 2341, 884, 418 },
		{ 2418, 868, 418 },
		{ 2230, 861, 418 },
		{ 2367, 868, 418 },
		{ 2230, 865, 418 },
		{ 2430, 887, 418 },
		{ 2402, 887, 418 },
		{ 2230, 897, 418 },
		{ 2430, 890, 418 },
		{ 2451, 1265, 418 },
		{ 2430, 891, 418 },
		{ 2314, 894, 418 },
		{ 3068, 1774, 414 },
		{ 2289, 1483, 0 },
		{ 3068, 1797, 415 },
		{ 3037, 2279, 0 },
		{ 2760, 4704, 0 },
		{ 2230, 905, 418 },
		{ 3083, 1949, 0 },
		{ 2393, 930, 418 },
		{ 2300, 915, 418 },
		{ 2430, 923, 418 },
		{ 2367, 946, 418 },
		{ 2367, 947, 418 },
		{ 2314, 956, 418 },
		{ 2393, 974, 418 },
		{ 2393, 975, 418 },
		{ 2418, 963, 418 },
		{ 2365, 960, 418 },
		{ 2393, 978, 418 },
		{ 2341, 983, 418 },
		{ 2456, 469, 418 },
		{ 2456, 471, 418 },
		{ 2425, 993, 418 },
		{ 2425, 994, 418 },
		{ 2393, 1009, 418 },
		{ 2300, 994, 418 },
		{ 2402, 1002, 418 },
		{ 2341, 1053, 418 },
		{ 2382, 1051, 418 },
		{ 2317, 1403, 0 },
		{ 2289, 0, 0 },
		{ 2994, 2558, 416 },
		{ 2319, 1404, 0 },
		{ 3065, 2904, 0 },
		{ 0, 0, 383 },
		{ 2393, 1051, 418 },
		{ 2932, 2766, 0 },
		{ 2456, 484, 418 },
		{ 2314, 1046, 418 },
		{ 2367, 1039, 418 },
		{ 2456, 578, 418 },
		{ 2397, 1083, 418 },
		{ 2230, 1066, 418 },
		{ 2312, 1086, 418 },
		{ 2456, 8, 418 },
		{ 2367, 1109, 418 },
		{ 2397, 1121, 418 },
		{ 2456, 123, 418 },
		{ 2367, 1111, 418 },
		{ 3081, 1826, 0 },
		{ 3040, 2212, 0 },
		{ 2425, 1113, 418 },
		{ 2230, 1117, 418 },
		{ 2418, 1116, 418 },
		{ 2230, 1132, 418 },
		{ 2367, 1142, 418 },
		{ 2230, 1151, 418 },
		{ 2230, 1141, 418 },
		{ 2317, 0, 0 },
		{ 2994, 2592, 414 },
		{ 2319, 0, 0 },
		{ 2994, 2626, 415 },
		{ 0, 0, 419 },
		{ 2418, 1147, 418 },
		{ 2347, 4828, 0 },
		{ 3076, 2062, 0 },
		{ 2341, 1166, 418 },
		{ 2456, 125, 418 },
		{ 3046, 1884, 0 },
		{ 2479, 6, 418 },
		{ 2425, 1186, 418 },
		{ 2341, 1205, 418 },
		{ 2367, 1187, 418 },
		{ 3040, 1930, 0 },
		{ 2456, 128, 418 },
		{ 2365, 1186, 418 },
		{ 3083, 1988, 0 },
		{ 2397, 1200, 418 },
		{ 3037, 2313, 0 },
		{ 3085, 2049, 0 },
		{ 3037, 2315, 0 },
		{ 2402, 1195, 418 },
		{ 2418, 1219, 418 },
		{ 2230, 1237, 418 },
		{ 2393, 1234, 418 },
		{ 2393, 1235, 418 },
		{ 2456, 130, 418 },
		{ 2397, 1261, 418 },
		{ 2456, 235, 418 },
		{ 3078, 2168, 0 },
		{ 2986, 2631, 0 },
		{ 2367, 1252, 418 },
		{ 3046, 1890, 0 },
		{ 3081, 1691, 0 },
		{ 3086, 4449, 0 },
		{ 3040, 4870, 391 },
		{ 2451, 1260, 418 },
		{ 2367, 1254, 418 },
		{ 2397, 1270, 418 },
		{ 3083, 2164, 0 },
		{ 3078, 2448, 0 },
		{ 2397, 1267, 418 },
		{ 2932, 2761, 0 },
		{ 2402, 1262, 418 },
		{ 2986, 2690, 0 },
		{ 3068, 2771, 0 },
		{ 2986, 2692, 0 },
		{ 2230, 1258, 418 },
		{ 2397, 1272, 418 },
		{ 2230, 1262, 418 },
		{ 2456, 237, 418 },
		{ 2456, 239, 418 },
		{ 3085, 1911, 0 },
		{ 2430, 1270, 418 },
		{ 3083, 1959, 0 },
		{ 3029, 3635, 0 },
		{ 2986, 2492, 0 },
		{ 3070, 2356, 0 },
		{ 2397, 1276, 418 },
		{ 3081, 1901, 0 },
		{ 3079, 2961, 0 },
		{ 2479, 121, 418 },
		{ 2402, 1271, 418 },
		{ 2402, 1272, 418 },
		{ 2230, 1284, 418 },
		{ 2846, 2117, 0 },
		{ 3085, 2047, 0 },
		{ 2430, 1275, 418 },
		{ 2413, 4763, 0 },
		{ 2430, 1276, 418 },
		{ 3081, 1923, 0 },
		{ 3068, 2812, 0 },
		{ 3081, 1925, 0 },
		{ 2393, 1286, 418 },
		{ 2430, 1278, 418 },
		{ 2230, 1288, 418 },
		{ 3083, 1761, 0 },
		{ 3040, 2216, 0 },
		{ 3068, 2826, 0 },
		{ 2230, 1285, 418 },
		{ 2932, 2765, 0 },
		{ 3033, 3308, 0 },
		{ 3081, 1936, 0 },
		{ 2986, 2671, 0 },
		{ 2230, 1280, 418 },
		{ 3079, 3005, 0 },
		{ 3081, 1947, 0 },
		{ 3086, 4377, 0 },
		{ 0, 0, 409 },
		{ 2418, 1278, 418 },
		{ 2430, 1283, 418 },
		{ 2456, 241, 418 },
		{ 3073, 1655, 0 },
		{ 3083, 2153, 0 },
		{ 2419, 1292, 418 },
		{ 3040, 1935, 0 },
		{ 2456, 243, 418 },
		{ 2441, 4702, 0 },
		{ 2442, 4732, 0 },
		{ 2443, 4724, 0 },
		{ 2230, 1283, 418 },
		{ 2230, 1295, 418 },
		{ 2456, 245, 418 },
		{ 3079, 2967, 0 },
		{ 2932, 2741, 0 },
		{ 3046, 2010, 0 },
		{ 3065, 2898, 0 },
		{ 2230, 1285, 418 },
		{ 2452, 4762, 0 },
		{ 3046, 2012, 0 },
		{ 3037, 2243, 0 },
		{ 3081, 1813, 0 },
		{ 2230, 1291, 418 },
		{ 3081, 1840, 0 },
		{ 3046, 2020, 0 },
		{ 2456, 349, 418 },
		{ 2456, 351, 418 },
		{ 3040, 2329, 0 },
		{ 3076, 2431, 0 },
		{ 3070, 2370, 0 },
		{ 2456, 355, 418 },
		{ 3085, 2046, 0 },
		{ 3040, 1941, 0 },
		{ 3081, 1830, 0 },
		{ 3065, 2571, 0 },
		{ 3081, 1878, 0 },
		{ 2456, 357, 418 },
		{ 2456, 370, 418 },
		{ 3080, 1928, 0 },
		{ 3085, 2058, 0 },
		{ 2932, 2739, 0 },
		{ 3076, 2455, 0 },
		{ 3073, 1599, 0 },
		{ 2456, 1549, 418 },
		{ 3083, 1882, 0 },
		{ 2482, 4840, 0 },
		{ 3068, 2784, 0 },
		{ 3086, 4441, 0 },
		{ 2479, 576, 418 },
		{ 3046, 1966, 0 },
		{ 3086, 4453, 0 },
		{ 3040, 2331, 0 },
		{ 3083, 1980, 0 },
		{ 3068, 2796, 0 },
		{ 3079, 2969, 0 },
		{ 2492, 4744, 0 },
		{ 3083, 1759, 0 },
		{ 3083, 2207, 0 },
		{ 3085, 2068, 0 },
		{ 3085, 2071, 0 },
		{ 3068, 2802, 0 },
		{ 3085, 2073, 0 },
		{ 3040, 1939, 0 },
		{ 3046, 1886, 0 },
		{ 3046, 1977, 0 },
		{ 3037, 2309, 0 },
		{ 2504, 4761, 0 },
		{ 3068, 2810, 0 },
		{ 3046, 1978, 0 },
		{ 3079, 2989, 0 },
		{ 3080, 2932, 0 },
		{ 2509, 811, 418 },
		{ 3068, 2813, 0 },
		{ 2846, 2121, 0 },
		{ 3086, 4585, 0 },
		{ 3046, 1981, 0 },
		{ 3040, 4789, 389 },
		{ 3046, 1892, 0 },
		{ 3086, 4599, 0 },
		{ 3040, 4816, 398 },
		{ 3083, 2152, 0 },
		{ 2846, 2125, 0 },
		{ 3037, 2325, 0 },
		{ 3081, 1932, 0 },
		{ 3078, 2438, 0 },
		{ 3079, 3011, 0 },
		{ 2932, 2744, 0 },
		{ 2888, 2727, 0 },
		{ 3083, 2155, 0 },
		{ 3068, 2833, 0 },
		{ 2846, 2137, 0 },
		{ 3068, 2836, 0 },
		{ 3085, 2084, 0 },
		{ 2986, 2565, 0 },
		{ 3050, 1462, 0 },
		{ 3073, 1536, 0 },
		{ 3046, 1894, 0 },
		{ 3037, 2236, 0 },
		{ 2846, 2120, 0 },
		{ 3037, 2238, 0 },
		{ 3068, 2848, 0 },
		{ 3086, 4375, 0 },
		{ 3040, 4785, 412 },
		{ 3037, 2239, 0 },
		{ 3081, 1940, 0 },
		{ 0, 0, 424 },
		{ 3046, 1992, 0 },
		{ 2986, 2664, 0 },
		{ 3040, 4810, 397 },
		{ 3079, 2978, 0 },
		{ 3068, 2856, 0 },
		{ 2986, 2665, 0 },
		{ 2986, 2666, 0 },
		{ 2986, 2667, 0 },
		{ 3085, 2096, 0 },
		{ 2932, 2762, 0 },
		{ 2549, 4755, 0 },
		{ 2604, 2951, 0 },
		{ 3068, 2871, 0 },
		{ 3081, 1941, 0 },
		{ 3068, 2873, 0 },
		{ 3083, 2171, 0 },
		{ 2541, 1376, 0 },
		{ 2556, 4756, 0 },
		{ 2846, 2129, 0 },
		{ 3080, 2923, 0 },
		{ 3081, 1946, 0 },
		{ 3085, 2105, 0 },
		{ 3065, 2903, 0 },
		{ 2562, 4780, 0 },
		{ 3068, 2880, 0 },
		{ 2986, 2680, 0 },
		{ 2565, 4669, 0 },
		{ 0, 1409, 0 },
		{ 3076, 2414, 0 },
		{ 3085, 2037, 0 },
		{ 3081, 1678, 0 },
		{ 3083, 2184, 0 },
		{ 3076, 2426, 0 },
		{ 3068, 2891, 0 },
		{ 3046, 1999, 0 },
		{ 3040, 2758, 0 },
		{ 3079, 2965, 0 },
		{ 2576, 4731, 0 },
		{ 3075, 2709, 0 },
		{ 2578, 4735, 0 },
		{ 2604, 2954, 0 },
		{ 3068, 2773, 0 },
		{ 3046, 1896, 0 },
		{ 3076, 2433, 0 },
		{ 3085, 2044, 0 },
		{ 3046, 2002, 0 },
		{ 2986, 2486, 0 },
		{ 2586, 4746, 0 },
		{ 3083, 1968, 0 },
		{ 3085, 2048, 0 },
		{ 3070, 2378, 0 },
		{ 3080, 2708, 0 },
		{ 3068, 2785, 0 },
		{ 3086, 4515, 0 },
		{ 3079, 2982, 0 },
		{ 3083, 2194, 0 },
		{ 3037, 2281, 0 },
		{ 3068, 2792, 0 },
		{ 3037, 2282, 0 },
		{ 2846, 2128, 0 },
		{ 3073, 1602, 0 },
		{ 2604, 2955, 0 },
		{ 3065, 2573, 0 },
		{ 2602, 4712, 0 },
		{ 3065, 2567, 0 },
		{ 3079, 2997, 0 },
		{ 3086, 4379, 0 },
		{ 3081, 1686, 0 },
		{ 3083, 2199, 0 },
		{ 2986, 2615, 0 },
		{ 2609, 4697, 0 },
		{ 3037, 2289, 0 },
		{ 3070, 2006, 0 },
		{ 2846, 2115, 0 },
		{ 3079, 3007, 0 },
		{ 2986, 2620, 0 },
		{ 3079, 3010, 0 },
		{ 3086, 4517, 0 },
		{ 3040, 4827, 410 },
		{ 3081, 1688, 0 },
		{ 3085, 2053, 0 },
		{ 3086, 4558, 0 },
		{ 3086, 4583, 0 },
		{ 3081, 1689, 0 },
		{ 3085, 2057, 0 },
		{ 2932, 2758, 0 },
		{ 2986, 2632, 0 },
		{ 3068, 2817, 0 },
		{ 3086, 4367, 0 },
		{ 3068, 2818, 0 },
		{ 0, 2956, 0 },
		{ 3040, 4858, 396 },
		{ 3079, 2960, 0 },
		{ 3081, 1690, 0 },
		{ 2846, 2122, 0 },
		{ 3083, 1974, 0 },
		{ 2888, 2730, 0 },
		{ 3083, 2217, 0 },
		{ 3068, 2824, 0 },
		{ 3081, 1692, 0 },
		{ 3046, 2013, 0 },
		{ 3046, 2014, 0 },
		{ 3040, 4901, 390 },
		{ 3083, 2145, 0 },
		{ 3046, 2015, 0 },
		{ 3040, 4764, 402 },
		{ 3040, 4775, 403 },
		{ 3046, 2016, 0 },
		{ 2986, 2675, 0 },
		{ 2932, 2749, 0 },
		{ 3076, 2420, 0 },
		{ 2846, 2133, 0 },
		{ 0, 0, 423 },
		{ 2846, 2135, 0 },
		{ 2986, 2682, 0 },
		{ 3081, 1693, 0 },
		{ 2649, 4701, 0 },
		{ 3081, 1718, 0 },
		{ 2846, 2139, 0 },
		{ 2652, 4713, 0 },
		{ 3065, 2912, 0 },
		{ 3085, 2069, 0 },
		{ 2986, 2691, 0 },
		{ 3079, 2995, 0 },
		{ 3068, 2850, 0 },
		{ 3085, 2070, 0 },
		{ 3086, 4412, 0 },
		{ 3086, 4414, 0 },
		{ 3037, 2332, 0 },
		{ 3068, 2853, 0 },
		{ 2986, 2696, 0 },
		{ 3081, 1722, 0 },
		{ 3081, 1726, 0 },
		{ 3076, 2445, 0 },
		{ 3046, 2021, 0 },
		{ 3046, 1898, 0 },
		{ 3086, 4513, 0 },
		{ 3068, 2864, 0 },
		{ 3083, 1986, 0 },
		{ 3068, 2867, 0 },
		{ 3079, 3018, 0 },
		{ 3083, 2165, 0 },
		{ 3081, 1740, 0 },
		{ 3046, 2027, 0 },
		{ 3086, 4587, 0 },
		{ 3085, 939, 393 },
		{ 3040, 4886, 405 },
		{ 2888, 2731, 0 },
		{ 3085, 2080, 0 },
		{ 3081, 1753, 0 },
		{ 3075, 2716, 0 },
		{ 3075, 2719, 0 },
		{ 2986, 2549, 0 },
		{ 2684, 4674, 0 },
		{ 3080, 2940, 0 },
		{ 3040, 4778, 401 },
		{ 3085, 2082, 0 },
		{ 2846, 2130, 0 },
		{ 3076, 2461, 0 },
		{ 3081, 1773, 0 },
		{ 3037, 2258, 0 },
		{ 2986, 2561, 0 },
		{ 2692, 4745, 0 },
		{ 3040, 4796, 392 },
		{ 3086, 4447, 0 },
		{ 2694, 4751, 0 },
		{ 2698, 1455, 0 },
		{ 2696, 4755, 0 },
		{ 2697, 4757, 0 },
		{ 3081, 1793, 0 },
		{ 3078, 2446, 0 },
		{ 3085, 2087, 0 },
		{ 3079, 2979, 0 },
		{ 3068, 2895, 0 },
		{ 3086, 4486, 0 },
		{ 3083, 2180, 0 },
		{ 3046, 1953, 0 },
		{ 3083, 2183, 0 },
		{ 3086, 4519, 0 },
		{ 3040, 4834, 407 },
		{ 3086, 4521, 0 },
		{ 3086, 4523, 0 },
		{ 3086, 4527, 0 },
		{ 3086, 4529, 0 },
		{ 0, 1456, 0 },
		{ 2986, 2622, 0 },
		{ 2986, 2625, 0 },
		{ 3081, 1799, 0 },
		{ 3085, 2091, 0 },
		{ 3040, 4851, 413 },
		{ 3085, 2092, 0 },
		{ 3086, 4591, 0 },
		{ 3037, 2275, 0 },
		{ 0, 0, 426 },
		{ 0, 0, 425 },
		{ 3040, 4860, 394 },
		{ 0, 0, 421 },
		{ 0, 0, 422 },
		{ 3086, 4595, 0 },
		{ 3076, 2422, 0 },
		{ 2846, 2119, 0 },
		{ 3083, 2191, 0 },
		{ 3079, 2999, 0 },
		{ 3086, 4369, 0 },
		{ 3040, 4872, 388 },
		{ 2726, 4780, 0 },
		{ 3040, 4877, 395 },
		{ 3068, 2789, 0 },
		{ 3081, 1801, 0 },
		{ 3085, 2095, 0 },
		{ 3081, 1803, 0 },
		{ 3040, 4896, 408 },
		{ 3040, 2214, 0 },
		{ 3086, 4381, 0 },
		{ 3086, 4383, 0 },
		{ 3086, 4385, 0 },
		{ 3083, 2196, 0 },
		{ 3081, 1807, 0 },
		{ 3040, 4768, 399 },
		{ 3040, 4770, 400 },
		{ 3040, 4772, 404 },
		{ 3085, 2098, 0 },
		{ 3068, 2799, 0 },
		{ 3086, 4443, 0 },
		{ 3085, 2100, 0 },
		{ 3040, 4782, 406 },
		{ 3079, 3013, 0 },
		{ 3081, 1809, 0 },
		{ 2986, 2676, 0 },
		{ 3083, 2202, 0 },
		{ 3037, 2294, 0 },
		{ 3046, 1967, 0 },
		{ 3086, 4484, 0 },
		{ 3040, 4802, 411 },
		{ 2760, 4678, 427 },
		{ 2753, 0, 385 },
		{ 0, 0, 386 },
		{ -2751, 20, 381 },
		{ -2752, 4722, 0 },
		{ 3040, 4699, 0 },
		{ 2760, 4687, 0 },
		{ 0, 0, 382 },
		{ 2760, 4681, 0 },
		{ -2757, 4902, 0 },
		{ -2758, 4714, 0 },
		{ 2761, 0, 383 },
		{ 0, 4689, 0 },
		{ 3040, 4820, 0 },
		{ 0, 0, 384 },
		{ 3033, 3281, 148 },
		{ 0, 0, 148 },
		{ 0, 0, 149 },
		{ 3046, 1971, 0 },
		{ 3068, 2809, 0 },
		{ 3085, 2036, 0 },
		{ 2770, 4739, 0 },
		{ 3078, 2436, 0 },
		{ 3073, 1639, 0 },
		{ 3037, 2302, 0 },
		{ 3080, 2946, 0 },
		{ 3081, 1820, 0 },
		{ 2986, 2693, 0 },
		{ 3083, 2214, 0 },
		{ 3037, 2306, 0 },
		{ 3046, 1974, 0 },
		{ 3086, 4589, 0 },
		{ 0, 0, 146 },
		{ 2919, 4497, 171 },
		{ 0, 0, 171 },
		{ 3081, 1825, 0 },
		{ 2785, 4763, 0 },
		{ 3068, 2495, 0 },
		{ 3079, 2975, 0 },
		{ 3080, 2935, 0 },
		{ 3075, 2706, 0 },
		{ 2790, 4771, 0 },
		{ 3040, 2325, 0 },
		{ 3068, 2825, 0 },
		{ 3037, 2312, 0 },
		{ 3068, 2827, 0 },
		{ 3085, 2043, 0 },
		{ 3079, 2984, 0 },
		{ 3081, 1831, 0 },
		{ 2986, 2488, 0 },
		{ 3083, 2220, 0 },
		{ 3037, 2317, 0 },
		{ 2801, 4799, 0 },
		{ 3040, 2756, 0 },
		{ 3068, 2835, 0 },
		{ 2932, 2756, 0 },
		{ 3083, 2142, 0 },
		{ 3085, 2045, 0 },
		{ 3068, 2839, 0 },
		{ 2808, 4658, 0 },
		{ 3085, 1923, 0 },
		{ 3068, 2841, 0 },
		{ 3065, 2905, 0 },
		{ 3073, 1657, 0 },
		{ 3080, 2924, 0 },
		{ 3068, 2843, 0 },
		{ 2815, 4684, 0 },
		{ 3078, 2434, 0 },
		{ 3073, 1532, 0 },
		{ 3037, 2327, 0 },
		{ 3080, 2933, 0 },
		{ 3081, 1856, 0 },
		{ 2986, 2560, 0 },
		{ 3083, 2148, 0 },
		{ 3037, 2330, 0 },
		{ 3086, 4525, 0 },
		{ 0, 0, 169 },
		{ 2826, 0, 1 },
		{ -2826, 1372, 260 },
		{ 3068, 2752, 266 },
		{ 0, 0, 266 },
		{ 3046, 1988, 0 },
		{ 3037, 2234, 0 },
		{ 3068, 2858, 0 },
		{ 3065, 2910, 0 },
		{ 3085, 2054, 0 },
		{ 0, 0, 265 },
		{ 2836, 4745, 0 },
		{ 3070, 1925, 0 },
		{ 3079, 2963, 0 },
		{ 2865, 2477, 0 },
		{ 3068, 2865, 0 },
		{ 2932, 2759, 0 },
		{ 2986, 2624, 0 },
		{ 3076, 2447, 0 },
		{ 3068, 2869, 0 },
		{ 2845, 4728, 0 },
		{ 3083, 1994, 0 },
		{ 0, 2131, 0 },
		{ 3081, 1876, 0 },
		{ 2986, 2630, 0 },
		{ 3083, 2160, 0 },
		{ 3037, 2241, 0 },
		{ 3046, 1993, 0 },
		{ 3086, 4387, 0 },
		{ 0, 0, 264 },
		{ 0, 4309, 174 },
		{ 0, 0, 174 },
		{ 3083, 2162, 0 },
		{ 3073, 1598, 0 },
		{ 3037, 2246, 0 },
		{ 3065, 2914, 0 },
		{ 2861, 4764, 0 },
		{ 3080, 2627, 0 },
		{ 3075, 2708, 0 },
		{ 3068, 2885, 0 },
		{ 3080, 2941, 0 },
		{ 0, 2478, 0 },
		{ 2986, 2669, 0 },
		{ 3037, 2248, 0 },
		{ 2888, 2722, 0 },
		{ 3086, 4511, 0 },
		{ 0, 0, 172 },
		{ 2919, 4511, 168 },
		{ 0, 0, 167 },
		{ 0, 0, 168 },
		{ 3081, 1898, 0 },
		{ 2876, 4768, 0 },
		{ 3081, 1854, 0 },
		{ 3075, 2717, 0 },
		{ 3068, 2894, 0 },
		{ 2880, 4791, 0 },
		{ 3040, 2754, 0 },
		{ 3068, 2896, 0 },
		{ 2888, 2728, 0 },
		{ 2986, 2674, 0 },
		{ 3037, 2254, 0 },
		{ 3037, 2255, 0 },
		{ 2986, 2677, 0 },
		{ 3037, 2256, 0 },
		{ 0, 2724, 0 },
		{ 2890, 4660, 0 },
		{ 3083, 1954, 0 },
		{ 2932, 2745, 0 },
		{ 2893, 4674, 0 },
		{ 3068, 2493, 0 },
		{ 3079, 3014, 0 },
		{ 3080, 2922, 0 },
		{ 3075, 2715, 0 },
		{ 2898, 4680, 0 },
		{ 3040, 2323, 0 },
		{ 3068, 2786, 0 },
		{ 3037, 2259, 0 },
		{ 3068, 2788, 0 },
		{ 3085, 2065, 0 },
		{ 3079, 3026, 0 },
		{ 3081, 1905, 0 },
		{ 2986, 2689, 0 },
		{ 3083, 2169, 0 },
		{ 3037, 2263, 0 },
		{ 2909, 4699, 0 },
		{ 3078, 2452, 0 },
		{ 3073, 1601, 0 },
		{ 3037, 2265, 0 },
		{ 3080, 2943, 0 },
		{ 3081, 1907, 0 },
		{ 2986, 2697, 0 },
		{ 3083, 2172, 0 },
		{ 3037, 2268, 0 },
		{ 3086, 4445, 0 },
		{ 0, 0, 161 },
		{ 0, 4375, 160 },
		{ 0, 0, 160 },
		{ 3081, 1908, 0 },
		{ 2923, 4707, 0 },
		{ 3081, 1858, 0 },
		{ 3075, 2707, 0 },
		{ 3068, 2806, 0 },
		{ 2927, 4726, 0 },
		{ 3068, 2499, 0 },
		{ 3037, 2271, 0 },
		{ 3065, 2906, 0 },
		{ 2931, 4722, 0 },
		{ 3083, 1956, 0 },
		{ 0, 2747, 0 },
		{ 2934, 4737, 0 },
		{ 3068, 2489, 0 },
		{ 3079, 2983, 0 },
		{ 3080, 2938, 0 },
		{ 3075, 2712, 0 },
		{ 2939, 4742, 0 },
		{ 3040, 2327, 0 },
		{ 3068, 2814, 0 },
		{ 3037, 2274, 0 },
		{ 3068, 2816, 0 },
		{ 3085, 2072, 0 },
		{ 3079, 2991, 0 },
		{ 3081, 1914, 0 },
		{ 2986, 2546, 0 },
		{ 3083, 2178, 0 },
		{ 3037, 2278, 0 },
		{ 2950, 4756, 0 },
		{ 3078, 2450, 0 },
		{ 3073, 1618, 0 },
		{ 3037, 2280, 0 },
		{ 3080, 2928, 0 },
		{ 3081, 1917, 0 },
		{ 2986, 2555, 0 },
		{ 3083, 2181, 0 },
		{ 3037, 2283, 0 },
		{ 3086, 4603, 0 },
		{ 0, 0, 158 },
		{ 0, 3937, 163 },
		{ 0, 0, 163 },
		{ 0, 0, 164 },
		{ 3037, 2284, 0 },
		{ 3046, 2008, 0 },
		{ 3081, 1919, 0 },
		{ 3068, 2832, 0 },
		{ 3079, 3012, 0 },
		{ 3065, 2916, 0 },
		{ 2970, 4787, 0 },
		{ 3068, 2487, 0 },
		{ 3050, 1492, 0 },
		{ 3079, 3017, 0 },
		{ 3076, 2463, 0 },
		{ 3073, 1620, 0 },
		{ 3079, 3020, 0 },
		{ 3081, 1924, 0 },
		{ 2986, 2619, 0 },
		{ 3083, 2187, 0 },
		{ 3037, 2291, 0 },
		{ 2981, 4667, 0 },
		{ 3078, 2440, 0 },
		{ 3073, 1621, 0 },
		{ 3037, 2293, 0 },
		{ 3080, 2930, 0 },
		{ 3081, 1927, 0 },
		{ 0, 2628, 0 },
		{ 3083, 2190, 0 },
		{ 3037, 2296, 0 },
		{ 3048, 4626, 0 },
		{ 0, 0, 162 },
		{ 3068, 2849, 427 },
		{ 3085, 1517, 25 },
		{ 3000, 0, 427 },
		{ 2229, 2648, 27 },
		{ 0, 0, 28 },
		{ 0, 0, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 3037, 2300, 0 },
		{ 3085, 661, 0 },
		{ 0, 0, 26 },
		{ 3065, 2920, 0 },
		{ 0, 0, 21 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 0, 3766, 37 },
		{ 0, 3498, 37 },
		{ 0, 0, 36 },
		{ 0, 0, 37 },
		{ 3029, 3638, 0 },
		{ 3041, 4282, 0 },
		{ 3033, 3283, 0 },
		{ 0, 0, 35 },
		{ 3036, 3350, 0 },
		{ 0, 3193, 0 },
		{ 2994, 1630, 0 },
		{ 0, 0, 34 },
		{ 3068, 2765, 47 },
		{ 0, 0, 47 },
		{ 3033, 3288, 47 },
		{ 3068, 2859, 47 },
		{ 0, 0, 50 },
		{ 3068, 2860, 0 },
		{ 3037, 2305, 0 },
		{ 3036, 3359, 0 },
		{ 3081, 1938, 0 },
		{ 3037, 2307, 0 },
		{ 3065, 2908, 0 },
		{ 0, 3603, 0 },
		{ 3073, 1656, 0 },
		{ 3083, 2200, 0 },
		{ 0, 0, 46 },
		{ 0, 3300, 0 },
		{ 3085, 2093, 0 },
		{ 3070, 2360, 0 },
		{ 0, 3368, 0 },
		{ 0, 2311, 0 },
		{ 3068, 2870, 0 },
		{ 0, 0, 48 },
		{ 0, 5, 51 },
		{ 0, 4249, 0 },
		{ 0, 0, 49 },
		{ 3076, 2448, 0 },
		{ 3079, 2993, 0 },
		{ 3046, 2024, 0 },
		{ 0, 2025, 0 },
		{ 3048, 4630, 0 },
		{ 0, 4631, 0 },
		{ 3068, 2874, 0 },
		{ 0, 1495, 0 },
		{ 3079, 2998, 0 },
		{ 3076, 2452, 0 },
		{ 3073, 1658, 0 },
		{ 3079, 3001, 0 },
		{ 3081, 1943, 0 },
		{ 3083, 2209, 0 },
		{ 3085, 2099, 0 },
		{ 3079, 2052, 0 },
		{ 3068, 2882, 0 },
		{ 3083, 2213, 0 },
		{ 3080, 2929, 0 },
		{ 3079, 3009, 0 },
		{ 3085, 2101, 0 },
		{ 3080, 2931, 0 },
		{ 0, 2907, 0 },
		{ 3068, 2790, 0 },
		{ 3073, 1673, 0 },
		{ 0, 2887, 0 },
		{ 3079, 3016, 0 },
		{ 0, 2381, 0 },
		{ 3085, 2103, 0 },
		{ 3080, 2939, 0 },
		{ 0, 1675, 0 },
		{ 3086, 4633, 0 },
		{ 0, 2714, 0 },
		{ 0, 2464, 0 },
		{ 0, 0, 43 },
		{ 3040, 2730, 0 },
		{ 0, 3024, 0 },
		{ 0, 2944, 0 },
		{ 0, 1948, 0 },
		{ 3086, 4635, 0 },
		{ 0, 2219, 0 },
		{ 0, 0, 44 },
		{ 0, 2106, 0 },
		{ 3048, 4637, 0 },
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
