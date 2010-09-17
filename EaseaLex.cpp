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
#line 689 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    //DEBUG_PRT_PRT("found new symbol %s",pSym->Object->sName);
    fprintf(fpOutputFile," ar & %s;\n",pSym->Object->sName);
  }
 
#line 968 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 699 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 983 "EaseaLex.cpp"
		}
		break;
	case 62:
		{
#line 708 "EaseaLex.l"

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
 
#line 1006 "EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 725 "EaseaLex.l"

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
 
#line 1029 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 743 "EaseaLex.l"

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
 
#line 1061 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 770 "EaseaLex.l"

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
 
#line 1082 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 786 "EaseaLex.l"

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
 
#line 1104 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 803 "EaseaLex.l"
       
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
 
#line 1132 "EaseaLex.cpp"
		}
		break;
	case 68:
		{
#line 825 "EaseaLex.l"

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
 
#line 1154 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 841 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 1169 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 850 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 1181 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 858 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 1193 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 865 "EaseaLex.l"

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
 
#line 1224 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 890 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 1237 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 897 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 1251 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 906 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISER;   
 
#line 1263 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 913 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 1276 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 921 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 1288 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 927 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 1300 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 933 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 1312 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 939 "EaseaLex.l"
      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 1325 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 946 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 1338 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 953 "EaseaLex.l"
 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 1352 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 962 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 1363 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 967 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 1377 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 976 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 1391 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 985 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 1405 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 995 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 1418 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 1003 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 1427 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 1007 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 1436 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 1011 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 1445 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 1015 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 1454 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 1019 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 1464 "EaseaLex.cpp"
		}
		break;
	case 93:
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

#line 1483 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 1037 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 1490 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 1038 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 1497 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 1039 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 1504 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 1040 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 1511 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 1041 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 1518 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 1042 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1525 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 1043 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1532 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 1044 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1539 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 1045 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1546 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 1046 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1553 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 1047 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",nELITE);
  ////DEBUG_PRT_PRT("elitism is %d, elite size is %d",bELITISM, nELITE);
 
#line 1563 "EaseaLex.cpp"
		}
		break;
	case 105:
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
 
#line 1582 "EaseaLex.cpp"
		}
		break;
	case 106:
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
 
#line 1601 "EaseaLex.cpp"
		}
		break;
	case 107:
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
 
#line 1620 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 1091 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1630 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 1095 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1637 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 1096 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1644 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 1097 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1651 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 1098 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1658 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 1099 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1665 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 1100 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1672 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 1101 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1679 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 1102 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1686 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 1103 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1693 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 1104 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1700 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 1105 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1707 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 1106 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1714 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 1108 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1721 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1109 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1728 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1110 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1735 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1111 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1742 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1112 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1749 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1114 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1763 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1122 "EaseaLex.l"

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
 
#line 1783 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1136 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1797 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1144 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1811 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1153 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1825 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1162 "EaseaLex.l"

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

#line 1888 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1219 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded with no warning.\n");
  printf ("Have a nice compile time.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 1905 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1231 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1912 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1237 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1924 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1243 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1937 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1250 "EaseaLex.l"

#line 1944 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1251 "EaseaLex.l"
lineCounter++;
#line 1951 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1253 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1963 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1259 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1976 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1267 "EaseaLex.l"

#line 1983 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1268 "EaseaLex.l"

  lineCounter++;
 
#line 1992 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1272 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2004 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1278 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2018 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1286 "EaseaLex.l"

#line 2025 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1287 "EaseaLex.l"

  lineCounter++;
 
#line 2034 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1291 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  
  BEGIN COPY;
 
#line 2048 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1299 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2063 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1308 "EaseaLex.l"

#line 2070 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1309 "EaseaLex.l"
lineCounter++;
#line 2077 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1314 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2091 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1323 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2105 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1331 "EaseaLex.l"

#line 2112 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1332 "EaseaLex.l"
lineCounter++;
#line 2119 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1335 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2135 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1346 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2151 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1356 "EaseaLex.l"

#line 2158 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1359 "EaseaLex.l"

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
 
#line 2176 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1372 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2193 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1384 "EaseaLex.l"

#line 2200 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1385 "EaseaLex.l"
lineCounter++;
#line 2207 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1387 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2223 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1399 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2239 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1409 "EaseaLex.l"
lineCounter++;
#line 2246 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1410 "EaseaLex.l"

#line 2253 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1414 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2268 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1424 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2283 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1433 "EaseaLex.l"

#line 2290 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1436 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2303 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1443 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2317 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1451 "EaseaLex.l"

#line 2324 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1455 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2332 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1457 "EaseaLex.l"

#line 2339 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1463 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2346 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1464 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2353 "EaseaLex.cpp"
		}
		break;
	case 175:
	case 176:
		{
#line 1467 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2364 "EaseaLex.cpp"
		}
		break;
	case 177:
	case 178:
		{
#line 1472 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2373 "EaseaLex.cpp"
		}
		break;
	case 179:
	case 180:
		{
#line 1475 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2382 "EaseaLex.cpp"
		}
		break;
	case 181:
	case 182:
		{
#line 1478 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2399 "EaseaLex.cpp"
		}
		break;
	case 183:
	case 184:
		{
#line 1489 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2413 "EaseaLex.cpp"
		}
		break;
	case 185:
	case 186:
		{
#line 1497 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2422 "EaseaLex.cpp"
		}
		break;
	case 187:
	case 188:
		{
#line 1500 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2431 "EaseaLex.cpp"
		}
		break;
	case 189:
	case 190:
		{
#line 1503 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2440 "EaseaLex.cpp"
		}
		break;
	case 191:
	case 192:
		{
#line 1506 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2449 "EaseaLex.cpp"
		}
		break;
	case 193:
	case 194:
		{
#line 1509 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2458 "EaseaLex.cpp"
		}
		break;
	case 195:
	case 196:
		{
#line 1513 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2470 "EaseaLex.cpp"
		}
		break;
	case 197:
		{
#line 1519 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2477 "EaseaLex.cpp"
		}
		break;
	case 198:
		{
#line 1520 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2484 "EaseaLex.cpp"
		}
		break;
	case 199:
		{
#line 1521 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2491 "EaseaLex.cpp"
		}
		break;
	case 200:
		{
#line 1522 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2501 "EaseaLex.cpp"
		}
		break;
	case 201:
		{
#line 1527 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2508 "EaseaLex.cpp"
		}
		break;
	case 202:
		{
#line 1528 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2515 "EaseaLex.cpp"
		}
		break;
	case 203:
		{
#line 1529 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2522 "EaseaLex.cpp"
		}
		break;
	case 204:
		{
#line 1530 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2529 "EaseaLex.cpp"
		}
		break;
	case 205:
		{
#line 1531 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2536 "EaseaLex.cpp"
		}
		break;
	case 206:
		{
#line 1532 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2543 "EaseaLex.cpp"
		}
		break;
	case 207:
		{
#line 1533 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2550 "EaseaLex.cpp"
		}
		break;
	case 208:
		{
#line 1534 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2557 "EaseaLex.cpp"
		}
		break;
	case 209:
		{
#line 1535 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2565 "EaseaLex.cpp"
		}
		break;
	case 210:
		{
#line 1537 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2573 "EaseaLex.cpp"
		}
		break;
	case 211:
		{
#line 1539 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2581 "EaseaLex.cpp"
		}
		break;
	case 212:
		{
#line 1541 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2591 "EaseaLex.cpp"
		}
		break;
	case 213:
		{
#line 1545 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2598 "EaseaLex.cpp"
		}
		break;
	case 214:
		{
#line 1546 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2605 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1547 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2616 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1552 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2623 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1553 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"Individual");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2632 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1556 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2644 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1562 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2653 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1565 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2665 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1571 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2676 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1576 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2692 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1586 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2699 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1589 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2708 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1592 "EaseaLex.l"
BEGIN COPY;
#line 2715 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1594 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2722 "EaseaLex.cpp"
		}
		break;
	case 227:
	case 228:
	case 229:
		{
#line 1597 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2735 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1602 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2746 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1607 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 2755 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1616 "EaseaLex.l"
;
#line 2762 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1617 "EaseaLex.l"
;
#line 2769 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1618 "EaseaLex.l"
;
#line 2776 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1619 "EaseaLex.l"
;
#line 2783 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1622 "EaseaLex.l"
 /* do nothing */ 
#line 2790 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1623 "EaseaLex.l"
 /*return '\n';*/ 
#line 2797 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1624 "EaseaLex.l"
 /*return '\n';*/ 
#line 2804 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1627 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("boolean");
  return BOOL;
#line 2813 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1630 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 2823 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1634 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
  printf("match gpnode\n");
  return GPNODE;
 
#line 2835 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1641 "EaseaLex.l"
return STATIC;
#line 2842 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1642 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 2849 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1643 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 2856 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1644 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 2863 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1645 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 2870 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1646 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 2877 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1648 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 2884 "EaseaLex.cpp"
		}
		break;
#line 1649 "EaseaLex.l"
  
#line 2889 "EaseaLex.cpp"
	case 249:
		{
#line 1650 "EaseaLex.l"
return GENOME; 
#line 2894 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1652 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 2904 "EaseaLex.cpp"
		}
		break;
	case 251:
	case 252:
	case 253:
		{
#line 1659 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 2913 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1660 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 2920 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1663 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 2928 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1665 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 2935 "EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1671 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 2947 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1677 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2960 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1684 "EaseaLex.l"

#line 2967 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1686 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 2978 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1697 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 2993 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1707 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3004 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1713 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3013 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1717 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3028 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1730 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3040 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1736 "EaseaLex.l"

#line 3047 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1737 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3060 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1744 "EaseaLex.l"

#line 3067 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1745 "EaseaLex.l"
lineCounter++;
#line 3074 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1746 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3087 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1753 "EaseaLex.l"

#line 3094 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1754 "EaseaLex.l"
lineCounter++;
#line 3101 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1756 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3114 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1763 "EaseaLex.l"

#line 3121 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1764 "EaseaLex.l"
lineCounter++;
#line 3128 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1766 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3141 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1773 "EaseaLex.l"

#line 3148 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1774 "EaseaLex.l"
lineCounter++;
#line 3155 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1780 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3162 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1781 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3169 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1782 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3176 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1783 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3183 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1784 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3190 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1785 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3197 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1786 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3204 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1788 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3213 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1791 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3226 "EaseaLex.cpp"
		}
		break;
	case 288:
	case 289:
		{
#line 1800 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3237 "EaseaLex.cpp"
		}
		break;
	case 290:
	case 291:
		{
#line 1805 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3246 "EaseaLex.cpp"
		}
		break;
	case 292:
	case 293:
		{
#line 1808 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3255 "EaseaLex.cpp"
		}
		break;
	case 294:
	case 295:
		{
#line 1811 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3267 "EaseaLex.cpp"
		}
		break;
	case 296:
	case 297:
		{
#line 1817 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3280 "EaseaLex.cpp"
		}
		break;
	case 298:
	case 299:
		{
#line 1824 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3289 "EaseaLex.cpp"
		}
		break;
	case 300:
	case 301:
		{
#line 1827 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3298 "EaseaLex.cpp"
		}
		break;
	case 302:
	case 303:
		{
#line 1830 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3307 "EaseaLex.cpp"
		}
		break;
	case 304:
	case 305:
		{
#line 1833 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3316 "EaseaLex.cpp"
		}
		break;
	case 306:
	case 307:
		{
#line 1836 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3325 "EaseaLex.cpp"
		}
		break;
	case 308:
		{
#line 1839 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3334 "EaseaLex.cpp"
		}
		break;
	case 309:
		{
#line 1842 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3344 "EaseaLex.cpp"
		}
		break;
	case 310:
		{
#line 1846 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3352 "EaseaLex.cpp"
		}
		break;
	case 311:
		{
#line 1848 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3363 "EaseaLex.cpp"
		}
		break;
	case 312:
		{
#line 1853 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3374 "EaseaLex.cpp"
		}
		break;
	case 313:
		{
#line 1858 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3382 "EaseaLex.cpp"
		}
		break;
	case 314:
		{
#line 1860 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3390 "EaseaLex.cpp"
		}
		break;
	case 315:
		{
#line 1862 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3398 "EaseaLex.cpp"
		}
		break;
	case 316:
		{
#line 1864 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3406 "EaseaLex.cpp"
		}
		break;
	case 317:
		{
#line 1866 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3414 "EaseaLex.cpp"
		}
		break;
	case 318:
		{
#line 1868 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3421 "EaseaLex.cpp"
		}
		break;
	case 319:
		{
#line 1869 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3428 "EaseaLex.cpp"
		}
		break;
	case 320:
		{
#line 1870 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3436 "EaseaLex.cpp"
		}
		break;
	case 321:
		{
#line 1872 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3444 "EaseaLex.cpp"
		}
		break;
	case 322:
		{
#line 1874 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3452 "EaseaLex.cpp"
		}
		break;
	case 323:
		{
#line 1876 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3459 "EaseaLex.cpp"
		}
		break;
	case 324:
		{
#line 1877 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3471 "EaseaLex.cpp"
		}
		break;
	case 325:
		{
#line 1883 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3480 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1886 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3490 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1890 "EaseaLex.l"
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
#line 3507 "EaseaLex.cpp"
		}
		break;
	case 328:
	case 329:
		{
#line 1902 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3517 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1905 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3524 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1912 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3531 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 1913 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3538 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 1914 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3545 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 1915 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3552 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 1916 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3559 "EaseaLex.cpp"
		}
		break;
	case 336:
		{
#line 1918 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3568 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 1922 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3581 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1930 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3594 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 1939 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3607 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 1948 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3622 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 1958 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3629 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 1959 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3636 "EaseaLex.cpp"
		}
		break;
	case 343:
	case 344:
		{
#line 1962 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3647 "EaseaLex.cpp"
		}
		break;
	case 345:
	case 346:
		{
#line 1967 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3656 "EaseaLex.cpp"
		}
		break;
	case 347:
	case 348:
		{
#line 1970 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3665 "EaseaLex.cpp"
		}
		break;
	case 349:
	case 350:
		{
#line 1973 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3678 "EaseaLex.cpp"
		}
		break;
	case 351:
	case 352:
		{
#line 1980 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3691 "EaseaLex.cpp"
		}
		break;
	case 353:
	case 354:
		{
#line 1987 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3700 "EaseaLex.cpp"
		}
		break;
	case 355:
		{
#line 1990 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3707 "EaseaLex.cpp"
		}
		break;
	case 356:
		{
#line 1991 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3714 "EaseaLex.cpp"
		}
		break;
	case 357:
		{
#line 1992 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3721 "EaseaLex.cpp"
		}
		break;
	case 358:
		{
#line 1993 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3731 "EaseaLex.cpp"
		}
		break;
	case 359:
		{
#line 1998 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3738 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 1999 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3745 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 2000 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3752 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 2001 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3759 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 2002 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3767 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 2004 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3775 "EaseaLex.cpp"
		}
		break;
	case 365:
		{
#line 2006 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3783 "EaseaLex.cpp"
		}
		break;
	case 366:
		{
#line 2008 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3791 "EaseaLex.cpp"
		}
		break;
	case 367:
		{
#line 2010 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 3799 "EaseaLex.cpp"
		}
		break;
	case 368:
		{
#line 2012 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 3807 "EaseaLex.cpp"
		}
		break;
	case 369:
		{
#line 2014 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 3815 "EaseaLex.cpp"
		}
		break;
	case 370:
		{
#line 2016 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3822 "EaseaLex.cpp"
		}
		break;
	case 371:
		{
#line 2017 "EaseaLex.l"
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
#line 3845 "EaseaLex.cpp"
		}
		break;
	case 372:
		{
#line 2034 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 3856 "EaseaLex.cpp"
		}
		break;
	case 373:
		{
#line 2039 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 3870 "EaseaLex.cpp"
		}
		break;
	case 374:
		{
#line 2047 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3877 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2053 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 3887 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2057 "EaseaLex.l"

#line 3894 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2060 "EaseaLex.l"
;
#line 3901 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2061 "EaseaLex.l"
;
#line 3908 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2062 "EaseaLex.l"
;
#line 3915 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2063 "EaseaLex.l"
;
#line 3922 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2065 "EaseaLex.l"
 /* do nothing */ 
#line 3929 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2066 "EaseaLex.l"
 /*return '\n';*/ 
#line 3936 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2067 "EaseaLex.l"
 /*return '\n';*/ 
#line 3943 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2069 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 3950 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2070 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 3957 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2071 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 3964 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2072 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 3971 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2073 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 3978 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2074 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 3985 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2075 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 3992 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2076 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 3999 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2077 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4006 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2079 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4013 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2080 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4020 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2081 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4027 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2082 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4034 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2083 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4041 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2085 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4052 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2090 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4059 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2092 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4070 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2097 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4077 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2100 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4084 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2101 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4091 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2102 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4098 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2103 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4105 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2104 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4112 "EaseaLex.cpp"
		}
		break;
#line 2105 "EaseaLex.l"
 
#line 4117 "EaseaLex.cpp"
	case 407:
	case 408:
	case 409:
		{
#line 2109 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4124 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2110 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4131 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2113 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4139 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2116 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4146 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2118 "EaseaLex.l"

  lineCounter++;

#line 4155 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2121 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4165 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2126 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4175 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2131 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4185 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2136 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4195 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2141 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4205 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2146 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4215 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2155 "EaseaLex.l"
return  (char)yytext[0];
#line 4222 "EaseaLex.cpp"
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
#line 2157 "EaseaLex.l"


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

#line 4410 "EaseaLex.cpp"

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

	yytransitionmax = 4794;
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
		{ 2900, 61 },
		{ 2900, 61 },
		{ 1808, 1911 },
		{ 1448, 1448 },
		{ 67, 61 },
		{ 2294, 2272 },
		{ 2294, 2272 },
		{ 2277, 2251 },
		{ 2277, 2251 },
		{ 2414, 2414 },
		{ 2414, 2414 },
		{ 71, 3 },
		{ 72, 3 },
		{ 2167, 43 },
		{ 2168, 43 },
		{ 1930, 39 },
		{ 69, 1 },
		{ 2440, 2440 },
		{ 2440, 2440 },
		{ 67, 1 },
		{ 165, 161 },
		{ 0, 1747 },
		{ 2133, 2129 },
		{ 2900, 61 },
		{ 1281, 1280 },
		{ 2898, 61 },
		{ 1448, 1448 },
		{ 2947, 2945 },
		{ 2294, 2272 },
		{ 1302, 1301 },
		{ 2277, 2251 },
		{ 2150, 2149 },
		{ 1433, 1416 },
		{ 1434, 1416 },
		{ 71, 3 },
		{ 2902, 61 },
		{ 2167, 43 },
		{ 2126, 2125 },
		{ 1930, 39 },
		{ 2899, 61 },
		{ 2899, 61 },
		{ 2899, 61 },
		{ 2899, 61 },
		{ 2899, 61 },
		{ 2899, 61 },
		{ 2899, 61 },
		{ 2899, 61 },
		{ 2899, 61 },
		{ 2899, 61 },
		{ 70, 3 },
		{ 2901, 61 },
		{ 2166, 43 },
		{ 1523, 1517 },
		{ 1916, 39 },
		{ 2295, 2272 },
		{ 1433, 1416 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 1525, 1519 },
		{ 2896, 61 },
		{ 1435, 1416 },
		{ 1396, 1375 },
		{ 2897, 61 },
		{ 1397, 1376 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2897, 61 },
		{ 2903, 61 },
		{ 2128, 40 },
		{ 1470, 1454 },
		{ 1471, 1454 },
		{ 1384, 1362 },
		{ 1915, 40 },
		{ 2341, 2318 },
		{ 2341, 2318 },
		{ 2292, 2270 },
		{ 2292, 2270 },
		{ 2309, 2286 },
		{ 2309, 2286 },
		{ 1741, 37 },
		{ 2311, 2288 },
		{ 2311, 2288 },
		{ 2331, 2308 },
		{ 2331, 2308 },
		{ 1385, 1363 },
		{ 1386, 1364 },
		{ 1389, 1367 },
		{ 1390, 1368 },
		{ 1391, 1369 },
		{ 1392, 1370 },
		{ 1393, 1371 },
		{ 2128, 40 },
		{ 1470, 1454 },
		{ 1918, 40 },
		{ 1395, 1374 },
		{ 1398, 1377 },
		{ 2341, 2318 },
		{ 1399, 1378 },
		{ 2292, 2270 },
		{ 1400, 1379 },
		{ 2309, 2286 },
		{ 1401, 1380 },
		{ 1741, 37 },
		{ 2311, 2288 },
		{ 1402, 1381 },
		{ 2331, 2308 },
		{ 2127, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1728, 37 },
		{ 1931, 40 },
		{ 1403, 1382 },
		{ 1404, 1383 },
		{ 1472, 1454 },
		{ 2342, 2318 },
		{ 1405, 1384 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1917, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1925, 40 },
		{ 1923, 40 },
		{ 1936, 40 },
		{ 1924, 40 },
		{ 1936, 40 },
		{ 1927, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1926, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1406, 1386 },
		{ 1919, 40 },
		{ 1921, 40 },
		{ 1409, 1389 },
		{ 1936, 40 },
		{ 1410, 1390 },
		{ 1936, 40 },
		{ 1934, 40 },
		{ 1922, 40 },
		{ 1936, 40 },
		{ 1935, 40 },
		{ 1928, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1933, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1920, 40 },
		{ 1936, 40 },
		{ 1932, 40 },
		{ 1936, 40 },
		{ 1929, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1936, 40 },
		{ 1317, 21 },
		{ 1473, 1455 },
		{ 1474, 1455 },
		{ 1411, 1391 },
		{ 1304, 21 },
		{ 2368, 2345 },
		{ 2368, 2345 },
		{ 2371, 2348 },
		{ 2371, 2348 },
		{ 2377, 2354 },
		{ 2377, 2354 },
		{ 2389, 2366 },
		{ 2389, 2366 },
		{ 2390, 2367 },
		{ 2390, 2367 },
		{ 2392, 2369 },
		{ 2392, 2369 },
		{ 1412, 1392 },
		{ 1413, 1393 },
		{ 1415, 1395 },
		{ 1416, 1396 },
		{ 1417, 1397 },
		{ 1418, 1398 },
		{ 1317, 21 },
		{ 1473, 1455 },
		{ 1305, 21 },
		{ 1318, 21 },
		{ 1419, 1399 },
		{ 2368, 2345 },
		{ 1420, 1400 },
		{ 2371, 2348 },
		{ 1421, 1401 },
		{ 2377, 2354 },
		{ 1422, 1402 },
		{ 2389, 2366 },
		{ 1423, 1403 },
		{ 2390, 2367 },
		{ 1424, 1404 },
		{ 2392, 2369 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1425, 1406 },
		{ 1428, 1409 },
		{ 1429, 1410 },
		{ 1430, 1411 },
		{ 1475, 1455 },
		{ 1431, 1413 },
		{ 1432, 1415 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1321, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1310, 21 },
		{ 1308, 21 },
		{ 1323, 21 },
		{ 1309, 21 },
		{ 1323, 21 },
		{ 1312, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1311, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1329, 1307 },
		{ 1306, 21 },
		{ 1319, 21 },
		{ 1436, 1417 },
		{ 1313, 21 },
		{ 1437, 1418 },
		{ 1323, 21 },
		{ 1324, 21 },
		{ 1307, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1314, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1322, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1325, 21 },
		{ 1323, 21 },
		{ 1320, 21 },
		{ 1323, 21 },
		{ 1315, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1323, 21 },
		{ 1902, 38 },
		{ 1476, 1456 },
		{ 1477, 1456 },
		{ 1438, 1419 },
		{ 1727, 38 },
		{ 2398, 2375 },
		{ 2398, 2375 },
		{ 2399, 2376 },
		{ 2399, 2376 },
		{ 1440, 1420 },
		{ 1439, 1419 },
		{ 2216, 2191 },
		{ 2216, 2191 },
		{ 2253, 2227 },
		{ 2253, 2227 },
		{ 1442, 1421 },
		{ 1441, 1420 },
		{ 1443, 1422 },
		{ 1444, 1423 },
		{ 1445, 1424 },
		{ 1446, 1425 },
		{ 1449, 1429 },
		{ 1450, 1430 },
		{ 1902, 38 },
		{ 1476, 1456 },
		{ 1732, 38 },
		{ 2254, 2228 },
		{ 2254, 2228 },
		{ 2398, 2375 },
		{ 1451, 1431 },
		{ 2399, 2376 },
		{ 1452, 1432 },
		{ 1479, 1457 },
		{ 1480, 1457 },
		{ 2216, 2191 },
		{ 1454, 1436 },
		{ 2253, 2227 },
		{ 1455, 1437 },
		{ 1901, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 2254, 2228 },
		{ 1742, 38 },
		{ 1456, 1438 },
		{ 1457, 1439 },
		{ 1478, 1456 },
		{ 1458, 1440 },
		{ 1479, 1457 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1729, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1737, 38 },
		{ 1735, 38 },
		{ 1745, 38 },
		{ 1736, 38 },
		{ 1745, 38 },
		{ 1739, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1738, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1459, 1441 },
		{ 1733, 38 },
		{ 1481, 1457 },
		{ 1460, 1442 },
		{ 1745, 38 },
		{ 1461, 1443 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1734, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1730, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1731, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1744, 38 },
		{ 1745, 38 },
		{ 1743, 38 },
		{ 1745, 38 },
		{ 1740, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 1745, 38 },
		{ 2654, 44 },
		{ 2655, 44 },
		{ 1482, 1458 },
		{ 1483, 1458 },
		{ 67, 44 },
		{ 2268, 2243 },
		{ 2268, 2243 },
		{ 1462, 1444 },
		{ 1463, 1445 },
		{ 1464, 1446 },
		{ 1466, 1449 },
		{ 2271, 2246 },
		{ 2271, 2246 },
		{ 1467, 1450 },
		{ 1468, 1451 },
		{ 1469, 1452 },
		{ 1332, 1308 },
		{ 1333, 1309 },
		{ 1337, 1311 },
		{ 1338, 1312 },
		{ 1339, 1313 },
		{ 1488, 1460 },
		{ 1489, 1461 },
		{ 2654, 44 },
		{ 1490, 1462 },
		{ 1482, 1458 },
		{ 1492, 1466 },
		{ 1493, 1467 },
		{ 2268, 2243 },
		{ 1494, 1468 },
		{ 1485, 1459 },
		{ 1486, 1459 },
		{ 1503, 1489 },
		{ 1504, 1489 },
		{ 2271, 2246 },
		{ 1495, 1469 },
		{ 1502, 1488 },
		{ 2182, 44 },
		{ 2653, 44 },
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
		{ 1340, 1314 },
		{ 1506, 1490 },
		{ 1508, 1492 },
		{ 1509, 1493 },
		{ 1485, 1459 },
		{ 1484, 1458 },
		{ 1503, 1489 },
		{ 2183, 44 },
		{ 2180, 44 },
		{ 2175, 44 },
		{ 2183, 44 },
		{ 2172, 44 },
		{ 2179, 44 },
		{ 2177, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2174, 44 },
		{ 2169, 44 },
		{ 2176, 44 },
		{ 2171, 44 },
		{ 2183, 44 },
		{ 2178, 44 },
		{ 2173, 44 },
		{ 2170, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 1487, 1459 },
		{ 2187, 44 },
		{ 1505, 1489 },
		{ 1510, 1494 },
		{ 2183, 44 },
		{ 1511, 1495 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2184, 44 },
		{ 2185, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2186, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 2183, 44 },
		{ 159, 4 },
		{ 160, 4 },
		{ 1512, 1502 },
		{ 1513, 1502 },
		{ 2412, 2386 },
		{ 2412, 2386 },
		{ 1336, 1310 },
		{ 1342, 1315 },
		{ 1517, 1508 },
		{ 1518, 1509 },
		{ 1341, 1315 },
		{ 1519, 1510 },
		{ 1520, 1511 },
		{ 1345, 1320 },
		{ 1335, 1310 },
		{ 1524, 1518 },
		{ 1346, 1321 },
		{ 1526, 1520 },
		{ 1529, 1524 },
		{ 1530, 1526 },
		{ 1532, 1529 },
		{ 1533, 1530 },
		{ 1534, 1532 },
		{ 159, 4 },
		{ 1535, 1533 },
		{ 1512, 1502 },
		{ 1334, 1310 },
		{ 2412, 2386 },
		{ 1330, 1534 },
		{ 1347, 1322 },
		{ 1348, 1324 },
		{ 1349, 1325 },
		{ 1352, 1329 },
		{ 1353, 1332 },
		{ 1354, 1333 },
		{ 1355, 1334 },
		{ 1356, 1335 },
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
		{ 1357, 1336 },
		{ 1358, 1337 },
		{ 1359, 1338 },
		{ 1362, 1340 },
		{ 0, 2386 },
		{ 1514, 1502 },
		{ 1363, 1341 },
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
		{ 1364, 1342 },
		{ 81, 4 },
		{ 1367, 1345 },
		{ 1368, 1346 },
		{ 85, 4 },
		{ 1369, 1347 },
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
		{ 2906, 2905 },
		{ 1360, 1339 },
		{ 1370, 1348 },
		{ 2905, 2905 },
		{ 1371, 1349 },
		{ 1361, 1339 },
		{ 1374, 1352 },
		{ 1375, 1353 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 1376, 1354 },
		{ 2905, 2905 },
		{ 1377, 1355 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 1378, 1356 },
		{ 1379, 1357 },
		{ 1380, 1358 },
		{ 1381, 1359 },
		{ 1382, 1360 },
		{ 1383, 1361 },
		{ 154, 152 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 97, 80 },
		{ 99, 82 },
		{ 104, 89 },
		{ 105, 90 },
		{ 106, 91 },
		{ 107, 92 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 2905, 2905 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 108, 93 },
		{ 109, 94 },
		{ 110, 95 },
		{ 111, 96 },
		{ 112, 97 },
		{ 114, 99 },
		{ 120, 104 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 121, 105 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 109 },
		{ 1330, 1536 },
		{ 125, 110 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 1330, 1536 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 126, 111 },
		{ 127, 112 },
		{ 129, 114 },
		{ 134, 120 },
		{ 135, 121 },
		{ 136, 122 },
		{ 137, 123 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 138, 124 },
		{ 139, 125 },
		{ 140, 127 },
		{ 141, 129 },
		{ 2183, 2406 },
		{ 142, 134 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 2183, 2406 },
		{ 1331, 1535 },
		{ 0, 1535 },
		{ 143, 135 },
		{ 144, 136 },
		{ 2190, 2169 },
		{ 2192, 2170 },
		{ 2195, 2171 },
		{ 2196, 2172 },
		{ 2198, 2173 },
		{ 2193, 2171 },
		{ 2201, 2174 },
		{ 2204, 2175 },
		{ 2194, 2171 },
		{ 2205, 2176 },
		{ 2567, 2567 },
		{ 2206, 2177 },
		{ 2207, 2178 },
		{ 2197, 2172 },
		{ 2208, 2179 },
		{ 2209, 2180 },
		{ 2183, 2183 },
		{ 2202, 2184 },
		{ 2191, 2185 },
		{ 1331, 1535 },
		{ 2199, 2173 },
		{ 2200, 2186 },
		{ 2215, 2190 },
		{ 145, 137 },
		{ 2217, 2192 },
		{ 2203, 2184 },
		{ 2218, 2193 },
		{ 2219, 2194 },
		{ 2220, 2195 },
		{ 2221, 2196 },
		{ 2222, 2197 },
		{ 2223, 2198 },
		{ 2224, 2199 },
		{ 2567, 2567 },
		{ 2225, 2200 },
		{ 2226, 2201 },
		{ 2227, 2202 },
		{ 2228, 2203 },
		{ 2229, 2204 },
		{ 2230, 2205 },
		{ 2231, 2206 },
		{ 2232, 2207 },
		{ 2233, 2208 },
		{ 2234, 2209 },
		{ 2241, 2215 },
		{ 2243, 2217 },
		{ 2244, 2218 },
		{ 2245, 2219 },
		{ 0, 1535 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2246, 2220 },
		{ 2247, 2221 },
		{ 2248, 2222 },
		{ 2249, 2223 },
		{ 2250, 2224 },
		{ 2251, 2225 },
		{ 2252, 2226 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 67, 7 },
		{ 146, 138 },
		{ 2255, 2229 },
		{ 2256, 2230 },
		{ 2567, 2567 },
		{ 1536, 1535 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2257, 2231 },
		{ 2258, 2232 },
		{ 2259, 2233 },
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
		{ 2260, 2234 },
		{ 2266, 2241 },
		{ 147, 140 },
		{ 2269, 2244 },
		{ 2270, 2245 },
		{ 148, 141 },
		{ 2274, 2248 },
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
		{ 2275, 2249 },
		{ 2272, 2247 },
		{ 2276, 2250 },
		{ 149, 142 },
		{ 1157, 7 },
		{ 2273, 2247 },
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
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 2278, 2252 },
		{ 2282, 2255 },
		{ 2283, 2256 },
		{ 2284, 2257 },
		{ 2285, 2258 },
		{ 2286, 2259 },
		{ 2287, 2260 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 2288, 2266 },
		{ 2291, 2269 },
		{ 150, 144 },
		{ 2296, 2273 },
		{ 0, 1394 },
		{ 2297, 2274 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1394 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 2298, 2275 },
		{ 2299, 2276 },
		{ 2301, 2278 },
		{ 2305, 2282 },
		{ 2306, 2283 },
		{ 2307, 2284 },
		{ 2308, 2285 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 2310, 2287 },
		{ 2315, 2291 },
		{ 2318, 2296 },
		{ 2319, 2297 },
		{ 0, 1807 },
		{ 2322, 2299 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 1807 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 2321, 2298 },
		{ 2324, 2301 },
		{ 2328, 2305 },
		{ 2329, 2306 },
		{ 2320, 2298 },
		{ 2330, 2307 },
		{ 151, 147 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 2333, 2310 },
		{ 2338, 2315 },
		{ 152, 148 },
		{ 2343, 2319 },
		{ 0, 2001 },
		{ 2344, 2320 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 0, 2001 },
		{ 2345, 2321 },
		{ 2346, 2322 },
		{ 2348, 2324 },
		{ 2352, 2328 },
		{ 2353, 2329 },
		{ 2354, 2330 },
		{ 2358, 2333 },
		{ 2363, 2338 },
		{ 2366, 2343 },
		{ 2367, 2344 },
		{ 2369, 2346 },
		{ 153, 150 },
		{ 2375, 2352 },
		{ 2376, 2353 },
		{ 2381, 2358 },
		{ 2386, 2363 },
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
		{ 0, 2731 },
		{ 2474, 2452 },
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
		{ 2484, 2462 },
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
		{ 0, 2381 },
		{ 0, 2381 },
		{ 130, 115 },
		{ 90, 74 },
		{ 130, 115 },
		{ 1170, 1166 },
		{ 2107, 2104 },
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
		{ 1725, 1724 },
		{ 0, 2381 },
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
		{ 2261, 2235 },
		{ 2263, 2238 },
		{ 2261, 2235 },
		{ 2263, 2238 },
		{ 1157, 1157 },
		{ 2730, 49 },
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
		{ 1209, 1208 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2236, 2211 },
		{ 2601, 2587 },
		{ 2236, 2211 },
		{ 2617, 2604 },
		{ 1683, 1682 },
		{ 1635, 1634 },
		{ 2406, 2381 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2453, 2427 },
		{ 1680, 1679 },
		{ 1206, 1205 },
		{ 2877, 2876 },
		{ 2897, 2897 },
		{ 1590, 1589 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 2897, 2897 },
		{ 182, 173 },
		{ 192, 173 },
		{ 184, 173 },
		{ 1564, 1563 },
		{ 179, 173 },
		{ 2956, 2955 },
		{ 183, 173 },
		{ 1758, 1734 },
		{ 181, 173 },
		{ 86, 49 },
		{ 1564, 1563 },
		{ 2454, 2428 },
		{ 190, 173 },
		{ 189, 173 },
		{ 180, 173 },
		{ 188, 173 },
		{ 1611, 1610 },
		{ 185, 173 },
		{ 187, 173 },
		{ 178, 173 },
		{ 1757, 1734 },
		{ 1944, 1922 },
		{ 186, 173 },
		{ 191, 173 },
		{ 1163, 1160 },
		{ 2405, 2380 },
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
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 2921, 2921 },
		{ 2921, 2921 },
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
		{ 1222, 1221 },
		{ 431, 391 },
		{ 436, 391 },
		{ 433, 391 },
		{ 432, 391 },
		{ 435, 391 },
		{ 430, 391 },
		{ 2927, 65 },
		{ 429, 391 },
		{ 1269, 1268 },
		{ 67, 65 },
		{ 101, 83 },
		{ 434, 391 },
		{ 1959, 1935 },
		{ 437, 391 },
		{ 2763, 2762 },
		{ 1269, 1268 },
		{ 1974, 1953 },
		{ 2512, 2490 },
		{ 2816, 2815 },
		{ 428, 391 },
		{ 1164, 1160 },
		{ 2210, 2181 },
		{ 2922, 2921 },
		{ 2181, 2181 },
		{ 2181, 2181 },
		{ 2181, 2181 },
		{ 2181, 2181 },
		{ 2181, 2181 },
		{ 2181, 2181 },
		{ 2181, 2181 },
		{ 2181, 2181 },
		{ 2181, 2181 },
		{ 2181, 2181 },
		{ 1958, 1935 },
		{ 2003, 1985 },
		{ 2017, 2000 },
		{ 2857, 2856 },
		{ 1783, 1764 },
		{ 2880, 2879 },
		{ 2888, 2887 },
		{ 1805, 1786 },
		{ 1696, 1695 },
		{ 101, 83 },
		{ 2147, 2146 },
		{ 2211, 2181 },
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
		{ 2152, 2151 },
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
		{ 1169, 1165 },
		{ 2392, 2392 },
		{ 2392, 2392 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2677, 2676 },
		{ 1638, 1637 },
		{ 2717, 2716 },
		{ 2939, 2935 },
		{ 2925, 65 },
		{ 2211, 2181 },
		{ 2235, 2210 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2212, 2212 },
		{ 2923, 65 },
		{ 2722, 2721 },
		{ 2392, 2392 },
		{ 2959, 2958 },
		{ 2331, 2331 },
		{ 2975, 2972 },
		{ 2981, 2978 },
		{ 2487, 2465 },
		{ 2365, 2340 },
		{ 2913, 63 },
		{ 1169, 1165 },
		{ 2238, 2212 },
		{ 67, 63 },
		{ 1843, 1828 },
		{ 1893, 1891 },
		{ 2519, 2497 },
		{ 2529, 2508 },
		{ 2531, 2510 },
		{ 2540, 2520 },
		{ 1640, 1639 },
		{ 2545, 2525 },
		{ 2235, 2210 },
		{ 2926, 65 },
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
		{ 2238, 2212 },
		{ 118, 102 },
		{ 2557, 2539 },
		{ 2253, 2253 },
		{ 2253, 2253 },
		{ 1159, 9 },
		{ 2559, 2541 },
		{ 2372, 2372 },
		{ 2372, 2372 },
		{ 67, 9 },
		{ 2572, 2554 },
		{ 115, 100 },
		{ 2573, 2555 },
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
		{ 1662, 1661 },
		{ 1675, 1674 },
		{ 2912, 63 },
		{ 2253, 2253 },
		{ 2582, 2564 },
		{ 1159, 9 },
		{ 2911, 63 },
		{ 2372, 2372 },
		{ 2781, 2781 },
		{ 2781, 2781 },
		{ 118, 102 },
		{ 1166, 1163 },
		{ 2828, 2828 },
		{ 2828, 2828 },
		{ 2317, 2293 },
		{ 2419, 2392 },
		{ 2418, 2392 },
		{ 2356, 2331 },
		{ 2355, 2331 },
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
		{ 2781, 2781 },
		{ 2374, 2374 },
		{ 2374, 2374 },
		{ 2587, 2571 },
		{ 2828, 2828 },
		{ 2382, 2382 },
		{ 2382, 2382 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2399, 2399 },
		{ 2399, 2399 },
		{ 2595, 2580 },
		{ 1166, 1163 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 2412, 2412 },
		{ 2412, 2412 },
		{ 2429, 2429 },
		{ 2429, 2429 },
		{ 2475, 2475 },
		{ 2475, 2475 },
		{ 2558, 2558 },
		{ 2558, 2558 },
		{ 1210, 1209 },
		{ 2374, 2374 },
		{ 2909, 63 },
		{ 2604, 2590 },
		{ 2910, 63 },
		{ 2382, 2382 },
		{ 1559, 1558 },
		{ 2292, 2292 },
		{ 2620, 2607 },
		{ 2399, 2399 },
		{ 2633, 2627 },
		{ 2309, 2309 },
		{ 2309, 2309 },
		{ 2271, 2271 },
		{ 2635, 2629 },
		{ 2412, 2412 },
		{ 2641, 2636 },
		{ 2429, 2429 },
		{ 2647, 2646 },
		{ 2475, 2475 },
		{ 1684, 1683 },
		{ 2558, 2558 },
		{ 2279, 2253 },
		{ 2713, 2713 },
		{ 2713, 2713 },
		{ 2741, 2741 },
		{ 2741, 2741 },
		{ 2015, 1998 },
		{ 2377, 2377 },
		{ 2377, 2377 },
		{ 2347, 2347 },
		{ 2347, 2347 },
		{ 2680, 2679 },
		{ 2280, 2253 },
		{ 2309, 2309 },
		{ 2371, 2371 },
		{ 2371, 2371 },
		{ 2395, 2372 },
		{ 2689, 2688 },
		{ 2154, 2154 },
		{ 2154, 2154 },
		{ 2398, 2398 },
		{ 2398, 2398 },
		{ 2795, 2795 },
		{ 2795, 2795 },
		{ 2702, 2701 },
		{ 2713, 2713 },
		{ 2385, 2362 },
		{ 2741, 2741 },
		{ 2277, 2277 },
		{ 2277, 2277 },
		{ 2377, 2377 },
		{ 2016, 1999 },
		{ 2347, 2347 },
		{ 2836, 2836 },
		{ 2836, 2836 },
		{ 2311, 2311 },
		{ 2311, 2311 },
		{ 2371, 2371 },
		{ 2480, 2480 },
		{ 2480, 2480 },
		{ 2782, 2781 },
		{ 2154, 2154 },
		{ 2387, 2364 },
		{ 2398, 2398 },
		{ 2829, 2828 },
		{ 2795, 2795 },
		{ 2725, 2724 },
		{ 2522, 2522 },
		{ 2522, 2522 },
		{ 1287, 1286 },
		{ 1699, 1698 },
		{ 2277, 2277 },
		{ 2034, 2021 },
		{ 2390, 2390 },
		{ 2390, 2390 },
		{ 2407, 2382 },
		{ 2836, 2836 },
		{ 2047, 2032 },
		{ 2311, 2311 },
		{ 2561, 2561 },
		{ 2561, 2561 },
		{ 2480, 2480 },
		{ 2048, 2033 },
		{ 2408, 2382 },
		{ 2397, 2374 },
		{ 2216, 2216 },
		{ 2216, 2216 },
		{ 2750, 2750 },
		{ 2750, 2750 },
		{ 2316, 2292 },
		{ 2522, 2522 },
		{ 2426, 2399 },
		{ 2742, 2741 },
		{ 2753, 2752 },
		{ 1298, 1297 },
		{ 2293, 2271 },
		{ 2390, 2390 },
		{ 2439, 2412 },
		{ 2780, 2779 },
		{ 2455, 2429 },
		{ 2402, 2377 },
		{ 2497, 2475 },
		{ 2561, 2561 },
		{ 2576, 2558 },
		{ 2108, 2105 },
		{ 2123, 2122 },
		{ 2810, 2809 },
		{ 2339, 2316 },
		{ 2216, 2216 },
		{ 2819, 2818 },
		{ 2750, 2750 },
		{ 2827, 2826 },
		{ 1189, 1188 },
		{ 1612, 1611 },
		{ 2851, 2850 },
		{ 2332, 2309 },
		{ 2149, 2148 },
		{ 2400, 2377 },
		{ 2860, 2859 },
		{ 2871, 2870 },
		{ 1614, 1613 },
		{ 2401, 2377 },
		{ 1786, 1767 },
		{ 2882, 2881 },
		{ 2742, 2741 },
		{ 2161, 2160 },
		{ 2891, 2890 },
		{ 2714, 2713 },
		{ 2349, 2325 },
		{ 2351, 2327 },
		{ 1792, 1773 },
		{ 1225, 1224 },
		{ 2370, 2347 },
		{ 1812, 1793 },
		{ 1264, 1263 },
		{ 1839, 1823 },
		{ 2935, 2931 },
		{ 2394, 2371 },
		{ 2460, 2436 },
		{ 2471, 2449 },
		{ 2361, 2336 },
		{ 2155, 2154 },
		{ 2961, 2960 },
		{ 2425, 2398 },
		{ 2796, 2795 },
		{ 2963, 2963 },
		{ 2478, 2456 },
		{ 1841, 1826 },
		{ 2988, 2986 },
		{ 1954, 1929 },
		{ 2300, 2277 },
		{ 1765, 1740 },
		{ 1953, 1929 },
		{ 1587, 1586 },
		{ 1764, 1740 },
		{ 2837, 2836 },
		{ 1588, 1587 },
		{ 2334, 2311 },
		{ 1240, 1239 },
		{ 1784, 1765 },
		{ 2502, 2480 },
		{ 2268, 2268 },
		{ 2268, 2268 },
		{ 1606, 1605 },
		{ 1791, 1772 },
		{ 2651, 2650 },
		{ 2413, 2387 },
		{ 2672, 2671 },
		{ 2963, 2963 },
		{ 2542, 2522 },
		{ 2164, 2163 },
		{ 1607, 1606 },
		{ 2684, 2683 },
		{ 1803, 1784 },
		{ 1241, 1240 },
		{ 2416, 2390 },
		{ 2427, 2400 },
		{ 2431, 2404 },
		{ 1243, 1242 },
		{ 2438, 2410 },
		{ 1516, 1507 },
		{ 2579, 2561 },
		{ 1630, 1629 },
		{ 1631, 1630 },
		{ 2268, 2268 },
		{ 2736, 2734 },
		{ 1257, 1256 },
		{ 2242, 2216 },
		{ 1854, 1841 },
		{ 2751, 2750 },
		{ 2461, 2438 },
		{ 2757, 2756 },
		{ 1873, 1863 },
		{ 1881, 1873 },
		{ 1258, 1257 },
		{ 1218, 1217 },
		{ 1656, 1655 },
		{ 2489, 2467 },
		{ 1657, 1656 },
		{ 2500, 2478 },
		{ 1193, 1192 },
		{ 1666, 1665 },
		{ 1522, 1516 },
		{ 1975, 1954 },
		{ 1994, 1973 },
		{ 2870, 2869 },
		{ 1996, 1975 },
		{ 1999, 1978 },
		{ 1178, 1177 },
		{ 2546, 2526 },
		{ 2547, 2527 },
		{ 2549, 2530 },
		{ 2550, 2531 },
		{ 2383, 2360 },
		{ 1552, 1551 },
		{ 1553, 1552 },
		{ 1692, 1691 },
		{ 2575, 2557 },
		{ 2388, 2365 },
		{ 1293, 1292 },
		{ 1233, 1232 },
		{ 2951, 2950 },
		{ 2952, 2951 },
		{ 2583, 2565 },
		{ 1715, 1714 },
		{ 1716, 1715 },
		{ 1721, 1720 },
		{ 1580, 1579 },
		{ 1581, 1580 },
		{ 2610, 2596 },
		{ 2674, 2673 },
		{ 2486, 2464 },
		{ 2157, 2156 },
		{ 2966, 2963 },
		{ 1677, 1676 },
		{ 2163, 2162 },
		{ 2700, 2699 },
		{ 2499, 2477 },
		{ 2965, 2963 },
		{ 2711, 2710 },
		{ 2964, 2963 },
		{ 2393, 2370 },
		{ 1965, 1944 },
		{ 2503, 2481 },
		{ 1787, 1768 },
		{ 1285, 1284 },
		{ 1986, 1965 },
		{ 2530, 2509 },
		{ 2303, 2280 },
		{ 2740, 2738 },
		{ 2532, 2511 },
		{ 1682, 1681 },
		{ 2403, 2378 },
		{ 1795, 1776 },
		{ 1566, 1565 },
		{ 1616, 1615 },
		{ 1187, 1186 },
		{ 1822, 1805 },
		{ 2808, 2807 },
		{ 1289, 1288 },
		{ 2290, 2268 },
		{ 2289, 2289 },
		{ 2289, 2289 },
		{ 2562, 2544 },
		{ 2566, 2548 },
		{ 2421, 2394 },
		{ 2422, 2395 },
		{ 2849, 2848 },
		{ 2424, 2397 },
		{ 1585, 1584 },
		{ 1202, 1201 },
		{ 1295, 1294 },
		{ 1642, 1641 },
		{ 2101, 2094 },
		{ 2586, 2570 },
		{ 2104, 2099 },
		{ 2593, 2578 },
		{ 1235, 1234 },
		{ 2451, 2425 },
		{ 1592, 1591 },
		{ 2606, 2592 },
		{ 2120, 2117 },
		{ 1888, 1883 },
		{ 1180, 1179 },
		{ 2289, 2289 },
		{ 2621, 2608 },
		{ 2622, 2609 },
		{ 1208, 1207 },
		{ 2942, 2939 },
		{ 2634, 2628 },
		{ 2466, 2444 },
		{ 1776, 1757 },
		{ 2642, 2640 },
		{ 2645, 2643 },
		{ 2963, 2962 },
		{ 2151, 2150 },
		{ 2971, 2968 },
		{ 1668, 1667 },
		{ 2979, 2976 },
		{ 2479, 2457 },
		{ 1271, 1270 },
		{ 2991, 2990 },
		{ 2631, 2631 },
		{ 2631, 2631 },
		{ 2254, 2254 },
		{ 2254, 2254 },
		{ 2627, 2619 },
		{ 2541, 2521 },
		{ 1297, 1296 },
		{ 2442, 2416 },
		{ 2449, 2423 },
		{ 2476, 2454 },
		{ 2511, 2489 },
		{ 1826, 1811 },
		{ 2554, 2535 },
		{ 2594, 2579 },
		{ 2555, 2537 },
		{ 1283, 1282 },
		{ 2752, 2751 },
		{ 2524, 2502 },
		{ 2560, 2542 },
		{ 1985, 1964 },
		{ 1661, 1660 },
		{ 2312, 2289 },
		{ 2457, 2431 },
		{ 2631, 2631 },
		{ 2436, 2408 },
		{ 2254, 2254 },
		{ 1722, 1721 },
		{ 1754, 1731 },
		{ 2441, 2415 },
		{ 2029, 2013 },
		{ 2727, 2726 },
		{ 2447, 2421 },
		{ 1836, 1820 },
		{ 1641, 1640 },
		{ 1589, 1588 },
		{ 2079, 2061 },
		{ 2080, 2062 },
		{ 1753, 1731 },
		{ 1650, 1649 },
		{ 1546, 1545 },
		{ 2581, 2563 },
		{ 2755, 2754 },
		{ 2106, 2103 },
		{ 2762, 2761 },
		{ 1300, 1299 },
		{ 1660, 1659 },
		{ 2473, 2451 },
		{ 2114, 2111 },
		{ 2117, 2115 },
		{ 1600, 1599 },
		{ 2812, 2811 },
		{ 1890, 1887 },
		{ 2313, 2289 },
		{ 1288, 1287 },
		{ 2821, 2820 },
		{ 1896, 1895 },
		{ 1769, 1746 },
		{ 2609, 2595 },
		{ 1407, 1387 },
		{ 2611, 2597 },
		{ 2853, 2852 },
		{ 1227, 1226 },
		{ 1350, 1326 },
		{ 2862, 2861 },
		{ 1574, 1573 },
		{ 1960, 1937 },
		{ 1964, 1943 },
		{ 2628, 2620 },
		{ 2509, 2487 },
		{ 2884, 2883 },
		{ 1615, 1614 },
		{ 1788, 1769 },
		{ 2893, 2892 },
		{ 1790, 1771 },
		{ 2640, 2635 },
		{ 2520, 2498 },
		{ 1979, 1958 },
		{ 1981, 1960 },
		{ 1983, 1962 },
		{ 2649, 2648 },
		{ 2420, 2393 },
		{ 1282, 1281 },
		{ 2940, 2936 },
		{ 1624, 1623 },
		{ 1191, 1190 },
		{ 1372, 1350 },
		{ 1586, 1585 },
		{ 2682, 2681 },
		{ 1201, 1200 },
		{ 2962, 2961 },
		{ 2014, 1997 },
		{ 2548, 2529 },
		{ 2968, 2965 },
		{ 1821, 1804 },
		{ 2704, 2703 },
		{ 2710, 2709 },
		{ 2636, 2631 },
		{ 1701, 1700 },
		{ 2281, 2254 },
		{ 1709, 1708 },
		{ 2990, 2988 },
		{ 1251, 1250 },
		{ 2803, 2803 },
		{ 2803, 2803 },
		{ 2695, 2695 },
		{ 2695, 2695 },
		{ 2844, 2844 },
		{ 2844, 2844 },
		{ 2368, 2368 },
		{ 2368, 2368 },
		{ 2389, 2389 },
		{ 2389, 2389 },
		{ 2737, 2735 },
		{ 1828, 1813 },
		{ 2304, 2281 },
		{ 2456, 2430 },
		{ 1998, 1977 },
		{ 2459, 2435 },
		{ 1213, 1212 },
		{ 2756, 2755 },
		{ 1838, 1822 },
		{ 1351, 1328 },
		{ 2384, 2361 },
		{ 2764, 2763 },
		{ 2773, 2772 },
		{ 2803, 2803 },
		{ 1623, 1622 },
		{ 2695, 2695 },
		{ 2596, 2581 },
		{ 2844, 2844 },
		{ 2790, 2789 },
		{ 2368, 2368 },
		{ 2791, 2790 },
		{ 2389, 2389 },
		{ 2793, 2792 },
		{ 1301, 1300 },
		{ 1262, 1261 },
		{ 2806, 2805 },
		{ 1862, 1850 },
		{ 2021, 2004 },
		{ 1771, 1749 },
		{ 2813, 2812 },
		{ 1875, 1866 },
		{ 2817, 2816 },
		{ 1775, 1756 },
		{ 1887, 1882 },
		{ 2822, 2821 },
		{ 2059, 2045 },
		{ 1387, 1365 },
		{ 2834, 2833 },
		{ 2624, 2611 },
		{ 1545, 1544 },
		{ 2847, 2846 },
		{ 2081, 2063 },
		{ 2092, 2079 },
		{ 2093, 2080 },
		{ 2854, 2853 },
		{ 2510, 2488 },
		{ 2858, 2857 },
		{ 1192, 1191 },
		{ 2103, 2098 },
		{ 2863, 2862 },
		{ 2869, 2868 },
		{ 1785, 1766 },
		{ 1897, 1896 },
		{ 2523, 2501 },
		{ 1687, 1686 },
		{ 1228, 1227 },
		{ 2650, 2649 },
		{ 2885, 2884 },
		{ 1250, 1249 },
		{ 2889, 2888 },
		{ 2116, 2114 },
		{ 1697, 1696 },
		{ 2894, 2893 },
		{ 1599, 1598 },
		{ 1649, 1648 },
		{ 2678, 2677 },
		{ 2907, 2904 },
		{ 1962, 1940 },
		{ 1963, 1942 },
		{ 2683, 2682 },
		{ 1793, 1774 },
		{ 2933, 2929 },
		{ 1702, 1701 },
		{ 2936, 2932 },
		{ 1708, 1707 },
		{ 2698, 2697 },
		{ 2428, 2401 },
		{ 2945, 2942 },
		{ 2153, 2152 },
		{ 1557, 1556 },
		{ 2705, 2704 },
		{ 2239, 2213 },
		{ 1976, 1955 },
		{ 1280, 1279 },
		{ 2804, 2803 },
		{ 1427, 1408 },
		{ 2696, 2695 },
		{ 113, 98 },
		{ 2845, 2844 },
		{ 2723, 2722 },
		{ 2391, 2368 },
		{ 2443, 2417 },
		{ 2415, 2389 },
		{ 2569, 2551 },
		{ 2728, 2727 },
		{ 1823, 1806 },
		{ 1573, 1572 },
		{ 1223, 1222 },
		{ 2302, 2279 },
		{ 2720, 2720 },
		{ 2720, 2720 },
		{ 1694, 1694 },
		{ 1694, 1694 },
		{ 1220, 1220 },
		{ 1220, 1220 },
		{ 2886, 2886 },
		{ 2886, 2886 },
		{ 2855, 2855 },
		{ 2855, 2855 },
		{ 2675, 2675 },
		{ 2675, 2675 },
		{ 2591, 2591 },
		{ 2591, 2591 },
		{ 2814, 2814 },
		{ 2814, 2814 },
		{ 1809, 1790 },
		{ 1252, 1251 },
		{ 1625, 1624 },
		{ 2943, 2940 },
		{ 2337, 2314 },
		{ 1664, 1663 },
		{ 2159, 2158 },
		{ 2720, 2720 },
		{ 2504, 2482 },
		{ 1694, 1694 },
		{ 1710, 1709 },
		{ 1220, 1220 },
		{ 2030, 2014 },
		{ 2886, 2886 },
		{ 2109, 2106 },
		{ 2855, 2855 },
		{ 1426, 1407 },
		{ 2675, 2675 },
		{ 1601, 1600 },
		{ 2591, 2591 },
		{ 1651, 1650 },
		{ 2814, 2814 },
		{ 1575, 1574 },
		{ 1837, 1821 },
		{ 2978, 2975 },
		{ 2125, 2124 },
		{ 2002, 1983 },
		{ 1892, 1890 },
		{ 1547, 1546 },
		{ 1610, 1609 },
		{ 2875, 2875 },
		{ 2875, 2875 },
		{ 2839, 2839 },
		{ 2839, 2839 },
		{ 1678, 1678 },
		{ 1678, 1678 },
		{ 2798, 2798 },
		{ 2798, 2798 },
		{ 2690, 2690 },
		{ 2690, 2690 },
		{ 1204, 1204 },
		{ 1204, 1204 },
		{ 2832, 2832 },
		{ 2832, 2832 },
		{ 1215, 1215 },
		{ 1215, 1215 },
		{ 1689, 1689 },
		{ 1689, 1689 },
		{ 1723, 1722 },
		{ 2020, 2003 },
		{ 2574, 2556 },
		{ 1691, 1690 },
		{ 2148, 2147 },
		{ 2875, 2875 },
		{ 1853, 1840 },
		{ 2839, 2839 },
		{ 1562, 1561 },
		{ 1678, 1678 },
		{ 1973, 1952 },
		{ 2798, 2798 },
		{ 1238, 1237 },
		{ 2690, 2690 },
		{ 1865, 1855 },
		{ 1204, 1204 },
		{ 2748, 2747 },
		{ 2832, 2832 },
		{ 1796, 1777 },
		{ 1215, 1215 },
		{ 2078, 2060 },
		{ 1689, 1689 },
		{ 1637, 1636 },
		{ 1255, 1254 },
		{ 1807, 1788 },
		{ 1491, 1465 },
		{ 1550, 1549 },
		{ 1772, 1752 },
		{ 2949, 2948 },
		{ 2721, 2720 },
		{ 1267, 1266 },
		{ 1695, 1694 },
		{ 1578, 1577 },
		{ 1221, 1220 },
		{ 1217, 1216 },
		{ 2887, 2886 },
		{ 2958, 2957 },
		{ 2856, 2855 },
		{ 2000, 1979 },
		{ 2676, 2675 },
		{ 2001, 1981 },
		{ 2605, 2591 },
		{ 2485, 2463 },
		{ 2815, 2814 },
		{ 2553, 2534 },
		{ 1394, 1372 },
		{ 2113, 2110 },
		{ 2879, 2878 },
		{ 1713, 1712 },
		{ 2626, 2618 },
		{ 1654, 1653 },
		{ 2498, 2476 },
		{ 2984, 2981 },
		{ 1604, 1603 },
		{ 1628, 1627 },
		{ 1185, 1184 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 1181, 1181 },
		{ 1181, 1181 },
		{ 2492, 2492 },
		{ 2492, 2492 },
		{ 2494, 2494 },
		{ 2494, 2494 },
		{ 2744, 2743 },
		{ 2771, 2770 },
		{ 0, 1161 },
		{ 0, 84 },
		{ 0, 2182 },
		{ 2766, 2766 },
		{ 2766, 2766 },
		{ 1746, 1902 },
		{ 1937, 2128 },
		{ 1326, 1317 },
		{ 2876, 2875 },
		{ 1700, 1699 },
		{ 2840, 2839 },
		{ 1941, 1919 },
		{ 1679, 1678 },
		{ 2373, 2373 },
		{ 2799, 2798 },
		{ 1181, 1181 },
		{ 2691, 2690 },
		{ 2492, 2492 },
		{ 1205, 1204 },
		{ 2494, 2494 },
		{ 2833, 2832 },
		{ 2590, 2574 },
		{ 1216, 1215 },
		{ 2122, 2120 },
		{ 1690, 1689 },
		{ 2852, 2851 },
		{ 2766, 2766 },
		{ 1827, 1812 },
		{ 2325, 2302 },
		{ 2327, 2304 },
		{ 2597, 2582 },
		{ 1665, 1664 },
		{ 0, 1161 },
		{ 0, 84 },
		{ 0, 2182 },
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
		{ 2237, 2237 },
		{ 2237, 2237 },
		{ 2237, 2237 },
		{ 2237, 2237 },
		{ 2237, 2237 },
		{ 2237, 2237 },
		{ 2237, 2237 },
		{ 2237, 2237 },
		{ 2237, 2237 },
		{ 2237, 2237 },
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
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2264, 2264 },
		{ 2726, 2725 },
		{ 2861, 2860 },
		{ 1777, 1758 },
		{ 1942, 1919 },
		{ 1226, 1225 },
		{ 1268, 1267 },
		{ 2396, 2373 },
		{ 2521, 2499 },
		{ 1182, 1181 },
		{ 1261, 1260 },
		{ 2514, 2492 },
		{ 2031, 2015 },
		{ 2516, 2494 },
		{ 2526, 2504 },
		{ 2618, 2605 },
		{ 2619, 2606 },
		{ 2452, 2426 },
		{ 2883, 2882 },
		{ 2767, 2766 },
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
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 2899, 2899 },
		{ 2899, 2899 },
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
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 2483, 2483 },
		{ 2483, 2483 },
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
		{ 2983, 2983 },
		{ 2785, 2785 },
		{ 2785, 2785 },
		{ 2706, 2706 },
		{ 2706, 2706 },
		{ 2468, 2468 },
		{ 2468, 2468 },
		{ 2747, 2746 },
		{ 1327, 1306 },
		{ 1236, 1236 },
		{ 2336, 2313 },
		{ 2483, 2483 },
		{ 2045, 2029 },
		{ 2534, 2513 },
		{ 2754, 2753 },
		{ 2892, 2891 },
		{ 1190, 1189 },
		{ 1850, 1836 },
		{ 1279, 1278 },
		{ 2061, 2047 },
		{ 2062, 2048 },
		{ 2462, 2439 },
		{ 2465, 2443 },
		{ 2983, 2983 },
		{ 2785, 2785 },
		{ 1263, 1262 },
		{ 2706, 2706 },
		{ 2772, 2771 },
		{ 2468, 2468 },
		{ 1720, 1719 },
		{ 2551, 2532 },
		{ 1296, 1295 },
		{ 2648, 2647 },
		{ 2789, 2788 },
		{ 1556, 1555 },
		{ 1686, 1685 },
		{ 2792, 2791 },
		{ 2556, 2538 },
		{ 1212, 1211 },
		{ 1558, 1557 },
		{ 2482, 2460 },
		{ 1408, 1388 },
		{ 2563, 2545 },
		{ 1755, 1733 },
		{ 2811, 2810 },
		{ 1560, 1559 },
		{ 2681, 2680 },
		{ 2571, 2553 },
		{ 1265, 1264 },
		{ 2362, 2337 },
		{ 1895, 1893 },
		{ 2971, 2971 },
		{ 2820, 2819 },
		{ 1563, 1562 },
		{ 2314, 2290 },
		{ 2111, 2108 },
		{ 1663, 1662 },
		{ 2501, 2479 },
		{ 2703, 2702 },
		{ 1299, 1298 },
		{ 1773, 1753 },
		{ 2694, 2693 },
		{ 2830, 2829 },
		{ 2768, 2767 },
		{ 1685, 1684 },
		{ 67, 5 },
		{ 1211, 1210 },
		{ 2843, 2842 },
		{ 2124, 2123 },
		{ 2983, 2980 },
		{ 2802, 2801 },
		{ 2493, 2471 },
		{ 2783, 2782 },
		{ 2588, 2572 },
		{ 2971, 2971 },
		{ 1328, 1306 },
		{ 1237, 1236 },
		{ 2589, 2573 },
		{ 2505, 2483 },
		{ 2774, 2773 },
		{ 2543, 2523 },
		{ 2794, 2793 },
		{ 1278, 1277 },
		{ 2446, 2420 },
		{ 1688, 1687 },
		{ 2788, 2787 },
		{ 2115, 2113 },
		{ 1214, 1213 },
		{ 2585, 2569 },
		{ 2734, 2732 },
		{ 2786, 2785 },
		{ 2404, 2379 },
		{ 2707, 2706 },
		{ 2985, 2983 },
		{ 2490, 2468 },
		{ 2928, 2923 },
		{ 1766, 1743 },
		{ 2467, 2445 },
		{ 2359, 2334 },
		{ 1904, 1901 },
		{ 1767, 1743 },
		{ 2145, 2144 },
		{ 2797, 2796 },
		{ 2379, 2356 },
		{ 1903, 1901 },
		{ 2735, 2732 },
		{ 2323, 2300 },
		{ 2838, 2837 },
		{ 2267, 2242 },
		{ 2552, 2533 },
		{ 1756, 1733 },
		{ 2709, 2708 },
		{ 1388, 1366 },
		{ 1748, 1728 },
		{ 2746, 2745 },
		{ 1939, 1916 },
		{ 2533, 2512 },
		{ 174, 5 },
		{ 1747, 1728 },
		{ 1200, 1199 },
		{ 1938, 1916 },
		{ 175, 5 },
		{ 1674, 1673 },
		{ 2445, 2419 },
		{ 2130, 2127 },
		{ 2929, 2923 },
		{ 1344, 1318 },
		{ 1978, 1957 },
		{ 176, 5 },
		{ 2129, 2127 },
		{ 2496, 2474 },
		{ 1565, 1564 },
		{ 1284, 1283 },
		{ 1825, 1809 },
		{ 1572, 1571 },
		{ 2144, 2143 },
		{ 1667, 1666 },
		{ 1249, 1248 },
		{ 2632, 2626 },
		{ 2805, 2804 },
		{ 1995, 1974 },
		{ 2807, 2806 },
		{ 1179, 1178 },
		{ 2974, 2971 },
		{ 173, 5 },
		{ 2506, 2484 },
		{ 1622, 1621 },
		{ 1749, 1729 },
		{ 1750, 1730 },
		{ 1751, 1730 },
		{ 1676, 1675 },
		{ 2643, 2641 },
		{ 1365, 1343 },
		{ 1576, 1575 },
		{ 2410, 2384 },
		{ 2160, 2159 },
		{ 2326, 2303 },
		{ 2004, 1986 },
		{ 2162, 2161 },
		{ 2417, 2391 },
		{ 2831, 2830 },
		{ 2673, 2672 },
		{ 2012, 1994 },
		{ 1366, 1344 },
		{ 1681, 1680 },
		{ 1626, 1625 },
		{ 2423, 2396 },
		{ 1544, 1543 },
		{ 2846, 2845 },
		{ 2535, 2514 },
		{ 2848, 2847 },
		{ 2537, 2516 },
		{ 1866, 1856 },
		{ 2019, 2002 },
		{ 1234, 1233 },
		{ 1207, 1206 },
		{ 2544, 2524 },
		{ 2697, 2696 },
		{ 2430, 2403 },
		{ 2699, 2698 },
		{ 1774, 1754 },
		{ 1882, 1874 },
		{ 2435, 2407 },
		{ 1883, 1875 },
		{ 1634, 1633 },
		{ 1292, 1291 },
		{ 2872, 2871 },
		{ 2708, 2707 },
		{ 1548, 1547 },
		{ 1253, 1252 },
		{ 1270, 1269 },
		{ 2712, 2711 },
		{ 2350, 2326 },
		{ 2715, 2714 },
		{ 1894, 1892 },
		{ 2719, 2718 },
		{ 1752, 1730 },
		{ 1183, 1182 },
		{ 2448, 2422 },
		{ 2063, 2049 },
		{ 2450, 2424 },
		{ 1693, 1692 },
		{ 2565, 2547 },
		{ 2213, 2187 },
		{ 2904, 2896 },
		{ 2357, 2332 },
		{ 2570, 2552 },
		{ 1591, 1590 },
		{ 1648, 1647 },
		{ 1277, 1276 },
		{ 1598, 1597 },
		{ 2738, 2736 },
		{ 2458, 2432 },
		{ 98, 81 },
		{ 2931, 2926 },
		{ 2932, 2928 },
		{ 2578, 2560 },
		{ 2745, 2744 },
		{ 2094, 2081 },
		{ 2098, 2091 },
		{ 1652, 1651 },
		{ 2749, 2748 },
		{ 2464, 2442 },
		{ 1940, 1917 },
		{ 2946, 2943 },
		{ 1203, 1202 },
		{ 1707, 1706 },
		{ 2470, 2448 },
		{ 2955, 2954 },
		{ 1802, 1783 },
		{ 2472, 2450 },
		{ 1199, 1198 },
		{ 1602, 1601 },
		{ 1806, 1787 },
		{ 2112, 2109 },
		{ 1447, 1426 },
		{ 2967, 2964 },
		{ 2769, 2768 },
		{ 2481, 2459 },
		{ 1711, 1710 },
		{ 1810, 1791 },
		{ 2976, 2973 },
		{ 1448, 1427 },
		{ 1813, 1795 },
		{ 2608, 2594 },
		{ 1242, 1241 },
		{ 2784, 2783 },
		{ 2488, 2466 },
		{ 2787, 2786 },
		{ 1219, 1218 },
		{ 1977, 1956 },
		{ 1719, 1718 },
		{ 1804, 1785 },
		{ 1997, 1976 },
		{ 2716, 2715 },
		{ 1794, 1775 },
		{ 2937, 2933 },
		{ 2265, 2239 },
		{ 2973, 2970 },
		{ 1814, 1796 },
		{ 2874, 2873 },
		{ 2739, 2737 },
		{ 1373, 1351 },
		{ 2380, 2357 },
		{ 1984, 1963 },
		{ 1956, 1933 },
		{ 2908, 2907 },
		{ 2765, 2764 },
		{ 1291, 1290 },
		{ 2835, 2834 },
		{ 128, 113 },
		{ 2801, 2800 },
		{ 2477, 2455 },
		{ 2718, 2717 },
		{ 1698, 1697 },
		{ 1842, 1827 },
		{ 1224, 1223 },
		{ 2859, 2858 },
		{ 2969, 2966 },
		{ 2890, 2889 },
		{ 2972, 2969 },
		{ 1294, 1293 },
		{ 2724, 2723 },
		{ 1184, 1183 },
		{ 2693, 2692 },
		{ 585, 531 },
		{ 2842, 2841 },
		{ 2980, 2977 },
		{ 2046, 2031 },
		{ 2770, 2769 },
		{ 2592, 2576 },
		{ 2818, 2817 },
		{ 2987, 2985 },
		{ 1584, 1583 },
		{ 2679, 2678 },
		{ 586, 531 },
		{ 2432, 2405 },
		{ 2538, 2517 },
		{ 2513, 2491 },
		{ 2495, 2473 },
		{ 1326, 1319 },
		{ 1937, 1931 },
		{ 1276, 1273 },
		{ 1746, 1742 },
		{ 2469, 2447 },
		{ 202, 179 },
		{ 1952, 1928 },
		{ 1655, 1654 },
		{ 200, 179 },
		{ 1613, 1612 },
		{ 201, 179 },
		{ 587, 531 },
		{ 1507, 1491 },
		{ 2692, 2691 },
		{ 1955, 1932 },
		{ 2607, 2593 },
		{ 2060, 2046 },
		{ 1724, 1723 },
		{ 1891, 1888 },
		{ 199, 179 },
		{ 2378, 2355 },
		{ 1636, 1635 },
		{ 2701, 2700 },
		{ 2508, 2486 },
		{ 2463, 2441 },
		{ 1266, 1265 },
		{ 2841, 2840 },
		{ 1256, 1255 },
		{ 2340, 2317 },
		{ 1639, 1638 },
		{ 2564, 2546 },
		{ 1186, 1185 },
		{ 1561, 1560 },
		{ 2629, 2621 },
		{ 2850, 2849 },
		{ 2517, 2495 },
		{ 2146, 2145 },
		{ 1605, 1604 },
		{ 2950, 2949 },
		{ 1551, 1550 },
		{ 1855, 1842 },
		{ 2099, 2092 },
		{ 2525, 2503 },
		{ 2957, 2956 },
		{ 1239, 1238 },
		{ 2527, 2505 },
		{ 2960, 2959 },
		{ 1863, 1853 },
		{ 2646, 2645 },
		{ 2580, 2562 },
		{ 1714, 1713 },
		{ 2156, 2155 },
		{ 2105, 2101 },
		{ 2800, 2799 },
		{ 2970, 2967 },
		{ 2873, 2872 },
		{ 2158, 2157 },
		{ 2444, 2418 },
		{ 1579, 1578 },
		{ 2878, 2877 },
		{ 1629, 1628 },
		{ 2977, 2974 },
		{ 1188, 1187 },
		{ 2881, 2880 },
		{ 2539, 2519 },
		{ 1286, 1285 },
		{ 2809, 2808 },
		{ 2032, 2017 },
		{ 2491, 2469 },
		{ 2986, 2984 },
		{ 2743, 2742 },
		{ 2364, 2339 },
		{ 2033, 2020 },
		{ 1768, 1744 },
		{ 641, 584 },
		{ 703, 650 },
		{ 806, 759 },
		{ 406, 368 },
		{ 1197, 13 },
		{ 1570, 25 },
		{ 1646, 31 },
		{ 67, 13 },
		{ 67, 25 },
		{ 67, 31 },
		{ 2670, 45 },
		{ 1176, 11 },
		{ 1275, 19 },
		{ 67, 45 },
		{ 67, 11 },
		{ 67, 19 },
		{ 532, 477 },
		{ 1672, 33 },
		{ 702, 650 },
		{ 407, 368 },
		{ 67, 33 },
		{ 2777, 55 },
		{ 642, 584 },
		{ 1620, 29 },
		{ 67, 55 },
		{ 536, 481 },
		{ 67, 29 },
		{ 1247, 17 },
		{ 807, 759 },
		{ 2867, 59 },
		{ 67, 17 },
		{ 1596, 27 },
		{ 67, 59 },
		{ 546, 491 },
		{ 67, 27 },
		{ 560, 503 },
		{ 561, 505 },
		{ 566, 510 },
		{ 233, 200 },
		{ 594, 535 },
		{ 607, 548 },
		{ 610, 551 },
		{ 618, 559 },
		{ 1778, 1759 },
		{ 1966, 1945 },
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
		{ 741, 687 },
		{ 750, 696 },
		{ 1988, 1967 },
		{ 1989, 1968 },
		{ 754, 700 },
		{ 1798, 1779 },
		{ 1799, 1780 },
		{ 755, 701 },
		{ 798, 751 },
		{ 266, 230 },
		{ 832, 787 },
		{ 840, 795 },
		{ 855, 810 },
		{ 890, 845 },
		{ 894, 849 },
		{ 2011, 1993 },
		{ 910, 868 },
		{ 924, 886 },
		{ 948, 914 },
		{ 1819, 1801 },
		{ 951, 917 },
		{ 958, 925 },
		{ 968, 935 },
		{ 984, 953 },
		{ 1003, 971 },
		{ 2027, 2010 },
		{ 1010, 979 },
		{ 1012, 981 },
		{ 1024, 997 },
		{ 1834, 1818 },
		{ 1037, 1011 },
		{ 1195, 13 },
		{ 1568, 25 },
		{ 1644, 31 },
		{ 1042, 1017 },
		{ 1049, 1025 },
		{ 1050, 1026 },
		{ 2668, 45 },
		{ 1174, 11 },
		{ 1273, 19 },
		{ 1064, 1046 },
		{ 1081, 1065 },
		{ 1086, 1071 },
		{ 1093, 1082 },
		{ 1670, 33 },
		{ 1104, 1094 },
		{ 1110, 1100 },
		{ 1145, 1144 },
		{ 2776, 55 },
		{ 273, 237 },
		{ 1618, 29 },
		{ 287, 248 },
		{ 292, 253 },
		{ 298, 258 },
		{ 1245, 17 },
		{ 307, 267 },
		{ 2865, 59 },
		{ 323, 282 },
		{ 1594, 27 },
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
		{ 513, 456 },
		{ 514, 457 },
		{ 516, 459 },
		{ 528, 473 },
		{ 67, 53 },
		{ 67, 41 },
		{ 67, 57 },
		{ 67, 35 },
		{ 67, 51 },
		{ 67, 23 },
		{ 67, 47 },
		{ 67, 15 },
		{ 2039, 2025 },
		{ 2041, 2026 },
		{ 2100, 2093 },
		{ 2921, 2920 },
		{ 443, 397 },
		{ 445, 397 },
		{ 390, 353 },
		{ 2040, 2025 },
		{ 2042, 2026 },
		{ 765, 711 },
		{ 410, 371 },
		{ 413, 372 },
		{ 411, 371 },
		{ 472, 425 },
		{ 553, 498 },
		{ 446, 397 },
		{ 681, 629 },
		{ 682, 630 },
		{ 2037, 2023 },
		{ 1846, 1832 },
		{ 444, 397 },
		{ 762, 708 },
		{ 480, 432 },
		{ 481, 432 },
		{ 1045, 1020 },
		{ 1949, 1925 },
		{ 319, 278 },
		{ 639, 582 },
		{ 211, 184 },
		{ 412, 371 },
		{ 482, 432 },
		{ 210, 184 },
		{ 493, 441 },
		{ 1948, 1925 },
		{ 282, 243 },
		{ 215, 187 },
		{ 1970, 1949 },
		{ 212, 184 },
		{ 419, 379 },
		{ 846, 801 },
		{ 591, 532 },
		{ 495, 441 },
		{ 847, 802 },
		{ 343, 301 },
		{ 590, 532 },
		{ 1947, 1925 },
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
		{ 1781, 1762 },
		{ 842, 797 },
		{ 1053, 1030 },
		{ 222, 190 },
		{ 1971, 1950 },
		{ 766, 712 },
		{ 905, 863 },
		{ 2360, 2335 },
		{ 2759, 53 },
		{ 2141, 41 },
		{ 2824, 57 },
		{ 1704, 35 },
		{ 2732, 51 },
		{ 1541, 23 },
		{ 2686, 47 },
		{ 1230, 15 },
		{ 485, 434 },
		{ 557, 502 },
		{ 274, 238 },
		{ 220, 188 },
		{ 2919, 2917 },
		{ 558, 502 },
		{ 218, 188 },
		{ 402, 364 },
		{ 771, 717 },
		{ 219, 188 },
		{ 671, 617 },
		{ 1078, 1062 },
		{ 2930, 2925 },
		{ 673, 619 },
		{ 1084, 1068 },
		{ 816, 770 },
		{ 559, 502 },
		{ 275, 238 },
		{ 234, 201 },
		{ 1101, 1091 },
		{ 1177, 1174 },
		{ 1761, 1737 },
		{ 1232, 1230 },
		{ 2941, 2938 },
		{ 1102, 1092 },
		{ 746, 692 },
		{ 988, 957 },
		{ 534, 479 },
		{ 700, 648 },
		{ 2671, 2668 },
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
		{ 2920, 2919 },
		{ 1072, 1056 },
		{ 368, 331 },
		{ 295, 256 },
		{ 1082, 1066 },
		{ 329, 287 },
		{ 377, 339 },
		{ 781, 730 },
		{ 1094, 1083 },
		{ 2934, 2930 },
		{ 1096, 1085 },
		{ 782, 731 },
		{ 789, 741 },
		{ 795, 748 },
		{ 328, 287 },
		{ 327, 287 },
		{ 1108, 1098 },
		{ 1780, 1761 },
		{ 1109, 1099 },
		{ 196, 178 },
		{ 2944, 2941 },
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
		{ 1968, 1947 },
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
		{ 1512, 1512 },
		{ 1433, 1433 },
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
		{ 1537, 1537 },
		{ 227, 194 },
		{ 801, 754 },
		{ 802, 755 },
		{ 944, 907 },
		{ 2006, 1988 },
		{ 945, 908 },
		{ 946, 910 },
		{ 1512, 1512 },
		{ 1433, 1433 },
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
		{ 1470, 1470 },
		{ 1537, 1537 },
		{ 1816, 1798 },
		{ 226, 193 },
		{ 1473, 1473 },
		{ 1476, 1476 },
		{ 1479, 1479 },
		{ 1482, 1482 },
		{ 1485, 1485 },
		{ 344, 304 },
		{ 616, 557 },
		{ 992, 960 },
		{ 375, 337 },
		{ 314, 273 },
		{ 1008, 977 },
		{ 636, 576 },
		{ 549, 494 },
		{ 241, 208 },
		{ 1503, 1503 },
		{ 1016, 985 },
		{ 415, 374 },
		{ 517, 460 },
		{ 881, 837 },
		{ 1470, 1470 },
		{ 1009, 978 },
		{ 1343, 1512 },
		{ 1343, 1433 },
		{ 1473, 1473 },
		{ 1476, 1476 },
		{ 1479, 1479 },
		{ 1482, 1482 },
		{ 1485, 1485 },
		{ 854, 809 },
		{ 729, 675 },
		{ 734, 680 },
		{ 857, 812 },
		{ 2024, 2007 },
		{ 860, 815 },
		{ 614, 555 },
		{ 1028, 1001 },
		{ 1343, 1537 },
		{ 1503, 1503 },
		{ 1031, 1004 },
		{ 1032, 1005 },
		{ 512, 455 },
		{ 870, 825 },
		{ 871, 826 },
		{ 2916, 2912 },
		{ 872, 827 },
		{ 1044, 1019 },
		{ 456, 408 },
		{ 1046, 1021 },
		{ 554, 499 },
		{ 403, 365 },
		{ 2057, 2043 },
		{ 2058, 2044 },
		{ 272, 236 },
		{ 378, 340 },
		{ 563, 507 },
		{ 643, 585 },
		{ 565, 509 },
		{ 468, 420 },
		{ 1343, 1470 },
		{ 291, 252 },
		{ 908, 866 },
		{ 764, 710 },
		{ 1343, 1473 },
		{ 1343, 1476 },
		{ 1343, 1479 },
		{ 1343, 1482 },
		{ 1343, 1485 },
		{ 342, 300 },
		{ 913, 874 },
		{ 918, 880 },
		{ 666, 608 },
		{ 922, 884 },
		{ 384, 346 },
		{ 1818, 1800 },
		{ 356, 317 },
		{ 251, 216 },
		{ 1343, 1503 },
		{ 931, 893 },
		{ 772, 718 },
		{ 778, 726 },
		{ 780, 728 },
		{ 363, 324 },
		{ 364, 326 },
		{ 783, 732 },
		{ 1833, 1817 },
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
		{ 1860, 1848 },
		{ 1861, 1849 },
		{ 214, 186 },
		{ 349, 309 },
		{ 322, 281 },
		{ 605, 546 },
		{ 2010, 1992 },
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
		{ 2938, 2934 },
		{ 297, 257 },
		{ 253, 218 },
		{ 834, 789 },
		{ 479, 431 },
		{ 1967, 1946 },
		{ 280, 242 },
		{ 714, 660 },
		{ 562, 506 },
		{ 2028, 2011 },
		{ 339, 297 },
		{ 719, 665 },
		{ 1972, 1951 },
		{ 768, 714 },
		{ 628, 567 },
		{ 1817, 1799 },
		{ 722, 668 },
		{ 1148, 1147 },
		{ 564, 508 },
		{ 2043, 2027 },
		{ 916, 877 },
		{ 1056, 1035 },
		{ 1779, 1760 },
		{ 1057, 1036 },
		{ 725, 671 },
		{ 1782, 1763 },
		{ 1059, 1038 },
		{ 440, 394 },
		{ 340, 298 },
		{ 1835, 1819 },
		{ 678, 626 },
		{ 597, 538 },
		{ 1074, 1058 },
		{ 548, 493 },
		{ 1943, 1920 },
		{ 489, 437 },
		{ 869, 824 },
		{ 2917, 2915 },
		{ 647, 589 },
		{ 937, 899 },
		{ 1848, 1834 },
		{ 2335, 2312 },
		{ 2007, 1989 },
		{ 648, 590 },
		{ 1091, 1080 },
		{ 1013, 982 },
		{ 649, 591 },
		{ 268, 232 },
		{ 265, 229 },
		{ 283, 244 },
		{ 712, 659 },
		{ 1115, 1106 },
		{ 1871, 1860 },
		{ 1126, 1117 },
		{ 1127, 1118 },
		{ 1034, 1007 },
		{ 1132, 1123 },
		{ 2076, 2057 },
		{ 555, 500 },
		{ 1140, 1135 },
		{ 1141, 1136 },
		{ 899, 855 },
		{ 954, 920 },
		{ 1040, 1014 },
		{ 224, 191 },
		{ 1811, 1792 },
		{ 841, 796 },
		{ 753, 699 },
		{ 256, 220 },
		{ 1047, 1023 },
		{ 1760, 1736 },
		{ 449, 400 },
		{ 1762, 1738 },
		{ 665, 607 },
		{ 912, 873 },
		{ 418, 378 },
		{ 668, 610 },
		{ 522, 468 },
		{ 728, 674 },
		{ 986, 955 },
		{ 809, 761 },
		{ 1946, 1924 },
		{ 459, 411 },
		{ 1950, 1926 },
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
		{ 1847, 1833 },
		{ 827, 782 },
		{ 938, 900 },
		{ 2038, 2024 },
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
		{ 1831, 1815 },
		{ 421, 381 },
		{ 886, 841 },
		{ 1065, 1047 },
		{ 1070, 1053 },
		{ 2022, 2005 },
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
		{ 2915, 2911 },
		{ 848, 803 },
		{ 1969, 1948 },
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
		{ 2008, 1990 },
		{ 2009, 1991 },
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
		{ 1993, 1972 },
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
		{ 1801, 1782 },
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
		{ 1331, 1331 },
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
		{ 1331, 1331 },
		{ 891, 846 },
		{ 966, 933 },
		{ 967, 934 },
		{ 892, 847 },
		{ 1136, 1129 },
		{ 711, 658 },
		{ 547, 492 },
		{ 575, 518 },
		{ 900, 856 },
		{ 1832, 1816 },
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
		{ 1990, 1969 },
		{ 1991, 1970 },
		{ 1077, 1061 },
		{ 367, 329 },
		{ 1079, 1063 },
		{ 592, 533 },
		{ 469, 421 },
		{ 794, 747 },
		{ 1007, 976 },
		{ 1343, 1331 },
		{ 859, 814 },
		{ 353, 314 },
		{ 796, 749 },
		{ 595, 536 },
		{ 936, 898 },
		{ 2868, 2865 },
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
		{ 2023, 2006 },
		{ 309, 269 },
		{ 2025, 2008 },
		{ 2026, 2009 },
		{ 1117, 1108 },
		{ 1118, 1109 },
		{ 386, 348 },
		{ 805, 758 },
		{ 994, 962 },
		{ 995, 963 },
		{ 1025, 998 },
		{ 932, 894 },
		{ 1797, 1778 },
		{ 1874, 1865 },
		{ 1987, 1966 },
		{ 1597, 1594 },
		{ 849, 804 },
		{ 2091, 2078 },
		{ 533, 478 },
		{ 652, 594 },
		{ 1621, 1618 },
		{ 320, 279 },
		{ 756, 702 },
		{ 664, 606 },
		{ 1706, 1704 },
		{ 1571, 1568 },
		{ 1856, 1843 },
		{ 2049, 2034 },
		{ 1647, 1644 },
		{ 1248, 1245 },
		{ 1080, 1064 },
		{ 1543, 1541 },
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
		{ 1951, 1927 },
		{ 248, 213 },
		{ 1128, 1119 },
		{ 1041, 1015 },
		{ 300, 260 },
		{ 2051, 2037 },
		{ 2053, 2039 },
		{ 2054, 2040 },
		{ 2055, 2041 },
		{ 2056, 2042 },
		{ 598, 539 },
		{ 810, 763 },
		{ 812, 766 },
		{ 961, 928 },
		{ 965, 932 },
		{ 302, 262 },
		{ 1815, 1797 },
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
		{ 1992, 1971 },
		{ 615, 556 },
		{ 1849, 1835 },
		{ 663, 605 },
		{ 423, 383 },
		{ 914, 875 },
		{ 1763, 1739 },
		{ 213, 185 },
		{ 1858, 1846 },
		{ 263, 227 },
		{ 1001, 969 },
		{ 917, 878 },
		{ 775, 721 },
		{ 2005, 1987 },
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
		{ 1800, 1781 },
		{ 803, 756 },
		{ 267, 231 },
		{ 316, 275 },
		{ 2044, 2028 },
		{ 2118, 2118 },
		{ 2118, 2118 },
		{ 1867, 1867 },
		{ 1867, 1867 },
		{ 1869, 1869 },
		{ 1869, 1869 },
		{ 1480, 1480 },
		{ 1480, 1480 },
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
		{ 2074, 2074 },
		{ 2074, 2074 },
		{ 1844, 1844 },
		{ 1844, 1844 },
		{ 864, 819 },
		{ 2118, 2118 },
		{ 448, 399 },
		{ 1867, 1867 },
		{ 301, 261 },
		{ 1869, 1869 },
		{ 535, 480 },
		{ 1480, 1480 },
		{ 651, 593 },
		{ 2064, 2064 },
		{ 515, 458 },
		{ 2066, 2066 },
		{ 269, 233 },
		{ 2068, 2068 },
		{ 638, 581 },
		{ 2070, 2070 },
		{ 621, 561 },
		{ 2072, 2072 },
		{ 246, 212 },
		{ 2074, 2074 },
		{ 800, 753 },
		{ 1844, 1844 },
		{ 619, 560 },
		{ 620, 560 },
		{ 1879, 1879 },
		{ 1879, 1879 },
		{ 2089, 2089 },
		{ 2089, 2089 },
		{ 1538, 1538 },
		{ 1538, 1538 },
		{ 2119, 2118 },
		{ 518, 462 },
		{ 1868, 1867 },
		{ 306, 266 },
		{ 1870, 1869 },
		{ 392, 355 },
		{ 1481, 1480 },
		{ 2143, 2141 },
		{ 2065, 2064 },
		{ 543, 488 },
		{ 2067, 2066 },
		{ 286, 247 },
		{ 2069, 2068 },
		{ 1015, 984 },
		{ 2071, 2070 },
		{ 717, 663 },
		{ 2073, 2072 },
		{ 1879, 1879 },
		{ 2075, 2074 },
		{ 2089, 2089 },
		{ 1845, 1844 },
		{ 1538, 1538 },
		{ 1884, 1884 },
		{ 1884, 1884 },
		{ 2035, 2035 },
		{ 2035, 2035 },
		{ 2095, 2095 },
		{ 2095, 2095 },
		{ 1483, 1483 },
		{ 1483, 1483 },
		{ 1474, 1474 },
		{ 1474, 1474 },
		{ 1486, 1486 },
		{ 1486, 1486 },
		{ 1434, 1434 },
		{ 1434, 1434 },
		{ 1513, 1513 },
		{ 1513, 1513 },
		{ 1477, 1477 },
		{ 1477, 1477 },
		{ 1898, 1898 },
		{ 1898, 1898 },
		{ 1504, 1504 },
		{ 1504, 1504 },
		{ 928, 890 },
		{ 1884, 1884 },
		{ 1880, 1879 },
		{ 2035, 2035 },
		{ 2090, 2089 },
		{ 2095, 2095 },
		{ 1539, 1538 },
		{ 1483, 1483 },
		{ 863, 818 },
		{ 1474, 1474 },
		{ 1019, 989 },
		{ 1486, 1486 },
		{ 1020, 990 },
		{ 1434, 1434 },
		{ 885, 840 },
		{ 1513, 1513 },
		{ 758, 704 },
		{ 1477, 1477 },
		{ 1103, 1093 },
		{ 1898, 1898 },
		{ 697, 645 },
		{ 1504, 1504 },
		{ 1471, 1471 },
		{ 1471, 1471 },
		{ 1105, 1095 },
		{ 331, 289 },
		{ 699, 647 },
		{ 1840, 1825 },
		{ 843, 798 },
		{ 1000, 968 },
		{ 1885, 1884 },
		{ 817, 771 },
		{ 2036, 2035 },
		{ 1113, 1104 },
		{ 2096, 2095 },
		{ 2948, 2946 },
		{ 1484, 1483 },
		{ 1151, 1150 },
		{ 1475, 1474 },
		{ 791, 744 },
		{ 1487, 1486 },
		{ 895, 850 },
		{ 1435, 1434 },
		{ 1627, 1626 },
		{ 1514, 1513 },
		{ 1471, 1471 },
		{ 1478, 1477 },
		{ 1577, 1576 },
		{ 1899, 1898 },
		{ 897, 852 },
		{ 1505, 1504 },
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
		{ 1254, 1253 },
		{ 1465, 1447 },
		{ 1872, 1861 },
		{ 1549, 1548 },
		{ 726, 672 },
		{ 1820, 1803 },
		{ 1066, 1048 },
		{ 2918, 2916 },
		{ 943, 905 },
		{ 2013, 1996 },
		{ 1653, 1652 },
		{ 1472, 1471 },
		{ 838, 793 },
		{ 1603, 1602 },
		{ 1097, 1086 },
		{ 993, 961 },
		{ 1099, 1089 },
		{ 2077, 2058 },
		{ 1712, 1711 },
		{ 1043, 1018 },
		{ 1878, 1871 },
		{ 1859, 1847 },
		{ 1198, 1195 },
		{ 1119, 1110 },
		{ 578, 522 },
		{ 1945, 1923 },
		{ 2052, 2038 },
		{ 1957, 1934 },
		{ 221, 189 },
		{ 293, 254 },
		{ 466, 418 },
		{ 1759, 1735 },
		{ 2088, 2076 },
		{ 2761, 2759 },
		{ 2413, 2413 },
		{ 2413, 2413 },
		{ 2433, 2433 },
		{ 2433, 2433 },
		{ 1616, 1616 },
		{ 1616, 1616 },
		{ 2559, 2559 },
		{ 2559, 2559 },
		{ 1642, 1642 },
		{ 1642, 1642 },
		{ 2651, 2651 },
		{ 2651, 2651 },
		{ 2506, 2506 },
		{ 2506, 2506 },
		{ 2822, 2822 },
		{ 2822, 2822 },
		{ 2598, 2598 },
		{ 2598, 2598 },
		{ 2600, 2600 },
		{ 2600, 2600 },
		{ 2601, 2601 },
		{ 2601, 2601 },
		{ 731, 677 },
		{ 2413, 2413 },
		{ 604, 545 },
		{ 2433, 2433 },
		{ 361, 322 },
		{ 1616, 1616 },
		{ 422, 382 },
		{ 2559, 2559 },
		{ 896, 851 },
		{ 1642, 1642 },
		{ 646, 588 },
		{ 2651, 2651 },
		{ 784, 734 },
		{ 2506, 2506 },
		{ 1027, 1000 },
		{ 2822, 2822 },
		{ 317, 276 },
		{ 2598, 2598 },
		{ 252, 217 },
		{ 2600, 2600 },
		{ 738, 684 },
		{ 2601, 2601 },
		{ 964, 931 },
		{ 2602, 2602 },
		{ 2602, 2602 },
		{ 2603, 2603 },
		{ 2603, 2603 },
		{ 2440, 2413 },
		{ 739, 685 },
		{ 2434, 2433 },
		{ 2826, 2824 },
		{ 1617, 1616 },
		{ 571, 515 },
		{ 2577, 2559 },
		{ 262, 226 },
		{ 1643, 1642 },
		{ 693, 641 },
		{ 2652, 2651 },
		{ 792, 745 },
		{ 2528, 2506 },
		{ 365, 327 },
		{ 2823, 2822 },
		{ 695, 643 },
		{ 2612, 2598 },
		{ 194, 175 },
		{ 2613, 2600 },
		{ 2602, 2602 },
		{ 2614, 2601 },
		{ 2603, 2603 },
		{ 308, 268 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 2774, 2774 },
		{ 2774, 2774 },
		{ 1271, 1271 },
		{ 1271, 1271 },
		{ 2485, 2485 },
		{ 2485, 2485 },
		{ 2164, 2164 },
		{ 2164, 2164 },
		{ 2728, 2728 },
		{ 2728, 2728 },
		{ 2566, 2566 },
		{ 2566, 2566 },
		{ 2294, 2294 },
		{ 2294, 2294 },
		{ 2610, 2610 },
		{ 2610, 2610 },
		{ 1566, 1566 },
		{ 1566, 1566 },
		{ 2617, 2617 },
		{ 2617, 2617 },
		{ 2615, 2602 },
		{ 1228, 1228 },
		{ 2616, 2603 },
		{ 2774, 2774 },
		{ 519, 463 },
		{ 1271, 1271 },
		{ 1107, 1097 },
		{ 2485, 2485 },
		{ 225, 192 },
		{ 2164, 2164 },
		{ 853, 808 },
		{ 2728, 2728 },
		{ 408, 369 },
		{ 2566, 2566 },
		{ 464, 416 },
		{ 2294, 2294 },
		{ 1112, 1103 },
		{ 2610, 2610 },
		{ 550, 495 },
		{ 1566, 1566 },
		{ 752, 698 },
		{ 2617, 2617 },
		{ 285, 246 },
		{ 2684, 2684 },
		{ 2684, 2684 },
		{ 2385, 2385 },
		{ 2385, 2385 },
		{ 1229, 1228 },
		{ 324, 283 },
		{ 2775, 2774 },
		{ 270, 234 },
		{ 1272, 1271 },
		{ 707, 654 },
		{ 2507, 2485 },
		{ 708, 655 },
		{ 2165, 2164 },
		{ 923, 885 },
		{ 2729, 2728 },
		{ 355, 316 },
		{ 2584, 2566 },
		{ 2688, 2686 },
		{ 2295, 2294 },
		{ 1122, 1113 },
		{ 2623, 2610 },
		{ 313, 272 },
		{ 1567, 1566 },
		{ 2684, 2684 },
		{ 2625, 2617 },
		{ 2385, 2385 },
		{ 865, 820 },
		{ 2515, 2515 },
		{ 2515, 2515 },
		{ 1243, 1243 },
		{ 1243, 1243 },
		{ 1592, 1592 },
		{ 1592, 1592 },
		{ 2622, 2622 },
		{ 2622, 2622 },
		{ 2388, 2388 },
		{ 2388, 2388 },
		{ 1668, 1668 },
		{ 1668, 1668 },
		{ 1725, 1725 },
		{ 1725, 1725 },
		{ 2549, 2549 },
		{ 2549, 2549 },
		{ 2550, 2550 },
		{ 2550, 2550 },
		{ 2632, 2632 },
		{ 2632, 2632 },
		{ 2633, 2633 },
		{ 2633, 2633 },
		{ 2685, 2684 },
		{ 2515, 2515 },
		{ 2411, 2385 },
		{ 1243, 1243 },
		{ 811, 765 },
		{ 1592, 1592 },
		{ 1673, 1670 },
		{ 2622, 2622 },
		{ 2110, 2107 },
		{ 2388, 2388 },
		{ 501, 446 },
		{ 1668, 1668 },
		{ 2779, 2776 },
		{ 1725, 1725 },
		{ 813, 767 },
		{ 2549, 2549 },
		{ 240, 207 },
		{ 2550, 2550 },
		{ 503, 448 },
		{ 2632, 2632 },
		{ 667, 609 },
		{ 2633, 2633 },
		{ 303, 263 },
		{ 2634, 2634 },
		{ 2634, 2634 },
		{ 2496, 2496 },
		{ 2496, 2496 },
		{ 2536, 2515 },
		{ 669, 613 },
		{ 1244, 1243 },
		{ 304, 264 },
		{ 1593, 1592 },
		{ 345, 305 },
		{ 2630, 2622 },
		{ 632, 572 },
		{ 2414, 2388 },
		{ 633, 573 },
		{ 1669, 1668 },
		{ 939, 901 },
		{ 1726, 1725 },
		{ 634, 574 },
		{ 2567, 2549 },
		{ 379, 341 },
		{ 2568, 2550 },
		{ 880, 836 },
		{ 2637, 2632 },
		{ 2634, 2634 },
		{ 2638, 2633 },
		{ 2496, 2496 },
		{ 599, 540 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2863, 2863 },
		{ 2863, 2863 },
		{ 2585, 2585 },
		{ 2585, 2585 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2642, 2642 },
		{ 2642, 2642 },
		{ 1702, 1702 },
		{ 1702, 1702 },
		{ 2757, 2757 },
		{ 2757, 2757 },
		{ 1193, 1193 },
		{ 1193, 1193 },
		{ 637, 580 },
		{ 600, 541 },
		{ 828, 783 },
		{ 453, 405 },
		{ 887, 842 },
		{ 888, 843 },
		{ 2639, 2634 },
		{ 2341, 2341 },
		{ 2518, 2496 },
		{ 2863, 2863 },
		{ 602, 543 },
		{ 2585, 2585 },
		{ 537, 482 },
		{ 2409, 2409 },
		{ 1150, 1149 },
		{ 2642, 2642 },
		{ 952, 918 },
		{ 1702, 1702 },
		{ 730, 676 },
		{ 2757, 2757 },
		{ 1453, 1435 },
		{ 1193, 1193 },
		{ 1889, 1885 },
		{ 2979, 2979 },
		{ 1876, 1868 },
		{ 2987, 2987 },
		{ 2991, 2991 },
		{ 1501, 1487 },
		{ 2082, 2065 },
		{ 2102, 2096 },
		{ 1877, 1870 },
		{ 2083, 2067 },
		{ 1499, 1481 },
		{ 2342, 2341 },
		{ 2084, 2069 },
		{ 2864, 2863 },
		{ 1496, 1472 },
		{ 2599, 2585 },
		{ 2085, 2071 },
		{ 2437, 2409 },
		{ 1498, 1478 },
		{ 2644, 2642 },
		{ 2086, 2073 },
		{ 1703, 1702 },
		{ 1500, 1484 },
		{ 2758, 2757 },
		{ 2979, 2979 },
		{ 1194, 1193 },
		{ 2987, 2987 },
		{ 2991, 2991 },
		{ 2087, 2075 },
		{ 1857, 1845 },
		{ 1540, 1539 },
		{ 2050, 2036 },
		{ 1515, 1505 },
		{ 1900, 1899 },
		{ 1886, 1880 },
		{ 1521, 1514 },
		{ 1497, 1475 },
		{ 2121, 2119 },
		{ 2097, 2090 },
		{ 2895, 2894 },
		{ 1554, 1553 },
		{ 1555, 1554 },
		{ 1582, 1581 },
		{ 2953, 2952 },
		{ 2954, 2953 },
		{ 1717, 1716 },
		{ 1718, 1717 },
		{ 1583, 1582 },
		{ 1632, 1631 },
		{ 1633, 1632 },
		{ 2982, 2979 },
		{ 1260, 1259 },
		{ 2989, 2987 },
		{ 2992, 2991 },
		{ 1531, 1527 },
		{ 1608, 1607 },
		{ 1609, 1608 },
		{ 1259, 1258 },
		{ 1527, 1522 },
		{ 1528, 1523 },
		{ 1303, 1302 },
		{ 1658, 1657 },
		{ 1659, 1658 },
		{ 2658, 2658 },
		{ 2655, 2658 },
		{ 163, 163 },
		{ 160, 163 },
		{ 1829, 1814 },
		{ 1830, 1814 },
		{ 1851, 1839 },
		{ 1852, 1839 },
		{ 2131, 2131 },
		{ 1905, 1905 },
		{ 168, 164 },
		{ 2136, 2132 },
		{ 2657, 2653 },
		{ 1910, 1906 },
		{ 88, 70 },
		{ 167, 164 },
		{ 2135, 2132 },
		{ 2656, 2653 },
		{ 1909, 1906 },
		{ 87, 70 },
		{ 2664, 2661 },
		{ 2663, 2659 },
		{ 162, 158 },
		{ 2658, 2658 },
		{ 2189, 2166 },
		{ 163, 163 },
		{ 2662, 2659 },
		{ 161, 158 },
		{ 2666, 2665 },
		{ 2188, 2166 },
		{ 2240, 2214 },
		{ 2131, 2131 },
		{ 1905, 1905 },
		{ 1789, 1770 },
		{ 1911, 1908 },
		{ 1913, 1912 },
		{ 2137, 2134 },
		{ 2139, 2138 },
		{ 2659, 2658 },
		{ 1982, 1961 },
		{ 164, 163 },
		{ 119, 103 },
		{ 169, 166 },
		{ 171, 170 },
		{ 1980, 1959 },
		{ 1864, 1854 },
		{ 2132, 2131 },
		{ 1906, 1905 },
		{ 2665, 2663 },
		{ 2134, 2130 },
		{ 1912, 1910 },
		{ 1961, 1939 },
		{ 170, 168 },
		{ 2214, 2189 },
		{ 2138, 2136 },
		{ 1908, 1904 },
		{ 2661, 2657 },
		{ 166, 162 },
		{ 1770, 1748 },
		{ 103, 88 },
		{ 0, 2472 },
		{ 2507, 2507 },
		{ 2507, 2507 },
		{ 0, 2323 },
		{ 1913, 1913 },
		{ 1914, 1913 },
		{ 2599, 2599 },
		{ 2599, 2599 },
		{ 0, 2813 },
		{ 2666, 2666 },
		{ 2667, 2666 },
		{ 0, 1235 },
		{ 0, 2446 },
		{ 0, 1693 },
		{ 0, 2740 },
		{ 0, 1203 },
		{ 0, 2349 },
		{ 0, 2674 },
		{ 0, 2350 },
		{ 2518, 2518 },
		{ 2518, 2518 },
		{ 0, 2351 },
		{ 2295, 2295 },
		{ 2295, 2295 },
		{ 2507, 2507 },
		{ 0, 2827 },
		{ 0, 2453 },
		{ 1913, 1913 },
		{ 0, 2749 },
		{ 2599, 2599 },
		{ 0, 2153 },
		{ 0, 2831 },
		{ 2666, 2666 },
		{ 2614, 2614 },
		{ 2614, 2614 },
		{ 0, 1677 },
		{ 0, 2835 },
		{ 2139, 2139 },
		{ 2140, 2139 },
		{ 0, 2458 },
		{ 0, 2838 },
		{ 0, 2402 },
		{ 2518, 2518 },
		{ 2568, 2568 },
		{ 2568, 2568 },
		{ 2295, 2295 },
		{ 0, 2740 },
		{ 0, 2689 },
		{ 2528, 2528 },
		{ 2528, 2528 },
		{ 0, 2843 },
		{ 2623, 2623 },
		{ 2623, 2623 },
		{ 0, 2624 },
		{ 2625, 2625 },
		{ 2625, 2625 },
		{ 2614, 2614 },
		{ 0, 2694 },
		{ 0, 2765 },
		{ 0, 2267 },
		{ 2139, 2139 },
		{ 0, 2493 },
		{ 0, 2461 },
		{ 0, 1180 },
		{ 2630, 2630 },
		{ 2630, 2630 },
		{ 2568, 2568 },
		{ 0, 2854 },
		{ 0, 2359 },
		{ 0, 2575 },
		{ 0, 2383 },
		{ 2528, 2528 },
		{ 171, 171 },
		{ 172, 171 },
		{ 2623, 2623 },
		{ 0, 2705 },
		{ 0, 2359 },
		{ 2625, 2625 },
		{ 2577, 2577 },
		{ 2577, 2577 },
		{ 2434, 2434 },
		{ 2434, 2434 },
		{ 0, 2780 },
		{ 2637, 2637 },
		{ 2637, 2637 },
		{ 2638, 2638 },
		{ 2638, 2638 },
		{ 2630, 2630 },
		{ 2639, 2639 },
		{ 2639, 2639 },
		{ 0, 2267 },
		{ 0, 2784 },
		{ 0, 2500 },
		{ 0, 1688 },
		{ 0, 2712 },
		{ 171, 171 },
		{ 0, 2540 },
		{ 0, 2583 },
		{ 0, 2874 },
		{ 2644, 2644 },
		{ 2644, 2644 },
		{ 2577, 2577 },
		{ 0, 1219 },
		{ 2434, 2434 },
		{ 2584, 2584 },
		{ 2584, 2584 },
		{ 2637, 2637 },
		{ 0, 2586 },
		{ 2638, 2638 },
		{ 0, 2794 },
		{ 0, 2719 },
		{ 2639, 2639 },
		{ 0, 1214 },
		{ 0, 2797 },
		{ 0, 2588 },
		{ 0, 2589 },
		{ 0, 2885 },
		{ 0, 2543 },
		{ 2652, 2652 },
		{ 2652, 2652 },
		{ 0, 2802 },
		{ 0, 2470 },
		{ 2644, 2644 },
		{ 2411, 2411 },
		{ 2411, 2411 },
		{ 1159, 1159 },
		{ 1290, 1289 },
		{ 2584, 2584 },
		{ 0, 1938 },
		{ 2660, 2656 },
		{ 1907, 1909 },
		{ 165, 167 },
		{ 1907, 1903 },
		{ 0, 2188 },
		{ 2660, 2662 },
		{ 2133, 2135 },
		{ 0, 87 },
		{ 1808, 1789 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2652, 2652 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2411, 2411 },
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
		{ -1175, 3063, 139 },
		{ 11, 0, 139 },
		{ -1196, 3056, 147 },
		{ 13, 0, 147 },
		{ -1231, 3194, 0 },
		{ 15, 0, 0 },
		{ -1246, 3079, 135 },
		{ 17, 0, 135 },
		{ -1274, 3064, 22 },
		{ 19, 0, 22 },
		{ -1316, 230, 0 },
		{ 21, 0, 0 },
		{ -1542, 3192, 0 },
		{ 23, 0, 0 },
		{ -1569, 3057, 0 },
		{ 25, 0, 0 },
		{ -1595, 3083, 0 },
		{ 27, 0, 0 },
		{ -1619, 3075, 0 },
		{ 29, 0, 0 },
		{ -1645, 3058, 0 },
		{ 31, 0, 0 },
		{ -1671, 3069, 151 },
		{ 33, 0, 151 },
		{ -1705, 3190, 258 },
		{ 35, 0, 258 },
		{ 38, 127, 0 },
		{ -1741, 344, 0 },
		{ 40, 16, 0 },
		{ -1930, 116, 0 },
		{ -2142, 3188, 0 },
		{ 41, 0, 0 },
		{ 44, 14, 0 },
		{ -86, 458, 0 },
		{ -2669, 3062, 143 },
		{ 45, 0, 143 },
		{ -2687, 3193, 166 },
		{ 47, 0, 166 },
		{ 2731, 1503, 0 },
		{ 49, 0, 0 },
		{ -2733, 3191, 264 },
		{ 51, 0, 264 },
		{ -2760, 3187, 169 },
		{ 53, 0, 169 },
		{ -2778, 3073, 162 },
		{ 55, 0, 162 },
		{ -2825, 3189, 155 },
		{ 57, 0, 155 },
		{ -2866, 3081, 161 },
		{ 59, 0, 161 },
		{ -86, 1, 0 },
		{ 61, 0, 0 },
		{ -2914, 1765, 0 },
		{ 63, 0, 0 },
		{ -2924, 1674, 42 },
		{ 65, 0, 42 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 413 },
		{ 2665, 4557, 420 },
		{ 0, 0, 236 },
		{ 0, 0, 238 },
		{ 157, 1270, 255 },
		{ 157, 1383, 255 },
		{ 157, 1279, 255 },
		{ 157, 1286, 255 },
		{ 157, 1286, 255 },
		{ 157, 1290, 255 },
		{ 157, 1295, 255 },
		{ 157, 641, 255 },
		{ 2973, 2800, 420 },
		{ 157, 652, 255 },
		{ 2973, 1619, 254 },
		{ 102, 2462, 420 },
		{ 157, 0, 255 },
		{ 0, 0, 420 },
		{ -87, 4771, 232 },
		{ -88, 4602, 0 },
		{ 157, 643, 255 },
		{ 157, 677, 255 },
		{ 157, 646, 255 },
		{ 157, 660, 255 },
		{ 157, 678, 255 },
		{ 157, 678, 255 },
		{ 157, 685, 255 },
		{ 157, 700, 255 },
		{ 157, 693, 255 },
		{ 2942, 2278, 0 },
		{ 157, 683, 255 },
		{ 2973, 1751, 251 },
		{ 117, 1385, 0 },
		{ 2973, 1741, 252 },
		{ 2665, 4579, 0 },
		{ 157, 692, 255 },
		{ 157, 716, 255 },
		{ 157, 717, 255 },
		{ 157, 713, 255 },
		{ 157, 0, 243 },
		{ 157, 732, 255 },
		{ 157, 735, 255 },
		{ 157, 755, 255 },
		{ 157, 760, 255 },
		{ 2970, 2858, 0 },
		{ 157, 767, 255 },
		{ 131, 1419, 0 },
		{ 117, 0, 0 },
		{ 2899, 2630, 253 },
		{ 133, 1436, 0 },
		{ 0, 0, 234 },
		{ 157, 771, 239 },
		{ 157, 773, 255 },
		{ 157, 765, 255 },
		{ 157, 770, 255 },
		{ 157, 794, 255 },
		{ 157, 787, 255 },
		{ 157, 0, 246 },
		{ 157, 788, 255 },
		{ 0, 0, 248 },
		{ 157, 794, 255 },
		{ 131, 0, 0 },
		{ 2899, 2517, 251 },
		{ 133, 0, 0 },
		{ 2899, 2586, 252 },
		{ 157, 810, 255 },
		{ 157, 835, 255 },
		{ 157, 836, 255 },
		{ 157, 862, 255 },
		{ 157, 930, 255 },
		{ 157, 0, 245 },
		{ 157, 976, 255 },
		{ 157, 980, 255 },
		{ 157, 1001, 255 },
		{ 157, 0, 241 },
		{ 157, 1118, 255 },
		{ 157, 0, 242 },
		{ 157, 0, 244 },
		{ 157, 1192, 255 },
		{ 157, 1220, 255 },
		{ 157, 0, 240 },
		{ 157, 1268, 255 },
		{ 157, 0, 247 },
		{ 157, 667, 255 },
		{ 157, 1285, 255 },
		{ 0, 0, 250 },
		{ 157, 1268, 255 },
		{ 157, 1269, 255 },
		{ 2990, 1337, 249 },
		{ 2665, 4565, 420 },
		{ 163, 0, 236 },
		{ 0, 0, 237 },
		{ -161, 20, 232 },
		{ -162, 4600, 0 },
		{ 2945, 4578, 0 },
		{ 2665, 4553, 0 },
		{ 0, 0, 233 },
		{ 2665, 4580, 0 },
		{ -167, 4766, 0 },
		{ -168, 4595, 0 },
		{ 171, 0, 234 },
		{ 2665, 4581, 0 },
		{ 2945, 4708, 0 },
		{ 0, 0, 235 },
		{ 2946, 1554, 133 },
		{ 2042, 3955, 133 },
		{ 2824, 4269, 133 },
		{ 2946, 4158, 133 },
		{ 0, 0, 133 },
		{ 2934, 3305, 0 },
		{ 2057, 2917, 0 },
		{ 2934, 3499, 0 },
		{ 2911, 3246, 0 },
		{ 2911, 3245, 0 },
		{ 2042, 4000, 0 },
		{ 2938, 3160, 0 },
		{ 2042, 3965, 0 },
		{ 2912, 3477, 0 },
		{ 2941, 3174, 0 },
		{ 2912, 3214, 0 },
		{ 2759, 4196, 0 },
		{ 2938, 3189, 0 },
		{ 2057, 3580, 0 },
		{ 2824, 4305, 0 },
		{ 1988, 3375, 0 },
		{ 1988, 3351, 0 },
		{ 2010, 3090, 0 },
		{ 1991, 3692, 0 },
		{ 1972, 3746, 0 },
		{ 1991, 3703, 0 },
		{ 2010, 3092, 0 },
		{ 2010, 3005, 0 },
		{ 2938, 3220, 0 },
		{ 2865, 3837, 0 },
		{ 2042, 3987, 0 },
		{ 1128, 3890, 0 },
		{ 1988, 3364, 0 },
		{ 2010, 3014, 0 },
		{ 2824, 4385, 0 },
		{ 1988, 3389, 0 },
		{ 2911, 3654, 0 },
		{ 2934, 3510, 0 },
		{ 2057, 3606, 0 },
		{ 2141, 4047, 0 },
		{ 2824, 3917, 0 },
		{ 2865, 3803, 0 },
		{ 1972, 3719, 0 },
		{ 2912, 3452, 0 },
		{ 2824, 4243, 0 },
		{ 2934, 3517, 0 },
		{ 2865, 3499, 0 },
		{ 2057, 3584, 0 },
		{ 2010, 3016, 0 },
		{ 2941, 3323, 0 },
		{ 2911, 3678, 0 },
		{ 1950, 3182, 0 },
		{ 1972, 3742, 0 },
		{ 2824, 4259, 0 },
		{ 2042, 3967, 0 },
		{ 2042, 3983, 0 },
		{ 2934, 3563, 0 },
		{ 2010, 3031, 0 },
		{ 2042, 4003, 0 },
		{ 2934, 3562, 0 },
		{ 2141, 4041, 0 },
		{ 2824, 4327, 0 },
		{ 2941, 3331, 0 },
		{ 2912, 3429, 0 },
		{ 2010, 3071, 0 },
		{ 2941, 3220, 0 },
		{ 2934, 3506, 0 },
		{ 1972, 3738, 0 },
		{ 2865, 3839, 0 },
		{ 2057, 3522, 0 },
		{ 1020, 3153, 0 },
		{ 2934, 3564, 0 },
		{ 2911, 3637, 0 },
		{ 2824, 4319, 0 },
		{ 2141, 4070, 0 },
		{ 2010, 3073, 0 },
		{ 2941, 3308, 0 },
		{ 2042, 3914, 0 },
		{ 1950, 3183, 0 },
		{ 2912, 3436, 0 },
		{ 2010, 3074, 0 },
		{ 2759, 4197, 0 },
		{ 2911, 3636, 0 },
		{ 2941, 3274, 0 },
		{ 2078, 3517, 0 },
		{ 2010, 3075, 0 },
		{ 2865, 3805, 0 },
		{ 2042, 3919, 0 },
		{ 2141, 4033, 0 },
		{ 2042, 3930, 0 },
		{ 2824, 4391, 0 },
		{ 2824, 4399, 0 },
		{ 1972, 3745, 0 },
		{ 2141, 4062, 0 },
		{ 2010, 3077, 0 },
		{ 2824, 4274, 0 },
		{ 2865, 3849, 0 },
		{ 1972, 3709, 0 },
		{ 2865, 3773, 0 },
		{ 2824, 4341, 0 },
		{ 1988, 3385, 0 },
		{ 2912, 3473, 0 },
		{ 2042, 4004, 0 },
		{ 2824, 4241, 0 },
		{ 1128, 3904, 0 },
		{ 1020, 3145, 0 },
		{ 2078, 3871, 0 },
		{ 1991, 3685, 0 },
		{ 2912, 3479, 0 },
		{ 2010, 3079, 0 },
		{ 2824, 4325, 0 },
		{ 2042, 3974, 0 },
		{ 2010, 3081, 0 },
		{ 0, 0, 69 },
		{ 2934, 3290, 0 },
		{ 2042, 3997, 0 },
		{ 2946, 4129, 0 },
		{ 2010, 3082, 0 },
		{ 2941, 3306, 0 },
		{ 1988, 3346, 0 },
		{ 1972, 3747, 0 },
		{ 2941, 3307, 0 },
		{ 2010, 3083, 0 },
		{ 2042, 3950, 0 },
		{ 2934, 3525, 0 },
		{ 2934, 3543, 0 },
		{ 1991, 3682, 0 },
		{ 2912, 3444, 0 },
		{ 802, 3164, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 1988, 3381, 0 },
		{ 2824, 4401, 0 },
		{ 2865, 3806, 0 },
		{ 1972, 3713, 0 },
		{ 2941, 3330, 0 },
		{ 2912, 3478, 0 },
		{ 0, 0, 67 },
		{ 2010, 3084, 0 },
		{ 1988, 3334, 0 },
		{ 2941, 3332, 0 },
		{ 2865, 3827, 0 },
		{ 2941, 3249, 0 },
		{ 2824, 4335, 0 },
		{ 2912, 3451, 0 },
		{ 1128, 3891, 0 },
		{ 1988, 3363, 0 },
		{ 2911, 3653, 0 },
		{ 2042, 3984, 0 },
		{ 2824, 4229, 0 },
		{ 2946, 4160, 0 },
		{ 2912, 3458, 0 },
		{ 0, 0, 63 },
		{ 2912, 3459, 0 },
		{ 2824, 4265, 0 },
		{ 1128, 3909, 0 },
		{ 2865, 3819, 0 },
		{ 0, 0, 72 },
		{ 2941, 3273, 0 },
		{ 2934, 3504, 0 },
		{ 2010, 3085, 0 },
		{ 2865, 3844, 0 },
		{ 2042, 3936, 0 },
		{ 2010, 3086, 0 },
		{ 1988, 3384, 0 },
		{ 2911, 3629, 0 },
		{ 2941, 3277, 0 },
		{ 2912, 3430, 0 },
		{ 2824, 4411, 0 },
		{ 2010, 3087, 0 },
		{ 2865, 3834, 0 },
		{ 2042, 3985, 0 },
		{ 2941, 3302, 0 },
		{ 2912, 3449, 0 },
		{ 2865, 3843, 0 },
		{ 963, 3855, 0 },
		{ 0, 0, 8 },
		{ 1988, 3345, 0 },
		{ 1991, 3693, 0 },
		{ 2865, 3802, 0 },
		{ 2023, 3124, 0 },
		{ 2010, 3088, 0 },
		{ 2141, 4064, 0 },
		{ 2042, 3943, 0 },
		{ 1988, 3361, 0 },
		{ 2042, 3951, 0 },
		{ 1991, 3686, 0 },
		{ 2010, 3089, 0 },
		{ 2941, 3316, 0 },
		{ 2941, 3172, 0 },
		{ 2042, 3981, 0 },
		{ 2938, 3209, 0 },
		{ 2912, 3426, 0 },
		{ 1128, 3889, 0 },
		{ 2911, 3644, 0 },
		{ 2057, 2986, 0 },
		{ 2824, 4309, 0 },
		{ 1128, 3899, 0 },
		{ 2946, 3151, 0 },
		{ 2023, 3129, 0 },
		{ 1991, 3705, 0 },
		{ 1988, 3392, 0 },
		{ 2941, 3252, 0 },
		{ 0, 0, 113 },
		{ 2010, 3091, 0 },
		{ 2057, 3591, 0 },
		{ 1949, 3158, 0 },
		{ 2934, 3513, 0 },
		{ 2911, 3657, 0 },
		{ 2824, 4231, 0 },
		{ 2042, 3962, 0 },
		{ 0, 0, 7 },
		{ 1991, 3706, 0 },
		{ 0, 0, 6 },
		{ 2865, 3845, 0 },
		{ 0, 0, 118 },
		{ 2911, 3668, 0 },
		{ 2042, 3980, 0 },
		{ 2946, 1612, 0 },
		{ 2010, 3093, 0 },
		{ 2911, 3634, 0 },
		{ 2934, 3542, 0 },
		{ 2010, 3094, 0 },
		{ 2042, 3988, 0 },
		{ 2946, 3145, 0 },
		{ 2042, 3998, 0 },
		{ 2141, 4031, 0 },
		{ 2057, 3587, 0 },
		{ 0, 0, 68 },
		{ 1972, 3752, 0 },
		{ 2010, 3095, 105 },
		{ 2010, 3096, 106 },
		{ 2824, 4438, 0 },
		{ 2912, 3484, 0 },
		{ 2934, 3514, 0 },
		{ 2912, 3423, 0 },
		{ 1128, 3887, 0 },
		{ 2938, 3236, 0 },
		{ 2057, 3598, 0 },
		{ 2865, 3804, 0 },
		{ 2042, 3957, 0 },
		{ 2010, 3097, 0 },
		{ 2941, 3320, 0 },
		{ 2824, 4311, 0 },
		{ 2865, 3812, 0 },
		{ 2759, 4198, 0 },
		{ 2865, 3813, 0 },
		{ 2912, 3434, 0 },
		{ 2865, 3822, 0 },
		{ 0, 0, 9 },
		{ 2010, 3098, 0 },
		{ 2865, 3832, 0 },
		{ 2023, 3131, 0 },
		{ 2078, 3882, 0 },
		{ 0, 0, 103 },
		{ 1988, 3338, 0 },
		{ 2911, 3676, 0 },
		{ 2934, 3502, 0 },
		{ 2057, 3520, 0 },
		{ 2911, 3152, 0 },
		{ 2865, 3847, 0 },
		{ 2938, 3202, 0 },
		{ 2865, 3801, 0 },
		{ 2938, 3181, 0 },
		{ 2934, 3550, 0 },
		{ 2042, 3940, 0 },
		{ 2941, 3250, 0 },
		{ 2912, 3470, 0 },
		{ 2938, 3169, 0 },
		{ 2911, 3662, 0 },
		{ 2941, 3264, 0 },
		{ 2865, 3774, 0 },
		{ 2941, 3269, 0 },
		{ 2824, 4379, 0 },
		{ 2010, 3099, 0 },
		{ 2824, 4387, 0 },
		{ 2912, 3482, 0 },
		{ 2042, 3978, 0 },
		{ 2934, 3495, 0 },
		{ 2934, 3497, 0 },
		{ 1972, 3733, 0 },
		{ 2010, 3100, 93 },
		{ 2912, 3417, 0 },
		{ 2010, 3101, 0 },
		{ 2010, 3102, 0 },
		{ 2141, 4039, 0 },
		{ 2010, 3103, 0 },
		{ 1988, 3393, 0 },
		{ 0, 0, 102 },
		{ 2141, 4060, 0 },
		{ 2824, 4301, 0 },
		{ 2941, 3295, 0 },
		{ 2941, 3298, 0 },
		{ 0, 0, 115 },
		{ 0, 0, 117 },
		{ 2057, 3593, 0 },
		{ 1988, 3339, 0 },
		{ 2042, 3256, 0 },
		{ 2941, 3303, 0 },
		{ 2042, 3933, 0 },
		{ 2010, 3104, 0 },
		{ 2042, 3937, 0 },
		{ 2865, 3814, 0 },
		{ 2911, 3639, 0 },
		{ 2010, 2983, 0 },
		{ 2078, 3868, 0 },
		{ 2938, 3229, 0 },
		{ 2141, 4035, 0 },
		{ 2010, 2992, 0 },
		{ 2824, 4447, 0 },
		{ 1988, 3366, 0 },
		{ 895, 3762, 0 },
		{ 2941, 3318, 0 },
		{ 2911, 3666, 0 },
		{ 2057, 3618, 0 },
		{ 2141, 4068, 0 },
		{ 2941, 3319, 0 },
		{ 1950, 3179, 0 },
		{ 2010, 3000, 0 },
		{ 2865, 3793, 0 },
		{ 2934, 3548, 0 },
		{ 1988, 3388, 0 },
		{ 2824, 4315, 0 },
		{ 2941, 3329, 0 },
		{ 2057, 3600, 0 },
		{ 2023, 3132, 0 },
		{ 2912, 3425, 0 },
		{ 2057, 3574, 0 },
		{ 1991, 3700, 0 },
		{ 2946, 3221, 0 },
		{ 2010, 3002, 0 },
		{ 0, 0, 64 },
		{ 2010, 3003, 0 },
		{ 2934, 3523, 0 },
		{ 2912, 3431, 0 },
		{ 2934, 3533, 0 },
		{ 2912, 3433, 0 },
		{ 2010, 3004, 107 },
		{ 2057, 3616, 0 },
		{ 1991, 3702, 0 },
		{ 1988, 3340, 0 },
		{ 1988, 3344, 0 },
		{ 2824, 4257, 0 },
		{ 2934, 3493, 0 },
		{ 2938, 3234, 0 },
		{ 2865, 3794, 0 },
		{ 2941, 3256, 0 },
		{ 1988, 3347, 0 },
		{ 0, 0, 119 },
		{ 2759, 4192, 0 },
		{ 1991, 3691, 0 },
		{ 2941, 3260, 0 },
		{ 2911, 3663, 0 },
		{ 0, 0, 114 },
		{ 0, 0, 104 },
		{ 1988, 3360, 0 },
		{ 2912, 3468, 0 },
		{ 2941, 3263, 0 },
		{ 2057, 2904, 0 },
		{ 2946, 3181, 0 },
		{ 2865, 3821, 0 },
		{ 2911, 3630, 0 },
		{ 2010, 3006, 0 },
		{ 2865, 3829, 0 },
		{ 1972, 3716, 0 },
		{ 2934, 3546, 0 },
		{ 2042, 3925, 0 },
		{ 2824, 4418, 0 },
		{ 2824, 4436, 0 },
		{ 1988, 3367, 0 },
		{ 2824, 4445, 0 },
		{ 2865, 3838, 0 },
		{ 2824, 4227, 0 },
		{ 2912, 3480, 0 },
		{ 2911, 3643, 0 },
		{ 2010, 3007, 0 },
		{ 2042, 3941, 0 },
		{ 2912, 3483, 0 },
		{ 2010, 3008, 0 },
		{ 2912, 3485, 0 },
		{ 2042, 3952, 0 },
		{ 2865, 3785, 0 },
		{ 2912, 3411, 0 },
		{ 2042, 3959, 0 },
		{ 1988, 3382, 0 },
		{ 2911, 3664, 0 },
		{ 2010, 3009, 0 },
		{ 2946, 4053, 0 },
		{ 2141, 4045, 0 },
		{ 2042, 3973, 0 },
		{ 1991, 3689, 0 },
		{ 2042, 3977, 0 },
		{ 1991, 3690, 0 },
		{ 2934, 3501, 0 },
		{ 2934, 3529, 0 },
		{ 0, 0, 95 },
		{ 2865, 3807, 0 },
		{ 2865, 3809, 0 },
		{ 2010, 3012, 0 },
		{ 2824, 4403, 0 },
		{ 2824, 4405, 0 },
		{ 2824, 4409, 0 },
		{ 1991, 3697, 0 },
		{ 1988, 3387, 0 },
		{ 0, 0, 122 },
		{ 0, 0, 116 },
		{ 0, 0, 120 },
		{ 2824, 4435, 0 },
		{ 2141, 4043, 0 },
		{ 1020, 3146, 0 },
		{ 2010, 3013, 0 },
		{ 2865, 2989, 0 },
		{ 2912, 3432, 0 },
		{ 1991, 3681, 0 },
		{ 1128, 3893, 0 },
		{ 2824, 4235, 0 },
		{ 2934, 3553, 0 },
		{ 2934, 3558, 0 },
		{ 2934, 3561, 0 },
		{ 2911, 3645, 0 },
		{ 2141, 4037, 0 },
		{ 2078, 3869, 0 },
		{ 2911, 3652, 0 },
		{ 2938, 3235, 0 },
		{ 1972, 3755, 0 },
		{ 1128, 3896, 0 },
		{ 2941, 3304, 0 },
		{ 1972, 3710, 0 },
		{ 1988, 3337, 0 },
		{ 2010, 3015, 0 },
		{ 1991, 3698, 0 },
		{ 1972, 3729, 0 },
		{ 2042, 3961, 0 },
		{ 2078, 3873, 0 },
		{ 2057, 3589, 0 },
		{ 2912, 3447, 0 },
		{ 2824, 4389, 0 },
		{ 2057, 3592, 0 },
		{ 0, 0, 57 },
		{ 0, 0, 58 },
		{ 2824, 4397, 0 },
		{ 0, 0, 66 },
		{ 0, 0, 111 },
		{ 1950, 3184, 0 },
		{ 2938, 3212, 0 },
		{ 1988, 3341, 0 },
		{ 2938, 3215, 0 },
		{ 2941, 3317, 0 },
		{ 2865, 3810, 0 },
		{ 2912, 3465, 0 },
		{ 0, 0, 97 },
		{ 2912, 3466, 0 },
		{ 0, 0, 99 },
		{ 2934, 3545, 0 },
		{ 2912, 3467, 0 },
		{ 2042, 3995, 0 },
		{ 2023, 3134, 0 },
		{ 2023, 3135, 0 },
		{ 2078, 3629, 0 },
		{ 2912, 3471, 0 },
		{ 895, 3764, 0 },
		{ 1972, 3740, 0 },
		{ 0, 0, 112 },
		{ 0, 0, 121 },
		{ 2912, 3472, 0 },
		{ 0, 0, 132 },
		{ 1988, 3349, 0 },
		{ 2946, 3781, 0 },
		{ 2824, 4261, 0 },
		{ 1128, 3906, 0 },
		{ 2824, 4267, 0 },
		{ 2042, 3935, 0 },
		{ 2946, 4124, 0 },
		{ 2912, 3474, 0 },
		{ 2946, 4130, 0 },
		{ 2938, 3230, 0 },
		{ 2938, 3233, 0 },
		{ 2911, 2985, 0 },
		{ 2010, 3017, 0 },
		{ 2042, 3949, 0 },
		{ 2865, 3781, 0 },
		{ 2824, 4329, 0 },
		{ 2824, 4331, 0 },
		{ 2865, 3784, 0 },
		{ 2057, 3620, 0 },
		{ 2865, 3792, 0 },
		{ 2057, 3566, 0 },
		{ 2057, 3523, 0 },
		{ 2865, 3799, 0 },
		{ 2010, 3018, 0 },
		{ 2141, 4074, 0 },
		{ 2010, 3019, 0 },
		{ 2934, 3526, 0 },
		{ 2010, 3020, 0 },
		{ 1991, 3695, 0 },
		{ 2934, 3531, 0 },
		{ 1972, 3743, 0 },
		{ 2010, 3021, 0 },
		{ 2934, 3539, 0 },
		{ 2946, 4171, 0 },
		{ 1128, 3910, 0 },
		{ 2057, 3594, 0 },
		{ 2912, 3406, 0 },
		{ 2824, 4453, 0 },
		{ 2824, 4225, 0 },
		{ 2042, 3986, 0 },
		{ 1991, 3704, 0 },
		{ 2912, 3407, 0 },
		{ 2042, 3989, 0 },
		{ 2042, 3991, 0 },
		{ 2042, 3992, 0 },
		{ 2824, 4245, 0 },
		{ 2824, 4253, 0 },
		{ 2042, 3993, 0 },
		{ 2010, 3022, 0 },
		{ 2941, 3253, 0 },
		{ 2941, 3254, 0 },
		{ 2042, 3999, 0 },
		{ 1972, 3718, 0 },
		{ 2938, 3227, 0 },
		{ 1972, 3723, 0 },
		{ 2946, 4161, 0 },
		{ 2941, 3258, 0 },
		{ 2010, 3023, 61 },
		{ 2941, 3262, 0 },
		{ 2824, 4317, 0 },
		{ 2057, 3583, 0 },
		{ 2010, 3026, 0 },
		{ 2010, 3029, 0 },
		{ 2078, 3872, 0 },
		{ 2865, 3846, 0 },
		{ 2946, 4120, 0 },
		{ 2911, 3623, 0 },
		{ 2941, 3265, 0 },
		{ 2941, 3266, 0 },
		{ 1020, 3140, 0 },
		{ 1972, 3754, 0 },
		{ 2912, 3438, 0 },
		{ 2023, 3127, 0 },
		{ 1950, 3190, 0 },
		{ 1950, 3178, 0 },
		{ 2934, 3528, 0 },
		{ 1988, 3343, 0 },
		{ 1128, 3903, 0 },
		{ 2938, 3210, 0 },
		{ 2912, 3455, 0 },
		{ 2946, 4162, 0 },
		{ 2946, 4163, 0 },
		{ 2042, 3970, 0 },
		{ 0, 0, 62 },
		{ 0, 0, 60 },
		{ 1128, 3907, 0 },
		{ 1972, 3724, 0 },
		{ 2912, 3456, 0 },
		{ 1128, 3886, 0 },
		{ 2912, 3457, 0 },
		{ 0, 0, 108 },
		{ 2941, 3278, 0 },
		{ 2941, 3282, 0 },
		{ 2912, 3460, 0 },
		{ 0, 0, 101 },
		{ 2824, 4237, 0 },
		{ 0, 0, 109 },
		{ 0, 0, 110 },
		{ 2911, 3665, 0 },
		{ 895, 3766, 0 },
		{ 1991, 3694, 0 },
		{ 1128, 3902, 0 },
		{ 2941, 3283, 0 },
		{ 0, 0, 3 },
		{ 2042, 3990, 0 },
		{ 2946, 4143, 0 },
		{ 2824, 4263, 0 },
		{ 2911, 3667, 0 },
		{ 2865, 3823, 0 },
		{ 2941, 3284, 0 },
		{ 2865, 3828, 0 },
		{ 2042, 3996, 0 },
		{ 2010, 3030, 0 },
		{ 1991, 3701, 0 },
		{ 2141, 4049, 0 },
		{ 1988, 3352, 0 },
		{ 1988, 3353, 0 },
		{ 2042, 4002, 0 },
		{ 2911, 3625, 0 },
		{ 963, 3856, 0 },
		{ 2042, 2995, 0 },
		{ 2865, 3842, 0 },
		{ 2057, 3596, 0 },
		{ 0, 0, 70 },
		{ 2042, 3926, 0 },
		{ 0, 0, 78 },
		{ 2824, 4373, 0 },
		{ 2042, 3927, 0 },
		{ 2824, 4383, 0 },
		{ 2941, 3297, 0 },
		{ 2042, 3932, 0 },
		{ 2938, 3217, 0 },
		{ 2946, 4135, 0 },
		{ 2042, 3934, 0 },
		{ 2057, 3602, 0 },
		{ 1972, 3728, 0 },
		{ 2941, 3299, 0 },
		{ 2865, 3771, 0 },
		{ 2057, 3607, 0 },
		{ 2865, 3783, 0 },
		{ 2042, 3946, 0 },
		{ 0, 0, 65 },
		{ 2057, 3609, 0 },
		{ 2057, 3611, 0 },
		{ 2824, 4437, 0 },
		{ 2941, 3300, 0 },
		{ 2057, 3617, 0 },
		{ 2042, 3954, 0 },
		{ 2010, 3032, 0 },
		{ 2865, 3797, 0 },
		{ 2934, 3518, 0 },
		{ 1991, 3696, 0 },
		{ 1972, 3748, 0 },
		{ 1988, 3370, 0 },
		{ 2946, 4179, 0 },
		{ 1988, 3371, 0 },
		{ 2010, 3033, 0 },
		{ 2057, 3582, 0 },
		{ 1950, 3186, 0 },
		{ 2946, 4132, 0 },
		{ 2042, 3975, 0 },
		{ 2042, 3976, 0 },
		{ 802, 3160, 0 },
		{ 0, 3163, 0 },
		{ 2911, 3674, 0 },
		{ 2078, 3866, 0 },
		{ 2042, 3982, 0 },
		{ 2912, 3486, 0 },
		{ 1128, 3892, 0 },
		{ 2824, 4307, 0 },
		{ 2912, 3405, 0 },
		{ 2010, 3034, 0 },
		{ 2941, 3309, 0 },
		{ 2912, 3408, 0 },
		{ 1972, 3731, 0 },
		{ 2865, 3826, 0 },
		{ 2912, 3410, 0 },
		{ 2911, 3633, 0 },
		{ 2941, 3310, 0 },
		{ 2141, 4111, 0 },
		{ 2141, 4029, 0 },
		{ 2824, 4346, 0 },
		{ 2042, 3994, 0 },
		{ 1972, 3741, 0 },
		{ 2941, 3311, 0 },
		{ 2934, 3551, 0 },
		{ 2912, 3418, 0 },
		{ 2912, 3419, 0 },
		{ 2912, 3421, 0 },
		{ 2941, 3312, 0 },
		{ 2057, 3614, 0 },
		{ 1991, 3699, 0 },
		{ 1128, 3897, 0 },
		{ 2941, 3313, 0 },
		{ 1972, 3757, 0 },
		{ 1972, 3708, 0 },
		{ 0, 0, 10 },
		{ 2824, 4413, 0 },
		{ 1988, 3394, 0 },
		{ 2941, 3314, 0 },
		{ 2824, 3889, 0 },
		{ 2946, 4118, 0 },
		{ 2911, 3658, 0 },
		{ 2824, 4439, 0 },
		{ 2824, 4440, 0 },
		{ 2941, 3315, 0 },
		{ 2010, 3035, 0 },
		{ 2865, 3787, 0 },
		{ 2865, 3790, 0 },
		{ 2042, 3938, 0 },
		{ 2010, 3036, 0 },
		{ 2946, 4145, 0 },
		{ 2824, 4233, 0 },
		{ 2946, 4153, 0 },
		{ 1972, 3720, 0 },
		{ 0, 0, 79 },
		{ 2057, 3577, 0 },
		{ 2865, 3795, 0 },
		{ 0, 0, 77 },
		{ 2938, 3237, 0 },
		{ 1991, 3683, 0 },
		{ 0, 0, 80 },
		{ 2946, 4164, 0 },
		{ 2865, 3800, 0 },
		{ 2938, 3192, 0 },
		{ 2042, 3953, 0 },
		{ 1988, 3342, 0 },
		{ 2912, 3437, 0 },
		{ 2042, 3956, 0 },
		{ 2010, 3038, 0 },
		{ 2941, 3321, 0 },
		{ 0, 0, 59 },
		{ 0, 0, 96 },
		{ 0, 0, 98 },
		{ 2057, 3590, 0 },
		{ 2912, 3445, 0 },
		{ 2042, 3963, 0 },
		{ 2865, 3808, 0 },
		{ 2934, 3535, 0 },
		{ 2042, 3969, 0 },
		{ 0, 0, 130 },
		{ 2912, 3446, 0 },
		{ 2042, 3972, 0 },
		{ 2865, 3811, 0 },
		{ 2941, 3322, 0 },
		{ 2912, 3448, 0 },
		{ 2824, 4333, 0 },
		{ 2010, 3039, 0 },
		{ 1972, 3750, 0 },
		{ 1972, 3751, 0 },
		{ 2042, 3979, 0 },
		{ 2141, 4103, 0 },
		{ 2941, 3324, 0 },
		{ 2941, 3325, 0 },
		{ 2912, 3454, 0 },
		{ 2078, 3861, 0 },
		{ 0, 3760, 0 },
		{ 2941, 3326, 0 },
		{ 2941, 3327, 0 },
		{ 2865, 3830, 0 },
		{ 2934, 3554, 0 },
		{ 2057, 3612, 0 },
		{ 2824, 4407, 0 },
		{ 2865, 3835, 0 },
		{ 2941, 3328, 0 },
		{ 2057, 3615, 0 },
		{ 2946, 4175, 0 },
		{ 0, 0, 19 },
		{ 1988, 3354, 0 },
		{ 1988, 3356, 0 },
		{ 0, 0, 123 },
		{ 1988, 3357, 0 },
		{ 0, 0, 125 },
		{ 2912, 3463, 0 },
		{ 0, 0, 94 },
		{ 2010, 3040, 0 },
		{ 1972, 3725, 0 },
		{ 1972, 3727, 0 },
		{ 2010, 3042, 0 },
		{ 2824, 4451, 0 },
		{ 1988, 3362, 0 },
		{ 2057, 3578, 0 },
		{ 2865, 3782, 0 },
		{ 0, 0, 75 },
		{ 1972, 3732, 0 },
		{ 1128, 3911, 0 },
		{ 2010, 3043, 0 },
		{ 1972, 3735, 0 },
		{ 2912, 3469, 0 },
		{ 2042, 3928, 0 },
		{ 2946, 4165, 0 },
		{ 2946, 4166, 0 },
		{ 2824, 4247, 0 },
		{ 2042, 3929, 0 },
		{ 2865, 3788, 0 },
		{ 2865, 3789, 0 },
		{ 2010, 3044, 0 },
		{ 1988, 3365, 0 },
		{ 2941, 3242, 0 },
		{ 2911, 3626, 0 },
		{ 2941, 3243, 0 },
		{ 1988, 3368, 0 },
		{ 2865, 3798, 0 },
		{ 2911, 3631, 0 },
		{ 2941, 3246, 0 },
		{ 2042, 3944, 0 },
		{ 0, 0, 83 },
		{ 2946, 4155, 0 },
		{ 0, 0, 100 },
		{ 2946, 3778, 0 },
		{ 2042, 3948, 0 },
		{ 0, 0, 128 },
		{ 2941, 3247, 0 },
		{ 2941, 3248, 0 },
		{ 2010, 3045, 56 },
		{ 2911, 3638, 0 },
		{ 2057, 3595, 0 },
		{ 1972, 3756, 0 },
		{ 2938, 3228, 0 },
		{ 2759, 3768, 0 },
		{ 0, 0, 84 },
		{ 1988, 3383, 0 },
		{ 2946, 4182, 0 },
		{ 963, 3857, 0 },
		{ 0, 3858, 0 },
		{ 2941, 3251, 0 },
		{ 2911, 3648, 0 },
		{ 2911, 3650, 0 },
		{ 2057, 3601, 0 },
		{ 2946, 4133, 0 },
		{ 2042, 3968, 0 },
		{ 2865, 3815, 0 },
		{ 2010, 3046, 0 },
		{ 2057, 3603, 0 },
		{ 2057, 3604, 0 },
		{ 2057, 3605, 0 },
		{ 0, 0, 88 },
		{ 2865, 3824, 0 },
		{ 1988, 3386, 0 },
		{ 2912, 3397, 0 },
		{ 2010, 3048, 0 },
		{ 2938, 3232, 0 },
		{ 2010, 3049, 0 },
		{ 2934, 3560, 0 },
		{ 2865, 3833, 0 },
		{ 2141, 4072, 0 },
		{ 1988, 3391, 0 },
		{ 2911, 3669, 0 },
		{ 0, 0, 90 },
		{ 2911, 3670, 0 },
		{ 2141, 4113, 0 },
		{ 2141, 4115, 0 },
		{ 2941, 3257, 0 },
		{ 0, 0, 15 },
		{ 1972, 3739, 0 },
		{ 0, 0, 17 },
		{ 0, 0, 18 },
		{ 2865, 3840, 0 },
		{ 2010, 3050, 0 },
		{ 2078, 3860, 0 },
		{ 2911, 3677, 0 },
		{ 2824, 4239, 0 },
		{ 2912, 3412, 0 },
		{ 2057, 3619, 0 },
		{ 1128, 3885, 0 },
		{ 2912, 3415, 0 },
		{ 2912, 3416, 0 },
		{ 2911, 3627, 0 },
		{ 2057, 3571, 0 },
		{ 0, 0, 55 },
		{ 2865, 3780, 0 },
		{ 2941, 3259, 0 },
		{ 2010, 3052, 0 },
		{ 2941, 3261, 0 },
		{ 1972, 3753, 0 },
		{ 2057, 3579, 0 },
		{ 2042, 3918, 0 },
		{ 0, 0, 73 },
		{ 2010, 3056, 0 },
		{ 2946, 4186, 0 },
		{ 2912, 3422, 0 },
		{ 0, 3143, 0 },
		{ 2912, 3424, 0 },
		{ 0, 0, 16 },
		{ 2057, 3585, 0 },
		{ 1128, 3908, 0 },
		{ 2010, 3057, 54 },
		{ 2010, 3058, 0 },
		{ 1972, 3712, 0 },
		{ 0, 0, 74 },
		{ 2911, 3646, 0 },
		{ 2946, 3188, 0 },
		{ 0, 0, 81 },
		{ 0, 0, 82 },
		{ 0, 0, 52 },
		{ 2911, 3649, 0 },
		{ 2934, 3536, 0 },
		{ 2934, 3538, 0 },
		{ 2941, 3267, 0 },
		{ 2934, 3541, 0 },
		{ 0, 0, 129 },
		{ 2911, 3655, 0 },
		{ 1128, 3894, 0 },
		{ 1128, 3895, 0 },
		{ 2941, 3268, 0 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 2010, 3062, 40 },
		{ 2911, 3659, 0 },
		{ 2946, 4173, 0 },
		{ 1128, 3900, 0 },
		{ 1128, 3901, 0 },
		{ 1972, 3730, 0 },
		{ 0, 0, 71 },
		{ 2911, 3660, 0 },
		{ 2941, 3270, 0 },
		{ 0, 0, 89 },
		{ 2941, 3272, 0 },
		{ 1972, 3734, 0 },
		{ 2934, 3547, 0 },
		{ 1972, 3737, 0 },
		{ 1988, 3348, 0 },
		{ 2865, 3818, 0 },
		{ 2938, 3213, 0 },
		{ 2865, 3820, 0 },
		{ 2078, 3880, 0 },
		{ 2010, 3063, 0 },
		{ 2941, 3275, 0 },
		{ 2946, 4156, 0 },
		{ 2938, 3216, 0 },
		{ 0, 0, 85 },
		{ 2946, 4159, 0 },
		{ 2010, 3064, 0 },
		{ 0, 0, 124 },
		{ 0, 0, 126 },
		{ 1972, 3744, 0 },
		{ 0, 0, 131 },
		{ 0, 0, 11 },
		{ 2911, 3671, 0 },
		{ 2911, 3672, 0 },
		{ 2057, 3608, 0 },
		{ 2934, 3559, 0 },
		{ 1128, 3898, 0 },
		{ 2010, 3065, 0 },
		{ 2941, 3279, 0 },
		{ 2911, 3679, 0 },
		{ 2941, 3281, 0 },
		{ 2946, 4181, 0 },
		{ 0, 0, 127 },
		{ 2865, 3836, 0 },
		{ 2946, 4183, 0 },
		{ 2911, 3624, 0 },
		{ 2938, 3221, 0 },
		{ 2938, 3226, 0 },
		{ 2946, 4122, 0 },
		{ 2010, 3067, 0 },
		{ 2946, 4128, 0 },
		{ 2865, 3841, 0 },
		{ 2824, 4303, 0 },
		{ 2941, 3287, 0 },
		{ 2941, 3289, 0 },
		{ 2010, 3068, 0 },
		{ 0, 0, 41 },
		{ 2911, 3632, 0 },
		{ 2824, 4313, 0 },
		{ 2946, 4137, 0 },
		{ 2941, 3292, 0 },
		{ 2057, 3567, 0 },
		{ 1972, 3714, 0 },
		{ 2865, 3852, 0 },
		{ 2865, 3853, 0 },
		{ 2759, 4191, 0 },
		{ 2946, 4157, 0 },
		{ 1972, 3715, 0 },
		{ 2824, 4339, 0 },
		{ 2865, 3779, 0 },
		{ 2911, 3635, 0 },
		{ 1972, 3717, 0 },
		{ 2057, 3569, 0 },
		{ 2057, 3570, 0 },
		{ 2042, 3917, 0 },
		{ 2941, 3293, 0 },
		{ 1972, 3721, 0 },
		{ 1972, 3722, 0 },
		{ 2057, 3572, 0 },
		{ 0, 0, 76 },
		{ 0, 0, 91 },
		{ 2911, 3640, 0 },
		{ 2911, 3641, 0 },
		{ 0, 3905, 0 },
		{ 2865, 3791, 0 },
		{ 0, 0, 86 },
		{ 1972, 3726, 0 },
		{ 2911, 3642, 0 },
		{ 1988, 3369, 0 },
		{ 0, 0, 12 },
		{ 2057, 3575, 0 },
		{ 2057, 3576, 0 },
		{ 0, 0, 87 },
		{ 0, 0, 53 },
		{ 0, 0, 92 },
		{ 2912, 3462, 0 },
		{ 2911, 3647, 0 },
		{ 2042, 3939, 0 },
		{ 0, 0, 14 },
		{ 2010, 3069, 0 },
		{ 2912, 3464, 0 },
		{ 2042, 3942, 0 },
		{ 2934, 3532, 0 },
		{ 1972, 3736, 0 },
		{ 2824, 4449, 0 },
		{ 2946, 4141, 0 },
		{ 2042, 3945, 0 },
		{ 1991, 3684, 0 },
		{ 2042, 3947, 0 },
		{ 2911, 3651, 0 },
		{ 2941, 3296, 0 },
		{ 0, 0, 13 },
		{ 2990, 1420, 224 },
		{ 0, 0, 225 },
		{ 2945, 4761, 226 },
		{ 2973, 1597, 230 },
		{ 1165, 2461, 231 },
		{ 0, 0, 231 },
		{ 2973, 1774, 227 },
		{ 1168, 1386, 0 },
		{ 2973, 1675, 228 },
		{ 1171, 1422, 0 },
		{ 1168, 0, 0 },
		{ 2899, 2576, 229 },
		{ 1173, 1435, 0 },
		{ 1171, 0, 0 },
		{ 2899, 2606, 227 },
		{ 1173, 0, 0 },
		{ 2899, 2616, 228 },
		{ 2938, 3222, 140 },
		{ 0, 0, 140 },
		{ 0, 0, 141 },
		{ 2951, 1980, 0 },
		{ 2973, 2729, 0 },
		{ 2990, 2056, 0 },
		{ 1181, 4608, 0 },
		{ 2970, 2513, 0 },
		{ 2973, 2784, 0 },
		{ 2985, 2872, 0 },
		{ 2981, 2411, 0 },
		{ 2984, 2922, 0 },
		{ 2990, 2029, 0 },
		{ 2984, 2953, 0 },
		{ 2986, 1883, 0 },
		{ 2891, 2599, 0 },
		{ 2988, 2162, 0 },
		{ 2942, 2238, 0 },
		{ 2951, 1972, 0 },
		{ 2991, 4489, 0 },
		{ 0, 0, 138 },
		{ 2759, 4190, 148 },
		{ 0, 0, 148 },
		{ 0, 0, 149 },
		{ 2973, 2818, 0 },
		{ 2837, 2705, 0 },
		{ 2988, 2166, 0 },
		{ 2990, 2043, 0 },
		{ 2973, 2812, 0 },
		{ 1204, 4559, 0 },
		{ 2973, 2447, 0 },
		{ 2955, 1471, 0 },
		{ 2973, 2762, 0 },
		{ 2990, 2060, 0 },
		{ 2604, 1422, 0 },
		{ 2986, 1770, 0 },
		{ 2980, 2650, 0 },
		{ 2891, 2621, 0 },
		{ 2942, 2197, 0 },
		{ 2793, 2672, 0 },
		{ 1215, 4656, 0 },
		{ 2973, 2451, 0 },
		{ 2981, 2390, 0 },
		{ 2951, 1967, 0 },
		{ 2973, 2836, 0 },
		{ 1220, 4645, 0 },
		{ 2983, 2395, 0 },
		{ 2978, 1560, 0 },
		{ 2942, 2288, 0 },
		{ 2985, 2865, 0 },
		{ 2986, 1902, 0 },
		{ 2891, 2504, 0 },
		{ 2988, 2139, 0 },
		{ 2942, 2246, 0 },
		{ 2991, 4331, 0 },
		{ 0, 0, 146 },
		{ 2938, 3224, 172 },
		{ 0, 0, 172 },
		{ 2951, 1992, 0 },
		{ 2973, 2761, 0 },
		{ 2990, 2050, 0 },
		{ 1236, 4557, 0 },
		{ 2985, 2665, 0 },
		{ 2981, 2368, 0 },
		{ 2984, 2935, 0 },
		{ 2951, 1930, 0 },
		{ 2951, 1946, 0 },
		{ 2973, 2832, 0 },
		{ 2951, 1950, 0 },
		{ 2991, 4405, 0 },
		{ 0, 0, 171 },
		{ 2078, 3879, 136 },
		{ 0, 0, 136 },
		{ 0, 0, 137 },
		{ 2973, 2724, 0 },
		{ 2942, 2249, 0 },
		{ 2988, 2179, 0 },
		{ 2975, 2308, 0 },
		{ 2973, 2776, 0 },
		{ 2946, 4167, 0 },
		{ 2981, 2379, 0 },
		{ 2984, 2918, 0 },
		{ 2951, 1958, 0 },
		{ 2951, 1966, 0 },
		{ 2953, 4521, 0 },
		{ 2953, 4515, 0 },
		{ 2891, 2509, 0 },
		{ 2942, 2215, 0 },
		{ 2891, 2608, 0 },
		{ 2986, 1905, 0 },
		{ 2891, 2631, 0 },
		{ 2984, 2916, 0 },
		{ 2981, 2386, 0 },
		{ 2891, 2505, 0 },
		{ 2951, 1571, 0 },
		{ 2973, 2777, 0 },
		{ 2990, 2073, 0 },
		{ 2991, 4335, 0 },
		{ 0, 0, 134 },
		{ 2517, 2892, 23 },
		{ 0, 0, 23 },
		{ 0, 0, 24 },
		{ 2973, 2796, 0 },
		{ 2793, 2667, 0 },
		{ 2891, 2601, 0 },
		{ 2942, 2274, 0 },
		{ 2945, 2, 0 },
		{ 2988, 2159, 0 },
		{ 2751, 2092, 0 },
		{ 2973, 2719, 0 },
		{ 2990, 2018, 0 },
		{ 2984, 2956, 0 },
		{ 2986, 1840, 0 },
		{ 2988, 2131, 0 },
		{ 2990, 2032, 0 },
		{ 2945, 4739, 0 },
		{ 2970, 2856, 0 },
		{ 2973, 2772, 0 },
		{ 2951, 1991, 0 },
		{ 2985, 2870, 0 },
		{ 2990, 2044, 0 },
		{ 2891, 2614, 0 },
		{ 2751, 2083, 0 },
		{ 2986, 1865, 0 },
		{ 2891, 2642, 0 },
		{ 2988, 2122, 0 },
		{ 2942, 2214, 0 },
		{ 2945, 7, 0 },
		{ 2953, 4524, 0 },
		{ 0, 0, 20 },
		{ 1319, 0, 1 },
		{ 1319, 0, 173 },
		{ 1319, 2662, 223 },
		{ 1534, 204, 223 },
		{ 1534, 417, 223 },
		{ 1534, 405, 223 },
		{ 1534, 522, 223 },
		{ 1534, 406, 223 },
		{ 1534, 417, 223 },
		{ 1534, 392, 223 },
		{ 1534, 419, 223 },
		{ 1534, 477, 223 },
		{ 1319, 0, 223 },
		{ 1331, 2439, 223 },
		{ 1319, 2714, 223 },
		{ 2517, 2890, 219 },
		{ 1534, 497, 223 },
		{ 1534, 496, 223 },
		{ 1534, 510, 223 },
		{ 1534, 0, 223 },
		{ 1534, 545, 223 },
		{ 1534, 532, 223 },
		{ 2988, 2140, 0 },
		{ 0, 0, 174 },
		{ 2942, 2200, 0 },
		{ 1534, 499, 0 },
		{ 1534, 0, 0 },
		{ 2945, 3827, 0 },
		{ 1534, 519, 0 },
		{ 1534, 535, 0 },
		{ 1534, 532, 0 },
		{ 1534, 539, 0 },
		{ 1534, 542, 0 },
		{ 1534, 545, 0 },
		{ 1534, 552, 0 },
		{ 1534, 596, 0 },
		{ 1534, 525, 0 },
		{ 1534, 519, 0 },
		{ 1534, 548, 0 },
		{ 2973, 2739, 0 },
		{ 2973, 2750, 0 },
		{ 1535, 555, 0 },
		{ 1535, 556, 0 },
		{ 1534, 567, 0 },
		{ 1534, 596, 0 },
		{ 1534, 588, 0 },
		{ 2988, 2163, 0 },
		{ 2970, 2850, 0 },
		{ 1534, 587, 0 },
		{ 1534, 631, 0 },
		{ 1534, 610, 0 },
		{ 1534, 612, 0 },
		{ 1534, 646, 0 },
		{ 1534, 647, 0 },
		{ 1534, 652, 0 },
		{ 1534, 646, 0 },
		{ 1534, 622, 0 },
		{ 1534, 613, 0 },
		{ 1534, 13, 0 },
		{ 1534, 40, 0 },
		{ 1534, 27, 0 },
		{ 2942, 2227, 0 },
		{ 2837, 2698, 0 },
		{ 1534, 43, 0 },
		{ 1534, 33, 0 },
		{ 1535, 35, 0 },
		{ 1534, 31, 0 },
		{ 1534, 35, 0 },
		{ 2981, 2401, 0 },
		{ 0, 0, 222 },
		{ 1534, 50, 0 },
		{ 1534, 26, 0 },
		{ 1534, 14, 0 },
		{ 1534, 72, 0 },
		{ 1534, 77, 0 },
		{ 1534, 79, 0 },
		{ 1534, 76, 0 },
		{ 1534, 66, 0 },
		{ 1534, 58, 0 },
		{ 1534, 62, 0 },
		{ 1534, 79, 0 },
		{ 1534, 0, 208 },
		{ 1534, 140, 0 },
		{ 2988, 2136, 0 },
		{ 2891, 2624, 0 },
		{ 1534, 99, 0 },
		{ 1534, 103, 0 },
		{ 1534, 126, 0 },
		{ 1534, 140, 0 },
		{ 1534, 140, 0 },
		{ -1414, 1092, 0 },
		{ 1535, 148, 0 },
		{ 1534, 181, 0 },
		{ 1534, 187, 0 },
		{ 1534, 179, 0 },
		{ 1534, 193, 0 },
		{ 1534, 195, 0 },
		{ 1534, 175, 0 },
		{ 1534, 192, 0 },
		{ 1534, 169, 0 },
		{ 1534, 160, 0 },
		{ 1534, 0, 207 },
		{ 1534, 177, 0 },
		{ 2975, 2323, 0 },
		{ 2942, 2276, 0 },
		{ 1534, 180, 0 },
		{ 1534, 189, 0 },
		{ 1534, 186, 0 },
		{ 1534, 0, 221 },
		{ 1534, 185, 0 },
		{ 0, 0, 209 },
		{ 1534, 178, 0 },
		{ 1536, 33, -4 },
		{ 1534, 234, 0 },
		{ 1534, 247, 0 },
		{ 1534, 273, 0 },
		{ 1534, 279, 0 },
		{ 1534, 288, 0 },
		{ 1534, 301, 0 },
		{ 1534, 272, 0 },
		{ 1534, 277, 0 },
		{ 1534, 268, 0 },
		{ 2973, 2822, 0 },
		{ 2973, 2829, 0 },
		{ 1534, 0, 211 },
		{ 1534, 307, 212 },
		{ 1534, 276, 0 },
		{ 1534, 285, 0 },
		{ 1534, 313, 0 },
		{ 1434, 3410, 0 },
		{ 2945, 4152, 0 },
		{ 2119, 4459, 198 },
		{ 1534, 319, 0 },
		{ 1534, 324, 0 },
		{ 1534, 335, 0 },
		{ 1534, 336, 0 },
		{ 1534, 338, 0 },
		{ 1534, 366, 0 },
		{ 1534, 356, 0 },
		{ 1534, 358, 0 },
		{ 1534, 373, 0 },
		{ 1534, 380, 0 },
		{ 1535, 366, 0 },
		{ 2946, 4168, 0 },
		{ 2945, 4, 214 },
		{ 1534, 369, 0 },
		{ 1534, 383, 0 },
		{ 1534, 365, 0 },
		{ 1534, 381, 0 },
		{ 0, 0, 178 },
		{ 1536, 117, -7 },
		{ 1536, 231, -10 },
		{ 1536, 345, -13 },
		{ 1536, 376, -16 },
		{ 1536, 460, -19 },
		{ 1536, 488, -22 },
		{ 1534, 409, 0 },
		{ 1534, 422, 0 },
		{ 1534, 396, 0 },
		{ 1534, 0, 196 },
		{ 1534, 0, 210 },
		{ 2981, 2381, 0 },
		{ 1534, 396, 0 },
		{ 1534, 386, 0 },
		{ 1534, 391, 0 },
		{ 1535, 392, 0 },
		{ 1471, 3446, 0 },
		{ 2945, 4184, 0 },
		{ 2119, 4475, 199 },
		{ 1474, 3450, 0 },
		{ 2945, 4148, 0 },
		{ 2119, 4497, 200 },
		{ 1477, 3451, 0 },
		{ 2945, 4156, 0 },
		{ 2119, 4479, 203 },
		{ 1480, 3452, 0 },
		{ 2945, 4072, 0 },
		{ 2119, 4471, 204 },
		{ 1483, 3453, 0 },
		{ 2945, 4146, 0 },
		{ 2119, 4483, 205 },
		{ 1486, 3454, 0 },
		{ 2945, 4150, 0 },
		{ 2119, 4466, 206 },
		{ 1534, 437, 0 },
		{ 1536, 490, -25 },
		{ 1534, 422, 0 },
		{ 2984, 2903, 0 },
		{ 1534, 403, 0 },
		{ 1534, 448, 0 },
		{ 1534, 441, 0 },
		{ 1534, 453, 0 },
		{ 0, 0, 180 },
		{ 0, 0, 182 },
		{ 0, 0, 188 },
		{ 0, 0, 190 },
		{ 0, 0, 192 },
		{ 0, 0, 194 },
		{ 1536, 574, -28 },
		{ 1504, 3464, 0 },
		{ 2945, 4160, 0 },
		{ 2119, 4493, 202 },
		{ 1534, 0, 195 },
		{ 2951, 1952, 0 },
		{ 1534, 474, 0 },
		{ 1534, 489, 0 },
		{ 1535, 482, 0 },
		{ 1534, 479, 0 },
		{ 1513, 3409, 0 },
		{ 2945, 4154, 0 },
		{ 2119, 4496, 201 },
		{ 0, 0, 186 },
		{ 2951, 1974, 0 },
		{ 1534, 4, 217 },
		{ 1535, 486, 0 },
		{ 1534, 1, 220 },
		{ 1534, 501, 0 },
		{ 0, 0, 184 },
		{ 2953, 4522, 0 },
		{ 2953, 4523, 0 },
		{ 1534, 488, 0 },
		{ 0, 0, 218 },
		{ 1534, 484, 0 },
		{ 2953, 4518, 0 },
		{ 0, 0, 216 },
		{ 1534, 492, 0 },
		{ 1534, 497, 0 },
		{ 0, 0, 215 },
		{ 1534, 502, 0 },
		{ 1534, 494, 0 },
		{ 1535, 499, 213 },
		{ 1536, 925, 0 },
		{ 1537, 736, -1 },
		{ 1538, 3424, 0 },
		{ 2945, 4116, 0 },
		{ 2119, 4491, 197 },
		{ 0, 0, 176 },
		{ 2078, 3881, 266 },
		{ 0, 0, 266 },
		{ 2973, 2754, 0 },
		{ 2942, 2230, 0 },
		{ 2988, 2117, 0 },
		{ 2975, 2335, 0 },
		{ 2973, 2775, 0 },
		{ 2946, 4170, 0 },
		{ 2981, 2382, 0 },
		{ 2984, 2930, 0 },
		{ 2951, 1986, 0 },
		{ 2951, 1987, 0 },
		{ 2953, 4504, 0 },
		{ 2953, 4505, 0 },
		{ 2891, 2617, 0 },
		{ 2942, 2270, 0 },
		{ 2891, 2622, 0 },
		{ 2986, 1776, 0 },
		{ 2891, 2628, 0 },
		{ 2984, 2923, 0 },
		{ 2981, 2364, 0 },
		{ 2891, 2636, 0 },
		{ 2951, 1507, 0 },
		{ 2973, 2718, 0 },
		{ 2990, 2027, 0 },
		{ 2991, 4349, 0 },
		{ 0, 0, 265 },
		{ 2078, 3875, 268 },
		{ 0, 0, 268 },
		{ 0, 0, 269 },
		{ 2973, 2721, 0 },
		{ 2942, 2287, 0 },
		{ 2988, 2142, 0 },
		{ 2975, 2329, 0 },
		{ 2973, 2740, 0 },
		{ 2946, 4151, 0 },
		{ 2981, 2388, 0 },
		{ 2984, 2949, 0 },
		{ 2951, 1999, 0 },
		{ 2951, 2000, 0 },
		{ 2953, 4506, 0 },
		{ 2953, 4511, 0 },
		{ 2985, 2882, 0 },
		{ 2990, 2042, 0 },
		{ 2988, 2164, 0 },
		{ 2951, 1925, 0 },
		{ 2951, 1928, 0 },
		{ 2988, 2112, 0 },
		{ 2955, 1474, 0 },
		{ 2973, 2794, 0 },
		{ 2990, 2052, 0 },
		{ 2991, 4407, 0 },
		{ 0, 0, 267 },
		{ 2078, 3865, 271 },
		{ 0, 0, 271 },
		{ 0, 0, 272 },
		{ 2973, 2797, 0 },
		{ 2942, 2254, 0 },
		{ 2988, 2127, 0 },
		{ 2975, 2325, 0 },
		{ 2973, 2819, 0 },
		{ 2946, 4180, 0 },
		{ 2981, 2409, 0 },
		{ 2984, 2928, 0 },
		{ 2951, 1935, 0 },
		{ 2951, 1943, 0 },
		{ 2953, 4519, 0 },
		{ 2953, 4520, 0 },
		{ 2975, 2336, 0 },
		{ 2978, 1518, 0 },
		{ 2986, 1884, 0 },
		{ 2984, 2900, 0 },
		{ 2986, 1891, 0 },
		{ 2988, 2148, 0 },
		{ 2990, 2028, 0 },
		{ 2991, 4263, 0 },
		{ 0, 0, 270 },
		{ 2078, 3870, 274 },
		{ 0, 0, 274 },
		{ 0, 0, 275 },
		{ 2973, 2733, 0 },
		{ 2942, 2205, 0 },
		{ 2988, 2161, 0 },
		{ 2975, 2309, 0 },
		{ 2973, 2752, 0 },
		{ 2946, 4147, 0 },
		{ 2981, 2410, 0 },
		{ 2984, 2951, 0 },
		{ 2951, 1954, 0 },
		{ 2951, 1955, 0 },
		{ 2953, 4512, 0 },
		{ 2953, 4513, 0 },
		{ 2973, 2771, 0 },
		{ 2955, 1441, 0 },
		{ 2984, 2912, 0 },
		{ 2981, 2378, 0 },
		{ 2978, 1633, 0 },
		{ 2984, 2920, 0 },
		{ 2986, 1669, 0 },
		{ 2988, 2111, 0 },
		{ 2990, 2045, 0 },
		{ 2991, 4267, 0 },
		{ 0, 0, 273 },
		{ 2078, 3878, 277 },
		{ 0, 0, 277 },
		{ 0, 0, 278 },
		{ 2973, 2795, 0 },
		{ 2942, 2255, 0 },
		{ 2988, 2116, 0 },
		{ 2975, 2327, 0 },
		{ 2973, 2807, 0 },
		{ 2946, 4177, 0 },
		{ 2981, 2406, 0 },
		{ 2984, 2898, 0 },
		{ 2951, 1968, 0 },
		{ 2951, 1970, 0 },
		{ 2953, 4525, 0 },
		{ 2953, 4526, 0 },
		{ 2988, 2123, 0 },
		{ 2751, 2097, 0 },
		{ 2986, 1716, 0 },
		{ 2891, 2639, 0 },
		{ 2975, 2312, 0 },
		{ 2891, 2456, 0 },
		{ 2951, 1973, 0 },
		{ 2973, 2723, 0 },
		{ 2990, 2070, 0 },
		{ 2991, 4413, 0 },
		{ 0, 0, 276 },
		{ 2824, 4375, 152 },
		{ 0, 0, 152 },
		{ 0, 0, 153 },
		{ 2837, 2708, 0 },
		{ 2986, 1717, 0 },
		{ 2973, 2737, 0 },
		{ 2990, 2007, 0 },
		{ 1678, 4579, 0 },
		{ 2973, 2441, 0 },
		{ 2955, 1470, 0 },
		{ 2973, 2751, 0 },
		{ 2990, 2024, 0 },
		{ 2604, 1437, 0 },
		{ 2986, 1790, 0 },
		{ 2980, 2648, 0 },
		{ 2891, 2618, 0 },
		{ 2942, 2245, 0 },
		{ 2793, 2669, 0 },
		{ 1689, 4637, 0 },
		{ 2973, 2453, 0 },
		{ 2981, 2359, 0 },
		{ 2951, 1988, 0 },
		{ 2973, 2788, 0 },
		{ 1694, 4556, 0 },
		{ 2983, 2393, 0 },
		{ 2978, 1602, 0 },
		{ 2942, 2252, 0 },
		{ 2985, 2863, 0 },
		{ 2986, 1841, 0 },
		{ 2891, 2434, 0 },
		{ 2988, 2175, 0 },
		{ 2942, 2263, 0 },
		{ 2991, 4485, 0 },
		{ 0, 0, 150 },
		{ 2078, 3874, 259 },
		{ 0, 0, 259 },
		{ 2973, 2813, 0 },
		{ 2942, 2265, 0 },
		{ 2988, 2177, 0 },
		{ 2975, 2317, 0 },
		{ 2973, 2826, 0 },
		{ 2946, 4185, 0 },
		{ 2981, 2404, 0 },
		{ 2984, 2941, 0 },
		{ 2951, 1996, 0 },
		{ 2951, 1997, 0 },
		{ 2953, 4509, 0 },
		{ 2953, 4510, 0 },
		{ 2970, 2839, 0 },
		{ 2891, 2612, 0 },
		{ 2951, 1998, 0 },
		{ 2751, 2103, 0 },
		{ 2981, 2356, 0 },
		{ 2984, 2908, 0 },
		{ 2604, 1361, 0 },
		{ 2991, 4415, 0 },
		{ 0, 0, 257 },
		{ 1742, 0, 1 },
		{ 1901, 2759, 374 },
		{ 2973, 2734, 374 },
		{ 2984, 2787, 374 },
		{ 2970, 2116, 374 },
		{ 1742, 0, 341 },
		{ 1742, 2697, 374 },
		{ 2980, 1522, 374 },
		{ 2759, 4199, 374 },
		{ 2057, 3586, 374 },
		{ 2938, 3223, 374 },
		{ 2057, 3588, 374 },
		{ 2042, 3964, 374 },
		{ 2990, 1927, 374 },
		{ 1742, 0, 374 },
		{ 2517, 2893, 372 },
		{ 2984, 2687, 374 },
		{ 2984, 2964, 374 },
		{ 0, 0, 374 },
		{ 2988, 2134, 0 },
		{ -1747, 21, 331 },
		{ -1748, 4601, 0 },
		{ 2942, 2219, 0 },
		{ 0, 0, 337 },
		{ 0, 0, 338 },
		{ 2981, 2383, 0 },
		{ 2891, 2643, 0 },
		{ 2973, 2767, 0 },
		{ 0, 0, 342 },
		{ 2942, 2223, 0 },
		{ 2990, 2064, 0 },
		{ 2891, 2502, 0 },
		{ 2010, 3010, 0 },
		{ 2934, 3537, 0 },
		{ 2941, 3288, 0 },
		{ 1950, 3185, 0 },
		{ 2934, 3540, 0 },
		{ 2978, 1598, 0 },
		{ 2951, 1931, 0 },
		{ 2942, 2242, 0 },
		{ 2986, 1893, 0 },
		{ 2990, 2017, 0 },
		{ 2988, 2149, 0 },
		{ 2665, 4571, 0 },
		{ 2988, 2151, 0 },
		{ 2951, 1936, 0 },
		{ 2986, 1901, 0 },
		{ 2942, 2261, 0 },
		{ 2970, 2843, 0 },
		{ 2990, 2026, 0 },
		{ 2981, 2374, 0 },
		{ 2078, 3862, 0 },
		{ 2010, 3027, 0 },
		{ 2010, 3028, 0 },
		{ 2042, 4001, 0 },
		{ 1972, 3749, 0 },
		{ 2973, 2816, 0 },
		{ 2951, 1945, 0 },
		{ 2970, 2840, 0 },
		{ 2978, 1601, 0 },
		{ 2973, 2820, 0 },
		{ 2981, 2380, 0 },
		{ 0, 4772, 334 },
		{ 2975, 2307, 0 },
		{ 2973, 2827, 0 },
		{ 2057, 3581, 0 },
		{ 2986, 1904, 0 },
		{ 0, 0, 373 },
		{ 2973, 2830, 0 },
		{ 2970, 2847, 0 },
		{ 2042, 3931, 0 },
		{ 1988, 3374, 0 },
		{ 2934, 3530, 0 },
		{ 2912, 3450, 0 },
		{ 2010, 3041, 0 },
		{ 0, 0, 362 },
		{ 2946, 4172, 0 },
		{ 2988, 2171, 0 },
		{ 2990, 2030, 0 },
		{ 2942, 2286, 0 },
		{ -1824, 1167, 0 },
		{ 0, 0, 333 },
		{ 2973, 2720, 0 },
		{ 0, 0, 361 },
		{ 2751, 2088, 0 },
		{ 2891, 2452, 0 },
		{ 2942, 2192, 0 },
		{ 1839, 4540, 0 },
		{ 2911, 3656, 0 },
		{ 2865, 3796, 0 },
		{ 2912, 3461, 0 },
		{ 2010, 3051, 0 },
		{ 2934, 3544, 0 },
		{ 2988, 2110, 0 },
		{ 2975, 2330, 0 },
		{ 2942, 2199, 0 },
		{ 2986, 1906, 0 },
		{ 0, 0, 363 },
		{ 2946, 4131, 340 },
		{ 2986, 1918, 0 },
		{ 2985, 2864, 0 },
		{ 2986, 1663, 0 },
		{ 0, 0, 366 },
		{ 0, 0, 367 },
		{ 1844, 0, -71 },
		{ 2023, 3137, 0 },
		{ 2057, 3610, 0 },
		{ 2934, 3555, 0 },
		{ 2042, 3960, 0 },
		{ 2891, 2600, 0 },
		{ 0, 0, 365 },
		{ 0, 0, 371 },
		{ 0, 4542, 0 },
		{ 2981, 2362, 0 },
		{ 2951, 1960, 0 },
		{ 2984, 2931, 0 },
		{ 2078, 3876, 0 },
		{ 2945, 4086, 0 },
		{ 2119, 4490, 356 },
		{ 2042, 3966, 0 },
		{ 2759, 4189, 0 },
		{ 2912, 3475, 0 },
		{ 2912, 3476, 0 },
		{ 2942, 2217, 0 },
		{ 0, 0, 368 },
		{ 0, 0, 369 },
		{ 2984, 2938, 0 },
		{ 2125, 4584, 0 },
		{ 2981, 2370, 0 },
		{ 2973, 2759, 0 },
		{ 0, 0, 346 },
		{ 1867, 0, -74 },
		{ 1869, 0, -77 },
		{ 2057, 3568, 0 },
		{ 2946, 4169, 0 },
		{ 0, 0, 364 },
		{ 2951, 1964, 0 },
		{ 0, 0, 339 },
		{ 2078, 3863, 0 },
		{ 2942, 2221, 0 },
		{ 2945, 4068, 0 },
		{ 2119, 4463, 357 },
		{ 2945, 4070, 0 },
		{ 2119, 4469, 358 },
		{ 2759, 4188, 0 },
		{ 1879, 0, -59 },
		{ 2951, 1965, 0 },
		{ 2973, 2768, 0 },
		{ 2973, 2770, 0 },
		{ 0, 0, 348 },
		{ 0, 0, 350 },
		{ 1884, 0, -65 },
		{ 2945, 4112, 0 },
		{ 2119, 4495, 360 },
		{ 0, 0, 336 },
		{ 2942, 2224, 0 },
		{ 2990, 2055, 0 },
		{ 2945, 4140, 0 },
		{ 2119, 4461, 359 },
		{ 0, 0, 354 },
		{ 2988, 2129, 0 },
		{ 2984, 2909, 0 },
		{ 0, 0, 352 },
		{ 2975, 2334, 0 },
		{ 2986, 1664, 0 },
		{ 2973, 2781, 0 },
		{ 2891, 2633, 0 },
		{ 0, 0, 370 },
		{ 2988, 2133, 0 },
		{ 2942, 2243, 0 },
		{ 1898, 0, -80 },
		{ 2945, 4158, 0 },
		{ 2119, 4494, 355 },
		{ 0, 0, 344 },
		{ 1742, 2745, 374 },
		{ 1905, 2437, 374 },
		{ -1903, 4767, 331 },
		{ -1904, 4598, 0 },
		{ 2945, 4585, 0 },
		{ 2665, 4556, 0 },
		{ 0, 0, 332 },
		{ 2665, 4572, 0 },
		{ -1909, 4765, 0 },
		{ -1910, 4593, 0 },
		{ 1913, 2, 334 },
		{ 2665, 4573, 0 },
		{ 2945, 4640, 0 },
		{ 0, 0, 335 },
		{ 1931, 0, 1 },
		{ 2127, 2761, 330 },
		{ 2973, 2810, 330 },
		{ 1931, 0, 284 },
		{ 1931, 2507, 330 },
		{ 2934, 3549, 330 },
		{ 1931, 0, 287 },
		{ 2978, 1523, 330 },
		{ 2759, 4193, 330 },
		{ 2057, 3597, 330 },
		{ 2938, 3168, 330 },
		{ 2057, 3599, 330 },
		{ 2042, 3915, 330 },
		{ 2984, 2897, 330 },
		{ 2990, 1925, 330 },
		{ 1931, 0, 330 },
		{ 2517, 2891, 327 },
		{ 2984, 2905, 330 },
		{ 2970, 2853, 330 },
		{ 2759, 4195, 330 },
		{ 2984, 1593, 330 },
		{ 0, 0, 330 },
		{ 2988, 2143, 0 },
		{ -1938, 4763, 279 },
		{ -1939, 4594, 0 },
		{ 2942, 2258, 0 },
		{ 0, 0, 285 },
		{ 2942, 2259, 0 },
		{ 2988, 2144, 0 },
		{ 2990, 2015, 0 },
		{ 2010, 3011, 0 },
		{ 2934, 3520, 0 },
		{ 2941, 3305, 0 },
		{ 2911, 3675, 0 },
		{ 0, 3156, 0 },
		{ 0, 3189, 0 },
		{ 2934, 3527, 0 },
		{ 2981, 2366, 0 },
		{ 2978, 1577, 0 },
		{ 2951, 1975, 0 },
		{ 2942, 2273, 0 },
		{ 2973, 2837, 0 },
		{ 2973, 2714, 0 },
		{ 2988, 2154, 0 },
		{ 2125, 4583, 0 },
		{ 2988, 2155, 0 },
		{ 2665, 4577, 0 },
		{ 2988, 2156, 0 },
		{ 2970, 2852, 0 },
		{ 2751, 2096, 0 },
		{ 2990, 2019, 0 },
		{ 2078, 3864, 0 },
		{ 2010, 3024, 0 },
		{ 2010, 3025, 0 },
		{ 2865, 3816, 0 },
		{ 2865, 3817, 0 },
		{ 2042, 3958, 0 },
		{ 0, 3711, 0 },
		{ 2951, 1976, 0 },
		{ 2973, 2727, 0 },
		{ 2951, 1978, 0 },
		{ 2970, 2841, 0 },
		{ 2942, 2195, 0 },
		{ 2951, 1979, 0 },
		{ 2981, 2394, 0 },
		{ 0, 0, 329 },
		{ 2981, 2396, 0 },
		{ 0, 0, 281 },
		{ 2975, 2333, 0 },
		{ 0, 0, 326 },
		{ 2978, 1595, 0 },
		{ 2973, 2744, 0 },
		{ 2042, 3971, 0 },
		{ 0, 3355, 0 },
		{ 2934, 3557, 0 },
		{ 1991, 3687, 0 },
		{ 0, 3688, 0 },
		{ 2912, 3481, 0 },
		{ 2010, 3037, 0 },
		{ 2973, 2749, 0 },
		{ 0, 0, 319 },
		{ 2946, 4176, 0 },
		{ 2988, 2168, 0 },
		{ 2986, 1797, 0 },
		{ 2986, 1822, 0 },
		{ 2978, 1596, 0 },
		{ -2018, 1242, 0 },
		{ 2973, 2760, 0 },
		{ 2981, 2357, 0 },
		{ 2942, 2218, 0 },
		{ 2911, 3661, 0 },
		{ 2865, 3848, 0 },
		{ 2912, 3409, 0 },
		{ 2865, 3850, 0 },
		{ 2865, 3851, 0 },
		{ 0, 3047, 0 },
		{ 2934, 3524, 0 },
		{ 0, 0, 318 },
		{ 2988, 2107, 0 },
		{ 2975, 2319, 0 },
		{ 2891, 2511, 0 },
		{ 0, 0, 325 },
		{ 2984, 2958, 0 },
		{ 0, 0, 320 },
		{ 0, 0, 283 },
		{ 2984, 2963, 0 },
		{ 2986, 1843, 0 },
		{ 2035, 0, -56 },
		{ 0, 3136, 0 },
		{ 2057, 3613, 0 },
		{ 2026, 3125, 0 },
		{ 2023, 3126, 0 },
		{ 2934, 3534, 0 },
		{ 2042, 4005, 0 },
		{ 2891, 2595, 0 },
		{ 0, 0, 322 },
		{ 2985, 2877, 0 },
		{ 2986, 1848, 0 },
		{ 2986, 1853, 0 },
		{ 2078, 3877, 0 },
		{ 2945, 4142, 0 },
		{ 2119, 4492, 309 },
		{ 2042, 3920, 0 },
		{ 2759, 4194, 0 },
		{ 2042, 3921, 0 },
		{ 2042, 3922, 0 },
		{ 2042, 3923, 0 },
		{ 0, 3924, 0 },
		{ 2912, 3427, 0 },
		{ 2912, 3428, 0 },
		{ 2942, 2226, 0 },
		{ 2984, 2907, 0 },
		{ 2891, 2602, 0 },
		{ 2891, 2603, 0 },
		{ 2973, 2786, 0 },
		{ 0, 0, 291 },
		{ 2064, 0, -35 },
		{ 2066, 0, -38 },
		{ 2068, 0, -44 },
		{ 2070, 0, -47 },
		{ 2072, 0, -50 },
		{ 2074, 0, -53 },
		{ 0, 3573, 0 },
		{ 2946, 4184, 0 },
		{ 0, 0, 321 },
		{ 2981, 2376, 0 },
		{ 2988, 2113, 0 },
		{ 2988, 2114, 0 },
		{ 2942, 2232, 0 },
		{ 2945, 4074, 0 },
		{ 2119, 4467, 310 },
		{ 2945, 4076, 0 },
		{ 2119, 4470, 311 },
		{ 2945, 4078, 0 },
		{ 2119, 4473, 314 },
		{ 2945, 4080, 0 },
		{ 2119, 4477, 315 },
		{ 2945, 4082, 0 },
		{ 2119, 4481, 316 },
		{ 2945, 4084, 0 },
		{ 2119, 4489, 317 },
		{ 2759, 4200, 0 },
		{ 2089, 0, -62 },
		{ 0, 3867, 0 },
		{ 2942, 2233, 0 },
		{ 2942, 2234, 0 },
		{ 2973, 2805, 0 },
		{ 0, 0, 293 },
		{ 0, 0, 295 },
		{ 0, 0, 301 },
		{ 0, 0, 303 },
		{ 0, 0, 305 },
		{ 0, 0, 307 },
		{ 2095, 0, -68 },
		{ 2945, 4114, 0 },
		{ 2119, 4499, 313 },
		{ 2973, 2806, 0 },
		{ 2984, 2932, 0 },
		{ 2920, 3119, 324 },
		{ 2990, 2046, 0 },
		{ 2945, 4144, 0 },
		{ 2119, 4468, 312 },
		{ 0, 0, 299 },
		{ 2942, 2239, 0 },
		{ 2990, 2048, 0 },
		{ 0, 0, 286 },
		{ 2984, 2943, 0 },
		{ 0, 0, 297 },
		{ 2988, 2120, 0 },
		{ 2604, 1345, 0 },
		{ 2986, 1875, 0 },
		{ 2975, 2321, 0 },
		{ 2824, 4377, 0 },
		{ 2891, 2638, 0 },
		{ 2973, 2821, 0 },
		{ 2981, 2402, 0 },
		{ 2988, 2125, 0 },
		{ 0, 0, 323 },
		{ 2793, 2671, 0 },
		{ 2942, 2251, 0 },
		{ 2988, 2126, 0 },
		{ 2118, 0, -41 },
		{ 2990, 2054, 0 },
		{ 2945, 4066, 0 },
		{ 0, 4498, 308 },
		{ 2891, 2448, 0 },
		{ 0, 0, 289 },
		{ 2986, 1876, 0 },
		{ 2980, 2652, 0 },
		{ 2975, 2332, 0 },
		{ 0, 1, 0 },
		{ 0, 0, 328 },
		{ 1931, 2770, 330 },
		{ 2131, 2438, 330 },
		{ -2129, 22, 279 },
		{ -2130, 4592, 0 },
		{ 2945, 4584, 0 },
		{ 2665, 4554, 0 },
		{ 0, 0, 280 },
		{ 2665, 4574, 0 },
		{ -2135, 4770, 0 },
		{ -2136, 4597, 0 },
		{ 2139, 0, 281 },
		{ 2665, 4575, 0 },
		{ 2945, 4673, 0 },
		{ 0, 0, 282 },
		{ 0, 4066, 376 },
		{ 0, 0, 376 },
		{ 2973, 2722, 0 },
		{ 2837, 2687, 0 },
		{ 2984, 2927, 0 },
		{ 2978, 1604, 0 },
		{ 2981, 2360, 0 },
		{ 2986, 1887, 0 },
		{ 2945, 9, 0 },
		{ 2990, 2068, 0 },
		{ 2978, 1616, 0 },
		{ 2942, 2269, 0 },
		{ 2154, 4563, 0 },
		{ 2945, 1916, 0 },
		{ 2984, 2942, 0 },
		{ 2990, 2005, 0 },
		{ 2984, 2947, 0 },
		{ 2975, 2313, 0 },
		{ 2973, 2742, 0 },
		{ 2986, 1896, 0 },
		{ 2973, 2745, 0 },
		{ 2990, 2008, 0 },
		{ 2951, 1942, 0 },
		{ 2991, 4339, 0 },
		{ 0, 0, 375 },
		{ 2665, 4567, 420 },
		{ 0, 0, 381 },
		{ 0, 0, 383 },
		{ 2185, 821, 411 },
		{ 2333, 834, 411 },
		{ 2352, 832, 411 },
		{ 2305, 833, 411 },
		{ 2186, 841, 411 },
		{ 2184, 827, 411 },
		{ 2352, 831, 411 },
		{ 2205, 845, 411 },
		{ 2330, 848, 411 },
		{ 2330, 849, 411 },
		{ 2333, 847, 411 },
		{ 2284, 856, 411 },
		{ 2973, 1653, 410 },
		{ 2212, 2463, 420 },
		{ 2381, 845, 411 },
		{ 2333, 858, 411 },
		{ 2215, 858, 411 },
		{ 2333, 854, 411 },
		{ 2973, 2790, 420 },
		{ -2188, 4768, 377 },
		{ -2189, 4596, 0 },
		{ 2381, 851, 411 },
		{ 2386, 355, 411 },
		{ 2381, 853, 411 },
		{ 2256, 852, 411 },
		{ 2333, 860, 411 },
		{ 2338, 855, 411 },
		{ 2333, 862, 411 },
		{ 2284, 871, 411 },
		{ 2259, 861, 411 },
		{ 2352, 856, 411 },
		{ 2183, 850, 411 },
		{ 2307, 857, 411 },
		{ 2183, 854, 411 },
		{ 2363, 865, 411 },
		{ 2338, 865, 411 },
		{ 2183, 875, 411 },
		{ 2363, 868, 411 },
		{ 2321, 879, 411 },
		{ 2363, 870, 411 },
		{ 2259, 873, 411 },
		{ 2973, 1686, 407 },
		{ 2237, 1511, 0 },
		{ 2973, 1708, 408 },
		{ 2942, 2272, 0 },
		{ 2665, 4568, 0 },
		{ 2183, 884, 411 },
		{ 2988, 1963, 0 },
		{ 2330, 882, 411 },
		{ 2248, 867, 411 },
		{ 2363, 875, 411 },
		{ 2307, 881, 411 },
		{ 2307, 882, 411 },
		{ 2259, 891, 411 },
		{ 2330, 899, 411 },
		{ 2305, 883, 411 },
		{ 2330, 901, 411 },
		{ 2284, 906, 411 },
		{ 2386, 357, 411 },
		{ 2386, 370, 411 },
		{ 2358, 917, 411 },
		{ 2358, 918, 411 },
		{ 2330, 961, 411 },
		{ 2248, 946, 411 },
		{ 2284, 967, 411 },
		{ 2321, 975, 411 },
		{ 2262, 1468, 0 },
		{ 2237, 0, 0 },
		{ 2899, 2527, 409 },
		{ 2264, 1469, 0 },
		{ 2970, 2845, 0 },
		{ 0, 0, 379 },
		{ 2330, 975, 411 },
		{ 2837, 2694, 0 },
		{ 2386, 463, 411 },
		{ 2259, 970, 411 },
		{ 2307, 963, 411 },
		{ 2386, 469, 411 },
		{ 2333, 1008, 411 },
		{ 2183, 964, 411 },
		{ 2258, 1009, 411 },
		{ 2333, 1005, 411 },
		{ 2386, 8, 411 },
		{ 2307, 1034, 411 },
		{ 2986, 1803, 0 },
		{ 2945, 2182, 0 },
		{ 2358, 1036, 411 },
		{ 2183, 1040, 411 },
		{ 2352, 1039, 411 },
		{ 2183, 1055, 411 },
		{ 2183, 1047, 411 },
		{ 2183, 1037, 411 },
		{ 2262, 0, 0 },
		{ 2899, 2537, 407 },
		{ 2264, 0, 0 },
		{ 2899, 2547, 408 },
		{ 0, 0, 412 },
		{ 2352, 1069, 411 },
		{ 2289, 4633, 0 },
		{ 2981, 2039, 0 },
		{ 2284, 1087, 411 },
		{ 2386, 123, 411 },
		{ 2951, 1867, 0 },
		{ 2406, 6, 411 },
		{ 2358, 1071, 411 },
		{ 2284, 1091, 411 },
		{ 2307, 1109, 411 },
		{ 2305, 1108, 411 },
		{ 2988, 1926, 0 },
		{ 2333, 1122, 411 },
		{ 2942, 2289, 0 },
		{ 2990, 2021, 0 },
		{ 2942, 2193, 0 },
		{ 2338, 1117, 411 },
		{ 2352, 1115, 411 },
		{ 2183, 1133, 411 },
		{ 2330, 1130, 411 },
		{ 2386, 125, 411 },
		{ 2333, 1153, 411 },
		{ 2386, 128, 411 },
		{ 2983, 2139, 0 },
		{ 2891, 2637, 0 },
		{ 2307, 1143, 411 },
		{ 2951, 1861, 0 },
		{ 2986, 1730, 0 },
		{ 2991, 4345, 0 },
		{ 2945, 4658, 387 },
		{ 2381, 1151, 411 },
		{ 2307, 1145, 411 },
		{ 2333, 1199, 411 },
		{ 2333, 1158, 411 },
		{ 2837, 2692, 0 },
		{ 2338, 1190, 411 },
		{ 2891, 2453, 0 },
		{ 2973, 2743, 0 },
		{ 2891, 2454, 0 },
		{ 2183, 1184, 411 },
		{ 2333, 1198, 411 },
		{ 2183, 1189, 411 },
		{ 2386, 130, 411 },
		{ 2990, 1888, 0 },
		{ 2363, 1223, 411 },
		{ 2988, 1933, 0 },
		{ 2934, 3556, 0 },
		{ 2891, 2593, 0 },
		{ 2975, 2311, 0 },
		{ 2333, 1229, 411 },
		{ 2986, 1878, 0 },
		{ 2984, 2919, 0 },
		{ 2406, 121, 411 },
		{ 2338, 1225, 411 },
		{ 2338, 1227, 411 },
		{ 2183, 1265, 411 },
		{ 2363, 1256, 411 },
		{ 2347, 4536, 0 },
		{ 2363, 1257, 411 },
		{ 2986, 1899, 0 },
		{ 2973, 2779, 0 },
		{ 2986, 1900, 0 },
		{ 2330, 1267, 411 },
		{ 2363, 1259, 411 },
		{ 2183, 1269, 411 },
		{ 2988, 1738, 0 },
		{ 2973, 2792, 0 },
		{ 2183, 1266, 411 },
		{ 2837, 2684, 0 },
		{ 2938, 3193, 0 },
		{ 2986, 1911, 0 },
		{ 2891, 2632, 0 },
		{ 2183, 1261, 411 },
		{ 2984, 2962, 0 },
		{ 2986, 1658, 0 },
		{ 2991, 4475, 0 },
		{ 0, 0, 403 },
		{ 2352, 1259, 411 },
		{ 2363, 1264, 411 },
		{ 2386, 235, 411 },
		{ 2353, 1272, 411 },
		{ 2945, 1907, 0 },
		{ 2386, 237, 411 },
		{ 2372, 4545, 0 },
		{ 2373, 4563, 0 },
		{ 2374, 4550, 0 },
		{ 2183, 1263, 411 },
		{ 2183, 1275, 411 },
		{ 2386, 239, 411 },
		{ 2984, 2911, 0 },
		{ 2837, 2689, 0 },
		{ 2970, 2851, 0 },
		{ 2183, 1264, 411 },
		{ 2382, 4610, 0 },
		{ 2951, 1985, 0 },
		{ 2942, 2201, 0 },
		{ 2986, 1817, 0 },
		{ 2183, 1270, 411 },
		{ 2986, 1833, 0 },
		{ 2951, 1990, 0 },
		{ 2386, 241, 411 },
		{ 2386, 243, 411 },
		{ 2945, 2288, 0 },
		{ 2386, 245, 411 },
		{ 2990, 2014, 0 },
		{ 2945, 1912, 0 },
		{ 2986, 1807, 0 },
		{ 2970, 2511, 0 },
		{ 2986, 1855, 0 },
		{ 2386, 349, 411 },
		{ 2386, 351, 411 },
		{ 2985, 1905, 0 },
		{ 2990, 2025, 0 },
		{ 2837, 2677, 0 },
		{ 2978, 1527, 0 },
		{ 2386, 1451, 411 },
		{ 2988, 1859, 0 },
		{ 2409, 4657, 0 },
		{ 2973, 2741, 0 },
		{ 2991, 4378, 0 },
		{ 2406, 576, 411 },
		{ 2951, 1938, 0 },
		{ 2991, 4411, 0 },
		{ 2945, 2290, 0 },
		{ 2988, 1951, 0 },
		{ 2973, 2746, 0 },
		{ 2988, 1736, 0 },
		{ 2988, 2158, 0 },
		{ 2990, 2038, 0 },
		{ 2990, 2039, 0 },
		{ 2973, 2753, 0 },
		{ 2990, 2041, 0 },
		{ 2945, 1918, 0 },
		{ 2951, 1863, 0 },
		{ 2951, 1948, 0 },
		{ 2942, 2267, 0 },
		{ 2429, 4571, 0 },
		{ 2973, 2765, 0 },
		{ 2951, 1949, 0 },
		{ 2985, 2885, 0 },
		{ 2433, 811, 411 },
		{ 2973, 2769, 0 },
		{ 2751, 2101, 0 },
		{ 2991, 4481, 0 },
		{ 2951, 1951, 0 },
		{ 2945, 4759, 385 },
		{ 2951, 1869, 0 },
		{ 2991, 4259, 0 },
		{ 2440, 10, 394 },
		{ 2988, 2106, 0 },
		{ 2751, 2084, 0 },
		{ 2942, 2282, 0 },
		{ 2984, 2948, 0 },
		{ 2837, 2709, 0 },
		{ 2793, 2668, 0 },
		{ 2988, 2109, 0 },
		{ 2973, 2785, 0 },
		{ 2751, 2085, 0 },
		{ 2973, 2787, 0 },
		{ 2990, 2051, 0 },
		{ 2891, 2516, 0 },
		{ 2955, 1469, 0 },
		{ 2978, 1513, 0 },
		{ 2951, 1871, 0 },
		{ 2942, 2194, 0 },
		{ 2751, 2099, 0 },
		{ 2973, 2799, 0 },
		{ 2991, 4261, 0 },
		{ 2945, 4716, 406 },
		{ 2942, 2196, 0 },
		{ 2986, 1909, 0 },
		{ 0, 0, 417 },
		{ 2951, 1962, 0 },
		{ 2891, 2604, 0 },
		{ 1789, 18, 393 },
		{ 2984, 2915, 0 },
		{ 2973, 2809, 0 },
		{ 2891, 2605, 0 },
		{ 2990, 2063, 0 },
		{ 2837, 2683, 0 },
		{ 2468, 4559, 0 },
		{ 2517, 2894, 0 },
		{ 2973, 2814, 0 },
		{ 2986, 1910, 0 },
		{ 2973, 2817, 0 },
		{ 2988, 2124, 0 },
		{ 2462, 1279, 0 },
		{ 2475, 4556, 0 },
		{ 2751, 2086, 0 },
		{ 2985, 2861, 0 },
		{ 2986, 1917, 0 },
		{ 2990, 2072, 0 },
		{ 2480, 4573, 0 },
		{ 2973, 2825, 0 },
		{ 2891, 2623, 0 },
		{ 2483, 4608, 0 },
		{ 0, 1311, 0 },
		{ 2981, 2398, 0 },
		{ 2990, 2004, 0 },
		{ 2986, 1657, 0 },
		{ 2973, 2834, 0 },
		{ 2951, 1969, 0 },
		{ 2945, 2684, 0 },
		{ 2984, 2959, 0 },
		{ 2492, 4666, 0 },
		{ 2980, 2655, 0 },
		{ 2494, 4545, 0 },
		{ 2517, 2889, 0 },
		{ 2973, 2717, 0 },
		{ 2951, 1873, 0 },
		{ 2981, 2407, 0 },
		{ 2990, 2010, 0 },
		{ 2951, 1971, 0 },
		{ 2891, 2640, 0 },
		{ 2988, 1936, 0 },
		{ 2990, 2016, 0 },
		{ 2975, 2315, 0 },
		{ 2985, 2667, 0 },
		{ 2973, 2732, 0 },
		{ 2991, 4337, 0 },
		{ 2984, 2914, 0 },
		{ 2988, 2146, 0 },
		{ 2942, 2236, 0 },
		{ 2751, 2087, 0 },
		{ 2978, 1578, 0 },
		{ 2517, 2888, 0 },
		{ 2970, 2515, 0 },
		{ 2515, 4648, 0 },
		{ 2970, 2517, 0 },
		{ 2984, 2926, 0 },
		{ 2991, 4450, 0 },
		{ 2986, 1665, 0 },
		{ 2988, 2153, 0 },
		{ 2891, 2507, 0 },
		{ 2522, 4626, 0 },
		{ 2942, 2244, 0 },
		{ 2751, 2094, 0 },
		{ 2984, 2933, 0 },
		{ 2891, 2513, 0 },
		{ 2984, 2936, 0 },
		{ 2991, 4271, 0 },
		{ 2945, 4637, 404 },
		{ 2986, 1666, 0 },
		{ 2990, 2020, 0 },
		{ 2986, 1667, 0 },
		{ 2990, 2023, 0 },
		{ 2837, 2702, 0 },
		{ 2891, 2596, 0 },
		{ 2973, 2756, 0 },
		{ 2991, 4403, 0 },
		{ 2973, 2758, 0 },
		{ 0, 2887, 0 },
		{ 2945, 4655, 392 },
		{ 2984, 2955, 0 },
		{ 2986, 1668, 0 },
		{ 2751, 2082, 0 },
		{ 2988, 1945, 0 },
		{ 2793, 2665, 0 },
		{ 2973, 2763, 0 },
		{ 2986, 1670, 0 },
		{ 2951, 1981, 0 },
		{ 2951, 1982, 0 },
		{ 2945, 4684, 386 },
		{ 2988, 2169, 0 },
		{ 2951, 1983, 0 },
		{ 2951, 1984, 0 },
		{ 2891, 2613, 0 },
		{ 2837, 2695, 0 },
		{ 2981, 2400, 0 },
		{ 2751, 2089, 0 },
		{ 0, 0, 416 },
		{ 2751, 2091, 0 },
		{ 2891, 2620, 0 },
		{ 2986, 1695, 0 },
		{ 2558, 4626, 0 },
		{ 2986, 1699, 0 },
		{ 2751, 2095, 0 },
		{ 2561, 4651, 0 },
		{ 2990, 2036, 0 },
		{ 2891, 2625, 0 },
		{ 2984, 2921, 0 },
		{ 2973, 2789, 0 },
		{ 2990, 2037, 0 },
		{ 2991, 4417, 0 },
		{ 2991, 4419, 0 },
		{ 2942, 2284, 0 },
		{ 2973, 2793, 0 },
		{ 2891, 2630, 0 },
		{ 2986, 1703, 0 },
		{ 2986, 1705, 0 },
		{ 2981, 2358, 0 },
		{ 2951, 1989, 0 },
		{ 2951, 1875, 0 },
		{ 2991, 4265, 0 },
		{ 2973, 2803, 0 },
		{ 2988, 1957, 0 },
		{ 2984, 2940, 0 },
		{ 2988, 2118, 0 },
		{ 2986, 1720, 0 },
		{ 2951, 1995, 0 },
		{ 2991, 4343, 0 },
		{ 2990, 939, 389 },
		{ 2945, 4679, 399 },
		{ 2793, 2673, 0 },
		{ 2990, 2047, 0 },
		{ 2986, 1750, 0 },
		{ 2980, 2657, 0 },
		{ 2980, 2661, 0 },
		{ 2891, 2446, 0 },
		{ 2591, 4612, 0 },
		{ 2985, 2879, 0 },
		{ 2945, 4714, 397 },
		{ 2990, 2049, 0 },
		{ 2751, 2090, 0 },
		{ 2986, 1758, 0 },
		{ 2942, 2207, 0 },
		{ 2891, 2455, 0 },
		{ 2598, 4684, 0 },
		{ 2945, 4740, 388 },
		{ 2991, 4479, 0 },
		{ 2600, 4694, 0 },
		{ 2604, 1434, 0 },
		{ 2602, 4701, 0 },
		{ 2603, 4702, 0 },
		{ 2986, 1773, 0 },
		{ 2983, 2403, 0 },
		{ 2990, 2053, 0 },
		{ 2984, 2906, 0 },
		{ 2973, 2831, 0 },
		{ 2988, 2135, 0 },
		{ 2951, 2001, 0 },
		{ 2988, 2137, 0 },
		{ 2991, 4275, 0 },
		{ 2945, 4642, 401 },
		{ 2991, 4277, 0 },
		{ 2991, 4279, 0 },
		{ 2991, 4304, 0 },
		{ 2991, 4306, 0 },
		{ 0, 1436, 0 },
		{ 2891, 2514, 0 },
		{ 2891, 2515, 0 },
		{ 2986, 1778, 0 },
		{ 2990, 2058, 0 },
		{ 2990, 2059, 0 },
		{ 2991, 4347, 0 },
		{ 2942, 2229, 0 },
		{ 0, 0, 419 },
		{ 0, 0, 418 },
		{ 2945, 4669, 390 },
		{ 0, 0, 414 },
		{ 0, 0, 415 },
		{ 2991, 4351, 0 },
		{ 2981, 2405, 0 },
		{ 2751, 2081, 0 },
		{ 2988, 2145, 0 },
		{ 2984, 2924, 0 },
		{ 2991, 4409, 0 },
		{ 2945, 4687, 384 },
		{ 2631, 4593, 0 },
		{ 2945, 4690, 391 },
		{ 2973, 2725, 0 },
		{ 2986, 1780, 0 },
		{ 2990, 2062, 0 },
		{ 2986, 1784, 0 },
		{ 2945, 4700, 402 },
		{ 2945, 2180, 0 },
		{ 2991, 4421, 0 },
		{ 2991, 4423, 0 },
		{ 2991, 4448, 0 },
		{ 2988, 2152, 0 },
		{ 2986, 1786, 0 },
		{ 2945, 4719, 395 },
		{ 2945, 4721, 396 },
		{ 2945, 4724, 398 },
		{ 2990, 2065, 0 },
		{ 2973, 2738, 0 },
		{ 2991, 4483, 0 },
		{ 2990, 2066, 0 },
		{ 2945, 4735, 400 },
		{ 2984, 2939, 0 },
		{ 2986, 1788, 0 },
		{ 2891, 2615, 0 },
		{ 2988, 2157, 0 },
		{ 2942, 2247, 0 },
		{ 2951, 1937, 0 },
		{ 2991, 4269, 0 },
		{ 2945, 4754, 405 },
		{ 2665, 4555, 420 },
		{ 2658, 0, 381 },
		{ 0, 0, 382 },
		{ -2656, 4764, 377 },
		{ -2657, 4599, 0 },
		{ 2945, 4576, 0 },
		{ 2665, 4564, 0 },
		{ 0, 0, 378 },
		{ 2665, 4558, 0 },
		{ -2662, 4769, 0 },
		{ -2663, 4591, 0 },
		{ 2666, 0, 379 },
		{ 0, 4566, 0 },
		{ 2945, 4645, 0 },
		{ 0, 0, 380 },
		{ 2938, 3231, 144 },
		{ 0, 0, 144 },
		{ 0, 0, 145 },
		{ 2951, 1939, 0 },
		{ 2973, 2748, 0 },
		{ 2990, 2003, 0 },
		{ 2675, 4560, 0 },
		{ 2983, 2401, 0 },
		{ 2978, 1632, 0 },
		{ 2942, 2256, 0 },
		{ 2985, 2883, 0 },
		{ 2986, 1802, 0 },
		{ 2891, 2629, 0 },
		{ 2988, 2165, 0 },
		{ 2942, 2260, 0 },
		{ 2951, 1944, 0 },
		{ 2991, 4376, 0 },
		{ 0, 0, 142 },
		{ 2824, 4337, 167 },
		{ 0, 0, 167 },
		{ 2986, 1808, 0 },
		{ 2690, 4591, 0 },
		{ 2973, 2445, 0 },
		{ 2984, 2904, 0 },
		{ 2985, 2873, 0 },
		{ 2980, 2645, 0 },
		{ 2695, 4599, 0 },
		{ 2945, 2284, 0 },
		{ 2973, 2764, 0 },
		{ 2942, 2266, 0 },
		{ 2973, 2766, 0 },
		{ 2990, 2009, 0 },
		{ 2984, 2913, 0 },
		{ 2986, 1815, 0 },
		{ 2891, 2641, 0 },
		{ 2988, 2172, 0 },
		{ 2942, 2271, 0 },
		{ 2706, 4622, 0 },
		{ 2945, 2682, 0 },
		{ 2973, 2774, 0 },
		{ 2837, 2697, 0 },
		{ 2988, 2173, 0 },
		{ 2990, 2012, 0 },
		{ 2973, 2778, 0 },
		{ 2713, 4625, 0 },
		{ 2990, 1900, 0 },
		{ 2973, 2780, 0 },
		{ 2970, 2842, 0 },
		{ 2978, 1634, 0 },
		{ 2985, 2862, 0 },
		{ 2973, 2782, 0 },
		{ 2720, 4653, 0 },
		{ 2983, 2391, 0 },
		{ 2978, 1650, 0 },
		{ 2942, 2280, 0 },
		{ 2985, 2871, 0 },
		{ 2986, 1837, 0 },
		{ 2891, 2500, 0 },
		{ 2988, 2108, 0 },
		{ 2942, 2285, 0 },
		{ 2991, 4341, 0 },
		{ 0, 0, 165 },
		{ 2731, 0, 1 },
		{ -2731, 1275, 256 },
		{ 2973, 2692, 262 },
		{ 0, 0, 262 },
		{ 2951, 1957, 0 },
		{ 2942, 2191, 0 },
		{ 2973, 2798, 0 },
		{ 2970, 2849, 0 },
		{ 2990, 2022, 0 },
		{ 0, 0, 261 },
		{ 2741, 4582, 0 },
		{ 2975, 1902, 0 },
		{ 2984, 2961, 0 },
		{ 2770, 2421, 0 },
		{ 2973, 2804, 0 },
		{ 2837, 2700, 0 },
		{ 2891, 2590, 0 },
		{ 2981, 2372, 0 },
		{ 2973, 2808, 0 },
		{ 2750, 4562, 0 },
		{ 2988, 1965, 0 },
		{ 0, 2093, 0 },
		{ 2986, 1864, 0 },
		{ 2891, 2597, 0 },
		{ 2988, 2119, 0 },
		{ 2942, 2198, 0 },
		{ 2951, 1963, 0 },
		{ 2991, 4487, 0 },
		{ 0, 0, 260 },
		{ 0, 4201, 170 },
		{ 0, 0, 170 },
		{ 2988, 2121, 0 },
		{ 2978, 1575, 0 },
		{ 2942, 2202, 0 },
		{ 2970, 2855, 0 },
		{ 2766, 4604, 0 },
		{ 2985, 2524, 0 },
		{ 2980, 2647, 0 },
		{ 2973, 2824, 0 },
		{ 2985, 2878, 0 },
		{ 0, 2422, 0 },
		{ 2891, 2610, 0 },
		{ 2942, 2203, 0 },
		{ 2793, 2664, 0 },
		{ 2991, 4333, 0 },
		{ 0, 0, 168 },
		{ 2824, 4381, 164 },
		{ 0, 0, 163 },
		{ 0, 0, 164 },
		{ 2986, 1869, 0 },
		{ 2781, 4611, 0 },
		{ 2986, 1831, 0 },
		{ 2980, 2656, 0 },
		{ 2973, 2833, 0 },
		{ 2785, 4638, 0 },
		{ 2945, 2680, 0 },
		{ 2973, 2835, 0 },
		{ 2793, 2670, 0 },
		{ 2891, 2616, 0 },
		{ 2942, 2209, 0 },
		{ 2942, 2211, 0 },
		{ 2891, 2619, 0 },
		{ 2942, 2213, 0 },
		{ 0, 2666, 0 },
		{ 2795, 4643, 0 },
		{ 2988, 1920, 0 },
		{ 2837, 2688, 0 },
		{ 2798, 4657, 0 },
		{ 2973, 2443, 0 },
		{ 2984, 2944, 0 },
		{ 2985, 2860, 0 },
		{ 2980, 2654, 0 },
		{ 2803, 4662, 0 },
		{ 2945, 2282, 0 },
		{ 2973, 2726, 0 },
		{ 2942, 2216, 0 },
		{ 2973, 2728, 0 },
		{ 2990, 2031, 0 },
		{ 2984, 2957, 0 },
		{ 2986, 1877, 0 },
		{ 2891, 2627, 0 },
		{ 2988, 2128, 0 },
		{ 2942, 2220, 0 },
		{ 2814, 4551, 0 },
		{ 2983, 2405, 0 },
		{ 2978, 1579, 0 },
		{ 2942, 2222, 0 },
		{ 2985, 2880, 0 },
		{ 2986, 1880, 0 },
		{ 2891, 2635, 0 },
		{ 2988, 2132, 0 },
		{ 2942, 2225, 0 },
		{ 2991, 4273, 0 },
		{ 0, 0, 157 },
		{ 0, 4255, 156 },
		{ 0, 0, 156 },
		{ 2986, 1882, 0 },
		{ 2828, 4554, 0 },
		{ 2986, 1835, 0 },
		{ 2980, 2646, 0 },
		{ 2973, 2747, 0 },
		{ 2832, 4575, 0 },
		{ 2973, 2449, 0 },
		{ 2942, 2228, 0 },
		{ 2970, 2857, 0 },
		{ 2836, 4570, 0 },
		{ 2988, 1931, 0 },
		{ 0, 2693, 0 },
		{ 2839, 4584, 0 },
		{ 2973, 2439, 0 },
		{ 2984, 2917, 0 },
		{ 2985, 2875, 0 },
		{ 2980, 2651, 0 },
		{ 2844, 4592, 0 },
		{ 2945, 2286, 0 },
		{ 2973, 2755, 0 },
		{ 2942, 2231, 0 },
		{ 2973, 2757, 0 },
		{ 2990, 2040, 0 },
		{ 2984, 2925, 0 },
		{ 2986, 1885, 0 },
		{ 2891, 2450, 0 },
		{ 2988, 2138, 0 },
		{ 2942, 2235, 0 },
		{ 2855, 4610, 0 },
		{ 2983, 2399, 0 },
		{ 2978, 1597, 0 },
		{ 2942, 2237, 0 },
		{ 2985, 2866, 0 },
		{ 2986, 1889, 0 },
		{ 2891, 2501, 0 },
		{ 2988, 2141, 0 },
		{ 2942, 2240, 0 },
		{ 2991, 4477, 0 },
		{ 0, 0, 154 },
		{ 0, 3831, 159 },
		{ 0, 0, 159 },
		{ 0, 0, 160 },
		{ 2942, 2241, 0 },
		{ 2951, 1977, 0 },
		{ 2986, 1890, 0 },
		{ 2973, 2773, 0 },
		{ 2984, 2946, 0 },
		{ 2970, 2848, 0 },
		{ 2875, 4642, 0 },
		{ 2973, 2437, 0 },
		{ 2955, 1472, 0 },
		{ 2984, 2950, 0 },
		{ 2981, 2403, 0 },
		{ 2978, 1599, 0 },
		{ 2984, 2954, 0 },
		{ 2986, 1894, 0 },
		{ 2891, 2517, 0 },
		{ 2988, 2147, 0 },
		{ 2942, 2248, 0 },
		{ 2886, 4659, 0 },
		{ 2983, 2397, 0 },
		{ 2978, 1600, 0 },
		{ 2942, 2250, 0 },
		{ 2985, 2868, 0 },
		{ 2986, 1897, 0 },
		{ 0, 2598, 0 },
		{ 2988, 2150, 0 },
		{ 2942, 2253, 0 },
		{ 2953, 4503, 0 },
		{ 0, 0, 158 },
		{ 2973, 2791, 420 },
		{ 2990, 1496, 25 },
		{ 2905, 0, 420 },
		{ 2182, 2596, 27 },
		{ 0, 0, 28 },
		{ 0, 0, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 2942, 2257, 0 },
		{ 2990, 661, 0 },
		{ 0, 0, 26 },
		{ 2970, 2854, 0 },
		{ 0, 0, 21 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 0, 3673, 37 },
		{ 0, 3420, 37 },
		{ 0, 0, 36 },
		{ 0, 0, 37 },
		{ 2934, 3552, 0 },
		{ 2946, 4174, 0 },
		{ 2938, 3206, 0 },
		{ 0, 0, 35 },
		{ 2941, 3271, 0 },
		{ 0, 3120, 0 },
		{ 2899, 1607, 0 },
		{ 0, 0, 34 },
		{ 2973, 2712, 47 },
		{ 0, 0, 47 },
		{ 2938, 3214, 47 },
		{ 2973, 2801, 47 },
		{ 0, 0, 50 },
		{ 2973, 2802, 0 },
		{ 2942, 2262, 0 },
		{ 2941, 3280, 0 },
		{ 2986, 1907, 0 },
		{ 2942, 2264, 0 },
		{ 2970, 2844, 0 },
		{ 0, 3515, 0 },
		{ 2978, 1635, 0 },
		{ 2988, 2160, 0 },
		{ 0, 0, 46 },
		{ 0, 3225, 0 },
		{ 2990, 2061, 0 },
		{ 2975, 2310, 0 },
		{ 0, 3291, 0 },
		{ 0, 2268, 0 },
		{ 2973, 2811, 0 },
		{ 0, 0, 48 },
		{ 0, 5, 51 },
		{ 0, 4139, 0 },
		{ 0, 0, 49 },
		{ 2981, 2384, 0 },
		{ 2984, 2929, 0 },
		{ 2951, 1993, 0 },
		{ 0, 1994, 0 },
		{ 2953, 4507, 0 },
		{ 0, 4508, 0 },
		{ 2973, 2815, 0 },
		{ 0, 1506, 0 },
		{ 2984, 2934, 0 },
		{ 2981, 2392, 0 },
		{ 2978, 1652, 0 },
		{ 2984, 2937, 0 },
		{ 2986, 1913, 0 },
		{ 2988, 2167, 0 },
		{ 2990, 2067, 0 },
		{ 2984, 2023, 0 },
		{ 2973, 2823, 0 },
		{ 2988, 2170, 0 },
		{ 2985, 2867, 0 },
		{ 2984, 2945, 0 },
		{ 2990, 2069, 0 },
		{ 2985, 2869, 0 },
		{ 0, 2846, 0 },
		{ 2973, 2730, 0 },
		{ 2978, 1654, 0 },
		{ 0, 2828, 0 },
		{ 2984, 2952, 0 },
		{ 0, 2331, 0 },
		{ 2990, 2071, 0 },
		{ 2985, 2876, 0 },
		{ 0, 1655, 0 },
		{ 2991, 4514, 0 },
		{ 0, 2653, 0 },
		{ 0, 2408, 0 },
		{ 0, 0, 43 },
		{ 2945, 2679, 0 },
		{ 0, 2960, 0 },
		{ 0, 2881, 0 },
		{ 0, 1919, 0 },
		{ 2991, 4516, 0 },
		{ 0, 2178, 0 },
		{ 0, 0, 44 },
		{ 0, 2074, 0 },
		{ 2953, 4517, 0 },
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
		0
	};
	yybackup = backup;
}
