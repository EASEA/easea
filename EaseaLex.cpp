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
#line 1118 "EaseaLex.l"
if(strlen(sIP_FILE)>0)fprintf(fpOutputFile,"%s",sIP_FILE); else fprintf(fpOutputFile,"NULL");
#line 1740 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1120 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1747 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1121 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1754 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1122 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1761 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1123 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1768 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1124 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1775 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1126 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 1782 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1127 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 1789 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1129 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1803 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1137 "EaseaLex.l"

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
 
#line 1823 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1151 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1837 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1159 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1851 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1168 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1865 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1177 "EaseaLex.l"

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

#line 1928 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1234 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded with no warning.\n");
  printf ("Have a nice compile time.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 1945 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1246 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1952 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1252 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1964 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1258 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 1977 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1265 "EaseaLex.l"

#line 1984 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1266 "EaseaLex.l"
lineCounter++;
#line 1991 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1268 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2003 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1274 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2016 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1282 "EaseaLex.l"

#line 2023 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1283 "EaseaLex.l"

  lineCounter++;
 
#line 2032 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1287 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2044 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1293 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2058 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1301 "EaseaLex.l"

#line 2065 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1302 "EaseaLex.l"

  lineCounter++;
 
#line 2074 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1306 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  
  BEGIN COPY;
 
#line 2088 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1314 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2103 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1323 "EaseaLex.l"

#line 2110 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1324 "EaseaLex.l"
lineCounter++;
#line 2117 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1329 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2131 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1338 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2145 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1346 "EaseaLex.l"

#line 2152 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1347 "EaseaLex.l"
lineCounter++;
#line 2159 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1350 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2175 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1361 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2191 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1371 "EaseaLex.l"

#line 2198 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1374 "EaseaLex.l"

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
 
#line 2216 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1387 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2233 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1399 "EaseaLex.l"

#line 2240 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1400 "EaseaLex.l"
lineCounter++;
#line 2247 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1402 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2263 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1414 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2279 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1424 "EaseaLex.l"
lineCounter++;
#line 2286 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1425 "EaseaLex.l"

#line 2293 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1429 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2308 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1439 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2323 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1448 "EaseaLex.l"

#line 2330 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1451 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2343 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1458 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2357 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1466 "EaseaLex.l"

#line 2364 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1470 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2372 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1472 "EaseaLex.l"

#line 2379 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1478 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2386 "EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 1479 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2393 "EaseaLex.cpp"
		}
		break;
	case 180:
	case 181:
		{
#line 1482 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2404 "EaseaLex.cpp"
		}
		break;
	case 182:
	case 183:
		{
#line 1487 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2413 "EaseaLex.cpp"
		}
		break;
	case 184:
	case 185:
		{
#line 1490 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2422 "EaseaLex.cpp"
		}
		break;
	case 186:
	case 187:
		{
#line 1493 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2439 "EaseaLex.cpp"
		}
		break;
	case 188:
	case 189:
		{
#line 1504 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2453 "EaseaLex.cpp"
		}
		break;
	case 190:
	case 191:
		{
#line 1512 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2462 "EaseaLex.cpp"
		}
		break;
	case 192:
	case 193:
		{
#line 1515 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2471 "EaseaLex.cpp"
		}
		break;
	case 194:
	case 195:
		{
#line 1518 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2480 "EaseaLex.cpp"
		}
		break;
	case 196:
	case 197:
		{
#line 1521 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2489 "EaseaLex.cpp"
		}
		break;
	case 198:
	case 199:
		{
#line 1524 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2498 "EaseaLex.cpp"
		}
		break;
	case 200:
	case 201:
		{
#line 1528 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2510 "EaseaLex.cpp"
		}
		break;
	case 202:
		{
#line 1534 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2517 "EaseaLex.cpp"
		}
		break;
	case 203:
		{
#line 1535 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2524 "EaseaLex.cpp"
		}
		break;
	case 204:
		{
#line 1536 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2531 "EaseaLex.cpp"
		}
		break;
	case 205:
		{
#line 1537 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2541 "EaseaLex.cpp"
		}
		break;
	case 206:
		{
#line 1542 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2548 "EaseaLex.cpp"
		}
		break;
	case 207:
		{
#line 1543 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
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
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2583 "EaseaLex.cpp"
		}
		break;
	case 212:
		{
#line 1548 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2590 "EaseaLex.cpp"
		}
		break;
	case 213:
		{
#line 1549 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2597 "EaseaLex.cpp"
		}
		break;
	case 214:
		{
#line 1550 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2605 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1552 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2613 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1554 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2621 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1556 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2631 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1560 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2638 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1561 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2645 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1562 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2656 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1567 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2663 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1568 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2672 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1571 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2684 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1577 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2693 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1580 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2705 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1586 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2716 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1591 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2732 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1601 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2739 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1604 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2748 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1607 "EaseaLex.l"
BEGIN COPY;
#line 2755 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1609 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2762 "EaseaLex.cpp"
		}
		break;
	case 232:
	case 233:
	case 234:
		{
#line 1612 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2775 "EaseaLex.cpp"
		}
		break;
	case 235:
		{
#line 1617 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2786 "EaseaLex.cpp"
		}
		break;
	case 236:
		{
#line 1622 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
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
#line 1634 "EaseaLex.l"
;
#line 2823 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1637 "EaseaLex.l"
 /* do nothing */ 
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
#line 1639 "EaseaLex.l"
 /*return '\n';*/ 
#line 2844 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1642 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 2853 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1645 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 2863 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1649 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
  printf("match gpnode\n");
  return GPNODE;
 
#line 2875 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1656 "EaseaLex.l"
return STATIC;
#line 2882 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1657 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 2889 "EaseaLex.cpp"
		}
		break;
	case 249:
		{
#line 1658 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 2896 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1659 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 2903 "EaseaLex.cpp"
		}
		break;
	case 251:
		{
#line 1660 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 2910 "EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1661 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 2917 "EaseaLex.cpp"
		}
		break;
	case 253:
		{
#line 1663 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 2924 "EaseaLex.cpp"
		}
		break;
#line 1664 "EaseaLex.l"
  
#line 2929 "EaseaLex.cpp"
	case 254:
		{
#line 1665 "EaseaLex.l"
return GENOME; 
#line 2934 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1667 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 2944 "EaseaLex.cpp"
		}
		break;
	case 256:
	case 257:
	case 258:
		{
#line 1674 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 2953 "EaseaLex.cpp"
		}
		break;
	case 259:
		{
#line 1675 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 2960 "EaseaLex.cpp"
		}
		break;
	case 260:
		{
#line 1678 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 2968 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1680 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 2975 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1686 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 2987 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1692 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 3000 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1699 "EaseaLex.l"

#line 3007 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1701 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3018 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1712 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3033 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1722 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3044 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1728 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3053 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1732 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3068 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1745 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3080 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1751 "EaseaLex.l"

#line 3087 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1752 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3100 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1759 "EaseaLex.l"

#line 3107 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1760 "EaseaLex.l"
lineCounter++;
#line 3114 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1761 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3127 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1768 "EaseaLex.l"

#line 3134 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1769 "EaseaLex.l"
lineCounter++;
#line 3141 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1771 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3154 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1778 "EaseaLex.l"

#line 3161 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1779 "EaseaLex.l"
lineCounter++;
#line 3168 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1781 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3181 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1788 "EaseaLex.l"

#line 3188 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1789 "EaseaLex.l"
lineCounter++;
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
fprintf(fpOutputFile,yytext);
#line 3223 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1799 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3230 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1800 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3237 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1801 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3244 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1803 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3253 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1806 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3266 "EaseaLex.cpp"
		}
		break;
	case 293:
	case 294:
		{
#line 1815 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3277 "EaseaLex.cpp"
		}
		break;
	case 295:
	case 296:
		{
#line 1820 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3286 "EaseaLex.cpp"
		}
		break;
	case 297:
	case 298:
		{
#line 1823 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3295 "EaseaLex.cpp"
		}
		break;
	case 299:
	case 300:
		{
#line 1826 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3307 "EaseaLex.cpp"
		}
		break;
	case 301:
	case 302:
		{
#line 1832 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3320 "EaseaLex.cpp"
		}
		break;
	case 303:
	case 304:
		{
#line 1839 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3329 "EaseaLex.cpp"
		}
		break;
	case 305:
	case 306:
		{
#line 1842 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3338 "EaseaLex.cpp"
		}
		break;
	case 307:
	case 308:
		{
#line 1845 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3347 "EaseaLex.cpp"
		}
		break;
	case 309:
	case 310:
		{
#line 1848 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3356 "EaseaLex.cpp"
		}
		break;
	case 311:
	case 312:
		{
#line 1851 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3365 "EaseaLex.cpp"
		}
		break;
	case 313:
		{
#line 1854 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3374 "EaseaLex.cpp"
		}
		break;
	case 314:
		{
#line 1857 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3384 "EaseaLex.cpp"
		}
		break;
	case 315:
		{
#line 1861 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3392 "EaseaLex.cpp"
		}
		break;
	case 316:
		{
#line 1863 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3403 "EaseaLex.cpp"
		}
		break;
	case 317:
		{
#line 1868 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3414 "EaseaLex.cpp"
		}
		break;
	case 318:
		{
#line 1873 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3422 "EaseaLex.cpp"
		}
		break;
	case 319:
		{
#line 1875 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3430 "EaseaLex.cpp"
		}
		break;
	case 320:
		{
#line 1877 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3438 "EaseaLex.cpp"
		}
		break;
	case 321:
		{
#line 1879 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3446 "EaseaLex.cpp"
		}
		break;
	case 322:
		{
#line 1881 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3454 "EaseaLex.cpp"
		}
		break;
	case 323:
		{
#line 1883 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3461 "EaseaLex.cpp"
		}
		break;
	case 324:
		{
#line 1884 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3468 "EaseaLex.cpp"
		}
		break;
	case 325:
		{
#line 1885 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3476 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1887 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3484 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1889 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3492 "EaseaLex.cpp"
		}
		break;
	case 328:
		{
#line 1891 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3499 "EaseaLex.cpp"
		}
		break;
	case 329:
		{
#line 1892 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3511 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1898 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3520 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1901 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3530 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 1905 "EaseaLex.l"
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
#line 3547 "EaseaLex.cpp"
		}
		break;
	case 333:
	case 334:
		{
#line 1917 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3557 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 1920 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
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
fprintf(fpOutputFile,yytext);
#line 3578 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1929 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
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
#line 1931 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3599 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 1933 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3608 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 1937 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3621 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 1945 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3634 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 1954 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3647 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 1963 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3662 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 1973 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3669 "EaseaLex.cpp"
		}
		break;
	case 347:
		{
#line 1974 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3676 "EaseaLex.cpp"
		}
		break;
	case 348:
	case 349:
		{
#line 1977 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3687 "EaseaLex.cpp"
		}
		break;
	case 350:
	case 351:
		{
#line 1982 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3696 "EaseaLex.cpp"
		}
		break;
	case 352:
	case 353:
		{
#line 1985 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3705 "EaseaLex.cpp"
		}
		break;
	case 354:
	case 355:
		{
#line 1988 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3718 "EaseaLex.cpp"
		}
		break;
	case 356:
	case 357:
		{
#line 1995 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3731 "EaseaLex.cpp"
		}
		break;
	case 358:
	case 359:
		{
#line 2002 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3740 "EaseaLex.cpp"
		}
		break;
	case 360:
		{
#line 2005 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3747 "EaseaLex.cpp"
		}
		break;
	case 361:
		{
#line 2006 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3754 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 2007 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3761 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 2008 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3771 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 2013 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3778 "EaseaLex.cpp"
		}
		break;
	case 365:
		{
#line 2014 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3785 "EaseaLex.cpp"
		}
		break;
	case 366:
		{
#line 2015 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3792 "EaseaLex.cpp"
		}
		break;
	case 367:
		{
#line 2016 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3799 "EaseaLex.cpp"
		}
		break;
	case 368:
		{
#line 2017 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3807 "EaseaLex.cpp"
		}
		break;
	case 369:
		{
#line 2019 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3815 "EaseaLex.cpp"
		}
		break;
	case 370:
		{
#line 2021 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3823 "EaseaLex.cpp"
		}
		break;
	case 371:
		{
#line 2023 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3831 "EaseaLex.cpp"
		}
		break;
	case 372:
		{
#line 2025 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 3839 "EaseaLex.cpp"
		}
		break;
	case 373:
		{
#line 2027 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 3847 "EaseaLex.cpp"
		}
		break;
	case 374:
		{
#line 2029 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 3855 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2031 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3862 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2032 "EaseaLex.l"
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
#line 3885 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2049 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 3896 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2054 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 3910 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2062 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3917 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2068 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 3927 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2072 "EaseaLex.l"

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
#line 2078 "EaseaLex.l"
;
#line 3962 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2080 "EaseaLex.l"
 /* do nothing */ 
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
#line 2082 "EaseaLex.l"
 /*return '\n';*/ 
#line 3983 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2084 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 3990 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2085 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 3997 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2086 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 4004 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2087 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4011 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2088 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4018 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2089 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4025 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2090 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4032 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2091 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4039 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2092 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4046 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2094 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4053 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2095 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4060 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2096 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4067 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2097 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4074 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2098 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4081 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2100 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4088 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2101 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4095 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2103 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4106 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2108 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4113 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2110 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4124 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2115 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4131 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2118 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4138 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2119 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4145 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2120 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4152 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2121 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4159 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2122 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4166 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2124 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4173 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2125 "EaseaLex.l"
if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;
#line 4180 "EaseaLex.cpp"
		}
		break;
#line 2127 "EaseaLex.l"
 
#line 4185 "EaseaLex.cpp"
	case 416:
	case 417:
	case 418:
		{
#line 2131 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4192 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2132 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4199 "EaseaLex.cpp"
		}
		break;
	case 420:
		{
#line 2135 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4207 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2138 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4214 "EaseaLex.cpp"
		}
		break;
	case 422:
		{
#line 2140 "EaseaLex.l"

  lineCounter++;

#line 4223 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2143 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4233 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2148 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4243 "EaseaLex.cpp"
		}
		break;
	case 425:
		{
#line 2153 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4253 "EaseaLex.cpp"
		}
		break;
	case 426:
		{
#line 2158 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4263 "EaseaLex.cpp"
		}
		break;
	case 427:
		{
#line 2163 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4273 "EaseaLex.cpp"
		}
		break;
	case 428:
		{
#line 2168 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4283 "EaseaLex.cpp"
		}
		break;
	case 429:
		{
#line 2177 "EaseaLex.l"
return  (char)yytext[0];
#line 4290 "EaseaLex.cpp"
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
#line 2179 "EaseaLex.l"


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

#line 4478 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
		0,
		180,
		-181,
		0,
		182,
		-183,
		0,
		184,
		-185,
		0,
		186,
		-187,
		0,
		192,
		-193,
		0,
		194,
		-195,
		0,
		196,
		-197,
		0,
		198,
		-199,
		0,
		190,
		-191,
		0,
		188,
		-189,
		0,
		-230,
		0,
		-236,
		0,
		305,
		-306,
		0,
		307,
		-308,
		0,
		309,
		-310,
		0,
		311,
		-312,
		0,
		295,
		-296,
		0,
		358,
		-359,
		0,
		303,
		-304,
		0,
		356,
		-357,
		0,
		301,
		-302,
		0,
		350,
		-351,
		0,
		352,
		-353,
		0,
		354,
		-355,
		0,
		348,
		-349,
		0,
		297,
		-298,
		0,
		299,
		-300,
		0,
		293,
		-294,
		0
	};
	yymatch = match;

	yytransitionmax = 4958;
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
		{ 3009, 61 },
		{ 3009, 61 },
		{ 1861, 1964 },
		{ 1501, 1501 },
		{ 67, 61 },
		{ 2363, 2337 },
		{ 2363, 2337 },
		{ 2336, 2307 },
		{ 2336, 2307 },
		{ 0, 1800 },
		{ 2186, 2182 },
		{ 71, 3 },
		{ 72, 3 },
		{ 2220, 43 },
		{ 2221, 43 },
		{ 1983, 39 },
		{ 69, 1 },
		{ 0, 1991 },
		{ 1960, 1962 },
		{ 67, 1 },
		{ 2769, 2765 },
		{ 165, 167 },
		{ 1960, 1956 },
		{ 3009, 61 },
		{ 2203, 2202 },
		{ 3007, 61 },
		{ 1501, 1501 },
		{ 3056, 3054 },
		{ 2363, 2337 },
		{ 1343, 1342 },
		{ 2336, 2307 },
		{ 1334, 1333 },
		{ 1486, 1469 },
		{ 1487, 1469 },
		{ 71, 3 },
		{ 3011, 61 },
		{ 2220, 43 },
		{ 1917, 1907 },
		{ 1983, 39 },
		{ 3008, 61 },
		{ 3008, 61 },
		{ 3008, 61 },
		{ 3008, 61 },
		{ 3008, 61 },
		{ 3008, 61 },
		{ 3008, 61 },
		{ 3008, 61 },
		{ 3008, 61 },
		{ 3008, 61 },
		{ 70, 3 },
		{ 3010, 61 },
		{ 2219, 43 },
		{ 1576, 1570 },
		{ 1969, 39 },
		{ 2364, 2337 },
		{ 1486, 1469 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 1578, 1572 },
		{ 3005, 61 },
		{ 1488, 1469 },
		{ 1449, 1428 },
		{ 3006, 61 },
		{ 1450, 1429 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3006, 61 },
		{ 3012, 61 },
		{ 2181, 40 },
		{ 1523, 1507 },
		{ 1524, 1507 },
		{ 1443, 1421 },
		{ 1968, 40 },
		{ 2418, 2391 },
		{ 2418, 2391 },
		{ 2341, 2311 },
		{ 2341, 2311 },
		{ 2344, 2314 },
		{ 2344, 2314 },
		{ 1794, 37 },
		{ 2361, 2335 },
		{ 2361, 2335 },
		{ 2369, 2342 },
		{ 2369, 2342 },
		{ 1444, 1422 },
		{ 1445, 1423 },
		{ 1446, 1424 },
		{ 1448, 1427 },
		{ 1451, 1430 },
		{ 1452, 1431 },
		{ 1453, 1432 },
		{ 2181, 40 },
		{ 1523, 1507 },
		{ 1971, 40 },
		{ 1454, 1433 },
		{ 1455, 1434 },
		{ 2418, 2391 },
		{ 1456, 1435 },
		{ 2341, 2311 },
		{ 1457, 1436 },
		{ 2344, 2314 },
		{ 1458, 1437 },
		{ 1794, 37 },
		{ 2361, 2335 },
		{ 1459, 1439 },
		{ 2369, 2342 },
		{ 2180, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1781, 37 },
		{ 1984, 40 },
		{ 1462, 1442 },
		{ 1463, 1443 },
		{ 1525, 1507 },
		{ 2419, 2391 },
		{ 1464, 1444 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1970, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1978, 40 },
		{ 1976, 40 },
		{ 1989, 40 },
		{ 1977, 40 },
		{ 1989, 40 },
		{ 1980, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1979, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1465, 1445 },
		{ 1972, 40 },
		{ 1974, 40 },
		{ 1466, 1446 },
		{ 1989, 40 },
		{ 1468, 1448 },
		{ 1989, 40 },
		{ 1987, 40 },
		{ 1975, 40 },
		{ 1989, 40 },
		{ 1988, 40 },
		{ 1981, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1986, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1973, 40 },
		{ 1989, 40 },
		{ 1985, 40 },
		{ 1989, 40 },
		{ 1982, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1989, 40 },
		{ 1370, 21 },
		{ 1526, 1508 },
		{ 1527, 1508 },
		{ 1469, 1449 },
		{ 1357, 21 },
		{ 2381, 2354 },
		{ 2381, 2354 },
		{ 2384, 2357 },
		{ 2384, 2357 },
		{ 2406, 2379 },
		{ 2406, 2379 },
		{ 2407, 2380 },
		{ 2407, 2380 },
		{ 2449, 2422 },
		{ 2449, 2422 },
		{ 2454, 2427 },
		{ 2454, 2427 },
		{ 1470, 1450 },
		{ 1471, 1451 },
		{ 1472, 1452 },
		{ 1473, 1453 },
		{ 1474, 1454 },
		{ 1475, 1455 },
		{ 1370, 21 },
		{ 1526, 1508 },
		{ 1358, 21 },
		{ 1371, 21 },
		{ 1476, 1456 },
		{ 2381, 2354 },
		{ 1477, 1457 },
		{ 2384, 2357 },
		{ 1478, 1459 },
		{ 2406, 2379 },
		{ 1481, 1462 },
		{ 2407, 2380 },
		{ 1482, 1463 },
		{ 2449, 2422 },
		{ 1483, 1464 },
		{ 2454, 2427 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1484, 1466 },
		{ 1485, 1468 },
		{ 1382, 1360 },
		{ 1489, 1470 },
		{ 1528, 1508 },
		{ 1490, 1471 },
		{ 1495, 1474 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1374, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1363, 21 },
		{ 1361, 21 },
		{ 1376, 21 },
		{ 1362, 21 },
		{ 1376, 21 },
		{ 1365, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1364, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1496, 1475 },
		{ 1359, 21 },
		{ 1372, 21 },
		{ 1497, 1476 },
		{ 1366, 21 },
		{ 1498, 1477 },
		{ 1376, 21 },
		{ 1377, 21 },
		{ 1360, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1367, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1375, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1378, 21 },
		{ 1376, 21 },
		{ 1373, 21 },
		{ 1376, 21 },
		{ 1368, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1376, 21 },
		{ 1955, 38 },
		{ 1529, 1509 },
		{ 1530, 1509 },
		{ 1491, 1472 },
		{ 1780, 38 },
		{ 2460, 2433 },
		{ 2460, 2433 },
		{ 2473, 2447 },
		{ 2473, 2447 },
		{ 1493, 1473 },
		{ 1492, 1472 },
		{ 2474, 2448 },
		{ 2474, 2448 },
		{ 2478, 2452 },
		{ 2478, 2452 },
		{ 1499, 1478 },
		{ 1494, 1473 },
		{ 1502, 1482 },
		{ 1503, 1483 },
		{ 1504, 1484 },
		{ 1505, 1485 },
		{ 1507, 1489 },
		{ 1508, 1490 },
		{ 1955, 38 },
		{ 1529, 1509 },
		{ 1785, 38 },
		{ 2484, 2458 },
		{ 2484, 2458 },
		{ 2460, 2433 },
		{ 1509, 1491 },
		{ 2473, 2447 },
		{ 1510, 1492 },
		{ 1532, 1510 },
		{ 1533, 1510 },
		{ 2474, 2448 },
		{ 1511, 1493 },
		{ 2478, 2452 },
		{ 1512, 1494 },
		{ 1954, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 2484, 2458 },
		{ 1795, 38 },
		{ 1513, 1495 },
		{ 1514, 1496 },
		{ 1531, 1509 },
		{ 1515, 1497 },
		{ 1532, 1510 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1782, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1790, 38 },
		{ 1788, 38 },
		{ 1798, 38 },
		{ 1789, 38 },
		{ 1798, 38 },
		{ 1792, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1791, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1516, 1498 },
		{ 1786, 38 },
		{ 1534, 1510 },
		{ 1517, 1499 },
		{ 1798, 38 },
		{ 1519, 1502 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1787, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1783, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1784, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1797, 38 },
		{ 1798, 38 },
		{ 1796, 38 },
		{ 1798, 38 },
		{ 1793, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 1798, 38 },
		{ 2763, 44 },
		{ 2764, 44 },
		{ 1535, 1511 },
		{ 1536, 1511 },
		{ 67, 44 },
		{ 2485, 2459 },
		{ 2485, 2459 },
		{ 1520, 1503 },
		{ 1521, 1504 },
		{ 1522, 1505 },
		{ 1385, 1361 },
		{ 2273, 2245 },
		{ 2273, 2245 },
		{ 2295, 2266 },
		{ 2295, 2266 },
		{ 1386, 1362 },
		{ 1390, 1364 },
		{ 1391, 1365 },
		{ 1392, 1366 },
		{ 1541, 1513 },
		{ 1542, 1514 },
		{ 1543, 1515 },
		{ 1545, 1519 },
		{ 2763, 44 },
		{ 1546, 1520 },
		{ 1535, 1511 },
		{ 2316, 2286 },
		{ 2316, 2286 },
		{ 2485, 2459 },
		{ 1547, 1521 },
		{ 1538, 1512 },
		{ 1539, 1512 },
		{ 1556, 1542 },
		{ 1557, 1542 },
		{ 2273, 2245 },
		{ 1548, 1522 },
		{ 2295, 2266 },
		{ 2236, 44 },
		{ 2762, 44 },
		{ 2235, 44 },
		{ 2235, 44 },
		{ 2235, 44 },
		{ 2235, 44 },
		{ 2235, 44 },
		{ 2235, 44 },
		{ 2235, 44 },
		{ 2235, 44 },
		{ 2235, 44 },
		{ 2235, 44 },
		{ 2316, 2286 },
		{ 1555, 1541 },
		{ 1393, 1367 },
		{ 1559, 1543 },
		{ 1538, 1512 },
		{ 1537, 1511 },
		{ 1556, 1542 },
		{ 2237, 44 },
		{ 2233, 44 },
		{ 2228, 44 },
		{ 2237, 44 },
		{ 2225, 44 },
		{ 2232, 44 },
		{ 2230, 44 },
		{ 2237, 44 },
		{ 2234, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2227, 44 },
		{ 2222, 44 },
		{ 2229, 44 },
		{ 2224, 44 },
		{ 2237, 44 },
		{ 2231, 44 },
		{ 2226, 44 },
		{ 2223, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 1540, 1512 },
		{ 2241, 44 },
		{ 1558, 1542 },
		{ 1561, 1545 },
		{ 2237, 44 },
		{ 1562, 1546 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2238, 44 },
		{ 2239, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2240, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 2237, 44 },
		{ 159, 4 },
		{ 160, 4 },
		{ 1565, 1555 },
		{ 1566, 1555 },
		{ 2499, 2470 },
		{ 2499, 2470 },
		{ 2317, 2287 },
		{ 2317, 2287 },
		{ 2333, 2304 },
		{ 2333, 2304 },
		{ 1389, 1363 },
		{ 1563, 1547 },
		{ 1564, 1548 },
		{ 1395, 1368 },
		{ 1570, 1561 },
		{ 1571, 1562 },
		{ 1394, 1368 },
		{ 1572, 1563 },
		{ 1388, 1363 },
		{ 1573, 1564 },
		{ 1398, 1373 },
		{ 1577, 1571 },
		{ 1399, 1374 },
		{ 159, 4 },
		{ 1579, 1573 },
		{ 1565, 1555 },
		{ 1582, 1577 },
		{ 2499, 2470 },
		{ 1583, 1579 },
		{ 2317, 2287 },
		{ 1387, 1363 },
		{ 2333, 2304 },
		{ 1585, 1582 },
		{ 1586, 1583 },
		{ 1587, 1585 },
		{ 1588, 1586 },
		{ 1383, 1587 },
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
		{ 1400, 1375 },
		{ 1401, 1377 },
		{ 1402, 1378 },
		{ 1405, 1382 },
		{ 0, 2470 },
		{ 1567, 1555 },
		{ 1406, 1385 },
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
		{ 1407, 1386 },
		{ 81, 4 },
		{ 1408, 1387 },
		{ 1409, 1388 },
		{ 85, 4 },
		{ 1410, 1389 },
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
		{ 3015, 3014 },
		{ 1411, 1390 },
		{ 1412, 1391 },
		{ 3014, 3014 },
		{ 1415, 1393 },
		{ 1416, 1394 },
		{ 1413, 1392 },
		{ 1417, 1395 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 1414, 1392 },
		{ 3014, 3014 },
		{ 1420, 1398 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 1421, 1399 },
		{ 1422, 1400 },
		{ 1423, 1401 },
		{ 1424, 1402 },
		{ 1427, 1405 },
		{ 1428, 1406 },
		{ 1429, 1407 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 1430, 1408 },
		{ 1431, 1409 },
		{ 1432, 1410 },
		{ 1433, 1411 },
		{ 1434, 1412 },
		{ 1435, 1413 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 3014, 3014 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1436, 1414 },
		{ 1437, 1415 },
		{ 1438, 1416 },
		{ 1439, 1417 },
		{ 1442, 1420 },
		{ 154, 152 },
		{ 105, 90 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 106, 91 },
		{ 107, 92 },
		{ 108, 93 },
		{ 109, 94 },
		{ 1383, 1589 },
		{ 110, 95 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 1383, 1589 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 111, 96 },
		{ 112, 97 },
		{ 114, 99 },
		{ 120, 104 },
		{ 121, 105 },
		{ 122, 106 },
		{ 123, 107 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 124, 109 },
		{ 125, 110 },
		{ 126, 111 },
		{ 127, 112 },
		{ 2237, 2493 },
		{ 129, 114 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 2237, 2493 },
		{ 1384, 1588 },
		{ 0, 1588 },
		{ 134, 120 },
		{ 135, 121 },
		{ 136, 122 },
		{ 137, 123 },
		{ 138, 124 },
		{ 139, 125 },
		{ 140, 127 },
		{ 141, 129 },
		{ 142, 134 },
		{ 143, 135 },
		{ 144, 136 },
		{ 2244, 2222 },
		{ 2673, 2673 },
		{ 2246, 2223 },
		{ 2249, 2224 },
		{ 2250, 2225 },
		{ 2257, 2227 },
		{ 2247, 2224 },
		{ 2253, 2226 },
		{ 2260, 2228 },
		{ 2248, 2224 },
		{ 1384, 1588 },
		{ 2252, 2226 },
		{ 2261, 2229 },
		{ 2262, 2230 },
		{ 2251, 2225 },
		{ 2263, 2231 },
		{ 2264, 2232 },
		{ 2265, 2233 },
		{ 2266, 2234 },
		{ 2237, 2237 },
		{ 2258, 2238 },
		{ 2245, 2239 },
		{ 2256, 2240 },
		{ 2272, 2244 },
		{ 2673, 2673 },
		{ 145, 137 },
		{ 2254, 2226 },
		{ 2255, 2226 },
		{ 2259, 2238 },
		{ 2274, 2246 },
		{ 2275, 2247 },
		{ 2276, 2248 },
		{ 2277, 2249 },
		{ 2278, 2250 },
		{ 2279, 2251 },
		{ 2280, 2252 },
		{ 2281, 2253 },
		{ 2282, 2254 },
		{ 2283, 2255 },
		{ 0, 1588 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2284, 2256 },
		{ 2285, 2257 },
		{ 2286, 2258 },
		{ 2287, 2259 },
		{ 2288, 2260 },
		{ 2289, 2261 },
		{ 2290, 2262 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 67, 7 },
		{ 2293, 2264 },
		{ 2294, 2265 },
		{ 146, 138 },
		{ 2673, 2673 },
		{ 1589, 1588 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2673, 2673 },
		{ 2302, 2272 },
		{ 2304, 2274 },
		{ 2305, 2275 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 2306, 2276 },
		{ 2307, 2277 },
		{ 2308, 2278 },
		{ 2309, 2279 },
		{ 2310, 2280 },
		{ 2311, 2281 },
		{ 2312, 2282 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 2313, 2283 },
		{ 2314, 2284 },
		{ 2315, 2285 },
		{ 147, 140 },
		{ 1210, 7 },
		{ 2318, 2288 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 1210, 7 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 2319, 2289 },
		{ 2320, 2290 },
		{ 2321, 2291 },
		{ 2322, 2292 },
		{ 2323, 2293 },
		{ 2324, 2294 },
		{ 2331, 2302 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 148, 141 },
		{ 2334, 2305 },
		{ 2335, 2306 },
		{ 149, 142 },
		{ 0, 1447 },
		{ 2339, 2309 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1447 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 2337, 2308 },
		{ 2340, 2310 },
		{ 2342, 2312 },
		{ 2343, 2313 },
		{ 2338, 2308 },
		{ 150, 144 },
		{ 2345, 2315 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 2349, 2318 },
		{ 2350, 2319 },
		{ 2351, 2320 },
		{ 2352, 2321 },
		{ 0, 1860 },
		{ 2353, 2322 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 1860 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 2354, 2323 },
		{ 2355, 2324 },
		{ 2357, 2331 },
		{ 2360, 2334 },
		{ 2365, 2338 },
		{ 2366, 2339 },
		{ 2367, 2340 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 151, 147 },
		{ 2370, 2343 },
		{ 2372, 2345 },
		{ 2376, 2349 },
		{ 0, 2054 },
		{ 2377, 2350 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 0, 2054 },
		{ 2291, 2263 },
		{ 2378, 2351 },
		{ 2379, 2352 },
		{ 2380, 2353 },
		{ 152, 148 },
		{ 2382, 2355 },
		{ 2388, 2360 },
		{ 2391, 2365 },
		{ 2392, 2366 },
		{ 2292, 2263 },
		{ 2394, 2367 },
		{ 2397, 2370 },
		{ 2399, 2372 },
		{ 2403, 2376 },
		{ 2393, 2367 },
		{ 2404, 2377 },
		{ 2405, 2378 },
		{ 153, 150 },
		{ 2409, 2382 },
		{ 2415, 2388 },
		{ 89, 73 },
		{ 2420, 2392 },
		{ 2421, 2393 },
		{ 2422, 2394 },
		{ 2425, 2397 },
		{ 2427, 2399 },
		{ 2431, 2403 },
		{ 2432, 2404 },
		{ 2433, 2405 },
		{ 2438, 2409 },
		{ 2444, 2415 },
		{ 2447, 2420 },
		{ 2448, 2421 },
		{ 155, 153 },
		{ 2452, 2425 },
		{ 156, 155 },
		{ 2458, 2431 },
		{ 2459, 2432 },
		{ 157, 156 },
		{ 2465, 2438 },
		{ 2470, 2444 },
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
		{ 104, 89 },
		{ 2326, 2296 },
		{ 0, 2840 },
		{ 2326, 2296 },
		{ 85, 157 },
		{ 2569, 2544 },
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
		{ 2328, 2299 },
		{ 116, 101 },
		{ 2328, 2299 },
		{ 116, 101 },
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
		{ 1220, 1217 },
		{ 130, 115 },
		{ 1220, 1217 },
		{ 130, 115 },
		{ 2839, 49 },
		{ 2580, 2555 },
		{ 91, 74 },
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
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1223, 1219 },
		{ 2297, 2268 },
		{ 1223, 1219 },
		{ 2297, 2268 },
		{ 1210, 1210 },
		{ 1778, 1777 },
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
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 1210, 1210 },
		{ 0, 2465 },
		{ 0, 2465 },
		{ 1225, 1222 },
		{ 132, 118 },
		{ 1225, 1222 },
		{ 132, 118 },
		{ 1262, 1261 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 1736, 1735 },
		{ 2709, 2694 },
		{ 2726, 2712 },
		{ 2160, 2157 },
		{ 1259, 1258 },
		{ 2545, 2516 },
		{ 0, 2465 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 1643, 1642 },
		{ 86, 49 },
		{ 1688, 1687 },
		{ 2986, 2985 },
		{ 3006, 3006 },
		{ 1733, 1732 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 3006, 3006 },
		{ 1322, 1321 },
		{ 2012, 1988 },
		{ 182, 173 },
		{ 192, 173 },
		{ 184, 173 },
		{ 1617, 1616 },
		{ 179, 173 },
		{ 1322, 1321 },
		{ 183, 173 },
		{ 3065, 3064 },
		{ 181, 173 },
		{ 2831, 2830 },
		{ 1617, 1616 },
		{ 1691, 1690 },
		{ 190, 173 },
		{ 189, 173 },
		{ 180, 173 },
		{ 188, 173 },
		{ 2493, 2465 },
		{ 185, 173 },
		{ 187, 173 },
		{ 178, 173 },
		{ 2011, 1988 },
		{ 2546, 2517 },
		{ 186, 173 },
		{ 191, 173 },
		{ 100, 83 },
		{ 1664, 1663 },
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
		{ 3030, 3030 },
		{ 3030, 3030 },
		{ 3030, 3030 },
		{ 3030, 3030 },
		{ 3030, 3030 },
		{ 3030, 3030 },
		{ 3030, 3030 },
		{ 3030, 3030 },
		{ 3030, 3030 },
		{ 3030, 3030 },
		{ 1216, 1213 },
		{ 101, 83 },
		{ 1213, 1213 },
		{ 1213, 1213 },
		{ 1213, 1213 },
		{ 1213, 1213 },
		{ 1213, 1213 },
		{ 1213, 1213 },
		{ 1213, 1213 },
		{ 1213, 1213 },
		{ 1213, 1213 },
		{ 1213, 1213 },
		{ 1811, 1787 },
		{ 449, 405 },
		{ 454, 405 },
		{ 451, 405 },
		{ 450, 405 },
		{ 453, 405 },
		{ 448, 405 },
		{ 3036, 65 },
		{ 447, 405 },
		{ 1997, 1975 },
		{ 67, 65 },
		{ 1217, 1213 },
		{ 452, 405 },
		{ 1810, 1787 },
		{ 455, 405 },
		{ 1275, 1274 },
		{ 2872, 2871 },
		{ 2492, 2464 },
		{ 2027, 2006 },
		{ 2925, 2924 },
		{ 446, 405 },
		{ 101, 83 },
		{ 2267, 2235 },
		{ 3031, 3030 },
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
		{ 2613, 2588 },
		{ 2056, 2038 },
		{ 2966, 2965 },
		{ 2070, 2053 },
		{ 2989, 2988 },
		{ 2997, 2996 },
		{ 1836, 1817 },
		{ 1858, 1839 },
		{ 1749, 1748 },
		{ 1217, 1213 },
		{ 2200, 2199 },
		{ 2268, 2235 },
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
		{ 2205, 2204 },
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
		{ 118, 102 },
		{ 2478, 2478 },
		{ 2478, 2478 },
		{ 2406, 2406 },
		{ 2406, 2406 },
		{ 2786, 2785 },
		{ 2450, 2423 },
		{ 3048, 3044 },
		{ 2826, 2825 },
		{ 3034, 65 },
		{ 2268, 2235 },
		{ 115, 100 },
		{ 1216, 1216 },
		{ 1216, 1216 },
		{ 1216, 1216 },
		{ 1216, 1216 },
		{ 1216, 1216 },
		{ 1216, 1216 },
		{ 1216, 1216 },
		{ 1216, 1216 },
		{ 1216, 1216 },
		{ 1216, 1216 },
		{ 3032, 65 },
		{ 3068, 3067 },
		{ 2478, 2478 },
		{ 3084, 3081 },
		{ 2406, 2406 },
		{ 3090, 3087 },
		{ 2583, 2558 },
		{ 2101, 2086 },
		{ 1351, 1350 },
		{ 3022, 63 },
		{ 118, 102 },
		{ 1219, 1216 },
		{ 67, 63 },
		{ 2161, 2158 },
		{ 2620, 2595 },
		{ 2176, 2175 },
		{ 2631, 2607 },
		{ 2635, 2611 },
		{ 2644, 2621 },
		{ 2390, 2362 },
		{ 2650, 2627 },
		{ 115, 100 },
		{ 3035, 65 },
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
		{ 2267, 2267 },
		{ 2267, 2267 },
		{ 2267, 2267 },
		{ 2267, 2267 },
		{ 2267, 2267 },
		{ 2267, 2267 },
		{ 2267, 2267 },
		{ 2267, 2267 },
		{ 2267, 2267 },
		{ 2267, 2267 },
		{ 1219, 1216 },
		{ 1222, 1218 },
		{ 2662, 2643 },
		{ 2316, 2316 },
		{ 2316, 2316 },
		{ 1212, 9 },
		{ 2664, 2645 },
		{ 2455, 2455 },
		{ 2455, 2455 },
		{ 67, 9 },
		{ 2678, 2659 },
		{ 2296, 2267 },
		{ 2679, 2660 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 2269, 2269 },
		{ 1242, 1241 },
		{ 1665, 1664 },
		{ 3021, 63 },
		{ 2316, 2316 },
		{ 2689, 2670 },
		{ 1212, 9 },
		{ 3020, 63 },
		{ 2455, 2455 },
		{ 2890, 2890 },
		{ 2890, 2890 },
		{ 1222, 1218 },
		{ 2299, 2269 },
		{ 2937, 2937 },
		{ 2937, 2937 },
		{ 2202, 2201 },
		{ 2508, 2478 },
		{ 2507, 2478 },
		{ 2435, 2406 },
		{ 2434, 2406 },
		{ 1214, 9 },
		{ 2296, 2267 },
		{ 1213, 9 },
		{ 1213, 9 },
		{ 1213, 9 },
		{ 1213, 9 },
		{ 1213, 9 },
		{ 1213, 9 },
		{ 1213, 9 },
		{ 1213, 9 },
		{ 1213, 9 },
		{ 1213, 9 },
		{ 2890, 2890 },
		{ 2457, 2457 },
		{ 2457, 2457 },
		{ 2694, 2677 },
		{ 2937, 2937 },
		{ 2466, 2466 },
		{ 2466, 2466 },
		{ 2336, 2336 },
		{ 2336, 2336 },
		{ 2485, 2485 },
		{ 2485, 2485 },
		{ 2703, 2687 },
		{ 2299, 2269 },
		{ 2361, 2361 },
		{ 2361, 2361 },
		{ 2499, 2499 },
		{ 2499, 2499 },
		{ 2518, 2518 },
		{ 2518, 2518 },
		{ 2570, 2570 },
		{ 2570, 2570 },
		{ 2663, 2663 },
		{ 2663, 2663 },
		{ 1667, 1666 },
		{ 2457, 2457 },
		{ 3018, 63 },
		{ 2712, 2697 },
		{ 3019, 63 },
		{ 2466, 2466 },
		{ 1839, 1820 },
		{ 2336, 2336 },
		{ 2729, 2715 },
		{ 2485, 2485 },
		{ 2742, 2736 },
		{ 2381, 2381 },
		{ 2381, 2381 },
		{ 2361, 2361 },
		{ 2744, 2738 },
		{ 2499, 2499 },
		{ 2750, 2745 },
		{ 2518, 2518 },
		{ 2756, 2755 },
		{ 2570, 2570 },
		{ 2214, 2213 },
		{ 2663, 2663 },
		{ 2346, 2316 },
		{ 2822, 2822 },
		{ 2822, 2822 },
		{ 2850, 2850 },
		{ 2850, 2850 },
		{ 2469, 2443 },
		{ 2460, 2460 },
		{ 2460, 2460 },
		{ 2341, 2341 },
		{ 2341, 2341 },
		{ 2789, 2788 },
		{ 2347, 2316 },
		{ 2381, 2381 },
		{ 2426, 2426 },
		{ 2426, 2426 },
		{ 2481, 2455 },
		{ 2798, 2797 },
		{ 2484, 2484 },
		{ 2484, 2484 },
		{ 2454, 2454 },
		{ 2454, 2454 },
		{ 2207, 2207 },
		{ 2207, 2207 },
		{ 2811, 2810 },
		{ 2822, 2822 },
		{ 1845, 1826 },
		{ 2850, 2850 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2460, 2460 },
		{ 2471, 2445 },
		{ 2341, 2341 },
		{ 2904, 2904 },
		{ 2904, 2904 },
		{ 2945, 2945 },
		{ 2945, 2945 },
		{ 2426, 2426 },
		{ 2384, 2384 },
		{ 2384, 2384 },
		{ 2891, 2890 },
		{ 2484, 2484 },
		{ 1278, 1277 },
		{ 2454, 2454 },
		{ 2938, 2937 },
		{ 2207, 2207 },
		{ 2834, 2833 },
		{ 2576, 2576 },
		{ 2576, 2576 },
		{ 1865, 1846 },
		{ 1317, 1316 },
		{ 2273, 2273 },
		{ 1892, 1876 },
		{ 2623, 2623 },
		{ 2623, 2623 },
		{ 2494, 2466 },
		{ 2904, 2904 },
		{ 1894, 1879 },
		{ 2945, 2945 },
		{ 2474, 2474 },
		{ 2474, 2474 },
		{ 2384, 2384 },
		{ 1896, 1881 },
		{ 2495, 2466 },
		{ 2483, 2457 },
		{ 2666, 2666 },
		{ 2666, 2666 },
		{ 2344, 2344 },
		{ 2344, 2344 },
		{ 2362, 2336 },
		{ 2576, 2576 },
		{ 2515, 2485 },
		{ 2851, 2850 },
		{ 2859, 2859 },
		{ 2859, 2859 },
		{ 2389, 2361 },
		{ 2623, 2623 },
		{ 2529, 2499 },
		{ 2862, 2861 },
		{ 2547, 2518 },
		{ 2488, 2460 },
		{ 2595, 2570 },
		{ 2474, 2474 },
		{ 2682, 2663 },
		{ 1946, 1944 },
		{ 2600, 2600 },
		{ 2600, 2600 },
		{ 2889, 2888 },
		{ 2666, 2666 },
		{ 2416, 2389 },
		{ 2344, 2344 },
		{ 1693, 1692 },
		{ 2919, 2918 },
		{ 1715, 1714 },
		{ 2928, 2927 },
		{ 2408, 2381 },
		{ 2859, 2859 },
		{ 2486, 2460 },
		{ 2936, 2935 },
		{ 1728, 1727 },
		{ 1263, 1262 },
		{ 2487, 2460 },
		{ 2960, 2959 },
		{ 1612, 1611 },
		{ 2851, 2850 },
		{ 2969, 2968 },
		{ 2980, 2979 },
		{ 2823, 2822 },
		{ 2600, 2600 },
		{ 1737, 1736 },
		{ 2428, 2400 },
		{ 2991, 2990 },
		{ 2368, 2341 },
		{ 2430, 2402 },
		{ 3000, 2999 },
		{ 2068, 2051 },
		{ 2069, 2052 },
		{ 2453, 2426 },
		{ 1340, 1339 },
		{ 2534, 2505 },
		{ 1752, 1751 },
		{ 2514, 2484 },
		{ 2087, 2074 },
		{ 2480, 2454 },
		{ 2442, 2413 },
		{ 2208, 2207 },
		{ 3044, 3040 },
		{ 2553, 2526 },
		{ 2566, 2541 },
		{ 2100, 2085 },
		{ 2303, 2273 },
		{ 3070, 3069 },
		{ 2573, 2548 },
		{ 3072, 3072 },
		{ 2446, 2417 },
		{ 2905, 2904 },
		{ 3097, 3095 },
		{ 2946, 2945 },
		{ 1818, 1793 },
		{ 2719, 2704 },
		{ 2411, 2384 },
		{ 1817, 1793 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 2007, 1982 },
		{ 1926, 1916 },
		{ 1934, 1926 },
		{ 2006, 1982 },
		{ 1311, 1310 },
		{ 2601, 2576 },
		{ 1271, 1270 },
		{ 1709, 1708 },
		{ 2500, 2471 },
		{ 2760, 2759 },
		{ 1710, 1709 },
		{ 2646, 2623 },
		{ 3072, 3072 },
		{ 2781, 2780 },
		{ 1246, 1245 },
		{ 1719, 1718 },
		{ 2793, 2792 },
		{ 2503, 2474 },
		{ 1575, 1569 },
		{ 2516, 2486 },
		{ 2520, 2490 },
		{ 2333, 2333 },
		{ 2028, 2007 },
		{ 2685, 2666 },
		{ 2528, 2497 },
		{ 2371, 2344 },
		{ 2047, 2026 },
		{ 2049, 2028 },
		{ 2052, 2031 },
		{ 1231, 1230 },
		{ 2845, 2843 },
		{ 2860, 2859 },
		{ 1605, 1604 },
		{ 1606, 1605 },
		{ 2554, 2528 },
		{ 2866, 2865 },
		{ 1745, 1744 },
		{ 1346, 1345 },
		{ 1286, 1285 },
		{ 1768, 1767 },
		{ 1769, 1768 },
		{ 2625, 2600 },
		{ 2587, 2562 },
		{ 1774, 1773 },
		{ 2598, 2573 },
		{ 1633, 1632 },
		{ 1634, 1633 },
		{ 1640, 1639 },
		{ 1641, 1640 },
		{ 1293, 1292 },
		{ 2979, 2978 },
		{ 1837, 1818 },
		{ 2463, 2436 },
		{ 1659, 1658 },
		{ 2467, 2441 },
		{ 2651, 2628 },
		{ 2652, 2629 },
		{ 2654, 2632 },
		{ 2655, 2635 },
		{ 1844, 1825 },
		{ 2217, 2216 },
		{ 1660, 1659 },
		{ 2472, 2446 },
		{ 2681, 2662 },
		{ 1856, 1837 },
		{ 1294, 1293 },
		{ 3060, 3059 },
		{ 3061, 3060 },
		{ 1296, 1295 },
		{ 2690, 2671 },
		{ 1569, 1560 },
		{ 1683, 1682 },
		{ 1684, 1683 },
		{ 1310, 1309 },
		{ 1907, 1894 },
		{ 1255, 1254 },
		{ 2783, 2782 },
		{ 3075, 3072 },
		{ 2582, 2557 },
		{ 1348, 1347 },
		{ 1695, 1694 },
		{ 2154, 2147 },
		{ 3074, 3072 },
		{ 2809, 2808 },
		{ 3073, 3072 },
		{ 2597, 2572 },
		{ 2820, 2819 },
		{ 2479, 2453 },
		{ 2424, 2396 },
		{ 2602, 2577 },
		{ 2374, 2347 },
		{ 2157, 2152 },
		{ 1288, 1287 },
		{ 1645, 1644 },
		{ 2632, 2608 },
		{ 2849, 2847 },
		{ 2173, 2170 },
		{ 2359, 2333 },
		{ 2636, 2612 },
		{ 2489, 2461 },
		{ 1941, 1936 },
		{ 1233, 1232 },
		{ 1261, 1260 },
		{ 1829, 1810 },
		{ 2358, 2358 },
		{ 2358, 2358 },
		{ 2917, 2916 },
		{ 2204, 2203 },
		{ 1721, 1720 },
		{ 2510, 2480 },
		{ 2668, 2649 },
		{ 2672, 2653 },
		{ 2511, 2481 },
		{ 2958, 2957 },
		{ 2513, 2483 },
		{ 1324, 1323 },
		{ 2210, 2209 },
		{ 1730, 1729 },
		{ 2216, 2215 },
		{ 2018, 1997 },
		{ 1840, 1821 },
		{ 2693, 2676 },
		{ 1338, 1337 },
		{ 2700, 2684 },
		{ 2039, 2018 },
		{ 2543, 2514 },
		{ 1735, 1734 },
		{ 2358, 2358 },
		{ 2714, 2699 },
		{ 1848, 1829 },
		{ 1619, 1618 },
		{ 1669, 1668 },
		{ 2730, 2716 },
		{ 2731, 2718 },
		{ 3051, 3048 },
		{ 1240, 1239 },
		{ 2743, 2737 },
		{ 2561, 2536 },
		{ 1875, 1858 },
		{ 2751, 2749 },
		{ 3072, 3071 },
		{ 2754, 2752 },
		{ 3080, 3077 },
		{ 1342, 1341 },
		{ 3088, 3085 },
		{ 1638, 1637 },
		{ 2574, 2549 },
		{ 3100, 3099 },
		{ 2317, 2317 },
		{ 2317, 2317 },
		{ 2740, 2740 },
		{ 2740, 2740 },
		{ 2407, 2407 },
		{ 2407, 2407 },
		{ 2626, 2601 },
		{ 1879, 1864 },
		{ 2423, 2395 },
		{ 1336, 1335 },
		{ 2736, 2728 },
		{ 2549, 2520 },
		{ 2526, 2495 },
		{ 2645, 2622 },
		{ 2038, 2017 },
		{ 1714, 1713 },
		{ 2532, 2503 },
		{ 1775, 1774 },
		{ 2385, 2358 },
		{ 2612, 2587 },
		{ 2571, 2546 },
		{ 2701, 2685 },
		{ 2861, 2860 },
		{ 2317, 2317 },
		{ 2659, 2639 },
		{ 2740, 2740 },
		{ 2660, 2641 },
		{ 2407, 2407 },
		{ 2541, 2512 },
		{ 1350, 1349 },
		{ 2665, 2646 },
		{ 1807, 1784 },
		{ 2819, 2818 },
		{ 2170, 2168 },
		{ 1653, 1652 },
		{ 2653, 2631 },
		{ 1943, 1940 },
		{ 1341, 1340 },
		{ 2836, 2835 },
		{ 1949, 1948 },
		{ 1822, 1799 },
		{ 1806, 1784 },
		{ 2531, 2502 },
		{ 2451, 2424 },
		{ 1460, 1440 },
		{ 2539, 2510 },
		{ 1280, 1279 },
		{ 1403, 1379 },
		{ 1627, 1626 },
		{ 2386, 2358 },
		{ 2864, 2863 },
		{ 2013, 1990 },
		{ 2871, 2870 },
		{ 2017, 1996 },
		{ 2395, 2368 },
		{ 2688, 2669 },
		{ 1668, 1667 },
		{ 1841, 1822 },
		{ 1843, 1824 },
		{ 2921, 2920 },
		{ 2032, 2011 },
		{ 2568, 2543 },
		{ 2930, 2929 },
		{ 2034, 2013 },
		{ 2036, 2015 },
		{ 1335, 1334 },
		{ 1677, 1676 },
		{ 1244, 1243 },
		{ 2962, 2961 },
		{ 1425, 1403 },
		{ 2718, 2703 },
		{ 2971, 2970 },
		{ 1639, 1638 },
		{ 2720, 2705 },
		{ 2584, 2559 },
		{ 1254, 1253 },
		{ 2067, 2050 },
		{ 2993, 2992 },
		{ 1874, 1857 },
		{ 1754, 1753 },
		{ 3002, 3001 },
		{ 2737, 2729 },
		{ 1762, 1761 },
		{ 1304, 1303 },
		{ 2608, 2583 },
		{ 2082, 2066 },
		{ 2749, 2744 },
		{ 1889, 1873 },
		{ 1694, 1693 },
		{ 2621, 2596 },
		{ 3049, 3045 },
		{ 1642, 1641 },
		{ 2758, 2757 },
		{ 2132, 2114 },
		{ 2133, 2115 },
		{ 1703, 1702 },
		{ 2348, 2317 },
		{ 2509, 2479 },
		{ 2745, 2740 },
		{ 3071, 3070 },
		{ 2436, 2407 },
		{ 1599, 1598 },
		{ 2159, 2156 },
		{ 3077, 3074 },
		{ 2791, 2790 },
		{ 1353, 1352 },
		{ 1713, 1712 },
		{ 2648, 2625 },
		{ 2167, 2164 },
		{ 3099, 3097 },
		{ 2813, 2812 },
		{ 2912, 2912 },
		{ 2912, 2912 },
		{ 2804, 2804 },
		{ 2804, 2804 },
		{ 2953, 2953 },
		{ 2953, 2953 },
		{ 2449, 2449 },
		{ 2449, 2449 },
		{ 2473, 2473 },
		{ 2473, 2473 },
		{ 1245, 1244 },
		{ 2156, 2151 },
		{ 2846, 2844 },
		{ 1838, 1819 },
		{ 2548, 2519 },
		{ 1950, 1949 },
		{ 2550, 2521 },
		{ 2552, 2525 },
		{ 1740, 1739 },
		{ 2865, 2864 },
		{ 1281, 1280 },
		{ 2468, 2442 },
		{ 1303, 1302 },
		{ 2912, 2912 },
		{ 2873, 2872 },
		{ 2804, 2804 },
		{ 2882, 2881 },
		{ 2953, 2953 },
		{ 2169, 2167 },
		{ 2449, 2449 },
		{ 1750, 1749 },
		{ 2473, 2473 },
		{ 2899, 2898 },
		{ 2900, 2899 },
		{ 2902, 2901 },
		{ 1652, 1651 },
		{ 2704, 2688 },
		{ 2915, 2914 },
		{ 1702, 1701 },
		{ 2015, 1993 },
		{ 2016, 1995 },
		{ 2922, 2921 },
		{ 1846, 1827 },
		{ 2926, 2925 },
		{ 1755, 1754 },
		{ 1761, 1760 },
		{ 2931, 2930 },
		{ 2206, 2205 },
		{ 1610, 1609 },
		{ 2943, 2942 },
		{ 2029, 2008 },
		{ 1333, 1332 },
		{ 2956, 2955 },
		{ 2733, 2720 },
		{ 1480, 1461 },
		{ 113, 98 },
		{ 2963, 2962 },
		{ 2300, 2270 },
		{ 2967, 2966 },
		{ 2609, 2584 },
		{ 2611, 2586 },
		{ 2972, 2971 },
		{ 2978, 2977 },
		{ 1876, 1859 },
		{ 1626, 1625 },
		{ 1276, 1275 },
		{ 1881, 1866 },
		{ 2624, 2599 },
		{ 2051, 2030 },
		{ 2994, 2993 },
		{ 1266, 1265 },
		{ 2998, 2997 },
		{ 2759, 2758 },
		{ 1891, 1875 },
		{ 3003, 3002 },
		{ 1404, 1381 },
		{ 1676, 1675 },
		{ 1354, 1353 },
		{ 3016, 3013 },
		{ 1315, 1314 },
		{ 2787, 2786 },
		{ 1915, 1903 },
		{ 2074, 2057 },
		{ 3042, 3038 },
		{ 2792, 2791 },
		{ 3045, 3041 },
		{ 1824, 1802 },
		{ 2517, 2487 },
		{ 1928, 1919 },
		{ 3054, 3051 },
		{ 2807, 2806 },
		{ 2373, 2346 },
		{ 1828, 1809 },
		{ 2375, 2348 },
		{ 2913, 2912 },
		{ 2814, 2813 },
		{ 2805, 2804 },
		{ 1940, 1935 },
		{ 2954, 2953 },
		{ 2112, 2098 },
		{ 2475, 2449 },
		{ 1440, 1418 },
		{ 2502, 2473 },
		{ 2533, 2504 },
		{ 1598, 1597 },
		{ 2832, 2831 },
		{ 2134, 2116 },
		{ 2145, 2132 },
		{ 2837, 2836 },
		{ 2146, 2133 },
		{ 2675, 2656 },
		{ 1273, 1273 },
		{ 1273, 1273 },
		{ 1747, 1747 },
		{ 1747, 1747 },
		{ 2829, 2829 },
		{ 2829, 2829 },
		{ 2964, 2964 },
		{ 2964, 2964 },
		{ 2698, 2698 },
		{ 2698, 2698 },
		{ 2295, 2295 },
		{ 2295, 2295 },
		{ 2923, 2923 },
		{ 2923, 2923 },
		{ 2995, 2995 },
		{ 2995, 2995 },
		{ 2506, 2506 },
		{ 2506, 2506 },
		{ 2784, 2784 },
		{ 2784, 2784 },
		{ 2369, 2369 },
		{ 2369, 2369 },
		{ 1862, 1843 },
		{ 1273, 1273 },
		{ 1305, 1304 },
		{ 1747, 1747 },
		{ 2414, 2387 },
		{ 2829, 2829 },
		{ 1678, 1677 },
		{ 2964, 2964 },
		{ 3052, 3049 },
		{ 2698, 2698 },
		{ 1717, 1716 },
		{ 2295, 2295 },
		{ 2212, 2211 },
		{ 2923, 2923 },
		{ 1763, 1762 },
		{ 2995, 2995 },
		{ 2083, 2067 },
		{ 2506, 2506 },
		{ 2477, 2451 },
		{ 2784, 2784 },
		{ 2162, 2159 },
		{ 2369, 2369 },
		{ 1479, 1460 },
		{ 1654, 1653 },
		{ 1704, 1703 },
		{ 2603, 2578 },
		{ 1628, 1627 },
		{ 1890, 1874 },
		{ 3087, 3084 },
		{ 2178, 2177 },
		{ 2055, 2036 },
		{ 1945, 1943 },
		{ 1600, 1599 },
		{ 1663, 1662 },
		{ 2984, 2984 },
		{ 2984, 2984 },
		{ 2948, 2948 },
		{ 2948, 2948 },
		{ 1742, 1742 },
		{ 1742, 1742 },
		{ 2907, 2907 },
		{ 2907, 2907 },
		{ 2799, 2799 },
		{ 2799, 2799 },
		{ 1731, 1731 },
		{ 1731, 1731 },
		{ 2941, 2941 },
		{ 2941, 2941 },
		{ 2201, 2200 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 1268, 1268 },
		{ 1268, 1268 },
		{ 1906, 1893 },
		{ 1615, 1614 },
		{ 2026, 2005 },
		{ 1291, 1290 },
		{ 2984, 2984 },
		{ 1918, 1908 },
		{ 2948, 2948 },
		{ 1849, 1830 },
		{ 1742, 1742 },
		{ 2131, 2113 },
		{ 2907, 2907 },
		{ 1690, 1689 },
		{ 2799, 2799 },
		{ 1308, 1307 },
		{ 1731, 1731 },
		{ 1860, 1841 },
		{ 2941, 2941 },
		{ 1544, 1518 },
		{ 1274, 1273 },
		{ 1257, 1257 },
		{ 1748, 1747 },
		{ 1268, 1268 },
		{ 2830, 2829 },
		{ 2581, 2556 },
		{ 2965, 2964 },
		{ 1603, 1602 },
		{ 2713, 2698 },
		{ 2735, 2727 },
		{ 2325, 2295 },
		{ 2658, 2638 },
		{ 2924, 2923 },
		{ 2383, 2356 },
		{ 2996, 2995 },
		{ 1825, 1805 },
		{ 2535, 2506 },
		{ 2585, 2560 },
		{ 2785, 2784 },
		{ 1320, 1319 },
		{ 2396, 2369 },
		{ 1631, 1630 },
		{ 2596, 2571 },
		{ 1270, 1269 },
		{ 2476, 2450 },
		{ 3058, 3057 },
		{ 2053, 2032 },
		{ 2857, 2856 },
		{ 2054, 2034 },
		{ 3067, 3066 },
		{ 2680, 2661 },
		{ 1447, 1425 },
		{ 2166, 2163 },
		{ 1766, 1765 },
		{ 1707, 1706 },
		{ 1657, 1656 },
		{ 1681, 1680 },
		{ 1238, 1237 },
		{ 1776, 1775 },
		{ 2491, 2463 },
		{ 2073, 2056 },
		{ 3093, 3090 },
		{ 2988, 2987 },
		{ 1744, 1743 },
		{ 2702, 2686 },
		{ 2592, 2592 },
		{ 2592, 2592 },
		{ 2853, 2852 },
		{ 1234, 1234 },
		{ 1234, 1234 },
		{ 2456, 2456 },
		{ 2456, 2456 },
		{ 2590, 2590 },
		{ 2590, 2590 },
		{ 2880, 2879 },
		{ 2985, 2984 },
		{ 1379, 1370 },
		{ 2949, 2948 },
		{ 1799, 1955 },
		{ 1743, 1742 },
		{ 1990, 2181 },
		{ 2908, 2907 },
		{ 1609, 1608 },
		{ 2800, 2799 },
		{ 1739, 1738 },
		{ 1732, 1731 },
		{ 0, 1214 },
		{ 2942, 2941 },
		{ 2592, 2592 },
		{ 0, 84 },
		{ 1258, 1257 },
		{ 1234, 1234 },
		{ 1269, 1268 },
		{ 2456, 2456 },
		{ 0, 2236 },
		{ 2590, 2590 },
		{ 2298, 2298 },
		{ 2298, 2298 },
		{ 2298, 2298 },
		{ 2298, 2298 },
		{ 2298, 2298 },
		{ 2298, 2298 },
		{ 2298, 2298 },
		{ 2298, 2298 },
		{ 2298, 2298 },
		{ 2298, 2298 },
		{ 1289, 1289 },
		{ 1289, 1289 },
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
		{ 0, 1214 },
		{ 2875, 2875 },
		{ 2875, 2875 },
		{ 0, 84 },
		{ 1265, 1264 },
		{ 2812, 2811 },
		{ 1611, 1610 },
		{ 2599, 2574 },
		{ 0, 2236 },
		{ 1461, 1441 },
		{ 2961, 2960 },
		{ 1289, 1289 },
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
		{ 1808, 1786 },
		{ 1613, 1612 },
		{ 2875, 2875 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 2329, 2329 },
		{ 1318, 1317 },
		{ 2697, 2680 },
		{ 2413, 2386 },
		{ 2617, 2592 },
		{ 1948, 1946 },
		{ 2970, 2969 },
		{ 1235, 1234 },
		{ 1616, 1615 },
		{ 2482, 2456 },
		{ 2164, 2161 },
		{ 2615, 2590 },
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
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 1221, 1221 },
		{ 1221, 1221 },
		{ 1221, 1221 },
		{ 1221, 1221 },
		{ 1221, 1221 },
		{ 1221, 1221 },
		{ 1221, 1221 },
		{ 1221, 1221 },
		{ 1221, 1221 },
		{ 1221, 1221 },
		{ 2579, 2579 },
		{ 2579, 2579 },
		{ 1290, 1289 },
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
		{ 1809, 1786 },
		{ 2835, 2834 },
		{ 2876, 2875 },
		{ 1716, 1715 },
		{ 2705, 2689 },
		{ 1352, 1351 },
		{ 1826, 1806 },
		{ 1753, 1752 },
		{ 3092, 3092 },
		{ 2544, 2515 },
		{ 2579, 2579 },
		{ 1224, 1224 },
		{ 1224, 1224 },
		{ 1224, 1224 },
		{ 1224, 1224 },
		{ 1224, 1224 },
		{ 1224, 1224 },
		{ 1224, 1224 },
		{ 1224, 1224 },
		{ 1224, 1224 },
		{ 1224, 1224 },
		{ 1226, 1226 },
		{ 1226, 1226 },
		{ 1226, 1226 },
		{ 1226, 1226 },
		{ 1226, 1226 },
		{ 1226, 1226 },
		{ 1226, 1226 },
		{ 1226, 1226 },
		{ 1226, 1226 },
		{ 1226, 1226 },
		{ 3092, 3092 },
		{ 2894, 2894 },
		{ 2894, 2894 },
		{ 2815, 2815 },
		{ 2815, 2815 },
		{ 2563, 2563 },
		{ 2563, 2563 },
		{ 2622, 2597 },
		{ 2992, 2991 },
		{ 1994, 1972 },
		{ 2175, 2173 },
		{ 1880, 1865 },
		{ 2856, 2855 },
		{ 2727, 2713 },
		{ 2728, 2714 },
		{ 3001, 3000 },
		{ 2628, 2603 },
		{ 1718, 1717 },
		{ 2863, 2862 },
		{ 1830, 1811 },
		{ 1279, 1278 },
		{ 1321, 1320 },
		{ 2638, 2614 },
		{ 1314, 1313 },
		{ 2894, 2894 },
		{ 2084, 2068 },
		{ 2815, 2815 },
		{ 2555, 2529 },
		{ 2563, 2563 },
		{ 2558, 2533 },
		{ 2881, 2880 },
		{ 2559, 2534 },
		{ 2560, 2535 },
		{ 1380, 1359 },
		{ 2387, 2359 },
		{ 2898, 2897 },
		{ 2098, 2082 },
		{ 1243, 1242 },
		{ 2901, 2900 },
		{ 2757, 2756 },
		{ 2656, 2636 },
		{ 2443, 2414 },
		{ 1903, 1889 },
		{ 2356, 2325 },
		{ 2661, 2642 },
		{ 2578, 2553 },
		{ 2604, 2579 },
		{ 2920, 2919 },
		{ 1332, 1331 },
		{ 2114, 2100 },
		{ 2115, 2101 },
		{ 3080, 3080 },
		{ 2669, 2650 },
		{ 2790, 2789 },
		{ 1316, 1315 },
		{ 2929, 2928 },
		{ 1773, 1772 },
		{ 2677, 2658 },
		{ 1349, 1348 },
		{ 2400, 2373 },
		{ 2402, 2375 },
		{ 2803, 2802 },
		{ 2939, 2938 },
		{ 2877, 2876 },
		{ 2591, 2566 },
		{ 2177, 2176 },
		{ 67, 5 },
		{ 2952, 2951 },
		{ 1738, 1737 },
		{ 3092, 3089 },
		{ 3094, 3092 },
		{ 2911, 2910 },
		{ 2695, 2678 },
		{ 2892, 2891 },
		{ 3080, 3080 },
		{ 2696, 2679 },
		{ 1995, 1972 },
		{ 1264, 1263 },
		{ 2883, 2882 },
		{ 1331, 1330 },
		{ 2903, 2902 },
		{ 1741, 1740 },
		{ 2168, 2166 },
		{ 2538, 2509 },
		{ 2897, 2896 },
		{ 2647, 2624 },
		{ 2692, 2675 },
		{ 1267, 1266 },
		{ 2843, 2841 },
		{ 3037, 3032 },
		{ 2895, 2894 },
		{ 1819, 1796 },
		{ 2816, 2815 },
		{ 1253, 1252 },
		{ 2588, 2563 },
		{ 1820, 1796 },
		{ 1727, 1726 },
		{ 2490, 2462 },
		{ 2462, 2435 },
		{ 2537, 2508 },
		{ 1381, 1359 },
		{ 2906, 2905 },
		{ 2183, 2180 },
		{ 1957, 1954 },
		{ 2844, 2841 },
		{ 2947, 2946 },
		{ 2657, 2637 },
		{ 2182, 2180 },
		{ 1956, 1954 },
		{ 2198, 2197 },
		{ 2818, 2817 },
		{ 2332, 2303 },
		{ 2637, 2613 },
		{ 2855, 2854 },
		{ 2398, 2371 },
		{ 2562, 2537 },
		{ 1801, 1781 },
		{ 2440, 2411 },
		{ 174, 5 },
		{ 3038, 3032 },
		{ 1441, 1419 },
		{ 1800, 1781 },
		{ 175, 5 },
		{ 1992, 1969 },
		{ 1397, 1371 },
		{ 2896, 2895 },
		{ 2716, 2701 },
		{ 1601, 1600 },
		{ 1991, 1969 },
		{ 176, 5 },
		{ 1306, 1305 },
		{ 2594, 2569 },
		{ 2401, 2374 },
		{ 1323, 1322 },
		{ 1947, 1945 },
		{ 1236, 1235 },
		{ 2116, 2102 },
		{ 1746, 1745 },
		{ 1644, 1643 },
		{ 2270, 2241 },
		{ 2914, 2913 },
		{ 1701, 1700 },
		{ 2916, 2915 },
		{ 3083, 3080 },
		{ 2605, 2580 },
		{ 173, 5 },
		{ 2741, 2735 },
		{ 2497, 2468 },
		{ 2410, 2383 },
		{ 2610, 2585 },
		{ 1330, 1329 },
		{ 1651, 1650 },
		{ 98, 81 },
		{ 2504, 2475 },
		{ 2752, 2750 },
		{ 2147, 2134 },
		{ 2151, 2144 },
		{ 1705, 1704 },
		{ 1993, 1970 },
		{ 2512, 2482 },
		{ 2940, 2939 },
		{ 1256, 1255 },
		{ 1760, 1759 },
		{ 1855, 1836 },
		{ 2782, 2781 },
		{ 1252, 1251 },
		{ 1655, 1654 },
		{ 2519, 2489 },
		{ 2955, 2954 },
		{ 1859, 1840 },
		{ 2957, 2956 },
		{ 2165, 2162 },
		{ 2525, 2494 },
		{ 2639, 2615 },
		{ 2641, 2617 },
		{ 2429, 2401 },
		{ 1500, 1479 },
		{ 1764, 1763 },
		{ 1863, 1844 },
		{ 1501, 1480 },
		{ 2806, 2805 },
		{ 2649, 2626 },
		{ 2808, 2807 },
		{ 1866, 1848 },
		{ 1295, 1294 },
		{ 2437, 2408 },
		{ 2981, 2980 },
		{ 1272, 1271 },
		{ 2030, 2009 },
		{ 2031, 2010 },
		{ 2817, 2816 },
		{ 2540, 2511 },
		{ 1618, 1617 },
		{ 2542, 2513 },
		{ 2821, 2820 },
		{ 1337, 1336 },
		{ 2824, 2823 },
		{ 1878, 1862 },
		{ 2828, 2827 },
		{ 1625, 1624 },
		{ 2197, 2196 },
		{ 1720, 1719 },
		{ 1302, 1301 },
		{ 3013, 3005 },
		{ 2048, 2027 },
		{ 1232, 1231 },
		{ 2671, 2652 },
		{ 2551, 2522 },
		{ 1675, 1674 },
		{ 2676, 2657 },
		{ 1802, 1782 },
		{ 1803, 1783 },
		{ 1804, 1783 },
		{ 1729, 1728 },
		{ 3040, 3035 },
		{ 3041, 3037 },
		{ 2847, 2845 },
		{ 2557, 2532 },
		{ 1418, 1396 },
		{ 1629, 1628 },
		{ 2854, 2853 },
		{ 2684, 2665 },
		{ 2213, 2212 },
		{ 2686, 2667 },
		{ 3055, 3052 },
		{ 2858, 2857 },
		{ 2057, 2039 },
		{ 2215, 2214 },
		{ 3064, 3063 },
		{ 2565, 2540 },
		{ 2065, 2047 },
		{ 2567, 2542 },
		{ 1419, 1397 },
		{ 1734, 1733 },
		{ 1679, 1678 },
		{ 1597, 1596 },
		{ 3076, 3073 },
		{ 1919, 1909 },
		{ 2577, 2552 },
		{ 2072, 2055 },
		{ 2878, 2877 },
		{ 3085, 3082 },
		{ 1287, 1286 },
		{ 1260, 1259 },
		{ 1827, 1807 },
		{ 1935, 1927 },
		{ 1936, 1928 },
		{ 1687, 1686 },
		{ 2586, 2561 },
		{ 2893, 2892 },
		{ 1345, 1344 },
		{ 2464, 2437 },
		{ 1857, 1838 },
		{ 2330, 2300 },
		{ 2050, 2029 },
		{ 1847, 1828 },
		{ 2575, 2550 },
		{ 1867, 1849 },
		{ 2825, 2824 },
		{ 1805, 1783 },
		{ 2944, 2943 },
		{ 3082, 3079 },
		{ 3046, 3042 },
		{ 1426, 1404 },
		{ 2848, 2846 },
		{ 2667, 2648 },
		{ 2037, 2016 },
		{ 2874, 2873 },
		{ 2009, 1986 },
		{ 2983, 2982 },
		{ 1344, 1343 },
		{ 128, 113 },
		{ 1772, 1771 },
		{ 3017, 3016 },
		{ 2910, 2909 },
		{ 2572, 2547 },
		{ 2827, 2826 },
		{ 1277, 1276 },
		{ 1347, 1346 },
		{ 1237, 1236 },
		{ 2968, 2967 },
		{ 3078, 3075 },
		{ 2999, 2998 },
		{ 3081, 3078 },
		{ 2522, 2492 },
		{ 2833, 2832 },
		{ 614, 556 },
		{ 2802, 2801 },
		{ 2099, 2084 },
		{ 2951, 2950 },
		{ 3089, 3086 },
		{ 2699, 2682 },
		{ 2879, 2878 },
		{ 1637, 1636 },
		{ 2927, 2926 },
		{ 3096, 3094 },
		{ 615, 556 },
		{ 1751, 1750 },
		{ 2788, 2787 },
		{ 1895, 1880 },
		{ 2564, 2539 },
		{ 1379, 1372 },
		{ 1990, 1984 },
		{ 1329, 1326 },
		{ 1799, 1795 },
		{ 2593, 2568 },
		{ 2614, 2589 },
		{ 2642, 2618 },
		{ 202, 179 },
		{ 2643, 2620 },
		{ 2417, 2390 },
		{ 200, 179 },
		{ 616, 556 },
		{ 201, 179 },
		{ 2852, 2851 },
		{ 2589, 2564 },
		{ 1821, 1797 },
		{ 2461, 2434 },
		{ 2005, 1981 },
		{ 2505, 2476 },
		{ 1708, 1707 },
		{ 1666, 1665 },
		{ 199, 179 },
		{ 1560, 1544 },
		{ 2008, 1985 },
		{ 2801, 2800 },
		{ 2113, 2099 },
		{ 1777, 1776 },
		{ 2556, 2531 },
		{ 2715, 2700 },
		{ 1944, 1941 },
		{ 1689, 1688 },
		{ 2607, 2582 },
		{ 2950, 2949 },
		{ 2810, 2809 },
		{ 1319, 1318 },
		{ 1309, 1308 },
		{ 1692, 1691 },
		{ 1239, 1238 },
		{ 2521, 2491 },
		{ 1614, 1613 },
		{ 2959, 2958 },
		{ 2199, 2198 },
		{ 3059, 3058 },
		{ 1658, 1657 },
		{ 2670, 2651 },
		{ 1604, 1603 },
		{ 2618, 2593 },
		{ 3066, 3065 },
		{ 2738, 2730 },
		{ 1908, 1895 },
		{ 3069, 3068 },
		{ 2152, 2145 },
		{ 1292, 1291 },
		{ 1916, 1906 },
		{ 2445, 2416 },
		{ 1767, 1766 },
		{ 2627, 2602 },
		{ 2209, 2208 },
		{ 3079, 3076 },
		{ 2629, 2604 },
		{ 2536, 2507 },
		{ 2982, 2981 },
		{ 2755, 2754 },
		{ 2909, 2908 },
		{ 2158, 2154 },
		{ 3086, 3083 },
		{ 2987, 2986 },
		{ 2687, 2668 },
		{ 2211, 2210 },
		{ 2990, 2989 },
		{ 1632, 1631 },
		{ 1682, 1681 },
		{ 1241, 1240 },
		{ 3095, 3093 },
		{ 1339, 1338 },
		{ 2918, 2917 },
		{ 2085, 2070 },
		{ 2086, 2073 },
		{ 851, 800 },
		{ 422, 381 },
		{ 674, 613 },
		{ 740, 683 },
		{ 1623, 25 },
		{ 1699, 31 },
		{ 1229, 11 },
		{ 67, 25 },
		{ 67, 31 },
		{ 67, 11 },
		{ 2779, 45 },
		{ 1328, 19 },
		{ 1725, 33 },
		{ 67, 45 },
		{ 67, 19 },
		{ 67, 33 },
		{ 1673, 29 },
		{ 423, 381 },
		{ 2886, 55 },
		{ 67, 29 },
		{ 739, 683 },
		{ 67, 55 },
		{ 1300, 17 },
		{ 332, 288 },
		{ 675, 613 },
		{ 67, 17 },
		{ 852, 800 },
		{ 1649, 27 },
		{ 2976, 59 },
		{ 1250, 13 },
		{ 67, 27 },
		{ 67, 59 },
		{ 67, 13 },
		{ 335, 291 },
		{ 342, 297 },
		{ 343, 298 },
		{ 348, 303 },
		{ 361, 319 },
		{ 2019, 1998 },
		{ 385, 344 },
		{ 388, 347 },
		{ 394, 353 },
		{ 405, 365 },
		{ 1831, 1812 },
		{ 412, 372 },
		{ 421, 380 },
		{ 230, 195 },
		{ 435, 391 },
		{ 234, 199 },
		{ 456, 406 },
		{ 459, 410 },
		{ 2041, 2020 },
		{ 2042, 2021 },
		{ 469, 418 },
		{ 470, 419 },
		{ 483, 432 },
		{ 491, 441 },
		{ 524, 465 },
		{ 534, 473 },
		{ 1851, 1832 },
		{ 1852, 1833 },
		{ 536, 475 },
		{ 537, 476 },
		{ 541, 480 },
		{ 2064, 2046 },
		{ 553, 494 },
		{ 557, 498 },
		{ 561, 502 },
		{ 571, 512 },
		{ 586, 525 },
		{ 587, 527 },
		{ 592, 532 },
		{ 235, 200 },
		{ 1872, 1854 },
		{ 2080, 2063 },
		{ 623, 560 },
		{ 636, 573 },
		{ 639, 576 },
		{ 648, 585 },
		{ 663, 599 },
		{ 664, 600 },
		{ 673, 612 },
		{ 241, 206 },
		{ 1887, 1871 },
		{ 693, 631 },
		{ 242, 207 },
		{ 1621, 25 },
		{ 1697, 31 },
		{ 1227, 11 },
		{ 741, 684 },
		{ 753, 695 },
		{ 755, 697 },
		{ 2777, 45 },
		{ 1326, 19 },
		{ 1723, 33 },
		{ 757, 699 },
		{ 762, 704 },
		{ 782, 724 },
		{ 1671, 29 },
		{ 791, 733 },
		{ 2885, 55 },
		{ 795, 737 },
		{ 796, 738 },
		{ 826, 772 },
		{ 1298, 17 },
		{ 843, 792 },
		{ 262, 223 },
		{ 881, 832 },
		{ 889, 840 },
		{ 1647, 27 },
		{ 2974, 59 },
		{ 1248, 13 },
		{ 904, 855 },
		{ 940, 894 },
		{ 944, 898 },
		{ 960, 917 },
		{ 975, 936 },
		{ 1000, 965 },
		{ 1003, 968 },
		{ 1010, 976 },
		{ 1020, 986 },
		{ 1037, 1005 },
		{ 1056, 1023 },
		{ 1063, 1032 },
		{ 1065, 1034 },
		{ 1077, 1050 },
		{ 1090, 1064 },
		{ 1095, 1070 },
		{ 1102, 1078 },
		{ 1103, 1079 },
		{ 1117, 1099 },
		{ 1134, 1118 },
		{ 1139, 1124 },
		{ 1146, 1135 },
		{ 1157, 1147 },
		{ 1163, 1153 },
		{ 1198, 1197 },
		{ 271, 232 },
		{ 278, 239 },
		{ 293, 251 },
		{ 300, 258 },
		{ 306, 263 },
		{ 315, 272 },
		{ 67, 15 },
		{ 67, 47 },
		{ 67, 23 },
		{ 428, 385 },
		{ 67, 51 },
		{ 429, 385 },
		{ 427, 385 },
		{ 67, 53 },
		{ 67, 35 },
		{ 67, 41 },
		{ 67, 57 },
		{ 218, 187 },
		{ 2092, 2078 },
		{ 2094, 2079 },
		{ 2153, 2146 },
		{ 216, 187 },
		{ 3030, 3029 },
		{ 461, 412 },
		{ 463, 412 },
		{ 2093, 2078 },
		{ 2095, 2079 },
		{ 717, 661 },
		{ 430, 385 },
		{ 431, 386 },
		{ 578, 519 },
		{ 404, 364 },
		{ 806, 748 },
		{ 493, 443 },
		{ 464, 412 },
		{ 2090, 2076 },
		{ 219, 187 },
		{ 217, 187 },
		{ 1899, 1885 },
		{ 462, 412 },
		{ 501, 450 },
		{ 502, 450 },
		{ 716, 660 },
		{ 1098, 1073 },
		{ 672, 611 },
		{ 803, 745 },
		{ 212, 184 },
		{ 288, 246 },
		{ 503, 450 },
		{ 211, 184 },
		{ 328, 284 },
		{ 514, 459 },
		{ 2023, 2002 },
		{ 2002, 1978 },
		{ 437, 393 },
		{ 213, 184 },
		{ 896, 847 },
		{ 354, 309 },
		{ 620, 557 },
		{ 895, 846 },
		{ 516, 459 },
		{ 2001, 1978 },
		{ 619, 557 },
		{ 505, 452 },
		{ 205, 181 },
		{ 515, 459 },
		{ 207, 181 },
		{ 225, 190 },
		{ 807, 749 },
		{ 206, 181 },
		{ 618, 557 },
		{ 617, 557 },
		{ 508, 454 },
		{ 2000, 1978 },
		{ 808, 750 },
		{ 509, 454 },
		{ 415, 374 },
		{ 414, 374 },
		{ 1107, 1083 },
		{ 224, 190 },
		{ 1834, 1815 },
		{ 265, 226 },
		{ 256, 218 },
		{ 506, 452 },
		{ 583, 524 },
		{ 1283, 15 },
		{ 2795, 47 },
		{ 1594, 23 },
		{ 584, 524 },
		{ 2841, 51 },
		{ 297, 255 },
		{ 2024, 2003 },
		{ 2868, 53 },
		{ 1757, 35 },
		{ 2194, 41 },
		{ 2933, 57 },
		{ 1106, 1083 },
		{ 222, 188 },
		{ 891, 842 },
		{ 585, 524 },
		{ 220, 188 },
		{ 570, 511 },
		{ 704, 646 },
		{ 221, 188 },
		{ 279, 240 },
		{ 521, 463 },
		{ 1814, 1790 },
		{ 1041, 1009 },
		{ 2780, 2777 },
		{ 787, 729 },
		{ 3028, 3026 },
		{ 559, 500 },
		{ 1064, 1033 },
		{ 538, 477 },
		{ 737, 681 },
		{ 3039, 3034 },
		{ 738, 682 },
		{ 236, 201 },
		{ 417, 376 },
		{ 280, 240 },
		{ 522, 463 },
		{ 951, 907 },
		{ 955, 912 },
		{ 687, 625 },
		{ 812, 754 },
		{ 1131, 1115 },
		{ 3050, 3047 },
		{ 603, 542 },
		{ 1137, 1121 },
		{ 479, 428 },
		{ 1285, 1283 },
		{ 705, 647 },
		{ 1230, 1227 },
		{ 1154, 1144 },
		{ 2441, 2412 },
		{ 1155, 1145 },
		{ 861, 811 },
		{ 707, 649 },
		{ 550, 491 },
		{ 208, 182 },
		{ 1135, 1119 },
		{ 397, 356 },
		{ 196, 178 },
		{ 823, 768 },
		{ 209, 182 },
		{ 1833, 1814 },
		{ 198, 178 },
		{ 1147, 1136 },
		{ 1149, 1138 },
		{ 824, 769 },
		{ 545, 485 },
		{ 834, 782 },
		{ 549, 491 },
		{ 197, 178 },
		{ 1161, 1151 },
		{ 1162, 1152 },
		{ 840, 789 },
		{ 1167, 1158 },
		{ 1182, 1173 },
		{ 546, 486 },
		{ 1209, 1208 },
		{ 2021, 2000 },
		{ 295, 253 },
		{ 859, 809 },
		{ 551, 492 },
		{ 866, 816 },
		{ 876, 827 },
		{ 344, 299 },
		{ 690, 628 },
		{ 347, 302 },
		{ 413, 373 },
		{ 3029, 3028 },
		{ 484, 433 },
		{ 565, 506 },
		{ 905, 856 },
		{ 911, 862 },
		{ 917, 869 },
		{ 922, 874 },
		{ 927, 881 },
		{ 932, 887 },
		{ 3043, 3039 },
		{ 939, 893 },
		{ 708, 650 },
		{ 569, 510 },
		{ 263, 224 },
		{ 718, 662 },
		{ 276, 237 },
		{ 961, 918 },
		{ 972, 933 },
		{ 3053, 3050 },
		{ 576, 517 },
		{ 980, 941 },
		{ 981, 942 },
		{ 985, 946 },
		{ 986, 947 },
		{ 992, 953 },
		{ 359, 316 },
		{ 303, 261 },
		{ 363, 321 },
		{ 512, 457 },
		{ 1022, 988 },
		{ 1024, 990 },
		{ 1028, 994 },
		{ 1035, 1003 },
		{ 1036, 1004 },
		{ 365, 323 },
		{ 518, 461 },
		{ 1049, 1016 },
		{ 605, 544 },
		{ 783, 725 },
		{ 784, 726 },
		{ 609, 549 },
		{ 1074, 1044 },
		{ 790, 732 },
		{ 1089, 1063 },
		{ 613, 555 },
		{ 1091, 1065 },
		{ 792, 734 },
		{ 434, 389 },
		{ 381, 341 },
		{ 801, 743 },
		{ 802, 744 },
		{ 1111, 1090 },
		{ 1116, 1096 },
		{ 338, 293 },
		{ 1124, 1107 },
		{ 1125, 1109 },
		{ 339, 294 },
		{ 391, 350 },
		{ 533, 472 },
		{ 2059, 2041 },
		{ 228, 193 },
		{ 1069, 1038 },
		{ 337, 293 },
		{ 336, 293 },
		{ 229, 194 },
		{ 669, 605 },
		{ 574, 515 },
		{ 1590, 1590 },
		{ 362, 320 },
		{ 931, 886 },
		{ 433, 388 },
		{ 580, 521 },
		{ 401, 361 },
		{ 692, 630 },
		{ 542, 481 },
		{ 495, 446 },
		{ 957, 914 },
		{ 345, 300 },
		{ 810, 752 },
		{ 706, 648 },
		{ 1129, 1113 },
		{ 596, 536 },
		{ 597, 537 },
		{ 548, 490 },
		{ 244, 209 },
		{ 1869, 1851 },
		{ 606, 545 },
		{ 726, 672 },
		{ 995, 957 },
		{ 1523, 1523 },
		{ 1590, 1590 },
		{ 1526, 1526 },
		{ 1529, 1529 },
		{ 1532, 1532 },
		{ 1535, 1535 },
		{ 1538, 1538 },
		{ 996, 958 },
		{ 997, 960 },
		{ 846, 795 },
		{ 847, 796 },
		{ 1005, 970 },
		{ 408, 368 },
		{ 611, 553 },
		{ 1021, 987 },
		{ 370, 328 },
		{ 1556, 1556 },
		{ 1192, 1186 },
		{ 322, 278 },
		{ 1025, 991 },
		{ 240, 205 },
		{ 355, 312 },
		{ 886, 837 },
		{ 1523, 1523 },
		{ 1565, 1565 },
		{ 1526, 1526 },
		{ 1529, 1529 },
		{ 1532, 1532 },
		{ 1535, 1535 },
		{ 1538, 1538 },
		{ 888, 839 },
		{ 630, 567 },
		{ 1486, 1486 },
		{ 1045, 1012 },
		{ 563, 504 },
		{ 389, 348 },
		{ 1061, 1029 },
		{ 646, 583 },
		{ 903, 854 },
		{ 1556, 1556 },
		{ 1396, 1590 },
		{ 644, 581 },
		{ 767, 709 },
		{ 906, 857 },
		{ 1081, 1054 },
		{ 1084, 1057 },
		{ 1085, 1058 },
		{ 1565, 1565 },
		{ 909, 860 },
		{ 773, 715 },
		{ 535, 474 },
		{ 919, 871 },
		{ 1097, 1072 },
		{ 920, 872 },
		{ 1099, 1074 },
		{ 1486, 1486 },
		{ 921, 873 },
		{ 1871, 1853 },
		{ 3025, 3021 },
		{ 478, 427 },
		{ 579, 520 },
		{ 215, 186 },
		{ 1396, 1523 },
		{ 392, 351 },
		{ 1396, 1526 },
		{ 1396, 1529 },
		{ 1396, 1532 },
		{ 1396, 1535 },
		{ 1396, 1538 },
		{ 323, 279 },
		{ 489, 438 },
		{ 589, 529 },
		{ 676, 614 },
		{ 591, 531 },
		{ 1886, 1870 },
		{ 353, 308 },
		{ 958, 915 },
		{ 398, 357 },
		{ 1396, 1556 },
		{ 805, 747 },
		{ 964, 924 },
		{ 969, 930 },
		{ 367, 325 },
		{ 973, 934 },
		{ 699, 637 },
		{ 703, 643 },
		{ 1396, 1565 },
		{ 255, 217 },
		{ 982, 943 },
		{ 375, 333 },
		{ 1913, 1901 },
		{ 1914, 1902 },
		{ 813, 755 },
		{ 820, 764 },
		{ 1396, 1486 },
		{ 2063, 2045 },
		{ 822, 766 },
		{ 1195, 1193 },
		{ 376, 335 },
		{ 1199, 1198 },
		{ 331, 287 },
		{ 998, 962 },
		{ 825, 770 },
		{ 513, 458 },
		{ 710, 652 },
		{ 711, 654 },
		{ 2077, 2060 },
		{ 1012, 978 },
		{ 713, 657 },
		{ 612, 554 },
		{ 277, 238 },
		{ 360, 317 },
		{ 722, 665 },
		{ 725, 670 },
		{ 299, 257 },
		{ 735, 679 },
		{ 418, 377 },
		{ 526, 467 },
		{ 634, 571 },
		{ 473, 422 },
		{ 638, 575 },
		{ 2110, 2096 },
		{ 2111, 2097 },
		{ 475, 424 },
		{ 1062, 1030 },
		{ 640, 577 },
		{ 900, 851 },
		{ 656, 592 },
		{ 657, 592 },
		{ 247, 211 },
		{ 497, 448 },
		{ 573, 514 },
		{ 246, 211 },
		{ 528, 469 },
		{ 529, 469 },
		{ 530, 470 },
		{ 531, 470 },
		{ 602, 541 },
		{ 282, 241 },
		{ 281, 241 },
		{ 601, 541 },
		{ 498, 448 },
		{ 204, 180 },
		{ 383, 342 },
		{ 382, 342 },
		{ 250, 214 },
		{ 287, 245 },
		{ 260, 221 },
		{ 251, 214 },
		{ 499, 449 },
		{ 750, 693 },
		{ 883, 834 },
		{ 203, 180 },
		{ 259, 221 },
		{ 304, 262 },
		{ 2096, 2080 },
		{ 286, 245 },
		{ 252, 214 },
		{ 438, 394 },
		{ 500, 449 },
		{ 751, 693 },
		{ 3047, 3043 },
		{ 305, 262 },
		{ 477, 426 },
		{ 1870, 1852 },
		{ 1109, 1088 },
		{ 1110, 1089 },
		{ 258, 220 },
		{ 1112, 1091 },
		{ 273, 234 },
		{ 809, 751 },
		{ 967, 927 },
		{ 1832, 1813 },
		{ 510, 455 },
		{ 1127, 1111 },
		{ 1835, 1816 },
		{ 756, 698 },
		{ 1888, 1872 },
		{ 458, 408 },
		{ 759, 701 },
		{ 1996, 1973 },
		{ 659, 594 },
		{ 763, 705 },
		{ 384, 343 },
		{ 2060, 2042 },
		{ 1144, 1133 },
		{ 270, 231 },
		{ 588, 528 },
		{ 1901, 1887 },
		{ 1066, 1035 },
		{ 988, 949 },
		{ 918, 870 },
		{ 712, 656 },
		{ 350, 305 },
		{ 2412, 2385 },
		{ 590, 530 },
		{ 626, 563 },
		{ 3026, 3024 },
		{ 351, 306 },
		{ 680, 618 },
		{ 2081, 2064 },
		{ 2020, 1999 },
		{ 681, 619 },
		{ 682, 620 },
		{ 289, 247 },
		{ 2025, 2004 },
		{ 474, 423 },
		{ 1201, 1200 },
		{ 1999, 1977 },
		{ 924, 876 },
		{ 2003, 1979 },
		{ 1815, 1791 },
		{ 226, 191 },
		{ 869, 819 },
		{ 1100, 1076 },
		{ 1006, 971 },
		{ 872, 823 },
		{ 873, 824 },
		{ 261, 222 },
		{ 2091, 2077 },
		{ 879, 830 },
		{ 949, 904 },
		{ 766, 708 },
		{ 698, 636 },
		{ 426, 384 },
		{ 701, 639 },
		{ 539, 478 },
		{ 1900, 1886 },
		{ 890, 841 },
		{ 1039, 1007 },
		{ 962, 922 },
		{ 594, 534 },
		{ 577, 518 },
		{ 1052, 1019 },
		{ 2129, 2110 },
		{ 436, 392 },
		{ 1143, 1132 },
		{ 1057, 1024 },
		{ 1058, 1025 },
		{ 1059, 1026 },
		{ 1924, 1913 },
		{ 294, 252 },
		{ 747, 690 },
		{ 828, 776 },
		{ 794, 736 },
		{ 749, 692 },
		{ 467, 415 },
		{ 581, 522 },
		{ 248, 212 },
		{ 1168, 1159 },
		{ 1179, 1170 },
		{ 1180, 1171 },
		{ 567, 508 },
		{ 1185, 1176 },
		{ 989, 950 },
		{ 1193, 1188 },
		{ 1194, 1189 },
		{ 1864, 1845 },
		{ 1082, 1055 },
		{ 854, 802 },
		{ 993, 954 },
		{ 1087, 1060 },
		{ 547, 489 },
		{ 480, 429 },
		{ 864, 814 },
		{ 1093, 1067 },
		{ 923, 875 },
		{ 1813, 1789 },
		{ 720, 664 },
		{ 1142, 1131 },
		{ 1070, 1039 },
		{ 1071, 1041 },
		{ 302, 260 },
		{ 245, 210 },
		{ 721, 664 },
		{ 1148, 1137 },
		{ 1079, 1052 },
		{ 1153, 1143 },
		{ 830, 778 },
		{ 264, 225 },
		{ 838, 787 },
		{ 897, 848 },
		{ 1086, 1059 },
		{ 390, 349 },
		{ 1164, 1155 },
		{ 1023, 989 },
		{ 622, 559 },
		{ 1177, 1168 },
		{ 1884, 1868 },
		{ 2075, 2058 },
		{ 800, 742 },
		{ 1027, 993 },
		{ 439, 395 },
		{ 849, 798 },
		{ 1186, 1179 },
		{ 1187, 1180 },
		{ 1191, 1185 },
		{ 444, 403 },
		{ 910, 861 },
		{ 1038, 1006 },
		{ 556, 497 },
		{ 1196, 1194 },
		{ 290, 248 },
		{ 1105, 1082 },
		{ 2022, 2001 },
		{ 635, 572 },
		{ 1208, 1207 },
		{ 1108, 1087 },
		{ 517, 460 },
		{ 1050, 1017 },
		{ 1051, 1018 },
		{ 714, 658 },
		{ 1113, 1093 },
		{ 371, 329 },
		{ 683, 621 },
		{ 1118, 1100 },
		{ 1123, 1106 },
		{ 686, 624 },
		{ 496, 447 },
		{ 877, 828 },
		{ 457, 407 },
		{ 566, 507 },
		{ 936, 890 },
		{ 3024, 3020 },
		{ 610, 550 },
		{ 647, 584 },
		{ 420, 379 },
		{ 1141, 1130 },
		{ 677, 615 },
		{ 952, 908 },
		{ 599, 539 },
		{ 330, 286 },
		{ 432, 387 },
		{ 653, 589 },
		{ 655, 591 },
		{ 442, 399 },
		{ 875, 826 },
		{ 411, 371 },
		{ 608, 548 },
		{ 231, 196 },
		{ 832, 780 },
		{ 758, 700 },
		{ 884, 835 },
		{ 1206, 1205 },
		{ 694, 632 },
		{ 925, 879 },
		{ 2061, 2043 },
		{ 2062, 2044 },
		{ 668, 604 },
		{ 402, 362 },
		{ 844, 793 },
		{ 582, 523 },
		{ 595, 535 },
		{ 771, 713 },
		{ 352, 307 },
		{ 233, 198 },
		{ 948, 902 },
		{ 1008, 974 },
		{ 724, 667 },
		{ 1011, 977 },
		{ 254, 216 },
		{ 598, 538 },
		{ 313, 270 },
		{ 232, 197 },
		{ 625, 562 },
		{ 318, 275 },
		{ 468, 417 },
		{ 1092, 1066 },
		{ 907, 858 },
		{ 865, 815 },
		{ 1031, 999 },
		{ 819, 763 },
		{ 1169, 1160 },
		{ 1174, 1165 },
		{ 867, 817 },
		{ 1178, 1169 },
		{ 916, 868 },
		{ 786, 728 },
		{ 284, 243 },
		{ 1183, 1174 },
		{ 1184, 1175 },
		{ 1104, 1080 },
		{ 1040, 1008 },
		{ 976, 937 },
		{ 1190, 1184 },
		{ 1043, 1010 },
		{ 977, 938 },
		{ 788, 730 },
		{ 688, 626 },
		{ 1044, 1010 },
		{ 346, 301 },
		{ 1042, 1010 },
		{ 691, 629 },
		{ 532, 471 },
		{ 829, 777 },
		{ 593, 533 },
		{ 1202, 1201 },
		{ 1122, 1104 },
		{ 266, 227 },
		{ 928, 882 },
		{ 929, 883 },
		{ 1126, 1110 },
		{ 885, 836 },
		{ 1128, 1112 },
		{ 1854, 1835 },
		{ 2046, 2025 },
		{ 695, 633 },
		{ 369, 327 },
		{ 760, 702 },
		{ 1001, 966 },
		{ 1002, 967 },
		{ 358, 315 },
		{ 1140, 1127 },
		{ 804, 746 },
		{ 1075, 1046 },
		{ 1384, 1384 },
		{ 831, 779 },
		{ 723, 666 },
		{ 984, 945 },
		{ 564, 505 },
		{ 520, 462 },
		{ 319, 276 },
		{ 2078, 2061 },
		{ 1033, 1000 },
		{ 519, 462 },
		{ 320, 276 },
		{ 728, 673 },
		{ 727, 673 },
		{ 1032, 1000 },
		{ 729, 673 },
		{ 2079, 2062 },
		{ 1019, 985 },
		{ 472, 421 },
		{ 868, 818 },
		{ 399, 358 },
		{ 870, 820 },
		{ 941, 895 },
		{ 942, 896 },
		{ 1384, 1384 },
		{ 1026, 992 },
		{ 285, 244 },
		{ 504, 451 },
		{ 743, 686 },
		{ 950, 905 },
		{ 746, 689 },
		{ 642, 579 },
		{ 954, 911 },
		{ 748, 691 },
		{ 572, 513 },
		{ 882, 833 },
		{ 443, 401 },
		{ 752, 694 },
		{ 507, 453 },
		{ 1130, 1114 },
		{ 604, 543 },
		{ 1132, 1116 },
		{ 966, 926 },
		{ 386, 345 },
		{ 403, 363 },
		{ 1055, 1022 },
		{ 971, 932 },
		{ 1885, 1869 },
		{ 237, 202 },
		{ 327, 283 },
		{ 660, 596 },
		{ 1060, 1028 },
		{ 761, 703 },
		{ 661, 597 },
		{ 709, 651 },
		{ 481, 430 },
		{ 1151, 1141 },
		{ 317, 274 },
		{ 253, 215 },
		{ 1067, 1036 },
		{ 486, 435 },
		{ 1159, 1149 },
		{ 2043, 2022 },
		{ 1396, 1384 },
		{ 2044, 2023 },
		{ 776, 718 },
		{ 987, 948 },
		{ 908, 859 },
		{ 777, 719 },
		{ 1076, 1049 },
		{ 991, 952 },
		{ 839, 788 },
		{ 1170, 1161 },
		{ 1171, 1162 },
		{ 555, 496 },
		{ 1176, 1167 },
		{ 841, 790 },
		{ 488, 437 },
		{ 2977, 2974 },
		{ 621, 558 },
		{ 379, 338 },
		{ 490, 439 },
		{ 1088, 1062 },
		{ 624, 561 },
		{ 364, 322 },
		{ 853, 801 },
		{ 492, 442 },
		{ 1189, 1182 },
		{ 395, 354 },
		{ 1007, 972 },
		{ 632, 569 },
		{ 357, 314 },
		{ 307, 264 },
		{ 798, 740 },
		{ 2076, 2059 },
		{ 1018, 984 },
		{ 1047, 1014 },
		{ 1048, 1015 },
		{ 850, 799 },
		{ 400, 359 },
		{ 1078, 1051 },
		{ 685, 623 },
		{ 1850, 1831 },
		{ 1927, 1918 },
		{ 2040, 2019 },
		{ 1650, 1647 },
		{ 558, 499 },
		{ 2144, 2131 },
		{ 697, 635 },
		{ 797, 739 },
		{ 1674, 1671 },
		{ 494, 444 },
		{ 329, 285 },
		{ 898, 849 },
		{ 1759, 1757 },
		{ 1624, 1621 },
		{ 1909, 1896 },
		{ 2102, 2087 },
		{ 1700, 1697 },
		{ 1301, 1298 },
		{ 1133, 1117 },
		{ 1596, 1594 },
		{ 983, 944 },
		{ 933, 888 },
		{ 1121, 1103 },
		{ 833, 781 },
		{ 719, 663 },
		{ 934, 888 },
		{ 476, 425 },
		{ 818, 762 },
		{ 368, 326 },
		{ 326, 282 },
		{ 821, 765 },
		{ 1101, 1077 },
		{ 419, 378 },
		{ 689, 627 },
		{ 678, 616 },
		{ 765, 707 },
		{ 901, 852 },
		{ 1083, 1056 },
		{ 378, 337 },
		{ 926, 880 },
		{ 731, 675 },
		{ 239, 204 },
		{ 1009, 975 },
		{ 1114, 1094 },
		{ 1115, 1095 },
		{ 1145, 1134 },
		{ 283, 242 },
		{ 425, 383 },
		{ 811, 753 },
		{ 1120, 1102 },
		{ 1188, 1181 },
		{ 396, 355 },
		{ 2097, 2081 },
		{ 860, 810 },
		{ 272, 233 },
		{ 863, 813 },
		{ 340, 295 },
		{ 627, 564 },
		{ 2104, 2090 },
		{ 2106, 2092 },
		{ 2107, 2093 },
		{ 2108, 2094 },
		{ 2109, 2095 },
		{ 1029, 995 },
		{ 733, 677 },
		{ 2004, 1980 },
		{ 943, 897 },
		{ 1853, 1834 },
		{ 1034, 1001 },
		{ 238, 203 },
		{ 324, 280 },
		{ 380, 339 },
		{ 552, 493 },
		{ 871, 821 },
		{ 210, 183 },
		{ 742, 685 },
		{ 637, 574 },
		{ 956, 913 },
		{ 554, 495 },
		{ 511, 456 },
		{ 959, 916 },
		{ 1868, 1850 },
		{ 878, 829 },
		{ 296, 254 },
		{ 880, 831 },
		{ 1054, 1021 },
		{ 641, 578 },
		{ 965, 925 },
		{ 308, 265 },
		{ 696, 634 },
		{ 968, 928 },
		{ 643, 580 },
		{ 970, 931 },
		{ 310, 267 },
		{ 816, 758 },
		{ 645, 582 },
		{ 407, 367 },
		{ 268, 229 },
		{ 409, 369 },
		{ 978, 939 },
		{ 893, 844 },
		{ 894, 845 },
		{ 2045, 2024 },
		{ 652, 588 },
		{ 410, 370 },
		{ 654, 590 },
		{ 482, 431 },
		{ 899, 850 },
		{ 441, 397 },
		{ 527, 468 },
		{ 1902, 1888 },
		{ 387, 346 },
		{ 770, 712 },
		{ 349, 304 },
		{ 298, 256 },
		{ 2058, 2040 },
		{ 774, 716 },
		{ 1911, 1899 },
		{ 775, 717 },
		{ 1181, 1172 },
		{ 835, 784 },
		{ 445, 404 },
		{ 1816, 1792 },
		{ 999, 963 },
		{ 715, 659 },
		{ 915, 866 },
		{ 778, 720 },
		{ 1094, 1068 },
		{ 781, 723 },
		{ 842, 791 },
		{ 269, 230 },
		{ 416, 375 },
		{ 334, 290 },
		{ 785, 727 },
		{ 848, 797 },
		{ 193, 174 },
		{ 1197, 1195 },
		{ 214, 185 },
		{ 1013, 979 },
		{ 1200, 1199 },
		{ 1017, 983 },
		{ 460, 411 },
		{ 1205, 1204 },
		{ 372, 330 },
		{ 1207, 1206 },
		{ 465, 413 },
		{ 855, 804 },
		{ 857, 807 },
		{ 2171, 2171 },
		{ 2171, 2171 },
		{ 1920, 1920 },
		{ 1920, 1920 },
		{ 1922, 1922 },
		{ 1922, 1922 },
		{ 1533, 1533 },
		{ 1533, 1533 },
		{ 2117, 2117 },
		{ 2117, 2117 },
		{ 2119, 2119 },
		{ 2119, 2119 },
		{ 2121, 2121 },
		{ 2121, 2121 },
		{ 2123, 2123 },
		{ 2123, 2123 },
		{ 2125, 2125 },
		{ 2125, 2125 },
		{ 2127, 2127 },
		{ 2127, 2127 },
		{ 1897, 1897 },
		{ 1897, 1897 },
		{ 568, 509 },
		{ 2171, 2171 },
		{ 979, 940 },
		{ 1920, 1920 },
		{ 651, 587 },
		{ 1922, 1922 },
		{ 2196, 2194 },
		{ 1533, 1533 },
		{ 671, 610 },
		{ 2117, 2117 },
		{ 912, 863 },
		{ 2119, 2119 },
		{ 913, 864 },
		{ 2121, 2121 },
		{ 249, 213 },
		{ 2123, 2123 },
		{ 963, 923 },
		{ 2125, 2125 },
		{ 314, 271 },
		{ 2127, 2127 },
		{ 466, 414 },
		{ 1897, 1897 },
		{ 649, 586 },
		{ 650, 586 },
		{ 1932, 1932 },
		{ 1932, 1932 },
		{ 2142, 2142 },
		{ 2142, 2142 },
		{ 1591, 1591 },
		{ 1591, 1591 },
		{ 2172, 2171 },
		{ 560, 501 },
		{ 1921, 1920 },
		{ 540, 479 },
		{ 1923, 1922 },
		{ 274, 235 },
		{ 1534, 1533 },
		{ 406, 366 },
		{ 2118, 2117 },
		{ 543, 483 },
		{ 2120, 2119 },
		{ 1068, 1037 },
		{ 2122, 2121 },
		{ 309, 266 },
		{ 2124, 2123 },
		{ 845, 794 },
		{ 2126, 2125 },
		{ 1932, 1932 },
		{ 2128, 2127 },
		{ 2142, 2142 },
		{ 1898, 1897 },
		{ 1591, 1591 },
		{ 1937, 1937 },
		{ 1937, 1937 },
		{ 2088, 2088 },
		{ 2088, 2088 },
		{ 2148, 2148 },
		{ 2148, 2148 },
		{ 1536, 1536 },
		{ 1536, 1536 },
		{ 1527, 1527 },
		{ 1527, 1527 },
		{ 1539, 1539 },
		{ 1539, 1539 },
		{ 1487, 1487 },
		{ 1487, 1487 },
		{ 1566, 1566 },
		{ 1566, 1566 },
		{ 1530, 1530 },
		{ 1530, 1530 },
		{ 1951, 1951 },
		{ 1951, 1951 },
		{ 1557, 1557 },
		{ 1557, 1557 },
		{ 292, 250 },
		{ 1937, 1937 },
		{ 1933, 1932 },
		{ 2088, 2088 },
		{ 2143, 2142 },
		{ 2148, 2148 },
		{ 1592, 1591 },
		{ 1536, 1536 },
		{ 1072, 1042 },
		{ 1527, 1527 },
		{ 1073, 1043 },
		{ 1539, 1539 },
		{ 684, 622 },
		{ 1487, 1487 },
		{ 772, 714 },
		{ 1566, 1566 },
		{ 754, 696 },
		{ 1530, 1530 },
		{ 1765, 1764 },
		{ 1951, 1951 },
		{ 1096, 1071 },
		{ 1557, 1557 },
		{ 1524, 1524 },
		{ 1524, 1524 },
		{ 815, 757 },
		{ 892, 843 },
		{ 1156, 1146 },
		{ 195, 176 },
		{ 1158, 1148 },
		{ 799, 741 },
		{ 1938, 1937 },
		{ 374, 332 },
		{ 2089, 2088 },
		{ 1893, 1878 },
		{ 2149, 2148 },
		{ 945, 899 },
		{ 1537, 1536 },
		{ 1053, 1020 },
		{ 1528, 1527 },
		{ 3057, 3055 },
		{ 1540, 1539 },
		{ 947, 901 },
		{ 1488, 1487 },
		{ 1166, 1157 },
		{ 1567, 1566 },
		{ 1524, 1524 },
		{ 1531, 1530 },
		{ 1204, 1203 },
		{ 1952, 1951 },
		{ 734, 678 },
		{ 1558, 1557 },
		{ 764, 706 },
		{ 1680, 1679 },
		{ 1630, 1629 },
		{ 1030, 997 },
		{ 341, 296 },
		{ 1136, 1120 },
		{ 1173, 1164 },
		{ 736, 680 },
		{ 1138, 1123 },
		{ 789, 731 },
		{ 953, 910 },
		{ 887, 838 },
		{ 862, 812 },
		{ 1014, 980 },
		{ 1015, 981 },
		{ 994, 955 },
		{ 1307, 1306 },
		{ 1518, 1500 },
		{ 1925, 1914 },
		{ 1602, 1601 },
		{ 935, 889 },
		{ 3027, 3025 },
		{ 1873, 1856 },
		{ 1525, 1524 },
		{ 1119, 1101 },
		{ 836, 785 },
		{ 2066, 2049 },
		{ 1706, 1705 },
		{ 814, 756 },
		{ 1656, 1655 },
		{ 1150, 1139 },
		{ 1046, 1013 },
		{ 1152, 1142 },
		{ 2130, 2111 },
		{ 2010, 1987 },
		{ 487, 436 },
		{ 223, 189 },
		{ 607, 547 },
		{ 1812, 1788 },
		{ 2141, 2129 },
		{ 1931, 1924 },
		{ 1912, 1900 },
		{ 1251, 1248 },
		{ 1172, 1163 },
		{ 301, 259 },
		{ 1998, 1976 },
		{ 2105, 2091 },
		{ 2870, 2868 },
		{ 2672, 2672 },
		{ 2672, 2672 },
		{ 2972, 2972 },
		{ 2972, 2972 },
		{ 2726, 2726 },
		{ 2726, 2726 },
		{ 2610, 2610 },
		{ 2610, 2610 },
		{ 2472, 2472 },
		{ 2472, 2472 },
		{ 2363, 2363 },
		{ 2363, 2363 },
		{ 1778, 1778 },
		{ 1778, 1778 },
		{ 2731, 2731 },
		{ 2731, 2731 },
		{ 1755, 1755 },
		{ 1755, 1755 },
		{ 1246, 1246 },
		{ 1246, 1246 },
		{ 2616, 2616 },
		{ 2616, 2616 },
		{ 312, 269 },
		{ 2672, 2672 },
		{ 628, 565 },
		{ 2972, 2972 },
		{ 665, 601 },
		{ 2726, 2726 },
		{ 666, 602 },
		{ 2610, 2610 },
		{ 667, 603 },
		{ 2472, 2472 },
		{ 629, 566 },
		{ 2363, 2363 },
		{ 424, 382 },
		{ 1778, 1778 },
		{ 990, 951 },
		{ 2731, 2731 },
		{ 670, 609 },
		{ 1755, 1755 },
		{ 631, 568 },
		{ 1246, 1246 },
		{ 325, 281 },
		{ 2616, 2616 },
		{ 633, 570 },
		{ 2496, 2496 },
		{ 2496, 2496 },
		{ 1669, 1669 },
		{ 1669, 1669 },
		{ 2691, 2672 },
		{ 930, 885 },
		{ 2973, 2972 },
		{ 1203, 1202 },
		{ 2734, 2726 },
		{ 243, 208 },
		{ 2634, 2610 },
		{ 817, 761 },
		{ 2501, 2472 },
		{ 874, 825 },
		{ 2364, 2363 },
		{ 275, 236 },
		{ 1779, 1778 },
		{ 768, 710 },
		{ 2739, 2731 },
		{ 2935, 2933 },
		{ 1756, 1755 },
		{ 937, 891 },
		{ 1247, 1246 },
		{ 2496, 2496 },
		{ 2640, 2616 },
		{ 1669, 1669 },
		{ 938, 892 },
		{ 2741, 2741 },
		{ 2741, 2741 },
		{ 2866, 2866 },
		{ 2866, 2866 },
		{ 2742, 2742 },
		{ 2742, 2742 },
		{ 2743, 2743 },
		{ 2743, 2743 },
		{ 1695, 1695 },
		{ 1695, 1695 },
		{ 2500, 2500 },
		{ 2500, 2500 },
		{ 2931, 2931 },
		{ 2931, 2931 },
		{ 2654, 2654 },
		{ 2654, 2654 },
		{ 2655, 2655 },
		{ 2655, 2655 },
		{ 2751, 2751 },
		{ 2751, 2751 },
		{ 2410, 2410 },
		{ 2410, 2410 },
		{ 2527, 2496 },
		{ 2741, 2741 },
		{ 1670, 1669 },
		{ 2866, 2866 },
		{ 769, 711 },
		{ 2742, 2742 },
		{ 1004, 969 },
		{ 2743, 2743 },
		{ 356, 313 },
		{ 1695, 1695 },
		{ 485, 434 },
		{ 2500, 2500 },
		{ 2163, 2160 },
		{ 2931, 2931 },
		{ 679, 617 },
		{ 2654, 2654 },
		{ 600, 540 },
		{ 2655, 2655 },
		{ 544, 484 },
		{ 2751, 2751 },
		{ 291, 249 },
		{ 2410, 2410 },
		{ 1080, 1053 },
		{ 2692, 2692 },
		{ 2692, 2692 },
		{ 2594, 2594 },
		{ 2594, 2594 },
		{ 2746, 2741 },
		{ 946, 900 },
		{ 2867, 2866 },
		{ 827, 773 },
		{ 2747, 2742 },
		{ 373, 331 },
		{ 2748, 2743 },
		{ 316, 273 },
		{ 1696, 1695 },
		{ 575, 516 },
		{ 2530, 2500 },
		{ 1016, 982 },
		{ 2932, 2931 },
		{ 779, 721 },
		{ 2673, 2654 },
		{ 780, 722 },
		{ 2674, 2655 },
		{ 730, 674 },
		{ 2753, 2751 },
		{ 2692, 2692 },
		{ 2439, 2410 },
		{ 2594, 2594 },
		{ 1726, 1723 },
		{ 1281, 1281 },
		{ 1281, 1281 },
		{ 2523, 2523 },
		{ 2523, 2523 },
		{ 1324, 1324 },
		{ 1324, 1324 },
		{ 2217, 2217 },
		{ 2217, 2217 },
		{ 2883, 2883 },
		{ 2883, 2883 },
		{ 2760, 2760 },
		{ 2760, 2760 },
		{ 1619, 1619 },
		{ 1619, 1619 },
		{ 2702, 2702 },
		{ 2702, 2702 },
		{ 1296, 1296 },
		{ 1296, 1296 },
		{ 2664, 2664 },
		{ 2664, 2664 },
		{ 1645, 1645 },
		{ 1645, 1645 },
		{ 2707, 2692 },
		{ 1281, 1281 },
		{ 2619, 2594 },
		{ 2523, 2523 },
		{ 393, 352 },
		{ 1324, 1324 },
		{ 732, 676 },
		{ 2217, 2217 },
		{ 1160, 1150 },
		{ 2883, 2883 },
		{ 227, 192 },
		{ 2760, 2760 },
		{ 837, 786 },
		{ 1619, 1619 },
		{ 523, 464 },
		{ 2702, 2702 },
		{ 257, 219 },
		{ 1296, 1296 },
		{ 1165, 1156 },
		{ 2664, 2664 },
		{ 2797, 2795 },
		{ 1645, 1645 },
		{ 525, 466 },
		{ 2706, 2706 },
		{ 2706, 2706 },
		{ 2837, 2837 },
		{ 2837, 2837 },
		{ 1282, 1281 },
		{ 377, 336 },
		{ 2524, 2523 },
		{ 471, 420 },
		{ 1325, 1324 },
		{ 267, 228 },
		{ 2218, 2217 },
		{ 2888, 2885 },
		{ 2884, 2883 },
		{ 440, 396 },
		{ 2761, 2760 },
		{ 902, 853 },
		{ 1620, 1619 },
		{ 333, 289 },
		{ 2717, 2702 },
		{ 793, 735 },
		{ 1297, 1296 },
		{ 321, 277 },
		{ 2683, 2664 },
		{ 2706, 2706 },
		{ 1646, 1645 },
		{ 2837, 2837 },
		{ 1175, 1166 },
		{ 2708, 2708 },
		{ 2708, 2708 },
		{ 2709, 2709 },
		{ 2709, 2709 },
		{ 2710, 2710 },
		{ 2710, 2710 },
		{ 2711, 2711 },
		{ 2711, 2711 },
		{ 2469, 2469 },
		{ 2469, 2469 },
		{ 1721, 1721 },
		{ 1721, 1721 },
		{ 2605, 2605 },
		{ 2605, 2605 },
		{ 2581, 2581 },
		{ 2581, 2581 },
		{ 2418, 2418 },
		{ 2418, 2418 },
		{ 2793, 2793 },
		{ 2793, 2793 },
		{ 2609, 2609 },
		{ 2609, 2609 },
		{ 2721, 2706 },
		{ 2708, 2708 },
		{ 2838, 2837 },
		{ 2709, 2709 },
		{ 744, 687 },
		{ 2710, 2710 },
		{ 745, 688 },
		{ 2711, 2711 },
		{ 194, 175 },
		{ 2469, 2469 },
		{ 658, 593 },
		{ 1721, 1721 },
		{ 311, 268 },
		{ 2605, 2605 },
		{ 974, 935 },
		{ 2581, 2581 },
		{ 700, 638 },
		{ 2418, 2418 },
		{ 562, 503 },
		{ 2793, 2793 },
		{ 856, 806 },
		{ 2609, 2609 },
		{ 914, 865 },
		{ 2719, 2719 },
		{ 2719, 2719 },
		{ 702, 642 },
		{ 858, 808 },
		{ 2722, 2708 },
		{ 366, 324 },
		{ 2723, 2709 },
		{ 662, 598 },
		{ 2724, 2710 },
		{ 3088, 3088 },
		{ 2725, 2711 },
		{ 3096, 3096 },
		{ 2498, 2469 },
		{ 3100, 3100 },
		{ 1722, 1721 },
		{ 2174, 2172 },
		{ 2630, 2605 },
		{ 2150, 2143 },
		{ 2606, 2581 },
		{ 1506, 1488 },
		{ 2419, 2418 },
		{ 1942, 1938 },
		{ 2794, 2793 },
		{ 2719, 2719 },
		{ 2633, 2609 },
		{ 1929, 1921 },
		{ 1554, 1540 },
		{ 2135, 2118 },
		{ 2155, 2149 },
		{ 1930, 1923 },
		{ 2136, 2120 },
		{ 1552, 1534 },
		{ 3088, 3088 },
		{ 2137, 2122 },
		{ 3096, 3096 },
		{ 1549, 1525 },
		{ 3100, 3100 },
		{ 2138, 2124 },
		{ 1551, 1531 },
		{ 2139, 2126 },
		{ 1553, 1537 },
		{ 2140, 2128 },
		{ 1910, 1898 },
		{ 1593, 1592 },
		{ 2103, 2089 },
		{ 1568, 1558 },
		{ 1953, 1952 },
		{ 1939, 1933 },
		{ 1574, 1567 },
		{ 2732, 2719 },
		{ 1550, 1528 },
		{ 3004, 3003 },
		{ 1581, 1576 },
		{ 1356, 1355 },
		{ 1711, 1710 },
		{ 3062, 3061 },
		{ 3063, 3062 },
		{ 1712, 1711 },
		{ 3091, 3088 },
		{ 1607, 1606 },
		{ 3098, 3096 },
		{ 1608, 1607 },
		{ 3101, 3100 },
		{ 1635, 1634 },
		{ 1770, 1769 },
		{ 1771, 1770 },
		{ 1636, 1635 },
		{ 1685, 1684 },
		{ 1686, 1685 },
		{ 1313, 1312 },
		{ 1584, 1580 },
		{ 1661, 1660 },
		{ 1662, 1661 },
		{ 1312, 1311 },
		{ 1580, 1575 },
		{ 2767, 2767 },
		{ 2764, 2767 },
		{ 163, 163 },
		{ 160, 163 },
		{ 1882, 1867 },
		{ 1883, 1867 },
		{ 1904, 1892 },
		{ 1905, 1892 },
		{ 2184, 2184 },
		{ 1958, 1958 },
		{ 1963, 1959 },
		{ 88, 70 },
		{ 2766, 2762 },
		{ 162, 158 },
		{ 2243, 2219 },
		{ 1962, 1959 },
		{ 87, 70 },
		{ 2765, 2762 },
		{ 161, 158 },
		{ 2242, 2219 },
		{ 2773, 2770 },
		{ 2772, 2768 },
		{ 168, 164 },
		{ 2767, 2767 },
		{ 2189, 2185 },
		{ 163, 163 },
		{ 2771, 2768 },
		{ 167, 164 },
		{ 2775, 2774 },
		{ 2188, 2185 },
		{ 1842, 1823 },
		{ 2184, 2184 },
		{ 1958, 1958 },
		{ 1964, 1961 },
		{ 1966, 1965 },
		{ 2190, 2187 },
		{ 2301, 2271 },
		{ 2192, 2191 },
		{ 2768, 2767 },
		{ 2035, 2014 },
		{ 164, 163 },
		{ 119, 103 },
		{ 169, 166 },
		{ 171, 170 },
		{ 2179, 2178 },
		{ 2033, 2012 },
		{ 2185, 2184 },
		{ 1959, 1958 },
		{ 2774, 2772 },
		{ 2187, 2183 },
		{ 1965, 1963 },
		{ 2014, 1992 },
		{ 170, 168 },
		{ 2191, 2189 },
		{ 2271, 2243 },
		{ 1961, 1957 },
		{ 2770, 2766 },
		{ 166, 162 },
		{ 1823, 1801 },
		{ 103, 88 },
		{ 2674, 2674 },
		{ 2674, 2674 },
		{ 2739, 2739 },
		{ 2739, 2739 },
		{ 0, 2994 },
		{ 0, 2488 },
		{ 0, 2440 },
		{ 0, 2903 },
		{ 0, 2551 },
		{ 0, 2821 },
		{ 0, 2906 },
		{ 2630, 2630 },
		{ 2630, 2630 },
		{ 0, 2332 },
		{ 0, 2440 },
		{ 2364, 2364 },
		{ 2364, 2364 },
		{ 2746, 2746 },
		{ 2746, 2746 },
		{ 0, 2911 },
		{ 2747, 2747 },
		{ 2747, 2747 },
		{ 0, 2828 },
		{ 2674, 2674 },
		{ 0, 2681 },
		{ 2739, 2739 },
		{ 2748, 2748 },
		{ 2748, 2748 },
		{ 2633, 2633 },
		{ 2633, 2633 },
		{ 2683, 2683 },
		{ 2683, 2683 },
		{ 2634, 2634 },
		{ 2634, 2634 },
		{ 2630, 2630 },
		{ 2753, 2753 },
		{ 2753, 2753 },
		{ 0, 2591 },
		{ 2364, 2364 },
		{ 0, 2554 },
		{ 2746, 2746 },
		{ 0, 2922 },
		{ 0, 2467 },
		{ 2747, 2747 },
		{ 0, 2332 },
		{ 0, 1730 },
		{ 0, 2690 },
		{ 2691, 2691 },
		{ 2691, 2691 },
		{ 2748, 2748 },
		{ 0, 2693 },
		{ 2633, 2633 },
		{ 0, 2695 },
		{ 2683, 2683 },
		{ 0, 2696 },
		{ 2634, 2634 },
		{ 2761, 2761 },
		{ 2761, 2761 },
		{ 2753, 2753 },
		{ 2192, 2192 },
		{ 2193, 2192 },
		{ 2524, 2524 },
		{ 2524, 2524 },
		{ 0, 2598 },
		{ 0, 2936 },
		{ 0, 2849 },
		{ 0, 2644 },
		{ 0, 1233 },
		{ 0, 2940 },
		{ 0, 2944 },
		{ 2691, 2691 },
		{ 171, 171 },
		{ 172, 171 },
		{ 2775, 2775 },
		{ 2776, 2775 },
		{ 0, 2647 },
		{ 2498, 2498 },
		{ 2498, 2498 },
		{ 0, 2947 },
		{ 2761, 2761 },
		{ 0, 1741 },
		{ 0, 2858 },
		{ 2192, 2192 },
		{ 0, 2565 },
		{ 2524, 2524 },
		{ 2530, 2530 },
		{ 2530, 2530 },
		{ 0, 2952 },
		{ 0, 2783 },
		{ 0, 1272 },
		{ 2707, 2707 },
		{ 2707, 2707 },
		{ 2606, 2606 },
		{ 2606, 2606 },
		{ 171, 171 },
		{ 0, 2567 },
		{ 2775, 2775 },
		{ 0, 2849 },
		{ 0, 1267 },
		{ 2498, 2498 },
		{ 2501, 2501 },
		{ 2501, 2501 },
		{ 0, 2428 },
		{ 0, 2429 },
		{ 0, 2477 },
		{ 0, 2963 },
		{ 0, 2430 },
		{ 0, 2538 },
		{ 2530, 2530 },
		{ 0, 2874 },
		{ 2717, 2717 },
		{ 2717, 2717 },
		{ 0, 2575 },
		{ 2707, 2707 },
		{ 0, 2798 },
		{ 2606, 2606 },
		{ 2723, 2723 },
		{ 2723, 2723 },
		{ 1966, 1966 },
		{ 1967, 1966 },
		{ 0, 2803 },
		{ 2619, 2619 },
		{ 2619, 2619 },
		{ 2501, 2501 },
		{ 0, 1288 },
		{ 0, 1746 },
		{ 0, 1256 },
		{ 0, 2889 },
		{ 0, 2545 },
		{ 0, 2983 },
		{ 2732, 2732 },
		{ 2732, 2732 },
		{ 0, 2733 },
		{ 2717, 2717 },
		{ 0, 2893 },
		{ 2734, 2734 },
		{ 2734, 2734 },
		{ 0, 2206 },
		{ 0, 2398 },
		{ 2723, 2723 },
		{ 0, 2814 },
		{ 1966, 1966 },
		{ 2439, 2439 },
		{ 2439, 2439 },
		{ 2619, 2619 },
		{ 1212, 1212 },
		{ 1355, 1354 },
		{ 2186, 2188 },
		{ 0, 2242 },
		{ 2769, 2771 },
		{ 0, 87 },
		{ 1861, 1842 },
		{ 165, 161 },
		{ 2732, 2732 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2734, 2734 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2439, 2439 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1212, 1212 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -68, 16, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 44, 572, 0 },
		{ -177, 2811, 0 },
		{ 5, 0, 0 },
		{ -1211, 1017, -31 },
		{ 7, 0, -31 },
		{ -1215, 1831, -33 },
		{ 9, 0, -33 },
		{ -1228, 3133, 144 },
		{ 11, 0, 144 },
		{ -1249, 3156, 152 },
		{ 13, 0, 152 },
		{ -1284, 3267, 0 },
		{ 15, 0, 0 },
		{ -1299, 3149, 140 },
		{ 17, 0, 140 },
		{ -1327, 3138, 22 },
		{ 19, 0, 22 },
		{ -1369, 230, 0 },
		{ 21, 0, 0 },
		{ -1595, 3269, 0 },
		{ 23, 0, 0 },
		{ -1622, 3131, 0 },
		{ 25, 0, 0 },
		{ -1648, 3154, 0 },
		{ 27, 0, 0 },
		{ -1672, 3143, 0 },
		{ 29, 0, 0 },
		{ -1698, 3132, 0 },
		{ 31, 0, 0 },
		{ -1724, 3139, 156 },
		{ 33, 0, 156 },
		{ -1758, 3275, 263 },
		{ 35, 0, 263 },
		{ 38, 127, 0 },
		{ -1794, 344, 0 },
		{ 40, 16, 0 },
		{ -1983, 116, 0 },
		{ -2195, 3276, 0 },
		{ 41, 0, 0 },
		{ 44, 14, 0 },
		{ -86, 458, 0 },
		{ -2778, 3137, 148 },
		{ 45, 0, 148 },
		{ -2796, 3268, 171 },
		{ 47, 0, 171 },
		{ 2840, 1487, 0 },
		{ 49, 0, 0 },
		{ -2842, 3271, 269 },
		{ 51, 0, 269 },
		{ -2869, 3274, 174 },
		{ 53, 0, 174 },
		{ -2887, 3145, 167 },
		{ 55, 0, 167 },
		{ -2934, 3277, 160 },
		{ 57, 0, 160 },
		{ -2975, 3155, 166 },
		{ 59, 0, 166 },
		{ -86, 1, 0 },
		{ 61, 0, 0 },
		{ -3023, 1791, 0 },
		{ 63, 0, 0 },
		{ -3033, 1700, 42 },
		{ 65, 0, 42 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 422 },
		{ 2774, 4698, 429 },
		{ 0, 0, 241 },
		{ 0, 0, 243 },
		{ 157, 1274, 260 },
		{ 157, 1401, 260 },
		{ 157, 1300, 260 },
		{ 157, 1307, 260 },
		{ 157, 1307, 260 },
		{ 157, 1311, 260 },
		{ 157, 1316, 260 },
		{ 157, 1310, 260 },
		{ 3082, 2809, 429 },
		{ 157, 1321, 260 },
		{ 3082, 1623, 259 },
		{ 102, 2540, 429 },
		{ 157, 0, 260 },
		{ 0, 0, 429 },
		{ -87, 4929, 237 },
		{ -88, 4746, 0 },
		{ 157, 1338, 260 },
		{ 157, 722, 260 },
		{ 157, 717, 260 },
		{ 157, 731, 260 },
		{ 157, 713, 260 },
		{ 157, 713, 260 },
		{ 157, 721, 260 },
		{ 157, 772, 260 },
		{ 157, 765, 260 },
		{ 3051, 2281, 0 },
		{ 157, 755, 260 },
		{ 3082, 1712, 256 },
		{ 117, 1440, 0 },
		{ 3082, 1701, 257 },
		{ 2774, 4723, 0 },
		{ 157, 764, 260 },
		{ 157, 762, 260 },
		{ 157, 763, 260 },
		{ 157, 759, 260 },
		{ 157, 0, 248 },
		{ 157, 804, 260 },
		{ 157, 806, 260 },
		{ 157, 790, 260 },
		{ 157, 795, 260 },
		{ 3079, 2929, 0 },
		{ 157, 803, 260 },
		{ 131, 1454, 0 },
		{ 117, 0, 0 },
		{ 3008, 2580, 258 },
		{ 133, 1521, 0 },
		{ 0, 0, 239 },
		{ 157, 835, 244 },
		{ 157, 837, 260 },
		{ 157, 829, 260 },
		{ 157, 834, 260 },
		{ 157, 832, 260 },
		{ 157, 825, 260 },
		{ 157, 0, 251 },
		{ 157, 826, 260 },
		{ 0, 0, 253 },
		{ 157, 832, 260 },
		{ 131, 0, 0 },
		{ 3008, 2636, 256 },
		{ 133, 0, 0 },
		{ 3008, 2669, 257 },
		{ 157, 847, 260 },
		{ 157, 844, 260 },
		{ 157, 845, 260 },
		{ 157, 873, 260 },
		{ 157, 932, 260 },
		{ 157, 0, 250 },
		{ 157, 1010, 260 },
		{ 157, 1083, 260 },
		{ 157, 1076, 260 },
		{ 157, 0, 246 },
		{ 157, 1163, 260 },
		{ 157, 0, 247 },
		{ 157, 0, 249 },
		{ 157, 1219, 260 },
		{ 157, 1254, 260 },
		{ 157, 0, 245 },
		{ 157, 1274, 260 },
		{ 157, 0, 252 },
		{ 157, 741, 260 },
		{ 157, 1301, 260 },
		{ 0, 0, 255 },
		{ 157, 1285, 260 },
		{ 157, 1288, 260 },
		{ 3099, 1358, 254 },
		{ 2774, 4700, 429 },
		{ 163, 0, 241 },
		{ 0, 0, 242 },
		{ -161, 4931, 237 },
		{ -162, 4744, 0 },
		{ 3054, 4722, 0 },
		{ 2774, 4709, 0 },
		{ 0, 0, 238 },
		{ 2774, 4724, 0 },
		{ -167, 21, 0 },
		{ -168, 4739, 0 },
		{ 171, 0, 239 },
		{ 2774, 4725, 0 },
		{ 3054, 4851, 0 },
		{ 0, 0, 240 },
		{ 3055, 1580, 138 },
		{ 2095, 4116, 138 },
		{ 2933, 4574, 138 },
		{ 3055, 4254, 138 },
		{ 0, 0, 138 },
		{ 3043, 3347, 0 },
		{ 2110, 2990, 0 },
		{ 3043, 3606, 0 },
		{ 3043, 3260, 0 },
		{ 3020, 3335, 0 },
		{ 2095, 4055, 0 },
		{ 3047, 3244, 0 },
		{ 2095, 4118, 0 },
		{ 3021, 3513, 0 },
		{ 3050, 3226, 0 },
		{ 3021, 3295, 0 },
		{ 2868, 4315, 0 },
		{ 3047, 3268, 0 },
		{ 2110, 3667, 0 },
		{ 2933, 4504, 0 },
		{ 2041, 3422, 0 },
		{ 2041, 3426, 0 },
		{ 2063, 3088, 0 },
		{ 2044, 3796, 0 },
		{ 2025, 3821, 0 },
		{ 2044, 3812, 0 },
		{ 2063, 3090, 0 },
		{ 2063, 3114, 0 },
		{ 3047, 3306, 0 },
		{ 2974, 3923, 0 },
		{ 2095, 4050, 0 },
		{ 1181, 4021, 0 },
		{ 2041, 3471, 0 },
		{ 2063, 3124, 0 },
		{ 2063, 3127, 0 },
		{ 2933, 4382, 0 },
		{ 2041, 3446, 0 },
		{ 3020, 3729, 0 },
		{ 3043, 3586, 0 },
		{ 2110, 3703, 0 },
		{ 2194, 4166, 0 },
		{ 2933, 3614, 0 },
		{ 2974, 3933, 0 },
		{ 2025, 3818, 0 },
		{ 3021, 3539, 0 },
		{ 2003, 3270, 0 },
		{ 2933, 4510, 0 },
		{ 3043, 3621, 0 },
		{ 2974, 3608, 0 },
		{ 2110, 3673, 0 },
		{ 2063, 3148, 0 },
		{ 3050, 3374, 0 },
		{ 3020, 3735, 0 },
		{ 2003, 3269, 0 },
		{ 2025, 3856, 0 },
		{ 2933, 4526, 0 },
		{ 2095, 4078, 0 },
		{ 2095, 4111, 0 },
		{ 3043, 3640, 0 },
		{ 2063, 3179, 0 },
		{ 2095, 4035, 0 },
		{ 3043, 3623, 0 },
		{ 2194, 4187, 0 },
		{ 2933, 4388, 0 },
		{ 3050, 3376, 0 },
		{ 3021, 3562, 0 },
		{ 2063, 3180, 0 },
		{ 3050, 3309, 0 },
		{ 3043, 3593, 0 },
		{ 1181, 4026, 0 },
		{ 2025, 3836, 0 },
		{ 2974, 3901, 0 },
		{ 2110, 3611, 0 },
		{ 1073, 3232, 0 },
		{ 3043, 3658, 0 },
		{ 3020, 3758, 0 },
		{ 2933, 4442, 0 },
		{ 2194, 4226, 0 },
		{ 2063, 3181, 0 },
		{ 2110, 3696, 0 },
		{ 3050, 3352, 0 },
		{ 2095, 4064, 0 },
		{ 2003, 3278, 0 },
		{ 2095, 4095, 0 },
		{ 3021, 3566, 0 },
		{ 2063, 3182, 0 },
		{ 2868, 4323, 0 },
		{ 3020, 3728, 0 },
		{ 3050, 3387, 0 },
		{ 2131, 3617, 0 },
		{ 2063, 3183, 0 },
		{ 2974, 3967, 0 },
		{ 2095, 4069, 0 },
		{ 2194, 4195, 0 },
		{ 2095, 4074, 0 },
		{ 2933, 4578, 0 },
		{ 2933, 4350, 0 },
		{ 2025, 3820, 0 },
		{ 2194, 4170, 0 },
		{ 2063, 3184, 0 },
		{ 2933, 4456, 0 },
		{ 2974, 3932, 0 },
		{ 2025, 3823, 0 },
		{ 2974, 3886, 0 },
		{ 2933, 4538, 0 },
		{ 2041, 3469, 0 },
		{ 3021, 3521, 0 },
		{ 2095, 4051, 0 },
		{ 2933, 4370, 0 },
		{ 1181, 4009, 0 },
		{ 2974, 3924, 0 },
		{ 1073, 3235, 0 },
		{ 2131, 3989, 0 },
		{ 2044, 3788, 0 },
		{ 3021, 3552, 0 },
		{ 2063, 3065, 0 },
		{ 2933, 4534, 0 },
		{ 2095, 4113, 0 },
		{ 2063, 3075, 0 },
		{ 0, 0, 70 },
		{ 3043, 3428, 0 },
		{ 3050, 3417, 0 },
		{ 2095, 4037, 0 },
		{ 3055, 4282, 0 },
		{ 2063, 3076, 0 },
		{ 2063, 3077, 0 },
		{ 3050, 3357, 0 },
		{ 2041, 3439, 0 },
		{ 2025, 3848, 0 },
		{ 3050, 3359, 0 },
		{ 2063, 3078, 0 },
		{ 2095, 4094, 0 },
		{ 3043, 3647, 0 },
		{ 3043, 3652, 0 },
		{ 2044, 3811, 0 },
		{ 3021, 3527, 0 },
		{ 847, 3244, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2041, 3472, 0 },
		{ 2933, 4430, 0 },
		{ 2974, 3966, 0 },
		{ 2025, 3869, 0 },
		{ 3050, 3386, 0 },
		{ 3021, 3563, 0 },
		{ 0, 0, 68 },
		{ 2063, 3079, 0 },
		{ 2041, 3430, 0 },
		{ 3050, 3388, 0 },
		{ 2974, 3959, 0 },
		{ 3050, 3395, 0 },
		{ 2933, 4594, 0 },
		{ 3021, 3534, 0 },
		{ 1181, 4008, 0 },
		{ 2025, 3865, 0 },
		{ 2041, 3466, 0 },
		{ 3020, 3769, 0 },
		{ 2095, 4124, 0 },
		{ 2933, 4454, 0 },
		{ 3055, 4258, 0 },
		{ 3021, 3541, 0 },
		{ 0, 0, 62 },
		{ 3021, 3550, 0 },
		{ 2933, 4522, 0 },
		{ 1181, 4018, 0 },
		{ 2974, 3955, 0 },
		{ 2095, 4052, 0 },
		{ 0, 0, 73 },
		{ 3050, 3409, 0 },
		{ 3043, 3598, 0 },
		{ 3043, 3637, 0 },
		{ 2063, 3081, 0 },
		{ 2974, 3918, 0 },
		{ 2095, 4092, 0 },
		{ 2063, 3082, 0 },
		{ 2041, 3486, 0 },
		{ 3020, 3739, 0 },
		{ 3050, 3418, 0 },
		{ 3021, 3515, 0 },
		{ 2933, 4498, 0 },
		{ 2063, 3083, 0 },
		{ 2974, 3963, 0 },
		{ 2095, 4032, 0 },
		{ 3050, 3331, 0 },
		{ 3021, 3529, 0 },
		{ 2974, 3895, 0 },
		{ 1015, 3975, 0 },
		{ 0, 0, 8 },
		{ 2041, 3434, 0 },
		{ 2044, 3806, 0 },
		{ 2974, 3919, 0 },
		{ 2076, 3215, 0 },
		{ 2063, 3084, 0 },
		{ 2194, 4189, 0 },
		{ 2095, 4077, 0 },
		{ 2041, 3463, 0 },
		{ 2095, 4079, 0 },
		{ 2095, 4085, 0 },
		{ 2044, 3794, 0 },
		{ 2063, 3086, 0 },
		{ 3050, 3360, 0 },
		{ 3050, 3266, 0 },
		{ 2095, 4112, 0 },
		{ 3047, 3307, 0 },
		{ 3021, 3568, 0 },
		{ 1181, 4012, 0 },
		{ 3020, 3782, 0 },
		{ 2063, 3087, 0 },
		{ 2110, 3059, 0 },
		{ 2933, 4362, 0 },
		{ 1181, 4027, 0 },
		{ 2110, 3679, 0 },
		{ 3055, 3216, 0 },
		{ 2076, 3213, 0 },
		{ 2044, 3789, 0 },
		{ 2041, 3432, 0 },
		{ 3050, 3408, 0 },
		{ 0, 0, 114 },
		{ 2063, 3089, 0 },
		{ 2110, 3690, 0 },
		{ 2002, 3240, 0 },
		{ 3043, 3612, 0 },
		{ 3020, 3748, 0 },
		{ 2933, 4530, 0 },
		{ 2095, 4089, 0 },
		{ 0, 0, 7 },
		{ 2044, 3792, 0 },
		{ 0, 0, 6 },
		{ 2974, 3911, 0 },
		{ 0, 0, 119 },
		{ 3020, 3753, 0 },
		{ 2095, 4102, 0 },
		{ 3055, 1638, 0 },
		{ 2063, 3091, 0 },
		{ 3020, 3776, 0 },
		{ 3043, 3632, 0 },
		{ 0, 0, 123 },
		{ 2063, 3092, 0 },
		{ 2095, 4122, 0 },
		{ 3055, 3230, 0 },
		{ 2095, 4126, 0 },
		{ 2194, 4172, 0 },
		{ 2110, 3701, 0 },
		{ 0, 0, 69 },
		{ 2025, 3824, 0 },
		{ 2063, 3095, 106 },
		{ 2063, 3096, 107 },
		{ 2933, 4524, 0 },
		{ 2974, 3893, 0 },
		{ 3021, 3571, 0 },
		{ 3043, 3660, 0 },
		{ 3021, 3575, 0 },
		{ 1181, 4006, 0 },
		{ 3043, 3617, 0 },
		{ 3021, 3511, 0 },
		{ 3047, 3318, 0 },
		{ 2110, 3718, 0 },
		{ 2974, 3930, 0 },
		{ 2095, 4087, 0 },
		{ 2063, 3097, 0 },
		{ 3050, 3362, 0 },
		{ 2933, 4432, 0 },
		{ 2974, 3935, 0 },
		{ 2868, 4314, 0 },
		{ 2974, 3952, 0 },
		{ 3021, 3522, 0 },
		{ 2974, 3956, 0 },
		{ 0, 0, 9 },
		{ 2063, 3098, 0 },
		{ 2974, 3961, 0 },
		{ 2076, 3217, 0 },
		{ 2131, 3988, 0 },
		{ 0, 0, 104 },
		{ 2041, 3437, 0 },
		{ 3020, 3774, 0 },
		{ 3043, 3595, 0 },
		{ 2110, 3614, 0 },
		{ 3020, 3236, 0 },
		{ 2974, 3902, 0 },
		{ 3047, 3272, 0 },
		{ 2974, 3913, 0 },
		{ 3047, 3264, 0 },
		{ 3043, 3627, 0 },
		{ 2095, 4060, 0 },
		{ 3050, 3389, 0 },
		{ 3021, 3555, 0 },
		{ 3047, 3254, 0 },
		{ 3020, 3764, 0 },
		{ 3050, 3396, 0 },
		{ 2974, 3885, 0 },
		{ 3050, 3310, 0 },
		{ 2933, 4508, 0 },
		{ 2063, 3099, 0 },
		{ 2933, 4516, 0 },
		{ 3021, 3569, 0 },
		{ 2095, 4090, 0 },
		{ 3043, 3588, 0 },
		{ 3043, 3590, 0 },
		{ 2025, 3851, 0 },
		{ 2041, 3420, 0 },
		{ 2063, 3100, 94 },
		{ 3021, 3502, 0 },
		{ 2063, 3103, 0 },
		{ 2063, 3104, 0 },
		{ 3047, 3302, 0 },
		{ 2110, 3681, 0 },
		{ 2194, 4185, 0 },
		{ 2063, 3105, 0 },
		{ 2041, 3436, 0 },
		{ 0, 0, 103 },
		{ 2194, 4191, 0 },
		{ 2933, 4440, 0 },
		{ 3050, 3340, 0 },
		{ 3050, 3349, 0 },
		{ 0, 0, 116 },
		{ 0, 0, 118 },
		{ 2110, 3717, 0 },
		{ 2041, 3445, 0 },
		{ 2095, 3343, 0 },
		{ 3050, 3354, 0 },
		{ 2095, 4053, 0 },
		{ 2063, 3107, 0 },
		{ 2095, 4059, 0 },
		{ 2974, 3949, 0 },
		{ 3020, 3756, 0 },
		{ 2063, 3108, 0 },
		{ 2131, 3983, 0 },
		{ 3047, 3300, 0 },
		{ 2194, 4183, 0 },
		{ 2063, 3109, 0 },
		{ 2933, 4584, 0 },
		{ 2041, 3485, 0 },
		{ 945, 3878, 0 },
		{ 3050, 3363, 0 },
		{ 3020, 3777, 0 },
		{ 2110, 3707, 0 },
		{ 2194, 4152, 0 },
		{ 3050, 3373, 0 },
		{ 2003, 3289, 0 },
		{ 2063, 3110, 0 },
		{ 2974, 3909, 0 },
		{ 3043, 3585, 0 },
		{ 2041, 3428, 0 },
		{ 2933, 4458, 0 },
		{ 3050, 3380, 0 },
		{ 2110, 3687, 0 },
		{ 2076, 3214, 0 },
		{ 3021, 3512, 0 },
		{ 2041, 3433, 0 },
		{ 2110, 3702, 0 },
		{ 2044, 3808, 0 },
		{ 3055, 3291, 0 },
		{ 2063, 3111, 0 },
		{ 0, 0, 63 },
		{ 2063, 3112, 0 },
		{ 3043, 3641, 0 },
		{ 3021, 3523, 0 },
		{ 3043, 3649, 0 },
		{ 3021, 3525, 0 },
		{ 2063, 3113, 108 },
		{ 2025, 3853, 0 },
		{ 2110, 3686, 0 },
		{ 2044, 3809, 0 },
		{ 2041, 3443, 0 },
		{ 2041, 3444, 0 },
		{ 2025, 3819, 0 },
		{ 2044, 3787, 0 },
		{ 2933, 4438, 0 },
		{ 3043, 3594, 0 },
		{ 3047, 3316, 0 },
		{ 2974, 3915, 0 },
		{ 3050, 3398, 0 },
		{ 2041, 3448, 0 },
		{ 0, 0, 120 },
		{ 2868, 4316, 0 },
		{ 2044, 3795, 0 },
		{ 3050, 3401, 0 },
		{ 3020, 3780, 0 },
		{ 0, 0, 115 },
		{ 0, 0, 105 },
		{ 2041, 3464, 0 },
		{ 3021, 3561, 0 },
		{ 3050, 3405, 0 },
		{ 2110, 2975, 0 },
		{ 3055, 3265, 0 },
		{ 2974, 3954, 0 },
		{ 3020, 3742, 0 },
		{ 2063, 3117, 0 },
		{ 2974, 3958, 0 },
		{ 2025, 3822, 0 },
		{ 3043, 3650, 0 },
		{ 2095, 4038, 0 },
		{ 2933, 4352, 0 },
		{ 2933, 4360, 0 },
		{ 2041, 3482, 0 },
		{ 2933, 4368, 0 },
		{ 2974, 3965, 0 },
		{ 2933, 4372, 0 },
		{ 3021, 3570, 0 },
		{ 3020, 3761, 0 },
		{ 2063, 3118, 0 },
		{ 2095, 4057, 0 },
		{ 3021, 3572, 0 },
		{ 2063, 3119, 0 },
		{ 3021, 3577, 0 },
		{ 2095, 4067, 0 },
		{ 2974, 3906, 0 },
		{ 2095, 4072, 0 },
		{ 3021, 3493, 0 },
		{ 2095, 4076, 0 },
		{ 2041, 3488, 0 },
		{ 3020, 3781, 0 },
		{ 2063, 3120, 0 },
		{ 3055, 4176, 0 },
		{ 2194, 4156, 0 },
		{ 2095, 4084, 0 },
		{ 2044, 3790, 0 },
		{ 2095, 4086, 0 },
		{ 2044, 3791, 0 },
		{ 3043, 3582, 0 },
		{ 2933, 4576, 0 },
		{ 3043, 3635, 0 },
		{ 0, 0, 96 },
		{ 2974, 3925, 0 },
		{ 2974, 3928, 0 },
		{ 2933, 4596, 0 },
		{ 2063, 3121, 0 },
		{ 2063, 3122, 0 },
		{ 2933, 4354, 0 },
		{ 2933, 4356, 0 },
		{ 2933, 4358, 0 },
		{ 2044, 3805, 0 },
		{ 2041, 3427, 0 },
		{ 0, 0, 125 },
		{ 0, 0, 117 },
		{ 0, 0, 121 },
		{ 2933, 4366, 0 },
		{ 2194, 4160, 0 },
		{ 1073, 3229, 0 },
		{ 2063, 3123, 0 },
		{ 2974, 3066, 0 },
		{ 3021, 3524, 0 },
		{ 2044, 3785, 0 },
		{ 1181, 4014, 0 },
		{ 2933, 4436, 0 },
		{ 3043, 3653, 0 },
		{ 3043, 3656, 0 },
		{ 3043, 3657, 0 },
		{ 3020, 3770, 0 },
		{ 2194, 4238, 0 },
		{ 2131, 3978, 0 },
		{ 3020, 3773, 0 },
		{ 3047, 3312, 0 },
		{ 2025, 3846, 0 },
		{ 1181, 4013, 0 },
		{ 3050, 3358, 0 },
		{ 2025, 3850, 0 },
		{ 2041, 3435, 0 },
		{ 2063, 3126, 0 },
		{ 2044, 3801, 0 },
		{ 2025, 3864, 0 },
		{ 2095, 4070, 0 },
		{ 2131, 3985, 0 },
		{ 2110, 3678, 0 },
		{ 3021, 3536, 0 },
		{ 2933, 4582, 0 },
		{ 2110, 3680, 0 },
		{ 0, 0, 57 },
		{ 0, 0, 58 },
		{ 2933, 4591, 0 },
		{ 3021, 3537, 0 },
		{ 0, 0, 67 },
		{ 0, 0, 112 },
		{ 2003, 3290, 0 },
		{ 3047, 3320, 0 },
		{ 2041, 3441, 0 },
		{ 3047, 3326, 0 },
		{ 3050, 3372, 0 },
		{ 2974, 3929, 0 },
		{ 3021, 3556, 0 },
		{ 0, 0, 98 },
		{ 3021, 3557, 0 },
		{ 0, 0, 100 },
		{ 3043, 3646, 0 },
		{ 3021, 3560, 0 },
		{ 3020, 3767, 0 },
		{ 2095, 4105, 0 },
		{ 2076, 3226, 0 },
		{ 2076, 3211, 0 },
		{ 3050, 3375, 0 },
		{ 1181, 4004, 0 },
		{ 2131, 3731, 0 },
		{ 3021, 3564, 0 },
		{ 945, 3876, 0 },
		{ 2025, 3816, 0 },
		{ 0, 0, 113 },
		{ 0, 0, 124 },
		{ 3021, 3565, 0 },
		{ 0, 0, 137 },
		{ 2041, 3449, 0 },
		{ 3055, 3893, 0 },
		{ 2933, 4466, 0 },
		{ 1181, 4020, 0 },
		{ 2933, 4500, 0 },
		{ 2095, 4045, 0 },
		{ 3055, 4276, 0 },
		{ 3021, 3567, 0 },
		{ 3055, 4285, 0 },
		{ 3047, 3303, 0 },
		{ 3047, 3305, 0 },
		{ 3020, 3062, 0 },
		{ 2063, 3131, 0 },
		{ 2095, 4056, 0 },
		{ 2974, 3903, 0 },
		{ 2933, 4570, 0 },
		{ 2933, 4572, 0 },
		{ 2974, 3905, 0 },
		{ 2110, 3697, 0 },
		{ 2974, 3908, 0 },
		{ 2110, 3700, 0 },
		{ 2110, 3615, 0 },
		{ 2974, 3912, 0 },
		{ 2063, 3132, 0 },
		{ 2194, 4242, 0 },
		{ 2063, 3133, 0 },
		{ 3043, 3630, 0 },
		{ 2063, 3137, 0 },
		{ 2044, 3798, 0 },
		{ 3043, 3633, 0 },
		{ 2025, 3866, 0 },
		{ 2974, 3927, 0 },
		{ 2063, 3138, 0 },
		{ 3043, 3636, 0 },
		{ 3055, 4278, 0 },
		{ 1181, 4015, 0 },
		{ 2110, 3677, 0 },
		{ 3021, 3494, 0 },
		{ 2933, 4390, 0 },
		{ 2933, 4426, 0 },
		{ 2095, 4093, 0 },
		{ 2044, 3810, 0 },
		{ 2194, 4240, 0 },
		{ 3021, 3501, 0 },
		{ 2095, 4097, 0 },
		{ 2095, 4099, 0 },
		{ 2974, 3940, 0 },
		{ 2974, 3943, 0 },
		{ 2095, 4107, 0 },
		{ 2933, 4462, 0 },
		{ 2933, 4464, 0 },
		{ 2095, 4109, 0 },
		{ 2063, 3139, 0 },
		{ 3050, 3399, 0 },
		{ 3050, 3400, 0 },
		{ 2095, 4114, 0 },
		{ 2025, 3835, 0 },
		{ 3047, 3298, 0 },
		{ 2025, 3845, 0 },
		{ 3055, 4287, 0 },
		{ 3050, 3403, 0 },
		{ 2063, 3141, 60 },
		{ 3050, 3407, 0 },
		{ 2933, 4536, 0 },
		{ 2110, 3699, 0 },
		{ 2063, 3143, 0 },
		{ 2063, 3144, 0 },
		{ 2131, 3986, 0 },
		{ 2974, 3968, 0 },
		{ 3055, 4256, 0 },
		{ 3020, 3746, 0 },
		{ 3050, 3410, 0 },
		{ 3050, 3411, 0 },
		{ 1073, 3230, 0 },
		{ 2025, 3871, 0 },
		{ 3021, 3531, 0 },
		{ 2076, 3216, 0 },
		{ 2003, 3256, 0 },
		{ 2003, 3262, 0 },
		{ 3043, 3624, 0 },
		{ 2041, 3440, 0 },
		{ 1181, 4028, 0 },
		{ 3047, 3313, 0 },
		{ 3021, 3544, 0 },
		{ 3055, 4306, 0 },
		{ 3055, 4251, 0 },
		{ 2095, 4075, 0 },
		{ 0, 0, 61 },
		{ 0, 0, 64 },
		{ 2933, 4384, 0 },
		{ 1181, 4007, 0 },
		{ 2025, 3829, 0 },
		{ 3021, 3545, 0 },
		{ 1181, 4010, 0 },
		{ 3021, 3548, 0 },
		{ 0, 0, 109 },
		{ 3050, 3333, 0 },
		{ 3050, 3339, 0 },
		{ 3021, 3554, 0 },
		{ 0, 0, 102 },
		{ 2063, 3145, 0 },
		{ 2933, 4452, 0 },
		{ 0, 0, 110 },
		{ 0, 0, 111 },
		{ 2110, 3698, 0 },
		{ 2025, 3852, 0 },
		{ 3020, 3734, 0 },
		{ 945, 3875, 0 },
		{ 2044, 3797, 0 },
		{ 1181, 4003, 0 },
		{ 3050, 3341, 0 },
		{ 0, 0, 3 },
		{ 2095, 4101, 0 },
		{ 3055, 4303, 0 },
		{ 2933, 4506, 0 },
		{ 3020, 3736, 0 },
		{ 2974, 3946, 0 },
		{ 3050, 3346, 0 },
		{ 2974, 3951, 0 },
		{ 2095, 4110, 0 },
		{ 2063, 3147, 0 },
		{ 2044, 3807, 0 },
		{ 2194, 4197, 0 },
		{ 2041, 3460, 0 },
		{ 2041, 3461, 0 },
		{ 2095, 4115, 0 },
		{ 3020, 3749, 0 },
		{ 1015, 3974, 0 },
		{ 2095, 3068, 0 },
		{ 2974, 3960, 0 },
		{ 2110, 3714, 0 },
		{ 0, 0, 71 },
		{ 2095, 4127, 0 },
		{ 0, 0, 79 },
		{ 2933, 4586, 0 },
		{ 2095, 4128, 0 },
		{ 2933, 4592, 0 },
		{ 3050, 3353, 0 },
		{ 2095, 4034, 0 },
		{ 3047, 3325, 0 },
		{ 3055, 4290, 0 },
		{ 2095, 4036, 0 },
		{ 2110, 3719, 0 },
		{ 2025, 3827, 0 },
		{ 3050, 3355, 0 },
		{ 2025, 3832, 0 },
		{ 2974, 3894, 0 },
		{ 2110, 3668, 0 },
		{ 2974, 3896, 0 },
		{ 2095, 4054, 0 },
		{ 0, 0, 66 },
		{ 2110, 3671, 0 },
		{ 2110, 3672, 0 },
		{ 2933, 4386, 0 },
		{ 2044, 3793, 0 },
		{ 3050, 3356, 0 },
		{ 3020, 3775, 0 },
		{ 2095, 4063, 0 },
		{ 2110, 3675, 0 },
		{ 2095, 4065, 0 },
		{ 2063, 3149, 0 },
		{ 2974, 3910, 0 },
		{ 3043, 3605, 0 },
		{ 2044, 3799, 0 },
		{ 2025, 3860, 0 },
		{ 2041, 3473, 0 },
		{ 3055, 4289, 0 },
		{ 2041, 3481, 0 },
		{ 2063, 3150, 0 },
		{ 2110, 3683, 0 },
		{ 2003, 3286, 0 },
		{ 3055, 4252, 0 },
		{ 2095, 4081, 0 },
		{ 2095, 4082, 0 },
		{ 847, 3246, 0 },
		{ 0, 3243, 0 },
		{ 3020, 3737, 0 },
		{ 2131, 3990, 0 },
		{ 2095, 4088, 0 },
		{ 3021, 3578, 0 },
		{ 1181, 4016, 0 },
		{ 2933, 4532, 0 },
		{ 3021, 3490, 0 },
		{ 2063, 3154, 0 },
		{ 3050, 3364, 0 },
		{ 3021, 3495, 0 },
		{ 2025, 3826, 0 },
		{ 2974, 3942, 0 },
		{ 3021, 3500, 0 },
		{ 3020, 3754, 0 },
		{ 3050, 3365, 0 },
		{ 2194, 4162, 0 },
		{ 2194, 4164, 0 },
		{ 2933, 4588, 0 },
		{ 2095, 4106, 0 },
		{ 0, 0, 65 },
		{ 2025, 3834, 0 },
		{ 3050, 3366, 0 },
		{ 3043, 3645, 0 },
		{ 3021, 3503, 0 },
		{ 3021, 3505, 0 },
		{ 3021, 3508, 0 },
		{ 3050, 3367, 0 },
		{ 2110, 3721, 0 },
		{ 2110, 3664, 0 },
		{ 0, 0, 129 },
		{ 0, 0, 130 },
		{ 2044, 3802, 0 },
		{ 1181, 4019, 0 },
		{ 3050, 3368, 0 },
		{ 2025, 3857, 0 },
		{ 2025, 3858, 0 },
		{ 0, 0, 10 },
		{ 2933, 4378, 0 },
		{ 2041, 3431, 0 },
		{ 3050, 3369, 0 },
		{ 2933, 4006, 0 },
		{ 3055, 4298, 0 },
		{ 3020, 3778, 0 },
		{ 2933, 4394, 0 },
		{ 2933, 4399, 0 },
		{ 3050, 3371, 0 },
		{ 2063, 3155, 0 },
		{ 2974, 3897, 0 },
		{ 2974, 3898, 0 },
		{ 2095, 4047, 0 },
		{ 2063, 3156, 0 },
		{ 3055, 4262, 0 },
		{ 2933, 4450, 0 },
		{ 3055, 4268, 0 },
		{ 2025, 3814, 0 },
		{ 0, 0, 80 },
		{ 2110, 3676, 0 },
		{ 2974, 3904, 0 },
		{ 0, 0, 78 },
		{ 3047, 3310, 0 },
		{ 2044, 3786, 0 },
		{ 0, 0, 81 },
		{ 3055, 4288, 0 },
		{ 2974, 3907, 0 },
		{ 3047, 3311, 0 },
		{ 2095, 4058, 0 },
		{ 2041, 3438, 0 },
		{ 3021, 3528, 0 },
		{ 2095, 4061, 0 },
		{ 2063, 3157, 0 },
		{ 3050, 3377, 0 },
		{ 0, 0, 59 },
		{ 0, 0, 97 },
		{ 0, 0, 99 },
		{ 2110, 3685, 0 },
		{ 2194, 4168, 0 },
		{ 3021, 3532, 0 },
		{ 2095, 4068, 0 },
		{ 2974, 3917, 0 },
		{ 3043, 3625, 0 },
		{ 2095, 4071, 0 },
		{ 0, 0, 135 },
		{ 3021, 3533, 0 },
		{ 2095, 4073, 0 },
		{ 2974, 3921, 0 },
		{ 3050, 3378, 0 },
		{ 3021, 3535, 0 },
		{ 2933, 4580, 0 },
		{ 2063, 3158, 0 },
		{ 2025, 3841, 0 },
		{ 2025, 3844, 0 },
		{ 2095, 4080, 0 },
		{ 2194, 4154, 0 },
		{ 3050, 3381, 0 },
		{ 3050, 3382, 0 },
		{ 3021, 3540, 0 },
		{ 2131, 3999, 0 },
		{ 0, 3877, 0 },
		{ 3050, 3383, 0 },
		{ 3050, 3384, 0 },
		{ 2974, 3941, 0 },
		{ 3043, 3644, 0 },
		{ 2110, 3709, 0 },
		{ 2933, 4364, 0 },
		{ 2974, 3945, 0 },
		{ 3050, 3385, 0 },
		{ 2110, 3715, 0 },
		{ 3055, 4293, 0 },
		{ 0, 0, 19 },
		{ 2041, 3450, 0 },
		{ 2041, 3458, 0 },
		{ 0, 0, 126 },
		{ 2041, 3459, 0 },
		{ 0, 0, 128 },
		{ 3021, 3553, 0 },
		{ 2095, 4104, 0 },
		{ 0, 0, 95 },
		{ 2063, 3159, 0 },
		{ 2025, 3867, 0 },
		{ 2025, 3868, 0 },
		{ 2063, 3160, 0 },
		{ 2933, 4428, 0 },
		{ 2041, 3462, 0 },
		{ 2110, 3670, 0 },
		{ 2974, 3964, 0 },
		{ 0, 0, 76 },
		{ 2025, 3815, 0 },
		{ 1181, 4022, 0 },
		{ 2063, 3161, 0 },
		{ 2025, 3817, 0 },
		{ 3021, 3559, 0 },
		{ 2095, 4119, 0 },
		{ 3055, 4291, 0 },
		{ 3055, 4292, 0 },
		{ 2933, 4460, 0 },
		{ 2095, 4121, 0 },
		{ 2974, 3970, 0 },
		{ 2974, 3892, 0 },
		{ 2063, 3162, 0 },
		{ 2041, 3465, 0 },
		{ 3050, 3390, 0 },
		{ 3020, 3741, 0 },
		{ 3050, 3391, 0 },
		{ 2041, 3470, 0 },
		{ 2974, 3900, 0 },
		{ 3020, 3747, 0 },
		{ 3050, 3392, 0 },
		{ 2095, 4044, 0 },
		{ 0, 0, 84 },
		{ 3055, 4281, 0 },
		{ 0, 0, 101 },
		{ 2025, 3828, 0 },
		{ 3055, 3890, 0 },
		{ 2095, 4049, 0 },
		{ 0, 0, 133 },
		{ 3050, 3393, 0 },
		{ 3050, 3394, 0 },
		{ 2063, 3163, 56 },
		{ 3020, 3755, 0 },
		{ 2110, 3684, 0 },
		{ 2025, 3840, 0 },
		{ 3047, 3296, 0 },
		{ 2868, 3853, 0 },
		{ 0, 0, 85 },
		{ 2041, 3484, 0 },
		{ 3055, 4309, 0 },
		{ 1015, 3972, 0 },
		{ 0, 3973, 0 },
		{ 3050, 3397, 0 },
		{ 3020, 3765, 0 },
		{ 3020, 3766, 0 },
		{ 2110, 3688, 0 },
		{ 3055, 4264, 0 },
		{ 2095, 4066, 0 },
		{ 2974, 3920, 0 },
		{ 2063, 3164, 0 },
		{ 2110, 3692, 0 },
		{ 2110, 3693, 0 },
		{ 2110, 3694, 0 },
		{ 0, 0, 89 },
		{ 2974, 3926, 0 },
		{ 2041, 3487, 0 },
		{ 3021, 3576, 0 },
		{ 0, 0, 122 },
		{ 2063, 3165, 0 },
		{ 3047, 3301, 0 },
		{ 2063, 3166, 0 },
		{ 3043, 3643, 0 },
		{ 2974, 3934, 0 },
		{ 2194, 4193, 0 },
		{ 2041, 3423, 0 },
		{ 3020, 3726, 0 },
		{ 0, 0, 91 },
		{ 3020, 3727, 0 },
		{ 2194, 4234, 0 },
		{ 2194, 4236, 0 },
		{ 3050, 3402, 0 },
		{ 0, 0, 15 },
		{ 2025, 3872, 0 },
		{ 0, 0, 17 },
		{ 0, 0, 18 },
		{ 2974, 3944, 0 },
		{ 2063, 3167, 0 },
		{ 2131, 3977, 0 },
		{ 3020, 3732, 0 },
		{ 2933, 4444, 0 },
		{ 3021, 3496, 0 },
		{ 2110, 3713, 0 },
		{ 1181, 4017, 0 },
		{ 3021, 3497, 0 },
		{ 3021, 3498, 0 },
		{ 3020, 3738, 0 },
		{ 2110, 3716, 0 },
		{ 0, 0, 55 },
		{ 2974, 3957, 0 },
		{ 3050, 3404, 0 },
		{ 2063, 3168, 0 },
		{ 3050, 3406, 0 },
		{ 2025, 3825, 0 },
		{ 2110, 3720, 0 },
		{ 2095, 4108, 0 },
		{ 0, 0, 74 },
		{ 2063, 3169, 0 },
		{ 3055, 4247, 0 },
		{ 3021, 3504, 0 },
		{ 0, 3228, 0 },
		{ 3021, 3506, 0 },
		{ 0, 0, 16 },
		{ 2110, 3669, 0 },
		{ 1181, 4011, 0 },
		{ 2063, 3170, 54 },
		{ 2063, 3171, 0 },
		{ 2025, 3839, 0 },
		{ 0, 0, 75 },
		{ 3020, 3759, 0 },
		{ 3055, 3285, 0 },
		{ 0, 0, 82 },
		{ 0, 0, 83 },
		{ 0, 0, 52 },
		{ 3020, 3763, 0 },
		{ 3043, 3619, 0 },
		{ 3043, 3620, 0 },
		{ 3050, 3412, 0 },
		{ 3043, 3622, 0 },
		{ 0, 0, 134 },
		{ 3020, 3768, 0 },
		{ 1181, 4023, 0 },
		{ 1181, 4024, 0 },
		{ 3050, 3413, 0 },
		{ 0, 0, 38 },
		{ 0, 0, 39 },
		{ 2063, 3172, 40 },
		{ 3020, 3771, 0 },
		{ 3055, 4302, 0 },
		{ 1181, 4029, 0 },
		{ 1181, 4002, 0 },
		{ 2025, 3855, 0 },
		{ 0, 0, 72 },
		{ 3020, 3772, 0 },
		{ 3050, 3415, 0 },
		{ 0, 0, 90 },
		{ 3050, 3416, 0 },
		{ 2025, 3859, 0 },
		{ 3043, 3628, 0 },
		{ 2025, 3861, 0 },
		{ 2041, 3442, 0 },
		{ 2974, 3914, 0 },
		{ 3047, 3314, 0 },
		{ 2974, 3916, 0 },
		{ 2131, 3997, 0 },
		{ 2063, 3173, 0 },
		{ 3050, 3330, 0 },
		{ 3055, 4283, 0 },
		{ 3047, 3317, 0 },
		{ 0, 0, 86 },
		{ 3055, 4286, 0 },
		{ 2063, 3174, 0 },
		{ 0, 0, 127 },
		{ 0, 0, 131 },
		{ 2025, 3870, 0 },
		{ 0, 0, 136 },
		{ 0, 0, 11 },
		{ 3020, 3783, 0 },
		{ 3020, 3725, 0 },
		{ 2110, 3691, 0 },
		{ 3043, 3639, 0 },
		{ 1181, 4025, 0 },
		{ 2063, 3175, 0 },
		{ 3050, 3337, 0 },
		{ 3020, 3731, 0 },
		{ 3050, 3338, 0 },
		{ 3055, 4308, 0 },
		{ 0, 0, 132 },
		{ 2974, 3931, 0 },
		{ 3055, 4310, 0 },
		{ 3020, 3733, 0 },
		{ 3047, 3322, 0 },
		{ 3047, 3324, 0 },
		{ 3055, 4253, 0 },
		{ 2063, 3176, 0 },
		{ 3055, 4255, 0 },
		{ 2974, 3936, 0 },
		{ 2933, 4502, 0 },
		{ 3050, 3344, 0 },
		{ 3050, 3345, 0 },
		{ 2063, 3177, 0 },
		{ 0, 0, 41 },
		{ 3020, 3740, 0 },
		{ 2933, 4512, 0 },
		{ 3055, 4270, 0 },
		{ 3050, 3347, 0 },
		{ 2110, 3704, 0 },
		{ 2025, 3830, 0 },
		{ 2974, 3947, 0 },
		{ 2974, 3948, 0 },
		{ 2868, 4322, 0 },
		{ 3055, 4284, 0 },
		{ 2025, 3831, 0 },
		{ 2933, 4543, 0 },
		{ 2974, 3950, 0 },
		{ 3020, 3743, 0 },
		{ 2025, 3833, 0 },
		{ 2110, 3705, 0 },
		{ 2110, 3706, 0 },
		{ 2095, 4100, 0 },
		{ 3050, 3348, 0 },
		{ 2025, 3837, 0 },
		{ 2025, 3838, 0 },
		{ 2110, 3708, 0 },
		{ 0, 0, 77 },
		{ 0, 0, 92 },
		{ 3020, 3750, 0 },
		{ 3020, 3751, 0 },
		{ 0, 4030, 0 },
		{ 2974, 3962, 0 },
		{ 0, 0, 87 },
		{ 2025, 3842, 0 },
		{ 3020, 3752, 0 },
		{ 2041, 3468, 0 },
		{ 0, 0, 12 },
		{ 2110, 3710, 0 },
		{ 2110, 3711, 0 },
		{ 0, 0, 88 },
		{ 0, 0, 53 },
		{ 0, 0, 93 },
		{ 3021, 3549, 0 },
		{ 3020, 3757, 0 },
		{ 2095, 4117, 0 },
		{ 0, 0, 14 },
		{ 2063, 3178, 0 },
		{ 3021, 3551, 0 },
		{ 2095, 4120, 0 },
		{ 3043, 3661, 0 },
		{ 2025, 3854, 0 },
		{ 2933, 4380, 0 },
		{ 3055, 4274, 0 },
		{ 2095, 4123, 0 },
		{ 2044, 3800, 0 },
		{ 2095, 4125, 0 },
		{ 3020, 3762, 0 },
		{ 3050, 3350, 0 },
		{ 0, 0, 13 },
		{ 3099, 1438, 229 },
		{ 0, 0, 230 },
		{ 3054, 4925, 231 },
		{ 3082, 1645, 235 },
		{ 1218, 2537, 236 },
		{ 0, 0, 236 },
		{ 3082, 1734, 232 },
		{ 1221, 1453, 0 },
		{ 3082, 1767, 233 },
		{ 1224, 1486, 0 },
		{ 1221, 0, 0 },
		{ 3008, 2656, 234 },
		{ 1226, 1520, 0 },
		{ 1224, 0, 0 },
		{ 3008, 2690, 232 },
		{ 1226, 0, 0 },
		{ 3008, 2700, 233 },
		{ 3047, 3321, 145 },
		{ 0, 0, 145 },
		{ 0, 0, 146 },
		{ 3060, 1991, 0 },
		{ 3082, 2862, 0 },
		{ 3099, 2064, 0 },
		{ 1234, 4756, 0 },
		{ 3079, 2579, 0 },
		{ 3082, 2792, 0 },
		{ 3094, 2938, 0 },
		{ 3090, 2469, 0 },
		{ 3093, 2999, 0 },
		{ 3099, 2098, 0 },
		{ 3093, 3034, 0 },
		{ 3095, 1742, 0 },
		{ 3000, 2690, 0 },
		{ 3097, 2181, 0 },
		{ 3051, 2236, 0 },
		{ 3060, 1976, 0 },
		{ 3100, 4402, 0 },
		{ 0, 0, 143 },
		{ 2868, 4321, 153 },
		{ 0, 0, 153 },
		{ 0, 0, 154 },
		{ 3082, 2822, 0 },
		{ 2946, 2749, 0 },
		{ 3097, 2189, 0 },
		{ 3099, 2038, 0 },
		{ 3082, 2818, 0 },
		{ 1257, 4814, 0 },
		{ 3082, 2509, 0 },
		{ 3064, 1464, 0 },
		{ 3082, 2900, 0 },
		{ 3099, 2065, 0 },
		{ 2712, 1446, 0 },
		{ 3095, 1917, 0 },
		{ 3089, 2731, 0 },
		{ 3000, 2537, 0 },
		{ 3051, 2296, 0 },
		{ 2902, 2742, 0 },
		{ 1268, 4786, 0 },
		{ 3082, 2511, 0 },
		{ 3090, 2455, 0 },
		{ 3060, 1968, 0 },
		{ 3082, 2844, 0 },
		{ 1273, 4776, 0 },
		{ 3092, 2438, 0 },
		{ 3087, 1601, 0 },
		{ 3051, 2291, 0 },
		{ 3094, 2936, 0 },
		{ 3095, 1859, 0 },
		{ 3000, 2673, 0 },
		{ 3097, 2160, 0 },
		{ 3051, 2246, 0 },
		{ 3100, 4528, 0 },
		{ 0, 0, 151 },
		{ 3047, 3319, 177 },
		{ 0, 0, 177 },
		{ 3060, 2000, 0 },
		{ 3082, 2899, 0 },
		{ 3099, 2055, 0 },
		{ 1289, 4814, 0 },
		{ 3094, 2617, 0 },
		{ 3090, 2417, 0 },
		{ 3093, 3014, 0 },
		{ 3060, 2011, 0 },
		{ 3060, 2027, 0 },
		{ 3082, 2841, 0 },
		{ 3060, 2030, 0 },
		{ 3100, 4544, 0 },
		{ 0, 0, 176 },
		{ 2131, 3996, 141 },
		{ 0, 0, 141 },
		{ 0, 0, 142 },
		{ 3082, 2859, 0 },
		{ 3051, 2248, 0 },
		{ 3097, 2197, 0 },
		{ 3084, 2362, 0 },
		{ 3082, 2787, 0 },
		{ 3055, 4294, 0 },
		{ 3090, 2427, 0 },
		{ 3093, 2997, 0 },
		{ 3060, 2035, 0 },
		{ 3060, 1966, 0 },
		{ 3062, 4669, 0 },
		{ 3062, 4665, 0 },
		{ 3000, 2676, 0 },
		{ 3051, 2305, 0 },
		{ 3000, 2707, 0 },
		{ 3095, 1867, 0 },
		{ 3000, 2568, 0 },
		{ 3093, 2996, 0 },
		{ 3090, 2451, 0 },
		{ 3000, 2674, 0 },
		{ 3060, 1528, 0 },
		{ 3082, 2790, 0 },
		{ 3099, 2078, 0 },
		{ 3100, 4532, 0 },
		{ 0, 0, 139 },
		{ 2618, 2963, 23 },
		{ 0, 0, 23 },
		{ 0, 0, 24 },
		{ 3082, 2807, 0 },
		{ 2902, 2734, 0 },
		{ 3000, 2701, 0 },
		{ 3051, 2277, 0 },
		{ 3054, 9, 0 },
		{ 3097, 2179, 0 },
		{ 2860, 2122, 0 },
		{ 3082, 2852, 0 },
		{ 3099, 2085, 0 },
		{ 3093, 3036, 0 },
		{ 3095, 1935, 0 },
		{ 3097, 2151, 0 },
		{ 3099, 2106, 0 },
		{ 3054, 7, 0 },
		{ 3079, 2928, 0 },
		{ 3082, 2907, 0 },
		{ 3060, 1999, 0 },
		{ 3094, 2937, 0 },
		{ 3099, 2042, 0 },
		{ 3000, 2711, 0 },
		{ 2860, 2142, 0 },
		{ 3095, 1684, 0 },
		{ 3000, 2627, 0 },
		{ 3097, 2219, 0 },
		{ 3051, 2303, 0 },
		{ 3054, 4903, 0 },
		{ 3062, 4649, 0 },
		{ 0, 0, 20 },
		{ 1372, 0, 1 },
		{ 1372, 0, 178 },
		{ 1372, 2757, 228 },
		{ 1587, 173, 228 },
		{ 1587, 411, 228 },
		{ 1587, 403, 228 },
		{ 1587, 526, 228 },
		{ 1587, 404, 228 },
		{ 1587, 415, 228 },
		{ 1587, 390, 228 },
		{ 1587, 421, 228 },
		{ 1587, 483, 228 },
		{ 1372, 0, 228 },
		{ 1384, 2498, 228 },
		{ 1372, 2782, 228 },
		{ 2618, 2961, 224 },
		{ 1587, 504, 228 },
		{ 1587, 502, 228 },
		{ 1587, 530, 228 },
		{ 1587, 0, 228 },
		{ 1587, 565, 228 },
		{ 1587, 552, 228 },
		{ 3097, 2161, 0 },
		{ 0, 0, 179 },
		{ 3051, 2301, 0 },
		{ 1587, 519, 0 },
		{ 1587, 0, 0 },
		{ 3054, 3940, 0 },
		{ 1587, 541, 0 },
		{ 1587, 583, 0 },
		{ 1587, 581, 0 },
		{ 1587, 588, 0 },
		{ 1587, 580, 0 },
		{ 1587, 610, 0 },
		{ 1587, 617, 0 },
		{ 1587, 601, 0 },
		{ 1587, 591, 0 },
		{ 1587, 583, 0 },
		{ 1587, 587, 0 },
		{ 3082, 2875, 0 },
		{ 3082, 2889, 0 },
		{ 1588, 597, 0 },
		{ 1588, 609, 0 },
		{ 1587, 619, 0 },
		{ 1587, 620, 0 },
		{ 1587, 611, 0 },
		{ 3097, 2183, 0 },
		{ 3079, 2921, 0 },
		{ 1587, 609, 0 },
		{ 1587, 653, 0 },
		{ 1587, 630, 0 },
		{ 1587, 657, 0 },
		{ 1587, 680, 0 },
		{ 1587, 681, 0 },
		{ 1587, 686, 0 },
		{ 1587, 680, 0 },
		{ 1587, 656, 0 },
		{ 1587, 683, 0 },
		{ 1587, 680, 0 },
		{ 1587, 695, 0 },
		{ 1587, 682, 0 },
		{ 3051, 2327, 0 },
		{ 2946, 2776, 0 },
		{ 1587, 698, 0 },
		{ 1587, 17, 0 },
		{ 1588, 31, 0 },
		{ 1587, 27, 0 },
		{ 1587, 31, 0 },
		{ 3090, 2463, 0 },
		{ 0, 0, 227 },
		{ 1587, 43, 0 },
		{ 1587, 26, 0 },
		{ 1587, 14, 0 },
		{ 1587, 65, 0 },
		{ 1587, 69, 0 },
		{ 1587, 70, 0 },
		{ 1587, 69, 0 },
		{ 1587, 57, 0 },
		{ 1587, 36, 0 },
		{ 1587, 41, 0 },
		{ 1587, 57, 0 },
		{ 1587, 0, 213 },
		{ 1587, 94, 0 },
		{ 3097, 2158, 0 },
		{ 3000, 2542, 0 },
		{ 1587, 65, 0 },
		{ 1587, 68, 0 },
		{ 1587, 64, 0 },
		{ 1587, 91, 0 },
		{ 1587, 93, 0 },
		{ -1467, 1092, 0 },
		{ 1588, 102, 0 },
		{ 1587, 164, 0 },
		{ 1587, 183, 0 },
		{ 1587, 175, 0 },
		{ 1587, 185, 0 },
		{ 1587, 186, 0 },
		{ 1587, 165, 0 },
		{ 1587, 181, 0 },
		{ 1587, 161, 0 },
		{ 1587, 152, 0 },
		{ 1587, 0, 212 },
		{ 1587, 159, 0 },
		{ 3084, 2382, 0 },
		{ 3051, 2280, 0 },
		{ 1587, 163, 0 },
		{ 1587, 173, 0 },
		{ 1587, 171, 0 },
		{ 1587, 0, 226 },
		{ 1587, 180, 0 },
		{ 0, 0, 214 },
		{ 1587, 173, 0 },
		{ 1589, 33, -4 },
		{ 1587, 201, 0 },
		{ 1587, 214, 0 },
		{ 1587, 273, 0 },
		{ 1587, 279, 0 },
		{ 1587, 214, 0 },
		{ 1587, 252, 0 },
		{ 1587, 225, 0 },
		{ 1587, 231, 0 },
		{ 1587, 263, 0 },
		{ 3082, 2833, 0 },
		{ 3082, 2836, 0 },
		{ 1587, 0, 216 },
		{ 1587, 303, 217 },
		{ 1587, 272, 0 },
		{ 1587, 275, 0 },
		{ 1587, 302, 0 },
		{ 1487, 3557, 0 },
		{ 3054, 4275, 0 },
		{ 2172, 4612, 203 },
		{ 1587, 305, 0 },
		{ 1587, 309, 0 },
		{ 1587, 313, 0 },
		{ 1587, 315, 0 },
		{ 1587, 319, 0 },
		{ 1587, 321, 0 },
		{ 1587, 322, 0 },
		{ 1587, 323, 0 },
		{ 1587, 306, 0 },
		{ 1587, 340, 0 },
		{ 1588, 328, 0 },
		{ 3055, 4295, 0 },
		{ 3054, 4, 219 },
		{ 1587, 332, 0 },
		{ 1587, 377, 0 },
		{ 1587, 359, 0 },
		{ 1587, 375, 0 },
		{ 0, 0, 183 },
		{ 1589, 117, -7 },
		{ 1589, 231, -10 },
		{ 1589, 345, -13 },
		{ 1589, 376, -16 },
		{ 1589, 460, -19 },
		{ 1589, 488, -22 },
		{ 1587, 407, 0 },
		{ 1587, 420, 0 },
		{ 1587, 393, 0 },
		{ 1587, 0, 201 },
		{ 1587, 0, 215 },
		{ 3090, 2431, 0 },
		{ 1587, 392, 0 },
		{ 1587, 383, 0 },
		{ 1587, 391, 0 },
		{ 1588, 392, 0 },
		{ 1524, 3525, 0 },
		{ 3054, 4307, 0 },
		{ 2172, 4628, 204 },
		{ 1527, 3527, 0 },
		{ 3054, 4271, 0 },
		{ 2172, 4643, 205 },
		{ 1530, 3528, 0 },
		{ 3054, 4279, 0 },
		{ 2172, 4631, 208 },
		{ 1533, 3529, 0 },
		{ 3054, 4195, 0 },
		{ 2172, 4624, 209 },
		{ 1536, 3530, 0 },
		{ 3054, 4269, 0 },
		{ 2172, 4633, 210 },
		{ 1539, 3531, 0 },
		{ 3054, 4273, 0 },
		{ 2172, 4619, 211 },
		{ 1587, 451, 0 },
		{ 1589, 490, -25 },
		{ 1587, 424, 0 },
		{ 3093, 2984, 0 },
		{ 1587, 437, 0 },
		{ 1587, 483, 0 },
		{ 1587, 481, 0 },
		{ 1587, 492, 0 },
		{ 0, 0, 185 },
		{ 0, 0, 187 },
		{ 0, 0, 193 },
		{ 0, 0, 195 },
		{ 0, 0, 197 },
		{ 0, 0, 199 },
		{ 1589, 574, -28 },
		{ 1557, 3541, 0 },
		{ 3054, 4283, 0 },
		{ 2172, 4638, 207 },
		{ 1587, 0, 200 },
		{ 3060, 2032, 0 },
		{ 1587, 480, 0 },
		{ 1587, 495, 0 },
		{ 1588, 488, 0 },
		{ 1587, 486, 0 },
		{ 1566, 3549, 0 },
		{ 3054, 4277, 0 },
		{ 2172, 4641, 206 },
		{ 0, 0, 191 },
		{ 3060, 1980, 0 },
		{ 1587, 4, 222 },
		{ 1588, 492, 0 },
		{ 1587, 1, 225 },
		{ 1587, 508, 0 },
		{ 0, 0, 189 },
		{ 3062, 4670, 0 },
		{ 3062, 4648, 0 },
		{ 1587, 496, 0 },
		{ 0, 0, 223 },
		{ 1587, 493, 0 },
		{ 3062, 4666, 0 },
		{ 0, 0, 221 },
		{ 1587, 504, 0 },
		{ 1587, 509, 0 },
		{ 0, 0, 220 },
		{ 1587, 514, 0 },
		{ 1587, 505, 0 },
		{ 1588, 507, 218 },
		{ 1589, 925, 0 },
		{ 1590, 736, -1 },
		{ 1591, 3503, 0 },
		{ 3054, 4239, 0 },
		{ 2172, 4636, 202 },
		{ 0, 0, 181 },
		{ 2131, 3998, 271 },
		{ 0, 0, 271 },
		{ 3082, 2892, 0 },
		{ 3051, 2330, 0 },
		{ 3097, 2215, 0 },
		{ 3084, 2392, 0 },
		{ 3082, 2784, 0 },
		{ 3055, 4297, 0 },
		{ 3090, 2439, 0 },
		{ 3093, 3007, 0 },
		{ 3060, 1994, 0 },
		{ 3060, 1995, 0 },
		{ 3062, 4655, 0 },
		{ 3062, 4657, 0 },
		{ 3000, 2497, 0 },
		{ 3051, 2274, 0 },
		{ 3000, 2539, 0 },
		{ 3095, 1920, 0 },
		{ 3000, 2556, 0 },
		{ 3093, 3001, 0 },
		{ 3090, 2415, 0 },
		{ 3000, 2575, 0 },
		{ 3060, 1533, 0 },
		{ 3082, 2849, 0 },
		{ 3099, 2093, 0 },
		{ 3100, 4540, 0 },
		{ 0, 0, 270 },
		{ 2131, 3992, 273 },
		{ 0, 0, 273 },
		{ 0, 0, 274 },
		{ 3082, 2856, 0 },
		{ 3051, 2290, 0 },
		{ 3097, 2162, 0 },
		{ 3084, 2386, 0 },
		{ 3082, 2876, 0 },
		{ 3055, 4280, 0 },
		{ 3090, 2453, 0 },
		{ 3093, 3032, 0 },
		{ 3060, 2007, 0 },
		{ 3060, 2008, 0 },
		{ 3062, 4659, 0 },
		{ 3062, 4662, 0 },
		{ 3094, 2952, 0 },
		{ 3099, 2108, 0 },
		{ 3097, 2186, 0 },
		{ 3060, 2009, 0 },
		{ 3060, 2010, 0 },
		{ 3097, 2205, 0 },
		{ 3064, 1493, 0 },
		{ 3082, 2795, 0 },
		{ 3099, 2056, 0 },
		{ 3100, 4548, 0 },
		{ 0, 0, 272 },
		{ 2131, 3982, 276 },
		{ 0, 0, 276 },
		{ 0, 0, 277 },
		{ 3082, 2808, 0 },
		{ 3051, 2261, 0 },
		{ 3097, 2148, 0 },
		{ 3084, 2383, 0 },
		{ 3082, 2823, 0 },
		{ 3055, 4307, 0 },
		{ 3090, 2467, 0 },
		{ 3093, 3005, 0 },
		{ 3060, 2015, 0 },
		{ 3060, 2023, 0 },
		{ 3062, 4667, 0 },
		{ 3062, 4668, 0 },
		{ 3084, 2393, 0 },
		{ 3087, 1553, 0 },
		{ 3095, 1743, 0 },
		{ 3093, 2982, 0 },
		{ 3095, 1796, 0 },
		{ 3097, 2170, 0 },
		{ 3099, 2094, 0 },
		{ 3100, 4431, 0 },
		{ 0, 0, 275 },
		{ 2131, 3987, 279 },
		{ 0, 0, 279 },
		{ 0, 0, 280 },
		{ 3082, 2865, 0 },
		{ 3051, 2302, 0 },
		{ 3097, 2180, 0 },
		{ 3084, 2366, 0 },
		{ 3082, 2891, 0 },
		{ 3055, 4279, 0 },
		{ 3090, 2468, 0 },
		{ 3093, 3033, 0 },
		{ 3060, 2033, 0 },
		{ 3060, 2034, 0 },
		{ 3062, 4663, 0 },
		{ 3062, 4664, 0 },
		{ 3082, 2904, 0 },
		{ 3064, 1495, 0 },
		{ 3093, 2992, 0 },
		{ 3090, 2425, 0 },
		{ 3087, 1539, 0 },
		{ 3093, 2998, 0 },
		{ 3095, 1908, 0 },
		{ 3097, 2202, 0 },
		{ 3099, 2043, 0 },
		{ 3100, 4464, 0 },
		{ 0, 0, 278 },
		{ 2131, 3995, 282 },
		{ 0, 0, 282 },
		{ 0, 0, 283 },
		{ 3082, 2798, 0 },
		{ 3051, 2264, 0 },
		{ 3097, 2209, 0 },
		{ 3084, 2384, 0 },
		{ 3082, 2814, 0 },
		{ 3055, 4305, 0 },
		{ 3090, 2466, 0 },
		{ 3093, 2981, 0 },
		{ 3060, 1969, 0 },
		{ 3060, 1972, 0 },
		{ 3062, 4650, 0 },
		{ 3062, 4653, 0 },
		{ 3097, 2220, 0 },
		{ 2860, 2128, 0 },
		{ 3095, 1910, 0 },
		{ 3000, 2625, 0 },
		{ 3084, 2370, 0 },
		{ 3000, 2670, 0 },
		{ 3060, 1977, 0 },
		{ 3082, 2858, 0 },
		{ 3099, 2071, 0 },
		{ 3100, 4610, 0 },
		{ 0, 0, 281 },
		{ 2933, 4471, 157 },
		{ 0, 0, 157 },
		{ 0, 0, 158 },
		{ 2946, 2752, 0 },
		{ 3095, 1916, 0 },
		{ 3082, 2870, 0 },
		{ 3099, 2080, 0 },
		{ 1731, 4733, 0 },
		{ 3082, 2504, 0 },
		{ 3064, 1498, 0 },
		{ 3082, 2890, 0 },
		{ 3099, 2089, 0 },
		{ 2712, 1457, 0 },
		{ 3095, 1926, 0 },
		{ 3089, 2722, 0 },
		{ 3000, 2499, 0 },
		{ 3051, 2244, 0 },
		{ 2902, 2736, 0 },
		{ 1742, 4768, 0 },
		{ 3082, 2498, 0 },
		{ 3090, 2475, 0 },
		{ 3060, 1998, 0 },
		{ 3082, 2794, 0 },
		{ 1747, 4812, 0 },
		{ 3092, 2440, 0 },
		{ 3087, 1628, 0 },
		{ 3051, 2256, 0 },
		{ 3094, 2956, 0 },
		{ 3095, 1937, 0 },
		{ 3000, 2629, 0 },
		{ 3097, 2193, 0 },
		{ 3051, 2270, 0 },
		{ 3100, 4400, 0 },
		{ 0, 0, 155 },
		{ 2131, 3991, 264 },
		{ 0, 0, 264 },
		{ 3082, 2819, 0 },
		{ 3051, 2271, 0 },
		{ 3097, 2196, 0 },
		{ 3084, 2374, 0 },
		{ 3082, 2834, 0 },
		{ 3055, 4245, 0 },
		{ 3090, 2465, 0 },
		{ 3093, 3017, 0 },
		{ 3060, 2001, 0 },
		{ 3060, 2002, 0 },
		{ 3062, 4660, 0 },
		{ 3062, 4661, 0 },
		{ 3079, 2930, 0 },
		{ 3000, 2709, 0 },
		{ 3060, 2005, 0 },
		{ 2860, 2130, 0 },
		{ 3090, 2470, 0 },
		{ 3093, 2988, 0 },
		{ 2712, 1413, 0 },
		{ 3100, 4396, 0 },
		{ 0, 0, 262 },
		{ 1795, 0, 1 },
		{ 1954, 2832, 379 },
		{ 3082, 2867, 379 },
		{ 3093, 2920, 379 },
		{ 3079, 2156, 379 },
		{ 1795, 0, 346 },
		{ 1795, 2626, 379 },
		{ 3089, 1599, 379 },
		{ 2868, 4317, 379 },
		{ 2110, 3722, 379 },
		{ 3047, 3295, 379 },
		{ 2110, 3666, 379 },
		{ 2095, 4103, 379 },
		{ 3099, 1960, 379 },
		{ 1795, 0, 379 },
		{ 2618, 2964, 377 },
		{ 3093, 2752, 379 },
		{ 3093, 2977, 379 },
		{ 0, 0, 379 },
		{ 3097, 2154, 0 },
		{ -1800, 9, 336 },
		{ -1801, 4745, 0 },
		{ 3051, 2312, 0 },
		{ 0, 0, 342 },
		{ 0, 0, 343 },
		{ 3090, 2447, 0 },
		{ 3000, 2628, 0 },
		{ 3082, 2901, 0 },
		{ 0, 0, 347 },
		{ 3051, 2318, 0 },
		{ 3099, 2066, 0 },
		{ 3000, 2672, 0 },
		{ 2063, 3085, 0 },
		{ 3043, 3626, 0 },
		{ 3050, 3335, 0 },
		{ 2003, 3268, 0 },
		{ 3043, 3629, 0 },
		{ 3087, 1626, 0 },
		{ 3060, 2013, 0 },
		{ 3051, 2239, 0 },
		{ 3095, 1802, 0 },
		{ 3099, 2083, 0 },
		{ 3097, 2171, 0 },
		{ 2774, 4712, 0 },
		{ 3097, 2172, 0 },
		{ 3060, 2021, 0 },
		{ 3095, 1843, 0 },
		{ 3051, 2268, 0 },
		{ 3079, 2913, 0 },
		{ 3099, 2092, 0 },
		{ 3090, 2421, 0 },
		{ 2131, 3979, 0 },
		{ 2063, 3101, 0 },
		{ 2063, 3102, 0 },
		{ 2095, 4048, 0 },
		{ 2025, 3862, 0 },
		{ 3082, 2820, 0 },
		{ 3060, 2026, 0 },
		{ 3079, 2910, 0 },
		{ 3087, 1627, 0 },
		{ 3082, 2826, 0 },
		{ 3090, 2429, 0 },
		{ 0, 4930, 339 },
		{ 3084, 2360, 0 },
		{ 3082, 2835, 0 },
		{ 2110, 3712, 0 },
		{ 3095, 1866, 0 },
		{ 0, 0, 378 },
		{ 3082, 2840, 0 },
		{ 3079, 2915, 0 },
		{ 2095, 4062, 0 },
		{ 2041, 3447, 0 },
		{ 3043, 3618, 0 },
		{ 3021, 3509, 0 },
		{ 2063, 3115, 0 },
		{ 0, 0, 367 },
		{ 3055, 4300, 0 },
		{ 3097, 2192, 0 },
		{ 3099, 2101, 0 },
		{ 3051, 2289, 0 },
		{ -1877, 1167, 0 },
		{ 0, 0, 338 },
		{ 3082, 2854, 0 },
		{ 0, 0, 366 },
		{ 2860, 2120, 0 },
		{ 3000, 2664, 0 },
		{ 3051, 2292, 0 },
		{ 1892, 4684, 0 },
		{ 3020, 3744, 0 },
		{ 2974, 3922, 0 },
		{ 3021, 3526, 0 },
		{ 2063, 3125, 0 },
		{ 3043, 3631, 0 },
		{ 3097, 2201, 0 },
		{ 3084, 2387, 0 },
		{ 3051, 2299, 0 },
		{ 3095, 1869, 0 },
		{ 0, 0, 368 },
		{ 3055, 4260, 345 },
		{ 3095, 1874, 0 },
		{ 3094, 2958, 0 },
		{ 3095, 1879, 0 },
		{ 0, 0, 371 },
		{ 0, 0, 372 },
		{ 1897, 0, -62 },
		{ 2076, 3222, 0 },
		{ 2110, 3682, 0 },
		{ 3043, 3642, 0 },
		{ 2095, 4091, 0 },
		{ 3000, 2695, 0 },
		{ 0, 0, 370 },
		{ 0, 0, 376 },
		{ 0, 4686, 0 },
		{ 3090, 2414, 0 },
		{ 3060, 2036, 0 },
		{ 3093, 3011, 0 },
		{ 2131, 3993, 0 },
		{ 3054, 4209, 0 },
		{ 2172, 4635, 361 },
		{ 2095, 4098, 0 },
		{ 2868, 4320, 0 },
		{ 3021, 3542, 0 },
		{ 3021, 3543, 0 },
		{ 3051, 2307, 0 },
		{ 0, 0, 373 },
		{ 0, 0, 374 },
		{ 3093, 3015, 0 },
		{ 2178, 1, 0 },
		{ 3090, 2419, 0 },
		{ 3082, 2894, 0 },
		{ 0, 0, 351 },
		{ 1920, 0, -65 },
		{ 1922, 0, -68 },
		{ 2110, 3695, 0 },
		{ 3055, 4296, 0 },
		{ 0, 0, 369 },
		{ 3060, 1963, 0 },
		{ 0, 0, 344 },
		{ 2131, 3980, 0 },
		{ 3051, 2314, 0 },
		{ 3054, 4191, 0 },
		{ 2172, 4618, 362 },
		{ 3054, 4193, 0 },
		{ 2172, 4622, 363 },
		{ 2868, 4319, 0 },
		{ 1932, 0, -50 },
		{ 3060, 1964, 0 },
		{ 3082, 2902, 0 },
		{ 3082, 2903, 0 },
		{ 0, 0, 353 },
		{ 0, 0, 355 },
		{ 1937, 0, -56 },
		{ 3054, 4235, 0 },
		{ 2172, 4640, 365 },
		{ 0, 0, 341 },
		{ 3051, 2323, 0 },
		{ 3099, 2063, 0 },
		{ 3054, 4263, 0 },
		{ 2172, 4614, 364 },
		{ 0, 0, 359 },
		{ 3097, 2150, 0 },
		{ 3093, 2991, 0 },
		{ 0, 0, 357 },
		{ 3084, 2391, 0 },
		{ 3095, 1901, 0 },
		{ 3082, 2791, 0 },
		{ 3000, 2572, 0 },
		{ 0, 0, 375 },
		{ 3097, 2153, 0 },
		{ 3051, 2241, 0 },
		{ 1951, 0, -71 },
		{ 3054, 4281, 0 },
		{ 2172, 4639, 360 },
		{ 0, 0, 349 },
		{ 1795, 2819, 379 },
		{ 1958, 2500, 379 },
		{ -1956, 22, 336 },
		{ -1957, 4742, 0 },
		{ 3054, 4729, 0 },
		{ 2774, 4697, 0 },
		{ 0, 0, 337 },
		{ 2774, 4715, 0 },
		{ -1962, 18, 0 },
		{ -1963, 4737, 0 },
		{ 1966, 2, 339 },
		{ 2774, 4716, 0 },
		{ 3054, 4898, 0 },
		{ 0, 0, 340 },
		{ 1984, 0, 1 },
		{ 2180, 2839, 335 },
		{ 3082, 2815, 335 },
		{ 1984, 0, 289 },
		{ 1984, 2733, 335 },
		{ 3043, 3634, 335 },
		{ 1984, 0, 292 },
		{ 3087, 1595, 335 },
		{ 2868, 4324, 335 },
		{ 2110, 3663, 335 },
		{ 3047, 3262, 335 },
		{ 2110, 3665, 335 },
		{ 2095, 4046, 335 },
		{ 3093, 2979, 335 },
		{ 3099, 1966, 335 },
		{ 1984, 0, 335 },
		{ 2618, 2962, 332 },
		{ 3093, 2985, 335 },
		{ 3079, 2926, 335 },
		{ 2868, 4313, 335 },
		{ 3093, 1547, 335 },
		{ 0, 0, 335 },
		{ 3097, 2165, 0 },
		{ -1991, 17, 284 },
		{ -1992, 4738, 0 },
		{ 3051, 2265, 0 },
		{ 0, 0, 290 },
		{ 3051, 2266, 0 },
		{ 3097, 2167, 0 },
		{ 3099, 2082, 0 },
		{ 2063, 3080, 0 },
		{ 3043, 3655, 0 },
		{ 3050, 3351, 0 },
		{ 3020, 3760, 0 },
		{ 0, 3238, 0 },
		{ 0, 3279, 0 },
		{ 3043, 3659, 0 },
		{ 3090, 2416, 0 },
		{ 3087, 1604, 0 },
		{ 3060, 1984, 0 },
		{ 3051, 2276, 0 },
		{ 3082, 2845, 0 },
		{ 3082, 2846, 0 },
		{ 3097, 2174, 0 },
		{ 2178, 4728, 0 },
		{ 3097, 2177, 0 },
		{ 2774, 4721, 0 },
		{ 3097, 2178, 0 },
		{ 3079, 2924, 0 },
		{ 2860, 2127, 0 },
		{ 3099, 2087, 0 },
		{ 2131, 3981, 0 },
		{ 2063, 3093, 0 },
		{ 2063, 3094, 0 },
		{ 2974, 3937, 0 },
		{ 2974, 3939, 0 },
		{ 2095, 4083, 0 },
		{ 0, 3863, 0 },
		{ 3060, 1988, 0 },
		{ 3082, 2861, 0 },
		{ 3060, 1989, 0 },
		{ 3079, 2912, 0 },
		{ 3051, 2294, 0 },
		{ 3060, 1990, 0 },
		{ 3090, 2458, 0 },
		{ 0, 0, 334 },
		{ 3090, 2460, 0 },
		{ 0, 0, 286 },
		{ 3084, 2390, 0 },
		{ 0, 0, 331 },
		{ 3087, 1621, 0 },
		{ 3082, 2883, 0 },
		{ 2095, 4096, 0 },
		{ 0, 3421, 0 },
		{ 3043, 3638, 0 },
		{ 2044, 3803, 0 },
		{ 0, 3804, 0 },
		{ 3021, 3547, 0 },
		{ 2063, 3106, 0 },
		{ 3082, 2887, 0 },
		{ 0, 0, 324 },
		{ 3055, 4304, 0 },
		{ 3097, 2190, 0 },
		{ 3095, 1932, 0 },
		{ 3095, 1933, 0 },
		{ 3087, 1623, 0 },
		{ -2071, 1242, 0 },
		{ 3082, 2896, 0 },
		{ 3090, 2472, 0 },
		{ 3051, 2308, 0 },
		{ 3020, 3745, 0 },
		{ 2974, 3969, 0 },
		{ 3021, 3558, 0 },
		{ 2974, 3883, 0 },
		{ 2974, 3891, 0 },
		{ 0, 3116, 0 },
		{ 3043, 3654, 0 },
		{ 0, 0, 323 },
		{ 3097, 2199, 0 },
		{ 3084, 2376, 0 },
		{ 3000, 2678, 0 },
		{ 0, 0, 330 },
		{ 3093, 3038, 0 },
		{ 0, 0, 325 },
		{ 0, 0, 288 },
		{ 3093, 3039, 0 },
		{ 3095, 1939, 0 },
		{ 2088, 0, -47 },
		{ 0, 3219, 0 },
		{ 2110, 3674, 0 },
		{ 2079, 3209, 0 },
		{ 2076, 3210, 0 },
		{ 3043, 3609, 0 },
		{ 2095, 4033, 0 },
		{ 3000, 2689, 0 },
		{ 0, 0, 327 },
		{ 3094, 2947, 0 },
		{ 3095, 1946, 0 },
		{ 3095, 1683, 0 },
		{ 2131, 3994, 0 },
		{ 3054, 4265, 0 },
		{ 2172, 4637, 314 },
		{ 2095, 4039, 0 },
		{ 2868, 4325, 0 },
		{ 2095, 4040, 0 },
		{ 2095, 4041, 0 },
		{ 2095, 4042, 0 },
		{ 0, 4043, 0 },
		{ 3021, 3573, 0 },
		{ 3021, 3574, 0 },
		{ 3051, 2325, 0 },
		{ 3093, 2987, 0 },
		{ 3000, 2702, 0 },
		{ 3000, 2703, 0 },
		{ 3082, 2793, 0 },
		{ 0, 0, 296 },
		{ 2117, 0, -74 },
		{ 2119, 0, -77 },
		{ 2121, 0, -35 },
		{ 2123, 0, -38 },
		{ 2125, 0, -41 },
		{ 2127, 0, -44 },
		{ 0, 3689, 0 },
		{ 3055, 4311, 0 },
		{ 0, 0, 326 },
		{ 3090, 2423, 0 },
		{ 3097, 2207, 0 },
		{ 3097, 2208, 0 },
		{ 3051, 2332, 0 },
		{ 3054, 4197, 0 },
		{ 2172, 4620, 315 },
		{ 3054, 4199, 0 },
		{ 2172, 4623, 316 },
		{ 3054, 4201, 0 },
		{ 2172, 4626, 319 },
		{ 3054, 4203, 0 },
		{ 2172, 4630, 320 },
		{ 3054, 4205, 0 },
		{ 2172, 4632, 321 },
		{ 3054, 4207, 0 },
		{ 2172, 4634, 322 },
		{ 2868, 4318, 0 },
		{ 2142, 0, -53 },
		{ 0, 3984, 0 },
		{ 3051, 2333, 0 },
		{ 3051, 2335, 0 },
		{ 3082, 2812, 0 },
		{ 0, 0, 298 },
		{ 0, 0, 300 },
		{ 0, 0, 306 },
		{ 0, 0, 308 },
		{ 0, 0, 310 },
		{ 0, 0, 312 },
		{ 2148, 0, -59 },
		{ 3054, 4237, 0 },
		{ 2172, 4610, 318 },
		{ 3082, 2813, 0 },
		{ 3093, 3013, 0 },
		{ 3029, 3203, 329 },
		{ 3099, 2044, 0 },
		{ 3054, 4267, 0 },
		{ 2172, 4621, 317 },
		{ 0, 0, 304 },
		{ 3051, 2237, 0 },
		{ 3099, 2054, 0 },
		{ 0, 0, 291 },
		{ 3093, 3026, 0 },
		{ 0, 0, 302 },
		{ 3097, 2216, 0 },
		{ 2712, 1460, 0 },
		{ 3095, 1689, 0 },
		{ 3084, 2380, 0 },
		{ 2933, 4434, 0 },
		{ 3000, 2577, 0 },
		{ 3082, 2828, 0 },
		{ 3090, 2464, 0 },
		{ 3097, 2222, 0 },
		{ 0, 0, 328 },
		{ 2902, 2737, 0 },
		{ 3051, 2254, 0 },
		{ 3097, 2147, 0 },
		{ 2171, 0, -80 },
		{ 3099, 2059, 0 },
		{ 3054, 4189, 0 },
		{ 0, 4608, 313 },
		{ 3000, 2663, 0 },
		{ 0, 0, 294 },
		{ 3095, 1691, 0 },
		{ 3089, 2719, 0 },
		{ 3084, 2389, 0 },
		{ 0, 4727, 0 },
		{ 0, 0, 333 },
		{ 1984, 2818, 335 },
		{ 2184, 2502, 335 },
		{ -2182, 10, 284 },
		{ -2183, 4736, 0 },
		{ 3054, 4728, 0 },
		{ 2774, 4711, 0 },
		{ 0, 0, 285 },
		{ 2774, 4717, 0 },
		{ -2188, 4926, 0 },
		{ -2189, 4740, 0 },
		{ 2192, 0, 286 },
		{ 2774, 4719, 0 },
		{ 3054, 4839, 0 },
		{ 0, 0, 287 },
		{ 0, 4158, 381 },
		{ 0, 0, 381 },
		{ 3082, 2857, 0 },
		{ 2946, 2765, 0 },
		{ 3093, 3003, 0 },
		{ 3087, 1630, 0 },
		{ 3090, 2409, 0 },
		{ 3095, 1756, 0 },
		{ 3054, 2, 0 },
		{ 3099, 2070, 0 },
		{ 3087, 1642, 0 },
		{ 3051, 2273, 0 },
		{ 2207, 4814, 0 },
		{ 3054, 1946, 0 },
		{ 3093, 3019, 0 },
		{ 3099, 2079, 0 },
		{ 3093, 3030, 0 },
		{ 3084, 2372, 0 },
		{ 3082, 2879, 0 },
		{ 3095, 1816, 0 },
		{ 3082, 2884, 0 },
		{ 3099, 2081, 0 },
		{ 3060, 2022, 0 },
		{ 3100, 4534, 0 },
		{ 0, 0, 380 },
		{ 2774, 4701, 429 },
		{ 0, 0, 386 },
		{ 0, 0, 388 },
		{ 2239, 830, 420 },
		{ 2409, 844, 420 },
		{ 2431, 842, 420 },
		{ 2376, 843, 420 },
		{ 2240, 857, 420 },
		{ 2238, 835, 420 },
		{ 2431, 841, 420 },
		{ 2261, 857, 420 },
		{ 2405, 859, 420 },
		{ 2405, 861, 420 },
		{ 2409, 858, 420 },
		{ 2351, 867, 420 },
		{ 2237, 885, 420 },
		{ 3082, 1679, 419 },
		{ 2269, 2545, 429 },
		{ 2465, 857, 420 },
		{ 2409, 870, 420 },
		{ 2272, 870, 420 },
		{ 2409, 864, 420 },
		{ 3082, 2796, 429 },
		{ -2242, 4927, 382 },
		{ -2243, 4741, 0 },
		{ 2465, 861, 420 },
		{ 2470, 469, 420 },
		{ 2465, 867, 420 },
		{ 2319, 865, 420 },
		{ 2409, 873, 420 },
		{ 2415, 868, 420 },
		{ 2409, 875, 420 },
		{ 2351, 884, 420 },
		{ 2323, 874, 420 },
		{ 2376, 865, 420 },
		{ 2351, 887, 420 },
		{ 2431, 871, 420 },
		{ 2237, 875, 420 },
		{ 2378, 882, 420 },
		{ 2237, 879, 420 },
		{ 2444, 890, 420 },
		{ 2415, 890, 420 },
		{ 2237, 900, 420 },
		{ 2444, 893, 420 },
		{ 2465, 1265, 420 },
		{ 2444, 921, 420 },
		{ 2323, 924, 420 },
		{ 2470, 471, 420 },
		{ 3082, 1777, 416 },
		{ 2298, 1487, 0 },
		{ 3082, 1800, 417 },
		{ 3051, 2283, 0 },
		{ 2774, 4718, 0 },
		{ 2237, 964, 420 },
		{ 3097, 1952, 0 },
		{ 2405, 962, 420 },
		{ 2309, 947, 420 },
		{ 2444, 965, 420 },
		{ 2378, 960, 420 },
		{ 2378, 961, 420 },
		{ 2323, 970, 420 },
		{ 2405, 978, 420 },
		{ 2405, 979, 420 },
		{ 2431, 967, 420 },
		{ 2376, 990, 420 },
		{ 2405, 1008, 420 },
		{ 2351, 1013, 420 },
		{ 2470, 484, 420 },
		{ 2470, 578, 420 },
		{ 2438, 998, 420 },
		{ 2438, 1035, 420 },
		{ 2405, 1050, 420 },
		{ 2309, 1035, 420 },
		{ 2415, 1042, 420 },
		{ 2351, 1057, 420 },
		{ 2394, 1055, 420 },
		{ 3092, 2448, 0 },
		{ 2327, 1407, 0 },
		{ 2298, 0, 0 },
		{ 3008, 2568, 418 },
		{ 2329, 1439, 0 },
		{ 3079, 2911, 0 },
		{ 0, 0, 384 },
		{ 2405, 1055, 420 },
		{ 2946, 2767, 0 },
		{ 2470, 580, 420 },
		{ 2323, 1076, 420 },
		{ 2378, 1069, 420 },
		{ 2470, 8, 420 },
		{ 2409, 1124, 420 },
		{ 2237, 1071, 420 },
		{ 2321, 1127, 420 },
		{ 2470, 123, 420 },
		{ 2378, 1111, 420 },
		{ 2409, 1123, 420 },
		{ 2470, 125, 420 },
		{ 2378, 1115, 420 },
		{ 3095, 1829, 0 },
		{ 3054, 2216, 0 },
		{ 2438, 1143, 420 },
		{ 2237, 1147, 420 },
		{ 2431, 1146, 420 },
		{ 2237, 1162, 420 },
		{ 2378, 1147, 420 },
		{ 2237, 1192, 420 },
		{ 2237, 1182, 420 },
		{ 3000, 2696, 0 },
		{ 2327, 0, 0 },
		{ 3008, 2602, 416 },
		{ 2329, 0, 0 },
		{ 3008, 2615, 417 },
		{ 0, 0, 421 },
		{ 2431, 1188, 420 },
		{ 2358, 4731, 0 },
		{ 3090, 2066, 0 },
		{ 2351, 1206, 420 },
		{ 2470, 128, 420 },
		{ 3060, 1887, 0 },
		{ 2493, 6, 420 },
		{ 2438, 1189, 420 },
		{ 2351, 1208, 420 },
		{ 2378, 1190, 420 },
		{ 3054, 1933, 0 },
		{ 2470, 130, 420 },
		{ 2376, 1216, 420 },
		{ 3097, 1991, 0 },
		{ 2409, 1230, 420 },
		{ 3051, 2317, 0 },
		{ 3099, 2053, 0 },
		{ 3051, 2319, 0 },
		{ 2415, 1225, 420 },
		{ 2431, 1224, 420 },
		{ 2237, 1269, 420 },
		{ 2405, 1266, 420 },
		{ 2405, 1267, 420 },
		{ 2470, 235, 420 },
		{ 2409, 1265, 420 },
		{ 3090, 2445, 0 },
		{ 2470, 237, 420 },
		{ 3092, 2172, 0 },
		{ 3000, 2687, 0 },
		{ 2378, 1255, 420 },
		{ 3060, 1893, 0 },
		{ 3095, 1695, 0 },
		{ 3100, 4394, 0 },
		{ 3054, 4795, 392 },
		{ 2465, 1263, 420 },
		{ 2378, 1257, 420 },
		{ 2409, 1274, 420 },
		{ 3097, 2168, 0 },
		{ 3092, 2458, 0 },
		{ 2409, 1271, 420 },
		{ 2946, 2770, 0 },
		{ 2415, 1266, 420 },
		{ 3000, 2712, 0 },
		{ 3082, 2789, 0 },
		{ 3000, 2713, 0 },
		{ 2237, 1260, 420 },
		{ 2409, 1275, 420 },
		{ 2237, 1265, 420 },
		{ 2470, 239, 420 },
		{ 2470, 241, 420 },
		{ 3099, 1914, 0 },
		{ 2444, 1273, 420 },
		{ 3082, 2805, 0 },
		{ 3097, 1962, 0 },
		{ 3043, 3648, 0 },
		{ 3000, 2570, 0 },
		{ 3084, 2364, 0 },
		{ 2409, 1279, 420 },
		{ 3095, 1906, 0 },
		{ 3093, 2971, 0 },
		{ 2493, 121, 420 },
		{ 2415, 1275, 420 },
		{ 2415, 1276, 420 },
		{ 2237, 1288, 420 },
		{ 2860, 2121, 0 },
		{ 3099, 2051, 0 },
		{ 2444, 1279, 420 },
		{ 2426, 4815, 0 },
		{ 2444, 1280, 420 },
		{ 3095, 1927, 0 },
		{ 3082, 2832, 0 },
		{ 3095, 1930, 0 },
		{ 2405, 1290, 420 },
		{ 2444, 1282, 420 },
		{ 2237, 1292, 420 },
		{ 3097, 1764, 0 },
		{ 3054, 2220, 0 },
		{ 3082, 2842, 0 },
		{ 2237, 1289, 420 },
		{ 3100, 4476, 0 },
		{ 2946, 2773, 0 },
		{ 3047, 3323, 0 },
		{ 3095, 1941, 0 },
		{ 3000, 2694, 0 },
		{ 2237, 1284, 420 },
		{ 3093, 3016, 0 },
		{ 3095, 1951, 0 },
		{ 3100, 4616, 0 },
		{ 0, 0, 410 },
		{ 2431, 1282, 420 },
		{ 2444, 1287, 420 },
		{ 2470, 243, 420 },
		{ 3087, 1659, 0 },
		{ 3097, 2157, 0 },
		{ 2432, 1296, 420 },
		{ 3054, 1938, 0 },
		{ 2470, 245, 420 },
		{ 2455, 4775, 0 },
		{ 2456, 4792, 0 },
		{ 2457, 4779, 0 },
		{ 2237, 1287, 420 },
		{ 2237, 1299, 420 },
		{ 2470, 349, 420 },
		{ 3093, 2978, 0 },
		{ 2946, 2754, 0 },
		{ 3060, 2014, 0 },
		{ 3079, 2909, 0 },
		{ 2237, 1289, 420 },
		{ 3054, 4922, 415 },
		{ 2466, 4692, 0 },
		{ 3060, 2016, 0 },
		{ 3051, 2247, 0 },
		{ 3095, 1823, 0 },
		{ 2237, 1295, 420 },
		{ 3095, 1848, 0 },
		{ 3060, 2024, 0 },
		{ 2470, 351, 420 },
		{ 2470, 355, 420 },
		{ 3054, 2333, 0 },
		{ 3090, 2456, 0 },
		{ 3084, 2378, 0 },
		{ 2470, 357, 420 },
		{ 3099, 2050, 0 },
		{ 3054, 1944, 0 },
		{ 3095, 1833, 0 },
		{ 3079, 2581, 0 },
		{ 3095, 1881, 0 },
		{ 2470, 370, 420 },
		{ 2470, 463, 420 },
		{ 3094, 1931, 0 },
		{ 3099, 2062, 0 },
		{ 2946, 2753, 0 },
		{ 3090, 2471, 0 },
		{ 3087, 1603, 0 },
		{ 2470, 1552, 420 },
		{ 3097, 1885, 0 },
		{ 2496, 4773, 0 },
		{ 3082, 2804, 0 },
		{ 3100, 4608, 0 },
		{ 2493, 576, 420 },
		{ 3060, 1970, 0 },
		{ 3100, 4392, 0 },
		{ 3054, 2335, 0 },
		{ 3097, 1983, 0 },
		{ 3082, 2810, 0 },
		{ 3093, 2980, 0 },
		{ 2506, 4791, 0 },
		{ 3097, 1762, 0 },
		{ 3097, 2211, 0 },
		{ 3099, 2072, 0 },
		{ 3099, 2075, 0 },
		{ 3082, 2816, 0 },
		{ 3099, 2077, 0 },
		{ 3054, 1942, 0 },
		{ 3060, 1889, 0 },
		{ 3060, 1981, 0 },
		{ 3051, 2313, 0 },
		{ 2518, 4679, 0 },
		{ 3082, 2824, 0 },
		{ 3060, 1982, 0 },
		{ 3093, 3000, 0 },
		{ 3094, 2943, 0 },
		{ 2523, 811, 420 },
		{ 3082, 2829, 0 },
		{ 2860, 2125, 0 },
		{ 3100, 4429, 0 },
		{ 3060, 1986, 0 },
		{ 3054, 4856, 390 },
		{ 3060, 1895, 0 },
		{ 3100, 4466, 0 },
		{ 3054, 4880, 399 },
		{ 3097, 2156, 0 },
		{ 2860, 2129, 0 },
		{ 3051, 2329, 0 },
		{ 3095, 1936, 0 },
		{ 3092, 2454, 0 },
		{ 3093, 3022, 0 },
		{ 2946, 2755, 0 },
		{ 2902, 2738, 0 },
		{ 3097, 2159, 0 },
		{ 3082, 2848, 0 },
		{ 2860, 2141, 0 },
		{ 3082, 2850, 0 },
		{ 3099, 2088, 0 },
		{ 3000, 2631, 0 },
		{ 3064, 1465, 0 },
		{ 3087, 1549, 0 },
		{ 3060, 1897, 0 },
		{ 3051, 2240, 0 },
		{ 2860, 2124, 0 },
		{ 3051, 2242, 0 },
		{ 3082, 2864, 0 },
		{ 3100, 4530, 0 },
		{ 3054, 4841, 413 },
		{ 3051, 2243, 0 },
		{ 3095, 1944, 0 },
		{ 0, 0, 426 },
		{ 3060, 1996, 0 },
		{ 3000, 2680, 0 },
		{ 3054, 4865, 398 },
		{ 3093, 2989, 0 },
		{ 3082, 2874, 0 },
		{ 3000, 2682, 0 },
		{ 3000, 2684, 0 },
		{ 3000, 2685, 0 },
		{ 3099, 2100, 0 },
		{ 2946, 2771, 0 },
		{ 2563, 4798, 0 },
		{ 2618, 2960, 0 },
		{ 3082, 2886, 0 },
		{ 3095, 1945, 0 },
		{ 3082, 2888, 0 },
		{ 3097, 2175, 0 },
		{ 2555, 1332, 0 },
		{ 2570, 4802, 0 },
		{ 2860, 2133, 0 },
		{ 3094, 2934, 0 },
		{ 3095, 1949, 0 },
		{ 3099, 2109, 0 },
		{ 3079, 2914, 0 },
		{ 2576, 4686, 0 },
		{ 3082, 2895, 0 },
		{ 3000, 2698, 0 },
		{ 2579, 4729, 0 },
		{ 0, 1379, 0 },
		{ 3090, 2437, 0 },
		{ 3099, 2041, 0 },
		{ 3095, 1682, 0 },
		{ 3097, 2188, 0 },
		{ 3090, 2449, 0 },
		{ 3082, 2905, 0 },
		{ 3060, 2004, 0 },
		{ 3054, 2754, 0 },
		{ 3093, 2976, 0 },
		{ 2590, 4772, 0 },
		{ 3089, 2718, 0 },
		{ 2592, 4784, 0 },
		{ 2618, 2965, 0 },
		{ 3082, 2788, 0 },
		{ 3060, 1899, 0 },
		{ 3090, 2454, 0 },
		{ 3099, 2048, 0 },
		{ 3060, 2006, 0 },
		{ 3000, 2540, 0 },
		{ 2600, 4792, 0 },
		{ 3097, 1971, 0 },
		{ 3099, 2052, 0 },
		{ 3084, 2385, 0 },
		{ 3094, 2705, 0 },
		{ 3082, 2801, 0 },
		{ 3100, 4614, 0 },
		{ 3093, 2993, 0 },
		{ 3097, 2198, 0 },
		{ 3051, 2285, 0 },
		{ 3082, 2806, 0 },
		{ 3051, 2286, 0 },
		{ 2860, 2132, 0 },
		{ 3087, 1620, 0 },
		{ 2618, 2966, 0 },
		{ 3079, 2583, 0 },
		{ 2616, 4768, 0 },
		{ 3079, 2576, 0 },
		{ 3093, 3008, 0 },
		{ 3100, 4503, 0 },
		{ 3095, 1690, 0 },
		{ 3097, 2203, 0 },
		{ 3000, 2660, 0 },
		{ 2623, 4741, 0 },
		{ 3051, 2293, 0 },
		{ 3084, 2009, 0 },
		{ 2860, 2119, 0 },
		{ 3093, 3018, 0 },
		{ 3000, 2669, 0 },
		{ 3093, 3021, 0 },
		{ 3100, 4612, 0 },
		{ 3054, 4872, 411 },
		{ 3095, 1692, 0 },
		{ 3099, 2057, 0 },
		{ 3100, 4620, 0 },
		{ 3100, 4390, 0 },
		{ 3095, 1693, 0 },
		{ 3099, 2061, 0 },
		{ 2946, 2768, 0 },
		{ 3000, 2675, 0 },
		{ 3082, 2830, 0 },
		{ 3100, 4404, 0 },
		{ 3082, 2831, 0 },
		{ 0, 2967, 0 },
		{ 3054, 4901, 397 },
		{ 3093, 2970, 0 },
		{ 3095, 1694, 0 },
		{ 2860, 2126, 0 },
		{ 3097, 1977, 0 },
		{ 2902, 2740, 0 },
		{ 3097, 2221, 0 },
		{ 3082, 2838, 0 },
		{ 3095, 1696, 0 },
		{ 3060, 2017, 0 },
		{ 3060, 2018, 0 },
		{ 3054, 4791, 391 },
		{ 3097, 2149, 0 },
		{ 3060, 2019, 0 },
		{ 3054, 4808, 403 },
		{ 3054, 4812, 404 },
		{ 3060, 2020, 0 },
		{ 3000, 2693, 0 },
		{ 2946, 2762, 0 },
		{ 3090, 2443, 0 },
		{ 2860, 2137, 0 },
		{ 0, 0, 425 },
		{ 2860, 2139, 0 },
		{ 3000, 2697, 0 },
		{ 3095, 1721, 0 },
		{ 2663, 4740, 0 },
		{ 3095, 1725, 0 },
		{ 2860, 2143, 0 },
		{ 2666, 4753, 0 },
		{ 3079, 2923, 0 },
		{ 3099, 2073, 0 },
		{ 3000, 2705, 0 },
		{ 3093, 3006, 0 },
		{ 3082, 2863, 0 },
		{ 3099, 2074, 0 },
		{ 3100, 4470, 0 },
		{ 3100, 4472, 0 },
		{ 3051, 2336, 0 },
		{ 3082, 2866, 0 },
		{ 3000, 2710, 0 },
		{ 3095, 1729, 0 },
		{ 3095, 1731, 0 },
		{ 3090, 2462, 0 },
		{ 3060, 2025, 0 },
		{ 3060, 1901, 0 },
		{ 3100, 4546, 0 },
		{ 3082, 2878, 0 },
		{ 3097, 1989, 0 },
		{ 3082, 2880, 0 },
		{ 3093, 3029, 0 },
		{ 3097, 2169, 0 },
		{ 3095, 1746, 0 },
		{ 3060, 2031, 0 },
		{ 3100, 4384, 0 },
		{ 3099, 939, 394 },
		{ 3054, 4780, 406 },
		{ 2902, 2741, 0 },
		{ 3099, 2084, 0 },
		{ 3095, 1776, 0 },
		{ 3089, 2726, 0 },
		{ 3089, 2729, 0 },
		{ 3000, 2569, 0 },
		{ 2698, 4711, 0 },
		{ 3094, 2950, 0 },
		{ 3054, 4810, 402 },
		{ 3099, 2086, 0 },
		{ 2860, 2134, 0 },
		{ 3090, 2476, 0 },
		{ 3095, 1784, 0 },
		{ 3051, 2262, 0 },
		{ 3000, 2626, 0 },
		{ 2706, 4777, 0 },
		{ 3054, 4827, 393 },
		{ 3100, 4501, 0 },
		{ 2708, 4781, 0 },
		{ 2712, 1458, 0 },
		{ 2710, 4783, 0 },
		{ 2711, 4785, 0 },
		{ 3095, 1799, 0 },
		{ 3092, 2446, 0 },
		{ 3099, 2091, 0 },
		{ 3093, 2990, 0 },
		{ 3082, 2783, 0 },
		{ 3100, 4542, 0 },
		{ 3097, 2184, 0 },
		{ 3060, 1957, 0 },
		{ 3097, 2187, 0 },
		{ 3100, 4573, 0 },
		{ 3054, 4870, 408 },
		{ 3100, 4600, 0 },
		{ 3100, 4602, 0 },
		{ 3100, 4604, 0 },
		{ 3100, 4606, 0 },
		{ 0, 1459, 0 },
		{ 3000, 2666, 0 },
		{ 3000, 2667, 0 },
		{ 3095, 1804, 0 },
		{ 3099, 2095, 0 },
		{ 3054, 4890, 414 },
		{ 3099, 2096, 0 },
		{ 3100, 4645, 0 },
		{ 3051, 2279, 0 },
		{ 0, 0, 428 },
		{ 0, 0, 427 },
		{ 3054, 4896, 395 },
		{ 0, 0, 423 },
		{ 0, 0, 424 },
		{ 3100, 4388, 0 },
		{ 3090, 2441, 0 },
		{ 2860, 2123, 0 },
		{ 3097, 2195, 0 },
		{ 3093, 3010, 0 },
		{ 3100, 4398, 0 },
		{ 3054, 4910, 389 },
		{ 2740, 4816, 0 },
		{ 3054, 4915, 396 },
		{ 3082, 2803, 0 },
		{ 3095, 1806, 0 },
		{ 3099, 2099, 0 },
		{ 3095, 1810, 0 },
		{ 3054, 4782, 409 },
		{ 3054, 2218, 0 },
		{ 3100, 4456, 0 },
		{ 3100, 4460, 0 },
		{ 3100, 4462, 0 },
		{ 3097, 2200, 0 },
		{ 3095, 1812, 0 },
		{ 3054, 4797, 400 },
		{ 3054, 4800, 401 },
		{ 3054, 4806, 405 },
		{ 3099, 2102, 0 },
		{ 3082, 2811, 0 },
		{ 3100, 4474, 0 },
		{ 3099, 2104, 0 },
		{ 3054, 4815, 407 },
		{ 3093, 3024, 0 },
		{ 3095, 1814, 0 },
		{ 3000, 2692, 0 },
		{ 3097, 2206, 0 },
		{ 3051, 2298, 0 },
		{ 3060, 1971, 0 },
		{ 3100, 4538, 0 },
		{ 3054, 4836, 412 },
		{ 2774, 4699, 429 },
		{ 2767, 0, 386 },
		{ 0, 0, 387 },
		{ -2765, 20, 382 },
		{ -2766, 4743, 0 },
		{ 3054, 4720, 0 },
		{ 2774, 4708, 0 },
		{ 0, 0, 383 },
		{ 2774, 4702, 0 },
		{ -2771, 4928, 0 },
		{ -2772, 4735, 0 },
		{ 2775, 0, 384 },
		{ 0, 4710, 0 },
		{ 3054, 4853, 0 },
		{ 0, 0, 385 },
		{ 3047, 3297, 149 },
		{ 0, 0, 149 },
		{ 0, 0, 150 },
		{ 3060, 1975, 0 },
		{ 3082, 2821, 0 },
		{ 3099, 2039, 0 },
		{ 2784, 4775, 0 },
		{ 3092, 2456, 0 },
		{ 3087, 1658, 0 },
		{ 3051, 2306, 0 },
		{ 3094, 2957, 0 },
		{ 3095, 1828, 0 },
		{ 3000, 2706, 0 },
		{ 3097, 2218, 0 },
		{ 3051, 2310, 0 },
		{ 3060, 1978, 0 },
		{ 3100, 4618, 0 },
		{ 0, 0, 147 },
		{ 2933, 4514, 172 },
		{ 0, 0, 172 },
		{ 3095, 1834, 0 },
		{ 2799, 4802, 0 },
		{ 3082, 2502, 0 },
		{ 3093, 2986, 0 },
		{ 3094, 2946, 0 },
		{ 3089, 2715, 0 },
		{ 2804, 4806, 0 },
		{ 3054, 2329, 0 },
		{ 3082, 2837, 0 },
		{ 3051, 2316, 0 },
		{ 3082, 2839, 0 },
		{ 3099, 2046, 0 },
		{ 3093, 2995, 0 },
		{ 3095, 1841, 0 },
		{ 3000, 2538, 0 },
		{ 3097, 2224, 0 },
		{ 3051, 2321, 0 },
		{ 2815, 4831, 0 },
		{ 3054, 2752, 0 },
		{ 3082, 2847, 0 },
		{ 2946, 2766, 0 },
		{ 3097, 2146, 0 },
		{ 3099, 2049, 0 },
		{ 3082, 2851, 0 },
		{ 2822, 4684, 0 },
		{ 3099, 1926, 0 },
		{ 3082, 2853, 0 },
		{ 3079, 2916, 0 },
		{ 3087, 1661, 0 },
		{ 3094, 2935, 0 },
		{ 3082, 2855, 0 },
		{ 2829, 4709, 0 },
		{ 3092, 2442, 0 },
		{ 3087, 1537, 0 },
		{ 3051, 2331, 0 },
		{ 3094, 2944, 0 },
		{ 3095, 1863, 0 },
		{ 3000, 2623, 0 },
		{ 3097, 2152, 0 },
		{ 3051, 2334, 0 },
		{ 3100, 4575, 0 },
		{ 0, 0, 170 },
		{ 2840, 0, 1 },
		{ -2840, 1326, 261 },
		{ 3082, 2761, 267 },
		{ 0, 0, 267 },
		{ 3060, 1992, 0 },
		{ 3051, 2238, 0 },
		{ 3082, 2873, 0 },
		{ 3079, 2922, 0 },
		{ 3099, 2058, 0 },
		{ 0, 0, 266 },
		{ 2850, 4777, 0 },
		{ 3084, 1928, 0 },
		{ 3093, 2975, 0 },
		{ 2879, 2480, 0 },
		{ 3082, 2877, 0 },
		{ 2946, 2769, 0 },
		{ 3000, 2665, 0 },
		{ 3090, 2459, 0 },
		{ 3082, 2882, 0 },
		{ 2859, 4759, 0 },
		{ 3097, 1997, 0 },
		{ 0, 2135, 0 },
		{ 3095, 1895, 0 },
		{ 3000, 2671, 0 },
		{ 3097, 2164, 0 },
		{ 3051, 2245, 0 },
		{ 3060, 1997, 0 },
		{ 3100, 4458, 0 },
		{ 0, 0, 265 },
		{ 0, 4326, 175 },
		{ 0, 0, 175 },
		{ 3097, 2166, 0 },
		{ 3087, 1602, 0 },
		{ 3051, 2250, 0 },
		{ 3079, 2925, 0 },
		{ 2875, 4799, 0 },
		{ 3094, 2630, 0 },
		{ 3089, 2717, 0 },
		{ 3082, 2897, 0 },
		{ 3094, 2951, 0 },
		{ 0, 2487, 0 },
		{ 3000, 2683, 0 },
		{ 3051, 2252, 0 },
		{ 2902, 2733, 0 },
		{ 3100, 4536, 0 },
		{ 0, 0, 173 },
		{ 2933, 4528, 169 },
		{ 0, 0, 168 },
		{ 0, 0, 169 },
		{ 3095, 1904, 0 },
		{ 2890, 4800, 0 },
		{ 3095, 1857, 0 },
		{ 3089, 2727, 0 },
		{ 3082, 2906, 0 },
		{ 2894, 4825, 0 },
		{ 3054, 2750, 0 },
		{ 3082, 2782, 0 },
		{ 2902, 2739, 0 },
		{ 3000, 2688, 0 },
		{ 3051, 2258, 0 },
		{ 3051, 2259, 0 },
		{ 3000, 2691, 0 },
		{ 3051, 2260, 0 },
		{ 0, 2735, 0 },
		{ 2904, 4685, 0 },
		{ 3097, 1957, 0 },
		{ 2946, 2757, 0 },
		{ 2907, 4698, 0 },
		{ 3082, 2500, 0 },
		{ 3093, 3025, 0 },
		{ 3094, 2933, 0 },
		{ 3089, 2725, 0 },
		{ 2912, 4705, 0 },
		{ 3054, 2327, 0 },
		{ 3082, 2797, 0 },
		{ 3051, 2263, 0 },
		{ 3082, 2799, 0 },
		{ 3099, 2069, 0 },
		{ 3093, 3037, 0 },
		{ 3095, 1909, 0 },
		{ 3000, 2700, 0 },
		{ 3097, 2173, 0 },
		{ 3051, 2267, 0 },
		{ 2923, 4728, 0 },
		{ 3092, 2450, 0 },
		{ 3087, 1605, 0 },
		{ 3051, 2269, 0 },
		{ 3094, 2953, 0 },
		{ 3095, 1911, 0 },
		{ 3000, 2708, 0 },
		{ 3097, 2176, 0 },
		{ 3051, 2272, 0 },
		{ 3100, 4468, 0 },
		{ 0, 0, 162 },
		{ 0, 4392, 161 },
		{ 0, 0, 161 },
		{ 3095, 1915, 0 },
		{ 2937, 4737, 0 },
		{ 3095, 1861, 0 },
		{ 3089, 2716, 0 },
		{ 3082, 2817, 0 },
		{ 2941, 4756, 0 },
		{ 3082, 2506, 0 },
		{ 3051, 2275, 0 },
		{ 3079, 2918, 0 },
		{ 2945, 4747, 0 },
		{ 3097, 1959, 0 },
		{ 0, 2761, 0 },
		{ 2948, 4766, 0 },
		{ 3082, 2496, 0 },
		{ 3093, 2994, 0 },
		{ 3094, 2948, 0 },
		{ 3089, 2721, 0 },
		{ 2953, 4773, 0 },
		{ 3054, 2331, 0 },
		{ 3082, 2825, 0 },
		{ 3051, 2278, 0 },
		{ 3082, 2827, 0 },
		{ 3099, 2076, 0 },
		{ 3093, 3002, 0 },
		{ 3095, 1919, 0 },
		{ 3000, 2543, 0 },
		{ 3097, 2182, 0 },
		{ 3051, 2282, 0 },
		{ 2964, 4792, 0 },
		{ 3092, 2444, 0 },
		{ 3087, 1622, 0 },
		{ 3051, 2284, 0 },
		{ 3094, 2939, 0 },
		{ 3095, 1922, 0 },
		{ 3000, 2573, 0 },
		{ 3097, 2185, 0 },
		{ 3051, 2287, 0 },
		{ 3100, 4386, 0 },
		{ 0, 0, 159 },
		{ 0, 3953, 164 },
		{ 0, 0, 164 },
		{ 0, 0, 165 },
		{ 3051, 2288, 0 },
		{ 3060, 2012, 0 },
		{ 3095, 1923, 0 },
		{ 3082, 2843, 0 },
		{ 3093, 3023, 0 },
		{ 3079, 2927, 0 },
		{ 2984, 4817, 0 },
		{ 3082, 2494, 0 },
		{ 3064, 1496, 0 },
		{ 3093, 3028, 0 },
		{ 3090, 2474, 0 },
		{ 3087, 1624, 0 },
		{ 3093, 3031, 0 },
		{ 3095, 1928, 0 },
		{ 3000, 2661, 0 },
		{ 3097, 2191, 0 },
		{ 3051, 2295, 0 },
		{ 2995, 4691, 0 },
		{ 3092, 2452, 0 },
		{ 3087, 1625, 0 },
		{ 3051, 2297, 0 },
		{ 3094, 2941, 0 },
		{ 3095, 1931, 0 },
		{ 0, 2668, 0 },
		{ 3097, 2194, 0 },
		{ 3051, 2300, 0 },
		{ 3062, 4647, 0 },
		{ 0, 0, 163 },
		{ 3082, 2860, 429 },
		{ 3099, 1520, 25 },
		{ 3014, 0, 429 },
		{ 2236, 2646, 27 },
		{ 0, 0, 28 },
		{ 0, 0, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 3051, 2304, 0 },
		{ 3099, 661, 0 },
		{ 0, 0, 26 },
		{ 3079, 2931, 0 },
		{ 0, 0, 21 },
		{ 0, 0, 32 },
		{ 0, 0, 33 },
		{ 0, 3779, 37 },
		{ 0, 3510, 37 },
		{ 0, 0, 36 },
		{ 0, 0, 37 },
		{ 3043, 3651, 0 },
		{ 3055, 4299, 0 },
		{ 3047, 3299, 0 },
		{ 0, 0, 35 },
		{ 3050, 3361, 0 },
		{ 0, 3205, 0 },
		{ 3008, 1633, 0 },
		{ 0, 0, 34 },
		{ 3082, 2776, 47 },
		{ 0, 0, 47 },
		{ 3047, 3304, 47 },
		{ 3082, 2871, 47 },
		{ 0, 0, 50 },
		{ 3082, 2872, 0 },
		{ 3051, 2309, 0 },
		{ 3050, 3370, 0 },
		{ 3095, 1943, 0 },
		{ 3051, 2311, 0 },
		{ 3079, 2920, 0 },
		{ 0, 3615, 0 },
		{ 3087, 1660, 0 },
		{ 3097, 2204, 0 },
		{ 0, 0, 46 },
		{ 0, 3315, 0 },
		{ 3099, 2097, 0 },
		{ 3084, 2368, 0 },
		{ 0, 3379, 0 },
		{ 0, 2315, 0 },
		{ 3082, 2881, 0 },
		{ 0, 0, 48 },
		{ 0, 5, 51 },
		{ 0, 4266, 0 },
		{ 0, 0, 49 },
		{ 3090, 2457, 0 },
		{ 3093, 3004, 0 },
		{ 3060, 2028, 0 },
		{ 0, 2029, 0 },
		{ 3062, 4651, 0 },
		{ 0, 4652, 0 },
		{ 3082, 2885, 0 },
		{ 0, 1534, 0 },
		{ 3093, 3009, 0 },
		{ 3090, 2461, 0 },
		{ 3087, 1676, 0 },
		{ 3093, 3012, 0 },
		{ 3095, 1948, 0 },
		{ 3097, 2213, 0 },
		{ 3099, 2103, 0 },
		{ 3093, 2057, 0 },
		{ 3082, 2893, 0 },
		{ 3097, 2217, 0 },
		{ 3094, 2940, 0 },
		{ 3093, 3020, 0 },
		{ 3099, 2105, 0 },
		{ 3094, 2942, 0 },
		{ 0, 2919, 0 },
		{ 3082, 2800, 0 },
		{ 3087, 1678, 0 },
		{ 0, 2898, 0 },
		{ 3093, 3027, 0 },
		{ 0, 2388, 0 },
		{ 3099, 2107, 0 },
		{ 3094, 2949, 0 },
		{ 0, 1680, 0 },
		{ 3100, 4654, 0 },
		{ 0, 2723, 0 },
		{ 0, 2473, 0 },
		{ 0, 0, 43 },
		{ 3054, 2726, 0 },
		{ 0, 3035, 0 },
		{ 0, 2954, 0 },
		{ 0, 1953, 0 },
		{ 3100, 4656, 0 },
		{ 0, 2223, 0 },
		{ 0, 0, 44 },
		{ 0, 2110, 0 },
		{ 3062, 4658, 0 },
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
		0
	};
	yybackup = backup;
}
