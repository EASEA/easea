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

#line 464 "EaseaLex.cpp"
		}
		break;
	case 27:
		{
#line 332 "EaseaLex.l"

  if( iGP_OPCODE_FIELD != 1 ){
    fprintf(stderr,"Error, op code real name must be given at the second place\n");
    exit(-1);
  }
  opDesc[iNoOp]->realName = new string(yytext);
 
#line 477 "EaseaLex.cpp"
		}
		break;
	case 28:
		{
#line 341 "EaseaLex.l"

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
 
#line 496 "EaseaLex.cpp"
		}
		break;
	case 29:
		{
#line 355 "EaseaLex.l"

#line 503 "EaseaLex.cpp"
		}
		break;
	case 30:
		{
#line 356 "EaseaLex.l"

  iGP_OPCODE_FIELD = 0;
  iNoOp++;
 
#line 513 "EaseaLex.cpp"
		}
		break;
	case 31:
		{
#line 361 "EaseaLex.l"

  if( bGPOPCODE_ANALYSIS ) iGP_OPCODE_FIELD++;
 
#line 522 "EaseaLex.cpp"
		}
		break;
	case 32:
		{
#line 366 "EaseaLex.l"

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
 
#line 544 "EaseaLex.cpp"
		}
		break;
	case 33:
		{
#line 387 "EaseaLex.l"

  accolade_counter++;
  opDesc[iNoOp]->cpuCodeStream << "{";
  opDesc[iNoOp]->gpuCodeStream << "{";
 
#line 555 "EaseaLex.cpp"
		}
		break;
	case 34:
		{
#line 393 "EaseaLex.l"

  accolade_counter--;
  if( accolade_counter==0 ){
    opDesc[iNoOp]->gpuCodeStream << "\n      stack[sp++] = RESULT;\n";

    BEGIN GP_RULE_ANALYSIS;
  }
  else{
    opDesc[iNoOp]->cpuCodeStream << "}";
    opDesc[iNoOp]->gpuCodeStream << "}";
  }
 
#line 573 "EaseaLex.cpp"
		}
		break;
	case 35:
		{
#line 406 "EaseaLex.l"

  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
//  printf("input no : %d\n",no_input);
  opDesc[iNoOp]->cpuCodeStream << "input["<< no_input <<"]" ;
  opDesc[iNoOp]->gpuCodeStream << "input["<< no_input << "]";  
 
#line 586 "EaseaLex.cpp"
		}
		break;
	case 36:
		{
#line 414 "EaseaLex.l"

  opDesc[iNoOp]->isERC = true;
  opDesc[iNoOp]->cpuCodeStream << "root->erc_value" ;
  opDesc[iNoOp]->gpuCodeStream << "k_progs[start_prog++];" ;
//  printf("ERC matched\n");

#line 598 "EaseaLex.cpp"
		}
		break;
	case 37:
		{
#line 421 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << "\n  ";
  opDesc[iNoOp]->gpuCodeStream << "\n    ";
 
#line 608 "EaseaLex.cpp"
		}
		break;
	case 38:
		{
#line 427 "EaseaLex.l"

  opDesc[iNoOp]->cpuCodeStream << yytext;
  opDesc[iNoOp]->gpuCodeStream << yytext;
 
#line 618 "EaseaLex.cpp"
		}
		break;
	case 39:
		{
#line 432 "EaseaLex.l"
 
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  if( bVERBOSE ) printf("Insert GP eval header\n");
  iCOPY_GP_EVAL_STATUS = EVAL_HDR;
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 634 "EaseaLex.cpp"
		}
		break;
	case 40:
		{
#line 443 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = false;
  BEGIN COPY_GP_EVAL;
 
#line 651 "EaseaLex.cpp"
		}
		break;
	case 41:
		{
#line 457 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 668 "EaseaLex.cpp"
		}
		break;
	case 42:
		{
#line 471 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 
#line 684 "EaseaLex.cpp"
		}
		break;
	case 43:
		{
#line 482 "EaseaLex.l"

  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 
#line 701 "EaseaLex.cpp"
		}
		break;
	case 44:
		{
#line 495 "EaseaLex.l"

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
 
#line 720 "EaseaLex.cpp"
		}
		break;
	case 45:
		{
#line 510 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_HDR){
    bIsCopyingGPEval = true;
  }
 
#line 731 "EaseaLex.cpp"
		}
		break;
	case 46:
		{
#line 516 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_BDY){
    bIsCopyingGPEval = true;
  }
 
#line 742 "EaseaLex.cpp"
		}
		break;
	case 47:
		{
#line 524 "EaseaLex.l"

  if( iCOPY_GP_EVAL_STATUS==EVAL_FTR){
    bIsCopyingGPEval = true;
  }
 
#line 753 "EaseaLex.cpp"
		}
		break;
	case 48:
		{
#line 530 "EaseaLex.l"

  if( bIsCopyingGPEval ){
    bIsCopyingGPEval = false;
    bCOPY_GP_EVAL_GPU = false;
    rewind(fpGenomeFile);
    yyin = fpTemplateFile;
    BEGIN TEMPLATE_ANALYSIS;
  }
 
#line 768 "EaseaLex.cpp"
		}
		break;
	case 49:
		{
#line 540 "EaseaLex.l"

  if( bIsCopyingGPEval ) fprintf(fpOutputFile,"%s",yytext);
 
#line 777 "EaseaLex.cpp"
		}
		break;
	case 50:
		{
#line 544 "EaseaLex.l"

  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[i*NUMTHREAD2+tid]" );
    else fprintf(fpOutputFile, "outputs[i]" );
  
 
#line 790 "EaseaLex.cpp"
		}
		break;
	case 51:
		{
#line 552 "EaseaLex.l"

  if( bIsCopyingGPEval )
    if( iCOPY_GP_EVAL_STATUS==EVAL_FTR )
      if( bCOPY_GP_EVAL_GPU ){
	fprintf(fpOutputFile,"k_results[index] =");
      }
      else fprintf(fpOutputFile,"%s",yytext);
 
#line 804 "EaseaLex.cpp"
		}
		break;
	case 52:
		{
#line 564 "EaseaLex.l"

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
 
#line 822 "EaseaLex.cpp"
		}
		break;
	case 53:
		{
#line 578 "EaseaLex.l"

  if( bIsCopyingGPEval )
    fprintf(fpOutputFile,"return fitness = "); 
 
#line 832 "EaseaLex.cpp"
		}
		break;
	case 54:
		{
#line 585 "EaseaLex.l"

  //DEBUG_PRT_PRT("insert beg");
  yyreset();
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Evaluation population in a single function!!.\n");
  lineCounter = 1;
  BEGIN COPY_INSTEAD_EVAL;
 
#line 846 "EaseaLex.cpp"
		}
		break;
	case 55:
		{
#line 594 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting at the end of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bEndGeneration = true;
  bBeginGeneration = false;
  BEGIN COPY_END_GENERATION_FUNCTION;
 
#line 860 "EaseaLex.cpp"
		}
		break;
	case 56:
		{
#line 603 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Bound Checking function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_BOUND_CHECKING_FUNCTION;
 
#line 872 "EaseaLex.cpp"
		}
		break;
	case 57:
		{
#line 610 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 
#line 884 "EaseaLex.cpp"
		}
		break;
	case 58:
		{
#line 617 "EaseaLex.l"

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
 
#line 913 "EaseaLex.cpp"
		}
		break;
	case 59:
		{
#line 640 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 
#line 930 "EaseaLex.cpp"
		}
		break;
	case 60:
		{
#line 652 "EaseaLex.l"

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
 
#line 956 "EaseaLex.cpp"
		}
		break;
	case 61:
		{
#line 673 "EaseaLex.l"

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
  
 
#line 977 "EaseaLex.cpp"
		}
		break;
#line 691 "EaseaLex.l"
  
#line 705 "EaseaLex.l"
      
#line 984 "EaseaLex.cpp"
	case 62:
		{
#line 713 "EaseaLex.l"
        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 
#line 997 "EaseaLex.cpp"
		}
		break;
	case 63:
		{
#line 722 "EaseaLex.l"

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
 
#line 1020 "EaseaLex.cpp"
		}
		break;
	case 64:
		{
#line 739 "EaseaLex.l"

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
 
#line 1043 "EaseaLex.cpp"
		}
		break;
	case 65:
		{
#line 757 "EaseaLex.l"

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
 
#line 1075 "EaseaLex.cpp"
		}
		break;
	case 66:
		{
#line 784 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  fprintf (fpOutputFile,"// Memberwise serialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->serializeIndividual(fpOutputFile, "this");
  //fprintf(fpOutputFile,"\tEASEA_Line << endl;\n");
 
#line 1089 "EaseaLex.cpp"
		}
		break;
	case 67:
		{
#line 793 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome deserializer.\n");
  fprintf (fpOutputFile,"// Memberwise deserialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->deserializeIndividual(fpOutputFile, "this");
 
#line 1102 "EaseaLex.cpp"
		}
		break;
	case 68:
		{
#line 801 "EaseaLex.l"

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
 
#line 1123 "EaseaLex.cpp"
		}
		break;
	case 69:
		{
#line 817 "EaseaLex.l"

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
 
#line 1145 "EaseaLex.cpp"
		}
		break;
	case 70:
		{
#line 834 "EaseaLex.l"
       
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
 
#line 1173 "EaseaLex.cpp"
		}
		break;
	case 71:
		{
#line 856 "EaseaLex.l"

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
 
#line 1195 "EaseaLex.cpp"
		}
		break;
	case 72:
		{
#line 872 "EaseaLex.l"

  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 
#line 1210 "EaseaLex.cpp"
		}
		break;
	case 73:
		{
#line 881 "EaseaLex.l"
        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 
#line 1222 "EaseaLex.cpp"
		}
		break;
	case 74:
		{
#line 889 "EaseaLex.l"

  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 
#line 1234 "EaseaLex.cpp"
		}
		break;
	case 75:
		{
#line 896 "EaseaLex.l"

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
 
#line 1265 "EaseaLex.cpp"
		}
		break;
	case 76:
		{
#line 921 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 
#line 1278 "EaseaLex.cpp"
		}
		break;
	case 77:
		{
#line 928 "EaseaLex.l"
        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 
#line 1292 "EaseaLex.cpp"
		}
		break;
	case 78:
		{
#line 937 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISER;   
 
#line 1304 "EaseaLex.cpp"
		}
		break;
	case 79:
		{
#line 944 "EaseaLex.l"

  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_FINALIZATION_FUNCTION;
 
#line 1317 "EaseaLex.cpp"
		}
		break;
	case 80:
		{
#line 952 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 
#line 1329 "EaseaLex.cpp"
		}
		break;
	case 81:
		{
#line 958 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 
#line 1341 "EaseaLex.cpp"
		}
		break;
	case 82:
		{
#line 964 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 
#line 1353 "EaseaLex.cpp"
		}
		break;
	case 83:
		{
#line 970 "EaseaLex.l"
      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 
#line 1366 "EaseaLex.cpp"
		}
		break;
	case 84:
		{
#line 977 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 
#line 1379 "EaseaLex.cpp"
		}
		break;
	case 85:
		{
#line 984 "EaseaLex.l"
 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 
#line 1393 "EaseaLex.cpp"
		}
		break;
	case 86:
		{
#line 993 "EaseaLex.l"
        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 
#line 1404 "EaseaLex.cpp"
		}
		break;
	case 87:
		{
#line 998 "EaseaLex.l"

  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 
#line 1418 "EaseaLex.cpp"
		}
		break;
	case 88:
		{
#line 1007 "EaseaLex.l"

  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 
#line 1432 "EaseaLex.cpp"
		}
		break;
	case 89:
		{
#line 1016 "EaseaLex.l"

  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 
#line 1446 "EaseaLex.cpp"
		}
		break;
	case 90:
		{
#line 1026 "EaseaLex.l"

  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 
#line 1459 "EaseaLex.cpp"
		}
		break;
	case 91:
		{
#line 1034 "EaseaLex.l"

    fprintf(fpOutputFile,"%d",bIsParentReduce);
 
#line 1468 "EaseaLex.cpp"
		}
		break;
	case 92:
		{
#line 1038 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 
#line 1477 "EaseaLex.cpp"
		}
		break;
	case 93:
		{
#line 1042 "EaseaLex.l"

  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 
#line 1486 "EaseaLex.cpp"
		}
		break;
	case 94:
		{
#line 1046 "EaseaLex.l"

  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 
#line 1495 "EaseaLex.cpp"
		}
		break;
	case 95:
		{
#line 1050 "EaseaLex.l"

  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 
#line 1505 "EaseaLex.cpp"
		}
		break;
	case 96:
		{
#line 1055 "EaseaLex.l"

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

#line 1524 "EaseaLex.cpp"
		}
		break;
	case 97:
		{
#line 1068 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);
#line 1531 "EaseaLex.cpp"
		}
		break;
	case 98:
		{
#line 1069 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fSELECT_PRM);
#line 1538 "EaseaLex.cpp"
		}
		break;
	case 99:
		{
#line 1070 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);
#line 1545 "EaseaLex.cpp"
		}
		break;
	case 100:
		{
#line 1071 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_PAR_PRM);
#line 1552 "EaseaLex.cpp"
		}
		break;
	case 101:
		{
#line 1072 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);
#line 1559 "EaseaLex.cpp"
		}
		break;
	case 102:
		{
#line 1073 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_OFF_PRM);
#line 1566 "EaseaLex.cpp"
		}
		break;
	case 103:
		{
#line 1074 "EaseaLex.l"
fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);
#line 1573 "EaseaLex.cpp"
		}
		break;
	case 104:
		{
#line 1075 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);
#line 1580 "EaseaLex.cpp"
		}
		break;
	case 105:
		{
#line 1076 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPOP_SIZE);
#line 1587 "EaseaLex.cpp"
		}
		break;
	case 106:
		{
#line 1077 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nOFF_SIZE);
#line 1594 "EaseaLex.cpp"
		}
		break;
	case 107:
		{
#line 1078 "EaseaLex.l"

  fprintf(fpOutputFile,"%d",nELITE);
  ////DEBUG_PRT_PRT("elitism is %d, elite size is %d",bELITISM, nELITE);
 
#line 1604 "EaseaLex.cpp"
		}
		break;
	case 108:
		{
#line 1083 "EaseaLex.l"

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
 
#line 1623 "EaseaLex.cpp"
		}
		break;
	case 109:
		{
#line 1096 "EaseaLex.l"

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
 
#line 1642 "EaseaLex.cpp"
		}
		break;
	case 110:
		{
#line 1109 "EaseaLex.l"

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
 
#line 1661 "EaseaLex.cpp"
		}
		break;
	case 111:
		{
#line 1122 "EaseaLex.l"

if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));

#line 1671 "EaseaLex.cpp"
		}
		break;
	case 112:
		{
#line 1126 "EaseaLex.l"
if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);
#line 1678 "EaseaLex.cpp"
		}
		break;
	case 113:
		{
#line 1127 "EaseaLex.l"
if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);
#line 1685 "EaseaLex.cpp"
		}
		break;
	case 114:
		{
#line 1128 "EaseaLex.l"
fprintf(fpOutputFile,"%s",nGENOME_NAME);
#line 1692 "EaseaLex.cpp"
		}
		break;
	case 115:
		{
#line 1129 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nPROBLEM_DIM);
#line 1699 "EaseaLex.cpp"
		}
		break;
	case 116:
		{
#line 1130 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_GEN);
#line 1706 "EaseaLex.cpp"
		}
		break;
	case 117:
		{
#line 1131 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nTIME_LIMIT);
#line 1713 "EaseaLex.cpp"
		}
		break;
	case 118:
		{
#line 1132 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMUT_PROB);
#line 1720 "EaseaLex.cpp"
		}
		break;
	case 119:
		{
#line 1133 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fXOVER_PROB);
#line 1727 "EaseaLex.cpp"
		}
		break;
	case 120:
		{
#line 1134 "EaseaLex.l"
fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); 
#line 1734 "EaseaLex.cpp"
		}
		break;
	case 121:
		{
#line 1135 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bELITISM);
#line 1741 "EaseaLex.cpp"
		}
		break;
	case 122:
		{
#line 1137 "EaseaLex.l"
fprintf(fpOutputFile,"%d",nNB_OPT_IT);
#line 1748 "EaseaLex.cpp"
		}
		break;
	case 123:
		{
#line 1138 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bBALDWINISM);
#line 1755 "EaseaLex.cpp"
		}
		break;
	case 124:
		{
#line 1140 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bREMOTE_ISLAND_MODEL);
#line 1762 "EaseaLex.cpp"
		}
		break;
	case 125:
		{
#line 1141 "EaseaLex.l"
if(strlen(sIP_FILE)>0)fprintf(fpOutputFile,"%s",sIP_FILE); else fprintf(fpOutputFile,"NULL");
#line 1769 "EaseaLex.cpp"
		}
		break;
	case 126:
		{
#line 1143 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1776 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1144 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1783 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1145 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1790 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1146 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1797 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1147 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1804 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1149 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 1811 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1150 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 1818 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1152 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1832 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1160 "EaseaLex.l"

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
 
#line 1852 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1174 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1866 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1182 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1880 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1191 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1894 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1200 "EaseaLex.l"

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

#line 1957 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1257 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded without warnings.\n");
  printf ("You can now type \"make\" to compile your project.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 1974 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1269 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1981 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1275 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 1993 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1281 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2006 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1288 "EaseaLex.l"

#line 2013 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1289 "EaseaLex.l"
lineCounter++;
#line 2020 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1291 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2032 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1297 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2045 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1305 "EaseaLex.l"

#line 2052 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1306 "EaseaLex.l"

  lineCounter++;
 
#line 2061 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1310 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2073 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1316 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2087 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1324 "EaseaLex.l"

#line 2094 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1325 "EaseaLex.l"

  lineCounter++;
 
#line 2103 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1329 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
      
  BEGIN COPY;
 
#line 2117 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1337 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2132 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1346 "EaseaLex.l"

#line 2139 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1347 "EaseaLex.l"
lineCounter++;
#line 2146 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1352 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2160 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1361 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2174 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1369 "EaseaLex.l"

#line 2181 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1370 "EaseaLex.l"
lineCounter++;
#line 2188 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1373 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2204 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1384 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2220 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1394 "EaseaLex.l"

#line 2227 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1397 "EaseaLex.l"

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
 
#line 2245 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1410 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2262 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1422 "EaseaLex.l"

#line 2269 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1423 "EaseaLex.l"
lineCounter++;
#line 2276 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1425 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2292 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1437 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2308 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1447 "EaseaLex.l"
lineCounter++;
#line 2315 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1448 "EaseaLex.l"

#line 2322 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1452 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2337 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1462 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2352 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1471 "EaseaLex.l"

#line 2359 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1474 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2372 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1481 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2386 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1489 "EaseaLex.l"

#line 2393 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1493 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2401 "EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 1495 "EaseaLex.l"

#line 2408 "EaseaLex.cpp"
		}
		break;
	case 180:
		{
#line 1501 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2415 "EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 1502 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2422 "EaseaLex.cpp"
		}
		break;
	case 182:
	case 183:
		{
#line 1505 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2433 "EaseaLex.cpp"
		}
		break;
	case 184:
	case 185:
		{
#line 1510 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2442 "EaseaLex.cpp"
		}
		break;
	case 186:
	case 187:
		{
#line 1513 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2451 "EaseaLex.cpp"
		}
		break;
	case 188:
	case 189:
		{
#line 1516 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2468 "EaseaLex.cpp"
		}
		break;
	case 190:
	case 191:
		{
#line 1527 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2482 "EaseaLex.cpp"
		}
		break;
	case 192:
	case 193:
		{
#line 1535 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2491 "EaseaLex.cpp"
		}
		break;
	case 194:
	case 195:
		{
#line 1538 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2500 "EaseaLex.cpp"
		}
		break;
	case 196:
	case 197:
		{
#line 1541 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2509 "EaseaLex.cpp"
		}
		break;
	case 198:
	case 199:
		{
#line 1544 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2518 "EaseaLex.cpp"
		}
		break;
	case 200:
	case 201:
		{
#line 1547 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2527 "EaseaLex.cpp"
		}
		break;
	case 202:
	case 203:
		{
#line 1551 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2539 "EaseaLex.cpp"
		}
		break;
	case 204:
		{
#line 1557 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2546 "EaseaLex.cpp"
		}
		break;
	case 205:
		{
#line 1558 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2553 "EaseaLex.cpp"
		}
		break;
	case 206:
		{
#line 1559 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2560 "EaseaLex.cpp"
		}
		break;
	case 207:
		{
#line 1560 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2570 "EaseaLex.cpp"
		}
		break;
	case 208:
		{
#line 1565 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2577 "EaseaLex.cpp"
		}
		break;
	case 209:
		{
#line 1566 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2584 "EaseaLex.cpp"
		}
		break;
	case 210:
		{
#line 1567 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2591 "EaseaLex.cpp"
		}
		break;
	case 211:
		{
#line 1568 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2598 "EaseaLex.cpp"
		}
		break;
	case 212:
		{
#line 1569 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2605 "EaseaLex.cpp"
		}
		break;
	case 213:
		{
#line 1570 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2612 "EaseaLex.cpp"
		}
		break;
	case 214:
		{
#line 1571 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2619 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1572 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2626 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1573 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2634 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1575 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2642 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1577 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2650 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1579 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2660 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1583 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2667 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1584 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2674 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1585 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2685 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1590 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2692 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1591 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2701 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1594 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2713 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1600 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2722 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1603 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2734 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1609 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2745 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1614 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2761 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1624 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2768 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1627 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2777 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1630 "EaseaLex.l"
BEGIN COPY;
#line 2784 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1632 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2791 "EaseaLex.cpp"
		}
		break;
	case 234:
	case 235:
	case 236:
		{
#line 1635 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2804 "EaseaLex.cpp"
		}
		break;
	case 237:
		{
#line 1640 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2815 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1645 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
#line 2824 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1654 "EaseaLex.l"
;
#line 2831 "EaseaLex.cpp"
		}
		break;
	case 240:
		{
#line 1655 "EaseaLex.l"
;
#line 2838 "EaseaLex.cpp"
		}
		break;
	case 241:
		{
#line 1656 "EaseaLex.l"
;
#line 2845 "EaseaLex.cpp"
		}
		break;
	case 242:
		{
#line 1657 "EaseaLex.l"
;
#line 2852 "EaseaLex.cpp"
		}
		break;
	case 243:
		{
#line 1660 "EaseaLex.l"
 /* do nothing */ 
#line 2859 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1661 "EaseaLex.l"
 /*return '\n';*/ 
#line 2866 "EaseaLex.cpp"
		}
		break;
	case 245:
		{
#line 1662 "EaseaLex.l"
 /*return '\n';*/ 
#line 2873 "EaseaLex.cpp"
		}
		break;
	case 246:
		{
#line 1665 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 2882 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1668 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 2892 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1672 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
//  printf("match gpnode\n");
  return GPNODE;
 
#line 2904 "EaseaLex.cpp"
		}
		break;
	case 249:
		{
#line 1679 "EaseaLex.l"
return STATIC;
#line 2911 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1680 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 2918 "EaseaLex.cpp"
		}
		break;
	case 251:
		{
#line 1681 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 2925 "EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1682 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 2932 "EaseaLex.cpp"
		}
		break;
	case 253:
		{
#line 1683 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 2939 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1684 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 2946 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1686 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 2953 "EaseaLex.cpp"
		}
		break;
#line 1687 "EaseaLex.l"
  
#line 2958 "EaseaLex.cpp"
	case 256:
		{
#line 1688 "EaseaLex.l"
return GENOME; 
#line 2963 "EaseaLex.cpp"
		}
		break;
	case 257:
		{
#line 1690 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 2973 "EaseaLex.cpp"
		}
		break;
	case 258:
	case 259:
	case 260:
		{
#line 1697 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 2982 "EaseaLex.cpp"
		}
		break;
	case 261:
		{
#line 1698 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 2989 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1701 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 2997 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1703 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 3004 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1709 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 3016 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1715 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 3029 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1722 "EaseaLex.l"

#line 3036 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1724 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3047 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1735 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3062 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1745 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3073 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1751 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3082 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1755 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3097 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1768 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3109 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1774 "EaseaLex.l"

#line 3116 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1775 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3129 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1782 "EaseaLex.l"

#line 3136 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1783 "EaseaLex.l"
lineCounter++;
#line 3143 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1784 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3156 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1791 "EaseaLex.l"

#line 3163 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1792 "EaseaLex.l"
lineCounter++;
#line 3170 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1794 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3183 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1801 "EaseaLex.l"

#line 3190 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1802 "EaseaLex.l"
lineCounter++;
#line 3197 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1804 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3210 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1811 "EaseaLex.l"

#line 3217 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1812 "EaseaLex.l"
lineCounter++;
#line 3224 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1818 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3231 "EaseaLex.cpp"
		}
		break;
	case 287:
		{
#line 1819 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3238 "EaseaLex.cpp"
		}
		break;
	case 288:
		{
#line 1820 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3245 "EaseaLex.cpp"
		}
		break;
	case 289:
		{
#line 1821 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3252 "EaseaLex.cpp"
		}
		break;
	case 290:
		{
#line 1822 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3259 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1823 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3266 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1824 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3273 "EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1826 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3282 "EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1829 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3295 "EaseaLex.cpp"
		}
		break;
	case 295:
	case 296:
		{
#line 1838 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3306 "EaseaLex.cpp"
		}
		break;
	case 297:
	case 298:
		{
#line 1843 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3315 "EaseaLex.cpp"
		}
		break;
	case 299:
	case 300:
		{
#line 1846 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3324 "EaseaLex.cpp"
		}
		break;
	case 301:
	case 302:
		{
#line 1849 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3336 "EaseaLex.cpp"
		}
		break;
	case 303:
	case 304:
		{
#line 1855 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3349 "EaseaLex.cpp"
		}
		break;
	case 305:
	case 306:
		{
#line 1862 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3358 "EaseaLex.cpp"
		}
		break;
	case 307:
	case 308:
		{
#line 1865 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3367 "EaseaLex.cpp"
		}
		break;
	case 309:
	case 310:
		{
#line 1868 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3376 "EaseaLex.cpp"
		}
		break;
	case 311:
	case 312:
		{
#line 1871 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3385 "EaseaLex.cpp"
		}
		break;
	case 313:
	case 314:
		{
#line 1874 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3394 "EaseaLex.cpp"
		}
		break;
	case 315:
		{
#line 1877 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3403 "EaseaLex.cpp"
		}
		break;
	case 316:
		{
#line 1880 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3413 "EaseaLex.cpp"
		}
		break;
	case 317:
		{
#line 1884 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3421 "EaseaLex.cpp"
		}
		break;
	case 318:
		{
#line 1886 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3432 "EaseaLex.cpp"
		}
		break;
	case 319:
		{
#line 1891 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3443 "EaseaLex.cpp"
		}
		break;
	case 320:
		{
#line 1896 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3451 "EaseaLex.cpp"
		}
		break;
	case 321:
		{
#line 1898 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3459 "EaseaLex.cpp"
		}
		break;
	case 322:
		{
#line 1900 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3467 "EaseaLex.cpp"
		}
		break;
	case 323:
		{
#line 1902 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3475 "EaseaLex.cpp"
		}
		break;
	case 324:
		{
#line 1904 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3483 "EaseaLex.cpp"
		}
		break;
	case 325:
		{
#line 1906 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3490 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1907 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3497 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1908 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3505 "EaseaLex.cpp"
		}
		break;
	case 328:
		{
#line 1910 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3513 "EaseaLex.cpp"
		}
		break;
	case 329:
		{
#line 1912 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3521 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1914 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3528 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1915 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3540 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 1921 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3549 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 1924 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3559 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 1928 "EaseaLex.l"
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
#line 3576 "EaseaLex.cpp"
		}
		break;
	case 335:
	case 336:
		{
#line 1940 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3586 "EaseaLex.cpp"
		}
		break;
	case 337:
		{
#line 1943 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3593 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1950 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3600 "EaseaLex.cpp"
		}
		break;
	case 339:
		{
#line 1951 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3607 "EaseaLex.cpp"
		}
		break;
	case 340:
		{
#line 1952 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
#line 3614 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 1953 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3621 "EaseaLex.cpp"
		}
		break;
	case 342:
		{
#line 1954 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3628 "EaseaLex.cpp"
		}
		break;
	case 343:
		{
#line 1956 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3637 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 1960 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3650 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 1968 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3663 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 1977 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3676 "EaseaLex.cpp"
		}
		break;
	case 347:
		{
#line 1986 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3691 "EaseaLex.cpp"
		}
		break;
	case 348:
		{
#line 1996 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3698 "EaseaLex.cpp"
		}
		break;
	case 349:
		{
#line 1997 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3705 "EaseaLex.cpp"
		}
		break;
	case 350:
	case 351:
		{
#line 2000 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3716 "EaseaLex.cpp"
		}
		break;
	case 352:
	case 353:
		{
#line 2005 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3725 "EaseaLex.cpp"
		}
		break;
	case 354:
	case 355:
		{
#line 2008 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3734 "EaseaLex.cpp"
		}
		break;
	case 356:
	case 357:
		{
#line 2011 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3747 "EaseaLex.cpp"
		}
		break;
	case 358:
	case 359:
		{
#line 2018 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3760 "EaseaLex.cpp"
		}
		break;
	case 360:
	case 361:
		{
#line 2025 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3769 "EaseaLex.cpp"
		}
		break;
	case 362:
		{
#line 2028 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3776 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 2029 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3783 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 2030 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3790 "EaseaLex.cpp"
		}
		break;
	case 365:
		{
#line 2031 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3800 "EaseaLex.cpp"
		}
		break;
	case 366:
		{
#line 2036 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3807 "EaseaLex.cpp"
		}
		break;
	case 367:
		{
#line 2037 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3814 "EaseaLex.cpp"
		}
		break;
	case 368:
		{
#line 2038 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3821 "EaseaLex.cpp"
		}
		break;
	case 369:
		{
#line 2039 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3828 "EaseaLex.cpp"
		}
		break;
	case 370:
		{
#line 2040 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3836 "EaseaLex.cpp"
		}
		break;
	case 371:
		{
#line 2042 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3844 "EaseaLex.cpp"
		}
		break;
	case 372:
		{
#line 2044 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3852 "EaseaLex.cpp"
		}
		break;
	case 373:
		{
#line 2046 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3860 "EaseaLex.cpp"
		}
		break;
	case 374:
		{
#line 2048 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 3868 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2050 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 3876 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2052 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 3884 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2054 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3891 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2055 "EaseaLex.l"
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
#line 3914 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2072 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 3925 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2077 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 3939 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2085 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3946 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2091 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 3956 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2095 "EaseaLex.l"

#line 3963 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2098 "EaseaLex.l"
;
#line 3970 "EaseaLex.cpp"
		}
		break;
	case 385:
		{
#line 2099 "EaseaLex.l"
;
#line 3977 "EaseaLex.cpp"
		}
		break;
	case 386:
		{
#line 2100 "EaseaLex.l"
;
#line 3984 "EaseaLex.cpp"
		}
		break;
	case 387:
		{
#line 2101 "EaseaLex.l"
;
#line 3991 "EaseaLex.cpp"
		}
		break;
	case 388:
		{
#line 2103 "EaseaLex.l"
 /* do nothing */ 
#line 3998 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2104 "EaseaLex.l"
 /*return '\n';*/ 
#line 4005 "EaseaLex.cpp"
		}
		break;
	case 390:
		{
#line 2105 "EaseaLex.l"
 /*return '\n';*/ 
#line 4012 "EaseaLex.cpp"
		}
		break;
	case 391:
		{
#line 2107 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 4019 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2108 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 4026 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2109 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 4033 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2110 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4040 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2111 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4047 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2112 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4054 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2113 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4061 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2114 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4068 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2115 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4075 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2117 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4082 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2118 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4089 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2119 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4096 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2120 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4103 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2121 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4110 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2123 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4117 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2124 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4124 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2126 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4135 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2131 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4142 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2133 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4153 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2138 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4160 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2141 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4167 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2142 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4174 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2143 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4181 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2144 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4188 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2145 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4195 "EaseaLex.cpp"
		}
		break;
#line 2146 "EaseaLex.l"
 
#line 4200 "EaseaLex.cpp"
	case 416:
		{
#line 2147 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4205 "EaseaLex.cpp"
		}
		break;
	case 417:
		{
#line 2148 "EaseaLex.l"
if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;
#line 4212 "EaseaLex.cpp"
		}
		break;
#line 2150 "EaseaLex.l"
 
#line 4217 "EaseaLex.cpp"
	case 418:
	case 419:
	case 420:
		{
#line 2154 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4224 "EaseaLex.cpp"
		}
		break;
	case 421:
		{
#line 2155 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4231 "EaseaLex.cpp"
		}
		break;
	case 422:
		{
#line 2158 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4239 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2161 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4246 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2163 "EaseaLex.l"

  lineCounter++;

#line 4255 "EaseaLex.cpp"
		}
		break;
	case 425:
		{
#line 2166 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4265 "EaseaLex.cpp"
		}
		break;
	case 426:
		{
#line 2171 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4275 "EaseaLex.cpp"
		}
		break;
	case 427:
		{
#line 2176 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4285 "EaseaLex.cpp"
		}
		break;
	case 428:
		{
#line 2181 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4295 "EaseaLex.cpp"
		}
		break;
	case 429:
		{
#line 2186 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4305 "EaseaLex.cpp"
		}
		break;
	case 430:
		{
#line 2191 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4315 "EaseaLex.cpp"
		}
		break;
	case 431:
		{
#line 2200 "EaseaLex.l"
return  (char)yytext[0];
#line 4322 "EaseaLex.cpp"
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
#line 2202 "EaseaLex.l"


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

#line 4519 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
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
		188,
		-189,
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
		200,
		-201,
		0,
		192,
		-193,
		0,
		190,
		-191,
		0,
		-232,
		0,
		-238,
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
		313,
		-314,
		0,
		297,
		-298,
		0,
		360,
		-361,
		0,
		305,
		-306,
		0,
		358,
		-359,
		0,
		303,
		-304,
		0,
		352,
		-353,
		0,
		354,
		-355,
		0,
		356,
		-357,
		0,
		350,
		-351,
		0,
		299,
		-300,
		0,
		301,
		-302,
		0,
		295,
		-296,
		0
	};
	yymatch = match;

	yytransitionmax = 4961;
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
		{ 3014, 61 },
		{ 3014, 61 },
		{ 1865, 1968 },
		{ 1490, 1473 },
		{ 1491, 1473 },
		{ 2367, 2341 },
		{ 2367, 2341 },
		{ 2773, 2775 },
		{ 165, 161 },
		{ 2340, 2311 },
		{ 2340, 2311 },
		{ 71, 3 },
		{ 72, 3 },
		{ 2224, 43 },
		{ 2225, 43 },
		{ 69, 1 },
		{ 1865, 1846 },
		{ 165, 167 },
		{ 67, 1 },
		{ 0, 1804 },
		{ 2190, 2186 },
		{ 0, 1995 },
		{ 1964, 1966 },
		{ 3014, 61 },
		{ 1347, 1346 },
		{ 3012, 61 },
		{ 1490, 1473 },
		{ 3063, 3061 },
		{ 2367, 2341 },
		{ 1338, 1337 },
		{ 1527, 1511 },
		{ 1528, 1511 },
		{ 2340, 2311 },
		{ 2191, 2187 },
		{ 71, 3 },
		{ 3016, 61 },
		{ 2224, 43 },
		{ 86, 61 },
		{ 3011, 61 },
		{ 3013, 61 },
		{ 3013, 61 },
		{ 3013, 61 },
		{ 3013, 61 },
		{ 3013, 61 },
		{ 3013, 61 },
		{ 3013, 61 },
		{ 3013, 61 },
		{ 3013, 61 },
		{ 3013, 61 },
		{ 70, 3 },
		{ 3015, 61 },
		{ 2223, 43 },
		{ 1580, 1574 },
		{ 1527, 1511 },
		{ 2368, 2341 },
		{ 1492, 1473 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 1529, 1511 },
		{ 3009, 61 },
		{ 1582, 1576 },
		{ 1453, 1432 },
		{ 3010, 61 },
		{ 1454, 1433 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3010, 61 },
		{ 3017, 61 },
		{ 2185, 40 },
		{ 1530, 1512 },
		{ 1531, 1512 },
		{ 1447, 1425 },
		{ 1972, 40 },
		{ 2422, 2395 },
		{ 2422, 2395 },
		{ 2345, 2315 },
		{ 2345, 2315 },
		{ 2348, 2318 },
		{ 2348, 2318 },
		{ 1987, 39 },
		{ 1448, 1426 },
		{ 1798, 37 },
		{ 2365, 2339 },
		{ 2365, 2339 },
		{ 1449, 1427 },
		{ 1450, 1428 },
		{ 1452, 1431 },
		{ 1455, 1434 },
		{ 1456, 1435 },
		{ 1457, 1436 },
		{ 1458, 1437 },
		{ 2185, 40 },
		{ 1530, 1512 },
		{ 1975, 40 },
		{ 1459, 1438 },
		{ 1460, 1439 },
		{ 2422, 2395 },
		{ 1461, 1440 },
		{ 2345, 2315 },
		{ 1462, 1441 },
		{ 2348, 2318 },
		{ 1463, 1443 },
		{ 1987, 39 },
		{ 1466, 1446 },
		{ 1798, 37 },
		{ 2365, 2339 },
		{ 2184, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1973, 39 },
		{ 1988, 40 },
		{ 1785, 37 },
		{ 1467, 1447 },
		{ 1532, 1512 },
		{ 2423, 2395 },
		{ 1468, 1448 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1974, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1982, 40 },
		{ 1980, 40 },
		{ 1993, 40 },
		{ 1981, 40 },
		{ 1993, 40 },
		{ 1984, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1983, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1469, 1449 },
		{ 1976, 40 },
		{ 1978, 40 },
		{ 1470, 1450 },
		{ 1993, 40 },
		{ 1472, 1452 },
		{ 1993, 40 },
		{ 1991, 40 },
		{ 1979, 40 },
		{ 1993, 40 },
		{ 1992, 40 },
		{ 1985, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1990, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1977, 40 },
		{ 1993, 40 },
		{ 1989, 40 },
		{ 1993, 40 },
		{ 1986, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1993, 40 },
		{ 1374, 21 },
		{ 1533, 1513 },
		{ 1534, 1513 },
		{ 1473, 1453 },
		{ 1361, 21 },
		{ 2373, 2346 },
		{ 2373, 2346 },
		{ 2385, 2358 },
		{ 2385, 2358 },
		{ 2388, 2361 },
		{ 2388, 2361 },
		{ 2410, 2383 },
		{ 2410, 2383 },
		{ 2411, 2384 },
		{ 2411, 2384 },
		{ 2453, 2426 },
		{ 2453, 2426 },
		{ 1474, 1454 },
		{ 1475, 1455 },
		{ 1476, 1456 },
		{ 1477, 1457 },
		{ 1478, 1458 },
		{ 1479, 1459 },
		{ 1374, 21 },
		{ 1533, 1513 },
		{ 1362, 21 },
		{ 1375, 21 },
		{ 1480, 1460 },
		{ 2373, 2346 },
		{ 1481, 1461 },
		{ 2385, 2358 },
		{ 1482, 1463 },
		{ 2388, 2361 },
		{ 1485, 1466 },
		{ 2410, 2383 },
		{ 1486, 1467 },
		{ 2411, 2384 },
		{ 1487, 1468 },
		{ 2453, 2426 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1488, 1470 },
		{ 1489, 1472 },
		{ 1386, 1364 },
		{ 1493, 1474 },
		{ 1535, 1513 },
		{ 1494, 1475 },
		{ 1499, 1478 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1378, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1367, 21 },
		{ 1365, 21 },
		{ 1380, 21 },
		{ 1366, 21 },
		{ 1380, 21 },
		{ 1369, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1368, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1500, 1479 },
		{ 1363, 21 },
		{ 1376, 21 },
		{ 1501, 1480 },
		{ 1370, 21 },
		{ 1502, 1481 },
		{ 1380, 21 },
		{ 1381, 21 },
		{ 1364, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1371, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1379, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1382, 21 },
		{ 1380, 21 },
		{ 1377, 21 },
		{ 1380, 21 },
		{ 1372, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1380, 21 },
		{ 1959, 38 },
		{ 1536, 1514 },
		{ 1537, 1514 },
		{ 1495, 1476 },
		{ 1784, 38 },
		{ 2458, 2431 },
		{ 2458, 2431 },
		{ 2464, 2437 },
		{ 2464, 2437 },
		{ 1497, 1477 },
		{ 1496, 1476 },
		{ 2477, 2451 },
		{ 2477, 2451 },
		{ 2478, 2452 },
		{ 2478, 2452 },
		{ 1503, 1482 },
		{ 1498, 1477 },
		{ 1506, 1486 },
		{ 1507, 1487 },
		{ 1508, 1488 },
		{ 1509, 1489 },
		{ 1511, 1493 },
		{ 1512, 1494 },
		{ 1959, 38 },
		{ 1536, 1514 },
		{ 1789, 38 },
		{ 2482, 2456 },
		{ 2482, 2456 },
		{ 2458, 2431 },
		{ 1513, 1495 },
		{ 2464, 2437 },
		{ 1514, 1496 },
		{ 1539, 1515 },
		{ 1540, 1515 },
		{ 2477, 2451 },
		{ 1515, 1497 },
		{ 2478, 2452 },
		{ 1516, 1498 },
		{ 1958, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 2482, 2456 },
		{ 1799, 38 },
		{ 1517, 1499 },
		{ 1518, 1500 },
		{ 1538, 1514 },
		{ 1519, 1501 },
		{ 1539, 1515 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1786, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1794, 38 },
		{ 1792, 38 },
		{ 1802, 38 },
		{ 1793, 38 },
		{ 1802, 38 },
		{ 1796, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1795, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1520, 1502 },
		{ 1790, 38 },
		{ 1541, 1515 },
		{ 1521, 1503 },
		{ 1802, 38 },
		{ 1523, 1506 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1791, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1787, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1788, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1801, 38 },
		{ 1802, 38 },
		{ 1800, 38 },
		{ 1802, 38 },
		{ 1797, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 1802, 38 },
		{ 2767, 44 },
		{ 2768, 44 },
		{ 1542, 1516 },
		{ 1543, 1516 },
		{ 67, 44 },
		{ 2488, 2462 },
		{ 2488, 2462 },
		{ 1524, 1507 },
		{ 1525, 1508 },
		{ 1526, 1509 },
		{ 1389, 1365 },
		{ 2489, 2463 },
		{ 2489, 2463 },
		{ 2277, 2249 },
		{ 2277, 2249 },
		{ 1390, 1366 },
		{ 1394, 1368 },
		{ 1395, 1369 },
		{ 1396, 1370 },
		{ 1545, 1517 },
		{ 1546, 1518 },
		{ 1547, 1519 },
		{ 1549, 1523 },
		{ 2767, 44 },
		{ 1550, 1524 },
		{ 1542, 1516 },
		{ 2299, 2270 },
		{ 2299, 2270 },
		{ 2488, 2462 },
		{ 1551, 1525 },
		{ 1560, 1546 },
		{ 1561, 1546 },
		{ 1569, 1559 },
		{ 1570, 1559 },
		{ 2489, 2463 },
		{ 1552, 1526 },
		{ 2277, 2249 },
		{ 2240, 44 },
		{ 2766, 44 },
		{ 2239, 44 },
		{ 2239, 44 },
		{ 2239, 44 },
		{ 2239, 44 },
		{ 2239, 44 },
		{ 2239, 44 },
		{ 2239, 44 },
		{ 2239, 44 },
		{ 2239, 44 },
		{ 2239, 44 },
		{ 2299, 2270 },
		{ 1559, 1545 },
		{ 1397, 1371 },
		{ 1563, 1547 },
		{ 1560, 1546 },
		{ 1544, 1516 },
		{ 1569, 1559 },
		{ 2241, 44 },
		{ 2237, 44 },
		{ 2232, 44 },
		{ 2241, 44 },
		{ 2229, 44 },
		{ 2236, 44 },
		{ 2234, 44 },
		{ 2241, 44 },
		{ 2238, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2231, 44 },
		{ 2226, 44 },
		{ 2233, 44 },
		{ 2228, 44 },
		{ 2241, 44 },
		{ 2235, 44 },
		{ 2230, 44 },
		{ 2227, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 1562, 1546 },
		{ 2245, 44 },
		{ 1571, 1559 },
		{ 1565, 1549 },
		{ 2241, 44 },
		{ 1566, 1550 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2242, 44 },
		{ 2243, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2244, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 2241, 44 },
		{ 159, 4 },
		{ 160, 4 },
		{ 2503, 2474 },
		{ 2503, 2474 },
		{ 2320, 2290 },
		{ 2320, 2290 },
		{ 2321, 2291 },
		{ 2321, 2291 },
		{ 2337, 2308 },
		{ 2337, 2308 },
		{ 1393, 1367 },
		{ 1567, 1551 },
		{ 1568, 1552 },
		{ 1399, 1372 },
		{ 1574, 1565 },
		{ 1575, 1566 },
		{ 1398, 1372 },
		{ 1576, 1567 },
		{ 1392, 1367 },
		{ 1577, 1568 },
		{ 1402, 1377 },
		{ 1581, 1575 },
		{ 1403, 1378 },
		{ 159, 4 },
		{ 1583, 1577 },
		{ 2503, 2474 },
		{ 1586, 1581 },
		{ 2320, 2290 },
		{ 1587, 1583 },
		{ 2321, 2291 },
		{ 1391, 1367 },
		{ 2337, 2308 },
		{ 1589, 1586 },
		{ 1590, 1587 },
		{ 1591, 1589 },
		{ 1592, 1590 },
		{ 1387, 1591 },
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
		{ 1404, 1379 },
		{ 1405, 1381 },
		{ 0, 2474 },
		{ 1406, 1382 },
		{ 1409, 1386 },
		{ 1410, 1389 },
		{ 1411, 1390 },
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
		{ 1412, 1391 },
		{ 81, 4 },
		{ 1413, 1392 },
		{ 1414, 1393 },
		{ 85, 4 },
		{ 1415, 1394 },
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
		{ 3021, 3020 },
		{ 1416, 1395 },
		{ 1417, 1396 },
		{ 3020, 3020 },
		{ 1419, 1397 },
		{ 1420, 1398 },
		{ 1418, 1396 },
		{ 1421, 1399 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 1424, 1402 },
		{ 3020, 3020 },
		{ 1425, 1403 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 1426, 1404 },
		{ 1427, 1405 },
		{ 1428, 1406 },
		{ 1431, 1409 },
		{ 1432, 1410 },
		{ 1433, 1411 },
		{ 1434, 1412 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 1435, 1413 },
		{ 1436, 1414 },
		{ 1437, 1415 },
		{ 1438, 1416 },
		{ 1439, 1417 },
		{ 1440, 1418 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 3020, 3020 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1441, 1419 },
		{ 1442, 1420 },
		{ 1443, 1421 },
		{ 1446, 1424 },
		{ 154, 152 },
		{ 105, 90 },
		{ 106, 91 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 107, 92 },
		{ 108, 93 },
		{ 109, 94 },
		{ 110, 95 },
		{ 1387, 1593 },
		{ 111, 96 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 1387, 1593 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 112, 97 },
		{ 114, 99 },
		{ 120, 104 },
		{ 121, 105 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 109 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 125, 110 },
		{ 126, 111 },
		{ 127, 112 },
		{ 129, 114 },
		{ 2241, 2497 },
		{ 134, 120 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 2241, 2497 },
		{ 1388, 1592 },
		{ 0, 1592 },
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
		{ 2248, 2226 },
		{ 2250, 2227 },
		{ 2677, 2677 },
		{ 2253, 2228 },
		{ 2254, 2229 },
		{ 2261, 2231 },
		{ 2251, 2228 },
		{ 2264, 2232 },
		{ 2257, 2230 },
		{ 2252, 2228 },
		{ 2265, 2233 },
		{ 1388, 1592 },
		{ 2256, 2230 },
		{ 2266, 2234 },
		{ 2255, 2229 },
		{ 2267, 2235 },
		{ 2268, 2236 },
		{ 2269, 2237 },
		{ 2270, 2238 },
		{ 2241, 2241 },
		{ 2249, 2243 },
		{ 2262, 2242 },
		{ 2260, 2244 },
		{ 2276, 2248 },
		{ 145, 137 },
		{ 2677, 2677 },
		{ 2278, 2250 },
		{ 2258, 2230 },
		{ 2259, 2230 },
		{ 2263, 2242 },
		{ 2279, 2251 },
		{ 2280, 2252 },
		{ 2281, 2253 },
		{ 2282, 2254 },
		{ 2283, 2255 },
		{ 2284, 2256 },
		{ 2285, 2257 },
		{ 2286, 2258 },
		{ 2287, 2259 },
		{ 2288, 2260 },
		{ 0, 1592 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2289, 2261 },
		{ 2290, 2262 },
		{ 2291, 2263 },
		{ 2292, 2264 },
		{ 2293, 2265 },
		{ 2294, 2266 },
		{ 2297, 2268 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 67, 7 },
		{ 2298, 2269 },
		{ 146, 138 },
		{ 2306, 2276 },
		{ 2677, 2677 },
		{ 1593, 1592 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2677, 2677 },
		{ 2308, 2278 },
		{ 2309, 2279 },
		{ 2310, 2280 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 2311, 2281 },
		{ 2312, 2282 },
		{ 2313, 2283 },
		{ 2314, 2284 },
		{ 2315, 2285 },
		{ 2316, 2286 },
		{ 2317, 2287 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 2318, 2288 },
		{ 2319, 2289 },
		{ 147, 140 },
		{ 2322, 2292 },
		{ 1214, 7 },
		{ 2323, 2293 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 1214, 7 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 2324, 2294 },
		{ 2325, 2295 },
		{ 2326, 2296 },
		{ 2327, 2297 },
		{ 2328, 2298 },
		{ 2335, 2306 },
		{ 148, 141 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 2338, 2309 },
		{ 2339, 2310 },
		{ 149, 142 },
		{ 2343, 2313 },
		{ 0, 1451 },
		{ 2344, 2314 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1451 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 2341, 2312 },
		{ 2346, 2316 },
		{ 2347, 2317 },
		{ 150, 144 },
		{ 2342, 2312 },
		{ 2349, 2319 },
		{ 2353, 2322 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 2354, 2323 },
		{ 2355, 2324 },
		{ 2356, 2325 },
		{ 2357, 2326 },
		{ 0, 1864 },
		{ 2358, 2327 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 1864 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 2359, 2328 },
		{ 2361, 2335 },
		{ 2364, 2338 },
		{ 2369, 2342 },
		{ 2370, 2343 },
		{ 2371, 2344 },
		{ 151, 147 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 2374, 2347 },
		{ 2376, 2349 },
		{ 2380, 2353 },
		{ 2381, 2354 },
		{ 0, 2058 },
		{ 2382, 2355 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 0, 2058 },
		{ 2295, 2267 },
		{ 2383, 2356 },
		{ 2384, 2357 },
		{ 152, 148 },
		{ 2386, 2359 },
		{ 2392, 2364 },
		{ 2395, 2369 },
		{ 2396, 2370 },
		{ 2398, 2371 },
		{ 2296, 2267 },
		{ 2401, 2374 },
		{ 2403, 2376 },
		{ 2397, 2371 },
		{ 2407, 2380 },
		{ 2408, 2381 },
		{ 2409, 2382 },
		{ 153, 150 },
		{ 2413, 2386 },
		{ 2419, 2392 },
		{ 89, 73 },
		{ 2424, 2396 },
		{ 2425, 2397 },
		{ 2426, 2398 },
		{ 2429, 2401 },
		{ 2431, 2403 },
		{ 2435, 2407 },
		{ 2436, 2408 },
		{ 2437, 2409 },
		{ 2442, 2413 },
		{ 2448, 2419 },
		{ 2451, 2424 },
		{ 2452, 2425 },
		{ 155, 153 },
		{ 2456, 2429 },
		{ 156, 155 },
		{ 2462, 2435 },
		{ 2463, 2436 },
		{ 157, 156 },
		{ 2469, 2442 },
		{ 2474, 2448 },
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
		{ 2330, 2300 },
		{ 0, 2844 },
		{ 2330, 2300 },
		{ 85, 157 },
		{ 2573, 2548 },
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
		{ 2332, 2303 },
		{ 116, 101 },
		{ 2332, 2303 },
		{ 116, 101 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1224, 1221 },
		{ 130, 115 },
		{ 1224, 1221 },
		{ 130, 115 },
		{ 2843, 49 },
		{ 2584, 2559 },
		{ 91, 74 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1227, 1223 },
		{ 2301, 2272 },
		{ 1227, 1223 },
		{ 2301, 2272 },
		{ 1214, 1214 },
		{ 1782, 1781 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 1214, 1214 },
		{ 0, 2469 },
		{ 0, 2469 },
		{ 1229, 1226 },
		{ 132, 118 },
		{ 1229, 1226 },
		{ 132, 118 },
		{ 1266, 1265 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 1740, 1739 },
		{ 2713, 2698 },
		{ 2730, 2716 },
		{ 2164, 2161 },
		{ 1263, 1262 },
		{ 2549, 2520 },
		{ 0, 2469 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 1647, 1646 },
		{ 86, 49 },
		{ 1692, 1691 },
		{ 2990, 2989 },
		{ 3010, 3010 },
		{ 1737, 1736 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 3010, 3010 },
		{ 1326, 1325 },
		{ 2016, 1992 },
		{ 182, 173 },
		{ 192, 173 },
		{ 184, 173 },
		{ 1621, 1620 },
		{ 179, 173 },
		{ 1326, 1325 },
		{ 183, 173 },
		{ 3072, 3071 },
		{ 181, 173 },
		{ 2835, 2834 },
		{ 1621, 1620 },
		{ 1695, 1694 },
		{ 190, 173 },
		{ 189, 173 },
		{ 180, 173 },
		{ 188, 173 },
		{ 2497, 2469 },
		{ 185, 173 },
		{ 187, 173 },
		{ 178, 173 },
		{ 2015, 1992 },
		{ 2550, 2521 },
		{ 186, 173 },
		{ 191, 173 },
		{ 100, 83 },
		{ 1668, 1667 },
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
		{ 3037, 3037 },
		{ 3037, 3037 },
		{ 3037, 3037 },
		{ 3037, 3037 },
		{ 3037, 3037 },
		{ 3037, 3037 },
		{ 3037, 3037 },
		{ 3037, 3037 },
		{ 3037, 3037 },
		{ 3037, 3037 },
		{ 1220, 1217 },
		{ 101, 83 },
		{ 1217, 1217 },
		{ 1217, 1217 },
		{ 1217, 1217 },
		{ 1217, 1217 },
		{ 1217, 1217 },
		{ 1217, 1217 },
		{ 1217, 1217 },
		{ 1217, 1217 },
		{ 1217, 1217 },
		{ 1217, 1217 },
		{ 1815, 1791 },
		{ 449, 405 },
		{ 454, 405 },
		{ 451, 405 },
		{ 450, 405 },
		{ 453, 405 },
		{ 448, 405 },
		{ 3043, 65 },
		{ 447, 405 },
		{ 2001, 1979 },
		{ 67, 65 },
		{ 1221, 1217 },
		{ 452, 405 },
		{ 1814, 1791 },
		{ 455, 405 },
		{ 1279, 1278 },
		{ 2876, 2875 },
		{ 2496, 2468 },
		{ 2031, 2010 },
		{ 2929, 2928 },
		{ 446, 405 },
		{ 101, 83 },
		{ 2271, 2239 },
		{ 3038, 3037 },
		{ 2239, 2239 },
		{ 2239, 2239 },
		{ 2239, 2239 },
		{ 2239, 2239 },
		{ 2239, 2239 },
		{ 2239, 2239 },
		{ 2239, 2239 },
		{ 2239, 2239 },
		{ 2239, 2239 },
		{ 2239, 2239 },
		{ 2617, 2592 },
		{ 2060, 2042 },
		{ 2970, 2969 },
		{ 2074, 2057 },
		{ 2993, 2992 },
		{ 3001, 3000 },
		{ 1840, 1821 },
		{ 1862, 1843 },
		{ 1753, 1752 },
		{ 1221, 1217 },
		{ 2204, 2203 },
		{ 2272, 2239 },
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
		{ 2209, 2208 },
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
		{ 2482, 2482 },
		{ 2482, 2482 },
		{ 2410, 2410 },
		{ 2410, 2410 },
		{ 2790, 2789 },
		{ 2454, 2427 },
		{ 3055, 3051 },
		{ 2830, 2829 },
		{ 3041, 65 },
		{ 2272, 2239 },
		{ 115, 100 },
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
		{ 3039, 65 },
		{ 3075, 3074 },
		{ 2482, 2482 },
		{ 3091, 3088 },
		{ 2410, 2410 },
		{ 3097, 3094 },
		{ 2587, 2562 },
		{ 2105, 2090 },
		{ 1355, 1354 },
		{ 3029, 63 },
		{ 118, 102 },
		{ 1223, 1220 },
		{ 67, 63 },
		{ 2165, 2162 },
		{ 2624, 2599 },
		{ 2180, 2179 },
		{ 2635, 2611 },
		{ 2639, 2615 },
		{ 2648, 2625 },
		{ 2394, 2366 },
		{ 2654, 2631 },
		{ 115, 100 },
		{ 3042, 65 },
		{ 1222, 1222 },
		{ 1222, 1222 },
		{ 1222, 1222 },
		{ 1222, 1222 },
		{ 1222, 1222 },
		{ 1222, 1222 },
		{ 1222, 1222 },
		{ 1222, 1222 },
		{ 1222, 1222 },
		{ 1222, 1222 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 2271, 2271 },
		{ 1223, 1220 },
		{ 1226, 1222 },
		{ 2666, 2647 },
		{ 2320, 2320 },
		{ 2320, 2320 },
		{ 1216, 9 },
		{ 2668, 2649 },
		{ 2459, 2459 },
		{ 2459, 2459 },
		{ 67, 9 },
		{ 2682, 2663 },
		{ 2300, 2271 },
		{ 2683, 2664 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 2273, 2273 },
		{ 1246, 1245 },
		{ 1669, 1668 },
		{ 3028, 63 },
		{ 2320, 2320 },
		{ 2693, 2674 },
		{ 1216, 9 },
		{ 3027, 63 },
		{ 2459, 2459 },
		{ 2894, 2894 },
		{ 2894, 2894 },
		{ 1226, 1222 },
		{ 2303, 2273 },
		{ 2941, 2941 },
		{ 2941, 2941 },
		{ 2206, 2205 },
		{ 2512, 2482 },
		{ 2511, 2482 },
		{ 2439, 2410 },
		{ 2438, 2410 },
		{ 1218, 9 },
		{ 2300, 2271 },
		{ 1217, 9 },
		{ 1217, 9 },
		{ 1217, 9 },
		{ 1217, 9 },
		{ 1217, 9 },
		{ 1217, 9 },
		{ 1217, 9 },
		{ 1217, 9 },
		{ 1217, 9 },
		{ 1217, 9 },
		{ 2894, 2894 },
		{ 2461, 2461 },
		{ 2461, 2461 },
		{ 2698, 2681 },
		{ 2941, 2941 },
		{ 2470, 2470 },
		{ 2470, 2470 },
		{ 2340, 2340 },
		{ 2340, 2340 },
		{ 2489, 2489 },
		{ 2489, 2489 },
		{ 2707, 2691 },
		{ 2303, 2273 },
		{ 2365, 2365 },
		{ 2365, 2365 },
		{ 2503, 2503 },
		{ 2503, 2503 },
		{ 2522, 2522 },
		{ 2522, 2522 },
		{ 2574, 2574 },
		{ 2574, 2574 },
		{ 2667, 2667 },
		{ 2667, 2667 },
		{ 1671, 1670 },
		{ 2461, 2461 },
		{ 3025, 63 },
		{ 2716, 2701 },
		{ 3026, 63 },
		{ 2470, 2470 },
		{ 1843, 1824 },
		{ 2340, 2340 },
		{ 2733, 2719 },
		{ 2489, 2489 },
		{ 2746, 2740 },
		{ 2385, 2385 },
		{ 2385, 2385 },
		{ 2365, 2365 },
		{ 2748, 2742 },
		{ 2503, 2503 },
		{ 2754, 2749 },
		{ 2522, 2522 },
		{ 2760, 2759 },
		{ 2574, 2574 },
		{ 2218, 2217 },
		{ 2667, 2667 },
		{ 2350, 2320 },
		{ 2826, 2826 },
		{ 2826, 2826 },
		{ 2854, 2854 },
		{ 2854, 2854 },
		{ 2473, 2447 },
		{ 2464, 2464 },
		{ 2464, 2464 },
		{ 2345, 2345 },
		{ 2345, 2345 },
		{ 2793, 2792 },
		{ 2351, 2320 },
		{ 2385, 2385 },
		{ 2430, 2430 },
		{ 2430, 2430 },
		{ 2485, 2459 },
		{ 2802, 2801 },
		{ 2488, 2488 },
		{ 2488, 2488 },
		{ 2458, 2458 },
		{ 2458, 2458 },
		{ 2211, 2211 },
		{ 2211, 2211 },
		{ 2815, 2814 },
		{ 2826, 2826 },
		{ 1849, 1830 },
		{ 2854, 2854 },
		{ 2277, 2277 },
		{ 2277, 2277 },
		{ 2464, 2464 },
		{ 2475, 2449 },
		{ 2345, 2345 },
		{ 2908, 2908 },
		{ 2908, 2908 },
		{ 2949, 2949 },
		{ 2949, 2949 },
		{ 2430, 2430 },
		{ 2388, 2388 },
		{ 2388, 2388 },
		{ 2895, 2894 },
		{ 2488, 2488 },
		{ 1282, 1281 },
		{ 2458, 2458 },
		{ 2942, 2941 },
		{ 2211, 2211 },
		{ 2838, 2837 },
		{ 2580, 2580 },
		{ 2580, 2580 },
		{ 1869, 1850 },
		{ 1321, 1320 },
		{ 2277, 2277 },
		{ 1896, 1880 },
		{ 2627, 2627 },
		{ 2627, 2627 },
		{ 2498, 2470 },
		{ 2908, 2908 },
		{ 1898, 1883 },
		{ 2949, 2949 },
		{ 2478, 2478 },
		{ 2478, 2478 },
		{ 2388, 2388 },
		{ 1900, 1885 },
		{ 2499, 2470 },
		{ 2487, 2461 },
		{ 2670, 2670 },
		{ 2670, 2670 },
		{ 2348, 2348 },
		{ 2348, 2348 },
		{ 2366, 2340 },
		{ 2580, 2580 },
		{ 2519, 2489 },
		{ 2855, 2854 },
		{ 2863, 2863 },
		{ 2863, 2863 },
		{ 2393, 2365 },
		{ 2627, 2627 },
		{ 2533, 2503 },
		{ 2866, 2865 },
		{ 2551, 2522 },
		{ 2492, 2464 },
		{ 2599, 2574 },
		{ 2478, 2478 },
		{ 2686, 2667 },
		{ 1950, 1948 },
		{ 2604, 2604 },
		{ 2604, 2604 },
		{ 2893, 2892 },
		{ 2670, 2670 },
		{ 2420, 2393 },
		{ 2348, 2348 },
		{ 1697, 1696 },
		{ 2923, 2922 },
		{ 1719, 1718 },
		{ 2932, 2931 },
		{ 2412, 2385 },
		{ 2863, 2863 },
		{ 2490, 2464 },
		{ 2940, 2939 },
		{ 1732, 1731 },
		{ 1267, 1266 },
		{ 2491, 2464 },
		{ 2964, 2963 },
		{ 1616, 1615 },
		{ 2855, 2854 },
		{ 2973, 2972 },
		{ 2984, 2983 },
		{ 2827, 2826 },
		{ 2604, 2604 },
		{ 1741, 1740 },
		{ 2432, 2404 },
		{ 2995, 2994 },
		{ 2372, 2345 },
		{ 2434, 2406 },
		{ 3004, 3003 },
		{ 2072, 2055 },
		{ 2073, 2056 },
		{ 2457, 2430 },
		{ 1344, 1343 },
		{ 2538, 2509 },
		{ 1756, 1755 },
		{ 2518, 2488 },
		{ 2091, 2078 },
		{ 2484, 2458 },
		{ 2446, 2417 },
		{ 2212, 2211 },
		{ 3051, 3047 },
		{ 2557, 2530 },
		{ 2570, 2545 },
		{ 2104, 2089 },
		{ 2307, 2277 },
		{ 3077, 3076 },
		{ 2577, 2552 },
		{ 3079, 3079 },
		{ 2450, 2421 },
		{ 2909, 2908 },
		{ 3104, 3102 },
		{ 2950, 2949 },
		{ 1822, 1797 },
		{ 2723, 2708 },
		{ 2415, 2388 },
		{ 1821, 1797 },
		{ 2337, 2337 },
		{ 2337, 2337 },
		{ 2011, 1986 },
		{ 1930, 1920 },
		{ 1938, 1930 },
		{ 2010, 1986 },
		{ 1315, 1314 },
		{ 2605, 2580 },
		{ 1275, 1274 },
		{ 1713, 1712 },
		{ 2504, 2475 },
		{ 2764, 2763 },
		{ 1714, 1713 },
		{ 2650, 2627 },
		{ 3079, 3079 },
		{ 2785, 2784 },
		{ 1250, 1249 },
		{ 1723, 1722 },
		{ 2797, 2796 },
		{ 2507, 2478 },
		{ 1579, 1573 },
		{ 2520, 2490 },
		{ 2524, 2494 },
		{ 2337, 2337 },
		{ 2032, 2011 },
		{ 2689, 2670 },
		{ 2532, 2501 },
		{ 2375, 2348 },
		{ 2051, 2030 },
		{ 2053, 2032 },
		{ 2056, 2035 },
		{ 1235, 1234 },
		{ 2849, 2847 },
		{ 2864, 2863 },
		{ 1609, 1608 },
		{ 1610, 1609 },
		{ 2558, 2532 },
		{ 2870, 2869 },
		{ 1749, 1748 },
		{ 1350, 1349 },
		{ 1290, 1289 },
		{ 1772, 1771 },
		{ 1773, 1772 },
		{ 2629, 2604 },
		{ 2591, 2566 },
		{ 1778, 1777 },
		{ 2602, 2577 },
		{ 1637, 1636 },
		{ 1638, 1637 },
		{ 1644, 1643 },
		{ 1645, 1644 },
		{ 1297, 1296 },
		{ 2983, 2982 },
		{ 1841, 1822 },
		{ 2467, 2440 },
		{ 1663, 1662 },
		{ 2471, 2445 },
		{ 2655, 2632 },
		{ 2656, 2633 },
		{ 2658, 2636 },
		{ 2659, 2639 },
		{ 1848, 1829 },
		{ 2221, 2220 },
		{ 1664, 1663 },
		{ 2476, 2450 },
		{ 2685, 2666 },
		{ 1860, 1841 },
		{ 1298, 1297 },
		{ 3067, 3066 },
		{ 3068, 3067 },
		{ 1300, 1299 },
		{ 2694, 2675 },
		{ 1573, 1564 },
		{ 1687, 1686 },
		{ 1688, 1687 },
		{ 1314, 1313 },
		{ 1911, 1898 },
		{ 1259, 1258 },
		{ 2787, 2786 },
		{ 3082, 3079 },
		{ 2586, 2561 },
		{ 1352, 1351 },
		{ 1699, 1698 },
		{ 2158, 2151 },
		{ 3081, 3079 },
		{ 2813, 2812 },
		{ 3080, 3079 },
		{ 2601, 2576 },
		{ 2824, 2823 },
		{ 2483, 2457 },
		{ 2428, 2400 },
		{ 2606, 2581 },
		{ 2378, 2351 },
		{ 2161, 2156 },
		{ 1292, 1291 },
		{ 1649, 1648 },
		{ 2636, 2612 },
		{ 2853, 2851 },
		{ 2177, 2174 },
		{ 2363, 2337 },
		{ 2640, 2616 },
		{ 2493, 2465 },
		{ 1945, 1940 },
		{ 1237, 1236 },
		{ 1265, 1264 },
		{ 1833, 1814 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2921, 2920 },
		{ 2208, 2207 },
		{ 1725, 1724 },
		{ 2514, 2484 },
		{ 2672, 2653 },
		{ 2676, 2657 },
		{ 2515, 2485 },
		{ 2962, 2961 },
		{ 2517, 2487 },
		{ 1328, 1327 },
		{ 2214, 2213 },
		{ 1734, 1733 },
		{ 2220, 2219 },
		{ 2022, 2001 },
		{ 1844, 1825 },
		{ 2697, 2680 },
		{ 1342, 1341 },
		{ 2704, 2688 },
		{ 2043, 2022 },
		{ 2547, 2518 },
		{ 1739, 1738 },
		{ 2362, 2362 },
		{ 2718, 2703 },
		{ 1852, 1833 },
		{ 1623, 1622 },
		{ 1673, 1672 },
		{ 2734, 2720 },
		{ 2735, 2722 },
		{ 3058, 3055 },
		{ 1244, 1243 },
		{ 2747, 2741 },
		{ 2565, 2540 },
		{ 1879, 1862 },
		{ 2755, 2753 },
		{ 3079, 3078 },
		{ 2758, 2756 },
		{ 3087, 3084 },
		{ 1346, 1345 },
		{ 3095, 3092 },
		{ 1642, 1641 },
		{ 2578, 2553 },
		{ 3107, 3106 },
		{ 2321, 2321 },
		{ 2321, 2321 },
		{ 2744, 2744 },
		{ 2744, 2744 },
		{ 2411, 2411 },
		{ 2411, 2411 },
		{ 2630, 2605 },
		{ 1883, 1868 },
		{ 2427, 2399 },
		{ 1340, 1339 },
		{ 2740, 2732 },
		{ 2553, 2524 },
		{ 2530, 2499 },
		{ 2649, 2626 },
		{ 2042, 2021 },
		{ 1718, 1717 },
		{ 2536, 2507 },
		{ 1779, 1778 },
		{ 2389, 2362 },
		{ 2616, 2591 },
		{ 2575, 2550 },
		{ 2705, 2689 },
		{ 2865, 2864 },
		{ 2321, 2321 },
		{ 2663, 2643 },
		{ 2744, 2744 },
		{ 2664, 2645 },
		{ 2411, 2411 },
		{ 2545, 2516 },
		{ 1354, 1353 },
		{ 2669, 2650 },
		{ 1811, 1788 },
		{ 2823, 2822 },
		{ 2174, 2172 },
		{ 1657, 1656 },
		{ 2657, 2635 },
		{ 1947, 1944 },
		{ 1345, 1344 },
		{ 2840, 2839 },
		{ 1953, 1952 },
		{ 1826, 1803 },
		{ 1810, 1788 },
		{ 2535, 2506 },
		{ 2455, 2428 },
		{ 1464, 1444 },
		{ 2543, 2514 },
		{ 1284, 1283 },
		{ 1407, 1383 },
		{ 1631, 1630 },
		{ 2390, 2362 },
		{ 2868, 2867 },
		{ 2017, 1994 },
		{ 2875, 2874 },
		{ 2021, 2000 },
		{ 2399, 2372 },
		{ 2692, 2673 },
		{ 1672, 1671 },
		{ 1845, 1826 },
		{ 1847, 1828 },
		{ 2925, 2924 },
		{ 2036, 2015 },
		{ 2572, 2547 },
		{ 2934, 2933 },
		{ 2038, 2017 },
		{ 2040, 2019 },
		{ 1339, 1338 },
		{ 1681, 1680 },
		{ 1248, 1247 },
		{ 2966, 2965 },
		{ 1429, 1407 },
		{ 2722, 2707 },
		{ 2975, 2974 },
		{ 1643, 1642 },
		{ 2724, 2709 },
		{ 2588, 2563 },
		{ 1258, 1257 },
		{ 2071, 2054 },
		{ 2997, 2996 },
		{ 1878, 1861 },
		{ 1758, 1757 },
		{ 3006, 3005 },
		{ 2741, 2733 },
		{ 1766, 1765 },
		{ 1308, 1307 },
		{ 2612, 2587 },
		{ 2086, 2070 },
		{ 2753, 2748 },
		{ 1893, 1877 },
		{ 1698, 1697 },
		{ 2625, 2600 },
		{ 3056, 3052 },
		{ 1646, 1645 },
		{ 2762, 2761 },
		{ 2136, 2118 },
		{ 2137, 2119 },
		{ 1707, 1706 },
		{ 2352, 2321 },
		{ 2513, 2483 },
		{ 2749, 2744 },
		{ 3078, 3077 },
		{ 2440, 2411 },
		{ 1603, 1602 },
		{ 2163, 2160 },
		{ 3084, 3081 },
		{ 2795, 2794 },
		{ 1357, 1356 },
		{ 1717, 1716 },
		{ 2652, 2629 },
		{ 2171, 2168 },
		{ 3106, 3104 },
		{ 2817, 2816 },
		{ 2916, 2916 },
		{ 2916, 2916 },
		{ 2808, 2808 },
		{ 2808, 2808 },
		{ 2957, 2957 },
		{ 2957, 2957 },
		{ 2453, 2453 },
		{ 2453, 2453 },
		{ 2477, 2477 },
		{ 2477, 2477 },
		{ 1249, 1248 },
		{ 2160, 2155 },
		{ 2850, 2848 },
		{ 1842, 1823 },
		{ 2552, 2523 },
		{ 1954, 1953 },
		{ 2554, 2525 },
		{ 2556, 2529 },
		{ 1744, 1743 },
		{ 2869, 2868 },
		{ 1285, 1284 },
		{ 2472, 2446 },
		{ 1307, 1306 },
		{ 2916, 2916 },
		{ 2877, 2876 },
		{ 2808, 2808 },
		{ 2886, 2885 },
		{ 2957, 2957 },
		{ 2173, 2171 },
		{ 2453, 2453 },
		{ 1754, 1753 },
		{ 2477, 2477 },
		{ 2903, 2902 },
		{ 2904, 2903 },
		{ 2906, 2905 },
		{ 1656, 1655 },
		{ 2708, 2692 },
		{ 2919, 2918 },
		{ 1706, 1705 },
		{ 2019, 1997 },
		{ 2020, 1999 },
		{ 2926, 2925 },
		{ 1850, 1831 },
		{ 2930, 2929 },
		{ 1759, 1758 },
		{ 1765, 1764 },
		{ 2935, 2934 },
		{ 2210, 2209 },
		{ 1614, 1613 },
		{ 2947, 2946 },
		{ 2033, 2012 },
		{ 1337, 1336 },
		{ 2960, 2959 },
		{ 2737, 2724 },
		{ 1484, 1465 },
		{ 113, 98 },
		{ 2967, 2966 },
		{ 2304, 2274 },
		{ 2971, 2970 },
		{ 2613, 2588 },
		{ 2615, 2590 },
		{ 2976, 2975 },
		{ 2982, 2981 },
		{ 1880, 1863 },
		{ 1630, 1629 },
		{ 1280, 1279 },
		{ 1885, 1870 },
		{ 2628, 2603 },
		{ 2055, 2034 },
		{ 2998, 2997 },
		{ 1270, 1269 },
		{ 3002, 3001 },
		{ 2763, 2762 },
		{ 1895, 1879 },
		{ 3007, 3006 },
		{ 1408, 1385 },
		{ 1680, 1679 },
		{ 1358, 1357 },
		{ 3022, 3018 },
		{ 1319, 1318 },
		{ 2791, 2790 },
		{ 1919, 1907 },
		{ 2078, 2061 },
		{ 3049, 3045 },
		{ 2796, 2795 },
		{ 3052, 3048 },
		{ 1828, 1806 },
		{ 2521, 2491 },
		{ 1932, 1923 },
		{ 3061, 3058 },
		{ 2811, 2810 },
		{ 2377, 2350 },
		{ 1832, 1813 },
		{ 2379, 2352 },
		{ 2917, 2916 },
		{ 2818, 2817 },
		{ 2809, 2808 },
		{ 1944, 1939 },
		{ 2958, 2957 },
		{ 2116, 2102 },
		{ 2479, 2453 },
		{ 1444, 1422 },
		{ 2506, 2477 },
		{ 2537, 2508 },
		{ 1602, 1601 },
		{ 2836, 2835 },
		{ 2138, 2120 },
		{ 2149, 2136 },
		{ 2841, 2840 },
		{ 2150, 2137 },
		{ 2679, 2660 },
		{ 1277, 1277 },
		{ 1277, 1277 },
		{ 1751, 1751 },
		{ 1751, 1751 },
		{ 2833, 2833 },
		{ 2833, 2833 },
		{ 2968, 2968 },
		{ 2968, 2968 },
		{ 2702, 2702 },
		{ 2702, 2702 },
		{ 2299, 2299 },
		{ 2299, 2299 },
		{ 2927, 2927 },
		{ 2927, 2927 },
		{ 2999, 2999 },
		{ 2999, 2999 },
		{ 2510, 2510 },
		{ 2510, 2510 },
		{ 2788, 2788 },
		{ 2788, 2788 },
		{ 2373, 2373 },
		{ 2373, 2373 },
		{ 1866, 1847 },
		{ 1277, 1277 },
		{ 1309, 1308 },
		{ 1751, 1751 },
		{ 2418, 2391 },
		{ 2833, 2833 },
		{ 1682, 1681 },
		{ 2968, 2968 },
		{ 3059, 3056 },
		{ 2702, 2702 },
		{ 1721, 1720 },
		{ 2299, 2299 },
		{ 2216, 2215 },
		{ 2927, 2927 },
		{ 1767, 1766 },
		{ 2999, 2999 },
		{ 2087, 2071 },
		{ 2510, 2510 },
		{ 2481, 2455 },
		{ 2788, 2788 },
		{ 2166, 2163 },
		{ 2373, 2373 },
		{ 1483, 1464 },
		{ 1658, 1657 },
		{ 1708, 1707 },
		{ 2607, 2582 },
		{ 1632, 1631 },
		{ 1894, 1878 },
		{ 3094, 3091 },
		{ 2182, 2181 },
		{ 2059, 2040 },
		{ 1949, 1947 },
		{ 1604, 1603 },
		{ 1667, 1666 },
		{ 2988, 2988 },
		{ 2988, 2988 },
		{ 2952, 2952 },
		{ 2952, 2952 },
		{ 1746, 1746 },
		{ 1746, 1746 },
		{ 2911, 2911 },
		{ 2911, 2911 },
		{ 2803, 2803 },
		{ 2803, 2803 },
		{ 1735, 1735 },
		{ 1735, 1735 },
		{ 2945, 2945 },
		{ 2945, 2945 },
		{ 2205, 2204 },
		{ 1261, 1261 },
		{ 1261, 1261 },
		{ 1272, 1272 },
		{ 1272, 1272 },
		{ 1910, 1897 },
		{ 1619, 1618 },
		{ 2030, 2009 },
		{ 1295, 1294 },
		{ 2988, 2988 },
		{ 1922, 1912 },
		{ 2952, 2952 },
		{ 1853, 1834 },
		{ 1746, 1746 },
		{ 2135, 2117 },
		{ 2911, 2911 },
		{ 1694, 1693 },
		{ 2803, 2803 },
		{ 1312, 1311 },
		{ 1735, 1735 },
		{ 1864, 1845 },
		{ 2945, 2945 },
		{ 1548, 1522 },
		{ 1278, 1277 },
		{ 1261, 1261 },
		{ 1752, 1751 },
		{ 1272, 1272 },
		{ 2834, 2833 },
		{ 2585, 2560 },
		{ 2969, 2968 },
		{ 1607, 1606 },
		{ 2717, 2702 },
		{ 2739, 2731 },
		{ 2329, 2299 },
		{ 2662, 2642 },
		{ 2928, 2927 },
		{ 2387, 2360 },
		{ 3000, 2999 },
		{ 1829, 1809 },
		{ 2539, 2510 },
		{ 2589, 2564 },
		{ 2789, 2788 },
		{ 1324, 1323 },
		{ 2400, 2373 },
		{ 1635, 1634 },
		{ 2600, 2575 },
		{ 1274, 1273 },
		{ 2480, 2454 },
		{ 3065, 3064 },
		{ 2057, 2036 },
		{ 2861, 2860 },
		{ 2058, 2038 },
		{ 3074, 3073 },
		{ 2684, 2665 },
		{ 1451, 1429 },
		{ 2170, 2167 },
		{ 1770, 1769 },
		{ 1711, 1710 },
		{ 1661, 1660 },
		{ 1685, 1684 },
		{ 1242, 1241 },
		{ 1780, 1779 },
		{ 2495, 2467 },
		{ 2077, 2060 },
		{ 3100, 3097 },
		{ 2992, 2991 },
		{ 1748, 1747 },
		{ 2706, 2690 },
		{ 2596, 2596 },
		{ 2596, 2596 },
		{ 2857, 2856 },
		{ 1238, 1238 },
		{ 1238, 1238 },
		{ 2460, 2460 },
		{ 2460, 2460 },
		{ 2594, 2594 },
		{ 2594, 2594 },
		{ 2884, 2883 },
		{ 2989, 2988 },
		{ 1383, 1374 },
		{ 2953, 2952 },
		{ 1803, 1959 },
		{ 1747, 1746 },
		{ 1994, 2185 },
		{ 2912, 2911 },
		{ 1613, 1612 },
		{ 2804, 2803 },
		{ 1743, 1742 },
		{ 1736, 1735 },
		{ 0, 1218 },
		{ 2946, 2945 },
		{ 2596, 2596 },
		{ 0, 84 },
		{ 1262, 1261 },
		{ 1238, 1238 },
		{ 1273, 1272 },
		{ 2460, 2460 },
		{ 0, 2240 },
		{ 2594, 2594 },
		{ 2302, 2302 },
		{ 2302, 2302 },
		{ 2302, 2302 },
		{ 2302, 2302 },
		{ 2302, 2302 },
		{ 2302, 2302 },
		{ 2302, 2302 },
		{ 2302, 2302 },
		{ 2302, 2302 },
		{ 2302, 2302 },
		{ 1293, 1293 },
		{ 1293, 1293 },
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
		{ 0, 1218 },
		{ 2879, 2879 },
		{ 2879, 2879 },
		{ 0, 84 },
		{ 1269, 1268 },
		{ 2816, 2815 },
		{ 1615, 1614 },
		{ 2603, 2578 },
		{ 0, 2240 },
		{ 1465, 1445 },
		{ 2965, 2964 },
		{ 1293, 1293 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 2331, 2331 },
		{ 1812, 1790 },
		{ 1617, 1616 },
		{ 2879, 2879 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 2333, 2333 },
		{ 1322, 1321 },
		{ 2701, 2684 },
		{ 2417, 2390 },
		{ 2621, 2596 },
		{ 1952, 1950 },
		{ 2974, 2973 },
		{ 1239, 1238 },
		{ 1620, 1619 },
		{ 2486, 2460 },
		{ 2168, 2165 },
		{ 2619, 2594 },
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
		{ 3013, 3013 },
		{ 3013, 3013 },
		{ 3013, 3013 },
		{ 3013, 3013 },
		{ 3013, 3013 },
		{ 3013, 3013 },
		{ 3013, 3013 },
		{ 3013, 3013 },
		{ 3013, 3013 },
		{ 3013, 3013 },
		{ 1225, 1225 },
		{ 1225, 1225 },
		{ 1225, 1225 },
		{ 1225, 1225 },
		{ 1225, 1225 },
		{ 1225, 1225 },
		{ 1225, 1225 },
		{ 1225, 1225 },
		{ 1225, 1225 },
		{ 1225, 1225 },
		{ 2583, 2583 },
		{ 2583, 2583 },
		{ 1294, 1293 },
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
		{ 1813, 1790 },
		{ 2839, 2838 },
		{ 2880, 2879 },
		{ 1720, 1719 },
		{ 2709, 2693 },
		{ 1356, 1355 },
		{ 1830, 1810 },
		{ 1757, 1756 },
		{ 3099, 3099 },
		{ 2548, 2519 },
		{ 2583, 2583 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 1228, 1228 },
		{ 1230, 1230 },
		{ 1230, 1230 },
		{ 1230, 1230 },
		{ 1230, 1230 },
		{ 1230, 1230 },
		{ 1230, 1230 },
		{ 1230, 1230 },
		{ 1230, 1230 },
		{ 1230, 1230 },
		{ 1230, 1230 },
		{ 3099, 3099 },
		{ 2898, 2898 },
		{ 2898, 2898 },
		{ 2819, 2819 },
		{ 2819, 2819 },
		{ 2567, 2567 },
		{ 2567, 2567 },
		{ 2626, 2601 },
		{ 2996, 2995 },
		{ 1998, 1976 },
		{ 2179, 2177 },
		{ 1884, 1869 },
		{ 2860, 2859 },
		{ 2731, 2717 },
		{ 2732, 2718 },
		{ 3005, 3004 },
		{ 2632, 2607 },
		{ 1722, 1721 },
		{ 2867, 2866 },
		{ 1834, 1815 },
		{ 1283, 1282 },
		{ 1325, 1324 },
		{ 2642, 2618 },
		{ 1318, 1317 },
		{ 2898, 2898 },
		{ 2088, 2072 },
		{ 2819, 2819 },
		{ 2559, 2533 },
		{ 2567, 2567 },
		{ 2562, 2537 },
		{ 2885, 2884 },
		{ 2563, 2538 },
		{ 2564, 2539 },
		{ 1384, 1363 },
		{ 2391, 2363 },
		{ 2902, 2901 },
		{ 2102, 2086 },
		{ 1247, 1246 },
		{ 2905, 2904 },
		{ 2761, 2760 },
		{ 2660, 2640 },
		{ 2447, 2418 },
		{ 1907, 1893 },
		{ 2360, 2329 },
		{ 2665, 2646 },
		{ 2582, 2557 },
		{ 2608, 2583 },
		{ 2924, 2923 },
		{ 1336, 1335 },
		{ 2118, 2104 },
		{ 2119, 2105 },
		{ 3087, 3087 },
		{ 2673, 2654 },
		{ 2794, 2793 },
		{ 1320, 1319 },
		{ 2933, 2932 },
		{ 1777, 1776 },
		{ 2681, 2662 },
		{ 1353, 1352 },
		{ 2404, 2377 },
		{ 2406, 2379 },
		{ 2807, 2806 },
		{ 2943, 2942 },
		{ 2881, 2880 },
		{ 2595, 2570 },
		{ 2181, 2180 },
		{ 67, 5 },
		{ 2956, 2955 },
		{ 1742, 1741 },
		{ 3099, 3096 },
		{ 3101, 3099 },
		{ 2915, 2914 },
		{ 2699, 2682 },
		{ 2896, 2895 },
		{ 3087, 3087 },
		{ 2700, 2683 },
		{ 1999, 1976 },
		{ 1268, 1267 },
		{ 2887, 2886 },
		{ 1335, 1334 },
		{ 2907, 2906 },
		{ 1745, 1744 },
		{ 2172, 2170 },
		{ 2542, 2513 },
		{ 2901, 2900 },
		{ 2651, 2628 },
		{ 2696, 2679 },
		{ 1271, 1270 },
		{ 2847, 2845 },
		{ 3044, 3039 },
		{ 2899, 2898 },
		{ 1823, 1800 },
		{ 2820, 2819 },
		{ 1257, 1256 },
		{ 2592, 2567 },
		{ 1824, 1800 },
		{ 1731, 1730 },
		{ 2494, 2466 },
		{ 2466, 2439 },
		{ 2541, 2512 },
		{ 1385, 1363 },
		{ 2910, 2909 },
		{ 2187, 2184 },
		{ 1961, 1958 },
		{ 2848, 2845 },
		{ 2951, 2950 },
		{ 2661, 2641 },
		{ 2186, 2184 },
		{ 1960, 1958 },
		{ 2202, 2201 },
		{ 2822, 2821 },
		{ 2336, 2307 },
		{ 2641, 2617 },
		{ 2859, 2858 },
		{ 2402, 2375 },
		{ 2566, 2541 },
		{ 1805, 1785 },
		{ 2444, 2415 },
		{ 174, 5 },
		{ 3045, 3039 },
		{ 1445, 1423 },
		{ 1804, 1785 },
		{ 175, 5 },
		{ 1996, 1973 },
		{ 1401, 1375 },
		{ 2900, 2899 },
		{ 2720, 2705 },
		{ 1605, 1604 },
		{ 1995, 1973 },
		{ 176, 5 },
		{ 1310, 1309 },
		{ 2598, 2573 },
		{ 2405, 2378 },
		{ 1327, 1326 },
		{ 1951, 1949 },
		{ 1240, 1239 },
		{ 2120, 2106 },
		{ 1750, 1749 },
		{ 1648, 1647 },
		{ 2274, 2245 },
		{ 2918, 2917 },
		{ 1705, 1704 },
		{ 2920, 2919 },
		{ 3090, 3087 },
		{ 2609, 2584 },
		{ 173, 5 },
		{ 2745, 2739 },
		{ 2501, 2472 },
		{ 2414, 2387 },
		{ 2614, 2589 },
		{ 1334, 1333 },
		{ 1655, 1654 },
		{ 98, 81 },
		{ 2508, 2479 },
		{ 2756, 2754 },
		{ 2151, 2138 },
		{ 2155, 2148 },
		{ 1709, 1708 },
		{ 1997, 1974 },
		{ 2516, 2486 },
		{ 2944, 2943 },
		{ 1260, 1259 },
		{ 1764, 1763 },
		{ 1859, 1840 },
		{ 2786, 2785 },
		{ 1256, 1255 },
		{ 1659, 1658 },
		{ 2523, 2493 },
		{ 2959, 2958 },
		{ 1863, 1844 },
		{ 2961, 2960 },
		{ 2169, 2166 },
		{ 2529, 2498 },
		{ 2643, 2619 },
		{ 2645, 2621 },
		{ 2433, 2405 },
		{ 1504, 1483 },
		{ 1768, 1767 },
		{ 1867, 1848 },
		{ 1505, 1484 },
		{ 2810, 2809 },
		{ 2653, 2630 },
		{ 2812, 2811 },
		{ 1870, 1852 },
		{ 1299, 1298 },
		{ 2441, 2412 },
		{ 2985, 2984 },
		{ 1276, 1275 },
		{ 2034, 2013 },
		{ 2035, 2014 },
		{ 2821, 2820 },
		{ 2544, 2515 },
		{ 1622, 1621 },
		{ 2546, 2517 },
		{ 2825, 2824 },
		{ 1341, 1340 },
		{ 2828, 2827 },
		{ 1882, 1866 },
		{ 2832, 2831 },
		{ 1629, 1628 },
		{ 2201, 2200 },
		{ 1724, 1723 },
		{ 1306, 1305 },
		{ 3018, 3009 },
		{ 2052, 2031 },
		{ 1236, 1235 },
		{ 2675, 2656 },
		{ 2555, 2526 },
		{ 1679, 1678 },
		{ 2680, 2661 },
		{ 1806, 1786 },
		{ 1807, 1787 },
		{ 1808, 1787 },
		{ 1733, 1732 },
		{ 3047, 3042 },
		{ 3048, 3044 },
		{ 2851, 2849 },
		{ 2561, 2536 },
		{ 1422, 1400 },
		{ 1633, 1632 },
		{ 2858, 2857 },
		{ 2688, 2669 },
		{ 2217, 2216 },
		{ 2690, 2671 },
		{ 3062, 3059 },
		{ 2862, 2861 },
		{ 2061, 2043 },
		{ 2219, 2218 },
		{ 3071, 3070 },
		{ 2569, 2544 },
		{ 2069, 2051 },
		{ 2571, 2546 },
		{ 1423, 1401 },
		{ 1738, 1737 },
		{ 1683, 1682 },
		{ 1601, 1600 },
		{ 3083, 3080 },
		{ 1923, 1913 },
		{ 2581, 2556 },
		{ 2076, 2059 },
		{ 2882, 2881 },
		{ 3092, 3089 },
		{ 1291, 1290 },
		{ 1264, 1263 },
		{ 1831, 1811 },
		{ 1939, 1931 },
		{ 1940, 1932 },
		{ 1691, 1690 },
		{ 2590, 2565 },
		{ 2897, 2896 },
		{ 1349, 1348 },
		{ 2468, 2441 },
		{ 1861, 1842 },
		{ 2334, 2304 },
		{ 2054, 2033 },
		{ 1851, 1832 },
		{ 2579, 2554 },
		{ 1871, 1853 },
		{ 2829, 2828 },
		{ 1809, 1787 },
		{ 2948, 2947 },
		{ 3089, 3086 },
		{ 3053, 3049 },
		{ 1430, 1408 },
		{ 2852, 2850 },
		{ 2671, 2652 },
		{ 2041, 2020 },
		{ 2878, 2877 },
		{ 2013, 1990 },
		{ 2987, 2986 },
		{ 1348, 1347 },
		{ 128, 113 },
		{ 1776, 1775 },
		{ 3024, 3022 },
		{ 2914, 2913 },
		{ 2576, 2551 },
		{ 2831, 2830 },
		{ 1281, 1280 },
		{ 1351, 1350 },
		{ 1241, 1240 },
		{ 2972, 2971 },
		{ 3085, 3082 },
		{ 3003, 3002 },
		{ 3088, 3085 },
		{ 2526, 2496 },
		{ 2837, 2836 },
		{ 614, 556 },
		{ 2806, 2805 },
		{ 2103, 2088 },
		{ 2955, 2954 },
		{ 3096, 3093 },
		{ 2703, 2686 },
		{ 2883, 2882 },
		{ 1641, 1640 },
		{ 2931, 2930 },
		{ 3103, 3101 },
		{ 615, 556 },
		{ 1755, 1754 },
		{ 2792, 2791 },
		{ 1899, 1884 },
		{ 2568, 2543 },
		{ 1383, 1376 },
		{ 1994, 1988 },
		{ 1333, 1330 },
		{ 1803, 1799 },
		{ 2597, 2572 },
		{ 2618, 2593 },
		{ 2646, 2622 },
		{ 202, 179 },
		{ 2647, 2624 },
		{ 2421, 2394 },
		{ 200, 179 },
		{ 616, 556 },
		{ 201, 179 },
		{ 2856, 2855 },
		{ 2593, 2568 },
		{ 1825, 1801 },
		{ 2465, 2438 },
		{ 2009, 1985 },
		{ 2509, 2480 },
		{ 1712, 1711 },
		{ 1670, 1669 },
		{ 199, 179 },
		{ 1564, 1548 },
		{ 2012, 1989 },
		{ 2805, 2804 },
		{ 2117, 2103 },
		{ 1781, 1780 },
		{ 2560, 2535 },
		{ 2719, 2704 },
		{ 1948, 1945 },
		{ 1693, 1692 },
		{ 2611, 2586 },
		{ 2954, 2953 },
		{ 2814, 2813 },
		{ 1323, 1322 },
		{ 1313, 1312 },
		{ 1696, 1695 },
		{ 1243, 1242 },
		{ 2525, 2495 },
		{ 1618, 1617 },
		{ 2963, 2962 },
		{ 2203, 2202 },
		{ 3066, 3065 },
		{ 1662, 1661 },
		{ 2674, 2655 },
		{ 1608, 1607 },
		{ 2622, 2597 },
		{ 3073, 3072 },
		{ 2742, 2734 },
		{ 1912, 1899 },
		{ 3076, 3075 },
		{ 2156, 2149 },
		{ 1296, 1295 },
		{ 1920, 1910 },
		{ 2449, 2420 },
		{ 1771, 1770 },
		{ 2631, 2606 },
		{ 2213, 2212 },
		{ 3086, 3083 },
		{ 2633, 2608 },
		{ 2540, 2511 },
		{ 2986, 2985 },
		{ 2759, 2758 },
		{ 2913, 2912 },
		{ 2162, 2158 },
		{ 3093, 3090 },
		{ 2991, 2990 },
		{ 2691, 2672 },
		{ 2215, 2214 },
		{ 2994, 2993 },
		{ 1636, 1635 },
		{ 1686, 1685 },
		{ 1245, 1244 },
		{ 3102, 3100 },
		{ 1343, 1342 },
		{ 2922, 2921 },
		{ 2089, 2074 },
		{ 2090, 2077 },
		{ 851, 800 },
		{ 422, 381 },
		{ 674, 613 },
		{ 740, 683 },
		{ 1627, 25 },
		{ 1703, 31 },
		{ 1233, 11 },
		{ 67, 25 },
		{ 67, 31 },
		{ 67, 11 },
		{ 2783, 45 },
		{ 1332, 19 },
		{ 1729, 33 },
		{ 67, 45 },
		{ 67, 19 },
		{ 67, 33 },
		{ 1677, 29 },
		{ 423, 381 },
		{ 2890, 55 },
		{ 67, 29 },
		{ 739, 683 },
		{ 67, 55 },
		{ 1304, 17 },
		{ 1117, 1098 },
		{ 675, 613 },
		{ 67, 17 },
		{ 852, 800 },
		{ 1653, 27 },
		{ 2980, 59 },
		{ 1254, 13 },
		{ 67, 27 },
		{ 67, 59 },
		{ 67, 13 },
		{ 1118, 1099 },
		{ 1136, 1119 },
		{ 1141, 1125 },
		{ 1149, 1137 },
		{ 1161, 1150 },
		{ 1167, 1156 },
		{ 1202, 1201 },
		{ 271, 232 },
		{ 278, 239 },
		{ 293, 251 },
		{ 300, 258 },
		{ 306, 263 },
		{ 315, 272 },
		{ 2023, 2002 },
		{ 332, 288 },
		{ 335, 291 },
		{ 342, 297 },
		{ 343, 298 },
		{ 348, 303 },
		{ 1835, 1816 },
		{ 361, 319 },
		{ 385, 344 },
		{ 388, 347 },
		{ 394, 353 },
		{ 405, 365 },
		{ 412, 372 },
		{ 2045, 2024 },
		{ 2046, 2025 },
		{ 421, 380 },
		{ 230, 195 },
		{ 435, 391 },
		{ 234, 199 },
		{ 456, 406 },
		{ 459, 410 },
		{ 469, 418 },
		{ 1855, 1836 },
		{ 1856, 1837 },
		{ 470, 419 },
		{ 483, 432 },
		{ 2068, 2050 },
		{ 491, 441 },
		{ 524, 465 },
		{ 534, 473 },
		{ 536, 475 },
		{ 537, 476 },
		{ 541, 480 },
		{ 553, 494 },
		{ 557, 498 },
		{ 561, 502 },
		{ 2084, 2067 },
		{ 1876, 1858 },
		{ 571, 512 },
		{ 586, 525 },
		{ 1625, 25 },
		{ 1701, 31 },
		{ 1231, 11 },
		{ 587, 527 },
		{ 592, 532 },
		{ 235, 200 },
		{ 2781, 45 },
		{ 1330, 19 },
		{ 1727, 33 },
		{ 623, 560 },
		{ 636, 573 },
		{ 639, 576 },
		{ 1675, 29 },
		{ 1891, 1875 },
		{ 2889, 55 },
		{ 648, 585 },
		{ 663, 599 },
		{ 664, 600 },
		{ 1302, 17 },
		{ 673, 612 },
		{ 241, 206 },
		{ 693, 631 },
		{ 242, 207 },
		{ 1651, 27 },
		{ 2978, 59 },
		{ 1252, 13 },
		{ 741, 684 },
		{ 753, 695 },
		{ 755, 697 },
		{ 757, 699 },
		{ 762, 704 },
		{ 782, 724 },
		{ 791, 733 },
		{ 795, 737 },
		{ 796, 738 },
		{ 826, 772 },
		{ 843, 792 },
		{ 262, 223 },
		{ 881, 832 },
		{ 889, 840 },
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
		{ 2096, 2082 },
		{ 2098, 2083 },
		{ 2157, 2150 },
		{ 216, 187 },
		{ 3037, 3036 },
		{ 461, 412 },
		{ 463, 412 },
		{ 2097, 2082 },
		{ 2099, 2083 },
		{ 431, 386 },
		{ 430, 385 },
		{ 578, 519 },
		{ 404, 364 },
		{ 806, 748 },
		{ 493, 443 },
		{ 716, 660 },
		{ 464, 412 },
		{ 2094, 2080 },
		{ 219, 187 },
		{ 217, 187 },
		{ 1903, 1889 },
		{ 462, 412 },
		{ 501, 450 },
		{ 502, 450 },
		{ 717, 661 },
		{ 672, 611 },
		{ 803, 745 },
		{ 288, 246 },
		{ 212, 184 },
		{ 328, 284 },
		{ 503, 450 },
		{ 211, 184 },
		{ 514, 459 },
		{ 1098, 1073 },
		{ 2006, 1982 },
		{ 2027, 2006 },
		{ 437, 393 },
		{ 213, 184 },
		{ 354, 309 },
		{ 620, 557 },
		{ 895, 846 },
		{ 516, 459 },
		{ 2005, 1982 },
		{ 619, 557 },
		{ 896, 847 },
		{ 205, 181 },
		{ 515, 459 },
		{ 207, 181 },
		{ 225, 190 },
		{ 808, 750 },
		{ 206, 181 },
		{ 618, 557 },
		{ 617, 557 },
		{ 508, 454 },
		{ 2004, 1982 },
		{ 1107, 1083 },
		{ 509, 454 },
		{ 415, 374 },
		{ 414, 374 },
		{ 505, 452 },
		{ 224, 190 },
		{ 265, 226 },
		{ 1838, 1819 },
		{ 256, 218 },
		{ 297, 255 },
		{ 891, 842 },
		{ 583, 524 },
		{ 1287, 15 },
		{ 2799, 47 },
		{ 1598, 23 },
		{ 584, 524 },
		{ 2845, 51 },
		{ 2028, 2007 },
		{ 1106, 1083 },
		{ 2872, 53 },
		{ 1761, 35 },
		{ 2198, 41 },
		{ 2937, 57 },
		{ 570, 511 },
		{ 506, 452 },
		{ 222, 188 },
		{ 585, 524 },
		{ 704, 646 },
		{ 220, 188 },
		{ 807, 749 },
		{ 521, 463 },
		{ 221, 188 },
		{ 279, 240 },
		{ 707, 649 },
		{ 1041, 1009 },
		{ 787, 729 },
		{ 559, 500 },
		{ 1064, 1033 },
		{ 538, 477 },
		{ 3035, 3033 },
		{ 737, 681 },
		{ 738, 682 },
		{ 1818, 1794 },
		{ 236, 201 },
		{ 3046, 3041 },
		{ 522, 463 },
		{ 417, 376 },
		{ 280, 240 },
		{ 2784, 2781 },
		{ 951, 907 },
		{ 955, 912 },
		{ 687, 625 },
		{ 812, 754 },
		{ 2445, 2416 },
		{ 603, 542 },
		{ 3057, 3054 },
		{ 1132, 1115 },
		{ 479, 428 },
		{ 1289, 1287 },
		{ 1139, 1122 },
		{ 705, 647 },
		{ 861, 811 },
		{ 1234, 1231 },
		{ 1157, 1146 },
		{ 1158, 1147 },
		{ 1159, 1148 },
		{ 550, 491 },
		{ 208, 182 },
		{ 397, 356 },
		{ 823, 768 },
		{ 824, 769 },
		{ 1150, 1138 },
		{ 209, 182 },
		{ 1837, 1818 },
		{ 1152, 1140 },
		{ 545, 485 },
		{ 834, 782 },
		{ 840, 789 },
		{ 546, 486 },
		{ 1165, 1154 },
		{ 549, 491 },
		{ 1166, 1155 },
		{ 295, 253 },
		{ 1171, 1162 },
		{ 1186, 1177 },
		{ 859, 809 },
		{ 1213, 1212 },
		{ 2025, 2004 },
		{ 551, 492 },
		{ 866, 816 },
		{ 876, 827 },
		{ 344, 299 },
		{ 690, 628 },
		{ 347, 302 },
		{ 413, 373 },
		{ 484, 433 },
		{ 565, 506 },
		{ 3036, 3035 },
		{ 905, 856 },
		{ 911, 862 },
		{ 917, 869 },
		{ 922, 874 },
		{ 927, 881 },
		{ 932, 887 },
		{ 939, 893 },
		{ 708, 650 },
		{ 3050, 3046 },
		{ 569, 510 },
		{ 263, 224 },
		{ 718, 662 },
		{ 276, 237 },
		{ 961, 918 },
		{ 972, 933 },
		{ 576, 517 },
		{ 980, 941 },
		{ 3060, 3057 },
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
		{ 339, 294 },
		{ 1125, 1107 },
		{ 1126, 1109 },
		{ 391, 350 },
		{ 1137, 1120 },
		{ 1069, 1038 },
		{ 196, 178 },
		{ 2063, 2045 },
		{ 337, 293 },
		{ 336, 293 },
		{ 198, 178 },
		{ 229, 194 },
		{ 669, 605 },
		{ 574, 515 },
		{ 362, 320 },
		{ 931, 886 },
		{ 1594, 1594 },
		{ 197, 178 },
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
		{ 596, 536 },
		{ 1130, 1113 },
		{ 597, 537 },
		{ 548, 490 },
		{ 244, 209 },
		{ 606, 545 },
		{ 726, 672 },
		{ 1873, 1855 },
		{ 995, 957 },
		{ 996, 958 },
		{ 997, 960 },
		{ 1594, 1594 },
		{ 1527, 1527 },
		{ 1530, 1530 },
		{ 1533, 1533 },
		{ 1536, 1536 },
		{ 1539, 1539 },
		{ 1542, 1542 },
		{ 846, 795 },
		{ 847, 796 },
		{ 1005, 970 },
		{ 408, 368 },
		{ 611, 553 },
		{ 1021, 987 },
		{ 370, 328 },
		{ 322, 278 },
		{ 1025, 991 },
		{ 1196, 1190 },
		{ 1560, 1560 },
		{ 240, 205 },
		{ 355, 312 },
		{ 886, 837 },
		{ 888, 839 },
		{ 630, 567 },
		{ 1569, 1569 },
		{ 1527, 1527 },
		{ 1530, 1530 },
		{ 1533, 1533 },
		{ 1536, 1536 },
		{ 1539, 1539 },
		{ 1542, 1542 },
		{ 1045, 1012 },
		{ 563, 504 },
		{ 1490, 1490 },
		{ 389, 348 },
		{ 1061, 1029 },
		{ 646, 583 },
		{ 533, 472 },
		{ 228, 193 },
		{ 767, 709 },
		{ 1400, 1594 },
		{ 1560, 1560 },
		{ 906, 857 },
		{ 1081, 1054 },
		{ 1084, 1057 },
		{ 1085, 1058 },
		{ 909, 860 },
		{ 1569, 1569 },
		{ 773, 715 },
		{ 535, 474 },
		{ 919, 871 },
		{ 1097, 1072 },
		{ 920, 872 },
		{ 1099, 1074 },
		{ 921, 873 },
		{ 478, 427 },
		{ 1490, 1490 },
		{ 579, 520 },
		{ 1875, 1857 },
		{ 3032, 3028 },
		{ 215, 186 },
		{ 392, 351 },
		{ 323, 279 },
		{ 489, 438 },
		{ 1400, 1527 },
		{ 1400, 1530 },
		{ 1400, 1533 },
		{ 1400, 1536 },
		{ 1400, 1539 },
		{ 1400, 1542 },
		{ 589, 529 },
		{ 676, 614 },
		{ 591, 531 },
		{ 353, 308 },
		{ 958, 915 },
		{ 1890, 1874 },
		{ 398, 357 },
		{ 805, 747 },
		{ 964, 924 },
		{ 969, 930 },
		{ 1400, 1560 },
		{ 367, 325 },
		{ 973, 934 },
		{ 699, 637 },
		{ 703, 643 },
		{ 255, 217 },
		{ 1400, 1569 },
		{ 982, 943 },
		{ 375, 333 },
		{ 813, 755 },
		{ 820, 764 },
		{ 1917, 1905 },
		{ 1918, 1906 },
		{ 822, 766 },
		{ 376, 335 },
		{ 1400, 1490 },
		{ 2067, 2049 },
		{ 331, 287 },
		{ 1199, 1197 },
		{ 998, 962 },
		{ 1203, 1202 },
		{ 825, 770 },
		{ 513, 458 },
		{ 710, 652 },
		{ 711, 654 },
		{ 1012, 978 },
		{ 713, 657 },
		{ 2081, 2064 },
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
		{ 475, 424 },
		{ 1062, 1030 },
		{ 2114, 2100 },
		{ 2115, 2101 },
		{ 640, 577 },
		{ 900, 851 },
		{ 903, 854 },
		{ 644, 581 },
		{ 497, 448 },
		{ 528, 469 },
		{ 529, 469 },
		{ 530, 470 },
		{ 531, 470 },
		{ 602, 541 },
		{ 282, 241 },
		{ 281, 241 },
		{ 601, 541 },
		{ 204, 180 },
		{ 1905, 1891 },
		{ 498, 448 },
		{ 383, 342 },
		{ 382, 342 },
		{ 656, 592 },
		{ 657, 592 },
		{ 750, 693 },
		{ 247, 211 },
		{ 250, 214 },
		{ 203, 180 },
		{ 246, 211 },
		{ 251, 214 },
		{ 260, 221 },
		{ 304, 262 },
		{ 287, 245 },
		{ 499, 449 },
		{ 751, 693 },
		{ 967, 927 },
		{ 259, 221 },
		{ 510, 455 },
		{ 252, 214 },
		{ 305, 262 },
		{ 756, 698 },
		{ 1128, 1111 },
		{ 286, 245 },
		{ 500, 449 },
		{ 3054, 3050 },
		{ 458, 408 },
		{ 759, 701 },
		{ 659, 594 },
		{ 2085, 2068 },
		{ 763, 705 },
		{ 2024, 2003 },
		{ 384, 343 },
		{ 270, 231 },
		{ 1146, 1134 },
		{ 2029, 2008 },
		{ 1147, 1135 },
		{ 588, 528 },
		{ 1066, 1035 },
		{ 2100, 2084 },
		{ 988, 949 },
		{ 918, 870 },
		{ 712, 656 },
		{ 350, 305 },
		{ 590, 530 },
		{ 1874, 1856 },
		{ 626, 563 },
		{ 351, 306 },
		{ 680, 618 },
		{ 681, 619 },
		{ 682, 620 },
		{ 289, 247 },
		{ 474, 423 },
		{ 573, 514 },
		{ 1836, 1817 },
		{ 883, 834 },
		{ 1205, 1204 },
		{ 1839, 1820 },
		{ 1892, 1876 },
		{ 438, 394 },
		{ 2000, 1977 },
		{ 3033, 3031 },
		{ 477, 426 },
		{ 1109, 1088 },
		{ 2064, 2046 },
		{ 1110, 1089 },
		{ 258, 220 },
		{ 1112, 1091 },
		{ 273, 234 },
		{ 2416, 2389 },
		{ 809, 751 },
		{ 2003, 1981 },
		{ 1100, 1076 },
		{ 2007, 1983 },
		{ 1819, 1795 },
		{ 1006, 971 },
		{ 872, 823 },
		{ 873, 824 },
		{ 261, 222 },
		{ 879, 830 },
		{ 949, 904 },
		{ 766, 708 },
		{ 2095, 2081 },
		{ 698, 636 },
		{ 426, 384 },
		{ 701, 639 },
		{ 539, 478 },
		{ 890, 841 },
		{ 1039, 1007 },
		{ 962, 922 },
		{ 1904, 1890 },
		{ 594, 534 },
		{ 577, 518 },
		{ 1052, 1019 },
		{ 436, 392 },
		{ 1057, 1024 },
		{ 1145, 1133 },
		{ 2133, 2114 },
		{ 1058, 1025 },
		{ 1059, 1026 },
		{ 294, 252 },
		{ 747, 690 },
		{ 828, 776 },
		{ 1928, 1917 },
		{ 794, 736 },
		{ 749, 692 },
		{ 467, 415 },
		{ 581, 522 },
		{ 248, 212 },
		{ 567, 508 },
		{ 989, 950 },
		{ 1082, 1055 },
		{ 1172, 1163 },
		{ 1183, 1174 },
		{ 1184, 1175 },
		{ 854, 802 },
		{ 1189, 1180 },
		{ 993, 954 },
		{ 1197, 1192 },
		{ 1198, 1193 },
		{ 1868, 1849 },
		{ 1087, 1060 },
		{ 547, 489 },
		{ 480, 429 },
		{ 864, 814 },
		{ 1093, 1067 },
		{ 923, 875 },
		{ 924, 876 },
		{ 226, 191 },
		{ 869, 819 },
		{ 1817, 1793 },
		{ 720, 664 },
		{ 302, 260 },
		{ 245, 210 },
		{ 1079, 1052 },
		{ 830, 778 },
		{ 1151, 1139 },
		{ 721, 664 },
		{ 264, 225 },
		{ 1156, 1145 },
		{ 838, 787 },
		{ 897, 848 },
		{ 1086, 1059 },
		{ 390, 349 },
		{ 1023, 989 },
		{ 622, 559 },
		{ 800, 742 },
		{ 1168, 1159 },
		{ 1027, 993 },
		{ 439, 395 },
		{ 1181, 1172 },
		{ 1888, 1872 },
		{ 2079, 2062 },
		{ 849, 798 },
		{ 444, 403 },
		{ 910, 861 },
		{ 1038, 1006 },
		{ 1190, 1183 },
		{ 1191, 1184 },
		{ 1195, 1189 },
		{ 556, 497 },
		{ 290, 248 },
		{ 1105, 1082 },
		{ 635, 572 },
		{ 1200, 1198 },
		{ 1108, 1087 },
		{ 517, 460 },
		{ 2026, 2005 },
		{ 1050, 1017 },
		{ 1212, 1211 },
		{ 1051, 1018 },
		{ 714, 658 },
		{ 1113, 1093 },
		{ 371, 329 },
		{ 683, 621 },
		{ 686, 624 },
		{ 1119, 1100 },
		{ 1124, 1106 },
		{ 496, 447 },
		{ 877, 828 },
		{ 457, 407 },
		{ 566, 507 },
		{ 936, 890 },
		{ 610, 550 },
		{ 647, 584 },
		{ 420, 379 },
		{ 3031, 3027 },
		{ 1070, 1039 },
		{ 1143, 1131 },
		{ 1144, 1132 },
		{ 1071, 1041 },
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
		{ 694, 632 },
		{ 925, 879 },
		{ 668, 604 },
		{ 1210, 1209 },
		{ 402, 362 },
		{ 844, 793 },
		{ 2065, 2047 },
		{ 2066, 2048 },
		{ 582, 523 },
		{ 595, 535 },
		{ 771, 713 },
		{ 352, 307 },
		{ 233, 198 },
		{ 677, 615 },
		{ 952, 908 },
		{ 599, 539 },
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
		{ 867, 817 },
		{ 916, 868 },
		{ 786, 728 },
		{ 1173, 1164 },
		{ 1178, 1169 },
		{ 284, 243 },
		{ 1182, 1173 },
		{ 1104, 1080 },
		{ 1040, 1008 },
		{ 976, 937 },
		{ 1187, 1178 },
		{ 1188, 1179 },
		{ 1043, 1010 },
		{ 977, 938 },
		{ 788, 730 },
		{ 1194, 1188 },
		{ 1044, 1010 },
		{ 688, 626 },
		{ 1042, 1010 },
		{ 346, 301 },
		{ 691, 629 },
		{ 532, 471 },
		{ 829, 777 },
		{ 593, 533 },
		{ 266, 227 },
		{ 1123, 1104 },
		{ 928, 882 },
		{ 1206, 1205 },
		{ 929, 883 },
		{ 885, 836 },
		{ 1127, 1110 },
		{ 695, 633 },
		{ 1129, 1112 },
		{ 369, 327 },
		{ 760, 702 },
		{ 1858, 1839 },
		{ 2050, 2029 },
		{ 1001, 966 },
		{ 1002, 967 },
		{ 358, 315 },
		{ 804, 746 },
		{ 1142, 1128 },
		{ 1075, 1046 },
		{ 948, 902 },
		{ 1008, 974 },
		{ 724, 667 },
		{ 1388, 1388 },
		{ 564, 505 },
		{ 831, 779 },
		{ 723, 666 },
		{ 984, 945 },
		{ 319, 276 },
		{ 520, 462 },
		{ 2082, 2065 },
		{ 1033, 1000 },
		{ 320, 276 },
		{ 519, 462 },
		{ 728, 673 },
		{ 727, 673 },
		{ 1032, 1000 },
		{ 729, 673 },
		{ 2083, 2066 },
		{ 399, 358 },
		{ 870, 820 },
		{ 941, 895 },
		{ 942, 896 },
		{ 1026, 992 },
		{ 285, 244 },
		{ 504, 451 },
		{ 1388, 1388 },
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
		{ 604, 543 },
		{ 1131, 1114 },
		{ 966, 926 },
		{ 1133, 1116 },
		{ 386, 345 },
		{ 403, 363 },
		{ 1055, 1022 },
		{ 971, 932 },
		{ 237, 202 },
		{ 327, 283 },
		{ 660, 596 },
		{ 1889, 1873 },
		{ 1060, 1028 },
		{ 761, 703 },
		{ 661, 597 },
		{ 709, 651 },
		{ 481, 430 },
		{ 317, 274 },
		{ 253, 215 },
		{ 1154, 1143 },
		{ 1067, 1036 },
		{ 486, 435 },
		{ 776, 718 },
		{ 987, 948 },
		{ 908, 859 },
		{ 1163, 1152 },
		{ 2047, 2026 },
		{ 1400, 1388 },
		{ 2048, 2027 },
		{ 777, 719 },
		{ 1076, 1049 },
		{ 991, 952 },
		{ 839, 788 },
		{ 555, 496 },
		{ 841, 790 },
		{ 488, 437 },
		{ 1174, 1165 },
		{ 1175, 1166 },
		{ 621, 558 },
		{ 1180, 1171 },
		{ 379, 338 },
		{ 490, 439 },
		{ 2981, 2978 },
		{ 1088, 1062 },
		{ 624, 561 },
		{ 364, 322 },
		{ 853, 801 },
		{ 492, 442 },
		{ 395, 354 },
		{ 1007, 972 },
		{ 632, 569 },
		{ 1193, 1186 },
		{ 357, 314 },
		{ 307, 264 },
		{ 798, 740 },
		{ 1018, 984 },
		{ 1019, 985 },
		{ 472, 421 },
		{ 2080, 2063 },
		{ 868, 818 },
		{ 400, 359 },
		{ 1047, 1014 },
		{ 1048, 1015 },
		{ 850, 799 },
		{ 898, 849 },
		{ 983, 944 },
		{ 1134, 1117 },
		{ 2106, 2091 },
		{ 1628, 1625 },
		{ 1305, 1302 },
		{ 1135, 1118 },
		{ 1078, 1051 },
		{ 685, 623 },
		{ 1913, 1900 },
		{ 1600, 1598 },
		{ 1654, 1651 },
		{ 558, 499 },
		{ 697, 635 },
		{ 1763, 1761 },
		{ 1678, 1675 },
		{ 2148, 2135 },
		{ 2044, 2023 },
		{ 797, 739 },
		{ 494, 444 },
		{ 1931, 1922 },
		{ 329, 285 },
		{ 1704, 1701 },
		{ 1854, 1835 },
		{ 933, 888 },
		{ 818, 762 },
		{ 368, 326 },
		{ 326, 282 },
		{ 934, 888 },
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
		{ 283, 242 },
		{ 425, 383 },
		{ 811, 753 },
		{ 1148, 1136 },
		{ 833, 781 },
		{ 1121, 1102 },
		{ 1122, 1103 },
		{ 719, 663 },
		{ 476, 425 },
		{ 1192, 1185 },
		{ 340, 295 },
		{ 2101, 2085 },
		{ 627, 564 },
		{ 1029, 995 },
		{ 733, 677 },
		{ 943, 897 },
		{ 1034, 1001 },
		{ 2108, 2094 },
		{ 2110, 2096 },
		{ 2111, 2097 },
		{ 2112, 2098 },
		{ 2113, 2099 },
		{ 238, 203 },
		{ 324, 280 },
		{ 2008, 1984 },
		{ 380, 339 },
		{ 1857, 1838 },
		{ 552, 493 },
		{ 871, 821 },
		{ 210, 183 },
		{ 742, 685 },
		{ 637, 574 },
		{ 956, 913 },
		{ 554, 495 },
		{ 511, 456 },
		{ 959, 916 },
		{ 878, 829 },
		{ 296, 254 },
		{ 880, 831 },
		{ 1054, 1021 },
		{ 1872, 1854 },
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
		{ 652, 588 },
		{ 410, 370 },
		{ 654, 590 },
		{ 482, 431 },
		{ 2049, 2028 },
		{ 899, 850 },
		{ 441, 397 },
		{ 527, 468 },
		{ 387, 346 },
		{ 770, 712 },
		{ 349, 304 },
		{ 298, 256 },
		{ 1906, 1892 },
		{ 774, 716 },
		{ 775, 717 },
		{ 835, 784 },
		{ 445, 404 },
		{ 2062, 2044 },
		{ 999, 963 },
		{ 1915, 1903 },
		{ 715, 659 },
		{ 1185, 1176 },
		{ 915, 866 },
		{ 778, 720 },
		{ 1820, 1796 },
		{ 1094, 1068 },
		{ 781, 723 },
		{ 842, 791 },
		{ 269, 230 },
		{ 416, 375 },
		{ 334, 290 },
		{ 785, 727 },
		{ 848, 797 },
		{ 193, 174 },
		{ 214, 185 },
		{ 1013, 979 },
		{ 1017, 983 },
		{ 460, 411 },
		{ 1201, 1199 },
		{ 372, 330 },
		{ 465, 413 },
		{ 1204, 1203 },
		{ 855, 804 },
		{ 857, 807 },
		{ 1209, 1208 },
		{ 396, 355 },
		{ 1211, 1210 },
		{ 860, 810 },
		{ 272, 233 },
		{ 863, 813 },
		{ 2175, 2175 },
		{ 2175, 2175 },
		{ 1924, 1924 },
		{ 1924, 1924 },
		{ 1926, 1926 },
		{ 1926, 1926 },
		{ 1537, 1537 },
		{ 1537, 1537 },
		{ 2121, 2121 },
		{ 2121, 2121 },
		{ 2123, 2123 },
		{ 2123, 2123 },
		{ 2125, 2125 },
		{ 2125, 2125 },
		{ 2127, 2127 },
		{ 2127, 2127 },
		{ 2129, 2129 },
		{ 2129, 2129 },
		{ 2131, 2131 },
		{ 2131, 2131 },
		{ 1901, 1901 },
		{ 1901, 1901 },
		{ 912, 863 },
		{ 2175, 2175 },
		{ 913, 864 },
		{ 1924, 1924 },
		{ 249, 213 },
		{ 1926, 1926 },
		{ 2200, 2198 },
		{ 1537, 1537 },
		{ 963, 923 },
		{ 2121, 2121 },
		{ 314, 271 },
		{ 2123, 2123 },
		{ 466, 414 },
		{ 2125, 2125 },
		{ 560, 501 },
		{ 2127, 2127 },
		{ 540, 479 },
		{ 2129, 2129 },
		{ 274, 235 },
		{ 2131, 2131 },
		{ 406, 366 },
		{ 1901, 1901 },
		{ 649, 586 },
		{ 650, 586 },
		{ 1936, 1936 },
		{ 1936, 1936 },
		{ 2146, 2146 },
		{ 2146, 2146 },
		{ 1595, 1595 },
		{ 1595, 1595 },
		{ 2176, 2175 },
		{ 543, 483 },
		{ 1925, 1924 },
		{ 1068, 1037 },
		{ 1927, 1926 },
		{ 309, 266 },
		{ 1538, 1537 },
		{ 845, 794 },
		{ 2122, 2121 },
		{ 292, 250 },
		{ 2124, 2123 },
		{ 1072, 1042 },
		{ 2126, 2125 },
		{ 1073, 1043 },
		{ 2128, 2127 },
		{ 684, 622 },
		{ 2130, 2129 },
		{ 1936, 1936 },
		{ 2132, 2131 },
		{ 2146, 2146 },
		{ 1902, 1901 },
		{ 1595, 1595 },
		{ 1941, 1941 },
		{ 1941, 1941 },
		{ 2092, 2092 },
		{ 2092, 2092 },
		{ 2152, 2152 },
		{ 2152, 2152 },
		{ 1540, 1540 },
		{ 1540, 1540 },
		{ 1531, 1531 },
		{ 1531, 1531 },
		{ 1543, 1543 },
		{ 1543, 1543 },
		{ 1491, 1491 },
		{ 1491, 1491 },
		{ 1570, 1570 },
		{ 1570, 1570 },
		{ 1534, 1534 },
		{ 1534, 1534 },
		{ 1955, 1955 },
		{ 1955, 1955 },
		{ 1561, 1561 },
		{ 1561, 1561 },
		{ 772, 714 },
		{ 1941, 1941 },
		{ 1937, 1936 },
		{ 2092, 2092 },
		{ 2147, 2146 },
		{ 2152, 2152 },
		{ 1596, 1595 },
		{ 1540, 1540 },
		{ 754, 696 },
		{ 1531, 1531 },
		{ 568, 509 },
		{ 1543, 1543 },
		{ 979, 940 },
		{ 1491, 1491 },
		{ 651, 587 },
		{ 1570, 1570 },
		{ 671, 610 },
		{ 1534, 1534 },
		{ 1769, 1768 },
		{ 1955, 1955 },
		{ 799, 741 },
		{ 1561, 1561 },
		{ 1528, 1528 },
		{ 1528, 1528 },
		{ 374, 332 },
		{ 945, 899 },
		{ 1160, 1149 },
		{ 1053, 1020 },
		{ 1162, 1151 },
		{ 947, 901 },
		{ 1942, 1941 },
		{ 734, 678 },
		{ 2093, 2092 },
		{ 1897, 1882 },
		{ 2153, 2152 },
		{ 764, 706 },
		{ 1541, 1540 },
		{ 1030, 997 },
		{ 1532, 1531 },
		{ 3064, 3062 },
		{ 1544, 1543 },
		{ 341, 296 },
		{ 1492, 1491 },
		{ 1170, 1161 },
		{ 1571, 1570 },
		{ 1528, 1528 },
		{ 1535, 1534 },
		{ 1208, 1207 },
		{ 1956, 1955 },
		{ 736, 680 },
		{ 1562, 1561 },
		{ 789, 731 },
		{ 1684, 1683 },
		{ 1634, 1633 },
		{ 1138, 1121 },
		{ 953, 910 },
		{ 1140, 1124 },
		{ 1177, 1168 },
		{ 887, 838 },
		{ 862, 812 },
		{ 1014, 980 },
		{ 1015, 981 },
		{ 994, 955 },
		{ 935, 889 },
		{ 836, 785 },
		{ 1120, 1101 },
		{ 814, 756 },
		{ 1311, 1310 },
		{ 1522, 1504 },
		{ 1929, 1918 },
		{ 1606, 1605 },
		{ 1046, 1013 },
		{ 3034, 3032 },
		{ 1877, 1860 },
		{ 1529, 1528 },
		{ 1096, 1071 },
		{ 815, 757 },
		{ 2070, 2053 },
		{ 1710, 1709 },
		{ 1153, 1141 },
		{ 1660, 1659 },
		{ 892, 843 },
		{ 1155, 1144 },
		{ 195, 176 },
		{ 2134, 2115 },
		{ 2014, 1991 },
		{ 487, 436 },
		{ 223, 189 },
		{ 607, 547 },
		{ 1816, 1792 },
		{ 2145, 2133 },
		{ 1935, 1928 },
		{ 1916, 1904 },
		{ 1255, 1252 },
		{ 1176, 1167 },
		{ 301, 259 },
		{ 2002, 1980 },
		{ 2109, 2095 },
		{ 2874, 2872 },
		{ 2676, 2676 },
		{ 2676, 2676 },
		{ 2976, 2976 },
		{ 2976, 2976 },
		{ 2730, 2730 },
		{ 2730, 2730 },
		{ 2614, 2614 },
		{ 2614, 2614 },
		{ 2476, 2476 },
		{ 2476, 2476 },
		{ 2367, 2367 },
		{ 2367, 2367 },
		{ 1782, 1782 },
		{ 1782, 1782 },
		{ 2735, 2735 },
		{ 2735, 2735 },
		{ 1759, 1759 },
		{ 1759, 1759 },
		{ 1250, 1250 },
		{ 1250, 1250 },
		{ 2620, 2620 },
		{ 2620, 2620 },
		{ 667, 603 },
		{ 2676, 2676 },
		{ 629, 566 },
		{ 2976, 2976 },
		{ 424, 382 },
		{ 2730, 2730 },
		{ 990, 951 },
		{ 2614, 2614 },
		{ 670, 609 },
		{ 2476, 2476 },
		{ 631, 568 },
		{ 2367, 2367 },
		{ 325, 281 },
		{ 1782, 1782 },
		{ 633, 570 },
		{ 2735, 2735 },
		{ 930, 885 },
		{ 1759, 1759 },
		{ 243, 208 },
		{ 1250, 1250 },
		{ 817, 761 },
		{ 2620, 2620 },
		{ 874, 825 },
		{ 2500, 2500 },
		{ 2500, 2500 },
		{ 1673, 1673 },
		{ 1673, 1673 },
		{ 2695, 2676 },
		{ 275, 236 },
		{ 2977, 2976 },
		{ 1207, 1206 },
		{ 2738, 2730 },
		{ 768, 710 },
		{ 2638, 2614 },
		{ 937, 891 },
		{ 2505, 2476 },
		{ 938, 892 },
		{ 2368, 2367 },
		{ 769, 711 },
		{ 1783, 1782 },
		{ 1004, 969 },
		{ 2743, 2735 },
		{ 2939, 2937 },
		{ 1760, 1759 },
		{ 356, 313 },
		{ 1251, 1250 },
		{ 2500, 2500 },
		{ 2644, 2620 },
		{ 1673, 1673 },
		{ 485, 434 },
		{ 2745, 2745 },
		{ 2745, 2745 },
		{ 2870, 2870 },
		{ 2870, 2870 },
		{ 2746, 2746 },
		{ 2746, 2746 },
		{ 2747, 2747 },
		{ 2747, 2747 },
		{ 1699, 1699 },
		{ 1699, 1699 },
		{ 2504, 2504 },
		{ 2504, 2504 },
		{ 2935, 2935 },
		{ 2935, 2935 },
		{ 2658, 2658 },
		{ 2658, 2658 },
		{ 2659, 2659 },
		{ 2659, 2659 },
		{ 2755, 2755 },
		{ 2755, 2755 },
		{ 2414, 2414 },
		{ 2414, 2414 },
		{ 2531, 2500 },
		{ 2745, 2745 },
		{ 1674, 1673 },
		{ 2870, 2870 },
		{ 679, 617 },
		{ 2746, 2746 },
		{ 600, 540 },
		{ 2747, 2747 },
		{ 544, 484 },
		{ 1699, 1699 },
		{ 291, 249 },
		{ 2504, 2504 },
		{ 2167, 2164 },
		{ 2935, 2935 },
		{ 1080, 1053 },
		{ 2658, 2658 },
		{ 946, 900 },
		{ 2659, 2659 },
		{ 827, 773 },
		{ 2755, 2755 },
		{ 373, 331 },
		{ 2414, 2414 },
		{ 316, 273 },
		{ 2696, 2696 },
		{ 2696, 2696 },
		{ 2598, 2598 },
		{ 2598, 2598 },
		{ 2750, 2745 },
		{ 575, 516 },
		{ 2871, 2870 },
		{ 1016, 982 },
		{ 2751, 2746 },
		{ 779, 721 },
		{ 2752, 2747 },
		{ 780, 722 },
		{ 1700, 1699 },
		{ 730, 674 },
		{ 2534, 2504 },
		{ 393, 352 },
		{ 2936, 2935 },
		{ 732, 676 },
		{ 2677, 2658 },
		{ 227, 192 },
		{ 2678, 2659 },
		{ 837, 786 },
		{ 2757, 2755 },
		{ 2696, 2696 },
		{ 2443, 2414 },
		{ 2598, 2598 },
		{ 1730, 1727 },
		{ 1285, 1285 },
		{ 1285, 1285 },
		{ 2527, 2527 },
		{ 2527, 2527 },
		{ 1328, 1328 },
		{ 1328, 1328 },
		{ 2221, 2221 },
		{ 2221, 2221 },
		{ 2887, 2887 },
		{ 2887, 2887 },
		{ 2764, 2764 },
		{ 2764, 2764 },
		{ 1623, 1623 },
		{ 1623, 1623 },
		{ 2706, 2706 },
		{ 2706, 2706 },
		{ 1300, 1300 },
		{ 1300, 1300 },
		{ 2668, 2668 },
		{ 2668, 2668 },
		{ 1649, 1649 },
		{ 1649, 1649 },
		{ 2711, 2696 },
		{ 1285, 1285 },
		{ 2623, 2598 },
		{ 2527, 2527 },
		{ 523, 464 },
		{ 1328, 1328 },
		{ 257, 219 },
		{ 2221, 2221 },
		{ 1164, 1153 },
		{ 2887, 2887 },
		{ 525, 466 },
		{ 2764, 2764 },
		{ 377, 336 },
		{ 1623, 1623 },
		{ 471, 420 },
		{ 2706, 2706 },
		{ 267, 228 },
		{ 1300, 1300 },
		{ 1169, 1160 },
		{ 2668, 2668 },
		{ 2801, 2799 },
		{ 1649, 1649 },
		{ 440, 396 },
		{ 2710, 2710 },
		{ 2710, 2710 },
		{ 2841, 2841 },
		{ 2841, 2841 },
		{ 1286, 1285 },
		{ 902, 853 },
		{ 2528, 2527 },
		{ 333, 289 },
		{ 1329, 1328 },
		{ 793, 735 },
		{ 2222, 2221 },
		{ 2892, 2889 },
		{ 2888, 2887 },
		{ 321, 277 },
		{ 2765, 2764 },
		{ 744, 687 },
		{ 1624, 1623 },
		{ 745, 688 },
		{ 2721, 2706 },
		{ 194, 175 },
		{ 1301, 1300 },
		{ 658, 593 },
		{ 2687, 2668 },
		{ 2710, 2710 },
		{ 1650, 1649 },
		{ 2841, 2841 },
		{ 1179, 1170 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2713, 2713 },
		{ 2713, 2713 },
		{ 2714, 2714 },
		{ 2714, 2714 },
		{ 2715, 2715 },
		{ 2715, 2715 },
		{ 2473, 2473 },
		{ 2473, 2473 },
		{ 1725, 1725 },
		{ 1725, 1725 },
		{ 2609, 2609 },
		{ 2609, 2609 },
		{ 2585, 2585 },
		{ 2585, 2585 },
		{ 2422, 2422 },
		{ 2422, 2422 },
		{ 2797, 2797 },
		{ 2797, 2797 },
		{ 2613, 2613 },
		{ 2613, 2613 },
		{ 2725, 2710 },
		{ 2712, 2712 },
		{ 2842, 2841 },
		{ 2713, 2713 },
		{ 311, 268 },
		{ 2714, 2714 },
		{ 974, 935 },
		{ 2715, 2715 },
		{ 700, 638 },
		{ 2473, 2473 },
		{ 562, 503 },
		{ 1725, 1725 },
		{ 856, 806 },
		{ 2609, 2609 },
		{ 914, 865 },
		{ 2585, 2585 },
		{ 702, 642 },
		{ 2422, 2422 },
		{ 858, 808 },
		{ 2797, 2797 },
		{ 366, 324 },
		{ 2613, 2613 },
		{ 662, 598 },
		{ 2723, 2723 },
		{ 2723, 2723 },
		{ 312, 269 },
		{ 628, 565 },
		{ 2726, 2712 },
		{ 665, 601 },
		{ 2727, 2713 },
		{ 666, 602 },
		{ 2728, 2714 },
		{ 3095, 3095 },
		{ 2729, 2715 },
		{ 3103, 3103 },
		{ 2502, 2473 },
		{ 3107, 3107 },
		{ 1726, 1725 },
		{ 2178, 2176 },
		{ 2634, 2609 },
		{ 2154, 2147 },
		{ 2610, 2585 },
		{ 1510, 1492 },
		{ 2423, 2422 },
		{ 1946, 1942 },
		{ 2798, 2797 },
		{ 2723, 2723 },
		{ 2637, 2613 },
		{ 1933, 1925 },
		{ 1558, 1544 },
		{ 2139, 2122 },
		{ 2159, 2153 },
		{ 1934, 1927 },
		{ 2140, 2124 },
		{ 1556, 1538 },
		{ 3095, 3095 },
		{ 2141, 2126 },
		{ 3103, 3103 },
		{ 1553, 1529 },
		{ 3107, 3107 },
		{ 2142, 2128 },
		{ 1555, 1535 },
		{ 2143, 2130 },
		{ 1557, 1541 },
		{ 2144, 2132 },
		{ 1914, 1902 },
		{ 1597, 1596 },
		{ 2107, 2093 },
		{ 1572, 1562 },
		{ 1957, 1956 },
		{ 1943, 1937 },
		{ 1578, 1571 },
		{ 2736, 2723 },
		{ 1554, 1532 },
		{ 3008, 3007 },
		{ 1585, 1580 },
		{ 1360, 1359 },
		{ 1715, 1714 },
		{ 3069, 3068 },
		{ 3070, 3069 },
		{ 1716, 1715 },
		{ 3098, 3095 },
		{ 1611, 1610 },
		{ 3105, 3103 },
		{ 1612, 1611 },
		{ 3108, 3107 },
		{ 1639, 1638 },
		{ 1774, 1773 },
		{ 1775, 1774 },
		{ 1640, 1639 },
		{ 1689, 1688 },
		{ 1690, 1689 },
		{ 1317, 1316 },
		{ 1588, 1584 },
		{ 1665, 1664 },
		{ 1666, 1665 },
		{ 1316, 1315 },
		{ 1584, 1579 },
		{ 2771, 2771 },
		{ 2768, 2771 },
		{ 163, 163 },
		{ 160, 163 },
		{ 1886, 1871 },
		{ 1887, 1871 },
		{ 1908, 1896 },
		{ 1909, 1896 },
		{ 2188, 2188 },
		{ 1962, 1962 },
		{ 1967, 1963 },
		{ 162, 158 },
		{ 2770, 2766 },
		{ 168, 164 },
		{ 2247, 2223 },
		{ 1966, 1963 },
		{ 161, 158 },
		{ 2769, 2766 },
		{ 167, 164 },
		{ 2246, 2223 },
		{ 2196, 2195 },
		{ 2776, 2772 },
		{ 88, 70 },
		{ 2771, 2771 },
		{ 2193, 2189 },
		{ 163, 163 },
		{ 2775, 2772 },
		{ 87, 70 },
		{ 3019, 3011 },
		{ 2192, 2189 },
		{ 2039, 2018 },
		{ 2188, 2188 },
		{ 1962, 1962 },
		{ 169, 166 },
		{ 171, 170 },
		{ 119, 103 },
		{ 2777, 2774 },
		{ 2779, 2778 },
		{ 2772, 2771 },
		{ 1846, 1827 },
		{ 164, 163 },
		{ 1968, 1965 },
		{ 1970, 1969 },
		{ 2194, 2191 },
		{ 2305, 2275 },
		{ 2183, 2182 },
		{ 2189, 2188 },
		{ 1963, 1962 },
		{ 2037, 2016 },
		{ 1921, 1911 },
		{ 1969, 1967 },
		{ 2018, 1996 },
		{ 166, 162 },
		{ 2195, 2193 },
		{ 2275, 2247 },
		{ 1965, 1961 },
		{ 2774, 2770 },
		{ 103, 88 },
		{ 1827, 1805 },
		{ 170, 168 },
		{ 2778, 2776 },
		{ 2743, 2743 },
		{ 2743, 2743 },
		{ 0, 2998 },
		{ 0, 2492 },
		{ 0, 2444 },
		{ 0, 2907 },
		{ 0, 2555 },
		{ 0, 2825 },
		{ 0, 2910 },
		{ 2634, 2634 },
		{ 2634, 2634 },
		{ 0, 2336 },
		{ 0, 2444 },
		{ 2368, 2368 },
		{ 2368, 2368 },
		{ 2750, 2750 },
		{ 2750, 2750 },
		{ 0, 2915 },
		{ 2751, 2751 },
		{ 2751, 2751 },
		{ 2752, 2752 },
		{ 2752, 2752 },
		{ 0, 2832 },
		{ 2743, 2743 },
		{ 0, 2685 },
		{ 2637, 2637 },
		{ 2637, 2637 },
		{ 2687, 2687 },
		{ 2687, 2687 },
		{ 2638, 2638 },
		{ 2638, 2638 },
		{ 0, 2595 },
		{ 2634, 2634 },
		{ 2757, 2757 },
		{ 2757, 2757 },
		{ 0, 2558 },
		{ 2368, 2368 },
		{ 0, 2926 },
		{ 2750, 2750 },
		{ 0, 2471 },
		{ 0, 1734 },
		{ 2751, 2751 },
		{ 0, 2336 },
		{ 2752, 2752 },
		{ 0, 2694 },
		{ 2695, 2695 },
		{ 2695, 2695 },
		{ 0, 2697 },
		{ 2637, 2637 },
		{ 0, 2699 },
		{ 2687, 2687 },
		{ 0, 2700 },
		{ 2638, 2638 },
		{ 2765, 2765 },
		{ 2765, 2765 },
		{ 0, 2602 },
		{ 2757, 2757 },
		{ 2196, 2196 },
		{ 2197, 2196 },
		{ 2528, 2528 },
		{ 2528, 2528 },
		{ 0, 2940 },
		{ 0, 2853 },
		{ 0, 2648 },
		{ 0, 1237 },
		{ 0, 2944 },
		{ 171, 171 },
		{ 172, 171 },
		{ 2695, 2695 },
		{ 2779, 2779 },
		{ 2780, 2779 },
		{ 0, 2948 },
		{ 0, 2651 },
		{ 2502, 2502 },
		{ 2502, 2502 },
		{ 0, 2951 },
		{ 2765, 2765 },
		{ 0, 1745 },
		{ 0, 2862 },
		{ 0, 2569 },
		{ 2196, 2196 },
		{ 0, 2956 },
		{ 2528, 2528 },
		{ 2534, 2534 },
		{ 2534, 2534 },
		{ 0, 2787 },
		{ 0, 1276 },
		{ 2711, 2711 },
		{ 2711, 2711 },
		{ 171, 171 },
		{ 2610, 2610 },
		{ 2610, 2610 },
		{ 2779, 2779 },
		{ 0, 2571 },
		{ 0, 2853 },
		{ 0, 1271 },
		{ 2502, 2502 },
		{ 2505, 2505 },
		{ 2505, 2505 },
		{ 0, 2432 },
		{ 0, 2433 },
		{ 0, 2481 },
		{ 0, 2967 },
		{ 0, 2434 },
		{ 0, 2542 },
		{ 0, 2878 },
		{ 2534, 2534 },
		{ 2721, 2721 },
		{ 2721, 2721 },
		{ 0, 2579 },
		{ 2711, 2711 },
		{ 0, 2802 },
		{ 0, 2807 },
		{ 2610, 2610 },
		{ 2727, 2727 },
		{ 2727, 2727 },
		{ 1970, 1970 },
		{ 1971, 1970 },
		{ 2623, 2623 },
		{ 2623, 2623 },
		{ 2505, 2505 },
		{ 0, 1292 },
		{ 0, 1750 },
		{ 0, 1260 },
		{ 0, 2893 },
		{ 0, 2549 },
		{ 0, 2987 },
		{ 2736, 2736 },
		{ 2736, 2736 },
		{ 0, 2737 },
		{ 2721, 2721 },
		{ 0, 2897 },
		{ 2738, 2738 },
		{ 2738, 2738 },
		{ 0, 2210 },
		{ 0, 2402 },
		{ 0, 2818 },
		{ 2727, 2727 },
		{ 1216, 1216 },
		{ 1970, 1970 },
		{ 1505, 1505 },
		{ 2623, 2623 },
		{ 2443, 2443 },
		{ 2443, 2443 },
		{ 2678, 2678 },
		{ 2678, 2678 },
		{ 1359, 1358 },
		{ 2207, 2206 },
		{ 2773, 2769 },
		{ 3023, 3019 },
		{ 2736, 2736 },
		{ 0, 87 },
		{ 1964, 1960 },
		{ 2190, 2192 },
		{ 0, 2246 },
		{ 2738, 2738 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 1216, 1216 },
		{ 0, 0 },
		{ 1505, 1505 },
		{ 0, 0 },
		{ 2443, 2443 },
		{ 0, 0 },
		{ 2678, 2678 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -68, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 44, 572, 0 },
		{ -177, 2810, 0 },
		{ 5, 0, 0 },
		{ -1215, 1017, -31 },
		{ 7, 0, -31 },
		{ -1219, 1830, -33 },
		{ 9, 0, -33 },
		{ -1232, 3132, 146 },
		{ 11, 0, 146 },
		{ -1253, 3155, 154 },
		{ 13, 0, 154 },
		{ -1288, 3267, 0 },
		{ 15, 0, 0 },
		{ -1303, 3148, 142 },
		{ 17, 0, 142 },
		{ -1331, 3137, 22 },
		{ 19, 0, 22 },
		{ -1373, 230, 0 },
		{ 21, 0, 0 },
		{ -1599, 3269, 0 },
		{ 23, 0, 0 },
		{ -1626, 3130, 0 },
		{ 25, 0, 0 },
		{ -1652, 3153, 0 },
		{ 27, 0, 0 },
		{ -1676, 3142, 0 },
		{ 29, 0, 0 },
		{ -1702, 3131, 0 },
		{ 31, 0, 0 },
		{ -1728, 3138, 158 },
		{ 33, 0, 158 },
		{ -1762, 3275, 265 },
		{ 35, 0, 265 },
		{ 38, 129, 0 },
		{ -1798, 344, 0 },
		{ 40, 127, 0 },
		{ -1987, 116, 0 },
		{ -2199, 3276, 0 },
		{ 41, 0, 0 },
		{ 44, 14, 0 },
		{ -86, 458, 0 },
		{ -2782, 3136, 150 },
		{ 45, 0, 150 },
		{ -2800, 3268, 173 },
		{ 47, 0, 173 },
		{ 2844, 1486, 0 },
		{ 49, 0, 0 },
		{ -2846, 3271, 271 },
		{ 51, 0, 271 },
		{ -2873, 3274, 176 },
		{ 53, 0, 176 },
		{ -2891, 3144, 169 },
		{ 55, 0, 169 },
		{ -2938, 3277, 162 },
		{ 57, 0, 162 },
		{ -2979, 3154, 168 },
		{ 59, 0, 168 },
		{ 44, 1, 0 },
		{ 61, 0, 0 },
		{ -3030, 1790, 0 },
		{ 63, 0, 0 },
		{ -3040, 1699, 44 },
		{ 65, 0, 44 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 424 },
		{ 3011, 4712, 431 },
		{ 0, 0, 243 },
		{ 0, 0, 245 },
		{ 157, 1273, 262 },
		{ 157, 1400, 262 },
		{ 157, 1299, 262 },
		{ 157, 1306, 262 },
		{ 157, 1306, 262 },
		{ 157, 1310, 262 },
		{ 157, 1315, 262 },
		{ 157, 1309, 262 },
		{ 3089, 2808, 431 },
		{ 157, 1320, 262 },
		{ 3089, 1622, 261 },
		{ 102, 2539, 431 },
		{ 157, 0, 262 },
		{ 0, 0, 431 },
		{ -87, 4934, 239 },
		{ -88, 4747, 0 },
		{ 157, 1337, 262 },
		{ 157, 721, 262 },
		{ 157, 690, 262 },
		{ 157, 730, 262 },
		{ 157, 712, 262 },
		{ 157, 712, 262 },
		{ 157, 719, 262 },
		{ 157, 735, 262 },
		{ 157, 764, 262 },
		{ 3058, 2280, 0 },
		{ 157, 754, 262 },
		{ 3089, 1711, 258 },
		{ 117, 1439, 0 },
		{ 3089, 1700, 259 },
		{ 3011, 4720, 0 },
		{ 157, 763, 262 },
		{ 157, 761, 262 },
		{ 157, 762, 262 },
		{ 157, 758, 262 },
		{ 157, 0, 250 },
		{ 157, 777, 262 },
		{ 157, 805, 262 },
		{ 157, 789, 262 },
		{ 157, 794, 262 },
		{ 3086, 2928, 0 },
		{ 157, 801, 262 },
		{ 131, 1453, 0 },
		{ 117, 0, 0 },
		{ 3013, 2579, 260 },
		{ 133, 1520, 0 },
		{ 0, 0, 241 },
		{ 157, 806, 246 },
		{ 157, 836, 262 },
		{ 157, 828, 262 },
		{ 157, 833, 262 },
		{ 157, 831, 262 },
		{ 157, 824, 262 },
		{ 157, 0, 253 },
		{ 157, 825, 262 },
		{ 0, 0, 255 },
		{ 157, 831, 262 },
		{ 131, 0, 0 },
		{ 3013, 2635, 258 },
		{ 133, 0, 0 },
		{ 3013, 2668, 259 },
		{ 157, 846, 262 },
		{ 157, 843, 262 },
		{ 157, 844, 262 },
		{ 157, 871, 262 },
		{ 157, 931, 262 },
		{ 157, 0, 252 },
		{ 157, 1009, 262 },
		{ 157, 1056, 262 },
		{ 157, 1075, 262 },
		{ 157, 0, 248 },
		{ 157, 1161, 262 },
		{ 157, 0, 249 },
		{ 157, 0, 251 },
		{ 157, 1192, 262 },
		{ 157, 1253, 262 },
		{ 157, 0, 247 },
		{ 157, 1273, 262 },
		{ 157, 0, 254 },
		{ 157, 740, 262 },
		{ 157, 1300, 262 },
		{ 0, 0, 257 },
		{ 157, 1284, 262 },
		{ 157, 1287, 262 },
		{ 3106, 1357, 256 },
		{ 3011, 4701, 431 },
		{ 163, 0, 243 },
		{ 0, 0, 244 },
		{ -161, 8, 239 },
		{ -162, 4742, 0 },
		{ 3061, 4725, 0 },
		{ 3011, 4703, 0 },
		{ 0, 0, 240 },
		{ 3011, 4718, 0 },
		{ -167, 17, 0 },
		{ -168, 4749, 0 },
		{ 171, 0, 241 },
		{ 3011, 4719, 0 },
		{ 3061, 4850, 0 },
		{ 0, 0, 242 },
		{ 3062, 1579, 140 },
		{ 2099, 4115, 140 },
		{ 2937, 4539, 140 },
		{ 3062, 4313, 140 },
		{ 0, 0, 140 },
		{ 3050, 3433, 0 },
		{ 2114, 2989, 0 },
		{ 3050, 3601, 0 },
		{ 3050, 3259, 0 },
		{ 3027, 3336, 0 },
		{ 2099, 4054, 0 },
		{ 3054, 3244, 0 },
		{ 2099, 4116, 0 },
		{ 3028, 3512, 0 },
		{ 3057, 3226, 0 },
		{ 3028, 3296, 0 },
		{ 2872, 4318, 0 },
		{ 3054, 3267, 0 },
		{ 2114, 3722, 0 },
		{ 2937, 4467, 0 },
		{ 2045, 3489, 0 },
		{ 2045, 3424, 0 },
		{ 2067, 3103, 0 },
		{ 2048, 3795, 0 },
		{ 2029, 3820, 0 },
		{ 2048, 3811, 0 },
		{ 2067, 3105, 0 },
		{ 2067, 3132, 0 },
		{ 3054, 3305, 0 },
		{ 2978, 3921, 0 },
		{ 2099, 4047, 0 },
		{ 1185, 4020, 0 },
		{ 2045, 3470, 0 },
		{ 2067, 3147, 0 },
		{ 2067, 3149, 0 },
		{ 2937, 4371, 0 },
		{ 2045, 3445, 0 },
		{ 3027, 3728, 0 },
		{ 3050, 3602, 0 },
		{ 2114, 3702, 0 },
		{ 2198, 4159, 0 },
		{ 2937, 3615, 0 },
		{ 2978, 3931, 0 },
		{ 2029, 3817, 0 },
		{ 3028, 3537, 0 },
		{ 2007, 3269, 0 },
		{ 2937, 4503, 0 },
		{ 3050, 3659, 0 },
		{ 2978, 3611, 0 },
		{ 2114, 3672, 0 },
		{ 2067, 3164, 0 },
		{ 3057, 3371, 0 },
		{ 3027, 3733, 0 },
		{ 2007, 3267, 0 },
		{ 2029, 3853, 0 },
		{ 2937, 4513, 0 },
		{ 2099, 4077, 0 },
		{ 2099, 4110, 0 },
		{ 3050, 3626, 0 },
		{ 2067, 3081, 0 },
		{ 2099, 4130, 0 },
		{ 3050, 3661, 0 },
		{ 2198, 4173, 0 },
		{ 2937, 4381, 0 },
		{ 3057, 3373, 0 },
		{ 3028, 3561, 0 },
		{ 2067, 3082, 0 },
		{ 3057, 3310, 0 },
		{ 3050, 3589, 0 },
		{ 1185, 4024, 0 },
		{ 2029, 3834, 0 },
		{ 2978, 3899, 0 },
		{ 2114, 3617, 0 },
		{ 1073, 3230, 0 },
		{ 3050, 3644, 0 },
		{ 3027, 3756, 0 },
		{ 2937, 4435, 0 },
		{ 2198, 4194, 0 },
		{ 2067, 3083, 0 },
		{ 2114, 3694, 0 },
		{ 3057, 3345, 0 },
		{ 2099, 4062, 0 },
		{ 2007, 3270, 0 },
		{ 2099, 4093, 0 },
		{ 3028, 3565, 0 },
		{ 2067, 3084, 0 },
		{ 2872, 4326, 0 },
		{ 3027, 3727, 0 },
		{ 3057, 3384, 0 },
		{ 2135, 3614, 0 },
		{ 2067, 3085, 0 },
		{ 2978, 3966, 0 },
		{ 2099, 4068, 0 },
		{ 2198, 4190, 0 },
		{ 2099, 4073, 0 },
		{ 2937, 4573, 0 },
		{ 2937, 4594, 0 },
		{ 2029, 3819, 0 },
		{ 2198, 4165, 0 },
		{ 2067, 3086, 0 },
		{ 2937, 4447, 0 },
		{ 2978, 3930, 0 },
		{ 2029, 3822, 0 },
		{ 2978, 3887, 0 },
		{ 2937, 4533, 0 },
		{ 2045, 3466, 0 },
		{ 3028, 3514, 0 },
		{ 2099, 4048, 0 },
		{ 2937, 4365, 0 },
		{ 1185, 4007, 0 },
		{ 2978, 3922, 0 },
		{ 1073, 3232, 0 },
		{ 2135, 4000, 0 },
		{ 2048, 3787, 0 },
		{ 3028, 3549, 0 },
		{ 2067, 3088, 0 },
		{ 2937, 4527, 0 },
		{ 2099, 4112, 0 },
		{ 2067, 3089, 0 },
		{ 0, 0, 72 },
		{ 3050, 3425, 0 },
		{ 3057, 3412, 0 },
		{ 2099, 4035, 0 },
		{ 3062, 4271, 0 },
		{ 2067, 3090, 0 },
		{ 2067, 3091, 0 },
		{ 3057, 3354, 0 },
		{ 2045, 3438, 0 },
		{ 2029, 3848, 0 },
		{ 3057, 3356, 0 },
		{ 2067, 3092, 0 },
		{ 2099, 4092, 0 },
		{ 3050, 3636, 0 },
		{ 3050, 3640, 0 },
		{ 2048, 3810, 0 },
		{ 3028, 3525, 0 },
		{ 847, 3243, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2045, 3471, 0 },
		{ 2937, 4397, 0 },
		{ 2978, 3965, 0 },
		{ 2029, 3868, 0 },
		{ 3057, 3383, 0 },
		{ 3028, 3562, 0 },
		{ 0, 0, 70 },
		{ 2067, 3094, 0 },
		{ 2045, 3427, 0 },
		{ 3057, 3385, 0 },
		{ 2978, 3958, 0 },
		{ 3057, 3392, 0 },
		{ 2937, 4589, 0 },
		{ 3028, 3533, 0 },
		{ 1185, 4006, 0 },
		{ 2029, 3862, 0 },
		{ 2045, 3465, 0 },
		{ 3027, 3768, 0 },
		{ 2099, 4121, 0 },
		{ 2937, 4445, 0 },
		{ 3062, 4254, 0 },
		{ 3028, 3540, 0 },
		{ 0, 0, 64 },
		{ 3028, 3546, 0 },
		{ 2937, 4509, 0 },
		{ 1185, 4017, 0 },
		{ 2978, 3953, 0 },
		{ 2099, 4050, 0 },
		{ 0, 0, 75 },
		{ 3057, 3406, 0 },
		{ 3050, 3595, 0 },
		{ 3050, 3625, 0 },
		{ 2067, 3095, 0 },
		{ 2978, 3917, 0 },
		{ 2099, 4090, 0 },
		{ 2067, 3096, 0 },
		{ 2045, 3485, 0 },
		{ 3027, 3738, 0 },
		{ 3057, 3415, 0 },
		{ 3028, 3513, 0 },
		{ 2937, 4463, 0 },
		{ 2067, 3097, 0 },
		{ 2978, 3961, 0 },
		{ 2099, 4127, 0 },
		{ 3057, 3331, 0 },
		{ 3028, 3528, 0 },
		{ 2978, 3894, 0 },
		{ 1015, 3974, 0 },
		{ 0, 0, 8 },
		{ 2045, 3433, 0 },
		{ 2048, 3803, 0 },
		{ 2978, 3918, 0 },
		{ 2080, 3214, 0 },
		{ 2067, 3098, 0 },
		{ 2198, 4175, 0 },
		{ 2099, 4076, 0 },
		{ 2045, 3462, 0 },
		{ 2099, 4078, 0 },
		{ 2099, 4083, 0 },
		{ 2048, 3793, 0 },
		{ 2067, 3099, 0 },
		{ 3057, 3357, 0 },
		{ 3057, 3265, 0 },
		{ 2099, 4111, 0 },
		{ 3054, 3308, 0 },
		{ 3028, 3567, 0 },
		{ 1185, 4011, 0 },
		{ 3027, 3780, 0 },
		{ 2067, 3102, 0 },
		{ 2114, 3058, 0 },
		{ 2937, 4357, 0 },
		{ 1185, 4025, 0 },
		{ 2114, 3678, 0 },
		{ 3062, 3216, 0 },
		{ 2080, 3211, 0 },
		{ 2048, 3788, 0 },
		{ 2045, 3431, 0 },
		{ 3057, 3405, 0 },
		{ 0, 0, 116 },
		{ 2067, 3104, 0 },
		{ 2114, 3688, 0 },
		{ 2006, 3240, 0 },
		{ 3050, 3652, 0 },
		{ 3027, 3744, 0 },
		{ 2937, 4519, 0 },
		{ 2099, 4088, 0 },
		{ 0, 0, 7 },
		{ 2048, 3791, 0 },
		{ 0, 0, 6 },
		{ 2978, 3910, 0 },
		{ 0, 0, 121 },
		{ 3027, 3749, 0 },
		{ 2099, 4098, 0 },
		{ 3062, 1637, 0 },
		{ 2067, 3106, 0 },
		{ 3027, 3775, 0 },
		{ 3050, 3619, 0 },
		{ 0, 0, 125 },
		{ 2067, 3107, 0 },
		{ 2099, 4119, 0 },
		{ 3062, 3230, 0 },
		{ 2099, 4122, 0 },
		{ 2198, 4167, 0 },
		{ 2114, 3700, 0 },
		{ 0, 0, 71 },
		{ 2029, 3823, 0 },
		{ 2067, 3108, 108 },
		{ 2067, 3111, 109 },
		{ 2937, 4511, 0 },
		{ 2978, 3970, 0 },
		{ 3028, 3570, 0 },
		{ 3050, 3645, 0 },
		{ 3028, 3572, 0 },
		{ 1185, 4032, 0 },
		{ 3050, 3655, 0 },
		{ 3028, 3507, 0 },
		{ 3054, 3319, 0 },
		{ 2114, 3717, 0 },
		{ 2978, 3929, 0 },
		{ 2099, 4085, 0 },
		{ 2067, 3112, 0 },
		{ 3057, 3358, 0 },
		{ 2937, 4402, 0 },
		{ 2978, 3934, 0 },
		{ 2872, 4317, 0 },
		{ 2978, 3948, 0 },
		{ 3028, 3515, 0 },
		{ 2978, 3954, 0 },
		{ 0, 0, 9 },
		{ 2067, 3114, 0 },
		{ 2978, 3960, 0 },
		{ 2080, 3216, 0 },
		{ 2135, 3998, 0 },
		{ 0, 0, 106 },
		{ 2045, 3436, 0 },
		{ 3027, 3773, 0 },
		{ 3050, 3593, 0 },
		{ 2114, 3618, 0 },
		{ 3027, 3236, 0 },
		{ 2978, 3900, 0 },
		{ 3054, 3286, 0 },
		{ 2978, 3912, 0 },
		{ 3054, 3263, 0 },
		{ 3050, 3611, 0 },
		{ 2099, 4059, 0 },
		{ 3057, 3386, 0 },
		{ 3028, 3554, 0 },
		{ 3054, 3253, 0 },
		{ 3027, 3761, 0 },
		{ 3057, 3393, 0 },
		{ 2978, 3888, 0 },
		{ 3057, 3308, 0 },
		{ 2937, 4501, 0 },
		{ 2067, 3115, 0 },
		{ 2937, 4507, 0 },
		{ 3028, 3568, 0 },
		{ 2099, 4089, 0 },
		{ 3050, 3584, 0 },
		{ 3050, 3586, 0 },
		{ 2029, 3850, 0 },
		{ 2045, 3488, 0 },
		{ 2067, 3116, 96 },
		{ 3028, 3501, 0 },
		{ 2067, 3117, 0 },
		{ 2067, 3118, 0 },
		{ 3054, 3300, 0 },
		{ 2114, 3680, 0 },
		{ 2198, 4171, 0 },
		{ 2067, 3119, 0 },
		{ 2045, 3435, 0 },
		{ 0, 0, 105 },
		{ 2198, 4186, 0 },
		{ 2937, 4433, 0 },
		{ 3057, 3338, 0 },
		{ 3057, 3341, 0 },
		{ 0, 0, 118 },
		{ 0, 0, 120 },
		{ 2114, 3716, 0 },
		{ 2045, 3444, 0 },
		{ 2099, 3344, 0 },
		{ 3057, 3351, 0 },
		{ 2099, 4052, 0 },
		{ 2067, 3120, 0 },
		{ 2099, 4058, 0 },
		{ 2978, 3946, 0 },
		{ 3027, 3755, 0 },
		{ 2067, 3121, 0 },
		{ 2135, 3991, 0 },
		{ 3054, 3298, 0 },
		{ 2198, 4169, 0 },
		{ 2067, 3122, 0 },
		{ 2937, 4579, 0 },
		{ 2045, 3483, 0 },
		{ 945, 3877, 0 },
		{ 3057, 3359, 0 },
		{ 3027, 3776, 0 },
		{ 2114, 3703, 0 },
		{ 2198, 4239, 0 },
		{ 3057, 3370, 0 },
		{ 2007, 3284, 0 },
		{ 2067, 3125, 0 },
		{ 2978, 3908, 0 },
		{ 3050, 3646, 0 },
		{ 2045, 3426, 0 },
		{ 2937, 4453, 0 },
		{ 3057, 3376, 0 },
		{ 2114, 3686, 0 },
		{ 2080, 3213, 0 },
		{ 3028, 3509, 0 },
		{ 2045, 3432, 0 },
		{ 2114, 3701, 0 },
		{ 2048, 3807, 0 },
		{ 3062, 3291, 0 },
		{ 2067, 3126, 0 },
		{ 0, 0, 65 },
		{ 2067, 3130, 0 },
		{ 3050, 3630, 0 },
		{ 3028, 3522, 0 },
		{ 3050, 3637, 0 },
		{ 3028, 3524, 0 },
		{ 2067, 3131, 110 },
		{ 2029, 3852, 0 },
		{ 2114, 3685, 0 },
		{ 2048, 3808, 0 },
		{ 2045, 3441, 0 },
		{ 2045, 3443, 0 },
		{ 2029, 3818, 0 },
		{ 2048, 3814, 0 },
		{ 2937, 4431, 0 },
		{ 3050, 3590, 0 },
		{ 3054, 3316, 0 },
		{ 2978, 3913, 0 },
		{ 3057, 3395, 0 },
		{ 2045, 3446, 0 },
		{ 0, 0, 122 },
		{ 2872, 4319, 0 },
		{ 2048, 3794, 0 },
		{ 3057, 3398, 0 },
		{ 3027, 3778, 0 },
		{ 0, 0, 117 },
		{ 0, 0, 107 },
		{ 2045, 3463, 0 },
		{ 3028, 3560, 0 },
		{ 3057, 3402, 0 },
		{ 2114, 2974, 0 },
		{ 3062, 3264, 0 },
		{ 2978, 3951, 0 },
		{ 3027, 3740, 0 },
		{ 2067, 3136, 0 },
		{ 2978, 3957, 0 },
		{ 2029, 3821, 0 },
		{ 3050, 3639, 0 },
		{ 2099, 4037, 0 },
		{ 2937, 4595, 0 },
		{ 2937, 4355, 0 },
		{ 2045, 3474, 0 },
		{ 2937, 4363, 0 },
		{ 2978, 3963, 0 },
		{ 2937, 4367, 0 },
		{ 3028, 3569, 0 },
		{ 3027, 3758, 0 },
		{ 2067, 3137, 0 },
		{ 2099, 4056, 0 },
		{ 3028, 3571, 0 },
		{ 2067, 3138, 0 },
		{ 3028, 3576, 0 },
		{ 2099, 4066, 0 },
		{ 2978, 3905, 0 },
		{ 2099, 4071, 0 },
		{ 3028, 3579, 0 },
		{ 2099, 4075, 0 },
		{ 2045, 3487, 0 },
		{ 3027, 3779, 0 },
		{ 2067, 3142, 0 },
		{ 3062, 4179, 0 },
		{ 2198, 4243, 0 },
		{ 2099, 4082, 0 },
		{ 2048, 3789, 0 },
		{ 2099, 4084, 0 },
		{ 2048, 3790, 0 },
		{ 3050, 3597, 0 },
		{ 2937, 4541, 0 },
		{ 3050, 3621, 0 },
		{ 0, 0, 98 },
		{ 2978, 3923, 0 },
		{ 2978, 3927, 0 },
		{ 2937, 4591, 0 },
		{ 2067, 3143, 0 },
		{ 2067, 3144, 0 },
		{ 2937, 4597, 0 },
		{ 2937, 4599, 0 },
		{ 2937, 4353, 0 },
		{ 2048, 3801, 0 },
		{ 2045, 3425, 0 },
		{ 0, 0, 127 },
		{ 0, 0, 119 },
		{ 0, 0, 123 },
		{ 2937, 4361, 0 },
		{ 2198, 4245, 0 },
		{ 1073, 3228, 0 },
		{ 2067, 3146, 0 },
		{ 2978, 3065, 0 },
		{ 3028, 3523, 0 },
		{ 2048, 3812, 0 },
		{ 1185, 4013, 0 },
		{ 2937, 4429, 0 },
		{ 3050, 3641, 0 },
		{ 3050, 3642, 0 },
		{ 3050, 3643, 0 },
		{ 3027, 3769, 0 },
		{ 2198, 4200, 0 },
		{ 2135, 3987, 0 },
		{ 3027, 3770, 0 },
		{ 3054, 3313, 0 },
		{ 2029, 3846, 0 },
		{ 1185, 4012, 0 },
		{ 3057, 3355, 0 },
		{ 2029, 3849, 0 },
		{ 2045, 3434, 0 },
		{ 2067, 3148, 0 },
		{ 2048, 3799, 0 },
		{ 2029, 3860, 0 },
		{ 2099, 4069, 0 },
		{ 2135, 3992, 0 },
		{ 2114, 3677, 0 },
		{ 3028, 3535, 0 },
		{ 2937, 4577, 0 },
		{ 2114, 3679, 0 },
		{ 0, 0, 59 },
		{ 0, 0, 60 },
		{ 2937, 4585, 0 },
		{ 3028, 3536, 0 },
		{ 0, 0, 69 },
		{ 0, 0, 114 },
		{ 2007, 3288, 0 },
		{ 3054, 3322, 0 },
		{ 2045, 3440, 0 },
		{ 3054, 3295, 0 },
		{ 3057, 3368, 0 },
		{ 2978, 3928, 0 },
		{ 3028, 3555, 0 },
		{ 0, 0, 100 },
		{ 3028, 3556, 0 },
		{ 0, 0, 102 },
		{ 3050, 3635, 0 },
		{ 3028, 3558, 0 },
		{ 3027, 3766, 0 },
		{ 2099, 4102, 0 },
		{ 2080, 3217, 0 },
		{ 2080, 3226, 0 },
		{ 3057, 3372, 0 },
		{ 1185, 4031, 0 },
		{ 2135, 3733, 0 },
		{ 3028, 3563, 0 },
		{ 945, 3879, 0 },
		{ 2029, 3874, 0 },
		{ 0, 0, 115 },
		{ 0, 0, 126 },
		{ 3028, 3564, 0 },
		{ 0, 0, 139 },
		{ 2045, 3447, 0 },
		{ 3062, 3895, 0 },
		{ 2937, 4461, 0 },
		{ 1185, 4019, 0 },
		{ 2937, 4465, 0 },
		{ 2099, 4039, 0 },
		{ 3062, 4261, 0 },
		{ 3028, 3566, 0 },
		{ 3062, 4279, 0 },
		{ 3054, 3302, 0 },
		{ 3054, 3303, 0 },
		{ 3027, 3061, 0 },
		{ 2067, 3153, 0 },
		{ 2099, 4055, 0 },
		{ 2978, 3902, 0 },
		{ 2937, 4535, 0 },
		{ 2937, 4537, 0 },
		{ 2978, 3904, 0 },
		{ 2114, 3695, 0 },
		{ 2978, 3907, 0 },
		{ 2114, 3699, 0 },
		{ 2114, 3609, 0 },
		{ 2978, 3911, 0 },
		{ 2067, 3154, 0 },
		{ 2198, 4237, 0 },
		{ 2067, 3155, 0 },
		{ 3050, 3614, 0 },
		{ 2067, 3156, 0 },
		{ 2048, 3797, 0 },
		{ 3050, 3620, 0 },
		{ 2029, 3863, 0 },
		{ 2978, 3926, 0 },
		{ 2067, 3157, 0 },
		{ 3050, 3623, 0 },
		{ 3062, 4265, 0 },
		{ 1185, 4014, 0 },
		{ 2114, 3675, 0 },
		{ 3028, 3491, 0 },
		{ 2937, 4385, 0 },
		{ 2937, 4391, 0 },
		{ 2099, 4091, 0 },
		{ 2048, 3809, 0 },
		{ 2198, 4229, 0 },
		{ 3028, 3500, 0 },
		{ 2099, 4095, 0 },
		{ 2099, 4096, 0 },
		{ 2978, 3935, 0 },
		{ 2978, 3942, 0 },
		{ 2099, 4105, 0 },
		{ 2937, 4457, 0 },
		{ 2937, 4459, 0 },
		{ 2099, 4108, 0 },
		{ 2067, 3158, 0 },
		{ 3057, 3396, 0 },
		{ 3057, 3397, 0 },
		{ 2099, 4113, 0 },
		{ 2029, 3831, 0 },
		{ 3054, 3297, 0 },
		{ 2029, 3843, 0 },
		{ 3062, 4281, 0 },
		{ 3057, 3400, 0 },
		{ 2067, 3159, 62 },
		{ 3057, 3404, 0 },
		{ 2937, 4529, 0 },
		{ 2114, 3698, 0 },
		{ 2067, 3160, 0 },
		{ 2067, 3161, 0 },
		{ 2135, 3997, 0 },
		{ 2978, 3967, 0 },
		{ 3062, 4250, 0 },
		{ 3027, 3741, 0 },
		{ 3057, 3407, 0 },
		{ 3057, 3408, 0 },
		{ 1073, 3229, 0 },
		{ 2029, 3869, 0 },
		{ 3028, 3529, 0 },
		{ 2080, 3215, 0 },
		{ 2007, 3290, 0 },
		{ 2007, 3255, 0 },
		{ 3050, 3663, 0 },
		{ 2045, 3439, 0 },
		{ 1185, 4026, 0 },
		{ 3054, 3314, 0 },
		{ 3028, 3541, 0 },
		{ 3062, 4296, 0 },
		{ 3062, 4306, 0 },
		{ 2099, 4074, 0 },
		{ 0, 0, 63 },
		{ 0, 0, 66 },
		{ 2937, 4373, 0 },
		{ 1185, 4005, 0 },
		{ 2029, 3828, 0 },
		{ 3028, 3542, 0 },
		{ 1185, 4009, 0 },
		{ 3028, 3545, 0 },
		{ 0, 0, 111 },
		{ 3057, 3332, 0 },
		{ 3057, 3333, 0 },
		{ 3028, 3553, 0 },
		{ 0, 0, 104 },
		{ 2067, 3162, 0 },
		{ 2937, 4443, 0 },
		{ 0, 0, 112 },
		{ 0, 0, 113 },
		{ 2114, 3696, 0 },
		{ 2029, 3851, 0 },
		{ 3027, 3730, 0 },
		{ 945, 3878, 0 },
		{ 2048, 3796, 0 },
		{ 1185, 4028, 0 },
		{ 3057, 3339, 0 },
		{ 0, 0, 3 },
		{ 2099, 4097, 0 },
		{ 3062, 4294, 0 },
		{ 2937, 4469, 0 },
		{ 3027, 3735, 0 },
		{ 2978, 3945, 0 },
		{ 3057, 3340, 0 },
		{ 2978, 3947, 0 },
		{ 2099, 4109, 0 },
		{ 2067, 3163, 0 },
		{ 2048, 3804, 0 },
		{ 2198, 4192, 0 },
		{ 2045, 3459, 0 },
		{ 2045, 3460, 0 },
		{ 2099, 4114, 0 },
		{ 3027, 3748, 0 },
		{ 1015, 3977, 0 },
		{ 2099, 3067, 0 },
		{ 2978, 3959, 0 },
		{ 2114, 3709, 0 },
		{ 0, 0, 73 },
		{ 2099, 4124, 0 },
		{ 0, 0, 81 },
		{ 2937, 4581, 0 },
		{ 2099, 4125, 0 },
		{ 2937, 4587, 0 },
		{ 3057, 3348, 0 },
		{ 2099, 4129, 0 },
		{ 3054, 3323, 0 },
		{ 3062, 4289, 0 },
		{ 2099, 4131, 0 },
		{ 2114, 3718, 0 },
		{ 2029, 3826, 0 },
		{ 3057, 3352, 0 },
		{ 2029, 3829, 0 },
		{ 2978, 3972, 0 },
		{ 2114, 3723, 0 },
		{ 2978, 3895, 0 },
		{ 2099, 4053, 0 },
		{ 0, 0, 68 },
		{ 2114, 3670, 0 },
		{ 2114, 3671, 0 },
		{ 2937, 4375, 0 },
		{ 2048, 3792, 0 },
		{ 3057, 3353, 0 },
		{ 3027, 3774, 0 },
		{ 2099, 4061, 0 },
		{ 2114, 3673, 0 },
		{ 2099, 4063, 0 },
		{ 2067, 3165, 0 },
		{ 2978, 3909, 0 },
		{ 3050, 3648, 0 },
		{ 2048, 3798, 0 },
		{ 2029, 3858, 0 },
		{ 2045, 3472, 0 },
		{ 3062, 4288, 0 },
		{ 2045, 3473, 0 },
		{ 2067, 3166, 0 },
		{ 2114, 3681, 0 },
		{ 2007, 3271, 0 },
		{ 3062, 4311, 0 },
		{ 2099, 4080, 0 },
		{ 2099, 4081, 0 },
		{ 847, 3245, 0 },
		{ 0, 3249, 0 },
		{ 3027, 3736, 0 },
		{ 2135, 3979, 0 },
		{ 2099, 4087, 0 },
		{ 3028, 3577, 0 },
		{ 1185, 4015, 0 },
		{ 2937, 4525, 0 },
		{ 3028, 3578, 0 },
		{ 2067, 3167, 0 },
		{ 3057, 3361, 0 },
		{ 3028, 3494, 0 },
		{ 2029, 3825, 0 },
		{ 2978, 3937, 0 },
		{ 3028, 3498, 0 },
		{ 3027, 3750, 0 },
		{ 3057, 3362, 0 },
		{ 2198, 4155, 0 },
		{ 2198, 4157, 0 },
		{ 2937, 4583, 0 },
		{ 2099, 4104, 0 },
		{ 0, 0, 67 },
		{ 2029, 3830, 0 },
		{ 3057, 3363, 0 },
		{ 3050, 3634, 0 },
		{ 3028, 3502, 0 },
		{ 3028, 3504, 0 },
		{ 3028, 3506, 0 },
		{ 3057, 3364, 0 },
		{ 2114, 3720, 0 },
		{ 2114, 3721, 0 },
		{ 0, 0, 131 },
		{ 0, 0, 132 },
		{ 2048, 3800, 0 },
		{ 1185, 4018, 0 },
		{ 3057, 3365, 0 },
		{ 2029, 3855, 0 },
		{ 2029, 3857, 0 },
		{ 0, 0, 10 },
		{ 2937, 4369, 0 },
		{ 2045, 3428, 0 },
		{ 3057, 3366, 0 },
		{ 2937, 4009, 0 },
		{ 3062, 4293, 0 },
		{ 3027, 3777, 0 },
		{ 2937, 4387, 0 },
		{ 2937, 4389, 0 },
		{ 3057, 3367, 0 },
		{ 2067, 3168, 0 },
		{ 2978, 3896, 0 },
		{ 2978, 3897, 0 },
		{ 2099, 4040, 0 },
		{ 2067, 3169, 0 },
		{ 3062, 4255, 0 },
		{ 2937, 4441, 0 },
		{ 3062, 4259, 0 },
		{ 2029, 3872, 0 },
		{ 0, 0, 82 },
		{ 2114, 3674, 0 },
		{ 2978, 3903, 0 },
		{ 0, 0, 80 },
		{ 3054, 3311, 0 },
		{ 2048, 3813, 0 },
		{ 0, 0, 83 },
		{ 3062, 4285, 0 },
		{ 2978, 3906, 0 },
		{ 3054, 3312, 0 },
		{ 2099, 4057, 0 },
		{ 2045, 3437, 0 },
		{ 3028, 3526, 0 },
		{ 2099, 4060, 0 },
		{ 2067, 3170, 0 },
		{ 3057, 3374, 0 },
		{ 0, 0, 61 },
		{ 0, 0, 99 },
		{ 0, 0, 101 },
		{ 2114, 3683, 0 },
		{ 2198, 4163, 0 },
		{ 3028, 3530, 0 },
		{ 2099, 4067, 0 },
		{ 2978, 3915, 0 },
		{ 3050, 3609, 0 },
		{ 2099, 4070, 0 },
		{ 0, 0, 137 },
		{ 3028, 3531, 0 },
		{ 2099, 4072, 0 },
		{ 2978, 3920, 0 },
		{ 3057, 3375, 0 },
		{ 3028, 3534, 0 },
		{ 2937, 4575, 0 },
		{ 2067, 3171, 0 },
		{ 2029, 3838, 0 },
		{ 2029, 3842, 0 },
		{ 2099, 4079, 0 },
		{ 2198, 4241, 0 },
		{ 3057, 3377, 0 },
		{ 3057, 3379, 0 },
		{ 3028, 3539, 0 },
		{ 2135, 3980, 0 },
		{ 0, 3880, 0 },
		{ 3057, 3380, 0 },
		{ 3057, 3381, 0 },
		{ 2978, 3936, 0 },
		{ 3050, 3633, 0 },
		{ 2114, 3704, 0 },
		{ 2937, 4359, 0 },
		{ 2978, 3944, 0 },
		{ 3057, 3382, 0 },
		{ 2114, 3711, 0 },
		{ 3062, 4292, 0 },
		{ 0, 0, 19 },
		{ 2045, 3449, 0 },
		{ 2045, 3450, 0 },
		{ 0, 0, 128 },
		{ 2045, 3451, 0 },
		{ 0, 0, 130 },
		{ 3028, 3551, 0 },
		{ 2099, 4100, 0 },
		{ 0, 0, 97 },
		{ 2067, 3172, 0 },
		{ 2029, 3866, 0 },
		{ 2029, 3867, 0 },
		{ 2067, 3173, 0 },
		{ 2937, 4393, 0 },
		{ 2045, 3461, 0 },
		{ 2114, 3669, 0 },
		{ 2978, 3962, 0 },
		{ 0, 0, 78 },
		{ 2029, 3873, 0 },
		{ 1185, 4021, 0 },
		{ 2067, 3174, 0 },
		{ 2029, 3816, 0 },
		{ 3028, 3557, 0 },
		{ 2099, 4117, 0 },
		{ 3062, 4290, 0 },
		{ 3062, 4291, 0 },
		{ 2937, 4455, 0 },
		{ 2099, 4118, 0 },
		{ 2978, 3968, 0 },
		{ 2978, 3969, 0 },
		{ 2067, 3175, 0 },
		{ 2045, 3464, 0 },
		{ 3057, 3387, 0 },
		{ 3027, 3739, 0 },
		{ 3057, 3388, 0 },
		{ 2045, 3467, 0 },
		{ 2978, 3898, 0 },
		{ 3027, 3743, 0 },
		{ 3057, 3389, 0 },
		{ 2099, 4038, 0 },
		{ 0, 0, 86 },
		{ 3062, 4267, 0 },
		{ 0, 0, 103 },
		{ 2029, 3827, 0 },
		{ 3062, 3892, 0 },
		{ 2099, 4041, 0 },
		{ 0, 0, 135 },
		{ 3057, 3390, 0 },
		{ 3057, 3391, 0 },
		{ 2067, 3176, 58 },
		{ 3027, 3751, 0 },
		{ 2114, 3682, 0 },
		{ 2029, 3837, 0 },
		{ 3054, 3296, 0 },
		{ 2872, 3851, 0 },
		{ 0, 0, 87 },
		{ 2045, 3482, 0 },
		{ 3062, 4301, 0 },
		{ 1015, 3975, 0 },
		{ 0, 3976, 0 },
		{ 3057, 3394, 0 },
		{ 3027, 3763, 0 },
		{ 3027, 3765, 0 },
		{ 2114, 3687, 0 },
		{ 3062, 4257, 0 },
		{ 2099, 4064, 0 },
		{ 2978, 3919, 0 },
		{ 2067, 3177, 0 },
		{ 2114, 3689, 0 },
		{ 2114, 3692, 0 },
		{ 2114, 3693, 0 },
		{ 0, 0, 91 },
		{ 2978, 3925, 0 },
		{ 2045, 3486, 0 },
		{ 3028, 3573, 0 },
		{ 0, 0, 124 },
		{ 2067, 3178, 0 },
		{ 3054, 3299, 0 },
		{ 2067, 3179, 0 },
		{ 3050, 3631, 0 },
		{ 2978, 3933, 0 },
		{ 2198, 4188, 0 },
		{ 2045, 3418, 0 },
		{ 3027, 3782, 0 },
		{ 0, 0, 93 },
		{ 3027, 3785, 0 },
		{ 2198, 4196, 0 },
		{ 2198, 4198, 0 },
		{ 3057, 3399, 0 },
		{ 0, 0, 15 },
		{ 2029, 3871, 0 },
		{ 0, 0, 17 },
		{ 0, 0, 18 },
		{ 2978, 3943, 0 },
		{ 2067, 3180, 0 },
		{ 2135, 3986, 0 },
		{ 3027, 3729, 0 },
		{ 2937, 4439, 0 },
		{ 3028, 3495, 0 },
		{ 2114, 3705, 0 },
		{ 1185, 4016, 0 },
		{ 3028, 3496, 0 },
		{ 3028, 3497, 0 },
		{ 3027, 3737, 0 },
		{ 2114, 3715, 0 },
		{ 0, 0, 57 },
		{ 2978, 3956, 0 },
		{ 3057, 3401, 0 },
		{ 2067, 3181, 0 },
		{ 3057, 3403, 0 },
		{ 2029, 3824, 0 },
		{ 2114, 3719, 0 },
		{ 2099, 4107, 0 },
		{ 0, 0, 76 },
		{ 2067, 3182, 0 },
		{ 3062, 4305, 0 },
		{ 3028, 3503, 0 },
		{ 0, 3236, 0 },
		{ 3028, 3505, 0 },
		{ 0, 0, 16 },
		{ 2114, 3666, 0 },
		{ 1185, 4010, 0 },
		{ 2067, 3183, 56 },
		{ 2067, 3184, 0 },
		{ 2029, 3836, 0 },
		{ 0, 0, 77 },
		{ 3027, 3757, 0 },
		{ 3062, 3280, 0 },
		{ 0, 0, 84 },
		{ 0, 0, 85 },
		{ 0, 0, 54 },
		{ 3027, 3760, 0 },
		{ 3050, 3656, 0 },
		{ 3050, 3658, 0 },
		{ 3057, 3409, 0 },
		{ 3050, 3660, 0 },
		{ 0, 0, 136 },
		{ 3027, 3767, 0 },
		{ 1185, 4022, 0 },
		{ 1185, 4023, 0 },
		{ 3057, 3410, 0 },
		{ 0, 0, 39 },
		{ 2067, 3064, 40 },
		{ 2067, 3074, 42 },
		{ 3027, 3771, 0 },
		{ 3062, 4295, 0 },
		{ 1185, 4029, 0 },
		{ 1185, 4030, 0 },
		{ 2029, 3854, 0 },
		{ 0, 0, 74 },
		{ 3027, 3772, 0 },
		{ 3057, 3413, 0 },
		{ 0, 0, 92 },
		{ 3057, 3414, 0 },
		{ 2029, 3859, 0 },
		{ 3050, 3615, 0 },
		{ 2029, 3861, 0 },
		{ 2045, 3442, 0 },
		{ 2978, 3914, 0 },
		{ 3054, 3318, 0 },
		{ 2978, 3916, 0 },
		{ 2135, 3981, 0 },
		{ 2135, 3985, 0 },
		{ 2067, 3075, 0 },
		{ 3057, 3416, 0 },
		{ 3062, 4284, 0 },
		{ 3054, 3321, 0 },
		{ 0, 0, 88 },
		{ 3062, 4286, 0 },
		{ 2067, 3076, 0 },
		{ 0, 0, 129 },
		{ 0, 0, 133 },
		{ 2029, 3870, 0 },
		{ 0, 0, 138 },
		{ 0, 0, 11 },
		{ 3027, 3783, 0 },
		{ 3027, 3784, 0 },
		{ 2114, 3690, 0 },
		{ 3050, 3627, 0 },
		{ 3050, 3629, 0 },
		{ 1185, 4027, 0 },
		{ 2067, 3077, 0 },
		{ 3057, 3334, 0 },
		{ 3027, 3731, 0 },
		{ 3057, 3337, 0 },
		{ 3062, 4309, 0 },
		{ 0, 0, 134 },
		{ 2978, 3932, 0 },
		{ 3062, 4312, 0 },
		{ 3027, 3734, 0 },
		{ 3054, 3325, 0 },
		{ 3054, 3326, 0 },
		{ 3054, 3327, 0 },
		{ 3062, 4256, 0 },
		{ 2067, 3078, 0 },
		{ 3062, 4258, 0 },
		{ 2978, 3938, 0 },
		{ 2937, 4505, 0 },
		{ 3057, 3342, 0 },
		{ 3057, 3344, 0 },
		{ 2067, 3079, 0 },
		{ 0, 0, 41 },
		{ 0, 0, 43 },
		{ 3027, 3742, 0 },
		{ 2937, 4515, 0 },
		{ 3062, 4273, 0 },
		{ 3057, 3346, 0 },
		{ 2114, 3706, 0 },
		{ 2029, 3832, 0 },
		{ 2978, 3949, 0 },
		{ 2978, 3950, 0 },
		{ 2872, 4325, 0 },
		{ 3062, 4287, 0 },
		{ 2029, 3833, 0 },
		{ 2937, 4546, 0 },
		{ 2978, 3952, 0 },
		{ 3027, 3745, 0 },
		{ 2029, 3835, 0 },
		{ 2114, 3707, 0 },
		{ 2114, 3708, 0 },
		{ 2099, 4103, 0 },
		{ 3057, 3347, 0 },
		{ 2029, 3839, 0 },
		{ 2029, 3840, 0 },
		{ 2114, 3710, 0 },
		{ 0, 0, 79 },
		{ 0, 0, 94 },
		{ 3027, 3752, 0 },
		{ 3027, 3753, 0 },
		{ 0, 4033, 0 },
		{ 2978, 3964, 0 },
		{ 0, 0, 89 },
		{ 2029, 3844, 0 },
		{ 3027, 3754, 0 },
		{ 2045, 3468, 0 },
		{ 0, 0, 12 },
		{ 2114, 3712, 0 },
		{ 2114, 3713, 0 },
		{ 0, 0, 90 },
		{ 0, 0, 55 },
		{ 0, 0, 95 },
		{ 3028, 3550, 0 },
		{ 3027, 3759, 0 },
		{ 2099, 4120, 0 },
		{ 0, 0, 14 },
		{ 2067, 3080, 0 },
		{ 3028, 3552, 0 },
		{ 2099, 4123, 0 },
		{ 3050, 3649, 0 },
		{ 2029, 3856, 0 },
		{ 2937, 4383, 0 },
		{ 3062, 4277, 0 },
		{ 2099, 4126, 0 },
		{ 2048, 3802, 0 },
		{ 2099, 4128, 0 },
		{ 3027, 3764, 0 },
		{ 3057, 3349, 0 },
		{ 0, 0, 13 },
		{ 3106, 1437, 231 },
		{ 0, 0, 232 },
		{ 3061, 4922, 233 },
		{ 3089, 1644, 237 },
		{ 1222, 2536, 238 },
		{ 0, 0, 238 },
		{ 3089, 1733, 234 },
		{ 1225, 1452, 0 },
		{ 3089, 1766, 235 },
		{ 1228, 1485, 0 },
		{ 1225, 0, 0 },
		{ 3013, 2655, 236 },
		{ 1230, 1519, 0 },
		{ 1228, 0, 0 },
		{ 3013, 2689, 234 },
		{ 1230, 0, 0 },
		{ 3013, 2699, 235 },
		{ 3054, 3324, 147 },
		{ 0, 0, 147 },
		{ 0, 0, 148 },
		{ 3067, 1990, 0 },
		{ 3089, 2861, 0 },
		{ 3106, 2063, 0 },
		{ 1238, 4757, 0 },
		{ 3086, 2578, 0 },
		{ 3089, 2791, 0 },
		{ 3101, 2937, 0 },
		{ 3097, 2468, 0 },
		{ 3100, 2998, 0 },
		{ 3106, 2097, 0 },
		{ 3100, 3033, 0 },
		{ 3102, 1741, 0 },
		{ 3004, 2689, 0 },
		{ 3104, 2180, 0 },
		{ 3058, 2235, 0 },
		{ 3067, 1975, 0 },
		{ 3107, 4405, 0 },
		{ 0, 0, 145 },
		{ 2872, 4324, 155 },
		{ 0, 0, 155 },
		{ 0, 0, 156 },
		{ 3089, 2821, 0 },
		{ 2950, 2748, 0 },
		{ 3104, 2188, 0 },
		{ 3106, 2037, 0 },
		{ 3089, 2817, 0 },
		{ 1261, 4815, 0 },
		{ 3089, 2508, 0 },
		{ 3071, 1463, 0 },
		{ 3089, 2899, 0 },
		{ 3106, 2064, 0 },
		{ 2716, 1445, 0 },
		{ 3102, 1916, 0 },
		{ 3096, 2730, 0 },
		{ 3004, 2536, 0 },
		{ 3058, 2295, 0 },
		{ 2906, 2741, 0 },
		{ 1272, 4787, 0 },
		{ 3089, 2510, 0 },
		{ 3097, 2454, 0 },
		{ 3067, 1967, 0 },
		{ 3089, 2843, 0 },
		{ 1277, 4777, 0 },
		{ 3099, 2437, 0 },
		{ 3094, 1600, 0 },
		{ 3058, 2290, 0 },
		{ 3101, 2935, 0 },
		{ 3102, 1858, 0 },
		{ 3004, 2672, 0 },
		{ 3104, 2159, 0 },
		{ 3058, 2245, 0 },
		{ 3107, 4531, 0 },
		{ 0, 0, 153 },
		{ 3054, 3320, 179 },
		{ 0, 0, 179 },
		{ 3067, 1999, 0 },
		{ 3089, 2898, 0 },
		{ 3106, 2054, 0 },
		{ 1293, 4815, 0 },
		{ 3101, 2616, 0 },
		{ 3097, 2416, 0 },
		{ 3100, 3013, 0 },
		{ 3067, 2010, 0 },
		{ 3067, 2026, 0 },
		{ 3089, 2840, 0 },
		{ 3067, 2029, 0 },
		{ 3107, 4547, 0 },
		{ 0, 0, 178 },
		{ 2135, 3984, 143 },
		{ 0, 0, 143 },
		{ 0, 0, 144 },
		{ 3089, 2858, 0 },
		{ 3058, 2247, 0 },
		{ 3104, 2196, 0 },
		{ 3091, 2361, 0 },
		{ 3089, 2786, 0 },
		{ 3062, 4297, 0 },
		{ 3097, 2426, 0 },
		{ 3100, 2996, 0 },
		{ 3067, 2034, 0 },
		{ 3067, 1965, 0 },
		{ 3069, 4672, 0 },
		{ 3069, 4668, 0 },
		{ 3004, 2675, 0 },
		{ 3058, 2304, 0 },
		{ 3004, 2706, 0 },
		{ 3102, 1866, 0 },
		{ 3004, 2567, 0 },
		{ 3100, 2995, 0 },
		{ 3097, 2450, 0 },
		{ 3004, 2673, 0 },
		{ 3067, 1527, 0 },
		{ 3089, 2789, 0 },
		{ 3106, 2077, 0 },
		{ 3107, 4535, 0 },
		{ 0, 0, 141 },
		{ 2622, 2962, 23 },
		{ 0, 0, 23 },
		{ 0, 0, 24 },
		{ 3089, 2806, 0 },
		{ 2906, 2733, 0 },
		{ 3004, 2700, 0 },
		{ 3058, 2276, 0 },
		{ 3061, 7, 0 },
		{ 3104, 2178, 0 },
		{ 2864, 2121, 0 },
		{ 3089, 2851, 0 },
		{ 3106, 2084, 0 },
		{ 3100, 3035, 0 },
		{ 3102, 1934, 0 },
		{ 3104, 2150, 0 },
		{ 3106, 2105, 0 },
		{ 3061, 2, 0 },
		{ 3086, 2927, 0 },
		{ 3089, 2906, 0 },
		{ 3067, 1998, 0 },
		{ 3101, 2936, 0 },
		{ 3106, 2041, 0 },
		{ 3004, 2710, 0 },
		{ 2864, 2141, 0 },
		{ 3102, 1683, 0 },
		{ 3004, 2626, 0 },
		{ 3104, 2218, 0 },
		{ 3058, 2302, 0 },
		{ 3061, 4907, 0 },
		{ 3069, 4652, 0 },
		{ 0, 0, 20 },
		{ 1376, 0, 1 },
		{ 1376, 0, 180 },
		{ 1376, 2756, 230 },
		{ 1591, 173, 230 },
		{ 1591, 411, 230 },
		{ 1591, 403, 230 },
		{ 1591, 526, 230 },
		{ 1591, 404, 230 },
		{ 1591, 415, 230 },
		{ 1591, 390, 230 },
		{ 1591, 421, 230 },
		{ 1591, 483, 230 },
		{ 1376, 0, 230 },
		{ 1388, 2497, 230 },
		{ 1376, 2781, 230 },
		{ 2622, 2960, 226 },
		{ 1591, 504, 230 },
		{ 1591, 502, 230 },
		{ 1591, 530, 230 },
		{ 1591, 0, 230 },
		{ 1591, 565, 230 },
		{ 1591, 553, 230 },
		{ 3104, 2160, 0 },
		{ 0, 0, 181 },
		{ 3058, 2300, 0 },
		{ 1591, 520, 0 },
		{ 1591, 0, 0 },
		{ 3061, 3942, 0 },
		{ 1591, 540, 0 },
		{ 1591, 556, 0 },
		{ 1591, 579, 0 },
		{ 1591, 587, 0 },
		{ 1591, 578, 0 },
		{ 1591, 582, 0 },
		{ 1591, 616, 0 },
		{ 1591, 597, 0 },
		{ 1591, 591, 0 },
		{ 1591, 583, 0 },
		{ 1591, 587, 0 },
		{ 3089, 2874, 0 },
		{ 3089, 2888, 0 },
		{ 1592, 595, 0 },
		{ 1592, 597, 0 },
		{ 1591, 618, 0 },
		{ 1591, 619, 0 },
		{ 1591, 610, 0 },
		{ 3104, 2182, 0 },
		{ 3086, 2920, 0 },
		{ 1591, 608, 0 },
		{ 1591, 652, 0 },
		{ 1591, 629, 0 },
		{ 1591, 630, 0 },
		{ 1591, 679, 0 },
		{ 1591, 680, 0 },
		{ 1591, 685, 0 },
		{ 1591, 679, 0 },
		{ 1591, 655, 0 },
		{ 1591, 646, 0 },
		{ 1591, 679, 0 },
		{ 1591, 694, 0 },
		{ 1591, 681, 0 },
		{ 3058, 2326, 0 },
		{ 2950, 2775, 0 },
		{ 1591, 697, 0 },
		{ 1591, 17, 0 },
		{ 1592, 27, 0 },
		{ 1591, 26, 0 },
		{ 1591, 30, 0 },
		{ 3097, 2462, 0 },
		{ 0, 0, 229 },
		{ 1591, 42, 0 },
		{ 1591, 26, 0 },
		{ 1591, 14, 0 },
		{ 1591, 64, 0 },
		{ 1591, 68, 0 },
		{ 1591, 69, 0 },
		{ 1591, 65, 0 },
		{ 1591, 56, 0 },
		{ 1591, 34, 0 },
		{ 1591, 39, 0 },
		{ 1591, 55, 0 },
		{ 1591, 0, 215 },
		{ 1591, 91, 0 },
		{ 3104, 2157, 0 },
		{ 3004, 2541, 0 },
		{ 1591, 49, 0 },
		{ 1591, 68, 0 },
		{ 1591, 64, 0 },
		{ 1591, 91, 0 },
		{ 1591, 93, 0 },
		{ -1471, 1092, 0 },
		{ 1592, 102, 0 },
		{ 1591, 164, 0 },
		{ 1591, 183, 0 },
		{ 1591, 175, 0 },
		{ 1591, 185, 0 },
		{ 1591, 186, 0 },
		{ 1591, 165, 0 },
		{ 1591, 181, 0 },
		{ 1591, 161, 0 },
		{ 1591, 152, 0 },
		{ 1591, 0, 214 },
		{ 1591, 159, 0 },
		{ 3091, 2381, 0 },
		{ 3058, 2279, 0 },
		{ 1591, 163, 0 },
		{ 1591, 173, 0 },
		{ 1591, 171, 0 },
		{ 1591, 0, 228 },
		{ 1591, 180, 0 },
		{ 0, 0, 216 },
		{ 1591, 173, 0 },
		{ 1593, 4, -4 },
		{ 1591, 201, 0 },
		{ 1591, 214, 0 },
		{ 1591, 273, 0 },
		{ 1591, 279, 0 },
		{ 1591, 214, 0 },
		{ 1591, 252, 0 },
		{ 1591, 225, 0 },
		{ 1591, 231, 0 },
		{ 1591, 263, 0 },
		{ 3089, 2832, 0 },
		{ 3089, 2835, 0 },
		{ 1591, 0, 218 },
		{ 1591, 303, 219 },
		{ 1591, 272, 0 },
		{ 1591, 275, 0 },
		{ 1591, 302, 0 },
		{ 1491, 3558, 0 },
		{ 3061, 4278, 0 },
		{ 2176, 4615, 205 },
		{ 1591, 305, 0 },
		{ 1591, 309, 0 },
		{ 1591, 313, 0 },
		{ 1591, 315, 0 },
		{ 1591, 319, 0 },
		{ 1591, 321, 0 },
		{ 1591, 322, 0 },
		{ 1591, 323, 0 },
		{ 1591, 306, 0 },
		{ 1591, 340, 0 },
		{ 1592, 328, 0 },
		{ 3062, 4298, 0 },
		{ 3061, 4924, 221 },
		{ 1591, 332, 0 },
		{ 1591, 377, 0 },
		{ 1591, 359, 0 },
		{ 1591, 375, 0 },
		{ 0, 0, 185 },
		{ 1593, 31, -7 },
		{ 1593, 117, -10 },
		{ 1593, 231, -13 },
		{ 1593, 345, -16 },
		{ 1593, 376, -19 },
		{ 1593, 460, -22 },
		{ 1591, 407, 0 },
		{ 1591, 420, 0 },
		{ 1591, 393, 0 },
		{ 1591, 0, 203 },
		{ 1591, 0, 217 },
		{ 3097, 2430, 0 },
		{ 1591, 392, 0 },
		{ 1591, 383, 0 },
		{ 1591, 391, 0 },
		{ 1592, 392, 0 },
		{ 1528, 3527, 0 },
		{ 3061, 4310, 0 },
		{ 2176, 4631, 206 },
		{ 1531, 3528, 0 },
		{ 3061, 4274, 0 },
		{ 2176, 4646, 207 },
		{ 1534, 3529, 0 },
		{ 3061, 4282, 0 },
		{ 2176, 4634, 210 },
		{ 1537, 3530, 0 },
		{ 3061, 4198, 0 },
		{ 2176, 4627, 211 },
		{ 1540, 3531, 0 },
		{ 3061, 4272, 0 },
		{ 2176, 4636, 212 },
		{ 1543, 3532, 0 },
		{ 3061, 4276, 0 },
		{ 2176, 4622, 213 },
		{ 1591, 451, 0 },
		{ 1593, 488, -25 },
		{ 1591, 424, 0 },
		{ 3100, 2983, 0 },
		{ 1591, 437, 0 },
		{ 1591, 483, 0 },
		{ 1591, 481, 0 },
		{ 1591, 492, 0 },
		{ 0, 0, 187 },
		{ 0, 0, 189 },
		{ 0, 0, 195 },
		{ 0, 0, 197 },
		{ 0, 0, 199 },
		{ 0, 0, 201 },
		{ 1593, 490, -28 },
		{ 1561, 3543, 0 },
		{ 3061, 4286, 0 },
		{ 2176, 4641, 209 },
		{ 1591, 0, 202 },
		{ 3067, 2031, 0 },
		{ 1591, 480, 0 },
		{ 1591, 495, 0 },
		{ 1592, 488, 0 },
		{ 1591, 486, 0 },
		{ 1570, 3549, 0 },
		{ 3061, 4280, 0 },
		{ 2176, 4644, 208 },
		{ 0, 0, 193 },
		{ 3067, 1979, 0 },
		{ 1591, 4, 224 },
		{ 1592, 492, 0 },
		{ 1591, 3, 227 },
		{ 1591, 508, 0 },
		{ 0, 0, 191 },
		{ 3069, 4673, 0 },
		{ 3069, 4651, 0 },
		{ 1591, 496, 0 },
		{ 0, 0, 225 },
		{ 1591, 493, 0 },
		{ 3069, 4669, 0 },
		{ 0, 0, 223 },
		{ 1591, 504, 0 },
		{ 1591, 509, 0 },
		{ 0, 0, 222 },
		{ 1591, 514, 0 },
		{ 1591, 505, 0 },
		{ 1592, 507, 220 },
		{ 1593, 925, 0 },
		{ 1594, 736, -1 },
		{ 1595, 3503, 0 },
		{ 3061, 4242, 0 },
		{ 2176, 4639, 204 },
		{ 0, 0, 183 },
		{ 2135, 3989, 273 },
		{ 0, 0, 273 },
		{ 3089, 2891, 0 },
		{ 3058, 2329, 0 },
		{ 3104, 2214, 0 },
		{ 3091, 2391, 0 },
		{ 3089, 2783, 0 },
		{ 3062, 4300, 0 },
		{ 3097, 2438, 0 },
		{ 3100, 3006, 0 },
		{ 3067, 1993, 0 },
		{ 3067, 1994, 0 },
		{ 3069, 4658, 0 },
		{ 3069, 4660, 0 },
		{ 3004, 2496, 0 },
		{ 3058, 2273, 0 },
		{ 3004, 2538, 0 },
		{ 3102, 1919, 0 },
		{ 3004, 2555, 0 },
		{ 3100, 3000, 0 },
		{ 3097, 2414, 0 },
		{ 3004, 2574, 0 },
		{ 3067, 1532, 0 },
		{ 3089, 2848, 0 },
		{ 3106, 2092, 0 },
		{ 3107, 4543, 0 },
		{ 0, 0, 272 },
		{ 2135, 3983, 275 },
		{ 0, 0, 275 },
		{ 0, 0, 276 },
		{ 3089, 2855, 0 },
		{ 3058, 2289, 0 },
		{ 3104, 2161, 0 },
		{ 3091, 2385, 0 },
		{ 3089, 2875, 0 },
		{ 3062, 4283, 0 },
		{ 3097, 2452, 0 },
		{ 3100, 3031, 0 },
		{ 3067, 2006, 0 },
		{ 3067, 2007, 0 },
		{ 3069, 4662, 0 },
		{ 3069, 4665, 0 },
		{ 3101, 2951, 0 },
		{ 3106, 2107, 0 },
		{ 3104, 2185, 0 },
		{ 3067, 2008, 0 },
		{ 3067, 2009, 0 },
		{ 3104, 2204, 0 },
		{ 3071, 1492, 0 },
		{ 3089, 2794, 0 },
		{ 3106, 2055, 0 },
		{ 3107, 4551, 0 },
		{ 0, 0, 274 },
		{ 2135, 3990, 278 },
		{ 0, 0, 278 },
		{ 0, 0, 279 },
		{ 3089, 2807, 0 },
		{ 3058, 2260, 0 },
		{ 3104, 2147, 0 },
		{ 3091, 2382, 0 },
		{ 3089, 2822, 0 },
		{ 3062, 4310, 0 },
		{ 3097, 2466, 0 },
		{ 3100, 3004, 0 },
		{ 3067, 2014, 0 },
		{ 3067, 2022, 0 },
		{ 3069, 4670, 0 },
		{ 3069, 4671, 0 },
		{ 3091, 2392, 0 },
		{ 3094, 1552, 0 },
		{ 3102, 1742, 0 },
		{ 3100, 2981, 0 },
		{ 3102, 1795, 0 },
		{ 3104, 2169, 0 },
		{ 3106, 2093, 0 },
		{ 3107, 4434, 0 },
		{ 0, 0, 277 },
		{ 2135, 3994, 281 },
		{ 0, 0, 281 },
		{ 0, 0, 282 },
		{ 3089, 2864, 0 },
		{ 3058, 2301, 0 },
		{ 3104, 2179, 0 },
		{ 3091, 2365, 0 },
		{ 3089, 2890, 0 },
		{ 3062, 4282, 0 },
		{ 3097, 2467, 0 },
		{ 3100, 3032, 0 },
		{ 3067, 2032, 0 },
		{ 3067, 2033, 0 },
		{ 3069, 4666, 0 },
		{ 3069, 4667, 0 },
		{ 3089, 2903, 0 },
		{ 3071, 1494, 0 },
		{ 3100, 2991, 0 },
		{ 3097, 2424, 0 },
		{ 3094, 1538, 0 },
		{ 3100, 2997, 0 },
		{ 3102, 1907, 0 },
		{ 3104, 2201, 0 },
		{ 3106, 2042, 0 },
		{ 3107, 4467, 0 },
		{ 0, 0, 280 },
		{ 2135, 4001, 284 },
		{ 0, 0, 284 },
		{ 0, 0, 285 },
		{ 3089, 2797, 0 },
		{ 3058, 2263, 0 },
		{ 3104, 2208, 0 },
		{ 3091, 2383, 0 },
		{ 3089, 2813, 0 },
		{ 3062, 4308, 0 },
		{ 3097, 2465, 0 },
		{ 3100, 2980, 0 },
		{ 3067, 1968, 0 },
		{ 3067, 1971, 0 },
		{ 3069, 4653, 0 },
		{ 3069, 4656, 0 },
		{ 3104, 2219, 0 },
		{ 2864, 2127, 0 },
		{ 3102, 1909, 0 },
		{ 3004, 2624, 0 },
		{ 3091, 2369, 0 },
		{ 3004, 2669, 0 },
		{ 3067, 1976, 0 },
		{ 3089, 2857, 0 },
		{ 3106, 2070, 0 },
		{ 3107, 4613, 0 },
		{ 0, 0, 283 },
		{ 2937, 4474, 159 },
		{ 0, 0, 159 },
		{ 0, 0, 160 },
		{ 2950, 2751, 0 },
		{ 3102, 1915, 0 },
		{ 3089, 2869, 0 },
		{ 3106, 2079, 0 },
		{ 1735, 4732, 0 },
		{ 3089, 2503, 0 },
		{ 3071, 1497, 0 },
		{ 3089, 2889, 0 },
		{ 3106, 2088, 0 },
		{ 2716, 1456, 0 },
		{ 3102, 1925, 0 },
		{ 3096, 2721, 0 },
		{ 3004, 2498, 0 },
		{ 3058, 2243, 0 },
		{ 2906, 2735, 0 },
		{ 1746, 4769, 0 },
		{ 3089, 2497, 0 },
		{ 3097, 2474, 0 },
		{ 3067, 1997, 0 },
		{ 3089, 2793, 0 },
		{ 1751, 4813, 0 },
		{ 3099, 2439, 0 },
		{ 3094, 1627, 0 },
		{ 3058, 2255, 0 },
		{ 3101, 2955, 0 },
		{ 3102, 1936, 0 },
		{ 3004, 2628, 0 },
		{ 3104, 2192, 0 },
		{ 3058, 2269, 0 },
		{ 3107, 4403, 0 },
		{ 0, 0, 157 },
		{ 2135, 3993, 266 },
		{ 0, 0, 266 },
		{ 3089, 2818, 0 },
		{ 3058, 2270, 0 },
		{ 3104, 2195, 0 },
		{ 3091, 2373, 0 },
		{ 3089, 2833, 0 },
		{ 3062, 4248, 0 },
		{ 3097, 2464, 0 },
		{ 3100, 3016, 0 },
		{ 3067, 2000, 0 },
		{ 3067, 2001, 0 },
		{ 3069, 4663, 0 },
		{ 3069, 4664, 0 },
		{ 3086, 2929, 0 },
		{ 3004, 2708, 0 },
		{ 3067, 2004, 0 },
		{ 2864, 2129, 0 },
		{ 3097, 2469, 0 },
		{ 3100, 2987, 0 },
		{ 2716, 1412, 0 },
		{ 3107, 4399, 0 },
		{ 0, 0, 264 },
		{ 1799, 0, 1 },
		{ 1958, 2831, 381 },
		{ 3089, 2866, 381 },
		{ 3100, 2919, 381 },
		{ 3086, 2155, 381 },
		{ 1799, 0, 348 },
		{ 1799, 2625, 381 },
		{ 3096, 1598, 381 },
		{ 2872, 4320, 381 },
		{ 2114, 3724, 381 },
		{ 3054, 3304, 381 },
		{ 2114, 3668, 381 },
		{ 2099, 4106, 381 },
		{ 3106, 1959, 381 },
		{ 1799, 0, 381 },
		{ 2622, 2963, 379 },
		{ 3100, 2751, 381 },
		{ 3100, 2976, 381 },
		{ 0, 0, 381 },
		{ 3104, 2153, 0 },
		{ -1804, 19, 338 },
		{ -1805, 4748, 0 },
		{ 3058, 2311, 0 },
		{ 0, 0, 344 },
		{ 0, 0, 345 },
		{ 3097, 2446, 0 },
		{ 3004, 2627, 0 },
		{ 3089, 2900, 0 },
		{ 0, 0, 349 },
		{ 3058, 2317, 0 },
		{ 3106, 2065, 0 },
		{ 3004, 2671, 0 },
		{ 2067, 3093, 0 },
		{ 3050, 3647, 0 },
		{ 3057, 3336, 0 },
		{ 2007, 3268, 0 },
		{ 3050, 3650, 0 },
		{ 3094, 1625, 0 },
		{ 3067, 2012, 0 },
		{ 3058, 2238, 0 },
		{ 3102, 1801, 0 },
		{ 3106, 2082, 0 },
		{ 3104, 2170, 0 },
		{ 3011, 4724, 0 },
		{ 3104, 2171, 0 },
		{ 3067, 2020, 0 },
		{ 3102, 1842, 0 },
		{ 3058, 2267, 0 },
		{ 3086, 2912, 0 },
		{ 3106, 2091, 0 },
		{ 3097, 2420, 0 },
		{ 2135, 4002, 0 },
		{ 2067, 3109, 0 },
		{ 2067, 3110, 0 },
		{ 2099, 4051, 0 },
		{ 2029, 3864, 0 },
		{ 3089, 2819, 0 },
		{ 3067, 2025, 0 },
		{ 3086, 2909, 0 },
		{ 3094, 1626, 0 },
		{ 3089, 2825, 0 },
		{ 3097, 2428, 0 },
		{ 0, 16, 341 },
		{ 3091, 2359, 0 },
		{ 3089, 2834, 0 },
		{ 2114, 3714, 0 },
		{ 3102, 1865, 0 },
		{ 0, 0, 380 },
		{ 3089, 2839, 0 },
		{ 3086, 2914, 0 },
		{ 2099, 4065, 0 },
		{ 2045, 3448, 0 },
		{ 3050, 3638, 0 },
		{ 3028, 3510, 0 },
		{ 2067, 3124, 0 },
		{ 0, 0, 369 },
		{ 3062, 4303, 0 },
		{ 3104, 2191, 0 },
		{ 3106, 2100, 0 },
		{ 3058, 2288, 0 },
		{ -1881, 1167, 0 },
		{ 0, 0, 340 },
		{ 3089, 2853, 0 },
		{ 0, 0, 368 },
		{ 2864, 2119, 0 },
		{ 3004, 2663, 0 },
		{ 3058, 2291, 0 },
		{ 1896, 4687, 0 },
		{ 3027, 3746, 0 },
		{ 2978, 3924, 0 },
		{ 3028, 3527, 0 },
		{ 2067, 3140, 0 },
		{ 3050, 3651, 0 },
		{ 3104, 2200, 0 },
		{ 3091, 2386, 0 },
		{ 3058, 2298, 0 },
		{ 3102, 1868, 0 },
		{ 0, 0, 370 },
		{ 3062, 4263, 347 },
		{ 3102, 1873, 0 },
		{ 3101, 2957, 0 },
		{ 3102, 1878, 0 },
		{ 0, 0, 373 },
		{ 0, 0, 374 },
		{ 1901, 0, -62 },
		{ 2080, 3222, 0 },
		{ 2114, 3684, 0 },
		{ 3050, 3592, 0 },
		{ 2099, 4094, 0 },
		{ 3004, 2694, 0 },
		{ 0, 0, 372 },
		{ 0, 0, 378 },
		{ 0, 4689, 0 },
		{ 3097, 2413, 0 },
		{ 3067, 2035, 0 },
		{ 3100, 3010, 0 },
		{ 2135, 3988, 0 },
		{ 3061, 4212, 0 },
		{ 2176, 4638, 363 },
		{ 2099, 4101, 0 },
		{ 2872, 4323, 0 },
		{ 3028, 3543, 0 },
		{ 3028, 3544, 0 },
		{ 3058, 2306, 0 },
		{ 0, 0, 375 },
		{ 0, 0, 376 },
		{ 3100, 3014, 0 },
		{ 2182, 4735, 0 },
		{ 3097, 2418, 0 },
		{ 3089, 2893, 0 },
		{ 0, 0, 353 },
		{ 1924, 0, -65 },
		{ 1926, 0, -68 },
		{ 2114, 3697, 0 },
		{ 3062, 4299, 0 },
		{ 0, 0, 371 },
		{ 3067, 1962, 0 },
		{ 0, 0, 346 },
		{ 2135, 3999, 0 },
		{ 3058, 2313, 0 },
		{ 3061, 4194, 0 },
		{ 2176, 4621, 364 },
		{ 3061, 4196, 0 },
		{ 2176, 4625, 365 },
		{ 2872, 4322, 0 },
		{ 1936, 0, -50 },
		{ 3067, 1963, 0 },
		{ 3089, 2901, 0 },
		{ 3089, 2902, 0 },
		{ 0, 0, 355 },
		{ 0, 0, 357 },
		{ 1941, 0, -56 },
		{ 3061, 4238, 0 },
		{ 2176, 4643, 367 },
		{ 0, 0, 343 },
		{ 3058, 2322, 0 },
		{ 3106, 2062, 0 },
		{ 3061, 4266, 0 },
		{ 2176, 4617, 366 },
		{ 0, 0, 361 },
		{ 3104, 2149, 0 },
		{ 3100, 2990, 0 },
		{ 0, 0, 359 },
		{ 3091, 2390, 0 },
		{ 3102, 1900, 0 },
		{ 3089, 2790, 0 },
		{ 3004, 2571, 0 },
		{ 0, 0, 377 },
		{ 3104, 2152, 0 },
		{ 3058, 2240, 0 },
		{ 1955, 0, -71 },
		{ 3061, 4284, 0 },
		{ 2176, 4642, 362 },
		{ 0, 0, 351 },
		{ 1799, 2818, 381 },
		{ 1962, 2499, 381 },
		{ -1960, 4935, 338 },
		{ -1961, 4745, 0 },
		{ 3061, 4732, 0 },
		{ 3011, 4700, 0 },
		{ 0, 0, 339 },
		{ 3011, 4726, 0 },
		{ -1966, 22, 0 },
		{ -1967, 4740, 0 },
		{ 1970, 2, 341 },
		{ 3011, 4727, 0 },
		{ 3061, 4900, 0 },
		{ 0, 0, 342 },
		{ 1988, 0, 1 },
		{ 2184, 2838, 337 },
		{ 3089, 2814, 337 },
		{ 1988, 0, 291 },
		{ 1988, 2732, 337 },
		{ 3050, 3653, 337 },
		{ 1988, 0, 294 },
		{ 3094, 1594, 337 },
		{ 2872, 4327, 337 },
		{ 2114, 3665, 337 },
		{ 3054, 3261, 337 },
		{ 2114, 3667, 337 },
		{ 2099, 4049, 337 },
		{ 3100, 2978, 337 },
		{ 3106, 1965, 337 },
		{ 1988, 0, 337 },
		{ 2622, 2961, 334 },
		{ 3100, 2984, 337 },
		{ 3086, 2925, 337 },
		{ 2872, 4316, 337 },
		{ 3100, 1546, 337 },
		{ 0, 0, 337 },
		{ 3104, 2164, 0 },
		{ -1995, 21, 286 },
		{ -1996, 4741, 0 },
		{ 3058, 2264, 0 },
		{ 0, 0, 292 },
		{ 3058, 2265, 0 },
		{ 3104, 2166, 0 },
		{ 3106, 2081, 0 },
		{ 2067, 3087, 0 },
		{ 3050, 3624, 0 },
		{ 3057, 3350, 0 },
		{ 3027, 3762, 0 },
		{ 0, 3239, 0 },
		{ 0, 3278, 0 },
		{ 3050, 3628, 0 },
		{ 3097, 2415, 0 },
		{ 3094, 1603, 0 },
		{ 3067, 1983, 0 },
		{ 3058, 2275, 0 },
		{ 3089, 2844, 0 },
		{ 3089, 2845, 0 },
		{ 3104, 2173, 0 },
		{ 2182, 4734, 0 },
		{ 3104, 2176, 0 },
		{ 3011, 4715, 0 },
		{ 3104, 2177, 0 },
		{ 3086, 2923, 0 },
		{ 2864, 2126, 0 },
		{ 3106, 2086, 0 },
		{ 2135, 3996, 0 },
		{ 2067, 3100, 0 },
		{ 2067, 3101, 0 },
		{ 2978, 3939, 0 },
		{ 2978, 3941, 0 },
		{ 2099, 4086, 0 },
		{ 0, 3865, 0 },
		{ 3067, 1987, 0 },
		{ 3089, 2860, 0 },
		{ 3067, 1988, 0 },
		{ 3086, 2911, 0 },
		{ 3058, 2293, 0 },
		{ 3067, 1989, 0 },
		{ 3097, 2457, 0 },
		{ 0, 0, 336 },
		{ 3097, 2459, 0 },
		{ 0, 0, 288 },
		{ 3091, 2389, 0 },
		{ 0, 0, 333 },
		{ 3094, 1620, 0 },
		{ 3089, 2882, 0 },
		{ 2099, 4099, 0 },
		{ 0, 3420, 0 },
		{ 3050, 3657, 0 },
		{ 2048, 3805, 0 },
		{ 0, 3806, 0 },
		{ 3028, 3548, 0 },
		{ 2067, 3113, 0 },
		{ 3089, 2886, 0 },
		{ 0, 0, 326 },
		{ 3062, 4307, 0 },
		{ 3104, 2189, 0 },
		{ 3102, 1931, 0 },
		{ 3102, 1932, 0 },
		{ 3094, 1622, 0 },
		{ -2075, 1242, 0 },
		{ 3089, 2895, 0 },
		{ 3097, 2471, 0 },
		{ 3058, 2307, 0 },
		{ 3027, 3747, 0 },
		{ 2978, 3971, 0 },
		{ 3028, 3559, 0 },
		{ 2978, 3885, 0 },
		{ 2978, 3893, 0 },
		{ 0, 3123, 0 },
		{ 3050, 3622, 0 },
		{ 0, 0, 325 },
		{ 3104, 2198, 0 },
		{ 3091, 2375, 0 },
		{ 3004, 2677, 0 },
		{ 0, 0, 332 },
		{ 3100, 3037, 0 },
		{ 0, 0, 327 },
		{ 0, 0, 290 },
		{ 3100, 3038, 0 },
		{ 3102, 1938, 0 },
		{ 2092, 0, -47 },
		{ 0, 3219, 0 },
		{ 2114, 3676, 0 },
		{ 2083, 3209, 0 },
		{ 2080, 3210, 0 },
		{ 3050, 3632, 0 },
		{ 2099, 4036, 0 },
		{ 3004, 2688, 0 },
		{ 0, 0, 329 },
		{ 3101, 2946, 0 },
		{ 3102, 1945, 0 },
		{ 3102, 1682, 0 },
		{ 2135, 3982, 0 },
		{ 3061, 4268, 0 },
		{ 2176, 4640, 316 },
		{ 2099, 4042, 0 },
		{ 2872, 4328, 0 },
		{ 2099, 4043, 0 },
		{ 2099, 4044, 0 },
		{ 2099, 4045, 0 },
		{ 0, 4046, 0 },
		{ 3028, 3574, 0 },
		{ 3028, 3575, 0 },
		{ 3058, 2324, 0 },
		{ 3100, 2986, 0 },
		{ 3004, 2701, 0 },
		{ 3004, 2702, 0 },
		{ 3089, 2792, 0 },
		{ 0, 0, 298 },
		{ 2121, 0, -74 },
		{ 2123, 0, -77 },
		{ 2125, 0, -35 },
		{ 2127, 0, -38 },
		{ 2129, 0, -41 },
		{ 2131, 0, -44 },
		{ 0, 3691, 0 },
		{ 3062, 4314, 0 },
		{ 0, 0, 328 },
		{ 3097, 2422, 0 },
		{ 3104, 2206, 0 },
		{ 3104, 2207, 0 },
		{ 3058, 2331, 0 },
		{ 3061, 4200, 0 },
		{ 2176, 4623, 317 },
		{ 3061, 4202, 0 },
		{ 2176, 4626, 318 },
		{ 3061, 4204, 0 },
		{ 2176, 4629, 321 },
		{ 3061, 4206, 0 },
		{ 2176, 4633, 322 },
		{ 3061, 4208, 0 },
		{ 2176, 4635, 323 },
		{ 3061, 4210, 0 },
		{ 2176, 4637, 324 },
		{ 2872, 4321, 0 },
		{ 2146, 0, -53 },
		{ 0, 3995, 0 },
		{ 3058, 2332, 0 },
		{ 3058, 2334, 0 },
		{ 3089, 2811, 0 },
		{ 0, 0, 300 },
		{ 0, 0, 302 },
		{ 0, 0, 308 },
		{ 0, 0, 310 },
		{ 0, 0, 312 },
		{ 0, 0, 314 },
		{ 2152, 0, -59 },
		{ 3061, 4240, 0 },
		{ 2176, 4613, 320 },
		{ 3089, 2812, 0 },
		{ 3100, 3012, 0 },
		{ 3036, 3203, 331 },
		{ 3106, 2043, 0 },
		{ 3061, 4270, 0 },
		{ 2176, 4624, 319 },
		{ 0, 0, 306 },
		{ 3058, 2236, 0 },
		{ 3106, 2053, 0 },
		{ 0, 0, 293 },
		{ 3100, 3025, 0 },
		{ 0, 0, 304 },
		{ 3104, 2215, 0 },
		{ 2716, 1459, 0 },
		{ 3102, 1688, 0 },
		{ 3091, 2379, 0 },
		{ 2937, 4437, 0 },
		{ 3004, 2576, 0 },
		{ 3089, 2827, 0 },
		{ 3097, 2463, 0 },
		{ 3104, 2221, 0 },
		{ 0, 0, 330 },
		{ 2906, 2736, 0 },
		{ 3058, 2253, 0 },
		{ 3104, 2146, 0 },
		{ 2175, 0, -80 },
		{ 3106, 2058, 0 },
		{ 3061, 4192, 0 },
		{ 0, 4611, 315 },
		{ 3004, 2662, 0 },
		{ 0, 0, 296 },
		{ 3102, 1690, 0 },
		{ 3096, 2718, 0 },
		{ 3091, 2388, 0 },
		{ 0, 4731, 0 },
		{ 0, 0, 335 },
		{ 1988, 2817, 337 },
		{ 2188, 2501, 337 },
		{ -2186, 20, 286 },
		{ -2187, 1, 0 },
		{ 3061, 4731, 0 },
		{ 3011, 4714, 0 },
		{ 0, 0, 287 },
		{ 3011, 4728, 0 },
		{ -2192, 4936, 0 },
		{ -2193, 4743, 0 },
		{ 2196, 0, 288 },
		{ 3011, 4705, 0 },
		{ 3061, 4841, 0 },
		{ 0, 0, 289 },
		{ 0, 4161, 383 },
		{ 0, 0, 383 },
		{ 3089, 2856, 0 },
		{ 2950, 2764, 0 },
		{ 3100, 3002, 0 },
		{ 3094, 1629, 0 },
		{ 3097, 2408, 0 },
		{ 3102, 1755, 0 },
		{ 3061, 4908, 0 },
		{ 3106, 2069, 0 },
		{ 3094, 1641, 0 },
		{ 3058, 2272, 0 },
		{ 2211, 4815, 0 },
		{ 3061, 1945, 0 },
		{ 3100, 3018, 0 },
		{ 3106, 2078, 0 },
		{ 3100, 3029, 0 },
		{ 3091, 2371, 0 },
		{ 3089, 2878, 0 },
		{ 3102, 1815, 0 },
		{ 3089, 2883, 0 },
		{ 3106, 2080, 0 },
		{ 3067, 2021, 0 },
		{ 3107, 4537, 0 },
		{ 0, 0, 382 },
		{ 3011, 4704, 431 },
		{ 0, 0, 388 },
		{ 0, 0, 390 },
		{ 2243, 829, 422 },
		{ 2413, 842, 422 },
		{ 2435, 841, 422 },
		{ 2380, 842, 422 },
		{ 2244, 857, 422 },
		{ 2242, 834, 422 },
		{ 2435, 839, 422 },
		{ 2265, 854, 422 },
		{ 2409, 858, 422 },
		{ 2409, 860, 422 },
		{ 2413, 857, 422 },
		{ 2355, 866, 422 },
		{ 2241, 884, 422 },
		{ 3089, 1678, 421 },
		{ 2273, 2544, 431 },
		{ 2469, 856, 422 },
		{ 2413, 870, 422 },
		{ 2276, 868, 422 },
		{ 2413, 863, 422 },
		{ 3089, 2795, 431 },
		{ -2246, 4937, 384 },
		{ -2247, 4744, 0 },
		{ 2469, 860, 422 },
		{ 2474, 471, 422 },
		{ 2469, 863, 422 },
		{ 2323, 864, 422 },
		{ 2413, 872, 422 },
		{ 2419, 867, 422 },
		{ 2413, 874, 422 },
		{ 2355, 883, 422 },
		{ 2327, 873, 422 },
		{ 2380, 864, 422 },
		{ 2355, 886, 422 },
		{ 2435, 870, 422 },
		{ 2241, 863, 422 },
		{ 2382, 881, 422 },
		{ 2241, 878, 422 },
		{ 2448, 889, 422 },
		{ 2419, 889, 422 },
		{ 2241, 899, 422 },
		{ 2448, 892, 422 },
		{ 2469, 1265, 422 },
		{ 2448, 893, 422 },
		{ 2327, 923, 422 },
		{ 2474, 484, 422 },
		{ 3089, 1776, 418 },
		{ 2302, 1486, 0 },
		{ 3089, 1799, 419 },
		{ 3058, 2282, 0 },
		{ 3011, 4729, 0 },
		{ 2241, 935, 422 },
		{ 3104, 1951, 0 },
		{ 2409, 961, 422 },
		{ 2313, 946, 422 },
		{ 2448, 954, 422 },
		{ 2382, 959, 422 },
		{ 2382, 960, 422 },
		{ 2327, 969, 422 },
		{ 2409, 977, 422 },
		{ 2409, 978, 422 },
		{ 2435, 966, 422 },
		{ 2380, 963, 422 },
		{ 2409, 1007, 422 },
		{ 2355, 1012, 422 },
		{ 2474, 576, 422 },
		{ 2474, 578, 422 },
		{ 2442, 996, 422 },
		{ 2442, 998, 422 },
		{ 2409, 1049, 422 },
		{ 2313, 1034, 422 },
		{ 2419, 1041, 422 },
		{ 2355, 1056, 422 },
		{ 2398, 1054, 422 },
		{ 3099, 2447, 0 },
		{ 2331, 1406, 0 },
		{ 2302, 0, 0 },
		{ 3013, 2567, 420 },
		{ 2333, 1438, 0 },
		{ 3086, 2910, 0 },
		{ 0, 0, 386 },
		{ 2409, 1054, 422 },
		{ 2950, 2766, 0 },
		{ 2474, 580, 422 },
		{ 2327, 1075, 422 },
		{ 2382, 1068, 422 },
		{ 2474, 10, 422 },
		{ 2413, 1124, 422 },
		{ 2241, 1069, 422 },
		{ 2325, 1089, 422 },
		{ 2474, 123, 422 },
		{ 2382, 1110, 422 },
		{ 2413, 1122, 422 },
		{ 2474, 125, 422 },
		{ 2382, 1114, 422 },
		{ 3102, 1828, 0 },
		{ 3061, 2215, 0 },
		{ 2442, 1116, 422 },
		{ 2241, 1146, 422 },
		{ 2435, 1145, 422 },
		{ 2241, 1161, 422 },
		{ 2382, 1145, 422 },
		{ 2241, 1155, 422 },
		{ 2241, 1181, 422 },
		{ 3004, 2695, 0 },
		{ 2331, 0, 0 },
		{ 3013, 2601, 418 },
		{ 2333, 0, 0 },
		{ 3013, 2614, 419 },
		{ 0, 0, 423 },
		{ 2435, 1187, 422 },
		{ 2362, 4733, 0 },
		{ 3097, 2065, 0 },
		{ 2355, 1205, 422 },
		{ 2474, 130, 422 },
		{ 3067, 1886, 0 },
		{ 2497, 6, 422 },
		{ 2442, 1188, 422 },
		{ 2355, 1207, 422 },
		{ 2382, 1189, 422 },
		{ 3061, 1932, 0 },
		{ 2474, 235, 422 },
		{ 2380, 1215, 422 },
		{ 3104, 1990, 0 },
		{ 2413, 1229, 422 },
		{ 3058, 2316, 0 },
		{ 3106, 2052, 0 },
		{ 3058, 2318, 0 },
		{ 2419, 1224, 422 },
		{ 2435, 1222, 422 },
		{ 2241, 1241, 422 },
		{ 2409, 1265, 422 },
		{ 2409, 1266, 422 },
		{ 2474, 237, 422 },
		{ 2413, 1264, 422 },
		{ 3097, 2444, 0 },
		{ 2474, 239, 422 },
		{ 3099, 2171, 0 },
		{ 3004, 2686, 0 },
		{ 2382, 1254, 422 },
		{ 3067, 1892, 0 },
		{ 3102, 1694, 0 },
		{ 3107, 4397, 0 },
		{ 3061, 4797, 394 },
		{ 2469, 1262, 422 },
		{ 2382, 1256, 422 },
		{ 2413, 1272, 422 },
		{ 3104, 2167, 0 },
		{ 3099, 2457, 0 },
		{ 2413, 1270, 422 },
		{ 2950, 2769, 0 },
		{ 2419, 1265, 422 },
		{ 3004, 2711, 0 },
		{ 3089, 2788, 0 },
		{ 3004, 2712, 0 },
		{ 2241, 1260, 422 },
		{ 2413, 1274, 422 },
		{ 2241, 1264, 422 },
		{ 2474, 241, 422 },
		{ 2474, 243, 422 },
		{ 3106, 1913, 0 },
		{ 2448, 1272, 422 },
		{ 3089, 2804, 0 },
		{ 3104, 1961, 0 },
		{ 3050, 3662, 0 },
		{ 3004, 2569, 0 },
		{ 3091, 2363, 0 },
		{ 2413, 1278, 422 },
		{ 3102, 1905, 0 },
		{ 3100, 2970, 0 },
		{ 2497, 121, 422 },
		{ 2419, 1274, 422 },
		{ 2419, 1275, 422 },
		{ 2241, 1287, 422 },
		{ 2864, 2120, 0 },
		{ 3106, 2050, 0 },
		{ 2448, 1278, 422 },
		{ 2430, 4816, 0 },
		{ 2448, 1279, 422 },
		{ 3102, 1926, 0 },
		{ 3089, 2831, 0 },
		{ 3102, 1929, 0 },
		{ 2409, 1289, 422 },
		{ 2448, 1281, 422 },
		{ 2241, 1291, 422 },
		{ 3104, 1763, 0 },
		{ 3061, 2219, 0 },
		{ 3089, 2841, 0 },
		{ 2241, 1288, 422 },
		{ 3107, 4479, 0 },
		{ 2950, 2772, 0 },
		{ 3054, 3315, 0 },
		{ 3102, 1940, 0 },
		{ 3004, 2693, 0 },
		{ 2241, 1283, 422 },
		{ 3100, 3015, 0 },
		{ 3102, 1950, 0 },
		{ 3107, 4619, 0 },
		{ 0, 0, 412 },
		{ 2435, 1281, 422 },
		{ 2448, 1286, 422 },
		{ 2474, 245, 422 },
		{ 3094, 1658, 0 },
		{ 3104, 2156, 0 },
		{ 2436, 1295, 422 },
		{ 3061, 1937, 0 },
		{ 2474, 349, 422 },
		{ 2459, 4776, 0 },
		{ 2460, 4793, 0 },
		{ 2461, 4780, 0 },
		{ 2241, 1286, 422 },
		{ 2241, 1298, 422 },
		{ 2474, 351, 422 },
		{ 3100, 2977, 0 },
		{ 2950, 2753, 0 },
		{ 3067, 2013, 0 },
		{ 3086, 2908, 0 },
		{ 2241, 1288, 422 },
		{ 3061, 4926, 417 },
		{ 2470, 4694, 0 },
		{ 3067, 2015, 0 },
		{ 3058, 2246, 0 },
		{ 3102, 1822, 0 },
		{ 2241, 1294, 422 },
		{ 3102, 1847, 0 },
		{ 3067, 2023, 0 },
		{ 2474, 355, 422 },
		{ 2474, 357, 422 },
		{ 3061, 2332, 0 },
		{ 3097, 2455, 0 },
		{ 3091, 2377, 0 },
		{ 2474, 370, 422 },
		{ 3106, 2049, 0 },
		{ 3061, 1943, 0 },
		{ 3102, 1832, 0 },
		{ 3086, 2580, 0 },
		{ 3102, 1880, 0 },
		{ 2474, 463, 422 },
		{ 2474, 469, 422 },
		{ 3101, 1930, 0 },
		{ 3106, 2061, 0 },
		{ 2950, 2752, 0 },
		{ 3097, 2470, 0 },
		{ 3094, 1602, 0 },
		{ 2474, 1551, 422 },
		{ 3104, 1884, 0 },
		{ 2500, 4774, 0 },
		{ 3089, 2803, 0 },
		{ 3107, 4611, 0 },
		{ 2497, 574, 422 },
		{ 3067, 1969, 0 },
		{ 3107, 4395, 0 },
		{ 3061, 2334, 0 },
		{ 3104, 1982, 0 },
		{ 3089, 2809, 0 },
		{ 3100, 2979, 0 },
		{ 2510, 4792, 0 },
		{ 3104, 1761, 0 },
		{ 3104, 2210, 0 },
		{ 3106, 2071, 0 },
		{ 3106, 2074, 0 },
		{ 3089, 2815, 0 },
		{ 3106, 2076, 0 },
		{ 3061, 1941, 0 },
		{ 3067, 1888, 0 },
		{ 3067, 1980, 0 },
		{ 3058, 2312, 0 },
		{ 2522, 4681, 0 },
		{ 3089, 2823, 0 },
		{ 3067, 1981, 0 },
		{ 3100, 2999, 0 },
		{ 3101, 2942, 0 },
		{ 2527, 811, 422 },
		{ 3089, 2828, 0 },
		{ 2864, 2124, 0 },
		{ 3107, 4432, 0 },
		{ 3067, 1985, 0 },
		{ 3061, 4857, 392 },
		{ 3067, 1894, 0 },
		{ 3107, 4469, 0 },
		{ 3061, 4881, 401 },
		{ 3104, 2155, 0 },
		{ 2864, 2128, 0 },
		{ 3058, 2328, 0 },
		{ 3102, 1935, 0 },
		{ 3099, 2453, 0 },
		{ 3100, 3021, 0 },
		{ 2950, 2754, 0 },
		{ 2906, 2737, 0 },
		{ 3104, 2158, 0 },
		{ 3089, 2847, 0 },
		{ 2864, 2140, 0 },
		{ 3089, 2849, 0 },
		{ 3106, 2087, 0 },
		{ 3004, 2630, 0 },
		{ 3071, 1464, 0 },
		{ 3094, 1548, 0 },
		{ 3067, 1896, 0 },
		{ 3058, 2239, 0 },
		{ 2864, 2123, 0 },
		{ 3058, 2241, 0 },
		{ 3089, 2863, 0 },
		{ 3107, 4533, 0 },
		{ 3061, 4843, 415 },
		{ 3058, 2242, 0 },
		{ 3102, 1943, 0 },
		{ 0, 0, 428 },
		{ 3067, 1995, 0 },
		{ 3004, 2679, 0 },
		{ 3061, 4867, 400 },
		{ 3100, 2988, 0 },
		{ 3089, 2873, 0 },
		{ 3004, 2681, 0 },
		{ 3004, 2683, 0 },
		{ 3004, 2684, 0 },
		{ 3106, 2099, 0 },
		{ 2950, 2770, 0 },
		{ 2567, 4799, 0 },
		{ 2622, 2959, 0 },
		{ 3089, 2885, 0 },
		{ 3102, 1944, 0 },
		{ 3089, 2887, 0 },
		{ 3104, 2174, 0 },
		{ 2559, 1331, 0 },
		{ 2574, 4803, 0 },
		{ 2864, 2132, 0 },
		{ 3101, 2933, 0 },
		{ 3102, 1948, 0 },
		{ 3106, 2108, 0 },
		{ 3086, 2913, 0 },
		{ 2580, 4688, 0 },
		{ 3089, 2894, 0 },
		{ 3004, 2697, 0 },
		{ 2583, 4729, 0 },
		{ 0, 1378, 0 },
		{ 3097, 2436, 0 },
		{ 3106, 2040, 0 },
		{ 3102, 1681, 0 },
		{ 3104, 2187, 0 },
		{ 3097, 2448, 0 },
		{ 3089, 2904, 0 },
		{ 3067, 2003, 0 },
		{ 3061, 2753, 0 },
		{ 3100, 2975, 0 },
		{ 2594, 4772, 0 },
		{ 3096, 2717, 0 },
		{ 2596, 4786, 0 },
		{ 2622, 2964, 0 },
		{ 3089, 2787, 0 },
		{ 3067, 1898, 0 },
		{ 3097, 2453, 0 },
		{ 3106, 2047, 0 },
		{ 3067, 2005, 0 },
		{ 3004, 2539, 0 },
		{ 2604, 4793, 0 },
		{ 3104, 1970, 0 },
		{ 3106, 2051, 0 },
		{ 3091, 2384, 0 },
		{ 3101, 2704, 0 },
		{ 3089, 2800, 0 },
		{ 3107, 4617, 0 },
		{ 3100, 2992, 0 },
		{ 3104, 2197, 0 },
		{ 3058, 2284, 0 },
		{ 3089, 2805, 0 },
		{ 3058, 2285, 0 },
		{ 2864, 2131, 0 },
		{ 3094, 1619, 0 },
		{ 2622, 2965, 0 },
		{ 3086, 2582, 0 },
		{ 2620, 4766, 0 },
		{ 3086, 2575, 0 },
		{ 3100, 3007, 0 },
		{ 3107, 4506, 0 },
		{ 3102, 1689, 0 },
		{ 3104, 2202, 0 },
		{ 3004, 2659, 0 },
		{ 2627, 4737, 0 },
		{ 3058, 2292, 0 },
		{ 3091, 2008, 0 },
		{ 2864, 2118, 0 },
		{ 3100, 3017, 0 },
		{ 3004, 2668, 0 },
		{ 3100, 3020, 0 },
		{ 3107, 4615, 0 },
		{ 3061, 4874, 413 },
		{ 3102, 1691, 0 },
		{ 3106, 2056, 0 },
		{ 3107, 4623, 0 },
		{ 3107, 4393, 0 },
		{ 3102, 1692, 0 },
		{ 3106, 2060, 0 },
		{ 2950, 2767, 0 },
		{ 3004, 2674, 0 },
		{ 3089, 2829, 0 },
		{ 3107, 4407, 0 },
		{ 3089, 2830, 0 },
		{ 0, 2966, 0 },
		{ 3061, 4902, 399 },
		{ 3100, 2969, 0 },
		{ 3102, 1693, 0 },
		{ 2864, 2125, 0 },
		{ 3104, 1976, 0 },
		{ 2906, 2739, 0 },
		{ 3104, 2220, 0 },
		{ 3089, 2837, 0 },
		{ 3102, 1695, 0 },
		{ 3067, 2016, 0 },
		{ 3067, 2017, 0 },
		{ 3061, 4793, 393 },
		{ 3104, 2148, 0 },
		{ 3067, 2018, 0 },
		{ 3061, 4809, 405 },
		{ 3061, 4813, 406 },
		{ 3067, 2019, 0 },
		{ 3004, 2692, 0 },
		{ 2950, 2761, 0 },
		{ 3097, 2442, 0 },
		{ 2864, 2136, 0 },
		{ 0, 0, 427 },
		{ 2864, 2138, 0 },
		{ 3004, 2696, 0 },
		{ 3102, 1720, 0 },
		{ 2667, 4741, 0 },
		{ 3102, 1724, 0 },
		{ 2864, 2142, 0 },
		{ 2670, 4754, 0 },
		{ 3086, 2922, 0 },
		{ 3106, 2072, 0 },
		{ 3004, 2704, 0 },
		{ 3100, 3005, 0 },
		{ 3089, 2862, 0 },
		{ 3106, 2073, 0 },
		{ 3107, 4473, 0 },
		{ 3107, 4475, 0 },
		{ 3058, 2335, 0 },
		{ 3089, 2865, 0 },
		{ 3004, 2709, 0 },
		{ 3102, 1728, 0 },
		{ 3102, 1730, 0 },
		{ 3097, 2461, 0 },
		{ 3067, 2024, 0 },
		{ 3067, 1900, 0 },
		{ 3107, 4549, 0 },
		{ 3089, 2877, 0 },
		{ 3104, 1988, 0 },
		{ 3089, 2879, 0 },
		{ 3100, 3028, 0 },
		{ 3104, 2168, 0 },
		{ 3102, 1745, 0 },
		{ 3067, 2030, 0 },
		{ 3107, 4387, 0 },
		{ 3106, 939, 396 },
		{ 3061, 4928, 408 },
		{ 2906, 2740, 0 },
		{ 3106, 2083, 0 },
		{ 3102, 1775, 0 },
		{ 3096, 2725, 0 },
		{ 3096, 2728, 0 },
		{ 3004, 2568, 0 },
		{ 2702, 4715, 0 },
		{ 3101, 2949, 0 },
		{ 3061, 4811, 404 },
		{ 3106, 2085, 0 },
		{ 2864, 2133, 0 },
		{ 3097, 2475, 0 },
		{ 3102, 1783, 0 },
		{ 3058, 2261, 0 },
		{ 3004, 2625, 0 },
		{ 2710, 4779, 0 },
		{ 3061, 4829, 395 },
		{ 3107, 4504, 0 },
		{ 2712, 4782, 0 },
		{ 2716, 1457, 0 },
		{ 2714, 4784, 0 },
		{ 2715, 4786, 0 },
		{ 3102, 1798, 0 },
		{ 3099, 2445, 0 },
		{ 3106, 2090, 0 },
		{ 3100, 2989, 0 },
		{ 3089, 2782, 0 },
		{ 3107, 4545, 0 },
		{ 3104, 2183, 0 },
		{ 3067, 1956, 0 },
		{ 3104, 2186, 0 },
		{ 3107, 4576, 0 },
		{ 3061, 4871, 410 },
		{ 3107, 4603, 0 },
		{ 3107, 4605, 0 },
		{ 3107, 4607, 0 },
		{ 3107, 4609, 0 },
		{ 0, 1458, 0 },
		{ 3004, 2665, 0 },
		{ 3004, 2666, 0 },
		{ 3102, 1803, 0 },
		{ 3106, 2094, 0 },
		{ 3061, 4891, 416 },
		{ 3106, 2095, 0 },
		{ 3107, 4648, 0 },
		{ 3058, 2278, 0 },
		{ 0, 0, 430 },
		{ 0, 0, 429 },
		{ 3061, 4898, 397 },
		{ 0, 0, 425 },
		{ 0, 0, 426 },
		{ 3107, 4391, 0 },
		{ 3097, 2440, 0 },
		{ 2864, 2122, 0 },
		{ 3104, 2194, 0 },
		{ 3100, 3009, 0 },
		{ 3107, 4401, 0 },
		{ 3061, 4911, 391 },
		{ 2744, 4817, 0 },
		{ 3061, 4916, 398 },
		{ 3089, 2802, 0 },
		{ 3102, 1805, 0 },
		{ 3106, 2098, 0 },
		{ 3102, 1809, 0 },
		{ 3061, 4784, 411 },
		{ 3061, 2217, 0 },
		{ 3107, 4459, 0 },
		{ 3107, 4463, 0 },
		{ 3107, 4465, 0 },
		{ 3104, 2199, 0 },
		{ 3102, 1811, 0 },
		{ 3061, 4799, 402 },
		{ 3061, 4802, 403 },
		{ 3061, 4804, 407 },
		{ 3106, 2101, 0 },
		{ 3089, 2810, 0 },
		{ 3107, 4477, 0 },
		{ 3106, 2103, 0 },
		{ 3061, 4817, 409 },
		{ 3100, 3023, 0 },
		{ 3102, 1813, 0 },
		{ 3004, 2691, 0 },
		{ 3104, 2205, 0 },
		{ 3058, 2297, 0 },
		{ 3067, 1970, 0 },
		{ 3107, 4541, 0 },
		{ 3061, 4837, 414 },
		{ 3011, 4702, 431 },
		{ 2771, 0, 388 },
		{ 0, 0, 389 },
		{ -2769, 4931, 384 },
		{ -2770, 4746, 0 },
		{ 3061, 4723, 0 },
		{ 3011, 4711, 0 },
		{ 0, 0, 385 },
		{ 3011, 4721, 0 },
		{ -2775, 7, 0 },
		{ -2776, 4750, 0 },
		{ 2779, 0, 386 },
		{ 3011, 4722, 0 },
		{ 3061, 4853, 0 },
		{ 0, 0, 387 },
		{ 3054, 3310, 151 },
		{ 0, 0, 151 },
		{ 0, 0, 152 },
		{ 3067, 1974, 0 },
		{ 3089, 2820, 0 },
		{ 3106, 2038, 0 },
		{ 2788, 4776, 0 },
		{ 3099, 2455, 0 },
		{ 3094, 1657, 0 },
		{ 3058, 2305, 0 },
		{ 3101, 2956, 0 },
		{ 3102, 1827, 0 },
		{ 3004, 2705, 0 },
		{ 3104, 2217, 0 },
		{ 3058, 2309, 0 },
		{ 3067, 1977, 0 },
		{ 3107, 4621, 0 },
		{ 0, 0, 149 },
		{ 2937, 4517, 174 },
		{ 0, 0, 174 },
		{ 3102, 1833, 0 },
		{ 2803, 4803, 0 },
		{ 3089, 2501, 0 },
		{ 3100, 2985, 0 },
		{ 3101, 2945, 0 },
		{ 3096, 2714, 0 },
		{ 2808, 4802, 0 },
		{ 3061, 2328, 0 },
		{ 3089, 2836, 0 },
		{ 3058, 2315, 0 },
		{ 3089, 2838, 0 },
		{ 3106, 2045, 0 },
		{ 3100, 2994, 0 },
		{ 3102, 1840, 0 },
		{ 3004, 2537, 0 },
		{ 3104, 2223, 0 },
		{ 3058, 2320, 0 },
		{ 2819, 4831, 0 },
		{ 3061, 2751, 0 },
		{ 3089, 2846, 0 },
		{ 2950, 2765, 0 },
		{ 3104, 2145, 0 },
		{ 3106, 2048, 0 },
		{ 3089, 2850, 0 },
		{ 2826, 4686, 0 },
		{ 3106, 1925, 0 },
		{ 3089, 2852, 0 },
		{ 3086, 2915, 0 },
		{ 3094, 1660, 0 },
		{ 3101, 2934, 0 },
		{ 3089, 2854, 0 },
		{ 2833, 4713, 0 },
		{ 3099, 2441, 0 },
		{ 3094, 1536, 0 },
		{ 3058, 2330, 0 },
		{ 3101, 2943, 0 },
		{ 3102, 1862, 0 },
		{ 3004, 2622, 0 },
		{ 3104, 2151, 0 },
		{ 3058, 2333, 0 },
		{ 3107, 4578, 0 },
		{ 0, 0, 172 },
		{ 2844, 0, 1 },
		{ -2844, 1325, 263 },
		{ 3089, 2760, 269 },
		{ 0, 0, 269 },
		{ 3067, 1991, 0 },
		{ 3058, 2237, 0 },
		{ 3089, 2872, 0 },
		{ 3086, 2921, 0 },
		{ 3106, 2057, 0 },
		{ 0, 0, 268 },
		{ 2854, 4778, 0 },
		{ 3091, 1927, 0 },
		{ 3100, 2974, 0 },
		{ 2883, 2479, 0 },
		{ 3089, 2876, 0 },
		{ 2950, 2768, 0 },
		{ 3004, 2664, 0 },
		{ 3097, 2458, 0 },
		{ 3089, 2881, 0 },
		{ 2863, 4760, 0 },
		{ 3104, 1996, 0 },
		{ 0, 2134, 0 },
		{ 3102, 1894, 0 },
		{ 3004, 2670, 0 },
		{ 3104, 2163, 0 },
		{ 3058, 2244, 0 },
		{ 3067, 1996, 0 },
		{ 3107, 4461, 0 },
		{ 0, 0, 267 },
		{ 0, 4329, 177 },
		{ 0, 0, 177 },
		{ 3104, 2165, 0 },
		{ 3094, 1601, 0 },
		{ 3058, 2249, 0 },
		{ 3086, 2924, 0 },
		{ 2879, 4799, 0 },
		{ 3101, 2629, 0 },
		{ 3096, 2716, 0 },
		{ 3089, 2896, 0 },
		{ 3101, 2950, 0 },
		{ 0, 2486, 0 },
		{ 3004, 2682, 0 },
		{ 3058, 2251, 0 },
		{ 2906, 2732, 0 },
		{ 3107, 4539, 0 },
		{ 0, 0, 175 },
		{ 2937, 4531, 171 },
		{ 0, 0, 170 },
		{ 0, 0, 171 },
		{ 3102, 1903, 0 },
		{ 2894, 4801, 0 },
		{ 3102, 1856, 0 },
		{ 3096, 2726, 0 },
		{ 3089, 2905, 0 },
		{ 2898, 4826, 0 },
		{ 3061, 2749, 0 },
		{ 3089, 2781, 0 },
		{ 2906, 2738, 0 },
		{ 3004, 2687, 0 },
		{ 3058, 2257, 0 },
		{ 3058, 2258, 0 },
		{ 3004, 2690, 0 },
		{ 3058, 2259, 0 },
		{ 0, 2734, 0 },
		{ 2908, 4687, 0 },
		{ 3104, 1956, 0 },
		{ 2950, 2756, 0 },
		{ 2911, 4700, 0 },
		{ 3089, 2499, 0 },
		{ 3100, 3024, 0 },
		{ 3101, 2932, 0 },
		{ 3096, 2724, 0 },
		{ 2916, 4707, 0 },
		{ 3061, 2326, 0 },
		{ 3089, 2796, 0 },
		{ 3058, 2262, 0 },
		{ 3089, 2798, 0 },
		{ 3106, 2068, 0 },
		{ 3100, 3036, 0 },
		{ 3102, 1908, 0 },
		{ 3004, 2699, 0 },
		{ 3104, 2172, 0 },
		{ 3058, 2266, 0 },
		{ 2927, 4728, 0 },
		{ 3099, 2449, 0 },
		{ 3094, 1604, 0 },
		{ 3058, 2268, 0 },
		{ 3101, 2952, 0 },
		{ 3102, 1910, 0 },
		{ 3004, 2707, 0 },
		{ 3104, 2175, 0 },
		{ 3058, 2271, 0 },
		{ 3107, 4471, 0 },
		{ 0, 0, 164 },
		{ 0, 4395, 163 },
		{ 0, 0, 163 },
		{ 3102, 1914, 0 },
		{ 2941, 4738, 0 },
		{ 3102, 1860, 0 },
		{ 3096, 2715, 0 },
		{ 3089, 2816, 0 },
		{ 2945, 4757, 0 },
		{ 3089, 2505, 0 },
		{ 3058, 2274, 0 },
		{ 3086, 2917, 0 },
		{ 2949, 4753, 0 },
		{ 3104, 1958, 0 },
		{ 0, 2760, 0 },
		{ 2952, 4767, 0 },
		{ 3089, 2495, 0 },
		{ 3100, 2993, 0 },
		{ 3101, 2947, 0 },
		{ 3096, 2720, 0 },
		{ 2957, 4771, 0 },
		{ 3061, 2330, 0 },
		{ 3089, 2824, 0 },
		{ 3058, 2277, 0 },
		{ 3089, 2826, 0 },
		{ 3106, 2075, 0 },
		{ 3100, 3001, 0 },
		{ 3102, 1918, 0 },
		{ 3004, 2542, 0 },
		{ 3104, 2181, 0 },
		{ 3058, 2281, 0 },
		{ 2968, 4793, 0 },
		{ 3099, 2443, 0 },
		{ 3094, 1621, 0 },
		{ 3058, 2283, 0 },
		{ 3101, 2938, 0 },
		{ 3102, 1921, 0 },
		{ 3004, 2572, 0 },
		{ 3104, 2184, 0 },
		{ 3058, 2286, 0 },
		{ 3107, 4389, 0 },
		{ 0, 0, 161 },
		{ 0, 3955, 166 },
		{ 0, 0, 166 },
		{ 0, 0, 167 },
		{ 3058, 2287, 0 },
		{ 3067, 2011, 0 },
		{ 3102, 1922, 0 },
		{ 3089, 2842, 0 },
		{ 3100, 3022, 0 },
		{ 3086, 2926, 0 },
		{ 2988, 4818, 0 },
		{ 3089, 2493, 0 },
		{ 3071, 1495, 0 },
		{ 3100, 3027, 0 },
		{ 3097, 2473, 0 },
		{ 3094, 1623, 0 },
		{ 3100, 3030, 0 },
		{ 3102, 1927, 0 },
		{ 3004, 2660, 0 },
		{ 3104, 2190, 0 },
		{ 3058, 2294, 0 },
		{ 2999, 4693, 0 },
		{ 3099, 2451, 0 },
		{ 3094, 1624, 0 },
		{ 3058, 2296, 0 },
		{ 3101, 2940, 0 },
		{ 3102, 1930, 0 },
		{ 0, 2667, 0 },
		{ 3104, 2193, 0 },
		{ 3058, 2299, 0 },
		{ 3069, 4650, 0 },
		{ 0, 0, 165 },
		{ 3089, 2859, 431 },
		{ 3106, 1519, 25 },
		{ 0, 4713, 431 },
		{ 3020, 0, 431 },
		{ 2240, 2645, 28 },
		{ 0, 0, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 0, 0, 32 },
		{ 3058, 2303, 0 },
		{ -3019, 4932, 0 },
		{ 3106, 661, 0 },
		{ 0, 0, 27 },
		{ 3086, 2930, 0 },
		{ 0, 0, 26 },
		{ 0, 0, 21 },
		{ 0, 0, 33 },
		{ 0, 0, 34 },
		{ 0, 3781, 38 },
		{ 0, 3511, 38 },
		{ 0, 0, 37 },
		{ 0, 0, 38 },
		{ 3050, 3654, 0 },
		{ 3062, 4302, 0 },
		{ 3054, 3301, 0 },
		{ 0, 0, 36 },
		{ 3057, 3360, 0 },
		{ 0, 3205, 0 },
		{ 3013, 1632, 0 },
		{ 0, 0, 35 },
		{ 3089, 2775, 49 },
		{ 0, 0, 49 },
		{ 3054, 3306, 49 },
		{ 3089, 2870, 49 },
		{ 0, 0, 52 },
		{ 3089, 2871, 0 },
		{ 3058, 2308, 0 },
		{ 3057, 3369, 0 },
		{ 3102, 1942, 0 },
		{ 3058, 2310, 0 },
		{ 3086, 2919, 0 },
		{ 0, 3618, 0 },
		{ 3094, 1659, 0 },
		{ 3104, 2203, 0 },
		{ 0, 0, 48 },
		{ 0, 3317, 0 },
		{ 3106, 2096, 0 },
		{ 3091, 2367, 0 },
		{ 0, 3378, 0 },
		{ 0, 2314, 0 },
		{ 3089, 2880, 0 },
		{ 0, 0, 50 },
		{ 0, 5, 53 },
		{ 0, 4269, 0 },
		{ 0, 0, 51 },
		{ 3097, 2456, 0 },
		{ 3100, 3003, 0 },
		{ 3067, 2027, 0 },
		{ 0, 2028, 0 },
		{ 3069, 4654, 0 },
		{ 0, 4655, 0 },
		{ 3089, 2884, 0 },
		{ 0, 1533, 0 },
		{ 3100, 3008, 0 },
		{ 3097, 2460, 0 },
		{ 3094, 1675, 0 },
		{ 3100, 3011, 0 },
		{ 3102, 1947, 0 },
		{ 3104, 2212, 0 },
		{ 3106, 2102, 0 },
		{ 3100, 2056, 0 },
		{ 3089, 2892, 0 },
		{ 3104, 2216, 0 },
		{ 3101, 2939, 0 },
		{ 3100, 3019, 0 },
		{ 3106, 2104, 0 },
		{ 3101, 2941, 0 },
		{ 0, 2918, 0 },
		{ 3089, 2799, 0 },
		{ 3094, 1677, 0 },
		{ 0, 2897, 0 },
		{ 3100, 3026, 0 },
		{ 0, 2387, 0 },
		{ 3106, 2106, 0 },
		{ 3101, 2948, 0 },
		{ 0, 1679, 0 },
		{ 3107, 4657, 0 },
		{ 0, 2722, 0 },
		{ 0, 2472, 0 },
		{ 0, 0, 45 },
		{ 3061, 2725, 0 },
		{ 0, 3034, 0 },
		{ 0, 2953, 0 },
		{ 0, 1952, 0 },
		{ 3107, 4659, 0 },
		{ 0, 2222, 0 },
		{ 0, 0, 46 },
		{ 0, 2109, 0 },
		{ 3069, 4661, 0 },
		{ 0, 0, 47 }
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
