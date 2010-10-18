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
#line 1142 "EaseaLex.l"
fprintf(fpOutputFile,"%f",fMIGRATION_PROBABILITY);
#line 1776 "EaseaLex.cpp"
		}
		break;
	case 127:
		{
#line 1144 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPRINT_STATS);
#line 1783 "EaseaLex.cpp"
		}
		break;
	case 128:
		{
#line 1145 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bPLOT_STATS);
#line 1790 "EaseaLex.cpp"
		}
		break;
	case 129:
		{
#line 1146 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);
#line 1797 "EaseaLex.cpp"
		}
		break;
	case 130:
		{
#line 1147 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);
#line 1804 "EaseaLex.cpp"
		}
		break;
	case 131:
		{
#line 1148 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);
#line 1811 "EaseaLex.cpp"
		}
		break;
	case 132:
		{
#line 1150 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSAVE_POPULATION);
#line 1818 "EaseaLex.cpp"
		}
		break;
	case 133:
		{
#line 1151 "EaseaLex.l"
fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);
#line 1825 "EaseaLex.cpp"
		}
		break;
	case 134:
		{
#line 1153 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 
#line 1839 "EaseaLex.cpp"
		}
		break;
	case 135:
		{
#line 1161 "EaseaLex.l"

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
 
#line 1859 "EaseaLex.cpp"
		}
		break;
	case 136:
		{
#line 1175 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1873 "EaseaLex.cpp"
		}
		break;
	case 137:
		{
#line 1183 "EaseaLex.l"

  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1887 "EaseaLex.cpp"
		}
		break;
	case 138:
		{
#line 1192 "EaseaLex.l"

  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 
#line 1901 "EaseaLex.cpp"
		}
		break;
	case 139:
		{
#line 1201 "EaseaLex.l"

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

#line 1964 "EaseaLex.cpp"
		}
		break;
	case 140:
		{
#line 1258 "EaseaLex.l"

  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded without warnings.\n");
  printf ("You can now type \"make\" to compile your project.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 
#line 1981 "EaseaLex.cpp"
		}
		break;
	case 141:
		{
#line 1270 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 1988 "EaseaLex.cpp"
		}
		break;
	case 142:
		{
#line 1276 "EaseaLex.l"

  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2000 "EaseaLex.cpp"
		}
		break;
	case 143:
		{
#line 1282 "EaseaLex.l"

  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2013 "EaseaLex.cpp"
		}
		break;
	case 144:
		{
#line 1289 "EaseaLex.l"

#line 2020 "EaseaLex.cpp"
		}
		break;
	case 145:
		{
#line 1290 "EaseaLex.l"
lineCounter++;
#line 2027 "EaseaLex.cpp"
		}
		break;
	case 146:
		{
#line 1292 "EaseaLex.l"

  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2039 "EaseaLex.cpp"
		}
		break;
	case 147:
		{
#line 1298 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2052 "EaseaLex.cpp"
		}
		break;
	case 148:
		{
#line 1306 "EaseaLex.l"

#line 2059 "EaseaLex.cpp"
		}
		break;
	case 149:
		{
#line 1307 "EaseaLex.l"

  lineCounter++;
 
#line 2068 "EaseaLex.cpp"
		}
		break;
	case 150:
		{
#line 1311 "EaseaLex.l"

  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 
#line 2080 "EaseaLex.cpp"
		}
		break;
	case 151:
		{
#line 1317 "EaseaLex.l"

  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2094 "EaseaLex.cpp"
		}
		break;
	case 152:
		{
#line 1325 "EaseaLex.l"

#line 2101 "EaseaLex.cpp"
		}
		break;
	case 153:
		{
#line 1326 "EaseaLex.l"

  lineCounter++;
 
#line 2110 "EaseaLex.cpp"
		}
		break;
	case 154:
		{
#line 1330 "EaseaLex.l"

    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
      
  BEGIN COPY;
 
#line 2124 "EaseaLex.cpp"
		}
		break;
	case 155:
		{
#line 1338 "EaseaLex.l"
bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2139 "EaseaLex.cpp"
		}
		break;
	case 156:
		{
#line 1347 "EaseaLex.l"

#line 2146 "EaseaLex.cpp"
		}
		break;
	case 157:
		{
#line 1348 "EaseaLex.l"
lineCounter++;
#line 2153 "EaseaLex.cpp"
		}
		break;
	case 158:
		{
#line 1353 "EaseaLex.l"

  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 
#line 2167 "EaseaLex.cpp"
		}
		break;
	case 159:
		{
#line 1362 "EaseaLex.l"
bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2181 "EaseaLex.cpp"
		}
		break;
	case 160:
		{
#line 1370 "EaseaLex.l"

#line 2188 "EaseaLex.cpp"
		}
		break;
	case 161:
		{
#line 1371 "EaseaLex.l"
lineCounter++;
#line 2195 "EaseaLex.cpp"
		}
		break;
	case 162:
		{
#line 1374 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 
#line 2211 "EaseaLex.cpp"
		}
		break;
	case 163:
		{
#line 1385 "EaseaLex.l"

  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2227 "EaseaLex.cpp"
		}
		break;
	case 164:
		{
#line 1395 "EaseaLex.l"

#line 2234 "EaseaLex.cpp"
		}
		break;
	case 165:
		{
#line 1398 "EaseaLex.l"

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
 
#line 2252 "EaseaLex.cpp"
		}
		break;
	case 166:
		{
#line 1411 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 
#line 2269 "EaseaLex.cpp"
		}
		break;
	case 167:
		{
#line 1423 "EaseaLex.l"

#line 2276 "EaseaLex.cpp"
		}
		break;
	case 168:
		{
#line 1424 "EaseaLex.l"
lineCounter++;
#line 2283 "EaseaLex.cpp"
		}
		break;
	case 169:
		{
#line 1426 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2299 "EaseaLex.cpp"
		}
		break;
	case 170:
		{
#line 1438 "EaseaLex.l"

  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2315 "EaseaLex.cpp"
		}
		break;
	case 171:
		{
#line 1448 "EaseaLex.l"
lineCounter++;
#line 2322 "EaseaLex.cpp"
		}
		break;
	case 172:
		{
#line 1449 "EaseaLex.l"

#line 2329 "EaseaLex.cpp"
		}
		break;
	case 173:
		{
#line 1453 "EaseaLex.l"

  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 
#line 2344 "EaseaLex.cpp"
		}
		break;
	case 174:
		{
#line 1463 "EaseaLex.l"

  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2359 "EaseaLex.cpp"
		}
		break;
	case 175:
		{
#line 1472 "EaseaLex.l"

#line 2366 "EaseaLex.cpp"
		}
		break;
	case 176:
		{
#line 1475 "EaseaLex.l"

  if(TARGET==CUDA || TARGET==STD){
    fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 
#line 2379 "EaseaLex.cpp"
		}
		break;
	case 177:
		{
#line 1482 "EaseaLex.l"
bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 2393 "EaseaLex.cpp"
		}
		break;
	case 178:
		{
#line 1490 "EaseaLex.l"

#line 2400 "EaseaLex.cpp"
		}
		break;
	case 179:
		{
#line 1494 "EaseaLex.l"

  BEGIN GENOME_ANALYSIS; return CLASSES;
#line 2408 "EaseaLex.cpp"
		}
		break;
	case 180:
		{
#line 1496 "EaseaLex.l"

#line 2415 "EaseaLex.cpp"
		}
		break;
	case 181:
		{
#line 1502 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 2422 "EaseaLex.cpp"
		}
		break;
	case 182:
		{
#line 1503 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 2429 "EaseaLex.cpp"
		}
		break;
	case 183:
	case 184:
		{
#line 1506 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 2440 "EaseaLex.cpp"
		}
		break;
	case 185:
	case 186:
		{
#line 1511 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 2449 "EaseaLex.cpp"
		}
		break;
	case 187:
	case 188:
		{
#line 1514 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 2458 "EaseaLex.cpp"
		}
		break;
	case 189:
	case 190:
		{
#line 1517 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 
#line 2475 "EaseaLex.cpp"
		}
		break;
	case 191:
	case 192:
		{
#line 1528 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 
#line 2489 "EaseaLex.cpp"
		}
		break;
	case 193:
	case 194:
		{
#line 1536 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 2498 "EaseaLex.cpp"
		}
		break;
	case 195:
	case 196:
		{
#line 1539 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");
#line 2507 "EaseaLex.cpp"
		}
		break;
	case 197:
	case 198:
		{
#line 1542 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");
#line 2516 "EaseaLex.cpp"
		}
		break;
	case 199:
	case 200:
		{
#line 1545 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");
#line 2525 "EaseaLex.cpp"
		}
		break;
	case 201:
	case 202:
		{
#line 1548 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");
#line 2534 "EaseaLex.cpp"
		}
		break;
	case 203:
	case 204:
		{
#line 1552 "EaseaLex.l"

  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 
#line 2546 "EaseaLex.cpp"
		}
		break;
	case 205:
		{
#line 1558 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 2553 "EaseaLex.cpp"
		}
		break;
	case 206:
		{
#line 1559 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2560 "EaseaLex.cpp"
		}
		break;
	case 207:
		{
#line 1560 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2567 "EaseaLex.cpp"
		}
		break;
	case 208:
		{
#line 1561 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 2577 "EaseaLex.cpp"
		}
		break;
	case 209:
		{
#line 1566 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2584 "EaseaLex.cpp"
		}
		break;
	case 210:
		{
#line 1567 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
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
fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 2619 "EaseaLex.cpp"
		}
		break;
	case 215:
		{
#line 1572 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 2626 "EaseaLex.cpp"
		}
		break;
	case 216:
		{
#line 1573 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 2633 "EaseaLex.cpp"
		}
		break;
	case 217:
		{
#line 1574 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 2641 "EaseaLex.cpp"
		}
		break;
	case 218:
		{
#line 1576 "EaseaLex.l"
 // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 2649 "EaseaLex.cpp"
		}
		break;
	case 219:
		{
#line 1578 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 2657 "EaseaLex.cpp"
		}
		break;
	case 220:
		{
#line 1580 "EaseaLex.l"

  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");
#line 2667 "EaseaLex.cpp"
		}
		break;
	case 221:
		{
#line 1584 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 2674 "EaseaLex.cpp"
		}
		break;
	case 222:
		{
#line 1585 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;
#line 2681 "EaseaLex.cpp"
		}
		break;
	case 223:
		{
#line 1586 "EaseaLex.l"

  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);
#line 2692 "EaseaLex.cpp"
		}
		break;
	case 224:
		{
#line 1591 "EaseaLex.l"
fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);
#line 2699 "EaseaLex.cpp"
		}
		break;
	case 225:
		{
#line 1592 "EaseaLex.l"

  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);
#line 2708 "EaseaLex.cpp"
		}
		break;
	case 226:
		{
#line 1595 "EaseaLex.l"

  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 
#line 2720 "EaseaLex.cpp"
		}
		break;
	case 227:
		{
#line 1601 "EaseaLex.l"

  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}
#line 2729 "EaseaLex.cpp"
		}
		break;
	case 228:
		{
#line 1604 "EaseaLex.l"

  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 
#line 2741 "EaseaLex.cpp"
		}
		break;
	case 229:
		{
#line 1610 "EaseaLex.l"

  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  
#line 2752 "EaseaLex.cpp"
		}
		break;
	case 230:
		{
#line 1615 "EaseaLex.l"

  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;
#line 2768 "EaseaLex.cpp"
		}
		break;
	case 231:
		{
#line 1625 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 2775 "EaseaLex.cpp"
		}
		break;
	case 232:
		{
#line 1628 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; 
#line 2784 "EaseaLex.cpp"
		}
		break;
	case 233:
		{
#line 1631 "EaseaLex.l"
BEGIN COPY;
#line 2791 "EaseaLex.cpp"
		}
		break;
	case 234:
		{
#line 1633 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
#line 2798 "EaseaLex.cpp"
		}
		break;
	case 235:
	case 236:
	case 237:
		{
#line 1636 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2811 "EaseaLex.cpp"
		}
		break;
	case 238:
		{
#line 1641 "EaseaLex.l"
fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;
#line 2822 "EaseaLex.cpp"
		}
		break;
	case 239:
		{
#line 1646 "EaseaLex.l"
if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;
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
#line 1658 "EaseaLex.l"
;
#line 2859 "EaseaLex.cpp"
		}
		break;
	case 244:
		{
#line 1661 "EaseaLex.l"
 /* do nothing */ 
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
#line 1663 "EaseaLex.l"
 /*return '\n';*/ 
#line 2880 "EaseaLex.cpp"
		}
		break;
	case 247:
		{
#line 1666 "EaseaLex.l"

  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;
#line 2889 "EaseaLex.cpp"
		}
		break;
	case 248:
		{
#line 1669 "EaseaLex.l"

    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    
#line 2899 "EaseaLex.cpp"
		}
		break;
	case 249:
		{
#line 1673 "EaseaLex.l"

  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
//  printf("match gpnode\n");
  return GPNODE;
 
#line 2911 "EaseaLex.cpp"
		}
		break;
	case 250:
		{
#line 1680 "EaseaLex.l"
return STATIC;
#line 2918 "EaseaLex.cpp"
		}
		break;
	case 251:
		{
#line 1681 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("int"); return INT;
#line 2925 "EaseaLex.cpp"
		}
		break;
	case 252:
		{
#line 1682 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;
#line 2932 "EaseaLex.cpp"
		}
		break;
	case 253:
		{
#line 1683 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;
#line 2939 "EaseaLex.cpp"
		}
		break;
	case 254:
		{
#line 1684 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("char"); return CHAR;
#line 2946 "EaseaLex.cpp"
		}
		break;
	case 255:
		{
#line 1685 "EaseaLex.l"
yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;
#line 2953 "EaseaLex.cpp"
		}
		break;
	case 256:
		{
#line 1687 "EaseaLex.l"
rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;
#line 2960 "EaseaLex.cpp"
		}
		break;
#line 1688 "EaseaLex.l"
  
#line 2965 "EaseaLex.cpp"
	case 257:
		{
#line 1689 "EaseaLex.l"
return GENOME; 
#line 2970 "EaseaLex.cpp"
		}
		break;
	case 258:
		{
#line 1691 "EaseaLex.l"
BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;
#line 2980 "EaseaLex.cpp"
		}
		break;
	case 259:
	case 260:
	case 261:
		{
#line 1698 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER; 
#line 2989 "EaseaLex.cpp"
		}
		break;
	case 262:
		{
#line 1699 "EaseaLex.l"
yylval.dValue=atoi(yytext); return NUMBER;
#line 2996 "EaseaLex.cpp"
		}
		break;
	case 263:
		{
#line 1702 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; 
#line 3004 "EaseaLex.cpp"
		}
		break;
	case 264:
		{
#line 1704 "EaseaLex.l"
BEGIN GENOME_ANALYSIS; return END_METHODS;
#line 3011 "EaseaLex.cpp"
		}
		break;
	case 265:
		{
#line 1710 "EaseaLex.l"
 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 
#line 3023 "EaseaLex.cpp"
		}
		break;
	case 266:
		{
#line 1716 "EaseaLex.l"
bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 
#line 3036 "EaseaLex.cpp"
		}
		break;
	case 267:
		{
#line 1723 "EaseaLex.l"

#line 3043 "EaseaLex.cpp"
		}
		break;
	case 268:
		{
#line 1725 "EaseaLex.l"

  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 
#line 3054 "EaseaLex.cpp"
		}
		break;
	case 269:
		{
#line 1736 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 
#line 3069 "EaseaLex.cpp"
		}
		break;
	case 270:
		{
#line 1746 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 
#line 3080 "EaseaLex.cpp"
		}
		break;
	case 271:
		{
#line 1752 "EaseaLex.l"

  if( bWithinMAKEFILEOPTION );
 
#line 3089 "EaseaLex.cpp"
		}
		break;
	case 272:
		{
#line 1756 "EaseaLex.l"
 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 
#line 3104 "EaseaLex.cpp"
		}
		break;
	case 273:
		{
#line 1769 "EaseaLex.l"

  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 
#line 3116 "EaseaLex.cpp"
		}
		break;
	case 274:
		{
#line 1775 "EaseaLex.l"

#line 3123 "EaseaLex.cpp"
		}
		break;
	case 275:
		{
#line 1776 "EaseaLex.l"

  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 
#line 3136 "EaseaLex.cpp"
		}
		break;
	case 276:
		{
#line 1783 "EaseaLex.l"

#line 3143 "EaseaLex.cpp"
		}
		break;
	case 277:
		{
#line 1784 "EaseaLex.l"
lineCounter++;
#line 3150 "EaseaLex.cpp"
		}
		break;
	case 278:
		{
#line 1785 "EaseaLex.l"

  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 
#line 3163 "EaseaLex.cpp"
		}
		break;
	case 279:
		{
#line 1792 "EaseaLex.l"

#line 3170 "EaseaLex.cpp"
		}
		break;
	case 280:
		{
#line 1793 "EaseaLex.l"
lineCounter++;
#line 3177 "EaseaLex.cpp"
		}
		break;
	case 281:
		{
#line 1795 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 
#line 3190 "EaseaLex.cpp"
		}
		break;
	case 282:
		{
#line 1802 "EaseaLex.l"

#line 3197 "EaseaLex.cpp"
		}
		break;
	case 283:
		{
#line 1803 "EaseaLex.l"
lineCounter++;
#line 3204 "EaseaLex.cpp"
		}
		break;
	case 284:
		{
#line 1805 "EaseaLex.l"

  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 
#line 3217 "EaseaLex.cpp"
		}
		break;
	case 285:
		{
#line 1812 "EaseaLex.l"

#line 3224 "EaseaLex.cpp"
		}
		break;
	case 286:
		{
#line 1813 "EaseaLex.l"
lineCounter++;
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
fprintf(fpOutputFile,yytext);
#line 3259 "EaseaLex.cpp"
		}
		break;
	case 291:
		{
#line 1823 "EaseaLex.l"
fprintf(fpOutputFile,"genome");
#line 3266 "EaseaLex.cpp"
		}
		break;
	case 292:
		{
#line 1824 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3273 "EaseaLex.cpp"
		}
		break;
	case 293:
		{
#line 1825 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3280 "EaseaLex.cpp"
		}
		break;
	case 294:
		{
#line 1827 "EaseaLex.l"
bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 
#line 3289 "EaseaLex.cpp"
		}
		break;
	case 295:
		{
#line 1830 "EaseaLex.l"

  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 
#line 3302 "EaseaLex.cpp"
		}
		break;
	case 296:
	case 297:
		{
#line 1839 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}
#line 3313 "EaseaLex.cpp"
		}
		break;
	case 298:
	case 299:
		{
#line 1844 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }
#line 3322 "EaseaLex.cpp"
		}
		break;
	case 300:
	case 301:
		{
#line 1847 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    
#line 3331 "EaseaLex.cpp"
		}
		break;
	case 302:
	case 303:
		{
#line 1850 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 
#line 3343 "EaseaLex.cpp"
		}
		break;
	case 304:
	case 305:
		{
#line 1856 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 
#line 3356 "EaseaLex.cpp"
		}
		break;
	case 306:
	case 307:
		{
#line 1863 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    
#line 3365 "EaseaLex.cpp"
		}
		break;
	case 308:
	case 309:
		{
#line 1866 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    
#line 3374 "EaseaLex.cpp"
		}
		break;
	case 310:
	case 311:
		{
#line 1869 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    
#line 3383 "EaseaLex.cpp"
		}
		break;
	case 312:
	case 313:
		{
#line 1872 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    
#line 3392 "EaseaLex.cpp"
		}
		break;
	case 314:
	case 315:
		{
#line 1875 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    
#line 3401 "EaseaLex.cpp"
		}
		break;
	case 316:
		{
#line 1878 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 
#line 3410 "EaseaLex.cpp"
		}
		break;
	case 317:
		{
#line 1881 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }
#line 3420 "EaseaLex.cpp"
		}
		break;
	case 318:
		{
#line 1885 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3428 "EaseaLex.cpp"
		}
		break;
	case 319:
		{
#line 1887 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 
#line 3439 "EaseaLex.cpp"
		}
		break;
	case 320:
		{
#line 1892 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 
#line 3450 "EaseaLex.cpp"
		}
		break;
	case 321:
		{
#line 1897 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3458 "EaseaLex.cpp"
		}
		break;
	case 322:
		{
#line 1899 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3466 "EaseaLex.cpp"
		}
		break;
	case 323:
		{
#line 1901 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3474 "EaseaLex.cpp"
		}
		break;
	case 324:
		{
#line 1903 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3482 "EaseaLex.cpp"
		}
		break;
	case 325:
		{
#line 1905 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    
#line 3490 "EaseaLex.cpp"
		}
		break;
	case 326:
		{
#line 1907 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3497 "EaseaLex.cpp"
		}
		break;
	case 327:
		{
#line 1908 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3504 "EaseaLex.cpp"
		}
		break;
	case 328:
		{
#line 1909 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3512 "EaseaLex.cpp"
		}
		break;
	case 329:
		{
#line 1911 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3520 "EaseaLex.cpp"
		}
		break;
	case 330:
		{
#line 1913 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3528 "EaseaLex.cpp"
		}
		break;
	case 331:
		{
#line 1915 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3535 "EaseaLex.cpp"
		}
		break;
	case 332:
		{
#line 1916 "EaseaLex.l"

  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 
#line 3547 "EaseaLex.cpp"
		}
		break;
	case 333:
		{
#line 1922 "EaseaLex.l"

  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 
#line 3556 "EaseaLex.cpp"
		}
		break;
	case 334:
		{
#line 1925 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");
#line 3566 "EaseaLex.cpp"
		}
		break;
	case 335:
		{
#line 1929 "EaseaLex.l"
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
#line 3583 "EaseaLex.cpp"
		}
		break;
	case 336:
	case 337:
		{
#line 1941 "EaseaLex.l"

  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 
#line 3593 "EaseaLex.cpp"
		}
		break;
	case 338:
		{
#line 1944 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
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
fprintf(fpOutputFile,yytext);
#line 3614 "EaseaLex.cpp"
		}
		break;
	case 341:
		{
#line 1953 "EaseaLex.l"
fprintf(fpOutputFile,yytext);printf("%s\n",yytext);
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
#line 1955 "EaseaLex.l"
fprintf(fpOutputFile,yytext);
#line 3635 "EaseaLex.cpp"
		}
		break;
	case 344:
		{
#line 1957 "EaseaLex.l"

  fprintf(fpOutputFile,"Genome.");
 
#line 3644 "EaseaLex.cpp"
		}
		break;
	case 345:
		{
#line 1961 "EaseaLex.l"

  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3657 "EaseaLex.cpp"
		}
		break;
	case 346:
		{
#line 1969 "EaseaLex.l"

  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3670 "EaseaLex.cpp"
		}
		break;
	case 347:
		{
#line 1978 "EaseaLex.l"

  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 
#line 3683 "EaseaLex.cpp"
		}
		break;
	case 348:
		{
#line 1987 "EaseaLex.l"


  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");
#line 3698 "EaseaLex.cpp"
		}
		break;
	case 349:
		{
#line 1997 "EaseaLex.l"
(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");
#line 3705 "EaseaLex.cpp"
		}
		break;
	case 350:
		{
#line 1998 "EaseaLex.l"
fprintf(fpOutputFile,"\\\"");
#line 3712 "EaseaLex.cpp"
		}
		break;
	case 351:
	case 352:
		{
#line 2001 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }
#line 3723 "EaseaLex.cpp"
		}
		break;
	case 353:
	case 354:
		{
#line 2006 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");
#line 3732 "EaseaLex.cpp"
		}
		break;
	case 355:
	case 356:
		{
#line 2009 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");
#line 3741 "EaseaLex.cpp"
		}
		break;
	case 357:
	case 358:
		{
#line 2012 "EaseaLex.l"

  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 
#line 3754 "EaseaLex.cpp"
		}
		break;
	case 359:
	case 360:
		{
#line 2019 "EaseaLex.l"

  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
#line 3767 "EaseaLex.cpp"
		}
		break;
	case 361:
	case 362:
		{
#line 2026 "EaseaLex.l"
if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");
#line 3776 "EaseaLex.cpp"
		}
		break;
	case 363:
		{
#line 2029 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);
#line 3783 "EaseaLex.cpp"
		}
		break;
	case 364:
		{
#line 2030 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3790 "EaseaLex.cpp"
		}
		break;
	case 365:
		{
#line 2031 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3797 "EaseaLex.cpp"
		}
		break;
	case 366:
		{
#line 2032 "EaseaLex.l"

  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);

#line 3807 "EaseaLex.cpp"
		}
		break;
	case 367:
		{
#line 2037 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3814 "EaseaLex.cpp"
		}
		break;
	case 368:
		{
#line 2038 "EaseaLex.l"
fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);
#line 3821 "EaseaLex.cpp"
		}
		break;
	case 369:
		{
#line 2039 "EaseaLex.l"
fprintf(fpOutputFile,"false");
#line 3828 "EaseaLex.cpp"
		}
		break;
	case 370:
		{
#line 2040 "EaseaLex.l"
fprintf(fpOutputFile,"true");
#line 3835 "EaseaLex.cpp"
		}
		break;
	case 371:
		{
#line 2041 "EaseaLex.l"

  fprintf(fpOutputFile,yytext);
#line 3843 "EaseaLex.cpp"
		}
		break;
	case 372:
		{
#line 2043 "EaseaLex.l"
 // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");
#line 3851 "EaseaLex.cpp"
		}
		break;
	case 373:
		{
#line 2045 "EaseaLex.l"

  fprintf(fpOutputFile,"globalRandomGenerator->random");
#line 3859 "EaseaLex.cpp"
		}
		break;
	case 374:
		{
#line 2047 "EaseaLex.l"
fprintf(fpOutputFile,"child1");
 
#line 3867 "EaseaLex.cpp"
		}
		break;
	case 375:
		{
#line 2049 "EaseaLex.l"
fprintf(fpOutputFile,"child2");
 
#line 3875 "EaseaLex.cpp"
		}
		break;
	case 376:
		{
#line 2051 "EaseaLex.l"
fprintf(fpOutputFile,"parent1");
 
#line 3883 "EaseaLex.cpp"
		}
		break;
	case 377:
		{
#line 2053 "EaseaLex.l"
fprintf(fpOutputFile,"parent2");
 
#line 3891 "EaseaLex.cpp"
		}
		break;
	case 378:
		{
#line 2055 "EaseaLex.l"
fprintf(fpOutputFile,"genome._evaluated");
#line 3898 "EaseaLex.cpp"
		}
		break;
	case 379:
		{
#line 2056 "EaseaLex.l"
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
#line 3921 "EaseaLex.cpp"
		}
		break;
	case 380:
		{
#line 2073 "EaseaLex.l"
if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 
#line 3932 "EaseaLex.cpp"
		}
		break;
	case 381:
		{
#line 2078 "EaseaLex.l"
rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;
#line 3946 "EaseaLex.cpp"
		}
		break;
	case 382:
		{
#line 2086 "EaseaLex.l"
putc(yytext[0],fpOutputFile);
#line 3953 "EaseaLex.cpp"
		}
		break;
	case 383:
		{
#line 2092 "EaseaLex.l"

  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 
#line 3963 "EaseaLex.cpp"
		}
		break;
	case 384:
		{
#line 2096 "EaseaLex.l"

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
#line 2102 "EaseaLex.l"
;
#line 3998 "EaseaLex.cpp"
		}
		break;
	case 389:
		{
#line 2104 "EaseaLex.l"
 /* do nothing */ 
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
#line 2106 "EaseaLex.l"
 /*return '\n';*/ 
#line 4019 "EaseaLex.cpp"
		}
		break;
	case 392:
		{
#line 2108 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;
#line 4026 "EaseaLex.cpp"
		}
		break;
	case 393:
		{
#line 2109 "EaseaLex.l"
if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;
#line 4033 "EaseaLex.cpp"
		}
		break;
	case 394:
		{
#line 2110 "EaseaLex.l"
if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;
#line 4040 "EaseaLex.cpp"
		}
		break;
	case 395:
		{
#line 2111 "EaseaLex.l"
if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;
#line 4047 "EaseaLex.cpp"
		}
		break;
	case 396:
		{
#line 2112 "EaseaLex.l"
if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;
#line 4054 "EaseaLex.cpp"
		}
		break;
	case 397:
		{
#line 2113 "EaseaLex.l"
if (bVERBOSE) printf ("\tSel Genitors...\n");
#line 4061 "EaseaLex.cpp"
		}
		break;
	case 398:
		{
#line 2114 "EaseaLex.l"
if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;
#line 4068 "EaseaLex.cpp"
		}
		break;
	case 399:
		{
#line 2115 "EaseaLex.l"
if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;
#line 4075 "EaseaLex.cpp"
		}
		break;
	case 400:
		{
#line 2116 "EaseaLex.l"
if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;
#line 4082 "EaseaLex.cpp"
		}
		break;
	case 401:
		{
#line 2118 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;
#line 4089 "EaseaLex.cpp"
		}
		break;
	case 402:
		{
#line 2119 "EaseaLex.l"
if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;
#line 4096 "EaseaLex.cpp"
		}
		break;
	case 403:
		{
#line 2120 "EaseaLex.l"
if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;
#line 4103 "EaseaLex.cpp"
		}
		break;
	case 404:
		{
#line 2121 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;
#line 4110 "EaseaLex.cpp"
		}
		break;
	case 405:
		{
#line 2122 "EaseaLex.l"
if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;
#line 4117 "EaseaLex.cpp"
		}
		break;
	case 406:
		{
#line 2124 "EaseaLex.l"
if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;
#line 4124 "EaseaLex.cpp"
		}
		break;
	case 407:
		{
#line 2125 "EaseaLex.l"
if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;
#line 4131 "EaseaLex.cpp"
		}
		break;
	case 408:
		{
#line 2127 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 
#line 4142 "EaseaLex.cpp"
		}
		break;
	case 409:
		{
#line 2132 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;
#line 4149 "EaseaLex.cpp"
		}
		break;
	case 410:
		{
#line 2134 "EaseaLex.l"

  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 
#line 4160 "EaseaLex.cpp"
		}
		break;
	case 411:
		{
#line 2139 "EaseaLex.l"
if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;
#line 4167 "EaseaLex.cpp"
		}
		break;
	case 412:
		{
#line 2142 "EaseaLex.l"
if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;
#line 4174 "EaseaLex.cpp"
		}
		break;
	case 413:
		{
#line 2143 "EaseaLex.l"
if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;
#line 4181 "EaseaLex.cpp"
		}
		break;
	case 414:
		{
#line 2144 "EaseaLex.l"
if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;
#line 4188 "EaseaLex.cpp"
		}
		break;
	case 415:
		{
#line 2145 "EaseaLex.l"
if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;
#line 4195 "EaseaLex.cpp"
		}
		break;
	case 416:
		{
#line 2146 "EaseaLex.l"
if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;
#line 4202 "EaseaLex.cpp"
		}
		break;
#line 2147 "EaseaLex.l"
 
#line 4207 "EaseaLex.cpp"
	case 417:
		{
#line 2148 "EaseaLex.l"
if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;
#line 4212 "EaseaLex.cpp"
		}
		break;
	case 418:
		{
#line 2149 "EaseaLex.l"
if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;
#line 4219 "EaseaLex.cpp"
		}
		break;
	case 419:
		{
#line 2150 "EaseaLex.l"
if(bVERBOSE) printf("\tMigration Probability...\n"); return MIGRATION_PROBABILITY;
#line 4226 "EaseaLex.cpp"
		}
		break;
#line 2152 "EaseaLex.l"
 
#line 4231 "EaseaLex.cpp"
	case 420:
	case 421:
	case 422:
		{
#line 2156 "EaseaLex.l"
 yylval.dValue = myStrtod(); return NUMBER2; 
#line 4238 "EaseaLex.cpp"
		}
		break;
	case 423:
		{
#line 2157 "EaseaLex.l"
yylval.dValue=atof(yytext); return NUMBER2;
#line 4245 "EaseaLex.cpp"
		}
		break;
	case 424:
		{
#line 2160 "EaseaLex.l"
 yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; 
#line 4253 "EaseaLex.cpp"
		}
		break;
	case 425:
		{
#line 2163 "EaseaLex.l"
rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;
#line 4260 "EaseaLex.cpp"
		}
		break;
	case 426:
		{
#line 2165 "EaseaLex.l"

  lineCounter++;

#line 4269 "EaseaLex.cpp"
		}
		break;
	case 427:
		{
#line 2168 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 
#line 4279 "EaseaLex.cpp"
		}
		break;
	case 428:
		{
#line 2173 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 
#line 4289 "EaseaLex.cpp"
		}
		break;
	case 429:
		{
#line 2178 "EaseaLex.l"

  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 
#line 4299 "EaseaLex.cpp"
		}
		break;
	case 430:
		{
#line 2183 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 
#line 4309 "EaseaLex.cpp"
		}
		break;
	case 431:
		{
#line 2188 "EaseaLex.l"

  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 
#line 4319 "EaseaLex.cpp"
		}
		break;
	case 432:
		{
#line 2193 "EaseaLex.l"

  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 
#line 4329 "EaseaLex.cpp"
		}
		break;
	case 433:
		{
#line 2202 "EaseaLex.l"
return  (char)yytext[0];
#line 4336 "EaseaLex.cpp"
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
#line 2204 "EaseaLex.l"


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

#line 4533 "EaseaLex.cpp"

void YYLEXNAME::yytables()
{
	yystext_size = YYTEXT_SIZE;
	yysunput_size = YYUNPUT_SIZE;

	static const yymatch_t YYNEARFAR YYBASED_CODE match[] = {
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
		195,
		-196,
		0,
		197,
		-198,
		0,
		199,
		-200,
		0,
		201,
		-202,
		0,
		193,
		-194,
		0,
		191,
		-192,
		0,
		-233,
		0,
		-239,
		0,
		359,
		-360,
		0,
		304,
		-305,
		0,
		353,
		-354,
		0,
		355,
		-356,
		0,
		357,
		-358,
		0,
		351,
		-352,
		0,
		300,
		-301,
		0,
		302,
		-303,
		0,
		296,
		-297,
		0,
		308,
		-309,
		0,
		310,
		-311,
		0,
		312,
		-313,
		0,
		314,
		-315,
		0,
		298,
		-299,
		0,
		361,
		-362,
		0,
		306,
		-307,
		0
	};
	yymatch = match;

	yytransitionmax = 5013;
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
		{ 3054, 61 },
		{ 3054, 61 },
		{ 1884, 1987 },
		{ 1509, 1492 },
		{ 1510, 1492 },
		{ 2390, 2363 },
		{ 2390, 2363 },
		{ 2813, 2815 },
		{ 0, 2265 },
		{ 2362, 2332 },
		{ 2362, 2332 },
		{ 71, 3 },
		{ 72, 3 },
		{ 2243, 43 },
		{ 2244, 43 },
		{ 69, 1 },
		{ 165, 161 },
		{ 1884, 1865 },
		{ 67, 1 },
		{ 165, 167 },
		{ 0, 1823 },
		{ 2209, 2205 },
		{ 0, 2014 },
		{ 3054, 61 },
		{ 1357, 1356 },
		{ 3052, 61 },
		{ 1509, 1492 },
		{ 3103, 3101 },
		{ 2390, 2363 },
		{ 1378, 1377 },
		{ 1546, 1530 },
		{ 1547, 1530 },
		{ 2362, 2332 },
		{ 2210, 2206 },
		{ 71, 3 },
		{ 3056, 61 },
		{ 2243, 43 },
		{ 86, 61 },
		{ 3051, 61 },
		{ 3053, 61 },
		{ 3053, 61 },
		{ 3053, 61 },
		{ 3053, 61 },
		{ 3053, 61 },
		{ 3053, 61 },
		{ 3053, 61 },
		{ 3053, 61 },
		{ 3053, 61 },
		{ 3053, 61 },
		{ 70, 3 },
		{ 3055, 61 },
		{ 2242, 43 },
		{ 1599, 1593 },
		{ 1546, 1530 },
		{ 2391, 2363 },
		{ 1511, 1492 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 1548, 1530 },
		{ 3049, 61 },
		{ 1601, 1595 },
		{ 1472, 1451 },
		{ 3050, 61 },
		{ 1473, 1452 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3050, 61 },
		{ 3057, 61 },
		{ 2204, 40 },
		{ 1549, 1531 },
		{ 1550, 1531 },
		{ 1467, 1445 },
		{ 1991, 40 },
		{ 2447, 2419 },
		{ 2447, 2419 },
		{ 2367, 2336 },
		{ 2367, 2336 },
		{ 2370, 2339 },
		{ 2370, 2339 },
		{ 2006, 39 },
		{ 1468, 1446 },
		{ 1817, 37 },
		{ 2388, 2361 },
		{ 2388, 2361 },
		{ 1469, 1447 },
		{ 1471, 1450 },
		{ 1474, 1453 },
		{ 1475, 1454 },
		{ 1476, 1455 },
		{ 1477, 1456 },
		{ 1478, 1457 },
		{ 2204, 40 },
		{ 1549, 1531 },
		{ 1994, 40 },
		{ 1479, 1458 },
		{ 1480, 1459 },
		{ 2447, 2419 },
		{ 1481, 1460 },
		{ 2367, 2336 },
		{ 1482, 1462 },
		{ 2370, 2339 },
		{ 1485, 1465 },
		{ 2006, 39 },
		{ 1486, 1466 },
		{ 1817, 37 },
		{ 2388, 2361 },
		{ 2203, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 1992, 39 },
		{ 2007, 40 },
		{ 1804, 37 },
		{ 1487, 1467 },
		{ 1551, 1531 },
		{ 2448, 2419 },
		{ 1488, 1468 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 1993, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2001, 40 },
		{ 1999, 40 },
		{ 2012, 40 },
		{ 2000, 40 },
		{ 2012, 40 },
		{ 2003, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2002, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 1489, 1469 },
		{ 1995, 40 },
		{ 1997, 40 },
		{ 1491, 1471 },
		{ 2012, 40 },
		{ 1492, 1472 },
		{ 2012, 40 },
		{ 2010, 40 },
		{ 1998, 40 },
		{ 2012, 40 },
		{ 2011, 40 },
		{ 2004, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2009, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 1996, 40 },
		{ 2012, 40 },
		{ 2008, 40 },
		{ 2012, 40 },
		{ 2005, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 2012, 40 },
		{ 1393, 21 },
		{ 1552, 1532 },
		{ 1553, 1532 },
		{ 1493, 1473 },
		{ 1380, 21 },
		{ 2396, 2368 },
		{ 2396, 2368 },
		{ 2409, 2381 },
		{ 2409, 2381 },
		{ 2412, 2384 },
		{ 2412, 2384 },
		{ 2435, 2407 },
		{ 2435, 2407 },
		{ 2436, 2408 },
		{ 2436, 2408 },
		{ 2479, 2451 },
		{ 2479, 2451 },
		{ 1494, 1474 },
		{ 1495, 1475 },
		{ 1496, 1476 },
		{ 1497, 1477 },
		{ 1498, 1478 },
		{ 1499, 1479 },
		{ 1393, 21 },
		{ 1552, 1532 },
		{ 1381, 21 },
		{ 1394, 21 },
		{ 1500, 1480 },
		{ 2396, 2368 },
		{ 1501, 1482 },
		{ 2409, 2381 },
		{ 1504, 1485 },
		{ 2412, 2384 },
		{ 1505, 1486 },
		{ 2435, 2407 },
		{ 1506, 1487 },
		{ 2436, 2408 },
		{ 1507, 1489 },
		{ 2479, 2451 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1508, 1491 },
		{ 1405, 1383 },
		{ 1512, 1493 },
		{ 1513, 1494 },
		{ 1554, 1532 },
		{ 1518, 1497 },
		{ 1519, 1498 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1397, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1386, 21 },
		{ 1384, 21 },
		{ 1399, 21 },
		{ 1385, 21 },
		{ 1399, 21 },
		{ 1388, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1387, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1520, 1499 },
		{ 1382, 21 },
		{ 1395, 21 },
		{ 1521, 1500 },
		{ 1389, 21 },
		{ 1522, 1501 },
		{ 1399, 21 },
		{ 1400, 21 },
		{ 1383, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1390, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1398, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1401, 21 },
		{ 1399, 21 },
		{ 1396, 21 },
		{ 1399, 21 },
		{ 1391, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1399, 21 },
		{ 1978, 38 },
		{ 1555, 1533 },
		{ 1556, 1533 },
		{ 1514, 1495 },
		{ 1803, 38 },
		{ 2484, 2456 },
		{ 2484, 2456 },
		{ 2491, 2463 },
		{ 2491, 2463 },
		{ 1516, 1496 },
		{ 1515, 1495 },
		{ 2504, 2477 },
		{ 2504, 2477 },
		{ 2505, 2478 },
		{ 2505, 2478 },
		{ 1525, 1505 },
		{ 1517, 1496 },
		{ 1526, 1506 },
		{ 1527, 1507 },
		{ 1528, 1508 },
		{ 1530, 1512 },
		{ 1531, 1513 },
		{ 1532, 1514 },
		{ 1978, 38 },
		{ 1555, 1533 },
		{ 1808, 38 },
		{ 2509, 2482 },
		{ 2509, 2482 },
		{ 2484, 2456 },
		{ 1533, 1515 },
		{ 2491, 2463 },
		{ 1534, 1516 },
		{ 1558, 1534 },
		{ 1559, 1534 },
		{ 2504, 2477 },
		{ 1535, 1517 },
		{ 2505, 2478 },
		{ 1536, 1518 },
		{ 1977, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 2509, 2482 },
		{ 1818, 38 },
		{ 1537, 1519 },
		{ 1538, 1520 },
		{ 1557, 1533 },
		{ 1539, 1521 },
		{ 1558, 1534 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1805, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1813, 38 },
		{ 1811, 38 },
		{ 1821, 38 },
		{ 1812, 38 },
		{ 1821, 38 },
		{ 1815, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1814, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1540, 1522 },
		{ 1809, 38 },
		{ 1560, 1534 },
		{ 1542, 1525 },
		{ 1821, 38 },
		{ 1543, 1526 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1810, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1806, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1807, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1820, 38 },
		{ 1821, 38 },
		{ 1819, 38 },
		{ 1821, 38 },
		{ 1816, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 1821, 38 },
		{ 2807, 44 },
		{ 2808, 44 },
		{ 1561, 1535 },
		{ 1562, 1535 },
		{ 67, 44 },
		{ 2512, 2485 },
		{ 2512, 2485 },
		{ 1544, 1527 },
		{ 1545, 1528 },
		{ 1408, 1384 },
		{ 1409, 1385 },
		{ 2516, 2489 },
		{ 2516, 2489 },
		{ 2517, 2490 },
		{ 2517, 2490 },
		{ 1413, 1387 },
		{ 1414, 1388 },
		{ 1415, 1389 },
		{ 1564, 1536 },
		{ 1565, 1537 },
		{ 1566, 1538 },
		{ 1568, 1542 },
		{ 1569, 1543 },
		{ 2807, 44 },
		{ 1570, 1544 },
		{ 1561, 1535 },
		{ 2297, 2268 },
		{ 2297, 2268 },
		{ 2512, 2485 },
		{ 1571, 1545 },
		{ 1579, 1565 },
		{ 1580, 1565 },
		{ 1588, 1578 },
		{ 1589, 1578 },
		{ 2516, 2489 },
		{ 1578, 1564 },
		{ 2517, 2490 },
		{ 2259, 44 },
		{ 2806, 44 },
		{ 2258, 44 },
		{ 2258, 44 },
		{ 2258, 44 },
		{ 2258, 44 },
		{ 2258, 44 },
		{ 2258, 44 },
		{ 2258, 44 },
		{ 2258, 44 },
		{ 2258, 44 },
		{ 2258, 44 },
		{ 2297, 2268 },
		{ 1416, 1390 },
		{ 1582, 1566 },
		{ 1584, 1568 },
		{ 1579, 1565 },
		{ 1563, 1535 },
		{ 1588, 1578 },
		{ 2260, 44 },
		{ 2256, 44 },
		{ 2251, 44 },
		{ 2260, 44 },
		{ 2248, 44 },
		{ 2255, 44 },
		{ 2253, 44 },
		{ 2260, 44 },
		{ 2257, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2250, 44 },
		{ 2245, 44 },
		{ 2252, 44 },
		{ 2247, 44 },
		{ 2260, 44 },
		{ 2254, 44 },
		{ 2249, 44 },
		{ 2246, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 1581, 1565 },
		{ 2264, 44 },
		{ 1590, 1578 },
		{ 1585, 1569 },
		{ 2260, 44 },
		{ 1586, 1570 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2261, 44 },
		{ 2262, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2263, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 2260, 44 },
		{ 159, 4 },
		{ 160, 4 },
		{ 2531, 2501 },
		{ 2531, 2501 },
		{ 2320, 2289 },
		{ 2320, 2289 },
		{ 2342, 2311 },
		{ 2342, 2311 },
		{ 2343, 2312 },
		{ 2343, 2312 },
		{ 2359, 2329 },
		{ 2359, 2329 },
		{ 1412, 1386 },
		{ 1587, 1571 },
		{ 1418, 1391 },
		{ 1593, 1584 },
		{ 1594, 1585 },
		{ 1417, 1391 },
		{ 1595, 1586 },
		{ 1596, 1587 },
		{ 1411, 1386 },
		{ 1421, 1396 },
		{ 1600, 1594 },
		{ 159, 4 },
		{ 1422, 1397 },
		{ 2531, 2501 },
		{ 1602, 1596 },
		{ 2320, 2289 },
		{ 1605, 1600 },
		{ 2342, 2311 },
		{ 1606, 1602 },
		{ 2343, 2312 },
		{ 1410, 1386 },
		{ 2359, 2329 },
		{ 1608, 1605 },
		{ 1609, 1606 },
		{ 1610, 1608 },
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
		{ 1611, 1609 },
		{ 1406, 1610 },
		{ 0, 2501 },
		{ 1423, 1398 },
		{ 1424, 1400 },
		{ 1425, 1401 },
		{ 1428, 1405 },
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
		{ 1429, 1408 },
		{ 81, 4 },
		{ 1430, 1409 },
		{ 1431, 1410 },
		{ 85, 4 },
		{ 1432, 1411 },
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
		{ 3061, 3060 },
		{ 1433, 1412 },
		{ 1434, 1413 },
		{ 3060, 3060 },
		{ 1435, 1414 },
		{ 1438, 1416 },
		{ 1436, 1415 },
		{ 1439, 1417 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 1437, 1415 },
		{ 3060, 3060 },
		{ 1440, 1418 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 1443, 1421 },
		{ 1444, 1422 },
		{ 1445, 1423 },
		{ 1446, 1424 },
		{ 1447, 1425 },
		{ 1450, 1428 },
		{ 1451, 1429 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 1452, 1430 },
		{ 1453, 1431 },
		{ 1454, 1432 },
		{ 1455, 1433 },
		{ 1456, 1434 },
		{ 1457, 1435 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 3060, 3060 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1458, 1436 },
		{ 1459, 1437 },
		{ 1460, 1438 },
		{ 1461, 1439 },
		{ 1462, 1440 },
		{ 1465, 1443 },
		{ 1466, 1444 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 154, 152 },
		{ 105, 90 },
		{ 106, 91 },
		{ 107, 92 },
		{ 1406, 1612 },
		{ 108, 93 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 1406, 1612 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 109, 94 },
		{ 110, 95 },
		{ 111, 96 },
		{ 112, 97 },
		{ 114, 99 },
		{ 120, 104 },
		{ 121, 105 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 122, 106 },
		{ 123, 107 },
		{ 124, 109 },
		{ 125, 110 },
		{ 2260, 2525 },
		{ 126, 111 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 2260, 2525 },
		{ 1407, 1611 },
		{ 0, 1611 },
		{ 127, 112 },
		{ 129, 114 },
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
		{ 2712, 2712 },
		{ 144, 136 },
		{ 2267, 2245 },
		{ 2269, 2246 },
		{ 2272, 2247 },
		{ 2273, 2248 },
		{ 2281, 2250 },
		{ 2270, 2247 },
		{ 2276, 2249 },
		{ 1407, 1611 },
		{ 2271, 2247 },
		{ 2283, 2251 },
		{ 2275, 2249 },
		{ 2284, 2252 },
		{ 2285, 2253 },
		{ 2274, 2248 },
		{ 2286, 2254 },
		{ 2287, 2255 },
		{ 2280, 2250 },
		{ 2288, 2256 },
		{ 2289, 2257 },
		{ 2260, 2260 },
		{ 2282, 2261 },
		{ 2712, 2712 },
		{ 2268, 2262 },
		{ 2279, 2263 },
		{ 2296, 2267 },
		{ 2277, 2249 },
		{ 2278, 2249 },
		{ 145, 137 },
		{ 2293, 2261 },
		{ 2298, 2269 },
		{ 2299, 2270 },
		{ 2300, 2271 },
		{ 2301, 2272 },
		{ 2302, 2273 },
		{ 2303, 2274 },
		{ 2304, 2275 },
		{ 0, 1611 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2305, 2276 },
		{ 2306, 2277 },
		{ 2307, 2278 },
		{ 2308, 2279 },
		{ 2309, 2280 },
		{ 2310, 2281 },
		{ 2312, 2282 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 67, 7 },
		{ 2313, 2283 },
		{ 2314, 2284 },
		{ 2315, 2285 },
		{ 2712, 2712 },
		{ 1612, 1611 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2712, 2712 },
		{ 2318, 2287 },
		{ 2319, 2288 },
		{ 146, 138 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 2311, 2293 },
		{ 2327, 2296 },
		{ 2329, 2298 },
		{ 2330, 2299 },
		{ 2331, 2300 },
		{ 2332, 2301 },
		{ 2333, 2302 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 2334, 2303 },
		{ 2335, 2304 },
		{ 2336, 2305 },
		{ 2337, 2306 },
		{ 1233, 7 },
		{ 2338, 2307 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 1233, 7 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 2339, 2308 },
		{ 2340, 2309 },
		{ 2341, 2310 },
		{ 147, 140 },
		{ 2344, 2313 },
		{ 2345, 2314 },
		{ 2346, 2315 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 2347, 2316 },
		{ 2348, 2317 },
		{ 2349, 2318 },
		{ 2350, 2319 },
		{ 0, 1883 },
		{ 2357, 2327 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 1883 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 148, 141 },
		{ 2360, 2330 },
		{ 2361, 2331 },
		{ 149, 142 },
		{ 2365, 2334 },
		{ 2366, 2335 },
		{ 2368, 2337 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 2369, 2338 },
		{ 2363, 2333 },
		{ 150, 144 },
		{ 2371, 2340 },
		{ 0, 2077 },
		{ 2364, 2333 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 0, 2077 },
		{ 2316, 2286 },
		{ 2372, 2341 },
		{ 2376, 2344 },
		{ 2377, 2345 },
		{ 2378, 2346 },
		{ 2379, 2347 },
		{ 2380, 2348 },
		{ 2381, 2349 },
		{ 2382, 2350 },
		{ 2317, 2286 },
		{ 2384, 2357 },
		{ 2387, 2360 },
		{ 2392, 2364 },
		{ 2393, 2365 },
		{ 2394, 2366 },
		{ 151, 147 },
		{ 2397, 2369 },
		{ 2399, 2371 },
		{ 2400, 2372 },
		{ 2404, 2376 },
		{ 2405, 2377 },
		{ 2406, 2378 },
		{ 2407, 2379 },
		{ 2408, 2380 },
		{ 152, 148 },
		{ 2410, 2382 },
		{ 2416, 2387 },
		{ 2419, 2392 },
		{ 2420, 2393 },
		{ 2422, 2394 },
		{ 2425, 2397 },
		{ 2427, 2399 },
		{ 2428, 2400 },
		{ 2421, 2394 },
		{ 2432, 2404 },
		{ 2433, 2405 },
		{ 2434, 2406 },
		{ 153, 150 },
		{ 2438, 2410 },
		{ 2444, 2416 },
		{ 89, 73 },
		{ 2449, 2420 },
		{ 2450, 2421 },
		{ 2451, 2422 },
		{ 2454, 2425 },
		{ 2456, 2427 },
		{ 2457, 2428 },
		{ 2461, 2432 },
		{ 2462, 2433 },
		{ 2463, 2434 },
		{ 2468, 2438 },
		{ 2474, 2444 },
		{ 2477, 2449 },
		{ 2478, 2450 },
		{ 155, 153 },
		{ 2482, 2454 },
		{ 156, 155 },
		{ 2485, 2457 },
		{ 2489, 2461 },
		{ 2490, 2462 },
		{ 157, 156 },
		{ 2496, 2468 },
		{ 2501, 2474 },
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
		{ 132, 118 },
		{ 0, 2884 },
		{ 132, 118 },
		{ 85, 157 },
		{ 2604, 2578 },
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
		{ 116, 101 },
		{ 1243, 1240 },
		{ 116, 101 },
		{ 1243, 1240 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 2352, 2321 },
		{ 2354, 2324 },
		{ 2352, 2321 },
		{ 2354, 2324 },
		{ 2883, 49 },
		{ 2615, 2589 },
		{ 91, 74 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 130, 115 },
		{ 1246, 1242 },
		{ 130, 115 },
		{ 1246, 1242 },
		{ 1233, 1233 },
		{ 2769, 2754 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 1233, 1233 },
		{ 0, 2496 },
		{ 0, 2496 },
		{ 2322, 2291 },
		{ 1248, 1245 },
		{ 2322, 2291 },
		{ 1248, 1245 },
		{ 2183, 2180 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 1801, 1800 },
		{ 1285, 1284 },
		{ 1759, 1758 },
		{ 2750, 2734 },
		{ 2766, 2751 },
		{ 2579, 2549 },
		{ 0, 2496 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 1711, 1710 },
		{ 86, 49 },
		{ 1756, 1755 },
		{ 1282, 1281 },
		{ 3050, 3050 },
		{ 3030, 3029 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 3050, 3050 },
		{ 1834, 1810 },
		{ 1666, 1665 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 3112, 3111 },
		{ 1833, 1810 },
		{ 2875, 2874 },
		{ 2580, 2550 },
		{ 2223, 2222 },
		{ 2228, 2227 },
		{ 2525, 2496 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 2524, 2495 },
		{ 1714, 1713 },
		{ 2916, 2915 },
		{ 1687, 1686 },
		{ 0, 1470 },
		{ 2649, 2623 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 0, 1470 },
		{ 182, 173 },
		{ 192, 173 },
		{ 184, 173 },
		{ 1345, 1344 },
		{ 179, 173 },
		{ 2969, 2968 },
		{ 183, 173 },
		{ 2020, 1998 },
		{ 181, 173 },
		{ 1640, 1639 },
		{ 1345, 1344 },
		{ 1298, 1297 },
		{ 190, 173 },
		{ 189, 173 },
		{ 180, 173 },
		{ 188, 173 },
		{ 1640, 1639 },
		{ 185, 173 },
		{ 187, 173 },
		{ 178, 173 },
		{ 3010, 3009 },
		{ 2050, 2029 },
		{ 186, 173 },
		{ 191, 173 },
		{ 2290, 2258 },
		{ 3033, 3032 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 2258, 2258 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 3077, 3077 },
		{ 100, 83 },
		{ 2291, 2258 },
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
		{ 3041, 3040 },
		{ 454, 409 },
		{ 459, 409 },
		{ 456, 409 },
		{ 455, 409 },
		{ 458, 409 },
		{ 453, 409 },
		{ 3083, 65 },
		{ 452, 409 },
		{ 2079, 2061 },
		{ 67, 65 },
		{ 101, 83 },
		{ 457, 409 },
		{ 2035, 2011 },
		{ 460, 409 },
		{ 2480, 2452 },
		{ 2093, 2076 },
		{ 1859, 1840 },
		{ 1881, 1862 },
		{ 2830, 2829 },
		{ 451, 409 },
		{ 2291, 2258 },
		{ 1239, 1236 },
		{ 3078, 3077 },
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 1236, 1236 },
		{ 2034, 2011 },
		{ 1772, 1771 },
		{ 3095, 3091 },
		{ 2870, 2869 },
		{ 3115, 3114 },
		{ 3131, 3128 },
		{ 3137, 3134 },
		{ 1862, 1843 },
		{ 2237, 2236 },
		{ 101, 83 },
		{ 1868, 1849 },
		{ 1240, 1236 },
		{ 1239, 1239 },
		{ 1239, 1239 },
		{ 1239, 1239 },
		{ 1239, 1239 },
		{ 1239, 1239 },
		{ 1239, 1239 },
		{ 1239, 1239 },
		{ 1239, 1239 },
		{ 1239, 1239 },
		{ 1239, 1239 },
		{ 2657, 2631 },
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
		{ 1242, 1239 },
		{ 2509, 2509 },
		{ 2509, 2509 },
		{ 2435, 2435 },
		{ 2435, 2435 },
		{ 1301, 1300 },
		{ 2668, 2643 },
		{ 2672, 2647 },
		{ 2682, 2658 },
		{ 3081, 65 },
		{ 1240, 1236 },
		{ 1245, 1241 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 2290, 2290 },
		{ 3079, 65 },
		{ 1888, 1869 },
		{ 2509, 2509 },
		{ 2688, 2664 },
		{ 2435, 2435 },
		{ 2701, 2681 },
		{ 2703, 2683 },
		{ 2718, 2698 },
		{ 2719, 2699 },
		{ 3069, 63 },
		{ 1242, 1239 },
		{ 2321, 2290 },
		{ 67, 63 },
		{ 1340, 1339 },
		{ 1915, 1899 },
		{ 2729, 2709 },
		{ 1917, 1902 },
		{ 2734, 2716 },
		{ 2744, 2727 },
		{ 1919, 1904 },
		{ 2751, 2735 },
		{ 1245, 1241 },
		{ 3082, 65 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
		{ 2292, 2292 },
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
		{ 2321, 2290 },
		{ 2324, 2292 },
		{ 2754, 2738 },
		{ 2343, 2343 },
		{ 2343, 2343 },
		{ 1235, 9 },
		{ 2500, 2473 },
		{ 2486, 2486 },
		{ 2486, 2486 },
		{ 67, 9 },
		{ 1969, 1967 },
		{ 118, 102 },
		{ 2772, 2757 },
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
		{ 2786, 2780 },
		{ 2788, 2782 },
		{ 3068, 63 },
		{ 2343, 2343 },
		{ 2794, 2789 },
		{ 1235, 9 },
		{ 3067, 63 },
		{ 2486, 2486 },
		{ 2934, 2934 },
		{ 2934, 2934 },
		{ 2324, 2292 },
		{ 115, 100 },
		{ 2981, 2981 },
		{ 2981, 2981 },
		{ 2800, 2799 },
		{ 2540, 2509 },
		{ 2539, 2509 },
		{ 2465, 2435 },
		{ 2464, 2435 },
		{ 1237, 9 },
		{ 118, 102 },
		{ 1236, 9 },
		{ 1236, 9 },
		{ 1236, 9 },
		{ 1236, 9 },
		{ 1236, 9 },
		{ 1236, 9 },
		{ 1236, 9 },
		{ 1236, 9 },
		{ 1236, 9 },
		{ 1236, 9 },
		{ 2934, 2934 },
		{ 2487, 2487 },
		{ 2487, 2487 },
		{ 2502, 2475 },
		{ 2981, 2981 },
		{ 2497, 2497 },
		{ 2497, 2497 },
		{ 2362, 2362 },
		{ 2362, 2362 },
		{ 2517, 2517 },
		{ 2517, 2517 },
		{ 1716, 1715 },
		{ 115, 100 },
		{ 2388, 2388 },
		{ 2388, 2388 },
		{ 2531, 2531 },
		{ 2531, 2531 },
		{ 2551, 2551 },
		{ 2551, 2551 },
		{ 2605, 2605 },
		{ 2605, 2605 },
		{ 2702, 2702 },
		{ 2702, 2702 },
		{ 2833, 2832 },
		{ 2487, 2487 },
		{ 3065, 63 },
		{ 2842, 2841 },
		{ 3066, 63 },
		{ 2497, 2497 },
		{ 2855, 2854 },
		{ 2362, 2362 },
		{ 1738, 1737 },
		{ 2517, 2517 },
		{ 1751, 1750 },
		{ 2409, 2409 },
		{ 2409, 2409 },
		{ 2388, 2388 },
		{ 1286, 1285 },
		{ 2531, 2531 },
		{ 2878, 2877 },
		{ 2551, 2551 },
		{ 1635, 1634 },
		{ 2605, 2605 },
		{ 2445, 2417 },
		{ 2702, 2702 },
		{ 2374, 2343 },
		{ 2866, 2866 },
		{ 2866, 2866 },
		{ 2894, 2894 },
		{ 2894, 2894 },
		{ 1760, 1759 },
		{ 2491, 2491 },
		{ 2491, 2491 },
		{ 2516, 2516 },
		{ 2516, 2516 },
		{ 2091, 2074 },
		{ 2375, 2343 },
		{ 2409, 2409 },
		{ 2367, 2367 },
		{ 2367, 2367 },
		{ 2513, 2486 },
		{ 2092, 2075 },
		{ 2455, 2455 },
		{ 2455, 2455 },
		{ 2230, 2230 },
		{ 2230, 2230 },
		{ 2484, 2484 },
		{ 2484, 2484 },
		{ 2906, 2905 },
		{ 2866, 2866 },
		{ 1363, 1362 },
		{ 2894, 2894 },
		{ 2512, 2512 },
		{ 2512, 2512 },
		{ 2491, 2491 },
		{ 2933, 2932 },
		{ 2516, 2516 },
		{ 2297, 2297 },
		{ 2297, 2297 },
		{ 2948, 2948 },
		{ 2948, 2948 },
		{ 2367, 2367 },
		{ 2505, 2505 },
		{ 2505, 2505 },
		{ 2935, 2934 },
		{ 2455, 2455 },
		{ 1775, 1774 },
		{ 2230, 2230 },
		{ 2982, 2981 },
		{ 2484, 2484 },
		{ 2110, 2097 },
		{ 2989, 2989 },
		{ 2989, 2989 },
		{ 2963, 2962 },
		{ 2123, 2108 },
		{ 2512, 2512 },
		{ 2972, 2971 },
		{ 2611, 2611 },
		{ 2611, 2611 },
		{ 2526, 2497 },
		{ 2297, 2297 },
		{ 2980, 2979 },
		{ 2948, 2948 },
		{ 2660, 2660 },
		{ 2660, 2660 },
		{ 2505, 2505 },
		{ 2458, 2429 },
		{ 2527, 2497 },
		{ 2514, 2487 },
		{ 2412, 2412 },
		{ 2412, 2412 },
		{ 2705, 2705 },
		{ 2705, 2705 },
		{ 2389, 2362 },
		{ 2989, 2989 },
		{ 2548, 2517 },
		{ 2895, 2894 },
		{ 2370, 2370 },
		{ 2370, 2370 },
		{ 2417, 2388 },
		{ 2611, 2611 },
		{ 2562, 2531 },
		{ 2459, 2430 },
		{ 2581, 2551 },
		{ 2520, 2491 },
		{ 2631, 2605 },
		{ 2660, 2660 },
		{ 2722, 2702 },
		{ 3004, 3003 },
		{ 2903, 2903 },
		{ 2903, 2903 },
		{ 2124, 2109 },
		{ 2412, 2412 },
		{ 3013, 3012 },
		{ 2705, 2705 },
		{ 2636, 2636 },
		{ 2636, 2636 },
		{ 3024, 3023 },
		{ 1374, 1373 },
		{ 2437, 2409 },
		{ 2370, 2370 },
		{ 2518, 2491 },
		{ 2184, 2181 },
		{ 3035, 3034 },
		{ 2199, 2198 },
		{ 2519, 2491 },
		{ 3044, 3043 },
		{ 1265, 1264 },
		{ 2895, 2894 },
		{ 2472, 2442 },
		{ 2567, 2537 },
		{ 2867, 2866 },
		{ 2903, 2903 },
		{ 1688, 1687 },
		{ 2476, 2446 },
		{ 2225, 2224 },
		{ 2547, 2516 },
		{ 2587, 2559 },
		{ 2636, 2636 },
		{ 3091, 3087 },
		{ 2602, 2576 },
		{ 2395, 2367 },
		{ 1690, 1689 },
		{ 2608, 2582 },
		{ 3117, 3116 },
		{ 2483, 2455 },
		{ 3119, 3119 },
		{ 2231, 2230 },
		{ 2418, 2389 },
		{ 2511, 2484 },
		{ 2618, 2592 },
		{ 3144, 3142 },
		{ 2070, 2049 },
		{ 1841, 1816 },
		{ 2030, 2005 },
		{ 2543, 2512 },
		{ 1840, 1816 },
		{ 2029, 2005 },
		{ 2072, 2051 },
		{ 2328, 2297 },
		{ 2075, 2054 },
		{ 2949, 2948 },
		{ 2532, 2502 },
		{ 1254, 1253 },
		{ 2535, 2505 },
		{ 2359, 2359 },
		{ 2359, 2359 },
		{ 1628, 1627 },
		{ 1629, 1628 },
		{ 3119, 3119 },
		{ 2804, 2803 },
		{ 1768, 1767 },
		{ 2825, 2824 },
		{ 2990, 2989 },
		{ 1369, 1368 },
		{ 2549, 2518 },
		{ 2837, 2836 },
		{ 2553, 2522 },
		{ 1309, 1308 },
		{ 2637, 2611 },
		{ 2561, 2529 },
		{ 1791, 1790 },
		{ 1792, 1791 },
		{ 1797, 1796 },
		{ 1656, 1655 },
		{ 2684, 2660 },
		{ 1657, 1656 },
		{ 1663, 1662 },
		{ 2359, 2359 },
		{ 2889, 2887 },
		{ 2588, 2561 },
		{ 2440, 2412 },
		{ 1664, 1663 },
		{ 2725, 2705 },
		{ 1316, 1315 },
		{ 2910, 2909 },
		{ 1860, 1841 },
		{ 1682, 1681 },
		{ 1867, 1848 },
		{ 2398, 2370 },
		{ 2622, 2596 },
		{ 2240, 2239 },
		{ 2634, 2608 },
		{ 1683, 1682 },
		{ 1879, 1860 },
		{ 1317, 1316 },
		{ 1319, 1318 },
		{ 2494, 2466 },
		{ 1592, 1583 },
		{ 2498, 2471 },
		{ 3023, 3022 },
		{ 2904, 2903 },
		{ 1706, 1705 },
		{ 1707, 1706 },
		{ 2689, 2665 },
		{ 2662, 2636 },
		{ 2690, 2666 },
		{ 2692, 2669 },
		{ 2693, 2672 },
		{ 1333, 1332 },
		{ 2503, 2476 },
		{ 1930, 1917 },
		{ 1949, 1939 },
		{ 2721, 2701 },
		{ 1957, 1949 },
		{ 1334, 1333 },
		{ 1294, 1293 },
		{ 2730, 2710 },
		{ 3107, 3106 },
		{ 3108, 3107 },
		{ 1732, 1731 },
		{ 1733, 1732 },
		{ 1269, 1268 },
		{ 1742, 1741 },
		{ 3122, 3119 },
		{ 1598, 1592 },
		{ 2051, 2030 },
		{ 2761, 2745 },
		{ 2453, 2424 },
		{ 3121, 3119 },
		{ 2617, 2591 },
		{ 3120, 3119 },
		{ 2827, 2826 },
		{ 1964, 1959 },
		{ 2510, 2483 },
		{ 1256, 1255 },
		{ 2633, 2607 },
		{ 2853, 2852 },
		{ 1284, 1283 },
		{ 2864, 2863 },
		{ 1852, 1833 },
		{ 2638, 2612 },
		{ 2227, 2226 },
		{ 1744, 1743 },
		{ 1347, 1346 },
		{ 2233, 2232 },
		{ 2669, 2644 },
		{ 2521, 2492 },
		{ 2893, 2891 },
		{ 2673, 2648 },
		{ 2385, 2385 },
		{ 2385, 2385 },
		{ 1753, 1752 },
		{ 2239, 2238 },
		{ 2386, 2359 },
		{ 2041, 2020 },
		{ 1863, 1844 },
		{ 1361, 1360 },
		{ 2062, 2041 },
		{ 2961, 2960 },
		{ 2542, 2511 },
		{ 1758, 1757 },
		{ 2707, 2687 },
		{ 2711, 2691 },
		{ 2544, 2513 },
		{ 2545, 2514 },
		{ 3002, 3001 },
		{ 1871, 1852 },
		{ 1642, 1641 },
		{ 1692, 1691 },
		{ 1263, 1262 },
		{ 1898, 1881 },
		{ 1365, 1364 },
		{ 2385, 2385 },
		{ 2733, 2715 },
		{ 1661, 1660 },
		{ 2741, 2724 },
		{ 1278, 1277 },
		{ 2573, 2543 },
		{ 2577, 2547 },
		{ 1371, 1370 },
		{ 2756, 2740 },
		{ 1718, 1717 },
		{ 2177, 2170 },
		{ 2180, 2175 },
		{ 1311, 1310 },
		{ 2773, 2758 },
		{ 3098, 3095 },
		{ 2774, 2760 },
		{ 2595, 2569 },
		{ 2787, 2781 },
		{ 1668, 1667 },
		{ 2196, 2193 },
		{ 3119, 3118 },
		{ 2795, 2793 },
		{ 3127, 3124 },
		{ 2798, 2796 },
		{ 3135, 3132 },
		{ 2403, 2375 },
		{ 2609, 2583 },
		{ 3147, 3146 },
		{ 2342, 2342 },
		{ 2342, 2342 },
		{ 2784, 2784 },
		{ 2784, 2784 },
		{ 2436, 2436 },
		{ 2436, 2436 },
		{ 1798, 1797 },
		{ 2559, 2527 },
		{ 2583, 2553 },
		{ 1373, 1372 },
		{ 2780, 2771 },
		{ 2413, 2385 },
		{ 2683, 2659 },
		{ 2452, 2423 },
		{ 2565, 2535 },
		{ 1902, 1887 },
		{ 1359, 1358 },
		{ 2648, 2622 },
		{ 2606, 2580 },
		{ 2742, 2725 },
		{ 2698, 2677 },
		{ 2699, 2678 },
		{ 2905, 2904 },
		{ 2342, 2342 },
		{ 2576, 2546 },
		{ 2784, 2784 },
		{ 2061, 2040 },
		{ 2436, 2436 },
		{ 2704, 2684 },
		{ 2663, 2637 },
		{ 1737, 1736 },
		{ 1830, 1807 },
		{ 2857, 2856 },
		{ 2863, 2862 },
		{ 2040, 2019 },
		{ 2691, 2668 },
		{ 2481, 2453 },
		{ 1691, 1690 },
		{ 1864, 1845 },
		{ 2880, 2879 },
		{ 2564, 2534 },
		{ 1829, 1807 },
		{ 2414, 2385 },
		{ 1866, 1847 },
		{ 2055, 2034 },
		{ 2572, 2542 },
		{ 2423, 2395 },
		{ 2057, 2036 },
		{ 2059, 2038 },
		{ 1358, 1357 },
		{ 1700, 1699 },
		{ 2908, 2907 },
		{ 1267, 1266 },
		{ 2915, 2914 },
		{ 1448, 1426 },
		{ 2728, 2708 },
		{ 1662, 1661 },
		{ 1277, 1276 },
		{ 2090, 2073 },
		{ 2599, 2573 },
		{ 2965, 2964 },
		{ 1897, 1880 },
		{ 2603, 2577 },
		{ 2974, 2973 },
		{ 1777, 1776 },
		{ 1785, 1784 },
		{ 1327, 1326 },
		{ 2105, 2089 },
		{ 1912, 1896 },
		{ 3006, 3005 },
		{ 1717, 1716 },
		{ 2760, 2744 },
		{ 3015, 3014 },
		{ 1665, 1664 },
		{ 2762, 2746 },
		{ 2619, 2593 },
		{ 2155, 2137 },
		{ 2156, 2138 },
		{ 3037, 3036 },
		{ 1726, 1725 },
		{ 1622, 1621 },
		{ 3046, 3045 },
		{ 2182, 2179 },
		{ 2781, 2772 },
		{ 1376, 1375 },
		{ 2644, 2618 },
		{ 1736, 1735 },
		{ 2190, 2187 },
		{ 2793, 2788 },
		{ 2193, 2191 },
		{ 2658, 2632 },
		{ 3096, 3092 },
		{ 1676, 1675 },
		{ 1966, 1963 },
		{ 2802, 2801 },
		{ 2541, 2510 },
		{ 2373, 2342 },
		{ 1364, 1363 },
		{ 2789, 2784 },
		{ 1972, 1971 },
		{ 2466, 2436 },
		{ 3118, 3117 },
		{ 1845, 1822 },
		{ 1483, 1463 },
		{ 3124, 3121 },
		{ 1303, 1302 },
		{ 2835, 2834 },
		{ 1426, 1402 },
		{ 2686, 2662 },
		{ 1650, 1649 },
		{ 3146, 3144 },
		{ 2036, 2013 },
		{ 2956, 2956 },
		{ 2956, 2956 },
		{ 2848, 2848 },
		{ 2848, 2848 },
		{ 2997, 2997 },
		{ 2997, 2997 },
		{ 2504, 2504 },
		{ 2504, 2504 },
		{ 2479, 2479 },
		{ 2479, 2479 },
		{ 2582, 2552 },
		{ 2499, 2472 },
		{ 2890, 2888 },
		{ 2584, 2554 },
		{ 2586, 2558 },
		{ 1869, 1850 },
		{ 1778, 1777 },
		{ 1784, 1783 },
		{ 2229, 2228 },
		{ 2909, 2908 },
		{ 1633, 1632 },
		{ 2052, 2031 },
		{ 1356, 1355 },
		{ 2956, 2956 },
		{ 2917, 2916 },
		{ 2848, 2848 },
		{ 2926, 2925 },
		{ 2997, 2997 },
		{ 1503, 1484 },
		{ 2504, 2504 },
		{ 2745, 2728 },
		{ 2479, 2479 },
		{ 2943, 2942 },
		{ 2944, 2943 },
		{ 2946, 2945 },
		{ 113, 98 },
		{ 1899, 1882 },
		{ 2959, 2958 },
		{ 2325, 2294 },
		{ 1649, 1648 },
		{ 1299, 1298 },
		{ 2966, 2965 },
		{ 1904, 1889 },
		{ 2970, 2969 },
		{ 2074, 2053 },
		{ 1289, 1288 },
		{ 2975, 2974 },
		{ 1914, 1898 },
		{ 1427, 1404 },
		{ 2987, 2986 },
		{ 1699, 1698 },
		{ 1377, 1376 },
		{ 3000, 2999 },
		{ 2776, 2762 },
		{ 1338, 1337 },
		{ 2645, 2619 },
		{ 3007, 3006 },
		{ 2647, 2621 },
		{ 3011, 3010 },
		{ 1938, 1926 },
		{ 2097, 2080 },
		{ 3016, 3015 },
		{ 3022, 3021 },
		{ 1847, 1825 },
		{ 1951, 1942 },
		{ 2661, 2635 },
		{ 1851, 1832 },
		{ 1963, 1958 },
		{ 2135, 2121 },
		{ 3038, 3037 },
		{ 1463, 1441 },
		{ 3042, 3041 },
		{ 2803, 2802 },
		{ 2401, 2373 },
		{ 3047, 3046 },
		{ 2402, 2374 },
		{ 1621, 1620 },
		{ 2157, 2139 },
		{ 3062, 3058 },
		{ 2550, 2519 },
		{ 2831, 2830 },
		{ 2168, 2155 },
		{ 2169, 2156 },
		{ 3089, 3085 },
		{ 2836, 2835 },
		{ 3092, 3088 },
		{ 1268, 1267 },
		{ 2179, 2174 },
		{ 1861, 1842 },
		{ 3101, 3098 },
		{ 2851, 2850 },
		{ 1973, 1972 },
		{ 1763, 1762 },
		{ 2566, 2536 },
		{ 2957, 2956 },
		{ 2858, 2857 },
		{ 2849, 2848 },
		{ 1304, 1303 },
		{ 2998, 2997 },
		{ 1326, 1325 },
		{ 2534, 2504 },
		{ 2192, 2190 },
		{ 2506, 2479 },
		{ 1773, 1772 },
		{ 1675, 1674 },
		{ 2876, 2875 },
		{ 1725, 1724 },
		{ 2714, 2694 },
		{ 2881, 2880 },
		{ 2038, 2016 },
		{ 2039, 2018 },
		{ 2320, 2320 },
		{ 2320, 2320 },
		{ 2739, 2739 },
		{ 2739, 2739 },
		{ 2873, 2873 },
		{ 2873, 2873 },
		{ 3008, 3008 },
		{ 3008, 3008 },
		{ 1296, 1296 },
		{ 1296, 1296 },
		{ 1770, 1770 },
		{ 1770, 1770 },
		{ 2967, 2967 },
		{ 2967, 2967 },
		{ 3039, 3039 },
		{ 3039, 3039 },
		{ 2396, 2396 },
		{ 2396, 2396 },
		{ 2828, 2828 },
		{ 2828, 2828 },
		{ 2538, 2538 },
		{ 2538, 2538 },
		{ 1913, 1897 },
		{ 2320, 2320 },
		{ 2201, 2200 },
		{ 2739, 2739 },
		{ 2508, 2481 },
		{ 2873, 2873 },
		{ 2078, 2059 },
		{ 3008, 3008 },
		{ 3099, 3096 },
		{ 1296, 1296 },
		{ 1968, 1966 },
		{ 1770, 1770 },
		{ 1623, 1622 },
		{ 2967, 2967 },
		{ 1686, 1685 },
		{ 3039, 3039 },
		{ 1885, 1866 },
		{ 2396, 2396 },
		{ 1328, 1327 },
		{ 2828, 2828 },
		{ 1701, 1700 },
		{ 2538, 2538 },
		{ 1740, 1739 },
		{ 2639, 2613 },
		{ 2235, 2234 },
		{ 1786, 1785 },
		{ 2106, 2090 },
		{ 2443, 2415 },
		{ 3134, 3131 },
		{ 2185, 2182 },
		{ 1502, 1483 },
		{ 1677, 1676 },
		{ 1727, 1726 },
		{ 1651, 1650 },
		{ 3028, 3028 },
		{ 3028, 3028 },
		{ 2992, 2992 },
		{ 2992, 2992 },
		{ 1754, 1754 },
		{ 1754, 1754 },
		{ 2951, 2951 },
		{ 2951, 2951 },
		{ 2843, 2843 },
		{ 2843, 2843 },
		{ 1280, 1280 },
		{ 1280, 1280 },
		{ 2985, 2985 },
		{ 2985, 2985 },
		{ 1883, 1864 },
		{ 1291, 1291 },
		{ 1291, 1291 },
		{ 1765, 1765 },
		{ 1765, 1765 },
		{ 2743, 2726 },
		{ 1567, 1541 },
		{ 1626, 1625 },
		{ 1848, 1828 },
		{ 3028, 3028 },
		{ 1343, 1342 },
		{ 2992, 2992 },
		{ 1654, 1653 },
		{ 1754, 1754 },
		{ 1293, 1292 },
		{ 2951, 2951 },
		{ 2411, 2383 },
		{ 2843, 2843 },
		{ 2076, 2055 },
		{ 1280, 1280 },
		{ 2077, 2057 },
		{ 2985, 2985 },
		{ 1470, 1448 },
		{ 2351, 2320 },
		{ 1291, 1291 },
		{ 2755, 2739 },
		{ 1765, 1765 },
		{ 2874, 2873 },
		{ 2189, 2186 },
		{ 3009, 3008 },
		{ 1789, 1788 },
		{ 1297, 1296 },
		{ 2616, 2590 },
		{ 1771, 1770 },
		{ 1730, 1729 },
		{ 2968, 2967 },
		{ 2696, 2675 },
		{ 3040, 3039 },
		{ 1680, 1679 },
		{ 2424, 2396 },
		{ 2779, 2770 },
		{ 2829, 2828 },
		{ 2507, 2480 },
		{ 2568, 2538 },
		{ 2620, 2594 },
		{ 1704, 1703 },
		{ 1261, 1260 },
		{ 2632, 2606 },
		{ 1799, 1798 },
		{ 3105, 3104 },
		{ 2096, 2079 },
		{ 2717, 2697 },
		{ 1767, 1766 },
		{ 3114, 3113 },
		{ 2901, 2900 },
		{ 2224, 2223 },
		{ 2720, 2700 },
		{ 1929, 1916 },
		{ 1638, 1637 },
		{ 2049, 2028 },
		{ 1314, 1313 },
		{ 1941, 1931 },
		{ 2523, 2494 },
		{ 1872, 1853 },
		{ 2154, 2136 },
		{ 3140, 3137 },
		{ 1713, 1712 },
		{ 3032, 3031 },
		{ 1331, 1330 },
		{ 2626, 2626 },
		{ 2626, 2626 },
		{ 2627, 2627 },
		{ 2627, 2627 },
		{ 1257, 1257 },
		{ 1257, 1257 },
		{ 2488, 2488 },
		{ 2488, 2488 },
		{ 2897, 2896 },
		{ 3029, 3028 },
		{ 2924, 2923 },
		{ 2993, 2992 },
		{ 2013, 2204 },
		{ 1755, 1754 },
		{ 1402, 1393 },
		{ 2952, 2951 },
		{ 1822, 1978 },
		{ 2844, 2843 },
		{ 1639, 1638 },
		{ 1281, 1280 },
		{ 2187, 2184 },
		{ 2986, 2985 },
		{ 1739, 1738 },
		{ 2626, 2626 },
		{ 1292, 1291 },
		{ 2627, 2627 },
		{ 1766, 1765 },
		{ 1257, 1257 },
		{ 1375, 1374 },
		{ 2488, 2488 },
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
		{ 1247, 1247 },
		{ 1247, 1247 },
		{ 1247, 1247 },
		{ 1247, 1247 },
		{ 1247, 1247 },
		{ 1247, 1247 },
		{ 1247, 1247 },
		{ 1247, 1247 },
		{ 1247, 1247 },
		{ 1247, 1247 },
		{ 2919, 2919 },
		{ 2919, 2919 },
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
		{ 2442, 2414 },
		{ 1312, 1312 },
		{ 1312, 1312 },
		{ 1849, 1829 },
		{ 2635, 2609 },
		{ 2856, 2855 },
		{ 1776, 1775 },
		{ 3005, 3004 },
		{ 2017, 1995 },
		{ 2198, 2196 },
		{ 1903, 1888 },
		{ 2919, 2919 },
		{ 1741, 1740 },
		{ 2735, 2717 },
		{ 0, 1237 },
		{ 0, 84 },
		{ 2738, 2720 },
		{ 3014, 3013 },
		{ 1853, 1834 },
		{ 2652, 2626 },
		{ 1302, 1301 },
		{ 2653, 2627 },
		{ 1344, 1343 },
		{ 1258, 1257 },
		{ 1312, 1312 },
		{ 2515, 2488 },
		{ 2323, 2323 },
		{ 2323, 2323 },
		{ 2323, 2323 },
		{ 2323, 2323 },
		{ 2323, 2323 },
		{ 2323, 2323 },
		{ 2323, 2323 },
		{ 2323, 2323 },
		{ 2323, 2323 },
		{ 2323, 2323 },
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
		{ 0, 1237 },
		{ 0, 84 },
		{ 2614, 2614 },
		{ 2614, 2614 },
		{ 3053, 3053 },
		{ 3053, 3053 },
		{ 3053, 3053 },
		{ 3053, 3053 },
		{ 3053, 3053 },
		{ 3053, 3053 },
		{ 3053, 3053 },
		{ 3053, 3053 },
		{ 3053, 3053 },
		{ 3053, 3053 },
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
		{ 0, 2259 },
		{ 2614, 2614 },
		{ 1337, 1336 },
		{ 2879, 2878 },
		{ 2107, 2091 },
		{ 2018, 1995 },
		{ 2746, 2729 },
		{ 1403, 1382 },
		{ 2920, 2919 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 2353, 2353 },
		{ 3139, 3139 },
		{ 2578, 2548 },
		{ 1313, 1312 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 2355, 2355 },
		{ 0, 2259 },
		{ 2938, 2938 },
		{ 2938, 2938 },
		{ 2597, 2597 },
		{ 2597, 2597 },
		{ 2859, 2859 },
		{ 2859, 2859 },
		{ 2121, 2105 },
		{ 3036, 3035 },
		{ 2659, 2633 },
		{ 3139, 3139 },
		{ 1266, 1265 },
		{ 1926, 1912 },
		{ 1355, 1354 },
		{ 2665, 2639 },
		{ 2900, 2899 },
		{ 3045, 3044 },
		{ 2415, 2386 },
		{ 2137, 2123 },
		{ 2770, 2755 },
		{ 2771, 2756 },
		{ 2907, 2906 },
		{ 2138, 2124 },
		{ 2383, 2351 },
		{ 2938, 2938 },
		{ 2675, 2650 },
		{ 2597, 2597 },
		{ 1339, 1338 },
		{ 2859, 2859 },
		{ 2589, 2562 },
		{ 2592, 2566 },
		{ 2593, 2567 },
		{ 2594, 2568 },
		{ 2925, 2924 },
		{ 2473, 2443 },
		{ 1796, 1795 },
		{ 2640, 2614 },
		{ 1372, 1371 },
		{ 1632, 1631 },
		{ 2942, 2941 },
		{ 1762, 1761 },
		{ 2694, 2673 },
		{ 1404, 1382 },
		{ 2945, 2944 },
		{ 1288, 1287 },
		{ 2697, 2676 },
		{ 2801, 2800 },
		{ 1634, 1633 },
		{ 1484, 1464 },
		{ 2700, 2680 },
		{ 2613, 2587 },
		{ 2964, 2963 },
		{ 2429, 2401 },
		{ 3127, 3127 },
		{ 2430, 2402 },
		{ 1831, 1809 },
		{ 2708, 2688 },
		{ 1636, 1635 },
		{ 1341, 1340 },
		{ 2973, 2972 },
		{ 2834, 2833 },
		{ 2716, 2696 },
		{ 1971, 1969 },
		{ 2847, 2846 },
		{ 2983, 2982 },
		{ 2921, 2920 },
		{ 2628, 2602 },
		{ 1287, 1286 },
		{ 2736, 2718 },
		{ 2996, 2995 },
		{ 2737, 2719 },
		{ 3139, 3136 },
		{ 2955, 2954 },
		{ 2200, 2199 },
		{ 2936, 2935 },
		{ 67, 5 },
		{ 3127, 3127 },
		{ 1761, 1760 },
		{ 2927, 2926 },
		{ 2191, 2189 },
		{ 3141, 3139 },
		{ 2947, 2946 },
		{ 1290, 1289 },
		{ 2732, 2714 },
		{ 2571, 2541 },
		{ 2941, 2940 },
		{ 1354, 1353 },
		{ 1764, 1763 },
		{ 2685, 2661 },
		{ 2887, 2885 },
		{ 2939, 2938 },
		{ 3084, 3079 },
		{ 2623, 2597 },
		{ 1842, 1819 },
		{ 2860, 2859 },
		{ 1980, 1977 },
		{ 2470, 2440 },
		{ 1843, 1819 },
		{ 2695, 2674 },
		{ 2221, 2220 },
		{ 1979, 1977 },
		{ 1824, 1804 },
		{ 2950, 2949 },
		{ 2674, 2649 },
		{ 2596, 2570 },
		{ 2888, 2885 },
		{ 1823, 1804 },
		{ 2991, 2990 },
		{ 1464, 1442 },
		{ 2358, 2328 },
		{ 2862, 2861 },
		{ 2015, 1992 },
		{ 2493, 2465 },
		{ 2899, 2898 },
		{ 2522, 2493 },
		{ 1276, 1275 },
		{ 2014, 1992 },
		{ 2426, 2398 },
		{ 2570, 2540 },
		{ 1750, 1749 },
		{ 2206, 2203 },
		{ 3085, 3079 },
		{ 1832, 1809 },
		{ 1420, 1394 },
		{ 2940, 2939 },
		{ 2205, 2203 },
		{ 2170, 2157 },
		{ 174, 5 },
		{ 2174, 2167 },
		{ 2439, 2411 },
		{ 1728, 1727 },
		{ 175, 5 },
		{ 2016, 1993 },
		{ 2529, 2499 },
		{ 1279, 1278 },
		{ 1783, 1782 },
		{ 2641, 2615 },
		{ 1878, 1859 },
		{ 176, 5 },
		{ 1275, 1274 },
		{ 2646, 2620 },
		{ 2958, 2957 },
		{ 2536, 2506 },
		{ 2960, 2959 },
		{ 1678, 1677 },
		{ 3130, 3127 },
		{ 2785, 2779 },
		{ 1882, 1863 },
		{ 2188, 2185 },
		{ 1523, 1502 },
		{ 1787, 1786 },
		{ 1886, 1867 },
		{ 2546, 2515 },
		{ 1524, 1503 },
		{ 173, 5 },
		{ 2796, 2794 },
		{ 1889, 1871 },
		{ 1318, 1317 },
		{ 1295, 1294 },
		{ 2552, 2521 },
		{ 2460, 2431 },
		{ 2984, 2983 },
		{ 2053, 2032 },
		{ 2558, 2526 },
		{ 2054, 2033 },
		{ 2826, 2825 },
		{ 1641, 1640 },
		{ 2677, 2652 },
		{ 2678, 2653 },
		{ 2999, 2998 },
		{ 1360, 1359 },
		{ 3001, 3000 },
		{ 1901, 1885 },
		{ 2467, 2437 },
		{ 1648, 1647 },
		{ 2220, 2219 },
		{ 2687, 2663 },
		{ 1743, 1742 },
		{ 1325, 1324 },
		{ 2071, 2050 },
		{ 1255, 1254 },
		{ 2850, 2849 },
		{ 1698, 1697 },
		{ 2852, 2851 },
		{ 2574, 2544 },
		{ 2575, 2545 },
		{ 1825, 1805 },
		{ 3025, 3024 },
		{ 1826, 1806 },
		{ 1827, 1806 },
		{ 1752, 1751 },
		{ 1441, 1419 },
		{ 2861, 2860 },
		{ 1652, 1651 },
		{ 2236, 2235 },
		{ 2080, 2062 },
		{ 2865, 2864 },
		{ 2238, 2237 },
		{ 2868, 2867 },
		{ 2088, 2070 },
		{ 2872, 2871 },
		{ 2585, 2555 },
		{ 1442, 1420 },
		{ 2710, 2690 },
		{ 1757, 1756 },
		{ 3058, 3049 },
		{ 1702, 1701 },
		{ 2715, 2695 },
		{ 1620, 1619 },
		{ 2591, 2565 },
		{ 1942, 1932 },
		{ 2095, 2078 },
		{ 1310, 1309 },
		{ 1283, 1282 },
		{ 1850, 1830 },
		{ 3087, 3082 },
		{ 3088, 3084 },
		{ 2891, 2889 },
		{ 2724, 2704 },
		{ 1958, 1950 },
		{ 2726, 2706 },
		{ 2898, 2897 },
		{ 2600, 2574 },
		{ 2601, 2575 },
		{ 1959, 1951 },
		{ 3102, 3099 },
		{ 2902, 2901 },
		{ 1710, 1709 },
		{ 1368, 1367 },
		{ 3111, 3110 },
		{ 1624, 1623 },
		{ 1329, 1328 },
		{ 1346, 1345 },
		{ 2612, 2586 },
		{ 1970, 1968 },
		{ 1259, 1258 },
		{ 1828, 1806 },
		{ 2139, 2125 },
		{ 3123, 3120 },
		{ 1769, 1768 },
		{ 1667, 1666 },
		{ 2431, 2403 },
		{ 2922, 2921 },
		{ 3132, 3129 },
		{ 2294, 2264 },
		{ 2621, 2595 },
		{ 1724, 1723 },
		{ 1353, 1352 },
		{ 1674, 1673 },
		{ 2630, 2604 },
		{ 98, 81 },
		{ 2937, 2936 },
		{ 2758, 2742 },
		{ 2610, 2584 },
		{ 1890, 1872 },
		{ 1449, 1427 },
		{ 2060, 2039 },
		{ 2706, 2686 },
		{ 2495, 2467 },
		{ 2032, 2009 },
		{ 2869, 2868 },
		{ 2988, 2987 },
		{ 3129, 3126 },
		{ 3093, 3089 },
		{ 1367, 1366 },
		{ 2892, 2890 },
		{ 128, 113 },
		{ 1795, 1794 },
		{ 2918, 2917 },
		{ 1880, 1861 },
		{ 3027, 3026 },
		{ 2073, 2052 },
		{ 1870, 1851 },
		{ 2356, 2325 },
		{ 3064, 3062 },
		{ 2954, 2953 },
		{ 1774, 1773 },
		{ 2871, 2870 },
		{ 1918, 1903 },
		{ 1300, 1299 },
		{ 1370, 1369 },
		{ 3012, 3011 },
		{ 3125, 3122 },
		{ 3043, 3042 },
		{ 3128, 3125 },
		{ 1260, 1259 },
		{ 2877, 2876 },
		{ 2740, 2722 },
		{ 2846, 2845 },
		{ 2607, 2581 },
		{ 2995, 2994 },
		{ 3136, 3133 },
		{ 622, 563 },
		{ 2923, 2922 },
		{ 2555, 2524 },
		{ 2971, 2970 },
		{ 3143, 3141 },
		{ 2122, 2107 },
		{ 2832, 2831 },
		{ 1660, 1659 },
		{ 2013, 2007 },
		{ 2625, 2599 },
		{ 623, 563 },
		{ 2676, 2651 },
		{ 2629, 2603 },
		{ 2680, 2655 },
		{ 2650, 2624 },
		{ 1352, 1349 },
		{ 1822, 1818 },
		{ 2598, 2572 },
		{ 1402, 1395 },
		{ 202, 179 },
		{ 2624, 2598 },
		{ 2681, 2657 },
		{ 200, 179 },
		{ 2537, 2507 },
		{ 201, 179 },
		{ 1342, 1341 },
		{ 624, 563 },
		{ 1332, 1331 },
		{ 2896, 2895 },
		{ 1715, 1714 },
		{ 1262, 1261 },
		{ 1637, 1636 },
		{ 2222, 2221 },
		{ 199, 179 },
		{ 1681, 1680 },
		{ 1627, 1626 },
		{ 2590, 2564 },
		{ 1931, 1918 },
		{ 2175, 2168 },
		{ 1315, 1314 },
		{ 2845, 2844 },
		{ 1939, 1929 },
		{ 2643, 2617 },
		{ 1790, 1789 },
		{ 2757, 2741 },
		{ 2554, 2523 },
		{ 2994, 2993 },
		{ 2232, 2231 },
		{ 2181, 2177 },
		{ 2854, 2853 },
		{ 2234, 2233 },
		{ 1655, 1654 },
		{ 1705, 1704 },
		{ 2651, 2625 },
		{ 3003, 3002 },
		{ 3106, 3105 },
		{ 2475, 2445 },
		{ 1264, 1263 },
		{ 2709, 2689 },
		{ 2655, 2629 },
		{ 3113, 3112 },
		{ 1362, 1361 },
		{ 2108, 2093 },
		{ 3116, 3115 },
		{ 2109, 2096 },
		{ 1844, 1820 },
		{ 2782, 2773 },
		{ 2569, 2539 },
		{ 2028, 2004 },
		{ 2664, 2638 },
		{ 1731, 1730 },
		{ 3126, 3123 },
		{ 2666, 2640 },
		{ 1689, 1688 },
		{ 1583, 1567 },
		{ 3026, 3025 },
		{ 2446, 2418 },
		{ 2031, 2008 },
		{ 3133, 3130 },
		{ 2953, 2952 },
		{ 3031, 3030 },
		{ 2727, 2707 },
		{ 2799, 2798 },
		{ 3034, 3033 },
		{ 2136, 2122 },
		{ 1800, 1799 },
		{ 3142, 3140 },
		{ 2492, 2464 },
		{ 1967, 1964 },
		{ 1712, 1711 },
		{ 2962, 2961 },
		{ 750, 692 },
		{ 863, 811 },
		{ 426, 384 },
		{ 683, 621 },
		{ 1722, 31 },
		{ 1252, 11 },
		{ 1351, 19 },
		{ 67, 31 },
		{ 67, 11 },
		{ 67, 19 },
		{ 2823, 45 },
		{ 1748, 33 },
		{ 1696, 29 },
		{ 67, 45 },
		{ 67, 33 },
		{ 67, 29 },
		{ 1323, 17 },
		{ 749, 692 },
		{ 427, 384 },
		{ 67, 17 },
		{ 2930, 55 },
		{ 1672, 27 },
		{ 1180, 1169 },
		{ 67, 55 },
		{ 67, 27 },
		{ 684, 621 },
		{ 1273, 13 },
		{ 864, 811 },
		{ 3020, 59 },
		{ 67, 13 },
		{ 1646, 25 },
		{ 67, 59 },
		{ 1186, 1175 },
		{ 67, 25 },
		{ 1221, 1220 },
		{ 2042, 2021 },
		{ 272, 232 },
		{ 279, 239 },
		{ 294, 251 },
		{ 301, 258 },
		{ 307, 263 },
		{ 317, 273 },
		{ 1854, 1835 },
		{ 334, 289 },
		{ 337, 292 },
		{ 344, 298 },
		{ 345, 299 },
		{ 350, 304 },
		{ 2064, 2043 },
		{ 2065, 2044 },
		{ 364, 321 },
		{ 388, 346 },
		{ 391, 349 },
		{ 398, 356 },
		{ 409, 368 },
		{ 416, 375 },
		{ 425, 383 },
		{ 230, 195 },
		{ 1874, 1855 },
		{ 1875, 1856 },
		{ 439, 394 },
		{ 2087, 2069 },
		{ 234, 199 },
		{ 461, 410 },
		{ 464, 414 },
		{ 474, 422 },
		{ 475, 423 },
		{ 488, 436 },
		{ 497, 446 },
		{ 530, 470 },
		{ 540, 478 },
		{ 2103, 2086 },
		{ 542, 480 },
		{ 1895, 1877 },
		{ 543, 481 },
		{ 547, 485 },
		{ 560, 500 },
		{ 564, 504 },
		{ 568, 508 },
		{ 578, 518 },
		{ 593, 531 },
		{ 594, 533 },
		{ 1910, 1894 },
		{ 599, 538 },
		{ 614, 553 },
		{ 235, 200 },
		{ 1720, 31 },
		{ 1250, 11 },
		{ 1349, 19 },
		{ 631, 567 },
		{ 644, 580 },
		{ 647, 583 },
		{ 2821, 45 },
		{ 1746, 33 },
		{ 1694, 29 },
		{ 656, 592 },
		{ 671, 606 },
		{ 672, 607 },
		{ 1321, 17 },
		{ 682, 620 },
		{ 241, 206 },
		{ 702, 639 },
		{ 2929, 55 },
		{ 1670, 27 },
		{ 242, 207 },
		{ 751, 693 },
		{ 763, 704 },
		{ 765, 706 },
		{ 1271, 13 },
		{ 767, 708 },
		{ 3018, 59 },
		{ 772, 713 },
		{ 1644, 25 },
		{ 792, 733 },
		{ 802, 743 },
		{ 806, 747 },
		{ 807, 748 },
		{ 837, 782 },
		{ 855, 803 },
		{ 262, 223 },
		{ 893, 843 },
		{ 902, 852 },
		{ 917, 867 },
		{ 954, 907 },
		{ 958, 911 },
		{ 974, 930 },
		{ 990, 950 },
		{ 1015, 979 },
		{ 1018, 982 },
		{ 1026, 991 },
		{ 1036, 1001 },
		{ 1054, 1021 },
		{ 1073, 1039 },
		{ 1080, 1048 },
		{ 1082, 1050 },
		{ 1095, 1067 },
		{ 1108, 1081 },
		{ 1114, 1088 },
		{ 1121, 1096 },
		{ 1122, 1097 },
		{ 1136, 1117 },
		{ 1137, 1118 },
		{ 1155, 1138 },
		{ 1160, 1144 },
		{ 1168, 1156 },
		{ 67, 41 },
		{ 67, 47 },
		{ 67, 15 },
		{ 67, 23 },
		{ 67, 51 },
		{ 67, 53 },
		{ 432, 388 },
		{ 67, 35 },
		{ 433, 388 },
		{ 431, 388 },
		{ 67, 57 },
		{ 218, 187 },
		{ 2117, 2102 },
		{ 2115, 2101 },
		{ 2176, 2169 },
		{ 216, 187 },
		{ 3077, 3076 },
		{ 466, 416 },
		{ 468, 416 },
		{ 2118, 2102 },
		{ 2116, 2101 },
		{ 726, 669 },
		{ 585, 525 },
		{ 408, 367 },
		{ 499, 448 },
		{ 434, 388 },
		{ 817, 758 },
		{ 435, 389 },
		{ 469, 416 },
		{ 725, 668 },
		{ 219, 187 },
		{ 217, 187 },
		{ 2113, 2099 },
		{ 467, 416 },
		{ 507, 455 },
		{ 508, 455 },
		{ 1922, 1908 },
		{ 2025, 2001 },
		{ 814, 755 },
		{ 289, 246 },
		{ 330, 285 },
		{ 520, 464 },
		{ 509, 455 },
		{ 1111, 1084 },
		{ 212, 184 },
		{ 2024, 2001 },
		{ 1117, 1091 },
		{ 211, 184 },
		{ 681, 619 },
		{ 441, 396 },
		{ 522, 464 },
		{ 2046, 2025 },
		{ 628, 564 },
		{ 213, 184 },
		{ 357, 311 },
		{ 521, 464 },
		{ 627, 564 },
		{ 2023, 2001 },
		{ 205, 181 },
		{ 908, 858 },
		{ 207, 181 },
		{ 909, 859 },
		{ 225, 190 },
		{ 206, 181 },
		{ 626, 564 },
		{ 625, 564 },
		{ 419, 377 },
		{ 418, 377 },
		{ 511, 457 },
		{ 514, 459 },
		{ 266, 226 },
		{ 2047, 2026 },
		{ 515, 459 },
		{ 1126, 1101 },
		{ 224, 190 },
		{ 713, 654 },
		{ 818, 759 },
		{ 819, 760 },
		{ 256, 218 },
		{ 2217, 41 },
		{ 2839, 47 },
		{ 1306, 15 },
		{ 1617, 23 },
		{ 2885, 51 },
		{ 2912, 53 },
		{ 1857, 1838 },
		{ 1780, 35 },
		{ 298, 255 },
		{ 512, 457 },
		{ 2977, 57 },
		{ 590, 530 },
		{ 1125, 1101 },
		{ 222, 188 },
		{ 577, 517 },
		{ 591, 530 },
		{ 220, 188 },
		{ 904, 854 },
		{ 280, 240 },
		{ 221, 188 },
		{ 527, 468 },
		{ 1158, 1141 },
		{ 873, 822 },
		{ 2471, 2441 },
		{ 716, 657 },
		{ 1253, 1250 },
		{ 592, 530 },
		{ 1176, 1165 },
		{ 3075, 3073 },
		{ 1177, 1166 },
		{ 1178, 1167 },
		{ 1058, 1025 },
		{ 798, 739 },
		{ 281, 240 },
		{ 3086, 3081 },
		{ 528, 468 },
		{ 544, 482 },
		{ 2824, 2821 },
		{ 1081, 1049 },
		{ 236, 201 },
		{ 747, 690 },
		{ 748, 691 },
		{ 421, 379 },
		{ 1837, 1813 },
		{ 3097, 3094 },
		{ 610, 548 },
		{ 965, 920 },
		{ 969, 925 },
		{ 696, 633 },
		{ 823, 764 },
		{ 484, 432 },
		{ 566, 506 },
		{ 1151, 1134 },
		{ 714, 655 },
		{ 1308, 1306 },
		{ 208, 182 },
		{ 557, 497 },
		{ 617, 556 },
		{ 803, 744 },
		{ 621, 562 },
		{ 209, 182 },
		{ 368, 325 },
		{ 812, 753 },
		{ 813, 754 },
		{ 1130, 1108 },
		{ 1856, 1837 },
		{ 1135, 1115 },
		{ 438, 392 },
		{ 384, 343 },
		{ 1144, 1126 },
		{ 556, 497 },
		{ 1145, 1128 },
		{ 340, 294 },
		{ 341, 295 },
		{ 1156, 1139 },
		{ 394, 352 },
		{ 834, 778 },
		{ 2044, 2023 },
		{ 835, 779 },
		{ 1169, 1157 },
		{ 1171, 1159 },
		{ 339, 294 },
		{ 338, 294 },
		{ 397, 355 },
		{ 845, 792 },
		{ 852, 800 },
		{ 551, 490 },
		{ 1184, 1173 },
		{ 1185, 1174 },
		{ 552, 491 },
		{ 3076, 3075 },
		{ 1190, 1181 },
		{ 1205, 1196 },
		{ 871, 820 },
		{ 1232, 1231 },
		{ 196, 178 },
		{ 878, 827 },
		{ 888, 838 },
		{ 558, 498 },
		{ 198, 178 },
		{ 3090, 3086 },
		{ 401, 359 },
		{ 699, 636 },
		{ 296, 253 },
		{ 346, 300 },
		{ 349, 303 },
		{ 197, 178 },
		{ 918, 868 },
		{ 924, 874 },
		{ 930, 881 },
		{ 3100, 3097 },
		{ 935, 886 },
		{ 940, 893 },
		{ 946, 900 },
		{ 953, 906 },
		{ 572, 512 },
		{ 717, 658 },
		{ 576, 516 },
		{ 489, 437 },
		{ 727, 670 },
		{ 975, 931 },
		{ 987, 947 },
		{ 417, 376 },
		{ 995, 955 },
		{ 996, 956 },
		{ 1000, 960 },
		{ 1001, 961 },
		{ 1007, 967 },
		{ 583, 523 },
		{ 263, 224 },
		{ 277, 237 },
		{ 362, 318 },
		{ 1038, 1003 },
		{ 1040, 1005 },
		{ 1044, 1009 },
		{ 1051, 1018 },
		{ 1053, 1020 },
		{ 304, 261 },
		{ 518, 462 },
		{ 1066, 1032 },
		{ 366, 323 },
		{ 612, 550 },
		{ 794, 735 },
		{ 795, 736 },
		{ 1084, 1052 },
		{ 1092, 1061 },
		{ 524, 466 },
		{ 1107, 1080 },
		{ 801, 742 },
		{ 1109, 1082 },
		{ 1149, 1132 },
		{ 613, 551 },
		{ 412, 371 },
		{ 736, 681 },
		{ 1010, 971 },
		{ 1011, 972 },
		{ 1892, 1874 },
		{ 1012, 974 },
		{ 858, 806 },
		{ 859, 807 },
		{ 1546, 1546 },
		{ 1549, 1549 },
		{ 1552, 1552 },
		{ 1555, 1555 },
		{ 1558, 1558 },
		{ 1561, 1561 },
		{ 1021, 985 },
		{ 324, 279 },
		{ 619, 560 },
		{ 1037, 1002 },
		{ 240, 205 },
		{ 358, 314 },
		{ 1041, 1006 },
		{ 392, 350 },
		{ 570, 510 },
		{ 1579, 1579 },
		{ 1215, 1209 },
		{ 899, 849 },
		{ 901, 851 },
		{ 638, 574 },
		{ 1062, 1028 },
		{ 228, 193 },
		{ 1588, 1588 },
		{ 1546, 1546 },
		{ 1549, 1549 },
		{ 1552, 1552 },
		{ 1555, 1555 },
		{ 1558, 1558 },
		{ 1561, 1561 },
		{ 539, 477 },
		{ 1078, 1045 },
		{ 1509, 1509 },
		{ 654, 590 },
		{ 229, 194 },
		{ 365, 322 },
		{ 581, 521 },
		{ 1087, 1055 },
		{ 677, 612 },
		{ 1579, 1579 },
		{ 2082, 2064 },
		{ 347, 301 },
		{ 437, 391 },
		{ 945, 899 },
		{ 587, 527 },
		{ 405, 364 },
		{ 1588, 1588 },
		{ 1613, 1613 },
		{ 548, 486 },
		{ 701, 638 },
		{ 501, 451 },
		{ 244, 209 },
		{ 971, 927 },
		{ 603, 542 },
		{ 821, 762 },
		{ 1509, 1509 },
		{ 715, 656 },
		{ 604, 543 },
		{ 555, 496 },
		{ 373, 330 },
		{ 356, 310 },
		{ 719, 660 },
		{ 720, 662 },
		{ 1419, 1546 },
		{ 1419, 1549 },
		{ 1419, 1552 },
		{ 1419, 1555 },
		{ 1419, 1558 },
		{ 1419, 1561 },
		{ 722, 665 },
		{ 1613, 1613 },
		{ 378, 335 },
		{ 620, 561 },
		{ 519, 463 },
		{ 731, 673 },
		{ 734, 678 },
		{ 735, 679 },
		{ 379, 337 },
		{ 1419, 1579 },
		{ 745, 688 },
		{ 215, 186 },
		{ 325, 280 },
		{ 1079, 1046 },
		{ 255, 217 },
		{ 642, 578 },
		{ 1419, 1588 },
		{ 532, 472 },
		{ 3072, 3068 },
		{ 913, 863 },
		{ 916, 866 },
		{ 1894, 1876 },
		{ 646, 582 },
		{ 363, 319 },
		{ 1099, 1071 },
		{ 1419, 1509 },
		{ 1102, 1074 },
		{ 1103, 1075 },
		{ 919, 869 },
		{ 922, 872 },
		{ 648, 584 },
		{ 777, 718 },
		{ 1909, 1893 },
		{ 932, 883 },
		{ 1116, 1090 },
		{ 933, 884 },
		{ 1118, 1092 },
		{ 934, 885 },
		{ 783, 724 },
		{ 652, 588 },
		{ 1419, 1613 },
		{ 478, 426 },
		{ 541, 479 },
		{ 586, 526 },
		{ 480, 428 },
		{ 483, 431 },
		{ 422, 380 },
		{ 1936, 1924 },
		{ 1937, 1925 },
		{ 2086, 2068 },
		{ 333, 288 },
		{ 596, 535 },
		{ 972, 928 },
		{ 685, 622 },
		{ 598, 537 },
		{ 978, 937 },
		{ 984, 944 },
		{ 395, 353 },
		{ 988, 948 },
		{ 816, 757 },
		{ 2100, 2083 },
		{ 495, 443 },
		{ 278, 238 },
		{ 997, 957 },
		{ 708, 645 },
		{ 712, 651 },
		{ 309, 265 },
		{ 824, 765 },
		{ 831, 774 },
		{ 833, 776 },
		{ 1218, 1216 },
		{ 1013, 976 },
		{ 1222, 1221 },
		{ 300, 257 },
		{ 402, 360 },
		{ 2133, 2119 },
		{ 2134, 2120 },
		{ 836, 780 },
		{ 370, 327 },
		{ 1028, 993 },
		{ 386, 344 },
		{ 385, 344 },
		{ 283, 241 },
		{ 282, 241 },
		{ 204, 180 },
		{ 664, 599 },
		{ 665, 599 },
		{ 503, 453 },
		{ 534, 474 },
		{ 535, 474 },
		{ 536, 475 },
		{ 537, 475 },
		{ 609, 547 },
		{ 247, 211 },
		{ 203, 180 },
		{ 608, 547 },
		{ 246, 211 },
		{ 250, 214 },
		{ 504, 453 },
		{ 305, 262 },
		{ 251, 214 },
		{ 505, 454 },
		{ 288, 245 },
		{ 760, 702 },
		{ 260, 221 },
		{ 634, 570 },
		{ 274, 234 },
		{ 306, 262 },
		{ 689, 626 },
		{ 252, 214 },
		{ 259, 221 },
		{ 506, 454 },
		{ 287, 245 },
		{ 761, 702 },
		{ 690, 627 },
		{ 691, 628 },
		{ 2104, 2087 },
		{ 3094, 3090 },
		{ 1224, 1223 },
		{ 479, 427 },
		{ 2043, 2022 },
		{ 895, 845 },
		{ 580, 520 },
		{ 1128, 1106 },
		{ 2048, 2027 },
		{ 1129, 1107 },
		{ 443, 398 },
		{ 2119, 2103 },
		{ 1131, 1109 },
		{ 482, 430 },
		{ 271, 231 },
		{ 981, 940 },
		{ 387, 345 },
		{ 820, 761 },
		{ 1147, 1130 },
		{ 516, 460 },
		{ 1893, 1875 },
		{ 463, 412 },
		{ 766, 707 },
		{ 290, 247 },
		{ 769, 710 },
		{ 667, 601 },
		{ 1165, 1153 },
		{ 1166, 1154 },
		{ 1083, 1051 },
		{ 773, 714 },
		{ 1855, 1836 },
		{ 1003, 963 },
		{ 352, 306 },
		{ 1911, 1895 },
		{ 2019, 1996 },
		{ 1858, 1839 },
		{ 2083, 2065 },
		{ 931, 882 },
		{ 3073, 3071 },
		{ 595, 534 },
		{ 353, 307 },
		{ 721, 664 },
		{ 2441, 2413 },
		{ 678, 614 },
		{ 597, 536 },
		{ 258, 220 },
		{ 1924, 1910 },
		{ 265, 225 },
		{ 1887, 1868 },
		{ 1076, 1042 },
		{ 903, 853 },
		{ 707, 644 },
		{ 261, 222 },
		{ 793, 734 },
		{ 264, 225 },
		{ 710, 647 },
		{ 472, 419 },
		{ 248, 212 },
		{ 2022, 2000 },
		{ 601, 540 },
		{ 2026, 2002 },
		{ 839, 786 },
		{ 485, 433 },
		{ 1836, 1812 },
		{ 584, 524 },
		{ 1838, 1814 },
		{ 1191, 1182 },
		{ 2114, 2100 },
		{ 1202, 1193 },
		{ 1203, 1194 },
		{ 1100, 1072 },
		{ 1208, 1199 },
		{ 1004, 964 },
		{ 1216, 1211 },
		{ 1217, 1212 },
		{ 805, 746 },
		{ 1105, 1077 },
		{ 1008, 968 },
		{ 757, 699 },
		{ 1923, 1909 },
		{ 759, 701 },
		{ 295, 252 },
		{ 2152, 2133 },
		{ 1112, 1085 },
		{ 866, 813 },
		{ 545, 483 },
		{ 936, 887 },
		{ 937, 888 },
		{ 1119, 1094 },
		{ 1022, 986 },
		{ 226, 191 },
		{ 876, 825 },
		{ 1947, 1936 },
		{ 588, 528 },
		{ 881, 830 },
		{ 884, 834 },
		{ 885, 835 },
		{ 963, 917 },
		{ 494, 442 },
		{ 891, 841 },
		{ 430, 387 },
		{ 574, 514 },
		{ 1056, 1023 },
		{ 440, 395 },
		{ 776, 717 },
		{ 976, 935 },
		{ 1069, 1035 },
		{ 554, 495 },
		{ 1074, 1040 },
		{ 1164, 1152 },
		{ 1075, 1041 },
		{ 729, 672 },
		{ 2098, 2081 },
		{ 923, 873 },
		{ 1055, 1022 },
		{ 393, 351 },
		{ 444, 399 },
		{ 730, 672 },
		{ 1209, 1202 },
		{ 1210, 1203 },
		{ 1214, 1208 },
		{ 723, 666 },
		{ 449, 407 },
		{ 1124, 1100 },
		{ 1067, 1033 },
		{ 1219, 1217 },
		{ 1127, 1105 },
		{ 1068, 1034 },
		{ 2045, 2024 },
		{ 692, 629 },
		{ 1231, 1230 },
		{ 695, 632 },
		{ 573, 513 },
		{ 1132, 1112 },
		{ 424, 382 },
		{ 618, 557 },
		{ 889, 839 },
		{ 1138, 1119 },
		{ 1143, 1125 },
		{ 655, 591 },
		{ 553, 494 },
		{ 950, 903 },
		{ 303, 260 },
		{ 462, 411 },
		{ 245, 210 },
		{ 291, 248 },
		{ 1088, 1056 },
		{ 1089, 1058 },
		{ 1162, 1150 },
		{ 1163, 1151 },
		{ 630, 566 },
		{ 841, 788 },
		{ 1097, 1069 },
		{ 523, 465 },
		{ 850, 798 },
		{ 1170, 1158 },
		{ 910, 860 },
		{ 1175, 1164 },
		{ 563, 503 },
		{ 1104, 1076 },
		{ 1039, 1004 },
		{ 502, 452 },
		{ 811, 752 },
		{ 1043, 1008 },
		{ 374, 331 },
		{ 1187, 1178 },
		{ 3071, 3067 },
		{ 861, 809 },
		{ 643, 579 },
		{ 1200, 1191 },
		{ 1907, 1891 },
		{ 2085, 2067 },
		{ 447, 403 },
		{ 686, 623 },
		{ 415, 374 },
		{ 354, 308 },
		{ 966, 921 },
		{ 661, 596 },
		{ 663, 598 },
		{ 406, 365 },
		{ 231, 196 },
		{ 616, 555 },
		{ 332, 287 },
		{ 887, 837 },
		{ 676, 611 },
		{ 703, 640 },
		{ 768, 709 },
		{ 843, 790 },
		{ 589, 529 },
		{ 897, 847 },
		{ 602, 541 },
		{ 938, 891 },
		{ 233, 198 },
		{ 436, 390 },
		{ 856, 804 },
		{ 1229, 1228 },
		{ 606, 545 },
		{ 781, 722 },
		{ 2084, 2066 },
		{ 1019, 983 },
		{ 815, 756 },
		{ 1161, 1147 },
		{ 1093, 1063 },
		{ 962, 915 },
		{ 1024, 989 },
		{ 473, 421 },
		{ 1027, 992 },
		{ 348, 302 },
		{ 633, 569 },
		{ 254, 216 },
		{ 372, 329 },
		{ 538, 476 },
		{ 361, 317 },
		{ 697, 634 },
		{ 1110, 1083 },
		{ 920, 870 },
		{ 877, 826 },
		{ 1047, 1014 },
		{ 830, 773 },
		{ 879, 828 },
		{ 929, 880 },
		{ 797, 738 },
		{ 1192, 1183 },
		{ 1197, 1188 },
		{ 600, 539 },
		{ 1201, 1192 },
		{ 1057, 1024 },
		{ 1123, 1098 },
		{ 991, 951 },
		{ 1206, 1197 },
		{ 1207, 1198 },
		{ 1060, 1026 },
		{ 992, 952 },
		{ 799, 740 },
		{ 1213, 1207 },
		{ 1061, 1026 },
		{ 700, 637 },
		{ 1059, 1026 },
		{ 232, 197 },
		{ 315, 271 },
		{ 285, 243 },
		{ 840, 787 },
		{ 704, 641 },
		{ 320, 276 },
		{ 2069, 2048 },
		{ 941, 894 },
		{ 1877, 1858 },
		{ 1142, 1123 },
		{ 1225, 1224 },
		{ 942, 895 },
		{ 898, 848 },
		{ 605, 544 },
		{ 1146, 1129 },
		{ 770, 711 },
		{ 1148, 1131 },
		{ 733, 675 },
		{ 1016, 980 },
		{ 1017, 981 },
		{ 267, 227 },
		{ 732, 674 },
		{ 1407, 1407 },
		{ 999, 959 },
		{ 571, 511 },
		{ 842, 789 },
		{ 526, 467 },
		{ 321, 277 },
		{ 669, 604 },
		{ 1049, 1015 },
		{ 525, 467 },
		{ 322, 277 },
		{ 738, 682 },
		{ 737, 682 },
		{ 1048, 1015 },
		{ 739, 682 },
		{ 1173, 1162 },
		{ 718, 659 },
		{ 407, 366 },
		{ 1085, 1053 },
		{ 329, 284 },
		{ 486, 434 },
		{ 1182, 1171 },
		{ 1002, 962 },
		{ 2099, 2082 },
		{ 1407, 1407 },
		{ 319, 275 },
		{ 2101, 2084 },
		{ 2102, 2085 },
		{ 921, 871 },
		{ 1094, 1066 },
		{ 1006, 966 },
		{ 786, 727 },
		{ 787, 728 },
		{ 1193, 1184 },
		{ 1194, 1185 },
		{ 851, 799 },
		{ 1199, 1190 },
		{ 308, 264 },
		{ 853, 801 },
		{ 491, 439 },
		{ 562, 502 },
		{ 1106, 1079 },
		{ 629, 565 },
		{ 493, 441 },
		{ 382, 340 },
		{ 632, 568 },
		{ 865, 812 },
		{ 1212, 1205 },
		{ 367, 324 },
		{ 1023, 987 },
		{ 496, 444 },
		{ 360, 316 },
		{ 640, 576 },
		{ 498, 447 },
		{ 1034, 999 },
		{ 1035, 1000 },
		{ 399, 357 },
		{ 809, 750 },
		{ 880, 829 },
		{ 286, 244 },
		{ 955, 908 },
		{ 956, 909 },
		{ 1042, 1007 },
		{ 1419, 1407 },
		{ 1908, 1892 },
		{ 882, 831 },
		{ 442, 397 },
		{ 477, 425 },
		{ 964, 918 },
		{ 237, 202 },
		{ 1052, 1019 },
		{ 753, 695 },
		{ 968, 924 },
		{ 756, 698 },
		{ 650, 586 },
		{ 758, 700 },
		{ 2066, 2045 },
		{ 2067, 2046 },
		{ 894, 844 },
		{ 403, 361 },
		{ 579, 519 },
		{ 1150, 1133 },
		{ 762, 703 },
		{ 3021, 3018 },
		{ 1152, 1135 },
		{ 980, 939 },
		{ 510, 456 },
		{ 983, 943 },
		{ 1072, 1038 },
		{ 611, 549 },
		{ 986, 946 },
		{ 389, 347 },
		{ 513, 458 },
		{ 1077, 1044 },
		{ 448, 405 },
		{ 253, 215 },
		{ 668, 603 },
		{ 771, 712 },
		{ 1064, 1030 },
		{ 1065, 1031 },
		{ 404, 362 },
		{ 862, 810 },
		{ 565, 505 },
		{ 1950, 1941 },
		{ 331, 286 },
		{ 1723, 1720 },
		{ 1873, 1854 },
		{ 808, 749 },
		{ 1096, 1068 },
		{ 1153, 1136 },
		{ 2125, 2110 },
		{ 1647, 1644 },
		{ 1324, 1321 },
		{ 1154, 1137 },
		{ 694, 631 },
		{ 500, 449 },
		{ 1932, 1919 },
		{ 1619, 1617 },
		{ 1673, 1670 },
		{ 911, 861 },
		{ 706, 643 },
		{ 1782, 1780 },
		{ 1697, 1694 },
		{ 2167, 2154 },
		{ 2063, 2042 },
		{ 998, 958 },
		{ 947, 901 },
		{ 829, 772 },
		{ 328, 283 },
		{ 1167, 1155 },
		{ 948, 901 },
		{ 698, 635 },
		{ 1140, 1121 },
		{ 1141, 1122 },
		{ 832, 775 },
		{ 687, 624 },
		{ 1211, 1204 },
		{ 481, 429 },
		{ 775, 716 },
		{ 381, 339 },
		{ 914, 864 },
		{ 1120, 1095 },
		{ 741, 684 },
		{ 939, 892 },
		{ 423, 381 },
		{ 284, 242 },
		{ 1101, 1073 },
		{ 728, 671 },
		{ 239, 204 },
		{ 1025, 990 },
		{ 822, 763 },
		{ 844, 791 },
		{ 371, 328 },
		{ 1133, 1113 },
		{ 1134, 1114 },
		{ 429, 386 },
		{ 413, 372 },
		{ 973, 929 },
		{ 890, 840 },
		{ 651, 587 },
		{ 892, 842 },
		{ 705, 642 },
		{ 979, 938 },
		{ 414, 373 },
		{ 1071, 1037 },
		{ 653, 589 },
		{ 982, 941 },
		{ 1891, 1873 },
		{ 351, 305 },
		{ 487, 435 },
		{ 985, 945 },
		{ 827, 768 },
		{ 446, 401 },
		{ 660, 595 },
		{ 270, 230 },
		{ 662, 597 },
		{ 906, 856 },
		{ 993, 953 },
		{ 907, 857 },
		{ 533, 473 },
		{ 193, 174 },
		{ 336, 291 },
		{ 450, 408 },
		{ 912, 862 },
		{ 420, 378 },
		{ 214, 185 },
		{ 780, 721 },
		{ 273, 233 },
		{ 2068, 2047 },
		{ 375, 332 },
		{ 784, 725 },
		{ 785, 726 },
		{ 724, 667 },
		{ 847, 795 },
		{ 238, 203 },
		{ 788, 729 },
		{ 1925, 1911 },
		{ 1014, 977 },
		{ 928, 878 },
		{ 791, 732 },
		{ 465, 415 },
		{ 2081, 2063 },
		{ 854, 802 },
		{ 1934, 1922 },
		{ 400, 358 },
		{ 1204, 1195 },
		{ 470, 417 },
		{ 342, 296 },
		{ 1839, 1815 },
		{ 1113, 1086 },
		{ 796, 737 },
		{ 860, 808 },
		{ 210, 183 },
		{ 326, 281 },
		{ 383, 341 },
		{ 297, 254 },
		{ 1029, 994 },
		{ 1033, 998 },
		{ 269, 229 },
		{ 867, 815 },
		{ 869, 818 },
		{ 635, 571 },
		{ 1220, 1218 },
		{ 872, 821 },
		{ 310, 266 },
		{ 1223, 1222 },
		{ 875, 824 },
		{ 312, 268 },
		{ 1228, 1227 },
		{ 743, 686 },
		{ 1230, 1229 },
		{ 559, 499 },
		{ 957, 910 },
		{ 1045, 1010 },
		{ 517, 461 },
		{ 2120, 2104 },
		{ 561, 501 },
		{ 1050, 1016 },
		{ 645, 581 },
		{ 299, 256 },
		{ 883, 832 },
		{ 2127, 2113 },
		{ 2129, 2115 },
		{ 2130, 2116 },
		{ 2131, 2117 },
		{ 2132, 2118 },
		{ 752, 694 },
		{ 411, 370 },
		{ 2027, 2003 },
		{ 390, 348 },
		{ 1876, 1857 },
		{ 970, 926 },
		{ 649, 585 },
		{ 1955, 1955 },
		{ 1955, 1955 },
		{ 2165, 2165 },
		{ 2165, 2165 },
		{ 1614, 1614 },
		{ 1614, 1614 },
		{ 1960, 1960 },
		{ 1960, 1960 },
		{ 2111, 2111 },
		{ 2111, 2111 },
		{ 2171, 2171 },
		{ 2171, 2171 },
		{ 1559, 1559 },
		{ 1559, 1559 },
		{ 1550, 1550 },
		{ 1550, 1550 },
		{ 1562, 1562 },
		{ 1562, 1562 },
		{ 1510, 1510 },
		{ 1510, 1510 },
		{ 1589, 1589 },
		{ 1589, 1589 },
		{ 567, 507 },
		{ 1955, 1955 },
		{ 925, 875 },
		{ 2165, 2165 },
		{ 926, 876 },
		{ 1614, 1614 },
		{ 249, 213 },
		{ 1960, 1960 },
		{ 977, 936 },
		{ 2111, 2111 },
		{ 275, 235 },
		{ 2171, 2171 },
		{ 549, 488 },
		{ 1559, 1559 },
		{ 316, 272 },
		{ 1550, 1550 },
		{ 293, 250 },
		{ 1562, 1562 },
		{ 2219, 2217 },
		{ 1510, 1510 },
		{ 311, 267 },
		{ 1589, 1589 },
		{ 1553, 1553 },
		{ 1553, 1553 },
		{ 1974, 1974 },
		{ 1974, 1974 },
		{ 1580, 1580 },
		{ 1580, 1580 },
		{ 1547, 1547 },
		{ 1547, 1547 },
		{ 1956, 1955 },
		{ 575, 515 },
		{ 2166, 2165 },
		{ 471, 418 },
		{ 1615, 1614 },
		{ 693, 630 },
		{ 1961, 1960 },
		{ 857, 805 },
		{ 2112, 2111 },
		{ 659, 594 },
		{ 2172, 2171 },
		{ 782, 723 },
		{ 1560, 1559 },
		{ 1086, 1054 },
		{ 1551, 1550 },
		{ 1553, 1553 },
		{ 1563, 1562 },
		{ 1974, 1974 },
		{ 1511, 1510 },
		{ 1580, 1580 },
		{ 1590, 1589 },
		{ 1547, 1547 },
		{ 2194, 2194 },
		{ 2194, 2194 },
		{ 1943, 1943 },
		{ 1943, 1943 },
		{ 1945, 1945 },
		{ 1945, 1945 },
		{ 1556, 1556 },
		{ 1556, 1556 },
		{ 2140, 2140 },
		{ 2140, 2140 },
		{ 2142, 2142 },
		{ 2142, 2142 },
		{ 2144, 2144 },
		{ 2144, 2144 },
		{ 2146, 2146 },
		{ 2146, 2146 },
		{ 2148, 2148 },
		{ 2148, 2148 },
		{ 2150, 2150 },
		{ 2150, 2150 },
		{ 1920, 1920 },
		{ 1920, 1920 },
		{ 1554, 1553 },
		{ 2194, 2194 },
		{ 1975, 1974 },
		{ 1943, 1943 },
		{ 1581, 1580 },
		{ 1945, 1945 },
		{ 1548, 1547 },
		{ 1556, 1556 },
		{ 764, 705 },
		{ 2140, 2140 },
		{ 410, 369 },
		{ 2142, 2142 },
		{ 680, 618 },
		{ 2144, 2144 },
		{ 1090, 1059 },
		{ 2146, 2146 },
		{ 1091, 1060 },
		{ 2148, 2148 },
		{ 546, 484 },
		{ 2150, 2150 },
		{ 994, 954 },
		{ 1920, 1920 },
		{ 657, 593 },
		{ 658, 593 },
		{ 1157, 1140 },
		{ 1030, 995 },
		{ 1159, 1143 },
		{ 1196, 1187 },
		{ 1031, 996 },
		{ 848, 796 },
		{ 2195, 2194 },
		{ 825, 766 },
		{ 1944, 1943 },
		{ 826, 767 },
		{ 1946, 1945 },
		{ 905, 855 },
		{ 1557, 1556 },
		{ 343, 297 },
		{ 2141, 2140 },
		{ 1063, 1029 },
		{ 2143, 2142 },
		{ 1139, 1120 },
		{ 2145, 2144 },
		{ 3104, 3102 },
		{ 2147, 2146 },
		{ 810, 751 },
		{ 2149, 2148 },
		{ 1330, 1329 },
		{ 2151, 2150 },
		{ 1541, 1523 },
		{ 1921, 1920 },
		{ 1948, 1937 },
		{ 1625, 1624 },
		{ 744, 687 },
		{ 1896, 1879 },
		{ 1115, 1089 },
		{ 959, 912 },
		{ 2089, 2072 },
		{ 1729, 1728 },
		{ 1172, 1160 },
		{ 1679, 1678 },
		{ 961, 914 },
		{ 1174, 1163 },
		{ 774, 715 },
		{ 2153, 2134 },
		{ 1788, 1787 },
		{ 195, 176 },
		{ 1070, 1036 },
		{ 746, 689 },
		{ 1179, 1168 },
		{ 377, 334 },
		{ 1181, 1170 },
		{ 1046, 1012 },
		{ 3074, 3072 },
		{ 800, 741 },
		{ 1916, 1901 },
		{ 967, 923 },
		{ 900, 850 },
		{ 874, 823 },
		{ 1189, 1180 },
		{ 1227, 1226 },
		{ 949, 902 },
		{ 1009, 969 },
		{ 1703, 1702 },
		{ 1653, 1652 },
		{ 2021, 1999 },
		{ 615, 554 },
		{ 1954, 1947 },
		{ 943, 896 },
		{ 223, 189 },
		{ 1835, 1811 },
		{ 1274, 1271 },
		{ 1935, 1923 },
		{ 2128, 2114 },
		{ 2914, 2912 },
		{ 1195, 1186 },
		{ 2164, 2152 },
		{ 846, 793 },
		{ 492, 440 },
		{ 2033, 2010 },
		{ 302, 259 },
		{ 2711, 2711 },
		{ 2711, 2711 },
		{ 1744, 1744 },
		{ 1744, 1744 },
		{ 2837, 2837 },
		{ 2837, 2837 },
		{ 3016, 3016 },
		{ 3016, 3016 },
		{ 1801, 1801 },
		{ 1801, 1801 },
		{ 2761, 2761 },
		{ 2761, 2761 },
		{ 2532, 2532 },
		{ 2532, 2532 },
		{ 2766, 2766 },
		{ 2766, 2766 },
		{ 2769, 2769 },
		{ 2769, 2769 },
		{ 1778, 1778 },
		{ 1778, 1778 },
		{ 1269, 1269 },
		{ 1269, 1269 },
		{ 396, 354 },
		{ 2711, 2711 },
		{ 531, 471 },
		{ 1744, 1744 },
		{ 227, 192 },
		{ 2837, 2837 },
		{ 915, 865 },
		{ 3016, 3016 },
		{ 318, 274 },
		{ 1801, 1801 },
		{ 476, 424 },
		{ 2761, 2761 },
		{ 804, 745 },
		{ 2532, 2532 },
		{ 257, 219 },
		{ 2766, 2766 },
		{ 754, 696 },
		{ 2769, 2769 },
		{ 1198, 1189 },
		{ 1778, 1778 },
		{ 755, 697 },
		{ 1269, 1269 },
		{ 380, 338 },
		{ 2654, 2654 },
		{ 2654, 2654 },
		{ 1692, 1692 },
		{ 1692, 1692 },
		{ 2731, 2711 },
		{ 666, 600 },
		{ 1745, 1744 },
		{ 989, 949 },
		{ 2838, 2837 },
		{ 445, 400 },
		{ 3017, 3016 },
		{ 709, 646 },
		{ 1802, 1801 },
		{ 268, 228 },
		{ 2775, 2761 },
		{ 927, 877 },
		{ 2563, 2532 },
		{ 868, 817 },
		{ 2777, 2766 },
		{ 711, 650 },
		{ 2778, 2769 },
		{ 870, 819 },
		{ 1779, 1778 },
		{ 2654, 2654 },
		{ 1270, 1269 },
		{ 1692, 1692 },
		{ 194, 175 },
		{ 2774, 2774 },
		{ 2774, 2774 },
		{ 2556, 2556 },
		{ 2556, 2556 },
		{ 2390, 2390 },
		{ 2390, 2390 },
		{ 2630, 2630 },
		{ 2630, 2630 },
		{ 2692, 2692 },
		{ 2692, 2692 },
		{ 2910, 2910 },
		{ 2910, 2910 },
		{ 2693, 2693 },
		{ 2693, 2693 },
		{ 2785, 2785 },
		{ 2785, 2785 },
		{ 2786, 2786 },
		{ 2786, 2786 },
		{ 2975, 2975 },
		{ 2975, 2975 },
		{ 2787, 2787 },
		{ 2787, 2787 },
		{ 2679, 2654 },
		{ 2774, 2774 },
		{ 1693, 1692 },
		{ 2556, 2556 },
		{ 670, 605 },
		{ 2390, 2390 },
		{ 2979, 2977 },
		{ 2630, 2630 },
		{ 569, 509 },
		{ 2692, 2692 },
		{ 335, 290 },
		{ 2910, 2910 },
		{ 673, 608 },
		{ 2693, 2693 },
		{ 674, 609 },
		{ 2785, 2785 },
		{ 2186, 2183 },
		{ 2786, 2786 },
		{ 675, 610 },
		{ 2975, 2975 },
		{ 636, 572 },
		{ 2787, 2787 },
		{ 1005, 965 },
		{ 1718, 1718 },
		{ 1718, 1718 },
		{ 2447, 2447 },
		{ 2447, 2447 },
		{ 2783, 2774 },
		{ 637, 573 },
		{ 2557, 2556 },
		{ 323, 278 },
		{ 2391, 2390 },
		{ 679, 617 },
		{ 2656, 2630 },
		{ 639, 575 },
		{ 2712, 2692 },
		{ 243, 208 },
		{ 2911, 2910 },
		{ 944, 898 },
		{ 2713, 2693 },
		{ 641, 577 },
		{ 2790, 2785 },
		{ 1226, 1225 },
		{ 2791, 2786 },
		{ 828, 771 },
		{ 2976, 2975 },
		{ 1718, 1718 },
		{ 2792, 2787 },
		{ 2447, 2447 },
		{ 886, 836 },
		{ 2500, 2500 },
		{ 2500, 2500 },
		{ 2732, 2732 },
		{ 2732, 2732 },
		{ 2795, 2795 },
		{ 2795, 2795 },
		{ 1304, 1304 },
		{ 1304, 1304 },
		{ 1347, 1347 },
		{ 1347, 1347 },
		{ 2503, 2503 },
		{ 2503, 2503 },
		{ 2240, 2240 },
		{ 2240, 2240 },
		{ 1642, 1642 },
		{ 1642, 1642 },
		{ 2927, 2927 },
		{ 2927, 2927 },
		{ 2616, 2616 },
		{ 2616, 2616 },
		{ 2703, 2703 },
		{ 2703, 2703 },
		{ 1719, 1718 },
		{ 2500, 2500 },
		{ 2448, 2447 },
		{ 2732, 2732 },
		{ 313, 269 },
		{ 2795, 2795 },
		{ 778, 719 },
		{ 1304, 1304 },
		{ 951, 904 },
		{ 1347, 1347 },
		{ 952, 905 },
		{ 2503, 2503 },
		{ 779, 720 },
		{ 2240, 2240 },
		{ 1020, 984 },
		{ 1642, 1642 },
		{ 428, 385 },
		{ 2927, 2927 },
		{ 369, 326 },
		{ 2616, 2616 },
		{ 314, 270 },
		{ 2703, 2703 },
		{ 688, 625 },
		{ 2804, 2804 },
		{ 2804, 2804 },
		{ 2743, 2743 },
		{ 2743, 2743 },
		{ 2530, 2500 },
		{ 607, 546 },
		{ 2748, 2732 },
		{ 1749, 1746 },
		{ 2797, 2795 },
		{ 355, 309 },
		{ 1305, 1304 },
		{ 1098, 1070 },
		{ 1348, 1347 },
		{ 960, 913 },
		{ 2533, 2503 },
		{ 896, 846 },
		{ 2241, 2240 },
		{ 2841, 2839 },
		{ 1643, 1642 },
		{ 838, 783 },
		{ 2928, 2927 },
		{ 550, 489 },
		{ 2642, 2616 },
		{ 2804, 2804 },
		{ 2723, 2703 },
		{ 2743, 2743 },
		{ 2932, 2929 },
		{ 2641, 2641 },
		{ 2641, 2641 },
		{ 1319, 1319 },
		{ 1319, 1319 },
		{ 1668, 1668 },
		{ 1668, 1668 },
		{ 2747, 2747 },
		{ 2747, 2747 },
		{ 2881, 2881 },
		{ 2881, 2881 },
		{ 2749, 2749 },
		{ 2749, 2749 },
		{ 2750, 2750 },
		{ 2750, 2750 },
		{ 2645, 2645 },
		{ 2645, 2645 },
		{ 2752, 2752 },
		{ 2752, 2752 },
		{ 2753, 2753 },
		{ 2753, 2753 },
		{ 2646, 2646 },
		{ 2646, 2646 },
		{ 2805, 2804 },
		{ 2641, 2641 },
		{ 2759, 2743 },
		{ 1319, 1319 },
		{ 490, 438 },
		{ 1668, 1668 },
		{ 1032, 997 },
		{ 2747, 2747 },
		{ 327, 282 },
		{ 2881, 2881 },
		{ 789, 730 },
		{ 2749, 2749 },
		{ 790, 731 },
		{ 2750, 2750 },
		{ 276, 236 },
		{ 2645, 2645 },
		{ 740, 683 },
		{ 2752, 2752 },
		{ 582, 522 },
		{ 2753, 2753 },
		{ 742, 685 },
		{ 2646, 2646 },
		{ 292, 249 },
		{ 2439, 2439 },
		{ 2439, 2439 },
		{ 2528, 2528 },
		{ 2528, 2528 },
		{ 2667, 2641 },
		{ 1183, 1172 },
		{ 1320, 1319 },
		{ 849, 797 },
		{ 1669, 1668 },
		{ 359, 315 },
		{ 2763, 2747 },
		{ 376, 333 },
		{ 2882, 2881 },
		{ 529, 469 },
		{ 2764, 2749 },
		{ 1188, 1179 },
		{ 2765, 2750 },
		{ 3135, 3135 },
		{ 2670, 2645 },
		{ 3143, 3143 },
		{ 2767, 2752 },
		{ 3147, 3147 },
		{ 2768, 2753 },
		{ 2439, 2439 },
		{ 2671, 2646 },
		{ 2528, 2528 },
		{ 2126, 2112 },
		{ 1591, 1581 },
		{ 1976, 1975 },
		{ 1962, 1956 },
		{ 1597, 1590 },
		{ 1573, 1551 },
		{ 2197, 2195 },
		{ 2173, 2166 },
		{ 1529, 1511 },
		{ 1965, 1961 },
		{ 1952, 1944 },
		{ 1577, 1563 },
		{ 2158, 2141 },
		{ 2178, 2172 },
		{ 3135, 3135 },
		{ 1953, 1946 },
		{ 3143, 3143 },
		{ 2159, 2143 },
		{ 3147, 3147 },
		{ 1575, 1557 },
		{ 2160, 2145 },
		{ 1572, 1548 },
		{ 2161, 2147 },
		{ 2469, 2439 },
		{ 1574, 1554 },
		{ 2560, 2528 },
		{ 2162, 2149 },
		{ 1576, 1560 },
		{ 2163, 2151 },
		{ 1933, 1921 },
		{ 1616, 1615 },
		{ 3048, 3047 },
		{ 1734, 1733 },
		{ 1735, 1734 },
		{ 1630, 1629 },
		{ 3109, 3108 },
		{ 3110, 3109 },
		{ 1631, 1630 },
		{ 1658, 1657 },
		{ 1793, 1792 },
		{ 3138, 3135 },
		{ 1794, 1793 },
		{ 3145, 3143 },
		{ 1659, 1658 },
		{ 3148, 3147 },
		{ 1708, 1707 },
		{ 1709, 1708 },
		{ 1336, 1335 },
		{ 1607, 1603 },
		{ 1684, 1683 },
		{ 1685, 1684 },
		{ 1335, 1334 },
		{ 1603, 1598 },
		{ 1604, 1599 },
		{ 1379, 1378 },
		{ 2811, 2811 },
		{ 2808, 2811 },
		{ 163, 163 },
		{ 160, 163 },
		{ 1905, 1890 },
		{ 1906, 1890 },
		{ 1927, 1915 },
		{ 1928, 1915 },
		{ 2207, 2207 },
		{ 1981, 1981 },
		{ 168, 164 },
		{ 2266, 2242 },
		{ 2810, 2806 },
		{ 88, 70 },
		{ 2212, 2208 },
		{ 167, 164 },
		{ 2265, 2242 },
		{ 2809, 2806 },
		{ 87, 70 },
		{ 2211, 2208 },
		{ 2326, 2295 },
		{ 2816, 2812 },
		{ 1986, 1982 },
		{ 2811, 2811 },
		{ 162, 158 },
		{ 163, 163 },
		{ 2815, 2812 },
		{ 1985, 1982 },
		{ 3059, 3051 },
		{ 161, 158 },
		{ 2058, 2037 },
		{ 2207, 2207 },
		{ 1981, 1981 },
		{ 169, 166 },
		{ 171, 170 },
		{ 119, 103 },
		{ 2817, 2814 },
		{ 2819, 2818 },
		{ 2812, 2811 },
		{ 1865, 1846 },
		{ 164, 163 },
		{ 1987, 1984 },
		{ 1989, 1988 },
		{ 2213, 2210 },
		{ 2215, 2214 },
		{ 1940, 1930 },
		{ 2208, 2207 },
		{ 1982, 1981 },
		{ 2202, 2201 },
		{ 2056, 2035 },
		{ 1988, 1986 },
		{ 2037, 2015 },
		{ 166, 162 },
		{ 2214, 2212 },
		{ 2295, 2266 },
		{ 1984, 1980 },
		{ 2814, 2810 },
		{ 103, 88 },
		{ 1846, 1824 },
		{ 170, 168 },
		{ 2818, 2816 },
		{ 0, 2858 },
		{ 2667, 2667 },
		{ 2667, 2667 },
		{ 0, 3038 },
		{ 1989, 1989 },
		{ 1990, 1989 },
		{ 0, 1311 },
		{ 2783, 2783 },
		{ 2783, 2783 },
		{ 0, 2947 },
		{ 2670, 2670 },
		{ 2670, 2670 },
		{ 2671, 2671 },
		{ 2671, 2671 },
		{ 0, 2950 },
		{ 0, 2865 },
		{ 0, 2628 },
		{ 0, 2721 },
		{ 0, 1769 },
		{ 0, 2955 },
		{ 2723, 2723 },
		{ 2723, 2723 },
		{ 2790, 2790 },
		{ 2790, 2790 },
		{ 2667, 2667 },
		{ 2791, 2791 },
		{ 2791, 2791 },
		{ 1989, 1989 },
		{ 0, 2872 },
		{ 0, 1279 },
		{ 2783, 2783 },
		{ 2792, 2792 },
		{ 2792, 2792 },
		{ 2670, 2670 },
		{ 0, 2229 },
		{ 2671, 2671 },
		{ 2557, 2557 },
		{ 2557, 2557 },
		{ 2530, 2530 },
		{ 2530, 2530 },
		{ 2797, 2797 },
		{ 2797, 2797 },
		{ 0, 2966 },
		{ 2723, 2723 },
		{ 0, 2634 },
		{ 2790, 2790 },
		{ 0, 2458 },
		{ 0, 2730 },
		{ 2791, 2791 },
		{ 2731, 2731 },
		{ 2731, 2731 },
		{ 2563, 2563 },
		{ 2563, 2563 },
		{ 0, 2733 },
		{ 2792, 2792 },
		{ 0, 2682 },
		{ 2805, 2805 },
		{ 2805, 2805 },
		{ 0, 2736 },
		{ 2557, 2557 },
		{ 0, 2737 },
		{ 2530, 2530 },
		{ 0, 2459 },
		{ 2797, 2797 },
		{ 2533, 2533 },
		{ 2533, 2533 },
		{ 0, 2980 },
		{ 0, 2685 },
		{ 0, 2893 },
		{ 0, 2600 },
		{ 0, 2984 },
		{ 0, 2601 },
		{ 2731, 2731 },
		{ 0, 2460 },
		{ 2563, 2563 },
		{ 0, 2988 },
		{ 2642, 2642 },
		{ 2642, 2642 },
		{ 0, 2991 },
		{ 2805, 2805 },
		{ 2819, 2819 },
		{ 2820, 2819 },
		{ 0, 2508 },
		{ 0, 2902 },
		{ 0, 1753 },
		{ 0, 2996 },
		{ 0, 2827 },
		{ 2533, 2533 },
		{ 2215, 2215 },
		{ 2216, 2215 },
		{ 2748, 2748 },
		{ 2748, 2748 },
		{ 0, 2571 },
		{ 0, 2358 },
		{ 2391, 2391 },
		{ 2391, 2391 },
		{ 0, 2610 },
		{ 0, 2426 },
		{ 0, 1256 },
		{ 2642, 2642 },
		{ 0, 2893 },
		{ 0, 3007 },
		{ 0, 2470 },
		{ 2819, 2819 },
		{ 171, 171 },
		{ 172, 171 },
		{ 2469, 2469 },
		{ 2469, 2469 },
		{ 2656, 2656 },
		{ 2656, 2656 },
		{ 0, 2470 },
		{ 2215, 2215 },
		{ 0, 2918 },
		{ 2748, 2748 },
		{ 2759, 2759 },
		{ 2759, 2759 },
		{ 0, 2579 },
		{ 2391, 2391 },
		{ 0, 1764 },
		{ 0, 2842 },
		{ 0, 2520 },
		{ 2765, 2765 },
		{ 2765, 2765 },
		{ 0, 1295 },
		{ 0, 2358 },
		{ 0, 1290 },
		{ 0, 2847 },
		{ 171, 171 },
		{ 0, 2585 },
		{ 2469, 2469 },
		{ 0, 2498 },
		{ 2656, 2656 },
		{ 0, 2933 },
		{ 0, 3027 },
		{ 2713, 2713 },
		{ 2713, 2713 },
		{ 0, 2937 },
		{ 2759, 2759 },
		{ 2775, 2775 },
		{ 2775, 2775 },
		{ 0, 2776 },
		{ 2777, 2777 },
		{ 2777, 2777 },
		{ 0, 2588 },
		{ 2765, 2765 },
		{ 2778, 2778 },
		{ 2778, 2778 },
		{ 1524, 1524 },
		{ 1235, 1235 },
		{ 2226, 2225 },
		{ 1366, 1365 },
		{ 2813, 2809 },
		{ 3063, 3059 },
		{ 1983, 1985 },
		{ 0, 87 },
		{ 1983, 1979 },
		{ 2209, 2211 },
		{ 2713, 2713 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2775, 2775 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2777, 2777 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 2778, 2778 },
		{ 0, 0 },
		{ 1524, 1524 },
		{ 1235, 1235 }
	};
	yytransition = transition;

	static const yystate_t YYNEARFAR YYBASED_CODE state[] = {
		{ 0, 0, 0 },
		{ -68, 15, 0 },
		{ 1, 0, 0 },
		{ 4, 12, 0 },
		{ 44, 572, 0 },
		{ -177, 2840, 0 },
		{ 5, 0, 0 },
		{ -1234, 1017, -31 },
		{ 7, 0, -31 },
		{ -1238, 1853, -33 },
		{ 9, 0, -33 },
		{ -1251, 3156, 147 },
		{ 11, 0, 147 },
		{ -1272, 3177, 155 },
		{ 13, 0, 155 },
		{ -1307, 3295, 0 },
		{ 15, 0, 0 },
		{ -1322, 3167, 143 },
		{ 17, 0, 143 },
		{ -1350, 3157, 22 },
		{ 19, 0, 22 },
		{ -1392, 230, 0 },
		{ 21, 0, 0 },
		{ -1618, 3296, 0 },
		{ 23, 0, 0 },
		{ -1645, 3181, 0 },
		{ 25, 0, 0 },
		{ -1671, 3172, 0 },
		{ 27, 0, 0 },
		{ -1695, 3163, 0 },
		{ 29, 0, 0 },
		{ -1721, 3155, 0 },
		{ 31, 0, 0 },
		{ -1747, 3162, 159 },
		{ 33, 0, 159 },
		{ -1781, 3300, 266 },
		{ 35, 0, 266 },
		{ 38, 129, 0 },
		{ -1817, 344, 0 },
		{ 40, 127, 0 },
		{ -2006, 116, 0 },
		{ -2218, 3293, 0 },
		{ 41, 0, 0 },
		{ 44, 14, 0 },
		{ -86, 458, 0 },
		{ -2822, 3161, 151 },
		{ 45, 0, 151 },
		{ -2840, 3294, 174 },
		{ 47, 0, 174 },
		{ 2884, 1434, 0 },
		{ 49, 0, 0 },
		{ -2886, 3297, 272 },
		{ 51, 0, 272 },
		{ -2913, 3298, 177 },
		{ 53, 0, 177 },
		{ -2931, 3171, 170 },
		{ 55, 0, 170 },
		{ -2978, 3303, 163 },
		{ 57, 0, 163 },
		{ -3019, 3179, 169 },
		{ 59, 0, 169 },
		{ 44, 1, 0 },
		{ 61, 0, 0 },
		{ -3070, 1813, 0 },
		{ 63, 0, 0 },
		{ -3080, 1722, 44 },
		{ 65, 0, 44 },
		{ 0, 0, 1 },
		{ 0, 0, 2 },
		{ 0, 0, 426 },
		{ 3051, 4751, 433 },
		{ 0, 0, 244 },
		{ 0, 0, 246 },
		{ 157, 1219, 263 },
		{ 157, 1348, 263 },
		{ 157, 1247, 263 },
		{ 157, 1254, 263 },
		{ 157, 1254, 263 },
		{ 157, 1258, 263 },
		{ 157, 1263, 263 },
		{ 157, 1257, 263 },
		{ 3129, 2927, 433 },
		{ 157, 1268, 263 },
		{ 3129, 1667, 262 },
		{ 102, 2622, 433 },
		{ 157, 0, 263 },
		{ 0, 0, 433 },
		{ -87, 4985, 240 },
		{ -88, 4795, 0 },
		{ 157, 1285, 263 },
		{ 157, 750, 263 },
		{ 157, 719, 263 },
		{ 157, 733, 263 },
		{ 157, 716, 263 },
		{ 157, 752, 263 },
		{ 157, 759, 263 },
		{ 157, 774, 263 },
		{ 157, 767, 263 },
		{ 3098, 2279, 0 },
		{ 157, 757, 263 },
		{ 3129, 1822, 259 },
		{ 117, 1386, 0 },
		{ 3129, 1799, 260 },
		{ 3051, 4768, 0 },
		{ 157, 766, 263 },
		{ 157, 764, 263 },
		{ 157, 791, 263 },
		{ 157, 787, 263 },
		{ 157, 0, 251 },
		{ 157, 806, 263 },
		{ 157, 808, 263 },
		{ 157, 793, 263 },
		{ 157, 826, 263 },
		{ 3126, 2944, 0 },
		{ 157, 833, 263 },
		{ 131, 1433, 0 },
		{ 117, 0, 0 },
		{ 3053, 2664, 261 },
		{ 133, 1354, 0 },
		{ 0, 0, 242 },
		{ 157, 837, 247 },
		{ 157, 839, 263 },
		{ 157, 831, 263 },
		{ 157, 836, 263 },
		{ 157, 834, 263 },
		{ 157, 827, 263 },
		{ 157, 0, 254 },
		{ 157, 828, 263 },
		{ 0, 0, 256 },
		{ 157, 834, 263 },
		{ 131, 0, 0 },
		{ 3053, 2688, 259 },
		{ 133, 0, 0 },
		{ 3053, 2596, 260 },
		{ 157, 849, 263 },
		{ 157, 846, 263 },
		{ 157, 848, 263 },
		{ 157, 878, 263 },
		{ 157, 963, 263 },
		{ 157, 0, 253 },
		{ 157, 1052, 263 },
		{ 157, 1125, 263 },
		{ 157, 1118, 263 },
		{ 157, 0, 249 },
		{ 157, 1193, 263 },
		{ 157, 0, 250 },
		{ 157, 0, 252 },
		{ 157, 1191, 263 },
		{ 157, 1199, 263 },
		{ 157, 0, 248 },
		{ 157, 1219, 263 },
		{ 157, 0, 255 },
		{ 157, 769, 263 },
		{ 157, 1247, 263 },
		{ 0, 0, 258 },
		{ 157, 1231, 263 },
		{ 157, 1235, 263 },
		{ 3146, 1305, 257 },
		{ 3051, 4762, 433 },
		{ 163, 0, 244 },
		{ 0, 0, 245 },
		{ -161, 16, 240 },
		{ -162, 4790, 0 },
		{ 3101, 4773, 0 },
		{ 3051, 4748, 0 },
		{ 0, 0, 241 },
		{ 3051, 4766, 0 },
		{ -167, 19, 0 },
		{ -168, 4797, 0 },
		{ 171, 0, 242 },
		{ 3051, 4767, 0 },
		{ 3101, 4936, 0 },
		{ 0, 0, 243 },
		{ 3102, 1602, 141 },
		{ 2118, 4099, 141 },
		{ 2977, 4444, 141 },
		{ 3102, 4336, 141 },
		{ 0, 0, 141 },
		{ 3090, 3411, 0 },
		{ 2133, 3013, 0 },
		{ 3090, 3627, 0 },
		{ 3090, 3286, 0 },
		{ 3067, 3362, 0 },
		{ 2118, 4131, 0 },
		{ 3094, 3274, 0 },
		{ 2118, 4104, 0 },
		{ 3068, 3542, 0 },
		{ 3097, 3252, 0 },
		{ 3068, 3322, 0 },
		{ 2912, 4360, 0 },
		{ 3094, 3295, 0 },
		{ 2133, 3740, 0 },
		{ 2977, 4399, 0 },
		{ 2064, 3483, 0 },
		{ 2064, 3495, 0 },
		{ 2086, 3123, 0 },
		{ 2067, 3832, 0 },
		{ 2048, 3891, 0 },
		{ 2067, 3844, 0 },
		{ 2086, 3128, 0 },
		{ 2086, 3151, 0 },
		{ 3094, 3339, 0 },
		{ 3018, 3984, 0 },
		{ 2118, 4113, 0 },
		{ 1204, 4066, 0 },
		{ 2064, 3472, 0 },
		{ 2086, 3166, 0 },
		{ 2086, 3170, 0 },
		{ 2977, 4503, 0 },
		{ 2064, 3512, 0 },
		{ 3067, 3795, 0 },
		{ 3090, 3629, 0 },
		{ 2133, 3707, 0 },
		{ 2217, 4201, 0 },
		{ 2977, 3645, 0 },
		{ 3018, 4010, 0 },
		{ 2048, 3862, 0 },
		{ 3068, 3545, 0 },
		{ 2026, 3298, 0 },
		{ 2977, 4409, 0 },
		{ 3090, 3694, 0 },
		{ 3018, 3644, 0 },
		{ 2133, 3702, 0 },
		{ 2086, 3185, 0 },
		{ 3097, 3430, 0 },
		{ 3067, 3705, 0 },
		{ 2026, 3290, 0 },
		{ 2048, 3911, 0 },
		{ 2977, 4431, 0 },
		{ 2118, 4137, 0 },
		{ 2118, 4093, 0 },
		{ 3090, 3663, 0 },
		{ 2086, 3102, 0 },
		{ 2118, 4106, 0 },
		{ 3090, 3639, 0 },
		{ 2217, 4205, 0 },
		{ 2977, 4625, 0 },
		{ 3097, 3431, 0 },
		{ 3068, 3593, 0 },
		{ 2086, 3103, 0 },
		{ 3097, 3334, 0 },
		{ 3090, 3616, 0 },
		{ 1204, 4063, 0 },
		{ 2048, 3893, 0 },
		{ 3018, 3974, 0 },
		{ 2133, 3646, 0 },
		{ 1091, 3256, 0 },
		{ 3090, 3672, 0 },
		{ 3067, 3796, 0 },
		{ 2977, 4633, 0 },
		{ 2217, 4211, 0 },
		{ 2086, 3104, 0 },
		{ 2133, 3731, 0 },
		{ 3097, 3404, 0 },
		{ 2118, 4134, 0 },
		{ 2026, 3307, 0 },
		{ 2118, 4158, 0 },
		{ 3068, 3604, 0 },
		{ 2086, 3105, 0 },
		{ 2912, 4371, 0 },
		{ 3067, 3793, 0 },
		{ 3097, 3438, 0 },
		{ 2154, 3641, 0 },
		{ 2086, 3106, 0 },
		{ 3018, 3952, 0 },
		{ 3068, 3597, 0 },
		{ 2118, 4143, 0 },
		{ 2217, 4215, 0 },
		{ 2118, 4146, 0 },
		{ 2977, 4543, 0 },
		{ 2977, 4559, 0 },
		{ 2048, 3892, 0 },
		{ 2217, 4209, 0 },
		{ 2086, 3107, 0 },
		{ 2977, 4403, 0 },
		{ 3018, 3940, 0 },
		{ 2048, 3896, 0 },
		{ 3018, 3925, 0 },
		{ 2977, 4497, 0 },
		{ 2064, 3469, 0 },
		{ 3068, 3543, 0 },
		{ 2118, 4132, 0 },
		{ 2977, 4619, 0 },
		{ 1204, 4046, 0 },
		{ 3018, 3934, 0 },
		{ 1091, 3257, 0 },
		{ 2154, 4021, 0 },
		{ 2067, 3834, 0 },
		{ 3068, 3581, 0 },
		{ 2086, 3109, 0 },
		{ 2977, 4477, 0 },
		{ 2118, 4100, 0 },
		{ 2086, 3110, 0 },
		{ 0, 0, 72 },
		{ 3090, 3387, 0 },
		{ 3097, 3374, 0 },
		{ 2118, 4126, 0 },
		{ 3102, 4307, 0 },
		{ 2086, 3111, 0 },
		{ 2086, 3112, 0 },
		{ 3097, 3405, 0 },
		{ 2064, 3502, 0 },
		{ 2048, 3860, 0 },
		{ 3097, 3406, 0 },
		{ 2086, 3113, 0 },
		{ 2118, 4087, 0 },
		{ 3090, 3681, 0 },
		{ 3090, 3689, 0 },
		{ 2067, 3827, 0 },
		{ 2977, 4571, 0 },
		{ 3068, 3522, 0 },
		{ 859, 3273, 0 },
		{ 0, 0, 4 },
		{ 0, 0, 5 },
		{ 2064, 3473, 0 },
		{ 2977, 4643, 0 },
		{ 3018, 3966, 0 },
		{ 2048, 3865, 0 },
		{ 3097, 3432, 0 },
		{ 3068, 3554, 0 },
		{ 0, 0, 70 },
		{ 2086, 3116, 0 },
		{ 2064, 3496, 0 },
		{ 3097, 3441, 0 },
		{ 3018, 3963, 0 },
		{ 3097, 3362, 0 },
		{ 2977, 4557, 0 },
		{ 3068, 3609, 0 },
		{ 1204, 4070, 0 },
		{ 2048, 3863, 0 },
		{ 2064, 3520, 0 },
		{ 3067, 3815, 0 },
		{ 2118, 4108, 0 },
		{ 2977, 4645, 0 },
		{ 3102, 4340, 0 },
		{ 3068, 3533, 0 },
		{ 0, 0, 64 },
		{ 3068, 3539, 0 },
		{ 2977, 4417, 0 },
		{ 1204, 4057, 0 },
		{ 3018, 3959, 0 },
		{ 2118, 4133, 0 },
		{ 0, 0, 75 },
		{ 3097, 3369, 0 },
		{ 3090, 3614, 0 },
		{ 3090, 3665, 0 },
		{ 2086, 3117, 0 },
		{ 3018, 4006, 0 },
		{ 2118, 4168, 0 },
		{ 2086, 3118, 0 },
		{ 2064, 3475, 0 },
		{ 3067, 3766, 0 },
		{ 3097, 3376, 0 },
		{ 3068, 3588, 0 },
		{ 2977, 4395, 0 },
		{ 3097, 3384, 0 },
		{ 2086, 3119, 0 },
		{ 3018, 3971, 0 },
		{ 2118, 4123, 0 },
		{ 3097, 3402, 0 },
		{ 3068, 3605, 0 },
		{ 3018, 3994, 0 },
		{ 1031, 4016, 0 },
		{ 0, 0, 8 },
		{ 2064, 3506, 0 },
		{ 2067, 3831, 0 },
		{ 3018, 3932, 0 },
		{ 2099, 3239, 0 },
		{ 2086, 3120, 0 },
		{ 2217, 4279, 0 },
		{ 2118, 4166, 0 },
		{ 2064, 3454, 0 },
		{ 2118, 4075, 0 },
		{ 2118, 4082, 0 },
		{ 2067, 3826, 0 },
		{ 2086, 3121, 0 },
		{ 3097, 3423, 0 },
		{ 3097, 3288, 0 },
		{ 2118, 4103, 0 },
		{ 3094, 3342, 0 },
		{ 3068, 3577, 0 },
		{ 1204, 4062, 0 },
		{ 3067, 3785, 0 },
		{ 2086, 3122, 0 },
		{ 2133, 3084, 0 },
		{ 2977, 4555, 0 },
		{ 1204, 4073, 0 },
		{ 2133, 3750, 0 },
		{ 3102, 3245, 0 },
		{ 2099, 3243, 0 },
		{ 2067, 3845, 0 },
		{ 2064, 3503, 0 },
		{ 3097, 3368, 0 },
		{ 0, 0, 116 },
		{ 2086, 3126, 0 },
		{ 2133, 3753, 0 },
		{ 2025, 3267, 0 },
		{ 3018, 3981, 0 },
		{ 3090, 3659, 0 },
		{ 3067, 3767, 0 },
		{ 2977, 4427, 0 },
		{ 2118, 4091, 0 },
		{ 0, 0, 7 },
		{ 2067, 3824, 0 },
		{ 0, 0, 6 },
		{ 3018, 4009, 0 },
		{ 0, 0, 121 },
		{ 3067, 3773, 0 },
		{ 2118, 4101, 0 },
		{ 3102, 1660, 0 },
		{ 2086, 3129, 0 },
		{ 3067, 3794, 0 },
		{ 3090, 3670, 0 },
		{ 0, 0, 125 },
		{ 2086, 3130, 0 },
		{ 2118, 4119, 0 },
		{ 3102, 3256, 0 },
		{ 2118, 4125, 0 },
		{ 2217, 4228, 0 },
		{ 2133, 3706, 0 },
		{ 0, 0, 71 },
		{ 2048, 3858, 0 },
		{ 2086, 3131, 108 },
		{ 2086, 3132, 109 },
		{ 2977, 4405, 0 },
		{ 3018, 3982, 0 },
		{ 3068, 3572, 0 },
		{ 3090, 3652, 0 },
		{ 3068, 3575, 0 },
		{ 1204, 4055, 0 },
		{ 3090, 3662, 0 },
		{ 3068, 3576, 0 },
		{ 3094, 3350, 0 },
		{ 2133, 3712, 0 },
		{ 3018, 3935, 0 },
		{ 2118, 4088, 0 },
		{ 2086, 3133, 0 },
		{ 3097, 3419, 0 },
		{ 2977, 4615, 0 },
		{ 3018, 3954, 0 },
		{ 2912, 4369, 0 },
		{ 3018, 3958, 0 },
		{ 2133, 3748, 0 },
		{ 3068, 3592, 0 },
		{ 3018, 3965, 0 },
		{ 0, 0, 9 },
		{ 2086, 3134, 0 },
		{ 3018, 3968, 0 },
		{ 2099, 3240, 0 },
		{ 2154, 4032, 0 },
		{ 0, 0, 106 },
		{ 2064, 3511, 0 },
		{ 3067, 3812, 0 },
		{ 3090, 3631, 0 },
		{ 2133, 3645, 0 },
		{ 3067, 3262, 0 },
		{ 3018, 4001, 0 },
		{ 3094, 3309, 0 },
		{ 3018, 4007, 0 },
		{ 3094, 3293, 0 },
		{ 3090, 3668, 0 },
		{ 2118, 4153, 0 },
		{ 3097, 3439, 0 },
		{ 3068, 3535, 0 },
		{ 3094, 3276, 0 },
		{ 3067, 3804, 0 },
		{ 3097, 3447, 0 },
		{ 3018, 3924, 0 },
		{ 3097, 3336, 0 },
		{ 2977, 4647, 0 },
		{ 2086, 3135, 0 },
		{ 2977, 4397, 0 },
		{ 3068, 3548, 0 },
		{ 2118, 4098, 0 },
		{ 3090, 3622, 0 },
		{ 3090, 3624, 0 },
		{ 2048, 3864, 0 },
		{ 2064, 3491, 0 },
		{ 2086, 3136, 96 },
		{ 3068, 3573, 0 },
		{ 2086, 3138, 0 },
		{ 2086, 3140, 0 },
		{ 3094, 3336, 0 },
		{ 2133, 3735, 0 },
		{ 2217, 4287, 0 },
		{ 2086, 3141, 0 },
		{ 2064, 3509, 0 },
		{ 0, 0, 105 },
		{ 2217, 4207, 0 },
		{ 2977, 4583, 0 },
		{ 3097, 3387, 0 },
		{ 3097, 3390, 0 },
		{ 0, 0, 118 },
		{ 0, 0, 120 },
		{ 3067, 3791, 0 },
		{ 2133, 3757, 0 },
		{ 2064, 3519, 0 },
		{ 2118, 3372, 0 },
		{ 3097, 3399, 0 },
		{ 2118, 4150, 0 },
		{ 2086, 3142, 0 },
		{ 2118, 4155, 0 },
		{ 3018, 3955, 0 },
		{ 3067, 3809, 0 },
		{ 2086, 3143, 0 },
		{ 2154, 4019, 0 },
		{ 3094, 3351, 0 },
		{ 2217, 4195, 0 },
		{ 2086, 3144, 0 },
		{ 2977, 4475, 0 },
		{ 2064, 3476, 0 },
		{ 959, 3916, 0 },
		{ 3097, 3416, 0 },
		{ 3067, 3783, 0 },
		{ 2133, 3751, 0 },
		{ 2217, 4226, 0 },
		{ 3097, 3418, 0 },
		{ 2026, 3313, 0 },
		{ 2086, 3145, 0 },
		{ 3018, 3995, 0 },
		{ 3090, 3655, 0 },
		{ 2064, 3497, 0 },
		{ 2977, 4629, 0 },
		{ 3097, 3429, 0 },
		{ 2133, 3714, 0 },
		{ 2099, 3238, 0 },
		{ 3068, 3574, 0 },
		{ 2064, 3505, 0 },
		{ 2133, 3743, 0 },
		{ 2067, 3840, 0 },
		{ 3102, 3329, 0 },
		{ 2086, 3146, 0 },
		{ 0, 0, 65 },
		{ 2086, 3147, 0 },
		{ 3090, 3688, 0 },
		{ 3068, 3582, 0 },
		{ 3090, 3693, 0 },
		{ 3068, 3585, 0 },
		{ 2086, 3149, 110 },
		{ 2048, 3877, 0 },
		{ 2133, 3709, 0 },
		{ 2067, 3842, 0 },
		{ 2064, 3514, 0 },
		{ 2064, 3518, 0 },
		{ 2048, 3904, 0 },
		{ 2067, 3848, 0 },
		{ 2977, 4567, 0 },
		{ 3090, 3628, 0 },
		{ 3094, 3345, 0 },
		{ 3018, 4004, 0 },
		{ 3097, 3442, 0 },
		{ 2064, 3453, 0 },
		{ 0, 0, 122 },
		{ 2086, 3150, 0 },
		{ 2912, 4357, 0 },
		{ 2067, 3833, 0 },
		{ 3097, 3358, 0 },
		{ 3067, 3786, 0 },
		{ 0, 0, 117 },
		{ 0, 0, 107 },
		{ 2064, 3470, 0 },
		{ 3068, 3534, 0 },
		{ 3097, 3360, 0 },
		{ 2133, 3001, 0 },
		{ 3102, 3291, 0 },
		{ 3018, 3957, 0 },
		{ 3067, 3801, 0 },
		{ 2086, 3155, 0 },
		{ 3018, 3960, 0 },
		{ 2048, 3861, 0 },
		{ 3090, 3638, 0 },
		{ 2118, 4140, 0 },
		{ 2977, 4487, 0 },
		{ 2977, 4495, 0 },
		{ 2064, 3481, 0 },
		{ 2977, 4501, 0 },
		{ 3018, 3967, 0 },
		{ 2977, 4507, 0 },
		{ 3068, 3546, 0 },
		{ 3067, 3819, 0 },
		{ 2086, 3156, 0 },
		{ 2118, 4157, 0 },
		{ 3068, 3553, 0 },
		{ 2086, 3157, 0 },
		{ 3068, 3561, 0 },
		{ 2118, 4171, 0 },
		{ 3018, 3989, 0 },
		{ 2118, 4078, 0 },
		{ 3068, 3570, 0 },
		{ 2118, 4084, 0 },
		{ 2064, 3494, 0 },
		{ 3067, 3790, 0 },
		{ 2086, 3161, 0 },
		{ 3102, 4293, 0 },
		{ 2217, 4234, 0 },
		{ 2118, 4092, 0 },
		{ 2067, 3829, 0 },
		{ 2118, 4094, 0 },
		{ 2067, 3830, 0 },
		{ 3090, 3619, 0 },
		{ 2977, 4423, 0 },
		{ 3090, 3674, 0 },
		{ 0, 0, 98 },
		{ 3018, 4011, 0 },
		{ 3018, 3922, 0 },
		{ 2977, 4471, 0 },
		{ 2086, 3162, 0 },
		{ 2086, 3163, 0 },
		{ 2977, 4479, 0 },
		{ 2977, 4481, 0 },
		{ 2977, 4485, 0 },
		{ 2067, 3836, 0 },
		{ 2064, 3499, 0 },
		{ 0, 0, 128 },
		{ 3090, 3692, 0 },
		{ 0, 0, 119 },
		{ 0, 0, 123 },
		{ 2977, 4499, 0 },
		{ 2217, 4281, 0 },
		{ 1091, 3265, 0 },
		{ 2086, 3165, 0 },
		{ 3018, 3091, 0 },
		{ 3068, 3584, 0 },
		{ 2067, 3825, 0 },
		{ 1204, 4053, 0 },
		{ 2977, 4561, 0 },
		{ 3090, 3641, 0 },
		{ 3090, 3647, 0 },
		{ 3090, 3648, 0 },
		{ 3067, 3780, 0 },
		{ 2217, 4230, 0 },
		{ 2154, 4031, 0 },
		{ 3067, 3782, 0 },
		{ 3094, 3348, 0 },
		{ 2048, 3866, 0 },
		{ 1204, 4049, 0 },
		{ 3097, 3403, 0 },
		{ 2048, 3889, 0 },
		{ 2064, 3510, 0 },
		{ 2086, 3167, 0 },
		{ 2067, 3837, 0 },
		{ 2048, 3895, 0 },
		{ 2118, 4080, 0 },
		{ 2154, 4037, 0 },
		{ 2133, 3701, 0 },
		{ 3068, 3595, 0 },
		{ 2977, 4429, 0 },
		{ 2133, 3705, 0 },
		{ 0, 0, 59 },
		{ 0, 0, 60 },
		{ 2977, 4437, 0 },
		{ 3068, 3596, 0 },
		{ 0, 0, 69 },
		{ 0, 0, 114 },
		{ 2026, 3295, 0 },
		{ 3094, 3353, 0 },
		{ 2064, 3517, 0 },
		{ 3094, 3324, 0 },
		{ 3097, 3417, 0 },
		{ 3018, 3931, 0 },
		{ 3068, 3523, 0 },
		{ 0, 0, 100 },
		{ 3068, 3524, 0 },
		{ 0, 0, 102 },
		{ 3090, 3690, 0 },
		{ 3068, 3531, 0 },
		{ 3067, 3772, 0 },
		{ 2118, 4111, 0 },
		{ 2099, 3245, 0 },
		{ 2099, 3237, 0 },
		{ 3097, 3420, 0 },
		{ 1204, 4065, 0 },
		{ 2154, 3769, 0 },
		{ 3068, 3536, 0 },
		{ 959, 3913, 0 },
		{ 2048, 3908, 0 },
		{ 0, 0, 115 },
		{ 0, 0, 127 },
		{ 3068, 3537, 0 },
		{ 3068, 3538, 0 },
		{ 0, 0, 140 },
		{ 2064, 3455, 0 },
		{ 3102, 3932, 0 },
		{ 2977, 4627, 0 },
		{ 1204, 4060, 0 },
		{ 2977, 4631, 0 },
		{ 2118, 4148, 0 },
		{ 3102, 4323, 0 },
		{ 3068, 3541, 0 },
		{ 3102, 4338, 0 },
		{ 3094, 3340, 0 },
		{ 3094, 3341, 0 },
		{ 3067, 3083, 0 },
		{ 2086, 3171, 0 },
		{ 2118, 4165, 0 },
		{ 3018, 3986, 0 },
		{ 2977, 4411, 0 },
		{ 2977, 4415, 0 },
		{ 3018, 3988, 0 },
		{ 2133, 3728, 0 },
		{ 3018, 3990, 0 },
		{ 2133, 3730, 0 },
		{ 2133, 3647, 0 },
		{ 3018, 3997, 0 },
		{ 2086, 3172, 0 },
		{ 2217, 4277, 0 },
		{ 2086, 3173, 0 },
		{ 3090, 3671, 0 },
		{ 2086, 3175, 0 },
		{ 2067, 3838, 0 },
		{ 3090, 3673, 0 },
		{ 2048, 3906, 0 },
		{ 3018, 4012, 0 },
		{ 2086, 3177, 0 },
		{ 3090, 3678, 0 },
		{ 3102, 4333, 0 },
		{ 1204, 4056, 0 },
		{ 2133, 3754, 0 },
		{ 3068, 3562, 0 },
		{ 2977, 4545, 0 },
		{ 2977, 4551, 0 },
		{ 2118, 4105, 0 },
		{ 2067, 3849, 0 },
		{ 2217, 4236, 0 },
		{ 3068, 3569, 0 },
		{ 2118, 4109, 0 },
		{ 2118, 4110, 0 },
		{ 3018, 3946, 0 },
		{ 3018, 3947, 0 },
		{ 2118, 4114, 0 },
		{ 2977, 4621, 0 },
		{ 2977, 4623, 0 },
		{ 2118, 4118, 0 },
		{ 2086, 3179, 0 },
		{ 2133, 3703, 0 },
		{ 3097, 3443, 0 },
		{ 3097, 3444, 0 },
		{ 2118, 4129, 0 },
		{ 2048, 3874, 0 },
		{ 3094, 3332, 0 },
		{ 2048, 3886, 0 },
		{ 3102, 4344, 0 },
		{ 3097, 3449, 0 },
		{ 2086, 3180, 62 },
		{ 3097, 3359, 0 },
		{ 2977, 4407, 0 },
		{ 2133, 3725, 0 },
		{ 2086, 3181, 0 },
		{ 2086, 3182, 0 },
		{ 2154, 4024, 0 },
		{ 3018, 3972, 0 },
		{ 3102, 4315, 0 },
		{ 3067, 3813, 0 },
		{ 3097, 3363, 0 },
		{ 3097, 3364, 0 },
		{ 1091, 3255, 0 },
		{ 2048, 3853, 0 },
		{ 3068, 3590, 0 },
		{ 2099, 3242, 0 },
		{ 2026, 3296, 0 },
		{ 2026, 3297, 0 },
		{ 3090, 3666, 0 },
		{ 2064, 3515, 0 },
		{ 1204, 4068, 0 },
		{ 3094, 3349, 0 },
		{ 3068, 3598, 0 },
		{ 3102, 4301, 0 },
		{ 3102, 4303, 0 },
		{ 2118, 4090, 0 },
		{ 0, 0, 63 },
		{ 0, 0, 66 },
		{ 2977, 4511, 0 },
		{ 1204, 4045, 0 },
		{ 2048, 3871, 0 },
		{ 3068, 3599, 0 },
		{ 1204, 4052, 0 },
		{ 3068, 3600, 0 },
		{ 0, 0, 111 },
		{ 3097, 3377, 0 },
		{ 3097, 3379, 0 },
		{ 3068, 3608, 0 },
		{ 0, 0, 104 },
		{ 2086, 3183, 0 },
		{ 2977, 4581, 0 },
		{ 0, 0, 112 },
		{ 0, 0, 113 },
		{ 2133, 3711, 0 },
		{ 2048, 3894, 0 },
		{ 3067, 3802, 0 },
		{ 959, 3917, 0 },
		{ 2067, 3839, 0 },
		{ 1204, 4069, 0 },
		{ 3097, 3385, 0 },
		{ 2912, 4368, 0 },
		{ 0, 0, 3 },
		{ 2118, 4112, 0 },
		{ 3102, 4299, 0 },
		{ 2977, 4641, 0 },
		{ 3067, 3805, 0 },
		{ 3018, 3950, 0 },
		{ 3097, 3386, 0 },
		{ 3018, 3953, 0 },
		{ 2118, 4121, 0 },
		{ 2086, 3184, 0 },
		{ 2067, 3846, 0 },
		{ 2217, 4232, 0 },
		{ 2064, 3460, 0 },
		{ 2064, 3461, 0 },
		{ 2118, 4130, 0 },
		{ 3067, 3818, 0 },
		{ 1031, 4017, 0 },
		{ 2118, 3093, 0 },
		{ 3018, 3961, 0 },
		{ 2133, 3734, 0 },
		{ 0, 0, 73 },
		{ 2118, 4138, 0 },
		{ 0, 0, 81 },
		{ 2977, 4435, 0 },
		{ 2118, 4139, 0 },
		{ 2977, 4439, 0 },
		{ 3097, 3394, 0 },
		{ 2118, 4142, 0 },
		{ 3094, 3322, 0 },
		{ 3102, 4348, 0 },
		{ 2118, 4145, 0 },
		{ 2133, 3741, 0 },
		{ 2048, 3869, 0 },
		{ 3097, 3397, 0 },
		{ 2048, 3872, 0 },
		{ 3018, 3973, 0 },
		{ 2133, 3744, 0 },
		{ 3018, 3980, 0 },
		{ 2118, 4159, 0 },
		{ 0, 0, 68 },
		{ 2133, 3745, 0 },
		{ 2133, 3746, 0 },
		{ 2977, 4516, 0 },
		{ 2067, 3835, 0 },
		{ 3097, 3398, 0 },
		{ 3067, 3787, 0 },
		{ 2118, 4077, 0 },
		{ 2133, 3749, 0 },
		{ 2118, 4079, 0 },
		{ 2086, 3186, 0 },
		{ 3018, 3993, 0 },
		{ 3090, 3654, 0 },
		{ 2977, 4577, 0 },
		{ 2067, 3841, 0 },
		{ 2048, 3903, 0 },
		{ 2064, 3479, 0 },
		{ 3102, 4347, 0 },
		{ 2064, 3480, 0 },
		{ 2086, 3187, 0 },
		{ 2133, 3700, 0 },
		{ 2026, 3316, 0 },
		{ 3102, 4305, 0 },
		{ 2118, 4095, 0 },
		{ 2118, 4097, 0 },
		{ 859, 3278, 0 },
		{ 0, 3280, 0 },
		{ 3067, 3807, 0 },
		{ 2154, 4036, 0 },
		{ 2118, 4102, 0 },
		{ 3068, 3550, 0 },
		{ 1204, 4058, 0 },
		{ 2977, 4401, 0 },
		{ 3068, 3551, 0 },
		{ 2086, 3188, 0 },
		{ 3097, 3408, 0 },
		{ 3068, 3559, 0 },
		{ 2048, 3868, 0 },
		{ 3018, 3943, 0 },
		{ 3068, 3560, 0 },
		{ 3067, 3764, 0 },
		{ 3097, 3409, 0 },
		{ 2217, 4197, 0 },
		{ 2217, 4199, 0 },
		{ 2977, 4433, 0 },
		{ 2118, 4117, 0 },
		{ 0, 0, 67 },
		{ 2048, 3873, 0 },
		{ 3097, 3410, 0 },
		{ 3090, 3686, 0 },
		{ 3068, 3564, 0 },
		{ 3068, 3566, 0 },
		{ 3068, 3568, 0 },
		{ 3097, 3412, 0 },
		{ 2133, 3736, 0 },
		{ 2133, 3737, 0 },
		{ 0, 0, 132 },
		{ 0, 0, 133 },
		{ 2067, 3843, 0 },
		{ 1204, 4061, 0 },
		{ 3097, 3413, 0 },
		{ 2048, 3898, 0 },
		{ 2048, 3902, 0 },
		{ 2912, 4359, 0 },
		{ 0, 0, 10 },
		{ 2977, 4505, 0 },
		{ 2064, 3504, 0 },
		{ 3097, 3414, 0 },
		{ 2977, 4049, 0 },
		{ 3102, 4351, 0 },
		{ 3067, 3792, 0 },
		{ 2977, 4547, 0 },
		{ 2977, 4549, 0 },
		{ 3097, 3415, 0 },
		{ 2086, 3189, 0 },
		{ 3018, 3975, 0 },
		{ 3018, 3976, 0 },
		{ 2118, 4151, 0 },
		{ 2086, 3190, 0 },
		{ 3102, 4326, 0 },
		{ 2977, 4575, 0 },
		{ 3102, 4331, 0 },
		{ 2048, 3856, 0 },
		{ 0, 0, 82 },
		{ 2133, 3747, 0 },
		{ 3018, 3983, 0 },
		{ 0, 0, 80 },
		{ 3094, 3346, 0 },
		{ 2067, 3828, 0 },
		{ 0, 0, 83 },
		{ 3102, 4346, 0 },
		{ 3018, 3987, 0 },
		{ 3094, 3347, 0 },
		{ 2118, 4170, 0 },
		{ 2064, 3513, 0 },
		{ 3068, 3583, 0 },
		{ 2118, 4076, 0 },
		{ 2086, 3191, 0 },
		{ 3097, 3421, 0 },
		{ 0, 0, 61 },
		{ 0, 0, 99 },
		{ 0, 0, 101 },
		{ 2133, 3755, 0 },
		{ 2217, 4203, 0 },
		{ 3068, 3586, 0 },
		{ 2118, 4081, 0 },
		{ 3018, 4000, 0 },
		{ 3090, 3664, 0 },
		{ 2118, 4085, 0 },
		{ 0, 0, 138 },
		{ 3018, 4002, 0 },
		{ 3068, 3587, 0 },
		{ 2118, 4089, 0 },
		{ 3018, 4005, 0 },
		{ 3097, 3422, 0 },
		{ 3068, 3589, 0 },
		{ 2977, 4425, 0 },
		{ 2086, 3192, 0 },
		{ 2048, 3881, 0 },
		{ 2048, 3885, 0 },
		{ 2118, 4096, 0 },
		{ 2217, 4289, 0 },
		{ 3097, 3424, 0 },
		{ 3097, 3425, 0 },
		{ 3068, 3594, 0 },
		{ 2154, 4042, 0 },
		{ 0, 3915, 0 },
		{ 3097, 3426, 0 },
		{ 3097, 3427, 0 },
		{ 3018, 3937, 0 },
		{ 3090, 3680, 0 },
		{ 2133, 3722, 0 },
		{ 2977, 4489, 0 },
		{ 3018, 3945, 0 },
		{ 3097, 3428, 0 },
		{ 2133, 3727, 0 },
		{ 3102, 4352, 0 },
		{ 0, 0, 19 },
		{ 2064, 3456, 0 },
		{ 2064, 3457, 0 },
		{ 0, 0, 129 },
		{ 2064, 3459, 0 },
		{ 0, 0, 131 },
		{ 3068, 3602, 0 },
		{ 2118, 4116, 0 },
		{ 0, 0, 97 },
		{ 2086, 3193, 0 },
		{ 2048, 3909, 0 },
		{ 2048, 3910, 0 },
		{ 2086, 3194, 0 },
		{ 2048, 3852, 0 },
		{ 2977, 4553, 0 },
		{ 2064, 3468, 0 },
		{ 2133, 3739, 0 },
		{ 3018, 3964, 0 },
		{ 0, 0, 78 },
		{ 2048, 3857, 0 },
		{ 1204, 4067, 0 },
		{ 2086, 3195, 0 },
		{ 2048, 3859, 0 },
		{ 3068, 3610, 0 },
		{ 2118, 4135, 0 },
		{ 3102, 4295, 0 },
		{ 3102, 4298, 0 },
		{ 2977, 4617, 0 },
		{ 2118, 4136, 0 },
		{ 3018, 3969, 0 },
		{ 3018, 3970, 0 },
		{ 2086, 3196, 0 },
		{ 2064, 3471, 0 },
		{ 3097, 3433, 0 },
		{ 3067, 3811, 0 },
		{ 3097, 3434, 0 },
		{ 2064, 3474, 0 },
		{ 3018, 3977, 0 },
		{ 3067, 3814, 0 },
		{ 3097, 3435, 0 },
		{ 2118, 4152, 0 },
		{ 0, 0, 86 },
		{ 3102, 4342, 0 },
		{ 0, 0, 103 },
		{ 2048, 3870, 0 },
		{ 3102, 3929, 0 },
		{ 2118, 4156, 0 },
		{ 0, 0, 136 },
		{ 3097, 3436, 0 },
		{ 3018, 3985, 0 },
		{ 3097, 3437, 0 },
		{ 2086, 3197, 58 },
		{ 3067, 3765, 0 },
		{ 2133, 3752, 0 },
		{ 2048, 3879, 0 },
		{ 3094, 3331, 0 },
		{ 2912, 3894, 0 },
		{ 0, 0, 87 },
		{ 2064, 3482, 0 },
		{ 3102, 4309, 0 },
		{ 1031, 4014, 0 },
		{ 0, 4015, 0 },
		{ 3097, 3440, 0 },
		{ 3067, 3775, 0 },
		{ 3067, 3778, 0 },
		{ 2133, 3756, 0 },
		{ 3102, 4337, 0 },
		{ 2118, 4083, 0 },
		{ 3018, 4003, 0 },
		{ 2086, 3198, 0 },
		{ 2133, 3758, 0 },
		{ 2133, 3760, 0 },
		{ 2133, 3699, 0 },
		{ 0, 0, 91 },
		{ 3018, 4008, 0 },
		{ 2064, 3492, 0 },
		{ 3068, 3544, 0 },
		{ 0, 0, 124 },
		{ 2086, 3199, 0 },
		{ 3094, 3338, 0 },
		{ 2086, 3200, 0 },
		{ 3090, 3677, 0 },
		{ 3097, 3445, 0 },
		{ 3018, 3933, 0 },
		{ 2217, 4238, 0 },
		{ 2064, 3498, 0 },
		{ 3067, 3797, 0 },
		{ 0, 0, 93 },
		{ 3067, 3798, 0 },
		{ 2217, 4283, 0 },
		{ 2217, 4285, 0 },
		{ 3097, 3446, 0 },
		{ 0, 0, 15 },
		{ 2048, 3855, 0 },
		{ 0, 0, 17 },
		{ 0, 0, 18 },
		{ 3018, 3944, 0 },
		{ 2086, 3201, 0 },
		{ 2154, 4025, 0 },
		{ 3067, 3803, 0 },
		{ 2977, 4573, 0 },
		{ 3068, 3555, 0 },
		{ 2133, 3720, 0 },
		{ 1204, 4064, 0 },
		{ 3068, 3557, 0 },
		{ 3068, 3558, 0 },
		{ 3067, 3810, 0 },
		{ 2133, 3726, 0 },
		{ 0, 0, 57 },
		{ 3018, 3956, 0 },
		{ 3097, 3448, 0 },
		{ 2086, 3202, 0 },
		{ 3097, 3450, 0 },
		{ 2048, 3867, 0 },
		{ 1091, 3260, 0 },
		{ 2133, 3733, 0 },
		{ 2118, 4128, 0 },
		{ 0, 0, 76 },
		{ 2086, 3203, 0 },
		{ 3102, 4325, 0 },
		{ 3068, 3565, 0 },
		{ 0, 3263, 0 },
		{ 3068, 3567, 0 },
		{ 0, 0, 16 },
		{ 2133, 3738, 0 },
		{ 1204, 4059, 0 },
		{ 2086, 3204, 56 },
		{ 2086, 3205, 0 },
		{ 2048, 3880, 0 },
		{ 0, 0, 77 },
		{ 3067, 3774, 0 },
		{ 3102, 3312, 0 },
		{ 0, 0, 84 },
		{ 0, 0, 85 },
		{ 0, 0, 54 },
		{ 3067, 3777, 0 },
		{ 3090, 3656, 0 },
		{ 3090, 3658, 0 },
		{ 3097, 3365, 0 },
		{ 3090, 3661, 0 },
		{ 0, 0, 137 },
		{ 0, 0, 126 },
		{ 3067, 3784, 0 },
		{ 1204, 4071, 0 },
		{ 1204, 4072, 0 },
		{ 3097, 3367, 0 },
		{ 0, 0, 39 },
		{ 2086, 3206, 40 },
		{ 2086, 3207, 42 },
		{ 3067, 3788, 0 },
		{ 3102, 4311, 0 },
		{ 1204, 4050, 0 },
		{ 1204, 4051, 0 },
		{ 2048, 3900, 0 },
		{ 0, 0, 74 },
		{ 3067, 3789, 0 },
		{ 3097, 3370, 0 },
		{ 0, 0, 92 },
		{ 3097, 3372, 0 },
		{ 2048, 3905, 0 },
		{ 3090, 3667, 0 },
		{ 2048, 3907, 0 },
		{ 2064, 3452, 0 },
		{ 3018, 3996, 0 },
		{ 3094, 3352, 0 },
		{ 3018, 3999, 0 },
		{ 2154, 4026, 0 },
		{ 2154, 4030, 0 },
		{ 2086, 3208, 0 },
		{ 3097, 3375, 0 },
		{ 3102, 4294, 0 },
		{ 3094, 3321, 0 },
		{ 0, 0, 88 },
		{ 3102, 4296, 0 },
		{ 2086, 3209, 0 },
		{ 0, 0, 130 },
		{ 0, 0, 134 },
		{ 2048, 3854, 0 },
		{ 0, 0, 139 },
		{ 0, 0, 11 },
		{ 3067, 3799, 0 },
		{ 3067, 3800, 0 },
		{ 2133, 3759, 0 },
		{ 3090, 3675, 0 },
		{ 3090, 3676, 0 },
		{ 1204, 4047, 0 },
		{ 2086, 3210, 0 },
		{ 3097, 3380, 0 },
		{ 3067, 3806, 0 },
		{ 3097, 3381, 0 },
		{ 3102, 4329, 0 },
		{ 0, 0, 135 },
		{ 3018, 3930, 0 },
		{ 3102, 4332, 0 },
		{ 3067, 3808, 0 },
		{ 3094, 3327, 0 },
		{ 3094, 3329, 0 },
		{ 3094, 3330, 0 },
		{ 3102, 4339, 0 },
		{ 2086, 3088, 0 },
		{ 3102, 4341, 0 },
		{ 3018, 3936, 0 },
		{ 2977, 4639, 0 },
		{ 3097, 3388, 0 },
		{ 3097, 3389, 0 },
		{ 2086, 3098, 0 },
		{ 0, 0, 41 },
		{ 0, 0, 43 },
		{ 3067, 3816, 0 },
		{ 2977, 4649, 0 },
		{ 3102, 4349, 0 },
		{ 3097, 3392, 0 },
		{ 2133, 3716, 0 },
		{ 2048, 3875, 0 },
		{ 3018, 3948, 0 },
		{ 3018, 3949, 0 },
		{ 2912, 4366, 0 },
		{ 3102, 4297, 0 },
		{ 2048, 3876, 0 },
		{ 2977, 4413, 0 },
		{ 3018, 3951, 0 },
		{ 3067, 3820, 0 },
		{ 2048, 3878, 0 },
		{ 2133, 3718, 0 },
		{ 2133, 3719, 0 },
		{ 2118, 4124, 0 },
		{ 3097, 3393, 0 },
		{ 2048, 3882, 0 },
		{ 2048, 3883, 0 },
		{ 2133, 3721, 0 },
		{ 0, 0, 79 },
		{ 0, 0, 94 },
		{ 3067, 3769, 0 },
		{ 3067, 3770, 0 },
		{ 0, 4054, 0 },
		{ 3018, 3962, 0 },
		{ 0, 0, 89 },
		{ 2048, 3887, 0 },
		{ 3067, 3771, 0 },
		{ 2064, 3478, 0 },
		{ 0, 0, 12 },
		{ 2133, 3723, 0 },
		{ 2133, 3724, 0 },
		{ 0, 0, 90 },
		{ 0, 0, 55 },
		{ 0, 0, 95 },
		{ 3068, 3601, 0 },
		{ 3067, 3776, 0 },
		{ 2118, 4141, 0 },
		{ 0, 0, 14 },
		{ 2086, 3100, 0 },
		{ 3068, 3603, 0 },
		{ 2118, 4144, 0 },
		{ 3090, 3651, 0 },
		{ 2048, 3901, 0 },
		{ 2977, 4509, 0 },
		{ 3102, 4350, 0 },
		{ 2118, 4147, 0 },
		{ 2067, 3847, 0 },
		{ 2118, 4149, 0 },
		{ 3067, 3781, 0 },
		{ 3097, 3395, 0 },
		{ 0, 0, 13 },
		{ 3146, 1385, 232 },
		{ 0, 0, 233 },
		{ 3101, 4980, 234 },
		{ 3129, 1701, 238 },
		{ 1241, 2621, 239 },
		{ 0, 0, 239 },
		{ 3129, 1723, 235 },
		{ 1244, 1387, 0 },
		{ 3129, 1734, 236 },
		{ 1247, 1434, 0 },
		{ 1244, 0, 0 },
		{ 3053, 2586, 237 },
		{ 1249, 1468, 0 },
		{ 1247, 0, 0 },
		{ 3053, 2606, 235 },
		{ 1249, 0, 0 },
		{ 3053, 2618, 236 },
		{ 3094, 3325, 148 },
		{ 0, 0, 148 },
		{ 0, 0, 149 },
		{ 3107, 1979, 0 },
		{ 3129, 2857, 0 },
		{ 3146, 2062, 0 },
		{ 1257, 4839, 0 },
		{ 3126, 2599, 0 },
		{ 3129, 2912, 0 },
		{ 3141, 2964, 0 },
		{ 3137, 2473, 0 },
		{ 3140, 3003, 0 },
		{ 3146, 2097, 0 },
		{ 3140, 3030, 0 },
		{ 3142, 1942, 0 },
		{ 3044, 2684, 0 },
		{ 3144, 2183, 0 },
		{ 3098, 2330, 0 },
		{ 3107, 2048, 0 },
		{ 3147, 4449, 0 },
		{ 0, 0, 146 },
		{ 2912, 4362, 156 },
		{ 0, 0, 156 },
		{ 0, 0, 157 },
		{ 3129, 2816, 0 },
		{ 2990, 2791, 0 },
		{ 3144, 2188, 0 },
		{ 3146, 2104, 0 },
		{ 3129, 2811, 0 },
		{ 1280, 4769, 0 },
		{ 3129, 2522, 0 },
		{ 3111, 1443, 0 },
		{ 3129, 2890, 0 },
		{ 3146, 2065, 0 },
		{ 2754, 1405, 0 },
		{ 3142, 1832, 0 },
		{ 3136, 2741, 0 },
		{ 3044, 2717, 0 },
		{ 3098, 2289, 0 },
		{ 2946, 2757, 0 },
		{ 1291, 4865, 0 },
		{ 3129, 2527, 0 },
		{ 3137, 2441, 0 },
		{ 3107, 2042, 0 },
		{ 3129, 2835, 0 },
		{ 1296, 4862, 0 },
		{ 3139, 2464, 0 },
		{ 3134, 1561, 0 },
		{ 3098, 2284, 0 },
		{ 3141, 2958, 0 },
		{ 3142, 1681, 0 },
		{ 3044, 2591, 0 },
		{ 3144, 2236, 0 },
		{ 3098, 2341, 0 },
		{ 3147, 4579, 0 },
		{ 0, 0, 154 },
		{ 3094, 3354, 180 },
		{ 0, 0, 180 },
		{ 3107, 1994, 0 },
		{ 3129, 2889, 0 },
		{ 3146, 2112, 0 },
		{ 1312, 4748, 0 },
		{ 3141, 2668, 0 },
		{ 3137, 2487, 0 },
		{ 3140, 3012, 0 },
		{ 3107, 2010, 0 },
		{ 3107, 2021, 0 },
		{ 3129, 2834, 0 },
		{ 3107, 2022, 0 },
		{ 3147, 4647, 0 },
		{ 0, 0, 179 },
		{ 2154, 4029, 144 },
		{ 0, 0, 144 },
		{ 0, 0, 145 },
		{ 3129, 2855, 0 },
		{ 3098, 2343, 0 },
		{ 3144, 2197, 0 },
		{ 3131, 2396, 0 },
		{ 3129, 2908, 0 },
		{ 3102, 4317, 0 },
		{ 3137, 2495, 0 },
		{ 3140, 3000, 0 },
		{ 3107, 2035, 0 },
		{ 3107, 2041, 0 },
		{ 3109, 4718, 0 },
		{ 3109, 4714, 0 },
		{ 3044, 2643, 0 },
		{ 3098, 2298, 0 },
		{ 3044, 2700, 0 },
		{ 3142, 1711, 0 },
		{ 3044, 2731, 0 },
		{ 3140, 2998, 0 },
		{ 3137, 2437, 0 },
		{ 3044, 2593, 0 },
		{ 3107, 1555, 0 },
		{ 3129, 2909, 0 },
		{ 3146, 2071, 0 },
		{ 3147, 4581, 0 },
		{ 0, 0, 142 },
		{ 2655, 2987, 23 },
		{ 0, 0, 23 },
		{ 0, 0, 24 },
		{ 3129, 2924, 0 },
		{ 2946, 2761, 0 },
		{ 3044, 2686, 0 },
		{ 3098, 2266, 0 },
		{ 3101, 2, 0 },
		{ 3144, 2180, 0 },
		{ 2904, 2146, 0 },
		{ 3129, 2847, 0 },
		{ 3146, 2084, 0 },
		{ 3140, 3034, 0 },
		{ 3142, 1865, 0 },
		{ 3144, 2228, 0 },
		{ 3146, 2099, 0 },
		{ 3101, 4959, 0 },
		{ 3126, 2942, 0 },
		{ 3129, 2905, 0 },
		{ 3107, 1990, 0 },
		{ 3141, 2959, 0 },
		{ 3146, 2107, 0 },
		{ 3044, 2710, 0 },
		{ 2904, 2139, 0 },
		{ 3142, 1933, 0 },
		{ 3044, 2527, 0 },
		{ 3144, 2215, 0 },
		{ 3098, 2295, 0 },
		{ 3101, 7, 0 },
		{ 3109, 4721, 0 },
		{ 0, 0, 20 },
		{ 1395, 0, 1 },
		{ 1395, 0, 181 },
		{ 1395, 2719, 231 },
		{ 1610, 172, 231 },
		{ 1610, 410, 231 },
		{ 1610, 398, 231 },
		{ 1610, 528, 231 },
		{ 1610, 403, 231 },
		{ 1610, 414, 231 },
		{ 1610, 389, 231 },
		{ 1610, 420, 231 },
		{ 1610, 484, 231 },
		{ 1395, 0, 231 },
		{ 1407, 2520, 231 },
		{ 1395, 2801, 231 },
		{ 2655, 2990, 227 },
		{ 1610, 505, 231 },
		{ 1610, 504, 231 },
		{ 1610, 533, 231 },
		{ 1610, 0, 231 },
		{ 1610, 568, 231 },
		{ 1610, 555, 231 },
		{ 3144, 2238, 0 },
		{ 0, 0, 182 },
		{ 3098, 2292, 0 },
		{ 1610, 522, 0 },
		{ 1610, 0, 0 },
		{ 3101, 3980, 0 },
		{ 1610, 568, 0 },
		{ 1610, 585, 0 },
		{ 1610, 582, 0 },
		{ 1610, 590, 0 },
		{ 1610, 608, 0 },
		{ 1610, 611, 0 },
		{ 1610, 619, 0 },
		{ 1610, 601, 0 },
		{ 1610, 592, 0 },
		{ 1610, 585, 0 },
		{ 1610, 592, 0 },
		{ 3129, 2868, 0 },
		{ 3129, 2879, 0 },
		{ 1611, 609, 0 },
		{ 1611, 610, 0 },
		{ 1610, 620, 0 },
		{ 1610, 621, 0 },
		{ 1610, 612, 0 },
		{ 3144, 2185, 0 },
		{ 3126, 2933, 0 },
		{ 1610, 610, 0 },
		{ 1610, 654, 0 },
		{ 1610, 657, 0 },
		{ 1610, 658, 0 },
		{ 1610, 681, 0 },
		{ 1610, 682, 0 },
		{ 1610, 687, 0 },
		{ 1610, 681, 0 },
		{ 1610, 693, 0 },
		{ 1610, 684, 0 },
		{ 1610, 681, 0 },
		{ 1610, 696, 0 },
		{ 1610, 683, 0 },
		{ 3098, 2314, 0 },
		{ 2990, 2784, 0 },
		{ 1610, 699, 0 },
		{ 1610, 689, 0 },
		{ 1611, 18, 0 },
		{ 1610, 22, 0 },
		{ 1610, 29, 0 },
		{ 3137, 2449, 0 },
		{ 0, 0, 230 },
		{ 1610, 41, 0 },
		{ 1610, 26, 0 },
		{ 1610, 14, 0 },
		{ 1610, 63, 0 },
		{ 1610, 67, 0 },
		{ 1610, 68, 0 },
		{ 1610, 64, 0 },
		{ 1610, 52, 0 },
		{ 1610, 33, 0 },
		{ 1610, 37, 0 },
		{ 1610, 53, 0 },
		{ 1610, 0, 216 },
		{ 1610, 89, 0 },
		{ 3144, 2234, 0 },
		{ 3044, 2721, 0 },
		{ 1610, 47, 0 },
		{ 1610, 51, 0 },
		{ 1610, 61, 0 },
		{ 1610, 64, 0 },
		{ 1610, 90, 0 },
		{ -1490, 1544, 0 },
		{ 1611, 100, 0 },
		{ 1610, 134, 0 },
		{ 1610, 169, 0 },
		{ 1610, 174, 0 },
		{ 1610, 184, 0 },
		{ 1610, 185, 0 },
		{ 1610, 164, 0 },
		{ 1610, 180, 0 },
		{ 1610, 156, 0 },
		{ 1610, 150, 0 },
		{ 1610, 0, 215 },
		{ 1610, 157, 0 },
		{ 3131, 2408, 0 },
		{ 3098, 2272, 0 },
		{ 1610, 161, 0 },
		{ 1610, 171, 0 },
		{ 1610, 169, 0 },
		{ 1610, 0, 229 },
		{ 1610, 168, 0 },
		{ 0, 0, 217 },
		{ 1610, 172, 0 },
		{ 1612, 4, -4 },
		{ 1610, 200, 0 },
		{ 1610, 212, 0 },
		{ 1610, 273, 0 },
		{ 1610, 279, 0 },
		{ 1610, 213, 0 },
		{ 1610, 225, 0 },
		{ 1610, 222, 0 },
		{ 1610, 229, 0 },
		{ 1610, 221, 0 },
		{ 3129, 2826, 0 },
		{ 3129, 2830, 0 },
		{ 1610, 0, 219 },
		{ 1610, 301, 220 },
		{ 1610, 271, 0 },
		{ 1610, 274, 0 },
		{ 1610, 301, 0 },
		{ 1510, 3567, 0 },
		{ 3101, 4250, 0 },
		{ 2195, 4672, 206 },
		{ 1610, 304, 0 },
		{ 1610, 308, 0 },
		{ 1610, 306, 0 },
		{ 1610, 313, 0 },
		{ 1610, 315, 0 },
		{ 1610, 319, 0 },
		{ 1610, 308, 0 },
		{ 1610, 322, 0 },
		{ 1610, 304, 0 },
		{ 1610, 312, 0 },
		{ 1611, 325, 0 },
		{ 3102, 4319, 0 },
		{ 3101, 4979, 222 },
		{ 1610, 330, 0 },
		{ 1610, 343, 0 },
		{ 1610, 358, 0 },
		{ 1610, 374, 0 },
		{ 0, 0, 186 },
		{ 1612, 31, -7 },
		{ 1612, 117, -10 },
		{ 1612, 231, -13 },
		{ 1612, 345, -16 },
		{ 1612, 376, -19 },
		{ 1612, 460, -22 },
		{ 1610, 406, 0 },
		{ 1610, 419, 0 },
		{ 1610, 392, 0 },
		{ 1610, 0, 204 },
		{ 1610, 0, 218 },
		{ 3137, 2433, 0 },
		{ 1610, 391, 0 },
		{ 1610, 381, 0 },
		{ 1610, 386, 0 },
		{ 1611, 386, 0 },
		{ 1547, 3536, 0 },
		{ 3101, 4282, 0 },
		{ 2195, 4685, 207 },
		{ 1550, 3537, 0 },
		{ 3101, 4246, 0 },
		{ 2195, 4669, 208 },
		{ 1553, 3538, 0 },
		{ 3101, 4276, 0 },
		{ 2195, 4688, 211 },
		{ 1556, 3539, 0 },
		{ 3101, 4312, 0 },
		{ 2195, 4683, 212 },
		{ 1559, 3540, 0 },
		{ 3101, 4244, 0 },
		{ 2195, 4691, 213 },
		{ 1562, 3541, 0 },
		{ 3101, 4248, 0 },
		{ 2195, 4675, 214 },
		{ 1610, 436, 0 },
		{ 1612, 488, -25 },
		{ 1610, 423, 0 },
		{ 3140, 3047, 0 },
		{ 1610, 404, 0 },
		{ 1610, 481, 0 },
		{ 1610, 443, 0 },
		{ 1610, 493, 0 },
		{ 0, 0, 188 },
		{ 0, 0, 190 },
		{ 0, 0, 196 },
		{ 0, 0, 198 },
		{ 0, 0, 200 },
		{ 0, 0, 202 },
		{ 1612, 490, -28 },
		{ 1580, 3551, 0 },
		{ 3101, 4280, 0 },
		{ 2195, 4665, 210 },
		{ 1610, 0, 203 },
		{ 3107, 2024, 0 },
		{ 1610, 481, 0 },
		{ 1610, 496, 0 },
		{ 1611, 489, 0 },
		{ 1610, 486, 0 },
		{ 1589, 3558, 0 },
		{ 3101, 4252, 0 },
		{ 2195, 4668, 209 },
		{ 0, 0, 194 },
		{ 3107, 2051, 0 },
		{ 1610, 4, 225 },
		{ 1611, 493, 0 },
		{ 1610, 3, 228 },
		{ 1610, 510, 0 },
		{ 0, 0, 192 },
		{ 3109, 4719, 0 },
		{ 3109, 4720, 0 },
		{ 1610, 498, 0 },
		{ 0, 0, 226 },
		{ 1610, 495, 0 },
		{ 3109, 4715, 0 },
		{ 0, 0, 224 },
		{ 1610, 506, 0 },
		{ 1610, 511, 0 },
		{ 0, 0, 223 },
		{ 1610, 516, 0 },
		{ 1610, 519, 0 },
		{ 1611, 521, 221 },
		{ 1612, 925, 0 },
		{ 1613, 736, -1 },
		{ 1614, 3582, 0 },
		{ 3101, 4236, 0 },
		{ 2195, 4694, 205 },
		{ 0, 0, 184 },
		{ 2154, 4034, 274 },
		{ 0, 0, 274 },
		{ 3129, 2885, 0 },
		{ 3098, 2320, 0 },
		{ 3144, 2211, 0 },
		{ 3131, 2390, 0 },
		{ 3129, 2907, 0 },
		{ 3102, 4322, 0 },
		{ 3137, 2434, 0 },
		{ 3140, 3008, 0 },
		{ 3107, 1983, 0 },
		{ 3107, 1984, 0 },
		{ 3109, 4701, 0 },
		{ 3109, 4704, 0 },
		{ 3044, 2711, 0 },
		{ 3098, 2264, 0 },
		{ 3044, 2720, 0 },
		{ 3142, 1836, 0 },
		{ 3044, 2730, 0 },
		{ 3140, 3004, 0 },
		{ 3137, 2485, 0 },
		{ 3044, 2517, 0 },
		{ 3107, 1561, 0 },
		{ 3129, 2843, 0 },
		{ 3146, 2095, 0 },
		{ 3147, 4587, 0 },
		{ 0, 0, 273 },
		{ 2154, 4028, 276 },
		{ 0, 0, 276 },
		{ 0, 0, 277 },
		{ 3129, 2851, 0 },
		{ 3098, 2283, 0 },
		{ 3144, 2240, 0 },
		{ 3131, 2411, 0 },
		{ 3129, 2870, 0 },
		{ 3102, 4354, 0 },
		{ 3137, 2439, 0 },
		{ 3140, 3024, 0 },
		{ 3107, 2000, 0 },
		{ 3107, 2002, 0 },
		{ 3109, 4705, 0 },
		{ 3109, 4710, 0 },
		{ 3141, 2978, 0 },
		{ 3146, 2102, 0 },
		{ 3144, 2187, 0 },
		{ 3107, 2003, 0 },
		{ 3107, 2008, 0 },
		{ 3144, 2204, 0 },
		{ 3111, 1473, 0 },
		{ 3129, 2917, 0 },
		{ 3146, 2118, 0 },
		{ 3147, 4649, 0 },
		{ 0, 0, 275 },
		{ 2154, 4035, 279 },
		{ 0, 0, 279 },
		{ 0, 0, 280 },
		{ 3129, 2925, 0 },
		{ 3098, 2348, 0 },
		{ 3144, 2223, 0 },
		{ 3131, 2409, 0 },
		{ 3129, 2821, 0 },
		{ 3102, 4330, 0 },
		{ 3137, 2465, 0 },
		{ 3140, 3007, 0 },
		{ 3107, 2013, 0 },
		{ 3107, 2019, 0 },
		{ 3109, 4716, 0 },
		{ 3109, 4717, 0 },
		{ 3131, 2392, 0 },
		{ 3134, 1521, 0 },
		{ 3142, 1948, 0 },
		{ 3140, 3046, 0 },
		{ 3142, 1957, 0 },
		{ 3144, 2168, 0 },
		{ 3146, 2096, 0 },
		{ 3147, 4476, 0 },
		{ 0, 0, 278 },
		{ 2154, 4039, 282 },
		{ 0, 0, 282 },
		{ 0, 0, 283 },
		{ 3129, 2859, 0 },
		{ 3098, 2294, 0 },
		{ 3144, 2181, 0 },
		{ 3131, 2398, 0 },
		{ 3129, 2883, 0 },
		{ 3102, 4353, 0 },
		{ 3137, 2472, 0 },
		{ 3140, 3025, 0 },
		{ 3107, 2028, 0 },
		{ 3107, 2029, 0 },
		{ 3109, 4712, 0 },
		{ 3109, 4713, 0 },
		{ 3129, 2904, 0 },
		{ 3111, 1440, 0 },
		{ 3140, 3062, 0 },
		{ 3137, 2493, 0 },
		{ 3134, 1519, 0 },
		{ 3140, 3002, 0 },
		{ 3142, 1806, 0 },
		{ 3144, 2201, 0 },
		{ 3146, 2109, 0 },
		{ 3147, 4546, 0 },
		{ 0, 0, 281 },
		{ 2154, 4022, 285 },
		{ 0, 0, 285 },
		{ 0, 0, 286 },
		{ 3129, 2923, 0 },
		{ 3098, 2350, 0 },
		{ 3144, 2210, 0 },
		{ 3131, 2410, 0 },
		{ 3129, 2807, 0 },
		{ 3102, 4328, 0 },
		{ 3137, 2461, 0 },
		{ 3140, 3043, 0 },
		{ 3107, 2046, 0 },
		{ 3107, 2047, 0 },
		{ 3109, 4699, 0 },
		{ 3109, 4700, 0 },
		{ 3144, 2217, 0 },
		{ 2904, 2160, 0 },
		{ 3142, 1826, 0 },
		{ 3044, 2521, 0 },
		{ 3131, 2400, 0 },
		{ 3044, 2583, 0 },
		{ 3107, 2049, 0 },
		{ 3129, 2854, 0 },
		{ 3146, 2070, 0 },
		{ 3147, 4431, 0 },
		{ 0, 0, 284 },
		{ 2977, 4569, 160 },
		{ 0, 0, 160 },
		{ 0, 0, 161 },
		{ 2990, 2795, 0 },
		{ 3142, 1828, 0 },
		{ 3129, 2867, 0 },
		{ 3146, 2079, 0 },
		{ 1754, 4824, 0 },
		{ 3129, 2516, 0 },
		{ 3111, 1442, 0 },
		{ 3129, 2881, 0 },
		{ 3146, 2088, 0 },
		{ 2754, 1406, 0 },
		{ 3142, 1845, 0 },
		{ 3136, 2751, 0 },
		{ 3044, 2713, 0 },
		{ 3098, 2336, 0 },
		{ 2946, 2762, 0 },
		{ 1765, 4858, 0 },
		{ 3129, 2529, 0 },
		{ 3137, 2479, 0 },
		{ 3107, 1987, 0 },
		{ 3129, 2916, 0 },
		{ 1770, 4757, 0 },
		{ 3139, 2466, 0 },
		{ 3134, 1643, 0 },
		{ 3098, 2347, 0 },
		{ 3141, 2955, 0 },
		{ 3142, 1881, 0 },
		{ 3044, 2577, 0 },
		{ 3144, 2195, 0 },
		{ 3098, 2260, 0 },
		{ 3147, 4447, 0 },
		{ 0, 0, 158 },
		{ 2154, 4038, 267 },
		{ 0, 0, 267 },
		{ 3129, 2812, 0 },
		{ 3098, 2261, 0 },
		{ 3144, 2196, 0 },
		{ 3131, 2403, 0 },
		{ 3129, 2827, 0 },
		{ 3102, 4335, 0 },
		{ 3137, 2457, 0 },
		{ 3140, 3016, 0 },
		{ 3107, 1997, 0 },
		{ 3107, 1998, 0 },
		{ 3109, 4706, 0 },
		{ 3109, 4708, 0 },
		{ 3126, 2945, 0 },
		{ 3044, 2708, 0 },
		{ 3107, 1999, 0 },
		{ 2904, 2136, 0 },
		{ 3137, 2475, 0 },
		{ 3140, 3058, 0 },
		{ 2754, 1404, 0 },
		{ 3147, 4437, 0 },
		{ 0, 0, 265 },
		{ 1818, 0, 1 },
		{ 1977, 2837, 382 },
		{ 3129, 2863, 382 },
		{ 3140, 2917, 382 },
		{ 3126, 2173, 382 },
		{ 1818, 0, 349 },
		{ 1818, 2799, 382 },
		{ 3136, 1486, 382 },
		{ 2912, 4361, 382 },
		{ 2133, 3713, 382 },
		{ 3094, 3343, 382 },
		{ 2133, 3715, 382 },
		{ 2118, 4127, 382 },
		{ 3146, 1973, 382 },
		{ 1818, 0, 382 },
		{ 2655, 2988, 380 },
		{ 3140, 2774, 382 },
		{ 3140, 3038, 382 },
		{ 0, 0, 382 },
		{ 3144, 2233, 0 },
		{ -1823, 20, 339 },
		{ -1824, 4796, 0 },
		{ 3098, 2307, 0 },
		{ 0, 0, 345 },
		{ 0, 0, 346 },
		{ 3137, 2435, 0 },
		{ 3044, 2574, 0 },
		{ 3129, 2891, 0 },
		{ 0, 0, 350 },
		{ 3098, 2310, 0 },
		{ 3146, 2067, 0 },
		{ 3044, 2589, 0 },
		{ 2086, 3108, 0 },
		{ 3090, 3679, 0 },
		{ 3097, 3366, 0 },
		{ 2026, 3305, 0 },
		{ 3090, 3684, 0 },
		{ 3134, 1625, 0 },
		{ 3107, 2012, 0 },
		{ 3098, 2332, 0 },
		{ 3142, 1650, 0 },
		{ 3146, 2083, 0 },
		{ 3144, 2169, 0 },
		{ 3051, 4772, 0 },
		{ 3144, 2174, 0 },
		{ 3107, 2014, 0 },
		{ 3142, 1653, 0 },
		{ 3098, 2259, 0 },
		{ 3126, 2950, 0 },
		{ 3146, 2094, 0 },
		{ 3137, 2490, 0 },
		{ 2154, 4023, 0 },
		{ 2086, 3124, 0 },
		{ 2086, 3125, 0 },
		{ 2118, 4169, 0 },
		{ 2048, 3899, 0 },
		{ 3129, 2814, 0 },
		{ 3107, 2020, 0 },
		{ 3126, 2947, 0 },
		{ 3134, 1626, 0 },
		{ 3129, 2824, 0 },
		{ 3137, 2427, 0 },
		{ 0, 17, 342 },
		{ 3131, 2394, 0 },
		{ 3129, 2828, 0 },
		{ 2133, 3698, 0 },
		{ 3142, 1699, 0 },
		{ 0, 0, 381 },
		{ 3129, 2833, 0 },
		{ 3126, 2932, 0 },
		{ 2118, 4086, 0 },
		{ 2064, 3458, 0 },
		{ 3090, 3669, 0 },
		{ 3068, 3552, 0 },
		{ 2086, 3139, 0 },
		{ 0, 0, 370 },
		{ 3102, 4324, 0 },
		{ 3144, 2192, 0 },
		{ 3146, 2098, 0 },
		{ 3098, 2280, 0 },
		{ -1900, 1092, 0 },
		{ 0, 0, 341 },
		{ 3129, 2849, 0 },
		{ 0, 0, 369 },
		{ 2904, 2145, 0 },
		{ 3044, 2581, 0 },
		{ 3098, 2286, 0 },
		{ 1915, 4735, 0 },
		{ 3067, 3821, 0 },
		{ 3018, 3979, 0 },
		{ 3068, 3563, 0 },
		{ 2086, 3148, 0 },
		{ 3090, 3682, 0 },
		{ 3144, 2199, 0 },
		{ 3131, 2378, 0 },
		{ 3098, 2291, 0 },
		{ 3142, 1712, 0 },
		{ 0, 0, 371 },
		{ 3102, 4345, 348 },
		{ 3142, 1714, 0 },
		{ 3141, 2957, 0 },
		{ 3142, 1717, 0 },
		{ 0, 0, 374 },
		{ 0, 0, 375 },
		{ 1920, 0, -41 },
		{ 2099, 3252, 0 },
		{ 2133, 3729, 0 },
		{ 3090, 3695, 0 },
		{ 2118, 4115, 0 },
		{ 3044, 2685, 0 },
		{ 0, 0, 373 },
		{ 0, 0, 379 },
		{ 0, 4737, 0 },
		{ 3137, 2484, 0 },
		{ 3107, 2037, 0 },
		{ 3140, 3010, 0 },
		{ 2154, 4033, 0 },
		{ 3101, 4326, 0 },
		{ 2195, 4693, 364 },
		{ 2118, 4122, 0 },
		{ 2912, 4363, 0 },
		{ 3068, 3578, 0 },
		{ 3068, 3579, 0 },
		{ 3098, 2303, 0 },
		{ 0, 0, 376 },
		{ 0, 0, 377 },
		{ 3140, 3014, 0 },
		{ 2201, 4779, 0 },
		{ 3137, 2488, 0 },
		{ 3129, 2887, 0 },
		{ 0, 0, 354 },
		{ 1943, 0, -44 },
		{ 1945, 0, -47 },
		{ 2133, 3742, 0 },
		{ 3102, 4321, 0 },
		{ 0, 0, 372 },
		{ 3107, 2038, 0 },
		{ 0, 0, 347 },
		{ 2154, 4020, 0 },
		{ 3098, 2308, 0 },
		{ 3101, 4308, 0 },
		{ 2195, 4674, 365 },
		{ 3101, 4310, 0 },
		{ 2195, 4679, 366 },
		{ 2912, 4358, 0 },
		{ 1955, 0, -77 },
		{ 3107, 2040, 0 },
		{ 3129, 2896, 0 },
		{ 3129, 2901, 0 },
		{ 0, 0, 356 },
		{ 0, 0, 358 },
		{ 1960, 0, -35 },
		{ 3101, 4232, 0 },
		{ 2195, 4667, 368 },
		{ 0, 0, 344 },
		{ 3098, 2311, 0 },
		{ 3146, 2060, 0 },
		{ 3101, 4238, 0 },
		{ 2195, 4673, 367 },
		{ 0, 0, 362 },
		{ 3144, 2224, 0 },
		{ 3140, 3061, 0 },
		{ 0, 0, 360 },
		{ 3131, 2388, 0 },
		{ 3142, 1751, 0 },
		{ 3129, 2911, 0 },
		{ 3044, 2735, 0 },
		{ 0, 0, 378 },
		{ 3144, 2230, 0 },
		{ 3098, 2335, 0 },
		{ 1974, 0, -50 },
		{ 3101, 4278, 0 },
		{ 2195, 4666, 363 },
		{ 0, 0, 352 },
		{ 1818, 2831, 382 },
		{ 1981, 2522, 382 },
		{ -1979, 4986, 339 },
		{ -1980, 4793, 0 },
		{ 3101, 4780, 0 },
		{ 3051, 4760, 0 },
		{ 0, 0, 340 },
		{ 3051, 4774, 0 },
		{ -1985, 4984, 0 },
		{ -1986, 4788, 0 },
		{ 1989, 2, 342 },
		{ 3051, 4775, 0 },
		{ 3101, 4836, 0 },
		{ 0, 0, 343 },
		{ 2007, 0, 1 },
		{ 2203, 2847, 338 },
		{ 3129, 2809, 338 },
		{ 2007, 0, 292 },
		{ 2007, 2650, 338 },
		{ 3090, 3683, 338 },
		{ 2007, 0, 295 },
		{ 3134, 1557, 338 },
		{ 2912, 4356, 338 },
		{ 2133, 3708, 338 },
		{ 3094, 3278, 338 },
		{ 2133, 3710, 338 },
		{ 2118, 4167, 338 },
		{ 3140, 3041, 338 },
		{ 3146, 1974, 338 },
		{ 2007, 0, 338 },
		{ 2655, 2980, 335 },
		{ 3140, 3050, 338 },
		{ 3126, 2937, 338 },
		{ 2912, 4370, 338 },
		{ 3140, 1641, 338 },
		{ 0, 0, 338 },
		{ 3144, 2242, 0 },
		{ -2014, 22, 287 },
		{ -2015, 4789, 0 },
		{ 3098, 2353, 0 },
		{ 0, 0, 293 },
		{ 3098, 2354, 0 },
		{ 3144, 2165, 0 },
		{ 3146, 2082, 0 },
		{ 2086, 3101, 0 },
		{ 3090, 3653, 0 },
		{ 3097, 3378, 0 },
		{ 3067, 3779, 0 },
		{ 0, 3269, 0 },
		{ 0, 3291, 0 },
		{ 3090, 3657, 0 },
		{ 3137, 2486, 0 },
		{ 3134, 1571, 0 },
		{ 3107, 2052, 0 },
		{ 3098, 2265, 0 },
		{ 3129, 2839, 0 },
		{ 3129, 2841, 0 },
		{ 3144, 2175, 0 },
		{ 2201, 4783, 0 },
		{ 3144, 2178, 0 },
		{ 3051, 4763, 0 },
		{ 3144, 2179, 0 },
		{ 3126, 2934, 0 },
		{ 2904, 2156, 0 },
		{ 3146, 2085, 0 },
		{ 2154, 4041, 0 },
		{ 2086, 3114, 0 },
		{ 2086, 3115, 0 },
		{ 3018, 3991, 0 },
		{ 3018, 3992, 0 },
		{ 2118, 4107, 0 },
		{ 0, 3897, 0 },
		{ 3107, 1968, 0 },
		{ 3129, 2856, 0 },
		{ 3107, 1974, 0 },
		{ 3126, 2949, 0 },
		{ 3098, 2288, 0 },
		{ 3107, 1976, 0 },
		{ 3137, 2445, 0 },
		{ 0, 0, 337 },
		{ 3137, 2447, 0 },
		{ 0, 0, 289 },
		{ 3131, 2384, 0 },
		{ 0, 0, 334 },
		{ 3134, 1617, 0 },
		{ 3129, 2872, 0 },
		{ 2118, 4120, 0 },
		{ 0, 3501, 0 },
		{ 3090, 3685, 0 },
		{ 2067, 3850, 0 },
		{ 0, 3823, 0 },
		{ 3068, 3580, 0 },
		{ 2086, 3127, 0 },
		{ 3129, 2876, 0 },
		{ 0, 0, 327 },
		{ 3102, 4327, 0 },
		{ 3144, 2189, 0 },
		{ 3142, 1850, 0 },
		{ 3142, 1856, 0 },
		{ 3134, 1624, 0 },
		{ -2094, 1167, 0 },
		{ 3129, 2888, 0 },
		{ 3137, 2477, 0 },
		{ 3098, 2304, 0 },
		{ 3067, 3763, 0 },
		{ 3018, 3938, 0 },
		{ 3068, 3591, 0 },
		{ 3018, 3941, 0 },
		{ 3018, 3942, 0 },
		{ 0, 3137, 0 },
		{ 3090, 3649, 0 },
		{ 0, 0, 326 },
		{ 3144, 2198, 0 },
		{ 3131, 2404, 0 },
		{ 3044, 2645, 0 },
		{ 0, 0, 333 },
		{ 3140, 3035, 0 },
		{ 0, 0, 328 },
		{ 0, 0, 291 },
		{ 3140, 3037, 0 },
		{ 3142, 1885, 0 },
		{ 2111, 0, -74 },
		{ 0, 3248, 0 },
		{ 2133, 3717, 0 },
		{ 2102, 3236, 0 },
		{ 2099, 3235, 0 },
		{ 3090, 3660, 0 },
		{ 2118, 4154, 0 },
		{ 3044, 2680, 0 },
		{ 0, 0, 330 },
		{ 3141, 2976, 0 },
		{ 3142, 1889, 0 },
		{ 3142, 1926, 0 },
		{ 2154, 4027, 0 },
		{ 3101, 4240, 0 },
		{ 2195, 4664, 317 },
		{ 2118, 4160, 0 },
		{ 2912, 4364, 0 },
		{ 2118, 4161, 0 },
		{ 2118, 4162, 0 },
		{ 2118, 4163, 0 },
		{ 0, 4164, 0 },
		{ 3068, 3606, 0 },
		{ 3068, 3607, 0 },
		{ 3098, 2312, 0 },
		{ 3140, 3057, 0 },
		{ 3044, 2691, 0 },
		{ 3044, 2695, 0 },
		{ 3129, 2914, 0 },
		{ 0, 0, 299 },
		{ 2140, 0, -53 },
		{ 2142, 0, -56 },
		{ 2144, 0, -62 },
		{ 2146, 0, -65 },
		{ 2148, 0, -68 },
		{ 2150, 0, -71 },
		{ 0, 3732, 0 },
		{ 3102, 4334, 0 },
		{ 0, 0, 329 },
		{ 3137, 2491, 0 },
		{ 3144, 2207, 0 },
		{ 3144, 2208, 0 },
		{ 3098, 2321, 0 },
		{ 3101, 4314, 0 },
		{ 2195, 4676, 318 },
		{ 3101, 4316, 0 },
		{ 2195, 4681, 319 },
		{ 3101, 4318, 0 },
		{ 2195, 4684, 322 },
		{ 3101, 4320, 0 },
		{ 2195, 4686, 323 },
		{ 3101, 4322, 0 },
		{ 2195, 4690, 324 },
		{ 3101, 4324, 0 },
		{ 2195, 4692, 325 },
		{ 2912, 4367, 0 },
		{ 2165, 0, -80 },
		{ 0, 4040, 0 },
		{ 3098, 2325, 0 },
		{ 3098, 2326, 0 },
		{ 3129, 2803, 0 },
		{ 0, 0, 301 },
		{ 0, 0, 303 },
		{ 0, 0, 309 },
		{ 0, 0, 311 },
		{ 0, 0, 313 },
		{ 0, 0, 315 },
		{ 2171, 0, -38 },
		{ 3101, 4234, 0 },
		{ 2195, 4671, 321 },
		{ 3129, 2805, 0 },
		{ 3140, 3011, 0 },
		{ 3076, 3229, 332 },
		{ 3146, 2110, 0 },
		{ 3101, 4242, 0 },
		{ 2195, 4677, 320 },
		{ 0, 0, 307 },
		{ 3098, 2331, 0 },
		{ 3146, 2111, 0 },
		{ 0, 0, 294 },
		{ 3140, 3021, 0 },
		{ 0, 0, 305 },
		{ 3144, 2213, 0 },
		{ 2754, 1393, 0 },
		{ 3142, 1937, 0 },
		{ 3131, 2407, 0 },
		{ 2977, 4483, 0 },
		{ 3044, 2519, 0 },
		{ 3129, 2825, 0 },
		{ 3137, 2455, 0 },
		{ 3144, 2218, 0 },
		{ 0, 0, 331 },
		{ 2946, 2754, 0 },
		{ 3098, 2345, 0 },
		{ 3144, 2220, 0 },
		{ 2194, 0, -59 },
		{ 3146, 2119, 0 },
		{ 3101, 4306, 0 },
		{ 0, 4670, 316 },
		{ 3044, 2580, 0 },
		{ 0, 0, 297 },
		{ 3142, 1939, 0 },
		{ 3136, 2747, 0 },
		{ 3131, 2380, 0 },
		{ 0, 4782, 0 },
		{ 0, 0, 336 },
		{ 2007, 2856, 338 },
		{ 2207, 2518, 338 },
		{ -2205, 21, 287 },
		{ -2206, 1, 0 },
		{ 3101, 4779, 0 },
		{ 3051, 4752, 0 },
		{ 0, 0, 288 },
		{ 3051, 4776, 0 },
		{ -2211, 4987, 0 },
		{ -2212, 4791, 0 },
		{ 2215, 0, 289 },
		{ 3051, 4777, 0 },
		{ 3101, 4920, 0 },
		{ 0, 0, 290 },
		{ 0, 4213, 384 },
		{ 0, 0, 384 },
		{ 3129, 2852, 0 },
		{ 2990, 2775, 0 },
		{ 3140, 3005, 0 },
		{ 3134, 1489, 0 },
		{ 3137, 2482, 0 },
		{ 3142, 1950, 0 },
		{ 3101, 4958, 0 },
		{ 3146, 2069, 0 },
		{ 3134, 1490, 0 },
		{ 3098, 2262, 0 },
		{ 2230, 4763, 0 },
		{ 3101, 1966, 0 },
		{ 3140, 3020, 0 },
		{ 3146, 2072, 0 },
		{ 3140, 3023, 0 },
		{ 3131, 2402, 0 },
		{ 3129, 2871, 0 },
		{ 3142, 1651, 0 },
		{ 3129, 2874, 0 },
		{ 3146, 2080, 0 },
		{ 3107, 2017, 0 },
		{ 3147, 4585, 0 },
		{ 0, 0, 383 },
		{ 3051, 4749, 433 },
		{ 0, 0, 389 },
		{ 0, 0, 391 },
		{ 2262, 833, 424 },
		{ 2438, 846, 424 },
		{ 2461, 844, 424 },
		{ 2404, 845, 424 },
		{ 2263, 859, 424 },
		{ 2261, 849, 424 },
		{ 2461, 845, 424 },
		{ 2284, 859, 424 },
		{ 2434, 861, 424 },
		{ 2434, 863, 424 },
		{ 2438, 860, 424 },
		{ 2378, 870, 424 },
		{ 2260, 888, 424 },
		{ 3129, 1645, 423 },
		{ 2292, 2677, 433 },
		{ 2496, 860, 424 },
		{ 2438, 873, 424 },
		{ 2296, 874, 424 },
		{ 2438, 868, 424 },
		{ 3129, 2921, 433 },
		{ -2265, 8, 385 },
		{ -2266, 4792, 0 },
		{ 2496, 865, 424 },
		{ 2501, 484, 424 },
		{ 2496, 870, 424 },
		{ 2345, 868, 424 },
		{ 2438, 876, 424 },
		{ 2444, 871, 424 },
		{ 2438, 878, 424 },
		{ 2378, 887, 424 },
		{ 2349, 877, 424 },
		{ 2404, 879, 424 },
		{ 2378, 901, 424 },
		{ 2461, 885, 424 },
		{ 2260, 878, 424 },
		{ 2406, 885, 424 },
		{ 2293, 899, 424 },
		{ 2260, 883, 424 },
		{ 2444, 920, 424 },
		{ 2260, 930, 424 },
		{ 2474, 923, 424 },
		{ 2496, 1190, 424 },
		{ 2474, 952, 424 },
		{ 2349, 955, 424 },
		{ 2501, 576, 424 },
		{ 3129, 1756, 420 },
		{ 2323, 1467, 0 },
		{ 3129, 1789, 421 },
		{ 2474, 965, 424 },
		{ 3098, 2282, 0 },
		{ 3051, 4753, 0 },
		{ 2260, 978, 424 },
		{ 3144, 1979, 0 },
		{ 2434, 976, 424 },
		{ 2334, 961, 424 },
		{ 2474, 969, 424 },
		{ 2406, 964, 424 },
		{ 2406, 965, 424 },
		{ 2349, 1000, 424 },
		{ 2434, 1008, 424 },
		{ 2434, 1009, 424 },
		{ 2461, 997, 424 },
		{ 2404, 995, 424 },
		{ 2434, 1049, 424 },
		{ 2378, 1054, 424 },
		{ 2461, 1038, 424 },
		{ 2501, 578, 424 },
		{ 2501, 580, 424 },
		{ 2468, 1039, 424 },
		{ 2468, 1040, 424 },
		{ 2434, 1055, 424 },
		{ 2334, 1066, 424 },
		{ 2444, 1073, 424 },
		{ 2378, 1088, 424 },
		{ 2422, 1086, 424 },
		{ 3139, 2456, 0 },
		{ 2353, 1400, 0 },
		{ 2323, 0, 0 },
		{ 3053, 2654, 422 },
		{ 2355, 1401, 0 },
		{ 3126, 2951, 0 },
		{ 0, 0, 387 },
		{ 2434, 1087, 424 },
		{ 2990, 2785, 0 },
		{ 2501, 582, 424 },
		{ 2349, 1118, 424 },
		{ 2406, 1111, 424 },
		{ 2501, 10, 424 },
		{ 2438, 1158, 424 },
		{ 2260, 1112, 424 },
		{ 2347, 1131, 424 },
		{ 2501, 123, 424 },
		{ 2406, 1115, 424 },
		{ 2438, 1153, 424 },
		{ 2501, 125, 424 },
		{ 2406, 1145, 424 },
		{ 2378, 1194, 424 },
		{ 3101, 2233, 0 },
		{ 3142, 1851, 0 },
		{ 2468, 1177, 424 },
		{ 2260, 1181, 424 },
		{ 2461, 1180, 424 },
		{ 2260, 1196, 424 },
		{ 2406, 1180, 424 },
		{ 2260, 1189, 424 },
		{ 2260, 1179, 424 },
		{ 3044, 2696, 0 },
		{ 2353, 0, 0 },
		{ 3053, 2707, 420 },
		{ 2355, 0, 0 },
		{ 3053, 2720, 421 },
		{ 0, 0, 425 },
		{ 2461, 1186, 424 },
		{ 2385, 4863, 0 },
		{ 3137, 2087, 0 },
		{ 2378, 1204, 424 },
		{ 2501, 130, 424 },
		{ 3107, 1909, 0 },
		{ 2525, 6, 424 },
		{ 2468, 1187, 424 },
		{ 2378, 1206, 424 },
		{ 2406, 1188, 424 },
		{ 3101, 1960, 0 },
		{ 2501, 235, 424 },
		{ 2404, 1188, 424 },
		{ 3144, 2019, 0 },
		{ 2438, 1202, 424 },
		{ 2406, 1192, 424 },
		{ 3098, 2317, 0 },
		{ 3098, 2319, 0 },
		{ 3146, 2125, 0 },
		{ 2444, 1198, 424 },
		{ 2461, 1196, 424 },
		{ 2260, 1214, 424 },
		{ 2434, 1211, 424 },
		{ 2434, 1212, 424 },
		{ 2501, 237, 424 },
		{ 2438, 1210, 424 },
		{ 3137, 2443, 0 },
		{ 2501, 239, 424 },
		{ 3139, 2182, 0 },
		{ 3044, 2690, 0 },
		{ 2406, 1200, 424 },
		{ 3107, 1915, 0 },
		{ 3142, 1963, 0 },
		{ 3147, 4505, 0 },
		{ 3101, 4926, 395 },
		{ 2496, 1208, 424 },
		{ 2406, 1202, 424 },
		{ 2438, 1218, 424 },
		{ 3144, 2177, 0 },
		{ 3139, 2472, 0 },
		{ 2438, 1215, 424 },
		{ 2990, 2793, 0 },
		{ 2444, 1210, 424 },
		{ 2438, 1217, 424 },
		{ 3044, 2725, 0 },
		{ 3044, 2727, 0 },
		{ 3129, 2918, 0 },
		{ 2260, 1206, 424 },
		{ 2438, 1220, 424 },
		{ 2260, 1210, 424 },
		{ 2501, 241, 424 },
		{ 2501, 243, 424 },
		{ 3146, 1936, 0 },
		{ 2474, 1218, 424 },
		{ 3129, 2806, 0 },
		{ 3144, 2011, 0 },
		{ 3090, 3691, 0 },
		{ 3044, 2571, 0 },
		{ 3131, 2405, 0 },
		{ 2438, 1224, 424 },
		{ 3142, 1838, 0 },
		{ 3140, 3049, 0 },
		{ 2525, 121, 424 },
		{ 2444, 1220, 424 },
		{ 2444, 1221, 424 },
		{ 2260, 1233, 424 },
		{ 2904, 2143, 0 },
		{ 3146, 2055, 0 },
		{ 2474, 1224, 424 },
		{ 2455, 4826, 0 },
		{ 2474, 1225, 424 },
		{ 2444, 1225, 424 },
		{ 3142, 1901, 0 },
		{ 3142, 1917, 0 },
		{ 3129, 2837, 0 },
		{ 2434, 1236, 424 },
		{ 2474, 1228, 424 },
		{ 2260, 1238, 424 },
		{ 3144, 1786, 0 },
		{ 3101, 2237, 0 },
		{ 3129, 2850, 0 },
		{ 2260, 1235, 424 },
		{ 3147, 4690, 0 },
		{ 2990, 2772, 0 },
		{ 3094, 3323, 0 },
		{ 3142, 1944, 0 },
		{ 3044, 2707, 0 },
		{ 2260, 1230, 424 },
		{ 3140, 3029, 0 },
		{ 3142, 1949, 0 },
		{ 3147, 4548, 0 },
		{ 0, 0, 413 },
		{ 2461, 1228, 424 },
		{ 2474, 1233, 424 },
		{ 2501, 245, 424 },
		{ 3134, 1623, 0 },
		{ 3144, 2167, 0 },
		{ 2462, 1242, 424 },
		{ 3101, 1964, 0 },
		{ 2501, 349, 424 },
		{ 2474, 1237, 424 },
		{ 2486, 4771, 0 },
		{ 2487, 4787, 0 },
		{ 2488, 4814, 0 },
		{ 2260, 1234, 424 },
		{ 2260, 1246, 424 },
		{ 2501, 351, 424 },
		{ 3140, 3060, 0 },
		{ 2990, 2788, 0 },
		{ 3107, 2023, 0 },
		{ 3126, 2936, 0 },
		{ 2260, 1236, 424 },
		{ 3101, 4938, 418 },
		{ 2497, 4840, 0 },
		{ 3107, 2025, 0 },
		{ 3098, 2255, 0 },
		{ 3142, 1747, 0 },
		{ 2260, 1242, 424 },
		{ 3142, 1798, 0 },
		{ 3107, 2036, 0 },
		{ 2501, 355, 424 },
		{ 2501, 357, 424 },
		{ 3101, 2353, 0 },
		{ 3137, 2469, 0 },
		{ 3131, 2382, 0 },
		{ 2501, 370, 424 },
		{ 3146, 2061, 0 },
		{ 3101, 1968, 0 },
		{ 2501, 463, 424 },
		{ 3142, 1855, 0 },
		{ 3142, 1903, 0 },
		{ 3126, 2601, 0 },
		{ 2501, 469, 424 },
		{ 2501, 471, 424 },
		{ 3141, 1953, 0 },
		{ 3146, 2074, 0 },
		{ 2990, 2790, 0 },
		{ 3137, 2489, 0 },
		{ 3134, 1518, 0 },
		{ 2501, 1499, 424 },
		{ 3144, 1907, 0 },
		{ 2528, 4913, 0 },
		{ 3129, 2810, 0 },
		{ 3147, 4573, 0 },
		{ 2525, 574, 424 },
		{ 3107, 1978, 0 },
		{ 3147, 4583, 0 },
		{ 3101, 2351, 0 },
		{ 3144, 1984, 0 },
		{ 3129, 2819, 0 },
		{ 3140, 2996, 0 },
		{ 2538, 4821, 0 },
		{ 3144, 1784, 0 },
		{ 3144, 2226, 0 },
		{ 3146, 2087, 0 },
		{ 3101, 1974, 0 },
		{ 3146, 2091, 0 },
		{ 3146, 2092, 0 },
		{ 3129, 2829, 0 },
		{ 3101, 1955, 0 },
		{ 3107, 1911, 0 },
		{ 3107, 1991, 0 },
		{ 3098, 2323, 0 },
		{ 2551, 4846, 0 },
		{ 3129, 2836, 0 },
		{ 3107, 1993, 0 },
		{ 3140, 3018, 0 },
		{ 3141, 2973, 0 },
		{ 2556, 811, 424 },
		{ 3129, 2840, 0 },
		{ 2904, 2137, 0 },
		{ 3147, 4692, 0 },
		{ 3107, 1996, 0 },
		{ 3101, 4870, 393 },
		{ 3107, 1917, 0 },
		{ 3147, 4441, 0 },
		{ 3101, 4896, 402 },
		{ 3144, 2171, 0 },
		{ 2904, 2144, 0 },
		{ 3098, 2337, 0 },
		{ 3142, 1945, 0 },
		{ 3139, 2476, 0 },
		{ 3140, 3040, 0 },
		{ 2990, 2794, 0 },
		{ 2946, 2759, 0 },
		{ 3144, 2176, 0 },
		{ 3146, 2105, 0 },
		{ 3129, 2861, 0 },
		{ 3129, 2862, 0 },
		{ 2904, 2154, 0 },
		{ 3146, 2106, 0 },
		{ 3044, 2661, 0 },
		{ 3111, 1412, 0 },
		{ 3134, 1488, 0 },
		{ 3107, 1919, 0 },
		{ 3098, 2254, 0 },
		{ 2904, 2138, 0 },
		{ 3098, 2257, 0 },
		{ 3129, 2878, 0 },
		{ 3147, 4503, 0 },
		{ 3101, 4868, 416 },
		{ 3098, 2258, 0 },
		{ 3142, 1952, 0 },
		{ 0, 0, 430 },
		{ 3107, 2006, 0 },
		{ 3044, 2702, 0 },
		{ 3101, 4883, 401 },
		{ 3140, 3009, 0 },
		{ 3129, 2886, 0 },
		{ 3044, 2703, 0 },
		{ 3044, 2704, 0 },
		{ 3044, 2705, 0 },
		{ 3146, 2116, 0 },
		{ 2990, 2780, 0 },
		{ 2597, 4835, 0 },
		{ 2655, 2989, 0 },
		{ 3144, 2190, 0 },
		{ 3129, 2899, 0 },
		{ 3129, 2900, 0 },
		{ 3142, 1955, 0 },
		{ 3144, 2193, 0 },
		{ 2589, 1279, 0 },
		{ 2605, 4842, 0 },
		{ 2904, 2148, 0 },
		{ 3141, 2968, 0 },
		{ 3142, 1958, 0 },
		{ 3146, 2126, 0 },
		{ 3126, 2931, 0 },
		{ 2611, 4858, 0 },
		{ 3129, 2910, 0 },
		{ 3044, 2723, 0 },
		{ 2614, 4885, 0 },
		{ 0, 1326, 0 },
		{ 3137, 2459, 0 },
		{ 3146, 2057, 0 },
		{ 3142, 1965, 0 },
		{ 3144, 2206, 0 },
		{ 3137, 2471, 0 },
		{ 3129, 2922, 0 },
		{ 3107, 2016, 0 },
		{ 3101, 2772, 0 },
		{ 3140, 2993, 0 },
		{ 2655, 2981, 0 },
		{ 2626, 4810, 0 },
		{ 2627, 4812, 0 },
		{ 3136, 2740, 0 },
		{ 2655, 2984, 0 },
		{ 3129, 2926, 0 },
		{ 3107, 1921, 0 },
		{ 3137, 2474, 0 },
		{ 3146, 2063, 0 },
		{ 3107, 2018, 0 },
		{ 3044, 2575, 0 },
		{ 2636, 4828, 0 },
		{ 3144, 1999, 0 },
		{ 3146, 2068, 0 },
		{ 3131, 2401, 0 },
		{ 3141, 2715, 0 },
		{ 3129, 2813, 0 },
		{ 3147, 4591, 0 },
		{ 3140, 3015, 0 },
		{ 3144, 2216, 0 },
		{ 3098, 2299, 0 },
		{ 3129, 2817, 0 },
		{ 3098, 2301, 0 },
		{ 2904, 2147, 0 },
		{ 3134, 1523, 0 },
		{ 2655, 2986, 0 },
		{ 3140, 3026, 0 },
		{ 3126, 2595, 0 },
		{ 3126, 2597, 0 },
		{ 2654, 4799, 0 },
		{ 3140, 3032, 0 },
		{ 3147, 4507, 0 },
		{ 3142, 1665, 0 },
		{ 3144, 2221, 0 },
		{ 3044, 2682, 0 },
		{ 2660, 4774, 0 },
		{ 3098, 2309, 0 },
		{ 3131, 2037, 0 },
		{ 2904, 2159, 0 },
		{ 3140, 3042, 0 },
		{ 3044, 2687, 0 },
		{ 3140, 3045, 0 },
		{ 3147, 4645, 0 },
		{ 3101, 4908, 414 },
		{ 3142, 1682, 0 },
		{ 3146, 2073, 0 },
		{ 3147, 4659, 0 },
		{ 3147, 4665, 0 },
		{ 3142, 1683, 0 },
		{ 3146, 2076, 0 },
		{ 2990, 2779, 0 },
		{ 3044, 2698, 0 },
		{ 2655, 2983, 0 },
		{ 3129, 2844, 0 },
		{ 3129, 2845, 0 },
		{ 3147, 4474, 0 },
		{ 0, 2985, 0 },
		{ 3101, 4940, 400 },
		{ 3140, 2994, 0 },
		{ 3142, 1684, 0 },
		{ 2904, 2142, 0 },
		{ 3144, 2005, 0 },
		{ 2946, 2763, 0 },
		{ 3144, 2239, 0 },
		{ 3129, 2853, 0 },
		{ 3142, 1701, 0 },
		{ 3107, 2030, 0 },
		{ 3107, 2032, 0 },
		{ 3101, 4833, 394 },
		{ 3144, 2166, 0 },
		{ 3107, 2033, 0 },
		{ 3101, 4842, 406 },
		{ 3101, 4844, 407 },
		{ 3107, 2034, 0 },
		{ 3044, 2714, 0 },
		{ 2990, 2774, 0 },
		{ 3137, 2463, 0 },
		{ 3044, 2718, 0 },
		{ 2904, 2150, 0 },
		{ 2904, 2151, 0 },
		{ 0, 0, 429 },
		{ 3044, 2722, 0 },
		{ 3142, 1703, 0 },
		{ 2702, 4781, 0 },
		{ 3142, 1704, 0 },
		{ 2904, 2158, 0 },
		{ 2705, 4797, 0 },
		{ 3126, 2935, 0 },
		{ 3146, 2089, 0 },
		{ 3044, 2729, 0 },
		{ 3140, 3031, 0 },
		{ 3129, 2880, 0 },
		{ 3146, 2090, 0 },
		{ 3147, 4509, 0 },
		{ 3147, 4513, 0 },
		{ 3098, 2351, 0 },
		{ 3129, 2884, 0 },
		{ 3044, 2734, 0 },
		{ 3137, 2478, 0 },
		{ 3142, 1705, 0 },
		{ 3142, 1706, 0 },
		{ 3137, 2483, 0 },
		{ 3107, 2039, 0 },
		{ 3107, 1923, 0 },
		{ 3147, 4593, 0 },
		{ 3129, 2895, 0 },
		{ 3144, 2013, 0 },
		{ 3129, 2897, 0 },
		{ 3140, 3054, 0 },
		{ 3144, 2186, 0 },
		{ 3142, 1713, 0 },
		{ 3107, 2043, 0 },
		{ 3147, 4429, 0 },
		{ 3146, 939, 397 },
		{ 3101, 4966, 409 },
		{ 2946, 2758, 0 },
		{ 3146, 2101, 0 },
		{ 3142, 1715, 0 },
		{ 3044, 2584, 0 },
		{ 3136, 2742, 0 },
		{ 3136, 2744, 0 },
		{ 3044, 2587, 0 },
		{ 2739, 4756, 0 },
		{ 3141, 2966, 0 },
		{ 3101, 4852, 405 },
		{ 3146, 2103, 0 },
		{ 2904, 2149, 0 },
		{ 3137, 2432, 0 },
		{ 3142, 1716, 0 },
		{ 3098, 2274, 0 },
		{ 3044, 2647, 0 },
		{ 2747, 4830, 0 },
		{ 3101, 4881, 396 },
		{ 3147, 4575, 0 },
		{ 2749, 4836, 0 },
		{ 2754, 1407, 0 },
		{ 3142, 1718, 0 },
		{ 2752, 4841, 0 },
		{ 2753, 4843, 0 },
		{ 3142, 1743, 0 },
		{ 3139, 2458, 0 },
		{ 3146, 2108, 0 },
		{ 3140, 3017, 0 },
		{ 3129, 2929, 0 },
		{ 3147, 4620, 0 },
		{ 3144, 2202, 0 },
		{ 3107, 2053, 0 },
		{ 3144, 2205, 0 },
		{ 3147, 4651, 0 },
		{ 3101, 4922, 411 },
		{ 3147, 4655, 0 },
		{ 3147, 4657, 0 },
		{ 2754, 1408, 0 },
		{ 3147, 4661, 0 },
		{ 3147, 4663, 0 },
		{ 0, 1360, 0 },
		{ 3044, 2692, 0 },
		{ 3044, 2693, 0 },
		{ 3142, 1753, 0 },
		{ 3146, 2113, 0 },
		{ 3101, 4946, 417 },
		{ 3146, 2115, 0 },
		{ 3147, 4439, 0 },
		{ 3098, 2297, 0 },
		{ 0, 0, 432 },
		{ 0, 0, 431 },
		{ 3101, 4953, 398 },
		{ 3147, 4443, 0 },
		{ 0, 0, 428 },
		{ 0, 0, 427 },
		{ 3147, 4445, 0 },
		{ 3137, 2467, 0 },
		{ 2904, 2140, 0 },
		{ 3144, 2214, 0 },
		{ 3140, 3039, 0 },
		{ 3147, 4501, 0 },
		{ 3101, 4970, 392 },
		{ 2784, 4876, 0 },
		{ 3101, 4973, 419 },
		{ 3101, 4977, 399 },
		{ 3129, 2823, 0 },
		{ 3142, 1764, 0 },
		{ 3146, 2117, 0 },
		{ 3142, 1765, 0 },
		{ 3101, 4839, 412 },
		{ 3101, 2235, 0 },
		{ 3147, 4515, 0 },
		{ 3147, 4517, 0 },
		{ 3147, 4521, 0 },
		{ 3144, 2219, 0 },
		{ 3142, 1768, 0 },
		{ 3101, 4854, 403 },
		{ 3101, 4857, 404 },
		{ 3101, 4863, 408 },
		{ 3146, 2121, 0 },
		{ 3129, 2832, 0 },
		{ 3147, 4577, 0 },
		{ 3146, 2123, 0 },
		{ 3101, 4872, 410 },
		{ 3140, 3055, 0 },
		{ 3142, 1778, 0 },
		{ 3044, 2719, 0 },
		{ 3144, 2225, 0 },
		{ 3098, 2316, 0 },
		{ 3107, 1986, 0 },
		{ 3147, 4618, 0 },
		{ 3101, 4888, 415 },
		{ 3051, 4750, 433 },
		{ 2811, 0, 389 },
		{ 0, 0, 390 },
		{ -2809, 4982, 385 },
		{ -2810, 4794, 0 },
		{ 3101, 4771, 0 },
		{ 3051, 4759, 0 },
		{ 0, 0, 386 },
		{ 3051, 4769, 0 },
		{ -2815, 7, 0 },
		{ -2816, 4798, 0 },
		{ 2819, 0, 387 },
		{ 3051, 4770, 0 },
		{ 3101, 4912, 0 },
		{ 0, 0, 388 },
		{ 3094, 3337, 152 },
		{ 0, 0, 152 },
		{ 0, 0, 153 },
		{ 3107, 1988, 0 },
		{ 3129, 2842, 0 },
		{ 3146, 2059, 0 },
		{ 2828, 4825, 0 },
		{ 3139, 2474, 0 },
		{ 3134, 1627, 0 },
		{ 3098, 2324, 0 },
		{ 3141, 2977, 0 },
		{ 3142, 1818, 0 },
		{ 3044, 2733, 0 },
		{ 3144, 2237, 0 },
		{ 3098, 2328, 0 },
		{ 3107, 1992, 0 },
		{ 3147, 4433, 0 },
		{ 0, 0, 150 },
		{ 2977, 4579, 175 },
		{ 0, 0, 175 },
		{ 3142, 1821, 0 },
		{ 2843, 4859, 0 },
		{ 3129, 2520, 0 },
		{ 3140, 3013, 0 },
		{ 3141, 2967, 0 },
		{ 3136, 2737, 0 },
		{ 2848, 4864, 0 },
		{ 3101, 2347, 0 },
		{ 3129, 2858, 0 },
		{ 3098, 2334, 0 },
		{ 3129, 2860, 0 },
		{ 3146, 2064, 0 },
		{ 3140, 3022, 0 },
		{ 3142, 1824, 0 },
		{ 3044, 2576, 0 },
		{ 3144, 2163, 0 },
		{ 3098, 2339, 0 },
		{ 2859, 4743, 0 },
		{ 3101, 2774, 0 },
		{ 3129, 2869, 0 },
		{ 2990, 2786, 0 },
		{ 3144, 2164, 0 },
		{ 3146, 2066, 0 },
		{ 3129, 2873, 0 },
		{ 2866, 4742, 0 },
		{ 3146, 1948, 0 },
		{ 3129, 2875, 0 },
		{ 3126, 2938, 0 },
		{ 3134, 1645, 0 },
		{ 3141, 2956, 0 },
		{ 3129, 2877, 0 },
		{ 2873, 4767, 0 },
		{ 3139, 2460, 0 },
		{ 3134, 1487, 0 },
		{ 3098, 2349, 0 },
		{ 3141, 2965, 0 },
		{ 3142, 1834, 0 },
		{ 3044, 2644, 0 },
		{ 3144, 2170, 0 },
		{ 3098, 2352, 0 },
		{ 3147, 4653, 0 },
		{ 0, 0, 173 },
		{ 2884, 0, 1 },
		{ -2884, 1273, 264 },
		{ 3129, 2782, 270 },
		{ 0, 0, 270 },
		{ 3107, 2005, 0 },
		{ 3098, 2256, 0 },
		{ 3129, 2894, 0 },
		{ 3126, 2943, 0 },
		{ 3146, 2075, 0 },
		{ 0, 0, 269 },
		{ 2894, 4832, 0 },
		{ 3131, 1950, 0 },
		{ 3140, 3001, 0 },
		{ 2923, 2505, 0 },
		{ 3129, 2898, 0 },
		{ 2990, 2789, 0 },
		{ 3044, 2688, 0 },
		{ 3137, 2481, 0 },
		{ 3129, 2903, 0 },
		{ 2903, 4813, 0 },
		{ 3144, 2031, 0 },
		{ 0, 2152, 0 },
		{ 3142, 1863, 0 },
		{ 3044, 2694, 0 },
		{ 3144, 2182, 0 },
		{ 3098, 2263, 0 },
		{ 3107, 2011, 0 },
		{ 3147, 4511, 0 },
		{ 0, 0, 268 },
		{ 0, 4365, 178 },
		{ 0, 0, 178 },
		{ 3144, 2184, 0 },
		{ 3134, 1520, 0 },
		{ 3098, 2268, 0 },
		{ 3126, 2946, 0 },
		{ 2919, 4854, 0 },
		{ 3141, 2655, 0 },
		{ 3136, 2739, 0 },
		{ 3129, 2919, 0 },
		{ 3141, 2972, 0 },
		{ 0, 2507, 0 },
		{ 3044, 2706, 0 },
		{ 3098, 2270, 0 },
		{ 2946, 2753, 0 },
		{ 3147, 4589, 0 },
		{ 0, 0, 176 },
		{ 2977, 4588, 172 },
		{ 0, 0, 171 },
		{ 0, 0, 172 },
		{ 3142, 1870, 0 },
		{ 2934, 4857, 0 },
		{ 3142, 1879, 0 },
		{ 3136, 2748, 0 },
		{ 3129, 2928, 0 },
		{ 2938, 4879, 0 },
		{ 3101, 2770, 0 },
		{ 3129, 2801, 0 },
		{ 2946, 2760, 0 },
		{ 3044, 2712, 0 },
		{ 3098, 2276, 0 },
		{ 3098, 2277, 0 },
		{ 3044, 2716, 0 },
		{ 3098, 2278, 0 },
		{ 0, 2756, 0 },
		{ 2948, 4739, 0 },
		{ 3144, 1981, 0 },
		{ 2990, 2778, 0 },
		{ 2951, 4754, 0 },
		{ 3129, 2518, 0 },
		{ 3140, 3052, 0 },
		{ 3141, 2954, 0 },
		{ 3136, 2746, 0 },
		{ 2956, 4757, 0 },
		{ 3101, 2345, 0 },
		{ 3129, 2818, 0 },
		{ 3098, 2281, 0 },
		{ 3129, 2820, 0 },
		{ 3146, 2086, 0 },
		{ 3140, 3063, 0 },
		{ 3142, 1888, 0 },
		{ 3044, 2724, 0 },
		{ 3144, 2191, 0 },
		{ 3098, 2285, 0 },
		{ 2967, 4781, 0 },
		{ 3139, 2468, 0 },
		{ 3134, 1555, 0 },
		{ 3098, 2287, 0 },
		{ 3141, 2974, 0 },
		{ 3142, 1891, 0 },
		{ 3044, 2732, 0 },
		{ 3144, 2194, 0 },
		{ 3098, 2290, 0 },
		{ 3147, 4519, 0 },
		{ 0, 0, 165 },
		{ 0, 4473, 164 },
		{ 0, 0, 164 },
		{ 3142, 1896, 0 },
		{ 2981, 4791, 0 },
		{ 3142, 1883, 0 },
		{ 3136, 2738, 0 },
		{ 3129, 2838, 0 },
		{ 2985, 4810, 0 },
		{ 3129, 2524, 0 },
		{ 3098, 2293, 0 },
		{ 3126, 2939, 0 },
		{ 2989, 4805, 0 },
		{ 3144, 1993, 0 },
		{ 0, 2783, 0 },
		{ 2992, 4818, 0 },
		{ 3129, 2514, 0 },
		{ 3140, 3019, 0 },
		{ 3141, 2969, 0 },
		{ 3136, 2743, 0 },
		{ 2997, 4823, 0 },
		{ 3101, 2349, 0 },
		{ 3129, 2846, 0 },
		{ 3098, 2296, 0 },
		{ 3129, 2848, 0 },
		{ 3146, 2093, 0 },
		{ 3140, 3027, 0 },
		{ 3142, 1923, 0 },
		{ 3044, 2578, 0 },
		{ 3144, 2200, 0 },
		{ 3098, 2300, 0 },
		{ 3008, 4840, 0 },
		{ 3139, 2462, 0 },
		{ 3134, 1570, 0 },
		{ 3098, 2302, 0 },
		{ 3141, 2960, 0 },
		{ 3142, 1928, 0 },
		{ 3044, 2588, 0 },
		{ 3144, 2203, 0 },
		{ 3098, 2305, 0 },
		{ 3147, 4435, 0 },
		{ 0, 0, 162 },
		{ 0, 3998, 167 },
		{ 0, 0, 167 },
		{ 0, 0, 168 },
		{ 3098, 2306, 0 },
		{ 3107, 2026, 0 },
		{ 3142, 1932, 0 },
		{ 3129, 2864, 0 },
		{ 3140, 3048, 0 },
		{ 3126, 2948, 0 },
		{ 3028, 4873, 0 },
		{ 3129, 2512, 0 },
		{ 3111, 1445, 0 },
		{ 3140, 3053, 0 },
		{ 3137, 2494, 0 },
		{ 3134, 1575, 0 },
		{ 3140, 3056, 0 },
		{ 3142, 1938, 0 },
		{ 3044, 2681, 0 },
		{ 3144, 2209, 0 },
		{ 3098, 2313, 0 },
		{ 3039, 4742, 0 },
		{ 3139, 2470, 0 },
		{ 3134, 1608, 0 },
		{ 3098, 2315, 0 },
		{ 3141, 2962, 0 },
		{ 3142, 1941, 0 },
		{ 0, 2689, 0 },
		{ 3144, 2212, 0 },
		{ 3098, 2318, 0 },
		{ 3109, 4698, 0 },
		{ 0, 0, 166 },
		{ 3129, 2882, 433 },
		{ 3146, 1467, 25 },
		{ 0, 4761, 433 },
		{ 3060, 0, 433 },
		{ 2259, 2678, 28 },
		{ 0, 0, 29 },
		{ 0, 0, 30 },
		{ 0, 0, 31 },
		{ 0, 0, 32 },
		{ 3098, 2322, 0 },
		{ -3059, 4983, 0 },
		{ 3146, 661, 0 },
		{ 0, 0, 27 },
		{ 3126, 2952, 0 },
		{ 0, 0, 26 },
		{ 0, 0, 21 },
		{ 0, 0, 33 },
		{ 0, 0, 34 },
		{ 0, 3817, 38 },
		{ 0, 3549, 38 },
		{ 0, 0, 37 },
		{ 0, 0, 38 },
		{ 3090, 3687, 0 },
		{ 3102, 4343, 0 },
		{ 3094, 3328, 0 },
		{ 0, 0, 36 },
		{ 3097, 3391, 0 },
		{ 0, 3231, 0 },
		{ 3053, 1655, 0 },
		{ 0, 0, 35 },
		{ 3129, 2798, 49 },
		{ 0, 0, 49 },
		{ 3094, 3334, 49 },
		{ 3129, 2892, 49 },
		{ 0, 0, 52 },
		{ 3129, 2893, 0 },
		{ 3098, 2327, 0 },
		{ 3097, 3401, 0 },
		{ 3142, 1954, 0 },
		{ 3098, 2329, 0 },
		{ 3126, 2941, 0 },
		{ 0, 3650, 0 },
		{ 3134, 1644, 0 },
		{ 3144, 2222, 0 },
		{ 0, 0, 48 },
		{ 0, 3344, 0 },
		{ 3146, 2114, 0 },
		{ 3131, 2386, 0 },
		{ 0, 3411, 0 },
		{ 0, 2333, 0 },
		{ 3129, 2902, 0 },
		{ 0, 0, 50 },
		{ 0, 5, 53 },
		{ 0, 4313, 0 },
		{ 0, 0, 51 },
		{ 3137, 2476, 0 },
		{ 3140, 3028, 0 },
		{ 3107, 2044, 0 },
		{ 0, 2045, 0 },
		{ 3109, 4702, 0 },
		{ 0, 4703, 0 },
		{ 3129, 2906, 0 },
		{ 0, 1484, 0 },
		{ 3140, 3033, 0 },
		{ 3137, 2480, 0 },
		{ 3134, 1646, 0 },
		{ 3140, 3036, 0 },
		{ 3142, 1959, 0 },
		{ 3144, 2232, 0 },
		{ 3146, 2120, 0 },
		{ 3140, 2068, 0 },
		{ 3129, 2915, 0 },
		{ 3144, 2235, 0 },
		{ 3141, 2961, 0 },
		{ 3140, 3044, 0 },
		{ 3146, 2122, 0 },
		{ 3141, 2963, 0 },
		{ 0, 2940, 0 },
		{ 3129, 2822, 0 },
		{ 3134, 1647, 0 },
		{ 0, 2920, 0 },
		{ 3140, 3051, 0 },
		{ 0, 2406, 0 },
		{ 3146, 2124, 0 },
		{ 3141, 2970, 0 },
		{ 0, 1648, 0 },
		{ 3147, 4707, 0 },
		{ 0, 2745, 0 },
		{ 0, 2492, 0 },
		{ 0, 0, 45 },
		{ 3101, 2756, 0 },
		{ 0, 3059, 0 },
		{ 0, 2975, 0 },
		{ 0, 1966, 0 },
		{ 3147, 4709, 0 },
		{ 0, 2241, 0 },
		{ 0, 0, 46 },
		{ 0, 2127, 0 },
		{ 3109, 4711, 0 },
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
