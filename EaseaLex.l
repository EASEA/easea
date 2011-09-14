%{
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
  %}

// include file
%include {
  // forward references
  class CEASEAParser;
  class CSymbolTable;
  class OPCodeDesc;
 }                                                     
                                         
%start GENOME_ANALYSIS TEMPLATE_ANALYSIS MACRO_IDENTIFIER MACRO_DEFINITION 
%start COPY_USER_DECLARATIONS COPY_USER_CUDA COPY_INITIALISATION_FUNCTION ANALYSE_USER_CLASSES
%start COPY_EO_INITIALISER COPY_GP_OPCODE
%start COPY COPY_INITIALISER COPY_CROSSOVER COPY_MUTATOR COPY_EVALUATOR COPY_OPTIMISER COPY_FINALIZATION_FUNCTION
%start COPY_DISPLAY COPY_USER_FUNCTION COPY_USER_GENERATION PARAMETERS_ANALYSIS GET_PARAMETERS 
%start COPY_USER_FUNCTIONS COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT GET_METHODS COPY_MAKEFILE_OPTION COPY_BOUND_CHECKING_FUNCTION COPY_BEG_GENERATION_FUNCTION COPY_END_GENERATION_FUNCTION COPY_INSTEAD_EVAL GP_RULE_ANALYSIS GP_COPY_OPCODE_CODE COPY_GP_EVAL

 // lexical analyser name and class definition
%name CEASEALexer {
 protected:
  CSymbolTable *pSymbolTable;   // the symbol table
  bool bSymbolInserted,bWithinEvaluator, bWithinOptimiser;  // used to change evalutor type from double to float 
  bool bInitFunction,bDisplayFunction,bFunction, bNotFinishedYet, bWithinEO_Function;
  bool bDoubleQuotes,bWithinDisplayFunction,bWithinInitialiser,bWithinMutator,bWithinXover;
  bool bWaitingForSemiColon,bFinishNB_GEN,bFinishMINIMISE,bFinishMINIMIZE,bGenerationReplacementFunction;
  bool bCatchNextSemiColon,bWaitingToClosePopulation, bMethodsInGenome, bFinalizationFunction;
  bool bWithinCUDA_Initializer, bWithinMAKEFILEOPTION, bWithinCUDA_Evaluator, bBoundCheckingFunction;
  bool bIsParentReduce, bIsOffspringReduce, bEndGeneration, bBeginGeneration, bEndGenerationFunction, bBeginGenerationFunction, bGenerationFunctionBeforeReplacement;
  bool bGPOPCODE_ANALYSIS,bCOPY_GP_EVAL_GPU;
  CSymbol *pASymbol;

  unsigned iGP_OPCODE_FIELD, accolade_counter;
  OPCodeDesc* opDesc[128];
  unsigned iNoOp;

  enum COPY_GP_EVAL_STATUS {EVAL_HDR,EVAL_BDY,EVAL_FTR};
  unsigned iCOPY_GP_EVAL_STATUS;
  bool bIsCopyingGPEval;

 public:
  int create(CEASEAParser* pParser, CSymbolTable* pSymbolTable);
  int yywrap();
  double myStrtod() const;                              
 }

// constructor
{                                
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
}

// macros
exponent  ([Ee][+-]?[0-9]+)



%%                                              

%{
  // extract yylval for use later on in actions
  YYSTYPE& yylval = *(YYSTYPE*)yyparserptr->yylvalptr;
  %}

// Gobbles up ^Ms, to be compatible with unix
\r {}

// switch immediately to TEMPLATE_ANALYSIS state
<INITIAL>.    {
  BEGIN TEMPLATE_ANALYSIS; yyless(yyleng-1);
 }

//****************************************
//  GA_GENERATION lexer (feeds on template file and occasionally on .ez file)
//****************************************

<TEMPLATE_ANALYSIS>"\\TEMPLATE_START"  {             
  char sFileName[1000];
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".cpp"); 
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"AESAE" {fprintf(fpOutputFile,"EASEA");}        
<TEMPLATE_ANALYSIS>"EASEA" {fprintf(fpOutputFile,"%s",sPROJECT_NAME);}        
<TEMPLATE_ANALYSIS>"\\EZ_PATH" {fprintf(fpOutputFile,"%s",sEZ_PATH);}        
<TEMPLATE_ANALYSIS>"\\TPL_DIR" {fprintf(fpOutputFile,"%s",sTPL_DIR);}
<TEMPLATE_ANALYSIS>"\\EO_DIR" {fprintf(fpOutputFile,"%s",sEO_DIR);}        
<TEMPLATE_ANALYSIS>"LC_EASEA" {fprintf(fpOutputFile,"%s",sLOWER_CASE_PROJECT_NAME);}        
<TEMPLATE_ANALYSIS>"\\OPERATING_SYSTEM"  {switch (OPERATING_SYSTEM) {
  case UNIX : fprintf(fpOutputFile,"UNIX_OS"); break;
  case WINDOWS : fprintf(fpOutputFile,"WINDOWS_OS"); break;
  case UNKNOWN_OS : fprintf(fpOutputFile,"UNKNOWN_OS"); break;
  }
 }
<TEMPLATE_ANALYSIS>"\\INSERT_USER_DECLARATIONS"  {
  if (bVERBOSE) printf ("Inserting user declarations.\n");
  yyreset();
  yyin = fpGenomeFile;                                                    // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_USER_DECLARATIONS;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_USER_CUDA"  {
  if (bVERBOSE) printf ("Inserting user cuda.\n");
  yyreset();
  yyin = fpGenomeFile;                                                    // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_USER_CUDA;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_INITIALISATION_FUNCTION"  {
  if (bVERBOSE) printf ("Inserting initialisation function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISATION_FUNCTION;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_GENERATION_FUNCTION_BEFORE_REPLACEMENT"  {
  if (bVERBOSE) printf ("Inserting generation before reduce function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bGenerationFunctionBeforeReplacement = true;
  BEGIN COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT;
 }


<TEMPLATE_ANALYSIS>"\\INSERT_BEGIN_GENERATION_FUNCTION"  {
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
 }



/*
  Here we will insert parmeters for GP template.
*/
<TEMPLATE_ANALYSIS>"\\INSERT_GP_PARAMETERS" {
  if( bVERBOSE )printf("inserting gp parameters\n");
  //  fprintf(fpOutputFile,"#define MAX_XOVER_DEPTH",%d
  fprintf(fpOutputFile,"#define TREE_DEPTH_MAX %d\n",iMAX_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MAX %d\n",iMAX_INIT_TREE_D);
  fprintf(fpOutputFile,"#define INIT_TREE_DEPTH_MIN %d\n",iMIN_INIT_TREE_D);

  fprintf(fpOutputFile,"#define MAX_PROGS_SIZE %d\n",iPRG_BUF_SIZE);
  fprintf(fpOutputFile,"#define NB_GPU %d\n",iNB_GPU);

  fprintf(fpOutputFile,"#define NO_FITNESS_CASES %d\n",iNO_FITNESS_CASES);
}

/*
  This part dump every declaration related to gp opcode.
  the enumeration, the array of name (to plot trees)
  and the arity of every opcode.
 */
<TEMPLATE_ANALYSIS>"\\INSERT_GP_OPCODE_DECL" {
  
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
 }

<TEMPLATE_ANALYSIS>"\\INSERT_GP_GPU_SWITCH" {
  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"    case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"      %s",opDesc[i]->gpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"      break;\n");

  }
 }

<TEMPLATE_ANALYSIS>"\\INSERT_GP_CPU_SWITCH" {
  for( unsigned i=0 ; i<iNoOp ; i++ ){
    fprintf(fpOutputFile,"  case %s :\n",opDesc[i]->opcode->c_str());
    fprintf(fpOutputFile,"    %s\n",opDesc[i]->cpuCodeStream.str().c_str());
    fprintf(fpOutputFile,"    break;\n");
  }
 }


<TEMPLATE_ANALYSIS>"\\ANALYSE_GP_OPCODE"  {
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Analysing GP OP code from ez file\n");
  BEGIN COPY_GP_OPCODE;
 }


<COPY_GP_OPCODE>"\\begin operator description :" {
  if (bVERBOSE) printf ("found begin section\n");
  bGPOPCODE_ANALYSIS = true;
  BEGIN GP_RULE_ANALYSIS;
 }

<GP_RULE_ANALYSIS>"\\end" { 
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
 }

<COPY_GP_OPCODE><<EOF>> {
  if (bVERBOSE) printf("*** No GP OP codes were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
 }
<COPY_GP_OPCODE>. {} //if( bGPOPCODE_ANALYSIS ) printf("%c",yytext[0]);}                                      
<COPY_GP_OPCODE>\n {if( bGPOPCODE_ANALYSIS )printf("\n");lineCounter++;}

/*
  This section analyse a GP rule which is written as : 
  OP_NAME, "realopname", arity (a number), { the c code for this operator};
  OP_NAME, "realopname", arity (a number), { the c code for this operator};
 */
<GP_RULE_ANALYSIS>[A-Za-z_][a-zA-Z0-9_]* {
  // this rule match the OP_NAME
  if( iGP_OPCODE_FIELD != 0 ){
    fprintf(stderr,"Error, OP_CODE name must be given first\n");
    exit(-1);
  }
  opDesc[iNoOp] = new OPCodeDesc();
  opDesc[iNoOp]->opcode = new string(yytext);
 }

<GP_RULE_ANALYSIS>"//".*"\n" {}

<GP_RULE_ANALYSIS>"\""([a-zA-Z0-9]|\*|\+|\-|\/|\%)*"\"" {
  if( iGP_OPCODE_FIELD != 1 ){
    fprintf(stderr,"Error, op code real name must be given at the second place\n");
    exit(-1);
  }
  opDesc[iNoOp]->realName = new string(yytext);
 }


<GP_RULE_ANALYSIS>[0-9]* {
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
 }

<GP_RULE_ANALYSIS>[ \t\n] {}
<GP_RULE_ANALYSIS>";"  {
  iGP_OPCODE_FIELD = 0;
  iNoOp++;
 }

<GP_RULE_ANALYSIS>"," {
  if( bGPOPCODE_ANALYSIS ) iGP_OPCODE_FIELD++;
 }


<GP_RULE_ANALYSIS>"{" {
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
 }

/*
  This section copy the code contained in the code part of the GP opcode rule
  It count the number of curly bracket, when it reaches 0, the code part is finished.
 */
<GP_COPY_OPCODE_CODE>"{" {
  accolade_counter++;
  opDesc[iNoOp]->cpuCodeStream << "{";
  opDesc[iNoOp]->gpuCodeStream << "{";
 }

<GP_COPY_OPCODE_CODE>"}" {
  accolade_counter--;
  if( accolade_counter==0 ){
    opDesc[iNoOp]->gpuCodeStream << "\n      stack[sp++] = RESULT;\n";

    BEGIN GP_RULE_ANALYSIS;
  }
  else{
    opDesc[iNoOp]->cpuCodeStream << "}";
    opDesc[iNoOp]->gpuCodeStream << "}";
  }
 }

<GP_COPY_OPCODE_CODE>"INPUT["[0-9]*"]" {
  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
//  printf("input no : %d\n",no_input);
  opDesc[iNoOp]->cpuCodeStream << "input["<< no_input <<"]" ;
  opDesc[iNoOp]->gpuCodeStream << "input["<< no_input << "]";  
 }

<GP_COPY_OPCODE_CODE>"ERC" {
  opDesc[iNoOp]->isERC = true;
  opDesc[iNoOp]->cpuCodeStream << "root->erc_value" ;
  opDesc[iNoOp]->gpuCodeStream << "k_progs[start_prog++];" ;
//  printf("ERC matched\n");
}

<GP_COPY_OPCODE_CODE>\n {
  opDesc[iNoOp]->cpuCodeStream << "\n  ";
  opDesc[iNoOp]->gpuCodeStream << "\n    ";
 }


<GP_COPY_OPCODE_CODE>. {
  opDesc[iNoOp]->cpuCodeStream << yytext;
  opDesc[iNoOp]->gpuCodeStream << yytext;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_GENOME_EVAL_HDR" { 
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  if( bVERBOSE ) printf("Insert GP eval header\n");
  iCOPY_GP_EVAL_STATUS = EVAL_HDR;
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_GENOME_EVAL_BDY" {
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = false;
  BEGIN COPY_GP_EVAL;
 }



<TEMPLATE_ANALYSIS>"\\INSERT_GENOME_EVAL_BDY_GPU" {
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_BDY;
  if( bVERBOSE ) printf("Insert GP eval body\n");
  fprintf(fpOutputFile,"      ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 }



<TEMPLATE_ANALYSIS>"\\INSERT_GENOME_EVAL_FTR" {
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  BEGIN COPY_GP_EVAL;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_GENOME_EVAL_FTR_GPU" {
  yyreset();
  rewind(fpGenomeFile);
  yyin = fpGenomeFile;
  bIsCopyingGPEval = false;
  iCOPY_GP_EVAL_STATUS = EVAL_FTR;
  if( bVERBOSE ) printf("Insert GP eval footer\n");
  fprintf(fpOutputFile,"  ");
  bCOPY_GP_EVAL_GPU = true;
  BEGIN COPY_GP_EVAL;
 }


<COPY_GP_EVAL><<EOF>> {
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
 }


<COPY_GP_EVAL>"\\GenomeClass::evaluator"[ \t]*"header"[ \t]*":" {
  if( iCOPY_GP_EVAL_STATUS==EVAL_HDR){
    bIsCopyingGPEval = true;
  }
 }

<COPY_GP_EVAL>"\\GenomeClass::evaluator"[ \t]*"for"[ \t]*"each"[ \t]*"fc"[ \t]*":" {
  if( iCOPY_GP_EVAL_STATUS==EVAL_BDY){
    bIsCopyingGPEval = true;
  }
 }



<COPY_GP_EVAL>"\\GenomeClass::evaluator"[ \t]*"accumulator"[ \t]*":" {
  if( iCOPY_GP_EVAL_STATUS==EVAL_FTR){
    bIsCopyingGPEval = true;
  }
 }

<COPY_GP_EVAL>"\\end" {
  if( bIsCopyingGPEval ){
    bIsCopyingGPEval = false;
    bCOPY_GP_EVAL_GPU = false;
    rewind(fpGenomeFile);
    yyin = fpTemplateFile;
    BEGIN TEMPLATE_ANALYSIS;
  }
 }

<COPY_GP_EVAL>. {
  if( bIsCopyingGPEval ) fprintf(fpOutputFile,"%s",yytext);
 }

<COPY_GP_EVAL>"OUTPUT" {
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[i*NUMTHREAD2+tid]" );
    else fprintf(fpOutputFile, "outputs[i]" );
  
 }

<COPY_GP_EVAL>"OUTPUT["[0-9]+"]" {
  char* endptr;
  unsigned no_output = strtol(yytext+strlen("OUTPUT["),&endptr,10);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[(i+%d)*NUMTHREAD2+tid]", no_output);
    else fprintf(fpOutputFile, "outputs[i+%d]", no_output );
  
 }

<COPY_GP_EVAL>"OUTPUT["[a-zA-Z][a-zA-Z0-9]*"]" {
	char *var;
	var = strndup(yytext+strlen("OUTPUT["), strlen(yytext) - strlen("OUTPUT[") - 1);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "outputs[(i+%s)*NUMTHREAD2+tid]", var);
    else fprintf(fpOutputFile, "outputs[i+%s]", var);
  
 }

<COPY_GP_EVAL>"INPUT" {
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[i*NUMTHREAD2+tid]" );
    else fprintf(fpOutputFile, "inputs[i][0]" );
  
 }

<COPY_GP_EVAL>"INPUT["[0-9]+"]" {
  char* endptr;
  unsigned no_input = strtol(yytext+strlen("INPUT["),&endptr,10);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[(i+%d)*NUMTHREAD2+tid]", no_input);
    else fprintf(fpOutputFile, "inputs[i+%d][0]", no_input );
  
 }

<COPY_GP_EVAL>"INPUT["[a-zA-Z][a-zA-Z0-9]*"]" {
	char *var;
	var = strndup(yytext+strlen("INPUT["), strlen(yytext) - strlen("INPUT[") - 1);
  if( bIsCopyingGPEval) 
    if( bCOPY_GP_EVAL_GPU )
      fprintf(fpOutputFile, "k_inputs[(i+%s)*NUMTHREAD2+tid]", var);
    else fprintf(fpOutputFile, "inputs[i+%s][0]", var);
  
 }

<COPY_GP_EVAL>"return " {
  if( bIsCopyingGPEval )
    if( iCOPY_GP_EVAL_STATUS==EVAL_FTR )
      if( bCOPY_GP_EVAL_GPU ){
	fprintf(fpOutputFile,"k_results[index] =");
      }
      else fprintf(fpOutputFile,"return fitness=");
 }




<COPY_GP_EVAL>\n {
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
 }


<COPY_GP_EVAL>"return"	{
  if( bIsCopyingGPEval )
    fprintf(fpOutputFile,"return fitness = "); 
 }



<TEMPLATE_ANALYSIS>"\\INSTEAD_EVAL_FUNCTION"  {
  //DEBUG_PRT_PRT("insert beg");
  yyreset();
  yyin = fpGenomeFile;
  if (bVERBOSE) printf ("Evaluation population in a single function!!.\n");
  lineCounter = 1;
  BEGIN COPY_INSTEAD_EVAL;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_END_GENERATION_FUNCTION"  {
  if (bVERBOSE) printf ("Inserting at the end of each generation function.\n");
  yyreset();
  yyin = fpGenomeFile;
  bEndGeneration = true;
  bBeginGeneration = false;
  BEGIN COPY_END_GENERATION_FUNCTION;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_BOUND_CHECKING"  {
  if (bVERBOSE) printf ("Inserting Bound Checking function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_BOUND_CHECKING_FUNCTION;
 }

<TEMPLATE_ANALYSIS>"\\ANALYSE_USER_CLASSES"  {
  if (bVERBOSE) printf ("Analysing user classes.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN ANALYSE_USER_CLASSES;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_USER_CLASSES"  {
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
 }
<TEMPLATE_ANALYSIS>"\\GENOME_CTOR"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome constructor.\n");
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
    if (pSym->Object->ObjectType==oPointer){
      fprintf(fpOutputFile,"    %s=NULL;\n",pSym->Object->sName);
    }
  }
 }

<TEMPLATE_ANALYSIS>"\\GENOME_SIZE" {
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
 }

<TEMPLATE_ANALYSIS>"\\COPY_CUDA_BUFFER" {
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
  
 }

/* <TEMPLATE_ANALYSIS>"\\GENOME_CUDA_MOTION" { */
/*   if (bVERBOSE) printf ("Inserting default genome cuda motion function.\n"); */
  
/*   CListItem<CSymbol*> *pSym; */
/*   if (bVERBOSE) printf ("Creating default copy constructor.\n"); */
/*   fprintf (fpOutputFile,"// Memberwise copy\n");              */
/*   pGENOME->pSymbolList->reset(); */
/*   while (pSym=pGENOME->pSymbolList->walkToNextItem()){ */
/*     if (pSym->Object->ObjectQualifier==1) continue; // 1=Static */
/*     if (pSym->Object->ObjectType==oObject) */
/*       fprintf(fpOutputFile,"    %s=genome.%s;\n",pSym->Object->sName,pSym->Object->sName); */
/*     if (pSym->Object->ObjectType==oPointer) */
/*       fprintf(fpOutputFile,"    %s=(genome.%s ? new %s(*(genome.%s)) : NULL);\n",pSym->Object->sName,pSym->Object->sName,pSym->Object->pType->sName,pSym->Object->sName); */
/*     if (pSym->Object->ObjectType==oArray){ */
/* /\*       fprintf(fpOutputFile,"    {for(int EASEA_Ndx=0; EASEA_Ndx<%d; EASEA_Ndx++)\n",pSym->Object->nSize/pSym->Object->pType->nSize); *\/ */
/* /\*       fprintf(fpOutputFile,"       %s[EASEA_Ndx]=genome.%s[EASEA_Ndx];}\n",pSym->Object->sName,pSym->Object->sName); *\/ */
      
/*     } */
/*   } */
/*  } */




<TEMPLATE_ANALYSIS>"\\INSERT_GENOME"  {        
  if (pGENOME->sString) {
    if (bVERBOSE) printf ("Inserting Methods into Genome Class.\n");
    fprintf(fpOutputFile,"// User-defined methods:\n\n");
    fprintf(fpOutputFile,"%s\n",pGENOME->sString);
  }
  if (bVERBOSE) printf ("Inserting genome.\n");
  pGENOME->print(fpOutputFile);
 }                                            
<TEMPLATE_ANALYSIS>"\\ASSIGNMENT_OP"  {
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
 }
<TEMPLATE_ANALYSIS>"\\CLONE"  {
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
 }

<TEMPLATE_ANALYSIS>"\\COPY_CTOR"  {
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
 }

<TEMPLATE_ANALYSIS>"\\GENOME_SERIAL"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome serializer.\n");
  fprintf (fpOutputFile,"// Memberwise serialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->serializeIndividual(fpOutputFile, "this");
  //fprintf(fpOutputFile,"\tEASEA_Line << endl;\n");
 }

<TEMPLATE_ANALYSIS>"\\GENOME_DESERIAL"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Inserting default genome deserializer.\n");
  fprintf (fpOutputFile,"// Memberwise deserialization\n");
  pGENOME->pSymbolList->reset();
  pGENOME->deserializeIndividual(fpOutputFile, "this");
 }

<TEMPLATE_ANALYSIS>"\\COPY_CUDA_CTOR"  {
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
 }

<TEMPLATE_ANALYSIS>"\\GENOME_DTOR"  {
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
 }

<TEMPLATE_ANALYSIS>"\\EQUAL"  {       
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
 }
<TEMPLATE_ANALYSIS>"\\COMPARE"  {
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
 }
<TEMPLATE_ANALYSIS>"\\READ"  {
  CListItem<CSymbol*> *pSym;
  if (bVERBOSE) printf ("Creating default read command.\n");
  fprintf (fpOutputFile,"// Default read command\n");             
  pGENOME->pSymbolList->reset();
  while (pSym=pGENOME->pSymbolList->walkToNextItem()){
    if (pSym->Object->ObjectQualifier==1) continue; // 1=Static
  }
 }
<TEMPLATE_ANALYSIS>"\\INSERT_DISPLAY"  {        
  if (bVERBOSE) printf ("Inserting genome display function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_DISPLAY;   
 }


<TEMPLATE_ANALYSIS>"\\INSERT_MAKEFILE_OPTION" {
  if( bVERBOSE ) printf("Inserting user LDFLAGS.\n");
  yyreset();
  yyin = fpGenomeFile;
  BEGIN COPY_MAKEFILE_OPTION;
 }

<TEMPLATE_ANALYSIS>"\\WRITE"  {
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
 }    
<TEMPLATE_ANALYSIS>"\\INSERT_USER_FUNCTIONS"  {
  if (bVERBOSE) printf ("Inserting user functions.\n");
  yyreset();
  yyin = fpGenomeFile;                    
  lineCounter=2;                                 // switch to .ez file and analyser
  BEGIN COPY_USER_FUNCTIONS;
 }
<TEMPLATE_ANALYSIS>"\\INSERT_EO_INITIALISER"  {        
  yyreset();
  bWithinEO_Function=1;
  lineCounter=1;
  if( TARGET==CUDA || TARGET==STD) bWithinCUDA_Initializer = 1;
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN COPY_EO_INITIALISER;                               // not implemented as a function !
 }

<TEMPLATE_ANALYSIS>"\\INSERT_INITIALISER"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter = 1;
  BEGIN COPY_INITIALISER;   
 }

<TEMPLATE_ANALYSIS>"\\INSERT_FINALIZATION_FUNCTION"  {
  if (bVERBOSE) printf ("Inserting Finalization function.\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_FINALIZATION_FUNCTION;
 }

<TEMPLATE_ANALYSIS>"\\INSERT_CROSSOVER"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_CROSSOVER;   
 }
<TEMPLATE_ANALYSIS>"\\INSERT_MUTATOR"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_MUTATOR;   
 }
<TEMPLATE_ANALYSIS>"\\INSERT_EVALUATOR"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_EVALUATOR;   
 }
<TEMPLATE_ANALYSIS>"\\INSERT_OPTIMISER"  {      
  if( bVERBOSE ) fprintf(stdout,"Inserting optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  lineCounter=1;
  BEGIN COPY_OPTIMISER;   
 }
<TEMPLATE_ANALYSIS>"\\INSERT_CUDA_EVALUATOR"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_EVALUATOR;
 }
<TEMPLATE_ANALYSIS>"\\INSERT_CUDA_OPTIMISOR"  { 
  if( bVERBOSE ) fprintf(stdout,"Inserting cuda optimization function\n");
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  bWithinCUDA_Evaluator = 1;
  lineCounter=1;
  BEGIN COPY_OPTIMISER;
 }

<TEMPLATE_ANALYSIS>"\\ANALYSE_PARAMETERS"  {        
  yyreset();
  yyin = fpGenomeFile;                                                     // switch to .ez file and analyser
  BEGIN PARAMETERS_ANALYSIS;   
 }
<TEMPLATE_ANALYSIS>"\\INSERT_GEN_FCT_CALL"  {
  if (bGenerationReplacementFunction) {
    if( bVERBOSE ) fprintf(stdout,"Inserting generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAGenerationFunction(this);");
    }
  }
 }     

<TEMPLATE_ANALYSIS>"\\INSERT_END_GEN_FCT_CALL"  {
  if( bEndGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting end generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEAEndGenerationFunction(this);");
    }
  }
 }     

<TEMPLATE_ANALYSIS>"\\INSERT_BEGINNING_GEN_FCT_CALL"  {
  if( bBeginGenerationFunction ) {
    if( bVERBOSE ) fprintf(stdout,"Inserting beginning generation function call\n");
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABeginningGenerationFunction(this);");
    }
  }
 }     


<TEMPLATE_ANALYSIS>"\\INSERT_BOUND_CHECKING_FCT_CALL"  {
  if (bBoundCheckingFunction) {
    if( TARGET==CUDA || TARGET==STD ){
      fprintf(fpOutputFile,"\n\tEASEABoundChecking(this);");
    }
  }
 }        

<TEMPLATE_ANALYSIS>"\\IS_PARENT_REDUCTION" {
    fprintf(fpOutputFile,"%d",bIsParentReduce);
 }

<TEMPLATE_ANALYSIS>"\\IS_OFFSPRING_REDUCTION" {
  fprintf(fpOutputFile,"%d",bIsOffspringReduce);
 }

<TEMPLATE_ANALYSIS>"\\INSERT_INIT_FCT_CALL"  {
  if (bInitFunction) fprintf(fpOutputFile,"\n  EASEAInitFunction(argc, argv);\n");
 }

<TEMPLATE_ANALYSIS>"\\INSERT_FINALIZATION_FCT_CALL"  {
  if (bFinalizationFunction) fprintf(fpOutputFile,"\n  EASEAFinalization(pop);\n");
 }        

<TEMPLATE_ANALYSIS>"\\INSERT_USER_CLASSES_DEFINITIONS"  {
  //DEBUG_PRT_PRT("Inserting user classe definitions");
  pGENOME->printUserClasses(fpOutputFile);
 }        

<TEMPLATE_ANALYSIS>"\\SELECTOR"  {
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
}
<TEMPLATE_ANALYSIS>"\\SELECTOR_OPERATOR" {fprintf(fpOutputFile,"%s",sSELECTOR_OPERATOR);}
<TEMPLATE_ANALYSIS>"\\SELECT_PRM"  {fprintf(fpOutputFile,"%f",fSELECT_PRM);}
<TEMPLATE_ANALYSIS>"\\RED_PAR_OPERATOR" {fprintf(fpOutputFile,"%s",sRED_PAR_OPERATOR);}
<TEMPLATE_ANALYSIS>"\\RED_PAR_PRM"  {fprintf(fpOutputFile,"%f",fRED_PAR_PRM);}
<TEMPLATE_ANALYSIS>"\\RED_OFF_OPERATOR" {fprintf(fpOutputFile,"%s",sRED_OFF_OPERATOR);}
<TEMPLATE_ANALYSIS>"\\RED_OFF_PRM"  {fprintf(fpOutputFile,"%f",fRED_OFF_PRM);}
<TEMPLATE_ANALYSIS>"\\RED_FINAL_OPERATOR" {fprintf(fpOutputFile,"%s",sRED_FINAL_OPERATOR);}
<TEMPLATE_ANALYSIS>"\\RED_FINAL_PRM"  {fprintf(fpOutputFile,"%f",fRED_FINAL_PRM);}
<TEMPLATE_ANALYSIS>"\\POP_SIZE"  {fprintf(fpOutputFile,"%d",nPOP_SIZE);}
<TEMPLATE_ANALYSIS>"\\OFF_SIZE"  {fprintf(fpOutputFile,"%d",nOFF_SIZE);}
<TEMPLATE_ANALYSIS>"\\ELITE_SIZE"  { fprintf(fpOutputFile,"%d",nELITE); }
<TEMPLATE_ANALYSIS>"\\FC_SIZE"  { fprintf(fpOutputFile,"%d",iNO_FITNESS_CASES); }


<TEMPLATE_ANALYSIS>"\\RED_PAR"  {
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
 }
<TEMPLATE_ANALYSIS>"\\RED_OFF"  {
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
 }
<TEMPLATE_ANALYSIS>"\\RED_FINAL"  {
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
 }
<TEMPLATE_ANALYSIS>"\\CUDA_RULE_DIR"  {
if(OPERATING_SYSTEM=WINDOWS)
	fprintf(fpOutputFile,"%s\\",getenv("NVSDKCUDA_ROOT"));
}
<TEMPLATE_ANALYSIS>"\\SURV_PAR_SIZE"  {if(fSURV_PAR_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_PAR_SIZE); else fprintf(fpOutputFile,"%f",(float)nPOP_SIZE);}
<TEMPLATE_ANALYSIS>"\\SURV_OFF_SIZE"  {if(fSURV_OFF_SIZE>=0.0)fprintf(fpOutputFile,"%f",fSURV_OFF_SIZE); else fprintf(fpOutputFile,"%f",(float)nOFF_SIZE);}
<TEMPLATE_ANALYSIS>"\\GENOME_NAME"  {fprintf(fpOutputFile,"%s",nGENOME_NAME);}
<TEMPLATE_ANALYSIS>"\\PROBLEM_DIM"  {fprintf(fpOutputFile,"%d",nPROBLEM_DIM);}
<TEMPLATE_ANALYSIS>"\\NB_GEN"  {fprintf(fpOutputFile,"%d",nNB_GEN);}
<TEMPLATE_ANALYSIS>"\\TIME_LIMIT" {fprintf(fpOutputFile,"%d",nTIME_LIMIT);}
<TEMPLATE_ANALYSIS>"\\MUT_PROB"  {fprintf(fpOutputFile,"%f",fMUT_PROB);}
<TEMPLATE_ANALYSIS>"\\XOVER_PROB"  {fprintf(fpOutputFile,"%f",fXOVER_PROB);}
<TEMPLATE_ANALYSIS>"\\MINIMAXI"  {fprintf(fpOutputFile,"%s",(nMINIMISE? "true" : "false")); }
<TEMPLATE_ANALYSIS>"\\ELITISM"  {fprintf(fpOutputFile,"%d",bELITISM);}

<TEMPLATE_ANALYSIS>"\\NB_OPT_IT"  {fprintf(fpOutputFile,"%d",nNB_OPT_IT);}
<TEMPLATE_ANALYSIS>"\\BALDWINISM" {fprintf(fpOutputFile,"%d",bBALDWINISM);}

<TEMPLATE_ANALYSIS>"\\REMOTE_ISLAND_MODEL" {fprintf(fpOutputFile,"%d",bREMOTE_ISLAND_MODEL);}
<TEMPLATE_ANALYSIS>"\\IP_FILE" {if(strlen(sIP_FILE)>0)fprintf(fpOutputFile,"%s",sIP_FILE); else fprintf(fpOutputFile,"NULL");}
<TEMPLATE_ANALYSIS>"\\MIGRATION_PROBABILITY" {fprintf(fpOutputFile,"%f",fMIGRATION_PROBABILITY);}
<TEMPLATE_ANALYSIS>"\\SERVER_PORT" {fprintf(fpOutputFile,"%d",nSERVER_PORT);}

<TEMPLATE_ANALYSIS>"\\PRINT_STATS" {fprintf(fpOutputFile,"%d",bPRINT_STATS);}
<TEMPLATE_ANALYSIS>"\\PLOT_STATS" {fprintf(fpOutputFile,"%d",bPLOT_STATS);}
<TEMPLATE_ANALYSIS>"\\GENERATE_CSV_FILE" {fprintf(fpOutputFile,"%d",bGENERATE_CSV_FILE);}
<TEMPLATE_ANALYSIS>"\\GENERATE_GNUPLOT_SCRIPT" {fprintf(fpOutputFile,"%d",bGENERATE_GNUPLOT_SCRIPT);}
<TEMPLATE_ANALYSIS>"\\GENERATE_R_SCRIPT" {fprintf(fpOutputFile,"%d",bGENERATE_R_SCRIPT);}

<TEMPLATE_ANALYSIS>"\\SAVE_POPULATION" {fprintf(fpOutputFile,"%d",bSAVE_POPULATION);}
<TEMPLATE_ANALYSIS>"\\START_FROM_FILE" {fprintf(fpOutputFile,"%d",bSTART_FROM_FILE);}

<TEMPLATE_ANALYSIS>"\\START_CUDA_GENOME_H_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,"Individual.hpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 }
<TEMPLATE_ANALYSIS>"\\START_CUDA_GENOME_CU_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  if( TARGET==CUDA )
    strcat(sFileName,"Individual.cu");
  else if( TARGET==STD )
    strcat(sFileName,"Individual.cpp");
  fpOutputFile=fopen(sFileName,"w");    
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
 }
<TEMPLATE_ANALYSIS>"\\START_EO_PARAM_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".prm");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }
<TEMPLATE_ANALYSIS>"\\START_EO_MAKEFILE_TPL"  {
  char sFileName[1000];
  fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".mak");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }

<TEMPLATE_ANALYSIS>"\\START_VISUAL_TPL"  {
  char sFileName[1000];
 fclose(fpOutputFile);
  strcpy(sFileName, sRAW_PROJECT_NAME);
  strcat(sFileName,".vcproj");
  if (bVERBOSE) printf("Creating %s...\n",sFileName);
  fpOutputFile=fopen(sFileName,"w");
 }

<TEMPLATE_ANALYSIS>"\\START_CUDA_MAKEFILE_TPL"  {
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
}
<TEMPLATE_ANALYSIS>"\\TEMPLATE_END"  {
  if (nWARNINGS) printf ("\nWARNING !!!\nTarget file(s) generation went through WITH %d WARNING(S) !\n",nWARNINGS);
  else printf ("\nCONGRATULATIONS !!!\nTarget file(s) generation succeeded without warnings.\n");
  printf ("You can now type \"make\" to compile your project.\n");
  if (TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"\n# That's all folks ! \n");
  else fprintf(fpOutputFile,"\n// That's all folks ! \n");
  fflush(fpOutputFile);
  fclose(fpOutputFile);
  fclose(fpTemplateFile);
  fclose(fpGenomeFile);
 }

<TEMPLATE_ANALYSIS>.|\n {putc(yytext[0],fpOutputFile);}                                      

//****************************************
//  use information found in .ez file
//****************************************

<COPY_EO_INITIALISER>"\\GenomeClass::initiali"[sz]"er"[ \t\n]*":" {
  fprintf (fpOutputFile,"// Genome Initialiser\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 }                                                               
<COPY_EO_INITIALISER><<EOF>> {
  if (bVERBOSE) printf("*** No genome initialiser was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_EO_INITIALISER>. {}                                      
<COPY_EO_INITIALISER>\n {lineCounter++;}

<COPY_USER_DECLARATIONS>"\\User"[ \t\n]+"declarations"[ \t\n]*":" {
  fprintf (fpOutputFile,"// User declarations\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 }                                                               
<COPY_USER_DECLARATIONS><<EOF>> {
  if (bVERBOSE) printf("*** No user declarations were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }

<COPY_USER_DECLARATIONS>. {}
<COPY_USER_DECLARATIONS>\n {
  lineCounter++;
 } 
               
<COPY_USER_CUDA>"\\User"[ \t\n]+"CUDA"[ \t\n]*":" {
  fprintf (fpOutputFile,"// User CUDA\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 }                                                               
<COPY_USER_CUDA><<EOF>> {
  if (bVERBOSE) printf("*** No user CUDA were found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }

<COPY_USER_CUDA>. {}
<COPY_USER_CUDA>\n {
  lineCounter++;
 }                

<COPY_USER_FUNCTIONS>"\\User"[ \t\n]+"functions"[ \t\n]*":" {
  fprintf (fpOutputFile,"// User functions\n\n"); 
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY;
 }                                                               
<COPY_USER_FUNCTIONS><<EOF>> {
  if (bVERBOSE) printf("*** No user functions were found. ***\n");
  fprintf(fpOutputFile,"\n// No user functions.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_USER_FUNCTIONS>. {}                                      
<COPY_USER_FUNCTIONS>\n {
  lineCounter++;
 }

<COPY_INITIALISATION_FUNCTION>"\\Before"[ \t\n]+"everything"[ \t\n]+"else"[ \t\n]+"function"[ \t\n]*":" {
    fprintf (fpOutputFile,"// Initialisation function\nvoid EASEAInitFunction(int argc, char *argv[]){\n");
  bFunction=1; bInitFunction=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
      
  BEGIN COPY;
 }
<COPY_INITIALISATION_FUNCTION><<EOF>> {bInitFunction=0; // No before everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No before everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No before everything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;

  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_INITIALISATION_FUNCTION>. {}                                      
<COPY_INITIALISATION_FUNCTION>\n {lineCounter++;} 

//****************************************
//  Easea GPGPU & EO Finalization function
//****************************************
<COPY_FINALIZATION_FUNCTION>"\\After"[ \t\n]+"everything"[ \t\n]+"else"[ \t\n]+"function"[ \t\n]*":" {
  fprintf (fpOutputFile,"// Finalization function\nvoid EASEAFinalization(CPopulation* population){\n");
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

  bFunction=1; bFinalizationFunction=1;
  BEGIN COPY;
 }

<COPY_FINALIZATION_FUNCTION><<EOF>> {bFinalizationFunction=0; // No after everything else function was found in the .ez file
  if (bVERBOSE) printf("*** No after everything else function was found. ***\n");
  fprintf(fpOutputFile,"\n// No after eveything else function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_FINALIZATION_FUNCTION>. {}                                      
<COPY_FINALIZATION_FUNCTION>\n {lineCounter++;}                                      


<COPY_END_GENERATION_FUNCTION>"\\At"[ \t\n]+"the"[ \t\n]+"end"[ \t\n]+"of"[ \t\n]+"each"[ \t\n]+"generation"[ \t\n]+"function"[ \t\n]*":" {
  //DEBUG_PRT_PRT("at each end");
  if( (TARGET==CUDA || TARGET==STD)  ){
    fprintf (fpOutputFile,"{\n");
    //fprintf (fpOutputFile,"// Function called at each new generation\nvoid EASEAEndGenerationFunction(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    bFunction=1; 
    bEndGenerationFunction = 1;
    BEGIN COPY_USER_GENERATION;
  }
 }

<COPY_END_GENERATION_FUNCTION><<EOF>> {
  bEndGenerationFunction=0; // No Generation function was found in the .ez file
  if( bVERBOSE) printf("*** No end generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at end of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_END_GENERATION_FUNCTION>.|\n {}                                      


<COPY_BEG_GENERATION_FUNCTION>"\\At"[ \t\n]+"the"[ \t\n]+"beginning"[ \t\n]+"of"[ \t\n]+"each"[ \t\n]+"generation"[ \t\n]+"function"[ \t\n]*":" {
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
 }

<COPY_INSTEAD_EVAL>"\\Instead"[ \t\n]+"evaluation"[ \t\n]+"function:" {
  //DEBUG_PRT_PRT("at each beg");
  if( (TARGET==CUDA || TARGET==STD)){
    fprintf (fpOutputFile,"{\n");
    bFunction=1;
    if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);

    BEGIN COPY_USER_GENERATION;
  }
 }

<COPY_INSTEAD_EVAL>. {}                                      
<COPY_INSTEAD_EVAL>\n {lineCounter++;}                                      

<COPY_INSTEAD_EVAL><<EOF>> {
  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No Instead evaluation step function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Instead evaluation step function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }


<COPY_BEG_GENERATION_FUNCTION><<EOF>> {
  bBeginGenerationFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No beginning generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No at beginning of generation function.\n");

  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_BEG_GENERATION_FUNCTION>\n {lineCounter++;}
<COPY_BEG_GENERATION_FUNCTION>. {}



<COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT>"\\At"[ \t\n]+"each"[ \t\n]+"generation"[ \t\n]+"before"[ \t\n]+"reduce"[ \t\n]+"function"[ \t\n]*":" {
  //DEBUG_PRT_PRT("at each");
  if((TARGET==CUDA || TARGET==STD)){// && !bBeginGeneration && !bEndGeneration ){
      fprintf (fpOutputFile,"{\n");
      bFunction=1; 
      bGenerationReplacementFunction=1;
      BEGIN COPY_USER_GENERATION;
  }
 }

<COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT><<EOF>> {
  bGenerationFunctionBeforeReplacement=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No generation function was found. ***\n");
  fprintf(fpOutputFile,"\n// No generation function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT>.|\n {}                                      


<COPY_BOUND_CHECKING_FUNCTION>"\\Bound"[ \t\n]+"checking"[ \t\n]*":" {
  if(TARGET==CUDA || TARGET==STD){
    //fprintf (fpOutputFile,"void EASEABoundChecking(EvolutionaryAlgorithm* evolutionaryAlgorithm){\n");
    fprintf (fpOutputFile,"{\n");
  }
  bFunction=1; bBoundCheckingFunction=1;
  BEGIN COPY_USER_GENERATION;
 }
<COPY_BOUND_CHECKING_FUNCTION><<EOF>> {bBoundCheckingFunction=0; // No Generation function was found in the .ez file
  if (bVERBOSE) printf("*** No bound checking function was found. ***\n");
  fprintf(fpOutputFile,"\n// No Bound checking function.\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_BOUND_CHECKING_FUNCTION>.|\n {}                                      



<ANALYSE_USER_CLASSES>"\\User"[ \t\n]+"classes"[ \t\n]*":" {
  BEGIN GENOME_ANALYSIS; return CLASSES;}
<ANALYSE_USER_CLASSES>.|\n {}                                      

//****************************************
//  Basic copy to cpp file with minor changes
//****************************************

<COPY>"\""  {(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");}
<COPY>"\\\""  {fprintf(fpOutputFile,"\\\"");}

<COPY>"currentGeneration" |
<COPY>"currentGeneration"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }} // local genome name
<COPY>"NB_GEN" |
<COPY>"NB_GEN"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");} // local genome name
<COPY>"POP_SIZE" |
<COPY>"POP_SIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");} // local genome name
<COPY>"MUT_PROB" |
<COPY>"MUT_PROB"/[ \t\n]*"==" {
  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"MUT_PROB");
  else
    if( TARGET==STD || TARGET==CUDA){
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
    else fprintf(fpOutputFile,"EZ_MUT_PROB");
  
 } // local genome name
<COPY>"XOVER_PROB" |
<COPY>"XOVER_PROB"/[ \t\n]*"==" {
  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");
 } // local genome name
<COPY>"REPL_PERC" |
<COPY>"REPL_PERC"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");} // local genome name
<COPY>"MINIMISE" |
<COPY>"MINIMISE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
  else fprintf(fpOutputFile,"EZ_MINIMISE");} // local genome name
<COPY>"MINIMIZE" |
<COPY>"MINIMIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
  else fprintf(fpOutputFile,"EZ_MINIMIZE");} // local genome name
<COPY>"MAXIMISE" |
<COPY>"MAXIMISE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
  else fprintf(fpOutputFile,"EZ_MAXIMISE");} // local genome name
<COPY>"MAXIMIZE" |
<COPY>"MAXIMIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
  else fprintf(fpOutputFile,"EZ_MAXIMIZE");} // local genome name

<COPY>"__device__" |
<COPY>"__host__" {
  if( TARGET==CUDA ){
    fprintf(fpOutputFile,"%s",yytext);
  }
 }

<COPY>"currentGeneration"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);}
<COPY>"NB_GEN"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"POP_SIZE"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"MUT_PROB"[ \t\n]*"=" {
  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);
}

<COPY>"XOVER_PROB"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"REPL_PERC"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"MINIMISE"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"MINIMIZE"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"MAXIMISE"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"MAXIMIZE"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The evaluation goal can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY>"false"  {fprintf(fpOutputFile,"false");} // local name
<COPY>"true"  {fprintf(fpOutputFile,"true");} // local name
<COPY>[^a-zA-Z0-9_]"bool"[^a-zA-Z0-9_]  {
  fprintf(fpOutputFile,yytext);}
<COPY>"tossCoin"  { // local random name 
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");}
<COPY>"random"  {
  fprintf(fpOutputFile,"globalRandomGenerator->random");}
<COPY>"Genome" {
  if (bWithinEO_Function && TARGET!=CUDA && TARGET!=STD) fprintf(fpOutputFile,"_genotype");
  else if(bWithinEO_Function && bWithinCUDA_Initializer )fprintf(fpOutputFile,"(*this)");
  else fprintf(fpOutputFile,"Genome");} // local genome name
<COPY>"identicalGenome"  {fprintf(fpOutputFile,"genome._evaluated");} // local name
<COPY>"#define"[ \t]*  {fprintf(fpOutputFile,"%s",yytext); BEGIN MACRO_IDENTIFIER;} 
<COPY>[a-zA-Z0-9_]*[ \t]+"GenomeClass::" {
  int i;
  for (i=0;(yytext[i]!=' ')&&(yytext[i]!=' ');i++);
  yytext[i]=0;
  fprintf(fpOutputFile,"template <class fitT> %s %sGenome<fitT>::",yytext,sPROJECT_NAME);}         
<COPY>"GenomeClass::"  {fprintf(fpOutputFile,"template <class fitT> %sGenome<fitT>::",sPROJECT_NAME);}
<COPY>"GenomeClass"  {
  if( TARGET==CUDA || TARGET==STD) fprintf(fpOutputFile,"IndividualImpl");
  else fprintf(fpOutputFile,"%sGenome",sPROJECT_NAME);} // local name
<COPY>"pPopulation[" {
  if(bFinalizationFunction){
	bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
  }
 }
<COPY>"]" {
  	if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  	else {fprintf(fpOutputFile,"])"); ;bWaitingToClosePopulation=false;}}
<COPY>"pPopulation" {
  if(bFinalizationFunction){
    fprintf(fpOutputFile,"pPopulation");
  }
  else fprintf(fpOutputFile,"/*pPopulation only in \"After everything else function\" this will cause an error*/ pPopulation");
 }
<COPY>"bBest" {
  if(bFinalizationFunction)
	fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
  else fprintf(fpOutputFile,"/*bBest only in \"After everything else function\" this will cause an error*/ bBest");
  }
<COPY>"\\end"  {
  if (bFunction==1 && bWithinCUDA_Initializer==0) {
    fprintf (fpOutputFile,"}\n"); 
    bFunction=0;
    bWithinCUDA_Initializer=0;
  }
  bWithinEO_Function=0;
  rewind(fpGenomeFile); 
  yyin = fpTemplateFile; 
  BEGIN TEMPLATE_ANALYSIS;} // Back to the template file
<COPY>.|\n {putc(yytext[0],fpOutputFile);}                                      

// Looking for an identifier
<MACRO_IDENTIFIER>[a-zA-Z0-9_]* {fprintf(fpOutputFile,"%s",yytext);
  pASymbol = new CSymbol(yytext); pASymbol->ObjectType=oMacro;
  BEGIN MACRO_DEFINITION; }
<MACRO_IDENTIFIER>/.|\n {BEGIN COPY;} // If no identifier was found, we're not interested
// Looking for the first number following the macro definition
<MACRO_DEFINITION>[ \t]*  {fprintf(fpOutputFile,"%s",yytext);} // gobbles up spaces and tabs
<MACRO_DEFINITION>[0-9]+"."[0-9]*{exponent}?  |
<MACRO_DEFINITION>"."[0-9]+{exponent}?  |
<MACRO_DEFINITION>[0-9]+{exponent} {fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = myStrtod();
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;}
<MACRO_DEFINITION>[0-9]+           {fprintf(fpOutputFile,"%s",yytext);
  pASymbol->dValue = atoi(yytext);
  pSymbolTable->insert(pASymbol);
  bSymbolInserted=1;
  BEGIN COPY;}
<MACRO_DEFINITION>/.|\n {if (!bSymbolInserted) delete pASymbol;
  else bSymbolInserted=0;
  BEGIN COPY;} // If no number was found, we're not interested

//****************************************
//  GENOME_ANALYSIS lexer
//****************************************

// takes care of C++-like comments
<GENOME_ANALYSIS>\/\/[^\n]* ;         // blah blah on a line containing sth else
<GENOME_ANALYSIS>^[ \t]*\/\/[^\n]*\n ; // blah blah on its own on a single line
<GENOME_ANALYSIS>\/\*[^\*]*\*\/ ;  /* blah blah on a line containing sth else */
<GENOME_ANALYSIS>^[ \t]*\/\*[^\*]*\*\/[ \t]*\n ; /* blah blah with nothing before the comment */

// gobbles up white spaces, tabs or carriage returns
<GENOME_ANALYSIS>[ \t]             { /* do nothing */ }
<GENOME_ANALYSIS>^[ \t]*\n      { /*return '\n';*/ }
<GENOME_ANALYSIS>\n                 { /*return '\n';*/ }

// keywords
<GENOME_ANALYSIS>"bool"                  {
  yylval.pSymbol = pSymbolTable->find("bool");
  return BOOL;}
<GENOME_ANALYSIS>"boolean"            {
    yylval.pSymbol = new CSymbol(yytext);
    return IDENTIFIER;
    }
<GENOME_ANALYSIS>"GPNode"            {
  yylval.pSymbol = new CSymbol("GPNode");
  //yylval.pSymbol->ObjectType = oPointer;
//  printf("match gpnode\n");
  return GPNODE;
 }

<GENOME_ANALYSIS>"static"                  {return STATIC;}
<GENOME_ANALYSIS>"int"                  {yylval.pSymbol = pSymbolTable->find("int"); return INT;}
<GENOME_ANALYSIS>"double"                  {yylval.pSymbol = pSymbolTable->find("double"); return DOUBLE;}
<GENOME_ANALYSIS>"float"                  {yylval.pSymbol = pSymbolTable->find("float"); return FLOAT;}
<GENOME_ANALYSIS>"char"                  {yylval.pSymbol = pSymbolTable->find("char"); return CHAR;}
<GENOME_ANALYSIS>"pointer"                  {yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;}
//<GENOME_ANALYSIS>"pointer"                  {yylval.pSymbol = pSymbolTable->find("pointer"); return POINTER;}
<GENOME_ANALYSIS>"\\end"   {rewind(fpGenomeFile);yyin = fpTemplateFile;BEGIN TEMPLATE_ANALYSIS;} // Back to the template file
  
<GENOME_ANALYSIS>"GenomeClass"        {return GENOME; }                         

<GENOME_ANALYSIS>"Methods:" {BEGIN GET_METHODS;
  yylval.szString=yytext;  
  bMethodsInGenome=1;
  return METHODS;}
// number
<GENOME_ANALYSIS>[0-9]+"."[0-9]*{exponent}?  |
<GENOME_ANALYSIS>"."[0-9]+{exponent}?  |
<GENOME_ANALYSIS>[0-9]+{exponent}     { yylval.dValue = myStrtod(); return NUMBER; }
<GENOME_ANALYSIS>[0-9]+ {yylval.dValue=atoi(yytext); return NUMBER;}

// identifier
<GENOME_ANALYSIS>[a-zA-Z_][a-zA-Z0-9_]*    { yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER; }
<GET_METHODS>[^\}]*   {BEGIN GENOME_ANALYSIS; return END_METHODS;}

//****************************************
//  Looking for a user-supplied display function.
//****************************************

<COPY_DISPLAY>"\\GenomeClass::display"[ \t\n]*":" { 
/*   //DEBUG_PRT_PRT("Display function is at %d line in %s.ez",yylineno,sRAW_PROJECT_NAME); */
/*   fprintf(fpOutputFile,"\n#line %d \"%s.ez\"\n",yylineno,sRAW_PROJECT_NAME); */
  bDisplayFunction=bWithinDisplayFunction=1;
  BEGIN COPY_USER_FUNCTION;
 }
<COPY_DISPLAY><<EOF>> {bDisplayFunction=0; // No display function was found in the .ez file
  if (bVERBOSE) printf("*** No display function was found. ***\n");
  rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bNotFinishedYet=1;
 }
<COPY_DISPLAY>.|\n {}                                      

<COPY_MAKEFILE_OPTION>"\\User"[ \t\n]+[Mm]"akefile"[ \t\n]+"options"[ \t\n]*":" {
  //DEBUG_PRT_PRT("LDFLAGS is beg: %s",yytext); 
  bWithinMAKEFILEOPTION=1;
  return MAKEFILE_OPTION;
 }


//****************************************
//  Looks for Makefile options (like 
//  LDFLAGS+= or CPPFLAGS...)
//****************************************
<COPY_MAKEFILE_OPTION>"\\end" {
  if( bWithinMAKEFILEOPTION ){
    //DEBUG_PRT_PRT("end of makefile options");
    yyin = fpTemplateFile;
    bWithinMAKEFILEOPTION = 0;
    BEGIN TEMPLATE_ANALYSIS;
    return END_OF_FUNCTION;
  }
 }

<COPY_MAKEFILE_OPTION>.|\n {
  if( bWithinMAKEFILEOPTION ){
    putc(yytext[0],fpOutputFile);
    }
 }

<COPY_MAKEFILE_OPTION>\n {
  if( bWithinMAKEFILEOPTION );
 }

<COPY_MAKEFILE_OPTION><<EOF>> { 
  //DEBUG_PRT_PRT("No makefile options defined.");

  yyin = fpTemplateFile;
  bWithinMAKEFILEOPTION = 0;
  BEGIN TEMPLATE_ANALYSIS;

  return MAKEFILE_OPTION;
 }

//****************************************
//  Looks for standard and user functions in the .ez file
//****************************************
<COPY_INITIALISER>"\\GenomeClass::initiali"[sz]"er"[ \t\n]*":" {
  bWithinInitialiser=1;
  BEGIN COPY_USER_FUNCTION;
  BEGIN TEMPLATE_ANALYSIS;
  return USER_CTOR;
 }
<COPY_INITIALISER>.|\n {}                                      
<COPY_CROSSOVER>"\\GenomeClass::crossover"[ \t\n]*":" {
  bWithinXover=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_XOVER;
 }
<COPY_CROSSOVER>. {}
<COPY_CROSSOVER>\n {lineCounter++;}
<COPY_MUTATOR>"\\GenomeClass::mutator"[ \t\n]*":" {
  bWithinMutator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  BEGIN COPY_USER_FUNCTION;
  return USER_MUTATOR;
 }
<COPY_MUTATOR>. {}
<COPY_MUTATOR>\n {lineCounter++;}

<COPY_EVALUATOR>"\\GenomeClass::evaluator"[ \t\n]*":" {
  BEGIN COPY_USER_FUNCTION;            
  bWithinEvaluator=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_EVALUATOR;
 }
<COPY_EVALUATOR>. {}
<COPY_EVALUATOR>\n {lineCounter++;}

<COPY_OPTIMISER>"\\GenomeClass::optimiser"[ \t\n]*":" {
  BEGIN COPY_USER_FUNCTION;
  bWithinOptimiser=1;
  if( bLINE_NUM_EZ_FILE )
    fprintf(fpOutputFile,"#line %d \"%s.ez\"\n",lineCounter, sRAW_PROJECT_NAME);
  return USER_OPTIMISER;
 }
<COPY_OPTIMISER>. {}
<COPY_OPTIMISER>\n {lineCounter++;}

//****************************************
//  Basic copy to .cpp file with major changes
//****************************************

<COPY_USER_GENERATION>\/\/[^\n]* {fprintf(fpOutputFile,yytext);}         // blah blah on a line containing sth else
<COPY_USER_GENERATION>^[ \t]*\/\/[^\n]*\n {fprintf(fpOutputFile,yytext);} // blah blah on its own on a single line
<COPY_USER_GENERATION>\/\*[^\*]*\*\/ {fprintf(fpOutputFile,yytext);}  /* blah blah on a line containing sth else */
<COPY_USER_GENERATION>^[ \t]*\/\*[^\*]*\*\/[ \t]*\n {fprintf(fpOutputFile,yytext);} /* blah blah with nothing before the comment */
<COPY_USER_GENERATION>"Genome"  {fprintf(fpOutputFile,"genome");} // local genome name
<COPY_USER_GENERATION>"\""  {(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");}
<COPY_USER_GENERATION>"\\\""  {fprintf(fpOutputFile,"\\\"");}

<COPY_USER_GENERATION>"pPopulation[" {bWaitingToClosePopulation=true;
    fprintf(fpOutputFile,"((IndividualImpl*)pPopulation[");
 }
<COPY_USER_GENERATION>"]" {
  if (!bWaitingToClosePopulation) fprintf(fpOutputFile,"]");
  else {
    fprintf(fpOutputFile,"])"); 
    bWaitingToClosePopulation=false;
  }
 }

<COPY_USER_GENERATION>"currentGeneration" |
<COPY_USER_GENERATION>"currentGeneration"/[ \t\n]*"==" {
  if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else {
      fprintf(fpOutputFile,"(*EZ_current_generation)");}} // local genome name
<COPY_USER_GENERATION>"NB_GEN" |
<COPY_USER_GENERATION>"NB_GEN"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else {fprintf(fpOutputFile,"(*EZ_NB_GEN)"); }} // local genome name                           
<COPY_USER_GENERATION>"POP_SIZE" |
<COPY_USER_GENERATION>"POP_SIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
    } // local genome name
<COPY_USER_GENERATION>"MUT_PROB" |
<COPY_USER_GENERATION>"MUT_PROB"/[ \t\n]*"==" {
  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else {fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
    }
 } // local genome name
<COPY_USER_GENERATION>"XOVER_PROB" |
<COPY_USER_GENERATION>"XOVER_PROB"/[ \t\n]*"==" {
  if (bDoubleQuotes) fprintf(fpOutputFile,"XOVER_PROB");
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");     
    }
 } // local genome name
<COPY_USER_GENERATION>"REPL_PERC" |
<COPY_USER_GENERATION>"REPL_PERC"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
    } // local genome name
<COPY_USER_GENERATION>"MINIMISE" |
<COPY_USER_GENERATION>"MINIMISE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMISE");
    } // local genome name
<COPY_USER_GENERATION>"MINIMIZE" |
<COPY_USER_GENERATION>"MINIMIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MINIMIZE");
    } // local genome name
<COPY_USER_GENERATION>"MAXIMISE" |
<COPY_USER_GENERATION>"MAXIMISE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMISE");
    } // local genome name
<COPY_USER_GENERATION>"MAXIMIZE" |
<COPY_USER_GENERATION>"MAXIMIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"MAXIMIZE");
    } // local genome name

<COPY_USER_GENERATION>"currentGeneration"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n    hint -> You must have meant \"NB_GEN=...\" rather than \"currentGeneration=...\"\n",sEZ_FILE_NAME,yylineno);
  exit(1);
 } 
<COPY_USER_GENERATION>"NB_GEN"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*EZ_NB_GEN)=");
    }} // local genome name
<COPY_USER_GENERATION>"POP_SIZE"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    } // local genome name
<COPY_USER_GENERATION>"MUT_PROB"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_MUT_PROB)=");
    }
 } // local genome name
<COPY_USER_GENERATION>"XOVER_PROB"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
  else {
      fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)=");
    }
 } // local genome name
<COPY_USER_GENERATION>"REPL_PERC"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    } // local genome name
<COPY_USER_GENERATION>"MINIMISE"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    }
<COPY_USER_GENERATION>"MINIMIZE"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    }
<COPY_USER_GENERATION>"MAXIMISE"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    }
<COPY_USER_GENERATION>"MAXIMIZE"[ \t\n]*"="  {if (bDoubleQuotes) fprintf(fpOutputFile,"%s",yytext);
    }
<COPY_USER_GENERATION>"false"  {fprintf(fpOutputFile,"false");} // local name
<COPY_USER_GENERATION>"true"  {fprintf(fpOutputFile,"true");} // local name
<COPY_USER_GENERATION>[^a-zA-Z0-9_]"bool"[^a-zA-Z0-9_]  {
  fprintf(fpOutputFile,yytext);}
<COPY_USER_GENERATION>"tossCoin"  { // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");}
<COPY_USER_GENERATION>"random"  {
  fprintf(fpOutputFile,"globalRandomGenerator->random");}
<COPY_USER_GENERATION>"identicalGenome"  {fprintf(fpOutputFile,"genome._evaluated");} // local name
<COPY_USER_GENERATION>"pPopulation" {
  if(bBeginGenerationFunction || bEndGenerationFunction || bGenerationFunctionBeforeReplacement){
    fprintf(fpOutputFile,"pPopulation)");
  }
  else fprintf(fpOutputFile,"pPopulation");
 }
<COPY_USER_GENERATION>"bBest" {
  fprintf(fpOutputFile,"((IndividualImpl*)bBest)");
 }
<COPY_USER_GENERATION>"\\end" {rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  fprintf(fpOutputFile,"}");} // Back to the template file
<COPY_USER_GENERATION>";" {if (bWaitingForSemiColon){
    bWaitingForSemiColon=0;
    if (bFinishMINIMISE) {fprintf(fpOutputFile,");\n  if (MINIMISE) g.minimize() else g.maximize();\n"); bFinishMINIMISE=0;}
    if (bFinishMINIMIZE) {fprintf(fpOutputFile,");\n  if (MINIMIZE) g.minimize() else g.maximize();\n"); bFinishMINIMISE=0;}
    if ((bFinishNB_GEN)&&(OPERATING_SYSTEM==UNIX))
      {fprintf(fpOutputFile,"));\n  if ( (EZ_daFITNESS = (double *) realloc(EZ_daFITNESS, ((*EZ_NB_GEN) +1)* sizeof (double) )) == NULL){\n");
	fprintf(fpOutputFile,"    fprintf(stderr,\"Not enough memory... bailing out.\");\n    exit(1);");}
    else if (bFinishNB_GEN) {fprintf(fpOutputFile,"));"); bFinishNB_GEN=0;}
    else fprintf(fpOutputFile,"));");
  }
  else fprintf(fpOutputFile,";");}
<COPY_USER_GENERATION>"evoluationaryAlgorithm." | 
<COPY_USER_GENERATION>"ea." {
  fprintf(fpOutputFile,"(*evoluationaryAlgorithm).");
 }
<COPY_USER_GENERATION>.|\n {putc(yytext[0],fpOutputFile);}                                      

//****************************************
//  Basic copy to .cpp file with minor changes
//****************************************

// takes care of C++-like comments
<COPY_USER_FUNCTION>\/\/[^\n]* {fprintf(fpOutputFile,yytext);}         // blah blah on a line containing sth else
<COPY_USER_FUNCTION>^[ \t]*\/\/[^\n]*\n {fprintf(fpOutputFile,yytext);} // blah blah on its own on a single line
<COPY_USER_FUNCTION>\/\*[^\*]*\*\/\n {fprintf(fpOutputFile,yytext);printf("%s\n",yytext);}  /* blah blah on a line containing sth else */
<COPY_USER_FUNCTION>\/\*[^\*]*\*\/ {fprintf(fpOutputFile,yytext);}  /* blah blah on a line containing sth else */
<COPY_USER_FUNCTION>^[ \t]*\/\*[^\*]*\*\/[ \t]*\n {fprintf(fpOutputFile,yytext);} /* blah blah with nothing before the comment */

<COPY_USER_FUNCTION>"GenomeClass"  {
  fprintf(fpOutputFile,"Genome.");
 }

<COPY_USER_FUNCTION>"f1"  {
  if( bWithinCUDA_Evaluator && TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[0])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 }

<COPY_USER_FUNCTION>"f2"  {
  if( bWithinCUDA_Evaluator &&  TARGET==CUDA && TARGET_FLAVOR==CUDA_FLAVOR_MO ){
    fprintf(fpOutputFile,"(f[1])");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 }


<COPY_USER_FUNCTION>"initOpts."  {
  if( ((bWithinEvaluator || bWithinOptimiser) && !bWithinCUDA_Evaluator) && ( TARGET==STD || TARGET==CUDA )){
    fprintf(fpOutputFile,"");
  }
  else
    fprintf(fpOutputFile,"%s",yytext);
 }


<COPY_USER_FUNCTION>"Genome"  {

  if(bWithinOptimiser || bWithinEvaluator || bWithinMutator || bWithinDisplayFunction){ 
    if( bWithinCUDA_Evaluator)
      fprintf(fpOutputFile, "(*INDIVIDUAL_ACCESS(devBuffer,id))");
    else fprintf(fpOutputFile, "(*this)");
  }

  else fprintf(fpOutputFile,"Genome");} // local genome name

<COPY_USER_FUNCTION>"\""  {(bDoubleQuotes ? bDoubleQuotes=0:bDoubleQuotes=1); fprintf(fpOutputFile,"\"");}
<COPY_USER_FUNCTION>"\\\""  {fprintf(fpOutputFile,"\\\"");}

<COPY_USER_FUNCTION>"currentGeneration" |
<COPY_USER_FUNCTION>"currentGeneration"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"currentGeneration");
  else switch  (TARGET) {
    case STD : fprintf(fpOutputFile,"(*EZ_current_generation)"); break;
    }} // local genome name
<COPY_USER_FUNCTION>"NB_GEN" |
<COPY_USER_FUNCTION>"NB_GEN"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"NB_GEN");
  else fprintf(fpOutputFile,"(*EZ_NB_GEN)");} // local genome name
<COPY_USER_FUNCTION>"POP_SIZE" |
<COPY_USER_FUNCTION>"POP_SIZE"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"POP_SIZE");
  else fprintf(fpOutputFile,"EZ_POP_SIZE");} // local genome name
<COPY_USER_FUNCTION>"MUT_PROB" |
<COPY_USER_FUNCTION>"MUT_PROB"/[ \t\n]*"==" {
  if (bDoubleQuotes) fprintf(fpOutputFile,"MUT_PROB");
  else if( TARGET==CUDA || TARGET==STD)
    fprintf(fpOutputFile,"(*pEZ_MUT_PROB)");
  else fprintf(fpOutputFile,"EZ_MUT_PROB");
 } // local genome name
<COPY_USER_FUNCTION>"XOVER_PROB" |
<COPY_USER_FUNCTION>"XOVER_PROB"/[ \t\n]*"==" {
  if (bDoubleQuotes) 
    fprintf(fpOutputFile,"XOVER_PROB");
  else if( TARGET==CUDA || TARGET==STD )
    fprintf(fpOutputFile,"(*pEZ_XOVER_PROB)");
  else fprintf(fpOutputFile,"EZ_XOVER_PROB");} // local genome name
<COPY_USER_FUNCTION>"REPL_PERC" |
<COPY_USER_FUNCTION>"REPL_PERC"/[ \t\n]*"==" {if (bDoubleQuotes) fprintf(fpOutputFile,"REPL_PERC");
  else fprintf(fpOutputFile,"EZ_REPL_PERC");} // local genome name

<COPY_USER_FUNCTION>"currentGeneration"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The current generation number cannot be changed (not an l-value).\n",sEZ_FILE_NAME,yylineno); exit(1);}
<COPY_USER_FUNCTION>"NB_GEN"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The number of generations can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY_USER_FUNCTION>"POP_SIZE"[ \t\n]*"="  {fprintf(stderr,"\n%s - Error line %d: The size of the population can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY_USER_FUNCTION>"MUT_PROB"[ \t\n]*"=" {
  fprintf(stderr,"\n%s - Error line %d: The mutation probability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); 
  exit (1);
}

<COPY_USER_FUNCTION>"XOVER_PROB"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The crossover proability can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY_USER_FUNCTION>"REPL_PERC"[ \t\n]*"=" {fprintf(stderr,"\n%s - Error line %d: The replacement percentage can only be changed within the generation function.\n",sEZ_FILE_NAME,yylineno); exit (1);}
<COPY_USER_FUNCTION>"false"  {fprintf(fpOutputFile,"false");} // local name
<COPY_USER_FUNCTION>"true"  {fprintf(fpOutputFile,"true");} // local name
<COPY_USER_FUNCTION>[^a-zA-Z0-9_]"bool"[^a-zA-Z0-9_]  {
  fprintf(fpOutputFile,yytext);}
<COPY_USER_FUNCTION>"tossCoin"  { // local random name
  fprintf(fpOutputFile,"globalRandomGenerator->tossCoin");}
<COPY_USER_FUNCTION>"random"  {
  fprintf(fpOutputFile,"globalRandomGenerator->random");}
<COPY_USER_FUNCTION>"child1"  {fprintf(fpOutputFile,"child1");
 }
<COPY_USER_FUNCTION>"child2"  {fprintf(fpOutputFile,"child2");
 }
<COPY_USER_FUNCTION>"parent1"  {fprintf(fpOutputFile,"parent1");
 }
<COPY_USER_FUNCTION>"parent2"  {fprintf(fpOutputFile,"parent2");
 }
<COPY_USER_FUNCTION>"identicalGenome"  {fprintf(fpOutputFile,"genome._evaluated");} // local name
<COPY_USER_FUNCTION>"return"  {if (bWithinEvaluator) {
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
  else fprintf(fpOutputFile,"return"); }
<COPY_USER_FUNCTION>";" {if (!bCatchNextSemiColon) fprintf(fpOutputFile,";");
  else if (bWithinMutator){fprintf(fpOutputFile,">0?true:false;");/* bWithinMutator=false;*/}
  else fprintf(fpOutputFile,"));");
  bCatchNextSemiColon=false;
 }
<COPY_USER_FUNCTION>"\\end" {rewind(fpGenomeFile);
  yyin = fpTemplateFile;
  BEGIN TEMPLATE_ANALYSIS;
  bWithinInitialiser=bWithinXover=bWithinMutator=bWithinEvaluator=bWithinOptimiser=bWithinCUDA_Evaluator=0;
  if (bWithinDisplayFunction){
    bWithinDisplayFunction=0; // display function
  }
  else return END_OF_FUNCTION;} // Back to the template file
<COPY_USER_FUNCTION>.|\n {putc(yytext[0],fpOutputFile);}                                      

//****************************************
//  Looks for run parameters in the .ez file
//****************************************

<PARAMETERS_ANALYSIS>"\\Default run"[ \t\n]+"parameters"[ \t\n]*":" {
  if (bVERBOSE) printf ("Analysing parameters...\n");
  BEGIN GET_PARAMETERS;
 }
<PARAMETERS_ANALYSIS>.|\n {}                                      

// takes care of C++-like comments
<GET_PARAMETERS>\/\/[^\n]* ;         // blah blah on a line containing sth else
<GET_PARAMETERS>^[ \t]*\/\/[^\n]*\n ; // blah blah on its own on a single line
<GET_PARAMETERS>\/\*[^\*]*\*\/ ;  /* blah blah on a line containing sth else */
<GET_PARAMETERS>^[ \t]*\/\*[^\*]*\*\/[ \t]*\n ; /* blah blah with nothing before the comment */
// gobbles up white spaces, tabs or carriage returns
<GET_PARAMETERS>[ \t]             { /* do nothing */ }
<GET_PARAMETERS>^[ \t]*\n      { /*return '\n';*/ }
<GET_PARAMETERS>\n                 { /*return '\n';*/ }

<GET_PARAMETERS>"Number"[ \t\n]+"of"[ \t\n]+"generations"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tNb of Gen...\n");return NB_GEN;}
<GET_PARAMETERS>"Time"[ \t\n]+"limit"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tTime Limit...\n");return TIME_LIMIT;}
<GET_PARAMETERS>"Population"[ \t\n]+"size"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tPop Size...\n");return POP_SIZE;}
<GET_PARAMETERS>"Elite"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tElite Size...\n");return ELITE;}
<GET_PARAMETERS>"Selection"[ \t\n]+"operator"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tSelection Operator...\n");return SELECTOR;}
<GET_PARAMETERS>"Selected"[ \t\n]+"genitors"[ \t\n]*":"[ \ta-zA-Z0-9_]* {if (bVERBOSE) printf ("\tSel Genitors...\n");}
<GET_PARAMETERS>"Mutation"[ \t\n]+"probability"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tMut Prob...\n");return MUT_PROB;}
<GET_PARAMETERS>"Crossover"[ \t\n]+"probability"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tXov Prob...\n");return XOVER_PROB;}
<GET_PARAMETERS>"Offspring"[ \t\n]+"size"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tOff Size...\n");return OFFSPRING;}

<GET_PARAMETERS>"Print"[ \t\n]+"stats"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf("\tPrint Stats...\n");return PRINT_STATS;}
<GET_PARAMETERS>"Plot"[ \t\n]+"stats"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf("\tPlot Stats with gnuplot...\n");return PLOT_STATS;}
<GET_PARAMETERS>"Generate"[ \t\n]+"csv"[ \t\n]+"stats"[ \t\n]+"file"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf("\tPrint Stats to csv File...\n");return GENERATE_CSV_FILE;}
<GET_PARAMETERS>"Generate"[ \t\n]+"gnuplot"[ \t\n]+"script"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf("\tGenerate Gnuplot Script...\n");return GENERATE_GNUPLOT_SCRIPT;}
<GET_PARAMETERS>"Generate"[ \t\n]+"R"[ \t\n]+"script"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf("\tGenerate R Script...\n");return GENERATE_R_SCRIPT;}

<GET_PARAMETERS>"Save"[ \t\n]+"population"[ \t\n]*":"[ \t\n]* {if(bVERBOSE) printf("\tSave population...\n"); return SAVE_POPULATION;}
<GET_PARAMETERS>"Start"[ \t\n]+"from"[ \t\n]+"file"[ \t\n]*":"[ \t\n]* {if(bVERBOSE) printf("\tStart from file...\n"); return START_FROM_FILE;}

<GET_PARAMETERS>"Reduce"[ \t\n]+"parents"[ \t\n]+"operator"[ \t\n]*":"[ \t\n]* {
  if (bVERBOSE) printf ("\tReduce Parents Operator...\n");
  bIsParentReduce = true;
  return RED_PAR;
 }
<GET_PARAMETERS>"Surviving"[ \t\n]+"parents"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tSurv Par...\n");return SURVPAR;}

<GET_PARAMETERS>"Reduce"[ \t\n]+"offspring"[ \t\n]+"operator"[ \t\n]*":"[ \t\n]* {
  if (bVERBOSE) printf ("\tReduce Offspring Operator...\n");
  bIsOffspringReduce = true;
  return RED_OFF;
 }
<GET_PARAMETERS>"Surviving"[ \t\n]+"offspring"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tSurv Off...\n");return SURVOFF;}


<GET_PARAMETERS>"Final"[ \t\n]+"reduce"[ \t\n]+"operator"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tFinal Reduce Operator...\n");return RED_FINAL;}// DISCARD;}
<GET_PARAMETERS>"Elitism"[ \t\n]*":" {if (bVERBOSE) printf ("\tElitism...\n");return ELITISM;}
<GET_PARAMETERS>"Evaluator"[ \t\n]+"goal"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tMinMax...\n");return MINIMAXI;}
<GET_PARAMETERS>"Number"[ \t\n]+"of"[ \t\n]+"optimisation"[ \t\n]+"iterations"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tNb of Optimisation It...\n");return NB_OPT_IT;}
<GET_PARAMETERS>"Baldwinism"[ \t\n]*":"[ \t\n]* {if (bVERBOSE) printf ("\tBaldwinism...\n");return BALDWINISM;}
 
<GET_PARAMETERS>"Remote"[ \t\n]+"island"[ \t\n]+"model"[ \t\n]*":"[ \t\n]* {if(bVERBOSE) printf ("\tRemote Island Model...\n"); return REMOTE_ISLAND_MODEL;}
<GET_PARAMETERS>"IP"[ \t\n]+"file"[ \t\n]*":"[ \t\n]* {if(bVERBOSE) printf("\tIP File...\n"); return IP_FILE;}
<GET_PARAMETERS>"Migration"[ \t\n]+"probability"[ \t\n]*":"[ \t\n]* {if(bVERBOSE) printf("\tMigration Probability...\n"); return MIGRATION_PROBABILITY;}
<GET_PARAMETERS>"Server"[ \t\n]+"port"[ \t\n]*":"[ \t\n]* {if(bVERBOSE) printf("\tServer port...\n"); return SERVER_PORT;}

 
// number
<GET_PARAMETERS>[0-9]+"."[0-9]*{exponent}?  |
<GET_PARAMETERS>"."[0-9]+{exponent}?  |
<GET_PARAMETERS>[0-9]+{exponent}     { yylval.dValue = myStrtod(); return NUMBER2; }
<GET_PARAMETERS>[0-9]+ {yylval.dValue=atof(yytext); return NUMBER2;}

// identifier
<GET_PARAMETERS>[a-zA-Z_][a-zA-Z0-9_]*    { yylval.pSymbol = new CSymbol(yytext);
  return IDENTIFIER2; }
// end of .ez file : resume tpl analysis
<GET_PARAMETERS>"\\end" {rewind(fpGenomeFile); yyin = fpTemplateFile; BEGIN TEMPLATE_ANALYSIS;}  // switch back to .tpl file and analyser

\n {
  lineCounter++;
}                          
<GET_PARAMETERS>[Mm]"ax"[ \t\n]+"init"[ \t\n]+"tree"[ \t\n]+"depth"[ \t\n]+":" {
  if( bVERBOSE ) printf("\tMax init tree depth...\n");
  return MAX_INIT_TREE_D;
 }            

<GET_PARAMETERS>[Mm]"in"[ \t\n]+"init"[ \t\n]+"tree"[ \t\n]+"depth"[ \t\n]+":" {
  if( bVERBOSE ) printf("\tMin init tree depth...\n");
  return MIN_INIT_TREE_D;
 }            

<GET_PARAMETERS>[Mm]"ax"[ \t\n]+"tree"[ \t\n]+"depth"[ \t\n]+":" {
  if( bVERBOSE ) printf("\tMax tree depth...\n");
  return MAX_TREE_D;
 }            

<GET_PARAMETERS>[Nn]"b"[ \t\n]+"of"[ \t\n]+"GPUs"[ \t\n]+":" {
  if( bVERBOSE ) printf("\tNo of GPUs...\n");
  return NB_GPU;
 }            

<GET_PARAMETERS>[Ss]"ize"[ \t\n]+"of"[ \t\n]+"prog"[ \t\n]+"buffer"[ \t\n]+":" {
  if( bVERBOSE ) printf("\tProgramm length buffer...\n");
  return PRG_BUF_SIZE;
 }            

<GET_PARAMETERS>[Nn]"b"[ \t\n]+"of"[ \t\n]+"fitness"[ \t\n]+"cases"[ \t\n]+":" {
  if( bVERBOSE ) printf("\tNo of fitness cases...\n");
  return NO_FITNESS_CASES;
 }            



//****************************************
// all other characters                      
.                      {return  (char)yytext[0];}

%%

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
