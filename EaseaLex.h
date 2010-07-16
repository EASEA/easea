#ifndef _EASEALEX_H
#define _EASEALEX_H

#include <clex.h>

#line 67 "EaseaLex.l"

  // forward references
  class CEASEAParser;
  class CSymbolTable;
  class OPCodeDesc;
 
#line 14 "EaseaLex.h"
#define GENOME_ANALYSIS 2
#define TEMPLATE_ANALYSIS 4
#define MACRO_IDENTIFIER 6
#define MACRO_DEFINITION 8
#define COPY_USER_DECLARATIONS 10
#define COPY_INITIALISATION_FUNCTION 12
#define ANALYSE_USER_CLASSES 14
#define COPY_EO_INITIALISER 16
#define COPY_GP_OPCODE 18
#define COPY 20
#define COPY_INITIALISER 22
#define COPY_CROSSOVER 24
#define COPY_MUTATOR 26
#define COPY_EVALUATOR 28
#define COPY_OPTIMISER 30
#define COPY_FINALIZATION_FUNCTION 32
#define COPY_DISPLAY 34
#define COPY_USER_FUNCTION 36
#define COPY_USER_GENERATION 38
#define PARAMETERS_ANALYSIS 40
#define GET_PARAMETERS 42
#define COPY_USER_FUNCTIONS 44
#define COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT 46
#define GET_METHODS 48
#define COPY_MAKEFILE_OPTION 50
#define COPY_BOUND_CHECKING_FUNCTION 52
#define COPY_BEG_GENERATION_FUNCTION 54
#define COPY_END_GENERATION_FUNCTION 56
#define COPY_INSTEAD_EVAL 58
#define GP_RULE_ANALYSIS 60
#define GP_COPY_OPCODE_CODE 62
#define COPY_GP_EVAL 64
/////////////////////////////////////////////////////////////////////////////
// CEASEALexer

#ifndef YYDECLSPEC
#define YYDECLSPEC
#endif

class YYFAR YYDECLSPEC CEASEALexer : public yyflexer {
public:
	CEASEALexer();

protected:
	void yytables();
	virtual int yyaction(int action);

public:
#line 82 "EaseaLex.l"

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
 
#line 90 "EaseaLex.h"
};

#ifndef YYLEXNAME
#define YYLEXNAME CEASEALexer
#endif

#ifndef INITIAL
#define INITIAL 0
#endif

#endif
