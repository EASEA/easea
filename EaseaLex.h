#ifndef _EASEALEX_H
#define _EASEALEX_H

#include <clex.h>

#line 67 "EaseaLex.l"

  // forward references
  class CEASEAParser;
  class CSymbolTable;
 
#line 13 "EaseaLex.h"
#define GENOME_ANALYSIS 2
#define TEMPLATE_ANALYSIS 4
#define MACRO_IDENTIFIER 6
#define MACRO_DEFINITION 8
#define COPY_USER_DECLARATIONS 10
#define COPY_INITIALISATION_FUNCTION 12
#define ANALYSE_USER_CLASSES 14
#define COPY_EO_INITIALISER 16
#define COPY 18
#define COPY_INITIALISER 20
#define COPY_CROSSOVER 22
#define COPY_MUTATOR 24
#define COPY_EVALUATOR 26
#define COPY_OPTIMISER 28
#define COPY_FINALIZATION_FUNCTION 30
#define COPY_DISPLAY 32
#define COPY_USER_FUNCTION 34
#define COPY_USER_GENERATION 36
#define PARAMETERS_ANALYSIS 38
#define GET_PARAMETERS 40
#define COPY_USER_FUNCTIONS 42
#define COPY_GENERATION_FUNCTION_BEFORE_REPLACEMENT 44
#define GET_METHODS 46
#define COPY_MAKEFILE_OPTION 48
#define COPY_BOUND_CHECKING_FUNCTION 50
#define COPY_BEG_GENERATION_FUNCTION 52
#define COPY_END_GENERATION_FUNCTION 54
#define COPY_INSTEAD_EVAL 56
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
#line 81 "EaseaLex.l"

 protected:
  CSymbolTable *pSymbolTable;   // the symbol table
  bool bSymbolInserted,bWithinEvaluator, bWithinOptimiser;  // used to change evalutor type from double to float 
  bool bInitFunction,bDisplayFunction,bFunction, bNotFinishedYet, bWithinEO_Function;
  bool bDoubleQuotes,bWithinDisplayFunction,bWithinInitialiser,bWithinMutator,bWithinXover;
  bool bWaitingForSemiColon,bFinishNB_GEN,bFinishMINIMISE,bFinishMINIMIZE,bGenerationReplacementFunction;
  bool bCatchNextSemiColon,bWaitingToClosePopulation, bMethodsInGenome, bFinalizationFunction;
  bool bWithinCUDA_Initializer, bWithinMAKEFILEOPTION, bWithinCUDA_Evaluator, bBoundCheckingFunction;
  bool bIsParentReduce, bIsOffspringReduce, bEndGeneration, bBeginGeneration, bEndGenerationFunction, bBeginGenerationFunction, bGenerationFunctionBeforeReplacement;
  CSymbol *pASymbol;

 public:
  int create(CEASEAParser* pParser, CSymbolTable* pSymbolTable);
  int yywrap();
  double myStrtod() const;                              
 
#line 76 "EaseaLex.h"
};

#ifndef YYLEXNAME
#define YYLEXNAME CEASEALexer
#endif

#ifndef INITIAL
#define INITIAL 0
#endif

#endif
