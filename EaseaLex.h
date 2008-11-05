#ifndef _EASEALEX_H
#define _EASEALEX_H

#include <clex.h>

#line 17 "C:\\repo\\src\\EaseaLex.l"

  // forward references
  class CEASEAParser;
  class CSymbolTable;
 
#line 13 "C:\\repo\\src\\EaseaLex.h"
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
#define COPY_DISPLAY 28
#define COPY_USER_FUNCTION 30
#define COPY_USER_GENERATION 32
#define PARAMETERS_ANALYSIS 34
#define GET_PARAMETERS 36
#define COPY_USER_FUNCTIONS 38
#define COPY_GENERATION_FUNCTION 40
#define GET_METHODS 42
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
#line 31 "C:\\repo\\src\\EaseaLex.l"

 protected:
  CSymbolTable *pSymbolTable;   // the symbol table
  bool bSymbolInserted,bWithinEvaluator;  // used to change evalutor type from double to float 
  bool bInitFunction,bDisplayFunction,bFunction, bNotFinishedYet, bWithinEO_Function;
  bool bDoubleQuotes,bWithinDisplayFunction,bWithinInitialiser,bWithinMutator,bWithinXover;
  bool bWaitingForSemiColon,bFinishNB_GEN,bFinishMINIMISE,bFinishMINIMIZE,bGenerationFunction;
  bool bCatchNextSemiColon,bWaitingToClosePopulation, bMethodsInGenome;
  CSymbol *pASymbol;

 public:
  int create(CEASEAParser* pParser, CSymbolTable* pSymbolTable);
  int yywrap();
  double myStrtod() const;                              
 
#line 67 "C:\\repo\\src\\EaseaLex.h"
};

#ifndef YYLEXNAME
#define YYLEXNAME CEASEALexer
#endif

#ifndef INITIAL
#define INITIAL 0
#endif

#endif
