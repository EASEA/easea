#ifndef _EASEAPARSE_H
#define _EASEAPARSE_H

#include <cyacc.h>

#line 63 "EaseaParse.y"

// forward references
class CSymbol;

#line 12 "EaseaParse.h"
#ifndef YYSTYPE
union tagYYSTYPE {
#line 69 "EaseaParse.y"

  CSymbol* pSymbol;
  double dValue;
  int ObjectQualifier;
  int nValue;
  char *szString;

#line 23 "EaseaParse.h"
};

#define YYSTYPE union tagYYSTYPE
#endif

#define UMINUS 257
#define CLASSES 258
#define GENOME 259
#define USER_CTOR 260
#define USER_XOVER 261
#define USER_MUTATOR 262
#define USER_EVALUATOR 263
#define USER_OPTIMISER 264
#define MAKEFILE_OPTION 265
#define END_OF_FUNCTION 266
#define END_METHODS 267
#define IDENTIFIER 268
#define IDENTIFIER2 269
#define BOOL 270
#define INT 271
#define DOUBLE 272
#define FLOAT 273
#define CHAR 274
#define POINTER 275
#define NUMBER 276
#define NUMBER2 277
#define METHODS 278
#define STATIC 279
#define NB_GEN 280
#define NB_OPT_IT 281
#define BALDWINISM 282
#define MUT_PROB 283
#define XOVER_PROB 284
#define POP_SIZE 285
#define SELECTOR 286
#define RED_PAR 287
#define RED_OFF 288
#define RED_FINAL 289
#define OFFSPRING 290
#define SURVPAR 291
#define SURVOFF 292
#define MINIMAXI 293
#define ELITISM 294
#define ELITE 295
#define PRINT_STATS 296
#define PLOT_STATS 297
#define GENERATE_CSV_FILE 298
#define GENERATE_GNUPLOT_SCRIPT 299
#define GENERATE_R_SCRIPT 300
#define TIME_LIMIT 301
#line 139 "EaseaParse.y"

#include "EaseaSym.h"
#include "EaseaLex.h"

#line 79 "EaseaParse.h"
/////////////////////////////////////////////////////////////////////////////
// CEASEAParser

#ifndef YYDECLSPEC
#define YYDECLSPEC
#endif

class YYFAR YYDECLSPEC CEASEAParser : public yyfparser {
public:
	CEASEAParser();

protected:
	void yytables();
	virtual void yyaction(int action);
#ifdef YYDEBUG
	void YYFAR* yyattribute1(int index) const;
	void yyinitdebug(void YYFAR** p, int count) const;
#endif

public:
#line 146 "EaseaParse.y"

protected:
  CEASEALexer EASEALexer;       // the lexical analyser
  
public:
  CSymbolTable SymbolTable;    // the symbol table

  int create();
  
  double assign(CSymbol* pIdentifier, double dValue);
  double divide(double dDividend, double dDivisor);
  CSymbol* insert() const;

#line 114 "EaseaParse.h"
};

#ifndef YYPARSENAME
#define YYPARSENAME CEASEAParser
#endif

#endif
