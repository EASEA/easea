#ifndef _EASEAPARSE_H
#define _EASEAPARSE_H

#include <cyacc.h>

#line 69 "EaseaParse.y"

// forward references
class CSymbol;

#line 12 "EaseaParse.h"
#ifndef YYSTYPE
union tagYYSTYPE {
#line 75 "EaseaParse.y"

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
#define GPNODE 274
#define CHAR 275
#define POINTER 276
#define NUMBER 277
#define NUMBER2 278
#define METHODS 279
#define STATIC 280
#define NB_GEN 281
#define NB_OPT_IT 282
#define BALDWINISM 283
#define MUT_PROB 284
#define XOVER_PROB 285
#define POP_SIZE 286
#define SELECTOR 287
#define RED_PAR 288
#define RED_OFF 289
#define RED_FINAL 290
#define OFFSPRING 291
#define SURVPAR 292
#define SURVOFF 293
#define MINIMAXI 294
#define ELITISM 295
#define ELITE 296
#define REMOTE_ISLAND_MODEL 297
#define IP_FILE 298
#define MIGRATION_PROBABILITY 299
#define PRINT_STATS 300
#define PLOT_STATS 301
#define GENERATE_CSV_FILE 302
#define GENERATE_GNUPLOT_SCRIPT 303
#define GENERATE_R_SCRIPT 304
#define SAVE_POPULATION 305
#define START_FROM_FILE 306
#define TIME_LIMIT 307
#define MAX_INIT_TREE_D 308
#define MIN_INIT_TREE_D 309
#define MAX_XOVER_DEPTH 310
#define MAX_MUTAT_DEPTH 311
#define MAX_TREE_D 312
#define NB_GPU 313
#define PRG_BUF_SIZE 314
#define NO_FITNESS_CASES 315
#line 159 "EaseaParse.y"

#include "EaseaSym.h"
#include "EaseaLex.h"

#line 93 "EaseaParse.h"
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
#line 166 "EaseaParse.y"

protected:
  CEASEALexer EASEALexer;       // the lexical analyser
  
public:
  CSymbolTable SymbolTable;    // the symbol table

  int create();
  
  double assign(CSymbol* pIdentifier, double dValue);
  double divide(double dDividend, double dDivisor);
  CSymbol* insert() const;

  virtual void yysyntaxerror();

#line 130 "EaseaParse.h"
};

#ifndef YYPARSENAME
#define YYPARSENAME CEASEAParser
#endif

#endif
