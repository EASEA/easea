#ifndef _EASEAPARSE_H
#define _EASEAPARSE_H

#include <cyacc.h>

#line 72 "EaseaParse.y"

// forward references
class CSymbol;

#line 12 "EaseaParse.h"
#ifndef YYSTYPE
union tagYYSTYPE {
#line 78 "EaseaParse.y"

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
#define PATH_IDENTIFIER 270
#define BOOL 271
#define INT 272
#define DOUBLE 273
#define FLOAT 274
#define GPNODE 275
#define CHAR 276
#define POINTER 277
#define NUMBER 278
#define NUMBER2 279
#define METHODS 280
#define STATIC 281
#define NB_GEN 282
#define NB_OPT_IT 283
#define BALDWINISM 284
#define MUT_PROB 285
#define XOVER_PROB 286
#define POP_SIZE 287
#define SELECTOR 288
#define RED_PAR 289
#define RED_OFF 290
#define RED_FINAL 291
#define OFFSPRING 292
#define SURVPAR 293
#define SURVOFF 294
#define MINIMAXI 295
#define ELITISM 296
#define ELITE 297
#define REMOTE_ISLAND_MODEL 298
#define IP_FILE 299
#define EXPID 300
#define WORKING_PATH 301
#define MIGRATION_PROBABILITY 302
#define SERVER_PORT 303
#define PRINT_STATS 304
#define PLOT_STATS 305
#define GENERATE_CSV_IND_FILE 306
#define GENERATE_TXT_GEN_FILE 307
#define GENERATE_CSV_FILE 308
#define GENERATE_GNUPLOT_SCRIPT 309
#define GENERATE_R_SCRIPT 310
#define SAVE_POPULATION 311
#define START_FROM_FILE 312
#define TIME_LIMIT 313
#define MAX_INIT_TREE_D 314
#define MIN_INIT_TREE_D 315
#define MAX_XOVER_DEPTH 316
#define MAX_MUTAT_DEPTH 317
#define MAX_TREE_D 318
#define NB_GPU 319
#define PRG_BUF_SIZE 320
#define NO_FITNESS_CASES 321
#line 170 "EaseaParse.y"

#include "EaseaSym.h"
#include "EaseaLex.h"

#line 99 "EaseaParse.h"
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
#line 177 "EaseaParse.y"

protected:
  CEASEALexer EASEALexer;       // the lexical analyser
  
public:
  CSymbolTable SymbolTable;    // the symbol table

  int create();
  
  double assign(CSymbol* pIdentifier, double dValue);
  double divide(double dDividend, double dDivisor);
  CSymbol* insert() const;

  virtual void yysyntaxerror();

#line 136 "EaseaParse.h"
};

#ifndef YYPARSENAME
#define YYPARSENAME CEASEAParser
#endif

#endif
