#ifndef _EASEAPARSE_H
#define _EASEAPARSE_H

#include <cyacc.h>

#line 58 "C:\\repo\\src\\EaseaParse.y"

// forward references
class CSymbol;

#line 12 "C:\\repo\\src\\EaseaParse.h"
#ifndef YYSTYPE
union tagYYSTYPE {
#line 64 "C:\\repo\\src\\EaseaParse.y"

  CSymbol* pSymbol;
  double dValue;
  int ObjectQualifier;
  int nValue;
  char *szString;

#line 23 "C:\\repo\\src\\EaseaParse.h"
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
#define END_OF_FUNCTION 264
#define END_METHODS 265
#define IDENTIFIER 266
#define IDENTIFIER2 267
#define BOOL 268
#define INT 269
#define DOUBLE 270
#define FLOAT 271
#define CHAR 272
#define POINTER 273
#define NUMBER 274
#define NUMBER2 275
#define METHODS 276
#define STATIC 277
#define NB_GEN 278
#define NB_ISLANDS 279
#define PROP_SEQ 280
#define MUT_PROB 281
#define XOVER_PROB 282
#define POP_SIZE 283
#define SELECTOR 284
#define RED_PAR 285
#define RED_OFF 286
#define RED_FINAL 287
#define OFFSPRING 288
#define SURVPAR 289
#define SURVOFF 290
#define REPLACEMENT 291
#define DISCARD 292
#define MINIMAXI 293
#define ELITISM 294
#define ELITE 295
#define MIG_CLONE 296
#define MIG_SEL 297
#define MIGRATOR 298
#define MIG_FREQ 299
#define NB_MIG 300
#define IMMIG_SEL 301
#define IMMIG_REPL 302
#line 136 "C:\\repo\\src\\EaseaParse.y"

#include "EaseaSym.h"
#include "EaseaLex.h"

#line 80 "C:\\repo\\src\\EaseaParse.h"
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
#line 143 "C:\\repo\\src\\EaseaParse.y"

protected:
  CEASEALexer EASEALexer;       // the lexical analyser
  
public:
  CSymbolTable SymbolTable;    // the symbol table

  int create();
  
  double assign(CSymbol* pIdentifier, double dValue);
  double divide(double dDividend, double dDivisor);
  void yysyntaxerror();
  CSymbol* insert() const;

#line 115 "C:\\repo\\src\\EaseaParse.h"
};

#ifndef YYPARSENAME
#define YYPARSENAME CEASEAParser
#endif

#endif
