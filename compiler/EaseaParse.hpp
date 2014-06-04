/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison interface for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     UMINUS = 258,
     CLASSES = 259,
     GENOME = 260,
     USER_CTOR = 261,
     USER_XOVER = 262,
     USER_MUTATOR = 263,
     USER_EVALUATOR = 264,
     USER_OPTIMISER = 265,
     MAKEFILE_OPTION = 266,
     END_OF_FUNCTION = 267,
     END_METHODS = 268,
     IDENTIFIER = 269,
     IDENTIFIER2 = 270,
     BOOL = 271,
     INT = 272,
     DOUBLE = 273,
     FLOAT = 274,
     GPNODE = 275,
     CHAR = 276,
     POINTER = 277,
     NUMBER = 278,
     NUMBER2 = 279,
     METHODS = 280,
     STATIC = 281,
     NB_GEN = 282,
     NB_OPT_IT = 283,
     BALDWINISM = 284,
     MUT_PROB = 285,
     XOVER_PROB = 286,
     POP_SIZE = 287,
     SELECTOR = 288,
     RED_PAR = 289,
     RED_OFF = 290,
     RED_FINAL = 291,
     OFFSPRING = 292,
     SURVPAR = 293,
     SURVOFF = 294,
     MINIMAXI = 295,
     ELITISM = 296,
     ELITE = 297,
     REMOTE_ISLAND_MODEL = 298,
     IP_FILE = 299,
     MIGRATION_PROBABILITY = 300,
     SERVER_PORT = 301,
     PRINT_STATS = 302,
     PLOT_STATS = 303,
     GENERATE_CSV_FILE = 304,
     GENERATE_GNUPLOT_SCRIPT = 305,
     GENERATE_R_SCRIPT = 306,
     SAVE_POPULATION = 307,
     START_FROM_FILE = 308,
     TIME_LIMIT = 309,
     MAX_INIT_TREE_D = 310,
     MIN_INIT_TREE_D = 311,
     MAX_XOVER_DEPTH = 312,
     MAX_MUTAT_DEPTH = 313,
     MAX_TREE_D = 314,
     NB_GPU = 315,
     PRG_BUF_SIZE = 316,
     NO_FITNESS_CASES = 317,
     TEMPLATE_END = 318
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{


  CSymbol* pSymbol;
  double dValue;
  int ObjectQualifier;
  int nValue;
  char *szString;



} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;


/* "%code provides" blocks.  */


// forward references
class CSymbol;



#include "EaseaSym.h"
#include "EaseaLex.h"

  int  CEASEAParser_create();
  
  double  CEASEAParser_assign(CSymbol* pIdentifier, double dValue);
  double  CEASEAParser_divide(double dDividend, double dDivisor);
  CSymbol*  CEASEAParser_insert();

  //virtual void yysyntaxerror();




