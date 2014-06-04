/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison implementation for Yacc-like parsers in C
   
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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Copy the first part of user declarations.  */


/****************************************************************************
EaseaLex.y
Parser for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@polytechnique.fr)
Ecole Polytechnique
Centre de Math�matiques Appliqu�es
91128 Palaiseau cedex
****************************************************************************/

#include "Easea.h"
#include "debug.h"
#include "EaseaYTools.h"
#include "EaseaParse.hpp"

#define YYEXIT_FAILURE	1
#define YYEXIT_SUCCESS	0



// Globals     
CSymbol *pCURRENT_CLASS;
CSymbol *pCURRENT_TYPE;
CSymbol *pGENOME;
CSymbol* pCLASSES[128];
char sRAW_PROJECT_NAME[1000];
int nClasses_nb = 0;
char sPROJECT_NAME[1000];
char sLOWER_CASE_PROJECT_NAME[1000];
char sEZ_FILE_NAME[1000];
char sEO_DIR[1000];
char sEZ_PATH[1000];
char sTPL_DIR[1000];
int TARGET,TARGET_FLAVOR;
int OPERATING_SYSTEM;
int nWARNINGS=0;
int nERRORS=0;
char sSELECTOR[50], sSELECTOR_OPERATOR[50];
float fSELECT_PRM=0.0;
char sRED_PAR[50], sRED_PAR_OPERATOR[50];
float fRED_PAR_PRM=0.0;//[50] = {0};
char sRED_OFF[50], sRED_OFF_OPERATOR[50];
float fRED_OFF_PRM;//[50] = {0};
char sRED_FINAL[50], sRED_FINAL_OPERATOR[50];
float fRED_FINAL_PRM=0.0;//[50];
int nMINIMISE=2;
int nELITE=0;
bool bELITISM=0;
bool bVERBOSE=0;
bool bLINE_NUM_EZ_FILE=1;
bool bPRINT_STATS=1;
bool bPLOT_STATS=0;
bool bGENERATE_CSV_FILE=0, bGENERATE_R_SCRIPT=0, bGENERATE_GNUPLOT_SCRIPT=0;
bool bSAVE_POPULATION=0, bSTART_FROM_FILE=0;
bool bBALDWINISM=0; //memetic
bool bREMOTE_ISLAND_MODEL=0; //remote island model
float fMIGRATION_PROBABILITY=0.0;
char sIP_FILE[128]; //remote island model
int nPOP_SIZE, nOFF_SIZE;
float fSURV_PAR_SIZE=-1.0, fSURV_OFF_SIZE=-1.0;
char *nGENOME_NAME;
int nPROBLEM_DIM;
int nNB_GEN=0;
int nNB_OPT_IT=0;
int nTIME_LIMIT=0;
int nSERVER_PORT=0;
float fMUT_PROB;
float fXOVER_PROB;
FILE *fpOutputFile, *fpTemplateFile, *fpGenomeFile;//, *fpExplodedGenomeFile;

CSymbolTable SymbolTable;    // the symbol table


 unsigned iMAX_INIT_TREE_D,iMIN_INIT_TREE_D,iMAX_TREE_D,iNB_GPU,iPRG_BUF_SIZE,iMAX_TREE_DEPTH,iMAX_XOVER_DEPTH,iNO_FITNESS_CASES;

extern int yylex();
extern char * yytext;

void yyerror(const char * s){
  printf("%s\nSyntax Error at line : %d (on text : %s)\nFor more details during the EASEA compiling, use the \"-v\" option\n",
	 s,yylineno,yytext);
}




/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


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





/* Copy the second part of user declarations.  */



#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  72
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   223

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  81
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  39
/* YYNRULES -- Number of rules.  */
#define YYNRULES  119
/* YYNRULES -- Number of states.  */
#define YYNSTATES  198

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   318

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,    77,     2,     2,
      79,    80,     6,     4,    73,     5,    78,     7,    74,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    72,    71,
       2,     3,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    75,     2,    76,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    69,     2,    70,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     7,     8,    13,    14,    18,    19,    23,
      25,    27,    30,    31,    37,    39,    42,    43,    48,    49,
      54,    56,    59,    61,    62,    65,    70,    73,    78,    80,
      82,    84,    86,    88,    90,    92,    94,    96,   100,   102,
     105,   108,   112,   117,   123,   125,   128,   130,   131,   137,
     139,   142,   144,   146,   148,   151,   152,   156,   157,   161,
     162,   166,   167,   171,   172,   176,   179,   181,   183,   186,
     189,   192,   195,   198,   201,   204,   207,   211,   214,   218,
     221,   225,   228,   232,   235,   239,   242,   246,   249,   253,
     256,   259,   263,   266,   269,   272,   277,   280,   283,   286,
     289,   292,   295,   298,   301,   304,   307,   310,   313,   316,
     319,   322,   326,   330,   334,   338,   342,   346,   349,   351
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      82,     0,    -1,   117,    83,    68,    -1,    -1,    86,   105,
      84,   110,    -1,    -1,   105,    85,   110,    -1,    -1,     9,
      87,    88,    -1,     9,    -1,    89,    -1,    88,    89,    -1,
      -1,   109,    90,    69,    91,    70,    -1,    92,    -1,    91,
      92,    -1,    -1,    96,    99,    93,    97,    -1,    -1,    96,
     100,    94,    98,    -1,    95,    -1,    30,    18,    -1,    31,
      -1,    -1,   101,    71,    -1,   101,    72,   103,    71,    -1,
     101,    71,    -1,   101,    72,   107,    71,    -1,    21,    -1,
      22,    -1,    23,    -1,    24,    -1,    26,    -1,    27,    -1,
      25,    -1,   109,    -1,   102,    -1,   101,    73,   102,    -1,
     109,    -1,     6,   109,    -1,    74,   109,    -1,     6,     6,
     109,    -1,   109,    75,   119,    76,    -1,     6,   109,    75,
     119,    76,    -1,   104,    -1,   103,   104,    -1,    28,    -1,
      -1,    10,   106,    69,    91,    70,    -1,   108,    -1,   107,
     108,    -1,   109,    -1,    19,    -1,   111,    -1,   110,   111,
      -1,    -1,    11,   112,    17,    -1,    -1,    12,   113,    17,
      -1,    -1,    13,   114,    17,    -1,    -1,    14,   115,    17,
      -1,    -1,    15,   116,    17,    -1,    16,    17,    -1,    16,
      -1,   118,    -1,   117,   118,    -1,    32,    29,    -1,    33,
      29,    -1,    59,    29,    -1,    35,    29,    -1,    36,    29,
      -1,    37,    29,    -1,    38,    20,    -1,    38,    20,    29,
      -1,    39,    20,    -1,    39,    20,    29,    -1,    40,    20,
      -1,    40,    20,    29,    -1,    41,    20,    -1,    41,    20,
      29,    -1,    42,    29,    -1,    42,    29,    77,    -1,    43,
      29,    -1,    43,    29,    77,    -1,    44,    29,    -1,    44,
      29,    77,    -1,    45,    20,    -1,    47,    29,    -1,    47,
      29,    77,    -1,    46,    20,    -1,    34,    20,    -1,    48,
      20,    -1,    49,    20,    78,    20,    -1,    50,    29,    -1,
      51,    29,    -1,    52,    20,    -1,    53,    20,    -1,    54,
      20,    -1,    55,    20,    -1,    56,    20,    -1,    57,    20,
      -1,    58,    20,    -1,    60,    29,    -1,    61,    29,    -1,
      64,    29,    -1,    65,    29,    -1,    66,    29,    -1,    67,
      29,    -1,    19,     3,   119,    -1,   119,     4,   119,    -1,
     119,     5,   119,    -1,   119,     6,   119,    -1,   119,     7,
     119,    -1,    79,   119,    80,    -1,     5,   119,    -1,    28,
      -1,    19,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   195,   195,   198,   198,   203,   203,   211,   211,   214,
     219,   220,   224,   224,   238,   239,   244,   244,   245,   245,
     246,   250,   258,   259,   263,   264,   268,   269,   273,   274,
     275,   276,   277,   278,   279,   283,   295,   296,   304,   316,
     326,   336,   349,   363,   386,   387,   391,   404,   404,   426,
     427,   431,   435,   439,   440,   444,   444,   448,   448,   452,
     452,   456,   456,   460,   460,   464,   467,   471,   472,   476,
     478,   480,   482,   484,   486,   488,   498,   508,   518,   528,
     538,   546,   555,   564,   565,   566,   567,   568,   569,   570,
     579,   582,   585,   592,   600,   607,   612,   615,   618,   625,
     632,   639,   646,   653,   660,   667,   668,   669,   670,   671,
     674,   678,   685,   686,   687,   688,   692,   693,   694,   695
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'='", "'+'", "'-'", "'*'", "'/'",
  "UMINUS", "CLASSES", "GENOME", "USER_CTOR", "USER_XOVER", "USER_MUTATOR",
  "USER_EVALUATOR", "USER_OPTIMISER", "MAKEFILE_OPTION", "END_OF_FUNCTION",
  "END_METHODS", "IDENTIFIER", "IDENTIFIER2", "BOOL", "INT", "DOUBLE",
  "FLOAT", "GPNODE", "CHAR", "POINTER", "NUMBER", "NUMBER2", "METHODS",
  "STATIC", "NB_GEN", "NB_OPT_IT", "BALDWINISM", "MUT_PROB", "XOVER_PROB",
  "POP_SIZE", "SELECTOR", "RED_PAR", "RED_OFF", "RED_FINAL", "OFFSPRING",
  "SURVPAR", "SURVOFF", "MINIMAXI", "ELITISM", "ELITE",
  "REMOTE_ISLAND_MODEL", "IP_FILE", "MIGRATION_PROBABILITY", "SERVER_PORT",
  "PRINT_STATS", "PLOT_STATS", "GENERATE_CSV_FILE",
  "GENERATE_GNUPLOT_SCRIPT", "GENERATE_R_SCRIPT", "SAVE_POPULATION",
  "START_FROM_FILE", "TIME_LIMIT", "MAX_INIT_TREE_D", "MIN_INIT_TREE_D",
  "MAX_XOVER_DEPTH", "MAX_MUTAT_DEPTH", "MAX_TREE_D", "NB_GPU",
  "PRG_BUF_SIZE", "NO_FITNESS_CASES", "TEMPLATE_END", "'{'", "'}'", "';'",
  "':'", "','", "'0'", "'['", "']'", "'%'", "'.'", "'('", "')'", "$accept",
  "EASEA", "GenomeAnalysis", "$@1", "$@2", "ClassDeclarationsSection",
  "$@3", "ClassDeclarations", "ClassDeclaration", "$@4",
  "VariablesDeclarations", "VariablesDeclaration", "$@5", "$@6",
  "MethodsDeclaration", "Qualifier", "BaseObjects", "UserObjects",
  "BaseType", "UserType", "Objects", "Object", "BaseConstructorParameters",
  "BaseConstructorParameter", "GenomeDeclarationSection", "$@7",
  "UserConstructorParameters", "UserConstructorParameter", "Symbol",
  "StandardFunctionsAnalysis", "StandardFunctionAnalysis", "$@8", "$@9",
  "$@10", "$@11", "$@12", "RunParameters", "Parameter", "Expr", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,    61,    43,    45,    42,    47,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,   294,   295,   296,   297,   298,   299,
     300,   301,   302,   303,   304,   305,   306,   307,   308,   309,
     310,   311,   312,   313,   314,   315,   316,   317,   318,   123,
     125,    59,    58,    44,    48,    91,    93,    37,    46,    40,
      41
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    81,    82,    84,    83,    85,    83,    87,    86,    86,
      88,    88,    90,    89,    91,    91,    93,    92,    94,    92,
      92,    95,    96,    96,    97,    97,    98,    98,    99,    99,
      99,    99,    99,    99,    99,   100,   101,   101,   102,   102,
     102,   102,   102,   102,   103,   103,   104,   106,   105,   107,
     107,   108,   109,   110,   110,   112,   111,   113,   111,   114,
     111,   115,   111,   116,   111,   111,   111,   117,   117,   118,
     118,   118,   118,   118,   118,   118,   118,   118,   118,   118,
     118,   118,   118,   118,   118,   118,   118,   118,   118,   118,
     118,   118,   118,   118,   118,   118,   118,   118,   118,   118,
     118,   118,   118,   118,   118,   118,   118,   118,   118,   118,
     118,   119,   119,   119,   119,   119,   119,   119,   119,   119
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     3,     0,     4,     0,     3,     0,     3,     1,
       1,     2,     0,     5,     1,     2,     0,     4,     0,     4,
       1,     2,     1,     0,     2,     4,     2,     4,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     1,     2,
       2,     3,     4,     5,     1,     2,     1,     0,     5,     1,
       2,     1,     1,     1,     2,     0,     3,     0,     3,     0,
       3,     0,     3,     0,     3,     2,     1,     1,     2,     2,
       2,     2,     2,     2,     2,     2,     3,     2,     3,     2,
       3,     2,     3,     2,     3,     2,     3,     2,     3,     2,
       2,     3,     2,     2,     2,     4,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     3,     3,     3,     3,     3,     3,     2,     1,     1
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    67,    69,    70,
      93,    72,    73,    74,    75,    77,    79,    81,    83,    85,
      87,    89,    92,    90,    94,     0,    96,    97,    98,    99,
     100,   101,   102,   103,   104,    71,   105,   106,   107,   108,
     109,   110,     1,     7,    47,     0,     0,     5,    68,    76,
      78,    80,    82,    84,    86,    88,    91,     0,     0,     0,
       2,     3,     0,    95,    52,     8,    10,    12,    23,     0,
      55,    57,    59,    61,    63,    66,     6,    53,    11,     0,
       0,    22,    23,    14,    20,     0,     4,     0,     0,     0,
       0,     0,    65,    54,    23,    21,    48,    15,    28,    29,
      30,    31,    34,    32,    33,    16,    18,    35,    56,    58,
      60,    62,    64,    23,     0,     0,    13,     0,     0,    17,
       0,    36,    38,    19,     0,     0,    39,    40,    24,     0,
       0,     0,    26,     0,    41,     0,    46,     0,    44,    37,
       0,   119,   118,     0,     0,     0,    49,    51,     0,    25,
      45,   117,     0,     0,     0,     0,     0,     0,    42,    27,
      50,    43,   111,   116,   112,   113,   114,   115
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    35,    75,    99,    92,    76,    88,    95,    96,   109,
     112,   113,   144,   145,   114,   115,   149,   153,   135,   136,
     150,   151,   167,   168,    77,    89,   175,   176,   152,   106,
     107,   117,   118,   119,   120,   121,    36,    37,   174
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -145
static const yytype_int16 yypact[] =
{
      92,    10,    16,    43,    45,    47,    49,    63,    66,    98,
      99,   125,   126,   135,   145,   146,   138,   148,   149,   141,
     142,   152,   153,   154,   155,   157,   158,   159,   147,   151,
     160,   161,   162,   163,   164,   181,    56,  -145,  -145,  -145,
    -145,  -145,  -145,  -145,   165,   166,   167,   168,   105,   106,
     107,  -145,  -145,   108,  -145,   109,  -145,  -145,  -145,  -145,
    -145,  -145,  -145,  -145,  -145,  -145,  -145,  -145,  -145,  -145,
    -145,  -145,  -145,   176,  -145,   120,   188,  -145,  -145,  -145,
    -145,  -145,  -145,  -145,  -145,  -145,  -145,   179,   182,   131,
    -145,  -145,    57,  -145,  -145,   182,  -145,  -145,     1,    57,
    -145,  -145,  -145,  -145,  -145,   185,    57,  -145,  -145,   134,
     186,  -145,    -8,  -145,  -145,    27,    57,   189,   190,   191,
     192,   193,  -145,  -145,     1,  -145,  -145,  -145,  -145,  -145,
    -145,  -145,  -145,  -145,  -145,  -145,  -145,  -145,  -145,  -145,
    -145,  -145,  -145,    -6,    11,    11,  -145,    14,   182,  -145,
     -36,  -145,   130,  -145,   -16,   182,   136,  -145,  -145,   184,
      11,     0,  -145,   182,  -145,     0,  -145,   -10,  -145,  -145,
       0,   210,  -145,     0,     4,   -13,  -145,  -145,     8,  -145,
    -145,  -145,     0,    -3,     0,     0,     0,     0,  -145,  -145,
    -145,  -145,   156,  -145,    75,    75,  -145,  -145
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -145,  -145,  -145,  -145,  -145,  -145,  -145,  -145,   119,  -145,
      91,   -96,  -145,  -145,  -145,  -145,  -145,  -145,  -145,  -145,
      71,    58,  -145,    50,   143,  -145,  -145,    46,   -88,   121,
     -72,  -145,  -145,  -145,  -145,  -145,  -145,   187,  -144
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -10
static const yytype_int16 yytable[] =
{
      97,   184,   185,   186,   187,   170,    94,    97,   184,   185,
     186,   187,   184,   185,   186,   187,   127,   147,   166,   171,
     155,   178,   110,   111,   110,   111,   181,   137,   172,   183,
      94,   110,   111,    94,   123,   158,   159,   160,   192,    38,
     194,   195,   196,   197,   123,    39,    94,   127,   128,   129,
     130,   131,   132,   133,   134,   162,   163,   160,   189,   156,
     157,   179,   126,    40,   146,    73,    74,   164,   100,   101,
     102,   103,   104,   105,    41,   177,    42,   193,    43,   173,
     188,   186,   187,    44,   191,   148,    45,   177,     1,     2,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    46,    47,
      31,    32,    33,    34,     1,     2,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    48,    49,    31,    32,    33,    34,
     184,   185,   186,   187,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    65,    62,    63,    64,
      66,    72,    83,    84,    85,    86,    -9,    87,    90,    67,
      68,    69,    70,    71,    79,    80,    81,    82,    74,    93,
      98,    94,   122,   124,   125,   161,   138,   139,   140,   141,
     142,   165,   166,   182,   108,   143,   154,   180,   169,    91,
     116,   190,     0,    78
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-145))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
      88,     4,     5,     6,     7,     5,    19,    95,     4,     5,
       6,     7,     4,     5,     6,     7,   112,     6,    28,    19,
       6,   165,    30,    31,    30,    31,   170,   115,    28,   173,
      19,    30,    31,    19,   106,    71,    72,    73,   182,    29,
     184,   185,   186,   187,   116,    29,    19,   143,    21,    22,
      23,    24,    25,    26,    27,    71,    72,    73,    71,   147,
     148,    71,    70,    20,    70,     9,    10,   155,    11,    12,
      13,    14,    15,    16,    29,   163,    29,    80,    29,    79,
      76,     6,     7,    20,    76,    74,    20,   175,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    20,    20,
      64,    65,    66,    67,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    29,    29,    64,    65,    66,    67,
       4,     5,     6,     7,    29,    20,    20,    29,    20,    20,
      29,    29,    20,    20,    20,    20,    29,    20,    20,    20,
      29,     0,    77,    77,    77,    77,    10,    78,    68,    29,
      29,    29,    29,    29,    29,    29,    29,    29,    10,    20,
      69,    19,    17,    69,    18,    75,    17,    17,    17,    17,
      17,    75,    28,     3,    95,   124,   145,   167,   160,    76,
      99,   175,    -1,    36
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    64,    65,    66,    67,    82,   117,   118,    29,    29,
      20,    29,    29,    29,    20,    20,    20,    20,    29,    29,
      29,    20,    20,    29,    20,    20,    29,    29,    20,    20,
      20,    20,    20,    20,    20,    29,    29,    29,    29,    29,
      29,    29,     0,     9,    10,    83,    86,   105,   118,    29,
      29,    29,    29,    77,    77,    77,    77,    78,    87,   106,
      68,   105,    85,    20,    19,    88,    89,   109,    69,    84,
      11,    12,    13,    14,    15,    16,   110,   111,    89,    90,
      30,    31,    91,    92,    95,    96,   110,   112,   113,   114,
     115,   116,    17,   111,    69,    18,    70,    92,    21,    22,
      23,    24,    25,    26,    27,    99,   100,   109,    17,    17,
      17,    17,    17,    91,    93,    94,    70,     6,    74,    97,
     101,   102,   109,    98,   101,     6,   109,   109,    71,    72,
      73,    75,    71,    72,   109,    75,    28,   103,   104,   102,
       5,    19,    28,    79,   119,   107,   108,   109,   119,    71,
     104,   119,     3,   119,     4,     5,     6,     7,    76,    71,
     108,    76,   119,    80,   119,   119,   119,   119
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL		goto yyerrlab
#if defined YYFAIL
  /* This is here to suppress warnings from the GCC cpp's
     -Wunused-macros.  Normally we don't worry about that warning, but
     some users do, and we want to make it easy for users to remove
     YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* This macro is provided for backward compatibility. */

#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = 0;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                yysize1 = yysize + yytnamerr (0, yytname[yyx]);
                if (! (yysize <= yysize1
                       && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                  return 2;
                yysize = yysize1;
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  yysize1 = yysize + yystrlen (yyformat);
  if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
    return 2;
  yysize = yysize1;

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:

    {return 0;}
    break;

  case 3:

    {
        if (bVERBOSE){ printf("                    _______________________________________\n");
        printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);}
      }
    break;

  case 5:

    {
        if (bVERBOSE) printf("                    _______________________________________\n");
        if (bVERBOSE) printf ("\nGeneration of the C++ source file for %s.\n\n",sPROJECT_NAME);
      }
    break;

  case 7:

    {
    if (bVERBOSE) printf("Declaration of user classes :\n\n");}
    break;

  case 9:

    {
      if (bVERBOSE) printf("No user class declaration found other than GenomeClass.\n");}
    break;

  case 12:

    {
      pCURRENT_CLASS=SymbolTable.insert((yyvsp[(1) - (1)].pSymbol));  
      pCURRENT_CLASS->pSymbolList=new CLList<CSymbol *>();
      (yyvsp[(1) - (1)].pSymbol)->ObjectType=oUserClass;
      //DEBUG_PRT("Yacc Symbol declaration %s %d",$1->sName,$1->nSize);
      pCLASSES[nClasses_nb++] = (yyvsp[(1) - (1)].pSymbol);
    }
    break;

  case 13:

    {
      if (bVERBOSE) printf("Class %s declared for %d bytes.\n\n",(yyvsp[(1) - (5)].pSymbol)->sName,(yyvsp[(1) - (5)].pSymbol)->nSize);
      //DEBUG_PRT("Yacc variable declaration %s %d",$1->sName,$1->nSize);
    }
    break;

  case 16:

    {pCURRENT_TYPE=(yyvsp[(2) - (2)].pSymbol); pCURRENT_TYPE->ObjectQualifier=(yyvsp[(1) - (2)].ObjectQualifier);}
    break;

  case 17:

    {}
    break;

  case 18:

    {pCURRENT_TYPE=(yyvsp[(2) - (2)].pSymbol); pCURRENT_TYPE->ObjectQualifier=(yyvsp[(1) - (2)].ObjectQualifier);}
    break;

  case 19:

    {}
    break;

  case 21:

    {
    pCURRENT_CLASS->sString = new char[strlen((yyvsp[(2) - (2)].szString)) + 1];
    strcpy(pCURRENT_CLASS->sString, (yyvsp[(2) - (2)].szString));      
    if (bVERBOSE) printf("\n    The following methods have been declared:\n\n%s\n\n",pCURRENT_CLASS->sString);
    }
    break;

  case 22:

    {(yyval.ObjectQualifier)=1;}
    break;

  case 23:

    {(yyval.ObjectQualifier)=0;}
    break;

  case 25:

    {}
    break;

  case 26:

    {}
    break;

  case 27:

    {}
    break;

  case 35:

    {  
      CSymbol *pSym=SymbolTable.find((yyvsp[(1) - (1)].pSymbol)->sName);
      if (pSym==NULL) {
        fprintf(stderr,"\n%s - Error line %d: Class \"%s\" was not defined.\n",sEZ_FILE_NAME,yylineno,(yyvsp[(1) - (1)].pSymbol)->sName);
        fprintf(stderr,"Only base types (bool, int, float, double, char) or new user classes defined\nwithin the \"User classes\" sections are allowed.\n");
        exit(1);
      }       
      else (yyval.pSymbol)=pSym;
    }
    break;

  case 38:

    {
//      CSymbol *pSym;
//      pSym=$1;
        (yyvsp[(1) - (1)].pSymbol)->nSize=pCURRENT_TYPE->nSize;
        (yyvsp[(1) - (1)].pSymbol)->pClass=pCURRENT_CLASS;
        (yyvsp[(1) - (1)].pSymbol)->pType=pCURRENT_TYPE;
        (yyvsp[(1) - (1)].pSymbol)->ObjectType=oObject;
        (yyvsp[(1) - (1)].pSymbol)->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
        pCURRENT_CLASS->nSize+=(yyvsp[(1) - (1)].pSymbol)->nSize;
        pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)((yyvsp[(1) - (1)].pSymbol)));
        if (bVERBOSE) printf("    %s variable declared (%d bytes)\n",(yyvsp[(1) - (1)].pSymbol)->sName,(yyvsp[(1) - (1)].pSymbol)->nSize);
    }
    break;

  case 39:

    {
      (yyvsp[(2) - (2)].pSymbol)->nSize=sizeof (char *);
      (yyvsp[(2) - (2)].pSymbol)->pClass=pCURRENT_CLASS;
      (yyvsp[(2) - (2)].pSymbol)->pType=pCURRENT_TYPE;
      (yyvsp[(2) - (2)].pSymbol)->ObjectType=oPointer;
      (yyvsp[(2) - (2)].pSymbol)->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=(yyvsp[(2) - (2)].pSymbol)->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)((yyvsp[(2) - (2)].pSymbol)));
      if (bVERBOSE) printf("    %s pointer declared (%d bytes)\n",(yyvsp[(2) - (2)].pSymbol)->sName,(yyvsp[(2) - (2)].pSymbol)->nSize);
    }
    break;

  case 40:

    {
      (yyvsp[(2) - (2)].pSymbol)->nSize=sizeof (char *);
      (yyvsp[(2) - (2)].pSymbol)->pClass=pCURRENT_CLASS;
      (yyvsp[(2) - (2)].pSymbol)->pType=pCURRENT_TYPE;
      (yyvsp[(2) - (2)].pSymbol)->ObjectType=oPointer;
      (yyvsp[(2) - (2)].pSymbol)->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=(yyvsp[(2) - (2)].pSymbol)->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)((yyvsp[(2) - (2)].pSymbol)));
      if (bVERBOSE) printf("    %s NULL pointer declared (%d bytes)\n",(yyvsp[(2) - (2)].pSymbol)->sName,(yyvsp[(2) - (2)].pSymbol)->nSize);
    }
    break;

  case 41:

    {
      (yyvsp[(3) - (3)].pSymbol)->nSize=sizeof (char *);
      (yyvsp[(3) - (3)].pSymbol)->pClass=pCURRENT_CLASS;
      (yyvsp[(3) - (3)].pSymbol)->pType=pCURRENT_TYPE;
      (yyvsp[(3) - (3)].pSymbol)->ObjectType=oPointer;
      (yyvsp[(3) - (3)].pSymbol)->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=(yyvsp[(3) - (3)].pSymbol)->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)((yyvsp[(3) - (3)].pSymbol)));
      if (bVERBOSE) printf("    %s pointer of pointer declared (%d bytes)\n",(yyvsp[(3) - (3)].pSymbol)->sName,(yyvsp[(3) - (3)].pSymbol)->nSize);
      fprintf(stderr,"Pointer of pointer doesn't work properly yet\n");
      exit(-1);
    }
    break;

  case 42:

    {
      if((TARGET_FLAVOR==CMAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName,"Genome")==0) { nGENOME_NAME=(yyvsp[(1) - (4)].pSymbol)->sName; nPROBLEM_DIM=(int)(yyvsp[(3) - (4)].dValue);}

      //printf("DEBUG : size of $3 %d nSize %d\n",(int)$3,pCURRENT_TYPE->nSize);

      (yyvsp[(1) - (4)].pSymbol)->nSize=pCURRENT_TYPE->nSize*(int)(yyvsp[(3) - (4)].dValue);
      (yyvsp[(1) - (4)].pSymbol)->pClass=pCURRENT_CLASS;
      (yyvsp[(1) - (4)].pSymbol)->pType=pCURRENT_TYPE;
      (yyvsp[(1) - (4)].pSymbol)->ObjectType=oArray;
      (yyvsp[(1) - (4)].pSymbol)->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=(yyvsp[(1) - (4)].pSymbol)->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)((yyvsp[(1) - (4)].pSymbol)));
      if (bVERBOSE) printf("    %s array declared (%d bytes)\n",(yyvsp[(1) - (4)].pSymbol)->sName,(yyvsp[(1) - (4)].pSymbol)->nSize);
    }
    break;

  case 43:

    {

    // this is for support of pointer array. This should be done in a more generic way in a later version
      if((TARGET_FLAVOR==CMAES) && nPROBLEM_DIM==0 && strcmp(pCURRENT_CLASS->sName,"Genome")==0) { 
	nGENOME_NAME=(yyvsp[(2) - (5)].pSymbol)->sName; nPROBLEM_DIM=(int)(yyvsp[(4) - (5)].dValue);
      }
      
      //pCURRENT_CLASS->nSize

      (yyvsp[(2) - (5)].pSymbol)->nSize=sizeof(char*)*(int)(yyvsp[(4) - (5)].dValue);
      (yyvsp[(2) - (5)].pSymbol)->pClass=pCURRENT_CLASS;
      (yyvsp[(2) - (5)].pSymbol)->pType=pCURRENT_TYPE;
      (yyvsp[(2) - (5)].pSymbol)->ObjectType=oArrayPointer;
      (yyvsp[(2) - (5)].pSymbol)->ObjectQualifier=pCURRENT_TYPE->ObjectQualifier;
      pCURRENT_CLASS->nSize+=(yyvsp[(2) - (5)].pSymbol)->nSize;
      pCURRENT_CLASS->pSymbolList->addFirst((CSymbol *)((yyvsp[(2) - (5)].pSymbol)));

      printf("DEBUG : size of $4 %d nSize %d\n",(int)(yyvsp[(4) - (5)].dValue),pCURRENT_TYPE->nSize);
      if (bVERBOSE) printf("    %s array of pointers declared (%d bytes)\n",(yyvsp[(2) - (5)].pSymbol)->sName,(yyvsp[(2) - (5)].pSymbol)->nSize);
    }
    break;

  case 46:

    {}
    break;

  case 47:

    {
    ////DEBUG_PRT("Yacc genome decl %s",$1.pSymbol->sName);
      if (bVERBOSE) printf ("\nGenome declaration analysis :\n\n");
      pGENOME=new CSymbol("Genome");
      pCURRENT_CLASS=SymbolTable.insert(pGENOME);  
      pGENOME->pSymbolList=new CLList<CSymbol *>();
      pGENOME->ObjectType=oUserClass;
      pGENOME->ObjectQualifier=0;
      pGENOME->sString=NULL;
    }
    break;

  case 48:

    {}
    break;

  case 51:

    {}
    break;

  case 52:

    {(yyval.pSymbol)=(yyvsp[(1) - (1)].pSymbol);}
    break;

  case 55:

    {         
      if (bVERBOSE) printf("Inserting genome initialiser (taken from .ez file).\n");
    }
    break;

  case 56:

    {}
    break;

  case 57:

    {
      if (bVERBOSE) printf("Inserting user genome crossover (taken from .ez file).\n");
    }
    break;

  case 58:

    {}
    break;

  case 59:

    {
      if (bVERBOSE) printf("Inserting user genome mutator (taken from .ez file).\n");
    }
    break;

  case 60:

    {}
    break;

  case 61:

    { 
      if (bVERBOSE) printf("Inserting user genome evaluator (taken from .ez file).\n");
    }
    break;

  case 62:

    {}
    break;

  case 63:

    { 
      if (bVERBOSE) printf("Inserting user genome optimiser (taken from .ez file).\n");
    }
    break;

  case 64:

    {}
    break;

  case 65:

    {
     //DEBUG_PRT("User makefile options have been reduced");
     }
    break;

  case 66:

    {}
    break;

  case 69:

    {nNB_GEN=(int)(yyvsp[(2) - (2)].dValue);}
    break;

  case 70:

    {nNB_OPT_IT=(int)(yyvsp[(2) - (2)].dValue);}
    break;

  case 71:

    {nTIME_LIMIT=(int)(yyvsp[(2) - (2)].dValue);}
    break;

  case 72:

    {fMUT_PROB=(float)(yyvsp[(2) - (2)].dValue);}
    break;

  case 73:

    {fXOVER_PROB=(float)(yyvsp[(2) - (2)].dValue);}
    break;

  case 74:

    {nPOP_SIZE=(int)(yyvsp[(2) - (2)].dValue);}
    break;

  case 75:

    {
      strcpy(sSELECTOR, (yyvsp[(2) - (2)].pSymbol)->sName);
      strcpy(sSELECTOR_OPERATOR, (yyvsp[(2) - (2)].pSymbol)->sName);
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelector(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME);
	break;
      }
    }
    break;

  case 76:

    {
      sprintf(sSELECTOR, (yyvsp[(2) - (3)].pSymbol)->sName);   
      sprintf(sSELECTOR_OPERATOR, (yyvsp[(2) - (3)].pSymbol)->sName);   
      switch (TARGET) {
      case CUDA:
      case STD:
	pickupSTDSelectorArgument(sSELECTOR,&fSELECT_PRM,sEZ_FILE_NAME,(float)(yyvsp[(3) - (3)].dValue));
	break;
      }
    }
    break;

  case 77:

    {
        sprintf(sRED_PAR, (yyvsp[(2) - (2)].pSymbol)->sName);
	sprintf(sRED_PAR_OPERATOR, (yyvsp[(2) - (2)].pSymbol)->sName);
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME);
	  break;
	}
    }
    break;

  case 78:

    {
        sprintf(sRED_PAR, (yyvsp[(2) - (3)].pSymbol)->sName);
	sprintf(sRED_PAR_OPERATOR, (yyvsp[(2) - (3)].pSymbol)->sName);
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_PAR,&fRED_PAR_PRM,sEZ_FILE_NAME,(float)(yyvsp[(3) - (3)].dValue));
	  break;
	}
    }
    break;

  case 79:

    {
	sprintf(sRED_OFF, (yyvsp[(2) - (2)].pSymbol)->sName);
	sprintf(sRED_OFF_OPERATOR, (yyvsp[(2) - (2)].pSymbol)->sName);
      switch (TARGET) {
      case STD:
      case CUDA:
	pickupSTDSelector(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME);
	break;
      }
    }
    break;

  case 80:

    {
        sprintf(sRED_OFF, (yyvsp[(2) - (3)].pSymbol)->sName);
	sprintf(sRED_OFF_OPERATOR, (yyvsp[(2) - (3)].pSymbol)->sName);
        switch (TARGET) {
	case STD:
	case CUDA:
	  pickupSTDSelectorArgument(sRED_OFF,&fRED_OFF_PRM,sEZ_FILE_NAME,(yyvsp[(3) - (3)].dValue));
       }}
    break;

  case 81:

    {
        sprintf(sRED_FINAL, (yyvsp[(2) - (2)].pSymbol)->sName);
        sprintf(sRED_FINAL_OPERATOR, (yyvsp[(2) - (2)].pSymbol)->sName);
        switch (TARGET) {
	case CUDA:
	case STD:
	  pickupSTDSelector(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME);
	  break;
       }}
    break;

  case 82:

    {
        sprintf(sRED_FINAL, (yyvsp[(2) - (3)].pSymbol)->sName);
        sprintf(sRED_FINAL_OPERATOR, (yyvsp[(2) - (3)].pSymbol)->sName);
        switch (TARGET) {
	case CUDA :
	case STD:
	  pickupSTDSelectorArgument(sRED_FINAL,&fRED_FINAL_PRM,sEZ_FILE_NAME,(yyvsp[(3) - (3)].dValue));
	  break;
	}}
    break;

  case 83:

    {nOFF_SIZE=(int)(yyvsp[(2) - (2)].dValue);}
    break;

  case 84:

    {nOFF_SIZE=(int)((yyvsp[(2) - (3)].dValue)*nPOP_SIZE/100);}
    break;

  case 85:

    {fSURV_PAR_SIZE=(float)(yyvsp[(2) - (2)].dValue);}
    break;

  case 86:

    {fSURV_PAR_SIZE=(float)((yyvsp[(2) - (3)].dValue)/100);}
    break;

  case 87:

    {fSURV_OFF_SIZE=(float)(yyvsp[(2) - (2)].dValue);}
    break;

  case 88:

    {fSURV_OFF_SIZE=(float)((yyvsp[(2) - (3)].dValue)/100);}
    break;

  case 89:

    {
      if ((!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"Maximise")) || (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"Maximize"))) nMINIMISE=0;
      else if ((!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"Minimise")) || (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"Minimize"))) nMINIMISE=1;
      else {
         fprintf(stderr,"\n%s - Error line %d: The evaluator goal default parameter can only take\n",sEZ_FILE_NAME,yylineno);
         fprintf(stderr,"two values : maximi[sz]e or minimi[sz]e.\n",sEZ_FILE_NAME,yylineno);
         exit(1);
       }
      }
    break;

  case 90:

    {
        nELITE=(int)(yyvsp[(2) - (2)].dValue);
        }
    break;

  case 91:

    {
        nELITE=(int)(yyvsp[(2) - (3)].dValue)*nPOP_SIZE/100;
        }
    break;

  case 92:

    {
      if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"Weak")) bELITISM=0;
      else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"Strong")) bELITISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Elitism must be \"Strong\" or \"Weak\".\nDefault value \"Strong\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bELITISM=1;
       }}
    break;

  case 93:

    {
      if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"False")) bBALDWINISM=0;
      else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"True")) bBALDWINISM=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Baldwinism must be \"True\" or \"False\".\nDefault value \"True\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bBALDWINISM=1;
       }}
    break;

  case 94:

    {
	if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"False")) bREMOTE_ISLAND_MODEL=0;
	else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"True")) bREMOTE_ISLAND_MODEL=1;
	else {
	  fprintf(stderr,"\n%s - Warning line %d: remote island model must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n",sEZ_FILE_NAME,yylineno);nWARNINGS++;
	  bREMOTE_ISLAND_MODEL=0;
	}}
    break;

  case 95:

    {
        sprintf(sIP_FILE, (yyvsp[(2) - (4)].pSymbol)->sName);
	strcat(sIP_FILE,".");
	strcat(sIP_FILE,(yyvsp[(4) - (4)].pSymbol)->sName);
	}
    break;

  case 96:

    {
	fMIGRATION_PROBABILITY=(float)(yyvsp[(2) - (2)].dValue);
	}
    break;

  case 97:

    {
      nSERVER_PORT=(int)(yyvsp[(2) - (2)].dValue);
    }
    break;

  case 98:

    {
      if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"False")) bPRINT_STATS=0;
      else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"True")) bPRINT_STATS=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Print stats must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bPRINT_STATS=0;
       }}
    break;

  case 99:

    {
      if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"False")) bPLOT_STATS=0;
      else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"True")) bPLOT_STATS=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate stats must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bPLOT_STATS=0;
       }}
    break;

  case 100:

    {
      if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"False")) bGENERATE_CSV_FILE=0;
      else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"True")) bGENERATE_CSV_FILE=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate csv file must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bGENERATE_CSV_FILE=0;
       }}
    break;

  case 101:

    {
      if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"False")) bGENERATE_GNUPLOT_SCRIPT=0;
      else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"True")) bGENERATE_GNUPLOT_SCRIPT=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate gnuplot script must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bGENERATE_GNUPLOT_SCRIPT=0;
       }}
    break;

  case 102:

    {
      if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"False")) bGENERATE_R_SCRIPT=0;
      else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"True")) bGENERATE_R_SCRIPT=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: Generate R script must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bGENERATE_R_SCRIPT=0;
       }}
    break;

  case 103:

    {
      if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"False")) bSAVE_POPULATION=0;
      else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"True")) bSAVE_POPULATION=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: SavePopulation must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bSAVE_POPULATION=0;
       }}
    break;

  case 104:

    {
      if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"False")) bSTART_FROM_FILE=0;
      else if (!mystricmp((yyvsp[(2) - (2)].pSymbol)->sName,"True")) bSTART_FROM_FILE=1;
      else {
         fprintf(stderr,"\n%s - Warning line %d: StartFromFile must be \"True\" or \"False\".\nDefault value \"False\" inserted.\n.",sEZ_FILE_NAME,yylineno);nWARNINGS++;
         bSTART_FROM_FILE=0;
       }}
    break;

  case 105:

    {iMAX_INIT_TREE_D = (unsigned)(yyvsp[(2) - (2)].dValue);}
    break;

  case 106:

    {iMIN_INIT_TREE_D = (unsigned)(yyvsp[(2) - (2)].dValue);}
    break;

  case 107:

    {iMAX_TREE_D = (unsigned)(yyvsp[(2) - (2)].dValue);}
    break;

  case 108:

    {iNB_GPU = (unsigned)(yyvsp[(2) - (2)].dValue);}
    break;

  case 109:

    {iPRG_BUF_SIZE = (unsigned)(yyvsp[(2) - (2)].dValue);}
    break;

  case 110:

    {iNO_FITNESS_CASES = (unsigned)(yyvsp[(2) - (2)].dValue);}
    break;

  case 111:

    { 
      if (SymbolTable.find((yyvsp[(1) - (3)].pSymbol)->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,yylineno,(yyvsp[(1) - (3)].pSymbol)->sName);
         exit(1);
      }
      (yyval.dValue) = CEASEAParser_assign(SymbolTable.find((yyvsp[(1) - (3)].pSymbol)->sName), (yyvsp[(3) - (3)].dValue));
    }
    break;

  case 112:

    { (yyval.dValue) = (yyvsp[(1) - (3)].dValue) + (yyvsp[(3) - (3)].dValue); }
    break;

  case 113:

    { (yyval.dValue) = (yyvsp[(1) - (3)].dValue) - (yyvsp[(3) - (3)].dValue); }
    break;

  case 114:

    { (yyval.dValue) = (yyvsp[(1) - (3)].dValue) * (yyvsp[(3) - (3)].dValue); }
    break;

  case 115:

    { /* CEASEAParser_divide can't be used because g++ can't goto the label YYERROR go,
                                           So I directely use its code. I don't know why g++ can't compile CEASEAParser_divide */
                                        if((yyvsp[(3) - (3)].dValue) == 0) {(yyval.dValue)=0; YYERROR; return 0;}
                                        else (yyval.dValue) = (yyvsp[(1) - (3)].dValue) / (yyvsp[(3) - (3)].dValue); }
    break;

  case 116:

    { (yyval.dValue) = (yyvsp[(2) - (3)].dValue); }
    break;

  case 117:

    { (yyval.dValue) = -(yyvsp[(2) - (2)].dValue); }
    break;

  case 118:

    { (yyval.dValue) = (yyvsp[(1) - (1)].dValue); }
    break;

  case 119:

    {
      if (SymbolTable.find((yyvsp[(1) - (1)].pSymbol)->sName)==NULL){
         fprintf(stderr,"\n%s - Error line %d: Symbol \"%s\" not found.\n",sEZ_FILE_NAME,yylineno,(yyvsp[(1) - (1)].pSymbol)->sName);
         exit(1);
      }
      (yyval.dValue) = (SymbolTable.find((yyvsp[(1) - (1)].pSymbol)->sName))->dValue;
    }
    break;



      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}




                       
/////////////////////////////////////////////////////////////////////////////
// main

int main(int argc, char *argv[]){
  int n = YYEXIT_FAILURE;
  int nParamNb=0;
  char *sTemp;
  int i=0;
  
  TARGET=STD;
  bVERBOSE=0;
  sRAW_PROJECT_NAME[0]=0; // used to ask for a filename if no filename is found on the command line.

  while ((++nParamNb) < argc) {
    sTemp=&(argv[nParamNb][0]);
    if ((argv[nParamNb][0]=='-')||(argv[nParamNb][0]=='/')) sTemp=&(argv[nParamNb][1]);
    if (!mystricmp(sTemp,"cuda")){
      TARGET=CUDA;
      TARGET_FLAVOR = CUDA_FLAVOR_SO;
    }
    else if( !mystricmp(sTemp,"cuda_mo") ){
      TARGET=CUDA;
      TARGET_FLAVOR = CUDA_FLAVOR_MO;
    }
    else if( !mystricmp(sTemp,"cuda_gp") ){
      printf("tpl is cuda gp\n");
      TARGET=CUDA;
      TARGET_FLAVOR = FLAVOR_GP;
    }
    else if( !mystricmp(sTemp,"gp") ){
      printf("tpl is gp\n");
      TARGET=STD;
      TARGET_FLAVOR = FLAVOR_GP;
    }
    else if (!mystricmp(sTemp,"std"))  {
      TARGET=STD;
      TARGET_FLAVOR = STD_FLAVOR_SO;
    }
    else if (!mystricmp(sTemp,"std_mo")) {
      TARGET=STD;
      TARGET_FLAVOR = STD_FLAVOR_MO;
    }
    else if (!mystricmp(sTemp,"cmaes"))  {
      TARGET_FLAVOR = CMAES;
    }
    else if (!mystricmp(sTemp,"memetic"))  {
      TARGET_FLAVOR = MEMETIC;
    }

    else if (!mystricmp(sTemp,"v"))  bVERBOSE=true;
    else if (!mystricmp(sTemp,"tl")){
      bLINE_NUM_EZ_FILE=false;
    }
    else if (!mystricmp(sTemp,"path"))  {
      if (argv[++nParamNb][0]=='"') {
        strcpy(sEZ_PATH,&(argv[nParamNb][1]));
        while (argv[++nParamNb][strlen(argv[nParamNb])]!='"')
          strcat(sEZ_PATH,argv[nParamNb]);
          argv[nParamNb][strlen(argv[nParamNb])]=0;
          strcat(sEZ_PATH,argv[nParamNb]);
      }
      else {
        if (argv[nParamNb][strlen(argv[nParamNb])-1]=='"') argv[nParamNb][strlen(argv[nParamNb])-1]=0;
        strcpy(sEZ_PATH,argv[nParamNb]);
      }
    }
    else if (!mystricmp(sTemp,"eo_dir"))  {
      if (argv[++nParamNb][0]=='"') {
        strcpy(sEO_DIR,&(argv[nParamNb][1]));
        while (argv[++nParamNb][strlen(argv[nParamNb])]!='"')
          strcat(sEO_DIR,argv[nParamNb]);
          argv[nParamNb][strlen(argv[nParamNb])]=0;
          strcat(sEO_DIR,argv[nParamNb]);
      }
      else {
        if (argv[nParamNb][strlen(argv[nParamNb])-1]=='"') argv[nParamNb][strlen(argv[nParamNb])-1]=0;
        strcpy(sEO_DIR,argv[nParamNb]);
      }
    }
    else strcpy(sRAW_PROJECT_NAME,argv[nParamNb]);
  }

  CEASEAParser_create();
  n = yyparse();
  exit(n);
  return n;
}

/////////////////////////////////////////////////////////////////////////////
// EASEAParser commands

int CEASEAParser_create()
{
  CSymbol *pNewBaseType;

  pNewBaseType=new CSymbol("bool");
  pNewBaseType->nSize=sizeof(bool);   
  pNewBaseType->ObjectType=oBaseClass;
  SymbolTable.insert(pNewBaseType);

  pNewBaseType=new CSymbol("int");
  pNewBaseType->nSize=sizeof(int);   
  pNewBaseType->ObjectType=oBaseClass;
  SymbolTable.insert(pNewBaseType);

  pNewBaseType=new CSymbol("double");
  pNewBaseType->nSize=sizeof(double);   
  pNewBaseType->ObjectType=oBaseClass;
  SymbolTable.insert(pNewBaseType);

  pNewBaseType=new CSymbol("float");
  pNewBaseType->nSize=sizeof(float);   
  pNewBaseType->ObjectType=oBaseClass;
  SymbolTable.insert(pNewBaseType);

  pNewBaseType=new CSymbol("char");
  pNewBaseType->nSize=sizeof(char);   
  pNewBaseType->ObjectType=oBaseClass;
  SymbolTable.insert(pNewBaseType);

  pNewBaseType=new CSymbol("pointer");
  pNewBaseType->nSize=sizeof(char *);   
  pNewBaseType->ObjectType=oBaseClass;
  SymbolTable.insert(pNewBaseType);


  //if (!yycreate(&EASEALexer)) {
  //  return 0;
  //}
  if (!CEASEALexer_create(&SymbolTable)) {
    return 0;
  }
  return 1; // success
}

/////////////////////////////////////////////////////////////////////////////
// calc_parser attribute commands

double CEASEAParser_assign(CSymbol* spIdentifier, double dValue)
{
  assert(spIdentifier != NULL);

  spIdentifier->dValue = dValue;
  return spIdentifier->dValue;
}

/*double CEASEAParser_divide(double a, double b)
{
  if (b == 0) {
    printf("division by zero\n");
    YYERROR;
    return 0;
  }
  else {
    return a / b;
  }
}
*/

void CEASEAParser_yysyntaxerror(){

  printf("Syntax Error at line : %d (on text : %s)\nFor more details during the EASEA compiling, use the \"-v\" option\n",
	 yylineno,yytext);
}


