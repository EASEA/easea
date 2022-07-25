%option noyywrap
%option prefix="nz"

%{
#include "config.h"
#ifdef OS_WINDOWS
	#define YY_NO_UNISTD_H
	#include <io.h>
	using ssize_t = signed long long;
#endif

#include "NeuralParse.hpp"
#include <stdio.h>

// To fix illogical issues from Flex
#define YY_DO_BEFORE_ACTION \
	(yytext_ptr) = yy_bp; \
	yyleng = (int) (yy_cp - yy_bp); \
	(yy_hold_char) = *yy_cp; \
	*yy_cp = '\0'; \
	(yy_c_buf_p) = yy_cp;
#define YY_NEW_FILE yyrestart( yyin )

%}

%%

[ \t]	; // ignore all whitespace
[0-9]+\.[0-9]+ 	{nzlval.fval = atof(nztext); return T_FLOAT;}
[0-9]+		{nzlval.ival = atoi(nztext); return T_INT;}
\n		{return T_NEWLINE;}
"+"		{return T_PLUS;}
"-"		{return T_MINUS;}
"*"		{return T_MULTIPLY;}
"/"		{return T_DIVIDE;}
"("		{return T_LEFT;}
")"		{return T_RIGHT;}
"exit"		{return T_QUIT;}
"quit"		{return T_QUIT;}

%%

