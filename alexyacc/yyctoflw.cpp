/************************************************************
yyctoflw.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include "clex.h"

void yylexer::yytextoverflow()
{
	yyassert(yyerr != NULL);
	fprintf(yyerr, "lex text buffer overflow (%d)\n", (int)yytext_size);
}
