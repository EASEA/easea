/************************************************************
yyctoflw.cpp
This file can be freely modified for the generation of
custom code.

[Ansi]

Copyright (c) 1999-2001 Bumble-Bee Software Ltd.
************************************************************/

#include "clex.h"

void yylexer::yytextoverflow()
{
	yyassert(yyerr != NULL);
	fprintf(yyerr, "lex text buffer overflow (%d)\n", (int)yytext_size);
}
