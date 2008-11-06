/************************************************************
yycldest.cpp
This file can be freely modified for the generation of
custom code.

[Ansi]

Copyright (c) 1999-2001 Bumble-Bee Software Ltd.
************************************************************/

#include "clex.h"
#include <stdlib.h>

void yylexer::yydestroy()
{
	yycleanup();
	free(yystext);
	yystext = NULL;
	yytext = NULL;
	free(yysstatebuf);
	yysstatebuf = NULL;
	yystatebuf = NULL;
	free(yysunputbufptr);
	yysunputbufptr = NULL;
	yyunputbufptr = NULL;
}
