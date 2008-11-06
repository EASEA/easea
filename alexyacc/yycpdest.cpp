/************************************************************
yycpdest.cpp
This file can be freely modified for the generation of
custom code.

[Ansi]

Copyright (c) 1999-2001 Bumble-Bee Software Ltd.
************************************************************/

#include "cyacc.h"
#include <stdlib.h>

void yyparser::yydestroy()
{
	yycleanup();
	free(yysstackptr);
	yysstackptr = NULL;
	yystackptr = NULL;
	free(yysattributestackptr);
	yysattributestackptr = NULL;
	yyattributestackptr = NULL;

	free(yylvalptr);
	yylvalptr = NULL;
	free(yyvalptr);
	yyvalptr = NULL;
}
