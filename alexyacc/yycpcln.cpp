/************************************************************
yycpcln.cpp
This file can be freely modified for the generation of
custom code.

[Ansi]

Copyright (c) 1999-2001 Bumble-Bee Software Ltd.
************************************************************/

#include <stdlib.h>
#include "cyacc.h"

void yyparser::yycleanup()
{
	if (yystackptr != yysstackptr) {
		free(yystackptr);
		yystackptr = yysstackptr;
	}
	if (yyattributestackptr != yysattributestackptr) {
		free(yyattributestackptr);
		yyattributestackptr = yysattributestackptr;
	}
	yystack_size = yysstack_size;

	yytop = -1;
}
