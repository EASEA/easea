/************************************************************
yycgetch.cpp
This file can be freely modified for the generation of
custom code.

[Ansi]

Copyright (c) 1999-2001 Bumble-Bee Software Ltd.
************************************************************/

#include <stdio.h>
#include "clex.h"

int yylexer::yygetchar()
{
	yyassert(yyin != NULL);
	return getc(yyin);
}
