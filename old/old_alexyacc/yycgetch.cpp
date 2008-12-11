/************************************************************
yycgetch.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include <stdio.h>
#include "clex.h"

int yylexer::yygetchar()
{
	yyassert(yyin != NULL);
	return getc(yyin);
}
