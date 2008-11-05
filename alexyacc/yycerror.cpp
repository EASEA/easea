/************************************************************
yycerror.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include <stdio.h>
#include "cyacc.h"

void yyparser::yyerror(const char YYFAR* text)
{
	yyassert(text != NULL);
	yyassert(yyerr != NULL);
	while (*text != '\0') {
		putc(*text++, yyerr);
	}
	putc('\n', yyerr);
}
