/************************************************************
yycuoflw.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include "clex.h"

void yylexer::yyunputoverflow()
{
	yyassert(yyerr != NULL);
	fprintf(yyerr, "lex unput buffer overflow (%d)\n", (int)yyunput_size);
}
