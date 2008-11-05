/************************************************************
yycgtok.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include "cyacc.h"
#include "clex.h"

int yyparser::yygettoken()
{
	yyassert(yylexerptr != NULL);
	return yylexerptr->yylex();
}
