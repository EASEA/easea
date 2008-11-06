/************************************************************
yycgtok.cpp
This file can be freely modified for the generation of
custom code.

[Ansi]

Copyright (c) 1999-2001 Bumble-Bee Software Ltd.
************************************************************/

#include "cyacc.h"
#include "clex.h"

int yyparser::yygettoken()
{
	yyassert(yylexerptr != NULL);
	return yylexerptr->yylex();
}
