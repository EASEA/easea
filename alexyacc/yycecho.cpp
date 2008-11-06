/************************************************************
yycecho.cpp
This file can be freely modified for the generation of
custom code.

[Ansi]

Copyright (c) 1999-2001 Bumble-Bee Software Ltd.
************************************************************/

#include <stdio.h>
#include "clex.h"

void yylexer::yyecho()
{
	for (int i = 0; i < yyleng; i++) {
		yyoutput(yytext[i]);
	}
}
