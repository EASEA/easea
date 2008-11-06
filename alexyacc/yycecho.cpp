/************************************************************
yycecho.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include <stdio.h>
#include "clex.h"

void yylexer::yyecho()
{
	for (int i = 0; i < yyleng; i++) {
		yyoutput(yytext[i]);
	}
}
