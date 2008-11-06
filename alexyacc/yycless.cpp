/************************************************************
yycless.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include "clex.h"

void yylexer::yyless(int length)
{
	yyassert(length >= 0 && length <= yyleng);
	while (yyleng > length) {
		yyunput((unsigned char)yytext[--yyleng]);
	}
	if (yyleng > 0) {
		yyeol = (unsigned char)(yytext[yyleng - 1] == '\n');
	}
	else {
		yyeol = yyoldeol;
	}
}
