/************************************************************
yycsyner.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include "cyacc.h"

void yyparser::yysyntaxerror(void)
{
	yyerror("syntax error");
}
