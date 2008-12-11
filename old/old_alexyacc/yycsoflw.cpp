/************************************************************
yycsoflw.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include <stdlib.h>
#include "cyacc.h"

void yyparser::yystackoverflow()
{
	yyerror("yacc stack overflow");
}
