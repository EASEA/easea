/************************************************************
yycsoflw.cpp
This file can be freely modified for the generation of
custom code.

[Ansi]

Copyright (c) 1999-2001 Bumble-Bee Software Ltd.
************************************************************/

#include <stdlib.h>
#include "cyacc.h"

void yyparser::yystackoverflow()
{
	yyerror("yacc stack overflow");
}
