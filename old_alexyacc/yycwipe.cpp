/************************************************************
yycwipe.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include <string.h>
#include "cyacc.h"

void yyparser::yywipe()
{
	yydestructpop(yytop + 1);
	yydestructclearin();
}
