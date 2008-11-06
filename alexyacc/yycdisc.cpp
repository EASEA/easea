/************************************************************
yycdisc.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include <stdlib.h>
#include "cyacc.h"

#ifdef __BORLANDC__
#pragma argsused
#endif

void yyparser::yydiscard(int token)
{
	yyassert(token > 0);
	// do nothing
}
