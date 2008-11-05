/************************************************************
yycparse.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include "cyacc.h"

int yyparser::yyparse()
{
	int n = yysetup();
	if (n != 0) {
		return n;
	}
	return yywork();
}
