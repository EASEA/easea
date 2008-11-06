/************************************************************
yycsetin.cpp
This file can be freely modified for the generation of
custom code.

[Ansi]

Copyright (c) 1999-2001 Bumble-Bee Software Ltd.
************************************************************/

#include "cyacc.h"

void yyparser::yysetin(int token)
{
	if (token < 0) {
		token = 0;
	}
	yychar = token;
	yylookahead = 1;
}
