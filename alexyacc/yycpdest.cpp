/************************************************************
yycpdest.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include "cyacc.h"
#include <stdlib.h>

void yyparser::yydestroy()
{
	yycleanup();
	free(yysstackptr);
	free(yysattributestackptr);

	free(yylvalptr);
	free(yyvalptr);
}
