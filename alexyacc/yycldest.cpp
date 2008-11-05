/************************************************************
yycldest.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include "clex.h"
#include <stdlib.h>

void yylexer::yydestroy()
{
	yycleanup();
	free(yystext);
	free(yysstatebuf);
	free(yysunputbufptr);
}
