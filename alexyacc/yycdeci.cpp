/************************************************************
yycdeci.cpp
This file can be freely modified for the generation of
custom code.

Copyright (c) 1997-99 P. D. Stearns
************************************************************/

#include <string.h>
#include "cyacc.h"

// the Visual C++ v1.52 compiler generates an error in NT if this isn't present!
#ifdef _MSC_VER
#if defined(M_I86HM) && defined(NDEBUG)
#pragma function(memcpy)
#endif
#endif

void yyfparser::yydestructclearin()
{
	if (yylookahead) {
		// clean up any token attributes
		if (yytokendestptr != NULL) {
			if (yychar >= 0 && yychar < yytokendest_size) {
				int action = yytokendestptr[yychar];
				if (action != -1) {
					// user actions in here
					memcpy(yyvalptr, yylvalptr, yyattribute_size);

					yyaction(action);

					memcpy(yylvalptr, yyvalptr, yyattribute_size);
				}
			}
		}
		yylookahead = 0;
	}
}
