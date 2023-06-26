#ifndef EZ_COMPILER_ERRORS_H
#define EZ_COMPILER_ERRORS_H

void yyerror(const char * str);
void print_better_error(const char* fname, const char* line, int lineno, int columnno, const char* error_string);

#endif
