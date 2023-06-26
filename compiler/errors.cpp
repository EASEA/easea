#include "errors.h"
#include "EaseaLex.h"
#include "Easea.h"
#include "EaseaLex.h"

extern unsigned iCOPY_GP_EVAL_STATUS;
extern FILE* yyin;
extern int lineCounter;
extern int yylineno;
extern int column;
extern char* lineptr;

void yyerror(const char * str) {
  const char* fname = NULL;
  int rline = 0;
  if (yyin == fpTemplateFile) {
  	fname = sTPL_FILE_NAME;
	rline = yylineno;
  } else if (yyin == fpGenomeFile) {
  	fname = sEZ_FILE_NAME;
	rline = lineCounter;
  } else {
  	fprintf(stderr, "Unknown yyin, dev need to fix this.\n");
	exit(1);
  }
  	
  fprintf(stderr, "\n%s:%d:%d : error: %s\n", fname, rline, column, str);
  fprintf(stderr, "%s", lineptr);
  for (int i = 0; i < column-1; ++i)
  	fputc('~', stderr);
  fprintf(stderr, "^\n");

  if (yyin == fpGenomeFile) {
  	print_better_error(fname, lineptr, rline, column, str);
  }

  fprintf(stderr, "\n=> For more details during compilation, use the \"-v\" option\n");
}

void print_better_error(const char* fname, const char* line, int lineno, int columnno, const char* error_string)
{
	(void)(fname);
	(void)(line);
	(void)(lineno);
	(void)(columnno);
	(void)(error_string);
	switch (get_lexer_state()) {
	case SGENOME_ANALYSIS:
		printf("=> error occured while parsing classes in section \"\\User classes:\"\n");
		printf("** One of these classes may be ill formed or empty.");
		break;
	case USER_CMAKE:
		printf("=> error occured while parsing section starting with \"\\User CMake:\"\n");
		break;
	case USER_DECLARATIONS:
		printf("=> error occured while parsing section starting with \"\\User declarations:\"");
		break;
	case USER_CUDA:
		printf("=> error occured while parsing section starting with \"User CUDA:\"");
		break;
	case INITIALISATION_FUNCTION:
		printf("=> error occured while parsing section starting with \"Before anything else function:\"");
		break;
	case GENERATION_FUNCTION_BEFORE_REPLACEMENT:
		printf("=> error occured while parsing section starting with \"\\At each generation before reduce function:\"");
		break;
	case BEG_GENERATION_FUNCTION:
		printf("=> error occured while parsing section starting with \"At the end of each generation function:\"");
		break;
	case GP_OPCODE:
		printf("=> error occured while parsing section starting with \"\\Begin operator description :\"");
		break;
	case GP_EVAL:
		printf("=> error occured while parsing section starting with \"\\GenomeClass::evaluator");
		if (iCOPY_GP_EVAL_STATUS == EVAL_HDR) {
			printf(" header:\"");
		} else if (iCOPY_GP_EVAL_STATUS == EVAL_BDY) {
			printf(" for each:\"");
		} else if (iCOPY_GP_EVAL_STATUS == EVAL_FTR) {
			printf(" accumulator:\"");
		}
		break;
	case INSTEAD_EVAL:
		printf("=> error occured while parsing section starting with \"\\Instead evaluation function:\"");
		break;
	case END_GENERATION_FUNCTION:
		printf("=> error occured while parsing section starting with \"At the end of each generation function:\"");
		break;
	case BOUND_CHECKING_FUNCTION:
		printf("=> error occured while parsing section starting with \"\\Bound checking:\"");
		break;
	case SANALYSE_USER_CLASSES:
		printf("=> error occured while parsing section starting with \"\\User classes:\"");
		break;
	case DISPLAY:
		printf("=> error occured while parsing section starting with \"\\GenomeClass::display:\"");
		break;
	case SMAKEFILE_OPTION:
		printf("=> error occured while parsing section starting with \"\\User Makefile options:\"");
		break;
	case USER_FUNCTIONS:
		printf("=> error occured while parsing section starting with \"\\User functions:\"");
		break;
	case EO_INITIALISER:
		printf("=> error occured while parsing section starting with \"\\GenomeClass::initializer:\"");
		break;
	case INIALISER:
		printf("=> error occured while parsing section starting with \"GenomeClass::initializer:\"");
		break;
	case FINALIZATION_FUNCTION:
		printf("=> error occured while parsing section starting with \"\\After everything else function:\"");
		break;
	case CROSSOVER:
		printf("=> error occured while parsing section starting with \"\\GenomeClass::crossover:\"");
		break;
	case MUTATOR:
		printf("=> error occured while parsing section starting with \"\\GenomeClass::mutator:\"");
		break;
	case EVALUATOR:
		printf("=> error occured while parsing section starting with \"\\GenomeClass::evaluator:\"");
		break;
	case OPTIMISER:
		printf("=> error occured while parsing section starting with \"\\GenomeClass::optimiser:\"");
		break;
	default:
		return;
	}
	printf("\n");
	return;
}
