#include "Easea.h"
//#include "EaseaLex.h"
#include "debug.h"

extern int yylineno;

void pickupSTDSelector(char* sSELECTOR, float* fSELECT_PRM,
                       char* sEZ_FILE_NAME);
void pickupSTDSelectorArgument(char* sSELECTOR, float* fSELECT_PRM,
                               char* sEZ_FILE_NAME, float thirdParam);
