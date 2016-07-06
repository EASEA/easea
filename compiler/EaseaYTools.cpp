#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "EaseaYTools.h"

void pickupSTDSelector(char* sSELECTOR, float* fSELECT_PRM,
                       char* sEZ_FILE_NAME) {
    //DEBUG_PRT("Picking up selector without argument %s",sSELECTOR);
    if (!mystricmp(sSELECTOR,"Roulette")) {
        if (nMINIMISE==1) {
            fprintf(stderr,
                    "\n%s - Error line %d: The Roulette selection scheme cannot be\n selected when \"minimising the fitness\" is the evaluator goal.\n",
                    sEZ_FILE_NAME,yylineno);
            exit(1);
        } else sprintf(sSELECTOR,"Roulette");
    } else if (!mystricmp(sSELECTOR,"Tournament")) {
        sprintf(sSELECTOR,"Tournament");
        // as there is no selection pressure, we put 2
        *fSELECT_PRM=(float)2.0;
    } else if (!mystricmp(sSELECTOR,"StochTrn")) sprintf(sSELECTOR,"StochTour");
    else if (!mystricmp(sSELECTOR,"Random")) {
        sprintf(sSELECTOR,"Random");
        if( TARGET==CUDA || TARGET==STD || TARGET==NSGA2 )
            *fSELECT_PRM = 0.0;
    } else if (!mystricmp(sSELECTOR,"Ranking")) sprintf(sSELECTOR,"Ranking");
    else if (!mystricmp(sSELECTOR,"Deterministic")) {
        sprintf(sSELECTOR,"Deterministic");
        if( TARGET==CUDA || TARGET==STD || TARGET==NSGA2) *fSELECT_PRM = 0.0;
    } else if (!mystricmp(sSELECTOR, "Identity")) {
        sprintf(sSELECTOR,"Identity");
    } else {
        fprintf(stderr,
                "\n%s - Error line %d: The %s selection scheme does not exist in CUDA/STD.\n",
                sEZ_FILE_NAME,yylineno, sSELECTOR);
        exit(1);
    }
}


void pickupSTDSelectorArgument(char* sSELECTOR, float* fSELECT_PRM,
                               char* sEZ_FILE_NAME, float thirdParam) {
    //DEBUG_PRT("Picking up selector with argument %s %d",sSELECTOR,(int) thirdParam);
    if (!mystricmp(sSELECTOR,"Tournament")||!mystricmp(sSELECTOR,"StochTrn")) {
        if (thirdParam>=2) {
            sprintf(sSELECTOR,"Tournament");
            *fSELECT_PRM = (float) thirdParam;
        } else if ((thirdParam>.5)&&(thirdParam<=1.0)) {
            sprintf(sSELECTOR,"StochTour");
            *fSELECT_PRM = (float) thirdParam;
        } else {
            fprintf(stderr,
                    "\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",
                    sEZ_FILE_NAME,yylineno);
            exit(1);
        }
    } else if (!mystricmp(sSELECTOR,"Roulette")) {
        sprintf(sSELECTOR,"Roulette");
        if (thirdParam<1) {
            fprintf(stderr,
                    "\n%s - Warning line %d: The parameter of Roulette must be greater than one.\nThe parameter will therefore be ignored.",
                    sEZ_FILE_NAME,yylineno);
            nWARNINGS++;
        } else *fSELECT_PRM = (float) thirdParam;
    } else if (!mystricmp(sSELECTOR,"Random")) {
        sprintf(sSELECTOR,"Random");
        fprintf(stderr,
                "\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in CUDA/STD.\nThe parameter will therefore be ignored.",
                sEZ_FILE_NAME,yylineno);
        nWARNINGS++;
    }  else if (!mystricmp(sSELECTOR,"Identity")) {
        sprintf(sSELECTOR,"Identity");
        fprintf(stderr,
                "\n%s - Warning line %d: The Identity selector does not take any parameter in CUDA/STD.\nThe parameter will therefore be ignored.",
                sEZ_FILE_NAME,yylineno);
        nWARNINGS++;
    } else if (!mystricmp(sSELECTOR,"Ranking")) {
        sprintf(sSELECTOR,"Ranking");
        if ((thirdParam<=1)||(thirdParam>2)) {
            fprintf(stderr,
                    "\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",
                    sEZ_FILE_NAME,yylineno);
            nWARNINGS++;
            *fSELECT_PRM = 2.0;
        } else *fSELECT_PRM = (float) thirdParam;
    } else if (!mystricmp(sSELECTOR,"Deterministic")) {
        sprintf(sSELECTOR,"Deterministic");

        if (thirdParam==0)
            if( TARGET==CUDA || TARGET==STD || TARGET==NSGA2 )
                *fSELECT_PRM = 0.0;
    } else {
        fprintf(stderr,
                "\n%s - Error line %d: The %s selection scheme does not exist in CUDA/STD.\n",
                sEZ_FILE_NAME,yylineno, sSELECTOR);
        exit(1);
    }
}
