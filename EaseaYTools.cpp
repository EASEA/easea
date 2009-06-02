#include "Easea.h"
#include "EaseaLex.h"
#include "debug.h"


void pickupSTDSelector(char* sSELECTOR, char* sSELECT_PRM, char* sEZ_FILE_NAME, CEASEALexer* EASEALexer){
  DEBUG_PRT("Picking up selector without argument %s",sSELECTOR);
  if (!mystricmp(sSELECTOR,"RouletteWheel")){
    if (nMINIMISE==1) {
      fprintf(stderr,"\n%s - Error line %d: The RouletteWheel selection scheme cannot be\n selected when \"minimising the fitness\" is the evaluator goal.\n",
	      sEZ_FILE_NAME,EASEALexer->yylineno);
      exit(1);
    }
    else sprintf(sSELECTOR,"Roulette");
  }
  else if (!mystricmp(sSELECTOR,"Tournament")){
    sprintf(sSELECTOR,"DetTour");
    // as there is no selection pressure, we put 2
    sprintf(sSELECT_PRM,"(2)");
  }
  else if (!mystricmp(sSELECTOR,"StochTrn")) sprintf(sSELECTOR,"StochTour");
  else if (!mystricmp(sSELECTOR,"Random")){
    sprintf(sSELECTOR,"Random");
    if( TARGET==CUDA || TARGET==STD )
      sprintf(sSELECT_PRM,"(0)");      
  }
  else if (!mystricmp(sSELECTOR,"Ranking")) sprintf(sSELECTOR,"Ranking");
  else if (!mystricmp(sSELECTOR,"Sequential")){
    sprintf(sSELECTOR,"Sequential");
    if( TARGET==CUDA || TARGET==STD) sprintf(sSELECT_PRM,"(0)");
    
  }
  else {
    fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist in CUDA/STD.\n",sEZ_FILE_NAME,EASEALexer->yylineno, sSELECTOR);
    exit(1);
  }
}


void pickupSTDSelectorArgument(char* sSELECTOR, char* sSELECTOR_PRM, char* sEZ_FILE_NAME, float thirdParam, CEASEALexer* EASEALexer){
  DEBUG_PRT("Picking up selector with argument %s %d",sSELECTOR,(int) thirdParam);
  if (!mystricmp(sSELECTOR,"Tournament")||!mystricmp(sSELECTOR,"StochTrn")) {
    if (thirdParam>=2) {sprintf(sSELECTOR,"DetTour");
      sprintf(sSELECTOR_PRM,"(%d)",(int) thirdParam);}
    else if ((thirdParam>.5)&&(thirdParam<=1.0)) {
      sprintf(sSELECTOR,"StochTour");
      sprintf(sSELECTOR_PRM,"(%f)",(float) thirdParam);
    }
    else {
      fprintf(stderr,"\n%s - Error line %d: The parameter of the Tournament selector must be either >=2 or within ]0.5, 1].\n",
	      sEZ_FILE_NAME,EASEALexer->yylineno); 
      exit(1);
    }
  }
  else if (!mystricmp(sSELECTOR,"RouletteWheel")) {
    sprintf(sSELECTOR,"Roulette");
    if (thirdParam<1) {fprintf(stderr,"\n%s - Warning line %d: The parameter of RouletteWheel must be greater than one.\nThe parameter will therefore be ignored.",
		       sEZ_FILE_NAME,EASEALexer->yylineno);
      nWARNINGS++;
    }
    else sprintf(sSELECTOR_PRM,"(%f)",(float) thirdParam);
  }
  else if (!mystricmp(sSELECTOR,"Random")) {
    sprintf(sSELECTOR,"Random");
    fprintf(stderr,"\n%s - Warning line %d: The Uniform selector does not (yet) take any parameter in CUDA/STD.\nThe parameter will therefore be ignored.",
	    sEZ_FILE_NAME,EASEALexer->yylineno);
    nWARNINGS++;
  }
  else if (!mystricmp(sSELECTOR,"Ranking")) {
    sprintf(sSELECTOR,"Ranking");
    if ((thirdParam<=1)||(thirdParam>2)) {
      fprintf(stderr,"\n%s - Warning line %d: The parameter of Ranking must be in (1,2].\nThe parameter will default to 2.",sEZ_FILE_NAME,EASEALexer->yylineno);
      nWARNINGS++;
      sprintf(sSELECTOR_PRM,"(2)");
    }
    else sprintf(sSELECTOR_PRM,"(%f)",(float) thirdParam);
  }
  else if (!mystricmp(sSELECTOR,"Sequential")) {
    sprintf(sSELECTOR,"Sequential");
			    
    if (thirdParam==0)
      if( TARGET==CUDA || TARGET==STD )
	sprintf(sSELECT_PRM,"(0)");
      else
	sprintf(sSELECT_PRM,"(unordered)");
    else if (thirdParam==1) sprintf(sSELECTOR_PRM,"(ordered)");
    else {
      fprintf(stderr,"\n%s - Warning line %d: The parameter of Sequential must be either 0 (unordered) or 1 (ordered).\nThe parameter will default to 1.",
	      sEZ_FILE_NAME,EASEALexer->yylineno);
      nWARNINGS++;
      sprintf(sSELECTOR_PRM,"(ordered)");
    }
  }
  else {
    fprintf(stderr,"\n%s - Error line %d: The %s selection scheme does not exist in CUDA/STD.\n",sEZ_FILE_NAME,EASEALexer->yylineno, sSELECTOR);
    exit(1);
  }
}
