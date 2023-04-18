#ifndef _EASEALEX_H
#define _EASEALEX_H

#include "EaseaSym.h"
  // forward references
  class CSymbolTable;
  class OPCodeDesc;

  enum COPY_GP_EVAL_STATUS {EVAL_HDR,EVAL_BDY,EVAL_FTR};

  int CEASEALexer_create(CSymbolTable&& pSymbolTable);
  double myStrtod();

#endif
