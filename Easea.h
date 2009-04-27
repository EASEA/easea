/****************************************************************************
Easea.h
General header for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@polytechnique.fr)
Ecole Polytechnique
Centre de Mathématiques Appliquées
91128 Palaiseau cedex
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

#define true 1
#define false 0

#define GALIB 1
#define EO 2
#define DREAM 3
#define CUDA 4
#define STD 5

#define STD_FLAVOR_SO 0
#define STD_FLAVOR_MO 1
#define CUDA_FLAVOR_SO 0
#define CUDA_FLAVOR_MO 0

#define UNIX 1
#define WINDOWS 2
#define UNKNOWN_OS 3
#define YYTEXT_SIZE 10000
class CSymbol;

extern CSymbol *pCURRENT_CLASS, *pCURRENT_TYPE, *pGENOME, *pCLASSES[128];
extern int nClasses_nb;

extern   FILE *fpOutputFile, *fpTemplateFile, *fpGenomeFile, *fpExplodedGenomeFile;  
extern char sRAW_PROJECT_NAME[], sPROJECT_NAME[], sEO_DIR[], sEZ_PATH[1000], sEZ_FILE_NAME[];
extern char sLOWER_CASE_PROJECT_NAME[];
extern char sREPLACEMENT[], sSELECTOR[], sSELECT_PRM[], sRED_PAR[], sRED_PAR_PRM[], sRED_FINAL[], sRED_FINAL_PRM[], sRED_OFF[], sRED_OFF_PRM[], sDISCARD[], sDISCARD_PRM[];
extern int nMINIMISE,nELITE;
extern bool bELITISM, bVERBOSE;
extern bool bPROP_SEQ;
extern int nPOP_SIZE, nNB_GEN, nNB_ISLANDS, nOFF_SIZE, nSURV_PAR_SIZE, nSURV_OFF_SIZE;
extern float fMUT_PROB, fXOVER_PROB, fREPL_PERC, fMIG_FREQ;
extern int nMIG_CLONE, nNB_MIG, nIMMIG_REPL;
extern char sMIG_SEL[], sMIGRATOR[], sIMMIG_SEL[],sMIG_TARGET_SELECTOR[];

extern int nWARNINGS, nERRORS;
extern int TARGET, OPERATING_SYSTEM;
extern int TARGET_FLAVOR;

// Prototypes
extern int mystricmp(char *, char *);
