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

#define CUDA 4
#define STD 5
#define CMAES 6
#define MEMETIC 7

#define STD_FLAVOR_SO 0
#define STD_FLAVOR_MO 1
#define CUDA_FLAVOR_SO 0
#define CUDA_FLAVOR_MO 1
#define CUDA_FLAVOR_GP 2


#define UNIX 1
#define WINDOWS 2
#define UNKNOWN_OS 3
#define YYTEXT_SIZE 10000
class CSymbol;

extern CSymbol *pCURRENT_CLASS, *pCURRENT_TYPE, *pGENOME, *pCLASSES[128];
extern int nClasses_nb;

extern   FILE *fpOutputFile, *fpTemplateFile, *fpGenomeFile, *fpExplodedGenomeFile;  
extern char sRAW_PROJECT_NAME[], sPROJECT_NAME[], sEO_DIR[], sEZ_PATH[1000],  sTPL_DIR[1000], sEZ_FILE_NAME[];
extern char sLOWER_CASE_PROJECT_NAME[];
extern char  sSELECTOR[], sSELECTOR_OPERATOR[], sRED_PAR[], sRED_PAR_OPERATOR[], sRED_FINAL[], sRED_FINAL_OPERATOR[], sRED_OFF[], sRED_OFF_OPERATOR[];
extern char sIP_FILE[];
extern bool bELITISM, bVERBOSE;
extern bool bBALDWINISM;
extern bool bPRINT_STATS, bPLOT_STATS, bGENERATE_CSV_FILE, bGENERATE_GNUPLOT_SCRIPT, bGENERATE_R_SCRIPT;
extern bool bSAVE_POPULATION, bSTART_FROM_FILE;
extern bool bREMOTE_ISLAND_MODEL;
extern bool bLINE_NUM_EZ_FILE;
extern char* nGENOME_NAME;
extern int nPOP_SIZE, nNB_GEN, nNB_OPT_IT, nOFF_SIZE, nPROBLEM_DIM, nTIME_LIMIT;
extern int nSERVER_PORT;
extern int nWARNINGS, nERRORS;
extern int TARGET, OPERATING_SYSTEM;
extern int TARGET_FLAVOR;
extern int nMINIMISE,nELITE;
extern float fMUT_PROB, fXOVER_PROB, fSURV_PAR_SIZE, fSURV_OFF_SIZE;
extern float fSELECT_PRM, fRED_PAR_PRM, fRED_FINAL_PRM, fRED_OFF_PRM;
extern float fMIGRATION_PROBABILITY;

extern unsigned iMAX_INIT_TREE_D,iMIN_INIT_TREE_D,iMAX_TREE_D,iNB_GPU,iPRG_BUF_SIZE,iMAX_TREE_DEPTH,iNO_FITNESS_CASES;

// Prototypes
extern int mystricmp(const char *, const char *);
