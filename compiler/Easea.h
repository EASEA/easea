#ifndef _EASEA__H
#define _EASEA__H

/****************************************************************************
Easea.h
General header for the EASEA language (EAsy Specification for Evolutionary Algorithms)

Pierre COLLET (Pierre.Collet@polytechnique.fr)
Ecole Polytechnique
Centre de Math?matiques Appliqu?es
91128 Palaiseau cedex
---------
Reformated on june 2022 by Léo Chéneau to improved portability with Windows
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

//#define true 1
//#define false 0 // BRUH

#define CUDA 4
#define STD 5
#define CMAES 6
#define MEMETIC 7

#define STD_FLAVOR_SO 0
#define STD_FLAVOR_MO 1
#define CUDA_FLAVOR_SO 0
#define CUDA_FLAVOR_MO 1
#define FLAVOR_GP 2
#define NSGAII 11
#define ASREA 12
#define FASTEMO 15
#define FASTEMOII 16
#define NSGAIII 17
#define MOEAD 18
#define IBEA 19
#define CDAS 20
#define QIEA 22
#define QAES 23
#define CUDA_QAES 24
#define NSGAII_CUDA 25

#define UNIX 1
#define WINDOWS 2
#define UNKNOWN_OS 3
#define YYTEXT_SIZE 10000

class CSymbol;

extern CSymbol *pCURRENT_CLASS, *pCURRENT_TYPE, *pGENOME, *pCLASSES[128];
extern int nClasses_nb;

extern FILE *fpOutputFile, *fpTemplateFile, *fpGenomeFile, *fpExplodedGenomeFile;
extern char sRAW_PROJECT_NAME[], sPROJECT_NAME[], sEO_DIR[], sEZ_PATH[1000], sTPL_DIR[1000], sEZ_FILE_NAME[];
extern char sLOWER_CASE_PROJECT_NAME[];
extern char sSELECTOR[], sSELECTOR_OPERATOR[], sRED_PAR[], sRED_PAR_OPERATOR[], sRED_FINAL[], sRED_FINAL_OPERATOR[],
	sRED_OFF[], sRED_OFF_OPERATOR[];
extern char sIP_FILE[];
extern bool bELITISM, bVERBOSE;
extern bool bBALDWINISM;
extern bool bPRINT_STATS, bPLOT_STATS, bGENERATE_CSV_FILE, bGENERATE_GNUPLOT_SCRIPT, bGENERATE_R_SCRIPT;
extern bool bSAVE_POPULATION, bSTART_FROM_FILE;
extern bool bREMOTE_ISLAND_MODEL;
extern bool bLINE_NUM_EZ_FILE;
extern char* nGENOME_NAME;
extern int nPOP_SIZE, nNB_GEN, nARCH_SIZE, nNB_OPT_IT, nOFF_SIZE, nPROBLEM_DIM, nTIME_LIMIT;
extern int nSERVER_PORT;
extern int nWARNINGS, nERRORS;
extern int TARGET, OPERATING_SYSTEM;
extern int TARGET_FLAVOR;
extern int nMINIMISE, nELITE;
extern float fMUT_PROB, fXOVER_PROB, fSURV_PAR_SIZE, fSURV_OFF_SIZE;
extern float fSELECT_PRM, fRED_PAR_PRM, fRED_FINAL_PRM, fRED_OFF_PRM;
extern float fMIGRATION_PROBABILITY;

extern unsigned iMAX_INIT_TREE_D, iMIN_INIT_TREE_D, iMAX_TREE_D, iNB_GPU, iPRG_BUF_SIZE, iMAX_TREE_DEPTH,
	iNO_FITNESS_CASES;

// Prototypes
static inline char mytolower(char c)
{
	return ((c >= 65) && (c <= 90)) ? c += 32 : c;
}

static inline int mystricmp(const char* string1, const char* string2)
{
	int i;
	for (i = 0; string1[i] && string2[i]; i++) {
		if (mytolower(string1[i]) < mytolower(string2[i]))
			return -(i + 1);
		if (mytolower(string1[i]) > mytolower(string2[i]))
			return i + 1;
	}
	if (string2[i])
		return -(i + 1);
	if (string1[i])
		return i + 1;
	return 0;
}

static inline int isLetter(char c)
{
	if (((c >= 65) && (c <= 90)) || ((c >= 97) && (c <= 122)))
		return 1;
	if ((c == 45) || (c == 46) || (c == 95))
		return 1;
	return 0;
}

static inline int isFigure(char c)
{
	if ((c >= 48) && (c <= 57))
		return 1;
	return 0;
}

#endif
