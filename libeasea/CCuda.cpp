#include <math.h>
#include <stdlib.h>
#include "include/CCuda.h"
#define NB_MP 16

CCuda::CCuda(size_t parentSize, size_t offSize, size_t individualImplSize){
	this->sizeOfIndividualImpl = individualImplSize;
	this->cudaParentBuffer = (void*)malloc(this->sizeOfIndividualImpl*parentSize);
	this->cudaOffspringBuffer = (void*)malloc(this->sizeOfIndividualImpl*offSize);
}

CCuda::~CCuda(){
}

inline size_t partieEntiereSup(float E){
        int fl = floor(E);
        if(fl==E)
                return E;
        else
                return floor(E=1);
}

inline int puissanceDeuxSup(float n){
        int tmp=2;
        while(tmp<n) tmp*=2;
        return tmp;
}

bool repartition(size_t popSize, size_t* nbBlock, size_t* nbThreadPB, size_t* nbThreadLB, size_t nbMP, size_t maxBlockSize){
	(*nbThreadLB) = 0;
  
  	//DEBUG_PRT("repartition : %d",popSize);
  
  	if( ((float)popSize / (float)nbMP) <= maxBlockSize ){
	//la population repartie sur les MP tient dans une bloc par MP
		(*nbThreadPB) = partieEntiereSup( (float)popSize/(float)nbMP);
		(*nbBlock) = popSize/(*nbThreadPB);
		if( popSize%nbMP != 0 ){
		//on fait MP-1 block de equivalent et un plus petit
			(*nbThreadLB) = popSize - (*nbThreadPB)*(*nbBlock);
		}
	}
	else{
	//la population est trop grande pour etre repartie sur les MP
	//directement
		//(*nbBlock) = partieEntiereSup( (float)popSize/((float)maxBlockSize*NB_MP));
		(*nbBlock) = puissanceDeuxSup( (float)popSize/((float)maxBlockSize*NB_MP));
		(*nbBlock) *= NB_MP;
		(*nbThreadPB) = popSize/(*nbBlock);
		if( popSize%maxBlockSize!=0){
			(*nbThreadLB) = popSize - (*nbThreadPB)*(*nbBlock);
                                                                        
			// Le rest est trop grand pour etre place dans un seul block (c'est possible uniquement qd 
			// le nombre de block depasse maxBlockSize 
			while( (*nbThreadLB) > maxBlockSize ){
			//on augmente le nombre de blocs principaux jusqu'a ce que nbthreadLB retombe en dessous de maxBlockSize
				//(*nbBlock) += nbMP;
				(*nbBlock) *= 2;
				(*nbThreadPB) = popSize/(*nbBlock);
				(*nbThreadLB) = popSize - (*nbThreadPB)*(*nbBlock);
			}
		}
	}
	if((((*nbBlock)*(*nbThreadPB) + (*nbThreadLB))  == popSize) && ((*nbThreadLB) <= maxBlockSize) && ((*nbThreadPB) <= maxBlockSize))
		return true;
	else 
		return false;
}

