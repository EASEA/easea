#include <math.h>
#include <stdlib.h>
#include "include/CCuda.h"
#include <stdio.h>
 

CCuda::CCuda(unsigned parentSize, unsigned offSize, unsigned individualImplSize){
	this->sizeOfIndividualImpl = individualImplSize;
	this->cudaBuffer = (void*)malloc(this->sizeOfIndividualImpl*( (parentSize>offSize) ? parentSize : offSize));
}

CCuda::~CCuda(){
}

bool repartition(struct my_struct_gpu* gpu_infos){
	
	//There is an implied minimum number of threads for each block
        if(gpu_infos->num_Warp > gpu_infos->num_thread_max){
            printf("You need to authorized at least %d threads on each block!\n",gpu_infos->num_Warp);
            exit(1);
        }

        gpu_infos->dimGrid = gpu_infos->num_MP;
        gpu_infos->dimBlock = gpu_infos->num_Warp;;
  	
        //While each element of the population can't be placed on the card
        while(gpu_infos->dimBlock * gpu_infos->dimGrid < gpu_infos->sh_pop_size) {
             //Every time we add the number of Warp to the value of dimBlock
             if( (gpu_infos->dimBlock += gpu_infos->num_Warp) > gpu_infos->num_thread_max ) {
                  //If the number of dimBlock exceeds the number of threads max, we add the number of MP to the value of dimGrid and we reset the value of dimBlock with the number of Warp
                  gpu_infos->dimGrid += gpu_infos->num_MP;
                  gpu_infos->dimBlock = gpu_infos->num_Warp;
             }
        }


	//Verification that we have enough place for all the population and that every constraints are respected
	if( (gpu_infos->dimBlock*gpu_infos->dimGrid >= gpu_infos->sh_pop_size) && (gpu_infos->dimBlock <= gpu_infos->num_thread_max))
		return true;
	else 
		return false;
}



