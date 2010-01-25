#define NUMTHREAD2 128
#define MAX_STACK 50
#define LOGNUMTHREAD2 7

#define NUMTHREAD 32
#define LOGNUMTHREAD 5

#define HIT_LEVEL  0.01f
#define PROBABLY_ZERO  1.11E-15f
#define BIG_NUMBER 1.0E15f


struct gpuArg{
  int threadId;
  sem_t sem_in;
  sem_t sem_out;

  float* progs_k;
  float* results_k;
  int* indexes_k;
  int* hits_k;
  float* inputs_k;
  float* outputs_k;

  int index_st;
  int index_end;

  int indiv_st;
  int indiv_end;
};

struct gpuArg* gpuArgs;
bool freeGPU = false;
int sh_pop_size = 0;
int sh_length = 0;




__global__ static void 
fastEvaluatePostFixIndividuals_32_mgpu( const float * k_progs,
					const int maxprogssize,
					const int popsize,
					const float * k_inputs,
					const float * k_outputs,
					const int trainingSetSize,
					float * k_results,
					int *k_hits,
					
					const int indivPerBlock,
					const int* indexes,
					const int start_index,
					const int gpu_id
					){

  extern __shared__ float scratch[];
  float* tmpresult = scratch+(threadIdx.y*NUMTHREAD);
  float* tmphits = scratch+(indivPerBlock*NUMTHREAD)+(threadIdx.y*NUMTHREAD);
  /* __shared__ float tmpresult[NUMTHREAD]; */
  /* __shared__ float tmphits[NUMTHREAD]; */

  const int tid = threadIdx.x; //0 to NUM_THREADS-1
  const int bid = blockIdx.x+threadIdx.y*(popsize/indivPerBlock)+blockIdx.y*gridDim.x; // 0 to NUM_BLOCKS-1

  int index;   // index of the prog processed by the block 
  float sum = 0.0;
  int hits = 0 ; // hits number

  float currentOutput;
  float result;
  int start_prog;
  int codop;
  float stack[MAX_STACK];
  int  sp, var_id;
  float op1, op2;

  index = bid; // one program per block => block ID = program number

  if (index >= popsize){ // idle block (should never occur)
    return;
  }
  if (indexes[index] == -1.0) // already evaluated
    return;


  // Here, it's a busy thread

  sum = 0.0;
  hits = 0 ; // hits number

  

  // Loop on training cases, per cluster of 32 cases (= number of thread)
  // (even if there are only 8 stream processors, we must spawn at least 32 threads) 
  // We loop from 0 to upper bound INCLUDED in case trainingSetSize is not 
  // a multiple of NUMTHREAD
  for (int i=0; i < ((trainingSetSize-1)>>LOGNUMTHREAD)+1; i++) {

    int fc_id = i*NUMTHREAD+tid;
    // are we on a busy thread?
    if ( fc_id >= trainingSetSize) // no!
      continue;

    currentOutput = k_inputs[(1+fc_id)*DRONE_VAR_LEN+DRONE_FCT_ID];
    const float* currentInputs = k_inputs+(fc_id)*DRONE_VAR_LEN;
    start_prog = indexes[index]-start_index; // index of first codop
    codop =  k_progs[start_prog++];
    
    sp = 0; // stack and stack pointer
    
    while (codop != OP_RETURN){
      switch(codop){
	case OP_VAR : 
	  var_id = k_progs[start_prog++];
	  stack[sp++] = currentInputs[var_id];
	  break;      
      case OP_ERC: stack[sp++] = k_progs[start_prog++]; break;
      case OP_MUL :
	op1 = stack[--sp]; op2 = stack[sp-1];
	stack[sp-1] = op1*op2; break;
      case OP_ADD :
	op1 = stack[--sp]; op2 = stack[sp-1];
	stack[sp-1] = op1+op2; break;
      case OP_SUB :
	op1 = stack[--sp]; op2 = stack[sp-1];
	stack[sp-1] = op2 - op1; break;
      case OP_DIV :
	op2 = stack[--sp]; op1 = stack[sp-1];
	if (op2 == 0.0) stack[sp-1] = DIV_ERR_VALUE;
	else stack[sp-1] = op1/op2;
	break;
      case OP_SIN : stack[sp-1] = sinf(stack[sp-1]); break;
      case OP_COS : stack[sp-1] = cosf(stack[sp-1]); break;
      }
      // get next codop
      codop =  k_progs[start_prog++];
    } // codop interpret loop

    result = fabsf(stack[0] - currentOutput);
    
    if (!(result < BIG_NUMBER))
      result = BIG_NUMBER;
    else if (result < PROBABLY_ZERO)
      result = 0.0;
    
    if (result <= HIT_LEVEL)
      hits++;
    
    sum += result; // sum raw error on all training cases
    
  } // LOOP ON TRAINING CASES

  // gather results from all threads => we need to synchronize (not sure, take a look to next comment)
  tmpresult[tid] = sum;
  tmphits[tid] = hits;

  //__syncthreads(); // this is useless, because warps are synchronized by nature

  if( tid == 0 ){
    for (int i = 1; i < NUMTHREAD; i++) {
      tmpresult[0] += tmpresult[i];
      tmphits[0] += tmphits[i];
    }    
    k_results[index] = tmpresult[0];
    k_hits[index] = tmphits[0];
    //printf("tid.y = %d k_results %d = %f\n",threadIdx.y,index,k_results[index]);
  }  
}




__global__ static void 
EvaluatePostFixIndividuals_128_mgpu(const float * k_progs,
				    const int maxprogssize,
				    const int popsize,
				    const float * k_inputs,
				    const float * k_outputs,
				    const int trainingSetSize,
				    float * k_results,
				    int *k_hits,
				    int* k_indexes,
				    int start_index,
				    int gpu_id
			       )
{
  __shared__ float tmpresult[NUMTHREAD2];
  __shared__ float tmphits[NUMTHREAD2];
  
  const int tid = threadIdx.x; //0 to NUM_THREADS-1
  const int bid = blockIdx.x; // 0 to NUM_BLOCKS-1

  
  int index;   // index of the prog processed by the block 
  float sum = 0.0;
  int hits = 0 ; // hits number

  float currentOutput;
  float result;
  int start_prog;
  int codop;
  float stack[MAX_STACK];
  int  sp, var_id;
  float op1, op2;

  index = bid; // one program per block => block ID = program number
 
  if (index >= popsize) // idle block (should never occur)
    return;
  if (k_progs[index] == -1.0) // already evaluated
    return;

  // Here, it's a busy thread

  sum = 0.0;
  hits = 0 ; // hits number
  
  // Loop on training cases, per cluster of 32 cases (= number of thread)
  // (even if there are only 8 stream processors, we must spawn at least 32 threads) 
  // We loop from 0 to upper bound INCLUDED in case trainingSetSize is not 
  // a multiple of NUMTHREAD
  for (int i=0; i < ((trainingSetSize-1)>>LOGNUMTHREAD2)+1; i++) {
    int fc_id = i*NUMTHREAD2+tid;
    // are we on a busy thread?
    if (fc_id >= trainingSetSize) // no!
      continue;
    
    currentOutput = k_inputs[(1+fc_id)*DRONE_VAR_LEN+DRONE_FCT_ID];
    start_prog = k_indexes[index]-start_index; // index of first codop
    codop =  k_progs[start_prog++];
    
    const float* currentInputs = k_inputs+(fc_id)*DRONE_VAR_LEN;
    sp = 0; // stack and stack pointer
    
    while (codop != OP_RETURN){
      switch(codop)
	{
	case OP_VAR : 
	  var_id = k_progs[start_prog++];
	  stack[sp++] = currentInputs[var_id];
	  break;
	case OP_ERC: stack[sp++] = k_progs[start_prog++]; break;
	case OP_MUL :
	  op1 = stack[--sp]; op2 = stack[sp-1];
	  stack[sp-1] = op1*op2; break;
	case OP_ADD :
	  op1 = stack[--sp]; op2 = stack[sp-1];
	  stack[sp-1] = op1+op2; break;
	case OP_SUB :
	  op1 = stack[--sp]; op2 = stack[sp-1];
	  stack[sp-1] = op2 - op1; break;
	case OP_DIV :
	  op2 = stack[--sp]; op1 = stack[sp-1];
	  if (op2 == 0.0) stack[sp-1] = DIV_ERR_VALUE;
	  else stack[sp-1] = op1/op2;
	  break;
	case OP_POW :
	  op2 = stack[--sp]; op1 = stack[sp-1];
	  stack[sp-1] = powf(op1,op2);
	  break;
        case OP_SIN : stack[sp-1] = sinf(stack[sp-1]); break;
        case OP_COS : stack[sp-1] = cosf(stack[sp-1]); break;
	}
      // get next codop
      codop =  k_progs[start_prog++];
    } // codop interpret loop
    result = fabsf(stack[0] - currentOutput);
    
    if (!(result < BIG_NUMBER)) result = BIG_NUMBER;
    else if (result < PROBABLY_ZERO) result = 0.0;

    if (result <= HIT_LEVEL) hits++;
    
    sum += result; // sum raw error on all training cases
    
  } // LOOP ON TRAINING CASES
  
  // gather results from all threads => we need to synchronize
  tmpresult[tid] = sum;
  tmphits[tid] = hits;
  __syncthreads();

  if (tid == 0) {
    for (int i = 1; i < NUMTHREAD2; i++) {
      tmpresult[0] += tmpresult[i];
      tmphits[0] += tmphits[i];
    }    
    k_results[index] = tmpresult[0];
    k_hits[index] = tmphits[0];
    //printf("g %d %d %f\n",gpu_id,bid,k_results[index]);
    //fflush(stdout);
  }  
  // here results and hits have been stored in their respective array: we can leave
}



void wake_up_gpu_thread(int nbGpu){
    for( int i=0 ; i<nbGPU ; i++ ){
    DEBUG_PRT("wake up th %d",i);
    //fflush(stdout);
    sem_post(&(gpuArgs[i].sem_in));
  }

  for( int i=0 ; i<nbGPU ; i++ ){
    sem_wait(&gpuArgs[i].sem_out);
  }
}

void notify_gpus(float* progs, int* indexes, int length, CIndividual** population, int popSize, int nbGpu){

  int pop_chunk_len = popSize / nbGpu;
  //cout << " population chunk length : " << pop_chunk_len << "/" << length << endl;
  assert(nbGpu==2);
#ifdef INSTRUMENTED  
  currentStats.gpu0Blen = indexes[pop_chunk_len];
  currentStats.gpu1Blen = length-indexes[pop_chunk_len];
#endif
  sh_pop_size = pop_chunk_len;
  sh_length = length;
  
  wake_up_gpu_thread(nbGpu);
}



/**
   Send input and output data on the GPU memory.
   Allocate
*/
void initialDataToMGPU(float* input_f, int length_input, float* output_f, int length_output, int gpu_id){
  // allocate and copy input/output arrays
  CUDA_SAFE_CALL(cudaMalloc((void**)(&(gpuArgs[gpu_id].inputs_k)),sizeof(float)*length_input));
  CUDA_SAFE_CALL(cudaMemcpy((gpuArgs[gpu_id].inputs_k),input_f,sizeof(float)*length_input,cudaMemcpyHostToDevice));

  if( output_f ){
    CUDA_SAFE_CALL(cudaMalloc((void**)(&(gpuArgs[gpu_id].outputs_k)),sizeof(float)*length_output));
  }
  else {
    printf("no output buffer, dont need to allocate it\n");
    gpuArgs[gpu_id].outputs_k = NULL;
  }

  if( output_f ){
    CUDA_SAFE_CALL(cudaMemcpy((gpuArgs[gpu_id].outputs_k),output_f,sizeof(float)*length_output,cudaMemcpyHostToDevice));
  }
  else {
    printf("no output buffer, dont need to copy it\n");
    gpuArgs[gpu_id].outputs_k = NULL;
  }				   
  

  // allocate indexes and programs arrays
  int maxPopSize = MAX(EA->population->parentPopulationSize,EA->population->offspringPopulationSize);
  CUDA_SAFE_CALL( cudaMalloc((void**)&(gpuArgs[gpu_id].indexes_k),sizeof(*indexes_k)*maxPopSize));
  CUDA_SAFE_CALL( cudaMalloc((void**)&(gpuArgs[gpu_id].progs_k),sizeof(*progs_k)*MAX_PROGS_SIZE));

  // allocate hits and results arrays
  CUDA_SAFE_CALL(cudaMalloc((void**)&(gpuArgs[gpu_id].results_k),sizeof(*indexes_k)*maxPopSize));
  CUDA_SAFE_CALL(cudaMalloc((void**)&(gpuArgs[gpu_id].hits_k),sizeof(*indexes_k)*maxPopSize));
}


void* gpuThreadMain(void* arg){
  struct gpuArg* localArg = (struct gpuArg*)arg;

  DEBUG_PRT("gpu th %d",localArg->threadId);
  CUDA_SAFE_CALL(cudaSetDevice(localArg->threadId));

  // Alloc memory for this thread
  initialDataToMGPU(inputs_f, fitnessCasesSetLength*DRONE_VAR_LEN, NULL, 0,localArg->threadId);
  
  DEBUG_PRT("allocation ok for th %d",localArg->threadId);
  //sem_post(&localArg->sem_out);

  // Wait for population to evaluate.
  while(1){
    //printf("gpu %d is evaluating\n",localArg->threadId);
    sem_wait(&localArg->sem_in);

    if( freeGPU )
      break;

    int indiv_st = localArg->threadId*sh_pop_size;
    int indiv_end = indiv_st+sh_pop_size;
    int index_st = indexes[indiv_st];
    int index_end = 0;
    if( localArg->threadId != nbGPU-1 ) index_end = indexes[indiv_end];
    else index_end = sh_length;
    
    /* cout << "gpu " << localArg->threadId << " has been notified" << endl; */
    /* cout << indiv_st << "|" << indiv_end << "|" << index_st << "|" << index_end << endl; */
    /* fflush(stdout); */

    int no_tries = 10;
    for( int i=0 ; i<no_tries ; i++ ){

      //here we copy assigned population chunk to the related GPU
      CUDA_SAFE_CALL(cudaMemcpy( localArg->indexes_k, indexes+indiv_st, (indiv_end-indiv_st)*sizeof(int), cudaMemcpyHostToDevice ));
      CUDA_SAFE_CALL(cudaMemcpy( localArg->progs_k, progs+index_st, (index_end-index_st)*sizeof(int), cudaMemcpyHostToDevice ));

#if 1
      EvaluatePostFixIndividuals_128_mgpu<<<sh_pop_size,128>>>(localArg->progs_k, index_end-index_st, sh_pop_size, localArg->inputs_k, localArg->outputs_k,
							       NB_FITNESS_CASES-1, localArg->results_k, localArg->hits_k, localArg->indexes_k, index_st, localArg->threadId);
#else
      int indivPerBlock = 4;
      dim3 numthreads;
      numthreads.x = 32;
      numthreads.y = indivPerBlock;
    
      fastEvaluatePostFixIndividuals_32_mgpu<<<sh_pop_size/indivPerBlock,numthreads,NUMTHREAD*sizeof(float)*2*indivPerBlock>>>
	(localArg->progs_k, index_end-index_st, sh_pop_size, localArg->inputs_k, localArg->outputs_k, NB_FITNESS_CASES, 
	 localArg->results_k, localArg->hits_k, indivPerBlock, localArg->indexes_k, index_st, localArg->threadId);
#endif
      /* cudaThreadSynchronize(); */
      cudaError_t kernel_status = cudaMemcpy( results+(localArg->threadId*sh_pop_size), localArg->results_k, (indiv_end-indiv_st)*sizeof(int), cudaMemcpyDeviceToHost);
      
      if( kernel_status==cudaSuccess)break;
      else cout << "try :" << i << "for generation " << EA->getCurrentGeneration() << " fails " << endl;
      
      if( i==no_tries-1){
	std::ostringstream oss;
	oss << "best-of-run-" << EA->params->seed << ".exp" ;
	EA->population->sortParentPopulation();
	ofstream fichier(oss.str().c_str(), ios::out | ios::trunc);
	fichier << EA->population->parents[0]->getFitness() << endl;
	fichier << ((IndividualImpl*)EA->population->parents[0])->hits << endl;
	fichier << treeGP_to_c( ((IndividualImpl*)EA->population->parents[0])->root[0] ) << endl;
	fichier.close();
	
	cerr << "Kernel fail to launch" << endl;
	exit(-1);
      }

    }

    CUDA_SAFE_CALL( cudaMemcpy( hits+(localArg->threadId*sh_pop_size), localArg->hits_k, (indiv_end-indiv_st)*sizeof(int), cudaMemcpyDeviceToHost));
    
    sem_post(&localArg->sem_out);

  }
  DEBUG_PRT("gpu : %d",localArg->threadId);
  DEBUG_PRT("addr k_prog : %p",localArg->progs_k);
  CUDA_SAFE_CALL(cudaFree(localArg->progs_k));
  CUDA_SAFE_CALL(cudaFree(localArg->results_k));
  CUDA_SAFE_CALL(cudaFree(localArg->hits_k));
  CUDA_SAFE_CALL(cudaFree(localArg->inputs_k));
  if( localArg->outputs_k ) CUDA_SAFE_CALL(cudaFree(localArg->outputs_k));
  sem_post(&localArg->sem_out);
  cout << "gpu " << localArg->threadId << " has been freed" << endl;
  fflush(stdout);

  return NULL;
}
