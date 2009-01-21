/* #ifdef DEBUG_KRNL */
/* #define CDC( lastError, errorMsg, fun )					\ */
/*   fun ;									\ */
/*   cout <<__FILE__<< " : "<<__LINE__<<" "<<errorMsg<<" : "<<endl;	\ */
/*   cout << "\t" <<(cudaGetErrorString(lastError=cudaGetLastError()))	\ */
/*   << endl;								\ */
/*   if( lastError != cudaSuccess )exit(-1) */
/* #else */
/* #define CDC( lastError, errorMsg, fun )		\ */
/*   fun */
/* #endif */



#ifdef DEBUG_KRNL
#define CDC( lastError, errorMsg, fun )					\
  fun ;									\
  if((lastError=cudaGetLastError())!=cudaSuccess ){			\
    cout <<__FILE__<< " : "<<__LINE__<<" "<<errorMsg<<" : "<<endl;	\
    cout << "\t" <<(cudaGetErrorString(lastError))			\
	 << endl;							\
    exit(-1);								\
  }

#else
#define CDC( lastError, errorMsg, fun )		\
  fun
#endif

