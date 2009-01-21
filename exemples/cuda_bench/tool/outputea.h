#ifndef OUTPUT_EA_H
#define OUTPUT_EA_H



#define TMP_BUF_LENGTH 512

#define TIMING
#include "timing.h"

#include <libxml/tree.h>
#include "basetype.h"

#ifdef __cplusplus
extern "C" {
#endif 


  typedef struct {
    DECLARE_TIME(gpu);
    DECLARE_TIME(cpu);
    DECLARE_TIME(init);
    
    DECLARE_TIME(krnl);
    DECLARE_TIME(memCpy1);
    DECLARE_TIME(memCpy2);
    DECLARE_TIME(alloc);
    BOOLEAN_EA cmp;
    BOOLEAN_EA repartition;
    size_t popSize;

    size_t nbBlock,nbThreadPB,nbThreadLB,memSize,sharedMemSize;
    size_t fakeIteration;

  }OutputEa;
  
  
  xmlNodePtr outputEaToXmlNode(OutputEa* tt);
  xmlChar* timevalToXmlChar(struct timeval* ts);
  void outputTss(const char* filename);
  void addTs(OutputEa* ts);
  void initTss(const char* args);
  void attachSignal();
  

#ifdef __cplusplus
}
#endif 

#endif
