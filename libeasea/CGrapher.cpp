#include "include/CGrapher.h"
#include "include/Parameters.h"
#include <stdio.h>
#include <string.h>

CGrapher::CGrapher(Parameters* param, char* title) {
    this->valid=0;
#ifdef WIN32
//TODO: Create a child process and and pipe some line from the father to the child
//(http://msdn.microsoft.com/en-us/library/windows/desktop/ms682499%28v=vs.85%29.aspx)
#endif

#ifndef WIN32
    int toFils[2];
    int toPere[2];
    int sonPid;
    this->valid=1;

    if(pipe(toFils)<0) {
        perror("PipeComOpen: Creating pipes");
        this->valid=0;
        return;
    }
    if(pipe(toPere)<0) {
        perror("PipeComOpen: Creating pipes");
        this->valid=0;
        return;
    }
    switch((sonPid=vfork())) {
    case -1:
        perror("PipeComOpen: fork failed");
        this->valid=0;
        break;
    case 0:
        /* --- here's the son --- */
        if(dup2(toFils[0], fileno(stdin))<0) {
            perror("PipeComOpen(son): could not connect\n");
            this->valid=0;
            abort();
        }
        if(dup2(toPere[1], fileno(stdout))<0) {
            perror("PipeComOpen(son): could not connect\n");
            this->valid=0;
            abort();
        }
        char* pPath;
        pPath = getenv("EZ_PATH");
        if(pPath != NULL) {
            pPath = strcat(pPath, "easeagrapher/EaseaGrapher.jar");
        } else {
            pPath = (char*)"../../easeagrapher/EaseaGrapher.jar";
        }
        char *arg[4];
        arg[0] = (char*)"java";
        arg[1] = (char*)"-jar";
        arg[2] = pPath;
        arg[3] = (char*)0;
        if(execvp("java",arg)<0) {
            perror("java not installed, please change plotStats parameter\n");
            abort();
            this->valid=0;
        }
        break;
    default :
        if(this->valid) {
            this->fWrit = (FILE *)fdopen(toFils[1],"w");
            this->fRead = (FILE *)fdopen(toPere[0],"r");
            this->pid = sonPid;
            /*fprintf(this->fWrit,"set term wxt persist\n");
            fprintf(this->fWrit,"set grid\n");
            fprintf(this->fWrit,"set xrange[0:%d]\n",nbEval);
            fprintf(this->fWrit,"set xlabel \"Number of Evaluations\"\n");
            fprintf(this->fWrit,"set ylabel \"Fitness\"\n");*/
            int nbEval = param->offspringPopulationSize*param->nbGen +
                         param->parentPopulationSize;
            fprintf(this->fWrit,"set max eval:%d\n",nbEval);
            fprintf(this->fWrit,"set title:%s\n",title);
            if(param->remoteIslandModel) {
                fprintf(this->fWrit,"set island model\n");
                fprintf(this->fWrit,"set max generation:%d\n",param->nbGen);
            }
            fflush(this->fWrit);
        }
    }
#endif
}

CGrapher::~CGrapher() {
#ifndef WIN32
    //fprintf(this->fWrit,"quit\n");
    fclose(this->fRead);
    fclose(this->fWrit);
#endif
}
