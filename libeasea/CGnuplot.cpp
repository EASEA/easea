#include "include/CGnuplot.h"
#include <stdio.h>

CGnuplot::CGnuplot(int nbEval){
#ifdef __linux__
	int toFils[2];
        int toPere[2];
        int sonPid;
	this->valid=1;

        if(pipe(toFils)<0){
                perror("PipeComOpen: Creating pipes");
                this->valid=0;	
		return;
        }
        if(pipe(toPere)<0){
                perror("PipeComOpen: Creating pipes");
                this->valid=0;
		return;
        }
        switch((sonPid=vfork())){
                case -1:
                        perror("PipeComOpen: fork failed");
			this->valid=0;
                        break;
                case 0:
                        /* --- here's the son --- */
                        if(dup2(toFils[0], fileno(stdin))<0){
                                perror("PipeComOpen(son): could not connect\n");
				this->valid=0;
                                abort();
			}
                        if(dup2(toPere[1], fileno(stdout))<0){
                                perror("PipeComOpen(son): could not connect\n");
				this->valid=0;
                                abort();
                        }
                        char *arg[2];
                        arg[0] = (char*)"-persist";
                        arg[1] = 0;
                        if(execvp("gnuplot",arg)<0){
                                perror("gnuplot not installed, please change plotStats parameter\n");
                                abort();
				this->valid=0;
                        }
                        break;
                default	:
			if(this->valid){
                        	this->fWrit = (FILE *)fdopen(toFils[1],"w");
                        	this->fRead = (FILE *)fdopen(toPere[0],"r");
                        	this->pid = sonPid;
				fprintf(this->fWrit,"set term wxt persist\n");
				fprintf(this->fWrit,"set grid\n");
				fprintf(this->fWrit,"set xrange[0:%d]\n",nbEval);
				fprintf(this->fWrit,"set xlabel	\"Number of Evaluations\"\n");
				fprintf(this->fWrit,"set ylabel \"Fitness\"\n");
				fflush(this->fWrit);
			}
        }
#endif
}

CGnuplot::~CGnuplot(){
#ifdef __linux__
	fprintf(this->fWrit,"quit\n");
	fclose(this->fRead);
	fclose(this->fWrit);
#endif
}
