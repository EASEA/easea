#include "include/CGnuplot.h"

CGnuplot::CGnuplot(){
#ifdef __linux__
	int toFils[2];
        int toPere[2];
        int sonPid;

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
                                perror("PipeComOpen(son): could not connect");
				this->valid=0;
				return;
                                //abort();
			}
                        if(dup2(toPere[1], fileno(stdout))<0){
                                perror("PipeComOpen(son): could not connect");
				this->valid=0;
				return;
                                //abort();
                        }
                        char *arg[2];
                        arg[0] = "-persist";
                        arg[1] = 0;
                        if(execvp("gnuplot",arg)<0){
                                perror("gnuplot not installed, please change plotStats parameter");
                                //abort();
				this->valid=0;
				break;
                        }
                        break;
                default:
                        this->fWrit = (FILE *)fdopen(toFils[1],"w");
                        this->fRead = (FILE *)fdopen(toPere[0],"r");
                        this->pid = sonPid;
			this->valid = 1;
			fprintf(this->fWrit,"set term x11 persist\n");
			fprintf(this->fWrit,"set grid\n");
			fflush(this->fWrit);
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
