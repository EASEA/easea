
#This is an example Makefile that will create the executables exampletiming and
#exampleexperiment. This Makefile can either be completed with rules to create
#the executables for your own experiment or the rules here can be added to your
#own Makefile.

#Define your compiler/linker if needed.
#CC=

#Strict options
LDFLAGS=
CFLAGS=-Wextra -Wall -ansi -pedantic
#Lenient: add/remove any compilation or linking flags that you feel are necessary or not.
# LDFLAGS=
# CFLAGS=

DIROK=dirOK.o
#Linux/Unix users can try to comment the above and uncomment the following line instead.
# DIROK=dirOK-linux.o

#Lists all executable, used by the "all" and "clean" rules.
EXECS=exampletiming exampleexperiment

all: $(EXECS)

#Add here any rules needed for your own experiments

#Example experiments:
exampletiming: MY_OPTIMIZER.o benchmarksdeclare.o $(DIROK) benchmarkshelper.o benchmarks.o benchmarksnoisy.o fgeneric.o exampletiming.o
	$(CC) $(LDFLAGS) -o $@ $^ -lm
exampleexperiment: MY_OPTIMIZER.o benchmarksdeclare.o $(DIROK) benchmarkshelper.o benchmarks.o benchmarksnoisy.o fgeneric.o exampleexperiment.o
	$(CC) $(LDFLAGS) -o $@ $^ -lm

#Your own experiment should have bbobStructures.h as a dependency and so should be added here.
exampletiming.o exampleexperiment.o benchmarkshelper.o fgeneric.o benchmarks.o benchmarksnoisy.o benchmarksdeclare.o : bbobStructures.h

#Benchmarks object files are optimized so they are exceptions to the implicit *.o generation rule.
benchmarks.o : benchmarks.c
	$(CC) -c $(CFLAGS) -O2 $<

benchmarksnoisy.o : benchmarksnoisy.c
	$(CC) -c $(CFLAGS) -O2 $<

#Clean
.PHONY: clean

#Comment the command starting with rm and uncomment the comment starting with del to make the makefile work under Windows.
clean:
	rm -f *.o $(EXECS)
#	del *.o exampleexperiment.exe exampletiming.exe 

