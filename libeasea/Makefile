UNAME := $(shell uname)

ifeq ($(UNAME),Darwin)
CXXFLAGS = -g -Wall -fmessage-length=0 -I../boost/
else ifeq ($(UNAME),Linux)
CXXFLAGS = -g -Wall -fmessage-length=0 
endif #OS

OBJS = CRandomGenerator.o CSelectionOperator.o CEvolutionaryAlgorithm.o\
	   CStoppingCriterion.o COptionParser.o CPopulation.o CIndividual.o\
	CGnuplot.o CCmaes.o CCuda.o CCmaesCuda.o Parameters.o CGPNode.o

ifeq ($(UNAME),Darwin)
LIBS = 
else ifeq ($(UNAME),Linux)
LIBS = -lboost_program_options 
endif #OS

TARGET =	libeasea.a
CPPFLAGS=-I/usr/local/cuda/include/ 

$(TARGET): $(OBJS)
ifeq ($(UNAME),Darwin)
	ar rcs $@ $^ ../boost/program_options.a 
else ifeq ($(UNAME),Linux)
	ar rcs $@ $^
endif #OS
#$(TARGET): $(OBJS)
#	ar rcs $@ $^

%.o:%.cpp
	$(CXX)  -c $(CXXFLAGS) $(CPPFLAGS) $^


all:	$(TARGET)
clean:
	rm -f $(OBJS) $(TARGET)
