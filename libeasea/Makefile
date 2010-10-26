UNAME := $(shell uname)

CXXFLAGS = -g -Wall -fmessage-length=0 -I../boost/

OBJS = CRandomGenerator.o CSelectionOperator.o CEvolutionaryAlgorithm.o\
	   CStoppingCriterion.o COptionParser.o CPopulation.o CIndividual.o\
	CGnuplot.o CCmaes.o CCmaesCuda.o Parameters.o CGPNode.o\
	CComUDPLayer.o CCuda.o

ifeq ($(EZ_CUDA),1)
OBJS += CCuda.o
endif

#ifeq ($(UNAME),Darwin)
LIBS = 
#else ifeq ($(UNAME),Linux)
#LIBS = -lboost_program_options 
#endif #OS

TARGET =	libeasea.a
CPPFLAGS=-I/usr/local/cuda/include/ -I../boost/ -pg

$(TARGET): $(OBJS)
	ar rcs $@ $^ #../boost/program_options.a 

%.o:%.cpp
	$(CXX)  -c $(CXXFLAGS) $(CPPFLAGS) $^

all:	$(TARGET)
clean:
	rm -f $(OBJS) $(TARGET)  CCuda.o
