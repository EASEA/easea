UNAME = $(shell uname)
ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif

ifeq ($(shell uname -o 2>/dev/null),Cygwin)
	OS := CYGWIN
endif

CXXFLAGS = -g -O1 -std=c++14 -fopenmp -Wall -fmessage-length=0 -I/usr/local/cuda/include
CXXFLAGS += -std=c++14 -Wold-style-cast -pedantic -ggdb 
OBJS = CRandomGenerator.o CSelectionOperator.o CEvolutionaryAlgorithm.o\
	   CStoppingCriterion.o COptionParser.o CPopulation.o CIndividual.o\
	CGrapher.o CCmaes.o CCmaesCuda.o Parameters.o CGPNode.o\
	CComUDPLayer.o CStats.o CGPNode.o

ifneq ("$(OS)","")
OBJS += inet_pton.o
endif

#ifeq ($(EZ_CUDA),1)
#OBJS += CCuda.o
#endif

ifneq ("$(OS)","")
LIBS = -L"C:\MinGW\lib" -lws2_32 -lwinmm
CXXFLAGS += -I"C:\MinGW\include"
endif
#CC=nvcc
#ARCH=-arch=sm_20
#CFLAGS=-dc -Iinclude $(ARCH)

#SOURCES=$(shell echo *.cu)
#HEADERS=$(shell echo include/*.h)
#OBJECTS=$(SOURCES:.cu=.o)
#STATIC_LIB=libcudaeasea.a



TARGET = libeasea.a
#ifneq ("$(OS)","")
#	CPPFLAGS=-I..\boost\ #-pg
#else
#	CPPFLAGS=-I../boost/ #-pg
#endif

$(TARGET): $(OBJS)
	ar rcs $@ $^ 

#$(STATIC_LIB): $(OBJECTS)
#	$(CC) $(ARCH) -lib $(OBJECTS) -o $@


%.o:%.cpp
	$(CXX) $(LIBS) -c -std=c++14 -I/home/anna/Documents/EASEA/easea-master/cxxopts -Wall -Wextra -Wold-style-cast -pedantic -ggdb -Wold-style-cast $(CXXFLAGS) $(CPPFLAGS) $^ 
#%.o: %.cu
#	$(CC) $(CFLAGS) $< -o $@


all:	$(TARGET)
clean:
	rm -f $(OBJS) $(TARGET)
