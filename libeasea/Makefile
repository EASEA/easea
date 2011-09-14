UNAME = $(shell uname)
ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif

CXXFLAGS = -g -Wall -fmessage-length=0 #-I../boost/

OBJS = CRandomGenerator.o CSelectionOperator.o CEvolutionaryAlgorithm.o\
	   CStoppingCriterion.o COptionParser.o CPopulation.o CIndividual.o\
	CGrapher.o CCmaes.o CCmaesCuda.o Parameters.o CGPNode.o\
	CComUDPLayer.o CCuda.o CStats.o

ifneq ("$(OS)","")
OBJS += inet_pton.o
endif

ifeq ($(EZ_CUDA),1)
OBJS += CCuda.o
endif

ifneq ("$(OS)","")
LIBS = -L"C:\MinGW\lib" -lws2_32 -lwinmm
endif

TARGET = libeasea.a
ifneq ("$(OS)","")
	CPPFLAGS=-I..\boost\ #-pg
else
	CPPFLAGS=-I/usr/local/cuda/include/ -I../boost/ #-pg
endif

$(TARGET): $(OBJS)
	ar rcs $@ $^ 

%.o:%.cpp
	$(CXX) $(LIBS) -c $(CXXFLAGS) $(CPPFLAGS) $^ 

all:	$(TARGET)
clean:
ifneq ("$(OS)","")
	-del $(OBJS) $(TARGET) CCuda.o
else
	rm -f $(OBJS) $(TARGET)  CCuda.o
endif
