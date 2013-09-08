UNAME = $(shell uname)
ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif

CXXFLAGS =  -O -g2 -Wall -fmessage-length=0 -I/usr/include/lfc -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include -I/usr/local/cuda/include #-I../boost/ -llcg_util -lgfal_pthr

OBJS = CRandomGenerator.o CSelectionOperator.o CEvolutionaryAlgorithm.o\
	   CStoppingCriterion.o COptionParser.o CPopulation.o CIndividual.o\
	CGrapher.o CCmaes.o CCmaesCuda.o Parameters.o CGPNode.o\
	CStats.o CGPNode.o CComWorker.o CComWorkerListmanager.o CComGridUDPServer.o CComWorkerListmanager_2.o CComWorkerListmanager_3.o gfal_utils.o\
	monitorworkerlistmanager_2.o worker_workerlistmanager.o networkcommworkercommunicator.o\
	filecommworkercommunicator.o abstractworkerlistmanager.o

ifneq ("$(OS)","")
OBJS += inet_pton.o
endif

#ifeq ($(EZ_CUDA),1)
#OBJS += CCuda.o
#endif

ifneq ("$(OS)","")
LIBS = -L"C:\MinGW\lib" -lws2_32 -lwinmm
endif

TARGET = libeasea.a
ifneq ("$(OS)","")
	CPPFLAGS= -I..\boost\ #-pg
else
	CPPFLAGS= -I../boost/ #-pg
endif

$(TARGET): $(OBJS)
	ar rcs $@ $^ 

%.o:%.cpp
	$(CXX) $(LIBS) -c $(CXXFLAGS) $(CPPFLAGS) $^ 

all:	$(TARGET)
clean:
ifneq ("$(OS)","")
	-del $(OBJS) $(TARGET)
else
	rm -f $(OBJS) $(TARGET)
endif
