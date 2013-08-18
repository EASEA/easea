ifneq ("$(OS)","")
	EZ_PATH=../../
endif

monitorLIB_PATH=$(EZ_PATH)/libeasea/

CXXFLAGS =	-O -g2 -Wall -fmessage-length=0 -I$(monitorLIB_PATH)include -I$(EZ_PATH)boost

OBJS = monitor.o experimetnmonitor.o 

LIBS = -lpthread 
ifneq ("$(OS)","")
	LIBS += -lws2_32 -lwinmm -L"C:\MinGW\lib"
endif

#USER MAKEFILE OPTIONS :
 
CXXFLAGS+=-I/usr/local/cuda/common/inc/ -I/usr/local/cuda/include/ -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include
LDFLAGS+=-lgfal_pthr -llcg_util

#END OF USER MAKEFILE OPTIONS

TARGET =	monitor

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS) -g $(monitorLIB_PATH)/libeasea.a $(EZ_PATH)boost/program_options.a $(LIBS)

	
#%.o:%.cpp
#	$(CXX) -c $(CXXFLAGS) $^

all:	$(TARGET)
clean:
ifneq ("$(OS)","")
	-del $(OBJS) $(TARGET).exe
else
	rm -f $(OBJS) $(TARGET)
endif
easeaclean:
ifneq ("$(OS)","")
	-del $(TARGET).exe *.o 
else
	rm -f $(TARGET) *.o 
endif

