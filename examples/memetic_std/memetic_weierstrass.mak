

UNAME := $(shell uname)

ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif

ifneq ("$(OS)","")
	EZ_PATH=../../
endif

memetic_weierstrassLIB_PATH=$(EZ_PATH)/libeasea/

CXXFLAGS =  -fopenmp	-O2 -g -Wall -fmessage-length=0 -I$(memetic_weierstrassLIB_PATH)include -I$(EZ_PATH)boost

OBJS = memetic_weierstrass.o memetic_weierstrassIndividual.o 

LIBS = -lpthread -fopenmp 
ifneq ("$(OS)","")
	LIBS += -lws2_32 -lwinmm -L"C:\MinGW\lib"
endif

#USER MAKEFILE OPTIONS :
 
CPPFLAGS+=

#END OF USER MAKEFILE OPTIONS

TARGET =	memetic_weierstrass

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS) -g $(memetic_weierstrassLIB_PATH)/libeasea.a $(EZ_PATH)boost/program_options.a $(LIBS)

	
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
	-del $(TARGET).exe *.o *.cpp *.hpp memetic_weierstrass.png memetic_weierstrass.dat memetic_weierstrass.prm memetic_weierstrass.mak Makefile memetic_weierstrass.vcproj memetic_weierstrass.csv memetic_weierstrass.r memetic_weierstrass.plot memetic_weierstrass.pop
else
	rm -f $(TARGET) *.o *.cpp *.hpp memetic_weierstrass.png memetic_weierstrass.dat memetic_weierstrass.prm memetic_weierstrass.mak Makefile memetic_weierstrass.vcproj memetic_weierstrass.csv memetic_weierstrass.r memetic_weierstrass.plot memetic_weierstrass.pop
endif

