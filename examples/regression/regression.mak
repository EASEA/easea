
NVCC= /usr/local/cuda-9.2/bin/nvcc
CPPC= g++
LIBEASEA=$(EZ_PATH)libeasea/
CXXFLAGS+= -std=c++11 -g -Wall -O2 -I$(LIBEASEA)include 
LDFLAGS= $(LIBEASEA)libeasea.a -lpthread 



regression_SRC= regressionIndividual.cpp
regression_MAIN_HDR= regression.cpp
regression_UC_HDR= regressionIndividual.hpp

regression_HDR= $(regression_SRC:.cpp=.hpp) 

SRC= $(regression_SRC) $(regression_MAIN_HDR)
CUDA_SRC = regressionIndividual.cu
HDR= $(regression_HDR) $(regression_UC_HDR)
OBJ= $(regression_SRC:.cpp=.o) $(regression_MAIN_HDR:.cpp=.o)

#USER MAKEFILE OPTIONS :
 

CXXFLAGS+=-I/usr/local/cuda/common/inc/ -I/usr/local/cuda/include/
LDFLAGS+=
#END OF USER MAKEFILE OPTIONS

CPPFLAGS+= -I$(LIBEASEA)include  -I/usr/local/cuda/include/
NVCCFLAGS+= -std=c++11 #--ptxas-options="-v"# --gpu-architecture sm_23 --compiler-options -fpermissive 


BIN= regression
  
all:$(BIN)

$(BIN):$(OBJ)
	$(NVCC) $^ -o $@ $(LDFLAGS) -Xcompiler -fopenmp

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -c -DTIMING $(CPPFLAGS) -g -Xcompiler -fopenmp 

easeaclean: clean
	rm -f Makefile regression.prm $(SRC) $(HDR) regression.mak $(CUDA_SRC) *.linkinfo regression.png regression.dat regression.vcproj regression.plot regression.r regression.csv regression.pop
clean:
	rm -f $(OBJ) $(BIN) 	
	
