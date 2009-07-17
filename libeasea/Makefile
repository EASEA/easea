CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS = CRandomGenerator.o CSelectionOperator.o CEvolutionaryAlgorithm.o\
	   CStoppingCriterion.o COptionParser.o CPopulation.o CIndividual.o\
	CGnuplot.o CCmaes.o CCuda.o  CCmaesCuda.o

LIBS = -lboost_program_options 

TARGET =	libeasea.a

$(TARGET): $(OBJS)
	ar rcs $@ $^

%.o:%.cpp
	$(CXX)  -c $(CXXFLAGS) $^


all:	$(TARGET)
clean:
	rm -f $(OBJS) $(TARGET)
	
