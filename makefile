EXEC = main.out
CPPFLAGS = -DUNIX_OS -Ialexyacc/include/
CPPC = g++

$(EXEC):EaseaSym.o EaseaLex.o EaseaParse.o alexyacc/libalex.so
	$(CPPC) $(CPPFLAGS) $^ -o $@


%.o:%.cpp
	$(CPPC) $(CPPFLAGS) -c -o $@ $<


alexyacc/libalex.so:alexyacc/*.cpp
	cd alexyacc && make libalex.so

clean:
	rm -f *.o $(EXEC) 
	cd alexyacc && make clean


# g++ -DUNIX_OS -Iinclude EaseaLex.o  EaseaParse.o EaseaSym.o alexyacc/*.o -o main.out
# g++ EaseaSym.cpp -Iinclude/ -DUNIX_OS -c
# g++ EaseaLex.cpp -Iinclude/ -DUNIX_OS -c
# g++ EaseaParse.cpp -Iinclude/ -DUNIX_OS -c