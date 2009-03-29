EXEC = main.out
CPPFLAGS += -DUNIX_OS -Ialexyacc/include/ -g
CPPC = g++
LDFLAGS = 


$(EXEC):EaseaSym.o EaseaParse.o EaseaLex.o alexyacc/libalex.so
	$(CPPC) $(CPPFLAGS) $(LDFLAGS) $^ -o $@


EaseaParse.o: EaseaParse.cpp EaseaLex.cpp	
	$(CPPC) $(CPPFLAGS) $< -o $@ -c

%.o:%.cpp
	$(CPPC) $(CPPFLAGS) -c -o $@ $<

#compile library for alex and ayacc unix version
alexyacc/libalex.so:alexyacc/*.cpp
	cd alexyacc && make libalex.so

clean:
	rm -f *.o $(EXEC)
	cd alexyacc && make clean

realclean: clean
	rm -f EaseaParse.cpp EaseaParse.h EaseaLex.cpp EaseaLex.h


EaseaParse.cpp: EaseaParse.y
	wine ~/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ayacc.exe $< -Tcpp -d

EaseaLex.cpp: EaseaLex.l
	wine ~/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ALex.exe $< -Tcpp -i