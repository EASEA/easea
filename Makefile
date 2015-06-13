UNAME := $(shell uname)
ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif
YACC=bison
LEX=flex
EXEC = bin/easea
CPPFLAGS += -DUNIX_OS -g -Wno-deprecated -DDEBUG -DLINE_NUM_EZ_FILE
LDFLAGS = 

OBJ= build/EaseaSym.o build/EaseaParse.o build/EaseaLex.o build/EaseaYTools.o boost/program_options.a libeasea/libeasea.a

#ifeq ($(UNAME),Darwin)
$(EXEC):build bin $(OBJ)
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(OBJ) -o $@ 
ifneq ("$(OS)","")
	#
	# Congratulations !  It looks like you compiled EASEA successfully.
	# 
	# You can use easea from this directory by typing :
	#       For example : 
	#      easea.exe examples\weierstrass_std\weierstrass.ez
	# Go to the target directory and type make -f weierstrass.mak
	#
	# Thanks for using EASEA.
	#
else
	#
	# Congratulations !  It looks like you compiled EASEA successfully.
	# 
	# You can now install easea into your system or use it from 
	# its current directory.
	#
	# Installation:
	# To install EASEA into your system, type:
	#	 ". install.sh ".
	# EASEA will be installed into /usr/local/easea/ directory,
	# including, the binary, its libraries and the templates.
	# Finaly, environment variables will be updated (EZ_PATH and PATH),
ifeq ($(UNAME),Darwin)
	# into your .bash_profile file.
else
	# into your .bashrc file.
endif
	# 
	# Local Usage:
	# All EASEA elements will stay in the current directory, 
	# but some environment variables need to be updated into your
ifeq ($(UNAME),Darwin)
	# .bash_profile file (EZ_PATH and). To do so type:
else
	# .bashrc file (EZ_PATH and). To do so type:
endif
	#	 ". install.sh local". 
	#
	# Thanks for using EASEA.
	#
endif

# $(EXEC):EaseaSym.o EaseaParse.o EaseaLex.o alexyacc/libalex.so
# 	$(CXX) $(CPPFLAGS) $(LDFLAGS) $^ -o $@


# $(EXEC)_bin:EaseaSym.o EaseaParse.o EaseaLex.o 
# 	$(CXX) $(CPPFLAGS) $(LDFLAGS) $^ -o $@ -lalex


install:
	mkdir -p $(DESTDIR)$(PREFIX)/easea
	mkdir -p $(DESTDIR)$(PREFIX)/easea/bin
	mkdir -p $(DESTDIR)$(PREFIX)/easea/tpl
	mkdir -p $(DESTDIR)$(PREFIX)/easea/include
	mkdir -p $(DESTDIR)$(PREFIX)/easea/boost
	mkdir -p $(DESTDIR)$(PREFIX)/easea/easeagrapher
	cp bin/easea $(DESTDIR)$(PREFIX)/easea/bin/
	cp tpl/* $(DESTDIR)$(PREFIX)/easea/tpl/
	cp libeasea/include/* $(DESTDIR)$(PREFIX)/easea/libeasea/include/
	cp libeasea/libeasea.a $(DESTDIR)$(PREFIX)/easea/libeasea/
	cp boost/program_options.a $(DESTDIR)$(PREFIX)/easea/boost
	cp -r boost/boost/ $(DESTDIR)$(PREFIX)/easea/boost/boost/
	cp easeagrapher/EaseaGrapher.jar $(DESTDIR)$(PREFIX)/easea/easeagrapher/

	 
build:
	@test -d build || mkdir build || echo "Cannot make dir build"
bin:
	@test -d bin || mkdir bin || echo "Cannot make dir bin"


build/EaseaParse.o: compiler/EaseaParse.cpp compiler/EaseaLex.cpp
	$(CXX) $(CPPFLAGS) $< -o $@ -c -w
build/EaseaLex.o:  compiler/EaseaLex.cpp compiler/EaseaParse.hpp
	$(CXX) $(CPPFLAGS) $< -o $@ -c -w


build/%.o:compiler/%.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $<

compiler/EaseaParse.cpp:compiler/EaseaParse.y
	$(YACC) -d -o $@ $<

compiler/EaseaLex.cpp:compiler/EaseaLex.l
	$(LEX) -o $@ $<

#ifeq ($(UNAME),Darwin)
boost/program_options.a:boost/*.cpp
	cd boost && make program_options.a
#endif #OS

#compile libeasea
libeasea/libeasea.a:libeasea/*.cpp
	cd libeasea && make libeasea.a

clean:
	rm -f build/*.o $(EXEC) $(EXEC)_bin
	cd libeasea && make clean
	cd boost && make clean
#ifeq ($(UNAME),Darwin)
	cd boost && make clean
#endif
	
#install:$(EXEC)
#	sudo cp $< /usr/bin/dev-easea

#ifeq ($(UNAME),Linux)
#realclean: clean
#	rm -f EaseaParse.cpp EaseaParse.h EaseaLex.cpp EaseaLex.h

# AT commented these lines, because they imply the presence of wine + programs in a specific location
#EaseaParse.cpp: EaseaParse.y
#	wine ~/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ayacc.exe $< -Tcpp -d

#EaseaLex.cpp: EaseaLex.l
#	wine ~/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ALex.exe $< -Tcpp -i
#endif
