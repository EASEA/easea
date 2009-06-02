EXEC = main.out
CPPFLAGS += -DUNIX_OS -Ialexyacc/include/ -g  -Wno-deprecated
CPPC = g++ 
LDFLAGS = 



$(EXEC):EaseaSym.o EaseaParse.o EaseaLex.o alexyacc/libalex.a EaseaYTools.o
	$(CPPC) $(CPPFLAGS) $(LDFLAGS) $^ -o $@
	#
	# Congratulations !  It looks like you compiled EASEA successfully.
	# 
	# Generated files depend on libboost-program-options,
	# be sure that the development version of this library 
	# is installed on you system :
	#       For example, on ubuntu :
	#       sudo apt-get install libboost-program-options-dev
	#
	# Easea could be moved to a bin directory or included in the PATH 
	# as long as users have defined a EZ_PATH environment variable 
	# pointing to the tpl/ directory.
	# To do this temporarly type : 
	#       export EZ_PATH=`pwd`/tpl/
	# Or define EZ_PATH in your bashrc file (for bash users) :
	#       For example :
	#       export EZ_PATH=/path/to/easea/directory/tpl/
	#
	# Otherwise you can use easea from this directory by typing :
	#       For example : 
	#       ./easea examples/weierstrass/weierstrass.ez
	# Go to the taget directory and type make
	#
	# Thanks for using EASEA.
	#

# $(EXEC):EaseaSym.o EaseaParse.o EaseaLex.o alexyacc/libalex.so
# 	$(CPPC) $(CPPFLAGS) $(LDFLAGS) $^ -o $@


# $(EXEC)_bin:EaseaSym.o EaseaParse.o EaseaLex.o 
# 	$(CPPC) $(CPPFLAGS) $(LDFLAGS) $^ -o $@ -lalex



EaseaParse.o: EaseaParse.cpp EaseaLex.cpp
	$(CPPC) $(CPPFLAGS) $< -o $@ -c

%.o:%.cpp
	$(CPPC) $(CPPFLAGS) -c -o $@ $<

#compile library for alex and ayacc unix version
alexyacc/libalex.so:alexyacc/*.cpp
	cd alexyacc && make libalex.so

alexyacc/libalex.a:alexyacc/*.cpp
	cd alexyacc && make libalex.a


clean:
	rm -f *.o $(EXEC) $(EXEC)_bin
	cd alexyacc && make clean

install:$(EXEC)
	sudo cp $< /usr/bin/dev-easea

realclean: clean
	rm -f EaseaParse.cpp EaseaParse.h EaseaLex.cpp EaseaLex.h


EaseaParse.cpp: EaseaParse.y
	wine /home/maitre/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ayacc.exe $< -Tcpp -d

EaseaLex.cpp: EaseaLex.l
	wine /home/maitre/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ALex.exe $< -Tcpp -i
