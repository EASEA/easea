UNAME := $(shell uname)
EXEC = easea 
CPPFLAGS += -DUNIX_OS -Ialexyacc/include/ -g  -Wno-deprecated -DDEBUG -DLINE_NUM_EZ_FILE
CPPC = g++ 
LDFLAGS = 



#ifeq ($(UNAME),Darwin)
$(EXEC):EaseaSym.o EaseaParse.o EaseaLex.o alexyacc/libalex.a EaseaYTools.o boost/program_options.a libeasea/libeasea.a
#else
#$(EXEC):EaseaSym.o EaseaParse.o EaseaLex.o alexyacc/libalex.a EaseaYTools.o libeasea/libeasea.a
#endif
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
	# EZ_PATH was automatically added to your .bashrc at the end of the compilation
	#
	# Easea could be moved to a bin directory or included in the PATH 
	# as long as users have defined a EZ_PATH environment variable 
	# pointing to the Easea directory.
	# To do this temporarly type : 
	#       export EZ_PATH=`pwd`/
	# Or define EZ_PATH in your bashrc file (for bash users) :
	#       For example :
	#       export EZ_PATH=/path/to/easea/directory/
	#
	# Otherwise you can use easea from this directory by typing :
	#       For example : 
	#       ./easea examples/weierstrass_std/weierstrass.ez
	# Go to the taget directory and type make
	#
	# To Activate the EZ_PATH variable type:
	# source ~/.profile
	#
	# Thanks for using EASEA.
	#

# $(EXEC):EaseaSym.o EaseaParse.o EaseaLex.o alexyacc/libalex.so
# 	$(CPPC) $(CPPFLAGS) $(LDFLAGS) $^ -o $@


# $(EXEC)_bin:EaseaSym.o EaseaParse.o EaseaLex.o 
# 	$(CPPC) $(CPPFLAGS) $(LDFLAGS) $^ -o $@ -lalex


install:
	mkdir -p /usr/local/easea/ /usr/local/easea/bin /usr/local/easea/tpl /usr/local/easea/libeasea/include /usr/local/easea/boost
	cp easea /usr/local/easea/bin/
	cp tpl/* /usr/local/easea/tpl/
	cp libeasea/include/* /usr/local/easea/libeasea/include/
	cp libeasea/libeasea.a /usr/local/easea/libeasea/
	cp boost/program_options.a /usr/local/easea/boost
	cp -r boost/boost/ /usr/local/easea/boost/boost/
vars:
ifeq ($(UNAME), Darwin)
	@sed '/EZ_PATH/d' $(HOME)/.profile>$(HOME)/.profile_save
	@mv $(HOME)/.profile_save $(HOME)/.profile
	@echo "\nexport EZ_PATH=/usr/local/easea/">>$(HOME)/.profile
	@echo "export PATH=\$$PATH:/usr/local/easea/bin" >>$(HOME)/.profile
else
	@echo "\nexport EZ_PATH=/usr/local/easea/">>$(HOME)/.bashrc
	@echo "export PATH=\$$PATH:/usr/local/easea/bin" >>$(HOME)/.bashrc
	@echo "PATH and EZ_PATH variables have been set"
endif


dev_vars:
	@echo "\nexport EZ_PATH=$(PWD)/">>$(HOME)/.bashrc ; fi


EaseaParse.o: EaseaParse.cpp EaseaLex.cpp
	$(CPPC) $(CPPFLAGS) $< -o $@ -c

%.o:%.cpp
	$(CPPC) $(CPPFLAGS) -c -o $@ $<

#compile library for alex and ayacc unix version
alexyacc/libalex.so:alexyacc/*.cpp
	cd alexyacc && make libalex.so

alexyacc/libalex.a:alexyacc/*.cpp
	cd alexyacc && make libalex.a

#ifeq ($(UNAME),Darwin)
boost/program_options.a:boost/*.cpp
	cd boost && make program_options.a
#endif #OS

#compile libeasea
libeasea/libeasea.a:libeasea/*.cpp
	cd libeasea && make libeasea.a

clean:
	rm -f *.o $(EXEC) $(EXEC)_bin
	cd alexyacc && make clean
	cd libeasea && make clean
#ifeq ($(UNAME),Darwin)
	cd boost && make clean
#endif

#install:$(EXEC)
#	sudo cp $< /usr/bin/dev-easea

ifeq ($(UNAME),Linux)
realclean: clean
	rm -f EaseaParse.cpp EaseaParse.h EaseaLex.cpp EaseaLex.h


EaseaParse.cpp: EaseaParse.y
	wine ~/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ayacc.exe $< -Tcpp -d

EaseaLex.cpp: EaseaLex.l
	wine ~/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ALex.exe $< -Tcpp -i
endif
