UNAME := $(shell uname)
ifeq ($(shell uname -o 2>/dev/null),Msys)
	OS := MINGW
endif
EXEC = bin/easea
CPPFLAGS += -DUNIX_OS -Ialexyacc/include/ -g  -Wno-deprecated -DDEBUG -DLINE_NUM_EZ_FILE
LDFLAGS = 

OBJ= build/EaseaSym.o build/EaseaParse.o build/EaseaLex.o alexyacc/libalex.a build/EaseaYTools.o boost/program_options.a libeasea/libeasea.a

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
	#	 "sudo make install".
	# EASEA will be installed into /usr/local/easa/ directory,
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
	#	 "make dev_vars". 
	#
	# Finally after having "install" or "dev_vars", reload bash config file
ifeq ($(UNAME),Darwin)
	# (by "exec bash -l" or "source ~/.bash_profile", use easea with:
else
	# (by "exec bash" or "source ~/.bashrc", use easea with:
endif
	#	easea weierstrass.ez
	#
	# Thanks for using EASEA.
	#
endif

# $(EXEC):EaseaSym.o EaseaParse.o EaseaLex.o alexyacc/libalex.so
# 	$(CXX) $(CPPFLAGS) $(LDFLAGS) $^ -o $@


# $(EXEC)_bin:EaseaSym.o EaseaParse.o EaseaLex.o 
# 	$(CXX) $(CPPFLAGS) $(LDFLAGS) $^ -o $@ -lalex


install:vars
	mkdir -p /usr/local/easea/ /usr/local/easea/bin /usr/local/easea/tpl /usr/local/easea/libeasea/include /usr/local/easea/boost /usr/local/easea/easeagrapher/
	cp bin/easea /usr/local/easea/bin/
	cp tpl/* /usr/local/easea/tpl/
	cp libeasea/include/* /usr/local/easea/libeasea/include/
	cp libeasea/libeasea.a /usr/local/easea/libeasea/
	cp boost/program_options.a /usr/local/easea/boost
	cp -r boost/boost/ /usr/local/easea/boost/boost/
	cp easeagrapher/EaseaGrapher.jar /usr/local/easea/easeagrapher/
vars:
ifeq ($(UNAME), Darwin)
	@sed '/EZ_PATH/d' $(HOME)/.bash_profile>$(HOME)/.bash_profile_save
	@mv $(HOME)/.bash_profile_save $(HOME)/.bash_profile
	@echo "export EZ_PATH=/usr/local/easea/">>$(HOME)/.bash_profile
	@echo "export PATH=\$$PATH:/usr/local/easea/bin:" >>$(HOME)/.bash_profile
else
	@echo "\nexport EZ_PATH=/usr/local/easea/">>$(HOME)/.bashrc
	@echo "export PATH=\$$PATH:/usr/local/easea/bin:" >>$(HOME)/.bashrc
	@echo "PATH and EZ_PATH variables have been set"
endif

build:
	@test -d build || mkdir build || echo "Cannot make dir build"
bin:
	@test -d bin || mkdir bin || echo "Cannot make dir bin"


dev_vars:
ifeq ($(UNAME), Darwin)
	@echo >> $(HOME)/.bash_profile
	@echo "export EZ_PATH=$(PWD)/">>$(HOME)/.bash_profile 
	@echo "export PATH=\$$PATH:$(PWD)/bin/">>$(HOME)/.bash_profile
else
	@echo >> $(HOME)/.bashrc
	@echo "export EZ_PATH=$(PWD)/">>$(HOME)/.bashrc
	@echo "export PATH=\$$PATH:$(PWD)/bin/">>$(HOME)/.bashrc 
endif


build/EaseaParse.o: EaseaParse.cpp EaseaLex.cpp
	$(CXX) $(CPPFLAGS) $< -o $@ -c -w
build/EaseaLex.o:  EaseaLex.cpp
	$(CXX) $(CPPFLAGS) $< -o $@ -c -w


build/%.o:%.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $<

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
	rm -f build/*.o $(EXEC) $(EXEC)_bin
	cd alexyacc && make clean
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
