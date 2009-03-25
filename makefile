EXEC = main.out
CPPFLAGS = -DUNIX_OS -Ialexyacc/include/ -DDEBUG -g
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
	rm -f EaseaParse.cpp EaseaParse.h EaseaLex.cpp EaseaLex.h
	cd alexyacc && make clean



EaseaParse.cpp: EaseaParse.y
	wine ~/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ayacc.exe $< -Tcpp -d

EaseaLex.cpp: EaseaLex.l
	wine ~/.wine/drive_c/Program\ Files/Parser\ Generator/BIN/ALex.exe $< -Tcpp -i


# #handle file between windows and local directory
# winclean:
# 	rm -f $(TMP_DIR)/EaseaLex.cpp $(TMP_DIR)/EaseaLex.h $(TMP_DIR)/EaseaParse.cpp $(TMP_DIR)/EaseaParse.h $(TMP_DIR)/EaseaParse.v

# winrealclean: winclean
# 	rm -f $(TMP_DIR)/EaseaParse.y $(TMP_DIR)/EaseaLex.l

# #send alex and ayacc files to windows
# winsend: winsend_l winsend_y
# winsend_l:	
# 	sudo cp EaseaLex.l $(TMP_DIR)/
# winsend_y:
# 	sudo cp EaseaParse.y $(TMP_DIR)/

# #receveive alex and ayacc files from windows
# winreceive: winreceive_l winreceive_y

# winreceive_l:
# 	cp $(TMP_DIR)/EaseaLex.cpp $(TMP_DIR)/EaseaLex.h ./
# 	chmod -x EaseaLex.cpp EaseaLex.h

# winreceive_y:
# 	cp $(TMP_DIR)/EaseaParse.cpp $(TMP_DIR)/EaseaParse.h ./
# 	chmod -x EaseaParse.cpp EaseaParse.h