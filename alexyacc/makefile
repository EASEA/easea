OBJS= yycback.o yyccdeci.o yycclex.o yyccwork.o yycdeci.o yycdepop.o \
yycdisc.o yycdoutp.o yycecho.o yycerror.o yycflex.o yycgetch.o yycgtok.o \
yycinput.o yyclcln.o yyclcon.o yyclcrea.o yycldbug.o yycldest.o yycless.o \
yycoutpt.o yycparse.o yycpcln.o yycpcon.o yycpcrea.o yycpdbug.o yycpdest.o \
yycpop.o yycpush.o yycreset.o yycsetin.o yycsetup.o yycsoflw.o yycsskip.o \
yycsssiz.o yycstsiz.o yycsusiz.o yycsyner.o yyctoflw.o yycuncin.o \
yycunput.o yycuoflw.o yycwipe.o yycwork.o yycwrap.o


CPPC = g++
CPPFLAGS = -Iinclude/
LIB_NAME=libalex.so
STATIC_LIB_NAME = libalex.a

$(STATIC_LIB_NAME): $(OBJS)
	ar -r $@ $(OBJS) && ranlib $@


$(LIB_NAME): $(OBJS)
	$(CPPC) -shared -o $@ $^

%.o:%.cpp
	$(CPPC) $(CPPFLAGS) -fpic -c $<

clean:
	rm -f *.o $(LIB_NAME) $(STATIC_LIB_NAME)