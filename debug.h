#ifdef DEBUG
#define DEBUG_PRT(format, args...) fprintf (stdout,"***DBG***  %s-%d: "format"\n",__FILE__,__LINE__,##args)
#define DEBUG_YACC(format, args...) fprintf (stdout,"***DBG_YACC***  %s-%d: "format"\n",__FILE__,__LINE__,##args)

#else
#ifndef WIN32
#define DEBUG_PRT(format, args...) 
#define DEBUG_YACC(format, args...)
#endif
#endif
