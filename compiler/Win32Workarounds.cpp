
// Set of "temporary" workarounds for windows
#ifdef _MSC_VER 
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>

#define isatty(...) 0

char* strndup(const char *s, size_t size)
{
    // HACK
    return strdup(s);
}

#if _MSC_VER >= 1800
#include <boost/filesystem.hpp>

char* basename(const char* path)
{
    char fname[1000];
    _splitpath(path, 0, 0, fname, 0);
    return strdup(fname);
}
#endif
#endif