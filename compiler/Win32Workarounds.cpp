
// Set of "temporary" workarounds for Windows / Visual Studio
#ifdef _MSC_VER 
#include <stdlib.h>
#include <string.h>

char* strndup(const char *s, size_t size)
{
    // HACK
    return strdup(s);
}

char* basename(const char* path)
{
    char fname[1000];
    _splitpath(path, 0, 0, fname, 0);
    return strdup(fname);
}
#endif