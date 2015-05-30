#pragma once

// Set of "temporary" workarounds for Windows / Visual Studio
#ifdef _MSC_VER 

#define isatty(...) 0

char* strndup(const char *s, size_t size);

char* basename(const char* path);
#endif
