#include "shared/CRandom.h"

// reduce compilation time and check for errors while compiling lib
template class easea::shared::CRandom<easea::DefaultGenerator_t>;
