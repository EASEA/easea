#include "core/CmoIndividual.h"

// reduce compilation time and check for errors while compiling lib
template class easea::CmoIndividual<double, std::vector<double>>;
template class easea::CmoIndividual<float, std::vector<float>>;
