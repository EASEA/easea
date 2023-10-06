#include "operators/crossover/base/CCrossover.h"
#include "operators/crossover/C2x2CrossoverLauncher.h"
#include "operators/crossover/C3x1CrossoverLauncher.h"
#include "operators/crossover/base/C2x2Crossover.h"
#include "operators/crossover/base/C3x1Crossover.h"
#include "operators/crossover/continuous/CbetaCrossover.h"
#include "operators/crossover/continuous/CdeCrossover.h"
#include "operators/crossover/continuous/CsbxCrossover.h"
#include "operators/crossover/wrapper/CWrap2x2Crossover.h"
#include "operators/crossover/wrapper/CWrap3x1Crossover.h"
#include "operators/crossover/wrapper/CWrapCrossover.h"

// reduce compilation time and check for errors while compiling lib
template class easea::operators::crossover::C2x2CrossoverLauncher<float, std::vector<float>, easea::DefaultGenerator_t>;
template class easea::operators::crossover::C2x2CrossoverLauncher<double, std::vector<double>, easea::DefaultGenerator_t>;

template class easea::operators::crossover::CWrap2x2Crossover<float, std::vector<float>>;
template class easea::operators::crossover::CWrap2x2Crossover<double, std::vector<double>>;

template class easea::operators::crossover::CCrossover<float, std::vector<float>>;
template class easea::operators::crossover::CCrossover<double, std::vector<double>>;

template class easea::operators::crossover::C2x2Crossover<float, std::vector<float>>;
template class easea::operators::crossover::C2x2Crossover<double, std::vector<double>>;

template class easea::operators::crossover::continuous::sbx::CsbxCrossover<float, easea::DefaultGenerator_t>;
template class easea::operators::crossover::continuous::sbx::CsbxCrossover<double, easea::DefaultGenerator_t>;

template class easea::operators::crossover::continuous::beta::CbetaCrossover<float, easea::DefaultGenerator_t>;
template class easea::operators::crossover::continuous::beta::CbetaCrossover<double, easea::DefaultGenerator_t>;

template class easea::operators::crossover::continuous::de::CdeCrossover<float, easea::DefaultGenerator_t>;
template class easea::operators::crossover::continuous::de::CdeCrossover<double, easea::DefaultGenerator_t>;

template class easea::operators::crossover::CWrapCrossover<float, std::vector<float>>;
template class easea::operators::crossover::CWrapCrossover<double, std::vector<double>>;

template class easea::operators::crossover::CWrap3x1Crossover<float, std::vector<float>>;
template class easea::operators::crossover::CWrap3x1Crossover<double, std::vector<double>>;

template class easea::operators::crossover::C3x1CrossoverLauncher<float, std::vector<float>, easea::DefaultGenerator_t>;
template class easea::operators::crossover::C3x1CrossoverLauncher<double, std::vector<double>, easea::DefaultGenerator_t>;

template class easea::operators::crossover::C3x1Crossover<float, std::vector<float>>;
template class easea::operators::crossover::C3x1Crossover<double, std::vector<double>>;
