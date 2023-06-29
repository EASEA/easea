#include <boost/multiprecision/cpp_bin_float.hpp>

using namespace boost::multiprecision;

using real_t = boost::multiprecision::cpp_bin_float_quad;

#include "shared/CBoundary.h"
template class easea::shared::CBoundary<real_t>;

#include "core/CConstraint.h"
template class easea::CConstraint<real_t>;


#include "core/CmoIndividual.h"
template class easea::CmoIndividual<real_t, std::vector<real_t>>;


#include "shared/CProbability.h"
template class easea::shared::CProbability<real_t>;

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
template class easea::operators::crossover::C2x2CrossoverLauncher<real_t, std::vector<real_t>, easea::DefaultGenerator_t>;
template class easea::operators::crossover::CWrap2x2Crossover<real_t, std::vector<real_t>>;
template class easea::operators::crossover::CCrossover<real_t, std::vector<real_t>>;
template class easea::operators::crossover::C2x2Crossover<real_t, std::vector<real_t>>;
template class easea::operators::crossover::continuous::sbx::CsbxCrossover<real_t, easea::DefaultGenerator_t>;
template class easea::operators::crossover::continuous::beta::CbetaCrossover<real_t, easea::DefaultGenerator_t>;
template class easea::operators::crossover::continuous::de::CdeCrossover<real_t, easea::DefaultGenerator_t>;
template class easea::operators::crossover::CWrapCrossover<real_t, std::vector<real_t>>;
template class easea::operators::crossover::CWrap3x1Crossover<real_t, std::vector<real_t>>;
template class easea::operators::crossover::C3x1CrossoverLauncher<real_t, std::vector<real_t>, easea::DefaultGenerator_t>;
template class easea::operators::crossover::C3x1Crossover<real_t, std::vector<real_t>>;

#include "operators/mutation/base/CMutation.h"
#include "operators/mutation/wrapper/CWrapMutation.h"
#include "operators/mutation/continuous/CGaussianMutation.h"
#include "operators/mutation/continuous/CPolynomialMutation.h"
#include "operators/mutation/continuous/CSelfGaussianMutation.h"
template class easea::operators::mutation::CMutation<real_t, std::vector<real_t>>;
template class easea::operators::mutation::continuous::pm::CGaussianMutation<real_t, easea::DefaultGenerator_t>;
template class easea::operators::mutation::continuous::pm::CPolynomialMutation<real_t, easea::DefaultGenerator_t>;
template class easea::operators::mutation::continuous::pm::CSelfGaussianMutation<real_t, easea::DefaultGenerator_t>;
template class easea::operators::mutation::CWrapMutation<real_t, std::vector<real_t>>;


#include "variables/continuous/uniform.h"

template std::vector<real_t> easea::variables::continuous::getUniform<easea::DefaultGenerator_t, real_t>(easea::DefaultGenerator_t&, const std::vector<std::pair<real_t, real_t>>&);





