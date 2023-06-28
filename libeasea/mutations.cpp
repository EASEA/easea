#include "operators/mutation/base/CMutation.h"
#include "operators/mutation/wrapper/CWrapMutation.h"
#include "operators/mutation/continuous/CGaussianMutation.h"
#include "operators/mutation/continuous/CPolynomialMutation.h"
#include "operators/mutation/continuous/CSelfGaussianMutation.h"

// reduce compilation time and check for errors while compiling lib
template class easea::operators::mutation::CMutation<float, std::vector<float>>;
template class easea::operators::mutation::CMutation<double, std::vector<double>>;

template class easea::operators::mutation::continuous::pm::CGaussianMutation<float, easea::DefaultGenerator_t>;
template class easea::operators::mutation::continuous::pm::CGaussianMutation<double, easea::DefaultGenerator_t>;

template class easea::operators::mutation::continuous::pm::CPolynomialMutation<float, easea::DefaultGenerator_t>;
template class easea::operators::mutation::continuous::pm::CPolynomialMutation<double, easea::DefaultGenerator_t>;

template class easea::operators::mutation::continuous::pm::CSelfGaussianMutation<float, easea::DefaultGenerator_t>;
template class easea::operators::mutation::continuous::pm::CSelfGaussianMutation<double, easea::DefaultGenerator_t>;

template class easea::operators::mutation::CWrapMutation<float, std::vector<float>>;
template class easea::operators::mutation::CWrapMutation<double, std::vector<double>>;
