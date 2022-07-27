/*
List of possible activation functions
*/
#include "distributionFunctions.hpp"

void printDistributionFunctionName() {
	std::cout << "The implemented functions are :" << std::endl;
	std::cout << "     - normal  [(float mean, float standard_deviation), return real]" << std::endl;
	std::cout << "     - bernoulli [(float probability_true), return bool]" << std::endl;
	std::cout << "     - poisson [(float mean), return int]" << std::endl;
	std::cout << "     - uniform [(float a, float b), return real]" << std::endl;
}

bool isImplementedDistributionFunction(const std::string& dist) {
	if (dist.compare("normal") == 0) {
		return true;
	} else if (dist.compare("bernoulli") == 0) {
		return true;
	} else if (dist.compare("poisson") == 0) {
		return true;
	} else if (dist.compare("uniform") == 0) {
		return true;
	} 
	return false;
}

pcg32 generateEngine(const int seed) {
    // Engine of the random generator
	if (seed) {
		// Make a random number engine
    	pcg32 rng(seed);
		return rng;
	} else {
		// Seed with a real random value, if available
    	pcg_extras::seed_seq_from<std::random_device> seed_source;
		// Make a random number engine
    	pcg32 rng(seed_source);
		return rng;
	}
}

float generateRandom(pcg32 &engine, const std::string& dist, const float param1, const float param2 ) {
	if (dist.compare("normal") == 0) {
		std::normal_distribution<float> distribution(param1, param2);
		return distribution(engine);
	} else if (dist.compare("bernoulli") == 0) {
		std::bernoulli_distribution distribution(static_cast<double>(param1));
		// In order to not define null weights, the possible values are -0.5 and 0.5
		return static_cast<float>(distribution(engine) - 0.5);
	} else if (dist.compare("poisson") == 0) {
		std::poisson_distribution<int> distribution(static_cast<double>(param1));
		// In order to be centered around zero, we substract the mean
		return static_cast<float>(distribution(engine) - param1);
	} else if (dist.compare("uniform") == 0) {
		std::uniform_real_distribution<float> distribution(param1, param2);
		return distribution(engine);
	} else {
		std::normal_distribution<float> distribution(param1, param2);
		return distribution(engine);
	}
	return 0.;
}
