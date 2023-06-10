/*
 * CRandomGenerator.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 *
 *  =====
 *  Updated on: 26 july 2022 by Léo Chéneau to use modern engine
 *  =====
 *  Updated again on 09/06/2023 by Léo Chéneau to use XorShift256++
 *
 */

#ifndef CRANDOMGENERATOR_H_
#define CRANDOMGENERATOR_H_

#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

#include "xoshiro.hpp"

namespace impl
{
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, T> fast_nextafter(T val)
{
	return val;
}

template <typename T>
std::enable_if_t<std::is_integral_v<T>, T> fast_nextafter(T val)
{
	return val + 1;
}
/* NOTE: this distribution was tested on :
	 * - mt19937
	 * - lcg
	 * It is almost perfectly uniform +- 0.5% but is 7x times faster than std::uniform_...
	 * Example of tests: https://godbolt.org/z/Kq7xsEdzo
	 */
template <typename T>
class fast_bounded_distribution
{
	using span_t = std::conditional_t<std::is_floating_point_v<T>, T, std::size_t>;
	T min, max;
	span_t span;

    public:
	// NOTE: [min, max[
	constexpr fast_bounded_distribution(T min_, T max_) : min(min_), max(max_), span(max - min)
	{
		assert(min < max && "[a; b[ with a < b required");
	}

	template <typename Gen, typename V = T>
	constexpr std::enable_if_t<std::is_floating_point_v<V>, V> operator()(Gen&& gen) const
	{
		return ((static_cast<V>(gen() - gen.min()) / static_cast<V>(gen.max())) * span) + min;
	}

	template <typename Gen, typename V = T>
	constexpr auto operator()(Gen&& gen) const
		-> std::enable_if_t<std::is_integral_v<V> && std::is_unsigned_v<decltype(gen())>, V>
	{
		return static_cast<V>(span_t{ gen() - gen.min() } % span) + min;
	}
};
} // namespace impl

class CRandomGenerator
{
    private:
	unsigned seed;
	Xoshiro::Xoshiro128PP engine;
	Xoshiro::Xoshiro256PP engine64;

    public:
	CRandomGenerator(unsigned int seed);
	CRandomGenerator();

	// NOTE: expected behavior: [min, max[, hence the -1
	// but because of old code : [0, 0[ => 0 and [0, 1[ => 0
	template <typename T>
	T random(T min, T max)
	{
		assert(min <= max && "range [a, b[ with b < a is impossible");
		if (min == max) // [a, a[ => a
			return min;
		impl::fast_bounded_distribution<T> dis(min, max);
		if constexpr (sizeof(T) <= sizeof(decltype(engine)::result_type))
			return dis(engine);
		else
			return dis(engine64);
	}

	template <typename T>
	T random()
	{
		return random(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
	}

	bool tossCoin()
	{
		return random(0, 2) == 1;
	}

	template <typename T>
	std::enable_if_t<std::is_floating_point_v<T>, bool> tossCoin(float bias)
	{
		assert(0.f <= bias && bias <= 1.f && "Probability above 1 or below 0 makes no sense");
		return random(0.f, 1.f) <= bias;
	}

	template <typename T>
	std::enable_if_t<std::is_integral_v<T>, bool> tossCoin(int prct)
	{
		assert(0 <= prct && prct <= 100 && "Probability above 100% or below 0% makes no sens");
		return random(0, 101) <= prct;
	}

	/*
	 * Old functions
	 */
	int getRandomIntMax(int max)
	{
		return random(0, max);
	}

	float random_gauss(float mean, float std_dev)
	{
		std::normal_distribution<float> dis(mean, std_dev);
		return dis(engine);
	}

	unsigned get_seed() const
	{
		return seed;
	}
};

std::ostream& operator<<(std::ostream& os, const CRandomGenerator& rg);

extern CRandomGenerator globGen;

template <typename T>
T random(T min, T max)
{
	return globGen.random(min, max);
}

template <typename T>
static inline bool tossCoin(T bias)
{
	return globGen.tossCoin<T>(bias);
}

static inline bool tossCoin()
{
	return globGen.tossCoin();
}

#endif /* CRANDOMGENERATOR_H_ */
