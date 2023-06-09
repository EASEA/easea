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
	// NOTE: to reproduce std::uniform, we need [min, max[ to become [min, max], hence the + 1
	//constexpr fast_bounded_distribution(T min_, T max_) : min(min_), max(max_+1), span(max - min) {
	// probably std::nextafter is better in order to convert from [min, max) to [min,max]
	constexpr fast_bounded_distribution(T min_, T max_)
		: min(min_), max(std::nextafter(max_, max_ + 1)), span(max - min)
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
	xoshiro::Xoshiro256PP engine;

    public:
	CRandomGenerator(unsigned int seed);
	CRandomGenerator();

	// NOTE: expected behavior: [min, max[, hence the -1
	// but because of old code : [0, 0[ => 0 and [0, 1[ => 0
	template <typename T>
	T random(T min, T max)
	{
		if constexpr (std::is_integral_v<T>) {
			assert(min < max && "range [a, b[ with b <= a is impossible");
			if (min == max - 1) // [a, a+1[ => a
				return min;
		}
		impl::fast_bounded_distribution<T> dis(min, max);
		return dis(engine);
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

	bool tossCoin(float bias)
	{
		assert(bias <= 1.f && "Probability above 1 makes no sense");
		if (random(0.f, 1.f) <= bias)
			return true;
		else
			return false;
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

template <typename T>
T random(T min, T max)
{
	extern CRandomGenerator* globalRandomGenerator;
	return globalRandomGenerator->random(min, max);
}

static inline bool tossCoin(float bias)
{
	extern CRandomGenerator* globalRandomGenerator;
	return globalRandomGenerator->tossCoin(bias);
}

static inline bool tossCoin()
{
	extern CRandomGenerator* globalRandomGenerator;
	return globalRandomGenerator->tossCoin();
}

#endif /* CRANDOMGENERATOR_H_ */
