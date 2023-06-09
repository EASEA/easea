/*
 * CRandomGenerator.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 *
 *  =====
 *  Updated on: 26 july 2022 by Léo Chéneau to use modern engine
 *
 */

#ifndef CRANDOMGENERATOR_H_
#define CRANDOMGENERATOR_H_

#include <iostream>
#include <random>
#include <cassert>
#include <math/mft/include/mft.hpp>

class CRandomGenerator
{
    private:
	unsigned seed;
	std::minstd_rand engine;

    public:
	CRandomGenerator(unsigned int seed);
	CRandomGenerator();

	int randInt(int min, int max);
	float randFloat(float min, float max);
	double randDouble(double low, double high);

	bool tossCoin();
	bool tossCoin(float bias);

	int randInt();
	int getRandomIntMax(int max);

	int random(int min, int max);
	float random(float min, float max);
	double random(double min, double max);
	int rnd(int low, int high);

	float random_gauss(float mean, float std_dev);

	unsigned get_seed() const;
};

std::ostream& operator<<(std::ostream& os, const CRandomGenerator& rg);

template <typename T>
T random(T min, T max) {
	extern CRandomGenerator* globalRandomGenerator;
	return globalRandomGenerator->random(min, max);
}

static inline bool tossCoin(float bias) {
	extern CRandomGenerator* globalRandomGenerator;
	return globalRandomGenerator->tossCoin(bias);
}

static inline bool tossCoin() {
	extern CRandomGenerator* globalRandomGenerator;
	return globalRandomGenerator->tossCoin();
}

namespace impl {
	/* NOTE: this distribution was tested on :
	 * - mt19937
	 * - lcg
	 * It is almost perfectly uniform +- 0.5% but is 7x times faster than std::uniform_...
	 * Example of tests: https://godbolt.org/z/Kq7xsEdzo
	 */
	template <typename T>
	class fast_bounded_distribution {
		using span_t = std::conditional_t<std::is_floating_point_v<T>, T, std::size_t>;
		T min, max;
		span_t span;

		public:
		// NOTE: to reproduce std::uniform, we need [min, max[ to become [min, max], hence the + 1
		//constexpr fast_bounded_distribution(T min_, T max_) : min(min_), max(max_+1), span(max - min) {
		// probably std::nextafter is better in order to convert from [min, max) to [min,max]
		constexpr fast_bounded_distribution(T min_, T max_) : min(min_), max(std::nextafter(max_, max_+1)), span(max - min) {
		        assert(!mft::impl::is_any_nan(min, max));
			assert(min < max && "[a; b[ with a < b required");
		}

		template <typename Gen, typename V = T>
		constexpr std::enable_if_t<std::is_floating_point_v<V>, V> operator()(Gen&& gen) const {
			return ((static_cast<V>(gen() - gen.min()) / static_cast<V>(gen.max())) * span) + min;
		}

		template <typename Gen, typename V = T>
		constexpr auto operator()(Gen&& gen) const -> std::enable_if_t<std::is_integral_v<V> && std::is_unsigned_v<decltype(gen())>, V> {
			return static_cast<V>(span_t{gen() - gen.min()} % span) + min;
		}
	};
}

#endif /* CRANDOMGENERATOR_H_ */
