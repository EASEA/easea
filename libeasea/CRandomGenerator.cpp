/*
 * CRandomGenerator.cpp
 *
 *  Updated on: 26 july 2022 by Léo Chéneau to use modern engine
 *
 */

#include "include/CRandomGenerator.h"
#include "include/global.h"

#include <chrono>
#include <limits>
#include <cassert>

CRandomGenerator::CRandomGenerator(unsigned int seed_) : seed(seed_), engine(seed_)
{
}

CRandomGenerator::CRandomGenerator() : CRandomGenerator(std::random_device{}())
{
}

int CRandomGenerator::randInt(int min, int max)
{
	//assert(min != max - 1 && "integer range [a; a[ is impossible.");
	if (min != max) {
		impl::fast_bounded_distribution<int> dis(min, max - 1);
		return dis(engine);
	} else {
		return min;
	}
}

int CRandomGenerator::randInt()
{
	return randInt(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
}

int CRandomGenerator::random(int min, int max)
{
	return randInt(min, max);
}

bool CRandomGenerator::tossCoin()
{
	return randInt(0, 2) == 1;
}

float CRandomGenerator::randFloat(float min, float max)
{
	impl::fast_bounded_distribution<float> dis(min, max);
	return dis(engine);
}

bool CRandomGenerator::tossCoin(float bias)
{
	if (randFloat(0.f, 1.f) <= bias)
		return true;
	else
		return false;
}

float CRandomGenerator::random(float min, float max)
{
	return randFloat(min, max);
}

double CRandomGenerator::randDouble(double low, double high)
{
	impl::fast_bounded_distribution<double> dis(low, high);
	return dis(engine);
}

double CRandomGenerator::random(double min, double max)
{
	return randDouble(min, max);
}

int CRandomGenerator::getRandomIntMax(int max)
{
	return randInt(0, max);
}

float CRandomGenerator::random_gauss(float mean, float std_dev)
{
	std::normal_distribution<float> dis(mean, std_dev);
	return dis(engine);
}

unsigned CRandomGenerator::get_seed() const
{
	return seed;
}

std::ostream& operator<<(std::ostream& os, const CRandomGenerator& rg)
{
	os << "s : " << rg.get_seed() << std::endl;
	return os;
}
