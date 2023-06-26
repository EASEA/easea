/*
 * CRandomGenerator.cpp
 *
 *  Updated on: 26 july 2022 by Léo Chéneau to use modern engine
 *
 */

#include "include/CRandomGenerator.h"

#include <chrono>
#include <limits>
#include <cassert>

CRandomGenerator globGen{};
CRandomGenerator* globalRandomGenerator = nullptr;

CRandomGenerator::CRandomGenerator(unsigned int seed_) : seed(seed_), engine(Xoshiro::splitmix64(seed_)), engine64(Xoshiro::splitmix64(seed_))
{
}

CRandomGenerator::CRandomGenerator() : CRandomGenerator(std::random_device{}())
{
}

std::ostream& operator<<(std::ostream& os, const CRandomGenerator& rg)
{
	os << "s : " << rg.get_seed() << std::endl;
	return os;
}
