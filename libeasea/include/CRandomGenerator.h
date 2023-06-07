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

class CRandomGenerator
{
    private:
	unsigned seed;
	std::mt19937 engine;

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

#endif /* CRANDOMGENERATOR_H_ */
