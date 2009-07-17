/*
 * CRandomGenerator.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef CRANDOMGENERATOR_H_
#define CRANDOMGENERATOR_H_

class CRandomGenerator {
public:
  CRandomGenerator(unsigned int seed);
  int randInt();
  bool tossCoin();
  bool tossCoin(float bias);
  int randInt(int min, int max);
  int getRandomIntMax(int max);
  float randFloat(float min, float max);
  int random(int min, int max);
  float random(float min, float max);
  double random(double min, double max);
};

#endif /* CRANDOMGENERATOR_H_ */
