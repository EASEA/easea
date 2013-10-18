/*
 *    Copyright (C) 2009  Ogier Maitre

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef CRANDOMGENERATOR_H_
#define CRANDOMGENERATOR_H_

#include <iostream>
#include "MersenneTwister.h"
class CRandomGenerator {
 private:
  unsigned seed;
  MTRand* mt_rnd;
public:
  CRandomGenerator(unsigned int seed);
  ~CRandomGenerator();
  int randInt();
  bool tossCoin();
  bool tossCoin(float bias);
  int randInt(int min, int max);
  int getRandomIntMax(int max);
  float randFloat(float min, float max);
  int random(int min, int max);
  float random(float min, float max);
  double random(double min, double max);

  void random_gauss(float min, float max, float* z_0, float* z_1);
  float random_gauss(float mean, float std_dev);

  unsigned get_seed()const {return this->seed;}
  friend std::ostream & operator << (std::ostream & os, const CRandomGenerator& rg);

};

#endif /* CRANDOMGENERATOR_H_ */
