/*
 * CDistance.cpp
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CDISTANCE_H
#define CDISTANCE_H

#include <float.h>
#include <CObjectiveComparator.h>
#include <math.h>
#include <limits>
#include <memory>

/* This class is used for assign the  distance between individuals.
 * This version of libeasea include only crowding distance (CD) for 2 MOAE:
 * NSGAII and ASREA.
 * CD is assigned to all individuals in the populations
 * and is calculated for the individuals with the same rank.
 * It estimates density of individuals which are laid surrounding a particular
 * individual in the population.
 * In fact, CD is used for controlling the diversity of the individuals and
 * more value of it shows a better individual that is laid in a less crowded area.
 * More info about the calculations of CD can be found here:
 * https://pdfs.semanticscholar.org/6866/eb061ee8dc36c584b184f5b94be9ab674e0d.pdf
 */


template <typename TIndividual, typename TPopulation>
class CDistance {

public:
    CDistance(){};

 /* CD is assigned to all individuals (solutions) int the populations */
    void allocCrowdingDistance( TPopulation * population, const int nbObjs){

        int size = population->size();
        // first of all check the size of population
        if (size == 0)
            return;
        if (size == 1) {
            population->pop_vect[0]->crowdingDistance_ = std::numeric_limits<double>::max();
            return;
        }
        if (size == 2) {
            population->pop_vect[0]->crowdingDistance_ = std::numeric_limits<double>::max();
            population->pop_vect[1]->crowdingDistance_ = std::numeric_limits<double>::max();
            return;
        }
        // Make copy of population
        auto front = std::make_unique<TPopulation>(size);
        for (auto i = 0; i < size; ++i){
            front->add(population->pop_vect[i]);
	    front->pop_vect[i]->crowdingDistance_ = 0.0;
	}

        auto distance = double{0.0};

        for (auto i = 0; i<nbObjs; ++i) {
            auto c = std::make_unique<CObjectiveComparator<TIndividual>>(i);
            front->sort(c.get()); // Sorting of population

            auto minObj = front->pop_vect[0]->objective_[i];
            auto maxObj = front->pop_vect[(front->size()-1)]->objective_[i];

            //Assign crowding distance
            front->pop_vect[0]->crowdingDistance_ = std::numeric_limits<double>::max();
            front->pop_vect[size-1]->crowdingDistance_ = std::numeric_limits<double>::max();

            for (auto j = 1; j < size-1; ++j) {
                distance = front->pop_vect[(j+1)]->objective_[i] - front->pop_vect[(j-1)]->objective_[i];
                distance = distance / (maxObj - minObj);
                distance += front->pop_vect[(j)]->crowdingDistance_;
                front->pop_vect[(j)]->crowdingDistance_ = distance;
            }
        }
        front->clear();
    };

};

#endif
