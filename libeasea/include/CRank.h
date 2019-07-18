
/*
 * CRanking .h
 *
 *  Created on: 31 jullet 2018
 *      Author: Anna Ouskova Leonteva
 */
#ifndef CRANK_H
#define CRANK_H

#include <CDominanceComparator.h>
#include <CBT2Selection.h>
#include <CBTSelection.h>
#include <CConstraintComparator.h>
#include <vector>
#include <memory>
#include <algorithm>
#include "CLogger.h"

/* In this class 2 schemes of individal's ranking are implemented.
 * The first one gets a object of population. Then it's individuals (solutions) are
 * ranked according the algorithm NSGAII.
 * The second ranking scheme is corresponded to algorithm ASREA, which gets a object
 * of population and object of archive population.
 * For both cases, as a result a set of fronts with ranks is taken.
 * The first rank number is 0 and this front contains the non-dominated
 * individuals. The next one has a rank number = 1 and etc.
 */

template <typename TIndividual, typename TPopulation>
class CRank {

private:
  TPopulation * population_;
  TPopulation ** ranking_;
  int nbSubfronts_;
  CDominanceComparator<TIndividual> *dominance_;
//  CConstraintComparator<TIndividual> *constraint_;

public:
    CRank (TPopulation * &population) : population_{population},
	dominance_ { new CDominanceComparator<TIndividual>()}{
	//constraint_ { new CConstraintComparator<TIndividual>()}

        // dominating having the number of individuals dominating current individual
    	auto dominating = std::make_unique<int[]>(population_->size());
  	if (dominating == nullptr)
 	    LOG_ERROR(errorCode::memory, "memory wasn't allocated!");
        
	// dominated - contains the list of solutions dominated by current individual
	vector<int> dominated[population_->size()];
  	
	// front contains the list of individuals belonging to the current front 
  	vector<int>  front[population_->size()+1];

  	auto flgDom = int{0};

  	// Non dominated sorting algorithm
  	for (size_t i = 0; i < population_->size(); i++){
    		dominating[i] = 0;
	}
        // For all individuals , calculate if it  dominates 
	
        for (size_t i = 0; i < (population_->size() - 1); i++) {
            for (size_t j = i + 1; j < population_->size(); j++) {
    //          flgDom =constraint_->match(population_->pop_vect[(p)], population_->get(q));
    //          if (flgDom == 0)
                flgDom =dominance_->match(population_->pop_vect[(i)], population_->pop_vect[(j)]);
                if (flgDom == -1) {
                    dominated[i].emplace_back(j);
                    dominating[j]++;
                } else if (flgDom == 1) {
                    dominated[j].emplace_back(i);
		    dominating[i]++;
                }
            }
        }
        for (size_t i = 0; i < population_->size(); i++) {
        // first front
            if (dominating[i] == 0) {
                front[0].emplace_back(i);
                population_->pop_vect[(i)]->setRank(0);
            }
        }
        // Other fronts
        auto i= int{0};
        vector<int>::iterator it1, it2;
        while (front[i].size()!=0) {
            i++;
            for (it1=front[i-1].begin(); it1<front[i-1].end(); it1++) {
                for (it2=dominated[*it1].begin(); it2 < dominated[*it1].end();it2++) {
                    dominating[*it2]--;
                    if (dominating[*it2]==0) {
                        front[i].emplace_back(*it2);
                        population_->pop_vect[(*it2)]->setRank(i);
                    }
                }
            }
        }
        ranking_ = new TPopulation*[i];
        if (ranking_ == nullptr)
            LOG_ERROR(errorCode::memory, "memory wasn't allocated!");
        //0,1,2,....,i-1 are front

        nbSubfronts_ = i;

        for (auto j = 0; j < i; j++) {
            ranking_[j] = new TPopulation(front[j].size());
            for (it1=front[j].begin(); it1<front[j].end(); it1++)
                ranking_[j]->add(new TIndividual(population_->get(*it1)));
        }

    };

    CRank (TPopulation * population, TPopulation * archive) : population_{population} {
        dominance_   = new CDominanceComparator<TIndividual>();
//        constraint_  = new CConstraintComparator<TIndividual>();

        auto dominating = std::make_unique<int[]>(population_->size());
        if (dominating == nullptr)
            LOG_ERROR(errorCode::memory, "memory wasn't allocated!");


        vector<int> front[population_->size()+1];


        auto flgDom = int{0};
        auto selection = new CBTSelection<TIndividual, TPopulation>();

        for (auto i = 0; i < population_->size(); i++)
            dominating[i] = 0;

        auto idx = int{0};
        for (auto i = 0; i < (population_->size() ); i++) {
            for (auto j = 0; j < archive->size(); j++){
                flgDom =dominance_->match(population_->get(i), population_->get(j));
                if (flgDom == 1)
                    dominating[i]= dominating[i]+1;
            }
        }
        // other fronts
        auto i = int{1};
        vector<int>::iterator it1, it2;
        for (auto j = 0; j < (population_->size() ); j++) {
            if (dominating[j] == 0)
                front[0].emplace_back(j);
                population_->get(j)->setRank(1+dominating[j]);
        }
        ranking_ = new TPopulation*[i];
        if (ranking_ == nullptr)
            LOG_ERROR(errorCode::memory, "memory wasn't allocated!");

        nbSubfronts_ = i;

        for (auto j = 0; j < i; j++) {
            ranking_[j] = new TPopulation(front[j].size());
            for (it1=front[j].begin(); it1<front[j].end(); it1++)
                ranking_[j]->add(new TIndividual(population_->get(*it1)));
        }

    };


    ~CRank(){
        for (auto i = 0; i < nbSubfronts_; i++) {
            delete ranking_[i];
        }
        delete [] ranking_;
        delete dominance_;
//        delete constraint_;
    };

    TPopulation * getSubfront(int rank){ return ranking_[rank];};
    int getNumberOfSubfronts(){  return nbSubfronts_; };

};
#endif
