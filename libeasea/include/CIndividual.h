/*
 * CIndividual.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre  
 *	Changed 13.09.2018 by Anna Ouskova Leonteva
 */

#ifndef CINDIVIDUAL_H_
#define CINDIVIDUAL_H_

#include <iostream>

#include "CVariable.h"

#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>

/* Class of individual's (solution) implementation */


class CIndividual
{

public:
    
    int nbVar_;           // Number of variables
    int nbObj_;           // Number of objectives
    int nbConstraints_;   // Number of constraints
    double * upperBound_;     // upper bound of variable
    double * lowerBound_;     // lower bound of variable
    double *objective_;      // objecives
    double * mstep_;
    double * cstep_;
    double constraint_;       // constraints
    int rank_;                // rank value
    double crowdingDistance_; // croawding distance
    CVariable **variable_;    //  variables

//typedef std::unique_ptr<double[]> dblPtr;

public :
    // Functions for multi_objectives algorithmes
    void setObjective(int index, double val);
    double getObjective(const int index);
    void setAllObjectives(double* __restrict out, double* __restrict in);
    void setAllVariables(CVariable ** __restrict out, CVariable ** __restrict in);

    int getRank()const  { return rank_; }
    void setRank(int val){ rank_ = val; }

    double getCrowdingDistance()const { return crowdingDistance_; }
    void setCrowdingDistance(double val){ crowdingDistance_ = val; }

    CVariable ** getIndividualVariables()  { return variable_; }

    inline int getNumberOfObjectives() const { return nbObj_; }
    inline int getNumberOfVariables() const { return nbVar_; }
    inline int getNumberOfConstraints() const { return nbConstraints_; }
    inline void setNumberOfObjectives(int val) { nbObj_ = val; }
    inline void setNumberOfVariables(int val) { nbVar_ = val; }
    inline void setNumberOfConstraints(int val) { nbConstraints_ = val; }


    double getConstraint()  { return constraint_; }
    void setConstraint(double val) {constraint_ = val; }

public:
    bool valid;
    bool isImmigrant;
    float fitness;

public:

    CIndividual();
    CIndividual(CIndividual const& genome) noexcept=default;
    CIndividual& operator=(CIndividual const& genome) noexcept =default;
    CIndividual(CIndividual&&) noexcept =default;
    CIndividual& operator=(CIndividual&&) noexcept =default;
    virtual ~CIndividual() =default;
    virtual bool evaluate_wrapper(bool force_evaluation);
    virtual float evaluate()  =  0;
    virtual void printOn(std::ostream& O) const = 0;
    virtual unsigned mutate(float pMutationPerGene)  = 0;
    virtual CIndividual* crossover(CIndividual** p2)  = 0;
    virtual CIndividual* clone() = 0;

    virtual void boundChecking() = 0;

    static unsigned getCrossoverArity(){ return 2; }
    float getFitness() const { return this->fitness; }

private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, [[maybe_unused]] const unsigned int version) {
	    ar & fitness;
    }
};

std::ostream& operator<<(std::ostream& os, CIndividual const& ind);

BOOST_SERIALIZATION_ASSUME_ABSTRACT(CIndividual)

#endif /* CINDIVIDUAL_H_ */
