/***********************************************************************
| CbetaIndividual.h 							|
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA		|
| (EAsy Specification of Evolutionary Algorithms) 			|
| https://github.com/EASEA/                                 		|
|    									|	
| Copyright (c)      							|
| ICUBE Strasbourg		                           		|
| Date: 2019-05                                                         |
|                                                                       |
 ***********************************************************************/


#pragma once

#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <shared/CRandom.h>
#include <shared/CProbability.h>
#include <shared/CBoundary.h>
#include <shared/functions/helper.h>
#include <operators/crossover/base/C2x2Crossover.h>
#include <operators/crossover/base/CCrossover.h>

namespace easea
{
namespace operators
{
namespace crossover
{
namespace continuous
{
namespace beta
{
template <typename TType, typename TRandom>
class CbetaCrossover : public C2x2Crossover<TType, std::vector<TType> >, public easea::shared::CRandom<TRandom>, public easea::shared::CProbability<TType>, public easea::shared::CBoundary<TType>
{
public:
        typedef TType TT;
        typedef TRandom TR;
        typedef std::vector<TT> TVariable;
        typedef CCrossover<TT, TVariable> TBase;
        typedef typename TBase::TI TI;
        typedef typename easea::shared::CBoundary<TT>::TRange TRange;
        typedef typename easea::shared::CBoundary<TT>::TBoundary TBoundary;

        CbetaCrossover(TRandom random, const TT probability, const TBoundary &boundary, const TT distributionIndex, const TT componentProbability = 0.5);
        ~CbetaCrossover(void);
        TT getProbability(void) const;

protected:

	void boundedCrossover(TR &random, const TT parent1, const TT parent2, TT &offspring1, TT &offspring2, const TT lower, const TT upper);
	void kfCrossover(TR &random, const TT parent1, const TT parent2, TT &offspring1, TT &offspring2, const TT lower, const TT upper, const size_t ind);

TT getGaussian(TRandom &random);

        void runCrossover(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2);
        void launch(const TVariable &parent1, const TVariable &parent2, const TVariable step1, const TVariable step2, TVariable &offspring1, TVariable &offspring2);
//	size_t getLimitGen(){ return TBase::getLimitGen();};
private:
        std::uniform_real_distribution<TT> m_distribution;
//        TT m_distributionributionIndex;
        TT m_probability;
	TT m_adapt;
	TT k1;
	TT k;
	TT m_local;
};



template <typename TType, typename TRandom>
CbetaCrossover<TType, TRandom>::CbetaCrossover(TRandom random, const TT probability, const TBoundary &boundary, const TT distributionIndex, const TT componentProbability)
        : easea::shared::CRandom<TRandom>(random), easea::shared::CProbability<TT>(probability), easea::shared::CBoundary<TT>(boundary)
        , m_distribution(0, 1) /*, m_distributionributionIndex(distributionIndex)*/, m_probability(componentProbability)
{
        assert(m_nbGen <= 0);
	assert(m_iGen < 0);
        assert(0 <= componentProbability && componentProbability <= 1);
	k1 =0;

}

template <typename TType, typename TRandom>
CbetaCrossover<TType, TRandom>::~CbetaCrossover(void)
{
}
/*std::vector<double>*/double getNormals(/*int n*/) {
//    std::vector<double> X(n);
    std::mt19937 engine(8);
    std::normal_distribution<> normal(0.0, 1.0);
/*    for (int i=0; i<n; i++) {
        X[i] = normal(engine);
    }*/
    return normal(engine);//X;
}


template <typename TType, typename TRandom> 
void CbetaCrossover<TType, TRandom>::boundedCrossover(TRandom &random, const TT parent1, const TT parent2, TT &offspring1, TT &offspring2,
 const TT lower, const TT upper)
{
static std::uniform_real_distribution<TT> dist(0, 1);
	k = upper*(1-m_local);//1-m_local);
	if (k < 1) k = 1;
double p1 = parent1;
double p2 = parent2;
bool t =false;
/*if (fabs(parent1-parent2)<0.001){
    p1 = p1 + 0.00001*(dist(random));
    p2 = p2 + 0.00001*dist(random);
//    m_local = m_local*1000;
//	t = true;
}*/
/*if (p1 == p2)
{
    t=true;
}*/
	TT range = k*fabs(p1 - p2);
	TT min01 = 0;
	TT max01 = 0;
	if (p1 <= p2)
	{
		min01 = p1 - range * (1- m_local);
		max01 = p2 + range * ( m_local);
	}
	else
	{
		min01 = p2 - range * ( m_local);
		max01 = p1 + range * (1- m_local);
	}
//if ((parent1==parent2))
//	    printf("HERE: %f %f %f\n", parent1, parent2, m_adapt);

	offspring1 = (min01 + ((dist(random)) * (max01 - min01)));
	offspring2 = (min01 + ((dist(random)) * (max01 - min01)));
	double o1 = offspring1;
	double o2 = offspring2;
	offspring1 = (  1-m_adapt) * offspring1 + ( m_adapt) * parent1;
	offspring2 = (  1-m_adapt) * offspring2 + ( m_adapt)* parent2;
/*
if (this->getCurrentGen()>3000)
{
//printf("LOCAL: %f\n", m_local);
printf("p1 : %f p2 : %f r : %f min : %f max : %f o1 : %f o2 : %f o11: %f o22 : %f loc: %f\n",
parent1, parent2, range, min01, max01, o1, o2, offspring1, offspring2, m_local);
}*/	
	//m_adapt = 
        assert(lower < upper);
//printf("O: %f %f\n", offspring1, offspring2);
//printf("ADA: %f\n",m_adapt);
        offspring1 = easea::shared::functions::helper::checkBoundary(offspring1, lower, upper);
        offspring2 = easea::shared::functions::helper::checkBoundary(offspring2, lower, upper);
        if (dist(random) < 0.09)
                std::swap(offspring1, offspring2);
}



template <typename TType, typename TRandom>
TType CbetaCrossover<TType, TRandom>::getProbability(void) const
{
        return m_probability;
}

template <typename TType, typename TRandom>
void CbetaCrossover<TType, TRandom>::runCrossover(const TI &parent1, const TI &parent2, TI &offspring1, TI &offspring2)
{

	m_adapt = 0.99* powf(1.088, powf(this->getCurrentGen()/(TT)this->getLimitGen(),2));

	k1 = /*0.99*/ 0.99*(this->getLimitGen()-this->getCurrentGen());
//m_adapt = 0.99 * powf(0.01/0.99, powf(this->getCurrentGen()/(TT)this->getLimitGen(),2));
//m_local = 0.0001 * powf(0.99/0.0001, (this->getCurrentGen()/(TT)this->getLimitGen()));
//	m_local = /*1/(double)this->getBoundary().size() */ 0.005*this->getBoundary().size() * powf(0.99*k, (this->getCurrentGen()/(TT)this->getLimitGen()));
/*std::default_random_engine generator;
double u =  std::accumulate(std::begin(parent1.m_objective), std::end(parent1.m_objective), 0.0) / (parent1.m_variable.size());
std::normal_distribution<double> dist(u, 15.);//5);
double n;
n =dist(generator);
m_local = n ;//powf(n, this->getCurrentGen()/(TT)this->getLimitGen() );
*/

double kk = 24613;//100;//300; //24613;//42613;
double sigma = 400;//50;//40;//;400;
double mu = 4000;//250;//400;//4000;

/*if (parent1.m_objective[0] == parent2.m_objective[0])
if (parent1.m_objective[1] == parent2.m_objective[1])
sigma = sigma*2;*/
	m_local = kk/(sigma*sqrt(2*3.14))*exp(-1*powf((this->getCurrentGen()-mu),2)/(2*powf(sigma,2)));
//m_local=(k1*0.00001*powf(1.088/0.0001, (this->getCurrentGen()/((TT)this->getLimitGen()))));
        launch(parent1.m_variable, parent2.m_variable, parent1.m_mutStep, parent2.m_mutStep, offspring1.m_variable, offspring2.m_variable);
//	m_local = 0.0001 * powf(0.99/0.0001, (this->getCurrentGen()/(TT)this->getLimitGen()));


        offspring1.m_mutStep.resize(parent1.m_mutStep.size());
        offspring2.m_mutStep.resize(parent2.m_mutStep.size());

        for (size_t i = 0; i < this->getBoundary().size(); ++i)
        {
    //            offspring1.m_variable[i] = parent1.m_variable[i];
//offspring2.m_variable[i] = parent2.m_variable[i];
		offspring1.m_mutStep[i] = (parent1.m_mutStep[i]);//+parent2.m_mutStep[i])/2.;
                offspring2.m_mutStep[i] = (parent1.m_mutStep[i]);//+parent1.m_mutStep[i])/2.;;
        }
}

        
template <typename TType, typename TRandom>
void CbetaCrossover<TType, TRandom>::launch(const TVariable &parent1, const TVariable &parent2, const TVariable step1, const TVariable step2, TVariable &offspring1, TVariable &offspring2)
{
        assert(!this->getBoundary().empty());
        assert(parent1.size() == this->getBoundary().size());
        assert(parent2.size() == this->getBoundary().size());
	
/*	m_CRmax = 1000/(double)this->getBoundary().size();
	m_CRmin = 1./(double)this->getBoundary().size();
	m_CR = (m_CRmin* powf(m_CRmin/m_CRmax, powf(this->getCurrentGen()/(TT)this->getLimitGen(),2)));
*/
/*	m_adapt = 0.99 * powf(0.01/0.99, powf(this->getCurrentGen()/(TT)this->getLimitGen(),2));
	m_local = 0.0001 * powf(0.99/0.0001, (this->getCurrentGen()/(TT)this->getLimitGen()));
        */
	

//	if (m_distribution(this->getRandom()) < this->getProbability())
//        {
                offspring1.resize(parent1.size());
                offspring2.resize(parent2.size());
                for (size_t i = 0; i < this->getBoundary().size(); ++i)
                {
                        const TRange &range = this->getBoundary()[i];
    		//	kfCrossover(this->getRandom(),  parent1[i], parent2[i], offspring1[i], offspring2[i], range.first, range.second, i);
		//	m_local = m_local+sqrt((step1[i] + step2[i])/2.);

                     boundedCrossover(this->getRandom(),  parent1[i], parent2[i], offspring1[i], offspring2[i], range.first, range.second);
                }
//        }
/*        else
        {
                offspring1 = parent1;
                offspring2 = parent2;
    //    }*/
}
}
}
}
}
}
