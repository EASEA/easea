/***********************************************************************
| Cnsga-iii.h                                                           |
|                                                                       |
| This file is part of Artificial Evolution plateform EASEA             |
| (EAsy Specification of Evolutionary Algorithms)                       |
| https://github.com/EASEA/                                             |
|                                                                       |
| Copyright (c)                                                         |
| ICUBE Strasbourg                                                      |
| Date: 2019-05                                                         |
|                                                                       |
 ***********************************************************************/
#pragma once


#include <cassert>
#include <vector>
#include <list>


#include <algorithms/moea/CmoeaAlgorithm.h>
#include <shared/CRandom.h>
#include <shared/CMatrix.h>
#include <shared/functions/breeding.h>
#include <operators/crossover/wrapper/CWrapCrossover.h>
#include <operators/mutation/wrapper/CWrapMutation.h>
#include <shared/functions/dominance.h>
#include <operators/selection/nondominateSelection.h>
#include <operators/selection/randomSelection.h>
#include <config.h>



namespace easea
{
namespace algorithms
{
namespace nsga_iii
{
template <typename TIndividual, typename TRandom>
class Cnsga_iii : public CmoeaAlgorithm<std::vector<TIndividual>, TRandom>, public easea::operators::crossover::CWrapCrossover<typename TIndividual::TO,typename TIndividual::TV>, public easea::operators::mutation::CWrapMutation<typename TIndividual::TO,typename  TIndividual::TV>
{
public:
  typedef TIndividual TI;
  typedef typename TI::TO TO;
  typedef typename TI::TV TV;
  
  typedef std::vector<TI> TPopulation;
  typedef CmoeaAlgorithm<TPopulation, TRandom> TBase;
  typedef typename TBase::TP TP;
  typedef typename easea::operators::crossover::CWrapCrossover<TO, TV>::TC TC;
  typedef typename easea::operators::mutation::CWrapMutation<TO, TV>::TM TM;
  typedef std::vector<TO> TPoint;
  typedef std::pair<size_t, std::list<const TI *> > TNiche;
  
  Cnsga_iii(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation, const std::vector<TPoint> &m_referenceSet, const TO epsilon = 1e-6);
  ~Cnsga_iii(void);
  const std::vector<TPoint> &getReferenceSet(void) const;
  double getEpsilon(void) const;
  TPopulation runBreeding(const TPopulation &parent);
  static bool isDominated(const TI &individual1, const TI &individual2);
  typename TI::TO Distance(const std::vector<typename TI::TO> &referencePoint,const std::vector<typename TI::TO> &objective) const;

  template <typename TIter> std::vector<typename TIndividual::TO> getIdealPoint(TIter begin, TIter end);
  template <typename TIter> void shifting(const std::vector<typename TIndividual::TO> &idealPoint, TIter begin, TIter end);
  typename TIndividual::TO ASF(const size_t axis, const typename TIndividual::TO epsilon, const std::vector<typename TIndividual::TO> &objective);
  template <typename TIter> TIter LocateExtremeIndividual(const size_t axis, const typename TIndividual::TO epsilon, TIter begin, TIter end);
  template <typename TIter> const CMatrix<typename TIndividual::TO> getExtremePoints(TIter begin, TIter end, const typename TIndividual::TO epsilon, const std::vector<typename TIndividual::TO> &idealPoint);
  std::vector<typename TIndividual::TO> getIntercept(CMatrix<typename TIndividual::TO> &extremePoints);
  template <typename TIter> void normalizePopulation(const std::vector<typename TIndividual::TO> &idealPoint, const std::vector<typename TIndividual::TO> &intercepts, TIter begin, TIter end);
  template <typename TIter> void normalize(TIter begin, TIter end, const typename TIndividual::TO epsilon);
  void on_individuals_received() override;
          
protected:
  std::list<TI *> m_noncritical;
  
  static bool Dominate(const TI *individual1, const TI *individual2);
  void makeOneGeneration(void) override;
  void initialize() override;
  template <typename TPtr, typename TIter> TIter selectNoncrit(std::list<TPtr> &front, TIter begin, TIter end);
  template <typename TPtr, typename TIter> TIter selectCrit(std::list<TPtr> &front, TIter begin, TIter end);
  size_t nnReferencePoint(TIndividual &individual) const;
  template <typename TIter> std::vector<std::list<const TIndividual *> > getListIndividuals(TIter begin, TIter end) const;
  template <typename TIter> TIter nicher(std::list<TNiche> &niches, TIter begin, TIter end);
  template <typename TIter> TIter getSmallestNich(TIter begin, TIter end);
  static const TI * comparer(const std::vector<const TI *> &comparator);
  
private:
  std::vector<TPoint> m_referenceSet;
  const TO m_epsilon;
};

template <typename TIndividual, typename TRandom>
Cnsga_iii<TIndividual, TRandom>::Cnsga_iii(TRandom random, TP &problem, const std::vector<TV> &initial, TC &crossover, TM &mutation, const std::vector<TPoint> &referenceSet, const TO epsilon)
  : TBase(random, problem, initial)
  , easea::operators::crossover::CWrapCrossover<TO, TV>(crossover)
  , easea::operators::mutation::CWrapMutation<TO, TV>(mutation), m_referenceSet(referenceSet), m_epsilon(epsilon)
  {
  }

template <typename TIndividual, typename TRandom>
void Cnsga_iii<TIndividual, TRandom>::initialize() {
	TBase::initialize();
	//NOTE: this was commented when initialize() and deferred evaluation were introduced
/*    TBase::m_population.resize(initial.size());
    for (size_t i = 0; i < initial.size(); ++i)
    {
      TIndividual &individual = TBase::m_population[i];
      individual.m_variable= initial[i];
      individual.m_mutStep.resize(individual.m_variable.size());
      for(size_t j = 0; j < individual.m_variable.size(); j++)
            TBase::m_population[i].m_mutStep[j] = 1.;

      TBase::getProblem()(individual);
    }*/
}

template <typename TIndividual, typename TRandom>
void Cnsga_iii<TIndividual, TRandom>::on_individuals_received() {
	// required for updating reObjective
	auto& pop = TBase::m_population;
	std::vector<TI *> ppop;
	ppop.reserve(pop.size());
	/*for (size_t i = 0; i < pop.size(); ++i)
		ppop.push_back(&pop[i]);*/
	std::transform(pop.begin(), pop.end(), std::back_inserter(ppop), [](auto& popit) { return &popit;});
	normalize(ppop.begin(), ppop.end(), m_epsilon);
}



template <typename TIndividual, typename TRandom>
Cnsga_iii<TIndividual, TRandom>::~Cnsga_iii(void)
{
}

template <typename TIndividual, typename TRandom>
const std::vector<typename Cnsga_iii<TIndividual, TRandom>::TPoint> &Cnsga_iii<TIndividual, TRandom>::getReferenceSet(void) const
{
  return m_referenceSet;
}

template <typename TIndividual, typename TRandom>
double Cnsga_iii<TIndividual, TRandom>::getEpsilon(void) const
{
  return m_epsilon;
}

template <typename TIndividual, typename TRandom>
typename Cnsga_iii<TIndividual, TRandom>::TPopulation Cnsga_iii<TIndividual, TRandom>::runBreeding(const TPopulation &parent)
{
    this->getCrossover().setLimitGen(this->getLimitGeneration());
    this->getCrossover().setCurrentGen(this->getCurrentGeneration());
    
    TPopulation offspring = easea::shared::functions::runBreeding(parent.size(), parent.begin(), parent.end(), this->getRandom(), &comparer, this->getCrossover());

#ifdef USE_OPENMP
    EASEA_PRAGMA_OMP_PARALLEL
#endif
    for (int i = 0; i < static_cast<int>(offspring.size()); ++i)
    {
	TIndividual &child = offspring[i];
	this->getMutation()(child);
	TBase::getProblem()(child);
    }
    return offspring;
}

template <typename TIndividual, typename TRandom>
bool Cnsga_iii<TIndividual, TRandom>::isDominated(const TIndividual &individual1, const TIndividual &individual2)
{
  return easea::shared::functions::isDominated(individual1.m_objective, individual2.m_objective);
}

template <typename TIndividual, typename TRandom>
bool Cnsga_iii<TIndividual, TRandom>::Dominate(const TIndividual *individual1, const TIndividual *individual2)
{
  return isDominated(*individual1, *individual2);
}

template <typename TIndividual, typename TRandom>
void Cnsga_iii<TIndividual, TRandom>::makeOneGeneration(void)
{
  TPopulation parent = TBase::m_population;
  TPopulation offspring = runBreeding(parent);
  typedef typename TPopulation::pointer TPtr;
  std::list<TPtr> unionPop;
  for (size_t i = 0; i < parent.size(); ++i)
    unionPop.push_back(&parent[i]);
  for (size_t i = 0; i < offspring.size(); ++i)
    unionPop.push_back(&offspring[i]);
  m_noncritical.clear();
  typedef typename TPopulation::iterator TIter;
  easea::operators::selection::nondominateSelection(unionPop, TBase::m_population.begin(), TBase::m_population.end(), &Dominate
                                                      , [this](std::list<TPtr> &front, TIter begin, TIter end)->TIter{return this->selectNoncrit(front, begin, end);}
                                                      , [this](std::list<TPtr> &front, TIter begin, TIter end)->TIter{return this->selectCrit(front, begin, end);}
  );
}

template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Cnsga_iii<TIndividual, TRandom>::selectNoncrit(std::list<TPtr> &front, TIter begin, TIter end)
{
  /* Note: end was unused before, this could create an overflow */
  TIter dest = begin;
  for (auto i = front.begin(); i != front.end() && dest != end; ++i, ++dest) // NOTE: && dest != end added to prevent OOB
    *dest = **i;
  m_noncritical.splice(m_noncritical.end(), front, front.begin(), front.end());
  return dest;
}

template <typename TIndividual, typename TRandom>
template <typename TPtr, typename TIter> TIter Cnsga_iii<TIndividual, TRandom>::selectCrit(std::list<TPtr> &front, TIter begin, TIter end)
{
  assert(static_cast<long>(front.size()) >= std::distance(begin, end));
  if (static_cast<long>(front.size()) == std::distance(begin, end))
    return selectNoncrit(front, begin, end);
  {
      std::list<TPtr> population(m_noncritical.begin(), m_noncritical.end());
      population.insert(population.end(), front.begin(), front.end());
      normalize<>(population.begin(), population.end(), m_epsilon);
  }
  auto association1 = getListIndividuals(m_noncritical.begin(), m_noncritical.end());
  assert(association1.size() == m_referenceSet.size());
  auto association2 = getListIndividuals(front.begin(), front.end());
  assert(association2.size() == m_referenceSet.size());
  std::list<TNiche> niches;
  for (size_t i = 0; i < m_referenceSet.size(); ++i)
  {
    auto &individuals = association2[i];
    if (!individuals.empty())
    {
      niches.push_back(TNiche());
      TNiche &niche = niches.back();
      niche.first = association1[i].size();
      niche.second.splice(niche.second.end(), individuals, individuals.begin(), individuals.end());
    }
  }
  return nicher(niches, begin, end);
}

template <typename TIndividual, typename TRandom>
size_t Cnsga_iii<TIndividual, TRandom>::nnReferencePoint(TI &individual) const
{
  size_t index = 0;
  typename TI::TO minDistance = Distance(m_referenceSet[index], individual.m_trObjective);
  for (size_t i = 1; i < m_referenceSet.size(); ++i)
  {
    const typename TIndividual::TO distance = Distance(m_referenceSet[i], individual.m_trObjective);
    if (distance < minDistance)
    {
      index = i;
      minDistance = distance;
    }
  }
  individual.m_minDistance = minDistance;
  return index;
}

template <typename TIndividual, typename TRandom>
template <typename TIter> std::vector<std::list<const typename Cnsga_iii<TIndividual, TRandom>::TI *> > Cnsga_iii<TIndividual, TRandom>::getListIndividuals(TIter begin, TIter end) const
{
  std::vector<std::list<const TIndividual *> > nearest(m_referenceSet.size());
  for (TIter i = begin; i != end; ++i)
  {
    const size_t index = nnReferencePoint(**i);
    nearest[index].push_back(*i);
  }
  return nearest;
}

template <typename TIndividual, typename TRandom>
template <typename TIter> TIter Cnsga_iii<TIndividual, TRandom>::nicher(std::list<TNiche> &niches, TIter begin, TIter end)
{
  TIter dest = begin;
  while (dest != end)
  {
    auto _niche = getSmallestNich(niches.begin(), niches.end());
    TNiche &niche = *_niche;
    assert(!niche.second.empty());
    auto elite = niche.first ? easea::operators::selection::randomSelection(this->getRandom(), niche.second.begin(), niche.second.end()) : std::min_element(niche.second.begin(), niche.second.end(), [](const TIndividual *individual1, const TIndividual *individual2)->bool{return individual1->m_minDistance < individual2->m_minDistance;});
    *dest = **elite;
    niche.second.erase(elite);
    if (niche.second.empty())
      niches.erase(_niche);
    else
      ++niche.first;
    ++dest;
  }
  return dest;
}

template <typename TIndividual, typename TRandom>
typename TIndividual::TO Cnsga_iii<TIndividual, TRandom>::Distance(const std::vector<typename TI::TO> &referencePoint, const std::vector< typename TI::TO> &objective) const
{
  typename  TI::TO factor = std::inner_product(objective.begin(), objective.end(), referencePoint.begin(), (TO)1) / std::inner_product(objective.begin(), objective.end(), objective.begin(), (TO)1);
  TO sum = 0;
  for (size_t i = 0; i < objective.size(); ++i)
  {
    TO temp = factor * objective[i] - referencePoint[i];
    sum += temp * temp;
  }
  assert(sum >= 0);
  return sqrt(sum);
}


template <typename TIndividual, typename TRandom>
template <typename TIter> TIter Cnsga_iii<TIndividual, TRandom>::getSmallestNich(TIter begin, TIter end)
{
  assert(begin != end);
  const size_t size = std::min_element(begin, end, [](const TNiche &niche1, const TNiche &niche2)->bool{return niche1.first < niche2.first;})->first;
  std::list<TIter> niches;
  for (TIter i = begin; i != end; ++i)
  {
    assert(i->first >= size);
    assert(!i->second.empty());
    if (i->first == size)
      niches.push_back(i);
  }
  assert(!niches.empty());
  return *easea::operators::selection::randomSelection(this->getRandom(), niches.begin(), niches.end());
}

template <typename TIndividual, typename TRandom>
const typename Cnsga_iii<TIndividual, TRandom>::TI *Cnsga_iii<TIndividual, TRandom>::comparer(const std::vector<const TI *> &comparator)
{
  if (isDominated(*comparator[0], *comparator[1]))
    return comparator[0];
  else if (isDominated(*comparator[1], *comparator[0]))
    return comparator[1];
  return comparator[0];
}

template <typename TIndividual, typename TRandom>
template <typename TIter>
std::vector<typename TIndividual::TO> Cnsga_iii<TIndividual, TRandom>::getIdealPoint(TIter begin, TIter end)
{
  typedef typename std::iterator_traits<TIter>::value_type TPtr;
  assert(begin != end);
  std::vector<typename TIndividual::TO> idealPoint((**begin).m_objective.size());
  for (size_t obj = 0; obj < idealPoint.size(); ++obj)
  {
    TPtr extreme = *std::min_element(begin, end, [obj](TPtr individual1, TPtr individual2)->bool{return individual1->m_objective[obj] < individual2->m_objective[obj];});
    idealPoint[obj] = extreme->m_objective[obj];
  }
  return idealPoint;
}

template <typename TIndividual, typename TRandom>
template <typename TIter>
void Cnsga_iii<TIndividual, TRandom>::shifting(const std::vector<typename TIndividual::TO> &idealPoint, TIter begin, TIter end)
{
  for (TIter individual = begin; individual != end; ++individual)
  {
    auto &_individual = **individual;
    _individual.m_trObjective.resize(idealPoint.size());
    for (size_t i = 0; i < idealPoint.size(); ++i)
    {
      assert(idealPoint[i] <= _individual.m_objective[i]);
      _individual.m_trObjective[i] = _individual.m_objective[i] - idealPoint[i];
    }
  }
}

  

template <typename TIndividual, typename TRandom>
typename TIndividual::TO Cnsga_iii<TIndividual, TRandom>::ASF(const size_t axis, const typename TIndividual::TO epsilon, const std::vector<typename TIndividual::TO> &objective)
{
  typename TIndividual::TO asf = axis ? objective[0] / epsilon : objective[0];
  for (size_t i = 1; i < objective.size(); ++i)
  {
    const typename TIndividual::TO value = axis == i ? objective[i] : objective[i] / epsilon;
    if (value > asf)
      asf = value;
  }
  return asf;
}

template <typename TIndividual, typename TRandom>
template <typename TIter>
TIter Cnsga_iii<TIndividual, TRandom>::LocateExtremeIndividual(const size_t axis, const typename TIndividual::TO epsilon, TIter begin, TIter end)
{
  assert(begin != end);
  typename TIndividual::TO minASF = ASF(axis, epsilon, (**begin).m_trObjective);
  TIter extremeIndividual = begin;
  for (TIter i = ++TIter(begin); i != end; ++i)
  {
    const typename TIndividual::TO asf = ASF(axis, epsilon, (**i).m_trObjective);
    if (asf < minASF)
    {
      minASF = asf;
      extremeIndividual = i;
    }
  }
  return extremeIndividual;
}

template <typename TIndividual, typename TRandom>
template <typename TIter>
const CMatrix<typename TIndividual::TO> Cnsga_iii<TIndividual, TRandom>::getExtremePoints(TIter begin, TIter end, const typename TIndividual::TO epsilon, const std::vector<typename TIndividual::TO> &idealPoint)
{
  CMatrix<typename TIndividual::TO> extremePoints(idealPoint.size(), idealPoint.size());
  for (size_t row = 0; row < idealPoint.size(); ++row)
  {
    auto extremeIndividual = LocateExtremeIndividual(row, epsilon, begin, end);
    for (size_t col = 0; col < idealPoint.size(); ++col){
      extremePoints[row][col]= (**extremeIndividual).m_trObjective[col];
      
    }
  }
  return extremePoints;
}
template <typename TIndividual, typename TRandom>
std::vector<typename TIndividual::TO> Cnsga_iii<TIndividual, TRandom>::getIntercept(CMatrix<typename TIndividual::TO> &extremePoints)
{
  try
  {
    auto inverse = extremePoints.Inverse();
    CMatrix<typename TIndividual::TO> temp1(inverse.Rows(),1, 1.);
    CMatrix<typename TIndividual::TO> temp = inverse*temp1;
    std::vector<typename TIndividual::TO> intercepts(temp.Rows());
    for (size_t i = 0; i < intercepts.size(); ++i) {
      /* Old line below was OOB because M = 1 and indices start at 0, not 1. */
      // intercepts[i] = 1 / temp[i][1];
      intercepts[i] = 1. / temp[i][0];
    }
    return intercepts;
  }
  catch (...)
  {
    std::vector<typename TIndividual::TO> intercepts(extremePoints.Rows());
    for (size_t i = 0; i < intercepts.size(); ++i)
      intercepts[i] = extremePoints[i][i];
    return intercepts;
  }
}

template <typename TIndividual, typename TRandom>
template <typename TIter>
void Cnsga_iii<TIndividual, TRandom>::normalizePopulation(const std::vector<typename TIndividual::TO> &idealPoint, const std::vector<typename TIndividual::TO> &intercepts, TIter begin, TIter end)
{
  assert(intercepts.size() == idealPoint.size());
  std::vector<typename TIndividual::TO> _intercepts(intercepts.size());
  for (size_t i = 0; i < intercepts.size(); ++i)
    _intercepts[i] = intercepts[i] - idealPoint[i];
  for (TIter i = begin; i != end; ++i)
  {
    for (size_t j = 0; j < intercepts.size(); ++j)
      (**i).m_trObjective[j] /= _intercepts[j];
  }
}

template <typename TIndividual, typename TRandom>
template <typename TIter>
void Cnsga_iii<TIndividual, TRandom>::normalize(TIter begin, TIter end, const typename TIndividual::TO epsilon)
{
  const std::vector<typename TIndividual::TO> idealPoint = getIdealPoint(begin, end);
  shifting(idealPoint, begin, end);
  auto extremePoints = getExtremePoints(begin, end, epsilon, idealPoint);
  const auto intercepts = getIntercept(extremePoints);
  normalizePopulation(idealPoint, intercepts, begin, end);
}
}
}
}
