/***********************************************************************
| dominance.h 							        |
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

#include <list>
#include <map>
#include <cassert>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <CLogger.h>



namespace easea
{
namespace shared
{
namespace functions
{
/*
 * \brief shared utility functions : to check dominance
 *
 */

template <typename TObjective, typename Comp>
bool isDominated(const std::vector<TObjective> &point1, const std::vector<TObjective> &point2, Comp&& comparator = std::less<TObjective>{})
{
	assert(point1.size() == point2.size() && "individuals have different size");
	return std::equal(point1.cbegin(), point1.cend(), point2.cbegin(),
			[&](auto const& l, auto const& r) {return comparator(l, r) || l == r;}); // for weak dominance
			// std::forward<Comp>(comparator)); // for strong dominance
	// IMPORTANT: weak dominance lambda is optimized to produce the same assembly as a code using std::less_equal or std::more_equal.
	// This was proved here: https://godbolt.org/z/s4KqvjKds
}

namespace impl {

template <typename TInd>
auto const& get_objectives(TInd&& ind) {
	return ind->m_objective;
}

template <typename TInd>
auto get_objective(TInd&& ind, size_t n_obj) {
	assert(n_obj < get_objectives(ind).size());
	return get_objectives(ind)[n_obj];
}

/*
 * \brief Check if individual is nondominated
 *
 * \param[in] 
 */
template <typename TIterator, typename TIndividual, typename Comp>
bool isNondominated(TIterator individual, std::list<TIndividual> &population, std::list<TIndividual> &lstNondominated, Comp&& comparator)
{
        for (TIterator nondominated = lstNondominated.begin(); nondominated != lstNondominated.end(); ++nondominated)
        {
                if (isDominated(get_objectives(*individual), get_objectives(*nondominated), std::forward<Comp>(comparator)))
                {
                        typename std::list<TIndividual>::iterator move = nondominated;
                        population.splice(population.begin(), lstNondominated, move);
                }
                else if (isDominated(get_objectives(*nondominated), get_objectives(*individual), std::forward<Comp>(comparator))) {
                        return false;
		}
        }
        return true;
}

// Complexity for extracting all layers : O(h * n^2)
template <typename TIndividual, typename Comp>
std::list<TIndividual> slow_extract_maxima_nd(std::list<TIndividual> &population, Comp&& comparator, [[maybe_unused]] bool first_layer)
{
        assert(!population.empty() && "Population is empty");
        typedef typename std::list<TIndividual>::iterator TIterator;
        std::list<TIndividual> lstNondominated;

	lstNondominated.splice(lstNondominated.end(), population, population.begin());
        for (TIterator individual = population.begin(); individual != population.end(); ++individual)
        {
                if (lstNondominated.empty()) LOG_ERROR(errorCode::value,"The is no nondominated solutions");
                if (isNondominated(individual, population, lstNondominated, std::forward<Comp>(comparator)))
                {
                        typename std::list<TIndividual>::iterator move = individual;
                        lstNondominated.splice(lstNondominated.begin(), population, move);
                }
        }
        return lstNondominated;
}

// Complexity for extracting all layers : O(h * n log n) with h being the number of layers
// TODO: extraction of all layers in O(n log n) : "Fast Algorithm for Three-Dimensional Layers of Maxima Problem", by Yakov Nekrich
template <typename TInd, typename Comp>
std::list<TInd> fast_extract_maxima_3d(std::list<TInd>& pop, Comp&& comparator, bool first_layer) {
	assert(pop.begin() != pop.end());
	assert(get_objectives(pop.front()).size() == 3 && "This only works for 3D objectives");

	// sort by (x1) if needed
	const auto internal_cmp = [&](auto const& lhs, auto const& rhs) {return comparator(get_objective(lhs, 0), get_objective(rhs, 0));};
	assert(first_layer || std::is_sorted(pop.cbegin(), pop.cend(), internal_cmp));
	if (first_layer)
		pop.sort(internal_cmp);

	// structure with log n insertion, search and remove containing (x2, x3)
	// sorted by x2 in inverse order of comparator
	using obj_t = decltype(get_objective(std::declval<TInd>(), 0));
	const auto lambda_map = [&](const auto& l, const auto& r) { return !comparator(l, r);};
	std::map<obj_t, decltype(pop.begin()), decltype(lambda_map)> set(lambda_map); // TODO : comp

	std::list<TInd> out;
	for (auto it = pop.begin(); it != pop.end();) {
		auto oit = it++;
		const auto x2 = get_objective(*oit, 1);
		// determine best elem j in set such that x2(ej) better than x2(cur)
		auto wj = set.lower_bound(x2); // wj <= vi, wj first j to respect
		if (wj != set.end()) { // found
			const auto x3_cur = get_objective(*oit, 2);

			// vi maxima of Ti-1, if x3(vi) < x3(wk), k = j .. end | log n complexity
			if (std::all_of(wj, set.end(), [&](auto const& pair) {return comparator(x3_cur, get_objective(*pair.second, 2));})) {
				const auto [iit, ok] = set.insert({x2, oit});
				assert(ok && "Error during insertion");
				out.splice(out.begin(), pop, oit);
			}
		} else {
			auto [cij, ok] = set.insert({x2, oit});
			assert(ok && "Error during insertion");
			out.splice(out.begin(), pop, oit);
		}
	}
	return out;
}

// complexity for extracting all layers : O(n log n)
template <typename TInd, typename Comp>
std::list<TInd> fast_extract_maxima_2d(std::list<TInd>& pop, Comp&& comparator, bool first_layer) {
	assert(pop.begin() != pop.end());
	assert(get_objectives(pop.front()).size() == 2 && "This only works for 2D objectives");

	// sort by x1 if needed
	const auto internal_cmp = [&](auto const& lhs, auto const& rhs) {return comparator(get_objective(lhs, 0), get_objective(rhs, 0));};
	assert(first_layer || std::is_sorted(pop.cbegin(), pop.cend(), internal_cmp));
	if (first_layer)
		pop.sort(internal_cmp);

	std::list<TInd> out;
	out.splice(out.begin(), pop, pop.begin());

	auto cur_best_y = get_objective(out.front(), 1);
	for (auto it = pop.begin(); it != pop.end();) {
		auto oit = it++;
		if (comparator(get_objective(*oit, 1), cur_best_y)) {
			out.splice(out.begin(), pop, oit);
			cur_best_y = get_objective(*oit, 1);
		}
	}
	return out;
}

// complexity for extracting all layers : O(n log n)
template <typename TInd, typename Comp>
std::list<TInd> fast_extract_maxima_1d(std::list<TInd>& pop, Comp&& comparator, bool first_layer) {
	// sort by x1 if needed (faster in the long run, O(n*log(n)) vs O(n^2))
	const auto internal_cmp = [&](auto const& lhs, auto const& rhs) {return comparator(get_objective(lhs, 0), get_objective(rhs, 0));};
	assert(first_layer || std::is_sorted(pop.cbegin(), pop.cend(), internal_cmp));
	if (first_layer)
		pop.sort(internal_cmp);

	auto best = pop.begin();
	std::list<TInd> ret;
	ret.splice(ret.begin(), pop, best);
	return ret;
}

}

template <typename TInd, typename Comp>
std::list<TInd> getNondominated(std::list<TInd>& pop, Comp&& comp, bool first_layer) {
	assert(pop.size() > 0);
	switch (impl::get_objectives(pop.front()).size()) {
		case 0:
			assert(false && "Objective size can't be 0");
			exit(1);
		case 1:
			return impl::fast_extract_maxima_1d(pop, std::forward<Comp>(comp), first_layer);
		case 2:
			// TODO: deduce if maximizing or minimizing instead of less
			return impl::fast_extract_maxima_2d(pop, std::forward<Comp>(comp), first_layer);
		case 3:
			// TODO: deduce if maximizing or minimizing instead of less
			return impl::fast_extract_maxima_3d(pop, std::forward<Comp>(comp), first_layer);
		default:
			// TODO: better algorithm
			return impl::slow_extract_maxima_nd(pop, std::forward<Comp>(comp), first_layer);
	}
}

template <typename Ind, typename Comp>
bool isDominated(Ind&& lhs, Ind&& rhs, Comp&& comparator = std::less<decltype(get_objective(std::declval<Ind>(), 0))>{}) {
	return isDominated(lhs.m_objective, rhs.m_objective, std::forward<Comp>(comparator));
}


}
}

}
