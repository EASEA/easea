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

template <typename TObjective>
bool isDominated(const std::vector<TObjective> &point1, const std::vector<TObjective> &point2)
{
        if (point1.size() != point2.size())	LOG_ERROR(errorCode::value, "individuals have different size");
        bool dominated = false;
        
        for (size_t i = 0; i < point1.size(); ++i)
        {
                if (point2[i] < point1[i])
                        return false;        // non dominated
                if (point1[i] < point2[i])
                        dominated = true;     // dominated, the new one is better
        }
        return dominated;
}

namespace impl {
/*
 * \brief Check if individual is nondominated
 *
 * \param[in] 
 */
template <typename TIterator, typename TIndividual, typename TDominate>
bool isNondominated(TIterator individual, std::list<TIndividual> &population, std::list<TIndividual> &lstNondominated, TDominate dominate)
{
        for (TIterator nondominated = lstNondominated.begin(); nondominated != lstNondominated.end();)
        {
                if (dominate(*individual, *nondominated))
                {
                        typename std::list<TIndividual>::iterator move = nondominated;
                        ++nondominated;
                        population.splice(population.begin(), lstNondominated, move);
                }
                else if (dominate(*nondominated, *individual))
                        return false;
                else
                        ++nondominated;
        }
        return true;
}

template <typename TIndividual, typename TDominate>
std::list<TIndividual> slow_extract_maxima_nd(std::list<TIndividual> &population, TDominate dominate)
{
        typedef typename std::list<TIndividual>::iterator TIterator;
        if (population.empty())		LOG_ERROR(errorCode::value,  "Population is empty");
        std::list<TIndividual> lstNondominated;

	lstNondominated.splice(lstNondominated.end(), population, population.begin());
        for (TIterator individual = population.begin(); individual != population.end();)
        {
                if (lstNondominated.empty()) LOG_ERROR(errorCode::value,"The is no nondominated solutions");
                if (isNondominated(individual, population, lstNondominated, dominate))
                {
                        typename std::list<TIndividual>::iterator move = individual;
                        ++individual;
                        lstNondominated.splice(lstNondominated.begin(), population, move);
                }
                else
                        ++individual;
        }
        return lstNondominated;
}

template <typename TObjective>
bool isEqual(const std::vector<TObjective> &point1, const std::vector<TObjective> &point2)
{
//	if (point1.size() != point2.size())	LOG_ERROR(errorCode::value, "individuals have different size");
	bool equal = true;

	for (size_t i = 0; i < point1.size(); ++i)
	{
	    if (point2[i] != point1[i])
		    equal = false;
	}
	return equal;
}

template <typename TInd>
auto const& get_objectives(TInd&& ind) {
	return ind->m_objective;
}

template <typename TInd>
auto get_objective(TInd&& ind, size_t n_obj) {
	assert(n_obj < get_objectives(ind).size());
	return get_objectives(ind)[n_obj];
}



// NOTE: 3D points ONLY
template <typename TInd, typename Comp>
std::list<TInd> fast_extract_maxima_3d(std::list<TInd>& nonsorted_pop, Comp&& comparator) {
	assert(nonsorted_pop.begin() != nonsorted_pop.end());
	assert(get_objectives(nonsorted_pop.front()).size() == 3 && "This only works for 3D objectives");

	// sort by (x1)
	nonsorted_pop.sort([&](auto const& lhs, auto const& rhs) {return get_objective(lhs, 0) < get_objective(rhs, 0);});

	// structure with log n insertion, search and remove containing (x2, x3)
	// sorted by x2
	using obj_t = decltype(get_objective(std::declval<TInd>(), 0));
	std::map<obj_t, decltype(nonsorted_pop.begin()), std::greater<obj_t>> set; // TODO : comp

	std::list<TInd> out;
	//out.splice(out.begin(), nonsorted_pop, nonsorted_pop.begin());
	//set.insert({out.begin(),)

	//auto cur_best_x2 = get_objective(out.front(), 1); // used to speed-up computation
	//auto cur_best_x3 = get_objective(out.front(), 2);
	for (auto it = nonsorted_pop.begin(); it != nonsorted_pop.end();) {
		auto oit = it++;

		constexpr auto print_list = [](auto const& l, auto cit, auto lb) {
			std::cerr << "DBG: \n";
			int i = 0;
			for (auto it = l.begin(); it != l.end(); ++it, ++i) {
				if (it == cit)
					std::cerr << "=> " << std::setw(3) << i;
				else if (it == lb)
					std::cerr << "LB " << std::setw(3) << i;
				else if (isDominated(get_objectives(*cit->second), get_objectives(*it->second)))
					std::cerr << "iD" << std::setw(4) << i;
				else if (isDominated(get_objectives(*it->second), get_objectives(*cit->second)))
					std::cerr << "D" << std::setw(5) << i;
				else
					std::cerr << std::setw(6) << i;
				std::cerr <<  ": (";
				for (size_t j = 0; j < 3; ++j) {
					if (j > 0)
						std::cerr << "; ";
					std::cerr << std::setw(4) << std::fixed << get_objective(*it->second, j);
				}
				std::cerr << ")\n";
			}
		};


		const auto x2 = get_objective(*oit, 1);
		// determine best elem j in set such that x2(ej) better than x2(cur)
		auto wj = set.lower_bound(x2); // wj <= vi, wj first j to respect
		if (wj != set.end()) { // found
			const auto x3_cur = get_objective(*oit, 2);
			//const auto x3_j = get_objective(*wj->second, 2);

			//if (comparator(x3_cur, x3_j)) { // vi maxima of Ti-1, if x3(vi) < x3(wj)
			if (std::all_of(wj, set.end(), [=](auto const& pair) {return x3_cur < get_objective(*pair.second, 2);})) { // vi maxima of Ti-1, if x3(vi) < x3(wj)
				// remove until better x3 is found
				const auto [iit, ok] = set.insert({x2, oit});
				assert(ok);
				//std::cerr << "DBG: \n";
				std::cerr << "\nx3(vi) < x3(wj)\n";
				print_list(set, iit, wj);
				/*if (iit != set.begin()) {
					for (auto remit = std::prev(iit); remit != set.begin(); remit--) {
						std::cerr << "remit : (" <<
							get_objective(*remit->second, 0) << "; " <<
							get_objective(*remit->second, 1) << "; " <<
							get_objective(*remit->second, 2) << ") & iit : (" <<
							get_objective(*oit, 0) << "; " <<
							get_objective(*oit, 1) << "; " <<
							get_objective(*oit, 2) << ")\n";

						if (get_objective(*remit->second, 2) < x3_cur)
							break;
						std::cerr << "Erasing : (" <<
							get_objective(*remit->second, 0) << "; " <<
							get_objective(*remit->second, 1) << "; " <<
							get_objective(*remit->second, 2) <<
							") because of (" <<
							get_objective(*oit, 0) << "; " <<
							get_objective(*oit, 1) << "; " <<
							get_objective(*oit, 2) << ")\n";

						remit = set.erase(remit); // TODO: 1 erase instead of N
					}
				}*/
			}
		} else {
			//std::cerr << "DBG: \n";
			std::cerr << "\nwj == set.end()\n";
			auto [cij, ok] = set.insert({x2, oit});
			assert(ok);
			print_list(set, cij , wj);
		}
	}

	for (auto const& [_, ind] : set)
		out.splice(out.begin(), nonsorted_pop, ind);

	return out;
}

// NOTE: 2D points ONLY
template <typename TInd, typename Comp>
std::list<TInd> fast_extract_maxima_2d(std::list<TInd>& nonsorted_pop, Comp&& comparator) {
	assert(nonsorted_pop.begin() != nonsorted_pop.end());
	assert(get_objectives(nonsorted_pop.front()).size() == 2 && "This only works for 2D objectives");

	nonsorted_pop.sort([&](auto const& lhs, auto const& rhs) {return comparator(get_objective(lhs, 0), get_objective(rhs, 0));});
	std::list<TInd> out;
	out.splice(out.begin(), nonsorted_pop, nonsorted_pop.begin());

	auto cur_best_y = get_objective(out.front(), 1);
	for (auto it = nonsorted_pop.begin(); it != nonsorted_pop.end();) {
		auto oit = it++;
		if (comparator(get_objective(*oit, 1), cur_best_y)) {
			out.splice(out.begin(), nonsorted_pop, oit);
			cur_best_y = get_objective(*oit, 1);
		}
	}
	return out;
}

template <typename TInd, typename Comp>
std::list<TInd> fast_extract_maxima_1d(std::list<TInd>& pop, Comp&& comparator) {
	auto best = std::min_element(pop.begin(), pop.end(), comparator);
	std::list<TInd> ret;
	ret.splice(ret.begin(), pop, best);
	return ret;
}

}

template <typename TInd, typename TDom>
std::list<TInd> getNondominated(std::list<TInd>& pop, TDom&& dom) {
	assert(pop.size() > 0);
	switch (impl::get_objectives(pop.front()).size()) {
		case 0:
			assert(false && "Objective size can't be 0");
			exit(1);
		case 1:
			return impl::fast_extract_maxima_1d(pop, dom);
		case 2:
			// TODO: deduce if maximizing or minimizing instead of less
			return impl::fast_extract_maxima_2d(pop, std::less<std::decay_t<decltype(impl::get_objectives(pop.front())[0])>>{});
		case 3:
			// TODO: deduce if maximizing or minimizing instead of less
			return impl::fast_extract_maxima_3d(pop, std::less<std::decay_t<decltype(impl::get_objectives(pop.front())[0])>>{});
		default:
			return impl::slow_extract_maxima_nd(pop, dom);
	}
}



}
}

}
