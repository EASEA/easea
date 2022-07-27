/***********************************************************************
| heighbors.h		                                                |
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

#include <vector>
#include <cassert>
#include <numeric>
#include <shared/CMatrix.h>


namespace easea
{
namespace shared
{
namespace function
{
 /*
  * \brief shared utility functions : 
  *
  */

template <typename TType, typename TIter>
CMatrix<TType> calcAdjacencyMatrix(TIter begin, TIter end)
{
        const size_t count = std::distance(begin, end);
        CMatrix<TType> adMatrix(count, count);
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < count; ++j)
                adMatrix[i][j] = 0;

        size_t i = 0;

        for (TIter point1 = begin; point1 != end; ++point1)
        {
                size_t j = 0;
        
                for (TIter point2 = ++TIter(point1); point2 != end; ++point2)
                {
                        if (j != i){
                    	    const std::vector<TType> &tmp_point1 = (*point1);
                    	    const std::vector<TType> &tmp_point2 = (*point2);
                    	    adMatrix[i][j] = sqrt(std::inner_product(tmp_point1.begin(), tmp_point1.end(), tmp_point2.begin(), (TType)0, std::plus<TType>()
                                , [](TType x, TType y)->TType{TType t = x - y;return t * t;}));
                        }
                        ++j;
                }
		++i;
        }
        return adMatrix;
                  


}

template <typename TType>
std::vector<std::vector<size_t> > initNeighbors(const CMatrix<TType> &adjacencyMatrix, const size_t nNeighbors)
{
	//assert(adjacencyMatrix.size1() == adjacencyMatrix.size2()); // what are these functions ?
        std::vector<std::vector<size_t> > neighbors(adjacencyMatrix.Rows());
        for (size_t i = 0; i < neighbors.size(); ++i)
        {
                typedef std::pair<TType, size_t> Adjacency_t;
                std::vector<Adjacency_t> adjacency(adjacencyMatrix.Cols());
                std::vector<const Adjacency_t *> adjacency_(adjacencyMatrix.Cols());
                for (size_t j = 0; j < adjacencyMatrix.Cols(); ++j)
                {
		        adjacency[j].first = adjacencyMatrix[i][j];
                        adjacency[j].second = j;
                        adjacency_[j] = &adjacency[j];
                }
                std::partial_sort(adjacency_.begin(), adjacency_.begin() + nNeighbors, adjacency_.end()
                        , [](const Adjacency_t *adjacency1, const Adjacency_t *adjacency2)->bool{return adjacency1->first < adjacency2->first;}
                );
                std::vector<size_t> &neighbors_ = neighbors[i];
                neighbors_.resize(nNeighbors);
                for (size_t j = 0; j < nNeighbors; ++j)
                        neighbors_[j] = adjacency_[j]->second;
                assert(std::find(neighbors_.begin(), neighbors_.end(), i) != neighbors_.end());
        }
        return neighbors;
}
template <typename TType>
std::vector<TType> computeDirection(const std::vector<TType> &obj, const std::vector<TType> &ideal)
{
        assert(obj.size() == ideal.size());

        std::vector<TType> dir(obj.size());

        for (size_t i = 0; i < dir.size(); ++i)
        {
                dir[i] = obj[i] - ideal[i];
                assert(dir[i] >= 0);
        }
        return dir;
}


}
}
}
