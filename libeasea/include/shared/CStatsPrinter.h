#pragma once

#include <ostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <limits>
#include <iomanip>
#include <cmath>

namespace impl
{
template <typename T>
size_t min_digits(size_t min_nb, T&& value)
{
	return value > 1 ? std::max(min_nb, static_cast<size_t>(std::floor(std::log10(std::forward<T>(value))))) :
			   min_nb;
}
} // namespace impl

template <typename PopulationOwner>
class CStatsPrinter
{
    public:
	std::ostream& print_header(std::ostream& os, size_t max_generation, size_t max_seconds)
	{
		using namespace impl;
		auto const& population = static_cast<PopulationOwner*>(this)->getPopulation();
		//auto const& max_generation = static_cast<PopulationOwner*>(this)->getLimitGeneration(); // not set
		assert(population.size() > 0);
		nb_individuals = population.size();
		nb_vars = population[0].m_variable.size();
		nb_objs = population[0].m_objective.size();
		max_gen_width = min_digits(5, max_generation);
		max_time_width = min_digits(3, max_seconds) + 4;
		max_evaluation_width = min_digits(5, max_generation * nb_individuals);

		os << std::left << std::setw(max_gen_width) << "GEN"
		   << " " << std::right << std::setw(10) << "ELAPSED"
		   << " " << std::setw(max_evaluation_width) << "EVALS"
		   << " ";
		for (std::size_t i = 0; i < nb_objs; ++i) {
			auto obj_str = std::string{ "MEAN_OBJ_" } + std::to_string(i + 1);
			auto var_str = std::string{ "VAR_OBJ_" } + std::to_string(i + 1);
			os << std::right << std::setw(11) << obj_str << " " << std::setw(11) << var_str << " ";
		}
		os << "\n" << std::flush;
		return os;
	}

	std::ostream& print_population_stats(std::ostream& os)
	{
		auto const& population = static_cast<PopulationOwner*>(this)->getPopulation();
		auto const& cur_generation = static_cast<PopulationOwner*>(this)->getCurrentGeneration();
		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
					  std::chrono::system_clock::now() - started_at)
					  .count();

		using var_t = std::remove_cv_t<std::remove_reference_t<decltype(population[0].m_variable[0])>>;
		using obj_t = std::remove_cv_t<std::remove_reference_t<decltype(population[0].m_objective[0])>>;

		// mean + min + max
		std::vector<var_t> mean_vars(nb_vars, 0.f);
		std::vector<obj_t> mean_objs(nb_objs, 0.f);
		std::vector<obj_t> min_objs(nb_objs, std::numeric_limits<obj_t>::max());
		std::vector<obj_t> max_objs(nb_objs, std::numeric_limits<obj_t>::min());

		var_t* const pmv = mean_vars.data();
		obj_t* const pmo = mean_objs.data();
		obj_t* const pmio = min_objs.data();
		obj_t* const pmao = max_objs.data();
		// NOTE: not supported on MSVC
#if defined(_OPENMP) && (_OPENMP >= 201511)
		#pragma omp parallel for reduction(+:pmv[:nb_vars]) reduction(+:pmo[:nb_objs]) reduction(min:pmio[:nb_objs]) reduction(max:pmao[:nb_objs])
#endif
		for (int i = 0; i < nb_individuals; ++i) {
			auto const& ind = population[i];
			for (std::size_t i = 0; i < nb_vars; ++i)
				pmv[i] += ind.m_variable[i];
			for (std::size_t i = 0; i < nb_objs; ++i) {
				const auto oi = ind.m_objective[i];
				pmo[i] += oi;
				if (oi < min_objs[i])
					pmio[i] = oi;
				if (oi > max_objs[i])
					pmao[i] = oi;
			}
		}

		for (auto& v : mean_vars)
			v /= static_cast<float>(nb_individuals);
		for (auto& v : mean_objs)
			v /= static_cast<float>(nb_individuals);

		// variance
		std::vector<var_t> var_vars(nb_vars, 0.f);
		std::vector<obj_t> var_objs(nb_objs, 0.f);

		var_t* const pvv = var_vars.data();
		obj_t* const pvo = var_objs.data();
#if defined(_OPENMP) && (_OPENMP >= 201511)
		#pragma omp parallel for reduction(+:pvv[:nb_vars]) reduction(+:pvo[:nb_objs])
#endif
		for (int i = 0; i < nb_individuals; ++i) {
			auto const& ind = population[i];
			for (std::size_t i = 0; i < nb_vars; ++i)
				pvv[i] += (ind.m_variable[i] - mean_vars[i]) * (ind.m_variable[i] - mean_vars[i]);
			for (std::size_t i = 0; i < nb_objs; ++i)
				pvo[i] +=
					(ind.m_objective[i] - mean_objs[i]) * (ind.m_objective[i] - mean_objs[i]);
		}

		for (auto& v : var_vars)
			v /= static_cast<float>(nb_individuals);
		for (auto& v : var_objs)
			v /= static_cast<float>(nb_individuals);

		os << std::left << std::setw(max_gen_width) << cur_generation << " " << std::right << std::setw(9)
		   << std::setprecision(4) << std::fixed << static_cast<float>(elapsed_ms) / 1e3f << "s"
		   << " " << std::left << std::setw(max_evaluation_width) << nb_individuals * cur_generation << " ";
		for (std::size_t i = 0; i < nb_objs; ++i)
			os << std::right << std::setw(11) << std::scientific << mean_objs[i] << " " << std::setw(11)
			   << std::scientific << var_objs[i] << " ";
		os << "\n" << std::flush;

		return os;
	}

    private:
	std::chrono::system_clock::time_point started_at = std::chrono::system_clock::now();
	size_t max_gen_width;
	size_t max_time_width;
	size_t max_evaluation_width;
	size_t nb_vars;
	size_t nb_objs;
	size_t nb_individuals;
};
