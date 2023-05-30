#pragma once

// here to bridge the gap between CEvolutionaryAlgorithm and CAlgorithm

#include <fstream>
#include <functional>
#include <random>

#include "CEvolutionaryAlgorithm.h"
#include "Parameters.h"

namespace impl
{
template <typename TIt>
bool lhs_better(TIt lhs, TIt rhs)
{
	size_t lhs_better = 0;
	for (std::size_t j = 0; j < lhs->m_objective.size(); ++j) {
		if (lhs->m_objective[j] < rhs->m_objective[j])
			lhs_better++;
	}
	auto rhs_better = lhs->m_objective.size() - lhs_better;
	return lhs_better > rhs_better;
}

template <typename TIt, typename Gen, typename BinaryOp = decltype(lhs_better<TIt>)>
TIt dumb_tournament(TIt begin, TIt end, size_t pressure, Gen&& generator, BinaryOp&& cmp = lhs_better<TIt>)
{
	std::uniform_int_distribution<size_t> pdis(0, std::distance(begin, end));
	auto best = begin + pdis(generator);
	for (size_t i = 0; i < pressure; ++i) {
		auto cand = begin + pdis(generator);
		if (cmp(cand, best))
			best = cand;
	}
	return best;
}
} // namespace impl

template <typename MOAlgorithm>
class CAlgorithmWrapper : public CEvolutionaryAlgorithm
{
    public:
	CAlgorithmWrapper(Parameters* params, MOAlgorithm& ralgo) : CEvolutionaryAlgorithm(params), m_algorithm(ralgo)
	{
		initializeParentPopulation();
	}
	virtual ~CAlgorithmWrapper() = default;

	void initializeParentPopulation() override
	{
		if (params->startFromFile) {
			std::ifstream ifs(params->inputFilename);
			ifs >> *m_algorithm;
		}
	}

	void savePopulation(std::string const& dst) const
	{
		std::ofstream ofs(dst);
		ofs << *m_algorithm;
	}

	void network_send()
	{
		if (!(params->remoteIslandModel && numberOfClients > 0))
			return;

		static std::uniform_real_distribution<float> dis(0., 1.);
		auto p = dis(m_algorithm->getRandom());
		if (p > params->migrationProbability)
			return;

		const auto& pop = m_algorithm->getPopulation();
		// TODO: as CLI parameter
		const size_t pressure = 100; // 99th percentile
		// NOTE: tournament
		const auto& best = *impl::dumb_tournament(pop.cbegin(), pop.cend(), pressure, m_algorithm->getRandom());

		std::uniform_int_distribution<size_t> cdis(0, Clients.size());
		auto cidx = cdis(m_algorithm->getRandom());
		auto& cli = Clients[cidx];
		auto now = std::chrono::system_clock::now();
		auto in_time_t = std::chrono::system_clock::to_time_t(now);
		if (cli->getClientName().size() > 0) {
			std::cout << "[" << std::put_time(std::localtime(&in_time_t), "%H:%M:%S") << "]"
				  << " Sending my best individual to " << cli->getClientName() << std::endl;
		} else {
			std::cout << "[" << std::put_time(std::localtime(&in_time_t), "%H:%M:%S") << "]"
				  << " Sending my best individual to " << cli->getIP() << ":" << cli->getPort()
				  << std::endl;
		}
		Clients[cidx]->send(best);
	}

	void network_receive()
	{
		if (!params->remoteIslandModel)
			return;

		auto& pop = m_algorithm->getPopulation();
		using base_t = std::remove_cv_t<std::remove_reference_t<decltype(pop[0])>>;
		while (server->has_data()) {
			auto cmoind = server->consume<base_t>();

			// anti tournament
			// TODO: as CLI parameter
			static constexpr size_t apressure = 10;
			auto worst =
				impl::dumb_tournament(pop.begin(), pop.end(), apressure, m_algorithm->getRandom(),
						      [](auto itl, auto itr) { return impl::lhs_better(itr, itl); });
			*worst = std::move(cmoind);
		}
	}

	void network_tasks()
	{
		network_send();
		network_receive();
	}

	void runEvolutionaryLoop() override
	{
		initializeParentPopulation();

		const auto max_gen = *params->generationalCriterion->getGenerationalLimit();
		const auto max_time_s = this->params->timeCriterion->getLimit();
		if (params->printStats) {
			m_algorithm->print_header(std::cout, max_gen, max_time_s);
			m_algorithm->print_stats(std::cout);
		}

		// TODO: --printInitialPopulation --printFinalPopulation --alwaysEvaluate --optimise --elitism --elite
		while (!this->allCriteria()) {
			EASEABeginningGenerationFunction(this);

			m_algorithm->run();
			m_algorithm->print_stats(std::cout);

			network_tasks();

			EASEAGenerationFunctionBeforeReplacement(this);
			EASEAEndGenerationFunction(this);
			currentGeneration++;
		}
		m_algorithm->print_footer(std::cout);
	}

    private:
	MOAlgorithm m_algorithm;
};
