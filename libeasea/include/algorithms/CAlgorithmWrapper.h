#pragma once

// here to bridge the gap between CEvolutionaryAlgorithm and CAlgorithm

#include <fstream>
#include <functional>
#include <random>

#include <boost/exception/diagnostic_information.hpp>

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
	std::uniform_int_distribution<size_t> pdis(0, std::distance(begin, end)-1);
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
	CAlgorithmWrapper(Parameters* params, MOAlgorithm& ralgo);
	virtual ~CAlgorithmWrapper() = default;

	void initializeParentPopulation() override;
	void savePopulation(std::string const& dst) const;

	void network_send();
	void network_receive();
	void network_tasks();

	void runEvolutionaryLoop() override;

    private:
	MOAlgorithm m_algorithm;
};

template <typename MOAlgorithm>
CAlgorithmWrapper<MOAlgorithm>::CAlgorithmWrapper(Parameters* params, MOAlgorithm& ralgo)
	: CEvolutionaryAlgorithm(params), m_algorithm(ralgo)
{
	initializeParentPopulation();
}

template <typename MOAlgorithm>
void CAlgorithmWrapper<MOAlgorithm>::initializeParentPopulation()
{
	if (params->startFromFile) {
		std::ifstream ifs(params->inputFilename);
		if (!ifs) {
			std::cerr << "Warning: Could not load population file " << params->inputFilename
				  << "\nYou may provide another one using --inputFile" << std::endl;
			return;
		}
		try {
			ifs >> *m_algorithm;
		} catch (std::exception const& e) {
			std::cerr << "Error: An error occured while loading population from file "
				  << params->inputFilename << " : " << e.what() << std::endl;
		} catch (boost::exception const& e) {
			std::cerr << "Error: An error occured while loading population from file "
				  << params->inputFilename << " : " << boost::diagnostic_information(e) << std::endl;
		}
	}
}

template <typename MOAlgorithm>
void CAlgorithmWrapper<MOAlgorithm>::savePopulation(std::string const& dst) const
{
	try {
		std::ofstream ofs(dst);
		ofs << *m_algorithm;
	} catch (std::exception const& e) {
		std::cerr << "Error: An error occured while saving population to file " << dst << " : " << e.what()
			  << std::endl;
	} catch (boost::exception const& e) {
		std::cerr << "Error: An error occured while saving population to file " << params->inputFilename
			  << " : " << boost::diagnostic_information(e) << std::endl;
	}
}

template <typename MOAlgorithm>
void CAlgorithmWrapper<MOAlgorithm>::network_send()
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

	std::uniform_int_distribution<size_t> cdis(0, Clients.size() - 1);
	auto cidx = cdis(m_algorithm->getRandom());
	Clients[cidx]->send(best);
}

template <typename MOAlgorithm>
void CAlgorithmWrapper<MOAlgorithm>::network_receive()
{
	if (!params->remoteIslandModel)
		return;

	auto& pop = m_algorithm->getPopulation();
	using base_t = std::remove_cv_t<std::remove_reference_t<decltype(pop[0])>>;
	bool received = false;
	while (server->has_data()) {
		auto cmoind = server->consume<base_t>();

		// anti tournament
		// TODO: as CLI parameter
		static constexpr size_t apressure = 10;
		auto worst = impl::dumb_tournament(pop.begin(), pop.end(), apressure, m_algorithm->getRandom(),
						   [](auto itl, auto itr) { return impl::lhs_better(itr, itl); });
		*worst = std::move(cmoind);
		received = true;
	}
	if (received)
		m_algorithm->on_individuals_received();
}

template <typename MOAlgorithm>
void CAlgorithmWrapper<MOAlgorithm>::network_tasks()
{
	network_send();
	network_receive();
}

template <typename MOAlgorithm>
void CAlgorithmWrapper<MOAlgorithm>::runEvolutionaryLoop()
{
	initializeParentPopulation();

	const auto max_gen = *params->generationalCriterion->getGenerationalLimit();
	const auto max_time_s = this->params->timeCriterion->getLimit();
	if (params->printStats) {
		m_algorithm->print_header(std::cout, max_gen, max_time_s);
	}

	m_algorithm->setLimitGeneration(max_gen);

	// TODO: --printInitialPopulation --printFinalPopulation --alwaysEvaluate --optimise --elitism --elite
	while (!this->allCriteria()) {
		EASEABeginningGenerationFunction(this);

		m_algorithm->run();

		if (params->printStats)
			m_algorithm->print_stats(std::cout);

		network_tasks();

		EASEAGenerationFunctionBeforeReplacement(this);
		EASEAEndGenerationFunction(this);
		currentGeneration++;
	}
	m_algorithm->print_footer(std::cout);

	if (params->savePopulation) {
		std::cout << "Saving population to " << params->outputFilename << ".pop\n";
		savePopulation(params->outputFilename + ".pop");
	}
}
