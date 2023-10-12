#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

template <typename Base>
class CSerializable
{
    private:
	friend class boost::serialization::access;

	template <typename Archive, typename B = Base>
	auto serialize_impl(Archive& ar, const unsigned version) -> decltype(std::declval<B>().getPopulation(), void())
	{
		(void)version;
		decltype(auto) pop_owner = static_cast<Base*>(this);
		decltype(auto) population = pop_owner->getPopulation();
		ar & population;
	}

	template <typename Archive, typename  B = Base>
	auto serialize_impl(Archive& ar, const unsigned version) -> decltype(std::declval<B>().m_variable, void())
	{
		(void)version;
		auto* pop_owner = static_cast<Base*>(this);
		auto& vars = pop_owner->m_variable;
		auto& objs = pop_owner->m_objective;
		auto& muts = pop_owner->m_mutStep;
		ar & vars;
		ar & muts;
		ar & objs;
	}

    public:
	template <typename Archive>
	auto serialize(Archive& ar, const unsigned version) -> decltype(serialize_impl(ar, version), void())
	{
		serialize_impl(ar, version);
	}

	template <typename Archive>
	auto serialize(Archive& ar, const unsigned version)
	{
		auto* impl = static_cast<Base*>(this);
		impl->serialize_impl(ar, version);
	}

	friend std::ostream& operator<<(std::ostream& os, CSerializable const& serializable) {
		boost::archive::text_oarchive oa{os};
		oa << serializable;
		return os;
	}

	friend std::istream& operator>>(std::istream& is, CSerializable& serializable) {
		boost::archive::text_iarchive ia{is};
		ia >> serializable;
		return is;
	}

};
