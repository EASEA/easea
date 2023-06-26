/*
 * COptionParser.h
 *
 *  Created on: 22 juin 2009
 *      Author: maitre
 */

#ifndef COPTIONPARSER_H
#define COPTIONPARSER_H

#include <type_traits>
#include <string>
#include <memory>

#include <boost/program_options.hpp>

using vm_t = boost::program_options::variables_map;

void parseArguments(const char* parametersFileName, int ac, char** av, std::unique_ptr<vm_t>&vm);

template <typename T>
struct is_c_str
	: std::integral_constant<
		  bool, (std::is_pointer<T>::value || std::is_array<T>::value ||
			 (std::is_reference<T>::value && std::is_array<std::remove_reference_t<T>>::value)) &&
				std::is_same<char, std::remove_all_extents_t<std::remove_const_t<std::remove_pointer_t<
							   std::remove_const_t<std::remove_reference_t<T>>>>>>::value> {
};

template <typename TypeVariable>
typename std::conditional_t<is_c_str<TypeVariable>::value, std::string, std::remove_cv_t<std::remove_reference_t<TypeVariable>>>
setVariable(const std::string& argumentName, TypeVariable&& defaultValue, std::unique_ptr<vm_t> const& vm)
{
	using ret_t = std::conditional_t<is_c_str<TypeVariable>::value, std::string, std::remove_cv_t<std::remove_reference_t<TypeVariable>>>;
	if (vm->count(argumentName)) {
		return (*vm)[argumentName].as<ret_t>();
	} else {
		return ret_t{ defaultValue };
	}
}

#endif /* COPTIONPARSER_H_ */
