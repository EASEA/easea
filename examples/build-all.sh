#!/bin/bash
# Attempts to build all examples and run them for less than a second
# Léo Chéneau - 2022
# $1 = easea binary to call
# $2 = --no-cuda
# $3 = --verbose
path_to_script=$(realpath $BASH_SOURCE)
examples_dir=$(dirname $path_to_script)
EZ_BINARY=""
VERBOSE=false
CUDA=true
EZARGS=""
CMARGS=""
IGNORE_ERRORS=""

### CLI args
function help {
	printf -- "usage: [ -v | --verbose ] [ -e | --ez-args <ARGS> ] [ -c | --cmake-args <ARGS> ] [ --ignore-errors <NAME>] [ --no-cuda ] <COMPILER>\n"
	printf -- "<COMPILER> : path to the EASEA compiler.\n"
	printf -- "--verbose or -v : (optional) print more informations, such as all ouputs of commands.\n"
	printf -- "--ez-args <ARGS> or -e <ARGS> : (optional) arguments to pass to the final EASEA program.\n"
	printf -- "--cmake-args <ARGS> or -c <ARGS> : (optional) arguments to pass to the first CMake command.\n"
	printf -- "--ignore-errors <NAME> : (optional) ignore example <NAME>. DO NOT USE ON ERRORS YOU CAN FIX!\n"
	printf -- "--no-cuda : (optional) replace CUDA templates with non-CUDA ones.\n"
	printf -- "--help : print this help message.\n"
}

NEXT_EZARGS=false
NEXT_CMARGS=false
NEXT_IGNORE=false
for var in "$@"; do
	if $NEXT_EZARGS; then
		EZARGS="$var"
		NEXT_EZARGS=false
	elif $NEXT_CMARGS; then
		CMARGS="$var"
		NEXT_CMARGS=false
	elif $NEXT_IGNORE; then
		IGNORE_ERRORS="$var $IGNORE_ERRORS"
		NEXT_IGNORE=false
	else
		if [[ "$var" == "--verbose" ]] || [[ "$var" == "-v" ]]; then
			VERBOSE=true
		elif [[ "$var" == "--no-cuda" ]]; then
			CUDA=false
		elif [[ "$var" == "--no-cuda" ]]; then
			help
			exit 0
		elif [[ "$var" == "--ez-args" ]] || [[ "$var" == "-e" ]]; then
			NEXT_EZARGS=true
		elif [[ "$var" == "--cmake-args" ]] || [[ "$var" == "-c" ]]; then
			NEXT_CMARGS=true
		elif [[ "$var" == "--ignore-errors" ]]; then
			NEXT_IGNORE=true
		else
			if [[ "$EZ_BINARY" == "" ]]; then
				EZ_BINARY="$(realpath "$var")"
			else
				printf "Unknow CLI args: \"%s\"\n" "$var"
				help
				exit 1
			fi
		fi
	fi
done

Color_Off='\033[0m'
Red='\033[0;31m'
Green='\033[0;32m'
Yellow='\033[0;33m'

if [[ $IGNORE_ERRORS != "" ]]; then
	printf "$Yellow"
	printf "Warning: Please DO NOT use '--ignore-errors' for errors you introduced!\n$Color_Off"
fi

SED=sed
# Use GNU version of sed
if command -v gsed 2>&1 > /dev/null
then
	echo "Found gnu-sed."
	SED=gsed
fi

TIMEOUT=timeout
# Use GNU version of timeout
if command -v gtimeout 2>&1 > /dev/null
then
	echo "Found gnu-timeout."
	TIMEOUT=gtimeout
fi

# Retrieves test directories
printf "Calculating examples list..."
all_examples_raw=$(find $examples_dir -mindepth 2 -type f -name "*.ez")
all_examples=""
for f in $all_examples_raw; do
	all_examples="$all_examples $(dirname $f | xargs realpath)"
done
nb_examples=$(echo $all_examples | wc -w | tr -d " ")
printf "$Green ok!\n$Color_Off"
echo "Found $nb_examples examples to compile."

# Build and run each example
passed=0
failed=0
failed_list=()
passed_list=()
for edir in $all_examples; do
	printf -- "- %d/%d %s:\n" $((passed+failed+1)) $nb_examples $(basename $edir)
	cd $edir

	printf "\treading README.txt..."
	EASEA_ARGS=$($SED -n 's/\$[[:space:]]*ease\(a\|na[[:space:]]\)\([^ ]*\)/\2/p' README.txt | head -n1 | xargs)
	EASEA_BUILD=$($SED -n 's/\$[[:space:]]*\(.*make.*$\)/\1/p' README.txt | head -n1 | xargs)
	EASEA_OUT=$($SED -n 's/\(\$[[:space:]]*\)*\(\.\/.*\)/\2/p' README.txt | head -n1 | xargs)
	if [[ "$EASEA_ARGS" == "" ]] || [[ "$EASEA_OUT" == "" ]]; then
		printf "$Red ko!$Color_Off\n"
		printf "\tError:$Red Bad README\n$Color_Off"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
		continue
	fi
	printf "$Green ok!$Color_Off\n"

	# Convert args if CUDA not available
	if ! $CUDA; then
		EASEA_ARGS=$(echo $EASEA_ARGS | $SED 's/\(cuda_\|_cuda\)//')
		EASEA_ARGS=$(echo $EASEA_ARGS | $SED 's/\(-cuda\)//')
	fi

	# build
	printf "\t%s %s..." "$EZ_BINARY" "$EASEA_ARGS"
	OUT=$("$EZ_BINARY" $EASEA_ARGS 2>&1)
	if [[ "$?" != "0" ]]; then # error
		printf "$Red ko!$Color_Off\n"
		printf "\tError:$Red %s\n$Color_Off" "$OUT"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
		continue
	fi
	printf "$Green ok!$Color_Off\n"
	if $VERBOSE; then
		printf "Output:\n===\n%s\n===\n" "$OUT"
	fi

	# compile
	EASEA_BUILD="$(echo "$EASEA_BUILD" | sed 's/cmake/cmake '"$CMARGS"'/')"
	printf -- "\t$EASEA_BUILD..."
	OUT=$(bash -c "$EASEA_BUILD" 2>&1)
	if [[ "$?" != "0" ]]; then # error
		printf "$Red ko!$Color_Off\n"
		#echo -e -- "\tError:$Red $OUT\n$Color_Off"
		printf "\tError:$Red %s\n$Color_Off" "$OUT"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
		continue
	fi
	printf "$Green ok!$Color_Off\n"
	if $VERBOSE; then
		printf "Output:\n===\n%s\n===\n" "$OUT"
	fi

	# Match output file
	CURATED_BIN=$(echo $EASEA_OUT | $SED -n 's/\.\///p')
	#echo "DEBUG: " "$CURATED_BIN"
	#echo "DEBUG: " "$(find . -type f -name "$CURATED_BIN*")"
	#echo "DEBUG: " "$(find . -type f -name "$CURATED_BIN*" | $SED -n 's/^\(\.\/\)*\([^.]*'"$CURATED_BIN"'\)\(\.exe\)*$/\2\3/p')"
	EASEA_OUT=$(find . -type f -name "$CURATED_BIN*" | $SED -n 's/^\(\.\/\)*\([^.]*'"$CURATED_BIN"'\)\(\.exe\)*$/\2\3/p' | head -n1)
	#echo "DEBUG: " "$EASEA_OUT"

	# run
	printf "\tExecuting %s %s..." "$EASEA_OUT" "$EZARGS"
	OUT=$($TIMEOUT -k 7s 5s "./$EASEA_OUT" $EZARGS)
	ret=$?
	if [[ "$ret" == "0" ]] || [[ "$ret" == "124" ]] || [[ "$ret" == "137" ]]; then # ok
		printf "$Green ok!$Color_Off\n"
		passed=$((passed + 1))
		passed_list+=($(basename $edir))
	else
		printf "$Red ko!$Color_Off\n"
		printf "\tError (RETURNED %d):$Red %s\n$Color_Off" "$ret" "$OUT"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
	fi
	if $VERBOSE; then
		printf "Output:\n===\n%s\n===\n" "$OUT"
	fi

	# clean
	make clean easeaclean >/dev/null 2>/dev/null
	rm -rf *.log *.prm CMakeLists.txt CMakeFiles CMakeCache *.plot
done

# Compute ignored tests
warnings=()
true_errors=()
nb_warning=0
for er in "${failed_list[@]}"; do
	IGNORE=false
	for si in $IGNORE_ERRORS; do
		if [[ "$si" == "$er" ]]; then
			IGNORE=true
			break
		fi
	done
	if $IGNORE; then
		warnings+=($er)
		failed=$((failed - 1))
		#passed=$((passed + 1))
		nb_warning=$((nb_warning + 1))
	else
		true_errors+=($er)
	fi
done

# Stats
printf "\n### Results ###\n"
printf "passed: $Green%d$Color_Off/%d\n" "$passed" "$nb_examples"

if [[ "$nb_warning" != "0" ]]; then
	printf "ignored:$Yellow %d$Color_Off/%d\n" "$nb_warning" "$nb_examples"
fi


EXIT_CODE=0
if [[ "$failed" != "0" ]]; then # at least one fail
	printf "failed:$Red %d$Color_Off/%d\n" "$failed" "$nb_examples"
	printf "test failed:$Red"
	for te in "${true_errors[@]}"; do
		printf " %s" "$te"
	done
	printf "$Red\nFAILED$Color_Off\n"
	EXIT_CODE=1
else
	printf "$Green"
	printf "PASSED$Color_Off\n"
	EXIT_CODE=0
fi

if [[ "$nb_warning" != "0" ]]; then
	printf "\n$Yellow"
	if [[ "$nb_warning" == "1" ]]; then
		printf "Warning:$Color_Off %d test failed but was ignored:$Yellow" "$nb_warning"
	else
		printf "Warning:$Color_Off %d tests failed but were ignored:$Yellow" "$nb_warning"
	fi
	for te in "${warnings[@]}"; do
		printf " %s" "$te"
	done
	printf "$Color_Off\n"
fi

exit $EXIT_CODE
