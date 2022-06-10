#!/bin/bash
# Attempts to build all examples and run them for less than a second
# Léo Chéneau - 2022
# $1 = easea binary to call
# $2 = --no-cuda
path_to_script=$(realpath $BASH_SOURCE)
examples_dir=$(dirname $path_to_script)
EZ_BINARY="$(realpath "$1")"
Color_Off='\033[0m'
Red='\033[0;31m'
Green='\033[0;32m'

SED=sed
# Use GNU version of sed
if command -v gsed 2>&1 > /dev/null
then
	echo "Found gnu-sed."
	SED=gsed
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
	EASEA_OUT=$($SED -n 's/\(\$[[:space:]]*\)*\(\.\/.*\)/\2/p' README.txt | xargs)
	if [[ "$EASEA_ARGS" == "" ]] || [[ "$EASEA_OUT" == "" ]]; then
		printf "$Red ko!$Color_Off\n"
		printf "\tError:$Red Bad README\n$Color_Off"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
		continue
	fi
	printf "$Green ok!$Color_Off\n"

	# Convert args if CUDA not available
	if [[ "$2" == "--no-cuda" ]]; then
		EASEA_ARGS=$(echo $EASEA_ARGS | $SED 's/\(cuda_\|-cuda\|_cuda\)//')
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

	# compile
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

	# Match output file
	CURATED_BIN=$(echo $EASEA_OUT | $SED -n 's/\.\///p')
	echo "DEBUG: " "$CURATED_BIN"
	echo "DEBUG: " "$(find . -type f -name "$CURATED_BIN*")"
	echo "DEBUG: " "$(find . -type f -name "$CURATED_BIN*" | $SED -n 's/^\(\.\/\)*\('"$CURATED_BIN"'\)\(\.exe\)*$/\2\3/p')"
	EASEA_OUT=$(find . -type f -name "$CURATED_BIN*" | $SED -n 's/^\(\.\/\)*\('"$CURATED_BIN"'\)\(\.exe\)*$/\2\3/p' | head -n1)
	echo "DEBUG: " "$EASEA_OUT"

	# run
	printf "\tExecuting %s ..." "$EASEA_OUT"
	OUT=$(timeout -k 2s -s KILL 1s "./$EASEA_OUT")
	ret=$?
	if [[ "$ret" == "0" ]] || [[ "$ret" == "124" ]]; then # ok
		printf "$Green ok!$Color_Off\n"
		passed=$((passed + 1))
		passed_list+=($(basename $edir))
	else
		printf "$Red ko!$Color_Off\n"
		printf "\tError (RETURNED %d):$Red %s\n$Color_Off" "$ret" "$OUT"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
	fi

	# clean
	make clean easeaclean >/dev/null
	rm -rf *.log
done

# Stats
printf "\n### Results ###\n"
printf "passed: $Green%d$Color_Off/%d\n" "$passed" "$nb_examples"

if [[ "$((nb_examples-passed))" != "0" ]]; then # at least one fail
	printf "failed:$Red %d$Color_Off/%d\n" "$failed" "$nb_examples"
	printf "test failed:$Red"
	for te in "${failed_list[@]}"; do
		printf " %s" "$te"
	done
	printf "$Red\nFAILED$Color_Off\n"
	exit 1
else
	printf "$Green"
	printf "PASSED$Color_Off\n"
	exit 0
fi
