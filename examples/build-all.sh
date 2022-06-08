#!/bin/bash
# Attempts to build all examples and run them for less than a second
# Léo Chéneau - 2022
# $1 = easea binary to call
# $2 = --no-cuda
path_to_script=$(realpath $BASH_SOURCE)
examples_dir=$(dirname $path_to_script)
Color_Off='\033[0m'
Red='\033[0;31m'
Green='\033[0;32m'

# Retrieves test directories
printf "Calculating examples list..."
all_examples=$(find $examples_dir -mindepth 1 -type d -exec sh -c '[ -f "$0"/*.ez ]' {} \; -print 2>&1 | sed -n 's/^\([./a-zA-Z0-9_\-]\+\)\(.*\)*$/\1/p')
nb_examples=$(echo $all_examples | wc -w)
printf "$Green ok!\n$Color_Off"
echo "Found $nb_examples examples to compile."

# Build and run each example
passed=0
failed=0
failed_list=()
passed_list=()
for edir in $all_examples; do
	printf -- "- ($((passed+failed+1))/$nb_examples) $(basename $edir):\n"
	cd $edir

	printf -- "\treading README.txt..."
	EASEA_ARGS=$(sed -n 's/\$ ease\(a\|na[[:space:]]\)\([^ ]*\)/\2/p' README.txt | head -n1 | xargs)
	EASEA_OUT=$(sed -n 's/\(\$[[:space:]]*\)*\(\.\/.*\)/\2/p' README.txt | xargs)
	if [[ "$EASEA_ARGS" == "" ]] || [[ "$EASEA_OUT" == "" ]]; then
		printf "$Red ko!$Color_Off\n"
		printf "\tError:$Red Bad README\n$Color_Off"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
		continue
	fi
	printf "$Green ok!$Color_Off\n"

	# Convert args if CUDA not available
	if [ $2 == "--no-cuda" ]; then
		EASEA_ARGS=$(echo $EASEA_ARGS | sed 's/\(cuda_\|-cuda\)//')
	fi

	# build
	printf -- "\t$1 $EASEA_ARGS..."
	OUT=$($1 $EASEA_ARGS 2>&1)
	if [[ "$?" != "0" ]]; then # error
		printf "$Red ko!$Color_Off\n"
		printf "\tError:$Red $OUT\n$Color_Off"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
		continue
	fi
	printf "$Green ok!$Color_Off\n"

	# compile
	printf -- "\tmake..."
	OUT=$(make 2>&1)
	if [[ "$?" != "0" ]]; then # error
		printf "$Red ko!$Color_Off\n"
		printf "\tError:$Red $OUT\n$Color_Off"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
		continue
	fi
	printf "$Green ok!$Color_Off\n"

	# run
	printf -- "\t$EASEA_OUT..."
	OUT=$(timeout -k 2 1 $EASEA_OUT)
	ret=$?
	if [[ "$ret" == "0" ]] || [[ "$ret" == "124" ]]; then # ok
		printf "$Green ok!$Color_Off\n"
		passed=$((passed + 1))
		passed_list+=($(basename $edir))
	else
		printf "$Red ko!$Color_Off\n"
		printf "\tError (RETURNED $ret):$Red $OUT\n$Color_Off"
		failed=$((failed + 1))
		failed_list+=($(basename $edir))
	fi

	# clean
	make clean easeaclean >/dev/null
	rm -rf *.log
done

# Stats
printf "\n### Results ###\n"
printf "passed: $Green$passed$Color_Off/$nb_examples\n"

if [[ "$((nb_examples-passed))" != "0" ]]; then # at least one fail
	printf "failed:$Red $failed$Color_Off/$nb_examples\n"
	printf "test failed:$Red"
	for te in "${failed_list[@]}"; do
		printf " $te"
	done
	printf "$Red\nFAILED$Color_Off\n"
	exit 1
else
	printf "$Green"
	printf "PASSED$Color_Off\n"
	exit 0
fi
