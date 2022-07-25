#!/bin/bash
# Start multiple servers from one IP file
# Léo Chéneau on 2022-06-23

seed=$2
if [[ $seed == "" ]]; then
	seed=0
fi

i=0
for port in $(cat ip.txt | cut -d ':'  -f 2); do
	printf "Launching $1 on port: $port\n"
	if [ $i == 0 ]; then 
		./$1 --serverPort $port --seed $((seed + port)) --ipFile ip.txt &
	else
		./$1 --serverPort $port --seed $((seed + port +i)) &
	fi
	i=$((i+1))
done
