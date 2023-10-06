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
	./$1 --serverPort $port --seed $((seed + port + i)) --ipFile ip.txt &
	i=$((i+1))
done
