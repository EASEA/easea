#!/bin/bash

if [ $# -lt 3 ]
then
    echo "Usage : $0 <directory_1> <directory_2> <result_directory>"
    exit
fi


mkdir -p $3
rm -rf $3/*

for directory in `ls $1`
do
    mkdir "$3/$directory"
    for file in `ls $1/$directory`
    do
        diff $1/$directory/$file $2/$directory/$file > $3/$directory/$file.diff
    done
done
