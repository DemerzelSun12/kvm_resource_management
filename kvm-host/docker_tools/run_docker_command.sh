#!/bin/bash
command=$1
for i in `seq 4 9`;do
    #echo ${command}
    docker exec ubuntu18.04-0${i} ${command}
done

for j in `seq 10 20`;do
    docker exec ubuntu18.04-${j} ${command}
done
