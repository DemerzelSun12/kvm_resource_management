#!/bin/bash
cpu_num=$1
mem_num=$2

echo $cpu_num $mem_num

for i in `seq 4 9`;do
    #first CPU: 2
    docker update --cpus ${cpu_num} ubuntu18.04-0${i} 

    #second MEM: 1024MB
    docker update --memory ${mem_num}M --memory-swap -1 ubuntu18.04-0${i}
    
    #third BANDWIDTH: 1
    docker exec ubuntu18.04-0${i} bash -c "tc class change dev eth0 parent 1: classid 1:21 htb rate 50Mbit ceil 50Mbit"
    echo ubuntu18.04-0${i} is done
done

for j in `seq 10 20`;do
    docker update --cpus ${cpu_num} ubuntu18.04-${j} 
    docker update --memory ${mem_num}M  --memory-swap -1  ubuntu18.04-${j} 
    docker exec ubuntu18.04-${j} bash -c "tc class change dev eth0 parent 1: classid 1:21 htb rate 50Mbit ceil 50Mbit"
    echo ubuntu18.04-${j} is done
done
