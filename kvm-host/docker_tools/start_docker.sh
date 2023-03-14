#!/bin/bash
for l in `seq 4 9`;do
    docker run -it --name=ubuntu18.04-0${l} --cap-add=NET_ADMIN ex:v2 /bin/bash &
done

for k in `seq 10 20`;do
    docker run -it --name=ubuntu18.04-${k} --cap-add=NET_ADMIN ex:v2 /bin/bash &
done 
