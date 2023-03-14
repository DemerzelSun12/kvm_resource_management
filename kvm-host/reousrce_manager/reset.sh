#!/bin/bash
#This is the basic example of 'for loop'.

vms="vm1 vm2 vm3 vm4 vm5 vm6 vm7 vm8 vm9 vm10 vm11 vm12 vm13 vm14 vm15"

for vm in $vms
do
sudo virsh setvcpus --live --guest $vm 4
sudo virsh setmem $vm --size 2048MB
done

declare -A myMap=(["vm1"]="52:54:00:23:39:ce" 
                  ["vm2"]="52:54:00:cc:d4:6a" 
                  ["vm3"]="52:54:00:ad:01:be" 
                  ["vm4"]="52:54:00:a4:b5:e9" 
                  ["vm5"]="52:54:00:54:02:d6" 
                  ["vm6"]="52:54:00:b6:09:68" 
                  ["vm7"]="52:54:00:da:31:d3" 
                  ["vm8"]="52:54:00:ee:7b:52" 
                  ["vm9"]="52:54:00:21:3c:0c" 
                  ["vm10"]="52:54:00:27:30:eb"
                  ["vm11"]="52:54:00:ca:55:94"
                  ["vm12"]="52:54:00:f3:3f:b5"
                  ["vm13"]="52:54:00:9f:f6:de"
                  ["vm14"]="52:54:00:be:11:9e"
                  ["vm15"]="52:54:00:b6:5e:31" )

for key in ${!myMap[*]};do
  echo $key
  echo ${myMap[$key]}
  sudo virsh domiftune $key ${myMap[$key]} --live --config --inbound 1024
  sudo virsh domiftune $key ${myMap[$key]} --live --config --outbound 1024
done



