#!/bin/bash
ips=("192.168.122.22" "192.168.122.133" "192.168.122.137" "192.168.122.103" "192.168.122.201" "192.168.122.2" "192.168.122.236" "192.168.122.223" "192.168.122.211" "192.168.122.81" "192.168.122.53" "192.168.122.12" "192.168.122.3" "192.168.122.217" "192.168.122.248" );
domains=("vm1" "vm2" "vm3" "vm4" "vm5" "vm6" "vm7" "vm8" "vm9" "vm10" "vm11" "vm12" "vm13" "vm14" "vm15");
macs=("52:54:00:23:39:ce" "52:54:00:cc:d4:6a" "52:54:00:ad:01:be" "52:54:00:a4:b5:e9" "52:54:00:54:02:d6" "52:54:00:b6:09:68" "52:54:00:da:31:d3" "52:54:00:ee:7b:52" "52:54:00:21:3c:0c" "52:54:00:27:30:eb" "52:54:00:ca:55:94" "52:54:00:f3:3f:b5" "52:54:00:9f:f6:de" "52:54:00:be:11:9e" "52:54:00:b6:5e:31");
memsize="2000MB"
cpunum=5
bandwidth=51200
for (( i = 0; i < 14; i++ )); do
    virsh setmem ${domains[$i]} $memsize 
    virsh setvcpus ${domains[$i]} --live --guest $cpunum
    virsh domiftune ${domains[$i]} ${macs[$i]} --live --config --inbound $bandwidth,$bandwidth,$bandwidth
    virsh domiftune ${domains[$i]} ${macs[$i]} --live --config --outbound $bandwidth,$bandwidth,$bandwidth
    echo "VM ${domains[$i]} has been cleand"    
done
