#!/bin/bash
mem_payload=200
cpu_payload=0
net_payload=2
exNo="1-4"
base_ip=192.168.122.202
ip=$(ifconfig  |grep 192.* | awk '{print $2}')

declare -A benchmarks=(["192.168.122.22"]="h2" ["192.168.122.133"]="tradebeans" ["192.168.122.137"]="tradesoap" ["192.168.122.103"]="eclipse" ["192.168.122.201"]="avrora" ["192.168.122.2"]="jython" ["192.168.122.236"]="sunflow" ["192.168.122.223"]="lusearch" ["192.168.122.211"]="lusearch-fix" ["192.168.122.81"]="pmd" ["192.168.122.53"]="luindex" ["192.168.122.12"]="batik" ["192.168.122.3"]="fop" ["192.168.122.217"]="xalan" ["192.168.122.248"]="eclipse")
benchmark=${benchmarks[$ip]}
echo ${benchmark}

#start static payload
python3 static_payload.py $mem_payload $cpu_payload > /dev/null &

#start network payload
if [ ${ip} \> ${base_ip} ]; then
	nohup iperf -c 10.1.5.39 -b ${net_payload}MB >/dev/null &
	bash net_init.sh
fi

# repeat java
while [[ 1 ]]; do
	./dacapo_test.sh ${benchmark} 10 ${benchmark} > /dev/null
	sleep 1
done
