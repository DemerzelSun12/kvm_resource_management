#!/bin/bash
mem_payload=200
cpu_payload=0
net_payload=2
exNo="1-5"
ip=$(ifconfig  |grep 192.* | awk '{print $2}')
base_ip=192.168.122.15
#step1: Start the web Server:
echo "run server.py"
python3 server.py $ip > ./weblog.txt &

#Step2: Run the static payload
echo "run static_payload.py"
python3 static_payload.py $mem_payload $cpu_payload > /dev/null &

#Step3: Run the benchmark 
#python3 benchmark_test.py $exNo $mem_payload $cpu_payload  > $exNo".txt" &

#bash restart.sh ${exNo} >/dev/null &
#Step4: if ip address is bigger than 202 you should add some network payload
#if [ ${ip} \> ${base_ip} ];then
#	nohup iperf -c 10.1.5.39 -b ${net_payload}MB >/dev/null & 
#	bash net_init.sh &
#fi

# nohup iperf -c 101.200.218.87 -b ${net_payload}MB > /dev/null &
bash net_init.sh &
#fi
python3 benchmark_test.py $exNo $mem_payload $cpu_payload  > $exNo".txt"
./clean.sh


