#!/bin/bash
#num=$(ps -axu|grep benchmark_test -wc)
exNo="1-4"

#set +e
bash clean.sh 1
bash init.sh

sleep 2
num=$(ps -axu|grep benchmark_test -wc)

while [ 1 ]
do
	#echo ${num}
	if ((num == 2));then
		#echo sleep!
		sleep 20
	else
		echo benchmark error!!!
		echo !!!!!!!!!!!!!!! >> benchmark.txt
		echo !!!!!!!!!!!!!!! >> bandwidth.txt
		cat $exNo.txt >> benchmark.txt
		#cat bandwidth.log >> bandwidth.txt
		bash clean.sh 1
		bash init.sh 
		sleep 20
		#ython3 benchmark_test.py $exNo 0 0 > $exNo".log" &
	fi
	sleep 40
	num=$(ps -axu|grep benchmark_test -wc)
done 

