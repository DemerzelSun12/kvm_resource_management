#!/bin/bash
restart=$1
dateStr=$(date "+%m-%d_%H:%M:%S")
alive_python=$(ps -axu|grep python3|grep -v /usr/bin -wc)
alive_java=$(ps -axu|grep java -wc)

#step1: kill all python process
while ((alive_python>1))
do
	killall -9 python3
    alive_python=$(ps -axu|grep python3|grep -v /usr/bin -wc)
done
echo "All The python Process has been Killed"

#step2: kill all java process
#while ((alive_java>1))
#do
#	killall -9 java
#	alive_java=$(ps -axu|grep java -wc)
#done

#echo "All The java Process has been Killed"


#step3: kill all net process

#alive_net=$(ps -axu|grep net_init -wc)
#alive_http=$(ps -axu|grep http_load -wc)
#iperf_pid=$(ps -axu|grep iperf -wc)

init_pid=$(ps -axu|grep init.sh|grep -v "grep"|awk -F " " '{print $2}')
net_pid=$(ps -axu|grep net_init|grep -v color -m 1|awk -F " " '{print $2}')
http_pid=$(ps -axu|grep http_load|grep -v color -m 1|awk -F " " '{print $2}')
iperf_pid=$(ps -axu|grep iperf -m 1|grep -v color -m 1|awk -F " " '{print $2}')
restart_pid=$(ps -axu|grep restart.sh|grep -v color -m 1|awk -F " " '{print $2}')
train_pid=$(ps -axu|grep train.sh|grep -v color|awk -F " " '{print $2}')
dacapo_pid=$(ps -axu|grep dacapo_test|grep -v color|awk -F " " '{print $2}')

kill -9 ${net_pid}
kill -9 ${http_pid}
kill -9 ${iperf_pid}
kill -9 ${train_pid}
kill -9 ${dacapo_pid}

while ((alive_java>1))
do
    killall -9 java
	alive_java=$(ps -axu|grep java -wc)
done

echo "All The java Process has been Killed"


if [ "$restart" = "" ];then
    echo 1111
    kill -9 ${restart_pid}
	rm benchmark.txt
	rm bandwidth.txt
fi

echo "All the network process has been killed"

#step5: delete all the previous log files
mv 1-4.txt ./log/1-4.txt.${dateStr}
#rm ./weblog.txt ./*.log

