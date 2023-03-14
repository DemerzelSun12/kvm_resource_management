#!/bin/bash

#!/bin/bash


#python3 static_payload.py 1000 4 &
#./mm_test_static 1024M &

start=$[$(date +%s%N)/1000000]

#./http_load-09Mar2016/http_load -f 1000 -p 50 /home/vm1/kvm-guest/url.txt > http_load_without.txt &


i=1
while(( $i<=10 ))
do
    echo "run $i times"
    # java -jar dacapo-9.12-MR1-bach.jar avrora &
    java -jar dacapo-9.12-MR1-bach.jar xalan
	let "i++"
done

end=$[$(date +%s%N)/1000000]
take=$(( end - start ))
echo Time taken to execute commands is ${take} mseconds.
