#!/bin/bash
i=1
while(( $i<=10 ))
do
	#http_load -p 100 -f 100 url.txt >> bandwidth_ddpg2.item
	#echo "++++++++++++++++++++++++++++++++" >> bandwidth_ddpg2.item
	let "i++"
done
