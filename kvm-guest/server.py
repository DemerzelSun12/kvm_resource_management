import socket
from multiprocessing import Process,Lock
import sys
import psutil
import time

ip = sys.argv[1]
port = 12345
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind((ip,port))

def get_key(netcard):
	key_info = psutil.net_io_counters(pernic=True).keys()
	key_info = list(key_info)
	recv = 0
	sent = 0
	for key in key_info:
		if key == netcard:   ##不同主机上要修改这个监听的网卡！！！！
			recv = psutil.net_io_counters(pernic=True).get(key).bytes_recv
			sent = psutil.net_io_counters(pernic=True).get(key).bytes_sent
	return recv, sent

def getNetworkFlow(netcard):
	old_recv, old_sent = get_key(netcard)
	time.sleep(1)
	now_recv, now_sent = get_key(netcard)
	net_in = 0
	net_out = 0
	net_in = float('%.4f' % ((now_recv - old_recv) / 1024))
	net_out = float('%.4f' % ((now_sent - old_sent) / 1024))
	return net_in, net_out

def WebServer():
	while True:
		msg,addr = s.recvfrom(1024)
		if int(msg.decode(encoding="utf-8")) == 1:
			mem_info = psutil.virtual_memory()
			cpu_num = psutil.cpu_count()
			cpu_usage = psutil.cpu_percent()
			mem_usage = str(mem_info.percent)
			net_in,net_out = getNetworkFlow("ens3")
			print("cpu num:{}  cpu usage:{}  mem size:{}  mem usage:{} ".format(cpu_num,cpu_usage,mem_info.total/1024/1024,mem_info.percent))
			s.sendto(mem_usage.encode(encoding="utf-8"),addr)
			s.sendto(str(net_in).encode(encoding="utf-8"),addr)
			s.sendto(str(net_out).encode(encoding="utf-8"),addr)

WebServer()


