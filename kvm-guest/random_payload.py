from multiprocessing import Lock,Process
import time
import sys
import numpy as np
import psutil
import socket
import math
import os

np.random.seed(1234)
step = 1

#total_memory = int(psutil.virtual_memory().total / 1024 / 1024) #划分区块

payload_map = {"192.168.122.195":[0.55,0.65,1,2],"192.168.122.196":[0.55,0.65,1,2],"192.168.122.197":[0.55,0.65,1,2],
			   "192.168.122.198":[0.1,0,2,3,6],"192.168.122.199":[0.35,0.5,2,4],"192.168.122.200":[0.2,0.4,2,4],
			   "192.168.122.201":[0.45,0.6,2,3],"192.168.122.202":[0.1,0.2,3,5],"192.168.122.203":[0.2,0.3,3,5],
			   "192.168.122.204":[0.2,0.3,2,4],"192.168.122.205":[0.25,0.35,2,3],"192.168.122.206":[0.4,0.5,2,3],"192.168.122.207":[0.1,0.3,3,5],
			   "192.168.122.208":[0.35,0.55,2,4],"192.168.122.209":[0.1,0.2,3,6],"192.168.122.210":[0.2,0.3,2,5]}


#实现占用内存
def mem(payload_rate,lock):
	index = 0
	chr_map = [' ','*','-','a','b','c']
	while True:
		total_memory = int(psutil.virtual_memory().total / 1024 / 1024)
		tmp_payload = math.ceil(payload_rate * total_memory)
		payload = tmp_payload + np.random.randint(-5,6)*7 #增加一些扰动
		s = chr_map[index] * (payload * 1024 * 1024)
		index = (index+1) % len(chr_map)
		if(lock.acquire(False)):
			break
		time.sleep(20)
	lock.release()

#cpu满载
def deadloop(lock):
	while True:
		if(lock.acquire(False)): #非阻塞
			break
		pass
	lock.release()

#根据传参来指定占满几个核
def cpu(cpunum,lock):
	lockcpu = []
	process = []
	#占用CPU
	for i in range(cpunum):
		lockcpu.append(Lock())
		lockcpu[i].acquire()
		process.append(Process(target=deadloop,args=(lockcpu[i],)).start())
	#释放CPU
	if(lock.acquire()):
		for i in range(cpunum):
			lockcpu[i].release()
	lock.release()

def gethostip():
	s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	s.connect(("8.8.8.8",80))
	ip = s.getsockname()[0]
	s.close()
	return ip


def main():
	global step
	ip = gethostip()
	mem_payload = np.random.uniform(payload_map[ip][0],payload_map[ip][1]) #内存的payload
	cpu_payload = np.random.randint(payload_map[ip][2],payload_map[ip][3])      #cpu的payload
	print("Step:{} Mem_payload:{}% Cpu_Payload:{}".format(step,mem_payload*100,cpu_payload))
	if ip > "192.168.122.202":
		os.system("bash net_init.sh &")
	clock = Lock()
	mlock = Lock()
	clock.acquire()
	mlock.acquire()
	memp = Process(target=mem,args=(mem_payload,mlock))
	cpup = Process(target=cpu,args=(cpu_payload,clock))
	memp.start()
	cpup.start()
	while True:
		time.sleep(20)
		mlock.release()
		clock.release()
		memp.join()
		cpup.join()
		step += 1
		mem_payload = np.random.uniform(payload_map[ip][0], payload_map[ip][1])  # 内存的payload
		cpu_payload = np.random.randint(payload_map[ip][2], payload_map[ip][3])  # cpu的payload
		print("Step:{} Mem_payload:{}% Cpu_Payload:{}".format(step, mem_payload*100 , cpu_payload))
		clock.acquire()
		mlock.acquire()
		memp = Process(target=mem, args=(mem_payload, mlock))
		cpup = Process(target=cpu, args=(cpu_payload, clock))
		memp.start()
		cpup.start()


if __name__ == '__main__':
	main()

	
#首先是随机生成负载
#应该针对不同的benchmark的
