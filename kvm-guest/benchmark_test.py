import os
import time
import sys
import psutil
import threading
import signal
import numpy as np
from multiprocessing import Process,cpu_count,Lock,Value
import multiprocessing
import subprocess
import csv
import traceback
import socket
# import pymysql

script_type = {"192.168.122.15":"h2","192.168.122.188":"tradebeans"}
tmpp = 0.0
domain = {"192.168.122.15":0,"192.168.122.188":1}


step = 1
# conn = pymysql.connect(host='101.200.218.87',user='root',passwd='123456')
# conn.select_db("resource_M")
# cur = conn.cursor()

#实现占用内存
def mem(payload,lock):
	s = ' ' * (payload * 1024 * 1024)
	lock.acquire() #由于父进程没有释放锁就会一直阻塞
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

def subprocess_popen(statement):
	result={}
	p = subprocess.Popen(statement, shell=True, stdout=subprocess.PIPE)  # 执行shell语句并定义输出格式
	while p.poll() is None:
		if p.wait() != 0:
			print("命令执行失败，请检查设备连接状态")
			return result
		else:
			re = p.stdout.readlines()  # 获取原始执行结果
			#result = {}
			for i in range(len(re)):  # 由于原始结果需要转换编码，所以循环转为utf8编码并且去除\n换行
				res = re[i].decode('utf-8').strip('\r\n')
				result[i] = res
			return result
	return result

def cal_delaytime(benchmark, exno, extime,vmid):
	signal.signal(signal.SIGTTOU, signal.SIG_IGN)
	extime = extime  ###!!!!!!修改这里可以改变benchmark运行次数
	elapsed = []
	errno_num = 0
	time.sleep(1)
	for i in range(extime):
		t = threading.Thread(target=thread_gather)
		t.start()
		print("run dapaco_test.sh")
		os.system("./dacapo_test.sh {} {} {}_{} > /dev/null".format(benchmark, 1, benchmark, exno))
		ret = subprocess_popen("cat {}_{}.e.log |grep PASSED|wc -l".format(benchmark, exno))
		t.join()
		if len(ret)<=0 or int(ret[0]) == 0:
			print("Benchmark {} casued Error!".format(benchmark))
			errno_num += 1
			continue
		passtime = subprocess_popen("grep -oP 'in (\d+)' %s_%s.e.log|awk -F ' ' '{print $2}'" % (benchmark, str(exno)))
		if len(passtime)>0:
			elapsed.append(float(passtime[0]) / 1000)
			print("The benchmark run time:{}".format(float(passtime[0])/1000),flush=True)
	
	if (errno_num < 6):
		it = errno_num + 2
		while (errno_num > 0 and it >= 0):
			it -= 1
			os.system("./dacapo_test.sh {} {} {}_{} > /dev/null".format(benchmark, 1, benchmark, exno))
			ret = subprocess_popen("cat {}_{}.e.log |grep PASSED|wc -l".format(benchmark, exno))
			if len(ret)<=0 or int(ret[0]) == 0:
				print("Benchmark {} casued Error!".format(benchmark))
				continue
			passtime = subprocess_popen("grep -oP 'in (\d+)' %s_%s.e.log|awk -F ' ' '{print $2}'" % (benchmark, str(exno)))
			if len(passtime)>0:
				elapsed.append(int(passtime[0]) / 1000)
				print("The benchmark run time:{}".format(float(passtime[0])/1000),flush=True)
				errno_num -= 1
			
		if (errno_num > 0):
			print("The Benchmark {} Execute Error Too Much In {} \n Please try another time!".format(benchmark, exno))
			elapsed = [-1] * extime
	else:
		print("The Benchmark {} Execute Error Too Much In {} \n Please try another time!".format(benchmark, exno))
		elapsed = [-1] * extime
    
	elapsed = np.array(elapsed)
	print("**********************************")
	print("Time average used:{}s ".format(elapsed.mean()),flush=True)
	print("Time std: {}".format(elapsed.std(),flush=True))
	print("Time stamp:{}".format(time.asctime(time.localtime(time.time()))))
	print("**********************************")
	global step
	if sum(elapsed==-1):
		pass
	else:
		time_stamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
		# sql = "insert into ex09(vm_id,benchmark,used_time,time_stamp,stds,exNo,step) value({},'{}',{},'{}',{},'{}',{})"
		f = open('test.csv','w',encoding='utf-8')
		writer = csv.writer(f)
		writer.writerow(["{},{},{},{},{},{},{}".format(vmid,benchmark,elapsed.mean(),time_stamp,elapsed.std(),exno,step)])
		# cur.execute(sql.format(vmid,benchmark,elapsed.mean(),time_stamp,elapsed.std(),exno,step))
		# conn.commit()
		step += 1

def thread_gather():
	cpu_usage = 0.0
	mem_usage = 0.0
	swap_usage = 0.0
	psutil.cpu_percent(interval=1)
	time.sleep(2)
	for i in range(10):
		cpu_usage += psutil.cpu_percent(interval=0.1)
		mem_usage += psutil.virtual_memory().percent
		swap_usage += psutil.swap_memory().percent
	cpu_usage = round(cpu_usage/10,4)
	mem_usage = round(mem_usage/10,4)
	swap_usage = round(swap_usage/10,4)
	print("Runing Time cpu:{}%; mem:{}%; swap:{}%".format(cpu_usage,mem_usage,swap_usage))

def gethostip():
	s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	s.connect(("8.8.8.8",80))
	ip = s.getsockname()[0]
	s.close()
	return ip

def main():
	print("in main before")
	signal.signal(signal.SIGTTOU,signal.SIG_IGN)
	print("in main")
	exNo = sys.argv[1]
	mem_payload = 0
	cpu_payload = 0
	#os.system("ifconfig  |grep 192.* | awk '{print $2}'")
	ip = gethostip()
	print(ip)
	benchmark = script_type[ip]
	vmid=domain[ip]
	print("------------------------------------")
	print("Payload mem is {}MB".format(mem_payload))
	print("Payload CPU is {} ".format(cpu_payload))
	print("The running Benchmark is {}".format(benchmark))
	sys.stdout.flush()
	while True:
		cal_delaytime(benchmark,exNo,11,vmid)
		time.sleep(1)
		print("++++++++++++++++++++++++++++++++++++++++++ \n\n")
		break

def print_help():
	print("No Such Command, Please try again!!!")

if __name__ == "__main__":
	print(1)
	main()
	# cur.close()
	# conn.close()
