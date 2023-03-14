import csv
import psutil
import time

def task_gather(writer):
	i = 0
	max_iterator = 3600 * 5
	while i < max_iterator:
		cpu_usage = psutil.cpu_percent(interval=1)
		mem_usage = psutil.virtual_memory().percent
		localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
		writer.writerow([localtime,i,cpu_usage,mem_usage])
		i += 1

if __name__ == '__main__':
	f = open("no_adjust.csv","w")
	writer = csv.writer(f)
	task_gather(writer)
	f.close()