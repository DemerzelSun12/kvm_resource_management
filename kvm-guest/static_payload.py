from multiprocessing import Lock,Process
import time
import sys
#实现占用内存
def mem(payload,lock):
    index = 0
    chr_map = [' ','*','-','a','b','c']
    while True:
        s = chr_map[index] * (payload * 1024 * 1024)
        index = (index+1) % len(chr_map)
        if(lock.acquire(False)):
            lock.acquire()
            time.sleep(10)
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

def main(mem_payload,cpu_payload):
    clock = Lock()
    mlock = Lock()
    clock.acquire()
    mlock.acquire()
    memp = Process(target=mem,args=(mem_payload,mlock))
    cpup = Process(target=cpu,args=(cpu_payload,clock))
    memp.start()
    cpup.start()
    while True:
        time.sleep(10000)
    mlock.release()
    clock.release()



if __name__ == '__main__':
    mem_payload = int(sys.argv[1])
    cpu_payload = int(sys.argv[2])
    main(mem_payload,cpu_payload)


