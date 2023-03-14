import psutil
import os
import sys
import time
import csv
import threading

domain =  {1:"vm1",2:"vm2",3:"vm3",4:"vm4",5:"vm5",6:"vm6",7:"vm7",8:"vm8",9:"vm9",10:"vm10",11:"vm11",12:"vm12",13:"vm13",14:"vm14",15:"vm15"}

vm_mac = {1: "52:54:00:23:39:ce", 2: "52:54:00:cc:d4:6a", 3: "52:54:00:ad:01:be",
                4: "52:54:00:a4:b5:e9", 5: "52:54:00:54:02:d6", 6: "52:54:00:b6:09:68", 7: "52:54:00:da:31:d3",
                8: "52:54:00:ee:7b:52", 9: "52:54:00:21:3c:0c",
                10: "52:54:00:27:30:eb", 11: "52:54:00:ca:55:94", 12: "52:54:00:f3:3f:b5",
                13: "52:54:00:9f:f6:de", 14: "52:54:00:be:11:9e", 15: "52:54:00:b6:5e:31"}




def task_gather(writer):
    i = 0
    max_iterator = 3600 * 5
    while i < max_iterator:
        cpu_usage = psutil.cpu_percent(interval=1)
        mem_usage = psutil.virtual_memory().percent
        localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        writer.writerow([localtime,i,cpu_usage,mem_usage])
        i += 1



def task_adjust_kvm():
    i = 0
    max_iterator = 60 * 5
    flag = 1
    while i < max_iterator:
        for idx,vm in domain.items():
            if flag == 1:
                os.system("virsh setvcpus --live --guest {} {} > /dev/null".format(vm,2))
                os.system("virsh setmem {} --size {}MB > /dev/null".format(vm,1280))
                os.system("virsh domiftune {} {} --live --config --inbound {},{},{} > /dev/null".format(vm,vm_mac[idx], round(
                    256), round(256), round(256)))
                os.system("virsh domiftune {} {} --live --config --outbound {},{},{} > /dev/null".format(vm, vm_mac[idx],
                                                                                                        round(
                                                                                                            256),
                                                                                                        round(256),
                                                                                                        round(256)))
            else:
                os.system("virsh setvcpus --live --guest {} {} > /dev/null".format(vm,1))
                os.system("virsh setmem {} --size {}MB > /dev/null".format(vm,1024))
                os.system("virsh domiftune {} {} --live --config --inbound {},{},{} > /dev/null".format(vm,vm_mac[idx], round(
                    128), round(128), round(128)))
                os.system("virsh domiftune {} {} --live --config --outbound {},{},{} > /dev/null".format(vm, vm_mac[idx],
                                                                                                        round(
                                                                                                        128),
                                                                                                        round(128),                                                                                           round(128)))
        flag = -1 * flag
        time.sleep(55)
        i += 1


def task_adjust_docker():
    i = 0
    max_iterator = 60 * 5
    flag = 1
    while i < max_iterator:
        for idx, vm in domain.items():
            if flag == 1:
                os.system("docker update --cpus {} {} > /dev/null".format(2,vm))
                os.system("docker update --memory {}M --memory-swap -1 {} > /dev/null".format(1280,vm))
                os.system(
                    'docker exec {} bash -c "tc class change dev eth0 parent 1: classid 1:21 htb rate 2Mbit ceil 2Mbit"'.format(vm))
            else:
                os.system("docker update --cpus {} {}".format(1, vm))
                os.system("docker update --memory {}M --memory-swap -1 {}".format(1024, vm))
                os.system(
                    'docker exec {} bash -c "tc class change dev eth0 parent 1: classid 1:21 htb rate 1Mbit ceil 1Mbit"'.format(vm))

        flag = -1 * flag
        time.sleep(55)
        i += 1


def main():
    fk = open("adjust_kvm.csv","w")
    writerk = csv.writer(fk)
    writerk.writerow(["TimeStmp","Iterator","CPU USAGE","MEM USAGE"])
    tg = threading.Thread(target=task_gather,args=(writerk,))
    tk = threading.Thread(target=task_adjust_kvm)
    tg.start()
    tk.start()
    tg.join()
    tk.join()
    fk.close()

    fd = open("adjust_docker.csv","w")
    writerd = csv.writer(fd)
    writerd.writerow(["TimeStmp", "Iterator", "CPU USAGE", "MEM USAGE"])
    tg = threading.Thread(target=task_gather, args=(writerd,))
    td = threading.Thread(target=task_adjust_docker)
    tg.start()
    td.start()
    tg.join()
    td.join()
    fd.close()


if __name__ == '__main__':
    main()
