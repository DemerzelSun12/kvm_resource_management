import numpy as np
import os
import re
import sys
#import libvirt
import psutil
import socket
import time
import subprocess
from xml.etree import ElementTree


class Env():
    def __init__(self, vm_num):
        self.vm_num = vm_num
        self.observations = {}  # 观测值
        self.pre_observations = {}  # 之前的观测值
        self.pre_variance = 0.0  # 调节前的方差
        self.variance = 0.0  # 调节后的方差
        self.bad_state_vm = 1  # 处于不良状态的vm数量
        self.correct_mem = 0  # 需要矫正的内存
        self.over_max_correct_mem = 0  # 超过最大mem的机器数量
        self.done = 0  # 是否结束
        self.bad_state_vm_pre = 0  # 处于bad_state虚拟机的数量比例

        self.action = []  # 动作
        self.step = 0  # 用于记录step,便于存储

        self.vm_state = []  # 虚拟机的状态,一共6维

        self.vm_curr_mem = [2048] * self.vm_num  # 保存内存的数量的标志
        self.vm_curr_cpu = [4] * self.vm_num  # 当前cpu数量
        self.vm_curr_bandwith = [1] * self.vm_num  # 初始带宽为1M

        self.cpu_payload = [0]*self.vm_num
        self.mem_payload = [0]*self.vm_num
        self.bandwidth_payload = [0]*self.vm_num

        self.vm_sum_mem = 8192 * self.vm_num  # 所以虚拟机所有的内存大小
        self.vm_sum_cpu = 12 * self.vm_num  # 所有可以分配的VCPU总和,不会使用到
        self.vm_sum_bandwith = 100  # 网卡的最大带宽

        self.vm_mem_usage_bound = [30, 90]  # 利用率的范围,超过或低于,都属于bad-state
        self.vm_cpu_usage_bound = [30, 90]  # cpu利用率范围,超过或低于都属于,bad-state
        # 对于带宽没有必要设置这一个阈值,

        self.vm_down_mem = 0  # 每次下调的内存总数量
        self.vm_up_count = 0  # 需要上调内存的机器数量
        self.vm_up_percent = 0.0  # 总的上调的比例


        self.vm_max_mem = 8192  # VM最大mem大小
        self.vm_min_mem = 2048
        self.vm_max_bandwith = 50
    
        self.vm_min_cpu = 1
        self.vm_max_cpu = 12  # 最大的cpu数量

        self.vm_min_bandwith = 0.5

        self.vm_bad_state_count = 0  # 处于不好状态的虚拟机的数量
        self.ok_time = 0

        # 惩罚系数:
        self.l1 = 1
        self.l2 = 1
        self.l3 = 1

        self.down_mem_per_train = 0

        # 连接数据库
        #self.conn = libvirt.open("qemu:///system")
        # self.conn = pymysql.connect('101.200.218.87','root','Asdf1234!@#$')
        # self.conn.select_db("resource_vm")
        # self.cur = self.conn.cursor()

        # 虚拟机的domain名
        #self.vm_domain = {1: "ubuntu18.04-05", 2: "ubuntu18.04-06", 3: "ubuntu18.04-07", 4: "ubuntu18.04-08",
        #				  5: "ubuntu18.04-09",
        #				  6: "ubuntu18.04-10", 7: "ubuntu18.04-11", 8: "ubuntu18.04-12", 9: "ubuntu18.04-13",
        #				  10: "ubuntu18.04-14",
        #				  11: "ubuntu18.04-15", 12: "ubuntu18.04-16", 13: "ubuntu18.04-17", 14: "ubuntu18.04-18",
        #				  15: "ubuntu18.04-19", 16: "ubuntu18.04-20"}
        #self.ip = {1: "192.168.122.195", 2: "192.168.122.196", 3: "192.168.122.197", 4: "192.168.122.198",
        #		   5: "192.168.122.199",
        #		   6: "192.168.122.200", 7: "192.168.122.201", 8: "192.168.122.202", 9: "192.168.122.203",
        #		   10: "192.168.122.204",
        #		   11: "192.168.122.205", 12: "192.168.122.206", 13: "192.168.122.207", 14: "192.168.122.208",
        #		   15: "192.168.122.209", 16: "192.168.122.210"}

        #self.vm_mac = {1: "52:54:00:6d:41:af", 2: "52:54:00:4a:85:98", 3: "52:54:00:5e:b1:c5", 4: "52:54:00:d6:97:c0",
        #			   5: "52:54:00:2e:54:3e", 6: "52:54:00:8f:44:6a", 7: "52:54:00:36:0f:5f", 8: "52:54:00:4b:a7:fd",
        #			   9: "52:54:00:01:67:78", 10: "52:54:00:23:3c:b8",
        #			   11: "52:54:00:32:9d:5c", 12: "52:54:00:a7:3c:93", 13: "52:54:00:3f:e5:dd",
        #			   14: "52:54:00:68:b6:fb", 15: "52:54:00:c2:26:b6", 16: "52:54:00:f8:9e:df"}

        #self.netcards = {1: "vnet5", 2: "vnet4", 3: "vnet6", 4: "vnet19", 5: "vnet8", 6: "vnet9", 7: "vnet10",
        #				 8: "vnet11", 9: "vnet12", 10: "vnet13", 11: "vnet7", 12: "vnet14", 13: "vnet15", 14: "vnet16",
        #				 15: "vnet17", 16: "vnet18"}

    """
    def subprocess_popen(statement):
        a = psutil.virtual_memory()
        result = {}
        p = subprocess.Popen(statement, shell=True, stdout=subprocess.PIPE)  # 执行shell语句并定义输出格式
        while p.poll() is None:  # 判断进程是否结束（Popen.poll()用于检查子进程（命令）是否已经执行结束，没结束返回None，结束后返回状态码）
            if p.wait() is not 0:  # 判断是否执行成功（Popen.wait()等待子进程结束，并返回状态码；如果设置并且在timeout指定的秒数之后进程还没有结束，将会抛出一个TimeoutExpired异常。）
                print("命令执行失败，请检查设备连接状态")
                return False
            else:
                re = p.stdout.readlines()  # 获取原始执行结果
                for i in range(len(re)):  # 由于原始结果需要转换编码，所以循环转为utf8编码并且去除\n换行
                    res = re[i].decode('utf-8').strip('\r\n')
                    result[i + 1] = res
                return result
        return result
    """
    def createFixPayload(self):
        noise = np.random.randint(-2,2,15)/15
        self.mem_payload = list((np.array(
            [1.5,1.4,0.87, 0.65, 0.45, 0.56, 0.4, 0.56, 0.55, 0.68, 0.51, 0.42, 0.41, 0.7, 0.50])+noise) * self.vm_curr_mem[0])
        noise_cpu = np.random.randint(0,1,15)
        self.cpu_payload = list(np.array([3,3,6,5,4,5,4,5,4,4,5,4,4,6,5])+ noise_cpu)
        noise_bandwidth = np.random.randint(1,2,15)
        self.bandwidth_payload = list(np.array([1,2,2,1,2,2,1,8,8,7,5,6,10,12,14])+noise_bandwidth)
        print("This time payload is")
        print("Mem payload:",self.mem_payload)
        print("Cpu payload:",self.cpu_payload)
        print("Band payload:", self.bandwidth_payload)

    def createPayload(self):
        for i in range(self.vm_num):
            if np.random.rand() > 0.6:
                self.mem_payload[i] = np.random.randint(int(self.vm_curr_mem[i]*0.8),round(self.vm_curr_mem[i]*1.7))
            else:
                self.mem_payload[i] = np.random.randint(int(self.vm_curr_mem[i]*0.3),int(self.vm_curr_mem[i]*0.79))

            if np.random.rand() > 0.4:
                self.cpu_payload[i] = np.random.randint(2,7)
            else:
                self.cpu_payload[i] = np.random.randint(0,4)
            self.bandwidth_payload[i] = np.random.randint(0,int(self.vm_curr_bandwith[i]*3))
        print("This time payload is")
        print("Mem payload:",self.mem_payload)
        print("Cpu payload:",self.cpu_payload)
        print("Bandwidth payload",self.bandwidth_payload)

    def init(self):
        self.gather()  # 获取新的状态
        print("+++++++++Init Successful++++++++++++")
        return np.array(self.vm_state)

    def startNewGame(self):
        self.vm_curr_mem = [1300] * self.vm_num
        self.vm_curr_cpu = [2] * self.vm_num
        #print("self.vm_num",self.vm_num,"len(cpu)",len(self.vm_curr_cpu))
        self.vm_curr_bandwith = [4] * self.vm_num

        #self.createFixPayload()
        self.createPayload()
        self.pre_observations = self.observations
        print("+++++++++startNewGame Successful++++++++++++")

    def reset(self):  # 我们需要返回一个东state的numpy数组
        self.vm_curr_mem = [1300] * self.vm_num
        self.vm_curr_cpu = [2] * self.vm_num
        self.vm_curr_bandwith = [4] * self.vm_num
        #self.createFixPayload()
        self.createPayload()
        self.pre_observations = self.observations
        print("+++++++++Reset Successful++++++++++++")

        #time.sleep(1)
        self.gather()  # 获取新的状态
        return np.array(self.vm_state)

    # [[x1,x2,x3,...],[x1,x2,....]] 共10条记录 10* feature_size

    def getCpuInfo(self):
        cpu_usage = [0]* self.vm_num
        for i in range(self.vm_num):
            cpu_usage[i] = self.cpu_payload[i] / self.vm_curr_cpu[i]
            if cpu_usage[i] > 1:
                cpu_usage[i] = 1 - np.random.rand() / 10 
        return cpu_usage

    def getMemInfo(self):
        mem_usage = [0]*self.vm_num
        for i in range(self.vm_num):
            mem_usage[i] = (self.mem_payload[i]/self.vm_curr_mem[i])
            if mem_usage[i] > 1:
                mem_usage[i]  = 1-np.random.rand() / 10
        return mem_usage

    def getBandwithInfo(self):
        bandwith_usage = [0]*self.vm_num
        for i in range(self.vm_num):
            bandwith_usage[i] = (self.bandwidth_payload[i]/self.vm_curr_bandwith[i])
            if bandwith_usage[i] > 1:
                bandwith_usage[i] = 1 - np.random.rand() / 4
        return bandwith_usage

    def gather(self):
        # 首先等待所有变量全部变为1
        self.vm_state = []

        cpu_nums = self.vm_curr_cpu
        mem_nums = self.vm_curr_mem
        bandwith_nums = self.vm_curr_bandwith
        

        cpu_usage = self.getCpuInfo()
        mem_usage = self.getMemInfo()
        bandwith_usage = self.getBandwithInfo()
        # print("cpu_nums:{}, cpu_usage:{}".format(cpu_nums, cpu_usage))
        # print("mem_nums:{}, mem_usage:{}".format(mem_nums, mem_usage))
        # print("bandwith_nums, bandwith_usage:{}".format(bandwith_nums, bandwith_usage))


        #print(len(cpu_usage),len(cpu_nums),len(mem_nums),len(mem_usage),len(bandwith_nums),len(bandwith_usage))
        #print("[TEST] ",cpu_usage)
        #print("[TEST] ",bandwith_usage)
        # 数据汇总以及归一化.
        observation = [[cpu_usage[i], cpu_nums[i] / self.vm_max_cpu, mem_usage[i], mem_nums[i] / self.vm_max_mem,bandwith_usage[i], bandwith_nums[i] / self.vm_max_bandwith] for i in range(self.vm_num)]

        self.vm_state = list(np.array(observation).flatten("F"))
        for i in range(self.vm_num):
            self.observations[i] = observation[i]
        #print("[TEST] ", observation)


    def clear(self):
        self.pre_variance = 0.0  # 调节前的方差
        self.variance = 0.0  # 调节后的方差
        self.bad_state_vm = 0  # 处于不良状态的vm数量
        self.bad_state_vm_pre = 0  # 上一次处于不良状态的VM数量
        self.correct_mem = 0  # 需要矫正的内存
        self.over_max_correct_mem = 0  # 超过最大mem的机器数量
        self.done = 0  # 是否结束
        self.vm_down_mem = 0  # 每次下调的内存总数量
        self.vm_up_count = 0  # 需要上调内存的机器数量
        self.vm_up_percent = 0.0  # 总的上调的比例

    def execute_cpu(self, op, vm_id):
        vm_domain = self.vm_domain[vm_id + 1]
        cpu_num = self.vm_curr_cpu[vm_id]
        # 增加cpu
        if op == 1 and cpu_num < self.vm_max_cpu:
            try:
                ret = os.system("virsh setvcpus --live --guest {} {} > /dev/null".format(vm_domain, cpu_num + 1))
            except:
                ret = -1

            if ret == 0:
                self.vm_curr_cpu[vm_id] += 1
                print("VM{} UP CPU TO {}".format(vm_id + 1, self.vm_curr_cpu[vm_id]), flush=True)
            if ret != 0:
                print("VM {} CPU UP ERROR".format(vm_id + 1), flush=True)
            # self.execute_cpu(op,vm_id)

        elif op == 2 and cpu_num > 1:
            try:
                ret = os.system("virsh setvcpus --live --guest {} {} > /dev/null".format(vm_domain, cpu_num - 1))
            except:
                ret = -1
            if ret == 0:
                self.vm_curr_cpu[vm_id] -= 1
                print("VM {} DOWN CPU TO {}".format(vm_id + 1, self.vm_curr_cpu[vm_id]), flush=True)
            else:
                print("VM {} CPU DOWN ERROR".format(vm_id + 1), flush=True)
        else:
            print("Operation Error!")

    # 执行函数,调节bandwidth
    def execute_bandwith(self, op, vm_id, percent):
        vm_domain = self.vm_domain[vm_id + 1]
        mac = self.vm_mac[vm_id + 1]
        band_size = self.vm_curr_bandwith[vm_id]
        newband_size = band_size
        # 如果是增加带宽
        if op == 1:
            newband_size = (1 + percent) * newband_size
            os.system("virsh domiftune {} {} --live --config --inbound {} > /dev/null".format(vm_domain, mac,
                                                                                                    round(
                                                                                                        newband_size * 128)))
            os.system("virsh domiftune {} {} --live --config --outbound {} > /dev/null".format(vm_domain, mac,
                                                                                                     round(
                                                                                                         newband_size * 128)))
            self.vm_curr_bandwith[vm_id] = newband_size
            print("VM {} UP BANDWIDTH {} TO {}".format(vm_id + 1, band_size, self.vm_curr_bandwith[vm_id]), flush=True)
        elif op == 2:
            newband_size = (-percent) * newband_size
            if newband_size > self.vm_min_bandwith:
                os.system(
                    "virsh domiftune {} {} --live --config --inbound {} > /dev/null".format(vm_domain, mac, round(
                        newband_size * 128)))
                os.system(
                    "virsh domiftune {} {} --live --config --outbound {} > /dev/null".format(vm_domain, mac,
                                                                                                   round(
                                                                                                       newband_size * 128)))
                self.vm_curr_bandwith[vm_id] = newband_size
                print("VM {} DOWN BANDWIDTH {} TO {}".format(vm_id + 1, band_size, self.vm_curr_bandwith[vm_id]),
                      flush=True)
        else:
            print("Op Error!!")

    # 执行函数
    def execute_mem(self, op, vm_id, percent):
        vm_domain = self.vm_domain[vm_id + 1]  # 获取虚拟机的域名
        mem_size = self.vm_curr_mem[vm_id]  # 获取对应虚拟机的mem大小
        newmem_size = mem_size
        # 增加内存
        if op == 1 and self.vm_up_count != 0:
            newmem_size = round(mem_size + self.vm_down_mem * (percent / self.vm_up_percent))
        elif op == 1:
            pass
        else:
            newmem_size = round(abs(mem_size * (-percent)))

        if op == 1 and newmem_size <= self.vm_max_mem and self.vm_up_count != 0:
            print("VM{} UP MEM {}MB To {}MB".format(vm_id + 1, mem_size, newmem_size), flush=True)
            val = os.system("virsh setmem {} --size {}MB > /dev/null".format(vm_domain, newmem_size))
            self.vm_curr_mem[vm_id] = newmem_size

        elif op == 1 and newmem_size > self.vm_max_mem and self.vm_up_count != 0:
            self.correct_mem += newmem_size - self.vm_max_mem
            # 防止内存泄露
            print("VM{} UP Max MEM {}MB To {}MB".format(vm_id + 1, mem_size, self.vm_max_mem), flush=True)
            # print("vm id:{}; mem_size:{}MB ; New mem_size:{}MB".format(vm_id,mem_size,newmem_size))
            val = os.system("virsh setmem {} --size {}MB > /dev/null".format(vm_domain, self.vm_max_mem))
            self.vm_curr_mem[vm_id] = self.vm_max_mem

        elif op == 2 and percent < 0 and mem_size * (-percent) > self.vm_min_mem:
            # 如果为减少,则说明该值为负数,那么我们就按照比例进行减小即可
            # print("vm id:{}; mem_size:{}MB ; New mem_size:{}MB".format(vm_id,mem_size,newmem_size))
            print("VM{} DOWN MEM {}MB To {}MB".format(vm_id + 1, mem_size, round(abs(mem_size * (-percent)))),
                  flush=True)
            val = os.system(
                "virsh setmem {} --size {}MB > /dev/null".format(vm_domain, round(abs(mem_size * (-percent)))))
            self.vm_down_mem += (
                        self.vm_curr_mem[vm_id] - round(abs(mem_size * (-percent))))  # round(mem_size*abs(1+percent))
            self.vm_curr_mem[vm_id] = round(abs(mem_size * (-percent)))
        # 有一个问题，如果处于bad_state的机器是无法进行分配的,这个如果是bad_state应当将其从
        # 没有人可以增加时,就没有机器去获取,所以需要修正
        elif op == 3:
            # 做内存更正用的
            self.correct_mem = 0
            if self.vm_up_count == 0:
                # 没有需要增加内存的机器，我们就需要修增
                self.correct_mem += (self.vm_down_mem + self.over_max_correct_mem)
                self.over_max_correct_mem = 0
            self.vm_down_mem = 0
            print("!!! correct: --------------------------", flush=True)
            for i in self.vm_domain.items():  # 遍历每一个虚拟机
                if self.vm_curr_mem[i[0] - 1] + round(self.correct_mem / self.vm_num) < self.vm_max_mem:
                    self.vm_curr_mem[i[0] - 1] += round(self.correct_mem / self.vm_num)
                else:
                    self.over_max_correct_mem += self.vm_curr_mem[i[0] - 1] + round(
                        self.correct_mem / self.vm_num) - self.vm_max_mem
                    self.vm_curr_mem[i[0] - 1] = self.vm_max_mem
            diff = self.vm_sum_mem - sum(self.vm_curr_mem)
            for i in self.vm_domain.items():
                if diff > 100 and (self.vm_curr_mem[i[0] - 1] + round(diff / self.vm_num)) < self.vm_max_mem:
                    self.vm_curr_mem[i[0] - 1] += round(diff / self.vm_num)
                    if i[0] == 1:
                        print("memory too small need to add some!", flush=True)
                elif diff < -100 and (self.vm_curr_mem[i[0] - 1] - round(diff / self.vm_num)) >= 250:
                    self.vm_curr_mem[i[0] - 1] -= round(diff / self.vm_num)
                    if i[0] == 1:
                        print("memory too big need to sub some!", flush=True)
                # print("vm id:{};mem_size:{}MB".format(i[0],self.curr_mem[i[0]-1]))
                os.system("virsh setmem %s --size %dMB >/dev/null" % (i[1], self.vm_curr_mem[i[0] - 1]))
            diff = 0
            self.correct_mem = 0

    def checkHealthy(self, vm_id=0):
        if (self.observations[vm_id][0] < 0.85 and self.observations[vm_id][2] < 0.8 and self.observations[vm_id][
            4] < 0.8):
            return True
        else:
            return False

    def waitForNextLoop(self):
        while True:
            self.gather()
            if (self.checkHealthy()):
                time.sleep(20)
                continue
            else:
                break

    def mem_correct(self):
        if(self.vm_up_count == 0): #如果此次没有增加内存的action,则平
            tmp_mem = self.down_mem_per_train/self.vm_num
            for i in range(self.vm_num):
                if(self.vm_curr_mem[i] + tmp_mem) >= self.vm_max_mem:
                    self.vm_curr_mem[i] = self.vm_max_mem
                else:
                    self.vm_curr_mem[i] += tmp_mem

        #然后要分配相应的self.mem_correct
        if abs(sum(self.vm_curr_mem)-self.vm_sum_mem)  > 100:
            tmp_mem = (self.vm_sum_mem - sum(self.vm_curr_mem)) / self.vm_num
            for j in range(self.vm_num):
                if(self.vm_curr_mem[j] + tmp_mem) >= self.vm_min_mem and self.vm_curr_mem[j]+tmp_mem <= self.vm_max_mem:
                    self.vm_curr_mem[j] += tmp_mem


    def steps(self, actions, mode="Train"):
        self.done = False
        #print("actions:", actions)
        self.step += 1
        # print("sum Mem:{}".format(sum(self.vm_curr_mem)),flush=True)
        sum_up_percent = 0.0
        action_cpu = actions[:self.vm_num]
        action_mem = actions[self.vm_num:self.vm_num * 2]
        action_bandwith = actions[self.vm_num * 2:]

        self.clear()  # 清空一些变量

        bad_action_mem = [0] * self.vm_num  # 做出了不好的行动
        bad_action_cpu = [0] * self.vm_num
        bad_action_bandwith = [0] * self.vm_num

        bad_state_mem = [0] * self.vm_num  # vm处于不好的状态
        bad_state_cpu = [0] * self.vm_num
        bad_state_bandwith = [0] * self.vm_num

        reword_mem = [0] * self.vm_num
        reword_cpu = [0] * self.vm_num
        reword_bandwith = [0] * self.vm_num  # 奖励
        self.down_mem_per_train = 0
        self.vm_up_percent = 0
        self.vm_up_count = 0

        # step1:先执行释放操作
        # action永远是-1 ~ 1之间
        for i in range(self.vm_num):
            if action_mem[i] < 0 and self.observations[i][2] <= 0.75 and self.vm_curr_mem[i] * (-action_mem[i]) >= self \
                    .vm_min_mem:
                #self.execute_mem(2, i, action_mem[i]) #就是把curr_mem给替换即可
                self.down_mem_per_train += self.vm_curr_mem[i] * (1+action_mem[i])
                self.vm_curr_mem[i] = self.vm_curr_mem[i] * (-action_mem[i])
            elif action_mem[i] > 0 and self.observations[i][2] > 0.8 and self.vm_curr_mem[i] <= self.vm_max_mem:
                self.vm_up_percent += action_mem[i]  # 统计总共增加的百分比
                self.vm_up_count += 1
            elif action_mem[i] == 0 and self.observations[i][2] < 0.75 and self.observations[i][2] > 0.65:
                pass
            else:
                pass

            #cpu down ---------------
            if action_cpu[i] < 0 and self.observations[i][0] <= 0.6 and self.vm_curr_cpu[i] - 1 >= self.vm_min_cpu:
                #self.execute_cpu(2, i)
                self.vm_curr_cpu[i] -= 1
            elif action_cpu[i] > 0 and self.observations[i][0] > 0.65 and self.vm_curr_cpu[i] <= self.vm_max_cpu:
                pass
            else:
                pass

            if action_bandwith[i] < 0 and self.observations[i][4] <= 0.75 and self.vm_curr_bandwith[i] * (
            -action_bandwith[i]) >= self.vm_min_bandwith:
                #self.execute_bandwith(2, i, action_bandwith[i])
                self.vm_curr_bandwith[i] = self.vm_curr_bandwith[i] * (-action_bandwith[i])
            elif action_bandwith[i] > 0 and self.observations[i][4] > 0.75 and self.vm_curr_bandwith[i] * (
                    1 + action_bandwith[i]) < self.vm_sum_bandwith:
                pass
            else:
                pass

        # step2:再执行增加
        # 这里我们需要进行实验,当内存利用率达到多少时会开始使用swap
        # 再执行增加
        for i in range(self.vm_num):
            ## 执行mem增操作
            if action_mem[i] > 0 and self.observations[i][2] > 0.8 and self.vm_curr_mem[i] <= self.vm_max_mem:
                #self.execute_mem(1, i, action_mem[i])
                # print("++++ VM:{} UP MEM{} MB mem_usage:{}% ++++".format(i + 1,self.vm_curr_mem[i],self.observations[i][2]),flush=True)
                tmp_mem = self.down_mem_per_train * (action_mem[i]/self.vm_up_percent)
                if self.vm_curr_mem[i] + tmp_mem > self.vm_max_mem:
                    self.vm_curr_mem[i] = self.vm_max_mem
                else:
                    self.vm_curr_mem[i] += tmp_mem

            elif action_mem[i] < 0 and self.observations[i][2] < 0.75 and self.vm_curr_mem[i] >= self.vm_min_mem:
                pass
            elif action_mem[i] == 0 and self.observations[i][2] < 0.75 and self.observations[i][2] > 0.4:
                pass
            else:
                bad_action_mem[i] = 1
                reword_mem[i] = -1  # 这个是

            ## 执行cpu增操作
            if action_cpu[i] > 0 and self.observations[i][0] > 0.65 and self.vm_curr_cpu[i] + 1 <= self.vm_max_cpu:
                self.vm_curr_cpu[i] += 1
            elif action_cpu[i] < 0 and self.observations[i][0] < 0.5 and self.vm_curr_cpu[i] >= 1:
                pass
            elif action_cpu[i] == 0 and self.observations[i][0] < 0.8 and self.observations[i][0] > 0.3:
                pass
            else:
                bad_action_cpu[i] = 1
                reword_cpu[i] = -1
            ## 执行bandwith增操作:

            if action_bandwith[i] > 0 and self.observations[i][4] > 0.75 and self.vm_curr_bandwith[i] * (
                    1 + action_bandwith[i]) < self.vm_max_bandwith:
                #self.execute_bandwith(1, i, action_bandwith[i])
                if self.vm_curr_bandwith[i] * (1 + action_bandwith[i]) > self.vm_max_bandwith:
                    self.vm_curr_bandwith[i] = self.vm_max_bandwith
                else:
                    self.vm_curr_bandwith[i]  = self.vm_curr_bandwith[i] * (1 + action_bandwith[i])

            # print("++++ VM:{} UP BANDWITH {} bandwith_usage:{}% ++++".format(i+1,self.vm_curr_bandwith[i],self.observations[i][4]),flush=True)
            elif action_bandwith[i] < 0 and self.observations[i][4] < 0.75 and self.vm_curr_bandwith[
                i] >= self.vm_min_bandwith:
                pass
            elif action_bandwith[i] == 0 and self.observations[i][4] > 0.5 and self.observations[i][4] < 0.75:
                pass
            else:
                bad_action_bandwith[i] = 1
                #reword_bandwith[i] = -3

        ## 执行内存矫正,避免内存泄露
        self.mem_correct() # 校正内存,防止内存泄露
        # 执行完操作后,保存原状态,获取当前新状态
        self.pre_observations = self.observations
        self.gather()  # 获取新的状态

        # 第一部分: 方差
        # 第二部分: bad num
        # 第三部分: swap的利用率
        # print(np.array(self.pre_observations.values()))
        pre_variance_mem = (np.array(list(self.pre_observations.values()))[:, 2]).std()
        variance_mem = (np.array(list(self.observations.values()))[:, 2]).std()
        reword_mem_var = (self.l1 * (pre_variance_mem - variance_mem))

        pre_variance_cpu = (np.array(list(self.pre_observations.values()))[:, 0]).std()
        variance_cpu = (np.array(list(self.observations.values()))[:, 0]).std()
        reword_cpu_var = (self.l2 * (pre_variance_cpu - variance_cpu))

        pre_variance_bandwith = (np.array(list(self.pre_observations.values()))[:, 4]).std()
        variance_bandwith = (np.array(list(self.observations.values()))[:, 4]).std()
        reword_bandwith_var = (self.l3 * (pre_variance_bandwith - variance_bandwith))

        # step3:计算reword()
        self.bad_state_vm_pre = 0
        self.bad_state_vm = [0] * self.vm_num


        for i in range(self.vm_num):
            flag = [0]*3
            # 首先计算内存
            if self.pre_observations[i][2] > 0.95 or self.pre_observations[i][0] > 0.95:
                self.bad_state_vm_pre += 1

            if self.observations[i][2] > 0.95 and self.vm_curr_mem[i] < self.vm_max_mem:
                self.bad_state_vm[i] = 1
                reword_mem[i] += -6  # 我们的目标就是调节到 bad_state_vm为0 即可了
                #print("THE VM {} MEM IN BAD STATE!!".format(i))
                sys.stdout.flush()

            # 1.虽然mem利用率下降,但是调节了bad_state
            if self.pre_observations[i][2] >= self.observations[i][2] and self.pre_observations[i][2] > 0.8:
                reword_mem[i] += -0.3  # 这是个优秀的操作

            # 2.mem利用率下降,且超过合理范围
            elif self.pre_observations[i][2] >= self.observations[i][2] and self.observations[i][2] < 0.45:
                reword_mem[i] += -4

            # 3.mem利用率下降,且没调节bad_state且在合理范围.
            elif self.pre_observations[i][2] >= self.observations[i][2]:
                reword_mem[i] += -0.8

            # 4.mem利用率上升,并且在合理范围内
            if self.pre_observations[i][2] < self.observations[i][2] and self.observations[i][0] < 0.75:
                reword_mem[i] += -0.6

            # 4.如果利用率上述,且不在合理范围内,那么就应该给予惩罚
            elif self.pre_observations[i][2] < self.observations[i][2]:
                reword_mem[i] += -4

            # cpu--------------------------
            if self.observations[i][0] > 0.75 and self.vm_curr_cpu[i] < self.vm_max_cpu:
                self.bad_state_vm[i] = 1
                reword_cpu[i] += -18  # 我们的目标就是调节到 bad_state_vm为0 即可了
                #print("THE VM {} CPU IN BAD STATE!!".format(i))
                sys.stdout.flush()

            # 1.虽然cpu利用率下降,但是调节了bad_state
            if self.pre_observations[i][0] >= self.observations[i][0] and self.pre_observations[i][0] > 0.75:
                reword_cpu[i] += -0.3  # 这是个优秀的操作

            # 2.cpu利用率下降,且超过合理范围
            elif self.pre_observations[i][0] >= self.observations[i][0] and self.observations[i][0] < 0.4:
                reword_cpu[i] += -0.9

            # 3.cpu利用率下降,且没调节bad_state且在合理范围.
            elif self.pre_observations[i][0] >= self.observations[i][0]:
                reword_cpu[i] += -0.7

            # 4.cpu利用率上升,并且在合理范围内
            if self.pre_observations[i][0] < self.observations[i][0] and self.observations[i][0] < 0.65:
                reword_cpu[i] += -1

            # 5.如果利用率上述,且不在合理范围内,那么就应该给予惩罚
            elif self.pre_observations[i][0] < self.observations[i][0]:
                reword_cpu[i] += -4

            # 带宽----------------------------------------------
            if self.observations[i][4] > 0.9 and self.vm_curr_bandwith[i] < self.vm_max_bandwith:
                self.bad_state_vm[i] = 1
                reword_bandwith[i] += -6  # 我们的目标就是调节到 bad_state_vm为0 即可了
                #print("THE VM {} BANDWIDTH IN BAD STATE!!".format(i))
                sys.stdout.flush()

            # 1.虽然带宽利用率下降,但是调节了bad_state
            if self.pre_observations[i][4] >= self.observations[i][4] and self.pre_observations[i][4] > 0.7:
                reword_bandwith[i] += -0.3  # 这是个优秀的操作

            # 2.band利用率下降,且超过合理范围
            elif self.pre_observations[i][4] >= self.observations[i][4] and self.observations[i][4] < 0.4:
                reword_bandwith[i] += -3

            # 3.band利用率下降,且没调节bad_state且在合理范围.
            elif self.pre_observations[i][4] >= self.observations[i][4]:
                reword_bandwith[i] += -0.5

            # 4.band利用率上升,并且在合理范围内
            if self.pre_observations[i][4] < self.observations[i][4] and self.observations[i][4] < 0.6:
                reword_bandwith[i] += -0.8
            # 5.如果利用率上述,且不在合理范围内,那么就应该给予惩罚
            elif self.pre_observations[i][4] < self.observations[i][4]:
                reword_bandwith[i] += -4
            else:
                reword_bandwith[i] += -0.5

        # print("old feature"); print(self.pre_observations)
        # print("feature");print(self.observations)

        reword_mem_sum = sum(reword_mem) + reword_mem_var
        reword_cpu_sum = sum(reword_cpu) + reword_cpu_var
        reword_bandwith_sum = sum(reword_bandwith) + reword_bandwith_var

        if self.step % 100 == 1 or self.bad_state_vm == 0:
            print("[INFO] Step{}".format(self.step), flush=True)
            print("[INFO] Reword:{},Sum_mem:[{}/{}],Sum cpus:{},Sum bandwidth:{} +++".format(
                reword_bandwith_sum + reword_mem_sum + reword_cpu_sum, sum(self.vm_curr_mem), self.vm_sum_mem,
                sum(self.vm_curr_cpu), sum(self.vm_curr_bandwith)))
            # print("current mem:",self.vm_curr_mem)
            print("[INFO] Now Bad VM num is [{}/{}] +++".format(sum(self.bad_state_vm), self.vm_num))
            # print("cpu",self.observations)
            print("-----------------------------------------------------------")
        #time.sleep(1)
        reword_sum = reword_mem_sum + reword_cpu_sum + 0.4 * reword_bandwith_sum  # 暂时不算bandwith
        if sum(self.bad_state_vm) == 0:
            reword_sum = reword_sum * 0.75
            self.ok_time += 1

        if self.ok_time >= 1:
            self.done = True
            self.ok_time = 0
        else:
            self.done = False

        if mode == "Train":
            return np.array(self.vm_state), np.array(reword_sum), self.done
        elif (self.checkHealthy()):
            return np.array(self.vm_state), np.array(reword_sum), True
        else:
            return np.array(self.vm_state), np.array(reword_sum), False

