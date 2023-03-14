import numpy as np
import os
import re
import sys
import libvirt
import psutil
import socket
import time
import subprocess
from xml.etree import ElementTree


class Env():
    def __init__(self, vm_num, writer):
        self.observations = {}  # 观测值
        self.pre_observations = {}  # 之前的观测值
        self.pre_variance = 0.0  # 调节前的方差
        self.variance = 0.0  # 调节后的方差
        self.bad_state_vm = 1  # 处于不良状态的vm数量
        self.correct_mem = 0  # 需要矫正的内存
        self.over_max_correct_mem = 0  # 超过最大mem的机器数量
        self.done = 0  # 是否结束
        self.bad_state_vm_pre = 0  # 处于bad_state虚拟机的数量比例
        self.writer = writer

        self.action = []  # 动作
        self.step = 1  # 用于记录step,便于存储

        self.vm_state = []  # 虚拟机的状态,一共6维
        self.vm_num = vm_num  # 虚拟机数量

        self.vm_curr_mem = [2048] * self.vm_num  # 保存内存的数量的标志
        self.vm_curr_cpu = [4] * self.vm_num  # 当前cpu数量
        self.vm_curr_bandwith = [1] * self.vm_num  # 初始带宽为1M

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
        self.vm_min_cpu = 2
        self.vm_max_cpu = 12  # 最大的cpu数量

        self.vm_min_bandwith = 4

        self.vm_bad_state_count = 0  # 处于不好状态的虚拟机的数量

        # 惩罚系数:
        self.l1 = 1
        self.l2 = 1
        self.l3 = 1

        self.ok_time = 0

        # 连接数据库
        self.conn = libvirt.open("qemu:///system")
        # self.conn = pymysql.connect('101.200.218.87','root','Asdf1234!@#$')
        # self.conn.select_db("resource_vm")
        # self.cur = self.conn.cursor()

        # 虚拟机的domain名
        self.vm_domain = {1: "vm1", 2: "vm2", 3: "vm3", 4: "vm4", 5: "vm5", 6: "vm6", 7: "vm7", 8: "vm8", 9: "vm9", 10: "vm10", 11: "vm11", 12: "vm12", 13: "vm13", 14: "vm14", 15: "vm15"}
        self.ip = {1: "192.168.122.22", 2: "192.168.122.133", 3: "192.168.122.137", 4: "192.168.122.103", 5: "192.168.122.201", 6: "192.168.122.2", 7: "192.168.122.236", 8: "192.168.122.223", 9: "192.168.122.211", 10: "192.168.122.81", 11: "192.168.122.53", 12: "192.168.122.12", 13: "192.168.122.3", 14: "192.168.122.217", 15: "192.168.122.248"}

        self.vm_mac = {1: "52:54:00:23:39:ce", 2: "52:54:00:cc:d4:6a", 3: "52:54:00:ad:01:be", 4: "52:54:00:a4:b5:e9", 5: "52:54:00:54:02:d6", 6: "52:54:00:b6:09:68", 7: "52:54:00:da:31:d3", 8: "52:54:00:ee:7b:52", 9: "52:54:00:21:3c:0c", 10: "52:54:00:27:30:eb", 11: "52:54:00:ca:55:94", 12: "52:54:00:f3:3f:b5", 13: "52:54:00:9f:f6:de", 14: "52:54:00:be:11:9e", 15: "52:54:00:b6:5e:31"}

        self.netcards = {1: "vnet0", 2: "vnet1", 3: "vnet2", 4: "vnet3", 5: "vnet4", 6: "vnet5", 7: "vnet6", 8: "vnet7", 9: "vnet8", 10: "vnet9", 11: "vnet10", 12: "vnet11", 13: "vnet12", 14: "vnet12", 15: "vnet14"}

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

    def init(self):
        # self.vm_mac = self.subprocess_popen("./getmac.sh")
        self.gather()  # 获取新的状态
        return np.array(self.vm_state)

    def reset(self):  # 我们需要返回一个东state的numpy数组
        self.vm_curr_mem = [1300] * self.vm_num
        self.vm_curr_cpu = [2] * self.vm_num
        self.vm_curr_bandwith = [4] * self.vm_num

        self.pre_observations = self.observations
        os.system("bash clean.sh >/dev/null")
        print("+++++++++Reset Successful++++++++++++")

        time.sleep(1)
        self.gather()  # 获取新的状态
        return np.array(self.vm_state)

    # [[x1,x2,x3,...],[x1,x2,....]] 共10条记录 10* feature_size

    def getCpuInfo(self):
        cpu_nums = self.vm_curr_cpu
        cpu_usage = []
        t1 = []
        c1 = []
        i = 0
        for id in self.vm_domain.values():
            domain = self.conn.lookupByName(id)
            t1.append(time.time())
            c1.append(int(domain.info()[4]))
        time.sleep(1)
        for i, id in enumerate(self.vm_domain.values()):
            domain = self.conn.lookupByName(id)
            t2 = time.time()
            c2 = int(domain.info()[4])
            # print(cpu_nums[i])
            # print(c2, c1[i], t2, t1[i], cpu_nums[i])
            usage = (c2 - c1[i]) / ((t2 - t1[i]) * cpu_nums[i] * 1e9)
            cpu_usage.append(usage)
            i += 1
        return cpu_nums, cpu_usage

    def getMemBandinfo(self):
        mems_usage = []
        bandwith_usage = []
        port = 12345
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i, ip in enumerate(self.ip.values()):
            print(ip)
            s.sendto("1".encode(encoding="utf-8"), (ip, port))
            mem_usage, _ = s.recvfrom(1024)
            net_in, _ = s.recvfrom(1024)
            net_out, _ = s.recvfrom(1024)
            mem_usage.decode(encoding="utf-8")
            net_in.decode(encoding="utf-8")
            net_out.decode(encoding="utf-8")
            mems_usage.append(float(mem_usage) / 100)
            if self.vm_curr_bandwith[i] != 0:
                bandwith_usage.append(max(float(net_in), float(net_out)) / self.vm_curr_bandwith[i])
            else:
                bandwith_usage.append(0)
        s.close()
        return mems_usage, bandwith_usage

    def getMemInfo(self):
        mem_usage = []
        for id in self.vm_domain.values():
            domain = self.conn.lookupByName(id)
            domain.setMemoryStatsPeriod(10)
            meminfo = domain.memoryStats()
            free_mem = float(meminfo['unused'])
            total_mem = float(meminfo['available'])
            used_mem = ((total_mem - free_mem) / total_mem)
            mem_usage.append(used_mem)
        return mem_usage

    def get_values(self, orders, values_dic):
        try:
            with open('/proc/net/dev') as f:
                lines = f.readlines()  # 内容不多，一次性读取较方便
                for arg in self.netcards.values():
                    for line in lines:
                        line = line.lstrip()  # 去掉行首的空格，以便下面split
                        if re.match(arg, line):
                            values = re.split("[ :]+", line)  # 以空格和:作为分隔符
                            values_dic[arg + 'r' + orders] = values[1]  # 1为接收值
                            values_dic[arg + 't' + orders] = values[9]  # 9为发送值
        except (FileExistsError, FileNotFoundError, PermissionError):
            print('open file error')
            sys.exit(-1)

    def getBandwithInfo(self):
        bandwith_nums = self.vm_curr_bandwith
        bandwith_usage = []
        values_dic = {}
        self.get_values('first', values_dic)  # 第一次取值
        time.sleep(4)
        self.get_values('second', values_dic)  # 10s后第二次取值
        for id, arg in enumerate(self.netcards.values()):
            r_bandwidth = (int(values_dic[arg + 'r' + 'second']) - int(
                values_dic[arg + 'r' + 'first'])) / 1024 / 1024 / 4
            #print(r_bandwidth)
            t_bandwidth = (int(values_dic[arg + 't' + 'second']) - int(
                values_dic[arg + 't' + 'first'])) / 1024 / 1024 / 4
            bandwith_usage.append(max(r_bandwidth, t_bandwidth) / bandwith_nums[id])
            #print(t_bandwidth)
        values_dic = {}  # 清空本次循环后字典的内容
        return bandwith_usage

    '''
    def getBandwithInfo(self):
        bandwith_nums = self.vm_curr_bandwith
        bandwith_usage = []
        i = 0
        iinfo1 = []
        iinfo2 = []
        for id in self.vm_domain.keys():
            domain = conn.lookupByID(id)
            tree = ElementTree.fromstring(domain.XMLDesc())
            ifaces = tree.findall('devices/interface/target')
            t1 = time.time()
            for j in ifaces:
                iface = j.get('dev')
                iinfo1.append(domain.inerfaceStates(iface))
            for j in  ifaces:
                iface = j.get('dev')
                iinfo2.append(domain.inerfaceStates(iface))
            t2 = time.time()
            bandtmp = min(iinfo2[i][0]-iinfo1[i][0],iinfo2[i][2]-iinfo1[i][2])/(t2-t1)/1024/1024
            bandwith_usage.append(bandtmp[i]/self.vm_curr_bandwith[i])
            i += 1
        return bandwith_nums,bandwith_usage
    '''

    def gather(self):
        # 首先等待所有变量全部变为1
        self.vm_state = []
        cpu_nums, cpu_usage = self.getCpuInfo()
        print("cpu_nums:{}, cpu_usage:{}\n".format(cpu_nums, cpu_usage))
        mem_nums = self.vm_curr_mem
        print("mem_nums:{}\n".format(mem_nums))
        mem_usage = self.getMemInfo()
        print("mem_usage:{}\n".format(mem_usage))
        bandwith_nums = self.vm_curr_bandwith
        print("bandwith_num:{}\n".format(bandwith_nums))
        bandwith_usage = self.getBandwithInfo()
        print("bandwith_usage:{}\n".format(bandwith_usage))
        if(self.step % 2 == 0):
            for i in range(self.vm_num):
                self.writer.writerow([self.step/2,i,mem_nums[i],mem_usage[i],cpu_nums[i],cpu_usage[i],bandwith_nums[i],bandwith_usage[i]])
        # mem_usage,bandwith_usage = self.getMemBandinfo()
        # mem_nums = self.vm_curr_mem
        # bandwith_nums = self.vm_curr_bandwith

        # 数据汇总以及归一化.
        observation = [[cpu_usage[i], cpu_nums[i] / self.vm_max_cpu, mem_usage[i], mem_nums[i] / self.vm_max_mem,
                        bandwith_usage[i], bandwith_nums[i] / self.vm_max_bandwith] for i in range(self.vm_num)]
        self.vm_state = list(np.array(observation).flatten("F"))
        for i in range(self.vm_num):
            self.observations[i] = observation[i]

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
                ret = os.system("sudo virsh setvcpus --live --guest {} {} > /dev/null".format(vm_domain, cpu_num + 1))
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
                ret = os.system("sudo virsh setvcpus --live --guest {} {} > /dev/null".format(vm_domain, cpu_num - 1))
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
        print("band_width_percent "+str(percent))
        vm_domain = self.vm_domain[vm_id + 1]
        mac = self.vm_mac[vm_id + 1]
        band_size = self.vm_curr_bandwith[vm_id]
        newband_size = band_size
        # 如果是增加带宽
        if op == 1:
            newband_size = (1 + abs(percent)) * newband_size
            os.system("sudo virsh domiftune {} {} --live --config --inbound {} > /dev/null".format(vm_domain, mac,
                                                                                                    round(
                                                                                                        newband_size * 1024)))
            os.system("sudo virsh domiftune {} {} --live --config --outbound {} > /dev/null".format(vm_domain, mac,
                                                                                                     round(
                                                                                                         newband_size * 1024)))
            self.vm_curr_bandwith[vm_id] = newband_size
            print("VM {} UP BANDWIDTH {} TO {}".format(vm_id + 1, band_size, self.vm_curr_bandwith[vm_id]), flush=True)
        elif op == 2:
            newband_size = (-percent) * newband_size
            if newband_size > self.vm_min_bandwith:
                os.system(
                    "sudo virsh domiftune {} {} --live --config --inbound {} > /dev/null".format(vm_domain, mac, round(
                        newband_size * 1024)))
                os.system(
                    "sudo virsh domiftune {} {} --live --config --outbound {} > /dev/null".format(vm_domain, mac,
                                                                                                   round(
                                                                                                       newband_size * 1024)))
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
            val = os.system("sudo virsh setmem {} --size {}MB > /dev/null".format(vm_domain, newmem_size))
            self.vm_curr_mem[vm_id] = newmem_size

        elif op == 1 and newmem_size > self.vm_max_mem and self.vm_up_count != 0:
            self.correct_mem += newmem_size - self.vm_max_mem
            # 防止内存泄露
            print("VM{} UP Max MEM {}MB To {}MB".format(vm_id + 1, mem_size, self.vm_max_mem), flush=True)
            # print("vm id:{}; mem_size:{}MB ; New mem_size:{}MB".format(vm_id,mem_size,newmem_size))
            val = os.system("sudo virsh setmem {} --size {}MB > /dev/null".format(vm_domain, self.vm_max_mem))
            self.vm_curr_mem[vm_id] = self.vm_max_mem

        elif op == 2 and percent < 0 and mem_size * (-percent) > 520:
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
                os.system("sudo virsh setmem %s --size %dMB >/dev/null" % (i[1], self.vm_curr_mem[i[0] - 1]))
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

    def steps(self, actions, mode="Train"):

        print("--------- step{} ---------".format(self.step), flush=True)
        print("actions:", actions)
        self.step += 1
        self.done = 0

        # print("sum Mem:{}".format(sum(self.vm_curr_mem)),flush=True)
        sum_up_percent = 0.0
        action_cpu = [actions[i] for i in range(0, self.vm_num * 3, 3)]
        action_mem = [actions[i + 1] for i in range(0, self.vm_num * 3, 3)]
        action_bandwith = [actions[i + 2] for i in range(0, self.vm_num * 3, 3)]

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

        # step1:先执行释放操作
        # action永远是-1 ~ 1之间
        for i in range(self.vm_num):
            if action_mem[i] < 0 and self.observations[i][2] <= 0.75 and self.vm_curr_mem[i] * (-action_mem[i]) >= self \
                    .vm_min_mem:
                self.execute_mem(2, i, action_mem[i])
            elif action_mem[i] > 0 and self.observations[i][2] > 0.8 and self.vm_curr_mem[i] <= self.vm_max_mem:
                self.vm_up_percent += action_mem[i]  # 统计总共增加的百分比
                self.vm_up_count += 1
            elif action_mem[i] == 0 and self.observations[i][2] < 0.75 and self.observations[i][2] > 0.65:
                pass
            else:
                pass

            if action_cpu[i] < 0 and self.observations[i][0] <= 0.7 and self.vm_curr_cpu[i] - 1 >= self.vm_min_cpu:
                self.execute_cpu(2, i)
            elif action_cpu[i] > 0 and self.observations[i][0] > 0.8 and self.vm_curr_cpu[i] <= self.vm_max_cpu:
                pass
            else:
                pass

            if action_bandwith[i] < 0 and self.observations[i][4] <= 0.75 and self.vm_curr_bandwith[i] * (
            -action_bandwith[i]) >= self.vm_min_bandwith:
                self.execute_bandwith(2, i, action_bandwith[i])
            # print("---- VM:{} DOWN BAND {} bandwith usage:{}% ----".format(i + 1, self.vm_curr_bandwith[i],
            #                                                                    self.observations[i][4]),flush=True)
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
                self.execute_mem(1, i, action_mem[i])
            elif action_mem[i] < 0 and self.observations[i][2] < 0.75 and self.vm_curr_mem[i] >= self.vm_min_mem:
                pass
            elif action_mem[i] == 0 and self.observations[i][2] < 0.75 and self.observations[i][2] > 0.7:
                pass
            else:
                bad_action_mem[i] = 1
                # 处于bad_state,当前的任务可以不去执行
                reword_mem[i] -= 1  # 这个是

            ## 执行cpu增操作
            if action_cpu[i] > 0 and self.observations[i][0] > 0.6 and self.vm_curr_cpu[i] + 1 <= self.vm_max_cpu:
                self.execute_cpu(1, i)
            elif action_cpu[i] < 0 and self.observations[i][0] < 0.6 and self.vm_curr_cpu[i] >= 1:
                pass
            elif action_cpu[i] == 0 and self.observations[i][0] < 0.6 and self.observations[i][0] > 0.5:
                pass
            else:
                bad_action_cpu[i] = 1
                reword_cpu[i] -= 1
            ## 执行bandwith增操作:

            if action_cpu[i] > 0 and self.observations[i][4] > 0.75 and self.vm_curr_bandwith[i] * (
                    1 + action_bandwith[i]) < self.vm_max_bandwith:
                self.execute_bandwith(1, i, action_bandwith[i])
            elif action_bandwith[i] < 0 and self.observations[i][4] < 0.75 and self.vm_curr_bandwith[
                i] >= self.vm_min_bandwith:
                pass
            elif action_bandwith[i] == 0 and self.observations[i][4] > 0.5 and self.observations[i][4] < 0.75:
                pass
            else:
                bad_action_bandwith[i] = 1
                reword_bandwith[i] -= 1

        ## 执行内存矫正,避免内存泄露
        self.execute_mem(3, 1, 1)  # 校正内存,防止内存泄露

        # 执行完操作后,保存原状态,获取当前新状态
        self.pre_observations = self.observations
        time.sleep(0.5)
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
        reword_cpu_var = (self.l1 * (pre_variance_cpu - variance_cpu))

        pre_variance_bandwith = (np.array(list(self.pre_observations.values()))[:, 4]).std()
        variance_bandwith = (np.array(list(self.observations.values()))[:, 4]).std()
        reword_bandwith_var = (self.l1 * (pre_variance_bandwith - variance_bandwith))

        # step3:计算reword()
        self.bad_state_vm_pre = 0
        self.bad_state_vm = [0] * self.vm_num

        for i in range(self.vm_num):
            # 首先计算内存
            if self.pre_observations[i][2] > 0.9 or self.pre_observations[i][0] > 0.86:
                self.bad_state_vm_pre += 1

            if self.observations[i][2] > 0.9 and self.vm_curr_mem[i] < self.vm_max_mem:
                self.bad_state_vm[i] = 1
                reword_mem[i] += -6  # 我们的目标就是调节到 bad_state_vm为0 即可了
                print("THE VM {} MEM IN BAD STATE!!".format(i))
                sys.stdout.flush()

            # 1.虽然mem利用率下降,但是调节了bad_state
            if self.pre_observations[i][2] >= self.observations[i][2] and self.pre_observations[i][2] > 0.8:
                reword_mem[i] -= 0.3  # 这是个优秀的操作

            # 2.mem利用率下降,且超过合理范围
            elif self.pre_observations[i][2] >= self.observations[i][2] and self.observations[i][2] < 0.45:
                reword_mem[i] += -4

            # 3.mem利用率下降,且没调节bad_state且在合理范围.
            elif self.pre_observations[i][2] >= self.observations[i][2]:
                reword_mem[i] += -2

            # 4.mem利用率上升,并且在合理范围内
            if self.pre_observations[i][2] < self.observations[i][2] and self.observations[i][2] < 0.78:
                reword_mem[i] += -1
            # 4.如果利用率上述,且不在合理范围内,那么就应该给予惩罚
            elif self.pre_observations[i][2] < self.observations[i][2]:
                reword_mem[i] += -5

            # cpu--------------------------
            if self.observations[i][0] > 0.7 and self.vm_curr_cpu[i] < self.vm_max_cpu:
                self.bad_state_vm[i] = 1
                reword_cpu[i] += -18  # 我们的目标就是调节到 bad_state_vm为0 即可了
                print("THE VM {} CPU IN BAD STATE!!".format(i))
                sys.stdout.flush()

            # 1.虽然cpu利用率下降,但是调节了bad_state
            if self.pre_observations[i][0] >= self.observations[i][0] and self.pre_observations[i][0] > 0.85:
                reword_cpu[i] -= 0.3  # 这是个优秀的操作

            # 2.cpu利用率下降,且超过合理范围
            elif self.pre_observations[i][0] >= self.observations[i][0] and self.observations[i][0] < 0.5:
                reword_cpu[i] += -1.2

            # 3.cpu利用率下降,且没调节bad_state且在合理范围.
            elif self.pre_observations[i][0] >= self.observations[i][0]:
                reword_cpu[i] += -0.7

            # 4.cpu利用率上升,并且在合理范围内
            if self.pre_observations[i][0] < self.observations[i][0] and self.observations[i][0] < 0.6:
                reword_cpu[i] += -1

            # 5.如果利用率上述,且不在合理范围内,那么就应该给予惩罚
            elif self.pre_observations[i][0] < self.observations[i][0]:
                reword_cpu[i] += -4

            # 带宽----------------------------------------------
            if self.observations[i][4] > 0.95 and self.vm_curr_bandwith[i] < self.vm_max_bandwith:
                self.bad_state_vm[i] = 1
                reword_bandwith[i] += -6  # 我们的目标就是调节到 bad_state_vm为0 即可了
                print("THE VM {} BANDWIDTH IN BAD STATE!!".format(i))
                sys.stdout.flush()

            # 1.虽然带宽利用率下降,但是调节了bad_state
            if self.pre_observations[i][4] >= self.observations[i][4] and self.pre_observations[i][4] > 0.8:
                reword_bandwith[i] += 8  # 这是个优秀的操作

            # 2.band利用率下降,且超过合理范围
            elif self.pre_observations[i][4] >= self.observations[i][4] and self.observations[i][4] < 0.5:
                reword_bandwith[i] += -3

            # 3.band利用率下降,且没调节bad_state且在合理范围.
            elif self.pre_observations[i][4] >= self.observations[i][4]:
                reword_bandwith[i] += -1

            # 4.band利用率上升,并且在合理范围内
            if self.pre_observations[i][4] < self.observations[i][4] and self.observations[i][4] < 0.6:
                reword_bandwith[i] += 1

            # 5.如果利用率上述,且不在合理范围内,那么就应该给予惩罚
            elif self.pre_observations[i][4] < self.observations[i][4]:
                reword_bandwith[i] += -4

        # print("old feature"); print(self.pre_observations)
        # print("feature");print(self.observations)

        reword_mem_sum = np.array(reword_mem) + reword_mem_var
        reword_cpu_sum = np.array(reword_cpu) + reword_cpu_var
        reword_bandwith_sum = np.array(reword_bandwith) + reword_bandwith_var

        #print("+++ Reword:{},Sum_mem:[{}/{}],Sum cpus:{},Sum bandwidth:{} +++".format(
        #	sum(reword_bandwith_sum + reword_mem_sum + reword_cpu_sum), sum(self.vm_curr_mem), self.vm_sum_mem,
        #	sum(self.vm_curr_cpu), sum(self.vm_curr_bandwith)))
        print("+++ Now Bad VM num is [{}/{}] +++".format(sum(self.bad_state_vm), self.vm_num))
        print("-----------------------------------------------------------")
        time.sleep(1)

        if sum(self.bad_state_vm) == 0:
            self.ok_time += 1

        if self.ok_time >=1 :
            self.done = 1
            self.ok_time = 0
        else:
            self.done = 0

        reword_sum = reword_mem_sum + reword_cpu_sum + 0.4 * reword_bandwith_sum  # 暂时不算bandwith

        return np.array(self.vm_state), np.array(reword_sum), self.done

