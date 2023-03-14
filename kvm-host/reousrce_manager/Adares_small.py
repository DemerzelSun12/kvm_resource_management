import numpy as np
import random
import csv
import psutil
import libvirt
import socket
import time
import os
import sys
class LinUCB:
    def __init__(self,vm_num,arm):
        self.alpha = 0.4
        self.r1 = 1             #正反馈
        self.r0 = -5            #负反馈
        self.r00 = -16          #bad state反馈状态
        self.d = 4              #用户特征维度
        self.Aa = {}            #Aa记录每个Arm的不想关属性
        self.AaI = {}           #Aa的逆矩阵
        self.step = 1
        self.num = vm_num  # 虚拟机数量
        self.curr_mem = [1300]*self.num
        self.curr_cpu = [2]*self.num
        self.arm = arm

        self.vm_domain = {1: "ubuntu18.04-05", 2: "ubuntu18.04-06", 3: "ubuntu18.04-07", 4: "ubuntu18.04-08",
                          5: "ubuntu18.04-09"}
                          #6: "ubuntu18.04-10", 7: "ubuntu18.04-11", 8: "ubuntu18.04-12", 9: "ubuntu18.04-13",
                          #10: "ubuntu18.04-14",
                          #11: "ubuntu18.04-15", 12: "ubuntu18.04-16", 13: "ubuntu18.04-17", 14: "ubuntu18.04-18",
                          #15: "ubuntu18.04-19", 16: "ubuntu18.04-20"}
        self.ip = {1: "192.168.122.195", 2: "192.168.122.196", 3: "192.168.122.197", 4: "192.168.122.198",
                   5: "192.168.122.199"}
                   #6: "192.168.122.200", 7: "192.168.122.201", 8: "192.168.122.202", 9: "192.168.122.203",
                   #10: "192.168.122.204",
                   #11: "192.168.122.205", 12: "192.168.122.206", 13: "192.168.122.207", 14: "192.168.122.208",
                   #15: "192.168.122.209", 16: "192.168.122.210"}

        self.conn = libvirt.open("qemu:///system")

        self.correct_mem = 0
        self.down_mem = 0
        self.up_vm_count = 0
        self.ba = {}            #ba 计算不想关属性的向量
        self.theta = {}         #θ需要更新的参数
        self.a_max = {}         #最优的action
        self.x = {}             #此时的输入　d*1维
        self.xT = {}            #转置
        self.max_mem = 4096       #最大的mem大小,这里我们以128为调整范围,那么最大为
        self.max_cpu = 8
        self.feature = []       #特征
        self.prev_feature = []
        self.sum_mem = 1300*vm_num     #总的内存的大小

    def getCpuInfo(self):
        cpu_nums = self.curr_cpu
        cpu_usage = []
        t1  = []
        c1 = []
        i = 0
        for id in self.vm_domain.values():
            domain = self.conn.lookupByName(id)
            t1.append(time.time())
            c1.append(int(domain.info()[4]))
        time.sleep(2)
        for i,id in enumerate(self.vm_domain.values()):
            domain = self.conn.lookupByName(id)
            t2 = time.time()
            c2 = int(domain.info()[4])
            if ((t2 - t1[i]) * cpu_nums[i] * 1e9) !=0:
                usage = (c2 - c1[i]) / ((t2 - t1[i]) * cpu_nums[i] * 1e9)
            else:
                usage = 1.0
            cpu_usage.append(usage)
            i += 1
        return cpu_nums,cpu_usage

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

    def get_values(self,orders, values_dic):
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
        self.get_values('first',values_dic)  # 第一次取值
        time.sleep(4)
        self.get_values('second',values_dic)  # 10s后第二次取值
        for id,arg in enumerate(self.netcards.values()):
            r_bandwidth = (int(values_dic[arg + 'r' + 'second']) - int(
                values_dic[arg + 'r' + 'first'])) / 1024 / 1024 / 4
            t_bandwidth = (int(values_dic[arg + 't' + 'second']) - int(
                values_dic[arg + 't' + 'first'])) / 1024 / 1024 / 4
            bandwith_usage.append(max(r_bandwidth,t_bandwidth)/bandwith_nums[id])
        values_dic = {}  #清空本次循环后字典的内容
        return bandwith_usage


    def initializes(self,arms):
        #cpu = list(random.sample(list(range(5, 10)), self.num))
        #self.max_vcpu = sum(cpu)+5
        #cpu_usage = [100-i*10 for i in cpu]
        #mem = list(random.sample(list(range(5, 10)), self.num))
        cpu_nums,cpu_usage = self.getCpuInfo()
        mem_usage = self.getMemInfo()
        mem_nums = self.curr_mem
        self.feature = [[cpu_usage[i], cpu_nums[i], mem_usage[i], mem_nums[i]] for i in range(self.num)]

        for i in range(self.num):
            self.Aa[i] = {}
            self.ba[i] = {}
            self.AaI[i] = {}
            self.theta[i] = {}
            for key in arms:
                self.Aa[i][key] = np.identity(self.d)  #对角矩阵d*d
                self.ba[i][key] = np.zeros((self.d,1)) #全0向量
                self.AaI[i][key] = np.identity(self.d)
                self.theta[i][key] = np.zeros((self.d,1)) #初始化θ
        print("initalizer ok----------------")

    def update(self,reword):
        for i in range(self.num):
            r = reword
            self.Aa[i][self.a_max[i]] += np.dot(self.x[i],self.xT[i])
            self.ba[i][self.a_max[i]] += r[i]*self.x[i]
            self.AaI[i][self.a_max[i]] = np.linalg.inv(self.Aa[i][self.a_max[i]])
            self.theta[i][self.a_max[i]] = np.dot(self.AaI[i][self.a_max[i]],self.ba[i][self.a_max[i]])

    def recommend(self,timestamp,arms):
        print("{} step allocation start------".format(self.step))
        #normalize
        features = np.array(self.feature) #行向量
        max_feature = np.array([1,self.max_cpu,1,self.max_mem])
        features = (features/max_feature).tolist()

        for i in range(self.num):
            xaT = np.array([features[i]]) #行向量
            xa = np.transpose(xaT)        #列向量

            #遍历每一个arm
            AaI_tmp = np.array([self.AaI[i][arm] for arm in arms])
            theta_tmp = np.array([self.theta[i][arm] for arm in arms])

            #tmp = np.dot(xaT,theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT,AaI_tmp),xa))
            art_max = arms[np.argmax(np.dot(xaT,theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT,AaI_tmp),xa)))]
            self.x[i] = xa
            self.xT[i] = xaT
            self.a_max[i] = art_max

            #sql = "insert into mem_actions_dqn values(%s,%s,%s);"
            #self.cur.execute(sql,(i+1,self.step,art_max))
            #self.conn.commit()

            #timestamp += 1;
            print ("vm %d arm为%d"%(i,art_max))

        self.step += 1 #做出了调整的建议

        return self.a_max

    def execute_mem_allcation(self,vm_id,op):
        mem_size = self.curr_mem[vm_id]
        newmem_size = mem_size
        vm_domain = self.vm_domain[vm_id+1] #获取虚拟机的域名

        if self.up_vm_count != 0: #如果有需要增加内存的虚拟机
            up_mem_size = round((self.down_mem)/self.up_vm_count)
            newmem_size += up_mem_size
        else:
            pass

        if op==1 and newmem_size <= self.max_mem:
            print("VM:{} UP MEM {}MB TO {}MB".format(vm_id+1,mem_size,newmem_size))
            val = os.system("virsh setmem %s --size %dMB >/dev/null"%(vm_domain,newmem_size))
            self.curr_mem[vm_id] = newmem_size
        elif op==1 and newmem_size > self.max_mem:
            #这里有点问题,内存能守恒吗?
            print("VM:{} MAX MEM {}MB TO {}MB ".format(vm_id,mem_size,self.max_mem))
            val = os.system("virsh setmem %s --size %dMB >/dev/null"%(vm_domain,self.max_mem))
            self.curr_mem[vm_id] = self.max_mem
        elif op==2:
            print("VM:{} DOWN MEM {}MB TO {}MB".format(vm_id,mem_size,mem_size-128))
            val = os.system("virsh setmem %s --size %dMB > /dev/null"%(vm_domain,mem_size-128))
            self.curr_mem[vm_id] = mem_size - 128
            self.down_mem += 128

        elif op==3: #用于内存校正.
            if self.up_vm_count == 0:
            #没有需要增加内存的机器，我们就需要修增
                self.correct_mem += (self.down_mem)
                self.down_mem = 0

            print("!!! correct: --------------------------",flush=True)
            for i in self.vm_domain.items(): #遍历每一个虚拟机
                if self.curr_mem[i[0]-1]+round(self.correct_mem/self.num) <= self.max_mem: #如果
                    self.curr_mem[i[0]-1] += round(self.correct_mem/self.num)
                else:
                    self.down_mem += self.curr_mem[i[0]-1]+round(self.correct_mem/self.num)-self.max_mem
                    self.curr_mem[i[0]-1] = self.max_mem

            diff = self.sum_mem-sum(self.curr_mem)
            for i in self.vm_domain.items():
                if diff>200 and (self.curr_mem[i[0]-1]+round(diff/self.num)) < self.max_mem:
                    self.curr_mem[i[0]-1] += round(diff/self.num)
                    if i[0]==1:
                        print("memory too small need to add some!",flush=True)
                elif diff<-200 and (self.curr_mem[i[0]-1]-round(diff/self.num)) >= 600:
                    self.curr_mem[i[0]-1] -= round(diff/self.num)
                    if i[0]==1:
                        print("memory too big need to sub some!",flush=True)
                #print("vm id:{};mem_size:{}MB".format(i[0],self.curr_mem[i[0]-1]))
                os.system("virsh setmem %s --size %dMB >/dev/null"%(i[1],self.curr_mem[i[0]-1]))
            diff = 0
            self.correct_mem = 0

    def execute_cpu_allcation(self,vmid,op):
        cpunum = self.curr_cpu[vmid]
        vm_domain = self.vm_domain[vmid+1]
        if op==1 and cpunum < self.max_cpu:
            print("VM:{} UP CPU to {} ".format(vmid+1,cpunum+1))
            ret = os.system("virsh setvcpus --live --guest {} {} > /dev/null".format(vm_domain,cpunum+1))
            if ret == 0:
                self.curr_cpu[vmid] += 1
            else:
                print("CPU UP ERROR!")
        elif op==2 and cpunum > 1:
            print("VM:{} DOWN CPU to {} ".format(vmid + 1, cpunum - 1))
            ret = os.system("virsh setvcpus --live --guest {} {} > /dev/null".format(vm_domain,cpunum-1))
            if ret == 0:
                self.curr_cpu[vmid] -= 1
            else:
                print("CPU DOWN ERROR")
        else:
            print("OPERATOR ERROR!")


    def reword_function(self,feature,arm,writer,it):

        old_feature = feature.copy() #保存st的数值
        reword = [0]*self.num       #每个vm对于的reword
        bad_state = [0]*self.num   #是否为bad_state
        bad_action = [0]*self.num  #判断action是不是good
        # 分别对应cpu+,mem+ ; cpu-,mem+ ; cpu,mem+ ;
        # cpu+,mem ; cpu-,mem ; cpu,mem
        # cpu+,mem-; cpu-,mem- ; cpu,mem-
        #生成一个随机序列,使用随机序列来模拟虚拟机分配资源的顺序,避免总是按照一个顺序分配资源
        order = np.random.permutation(range(self.num))
        #self.sum_mem = sum(self.curr_mem)   #MB
        #self.sum_swap  = sum([line[3] for line in old_feature]) #MB
        self.up_vm_count = 0
        self.down_mem = 0

        #先执行降低操作:
        for i in order:
            #减小cpu
            if (arm[i]==1 or arm[i]==4 or arm[i]==7)  and feature[i][0]<0.8 and feature[i][1]>1:
                self.execute_cpu_allcation(i,2)
            elif (arm[i]==1 or arm[i]==4 or arm[i]==7):
                bad_action[i] += 1
                reword[i] = -1

            if (arm[i]==6 or arm[i]==7 or arm[i]==8) and feature[i][2]<0.75 and feature[i][3]>= 680:
                self.execute_mem_allcation(i,2)
            elif(arm[i]==6 or arm[i]==7 or arm[i]==8):
                bad_action[i] += 1
                reword[i] = -1

            if (arm[i]==0 or arm[i]==1 or arm[i]==2) and feature[i][2]>0.7 and feature[i][1] <= (self.max_mem):
                self.up_vm_count += 1

        print("+++++ this time down mem:{} up vm num:{}++++".format(self.down_mem,self.up_vm_count))


        for i in order:
            #增加cpu:
            if (arm[i]== 0 or arm[i]==3 or arm[i]==6) and feature[i][0]>0.75 and feature[i][1] < (self.max_cpu):
                self.execute_cpu_allcation(i,1)
            elif (arm[i]== 2 or arm[i]== 5 or arm[i]==8) and feature[i][0]<0.8 and feature[i][0]>0.5:
                pass
            elif (arm[i]== 0 or arm[i]==3 or arm[i]==6 or arm[i]== 2 or arm[i]== 5 or arm[i]==8):
                bad_action[i] += 1
                reword[i] = -1

            #增加mem
            if (arm[i]==0 or arm[i]==1 or arm[i]==2) and feature[i][2]>0.75 and feature[i][3] <= (self.max_mem):
                self.execute_mem_allcation(i,1)
            elif (arm[i]==3 or arm[i]==4 or arm[i]==5) and feature[i][2]<0.75 and feature[i][2] >0.5:
                pass
            elif(arm[i]==0 or arm[i]==1 or arm[i]==2 or arm[i]==3 or arm[i]==4 or arm[i]==5) :
                bad_action[i] += 1
                reword[i] = -1   #bad state

        self.execute_mem_allcation(0, 3)
        #2表明需要vm再采集新的状态
        #sql = "update signals set gather_flag=2 where 1=1"
        #self.cur.execute(sql)
        #self.conn.commit()

        self.prev_feature = feature
        cpu_nums, cpu_usage = self.getCpuInfo()
        mem_usage = self.getMemInfo()
        mem_nums = self.curr_mem
        self.feature = [[cpu_usage[i], cpu_nums[i], mem_usage[i], mem_nums[i]] for i in range(self.num)]
        for i in range(self.num):
            #time_stap = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
            writer.writerow([self.step,i,cpu_nums[i],cpu_usage[i],mem_nums[i],mem_usage[i]])
        #self.conn.commit()
        #print(self.step,self.feature,old_feature)
        #下面我们要分情况讨论
        #1.自身性能下降,但是整体性能上升 (较小的正反馈)
        #2.自身性能下降,整体性能也下降 (负反馈1)
        #3.自身性能有缺陷        (负反馈2)
        #4.自身性能上升,整体下降 (负反馈3)
        #5.自身性能上升,整体上升 (正反馈2)
        #总利用率


        #old_mem_usage = sum([line[0] for line in old_feature])
        #new_mem_usage = sum([line[0] for line in new_feature])
        #old_swap_usage = sum([line[2] for line in old_feature])
        #new_swap_usage = sum([line[2] for line in new_feature])

        #print("old mem usgae:%d new mem usage%d \n old swap usage:%d new swap usage:%d"%(old_mem_usage,new_mem_usage,old_swap_usage,new_swap_usage))

        bad_num = [0] * self.num
        is_bad = False
        for i in range(self.num):
            if self.feature[i][0]>0.9:
                is_bad = True
                reword[i] -= 1
                bad_num[i] = 1
                print("VM {} CPU in BAD STATE!".format(i+1))

            if self.feature[i][2]>0.8:
                is_bad = True
                reword[i] -= 1
                bad_num[i] = 1
                print("VM {} MEM IN BAD STATE!".format(i+1))

            #1.mem利用率下降,调节了bad_state
            #2.mem利用率上升
            #1.虽然cpu利用率下降,但是调节了bad_state
            if self.prev_feature[i][0] > self.feature[i][0] and self.prev_feature[i][0] >= 0.85:
                reword[i] += 1
            #2.cpu利用率下降
            elif self.prev_feature[i][0] > self.feature[i][0] and self.feature[i][0] < 0.5:
                reword[i] += -0.8
            #3.cpu利用率上升
            elif self.prev_feature[i][0] < self.feature[i][0] and self.feature[i][0] < 0.8:
                reword[i] += 0.6
            else:
                reword[i] -= 0.3

            #4.虽然mem利用率下降了,但是调节了bad_state
            if self.prev_feature[i][2] > self.feature[i][2] and self.prev_feature[i][2] >= 0.8:
                reword[i] += 1
            #5.mem利用率下降
            elif self.prev_feature[i][2] > self.feature[i][2] and self.feature[i][2] < 0.5:
                reword[i] += -0.3
            #6.mem利用率上升
            elif self.prev_feature[i][2] < self.feature[i][2] and self.feature[i][2] < 0.75:
                reword[i] += 0.45
            elif self.prev_feature[i][2]>self.feature[i][2]:
                reword[i] += -0.1
            elif self.prev_feature[i][2]<=self.feature[i][2]:
                reword[i] += -0.2
            else:
                reword[i] += -0.2

        print("action: ",arm)
        print("reword: ",sum(reword))
        print("sum mem:[{}/{}] ".format(sum(self.curr_mem),self.sum_mem))
        print("time:",time.asctime( time.localtime(time.time())))
        print("bad state vm: [{}/{}] ".format(sum(bad_num),self.num))
        print("-----------------------------------------------------------")

        self.update(reword)
        return is_bad

    def clean(self):
        self.curr_mem = [1300]*self.num
        self.curr_cpu = [2]*self.num
        self.down_mem = 0
        os.system("./clean.sh")
        self.initializes(self.arm)


vm_num = 5
arm = [0,1,2,3,4,5,6,7,8]        #对应mem+,mem-,mem不变
UCB = LinUCB(vm_num,arm) #10台vm进行动态调整

#分别对应cpu+,mem+ ; cpu-,mem+ ; cpu,mem+ ; 
#cpu+,mem ; cpu-,mem ; cpu,mem 
#cpu+,mem-; cpu-,mem- ; cpu,mem-

UCB.initializes(arm)
t = 0
is_bad = False
pre_bad = False
filename = sys.argv[1]
f = open("./{}.csv".format(filename),'w')
writer = csv.writer(f)
writer.writerow(["Iterator","vm_id","CPUS","CPU USAGE","MEMS","MEM USAGE"])

for i in range(3000):
    best_arm = UCB.recommend(t,arm)
    pre_bad = is_bad
    is_bad = UCB.reword_function(UCB.feature,best_arm,writer,i)
    time.sleep(2)
    if pre_bad == False and is_bad == False:
        print("Step:{} now everything is ok".format(UCB.step))
        time.sleep(30)  #如果调节成功则睡眠一分钟

f.close()

