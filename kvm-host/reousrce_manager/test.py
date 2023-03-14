# import socket



# def getMemBandinfo():
#         mems_usage = []
#         bandwith_usage = []
#         port = 12345
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         ip = "192.168.122.22"
#         s.sendto("1".encode(encoding="utf-8"), (ip, port))
#         mem_usage, _ = s.recvfrom(1024)
#         net_in, _ = s.recvfrom(1024)
#         net_out, _ = s.recvfrom(1024)
#         mem_usage.decode(encoding="utf-8")
#         net_in.decode(encoding="utf-8")
#         net_out.decode(encoding="utf-8")
#         mems_usage.append(float(mem_usage) / 100)
#         # if self.vm_curr_bandwith[i] != 0:
#         #     bandwith_usage.append(max(float(net_in), float(net_out)) / self.vm_curr_bandwith[i])
#         # else:
#         #     bandwith_usage.append(0)
#         s.close()
#         print(net_in)
#         print(net_out)
#         return mems_usage, bandwith_usage

# getMemBandinfo()
import re
import time

netcards = {1: "vnet0", 2: "vnet1", 3: "vnet2", 4: "vnet3", 5: "vnet4", 6: "vnet5", 7: "vnet6", 8: "vnet7", 9: "vnet8", 10: "vnet9", 11: "vnet10", 12: "vnet11", 13: "vnet12", 14: "vnet12", 15: "vnet14"}

vm_curr_bandwith = [0.5] *15
def get_values( orders, values_dic):
        try:
            with open('/proc/net/dev') as f:
                lines = f.readlines()  # 内容不多，一次性读取较方便
                for arg in netcards.values():
                    for line in lines:
                        line = line.lstrip()  # 去掉行首的空格，以便下面split
                        if re.match(arg, line):
                            values = re.split("[ :]+", line)  # 以空格和:作为分隔符
                            values_dic[arg + 'r' + orders] = values[1]  # 1为接收值
                            values_dic[arg + 't' + orders] = values[9]  # 9为发送值
        except (FileExistsError, FileNotFoundError, PermissionError):
            print('open file error')
            sys.exit(-1)

def getBandwithInfo():
        bandwith_nums = vm_curr_bandwith
        bandwith_usage = []
        values_dic = {}
        get_values('first', values_dic)  # 第一次取值
        time.sleep(4)
        get_values('second', values_dic)  # 10s后第二次取值
        for id, arg in enumerate(netcards.values()):
                r_bandwidth = (int(values_dic[arg + 'r' + 'second']) - int(
                values_dic[arg + 'r' + 'first'])) / 1024 / 1024 / 4
                #print(r_bandwidth)
                t_bandwidth = (int(values_dic[arg + 't' + 'second']) - int(
                values_dic[arg + 't' + 'first'])) / 1024 / 1024 / 4
                bandwith_usage.append(max(r_bandwidth, t_bandwidth) / bandwith_nums[id])
                #print(t_bandwidth)
        values_dic = {}  # 清空本次循环后字典的内容
        return bandwith_usage

print(getBandwithInfo())
