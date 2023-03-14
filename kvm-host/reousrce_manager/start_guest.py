import paramiko
import time

ip = {1: "192.168.122.22", 2: "192.168.122.133", 3: "192.168.122.137", 4: "192.168.122.103", 5: "192.168.122.201", 6: "192.168.122.2", 7: "192.168.122.236", 8: "192.168.122.223", 9: "192.168.122.211", 10: "192.168.122.81", 11: "192.168.122.53", 12: "192.168.122.12", 13: "192.168.122.3", 14: "192.168.122.217", 15: "192.168.122.248"}
list=[]

for i in range(1, 11):

    # 建立一个sshclient对象
    list.append(paramiko.SSHClient())
    ssh = list[i-1]
    # 允许将信任的主机自动加入到host_allow 列表，此方法必须放在connect方法的前面
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # 调用connect方法连接服务器
    ssh.connect(hostname=ip[i], port=22, username='root', password='vm1')
    # 执行命令
    # conn = ssh.invoke_shell()
    # conn.send("nohup /home/vm1/kvm-guest/dacapo_without_payload.sh &")
    stdin, stdout, stderr = ssh.exec_command('nohup /home/vm1/kvm-guest/dacapo_without_payload.sh &')
    # stdin, stdout, stderr = ssh.exec_command('df -hl')
    # 结果放到stdout中，如果有错误将放到stderr中
    # print(stdout.read().decode())
    # 关闭连接
    ssh.close()
time.sleep(10000000)