from socket import *

serverList= {1: "192.168.122.22", 2: "192.168.122.133", 3: "192.168.122.137", 4: "192.168.122.103", 5: "192.168.122.201", 6: "192.168.122.2", 7: "192.168.122.236", 8: "192.168.122.223", 9: "192.168.122.211", 10: "192.168.122.81", 11: "192.168.122.53", 12: "192.168.122.12", 13: "192.168.122.3", 14: "192.168.122.217", 15: "192.168.122.248"}
serverPort = 12000


class UdpClient:
    # serverName = '192.168.122.22'

    def __init__(self):
        # for i in [1,3,4,5,6,8,9,10]:
        for i in range(1, 11):
            print(i)
            self.socketAddress = (serverList[i], serverPort)
            #define the type of socket is IPv4 and Udp
            self.clientSocket = socket(AF_INET, SOCK_DGRAM)

            message1 = "kill -9 $(ps -aux | grep mm_test_static|grep -v color|awk -F \" \" '{print $2}')"
            print(message1)
            self.clientSocket.sendto(message1.encode('utf-8'), self.socketAddress)

            # message2 = "/home/vm1/kvm-guest/dacapo.sh > /home/vm1/kvm-guest/result.log"
            # self.clientSocket.sendto(message2.encode('utf-8'), self.socketAddress)
            
            # returnMessage, serverAddress = self.clientSocket.recvfrom(2048)


if __name__ == '__main__':
    client = UdpClient()
