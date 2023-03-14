# code_resource_management

## 项目说明

本项目是一个基于深度强化学习的，虚拟机或容器的参数自动调优系统。通过对虚拟机的状态进行获取，动态调节虚拟机的内存、vCPU核数、带宽大小。

## 代码说明：

### KVM-GUEST：

虚拟机使用kvm搭建，推荐使用ubuntu桌面版作为镜像安装，不推荐使用mini镜像安装。在虚拟机中部署 kvm-guest 端代码。

**作用：**部署在虚拟机段、容器端，用于产生负载。

#### 文件说明：


- 组1：train.sh (启动脚本) -> static_payload.py (制造固定负载)
- 组2: init.sh (启动脚本) -> benchmark_test.py (运行dacapo负载)

### KVM-HOST:

作用: 部署在物理机上,用于进行资源调节

### clean-tools:

- 用于在实验后, 快速进行资源配置的还原

### system-consumer:

- 用于统计物理机CPU/内存等资源随时间的变化情况

### resource_manager:

- 算法代码目录
- A2C* 为算法代码
  - 
- Adares 为对比算法代码

## 部署说明：

1. 安装 kvm
2. 通过virt-manager虚拟机管理工具创建一定数量的虚拟机，不要使用virt-install命令创建，会导致读取不到网卡信息
3. 通过ssh进入虚拟机，获取ip地址和mac地址
4. 在虚拟机内部安装qemu
5. 将 `kvm-guest` 目录下文件放在虚拟机内部，修改 `benchmark_test.py` 文件中 `script_type` 字段对应的 `dacappo` 测试程序项目
6. 将 `kvm-host` 放置在主机相应目录下，


## 测试步骤

### 在关闭虚拟机调度管理机制时，测量虚拟机运行时间。

1. 执行各个虚拟机的配置更新脚本，初始化所有虚拟机。bash ./reset.sh
2. 执行清除各个虚拟机当前工作状态脚本，清空虚拟机工作负载。python3 server.py。
3. 通过物理主机ssh连接进入虚拟机，ssh root@192.168.122.{xxx}
4. 进入到虚拟机目录下，cd /home/vm1/kvm-guest
5. 提起程序负载，运行命令 java Test，提起内存与CPU负载。
6. 运行命令/home/vm1/kvm-guest/dacapo.sh > /home/vm1/kvm-guest/result.log，执行测试脚本文件。
7. 查看文件，vim result.log，benchmark运行总时间。
8. 查看文件，vim dacapo.e.log，查看10次benchmark每次运行时间。


### 在开启虚拟机调度管理机制后，测量虚拟机执行时间。

1. 在物理主机上执行命令：cd ~/code_resource_manager/kvm-host/reousrce_manager，进入程序目录
2. 执行各个虚拟机的配置更新脚本，初始化所有虚拟机。bash ./reset.sh
3. 执行清除各个虚拟机当前工作状态脚本，清空虚拟机工作负载。python3 server.py。
4. 开启虚拟机调度管理优化：conda activate py37
5. python3 A2C_pertrain.py --mode test
6. 通过物理主机ssh连接进入虚拟机，ssh root@192.168.122.{xxx}
7. 进入到虚拟机目录下，cd /home/vm1/kvm-guest
8. 提起程序负载，运行命令 java Test，提起内存与CPU负载。
9. 运行命令/home/vm1/kvm-guest/dacapo.sh > /home/vm1/kvm-guest/result.log，执行测试脚本文件。
10. 查看文件，vim result.log，benchmark运行总时间。
11. 查看文件，vim dacapo.e.log，查看10次benchmark每次运行时间。
