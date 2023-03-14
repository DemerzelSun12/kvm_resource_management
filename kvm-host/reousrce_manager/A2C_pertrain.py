import argparse
from itertools import count
from src.logger import *
import csv
import os, sys, random, time
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from A2C_util_pretrain import Env
from A2C_pertrain_util import Env
#from A2C_docker_util import Env


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
parser.add_argument("--env_name", default="Env")
parser.add_argument('--tau', default=0.1, type=float)  # target smoothing coefficient
parser.add_argument('--test_iteration', default=5, type=int)
parser.add_argument('--filename',default='KVM_15_2021-5-8',type=str)

parser.add_argument('--learning_rate', default=0.1, type=float)  # 1e-3
parser.add_argument('--gamma', default=1, type=int)  # discounted factor
parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size
parser.add_argument('--batch_size', default=100, type=int)  # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)

parser.add_argument('--log_interval', default=40, type=int)  #
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--exploration_noise', type=float)
parser.add_argument('--max_episode', default=100000, type=int)  # num of max train
parser.add_argument('--update_iteration', default=200, type=int)  # 每100轮更新一次
parser.add_argument('--vm_num', default=15, type=int)
parser.add_argument('--e_value', default=0.9, type=float)

sys.stdout = Logger(sys.stdout) # 打印正常的输出
sys.stderr = Logger(sys.stderr) # 打印错误信息

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = "ddpg_new"
f = open('{}.csv'.format(args.filename),'w',encoding='utf-8')
writer = csv.writer(f)
writer.writerow(["Iterator","Vmid","Mems","Mem_usage","CPUs","CPU_usage","Bands","Band_usage"])
env = Env(args.vm_num, writer)
#env = Env(args.vm_num)


if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = 6 * args.vm_num  # 状态dim cpu利用率，cpu数量，mem利用率，mem数量，通信量，当前带宽分配量
action_dim = 3 * args.vm_num  # 动作dim
max_action = 1  # 最大的action

directory = './' + script_name


# 不需要修改
class Replay_buffer():
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


# 不需要修改
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


# 不需要修改
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# 需要修改：
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):

        # ---------------------------------------------------
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        # ----------------------------------------------------
        # self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #print("action:")
        #print(self.actor(state).cpu().data.numpy().flatten())
        return self.actor(state).cpu().data.numpy().flatten()  # 这个都不太

    def update(self):
        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # 不需要修改
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            # print(target_Q)
            target_Q = reward + (done * args.gamma * target_Q).detach()
            current_Q = self.critic(state, action)
            # print("target_Q:{} current_Q:{}".format(target_Q,current_Q))

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor_pretrain.pt')
        torch.save(self.critic.state_dict(), directory + 'critic_pretrain.pt')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        # print(directory + 'actor_pretrain.pt')
        self.actor.load_state_dict(torch.load(directory + 'actor_pretrain.pt'))
        self.critic.load_state_dict(torch.load(directory + 'critic_pretrain.pt'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    # time.sleep(5)
    agent = DDPG(state_dim, action_dim, max_action)  # 建立一个DDPG模型
    ep_r = 0
    # os.system("virsh domiftune vm6 52:54:00:b6:09:68 --live --config --inbound 512")
    # os.system("virsh domiftune vm6 52:54:00:b6:09:68 --live --config --outbound 512")
    # os.system("virsh domiftune vm2 52:54:00:cc:d4:6a --live --config --inbound 2048")
    # os.system("virsh domiftune vm2 52:54:00:cc:d4:6a --live --config --outbound 2048")
    # os.system("virsh domiftune vm3 52:54:00:ad:01:be --live --config --inbound 2048")
    # os.system("virsh domiftune vm3 52:54:00:ad:01:be --live --config --outbound 2048")
    # os.system("virsh domiftune vm4 52:54:00:a4:b5:e9 --live --config --inbound 2048")
    # os.system("virsh domiftune vm4 52:54:00:a4:b5:e9 --live --config --outbound 2048")
    # os.system("virsh domiftune vm5 52:54:00:54:02:d6 --live --config --inbound 2048")
    # os.system("virsh domiftune vm5 52:54:00:54:02:d6 --live --config --outbound 2048")
    if args.mode == 'test':
        print("[INFO] Test model ")
        agent.load()  # 加载模型
        cnt = 0
        state = env.init()  # 初始化环境
        done = 0
        for i in range(args.test_iteration):  # 最大调节次数
            # 对于测试来说
            # 1.进行调节,调节如果成功后则进行loop,直到再次出现不良状态再继续调节
            for t in range(10000):
                cnt += 1
                #print("current state", state)
                action = agent.select_action(state)
                #print("current action", action)
                action = action.clip(-1, 1)
                next_state, reward, done = env.steps(action, "test")
                ep_r += reward
                state = next_state
                f.flush()
                time.sleep(10)
                if done:
                    print("[FINISH] This envrioment is balance!")
                    print("[FINISH] We have done {} times adjustment!".format(cnt))
                    env.waitForNextLoop()
                    done = 0
                    cnt = 0

    #state：ndarray 4*10 维度 但是安排的时候应当是 1*10+1*10+1*10
    elif args.mode == 'train':
        if args.load: agent.load()
        step = 0
        done = 0
        for maps in range(1):
            env.startNewGame()
            state = env.init()  # 获取初始的
            args.exploration_noise = 1
            for i in range(10000):
               #  print("begin train")
                if done == 1:  # 如果经过当前调整已经稳定了，那么就可以重新配置了。
                    state = env.reset()
                    done = 0	 # 如果重置的话
                total_reward = 0
                for t in range(1500):
                    action = agent.select_action(state)
                    # 选择一个action:
                    action = (action + np.random.normal(0, args.exploration_noise, size=action_dim)).clip(-1, 1)
                    #print(action)
                    next_state, reward, done = env.steps(action)  # step需要获取下一次的元素
                    agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
                    state = next_state
                    if done == True:
                        print("[INFO] DONE BREAK!")
                        break
                    step += 1
                    total_reward += reward

                print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(step, i, total_reward))
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                sys.stdout.flush()
                agent.update()

                # "Total T: %d Episode Num: %d Episode T: %d Reward: %f
                if i % 50 == 0:
                    args.exploration_noise -= 0.2
                    if args.exploration_noise < 0.1:
                        args.exploration_noise = 0.1
                if i%100 ==0 and i!=0:
                    agent.save()
    else:
        raise NameError("mode wrong!!!")
    f.close()


if __name__ == '__main__':
    main()

# 问题集中在bad_action 和 bad_states之间的问题
# bad_action = 什么都不做
#
