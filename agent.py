
import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import traci

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from network import Actor, Critic
from memory import ReplayBuffer

from collections import defaultdict

from traffic import *

WEIGHT_DECAY = 0

class OUNoise(object):
    '''Ornstein–Uhlenbeck
    '''
    def __init__(self, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = 1
        self.low          = -2.0
        self.high         = 2.0
        self.reset()
        
    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu
        
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs
    
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = cfg.device
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)

        # copy parameters to target net
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),  lr=cfg.critic_lr, weight_decay=WEIGHT_DECAY)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        self.noise = OUNoise()
        
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.cfg = cfg

    def get_current_state(self):
        vehs = traci.vehicle.getIDList()
        
        current_state_dict = defaultdict(list)
        for veh in vehs:
            tl_info = get_tls_info(veh)
            if tl_info != None:
                # velocity, accel
                curr_pos = traci.vehicle.getLanePosition(veh)
                current_state_dict[veh].append(traci.vehicle.getSpeed(veh))
                current_state_dict[veh].append(traci.vehicle.getAcceleration(veh))
                leader_veh_info = get_leader_veh_info(veh, vehs, self.cfg)
                leader_flag, leader_speed, dist_to_leader = leader_veh_info[0], leader_veh_info[1], leader_veh_info[2]
                # 标记当前车辆前方有无汽车
                current_state_dict[veh].append(leader_flag)
                # 前车的速度
                current_state_dict[veh].append(leader_speed)
                # 前车与当前车辆的距离
                current_state_dict[veh].append(dist_to_leader)
                # 再获取当前车辆到交叉路口处的距离
                tl_flag, tl_dur, tl_dist = tl_info[0], tl_info[1], tl_info[2]
                current_state_dict[veh].append(tl_dist)
                current_state_dict[veh].append(tl_flag)
                current_state_dict[veh].append(tl_dur)  
        return current_state_dict    

    def choose_action(self, current_state_dict, curr_step, add_noise=True):
        action_dict = defaultdict(float)
        for key in current_state_dict:
            state = torch.FloatTensor(current_state_dict[key]).unsqueeze(0).to(self.device)
            # 在这里eval,batchnorm层会停止计算和更新mean和var，加速运算
            self.actor.eval()
            # no_grad层的作用:用于停止autograd模块的工作，以起到加速和节省显存的作用，但不会影响batch norm和dropout层的行为
            with torch.no_grad():
                action = self.actor(state)   
                action = action.detach().cpu().numpy()[0, 0]  
            self.actor.train()
            # print('添加噪声之前的动作:{}'.format(action))
            if add_noise:
                action = self.noise.get_action(action, curr_step)
            action_dict[key] = np.clip(action, -2.0, 2.0)
            # print('添加噪声之后的动作:{}'.format(action_dict[key]))
        return action_dict

    def step(self, current_state_dict, action_dict):
        for key in current_state_dict:
            current_speed = current_state_dict[key][0]
            desired_speed = current_speed + action_dict[key]
            traci.vehicle.setSpeed(key, desired_speed)

        #未已经通过十字路口的车辆设置速度，跟驰模型为IDM
        for veh in traci.vehicle.getIDList():
            if veh not in current_state_dict:
                curr_speed = traci.vehicle.getSpeed(veh)
                traci.vehicle.setSpeed(veh, 20.)
        #根据刚才给每一辆车的赋予的速度，进行单步仿真
        traci.simulationStep()
        #在仿真中，你想要设定的加速度，未必是车辆真正在下一步仿真的加速度(原因是可能由于拥堵，导致车辆此时的加速度无法达到想要的加速度)
        for key in action_dict:    
            action_dict[key] = traci.vehicle.getAcceleration(key)
        next_state_dict = self.get_current_state()
        return next_state_dict, action_dict

    def get_reward(self, next_state_dict, action_dict):
        reward_dict = defaultdict(list)
        reward_speed = 0.0
        reward_tls = 0.0

        #必须是存在于上一时刻和当前时刻的车辆，才能获取奖励
        for key in next_state_dict:
            current_speed = next_state_dict[key][0]     # 当前车辆的速度
            dist_to_intersec = next_state_dict[key][-3] # 当前车辆到十字路口的距离
            light_flag = next_state_dict[key][-2]       # 当前车辆行驶方向的车道是绿灯还是红灯
            time_to_green = next_state_dict[key][-1]    # 当前相位还有多少时间变绿/或者当前绿灯相位持续多少时间
            #-----------车辆行驶过程速度获得的奖励，速度太慢和太快都会获得惩罚----------#
            reward_speed = current_speed / 20.
            #------------------------获取通过交通灯时得到的奖励-----------------------#
            if light_flag == 0:
                # 此时为红灯
                # 若车辆以当前车速行驶，通过十字路口所需的时间小于交通灯由红变绿的时间,则车辆会停在十字路口
                if current_speed * time_to_green > dist_to_intersec:   
                    reward_tls = -2.5
                else:
                    reward_tls = 0.5
            else:
                # 此时为绿灯,这里的time_to_green实际是当前绿灯的剩余时长
                # 如果车辆以当前速度行驶，未能在绿灯结束前通过十字路口，则获得惩罚
                if current_speed * time_to_green < dist_to_intersec:
                    reward_tls = -2.5
                else:
                    reward_tls = 0.5
            # #---------------------------前车与后车交互过程获得的奖励--------------------#           
            # #--------------------------------------------------------------------------#
            total_reward = reward_speed + reward_tls
            #这里分别记录每一个单独的奖励在训练过程中的变化图，
            reward_dict[key].append(total_reward)
            reward_dict[key].append(reward_speed)
            reward_dict[key].append(reward_tls)
        return reward_dict

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state = self.memory.sample(batch_size)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
       
        #更新critic的参数
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward +  self.cfg.gamma * target_value
        eval_value = self.critic(state, action)
        value_loss = nn.MSELoss()(eval_value, expected_value.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        #更新actor的参数
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        #软更新target actor和target critic的网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.cfg.soft_tau) +
                param.data * self.cfg.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.cfg.soft_tau) +
                param.data * self.cfg.soft_tau
            )
            
    def reset(self):
        self.noise.reset()
                 
