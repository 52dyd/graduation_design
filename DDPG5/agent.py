import traci
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from network import Actor, Critic
from memory import ReplayBuffer

from collections import defaultdict

from traffic import *

WEIGHT_DECAY = 0    # L2 weight decay,论文的设置是0.01

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

    def getCurrentStateNCC(self):
        current_state_dict = defaultdict(list)
        veh_list = traci.vehicle.getIDList()
        for veh in veh_list:
            tl_info2 = getNCCTlsInfo2(veh)
            tl_info1 = getNCCTlsInfo2(veh)
            if tl_info2:
                f, l = get_neighbor_cars(veh)
                l_id, l_pos, l_speed = l[0], l[1], l[2]
                f_id, f_pos, f_speed = f[0], f[1], f[2]
                c_pos = traci.vehicle.getLanePosition(veh)
                dist_to_cl = l_pos - c_pos - 4.9
                dist_to_cf = c_pos - f_pos - 4.9 
                #############################################################
                current_state_dict[veh].append(traci.vehicle.getSpeed(veh))             # 当前车的速度
                current_state_dict[veh].append(l_speed)                                 # 前车的速度
                current_state_dict[veh].append(f_speed)                                 # 后车的速度
                ############################################################
                current_state_dict[veh].append(dist_to_cl)                             # 当前车距离前车的距离
                current_state_dict[veh].append(dist_to_cf)                             # 当前车和后车的距离
                #############################################################  
                tl_flag, tl_min, tl_max, tl_dist = tl_info2[0], tl_info2[1], tl_info2[2], tl_info2[3]
                # current_state_dict[veh].append(tl_flag)
                # current_state_dict[veh].append(tl_info1[1])
                current_state_dict[veh].append(tl_dist)                                # 距离交叉路口的距离
                current_state_dict[veh].append(tl_min)                                 # 通过交叉路口的最小时间
                current_state_dict[veh].append(tl_max)                                 # 通过交叉路口的最大时间
                #############################################################
                target_speed, min_speed, max_speed = getGlosaSpeedNCC(veh, 0.9)
                # current_state_dict[veh].append(target_speed)                           # 当前车建议通过交通路况的参考速度
                # current_state_dict[veh].append(traci.vehicle.getAcceleration(veh))
                current_state_dict[veh].append(min_speed)
                current_state_dict[veh].append(max_speed)
        return current_state_dict    

    def get_current_state(self):
        current_state_dict = defaultdict(list)
        veh_list = traci.vehicle.getIDList()
        for veh in veh_list:
            tl_info2 = get_tls_info2(veh)
            tl_info1 = get_tls_info(veh)
            if tl_info2:
                f, l = get_neighbor_cars(veh)
                l_id, l_pos, l_speed = l[0], l[1], l[2]
                f_id, f_pos, f_speed = f[0], f[1], f[2]
                c_pos = traci.vehicle.getLanePosition(veh)
                dist_to_cl = l_pos - c_pos - 4.9
                dist_to_cf = c_pos - f_pos - 4.9 
                #############################################################
                current_state_dict[veh].append(traci.vehicle.getSpeed(veh))             # 当前车的速度
                current_state_dict[veh].append(l_speed)                                 # 前车的速度
                current_state_dict[veh].append(f_speed)                                 # 后车的速度
                ############################################################
                current_state_dict[veh].append(dist_to_cl)                             # 当前车距离前车的距离
                current_state_dict[veh].append(dist_to_cf)                             # 当前车和后车的距离
                #############################################################  
                tl_flag, tl_min, tl_max, tl_dist = tl_info2[0], tl_info2[1], tl_info2[2], tl_info2[3]
                # current_state_dict[veh].append(tl_flag)
                # current_state_dict[veh].append(tl_info1[1])
                current_state_dict[veh].append(tl_dist)                                # 距离交叉路口的距离
                current_state_dict[veh].append(tl_min)                                 # 通过交叉路口的最小时间
                current_state_dict[veh].append(tl_max)                                 # 通过交叉路口的最大时间
                #############################################################
                target_speed, min_speed, max_speed = get_glosa_speed(veh, 0.9)
                # current_state_dict[veh].append(target_speed)                           # 当前车建议通过交通路况的参考速度
                # current_state_dict[veh].append(traci.vehicle.getAcceleration(veh))
                current_state_dict[veh].append(min_speed)
                current_state_dict[veh].append(max_speed)
        return current_state_dict    
    
    def get_current_state2(self):
        current_state_dict = defaultdict(list)
        veh_list = traci.vehicle.getIDList()
        for veh in veh_list:
            tl_info2 = get_tls_info2(veh)
            tl_info1 = get_tls_info(veh)
            if tl_info2:
                f, l = get_neighbor_cars(veh)
                l_id, l_pos, l_speed = l[0], l[1], l[2]
                f_id, f_pos, f_speed = f[0], f[1], f[2]
                c_pos = traci.vehicle.getLanePosition(veh)
                dist_to_cl = l_pos - c_pos - 4.9
                dist_to_cf = c_pos - f_pos - 4.9 
                #############################################################
                current_state_dict[veh].append(traci.vehicle.getSpeed(veh))             # 当前车的速度
                current_state_dict[veh].append(l_speed)                                 # 前车的速度
                current_state_dict[veh].append(f_speed)                                 # 后车的速度
                ############################################################
                current_state_dict[veh].append(dist_to_cl)                             # 当前车距离前车的距离
                current_state_dict[veh].append(dist_to_cf)                             # 当前车和后车的距离
                #############################################################  
                current_state_dict[veh].append(tl_info1[0])
                current_state_dict[veh].append(tl_info1[1])                            
                current_state_dict[veh].append(tl_info1[2])                            # 距离交叉路口的距离
                target_speed, min_speed, max_speed = get_glosa_speed(veh, 0.9)
                current_state_dict[veh].append(min_speed)                            
                current_state_dict[veh].append(max_speed) 
                
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
            if add_noise:
                action = self.noise.get_action(action, curr_step)
            action_dict[key] = np.clip(action, -2.0, 2.0)
        return action_dict

    def step(self, current_state_dict, action_dict):
        for key in current_state_dict:
            current_speed = current_state_dict[key][0]
            desired_speed = current_speed + action_dict[key]
            desired_speed = np.clip(desired_speed, 0, 20)
                
            traci.vehicle.setSpeed(key, desired_speed)

        # 已经通过十字路口的车辆设置速度，跟驰模型为IDM
        veh_list = traci.vehicle.getIDList()
        for veh in veh_list:
            if veh not in current_state_dict:
                next_speed = traci.vehicle.getSpeed(veh)
                next_speed = np.clip(next_speed + 1.5, 0, 20)
                traci.vehicle.setSpeed(veh, next_speed)
        #根据刚才给每一辆车的赋予的速度，进行单步仿真
        traci.simulationStep()
        #在仿真中，你想要设定的加速度，未必是车辆真正在下一步仿真的加速度(原因是可能由于拥堵，导致车辆此时的加速度无法达到想要的加速度)
        old_action_dict = copy.deepcopy(action_dict)
        for key in action_dict:    
            action_dict[key] = traci.vehicle.getAcceleration(key)
        
        return action_dict

    def get_reward(self, state_dict, next_state_dict, action_dict=None):
        reward_dict = defaultdict(list)
        reward_speed = 0.0
        reward_accel = 0.0
        reward_target = 0.0
        reward_safe = 0.0

        #必须是存在于上一时刻和当前时刻的车辆，才能获取奖励
        for key in next_state_dict:
            if key in state_dict:
                # [c_speed, l_speed, f_speed, dist_cl, dist_cf, tl_flag, tl_dur, dist_tl]
                accel = action_dict[key]
                c_speed = next_state_dict[key][0]           # 当前车的速度
                # l_speed = next_state_dict[key][1]           # 前车的速度
                # f_speed = next_state_dict[key][2]           # 后车的速度
                # dist_to_cl = next_state_dict[key][3]        # 当前车到前车的距离
                # dist_to_cf = next_state_dict[key][4]        # 当前车到后车的距离
                # target_speed = next_state_dict[key][-1]     # 建议速度
                min_speed = next_state_dict[key][-2]
                max_speed = next_state_dict[key][-1]

                reward_target = (c_speed -  min_speed) / (max_speed - min_speed + 1e-6)
                if c_speed > max_speed:
                    reward_target = -1.
                if c_speed < min_speed:
                    reward_target = -1.
                reward_target = np.clip(reward_target, -1., 1.)
                
                reward_accel = -0.01 * (accel * accel)

            reward_speed *= 0
            reward_accel *= 0.8
            reward_target *= 0.2
            reward_safe *= 0

            total_reward = reward_speed + reward_accel + reward_target + reward_safe
            reward_dict[key].append(total_reward)
            reward_dict[key].append(reward_speed)
            reward_dict[key].append(reward_accel)
            reward_dict[key].append(reward_target)
            reward_dict[key].append(reward_safe)
        return reward_dict

    def get_reward2(self, state_dict, next_state_dict, action_dict=None):
        reward_dict = defaultdict(list)
        reward_speed = 0.0
        reward_accel = 0.0
        reward_tls = 0.0
        reward_safe = 0.0

        #必须是存在于上一时刻和当前时刻的车辆，才能获取奖励
        for key in next_state_dict:
            if key in state_dict:
                # [c_speed, l_speed, f_speed, dist_cl, dist_cf, tl_flag, tl_dur, dist_tl]
                accel = action_dict[key]
                c_speed = next_state_dict[key][0]           # 当前车的速度
                # l_speed = next_state_dict[key][1]           # 前车的速度
                # f_speed = next_state_dict[key][2]           # 后车的速度
                # dist_to_cl = next_state_dict[key][3]        # 当前车到前车的距离
                # dist_to_cf = next_state_dict[key][4]        # 当前车到后车的距离
                tl_flag = next_state_dict[key][5]           # 当前交通灯红 or 绿
                tl_dur = next_state_dict[key][6]            # 当前变绿或者绿灯的剩余时长
                tl_dist = next_state_dict[key][7]           # 距离十字路口的距离
                min_speed = next_state_dict[key][8]
                max_speed = next_state_dict[key][9]
                # print('--------------------------------------------')
                # print('在绿灯周期通过的最小速度 {}, 最大速度是 {}'.format(min_speed, max_speed))
                # print('当前实际车速 {}, 距离十字路口的距离 {}'.format(c_speed, tl_dist))
                # print('当前交通灯相位 {}, 当前交通灯的剩余时间 {}'.format(tl_flag, tl_dur))
                
                reward_speed = 0.1 * (min(c_speed - min_speed, 0) + min(max_speed - c_speed, 0))
                
                if tl_flag == 0:
                    if c_speed * tl_dur > tl_dist:   
                        reward_tls = -2 #-1.2是个不错的值
                    else:
                        reward_tls = 0.7
                else:
                    if c_speed * tl_dur < tl_dist:
                        reward_tls = -2 #从-1.8往后回退
                    else:
                        reward_tls = 0.7
                # print('获得的交通灯奖励为{}'.format(reward_tls))        
                # print('--------------------------------------------')
                        
                reward_accel = -0.01 * (accel * accel)

        
            total_reward = reward_speed + reward_accel + reward_tls + reward_safe
            reward_dict[key].append(total_reward)
            reward_dict[key].append(reward_speed)
            reward_dict[key].append(reward_accel)
            reward_dict[key].append(reward_tls)
            reward_dict[key].append(reward_safe)
            
        return reward_dict

    def get_reward3(self, state_dict, next_state_dict, action_dict=None, selfish_score=0):
        reward_dict = defaultdict(list)
        reward_speed = 0.0
        reward_accel = 0.0
        reward_target = 0.0
        reward_safe = 0.0

        #必须是存在于上一时刻和当前时刻的车辆，才能获取奖励
        for key in next_state_dict:
            if key in state_dict:
                # [c_speed, l_speed, f_speed, dist_cl, dist_cf, tl_flag, tl_dur, dist_tl]
                
                c_speed = next_state_dict[key][0]           # 当前车的速度
                l_speed = next_state_dict[key][1]           # 前车的速度
                f_speed = next_state_dict[key][2]           # 后车的速度
                dist_to_cl = next_state_dict[key][3]        # 当前车到前车的距离
                dist_to_cf = next_state_dict[key][4]        # 当前车到后车的距离

                accel = next_state_dict[key][-3]
                min_speed = next_state_dict[key][-2]
                max_speed = next_state_dict[key][-1]
                #--------------------当前车速与建议车速的速度偏差--------#  
                # reward_target = -0.01 * (target_speed - c_speed) ** 2         
                #------------------------------------------------------#
                reward_target = (c_speed -  min_speed) / (max_speed - min_speed + 1e-6)
                if c_speed > max_speed:
                    reward_target = -1.
                if c_speed < min_speed:
                    reward_target = -1.
                reward_target = np.clip(reward_target, -1., 1.)
                
                reward_accel = -0.01 * (accel * accel)

        
            total_reward = reward_speed + reward_accel + reward_target + reward_safe
            reward_dict[key].append(total_reward)
            reward_dict[key].append(reward_speed)
            reward_dict[key].append(reward_accel)
            reward_dict[key].append(reward_target)
            reward_dict[key].append(reward_safe)
        
        coo_reward_dict = copy.deepcopy(reward_dict)    
        for key in reward_dict:
            f,l = get_neighbor_cars(key)
            f_id, l_id = f[0], l[0]
            if f_id != 'f':
                if f_id in reward_dict:
                    for i in range(len(reward_dict[key])):
                        coo_reward_dict[key][i] += selfish_score * reward_dict[f_id][i]
            if l_id != 'l':
                if l_id in reward_dict:
                    for i in range(len(reward_dict[key])):
                        coo_reward_dict[key][i] += selfish_score * reward_dict[l_id][i]
                
        return coo_reward_dict
    
    def get_reward4(self, state_dict, next_state_dict, action_dict=None, selfish_score=0):
        reward_dict = defaultdict(list)
        reward_speed = 0.0
        reward_accel = 0.0
        reward_target = 0.0
        reward_safe = 0.0

        #必须是存在于上一时刻和当前时刻的车辆，才能获取奖励
        for key in next_state_dict:
            if key in state_dict:
                
                c_speed = next_state_dict[key][0]           # 当前车的速度

                accel = next_state_dict[key][-3]
                min_speed = next_state_dict[key][-2]
                max_speed = next_state_dict[key][-1]

                reward_target = (c_speed -  min_speed) / (max_speed - min_speed + 1e-6)
                if c_speed > max_speed:
                    reward_target = -1.
                if c_speed < min_speed:
                    reward_target = -1.
                reward_target = np.clip(reward_target, -1., 1.)
                
                reward_accel = -0.01 * (accel * accel)

        
                total_reward = reward_speed + reward_accel + reward_target + reward_safe
                reward_dict[key].append(total_reward)
                reward_dict[key].append(reward_speed)
                reward_dict[key].append(reward_accel)
                reward_dict[key].append(reward_target)
                reward_dict[key].append(reward_safe)
                total_reward_sum = 0.
                
        total_tl_reward_sum = 0.
        total_accel_reward_sum = 0.
        
        for key in reward_dict:
            total_reward_sum += reward_dict[key][0]
            total_tl_reward_sum += reward_dict[key][2]
            total_accel_reward_sum += reward_dict[key][3]
        
        if len(reward_dict) > 0:
            avg_reward = total_reward_sum / len(reward_dict)
            avg_tl_reward = total_tl_reward_sum / len(reward_dict)
            avg_accel_reward = total_accel_reward_sum / len(reward_dict)
        else:
            avg_reward = 0.
            avg_accel_reward = 0.
            avg_tl_reward = 0.
        
        for key in reward_dict:
            reward_dict[key][0] = avg_reward
            reward_dict[key][2] = avg_tl_reward
            reward_dict[key][3] = avg_accel_reward
                
        return reward_dict
         
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state = self.memory.sample(batch_size)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
       
        # 更新critic的参数
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward +  self.cfg.gamma * target_value
        eval_value = self.critic(state, action)
        value_loss = nn.MSELoss()(eval_value, expected_value.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        # 更新actor的参数
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
                 