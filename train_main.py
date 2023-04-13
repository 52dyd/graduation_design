from collections import defaultdict
import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import traci
import datetime
import torch
import random
import contextlib
import datetime
import numpy as np

from traffic import get_avg_speed
from traffic import get_avg_halting_num

from sumocfg import generate_cfg_file
from sumocfg import generate_rou_file
from sumocfg import set_sumo
from functools import partial
from torch.utils.tensorboard import SummaryWriter

from agent import DDPGAgent

def init_rand_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time


class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'
        self.env = 'SUMO' # env name

        self.gamma = 0.99
        self.epsilon = 0.0005

        self.critic_lr = 5e-3
        self.actor_lr = 1e-4

        self.memory_capacity = 1000000
        self.batch_size = 512

        self.train_eps = 150
        self.eval_eps = 10
        
        self.epsilon_start = 3

        self.max_speed = 40

        self.target_update = 4
        self.hidden_dim = 256
        self.soft_tau = 0.01
        self.max_action = 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.simulation_steps = 3600

    
def train():
    rewards = []
    speed_reward = []
    tls_reward = []
    cfg = DDPGConfig()

    init_rand_seed(42)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))

    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    
    #############################3
    torch.cuda.set_device(1)
    ##############################
    now = datetime.datetime.now()
    dest_path = 'models/' + now.strftime("%Y-%m-%d %H:%M:%S") + '.pth'

    print('-----------------------------------------')
    print(dest_path)
    print('-----------------------------------------')
    # train
    agent = DDPGAgent(state_dim=8, action_dim=1, cfg=cfg)

    episode_avg_halt_list = []
    learning_step = 0
    for i_episode in range(cfg.train_eps):
        generate_rou_file(ep = i_episode + 1)
        generate_cfg_file(ep = i_episode + 1, path='rou_net')    #######
        cfg_file_name = 'rou_net/intersection' + str(i_episode + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)

        agent.reset() #重置噪声
        
        traci.start(sumo_cmd)
        print('Simulation......')
        ep_reward = 0
        ep_speed_reward = 0
        ep_tls_reward = 0

        avg_halt_num_list = []
        
        i_step = 0
        info_dict = defaultdict(partial(defaultdict, list))
        step_reward_list = []
        step_speed_reward_list = []
        step_tl_reward_list = []
        
        while traci.simulation.getMinExpectedNumber() > 0 and i_step <= 2 * cfg.simulation_steps:
            i_step += 1
            # 只有有车才进行学习
            if traci.vehicle.getIDCount() > 0:
                current_state_dict = agent.get_current_state()         
                action_dict = agent.choose_action(current_state_dict, i_step, add_noise=True)
                next_state_dict, action_dict = agent.step(current_state_dict, action_dict)
                reward_dict = agent.get_reward(next_state_dict, action_dict)
                # 再将刚才与环境互动得到的四元组存储起来
                agent.memory.push(current_state_dict, action_dict, reward_dict, next_state_dict)
                # 如果互动得到的经验超过batch_size，则进行学习
                if i_episode + 1 >= cfg.epsilon_start:
                    agent.update(cfg.batch_size)

                step_speed_reward = 0.
                step_tl_reward = 0.
                step_reward = 0.
                step_stop_count = 0
                
                for key in reward_dict:
                    step_reward += reward_dict[key][0]
                    step_speed_reward += reward_dict[key][1]
                    step_tl_reward += reward_dict[key][2]
                    
                if len(reward_dict) > 0:
                    step_reward_list.append(step_reward / len(reward_dict))
                    step_speed_reward_list.append(step_speed_reward / len(reward_dict))
                    step_tl_reward_list.append(step_tl_reward / len(reward_dict))
                else:
                    step_reward_list.append(0.)
                    step_speed_reward_list.append(0.)
                    step_tl_reward_list.append(0.)

                ep_reward += step_reward
                ep_speed_reward += step_speed_reward
                ep_tls_reward += step_tl_reward
                
                # 记录中间的各项信息
                for veh in traci.vehicle.getIDList():
                    info_dict[veh]['speed'].append(traci.vehicle.getSpeed(veh))
                    info_dict[veh]['accel'].append(traci.vehicle.getAcceleration(veh))
                    info_dict[veh]['dist'].append(traci.vehicle.getDistance(veh))
                    info_dict[veh]['pos'].append(traci.vehicle.getLanePosition(veh))
                    info_dict[veh]['ec'].append(traci.vehicle.getElectricityConsumption(veh))
                ##-------------------停下来的数量-----------------##
                lane_list = traci.lane.getIDList()
                for lane in lane_list:
                    step_stop_count += traci.lane.getLastStepHaltingNumber(lane)
                avg_halt_num_list.append(step_stop_count)
                #############################################################
            else:
                traci.simulationStep()
        traci.close()
        
        np_speed_list = []
        np_ec_list = []
        np_jerk_list = []
        # 首先统计总里程
        for key in info_dict:
            np_speed_list.append(info_dict[key]['dist'][-1] / len(info_dict[key]['dist']))
            np_ec_list.append(np.sum(info_dict[key]['ec']))
            
            # 统计加速度的抖动
            jerk = 0.
            for i in range(1, len(info_dict[key]['accel'])):
                jerk += (info_dict[key]['accel'][i] - info_dict[key]['accel'][i - 1]) ** 2
            np_jerk_list.append(jerk / len(info_dict[key]['accel']))
        
        # np_ec_list中存放的是车辆通过十字路口的能耗
        avg_pass_ec = np.mean(np_ec_list)
        # 需要将其转化为对应的百公里能耗
        hundred_ec = avg_pass_ec / 0.9 * 0.1
        
        ep_reward_list = [ep_reward, ep_speed_reward, ep_tls_reward]
        print('Episode:{}/{}, Reward:{}'.format(i_episode + 1, cfg.train_eps, np.mean(step_reward_list)))
        print('Episode:{}/{}, speed Reward:{}'.format(i_episode + 1, cfg.train_eps, np.mean(step_speed_reward_list)))
        print('Episode:{}/{}, tls Reward:{}'.format(i_episode + 1, cfg.train_eps, np.mean(step_tl_reward_list)))
        print('Episode:{}/{}, avg speed:{} m/s'.format(i_episode + 1, cfg.train_eps, np.mean(np_speed_list)))
        print('Episode:{}/{}, hundred miles ec:{} kwh/100km'.format(i_episode + 1, cfg.train_eps, hundred_ec))
        print('Episode:{}/{}, avg jerk:{} '.format(i_episode + 1, cfg.train_eps, np.mean(np_jerk_list)))
        print('Episode:{}/{}, avg halt num:{}'.format(i_episode + 1, cfg.train_eps, np.mean(avg_halt_num_list)))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))
        rewards.append(np.mean(step_reward_list))
        speed_reward.append(np.mean(step_speed_reward_list))
        tls_reward.append(np.mean(step_tl_reward_list))

    actor_pth_file_path = os.path.join(curr_path, dest_path)  ##########这个无需注释
  
    torch.save(agent.actor.state_dict(), actor_pth_file_path)

    print('model saved successfully!')
    print('Complete training!')
    print("final reward list:{}".format(rewards))
    print("final halt list:{}".format(episode_avg_halt_list))
    print('speed reward list:{}'.format(speed_reward))
    print('tls reward list:{}'.format(tls_reward))

if __name__ == "__main__":
    train()

    
