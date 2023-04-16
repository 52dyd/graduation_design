import os
import traci
import torch
import random
import numpy as np

from traffic import *

from sumocfg import generate_test_cfg_file
from sumocfg import generate_normal_rou_file
from sumocfg import generate_uniform_rou_file
from sumocfg import set_sumo
import json
from functools import partial
from collections import defaultdict
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


class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'
        self.env = 'SUMO' # env name

        self.gamma = 0.99
        self.epsilon = 0.001

        self.critic_lr = 5e-3
        self.actor_lr = 1e-4

        self.memory_capacity = 1000000
        self.batch_size = 512

        self.eval_eps = 1
        
        self.epsilon_start = 3

        self.max_speed = 20

        self.target_update = 4
        self.hidden_dim = 256
        self.soft_tau = 0.005
        self.max_action = 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.simulation_steps = 3600

    
def test():
    cfg = DDPGConfig()

    init_rand_seed(42)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))

    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    
    #############################3
    # torch.cuda.set_device(1)
    ##############################
    print(curr_path)
    # 13
    target_path = './models/cpu5.pth'

    print('-----------')
    print(target_path)
    print('------------')
    # train
    agent = DDPGAgent(state_dim=10, action_dim=1, cfg=cfg)
    agent.actor.load_state_dict(torch.load(target_path))
    print('模型参数导入成功')

    episode_avg_halt_list = []
    

    for i_episode in range(cfg.eval_eps):
        ########################################################################
        generate_uniform_rou_file(ep = i_episode + 1, arr_time = 0, depart_speed = 10, car_count_per_lane=1200, path='test_rou_net')    #######第二次是不需要更新的
        # generate_normal_rou_file(ep = i_episode + 1, simulation_steps=2400, arr_time = 0, depart_speed = 10, car_count_per_lane=800, path='test_rou_net')
        generate_test_cfg_file(ep = i_episode + 1, path='test_rou_net')    #######
        cfg_file_name = 'test_rou_net/intersection' + str(i_episode + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)

        agent.reset() #重置噪声
        
        traci.start(sumo_cmd)
        print('Simulation......')

        avg_halt_num_list = []
        avg_cl_dist_list = []
        
        i_step = 0
        
        info_dict = defaultdict(partial(defaultdict, list))
        
        other_info_dict = defaultdict(list)
        while traci.simulation.getMinExpectedNumber() > 0 and i_step <= 2 * cfg.simulation_steps:
            i_step += 1
            # 只有有车才进行学习
            if traci.vehicle.getIDCount() > 0:
                current_state_dict = agent.get_current_state()         
                action_dict = agent.choose_action(current_state_dict, i_step, add_noise=False)
                action_dict = agent.step(current_state_dict, action_dict)
                
                # traci.simulationStep()    
                ####----------------用json文件记录加速度，位置，行驶里程等信息-----------####
                
                step_jerk = 0.
                
                for veh in traci.vehicle.getIDList():
                    
                    if len(info_dict[veh]['accel']) != 0:
                        step_jerk += abs(traci.vehicle.getAcceleration(veh) - info_dict[veh]['accel'][-1])
                    
                    info_dict[veh]['speed'].append(traci.vehicle.getSpeed(veh))
                    info_dict[veh]['accel'].append(traci.vehicle.getAcceleration(veh))
                    info_dict[veh]['dist'].append(traci.vehicle.getDistance(veh))
                    info_dict[veh]['pos'].append(traci.vehicle.getLanePosition(veh))
                    info_dict[veh]['ec'].append(get_electricity_cons_per_car(veh, speed_thresh=2, idle=0.5))
                
                if len(traci.vehicle.getIDList()) == 0:
                    step_jerk = 0.
                else:   
                    step_jerk /= len(traci.vehicle.getIDList())
                    
                other_info_dict['jerk'].append(step_jerk)
                other_info_dict['speed'].append(get_step_mean_speed())
                other_info_dict['ec'].append(get_step_total_ec(speed_thresh=2, idle=0.5))
                #----------------------------------------------------------------------#
                    
                ##-----------------停下来的数量---------------##
                step_stop_count = 0
                lane_list = traci.lane.getIDList()
                for lane in lane_list:
                    step_stop_count += traci.lane.getLastStepHaltingNumber(lane)
                avg_halt_num_list.append(step_stop_count)
                ##----------------计算能耗----------------##
                veh_list = traci.vehicle.getIDList()
                ##------------------计算平均跟驰距离--------------------##
                has_l_cnt = 0
                dist_cl = 0.
                for veh in veh_list:
                    f, l = get_neighbor_cars(veh)
                    f_id, l_id = f[0], l[0]
                    if l_id != 'l':
                        has_l_cnt += 1
                        dist_cl += traci.vehicle.getDistance(l_id) - traci.vehicle.getDistance(veh) - 4.9
                if has_l_cnt > 0:
                    avg_dist = dist_cl / has_l_cnt
                    avg_cl_dist_list.append(avg_dist)
                    
            else:
                traci.simulationStep()
                print("nxt", i_step)
        traci.close()
        
        np_speed_list = []
        np_ec_list = []
        np_jerk_list = []
        # 首先统计总里程
        for key in info_dict:
            np_speed_list.append(info_dict[key]['dist'][-1] / len(info_dict[key]['dist']))
            
            # 统计加速度的抖动
            jerk = 0.
            for i in range(1, len(info_dict[key]['accel'])):
                jerk += abs(info_dict[key]['accel'][i] - info_dict[key]['accel'][i - 1])
            np_jerk_list.append(jerk / len(info_dict[key]['accel']))
            
        for i in range(len(info_dict[key]['accel'])):
            total_ec = 0.
            for i in range(len(info_dict[key]['speed'])):
                v = info_dict[key]['speed'][i]
                a = info_dict[key]['accel'][i]
                f = calc_force(v, a)
                p = get_output_power(f, v, 0.9, 0.85)
                total_ec += p
                # 将总能耗转化为百公里能耗
            total_ec = total_ec / 3600000 / 0.6 * 100
            np_ec_list.append(total_ec)
        
        print('simulation end in {} s !'.format(i_step))
        #-----------写一些平均速度，平均能耗，平均-------------#
        # save_json_to_dict(info_dict, target_path='rl/SARL_1200_x_1.json')
        save_json_to_dict(other_info_dict, target_path='rl/SARL_avg2.json')
        #---------------------------------------------------#
        total_pass_time = 0
        for key in info_dict:
            total_pass_time += len(info_dict[key]['dist'])
        print('avg speed:{} m/s'.format(np.mean(np_speed_list)))
        print('hundred miles ec:{} kwh/100km'.format( np.mean(np_ec_list)))
        print('avg jerk:{} m/s^3'.format(np.mean(np_jerk_list)))
        print('avg following dist = {} m'.format(np.sum(avg_cl_dist_list) / len(avg_cl_dist_list)))
        print('total pass time {}, avg pass time {}'.format(total_pass_time, total_pass_time / len(info_dict)))
        print('total halt num {}'.format(np.mean(avg_halt_num_list)))
    
    print('test finish !')

if __name__ == "__main__":
    test()
    
