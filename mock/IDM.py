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
        self.algo = 'IDM'
        self.env = 'SUMO' # env name

        self.eval_eps = 1
        
        self.epsilon_start = 3

        self.max_speed = 20

        self.max_action = 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.simulation_steps = 3600

    
def test():
    cfg = DDPGConfig()

    init_rand_seed(42)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))

    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    
    for i_episode in range(cfg.eval_eps):
        ########################################################################
        # generate_normal_rou_file(ep = i_episode + 1, simulation_steps=600, arr_time = 0, depart_speed = 10, car_count_per_lane=250, path='test_rou_net')
        generate_uniform_rou_file(ep = i_episode + 1, arr_time=0, car_count_per_lane=1800, path='test_rou_net')    #######第二次是不需要更新的
        generate_test_cfg_file(ep = i_episode + 1, path='test_rou_net')    #######
        cfg_file_name = 'test_rou_net/intersection' + str(i_episode + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)

        traci.start(sumo_cmd)
        print('Simulation......')

        avg_halt_num_list = []
        
        i_step = 0
        
        info_dict = defaultdict(partial(defaultdict, list))
        other_info_dict = defaultdict(list)
        
        while traci.simulation.getMinExpectedNumber() > 0 and i_step <= 2 * cfg.simulation_steps:
            i_step += 1
            veh_list = traci.vehicle.getIDList()
            ####----------------用json文件记录加速度，位置，行驶里程等信息-----------####
            step_jerk = 0.
            for veh in veh_list:
                get_IDM_speed(veh, glosa_flag=True)
                
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
            
            ##-----------------停下来的数量---------------##
            step_stop_count = 0
            lane_list = traci.lane.getIDList()
            for lane in lane_list:
                step_stop_count += traci.lane.getLastStepHaltingNumber(lane)
            avg_halt_num_list.append(step_stop_count)
            
            traci.simulationStep()
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
        # save_json_to_dict(info_dict, target_path='IDM/IDM_glosa_1200_x.json')
        # save_json_to_dict(other_info_dict, target_path='IDM/IDM_avg.json')
        total_pass_time = 0
        for key in info_dict:
            total_pass_time += len(info_dict[key]['dist'])
        print('avg speed:{} m/s'.format(np.mean(np_speed_list)))
        print('hundred miles ec:{} kwh/100km'.format(np.mean(np_ec_list)))
        print('avg jerk:{} '.format(np.mean(np_jerk_list)))
        print('total pass time {}, avg pass time {}'.format(total_pass_time, total_pass_time / len(info_dict)))
        print('total halt num {}'.format(np.mean(avg_halt_num_list)))
    
    print('test finish !')

if __name__ == "__main__":
    test()
    