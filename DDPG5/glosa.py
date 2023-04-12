import json
import os

import traci
from traffic import *
import torch
import random
import numpy as np


from sumocfg import generate_test_cfg_file
from sumocfg import generate_uniform_rou_file
from sumocfg import set_sumo

from functools import partial


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
        self.algo = 'GLOSA'
        self.env = 'SUMO' # env name
   
        self.max_speed = 20

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.simulation_steps = 3600

    
def train():
    cfg = DDPGConfig()

    init_rand_seed(42)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))
    
    ##############################
    # torch.cuda.set_device(1)
    ##############################

    
    
    episode_avg_halt_list = []
    for ep in range(1):
        ########################################################################
        generate_uniform_rou_file(ep = ep + 1, car_count_per_lane=700, path='test_rou_net')
        generate_test_cfg_file(ep + 1, path='test_rou_net')    #######
        cfg_file_name = 'test_rou_net/intersection' + str(ep + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)
        
        traci.start(sumo_cmd)
        print('Simulation......')

        avg_halt_num_list = []
        avg_speed_list = []
        avg_cl_dist_list = []
        avg_elec_list = []
        
        total_elec_cons = 0.
        
        i_step = 0
        info_dict = defaultdict(partial(defaultdict, list))
        while traci.simulation.getMinExpectedNumber() > 0 and i_step <= 2 * cfg.simulation_steps:
            i_step += 1
            veh_list = traci.vehicle.getIDList()
            
            for veh in veh_list:
                if get_tls_info(veh):
                    target_speed, min_speed, max_speed = get_glosa_speed(veh, 0.9)
                    traci.vehicle.setSpeed(veh, target_speed)
                else:
                    target_speed = traci.vehicle.getSpeed(veh) + 1.5
                    traci.vehicle.setSpeed(veh, target_speed)
                    
            step_stop_num = 0
            step_speed = 0.
            
            for lane in traci.lane.getIDList():        
                step_stop_num += traci.lane.getLastStepHaltingNumber(lane)
                step_speed += traci.lane.getLastStepMeanSpeed(lane)
                
            total_veh_num = len(veh_list)    
            
            ####----------------用json文件记录加速度，位置，行驶里程等信息-----------####
            for veh in traci.vehicle.getIDList():
                info_dict[veh]['accel'].append(traci.vehicle.getAcceleration(veh))
                info_dict[veh]['dist'].append(traci.vehicle.getDistance(veh))
                info_dict[veh]['pos'].append(traci.vehicle.getLanePosition(veh))
            #----------------------------------------------------------------------#
            
            ##-------------------停下来的数量-----------------##
            step_stop_count = 0
            lane_list = traci.lane.getIDList()
            for lane in lane_list:
                step_stop_count += traci.lane.getLastStepHaltingNumber(lane)
            avg_halt_num_list.append(step_stop_count)
            ##-----------------------计算平均速度------------------------##
            mean_speed = 0.
            for lane in lane_list:
                mean_speed += traci.lane.getLastStepMeanSpeed(lane)
            step_speed = mean_speed / len(lane_list)
            avg_speed_list.append(step_speed)
            step_e_cons = 0.
            veh_list = traci.vehicle.getIDList()
            ##-----------------------计算平均能耗--------------##
            if len(veh_list) > 0:
                for veh in veh_list:
                    # 能耗的单位是wh/s
                    step_e_cons += traci.vehicle.getElectricityConsumption(veh)
                total_elec_cons += step_e_cons
                avg_elec_list.append(step_e_cons / len(veh_list))
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
            #############################################################
            
            traci.simulationStep()
        traci.close()
        
        print('simulation end in {} s !'.format(i_step))
        json_str = json.dumps(info_dict)
        print(curr_path)
        target_json = 'glosa/glosa_700_0.9.json'
        
        with open(target_json, 'w') as json_file:
            json_file.write(json_str)

        print('Episode:{}, avg speed:{}'.format(ep + 1,  np.sum(avg_speed_list) / len(avg_speed_list)))
        print('Episode:{}, avg halt num:{}'.format(ep + 1, np.sum(avg_halt_num_list) / len(avg_halt_num_list)))
        print('Episode:{}, avg following dist = {} m'.format(ep + 1, np.sum(avg_cl_dist_list) / len(avg_cl_dist_list)))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))


    print("final halt list:{}".format(episode_avg_halt_list))

if __name__ == "__main__":
    train()
    