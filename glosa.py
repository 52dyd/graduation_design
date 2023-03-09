
import os

import traci
from traffic import *
import torch
import random
import numpy as np

from traffic import get_avg_speed
from traffic import get_avg_halting_num

from sumocfg import generate_cfg_file
from sumocfg import generate_rou_file
from sumocfg import set_sumo


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


        self.entering_lanes = ['WE_0', 'EW_0', 'NS_0', 'SN_0']
        self.depart_lanes = ['-WE_0', '-EW_0', '-NS_0', '-SN_0']
        self.intersection_lanes = [':intersection_0_0', ':intersection_1_0', ':intersection_2_0', ':intersection_3_0']
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
    for ep in range(100):
        ########################################################################
        #generate_rou_file(ep + 1, path='rou_net')    #######第二次是不需要更新的
        generate_cfg_file(ep + 1, path='test_rou_net')    #######
        cfg_file_name = 'test_rou_net/intersection' + str(ep + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)
        
        traci.start(sumo_cmd)
        print('Simulation......')

        avg_halt_num_list = []
        avg_speed_list = []
        
        i_step = 0
        
        while traci.simulation.getMinExpectedNumber() > 0 and i_step <= 2 * cfg.simulation_steps:
            veh_list = traci.vehicle.getIDList()
            
            for veh in veh_list:
                if traci.vehicle.getLaneID(veh) in cfg.entering_lanes:
                    glosa(veh, 0.5)
                else:
                    target_speed = traci.vehicle.getSpeed(veh) + 1.5
                    traci.vehicle.setSpeed(veh, target_speed)
                    
            step_stop_num = 0
            step_speed = 0.
            
            for lane in traci.lane.getIDList():        
                step_stop_num += traci.lane.getLastStepHaltingNumber(lane)
                step_speed += traci.lane.getLastStepMeanSpeed(lane)
                
            total_veh_num = len(veh_list)    
            
            if total_veh_num > 0:
                avg_halt_num_list.append(step_stop_num / total_veh_num)
                avg_speed_list.append(step_speed / len(traci.lane.getIDList()))
            
            traci.simulationStep()
        traci.close()

        print('Episode:{}, avg speed:{}'.format(ep + 1,  np.sum(avg_speed_list) / len(avg_speed_list)))
        print('Episode:{}, avg halt num:{}'.format(ep + 1, np.sum(avg_halt_num_list) / len(avg_halt_num_list)))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))


    print("final halt list:{}".format(episode_avg_halt_list))

if __name__ == "__main__":
    train()
    