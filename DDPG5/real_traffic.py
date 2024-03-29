import os
import sys

from collections import defaultdict
from functools import partial
import json

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import traci
import timeit
import torch
import numpy as np

from sumocfg import set_sumo
from sumocfg import generate_test_cfg_file
from sumocfg import generate_uniform_rou_file
from sumocfg import generate_single_rou_file

from traffic import get_avg_speed
from traffic import get_avg_halting_num
from traffic import get_avg_elec_cons

from agent import DDPGAgent

entering_lanes = ['WE_0', 'EW_0', 'NS_0', 'SN_0']

class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'
        self.env = 'SUMO' # env name

        self.random_seed = 2020
        self.gamma = 0.99
        self.epsilon = 0.001

        self.critic_lr = 3e-3
        self.actor_lr = 1e-4

        self.memory_capacity = 1000000
        self.batch_size = 512

        self.train_eps = 100
        self.eval_eps = 1

        self.max_speed = 20
        self.target_speed = 13.8

        self.target_update = 4
        self.hidden_dim = 256
        self.soft_tau = 0.005
        self.max_action = 2


        self.entering_lanes = ['WE_0', 'EW_0', 'NS_0', 'SN_0']
        self.depart_lanes = ['-WE_0', '-EW_0', '-NS_0', '-SN_0']
        self.intersection_lanes = [':intersection_0_0', ':intersection_1_0', ':intersection_2_0', ':intersection_3_0']
        self.yellow_duration = 4
        self.green_phase_duration = 30
        #self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.simulation_steps = 3600


def run_simulation_eval(agent, sumo_cmd, cfg):
    '''测试过程中用的应该是固定的路网文件'''
    traci.start(sumo_cmd)
    print('Simulation......')
    i_step = 0
    start_time = timeit.default_timer()

    avg_speed_list = []
    avg_halt_num_list = []
    avg_elec_cons_list = []
    total_elec_cons = 0.0
    danger_count = 0

    position_time_dict = defaultdict(partial(defaultdict, list))

    while traci.simulation.getMinExpectedNumber() > 0 and i_step <= cfg.simulation_steps:
        i_step += 1
        if traci.vehicle.getIDCount() > 0:
            current_state_dict = agent.get_current_state()
            vehicle_id_entering = []
            for lane in entering_lanes:
                vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))

            total_vehicle_list = traci.vehicle.getIDList()
            for i in range(len(total_vehicle_list)):
                if traci.vehicle.getLaneID(total_vehicle_list[i]) in cfg.entering_lanes:
                    curr_position = traci.vehicle.getLanePosition(total_vehicle_list[i]) - 250
                    position_time_dict[total_vehicle_list[i]]['position'].append(curr_position)

                    position_time_dict[total_vehicle_list[i]]['tls_phase'].append(traci.trafficlight.getPhaseName('J10'))  
                    #记录每个仿真中的能耗. wh/s
                    position_time_dict[total_vehicle_list[i]]['electricity'].append(traci.vehicle.getElectricityConsumption(total_vehicle_list[i]))    
                elif traci.vehicle.getLaneID(total_vehicle_list[i]) in cfg.intersection_lanes: 
                    #记录当前车辆每个仿真中的位置[0, 250]m
                    #print(traci.vehicle.getLaneID(total_vehicle_list[i]))
                    position_time_dict[total_vehicle_list[i]]['position'].append(traci.vehicle.getLanePosition(total_vehicle_list[i]))
                    # #另外一方面记录当前交通灯的相位，以及持续时间  
                    position_time_dict[total_vehicle_list[i]]['tls_phase'].append(traci.trafficlight.getPhaseName('J10'))  
                    #记录每个仿真中的能耗. wh/s
                    position_time_dict[total_vehicle_list[i]]['electricity'].append(traci.vehicle.getElectricityConsumption(total_vehicle_list[i]))    
                else:
                    #十字路口的长度为10m
                    curr_position = traci.vehicle.getLanePosition(total_vehicle_list[i]) + 10
                    position_time_dict[total_vehicle_list[i]]['position'].append(curr_position)
                    position_time_dict[total_vehicle_list[i]]['tls_phase'].append(traci.trafficlight.getPhaseName('J10'))  
                    #记录每个仿真中的能耗. wh/s
                    position_time_dict[total_vehicle_list[i]]['electricity'].append(traci.vehicle.getElectricityConsumption(total_vehicle_list[i]))        
            for i in range(len(vehicle_id_entering)):        
                total_elec_cons += traci.vehicle.getElectricityConsumption(vehicle_id_entering[i])
            action_dict = agent.choose_action(current_state_dict, i_step, add_noise=False)
            next_state_dict, action_dict = agent.step(current_state_dict, action_dict)

            avg_speed_step = get_avg_speed(entering_lanes)
            avg_halt_num_step = get_avg_halting_num(entering_lanes)
            avg_elec_cons = get_avg_elec_cons(entering_lanes)

            avg_speed_list.append(avg_speed_step)
            avg_halt_num_list.append(avg_halt_num_step)
            avg_elec_cons_list.append(avg_elec_cons)

            reward_dict, danger_count = agent.get_reward(current_state_dict, action_dict, danger_count)
        else:
            traci.simulationStep()
    traci.close()

    #计算当前仿真所有车辆通过十字路口时间的和
    episode_travel_time = 0
    for key in position_time_dict.keys():
        episode_travel_time += len(position_time_dict[key]['position'])

    #将position_dict中的内容存储下
    json_str = json.dumps(position_time_dict)
    with open('json/DDPG_data_500.json', 'w') as json_file:
        json_file.write(json_str)

    simulation_time = round(timeit.default_timer() - start_time, 1)
    print('Simulation time:{}'.format(simulation_time))

  
    return avg_speed_list, avg_halt_num_list, avg_elec_cons_list, total_elec_cons, episode_travel_time, danger_count

def eval():
    cfg = DDPGConfig()

    curr_path = os.path.dirname(os.path.abspath(__file__))
     
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    agent = DDPGAgent(state_dim=8, action_dim=1, cfg=cfg)

    ####################################
    # torch.cuda.set_device(1)
    ####################################

    dest_pth = 'models/reward_s0.05_a7e-2_t(-2_2)_safe(4,-0.8).pth'
    print('------------------------------------------')
    print(dest_pth)
    print('------------------------------------------')

    curr_actor_path = os.path.dirname(os.path.abspath(__file__))
    actor_file_path = os.path.join(curr_actor_path, dest_pth) #######
    agent.actor.load_state_dict(torch.load(actor_file_path))

    episode_avg_speed_list = []
    episode_avg_elec_list = []
    episode_avg_halt_list = []
    total_elec_cons_list = []
    travel_time_list = []
    
    print('网络参数导入成功!')
    agent.actor.eval()
    for i_ep in range(cfg.eval_eps):
        #########################################################################
        #测试的过程中生成一遍新的路网文件，这里不能与训练中用的路网一样
        #generate_rou_file(train_eps = i_episode + 1)
        generate_single_rou_file(i_ep + 1, car_count_per_lane=500)
        generate_test_cfg_file(train_eps = i_ep + 1, path='uniform_rou_net')    #######

        cfg_file_name = 'uniform_rou_net/intersection' + str(i_ep + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)
        ###########################################################################

        avg_speed_list, avg_halt_num_list, avg_elec_cons_list, total_elec_cons, episode_travel_time, danger_count = run_simulation_eval(agent, sumo_cmd, cfg)
        print('Episode:{}/{}, avg speed:{}'.format(i_ep + 1, cfg.eval_eps, np.sum(avg_speed_list) / len(avg_speed_list)))
        print('Episode:{}/{}, avg halt num:{}'.format(i_ep + 1, cfg.eval_eps, np.sum(avg_halt_num_list) / len(avg_halt_num_list)))
        print('Episode:{}/{}, total elec cons:{}'.format(i_ep + 1, cfg.eval_eps, total_elec_cons))
        print('danger count:{}'.format(danger_count))
        episode_avg_speed_list.append(np.sum(avg_speed_list) / len(avg_speed_list))
        episode_avg_elec_list.append(np.sum(avg_elec_cons_list) / len(avg_elec_cons_list))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))
        total_elec_cons_list.append(total_elec_cons)
        travel_time_list.append(episode_travel_time)

        
    print('Complete eval! ')
    print("final speed list:{}".format(episode_avg_speed_list))
    print("final halt list:{}".format(episode_avg_halt_list))
    print("final episode elec cons list:{}".format(total_elec_cons_list))
    print('time list:{}'.format(travel_time_list))

if __name__ == '__main__':
    eval()