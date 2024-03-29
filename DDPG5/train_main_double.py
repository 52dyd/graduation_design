import os
import traci
import torch
import random
import datetime

import multiprocessing
import numpy as np
from collections import defaultdict

from queue import Queue
from time import sleep

from traffic import *

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

class DDPGConfig:
    def __init__(self, GPUID = '0', gamma=0.99, tau=0.005):
        self.algo = 'MADDPG'
        self.env = 'SUMO' # env name

        self.seedVal = 20230427

        self.gamma = gamma
        self.epsilon = 0.001

        self.critic_lr = 5e-3
        self.actor_lr = 1e-4

        self.memory_capacity = 1000000
        self.batch_size = 512

        self.train_eps = 80 # 原来是80
        self.eval_eps = 10 # 原来是10
        
        self.epsilon_start = 3

        self.max_speed = 28

        self.target_update = 4
        self.hidden_dim = 256
        self.soft_tau = tau
        self.max_action = 2

        self.device = torch.device("cuda:" + GPUID if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        self.simulation_steps = 3600
        
        self.pathToSumoFiles = "rou_net2_double"

        self.model_dest_path = "models/"
        self.model_dest_path_leader = "double_acce8_speed2_"
        self.model_dest_path_follow = ""

        self.logName = '_acce2_speed_8_gamma%5f_tua%5f.txt'%(self.gamma, self.soft_tau)
        self.procID = -1 # 多进程id

def train(gamma_idx, tau_idx, procID, outputPath):
    rewards = []
    speed_reward = []
    tls_reward = []
    target_reward = []
    safe_reward = []
    ecs = []
    speeds = []
    jerks = []
    gamma_arr = [0.7, 0.8, 0.9, 0.99, 0.999]
    tau_arr = [0.001, 0.005, 0.01, 0.05, 0.1]
    # cfg = DDPGConfig("%1d"%(procID%2), gamma=gamma_arr[gamma_idx], tau=tau_arr[tau_idx])
    cfg = DDPGConfig('0', gamma=gamma_arr[gamma_idx], tau=tau_arr[tau_idx])
    cfg.procID = procID

    init_rand_seed(cfg.seedVal)
    outputFile = open(outputPath + cfg.logName, "w")
    curr_path = os.path.dirname(os.path.abspath(__file__))
    outputFile.write('\n'+f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    
    #############################3
    # torch.cuda.set_device(1)
    ##############################
    # 超参数 gamma soft_tau

    # dest_path = 'models/test4.pth'
    
    # single1.pth
    # writer_path = 'tensorboard/7'
    # writer = SummaryWriter(writer_path)

    outputFile.write('\n'+'-----------------------------------------')
    outputFile.write('\n'+str(vars(cfg)))
    outputFile.write('\n'+'seed {}'.format(cfg.seedVal))

    outputFile.write('当前的gamma是{}, 当前的soft tau是{}'.format(gamma_arr[gamma_idx], tau_arr[tau_idx]))
    cfg.gamma = gamma_arr[gamma_idx]
    cfg.soft_tau = tau_arr[tau_idx]
    outputFile.write('-----------------------------------------')
    # train
    agent = DDPGAgent(state_dim=10, action_dim=1, cfg=cfg)

    episode_avg_halt_list = []
    for i_episode in range(cfg.train_eps):
        ########################################################################
        # car_count_list = [100, 110, 120, 130, 140, 150]
        # generate_rou_file_double(ep = i_episode + 1, car_count_per_lane=random.choice(car_count_list), path=cfg.pathToSumoFiles, simulation_steps=cfg.simulation_steps)    #######第二次是不需要更新的
        # generate_cfg_file(ep = i_episode + 1, path=cfg.pathToSumoFiles)    #######
        cfg_file_name = cfg.pathToSumoFiles + '/intersection' + str(i_episode + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)

        agent.reset() #重置噪声
        
        print('proc No.%d simulating......%d/%d\n'%(cfg.procID, i_episode+1, cfg.train_eps))
        traci.start(sumo_cmd)
        ep_reward = 0.
        ep_speed_reward = 0.
        ep_tls_reward = 0.
        ep_target_reward = 0.
        ep_cf_reward = 0.

        avg_halt_num_list = []
        
        step_reward_list = []
        step_speed_reward_list = []
        step_tl_reward_list = []
        step_target_reward_list = []
        step_safe_reward_list = []
        
        
        i_step = 0
        info_dict = defaultdict(partial(defaultdict, list))
        while traci.simulation.getMinExpectedNumber() > 0 and i_step <= 2 * cfg.simulation_steps:
            i_step += 1
            # 只有有车才进行学习
            if traci.vehicle.getIDCount() > 0:
                current_state_dict = agent.get_current_state()         
                action_dict = agent.choose_action(current_state_dict, i_step, add_noise=True)
                action_dict = agent.step(current_state_dict, action_dict)
                next_state_dict = agent.get_current_state()
                reward_dict = agent.get_reward(current_state_dict, next_state_dict, action_dict)
                agent.memory.push(current_state_dict, action_dict, reward_dict, next_state_dict)
                # 如果互动得到的经验超过batch_size，则进行学习
                if i_episode + 1 >= 0:
                    agent.update(cfg.batch_size)

                step_speed_reward = 0.
                step_tl_reward = 0.
                step_reward = 0.
                step_stop_count = 0
                step_target_reward = 0.
                step_cf_reward = 0.
                
                for key in reward_dict:
                    step_reward += reward_dict[key][0]
                    step_speed_reward += reward_dict[key][1]
                    step_tl_reward += reward_dict[key][2]
                    step_target_reward += reward_dict[key][3]
                    step_cf_reward += reward_dict[key][4]

                if len(reward_dict) > 0:
                    step_reward_list.append(step_reward / len(reward_dict))
                    step_speed_reward_list.append(step_speed_reward / len(reward_dict))
                    step_tl_reward_list.append(step_tl_reward / len(reward_dict))
                    step_target_reward_list.append(step_target_reward / len(reward_dict))
                    step_safe_reward_list.append(step_cf_reward / len(reward_dict))
                else:
                    step_reward_list.append(0.)
                    step_speed_reward_list.append(0.)
                    step_tl_reward_list.append(0.)
                    step_target_reward_list.append(0.)
                    step_safe_reward_list.append(0.)
                    
                ep_reward += step_reward
                ep_speed_reward += step_speed_reward    
                ep_tls_reward += step_tl_reward
                ep_target_reward += step_target_reward
                ep_cf_reward += step_cf_reward
                # 记录中间的各项信息
                for veh in traci.vehicle.getIDList():
                    info_dict[veh]['speed'].append(traci.vehicle.getSpeed(veh))
                    info_dict[veh]['accel'].append(traci.vehicle.getAcceleration(veh))
                    info_dict[veh]['dist'].append(traci.vehicle.getDistance(veh))
                    info_dict[veh]['pos'].append(traci.vehicle.getLanePosition(veh))
                    info_dict[veh]['ec'].append(get_electricity_cons_per_car(veh, speed_thresh=2, idle=0.5))
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
                jerk += abs(info_dict[key]['accel'][i] - info_dict[key]['accel'][i - 1])
            np_jerk_list.append(jerk / len(info_dict[key]['accel']))
        
        # np_ec_list中存放的是车辆通过十字路口的能耗
        avg_pass_ec = np.mean(np_ec_list)
        # 需要将其转化为对应的百公里能耗
        hundred_ec = avg_pass_ec / 0.6 * 0.1
        # outputFile.write('Episode:{}/{}, Reward:{}'.format(i_episode + 1, cfg.train_eps, np.sum(step_reward_list) / len(step_reward_list)))
        # outputFile.write('Episode:{}/{}, speed Reward:{}'.format(i_episode + 1, cfg.train_eps, np.sum(step_speed_reward_list) / len(step_speed_reward_list)))
        # outputFile.write('Episode:{}/{}, tls Reward:{}'.format(i_episode + 1, cfg.train_eps, np.sum(step_tl_reward_list) / len(step_tl_reward_list)))
        # outputFile.write('Episode:{}/{}, target Reward:{}'.format(i_episode + 1, cfg.train_eps, np.sum(step_target_reward_list) / len(step_target_reward_list)))
        # outputFile.write('Episode:{}/{}, safe Reward:{}'.format(i_episode + 1, cfg.train_eps, np.sum(step_safe_reward_list) / len(step_safe_reward_list)))
        # outputFile.write('Episode:{}/{}, avg speed:{} m/s'.format(i_episode + 1, cfg.train_eps, np.mean(np_speed_list)))
        # outputFile.write('Episode:{}/{}, hundred miles ec:{} kwh/100km'.format(i_episode + 1, cfg.train_eps, hundred_ec))
        # outputFile.write('Episode:{}/{}, avg jerk:{} '.format(i_episode + 1, cfg.train_eps, np.mean(np_jerk_list)))
        # outputFile.write('Episode:{}/{}, avg halt num:{}'.format(i_episode + 1, cfg.train_eps, np.sum(avg_halt_num_list) / len(avg_halt_num_list)))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))
        rewards.append(np.sum(step_reward_list) / len(step_reward_list))
        speed_reward.append(np.sum(step_speed_reward_list) / len(step_speed_reward_list))
        tls_reward.append(np.sum(step_tl_reward_list) / len(step_tl_reward_list))
        target_reward.append(np.sum(step_target_reward_list) / len(step_target_reward_list))
        safe_reward.append(np.sum(step_safe_reward_list) / len(step_safe_reward_list))
        speeds.append(np.mean(np_speed_list))
        ecs.append(hundred_ec)
        jerks.append(np.mean(np_jerk_list))
        
    now = datetime.datetime.now()
    dest_path = cfg.model_dest_path + cfg.model_dest_path_leader + now.strftime("%Y%m%d_%H%M%S_") + ("gamma=%f_tau=%f.pth" % (gamma_arr[gamma_idx], tau_arr[tau_idx])) + cfg.model_dest_path_follow

    actor_pth_file_path = os.path.join(curr_path, dest_path)  ##########这个无需注释
  
    torch.save(agent.actor.state_dict(), actor_pth_file_path)

    outputFile.write('model saved successfully!')
    outputFile.write('Complete training!')
    outputFile.write("final reward list:{}".format(rewards))
    outputFile.write("final halt list:{}".format(episode_avg_halt_list))
    outputFile.write('speed reward list:{}'.format(speed_reward))
    outputFile.write('tls reward list:{}'.format(tls_reward))
    outputFile.write('target reward list:{}'.format(target_reward))
    outputFile.write('safe reward list {}'.format(safe_reward))
    outputFile.write('speed list {}'.format(speeds))
    outputFile.write('ec list {}'.format(ecs))
    outputFile.write('jerk list {}'.format(jerks))
    outputFile.close()

if __name__ == "__main__":
    maxThread = 12

    procs = []
    checkPoolQueue = Queue()
    checkPoolWait = Queue()
    now = datetime.datetime.now()
    destDirPth = './doubleOutput/' + now.strftime("%Y%m%d_%H%M%S/")
    os.makedirs(destDirPth)
    for i in range(5):
        for j in range(5):
            procs.append(multiprocessing.Process(target=train, args=(i, j, (i * 5 + j), destDirPth, )))
    
    for i in range(len(procs)):
        checkPoolWait.put(i)

    for i in range(max(maxThread, checkPoolQueue.qsize())):
        procID = checkPoolWait.get()
        procs[procID].start()
        checkPoolQueue.put(procID)

    procID = -1
    while not checkPoolQueue.empty():
        procID = checkPoolQueue.get()
        if procs[procID].is_alive():
            checkPoolQueue.put(procID)
            sleep(10)
            for i in range(checkPoolWait.qsize()):
                print("No.%d"%(checkPoolWait.queue[i]), end=' ')
            if checkPoolWait.qsize() > 0:
                print("processings are waiting")

            for i in range(checkPoolQueue.qsize()):
                print("No.%d"%(checkPoolQueue.queue[i]), end=' ')
            if checkPoolQueue.qsize() > 0:
                print("processings are running")
        else:
            if not checkPoolWait.empty():
                tmp = checkPoolWait.get()
                procs[tmp].start()
                checkPoolQueue.put(tmp)
            print("process No.%d well down" % procID)
