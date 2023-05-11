import os
import traci
import torch
import random
import numpy as np

from traffic import *
from sumocfg import set_sumo
from sumocfg import generate_test_cfg_file
from sumocfg import generateCfgTestFileDouble
from sumocfg import generate_normal_rou_file
from sumocfg import generateRouTestFileDoueble
import json
from functools import partial
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from agent import DDPGAgent

def initRandSeed(seed_value):
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

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.simulation_steps = 2000
        self.modelPth = 'models/double20230507_061154_gamma=0.999000_tau=0.010000.pth'
        self.sumocfgPth = 'test1/test.sumocfg'

def test():
    cfg = DDPGConfig()
    initRandSeed(42)
    currPath = os.path.dirname(os.path.abspath(__file__))
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    print(currPath)
    targetPath = os.path.join(currPath, cfg.modelPth)
    print('-------\n', targetPath, '-------\n')

    agent = DDPGAgent(state_dim=10, action_dim=1, cfg=cfg)
    agent.actor.load_state_dict(torch.load(targetPath, map_location=torch.device("cpu")))
    print('模型参数导入成功')

    # episode_avg_halt_list = []

    for i_episode in range(cfg.eval_eps):
        generateRouTestFileDoueble(ep = i_episode + 1)
        sumocfgPth = generateCfgTestFileDouble(ep = i_episode + 1)    #######
        sumocfgPth = os.path.join(currPath, sumocfgPth)
        sumo_cmd = set_sumo(gui=True, sumocfg_file_name=sumocfgPth, max_steps=cfg.simulation_steps)

        agent.reset()
        traci.start(sumo_cmd)

        # avg_halt_num_list = []
        # avg_cl_dist_list = []
    
        i_step = 0

        # fstInfo = defaultdict(partial(defaultdict, list))
        # secInfo = defaultdict(list)

        while traci.simulation.getMinExpectedNumber() > 0 and i_step <= 2 * cfg.simulation_steps:
            i_step += 1
            if traci.vehicle.getIDCount() > 0:
                currState = agent.getCurrentStateNCC()
                action = agent.choose_action(currState, i_step, add_noise=True)
                action = agent.step(currState, action)

                # step_jerk = 0.
                # for veh in traci.vehicle.getIDList():
                #     if len(fstInfo[veh]['accel'] != 0):
                #         step_jerk += abs(traci.vehicle.getAcceleration(veh) - fstInfo[veh]['accel'][-1])
                #     fstInfo[veh]['speed'].append(traci.vehicle.getSpeed(veh))
                #     fstInfo[veh]['accel'].append(traci.vehicle.getAcceleration(veh))
                #     fstInfo[veh]['dist'].append(traci.vehicle.getDistance(veh))
                #     fstInfo[veh]['pos'].append(traci.vehicle.getLanePosition(veh))
                #     fstInfo[veh]['ec'].append(get_electricity_cons_per_car(veh, speed_thresh=2, idle=0.5))
                
                # if len(traci.vehicle.getIDList()) == 0:
                #     step_jerk = 0.
                # else:
                #     step_jerk /= len(traci.vehicle.getIDList())
            else:
                traci.simulationStep()
        traci.close()

    print('test finish !')

if __name__ == "__main__":
    test()
