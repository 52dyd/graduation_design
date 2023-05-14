import os
import traci
import torch
import random
import numpy as np

from traffic import *
from sumocfg import set_sumo
from sumocfg import generate_test_cfg_file
from sumocfg import generateCfgTestFileSingle
from sumocfg import generate_normal_rou_file
from sumocfg import generateRouTestFileSingle
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
        self.modelPth = 'models/single20230507_061421_gamma=0.999000_tau=0.050000.pth'
        self.sumocfgPth = 'test1/test.sumocfg'

def test():
    cfg = DDPGConfig()
    # initRandSeed(42)
    currPath = os.path.dirname(os.path.abspath(__file__))
    # print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    print(currPath)
    targetPath = os.path.join(currPath, cfg.modelPth)
    print('-------\n', targetPath, '-------\n')
    
    # agent = DDPGAgent(state_dim=10, action_dim=1, cfg=cfg)
    # agent.actor.load_state_dict(torch.load(targetPath, map_location=torch.device("cpu")))
    # print('模型参数导入成功')

    # episode_avg_halt_list = []

    for i_episode in range(cfg.eval_eps):
        generateRouTestFileSingle(ep = i_episode + 1, car_count_per_lane = 500)
        sumocfgPth = generateCfgTestFileSingle(ep = i_episode + 1)    #######
        sumocfgPth = os.path.join(currPath, sumocfgPth)
        sumo_cmd = set_sumo(gui=True, sumocfg_file_name=sumocfgPth, max_steps=cfg.simulation_steps)

        # agent.reset()
        traci.start(sumo_cmd)
        
        # avg_halt_num_list = []
        # avg_cl_dist_list = []
    
        i_step = 0

        # fstInfo = defaultdict(partial(defaultdict, list))
        # secInfo = defaultdict(list)

        while traci.simulation.getMinExpectedNumber() > 0 and i_step <= 2 * cfg.simulation_steps:
            i_step += 1
            if traci.vehicle.getIDCount() > 0:

# 0当前速度 1前车速 2后车速 3前车距 4后车距 5TLS距离 6最大时间 7最小时间 8最小速 9最大速
# 前后没有车时，车距为最远通讯距离250-车长4.9=245.1，车速就是自己
                currState = get_current_state()
                for vehID in traci.vehicle.getIDList(): # 前方没有路口
                    if not vehID in currState:
                        traci.vehicle.setSpeed(vehID, cfg.max_speed) # 前方没有路口
                    elif len(currState[vehID]) == 0:
                        traci.vehicle.setSpeed(vehID, cfg.max_speed) # 前方有路口
                    else:
                        traci.vehicle.setSpeed(vehID, currState[vehID][9])
            traci.simulationStep()

if __name__ == "__main__":
    test()