from sumocfg import generate_cfg_file
from sumocfg import generate_rou_file_single
import random

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

        self.train_eps = 200 # 原来是80
        self.eval_eps = 10 # 原来是10
        
        self.epsilon_start = 3

        self.max_speed = 28

        self.target_update = 4
        self.hidden_dim = 256
        self.soft_tau = tau
        self.max_action = 2


        self.simulation_steps = 3600
        
        self.pathToSumoFiles = "rou_net2_single"

        self.model_dest_path = "models/"
        self.model_dest_path_leader = "double_acce8_speed2_"
        self.model_dest_path_follow = ""

        self.logName = '_acce2_speed_8_gamma%5f_tua%5f.txt'%(self.gamma, self.soft_tau)
        self.procID = -1 # 多进程id

cfg = DDPGConfig('0', gamma=0.999, tau=0.1)

for i_episode in range(400):
    car_count_list = [100, 150, 200, 250, 50, 300]
    generate_rou_file_single(ep = i_episode + 1, car_count_per_lane=random.choice(car_count_list), path=cfg.pathToSumoFiles, simulation_steps=cfg.simulation_steps)    #######第二次是不需要更新的
    generate_cfg_file(ep = i_episode + 1, path=cfg.pathToSumoFiles)    #######
    