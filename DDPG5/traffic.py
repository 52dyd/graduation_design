import numpy as np
import traci
import math
from collections import defaultdict
import json 

def save_json_to_dict(info_dict, target_path):
    json_str = json.dumps(info_dict)
    with open(target_path, 'w') as json_file:
        json_file.write(json_str)
        
def get_step_mean_speed():
    #获取每一个车道上车辆的平均速度，然后除以车道数，最后得到整个交叉路口的车辆平均速度
    veh_list = traci.vehicle.getIDList()
    
    speed_sum = 0.
    for veh in veh_list:
        speed_sum += traci.vehicle.getSpeed(veh)
    
    if len(veh_list) == 0:
        return 0.
    return speed_sum / len(veh_list)

def get_step_total_ec(speed_thresh = 2, idle=0.5):
    veh_list = traci.vehicle.getIDList()
    
    ec_sum = 0.
    for veh in veh_list:
        ec_sum += get_electricity_cons_per_car(veh, speed_thresh, idle)
    
    return ec_sum      

def get_electricity_cons_per_car(veh, speed_thresh=2, idle=0.5):
    ec = 0.
    if traci.vehicle.getSpeed(veh) < speed_thresh:
        ec += traci.vehicle.getElectricityConsumption(veh) + idle
    else:
        ec += traci.vehicle.getElectricityConsumption(veh) + 0.6
    return ec 

def calc_force(v, a):
    '''计算车辆的牵引力 or 制动力'''
    rou = 1.2256
    af = 2.5
    cd = 0.3
    theta = 0
    m = 1800
    g = 9.8
    cr = 0.02
    
    '''1. 空气阻力'''
    f_air = 0.5 * rou * af * cd * (v**2)
    # print('空气阻力为 {} N'.format(f_air))
    '''2. 滚动阻力'''
    f_roll = m * g * cr * math.cos(theta)
    # print('滚动阻力{}'.format(f_roll))
    '''3. 牵引力'''
    f = m * a + f_air + f_roll
    return f

def get_output_power(f, v, eta_m, eta_g, idling=400):
    '''idling 代表空转功率是400 瓦'''
    if f > 0:
        return f * v / eta_m + idling 
    else:
        return f * v * eta_g + idling

def get_neighbor_cars(veh_id):
    min_gap = 1.5
    max_communication_range = 250. # 车辆最大通信范围是250m
    curr_pos = traci.vehicle.getLanePosition(veh_id)
    
    follower = traci.vehicle.getFollower(veh_id, max_communication_range)
    leader = traci.vehicle.getLeader(veh_id, max_communication_range)
    
    leader_veh_info = []   # id, position, speed
    follower_veh_info = [] # id, position, speed
    
    #若后方没有车辆则车辆ID为'',而距离为-1
    if follower[0] != '':
        ############################################################
        follower_veh_id, dist_to_follower = follower[0], follower[1] 
        follower_veh_speed = traci.vehicle.getSpeed(follower_veh_id)
        ###################把后车的信息添加到列表中##################
        follower_veh_info.append(follower_veh_id)
        follower_veh_info.append(curr_pos - dist_to_follower - 4.9 - min_gap)
        follower_veh_info.append(follower_veh_speed)
    else:
        # 后方没有车辆
        follower_veh_info.append('f')
        follower_veh_info.append(curr_pos - max_communication_range) 
        follower_veh_info.append(traci.vehicle.getSpeed(veh_id))
    
    if leader:
        ###########################获取前车的相关信息############################
        leader_veh_id, dist_to_leader = leader[0], leader[1] 
        leader_speed = traci.vehicle.getSpeed(leader_veh_id)
        #################把前车的信息添加到列表中########################
        leader_veh_info.append(leader_veh_id)
        leader_veh_info.append(curr_pos + dist_to_leader + 4.9 + min_gap)
        leader_veh_info.append(leader_speed)  
        ###############################################################
    else:
        # 前方没有车辆
        leader_veh_info.append('l')
        leader_veh_info.append(curr_pos + max_communication_range)
        leader_veh_info.append(traci.vehicle.getSpeed(veh_id))
    return follower_veh_info, leader_veh_info

def calc_ttc(v1, v2, a1, a2, dist, minGap = 1.5):
    '''
    v1: 当前车的速度
    v2: 其前车的速度
    a1, a2类似
    '''
    if v1 > v2:
        if a1 == a2:
            ttc = (dist - minGap) / (v1 - v2 + 1e-6)
        else:
            v_rel = v1 - v2
            a_rel = a1 - a2 
            if v_rel ** 2 + 2 * a_rel * (dist - minGap) >= 0:
                ttc = (-v_rel + (v_rel ** 2 + 2 * a_rel * (dist - minGap))   ** 0.5) / a_rel
            else:
                ttc = 1e6
    else:
        ttc = 1e6 # 前车速度大于等于当前车，那么这时候的驾驶是相对安全的
        
    return ttc
    
def get_tls_info(veh_id):
    tls_info = traci.vehicle.getNextTLS(veh_id)
    # print('-------------------------------------------------------')
    # print('veh {}, lane {}'.format(veh_id, traci.vehicle.getLaneID(veh_id)))
    veh_lane = traci.vehicle.getLaneID(veh_id)
    if len(tls_info) != 0:
        tls_id = tls_info[0][0]  # 当前车辆即将到达的路口ID
        # print('路口id {}'.format(tls_id))
        dist_to_intersec = tls_info[0][2] # 车辆距离下个十字路口的距离
        
        tls_phase_name = traci.trafficlight.getPhaseName(tls_id)
        current_phase_duration = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
        # tls_phases存储四个元组 Phase(duration=4.0, state='ryry', minDur=4.0, maxDur=4.0, next=(), name='wey_nsr')
        tls_phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases
        tls_phase_dict = defaultdict()
        for i in range(len(tls_phases)):
            tls_phase_dict[tls_phases[i].name]=tls_phases[i].duration

        # 只有当交通灯为绿灯时，tls_flag才为1，其他情况下tls_flag都是0
        if tls_phase_name == "wer_nsg": #东西向红灯，南北向绿灯
            time_to_green_ns = current_phase_duration
            time_to_green_we = current_phase_duration + tls_phase_dict['wer_nsy']
        elif tls_phase_name == "wer_nsy": #东西向红灯，南北向黄灯
            time_to_green_ns = current_phase_duration + tls_phase_dict['weg_nsr'] + tls_phase_dict['wey_nsr']
            time_to_green_we = current_phase_duration
        elif tls_phase_name == "weg_nsr": #东西向绿灯，南北向红灯
            time_to_green_ns = current_phase_duration + tls_phase_dict['wey_nsr']
            time_to_green_we = current_phase_duration
        else: #东西向黄灯，南北向红灯
            time_to_green_ns = current_phase_duration
            time_to_green_we = current_phase_duration + tls_phase_dict['wer_nsg'] + tls_phase_dict['wer_nsy']
            
        if veh_id[0:3] == 'W2E' or veh_id[0:3] == 'E2W':
            # 当前车辆在东西向的车道上,先判断此时的交通相位
            if tls_phase_name == 'weg_nsr':
                return 1, time_to_green_we, dist_to_intersec
            else:
                return 0, time_to_green_we, dist_to_intersec
        else:
            #  当前车辆在南北向的车道上
            if tls_phase_name == 'wer_nsg':
                return 1, time_to_green_ns, dist_to_intersec
            else:
                return 0, time_to_green_ns, dist_to_intersec   

def glosa(veh_id, alpha=0.9):
    if veh_id == 'l' or veh_id == 'f':
        return None
    tls_info = get_tls_info2(veh_id)
    follower_veh_info, leader_veh_info = get_neighbor_cars(veh_id)
    follower_id = follower_veh_info[0]
    if tls_info != None:
        dist_to_tl = tls_info[3]
        min_time = tls_info[1]
        max_time = tls_info[2]
            
        # print('veh {}, dist to tl {}, min time {}, max time {}'.format(veh_id, dist_to_tl, min_time, max_time))
        min_speed = dist_to_tl / max_time 
        max_speed = dist_to_tl / min_time
        # print('min speed {}, max speed {}'.format(min_speed, max_speed))
        if min_speed > 20.:
            min_speed = 20.
        if max_speed > 20.:
            max_speed = 20.
        
        target_speed = max(alpha * max_speed, (2 - alpha) * min_speed)

        return target_speed
        
def get_glosa_speed(veh_id, alpha=0.9):
    if veh_id == 'l' or veh_id == 'f':
        return None
    tls_info = get_tls_info2(veh_id)
    if tls_info != None:
        tl_flag = tls_info[0]
        tls_phase_dict = tls_info[4]
        dist_to_tl = tls_info[3]
        min_time = tls_info[1]
        max_time = tls_info[2]
        
        # 根据前车的情况实际调整速度
        f, l = get_neighbor_cars(veh_id)
        follower_id, leader_id = f[0], l[0]
        follower_speed, leader_speed = f[2], l[2]
        # print('--------------------------------------')
        # if leader_id != 'l':
        # # 只有前车和当前车在同一个十字路口上时，前车的速度会影响到当前车的通行时间
        #     if traci.vehicle.getLaneID(veh_id) == traci.vehicle.getLaneID(leader_id):
        #         # 得到前车到达十字路口的预计时间
        #         tls_info_leader = get_tls_info(leader_id)
        #         leader_dist_tl = tls_info_leader[2]
        #         leader_tl_flag = tls_info_leader[0] 
        #         if leader_tl_flag == 1:
        #             if leader_speed > 0.5:
        #                 leader_pass_time = leader_dist_tl / leader_speed
        #                 # print('前车{}的预计通过时间{}'.format(leader_id, leader_pass_time))
        #                 if leader_pass_time > max_time:
        #                     # 当前车因为前车在当前绿灯周期通过，因此选择下一个绿灯周期通过
        #                     min_time = max_time + tls_phase_dict['wey_nsr'] + tls_phase_dict['wer_nsg'] + tls_phase_dict['wer_nsy']
        #                     max_time = max_time + tls_phase_dict['wey_nsr'] + tls_phase_dict['wer_nsg'] + tls_phase_dict['wer_nsy'] + tls_phase_dict['weg_nsr']
        #                 else:
        #                     if leader_pass_time > min_time:
        #                         min_time = leader_pass_time
        # if follower_id != 'f':     
        #     if traci.vehicle.getLaneID(veh_id) == traci.vehicle.getLaneID(follower_id):
        #         # 得到前车到达十字路口的预计时间
        #         tls_info_follower = get_tls_info(follower_id)
        #         follower_dist_tl = tls_info_follower[2]
        #         follower_tl_flag = tls_info_follower[0] 
        #         if follower_tl_flag == 1:
        #             if follower_speed > 0.5:
        #                 follower_pass_time = follower_dist_tl / follower_speed

        #                 if follower_pass_time < max_time:
        #                     if follower_pass_time > min_time:
        #                         max_time = follower_pass_time
        # print('--------------------------------------')                        
        # print('最早通过的时间 {}, 最迟通过时间 {}'.format(min_time, max_time))   
        # print('到十字路口的距离{}'.format(dist_to_tl))             
        min_speed = dist_to_tl / max_time 
        max_speed = dist_to_tl / min_time 
        min_speed = np.clip(min_speed, 0., 20.)
        max_speed = np.clip(max_speed, 0., 20.)    
        
        target_speed = max(alpha * max_speed, (2 - alpha) * min_speed)
        
    return target_speed, min_speed, max_speed

def get_IDM_speed(veh_id, glosa_flag=True, speed_flag=True):
    v1 = traci.vehicle.getSpeed(veh_id)
    a1 = traci.vehicle.getAcceleration(veh_id)
    f, l = get_neighbor_cars(veh_id)
    tl_info = get_tls_info2(veh_id)
    target_speed = 20.
    max_a = 3.                             # 当前车最大的加速度
    b = 3.                                 # 舒适减速度
    T = 1.                                 # time headway
    s0 = 1.                                # min gap
    if tl_info:
        if glosa_flag:
            target_speed, min_speed, max_speed = get_glosa_speed(veh_id, alpha=0.9)      # IDM中的target speed
        if l[0] == 'l':
            # 前方没有车辆的情况
            next_a = max_a * (1 - (v1 / (target_speed + 1e-6)) ** 4)
        else:
            # 前方有其他车辆的情况
            actual_gap = traci.vehicle.getDistance(l[0]) - traci.vehicle.getDistance(veh_id) - 4.9
            v2 = traci.vehicle.getSpeed(l[0])
            desired_gap = s0 + max(0, v1 * T + v1 * (v1 - v2) / (2 * (max_a * b) ** 0.5))
            next_a = max_a * (1 - (v1 / (target_speed + 1e-6)) ** 4 - (desired_gap / actual_gap) ** 2)
    else:
        if l[0] == 'l':
            # 前方没有车辆的情况
            next_a = max_a * (1 - (v1 / (target_speed + 1e-6)) ** 4)
        else:
            # 前方有其他车辆的情况
            actual_gap = traci.vehicle.getDistance(l[0]) - traci.vehicle.getDistance(veh_id) - 4.9
            v2 = traci.vehicle.getSpeed(l[0])
            desired_gap = s0 + max(0, v1 * T + v1 * (v1 - v2) / (2 * (max_a * b) ** 0.5))
            next_a = max_a * (1 - (v1 / (target_speed + 1e-6)) ** 4 - (desired_gap / actual_gap) ** 2)
    # print(next_a)        
    next_a = np.clip(next_a, -2., 1.67)
    next_speed = v1 + next_a
    next_speed = np.clip(next_speed, 0., 20.)
    if speed_flag:
        traci.vehicle.setSpeed(veh_id, next_speed)
    return next_a

def get_tls_info2(veh_id):
    tls_info = traci.vehicle.getNextTLS(veh_id)
    if len(tls_info) != 0:
        # print('123')
        tls_id = tls_info[0][0]  # 当前车辆即将到达的路口ID
        # print('路口id {}'.format(tls_id))
        dist_to_intersec = tls_info[0][2] # 车辆距离下个十字路口的距离
        
        tls_phase_name = traci.trafficlight.getPhaseName(tls_id)
        current_phase_duration = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
        # print('当前的相位名 {}, 当前相位的剩余时间:{}'.format(tls_phase_name, current_phase_duration))
        # tls_phases存储四个元组 Phase(duration=4.0, state='ryry', minDur=4.0, maxDur=4.0, next=(), name='wey_nsr')
        tls_phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
        tls_phase_dict = defaultdict()
        all_phase_duration = 0.
        for i in range(len(tls_phases)):
            tls_phase_dict[tls_phases[i].name] = tls_phases[i].duration
            all_phase_duration += tls_phases[i].duration
        # print(tls_phase_dict)

        # 只有当交通灯为绿灯时，tls_flag才为1，其他情况下tls_flag都是0
        if tls_phase_name == "wer_nsg": #东西向红灯，南北向绿灯
            time_to_green_we = current_phase_duration + tls_phase_dict['wer_nsy']
        elif tls_phase_name == "wer_nsy": #东西向红灯，南北向黄灯
            time_to_green_we = current_phase_duration
        elif tls_phase_name == "weg_nsr": #东西向绿灯，南北向红灯
            time_to_green_we = current_phase_duration
        else: #东西向黄灯，南北向红灯
            time_to_green_we = current_phase_duration + tls_phase_dict['wer_nsg'] + tls_phase_dict['wer_nsy']
        
        if veh_id[0:3] == 'W2E' or veh_id[0:3] == 'E2W':
            # 当前车辆在东西向的车道上,先判断此时的交通相位
            if tls_phase_name == 'weg_nsr':
                # 如果无法在绿灯期间通过，则选择下一个绿灯通过
                if dist_to_intersec / 20. <= 0.9 * time_to_green_we:
                    return [1, 0.01, time_to_green_we + 0.01, dist_to_intersec, tls_phase_dict]
                else:
                    return [1, 
                            time_to_green_we + tls_phase_dict['wey_nsr'] + tls_phase_dict['wer_nsg'] + tls_phase_dict['wer_nsy'],
                            time_to_green_we + tls_phase_dict['wey_nsr'] + tls_phase_dict['wer_nsg'] + tls_phase_dict['wer_nsy'] + tls_phase_dict['weg_nsr'],
                            dist_to_intersec,
                            tls_phase_dict]
            else:
                if time_to_green_we == 0:
                    return [0, 0.01, time_to_green_we + tls_phase_dict['weg_nsr'], dist_to_intersec, tls_phase_dict]
                else:
                    return [0, time_to_green_we, time_to_green_we + tls_phase_dict['weg_nsr'], dist_to_intersec, tls_phase_dict]
        
def get_safe_speed(veh):
    # 保证车与车之间的安全交互
    f, l = get_neighbor_cars(veh)
    curr_speed = traci.vehicle.getSpeed(veh)
    
    if l[0] != 'l':
        dist_to_leader = traci.vehicle.getDistance(l[0]) - traci.vehicle.getDistance(veh) - 4.9
        leader_speed = traci.vehicle.getSpeed(l[0])
        safe_speed = leader_speed + (dist_to_leader - leader_speed * 1) / ((curr_speed + leader_speed) / (2 * 3) + 1.)
        print(safe_speed)
        return safe_speed
    