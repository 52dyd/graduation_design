import random
import numpy as np
import os
import traci
import math
from collections import defaultdict

def get_avg_speed(entering_lanes):
    #获取每一个车道上车辆的平均速度，然后除以车道数，最后得到整个交叉路口的车辆平均速度
    avg_speed = 0.0
    for lane in entering_lanes:
        avg_speed += traci.lane.getLastStepMeanSpeed(lane)
    return avg_speed / len(entering_lanes)

def get_avg_elec_cons(entering_lanes):
    #获取每一个车道上车辆的平均速度，然后除以车道数，最后得到整个交叉路口的车辆平均速度
    total_elec_cons = 0.0
    for lane in entering_lanes:
        total_elec_cons += traci.lane.getElectricityConsumption(lane)
    return total_elec_cons / len(entering_lanes)

def get_avg_halting_num(entering_lanes):
    total_halting_num = 0
    for lane in entering_lanes:
        total_halting_num += traci.lane.getLastStepHaltingNumber(lane)
    return total_halting_num / len(entering_lanes)


def get_leader_veh_info(veh_id, vehicle_id_entering, cfg):
    #leading_veh_flag:1表示当前车辆前方有车辆；0表示没有
    leading_veh_flag = 0
    leading_veh_speed = cfg.max_speed
    leading_veh_id = None
    dist = 249.0         #lane length
    leading_veh_info = traci.vehicle.getLeader(veh_id, 250)

    if leading_veh_info != None:
        leading_veh_id, dist = leading_veh_info
        if veh_id in vehicle_id_entering and leading_veh_id in vehicle_id_entering:
            leading_veh_flag = 1
            leading_veh_speed = traci.vehicle.getSpeed(leading_veh_id)
        if dist > 249.0:
            dist = 249.0
    return leading_veh_flag, leading_veh_speed,  dist + 1, leading_veh_id

def get_neighbor_cars(veh_id):
    max_communication_range = 250. # 车辆最多只能获得250m范围
    curr_lane = traci.vehicle.getLaneID(veh_id)
    curr_pos = traci.vehicle.getLanePosition(veh_id)
    
    follower = traci.vehicle.getFollower(veh_id, max_communication_range)
    leader = traci.vehicle.getLeader(veh_id, max_communication_range)
    
    leader_veh_info = []   # id, position, speed
    follower_veh_info = [] # id, position, speed
    
    #若后方没有车辆则车辆ID为'',而距离为-1
    if follower[0] != '':
        ##############################################################################
        follower_veh_id, dist_to_follower = follower[0], follower[1] #注意dist_to_follower可能不是后车前保险杠，到前车后保险杠的距离
        follower_veh_speed = traci.vehicle.getSpeed(follower_veh_id)
        ############################################################################
        follower_veh_info.append(follower_veh_id)
        follower_veh_info.append(curr_pos - dist_to_follower - 4.9 - 1.) # 4.9是车的长度， 1是min gap
        follower_veh_info.append(round(follower_veh_speed, 6))
    else:
        # 后方没有车辆，则假设当前车探测范围内最远处有一辆以最大速度行驶的车辆
        follower_veh_info.append('f')
        follower_veh_info.append(curr_pos - max_communication_range) #0表示该车还未到达当前车辆所在的十字路口
        follower_veh_info.append(20.)
    
    if leader:
        ###########################获取前车的相关信息############################
        leader_veh_id, dist_to_leader = leader[0], leader[1] 
        leader_speed = traci.vehicle.getSpeed(leader_veh_id)
        leader_pos = traci.vehicle.getLanePosition(leader_veh_id)
        #######################################################################
        if traci.vehicle.getLaneID(leader_veh_id) == traci.vehicle.getLaneID(veh_id):
            #################把前车的信息添加到列表中########################
            leader_veh_info.append(leader_veh_id)
            leader_veh_info.append(leader_pos)
            leader_veh_info.append(leader_speed)  
        else:
            leader_veh_info.append('l')
            leader_veh_info.append(curr_pos + max_communication_range)
            leader_veh_info.append(20.)
        ###############################################################
    else:
        # 前方没有车辆的情况, 则假设领先其250米处有一辆车以最大速度行驶
        leader_veh_info.append('l')
        leader_veh_info.append(curr_pos + max_communication_range)
        leader_veh_info.append(20.)
    return follower_veh_info, leader_veh_info

def get_green_phase_duration(veh_id, cfg):
    veh_lane = traci.vehicle.getLaneID(veh_id)
    #计算当前交通灯相位还剩下多少时间
    current_phase_duration = traci.trafficlight.getNextSwitch('J10') - traci.simulation.getTime()
    time_to_green_ns = 0.0
    time_to_green_ew = 0.0
    current_phase_name = traci.trafficlight.getPhaseName('J10')
    if current_phase_name == 'wer_nsg':
        time_to_green_ns = current_phase_duration
        time_to_green_ew = current_phase_duration + cfg.yellow_duration
    elif current_phase_name == 'weg_nsr':
        time_to_green_ew = current_phase_duration
        time_to_green_ns = current_phase_duration + cfg.yellow_duration
    elif current_phase_name == 'wey_nsr':
        time_to_green_ew = current_phase_duration + cfg.green_phase_duration
        time_to_green_ns = current_phase_duration
    else:
        time_to_green_ns = current_phase_duration + cfg.green_phase_duration
        time_to_green_ew = current_phase_duration
    if veh_lane == cfg.entering_lanes[0] or veh_lane == cfg.entering_lanes[1]:
        #当前车辆在东西向的车道上,先判断此时的交通相位
        if current_phase_name == 'weg_nsr':
            return 1, time_to_green_ew
        else:
            return 0, time_to_green_ew
    else:
        #当前车辆在南北向的车道上
        if current_phase_name == 'wer_nsg':
            return 1, time_to_green_ns
        else:
            return 0, time_to_green_ns

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
            
        if veh_lane == 'WE_0' or veh_lane == 'EW_0':
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
            
#得到当前二车的最小安全距离
def get_safe_dist(current_speed, leader_speed, max_deceleration = 3, 
                        min_gap = 1, time_react = 0.5, time_delay = 0.3, time_i = 0.1):
    '''
    time_react:反应时间，因为是AV，所以在这里设置0.5
    time_delay:制动协调时间
    time_i:减速增长时间
    a_max:汽车减速过程中的最大加速度
    '''  
    safe_dist = current_speed * (time_react + time_delay + time_i / 2) + \
        ((current_speed - leader_speed)**2) / 2 * max_deceleration + min_gap
    return safe_dist

def IDM(veh_id, leader_veh_info):
    target_speed = 13.8
    max_accel = 2.0
    max_decel = 2.0
    min_gap = 1.0
    safe_time_headway = 1.5
    speed_difference = traci.vehicle.getSpeed(veh_id) - leader_veh_info[1]
    s = leader_veh_info[2] + min_gap

    if leader_veh_info[0] == 0:
        accel_value = max_accel * (1 - math.pow(traci.vehicle.getSpeed(veh_id) / target_speed, 2))
    else:
        accel_value =  max_accel * (1 - math.pow(traci.vehicle.getSpeed(veh_id) / target_speed, 4) - \
            math.pow((min_gap + traci.vehicle.getSpeed(veh_id) * safe_time_headway + \
            (traci.vehicle.getSpeed(veh_id) * speed_difference / 2 * math.sqrt(max_accel * max_decel))) / s, 2))

    if accel_value > 2:
        accel_value = 2
    if accel_value < -2:
        accel_value = -2
    desired_speed = traci.vehicle.getSpeed(veh_id) + accel_value
    if desired_speed < 0:
        desired_speed = 0.0
    elif desired_speed > 13.8:
        desired_speed = 13.8
    return desired_speed


def glosa(veh_id, alpha=0.9):
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
        

        traci.vehicle.setSpeed(veh_id, target_speed)
    
def get_tls_info2(veh_id):
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
        tls_phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
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
        
        curr_speed = traci.vehicle.getSpeed(veh_id)    
        if veh_lane == 'WE_0' or veh_lane == 'EW_0':
            # 当前车辆在东西向的车道上,先判断此时的交通相位
            if tls_phase_name == 'weg_nsr':
                if dist_to_intersec / 20. > time_to_green_we:
                    return [1, 
                            time_to_green_we + tls_phase_dict['wey_nsr'] + tls_phase_dict['wer_nsg'] + tls_phase_dict['wer_nsy'],
                            time_to_green_we + tls_phase_dict['wey_nsr'] + tls_phase_dict['wer_nsg'] + tls_phase_dict['wer_nsy'] + tls_phase_dict['weg_nsr'],
                            dist_to_intersec]
                else:
                    return [1, 0.01, time_to_green_we + 0.01, dist_to_intersec]
            else:
                if time_to_green_we == 0:
                    [0, 0.01, time_to_green_we + tls_phase_dict['weg_nsr'], dist_to_intersec]
                else:
                    return [0, time_to_green_we, time_to_green_we + tls_phase_dict['weg_nsr'], dist_to_intersec]
        else:
            #  当前车辆在南北向的车道上
            if tls_phase_name == 'wer_nsg':
                # 无法在绿灯剩余期间内通过, 则以下一个绿灯周期作为通过的目标
                if dist_to_intersec / 20. > time_to_green_ns:
                    return [1, 
                            time_to_green_ns + tls_phase_dict['wer_nsy'] + tls_phase_dict['weg_nsr'] + tls_phase_dict['wey_nsr'],
                            time_to_green_ns + tls_phase_dict['wer_nsy'] + tls_phase_dict['weg_nsr'] + tls_phase_dict['wey_nsr'] + tls_phase_dict['wer_nsg'],
                            dist_to_intersec]
                else:
                    # 能在绿灯剩余期间内通过，则最快的通过时间也是下一秒
                    return [1, 0.01, time_to_green_ns+0.01, dist_to_intersec]
            else:
                if time_to_green_ns == 0:
                    return [0, 0.01, time_to_green_ns + tls_phase_dict['wer_nsg'] , dist_to_intersec] 
                else:
                    return [0, time_to_green_ns, time_to_green_ns + tls_phase_dict['wer_nsg'] , dist_to_intersec] 