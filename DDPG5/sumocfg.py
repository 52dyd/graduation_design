import math
import random
import os
import numpy as np
import sys
from sumolib import checkBinary

def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
 
    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", sumocfg_file_name, "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd

def generate_cfg_file(ep, path='test_rou_net'):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file_name = path+'/intersection' + str(ep) + '.sumocfg'
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    net_file_name = path+'/intersection1.net.xml'
    cfg_file = os.path.join(curr_path, cfg_file_name)
    rou_file = os.path.join(curr_path, rou_file_name)
    net_file = os.path.join(curr_path, net_file_name)

    with open(cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<configuration>
    <input>
        <net-file value="{}"/>
        <route-files value="{}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <tripinfo-output value="tripinfo.xml"/>
</configuration>""".format(net_file, rou_file), file=route)
        
def generate_test_cfg_file(ep, path='test_rou_net'):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file_name = path+'/intersection' + str(ep) + '.sumocfg'
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    net_file_name = path+'/intersection.net.xml'
    cfg_file = os.path.join(curr_path, cfg_file_name)
    rou_file = os.path.join(curr_path, rou_file_name)
    net_file = os.path.join(curr_path, net_file_name)

    with open(cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<configuration>
    <input>
        <net-file value="{}"/>
        <route-files value="{}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <tripinfo-output value="tripinfo.xml"/>
</configuration>""".format(net_file, rou_file), file=route)

def generate_rou_file(ep, simulation_steps = 3600, car_count_per_lane = 100, path='rou_net'):
    #在4000s内随机生成400辆车
    random.seed(42)  #设置随机数种子，能够让结果重现
    timings_we = np.random.weibull(2, car_count_per_lane)
    timings_ew = np.random.weibull(2, car_count_per_lane)
    timings_we = np.sort(timings_we)
    timings_ew = np.sort(timings_ew)
    

    #reshape the distribution to fit the interval 0:max_steps
    car_gen_steps_we = []
    car_gen_steps_ew = []
    min_old_we = math.floor(timings_we[1])
    max_old_we = math.ceil(timings_we[-1])
    min_old_ew = math.floor(timings_ew[1])
    max_old_ew = math.ceil(timings_ew[-1])
    min_new = 0
    max_new = simulation_steps
    for i in range(len(timings_we)):
        car_gen_steps_we.append(((max_new - min_new) / (max_old_we - min_old_we)) * (timings_we[i] - max_old_we) + max_new)
        car_gen_steps_ew.append(((max_new - min_new) / (max_old_ew - min_old_ew)) * (timings_ew[i] - max_old_ew) + max_new)
 
    car_gen_steps_we = np.rint(car_gen_steps_we) 
    car_gen_steps_ew = np.rint(car_gen_steps_ew) 

    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy/unknown" tau="1" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1" maxSpeed="20" guiShape="passenger">
            <param key="has.battery.device" value="true"/>
            <param key="maximumBatteryCapacity" value="38000"/> 
            <param key="maximumPower" value="200000"/>
            <param key="vehicleMass" value="1830"/>
            <param key="frontSurfaceArea" value="2.6"/>
            <param key="airDragCoefficient" value="0.35"/>
            <param key="internalMomentOfInertia" value="0.01"/>
            <param key="radialDragCoefficient" value="0.1"/>
            <param key="rollDragCoefficient" value="0.01"/>
            <param key="constantPowerIntake" value="100"/>
            <param key="propulsionEfficiency" value="0.9"/>
            <param key="recuperationEfficiency" value="0.9"/>
            <param key="stoppingThreshold" value="0.1"/>
        </vType>
        
        <route id="W2E" edges="WE0 WE1 WE2"/>
        <route id="E2W" edges="EW0 EW1 EW2"/>""", file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型
            depart_list.append(['W2E', car_gen_steps_we[i]])
            depart_list.append(['E2W', car_gen_steps_ew[i]])
        depart_list = sorted(depart_list, key = lambda x:x[1])
        for i in range(car_count_per_lane * 2):
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]), file = route)             
        print('</routes>', file=route)
            
def generate_test_rou_file(ep, simulation_steps = 3600, car_count_per_lane = 100, path='test_rou_net'):
    #在4000s内随机生成400辆车
    random.seed(42)  #设置随机数种子，能够让结果重现
    timings_we = np.random.weibull(2, car_count_per_lane)
    timings_ew = np.random.weibull(2, car_count_per_lane)
    timings_we = np.sort(timings_we)
    timings_ew = np.sort(timings_ew)
    

    #reshape the distribution to fit the interval 0:max_steps
    car_gen_steps_we = []
    car_gen_steps_ew = []
    min_old_we = math.floor(timings_we[1])
    max_old_we = math.ceil(timings_we[-1])
    min_old_ew = math.floor(timings_ew[1])
    max_old_ew = math.ceil(timings_ew[-1])
    min_new = 0
    max_new = simulation_steps
    for i in range(len(timings_we)):
        car_gen_steps_we.append(((max_new - min_new) / (max_old_we - min_old_we)) * (timings_we[i] - max_old_we) + max_new)
        car_gen_steps_ew.append(((max_new - min_new) / (max_old_ew - min_old_ew)) * (timings_ew[i] - max_old_ew) + max_new)

    car_gen_steps_we = np.rint(car_gen_steps_we) 
    car_gen_steps_ew = np.rint(car_gen_steps_ew) 

    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy/unknown" tau="1.5" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1.0" maxSpeed="20" carFollowing="IDM" guiShape="passenger">
            <param key="has.battery.device" value="true"/>
            <param key="maximumBatteryCapacity" value="38000"/> 
            <param key="maximumPower" value="200000"/>
            <param key="vehicleMass" value="1830"/>
            <param key="frontSurfaceArea" value="2.6"/>
            <param key="airDragCoefficient" value="0.35"/>
            <param key="internalMomentOfInertia" value="0.01"/>
            <param key="radialDragCoefficient" value="0.1"/>
            <param key="rollDragCoefficient" value="0.01"/>
            <param key="constantPowerIntake" value="100"/>
            <param key="propulsionEfficiency" value="0.9"/>
            <param key="recuperationEfficiency" value="0.9"/>
            <param key="stoppingThreshold" value="0.1"/>
        </vType>
        
        <route id="W2E" edges="WE1 WE2 WE3"/>
        <route id="E2W" edges="EW1 EW2 EW3"/>""", file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型
            depart_list.append(['W2E', car_gen_steps_we[i]])
            depart_list.append(['E2W', car_gen_steps_ew[i]])
        depart_list = sorted(depart_list, key = lambda x:x[1])
        for i in range(car_count_per_lane * 2):
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best" insertionChecks="none"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]), file = route)             
        print('</routes>', file=route)

def generate_uniform_rou_file(ep, simulation_steps = 3600, arr_time=15, depart_speed=10, car_count_per_lane = 100, path='uniform_rou_net', carfollowingModel='IDM'):
    depart_margin = simulation_steps // car_count_per_lane
    car_gen_steps_we = [depart_margin * i for i in range(car_count_per_lane)]
    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
   
    with open(rou_cfg_file, "w") as route:
        print("""<routes>
        
        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy/unknown" tau="1" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1.0" maxSpeed="20" carFollowing="%s" guiShape="passenger">
            <param key="has.battery.device" value="true"/>
            <param key="maximumBatteryCapacity" value="38000"/> 
            <param key="maximumPower" value="200000"/>
            <param key="vehicleMass" value="1830"/>
            <param key="frontSurfaceArea" value="2.6"/>
            <param key="airDragCoefficient" value="0.35"/>
            <param key="internalMomentOfInertia" value="0.01"/>
            <param key="radialDragCoefficient" value="0.1"/>
            <param key="rollDragCoefficient" value="0.01"/>
            <param key="constantPowerIntake" value="100"/>
            <param key="propulsionEfficiency" value="0.9"/>
            <param key="recuperationEfficiency" value="0.9"/>
            <param key="stoppingThreshold" value="0.1"/>
        </vType>

        <route id="W2E" edges="WE0 WE1"/>""" % (carfollowingModel), file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型          
            depart_list.append(['W2E', car_gen_steps_we[i]]) # insertionChecks="none"
        for i in range(car_count_per_lane):
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="%i" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]+arr_time, depart_speed), file = route)             
        print('</routes>', file=route)

def generate_normal_rou_file(ep, simulation_steps = 3600, arr_time=15, depart_speed=10, car_count_per_lane = 400, path='normal_rou_net', carfollowingModel='IDM'):
    mean = 0
    std = 10
    accel_list = [np.random.normal(mean, std) for i in range(car_count_per_lane)]
    max_accel = max(accel_list)
    min_accel = min(accel_list)

    accel_list = [(accel_list[i] - min_accel) / (max_accel - min_accel) for i in range(len(accel_list))]


    depart_list = [int(accel_list[i] * simulation_steps) for i in range(len(accel_list))]
    car_gen_steps_we = sorted(depart_list)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
   
    with open(rou_cfg_file, "w") as route:
        # recuperationEfficiency 电动汽车再生效率
        print("""<routes>
        
        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy/unknown" tau="1" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1.0" maxSpeed="20" carFollowing="%s" guiShape="passenger">
            <param key="has.battery.device" value="true"/>
            <param key="maximumBatteryCapacity" value="38000"/> 
            <param key="maximumPower" value="200000"/>
            <param key="vehicleMass" value="1830"/>
            <param key="frontSurfaceArea" value="2.6"/>
            <param key="airDragCoefficient" value="0.35"/>
            <param key="internalMomentOfInertia" value="0.01"/>
            <param key="radialDragCoefficient" value="0.1"/>
            <param key="rollDragCoefficient" value="0.01"/>
            <param key="constantPowerIntake" value="100"/>
            <param key="propulsionEfficiency" value="0.9"/>
            <param key="recuperationEfficiency" value="0.9"/>
            <param key="stoppingThreshold" value="0.1"/>
        </vType>

        <route id="W2E" edges="WE0 WE1 WE2 WE3 WE4 WE5"/>""" % (carfollowingModel), file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型          
            depart_list.append(['W2E', car_gen_steps_we[i]]) # insertionChecks="none"
        for i in range(car_count_per_lane):
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="%i" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]+arr_time, depart_speed), file = route)             
        print('</routes>', file=route)      
        
def generate_rou_file_single(ep, simulation_steps = 3600, car_count_per_lane = 100, path='rou_net'):
    #在4000s内随机生成400辆车
    random.seed(42)  #设置随机数种子，能够让结果重现
    timings_we = np.random.weibull(2, car_count_per_lane)
    timings_ew = np.random.weibull(2, car_count_per_lane)
    timings_we = np.sort(timings_we)
    timings_ew = np.sort(timings_ew)
    

    #reshape the distribution to fit the interval 0:max_steps
    car_gen_steps_we = []
    car_gen_steps_ew = []
    min_old_we = math.floor(timings_we[1])
    max_old_we = math.ceil(timings_we[-1])
    min_old_ew = math.floor(timings_ew[1])
    max_old_ew = math.ceil(timings_ew[-1])
    min_new = 0
    max_new = simulation_steps
    for i in range(len(timings_we)):
        car_gen_steps_we.append(((max_new - min_new) / (max_old_we - min_old_we)) * (timings_we[i] - max_old_we) + max_new)
        car_gen_steps_ew.append(((max_new - min_new) / (max_old_ew - min_old_ew)) * (timings_ew[i] - max_old_ew) + max_new)
 
    car_gen_steps_we = np.rint(car_gen_steps_we) 
    car_gen_steps_ew = np.rint(car_gen_steps_ew) 

    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy/unknown" tau="1" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1" maxSpeed="20" guiShape="passenger">
            <param key="has.battery.device" value="true"/>
            <param key="maximumBatteryCapacity" value="38000"/> 
            <param key="maximumPower" value="200000"/>
            <param key="vehicleMass" value="1830"/>
            <param key="frontSurfaceArea" value="2.6"/>
            <param key="airDragCoefficient" value="0.35"/>
            <param key="internalMomentOfInertia" value="0.01"/>
            <param key="radialDragCoefficient" value="0.1"/>
            <param key="rollDragCoefficient" value="0.01"/>
            <param key="constantPowerIntake" value="100"/>
            <param key="propulsionEfficiency" value="0.9"/>
            <param key="recuperationEfficiency" value="0.9"/>
            <param key="stoppingThreshold" value="0.1"/>
        </vType>
        
        <route id="W2E" edges="WE0 WE1"/>
        <route id="E2W" edges="EW0 EW1"/>""", file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型
            depart_list.append(['W2E', car_gen_steps_we[i]])
            depart_list.append(['E2W', car_gen_steps_ew[i]])
        depart_list = sorted(depart_list, key = lambda x:x[1])
        for i in range(car_count_per_lane * 2):
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]), file = route)             
        print('</routes>', file=route)
            
def generate_rou_file_double(ep, simulation_steps = 3600, car_count_per_lane = 100, path='rou_net'):
    #在4000s内随机生成400辆车
    random.seed(42)  #设置随机数种子，能够让结果重现
    timings_we = np.random.weibull(2, car_count_per_lane)
    timings_ew = np.random.weibull(2, car_count_per_lane)
    timings_we = np.sort(timings_we)
    timings_ew = np.sort(timings_ew)
    

    #reshape the distribution to fit the interval 0:max_steps
    car_gen_steps_we = []
    car_gen_steps_ew = []
    min_old_we = math.floor(timings_we[1])
    max_old_we = math.ceil(timings_we[-1])
    min_old_ew = math.floor(timings_ew[1])
    max_old_ew = math.ceil(timings_ew[-1])
    min_new = 0
    max_new = simulation_steps
    for i in range(len(timings_we)):
        car_gen_steps_we.append(((max_new - min_new) / (max_old_we - min_old_we)) * (timings_we[i] - max_old_we) + max_new)
        car_gen_steps_ew.append(((max_new - min_new) / (max_old_ew - min_old_ew)) * (timings_ew[i] - max_old_ew) + max_new)
 
    car_gen_steps_we = np.rint(car_gen_steps_we) 
    car_gen_steps_ew = np.rint(car_gen_steps_ew) 

    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy/unknown" tau="1" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1" maxSpeed="20" guiShape="passenger">
            <param key="has.battery.device" value="true"/>
            <param key="maximumBatteryCapacity" value="38000"/> 
            <param key="maximumPower" value="200000"/>
            <param key="vehicleMass" value="1830"/>
            <param key="frontSurfaceArea" value="2.6"/>
            <param key="airDragCoefficient" value="0.35"/>
            <param key="internalMomentOfInertia" value="0.01"/>
            <param key="radialDragCoefficient" value="0.1"/>
            <param key="rollDragCoefficient" value="0.01"/>
            <param key="constantPowerIntake" value="100"/>
            <param key="propulsionEfficiency" value="0.9"/>
            <param key="recuperationEfficiency" value="0.9"/>
            <param key="stoppingThreshold" value="0.1"/>
        </vType>
        
        <route id="W2E" edges="WE0 WE1 WE2"/>
        <route id="E2W" edges="EW0 EW1 EW2"/>""", file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型
            depart_list.append(['W2E', car_gen_steps_we[i]])
            depart_list.append(['E2W', car_gen_steps_ew[i]])
        depart_list = sorted(depart_list, key = lambda x:x[1])
        for i in range(car_count_per_lane * 2):
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]), file = route)             
        print('</routes>', file=route)

def generateRouTestFileDoueble(ep, simulation_steps = 3600, car_count_per_lane = 100, path='double_test'):
    #在4000s内随机生成400辆车
    random.seed(42)  #设置随机数种子，能够让结果重现
    timings_we = np.random.weibull(2, car_count_per_lane)
    timings_ew = np.random.weibull(2, car_count_per_lane)
    timings_we = np.sort(timings_we)
    timings_ew = np.sort(timings_ew)
    

    #reshape the distribution to fit the interval 0:max_steps
    car_gen_steps_we = []
    car_gen_steps_ew = []
    min_old_we = math.floor(timings_we[1])
    max_old_we = math.ceil(timings_we[-1])
    min_old_ew = math.floor(timings_ew[1])
    max_old_ew = math.ceil(timings_ew[-1])
    min_new = 0
    max_new = simulation_steps
    for i in range(len(timings_we)):
        car_gen_steps_we.append(((max_new - min_new) / (max_old_we - min_old_we)) * (timings_we[i] - max_old_we) + max_new)
        car_gen_steps_ew.append(((max_new - min_new) / (max_old_ew - min_old_ew)) * (timings_ew[i] - max_old_ew) + max_new)
 
    car_gen_steps_we = np.rint(car_gen_steps_we) 
    car_gen_steps_ew = np.rint(car_gen_steps_ew) 

    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy/unknown" tau="1" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1" maxSpeed="20" guiShape="passenger">
            <param key="has.battery.device" value="true"/>
            <param key="maximumBatteryCapacity" value="38000"/> 
            <param key="maximumPower" value="200000"/>
            <param key="vehicleMass" value="1830"/>
            <param key="frontSurfaceArea" value="2.6"/>
            <param key="airDragCoefficient" value="0.35"/>
            <param key="internalMomentOfInertia" value="0.01"/>
            <param key="radialDragCoefficient" value="0.1"/>
            <param key="rollDragCoefficient" value="0.01"/>
            <param key="constantPowerIntake" value="100"/>
            <param key="propulsionEfficiency" value="0.9"/>
            <param key="recuperationEfficiency" value="0.9"/>
            <param key="stoppingThreshold" value="0.1"/>
        </vType>
        
        <route id="W2E" edges="WE0 WE1 WE2"/>
        <route id="E2W" edges="EW0 EW1 EW2"/>""", file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型
            depart_list.append(['W2E', car_gen_steps_we[i]])
            depart_list.append(['E2W', car_gen_steps_ew[i]])
        depart_list = sorted(depart_list, key = lambda x:x[1])
        for i in range(car_count_per_lane * 2):
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]), file = route)             
        print('</routes>', file=route)

def generateCfgTestFileDouble(ep, path='double_test'):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file_name = path+'/intersection' + str(ep) + '.sumocfg'
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    net_file_name = path+'/intersection.net.xml'
    cfg_file = os.path.join(curr_path, cfg_file_name)
    rou_file = os.path.join(curr_path, rou_file_name)
    net_file = os.path.join(curr_path, net_file_name)

    with open(cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<configuration>
    <input>
        <net-file value="{}"/>
        <route-files value="{}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <tripinfo-output value="tripinfo.xml"/>
</configuration>""".format(net_file, rou_file), file=route)
    return cfg_file_name
        
def generateRouTestFileSingle(ep, simulation_steps = 3600, car_count_per_lane = 100, path='single_test'):
    #在4000s内随机生成400辆车
    random.seed(42)  #设置随机数种子，能够让结果重现
    timings_we = np.random.weibull(2, car_count_per_lane)
    timings_ew = np.random.weibull(2, car_count_per_lane)
    timings_we = np.sort(timings_we)
    timings_ew = np.sort(timings_ew)
    

    #reshape the distribution to fit the interval 0:max_steps
    car_gen_steps_we = []
    car_gen_steps_ew = []
    min_old_we = math.floor(timings_we[1])
    max_old_we = math.ceil(timings_we[-1])
    min_old_ew = math.floor(timings_ew[1])
    max_old_ew = math.ceil(timings_ew[-1])
    min_new = 0
    max_new = simulation_steps
    for i in range(len(timings_we)):
        car_gen_steps_we.append(((max_new - min_new) / (max_old_we - min_old_we)) * (timings_we[i] - max_old_we) + max_new)
        car_gen_steps_ew.append(((max_new - min_new) / (max_old_ew - min_old_ew)) * (timings_ew[i] - max_old_ew) + max_new)
 
    car_gen_steps_we = np.rint(car_gen_steps_we) 
    car_gen_steps_ew = np.rint(car_gen_steps_ew) 

    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy/unknown" tau="1" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1" maxSpeed="20" guiShape="passenger">
            <param key="has.battery.device" value="true"/>
            <param key="maximumBatteryCapacity" value="38000"/> 
            <param key="maximumPower" value="200000"/>
            <param key="vehicleMass" value="1830"/>
            <param key="frontSurfaceArea" value="2.6"/>
            <param key="airDragCoefficient" value="0.35"/>
            <param key="internalMomentOfInertia" value="0.01"/>
            <param key="radialDragCoefficient" value="0.1"/>
            <param key="rollDragCoefficient" value="0.01"/>
            <param key="constantPowerIntake" value="100"/>
            <param key="propulsionEfficiency" value="0.9"/>
            <param key="recuperationEfficiency" value="0.9"/>
            <param key="stoppingThreshold" value="0.1"/>
        </vType>
        
        <route id="W2E" edges="WE0 WE1"/>
        <route id="E2W" edges="EW0 EW1"/>""", file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型
            depart_list.append(['W2E', car_gen_steps_we[i]])
            depart_list.append(['E2W', car_gen_steps_ew[i]])
        depart_list = sorted(depart_list, key = lambda x:x[1])
        for i in range(car_count_per_lane * 2):
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]), file = route)             
        print('</routes>', file=route)

def generateCfgTestFileSingle(ep, path='single_test'):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file_name = path+'/intersection' + str(ep) + '.sumocfg'
    rou_file_name = path+'/intersection' + str(ep) + '.rou.xml'
    net_file_name = path+'/intersection.net.xml'
    cfg_file = os.path.join(curr_path, cfg_file_name)
    rou_file = os.path.join(curr_path, rou_file_name)
    net_file = os.path.join(curr_path, net_file_name)

    with open(cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<configuration>
    <input>
        <net-file value="{}"/>
        <route-files value="{}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <tripinfo-output value="tripinfo.xml"/>
</configuration>""".format(net_file, rou_file), file=route)
    return cfg_file_name