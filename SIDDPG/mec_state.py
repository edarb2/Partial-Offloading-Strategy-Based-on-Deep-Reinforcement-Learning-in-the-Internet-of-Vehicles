import math
import random
A = 5
import tensorflow as tf
import time
import matplotlib.pyplot as plt

import numpy as np

MAX_EPISODES = 500
# MAX_EPISODES = 50000

BATCH_SIZE = 64
OUTPUT_GRAPH = False

MIN_LIMIT = 0.01


class MEC_STATE(object):
    mec_range = 600
    mec_height = 15 # mec服务器高度
    m = 5  # 5个边服务器
    ground_length = 1000*m  # 场地长为5000m
    loc_mec = [1000*i+500 for i in range(m)]
    bandwidth_nums = 10
    p_noisy_nlos = 10 ** (-11)  # 噪声功率-80dBm
    f_mec = 1.2e6  # mec的计算频率1.2GHz
    r = 10 ** (-27)  # 芯片结构对cpu处理的影响因子
    s = 1000  # 单位bit处理所需cpu圈数1000
    p_uplink = 0.1  # 上行链路传输功率0.1W
    alpha0 = 1e-5  # 距离为1m时的参考信道增益-30dB = 0.001， -50dB = 1e-5
    #com_mec=np.array[10485760 * m] # 每个mec计算资源1048576kb
    #com_mec=np.array[10485760 * m] # 每个mec计算资源1048576kb
    com_mec = 1048576
    B = bandwidth_nums * 10 ** 6  # 每个mec带宽1MH
    slot_num = 30
    t_cell = 0.1 # 每0.1秒刷新环境
    #    Theta = 0.1 #比例a的调整幅度θ
    #    Theta_change = True #调整方向正负
    STEP = 0

    #################### ues ####################
    M = 50  # UE数量
    ue_id = [i for i in range(M)] #车辆编号
    ue_com = np.random.randint(150000, 175000, M) #车辆计算能力
    #ue_com = np.zeros(M)
    loc_ue_list = np.random.uniform(0, ground_length, M)  # 位置信息:x在0-1500随机
    loc_ue_direction = np.random.randint(0, 3, M) #车辆的行驶方向
    for i in range(M):
        loc_ue_direction[i] = loc_ue_direction[i] - 1
    ue_speed = np.random.uniform(20, 30, M) #车辆的行驶速度
    # task_list = np.random.randint(1572864, 2097153, M)      # 随机计算任务1.5~2Mbits ->对应总任务大小60
    #task_store_size = np.random.uniform(1097153, 2621440, M)  # 随机计算任务存储空间2~2.5Mbits -> 80
    #task_cpu_size = np.random.uniform(1097153, 2621440, M) # 随机计算任务cpu周期2~2.5Mbits -> 80
    task_store_size = np.random.uniform(209715, 262144, M)  # 随机计算任务存储空间2~2.5Mbits -> 80
    task_store_size_mark = task_store_size
    task_cpu_size = np.random.uniform(209715, 262144, M) # 随机计算任务cpu周期2~2.5Mbits -> 80
    task_cpu_size_mark = task_cpu_size
    sum_store_size = sum(task_store_size) #任务总和
    task_priority = np.random.randint(1, 6, M)  # 随机生成任务优先级
    most_toler_delay = np.random.uniform(1,2,M) #任务最大容忍时延
    most_toler_delay = most_toler_delay * task_cpu_size / (np.ones(M) * 209715)
#    for i in range(M):
#       most_toler_delay[i] =  most_toler_delay[i] * (np.log10(task_priority[i]+5)/np.log10(5)) * (task_store_size[i] * 2 / (109715+262144))
    offloading_ratio = np.zeros(M) #卸载率
    offloadingpolicy = np.zeros(M, dtype=int) #卸载策略，0为本地计算，1-5对应mec服务器编号
    loc_com_time = np.zeros(M) #本地计算时间
    task_trans_time = np.zeros(M) #任务传输时延
    task_wait_time = np.zeros(M) #任务等待时延
    task_com_time = np.zeros(M)  # 任务计算时延
    task_create_time = np.zeros(M) #任务生成时间
    distance =  np.zeros(M) #车辆与边缘服务器距离
    mec_list_num = np.zeros(m, dtype=int) #服务器车辆数
    mec_list = np.zeros((m, 50), dtype=int) #服务器队列
    task_finish_num = 0 #任务完成数


    offloading_store_size = np.zeros(M) #中间状态，卸载任务大小
    offloading_cpu_size = np.zeros(M) #中间状态，卸载需cpu大小
    offloading_spect_ratio = np.zeros(M)#频谱分配比例
    offloading_cpu_ratio = np.zeros(M)#CPU分配比例
    local_com_time = np.zeros(M)#本地计算时间
    offloading_time = np.zeros(M)#卸载计算时间
    remain_ratio = np.ones(M) #任务剩余比例
    # ue位置转移概率
    # 0:位置不变; 1:x+1,y; 2:x,y+1; 3:x-1,y; 4:x,y-1
    # loc_ue_trans_pro = np.array([[.6, .1, .1, .1, .1],
    #                              [.6, .1, .1, .1, .1],
    #                              [.6, .1, .1, .1, .1],
    #                              [.6, .1, .1, .1, .1]])
    action_bound = [0, 1]  # 对应tahn激活函数
    STATE_dim = M * 2  # 车辆位置x，车辆所需空间，车辆所需cpu
    state_dim = M * 3# 切分后车辆所需空间，车辆所需cpu
    action1_dim = M # 切分比例
    action2_dim = M * 2 # 空间分配比例，cpu分配比例


    def __init__(self): #初始化
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append(self.task_cpu_size, self.task_store_size)
        self.start_state = np.append(self.state, self.loc_ue_list)
        self.state_s = self.start_state
        self.state_S = self.state

    def reset_env(self): #重设环境
        self.loc_ue_list = np.random.uniform(0, self.ground_length, self.M)  # 车辆位置
        self.ue_direction = np.random.randint(0, 2, self.M)  # 车辆的行驶方向
        for i in range(self.M):
            if self.ue_direction[i] == 0:
                self.ue_direction[i] = 1
        self.ue_speed = np.random.uniform(10, 20, self.M)  # 车辆的行驶速度
        self.task_store_size = np.random.uniform(209715, 262144, self.M)  # 随机计算任务存储空间2~2.5Mbits -> 80
        self.task_store_size_mark = self.task_store_size
        self.task_cpu_size = np.random.uniform(209715, 262144, self.M)  # 随机计算任务cpu周期2~2.5Mbits -> 80
        self.task_cpu_size_mark = self.task_cpu_size
        self.sum_store_size = sum(self.task_store_size)  # 任务总和
        self.task_priority = np.random.randint(1, 6, self.M)  # 随机生成任务优先级
        self.most_toler_delay = np.random.uniform(1, 2, self.M)  # 任务最大容忍时延
        self.most_toler_delay = self.most_toler_delay * self.task_cpu_size / (np.ones(self.M) * 209715)
        # for i in range(self.M):
        #      self.most_toler_delay[i] = self.most_toler_delay[i] * (np.log10(self.task_priority[i] + 5) / np.log10(5)) * (
        #                  self.task_store_size[i] * 2 / (109715 + 262144))
        self.offloading_ratio = np.zeros(self.M)  # 卸载率
        self.offloadingpolicy = np.zeros(self.M, dtype=int)  # 卸载策略，0为本地计算，1-5对应mec服务器编号
        self.loc_com_time = np.zeros(self.M)  # 本地计算时间
        self.task_trans_time = np.zeros(self.M)  # 任务传输时延
        self.task_wait_time = np.zeros(self.M)  # 任务等待时延
        self.task_com_time = np.zeros(self.M)  # 任务计算时延
        self.task_create_time = np.zeros(self.M)  # 任务生成时间
        self.distance = np.zeros(self.M)  # 车辆与边缘服务器距离
        self.mec_list_num = np.zeros(self.m, dtype=int)  # 服务器车辆数
        self.mec_list = np.zeros((self.m, 50), dtype=int)  # 服务器队列
        self.task_finish_num = 0

        self.offloading_store_size = np.zeros(self.M)  # 中间状态，卸载任务大小
        self.offloading_cpu_size = np.zeros(self.M)  # 中间状态，卸载需cpu大小\
        self.offloading_spect_ratio = np.zeros(self.M)  # 频谱分配比例
        self.offloading_cpu_ratio = np.zeros(self.M)  # CPU分配比例
        self.local_com_time = np.zeros(self.M)  # 本地计算时间
        self.offloading_time = np.zeros(self.M)  # 卸载计算时间
        self.remain_ratio = np.ones(self.M)  # 任务剩余比例
        self.init_offloadingpolicy()

        return self._get_obs_s()


    # def reset_env(self): #重设环境
    #     self.loc_ue_list = np.random.uniform(0, self.ground_length, self.M)  # 车辆位置
    #     self.ue_direction = np.random.randint(0, 2, self.M)  # 车辆的行驶方向
    #     for i in range(self.M):
    #         if self.ue_direction[i] == 0:
    #             self.ue_direction[i] = 1
    #     self.ue_speed = np.random.uniform(13, 17, self.M)  # 车辆的行驶速度
    #     self.task_store_size = np.random.uniform(150000, 200000, self.M)  # 随机计算任务存储空间2~2.5Mbits -> 80
    #     self.task_store_size_mark = self.task_store_size
    #     self.task_cpu_size = np.random.uniform(150000, 200000,self.M)  # 随机计算任务cpu周期2~2.5Mbits -> 80
    #     self.task_cpu_size_mark = self.task_cpu_size
    #     self.sum_store_size = sum(self.task_store_size)  # 任务总和
    #     self.task_priority = np.random.randint(1, 6, self.M)  # 随机生成任务优先级
    #     self.most_toler_delay = np.random.uniform(2, 4, self.M)  # 任务最大容忍时延
    #     # for i in range(self.M):
    #     #     self.most_toler_delay[i] = self.most_toler_delay[i] * (np.log10(self.task_priority[i] + 5) / np.log10(5)) * (
    #     #                 self.task_store_size[i] * 2 / (109715 + 262144))
    #     self.offloading_ratio = np.zeros(self.M)  # 卸载率
    #     self.offloadingpolicy = np.zeros(self.M, dtype=int)  # 卸载策略，0为本地计算，1-5对应mec服务器编号
    #     self.loc_com_time = np.zeros(self.M)  # 本地计算时间
    #     self.task_trans_time = np.zeros(self.M)  # 任务传输时延
    #     self.task_wait_time = np.zeros(self.M)  # 任务等待时延
    #     self.task_com_time = np.zeros(self.M)  # 任务计算时延
    #     self.distance = np.zeros(self.M)  # 车辆与边缘服务器距离
    #     self.mec_list_num = np.zeros(self.m, dtype=int)  # 服务器车辆数
    #     self.mec_list = np.zeros((self.m, 50), dtype=int)  # 服务器队列
    #
    #     self.offloading_store_size = np.zeros(self.M)  # 中间状态，卸载任务大小
    #     self.offloading_cpu_size = np.zeros(self.M)  # 中间状态，卸载需cpu大小\
    #     self.offloading_spect_ratio = np.zeros(self.M)  # 频谱分配比例
    #     self.offloading_cpu_ratio = np.zeros(self.M)  # CPU分配比例
    #     self.local_com_time = np.zeros(self.M)  # 本地计算时间
    #     self.offloading_time = np.zeros(self.M)  # 卸载计算时间
    #     self.remain_ratio = np.ones(self.M)  # 任务剩余比例
    #     self.init_offloadingpolicy()
    #
    #     return self._get_obs_s()

    # def reset_step(self): #重设车辆任务大小和优先级
    #     # self.task_list = np.random.randint(1572864, 2097153, self.M)  # 随机计算任务1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
    #     # self.task_list = np.random.randint(2097152, 2621441, self.M)  # 随机计算任务1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
    #     # self.task_list = np.random.randint(2621440, 3145729, self.M)  # 随机计算任务1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
    #     self.task_list = np.random.randint(1097153, 2621440, self.M)  # 随机计算任务1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
    #     # self.task_list = np.random.randint(3145728, 3670017, self.M)  # 随机计算任务1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
    #     # self.task_list = np.random.randint(3670016, 4194305, self.M)  # 随机计算任务1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
    #     self.task_priority = np.random.randint(1, 6, self.M)
    #     self.sum_task_size = sum(self.task_list)

    # def reset(self): #重设环境并形成数组
    #     self.reset_env()
    #     # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
    #     self.state = np.append(self.state, self.loc_ue_list)
    #     self.state = np.append(self.state, self.task_store_size)
    #     self.state = np.append(self.state, self.task_cpu_size)
    #     return self._get_obs_S()

    # def _get_obs(self):
    #     # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
    #     self.state = np.append(self.com_mec, self.B)
    #     self.state = np.append(self.state, self.sum_task_size)
    #     self.state = np.append(self.state, self.ue_id)
    #     self.state = np.append(self.state, self.ue_com)
    #     self.state = np.append(self.state, self.loc_ue_list)
    #     self.state = np.append(self.state, self.loc_ue_direction)
    #     self.state = np.append(self.state, self.ue_speed)
    #     self.state = np.append(self.state, self.task_list)
    #     self.state = np.append(self.state, self.task_priority)
    #     return self.state

    def _get_obs_s(self):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.loc_ue_list += (self.loc_ue_direction * self.ue_speed * self.t_cell)
        for i in range(self.M): #车辆越界调整
            if self.loc_ue_list[i] > self.ground_length:
                self.loc_ue_list[i] = self.ground_length - (self.loc_ue_list[i] - self.ground_length)
                self.loc_ue_direction[i] = - self.loc_ue_direction[i]
            elif self.loc_ue_list[i] < 0:
                self.loc_ue_list[i] = - self.loc_ue_list[i]
                self.loc_ue_direction[i] = - self.loc_ue_direction[i]
        self.state_s = np.append(self.loc_ue_list, self.task_store_size)
        self.state_s = np.append(self.state_s, self.task_cpu_size)
        return self.state_s

    def _get_obs_S(self):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state_S = np.append(self.task_store_size, self.task_cpu_size)
        return self.state_S

    def resetone(self,x,j): #重置某车辆
        self.task_cpu_size[x] = np.random.randint(209715, 262144)  # 车辆任务大小
        self.task_store_size[x] = np.random.randint(209715, 262144)  # 车辆任务大小
        self.task_priority[x] = np.random.randint(1, 6)  # 任务优先级
        self.task_create_time[x] = self.t_cell * j #任务生成时间
        self.most_toler_delay[x] = np.random.uniform(1, 2) # 任务最大容忍时延
        self.most_toler_delay[x] = self.most_toler_delay[x] * self.task_cpu_size[x] / 209715
        self.remain_ratio[x] = 1


    # def init_time(i):
    #     loc_com_time[i] = task_list[i] /ue_com[i]  # 本地计算时间
    #     offloadingpolicy[i] = (int)(loc_ue_list[i])//1000
    #     if loc_ue_list[i]-loc_mec[offloadingpolicy[i]] > 400 and offloadingpolicy[i] < 4:
    #         offloadingpolicy[i] = offloadingpolicy[i]+1 if loc_ue_direction[i]>0 else offloadingpolicy[i]
    #     elif loc_ue_list[i]-loc_mec[offloadingpolicy[i]] < -400 and offloadingpolicy[i] > 0:
    #         offloadingpolicy[i] = offloadingpolicy[i]-1 if loc_ue_direction[i]<0 else offloadingpolicy[i]
    #     dx = loc_ue_list[i]-loc_mec[offloadingpolicy[i]]
    #     dy = mec_height
    #     distance[i] = np.sqrt(dx * dx + dy * dy)
    #     p_noise = p_noisy_nlos
    #     g_uav_ue = abs(alpha0 / distance[i] ** 2)  # 信道增益
    #     trans_rate = B * math.log2(1 + p_uplink * g_uav_ue / p_noise)  # 上行链路传输速率bps
    #     task_trans_time[i] = task_list[i]/trans_rate #传输时延
    #     task_com_time[i] = task_list[i] / com_mec #计算时延
    #     task_wait_time[i] = 0 #队列等待，初始化为0
    #     return offloadingpolicy[i],loc_com_time[i],task_trans_time[i],task_wait_time[i],task_com_time[i]

    # def init_offloading_ratio(loc_com_time,task_trans_time,task_wait_time,task_com_time):#根据初始时间参数，通过公式计算比例a
    #     offloading_ratio = (loc_com_time - task_trans_time - task_wait_time)/(loc_com_time + task_com_time)
    #     if offloading_ratio < 0: offloading_ratio = 0
    #     return offloading_ratio

    def init_offloadingpolicy(self):
        for i in range(self.M):
            self.offloadingpolicy[i] = (int)(self.loc_ue_list[i]) // 1000
            if self.loc_ue_list[i]-self.loc_mec[self.offloadingpolicy[i]] > 400 and self.offloadingpolicy[i] < 4:
                self.offloadingpolicy[i] = self.offloadingpolicy[i]+1 if self.loc_ue_direction[i]>0 else self.offloadingpolicy[i]
            elif self.loc_ue_list[i]-self.loc_mec[self.offloadingpolicy[i]] < -400 and self.offloadingpolicy[i] > 0:
                self.offloadingpolicy[i] = self.offloadingpolicy[i]-1 if self.loc_ue_direction[i]<0 else self.offloadingpolicy[i]
        self.MEC_LIST(self.offloadingpolicy)

    def task_update(self,action2,j):
        print("task_finish_num:",self.task_finish_num)
        print("t_cell * j:",self.t_cell * j)
        for i in range (self.M):
            self.remain_ratio[i] = max(0,self.remain_ratio[i] - (self.ue_com[i] + self.f_mec * action2[i+50]) * (self.t_cell * j - self.task_create_time[i]) / self.task_cpu_size_mark[i])
        print("remain_ratio:",self.remain_ratio)
        self.task_cpu_size = self.task_cpu_size_mark * self.remain_ratio
        self.task_store_size = self.task_store_size_mark * self.remain_ratio
        #print("self.task_store_size",self.task_store_size,"self.task_cpu_size",self.task_cpu_size)


    def step1 (self, action1,j):
        self.offloading_ratio = action1
        #print("offloading_ratio：",self.offloading_ratio,"task_cpu_size：",self.task_cpu_size,"ue_com:",self.ue_com)
        for i in range (self.M):
            if self.remain_ratio[i] != 0:
                self.local_com_time[i] = (self.task_cpu_size[i] - self.offloading_ratio[i] * self.task_cpu_size[i]) / self.ue_com[i] + (self.t_cell * j - self.task_create_time[i])
        self.offloading_store_size = np.multiply(self.offloading_ratio,self.task_store_size)
        self.offloading_cpu_size = np.multiply(self.offloading_ratio,self.task_cpu_size)
        return self._get_obs_S()

    def step2 (self, action2,j):
        is_terminal = False
        action2_mark = np.array_split(action2, 2)
        self.offloading_spect_ratio = action2_mark[0]
        self.offloading_cpu_ratio = action2_mark[1]
        # self.task_store_size = self.arr_max_ele(np.zeros(50),self.task_store_size - (self.com_mec + self.ue_com) * self.t_cell)
        offloading_transmit_time = self.offloading_store_size/((self.offloading_spect_ratio) * self.B)
        offloading_com_time = self.offloading_cpu_size/((self.offloading_cpu_ratio) * self.f_mec)
        for i in range (self.M):
            if self.remain_ratio[i] != 0:
                self.offloading_time[i] = offloading_transmit_time[i] + offloading_com_time[i]
        #print("频谱资源比例：",self.offloading_spect_ratio,"计算资源比例：",self.offloading_cpu_ratio)
        print("传输time:",offloading_transmit_time,"计算time：",offloading_com_time)
        time_gap = abs(self.local_com_time - self.offloading_time)
        print("time_gap:",time_gap)
        reward1_mark = np.zeros(self.M)
        for i in range(self.M):
        #    if abs(self.local_com_time[i] - self.offloading_time[i]) == 0:
        #        reward1_mark[i] = 10
        #    else:
        #        reward1_mark[i] = 10 - abs(self.local_com_time[i] - self.offloading_time[i])
            reward1_mark[i] = (1-abs(math.tanh(time_gap[i])))
            # if reward1_mark[i] < -5 :
            #     reward1_mark[i] = -5
        #print("reward1_mark",reward1_mark,"time_gap",time_gap)
        #print("时间差:",self.local_com_time - self.offloading_time,"时间差和：",sum(self.local_com_time - self.offloading_time))
        print("reward1:",reward1_mark)
        print("本地时间：",self.local_com_time,"卸载时间：",self.offloading_time,"时间差:",self.local_com_time - self.offloading_time)
        self.task_update(action2,j)
        print("task_create_time:",self.task_create_time)
        for i in range (self.M):
            if self.remain_ratio[i] == 0:
                self.resetone(i,j)
                self.task_finish_num += 1
            # else:
            #     reward1_mark[i] = 0
        num = self.get_disfinish_number()
        #print("任务剩余数:",num)
#        print("reward1mark:",reward1_mark)
        if num != 0:
            reward1 =  sum(reward1_mark) #* 0.9 ** j
            reward2 =  self.delay_punish(self.offloading_time,self.local_com_time,num) #* 0.9 ** j
        else:
            reward1 = 0
            reward2 = 0
            is_terminal = True
        #print("cpu.size:",self.task_cpu_size)
        #print('任务剩余比例：',self.remain_ratio)
        self.sum_store_size = sum(self.task_store_size)
        self.STEP += 1
        return self._get_obs_s(),reward1,reward2,self.STEP,self.sum_store_size,is_terminal

    def get_disfinish_number(self):
        num = 0
        for i in range(self.M):
            if self.task_cpu_size[i] != 0:
                num += 1
        return num

    # def update_offloading_ratio(offloading_ratio,com_delay,loc_com_time,up_limit,down_limit):
    #     print("----------------------")
    #     print(offloading_ratio[1])
    #     leng = len(offloading_ratio)
    #     for i in range(leng):
    #         time1 = loc_com_time[i]*(1 - offloading_ratio[i])
    #         time2 = com_delay[i]
    #     #        if time1 > time2 and self.Theta_change == "True" :
    #     #            self.offloading_ratio = self.offloading_ratio+
    #     #        elif time1 < time2 and self.Theta_change == "True" :
    #     #            self.offloading_ratio -= self.Theta
    #     #        elif time1 > time2 and self.Theta_change == "False":
    #     #            self.offloading_ratio += self.Theta/2
    #     #        elif time1 < time2 and self.Theta_change == "False":
    #     #            self.offloading_ratio -= self.Theta/2
    #         if time1 > time2:
    #             offloading_ratio[i] = (offloading_ratio[i] + up_limit[i]) / 2
    #         elif time1 < time2:
    #             offloading_ratio[i] = (offloading_ratio[i] +down_limit[i]) / 2
    #     print(offloading_ratio[1])
    #     return offloading_ratio
    # def update_offloading_ratio(offloading_ratio_before,offloading_ratio_after,com_delay,loc_com_time,limit):
    #     leng = len(offloading_ratio_before)
    #     for i in range(leng):
    #         time1 = loc_com_time[i]*(1 - offloading_ratio_after[i])
    #         time2 = com_delay[i]
    #     #        if time1 > time2 and self.Theta_change == "True" :
    #     #            self.offloading_ratio = self.offloading_ratio+
    #     #        elif time1 < time2 and self.Theta_change == "True" :
    #     #            self.offloading_ratio -= self.Theta
    #     #        elif time1 > time2 and self.Theta_change == "False":
    #     #            self.offloading_ratio += self.Theta/2
    #     #        elif time1 < time2 and self.Theta_change == "False":
    #     #            self.offloading_ratio -= self.Theta/2
    #         if time1 > time2 and limit[i]==1:
    #             offloading_ratio_after[i] = (offloading_ratio_after[i] + 1) / 2
    #         elif time1 < time2 and limit[i]==1:
    #             offloading_ratio_after[i] = (offloading_ratio_before[i]+offloading_ratio_after[i]) / 2
    #         if time1 > time2 and limit[i]==0:
    #             offloading_ratio_after[i] = (offloading_ratio_before[i]+offloading_ratio_after[i]) / 2
    #         elif time1 < time2 and limit[i]==0:
    #             offloading_ratio_after[i] = (offloading_ratio_after[i]) / 2
    #         offloading_ratio_before[i] = offloading_ratio_after[i]
    #     return offloading_ratio_before,offloading_ratio_after


    def MEC_LIST(self,offloadingpolicy):
        # 生成各服务器内部队列
        #print("offloadingpolicy",self.offloadingpolicy)
        for i in range(self.M):
            if offloadingpolicy[i] == 0:
                self.mec_list[0, self.mec_list_num[0]] = i
                self.mec_list_num[0] += 1
            elif offloadingpolicy[i] == 1:
                self.mec_list[1, self.mec_list_num[1]] = i
                self.mec_list_num[1] += 1
            elif offloadingpolicy[i] == 2:
                self.mec_list[2, self.mec_list_num[2]] = i
                self.mec_list_num[2] += 1
            elif offloadingpolicy[i] == 3:
                self.mec_list[3, self.mec_list_num[3]] = i
                self.mec_list_num[3] += 1
            elif offloadingpolicy[i] == 4:
                self.mec_list[4, self.mec_list_num[4]] = i
                self.mec_list_num[4] += 1
        #冒泡排序进行队列内部排序(主要依据：优先级，次要依据：到达顺序)
        for i in range (0,5):
            for n in range (1,self.mec_list_num[i]):
                for m in range (0,self.mec_list_num[i]-n):
                    if  self.task_priority[self.mec_list[i][m]] > self.task_priority[self.mec_list[i][m+1]]:
                        self.mec_list[i][m],self.mec_list[i][m+1] = self.mec_list[i][m+1],self.mec_list[i][m]

    # def load_balance(mec_list_num,mec_list,task_wait_time,task_com_time):
    #     judge = True
    #     maxtime = 0
    #     maxtime_next = 0
    #     print("---------------load_balance---------------")
    #     while judge:
    #         last_task_finish_time = np.zeros(m)
    #         for i in range (m):
    #             last_task_finish_time[i] = task_wait_time[mec_list[i,mec_list_num[i]]] + task_com_time[mec_list[i,mec_list_num[i]]]
    #         fastest_mec = select_min(last_task_finish_time)
    #         lastest_mec = select_max(last_task_finish_time)
    #         maxtime = last_task_finish_time[lastest_mec]
    #         print(last_task_finish_time[fastest_mec])
    #         print(last_task_finish_time[lastest_mec])
    #         print("----------")
    #         mec_list[fastest_mec, mec_list_num[fastest_mec]+1] = mec_list[lastest_mec, mec_list_num[lastest_mec]]
    #         mec_list[lastest_mec, mec_list_num[lastest_mec]] = 0
    #         task_wait_time[mec_list[fastest_mec, mec_list_num[fastest_mec]+1]] = task_wait_time[mec_list[fastest_mec, mec_list_num[fastest_mec]]] + task_com_time[mec_list[fastest_mec, mec_list_num[fastest_mec]]]
    #         mec_list_num[fastest_mec] += 1
    #         mec_list_num[lastest_mec] -= 1
    #         lastest_mec_next = select_max(last_task_finish_time)
    #         for i in range (m):
    #             last_task_finish_time[i] = task_wait_time[mec_list[i,mec_list_num[i]]] + task_com_time[mec_list[i,mec_list_num[i]]]
    #         maxtime_next = last_task_finish_time[lastest_mec_next]
    #         if maxtime_next < maxtime:
    #             judge = True
    #         else:
    #             judge = False
    #         print(last_task_finish_time[fastest_mec])
    #         print(last_task_finish_time[lastest_mec])
    #         print("----------")
    #     return mec_list_num,mec_list,task_wait_time,task_com_time

    # def update_time(offloading_ratio,mec_list_num,mec_list):
    #     for i in range(M):
    #         task_list[i] = task_list[i] * offloading_ratio[i]
    #     wait_time = 0
    #     com_time = 0
    #     for i in range(m):
    #         for j in range(mec_list_num[i]):
    #             wait_time = wait_time + com_time
    #             com_time = task_list[mec_list[i,j]]/com_mec
    #             task_wait_time[mec_list[i,j]] = wait_time
    #             task_com_time[mec_list[i,j]] = com_time
    #
    #     return task_wait_time,task_com_time

    #            #越界参数
    #            over_road_list = np.zeros(self.M) #越界车辆号
    #            over_road_num = 0 #越界车辆数
    #            delay = np.zeros(self.M)
    #            reward = 0
    #
    #            for i in range(self.M): #判断是否有车辆超界
    #               if self.loc_ue_list[i] < 0 or self.loc_ue_list[i] > 1500:
    #                    over_road_num += 1
    #                    over_road_list[over_road_num-1] = i
    #
    #            if over_road_num != 0 :  # ue位置超界
    #                reset_dist = True
    #                for i in range(over_road_num):
    #                    self.resetone(over_road_list[i])
    #                    self.loc_ue_list[i] = self.loc_ue_list[i]
    #                    self.ue_direction[i] = self.loc_ue_direction[i]
    #                # delay = self.com_delay(self.loc_ue_list, self.ue_id, self.loc_mec, self.mec_height, offloadingpolicy,offloading_ratio, task_size)  # 计算delay
    #                reward = 0
    #                # 更新下一时刻状态
    #                # self.reset2(delay, self.offloading_ratio, task_size, self.ue_id)
    #
    #    #        for i in range(self.M) :
    #    #            t_down = self.com_delay(loc_ue, i, offloadingpolicy, offloading_ratio, task_size)
    #
    #            #if t_up < 0 or t_up < 0 or t_server < 0:
    #            #    raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
    #            #t_tr = t_up + t_down + t_server
    #
    #            # MEC服务器计算资源调整
    #            for i in range(self.M):
    #                if self.task_state[i] == 0 and self.com_mec[offloadingpolicy[i]] >= task_size[i] :
    #                    self.com_mec[offloadingpolicy[i]] -= task_size[i]
    #                    self.task_state[i] = 1
    #                elif self.task_state[i] == 2 :
    #                    self.com_mec[offloadingpolicy[i]] += task_size[i]
    #                else : continue
    #
    #            #情况分类
    #            if self.sum_task_size == 0:  # 计算任务全部完成
    #                is_terminal = True
    #                reward = 0
    #                STEP += 1
    #            #else : #正常运行
    #
    #            return self._get_obs(), reward, is_terminal, step_redo, reset_dist, STEP, offloading_ratio, offloadingpolicy

    # 重置ue任务大小，剩余总任务大小，ue位置，并记录到文件
    # def reset2(self, delay, offloading_ratio, task_size, ue_id):
    #     self.sum_task_size -= self.task_list[ue_id]  # 剩余任务量
    #     self.ue_move(delay)
    #     self.reset_step()  # ue随机计算任务1~2Mbits # 4个ue，ue的遮挡情况
    #     # 记录UE花费
    #     file_name = 'output.txt'
    #     # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
    #     with open(file_name, 'a') as file_obj:
    #         file_obj.write("\nUE-" + '{:d}'.format(ue_id) + ", task size: " + '{:d}'.format(int(task_size)) + ", offloading ratio:" + '{:.2f}'.format(offloading_ratio))
    #         file_obj.write("\ndelay:" + '{:.2f}'.format(delay))

    # 通信时延
    #def com_delay(self, loc_ue_list, ue_id, offloadingpolicy, offloading_ratio, task_size):
    #    dx = self.loc_mec[offloadingpolicy] - loc_ue_list[ue_id]
    #    dh = self.mec_height
    #    dist_uav_ue = np.sqrt(dx * dx + dh * dh)
    #    p_noise = self.p_noisy_nlos
    #    g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)  # 信道增益
    #    trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)  # 上行链路传输速率bps
    #    t_tr = offloading_ratio * task_size / trans_rate  # 上传时延,1B=8bit
    #    return t_tr

    #def com_delay(self,task_wait_time,task_com_time):
    #    offloading_delay = task_wait_time+task_com_time
    #    total_delay = task_trans_time+task_wait_time+task_com_time
    #    return total_delay


    #def delay_punish(self, most_toler_delay, loc_ue, ue_id, loc_mec,mec_height, offloadingpolicy, offloading_ratio, task_size):
    #    delay = self.com_delay(loc_ue, ue_id,offloadingpolicy, offloading_ratio, task_size)
    #    diff = delay - most_toler_delay[ue_id]
    #    return 1 - 2 ** diff

    def delay_punish(self,offloading_time,local_com_time,num):
        #print("self.offloading_time",offloading_time,"local_com_time",local_com_time)
        delay = self.arr_max_ele(offloading_time, local_com_time)
        #print("delay",delay)
        #print("self.most_toler_delay", self.most_toler_delay)
        # delay_ratio = self.arr_max_ele(-np.ones(self.M),(self.most_toler_delay - delay)/self.most_toler_delay)
        delay_ratio = self.most_toler_delay - delay
        for i in range(self.M):
            # if delay_ratio[i] < - 10:
            #     delay_ratio[i] = - 10
            delay_ratio[i] = math.tanh(delay_ratio[i])
            if self.remain_ratio[i] == 0:
                delay_ratio[i] = 0
        print("delay_ratio:",delay_ratio)
        reward = sum(delay_ratio) * 2

        return reward

    def take_min_array(self,a):
        min_num = a[0]
        mark = 0
        for i in range (1,len(a)):
            if a[i]<min_num :
                min_num = a[i]
                mark = i
        for i in range (mark,len(a)-1):
            a[i] = a[i+1]
        a[len(a)] = 0
        return mark, a

    def inverted_order_array(self,a):
        a = a.T[np.lexsort(a)].T
        return a

    def select_max(self,a):
        max_num = a[0]
        mark_max = 0
        for i in range (len(a)):
            if a[i] > max_num:
                max_num = a[i]
                mark_max = i
        return mark_max

    def select_min(self,a):
        min_num = a[0]
        mark_min = 0
        for i in range(len(a)):
            if a[i] < min_num:
                min_num = a[i]
                mark_min = i
        return mark_min

    def judge_convergence(self,array1,array2):
        count = 0
        count_judge = True
        leng = len(array1)
        for i in range(leng):
            if array1[i] - array2[i] < 0.001:
                count = count + 1
            if count == leng:
                count_judge = False
            return count_judge

    def arr_max_ele(self,array1,array2):
        length = min(len(array1),len(array2))
        if len(array1)>len(array2):
            array = array1
            for i in range(length):
                if array2[i] > array1[i]:
                    array[i] = array2[i]
        else :
            array = array2
            for i in range(length):
                if array1[i] > array2[i]:
                    array[i] = array1[i]
        return array
###############################  training  ####################################
#np.random.seed(1)
#tf.compat.v1.set_random_seed(1)

#MAX_EP_STEPS = Env.slot_num
#s_dim = Env.state_dim
#a_dim = Env.action_dim
#a_bound = Env.action_bound  # [-1,1]

#var = 1  # control exploration
#var = 0.01  # control exploration
#t1 = time.time()
#ep_reward_list = []
#mec_list = Env.MEC_LIST

#for i in range(MAX_EPISODES):
#    s = Env.reset
#    ep_reward = 0
#    S_mark = s
#    j = 0
#    while j < MAX_EP_STEPS:

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #   eval_policy(ddpg, env)
# reward = 0
# Episode = 0
# offloading_ratio_before = offloading_ratio
# offloading_ratio_after = offloading_ratio
# reward_list = []
# limit = np.zeros(M)
# unconverg = True
# for i in range (M):
#     offloadingpolicy[i],loc_com_time[i], task_trans_time[i], task_wait_time[i], task_com_time[i] = init_time(i)
#     offloading_ratio_after[i] = init_offloading_ratio(loc_com_time[i], task_trans_time[i], task_wait_time[i], task_com_time[i])
# mec_list_num, mec_list = MEC_LIST(offloadingpolicy)
# #mec_list_num,mec_list,task_wait_time,task_com_time = load_balance(mec_list_num,mec_list,task_wait_time,task_com_time)
#
# while unconverg :
#     count = 0
#     task_wait_time,task_com_time = update_time(offloading_ratio_after,mec_list_num, mec_list)
#     total_delay = com_delay(task_wait_time,task_com_time)
#     # for i in range(M):
#     #     if offloading_ratio_before[i] == offloading_ratio_after[i]:
#     #         limit[i] = offloading_ratio_before[i]
#     for i in range (M):
#         if offloading_ratio_before[i] <= offloading_ratio_after[i]:limit[i] = 1
#         else:limit[i] = 0
#     offloading_ratio_before,offloading_ratio_after = update_offloading_ratio(offloading_ratio_before,offloading_ratio_after,total_delay,loc_com_time,limit)
#     reward = delay_punish(total_delay,most_toler_delay,loc_com_time,offloading_ratio)
#     reward_list = np.append(reward_list,reward)
#     unconverg = judge_convergence(offloading_ratio_before,offloading_ratio_after)
#     Episode = Episode + 1
#     print("Episode")
#     print(Episode)
#     print("offloading_ratio：")
#     print(offloading_ratio_after)
#     print("reward：")
#     print(reward)
#     if Episode == 100:break
#
#
# # print('Running time: ', time.time() - t1)
# #plt.plot(offloading_ratio_before)
# plt.plot(reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# #plt.ylabel("offloadingpolicy_before")
# plt.show()
