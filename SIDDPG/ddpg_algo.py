"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.

Using:
tensorflow 1.14.0
gym 0.15.3
"""
import copy

import tensorflow as tf
import numpy as np
from mec_state import MEC_STATE
import time
import matplotlib.pyplot as plt
from state_normalization import StateNormalization1
from state_normalization import StateNormalization2

#####################  hyper parameters  ####################
MAX_EPISODES =10
# MAX_EPISODES = 50000

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
# LR_A = 0.1  # learning rate for actor
# LR_C = 0.2  # learning rate for critic
GAMMA = 0.001  # optimal reward discount
# GAMMA = 0.999  # reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01
# MEMORY_CAPACITY = 5000
MEMORY_CAPACITY = 300# 经验池容量
BATCH_SIZE = 16#32#64
OUTPUT_GRAPH = False


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
# def eval_policy(ddpg, eval_episodes=10):
#     # eval_env = gym.make(env_name)
#     eval_env = MEC_STATE()
#     # eval_env.seed(seed + 100)
#     avg_reward1 = 0
#     avg_reward2 = 0
#     for i in range(eval_episodes):
#         state = eval_env.reset_env()
#         # while not done:
#         for j in range(int(len(eval_env.loc_ue_list))):
#             action1 = ddpg.choose_action1(state)
#             action1 = action_limit1(action1, mec_list_num, mec_list)
#             STATE, local_com_time = eval_env.step1(action1)
#             action2 = ddpg.choose_action2(STATE)
#             action2 = action_limit2(action2, mec_list_num, mec_list)
#             state,reward1,reward2 = eval_env.step1(action2)
#             avg_reward1 += reward1
#             avg_reward2 += reward2
#
#     avg_reward1 /= eval_episodes
#     avg_reward2 /= eval_episodes
#     print("---------------------------------------")
#     print(f"Evaluation over {eval_episodes} episodes: {avg_reward1:.3f}")
#     print("---------------------------------------")
#     return avg_reward1,avg_reward2


###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a1_dim, a2_dim, s_dim, S_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, a1_dim + a2_dim + s_dim * 2 + S_dim * 2 + 2), dtype=np.float32)  # memory里存放当前和下一个state，动作和奖励
        self.pointer = 0
        self.sess = tf.compat.v1.Session()

        self.a1_dim,self.a2_dim,self.s_dim,self.S_dim,self.a_bound = a1_dim, a2_dim, s_dim, S_dim, a_bound
        self.s = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S = tf.compat.v1.placeholder(tf.float32, [None, S_dim], 'S')
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, S_dim], 'S_')
        self.R1 = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r1')
        self.R2 = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r2')

        with tf.compat.v1.variable_scope('Actor1'):
            self.a1 = self._build_a1(self.s, scope='eval1', trainable=True)
            a1_ = self._build_a1(self.s_, scope='target1', trainable=False)
        with tf.compat.v1.variable_scope('Actor2'):
            self.a2 = self._build_a2(self.S, scope='eval2', trainable=True)
            a2_ = self._build_a2(self.S_, scope='target2', trainable=False)
        with tf.variable_scope('Critic1'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q1 = self._build_c1(self.s, self.a1, scope='eval1', trainable=True)
            q1_ = self._build_c1(self.s_, a1_, scope='target1', trainable=False)
        with tf.variable_scope('Critic2'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q2 = self._build_c2(self.S, self.a2, scope='eval2', trainable=True)
            q2_ = self._build_c2(self.S_, a2_, scope='target2', trainable=False)

        self.a1e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor1/eval1')
        self.a1t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor1/target1')
        self.c1e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic1/eval1')
        self.c1t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic1/target1')
        self.a2e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor2/eval2')
        self.a2t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor2/target2')
        self.c2e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic2/eval2')
        self.c2t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic2/target2')

        # target net replacement
        self.soft_replace1 = [tf.assign(t1, (1 - TAU) * t1 + TAU * e1)
                             for t1, e1 in zip(self.a1t_params + self.c1t_params, self.a1e_params + self.c1e_params)]
        self.soft_replace2 = [tf.assign(t2, (1 - TAU) * t2 + TAU * e2)
                             for t2, e2 in zip(self.a2t_params + self.c2t_params, self.a2e_params + self.c2e_params)]

        q1_target = self.R1 + GAMMA * q1_
        q2_target = self.R2 + GAMMA * q2_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error1 = tf.losses.mean_squared_error(labels=q1_target, predictions=q1)
        td_error2 = tf.losses.mean_squared_error(labels=q2_target, predictions=q2)
        self.c1train = tf.train.AdamOptimizer(LR_C).minimize(td_error1, var_list=self.c1e_params)
        self.c2train = tf.train.AdamOptimizer(LR_C).minimize(td_error2, var_list=self.c2e_params)

        a1_loss =  -tf.reduce_mean(q1)  # maximize the q
        a2_loss =  -tf.reduce_mean(q2)

        self.a1train = tf.train.AdamOptimizer(LR_A).minimize(a1_loss, var_list=self.a1e_params)
        self.a2train = tf.train.AdamOptimizer(LR_A).minimize(a2_loss, var_list=self.a2e_params)

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action1(self, s):
        temp = self.sess.run(self.a1, {self.s: s[np.newaxis, :]})
        return temp

    def choose_action2(self, S):
        temp = self.sess.run(self.a2, {self.S: S[np.newaxis, :]})
        return temp

    def learn(self):
        self.sess.run(self.soft_replace1)
        self.sess.run(self.soft_replace2)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt  = self.memory[indices, :]
        bs  = bt[:, :self.s_dim]
        ba1 = bt[:, self.s_dim: self.s_dim + self.a1_dim]
        br1 = bt[:, self.s_dim + self.a1_dim: self.s_dim + self.a1_dim + 1]
        bS  = bt[:, self.s_dim + self.a1_dim + 1:self.s_dim + self.a1_dim + self.S_dim + 1]
        ba2 = bt[:, self.s_dim + self.a1_dim + self.S_dim +1: self.s_dim + self.a1_dim + self.S_dim +self.a2_dim+ 1]
        br2 = bt[:, self.s_dim + self.a1_dim + self.S_dim +self.a2_dim+ 1: self.s_dim + self.a1_dim + self.S_dim +self.a2_dim+ 2]
        bs_ = bt[:, self.s_dim + self.a1_dim + self.S_dim +self.a2_dim+ 2: self.s_dim * 2 + self.a1_dim + self.S_dim +self.a2_dim+ 2]
        bS_ = bt[:, -self.S_dim:]


        self.sess.run(self.a1train, {self.s: bs})
        self.sess.run(self.c1train, {self.s: bs, self.a1: ba1, self.R1: br1, self.s_: bs_})
        self.sess.run(self.a2train, {self.S: bS})
        self.sess.run(self.c2train, {self.S: bS, self.a2: ba2, self.R2: br2, self.S_: bS_})

    def store_transition(self,s, a1, r1, S, a2, r2, s_, S_):
        transition = np.hstack((s, a1, r1, S, a2, r2, s_, S_))
        # transition = np.hstack((s, [a], [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a1(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l11', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.tanh, name='l13', trainable=trainable)
            a1 = tf.layers.dense(net, self.a1_dim, activation=tf.nn.tanh, name='a1', trainable=trainable)
            return tf.multiply(a1, self.a_bound[1], name='scaled_a1')

    def _build_c1(self, s, a1, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 256
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a1 = tf.get_variable('w1_a', [self.a1_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a1, w1_a1) + b1)
            net = tf.layers.dense(net, 128, activation=tf.nn.relu6, name='l12', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu6, name='l13', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def _build_a2(self, S, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.layers.dense(S, 256, activation=tf.nn.relu, name='l21', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.tanh, name='l23', trainable=trainable)
            a2 = tf.layers.dense(net, self.a2_dim, activation=tf.nn.tanh, name='a2', trainable=trainable)
            return tf.multiply(a2, self.a_bound[1], name='scaled_a2')

    def _build_c2(self, S, a2, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 256
            w1_S = tf.get_variable('w2_s', [self.S_dim, n_l1], trainable=trainable)
            w1_a2 = tf.get_variable('w2_a', [self.a2_dim, n_l1], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(S, w1_S) + tf.matmul(a2, w1_a2) + b2)
            net = tf.layers.dense(net, 128, activation=tf.nn.relu6, name='l22', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu6, name='l23', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    # def _build_a1(self, s, scope, trainable):
    #     with tf.compat.v1.variable_scope(scope):
    #         net = tf.layers.dense(s, 50, activation=tf.nn.relu6, name='l11', trainable=trainable)
    #         net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l13', trainable=trainable)
    #         a1 = tf.layers.dense(net, self.a1_dim, activation=tf.nn.tanh, name='a1', trainable=trainable)
    #         return tf.multiply(a1, self.a_bound[1], name='scaled_a1')
    #
    # def _build_c1(self, s, a1, scope, trainable):
    #     with tf.variable_scope(scope):
    #         n_l1 = 400
    #         w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
    #         w1_a1 = tf.get_variable('w1_a', [self.a1_dim, n_l1], trainable=trainable)
    #         b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
    #         net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a1, w1_a1) + b1)
    #         net = tf.layers.dense(net, 25, activation=tf.nn.relu, name='l13', trainable=trainable)
    #         return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
    #
    # def _build_a2(self, S, scope, trainable):
    #     with tf.compat.v1.variable_scope(scope):
    #         net = tf.layers.dense(S, 50, activation=tf.nn.relu6, name='l21', trainable=trainable)
    #         net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l23', trainable=trainable)
    #         a2 = tf.layers.dense(net, self.a2_dim, activation=tf.nn.tanh, name='a2', trainable=trainable)
    #         return tf.multiply(a2, self.a_bound[1], name='scaled_a2')
    #
    # def _build_c2(self, S, a2, scope, trainable):
    #     with tf.variable_scope(scope):
    #         n_l1 = 400
    #         w1_S = tf.get_variable('w2_s', [self.S_dim, n_l1], trainable=trainable)
    #         w1_a2 = tf.get_variable('w2_a', [self.a2_dim, n_l1], trainable=trainable)
    #         b2 = tf.get_variable('b2', [1, n_l1], trainable=trainable)
    #         net = tf.nn.relu6(tf.matmul(S, w1_S) + tf.matmul(a2, w1_a2) + b2)
    #         net = tf.layers.dense(net, 25, activation=tf.nn.relu, name='l23', trainable=trainable)
    #         return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

def action_limit11(a):
    a1 = np.zeros(50)
    for i in range (0,50):
        a1[i] = a[0][i]
    return a1

def action_limit12(a):
    a1 = a
    for i in range(50):
        if MEC_STATE.remain_ratio[i] == 0:
            continue
        a1[i] = float(a1[i] + 1) / 2
    # for i in range(5):
    #     sum = 0
    #     for j in range(mec_list_num[i]):
    #         sum += a1[mec_list[i][j]]
    #     for j in range(mec_list_num[i]):
    #         a1[mec_list[i][j]] /= sum
    return a1

def action_limit21(a):
    a1 = np.zeros(100)
    for i in range (0,100):
        a1[i] = a[0][i]
    return a1

def action_limit22(a1,a,mec_list_num,mec_list):
    a2 = a
    for i in range(100):
#        a2[i] = float(a2[i] - np.min(a2)) / (np.max(a2) - np.min(a2))
         a2[i] = (a2[i] + 1) / 2
    for i in range (MEC_STATE.M):
        if MEC_STATE.remain_ratio[i] != 0 or a1[i] == 0:
            if a2[i] < 0.1:
               a2[i] = 0.1
            if a2[i+50] < 0.1:
               a2[i+50] = 0.1
    for i in range(5):
        sum1 = 0
        sum2 = 0
        for j in range(mec_list_num[i]):
            if MEC_STATE.remain_ratio[mec_list[i][j]] == 0 :
                continue
            sum1 += a2[mec_list[i][j]]
            sum2 += a2[mec_list[i][j] + 50]
        for j in range(mec_list_num[i]):
            if MEC_STATE.remain_ratio[mec_list[i][j]] == 0:
                a2[mec_list[i][j]] = 0
                a2[mec_list[i][j] + 50] = 0
            if sum1 == 0:
                a2[mec_list[i][j]] = 0
            elif sum2 == 0:
                a2[mec_list[i][j] + 50] = 0
            else:
                a2[mec_list[i][j]] = a2[mec_list[i][j]]/sum1
                a2[mec_list[i][j] + 50] = a2[mec_list[i][j] + 50]/sum2
#    for i in range(100):
#        a2[i]=0.1
    return a2

def action_limit3(a1,a2):
    all_offloading = MEC_STATE.task_cpu_size / MEC_STATE.ue_com
    for i in range(50):
        if MEC_STATE.offloading_time[i] > all_offloading[i]:
            a1[i] = 0
            a2[i] = 0
            a2[i+50] = 0
    for i in range(0, 50):
        if a2[i] == 0:
            a1[i] = 0
    for i in range(50, 100):
        if a2[i] == 0:
            a1[i-50] = 0

#    for i in range(50):
#        a1[i] = 0.5
    return a1,a2
###############################  training  ####################################
np.random.seed(1)
tf.compat.v1.set_random_seed(1)

env = MEC_STATE()
MAX_EP_STEPS = env.slot_num
s_dim = env.state_dim
S_dim = env.STATE_dim
a1_dim = env.action1_dim
a2_dim = env.action2_dim
a_bound = env.action_bound  # [-1,1]
mec_list = np.zeros(env.M)

ddpg = DDPG(a1_dim, a2_dim, s_dim, S_dim, a_bound)
var = 1  # control exploration
#var = 0.01  # control exploration
t1 = time.time()
ep_reward_list = []
s_normal1 = StateNormalization1()
s_normal2 = StateNormalization2()
#print(env.sum_store_size)
#s1 = env.reset_env()

for i in range(MAX_EPISODES):
#    S_mark = copy.deepcopy(s1)
#    print("Smark:",S_mark)
    S_mark = env.reset_env()
    local_time = np.zeros(env.M)
    local_time_next = np.zeros(env.M)
    ep_reward1 = 0
    ep_reward2 = 0
    ep_reward = 0
#    S_mark = s
    j = 0
    while j < MAX_EP_STEPS:
        s = S_mark
        #print("s:",s_normal1.state_normal(s))
        # Add exploration noise
        #print("s:",s_normal1.state_normal(s))
        a1 = ddpg.choose_action1(s_normal1.state_normal(s))
        #print("a1:", a1)
        a1 = action_limit11(a1)
        #print("a1:", a1)
        a12 = action_limit12(a1)
        #print("a12:",a12)
        # a = np.clip(np.random.normal(a, var), *a_bound)  # 高斯噪声add randomness to action selection for exploration
        # s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist,STEP,Offloading_ratio,a[0],a[1],a[2],a[3],sumtask = env.step1(a)
        S = env.step1(a12,j)
        #print("a1",a1)
        # if step_redo:
        #     continue
        # if reset_dist:
        #     a[2] = -1
        # if offloading_ratio_change:
        #     a[3] = -1
        a2 = ddpg.choose_action2(s_normal2.state_normal(S))
#        print("a2:",a2)
        a2 = action_limit21(a2)
        #print("a2:",a2)
        a22 = action_limit22(a1,a2,env.mec_list_num,env.mec_list)
        print("a22:",a22)
        a12,a22 = action_limit3(a12,a22)
        s_, r1, r2, STEP, sumtask, is_terminal = env.step2(a22,j)
        a1_ = ddpg.choose_action1(s_normal1.state_normal(s_))
        a1_ = action_limit11(a1_)
        a11_ = action_limit12(a1_)
#        a1_ = action_limit1(a1_, env.mec_list_num, env.mec_list)
        S_ = env.step1(a11_,j)
        ddpg.store_transition(s_normal1.state_normal(s), a1, r1, s_normal2.state_normal(S), a2, r2, s_normal1.state_normal(s_), s_normal2.state_normal(S_))  # 训练奖励缩小10倍

        if ddpg.pointer > MEMORY_CAPACITY:
            var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
            ddpg.learn()
        s = s_
        ep_reward1 += r1
        ep_reward2 += r2
        ep_reward = ep_reward1 + ep_reward2
        # if i == 0:
        #     ep_reward=0
        print('reward:', ep_reward,ep_reward1,ep_reward2,' r:',r1,r2,' IS_T',is_terminal,' STEP',STEP,' 卸载率',a12,'任务总量：',sumtask)
        print("--------------------------------------------------------------------------------")
        if j == MAX_EP_STEPS - 1 or is_terminal :
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % var,'J:',j)
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
            file_name = 'output.txt'
            with open(file_name, 'a') as file_obj:
                file_obj.write("\n======== This episode is done ========")  # 本episode结束
            break
        j = j + 1
    finial_reward = ep_reward
    print("MAX_EPISODES",MAX_EPISODES,"finial_reward",finial_reward)

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)

print('Running time: ', time.time() - t1)
# plt.plot(ep_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.show()
plt.plot(ep_reward_list)
# plt.xlabel("MAX_EPISODES")
# plt.ylabel("Reward")
plt.xlabel("MAX_EPISODES")
#plt.xlim(1,20)
plt.ylabel("Reward")
plt.show()