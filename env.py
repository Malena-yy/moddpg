import sys
import numpy as np
import time
import math
import copy
from gym import spaces
from gym.utils import seeding

#   以v in [0,1],theta in [-1,1]为动作
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
'''
地图大小为400*400
无人机数量N_UAV=1
服务点数量为100，所有服务点在限定区域内随机分布，服务点具有不同数据收集特性
固定地图
'''
WIDTH = 10  # 地图的宽度
HEIGHT = 10  # 地图的高度
UNIT = 40  # 每个方块的大小（像素值）   400m*400m的地图
LDA = [4., 8., 15., 20.]  # 假设有4类传感器，即有4个不同的泊松参数,传感器数据生成服从泊松分布
max_LDA = max(LDA)
C = 5000  # 传感器的容量都假设为5000
P_u = pow(10, -5)  # 传感器的发射功率 0.01mW,-20dbm
P_d = 10  # 无人机下行发射功率 10W,40dBm
H = 10.  # 无人机固定悬停高度 10m
R_d = 30.  # 无人机充电覆盖范围 10m能接受到0.1mW,30m能接收到0.01mW
N_S_ = 100  # 设备个数
V = 20  # 无人机最大速度 20m/s
b_S_ = np.random.randint(0, 500, N_S_)  # 初始化传感器当前数据缓存量

# 非线性能量接收机模型
Mj = 9.079 * pow(10, -6)
aj = 47083
bj = 2.9 * pow(10, -6)
Oj = 1 / (1 + math.exp(aj * bj))

np.random.seed(1)


# 定义无人机类
class UAV(tk.Tk, object):
    def __init__(self, R_dc=10., R_eh=30.):
        super(UAV, self).__init__()
        # POI位置
        self.N_POI = N_S_  # 传感器数量
        self.dis = np.zeros(self.N_POI)  # 距离的平方
        self.elevation = np.zeros(self.N_POI)  # 仰角
        self.pro = np.zeros(self.N_POI)  # 视距概率
        self.h = np.zeros(self.N_POI)  # 信道增益
        self.N_UAV = 1
        self.max_speed = V  # 无人机最大速度 20m/s
        self.H = 10.  # 无人机飞行高度 10m
        self.X_min = 0
        self.Y_min = 0
        self.X_max = (WIDTH) * UNIT
        self.Y_max = (HEIGHT) * UNIT  # 地图边界
        self.R_dc = R_dc  # 水平覆盖距离 10m
        self.R_eh = R_eh  # 水平覆盖距离 30m
        self.sdc = math.sqrt(pow(self.R_dc, 2) + pow(self.H, 2))  # 最大DC服务距离
        self.seh = math.sqrt(pow(self.R_eh, 2) + pow(self.H, 2))  # 最大EH服务距离
        self.noise = pow(10, -12)  # 噪声功率为-90dbm
        self.AutoUAV = []
        self.Aim = []
        self.N_AIM = 1  # 选择服务的用户数
        self.FX = 0.
        self.SoPcenter = np.random.randint(10, 390, size=[self.N_POI, 2])
        #   以 v in [0,1],theta in [-1,1]为动作
        self.action_space = spaces.Box(low=np.array([0., -1.]), high=np.array([1., 1.]),
                                       dtype=np.float32)
        self.state_dim = 6  # 状态空间为最高优先级用户位置与无人机的相对位置，无人机位置，是否撞墙，数据溢出数
        self.state = np.zeros(self.state_dim)
        self.xy = np.zeros((self.N_UAV, 2))  # 无人机位置

        # 假设有4类传感器，即有4个不同的泊松参数,随机给传感器分配泊松参数
        CoLDA = np.random.randint(0, len(LDA), self.N_POI)
        self.lda = [LDA[CoLDA[i]] for i in range(self.N_POI)]  # 给传感器们指定数据增长速度
        self.b_S = np.random.randint(0., 500., self.N_POI).astype(np.float32)  # 初始化传感器当前数据缓存量
        self.Fully_buffer = C
        self.N_Data_overflow = 0  # 数据溢出计数
        self.Q = np.array(
            [self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)])  # 数据收集优先级
        self.idx_target = np.argmax(self.Q)
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        '''
        # 指定一块区域环境对传感器有影响
        for i in range(self.N_POI):
            if all(self.SoPcenter[i] >= [120, 120]) and all(self.SoPcenter[i] <= [280, 280]):
                self.lda[i] += 3.
        '''

        self.title('MAP')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))  # Tkinter 的几何形状
        self.build_maze()

    # 创建地图
    def build_maze(self):
        # 创建画布 Canvas.白色背景，宽高。
        self.canvas = tk.Canvas(self, bg='white', width=WIDTH * UNIT, height=HEIGHT * UNIT)

        '''
        # 标记出特殊区域
        for c in range(120, 280, UNIT * 4 - 1):
            x0, y0, x1, y1 = c, 120, c, 280
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(120, 280, UNIT * 4 - 1):
            x0, y0, x1, y1 = 120, r, 280, r
            self.canvas.create_line(x0, y0, x1, y1)
        '''
        # 创建用户
        for i in range(self.N_POI):
            # 创建椭圆，指定起始位置。填充颜色
            if self.lda[i] == LDA[0]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='pink')
            elif self.lda[i] == LDA[1]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='blue')
            elif self.lda[i] == LDA[2]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='green')
            elif self.lda[i] == LDA[3]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='red')

        # 创建无人机
        self.xy = np.random.randint(100., 300., size=[self.N_UAV, 2])

        for i in range(self.N_UAV):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - R_d, self.xy[i][1] - R_d,
                self.xy[i][0] + R_d, self.xy[i][1] + R_d,
                fill='yellow')
            self.AutoUAV.append(L_UAV)

        # 用户选择
        pxy = self.SoPcenter[np.argmax(self.Q)]
        L_AIM = self.canvas.create_rectangle(
            pxy[0] - 10, pxy[1] - 10,
            pxy[0] + 10, pxy[1] + 10,
            fill='red')
        self.Aim.append(L_AIM)

        self.canvas.pack()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # 重置，随机初始化无人机的位置
    def reset(self):
        self.render()
        for i in range(self.N_UAV):
            self.canvas.delete(self.AutoUAV[i])
        self.AutoUAV = []

        for i in range(len(self.Aim)):
            self.canvas.delete(self.Aim[i])

        # 随机初始化无人机位置
        self.xy = np.random.randint(100, 300, size=[self.N_UAV, 2]).astype(float)
        for i in range(self.N_UAV):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - R_d, self.xy[i][1] - R_d,
                self.xy[i][0] + R_d, self.xy[i][1] + R_d,
                fill='yellow')
            self.AutoUAV.append(L_UAV)
        self.FX = 0.

        self.b_S = np.random.randint(0, 500, self.N_POI)  # 初始化传感器当前数据缓存量
        self.b_S = np.asarray(self.b_S, dtype=np.float32)
        self.N_Data_overflow = 0  # 数据溢出计数
        self.Q = np.array([self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)])  # 数据收集优先级

        # 初始化状态空间值
        self.idx_target = np.argmax(self.Q)
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.pxy = self.SoPcenter[self.idx_target]  # 初始选择优先级最大的

        L_AIM = self.canvas.create_rectangle(
            self.pxy[0] - 10, self.pxy[1] - 10,
            self.pxy[0] + 10, self.pxy[1] + 10,
            fill='red')
        self.Aim.append(L_AIM)

        self.state = np.concatenate(((self.pxy - self.xy[0]).flatten() / 400., self.xy.flatten() / 400., [0., 0.]))

        return self.state

    # 传入当前状态和输入动作输出下一个状态和奖励
    def step_move(self, action, above=False):
        if above == True:
            detX = action[:self.N_UAV] * self.max_speed
            detY = action[self.N_UAV:] * self.max_speed
        else:
            detX = action[0] * self.max_speed * math.cos(action[1] * math.pi)
            detY = action[0] * self.max_speed * math.sin(action[1] * math.pi)
        state_ = np.zeros(self.state_dim)
        xy_ = copy.deepcopy(self.xy)  # 位置更新
        Flag = False  # 无人机是否飞行标识
        for i in range(self.N_UAV):  # 无人机位置更新
            xy_[i][0] += detX
            xy_[i][1] += detY
            # 当无人机更新后的位置超出地图范围时
            if xy_[i][0] >= self.X_min and xy_[i][0] <= self.X_max:
                if xy_[i][1] >= self.Y_min and xy_[i][1] <= self.Y_max:
                    self.FX = 0.
                    Flag = True
                else:
                    xy_[i][0] -= detX
                    xy_[i][1] -= detY
                    self.FX += 1.
            else:
                xy_[i][0] -= detX
                xy_[i][1] -= detY
                self.FX += 1.
        if Flag:
            # 飞行能耗
            V = math.sqrt(pow(detX, 2) + pow(detY, 2))
            ec = 79.86 * (1 + 0.000208 * pow(V, 2)) + 88.63 * math.sqrt(
                math.sqrt(1 + pow(V, 4) / 1055.0673312400002) - pow(V, 2) / 32.4818) + 0.009242625 * pow(V, 3)
        else:
            ec = 168.49  # 悬停能耗

        for i in range(self.N_UAV):
            self.canvas.move(self.AutoUAV[i], xy_[i][0] - self.xy[i][0], xy_[i][1] - self.xy[i][1])

        self.xy = xy_
        # 无人机位置更新后，判断服务点接受服务的情况
        self.N_Data_overflow = 0  # 记录每时隙数据溢出用户数
        self.b_S += [np.random.poisson(self.lda[i]) for i in range(self.N_POI)]  # 传感器数据缓存量更新
        for i in range(self.N_POI):  # 数据溢出处理
            if self.b_S[i] >= self.Fully_buffer:
                self.N_Data_overflow += 1  # 数据溢出用户计数
                self.b_S[i] = self.Fully_buffer
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer

        # 状态空间归一化
        state_[:2] = (self.pxy - xy_).flatten() / 400.  # 更新用户与无人机相对位置
        state_[2:4] = xy_.flatten() / 400.  # 无人机绝对位置
        state_[4] = self.FX / 400.  # 无人机越境次数/总步数
        state_[5] = self.N_Data_overflow / self.N_POI  # 数据溢出用户占比

        Done = False

        # 奖励的定义——尽快到目的地/不要撞墙/减小能耗
        reward = -(abs(state_[0]) + abs(state_[1])) * 100 - self.FX * 10 - self.N_Data_overflow * 5
        self.Q_dis()  # 获取所有用户与无人机的信道增益
        ehu = 0  # 充电覆盖用户
        data_rate = 0  # 数据率
        eh = 0  # 总充电量
        if (above == False and self.dis[self.idx_target] <= self.sdc) or (
                above == True and abs(state_[0]) <= 0.002 and abs(state_[1]) <= 0.002):
            Done = True
            reward += 500
            # 只给目标用户收集数据
            data_rate = math.log(1 + P_u * self.h[self.idx_target] / self.noise, 2)  # 2.397~4.615
            self.b_S[self.idx_target] = 0
            for i in range(self.N_POI):
                if self.dis[i] <= self.seh and i != self.idx_target:
                    ehu += 1
                    eh += self.Non_linear_EH(P_d * self.h[i])  # 输入是10-4W~10-5W,输出是0.6751969599046135~7.418403066937866
        # print(sum_rate,ehu,eh)
        self.state = state_

        return state_, reward, Done, data_rate, ehu, eh, ec  # 状态值，奖励，是否到达目标，总数据率, 覆盖到的用户，收集能量，无人机能耗

    def step_hover(self, hover_time):
        # 无人机不动，所以是s[:5]不变
        self.N_Data_overflow = 0  # 记录每时隙数据溢出用户数
        self.b_S += [np.random.poisson(self.lda[i]) * hover_time for i in range(self.N_POI)]  # 传感器数据缓存量更新
        for i in range(self.N_POI):  # 数据溢出处理
            if self.b_S[i] >= self.Fully_buffer:
                self.N_Data_overflow += 1  # 数据溢出用户计数
                self.b_S[i] = self.Fully_buffer
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.state[5] = self.N_Data_overflow / self.N_POI  # 数据溢出用户占比

    # 每次无人机更新位置后，计算无人机与所有用户的距离与仰角，以及路径增益
    def Q_dis(self):
        for i in range(self.N_POI):
            self.dis[i] = math.sqrt(
                pow(self.SoPcenter[i][0] - self.xy[0][0], 2) + pow(self.SoPcenter[i][1] - self.xy[0][1], 2) + pow(
                    self.H, 2))  # 原始距离
            self.elevation[i] = 180 / math.pi * np.arcsin(self.H / self.dis[i])  # 仰角
            self.pro[i] = 1 / (1 + 10 * math.exp(-0.6 * (self.elevation[i] - 10)))  # 视距概率
            self.h[i] = (self.pro[i] + (1 - self.pro[i]) * 0.2) * pow(self.dis[i], -2.3) * pow(10,
                                                                                               -30 / 10)  # 参考距离增益为-30db

    # 输入是10-4W~10-5W,输出是0~9.079muW
    def Non_linear_EH(self, Pr):
        if Pr == 0:
            return 0
        P_prac = Mj / (1 + math.exp(-aj * (Pr - bj)))
        Peh = (P_prac - Mj * Oj) / (1 - Oj)  # 以W为单位
        return Peh * pow(10, 6)

    # 输入是10-4W~10-5W,输出是0~9.079muW
    def linear_EH(self, Pr):
        if Pr == 0:
            return 0
        return Pr * pow(10, 6) * 0.2

    # 重选目标用户
    def CHOOSE_AIM(self):
        for i in range(len(self.Aim)):
            self.canvas.delete(self.Aim[i])

        # 重选目标用户
        self.Q = np.array([self.lda[i] * self.b_S[i] / C for i in range(self.N_POI)])  # 数据收集优先级
        self.idx_target = np.argmax(self.Q)
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.pxy = self.SoPcenter[self.idx_target]
        L_AIM = self.canvas.create_rectangle(
            self.pxy[0] - 10, self.pxy[1] - 10,
            self.pxy[0] + 10, self.pxy[1] + 10,
            fill='red')
        self.Aim.append(L_AIM)

        self.state[:2] = (self.pxy - self.xy[0]).flatten() / 400.
        self.render()
        return self.state

    # 调用Tkinter的update方法, 0.01秒去走一步。
    def render(self, t=0.01):
        time.sleep(t)
        self.update()

    def sample_action(self):
        v = np.random.rand()
        theta = -1 + 2 * np.random.rand()
        return [v, theta]


def update():
    for t in range(10):
        env.reset()
        while True:
            env.render()
            paras = env.sample_action()
            s, r, done, sum_rate, cover_u, eh, ec = env.step_move(paras)
            if done:
                break


if __name__ == '__main__':
    env = UAV()
    env.after(10, update)
    env.mainloop()
