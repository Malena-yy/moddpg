import os
import time
from env import UAV
from ddpg import AGENT
import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='Evaluate the trained model.')
parser.add_argument('--is_train', type=int, default=1, metavar='train(1) or eval(0)',
                    help='train model of evaluate the trained model')

# TRAINING
parser.add_argument('--gamma', type=float, default=0.9, metavar='discount rate',
                    help='The discount rate of long-term returns')
parser.add_argument('--mem_size', type=int, default=8000, metavar='memorize size',
                    help='max size of the replay memory')
parser.add_argument('--batch_size', type=int, default=64, metavar='batch size',
                    help='batch size')
parser.add_argument('--lr_actor', type=float, default=0.001, metavar='learning rate of actor',
                    help='learning rate of actor network')
parser.add_argument('--lr_critic', type=float, default=0.001, metavar='learning rate of critic',
                    help='learning rate of critic network')
parser.add_argument('--replace_tau', type=float, default=0.001, metavar='replace_tau',
                    help='soft replace_tau')
parser.add_argument('--episode_num', type=int, default=301, metavar='episode number',
                    help='number of episodes for training')
parser.add_argument('--Num_episode_plot', type=int, default=10, metavar='plot freq',
                    help='frequent of episodes to plot')
parser.add_argument('--save_model_freq', type=int, default=100, metavar='save freq',
                    help='frequent to save network parameters')
parser.add_argument('--R_dc', type=float, default=10., metavar='R_DC',
                    help='the radius of data collection')
parser.add_argument('--R_eh', type=float, default=30., metavar='R_EH',
                    help='the radius of energy harvesting')
parser.add_argument('--w_dc', type=float, default=100., metavar='W_DC',
                    help='the weight of data collection')
parser.add_argument('--w_eh', type=float, default=100., metavar='W_EH',
                    help='the weight of energy harvesting')
parser.add_argument('--w_ec', type=float, default=1., metavar='W_EC',
                    help='the weight of energy consumption')

# evaluation
parser.add_argument('--eval_num', type=int, default=20, metavar='EN',
                    help='number of episodes for evaluation')
parser.add_argument('--model', type=str, default='P_moddpg', metavar='model path',
                    help='the path of the trained model')
parser.add_argument('--save_path', type=str, default='eval_result', metavar='save path',
                    help='the save path of the evaluation result')
args = parser.parse_args()

#####################  set the save path  ####################
logs_path = '/{}/{}/'.format(args.save_path, 'logs')
path = os.getcwd() + logs_path
if not os.path.exists(path):
    os.makedirs(path)
figs_path = '/{}/{}/'.format(args.save_path, 'figs')
path = os.getcwd() + figs_path
if not os.path.exists(path):
    os.makedirs(path)

# 设置画图横纵坐标字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }

Q = 10.
V_me = 10.0
V_max = 20.0
EC_min = 126.
EC_hov = 168.49  # 悬停能耗
EC_grd = 178.


def evaluation(i):
    policy = comp_policies[i]
    load_ep = eps[i]
    if policy.find('P_Vmax') == -1 and policy.find('P_V_ME') == -1:
        ddpg = AGENT(args, a_num, a_dim, s_dim, a_bound, False)
        ddpg.load_ckpt(args.model, load_ep)
    file = open(os.path.join('.{}{}.txt'.format(logs_path, str(policy))), 'w+')
    file.write(
        'Episode|data rate|Harvasted energy|fly energy consumption|Total number of EH user|sum rate|Total number of ID user|Average harvasted energy|Average number of EH user' + '\n')
    file.flush()
    ep = 0
    while ep < args.eval_num:
        s = env.reset()
        idu = 0
        N_DO = 0  # 每EPISODES总溢出数据用户数
        DQ = 0
        FX = 0
        sum_rate = 0
        ehu = 0
        Eh = 0
        Ec = 0
        action = np.asarray([0., 0.])
        Ht = 0
        Ft = 0
        while True:
            # env.render()
            if policy.find('P_Vmax') != -1:
                above = True
                if s[0] < 0:
                    action[0] = max(s[0] * V_max / math.sqrt(s[0] ** 2 + s[1] ** 2), s[0] * 400) / env.max_speed
                else:
                    action[0] = min(s[0] * V_max / math.sqrt(s[0] ** 2 + s[1] ** 2), s[0] * 400) / env.max_speed
                if s[1] < 0:
                    action[1] = max(s[1] * V_max / math.sqrt(s[0] ** 2 + s[1] ** 2), s[1] * 400) / env.max_speed
                else:
                    action[1] = min(s[1] * V_max / math.sqrt(s[0] ** 2 + s[1] ** 2), s[1] * 400) / env.max_speed
                s_, r, done, dr, cu, eh, ec = env.step_move(action, above)
                ft = math.sqrt((action[0]*env.max_speed) ** 2 + (action[1]*env.max_speed) ** 2) / V_max   # 飞行时间
                ec_fly = EC_grd * ft  # 飞行能耗
            elif policy.find('P_V_ME') != -1:
                above = True
                if s[0] < 0:
                    action[0] = max(s[0] * V_me / math.sqrt(s[0] ** 2 + s[1] ** 2), s[0] * 400) / env.max_speed
                else:
                    action[0] = min(s[0] * V_me / math.sqrt(s[0] ** 2 + s[1] ** 2), s[0] * 400) / env.max_speed
                if s[1] < 0:
                    action[1] = max(s[1] * V_me / math.sqrt(s[0] ** 2 + s[1] ** 2), s[1] * 400) / env.max_speed
                else:
                    action[1] = min(s[1] * V_me / math.sqrt(s[0] ** 2 + s[1] ** 2), s[1] * 400) / env.max_speed
                ft = math.sqrt((action[0]*env.max_speed) ** 2 + (action[1]*env.max_speed) ** 2) / V_me  # 飞行时间
                s_, r, done, dr, cu, eh, ec = env.step_move(action, above)
                ec_fly = EC_min * ft  # 飞行能耗
            else:
                action = ddpg.choose_action(s)
                ft = 1.
                s_, r, done, dr, cu, eh, ec = env.step_move(action)

            Ft += ft
            N_DO += env.N_Data_overflow
            DQ += sum(env.b_S / env.Fully_buffer)
            FX += env.FX
            if policy.find('P_Vmax') != -1 or policy.find('P_V_ME') != -1:
                ec = ec_fly
            Ec += ec

            if done:  # 悬停收集数据
                idu += 1  # 服务用户计数
                ht = Q * env.updata / dr  # 计算悬停时间
                sum_rate += dr
                ehu += cu
                Eh += eh * ht
                Ht += ht
                Ec += EC_hov * ht
                env.step_hover(ht)
                N_DO += env.N_Data_overflow
                DQ += sum(env.b_S / env.Fully_buffer)
                s = env.CHOOSE_AIM()
            else:
                s = s_

            if Ht+Ft >= 600:
                N_DO /= (Ht+Ft)  # 平均溢出数据用户数
                DQ /= (Ht+Ft)
                DQ /= env.N_POI  # 平均用户数据缓存量
                FX /= Ft
                Ec /= (Ht+Ft)  # 平均每步消耗能量
                if idu:
                    aEh = Eh / idu
                else:
                    aEh = 0

                ep += 1
                plot_x.append(ep)
                plot_N_DO.append(N_DO)  # 数据溢出用户计数
                plot_DQ.append(DQ)  # 平均用户数据缓存量
                plot_sr.append(sum_rate)  # 回合总悬停吞吐量
                plot_Eh.append(Eh)  # 回合总收集能量
                plot_ehu.append(ehu)  # 回合总充电用户
                plot_idu.append(idu)  # 回合总收集数据用户
                plot_Ec.append(Ec)  # 回合每步平均能耗
                plot_HT.append(Ht)  # 悬停时间
                plot_FT.append(Ft)  # 飞行时间
                # 实时输出训练数据
                print(
                    'Episode:%i |sum rate:%.3f |idu:%i |ehu:%i |total EH:%.2f |avg EH:%.2f |energy_coums:%.2f |N_Data_overflow:%.2f' % (
                        ep, sum_rate, idu, ehu, Eh, aEh, Ec, N_DO))
                # 将相关数据写入文档
                write_str = '%i|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.2f|%i|%i\n' % (
                    ep, DQ, sum_rate, Eh, aEh, Ec, N_DO, FX, ehu, idu)
                file.write(write_str)
                file.flush()
                file.close
                break


if __name__ == '__main__':
    comp_policies = ['P_moddpg', 'P_Vmax', 'P_V_ME']
    Label = [r'$P_{moddpg}$', r'$P_{V_{\max}}$', r'$P_{V_{ME}}$']
    eps = [1200, 0, 0]

    t1 = time.time()
    plot_x_avg = []
    plot_sr_avg = []  # 回合收集数据量
    plot_r_avg = []  # 每次悬停平均数据率
    plot_Eh_avg = []  # 总收集能量
    plot_avg_Eh_avg = []  # 平均收集能量
    plot_avg_Eh_rate = []  # 平均收集能量速率
    plot_Ec_avg = []  # 平均能耗
    plot_idu_avg = []  # 上传数据用户数
    plot_ehu_avg = []  # 总充电用户数
    plot_avg_ehu_avg = []  # 平均每次悬停充电用户数
    plot_DQ_avg = []  # 平均用户数据缓存量
    plot_N_DO_avg = []  # 数据溢出用户计数
    plot_HT_avg = []  # 悬停时间
    plot_FT_avg = []  # 飞行时间
    for i in range(len(comp_policies)):
        print(comp_policies[i])
        np.random.seed(1)
        env = UAV()
        # 定义状态空间，动作空间，动作幅度范围
        s_dim = env.state_dim
        a_num = 2
        a_dim = env.action_space.shape[0]
        a_bound = env.action_space

        plot_x = []
        plot_N_DO = []  # 数据溢出用户计数
        plot_DQ = []  # 平均用户数据缓存量
        plot_sr = []  # 每回合总悬停吞吐量
        plot_Eh = []  # 每回合总收集能量
        plot_Ec = []  # 平均每步能耗
        plot_ehu = []  # 回合总充电用户
        plot_idu = []  # 回合总收集数据用户
        plot_HT = []  # 悬停时间
        plot_FT = []  # 飞行时间

        evaluation(i)

        avg_ehu = [plot_ehu[i] / plot_idu[i] if plot_idu[i] != 0 else 0 for i in range(len(plot_ehu))]  # 每次悬停平均充电用户数
        avg_eh = [plot_Eh[i] / plot_idu[i] if plot_idu[i] != 0 else 0 for i in range(len(plot_Eh))]  # 每次悬停平均充电量
        eh_rate = [plot_Eh[i] / plot_HT[i] if plot_idu[i] != 0 else 0 for i in range(len(plot_Eh))]  # 平均收集能量速率
        plot_r = [plot_sr[i] / plot_idu[i] if plot_idu[i] != 0 else 0 for i in range(len(plot_sr))]  # 每次悬停平均数据率

        plot_x_avg.append(plot_x)
        plot_sr_avg.append(plot_sr)  # 回合收集数据量
        plot_r_avg.append(plot_r)  # 每次悬停平均数据率
        plot_Eh_avg.append(plot_Eh)  # 总收集能量
        plot_avg_Eh_avg.append(avg_eh)  # 平均收集能量 --除以悬停次数
        plot_avg_Eh_rate.append(eh_rate)  # 平均收集能量速率--除以悬停时间
        plot_Ec_avg.append(plot_Ec)  # 平均能耗
        plot_idu_avg.append(plot_idu)  # 上传数据用户数
        plot_ehu_avg.append(plot_ehu)  # 总充电用户数
        plot_avg_ehu_avg.append(avg_ehu)  # 平均每次悬停充电用户数
        plot_DQ_avg.append(plot_DQ)  # 平均用户数据缓存量
        plot_N_DO_avg.append(plot_N_DO)  # 数据溢出用户计数
        plot_HT_avg.append(plot_HT)  # 悬停时间
        plot_FT_avg.append(plot_FT)  # 飞行时间
    '''
        # 画图
        1、累积奖励Accumulated reward，2、回合收集数据量 sum rate 3、 平均每次悬停收集数据量 data rate
        4、 回合总收集能量harvested energy  5、平均每次悬停收集能量Average harvested energy
        6、回合平均每步飞行能耗 fly energy consumption  7、上传数据用户数 The number of ID user 
        8、总充电用户数 The number of EH user 9、平均每次悬停充电用户数 Average number of EH user 
        10、系统平均数据水平 Average data buffer length 11、 数据溢出用户数 N_d
    '''
    # Fig 1:Average data rate(sum_rate/idu)/Total harvested energy(Ec)/Average flying energy consumption(Ec/Ft)
    # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
    ############################################data_rate/harvested energy########################
    ####################################fly energy consumption/total number of EH user########
    p1 = plt.figure(figsize=(28, 14))  # 第一幅子图,并确定画布大小

    ax1 = p1.add_subplot(2, 4, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')
    ax2 = p1.add_subplot(2, 4, 2)
    ax2.tick_params(labelsize=12)
    ax2.grid(linestyle='-.')
    ax3 = p1.add_subplot(2, 4, 3)
    ax3.tick_params(labelsize=12)
    ax3.grid(linestyle='-.')
    ax4 = p1.add_subplot(2, 4, 4)
    ax4.tick_params(labelsize=12)
    ax4.grid(linestyle='-.')
    ax5 = p1.add_subplot(2, 4, 5)
    ax5.tick_params(labelsize=12)
    ax5.grid(linestyle='-.')
    ax6 = p1.add_subplot(2, 4, 6)
    ax6.tick_params(labelsize=12)
    ax6.grid(linestyle='-.')
    ax7 = p1.add_subplot(2, 4, 7)
    ax7.tick_params(labelsize=12)
    ax7.grid(linestyle='-.')
    ax8 = p1.add_subplot(2, 4, 8)
    ax8.tick_params(labelsize=12)
    ax8.grid(linestyle='-.')

    for i in range(len(plot_x_avg)):
        ax1.plot(plot_x_avg[i], plot_r_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_r_avg[i])))
    ax1.set_xlabel('Number of evaluation episodes', font1)
    ax1.set_ylabel('data rate (bps/Hz)', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]
    ax1.legend(prop=font2)

    for i in range(len(plot_x_avg)):
        ax2.plot(plot_x_avg[i], plot_Eh_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_Eh_avg[i])))
    ax2.set_xlabel('Number of evaluation episodes', font1)
    ax2.set_ylabel(r'Harvested energy ($\mu$W)', font1)

    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]
    ax2.legend(prop=font2)

    for i in range(len(plot_x_avg)):
        ax3.plot(plot_x_avg[i], plot_Ec_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_Ec_avg[i])))
    ax3.set_xlabel('Number of evaluation episodes', font1)
    ax3.set_ylabel('Average fly energy consumption (W)', font1)

    label3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label3]
    ax3.legend(prop=font2)

    for i in range(len(plot_x_avg)):
        ax4.plot(plot_x_avg[i], plot_ehu_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_ehu_avg[i])))
    ax4.set_xlabel('Number of evaluation episodes', font1)
    ax4.set_ylabel('Total number of EH user', font1)

    label4 = ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label4]
    ax4.legend(prop=font2)

    for i in range(len(plot_x_avg)):
        ax5.plot(plot_x_avg[i], plot_sr_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_sr_avg[i])))
    ax5.set_xlabel('Number of evaluation episodes', font1)
    ax5.set_ylabel('sum rate (bps/Hz)', font1)

    label5 = ax5.get_xticklabels() + ax5.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label5]
    ax5.legend(prop=font2)

    for i in range(len(plot_x_avg)):
        ax6.plot(plot_x_avg[i], plot_idu_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_idu_avg[i])))
    ax6.set_xlabel('Number of evaluation episodes', font1)
    ax6.set_ylabel('Total number of ID user', font1)

    label6 = ax6.get_xticklabels() + ax6.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label6]
    ax6.legend(prop=font2)

    for i in range(len(plot_x_avg)):
        ax7.plot(plot_x_avg[i], plot_avg_Eh_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_avg_Eh_avg[i])))
    ax7.set_xlabel('Number of evaluation episodes', font1)
    ax7.set_ylabel(r'Average harvested energy ($\mu$W)', font1)

    label7 = ax7.get_xticklabels() + ax7.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label7]
    ax7.legend(prop=font2)

    for i in range(len(plot_x_avg)):
        ax8.plot(plot_x_avg[i], plot_avg_ehu_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_avg_ehu_avg[i])))
    ax8.set_xlabel('Number of evaluation episodes', font1)
    ax8.set_ylabel('Average number of EH user', font1)

    label8 = ax8.get_xticklabels() + ax8.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label8]
    ax8.legend(prop=font2)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}sum_up.jpg'.format(figs_path))
    plt.clf()
    ############################################10、系统平均数据水平 Average data buffer length#####################################################
    # 画图
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.tick_params(labelsize=12)
    ax2.grid(linestyle='-.')

    for i in range(len(plot_x_avg)):
        ax1.plot(plot_x_avg[i], plot_DQ_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_DQ_avg[i])))
    ax1.set_xlabel('Number of evaluation episodes', font1)
    ax1.set_ylabel('Average data buffer length (%)', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]
    ax1.legend(prop=font2)

    for i in range(len(plot_x_avg)):
        ax2.plot(plot_x_avg[i], plot_N_DO_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_N_DO_avg[i])))
    ax2.set_xlabel('Number of evaluation episodes', font1)
    ax2.set_ylabel(r'$N_d^{AVG}$', font1)

    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]
    ax2.legend(prop=font2)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}system_performance.jpg'.format(figs_path))
    plt.clf()
    ##################################飞行时间 悬停时间#########################################
    # 画图
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.tick_params(labelsize=12)
    ax2.grid(linestyle='-.')

    for i in range(len(plot_x_avg)):
        ax1.plot(plot_x_avg[i], plot_HT_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_HT_avg[i])))
    ax1.set_xlabel('Number of evaluation episodes', font1)
    ax1.set_ylabel('hovering time (s)', font1)
    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]
    ax1.legend(prop=font2)

    for i in range(len(plot_x_avg)):
        ax2.plot(plot_x_avg[i], plot_FT_avg[i], marker='*', markersize='10', linewidth='2',
                 label='{}:{:.2f}'.format(str(Label[i]), np.mean(plot_FT_avg[i])))
    ax2.set_xlabel('Number of evaluation episodes', font1)
    ax2.set_ylabel('flying time (s)', font1)
    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]
    ax2.legend(prop=font2)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}time.jpg'.format(figs_path))
    plt.clf()
    #########################################################################
    now_time = datetime.datetime.now()
    date = now_time.strftime('%Y-%m-%d %H_%M_%S')
    print('Running time: ', time.time() - t1)
