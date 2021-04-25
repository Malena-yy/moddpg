import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl

###############################  DDPG  ####################################

class AGENT(object):
    """
    DDPG class
    """

    def __init__(self, args, a_num, a_dim, s_dim, a_bound, is_train):
        self.memory = np.zeros((args.mem_size, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.replace_tau = args.replace_tau
        self.a_num = a_num
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound.high
        self.is_train = is_train
        self.gamma = args.gamma

        w_init = tf.initializers.he_normal()
        b_init = tf.constant_initializer(0.001)

        # 建立actor网络，输入s, 输出a
        def get_actor(input_state, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            s = tl.layers.Input(input_state, name='A_s_input')
            x = tl.layers.Dense(n_units=400, act=tf.nn.relu6, W_init=w_init, b_init=b_init, name='A_l1')(s)
            x = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, b_init=b_init, name='A_l2')(x)
            x_v = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, b_init=b_init, name='v_l3')(x)
            x_angle = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, b_init=b_init, name='angle_l3')(x)
            v = tl.layers.Dense(n_units=1, act=tf.nn.sigmoid, W_init=w_init, name='action_v')(x_v)
            angle = tl.layers.Dense(n_units=1, act=tf.nn.tanh, W_init=w_init, name='action_angle')(x_angle)
            output = tl.layers.Concat(1)([v, angle])

            return tl.models.Model(inputs=s, outputs=output, name='Actor' + name)

        # 建立Critic网络，输入s, a。输出Q值
        def get_critic(input_state, input_action_params, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state, name='C_s_input')
            a_paras = tl.layers.Input(input_action_params, name='C_a_paras_input')
            inputs = tl.layers.Concat(1)([s, a_paras])
            x = tl.layers.Dense(n_units=400, act=tf.nn.relu6, W_init=w_init, b_init=tf.constant_initializer(0.01), name='C_l1')(inputs)
            x = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, b_init=tf.constant_initializer(0.01), name='C_l2')(x)
            q = tl.layers.Dense(n_units=1, W_init=w_init, b_init=tf.constant_initializer(0.01), name='C_svalue')(x)
            return tl.models.Model(inputs=[s, a_paras], outputs=q, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        # 更新参数，只用于首次赋值，之后就没用了
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        # 建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        # 建立critic_target网络，并和critic参数一致，不能训练
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        # 建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - args.replace_tau)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(lr=args.lr_actor)
        self.critic_opt = tf.optimizers.Adam(lr=args.lr_critic)

    def ema_update(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights  # 获取要更新的参数包括actor和critic的
        self.ema.apply(paras)  # 主要是建立影子参数
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))  # 用滑动平均赋值

    def soft_replace(self):
        paras = self.actor.trainable_weights + self.critic.trainable_weights  # 获取要更新的参数包括actor和critic的
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign((1 - self.replace_tau) * i + self.replace_tau * j)

    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        s = np.array([s], dtype=np.float32)
        a = self.actor(s)
        return a.numpy().squeeze()

    def learn(self):
        indices = np.random.choice(self.mem_size, size=self.batch_size)  # 随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]  # 根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]  # 从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  # 从bt获得数据a
        br = bt[:, self.s_dim + self.a_dim:-self.s_dim]  # 从bt获得数据r
        bs_ = bt[:, -self.s_dim:]  # 从bt获得数据s_

        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            # batch_size*2
            a_ = self.actor_target(bs_)
            # batch_size*4
            q_ = self.critic_target([bs_, a_])
            y = br + self.gamma * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(tf.reshape(y, [-1]), tf.reshape(q, [-1]))
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # 【敲黑板】：注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        # self.ema_update()
        self.soft_replace()
        return td_error.numpy(), -a_loss.numpy()

    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        # 把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_))

        # pointer是记录了曾经有多少数据进来。
        # index是记录当前最新进来的数据位置。
        # 所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % self.mem_size  # replace the old memory with new memory
        # 把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self, model_path, eps):
        """
        save trained weights
        :return: None
        """
        model_path = os.getcwd() + model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        tl.files.save_weights_to_hdf5('{}actor_{}.hdf5'.format(model_path, str(eps)), self.actor)
        tl.files.save_weights_to_hdf5('{}actor_target_{}.hdf5'.format(model_path, str(eps)), self.actor_target)
        tl.files.save_weights_to_hdf5('{}critic_{}.hdf5'.format(model_path, str(eps)), self.critic)
        tl.files.save_weights_to_hdf5('{}critic_target_{}.hdf5'.format(model_path, str(eps)), self.critic_target)

    def load_ckpt(self, version, eps):
        """
        load trained weights
        :return: None
        """
        model_path = os.getcwd() + '/{}/{}/'.format(version, 'models')
        tl.files.load_hdf5_to_weights_in_order('{}actor_{}.hdf5'.format(model_path, str(eps)), self.actor)
        tl.files.load_hdf5_to_weights_in_order('{}actor_target_{}.hdf5'.format(model_path, str(eps)), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('{}critic_{}.hdf5'.format(model_path, str(eps)), self.critic)
        tl.files.load_hdf5_to_weights_in_order('{}critic_target_{}.hdf5'.format(model_path, str(eps)), self.critic_target)