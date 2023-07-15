import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from Env import ContinuousEnv
from ENV import ContinuousEnv
import numpy as np
import random
import tensorboard_logger as tb_logger
from model import DNC_PPO
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='MPGD',
                        choices=['MPGD', 'DNC_A2C', 'MAPPO', 'MAA2C', 'Greedy', 'Random', 'EdgeAll'],
                        type=str, help='algorithm name, including MPGD, MAPPO, MAA2C, Greedy, Random')
    parser.add_argument('--ppo_mode', default='clip', choices=['clip', 'kl_pen'], help='coefficient fo regulation term')
    parser.add_argument('--user_num', default=4, type=int, help='the number of user')
    parser.add_argument('--hist_len', default=5, type=int, help='the length of history')
    # parser.add_argument('--a_dim', default=1, type=int, help='the dimension of action')
    parser.add_argument('--env_bw', default='fixed', choices=['fixed', 'var_single', 'var_all','gama_8']
                        , help='the bandwidth of environment')
    parser.add_argument('--epoch_num', default=200, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epoch_len', default=20, type=int, help='the length of each epoch')
    parser.add_argument('--a_update_step', default=5, type=int, help='the update frequency of actor')
    parser.add_argument('--c_update_step', default=5, type=int, help='the update frequency of critic')
    parser.add_argument('--gamma', default=0.95, type=float, help='discount factor')
    parser.add_argument('--a_lr', default=6e-5, type=float, help='learning rate of actor')
    parser.add_argument('--c_lr', default=6e-5, type=float, help='learning rate of critic')
    parser.add_argument('--dcy', default=0.5, type=float, help='decay')
    parser.add_argument('--reg_term', default=3e-5, type=float, help='coefficient fo regulation term')
    parser.add_argument('--ppo_clip_eps', default=0.1, type=float, help='epsilon for ppo clip')
    parser.add_argument('--ppo_kl_tgt', default=0.01, type=float, help='kl target for ppo kl penalty')
    parser.add_argument('--ppo_kl_lam', default=0.5, type=float, help='lambda for ppo ppo kl penalty')
    parser.add_argument('--log_dir', default='logs', type=str, help='the dir of logs')
    # parser.add_argument('--log_dir', default='./var_env_logs2/', type=str, help='the dir of logs')
    # parser.add_argument('--log_dir', default='./allvar_env_logs2/', type=str, help='the dir of logs')
    parser.add_argument('--restore_dir', default='', type=str, help='the dir for restoring model')
    parser.add_argument('--upper_bound', default=0.99, type=float, help='the upper bound for precision')

    FLAGS, _ = parser.parse_known_args()
    return FLAGS


def get_model(args, user_idx, load=False):
    s_dim = (args.user_num) * args.hist_len
    a_dim = 1
    method = {'name': args.ppo_mode,
              'kl_target': args.ppo_kl_tgt,
              'lam': args.ppo_kl_lam,
              'epsilon': args.ppo_clip_eps,
              'DNC': False}
    if args.algo == 'MAPPO':
        print('get MAPPO model for user', user_idx)
        # model = MAPPO(s_dim, a_dim, args.batch_size, args.a_update_step, args.c_update_step,
        #               user_idx, method, args.reg_term, args.restore_dir)
        model = DNC_PPO(s_dim, a_dim, args.batch_size, args.a_update_step, args.c_update_step,
                        user_idx, method, args.reg_term, args.restore_dir)
    elif args.algo == 'MAA2C':
        print('get MAA2C model for user', user_idx)
        method['name'] = 'a2c'
        model = DNC_PPO(s_dim, a_dim, args.batch_size, args.a_update_step, args.c_update_step,
                        user_idx, method, args.reg_term, args.restore_dir)
    elif args.algo == 'DNC_A2C':
        print('get MAA2C model for user', user_idx)
        method['name'] = 'a2c'
        method['DNC'] = True
        model = DNC_PPO(s_dim, a_dim, args.batch_size, args.a_update_step, args.c_update_step,
                        user_idx, method, args.reg_term, args.restore_dir)
    elif args.algo == 'MPGD':
        print('get MPGD model for user', user_idx)
        method['DNC'] = True
        model = DNC_PPO(s_dim, a_dim, args.batch_size, args.a_update_step, args.c_update_step,
                        user_idx, method, args.reg_term, args.restore_dir)
    else:
        model = None
    if model != None and load == True:
        model.loader()
    return model


def get_env(args):
    env = ContinuousEnv(args.user_num, args.hist_len, args.upper_bound)
    return env


def main():
    args = parse_args()
    train(args)


def train(args):
    A_LR = args.a_lr
    C_LR = args.c_lr
    v_s = np.zeros(args.user_num)
    ppo = []
    loggers = []
    loggers_NE = []
    dec = args.dcy
    action = np.zeros(args.user_num)

    max_r = np.zeros(args.user_num)
    max_a = np.random.random(args.user_num)

    step = 0
    env = get_env(args)
    UB = args.upper_bound
    for i in range(args.user_num):
        load = False;
        if os.path.exists("./ckpt"):
            load = True
        ppo.append(get_model(args, i, load))
        print(args.log_dir + '/' + args.env_bw + '/run-' + args.algo + '_user_' + str(i) + '/')
        loggers.append(
            tb_logger.Logger(args.log_dir + '/' + args.env_bw + '/run-' + args.algo + '_user_' + str(i) + '/'))

        # loggers_NE.append(
        #    tb_logger.Logger(args.log_dir + '/' + args.env_bw + '/run-' + args.algo + '_user_' + str(i) + '_NE' + '/'))

    rewards = []
    actions = []
    closs = []
    aloss = []

    for ep in range(args.epoch_num):
        precision = 0
        cur_state = env.reset()
        if ep % 50 == 0:
            dec = dec * 1
            A_LR = A_LR * 0.8
            C_LR = C_LR * 0.8
        '''buffer_s = [[] for _ in range(args.user_num)]
        buffer_a = [[] for _ in range(args.user_num)]
        buffer_r = [[] for _ in range(args.user_num)]'''
        sum_reward = np.zeros(args.user_num)
        sum_action = np.zeros(args.user_num)
        sum_closs = np.zeros(args.user_num)
        sum_aloss = np.zeros(args.user_num)
        sum_util_ne = np.zeros(args.user_num)
        sum_action_ne = np.zeros(args.user_num)
        for t in range(args.epoch_len):
            buffer_s = [[] for _ in range(args.user_num)]
            buffer_a = [[] for _ in range(args.user_num)]
            buffer_r = [[] for _ in range(args.user_num)]
            if precision > UB*100:##??????:
                break
            for i in range(args.user_num):
                if args.algo == 'Greedy':
                    # Greedy algorithm
                    if np.random.random() < 0.1:
                        action[i] = np.random.random()
                    else:
                        action[i] = max_a[i]
                elif args.algo == 'Random':
                    action[i] = np.random.random()
                elif args.algo == 'EdgeAll':
                    action[i] = 1.0
                else:
                    action[i] = ppo[i].choose_action(cur_state[i], dec)
            #print(action)
            #action = np.array([max(0.75,i) for i in action]) #altered!!!!!!!!!!!!!!!!!!
            # last three are delay_ne, energy_ne, utility_edgeall
            #print(action)
            next_state, reward, precision = env.step(action)
            if args.algo == 'EdgeAll':
                pass
            else:
                sum_reward += reward
                #print(reward)
            sum_action += action
            #print(action)
            #print(sum_action)
            # sum_util_ne += util_ne
            # sum_action_ne += action_ne
            step += 1

            # update
            if args.algo == 'Greedy':
                # Greedy algorithm
                for i in range(args.user_num):
                    if reward[i] > max_r[i]:
                        max_r[i] = reward[i]
                        max_a[i] = action[i]
                    if max_a[i] == action[i]:
                        max_r[i] = reward[i]
            elif args.algo == 'Random' or args.algo == 'EdgeAll':
                pass
            else:
                for i in range(args.user_num):
                    v_s[i] = ppo[i].get_v(next_state[i])

                for i in range(args.user_num):
                    buffer_a[i].append(action[i])
                    buffer_s[i].append(cur_state[i])
                    buffer_r[i].append(reward[i])

                cur_state = next_state
                # update ppo
                if (t + 1) % args.batch_size == 0:
                    for i in range(args.user_num):
                        discounted_r = np.zeros(len(buffer_r[i]), 'float32')
                        v_s[i] = ppo[i].get_v(next_state[i])
                        running_add = v_s[i]

                        for rd in reversed(range(len(buffer_r[i]))):
                            running_add = running_add * args.gamma + buffer_r[i][rd]
                            discounted_r[rd] = running_add

                        discounted_r = discounted_r[np.newaxis, :]
                        discounted_r = np.transpose(discounted_r)
                        l_c, l_a = ppo[i].update(np.vstack(buffer_s[i]), np.vstack(
                            buffer_a[i]), discounted_r, dec, A_LR, C_LR, ep)
                        sum_closs[i] += l_c
                        sum_aloss[i] += l_a
        #!!!!!!!!!!altered here
        env.reset()

        if True:#ep % 10 == 0:
            print('ep:', ep)
            print("action:", action)
            #print("t is ", t)
            #print(args.epoch_len)
            mean_reward = sum_reward / (t+1)#args.epoch_len
            mean_action = sum_action / (t+1)#args.epoch_len
            mean_closs = sum_closs / (t+1)#args.epoch_len
            mean_aloss = sum_closs / (t+1)#args.epoch_len
            # mean_util_ne = sum_util_ne / args.epoch_len
            # mean_action_ne = sum_action_ne / args.epoch_len
            print("Sum reward for individual:", sum_reward)
            print("Total reward",sum(sum_reward))
            rewards.append(mean_reward)
            actions.append(mean_action)
            closs.append(mean_closs)
            aloss.append(mean_aloss)
            print("Precision:", precision)
            print("average reward:", mean_reward)
            print("average action:", mean_action)
            print("average closs:", mean_closs)
            print("average aloss:", mean_aloss)
            for i in range(args.user_num):
                loggers[i].log_value("utility", mean_reward[i], step)
                loggers[i].log_value("action", mean_action[i], step)
                loggers[i].log_value("critic loss", mean_closs[i], step)
                loggers[i].log_value("action loss", mean_aloss[i], step)
                # loggers_NE[i].log_value("utility", mean_util_ne[i], step)
                # loggers_NE[i].log_value("action", mean_action_ne[i], step)
            loggers[0].log_value('overall utility', sum(sum_reward), step)


if __name__ == '__main__':
    main()
