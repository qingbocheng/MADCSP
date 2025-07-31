import warnings

import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" 
import sys
import gc
from pathlib import Path
import torch
import argparse
from mmengine.config import Config, DictAction
import numpy as np
import random
import gym
from copy import deepcopy
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)
from pm.registry import ENVIRONMENT
from pm.registry import AGENT
from pm.registry import DATASET
from pm.utils import update_data_root
from pm.utils import ReplayBuffer

def init_before_training(seed=3407):
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)


def make_env(env_id, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        return env

    return thunk


def parse_args():
    parser = argparse.ArgumentParser(description='PM train script')
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "mask_sac_portfolio_management_cap.py"),
                        help="config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--if_remove", action="store_true", default=True)
    args = parser.parse_args()
    return args


def main():
    # torch.cuda.set_per_process_memory_fraction(0.9)
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        args.cfg_options["root"] = args.root
    if args.workdir is not None:
        args.cfg_options["workdir"] = args.workdir
    if args.tag is not None:
        args.cfg_options["tag"] = args.tag
    cfg.merge_from_dict(args.cfg_options)
    # print(cfg)

    update_data_root(cfg, root=args.root)

    init_before_training(cfg.rand_seed)

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    print(50 * "-" + "build dataset" + "-" * 50)
    dataset = DATASET.build(cfg.dataset)

    train_sets, valid_sets = rolling_train_split(cfg.rolling_split_path, cfg.train_start_date, cfg.test_end_date,
        cfg.train_size, cfg.valid_size, cfg.test_size, cfg.offset)

    print(50 * "-" + "build train enviroment" + "-" * 50)
    '''pre train env'''
    cfg.environment.update(dict(
        mode="train",
        if_norm=True,
        dataset=dataset,
        start_date=train_sets[0][0],
        end_date=train_sets[0][1]
    ))
    pre_train_environment = ENVIRONMENT.build(cfg.environment)
    pre_train_envs = gym.vector.SyncVectorEnv(
        [make_env("PortfolioManagement-v0",
                  env_params=dict(env=deepcopy(pre_train_environment),
                                  transition_shape=cfg.transition_shape, seed=cfg.seed + i)) for i in
         range(cfg.num_envs)]
    )
    '''train env'''
    train_environment = ENVIRONMENT.build(cfg.environment)
    train_envs = gym.vector.SyncVectorEnv(
        [make_env("PortfolioManagement-v0",
                  env_params=dict(env=deepcopy(train_environment),
                                  transition_shape=cfg.transition_shape, seed=cfg.seed + i)) for i in
         range(cfg.num_envs)]
    )

    print(50 * "-" + "build val enviroment" + "-" * 50)
    cfg.environment.update(dict(
        mode="valid",
        if_norm=True,
        dataset=dataset,
        scaler=train_environment.scaler,
        start_date=valid_sets[0][0],
        end_date=valid_sets[0][1]
    ))
    val_environment = ENVIRONMENT.build(cfg.environment)
    val_envs = gym.vector.SyncVectorEnv(
        [make_env("PortfolioManagement-v0",
                  env_params=dict(env=deepcopy(val_environment),
                                  transition_shape=cfg.transition_shape)) for i in range(cfg.test_num_envs)]
    )

    print(50 * "-" + "build agent" + "-" * 50)
    cfg.agent.update(dict(device=device))
    agent = AGENT.build(cfg.agent)

    '''init agent.last_state'''
    dataset.init_all()
    make_market_obs(pre_train_envs,dataset)
    make_market_obs(train_envs,dataset)
    make_market_obs(val_envs,dataset)
 
    '''init buffer'''
    buffer = ReplayBuffer(
        buffer_size=cfg.buffer_size,
        transition=cfg.transition,
        transition_shape=cfg.transition_shape,
        if_use_per=cfg.if_use_per,
        device=device
        # device=torch.device("cpu")
    )
    print(cfg)
    '''pre train market'''
    Rolling_data(pre_train_envs, train_sets, 0, 0)
    for episode in range(1,1):
        state = pre_train_envs.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        agent.last_state = state
        ######################pre train######################
        print("############################pre Train Episode: [{}/{}]#############################".format(episode,
                                                                                                       20))
        while True:
            done=agent.pre_explore_env(pre_train_envs, cfg.horizon_len)
            if done[0]:
                break

    for episode in range(1, 2):
        for i in range(0, len(train_sets)):
            ######################################################
            Rolling_data(train_envs,train_sets,i,episode)
            Rolling_data(val_envs,valid_sets,i,episode,train_envs.envs[0])
            # Rolling_data(test_envs,test_sets,i,episode)
            state = train_envs.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            agent.last_state = state
            torch.cuda.empty_cache()
            gc.collect()
            ######################train######################
            print("############################Train Episode: [{}/{}]#############################".format(episode,
                                                                                                           cfg.num_episodes))
            print("############################Rolling Episode: [{}/{}]#############################".format(i + 1,
                                                                                                             len(train_sets)))
            train_one_episode(train_envs, buffer, agent, cfg.horizon_len)

            ######################val#########################
            print("################################Validate Episode: [{}/{}]#############################".format(episode,
                                                                                                                cfg.num_episodes))
            print("############################Rolling Episode: [{}/{}]#############################".format(i + 1,
                                                                                                             len(train_sets)))
            with torch.no_grad():
                validate(val_envs, agent)

def train_one_episode(environment, buffer, agent, horizon_len):

    # reset environment
    environment.reset()
    while True:
        buffer_items = agent.explore_env(environment, horizon_len)
        with torch.no_grad():
            buffer.update(buffer_items)
        with torch.set_grad_enabled(True):
        # torch.set_grad_enabled(True)
        #     agent.update_net(buffer,environment.envs[0])
            agent.update_net(buffer)
        # torch.set_grad_enabled(False)
        # if done is True in dones, find the min row index
        positive_indices = torch.nonzero(buffer_items[-2] > 0)
        if positive_indices.numel() == 0:
            min_row_index = horizon_len - 1
        else:
            min_row_index = torch.min(positive_indices[:, 0]).item()

        if buffer_items[-2][min_row_index] == True:
            break

def validate(environment, agent):

    agent.validate_net(environment)


def rolling_train_test_split(data_path, start_trade, end_trade, train_size, valid_size, test_size, offset):
    tradeday_data = pd.read_csv(data_path)[['Date']]
    tradeday_data = tradeday_data[
        (tradeday_data['Date'] >= start_trade) & (tradeday_data['Date'] <= end_trade)].reset_index(drop=True)
    tradedays = len(tradeday_data)
    train_start_index = 0
    train_end_index = train_start_index + train_size - 1
    valid_start_index = train_end_index + 1
    valid_end_index = valid_start_index + valid_size - 1
    # test_start_index = valid_end_index + 1
    # test_end_index = test_start_index + test_size - 1
    train_start_date = tradeday_data.iloc[train_start_index]['Date']
    train_end_date = tradeday_data.iloc[train_end_index]['Date']
    valid_start_date = tradeday_data.iloc[valid_start_index]['Date']
    valid_end_date = tradeday_data.iloc[valid_end_index]['Date']
    # test_start_date = tradeday_data.iloc[test_start_index]['Date']
    # test_end_date = tradeday_data.iloc[test_end_index]['Date']
    train_sets = []
    valid_sets = []
    # test_sets = []

    while valid_end_index + 64 <= tradedays:
        train_sets.append([train_start_date, train_end_date])
        valid_sets.append([valid_start_date, valid_end_date])
        # test_sets.append([test_start_date, test_end_date])

        train_start_index += offset
        train_end_index = train_start_index + train_size - 1
        valid_start_index = train_end_index + 1
        valid_end_index = valid_start_index + valid_size - 1
        # test_start_index = valid_end_index + 1
        # test_end_index = test_start_index + test_size - 1
        train_start_date = tradeday_data.iloc[train_start_index]['Date']
        train_end_date = tradeday_data.iloc[train_end_index]['Date']
        valid_start_date = tradeday_data.iloc[valid_start_index]['Date']
        valid_end_date = tradeday_data.iloc[valid_end_index]['Date']
        # test_start_date = tradeday_data.iloc[test_start_index]['Date']
        # test_end_date = tradeday_data.iloc[test_end_index]['Date']
    if len(train_sets) == 0:
        train_sets.append([train_start_date, train_end_date])
        valid_sets.append([valid_start_date, valid_end_date])
    else:
        train_sets.append([train_start_date, train_end_date])
        valid_sets.append([valid_start_date, valid_end_date])
    return train_sets, valid_sets

def rolling_train_split(data_path, start_trade, end_trade, train_size, valid_size, test_size, offset):
    tradeday_data = pd.read_csv(data_path)[['date']]
    tradeday_data = tradeday_data[
        (tradeday_data['date'] >= start_trade) & (tradeday_data['date'] <= end_trade)].reset_index(drop=True)
    tradedays = len(tradeday_data)
    train_start_index = 0
    train_end_index = train_start_index + train_size - 1
    valid_start_index = train_end_index + 1
    valid_end_index = valid_start_index + valid_size - 1
    # test_start_index = valid_end_index + 1
    # test_end_index = test_start_index + test_size - 1
    train_start_date = tradeday_data.iloc[train_start_index]['date']
    train_end_date = tradeday_data.iloc[train_end_index]['date']
    valid_start_date = tradeday_data.iloc[valid_start_index]['date']
    valid_end_date = tradeday_data.iloc[valid_end_index]['date']
    # test_start_date = tradeday_data.iloc[test_start_index]['Date']
    # test_end_date = tradeday_data.iloc[test_end_index]['Date']

    train_sets = []
    valid_sets = []
    # test_sets = []

    while valid_end_index + 64 <= tradedays:
        train_sets.append([train_start_date, train_end_date])
        valid_sets.append(['2022-08-06', '2025-03-01'])
        #valid_sets.append(['2021-10-20', '2024-12-31'])
        # test_sets.append([test_start_date, test_end_date])

        train_start_index += offset
        train_end_index = train_start_index + train_size - 1
        valid_start_index = train_end_index + 1
        valid_end_index = valid_start_index + valid_size - 1
        # test_start_index = valid_end_index + 1
        # test_end_index = test_start_index + test_size - 1
        train_start_date = tradeday_data.iloc[train_start_index]['date']
        train_end_date = tradeday_data.iloc[train_end_index]['date']
        valid_start_date = tradeday_data.iloc[valid_start_index]['date']
        valid_end_date = tradeday_data.iloc[valid_end_index]['date']
        # test_start_date = tradeday_data.iloc[test_start_index]['Date']
        # test_end_date = tradeday_data.iloc[test_end_index]['Date']
    if len(train_sets) == 0:
        train_sets.append([train_start_date, train_end_date])
        valid_sets.append(['2022-08-06', '2025-03-01'])
        #valid_sets.append(['2021-10-20', '2024-12-31'])

    else:
        train_sets.append([train_start_date, train_end_date])
        valid_sets.append(['2022-08-06', '2025-03-01'])
        #valid_sets.append(['2021-10-20', '2024-12-31'])


    return train_sets, valid_sets

def Rolling_data(environment, dataset, num, epoch,train_env=None):
    num_envs = environment.num_envs
    stock_types = {0:'GSP',1:'CSP1',2:'CSP2',3:'CSP3'}
    for i in range(num_envs):
        if environment.envs[i].mode != 'train':
            environment.envs[i].scaler_init(train_env)
        environment.envs[i].Rolling(dataset[num][0], dataset[num][1], num+1, epoch,stock_types[i])


def make_market_obs(environment,dataset):
    num_envs = environment.num_envs
    for i in range(num_envs):
        environment.envs[i].env_data_init(dataset)


if __name__ == '__main__':
    main()


