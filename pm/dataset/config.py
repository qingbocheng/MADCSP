#！/usr/bin/python
# -*- coding: utf-8 -*-#
import os
import pandas as pd
import datetime

class Config():
    def __init__(self, seed_num=2022, current_date=None):

        self.notes = 'MADCSP Implementation'
        self.benchmark_algo = 'MADCSP-mlp'
        self.market_name = 'NDX' # Financial Index: 'DJIA', 'SP500', 'CSI300','CRP'
        self.topK = 100 # Number of assets in a portfolio (10, 20, 30)
        self.num_epochs = 50 # episode.
        self.if_cap = True
        self.if_crp = False
        self.gmv = 'cvar'
        self.csp = 'ndx'

        if 'TD3' in self.benchmark_algo:
            self.rl_model_name = 'TD3'
            self.mode = 'RLonly'
            self.mktobs_algo = None
            obj_name = self.benchmark_algo.split('-')[1]
            if obj_name == 'Profit':
                self.trained_best_model_type = 'max_capital'
            elif obj_name == 'PR':
                self.trained_best_model_type = 'pr_loss'
            elif obj_name == 'SR':
                self.trained_best_model_type = 'sr_loss'
            else:
                raise ValueError("Undefined obj_name [{}] of {}.".format(obj_name, self.benchmark_algo))

        elif 'MADCSP' in self.benchmark_algo:
            self.rl_model_name = 'TD3' # RL-based agent is implemented by TD3 in the paper, and can be replaced by other RL approaches.
            self.mode = 'RLcontroller'
            # self.rl_model_name = 'SAC' # RL-based agent is implemented by TD3 in the paper, and can be replaced by other RL approaches.
            self.mktobs_algo = '{}_1'.format(self.benchmark_algo.split('-')[1]) # 'dc_1', 'ma_1', 'mlp_1'
            self.trained_best_model_type = 'js_loss'
        else:
            # Baseline models
            self.rl_model_name = self.benchmark_algo
            self.mode = 'Benchmark'
            self.mktobs_algo = None
            self.trained_best_model_type = 'max_capital'

        self.is_enable_dynamic_risk_bound = False # True if enabling that the market observer sends info to solver-based agents. 
        self.enable_controller = True # True if enabling the solver-based agent.
        self.enable_market_observer = True # True if enabling the market observer.
        self.enable_market_observer_CSP = True # True if enabling the market observer.

        self.trade_pattern = 1 # 1: Long only, 2: Long and short (Not applicable), 3: short only (Not applicable)
        self.lambda_1 = 100.0 # return reward weight
        self.lambda_2 = 10.0 # action reward weight
        self.lambda_3 = 0.5
        self.train_freq = [1, 'episode'] 
        self.risk_default = 0.017
        if (self.market_name == 'DJIA') and (self.topK == 30):
            self.topK = 30 # Only 29 stocks having complete data in the DJIA during that period.
        self.risk_up_bound = 0.012 # Decided by the observation of the training data set.  
        self.risk_hold_bound = 0.014 
        self.risk_down_bound = 0.017

        self.period_mode = 1 
        self.tmp_name = 'Cls3_{}_{}_K{}_M{}_{}_{}'.format(self.mode, self.mktobs_algo, self.topK, self.period_mode, self.market_name, self.trained_best_model_type)
        self.dataDir = '/mnt/f/R_test'
        # self.dataDir = '../../all_data'

        self.pricePredModel = 'MA'
        self.cov_lookback = 21
        self.norm_method = 'sum'

        if self.mode == 'Benchmark':
            self.trained_best_model_type = 'max_capital'
        if self.mode == 'RLonly':
            if self.trained_best_model_type not in ['max_capital', 'pr_loss', 'sr_loss']:
                raise ValueError("The trained_best_model_type[{}] of {} should be in [\'max_capital\', \'pr_loss\', \'sr_loss\']".format(self.trained_best_model_type, self.mode))

        self.risk_market = 0.001 # \Sigma_beta
        self.cbf_gamma = 0.7

        self.reward_scaling = 1 
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.gradient_steps = 1 
        self.ars_trial = 10

        if current_date is None:
            self.cur_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.cur_datetime = current_date
        self.res_dir = os.path.join('/root/res', self.mode, '{}-{}'.format(self.market_name, self.topK), self.cur_datetime)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        self.res_model_dir = os.path.join(self.res_dir, 'model')
        if not os.path.exists(self.res_model_dir):
            os.makedirs(self.res_model_dir)
        self.res_img_dir = os.path.join(self.res_dir, 'graph')
        if not os.path.exists(self.res_img_dir):
            os.makedirs(self.res_img_dir)
        self.tradeDays_per_year = 252
        self.tradeDays_per_month = 21
        self.seed_num = seed_num
        date_split_dict = {
            # 1: {'train_date_start': '2016-01-01 00:00:00',
            #     'train_date_end': '2020-08-31 23:59:59',
            #     'valid_date_start': '2020-09-01 00:00:0',
            #     # 'valid_date_start': '2019-07-22 00:00:0',
            #     'valid_date_end': '2021-10-19 23:59:59',
            #     # 'test_date_start': '2024-01-01 00:00:00',
            #     'test_date_start': '2021-10-20 00:00:00',
            #     'test_date_end': '2024-12-31 23:59:59'},
            1: {'train_date_start': '2017-01-01 00:00:00',
                 'train_date_end': '2021-12-10 23:59:59',
                 'valid_date_start': '2021-12-11 00:00:0',
                 # 'valid_date_start': '2019-07-22 00:00:0',
                 'valid_date_end': '2022-08-05 23:59:59',
                 # 'test_date_start': '2024-01-01 00:00:00',
                 'test_date_start': '2022-08-06 00:00:00',
                 'test_date_end': '2025-03-01 23:59:59'},
        }
        
        self.train_date_start = pd.Timestamp(date_split_dict[self.period_mode]['train_date_start'])
        self.train_date_end = pd.Timestamp(date_split_dict[self.period_mode]['train_date_end'])
        if (date_split_dict[self.period_mode]['valid_date_start'] is not None) and (date_split_dict[self.period_mode]['valid_date_end'] is not None):
            self.valid_date_start = pd.Timestamp(date_split_dict[self.period_mode]['valid_date_start'])
            self.valid_date_end = pd.Timestamp(date_split_dict[self.period_mode]['valid_date_end'])
        else:
            self.valid_date_start = None
            self.valid_date_end = None
        if (date_split_dict[self.period_mode]['test_date_start'] is not None) and (date_split_dict[self.period_mode]['test_date_end'] is not None):
            self.test_date_start = pd.Timestamp(date_split_dict[self.period_mode]['test_date_start'])
            self.test_date_end = pd.Timestamp(date_split_dict[self.period_mode]['test_date_end'])
        else:
            self.test_date_start = None
            self.test_date_end = None        

        self.tech_indicator_talib_lst = []
        # self.tech_indicator_extra_lst = ['CHANGE']
        self.tech_indicator_extra_lst = []
        self.tech_indicator_input_lst = self.tech_indicator_talib_lst + self.tech_indicator_extra_lst
        self.dailyRetun_lookback = self.cov_lookback
        self.otherRef_indicator_ma_window = 21
        self.enable_cov_features = False # enable using the cov features in RL-based agent

        self.otherRef_indicator_lst = ['MA-{}'.format(self.otherRef_indicator_ma_window), 'DAILYRETURNS-{}'.format(self.dailyRetun_lookback)]

        self.mkt_rf = { 
            'NDX': 1.6575,
            'CSI300': 1.037,
            'SH50': 1.037,
            'DJIA': 1.6575,
            'HSI': 1.037,
        }

        self.market_close_time = {
            'CSI300': '15:00:00',
            'SH50': '15:00:00',
        }
        self.invest_env_para = {
            'max_shares': 100, 'initial_asset': 100000, 'reward_scaling': self.reward_scaling, 'norm_method': self.norm_method,
            'transaction_cost': 0.0005, 'slippage': 0.001, 'seed_num': self.seed_num
        } 

        self.only_long_algo_lst = ['CRP', 'EG', 'OLMAR', 'PAMR', 'RMR']
        self.use_cash_algo_lst = ['RAT', 'EIIE', 'PPN'] 
        if self.mode != 'RLcontroller':
            self.enable_controller = True
            self.enable_market_observer = True
            self.is_enable_dynamic_risk_bound = True

        if self.risk_default <= self.risk_market:
            raise ValueError("The boundary of safe risk[{}] should not be less than/ equal to the market risk[{}].".format(self.risk_default, self.risk_market))

        if self.mktobs_algo is not None:
            if 'dc' in self.mktobs_algo:
                self.is_gen_dc_feat = True
                self.dc_threshold = [0.01] 
            else:
                self.is_gen_dc_feat = False
        else:
            self.is_gen_dc_feat = False

        self.load_para()  
        self.load_model_config()
        self.load_market_observer_config()

    def load_model_config(self):
        self.use_features = ['close', 'open', 'high', 'low'] 
        self.window_size = 31
        self.po_lr = 0.0001
        self.po_weight_decay = 0.001

    def load_market_observer_config(self):
        self.freq = '1d'
        self.finefreq = '60m'
        self.fine_window_size = 4
        self.feat_scaler = 10 
        
        self.hidden_vec_loss_weight = 1
        self.sigma_loss_weight = 1e3
        self.lambda_min = 0.0
        self.lambda_max = 1.0
        self.sigma_min = 0.0  
        self.sigma_max = 1.0  
        
        self.finestock_feat_cols_lst = []
        self.finemkt_feat_cols_lst = []
        for ifeat in self.use_features:
            for iwin in range(1, self.fine_window_size+1):
                self.finestock_feat_cols_lst.append('stock_{}_{}_w{}'.format(self.finefreq, ifeat, iwin))
                self.finemkt_feat_cols_lst.append('mkt_{}_{}_w{}'.format(self.finefreq, ifeat, iwin))

    def load_para(self):
        if self.enable_market_observer:
            if self.rl_model_name == 'TD3':
                policy_name = 'TD3PolicyAdj'
            else:
                raise ValueError("Cannot specify the {} policy name when enabling market observer.".format(self.rl_model_name))
        else:
            if self.rl_model_name == 'TD3':
                policy_name = None
            else:
                if self.mode in ['RLonly', 'RLcontroller']:
                    raise ValueError("Cannot specify the {} policy name when using stable-baseline.".format(self.rl_model_name))
                else:
                    policy_name = "MlpPolicy"
        base_para = {
            'policy': policy_name, 'learning_rate': self.learning_rate, 'buffer_size': 100000,
            'learning_starts': 100, 'batch_size': self.batch_size, 'tau': 0.005, 'gamma': 0.99, 'train_freq': (self.train_freq[0], self.train_freq[1]),
            'gradient_steps': self.gradient_steps, 'action_noise': None,  'replay_buffer_class': None, 'replay_buffer_kwargs': None, 
            'optimize_memory_usage': False, 'tensorboard_log': None, 'policy_kwargs': None, 
            'verbose': 1, 'seed': self.seed_num, 'device': 'auto', '_init_setup_model': True,
        }
        algo_para = {
            'TD3': {'policy_delay': 2, 'target_policy_noise': 0.2, 'target_noise_clip': 0.5,},
            'SAC': {'ent_coef': 'auto', 'target_update_interval': 1, 'target_entropy': 'auto', 'use_sde': False, 'sde_sample_freq': -1, 'use_sde_at_warmup': False,},
            'PPO': {'n_steps': 100},
        }
        if (self.topK == 20) or (self.topK == 30):
            algo_para['TD3']['policy_kwargs'] = {
                'net_arch': [1024, 512, 128], # [400, 300]
            }
        algo_para_rm_from_base = {
            'PPO': ['buffer_size', 'learning_starts', 'tau', 'train_freq', 'gradient_steps', 'action_noise', 'replay_buffer_class', 'replay_buffer_kwargs', 'optimize_memory_usage']
        }
        if self.rl_model_name in algo_para.keys():
            self.model_para = {**base_para, **algo_para[self.rl_model_name]}
        else:
            self.model_para = base_para
        if self.rl_model_name in algo_para_rm_from_base.keys():
            for rm_field in algo_para_rm_from_base[self.rl_model_name]:
                del self.model_para[rm_field]


    def print_config(self):
        log_str = '=' * 30 + '\n'
        para_str = '{} \n'.format(self.notes)
        log_str = log_str + para_str
        para_str = 'mode: {}, rl_model_name: {}, market_name: {}, topK: {}, dataDir: {}, enable_controller: {}, \n'.format(self.mode, self.rl_model_name, self.market_name, self.topK, self.dataDir, self.enable_controller)
        log_str = log_str + para_str
        para_str = 'trade_pattern: {} \n'.format(self.trade_pattern)
        log_str = log_str + para_str
        para_str = 'period_mode: {}, num_epochs: {}, cov_lookback: {}, norm_method: {}, benchmark_algo: {}, trained_best_model_type: {}, pricePredModel: {}, \n'.format(self.period_mode, self.num_epochs, self.cov_lookback, self.norm_method, self.benchmark_algo, self.trained_best_model_type, self.pricePredModel)
        log_str = log_str + para_str
        para_str = 'is_enable_dynamic_risk_bound: {}, risk_market: {}, risk_default: {}, cbf_gamma: {}, ars_trial: {} \n'.format(self.is_enable_dynamic_risk_bound, self.risk_market, self.risk_default, self.cbf_gamma, self.ars_trial)
        log_str = log_str + para_str
        para_str = 'cur_datetime: {}, res_dir: {}, tradeDays_per_year: {}, tradeDays_per_month: {}, seed_num: {}, \n'.format(self.cur_datetime, self.res_dir, self.tradeDays_per_year, self.tradeDays_per_month, self.seed_num)
        log_str = log_str + para_str
        para_str = 'train_date_start: {}, train_date_end: {}, valid_date_start: {}, valid_date_end: {}, test_date_start: {}, test_date_end: {}, \n'.format(self.train_date_start, self.train_date_end, self.valid_date_start, self.valid_date_end, self.test_date_start, self.test_date_end)
        log_str = log_str + para_str
        para_str = 'tech_indicator_input_lst: {}, \n'.format(self.tech_indicator_input_lst)
        log_str = log_str + para_str
        para_str = 'otherRef_indicator_lst: {}, enable_cov_features: {} \n'.format(self.otherRef_indicator_lst, self.enable_cov_features)
        log_str = log_str + para_str
        para_str = 'tmp_name: {}, mkt_rf: {} \n'.format(self.tmp_name, self.mkt_rf)
        log_str = log_str + para_str
        para_str = 'invest_env_para: {}, \n'.format(self.invest_env_para)
        log_str = log_str + para_str
        para_str = 'model_para: {}, \n'.format(self.model_para)
        log_str = log_str + para_str
        para_str = 'only_long_algo_lst: {}, \n'.format(self.only_long_algo_lst)
        log_str = log_str + para_str
        para_str = 'lstr para: use_features: {}, window_size: {}, freq: {}, finefreq: {}, fine_window_size: {}, \n'.format(self.use_features, self.window_size, self.freq, self.finefreq, self.fine_window_size)
        log_str = log_str + para_str
        para_str = 'enable_market_observer: {}, mktobs_algo: {}, feat_scaler: {} \n'.format(self.enable_market_observer, self.mktobs_algo, self.feat_scaler)
        log_str = log_str + para_str
        log_str = log_str + '=' * 30 + '\n'

        print(log_str, flush=True)




