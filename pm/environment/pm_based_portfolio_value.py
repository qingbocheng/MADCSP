import numpy as np
import pandas as pd
import os
from typing import List, Any
from sklearn.preprocessing import StandardScaler
import random
import gym
from gym import spaces
import copy
from pathlib import Path
from pm.registry import ENVIRONMENT
# from stable_baselines3.common.vec_env import DummyVecEnv
import scipy.stats as spstats
from scipy.stats import entropy
from sklearn.covariance import ledoit_wolf,oas,ledoit_wolf_shrinkage



@ENVIRONMENT.register_module()
class EnvironmentPV(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 if_norm: bool = True,
                 if_norm_temporal: bool = True,
                 scaler: List[StandardScaler] = None,
                 days: int = 10,
                 start_date: str = None,
                 end_date: str = None,
                 initial_amount: int = 1e4,
                 transaction_cost_pct: float = 1e-3
                 ):
        super(EnvironmentPV, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.if_norm = if_norm
        self.if_norm_temporal = if_norm_temporal
        self.scaler = scaler
        self.days = days
        self.start_date = start_date
        self.end_date = end_date
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.epoch = 0
        self.stepcount = 0

        if end_date is not None:
            assert end_date > start_date, "start date {}, end date {}, end date should be greater than start date".format(
                start_date, end_date)

        self.stocks = self.dataset.stocks
        self.stocks2id = self.dataset.stocks2id
        self.id2stocks = self.dataset.id2stocks
        self.aux_stocks = self.dataset.aux_stocks

        self.features_name = self.dataset.features_name
        self.prices_name = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        self.temporals_name = self.dataset.temporals_name
        self.labels_name = self.dataset.labels_name
        self.done = False
        self.opt=0
        if self.mode == "train":
            # self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
            self.day = 0
        else:
            # self.day = self.days - 1
            self.day = 0

        ##########################初始化##############################
        self.curTradeDay = 0
        self.config = self.dataset.config
        self.rolling_num=0
        self.mkt_observer = None
        # self.mkt_observer = self.dataset.market_obs
        self.env_mask = None
        self.cur_hidden_vector_ay = []
        ##############################################################################################
        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]
        self.var_lst = [0.0]
        self.cur_capital = self.initial_amount
        self.asset_lst = [self.initial_amount]
        self.profit_lst = [0] # percentage of portfolio daily returns
        self.is_last_ctrl_solvable = False
        self.risk_raw_lst = [0] # For performance analysis. Record the risk without using risk controllrt during the validation/test period.
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_amount] 
        self.risk_pred_lst = []
        self.rl_reward_risk_lst = []
        self.rl_reward_profit_lst = []
        self.reward_lst = [0]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.config.topK + 1 if self.config.if_cap else self.config.topK,))
        self.bound_flag = 1  # 1 for long and long+short, -1 for short
        self.action_cbf_memeory = [np.array([0] * ((self.config.topK+ 1) if self.config.if_cap else self.config.topK))]
        self.actions_memory = [np.random.dirichlet(np.ones(self.config.topK+ 1 if self.config.if_cap else self.config.topK))* self.bound_flag]
        self.action_rl_memory = [self.actions_memory[-1]]
        # self.action_cbf_memeory = []
        # self.actions_memory = []
        # self.action_rl_memory = []
        self.ctrl_weight_lst = [1.0]
        self.slippage = 0.001 # 0.001 for one-side, 0.002 for two-side
        self.cur_slippage_drift = np.random.random(self.config.topK) * (self.slippage * 2) - self.slippage # 拟与价格滑点（slippage）相关的随机偏差
        self.solvable_flag = []
        self.solver_stat = {'solvable': 0, 'insolvable': 0, 'stochastic_solvable': 0, 'stochastic_time': [], 'socp_solvable': 0, 'socp_time': []}
        # For saveing profile
        self.profile_hist_field_lst = [
            'ep','rolling_num','stock_type','start_day','end_day', 'trading_days', 'annualReturn_pct', 'mdd', 'sharpeRatio', 'final_capital', 'volatility',
            'calmarRatio', 'sterlingRatio',
            'netProfit', 'netProfit_pct', 'winRate',
            'vol_max', 'vol_min', 'vol_avg',
            'risk_max', 'risk_min', 'risk_avg', 'riskRaw_max', 'riskRaw_min', 'riskRaw_avg',
            'dailySR_max', 'dailySR_min', 'dailySR_avg', 'dailySR_wocbf_max', 'dailySR_wocbf_min', 'dailySR_wocbf_avg',
            'dailyReturn_pct_max', 'dailyReturn_pct_min', 'dailyReturn_pct_avg',
            'sigReturn_max', 'sigReturn_min', 'mdd_high', 'mdd_low', 'mdd_high_date', 'mdd_low_date', 'sharpeRatio_wocbf',
            'reward_sum', 'final_capital_wocbf', 'cbf_contribution',
            'risk_downsideAtVol', 'risk_downsideAtVol_daily_max', 'risk_downsideAtVol_daily_min', 'risk_downsideAtVol_daily_avg',
            'risk_downsideAtValue_daily_max', 'risk_downsideAtValue_daily_min', 'risk_downsideAtValue_daily_avg',
            'cvar_max', 'cvar_min', 'cvar_avg', 'cvar_raw_max', 'cvar_raw_min', 'cvar_raw_avg',
            'solver_solvable', 'solver_insolvable','seed']
        self.profile_hist_ep = {k: [] for k in self.profile_hist_field_lst}


    def get_current_date(self):
        # return self.stocks_df[0].index[self.day]
        return self.stocks_df[0].index[self.curTradeDay]

    def reset(self):
        # self.epoch = self.epoch + 1
        self.stepcount = 0
        if self.mode == "train":
            # self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
            self.day = 0
        else:
            # self.day = self.days - 1
            self.day = 0
        ############################################################################
        self.curTradeDay = 0

        state = self.features[:, self.curTradeDay - self.days + 1: self.curTradeDay + 1, :]
        self.state = state

        self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
        self.curData.sort_values(['stock'], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)
        self.date_memory = [self.curData['date'].unique()[0]]
        self.extra_data = self.data_dict['extra_data']
        self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst}
        # self.cur_hidden_vector_ay = []
        cur_risk_boundary, stock_ma_price = self.run_mkt_return_observer(stage='reset') # after curData and state, before cur_risk_boundary
        if stock_ma_price is not None:
            self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price

        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]
        self.var_lst = [0.0]
        self.cur_capital = self.initial_amount
        self.asset_lst = [self.initial_amount]
        self.profit_lst = [0] # percentage of portfolio daily returns
        self.risk_adj_lst = [cur_risk_boundary]
        self.risk_adj_price_lst = [cur_risk_boundary]
        self.is_last_ctrl_solvable = False
        self.risk_raw_lst = [0] # For performance analysis. Record the risk without using risk controllrt during the validation/test period.
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_amount]
        self.risk_pred_lst = []
        self.rl_reward_risk_lst = []
        self.rl_reward_profit_lst = []
        self.reward_lst = [0]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.config.topK+ 1 if self.config.if_cap else self.config.topK,))
        self.bound_flag = 1  # 1 for long and long+short, -1 for short
        self.action_cbf_memeory = [np.array([0] * ((self.config.topK+ 1) if self.config.if_cap else self.config.topK))]
        self.actions_memory = [np.random.dirichlet(np.ones(self.config.topK+ 1 if self.config.if_cap else self.config.topK))* self.bound_flag]
        self.action_rl_memory = [self.actions_memory[-1]]

        self.ctrl_weight_lst = [1.0]
        self.solvable_flag = []
        self.solver_stat = {'epoch':self.epoch,'rolling_num':self.rolling_num,'solvable': 0, 'insolvable': 0, 'stochastic_solvable': 0, 'stochastic_time': [], 'socp_solvable': 0, 'socp_time': []}

        return state

    def get_prices(self):
        prices = self.prices[:, self.curTradeDay, :]
        o, h, l, c = prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3]
        return o, h, l, c

    def step(self, action: np.array = None):

        date = self.get_current_date()
        if self.curTradeDay < self.num_days - 1:
            self.done = False
        else:
            self.done = True
        #########################################################################
        if self.done:
            self.cur_capital = self.cur_capital
            self.asset_lst[-1] = self.cur_capital
            self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]
            if len(self.action_rl_memory) > 1:
                self.return_raw_lst[-1] = self.return_raw_lst[-1]*(1-self.transaction_cost_pct)
                # self.return_raw_lst[-1] = self.return_raw_lst[-1]
            if self.config.enable_market_observer and (self.mode == 'train'):
                # Training at the end of epoch
                ori_profit_rate = np.append([1], np.array(self.return_raw_lst)[1:] / np.array(self.return_raw_lst)[:-1],
                                            axis=0)
                adj_profit_rate = np.array(self.profit_lst) + 1
                label_kwargs = {'mode': self.mode, 'ori_profit': ori_profit_rate, 'adj_profit': adj_profit_rate,
                                'ori_risk': np.array(self.risk_raw_lst), 'adj_risk': np.array(self.risk_cbf_lst)}
                self.mkt_observer.train(**label_kwargs)
            else:
                label_kwargs = {}
            invest_profile = self.get_results()
            self.save_profile(invest_profile=invest_profile)
            return self.state, self.reward, self.done, label_kwargs
        else:
            weights = np.reshape(action, (-1)) # [1, num_of_stocks] or [num_of_stocks, ]
            self.actions_memory.append(weights)
            if not self.config.enable_controller:
                self.action_rl_memory = self.actions_memory
            if self.curTradeDay == 0:
                self.cur_capital = self.cur_capital
            else:
                cur_p = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
                #last_p = np.array(self.curData['open'].values) * (1 + self.cur_slippage_drift)
                last_p = np.array(self.lastDayData['close'].values) * (1 + self.cur_slippage_drift)
                x_p = cur_p / last_p
                last_action = np.array(self.actions_memory[-2])
                if weights.shape[0] != self.config.topK:
                    sgn = np.sign(last_action)
                    x_p = np.hstack(([1], x_p))
                    x_p_adj = np.where((x_p >= 2) & (last_action < 0), 2, x_p)
                    adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
                    adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
                    last_w_adj = adj_w_ay / adj_cap
                    self.cur_capital = self.cur_capital * (1-weights[0]) * (1 - (np.sum(np.abs(weights[1:] - last_w_adj[1:]) * self.transaction_cost_pct))) + self.cur_capital * weights[0]
                else:
                    sgn = np.sign(last_action)
                    x_p_adj = np.where((x_p >= 2) & (last_action < 0), 2, x_p)
                    adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
                    adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
                    last_w_adj = adj_w_ay / adj_cap
                    self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(weights - last_w_adj) * self.transaction_cost_pct)))


                # Check if loss the whole capital
                if (adj_cap <= 0) or np.all(adj_w_ay == 0):
                    raise ValueError("Loss the whole capital! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                        self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                self.asset_lst[-1] = self.cur_capital
                self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]
                if len(self.action_rl_memory) > 1:
                    last_rl_action = np.array(self.action_rl_memory[-2])
                    prev_rl_cap = self.return_raw_lst[-1]
                    if weights.shape[0] != self.config.topK:
                        sgn_rl = np.sign(last_rl_action)
                        x_p_adjrl = np.where((x_p >= 2) & (last_rl_action < 0), 2, x_p)
                        adj_w_ay = sgn_rl * (last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action))
                        adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
                    else:
                        sgn_rl = np.sign(last_rl_action)
                        x_p_adjrl = np.where((x_p >= 2) & (last_rl_action < 0), 2, x_p)
                        adj_w_ay = sgn_rl * (last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action))
                        adj_cap = np.sum((x_p_adjrl - 1) * last_rl_action) + 1

                    if (adj_cap <= 0) or np.all(adj_w_ay == 0):
                        print(
                            "Loss the whole capital if using RL actions only! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                                self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))

                        adj_w_ay = np.array([1 / self.config.topK] * self.config.topK) * self.bound_flag
                        adj_cap = 1
                    last_rlw_adj = adj_w_ay / adj_cap
                    if weights.shape[0] != self.config.topK:
                        return_raw = prev_rl_cap * (1 - self.action_rl_memory[-1][0]) * (1 - (np.sum(np.abs(self.action_rl_memory[-1][1:] - last_rlw_adj[1:]) * self.transaction_cost_pct))) + prev_rl_cap * self.action_rl_memory[-1][0]
                    else:
                        return_raw = prev_rl_cap * (1 - (np.sum(np.abs(self.action_rl_memory[-1] - last_rlw_adj) * self.transaction_cost_pct)))
                    self.return_raw_lst[-1] = return_raw

            # Jump to the next day
            self.curTradeDay = self.curTradeDay + 1

            next_state = self.features[:, self.curTradeDay - self.days + 1: self.curTradeDay + 1, :]
            self.state = next_state
            self.lastDayData = self.curData
            self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
            self.ctl_state = {k: np.array(list(self.curData[k].values)) for k in
                              self.config.otherRef_indicator_lst}  # State data for the controller
            cur_date = self.curData['date'].unique()[0]
            self.date_memory.append(cur_date)
            self.cur_slippage_drift = np.random.random(self.config.topK) * (self.slippage * 2) - self.slippage
            curDay_ClosePrice_withSlippage = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
            #lastDay_ClosePrice_withSlippage = np.array(self.curData['open'].values) * (1 + self.cur_slippage_drift)
            lastDay_ClosePrice_withSlippage = np.array(self.lastDayData['close'].values) * (1 + self.cur_slippage_drift)
            rate_of_price_change = curDay_ClosePrice_withSlippage / lastDay_ClosePrice_withSlippage
            #rate_of_price_change_market = np.array(self.curData['close'].values) / np.array(self.curData['open'].values)
            if weights.shape[0] != self.config.topK:
                rate_of_price_change_adj = np.where((rate_of_price_change >= 2) & (weights[1:] < 0), 2, rate_of_price_change)
                sigDayReturn = (rate_of_price_change_adj - 1) * weights[1:]  # [s1_pct, s2_pct, .., px_pct_returns]
            else:
                rate_of_price_change_adj = np.where((rate_of_price_change >= 2) & (weights < 0), 2, rate_of_price_change)
                sigDayReturn = (rate_of_price_change_adj - 1) * weights  # [s1_pct, s2_pct, .., px_pct_returns]
            poDayReturn = np.sum(sigDayReturn)

            if poDayReturn <= (-1):

                raise ValueError("Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(self.curTradeDay,
                                                                                                   self.date_memory[-1], poDayReturn))
            if weights.shape[0] != self.config.topK:
                updatePoValue = self.cur_capital * (1-weights[0]) * (poDayReturn + 1) + self.cur_capital * weights[0]
            else:
                updatePoValue = self.cur_capital * (poDayReturn + 1)
            poDayReturn_withcost = (updatePoValue - self.cur_capital) / self.cur_capital  # Include the cost in the last timestamp
            # updatePoValue1 = self.cur_capital * ((poDayReturn + 1 - np.abs(weights[0])) + np.abs(weights[0]))
            self.cur_capital = updatePoValue
            # self.state = np.append(self.state, [np.log(self.cur_capital / self.initial_asset)],axis=0)  # current portfolio value observation
            self.profit_lst.append(poDayReturn_withcost)  # Daily return
            self.asset_lst.append(self.cur_capital)
            # Receive info from the market observer
            cur_risk_boundary, stock_ma_price = self.run_mkt_return_observer(stage='run', rate_of_price_change=np.array([rate_of_price_change]))
            if stock_ma_price is not None:
                self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
            self.risk_adj_lst.append(cur_risk_boundary)
            self.ctrl_weight_lst.append(1.0)
            daily_return_ay = np.array(list(self.curData['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)].values))
            daily_return_ay = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in daily_return_ay])
            if self.config.gmv != 'entry':
                cur_cov = np.cov(daily_return_ay)
            w_rl = self.action_rl_memory[-1] # weights - self.action_cbf_memeory[-1]
            w_rl = w_rl / np.sum(np.abs(w_rl))
            # Daily risk
            if self.curTradeDay == 1:
                prev_rl_cap = self.return_raw_lst[-1]
            else:
                prev_rl_cap = self.return_raw_lst[-1]
            if weights.shape[0] != self.config.topK:
                self.risk_cbf_lst.append(np.sqrt(np.matmul(np.matmul(weights[1:], cur_cov), weights[1:].T)))
                self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl[1:], cur_cov), w_rl[1:].T)))
                rate_of_price_change = np.hstack(([1], rate_of_price_change))
                rate_of_price_change_adj_rawrl = np.where((rate_of_price_change >= 2) & (w_rl < 0), 2,
                                                          rate_of_price_change)
                po_r_rl = np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl)
                return_raw = prev_rl_cap * (1-w_rl[0]) * (po_r_rl + 1) + prev_rl_cap * w_rl[0]
            else:
                self.risk_cbf_lst.append(np.sqrt(np.matmul(np.matmul(weights, cur_cov), weights.T)))
                self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl, cur_cov), w_rl.T)))
                rate_of_price_change_adj_rawrl = np.where((rate_of_price_change >= 2) & (w_rl < 0), 2,
                                                          rate_of_price_change)
                po_r_rl = np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl)
                return_raw = prev_rl_cap * (po_r_rl + 1)

            if po_r_rl <= (-1):
                raise ValueError("Loss the whole capital if using RL actions only! [Day: {}, date: {}, po_r_rl: {}]".format(self.curTradeDay, self.date_memory[-1], po_r_rl))
            self.return_raw_lst.append(return_raw)

            # CVaR
            if self.mode != 'train':
                if self.stock_type == 'CSP1':
                    expected_r_series = daily_return_ay[:, -self.config.dailyRetun_lookback:] * (1 - self.aux_stocks[1]['mask'].reshape(-1, 1))
                elif self.stock_type == 'CSP2':
                    expected_r_series = daily_return_ay[:, -self.config.dailyRetun_lookback:] * (1 - self.aux_stocks[2]['mask'].reshape(-1, 1))
                elif self.stock_type == 'CSP3':
                    expected_r_series = daily_return_ay[:, -self.config.dailyRetun_lookback:] * (1 - self.aux_stocks[3]['mask'].reshape(-1, 1))
                else:
                    expected_r_series = daily_return_ay[:, -self.config.dailyRetun_lookback:]
            else:
                expected_r_series = daily_return_ay[:, -self.config.dailyRetun_lookback:]
            ######################################################################################
            expected_r_prev = np.mean(expected_r_series[:, -1:], axis=1)
            expected_cov = np.cov(expected_r_series)
            if weights.shape[0] != self.config.topK:
                expected_r_prev = np.where((expected_r_prev>=1)&(weights[1:]<0), 1, expected_r_prev)
                expected_r = np.sum(np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights[1:], (-1, 1)))
                expected_std = np.sum(
                    np.sqrt(np.reshape(weights[1:], (1, -1)) @ expected_cov @ np.reshape(weights[1:], (-1, 1))))
            else:
                expected_r_prev = np.where((expected_r_prev>=1)&(weights<0), 1, expected_r_prev)
                expected_r = np.sum(np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights, (-1, 1)))
                expected_std = np.sum(
                    np.sqrt(np.reshape(weights, (1, -1)) @ expected_cov @ np.reshape(weights, (-1, 1))))

            cvar_lz = spstats.norm.ppf(1-0.05) # positive 1.65 for 95%(=1-alpha) confidence level.
            var_expected = -expected_r + expected_std * cvar_lz
            self.var_lst.append(var_expected)
            cvar_Z = np.exp(-0.5*np.power(cvar_lz, 2)) / 0.05 / np.sqrt(2*np.pi)
            cvar_expected = -expected_r + expected_std * cvar_Z
            self.cvar_lst.append(cvar_expected)
            ###############################################################################################################
            # CVaR without risk controller
            expected_r_prevrl = expected_r_prev
            if weights.shape[0] != self.config.topK:
                expected_r_prevrl = np.where((expected_r_prevrl >= 1) & (w_rl[1:] < 0), 1, expected_r_prevrl)
                expected_r_raw = np.sum(np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl[1:], (-1, 1)))
                expected_std_raw = np.sum(np.sqrt(np.reshape(w_rl[1:], (1, -1)) @ expected_cov @ np.reshape(w_rl[1:], (-1, 1))))

            else:
                expected_r_prevrl = np.where((expected_r_prevrl >= 1) & (w_rl < 0), 1, expected_r_prevrl)
                expected_r_raw = np.sum(np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl, (-1, 1)))
                expected_std_raw = np.sum(np.sqrt(np.reshape(w_rl, (1, -1)) @ expected_cov @ np.reshape(w_rl, (-1, 1))))

            cvar_expected_raw = -expected_r_raw + expected_std_raw * cvar_Z
            self.cvar_raw_lst.append(cvar_expected_raw)

            profit_part = np.log(poDayReturn_withcost + 1)
            if (self.config.trained_best_model_type == 'js_loss') and self.config.enable_controller:
                # Action reward guiding mechanism
                if self.config.trade_pattern == 1:
                    weights_norm = weights
                    w_rl_norm = w_rl
                elif self.config.trade_pattern == 2:
                    # [-1, 1] -> [0, 1]
                    weights_norm = (weights + 1) / 2
                    w_rl_norm = (w_rl + 1) / 2
                elif self.config.trade_pattern == 3:
                    # [-1, 0] -> [0, 1]
                    weights_norm = -weights
                    w_rl_norm = -w_rl
                else:
                    raise ValueError("Unexpected trade pattern: {}".format(self.config.trade_pattern))

                js_m = 0.5 * (w_rl_norm + weights_norm)
                js_divergence = (0.5 * entropy(pk=w_rl_norm, qk=js_m, base=2)) + (
                            0.5 * entropy(pk=weights_norm, qk=js_m, base=2))
                js_divergence = np.clip(js_divergence, 0, 1)
                risk_part = (-1) * js_divergence
                # kl_part=-entropy(pk=w_rl_norm, qk=weights_norm, base=2)
                diff_part = np.log(self.cur_capital/self.asset_lst[-2])
                scaled_profit_part = profit_part * self.config.lambda_1
                scaled_risk_part = self.config.lambda_2 * risk_part
                weights = np.clip(weights,1e-8,1)
                entropy_part = -np.sum(weights*np.log(weights))
                cur_reward = 200*diff_part+0.1*risk_part+0.05*entropy_part # now 50 5 0.1 

            elif (self.config.mode == 'RLonly') and (self.config.trained_best_model_type == 'pr_loss'):
                # overall return maximisation + risk minimisation
                cov_r_t0 = np.cov(self.ctl_state['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)])
                if weights.shape[0] != self.config.topK:
                    risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights[1:]]), cov_r_t0), np.array([weights[1:]]).T)[0][0])
                else:
                    risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights]), cov_r_t0), np.array([weights]).T)[0][0])
                scaled_risk_part = (-1) * risk_part * 50
                scaled_profit_part = profit_part * self.config.lambda_1
                cur_reward = scaled_profit_part + scaled_risk_part

            elif (self.config.mode == 'RLonly') and (self.config.trained_best_model_type == 'sr_loss'):
                # Sharpe ratio maximisation
                cov_r_t0 = np.cov(self.ctl_state['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)])
                if weights.shape[0] != self.config.topK:
                    risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights[1:]]), cov_r_t0), np.array([weights[1:]]).T)[0][0])
                else:
                    risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights]), cov_r_t0), np.array([weights]).T)[0][0])
                profit_part = poDayReturn_withcost
                scaled_profit_part = profit_part
                scaled_risk_part = risk_part
                cur_reward = (scaled_profit_part - (
                            self.config.mkt_rf[self.config.market_name] * 0.01)) / scaled_risk_part
            else:
                risk_part = 0
                scaled_risk_part = 0
                scaled_profit_part = profit_part * self.config.lambda_1
                # scaled_profit_part_1 = self.config.lambda_1*(self.cur_capital-self.asset_lst[-2])/self.asset_lst[-2]
                diff_part = np.log(self.cur_capital/self.asset_lst[-2])
                cur_reward = self.config.lambda_1*diff_part + scaled_risk_part
                # cur_reward = scaled_profit_part + scaled_risk_part

            self.rl_reward_risk_lst.append(scaled_risk_part)
            self.rl_reward_profit_lst.append(scaled_profit_part)
            self.reward = cur_reward
            self.reward_lst.append(self.reward)

            return next_state, self.reward, self.done, {}

    def run_mkt_return_observer(self, stage=None, rate_of_price_change=None):
        if self.mkt_observer is None:
            return None,None
        cur_date = self.curData['date'].unique()[0]
        if self.config.enable_market_observer:
            if stage in ['reset', 'init'] and (self.mode == 'train'):
                self.mkt_observer.reset()

            finemkt_feat = self.extra_data['fine_market']
            ma_close = finemkt_feat[finemkt_feat['date'] == cur_date][
                ['mkt_{}_close'.format(self.config.finefreq), 'mkt_{}_ma'.format(self.config.finefreq)]].values[-1]
            mkt_cur_close_price = ma_close[0]
            mkt_ma_price = ma_close[1]
            finemkt_feat = finemkt_feat[finemkt_feat['date'] == cur_date][self.config.finemkt_feat_cols_lst].values
            finemkt_feat = np.reshape(finemkt_feat, (
            len(self.config.use_features), self.config.fine_window_size))  # -> (features, window_size)
            finemkt_feat = np.expand_dims(finemkt_feat, axis=0)  # -> (batch=1, features, window_size)

            if (rate_of_price_change is not None) and (self.mode == 'train'):
                if mkt_cur_close_price > self.mkt_last_close_price:
                    mkt_direction = 0
                elif mkt_cur_close_price < self.mkt_last_close_price:
                    mkt_direction = 2
                else:
                    mkt_direction = 1
                mkt_direction = np.array([mkt_direction])
                self.mkt_observer.update_hidden_vec_reward(mode=self.mode, rate_of_price_change=rate_of_price_change,
                                                           mkt_direction=mkt_direction)

            finestock_feat = self.extra_data['fine_stock']
            stock_cur_close_price = finestock_feat[finestock_feat['date'] == cur_date][
                'stock_{}_close'.format(self.config.finefreq)].values  # (num_of_stock, )
            stock_ma_price = finestock_feat[finestock_feat['date'] == cur_date][
                'stock_{}_ma'.format(self.config.finefreq)].values  # (num_of_stock, )
            if self.config.is_gen_dc_feat:
                dc_events = finestock_feat[finestock_feat['date'] == cur_date][
                    'stock_{}_dc'.format(self.config.finefreq)].values  # (num_of_stocks, )
            else:
                dc_events = None

            finestock_feat = copy.deepcopy(self.state)
            finestock_feat = np.transpose(finestock_feat, (2, 0, 1))  # -> (features, num_of_stocks, window_size)
            finestock_feat = np.expand_dims(finestock_feat,
                                            axis=0)  # -> (batch=1, features, num_of_stocks, window_size)
            input_kwargs = {'mode': self.mode, 'stock_ma_price': np.array([stock_ma_price]),
                            'stock_cur_close_price': np.array([stock_cur_close_price]),
                            'dc_events': np.array([dc_events])}

            cur_hidden_vector_ay, lambda_val, sigma_val = self.mkt_observer.predict(finemkt_feat=finemkt_feat,
                                                                                    finestock_feat=finestock_feat,
                                                                                    **input_kwargs)  # lambda_val: not applicable
            self.cur_hidden_vector_ay.append(cur_hidden_vector_ay)
            if self.config.is_enable_dynamic_risk_bound:
                if int(sigma_val[-1]) == 0:
                    # up
                    cur_risk_boundary = self.config.risk_up_bound
                elif int(sigma_val[-1]) == 1:
                    # hold
                    cur_risk_boundary = self.config.risk_hold_bound
                elif int(sigma_val[-1]) == 2:
                    # down
                    cur_risk_boundary = self.config.risk_down_bound
                else:
                    raise ValueError('Unknown sigma value [{}]..'.format(sigma_val[-1]))

            else:
                cur_risk_boundary = self.config.risk_default

            # self.state = np.append(self.state, cur_hidden_vector_ay[-1], axis=0)
            self.mkt_last_close_price = mkt_cur_close_price
        else:
            cur_risk_boundary = self.config.risk_default
            if self.config.mode == 'RLcontroller':
                finestock_feat = self.extra_data['fine_stock']
                stock_cur_close_price = finestock_feat[finestock_feat['date'] == cur_date][
                    'stock_{}_close'.format(self.config.finefreq)].values  # (num_of_stock, )
                stock_ma_price = finestock_feat[finestock_feat['date'] == cur_date][
                    'stock_{}_ma'.format(self.config.finefreq)].values  # (num_of_stock, )
            else:
                stock_ma_price = None

        return cur_risk_boundary, stock_ma_price

    def sum_normalization(self, actions):
        if np.sum(np.abs(actions)) == 0:
            norm_weights = np.array([1/len(actions)]*len(actions)) * self.bound_flag
        else:
            norm_weights = actions / np.sum(np.abs(actions))
        return norm_weights

    def get_results(self):
        self.profit_lst = np.array(self.profit_lst)
        self.asset_lst = np.array(self.asset_lst)

        netProfit = self.cur_capital - self.initial_amount  # Profits
        netProfit_pct = netProfit / self.initial_amount  # Rate of overall returns

        diffPeriodAsset = np.diff(self.asset_lst)
        sigReturn_max = np.max(diffPeriodAsset)  # Maximal returns in a single transaction.
        sigReturn_min = np.min(diffPeriodAsset)  # Minimal returns in a single transaction

        # Annual Returns
        annualReturn_pct = np.power((1 + netProfit_pct), (self.config.tradeDays_per_year / len(self.asset_lst))) - 1

        dailyReturn_pct_max = np.max(self.profit_lst)
        dailyReturn_pct_min = np.min(self.profit_lst)
        avg_dailyReturn_pct = np.mean(self.profit_lst)
        # strategy volatility
        volatility = np.sqrt(
            np.sum(np.power((self.profit_lst - avg_dailyReturn_pct), 2)) * self.config.tradeDays_per_year / (
                        len(self.profit_lst) - 1))

        # SR_Vol, Long-term risk
        sharpeRatio = ((annualReturn_pct * 100) - self.config.mkt_rf[self.config.market_name]) / (volatility * 100)
        # sharpeRatio = np.max([sharpeRatio, 0])

        dailyAnnualReturn_lst = np.power((1 + np.array(self.profit_lst)), self.config.tradeDays_per_year) - 1
        dailyRisk_lst = np.array(self.risk_cbf_lst) * np.sqrt(
            self.config.tradeDays_per_year)  # Daily Risk to Anuual Risk
        dailySR = ((dailyAnnualReturn_lst[1:] * 100) - self.config.mkt_rf[self.config.market_name]) / (
                    dailyRisk_lst[1:] * 100)
        dailySR = np.append(0, dailySR)
        # dailySR = np.where(dailySR < 0, 0, dailySR)
        dailySR_max = np.max(dailySR)
        dailySR_min = np.min(dailySR[dailySR != 0])
        dailySR_avg = np.mean(dailySR)

        # For performance analysis
        dailyReturnRate_wocbf = np.diff(self.return_raw_lst) / np.array(self.return_raw_lst)[:-1]
        dailyReturnRate_wocbf = np.append(0, dailyReturnRate_wocbf)
        dailyAnnualReturn_wocbf_lst = np.power((1 + dailyReturnRate_wocbf), self.config.tradeDays_per_year) - 1
        dailyRisk_wocbf_lst = np.array(self.risk_raw_lst) * np.sqrt(self.config.tradeDays_per_year)
        dailySR_wocbf = ((dailyAnnualReturn_wocbf_lst[1:] * 100) - self.config.mkt_rf[self.config.market_name]) / (
                    dailyRisk_wocbf_lst[1:] * 100)
        dailySR_wocbf = np.append(0, dailySR_wocbf)
        # dailySR_wocbf = np.where(dailySR_wocbf < 0, 0, dailySR_wocbf)
        dailySR_wocbf_max = np.max(dailySR_wocbf)
        dailySR_wocbf_min = np.min(dailySR_wocbf[dailySR_wocbf != 0])
        dailySR_wocbf_avg = np.mean(dailySR_wocbf)

        annualReturn_wocbf_pct = np.power((1 + ((self.return_raw_lst[-1] - self.initial_amount) / self.initial_amount)),
                                          (self.config.tradeDays_per_year / len(self.return_raw_lst))) - 1
        volatility_wocbf = np.sqrt((np.sum(
            np.power((dailyReturnRate_wocbf - np.mean(dailyReturnRate_wocbf)), 2)) * self.config.tradeDays_per_year / (
                                                len(self.return_raw_lst) - 1)))
        sharpeRatio_woCBF = ((annualReturn_wocbf_pct * 100) - self.config.mkt_rf[self.config.market_name]) / (
                    volatility_wocbf * 100)
        # sharpeRatio_woCBF = np.max([sharpeRatio_woCBF, 0])

        winRate = len(np.argwhere(diffPeriodAsset > 0)) / (len(diffPeriodAsset) + 1)

        # MDD
        repeat_asset_lst = np.tile(self.asset_lst, (len(self.asset_lst), 1))
        mdd_mtix = np.triu(1 - repeat_asset_lst / np.reshape(self.asset_lst, (-1, 1)), k=1)
        mddmaxidx = np.argmax(mdd_mtix)
        mdd_highidx = mddmaxidx // len(self.asset_lst)
        mdd_lowidx = mddmaxidx % len(self.asset_lst)
        self.mdd = np.max(mdd_mtix)
        self.mdd_high = self.asset_lst[mdd_highidx]
        self.mdd_low = self.asset_lst[mdd_lowidx]
        self.mdd_highTimepoint = self.date_memory[mdd_highidx]
        self.mdd_lowTimepoint = self.date_memory[mdd_lowidx]

        # Strategy volatility during trading
        cumsum_r = np.cumsum(self.profit_lst) / np.arange(1, self.num_days + 1)  # average cumulative returns rate
        repeat_profit_lst = np.tile(self.profit_lst, (len(self.profit_lst), 1))
        stg_vol_lst = np.sqrt(
            np.sum(np.power(np.tril(repeat_profit_lst - np.reshape(cumsum_r, (-1, 1)), k=0), 2), axis=1)[
            1:] / np.arange(1, len(repeat_profit_lst)) * self.config.tradeDays_per_year)
        stg_vol_lst = np.append([0], stg_vol_lst, axis=0)
        # stg_vol_lst  = np.sqrt((np.cumsum(np.power((self.profit_lst - cumsum_r), 2))/np.arange(1, self.totalTradeDay+1)) * self.config.tradeDays_per_year)

        vol_max = np.max(stg_vol_lst)
        vol_min = np.min(np.array(stg_vol_lst)[np.array(stg_vol_lst) != 0])
        vol_avg = np.mean(stg_vol_lst)

        # short-term risk
        risk_max = np.max(self.risk_cbf_lst)
        risk_min = np.min(np.array(self.risk_cbf_lst)[np.array(self.risk_cbf_lst) != 0])
        risk_avg = np.mean(self.risk_cbf_lst)

        risk_raw_max = np.max(self.risk_raw_lst)
        risk_raw_min = np.min(np.array(self.risk_raw_lst)[np.array(self.risk_raw_lst) != 0])
        risk_raw_avg = np.mean(self.risk_raw_lst)

        # Downside risk at volatility
        risk_downsideAtVol_daily = np.sqrt(np.sum(np.power(np.tril(
            (repeat_profit_lst - np.reshape(cumsum_r, (-1, 1))) * (repeat_profit_lst < np.reshape(cumsum_r, (-1, 1))),
            k=0), 2), axis=1)[1:] / np.arange(1, len(repeat_profit_lst)) * self.config.tradeDays_per_year)
        risk_downsideAtVol_daily = np.append([0], risk_downsideAtVol_daily, axis=0)
        risk_downsideAtVol = risk_downsideAtVol_daily[-1]
        risk_downsideAtVol_daily_max = np.max(risk_downsideAtVol_daily)
        risk_downsideAtVol_daily_min = np.min(risk_downsideAtVol_daily)
        risk_downsideAtVol_daily_avg = np.mean(risk_downsideAtVol_daily)

        # Downside risk at value against initial capital
        risk_downsideAtValue_daily = (self.asset_lst / self.initial_amount) - 1
        risk_downsideAtValue_daily_max = np.max(risk_downsideAtValue_daily)
        risk_downsideAtValue_daily_min = np.min(risk_downsideAtValue_daily)
        risk_downsideAtValue_daily_avg = np.mean(risk_downsideAtValue_daily)

        # CVaR curve
        cvar_max = np.max(self.cvar_lst)
        cvar_min = np.min(np.array(self.cvar_lst)[np.array(self.cvar_lst) != 0])
        cvar_avg = np.mean(self.cvar_lst)

        cvar_raw_max = np.max(self.cvar_raw_lst)
        cvar_raw_min = np.min(np.array(self.cvar_raw_lst)[np.array(self.cvar_raw_lst) != 0])
        cvar_raw_avg = np.mean(self.cvar_raw_lst)

        # Calmar ratio
        time_T = len(self.profit_lst)
        avg_return = netProfit_pct / time_T
        variance_r = np.sum(np.power((self.profit_lst - avg_dailyReturn_pct), 2)) / (len(self.profit_lst) - 1)
        volatility_daily = np.sqrt(variance_r)

        if netProfit_pct > 0:
            shrp = avg_return / volatility_daily
            calmarRatio = (time_T * np.power(shrp, 2)) / (0.63519 + 0.5 * np.log(time_T) + np.log(shrp))
        elif netProfit_pct == 0:
            calmarRatio = (netProfit_pct) / (1.2533 * volatility_daily * np.sqrt(time_T))
        else:
            # netProfit_pct < 0
            calmarRatio = (netProfit_pct) / (-(avg_return * time_T) - (variance_r / avg_return))

        # Sterling ratio
        move_mdd_mask = np.where(np.array(self.profit_lst) < 0, 1, 0)
        moving_mdd = np.sqrt(np.sum(np.power(self.profit_lst * move_mdd_mask, 2)) * self.config.tradeDays_per_year / (
                    len(self.profit_lst) - 1))
        sterlingRatio = ((annualReturn_pct * 100) - self.config.mkt_rf[self.config.market_name]) / (moving_mdd * 100)

        cbf_abssum_contribution = np.sum(np.abs(self.action_cbf_memeory[:-1]))

        info_dict = {
            'ep':self.epoch, 'rolling_num':self.rolling_num,'stock_type':self.stock_type,'start_day':self.start_date,'end_day':self.end_date,'trading_days': self.num_days, 'annualReturn_pct': annualReturn_pct,
            'volatility': volatility, 'sharpeRatio': sharpeRatio, 'sharpeRatio_wocbf': sharpeRatio_woCBF,
            'mdd': self.mdd, 'calmarRatio': calmarRatio, 'sterlingRatio': sterlingRatio, 'netProfit': netProfit,
            'netProfit_pct': netProfit_pct, 'winRate': winRate,
            'vol_max': vol_max, 'vol_min': vol_min, 'vol_avg': vol_avg,
            'risk_max': risk_max, 'risk_min': risk_min, 'risk_avg': risk_avg,
            'riskRaw_max': risk_raw_max, 'riskRaw_min': risk_raw_min, 'riskRaw_avg': risk_raw_avg,
            'dailySR_max': dailySR_max, 'dailySR_min': dailySR_min, 'dailySR_avg': dailySR_avg,
            'dailySR_wocbf_max': dailySR_wocbf_max, 'dailySR_wocbf_min': dailySR_wocbf_min,
            'dailySR_wocbf_avg': dailySR_wocbf_avg,
            'dailyReturn_pct_max': dailyReturn_pct_max, 'dailyReturn_pct_min': dailyReturn_pct_min,
            'dailyReturn_pct_avg': avg_dailyReturn_pct,
            'sigReturn_max': sigReturn_max, 'sigReturn_min': sigReturn_min,
            'mdd_high': self.mdd_high, 'mdd_low': self.mdd_low, 'mdd_high_date': self.mdd_highTimepoint,
            'mdd_low_date': self.mdd_lowTimepoint,
            'final_capital': self.cur_capital, 'reward_sum': np.sum(self.reward_lst),
            'final_capital_wocbf': self.return_raw_lst[-1],
            'cbf_contribution': cbf_abssum_contribution,
            'risk_downsideAtVol': risk_downsideAtVol, 'risk_downsideAtVol_daily_max': risk_downsideAtVol_daily_max,
            'risk_downsideAtVol_daily_min': risk_downsideAtVol_daily_min,
            'risk_downsideAtVol_daily_avg': risk_downsideAtVol_daily_avg,
            'risk_downsideAtValue_daily_max': risk_downsideAtValue_daily_max,
            'risk_downsideAtValue_daily_min': risk_downsideAtValue_daily_min,
            'risk_downsideAtValue_daily_avg': risk_downsideAtValue_daily_avg,
            'cvar_max': cvar_max, 'cvar_min': cvar_min, 'cvar_avg': cvar_avg, 'cvar_raw_max': cvar_raw_max,
            'cvar_raw_min': cvar_raw_min, 'cvar_raw_avg': cvar_raw_avg,
            'solver_solvable': self.solver_stat['solvable'], 'solver_insolvable': self.solver_stat['insolvable'],
            'asset_lst': copy.deepcopy(self.asset_lst), 'daily_return_lst': copy.deepcopy(self.profit_lst),
            'reward_lst': copy.deepcopy(self.reward_lst),
            'stg_vol_lst': copy.deepcopy(stg_vol_lst), 'risk_lst': copy.deepcopy(self.risk_cbf_lst),
            'risk_wocbf_lst': copy.deepcopy(self.risk_raw_lst),
            'capital_wocbf_lst': copy.deepcopy(self.return_raw_lst), 'daily_sr_lst': copy.deepcopy(dailySR),
            'daily_sr_wocbf_lst': copy.deepcopy(dailySR_wocbf),
            'risk_adj_lst': copy.deepcopy(self.risk_adj_lst), 'ctrl_weight_lst': copy.deepcopy(self.ctrl_weight_lst),
            'solvable_flag': copy.deepcopy(self.solvable_flag), 'risk_pred_lst': copy.deepcopy(self.risk_pred_lst),
            'final_action_abssum_lst': copy.deepcopy(np.sum(np.abs(np.array(self.actions_memory)), axis=1)),
            'rl_action_abssum_lst': copy.deepcopy(np.sum(np.abs(np.array(self.action_rl_memory)), axis=1)[:-1]),
            'cbf_action_abssum_lst': copy.deepcopy(np.sum(np.abs(np.array(self.action_cbf_memeory)), axis=1)[:-1]),
            'daily_downsideAtVol_risk_lst': copy.deepcopy(risk_downsideAtVol_daily),
            'daily_downsideAtValue_risk_lst': copy.deepcopy(risk_downsideAtValue_daily),
            'cvar_lst': copy.deepcopy(self.cvar_lst), 'cvar_raw_lst': copy.deepcopy(self.cvar_raw_lst),
            'seed': self.config.seed_num,
        }

        return info_dict

    def save_profile(self, invest_profile):
        # basic data
        for fname in self.profile_hist_field_lst:
            if fname in list(invest_profile.keys()):
                self.profile_hist_ep[fname].append(invest_profile[fname])
            else:
                raise ValueError('Cannot find the field [{}] in invest profile..'.format(fname))
        phist_df = pd.DataFrame(self.profile_hist_ep, columns=self.profile_hist_field_lst)
        # print(self.config.res_dir,self.config.res_dir+'/{}'.format(self.mode))
        if self.mode != 'train':
            if not os.path.exists(self.config.res_dir + '/{}/{}'.format(self.mode,self.stock_type)):
                os.makedirs(self.config.res_dir + '/{}/{}'.format(self.mode,self.stock_type))
            phist_df.to_csv(os.path.join(self.config.res_dir + '/{}/{}'.format(self.mode,self.stock_type), '{}_profile.csv'.format(self.mode)), index=False)
            pd.DataFrame(self.profit_lst).to_csv(os.path.join(self.config.res_dir + '/{}/{}'.format(self.mode,self.stock_type), '{}_{}_{}_profit.csv'.format(self.mode,self.epoch,self.rolling_num)), index=False)

        else:
            if not os.path.exists(self.config.res_dir + '/train'):
                os.makedirs(self.config.res_dir + '/train')
            phist_df.to_csv(os.path.join(self.config.res_dir + '/train', '{}_profile.csv'.format(self.mode)), index=False)
            # print(os.path.join(self.config.res_dir + '/train', '{}_profile.csv'.format(self.mode)))

 

        # bestmodel_dict = {}
        if self.config.trained_best_model_type == 'max_capital':
            field_name = 'final_capital'
            v = np.max(phist_df[field_name])  # Please noted that the maximum value will be recorded.
            max_index = np.argmax(phist_df[field_name])
            e = phist_df.iloc[max_index].ep
        elif 'loss' in self.config.trained_best_model_type:
            field_name = 'reward_sum'
            v = np.max(phist_df[field_name])  # Please noted that the maximum value will be recorded.
            max_index = np.argmax(phist_df[field_name])
            e = phist_df.iloc[max_index].ep
        elif self.config.trained_best_model_type == 'sharpeRatio':
            field_name = 'sharpeRatio'
            v = np.max(phist_df[field_name])  # Please noted that the maximum value will be recorded.
            max_index = np.argmax(phist_df[field_name])
            e = phist_df.iloc[max_index].ep
        elif self.config.trained_best_model_type == 'volatility':
            field_name = 'volatility'
            v = np.min(phist_df[field_name])  # Please noted that the minimum value will be recorded.
            max_index = np.argmax(phist_df[field_name])
            e = phist_df.iloc[max_index].ep
        elif self.config.trained_best_model_type == 'mdd':
            field_name = 'mdd'
            v = np.min(phist_df[field_name])  # Please noted that the minimum value will be recorded.
            max_index = np.argmax(phist_df[field_name])
            e = phist_df.iloc[max_index].ep
        else:
            raise ValueError(
                'Unknown implementation with the best model type [{}]..'.format(self.config.trained_best_model_type))

        if True:
            print("-" * 50)
            # log_str = "Mode: {}, Ep: {}, Current epoch capital: {}, historical best captial ({} ep): {}, cputime cur: {} s, avg: {} s, system time cur: {} s/ep, avg: {} s/ep..".format(self.mode, self.epoch, self.cur_capital, v_ep, v, np.round(np.array(phist_df['cputime'])[-1], 2), np.round(cputime_avg, 2), np.round(np.array(phist_df['systime'])[-1], 2), np.round(systime_avg, 2))
            log_str = "Mode: {}，Solver: {}，Epoch：{} , Rolling_num：{}, Stock_type：{}，Current epoch capital: {}, WithoutRisk capital：{}, historical best reward_sum (ep: {}): {} | solvable: {}, insolvable: {} ".format(
                self.mode, self.config.gmv,self.epoch, self.rolling_num, self.stock_type,self.asset_lst[-1], self.return_raw_lst[-1], e, np.sum(self.reward_lst), np.array(phist_df['solver_solvable'])[-1],
                np.array(phist_df['solver_insolvable'])[-1])
                # , self.stepcount)
            print(log_str)

    def Rolling(self,start_date,end_date,rolling_num, eposide,stock_type='GSP'):
        print('-------------------init---------------------------')
        self.done = False

        self.stocks_df = []
        self.rolling_num = rolling_num
        self.epoch = eposide
        prices = []
        self.start_date = start_date
        self.end_date = end_date
        if self.if_norm:
            print("normalize datasets")

            if self.mode == "train":
                self.scaler = []
                for df in self.dataset_stocks_df:
                    if end_date is not None:
                        df = df.loc[start_date:end_date]
                    else:
                        df = df.loc[start_date:]

                    df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)

                    scaler = StandardScaler()
                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.fit_transform(
                            df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.fit_transform(df[self.features_name])

                    self.scaler.append(scaler)
                    self.stocks_df.append(df)
            else:
                assert self.scaler is not None, "val mode or test mode is not None."

                for index, df in enumerate(self.dataset_stocks_df):

                    if end_date is not None:
                        df = df.loc[start_date:end_date]
                    else:
                        df = df.loc[start_date:]

                    df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)

                    scaler = self.scaler[index]

                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.transform(
                            df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.transform(df[self.features_name])

                    self.stocks_df.append(df)
        else:
            print("no normalize datasets")

        self.features = []
        for df in self.stocks_df:
            df = df[self.features_name + self.temporals_name]
            self.features.append(df.values)
        self.features = np.stack(self.features)

        self.prices = np.stack(prices)

        self.labels = []
        for df in self.stocks_df:
            df = df[self.labels_name]
            self.labels.append(df.values)
        self.labels = np.stack(self.labels)

        print("features shape {}, prices shape {}, labels shape {}, num days {}".format(self.features.shape,
                                                                                        self.prices.shape,
                                                                                        self.labels.shape,
                                                                                        self.features.shape[1]))

        self.num_days = self.features.shape[1]
        #########################################################################################
        self.data_dict = {}
        self.data_dict['extra_data'] = {}
        all_data = self.all_data_dict['all_data']
        fine_mkt_data = self.all_data_dict['extra_all_data']['fine_market']
        fine_stock_data = self.all_data_dict['extra_all_data']['fine_stock']
        # train_data = copy.deepcopy(all_data[(all_data['date'] >= start_date) & (all_data['date'] <= end_date)])
        train_data = all_data[(all_data['date'] >= start_date) & (all_data['date'] <= end_date)]
        train_data.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
        # fmd_train = copy.deepcopy(fine_mkt_data[(fine_mkt_data['date'] >= start_date) & (fine_mkt_data['date'] <= end_date)])
        fmd_train = fine_mkt_data[(fine_mkt_data['date'] >= start_date) & (fine_mkt_data['date'] <= end_date)]
        fmd_train.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
        # fsd_train = copy.deepcopy(fine_stock_data[(fine_stock_data['date'] >= start_date) & (fine_stock_data['date'] <= end_date)])
        fsd_train = fine_stock_data[(fine_stock_data['date'] >= start_date) & (fine_stock_data['date'] <= end_date)]
        fsd_train.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
        self.data_dict['data'] = train_data
        self.data_dict['extra_data']['fine_market'] = fmd_train
        self.data_dict['extra_data']['fine_stock'] = fsd_train
        self.rawdata = self.data_dict['data']
        self.rawdata.sort_values(['date', 'stock'], ascending=True, inplace=True)
        self.rawdata.index = self.rawdata.date.factorize()[0]
        self.stock_type = stock_type
        self.cur_hidden_vector_ay = []
        self.opt+=1

    def env_data_init(self,dataset):
        self.dataset_stocks_df = dataset.stocks_df
        self.mkt_observer = dataset.market_obs
        self.all_data_dict = dataset.all_data_dict

    def scaler_init(self,env):
        if self.mode != 'train':
            self.scaler = env.scaler
