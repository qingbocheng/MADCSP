# ！/usr/bin/python
# -*- coding: utf-8 -*-#

#from cvxopt import matrix, solvers
import cvxpy as cp
from scipy.linalg import sqrtm
import pandas as pd
import numpy as np
from scipy.integrate import quad
import scipy.stats as spstats
#solvers.options['show_progress'] = False
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
from sklearn.covariance import ledoit_wolf, oas, ledoit_wolf_shrinkage
import os


def RL_withoutController(a_rl, env=None):
    a_cbf = np.array([0] * env.config.topK)
    a_rl = np.array(a_rl)
    env.action_cbf_memeory.append(a_cbf)
    env.action_rl_memory.append(a_rl)
    a_final = a_rl + a_cbf
    return a_final

def RL_withController(a_rl, a_buffer=None,env=None, mask=None):
    a_rl = np.array(a_rl)
    env.action_rl_memory.append(a_rl)
    if env.config.pricePredModel == 'MA':
        pred_prices_change = get_pred_price_change(env=env)
        pred_dict = {'shortterm': pred_prices_change}
    else:
        raise ValueError("Cannot find the price prediction model [{}]..".format(env.config.pricePredModel))
    if env.config.gmv == 'cvar':
        
        a_cbf, is_solvable_status = mask_cvar_cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict, mask=mask)


    cur_dcm_weight = 1.0
    cur_rl_weight = 1.0
    if is_solvable_status:
        a_cbf_weighted = a_cbf * cur_dcm_weight
        env.action_cbf_memeory.append(a_cbf_weighted)
        a_rl_weighted = a_rl * cur_rl_weight
        a_final = a_rl_weighted + a_cbf_weighted
    else:
        env.action_cbf_memeory.append(
            np.array([0] * (env.config.topK + 1)) if a_rl.shape[0] != env.config.topK else np.array(
                [0] * env.config.topK))
        a_final = a_rl
    return a_final


def get_pred_price_change(env):
    ma_lst = env.ctl_state['MA-{}'.format(env.config.otherRef_indicator_ma_window)]
    pred_prices = ma_lst
    cur_close_price = np.array(env.curData['close'].values)
    pred_prices_change = (pred_prices - cur_close_price) / cur_close_price
    return pred_prices_change

def mask_cvar_cbf_opt(env, a_rl, pred_dict, mask):
    """
    The risk constraint is based on CVaR method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk.
    """
    cur_date_1 = str(env.curData['date'].unique()[0]).split('T')[0]
    a_rl = np.array(a_rl)
    assert np.sum(
        np.abs(a_rl)) - 1 < 0.0001, "The sum of a_rl is not equal to 1, the value is {}, a_rl list: {}".format(
        np.sum(a_rl), a_rl)
    N = env.config.topK
    risk_safe_t1 = env.risk_adj_lst[-1]
    daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    daily_return_ay = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in daily_return_ay])
    cov_r_t1 = np.cov(daily_return_ay)
    w_lb = 0
    w_ub = 1

    a_rl_re_sign = np.reshape(a_rl, (-1, 1))
    sign_mul = np.ones((1, N + 1 if a_rl.shape[0] != N else N))
    w_lb_sign = w_lb
    w_ub_sign = w_ub
    step_add_lst = [0.002, 0.004, 0.006, 0.008, 0.008, 0.01, 0.01, 0.01, 0.015, 0.02]
    cnt = 1

    if True:
        # Complete solver
        # ++ Implemented by cvxpy
        if len(env.var_lst) < 5:
            var = env.var_lst[-1]
        else:
            var = np.mean(env.var_lst[-5:])
        # var = env.var_lst[-1]
        if a_rl.shape[0] != N:
            mask = np.hstack(([[0]], mask))
            num_count = np.count_nonzero(mask == 0)
            risk_safe_return_t1 = 1 / (num_count)

            weight_safe = min(a_rl[0], risk_safe_return_t1)
            cp_x = cp.Variable((N + 1, 1))
            z_i = cp.Variable((5, 1), nonneg=True)
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            mask = (1 - mask)
            cp_constraint.append(a_rl_re[0] + cp_x[0] <= weight_safe)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            tmp = [var + (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:]).T @ env.cur_hidden_vector_ay[-1].T + z_i[0] >= 0]
            tmp += [var + (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:]).T @ daily_return_ay[:,-i] + z_i[i] >= 0 for i in range(1,5)]
            cp_constraint += tmp
        else:
            cp_x = cp.Variable((N, 1))
            z_i = cp.Variable((1, 1), nonneg=True)
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            mask = (1 - mask)
            num_count = np.count_nonzero(mask == 0)
            weight_safe = 1 / num_count
            cp_constraint.append(a_rl_re + cp_x == a_rl[0])
            # cp_constraint.append(a_rl_re[1:]+cp_x[1:]<=0.5)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            tmp = [var + (a_rl_re + cp.multiply(mask.T, cp_x)).T @ env.cur_hidden_vector_ay[-1].T + z_i[0] >= 0]
            tmp += [var + (a_rl_re + cp.multiply(mask.T, cp_x)).T @ daily_return_ay[:,-i] + z_i[i] >= 0 for i in range(1,5)]
            cp_constraint += tmp
        # cvxpy
        try:
            if a_rl.shape[0] != N:
                obj_f2 = (var + (1 / (5*(1 - 0.05))) * cp.sum(z_i))
                obj_f1 = cp.sum_squares(cp_x)
                obj_f3 = cp.sum(cp.abs(cp_x))
            else:
                obj_f2 = (var + (1 / (5*(1 - 0.05))) * z_i)
                obj_f1 = cp.sum_squares(cp_x)
                obj_f3 = cp.sum(cp.abs(cp_x))
            cp_obj = cp.Minimize(obj_f2+obj_f1*0.000001)
            cp_prob = cp.Problem(cp_obj, cp_constraint)
            cp_prob.solve(solver=cp.ECOS, verbose=False)

            if cp_prob.status == 'optimal':
                solver_flag = True
            else:
                raise
        except:
            solver_flag = False
            cnt += 1
            risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt -2] + 0.01
        if (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            env.risk_adj_lst[-1] = risk_safe_t1
            cur_alpha_risk = cp_prob.value
            assert np.abs(np.sum(np.abs(a_rl + a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            if a_rl.shape[0] != N:
                a_cbf = np.zeros(N + 1)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl[1:]), cov_r_t1), (a_rl[1:]).T))
            else:
                a_cbf = np.zeros(N)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            env.risk_adj_lst[-1] = risk_safe_t1
    env.risk_pred_lst.append(cur_alpha_risk)
    env.is_last_ctrl_solvable = is_solvable_status
    if not is_solvable_status:
        print(cur_date_1, "None")
    if cnt > 1:
        env.stepcount = env.stepcount + 1

    return a_cbf, is_solvable_status
