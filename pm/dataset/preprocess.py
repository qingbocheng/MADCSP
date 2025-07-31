import shutil

try:
    import hfai_env

    hfai_env.set_env("mae")
except:
    pass

import os
from glob import glob
import pathlib
import sys
import pandas as pd
import numpy as np
import warnings
import time
import datetime
warnings.filterwarnings("ignore")
from config import Config
from featGen import FeatureProcesser

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def my_rank(x):
   return pd.Series(x).rank(pct=True).iloc[-1]

def cal_feature(df):
    # intermediate values
    df['max_oc'] = df[["open", "close"]].max(axis=1)
    df['min_oc'] = df[["open", "close"]].min(axis=1)
    df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
    df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + 1e-12)
    df['klow'] = (df['min_oc'] - df['low']) / df['open']
    df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df.drop(columns=['max_oc', 'min_oc'], inplace=True)

    window = [5, 10, 20, 30, 60]
    for w in window:
        df['roc_{}'.format(w)] = df['close'].shift(w) / df['close']

    for w in window:
        df['ma_{}'.format(w)] = df['close'].rolling(w).mean() / df['close']

    for w in window:
        df['std_{}'.format(w)] = df['close'].rolling(w).std() / df['close']

    for w in window:
        df['beta_{}'.format(w)] = (df['close'].shift(w) - df['close']) / (w * df['close'])

    for w in window:
        df['max_{}'.format(w)] = df['close'].rolling(w).max() / df['close']

    for w in window:
        df['min_{}'.format(w)] = df['close'].rolling(w).min() / df['close']

    for w in window:
        df['qtlu_{}'.format(w)] = df['close'].rolling(w).quantile(0.8) / df['close']

    for w in window:
        df['qtld_{}'.format(w)] = df['close'].rolling(w).quantile(0.2) / df['close']

    for w in window:
        df['rank_{}'.format(w)] = df['close'].rolling(w).apply(my_rank) / w

    for w in window:
        df['imax_{}'.format(w)] = df['high'].rolling(w).apply(np.argmax) / w

    for w in window:
        df['imin_{}'.format(w)] = df['low'].rolling(w).apply(np.argmin) / w

    for w in window:
        df['imxd_{}'.format(w)] = (df['high'].rolling(w).apply(np.argmax) - df['low'].rolling(w).apply(np.argmin)) / w

    df['ret1'] = df['close'].pct_change(1)
    for w in window:
        df['cntp_{}'.format(w)] = (df['ret1'].gt(0)).rolling(w).sum() / w

    for w in window:
        df['cntn_{}'.format(w)] = (df['ret1'].lt(0)).rolling(w).sum() / w

    for w in window:
        df['cntd_{}'.format(w)] = df['cntp_{}'.format(w)] - df['cntn_{}'.format(w)]

    df['abs_ret1'] = np.abs(df['ret1'])
    df['pos_ret1'] = df['ret1']
    df['pos_ret1'][df['pos_ret1'].lt(0)] = 0

    for w in window:
        df['sump_{}'.format(w)] = df['pos_ret1'].rolling(w).sum() / (df['abs_ret1'].rolling(w).sum() + 1e-12)

    for w in window:
        df['sumn_{}'.format(w)] = 1 - df['sump_{}'.format(w)]

    for w in window:
        df['sumd_{}'.format(w)] = 2 * df['sump_{}'.format(w)] - 1
    df['vchg1'] = df['volume'] - df['volume'].shift(1)
    df['abs_vchg1'] = np.abs(df['vchg1'])
    df['pos_vchg1'] = df['vchg1']
    df['pos_vchg1'][df['pos_vchg1'].lt(0)] = 0

    df["weekday"] = pd.to_datetime(df.index).weekday
    df["day"] = pd.to_datetime(df.index).day
    df["month"] = pd.to_datetime(df.index).month

    df.drop(columns=['ret1', 'abs_ret1', 'pos_ret1', 'vchg1', 'abs_vchg1', 'pos_vchg1'], inplace=True)

def cal_target(df):
    df['ret1'] = df['close'].pct_change(1)
    df['mov1'] = (df['ret1'] > 0)
    df['mov1'] = df['mov1'].astype(int)

def masa_cal():
    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    rand_seed = int(time.mktime(datetime.datetime.strptime(current_date, '%Y-%m-%d-%H-%M-%S').timetuple()))
    config = Config(seed_num=rand_seed, current_date=current_date)
    # 数据初始化
    fpath = os.path.join('./MADCSP/NDX_100_1d.csv')
    if not os.path.exists(fpath):
        raise ValueError("Cannot load the data from {}".format(fpath))
    data = pd.DataFrame(pd.read_csv(fpath, header=0))
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    # print(data_dict)
    return data_dict

def main(inpath, outpath, masa_data):
    features = [
        'kmid2',
        'kup2',
        'klow',
        'klow2',
        'ksft2',
        'roc_5',
        'roc_10',
        'roc_20',
        'roc_30',
        'roc_60',
        'ma_5',
        'ma_10',
        'ma_20',
        'ma_30',
        'ma_60',
        'std_5',
        'std_10',
        'std_20',
        'std_30',
        'std_60',
        'beta_5',
        'beta_10',
        'beta_20',
        'beta_30',
        'beta_60',
        'max_5',
        'max_10',
        'max_20',
        'max_30',
        'max_60',
        'min_5',
        'min_10',
        'min_20',
        'min_30',
        'min_60',
        'qtlu_5',
        'qtlu_10',
        'qtlu_20',
        'qtlu_30',
        'qtlu_60',
        'qtld_5',
        'qtld_10',
        'qtld_20',
        'qtld_30',
        'qtld_60',
        'rank_5',
        'rank_10',
        'rank_20',
        'rank_30',
        'rank_60',
        'imax_5',
        'imax_10',
        'imax_20',
        'imax_30',
        'imax_60',
        'imin_5',
        'imin_10',
        'imin_20',
        'imin_30',
        'imin_60',
        'imxd_5',
        'imxd_10',
        'imxd_20',
        'imxd_30',
        'imxd_60',
        'cntp_5',
        'cntp_10',
        'cntp_20',
        'cntp_30',
        'cntp_60',
        'cntn_5',
        'cntn_10',
        'cntn_20',
        'cntn_30',
        'cntn_60',
        'cntd_5',
        'cntd_10',
        'cntd_20',
        'cntd_30',
        'cntd_60',
        'sump_5',
        'sump_10',
        'sump_20',
        'sump_30',
        'sump_60',
        'sumn_5',
        'sumn_10',
        'sumn_20',
        'sumn_30',
        'sumn_60',
        'sumd_5',
        'sumd_10',
        'sumd_20',
        'sumd_30',
        'sumd_60',
    ]
    market = 'ndx'
    pd.DataFrame(features).to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}/madcsp/techIndicatorLst.csv')
    paths = glob(os.path.join(inpath, "*.csv"))
    pathlib.Path(outpath).mkdir(parents=True,exist_ok=True)
    masa_bftrain = []
    masa_train = []
    masa_valid = []
    masa_test = []
    for path in paths:
        name = os.path.basename(path)

        df = pd.read_csv(path)
        # df.rename({'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, axis=1, inplace=True)
        # df.rename({'date': 'Date'}, axis=1, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        cal_feature(df)
        cal_target(df)

        # df = df.iloc[60:-1].reset_index()
        df = df[df["date"] >= "2016-09-01"]
        df = df[df["date"] <= "2025-03-01"]
        # df = df.reset_index(drop=True)

        # df.drop(['amount','outstanding_share'], axis=1, inplace=True)
        df.to_csv(os.path.join(outpath, name),index=False)
        df['stock']=path.split('/')[-1].split('.')[0].upper()
        # df.rename({'Date': 'date'}, axis=1, inplace=True)
        df.drop(['open','high','low','close','volume'], axis=1, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        ################################多智能体数据处理##########################################
        df_bftrain = pd.merge(masa_data['bftrain'],df, on=['stock', 'date'])
        df_train = pd.merge(masa_data['train'],df, on=['stock', 'date'])
        df_valid = pd.merge(masa_data['valid'],df, on=['stock', 'date'])
        df_test = pd.merge(masa_data['test'],df, on=['stock', 'date'])
        masa_bftrain.append(df_bftrain)
        masa_train.append(df_train)
        masa_valid.append(df_valid)
        masa_test.append(df_test)
    pd.concat(masa_train).dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}/madcsp/train.csv',index=False)
    pd.concat(masa_valid).dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}//madcsp/valid.csv',index=False)
    pd.concat(masa_test).dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}//madcsp/test.csv',index=False)
    pd.concat(masa_bftrain).dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}//madcsp/bftrain.csv',index=False)
    masa_data['extra_train']['fine_market'].dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}/madcsp/extra_train_fine_market.csv',index=False)
    masa_data['extra_train']['fine_stock'].dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}/madcsp/extra_train_fine_stock.csv',index=False)
    masa_data['extra_valid']['fine_market'].dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}/madcsp/extra_valid_fine_market.csv',index=False)
    masa_data['extra_valid']['fine_stock'].dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}/madcsp/extra_valid_fine_stock.csv',index=False)
    masa_data['extra_test']['fine_market'].dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}/madcsp/extra_test_fine_market.csv',index=False)
    masa_data['extra_test']['fine_stock'].dropna().to_csv(f'/root/quant-ml-qlib/MADCSP/datasets/{market}/madcsp/extra_test_fine_stock.csv',index=False)



if __name__ == '__main__':

    market = 'ndx'
    data_dict = masa_cal()
    main(f'/root/quant-ml-qlib/MADCSP/datasets/{market}/raw',
         f'/root/quant-ml-qlib/MADCSP/datasets/{market}/features',data_dict)
