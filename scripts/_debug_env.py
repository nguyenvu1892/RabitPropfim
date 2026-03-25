import sys, numpy as np, torch, json, yaml
from pathlib import Path
sys.path.insert(0, 'rabit_propfirm_drl')
from environments.prop_env import MultiTFTradingEnv
from models.attention_ppo import AttentionPPO
DATA = Path('data')
sd = {}
for tf in ['M1', 'M5', 'M15', 'H1']:
    sd[tf] = np.load(DATA / f'XAUUSD_{tf}_50dim.npy')
ohlcv = np.load(DATA / 'XAUUSD_M5_ohlcv.npy')
with open('rabit_propfirm_drl/configs/prop_rules.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['stage1_mode'] = True
cfg['symbol'] = 'XAUUSD'
env = MultiTFTradingEnv(data_m1=sd['M1'], data_m5=sd['M5'], data_m15=sd['M15'], data_h1=sd['H1'], config=cfg, n_features=50, initial_balance=10000, episode_length=2000, ohlcv_m5=ohlcv, action_mode='discrete')
obs, _ = env.reset(seed=42)
attrs = [a for a in dir(env) if 'trade' in a.lower() or 'histor' in a.lower() or 'log' in a.lower() or 'pnl' in a.lower()]
print('Trade attrs:', attrs)
for i in range(50):
    obs, r, t, tr, info = env.step(0)
for i in range(50):
    obs, r, t, tr, info = env.step(3)
print('Positions:', len(env.positions))
print('Balance:', env.balance)
for a in attrs:
    val = getattr(env, a, None)
    if val is not None and hasattr(val, '__len__'):
        print(f'{a}: len={len(val)}')
        if len(val) > 0:
            print(f'  sample: {list(val)[:1]}')
    else:
        print(f'{a}: {val}')
